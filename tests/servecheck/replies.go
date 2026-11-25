package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

var replyClient = &http.Client{Timeout: 90 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}

type replyMessage struct {
	Role      string `json:"role,omitempty"`
	Content   string `json:"content"`
	Reasoning string `json:"reasoning_content,omitempty"`
}
type replyUsage struct {
	Prompt    int `json:"prompt_tokens"`
	Generated int `json:"completion_tokens"`
	Details   struct {
		Cached *int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}
type replyEvent struct {
	ID      string `json:"id"`
	Choices []struct {
		Message replyMessage
		Delta   replyMessage
		Finish  string `json:"finish_reason"`
	}
	Usage replyUsage
}
type replySample struct {
	message replyMessage
	finish  string
	usage   replyUsage
}

func (s replySample) valid() bool {
	return s.message.Role == "assistant" && (s.message.Content != "" || s.message.Reasoning != "") &&
		(s.finish == "length" || s.finish == "stop") && s.usage.Generated > 0 && s.usage.Prompt > 0
}

func (g *gate) sampleReply(body string, stream bool, first func(string)) (replySample, error) {
	var result replySample
	req, err := http.NewRequestWithContext(g.ctx, http.MethodPost, g.url+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		return result, err
	}
	req.Header.Set("Authorization", "Bearer "+g.key)
	req.Header.Set("Content-Type", "application/json")
	resp, err := replyClient.Do(req)
	if err != nil {
		return result, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return result, fmt.Errorf("reply HTTP %d: %s", resp.StatusCode, msg)
	}
	if !stream {
		var event replyEvent
		if err := json.NewDecoder(resp.Body).Decode(&event); err != nil {
			return result, err
		}
		if len(event.Choices) != 1 {
			return result, fmt.Errorf("got %d choices", len(event.Choices))
		}
		ch := event.Choices[0]
		result = replySample{ch.Message, ch.Finish, event.Usage}
	} else {
		lines := bufio.NewScanner(resp.Body)
		id := ""
		done := false
		for lines.Scan() {
			line := lines.Text()
			if line == "" {
				continue
			}
			data, ok := strings.CutPrefix(line, "data: ")
			if !ok {
				return result, fmt.Errorf("unexpected SSE line %q", line)
			}
			if data == "[DONE]" {
				done = true
				break
			}
			var event replyEvent
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				return result, err
			}
			if len(event.Choices) != 1 || event.ID == "" || (id != "" && id != event.ID) {
				return result, fmt.Errorf("inconsistent stream ID or choice count")
			}
			id = event.ID
			ch := event.Choices[0]
			if ch.Delta.Role != "" {
				result.message.Role = ch.Delta.Role
			}
			result.message.Content += ch.Delta.Content
			result.message.Reasoning += ch.Delta.Reasoning
			if first != nil && (ch.Delta.Content != "" || ch.Delta.Reasoning != "") {
				first(id)
				first = nil
			}
			if ch.Finish != "" {
				result.finish = ch.Finish
				result.usage = event.Usage
			}
		}
		if err := lines.Err(); err != nil {
			return result, err
		}
		if !done {
			return result, fmt.Errorf("missing SSE [DONE]")
		}
	}
	if !result.valid() {
		return result, fmt.Errorf("missing generated text, role, usage, or finish reason")
	}
	return result, nil
}

// Cached prompt tokens and request IDs describe execution, not sampled text.
func sameReply(a, b replySample) bool {
	return a.valid() && b.valid() && a.message == b.message && a.finish == b.finish &&
		a.usage.Prompt == b.usage.Prompt && a.usage.Generated == b.usage.Generated
}

func (g *gate) seededReplies() error {
	// Two distinct prompts (different topics, token lengths, and seeds) overlap on separate lanes so any cross-lane mix-up fails immediately.
	bodies := [2]string{
		`{"messages":[{"role":"user","content":"Write the integers 1 through 10000 in order, separated by commas. Do not summarize."}],"max_tokens":64,"seed":1729,"temperature":0.8,"stream":true}`,
		`{"messages":[{"role":"user","content":"List alternating Greek letter names alpha beta gamma delta repeated 200 times, separated by spaces, with no headings and no summary."}],"max_tokens":64,"seed":2029,"temperature":0.8,"stream":true}`,
	}
	var refs [2]replySample
	for i, body := range bodies {
		r, err := g.sampleReply(body, true, nil)
		if err != nil {
			return fmt.Errorf("reference prompt %d: %w", i, err)
		}
		refs[i] = r
	}
	if sameReply(refs[0], refs[1]) || refs[0].message == refs[1].message || refs[0].usage.Prompt == refs[1].usage.Prompt {
		return fmt.Errorf("distinct seeded reference prompts did not produce distinct replies and prompt counts")
	}
	type outcome struct {
		idx    int
		sample replySample
		err    error
	}
	first := make(chan string, 2)
	ends := make(chan outcome, 2)
	for i, body := range bodies {
		go func(idx int, reqBody string) {
			r, e := g.sampleReply(reqBody, true, func(id string) { first <- id })
			ends <- outcome{idx, r, e}
		}(i, body)
	}
	ids := make(map[string]bool)
	for len(ids) < 2 {
		select {
		case id := <-first:
			if !strings.HasPrefix(id, "chatcmpl-") || ids[id] {
				return fmt.Errorf("invalid or duplicated chat id %q", id)
			}
			ids[id] = true
		case end := <-ends:
			if end.err != nil {
				return end.err
			}
			return fmt.Errorf("one reply finished before both returned a text delta")
		}
	}
	serverLog, err := os.ReadFile(g.log)
	if err != nil {
		return err
	}
	for id := range ids {
		if strings.Contains(string(serverLog), "[serve] #"+strings.TrimPrefix(id, "chatcmpl-")+" prompt ") {
			return fmt.Errorf("%s already finished before both first deltas", id)
		}
	}
	var got [2]replySample
	for i := 0; i < 2; i++ {
		o := <-ends
		if o.err != nil {
			return o.err
		}
		got[o.idx] = o.sample
	}
	if sameReply(got[0], refs[1]) || sameReply(got[1], refs[0]) {
		return fmt.Errorf("cross-lane reply mix-up between concurrent lanes")
	}
	for i := 0; i < 2; i++ {
		if !sameReply(got[i], refs[i]) {
			return fmt.Errorf("overlapping seeded reply %d differs from its solo reference", i)
		}
	}
	if sameReply(got[0], got[1]) {
		return fmt.Errorf("overlapping distinct prompts collapsed to the same reply")
	}
	got[1].message.Content += "changed"
	if sameReply(got[1], refs[1]) {
		return fmt.Errorf("accepted a changed seeded reply")
	}
	return nil
}

func (g *gate) requestReply(messages []replyMessage, limit int) (replySample, error) {
	body, err := json.Marshal(map[string]any{"messages": messages, "max_tokens": limit, "seed": 1729, "temperature": 0.8})
	if err != nil {
		return replySample{}, err
	}
	return g.sampleReply(string(body), false, nil)
}

func (g *gate) followupCache() error {
	user := replyMessage{Role: "user", Content: strings.Repeat("Count carefully. ", 24) + "What is two plus two? Give just the number."}
	first, err := g.requestReply([]replyMessage{user}, 64)
	if err != nil {
		return err
	}
	next, err := g.requestReply([]replyMessage{user, first.message, {Role: "user", Content: "Add one to that number."}}, 12)
	if err != nil {
		return err
	}
	// Layer 3 seals 64-token KV blocks (`BLOCK = 64`).
	wantCached := (first.usage.Prompt / 64) * 64
	matches := func(cached int) bool {
		return wantCached >= 64 && cached >= wantCached && cached <= next.usage.Prompt-1 && cached%64 == 0
	}
	cached := next.usage.Details.Cached
	if cached == nil {
		return fmt.Errorf("follow-up omitted cached_tokens")
	}
	if !matches(*cached) {
		return fmt.Errorf("previous prompt=%d (wantCached>=%d), cached=%d, follow-up prompt=%d", first.usage.Prompt, wantCached, *cached, next.usage.Prompt)
	}
	if matches(0) {
		return fmt.Errorf("accepted zero cached tokens")
	}
	return nil
}

func windowMessages() []replyMessage {
	return []replyMessage{
		{Role: "user", Content: strings.Repeat(" apple", 1040) + " Write a detailed recipe using apples, with five steps."},
		{Role: "user", Content: "Include the time for each step."},
	}
}

// The base and follow-up are the first two requests to a fresh single-lane server.
// Any checkpoint the follow-up reuses must have come from the base.
func (g *gate) overwrittenWindow(messages []replyMessage, cold replySample) error {
	base, err := g.requestReply(messages[:1], 32)
	if err != nil {
		return fmt.Errorf("warm-up base: %w", err)
	}
	next, err := g.requestReply(messages, 32)
	if err != nil {
		return fmt.Errorf("warm follow-up: %w", err)
	}
	return verifyWindowReuse(cold, base, next)
}

func verifyWindowReuse(cold, base, next replySample) error {
	if cached := cold.usage.Details.Cached; cached == nil || *cached != 0 {
		return fmt.Errorf("cold reference must report zero cached tokens")
	}
	if cached := base.usage.Details.Cached; cached == nil || *cached != 0 {
		return fmt.Errorf("warm-up base must report zero cached tokens")
	}
	if base.usage.Prompt < 1024 || base.usage.Generated < 3 {
		return fmt.Errorf("fixture did not overwrite retained window rows: %+v", base.usage)
	}
	if next.usage.Prompt <= base.usage.Prompt {
		return fmt.Errorf("follow-up prompt %d must extend base prompt %d", next.usage.Prompt, base.usage.Prompt)
	}
	cached := next.usage.Details.Cached
	if cached == nil {
		return fmt.Errorf("follow-up omitted cached_tokens")
	}
	if *cached < 1024 || *cached%64 != 0 || *cached > base.usage.Prompt {
		return fmt.Errorf("follow-up cached %d tokens; expected sealed 64-token blocks from the %d-token base", *cached, base.usage.Prompt)
	}
	if !sameReply(cold, next) {
		return fmt.Errorf("follow-up differs from cold generation")
	}
	return nil
}

func (g *gate) defaultSeed() error {
	const messages = `[{"role":"user","content":"Give a short list of interesting round objects."}]`
	original := `{"temperature":1,"max_tokens":32,"stream":false,"messages":` + messages + `}`
	shuffled := `{"messages":` + messages + `,"stream":false,"max_tokens":32,"temperature":1}`
	ref, err := g.sampleReply(original, false, nil)
	if err != nil {
		return err
	}
	for _, tc := range []struct {
		name, body string
		stream     bool
	}{
		{"repeat", original, false}, {"key order", shuffled, false}, {"stream", strings.Replace(original, `"stream":false`, `"stream":true`, 1), true},
	} {
		got, err := g.sampleReply(tc.body, tc.stream, nil)
		if err != nil {
			return fmt.Errorf("%s: %w", tc.name, err)
		}
		if !sameReply(ref, got) {
			return fmt.Errorf("unseeded reply differs for %s", tc.name)
		}
	}
	wrong := ref
	wrong.message.Reasoning += "changed"
	if sameReply(ref, wrong) {
		return fmt.Errorf("accepted different unseeded reasoning")
	}
	return nil
}

func (g *gate) replies() {
	g.run("overlapping seeded replies", (*gate).seededReplies)
	g.run("follow-up cached prompt length", (*gate).followupCache)
	g.run("unseeded replies across repeats, key order, and streaming", (*gate).defaultSeed)
}
