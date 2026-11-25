package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

type toolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type toolCallItem struct {
	Index    int              `json:"index"`
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function toolCallFunction `json:"function"`
}

type toolMessage struct {
	Role      string         `json:"role,omitempty"`
	Content   string         `json:"content"`
	Reasoning string         `json:"reasoning_content,omitempty"`
	ToolCalls []toolCallItem `json:"tool_calls,omitempty"`
}

type toolEvent struct {
	ID      string `json:"id"`
	Choices []struct {
		Message toolMessage `json:"message"`
		Delta   toolMessage `json:"delta"`
		Finish  string      `json:"finish_reason"`
	} `json:"choices"`
	Usage replyUsage `json:"usage"`
}

type toolSample struct {
	ID         string
	Role       string
	Content    string
	Reasoning  string
	Finish     string
	ToolCalls  []toolCallItem
	Usage      replyUsage
	StartedAt  time.Time
	ActiveAt   time.Time
	FinishedAt time.Time
}

func toolCallRequestBody(stream bool) (string, error) {
	payload := map[string]any{
		"messages": []map[string]string{
			{
				"role":    "system",
				"content": "You are a tool-using assistant. Whenever the user asks about weather, immediately invoke get_weather with the requested city and do not ask follow-up questions.",
			},
			{
				"role":    "user",
				"content": "What is the current weather in Seattle? Call get_weather now.",
			},
		},
		"tools": []map[string]any{
			{
				"type": "function",
				"function": map[string]any{
					"name":        "get_weather",
					"description": "Look up current weather for a city.",
					"parameters": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"city": map[string]any{"type": "string", "description": "City name"},
						},
						"required": []string{"city"},
					},
				},
			},
		},
		"max_tokens":       128,
		"seed":             4242,
		"temperature":      0.0,
		"reasoning_effort": "none",
		"stream":           stream,
	}
	b, err := json.Marshal(payload)
	return string(b), err
}

func (g *gate) sampleToolReply(ctx context.Context, body string, stream bool) (toolSample, error) {
	var result toolSample
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, g.url+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		return result, err
	}
	req.Header.Set("Authorization", "Bearer "+g.key)
	req.Header.Set("Content-Type", "application/json")
	startedAt := time.Now()
	resp, err := replyClient.Do(req)
	if err != nil {
		return result, err
	}
	activeAt := time.Now()
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return result, fmt.Errorf("tool reply HTTP %d: %s", resp.StatusCode, msg)
	}
	if !stream {
		var event toolEvent
		if err := json.NewDecoder(resp.Body).Decode(&event); err != nil {
			return result, err
		}
		if len(event.Choices) != 1 {
			return result, fmt.Errorf("got %d choices", len(event.Choices))
		}
		ch := event.Choices[0]
		result = toolSample{
			ID:         event.ID,
			Role:       ch.Message.Role,
			Content:    ch.Message.Content,
			Reasoning:  ch.Message.Reasoning,
			Finish:     ch.Finish,
			ToolCalls:  ch.Message.ToolCalls,
			Usage:      event.Usage,
			StartedAt:  startedAt,
			ActiveAt:   activeAt,
			FinishedAt: time.Now(),
		}
		return result, nil
	}
	result.StartedAt = startedAt
	result.ActiveAt = activeAt
	lines := bufio.NewScanner(resp.Body)
	lines.Buffer(make([]byte, 0, 64*1024), 1<<20)
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
		var event toolEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return result, err
		}
		if len(event.Choices) != 1 || event.ID == "" || (result.ID != "" && result.ID != event.ID) {
			return result, fmt.Errorf("inconsistent stream ID %q or choice count %d", event.ID, len(event.Choices))
		}
		result.ID = event.ID
		ch := event.Choices[0]
		if ch.Delta.Role != "" {
			result.Role = ch.Delta.Role
		}
		result.Content += ch.Delta.Content
		result.Reasoning += ch.Delta.Reasoning
		if len(ch.Delta.ToolCalls) > 0 {
			result.ToolCalls = append(result.ToolCalls, ch.Delta.ToolCalls...)
		}
		if ch.Finish != "" {
			result.Finish = ch.Finish
			result.Usage = event.Usage
		}
	}
	if err := lines.Err(); err != nil {
		return result, err
	}
	if !done {
		return result, fmt.Errorf("missing SSE [DONE]")
	}
	result.FinishedAt = time.Now()
	return result, nil
}

func parseToolArgsObject(raw string) (map[string]any, error) {
	var args map[string]any
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return nil, fmt.Errorf("invalid tool arguments JSON %q: %w", raw, err)
	}
	if args == nil {
		return nil, fmt.Errorf("tool arguments JSON must be an object, got %q", raw)
	}
	if _, unparsed := args["_unparsed"]; unparsed {
		return nil, fmt.Errorf("server emitted unparsed tool arguments: %q", raw)
	}
	return args, nil
}

func verifyConcurrentToolCalls(wantName string, ref toolSample, got []toolSample) error {
	if len(got) < 4 {
		return fmt.Errorf("expected at least 4 concurrent tool replies, got %d", len(got))
	}
	if ref.Finish != "tool_calls" || len(ref.ToolCalls) == 0 {
		return fmt.Errorf("reference did not trigger tool call: finish=%q calls=%d content=%q", ref.Finish, len(ref.ToolCalls), ref.Content)
	}
	refTC := ref.ToolCalls[0]
	if refTC.Function.Name != wantName {
		return fmt.Errorf("reference tool name %q, want %q", refTC.Function.Name, wantName)
	}
	if _, err := parseToolArgsObject(refTC.Function.Arguments); err != nil {
		return fmt.Errorf("reference: %w", err)
	}
	seenRespIDs := make(map[string]bool, len(got))
	seenCallIDs := make(map[string]bool, len(got))
	for i, s := range got {
		if s.ID == "" || !strings.HasPrefix(s.ID, "chatcmpl-") {
			return fmt.Errorf("reply %d: invalid response id %q", i, s.ID)
		}
		if seenRespIDs[s.ID] {
			return fmt.Errorf("reply %d: duplicate response id %q", i, s.ID)
		}
		seenRespIDs[s.ID] = true
		if s.Role != "assistant" {
			return fmt.Errorf("reply %d: role=%q, want assistant", i, s.Role)
		}
		if s.Finish != "tool_calls" {
			return fmt.Errorf("reply %d: finish_reason=%q, want tool_calls", i, s.Finish)
		}
		if len(s.ToolCalls) != len(ref.ToolCalls) {
			return fmt.Errorf("reply %d: tool_calls count %d, want %d", i, len(s.ToolCalls), len(ref.ToolCalls))
		}
		tc := s.ToolCalls[0]
		if tc.ID == "" || !strings.HasPrefix(tc.ID, "call_") {
			return fmt.Errorf("reply %d: invalid tool_calls[0].id %q", i, tc.ID)
		}
		if seenCallIDs[tc.ID] {
			return fmt.Errorf("reply %d: duplicate tool_calls[0].id %q across concurrent cards", i, tc.ID)
		}
		seenCallIDs[tc.ID] = true
		if tc.Type != "function" {
			return fmt.Errorf("reply %d: tool_calls[0].type=%q, want function", i, tc.Type)
		}
		if tc.Function.Name != wantName {
			return fmt.Errorf("reply %d: function.name=%q, want %q", i, tc.Function.Name, wantName)
		}
		if _, err := parseToolArgsObject(tc.Function.Arguments); err != nil {
			return fmt.Errorf("reply %d: %w", i, err)
		}
		if tc.Function.Arguments != refTC.Function.Arguments {
			return fmt.Errorf("reply %d: argument bleed: got %q, want %q", i, tc.Function.Arguments, refTC.Function.Arguments)
		}
		if s.Content != ref.Content || s.Reasoning != ref.Reasoning {
			return fmt.Errorf("reply %d: text/reasoning bleed: got (%q, %q), want (%q, %q)", i, s.Content, s.Reasoning, ref.Content, ref.Reasoning)
		}
		if s.Usage.Prompt != ref.Usage.Prompt || s.Usage.Generated != ref.Usage.Generated {
			return fmt.Errorf("reply %d: usage mismatch: got (%d, %d), want (%d, %d)", i, s.Usage.Prompt, s.Usage.Generated, ref.Usage.Prompt, ref.Usage.Generated)
		}
		if s.StartedAt.IsZero() || s.FinishedAt.IsZero() || s.FinishedAt.Before(s.StartedAt) {
			return fmt.Errorf("reply %d: missing or inverted request timing (%v .. %v)", i, s.StartedAt, s.FinishedAt)
		}
	}
	latestStart := got[0].StartedAt
	earliestFinish := got[0].FinishedAt
	for _, s := range got[1:] {
		if s.StartedAt.After(latestStart) {
			latestStart = s.StartedAt
		}
		if s.FinishedAt.Before(earliestFinish) {
			earliestFinish = s.FinishedAt
		}
	}
	if !latestStart.Before(earliestFinish) {
		return fmt.Errorf("concurrent tool calls did not overlap in flight: latest start %s >= earliest finish %s",
			latestStart.Format(time.RFC3339Nano), earliestFinish.Format(time.RFC3339Nano))
	}
	return nil
}

func (g *gate) concurrentToolCalls() error {
	const concurrency = 4
	ctx, cancel := context.WithTimeout(g.ctx, 90*time.Second)
	defer cancel()
	nonStreamBody, err := toolCallRequestBody(false)
	if err != nil {
		return err
	}
	streamBody, err := toolCallRequestBody(true)
	if err != nil {
		return err
	}
	ref, err := g.sampleToolReply(ctx, nonStreamBody, false)
	if err != nil {
		return fmt.Errorf("solo reference: %w", err)
	}
	for _, tc := range []struct {
		name   string
		body   string
		stream bool
	}{
		{"stream", streamBody, true},
		{"nonstream", nonStreamBody, false},
	} {
		type outcome struct {
			idx    int
			sample toolSample
			err    error
		}
		startBarrier := make(chan struct{})
		results := make(chan outcome, concurrency)
		for i := range concurrency {
			go func(idx int) {
				<-startBarrier
				s, e := g.sampleToolReply(ctx, tc.body, tc.stream)
				results <- outcome{idx, s, e}
			}(i)
		}
		close(startBarrier)
		got := make([]toolSample, concurrency)
		for range concurrency {
			o := <-results
			if o.err != nil {
				return fmt.Errorf("%s worker %d: %w", tc.name, o.idx, o.err)
			}
			got[o.idx] = o.sample
		}
		if g.sabotageEnabled("concurrent-tool-calls") && tc.stream {
			// Negative control: inject duplicate call ID + argument bleed to prove validator catches it.
			got[1].ToolCalls[0].ID = got[0].ToolCalls[0].ID
			got[1].ToolCalls[0].Function.Arguments += "bleed"
			if err := verifyConcurrentToolCalls("get_weather", ref, got); err != nil {
				return fmt.Errorf("sabotage concurrent-tool-calls: %w", err)
			}
			return fmt.Errorf("sabotage concurrent-tool-calls was not caught")
		}
		if err := verifyConcurrentToolCalls("get_weather", ref, got); err != nil {
			return fmt.Errorf("%s: %w", tc.name, err)
		}
		if _, err := g.waitHealthDrained(5 * time.Second); err != nil {
			return fmt.Errorf("%s drain: %w", tc.name, err)
		}
	}
	return nil
}

func verifyDisconnectMidThinking(logBytes []byte, maxTokens int, drained healthSnapshot, ref, followUp replySample) error {
	if err := checkFreed(logBytes, false, maxTokens); err != nil {
		return err
	}
	if !drained.OK ||
		drained.ActiveLanes == nil || *drained.ActiveLanes != 0 ||
		drained.Inflight == nil || *drained.Inflight != 0 ||
		drained.QueuedRequests == nil || *drained.QueuedRequests != 0 {
		return fmt.Errorf("/health not drained after mid-thinking disconnect: active_lanes=%s inflight=%s queued=%s",
			ptrInt(drained.ActiveLanes), ptrInt(drained.Inflight), ptrInt(drained.QueuedRequests))
	}
	if !sameReply(ref, followUp) {
		return fmt.Errorf("follow-up seeded reply on freed lane differs from solo reference: got %+v, want %+v", followUp.message, ref.message)
	}
	return nil
}

func (g *gate) disconnectMidThinking() error {
	const thinkMaxTokens = 512
	ctx, cancel := context.WithTimeout(g.ctx, 60*time.Second)
	defer cancel()
	// Solo seeded reference before the disconnect trial (same 1-lane server guarantees exact lane reuse).
	refBody := `{"messages":[{"role":"user","content":"What is 17 times 23? Answer with only the integer."}],"max_tokens":32,"seed":9001,"temperature":0.8,"reasoning_effort":"none","stream":false}`
	ref, err := g.sampleReply(refBody, false, nil)
	if err != nil {
		return fmt.Errorf("solo reference before mid-thinking disconnect: %w", err)
	}
	offset, err := g.logOffset()
	if err != nil {
		return err
	}
	thinkPayload := map[string]any{
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": "Solve step by step in detail: find the smallest positive integer solutions to x^2 - 7y^2 = 1 and prove carefully why no smaller positive solutions exist.",
			},
		},
		"max_tokens":       thinkMaxTokens,
		"thinking_budget":  384,
		"reasoning_effort": "high",
		"seed":             31415,
		"temperature":      0.7,
		"stream":           true,
	}
	thinkBody, err := json.Marshal(thinkPayload)
	if err != nil {
		return err
	}
	// Dedicated non-keepalive client so Body.Close() closes the underlying TCP connection immediately.
	disconnectClient := &http.Client{
		Transport:     &http.Transport{DisableKeepAlives: true, Proxy: nil},
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, g.url+"/v1/chat/completions", strings.NewReader(string(thinkBody)))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+g.key)
	req.Header.Set("Content-Type", "application/json")
	resp, err := disconnectClient.Do(req)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		resp.Body.Close()
		return fmt.Errorf("expected SSE stream for thinking request, got HTTP %d: %s", resp.StatusCode, msg)
	}
	scanner := bufio.NewScanner(resp.Body)
	thinkingChunks := 0
	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			resp.Body.Close()
			return fmt.Errorf("unexpected SSE line %q", line)
		}
		if data == "[DONE]" {
			resp.Body.Close()
			return fmt.Errorf("stream finished before entering thinking phase")
		}
		var ev jobEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			resp.Body.Close()
			return err
		}
		if len(ev.Choices) != 1 {
			continue
		}
		c := ev.Choices[0]
		if c.Delta.Content != "" {
			resp.Body.Close()
			return fmt.Errorf("received content delta %q before thinking chunk", c.Delta.Content)
		}
		if c.Delta.Reasoning != "" {
			thinkingChunks++
			if thinkingChunks >= 2 {
				break
			}
		}
	}
	if thinkingChunks == 0 {
		resp.Body.Close()
		return fmt.Errorf("did not receive any reasoning_content chunk before disconnect (scan err: %v)", scanner.Err())
	}
	// Tear down TCP connection immediately mid-thinking.
	disconnectStart := time.Now()
	resp.Body.Close()
	disconnectClient.CloseIdleConnections()

	drained, err := g.waitHealthDrained(laneStartLimit)
	if err != nil {
		return err
	}
	if elapsed := time.Since(disconnectStart); elapsed > laneStartLimit {
		return fmt.Errorf("lane took %v to drain after mid-thinking disconnect (limit %v)", elapsed, laneStartLimit)
	}
	// Run follow-up seeded request on the freed lane and verify exact solo-reference output.
	followUp, err := g.sampleReply(refBody, false, nil)
	if err != nil {
		return fmt.Errorf("follow-up seeded reply after mid-thinking disconnect: %w", err)
	}
	f, err := os.Open(g.log)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		return err
	}
	logBytes, err := io.ReadAll(f)
	if err != nil {
		return err
	}
	if g.sabotageEnabled("disconnect-mid-thinking") {
		// Negative control: simulate an uncancelled lane that ran to max_tokens.
		fakeLog := []byte(fmt.Sprintf("[serve] client gone after %d generated tokens; lane freed\n", thinkMaxTokens))
		if err := verifyDisconnectMidThinking(fakeLog, thinkMaxTokens, drained, ref, followUp); err != nil {
			return fmt.Errorf("sabotage disconnect-mid-thinking: %w", err)
		}
		return fmt.Errorf("sabotage disconnect-mid-thinking was not caught")
	}
	return verifyDisconnectMidThinking(logBytes, thinkMaxTokens, drained, ref, followUp)
}
