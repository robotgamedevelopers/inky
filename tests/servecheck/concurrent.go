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

// The Layer 5 gate's last two checks (roadmap t-l6-gate-concurrent):
// concurrent cards issuing identical tool calls, and a disconnect in
// mid-thinking stopping generation. Both run against the real server
// here; unit tests stay free of sockets and real time.

type toolCall struct {
	Index int
	ID    string
	Name  string
	Args  string
}

type toolEvent struct {
	ID      string `json:"id"`
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			Reasoning string `json:"reasoning_content"`
			Calls     []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		}
		Finish string `json:"finish_reason"`
	}
}

const toolBody = `{"messages":[{"role":"system","content":"Always answer by calling the get_weather tool. Do not answer in plain text."},{"role":"user","content":"What is the weather in San Francisco? Call get_weather for San Francisco."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":128,"seed":1729,"temperature":0.8,"stream":true}`

func (g *gate) openStream(body string) (io.ReadCloser, *bufio.Scanner, error) {
	req, err := http.NewRequest(http.MethodPost, g.url+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Authorization", "Bearer "+g.key)
	req.Header.Set("Content-Type", "application/json")
	resp, err := replyClient.Do(req)
	if err != nil {
		return nil, nil, err
	}
	if resp.StatusCode != 200 || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		resp.Body.Close()
		return nil, nil, fmt.Errorf("expected accepted stream, got HTTP %d: %s", resp.StatusCode, msg)
	}
	return resp.Body, bufio.NewScanner(resp.Body), nil
}

// One streamed reply, read incrementally: first to its first generated
// delta (the overlap proof), then drained to its tool calls.
type toolStream struct {
	lines     *bufio.Scanner
	id        string
	calls     []toolCall
	finish    string
	lastDelta bool
}

func (s *toolStream) next() (done bool, err error) {
	for s.lines.Scan() {
		line := s.lines.Text()
		if line == "" {
			continue
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			return false, fmt.Errorf("unexpected SSE line %q", line)
		}
		if data == "[DONE]" {
			return true, nil
		}
		var e toolEvent
		if err := json.Unmarshal([]byte(data), &e); err != nil {
			return false, err
		}
		if len(e.Choices) != 1 || e.ID == "" || (s.id != "" && s.id != e.ID) {
			return false, fmt.Errorf("inconsistent stream ID or choice count: %s", data)
		}
		s.id = e.ID
		ch := e.Choices[0]
		for _, c := range ch.Delta.Calls {
			for len(s.calls) <= c.Index {
				s.calls = append(s.calls, toolCall{Index: len(s.calls)})
			}
			if c.ID != "" {
				s.calls[c.Index].ID = c.ID
			}
			if c.Function.Name != "" {
				s.calls[c.Index].Name = c.Function.Name
			}
			s.calls[c.Index].Args += c.Function.Arguments
		}
		if ch.Finish != "" {
			s.finish = ch.Finish
		}
		s.lastDelta = ch.Delta.Content != "" || ch.Delta.Reasoning != "" || len(ch.Delta.Calls) > 0
		return false, nil
	}
	if err := s.lines.Err(); err != nil {
		return false, err
	}
	return false, io.ErrUnexpectedEOF
}

// Read events until the reply has generated something (a content or
// reasoning delta, or a tool call): the stream is mid-generation here.
func (s *toolStream) firstDelta() error {
	for {
		done, err := s.next()
		if err != nil {
			return err
		}
		if done {
			return fmt.Errorf("stream ended before any generated delta")
		}
		if s.lastDelta {
			return nil
		}
	}
}

// Drain the rest of the stream to [DONE].
func (s *toolStream) drain() error {
	for {
		done, err := s.next()
		if err != nil {
			return err
		}
		if done {
			return nil
		}
	}
}

func (g *gate) concurrentToolCalls() error {
	// Both streams are opened, then each is read to its first generated
	// delta before either is drained: as in seededReplies, that plus the
	// serve log (neither request finished yet) proves the two requests
	// really were generating together, not serialised by the server.
	streams := make([]*toolStream, 2)
	for i := range streams {
		body, lines, err := g.openStream(toolBody)
		if err != nil {
			return fmt.Errorf("open stream %d: %w", i, err)
		}
		defer body.Close()
		streams[i] = &toolStream{lines: lines}
	}
	type firstResult struct {
		i   int
		id  string
		err error
	}
	firsts := make(chan firstResult, 2)
	for i, s := range streams {
		go func() {
			err := s.firstDelta()
			firsts <- firstResult{i, s.id, err}
		}()
	}
	ids := make(map[string]bool)
	seen := 0
	timeout := time.After(60 * time.Second)
	for seen < 2 {
		select {
		case f := <-firsts:
			if f.err != nil {
				return fmt.Errorf("stream %d first delta: %w", f.i, f.err)
			}
			if !strings.HasPrefix(f.id, "chatcmpl-") || ids[f.id] {
				return fmt.Errorf("invalid or duplicated chat id %q", f.id)
			}
			ids[f.id] = true
			seen++
		case <-timeout:
			return fmt.Errorf("only %d of 2 streams produced a first delta in 60s; not concurrent", seen)
		}
	}
	serverLog, err := os.ReadFile(g.log)
	if err != nil {
		return err
	}
	for id := range ids {
		if strings.Contains(string(serverLog), "[serve] #"+strings.TrimPrefix(id, "chatcmpl-")+" prompt ") {
			return fmt.Errorf("%s already finished before both first deltas; the requests did not overlap", id)
		}
	}
	type outcome struct {
		i   int
		err error
	}
	ends := make(chan outcome, 2)
	for i, s := range streams {
		go func() { ends <- outcome{i, s.drain()} }()
	}
	for range streams {
		if o := <-ends; o.err != nil {
			return fmt.Errorf("stream %d: %w", o.i, o.err)
		}
	}
	got := [2]*toolStream{streams[0], streams[1]}
	for i, s := range got {
		if s.finish != "tool_calls" {
			return fmt.Errorf("stream %d finished %q, want tool_calls", i, s.finish)
		}
		if len(s.calls) == 0 {
			return fmt.Errorf("stream %d issued no tool call", i)
		}
		for _, c := range s.calls {
			if c.Name != "get_weather" {
				return fmt.Errorf("stream %d called %q, want get_weather", i, c.Name)
			}
			var args map[string]any
			if err := json.Unmarshal([]byte(c.Args), &args); err != nil {
				return fmt.Errorf("stream %d tool arguments are not JSON: %q", i, c.Args)
			}
			if city, _ := args["city"].(string); !strings.Contains(strings.ToLower(city), "san francisco") {
				return fmt.Errorf("stream %d tool city %q, want San Francisco", i, city)
			}
		}
	}
	// Identical cards: same calls, same arguments. The per-request call
	// ids carry the request sequence and are expected to differ.
	a, b := got[0].calls, got[1].calls
	if len(a) != len(b) {
		return fmt.Errorf("concurrent identical requests issued %d and %d tool calls", len(a), len(b))
	}
	for i := range a {
		if a[i].Name != b[i].Name || a[i].Args != b[i].Args {
			return fmt.Errorf("concurrent identical requests differ at call %d: (%s %s) vs (%s %s)", i, a[i].Name, a[i].Args, b[i].Name, b[i].Args)
		}
	}
	if a[0].ID == b[0].ID {
		return fmt.Errorf("two requests share tool call id %q", a[0].ID)
	}
	return nil
}

func (g *gate) thinkingDisconnect() error {
	offset, err := g.logOffset()
	if err != nil {
		return err
	}
	body := `{"messages":[{"role":"user","content":"` + countPrompt + `"}],"max_tokens":256,"seed":1729,"temperature":0.8,"stream":true,"reasoning_effort":"high"}`
	rc, lines, err := g.openStream(body)
	if err != nil {
		return err
	}
	defer rc.Close()
	// Read until the reply is actually thinking (a reasoning_content
	// delta), not merely open: the disconnect must land mid-thinking.
	thinking := false
	for lines.Scan() {
		line := lines.Text()
		if line == "" {
			continue
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			return fmt.Errorf("unexpected SSE line %q", line)
		}
		if data == "[DONE]" {
			break
		}
		var e toolEvent
		if err := json.Unmarshal([]byte(data), &e); err != nil {
			return err
		}
		if len(e.Choices) == 1 && e.Choices[0].Delta.Reasoning != "" {
			thinking = true
			break
		}
		if len(e.Choices) == 1 && e.Choices[0].Delta.Content != "" {
			return fmt.Errorf("reply produced text before any thinking delta; cannot place a mid-thinking disconnect")
		}
	}
	if err := lines.Err(); err != nil {
		return err
	}
	if !thinking {
		return fmt.Errorf("stream ended before any thinking delta")
	}
	start := time.Now()
	rc.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if err := g.canary(ctx, start); err != nil {
		return err
	}
	return g.confirmFreed(offset, false, 256)
}
