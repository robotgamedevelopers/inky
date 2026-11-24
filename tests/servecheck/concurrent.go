package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
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

// Drain one streamed reply to its tool calls, in call order.
func drainToolCalls(lines *bufio.Scanner) ([]toolCall, string, error) {
	var calls []toolCall
	finish := ""
	for lines.Scan() {
		line := lines.Text()
		if line == "" {
			continue
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			return nil, "", fmt.Errorf("unexpected SSE line %q", line)
		}
		if data == "[DONE]" {
			return calls, finish, nil
		}
		var e toolEvent
		if err := json.Unmarshal([]byte(data), &e); err != nil {
			return nil, "", err
		}
		if len(e.Choices) != 1 {
			return nil, "", fmt.Errorf("missing completion choice: %s", data)
		}
		ch := e.Choices[0]
		for _, c := range ch.Delta.Calls {
			for len(calls) <= c.Index {
				calls = append(calls, toolCall{Index: len(calls)})
			}
			if c.ID != "" {
				calls[c.Index].ID = c.ID
			}
			if c.Function.Name != "" {
				calls[c.Index].Name = c.Function.Name
			}
			calls[c.Index].Args += c.Function.Arguments
		}
		if ch.Finish != "" {
			finish = ch.Finish
		}
	}
	if err := lines.Err(); err != nil {
		return nil, "", err
	}
	return nil, "", io.ErrUnexpectedEOF
}

func (g *gate) concurrentToolCalls() error {
	// Both streams are open before either is drained: the requests really
	// are in flight together, on two lanes, with the same seed.
	bodies := make([]io.ReadCloser, 2)
	scans := make([]*bufio.Scanner, 2)
	for i := range bodies {
		body, lines, err := g.openStream(toolBody)
		if err != nil {
			return fmt.Errorf("open stream %d: %w", i, err)
		}
		defer body.Close()
		bodies[i], scans[i] = body, lines
	}
	type outcome struct {
		calls  []toolCall
		finish string
		err    error
	}
	ends := make(chan outcome, 2)
	for i := range scans {
		go func(lines *bufio.Scanner) {
			c, f, e := drainToolCalls(lines)
			ends <- outcome{c, f, e}
		}(scans[i])
	}
	var got [2]outcome
	for i := range got {
		got[i] = <-ends
		if got[i].err != nil {
			return fmt.Errorf("stream %d: %w", i, got[i].err)
		}
	}
	for i, o := range got {
		if o.finish != "tool_calls" {
			return fmt.Errorf("stream %d finished %q, want tool_calls", i, o.finish)
		}
		if len(o.calls) == 0 {
			return fmt.Errorf("stream %d issued no tool call", i)
		}
		for _, c := range o.calls {
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
