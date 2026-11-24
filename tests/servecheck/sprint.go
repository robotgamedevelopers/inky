package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

type anthropicToolUse struct {
	ID    string
	Name  string
	Input map[string]any
}

type anthropicStreamSummary struct {
	Model          string
	StopReason     string
	InputTokens    int
	CacheRead      int
	OutputTokens   int
	Text           string
	ToolUses       []anthropicToolUse
	GotMessageStop bool
}

func parseAnthropicSSE(r io.Reader) (anthropicStreamSummary, error) {
	var out anthropicStreamSummary
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)

	type openTool struct {
		id   string
		name string
		json strings.Builder
	}
	openTools := make(map[int]*openTool)
	var curEvent string

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		if ev, ok := strings.CutPrefix(line, "event: "); ok {
			curEvent = ev
			continue
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			return out, fmt.Errorf("unexpected Anthropic SSE line %q", line)
		}
		var payload map[string]any
		if err := json.Unmarshal([]byte(data), &payload); err != nil {
			return out, fmt.Errorf("invalid Anthropic SSE JSON for %s: %w", curEvent, err)
		}
		switch curEvent {
		case "ping":
		case "message_start":
			msg, _ := payload["message"].(map[string]any)
			if msg == nil {
				return out, fmt.Errorf("message_start missing message object")
			}
			out.Model, _ = msg["model"].(string)
			if usage, _ := msg["usage"].(map[string]any); usage != nil {
				if v, ok := usage["input_tokens"].(float64); ok {
					out.InputTokens = int(v)
				}
			}
		case "content_block_start":
			idx := int(payload["index"].(float64))
			cb, _ := payload["content_block"].(map[string]any)
			if cb == nil {
				return out, fmt.Errorf("content_block_start missing content_block")
			}
			if cb["type"] == "tool_use" {
				id, _ := cb["id"].(string)
				name, _ := cb["name"].(string)
				openTools[idx] = &openTool{id: id, name: name}
			}
		case "content_block_delta":
			idx := int(payload["index"].(float64))
			delta, _ := payload["delta"].(map[string]any)
			if delta == nil {
				return out, fmt.Errorf("content_block_delta missing delta")
			}
			switch delta["type"] {
			case "text_delta":
				if t, ok := delta["text"].(string); ok {
					out.Text += t
				}
			case "input_json_delta":
				if pj, ok := delta["partial_json"].(string); ok && openTools[idx] != nil {
					openTools[idx].json.WriteString(pj)
				}
			}
		case "content_block_stop":
			idx := int(payload["index"].(float64))
			if ot := openTools[idx]; ot != nil {
				args, err := parseToolArgsObject(ot.json.String())
				if err != nil {
					return out, fmt.Errorf("tool_use %q: %w", ot.name, err)
				}
				out.ToolUses = append(out.ToolUses, anthropicToolUse{ID: ot.id, Name: ot.name, Input: args})
				delete(openTools, idx)
			}
		case "message_delta":
			if delta, _ := payload["delta"].(map[string]any); delta != nil {
				if sr, ok := delta["stop_reason"].(string); ok {
					out.StopReason = sr
				}
			}
			if usage, _ := payload["usage"].(map[string]any); usage != nil {
				if v, ok := usage["input_tokens"].(float64); ok {
					out.InputTokens = int(v)
				}
				if v, ok := usage["cache_read_input_tokens"].(float64); ok {
					out.CacheRead = int(v)
				}
				if v, ok := usage["output_tokens"].(float64); ok {
					out.OutputTokens = int(v)
				}
			}
		case "message_stop":
			out.GotMessageStop = true
		case "error":
			return out, fmt.Errorf("Anthropic SSE error event: %s", data)
		}
	}
	if err := scanner.Err(); err != nil {
		return out, err
	}
	if !out.GotMessageStop {
		return out, fmt.Errorf("missing Anthropic message_stop event")
	}
	return out, nil
}

type openAIUsageStreamSummary struct {
	Model          string
	Finish         string
	Content        string
	ToolCalls      []toolCallItem
	Usage          replyUsage
	TotalTokens    int
	SawUsageChunk  bool
	GotDone        bool
}

func parseOpenAIUsageStream(r io.Reader) (openAIUsageStreamSummary, error) {
	var out openAIUsageStreamSummary
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			return out, fmt.Errorf("unexpected OpenAI SSE line %q", line)
		}
		if data == "[DONE]" {
			out.GotDone = true
			break
		}
		var chunk struct {
			ID      string `json:"id"`
			Model   string `json:"model"`
			Choices []struct {
				Delta  toolMessage `json:"delta"`
				Finish string      `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				Prompt    int `json:"prompt_tokens"`
				Generated int `json:"completion_tokens"`
				Total     int `json:"total_tokens"`
				Details   struct {
					Cached *int `json:"cached_tokens"`
				} `json:"prompt_tokens_details"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return out, fmt.Errorf("invalid OpenAI SSE JSON: %w", err)
		}
		if chunk.Model != "" {
			out.Model = chunk.Model
		}
		if len(chunk.Choices) == 0 && chunk.Usage != nil {
			out.SawUsageChunk = true
			out.Usage.Prompt = chunk.Usage.Prompt
			out.Usage.Generated = chunk.Usage.Generated
			out.Usage.Details.Cached = chunk.Usage.Details.Cached
			out.TotalTokens = chunk.Usage.Total
			continue
		}
		if len(chunk.Choices) != 1 {
			return out, fmt.Errorf("expected 1 choice or final usage chunk, got %d choices in %s", len(chunk.Choices), data)
		}
		ch := chunk.Choices[0]
		out.Content += ch.Delta.Content
		if len(ch.Delta.ToolCalls) > 0 {
			out.ToolCalls = append(out.ToolCalls, ch.Delta.ToolCalls...)
		}
		if ch.Finish != "" {
			out.Finish = ch.Finish
		}
	}
	if err := scanner.Err(); err != nil {
		return out, err
	}
	if !out.GotDone {
		return out, fmt.Errorf("missing OpenAI SSE [DONE]")
	}
	if !out.SawUsageChunk {
		return out, fmt.Errorf("missing final choices:[] usage chunk for stream_options.include_usage=true")
	}
	return out, nil
}

func (g *gate) claudeHarnessMessagesLoop() error {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	// Reject malformed thinking budget_tokens < 1024 with HTTP 400 (PR #36).
	badReq := `{"model":"inky/inkling-2-mini","max_tokens":32,"thinking":{"type":"enabled","budget_tokens":512},"messages":[{"role":"user","content":"Hi"}]}`
	badObs, err := g.observeHeaders("/v1/messages", map[string]string{"x-api-key": g.key}, []byte(badReq))
	if err != nil {
		return err
	}
	if err := badObs.expect(400); err != nil {
		return fmt.Errorf("expected 400 for budget_tokens=512: %w", err)
	}

	system := []map[string]any{
		{
			"type":          "text",
			"text":          "You are a tool-using assistant. Whenever the user asks about weather, immediately invoke get_weather with the requested city and do not ask follow-up questions.",
			"cache_control": map[string]string{"type": "ephemeral"},
		},
	}
	tools := []map[string]any{
		{
			"name":          "get_weather",
			"description":   "Look up current weather for a city.",
			"cache_control": map[string]string{"type": "ephemeral"},
			"input_schema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string", "description": "City name"},
				},
				"required": []string{"city"},
			},
		},
	}

	postMessages := func(payload map[string]any) (*http.Response, error) {
		raw, err := json.Marshal(payload)
		if err != nil {
			return nil, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, g.url+"/v1/messages?beta=true", strings.NewReader(string(raw)))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-api-key", g.key)
		return replyClient.Do(req)
	}

	// Turn 1: streaming tool_use with thinking enabled (budget_tokens=1024 -> low effort) and cache_control.
	turn1Payload := map[string]any{
		"model":       "inky/inkling-2-mini",
		"max_tokens":  128,
		"temperature": 0.0,
		"stream":      true,
		"thinking":    map[string]any{"type": "enabled", "budget_tokens": 1024},
		"output_config": map[string]any{"effort": "none"},
		"system":      system,
		"tools":       tools,
		"messages": []map[string]any{
			{
				"role": "user",
				"content": []map[string]any{
					{"type": "text", "text": "What is the current weather in Seattle? Call get_weather now.", "cache_control": map[string]string{"type": "ephemeral"}},
				},
			},
		},
	}
	resp1, err := postMessages(turn1Payload)
	if err != nil {
		return err
	}
	defer resp1.Body.Close()
	if resp1.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp1.Body, 512))
		return fmt.Errorf("turn 1 HTTP %d: %s", resp1.StatusCode, msg)
	}
	sum1, err := parseAnthropicSSE(resp1.Body)
	if err != nil {
		return fmt.Errorf("turn 1 SSE: %w", err)
	}
	if sum1.Model != "inky/inkling-2-mini" {
		return fmt.Errorf("turn 1 model=%q, want inky/inkling-2-mini", sum1.Model)
	}
	if sum1.StopReason != "tool_use" || len(sum1.ToolUses) == 0 {
		return fmt.Errorf("turn 1 expected tool_use, got stop=%q tools=%d text=%q", sum1.StopReason, len(sum1.ToolUses), sum1.Text)
	}
	tu := sum1.ToolUses[0]
	if tu.Name != "get_weather" || !strings.HasPrefix(tu.ID, "toolu_") {
		return fmt.Errorf("turn 1 unexpected tool_use: %+v", tu)
	}

	// Turn 2: multi-turn tool_result (array-of-blocks content with cache_control + is_error=false) -> final answer + timing headers.
	turn2Payload := map[string]any{
		"model":         "inky/inkling-2-mini",
		"max_tokens":    64,
		"temperature":   0.0,
		"stream":        false,
		"thinking":      map[string]any{"type": "disabled"},
		"system":        system,
		"tools":         tools,
		"messages": []map[string]any{
			{
				"role": "user",
				"content": []map[string]any{
					{"type": "text", "text": "What is the current weather in Seattle? Call get_weather now.", "cache_control": map[string]string{"type": "ephemeral"}},
				},
			},
			{
				"role": "assistant",
				"content": []map[string]any{
					{"type": "tool_use", "id": tu.ID, "name": tu.Name, "input": tu.Input},
				},
			},
			{
				"role": "user",
				"content": []map[string]any{
					{
						"type":          "tool_result",
						"tool_use_id":   tu.ID,
						"is_error":      false,
						"cache_control": map[string]string{"type": "ephemeral"},
						"content": []map[string]any{
							{"type": "text", "text": "Seattle is 58F and rainy.", "cache_control": map[string]string{"type": "ephemeral"}},
						},
					},
				},
			},
		},
	}
	resp2, err := postMessages(turn2Payload)
	if err != nil {
		return err
	}
	defer resp2.Body.Close()
	if resp2.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp2.Body, 512))
		return fmt.Errorf("turn 2 HTTP %d: %s", resp2.StatusCode, msg)
	}
	for _, hdr := range []string{"X-Inky-Queue-Ms", "X-Inky-Prefill-Ms", "X-Inky-Decode-Ms"} {
		val := resp2.Header.Get(hdr)
		if _, err := strconv.ParseUint(val, 10, 64); err != nil {
			return fmt.Errorf("turn 2 missing/invalid timing header %s=%q", hdr, val)
		}
	}
	var msg2 struct {
		Model      string `json:"model"`
		StopReason string `json:"stop_reason"`
		Content    []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		Usage struct {
			InputTokens          int `json:"input_tokens"`
			CacheReadInputTokens int `json:"cache_read_input_tokens"`
			OutputTokens         int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp2.Body).Decode(&msg2); err != nil {
		return fmt.Errorf("turn 2 decode: %w", err)
	}
	if msg2.StopReason != "end_turn" && msg2.StopReason != "max_tokens" {
		return fmt.Errorf("turn 2 unexpected stop_reason %q", msg2.StopReason)
	}
	if msg2.Usage.OutputTokens <= 0 || msg2.Usage.CacheReadInputTokens <= 0 {
		return fmt.Errorf("turn 2 expected cached prompt reuse and output tokens, got %+v", msg2.Usage)
	}
	serverLog, err := os.ReadFile(g.log)
	if err != nil {
		return err
	}
	if !strings.Contains(string(serverLog), "queue_ms=") || !strings.Contains(string(serverLog), "prefill_ms=") || !strings.Contains(string(serverLog), "decode_ms=") {
		return fmt.Errorf("server log missing queue_ms/prefill_ms/decode_ms timing telemetry")
	}
	return nil
}

func (g *gate) opencodeHarnessChatLoop() error {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	tools := []map[string]any{
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
	}

	postChat := func(payload map[string]any) (openAIUsageStreamSummary, error) {
		raw, err := json.Marshal(payload)
		if err != nil {
			return openAIUsageStreamSummary{}, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, g.url+"/v1/chat/completions", strings.NewReader(string(raw)))
		if err != nil {
			return openAIUsageStreamSummary{}, err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+g.key)
		resp, err := replyClient.Do(req)
		if err != nil {
			return openAIUsageStreamSummary{}, err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
			return openAIUsageStreamSummary{}, fmt.Errorf("HTTP %d: %s", resp.StatusCode, msg)
		}
		return parseOpenAIUsageStream(resp.Body)
	}

	// Turn 1: streaming with stream_options.include_usage=true, opencode's default
	// max_tokens=128000 (clamped cleanly to lane context), and tool_calls.
	sum1, err := postChat(map[string]any{
		"model":            "inky/inkling-2-mini",
		"stream":           true,
		"stream_options":   map[string]any{"include_usage": true},
		"max_tokens":       128000,
		"seed":             4242,
		"temperature":      0.0,
		"reasoning_effort": "none",
		"tools":            tools,
		"messages": []map[string]any{
			{"role": "system", "content": "You are a tool-using assistant. Whenever the user asks about weather, immediately invoke get_weather with the requested city and do not ask follow-up questions."},
			{"role": "user", "content": "What is the current weather in Seattle? Call get_weather now."},
		},
	})
	if err != nil {
		return fmt.Errorf("turn 1: %w", err)
	}
	if sum1.Model != "inky/inkling-2-mini" || sum1.Finish != "tool_calls" || len(sum1.ToolCalls) == 0 {
		return fmt.Errorf("turn 1 unexpected result: model=%q finish=%q calls=%d", sum1.Model, sum1.Finish, len(sum1.ToolCalls))
	}
	if sum1.Usage.Prompt <= 0 || sum1.Usage.Generated <= 0 || sum1.TotalTokens != sum1.Usage.Prompt+sum1.Usage.Generated {
		return fmt.Errorf("turn 1 invalid final usage chunk: %+v total=%d", sum1.Usage, sum1.TotalTokens)
	}

	// Turn 2: feed back tool_calls + tool result, verify final answer + cached_tokens + include_usage chunk.
	tc := sum1.ToolCalls[0]
	sum2, err := postChat(map[string]any{
		"model":            "inky/inkling-2-mini",
		"stream":           true,
		"stream_options":   map[string]any{"include_usage": true},
		"max_tokens":       64,
		"seed":             4242,
		"temperature":      0.0,
		"reasoning_effort": "none",
		"tools":            tools,
		"messages": []map[string]any{
			{"role": "system", "content": "You are a tool-using assistant. Whenever the user asks about weather, immediately invoke get_weather with the requested city and do not ask follow-up questions."},
			{"role": "user", "content": "What is the current weather in Seattle? Call get_weather now."},
			{"role": "assistant", "content": nil, "tool_calls": []toolCallItem{tc}},
			{"role": "tool", "tool_call_id": tc.ID, "content": "Seattle is 58F and rainy."},
		},
	})
	if err != nil {
		return fmt.Errorf("turn 2: %w", err)
	}
	if sum2.Finish != "stop" && sum2.Finish != "length" {
		return fmt.Errorf("turn 2 unexpected finish %q", sum2.Finish)
	}
	if strings.TrimSpace(sum2.Content) == "" {
		return fmt.Errorf("turn 2 empty assistant content")
	}
	if sum2.TotalTokens != sum2.Usage.Prompt+sum2.Usage.Generated || sum2.Usage.Details.Cached == nil || *sum2.Usage.Details.Cached <= 0 {
		return fmt.Errorf("turn 2 expected positive cached_tokens and valid total_tokens, got %+v total=%d", sum2.Usage, sum2.TotalTokens)
	}
	return nil
}
