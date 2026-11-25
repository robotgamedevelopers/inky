package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestExecuteTurnMeasuresCompletedReplies(t *testing.T) {
	const chat = "/v1/chat/completions"
	const messages = "/v1/messages?beta=true"
	const buffered = "{}"
	const stream = `{"stream":true}`
	const separateUsage = `{"stream":true,"stream_options":{"include_usage":true}}`
	const chatBody = `{"choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"completion_tokens":2}}`
	const msgBody = `{"type":"message","role":"assistant","content":[],"stop_reason":"end_turn","usage":{"output_tokens":0}}`
	frame := func(payload string) string { return "data: " + payload + "\n\n" }
	tool := frame(`{"choices":[{"delta":{"tool_calls":[{"function":{"name":"run"}}]}}]}`)
	stop := frame(`{"choices":[{"delta":{},"finish_reason":"stop"}]}`)
	usage := frame(`{"choices":[],"usage":{"completion_tokens":7}}`)
	inline := frame(`{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":0}}`)
	done := frame("[DONE]")
	start := frame(`{"type":"message_start","message":{"type":"message","role":"assistant","usage":{"output_tokens":0}}}`)
	block := frame(`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)
	textDelta := frame(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}`)
	closeBlock := frame(`{"type":"content_block_stop","index":0}`)
	delta := frame(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}`)
	end := frame(`{"type":"message_stop"}`)
	for _, tt := range []struct {
		name, endpoint, request, body string
		tokens                        int // -1 for invalid replies
		hasTTFT                       bool
	}{
		{"buffered chat", chat, buffered, chatBody, 2, false},
		{"buffered messages zero", messages, buffered, msgBody, 0, false},
		{"buffered missing choice", chat, buffered, `{"usage":{"completion_tokens":5}}`, -1, false},
		{"buffered missing finish", chat, buffered, strings.Replace(chatBody, `"stop"`, "null", 1), -1, false},
		{"buffered messages missing finish", messages, buffered, strings.Replace(msgBody, `"end_turn"`, "null", 1), -1, false},
		{"buffered wrong dialect", messages, buffered, chatBody, -1, false},
		{"buffered missing usage", messages, buffered, strings.Replace(msgBody, "output_tokens", "completion_tokens", 1), -1, false},
		{"buffered negative usage", chat, buffered, strings.Replace(chatBody, ":2", ":-1", 1), -1, false},
		{"chat tool call", chat, separateUsage, tool + stop + usage + done, 7, true},
		{"chat empty zero", chat, stream, inline + done, 0, false},
		{"chat CRLF and SSE comment", chat, stream, strings.ReplaceAll(": keepalive\n\n"+inline+done, "\n", "\r\n"), 0, false},
		{"chat missing DONE", chat, stream, inline, -1, false},
		{"chat missing finish", chat, stream, tool + usage + done, -1, false},
		{"chat partial DONE frame", chat, stream, inline + strings.TrimSuffix(done, "\n"), -1, false},
		{"chat missing usage", chat, stream, tool + stop + done, -1, false},
		{"chat missing requested usage envelope", chat, separateUsage, inline + done, -1, false},
		{"chat unrecognized chunk", chat, stream, frame("{}") + inline + done, -1, false},
		{"chat early DONE", chat, stream, done + inline, -1, false},
		{"messages text", messages, stream, start + block + textDelta + closeBlock + delta + end, 3, true},
		{"messages tool call", messages, stream, start + frame(`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","name":"run"}}`) + closeBlock + delta + end, 3, true},
		{"messages event mismatch", messages, stream, "event: message_stop\n" + start + delta + end, -1, false},
		{"messages empty zero", messages, stream, start + strings.Replace(delta, ":3", ":0", 1) + end, 0, false},
		{"messages missing start", messages, stream, delta + end, -1, false},
		{"messages missing stop", messages, stream, start + delta, -1, false},
		{"messages missing delta", messages, stream, start + end, -1, false},
		{"messages missing output count", messages, stream, start + strings.Replace(delta, "output_tokens", "input_tokens", 1) + end, -1, false},
		{"messages truncated block", messages, stream, start + block + textDelta + delta + end, -1, false},
		{"messages misplaced text", messages, stream, start + textDelta + delta + end, -1, false},
		{"error after content", messages, stream, start + block + textDelta + frame(`{"type":"error","error":{"message":"failed"}}`), -1, false},
		{"wrong streaming dialect", chat, stream, start + delta + end, -1, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, tt.body) }))
			defer srv.Close()
			turn := RecordedEntry{Endpoint: tt.endpoint, Request: []byte(tt.request)}
			got, ttft, _, err := executeTurn(context.Background(), srv.Client(), newFakeClock(time.Unix(0, 0)), srv.URL, "", "card", turn, 0)
			if tt.tokens < 0 {
				if err == nil {
					t.Fatalf("accepted incomplete response (tokens=%d, TTFT=%v)", got, ttft)
				}
				return
			}
			if err != nil || got != tt.tokens || (tt.hasTTFT && ttft <= 0) || (!tt.hasTTFT && ttft != -1) {
				t.Errorf("tokens=%d, TTFT=%v, err=%v; want tokens=%d, observable TTFT=%v", got, ttft, err, tt.tokens, tt.hasTTFT)
			}
		})
	}
}
