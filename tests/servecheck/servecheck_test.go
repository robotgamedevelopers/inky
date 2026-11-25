package main

import (
	"fmt"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func intPtr(v int) *int       { return &v }
func int64Ptr(v int64) *int64 { return &v }

func sampleToolReplyFixture(seq int, args string) toolSample {
	base := time.Date(2026, 11, 25, 0, 0, 0, 0, time.UTC)
	start := base.Add(time.Duration(seq) * time.Millisecond)
	return toolSample{
		ID:     fmt.Sprintf("chatcmpl-%d", seq),
		Role:   "assistant",
		Finish: "tool_calls",
		ToolCalls: []toolCallItem{
			{
				Index: 0,
				ID:    fmt.Sprintf("call_%d_0", seq),
				Type:  "function",
				Function: toolCallFunction{
					Name:      "get_weather",
					Arguments: args,
				},
			},
		},
		Usage:      replyUsage{Prompt: 42, Generated: 15},
		StartedAt:  start,
		ActiveAt:   start.Add(5 * time.Millisecond),
		FinishedAt: base.Add(100*time.Millisecond + time.Duration(seq)*time.Millisecond),
	}
}

func TestVerifyConcurrentToolCalls(t *testing.T) {
	ref := sampleToolReplyFixture(1, `{"city":"Seattle"}`)
	good := []toolSample{
		sampleToolReplyFixture(2, `{"city":"Seattle"}`),
		sampleToolReplyFixture(3, `{"city":"Seattle"}`),
		sampleToolReplyFixture(4, `{"city":"Seattle"}`),
		sampleToolReplyFixture(5, `{"city":"Seattle"}`),
	}
	if err := verifyConcurrentToolCalls("get_weather", ref, good); err != nil {
		t.Fatalf("expected valid concurrent tool calls to pass, got %v", err)
	}

	// Duplicate tool_calls[0].id across concurrent cards must fail.
	dupID := append([]toolSample(nil), good...)
	dupID[2].ToolCalls = append([]toolCallItem(nil), dupID[2].ToolCalls...)
	dupID[2].ToolCalls[0].ID = dupID[0].ToolCalls[0].ID
	if err := verifyConcurrentToolCalls("get_weather", ref, dupID); err == nil || !strings.Contains(err.Error(), "duplicate tool_calls[0].id") {
		t.Fatalf("expected duplicate tool_calls[0].id rejection, got %v", err)
	}

	// Cross-lane argument bleed must fail.
	bleed := append([]toolSample(nil), good...)
	bleed[1].ToolCalls = append([]toolCallItem(nil), bleed[1].ToolCalls...)
	bleed[1].ToolCalls[0].Function.Arguments = `{"city":"Seattle","extra":"bleed"}`
	if err := verifyConcurrentToolCalls("get_weather", ref, bleed); err == nil || !strings.Contains(err.Error(), "argument bleed") {
		t.Fatalf("expected argument bleed rejection, got %v", err)
	}

	// Unparsed arguments must fail.
	unparsed := append([]toolSample(nil), good...)
	unparsed[0].ToolCalls = append([]toolCallItem(nil), unparsed[0].ToolCalls...)
	unparsed[0].ToolCalls[0].Function.Arguments = `{"_unparsed":"broken"}`
	if err := verifyConcurrentToolCalls("get_weather", ref, unparsed); err == nil || !strings.Contains(err.Error(), "unparsed") {
		t.Fatalf("expected unparsed tool arguments rejection, got %v", err)
	}

	// Fewer than 4 concurrent replies must fail.
	if err := verifyConcurrentToolCalls("get_weather", ref, good[:3]); err == nil {
		t.Fatalf("expected <4 concurrent replies to fail")
	}

	// Missing timing metadata must fail.
	noTiming := append([]toolSample(nil), good...)
	noTiming[2].StartedAt = time.Time{}
	if err := verifyConcurrentToolCalls("get_weather", ref, noTiming); err == nil || !strings.Contains(err.Error(), "missing or inverted request timing") {
		t.Fatalf("expected missing timing rejection, got %v", err)
	}

	// Serialized (non-overlapping) requests where request 3 started after request 0 finished must fail.
	serial := append([]toolSample(nil), good...)
	serial[3].StartedAt = serial[0].FinishedAt
	serial[3].FinishedAt = serial[3].StartedAt.Add(50 * time.Millisecond)
	if err := verifyConcurrentToolCalls("get_weather", ref, serial); err == nil || !strings.Contains(err.Error(), "did not overlap in flight") {
		t.Fatalf("expected non-overlapping concurrent tool calls to fail, got %v", err)
	}
}

func TestVerifyDisconnectMidThinking(t *testing.T) {
	ref := replySample{
		message: replyMessage{Role: "assistant", Content: "391"},
		finish:  "stop",
		usage:   replyUsage{Prompt: 20, Generated: 3},
	}
	drained := healthSnapshot{
		OK:             true,
		ActiveLanes:    intPtr(0),
		Inflight:       intPtr(0),
		QueuedRequests: intPtr(0),
	}
	goodLog := []byte("[serve] client gone after 14 generated tokens; lane freed\n")
	if err := verifyDisconnectMidThinking(goodLog, 512, drained, ref, ref); err != nil {
		t.Fatalf("expected clean mid-thinking disconnect to pass, got %v", err)
	}

	// Late cancellation (reaching max_tokens) must fail.
	lateLog := []byte("[serve] client gone after 512 generated tokens; lane freed\n")
	if err := verifyDisconnectMidThinking(lateLog, 512, drained, ref, ref); err == nil {
		t.Fatalf("expected exhausted budget cancellation to fail")
	}

	// Leaked active lane must fail.
	leaked := drained
	leaked.ActiveLanes = intPtr(1)
	if err := verifyDisconnectMidThinking(goodLog, 512, leaked, ref, ref); err == nil {
		t.Fatalf("expected non-zero active_lanes to fail")
	}

	// Follow-up mismatch on freed lane must fail.
	badFollow := ref
	badFollow.message.Content = "392"
	if err := verifyDisconnectMidThinking(goodLog, 512, drained, ref, badFollow); err == nil {
		t.Fatalf("expected mismatched follow-up reply to fail")
	}
}

func TestVerifyHealthBudget(t *testing.T) {
	gib := int64(1 << 30)
	good := healthSnapshot{
		OK:                         true,
		ActiveLanes:                intPtr(0),
		Inflight:                   intPtr(0),
		RecommendedWorkingSetBytes: int64Ptr(115 * gib),
		ModelBytes:                 int64Ptr(16 * gib),
		PerLaneWorstCaseBytes:      int64Ptr(360 * (1 << 20)),
		MaxAdmittedLanes:           intPtr(2),
		QueuedRequests:             intPtr(0),
	}
	if err := verifyHealthBudget(good, 2); err != nil {
		t.Fatalf("expected valid health budget to pass, got %v", err)
	}

	zeroWS := good
	zeroWS.RecommendedWorkingSetBytes = int64Ptr(0)
	if err := verifyHealthBudget(zeroWS, 2); err == nil {
		t.Fatalf("expected zero working set to fail")
	}

	modelTooBig := good
	modelTooBig.ModelBytes = int64Ptr(120 * gib)
	if err := verifyHealthBudget(modelTooBig, 2); err == nil {
		t.Fatalf("expected model_bytes >= recommended_working_set_bytes to fail")
	}

	wrongLanes := good
	wrongLanes.MaxAdmittedLanes = intPtr(1)
	if err := verifyHealthBudget(wrongLanes, 2); err == nil {
		t.Fatalf("expected mismatched max_admitted_lanes to fail")
	}

	leakedQueue := good
	leakedQueue.QueuedRequests = intPtr(1)
	if err := verifyHealthBudget(leakedQueue, 2); err == nil {
		t.Fatalf("expected non-zero queued_requests to fail")
	}
}

func TestVerifyModelsList(t *testing.T) {
	good := []struct{ ID string }{{ID: "inkling-2-mini"}, {ID: "inky/inkling-2-mini"}}
	if err := verifyModelsList(good); err != nil {
		t.Fatalf("expected dual model aliases to pass, got %v", err)
	}
	missing := []struct{ ID string }{{ID: "inkling-2-mini"}}
	if err := verifyModelsList(missing); err == nil {
		t.Fatalf("expected missing namespaced alias to fail")
	}
}

func TestParseAnthropicSSEAndOpenAIUsageStream(t *testing.T) {
	anthropicBody := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"inky/inkling-2-mini","content":[],"stop_reason":null,"usage":{"input_tokens":50,"output_tokens":0}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Checking."}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_n_1_0","name":"get_weather","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Seattle\"}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":1}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"input_tokens":10,"cache_read_input_tokens":40,"output_tokens":12}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	asum, err := parseAnthropicSSE(strings.NewReader(anthropicBody))
	if err != nil {
		t.Fatalf("parseAnthropicSSE failed: %v", err)
	}
	if asum.Model != "inky/inkling-2-mini" || asum.StopReason != "tool_use" || asum.Text != "Checking." || len(asum.ToolUses) != 1 || asum.ToolUses[0].Name != "get_weather" || asum.CacheRead != 40 || asum.OutputTokens != 12 {
		t.Fatalf("unexpected Anthropic SSE summary: %+v", asum)
	}
	if _, err := parseAnthropicSSE(strings.NewReader("event: ping\ndata: {\"type\":\"ping\"}\n\n")); err == nil {
		t.Fatalf("expected missing message_stop to fail")
	}

	openaiBody := strings.Join([]string{
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"inky/inkling-2-mini","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"inky/inkling-2-mini","choices":[{"index":0,"delta":{"content":"58F"},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"inky/inkling-2-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"inky/inkling-2-mini","choices":[],"usage":{"prompt_tokens":40,"completion_tokens":5,"total_tokens":45,"prompt_tokens_details":{"cached_tokens":30}}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	osum, err := parseOpenAIUsageStream(strings.NewReader(openaiBody))
	if err != nil {
		t.Fatalf("parseOpenAIUsageStream failed: %v", err)
	}
	if osum.Content != "58F" || osum.Finish != "stop" || osum.TotalTokens != 45 || !osum.SawUsageChunk || osum.Usage.Details.Cached == nil || *osum.Usage.Details.Cached != 30 {
		t.Fatalf("unexpected OpenAI usage stream summary: %+v", osum)
	}
	noUsageBody := `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\ndata: [DONE]\n\n"
	if _, err := parseOpenAIUsageStream(strings.NewReader(noUsageBody)); err == nil {
		t.Fatalf("expected missing final usage chunk to fail")
	}
}

func TestNegativeControlVerdict(t *testing.T) {
	budget := healthSnapshot{OK: true, RecommendedWorkingSetBytes: int64Ptr(0)}
	late := []byte("[serve] client gone after 512 generated tokens; lane freed\n")
	ref := sampleToolReplyFixture(1, `{"city":"Seattle"}`)
	replies := make([]toolSample, 4)
	for i := range replies {
		replies[i] = sampleToolReplyFixture(i+2, ref.ToolCalls[0].Function.Arguments)
	}
	replies[1].ToolCalls[0].ID = replies[0].ToolCalls[0].ID
	lines := []string{
		fmt.Sprintf("FAIL bearer authentication and public health: sabotage health-status: %v", (apiObservation{code: 200}).expect(418)),
		fmt.Sprintf("FAIL health reports Metal memory budget and admission limits: sabotage memory-budget: %v", verifyHealthBudget(budget, 1)),
		fmt.Sprintf("FAIL disconnect in mid-thinking stops generation: sabotage disconnect-mid-thinking: %v", verifyDisconnectMidThinking(late, 512, healthSnapshot{}, replySample{}, replySample{})),
		fmt.Sprintf("FAIL concurrent identical tool calls: sabotage concurrent-tool-calls: %v", verifyConcurrentToolCalls("get_weather", ref, replies)),
		"one or more serve checks failed",
	}
	complete := strings.Join(lines, "\n")
	for _, tc := range []struct {
		name, log, exit string
		ok              bool
	}{
		{"four observed controls", complete, "1", true},
		{"probe silently passed", strings.Replace(complete, lines[1], "PASS memory budget", 1), "1", false},
		{"wrong diagnostic", strings.Replace(complete, lines[1], "FAIL memory budget: timed out", 1), "1", false},
		{"extra failure", complete + "\nFAIL another probe", "1", false},
		{"incomplete run", strings.Join(lines[:4], "\n"), "1", false},
		{"unexpected success", complete, "0", false},
		{"crash", complete, "2", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cmd := exec.Command("../serve-negative.sh", "sh", "-c", `test "$SERVECHECK_SABOTAGE" = all || exit 3; printf '%s\n' "$1"; exit "$2"`, "runner", tc.log, tc.exit)
			out, err := cmd.CombinedOutput()
			if (err == nil) != tc.ok {
				t.Fatalf("negative control verdict: %v\n%s", err, out)
			}
		})
	}
}
