package main

import (
	"fmt"
	"strings"
	"testing"
)

func intPtr(v int) *int       { return &v }
func int64Ptr(v int64) *int64 { return &v }

func sampleToolReplyFixture(seq int, args string) toolSample {
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
		Usage: replyUsage{Prompt: 42, Generated: 15},
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
