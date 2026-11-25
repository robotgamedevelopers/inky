package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func memResponse(status int, body string, headers map[string]string) *http.Response {
	hdr := make(http.Header)
	hdr.Set("Content-Type", "application/json")
	for k, v := range headers {
		hdr.Set(k, v)
	}
	return &http.Response{
		StatusCode: status,
		Header:     hdr,
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

type fakeClock struct {
	mu   sync.Mutex
	now  time.Time
	step time.Duration
}

func newFakeClock() *fakeClock {
	return &fakeClock{now: time.Date(2026, 11, 25, 12, 0, 0, 0, time.UTC), step: 25 * time.Millisecond}
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(c.step)
	return c.now
}

func (c *fakeClock) Sleep(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

func TestLoadCuratedCardsSuite(t *testing.T) {
	cards, err := LoadCards(filepath.Join("..", "..", "tests", "quality", "cards.jsonl"))
	if err != nil {
		t.Fatalf("LoadCards failed: %v", err)
	}
	if len(cards) != 9 {
		t.Fatalf("expected 9 curated cards, got %d", len(cards))
	}
	byKind := map[string]int{}
	toolCards := 0
	diffCards := 0
	goParseCards := 0
	for _, c := range cards {
		byKind[c.Kind]++
		if c.Checks.RequireValidToolCall {
			toolCards++
		}
		if c.Checks.RequireUnifiedDiff {
			diffCards++
		}
		if c.Checks.RequireValidGoParse {
			goParseCards++
		}
	}
	for _, kind := range []string{KindMechanicalEdit, KindBugfix, KindMultiStepAlgorithm} {
		if byKind[kind] != 3 {
			t.Fatalf("kind %s count = %d, want 3", kind, byKind[kind])
		}
	}
	if toolCards < 3 || diffCards < 3 || goParseCards != 9 {
		t.Fatalf("unexpected check mix: tool=%d diff=%d goParse=%d", toolCards, diffCards, goParseCards)
	}
}

func TestParseAndApplyUnifiedDiff(t *testing.T) {
	orig := "package mathutil\n\nfunc Clamp(v, lo, hi int) int {\n\tif v < lo {\n\t\treturn lo\n\t}\n\tif v > hi {\n\t\treturn lo\n\t}\n\treturn v\n}\n"
	goodDiff := "```diff\n--- a/clamp.go\n+++ b/clamp.go\n@@ -5,6 +5,6 @@\n \t\treturn lo\n \t}\n \tif v > hi {\n-\t\treturn lo\n+\t\treturn hi\n \t}\n \treturn v\n```"
	patched, err := ApplyUnifiedDiff(orig, goodDiff)
	if err != nil {
		t.Fatalf("ApplyUnifiedDiff failed: %v", err)
	}
	if !strings.Contains(patched, "\tif v > hi {\n\t\treturn hi\n\t}") {
		t.Fatalf("unexpected patched output:\n%s", patched)
	}
	if _, err := ParseAndCollectGoSymbols(patched); err != nil {
		t.Fatalf("patched Go source failed to parse: %v", err)
	}

	// Negative control: strict hunk line count mismatch
	badCount := "--- a/clamp.go\n+++ b/clamp.go\n@@ -5,9 +5,6 @@\n-\t\treturn lo\n+\t\treturn hi\n"
	if _, err := ParseUnifiedDiffStrict(badCount); err == nil {
		t.Fatal("expected strict hunk line count mismatch error")
	}
	// Tolerant ApplyUnifiedDiff still accepts off-by-1 hunk counts when context matches
	recountDiff := "--- a/clamp.go\n+++ b/clamp.go\n@@ -1,10 +1,10 @@\n package mathutil\n \n func Clamp(v, lo, hi int) int {\n \tif v < lo {\n \t\treturn lo\n \t}\n \tif v > hi {\n-\t\treturn lo\n+\t\treturn hi\n \t}\n \treturn v\n }\n"
	if _, err := ApplyUnifiedDiff(orig, recountDiff); err != nil {
		t.Fatalf("expected tolerant ApplyUnifiedDiff to apply off-by-1 hunk count: %v", err)
	}

	// Negative control: context does not match source
	badCtx := "--- a/clamp.go\n+++ b/clamp.go\n@@ -1,2 +1,2 @@\n-func Missing() {}\n+func Fixed() {}\n"
	if _, err := ApplyUnifiedDiff(orig, badCtx); err == nil {
		t.Fatal("expected context mismatch error")
	}
}

func TestEvaluateCardAllCheckModes(t *testing.T) {
	cards, err := LoadCards(filepath.Join("..", "..", "tests", "quality", "cards.jsonl"))
	if err != nil {
		t.Fatalf("LoadCards: %v", err)
	}
	byID := map[string]Card{}
	for _, c := range cards {
		byID[c.ID] = c
	}

	// 1. Tool call with unified diff inside JSON args (bugfix-clamp-upper)
	clamp := byID["bugfix-clamp-upper"]
	goodPatchArgs, _ := json.Marshal(map[string]string{
		"path":  "clamp.go",
		"patch": "--- a/clamp.go\n+++ b/clamp.go\n@@ -6,5 +6,5 @@\n \t}\n \tif v > hi {\n-\t\treturn lo\n+\t\treturn hi\n \t}\n \treturn v\n",
	})
	res := EvaluateCard(clamp, ChatResult{
		ToolCalls: []ToolCall{{
			ID:       "call_1",
			Type:     "function",
			Function: ToolFunction{Name: "apply_patch", Arguments: string(goodPatchArgs)},
		}},
		FinishReason: "tool_calls",
	})
	if !res.Passed {
		t.Fatalf("expected bugfix-clamp-upper to pass, outcomes=%+v", res.Outcomes)
	}

	// Negative control: malformed tool JSON fails cleanly
	badJSONRes := EvaluateCard(clamp, ChatResult{
		ToolCalls: []ToolCall{{
			ID:       "call_1",
			Type:     "function",
			Function: ToolFunction{Name: "apply_patch", Arguments: "{not-json"},
		}},
	})
	if badJSONRes.Passed {
		t.Fatal("expected malformed tool JSON to fail")
	}

	// 2. Mechanical edit via edit_file tool call (mech-add-field-tag)
	addField := byID["mech-add-field-tag"]
	goodEditArgs, _ := json.Marshal(map[string]string{
		"path":     "config.go",
		"old_text": "\tTimeoutS int    `json:\"timeout_s\"`\n}\n\nfunc DefaultConfig() ServerConfig {\n\treturn ServerConfig{\n\t\tBindAddr: \"127.0.0.1:8090\",\n\t\tTimeoutS: 30,\n\t}\n}",
		"new_text": "\tTimeoutS   int    `json:\"timeout_s\"`\n\tMaxRetries int    `json:\"max_retries\"`\n}\n\nfunc DefaultConfig() ServerConfig {\n\treturn ServerConfig{\n\t\tBindAddr:   \"127.0.0.1:8090\",\n\t\tTimeoutS:   30,\n\t\tMaxRetries: 3,\n\t}\n}",
	})
	editRes := EvaluateCard(addField, ChatResult{
		ToolCalls: []ToolCall{{
			ID:       "call_2",
			Type:     "function",
			Function: ToolFunction{Name: "edit_file", Arguments: string(goodEditArgs)},
		}},
	})
	if !editRes.Passed {
		t.Fatalf("expected mech-add-field-tag to pass, outcomes=%+v", editRes.Outcomes)
	}

	// 3. Multi-step algorithm Go parse + symbols (algo-lru-cache)
	lru := byID["algo-lru-cache"]
	lruCode := "```go\npackage cache\n\nimport \"container/list\"\n\ntype entry struct {\n\tkey string\n\tval int\n}\n\ntype LRUCache struct {\n\tcap   int\n\tll    *list.List\n\titems map[string]*list.Element\n}\n\nfunc NewLRUCache(cap int) *LRUCache {\n\treturn &LRUCache{cap: cap, ll: list.New(), items: make(map[string]*list.Element)}\n}\n\nfunc (c *LRUCache) Get(key string) (int, bool) {\n\tif el, ok := c.items[key]; ok {\n\t\tc.ll.MoveToFront(el)\n\t\treturn el.Value.(*entry).val, true\n\t}\n\treturn 0, false\n}\n\nfunc (c *LRUCache) Put(key string, val int) {\n\tif c.cap <= 0 {\n\t\treturn\n\t}\n\tif el, ok := c.items[key]; ok {\n\t\tel.Value.(*entry).val = val\n\t\tc.ll.MoveToFront(el)\n\t\treturn\n\t}\n\tif c.ll.Len() == c.cap {\n\t\tback := c.ll.Back()\n\t\tc.ll.Remove(back)\n\t\tdelete(c.items, back.Value.(*entry).key)\n\t}\n\tel := c.ll.PushFront(&entry{key: key, val: val})\n\tc.items[key] = el\n}\n```"
	lruRes := EvaluateCard(lru, ChatResult{Content: lruCode, FinishReason: "stop"})
	if !lruRes.Passed {
		t.Fatalf("expected algo-lru-cache to pass, outcomes=%+v", lruRes.Outcomes)
	}
}

func TestCompareModesAndEffortSweep(t *testing.T) {
	ref := ModeRun{
		Name:            "fp16-medium",
		KVCacheMode:     "fp16",
		ReasoningEffort: "medium",
		Health: HealthInfo{
			KV8:                   false,
			KVBytesPerElem:        2,
			PerLaneWorstCaseBytes: 4 << 30,
			MaxAdmittedLanes:      8,
		},
		CardsTotal:  3,
		CardsPassed: 3,
		PassRate:    1.0,
		AvgWallMs:   1200,
		AvgDecodeMs: 900,
		Results: []CardResult{
			{CardID: "c1", Kind: KindMechanicalEdit, Passed: true, NormalizedOutput: "func A() int { return 1 }", Response: ChatResult{CompletionTokens: 10, WallMs: 1000}},
			{CardID: "c2", Kind: KindBugfix, Passed: true, NormalizedOutput: "if v > hi { return hi }", Response: ChatResult{CompletionTokens: 20, WallMs: 1200}},
			{CardID: "c3", Kind: KindMultiStepAlgorithm, Passed: true, NormalizedOutput: "func TopoSort() {}", Response: ChatResult{CompletionTokens: 30, WallMs: 1400}},
		},
	}
	cand := ModeRun{
		Name:            "kv8-medium",
		KVCacheMode:     "kv8",
		ReasoningEffort: "medium",
		Health: HealthInfo{
			KV8:                   true,
			KVBytesPerElem:        1,
			PerLaneWorstCaseBytes: 2 << 30,
			MaxAdmittedLanes:      16,
		},
		CardsTotal:  3,
		CardsPassed: 3,
		PassRate:    1.0,
		AvgWallMs:   1000,
		AvgDecodeMs: 750,
		Results: []CardResult{
			{CardID: "c1", Kind: KindMechanicalEdit, Passed: true, NormalizedOutput: "func A() int { return 1 }", Response: ChatResult{CompletionTokens: 10, WallMs: 900}},
			{CardID: "c2", Kind: KindBugfix, Passed: true, NormalizedOutput: "if v > hi { return hi }", Response: ChatResult{CompletionTokens: 20, WallMs: 1000}},
			{CardID: "c3", Kind: KindMultiStepAlgorithm, Passed: true, NormalizedOutput: "func TopoSort() { return }", Response: ChatResult{CompletionTokens: 32, WallMs: 1100}},
		},
	}

	cmp, err := CompareModes(ref, cand)
	if err != nil {
		t.Fatalf("CompareModes failed: %v", err)
	}
	if cmp.KVElementReductionRatio != 2.0 || cmp.PerLaneMemorySavingRatio != 2.0 {
		t.Fatalf("unexpected memory ratios: elem=%.2f lane=%.2f", cmp.KVElementReductionRatio, cmp.PerLaneMemorySavingRatio)
	}
	if cmp.CheckParityRate != 1.0 {
		t.Fatalf("expected check parity 1.0, got %.2f", cmp.CheckParityRate)
	}
	if cmp.ExactOutputAgreement < 0.66 || cmp.ExactOutputAgreement > 0.67 {
		t.Fatalf("expected 2/3 exact output match, got %.3f", cmp.ExactOutputAgreement)
	}
	if cmp.MeanTokenAgreement <= 0.8 || cmp.MeanTokenAgreement >= 1.0 {
		t.Fatalf("unexpected mean token agreement %.3f", cmp.MeanTokenAgreement)
	}

	runs := []ModeRun{
		{
			ReasoningEffort: "none",
			Results: []CardResult{
				{CardID: "m1", Kind: KindMechanicalEdit, Passed: true, Response: ChatResult{CompletionTokens: 30, WallMs: 300}},
				{CardID: "b1", Kind: KindBugfix, Passed: false, Response: ChatResult{CompletionTokens: 40, WallMs: 400}},
				{CardID: "a1", Kind: KindMultiStepAlgorithm, Passed: false, Response: ChatResult{CompletionTokens: 80, WallMs: 800}},
			},
		},
		{
			ReasoningEffort: "low",
			Results: []CardResult{
				{CardID: "m1", Kind: KindMechanicalEdit, Passed: true, Response: ChatResult{CompletionTokens: 45, WallMs: 450}},
				{CardID: "b1", Kind: KindBugfix, Passed: true, Response: ChatResult{CompletionTokens: 70, WallMs: 700}},
				{CardID: "a1", Kind: KindMultiStepAlgorithm, Passed: false, Response: ChatResult{CompletionTokens: 110, WallMs: 1100}},
			},
		},
		{
			ReasoningEffort: "medium",
			Results: []CardResult{
				{CardID: "m1", Kind: KindMechanicalEdit, Passed: true, Response: ChatResult{CompletionTokens: 60, WallMs: 600}},
				{CardID: "b1", Kind: KindBugfix, Passed: true, Response: ChatResult{CompletionTokens: 90, WallMs: 900}},
				{CardID: "a1", Kind: KindMultiStepAlgorithm, Passed: true, Response: ChatResult{CompletionTokens: 160, WallMs: 1600}},
			},
		},
	}
	sweep := BuildEffortSweep(runs, []string{"none", "low", "medium", "high"})
	got := map[string]string{}
	for _, rec := range sweep.Recommendations {
		got[rec.Kind] = rec.LowestPassingEffort
	}
	if got[KindMechanicalEdit] != "none" || got[KindBugfix] != "low" || got[KindMultiStepAlgorithm] != "medium" {
		t.Fatalf("unexpected lowest passing efforts: %+v", got)
	}
}

func TestRunCLILiveAndMergeWithMockServers(t *testing.T) {
	tmp := t.TempDir()
	cardsPath := filepath.Join(tmp, "cards.jsonl")
	cardLine := `{"id":"c1","kind":"mechanical-edit","title":"const","max_tokens":128,"messages":[{"role":"user","content":"add const X = 1"}],"checks":{"require_valid_go_parse":true,"expected_go_symbols":["X"]}}`
	if err := os.WriteFile(cardsPath, []byte(cardLine+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	makeTransport := func(kv8 bool) http.RoundTripper {
		return roundTripFunc(func(r *http.Request) (*http.Response, error) {
			if r.URL.Path == "/health" {
				kvb := 2
				lanes := 8
				perLane := uint64(4 << 30)
				if kv8 {
					kvb = 1
					lanes = 16
					perLane = 2 << 30
				}
				b, _ := json.Marshal(HealthInfo{
					OK:                    true,
					KV8:                   kv8,
					KVBytesPerElem:        kvb,
					MaxAdmittedLanes:      lanes,
					PerLaneWorstCaseBytes: perLane,
				})
				return memResponse(200, string(b), nil), nil
			}
			body, _ := io.ReadAll(r.Body)
			var req map[string]any
			_ = json.Unmarshal(body, &req)
			eff, _ := req["reasoning_effort"].(string)
			toks := 15
			if eff == "high" {
				toks = 45
			}
			reply, _ := json.Marshal(map[string]any{
				"choices": []map[string]any{{
					"message":       map[string]any{"role": "assistant", "content": "```go\npackage p\nconst X = 1\n```"},
					"finish_reason": "stop",
				}},
				"usage": map[string]any{"prompt_tokens": 50, "completion_tokens": toks},
			})
			return memResponse(200, string(reply), map[string]string{
				"X-Inky-Queue-Ms":   "1",
				"X-Inky-Prefill-Ms": "10",
				"X-Inky-Decode-Ms":  "20",
			}), nil
		})
	}

	// Test side-by-side fp16 vs kv8 CLI mode via distinct URLs handled by one router transport
	router := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.Host == "kv8.local" {
			return makeTransport(true).RoundTrip(r)
		}
		return makeTransport(false).RoundTrip(r)
	})
	client := &http.Client{Transport: router}
	outJSON := filepath.Join(tmp, "cmp.json")
	outMD := filepath.Join(tmp, "cmp.md")
	var stdout, stderr bytes.Buffer
	code := runCLI([]string{
		"-cards", cardsPath,
		"-ref-url", "http://fp16.local",
		"-kv8-url", "http://kv8.local",
		"-efforts", "medium",
		"-out-json", outJSON,
		"-out-md", outMD,
	}, func(string) string { return "" }, client, newFakeClock(), &stdout, &stderr)
	if code != 0 {
		t.Fatalf("runCLI comparison failed (%d): %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "2.0x reduction") {
		t.Fatalf("expected 2.0x reduction in markdown output, got:\n%s", stdout.String())
	}

	// Test effort sweep + merge
	fp16SweepJSON := filepath.Join(tmp, "sweep.json")
	stdout.Reset()
	stderr.Reset()
	code = runCLI([]string{
		"-cards", cardsPath,
		"-url", "http://fp16.local",
		"-name", "fp16",
		"-efforts", "none,high",
		"-out-json", fp16SweepJSON,
	}, func(string) string { return "" }, client, newFakeClock(), &stdout, &stderr)
	if code != 0 {
		t.Fatalf("runCLI sweep failed (%d): %s", code, stderr.String())
	}
	mergedJSON := filepath.Join(tmp, "merged.json")
	stdout.Reset()
	stderr.Reset()
	code = runCLI([]string{
		"-merge", outJSON + "," + fp16SweepJSON,
		"-ref-run", "fp16-medium",
		"-kv8-run", "kv8-medium",
		"-out-json", mergedJSON,
	}, func(string) string { return "" }, nil, nil, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("runCLI merge failed (%d): %s", code, stderr.String())
	}
	rawMerged, err := os.ReadFile(mergedJSON)
	if err != nil {
		t.Fatal(err)
	}
	var merged FullReport
	if err := json.Unmarshal(rawMerged, &merged); err != nil {
		t.Fatal(err)
	}
	if merged.Comparison == nil || merged.EffortSweep == nil {
		t.Fatalf("expected both Comparison and EffortSweep in merged report, got %+v", merged)
	}
}
