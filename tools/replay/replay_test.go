package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type fakeClock struct {
	mu      sync.Mutex
	now     time.Time
	sleeps  []time.Duration
	stepPer time.Duration
}

func newFakeClock(start time.Time) *fakeClock {
	return &fakeClock{now: start, stepPer: 10 * time.Millisecond}
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(c.stepPer)
	return c.now
}

func (c *fakeClock) Sleep(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.sleeps = append(c.sleeps, d)
	c.now = c.now.Add(d)
}

func (c *fakeClock) Sleeps() []time.Duration {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]time.Duration, len(c.sleeps))
	copy(out, c.sleeps)
	return out
}

func localTempDir(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp(".", ".tmp-replay-test-*")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	return dir
}

func TestLoadReplaySet_GroupsAndOrdersMultiTurnCards(t *testing.T) {
	cards, err := LoadReplaySet(filepath.Join("testdata", "sample_cards.jsonl"))
	if err != nil {
		t.Fatalf("LoadReplaySet sample_cards.jsonl: %v", err)
	}
	if len(cards) != 4 {
		t.Fatalf("got %d cards, want 4", len(cards))
	}
	if cards[0].ID != "card-fix-1" || len(cards[0].Turns) != 3 {
		t.Fatalf("unexpected card[0]: %+v", cards[0])
	}
	for i, turn := range cards[0].Turns {
		if turn.TurnIndex != uint64(i) {
			t.Fatalf("card-fix-1 turn %d has turn_index=%d", i, turn.TurnIndex)
		}
	}

	// Directory loading with out-of-order lines and a 400 entry that should be skipped.
	dir := localTempDir(t)
	lineA1 := `{"schema_version":1,"recorded_at_ms":2000,"card_id":"c-multi","turn_index":1,"endpoint":"/v1/chat/completions","request":{"messages":[{"role":"user","content":"t1"}],"stream":false},"response":{"prompt_tokens":10,"completion_tokens":5,"ttft_ms":1,"total_ms":2,"finish_reason":"stop","status_code":200}}`
	lineA0 := `{"schema_version":1,"recorded_at_ms":1000,"card_id":"c-multi","turn_index":0,"endpoint":"/v1/chat/completions","request":{"messages":[{"role":"user","content":"t0"}],"stream":false},"response":{"prompt_tokens":5,"completion_tokens":5,"ttft_ms":1,"total_ms":2,"finish_reason":"stop","status_code":200}}`
	lineBad := `{"schema_version":1,"recorded_at_ms":1500,"card_id":"c-bad","turn_index":0,"endpoint":"/v1/chat/completions","request":{"messages":[]},"response":{"prompt_tokens":0,"completion_tokens":0,"ttft_ms":0,"total_ms":1,"finish_reason":"invalid_request","status_code":400}}`
	if err := os.WriteFile(filepath.Join(dir, "cards-20261124.jsonl"), []byte(lineA1+"\n"+lineBad+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "cards-20261124.jsonl.1"), []byte(lineA0+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	dirCards, err := LoadReplaySet(dir)
	if err != nil {
		t.Fatalf("LoadReplaySet dir: %v", err)
	}
	if len(dirCards) != 1 || dirCards[0].ID != "c-multi" || len(dirCards[0].Turns) != 2 {
		t.Fatalf("unexpected dirCards: %+v", dirCards)
	}
	if dirCards[0].Turns[0].TurnIndex != 0 || dirCards[0].Turns[1].TurnIndex != 1 {
		t.Fatalf("turns not sorted: %+v", dirCards[0].Turns)
	}
}

func TestLoadReplaySetKeepsGenerationAndDeliveryOutcomes(t *testing.T) {
	const rows = `{"schema_version":1,"card_id":"sample","turn_index":0,"request":{"messages":[]},"response":{"finish_reason":"stop","status_code":200}}
{"schema_version":1,"card_id":"sample","turn_index":1,"request":{"messages":[]},"response":{"finish_reason":"cancelled","status_code":499,"generation_finish_reason":"end_turn","delivery_status":"failed"}}
{"schema_version":1,"card_id":"sample","turn_index":2,"request":{"messages":[]},"response":{"finish_reason":"length","status_code":200,"generation_finish_reason":"length","delivery_status":"written"}}`
	path := filepath.Join(t.TempDir(), "cards.jsonl")
	if err := os.WriteFile(path, []byte(rows), 0o600); err != nil {
		t.Fatal(err)
	}
	cards, err := LoadReplaySet(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(cards) != 1 || len(cards[0].Turns) != 3 {
		t.Fatalf("unexpected replay set: %+v", cards)
	}
	want := []ResponseMetrics{
		{StatusCode: 200, FinishReason: "stop"},
		{StatusCode: 499, FinishReason: "cancelled", GenerationFinishReason: "end_turn", DeliveryStatus: "failed"},
		{StatusCode: 200, FinishReason: "length", GenerationFinishReason: "length", DeliveryStatus: "written"},
	}
	for i, turn := range cards[0].Turns {
		if turn.Response != want[i] {
			t.Errorf("turn %d: got %+v, want %+v", i, turn.Response, want[i])
		}
	}
}

func TestParseRetryAfter(t *testing.T) {
	now := time.Date(2026, 11, 24, 18, 0, 0, 0, time.UTC)
	if d, ok := ParseRetryAfter("5", now); !ok || d != 5*time.Second {
		t.Fatalf("got (%v, %v), want (5s, true)", d, ok)
	}
	if d, ok := ParseRetryAfter("999", now); !ok || d != 60*time.Second {
		t.Fatalf("expected clamp to 60s, got (%v, %v)", d, ok)
	}
	httpDate := now.Add(3 * time.Second).Format(http.TimeFormat)
	if d, ok := ParseRetryAfter(httpDate, now); !ok || d != 3*time.Second {
		t.Fatalf("HTTP date: got (%v, %v), want (3s, true)", d, ok)
	}
	if _, ok := ParseRetryAfter("garbage", now); ok {
		t.Fatal("expected false for invalid Retry-After")
	}
}

func TestPickKnee(t *testing.T) {
	t.Run("saturation curve picks knee below 5 percent per doubling", func(t *testing.T) {
		rungs := []RungMetrics{
			{Lanes: 8, TotalTokS: 320, PerLaneTokS: 40, TTFTP95Ms: 40, CardsCompleted: 16, CardsPerHour: 100},
			{Lanes: 16, TotalTokS: 600, PerLaneTokS: 37.5, TTFTP95Ms: 55, CardsCompleted: 32, CardsPerHour: 185},
			{Lanes: 32, TotalTokS: 960, PerLaneTokS: 30, TTFTP95Ms: 80, CardsCompleted: 64, CardsPerHour: 295},
			{Lanes: 48, TotalTokS: 1120, PerLaneTokS: 23.3, TTFTP95Ms: 110, CardsCompleted: 96, CardsPerHour: 345},
			{Lanes: 64, TotalTokS: 1210, PerLaneTokS: 18.9, TTFTP95Ms: 140, CardsCompleted: 128, CardsPerHour: 375},
			{Lanes: 96, TotalTokS: 1220, PerLaneTokS: 12.7, TTFTP95Ms: 260, CardsCompleted: 128, CardsPerHour: 378},
			{Lanes: 128, TotalTokS: 1180, PerLaneTokS: 9.2, TTFTP95Ms: 480, CardsCompleted: 128, CardsPerHour: 360},
		}
		res, err := PickKnee(rungs, DefaultKneeConfig())
		if err != nil {
			t.Fatal(err)
		}
		if res.RecommendedLanes != 64 {
			t.Fatalf("got %d lanes (%s), want 64", res.RecommendedLanes, res.Reason)
		}
	})

	t.Run("p95 TTFT ceiling and per-lane floor stop earlier", func(t *testing.T) {
		rungs := []RungMetrics{
			{Lanes: 8, TotalTokS: 320, PerLaneTokS: 40, TTFTSamples: 1, TTFTP95Ms: 50, CardsCompleted: 8, CardsPerHour: 100},
			{Lanes: 16, TotalTokS: 600, PerLaneTokS: 37.5, TTFTSamples: 1, TTFTP95Ms: 90, CardsCompleted: 16, CardsPerHour: 190},
			{Lanes: 32, TotalTokS: 1000, PerLaneTokS: 31.25, TTFTSamples: 1, TTFTP95Ms: 250, CardsCompleted: 32, CardsPerHour: 320},
		}
		res, err := PickKnee(rungs, KneeConfig{MinMarginalGain: 0.05, MaxTTFTP95Ms: 200})
		if err != nil {
			t.Fatal(err)
		}
		if res.RecommendedLanes != 16 {
			t.Fatalf("TTFT ceiling: got %d, want 16 (%s)", res.RecommendedLanes, res.Reason)
		}

		res2, err := PickKnee(rungs, KneeConfig{MinMarginalGain: 0.05, MinPerLaneTokS: 35})
		if err != nil {
			t.Fatal(err)
		}
		if res2.RecommendedLanes != 16 {
			t.Fatalf("per-lane floor: got %d, want 16 (%s)", res2.RecommendedLanes, res2.Reason)
		}
	})

	t.Run("errors halt scaling", func(t *testing.T) {
		rungs := []RungMetrics{
			{Lanes: 8, TotalTokS: 300, PerLaneTokS: 37.5, CardsCompleted: 8, CardsPerHour: 100, Errors: 0},
			{Lanes: 16, TotalTokS: 500, PerLaneTokS: 31.2, CardsCompleted: 15, CardsPerHour: 170, Errors: 1},
		}
		res, err := PickKnee(rungs, DefaultKneeConfig())
		if err != nil {
			t.Fatal(err)
		}
		if res.RecommendedLanes != 8 {
			t.Fatalf("got %d, want 8", res.RecommendedLanes)
		}
	})

	t.Run("empty or zero completed returns error", func(t *testing.T) {
		if _, err := PickKnee(nil, DefaultKneeConfig()); err == nil {
			t.Fatal("expected error for empty rungs")
		}
		if _, err := PickKnee([]RungMetrics{{Lanes: 8, Errors: 4}}, DefaultKneeConfig()); err == nil {
			t.Fatal("expected error for zero completed cards")
		}
	})
}

func TestRunSweep_HttptestMultiTurnAndHealthAnd429Retry(t *testing.T) {
	cards, err := LoadReplaySet(filepath.Join("testdata", "sample_cards.jsonl"))
	if err != nil {
		t.Fatal(err)
	}

	var mu sync.Mutex
	turnSeqByCard := make(map[string][]int)
	var reqCount atomic.Int64
	var sawAuth atomic.Bool

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"ok":true,"active_lanes":4,"inflight":5,"model_bytes":1000000,"per_lane_worst_case_bytes":250000}`)
			return
		}
		if r.Header.Get("Authorization") == "Bearer secret-replay-token" {
			sawAuth.Store(true)
		}
		n := reqCount.Add(1)
		// Inject deterministic 429 + Retry-After on the 2nd request.
		if n == 2 {
			w.Header().Set("Retry-After", "2")
			w.WriteHeader(http.StatusTooManyRequests)
			fmt.Fprint(w, `{"error":{"message":"busy"}}`)
			return
		}

		var body struct {
			Stream   bool              `json:"stream"`
			Messages []json.RawMessage `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		cardID := r.Header.Get("X-Inky-Card-Id")
		mu.Lock()
		turnSeqByCard[cardID] = append(turnSeqByCard[cardID], len(body.Messages))
		mu.Unlock()

		if body.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":16}}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":12}}`)
	}))
	defer srv.Close()

	clk := newFakeClock(time.Date(2026, 11, 24, 18, 0, 0, 0, time.UTC))
	cfg := SweepConfig{
		BaseURL:        srv.URL,
		APIKey:         "secret-replay-token",
		LaneCounts:     []int{2, 4},
		CardsPerRung:   4,
		MaxRetries:     4,
		HealthInterval: 50 * time.Millisecond,
		Knee:           DefaultKneeConfig(),
		Client:         srv.Client(),
		Clock:          clk,
	}

	rungs, knee, err := RunSweep(context.Background(), cfg, cards)
	if err != nil {
		t.Fatalf("RunSweep: %v", err)
	}
	if len(rungs) != 2 {
		t.Fatalf("got %d rungs, want 2", len(rungs))
	}
	if !sawAuth.Load() {
		t.Fatal("expected Authorization bearer header to be sent")
	}
	if rungs[0].Retries429 != 1 {
		t.Fatalf("rung 0 retries_429=%d, want 1", rungs[0].Retries429)
	}
	sleeps := clk.Sleeps()
	if len(sleeps) != 1 || sleeps[0] != 2*time.Second {
		t.Fatalf("expected single 2s Retry-After sleep on fakeClock, got %v", sleeps)
	}

	for _, r := range rungs {
		if r.CardsCompleted != 4 || r.Errors != 0 {
			t.Fatalf("unexpected rung %+v", r)
		}
		if r.TotalTokS <= 0 || r.PerLaneTokS <= 0 || r.CardsPerHour <= 0 {
			t.Fatalf("expected positive rates in %+v", r)
		}
		// 1_000_000 + 4 * 250_000 = 2_000_000
		if r.PeakBudgetEstimateBytes != 2_000_000 || r.PeakMemoryBytes != 0 {
			t.Fatalf("budget-only health promoted to measured memory or estimated incorrectly: %+v", r)
		}
		if r.TTFTSamples != 7 {
			t.Fatalf("expected only the seven streaming turns to contribute to TTFT: %+v", r)
		}
	}

	// Verify multi-turn cards executed turns in strictly increasing message-count order.
	mu.Lock()
	for cid, lens := range turnSeqByCard {
		for i := 1; i < len(lens); i++ {
			if lens[i] <= lens[i-1] {
				t.Fatalf("card %s turns executed out of order: message counts %v", cid, lens)
			}
		}
	}
	mu.Unlock()

	md := FormatMarkdown(rungs, knee)
	if strings.Contains(md, "secret-replay-token") {
		t.Fatal("API key must never appear in Markdown output")
	}
	if !strings.Contains(md, "Recommended default lane count:") {
		t.Fatalf("missing recommendation in markdown:\n%s", md)
	}
	tsv := FormatTSV(rungs)
	if !strings.HasPrefix(tsv, "lanes\ttotal_tok_s\t") {
		t.Fatalf("unexpected TSV header:\n%s", tsv)
	}
}

func TestRunCLI_MarkdownAndTSVOutputs(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			fmt.Fprint(w, `{"ok":true,"active_lanes":2,"inflight":2,"peak_memory_bytes":4096}`)
			return
		}
		var body struct {
			Stream bool `json:"stream"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":8}}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"x"},"finish_reason":"stop"}],"usage":{"completion_tokens":8}}`)
	}))
	defer srv.Close()

	dir := localTempDir(t)
	outMD := filepath.Join(dir, "summary.md")
	outTSV := filepath.Join(dir, "summary.tsv")
	var stdout, stderr bytes.Buffer

	code := runCLI(
		[]string{
			"--url", srv.URL,
			"--input", filepath.Join("testdata", "sample_cards.jsonl"),
			"--lanes", "2,4",
			"--cards", "4",
			"--out-md", outMD,
			"--out-tsv", outTSV,
		},
		func(string) string { return "" },
		&stdout,
		&stderr,
	)
	if code != 0 {
		t.Fatalf("runCLI exited %d, stderr=%s", code, stderr.String())
	}
	mdBytes, err := os.ReadFile(outMD)
	if err != nil {
		t.Fatal(err)
	}
	tsvBytes, err := os.ReadFile(outTSV)
	if err != nil {
		t.Fatal(err)
	}
	if string(mdBytes) != stdout.String() {
		t.Fatalf("written markdown does not match stdout")
	}
	if !strings.Contains(string(tsvBytes), "\n2\t") || !strings.Contains(string(tsvBytes), "\n4\t") {
		t.Fatalf("missing lane rows in TSV:\n%s", tsvBytes)
	}
}

func TestReplayMissingIDsAndLegacyRestarts(t *testing.T) {
	dir := localTempDir(t)
	for i, name := range []string{"cards.jsonl", "cards.jsonl.1"} {
		data := fmt.Sprintf(`{"request":{}, "recorded_at_ms":%d}
{"card_id":"card-1", "turn_index":0, "request":{}, "recorded_at_ms":%d}
`, i+1, i+1)
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	cards, err := LoadReplaySet(dir)
	if err != nil || len(cards) != 4 {
		t.Fatalf("cards = %+v, err = %v", cards, err)
	}
	for _, card := range cards {
		if len(card.Turns) != 1 {
			t.Fatalf("unrelated turns were grouped as one card: %+v", card)
		}
	}
}

func TestKneeCannotCrossFailedRung(t *testing.T) {
	rungs := []RungMetrics{
		{Lanes: 32, CardsCompleted: 32, TotalTokS: 300},
		{Lanes: 8, CardsCompleted: 8, TotalTokS: 100},
		{Lanes: 16, Errors: 16},
	}
	res, err := PickKnee(rungs, DefaultKneeConfig())
	if err != nil || res.RecommendedLanes != 8 || !strings.Contains(res.Reason, "16 lanes") {
		t.Fatalf("failed rung was skipped: %+v, %v", res, err)
	}
	for _, cfg := range []KneeConfig{{MinPerLaneTokS: 1}, {MaxTTFTP95Ms: 1000}} {
		if res, err := PickKnee(rungs, cfg); err == nil || res.RecommendedLanes != 0 {
			t.Fatalf("no rung satisfies the limits: %+v, %v", res, err)
		}
	}
}

func TestNegativeDurationRejectedBeforeTraffic(t *testing.T) {
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(500)
	}))
	defer srv.Close()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	cfg := SweepConfig{BaseURL: srv.URL, Duration: -time.Second, CardsPerRung: 1, LaneCounts: []int{1}}
	if _, _, err := RunSweep(ctx, cfg, []Card{{ID: "sample"}}); err == nil || !strings.Contains(err.Error(), "duration") || calls.Load() != 0 {
		t.Fatalf("err = %v, requests = %d", err, calls.Load())
	}
	var out, stderr bytes.Buffer
	code := runCLI([]string{"--input", "testdata/sample_cards.jsonl", "--duration=-1s", "--url", srv.URL}, os.Getenv, &out, &stderr)
	if code != 2 || !strings.Contains(stderr.String(), "--duration") || calls.Load() != 0 {
		t.Fatalf("exit = %d, requests = %d, stderr = %s", code, calls.Load(), stderr.String())
	}
}

func TestMemorySamplePeaksAndUnavailableOutput(t *testing.T) {
	var samples memorySamples
	samples.observe(healthTelemetry{ProcessFootprintBytes: 360, ActiveLanes: 1, ModelBytes: 100, PerLaneWorstCaseBytes: 100})
	samples.observe(healthTelemetry{ProcessFootprintBytes: 220, ActiveLanes: 4, ModelBytes: 100, PerLaneWorstCaseBytes: 100})
	if samples.observed != 360 || samples.budget != 500 {
		t.Fatalf("unexpected independent peaks: %+v", samples)
	}
	rungs := []RungMetrics{{Lanes: 1, PeakBudgetEstimateBytes: 500, CardsCompleted: 1}}
	tsv := strings.Split(strings.TrimSpace(FormatTSV(rungs)), "\n")
	head, row := strings.Split(tsv[0], "\t"), strings.Split(tsv[1], "\t")
	values := map[string]string{}
	for i, name := range head {
		values[name] = row[i]
	}
	for field, want := range map[string]string{"ttft_samples": "0", "ttft_p95_ms": "n/a", "ttft_p50_ms": "n/a", "peak_memory_bytes": "n/a", "peak_budget_estimate_bytes": "500"} {
		if values[field] != want {
			t.Fatalf("%s = %q, want %q", field, values[field], want)
		}
	}
}

func TestCLIReportsRungWhenRequestedTTFTIsUnavailable(t *testing.T) {
	dir := localTempDir(t)
	input := filepath.Join(dir, "recording.jsonl")
	if err := os.WriteFile(input, []byte(`{"request":{"stream":false},"card_id":"buffered"}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			fmt.Fprint(w, `{"ok":true,"process_footprint_bytes":321}`)
			return
		}
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}`)
	}))
	defer srv.Close()
	var out, stderr bytes.Buffer
	tsvPath := filepath.Join(dir, "summary.tsv")
	code := runCLI([]string{"--input", input, "--url", srv.URL, "--lanes", "1", "--max-ttft-p95-ms", "1000", "--out-tsv", tsvPath}, os.Getenv, &out, &stderr)
	if code != 1 || !strings.Contains(stderr.String(), "no streaming TTFT samples") || strings.Contains(out.String(), "Recommended default") {
		t.Fatalf("exit=%d, stderr=%s, report=%s", code, stderr.String(), out.String())
	}
	tsv, err := os.ReadFile(tsvPath)
	if err != nil || !strings.Contains(string(tsv), "\tn/a\tn/a\t321\t1\t") {
		t.Fatalf("rung measurements missing: %q, %v", tsv, err)
	}
}

func TestLoadReplaySetReadsLargeRecorderRowAndFollowingTurn(t *testing.T) {
	path := filepath.Join(localTempDir(t), "cards.jsonl")
	first := `{"card_id":"large","turn_index":0,"request":{"metadata":"` + strings.Repeat("x", 16<<20) + `"}}`
	second := `{"card_id":"large","turn_index":1,"request":{"stream":true}}`
	if err := os.WriteFile(path, []byte(first+"\n"+second), 0o600); err != nil {
		t.Fatal(err)
	}
	cards, err := LoadReplaySet(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(cards) != 1 || len(cards[0].Turns) != 2 {
		t.Fatalf("expected one two-turn card, got %d cards", len(cards))
	}
	turns := cards[0].Turns
	if len(turns[0].Request) != len(`{"metadata":""}`)+(16<<20) || string(turns[1].Request) != `{"stream":true}` {
		t.Fatal("the loader lost part of the large request or its following turn")
	}
}
