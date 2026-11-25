package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func memResponse(status int, contentType, body string, headers map[string]string) *http.Response {
	hdr := make(http.Header)
	if contentType != "" {
		hdr.Set("Content-Type", contentType)
	}
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
	if err := os.WriteFile(filepath.Join(dir, "cards-20261123.jsonl"), []byte(lineA0+"\n"), 0o600); err != nil {
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
			{Lanes: 8, TotalTokS: 320, PerLaneTokS: 40, TTFTP95Ms: 50, CardsCompleted: 8, CardsPerHour: 100},
			{Lanes: 16, TotalTokS: 600, PerLaneTokS: 37.5, TTFTP95Ms: 90, CardsCompleted: 16, CardsPerHour: 190},
			{Lanes: 32, TotalTokS: 1000, PerLaneTokS: 31.25, TTFTP95Ms: 250, CardsCompleted: 32, CardsPerHour: 320},
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

func TestRunSweep_InMemoryMultiTurnAndHealthAnd429Retry(t *testing.T) {
	cards, err := LoadReplaySet(filepath.Join("testdata", "sample_cards.jsonl"))
	if err != nil {
		t.Fatal(err)
	}

	var mu sync.Mutex
	turnSeqByCard := make(map[string][]int)
	var reqCount atomic.Int64
	var healthPolls atomic.Int64
	var sawAuth atomic.Bool

	transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.Path == "/health" {
			healthPolls.Add(1)
			return memResponse(
				http.StatusOK,
				"application/json",
				`{"ok":true,"active_lanes":4,"inflight":5,"model_bytes":1000000,"per_lane_worst_case_bytes":250000}`,
				nil,
			), nil
		}
		if r.Header.Get("Authorization") == "Bearer secret-replay-token" {
			sawAuth.Store(true)
		}
		n := reqCount.Add(1)
		// Inject deterministic 429 + Retry-After on the 2nd request.
		if n == 2 {
			return memResponse(
				http.StatusTooManyRequests,
				"application/json",
				`{"error":{"message":"busy"}}`,
				map[string]string{"Retry-After": "2"},
			), nil
		}

		var body struct {
			Stream   bool              `json:"stream"`
			Messages []json.RawMessage `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			return memResponse(http.StatusBadRequest, "application/json", `{"error":{"message":"bad json"}}`, nil), nil
		}
		cardID := r.Header.Get("X-Inky-Card-Id")
		mu.Lock()
		turnSeqByCard[cardID] = append(turnSeqByCard[cardID], len(body.Messages))
		mu.Unlock()

		if body.Stream {
			sse := "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n" +
				"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":16}}\n\n" +
				"data: [DONE]\n\n"
			return memResponse(http.StatusOK, "text/event-stream", sse, nil), nil
		}
		return memResponse(
			http.StatusOK,
			"application/json",
			`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":12}}`,
			nil,
		), nil
	})

	clk := newFakeClock(time.Date(2026, 11, 24, 18, 0, 0, 0, time.UTC))
	cfg := SweepConfig{
		BaseURL:        "http://inky.invalid:8090",
		APIKey:         "secret-replay-token",
		LaneCounts:     []int{2, 4},
		CardsPerRung:   4,
		MaxRetries:     4,
		HealthInterval: 50 * time.Millisecond,
		Knee:           DefaultKneeConfig(),
		Client:         &http.Client{Transport: transport},
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
	if healthPolls.Load() < 4 {
		t.Fatalf("expected initial, periodic, and final /health polls without real tickers, got %d", healthPolls.Load())
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
		if r.PeakMemoryBytes != 2_000_000 {
			t.Fatalf("peak_memory_bytes=%d, want 2000000", r.PeakMemoryBytes)
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
	transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.Path == "/health" {
			return memResponse(http.StatusOK, "application/json", `{"ok":true,"active_lanes":2,"inflight":2,"peak_memory_bytes":4096}`, nil), nil
		}
		var body struct {
			Stream bool `json:"stream"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body.Stream {
			sse := "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":8}}\n\ndata: [DONE]\n\n"
			return memResponse(http.StatusOK, "text/event-stream", sse, nil), nil
		}
		return memResponse(http.StatusOK, "application/json", `{"usage":{"completion_tokens":8}}`, nil), nil
	})

	dir := localTempDir(t)
	outMD := filepath.Join(dir, "summary.md")
	outTSV := filepath.Join(dir, "summary.tsv")
	var stdout, stderr bytes.Buffer
	clk := newFakeClock(time.Date(2026, 11, 24, 18, 0, 0, 0, time.UTC))

	code := runCLIWithDeps(
		[]string{
			"--url", "http://inky.invalid:8090",
			"--input", filepath.Join("testdata", "sample_cards.jsonl"),
			"--lanes", "2,4",
			"--cards", "4",
			"--out-md", outMD,
			"--out-tsv", outTSV,
		},
		func(string) string { return "" },
		&http.Client{Transport: transport},
		clk,
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
