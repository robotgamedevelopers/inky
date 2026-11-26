package bestofn

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

func writeSSE(w http.ResponseWriter, flusher http.Flusher, content string, promptToks, cachedToks, compToks int) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	deltaChunk := map[string]any{
		"choices": []map[string]any{
			{"index": 0, "delta": map[string]any{"content": content}},
		},
	}
	b, _ := json.Marshal(deltaChunk)
	fmt.Fprintf(w, "data: %s\n\n", b)
	flusher.Flush()

	usageChunk := map[string]any{
		"choices": []map[string]any{},
		"usage": map[string]any{
			"prompt_tokens":     promptToks,
			"completion_tokens": compToks,
			"total_tokens":      promptToks + compToks,
			"prompt_tokens_details": map[string]any{
				"cached_tokens": cachedToks,
			},
		},
	}
	b, _ = json.Marshal(usageChunk)
	fmt.Fprintf(w, "data: %s\n\n", b)
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func TestBestOfN_FirstPassingCancelsRemainingLanes(t *testing.T) {
	var (
		mu              sync.Mutex
		seenPrompts     []string
		seenSeeds       []uint64
		seenTemps       []float64
		cancelledOnSrv  atomic.Int32
		activeBeforeWin atomic.Int32
	)
	allArrived := make(chan struct{})
	bothSlowReady := make(chan struct{})
	slowHandlerDone := make(chan struct{}, 2)
	var arrivedCount atomic.Int32

	fifoPath := filepath.Join(t.TempDir(), "lane0.fifo")
	if err := syscall.Mkfifo(fifoPath, 0o600); err != nil {
		t.Fatalf("mkfifo: %v", err)
	}
	fifoFile, err := os.OpenFile(fifoPath, os.O_RDWR, 0o600)
	if err != nil {
		t.Fatalf("open fifo: %v", err)
	}
	defer fifoFile.Close()
	lane0Checked := make(chan struct{})
	go func() {
		var b [1]byte
		_, _ = fifoFile.Read(b[:])
		close(lane0Checked)
	}()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		mu.Lock()
		rawMsgs, _ := json.Marshal(req.Messages)
		seenPrompts = append(seenPrompts, string(rawMsgs))
		seenSeeds = append(seenSeeds, req.Seed)
		seenTemps = append(seenTemps, req.Temperature)
		mu.Unlock()

		if arrivedCount.Add(1) == 4 {
			close(allArrived)
		}
		select {
		case <-allArrived:
		case <-r.Context().Done():
			cancelledOnSrv.Add(1)
			return
		}

		flusher := w.(http.Flusher)
		switch req.Seed {
		case 1001:
			// Lane 0: fast response, fails deterministic check.
			writeSSE(w, flusher, "```txt\nWRONG\n```", 256, 0, 8)
		case 1003:
			// Lane 2: wait until both slow lanes have flushed their initial chunk and
			// lane 0 has entered its deterministic check, then emit the passing response.
			select {
			case <-bothSlowReady:
			case <-r.Context().Done():
				cancelledOnSrv.Add(1)
				return
			}
			select {
			case <-lane0Checked:
			case <-r.Context().Done():
				cancelledOnSrv.Add(1)
				return
			}
			writeSSE(w, flusher, "```txt\nPASS_TOKEN\n```", 256, 256, 8)
		default:
			// Lanes 1 and 3: slow streams that wait until cancelled by winner.
			defer func() { slowHandlerDone <- struct{}{} }()
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"thinking...\"}}]}\n\n")
			flusher.Flush()
			if activeBeforeWin.Add(1) == 2 {
				close(bothSlowReady)
			}
			select {
			case <-r.Context().Done():
				cancelledOnSrv.Add(1)
			case <-time.After(10 * time.Second):
				writeSSE(w, flusher, "TOO_LATE", 256, 256, 8)
			}
		}
	}))
	defer srv.Close()

	cfg := DefaultConfig()
	cfg.Endpoint = srv.URL
	cfg.N = 4
	cfg.Temperature = 0.6
	cfg.BaseSeed = 1001
	cfg.Timeout = 5 * time.Second
	cfg.SystemPrompt = "You are a concise coding assistant."

	card := Card{
		ID:         "cancel-test",
		Tier:       "flash+",
		Prompt:     "Print PASS_TOKEN into out.txt.",
		TargetFile: "out.txt",
		CheckCmd:   fmt.Sprintf(`if [ "$BESTOFN_SEED" = "1001" ]; then printf x > %q; fi; grep -qx "PASS_TOKEN" out.txt`, fifoPath),
	}

	res, err := RunCard(context.Background(), cfg, card)
	if err != nil {
		t.Fatalf("RunCard error: %v", err)
	}
	if !res.BestOfNPassed {
		t.Fatalf("expected BestOfNPassed=true, got false: %+v", res)
	}
	if res.WinnerIndex != 2 {
		t.Fatalf("expected WinnerIndex=2, got %d", res.WinnerIndex)
	}
	if res.PassAt1 {
		t.Fatalf("expected PassAt1=false since lane 0 failed")
	}
	if res.CancelledLanes != 2 {
		t.Fatalf("expected 2 cancelled lanes, got %d (%+v)", res.CancelledLanes, res.Attempts)
	}
	if res.WallDuration >= 2*time.Second {
		t.Fatalf("expected early return well before 10s slow lanes, got %v", res.WallDuration)
	}
	teardownTimer := time.NewTimer(2 * time.Second)
	defer teardownTimer.Stop()
	for i := 0; i < 2; i++ {
		select {
		case <-slowHandlerDone:
		case <-teardownTimer.C:
			t.Fatalf("timed out waiting for slow server handler %d/2 to observe cancellation", i+1)
		}
	}
	if got := cancelledOnSrv.Load(); got != 2 {
		t.Fatalf("expected server to observe 2 cancelled requests, got %d", got)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(seenPrompts) != 4 {
		t.Fatalf("expected 4 requests, got %d", len(seenPrompts))
	}
	for i := 1; i < len(seenPrompts); i++ {
		if seenPrompts[i] != seenPrompts[0] {
			t.Fatalf("lane %d prompt differs from lane 0: %s vs %s", i, seenPrompts[i], seenPrompts[0])
		}
	}
	seedSet := make(map[uint64]bool)
	for _, s := range seenSeeds {
		seedSet[s] = true
	}
	if len(seedSet) != 4 {
		t.Fatalf("expected 4 distinct seeds, got %v", seenSeeds)
	}
	for _, temp := range seenTemps {
		if temp != 0.6 {
			t.Fatalf("expected temp 0.6, got %v", temp)
		}
	}
	if res.TotalCachedTokens != 256 || res.TotalPromptTokens != 512 {
		t.Fatalf("unexpected token accounting: cached=%d prompt=%d", res.TotalCachedTokens, res.TotalPromptTokens)
	}
	if res.PrefixCacheRate != 0.5 {
		t.Fatalf("expected prefix cache rate 0.5, got %v", res.PrefixCacheRate)
	}
}

func TestBestOfN_AllFailAndServerErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Seed == 2000 {
			http.Error(w, "overloaded", http.StatusTooManyRequests)
			return
		}
		flusher := w.(http.Flusher)
		writeSSE(w, flusher, "```txt\nBAD\n```", 128, 64, 4)
	}))
	defer srv.Close()

	cfg := DefaultConfig()
	cfg.Endpoint = srv.URL
	cfg.N = 4
	cfg.BaseSeed = 2000
	card := Card{
		ID:         "all-fail",
		Prompt:     "Emit GOOD.",
		TargetFile: "out.txt",
		CheckCmd:   `grep -qx "GOOD" out.txt`,
	}

	res, err := RunCard(context.Background(), cfg, card)
	if err != nil {
		t.Fatalf("RunCard error: %v", err)
	}
	if res.BestOfNPassed || res.WinnerIndex != -1 {
		t.Fatalf("expected no winner, got %+v", res)
	}
	if res.Attempts[0].Status != StatusRequestError {
		t.Fatalf("expected lane 0 StatusRequestError, got %s", res.Attempts[0].Status)
	}
	for i := 1; i < 4; i++ {
		if res.Attempts[i].Status != StatusFailedCheck {
			t.Fatalf("expected lane %d StatusFailedCheck, got %s", i, res.Attempts[i].Status)
		}
	}
}

func TestBestOfN_SuitePassAt1VsBestOfNByTier(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		flusher := w.(http.Flusher)
		prompt := req.Messages[len(req.Messages)-1].Content
		cached := 200
		if req.Seed == 3000 {
			cached = 0
		}
		switch prompt {
		case "card-easy":
			// All lanes pass (pass@1 = true, best-of-4 = true)
			writeSSE(w, flusher, "```txt\nOK\n```", 200, cached, 5)
		case "card-medium":
			// Only lane 3 passes (pass@1 = false, best-of-4 = true)
			if req.Seed == 3003 {
				writeSSE(w, flusher, "```txt\nOK\n```", 200, cached, 5)
			} else {
				writeSSE(w, flusher, "```txt\nNO\n```", 200, cached, 5)
			}
		default:
			// None pass
			writeSSE(w, flusher, "```txt\nNO\n```", 200, cached, 5)
		}
	}))
	defer srv.Close()

	cfg := DefaultConfig()
	cfg.Endpoint = srv.URL
	cfg.N = 4
	cfg.BaseSeed = 3000
	cfg.CancelOnFirstPass = false // run all lanes to verify deterministic accounting

	cards := []Card{
		{ID: "c1", Tier: "flash", Prompt: "card-easy", TargetFile: "a.txt", CheckCmd: `grep -qx "OK" a.txt`},
		{ID: "c2", Tier: "flash+", Prompt: "card-medium", TargetFile: "a.txt", CheckCmd: `grep -qx "OK" a.txt`},
		{ID: "c3", Tier: "flash+", Prompt: "card-hard", TargetFile: "a.txt", CheckCmd: `grep -qx "OK" a.txt`},
	}

	rep, err := RunSuite(context.Background(), cfg, cards)
	if err != nil {
		t.Fatalf("RunSuite error: %v", err)
	}
	if rep.PassAt1Count != 1 || rep.BestOfNCount != 2 {
		t.Fatalf("expected pass@1=1 best-of-4=2, got %d / %d", rep.PassAt1Count, rep.BestOfNCount)
	}
	if len(rep.ByTier) != 2 {
		t.Fatalf("expected 2 tiers, got %+v", rep.ByTier)
	}
	if rep.ByTier[0].Tier != "flash" || rep.ByTier[0].PassAt1Rate != 1.0 || rep.ByTier[0].BestOfNRate != 1.0 {
		t.Fatalf("unexpected flash tier stats: %+v", rep.ByTier[0])
	}
	if rep.ByTier[1].Tier != "flash+" || rep.ByTier[1].PassAt1Rate != 0.0 || rep.ByTier[1].BestOfNRate != 0.5 {
		t.Fatalf("unexpected flash+ tier stats: %+v", rep.ByTier[1])
	}
	if rep.PrefixCacheRate != 0.75 {
		t.Fatalf("expected 75%% prefix reuse (3/4 lanes cached), got %.4f", rep.PrefixCacheRate)
	}
	summary := FormatSummary(rep)
	if summary == "" {
		t.Fatal("empty summary")
	}
}

func TestExtractFiles(t *testing.T) {
	multi := "Here is the fix:\n```go:pkg/add.go\npackage pkg\nfunc Add(a,b int) int { return a+b }\n```\n```go file:pkg/add_test.go\npackage pkg\n```"
	files := ExtractFiles(multi, "ignored.go")
	if len(files) != 2 || files["pkg/add.go"] == "" || files["pkg/add_test.go"] == "" {
		t.Fatalf("unexpected multi-file extract: %+v", files)
	}

	single := "Answer:\n```python\nprint(42)\n```\n"
	files = ExtractFiles(single, "solve.py")
	if files["solve.py"] != "print(42)\n" {
		t.Fatalf("unexpected single extract: %q", files["solve.py"])
	}

	raw := ExtractFiles("plain text", "out.txt")
	if raw["out.txt"] != "plain text\n" {
		t.Fatalf("unexpected raw extract: %q", raw["out.txt"])
	}

	unclosed := ExtractFiles("```python\nprint(123)\n", "out.py")
	if unclosed["out.py"] != "print(123)\n" {
		t.Fatalf("unexpected unclosed-fence extract: %q", unclosed["out.py"])
	}
}

func TestBestOfN_MidStreamSSEErrorAndTruncationFailAsRequestError(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{
			name: "mid-stream error chunk",
			body: "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"```txt\\nOK\\n```\"}}]}\n\ndata: {\"error\":{\"message\":\"upstream exploded\",\"type\":\"server_error\"}}\n\ndata: [DONE]\n\n",
		},
		{
			name: "truncated stream missing DONE",
			body: "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"```txt\\nOK\\n```\"}}]}\n\n",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				_, _ = fmt.Fprint(w, tc.body)
			}))
			defer srv.Close()

			cfg := DefaultConfig()
			cfg.Endpoint = srv.URL
			cfg.N = 1
			card := Card{ID: "err-card", Prompt: "test", TargetFile: "out.txt", CheckCmd: `grep -qx "OK" out.txt`}
			res, err := RunCard(context.Background(), cfg, card)
			if err != nil {
				t.Fatalf("RunCard returned unexpected error: %v", err)
			}
			if res.Attempts[0].Status != StatusRequestError {
				t.Fatalf("expected StatusRequestError, got %s (err=%q)", res.Attempts[0].Status, res.Attempts[0].Error)
			}
		})
	}
}
