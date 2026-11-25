package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestValidateStream(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		ok         bool
	}{
		{"complete", "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", true},
		{"unterminated", "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n", false},
		{"truncated", "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\n", false},
		{"no finish", "data: {\"choices\":[{\"finish_reason\":null}]}\n\ndata: [DONE]\n\n", false},
		{"error", "data: {\"error\":{\"message\":\"failed\"}}\n\ndata: [DONE]\n\n", false},
		{"bad JSON", "data: nope\n\ndata: [DONE]\n\n", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := validateStream(strings.NewReader(tc.body)) == nil; got != tc.ok {
				t.Fatalf("valid = %v, want %v", got, tc.ok)
			}
		})
	}
}

func TestStatusAccounting(t *testing.T) {
	var tot totals
	tot.note(400)
	tot.note(429)
	tot.note(302)
	if tot.unexpected.Load() != 2 || tot.http429.Load() != 1 {
		t.Fatalf("unexpected=%d 429=%d", tot.unexpected.Load(), tot.http429.Load())
	}
}

func TestParseFlexibleDurationAndEpoch(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want time.Duration
		ok   bool
	}{
		{"60", 60 * time.Second, true},
		{"3600", 3600 * time.Second, true},
		{"60s", 60 * time.Second, true},
		{"72h", 72 * time.Hour, true},
		{"1h30m", 90 * time.Minute, true},
		{"0", 0, false},
		{"500ms", 0, false},
		{"", 0, false},
		{"bad", 0, false},
	} {
		got, err := parseFlexibleDuration(tc.in)
		if (err == nil) != tc.ok || (tc.ok && got != tc.want) {
			t.Fatalf("parseFlexibleDuration(%q) = (%v, %v), want (%v, ok=%v)", tc.in, got, err, tc.want, tc.ok)
		}
	}

	if got := chooseEpochDuration(2*time.Second, 0); got != 2*time.Second {
		t.Fatalf("short epoch = %v, want 2s", got)
	}
	if got := chooseEpochDuration(60*time.Second, 0); got != 10*time.Second {
		t.Fatalf("60s epoch = %v, want 10s", got)
	}
	if got := chooseEpochDuration(72*time.Hour, 0); got != 5*time.Minute {
		t.Fatalf("72h epoch = %v, want 5m", got)
	}
}

func makeEpoch(idx int, active, inflight, queued int, rss, recBytes, spillBytes, modelBytes int64) epochSample {
	wset := modelBytes * 4
	return epochSample{
		Epoch:          idx,
		ElapsedS:       float64(idx * 10),
		HealthOK:       true,
		ServerRSSBytes: rss,
		RecordDirBytes: recBytes,
		SpillDirBytes:  spillBytes,
		Health: healthSnapshot{
			OK:                         true,
			ActiveLanes:                &active,
			Inflight:                   &inflight,
			QueuedRequests:             &queued,
			ModelBytes:                 &modelBytes,
			RecommendedWorkingSetBytes: &wset,
		},
	}
}

func TestDriftDetection(t *testing.T) {
	const mib = 1024 * 1024
	const model = 16000 * mib
	cfg := driftConfig{
		WarmupEpochs:        1,
		MaxRSSDriftPct:      10.0,
		MaxRSSDriftBytes:    64 * mib,
		RecordMaxTotalBytes: 20 * mib,
		SpillMaxTotalBytes:  10 * mib,
	}

	t.Run("warmup spike then flat steady state passes", func(t *testing.T) {
		samples := []epochSample{
			makeEpoch(0, 0, 0, 0, 100*mib, 5*mib, 0, model), // warmup
			makeEpoch(1, 0, 0, 0, 250*mib, 19*mib, 0, model),
			makeEpoch(2, 0, 0, 0, 254*mib, 12*mib, 0, model),
			makeEpoch(3, 0, 0, 0, 249*mib, 18*mib, 0, model),
			makeEpoch(4, 0, 0, 0, 252*mib, 15*mib, 0, model),
		}
		rep := analyzeDrift(samples, cfg)
		if !rep.OK {
			t.Fatalf("expected steady flat soak to pass, got %+v", rep)
		}
	})

	t.Run("steady-state upward RSS leak fails", func(t *testing.T) {
		samples := []epochSample{
			makeEpoch(0, 0, 0, 0, 200*mib, 5*mib, 0, model),
			makeEpoch(1, 0, 0, 0, 210*mib, 6*mib, 0, model),
			makeEpoch(2, 0, 0, 0, 260*mib, 7*mib, 0, model),
			makeEpoch(3, 0, 0, 0, 310*mib, 8*mib, 0, model),
		}
		rep := analyzeDrift(samples, cfg)
		if rep.OK {
			t.Fatalf("expected upward RSS drift to fail, got %+v", rep)
		}
	})

	t.Run("monotonic RSS creep above half floor fails", func(t *testing.T) {
		samples := []epochSample{
			makeEpoch(0, 0, 0, 0, 200*mib, 5*mib, 0, model),
			makeEpoch(1, 0, 0, 0, 210*mib, 5*mib, 0, model),
			makeEpoch(2, 0, 0, 0, 222*mib, 5*mib, 0, model),
			makeEpoch(3, 0, 0, 0, 235*mib, 5*mib, 0, model),
			makeEpoch(4, 0, 0, 0, 250*mib, 5*mib, 0, model),
		}
		rep := analyzeDrift(samples, cfg)
		if rep.OK {
			t.Fatalf("expected monotonic RSS creep to fail, got %+v", rep)
		}
	})

	t.Run("record-dir exceeding bounded rotation cap fails", func(t *testing.T) {
		samples := []epochSample{
			makeEpoch(0, 0, 0, 0, 200*mib, 10*mib, 0, model),
			makeEpoch(1, 0, 0, 0, 200*mib, 21*mib, 0, model),
		}
		rep := analyzeDrift(samples, cfg)
		if rep.OK {
			t.Fatalf("expected oversized record-dir to fail, got %+v", rep)
		}
	})

	t.Run("monotonic spill-dir growth fails", func(t *testing.T) {
		samples := []epochSample{
			makeEpoch(0, 0, 0, 0, 200*mib, 5*mib, 1*mib, model),
			makeEpoch(1, 0, 0, 0, 200*mib, 5*mib, 2*mib, model),
			makeEpoch(2, 0, 0, 0, 200*mib, 5*mib, 3*mib, model),
			makeEpoch(3, 0, 0, 0, 200*mib, 5*mib, 4*mib, model),
		}
		rep := analyzeDrift(samples, cfg)
		if rep.OK {
			t.Fatalf("expected monotonic spill growth to fail, got %+v", rep)
		}
	})

	t.Run("leaked queued request fails", func(t *testing.T) {
		samples := []epochSample{
			makeEpoch(0, 0, 0, 0, 200*mib, 5*mib, 0, model),
			makeEpoch(1, 0, 0, 1, 200*mib, 5*mib, 0, model),
		}
		rep := analyzeDrift(samples, cfg)
		if rep.OK {
			t.Fatalf("expected leaked queued request to fail, got %+v", rep)
		}
	})

	t.Run("changed model_bytes fails", func(t *testing.T) {
		samples := []epochSample{
			makeEpoch(0, 0, 0, 0, 200*mib, 5*mib, 0, model),
			makeEpoch(1, 0, 0, 0, 200*mib, 5*mib, 0, model+1),
		}
		rep := analyzeDrift(samples, cfg)
		if rep.OK {
			t.Fatalf("expected model_bytes mutation to fail, got %+v", rep)
		}
	})

	t.Run("RSS leak net of model_bytes fails even when small fraction of model", func(t *testing.T) {
		// Total RSS is ~16.2 GiB (dominated by 16 GiB model), so +100 MiB is < 1% of total RSS
		// (would have passed 10% of 16.2 GiB = 1.62 GiB!) but is 50% of 200 MiB non-model RSS (> 64 MiB cap).
		samples := []epochSample{
			makeEpoch(0, 0, 0, 0, model+180*mib, 5*mib, 0, model),
			makeEpoch(1, 0, 0, 0, model+200*mib, 5*mib, 0, model),
			makeEpoch(2, 0, 0, 0, model+250*mib, 5*mib, 0, model),
			makeEpoch(3, 0, 0, 0, model+300*mib, 5*mib, 0, model),
		}
		rep := analyzeDrift(samples, cfg)
		if rep.OK {
			t.Fatalf("expected +100 MiB non-model RSS leak to fail net of 16 GiB model, got %+v", rep)
		}
		if rep.RSSNetSteadyFirstBytes != 200*mib {
			t.Fatalf("net_first=%d, want %d", rep.RSSNetSteadyFirstBytes, 200*mib)
		}
	})

	t.Run("upward RSS creep with a flat epoch still fails", func(t *testing.T) {
		samples := []epochSample{
			makeEpoch(0, 0, 0, 0, model+200*mib, 5*mib, 0, model),
			makeEpoch(1, 0, 0, 0, model+210*mib, 5*mib, 0, model),
			makeEpoch(2, 0, 0, 0, model+226*mib, 5*mib, 0, model),
			makeEpoch(3, 0, 0, 0, model+226*mib, 5*mib, 0, model), // flat epoch must not reset creep guard!
			makeEpoch(4, 0, 0, 0, model+242*mib, 5*mib, 0, model),
			makeEpoch(5, 0, 0, 0, model+256*mib, 5*mib, 0, model),
		}
		rep := analyzeDrift(samples, cfg)
		if rep.OK {
			t.Fatalf("expected upward RSS creep with flat epoch to fail, got %+v", rep)
		}
	})

	t.Run("upward RSS linear trend with minor 1 MiB dip still fails", func(t *testing.T) {
		samples := []epochSample{
			makeEpoch(0, 0, 0, 0, model+200*mib, 5*mib, 0, model),
			makeEpoch(1, 0, 0, 0, model+210*mib, 5*mib, 0, model),
			makeEpoch(2, 0, 0, 0, model+228*mib, 5*mib, 0, model),
			makeEpoch(3, 0, 0, 0, model+227*mib, 5*mib, 0, model),
			makeEpoch(4, 0, 0, 0, model+244*mib, 5*mib, 0, model),
			makeEpoch(5, 0, 0, 0, model+258*mib, 5*mib, 0, model),
		}
		rep := analyzeDrift(samples, cfg)
		if rep.OK {
			t.Fatalf("expected sustained upward linear trend to fail, got %+v", rep)
		}
	})
}

func TestUnderCapacity429Verdict(t *testing.T) {
	var reqNum atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"ok":true,"active_lanes":0,"inflight":0,"queued_requests":0}`)
			return
		}
		var req chatReq
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.MaxTokens == 8 {
			fmt.Fprint(w, "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
			return
		}
		if reqNum.Add(1) == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			fmt.Fprint(w, `{"error":{"message":"busy"}}`)
			return
		}
		fmt.Fprint(w, "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()

	baseCfg := clientConfig{
		chatURL:          server.URL + "/v1/chat/completions",
		key:              "test",
		duration:         1 * time.Millisecond,
		epoch:            1 * time.Millisecond,
		minEpochs:        1,
		warmupEpochs:     0,
		maxTokensChoices: []int{16},
		clients:          2,
		lanes:            2,
		queue:            4,
		seed:             1,
		maxRSSDriftPct:   15.0,
		maxRSSDriftBytes: 64 << 20,
		noSleep:          true,
	}

	// Under capacity (clients=2 <= lanes+queue=6) without -allow-429: single 429 MUST fail soak verdict (U36).
	reqNum.Store(0)
	if got := runSoak(newSoakHTTPClient(), baseCfg, func(int) (int64, error) { return 0, nil }); got != 1 {
		t.Fatalf("expected 429 under capacity to fail soak verdict, got %d", got)
	}

	// With explicit allow429=true: passes when workload completes.
	reqNum.Store(0)
	allowCfg := baseCfg
	allowCfg.allow429 = true
	if got := runSoak(newSoakHTTPClient(), allowCfg, func(int) (int64, error) { return 0, nil }); got != 0 {
		t.Fatalf("expected allow429=true soak to pass, got %d", got)
	}

	// Over capacity (clients=8 > lanes+queue=6): 429 is expected admission shedding and passes.
	reqNum.Store(0)
	overCfg := baseCfg
	overCfg.clients = 8
	if got := runSoak(newSoakHTTPClient(), overCfg, func(int) (int64, error) { return 0, nil }); got != 0 {
		t.Fatalf("expected over-capacity 429 shedding to pass, got %d", got)
	}
}

func TestSoakVerdict(t *testing.T) {
	if os.Getenv("SOAK_TEST_CHILD") == "1" {
		os.Args = []string{"soak", "-url", os.Getenv("SOAK_TEST_URL"), "-duration", "1", "-clients", "1", "-lanes", "1", "-disconnect-pct", "0"}
		main()
		return
	}
	for _, tc := range []struct {
		name           string
		workloadStatus int
		body           string
		healthBody     string
		success        bool
	}{
		{"success", 200, "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", `{"ok":true,"active_lanes":0,"inflight":0}`, true},
		{"leaked server lane fails", 200, "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", `{"ok":true,"active_lanes":1,"inflight":0}`, false},
		{"leaked server inflight fails", 200, "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", `{"ok":true,"active_lanes":0,"inflight":1}`, false},
		{"all workload rejected", 400, "bad request", `{"ok":true,"active_lanes":0,"inflight":0}`, false},
		{"all 429 zero workload fails", 429, "overloaded", `{"ok":true,"active_lanes":0,"inflight":0}`, false},
		{"truncated stream", 200, "data: [DONE]\n\n", `{"ok":true,"active_lanes":0,"inflight":0}`, false},
		{"mid-stream error", 200, "data: {\"error\":{\"message\":\"boom\"}}\n\ndata: [DONE]\n\n", `{"ok":true,"active_lanes":0,"inflight":0}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var count atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/health" {
					w.Header().Set("Content-Type", "application/json")
					fmt.Fprint(w, tc.healthBody)
					return
				}
				count.Add(1)
				var req chatReq
				_ = json.NewDecoder(r.Body).Decode(&req)
				if (tc.name == "all workload rejected" || tc.name == "all 429 zero workload fails") && req.MaxTokens == 8 {
					fmt.Fprint(w, "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
					return
				}
				w.WriteHeader(tc.workloadStatus)
				fmt.Fprint(w, tc.body)
			}))
			defer server.Close()
			cmd := exec.Command(os.Args[0], "-test.run=^TestSoakVerdict$")
			cmd.Env = append(os.Environ(), "SOAK_TEST_CHILD=1", "SOAK_TEST_URL="+server.URL+"/v1/chat/completions", "INKY_SERVE_KEY=test")
			out, err := cmd.CombinedOutput()
			if (err == nil) != tc.success {
				t.Fatalf("err=%v output=%s", err, out)
			}
			if count.Load() < 2 {
				t.Fatalf("missing workload/recovery requests")
			}
		})
	}
}

func TestDisconnectStormVerdict(t *testing.T) {
	for _, tc := range []struct {
		name              string
		corruptAfter      bool
		leakHealth        bool
		ignoreEarlyCancel bool
		preTTFTRuns       bool
		wantCode          int
	}{
		{"clean storm drains and stops early aborts", false, false, false, false, 0},
		{"corrupted post-storm reply fails", true, false, false, false, 1},
		{"leaked health after storm fails", false, true, false, false, 1},
		{"ttft early abort running to max_tokens fails", false, false, true, false, 1},
		{"header-done partial request generating tokens fails", false, false, false, true, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var reqCount atomic.Int64
			var outTokens atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/health" {
					w.Header().Set("Content-Type", "application/json")
					if tc.leakHealth && reqCount.Load() > 20 {
						fmt.Fprintf(w, `{"ok":true,"active_lanes":1,"inflight":1,"queued_requests":0,"output_tokens":%d}`, outTokens.Load())
					} else {
						fmt.Fprintf(w, `{"ok":true,"active_lanes":0,"inflight":0,"queued_requests":0,"output_tokens":%d}`, outTokens.Load())
					}
					return
				}
				n := reqCount.Add(1)
				bodyBytes, readErr := io.ReadAll(r.Body)
				if readErr != nil || len(bodyBytes) == 0 {
					if tc.preTTFTRuns {
						outTokens.Add(256)
					}
					return
				}
				var req chatReq
				if json.Unmarshal(bodyBytes, &req) != nil {
					if tc.preTTFTRuns {
						outTokens.Add(256)
					}
					return
				}
				if !req.Stream {
					content := "391"
					if tc.corruptAfter && n > 10 {
						content = "392"
					}
					outTokens.Add(3)
					w.Header().Set("Content-Type", "application/json")
					fmt.Fprintf(w, `{"choices":[{"finish_reason":"stop","message":{"content":%q}}],"usage":{"prompt_tokens":12,"completion_tokens":3}}`, content)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				flusher, _ := w.(http.Flusher)
				if _, err := fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think...\"},\"finish_reason\":null}]}\n\n"); err != nil {
					if tc.ignoreEarlyCancel {
						outTokens.Add(int64(req.MaxTokens))
					}
					return
				}
				if flusher != nil {
					flusher.Flush()
				}
				outTokens.Add(1)
				if tc.ignoreEarlyCancel && req.MaxTokens == 256 {
					outTokens.Add(int64(req.MaxTokens - 1))
					return
				}
				if r.Context().Err() != nil {
					return
				}
				if _, err := fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"); err == nil {
					outTokens.Add(1)
					if flusher != nil {
						flusher.Flush()
					}
				}
			}))
			defer server.Close()

			cfg := clientConfig{
				chatURL:       server.URL + "/v1/chat/completions",
				key:           "test",
				clients:       8,
				lanes:         2,
				queue:         64,
				seed:          42,
				stormRequests: 64,
				noSleep:       true,
				drainTimeout:  50 * time.Millisecond,
			}
			code := runDisconnectStorm(newSoakHTTPClient(), cfg)
			if code != tc.wantCode {
				t.Fatalf("runDisconnectStorm exit=%d, want %d", code, tc.wantCode)
			}
		})
	}
}

func TestFaultCheckAndReferenceCheck(t *testing.T) {
	var writeErrors atomic.Int64
	var replyValue atomic.Int64
	replyValue.Store(391)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `{"ok":true,"active_lanes":0,"inflight":0,"queued_requests":0,"record_written":0,"record_write_errors":%d}`, writeErrors.Load())
			return
		}
		var req chatReq
		_ = json.NewDecoder(r.Body).Decode(&req)
		if !req.Stream {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `{"choices":[{"finish_reason":"stop","message":{"content":"%d"}}],"usage":{"prompt_tokens":10,"completion_tokens":2}}`, replyValue.Load())
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()

	cfg := clientConfig{
		chatURL:               server.URL + "/v1/chat/completions",
		key:                   "test",
		duration:              1 * time.Second,
		clients:               2,
		lanes:                 2,
		seed:                  7,
		expectRecordWriteErrs: true,
	}
	if got := runFaultCheck(newSoakHTTPClient(), cfg); got != 1 {
		t.Fatalf("expected faultCheck without write errors to fail, got %d", got)
	}
	writeErrors.Store(4)
	if got := runFaultCheck(newSoakHTTPClient(), cfg); got != 0 {
		t.Fatalf("expected faultCheck with write errors to pass, got %d", got)
	}

	refPath := filepath.Join(t.TempDir(), "ref.json")
	refCfg := clientConfig{
		chatURL:       server.URL + "/v1/chat/completions",
		key:           "test",
		lanes:         2,
		seed:          7,
		referenceFile: refPath,
	}
	if got := runReferenceCheck(newSoakHTTPClient(), refCfg); got != 0 {
		t.Fatalf("initial reference save failed: %d", got)
	}
	if got := runReferenceCheck(newSoakHTTPClient(), refCfg); got != 0 {
		t.Fatalf("matching post-restart reference check failed: %d", got)
	}
	replyValue.Store(999)
	if got := runReferenceCheck(newSoakHTTPClient(), refCfg); got != 1 {
		t.Fatalf("mismatched post-restart reference check should fail, got %d", got)
	}
}
