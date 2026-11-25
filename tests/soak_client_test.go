package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// fakeClock advances virtual time deterministically with zero wall-clock sleeps.
type fakeClock struct {
	base time.Time
	ns   atomic.Int64
}

func newFakeClock() *fakeClock {
	return &fakeClock{base: time.Unix(1_700_000_000, 0)}
}

func (f *fakeClock) Now() time.Time {
	return f.base.Add(time.Duration(f.ns.Load()))
}

func (f *fakeClock) Sleep(d time.Duration) {
	if d > 0 {
		f.ns.Add(int64(d))
	}
}

// memResponseWriter records status, headers, and body in memory with zero TCP sockets.
type memResponseWriter struct {
	header http.Header
	code   int
	body   bytes.Buffer
}

func (w *memResponseWriter) Header() http.Header { return w.header }
func (w *memResponseWriter) WriteHeader(statusCode int) {
	if w.code == 0 {
		w.code = statusCode
	}
}
func (w *memResponseWriter) Write(p []byte) (int, error) {
	if w.code == 0 {
		w.code = http.StatusOK
	}
	return w.body.Write(p)
}
func (w *memResponseWriter) Flush() {}

type abortConn struct {
	closed atomic.Bool
	close  func() error
}

func (c *abortConn) Read([]byte) (int, error)         { return 0, io.EOF }
func (c *abortConn) Write(p []byte) (int, error)      { return len(p), nil }
func (c *abortConn) LocalAddr() net.Addr              { return dummyAddr("mem-local") }
func (c *abortConn) RemoteAddr() net.Addr             { return dummyAddr("mem-remote") }
func (c *abortConn) SetDeadline(time.Time) error      { return nil }
func (c *abortConn) SetReadDeadline(time.Time) error  { return nil }
func (c *abortConn) SetWriteDeadline(time.Time) error { return nil }
func (c *abortConn) Close() error {
	c.closed.Store(true)
	if c.close != nil {
		return c.close()
	}
	return nil
}

type dummyAddr string

func (a dummyAddr) Network() string { return "mem" }
func (a dummyAddr) String() string  { return string(a) }

type abortableBody struct {
	r      *bytes.Reader
	aborted *atomic.Bool
}

func (b *abortableBody) Read(p []byte) (int, error) {
	if b.aborted.Load() {
		return 0, io.ErrClosedPipe
	}
	return b.r.Read(p)
}

func (b *abortableBody) Close() error {
	b.aborted.Store(true)
	return nil
}

// newInMemoryEnv creates an HTTP client and dialer backed purely by in-memory handlers and pipes
// (zero loopback sockets, zero sleeps, zero child processes).
func newInMemoryEnv(handler http.HandlerFunc) (*http.Client, *fakeClock, dialFunc) {
	clk := newFakeClock()
	rt := roundTripFunc(func(req *http.Request) (*http.Response, error) {
		clk.Sleep(50 * time.Millisecond)
		rec := &memResponseWriter{header: make(http.Header)}
		handler(rec, req)
		if rec.code == 0 {
			rec.code = http.StatusOK
		}
		var aborted atomic.Bool
		body := &abortableBody{r: bytes.NewReader(rec.body.Bytes()), aborted: &aborted}
		if box, ok := req.Context().Value(ctxKey{}).(*dialBox); ok {
			box.c.Store(&abortConn{close: body.Close})
		}
		return &http.Response{
			StatusCode: rec.code,
			Status:     fmt.Sprintf("%d %s", rec.code, http.StatusText(rec.code)),
			Header:     rec.header,
			Body:       body,
			Request:    req,
		}, nil
	})
	client := &http.Client{
		Transport: rt,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	dial := func(network, addr string, timeout time.Duration) (net.Conn, error) {
		c1, c2 := net.Pipe()
		go func() {
			_, _ = io.Copy(io.Discard, c2)
			_ = c2.Close()
		}()
		return c1, nil
	}
	return client, clk, dial
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

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
}

func TestSoakVerdict(t *testing.T) {
	start := time.Now()
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
			client, clk, dial := newInMemoryEnv(func(w http.ResponseWriter, r *http.Request) {
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
			})
			var stdout, stderr bytes.Buffer
			args := []string{
				"-url", "http://mem.test:18090/v1/chat/completions",
				"-duration", "1",
				"-clients", "1",
				"-lanes", "1",
				"-disconnect-pct", "0",
				"-max-tokens", "16,32,64",
			}
			getenv := func(k string) string {
				if k == "INKY_SERVE_KEY" {
					return "test"
				}
				return ""
			}
			code := runCLI(args, getenv, client, clientConfig{clock: clk, dial: dial, stdout: &stdout, stderr: &stderr})
			if (code == 0) != tc.success {
				t.Fatalf("code=%d wantSuccess=%v stdout=%s stderr=%s", code, tc.success, stdout.String(), stderr.String())
			}
			if tc.success && !strings.Contains(stdout.String(), "\nmax_tokens=16,32,64\n") {
				t.Fatalf("receipt did not report configured token mix: %s", stdout.String())
			}
			if count.Load() < 2 {
				t.Fatalf("missing workload/recovery requests (got %d)", count.Load())
			}
		})
	}
	if elapsed := time.Since(start); elapsed > 1500*time.Millisecond {
		t.Fatalf("TestSoakVerdict exceeded unit budget without sleeps: %v", elapsed)
	}
}

func TestDisconnectStormVerdict(t *testing.T) {
	start := time.Now()
	for _, tc := range []struct {
		name         string
		corruptAfter bool
		leakHealth   bool
		wantCode     int
	}{
		{"clean storm drains and matches reference", false, false, 0},
		{"corrupted post-storm reply fails", true, false, 1},
		{"leaked health after storm fails", false, true, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var reqCount atomic.Int64
			client, clk, dial := newInMemoryEnv(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/health" {
					w.Header().Set("Content-Type", "application/json")
					if tc.leakHealth && reqCount.Load() > 5 {
						fmt.Fprint(w, `{"ok":true,"active_lanes":1,"inflight":1,"queued_requests":0}`)
					} else {
						fmt.Fprint(w, `{"ok":true,"active_lanes":0,"inflight":0,"queued_requests":0}`)
					}
					return
				}
				n := reqCount.Add(1)
				var req chatReq
				_ = json.NewDecoder(r.Body).Decode(&req)
				if !req.Stream {
					content := "391"
					if tc.corruptAfter && n > 2 {
						content = "392"
					}
					w.Header().Set("Content-Type", "application/json")
					fmt.Fprintf(w, `{"choices":[{"finish_reason":"stop","message":{"content":%q}}],"usage":{"prompt_tokens":12,"completion_tokens":3}}`, content)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think...\"},\"finish_reason\":null}]}\n\n")
				fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
			})

			var stdout, stderr bytes.Buffer
			cfg := clientConfig{
				chatURL:       "http://mem.test:18090/v1/chat/completions",
				key:           "test",
				clients:       8,
				lanes:         2,
				seed:          42,
				stormRequests: 64,
				clock:         clk,
				dial:          dial,
				stdout:        &stdout,
				stderr:        &stderr,
			}
			code := runDisconnectStorm(client, cfg)
			if code != tc.wantCode {
				t.Fatalf("runDisconnectStorm exit=%d, want %d (stderr=%s)", code, tc.wantCode, stderr.String())
			}
		})
	}
	if elapsed := time.Since(start); elapsed > 1500*time.Millisecond {
		t.Fatalf("TestDisconnectStormVerdict exceeded unit budget without sleeps: %v", elapsed)
	}
}

func TestFaultCheckAndReferenceCheck(t *testing.T) {
	var writeErrors atomic.Int64
	var replyValue atomic.Int64
	replyValue.Store(391)
	client, clk, dial := newInMemoryEnv(func(w http.ResponseWriter, r *http.Request) {
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
	})

	var stdout, stderr bytes.Buffer
	cfg := clientConfig{
		chatURL:               "http://mem.test:18090/v1/chat/completions",
		key:                   "test",
		duration:              1 * time.Second,
		clients:               2,
		lanes:                 2,
		seed:                  7,
		expectRecordWriteErrs: true,
		clock:                 clk,
		dial:                  dial,
		stdout:                &stdout,
		stderr:                &stderr,
	}
	if got := runFaultCheck(client, cfg); got != 1 {
		t.Fatalf("expected faultCheck without write errors to fail, got %d", got)
	}
	writeErrors.Store(4)
	if got := runFaultCheck(client, cfg); got != 0 {
		t.Fatalf("expected faultCheck with write errors to pass, got %d", got)
	}

	refPath := filepath.Join(t.TempDir(), "ref.json")
	refCfg := clientConfig{
		chatURL:       "http://mem.test:18090/v1/chat/completions",
		key:           "test",
		lanes:         2,
		seed:          7,
		referenceFile: refPath,
		clock:         clk,
		dial:          dial,
		stdout:        &stdout,
		stderr:        &stderr,
	}
	if got := runReferenceCheck(client, refCfg); got != 0 {
		t.Fatalf("initial reference save failed: %d", got)
	}
	if got := runReferenceCheck(client, refCfg); got != 0 {
		t.Fatalf("matching post-restart reference check failed: %d", got)
	}
	replyValue.Store(999)
	if got := runReferenceCheck(client, refCfg); got != 1 {
		t.Fatalf("mismatched post-restart reference check should fail, got %d", got)
	}
}
