package main

import (
	"bytes"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestLoadProgramWithoutSocketsOrRealTime(t *testing.T) {
	for _, tc := range []struct {
		name     string
		args     []string
		status   int
		body     string
		wantExit int
	}{
		{"successful response without timings", []string{"-url", "http://inky.invalid/v1/chat/completions", "-n", "1"}, 200, `{"usage":{"completion_tokens":10}}`, 0},
		{"server timings preferred when present", []string{"-url", "http://inky.invalid/v1/chat/completions", "-n", "4", "-pad", "2"}, 200, `{"usage":{"completion_tokens":10},"timings":{"predicted_per_second":88.5}}`, 0},
		{"http error", []string{"-url", "http://inky.invalid/v1/chat/completions", "-n", "1"}, 500, `{"usage":{"completion_tokens":10}}`, 1},
		{"missing usage", []string{"-url", "http://inky.invalid/v1/chat/completions", "-n", "1"}, 200, `{}`, 1},
		{"invalid negative pad", []string{"-pad", "-1"}, 200, `{"usage":{"completion_tokens":10}}`, 2},
		{"invalid zero streams", []string{"-n", "0"}, 200, `{"usage":{"completion_tokens":10}}`, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var ticks atomic.Int64
			base := time.Date(2026, 11, 25, 0, 0, 0, 0, time.UTC)
			fakeNow := func() time.Time {
				n := ticks.Add(1)
				return base.Add(time.Duration(n) * 25 * time.Millisecond)
			}
			client := &http.Client{
				Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
					_, _ = io.Copy(io.Discard, r.Body)
					_ = r.Body.Close()
					return &http.Response{
						StatusCode: tc.status,
						Header:     http.Header{"Content-Type": []string{"application/json"}},
						Body:       io.NopCloser(strings.NewReader(tc.body)),
					}, nil
				}),
			}
			var stdout, stderr bytes.Buffer
			code := runLoad(tc.args, client, fakeNow, &stdout, &stderr)
			if code != tc.wantExit {
				t.Fatalf("got exit %d, want %d (stdout=%q stderr=%q)", code, tc.wantExit, stdout.String(), stderr.String())
			}
			out := stdout.String()
			if tc.wantExit == 0 {
				if !strings.Contains(out, "per-request-median=") || strings.Contains(out, "per-request-median=0.0") {
					t.Fatalf("missing measured rate: %s", out)
				}
			}
			if tc.name == "server timings preferred when present" && !strings.Contains(out, "per-request-median=88.5") {
				t.Fatalf("expected server timings rate 88.5, got: %s", out)
			}
		})
	}
}
