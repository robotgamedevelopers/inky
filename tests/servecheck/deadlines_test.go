package main

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestRunGivesEachCheckItsOwnDeadline(t *testing.T) {
	suite, cancel := context.WithTimeout(context.Background(), suiteTimeout)
	defer cancel()
	g := gate{ctx: suite}
	for _, name := range []string{"first", "second"} {
		var ctx context.Context
		start := time.Now()
		g.run(name, func(check *gate) error {
			ctx = check.ctx
			deadline, ok := ctx.Deadline()
			if !ok || deadline.Before(start.Add(checkTimeout)) || deadline.After(time.Now().Add(checkTimeout)) {
				t.Fatalf("%s deadline = %v, want a fresh %v budget", name, deadline, checkTimeout)
			}
			suiteDeadline, _ := suite.Deadline()
			if !suiteDeadline.After(deadline) {
				t.Fatalf("suite deadline %v leaves no time after check deadline %v", suiteDeadline, deadline)
			}
			return ctx.Err()
		})
		if ctx == nil || ctx.Err() != context.Canceled || g.failed || suite.Err() != nil {
			t.Fatalf("%s did not complete and release its own context", name)
		}
	}
}

func TestCheckCancellationReachesHTTPRequests(t *testing.T) {
	for _, tc := range []struct {
		name string
		call func(*gate) error
	}{
		{"health", func(g *gate) error { _, _, err := g.fetchHealth("/health"); return err }},
		{"reply", func(g *gate) error { _, err := g.sampleReply(`{}`, true, nil); return err }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			suite, cancel := context.WithCancel(context.Background())
			defer cancel()
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// net/http observes a disconnect once the request body has been read.
				_, _ = io.Copy(io.Discard, r.Body)
				cancel()
				select {
				case <-r.Context().Done():
				case <-time.After(2 * time.Second):
					t.Error("client kept the request open after suite cancellation")
				}
			}))
			defer server.Close()
			g := gate{ctx: suite, url: server.URL}
			var requestErr error
			g.run(tc.name, func(check *gate) error {
				requestErr = tc.call(check)
				return requestErr
			})
			if !errors.Is(requestErr, context.Canceled) || !g.failed {
				t.Fatalf("request error = %v, gate failed = %t", requestErr, g.failed)
			}
		})
	}
}

func TestServerReapsProcessAfterChecks(t *testing.T) {
	const standIn = `#!/bin/sh
printf '%s\n' "$$" > "$3"
printf '%s\n' '[serve] http://127.0.0.1:4123/v1: ready'
exec sleep 30
`
	for _, tc := range []struct {
		name   string
		cancel bool
	}{
		{"completed", false},
		{"canceled", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			bin, pidPath := filepath.Join(dir, "server"), filepath.Join(dir, "pid")
			if err := os.WriteFile(bin, []byte(standIn), 0700); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), suiteTimeout)
			defer cancel()
			g := gate{ctx: ctx}
			var ready bool
			err := g.server(ctx, bin, pidPath, dir, 1, 1, func(server *gate) {
				ready = true
				serverDeadline, serverBounded := server.ctx.Deadline()
				suiteDeadline, _ := ctx.Deadline()
				if !serverBounded || !serverDeadline.Equal(suiteDeadline) {
					t.Fatalf("server deadline %v must cover the entire suite through %v", serverDeadline, suiteDeadline)
				}
				if tc.cancel {
					cancel()
				}
			})
			if !ready || (tc.cancel && !errors.Is(err, context.Canceled)) || (!tc.cancel && err != nil) {
				t.Fatalf("server ready = %t, returned %v", ready, err)
			}
			data, err := os.ReadFile(pidPath)
			if err != nil {
				t.Fatal(err)
			}
			pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
			if err != nil || pid < 1 {
				t.Fatalf("invalid server PID %q", data)
			}
			if err := syscall.Kill(pid, 0); !errors.Is(err, syscall.ESRCH) {
				t.Fatalf("server PID %d still exists after server returned (signal 0: %v)", pid, err)
			}
		})
	}
}
