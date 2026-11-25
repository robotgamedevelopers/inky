package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestQueueJobs(t *testing.T) {
	for _, tc := range []struct {
		name        string
		wrongQueue  bool
		badOverflow int32
		wantPosts   int32
	}{
		{name: "bounded streams drain and capacity recovers", wantPosts: 8},
		{name: "third request missing from queue", wrongQueue: true, wantPosts: 3},
		{name: "fourth request accepted", badOverflow: 4, wantPosts: 4},
		{name: "fifth request missing Retry-After", badOverflow: 5, wantPosts: 5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			bothHolders := make(chan struct{})
			queued := make(chan struct{})
			overflowSent := make(chan struct{})
			freeLane := make(chan struct{}, 2)
			var posts, busyPolls, idlePolls, completed atomic.Int32
			mux := http.NewServeMux()
			mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
				if completed.Load() >= 3 {
					idlePolls.Add(1)
					io.WriteString(w, `{"ok":true,"active_lanes":0,"queued_requests":0,"inflight":0}`)
					return
				}
				select {
				case <-queued:
				case <-r.Context().Done():
					return
				}
				poll := busyPolls.Add(1)
				queuedRequests := 1
				if tc.wrongQueue {
					queuedRequests = 0
					if poll == 2 {
						// A retry proves the first incorrect snapshot did not admit the overflow probes.
						cancel()
					}
				}
				fmt.Fprintf(w, `{"ok":true,"active_lanes":2,"queued_requests":%d,"inflight":3}`, queuedRequests)
			})
			mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
				var req struct {
					Limit  int `json:"max_tokens"`
					Stream bool
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil || !req.Stream {
					t.Errorf("expected streaming request, got %+v, error %v", req, err)
					return
				}
				n := posts.Add(1)
				if n == 4 || n == 5 {
					code := http.StatusTooManyRequests
					if !(n == 5 && tc.badOverflow == 5) {
						w.Header().Set("Retry-After", "1")
					}
					if n == 4 && tc.badOverflow == 4 {
						code = http.StatusOK
					}
					w.WriteHeader(code)
					io.WriteString(w, `{"error":{"message":"queue full"}}`)
					if n == 5 {
						close(overflowSent)
					}
					return
				}
				if n == 3 {
					close(queued)
					select {
					case <-freeLane:
					case <-r.Context().Done():
						return
					}
				}
				w.Header().Set("Content-Type", "text/event-stream")
				if n <= 2 {
					if req.Limit < 1 || req.Limit > 64 {
						t.Errorf("holder %d max_tokens = %d, want 1..64 for the queue check budget", n, req.Limit)
					}
					w.(http.Flusher).Flush()
					if n == 2 {
						close(bothHolders)
					}
					select {
					case <-bothHolders:
					case <-r.Context().Done():
						return
					}
				}
				io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"1\"}}]}\n\n")
				w.(http.Flusher).Flush()
				if n <= 2 {
					select {
					case <-overflowSent:
					case <-r.Context().Done():
						return
					}
					freeLane <- struct{}{}
				}
				completed.Add(1)
				io.WriteString(w, "data: {\"choices\":[{\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":1}}\n\ndata: [DONE]\n\n")
			})
			server := httptest.NewServer(mux)
			defer server.Close()
			g := gate{ctx: ctx, url: server.URL}
			g.queueJobs()
			wantFailure := tc.wrongQueue || tc.badOverflow != 0
			if g.failed != wantFailure || posts.Load() != tc.wantPosts || busyPolls.Load() < 1 {
				t.Fatalf("failed=%t, posts=%d, queue polls=%d; want failed=%t and %d posts", g.failed, posts.Load(), busyPolls.Load(), wantFailure, tc.wantPosts)
			}
			if tc.wrongQueue && busyPolls.Load() != 2 {
				t.Fatalf("incorrect queue snapshot was not retried, polls=%d", busyPolls.Load())
			}
			if !wantFailure && (completed.Load() != 6 || idlePolls.Load() != 2) {
				t.Fatalf("completed=%d, idle polls=%d; want 3 initial and 3 recovery streams, both groups drained", completed.Load(), idlePolls.Load())
			}
		})
	}
}
