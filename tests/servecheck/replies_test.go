package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func sseChunk(id, role, delta, finish string, promptToks, genToks int) string {
	if finish == "" {
		return fmt.Sprintf("data: {\"id\":%q,\"choices\":[{\"delta\":{\"role\":%q,\"content\":%q}}]}\n\n", id, role, delta)
	}
	return fmt.Sprintf(
		"data: {\"id\":%q,\"choices\":[{\"delta\":{\"content\":%q},\"finish_reason\":%q}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d}}\n\ndata: [DONE]\n\n",
		id, delta, finish, promptToks, genToks,
	)
}

func TestSeededRepliesUsesDistinctPromptsAndCatchesCrossLaneMixup(t *testing.T) {
	logPath := filepath.Join(t.TempDir(), "serve.log")
	if err := os.WriteFile(logPath, nil, 0o600); err != nil {
		t.Fatal(err)
	}

	origClient := replyClient
	defer func() { replyClient = origClient }()

	runScenario := func(swapConcurrent bool, collapseIdentical bool) error {
		var seq atomic.Int64
		var overlapMu sync.Mutex
		overlapSeen := 0
		bothStarted := make(chan struct{})

		replyClient = &http.Client{
			Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				bodyBytes, err := io.ReadAll(req.Body)
				if err != nil {
					return nil, err
				}
				body := string(bodyBytes)
				call := int(seq.Add(1))
				id := fmt.Sprintf("chatcmpl-%d", call)

				isIntegers := strings.Contains(body, "integers 1 through 10000")
				which := 1
				if isIntegers {
					which = 0
				}
				// Calls 1 and 2 are the solo reference runs; calls 3 and 4 are the concurrent pair.
				isConcurrent := call >= 3
				if isConcurrent && swapConcurrent {
					which = 1 - which
				}
				if collapseIdentical {
					which = 0
				}

				firstDelta, tailDelta, promptToks := "1, 2, 3, ", "4, 5, 6", 24
				if which == 1 {
					firstDelta, tailDelta, promptToks = "alpha beta ", "gamma delta", 31
				}

				pr, pw := io.Pipe()
				go func() {
					defer pw.Close()
					_, _ = io.WriteString(pw, sseChunk(id, "assistant", firstDelta, "", 0, 0))
					if isConcurrent {
						overlapMu.Lock()
						overlapSeen++
						if overlapSeen == 2 {
							close(bothStarted)
						}
						overlapMu.Unlock()
						<-bothStarted
					}
					_, _ = io.WriteString(pw, sseChunk(id, "", tailDelta, "length", promptToks, 64))
				}()

				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
					Body:       pr,
					Request:    req,
				}, nil
			}),
		}

		g := &gate{
			ctx: context.Background(),
			url: "http://inky.invalid",
			key: "test-key",
			log: logPath,
		}
		return g.seededReplies()
	}

	if err := runScenario(false, false); err != nil {
		t.Fatalf("expected distinct overlapping lanes to pass, got: %v", err)
	}

	if err := runScenario(true, false); err == nil || !strings.Contains(err.Error(), "cross-lane reply mix-up") {
		t.Fatalf("expected cross-lane swap to fail with mix-up error, got: %v", err)
	}

	if err := runScenario(false, true); err == nil || !strings.Contains(err.Error(), "distinct") {
		t.Fatalf("expected collapsed identical replies to fail, got: %v", err)
	}
}
