package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"sync/atomic"
	"testing"
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
		success        bool
	}{
		{"success", 200, "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", true},
		{"all workload rejected", 400, "bad request", false},
		{"truncated stream", 200, "data: [DONE]\n\n", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var count atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				count.Add(1)
				var req chatReq
				_ = json.NewDecoder(r.Body).Decode(&req)
				if tc.name == "all workload rejected" && req.MaxTokens == 8 {
					fmt.Fprint(w, "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
					return
				}
				w.WriteHeader(tc.workloadStatus)
				fmt.Fprint(w, tc.body)
			}))
			defer server.Close()
			cmd := exec.Command(os.Args[0], "-test.run=^TestSoakVerdict$")
			cmd.Env = append(os.Environ(), "SOAK_TEST_CHILD=1", "SOAK_TEST_URL="+server.URL, "INKY_SERVE_KEY=test")
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
