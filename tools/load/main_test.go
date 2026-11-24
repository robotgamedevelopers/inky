package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func TestLoadProgram(t *testing.T) {
	if os.Getenv("LOAD_TEST_CHILD") == "1" {
		os.Args = []string{"load", "-url", os.Getenv("LOAD_TEST_URL"), "-n", "1"}
		main()
		return
	}
	for _, tc := range []struct {
		name    string
		status  int
		body    string
		success bool
	}{
		{"successful response without timings", 200, `{"usage":{"completion_tokens":10}}`, true},
		{"http error", 500, `{"usage":{"completion_tokens":10}}`, false},
		{"missing usage", 200, `{}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				time.Sleep(30 * time.Millisecond)
				w.WriteHeader(tc.status)
				fmt.Fprint(w, tc.body)
			}))
			defer server.Close()
			cmd := exec.Command(os.Args[0], "-test.run=^TestLoadProgram$")
			cmd.Env = append(os.Environ(), "LOAD_TEST_CHILD=1", "LOAD_TEST_URL="+server.URL)
			out, err := cmd.CombinedOutput()
			if (err == nil) != tc.success {
				t.Fatalf("err=%v output=%s", err, out)
			}
			if tc.success && (!strings.Contains(string(out), "per-request-median=") || strings.Contains(string(out), "per-request-median=0.0")) {
				t.Fatalf("missing measured rate: %s", out)
			}
		})
	}
}
