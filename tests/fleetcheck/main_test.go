package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

func TestSafeJoinPreventsTraversal(t *testing.T) {
	root := t.TempDir()
	if _, err := safeJoin(root, "../etc/passwd"); err == nil {
		t.Fatal("expected traversal to fail")
	}
	if _, err := safeJoin(root, "/abs/path"); err == nil {
		t.Fatal("expected absolute path to fail")
	}
	got, err := safeJoin(root, "sub/file.go")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := filepath.Join(root, "sub", "file.go")
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestExecToolReadEditRunTestRoundTrip(t *testing.T) {
	dir := t.TempDir()
	if err := setupFleetWorkspace(dir); err != nil {
		t.Fatalf("setupFleetWorkspace: %v", err)
	}

	// Initially go test must fail due to the Clamp upper-bound bug.
	out, ok := execTool(dir, "run_test", `{"command":"go test -v ./..."}`)
	if ok {
		t.Fatalf("expected initial run_test to fail, got ok=true: %s", out)
	}

	// read_file reads mathutil.go.
	src, ok := execTool(dir, "read_file", `{"path":"mathutil.go"}`)
	if !ok || !strings.Contains(src, "return lo") {
		t.Fatalf("read_file failed: ok=%v src=%q", ok, src)
	}

	// edit_file fixes the upper-bound bug.
	editArgs := `{"path":"mathutil.go","old_text":"\tif v > hi {\n\t\treturn lo\n\t}","new_text":"\tif v > hi {\n\t\treturn hi\n\t}"}`
	msg, ok := execTool(dir, "edit_file", editArgs)
	if !ok {
		t.Fatalf("edit_file failed: %s", msg)
	}

	// Now go test must pass.
	out, ok = execTool(dir, "run_test", `{"command":"go test -v ./..."}`)
	if !ok {
		t.Fatalf("expected post-edit run_test to pass, got: %s", out)
	}
	if _, err := os.Stat(filepath.Join(dir, "mathutil.go")); err != nil {
		t.Fatal(err)
	}
}

func TestFixtureFilesAreReadOnlyToModel(t *testing.T) {
	dir := t.TempDir()
	if err := setupFleetWorkspace(dir); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"mathutil_test.go", "go.mod"} {
		read, ok := execTool(dir, "read_file", fmt.Sprintf(`{"path":%q}`, name))
		if !ok {
			t.Fatalf("read %s: %s", name, read)
		}
		out, ok := execTool(dir, "edit_file", fmt.Sprintf(`{"path":%q,"old_text":"","new_text":"change"}`, name))
		if ok {
			t.Fatalf("accepted edit to %s: %s", name, out)
		}
		bytes, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil || string(bytes) != read {
			t.Fatalf("%s changed: %v", name, err)
		}
	}
}

func TestSandboxHidesCallerAndMountsFixtureReadOnly(t *testing.T) {
	dir := t.TempDir()
	if err := setupFleetWorkspace(dir); err != nil {
		t.Fatal(err)
	}
	marker := filepath.Join(t.TempDir(), "fleet-marker")
	if err := os.WriteFile(marker, []byte("test"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FLEET_TEST_MARKER", "test")
	src := fmt.Sprintf(`package fleetcard
import (
 "os"
 "testing"
)
func TestClamp(t *testing.T) {
	if os.Getenv("FLEET_TEST_MARKER") != "" { t.Error("inherited caller environment") }
	if _, err := os.Stat(%q); !os.IsNotExist(err) { t.Error("caller marker visible") }
	if err := os.WriteFile("card-scratch", nil, 0600); err == nil { t.Error("fixture is writable") }
}
`, marker)
	if err := os.WriteFile(filepath.Join(dir, "mathutil_test.go"), []byte(src), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := runCardTests(dir); err != nil {
		t.Fatalf("sandbox controls: %v\n%s", err, out)
	}
}

func TestCardVerifiesOnlySubmittedFunction(t *testing.T) {
	dir := t.TempDir()
	for _, src := range []string{
		"package fleetcard\nfunc Clamp(v, lo, hi int) int { return v }\nfunc helper() {}",
		"package fleetcard\n//go:noinline\nfunc Clamp(v, lo, hi int) int { return v }",
	} {
		if err := os.WriteFile(filepath.Join(dir, "mathutil.go"), []byte(src), 0600); err != nil {
			t.Fatal(err)
		}
		if out, err := runCardTests(dir); err == nil || !strings.Contains(err.Error(), "card") || out != "" {
			t.Fatalf("unexpected card verification: %q %v", out, err)
		}
	}
}

func TestMissingSandboxRefusesToRun(t *testing.T) {
	dir := t.TempDir()
	if err := setupFleetWorkspace(dir); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", t.TempDir())
	if out, err := runCardTests(dir); err == nil || !strings.Contains(err.Error(), "bwrap") {
		t.Fatalf("ran without sandbox: %s", out)
	}
}

func TestSandboxVerificationDeadline(t *testing.T) {
	dir := t.TempDir()
	if err := setupFleetWorkspace(dir); err != nil {
		t.Fatal(err)
	}
	if _, err := runCardTestsWithin(dir, time.Millisecond); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected verification deadline, got %v", err)
	}
}

func TestBoundedToolOutput(t *testing.T) {
	var output cardOutput
	const size = 64 << 10
	n, err := output.Write(bytes.Repeat([]byte{'a'}, size))
	if err != nil || n != size || len(output.String()) >= size || !strings.HasSuffix(output.String(), "[tool output truncated]") {
		t.Fatalf("output bound not enforced: wrote %d, retained %d: %v", n, len(output.String()), err)
	}
}

type fleetStep struct{ name, args string }

func TestCardKeepsWorkdirAndRequiresSuccessfulTest(t *testing.T) {
	passingRun := fleetStep{"run_test", `{"command":"go test ./..."}`}
	rejectedRun := fleetStep{"run_test", `{"command":"go version"}`}
	commentEdit := fleetStep{"edit_file", `{"path":"mathutil.go","old_text":"// Clamp restricts","new_text":"// Clamp limits"}`}
	for _, tc := range []struct {
		name     string
		steps    []fleetStep
		succeeds bool
	}{
		{"failed verification", []fleetStep{rejectedRun}, false},
		{"failure after a passing test", []fleetStep{passingRun, rejectedRun}, false},
		{"edit after a passing test", []fleetStep{passingRun, commentEdit}, false},
		{"verified fix", []fleetStep{passingRun}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			checkFleetCard(t, tc.steps, tc.succeeds)
		})
	}
}

func checkFleetCard(t *testing.T, verification []fleetStep, succeeds bool) {
	t.Helper()
	dir := t.TempDir()
	parentFile := filepath.Join(dir, "go.mod")
	const sentinel = "existing project"
	if err := os.WriteFile(parentFile, []byte(sentinel), 0600); err != nil {
		t.Fatal(err)
	}
	var turn atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var calls []toolCall
		if turn.Add(1) == 1 {
			steps := []fleetStep{
				{"read_file", `{"path":"mathutil.go"}`},
				{"edit_file", `{"path":"mathutil.go","old_text":"if v > hi {\n\t\treturn lo","new_text":"if v > hi {\n\t\treturn hi"}`},
			}
			for i, call := range append(steps, verification...) {
				tc := toolCall{ID: fmt.Sprint(i), Type: "function"}
				tc.Function.Name, tc.Function.Arguments = call.name, call.args
				calls = append(calls, tc)
			}
		}
		json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": chatMessage{Role: "assistant", ToolCalls: calls}}}})
	}))
	defer srv.Close()
	err := runFleetCard(srv.Client(), srv.URL, "", "test-card", dir)
	if turn.Load() != 2 {
		t.Fatalf("turns=%d err=%v", turn.Load(), err)
	}
	if succeeds && err != nil {
		t.Fatalf("card verification: %v", err)
	}
	if !succeeds && (err == nil || !strings.Contains(err.Error(), "expected successful")) {
		t.Fatalf("accepted failed run_test or failed early: %v", err)
	}
	got, err := os.ReadFile(parentFile)
	if err != nil || string(got) != sentinel {
		t.Fatalf("parent fixture overwritten: %q %v", got, err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 2 {
		t.Fatalf("expected parent file and fresh fixture dir, got %v (%v)", entries, err)
	}
}

func TestLiveGateRequiresEndpoint(t *testing.T) {
	cmd := exec.Command("bash", "../fleet_gate.sh")
	cmd.Env = []string{"PATH=" + os.Getenv("PATH")}
	out, err := cmd.CombinedOutput()
	if err == nil || !strings.Contains(string(out), "requires --url or INKY_URL") {
		t.Fatalf("live gate accepted missing endpoint: %s %v", out, err)
	}
}
