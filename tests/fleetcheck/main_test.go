package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
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
