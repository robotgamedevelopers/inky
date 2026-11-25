package main

import (
	"bytes"
	"context"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// The card has no third-party dependencies. Only the Go distribution and the
// read-only fixture are visible to the isolated compiler and test process.
func runCardTests(dir string) (string, error) {
	return runCardTestsWithin(dir, 90*time.Second)
}

func runCardTestsWithin(dir string, timeout time.Duration) (string, error) {
	if runtime.GOOS != "linux" {
		return "", errors.New("fleet-card requires a Linux worker with bubblewrap")
	}
	root := runtime.GOROOT()
	if !filepath.IsAbs(root) {
		return "", errors.New("fleet-card requires the absolute path of a local Go toolchain in GOROOT")
	}
	absDir, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	// Only the submitted integer function belongs in this card. Keep package
	// startup, imports and compiler directives under the acceptance harness's
	// control so only the harness can terminate verification successfully.
	f, err := parser.ParseFile(token.NewFileSet(), filepath.Join(absDir, "mathutil.go"), nil, parser.ParseComments|parser.SkipObjectResolution)
	if err != nil {
		return "", err
	}
	if f.Name.Name != "fleetcard" || len(f.Decls) != 1 {
		return "", errors.New("the card must contain just the Clamp function in package fleetcard")
	}
	clamp, ok := f.Decls[0].(*ast.FuncDecl)
	if !ok || clamp.Recv != nil || clamp.Name.Name != "Clamp" {
		return "", errors.New("the card requires the Clamp function")
	}
	for _, group := range f.Comments {
		for _, comment := range group.List {
			if strings.HasPrefix(comment.Text, "//go:") {
				return "", errors.New("compiler directives are unavailable in the card")
			}
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "bwrap",
		"--unshare-all", "--unshare-user", "--disable-userns", "--die-with-parent", "--new-session",
		"--cap-drop", "ALL", "--ro-bind", root, "/go", "--ro-bind", absDir, "/card",
		"--dev", "/dev", "--proc", "/proc", "--size", "1073741824", "--tmpfs", "/tmp",
		"--remount-ro", "/", "--chdir", "/card", "--clearenv",
		"--setenv", "HOME", "/tmp", "--setenv", "PATH", "/go/bin",
		"--setenv", "GOROOT", "/go", "--setenv", "GOCACHE", "/tmp/cache",
		"--setenv", "CGO_ENABLED", "0", "--setenv", "GOPROXY", "off",
		"--setenv", "GOTOOLCHAIN", "local", "--setenv", "GOMAXPROCS", "2",
		"--", "/go/bin/go", "test", "-count=1", "-v", "./...")
	cmd.Env = []string{}
	cmd.WaitDelay = 2 * time.Second
	var out cardOutput
	cmd.Stdout, cmd.Stderr = &out, &out
	err = cmd.Run()
	if ctx.Err() != nil {
		return out.String(), ctx.Err()
	}
	if err == nil && !strings.Contains(out.String(), "--- PASS: TestClamp") {
		return out.String(), errors.New("Clamp acceptance test did not report a pass")
	}
	return out.String(), err
}

type cardOutput struct {
	data      bytes.Buffer
	truncated bool
}

func (w *cardOutput) Write(p []byte) (int, error) {
	const limit = 32 << 10
	n := min(len(p), limit-w.data.Len())
	w.data.Write(p[:n])
	if n < len(p) {
		w.truncated = true
	}
	return len(p), nil
}

func (w *cardOutput) String() string {
	if w.truncated {
		return w.data.String() + "\n[tool output truncated]"
	}
	return w.data.String()
}
