package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func makeGLine(tokID, baseExp int) string {
	passes := make([]string, L)
	for l := 0; l < L; l++ {
		exps := make([]string, 8)
		for k := 0; k < 8; k++ {
			exps[k] = fmt.Sprintf("%d", (baseExp+l+k)%E)
		}
		passes[l] = strings.Join(exps, ",")
	}
	return fmt.Sprintf("G %d %s", tokID, strings.Join(passes, ";"))
}

func TestXsimRefusesEdgeInputsAndHandlesBlankLines(t *testing.T) {
	var out, errBuf bytes.Buffer

	// Empty argv refuses with usage exit 2 instead of panicking on Intn(0).
	if code := runXsim(nil, 2, &out, &errBuf); code != 2 {
		t.Fatalf("expected exit 2 on empty args, got %d", code)
	}

	// Missing file fails cleanly with exit 1 instead of nil os.File dereference.
	errBuf.Reset()
	if code := runXsim([]string{"nonexistent-routes-file.txt"}, 2, &out, &errBuf); code != 1 {
		t.Fatalf("expected exit 1 on missing file, got %d", code)
	}

	// Malformed inputs rejected by loadReader.
	for _, tc := range []struct {
		name string
		data string
	}{
		{"empty file", "\n   \n"},
		{"only one G token", "P 10\n" + makeGLine(11, 0) + "\n"},
		{"short G line", "G 11\n" + makeGLine(12, 1) + "\n"},
		{"wrong layer count", "G 11 0,1,2;3,4,5\n" + makeGLine(12, 1) + "\n"},
		{"expert out of range", strings.Replace(makeGLine(11, 0), "0,1", "0,128", 1) + "\n" + makeGLine(12, 1) + "\n"},
		{"negative expert", strings.Replace(makeGLine(11, 0), "0,1", "-1,1", 1) + "\n" + makeGLine(12, 1) + "\n"},
		{"unknown tag", "X 11\n" + makeGLine(12, 1) + "\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := loadReader(strings.NewReader(tc.data)); err == nil {
				t.Fatalf("expected error for %s", tc.name)
			}
		})
	}

	// Valid routes file with blank lines and P lines runs cleanly.
	dir, err := os.MkdirTemp(".", ".tmp-xsim-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	p := filepath.Join(dir, "routes.txt")
	content := "\nP 100\n   \n" + makeGLine(101, 0) + "\n\n" + makeGLine(102, 8) + "\n" + makeGLine(103, 64) + "\n"
	if err := os.WriteFile(p, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errBuf.Reset()
	if code := runXsim([]string{p}, 3, &out, &errBuf); code != 0 {
		t.Fatalf("runXsim valid failed (code=%d): %s", code, errBuf.String())
	}
	if !strings.Contains(out.String(), "greedy pool 4B") {
		t.Fatalf("unexpected xsim output:\n%s", out.String())
	}
}
