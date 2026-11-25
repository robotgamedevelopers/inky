package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeGGUFFixture(t *testing.T, path string) {
	t.Helper()
	// Construct a complete two-tensor file independently of the writer.
	var b []byte
	b = append(b, "GGUF"...)
	b = binary.LittleEndian.AppendUint32(b, 3)
	b = binary.LittleEndian.AppendUint64(b, 2)
	b = binary.LittleEndian.AppendUint64(b, 0)
	for i, name := range []string{"first", "last"} {
		b = binary.LittleEndian.AppendUint64(b, uint64(len(name)))
		b = append(b, name...)
		b = binary.LittleEndian.AppendUint32(b, 1)
		b = binary.LittleEndian.AppendUint64(b, 1)
		b = binary.LittleEndian.AppendUint32(b, tF32)
		b = binary.LittleEndian.AppendUint64(b, uint64(i*ggufAlign))
	}
	b = append(b, make([]byte, (-len(b))&(ggufAlign-1))...)
	b = binary.LittleEndian.AppendUint32(b, math.Float32bits(1.25))
	b = append(b, make([]byte, ggufAlign-4)...)
	b = binary.LittleEndian.AppendUint32(b, math.Float32bits(-2.5))
	if err := os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
}

func emptySafetensorsFixture(t *testing.T, path string) {
	t.Helper()
	header := binary.LittleEndian.AppendUint64(nil, 2)
	if err := os.WriteFile(path, append(header, "{}"...), 0600); err != nil {
		t.Fatal(err)
	}
}

func fileContents(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestConvertPublishesCompleteTensors(t *testing.T) {
	dir := t.TempDir()
	base, official, out := filepath.Join(dir, "base"), filepath.Join(dir, "official"), filepath.Join(dir, "out")
	writeGGUFFixture(t, base)
	emptySafetensorsFixture(t, official)
	if err := os.WriteFile(out, []byte("previous model"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := convert(official, base, "", "", "", "base", out, "interleaved"); err != nil {
		t.Fatal(err)
	}
	g, err := openGGUF(out)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	for name, want := range map[string]float32{"first": 1.25, "last": -2.5} {
		tensor, ok := g.tensor(name)
		if !ok {
			t.Fatalf("missing %s", name)
		}
		b, err := g.readRows(tensor, 0, 1)
		if err != nil || math.Float32frombits(binary.LittleEndian.Uint32(b)) != want {
			t.Fatalf("reading %s: %x, %v", name, b, err)
		}
	}
}

func TestConvertRejectsInputAliases(t *testing.T) {
	dir := t.TempDir()
	base, official, q8 := filepath.Join(dir, "base"), filepath.Join(dir, "official"), filepath.Join(dir, "q8")
	writeGGUFFixture(t, base)
	writeGGUFFixture(t, q8)
	emptySafetensorsFixture(t, official)
	hardlink, symlink := filepath.Join(dir, "hard"), filepath.Join(dir, "sym")
	if err := os.Link(base, hardlink); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(official, symlink); err != nil {
		t.Fatal(err)
	}
	for _, out := range []string{base, hardlink, symlink, q8} {
		before := fileContents(t, out)
		err := convert(official, base, q8, "last", "", "base", out, "interleaved")
		if err == nil || !strings.Contains(err.Error(), "aliases input") {
			t.Fatalf("output %s: %v", out, err)
		}
		if !bytes.Equal(before, fileContents(t, out)) {
			t.Fatalf("changed input %s", out)
		}
	}
}

func TestCopyRejectsShortTensor(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "base")
	writeGGUFFixture(t, path)
	g, err := openGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	last, _ := g.tensor("last")
	if err := os.Truncate(path, g.dataBase+int64(last.Off)+2); err != nil {
		t.Fatal(err)
	}
	if err := g.copyTensor(io.Discard, last); !errors.Is(err, io.EOF) {
		t.Fatalf("source truncated after open: %v", err)
	}
	if got, err := openGGUF(path); err == nil {
		got.Close()
		t.Fatal("opened incomplete tensor data")
	}
}

func TestMTPFailurePreservesDestination(t *testing.T) {
	dir := t.TempDir()
	mtp, official, out := filepath.Join(dir, "mtp"), filepath.Join(dir, "official"), filepath.Join(dir, "out")
	emptySafetensorsFixture(t, mtp)
	emptySafetensorsFixture(t, official)
	if err := convertMTP(mtp, official, mtp); err == nil || !strings.Contains(err.Error(), "aliases input") {
		t.Fatalf("MTP alias: %v", err)
	}
	previous := []byte("previously converted model")
	if err := os.WriteFile(out, previous, 0600); err != nil {
		t.Fatal(err)
	}
	// Missing weights fail after creating the temporary GGUF and writing its header.
	if err := convertMTP(mtp, official, out); err == nil || !strings.Contains(err.Error(), "no tensor") {
		t.Fatalf("missing MTP weights: %v", err)
	}
	if !bytes.Equal(previous, fileContents(t, out)) {
		t.Fatal("failed conversion replaced the destination")
	}
	leftovers, err := filepath.Glob(filepath.Join(dir, ".inkconvert-*"))
	if err != nil || len(leftovers) != 0 {
		t.Fatalf("temporary outputs after error: %v, %v", leftovers, err)
	}
}
