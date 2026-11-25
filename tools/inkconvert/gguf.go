package main

import (
	"bufio"
	"crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

// GGUF v3, only as much as this tool needs: the key-value section is kept as raw bytes (and can be appended to),
// tensors are described by name, dims, type and offset.

const conversionKey = "inkserve.experts"

const ggufAlign = 32 // general.alignment's default; the Inkling files don't set it

// GGML type ids. NV4 and MX8 are private to inkserve's erisinky branch; llama.cpp can't load them.
const (
	tF32  = 0
	tF16  = 1
	tQ4_0 = 2
	tQ5_0 = 6
	tQ8_0 = 8
	tQ4_K = 12
	tQ5_K = 13
	tQ6_K = 14
	tBF16 = 30
	tNV4  = 200 // 32 weights: 2 FP8 E4M3 scales (one per 16) + 16 bytes of FP4 E2M1 nibbles, 18 bytes
	tMX8  = 201 // 32 weights: fp16 power-of-two scale + 32 FP8 E4M3 values, 34 bytes
)

// typeBlock returns (bytes per block, weights per block).
func typeBlock(t uint32) (int, int, error) {
	switch t {
	case tF32:
		return 4, 1, nil
	case tF16, tBF16:
		return 2, 1, nil
	case tQ4_0:
		return 18, 32, nil
	case tQ5_0:
		return 22, 32, nil
	case tQ8_0:
		return 34, 32, nil
	case tQ4_K:
		return 144, 256, nil
	case tQ5_K:
		return 176, 256, nil
	case tQ6_K:
		return 210, 256, nil
	case tNV4:
		return 18, 32, nil
	case tMX8:
		return 34, 32, nil
	}
	return 0, 0, fmt.Errorf("unknown ggml type %d", t)
}

type tensorInfo struct {
	Name string
	Dims []uint64 // ne0 (innermost, the row length) first
	Type uint32
	Off  uint64 // from the start of the data section
}

func (t tensorInfo) rowLen() int { return int(t.Dims[0]) }

func (t tensorInfo) nRows() int {
	n := 1
	for _, d := range t.Dims[1:] {
		n *= int(d)
	}
	return n
}

func (t tensorInfo) size() (uint64, error) {
	bs, bw, err := typeBlock(t.Type)
	if err != nil {
		return 0, fmt.Errorf("%s: %w", t.Name, err)
	}
	if len(t.Dims) == 0 || len(t.Dims) > 4 || t.Dims[0]%uint64(bw) != 0 {
		return 0, fmt.Errorf("%s: invalid dimensions %v for ggml type %d", t.Name, t.Dims, t.Type)
	}
	size := uint64(bs)
	for i, dim := range t.Dims {
		if dim == 0 || dim > math.MaxInt {
			return 0, fmt.Errorf("%s: invalid dimension %d", t.Name, dim)
		}
		if i == 0 {
			dim /= uint64(bw)
		}
		if size > math.MaxInt/dim {
			return 0, fmt.Errorf("%s: tensor size overflows", t.Name)
		}
		size *= dim
	}
	return size, nil
}

func (t tensorInfo) rowBytes() int {
	bs, bw, _ := typeBlock(t.Type)
	return t.rowLen() / bw * bs
}

type gguf struct {
	f        *os.File
	nKV      uint64
	kvRaw    []byte
	tensors  []tensorInfo
	byName   map[string]int
	dataBase int64
}

func openGGUF(path string) (*gguf, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	g := &gguf{f: f, byName: map[string]int{}}
	if err := g.parse(); err != nil {
		f.Close()
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return g, nil
}

func (g *gguf) Close() error { return g.f.Close() }

// countingReader tracks the offset, so the raw key-value bytes can be sliced out of the file.
type countingReader struct {
	r *bufio.Reader
	n int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}

func (g *gguf) parse() error {
	cr := &countingReader{r: bufio.NewReaderSize(g.f, 1<<20)}
	var hdr struct {
		Magic   [4]byte
		Version uint32
		NT, NKV uint64
	}
	if err := binary.Read(cr, binary.LittleEndian, &hdr); err != nil {
		return err
	}
	if string(hdr.Magic[:]) != "GGUF" || hdr.Version != 3 {
		return errors.New("not a GGUF v3 file")
	}
	g.nKV = hdr.NKV
	kvStart := cr.n
	var provenanceStart, provenanceEnd int64
	for i := uint64(0); i < hdr.NKV; i++ {
		entryStart := cr.n
		key, err := readString(cr)
		if err != nil {
			return err
		}
		var typ uint32
		if err := binary.Read(cr, binary.LittleEndian, &typ); err != nil {
			return err
		}
		if key == "general.alignment" {
			return errors.New("general.alignment is set; this tool assumes the default of 32")
		}
		if err := skipValue(cr, typ); err != nil {
			return fmt.Errorf("key %s: %w", key, err)
		}
		if key == conversionKey {
			if provenanceEnd != 0 {
				return fmt.Errorf("duplicate metadata key %s", key)
			}
			provenanceStart, provenanceEnd = entryStart-kvStart, cr.n-kvStart
		}
	}
	kvEnd := cr.n
	g.kvRaw = make([]byte, kvEnd-kvStart)
	if _, err := g.f.ReadAt(g.kvRaw, kvStart); err != nil {
		return err
	}
	// A new conversion writes its own provenance. Keep every other metadata record byte-for-byte.
	if provenanceEnd != 0 {
		g.kvRaw = append(g.kvRaw[:provenanceStart], g.kvRaw[provenanceEnd:]...)
		g.nKV--
	}
	for i := uint64(0); i < hdr.NT; i++ {
		name, err := readString(cr)
		if err != nil {
			return err
		}
		var nd uint32
		if err := binary.Read(cr, binary.LittleEndian, &nd); err != nil {
			return err
		}
		if nd < 1 || nd > 4 {
			return fmt.Errorf("%s: invalid tensor rank %d", name, nd)
		}
		t := tensorInfo{Name: name, Dims: make([]uint64, nd)}
		if err := binary.Read(cr, binary.LittleEndian, t.Dims); err != nil {
			return err
		}
		if err := binary.Read(cr, binary.LittleEndian, &t.Type); err != nil {
			return err
		}
		if err := binary.Read(cr, binary.LittleEndian, &t.Off); err != nil {
			return err
		}
		g.byName[name] = len(g.tensors)
		g.tensors = append(g.tensors, t)
	}
	g.dataBase = (cr.n + ggufAlign - 1) / ggufAlign * ggufAlign
	info, err := g.f.Stat()
	if err != nil {
		return err
	}
	for _, t := range g.tensors {
		size, err := t.size()
		if err != nil {
			return err
		}
		if t.Off%ggufAlign != 0 || info.Size() < g.dataBase {
			return fmt.Errorf("%s: invalid tensor offset", t.Name)
		}
		dataSize := uint64(info.Size() - g.dataBase)
		if t.Off > dataSize || size > dataSize-t.Off {
			return fmt.Errorf("%s: tensor extends past end of file", t.Name)
		}
	}
	return nil
}

func (g *gguf) tensor(name string) (tensorInfo, bool) {
	i, ok := g.byName[name]
	if !ok {
		return tensorInfo{}, false
	}
	return g.tensors[i], true
}

// readRows reads n rows starting at row r0 of a tensor.
func (g *gguf) readRows(t tensorInfo, r0, n int) ([]byte, error) {
	if len(t.Dims) == 0 || r0 < 0 || n < 0 || r0 > t.nRows() || n > t.nRows()-r0 {
		return nil, fmt.Errorf("%s: row range is outside tensor", t.Name)
	}
	rb := t.rowBytes()
	buf := make([]byte, n*rb)
	_, err := g.f.ReadAt(buf, g.dataBase+int64(t.Off)+int64(r0)*int64(rb))
	return buf, err
}

// The source can become shorter after the table has been validated.
func (g *gguf) copyTensor(w io.Writer, t tensorInfo) error {
	size, err := t.size()
	if err != nil {
		return err
	}
	r := io.NewSectionReader(g.f, g.dataBase+int64(t.Off), int64(size))
	if _, err := io.CopyN(w, r, int64(size)); err != nil {
		return fmt.Errorf("copying %s: %w", t.Name, err)
	}
	return nil
}

func readString(r io.Reader) (string, error) {
	var n uint64
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return "", err
	}
	if n > 1<<20 {
		return "", fmt.Errorf("string of %d bytes", n)
	}
	b := make([]byte, n)
	_, err := io.ReadFull(r, b)
	return string(b), err
}

func skipValue(r io.Reader, typ uint32) error {
	fixed := map[uint32]int64{0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
	if n, ok := fixed[typ]; ok {
		_, err := io.CopyN(io.Discard, r, n)
		return err
	}
	switch typ {
	case 8:
		_, err := readString(r)
		return err
	case 9:
		var at uint32
		var n uint64
		if err := binary.Read(r, binary.LittleEndian, &at); err != nil {
			return err
		}
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return err
		}
		for i := uint64(0); i < n; i++ {
			if err := skipValue(r, at); err != nil {
				return err
			}
		}
		return nil
	}
	return fmt.Errorf("unknown value type %d", typ)
}

// appendKVString encodes one string key-value pair in GGUF form.
func appendKVString(b []byte, key, val string) []byte {
	b = binary.LittleEndian.AppendUint64(b, uint64(len(key)))
	b = append(b, key...)
	b = binary.LittleEndian.AppendUint32(b, 8)
	b = binary.LittleEndian.AppendUint64(b, uint64(len(val)))
	return append(b, val...)
}

// ggufWriter writes a GGUF file whose tensor list is fixed up front; tensor data then arrives in order.
type ggufWriter struct {
	w       *bufio.Writer
	f       *os.File
	path    string
	tensors []tensorInfo
	next    int
	written uint64
	end     uint64
}

func outputDestination(path string) (string, os.FileInfo, error) {
	// Resolve the parent before interpreting a relative final symlink. Atomic rename
	// must publish at the target even when that target has not been created yet.
	for hops := 0; ; hops++ {
		parent, leaf := filepath.Split(path)
		if parent == "" {
			parent = "."
		}
		parent, err := filepath.EvalSymlinks(parent)
		if err != nil {
			return "", nil, err
		}
		path = filepath.Join(parent, leaf)
		linkInfo, err := os.Lstat(path)
		if os.IsNotExist(err) {
			return path, nil, nil
		}
		if err != nil {
			return "", nil, err
		}
		if linkInfo.Mode()&os.ModeSymlink == 0 {
			return path, linkInfo, nil
		}
		if hops >= 40 {
			return "", nil, fmt.Errorf("too many symlinks resolving output %s", path)
		}
		target, err := os.Readlink(path)
		if err != nil {
			return "", nil, err
		}
		path = target
		if !filepath.IsAbs(target) {
			// Preserve any intermediate symlink/.. until EvalSymlinks resolves it.
			path = parent + string(filepath.Separator) + target
		}
	}
}

func createGGUF(path string, nKV uint64, kvRaw []byte, tensors []tensorInfo, inputs ...*os.File) (*ggufWriter, error) {
	off := uint64(0)
	for i := range tensors {
		sz, err := tensors[i].size()
		if err != nil {
			return nil, err
		}
		tensors[i].Off = off
		if sz > math.MaxInt64-off-(ggufAlign-1) {
			return nil, fmt.Errorf("%s: output offset overflows", tensors[i].Name)
		}
		off = (off + sz + ggufAlign - 1) / ggufAlign * ggufAlign
	}
	path, info, err := outputDestination(path)
	if err != nil {
		return nil, err
	}
	if info != nil {
		if !info.Mode().IsRegular() {
			return nil, fmt.Errorf("%s: output must be a regular file", path)
		}
		for _, input := range inputs {
			src, err := input.Stat()
			if err != nil {
				return nil, err
			}
			if os.SameFile(info, src) {
				return nil, fmt.Errorf("%s: output aliases input %s", path, input.Name())
			}
		}
	}
	temp := filepath.Join(filepath.Dir(path), ".inkconvert-"+rand.Text())
	f, err := os.OpenFile(temp, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0666)
	if err != nil {
		return nil, err
	}
	gw := &ggufWriter{w: bufio.NewWriterSize(f, 8<<20), f: f, path: path, tensors: tensors}
	if info != nil {
		if err := f.Chmod(info.Mode().Perm()); err != nil {
			gw.discard()
			return nil, err
		}
	}
	var hdr []byte
	hdr = append(hdr, "GGUF"...)
	hdr = binary.LittleEndian.AppendUint32(hdr, 3)
	hdr = binary.LittleEndian.AppendUint64(hdr, uint64(len(tensors)))
	hdr = binary.LittleEndian.AppendUint64(hdr, nKV)
	hdr = append(hdr, kvRaw...)
	for _, t := range tensors {
		hdr = binary.LittleEndian.AppendUint64(hdr, uint64(len(t.Name)))
		hdr = append(hdr, t.Name...)
		hdr = binary.LittleEndian.AppendUint32(hdr, uint32(len(t.Dims)))
		for _, d := range t.Dims {
			hdr = binary.LittleEndian.AppendUint64(hdr, d)
		}
		hdr = binary.LittleEndian.AppendUint32(hdr, t.Type)
		hdr = binary.LittleEndian.AppendUint64(hdr, t.Off)
	}
	for len(hdr)%ggufAlign != 0 {
		hdr = append(hdr, 0)
	}
	if _, err := gw.w.Write(hdr); err != nil {
		gw.discard()
		return nil, err
	}
	return gw, nil
}

// The caller defers discard immediately after createGGUF. A successful publish disarms it.
func (gw *ggufWriter) discard() {
	if gw.f != nil {
		gw.f.Close()
		os.Remove(gw.f.Name())
		gw.f = nil
	}
}

// beginTensor pads to the next tensor's offset and returns it; the caller then writes exactly its size.
func (gw *ggufWriter) beginTensor(name string) (tensorInfo, error) {
	if gw.written != gw.end {
		return tensorInfo{}, fmt.Errorf("incomplete preceding tensor before %s", name)
	}
	if gw.next >= len(gw.tensors) || gw.tensors[gw.next].Name != name {
		return tensorInfo{}, fmt.Errorf("tensor %s written out of order", name)
	}
	t := gw.tensors[gw.next]
	gw.next++
	size, _ := t.size() // validated by createGGUF
	gw.end = t.Off + size
	for gw.written < t.Off {
		if err := gw.w.WriteByte(0); err != nil {
			return t, err
		}
		gw.written++
	}
	return t, nil
}

func (gw *ggufWriter) Write(p []byte) (int, error) {
	if uint64(len(p)) > gw.end-gw.written {
		return 0, fmt.Errorf("write exceeds declared tensor size")
	}
	n, err := gw.w.Write(p)
	gw.written += uint64(n)
	return n, err
}

func (gw *ggufWriter) publish() error {
	if gw.next != len(gw.tensors) {
		return fmt.Errorf("only %d of %d tensors written", gw.next, len(gw.tensors))
	}
	if gw.written != gw.end {
		return fmt.Errorf("incomplete final tensor")
	}
	if err := gw.w.Flush(); err != nil {
		return err
	}
	if err := gw.f.Close(); err != nil {
		return err
	}
	if err := os.Rename(gw.f.Name(), gw.path); err != nil {
		return err
	}
	gw.f = nil
	return nil
}
