package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

// GGUF v3, only as much as this tool needs: the key-value section is kept as raw bytes (and can be appended to),
// tensors are described by name, dims, type and offset.

const ggufAlign = 32 // general.alignment's default; the Inkling files don't set it

// GGML type ids. NV4 and MX8 are private to inkserve's erisinky branch; llama.cpp can't load them.
const (
	tF32  = 0
	tQ5_0 = 6
	tQ8_0 = 8
	tQ4_K = 12
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
	case tBF16:
		return 2, 1, nil
	case tQ5_0:
		return 22, 32, nil
	case tQ8_0:
		return 34, 32, nil
	case tQ4_K:
		return 144, 256, nil
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
	if t.rowLen()%bw != 0 {
		return 0, fmt.Errorf("%s: row length %d is not a multiple of %d", t.Name, t.rowLen(), bw)
	}
	return uint64(t.nRows()) * uint64(t.rowLen()/bw*bs), nil
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
	for i := uint64(0); i < hdr.NKV; i++ {
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
	}
	kvEnd := cr.n
	g.kvRaw = make([]byte, kvEnd-kvStart)
	if _, err := g.f.ReadAt(g.kvRaw, kvStart); err != nil {
		return err
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
	rb := t.rowBytes()
	buf := make([]byte, n*rb)
	_, err := g.f.ReadAt(buf, g.dataBase+int64(t.Off)+int64(r0)*int64(rb))
	return buf, err
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
	tensors []tensorInfo
	next    int
	written uint64
}

func createGGUF(path string, nKV uint64, kvRaw []byte, tensors []tensorInfo) (*ggufWriter, error) {
	off := uint64(0)
	for i := range tensors {
		sz, err := tensors[i].size()
		if err != nil {
			return nil, err
		}
		tensors[i].Off = off
		off = (off + sz + ggufAlign - 1) / ggufAlign * ggufAlign
	}
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	w := bufio.NewWriterSize(f, 8<<20)
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
	if _, err := w.Write(hdr); err != nil {
		f.Close()
		return nil, err
	}
	return &ggufWriter{w: w, f: f, tensors: tensors}, nil
}

// beginTensor pads to the next tensor's offset and returns it; the caller then writes exactly its size.
func (gw *ggufWriter) beginTensor(name string) (tensorInfo, error) {
	if gw.next >= len(gw.tensors) || gw.tensors[gw.next].Name != name {
		return tensorInfo{}, fmt.Errorf("tensor %s written out of order", name)
	}
	t := gw.tensors[gw.next]
	gw.next++
	for gw.written < t.Off {
		if err := gw.w.WriteByte(0); err != nil {
			return t, err
		}
		gw.written++
	}
	return t, nil
}

func (gw *ggufWriter) Write(p []byte) (int, error) {
	n, err := gw.w.Write(p)
	gw.written += uint64(n)
	return n, err
}

func (gw *ggufWriter) Close() error {
	if gw.next != len(gw.tensors) {
		gw.f.Close()
		return fmt.Errorf("only %d of %d tensors written", gw.next, len(gw.tensors))
	}
	if err := gw.w.Flush(); err != nil {
		gw.f.Close()
		return err
	}
	return gw.f.Close()
}
