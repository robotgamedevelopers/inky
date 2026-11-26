package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
)

// A safetensors file: an 8-byte header length, a JSON header, then raw tensor data.
type safetensors struct {
	f       *os.File
	base    int64
	tensors map[string]stTensor
}

type stTensor struct {
	DType   string   `json:"dtype"`
	Shape   []int    `json:"shape"`
	Offsets [2]int64 `json:"data_offsets"`
}

func openSafetensors(path string) (*safetensors, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var n [8]byte
	if _, err := f.ReadAt(n[:], 0); err != nil {
		f.Close()
		return nil, err
	}
	hl := binary.LittleEndian.Uint64(n[:])
	if hl > 64<<20 {
		f.Close()
		return nil, fmt.Errorf("%s: header of %d bytes", path, hl)
	}
	hdr := make([]byte, hl)
	if _, err := f.ReadAt(hdr, 8); err != nil {
		f.Close()
		return nil, err
	}
	raw := map[string]json.RawMessage{}
	if err := json.Unmarshal(hdr, &raw); err != nil {
		f.Close()
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	s := &safetensors{f: f, base: 8 + int64(hl), tensors: map[string]stTensor{}}
	stInfo, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	fileLen := stInfo.Size()
	for k, v := range raw {
		if k == "__metadata__" {
			continue
		}
		var t stTensor
		if err := json.Unmarshal(v, &t); err != nil {
			f.Close()
			return nil, fmt.Errorf("%s: tensor %s: %w", path, k, err)
		}
		for _, d := range t.Shape {
			if d < 0 {
				f.Close()
				return nil, fmt.Errorf("%s: tensor %s: negative dimension in %v", path, k, t.Shape)
			}
		}
		if t.Offsets[0] < 0 || t.Offsets[1] < t.Offsets[0] || fileLen < s.base || t.Offsets[1] > fileLen-s.base {
			f.Close()
			return nil, fmt.Errorf("%s: tensor %s: offsets %v out of file bounds (%d bytes)", path, k, t.Offsets, fileLen)
		}
		s.tensors[k] = t
	}
	return s, nil
}

func (s *safetensors) Close() error { return s.f.Close() }

// read returns bytes [from, from+n) of a tensor's data.
func (s *safetensors) read(name string, from, n int64) ([]byte, error) {
	t, ok := s.tensors[name]
	if !ok {
		return nil, fmt.Errorf("no tensor %s", name)
	}
	size := t.Offsets[1] - t.Offsets[0]
	if from < 0 || n < 0 || from > size || n > size-from {
		return nil, fmt.Errorf("%s: read [%d, %d+%d) out of range (size %d)", name, from, from, n, size)
	}
	buf := make([]byte, n)
	_, err := s.f.ReadAt(buf, s.base+t.Offsets[0]+from)
	return buf, err
}

func (s *safetensors) expect(name, dtype string, shape ...int) (stTensor, error) {
	t, ok := s.tensors[name]
	if !ok {
		return t, fmt.Errorf("no tensor %s", name)
	}
	if t.DType != dtype || fmt.Sprint(t.Shape) != fmt.Sprint(shape) {
		return t, fmt.Errorf("%s: %s %v, want %s %v", name, t.DType, t.Shape, dtype, shape)
	}
	return t, nil
}

func (s *safetensors) float32s(name string, n int) ([]float32, error) {
	if _, err := s.expect(name, "F32", n); err != nil {
		if _, err2 := s.expect(name, "F32"); err2 == nil && n == 1 {
			// a scalar stored with an empty shape
		} else {
			return nil, err
		}
	}
	b, err := s.read(name, 0, int64(4*n))
	if err != nil {
		return nil, err
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out, nil
}
