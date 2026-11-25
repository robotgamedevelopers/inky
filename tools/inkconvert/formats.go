package main

import (
	"encoding/binary"
	"fmt"
	"math"
)

func float32frombits(u uint32) float32 { return math.Float32frombits(u) }

// fp4 decodes an FP4 E2M1 nibble: sign, 2 exponent bits, 1 mantissa bit.
func fp4(n byte) float32 {
	v := [8]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6}[n&7]
	if n&8 != 0 {
		return -v
	}
	return v
}

// e4m3 decodes an FP8 E4M3 (fn) byte: bias 7, no infinities, 0x7F/0xFF are NaN.
func e4m3(b byte) float32 {
	e, m := int(b>>3&15), float64(b&7)
	var v float64
	if e == 0 {
		v = m / 8 * math.Ldexp(1, -6)
	} else {
		v = (1 + m/8) * math.Ldexp(1, e-7)
	}
	if b&0x7F == 0x7F {
		v = math.NaN()
	}
	if b&0x80 != 0 {
		v = -v
	}
	return float32(v)
}

func f16bits(f float32) uint16 {
	// Only exact powers of two in the normal or subnormal half range reach this (asserted by the caller).
	b := math.Float32bits(f)
	exp := int(b>>23&0xFF) - 127
	if f == 0 {
		return 0
	}
	if exp >= -14 {
		return uint16(exp+15) << 10
	}
	return uint16(1) << uint(exp+24) // subnormal: 2^exp = 2^-24 * 2^(exp+24)
}

func f16(h uint16) float32 {
	e, m := int(h>>10&31), float64(h&1023)
	var v float64
	if e == 0 {
		v = m / 1024 * math.Ldexp(1, -14)
	} else if e == 31 {
		v = math.Inf(1)
		if m != 0 {
			v = math.NaN()
		}
	} else {
		v = (1 + m/1024) * math.Ldexp(1, e-15)
	}
	if h&0x8000 != 0 {
		v = -v
	}
	return float32(v)
}

// The official checkpoint's NVFP4 rows: K/2 bytes with element 2i in the low nibble of byte i, plus K/16 E4M3
// scales. A weight is fp4 * scale * gscale, with gscale = amax / outer_scale_denominator per expert.

// nvfp4Elem returns element i of an official NVFP4 row.
func nvfp4Elem(row []byte, i int) byte {
	b := row[i/2]
	if i%2 == 0 {
		return b & 15
	}
	return b >> 4
}

// packNV4 converts one official NVFP4 row (codes and scales) into NV4 blocks: per 32 weights, 2 scale bytes,
// then 16 bytes whose low nibble j is weight j and high nibble is weight j+16 (Q4_0's order).
func packNV4(dst, codes, scales []byte, k int) error {
	if k < 0 || k%32 != 0 {
		return fmt.Errorf("NVFP4 row length %d is not a non-negative multiple of 32", k)
	}
	if len(dst) < (k/32)*18 || len(codes) < k/2 || len(scales) < k/16 {
		return fmt.Errorf("NVFP4 buffer too short for k=%d", k)
	}
	for b := 0; b < k/32; b++ {
		s0, s1 := scales[2*b], scales[2*b+1]
		if s0&0x7F == 0x7F {
			return fmt.Errorf("NVFP4 scale %#02x is E4M3 NaN at block %d", s0, b)
		}
		if s1&0x7F == 0x7F {
			return fmt.Errorf("NVFP4 scale %#02x is E4M3 NaN at block %d", s1, b)
		}
		o := dst[b*18 : b*18+18]
		o[0], o[1] = s0, s1
		for j := 0; j < 16; j++ {
			o[2+j] = nvfp4Elem(codes, 32*b+j) | nvfp4Elem(codes, 32*b+16+j)<<4
		}
	}
	return nil
}

// packMX8 converts one official MXFP8 row (K E4M3 bytes, K/32 E8M0 scales) into MX8 blocks: an fp16 scale
// (the exact power of two) followed by the 32 E4M3 bytes unchanged.
func packMX8(dst, vals, scales []byte, k int) error {
	if k < 0 || k%32 != 0 {
		return fmt.Errorf("MXFP8 row length %d is not a non-negative multiple of 32", k)
	}
	if len(dst) < (k/32)*34 || len(vals) < k || len(scales) < k/32 {
		return fmt.Errorf("MXFP8 buffer too short for k=%d", k)
	}
	for b := 0; b < k/32; b++ {
		if scales[b] == 0xFF {
			return fmt.Errorf("MXFP8 scale 0xff is E8M0 NaN at block %d", b)
		}
		e := int(scales[b]) - 127
		if e < -24 || e > 15 {
			return fmt.Errorf("MXFP8 scale 2^%d does not fit fp16", e)
		}
		blockVals := vals[32*b : 32*b+32]
		for _, v := range blockVals {
			if v&0x7F == 0x7F {
				return fmt.Errorf("MXFP8 input value %#02x is E4M3 NaN at block %d", v, b)
			}
		}
		h := f16bits(float32(math.Ldexp(1, e)))
		if h&0x7C00 == 0x7C00 {
			return fmt.Errorf("MXFP8 encoded fp16 scale %#04x is non-finite", h)
		}
		o := dst[b*34 : b*34+34]
		binary.LittleEndian.PutUint16(o, h)
		copy(o[2:], blockVals)
	}
	return nil
}

// dequantRow expands one row of a GGUF tensor (types this tool writes or verifies against) to float32.
// gscale multiplies NV4 rows; pass 1 for other types.
func dequantRow(typ uint32, row []byte, k int, gscale float32) ([]float32, error) {
	if math.IsNaN(float64(gscale)) || math.IsInf(float64(gscale), 0) {
		return nil, fmt.Errorf("non-finite gscale %v", gscale)
	}
	out := make([]float32, k)
	switch typ {
	case tQ8_0:
		for b := 0; b < k/32; b++ {
			o := row[b*34:]
			d := f16(binary.LittleEndian.Uint16(o))
			for j := 0; j < 32; j++ {
				out[32*b+j] = d * float32(int8(o[2+j]))
			}
		}
	case tNV4:
		for b := 0; b < k/32; b++ {
			o := row[b*18:]
			s0, s1 := e4m3(o[0])*gscale, e4m3(o[1])*gscale
			for j := 0; j < 16; j++ {
				out[32*b+j] = fp4(o[2+j]&15) * s0
				out[32*b+16+j] = fp4(o[2+j]>>4) * s1
			}
		}
	case tMX8:
		for b := 0; b < k/32; b++ {
			o := row[b*34:]
			d := f16(binary.LittleEndian.Uint16(o))
			for j := 0; j < 32; j++ {
				out[32*b+j] = e4m3(o[2+j]) * d
			}
		}
	default:
		return nil, fmt.Errorf("dequant of type %d not implemented", typ)
	}
	return out, nil
}

// relErr returns ||a-b|| / ||b|| and the best least-squares scale s minimizing ||s*a - b||.
func relErr(a, b []float32) (rel, scale float64, err error) {
	if len(a) != len(b) {
		return 0, 0, fmt.Errorf("row widths differ: %d and %d", len(a), len(b))
	}
	var ab, aa, bb, dd float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		ab += x * y
		aa += x * x
		bb += y * y
		dd += (x - y) * (x - y)
	}
	// A finite float32 row cannot overflow these float64 sums. Any nonfinite
	// component produces an infinite or NaN squared difference, including Inf - Inf.
	if math.IsNaN(dd) || math.IsInf(dd, 0) {
		return 0, 0, fmt.Errorf("row contains a nonfinite weight")
	}
	if bb == 0 {
		if dd == 0 {
			return 0, 0, nil
		}
		return 0, 0, fmt.Errorf("nonzero row differs from all-zero reference")
	}
	if aa > 0 {
		scale = ab / aa
	}
	return math.Sqrt(dd / bb), scale, nil
}
