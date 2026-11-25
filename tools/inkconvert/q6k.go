package main

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Q6_K, as llama.cpp defines it: 256 weights per block, 16 sub-blocks of 16 with an int8 scale each, one fp16 scale.
// Layout (210 bytes): ql[128] (low 4 bits), qh[64] (high 2 bits), scales[16] (int8), d (fp16).
// The encoder follows llama.cpp's quantize_row_q6_K_ref (ggml-quants.c), including make_qx_quants' scale search,
// so a tensor re-quantized here matches what llama-quantize would make from the same input.

const q6kBytes = 210

func nearestInt(f float32) int { return int(math.RoundToEven(float64(f))) }

// makeQxQuants is llama.cpp's make_qx_quants with rmse_type 1 (weights x^2) and no importance matrix.
func makeQxQuants(nmax int, x []float32, l []int8) float32 {
	var max, amax float32
	for _, v := range x {
		if a := float32(math.Abs(float64(v))); a > amax {
			amax, max = a, v
		}
	}
	if amax < 1e-15 {
		for i := range l {
			l[i] = 0
		}
		return 0
	}
	clamp := func(q int) int { return max2(-nmax, min2(nmax-1, q)) }
	iscale := -float32(nmax) / max
	var sumlx, suml2 float32
	for i, v := range x {
		q := clamp(nearestInt(iscale * v))
		l[i] = int8(q + nmax)
		w := v * v
		sumlx += w * v * float32(q)
		suml2 += w * float32(q*q)
	}
	scale := float32(0)
	if suml2 != 0 {
		scale = sumlx / suml2
	}
	best := scale * sumlx
	for is := -9; is <= 9; is++ {
		if is == 0 {
			continue
		}
		iscale = -(float32(nmax) + 0.1*float32(is)) / max
		sumlx, suml2 = 0, 0
		for _, v := range x {
			q := clamp(nearestInt(iscale * v))
			w := v * v
			sumlx += w * v * float32(q)
			suml2 += w * float32(q*q)
		}
		if suml2 > 0 && sumlx*sumlx > best*suml2 {
			for i, v := range x {
				l[i] = int8(nmax + clamp(nearestInt(iscale*v)))
			}
			scale = sumlx / suml2
			best = scale * sumlx
		}
	}
	return scale
}

func max2(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min2(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// f32to16 rounds a float32 to the nearest fp16 (ties to even), as GGML_FP32_TO_FP16 does for finite values.
func f32to16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int(b>>23&0xFF) - 127 + 15
	man := b & 0x7FFFFF
	if exp >= 31 {
		return sign | 0x7C00
	}
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		man |= 0x800000
		shift := uint(14 - exp)
		h := man >> shift
		rem := man & (1<<shift - 1)
		half := uint32(1) << (shift - 1)
		if rem > half || (rem == half && h&1 == 1) {
			h++
		}
		return sign | uint16(h)
	}
	h := uint32(exp)<<10 | man>>13
	rem := man & 0x1FFF
	if rem > 0x1000 || (rem == 0x1000 && h&1 == 1) {
		h++
	}
	return sign | uint16(h)
}

// quantizeQ6K encodes len(x)/256 blocks into dst.
func quantizeQ6K(dst []byte, x []float32) {
	var l [256]int8
	var scales [16]float32
	for blk := 0; blk < len(x)/256; blk++ {
		xb := x[blk*256 : blk*256+256]
		o := dst[blk*q6kBytes : blk*q6kBytes+q6kBytes]
		var maxScale, maxAbs float32
		for ib := 0; ib < 16; ib++ {
			s := makeQxQuants(32, xb[16*ib:16*ib+16], l[16*ib:16*ib+16])
			scales[ib] = s
			if a := float32(math.Abs(float64(s))); a > maxAbs {
				maxAbs, maxScale = a, s
			}
		}
		for i := range o {
			o[i] = 0
		}
		if maxAbs < 1e-15 {
			continue
		}
		iscale := -128 / maxScale
		dh := f32to16(1 / iscale)
		binary.LittleEndian.PutUint16(o[208:], dh)
		sc := o[192:208]
		for ib := 0; ib < 16; ib++ {
			sc[ib] = byte(int8(min2(127, nearestInt(iscale*scales[ib]))))
		}
		d := f16(dh)
		for j := 0; j < 16; j++ {
			dj := d * float32(int8(sc[j]))
			if dj == 0 {
				continue
			}
			for ii := 0; ii < 16; ii++ {
				q := max2(-32, min2(31, nearestInt(xb[16*j+ii]/dj)))
				l[16*j+ii] = int8(q + 32)
			}
		}
		ql, qh := o[0:128], o[128:192]
		for j := 0; j < 256; j += 128 {
			for i := 0; i < 32; i++ {
				q1, q2, q3, q4 := byte(l[j+i]), byte(l[j+i+32]), byte(l[j+i+64]), byte(l[j+i+96])
				ql[j/2+i] = q1&15 | (q3&15)<<4
				ql[j/2+i+32] = q2&15 | (q4&15)<<4
				qh[j/4+i] = q1>>4 | (q2>>4)<<2 | (q3>>4)<<4 | (q4>>4)<<6
			}
		}
	}
}

// dequantQ6K decodes len(row)/210 blocks, as llama.cpp's dequantize_row_q6_K.
func dequantQ6K(row []byte, k int) []float32 {
	out := make([]float32, k)
	for blk := 0; blk < k/256; blk++ {
		o := row[blk*q6kBytes:]
		d := f16(binary.LittleEndian.Uint16(o[208:]))
		y := out[blk*256:]
		for n := 0; n < 2; n++ {
			ql, qh, sc := o[64*n:], o[128+32*n:], o[192+8*n:]
			for i := 0; i < 32; i++ {
				is := i / 16
				q1 := int(ql[i]&15|(qh[i]&3)<<4) - 32
				q2 := int(ql[i+32]&15|(qh[i]>>2&3)<<4) - 32
				q3 := int(ql[i]>>4|(qh[i]>>4&3)<<4) - 32
				q4 := int(ql[i+32]>>4|(qh[i]>>6&3)<<4) - 32
				y[128*n+i] = d * float32(int8(sc[is])) * float32(q1)
				y[128*n+i+32] = d * float32(int8(sc[is+2])) * float32(q2)
				y[128*n+i+64] = d * float32(int8(sc[is+4])) * float32(q3)
				y[128*n+i+96] = d * float32(int8(sc[is+6])) * float32(q4)
			}
		}
	}
	return out
}

// q6check validates the Q6_K code against llama.cpp's own output: the decoder on the base file's Q6_K tensors
// (llama.cpp-made) must land near Q8_0, and the encoder on Q8_0 rows must land about as near.
func q6check(basePath, refPath string) error {
	base, err := openGGUF(basePath)
	if err != nil {
		return err
	}
	defer base.Close()
	ref, err := openGGUF(refPath)
	if err != nil {
		return err
	}
	defer ref.Close()
	for _, name := range []string{"output.weight", "blk.0.attn_v.weight", "blk.23.attn_v.weight"} {
		bt, ok1 := base.tensor(name)
		rt, ok2 := ref.tensor(name)
		if !ok1 || !ok2 || rt.Type != tQ8_0 {
			return fmt.Errorf("%s: missing, or the reference isn't Q8_0", name)
		}
		if !sameDims(bt.Dims, rt.Dims) || rt.rowLen() < 256 || rt.rowLen()%256 != 0 || bt.nRows() < 1 {
			return fmt.Errorf("%s: incompatible dimensions base=%v ref=%v", name, bt.Dims, rt.Dims)
		}
		var relLC, relOurs float64
		n := 0
		for _, r := range []int{0, min(1, bt.nRows()-1), min(17, bt.nRows()-1), min(333, bt.nRows()-1), bt.nRows() - 1} {
			rb, err := ref.readRows(rt, r, 1)
			if err != nil {
				return err
			}
			want, err := dequantRow(tQ8_0, rb, rt.rowLen(), 1)
			if err != nil {
				return err
			}
			ours := make([]byte, rt.rowLen()/256*q6kBytes)
			quantizeQ6K(ours, want)
			e2, _ := relErr(dequantQ6K(ours, rt.rowLen()), want)
			relOurs += e2
			if bt.Type == tQ6_K {
				bb, err := base.readRows(bt, r, 1)
				if err != nil {
					return err
				}
				e1, _ := relErr(dequantQ6K(bb, bt.rowLen()), want)
				relLC += e1
			}
			n++
		}
		lc := "n/a (base tensor is not Q6_K)"
		if bt.Type == tQ6_K {
			lc = fmt.Sprintf("%.4f", relLC/float64(n))
		}
		fmt.Printf("%-22s llama.cpp Q6_K vs Q8_0: %s | ours (from Q8_0) vs Q8_0: %.4f\n", name, lc, relOurs/float64(n))
	}
	return nil
}
