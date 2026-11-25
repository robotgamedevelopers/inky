package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"regexp"
	"runtime"
	"sync"
)

const (
	q2kBytes = 84
	q3kBytes = 110
	q30Bytes = 14
	q4kBytes = 144
	q50Bytes = 22
)

// quantizeQ3K encodes len(x)/256 super-blocks in GGML Q3_K layout (110 bytes / 256 weights = 3.4375 bpw):
// hmask[32] (high bit), qs[64] (low 2 bits), scales[12] (16 signed 6-bit scales), d (fp16).
func quantizeQ3K(dst []byte, x []float32) {
	var l [256]int8
	var scales [16]float32
	for blk := 0; blk < len(x)/256; blk++ {
		xb := x[blk*256 : blk*256+256]
		o := dst[blk*q3kBytes : blk*q3kBytes+q3kBytes]
		for i := range o {
			o[i] = 0
		}
		var maxScale, maxAbs float32
		for ib := 0; ib < 16; ib++ {
			s := makeQxQuants(4, xb[16*ib:16*ib+16], l[16*ib:16*ib+16])
			scales[ib] = s
			if a := float32(math.Abs(float64(s))); a > maxAbs {
				maxAbs, maxScale = a, s
			}
		}
		if maxAbs < 1e-15 {
			continue
		}
		iscale := -32.0 / maxScale
		dh := f32to16(1.0 / iscale)
		binary.LittleEndian.PutUint16(o[108:], dh)
		d := f16(dh)

		var scInt [16]int
		for ib := 0; ib < 16; ib++ {
			sv := max2(-32, min2(31, nearestInt(iscale*scales[ib])))
			scInt[ib] = sv
			u := byte(sv + 32)
			j := ib & 3
			if ib < 8 {
				o[96+ib] |= u & 0x0F
			} else {
				o[96+ib-8] |= (u & 0x0F) << 4
			}
			o[96+8+j] |= ((u >> 4) & 0x03) << uint(2*(ib>>2))
		}
		for ib := 0; ib < 16; ib++ {
			dj := d * float32(scInt[ib])
			if dj == 0 {
				for ii := 0; ii < 16; ii++ {
					l[16*ib+ii] = 4
				}
				continue
			}
			for ii := 0; ii < 16; ii++ {
				q := max2(-4, min2(3, nearestInt(xb[16*ib+ii]/dj)))
				l[16*ib+ii] = int8(q + 4)
			}
		}
		hmask, qs := o[0:32], o[32:96]
		for n := 0; n < 2; n++ {
			for g := 0; g < 4; g++ {
				sh := uint(2 * g)
				m := byte(1 << uint(4*n+g))
				for hp := 0; hp < 2; hp++ {
					for i := 0; i < 16; i++ {
						v := byte(l[128*n+32*g+16*hp+i]) // in 0..7 representing q+4
						idx := 16*hp + i
						qs[32*n+idx] |= (v & 3) << sh
						if v >= 4 {
							hmask[idx] |= m
						}
					}
				}
			}
		}
	}
}

func q3kScale(scales []byte, is int) int {
	j := is & 3
	var lo byte
	if is < 8 {
		lo = scales[is] & 0x0F
	} else {
		lo = (scales[is-8] >> 4) & 0x0F
	}
	hi := (scales[8+j] >> uint(2*(is>>2))) & 0x03
	return int(lo|(hi<<4)) - 32
}

// dequantQ3K decodes len(row)/110 Q3_K super-blocks matching llama.cpp and src/kernels.metal.
func dequantQ3K(row []byte, k int) []float32 {
	out := make([]float32, k)
	for blk := 0; blk < k/256; blk++ {
		o := row[blk*q3kBytes:]
		hmask, qs, sc := o[0:32], o[32:96], o[96:108]
		d := f16(binary.LittleEndian.Uint16(o[108:]))
		y := out[blk*256:]
		for n := 0; n < 2; n++ {
			for g := 0; g < 4; g++ {
				sh := uint(2 * g)
				m := byte(1 << uint(4*n+g))
				for hp := 0; hp < 2; hp++ {
					dl := d * float32(q3kScale(sc, 8*n+2*g+hp))
					for i := 0; i < 16; i++ {
						idx := 16*hp + i
						q2 := int((qs[32*n+idx] >> sh) & 3)
						qh := 4
						if hmask[idx]&m != 0 {
							qh = 0
						}
						y[128*n+32*g+16*hp+i] = dl * float32(q2-qh)
					}
				}
			}
		}
	}
	return out
}

// quantizeQ2K encodes len(x)/256 super-blocks in GGML Q2_K layout (84 bytes / 256 weights = 2.625 bpw):
// scales[16] (low 4b scale, high 4b min), qs[64] (2-bit quants), d (fp16), dmin (fp16).
func quantizeQ2K(dst []byte, x []float32) {
	var scF [16]float32
	var minF [16]float32
	for blk := 0; blk < len(x)/256; blk++ {
		xb := x[blk*256 : blk*256+256]
		o := dst[blk*q2kBytes : blk*q2kBytes+q2kBytes]
		for i := range o {
			o[i] = 0
		}
		var maxScale, maxMin float32
		for ib := 0; ib < 16; ib++ {
			sub := xb[16*ib : 16*ib+16]
			lo, hi := sub[0], sub[0]
			for _, v := range sub[1:] {
				if v < lo {
					lo = v
				}
				if v > hi {
					hi = v
				}
			}
			if lo > 0 {
				lo = 0
			}
			m := -lo
			s := (hi - lo) / 3.0
			scF[ib] = s
			minF[ib] = m
			if s > maxScale {
				maxScale = s
			}
			if m > maxMin {
				maxMin = m
			}
		}
		var d, dmin float32
		if maxScale > 1e-15 {
			dh := f32to16(maxScale / 15.0)
			binary.LittleEndian.PutUint16(o[80:], dh)
			d = f16(dh)
		}
		if maxMin > 1e-15 {
			mh := f32to16(maxMin / 15.0)
			binary.LittleEndian.PutUint16(o[82:], mh)
			dmin = f16(mh)
		}
		for ib := 0; ib < 16; ib++ {
			var ls, lm int
			if d > 0 {
				ls = max2(0, min2(15, nearestInt(scF[ib]/d)))
			}
			if dmin > 0 {
				lm = max2(0, min2(15, nearestInt(minF[ib]/dmin)))
			}
			o[ib] = byte(ls | (lm << 4))
		}
		qs := o[16:80]
		for n := 0; n < 2; n++ {
			for g := 0; g < 4; g++ {
				sh := uint(2 * g)
				for hp := 0; hp < 2; hp++ {
					sc := o[8*n+2*g+hp]
					dl := d * float32(sc&0x0F)
					ml := dmin * float32(sc>>4)
					for i := 0; i < 16; i++ {
						var q int
						if dl > 0 {
							q = max2(0, min2(3, nearestInt((xb[128*n+32*g+16*hp+i]+ml)/dl)))
						}
						qs[32*n+16*hp+i] |= byte(q) << sh
					}
				}
			}
		}
	}
}

// dequantQ2K decodes len(row)/84 Q2_K super-blocks matching llama.cpp and src/kernels.metal.
func dequantQ2K(row []byte, k int) []float32 {
	out := make([]float32, k)
	for blk := 0; blk < k/256; blk++ {
		o := row[blk*q2kBytes:]
		sc, qs := o[0:16], o[16:80]
		d := f16(binary.LittleEndian.Uint16(o[80:]))
		dmin := f16(binary.LittleEndian.Uint16(o[82:]))
		y := out[blk*256:]
		for n := 0; n < 2; n++ {
			for g := 0; g < 4; g++ {
				sh := uint(2 * g)
				for hp := 0; hp < 2; hp++ {
					s := sc[8*n+2*g+hp]
					dl := d * float32(s&0x0F)
					ml := dmin * float32(s>>4)
					for i := 0; i < 16; i++ {
						q := (qs[32*n+16*hp+i] >> sh) & 3
						y[128*n+32*g+16*hp+i] = dl*float32(q) - ml
					}
				}
			}
		}
	}
	return out
}

// quantizeQ3_0 encodes len(x)/32 blocks in compact Q3_0 layout (14 bytes / 32 weights = 3.50 bpw):
// fp16 d (2B) + qh[4] (32 high bits) + qs[8] (32 low 2-bit quants).
func quantizeQ3_0(dst []byte, x []float32) {
	var l [32]int8
	for blk := 0; blk < len(x)/32; blk++ {
		xb := x[blk*32 : blk*32+32]
		o := dst[blk*q30Bytes : blk*q30Bytes+q30Bytes]
		for i := range o {
			o[i] = 0
		}
		s := makeQxQuants(4, xb, l[:])
		if math.Abs(float64(s)) < 1e-15 {
			continue
		}
		dh := f32to16(s)
		binary.LittleEndian.PutUint16(o[0:2], dh)
		d := f16(dh)
		if d == 0 {
			continue
		}
		var qh uint32
		for j := 0; j < 32; j++ {
			q := max2(-4, min2(3, nearestInt(xb[j]/d)))
			u := uint32(q + 4) // 0..7
			o[6+(j>>2)] |= byte((u & 3) << uint((j&3)*2))
			if u&4 != 0 {
				qh |= 1 << uint(j)
			}
		}
		binary.LittleEndian.PutUint32(o[2:6], qh)
	}
}

// dequantQ3_0 decodes len(row)/14 Q3_0 blocks matching src/kernels.metal.
func dequantQ3_0(row []byte, k int) []float32 {
	out := make([]float32, k)
	for blk := 0; blk < k/32; blk++ {
		o := row[blk*q30Bytes:]
		d := f16(binary.LittleEndian.Uint16(o[0:2]))
		qh := binary.LittleEndian.Uint32(o[2:6])
		qs := o[6:14]
		for j := 0; j < 32; j++ {
			q2 := uint32((qs[j>>2] >> uint((j&3)*2)) & 3)
			hb := ((qh >> uint(j)) & 1) << 2
			out[blk*32+j] = d * float32(int(q2|hb)-4)
		}
	}
	return out
}

func dequantQ4K(row []byte, k int) []float32 {
	out := make([]float32, k)
	for blk := 0; blk < k/256; blk++ {
		o := row[blk*q4kBytes:]
		d := f16(binary.LittleEndian.Uint16(o[0:2]))
		dmin := f16(binary.LittleEndian.Uint16(o[2:4]))
		sc := o[4:16]
		qs := o[16:144]
		y := out[blk*256:]
		for j := 0; j < 8; j++ {
			var s, m byte
			if j < 4 {
				s = sc[j] & 63
				m = sc[j+4] & 63
			} else {
				s = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4)
				m = (sc[j+4] >> 4) | ((sc[j] >> 6) << 4)
			}
			df := d * float32(s)
			mf := dmin * float32(m)
			qbase := (j / 2) * 32
			sh := uint((j & 1) * 4)
			for i := 0; i < 32; i++ {
				q := (qs[qbase+i] >> sh) & 0x0F
				y[j*32+i] = df*float32(q) - mf
			}
		}
	}
	return out
}

func dequantQ5_0(row []byte, k int) []float32 {
	out := make([]float32, k)
	for blk := 0; blk < k/32; blk++ {
		o := row[blk*q50Bytes:]
		d := f16(binary.LittleEndian.Uint16(o[0:2]))
		qh := binary.LittleEndian.Uint32(o[2:6])
		qs := o[6:22]
		for i := 0; i < 16; i++ {
			q0 := int(qs[i]&0x0F) | int(((qh>>uint(i))&1)<<4) - 16
			q1 := int(qs[i]>>4) | int(((qh>>uint(i+16))&1)<<4) - 16
			out[blk*32+i] = d * float32(q0)
			out[blk*32+16+i] = d * float32(q1)
		}
	}
	return out
}

func dequantAnyRow(typ uint32, row []byte, k int) ([]float32, error) {
	switch typ {
	case tQ8_0, tMX8:
		return dequantRow(typ, row, k, 1)
	case tQ6_K:
		return dequantQ6K(row, k), nil
	case tQ4_K:
		return dequantQ4K(row, k), nil
	case tQ5_0:
		return dequantQ5_0(row, k), nil
	case tQ3_K:
		return dequantQ3K(row, k), nil
	case tQ2_K:
		return dequantQ2K(row, k), nil
	case tQ3_0:
		return dequantQ3_0(row, k), nil
	default:
		return nil, fmt.Errorf("cannot dequantize ggml type %d", typ)
	}
}

func targetSub4Type(quant string, rowLen int) (uint32, error) {
	switch quant {
	case "q3k":
		if rowLen%256 == 0 {
			return tQ3_K, nil
		}
		if rowLen%32 == 0 {
			return tQ3_0, nil
		}
	case "q2k":
		if rowLen%256 == 0 {
			return tQ2_K, nil
		}
		if rowLen%32 == 0 {
			return tQ3_0, nil
		}
	case "q3_0":
		if rowLen%32 == 0 {
			return tQ3_0, nil
		}
	default:
		return 0, fmt.Errorf("unknown -quant %q: expected q3k, q2k, or q3_0", quant)
	}
	return 0, fmt.Errorf("row length %d is not compatible with -quant %q", rowLen, quant)
}

func quantizeRow(typ uint32, dst []byte, x []float32) error {
	switch typ {
	case tQ3_K:
		quantizeQ3K(dst, x)
	case tQ2_K:
		quantizeQ2K(dst, x)
	case tQ3_0:
		quantizeQ3_0(dst, x)
	default:
		return fmt.Errorf("unsupported target quant type %d", typ)
	}
	return nil
}

// requantGGUF re-quantizes tensors matching pattern from inPath into compact sub-4-bit
// types (Q3_K / Q2_K for K % 256 == 0, Q3_0 for K % 32 == 0) and writes outPath atomically.
func requantGGUF(inPath, outPath, quant, pattern string) error {
	if inPath == "" || outPath == "" {
		return fmt.Errorf("requant needs -in and -out")
	}
	re, err := regexp.Compile(pattern)
	if err != nil {
		return fmt.Errorf("compile -tensors %q: %w", pattern, err)
	}
	in, err := openGGUF(inPath)
	if err != nil {
		return err
	}
	defer in.Close()

	type rplan struct {
		src     tensorInfo
		dst     tensorInfo
		requant bool
	}
	var plans []rplan
	var infos []tensorInfo
	var matched int
	var inBytesTotal, outBytesTotal uint64

	for _, t := range in.tensors {
		sz, err := t.size()
		if err != nil {
			return err
		}
		inBytesTotal += sz
		if re.MatchString(t.Name) && (t.Type == tQ4_K || t.Type == tQ5_0 || t.Type == tQ6_K || t.Type == tQ8_0) {
			nt, err := targetSub4Type(quant, t.rowLen())
			if err != nil {
				return fmt.Errorf("%s: %w", t.Name, err)
			}
			dt := tensorInfo{Name: t.Name, Dims: t.Dims, Type: nt}
			dsz, err := dt.size()
			if err != nil {
				return err
			}
			outBytesTotal += dsz
			matched++
			plans = append(plans, rplan{src: t, dst: dt, requant: true})
			infos = append(infos, dt)
			continue
		}
		outBytesTotal += sz
		plans = append(plans, rplan{src: t, dst: t, requant: false})
		infos = append(infos, t)
	}
	if matched == 0 {
		return fmt.Errorf("-tensors %q matched 0 re-quantizable tensors in %s", pattern, inPath)
	}

	note := fmt.Sprintf("requantized %d tensors to %s matching %s", matched, quant, pattern)
	kv := appendKVString(append([]byte(nil), in.kvRaw...), conversionKey, note)
	w, err := createGGUF(outPath, in.nKV+1, kv, infos, in.f)
	if err != nil {
		return err
	}
	defer w.discard()

	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}

	for _, p := range plans {
		dt, err := w.beginTensor(p.dst.Name)
		if err != nil {
			return err
		}
		if !p.requant {
			if err := in.copyTensor(w, p.src); err != nil {
				return err
			}
			continue
		}
		nRows := p.src.nRows()
		k := p.src.rowLen()
		dstRB := dt.rowBytes()
		const batchRows = 256
		var relSum float64
		var relCount int
		for r0 := 0; r0 < nRows; r0 += batchRows {
			cnt := min(batchRows, nRows-r0)
			rawIn, err := in.readRows(p.src, r0, cnt)
			if err != nil {
				return err
			}
			srcRB := p.src.rowBytes()
			rawOut := make([]byte, cnt*dstRB)
			var wg sync.WaitGroup
			var firstErr error
			var errMu sync.Mutex
			chunk := (cnt + workers - 1) / workers
			for wIdx := 0; wIdx < workers; wIdx++ {
				lo := wIdx * chunk
				if lo >= cnt {
					break
				}
				hi := min(cnt, lo+chunk)
				wg.Add(1)
				go func(lo, hi int) {
					defer wg.Done()
					for r := lo; r < hi; r++ {
						xf, err := dequantAnyRow(p.src.Type, rawIn[r*srcRB:(r+1)*srcRB], k)
						if err != nil {
							errMu.Lock()
							if firstErr == nil {
								firstErr = err
							}
							errMu.Unlock()
							return
						}
						dstRow := rawOut[r*dstRB : (r+1)*dstRB]
						if err := quantizeRow(dt.Type, dstRow, xf); err != nil {
							errMu.Lock()
							if firstErr == nil {
								firstErr = err
							}
							errMu.Unlock()
							return
						}
						if r0 == 0 && r < 4 {
							dq, _ := dequantAnyRow(dt.Type, dstRow, k)
							if reVal, _, rerr := relErr(dq, xf); rerr == nil {
								errMu.Lock()
								relSum += reVal
								relCount++
								errMu.Unlock()
							}
						}
					}
				}(lo, hi)
			}
			wg.Wait()
			if firstErr != nil {
				return fmt.Errorf("%s: %w", p.dst.Name, firstErr)
			}
			if _, err := w.Write(rawOut); err != nil {
				return err
			}
		}
		meanRel := 0.0
		if relCount > 0 {
			meanRel = relSum / float64(relCount)
		}
		srcSz, _ := p.src.size()
		dstSz, _ := dt.size()
		log.Printf("%s: type %d -> %d | %d -> %d bytes (%.1f%% saved) | sample rel err %.4f",
			p.dst.Name, p.src.Type, dt.Type, srcSz, dstSz, 100.0*(1.0-float64(dstSz)/float64(srcSz)), meanRel)
	}
	if err := w.publish(); err != nil {
		return err
	}
	log.Printf("requant complete: %d tensors | total tensor payload %d -> %d bytes (%.1f%% saved)",
		matched, inBytesTotal, outBytesTotal, 100.0*(1.0-float64(outBytesTotal)/float64(inBytesTotal)))
	return nil
}
