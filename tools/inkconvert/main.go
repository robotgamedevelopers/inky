// Command inkconvert builds an Inkling 2 Mini GGUF for inkserve's erisinky branch whose routed experts keep the
// formats of the official checkpoint (thinkingmachines/Inkling-2-Mini): NVFP4 in layers 0-19, MXFP8 in 20-23.
// Everything else is copied from a base GGUF (the Q4_K_M file erisinky loads), or optionally from a Q8_0 GGUF.
//
//	inkconvert probe   -official model.safetensors -ref Q8_0.gguf
//	inkconvert convert -official model.safetensors -base Q4_K_M.gguf [-q8 Q8_0.gguf -q8-tensors REGEX] -out OUT.gguf
//	inkconvert verify  -in OUT.gguf -ref Q8_0.gguf
//
// probe checks this tool's reading of the official layout (nibble order, gate/up row order, global scale) against
// the Q8_0 GGUF, which llama.cpp converted from the same checkpoint. verify compares a converted file's experts
// with the Q8_0 file the same way. See NATIVE.md at the repository root.
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"regexp"
	"slices"
	"strings"
)

const (
	nLayers  = 24
	nExperts = 128
	hidden   = 2304
	ff       = 1152
)

func main() {
	log.SetFlags(0)
	if len(os.Args) < 2 {
		log.Fatal("usage: inkconvert probe|convert|verify|q6check|mtp [flags]  (mtp: -in mtp.safetensors -official model-00001-of-00001.safetensors -out drafter.gguf)")
	}
	fs := flag.NewFlagSet(os.Args[1], flag.ExitOnError)
	official := fs.String("official", "", "official model-00001-of-00001.safetensors")
	ref := fs.String("ref", "", "Q8_0 GGUF to compare against")
	base := fs.String("base", "", "GGUF to copy the non-expert tensors from")
	q8 := fs.String("q8", "", "optional GGUF to take the tensors matching -q8-tensors from")
	q8re := fs.String("q8-tensors", "", "regexp of tensor names to take from -q8")
	q6re := fs.String("q6-tensors", "", "regexp of tensor names to re-quantize from -q8's Q8_0 to Q6_K (checked before -q8-tensors)")
	experts := fs.String("experts", "native", "routed experts: native (NV4 / MX8 from -official) or base (copied from -base)")
	in := fs.String("in", "", "converted GGUF to verify")
	out := fs.String("out", "", "output GGUF")
	w13 := fs.String("w13", "interleaved", "official w13 row order: interleaved (gate, up, gate, ...; what probe found in the release) or stacked")
	_ = fs.Parse(os.Args[2:])
	var err error
	switch os.Args[1] {
	case "probe":
		err = probe(*official, *ref)
	case "convert":
		err = convert(*official, *base, *q8, *q8re, *q6re, *experts, *out, *w13)
	case "q6check":
		err = q6check(*base, *ref)
	case "verify":
		err = verify(*in, *ref)
	case "mtp":
		err = convertMTP(*in, *official, *out)
	default:
		err = fmt.Errorf("unknown command %q", os.Args[1])
	}
	if err != nil {
		log.Fatal(err)
	}
}

func stName(layer int, which, part string) string {
	return fmt.Sprintf("model.llm.layers.%d.mlp.experts.%s_%s", layer, which, part)
}

// expertSrc reads one layer's routed-expert projection (w13 or w2) from the official checkpoint.
type expertSrc struct {
	st     *safetensors
	layer  int
	which  string
	nvfp4  bool
	rows   int // output rows per expert
	k      int // inputs per row
	gscale []float32
}

func newExpertSrc(st *safetensors, layer int, which string) (*expertSrc, error) {
	s := &expertSrc{st: st, layer: layer, which: which, rows: hidden, k: hidden}
	if which == "w2" {
		s.k = ff
	}
	w, ok := st.tensors[stName(layer, which, "weight")]
	if !ok {
		return nil, fmt.Errorf("layer %d: no %s", layer, which)
	}
	s.nvfp4 = w.DType == "U8"
	if s.nvfp4 {
		if _, err := st.expect(stName(layer, which, "weight"), "U8", nExperts, s.rows, s.k/2); err != nil {
			return nil, err
		}
		if _, err := st.expect(stName(layer, which, "weight_scale"), "F8_E4M3", nExperts, s.rows, s.k/16); err != nil {
			return nil, err
		}
		amax, err := st.float32s(stName(layer, which, "weight_amax"), nExperts)
		if err != nil {
			return nil, err
		}
		den, err := st.float32s(stName(layer, which, "weight_outer_scale_denominator"), 1)
		if err != nil {
			return nil, err
		}
		if len(den) == 0 || math.IsNaN(float64(den[0])) || math.IsInf(float64(den[0]), 0) || den[0] == 0 {
			return nil, fmt.Errorf("layer %d %s: invalid outer scale denominator %v", layer, which, den)
		}
		s.gscale = make([]float32, nExperts)
		for e := range amax {
			gs := amax[e] / den[0]
			if math.IsNaN(float64(gs)) || math.IsInf(float64(gs), 0) {
				return nil, fmt.Errorf("layer %d %s expert %d: non-finite gscale %v", layer, which, e, gs)
			}
			s.gscale[e] = gs
		}
	} else {
		if _, err := st.expect(stName(layer, which, "weight"), "F8_E4M3", nExperts, s.rows, s.k); err != nil {
			return nil, err
		}
		if _, err := st.expect(stName(layer, which, "weight_scale"), "U8", nExperts, s.rows, s.k/32); err != nil {
			return nil, err
		}
	}
	return s, nil
}

func (s *expertSrc) codeBytes() int {
	if s.nvfp4 {
		return s.k / 2
	}
	return s.k
}

func (s *expertSrc) scaleBytes() int {
	if s.nvfp4 {
		return s.k / 16
	}
	return s.k / 32
}

// expert reads all rows of one expert: codes and scales, each rows × bytes-per-row.
func (s *expertSrc) expert(e int) (codes, scales []byte, err error) {
	cb, sb := s.codeBytes(), s.scaleBytes()
	codes, err = s.st.read(stName(s.layer, s.which, "weight"), int64(e*s.rows*cb), int64(s.rows*cb))
	if err != nil {
		return nil, nil, err
	}
	scales, err = s.st.read(stName(s.layer, s.which, "weight_scale"), int64(e*s.rows*sb), int64(s.rows*sb))
	return codes, scales, err
}

func (s *expertSrc) ggmlType() uint32 {
	if s.nvfp4 {
		return tNV4
	}
	return tMX8
}

// packRow packs official row r of expert e (codes and scales as returned by expert) into dst.
func (s *expertSrc) packRow(dst, codes, scales []byte, r int) error {
	cb, sb := s.codeBytes(), s.scaleBytes()
	c, sc := codes[r*cb:(r+1)*cb], scales[r*sb:(r+1)*sb]
	if s.nvfp4 {
		return packNV4(dst, c, sc, s.k)
	}
	return packMX8(dst, c, sc, s.k)
}

// srcRow maps a GGUF output row to the official row: GGUF's fused gate_up puts all gate rows first.
func srcRow(which string, interleaved bool, r int) int {
	if which != "w13" || !interleaved {
		return r
	}
	if r < ff {
		return 2 * r
	}
	return 2*(r-ff) + 1
}

func gname(layer int, which string) string {
	if which == "w13" {
		return fmt.Sprintf("blk.%d.ffn_gate_up_exps.weight", layer)
	}
	return fmt.Sprintf("blk.%d.ffn_down_exps.weight", layer)
}

func gscaleName(layer int, which string) string {
	if which == "w13" {
		return fmt.Sprintf("blk.%d.ffn_gate_up_exps.gscale", layer)
	}
	return fmt.Sprintf("blk.%d.ffn_down_exps.gscale", layer)
}

// dequantSrcRow decodes official row r of expert e under a nibble-order hypothesis (hiFirst swaps the nibbles).
func (s *expertSrc) dequantSrcRow(codes, scales []byte, e, r int, hiFirst bool) ([]float32, error) {
	tmp := make([]byte, s.k/32*34)
	if s.nvfp4 {
		cb := s.codeBytes()
		c := append([]byte(nil), codes[r*cb:(r+1)*cb]...)
		if hiFirst {
			for i := range c {
				c[i] = c[i]>>4 | c[i]<<4
			}
		}
		if err := packNV4(tmp, c, scales[r*s.scaleBytes():(r+1)*s.scaleBytes()], s.k); err != nil {
			return nil, err
		}
		return dequantRow(tNV4, tmp, s.k, s.gscale[e])
	}
	if err := s.packRow(tmp, codes, scales, r); err != nil {
		return nil, err
	}
	return dequantRow(tMX8, tmp, s.k, 1)
}

func probe(official, refPath string) error {
	st, err := openSafetensors(official)
	if err != nil {
		return err
	}
	defer st.Close()
	ref, err := openGGUF(refPath)
	if err != nil {
		return err
	}
	defer ref.Close()
	for _, layer := range []int{0, 19, 20, 23} {
		for _, which := range []string{"w13", "w2"} {
			s, err := newExpertSrc(st, layer, which)
			if err != nil {
				return err
			}
			rt, ok := ref.tensor(gname(layer, which))
			if !ok || rt.Type != tQ8_0 {
				return fmt.Errorf("reference has no Q8_0 %s", gname(layer, which))
			}
			for _, e := range []int{0, 77} {
				codes, scales, err := s.expert(e)
				if err != nil {
					return err
				}
				for _, rr := range []int{0, 1, ff - 1, ff, ff + 1, hidden - 1} {
					rowB, err := ref.readRows(rt, e*s.rows+rr, 1)
					if err != nil {
						return err
					}
					want, err := dequantRow(tQ8_0, rowB, s.k, 1)
					if err != nil {
						return err
					}
					line := fmt.Sprintf("L%-2d %-3s e%-3d row %-4d", layer, which, e, rr)
					for _, il := range []bool{false, true} {
						if which == "w2" && il {
							continue
						}
						for _, hi := range []bool{false, true} {
							if !s.nvfp4 && hi {
								continue
							}
							got, err := s.dequantSrcRow(codes, scales, e, srcRow(which, il, rr), hi)
							if err != nil {
								return err
							}
							rel, sc, err := relErr(got, want)
							if err != nil {
								return fmt.Errorf("%s: %w", line, err)
							}
							line += fmt.Sprintf(" | il=%-5v hi=%-5v rel %.4f fit %.4f", il, hi, rel, sc)
						}
					}
					fmt.Println(line)
				}
			}
		}
	}
	return nil
}

func convert(official, basePath, q8Path, q8re, q6re, experts, outPath, layout string) error {
	if layout != "interleaved" && layout != "stacked" {
		return fmt.Errorf("unknown -w13 layout %q: expected interleaved or stacked", layout)
	}
	interleaved := layout == "interleaved"
	if official == "" || basePath == "" || outPath == "" {
		return fmt.Errorf("convert needs -official, -base and -out")
	}
	st, err := openSafetensors(official)
	if err != nil {
		return err
	}
	defer st.Close()
	base, err := openGGUF(basePath)
	if err != nil {
		return err
	}
	defer base.Close()
	inputs := []*os.File{st.f, base.f}
	var q8 *gguf
	var re, re6 *regexp.Regexp
	if q8Path != "" {
		if q8, err = openGGUF(q8Path); err != nil {
			return err
		}
		defer q8.Close()
		inputs = append(inputs, q8.f)
		if re, err = regexp.Compile(q8re); err != nil {
			return err
		}
		if q6re != "" {
			if re6, err = regexp.Compile(q6re); err != nil {
				return err
			}
		}
	} else if q6re != "" {
		return fmt.Errorf("-q6-tensors needs -q8 (the Q8_0 source)")
	}
	if experts != "native" && experts != "base" {
		return fmt.Errorf("-experts must be native or base")
	}

	srcs := map[string]*expertSrc{}
	for layer := 0; layer < nLayers && experts == "native"; layer++ {
		for _, which := range []string{"w13", "w2"} {
			s, err := newExpertSrc(st, layer, which)
			if err != nil {
				return err
			}
			srcs[gname(layer, which)] = s
		}
	}

	// The output tensor list: the base's, with experts retyped, gscale tensors after NVFP4 experts, and the
	// -q8 matches swapped for the Q8_0 file's version.
	type plan struct {
		t    tensorInfo
		from *gguf // copy from this file (nil: generated)
		src  tensorInfo
		q6   bool // generated: src (a Q8_0 tensor of -q8) re-quantized to Q6_K
	}
	var plans []plan
	var infos []tensorInfo
	for _, t := range base.tensors {
		if weight, ok := strings.CutSuffix(t.Name, ".gscale"); ok && srcs[weight+".weight"] != nil {
			// The generated native projection supplies its own scale when needed.
			continue
		}
		if s, ok := srcs[t.Name]; ok {
			nt := tensorInfo{Name: t.Name, Dims: t.Dims, Type: s.ggmlType()}
			if nt.rowLen() != s.k || nt.nRows() != nExperts*s.rows {
				return fmt.Errorf("%s: base shape %v does not match the official %d x %d", t.Name, t.Dims, s.rows, s.k)
			}
			plans = append(plans, plan{t: nt})
			if s.nvfp4 {
				plans = append(plans, plan{t: tensorInfo{Name: gscaleName(s.layer, s.which), Dims: []uint64{nExperts}, Type: tF32}})
			}
			continue
		}
		if re6 != nil && re6.MatchString(t.Name) {
			qt, ok := q8.tensor(t.Name)
			if ok && qt.Type == tQ8_0 && qt.rowLen()%256 == 0 {
				plans = append(plans, plan{t: tensorInfo{Name: t.Name, Dims: qt.Dims, Type: tQ6_K}, src: qt, q6: true})
				continue
			}
		}
		if q8 != nil && re.MatchString(t.Name) {
			qt, ok := q8.tensor(t.Name)
			if !ok {
				return fmt.Errorf("%s: not in %s", t.Name, q8Path)
			}
			plans = append(plans, plan{t: tensorInfo{Name: t.Name, Dims: qt.Dims, Type: qt.Type}, from: q8, src: qt})
			continue
		}
		plans = append(plans, plan{t: tensorInfo{Name: t.Name, Dims: t.Dims, Type: t.Type}, from: base, src: t})
	}
	for _, p := range plans {
		infos = append(infos, p.t)
	}
	note := "routed experts copied from the base file"
	if experts == "native" {
		note = fmt.Sprintf("routed experts from thinkingmachines/Inkling-2-Mini@b85c3cfa: NV4 (NVFP4) or MX8 (MXFP8), w13 %s",
			layout)
	}
	if re6 != nil {
		note += "; Q6_K (from Q8_0) tensors matching " + q6re
	}
	if q8 != nil {
		note += "; Q8_0 tensors matching " + q8re
	}
	kv := appendKVString(append([]byte(nil), base.kvRaw...), conversionKey, note)
	w, err := createGGUF(outPath, base.nKV+1, kv, infos, inputs...)
	if err != nil {
		return err
	}
	defer w.discard()
	for i, p := range plans {
		t, err := w.beginTensor(p.t.Name)
		if err != nil {
			return err
		}
		switch {
		case p.q6:
			k := p.src.rowLen()
			out := make([]byte, t.rowBytes())
			for r := 0; r < p.src.nRows(); r++ {
				b, err := q8.readRows(p.src, r, 1)
				if err != nil {
					return err
				}
				x, err := dequantRow(tQ8_0, b, k, 1)
				if err != nil {
					return err
				}
				quantizeQ6K(out, x)
				if _, err := w.Write(out); err != nil {
					return err
				}
			}
			log.Printf("%s: Q6_K from Q8_0", p.t.Name)
		case p.from != nil:
			if err := p.from.copyTensor(w, p.src); err != nil {
				return err
			}
		case p.t.Type == tF32: // a gscale tensor: the previous plan is its expert tensor
			s := srcs[plans[i-1].t.Name]
			buf := make([]byte, 0, 4*nExperts)
			for _, g := range s.gscale {
				buf = appendF32(buf, g)
			}
			if _, err := w.Write(buf); err != nil {
				return err
			}
		default:
			s := srcs[p.t.Name]
			row := make([]byte, t.rowBytes())
			for e := 0; e < nExperts; e++ {
				codes, scales, err := s.expert(e)
				if err != nil {
					return err
				}
				for r := 0; r < s.rows; r++ {
					if err := s.packRow(row, codes, scales, srcRow(s.which, interleaved, r)); err != nil {
						return fmt.Errorf("%s expert %d row %d: %w", p.t.Name, e, r, err)
					}
					if _, err := w.Write(row); err != nil {
						return err
					}
				}
			}
			log.Printf("%s: %s", p.t.Name, map[uint32]string{tNV4: "NV4", tMX8: "MX8"}[p.t.Type])
		}
	}
	return w.publish()
}

func appendF32(b []byte, f float32) []byte {
	u := math.Float32bits(f)
	return append(b, byte(u), byte(u>>8), byte(u>>16), byte(u>>24))
}

func verify(inPath, refPath string) error {
	in, err := openGGUF(inPath)
	if err != nil {
		return err
	}
	defer in.Close()
	ref, err := openGGUF(refPath)
	if err != nil {
		return err
	}
	defer ref.Close()
	worst := 0.0
	for layer := 0; layer < nLayers; layer++ {
		for _, which := range []string{"w13", "w2"} {
			t, ok := in.tensor(gname(layer, which))
			if !ok {
				return fmt.Errorf("no %s", gname(layer, which))
			}
			rt, ok := ref.tensor(t.Name)
			width, rows := ff, hidden
			if which == "w13" {
				width, rows = hidden, 2*ff
			}
			if !ok || rt.Type != tQ8_0 || !slices.Equal(rt.Dims, []uint64{uint64(width), uint64(rows), nExperts}) {
				return fmt.Errorf("%s: expected Q8_0 expert tensor with shape [%d %d %d] in reference", t.Name, width, rows, nExperts)
			}
			if !slices.Equal(t.Dims, rt.Dims) {
				return fmt.Errorf("%s: candidate shape %v differs from reference %v", t.Name, t.Dims, rt.Dims)
			}
			var gs []float32
			if t.Type == tNV4 {
				g, ok := in.tensor(gscaleName(layer, which))
				if !ok || g.Type != tF32 || !slices.Equal(g.Dims, []uint64{nExperts}) {
					return fmt.Errorf("%s: expected %d F32 expert scales", t.Name, nExperts)
				}
				b, err := in.readRows(g, 0, 1)
				if err != nil {
					return err
				}
				for i := 0; i < nExperts; i++ {
					gs = append(gs, math.Float32frombits(uint32(b[4*i])|uint32(b[4*i+1])<<8|uint32(b[4*i+2])<<16|uint32(b[4*i+3])<<24))
				}
			}
			sum, n := 0.0, 0
			for _, e := range []int{0, 31, 64, 127} {
				for _, r := range []int{0, 7, ff - 1, ff, rows - 1} {
					a, err := in.readRows(t, e*rows+r, 1)
					if err != nil {
						return err
					}
					b, err := ref.readRows(rt, e*rows+r, 1)
					if err != nil {
						return err
					}
					g := float32(1)
					if gs != nil {
						g = gs[e]
					}
					got, err := dequantRow(t.Type, a, t.rowLen(), g)
					if err != nil {
						return err
					}
					want, err := dequantRow(rt.Type, b, rt.rowLen(), 1)
					if err != nil {
						return err
					}
					rel, _, err := relErr(got, want)
					if err != nil {
						return fmt.Errorf("%s expert %d row %d: %w", t.Name, e, r, err)
					}
					sum += rel
					n++
					worst = max(worst, rel)
				}
			}
			fmt.Printf("L%-2d %-3s %s: mean rel err vs Q8_0 %.4f\n", layer, which, map[uint32]string{tNV4: "NV4", tMX8: "MX8"}[t.Type], sum/float64(n))
		}
	}
	fmt.Printf("worst row: %.4f\n", worst)
	if worst > 0.05 {
		return fmt.Errorf("a row differs from Q8_0 by %.1f%%: the conversion is wrong", 100*worst)
	}
	return nil
}
