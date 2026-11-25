package main

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestE4M3(t *testing.T) {
	cases := map[byte]float32{
		0x00: 0, 0x38: 1, 0xB8: -1, 0x7E: 448, 0x01: 1.0 / 512, 0x08: 1.0 / 64, 0x3C: 1.5,
	}
	for b, want := range cases {
		if got := e4m3(b); got != want {
			t.Errorf("e4m3(%#x) = %v, want %v", b, got, want)
		}
	}
	if !math.IsNaN(float64(e4m3(0x7F))) {
		t.Error("0x7F should be NaN")
	}
}

func TestFP4(t *testing.T) {
	want := []float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}
	for n, w := range want {
		if got := fp4(byte(n)); got != w {
			t.Errorf("fp4(%d) = %v, want %v", n, got, w)
		}
	}
}

func TestF16PowersOfTwo(t *testing.T) {
	for e := -24; e <= 15; e++ {
		v := float32(math.Ldexp(1, e))
		if got := f16(f16bits(v)); got != v {
			t.Errorf("2^%d round-trips to %v", e, got)
		}
	}
}

// The packed NV4 block must decode to exactly fp4(code) * scale * gscale, element by element.
func TestPackNV4(t *testing.T) {
	const k = 64
	codes := make([]byte, k/2)
	for i := range codes {
		codes[i] = byte(i*37 + 11)
	}
	scales := make([]byte, k/16)
	for i := range scales {
		scales[i] = byte(0x30 + 5*i)
	}
	dst := make([]byte, k/32*18)
	if err := packNV4(dst, codes, scales, k); err != nil {
		t.Fatal(err)
	}
	for b := 0; b < k/32; b++ {
		o := dst[b*18 : b*18+18]
		if o[0]&0x7F == 0x7F || o[1]&0x7F == 0x7F {
			t.Fatalf("block %d emitted E4M3 NaN scale byte: %#02x, %#02x", b, o[0], o[1])
		}
	}
	got, err := dequantRow(tNV4, dst, k, 0.25)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < k; i++ {
		want := fp4(nvfp4Elem(codes, i)) * e4m3(scales[i/16]) * 0.25
		if got[i] != want {
			t.Fatalf("element %d: %v, want %v", i, got[i], want)
		}
	}
	for _, nanByte := range []byte{0x7F, 0xFF} {
		badScales := append([]byte(nil), scales...)
		badScales[2] = nanByte
		if err := packNV4(dst, codes, badScales, k); err == nil {
			t.Fatalf("expected E4M3 NaN scale %#02x to be rejected by packNV4", nanByte)
		}
	}
}

func TestPackMX8(t *testing.T) {
	const k = 64
	vals := make([]byte, k)
	for i := range vals {
		vals[i] = byte(i * 3)
	}
	scales := []byte{127 - 10, 127 + 3}
	dst := make([]byte, k/32*34)
	if err := packMX8(dst, vals, scales, k); err != nil {
		t.Fatal(err)
	}
	got, err := dequantRow(tMX8, dst, k, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < k; i++ {
		want := e4m3(vals[i]) * float32(math.Ldexp(1, int(scales[i/32])-127))
		if got[i] != want {
			t.Fatalf("element %d: %v, want %v", i, got[i], want)
		}
	}
	if err := packMX8(dst, vals, []byte{0, 127}, k); err == nil {
		t.Error("a scale of 2^-127 should be rejected")
	}
	if err := packMX8(dst, vals, []byte{0xFF, 127}, k); err == nil {
		t.Error("E8M0 NaN scale 0xFF should be rejected")
	}
	for _, nanByte := range []byte{0x7F, 0xFF} {
		badVals := append([]byte(nil), vals...)
		badVals[33] = nanByte
		if err := packMX8(dst, badVals, scales, k); err == nil {
			t.Fatalf("expected E4M3 NaN value %#02x to be rejected by packMX8", nanByte)
		}
	}
}

func TestDequantRowRejectsNonFiniteGscale(t *testing.T) {
	dst := make([]byte, 18)
	for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if _, err := dequantRow(tNV4, dst, 32, bad); err == nil {
			t.Fatalf("expected non-finite gscale %v to be rejected", bad)
		}
	}
}

func TestF32to16(t *testing.T) {
	for _, h := range []uint16{0x0000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x3C01, 0x7BFF, 0x8001, 0xBC00} {
		if got := f32to16(f16(h)); got != h {
			t.Errorf("fp16 %#04x round-trips to %#04x", h, got)
		}
	}
}

// A Q6_K round trip on smooth data must stay within a few percent, and the decoder must invert the packing.
func TestQ6KRoundTrip(t *testing.T) {
	x := make([]float32, 512)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.37) * (1 + float64(i%16)/8))
	}
	buf := make([]byte, 2*q6kBytes)
	quantizeQ6K(buf, x)
	rel, _ := relErr(dequantQ6K(buf, len(x)), x)
	if rel > 0.03 {
		t.Errorf("Q6_K round trip error %.4f", rel)
	}
}

func TestQuantizeQ8_0NeverWrapsInt8AndValidatesInputs(t *testing.T) {
	// Subnormal scale case: d = amax/127 rounds down to h=0x0001 (2^-24), so amax/f16(h) ~= 189 > 127.5.
	// Without clamping, int8(math.RoundToEven(v * inv)) wrapped 189 -> -67 (flipping the sign of the peak weight!).
	subStep := float32(math.Ldexp(1, -24))
	peakPos := 127 * 1.49 * subStep
	row := make([]float32, 32)
	row[0] = peakPos
	row[1] = -peakPos
	row[2] = 1e-45 // tiny subnormal in a normal/subnormal block

	enc, err := quantizeQ8_0(row, 32)
	if err != nil {
		t.Fatalf("quantizeQ8_0 subnormal: %v", err)
	}
	if got0, got1 := int8(enc[2]), int8(enc[3]); got0 != 127 || got1 != -127 {
		t.Fatalf("expected clamped (+127, -127), got (%d, %d)", got0, got1)
	}

	// Ultra-tiny subnormal where f32to16(amax/127) underflows to 0 must emit 0s instead of dividing by 0 -> -128!
	tiny := make([]float32, 32)
	tiny[0] = float32(math.Ldexp(1, -30))
	encTiny, err := quantizeQ8_0(tiny, 32)
	if err != nil {
		t.Fatalf("quantizeQ8_0 underflow: %v", err)
	}
	for i := 2; i < 34; i++ {
		if int8(encTiny[i]) != 0 {
			t.Fatalf("expected 0 on fp16 scale underflow, got %d at byte %d", int8(encTiny[i]), i)
		}
	}

	// Invalid dimensions, NaN/Inf, and fp16 scale overflow must fail cleanly.
	if _, err := quantizeQ8_0(make([]float32, 30), 30); err == nil {
		t.Fatal("expected non-multiple-of-32 k to fail")
	}
	if _, err := quantizeQ8_0(make([]float32, 48), 32); err == nil {
		t.Fatal("expected len(x)%k != 0 to fail")
	}
	nanRow := make([]float32, 32)
	nanRow[7] = float32(math.NaN())
	if _, err := quantizeQ8_0(nanRow, 32); err == nil {
		t.Fatal("expected NaN weight to fail")
	}
	hugeRow := make([]float32, 32)
	hugeRow[0] = 1e7 // amax/127 > 65504 -> fp16 Inf
	if _, err := quantizeQ8_0(hugeRow, 32); err == nil {
		t.Fatal("expected fp16 scale overflow to fail")
	}
}

func localTestDir(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp(".", ".tmp-inkconvert-test-*")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	return dir
}

func writeTestSafetensors(t *testing.T, path string, arch archSpec) {
	t.Helper()
	type entry struct {
		dtype string
		shape []int
		data  []byte
	}
	var names []string
	byName := map[string]entry{}
	put := func(name, dtype string, shape []int, data []byte) {
		names = append(names, name)
		byName[name] = entry{dtype: dtype, shape: shape, data: data}
	}
	for l := 0; l < arch.nLayers; l++ {
		nvfp4 := l == 0
		for _, which := range []string{"w13", "w2"} {
			k := arch.hidden
			if which == "w2" {
				k = arch.ff
			}
			if nvfp4 {
				codes := make([]byte, arch.nExperts*arch.hidden*(k/2))
				for i := range codes {
					codes[i] = 0x22 // fp4(2)=1.0 on both nibbles
				}
				scales := make([]byte, arch.nExperts*arch.hidden*(k/16))
				for i := range scales {
					scales[i] = 0x38 // e4m3 = 1.0
				}
				var amax []byte
				for e := 0; e < arch.nExperts; e++ {
					amax = appendF32(amax, 0.5)
				}
				den := appendF32(nil, 1.0)
				put(stName(l, which, "weight"), "U8", []int{arch.nExperts, arch.hidden, k / 2}, codes)
				put(stName(l, which, "weight_scale"), "F8_E4M3", []int{arch.nExperts, arch.hidden, k / 16}, scales)
				put(stName(l, which, "weight_amax"), "F32", []int{arch.nExperts}, amax)
				put(stName(l, which, "weight_outer_scale_denominator"), "F32", []int{1}, den)
			} else {
				vals := make([]byte, arch.nExperts*arch.hidden*k)
				for i := range vals {
					vals[i] = 0x38 // e4m3 = 1.0
				}
				scales := make([]byte, arch.nExperts*arch.hidden*(k/32))
				for i := range scales {
					scales[i] = 126 // 2^-1 = 0.5
				}
				put(stName(l, which, "weight"), "F8_E4M3", []int{arch.nExperts, arch.hidden, k}, vals)
				put(stName(l, which, "weight_scale"), "U8", []int{arch.nExperts, arch.hidden, k / 32}, scales)
			}
		}
	}

	hdrMap := map[string]any{}
	var payload []byte
	for _, name := range names {
		e := byName[name]
		start := int64(len(payload))
		payload = append(payload, e.data...)
		end := int64(len(payload))
		hdrMap[name] = stTensor{DType: e.dtype, Shape: e.shape, Offsets: [2]int64{start, end}}
	}
	hdrJSON, err := json.Marshal(hdrMap)
	if err != nil {
		t.Fatal(err)
	}
	var raw []byte
	raw = binary.LittleEndian.AppendUint64(raw, uint64(len(hdrJSON)))
	raw = append(raw, hdrJSON...)
	raw = append(raw, payload...)
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeTestGGUF(t *testing.T, path string, nKV uint64, kv []byte, tensors []tensorInfo, fillVal float32) {
	t.Helper()
	gw, err := createGGUF(path, nKV, kv, tensors)
	if err != nil {
		t.Fatal(err)
	}
	for _, ti := range tensors {
		if _, err := gw.beginTensor(ti.Name); err != nil {
			t.Fatal(err)
		}
		switch ti.Type {
		case tQ8_0:
			vals := make([]float32, ti.rowLen()*ti.nRows())
			for i := range vals {
				vals[i] = fillVal
			}
			b, err := quantizeQ8_0(vals, ti.rowLen())
			if err != nil {
				t.Fatal(err)
			}
			if _, err := gw.Write(b); err != nil {
				t.Fatal(err)
			}
		default:
			sz, err := ti.size()
			if err != nil {
				t.Fatal(err)
			}
			if _, err := gw.Write(make([]byte, sz)); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := gw.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestConvertAndVerifyChecksCountsDimsAlreadyConvertedAndQ6Tensors(t *testing.T) {
	arch := archSpec{nLayers: 2, nExperts: 2, hidden: 512, ff: 256}
	dir := localTestDir(t)
	stPath := filepath.Join(dir, "official.safetensors")
	writeTestSafetensors(t, stPath, arch)

	makeTensors := func(expType, nonExpType uint32, nonExpRow uint64) []tensorInfo {
		ts := []tensorInfo{
			{Name: "output.weight", Dims: []uint64{nonExpRow, 4}, Type: nonExpType},
		}
		for l := 0; l < arch.nLayers; l++ {
			ts = append(ts,
				tensorInfo{Name: gname(l, "w13"), Dims: []uint64{uint64(arch.hidden), uint64(arch.hidden), uint64(arch.nExperts)}, Type: expType},
				tensorInfo{Name: gname(l, "w2"), Dims: []uint64{uint64(arch.ff), uint64(arch.hidden), uint64(arch.nExperts)}, Type: expType},
			)
		}
		return ts
	}

	basePath := filepath.Join(dir, "base.gguf")
	q8Path := filepath.Join(dir, "ref_q8.gguf")
	outPath := filepath.Join(dir, "converted.gguf")
	writeTestGGUF(t, basePath, 0, nil, makeTensors(tQ4_K, tQ4_K, 256), 0.5)
	writeTestGGUF(t, q8Path, 0, nil, makeTensors(tQ8_0, tQ8_0, 256), 0.5)

	// 1. Valid conversion with -q6-tensors matching output.weight + full verify pass.
	if err := convertArch(arch, stPath, basePath, q8Path, "", `^output\.weight$`, "native", outPath, true); err != nil {
		t.Fatalf("convertArch valid: %v", err)
	}
	if err := verifyArch(arch, outPath, q8Path); err != nil {
		t.Fatalf("verifyArch valid: %v", err)
	}

	// 2. Feeding an already-converted GGUF back as -base must be refused.
	outAgain := filepath.Join(dir, "again.gguf")
	err := convertArch(arch, stPath, outPath, "", "", "", "native", outAgain, true)
	if err == nil || !strings.Contains(err.Error(), "already converted") {
		t.Fatalf("expected already-converted base to be refused, got: %v", err)
	}

	// 3. Base GGUF with a pre-existing .gscale tensor (even without inkserve.experts metadata) must be refused.
	gscaleBase := filepath.Join(dir, "gscale_base.gguf")
	withGscale := append(makeTensors(tQ4_K, tQ4_K, 256), tensorInfo{Name: "blk.0.ffn_gate_up_exps.gscale", Dims: []uint64{2}, Type: tF32})
	writeTestGGUF(t, gscaleBase, 0, nil, withGscale, 0.5)
	if err := convertArch(arch, stPath, gscaleBase, "", "", "", "native", outAgain, true); err == nil || !strings.Contains(err.Error(), "already converted") {
		t.Fatalf("expected .gscale base to be refused, got: %v", err)
	}

	// 4. Unmatched -q6-tensors and unmatched -q8-tensors must fail instead of silently doing nothing.
	if err := convertArch(arch, stPath, basePath, q8Path, "", `^does\.not\.match$`, "native", outAgain, true); err == nil || !strings.Contains(err.Error(), "matched 0") {
		t.Fatalf("expected unmatched -q6-tensors error, got: %v", err)
	}
	if err := convertArch(arch, stPath, basePath, q8Path, `^does\.not\.match$`, "", "native", outAgain, true); err == nil || !strings.Contains(err.Error(), "matched 0") {
		t.Fatalf("expected unmatched -q8-tensors error, got: %v", err)
	}

	// 5. -q6-tensors matching a non-256-multiple row width must fail instead of silently falling back to base.
	narrowBase := filepath.Join(dir, "narrow_base.gguf")
	narrowQ8 := filepath.Join(dir, "narrow_q8.gguf")
	writeTestGGUF(t, narrowBase, 0, nil, makeTensors(tQ5_0, tQ5_0, 64), 0.5)
	writeTestGGUF(t, narrowQ8, 0, nil, makeTensors(tQ8_0, tQ8_0, 64), 0.5)
	if err := convertArch(arch, stPath, narrowBase, narrowQ8, "", `^output\.weight$`, "native", outAgain, true); err == nil || !strings.Contains(err.Error(), "multiple of 256") {
		t.Fatalf("expected non-256 -q6-tensors error, got: %v", err)
	}

	// 6. Missing expert tensor or mismatched expert dimensions must fail in both convert and verify.
	incompleteBase := filepath.Join(dir, "incomplete_base.gguf")
	tsIncomplete := makeTensors(tQ4_K, tQ4_K, 256)
	tsIncomplete = tsIncomplete[:len(tsIncomplete)-1]
	writeTestGGUF(t, incompleteBase, 0, nil, tsIncomplete, 0.5)
	if err := convertArch(arch, stPath, incompleteBase, "", "", "", "native", outAgain, true); err == nil || !strings.Contains(err.Error(), "expected routed-expert tensors") {
		t.Fatalf("expected missing expert tensor error, got: %v", err)
	}
	if err := verifyArch(arch, basePath, q8Path); err == nil {
		t.Fatal("expected verifyArch to reject unconverted Q4_K GGUF")
	}
}
