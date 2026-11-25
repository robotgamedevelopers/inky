package main

import (
	"math"
	"path/filepath"
	"testing"
)

func cosineSim(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		ab += x * y
		aa += x * x
		bb += y * y
	}
	if aa == 0 || bb == 0 {
		return 0
	}
	return ab / math.Sqrt(aa*bb)
}

func syntheticWeights(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		// Deterministic smooth + pseudo-random Gaussian-like distribution.
		x := math.Sin(float64(i)*0.173+0.3) * 0.45 + math.Cos(float64(i)*0.719) * 0.35 + math.Sin(float64(i)*2.31) * 0.2
		out[i] = float32(x)
	}
	return out
}

func TestSub4BitQuantRoundTripAndByteSavings(t *testing.T) {
	w256 := syntheticWeights(2304)
	q4kBytesRow := (len(w256) / 256) * q4kBytes
	q80BytesRow := (len(w256) / 32) * 34

	// Q3_K: 110B / 256 weights = 3.4375 bpw (-23.6% vs Q4_K, -59.6% vs Q8_0)
	buf3k := make([]byte, (len(w256)/256)*q3kBytes)
	quantizeQ3K(buf3k, w256)
	dq3k := dequantQ3K(buf3k, len(w256))
	cos3k := cosineSim(dq3k, w256)
	if len(buf3k) >= q4kBytesRow || len(buf3k) >= q80BytesRow {
		t.Fatalf("Q3_K row bytes %d not smaller than Q4_K %d / Q8_0 %d", len(buf3k), q4kBytesRow, q80BytesRow)
	}
	if cos3k < 0.985 {
		t.Fatalf("Q3_K cosine similarity too low: %.5f", cos3k)
	}

	// Q2_K: 84B / 256 weights = 2.625 bpw (-41.7% vs Q4_K, -69.1% vs Q8_0)
	buf2k := make([]byte, (len(w256)/256)*q2kBytes)
	quantizeQ2K(buf2k, w256)
	dq2k := dequantQ2K(buf2k, len(w256))
	cos2k := cosineSim(dq2k, w256)
	if len(buf2k) >= len(buf3k) {
		t.Fatalf("Q2_K row bytes %d not smaller than Q3_K %d", len(buf2k), len(buf3k))
	}
	if cos2k < 0.94 {
		t.Fatalf("Q2_K cosine similarity too low: %.5f", cos2k)
	}

	// Q3_0: 14B / 32 weights = 3.50 bpw (-36.4% vs Q5_0, -58.8% vs Q8_0)
	w1152 := syntheticWeights(1152)
	buf30 := make([]byte, (len(w1152)/32)*q30Bytes)
	quantizeQ3_0(buf30, w1152)
	dq30 := dequantQ3_0(buf30, len(w1152))
	cos30 := cosineSim(dq30, w1152)
	q50BytesRow := (len(w1152) / 32) * q50Bytes
	if len(buf30) >= q50BytesRow {
		t.Fatalf("Q3_0 row bytes %d not smaller than Q5_0 %d", len(buf30), q50BytesRow)
	}
	if cos30 < 0.98 {
		t.Fatalf("Q3_0 cosine similarity too low: %.5f", cos30)
	}
}

func TestRequantGGUFEndToEnd(t *testing.T) {
	dir := t.TempDir()
	inPath := filepath.Join(dir, "base.gguf")
	outQ3 := filepath.Join(dir, "out-q3k.gguf")
	outQ2 := filepath.Join(dir, "out-q2k.gguf")

	// Build a minimal valid GGUF containing a Q8_0 gate_up tensor (K=256) and Q8_0 down tensor (K=128).
	guW := syntheticWeights(256 * 4)
	dnW := syntheticWeights(128 * 4)
	encodeQ8 := func(w []float32) []byte {
		out := make([]byte, (len(w)/32)*34)
		for b := 0; b < len(w)/32; b++ {
			sub := w[32*b : 32*b+32]
			var amax float32
			for _, v := range sub {
				if a := float32(math.Abs(float64(v))); a > amax {
					amax = a
				}
			}
			d := amax / 127.0
			dh := f32to16(d)
			o := out[b*34 : b*34+34]
			o[0], o[1] = byte(dh), byte(dh>>8)
			df := f16(dh)
			if df > 0 {
				for j := 0; j < 32; j++ {
					q := max2(-127, min2(127, nearestInt(sub[j]/df)))
					o[2+j] = byte(int8(q))
				}
			}
		}
		return out
	}

	infos := []tensorInfo{
		{Name: "blk.0.ffn_gate_up_exps.weight", Dims: []uint64{256, 2, 2}, Type: tQ8_0},
		{Name: "blk.0.ffn_down_exps.weight", Dims: []uint64{128, 2, 2}, Type: tQ8_0},
	}
	kv := appendKVString(nil, "general.architecture", "inkling2_loop")
	gw, err := createGGUF(inPath, 1, kv, infos)
	if err != nil {
		t.Fatal(err)
	}
	defer gw.discard()
	if _, err := gw.beginTensor(infos[0].Name); err != nil {
		t.Fatal(err)
	}
	if _, err := gw.Write(encodeQ8(guW)); err != nil {
		t.Fatal(err)
	}
	if _, err := gw.beginTensor(infos[1].Name); err != nil {
		t.Fatal(err)
	}
	if _, err := gw.Write(encodeQ8(dnW)); err != nil {
		t.Fatal(err)
	}
	if err := gw.publish(); err != nil {
		t.Fatal(err)
	}

	if err := requantGGUF(inPath, outQ3, "q3k", `ffn_(gate_up|down)_exps\.weight$`); err != nil {
		t.Fatalf("requant q3k: %v", err)
	}
	if err := requantGGUF(inPath, outQ2, "q2k", `ffn_(gate_up|down)_exps\.weight$`); err != nil {
		t.Fatalf("requant q2k: %v", err)
	}

	g3, err := openGGUF(outQ3)
	if err != nil {
		t.Fatal(err)
	}
	defer g3.Close()
	gu3, ok := g3.tensor("blk.0.ffn_gate_up_exps.weight")
	if !ok || gu3.Type != tQ3_K {
		t.Fatalf("expected Q3_K gate_up, got %+v", gu3)
	}
	dn3, ok := g3.tensor("blk.0.ffn_down_exps.weight")
	if !ok || dn3.Type != tQ3_0 {
		t.Fatalf("expected Q3_0 down, got %+v", dn3)
	}

	g2, err := openGGUF(outQ2)
	if err != nil {
		t.Fatal(err)
	}
	defer g2.Close()
	gu2, ok := g2.tensor("blk.0.ffn_gate_up_exps.weight")
	if !ok || gu2.Type != tQ2_K {
		t.Fatalf("expected Q2_K gate_up, got %+v", gu2)
	}
}
