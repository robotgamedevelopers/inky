package main

import (
	"math"
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
	rel, _, err := relErr(dequantQ6K(buf, len(x)), x)
	if err != nil {
		t.Fatal(err)
	}
	if rel > 0.03 {
		t.Errorf("Q6_K round trip error %.4f", rel)
	}
}

func TestRelativeErrorVerifiesZeroAndNonfiniteWeights(t *testing.T) {
	for _, tc := range []struct {
		name      string
		got, want []float32
		rel       float64
		fail      bool
	}{
		{"zero rows", []float32{0, 0}, []float32{0, 0}, 0, false},
		{"zero output", []float32{0}, []float32{2}, 1, false},
		{"nonzero against zero", []float32{1}, []float32{0}, 0, true},
		{"nan after mismatch", []float32{2, float32(math.NaN())}, []float32{1, 1}, 0, true},
		{"infinite reference", []float32{1}, []float32{float32(math.Inf(1))}, 0, true},
		{"matching infinities", []float32{float32(math.Inf(-1))}, []float32{float32(math.Inf(-1))}, 0, true},
		{"different widths", []float32{1}, []float32{1, 0}, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rel, _, err := relErr(tc.got, tc.want)
			if (err != nil) != tc.fail || rel != tc.rel {
				t.Fatalf("relErr = %v, %v", rel, err)
			}
		})
	}
}

func TestHalfNonfiniteScale(t *testing.T) {
	if !math.IsInf(float64(f16(0x7c00)), 1) || !math.IsInf(float64(f16(0xfc00)), -1) ||
		!math.IsNaN(float64(f16(0x7c01))) {
		t.Fatal("nonfinite fp16 weight scales must decode as nonfinite")
	}
}

func TestRejectUnknownExpertLayoutBeforeOpeningFiles(t *testing.T) {
	err := convert("", "", "", "", "", "native", "", "interleavd")
	if err == nil || !strings.Contains(err.Error(), "-w13") {
		t.Fatalf("invalid -w13 layout: got %v", err)
	}
}

func TestExpertRowOrder(t *testing.T) {
	for _, tc := range []struct {
		name        string
		interleaved bool
		rows        []int
	}{
		{"interleaved", true, []int{0, 2, 1, 3}},
		{"stacked", false, []int{0, 1, ff, ff + 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for i, output := range []int{0, 1, ff, ff + 1} {
				if got := srcRow("w13", tc.interleaved, output); got != tc.rows[i] {
					t.Errorf("W13 source for row %d = %d, want %d", output, got, tc.rows[i])
				}
			}
		})
	}
}
