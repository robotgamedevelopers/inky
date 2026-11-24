package main

import (
	"math"
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
