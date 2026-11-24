package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"runtime"
	"sync"
	"time"
)

// The DSpark drafter (mtp.safetensors, BF16) as a GGUF for the engine (DSPARK.md → Drafter spec):
//   - each draft layer's nine projections as Q8_0
//   - the Markov head's token embedding and rnn_proj as BF16, its folded output W_f = unembed · hidden_out as Q8_0
//   - norms, mask_embed, biases and α = softmax(fuse_weight) as F32
// The fold uses the release's own output head (model.llm.unembed.weight in the official checkpoint).

const (
	mtpLayers = 3
	mtpHidden = 2304
	mtpRank   = 256
	mtpVocab  = 201024
)

// bf16s reads a BF16 tensor as float32.
func (s *safetensors) bf16s(name string, shape ...int) ([]float32, error) {
	if _, err := s.expect(name, "BF16", shape...); err != nil {
		return nil, err
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	b, err := s.read(name, 0, int64(2*n))
	if err != nil {
		return nil, err
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*i:])) << 16)
	}
	return out, nil
}

// quantizeQ8_0 packs rows of k values (k % 32 == 0) as Q8_0: per 32 values an f16 scale and 32 int8.
func quantizeQ8_0(x []float32, k int) []byte {
	out := make([]byte, 0, len(x)/32*34)
	for b := 0; b < len(x); b += 32 {
		amax := float32(0)
		for _, v := range x[b : b+32] {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		d := amax / 127
		h := f32to16(d)
		out = binary.LittleEndian.AppendUint16(out, h)
		inv := float32(0)
		if d > 0 {
			inv = 1 / f16(h)
		}
		for _, v := range x[b : b+32] {
			out = append(out, byte(int8(math.RoundToEven(float64(v*inv)))))
		}
	}
	return out
}

func bf16bytes(x []float32) []byte {
	out := make([]byte, 0, 2*len(x))
	for _, v := range x {
		u := math.Float32bits(v)
		u += 0x7FFF + (u>>16)&1 // round to nearest even
		out = binary.LittleEndian.AppendUint16(out, uint16(u>>16))
	}
	return out
}

func f32bytes(x []float32) []byte {
	out := make([]byte, 0, 4*len(x))
	for _, v := range x {
		out = appendF32(out, v)
	}
	return out
}

type mtpTensor struct {
	name string
	typ  uint32
	dims []uint64
	data func() ([]byte, error)
}

func convertMTP(mtpPath, official, outPath string) error {
	st, err := openSafetensors(mtpPath)
	if err != nil {
		return err
	}
	defer st.Close()
	off, err := openSafetensors(official)
	if err != nil {
		return err
	}
	defer off.Close()

	var ts []mtpTensor
	add := func(name string, typ uint32, k, rows int, data func() ([]byte, error)) {
		dims := []uint64{uint64(k)}
		if rows > 1 {
			dims = append(dims, uint64(rows))
		}
		ts = append(ts, mtpTensor{name, typ, dims, data})
	}
	vec := func(src, dst string, n int) {
		add(dst, tF32, n, 1, func() ([]byte, error) {
			v, err := st.bf16s(src, n)
			if err != nil {
				return nil, err
			}
			return f32bytes(v), nil
		})
	}
	q8 := func(src, dst string, rows, k int) {
		add(dst, tQ8_0, k, rows, func() ([]byte, error) {
			v, err := st.bf16s(src, rows, k)
			if err != nil {
				return nil, err
			}
			return quantizeQ8_0(v, k), nil
		})
	}
	const nh, nkv, hd, ff = 24, 8, 128, 9216
	for l := 0; l < mtpLayers; l++ {
		p, g := fmt.Sprintf("layers.%d.", l), fmt.Sprintf("mtp.%d.", l)
		vec(p+"input_layernorm.weight", g+"attn_norm", mtpHidden)
		vec(p+"post_attention_layernorm.weight", g+"ffn_norm", mtpHidden)
		vec(p+"dflare_norm.weight", g+"ctx_norm", mtpHidden)
		vec(p+"self_attn.q_norm.weight", g+"q_norm", hd)
		vec(p+"self_attn.k_norm.weight", g+"k_norm", hd)
		vec(p+"self_attn.k_norm_ctx.weight", g+"k_norm_ctx", hd)
		q8(p+"self_attn.q_proj.weight", g+"wq", nh*hd, mtpHidden)
		q8(p+"self_attn.k_proj.weight", g+"wk", nkv*hd, mtpHidden)
		q8(p+"self_attn.v_proj.weight", g+"wv", nkv*hd, mtpHidden)
		q8(p+"self_attn.k_proj_ctx.weight", g+"wk_ctx", nkv*hd, mtpHidden)
		q8(p+"self_attn.v_proj_ctx.weight", g+"wv_ctx", nkv*hd, mtpHidden)
		q8(p+"self_attn.o_proj.weight", g+"wo", mtpHidden, nh*hd)
		q8(p+"mlp.gate_proj.weight", g+"gate", ff, mtpHidden)
		q8(p+"mlp.up_proj.weight", g+"up", ff, mtpHidden)
		q8(p+"mlp.down_proj.weight", g+"down", mtpHidden, ff)
	}
	vec("mask_embed", "mtp.mask_embed", mtpHidden)
	vec("norm.weight", "mtp.norm", mtpHidden)
	vec("markov_head.dflare_norm.weight", "mtp.markov.ctx_norm", mtpHidden)
	vec("markov_head.rnn_proj.bias", "mtp.markov.rnn_bias", 3*mtpRank)
	// α: softmax over the 4 taps of each fuse_weight row (3 layers, then the Markov head)
	add("mtp.alpha", tF32, 4, mtpLayers+1, func() ([]byte, error) {
		w, err := st.bf16s("fuse_weight", mtpLayers+1, 4)
		if err != nil {
			return nil, err
		}
		a := make([]float32, len(w))
		for r := 0; r < mtpLayers+1; r++ {
			m := float32(math.Inf(-1))
			for c := 0; c < 4; c++ {
				m = max(m, w[r*4+c])
			}
			sum := float32(0)
			for c := 0; c < 4; c++ {
				a[r*4+c] = float32(math.Exp(float64(w[r*4+c] - m)))
				sum += a[r*4+c]
			}
			for c := 0; c < 4; c++ {
				a[r*4+c] /= sum
			}
		}
		return f32bytes(a), nil
	})
	add("mtp.markov.w1", tBF16, mtpRank, mtpVocab, func() ([]byte, error) {
		v, err := st.bf16s("markov_head.markov_w1.weight", mtpVocab, mtpRank)
		if err != nil {
			return nil, err
		}
		return bf16bytes(v), nil
	})
	add("mtp.markov.rnn", tBF16, 2*mtpRank+2*mtpHidden, 3*mtpRank, func() ([]byte, error) {
		v, err := st.bf16s("markov_head.rnn_proj.weight", 3*mtpRank, 2*mtpRank+2*mtpHidden)
		if err != nil {
			return nil, err
		}
		return bf16bytes(v), nil
	})
	// the fold: W_f = U · hidden_out.weight ([V][256]) and b_f = U · hidden_out.bias, U = the release's output head
	var fw, fb []float32
	fold := func() error {
		if fw != nil {
			return nil
		}
		hw, err := st.bf16s("markov_head.hidden_out.weight", mtpHidden, mtpRank)
		if err != nil {
			return err
		}
		hb, err := st.bf16s("markov_head.hidden_out.bias", mtpHidden)
		if err != nil {
			return err
		}
		u, err := off.bf16s("model.llm.unembed.weight", mtpVocab, mtpHidden)
		if err != nil {
			return err
		}
		ht := make([]float32, mtpRank*mtpHidden) // hidden_out transposed: [256][2304]
		for i := 0; i < mtpHidden; i++ {
			for r := 0; r < mtpRank; r++ {
				ht[r*mtpHidden+i] = hw[i*mtpRank+r]
			}
		}
		fw, fb = make([]float32, mtpVocab*mtpRank), make([]float32, mtpVocab)
		t0 := time.Now()
		var wg sync.WaitGroup
		rows := make(chan int, 1024)
		for w := 0; w < runtime.NumCPU(); w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for v := range rows {
					uv := u[v*mtpHidden : (v+1)*mtpHidden]
					for r := 0; r < mtpRank; r++ {
						h := ht[r*mtpHidden : (r+1)*mtpHidden]
						var s float64
						for i, x := range uv {
							s += float64(x * h[i])
						}
						fw[v*mtpRank+r] = float32(s)
					}
					var s float64
					for i, x := range uv {
						s += float64(x * hb[i])
					}
					fb[v] = float32(s)
				}
			}()
		}
		for v := 0; v < mtpVocab; v++ {
			rows <- v
		}
		close(rows)
		wg.Wait()
		log.Printf("folded the output head into the Markov head (%d x %d) in %v", mtpVocab, mtpRank, time.Since(t0).Round(time.Second))
		return nil
	}
	add("mtp.markov.fold_w", tQ8_0, mtpRank, mtpVocab, func() ([]byte, error) {
		if err := fold(); err != nil {
			return nil, err
		}
		return quantizeQ8_0(fw, mtpRank), nil
	})
	add("mtp.markov.fold_b", tF32, mtpVocab, 1, func() ([]byte, error) {
		if err := fold(); err != nil {
			return nil, err
		}
		return f32bytes(fb), nil
	})

	infos := make([]tensorInfo, len(ts))
	for i, t := range ts {
		infos[i] = tensorInfo{Name: t.name, Dims: t.dims, Type: t.typ}
	}
	kv := appendKVString(nil, "inkserve.mtp", "Inkling 2 Mini DSpark drafter (mtp.safetensors) for inkserve's erisinky branch; see DSPARK.md")
	gw, err := createGGUF(outPath, 1, kv, infos)
	if err != nil {
		return err
	}
	for _, t := range ts {
		info, err := gw.beginTensor(t.name)
		if err != nil {
			return err
		}
		b, err := t.data()
		if err != nil {
			return fmt.Errorf("%s: %w", t.name, err)
		}
		want, _ := info.size()
		if uint64(len(b)) != want {
			return fmt.Errorf("%s: %d bytes, want %d", t.name, len(b), want)
		}
		if _, err := gw.Write(b); err != nil {
			return err
		}
	}
	return gw.Close()
}
