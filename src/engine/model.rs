//! Model validation, weight binding, and Metal engine initialization.

use super::*;

const MAX_PREFILL_CHUNK: usize = 4096;

pub(super) fn configured_prefill_chunk(value: Result<String, std::env::VarError>, context: usize) -> Result<usize, &'static str> {
    let chunk = match value {
        Err(std::env::VarError::NotPresent) => Some(MAX_PREFILL_CHUNK),
        Ok(value) => value.parse::<usize>().ok(),
        Err(std::env::VarError::NotUnicode(_)) => None,
    };
    chunk.filter(|&n| (32..=MAX_PREFILL_CHUNK).contains(&n) && n % 32 == 0)
        .map(|n| n.min(context))
        .ok_or("INK_CHUNK must be a multiple of 32 between 32 and 4096")
}

// ----------------------------------------------------------------------------- weights
#[derive(Clone, Copy, Default, Debug)]
pub struct QW { pub off: u64, pub row_bytes: u32, pub ty: u32, pub rows: u32, pub k: u32, pub expert_bytes: u64, pub gs: u64 }

fn qw(g: &Gguf, name: &str, shape: &[usize], types: &[u32]) -> io::Result<QW> {
    let t = g.tensor(name, shape, types)?;
    let rb = t.row_bytes;
    // All callers pass the target model's fixed geometry. The shaders use 32-bit strides.
    let expert_bytes = u32::try_from(rb * t.ne[1])
        .map_err(|_| invalid_model(format_args!("tensor {name} exceeds the GPU stride limit")))?;
    let gs = if t.ty == T_NV4 {
        let gname = name.replace(".weight", ".gscale");
        g.tensor(&gname, &[NEXP], &[T_F32])?.file_off as u64
    } else {
        0
    };
    Ok(QW {
        off: t.file_off as u64,
        row_bytes: rb as u32,
        ty: t.ty,
        rows: t.ne[1] as u32,
        k: t.ne[0] as u32,
        expert_bytes: u64::from(expert_bytes),
        gs,
    })
}

fn validate_vocab(g: &Gguf, unpadded: u64, padded: usize) -> io::Result<usize> {
    let token_count = match g.kv.get("tokenizer.ggml.tokens") {
        Some(Value::AS(tokens)) => tokens.len(),
        _ => return Err(invalid_model("missing GGUF tokenizer tokens")),
    };
    let type_count = match g.kv.get("tokenizer.ggml.token_type") {
        Some(Value::AI(types)) => types.len(),
        _ => return Err(invalid_model("missing GGUF tokenizer token types")),
    };
    let eos = g.uint("tokenizer.ggml.eos_token_id")?;
    check_vocab_lengths(unpadded, padded, token_count, type_count, eos)
}

pub(super) fn check_vocab_lengths(unpadded: u64, padded: usize, tokens: usize, types: usize, eos: u64) -> io::Result<usize> {
    if unpadded == 0 || unpadded > padded as u64 || unpadded != tokens as u64 || tokens != types {
        return Err(invalid_model(format_args!(
            "vocabulary length {unpadded} must fit {padded} rows and match the tokenizer's {tokens} tokens and {types} token types",
        )));
    }
    if eos >= unpadded {
        return Err(invalid_model(format_args!("EOS token ID {eos} is outside the vocabulary")));
    }
    Ok(tokens)
}

pub struct Layer {
    pub swa: bool,
    pub wq: QW, pub wgate: QW, pub wk: QW, pub wv: QW, pub wo: QW, pub wrel: QW, pub wsink: QW,
    pub kconvp: Option<QW>, pub vconvp: Option<QW>,
    pub attn_norm: i32, pub q_inp_norm: i32, pub k_inp_norm: i32, pub v_inp_norm: i32,
    pub attn_res_query: i32, pub n_attn_q: u32,
    pub ffn_res_query: u32, pub ffn_norm: u32,
    pub q_clip: u32, pub rel_clip: u32, pub rel_bias: u32, pub k_conv: u32, pub v_conv: u32,
    pub attn_post_conv: u32, pub ffn_post_conv: u32,
    pub router: u32, pub exp_bias: u32,
    pub gu: QW, pub dn: QW,
}

pub struct LoopW { pub msg_query: u32, pub down: [u32; 4], pub gain: [u32; 4], pub forget: [u32; 4] }

struct Pb { data: Vec<u8> }
impl Pb {
    fn align(&mut self) { while self.data.len() % 256 != 0 { self.data.push(0); } }
    fn f32(&mut self, g: &Gguf, name: &str, shape: &[usize]) -> io::Result<u32> {
        let t = g.tensor(name, shape, &[T_F32])?;
        self.align();
        let off = self.data.len() as u32;
        self.data.extend_from_slice(g.bytes(t));
        Ok(off)
    }
    /// F32 -> BF16 (round to nearest even). Lossless for this model's bf16-origin matrices.
    fn bf16(&mut self, g: &Gguf, name: &str, shape: &[usize]) -> io::Result<u32> {
        let t = g.tensor(name, shape, &[T_F32])?;
        self.align();
        let off = self.data.len() as u32;
        for c in g.bytes(t).chunks_exact(4) {
            let u = u32::from_le_bytes(c.try_into().unwrap());
            let r = u.wrapping_add(0x7FFF + ((u >> 16) & 1)) >> 16;
            self.data.extend_from_slice(&(r as u16).to_le_bytes());
        }
        Ok(off)
    }
}

impl Engine {
    pub fn load(path: &str, max_ctx: usize) -> Engine {
        let chunk = configured_prefill_chunk(std::env::var("INK_CHUNK"), max_ctx).unwrap_or_else(|err| {
            eprintln!("[inky] {err}");
            std::process::exit(2);
        });
        Self::load_with(path, max_ctx, KV8.load(std::sync::atomic::Ordering::Relaxed), chunk)
            .unwrap_or_else(|err| {
                eprintln!("[inky] could not load {path}: {err}");
                std::process::exit(1);
            })
    }
    /// `kv8`: keep the K/V cache as 8-bit integers with a per-vector scale (half the memory, not exact) instead of fp16
    fn load_with(path: &str, max_ctx: usize, kv8: bool, chunk: usize) -> io::Result<Engine> {
        let t0 = std::time::Instant::now();
        let g = Gguf::open(path)?;
        if !matches!(g.kv.get("general.architecture"), Some(Value::S(s)) if s == "inkling2_loop") {
            return Err(invalid_model("expected inkling2_loop architecture"));
        }
        let a = |k: &str| format!("inkling2_loop.{k}");
        for (name, expected) in [
            ("embedding_length", E), ("block_count", N_PHYS), ("expert_count", NEXP),
            ("expert_used_count", TOPK), ("loop.start", LOOP_START), ("loop.end", LOOP_END),
            ("loop.count", N_PASS), ("attn_res.block_size", BLOCK), ("attention.sliding_window", EXT),
            ("attention.relative_extent", EXT), ("vocab_size", N_VOCAB_PAD),
        ] {
            let value = g.uint(&a(name))?;
            if value != expected as u64 {
                return Err(invalid_model(format_args!("{} is {value}, expected {expected}", a(name))));
            }
        }
        let swa_pat = match g.kv.get(&a("attention.sliding_window_pattern")) {
            Some(Value::AB(v)) if v.len() == N_PHYS
                && v.iter().copied().eq((0..N_PHYS).map(|i| (i + 1) % BLOCK != 0)) => v,
            _ => return Err(invalid_model("unexpected sliding-window attention pattern")),
        };

        let n_vocab = validate_vocab(&g, g.uint(&a("vocab_size_unpadded"))?, N_VOCAB_PAD)?;
        let _ = crate::tokenizer::Tokenizer::try_load(&g)?;
        let embed_shape = [E, N_VOCAB_PAD];
        let tok_embd = qw(&g, "token_embd.weight", &embed_shape, &[T_Q4_K, T_Q8_0])?;
        let output = qw(&g, "output.weight", &embed_shape, &[T_Q6_K, T_Q8_0])?;

        let eps = g.f(&a("attention.layer_norm_rms_epsilon"))?;
        let log_alpha = g.f(&a("attention.log_scale_alpha"))?;
        let log_floor = g.f(&a("attention.log_scale_floor"))?;
        let situ_beta = g.f(&a("activation.situ_beta"))?;
        let situ_lbeta = g.f(&a("activation.situ_linear_beta"))?;
        let wscale = g.f(&a("expert_weights_scale"))?;
        let softcap = g.f(&a("final_logit_softcapping"))?;

        let mut pb = Pb { data: Vec::new() };
        let mut layers = Vec::new();
        for (i, &swa) in swa_pat.iter().enumerate() {
            let n = |s: &str| format!("blk.{i}.{s}");
            let (attn_norm, q_inp_norm, k_inp_norm, v_inp_norm, attn_res_query, n_attn_q, kconvp, vconvp) =
                if swa {
                    let query = n("attn_res_query.weight");
                    let res_query = match g.opt(&query) {
                        Some(t) => {
                            let shape: &[usize] = if t.ne.len() == 2 { &[E, 1] } else { &[E] };
                            pb.f32(&g, &query, shape)? as i32
                        }
                        None => -1,
                    };
                    (pb.f32(&g, &n("attn_norm.weight"), &[E])? as i32, -1, -1, -1, res_query, 1, None, None)
                } else {
                    (
                        -1,
                        pb.f32(&g, &n("attn_q_inp_norm.weight"), &[E])? as i32,
                        pb.f32(&g, &n("attn_k_inp_norm.weight"), &[E])? as i32,
                        pb.f32(&g, &n("attn_v_inp_norm.weight"), &[E])? as i32,
                        pb.f32(&g, &n("attn_res_query.weight"), &[E, 3])? as i32,
                        3,
                        Some(qw(&g, &n("attn_k_conv_proj.weight"), &[E, NKV * CONV_TAPS], &[T_Q4_K, T_Q6_K, T_Q8_0])?),
                        Some(qw(&g, &n("attn_v_conv_proj.weight"), &[E, NKV * CONV_TAPS], &[T_Q4_K, T_Q6_K, T_Q8_0])?),
                    )
                };
            layers.push(Layer {
                swa,
                wq: qw(&g, &n("attn_q.weight"), &[E, NH * HD], &[T_Q4_K, T_Q8_0])?,
                wgate: qw(&g, &n("attn_gate.weight"), &[E, NH * HD], &[T_Q4_K, T_Q8_0])?,
                wk: qw(&g, &n("attn_k.weight"), &[E, NKV * HD], &[T_Q4_K, T_Q8_0])?,
                wv: qw(&g, &n("attn_v.weight"), &[E, NKV * HD], &[T_Q4_K, T_Q6_K, T_Q8_0])?,
                wo: qw(&g, &n("attn_output.weight"), &[NH * HD, E], &[T_Q4_K, T_Q8_0])?,
                wrel: qw(&g, &n("attn_rel.weight"), &[E, NH * NREL], &[T_Q4_K, T_Q8_0])?,
                wsink: qw(&g, &n("attn_sink_proj.weight"), &[E, NH], &[T_Q4_K, T_Q8_0])?,
                kconvp, vconvp,
                attn_norm, q_inp_norm, k_inp_norm, v_inp_norm, attn_res_query, n_attn_q,
                ffn_res_query: pb.f32(&g, &n("ffn_res_query.weight"), &[E])?,
                ffn_norm: pb.f32(&g, &n("ffn_norm.weight"), &[E])?,
                q_clip: pb.f32(&g, &n("attn_q_clip.weight"), &[HD, NH])?,
                rel_clip: pb.f32(&g, &n("attn_rel_clip.weight"), &[NREL, NH])?,
                rel_bias: pb.f32(&g, &n("attn_rel_bias.weight"), &[NREL, EXT])?,
                k_conv: pb.f32(&g, &n("attn_k_conv.weight"), &[CONV_TAPS, NKV * HD])?,
                v_conv: pb.f32(&g, &n("attn_v_conv.weight"), &[CONV_TAPS, NKV * HD])?,
                attn_post_conv: pb.f32(&g, &n("attn_post_conv.weight"), &[CONV_TAPS, E])?,
                ffn_post_conv: pb.f32(&g, &n("ffn_post_conv.weight"), &[CONV_TAPS, E])?,
                router: pb.bf16(&g, &n("ffn_gate_inp.weight"), &[E, NEXP])?,
                exp_bias: pb.f32(&g, &n("exp_probs_b.bias"), &[NEXP])?,
                gu: qw(&g, &n("ffn_gate_up_exps.weight"), &[E, 2 * FF, NEXP], &[T_Q4_K, T_Q8_0, T_NV4, T_MX8])?,
                dn: qw(&g, &n("ffn_down_exps.weight"), &[FF, E, NEXP], &[T_Q8_0, T_Q5_0, T_NV4, T_MX8])?,
            });
        }
        let mut loops = Vec::new();
        for i in LOOP_START..LOOP_END {
            let n = |s: &str| format!("blk.{i}.{s}");
            let mut lw = LoopW {
                msg_query: pb.f32(&g, &n("loop_msg_query.weight"), &[E, 4])?,
                down: [0; 4], gain: [0; 4], forget: [0; 4],
            };
            for (c, name) in ["q", "k", "v", "ffn"].into_iter().enumerate() {
                lw.down[c] = pb.bf16(&g, &n(&format!("loop_{name}_ctrl_down.weight")), &[2 * E, CTRL_RANK])?;
                lw.gain[c] = pb.bf16(&g, &n(&format!("loop_{name}_ctrl_gain.weight")), &[CTRL_RANK, E])?;
                lw.forget[c] = pb.bf16(&g, &n(&format!("loop_{name}_ctrl_forget.weight")), &[CTRL_RANK, E])?;
            }
            loops.push(lw);
        }
        let tok_norm = pb.f32(&g, "token_embd_norm.weight", &[E])?;
        let output_norm = pb.f32(&g, "output_norm.weight", &[E])?;
        let output_res_query = pb.f32(&g, "output_res_query.weight", &[E])?;

        let (chip, mem) = crate::tuning::this_machine();
        let tuning = crate::tuning::select(&chip, mem);
        let disable_mpp = DISABLE_MPP.load(std::sync::atomic::Ordering::Relaxed)
            || std::env::var("INKY_DISABLE_MPP").is_ok_and(|v| !v.is_empty() && v != "0")
            || std::env::var("INK_NO_TENSOR").is_ok()
            || !tuning.use_mpp;
        let use_mpp = !disable_mpp;
        let src = format!(
            "{}{}{}\n{}\n{}",
            if kv8 { "#define KV8 1\n" } else { "" },
            if !use_mpp { "#define NO_MPP 1\n" } else { "" },
            include_str!("../kernels.metal"),
            include_str!("../prefill.metal"),
            include_str!("../bench_kernels.metal")
        );
        let kvb = if kv8 { 1 } else { 2 };
        let gpu = Gpu::new_with_lang(&src, if use_mpp { KERNELS } else { KERNELS_NON_MPP }, use_mpp);
        if g.map_len() > gpu.dev.maxBufferLength() {
            return Err(invalid_model("GGUF exceeds the Metal buffer size limit"));
        }
        let w = gpu.buffer_nocopy(g.ptr(), g.map_len());
        pb.data.extend(std::iter::repeat(0u8).take(128 * NREL * 4));   // k_attn_h reads up to 95 rows past a relative-bias table
        let pbuf = gpu.buffer_with(&pb.data);

        let max_split = MAX_SPLIT;
        let ar = Arena::new(max_split, N_VOCAB_PAD);
        assert!(max_ctx % 64 == 0, "context size must be a multiple of 64");
        // the batch arena is allocated when a prefill needs it and released when generation starts
        let act = gpu.buffer(ar.total as usize * 4);
        // K/V cache layout: per slot [K: NKV x cap x HD][V: NKV x cap x HD] halfs, sliding-window and full slots in separate
        // buffers that both grow with the context (Metal commits a buffer when it is created, so capacity is real memory)
        let mut slot_kind = Vec::new();
        let (mut n_swa, mut n_full) = (0usize, 0usize);
        for s in 0..N_SLOT {
            let phys = if s < N_PHYS { s } else { LOOP_START + (s - N_PHYS) % (LOOP_END - LOOP_START) };
            if layers[phys].swa { slot_kind.push((true, n_swa)); n_swa += 1; } else { slot_kind.push((false, n_full)); n_full += 1; }
        }
        assert!(n_full * 2 * NKV * max_ctx * HD < u32::MAX as usize, "KV cache too large for 32-bit offsets");
        let ctr = gpu.buffer(4096 * 4);
        eprintln!("[inky] loaded {} ({:.1} GiB mapped, {:.1} MiB params) in {:.2?}",
            path.rsplit('/').next().unwrap_or(path), g.file_len() as f64 / (1u64 << 30) as f64, pb.data.len() as f64 / (1 << 20) as f64, t0.elapsed());
        // KV_CAP0 positions is one block. Further slots come from `BlockPool::grow`.
        let block_pool = std::cell::RefCell::new(crate::blocks::BlockPools::gpu(&gpu.dev, 1, 1, kvb as u32));
        let blocked_slab = Rc::new(RefCell::new(BlockedSlab::new(max_ctx.div_ceil(crate::blocks::BLOCK))));
        let eng = Engine {
            eps, log_alpha, log_floor, situ_beta, situ_lbeta, wscale, softcap,
            g, gpu, w, pb: pbuf, act: std::cell::RefCell::new(act),
            slot_kind, qcap: RefCell::new(None), ctr, layers, loops, tok_embd, tok_norm, output, output_norm, output_res_query,
            ar, abase: Cell::new(0), dstart: Cell::new(0), kv8, kvb, ba: Default::default(), chunk,
            use_mpp,
            kv_window_pct: tuning.kv_window_pct,
            decode_splits: tuning.decode_splits,
            tensor: use_mpp,
            hgemm: use_mpp && std::env::var("INK_NO_HGEMM").is_err(),
            sep_comb: std::env::var("INK_FUSE_COMB").is_err(), sep_topk: std::env::var("INK_SEP_TOPK").is_ok(),
            tail: std::cell::Cell::new(std::env::var("INK_NO_TAIL").is_err()),
            skip: Default::default(), stop: std::cell::Cell::new(usize::MAX), done: std::cell::Cell::new(0),
            last_out: Default::default(), last_mid: Default::default(),
            max_ctx, n_vocab,
            verify_keep: RefCell::new(None), verify_record: Cell::new(false),
            block_pool,
            blocked_slab,
            loopcd_omega: Cell::new(0.0),
            guided_logit_rows: Cell::new(0),
            lg_weak: RefCell::new(None),
            no_grp_dec: Cell::new(std::env::var("INK_NO_GRP_DEC").is_ok_and(|v| !v.is_empty() && v != "0")),
        };
        let mut eng = eng;
        eng.ensure_batch(64);
        let ok = !eng.hgemm || eng.layout_ok();
        eng.release_batch();
        if !ok {
            eprintln!("[inky] unexpected tensor accumulator layout on this GPU/OS: using the slower prefill path");
            eng.hgemm = false;
        }
        if std::env::var("INK_NO_WARM").is_err() { eng.warm(); }
        Ok(eng)
    }
}
