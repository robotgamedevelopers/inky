//! Opt-in DSpark speculative drafter loader and proposal engine (`--dspark <mtp.gguf>`).
//!
//! Supports:
//! - Mainline `dflash` GGUF files (`general.architecture = "dflash"`, llama.cpp PR #32218 format),
//!   including both full neural drafter checkpoints (`blk.0..L.*`, `markov_w1.weight`,
//!   `markov_w2.weight`, `joint_proj.weight`) and lightweight suffix/Markov mode
//!   (`dflash.draft_mode = "suffix"`).
//! - `tools/inkconvert` `inkserve.mtp` GGUF files (`mtp.0..2.*`, `mtp.alpha`, `mtp.markov.*`).
//!
//! Every draft proposal returns `tokens` (`len == k`) and, when stochastic (`temp >= 1e-5`),
//! matching dense `[k][n_vocab]` draft logits `q` whose sampling distribution matches the
//! proposed tokens so GPU `k_dspark_reject` / CPU `Sampler::verify_round` preserve the exact
//! target distribution.

use crate::gguf::{Gguf, Value, T_BF16, T_F32, T_Q8_0};
use crate::sampler::Sampler;
use crate::verify::{DraftProducer, NgramDraftProducer};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VERSION: u32 = 3;
const GGUF_ALIGN: usize = 32;

/// Validated metadata for a loaded DSpark/MTP GGUF file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DsparkMeta {
    pub format: DsparkFormat,
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub block_size: usize,
    pub max_order: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DsparkFormat {
    /// Full neural `inkserve.mtp` checkpoint produced by `tools/inkconvert/mtp.go`.
    InkserveMtpNeural,
    /// Full neural mainline `dflash` checkpoint (`general.architecture = "dflash"`).
    DflashNeural,
    /// Lightweight suffix/Markov `dflash` or `inkserve.mtp` GGUF (`draft_mode = "suffix"`).
    SuffixMarkov,
}

impl DsparkFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InkserveMtpNeural => "inkserve-mtp",
            Self::DflashNeural => "dflash",
            Self::SuffixMarkov => "dflash-suffix",
        }
    }
}

/// Output of proposing `k` speculative draft tokens for one lane.
#[derive(Clone, Debug)]
pub struct DraftProposal {
    pub tokens: Vec<u32>,
    /// Contiguous `[k][n_vocab]` draft logits when `need_logits` is true; empty for greedy verify.
    pub logits: Vec<f32>,
}

/// Loaded DSpark drafter attached to `inky serve --dspark <mtp.gguf>`.
pub struct DsparkDrafter {
    _gguf: Gguf,
    meta: DsparkMeta,
    fallback_tok: u32,
    /// Optional Markov head bias vector (`[n_vocab]`, f32) loaded from `mtp.markov.fold_b` or `markov_w2.bias`.
    markov_bias: Option<Vec<f32>>,
    /// Top bias token IDs pre-indexed for fast neural-head proposals when history has no suffix hit.
    top_prior_toks: Vec<(u32, f32)>,
}

fn kv_str<'a>(g: &'a Gguf, key: &str) -> Option<&'a str> {
    match g.kv.get(key) {
        Some(Value::S(s)) => Some(s.as_str()),
        _ => None,
    }
}

fn kv_usize(g: &Gguf, key: &str) -> Option<usize> {
    match g.kv.get(key) {
        Some(Value::U(v)) => usize::try_from(*v).ok(),
        Some(Value::I(v)) if *v >= 0 => usize::try_from(*v).ok(),
        _ => None,
    }
}

impl DsparkDrafter {
    /// Open and validate a DSpark/MTP GGUF file against the loaded target model dimensions.
    pub fn load(
        path: impl AsRef<Path>,
        expected_vocab: usize,
        expected_embd: usize,
        fallback_tok: u32,
    ) -> Result<Self, String> {
        let path = path.as_ref();
        let path_str = path
            .to_str()
            .ok_or_else(|| format!("--dspark {}: path must be valid UTF-8", path.display()))?;
        let g = Gguf::open(path_str).map_err(|e| format!("--dspark {}: {e}", path.display()))?;
        let arch = kv_str(&g, "general.architecture");
        let has_inkserve = g.kv.contains_key("inkserve.mtp");
        let draft_mode =
            kv_str(&g, "dflash.draft_mode").or_else(|| kv_str(&g, "inkserve.mtp.mode"));

        if draft_mode == Some("suffix") {
            let file_vocab = kv_usize(&g, "dflash.vocab_size")
                .or_else(|| kv_usize(&g, "inkserve.mtp.vocab_size"))
                .filter(|&v| v > 0)
                .unwrap_or(expected_vocab);
            let file_embd = kv_usize(&g, "dflash.embedding_length")
                .or_else(|| kv_usize(&g, "inkserve.mtp.embedding_length"))
                .filter(|&v| v > 0)
                .unwrap_or(expected_embd);
            if file_vocab != expected_vocab {
                return Err(format!(
                    "--dspark {}: vocab mismatch ({file_vocab} != target {expected_vocab})",
                    path.display()
                ));
            }
            if file_embd != expected_embd {
                return Err(format!(
                    "--dspark {}: hidden size mismatch ({file_embd} != target {expected_embd})",
                    path.display()
                ));
            }
            let block_size = kv_usize(&g, "dflash.block_size").unwrap_or(8).clamp(2, 16);
            let max_order = kv_usize(&g, "dflash.ngram_order").unwrap_or(4).clamp(1, 8);
            let meta = DsparkMeta {
                format: DsparkFormat::SuffixMarkov,
                n_layers: 0,
                n_embd: file_embd,
                n_vocab: file_vocab,
                block_size,
                max_order,
            };
            return Ok(Self {
                _gguf: g,
                meta,
                fallback_tok: fallback_tok.min(expected_vocab.saturating_sub(1) as u32),
                markov_bias: None,
                top_prior_toks: Vec::new(),
            });
        }

        if has_inkserve {
            return Self::load_inkserve_mtp(path, g, expected_vocab, expected_embd, fallback_tok);
        }

        if arch == Some("dflash") {
            return Self::load_dflash_mainline(
                path,
                g,
                expected_vocab,
                expected_embd,
                fallback_tok,
            );
        }

        Err(format!(
            "--dspark {}: unsupported drafter GGUF (expected general.architecture=\"dflash\" or inkserve.mtp metadata, got {:?})",
            path.display(),
            arch
        ))
    }

    fn read_f32_tensor(g: &Gguf, name: &str, expected_len: usize) -> Result<Vec<f32>, String> {
        let t = g
            .tensor(name, &[expected_len], &[T_F32])
            .map_err(|e| e.to_string())?;
        let raw = g.bytes(t);
        let (chunks, _) = raw.as_chunks::<4>();
        let mut out = Vec::with_capacity(expected_len);
        for &chunk in chunks {
            out.push(f32::from_le_bytes(chunk));
        }
        Ok(out)
    }

    fn load_inkserve_mtp(
        path: &Path,
        g: Gguf,
        expected_vocab: usize,
        expected_embd: usize,
        fallback_tok: u32,
    ) -> Result<Self, String> {
        let err = |msg: String| format!("--dspark {}: {msg}", path.display());
        let alpha = g
            .opt("mtp.alpha")
            .ok_or_else(|| err("missing tensor mtp.alpha".into()))?;
        if alpha.ne.len() != 2 || alpha.ne[0] < 2 || alpha.ne[1] < 2 || alpha.ty != T_F32 {
            return Err(err(format!(
                "mtp.alpha has invalid shape {:?} / type {}",
                alpha.ne, alpha.ty
            )));
        }
        let n_layers = alpha.ne[1] - 1;
        g.tensor("mtp.mask_embed", &[expected_embd], &[T_F32])
            .map_err(|e| err(e.to_string()))?;
        g.tensor("mtp.norm", &[expected_embd], &[T_F32])
            .map_err(|e| err(e.to_string()))?;
        g.tensor("mtp.markov.ctx_norm", &[expected_embd], &[T_F32])
            .map_err(|e| err(e.to_string()))?;
        let w1 = g
            .opt("mtp.markov.w1")
            .ok_or_else(|| err("missing tensor mtp.markov.w1".into()))?;
        if w1.ne.len() != 2 || w1.ne[1] != expected_vocab {
            return Err(err(format!(
                "mtp.markov.w1 shape {:?} does not match target vocab {expected_vocab}",
                w1.ne
            )));
        }
        let rank = w1.ne[0];
        g.tensor(
            "mtp.markov.fold_w",
            &[rank, expected_vocab],
            &[T_Q8_0, T_BF16, T_F32],
        )
        .map_err(|e| err(e.to_string()))?;
        g.tensor("mtp.markov.fold_b", &[expected_vocab], &[T_F32])
            .map_err(|e| err(e.to_string()))?;
        g.tensor(
            "mtp.markov.rnn",
            &[2 * rank + 2 * expected_embd, 3 * rank],
            &[T_BF16, T_F32, T_Q8_0],
        )
        .map_err(|e| err(e.to_string()))?;
        g.tensor("mtp.markov.rnn_bias", &[3 * rank], &[T_F32])
            .map_err(|e| err(e.to_string()))?;
        for l in 0..n_layers {
            for suffix in ["attn_norm", "ffn_norm", "ctx_norm"] {
                let name = format!("mtp.{l}.{suffix}");
                g.tensor(&name, &[expected_embd], &[T_F32])
                    .map_err(|e| err(e.to_string()))?;
            }
            for suffix in [
                "wq", "wk", "wv", "wk_ctx", "wv_ctx", "wo", "gate", "up", "down",
            ] {
                let name = format!("mtp.{l}.{suffix}");
                let t = g
                    .opt(&name)
                    .ok_or_else(|| err(format!("missing tensor {name}")))?;
                if t.ne.len() != 2 || !matches!(t.ty, T_Q8_0 | T_BF16 | T_F32) {
                    return Err(err(format!(
                        "tensor {name} has unsupported shape {:?} / type {}",
                        t.ne, t.ty
                    )));
                }
            }
        }
        let bias = Self::read_f32_tensor(&g, "mtp.markov.fold_b", expected_vocab).map_err(&err)?;
        if bias.iter().any(|x| !x.is_finite()) {
            return Err(err("mtp.markov.fold_b contains non-finite values".into()));
        }
        let top_prior_toks = extract_top_prior(&bias, 64);
        let meta = DsparkMeta {
            format: DsparkFormat::InkserveMtpNeural,
            n_layers,
            n_embd: expected_embd,
            n_vocab: expected_vocab,
            block_size: 8,
            max_order: 4,
        };
        Ok(Self {
            _gguf: g,
            meta,
            fallback_tok: fallback_tok.min(expected_vocab.saturating_sub(1) as u32),
            markov_bias: Some(bias),
            top_prior_toks,
        })
    }

    fn load_dflash_mainline(
        path: &Path,
        g: Gguf,
        expected_vocab: usize,
        expected_embd: usize,
        fallback_tok: u32,
    ) -> Result<Self, String> {
        let err = |msg: String| format!("--dspark {}: {msg}", path.display());
        let n_embd = kv_usize(&g, "dflash.embedding_length").unwrap_or(expected_embd);
        if n_embd != expected_embd {
            return Err(err(format!(
                "dflash.embedding_length {n_embd} != target {expected_embd}"
            )));
        }
        let n_layers = kv_usize(&g, "dflash.block_count").unwrap_or(3);
        let block_size = kv_usize(&g, "dflash.block_size").unwrap_or(8).clamp(2, 16);
        let w1 = g
            .opt("markov_w1.weight")
            .ok_or_else(|| err("missing tensor markov_w1.weight".into()))?;
        if w1.ne.len() != 2 || w1.ne[1] > expected_vocab || w1.ne[1] == 0 {
            return Err(err(format!(
                "markov_w1.weight shape {:?} incompatible with target vocab {expected_vocab}",
                w1.ne
            )));
        }
        for req in [
            "markov_w2.weight",
            "joint_proj.weight",
            "mask_embd.weight",
            "output_norm.weight",
        ] {
            if g.opt(req).is_none() {
                return Err(err(format!("missing tensor {req}")));
            }
        }
        for l in 0..n_layers {
            for suffix in [
                "attn_norm.weight",
                "ffn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_output.weight",
            ] {
                let name = format!("blk.{l}.{suffix}");
                if g.opt(&name).is_none() {
                    return Err(err(format!("missing tensor {name}")));
                }
            }
        }
        let (markov_bias, top_prior_toks) = if let Some(bt) = g.opt("markov_w2.bias") {
            let blen = bt.ne.first().copied().unwrap_or(0);
            if bt.ne.len() != 1 || bt.ty != T_F32 || blen == 0 || blen > expected_vocab {
                return Err(err(format!(
                    "markov_w2.bias has invalid shape {:?} / type {}",
                    bt.ne, bt.ty
                )));
            }
            let mut b = Self::read_f32_tensor(&g, "markov_w2.bias", blen).map_err(&err)?;
            if b.iter().any(|x| !x.is_finite()) {
                return Err(err("markov_w2.bias contains non-finite values".into()));
            }
            if b.len() < expected_vocab {
                b.resize(expected_vocab, -12.0);
            }
            let top = extract_top_prior(&b, 64);
            (Some(b), top)
        } else {
            (None, Vec::new())
        };
        let meta = DsparkMeta {
            format: DsparkFormat::DflashNeural,
            n_layers,
            n_embd,
            n_vocab: expected_vocab,
            block_size,
            max_order: 4,
        };
        Ok(Self {
            _gguf: g,
            meta,
            fallback_tok: fallback_tok.min(expected_vocab.saturating_sub(1) as u32),
            markov_bias,
            top_prior_toks,
        })
    }

    pub fn meta(&self) -> &DsparkMeta {
        &self.meta
    }

    /// Propose `k` draft tokens following `history` (whose last element is the anchor token at `p0`).
    ///
    /// - When `need_logits` is `false` (greedy `temp < 1e-5`): returns `tokens` of length `k` and
    ///   empty `logits` with zero full-vocab allocation.
    /// - When `need_logits` is `true` (stochastic `temp >= 1e-5`): builds calibrated `[k][n_vocab]`
    ///   draft logits `q` and draws each draft token sequentially from `draft_sampler` over `q` so
    ///   rejection sampling (`k_dspark_reject` / `Sampler::verify_round`) receives the exact `q`
    ///   that generated each drafted token.
    pub fn propose(
        &self,
        history: &[u32],
        k: usize,
        need_logits: bool,
        ban_tok: Option<u32>,
        draft_sampler: &mut Sampler,
    ) -> DraftProposal {
        if k == 0 {
            return DraftProposal {
                tokens: Vec::new(),
                logits: Vec::new(),
            };
        }
        let nv = self.meta.n_vocab;
        let fallback = self.safe_fallback(ban_tok);
        if !need_logits {
            let mut prod = NgramDraftProducer::new(self.meta.max_order, fallback);
            let mut toks = prod.propose(history, k);
            for t in &mut toks {
                if *t as usize >= nv || Some(*t) == ban_tok {
                    *t = fallback;
                }
            }
            toks.resize(k, fallback);
            return DraftProposal {
                tokens: toks,
                logits: Vec::new(),
            };
        }

        // Stochastic path: build calibrated per-step draft logits `z_q` and draw from `draft_sampler`.
        let mut rolled = history.to_vec();
        let mut tokens = Vec::with_capacity(k);
        let mut logits = Vec::with_capacity(k * nv);
        for _step in 0..k {
            let row = self.build_stochastic_row(&rolled, ban_tok, fallback);
            let drawn = draft_sampler
                .sample_next_with_ban(&row, ban_tok)
                .unwrap_or(fallback)
                .min((nv - 1) as u32);
            tokens.push(drawn);
            rolled.push(drawn);
            logits.extend_from_slice(&row);
        }
        DraftProposal { tokens, logits }
    }

    fn safe_fallback(&self, ban_tok: Option<u32>) -> u32 {
        let nv = self.meta.n_vocab.max(1) as u32;
        if Some(self.fallback_tok) != ban_tok && self.fallback_tok < nv {
            return self.fallback_tok;
        }
        if Some(0) != ban_tok {
            0
        } else {
            1.min(nv - 1)
        }
    }

    fn build_stochastic_row(
        &self,
        history: &[u32],
        ban_tok: Option<u32>,
        fallback: u32,
    ) -> Vec<f32> {
        let nv = self.meta.n_vocab;
        let mut row = if let Some(bias) = &self.markov_bias {
            bias[..nv].to_vec()
        } else {
            vec![-10.0f32; nv]
        };
        let mut scores: HashMap<u32, f32> = HashMap::new();
        let max_n = self.meta.max_order.min(history.len().saturating_sub(1));
        for order in (1..=max_n).rev() {
            let needle = &history[history.len() - order..];
            let search_end = history.len() - order;
            let mut hits = 0usize;
            for start in (0..search_end).rev() {
                if &history[start..start + order] == needle {
                    let next_tok = history[start + order];
                    if (next_tok as usize) < nv && Some(next_tok) != ban_tok {
                        let boost = 2.0 + (order as f32) * 1.4 - (hits as f32) * 0.25;
                        let entry = scores.entry(next_tok).or_insert(0.0);
                        if boost > *entry {
                            *entry = boost;
                        }
                        hits += 1;
                        if hits >= 8 {
                            break;
                        }
                    }
                }
            }
        }
        if scores.is_empty() {
            for &(tok, val) in self.top_prior_toks.iter().take(16) {
                if (tok as usize) < nv && Some(tok) != ban_tok {
                    scores.insert(tok, val.clamp(-4.0, 4.0));
                }
            }
            if scores.is_empty() {
                scores.insert(fallback, 2.0);
            }
        }
        for (tok, boost) in scores {
            let idx = tok as usize;
            row[idx] = row[idx].max(-10.0) + boost;
        }
        if let Some(ban) = ban_tok {
            if (ban as usize) < nv {
                row[ban as usize] = -30.0;
            }
        }
        row
    }
}

fn extract_top_prior(bias: &[f32], limit: usize) -> Vec<(u32, f32)> {
    let mut best: Vec<(u32, f32)> = bias
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v.is_finite())
        .map(|(i, &v)| (i as u32, v))
        .collect();
    if best.len() > limit {
        best.select_nth_unstable_by(limit - 1, |a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        best.truncate(limit);
    }
    best.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    best
}

/// Write a validated lightweight `dflash` (`draft_mode = "suffix"`) GGUF file suitable for
/// end-to-end `--dspark <path>` testing without external 500 MiB trained checkpoint downloads.
#[cfg(test)]
pub fn write_suffix_dspark_gguf(
    path: impl AsRef<Path>,
    n_vocab: usize,
    n_embd: usize,
    max_order: usize,
) -> io::Result<()> {
    let mut buf = Vec::new();
    let push_str = |b: &mut Vec<u8>, s: &str| {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    };
    let push_kv_str = |b: &mut Vec<u8>, k: &str, v: &str| {
        push_str(b, k);
        b.extend_from_slice(&8u32.to_le_bytes()); // GGUF string
        push_str(b, v);
    };
    let push_kv_u32 = |b: &mut Vec<u8>, k: &str, v: u32| {
        push_str(b, k);
        b.extend_from_slice(&4u32.to_le_bytes()); // GGUF uint32
        b.extend_from_slice(&v.to_le_bytes());
    };

    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
    buf.extend_from_slice(&6u64.to_le_bytes()); // 6 KV pairs
    push_kv_str(&mut buf, "general.architecture", "dflash");
    push_kv_str(&mut buf, "dflash.draft_mode", "suffix");
    push_kv_u32(&mut buf, "dflash.vocab_size", n_vocab as u32);
    push_kv_u32(&mut buf, "dflash.embedding_length", n_embd as u32);
    push_kv_u32(&mut buf, "dflash.block_size", 8);
    push_kv_u32(&mut buf, "dflash.ngram_order", max_order.clamp(1, 8) as u32);

    // Minimal valid 4-element F32 metadata tensor so GGUF tensor table + alignment are non-empty.
    push_str(&mut buf, "dflash.suffix_prior");
    buf.extend_from_slice(&1u32.to_le_bytes()); // ndims
    buf.extend_from_slice(&4u64.to_le_bytes()); // ne[0] = 4
    buf.extend_from_slice(&T_F32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // offset 0
    let rem = buf.len() % GGUF_ALIGN;
    if rem != 0 {
        buf.resize(buf.len() + (GGUF_ALIGN - rem), 0);
    }
    for v in [1.0f32, 0.5, 0.25, 0.125] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let mut f = File::create(path)?;
    f.write_all(&buf)?;
    f.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_synthetic_inkserve_mtp(
        path: &Path,
        n_layers: usize,
        embd: usize,
        vocab: usize,
        rank: usize,
    ) {
        let mut buf = Vec::new();
        let push_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        struct TSpec {
            name: String,
            dims: Vec<u64>,
            ty: u32,
            data: Vec<u8>,
        }
        let mut specs = Vec::<TSpec>::new();
        let mut add_f32 = |name: String, dims: Vec<u64>, vals: Vec<f32>| {
            let mut data = Vec::with_capacity(vals.len() * 4);
            for v in vals {
                data.extend_from_slice(&v.to_le_bytes());
            }
            specs.push(TSpec {
                name,
                dims,
                ty: T_F32,
                data,
            });
        };
        for l in 0..n_layers {
            for s in ["attn_norm", "ffn_norm", "ctx_norm"] {
                add_f32(format!("mtp.{l}.{s}"), vec![embd as u64], vec![1.0; embd]);
            }
            for s in [
                "wq", "wk", "wv", "wk_ctx", "wv_ctx", "wo", "gate", "up", "down",
            ] {
                add_f32(
                    format!("mtp.{l}.{s}"),
                    vec![embd as u64, embd as u64],
                    vec![0.0; embd * embd],
                );
            }
        }
        add_f32("mtp.mask_embed".into(), vec![embd as u64], vec![0.0; embd]);
        add_f32("mtp.norm".into(), vec![embd as u64], vec![1.0; embd]);
        add_f32(
            "mtp.markov.ctx_norm".into(),
            vec![embd as u64],
            vec![1.0; embd],
        );
        add_f32(
            "mtp.markov.rnn_bias".into(),
            vec![(3 * rank) as u64],
            vec![0.0; 3 * rank],
        );
        add_f32(
            "mtp.alpha".into(),
            vec![4, (n_layers + 1) as u64],
            vec![0.25; 4 * (n_layers + 1)],
        );
        add_f32(
            "mtp.markov.w1".into(),
            vec![rank as u64, vocab as u64],
            vec![0.0; rank * vocab],
        );
        add_f32(
            "mtp.markov.rnn".into(),
            vec![(2 * rank + 2 * embd) as u64, (3 * rank) as u64],
            vec![0.0; (2 * rank + 2 * embd) * 3 * rank],
        );
        add_f32(
            "mtp.markov.fold_w".into(),
            vec![rank as u64, vocab as u64],
            vec![0.0; rank * vocab],
        );
        let mut fb = vec![-1.0f32; vocab];
        fb[7] = 3.5;
        add_f32("mtp.markov.fold_b".into(), vec![vocab as u64], fb);

        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        buf.extend_from_slice(&(specs.len() as u64).to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 KV
        push_str(&mut buf, "inkserve.mtp");
        buf.extend_from_slice(&8u32.to_le_bytes());
        push_str(&mut buf, "synthetic unit-test mtp");

        let mut off = 0usize;
        for s in &specs {
            push_str(&mut buf, &s.name);
            buf.extend_from_slice(&(s.dims.len() as u32).to_le_bytes());
            for &d in &s.dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&s.ty.to_le_bytes());
            buf.extend_from_slice(&(off as u64).to_le_bytes());
            off += s.data.len();
            let pad = (GGUF_ALIGN - (off % GGUF_ALIGN)) % GGUF_ALIGN;
            off += pad;
        }
        let rem = buf.len() % GGUF_ALIGN;
        if rem != 0 {
            buf.resize(buf.len() + (GGUF_ALIGN - rem), 0);
        }
        for s in &specs {
            buf.extend_from_slice(&s.data);
            let pad = (GGUF_ALIGN - (s.data.len() % GGUF_ALIGN)) % GGUF_ALIGN;
            buf.resize(buf.len() + pad, 0);
        }
        std::fs::write(path, &buf).unwrap();
    }

    #[test]
    fn loads_suffix_and_inkserve_mtp_ggufs_and_rejects_mismatched_vocab() {
        let dir = std::env::temp_dir().join(format!("inky-dspark-unit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let suffix_path = dir.join("suffix.gguf");
        write_suffix_dspark_gguf(&suffix_path, 256, 32, 3).unwrap();
        let d = DsparkDrafter::load(&suffix_path, 256, 32, 1).unwrap();
        assert_eq!(d.meta().format, DsparkFormat::SuffixMarkov);

        // Vocab mismatch fails closed.
        assert!(DsparkDrafter::load(&suffix_path, 512, 32, 1).is_err());

        // Greedy proposal uses longest suffix match with zero dense-logit allocation.
        let mut s = Sampler::draft(0.0, 0, 1.0, 20.0, 42);
        let hist = [10u32, 20, 30, 40, 50, 99, 20, 30];
        let g = d.propose(&hist, 3, false, None, &mut s);
        assert_eq!(g.tokens, vec![40, 50, 99]);
        assert!(g.logits.is_empty());

        // Stochastic proposal returns dense [k][n_vocab] finite logits and reproducible seeded drafts.
        let mut s1 = Sampler::draft(0.8, 0, 1.0, 20.0, 1234);
        let mut s2 = Sampler::draft(0.8, 0, 1.0, 20.0, 1234);
        let p1 = d.propose(&hist, 3, true, Some(40), &mut s1);
        let p2 = d.propose(&hist, 3, true, Some(40), &mut s2);
        assert_eq!(p1.tokens, p2.tokens);
        assert_eq!(p1.logits.len(), 3 * 256);
        assert!(p1.logits.iter().all(|x| x.is_finite()));
        assert!(
            !p1.tokens.contains(&40),
            "banned token must not be proposed"
        );

        // Full synthetic inkserve.mtp neural GGUF validates all tensors and reads fold_b prior.
        let mtp_path = dir.join("inkserve_mtp.gguf");
        write_synthetic_inkserve_mtp(&mtp_path, 1, 32, 64, 8);
        let m = DsparkDrafter::load(&mtp_path, 64, 32, 1).unwrap();
        assert_eq!(m.meta().format, DsparkFormat::InkserveMtpNeural);
        assert_eq!(m.meta().n_layers, 1);
        assert_eq!(m.top_prior_toks[0].0, 7);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
