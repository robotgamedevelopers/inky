//! A generation session: owns the context (tokens + KV/conv state on the GPU), prefill and decode loops.

use crate::engine::{Engine, Lane, Run};
use crate::metal::CmdBuf;
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;
use objc2_metal::MTLCommandBuffer;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

pub struct Session<'a> {
    pub eng: &'a Engine,
    /// this session's stream state (cache, convolution rows, tokens)
    pub lane: Lane,
    pub tk: &'a Tokenizer,
    /// tokens whose K/V and conv state are in the cache
    pub n_past: usize,
    /// a sampled token not yet fed through the model (it becomes the first input of the next step)
    pending: Option<u32>,
    pub stats: Stats,
}

#[derive(Default, Clone, Copy)]
pub struct Stats { pub prompt_tokens: usize, pub prompt_time: Duration, pub gen_tokens: usize, pub gen_time: Duration,
                   /// largest activation buffer (decode arena + prefill arena) any prefill needed, bytes
                   pub peak_scratch: usize }

impl<'a> Session<'a> {
    pub fn new(eng: &'a Engine, tk: &'a Tokenizer) -> Session<'a> { Self::with_lane(eng, tk, Lane::new(eng)) }
    /// a new session on an existing lane: its buffers are kept (no regrowth), its contents forgotten
    pub fn with_lane(eng: &'a Engine, tk: &'a Tokenizer, lane: Lane) -> Session<'a> {
        eng.on(&lane).reset();
        Session { eng, tk, lane, n_past: 0, pending: None, stats: Stats::default() }
    }
    /// give the lane back, to start another session on it
    pub fn into_lane(self) -> Lane { self.lane }
    /// the engine working on this session's lane
    pub fn run(&self) -> Run<'_> { self.eng.on(&self.lane) }

    pub fn ctx_left(&self) -> usize { self.eng.max_ctx.saturating_sub(self.n_past + 1) }

    /// Feed `toks` (preceded by any pending sampled token) through the model in prefill chunks.
    /// Returns the greedy next token (argmax of the last position); logits stay available in the engine.
    pub fn prefill(&mut self, toks: &[u32]) -> u32 {
        self.try_prefill(toks).unwrap_or_else(|e| {
            eprintln!("[inky] {e}");
            std::process::exit(1);
        })
    }

    pub fn try_prefill(&mut self, toks: &[u32]) -> Result<u32, String> {
        let mut all: Vec<u32> = self.pending.take().into_iter().collect();
        all.extend_from_slice(toks);
        if all.is_empty() {
            return Err("cannot prefill an empty token sequence".into());
        }
        if self.n_past.saturating_add(all.len()) >= self.eng.max_ctx {
            return Err(format!("context full ({} tokens)", self.eng.max_ctx));
        }
        let t0 = Instant::now();
        self.run().tier_leave(self.n_past);       // (--kv-disk) a prefill attends through the whole cache in memory
        self.run().set_tokens(self.n_past, &all);
        let end = self.n_past + all.len();
        self.run().reserve(end + 1, true);
        self.eng.ensure_batch(all.len().min(self.eng.chunk));
        self.stats.peak_scratch = self.stats.peak_scratch.max(self.run().mem().1);
        let mut p0 = self.n_past;
        let mut last: Option<CmdBuf> = None;
        while p0 < end {
            let t = (end - p0).min(self.eng.chunk);
            let cb = self.eng.gpu.cmd();
            self.run().encode_prefill(&cb, p0, t, p0 + t == end, true);
            cb.commit();
            last = Some(cb);
            p0 += t;
        }
        if let Some(cb) = last {
            cb.waitUntilCompleted();
        }
        self.n_past = end;
        self.stats.prompt_tokens += all.len();
        self.stats.prompt_time += t0.elapsed();
        Ok(self.run().token_at(end))
    }

    /// Generate up to `max` tokens after a prefill. `on_token` returns false to stop early.
    /// Greedy decoding is fully pipelined on the GPU: the argmax of step p is the embedding input of step p+1, so the CPU
    /// never sits between steps. Exactly two steps are kept in flight, so the in-flight step always processes the token
    /// just emitted: when generation stops, the context (KV + conv state) is exactly "everything emitted so far".
    /// Sampling reads the logits back each step.
    pub fn generate(&mut self, first: u32, max: usize, sampler: &mut Sampler, on_token: impl FnMut(u32) -> bool) -> usize {
        self.try_generate(first, max, sampler, on_token).unwrap_or_else(|e| {
            eprintln!("[inky] {e}");
            std::process::exit(1);
        })
    }

    pub fn try_generate(&mut self, first: u32, max: usize, sampler: &mut Sampler, mut on_token: impl FnMut(u32) -> bool) -> Result<usize, &'static str> {
        let t0 = Instant::now();
        let eng = self.eng.on(&self.lane);
        let eos = self.tk.eos;
        eng.release_batch();      // the prefill arena is not needed while generating
        if max == 0 { return Ok(0); }
        let mut tok = if sampler.greedy() {
            first
        } else {
            sampler.sample_next_with_ban_and_softcap(eng.logits(), None, self.eng.effective_sampler_softcap())?
        };
        let mut n = 1usize;
        let mut stop = tok == eos || !on_token(tok) || n >= max || self.ctx_left() < 2;
        if sampler.greedy() {
            if stop { self.pending = Some(tok); }
            else {
                eng.set_tokens(self.n_past, &[tok]);
                let mut inflight: VecDeque<(usize, CmdBuf)> = VecDeque::new();
                let step = |pos: usize| (pos, eng.step(pos, true, true));
                eng.gen_reserve(self.n_past);
                inflight.push_back(step(self.n_past));
                loop {
                    if inflight.len() < 2 {
                        if let Some(&(last_pos, _)) = inflight.back() {
                            let pos = last_pos + 1;
                            if pos >= eng.gen_capacity() {
                                // the K/V buffers must grow (or, with --kv-disk, be re-sized): nothing may be in flight meanwhile
                                for (_, cb) in inflight.iter() { cb.waitUntilCompleted(); }
                                eng.gen_reserve(pos);
                            }
                            inflight.push_back(step(pos));
                        }
                    }
                    let Some((pos, cb)) = inflight.pop_front() else { break };
                    cb.waitUntilCompleted();
                    self.n_past = pos + 1;
                    tok = eng.token_at(pos + 1);
                    n += 1;
                    stop = tok == eos || !on_token(tok) || n >= max || self.ctx_left() < 3;
                    if stop {
                        // the remaining in-flight step consumes `tok` (a real token): keep it in the context
                        match inflight.pop_front() {
                            Some((p2, cb2)) => { cb2.waitUntilCompleted(); self.n_past = p2 + 1; self.pending = None; }
                            None => self.pending = Some(tok),
                        }
                        break;
                    }
                }
            }
        } else {
            while !stop {
                eng.set_tokens(self.n_past, &[tok]);
                eng.gen_reserve(self.n_past);
                eng.step(self.n_past, true, false).waitUntilCompleted();
                self.n_past += 1;
                tok = sampler.sample_next_with_ban_and_softcap(eng.logits(), None, self.eng.effective_sampler_softcap())?;
                n += 1;
                stop = tok == eos || !on_token(tok) || n >= max || self.ctx_left() < 2;
            }
            self.pending = Some(tok);
        }
        self.stats.gen_tokens += n;
        self.stats.gen_time += t0.elapsed();
        Ok(n)
    }
}

/// Streams model output: hides protocol tokens, dims the reasoning, emits valid UTF-8 only.
pub struct Renderer { buf: Vec<u8>, pub thinking: bool, pub show_thinking: bool, pub raw: bool, think_note: bool }

impl Renderer {
    pub fn new(show_thinking: bool, raw: bool) -> Renderer { Renderer { buf: Vec::new(), thinking: false, show_thinking, raw, think_note: false } }

    pub fn push(&mut self, tk: &Tokenizer, t: u32, out: &mut impl std::io::Write) {
        if self.raw { self.emit(&tk.decode_bytes(t), out); return; }
        if tk.is_special(t) {
            let s = String::from_utf8_lossy(&tk.decode_bytes(t)).to_string();
            match s.as_str() {
                "<|content_thinking|>" => {
                    self.thinking = true;
                    if self.show_thinking { let _ = write!(out, "\x1b[2m"); } else if !self.think_note { let _ = write!(out, "\x1b[2m(thinking…)\x1b[0m "); self.think_note = true; }
                }
                "<|content_text|>" => { self.thinking = false; }
                "<|end_message|>" => { if self.thinking && self.show_thinking { let _ = write!(out, "\x1b[0m\n\n"); } self.thinking = false; }
                "<|message_model|>" | "<|content_model_end_sampling|>" => {}
                _ => { let _ = write!(out, "{s}"); }
            }
            let _ = out.flush();
            return;
        }
        if self.thinking && !self.show_thinking { return; }
        self.emit(&tk.decode_bytes(t), out);
    }

    fn emit(&mut self, bytes: &[u8], out: &mut impl std::io::Write) {
        self.buf.extend_from_slice(bytes);
        let valid = match std::str::from_utf8(&self.buf) { Ok(_) => self.buf.len(), Err(e) => e.valid_up_to() };
        if valid > 0 {
            let _ = out.write_all(&self.buf[..valid]);
            let _ = out.flush();
            self.buf.drain(..valid);
        }
        if self.buf.len() > 8 { self.buf.clear(); }
    }

    pub fn end(&mut self, out: &mut impl std::io::Write) {
        if self.thinking && self.show_thinking { let _ = write!(out, "\x1b[0m"); }
        let _ = out.flush();
    }
}
