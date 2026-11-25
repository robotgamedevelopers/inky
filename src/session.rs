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
        let mut all: Vec<u32> = self.pending.take().into_iter().collect();
        all.extend_from_slice(toks);
        assert!(!all.is_empty());
        assert!(self.n_past + all.len() < self.eng.max_ctx, "context full ({} tokens)", self.eng.max_ctx);
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
        last.unwrap().waitUntilCompleted();
        self.n_past = end;
        self.stats.prompt_tokens += all.len();
        self.stats.prompt_time += t0.elapsed();
        self.run().token_at(end)
    }

    /// Generate up to `max` tokens after a prefill. `on_token` returns false to stop early.
    /// Greedy decoding is fully pipelined on the GPU: the argmax of step p is the embedding input of step p+1, so the CPU
    /// never sits between steps. Exactly two steps are kept in flight, so the in-flight step always processes the token
    /// just emitted: when generation stops, the context (KV + conv state) is exactly "everything emitted so far".
    /// Sampling reads the logits back each step.
    pub fn generate(&mut self, first: u32, max: usize, sampler: &mut Sampler, mut on_token: impl FnMut(u32) -> bool) -> usize {
        let t0 = Instant::now();
        let eng = self.eng.on(&self.lane);
        let eos = self.tk.eos;
        eng.release_batch();      // the prefill arena is not needed while generating
        if max == 0 { return 0; }
        let mut tok = if sampler.greedy() {
            first
        } else {
            sampler
                .sample_next_with_ban_and_softcap(eng.logits(), None, self.eng.effective_sampler_softcap())
                .expect("model sampling failed")
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
                        let pos = inflight.back().unwrap().0 + 1;
                        if pos >= eng.gen_capacity() {
                            // the K/V buffers must grow (or, with --kv-disk, be re-sized): nothing may be in flight meanwhile
                            for (_, cb) in inflight.iter() { cb.waitUntilCompleted(); }
                            eng.gen_reserve(pos);
                        }
                        inflight.push_back(step(pos));
                    }
                    let (pos, cb) = inflight.pop_front().unwrap();
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
                tok = sampler
                    .sample_next_with_ban_and_softcap(eng.logits(), None, self.eng.effective_sampler_softcap())
                    .expect("model sampling failed");
                n += 1;
                stop = tok == eos || !on_token(tok) || n >= max || self.ctx_left() < 2;
            }
            self.pending = Some(tok);
        }
        self.stats.gen_tokens += n;
        self.stats.gen_time += t0.elapsed();
        n
    }
}

/// Streams model output: hides protocol tokens, dims the reasoning, emits valid UTF-8 only.
pub struct Renderer { buf: Vec<u8>, pub thinking: bool, pub show_thinking: bool, pub raw: bool, think_note: bool }

const UTF8_REPLACEMENT: &[u8] = "\u{FFFD}".as_bytes();

impl Renderer {
    pub fn new(show_thinking: bool, raw: bool) -> Renderer { Renderer { buf: Vec::new(), thinking: false, show_thinking, raw, think_note: false } }

    pub fn push(&mut self, tk: &Tokenizer, t: u32, out: &mut impl std::io::Write) {
        if self.raw { self.emit(&tk.decode_bytes(t), out); return; }
        if tk.is_special(t) {
            self.flush_incomplete(out);
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
        let mut wrote = false;
        loop {
            match std::str::from_utf8(&self.buf) {
                Ok(valid) => {
                    if !valid.is_empty() {
                        let _ = out.write_all(valid.as_bytes());
                        wrote = true;
                    }
                    self.buf.clear();
                    break;
                }
                Err(err) => {
                    let valid_len = err.valid_up_to();
                    if valid_len > 0 {
                        let _ = out.write_all(&self.buf[..valid_len]);
                        wrote = true;
                    }
                    match err.error_len() {
                        Some(bad_len) => {
                            let _ = out.write_all(UTF8_REPLACEMENT);
                            wrote = true;
                            self.buf.drain(..valid_len + bad_len);
                        }
                        None => {
                            if valid_len > 0 {
                                self.buf.drain(..valid_len);
                            }
                            break;
                        }
                    }
                }
            }
        }
        if wrote {
            let _ = out.flush();
        }
    }

    fn flush_incomplete(&mut self, out: &mut impl std::io::Write) {
        if !self.buf.is_empty() {
            self.buf.clear();
            let _ = out.write_all(UTF8_REPLACEMENT);
        }
    }

    pub fn end(&mut self, out: &mut impl std::io::Write) {
        self.flush_incomplete(out);
        if self.thinking && self.show_thinking { let _ = write!(out, "\x1b[0m"); }
        let _ = out.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_replaces_bad_utf8_with_replacement_char_and_keeps_following_text() {
        // Standalone invalid byte followed by >8 bytes of valid ASCII in one chunk and subsequent chunks.
        let mut r = Renderer::new(true, true);
        let mut out = Vec::new();
        r.emit(b"\xffhello, world!", &mut out);
        r.emit(b" more text", &mut out);
        r.end(&mut out);
        assert_eq!(String::from_utf8(out).unwrap(), "\u{FFFD}hello, world! more text");

        // Split valid 4-byte emoji across four 1-byte chunks, then an incomplete 3-byte prefix followed by valid ASCII,
        // then a trailing incomplete 2-byte prefix flushed at end().
        let mut r2 = Renderer::new(true, true);
        let mut out2 = Vec::new();
        for &b in "🦀".as_bytes() {
            r2.emit(&[b], &mut out2);
        }
        r2.emit(&[0xE2, 0x82], &mut out2);
        assert_eq!(String::from_utf8(out2.clone()).unwrap(), "🦀");
        r2.emit(b"kept after truncated sequence", &mut out2);
        r2.emit(&[0xC3], &mut out2);
        r2.end(&mut out2);
        assert_eq!(String::from_utf8(out2).unwrap(), "🦀\u{FFFD}kept after truncated sequence\u{FFFD}");
    }
}
