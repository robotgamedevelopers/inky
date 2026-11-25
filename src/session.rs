//! A generation session: owns the context (tokens + KV/conv state on the GPU), prefill and decode loops.

use crate::engine::{Engine, Lane, Run};
use crate::metal::{wait_all, wait_completed, wait_or_drain};
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;
use crate::stream_text::stream_text;
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

/// Snapshot of a [`Session`] before an interactive CLI turn (`t-audit-engine-cli-step-status`),
/// allowing a failed prefill chunk or decode step to roll back K/V, convolution state, `n_past`,
/// `pending`, and `stats` cleanly for the next turn.
pub struct TurnCheckpoint {
    n_past: usize,
    pending: Option<u32>,
    stats: Stats,
    cs: Vec<u8>,
    window_kv: Vec<crate::engine::KvUndo>,
}

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

    /// Snapshot valid session context (`0..n_past`) before a chat turn so a failed Metal command
    /// buffer leaves the session state intact for the next turn.
    pub fn checkpoint(&self) -> TurnCheckpoint {
        let win_lo = self.n_past.saturating_sub(crate::blocks::WINDOW);
        let win_cnt = self.n_past - win_lo;
        let window_kv = if win_cnt > 0 && !self.run().tier_on() {
            self.eng.snapshot_kv_positions(&self.lane, win_lo, win_cnt)
        } else {
            Vec::new()
        };
        TurnCheckpoint {
            n_past: self.n_past,
            pending: self.pending,
            stats: self.stats,
            cs: self.run().get_cs(),
            window_kv,
        }
    }

    /// Restore `self` to `ckpt` after a failed turn. When `ckpt.n_past == 0` (or `--kv-disk` is
    /// active, where file blocks cannot be un-staged cheaply), resets the lane cleanly when at 0;
    /// otherwise restores `cs` and the live sliding-window K/V rows (`0..ckpt.n_past` in `kv_full`
    /// is never overwritten by writes at `>= ckpt.n_past`).
    pub fn rollback(&mut self, ckpt: TurnCheckpoint) {
        if ckpt.n_past == 0 {
            self.run().reset();
        } else {
            self.run().set_cs(&ckpt.cs);
            for undo in &ckpt.window_kv {
                self.eng.restore_kv_pos(&self.lane, undo);
            }
            self.eng.clear_kv_tail(&self.lane, ckpt.n_past);
        }
        self.n_past = ckpt.n_past;
        self.pending = ckpt.pending;
        self.stats = ckpt.stats;
    }

    /// Execute one complete chat turn (`prefill` + `generate` + `Renderer::end`), rolling back
    /// session state and closing any open ANSI styling if any Metal step or sample fails.
    pub fn run_chat_turn(
        &mut self,
        toks: &[u32],
        max_gen: usize,
        show_thinking: bool,
        raw: bool,
        sampler: &mut Sampler,
        out: &mut impl std::io::Write,
    ) -> Result<usize, String> {
        let ckpt = self.checkpoint();
        let first = match self.prefill(toks) {
            Ok(tok) => tok,
            Err(err) => {
                self.rollback(ckpt);
                return Err(err);
            }
        };
        let mut r = Renderer::new(show_thinking, raw);
        let max = max_gen.min(self.ctx_left().saturating_sub(4));
        let tk = self.tk;
        let res = self.generate(first, max, sampler, |t| {
            r.push(tk, t, out);
            true
        });
        r.end(out);
        if let Err(err) = res {
            self.rollback(ckpt);
            return Err(err);
        }
        res
    }

    /// Feed `toks` (preceded by any pending sampled token) through the model in prefill chunks.
    /// Returns the greedy next token (argmax of the last position); logits stay available in the engine.
    /// After an error, discard or reset the lane before using it for another inference.
    pub fn prefill(&mut self, toks: &[u32]) -> Result<u32, String> {
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
        let mut last = None;
        while p0 < end {
            let t = (end - p0).min(self.eng.chunk);
            let cb = self.eng.gpu.cmd();
            self.run().encode_prefill(&cb, p0, t, p0 + t == end, true);
            cb.commit();
            // Let Metal run the next chunk while the CPU checks the previous one.
            if let Some((pos, prev)) = last.replace((p0, cb)) {
                wait_or_drain((pos, &prev), last.as_ref().map(|(pos, cb)| (*pos, cb)),
                              |(pos, cb)| wait_completed(cb, "prefill", pos))?;
            }
            p0 += t;
        }
        if let Some((pos, cb)) = last {
            wait_completed(&cb, "prefill", pos)?;
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
    /// After an error, discard or reset the lane before using it for another inference.
    pub fn generate(&mut self, first: u32, max: usize, sampler: &mut Sampler, mut on_token: impl FnMut(u32) -> bool) -> Result<usize, String> {
        let t0 = Instant::now();
        let eng = self.eng.on(&self.lane);
        let eos = self.tk.eos;
        eng.release_batch();      // the prefill arena is not needed while generating
        if max == 0 { return Ok(0); }
        let mut tok = if sampler.greedy() {
            first
        } else {
            sampler
                .sample_next_with_ban_and_softcap(&eng.logits(), None, self.eng.effective_sampler_softcap(0))
                .map_err(|err| format!("model sampling failed: {err:?}"))?
        };
        let mut n = 1usize;
        let mut stop = tok == eos || !on_token(tok) || n >= max || self.ctx_left() < 2;
        if sampler.greedy() {
            if stop { self.pending = Some(tok); }
            else {
                eng.set_tokens(self.n_past, &[tok]);
                let mut inflight = VecDeque::with_capacity(2);
                let step = |pos: usize| (pos, eng.step(pos, true, true));
                eng.gen_reserve(self.n_past);
                inflight.push_back(step(self.n_past));
                loop {
                    if inflight.len() < 2 {
                        if let Some(&(last_pos, _)) = inflight.back() {
                            let pos = last_pos + 1;
                            if pos >= eng.gen_capacity() {
                                // the K/V buffers must grow (or, with --kv-disk, be re-sized): nothing may be in flight meanwhile
                                wait_all(&inflight, |(pos, cmd)| cmd.wait(*pos))?;
                                eng.gen_reserve(pos);
                            }
                            inflight.push_back(step(pos));
                        }
                    }
                    let Some((pos, cmd)) = inflight.pop_front() else { break };
                    wait_or_drain(&(pos, cmd), &inflight, |(pos, cmd)| cmd.wait(*pos))?;
                    self.n_past = pos + 1;
                    tok = eng.token_at(pos + 1);
                    n += 1;
                    stop = tok == eos || !on_token(tok) || n >= max || self.ctx_left() < 3;
                    if stop {
                        // the remaining in-flight step consumes `tok` (a real token): keep it in the context
                        match inflight.pop_front() {
                            Some((p2, cmd)) => {
                                cmd.wait(p2)?;
                                self.n_past = p2 + 1;
                                self.pending = None;
                            }
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
                eng.step(self.n_past, true, false).wait(self.n_past)?;
                self.n_past += 1;
                tok = sampler
                    .sample_next_with_ban_and_softcap(&eng.logits(), None, self.eng.effective_sampler_softcap(0))
                    .map_err(|err| format!("model sampling failed: {err:?}"))?;
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
            self.flush_text(true, out);
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
        self.flush_text(false, out);
    }

    fn flush_text(&mut self, end: bool, out: &mut impl std::io::Write) {
        stream_text(&mut self.buf, end, |text| {
            let _ = out.write_all(text.as_bytes());
            let _ = out.flush();
        });
    }

    pub fn end(&mut self, out: &mut impl std::io::Write) {
        self.flush_text(true, out);
        if self.thinking && self.show_thinking { let _ = write!(out, "\x1b[0m"); }
        let _ = out.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_bytes(renderer: &mut Renderer, tk: &Tokenizer, bytes: &[u8], out: &mut Vec<u8>) {
        for &byte in bytes {
            renderer.push(tk, tk.eos + 1 + u32::from(byte), out);
        }
    }

    #[test]
    fn renderer_recovers_valid_text_and_flushes_at_end() {
        let tk = Tokenizer::for_byte_test();
        for raw in [false, true] {
            let mut renderer = Renderer::new(false, raw);
            let mut out = Vec::new();
            push_bytes(&mut renderer, &tk, b"\xffvalid text\xe2", &mut out);
            assert_eq!(out, "�valid text".as_bytes());
            push_bytes(&mut renderer, &tk, b"\x82\xac\xf0\x9f", &mut out);
            assert_eq!(out, "�valid text€".as_bytes());
            renderer.end(&mut out);
            assert_eq!(out, "�valid text€�".as_bytes());
        }
    }

    #[test]
    fn renderer_does_not_join_utf8_across_content_boundaries() {
        let tk = Tokenizer::for_byte_test();
        let mut renderer = Renderer::new(false, false);
        let mut out = Vec::new();
        push_bytes(&mut renderer, &tk, b"\xe2", &mut out);
        for token in tk.encode("<|content_thinking|>hidden<|end_message|><|content_text|>", true) {
            renderer.push(&tk, token, &mut out);
        }
        push_bytes(&mut renderer, &tk, b"\x82\xactext", &mut out);
        renderer.end(&mut out);
        assert_eq!(out, "�\x1b[2m(thinking…)\x1b[0m ��text".as_bytes());
    }

    #[test]
    fn failed_command_drains_later_work_without_consuming_results() {
        let mut waited = Vec::new();
        let mut consumed = false;
        let result = wait_or_drain(0, [1, 2], |command| {
            waited.push(command);
            if command == 0 { Err("first command failed".into()) } else { Ok(()) }
        }).map(|_| consumed = true);
        assert_eq!(result, Err("first command failed".into()));
        assert_eq!(waited, [0, 1, 2]);
        assert!(!consumed);

        waited.clear();
        wait_or_drain(0, [1], |command| { waited.push(command); Ok(()) }).unwrap();
        assert_eq!(waited, [0]);
    }

    #[test]
    fn wait_all_preserves_first_error_and_checks_final_command() {
        for (failures, expected) in [
            ([true, true, false], Err("command 0".into())),
            ([false, false, true], Err("command 2".into())),
            ([false, false, false], Ok(())),
        ] {
            let mut waited = Vec::new();
            let result = wait_all(failures.into_iter().enumerate(), |(command, failed)| {
                waited.push(command);
                if failed { Err(format!("command {command}")) } else { Ok(()) }
            });
            assert_eq!(result, expected);
            assert_eq!(waited, [0, 1, 2]);
        }
    }

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

    #[test]
    #[ignore = "requires local Inkling-2-Mini GGUF on M5"]
    fn cli_session_recovers_from_injected_prefill_and_decode_failures() {
        use crate::metal::{clear_wait_failure, inject_wait_failure};

        let model = std::env::var("INKY_TEST_MODEL").unwrap_or_else(|_| crate::default_model());
        let mut eng = Engine::load(&model, 256);
        eng.chunk = 32;
        let tk = Tokenizer::load(&eng.g);

        let turn1 = tk.encode("<|message_user|><|content_text|>Count to three.<|end_message|><|message_model|>", true);
        // Make a 48-token turn-2 prompt so prefill splits into an early chunk (32) + final chunk (16 + 1 pending).
        let mut turn2 = tk.encode("<|message_user|><|content_text|>", true);
        while turn2.len() < 46 {
            turn2.extend(tk.encode(" apple banana cherry", false));
        }
        turn2.extend(tk.encode("<|end_message|><|message_model|>", true));
        assert!(turn2.len() > eng.chunk, "turn2 ({}) must span at least 2 prefill chunks ({})", turn2.len(), eng.chunk);

        // Reference clean 2-turn greedy run AND clean 2-turn sampled run.
        let run_clean = |temp: f32| -> (usize, u64, Vec<u8>) {
            let mut sess = Session::new(&eng, &tk);
            let mut s1 = Sampler::new(temp, 64, 0.95, eng.softcap, 12345);
            let mut out1 = Vec::new();
            sess.run_chat_turn(&turn1, 4, true, true, &mut s1, &mut out1).unwrap();
            let mut s2 = Sampler::new(temp, 64, 0.95, eng.softcap, 67890);
            let mut out2 = Vec::new();
            sess.run_chat_turn(&turn2, 4, true, true, &mut s2, &mut out2).unwrap();
            (sess.n_past, eng.lane_hash(&sess.lane), out2)
        };
        let (clean_greedy_npast, clean_greedy_hash, clean_greedy_out2) = run_clean(0.0);
        let (clean_samp_npast, clean_samp_hash, clean_samp_out2) = run_clean(0.8);

        // Exercise all 4 required failure points on turn 2 after turn 1 has already populated context:
        //  (1) early prefill chunk (`prefill`, call 0)
        //  (2) final prefill chunk (`prefill`, call 1)
        //  (3) greedy decode step (`decode`, call 0 — drains pipelined step 1 without consuming!)
        //  (4) sampled decode step (`decode`, call 0)
        for (label, temp, stage, call_idx, want_npast, want_hash, want_out2) in [
            ("early prefill chunk", 0.0f32, "prefill", 0usize, clean_greedy_npast, clean_greedy_hash, &clean_greedy_out2),
            ("final prefill chunk", 0.0f32, "prefill", 1usize, clean_greedy_npast, clean_greedy_hash, &clean_greedy_out2),
            ("greedy decode step", 0.0f32, "decode", 0usize, clean_greedy_npast, clean_greedy_hash, &clean_greedy_out2),
            ("sampled decode step", 0.8f32, "decode", 0usize, clean_samp_npast, clean_samp_hash, &clean_samp_out2),
        ] {
            let mut sess = Session::new(&eng, &tk);
            let mut s1 = Sampler::new(temp, 64, 0.95, eng.softcap, 12345);
            let mut out1 = Vec::new();
            sess.run_chat_turn(&turn1, 4, true, true, &mut s1, &mut out1).unwrap();
            let turn1_npast = sess.n_past;
            let turn1_cs = sess.run().get_cs();
            let turn1_kv: Vec<Vec<u8>> = eng
                .snapshot_kv_positions(&sess.lane, 0, turn1_npast)
                .into_iter()
                .map(|u| u.rows)
                .collect();

            // Inject device/command-buffer failure on turn 2: must return Err and leave session at turn 1 state!
            inject_wait_failure(stage, call_idx);
            let mut s_fail = Sampler::new(temp, 64, 0.95, eng.softcap, 67890);
            let mut fail_out = Vec::new();
            let err = sess.run_chat_turn(&turn2, 4, true, true, &mut s_fail, &mut fail_out);
            clear_wait_failure();
            assert!(err.is_err(), "{label}: injected {stage}[{call_idx}] failure was not reported");
            assert!(std::str::from_utf8(&fail_out).is_ok(), "{label}: output left invalid UTF-8");
            assert_eq!(sess.n_past, turn1_npast, "{label}: n_past mutated after failed turn");
            assert_eq!(sess.run().get_cs(), turn1_cs, "{label}: CS corrupted after failed turn");
            let after_kv: Vec<Vec<u8>> = eng
                .snapshot_kv_positions(&sess.lane, 0, turn1_npast)
                .into_iter()
                .map(|u| u.rows)
                .collect();
            assert_eq!(after_kv, turn1_kv, "{label}: committed KV[0..turn1_npast) corrupted after failed turn");

            // Retry turn 2 on the same session: must match the clean 2-turn session bit-for-bit!
            let mut s2 = Sampler::new(temp, 64, 0.95, eng.softcap, 67890);
            let mut retry_out = Vec::new();
            sess.run_chat_turn(&turn2, 4, true, true, &mut s2, &mut retry_out).unwrap();
            assert_eq!(sess.n_past, want_npast, "{label}: retry n_past mismatch");
            assert_eq!(eng.lane_hash(&sess.lane), want_hash, "{label}: retry lane_hash mismatch");
            assert_eq!(&retry_out, want_out2, "{label}: retry output mismatch");
        }
    }
}
