//! Bounded speculative verification planner, multi-token acceptance budget accounting,
//! and greedy acceptance helpers (`PR #42` / `PR #80` / `t-audit-pr42-port`).
//!
//! Pure `std` logic with zero Metal dependency so `ServeStepPlanner` and scheduler tests
//! exercise verification arena bounds (`spans * (1 + k) <= lg_rows = 128`), context/reply
//! clamping, stop-in-middle truncation, and multi-token reply budget accounting directly.

#![allow(dead_code)]

use std::collections::VecDeque;

/// FNV-1a over the raw IEEE-754 bits of one vocabulary row.
pub fn logit_hash(row: &[f32]) -> u64 {
    let mut h = 0u64;
    for &x in row {
        h = (h ^ u64::from(x.to_bits())).wrapping_mul(0x100000001b3);
    }
    h
}

/// One lane ready for speculative verification. `lane` is the caller's lane index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifyCand {
    pub lane: usize,
    pub p0: usize,
    pub k: usize,
    pub kv_cap: usize,
}

/// One verification span packed into the current GPU verify step (`t = 1 + k`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifySpanPlan {
    pub lane: usize,
    pub p0: usize,
    pub k: usize,
}

/// Largest `k` such that `p0 + 1 + k <= kv_cap`, or `None` when the committed token at `p0`
/// itself does not fit the lane's K/V capacity.
pub fn clamp_k(p0: usize, k: usize, kv_cap: usize) -> Option<usize> {
    let room = kv_cap.saturating_sub(p0);
    if room == 0 {
        return None;
    }
    Some(k.min(room - 1))
}

/// Remaining tokens a lane may still emit before hitting either `max` generated tokens or
/// `ctx - 1` (`prompt + gen + 1 <= ctx`).
pub fn reply_budget(prompt: usize, gen: usize, max: usize, ctx: usize) -> usize {
    max.saturating_sub(gen)
        .min(ctx.saturating_sub(prompt.saturating_add(gen).saturating_add(1)))
}

/// Pack verify spans into one GPU verify step bounded by `rows` (`Engine::verify_row_capacity()`,
/// at most `MAX_SPANS = 128` vocabulary rows in `b.lg`).
///
/// - Each span position consumes one activation row and one vocabulary row (`t = 1 + k`), so
///   `spans.len() * (1 + k) <= rows` always holds (preventing the 256-into-128 row overflow).
/// - A span that does not fit the remaining `rows` budget stops packing so that lane and all
///   lanes behind it wait in order for the next step; verify spans are never split or mixed
///   into prefill.
/// - Lanes in one `encode_verify` call share one `k`.
/// - Without `multi` (`!hgemm`), at most one span runs per step.
pub fn plan_verify(cands: &[VerifyCand], rows: usize, multi: bool) -> Vec<VerifySpanPlan> {
    let mut budget = rows;
    let mut out = Vec::new();
    if rows == 0 {
        return out;
    }
    for c in cands {
        if !multi && !out.is_empty() {
            break;
        }
        let Some(k) = clamp_k(c.p0, c.k.min(rows - 1), c.kv_cap) else {
            continue;
        };
        let t = 1 + k;
        if t > budget {
            break;
        }
        if out.first().is_some_and(|p: &VerifySpanPlan| p.k != k) {
            break;
        }
        budget -= t;
        out.push(VerifySpanPlan {
            lane: c.lane,
            p0: c.p0,
            k,
        });
    }
    out
}

/// Result of greedy (`temp == 0`) verification over `k` draft tokens and `k + 1` target argmaxes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GreedyKeep {
    pub emitted: Vec<u32>,
    pub n_keep: usize,
    pub finished: bool,
    pub rows: Vec<usize>,
    /// Token past the accepted drafts (`None` when a stop token inside the accepted drafts ended the block).
    pub bonus: Option<u32>,
}

/// Greedy verification (`temperature == 0`).
/// A draft is kept iff it equals that row's target argmax (`rows[i] == drafts[i]`).
/// On mismatch, emits the target's correction token `rows[n_acc]` and stops.
/// If an accepted draft is in `stop`, emits only through that stop token with `bonus = None`.
pub fn accept_greedy(drafts: &[u32], rows: &[u32], stop: &[u32]) -> GreedyKeep {
    assert_eq!(
        rows.len(),
        drafts.len() + 1,
        "one argmax per draft, plus the bonus row"
    );
    let mut emitted = Vec::new();
    let mut used = Vec::new();
    let mut n_acc = 0usize;
    for i in 0..drafts.len() {
        if rows[i] != drafts[i] {
            break;
        }
        emitted.push(drafts[i]);
        used.push(i);
        n_acc += 1;
        if stop.contains(&drafts[i]) {
            return GreedyKeep {
                emitted,
                n_keep: 1 + n_acc,
                finished: true,
                rows: used,
                bonus: None,
            };
        }
    }
    let bonus = rows[n_acc];
    emitted.push(bonus);
    used.push(n_acc);
    GreedyKeep {
        emitted,
        n_keep: 1 + n_acc,
        finished: stop.contains(&bonus),
        rows: used,
        bonus: Some(bonus),
    }
}

/// Multi-token emission accounting for a verified lane (`emit_q` + `reply_budget`).
/// Ensures up to `min(accepted + 1, remaining_max_tokens)` tokens are emitted per verify step,
/// never stopping after 1 token when `reply_budget == 2`.
#[derive(Clone, Debug)]
pub struct SpecEmitState {
    pub pf_end: usize,
    pub gen: usize,
    pub max_tokens: usize,
    pub ctx: usize,
    pub emit_q: VecDeque<u32>,
}

impl SpecEmitState {
    pub fn new(pf_end: usize, max_tokens: usize, ctx: usize) -> Self {
        Self {
            pf_end,
            gen: 0,
            max_tokens,
            ctx,
            emit_q: VecDeque::new(),
        }
    }

    /// Remaining reply budget at the current `(pf_end, gen)` cursor.
    pub fn remaining_budget(&self) -> usize {
        reply_budget(self.pf_end, self.gen, self.max_tokens, self.ctx)
    }

    /// Clamp a candidate draft slice `drafts` so `1 + k <= remaining_budget()`.
    /// With `remaining_budget == 2`, `k == 1` (`1` draft + `1` bonus = `2` emitted tokens).
    pub fn clamp_draft_count(&self, drafts_len: usize) -> usize {
        drafts_len.min(self.remaining_budget().saturating_sub(1))
    }

    /// Emit one token through `send_tok` (`false` if client disconnected), returning
    /// `Some(finish_reason)` when the lane finishes (`"cancelled"`, `"stop"`, or `"length"`).
    pub fn emit_token(
        &mut self,
        tok: u32,
        stop: &[u32],
        mut send_tok: impl FnMut(u32) -> bool,
    ) -> Option<&'static str> {
        let ok = send_tok(tok);
        self.gen += 1;
        let budget = self.remaining_budget();
        if !ok {
            Some("cancelled")
        } else if stop.contains(&tok) {
            Some("stop")
        } else if budget == 0 {
            Some("length")
        } else {
            None
        }
    }

    /// Drain all queued tokens in `emit_q`, stopping immediately on disconnect, stop token,
    /// or exhausted `reply_budget`.
    pub fn drain_queue(
        &mut self,
        stop: &[u32],
        mut send_tok: impl FnMut(u32) -> bool,
    ) -> Option<&'static str> {
        while let Some(tok) = self.emit_q.pop_front() {
            if let Some(reason) = self.emit_token(tok, stop, &mut send_tok) {
                self.emit_q.clear();
                return Some(reason);
            }
        }
        None
    }
}

#[cfg(test)]
mod verify_planner_tests {
    use super::*;
    use crate::scheduler::{ServeLaneInput, ServeStepPlan, ServeStepPlanner};

    #[test]
    fn verify_budget_waits_and_clamps_to_capacity() {
        let c = |lane, p0, k, kv_cap| VerifyCand { lane, p0, k, kv_cap };
        // Two spans of 4, budget 4: the second does not fit and waits.
        let p = plan_verify(&[c(0, 10, 3, 100), c(1, 10, 3, 100)], 4, true);
        assert_eq!(p, vec![VerifySpanPlan { lane: 0, p0: 10, k: 3 }]);
        // Without hgemm a second lane waits even when budget remains.
        let p = plan_verify(&[c(0, 1, 1, 100), c(1, 1, 1, 100)], 100, false);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].lane, 0);
        // Near K/V boundary: p0=60, cap=64, k=7 -> k=3 because 60+1+3 == 64.
        let p = plan_verify(&[c(0, 60, 7, 64)], 4096, false);
        assert_eq!(p, vec![VerifySpanPlan { lane: 0, p0: 60, k: 3 }]);
        assert!(60 + 1 + p[0].k <= 64);
        // Only committed token fits: k clamps to 0 (t=1).
        let p = plan_verify(&[c(0, 63, 7, 64)], 4096, true);
        assert_eq!(p[0].k, 0);
        // Committed token past cache: lane skipped cleanly without calling encode.
        assert!(plan_verify(&[c(0, 64, 1, 64)], 4096, true).is_empty());
        // Different k is not packed into a step that already fixed k.
        let p = plan_verify(&[c(0, 10, 3, 100), c(1, 10, 1, 100)], 100, true);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].k, 3);
    }

    #[test]
    fn verify_logit_arena_bounds_256_requested_rows_to_128() {
        let cands: Vec<_> = (0..64)
            .map(|lane| VerifyCand {
                lane,
                p0: 10,
                k: 3,
                kv_cap: 64,
            })
            .collect();
        // 64 lanes * (1 + 3) = 256 requested rows; 128-row arena packs 32 spans (128 rows).
        let plans = plan_verify(&cands, 128, true);
        assert_eq!(plans.len(), 32);
        assert_eq!(plans.iter().map(|p| p.k + 1).sum::<usize>(), 128);
        let waiting = plan_verify(&cands[32..], 128, true);
        assert_eq!(waiting.len(), 32);
        assert_eq!(waiting.first().unwrap().lane, 32);
        assert_eq!(waiting.last().unwrap().lane, 63);
        assert_eq!(
            plan_verify(&cands, 2, true),
            vec![VerifySpanPlan {
                lane: 0,
                p0: 10,
                k: 1
            }]
        );
    }

    #[test]
    fn final_verify_block_emits_full_two_token_reply_budget() {
        for (max_tokens, ctx, stop, want_reason) in [
            (2usize, 64usize, vec![], "length"),
            (10, 63, vec![], "length"),
            (2, 64, vec![8u32], "stop"),
        ] {
            let mut st = SpecEmitState::new(60, max_tokens, ctx);
            assert_eq!(st.remaining_budget(), 2);
            assert_eq!(st.clamp_draft_count(4), 1);
            let keep = accept_greedy(&[7], &[7, 8], &stop);
            assert_eq!(keep.emitted, vec![7, 8]);
            st.emit_q.extend(keep.emitted);
            let mut delivered = Vec::new();
            let reason = st.drain_queue(&stop, |tok| {
                delivered.push(tok);
                true
            });
            assert_eq!(delivered, vec![7, 8]);
            assert_eq!(st.gen, 2);
            assert_eq!(reason, Some(want_reason));
        }
    }

    #[test]
    fn verify_stop_in_middle_and_mismatch_semantics() {
        let g = accept_greedy(&[1u32, 7, 2], &[1u32, 7, 2, 9], &[7]);
        assert_eq!(g.emitted, vec![1, 7]);
        assert!(g.finished);
        assert_eq!(g.bonus, None);
        assert_eq!(g.n_keep, 3);

        let g = accept_greedy(&[1, 99, 3], &[1, 2, 3, 4], &[]);
        assert_eq!(g.emitted, vec![1, 2]);
        assert_eq!(g.bonus, Some(2));
        assert!(!g.finished);
        assert_eq!(g.n_keep, 2);

        let g = accept_greedy(&[1, 2, 3], &[1, 2, 3, 8], &[]);
        assert_eq!(g.emitted, vec![1, 2, 3, 8]);
        assert_eq!(g.bonus, Some(8));
        assert_eq!(g.n_keep, 4);
        assert_eq!(g.rows, vec![0, 1, 2, 3]);

        let g = accept_greedy(&[1, 2], &[1, 2, 7], &[7]);
        assert_eq!(g.emitted, vec![1, 2, 7]);
        assert!(g.finished);
        assert_eq!(g.bonus, Some(7));
    }

    #[test]
    fn scheduler_interleaves_prefill_and_bounded_verify_without_mixing_spans() {
        // 3 lanes: lanes 0 and 1 finish prefill and enter generation with drafts (verify),
        // while lane 2 arrives with a multi-chunk prompt and ages until preemption.
        let mut planner = ServeStepPlanner::new(3, 1);
        planner.on_admit(0);
        planner.on_admit(1);
        // Step 1: prefill lanes 0 & 1 to completion (16 tokens each).
        let p1 = planner
            .plan_and_begin_step(
                &[
                    ServeLaneInput { runnable: true, n_past: 0, pf_end: 16, has_pending_decode: false },
                    ServeLaneInput { runnable: true, n_past: 0, pf_end: 16, has_pending_decode: false },
                    ServeLaneInput { runnable: false, n_past: 0, pf_end: 0, has_pending_decode: false },
                ],
                64,
                64,
            )
            .unwrap();
        assert_eq!(p1, ServeStepPlan::Prefill(vec![(0, 0, 16), (1, 0, 16)]));
        assert!(planner.complete_step().is_empty());

        // Admit lane 2 needing prefill (128 tokens), while lanes 0 & 1 are ready to verify/decode.
        planner.on_admit(2);
        let p2 = planner
            .plan_and_begin_step(
                &[
                    ServeLaneInput { runnable: true, n_past: 16, pf_end: 16, has_pending_decode: true },
                    ServeLaneInput { runnable: true, n_past: 16, pf_end: 16, has_pending_decode: true },
                    ServeLaneInput { runnable: true, n_past: 0, pf_end: 128, has_pending_decode: false },
                ],
                64,
                128,
            )
            .unwrap();
        // Generation burst runs lanes 0 & 1 without mixing lane 2's prefill into the step;
        // verify planner packs both spans within the 128-row arena.
        assert_eq!(p2, ServeStepPlan::Decode(vec![0, 1]));
        let cands = [
            VerifyCand { lane: 0, p0: 15, k: 3, kv_cap: 64 },
            VerifyCand { lane: 1, p0: 15, k: 3, kv_cap: 64 },
        ];
        let vplans = plan_verify(&cands, 128, true);
        assert_eq!(
            vplans,
            vec![
                VerifySpanPlan { lane: 0, p0: 15, k: 3 },
                VerifySpanPlan { lane: 1, p0: 15, k: 3 },
            ]
        );
        assert!(planner.complete_step().is_empty());
    }
}
