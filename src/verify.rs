//! Engine-backed speculative verification driver, draft producers, and Layer 3 paged
//! verification harness (`PR #42` / `PR #80` / `t-audit-pr42-port`).

use crate::blocks::{BlockIndex, BlockKind, BlockPools, BLOCK, WINDOW};
use crate::engine::{Engine, Lane, Span};
use crate::metal::wait_completed;
use crate::sampler::{checked_argmax, Sampler};
use crate::scheduler::{accept_greedy, logit_hash, plan_verify, SpecEmitState, VerifyCand};
use objc2_metal::MTLCommandBuffer;

/// Pluggable draft token producer for speculative verification (`t-audit-pr42-port`).
pub trait DraftProducer {
    /// Propose up to `max_k` draft tokens following `history` (whose last element is the
    /// committed token at `p0`).
    fn propose(&mut self, history: &[u32], max_k: usize) -> Vec<u32>;
}

/// Deterministic fixed draft producer for synthetic gate & fault-injection tests.
#[cfg_attr(not(test), allow(dead_code))]
pub struct FixedDraftProducer {
    pub drafts: Vec<u32>,
}

impl DraftProducer for FixedDraftProducer {
    fn propose(&mut self, _history: &[u32], max_k: usize) -> Vec<u32> {
        self.drafts[..self.drafts.len().min(max_k)].to_vec()
    }
}

/// Prompt/context suffix n-gram draft producer: scans `history` from most recent to oldest
/// for the longest matching suffix (3-gram down to 1-gram) and proposes the immediately
/// following `1..=max_k` tokens with zero extra model weights or GPU dispatches.
pub struct NgramDraftProducer {
    pub max_order: usize,
    pub fallback_tok: u32,
}

impl NgramDraftProducer {
    pub fn new(max_order: usize, fallback_tok: u32) -> Self {
        Self { max_order: max_order.max(1), fallback_tok }
    }
}

impl DraftProducer for NgramDraftProducer {
    fn propose(&mut self, history: &[u32], max_k: usize) -> Vec<u32> {
        if max_k == 0 || history.is_empty() {
            return Vec::new();
        }
        let max_n = self.max_order.min(history.len().saturating_sub(1));
        for order in (1..=max_n).rev() {
            let needle = &history[history.len() - order..];
            let search_end = history.len() - order;
            for start in (0..search_end).rev() {
                if &history[start..start + order] == needle {
                    let avail = (history.len() - (start + order)).min(max_k);
                    if avail > 0 {
                        return history[start + order..start + order + avail].to_vec();
                    }
                }
            }
        }
        vec![self.fallback_tok; max_k]
    }
}

/// Stochastic draft producer backed by `Sampler::draft` over per-position draft logits,
/// sharing `(logit_desc, token_id_asc)` `top_k` tie-breaking with target verification.
#[allow(dead_code)]
pub struct SamplerDraftProducer<'a> {
    pub draft_sampler: &'a mut Sampler,
    pub draft_logits: &'a [&'a [f32]],
}

impl DraftProducer for SamplerDraftProducer<'_> {
    fn propose(&mut self, _history: &[u32], max_k: usize) -> Vec<u32> {
        let n = self.draft_logits.len().min(max_k);
        let mut out = Vec::with_capacity(n);
        for row in &self.draft_logits[..n] {
            match self.draft_sampler.sample_next(row) {
                Ok(tok) => out.push(tok),
                Err(_) => break,
            }
        }
        out
    }
}

/// One lane submitted to [`verify_step`] or [`verify_step_sampled`].
/// `cs_before` is the convolution state from immediately before `p0`.
pub struct VerifyIn<'a> {
    pub lane: &'a Lane,
    pub p0: usize,
    pub cs_before: &'a [u8],
    pub drafts: &'a [u32],
}

/// Outcome of one lane in a completed verify step (`None` means the lane waited for a later step).
#[derive(Clone, Debug)]
pub struct VerifyKept {
    pub p0: usize,
    pub k: usize,
    pub emitted: Vec<u32>,
    pub n_keep: usize,
    pub finished: bool,
    pub hashes: Vec<u64>,
    pub bonus: Option<u32>,
}

/// Execute one greedy (`temp == 0`) speculative verification step over Layer 2 contiguous or
/// Layer 3 paged lanes (`spans * (1 + k) <= eng.verify_row_capacity()`, checking `wait_completed`
/// and rolling back `n_keep = 0` on any Metal failure).
pub fn verify_step(eng: &Engine, lanes: &[VerifyIn<'_>], stop: &[u32]) -> Vec<Option<Result<VerifyKept, String>>> {
    let cands: Vec<VerifyCand> = lanes
        .iter()
        .enumerate()
        .map(|(i, v)| VerifyCand { lane: i, p0: v.p0, k: v.drafts.len(), kv_cap: eng.on(v.lane).kv_capacity() })
        .collect();
    let plans = plan_verify(&cands, eng.verify_row_capacity(), eng.hgemm);
    let mut out = vec![None; lanes.len()];
    if plans.is_empty() {
        return out;
    }
    let k = plans[0].k;
    let spans: Vec<Span<'_>> = plans.iter().map(|p| Span { lane: lanes[p.lane].lane, p0: p.p0, t: 1 + k }).collect();
    let cb = eng.gpu.cmd();
    // Snapshot lane KV + CS + tokens BEFORE staging draft tokens and pre-p0 CS into shared buffers.
    eng.encode_verify(&cb, &spans, k);
    for p in &plans {
        let v = &lanes[p.lane];
        let run = eng.on(v.lane);
        run.set_cs(v.cs_before);
        if k > 0 {
            run.set_tokens(p.p0 + 1, &v.drafts[..k]);
        }
    }
    cb.commit();
    let gpu = wait_completed(&cb, "verify", plans[0].p0);
    for (si, p) in plans.iter().enumerate() {
        let result = gpu.clone().and_then(|()| {
            let base = si * (1 + k);
            let logits: Vec<Vec<f32>> = (0..=k).map(|r| eng.span_logits(base + r).to_vec()).collect();
            let argmaxes: Result<Vec<u32>, String> =
                logits.iter().map(|row| checked_argmax(row).map_err(String::from)).collect();
            let keep = accept_greedy(&lanes[p.lane].drafts[..k], &argmaxes?, stop);
            let hashes = keep.rows.iter().map(|&r| logit_hash(&logits[r])).collect();
            Ok(VerifyKept {
                p0: p.p0,
                k,
                emitted: keep.emitted,
                n_keep: keep.n_keep,
                finished: keep.finished,
                hashes,
                bonus: keep.bonus,
            })
        });
        eng.rollback_verify(lanes[p.lane].lane, si, result.as_ref().map_or(0, |kept| kept.n_keep));
        out[p.lane] = Some(result);
    }
    out
}

/// Execute one stochastic (`temp > 0` or `temp == 0`) speculative verification step for a lane
/// using `Sampler::verify_round` over target logits and the draft model's `draft_logits`,
/// preserving exact target distribution and `(logit_desc, token_id_asc)` `top_k` tie-breaking.
pub fn verify_step_sampled(
    eng: &Engine,
    v: &VerifyIn<'_>,
    draft_logits: &[&[f32]],
    sampler: &mut Sampler,
    stop: &[u32],
) -> Result<Option<VerifyKept>, String> {
    assert_eq!(draft_logits.len(), v.drafts.len());
    let cands = [VerifyCand { lane: 0, p0: v.p0, k: v.drafts.len(), kv_cap: eng.on(v.lane).kv_capacity() }];
    let plans = plan_verify(&cands, eng.verify_row_capacity(), eng.hgemm);
    let Some(p) = plans.first().copied() else {
        return Ok(None);
    };
    let k = p.k;
    let cb = eng.gpu.cmd();
    eng.encode_verify(&cb, &[Span { lane: v.lane, p0: p.p0, t: 1 + k }], k);
    let run = eng.on(v.lane);
    run.set_cs(v.cs_before);
    if k > 0 {
        run.set_tokens(p.p0 + 1, &v.drafts[..k]);
    }
    cb.commit();
    if let Err(err) = wait_completed(&cb, "verify", p.p0) {
        eng.rollback_verify(v.lane, 0, 0);
        return Err(err);
    }
    let logits: Vec<Vec<f32>> = (0..=k).map(|r| eng.span_logits(r).to_vec()).collect();
    let argmaxes: Result<Vec<u32>, String> =
        logits.iter().map(|row| checked_argmax(row).map_err(String::from)).collect();
    let argmaxes = argmaxes?;
    if sampler.greedy() {
        let keep = accept_greedy(&v.drafts[..k], &argmaxes, stop);
        let hashes = keep.rows.iter().map(|&r| logit_hash(&logits[r])).collect();
        eng.rollback_verify(v.lane, 0, keep.n_keep);
        return Ok(Some(VerifyKept {
            p0: p.p0,
            k,
            emitted: keep.emitted,
            n_keep: keep.n_keep,
            finished: keep.finished,
            hashes,
            bonus: keep.bonus,
        }));
    }
    let target_refs: Vec<&[f32]> = logits.iter().map(Vec::as_slice).collect();
    let raw_emitted = sampler.verify_round(&v.drafts[..k], &target_refs, &draft_logits[..k]);
    let mut emitted = Vec::with_capacity(raw_emitted.len());
    let mut rows = Vec::with_capacity(raw_emitted.len());
    let mut finished = false;
    let mut stopped_in_drafts = false;
    for (idx, &tok) in raw_emitted.iter().enumerate() {
        emitted.push(tok);
        rows.push(idx);
        if stop.contains(&tok) {
            finished = true;
            if idx + 1 < raw_emitted.len() {
                stopped_in_drafts = true;
            }
            break;
        }
    }
    let n_keep = if stopped_in_drafts { 1 + emitted.len() } else { emitted.len() };
    let bonus = if stopped_in_drafts { None } else { emitted.last().copied() };
    let hashes = rows.iter().map(|&r| logit_hash(&logits[r])).collect();
    eng.rollback_verify(v.lane, 0, n_keep);
    Ok(Some(VerifyKept { p0: p.p0, k, emitted, n_keep, finished, hashes, bonus }))
}

/// Layer 3 paged lane wrapper for speculative verification + rollback across `Window`, `Full`,
/// and recurrent `cs` state (`t-audit-pr42-port`).
///
/// Tracks newly allocated unsealed draft blocks so a rejected tail releases blocks strictly past
/// `accepted_end`, keeps `Window` blocks live across `EXT = 1024` until `accepted_end` actually
/// passes the window, and never seals a rejected block.
pub struct PagedVerifyLane<'a> {
    pub eng: &'a Engine,
    pub lane: Lane,
    pub written: usize,
}

impl<'a> PagedVerifyLane<'a> {
    pub fn new(eng: &'a Engine, cap: usize) -> Self {
        let lane = Lane::new_blocked(eng, cap);
        Self { eng, lane, written: 0 }
    }

    fn take_slot(pools: &mut BlockPools, kind: BlockKind) -> (crate::blocks::Slot, u32) {
        let pool = pools.of_mut(kind);
        let slot = match pool.lowest_free() {
            Some(s) => s,
            None => pool.grow().expect("grow Layer 3 block pool for verify"),
        };
        let _ = pool.take(slot).expect("take zeroed slot for verify");
        let gen = pool.record(slot).unwrap().generation();
        (slot, gen)
    }

    /// Ensure unsealed `Full` and `Window` blocks exist in `self.lane.block_table` for `[p0, p0 + t)`,
    /// returning the list of `(kind, BlockIndex)` entries newly allocated for this span.
    pub fn ensure_span_blocks(&self, p0: usize, t: usize) -> Vec<(BlockKind, BlockIndex)> {
        let lo_blk = p0 / BLOCK;
        let hi_blk = (p0 + t - 1) / BLOCK;
        let mut allocated = Vec::new();
        let mut pools = self.eng.block_pool().borrow_mut();
        let mut tbl = self.lane.block_table().borrow_mut();
        for b in lo_blk..=hi_blk {
            let bi = BlockIndex::new(b as u32);
            for kind in [BlockKind::Full, BlockKind::Window] {
                if let Some(entry) = tbl.get(kind, bi).expect("block in table range") {
                    let rec = pools.of(kind).record(entry.slot()).expect("slot record");
                    assert_eq!(rec.generation(), entry.generation(), "stale block entry");
                    assert!(!rec.sealed(), "verify/step must never write into a sealed {kind:?} block");
                } else {
                    let (slot, gen) = Self::take_slot(&mut pools, kind);
                    tbl.fill(pools.of_mut(kind), bi, slot, gen).expect("fill newly allocated verify block");
                    allocated.push((kind, bi));
                }
            }
        }
        allocated
    }

    /// Run one 1-token reference/prefill step at `pos` (`encode_spans` of `t = 1`), updating
    /// `SlotRecord::length` and pruning expired `Window` blocks past `WINDOW = 1024`.
    pub fn step_one(&mut self, pos: usize, logits: bool) -> Result<(), String> {
        let _ = self.ensure_span_blocks(pos, 1);
        let cb = self.eng.gpu.cmd();
        self.eng.encode_spans(&cb, &[Span { lane: &self.lane, p0: pos, t: 1 }], logits, false);
        cb.commit();
        wait_completed(&cb, "paged step1", pos)?;
        let end = pos + 1;
        self.written = end;
        let b = pos / BLOCK;
        let bi = BlockIndex::new(b as u32);
        let block_len = (end - b * BLOCK).min(BLOCK) as u32;
        {
            let tbl = self.lane.block_table().borrow();
            let mut pools = self.eng.block_pool().borrow_mut();
            for kind in [BlockKind::Full, BlockKind::Window] {
                let e = tbl.get(kind, bi).unwrap().expect("active block entry");
                pools.of_mut(kind).record_mut(e.slot()).unwrap().set_length(block_len).expect("set block length");
            }
        }
        self.prune_expired_window(end);
        Ok(())
    }

    /// Reconcile `BlockTable` and `BlockPools` after `rollback_verify` commits `accepted_end`:
    /// releases any newly allocated draft blocks strictly past `accepted_end`, leaves retained
    /// blocks unsealed with exact `SlotRecord::length`, and prunes expired `Window` blocks ONLY
    /// up to `accepted_end` (never to the rejected draft end!).
    pub fn finish_verify_rollback(
        &mut self,
        allocated: &[(BlockKind, BlockIndex)],
        pre_written: usize,
        accepted_end: usize,
    ) {
        let target_end = if accepted_end == 0 { pre_written } else { accepted_end };
        {
            let mut pools = self.eng.block_pool().borrow_mut();
            let mut tbl = self.lane.block_table().borrow_mut();
            for &(kind, bi) in allocated {
                let blk_start = (bi.get() as usize) * BLOCK;
                if blk_start >= target_end {
                    match kind {
                        BlockKind::Full => tbl.clear_kind_entry(&mut pools.full, bi).unwrap(),
                        BlockKind::Window => tbl.clear_kind_entry(&mut pools.window, bi).unwrap(),
                    }
                }
            }
            if target_end > 0 {
                let active_b = (target_end - 1) / BLOCK;
                let bi = BlockIndex::new(active_b as u32);
                let block_len = (target_end - active_b * BLOCK).min(BLOCK) as u32;
                for kind in [BlockKind::Full, BlockKind::Window] {
                    if let Some(e) = tbl.get(kind, bi).unwrap() {
                        let rec = pools.of_mut(kind).record_mut(e.slot()).unwrap();
                        assert!(!rec.sealed(), "verify must never seal active/rejected block");
                        rec.set_length(block_len).unwrap();
                    }
                }
            }
        }
        self.written = target_end;
        // Sync any cleared draft-block GPU table entries back to `CLEARED_GPU_TABLE_ENTRY` (`u64::MAX`).
        if target_end > 0 {
            self.eng.sync_lane_block_table(&self.lane, target_end - 1, 1);
        }
        self.prune_expired_window(target_end);
    }

    pub fn prune_expired_window(&self, end_pos: usize) {
        if end_pos == 0 {
            return;
        }
        let min_win_blk = end_pos.saturating_sub(WINDOW) / BLOCK;
        if min_win_blk == 0 {
            return;
        }
        let mut pools = self.eng.block_pool().borrow_mut();
        let mut tbl = self.lane.block_table().borrow_mut();
        for b in 0..min_win_blk {
            let bi = BlockIndex::new(b as u32);
            if tbl.get(BlockKind::Window, bi).unwrap().is_some() {
                tbl.clear_kind_entry(&mut pools.window, bi).unwrap();
            }
        }
    }

    pub fn release(self) {
        let mut pools = self.eng.block_pool().borrow_mut();
        self.lane.block_table().borrow_mut().clear_entries(&mut pools).unwrap();
    }
}

/// Drive a multi-step speculative loop with `DraftProducer` and `SpecEmitState` on a Layer 2
/// lane, returning `(emitted_tokens, finish_reason)` for end-to-end verification against 1-token
/// sequential execution.
pub fn run_speculative_loop(
    eng: &Engine,
    prompt: &[u32],
    max_tokens: usize,
    ctx: usize,
    k_target: usize,
    stop: &[u32],
    drafter: &mut dyn DraftProducer,
) -> Result<(Vec<u32>, &'static str), String> {
    assert!(!prompt.is_empty());
    let lane = Lane::new(eng);
    let run = eng.on(&lane);
    run.reserve(ctx, false);
    run.set_tokens(0, prompt);
    for pos in 0..prompt.len() - 1 {
        let cb = eng.gpu.cmd();
        eng.encode_spans(&cb, &[Span { lane: &lane, p0: pos, t: 1 }], false, false);
        cb.commit();
        wait_completed(&cb, "spec prefill", pos)?;
    }
    let mut cs_before = run.get_cs();
    let mut p0 = prompt.len() - 1;
    {
        let cb = eng.gpu.cmd();
        eng.encode_spans(&cb, &[Span { lane: &lane, p0, t: 1 }], false, false);
        cb.commit();
        wait_completed(&cb, "spec anchor", p0)?;
    }
    let mut history = prompt.to_vec();
    let mut emit_state = SpecEmitState::new(prompt.len(), max_tokens, ctx);
    let mut delivered = Vec::new();
    loop {
        let k_max = emit_state.clamp_draft_count(k_target);
        let drafts = drafter.propose(&history, k_max);
        let res = verify_step(eng, &[VerifyIn { lane: &lane, p0, cs_before: &cs_before, drafts: &drafts }], stop)
            .pop()
            .unwrap()
            .ok_or_else(|| String::from("verify planner made no progress"))??;
        emit_state.emit_q.extend(res.emitted.iter().copied());
        if let Some(reason) = emit_state.drain_queue(stop, |tok| {
            delivered.push(tok);
            history.push(tok);
            true
        }) {
            return Ok((delivered, reason));
        }
        // Encode the accepted bonus token at `p0 + n_keep` while capturing `cs_before` for the next verify round.
        let bonus = res.bonus.expect("unfinished verify step has bonus token");
        let next_p0 = res.p0 + res.n_keep;
        cs_before = run.get_cs();
        run.set_tokens(next_p0, &[bonus]);
        let cb = eng.gpu.cmd();
        eng.encode_spans(&cb, &[Span { lane: &lane, p0: next_p0, t: 1 }], false, false);
        cb.commit();
        wait_completed(&cb, "spec bonus", next_p0)?;
        p0 = next_p0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_draft_producer_proposes_longest_matching_suffix_continuation() {
        let mut fixed = FixedDraftProducer { drafts: vec![11, 22, 33] };
        assert_eq!(fixed.propose(&[1, 2], 2), vec![11, 22]);
        assert_eq!(fixed.propose(&[1, 2], 5), vec![11, 22, 33]);

        let mut prod = NgramDraftProducer::new(3, 99);
        // History has pattern `[10, 20, 30, 40, 50]` earlier and ends with `[20, 30]`.
        let hist = [5u32, 10, 20, 30, 40, 50, 7, 20, 30];
        assert_eq!(prod.propose(&hist, 3), vec![40, 50, 7]);
        assert_eq!(prod.propose(&hist, 1), vec![40]);
        assert_eq!(prod.propose(&hist, 0), Vec::<u32>::new());
        // Unseen suffix falls back deterministically.
        assert_eq!(prod.propose(&[1, 2, 88], 2), vec![99, 99]);
    }
}
