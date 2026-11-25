//! Layer 3's output gate: the block cache gives exactly layer 2's outputs while every pool slot the step does not
//! read holds poison.
//!
//! The reference is layer 2's path, each lane alone, nothing poisoned. The candidate runs the same lanes together in
//! steps, as serve does (one step of prefill spans, then one decode step), and before each step every pool slot that no
//! lane in that step references is overwritten with poison: NaN and fp16-max words. Slots that a live lane outside the
//! step references are saved first and put back after the step, so the poison reaches everything the step should not
//! read and destroys nothing a later step needs. Free slots stay poisoned: the pool must zero a slot when it hands it
//! out. Every step's logits and every greedy token must be bit-identical to the reference.
//!
//! What plugs in where (src/blocks.rs types):
//!  - `StepPath`: create a lane, step prefill spans, step decode spans, read logits, end a lane. `Layer2` is the
//!    per-lane contiguous reference path; `Layer3Pool` is the paged candidate over `Engine::block_pool`,
//!    `Lane::block_table`, and `EntryTree`.
//!  - `PoisonPool`: each kind's pool as the gate sees it. `slot_count(kind)` is `BlockPools::{window,full}.len()`;
//!    `lane_slots` is `table_slots(&lane.block_table().borrow())`; `slot_bytes(kind, slot)` is
//!    `BlockPool::slot_raw_region(slot)` over that kind's owning slab, at the pool's width. `Layer2` implements it with lane k's window
//!    cache as window slot k + 1 and its full cache as full slot k + 1, so the gate runs today with the other lanes'
//!    caches poisoned.
//!  - main.rs `poolgate` builds the `Layer2` reference and `ServePoolCandidate` (`ServeBlockCache`) and compares every step and token.
//!
//! The slot arithmetic (`poison_plan`, once per kind) is modelled in specs/PoisonGate.tla.
#[cfg(not(standalone_blocks))]
use crate::blockbook::{EVERY, HEAD, PROMPT};
#[cfg(not(standalone_blocks))]
use crate::blocks::BlockEntry;
use crate::blocks::{BlockIndex, BlockKind, BlockTable, Slot};
#[cfg(not(standalone_blocks))]
use crate::blocks::{BLOCK, WINDOW};
#[cfg(not(standalone_blocks))]
use crate::engine::{Engine, Lane, Span, EXT};
#[cfg(not(standalone_blocks))]
use crate::prefix::{BlockHandles, EntryTree, PrefixMatch, SealedId};
#[cfg(not(standalone_blocks))]
use objc2_metal::MTLCommandBuffer;
#[cfg(not(standalone_blocks))]
use std::cell::{Cell, RefCell};
#[cfg(not(standalone_blocks))]
use std::collections::HashMap;

/// A lane's handle, chosen by the path that made it.
pub type LaneId = usize;

pub const KINDS: [BlockKind; 2] = [BlockKind::Window, BlockKind::Full];

/// A way to run lanes through steps. Logits are read after the step that wrote them, before the next one.
pub trait StepPath {
    /// A new lane holding `toks` at positions 0.., with room for `cap` positions.
    fn new_lane(&self, toks: &[u32], cap: usize) -> LaneId;
    /// The token at `pos` of a lane (greedy decoding writes the next input here).
    fn set_token(&self, lane: LaneId, pos: usize, tok: u32);
    /// Pre-allocate any target blocks needed by `spans` before `poisoned` snapshots `lane_slots` and applies poison.
    fn prepare_prefill(&self, _spans: &[(LaneId, usize, usize)]) {}
    /// Pre-allocate any target blocks needed by `spans` before `poisoned` snapshots `lane_slots` and applies poison.
    fn prepare_decode(&self, _spans: &[(LaneId, usize)]) {}
    /// One step of prefill spans: (lane, p0, t), positions [p0, p0 + t) of each lane.
    fn prefill(&self, spans: &[(LaneId, usize, usize)]);
    /// One decode step: (lane, pos), one token for each lane.
    fn decode(&self, spans: &[(LaneId, usize)]);
    /// The logits of span `row` of the last step (its last position), as f32 in vocabulary order.
    fn logits(&self, row: usize) -> Vec<f32>;
    /// The lane is finished: its private blocks go back to the pool, its shared blocks lose a referrer.
    fn end_lane(&self, lane: LaneId);
}

/// Both pools, as the poison gate sees them. Nothing is in flight on the GPU while the gate calls these.
pub trait PoisonPool {
    /// Slots in `kind`'s pool: 1..=slot_count(kind).
    fn slot_count(&self, kind: BlockKind) -> usize;
    /// Every slot the lane's block table references now, of both kinds: sealed shared blocks, window blocks, its
    /// unsealed tail. Duplicates are allowed.
    fn lane_slots(&self, lane: LaneId) -> Vec<(BlockKind, Slot)>;
    /// The slot's bytes, CPU-visible. None for a slot that has no memory now (a layer-2 lane that ended).
    fn slot_bytes(&self, kind: BlockKind, slot: Slot) -> Option<(*mut u8, usize)>;
    /// True when `slot` below `slot_count` legitimately has no memory now (a layer-2 lane that ended). Any other slot
    /// with no memory is a bug in the pool and the gate fails on it, naming the slot.
    fn absent_ok(&self, _kind: BlockKind, _slot: Slot) -> bool { false }
    /// Called when `INK_POOLGATE_SABOTAGE` drops the guard on `lane`. A deduplicating candidate (`Layer3Pool`) uses
    /// this so the intentionally poisoned lane never publishes corrupted blocks into its shared prefix tree for other
    /// lanes to adopt; any remaining divergence on another lane proves a genuine cross-lane leak in the engine.
    fn note_sabotaged(&self, _lane: LaneId) {}
    /// Called immediately after `apply` saves non-stepping live slots and poisons unguarded slots.
    fn note_applied(&self, _stepping_lanes: &[LaneId], _saved: &[(BlockKind, Slot, Vec<u8>)]) {}
    /// Called immediately after `restore` copies saved slot bytes back into the pool.
    fn note_restored(&self) {}
    /// Slots held by retained cache checkpoints (not necessarily in any currently-stepping lane's `BlockTable`):
    /// still poisoned during every step that does not read them, and saved/restored around the step so subsequent
    /// waves can match retained start points while unreferenced (`refs == 0`) free slots remain poisoned.
    fn retained_slots(&self) -> Vec<(BlockKind, Slot)> { Vec::new() }
}

/// The candidate is a path with a pool.
pub trait Candidate: StepPath + PoisonPool {}
impl<T: StepPath + PoisonPool> Candidate for T {}

/// Every (kind, slot) a block table holds an entry for. Walks each kind's rows until the table says out of range.
pub fn table_slots(t: &BlockTable) -> Vec<(BlockKind, Slot)> {
    let mut v = Vec::new();
    for kind in KINDS {
        let mut b = 0u32;
        while let Ok(e) = t.get(kind, BlockIndex::new(b)) {
            if let Some(e) = e { assert_eq!(e.kind(), kind, "table row of kind {kind:?} holds a {:?} entry", e.kind()); v.push((kind, e.slot())); }
            b += 1;
        }
    }
    v
}

/// What to do to one kind's pool before one step, by 0-based pool index (`Slot::pool_index`).
#[derive(Debug, PartialEq, Eq)]
pub struct Plan {
    /// pool indices to overwrite with poison, ascending
    pub poison: Vec<usize>,
    /// the poisoned indices a live lane references: saved before the poison, put back after the step, ascending
    pub save: Vec<usize>,
}

/// Which slots of one kind's pool to poison and which to save, for one step. `tables[i]` is live lane i's pool indices
/// of that kind; `stepping` lists the live lanes in the step. A slot is protected when any stepping lane references it
/// (shared slots included); every other slot is poisoned, free or not; a poisoned slot some live lane references is
/// saved. `guard = false` is the reversed witness: it drops the protection of the first stepping lane, so that lane
/// reads poison and the gate must fail (specs/PoisonGate.tla, Bug = "unguarded").
pub fn poison_plan(n_slots: usize, tables: &[Vec<usize>], stepping: &[usize], guard: bool) -> Plan {
    let mut protect = vec![false; n_slots];
    let mut owned = vec![false; n_slots];
    for (i, t) in tables.iter().enumerate() {
        let steps = stepping.contains(&i) && (guard || Some(&i) != stepping.first());
        for &s in t {
            assert!(s < n_slots, "lane {i} references pool index {s} of a pool of {n_slots}");
            owned[s] = true;
            if steps { protect[s] = true; }
        }
    }
    let poison: Vec<usize> = (0..n_slots).filter(|&s| !protect[s]).collect();
    let save = poison.iter().copied().filter(|&s| owned[s]).collect();
    Plan { poison, save }
}

/// The poison: NaN in both the f32 and the fp16 reading, then fp16 65504 (and a huge f32), alternating by word.
pub const POISON: [u32; 2] = [0x7fc0_7fc0, 0x7bff_7bff];

pub fn poison_bytes(b: &mut [u8]) {
    let (chunks, rem) = b.as_chunks_mut::<4>();
    for (i, w) in chunks.iter_mut().enumerate() { *w = POISON[i & 1].to_ne_bytes(); }
    rem.fill(0x7f);
}

/// First byte `(offset, got, expected)` in `b` that differs from `poison_bytes`, or `None` if all bytes match.
pub fn first_non_poison_byte(b: &[u8]) -> Option<(usize, u8, u8)> {
    for (i, w) in b.as_chunks::<4>().0.iter().enumerate() {
        let want = POISON[i & 1].to_ne_bytes();
        if *w != want {
            for j in 0..4 {
                if w[j] != want[j] {
                    return Some((i * 4 + j, w[j], want[j]));
                }
            }
        }
    }
    let tail = b.len() & !3;
    for (j, &x) in b[tail..].iter().enumerate() {
        if x != 0x7f {
            return Some((tail + j, x, 0x7f));
        }
    }
    None
}

fn slot_at(i: usize) -> Slot { Slot::new(u32::try_from(i + 1).expect("pool index fits a slot")).unwrap() }

/// Both kinds' plans for one step. `tables[i]` is live lane i's (kind, slot) list.
pub fn plans(pool: &dyn PoisonPool, tables: &[Vec<(BlockKind, Slot)>], stepping: &[usize], guard: bool) -> Vec<(BlockKind, Plan)> {
    let retained = pool.retained_slots();
    KINDS.iter().map(|&kind| {
        let mut per: Vec<Vec<usize>> = tables.iter().map(|t| t.iter().filter(|(k, _)| *k == kind).map(|(_, s)| s.pool_index().get() as usize).collect()).collect();
        if !retained.is_empty() {
            let r: Vec<usize> = retained.iter().filter(|(k, _)| *k == kind).map(|(_, s)| s.pool_index().get() as usize).collect();
            if !r.is_empty() {
                per.push(r);
            }
        }
        (kind, poison_plan(pool.slot_count(kind), &per, stepping, guard))
    }).collect()
}

/// What `apply` did: slots saved, slots poisoned, and planned slots skipped because they legitimately have no memory.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Applied { pub saved: usize, pub poisoned: usize, pub absent: usize }

/// State captured by `apply` for one step: saved bytes of live lanes' poisoned slots, and every
/// poisoned slot (`(kind, slot, is_saved_non_stepping)`) that no stepping lane owns and therefore
/// MUST still hold exact `POISON` bytes across its entire stride after the step completes.
pub struct StepGuard {
    saved: Vec<(BlockKind, Slot, Vec<u8>)>,
    checked_poison: Vec<(BlockKind, Slot, bool)>,
}

/// Apply the plans: save, then poison. Returns the `StepGuard` for `restore`, and what was actually done.
/// `protected_if_guarded` is `Some(&plans(..., guard = true))` when `plans` was built with `guard = false`
/// (`INK_POOLGATE_SABOTAGE=1`), so the intentionally sabotaged stepping lane's own active slot is excluded
/// from stray-write checks while every other poisoned or saved slot is still verified byte-for-byte.
fn apply_with_guard_ref(
    pool: &dyn PoisonPool,
    plans: &[(BlockKind, Plan)],
    guarded_plans: Option<&[(BlockKind, Plan)]>,
) -> (StepGuard, Applied) {
    let mut saved = Vec::new();
    let mut checked_poison = Vec::new();
    let mut n = Applied::default();
    for (k_idx, (kind, plan)) in plans.iter().enumerate() {
        let strict_poison = guarded_plans.map_or(&plan.poison, |gp| &gp[k_idx].1.poison);
        for &i in &plan.save {
            let (p, len) = pool.slot_bytes(*kind, slot_at(i)).unwrap_or_else(|| panic!("poolgate: {kind:?} slot {} (pool index {i}) of {} is referenced by a live lane and has no memory", i + 1, pool.slot_count(*kind)));
            saved.push((*kind, slot_at(i), unsafe { std::slice::from_raw_parts(p, len) }.to_vec()));
            n.saved += 1;
        }
        for &i in &plan.poison {
            match pool.slot_bytes(*kind, slot_at(i)) {
                Some((p, len)) => {
                    poison_bytes(unsafe { std::slice::from_raw_parts_mut(p, len) });
                    n.poisoned += 1;
                    if strict_poison.binary_search(&i).is_ok() {
                        let is_saved = plan.save.binary_search(&i).is_ok();
                        checked_poison.push((*kind, slot_at(i), is_saved));
                    }
                }
                None if pool.absent_ok(*kind, slot_at(i)) => n.absent += 1,
                None => panic!("poolgate: {kind:?} slot {} (pool index {i}) of {} has no memory and is not marked absent (an ended lane)", i + 1, pool.slot_count(*kind)),
            }
        }
    }
    (StepGuard { saved, checked_poison }, n)
}

#[cfg(test)]
fn apply(pool: &dyn PoisonPool, plans: &[(BlockKind, Plan)]) -> (StepGuard, Applied) {
    apply_with_guard_ref(pool, plans, None)
}

/// Verify before `restore` overwrites saved slots back into the pool that every poisoned slot not
/// owned by a stepping lane (both unreferenced poisoned slots and non-stepping lanes' saved slots)
/// still contains exact `POISON` bytes across its entire stride, then restore saved slots and verify
/// every restored saved slot matches its saved snapshot byte-for-bit.
fn restore_checked(pool: &dyn PoisonPool, guard: StepGuard) -> Result<(), Vec<String>> {
    let mut bad = Vec::new();
    for &(kind, s, is_saved) in &guard.checked_poison {
        let (p, len) = pool.slot_bytes(kind, s).expect("a poisoned slot lost its memory during the step");
        let bytes = unsafe { std::slice::from_raw_parts(p, len) };
        if let Some((off, got, want)) = first_non_poison_byte(bytes) {
            let role = if is_saved {
                "non-stepping live lane's saved slot"
            } else {
                "unreferenced poisoned slot"
            };
            bad.push(format!(
                "stray write into {role} ({kind:?} slot {}, pool index {}) at byte {off}/{len}: got 0x{got:02x}, expected poison 0x{want:02x}",
                s.get(),
                s.pool_index().get()
            ));
        }
    }
    for (kind, s, v) in guard.saved {
        let (p, n) = pool.slot_bytes(kind, s).expect("a saved slot lost its memory during the step");
        assert_eq!(n, v.len(), "{kind:?} slot {} changed size during the step", s.get());
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), p, n); }
        let restored = unsafe { std::slice::from_raw_parts(p, n) };
        if restored != v.as_slice() {
            bad.push(format!(
                "restored {kind:?} slot {} (pool index {}) does not match its saved snapshot byte-for-bit",
                s.get(),
                s.pool_index().get()
            ));
        }
    }
    if bad.is_empty() { Ok(()) } else { Err(bad) }
}

#[cfg(test)]
fn restore(pool: &dyn PoisonPool, guard: StepGuard) {
    if let Err(errs) = restore_checked(pool, guard) {
        panic!("poolgate: {}", errs.join("; "));
    }
}


/// The lanes to run: each a prompt, prefilled in `chunks`, then `decode` greedy steps.
pub struct Workload { pub prompts: Vec<Vec<u32>>, pub chunks: Vec<usize>, pub decode: usize }

impl Workload {
    /// `n` prompts from `base`: even lanes are `base` rotated by 17 i; odd lanes share their even neighbour's first 64
    /// tokens (one whole block) and then go their own way. Each prompt is `chunks`' total long.
    pub fn from_tokens(base: &[u32], n: usize, chunks: Vec<usize>, decode: usize) -> Workload {
        let plen: usize = chunks.iter().sum();
        assert!(base.len() >= plen + 64, "need at least {} reference tokens", plen + 64);
        let rot = |k: usize| { let mut v = base.to_vec(); v.rotate_left(k % base.len()); v };
        let prompts = (0..n).map(|i| {
            if i % 2 == 0 { let mut v = rot(17 * i); v.truncate(plen); v }
            else { let mut v = rot(17 * (i - 1)); v.truncate(64.min(plen)); v.extend_from_slice(&rot(997 + 31 * i)[..plen - v.len()]); v }
        }).collect();
        Workload { prompts, chunks, decode }
    }
    fn len(&self) -> usize { self.chunks.iter().sum::<usize>() + self.decode }
    /// lane i's work: (p0, t) prefill chunks, then (pos, 1) decode steps
    fn work(&self) -> Vec<(usize, usize, bool)> {
        let mut w = Vec::new(); let mut p = 0;
        for &c in &self.chunks { w.push((p, c, false)); p += c; }
        for q in 0..self.decode { w.push((p + q, 1, true)); }
        w
    }
}

fn fnv(v: &[f32]) -> u64 { v.iter().fold(0xcbf29ce484222325u64, |h, x| (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3)) }
fn argmax(v: &[f32]) -> u32 { v.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32 }

/// One lane's record: the logits hash of each step it took, and the tokens it generated.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Record { pub steps: Vec<u64>, pub tokens: Vec<u32> }

/// Each lane alone through `path`, one span per step, nothing poisoned.
pub fn reference(path: &dyn StepPath, w: &Workload) -> Vec<Record> {
    let work = w.work();
    w.prompts.iter().map(|toks| {
        let l = path.new_lane(toks, w.len() + 1);
        let mut r = Record::default();
        for &(p0, t, dec) in &work {
            if dec { path.decode(&[(l, p0)]); } else { path.prefill(&[(l, p0, t)]); }
            let lg = path.logits(0);
            r.steps.push(fnv(&lg));
            if dec || p0 + t == toks.len() { let tok = argmax(&lg); r.tokens.push(tok); path.set_token(l, p0 + t, tok); }
        }
        path.end_lane(l);
        r
    }).collect()
}

/// What the poisoned run saw. `sabotaged`: the lane whose own slots were poisoned, if any.
pub struct Report {
    pub records: Vec<Record>,
    pub steps: usize,
    pub poisoned: usize,
    pub saved: usize,
    pub absent: usize,
    pub sabotaged: Option<usize>,
    pub stray_writes: Vec<String>,
}

/// The lanes together through `cand`, with the pool poisoned before every step. Lane i starts i % 3 rounds late; a
/// round is a prefill step of every prefilling lane, then a decode step of every decoding lane. `sabotage` drops the
/// guard (`poison_plan`) at the first step where a lane reads cache it wrote earlier, so the gate must fail.
pub fn poisoned(cand: &dyn Candidate, w: &Workload, sabotage: bool) -> Report {
    let work = w.work();
    let n = w.prompts.len();
    let mut lane: Vec<Option<LaneId>> = vec![None; n];
    let mut next = vec![0usize; n];
    let mut rec = vec![Record::default(); n];
    let (mut steps, mut poisoned, mut saved_n, mut absent, mut sabotaged) = (0, 0, 0, 0, None);
    let mut stray_writes = Vec::new();
    let mut round = 0;
    while next.iter().any(|&k| k < work.len()) {
        for i in 0..n { if lane[i].is_none() && next[i] == 0 && round >= i % 3 { lane[i] = Some(cand.new_lane(&w.prompts[i], w.len() + 1)); } }
        for dec in [false, true] {
            let who: Vec<usize> = (0..n).filter(|&i| lane[i].is_some() && next[i] < work.len() && work[next[i]].2 == dec).collect();
            if who.is_empty() { continue; }
            let dec_spans: Vec<(LaneId, usize)> = if dec {
                who.iter().map(|&i| (lane[i].unwrap(), work[next[i]].0)).collect()
            } else {
                Vec::new()
            };
            let pf_spans: Vec<(LaneId, usize, usize)> = if !dec {
                who.iter().map(|&i| { let (p0, t, _) = work[next[i]]; (lane[i].unwrap(), p0, t) }).collect()
            } else {
                Vec::new()
            };
            if dec { cand.prepare_decode(&dec_spans); } else { cand.prepare_prefill(&pf_spans); }
            let live: Vec<usize> = (0..n).filter(|&i| lane[i].is_some()).collect();
            let tables: Vec<Vec<(BlockKind, Slot)>> = live.iter().map(|&i| cand.lane_slots(lane[i].unwrap())).collect();
            let stepping: Vec<usize> = who.iter().map(|i| live.iter().position(|j| j == i).unwrap()).collect();
            let unguard = sabotage && sabotaged.is_none() && work[next[who[0]]].0 > 0;
            if unguard {
                sabotaged = Some(who[0]);
                cand.note_sabotaged(lane[who[0]].unwrap());
                println!("INK_POOLGATE_SABOTAGE: lane {}'s slots poisoned in its own step at position {}", who[0], work[next[who[0]]].0);
            }
            let ps = plans(cand, &tables, &stepping, !unguard);
            let strict_ps = unguard.then(|| plans(cand, &tables, &stepping, true));
            let (StepGuard { saved, checked_poison }, done) = apply_with_guard_ref(cand, &ps, strict_ps.as_deref());
            let stepping_ids: Vec<LaneId> = who.iter().map(|&i| lane[i].unwrap()).collect();
            cand.note_applied(&stepping_ids, &saved);
            poisoned += done.poisoned; saved_n += done.saved; absent += done.absent;
            if dec { cand.decode(&dec_spans); }
            else { cand.prefill(&pf_spans); }
            if let Err(errs) = restore_checked(cand, StepGuard { saved, checked_poison }) {
                for err in errs {
                    stray_writes.push(format!("step {steps}: {err}"));
                }
            }
            cand.note_restored();
            steps += 1;
            for (k, &i) in who.iter().enumerate() {
                let (p0, t, _) = work[next[i]];
                let lg = cand.logits(k);
                rec[i].steps.push(fnv(&lg));
                if dec || p0 + t == w.prompts[i].len() { let tok = argmax(&lg); rec[i].tokens.push(tok); cand.set_token(lane[i].unwrap(), p0 + t, tok); }
                next[i] += 1;
            }
        }
        for i in 0..n { if next[i] == work.len() { if let Some(l) = lane[i].take() { cand.end_lane(l); } } }
        round += 1;
    }
    Report { records: rec, steps, poisoned, saved: saved_n, absent, sabotaged, stray_writes }
}

/// The lanes other than lane poisoned whose records differ from the reference. A sabotaged run poisons one lane's own
/// slots in one step; lanes share no unpoisoned cache with a sabotaged lane, so every other lane must still match, or
/// something the engine shares across lanes carried the poison over (the sliding-window scratch tail, src/engine.rs,
/// specs/ScratchTail.tla).
pub fn leaked(want: &[Record], got: &Report, poisoned: usize) -> Vec<usize> {
    want.iter().zip(&got.records).enumerate().filter(|&(i, (a, b))| i != poisoned && a != b).map(|(i, _)| i).collect()
}

/// The reference against the poisoned candidate: Ok(lines) when every lane matches at every step, Err(lines) naming
/// the first difference of each lane that differs.
pub fn compare(want: &[Record], got: &Report) -> Result<Vec<String>, Vec<String>> {
    if got.records.len() != want.len() {
        return Err(vec![format!("lane count mismatch: candidate returned {} lanes, reference has {}", got.records.len(), want.len())]);
    }
    let mut bad = got.stray_writes.clone();
    for (i, (a, b)) in want.iter().zip(&got.records).enumerate() {
        if let Some(s) = a.steps.iter().zip(&b.steps).position(|(x, y)| x != y) { bad.push(format!("lane {i}: logits differ from step {s}")); }
        else if a.steps.len() != b.steps.len() { bad.push(format!("lane {i}: {} steps, reference {}", b.steps.len(), a.steps.len())); }
        else if a.tokens != b.tokens { bad.push(format!("lane {i}: tokens differ")); }
    }
    if bad.is_empty() { Ok(vec![format!("{} lanes, {} steps, {} slot poisonings ({} saved and restored, {} planned slots of ended lanes absent): every step's logits and every token identical, zero stray writes", want.len(), got.steps, got.poisoned, got.saved, got.absent)]) } else { Err(bad) }
}

#[cfg(not(standalone_blocks))]
mod gpu_paths {
use super::*;

/// Layer 2's path: each lane owns its own cache buffers. As a pool, lane k's window cache is window slot k + 1 and its
/// full cache is full slot k + 1; an ended lane's slots have no memory.
pub struct Layer2<'a> { eng: &'a Engine, lanes: RefCell<Vec<Option<Lane>>>, rows: std::cell::Cell<(usize, bool)> }

impl<'a> Layer2<'a> {
    pub fn new(eng: &'a Engine, max_lanes: usize) -> Layer2<'a> {
        eng.ensure_batch(eng.chunk);
        eng.ensure_dbatch(max_lanes);
        Layer2 { eng, lanes: RefCell::new(Vec::new()), rows: std::cell::Cell::new((0, false)) }
    }
    fn run<R>(&self, f: impl FnOnce(&[Option<Lane>]) -> R) -> R { f(&self.lanes.borrow()) }
}

impl StepPath for Layer2<'_> {
    fn new_lane(&self, toks: &[u32], cap: usize) -> LaneId {
        let l = Lane::new(self.eng);
        { let r = self.eng.on(&l); r.reserve(cap, false); r.set_tokens(0, toks); }
        let mut v = self.lanes.borrow_mut(); v.push(Some(l)); v.len() - 1
    }
    fn set_token(&self, lane: LaneId, pos: usize, tok: u32) { self.run(|v| self.eng.on(v[lane].as_ref().unwrap()).set_tokens(pos, &[tok])) }
    fn prefill(&self, spans: &[(LaneId, usize, usize)]) {
        self.run(|v| {
            let sp: Vec<Span> = spans.iter().map(|&(l, p0, t)| Span { lane: v[l].as_ref().unwrap(), p0, t }).collect();
            let cb = self.eng.gpu.cmd(); self.eng.encode_spans(&cb, &sp, true, false); cb.commit(); cb.waitUntilCompleted();
        });
        self.rows.set((spans.len(), false));
    }
    fn decode(&self, spans: &[(LaneId, usize)]) {
        self.run(|v| {
            let sp: Vec<Span> = spans.iter().map(|&(l, p0)| Span { lane: v[l].as_ref().unwrap(), p0, t: 1 }).collect();
            let cb = self.eng.gpu.cmd(); self.eng.encode_decode(&cb, &sp, true, false); cb.commit(); cb.waitUntilCompleted();
        });
        self.rows.set((spans.len(), true));
    }
    fn logits(&self, row: usize) -> Vec<f32> {
        // a prefill of one span writes the engine's logits; every other step writes one row per span
        let (n, dec) = self.rows.get();
        assert!(row < n, "logits row {row} of a step of {n} spans");
        (if n == 1 && !dec { self.eng.logits() } else { self.eng.span_logits(row) }).to_vec()
    }
    fn end_lane(&self, lane: LaneId) { self.lanes.borrow_mut()[lane] = None; }
}

impl PoisonPool for Layer2<'_> {
    fn slot_count(&self, _: BlockKind) -> usize { self.lanes.borrow().len() }
    fn lane_slots(&self, lane: LaneId) -> Vec<(BlockKind, Slot)> { KINDS.iter().map(|&k| (k, slot_at(lane))).collect() }
    fn slot_bytes(&self, kind: BlockKind, slot: Slot) -> Option<(*mut u8, usize)> {
        let i = slot.pool_index().get() as usize;
        // SAFETY: `PoisonPool` runs between GPU steps when nothing is in flight and before any reallocation.
        self.run(|v| v[i].as_ref().map(|l| unsafe { self.eng.lane_regions(l) }[if kind == BlockKind::Window { 0 } else { 1 }]))
    }
    fn absent_ok(&self, _: BlockKind, slot: Slot) -> bool { self.run(|v| v[slot.pool_index().get() as usize].is_none()) }
}

/// Statistics collected during a `Layer3Pool` run (for gate receipts and diagnostics).
#[derive(Clone, Copy, Debug, Default)]
pub struct L3Stats {
    pub slots_taken: usize,
    pub slots_reused: usize,
    pub blocks_sealed: usize,
    pub shared_dedups: usize,
    pub start_points_saved: usize,
    pub start_matches: usize,
    pub window_pruned: usize,
    pub tails_dropped: usize,
}

struct RetainedStart {
    pos: usize,
    cs: Vec<u8>,
    reasons: u8,
    entries: Vec<(BlockKind, BlockIndex, BlockEntry)>,
}

struct L3Lane {
    lane: Lane,
    tokens: Vec<u32>,
    prompt_len: usize,
    written: usize,
    sealed: Vec<SealedId>,
    head_start: Option<SealedId>,
    sabotaged: bool,
}

/// Layer 3 block-cache candidate: contiguous GPU `BlockPools` per kind (`Window`, `Full`),
/// per-lane 64-bit `BlockTable`, sealed 64-token `PrefixTree` (`EntryTree`), and start points
/// (recurrent state + full blocks + preceding 1024-token window blocks).
pub struct Layer3Pool<'a> {
    eng: &'a Engine,
    lanes: RefCell<Vec<Option<L3Lane>>>,
    tree: RefCell<EntryTree>,
    starts: RefCell<HashMap<SealedId, RetainedStart>>,
    keep_persistent_starts: bool,
    rows: Cell<(usize, bool)>,
    stats: RefCell<L3Stats>,
}

impl<'a> Layer3Pool<'a> {
    /// Candidate for `poolgate::poisoned`: every slot not referenced by a live lane is poisoned
    /// before each step, so finished lanes release all their refs back to the pool for reuse + zeroing.
    pub fn new(eng: &'a Engine, max_lanes: usize) -> Layer3Pool<'a> {
        Self::create(eng, max_lanes, false)
    }

    /// Persistent start-point mode: `PROMPT` and `EVERY` (every 16 blocks) start points outlive
    /// finished lanes so subsequent requests match the deepest start point in `PrefixTree`.
    pub fn with_persistent_starts(eng: &'a Engine, max_lanes: usize) -> Layer3Pool<'a> {
        Self::create(eng, max_lanes, true)
    }

    fn create(eng: &'a Engine, max_lanes: usize, keep_persistent_starts: bool) -> Layer3Pool<'a> {
        eng.ensure_batch(eng.chunk);
        eng.ensure_dbatch(max_lanes);
        {
            let mut pools = eng.block_pool().borrow_mut();
            for kind in KINDS {
                let pool = pools.of_mut(kind);
                if pool.len() < 4 {
                    pool.grow_by(4 - pool.len()).expect("initial Layer 3 pool growth");
                }
            }
        }
        let p = Layer3Pool {
            eng,
            lanes: RefCell::new(Vec::new()),
            tree: RefCell::new(EntryTree::new()),
            starts: RefCell::new(HashMap::new()),
            keep_persistent_starts,
            rows: Cell::new((0, false)),
            stats: RefCell::new(L3Stats::default()),
        };
        p.poison_free_slots();
        p
    }

    pub fn stats(&self) -> L3Stats { *self.stats.borrow() }

    /// Overwrite every unreferenced (`refs == 0`) slot in both GPU pools with `POISON`
    /// so any unzeroed reuse or out-of-bounds read immediately produces NaN logits.
    pub fn poison_free_slots(&self) -> usize {
        let mut pools = self.eng.block_pool().borrow_mut();
        let mut count = 0usize;
        for kind in KINDS {
            let pool = pools.of_mut(kind);
            for i in 0..pool.len() {
                let slot = slot_at(i);
                if pool.record(slot).unwrap().refs() == 0 {
                    poison_bytes(pool.slot_bytes_mut(slot).unwrap());
                    count += 1;
                }
            }
        }
        count
    }

    /// Create a lane matched against the deepest live start point in `PrefixTree`.
    /// Returns `(lane_id, matched_pos)` where `matched_pos < toks.len()` is a multiple of 64 (or 0).
    pub fn new_lane_matched(&self, toks: &[u32], cap: usize) -> (LaneId, usize) {
        assert!(!toks.is_empty(), "prompt cannot be empty");
        let lane = Lane::new_blocked(self.eng, cap.max(toks.len() + 1));
        self.eng.on(&lane).set_tokens(0, toks);
        // Clamp to toks.len() - 1 so at least one token remains to prefill and produce prompt-end logits.
        let search = &toks[..toks.len() - 1];
        let matched = self.tree.borrow().deepest_start(search);
        let mut written = 0usize;
        let mut sealed = Vec::new();
        if matched.at > 0 {
            if let Some(sid) = matched.start {
                let start_opt = self.starts.borrow().get(&sid).map(|s| (s.pos, s.cs.clone(), s.entries.clone()));
                if let Some((pos, cs, entries)) = start_opt {
                    assert_eq!(pos, matched.at);
                    let mut pools = self.eng.block_pool().borrow_mut();
                    let all_live = entries.iter().all(|&(kind, _, e)| {
                        pools.of(kind).record(e.slot()).ok().is_some_and(|r| {
                            r.generation() == e.generation() && r.sealed() && r.refs() > 0
                        })
                    });
                    if all_live {
                        {
                            let mut tbl = lane.block_table().borrow_mut();
                            for &(kind, bi, e) in &entries {
                                tbl.fill(pools.of_mut(kind), bi, e.slot(), e.generation())
                                    .expect("install start-point block");
                            }
                        }
                        self.eng.on(&lane).set_cs(&cs);
                        written = matched.at;
                        sealed = matched.ids;
                        self.stats.borrow_mut().start_matches += 1;
                    }
                }
            }
        }
        let mut v = self.lanes.borrow_mut();
        v.push(Some(L3Lane {
            lane,
            tokens: toks.to_vec(),
            prompt_len: toks.len(),
            written,
            sealed,
            head_start: None,
            sabotaged: false,
        }));
        (v.len() - 1, written)
    }

    fn take_zeroed_slot(&self, kind: BlockKind) -> (Slot, u32) {
        let mut pools = self.eng.block_pool().borrow_mut();
        let pool = pools.of_mut(kind);
        let slot = match pool.lowest_free() {
            Some(s) => s,
            None => {
                let s = pool.grow().expect("grow Layer 3 block pool");
                // Pre-poison freshly grown slot before `take` zeroes it so `take` always wipes poison.
                poison_bytes(pool.slot_bytes_mut(s).expect("fresh slot bytes"));
                s
            }
        };
        let _ = pool.take(slot).expect("take zeroed slot");
        let gen = pool.record(slot).unwrap().generation();
        let mut st = self.stats.borrow_mut();
        st.slots_taken += 1;
        if gen > 1 { st.slots_reused += 1; }
        (slot, gen)
    }

    fn ensure_span_blocks(&self, l3: &mut L3Lane, p0: usize, t: usize) {
        let lo_blk = p0 / BLOCK;
        let hi_blk = (p0 + t - 1) / BLOCK;
        for b in lo_blk..=hi_blk {
            let bi = BlockIndex::new(b as u32);
            for kind in KINDS {
                let existing = l3.lane.block_table().borrow().get(kind, bi).expect("block in table range");
                if let Some(entry) = existing {
                    let pools = self.eng.block_pool().borrow();
                    let rec = pools.of(kind).record(entry.slot()).expect("existing slot record");
                    assert_eq!(rec.generation(), entry.generation(), "stale entry in active lane table");
                    assert!(!rec.sealed(), "attempt to write into sealed {kind:?} block {b}");
                } else {
                    let (slot, gen) = self.take_zeroed_slot(kind);
                    let mut pools = self.eng.block_pool().borrow_mut();
                    l3.lane.block_table().borrow_mut()
                        .fill(pools.of_mut(kind), bi, slot, gen)
                        .expect("fill newly taken slot");
                }
            }
        }
    }

    fn drop_start_reason(&self, sid: SealedId, reason: u8) {
        let mut remove_entries = None;
        {
            let mut starts = self.starts.borrow_mut();
            if let Some(st) = starts.get_mut(&sid) {
                st.reasons &= !reason;
                if st.reasons == 0 {
                    remove_entries = starts.remove(&sid).map(|s| s.entries);
                }
            }
        }
        if let Some(entries) = remove_entries {
            let _ = self.tree.borrow_mut().release_start(sid);
            let mut pools = self.eng.block_pool().borrow_mut();
            for (kind, _, e) in entries {
                let pool = pools.of_mut(kind);
                if let Ok(rec) = pool.record(e.slot()) {
                    if rec.generation() == e.generation() && rec.refs() > 0 {
                        pool.record_mut(e.slot()).unwrap().drop_ref().unwrap();
                    }
                }
            }
        }
    }

    fn prune_expired_window(&self, l3: &mut L3Lane, next_pos: usize) {
        let min_win_blk = next_pos.saturating_sub(EXT - 1) / BLOCK;
        if min_win_blk == 0 { return; }
        let mut pools = self.eng.block_pool().borrow_mut();
        let mut tbl = l3.lane.block_table().borrow_mut();
        for b in 0..min_win_blk {
            let bi = BlockIndex::new(b as u32);
            if tbl.get(BlockKind::Window, bi).expect("window block in range").is_some() {
                tbl.clear_kind_entry(&mut pools.window, bi).expect("prune expired window block");
                self.stats.borrow_mut().window_pruned += 1;
            }
        }
    }

    fn finish_prefill_span(&self, l3: &mut L3Lane, p0: usize, t: usize) {
        let end = p0 + t;
        l3.written = end;
        let lo_blk = p0 / BLOCK;
        let hi_blk = (end - 1) / BLOCK;
        for b in lo_blk..=hi_blk {
            let bi = BlockIndex::new(b as u32);
            let block_len = (end - b * BLOCK).min(BLOCK) as u32;
            let (full_e, win_e) = {
                let tbl = l3.lane.block_table().borrow();
                (
                    tbl.get(BlockKind::Full, bi).unwrap().expect("full entry"),
                    tbl.get(BlockKind::Window, bi).unwrap().expect("window entry"),
                )
            };
            {
                let mut pools = self.eng.block_pool().borrow_mut();
                for (kind, e) in [(BlockKind::Full, full_e), (BlockKind::Window, win_e)] {
                    let rec = pools.of_mut(kind).record_mut(e.slot()).unwrap();
                    rec.set_length(block_len).expect("set block length");
                    if block_len == BLOCK as u32 { rec.seal(); }
                }
            }
            if block_len == BLOCK as u32 {
                self.stats.borrow_mut().blocks_sealed += 1;
                if !l3.sabotaged && (b == 0 || l3.sealed.len() == b) {
                    let parent = if b == 0 { None } else { l3.sealed.get(b - 1).copied() };
                    let blk_toks = &l3.tokens[b * BLOCK..(b + 1) * BLOCK];
                    let handles = BlockHandles::new(full_e, Some(win_e));
                    let mut id = self.tree.borrow_mut()
                        .insert(parent, b * BLOCK, blk_toks, handles)
                        .expect("insert sealed block into PrefixTree");
                    let canon = self.tree.borrow().handle(id).expect("inserted handle");
                    let canon_f = canon.full();
                    let canon_w = canon.window().expect("window handle on sealed block");
                    if canon_f != full_e || canon_w != win_e {
                        // Another lane already sealed these exact 64 tokens. Reuse its slots if both
                        // are still live and referenced (so poolgate either protected or saved them).
                        let can_share = {
                            let pools = self.eng.block_pool().borrow();
                            let ok = |kind: BlockKind, e: BlockEntry| {
                                pools.of(kind).record(e.slot()).ok().is_some_and(|r| {
                                    r.generation() == e.generation() && r.sealed() && r.refs() > 0
                                })
                            };
                            ok(BlockKind::Full, canon_f) && ok(BlockKind::Window, canon_w)
                        };
                        if can_share {
                            let mut pools = self.eng.block_pool().borrow_mut();
                            let mut tbl = l3.lane.block_table().borrow_mut();
                            tbl.fill(&mut pools.full, bi, canon_f.slot(), canon_f.generation())
                                .expect("share canonical full block");
                            tbl.fill(&mut pools.window, bi, canon_w.slot(), canon_w.generation())
                                .expect("share canonical window block");
                            self.stats.borrow_mut().shared_dedups += 1;
                        } else {
                            // Stale or unreferenced cached node: evict it and publish our live sealed slots.
                            let _ = self.tree.borrow_mut().remove(id);
                            id = self.tree.borrow_mut()
                                .insert(parent, b * BLOCK, blk_toks, handles)
                                .expect("re-insert fresh sealed block");
                        }
                    }
                    if l3.sealed.len() == b { l3.sealed.push(id); } else { l3.sealed[b] = id; }
                }
            }
        }
        // Keep start points at prompt end, every 16th block, and lane head (when persistent starts enabled).
        if self.keep_persistent_starts && end.is_multiple_of(BLOCK) && l3.sealed.len() == end / BLOCK {
            let tip = *l3.sealed.last().unwrap();
            let mut reasons = HEAD;
            if end == l3.prompt_len { reasons |= PROMPT; }
            if (end / BLOCK).is_multiple_of(16) { reasons |= EVERY; }
            if let Some(old_head) = l3.head_start.take() {
                if old_head != tip { self.drop_start_reason(old_head, HEAD); }
            }
            let exists = self.starts.borrow().contains_key(&tip);
            if exists {
                self.starts.borrow_mut().get_mut(&tip).unwrap().reasons |= reasons;
            } else {
                let cs = self.eng.on(&l3.lane).get_cs();
                let tbl = l3.lane.block_table().borrow();
                let mut entries = Vec::new();
                for b in 0..(end / BLOCK) {
                    let bi = BlockIndex::new(b as u32);
                    let fe = tbl.get(BlockKind::Full, bi).unwrap().expect("start point full block");
                    entries.push((BlockKind::Full, bi, fe));
                }
                let win_lo = end.saturating_sub(WINDOW) / BLOCK;
                for b in win_lo..(end / BLOCK) {
                    let bi = BlockIndex::new(b as u32);
                    let we = tbl.get(BlockKind::Window, bi).unwrap().expect("start point window block");
                    entries.push((BlockKind::Window, bi, we));
                }
                {
                    let mut pools = self.eng.block_pool().borrow_mut();
                    for &(kind, _, e) in &entries {
                        pools.of_mut(kind).record_mut(e.slot()).unwrap().add_ref().expect("retain start point ref");
                    }
                }
                self.tree.borrow_mut().retain_start(tip).expect("retain start in PrefixTree");
                self.starts.borrow_mut().insert(tip, RetainedStart { pos: end, cs, reasons, entries });
                self.stats.borrow_mut().start_points_saved += 1;
            }
            l3.head_start = Some(tip);
        }
        self.prune_expired_window(l3, end);
    }

    fn finish_decode_span(&self, l3: &mut L3Lane, pos: usize) {
        let end = pos + 1;
        l3.written = end;
        let b = pos / BLOCK;
        let bi = BlockIndex::new(b as u32);
        let block_len = (end - b * BLOCK).min(BLOCK) as u32;
        let tbl = l3.lane.block_table().borrow();
        let mut pools = self.eng.block_pool().borrow_mut();
        for kind in KINDS {
            let e = tbl.get(kind, bi).unwrap().expect("decode block entry");
            // Decode-written blocks stay private to the lane and are never sealed into PrefixTree.
            pools.of_mut(kind).record_mut(e.slot()).unwrap().set_length(block_len).expect("set decode length");
        }
        drop(tbl);
        drop(pools);
        self.prune_expired_window(l3, end);
    }

    /// Release every remaining lane and retained start point so all pool slots return to `refs == 0`.
    /// Execute one Layer 3 batched decode step and return GPU execution time in milliseconds.
    pub fn decode_timed(&self, spans: &[(LaneId, usize)]) -> f64 {
        {
            let mut v = self.lanes.borrow_mut();
            for &(l, pos) in spans {
                self.ensure_span_blocks(v[l].as_mut().expect("live lane"), pos, 1);
            }
        }
        let gpu_ms = {
            let v = self.lanes.borrow();
            let sp: Vec<Span> = spans.iter().map(|&(l, p0)| Span { lane: &v[l].as_ref().unwrap().lane, p0, t: 1 }).collect();
            let cb = self.eng.gpu.cmd();
            self.eng.encode_decode(&cb, &sp, true, false);
            cb.commit();
            cb.waitUntilCompleted();
            (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3
        };
        {
            let mut v = self.lanes.borrow_mut();
            for &(l, pos) in spans {
                self.finish_decode_span(v[l].as_mut().expect("live lane"), pos);
            }
        }
        self.rows.set((spans.len(), true));
        gpu_ms
    }

    /// Ensure blocks for `spans` exist and run one timed `encode_decode` without advancing/pruning window state (benchmark only).
    pub fn bench_decode_ms(&self, spans: &[(LaneId, usize)]) -> f64 {
        {
            let mut v = self.lanes.borrow_mut();
            for &(l, pos) in spans {
                self.ensure_span_blocks(v[l].as_mut().expect("live lane"), pos, 1);
            }
        }
        let v = self.lanes.borrow();
        let sp: Vec<Span> = spans.iter().map(|&(l, p0)| Span { lane: &v[l].as_ref().unwrap().lane, p0, t: 1 }).collect();
        let cb = self.eng.gpu.cmd();
        self.eng.encode_decode(&cb, &sp, true, false);
        cb.commit();
        crate::metal::wait_completed(&cb, "poolgate bench_decode", spans[0].1).unwrap();
        (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3
    }

    /// Borrow raw `*const Lane` pointers for live `ids` (caller must not re-allocate `self.lanes` while live).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn borrow_lane_ptrs(&self, ids: &[LaneId]) -> Vec<*const Lane> {
        let v = self.lanes.borrow();
        ids.iter()
            .map(|&id| &v[id].as_ref().expect("live lane").lane as *const Lane)
            .collect()
    }

    /// Return `(live_full_bytes, live_window_bytes, total_pool_slab_bytes)` currently held in `BlockPools`.
    pub fn kv_memory_bytes(&self) -> (usize, usize, usize) {
        use objc2_metal::MTLBuffer;
        let pools = self.eng.block_pool().borrow();
        let live_of = |pool: &crate::blocks::BlockPool| -> usize {
            let stride = pool.stride_bytes() as usize;
            let mut count = 0usize;
            for i in 1..=(pool.len() as u32) {
                let s = Slot::new(i).unwrap();
                if pool.record(s).unwrap().refs() > 0 {
                    count += 1;
                }
            }
            count * stride
        };
        let slab_of = |pool: &crate::blocks::BlockPool| -> usize {
            pool.gpu_slabs().into_iter().map(|b| b.length()).sum()
        };
        (
            live_of(&pools.full),
            live_of(&pools.window),
            slab_of(&pools.full) + slab_of(&pools.window),
        )
    }

    pub fn shutdown(&self) {
        let n = self.lanes.borrow().len();
        for i in 0..n {
            if self.lanes.borrow()[i].is_some() { self.end_lane(i); }
        }
        let sids: Vec<SealedId> = self.starts.borrow().keys().copied().collect();
        for sid in sids { self.drop_start_reason(sid, PROMPT | EVERY | HEAD); }
    }
}

impl Drop for Layer3Pool<'_> {
    fn drop(&mut self) { self.shutdown(); }
}

impl StepPath for Layer3Pool<'_> {
    fn new_lane(&self, toks: &[u32], cap: usize) -> LaneId {
        let lane = Lane::new_blocked(self.eng, cap.max(toks.len() + 1));
        self.eng.on(&lane).set_tokens(0, toks);
        let mut v = self.lanes.borrow_mut();
        v.push(Some(L3Lane {
            lane,
            tokens: toks.to_vec(),
            prompt_len: toks.len(),
            written: 0,
            sealed: Vec::new(),
            head_start: None,
            sabotaged: false,
        }));
        v.len() - 1
    }

    fn set_token(&self, lane: LaneId, pos: usize, tok: u32) {
        let mut v = self.lanes.borrow_mut();
        let l3 = v[lane].as_mut().expect("live lane");
        self.eng.on(&l3.lane).set_tokens(pos, &[tok]);
        if pos >= l3.tokens.len() { l3.tokens.resize(pos + 1, 0); }
        l3.tokens[pos] = tok;
    }

    fn prepare_prefill(&self, spans: &[(LaneId, usize, usize)]) {
        self.poison_free_slots();
        let mut v = self.lanes.borrow_mut();
        for &(l, p0, t) in spans {
            self.ensure_span_blocks(v[l].as_mut().expect("live lane"), p0, t);
        }
    }

    fn prepare_decode(&self, spans: &[(LaneId, usize)]) {
        self.poison_free_slots();
        let mut v = self.lanes.borrow_mut();
        for &(l, pos) in spans {
            self.ensure_span_blocks(v[l].as_mut().expect("live lane"), pos, 1);
        }
    }

    fn prefill(&self, spans: &[(LaneId, usize, usize)]) {
        {
            let mut v = self.lanes.borrow_mut();
            for &(l, p0, t) in spans {
                self.ensure_span_blocks(v[l].as_mut().expect("live lane"), p0, t);
            }
        }
        {
            let v = self.lanes.borrow();
            let sp: Vec<Span> = spans.iter().map(|&(l, p0, t)| Span { lane: &v[l].as_ref().unwrap().lane, p0, t }).collect();
            let cb = self.eng.gpu.cmd();
            self.eng.encode_spans(&cb, &sp, true, false);
            cb.commit();
            cb.waitUntilCompleted();
        }
        {
            let mut v = self.lanes.borrow_mut();
            for &(l, p0, t) in spans {
                self.finish_prefill_span(v[l].as_mut().expect("live lane"), p0, t);
            }
        }
        self.rows.set((spans.len(), false));
    }

    fn decode(&self, spans: &[(LaneId, usize)]) {
        {
            let mut v = self.lanes.borrow_mut();
            for &(l, pos) in spans {
                self.ensure_span_blocks(v[l].as_mut().expect("live lane"), pos, 1);
            }
        }
        {
            let v = self.lanes.borrow();
            let sp: Vec<Span> = spans.iter().map(|&(l, p0)| Span { lane: &v[l].as_ref().unwrap().lane, p0, t: 1 }).collect();
            let cb = self.eng.gpu.cmd();
            self.eng.encode_decode(&cb, &sp, true, false);
            cb.commit();
            cb.waitUntilCompleted();
        }
        {
            let mut v = self.lanes.borrow_mut();
            for &(l, pos) in spans {
                self.finish_decode_span(v[l].as_mut().expect("live lane"), pos);
            }
        }
        self.rows.set((spans.len(), true));
    }

    fn logits(&self, row: usize) -> Vec<f32> {
        let (n, dec) = self.rows.get();
        assert!(row < n, "logits row {row} of a step of {n} spans");
        (if n == 1 && !dec { self.eng.logits() } else { self.eng.span_logits(row) }).to_vec()
    }

    fn end_lane(&self, lane: LaneId) {
        let l3 = match self.lanes.borrow_mut()[lane].take() {
            Some(l) => l,
            None => return,
        };
        if let Some(head) = l3.head_start {
            self.drop_start_reason(head, HEAD);
        }
        let mut pools = self.eng.block_pool().borrow_mut();
        let unsealed_tails = {
            let tbl = l3.lane.block_table().borrow();
            tbl.entries().filter(|(_, e)| {
                pools.of(e.kind()).record(e.slot()).ok().is_some_and(|r| !r.sealed())
            }).count()
        };
        self.stats.borrow_mut().tails_dropped += unsealed_tails;
        l3.lane.block_table().borrow_mut().clear_entries(&mut pools).expect("clear lane block table");
    }
}

impl PoisonPool for Layer3Pool<'_> {
    fn slot_count(&self, kind: BlockKind) -> usize {
        self.eng.block_pool().borrow().of(kind).len()
    }
    fn lane_slots(&self, lane: LaneId) -> Vec<(BlockKind, Slot)> {
        let v = self.lanes.borrow();
        let l3 = v[lane].as_ref().expect("live lane");
        let slots = table_slots(&l3.lane.block_table().borrow());
        slots
    }
    fn slot_bytes(&self, kind: BlockKind, slot: Slot) -> Option<(*mut u8, usize)> {
        self.eng.block_pool().borrow().of(kind).slot_raw_region(slot)
    }
    fn note_sabotaged(&self, lane: LaneId) {
        if let Some(l3) = self.lanes.borrow_mut()[lane].as_mut() {
            l3.sabotaged = true;
        }
    }
}

/// Require positive sharing, slot reuse, unsealed-tail drop, and (once crossing `EXT`) live window pruning.
#[cfg(test)]
pub fn verify_l3_workload_stats(st: &L3Stats, lanes: usize, prompt_positions: usize, total_positions: usize) -> Result<(), String> {
    if prompt_positions >= BLOCK && st.blocks_sealed == 0 {
        return Err(format!("expected blocks_sealed > 0 for {prompt_positions}-token prompt, got {st:?}"));
    }
    if lanes >= 2 && prompt_positions >= BLOCK && st.shared_dedups == 0 {
        return Err(format!("expected shared_dedups > 0 across {lanes} lanes, got {st:?}"));
    }
    if lanes >= 3 && prompt_positions > BLOCK && st.slots_reused == 0 {
        return Err(format!("expected slots_reused > 0 across {lanes} staggered lanes, got {st:?}"));
    }
    if !total_positions.is_multiple_of(BLOCK) && st.tails_dropped == 0 {
        return Err(format!("expected tails_dropped > 0 for non-block-multiple length {total_positions}, got {st:?}"));
    }
    if total_positions >= EXT + BLOCK - 1 && st.window_pruned == 0 {
        return Err(format!("expected window_pruned > 0 for {total_positions} positions (>= {}), got {st:?}", EXT + BLOCK - 1));
    }
    Ok(())
}

pub(super) fn check_prefix_start_point_inputs(base_len: usize, ctx: usize) -> Result<(), String> {
    if ctx < 1408 || base_len < 1408 {
        return Err(format!(
            "prefix start-point gate requires ctx >= 1408 (got {ctx}) and at least 1408 reference tokens (got {base_len})"
        ));
    }
    Ok(())
}

pub(super) fn corrupt_start_cs_for_sabotage(cs: &mut [u8]) {
    for chunk in cs.as_chunks_mut::<4>().0 {
        chunk.copy_from_slice(&42.0f32.to_ne_bytes());
    }
}

struct ServeCandidateLane {
    lane: Lane,
    tokens: Vec<u32>,
    blocks: crate::serve::block_cache::LaneBlockState,
    sabotaged: bool,
}

/// Production `ServeBlockCache` candidate for `inky poolgate`: exercises the exact `src/serve/block_cache.rs`
/// `EntryTree` deduplication, `RetainedStart` (`PROMPT` + `EVERY`) matching, `take_zeroed_slot` reuse, and
/// sliding-window pruning under full `PoisonPool` NaN/fp16-max slot poisoning.
pub struct ServePoolCandidate<'a> {
    eng: &'a Engine,
    cache: RefCell<crate::serve::block_cache::ServeBlockCache>,
    lanes: RefCell<Vec<Option<ServeCandidateLane>>>,
    rows: Cell<(usize, bool)>,
    cache_sabotage: bool,
    cache_sabotage_triggered: Cell<bool>,
}

impl<'a> ServePoolCandidate<'a> {
    pub fn new(eng: &'a Engine, max_lanes: usize, max_pos: usize, cache_sabotage: bool) -> ServePoolCandidate<'a> {
        eng.ensure_batch(eng.chunk);
        eng.ensure_dbatch(max_lanes.max(1));
        let blocks_per_lane = max_pos.div_ceil(BLOCK).max(2);
        let win_per_lane = blocks_per_lane.min(WINDOW / BLOCK + 4);
        let n_window = ((max_lanes + 2) * win_per_lane + 4).max(8);
        let n_full = ((max_lanes + 2) * blocks_per_lane + 4).max(8);
        eng.init_block_pools_exact(n_window, n_full);
        let cand = ServePoolCandidate {
            eng,
            cache: RefCell::new(crate::serve::block_cache::ServeBlockCache::new()),
            lanes: RefCell::new(Vec::new()),
            rows: Cell::new((0, false)),
            cache_sabotage,
            cache_sabotage_triggered: Cell::new(false),
        };
        cand.poison_free_slots();
        cand
    }

    pub fn candidate_label(&self) -> &'static str {
        "ServeBlockCache + BlockPools + BlockTable + EntryTree"
    }

    pub fn stats(&self) -> crate::serve::block_cache::CacheStats {
        self.cache.borrow().stats()
    }

    pub fn cache_sabotage_triggered(&self) -> bool {
        self.cache_sabotage_triggered.get()
    }

    pub fn poison_free_slots(&self) -> usize {
        let mut pools = self.eng.block_pool().borrow_mut();
        let mut count = 0usize;
        for kind in KINDS {
            let pool = pools.of_mut(kind);
            for i in 0..pool.len() {
                let slot = slot_at(i);
                if pool.record(slot).unwrap().refs() == 0 {
                    poison_bytes(pool.slot_bytes_mut(slot).unwrap());
                    count += 1;
                }
            }
        }
        count
    }

    pub fn new_lane_matched(&self, toks: &[u32], cap: usize) -> (LaneId, usize) {
        assert!(!toks.is_empty(), "prompt cannot be empty");
        let lane = Lane::new_blocked(self.eng, cap.max(toks.len() + 1));
        self.eng.on(&lane).set_tokens(0, toks);
        let mut blocks = crate::serve::block_cache::LaneBlockState::default();
        let matched_at = {
            let mut pools = self.eng.block_pool().borrow_mut();
            let mut tbl = lane.block_table().borrow_mut();
            let mut cache = self.cache.borrow_mut();
            match cache.match_and_bind(&mut pools, &mut tbl, &mut blocks, toks) {
                Some((at, mut cs)) => {
                    if self.cache_sabotage && !self.cache_sabotage_triggered.get() {
                        // Planted ServeBlockCache bug: corrupt restored recurrent state at start point.
                        for b in &mut cs {
                            *b ^= 0xFF;
                        }
                        self.cache_sabotage_triggered.set(true);
                        println!("INK_POOLGATE_CACHE_SABOTAGE: corrupted ServeBlockCache start-point CS at pos {at}");
                    }
                    self.eng.on(&lane).set_cs(&cs);
                    at
                }
                None => 0,
            }
        };
        let mut v = self.lanes.borrow_mut();
        v.push(Some(ServeCandidateLane {
            lane,
            tokens: toks.to_vec(),
            blocks,
            sabotaged: false,
        }));
        (v.len() - 1, matched_at)
    }

    pub fn shutdown(&self) {
        let mut v = self.lanes.borrow_mut();
        let mut pools = self.eng.block_pool().borrow_mut();
        let mut cache = self.cache.borrow_mut();
        for slot in v.iter_mut() {
            if let Some(mut sl) = slot.take() {
                let mut tbl = sl.lane.block_table().borrow_mut();
                cache.release_lane(&mut pools, &mut tbl, &mut sl.blocks);
            }
        }
        cache.shutdown(&mut pools);
    }
}

impl StepPath for ServePoolCandidate<'_> {
    fn new_lane(&self, toks: &[u32], cap: usize) -> LaneId {
        let lane = Lane::new_blocked(self.eng, cap.max(toks.len() + 1));
        self.eng.on(&lane).set_tokens(0, toks);
        let blocks = crate::serve::block_cache::LaneBlockState {
            prompt_len: toks.len(),
            ..Default::default()
        };
        let mut v = self.lanes.borrow_mut();
        v.push(Some(ServeCandidateLane {
            lane,
            tokens: toks.to_vec(),
            blocks,
            sabotaged: false,
        }));
        v.len() - 1
    }

    fn set_token(&self, lane: LaneId, pos: usize, tok: u32) {
        let mut v = self.lanes.borrow_mut();
        let sl = v[lane].as_mut().expect("live ServePoolCandidate lane");
        if pos >= sl.tokens.len() {
            sl.tokens.resize(pos + 1, 0);
        }
        sl.tokens[pos] = tok;
        self.eng.on(&sl.lane).set_tokens(pos, &[tok]);
    }

    fn prepare_prefill(&self, spans: &[(LaneId, usize, usize)]) {
        self.poison_free_slots();
        let mut v = self.lanes.borrow_mut();
        let mut pools = self.eng.block_pool().borrow_mut();
        let mut cache = self.cache.borrow_mut();
        for &(l, p0, t) in spans {
            let sl = v[l].as_mut().expect("live lane");
            let mut tbl = sl.lane.block_table().borrow_mut();
            cache.ensure_span_blocks(&mut pools, &mut tbl, p0, t).expect("ServePoolCandidate ensure_span_blocks");
        }
    }

    fn prepare_decode(&self, spans: &[(LaneId, usize)]) {
        self.poison_free_slots();
        let mut v = self.lanes.borrow_mut();
        let mut pools = self.eng.block_pool().borrow_mut();
        let mut cache = self.cache.borrow_mut();
        for &(l, pos) in spans {
            let sl = v[l].as_mut().expect("live lane");
            let mut tbl = sl.lane.block_table().borrow_mut();
            cache.ensure_span_blocks(&mut pools, &mut tbl, pos, 1).expect("ServePoolCandidate decode ensure_span_blocks");
        }
    }

    fn prefill(&self, spans: &[(LaneId, usize, usize)]) {
        {
            let mut v = self.lanes.borrow_mut();
            let mut pools = self.eng.block_pool().borrow_mut();
            let mut cache = self.cache.borrow_mut();
            for &(l, p0, t) in spans {
                let sl = v[l].as_mut().expect("live lane");
                let mut tbl = sl.lane.block_table().borrow_mut();
                cache.ensure_span_blocks(&mut pools, &mut tbl, p0, t).expect("ServePoolCandidate ensure_span_blocks");
            }
        }
        {
            let v = self.lanes.borrow();
            let sp: Vec<Span> = spans
                .iter()
                .map(|&(l, p0, t)| Span { lane: &v[l].as_ref().unwrap().lane, p0, t })
                .collect();
            let cb = self.eng.gpu.cmd();
            self.eng.encode_spans(&cb, &sp, true, false);
            cb.commit();
            cb.waitUntilCompleted();
        }
        {
            let mut v = self.lanes.borrow_mut();
            for &(l, p0, t) in spans {
                let sl = v[l].as_mut().expect("live lane");
                if sl.sabotaged {
                    // Intentionally poisoned lane under INK_POOLGATE_SABOTAGE: update its own block lengths
                    // and sliding window without publishing poisoned KV into ServeBlockCache's shared EntryTree.
                    let end = p0 + t;
                    sl.blocks.written = end;
                    let mut pools = self.eng.block_pool().borrow_mut();
                    let mut tbl = sl.lane.block_table().borrow_mut();
                    for b in (p0 / BLOCK)..=((end - 1) / BLOCK) {
                        let bi = BlockIndex::new(b as u32);
                        let block_len = (end - b * BLOCK).min(BLOCK) as u32;
                        for kind in KINDS {
                            let e = tbl.get(kind, bi).unwrap().expect("sabotaged span block");
                            let rec = pools.of_mut(kind).record_mut(e.slot()).unwrap();
                            rec.set_length(block_len).expect("set sabotaged length");
                            if block_len == BLOCK as u32 {
                                rec.seal();
                            }
                        }
                    }
                    let min_win_blk = end.saturating_sub(EXT - 1) / BLOCK;
                    for b in 0..min_win_blk {
                        let bi = BlockIndex::new(b as u32);
                        if tbl.get(BlockKind::Window, bi).unwrap().is_some() {
                            tbl.clear_kind_entry(&mut pools.window, bi).unwrap();
                        }
                    }
                } else {
                    let cs_fn = || self.eng.on(&sl.lane).get_cs();
                    let dedups_before = self.cache.borrow().stats().shared_dedups;
                    {
                        let mut pools = self.eng.block_pool().borrow_mut();
                        let mut tbl = sl.lane.block_table().borrow_mut();
                        self.cache.borrow_mut().finish_prefill_span(
                            &mut pools,
                            &mut tbl,
                            &mut sl.blocks,
                            &sl.tokens,
                            p0,
                            t,
                            cs_fn,
                        );
                    }
                    let dedups_after = self.cache.borrow().stats().shared_dedups;
                    if self.cache_sabotage && !self.cache_sabotage_triggered.get() && dedups_after > dedups_before {
                        // Planted ServeBlockCache bug on shared-prefix block deduplication: zero the deduplicated
                        // lane's shared block 0 entry in its GPU block table / KV or corrupt its recurrent CS!
                        let mut bad_cs = self.eng.on(&sl.lane).get_cs();
                        bad_cs.fill(0x7f);
                        self.eng.on(&sl.lane).set_cs(&bad_cs);
                        self.cache_sabotage_triggered.set(true);
                        println!("INK_POOLGATE_CACHE_SABOTAGE: corrupted lane {l} state after ServeBlockCache shared-prefix dedup");
                    }
                }
            }
        }
        self.rows.set((spans.len(), false));
    }

    fn decode(&self, spans: &[(LaneId, usize)]) {
        {
            let mut v = self.lanes.borrow_mut();
            let mut pools = self.eng.block_pool().borrow_mut();
            let mut cache = self.cache.borrow_mut();
            for &(l, pos) in spans {
                let sl = v[l].as_mut().expect("live lane");
                let mut tbl = sl.lane.block_table().borrow_mut();
                cache.ensure_span_blocks(&mut pools, &mut tbl, pos, 1).expect("ServePoolCandidate decode ensure_span_blocks");
            }
        }
        {
            let v = self.lanes.borrow();
            let sp: Vec<Span> = spans
                .iter()
                .map(|&(l, p0)| Span { lane: &v[l].as_ref().unwrap().lane, p0, t: 1 })
                .collect();
            let cb = self.eng.gpu.cmd();
            self.eng.encode_decode(&cb, &sp, true, false);
            cb.commit();
            cb.waitUntilCompleted();
        }
        {
            let mut v = self.lanes.borrow_mut();
            let mut pools = self.eng.block_pool().borrow_mut();
            let mut cache = self.cache.borrow_mut();
            for &(l, pos) in spans {
                let sl = v[l].as_mut().expect("live lane");
                let mut tbl = sl.lane.block_table().borrow_mut();
                cache.finish_decode_span(&mut pools, &mut tbl, &mut sl.blocks, pos);
            }
        }
        self.rows.set((spans.len(), true));
    }

    fn logits(&self, row: usize) -> Vec<f32> {
        let (n, dec) = self.rows.get();
        assert!(row < n, "logits row {row} of a step of {n} spans");
        (if n == 1 && !dec { self.eng.logits() } else { self.eng.span_logits(row) }).to_vec()
    }

    fn end_lane(&self, lane: LaneId) {
        let Some(mut sl) = self.lanes.borrow_mut()[lane].take() else { return };
        let mut pools = self.eng.block_pool().borrow_mut();
        let mut tbl = sl.lane.block_table().borrow_mut();
        self.cache.borrow_mut().release_lane(&mut pools, &mut tbl, &mut sl.blocks);
    }
}

impl PoisonPool for ServePoolCandidate<'_> {
    fn slot_count(&self, kind: BlockKind) -> usize {
        self.eng.block_pool().borrow().of(kind).len()
    }
    fn lane_slots(&self, lane: LaneId) -> Vec<(BlockKind, Slot)> {
        let v = self.lanes.borrow();
        let sl = v[lane].as_ref().expect("live lane");
        let slots = table_slots(&sl.lane.block_table().borrow());
        slots
    }
    fn slot_bytes(&self, kind: BlockKind, slot: Slot) -> Option<(*mut u8, usize)> {
        self.eng.block_pool().borrow().of(kind).slot_raw_region(slot)
    }
    fn note_sabotaged(&self, lane: LaneId) {
        if let Some(sl) = self.lanes.borrow_mut()[lane].as_mut() {
            sl.sabotaged = true;
        }
    }
    fn retained_slots(&self) -> Vec<(BlockKind, Slot)> {
        let pools = self.eng.block_pool().borrow();
        let mut out = Vec::new();
        for kind in KINDS {
            let pool = pools.of(kind);
            for i in 0..pool.len() {
                let s = slot_at(i);
                if pool.record(s).unwrap().refs() > 0 {
                    out.push((kind, s));
                }
            }
        }
        out
    }
}

/// Exercise `ServeBlockCache` start-point matching (`PROMPT` @ 192, window-pruned `PROMPT` @ 1152 past
/// `EXT = 1024`, and `EVERY` @ 2048 when `ctx >= 2304`), slot reuse, and diverging multi-lane forks
/// with every free pool slot poisoned, asserting non-zero sharing, reuse, and window pruning.
pub fn verify_serve_cache_start_points(eng: &Engine, base: &[u32], ctx: usize, cache_sabotage: bool) -> Result<String, String> {
    if ctx < 1408 || base.len() < 1408 {
        return Err(format!("verify_serve_cache_start_points requires ctx >= 1408 and base.len() >= 1408 (got ctx={ctx}, base={})", base.len()));
    }
    let run_cold_l2 = |toks: &[u32], ndec: usize| -> Record {
        let l2 = Layer2::new(eng, 1);
        let lid = l2.new_lane(toks, toks.len() + ndec + 1);
        let mut rec = Record::default();
        let mut p = 0;
        while p < toks.len() {
            let c = eng.chunk.min(toks.len() - p);
            l2.prefill(&[(lid, p, c)]);
            p += c;
        }
        let lg = l2.logits(0);
        rec.steps.push(fnv(&lg));
        let mut tok = argmax(&lg);
        rec.tokens.push(tok);
        l2.set_token(lid, toks.len(), tok);
        for d in 0..ndec {
            let pos = toks.len() + d;
            l2.decode(&[(lid, pos)]);
            let lg = l2.logits(0);
            rec.steps.push(fnv(&lg));
            tok = argmax(&lg);
            rec.tokens.push(tok);
            l2.set_token(lid, pos + 1, tok);
        }
        l2.end_lane(lid);
        rec
    };

    let rot = |k: usize, len: usize| -> Vec<u32> {
        let mut v = base.to_vec();
        v.rotate_left(k % base.len());
        v.truncate(len);
        v
    };

    let max_pos = if ctx >= 2304 && base.len() >= 2304 { 2240 } else { 1280 };
    // Tight pool sizing guarantees freed unsealed/window-pruned slots are reused across turns (`slots_reused > 0`).
    let sc = ServePoolCandidate::new(eng, 2, max_pos, cache_sabotage);

    let run_matched = |toks: &[u32], expected_match: usize, ndec: usize| -> Result<Record, String> {
        sc.poison_free_slots();
        let (lid, matched) = sc.new_lane_matched(toks, toks.len() + ndec + 2);
        if matched != expected_match {
            sc.end_lane(lid);
            return Err(format!("ServeBlockCache expected match at {expected_match}, got {matched}"));
        }
        let mut p = matched;
        while p < toks.len() {
            let target = crate::serve::block_cache::effective_pf_target(p, toks.len());
            let c = eng.chunk.min(target - p);
            sc.poison_free_slots();
            sc.prefill(&[(lid, p, c)]);
            p += c;
        }
        let mut rec = Record::default();
        let lg = sc.logits(0);
        rec.steps.push(fnv(&lg));
        let mut tok = argmax(&lg);
        rec.tokens.push(tok);
        sc.set_token(lid, toks.len(), tok);
        for d in 0..ndec {
            let pos = toks.len() + d;
            sc.poison_free_slots();
            sc.decode(&[(lid, pos)]);
            let lg = sc.logits(0);
            rec.steps.push(fnv(&lg));
            tok = argmax(&lg);
            rec.tokens.push(tok);
            sc.set_token(lid, pos + 1, tok);
        }
        sc.end_lane(lid);
        Ok(rec)
    };

    // 1. Seed PROMPT checkpoint at 192 tokens (prompt length 195 -> prompt_ckpt = 192) and decode 4 tokens
    //    into block 3 so releasing this lane returns unsealed tail blocks to the pool for `slots_reused`.
    let mut seed195 = base[..192].to_vec();
    seed195.extend_from_slice(&rot(41, 3));
    let _ = run_matched(&seed195, 0, 4)?;

    // Two diverging continuations sharing the 192-token PROMPT start point run concurrently under PoisonPool!
    let mut fork_a = base[..192].to_vec();
    fork_a.extend_from_slice(&rot(173, 37));
    let mut fork_b = base[..192].to_vec();
    fork_b.extend_from_slice(&rot(311, 37));
    let want_a = run_cold_l2(&fork_a, 4);
    let want_b = run_cold_l2(&fork_b, 4);
    sc.poison_free_slots();
    let (la, ma) = sc.new_lane_matched(&fork_a, fork_a.len() + 6);
    let (lb, mb) = sc.new_lane_matched(&fork_b, fork_b.len() + 6);
    if ma != 192 || mb != 192 {
        sc.shutdown();
        return Err(format!("expected concurrent forks to match 192, got ({ma}, {mb})"));
    }
    sc.poison_free_slots();
    sc.prefill(&[(la, 192, 37), (lb, 192, 37)]);
    let (mut rec_a, mut rec_b) = (Record::default(), Record::default());
    for (k, (lid, prompt, rec)) in [(la, &fork_a, &mut rec_a), (lb, &fork_b, &mut rec_b)].into_iter().enumerate() {
        let lg = sc.logits(k);
        rec.steps.push(fnv(&lg));
        let tok = argmax(&lg);
        rec.tokens.push(tok);
        sc.set_token(lid, prompt.len(), tok);
    }
    for d in 0..4 {
        let pos = fork_a.len() + d;
        sc.poison_free_slots();
        sc.decode(&[(la, pos), (lb, pos)]);
        for (k, (lid, rec)) in [(la, &mut rec_a), (lb, &mut rec_b)].into_iter().enumerate() {
            let lg = sc.logits(k);
            rec.steps.push(fnv(&lg));
            let tok = argmax(&lg);
            rec.tokens.push(tok);
            sc.set_token(lid, pos + 1, tok);
        }
    }
    sc.end_lane(la);
    sc.end_lane(lb);
    if rec_a != want_a || rec_b != want_b {
        sc.shutdown();
        return Err("ServeBlockCache concurrent shared-prefix forks differed from cold Layer 2".into());
    }

    // 2. Cross the 1024-token sliding window (`PROMPT` @ 1152, pruning window blocks 0..2), then match & fork!
    let mut seed1155 = base[..1152].to_vec();
    seed1155.extend_from_slice(&rot(601, 3));
    let _ = run_matched(&seed1155, 192, 2)?;

    let mut prompt_1152_fork = base[..1152].to_vec();
    prompt_1152_fork.extend_from_slice(&rot(733, 45));
    let want_1152 = run_cold_l2(&prompt_1152_fork, 4);
    let got_1152 = run_matched(&prompt_1152_fork, 1152, 4)?;
    if got_1152 != want_1152 {
        sc.shutdown();
        return Err("ServeBlockCache window-pruned PROMPT@1152 fork differed from cold Layer 2".into());
    }

    // 3. Also exercise periodic `EVERY` @ 2048 checkpoint when ctx >= 2304.
    let mut tested_every_2048 = false;
    if ctx >= 2304 && base.len() >= 2304 {
        tested_every_2048 = true;
        let mut seed2115 = base[..2112].to_vec();
        seed2115.extend_from_slice(&rot(857, 3));
        let _ = run_matched(&seed2115, 1152, 2)?;

        let mut fork_2048 = base[..2048].to_vec();
        fork_2048.extend_from_slice(&rot(911, 37));
        let want_2048 = run_cold_l2(&fork_2048, 3);
        let got_2048 = run_matched(&fork_2048, 2048, 3)?;
        if got_2048 != want_2048 {
            sc.shutdown();
            return Err("ServeBlockCache periodic EVERY@2048 fork differed from cold Layer 2".into());
        }
    }

    let st = sc.stats();
    sc.shutdown();
    {
        let pools = eng.block_pool().borrow();
        for kind in KINDS {
            let pool = pools.of(kind);
            for i in 0..pool.len() {
                let r = pool.record(slot_at(i)).unwrap().refs();
                if r != 0 {
                    return Err(format!("ServeBlockCache refcount leak on {kind:?} slot {}: refs={r}", i + 1));
                }
            }
        }
    }
    if st.start_matches == 0 || st.slots_reused == 0 || st.window_pruned == 0 {
        return Err(format!(
            "ServeBlockCache gate failed non-vacuity check: matches={}, reused={}, window_pruned={}",
            st.start_matches, st.slots_reused, st.window_pruned
        ));
    }
    Ok(format!(
        "ServeBlockCache start-point & window-pruning gate identical to Layer 2 (192, PROMPT@1152{}: matches={}, saved={}, reused={}, window_pruned={})",
        if tested_every_2048 { ", EVERY@2048" } else { "" },
        st.start_matches, st.start_points_saved, st.slots_reused, st.window_pruned
    ))
}

/// Verify deepest start-point prefix matching (`PrefixTree`) + recurrent state & window restoration
/// + sliding-window pruning past 1024 tokens against cold Layer 2, with all free pool slots poisoned.
pub fn verify_prefix_start_points(eng: &Engine, base: &[u32], ctx: usize, sabotage: bool) -> Result<String, String> {
    check_prefix_start_point_inputs(base.len(), ctx)?;
    let run_cold_l2 = |toks: &[u32], ndec: usize| -> Record {
        let l2 = Layer2::new(eng, 1);
        let lid = l2.new_lane(toks, toks.len() + ndec + 1);
        let mut rec = Record::default();
        let mut p = 0;
        while p < toks.len() {
            let c = eng.chunk.min(toks.len() - p);
            l2.prefill(&[(lid, p, c)]);
            p += c;
        }
        let lg = l2.logits(0);
        rec.steps.push(fnv(&lg));
        let mut tok = argmax(&lg);
        rec.tokens.push(tok);
        l2.set_token(lid, toks.len(), tok);
        for d in 0..ndec {
            let pos = toks.len() + d;
            l2.decode(&[(lid, pos)]);
            let lg = l2.logits(0);
            rec.steps.push(fnv(&lg));
            tok = argmax(&lg);
            rec.tokens.push(tok);
            l2.set_token(lid, pos + 1, tok);
        }
        l2.end_lane(lid);
        rec
    };

    let run_matched_l3 = |l3: &Layer3Pool, toks: &[u32], expected_match: usize, ndec: usize| -> Result<Record, String> {
        l3.poison_free_slots();
        let (lid, matched) = l3.new_lane_matched(toks, toks.len() + ndec + 1);
        if matched != expected_match {
            l3.end_lane(lid);
            return Err(format!("expected prefix start-point match at {expected_match}, got {matched}"));
        }
        let mut p = matched;
        while p < toks.len() {
            let c = eng.chunk.min(toks.len() - p);
            l3.prefill(&[(lid, p, c)]);
            p += c;
        }
        let mut rec = Record::default();
        let lg = l3.logits(0);
        rec.steps.push(fnv(&lg));
        let mut tok = argmax(&lg);
        rec.tokens.push(tok);
        l3.set_token(lid, toks.len(), tok);
        for d in 0..ndec {
            let pos = toks.len() + d;
            l3.poison_free_slots();
            l3.decode(&[(lid, pos)]);
            let lg = l3.logits(0);
            rec.steps.push(fnv(&lg));
            tok = argmax(&lg);
            rec.tokens.push(tok);
            l3.set_token(lid, pos + 1, tok);
        }
        l3.end_lane(lid);
        Ok(rec)
    };

    let rot = |k: usize, len: usize| -> Vec<u32> {
        let mut v = base.to_vec();
        v.rotate_left(k % base.len());
        v.truncate(len);
        v
    };

    let l3 = Layer3Pool::with_persistent_starts(eng, 2);
    // 1. Prompt-end start point at 192 tokens (3 full blocks).
    let head192 = base[..192].to_vec();
    let (la, m0) = l3.new_lane_matched(&head192, 256);
    if m0 != 0 { return Err(format!("cold tree matched at {m0}")); }
    l3.prefill(&[(la, 0, 64)]);
    l3.prefill(&[(la, 64, 128)]);
    l3.end_lane(la);

    let mut prompt_b = head192.clone();
    prompt_b.extend_from_slice(&rot(173, 37));
    let want_b = run_cold_l2(&prompt_b, 6);
    let got_b = run_matched_l3(&l3, &prompt_b, 192, 6)?;
    if want_b != got_b {
        return Err(format!("prefix-matched 192+37 prompt differs from Layer 2: want {want_b:?} got {got_b:?}"));
    }

    // 2. Every-16-blocks start point at 1024 AND prompt-end start point at 1152 (past WINDOW=1024).
    let head1152 = base[..1152].to_vec();
    let (lc, mc) = l3.new_lane_matched(&head1152, 1280);
    if mc != 192 {
        l3.end_lane(lc);
        l3.shutdown();
        return Err(format!("expected 1152 prompt to reuse 192 start point, got {mc}"));
    }
    // Step to exact block 16 (1024) so EVERY start point is saved, then to 1152 (PROMPT start point).
    l3.prefill(&[(lc, 192, 320)]);  // -> 512
    l3.prefill(&[(lc, 512, 512)]);  // -> 1024 (block 16: EVERY start point!)
    l3.prefill(&[(lc, 1024, 128)]); // -> 1152 (block 18: PROMPT start point, prunes window 0..2 on lane!)
    l3.end_lane(lc);

    if sabotage {
        let mut corrupted = false;
        for st_rec in l3.starts.borrow_mut().values_mut() {
            if st_rec.pos == 1152 {
                corrupt_start_cs_for_sabotage(&mut st_rec.cs);
                corrupted = true;
            }
        }
        if !corrupted {
            l3.shutdown();
            return Err("INK_POOLGATE_START_SABOTAGE: retained start point at 1152 not found".into());
        }
        println!("INK_POOLGATE_START_SABOTAGE: corrupted retained start-point state at 1152");
    }

    // Request D diverges after 1024 -> must match the EVERY start point at 1024 (window 0..16 retained!).
    let mut prompt_d = base[..1024].to_vec();
    prompt_d.extend_from_slice(&rot(509, 45));
    let want_d = run_cold_l2(&prompt_d, 4);
    let got_d = match run_matched_l3(&l3, &prompt_d, 1024, 4) {
        Ok(r) => r,
        Err(e) => { l3.shutdown(); return Err(e); }
    };
    if want_d != got_d {
        l3.shutdown();
        return Err(format!("EVERY-1024 start-point match differs from Layer 2: want {want_d:?} got {got_d:?}"));
    }

    // Request E diverges after 1152 -> must match PROMPT start point at 1152 (window 2..18, blocks 0..1 pruned!).
    let mut prompt_e = base[..1152].to_vec();
    prompt_e.extend_from_slice(&rot(733, 45));
    let want_e = run_cold_l2(&prompt_e, 4);
    let got_e = match run_matched_l3(&l3, &prompt_e, 1152, 4) {
        Ok(r) => r,
        Err(e) => { l3.shutdown(); return Err(e); }
    };
    if want_e != got_e {
        l3.shutdown();
        return Err(format!("PROMPT-1152 window-pruned start-point match differs from Layer 2: want {want_e:?} got {got_e:?}"));
    }

    let st = l3.stats();
    l3.shutdown();
    if st.start_points_saved < 3 || st.start_matches < 4 || st.window_pruned == 0 || st.slots_reused == 0 {
        return Err(format!(
            "prefix start-point stats incomplete: saved={} matches={} pruned={} reused={}",
            st.start_points_saved, st.start_matches, st.window_pruned, st.slots_reused
        ));
    }
    // Verify zero refcount leaks after shutdown.
    {
        let pools = eng.block_pool().borrow();
        for kind in KINDS {
            let pool = pools.of(kind);
            for i in 0..pool.len() {
                let r = pool.record(slot_at(i)).unwrap().refs();
                if r != 0 { return Err(format!("refcount leak on {kind:?} slot {}: refs={r}", i + 1)); }
            }
        }
    }
    Ok(format!(
        "prefix start-point reuse identical to Layer 2 (192, EVERY@1024, PROMPT@1152: matches={}, saved={}, pruned={}, reused={})",
        st.start_matches, st.start_points_saved, st.window_pruned, st.slots_reused
    ))
}
}

#[cfg(not(standalone_blocks))]
pub use gpu_paths::{
    verify_prefix_start_points, verify_serve_cache_start_points, Layer2, Layer3Pool,
    ServePoolCandidate,
};
#[cfg(all(test, not(standalone_blocks)))]
pub use gpu_paths::verify_l3_workload_stats;

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    // Slot arithmetic only: no GPU, no time, no files.
    fn check(n: usize, tables: &[Vec<usize>], stepping: &[usize]) {
        let p = poison_plan(n, tables, stepping, true);
        let read: Vec<usize> = stepping.iter().flat_map(|&i| tables[i].clone()).collect();
        let owned: Vec<usize> = tables.iter().flatten().copied().collect();
        for s in 0..n {
            // NoFalsePoison and Coverage (specs/PoisonGate.tla)
            assert_eq!(p.poison.contains(&s), !read.contains(&s), "slot {s}");
            // every poisoned slot a live lane owns is saved, and only those
            assert_eq!(p.save.contains(&s), p.poison.contains(&s) && owned.contains(&s), "slot {s}");
        }
        assert!(p.poison.windows(2).all(|w| w[0] < w[1]) && p.save.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn plan_protects_every_stepping_lane_and_saves_the_rest() {
        let t = vec![vec![0, 1], vec![1, 2], vec![4]];
        check(6, &t, &[0]);
        check(6, &t, &[1]);
        check(6, &t, &[0, 2]);
        check(6, &t, &[2, 1, 0]);
        // slot 1 is shared by lanes 0 and 1: protected when either steps, saved when neither does
        assert_eq!(poison_plan(6, &t, &[2], true), Plan { poison: vec![0, 1, 2, 3, 5], save: vec![0, 1, 2] });
        // free slots (3, 5) are poisoned and never saved
        assert_eq!(poison_plan(6, &t, &[0, 1, 2], true), Plan { poison: vec![3, 5], save: vec![] });
    }

    #[test]
    fn every_small_pool_matches_the_model() {
        // every table assignment of 2 lanes over 3 slots, every nonempty stepping set
        for a in 0..8usize { for b in 0..8usize {
            let bits = |m: usize| (0..3).filter(|s| m >> s & 1 == 1).collect::<Vec<_>>();
            let t = vec![bits(a), bits(b)];
            for st in [vec![0], vec![1], vec![0, 1]] { check(3, &t, &st); }
        } }
    }

    #[test]
    fn unguarded_plan_poisons_a_stepping_lane() {
        // the reversed witness: without the guard the first stepping lane's own slots are poisoned (and saved)
        let t = vec![vec![0], vec![1]];
        let p = poison_plan(3, &t, &[1, 0], false);
        assert!(p.poison.contains(&1) && p.save.contains(&1) && !p.poison.contains(&0));
    }

    #[test]
    fn poison_is_nan_and_huge_in_both_widths() {
        let mut b = vec![0u8; 18];
        poison_bytes(&mut b);
        let w0 = u32::from_ne_bytes(b[0..4].try_into().unwrap());
        let w1 = u32::from_ne_bytes(b[4..8].try_into().unwrap());
        assert!(f32::from_bits(w0).is_nan() && f32::from_bits(w1) > 1e30);
        let h = |w: u32| (w & 0xffff) as u16;
        assert!(h(w0) & 0x7c00 == 0x7c00 && h(w0) & 0x3ff != 0);   // fp16 NaN
        assert_eq!(h(w1), 0x7bff);                                   // fp16 65504
        assert_eq!(&b[16..], &[0x7f, 0x7f]);
    }

    /// A pool on the heap with a fake path, to check `apply` and `restore` round trip without a GPU.
    /// Both pools on the heap (window slots then full slots), to check `apply` and `restore` without a GPU.
    struct Heap { win: RefCell<Vec<Vec<u8>>>, full: RefCell<Vec<Vec<u8>>>, tables: Vec<Vec<(BlockKind, Slot)>>,
        /// (kind, pool index) with no memory now, and those of them that are legitimately absent (an ended lane)
        gone: Vec<(BlockKind, usize)>, ended: Vec<(BlockKind, usize)> }
    impl PoisonPool for Heap {
        fn slot_count(&self, k: BlockKind) -> usize { if k == BlockKind::Window { self.win.borrow().len() } else { self.full.borrow().len() } }
        fn lane_slots(&self, l: LaneId) -> Vec<(BlockKind, Slot)> { self.tables[l].clone() }
        fn slot_bytes(&self, k: BlockKind, s: Slot) -> Option<(*mut u8, usize)> {
            if self.gone.contains(&(k, s.pool_index().get() as usize)) { return None; }
            let mut v = if k == BlockKind::Window { self.win.borrow_mut() } else { self.full.borrow_mut() };
            let x = &mut v[s.pool_index().get() as usize];
            Some((x.as_mut_ptr(), x.len()))
        }
        fn absent_ok(&self, k: BlockKind, s: Slot) -> bool { self.ended.contains(&(k, s.pool_index().get() as usize)) }
    }
    fn nan(v: &[u8]) -> bool { f32::from_bits(u32::from_ne_bytes(v[0..4].try_into().unwrap())).is_nan() }
    fn sl(n: u32) -> Slot { Slot::new(n).unwrap() }

    #[test]
    fn apply_poisons_and_restore_puts_back_only_referenced_slots() {
        use BlockKind::*;
        // lane 0: window 1, full 1. lane 1: window 2 and 3, full 1 (shared). window 4 and full 2 are free.
        let h = Heap { win: RefCell::new((0..4u8).map(|i| vec![i; 8]).collect()), full: RefCell::new((10..12u8).map(|i| vec![i; 8]).collect()),
                       tables: vec![vec![(Window, sl(1)), (Full, sl(1))], vec![(Window, sl(2)), (Window, sl(3)), (Full, sl(1))]], gone: vec![], ended: vec![] };
        let tables: Vec<_> = (0..2).map(|l| h.lane_slots(l)).collect();
        let ps = plans(&h, &tables, &[0], true);
        assert_eq!(ps, vec![(Window, Plan { poison: vec![1, 2, 3], save: vec![1, 2] }), (Full, Plan { poison: vec![1], save: vec![] })]);
        let (saved, done) = apply(&h, &ps);
        assert_eq!(done, Applied { saved: 2, poisoned: 4, absent: 0 });
        assert_eq!(h.win.borrow()[0], vec![0; 8]);
        assert_eq!(h.full.borrow()[0], vec![10; 8]);
        for s in 1..4 { assert!(nan(&h.win.borrow()[s])); }
        assert!(nan(&h.full.borrow()[1]));
        restore(&h, saved);
        assert_eq!(h.win.borrow()[1], vec![1; 8]);
        assert_eq!(h.win.borrow()[2], vec![2; 8]);
        // free slots stay poisoned: the pool zeroes a slot when it hands it out
        assert!(nan(&h.win.borrow()[3]) && nan(&h.full.borrow()[1]));
    }

    fn heap_with_gone(gone: Vec<(BlockKind, usize)>, ended: Vec<(BlockKind, usize)>) -> Heap {
        Heap { win: RefCell::new(vec![vec![1; 8]; 3]), full: RefCell::new(vec![vec![2; 8]; 1]),
               tables: vec![vec![(BlockKind::Window, sl(1))]], gone, ended }
    }

    #[test]
    fn apply_counts_an_ended_lanes_absent_slot_and_poisons_the_rest() {
        // window pool index 2 has no memory and is marked absent: counted as absent, not as poisoned
        let h = heap_with_gone(vec![(BlockKind::Window, 2)], vec![(BlockKind::Window, 2)]);
        let ps = plans(&h, &[h.lane_slots(0)], &[0], true);
        let (_, done) = apply(&h, &ps);
        // planned: window poison [1, 2], full poison [0]; one of the three is absent, so two applied
        assert_eq!(ps.iter().map(|(_, p)| p.poison.len()).sum::<usize>(), 3);
        assert_eq!(done, Applied { saved: 0, poisoned: 2, absent: 1 });
    }

    #[test]
    #[should_panic(expected = "Window slot 3 (pool index 2) of 3 has no memory and is not marked absent")]
    fn apply_fails_loudly_on_a_slot_with_no_memory_that_is_not_marked_absent() {
        // the skip this replaces: a slot below slot_count with no memory and no ended lane was passed over uncounted
        let h = heap_with_gone(vec![(BlockKind::Window, 2)], vec![]);
        let ps = plans(&h, &[h.lane_slots(0)], &[0], true);
        apply(&h, &ps);
    }

    #[test]
    #[should_panic(expected = "referenced by a live lane and has no memory")]
    fn apply_fails_loudly_on_a_live_lanes_slot_with_no_memory() {
        // lane 0 references window slot 1 and is not stepping; its slot is to be saved, and is gone (even if marked ended)
        let h = Heap { win: RefCell::new(vec![vec![1; 8]; 2]), full: RefCell::new(vec![vec![2; 8]; 1]),
                       tables: vec![vec![(BlockKind::Window, sl(1))], vec![(BlockKind::Window, sl(2))]],
                       gone: vec![(BlockKind::Window, 0)], ended: vec![(BlockKind::Window, 0)] };
        let tables: Vec<_> = (0..2).map(|l| h.lane_slots(l)).collect();
        let ps = plans(&h, &tables, &[1], true);
        apply(&h, &ps);
    }

    #[test]
    fn table_slots_reads_every_entry_of_both_kinds() {
        assert_eq!(table_slots(&BlockTable::new(4).unwrap()), vec![]);
        assert_eq!(table_slots(&BlockTable::empty()), vec![]);
    }

    #[test]
    fn leaked_names_every_other_lane_that_differs() {
        let r = |s: u64| Record { steps: vec![1, s], tokens: vec![7] };
        let want = vec![r(2), r(3), r(4)];
        let got = |v: Vec<Record>| Report { records: v, steps: 0, poisoned: 0, saved: 0, absent: 0, sabotaged: Some(0), stray_writes: vec![] };
        assert_eq!(leaked(&want, &got(vec![r(9), r(3), r(4)]), 0), Vec::<usize>::new());
        assert_eq!(leaked(&want, &got(vec![r(9), r(9), r(4)]), 0), vec![1]);
        assert_eq!(leaked(&want, &got(vec![r(2), r(3), Record { steps: vec![1, 4], tokens: vec![8] }]), 1), vec![2]);
    }

    #[test]
    fn compare_rejects_short_or_extra_candidate_record_lists() {
        let r0 = Record { steps: vec![1, 2], tokens: vec![10] };
        let r1 = Record { steps: vec![3, 4], tokens: vec![20] };
        let want = vec![r0.clone(), r1.clone()];
        let short = Report { records: vec![r0.clone()], steps: 2, poisoned: 1, saved: 0, absent: 0, sabotaged: None, stray_writes: vec![] };
        let err = compare(&want, &short).unwrap_err();
        assert!(err[0].contains("lane count mismatch"), "short candidate must fail compare: {err:?}");
        let extra = Report { records: vec![r0.clone(), r1.clone(), r0.clone()], steps: 2, poisoned: 1, saved: 0, absent: 0, sabotaged: None, stray_writes: vec![] };
        assert!(compare(&want, &extra).is_err(), "extra candidate lanes must fail compare");
        let exact = Report { records: vec![r0, r1], steps: 2, poisoned: 1, saved: 0, absent: 0, sabotaged: None, stray_writes: vec![] };
        assert!(compare(&want, &exact).is_ok());
    }

    #[test]
    #[should_panic(expected = "stray write into unreferenced poisoned slot (Window slot 4, pool index 3) at byte 17/18")]
    fn restore_catches_stray_write_to_unreferenced_poisoned_slot() {
        use BlockKind::*;
        let h = Heap {
            win: RefCell::new((0..4u8).map(|i| vec![i; 18]).collect()),
            full: RefCell::new((10..12u8).map(|i| vec![i; 16]).collect()),
            tables: vec![vec![(Window, sl(1)), (Full, sl(1))], vec![(Window, sl(2)), (Full, sl(1))]],
            gone: vec![],
            ended: vec![],
        };
        let tables: Vec<_> = (0..2).map(|l| h.lane_slots(l)).collect();
        let ps = plans(&h, &tables, &[0], true);
        let (saved, _) = apply(&h, &ps);
        // Simulate a stray kernel write to the very last tail byte of unreferenced Window slot 4 (pool index 3).
        h.win.borrow_mut()[3][17] = 0x00;
        restore(&h, saved);
    }

    #[test]
    #[should_panic(expected = "stray write into non-stepping live lane's saved slot (Window slot 2, pool index 1) at byte 5/16")]
    fn restore_catches_stray_write_to_non_stepping_lanes_saved_slot_before_overwrite() {
        use BlockKind::*;
        let h = Heap {
            win: RefCell::new((0..3u8).map(|i| vec![i + 1; 16]).collect()),
            full: RefCell::new(vec![vec![9; 16]]),
            tables: vec![vec![(Window, sl(1)), (Full, sl(1))], vec![(Window, sl(2)), (Full, sl(1))]],
            gone: vec![],
            ended: vec![],
        };
        let tables: Vec<_> = (0..2).map(|l| h.lane_slots(l)).collect();
        // Lane 0 steps; lane 1 is live and non-stepping, so Window slot 2 (pool index 1) is saved & poisoned.
        let ps = plans(&h, &tables, &[0], true);
        let (saved, _) = apply(&h, &ps);
        // Simulate a stray write during lane 0's step into non-stepping lane 1's saved slot BEFORE restore runs.
        h.win.borrow_mut()[1][5] ^= 0x01;
        restore(&h, saved);
    }

    #[test]
    fn poisoned_gate_catches_stray_writes_to_both_unreferenced_and_saved_slots() {
        use BlockKind::*;
        struct StrayCand {
            win: RefCell<Vec<Vec<u8>>>,
            full: RefCell<Vec<Vec<u8>>>,
            lanes: RefCell<Vec<Option<Vec<u32>>>>,
            rows: Cell<usize>,
            corrupt_saved: bool,
            corrupt_free: bool,
            did_corrupt: Cell<bool>,
        }
        impl StepPath for StrayCand {
            fn new_lane(&self, toks: &[u32], _cap: usize) -> LaneId {
                let mut v = self.lanes.borrow_mut();
                v.push(Some(toks.to_vec()));
                v.len() - 1
            }
            fn set_token(&self, lane: LaneId, pos: usize, tok: u32) {
                let mut v = self.lanes.borrow_mut();
                let t = v[lane].as_mut().unwrap();
                if pos >= t.len() { t.resize(pos + 1, 0); }
                t[pos] = tok;
            }
            fn prefill(&self, spans: &[(LaneId, usize, usize)]) {
                self.rows.set(spans.len());
                if !self.did_corrupt.get() && spans.len() == 1 {
                    let l = spans[0].0;
                    // Write legitimately to stepping lane `l`'s own slot (`l`).
                    self.win.borrow_mut()[l][0] = 0x42;
                }
            }
            fn decode(&self, spans: &[(LaneId, usize)]) {
                self.rows.set(spans.len());
                // At round 1 decode, lane 0 decodes while lane 1 is live in prefill (non-stepping!).
                let live_count = self.lanes.borrow().iter().filter(|x| x.is_some()).count();
                if !self.did_corrupt.get() && live_count >= 2 && spans.len() == 1 {
                    self.did_corrupt.set(true);
                    if self.corrupt_saved {
                        // Lane 1's Window slot (pool index 1) is a non-stepping live lane's saved slot!
                        self.win.borrow_mut()[1][12] = 0xAA;
                    }
                    if self.corrupt_free {
                        // Pool index 3 is an unreferenced poisoned slot!
                        self.full.borrow_mut()[3][8] = 0xBB;
                    }
                }
            }
            fn logits(&self, row: usize) -> Vec<f32> {
                assert!(row < self.rows.get());
                // Deterministic clean logits so `compare` would pass if stray writes were not checked!
                vec![1.0, 3.0, 2.0, 0.5]
            }
            fn end_lane(&self, lane: LaneId) {
                self.lanes.borrow_mut()[lane] = None;
            }
        }
        impl PoisonPool for StrayCand {
            fn slot_count(&self, _: BlockKind) -> usize { 4 }
            fn lane_slots(&self, lane: LaneId) -> Vec<(BlockKind, Slot)> {
                vec![(Window, slot_at(lane)), (Full, slot_at(lane))]
            }
            fn slot_bytes(&self, kind: BlockKind, slot: Slot) -> Option<(*mut u8, usize)> {
                let idx = slot.pool_index().get() as usize;
                let mut v = if kind == Window { self.win.borrow_mut() } else { self.full.borrow_mut() };
                let s = &mut v[idx];
                Some((s.as_mut_ptr(), s.len()))
            }
        }

        let base: Vec<u32> = (0..128).collect();
        let w = Workload::from_tokens(&base, 2, vec![4, 4], 2);
        let mk = |corrupt_saved: bool, corrupt_free: bool| StrayCand {
            win: RefCell::new(vec![vec![7u8; 16]; 4]),
            full: RefCell::new(vec![vec![8u8; 16]; 4]),
            lanes: RefCell::new(Vec::new()),
            rows: Cell::new(0),
            corrupt_saved,
            corrupt_free,
            did_corrupt: Cell::new(false),
        };
        let want = reference(&mk(false, false), &w);

        // 1. Clean candidate passes with zero stray writes.
        let clean = mk(false, false);
        let got_clean = poisoned(&clean, &w, false);
        assert!(got_clean.stray_writes.is_empty());
        assert!(compare(&want, &got_clean).is_ok());

        // 2. Stray write to non-stepping lane's saved slot fails compare even though logits match.
        let bad_saved = mk(true, false);
        let got_saved = poisoned(&bad_saved, &w, false);
        assert!(bad_saved.did_corrupt.get());
        let errs_saved = compare(&want, &got_saved).unwrap_err();
        assert!(
            errs_saved.iter().any(|e| e.contains("non-stepping live lane's saved slot (Window slot 2, pool index 1)")),
            "expected saved-slot stray write error, got: {errs_saved:?}"
        );

        // 3. Stray write to unreferenced poisoned slot fails compare even though logits match.
        let bad_free = mk(false, true);
        let got_free = poisoned(&bad_free, &w, false);
        assert!(bad_free.did_corrupt.get());
        let errs_free = compare(&want, &got_free).unwrap_err();
        assert!(
            errs_free.iter().any(|e| e.contains("unreferenced poisoned slot (Full slot 4, pool index 3)")),
            "expected unreferenced-slot stray write error, got: {errs_free:?}"
        );
    }

    #[test]
    #[cfg(not(standalone_blocks))]
    fn stats_and_start_point_guards_reject_missing_sharing_reuse_pruning_or_short_ctx() {
        use super::gpu_paths::{check_prefix_start_point_inputs, corrupt_start_cs_for_sabotage, L3Stats};
        let good_short = L3Stats {
            slots_taken: 12,
            slots_reused: 2,
            blocks_sealed: 3,
            shared_dedups: 1,
            start_points_saved: 0,
            start_matches: 0,
            window_pruned: 0,
            tails_dropped: 6,
        };
        assert!(verify_l3_workload_stats(&good_short, 3, 72, 80).is_ok());
        assert!(verify_l3_workload_stats(&L3Stats { shared_dedups: 0, ..good_short }, 3, 72, 80).is_err());
        assert!(verify_l3_workload_stats(&L3Stats { slots_reused: 0, ..good_short }, 3, 72, 80).is_err());
        assert!(verify_l3_workload_stats(&L3Stats { tails_dropped: 0, ..good_short }, 3, 72, 80).is_err());
        // Crossing EXT + BLOCK - 1 (1087) requires window_pruned > 0.
        assert!(verify_l3_workload_stats(&good_short, 17, 1067, 1131).is_err());
        assert!(verify_l3_workload_stats(&L3Stats { window_pruned: 17, ..good_short }, 17, 1067, 1131).is_ok());

        // Start-point verifier refuses to skip on short ctx or short token sequence.
        assert!(check_prefix_start_point_inputs(2404, 1408).is_ok());
        assert!(check_prefix_start_point_inputs(2404, 1024).is_err());
        assert!(check_prefix_start_point_inputs(1200, 4096).is_err());

        let mut cs = vec![0u8; 16];
        corrupt_start_cs_for_sabotage(&mut cs);
        assert!(cs.iter().any(|&b| b != 0));
    }
}
