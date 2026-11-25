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
//!  - `StepPath`: create a lane, step prefill spans, step decode spans, read logits, end a lane. `Layer2` is today's
//!    per-lane caches; the block cache implements it as the candidate, over `Engine::block_pool` and
//!    `Lane::block_table`.
//!  - `PoisonPool`: each kind's pool as the gate sees it. `slot_count(kind)` is `BlockPools::{window,full}.len()`;
//!    `lane_slots` is `table_slots(&lane.block_table().borrow())`; `slot_bytes(kind, slot)` is
//!    `slot_region(base, len, pool, slot)` over that kind's GPU buffer, at the pool's width. `Layer2` implements it with lane k's window
//!    cache as window slot k + 1 and its full cache as full slot k + 1, so the gate runs today with the other lanes'
//!    caches poisoned.
//!  - main.rs `poolgate` builds the reference and the candidate; the candidate line is where the block cache goes.
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
}

/// The candidate is a path with a pool.
pub trait Candidate: StepPath + PoisonPool {}
impl<T: StepPath + PoisonPool> Candidate for T {}

/// Every (kind, slot) a block table holds an entry for. Walks each kind's rows until the table says out of range.
#[allow(dead_code)] // for the block cache's PoisonPool; Layer2 has no table
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

/// The bytes of `slot` in `pool`'s buffer at `base` of `len` bytes, at the pool's element width.
/// None when the slot does not fit in the buffer.
#[allow(dead_code)] // for the block cache's PoisonPool; Layer2 has no pool buffer
pub fn slot_region(base: *mut u8, len: usize, pool: &crate::blocks::BlockPool, slot: Slot) -> Option<(*mut u8, usize)> {
    let stride = pool.kind().stride_bytes(pool.kvb()).ok()?;
    let off = pool.offset(slot).ok()?;
    let end = off.checked_add(stride)?;
    if end > len as u64 { return None; }
    Some((unsafe { base.add(off as usize) }, stride as usize))
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
    for (i, w) in b.chunks_exact_mut(4).enumerate() { w.copy_from_slice(&POISON[i & 1].to_ne_bytes()); }
    let tail = b.len() & !3;
    for x in &mut b[tail..] { *x = 0x7f; }
}

fn slot_at(i: usize) -> Slot { Slot::new(u32::try_from(i + 1).expect("pool index fits a slot")).unwrap() }

/// Both kinds' plans for one step. `tables[i]` is live lane i's (kind, slot) list.
pub fn plans(pool: &dyn PoisonPool, tables: &[Vec<(BlockKind, Slot)>], stepping: &[usize], guard: bool) -> Vec<(BlockKind, Plan)> {
    KINDS.iter().map(|&kind| {
        let per: Vec<Vec<usize>> = tables.iter().map(|t| t.iter().filter(|(k, _)| *k == kind).map(|(_, s)| s.pool_index().get() as usize).collect()).collect();
        (kind, poison_plan(pool.slot_count(kind), &per, stepping, guard))
    }).collect()
}

/// What `apply` did: slots saved, slots poisoned, and planned slots skipped because they legitimately have no memory.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Applied { pub saved: usize, pub poisoned: usize, pub absent: usize }

/// Apply the plans: save, then poison. Returns the saved bytes, for `restore`, and what was actually done. A planned
/// slot with no memory is counted as absent when the pool says it may be (`absent_ok`) and panics, naming the slot,
/// otherwise; a slot is never skipped without being counted. A slot to save is a live lane's, so it is never absent.
fn apply(pool: &dyn PoisonPool, plans: &[(BlockKind, Plan)]) -> (Vec<(BlockKind, Slot, Vec<u8>)>, Applied) {
    let mut saved = Vec::new();
    let mut n = Applied::default();
    for (kind, plan) in plans {
        for &i in &plan.save {
            let (p, len) = pool.slot_bytes(*kind, slot_at(i)).unwrap_or_else(|| panic!("poolgate: {kind:?} slot {} (pool index {i}) of {} is referenced by a live lane and has no memory", i + 1, pool.slot_count(*kind)));
            saved.push((*kind, slot_at(i), unsafe { std::slice::from_raw_parts(p, len) }.to_vec()));
            n.saved += 1;
        }
        for &i in &plan.poison {
            match pool.slot_bytes(*kind, slot_at(i)) {
                Some((p, len)) => { poison_bytes(unsafe { std::slice::from_raw_parts_mut(p, len) }); n.poisoned += 1; }
                None if pool.absent_ok(*kind, slot_at(i)) => n.absent += 1,
                None => panic!("poolgate: {kind:?} slot {} (pool index {i}) of {} has no memory and is not marked absent (an ended lane)", i + 1, pool.slot_count(*kind)),
            }
        }
    }
    (saved, n)
}

fn restore(pool: &dyn PoisonPool, saved: Vec<(BlockKind, Slot, Vec<u8>)>) {
    for (kind, s, v) in saved {
        let (p, n) = pool.slot_bytes(kind, s).expect("a saved slot lost its memory during the step");
        assert_eq!(n, v.len(), "{kind:?} slot {} changed size during the step", s.get());
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), p, n); }
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
pub struct Report { pub records: Vec<Record>, pub steps: usize, pub poisoned: usize, pub saved: usize, pub absent: usize, pub sabotaged: Option<usize> }

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
    let mut round = 0;
    while next.iter().any(|&k| k < work.len()) {
        for i in 0..n { if lane[i].is_none() && next[i] == 0 && round >= i % 3 { lane[i] = Some(cand.new_lane(&w.prompts[i], w.len() + 1)); } }
        for dec in [false, true] {
            let who: Vec<usize> = (0..n).filter(|&i| lane[i].is_some() && next[i] < work.len() && work[next[i]].2 == dec).collect();
            if who.is_empty() { continue; }
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
            let (saved, done) = apply(cand, &ps);
            let stepping_ids: Vec<LaneId> = who.iter().map(|&i| lane[i].unwrap()).collect();
            cand.note_applied(&stepping_ids, &saved);
            poisoned += done.poisoned; saved_n += done.saved; absent += done.absent;
            if dec { cand.decode(&who.iter().map(|&i| (lane[i].unwrap(), work[next[i]].0)).collect::<Vec<_>>()); }
            else { cand.prefill(&who.iter().map(|&i| { let (p0, t, _) = work[next[i]]; (lane[i].unwrap(), p0, t) }).collect::<Vec<_>>()); }
            restore(cand, saved);
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
    Report { records: rec, steps, poisoned, saved: saved_n, absent, sabotaged }
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
    let mut bad = Vec::new();
    for (i, (a, b)) in want.iter().zip(&got.records).enumerate() {
        if let Some(s) = a.steps.iter().zip(&b.steps).position(|(x, y)| x != y) { bad.push(format!("lane {i}: logits differ from step {s}")); }
        else if a.steps.len() != b.steps.len() { bad.push(format!("lane {i}: {} steps, reference {}", b.steps.len(), a.steps.len())); }
        else if a.tokens != b.tokens { bad.push(format!("lane {i}: tokens differ")); }
    }
    if bad.is_empty() { Ok(vec![format!("{} lanes, {} steps, {} slot poisonings ({} saved and restored, {} planned slots of ended lanes absent): every step's logits and every token identical", want.len(), got.steps, got.poisoned, got.saved, got.absent)]) } else { Err(bad) }
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
    #[allow(dead_code)]
    pub fn candidate_label(&self) -> &'static str {
        "Layer2 self-check (BlockPools candidate not yet wired)"
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

    #[allow(dead_code)]
    pub fn candidate_label(&self) -> &'static str {
        "Layer3 BlockPool + BlockTable + PrefixTree"
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
        if self.keep_persistent_starts && end % BLOCK == 0 && l3.sealed.len() == end / BLOCK {
            let tip = *l3.sealed.last().unwrap();
            let mut reasons = HEAD;
            if end == l3.prompt_len { reasons |= PROMPT; }
            if (end / BLOCK) % 16 == 0 { reasons |= EVERY; }
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
        cb.waitUntilCompleted();
        (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3
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
    if total_positions % BLOCK != 0 && st.tails_dropped == 0 {
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
    for chunk in cs.chunks_exact_mut(4) {
        chunk.copy_from_slice(&42.0f32.to_ne_bytes());
    }
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
pub use gpu_paths::{verify_l3_workload_stats, verify_prefix_start_points, Layer2, Layer3Pool};

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
    fn slot_region_is_the_stride_at_the_offset_and_refuses_past_the_buffer() {
        use crate::blocks::BlockPool;
        let base = 0x1000 as *mut u8;
        let len = 2 * crate::blocks::FULL_STRIDE as usize;
        let full = BlockPool::new(BlockKind::Full, 4);
        let narrow = BlockPool::with_kvb(BlockKind::Full, 4, 1).unwrap();
        let window = BlockPool::new(BlockKind::Window, 4);
        assert_eq!(slot_region(base, len, &full, sl(1)), Some((base, crate::blocks::FULL_STRIDE as usize)));
        assert_eq!(slot_region(base, len, &full, sl(2)).map(|r| r.0 as usize - 0x1000), Some(crate::blocks::FULL_STRIDE as usize));
        assert_eq!(slot_region(base, len, &full, sl(3)), None);
        assert_eq!(slot_region(base, len, &window, sl(1)), None);
        let half = crate::blocks::FULL_STRIDE as usize / 2;
        assert_eq!(slot_region(base, 4 * half, &narrow, sl(2)).map(|r| r.1), Some(half));
        assert!(half < crate::blocks::FULL_STRIDE as usize);
    }

    #[test]
    fn leaked_names_every_other_lane_that_differs() {
        let r = |s: u64| Record { steps: vec![1, s], tokens: vec![7] };
        let want = vec![r(2), r(3), r(4)];
        let got = |v: Vec<Record>| Report { records: v, steps: 0, poisoned: 0, saved: 0, absent: 0, sabotaged: Some(0) };
        assert_eq!(leaked(&want, &got(vec![r(9), r(3), r(4)]), 0), Vec::<usize>::new());
        assert_eq!(leaked(&want, &got(vec![r(9), r(9), r(4)]), 0), vec![1]);
        assert_eq!(leaked(&want, &got(vec![r(2), r(3), Record { steps: vec![1, 4], tokens: vec![8] }]), 1), vec![2]);
    }

    #[test]
    fn compare_rejects_short_or_extra_candidate_record_lists() {
        let r0 = Record { steps: vec![1, 2], tokens: vec![10] };
        let r1 = Record { steps: vec![3, 4], tokens: vec![20] };
        let want = vec![r0.clone(), r1.clone()];
        let short = Report { records: vec![r0.clone()], steps: 2, poisoned: 1, saved: 0, absent: 0, sabotaged: None };
        let err = compare(&want, &short).unwrap_err();
        assert!(err[0].contains("lane count mismatch"), "short candidate must fail compare: {err:?}");
        let extra = Report { records: vec![r0.clone(), r1.clone(), r0.clone()], steps: 2, poisoned: 1, saved: 0, absent: 0, sabotaged: None };
        assert!(compare(&want, &extra).is_err(), "extra candidate lanes must fail compare");
        let exact = Report { records: vec![r0, r1], steps: 2, poisoned: 1, saved: 0, absent: 0, sabotaged: None };
        assert!(compare(&want, &exact).is_ok());
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
