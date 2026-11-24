//! Layer 3's output gate: the block cache gives exactly layer 2's outputs while every pool slot the step does not
//! read holds poison.
//!
//! The reference is layer 2's path, each lane alone, nothing poisoned. The candidate runs the same lanes together in
//! steps, as serve does (one step of prefill spans, then one decode step), and before each step every pool slot that no
//! lane in that step references is overwritten with poison: NaN and fp16-max words. Slots that a live lane outside the
//! step owns are saved first and put back after the step, so the poison reaches everything the step should not read
//! and destroys nothing a later step needs. Every step's logits and every greedy token must be bit-identical to the
//! reference.
//!
//! What plugs in where:
//!  - `StepPath` is a way to run lanes: create, step prefill spans, step decode spans, read logits, end. `Layer2` is
//!    today's per-lane caches; the block cache implements it as the candidate.
//!  - `PoisonPool` is the pool as the gate sees it: how many slots, which slots a lane's block table references, and
//!    each slot's bytes. The block cache implements it over its pool of GPU buffers. `Layer2` implements it over each
//!    lane's own buffers (three slots per lane), so the gate runs today and checks lane isolation with the other lanes'
//!    caches poisoned.
//!  - `run` takes the reference and the candidate; `gate` in main.rs today passes `Layer2` as both.
//!
//! The slot arithmetic (`poison_plan`) is modelled in specs/PoisonGate.tla.
use crate::engine::{Engine, Lane, Span};
use objc2_metal::MTLCommandBuffer;
use std::cell::RefCell;

/// A lane's handle, chosen by the path that made it.
pub type LaneId = usize;

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

/// The pool, as the poison gate sees it. Slots are numbered 0..slot_count(); a pool whose own slot numbers start at 1
/// maps slot s to s - 1. Nothing is in flight on the GPU while the gate calls these.
pub trait PoisonPool {
    fn slot_count(&self) -> usize;
    /// Every slot the lane's block table references now: sealed shared blocks, window blocks, its unsealed tail.
    fn lane_slots(&self, lane: LaneId) -> Vec<usize>;
    /// The slot's bytes, CPU-visible. None for a slot that has no memory now (a layer-2 lane that ended).
    fn slot_bytes(&self, slot: usize) -> Option<(*mut u8, usize)>;
}

/// The candidate is a path with a pool.
pub trait Candidate: StepPath + PoisonPool {}
impl<T: StepPath + PoisonPool> Candidate for T {}

/// What to do to the pool before one step.
#[derive(Debug, PartialEq, Eq)]
pub struct Plan {
    /// slots to overwrite with poison, ascending
    pub poison: Vec<usize>,
    /// the poisoned slots a live lane owns: saved before the poison, put back after the step, ascending
    pub save: Vec<usize>,
}

/// Which slots to poison and which to save, for one step. `tables[i]` is live lane i's slots; `stepping` lists the
/// live lanes in the step. A slot is protected when any stepping lane references it (shared slots included); every
/// other slot is poisoned, free or not; a poisoned slot some live lane references is saved. `guard = false` is the
/// reversed witness: it drops the protection of the first stepping lane, so that lane reads poison and the gate must
/// fail (specs/PoisonGate.tla, Bug = "unguarded").
pub fn poison_plan(n_slots: usize, tables: &[Vec<usize>], stepping: &[usize], guard: bool) -> Plan {
    let mut protect = vec![false; n_slots];
    let mut owned = vec![false; n_slots];
    for (i, t) in tables.iter().enumerate() {
        let steps = stepping.contains(&i) && (guard || Some(&i) != stepping.first());
        for &s in t {
            assert!(s < n_slots, "lane {i} references slot {s} of a pool of {n_slots}");
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

/// Apply a plan: save, then poison. Returns the saved bytes, for `restore`.
fn apply(pool: &dyn PoisonPool, plan: &Plan) -> Vec<(usize, Vec<u8>)> {
    let mut saved = Vec::with_capacity(plan.save.len());
    for &s in &plan.save {
        if let Some((p, n)) = pool.slot_bytes(s) { saved.push((s, unsafe { std::slice::from_raw_parts(p, n) }.to_vec())); }
    }
    for &s in &plan.poison {
        if let Some((p, n)) = pool.slot_bytes(s) { poison_bytes(unsafe { std::slice::from_raw_parts_mut(p, n) }); }
    }
    saved
}

fn restore(pool: &dyn PoisonPool, saved: Vec<(usize, Vec<u8>)>) {
    for (s, v) in saved {
        let (p, n) = pool.slot_bytes(s).expect("a saved slot lost its memory during the step");
        assert_eq!(n, v.len(), "slot {s} changed size during the step");
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

/// What the poisoned run saw.
pub struct Report { pub records: Vec<Record>, pub steps: usize, pub poisoned: usize, pub saved: usize, pub sabotaged: bool }

/// The lanes together through `cand`, with the pool poisoned before every step. Lane i starts i % 3 rounds late; a
/// round is a prefill step of every prefilling lane, then a decode step of every decoding lane. `sabotage` drops the
/// guard (`poison_plan`) at the first step where a lane reads cache it wrote earlier, so the gate must fail.
pub fn poisoned(cand: &dyn Candidate, w: &Workload, sabotage: bool) -> Report {
    let work = w.work();
    let n = w.prompts.len();
    let mut lane: Vec<Option<LaneId>> = vec![None; n];
    let mut next = vec![0usize; n];
    let mut rec = vec![Record::default(); n];
    let (mut steps, mut poisoned, mut saved_n, mut sabotaged) = (0, 0, 0, false);
    let mut round = 0;
    while next.iter().any(|&k| k < work.len()) {
        for i in 0..n { if lane[i].is_none() && next[i] == 0 && round >= i % 3 { lane[i] = Some(cand.new_lane(&w.prompts[i], w.len() + 1)); } }
        for dec in [false, true] {
            let who: Vec<usize> = (0..n).filter(|&i| lane[i].is_some() && next[i] < work.len() && work[next[i]].2 == dec).collect();
            if who.is_empty() { continue; }
            let live: Vec<usize> = (0..n).filter(|&i| lane[i].is_some()).collect();
            let tables: Vec<Vec<usize>> = live.iter().map(|&i| cand.lane_slots(lane[i].unwrap())).collect();
            let stepping: Vec<usize> = who.iter().map(|i| live.iter().position(|j| j == i).unwrap()).collect();
            let unguard = sabotage && !sabotaged && work[next[who[0]]].0 > 0;
            if unguard { sabotaged = true; println!("INK_POOLGATE_SABOTAGE: lane {}'s slots poisoned in its own step at position {}", who[0], work[next[who[0]]].0); }
            let plan = poison_plan(cand.slot_count(), &tables, &stepping, !unguard);
            poisoned += plan.poison.len(); saved_n += plan.save.len();
            let saved = apply(cand, &plan);
            if dec { cand.decode(&who.iter().map(|&i| (lane[i].unwrap(), work[next[i]].0)).collect::<Vec<_>>()); }
            else { cand.prefill(&who.iter().map(|&i| { let (p0, t, _) = work[next[i]]; (lane[i].unwrap(), p0, t) }).collect::<Vec<_>>()); }
            restore(cand, saved);
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
    Report { records: rec, steps, poisoned, saved: saved_n, sabotaged }
}

/// The reference against the poisoned candidate: Ok(lines) when every lane matches at every step, Err(lines) naming
/// the first difference of each lane that differs.
pub fn compare(want: &[Record], got: &Report) -> Result<Vec<String>, Vec<String>> {
    let mut bad = Vec::new();
    for (i, (a, b)) in want.iter().zip(&got.records).enumerate() {
        if let Some(s) = a.steps.iter().zip(&b.steps).position(|(x, y)| x != y) { bad.push(format!("lane {i}: logits differ from step {s}")); }
        else if a.steps.len() != b.steps.len() { bad.push(format!("lane {i}: {} steps, reference {}", b.steps.len(), a.steps.len())); }
        else if a.tokens != b.tokens { bad.push(format!("lane {i}: tokens differ")); }
    }
    if bad.is_empty() { Ok(vec![format!("{} lanes, {} steps, {} slot poisonings ({} saved and restored): every step's logits and every token identical", want.len(), got.steps, got.poisoned, got.saved)]) } else { Err(bad) }
}

/// Layer 2's path: each lane owns its own cache buffers and convolution rows. As a pool, lane k owns slots 3k (window
/// cache), 3k + 1 (full cache) and 3k + 2 (convolution rows); an ended lane's slots have no memory.
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
    fn slot_count(&self) -> usize { 3 * self.lanes.borrow().len() }
    fn lane_slots(&self, lane: LaneId) -> Vec<usize> { vec![3 * lane, 3 * lane + 1, 3 * lane + 2] }
    fn slot_bytes(&self, slot: usize) -> Option<(*mut u8, usize)> {
        self.run(|v| v[slot / 3].as_ref().map(|l| self.eng.lane_regions(l)[slot % 3]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    struct Heap { slots: RefCell<Vec<Vec<u8>>>, tables: Vec<Vec<usize>> }
    impl PoisonPool for Heap {
        fn slot_count(&self) -> usize { self.slots.borrow().len() }
        fn lane_slots(&self, l: LaneId) -> Vec<usize> { self.tables[l].clone() }
        fn slot_bytes(&self, s: usize) -> Option<(*mut u8, usize)> { let mut v = self.slots.borrow_mut(); Some((v[s].as_mut_ptr(), v[s].len())) }
    }

    #[test]
    fn apply_poisons_and_restore_puts_back_only_owned_slots() {
        let h = Heap { slots: RefCell::new((0..4u8).map(|i| vec![i; 8]).collect()), tables: vec![vec![0], vec![1, 2]] };
        let plan = poison_plan(4, &h.tables, &[0], true);
        let saved = apply(&h, &plan);
        assert_eq!(h.slots.borrow()[0], vec![0; 8]);
        for s in 1..4 { assert!(f32::from_bits(u32::from_ne_bytes(h.slots.borrow()[s][0..4].try_into().unwrap())).is_nan()); }
        restore(&h, saved);
        assert_eq!(h.slots.borrow()[1], vec![1; 8]);
        assert_eq!(h.slots.borrow()[2], vec![2; 8]);
        // slot 3 is free: it stays poisoned, and the pool zeroes it when it hands it out
        assert!(f32::from_bits(u32::from_ne_bytes(h.slots.borrow()[3][0..4].try_into().unwrap())).is_nan());
    }
}
