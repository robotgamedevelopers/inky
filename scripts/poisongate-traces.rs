// Bind `src/poolgate.rs` (`poisoned`, `plans`, `poison_plan`, `apply`, `restore`,
// `table_slots`, `poison_bytes`) + `src/blocks.rs` (`BlockPools`, `BlockTable`) to
// `specs/PoisonGate.tla` via `specs/PoisonGateTrace.tla` (`U64, U179`).
//
// Exercises `Take`, `Share`, and `Release` both between steps (`phase = "idle"`)
// and inside the poisoned phase (`ensure_span_blocks` `Take` during `"poisoned"`,
// `Write` during `"poisoned"`, and `finish_prefill_span` / `prune_expired_window`
// `Share` + `Release` during `"finish"` before `Restore`).
#[path = "blocks.rs"]
#[allow(dead_code)]
mod blocks;
#[path = "poolgate.rs"]
#[allow(dead_code)]
mod poolgate;

use blocks::{BlockEntry, BlockIndex, BlockKind, BlockPools, BlockTable, Slot, BLOCK};
use poolgate::{poison_bytes, poisoned, table_slots, LaneId, PoisonPool, StepPath, Workload, KINDS, POISON};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

const LANES: usize = 3;
const CAPACITY: usize = 10;
const MAX_WRITE: i32 = 3;
const SLOT_BYTES: usize = 16;
const SIM_WIN_BLOCKS: usize = 2;

#[derive(Clone)]
struct StepRec {
    action: &'static str,
    arg_lane: i32,
    arg_slot: usize,
    arg_set: Vec<usize>,
    table: [BTreeSet<usize>; LANES],
    data: [i32; CAPACITY],
    truth: [i32; CAPACITY],
    saved: [i32; CAPACITY],
    stepping: BTreeSet<usize>,
    phase: &'static str,
}

#[derive(Default, Clone, Copy)]
struct Coverage {
    poison_step: usize,
    restore: usize,
    write: usize,
    take_idle: usize,
    share_idle: usize,
    release_idle: usize,
    take_poisoned: usize,
    share_poisoned: usize,
    release_poisoned: usize,
    saved_slots: usize,
}

struct LaneEntry {
    table: BlockTable,
    tokens: Vec<u32>,
    written: usize,
}

struct Inner {
    pools: BlockPools,
    mem: [[Vec<u8>; CAPACITY]; 2],
    truth: [[i32; CAPACITY]; 2],
    saved: [[i32; CAPACITY]; 2],
    stepping: BTreeSet<usize>,
    phase: [&'static str; 2],
    lanes: Vec<Option<LaneEntry>>,
    canon: BTreeMap<(usize, Vec<u32>), (BlockEntry, BlockEntry)>,
    last_rows: usize,
    traces: [Vec<StepRec>; 2],
    cov: Coverage,
    pre_take_on_new: bool,
}

fn kind_idx(k: BlockKind) -> usize {
    match k {
        BlockKind::Window => 0,
        BlockKind::Full => 1,
    }
}

fn decode_slot_bytes(buf: &[u8]) -> i32 {
    let w0 = u32::from_ne_bytes(buf[0..4].try_into().unwrap());
    let w1 = u32::from_ne_bytes(buf[4..8].try_into().unwrap());
    if w0 == POISON[0] && w1 == POISON[1] {
        return -1;
    }
    let val = i32::try_from(w0).unwrap_or(-1);
    if (0..=MAX_WRITE).contains(&val) && w1 == 0 && buf[8..].iter().all(|&b| b == 0) {
        val
    } else {
        -1
    }
}

fn write_slot_int(buf: &mut [u8], val: i32) {
    buf.fill(0);
    let w = u32::try_from(val).expect("non-negative truth value");
    buf[0..4].copy_from_slice(&w.to_ne_bytes());
}

impl Inner {
    fn new(pre_take_on_new: bool) -> Self {
        let pools = BlockPools::new(CAPACITY);
        let mut mem = [
            std::array::from_fn(|_| vec![0u8; SLOT_BYTES]),
            std::array::from_fn(|_| vec![0u8; SLOT_BYTES]),
        ];
        for ki in 0..2 {
            for s in 0..CAPACITY {
                poison_bytes(&mut mem[ki][s]);
            }
        }
        let mut inner = Inner {
            pools,
            mem,
            truth: [[0; CAPACITY]; 2],
            saved: [[-2; CAPACITY]; 2],
            stepping: BTreeSet::new(),
            phase: ["idle", "idle"],
            lanes: Vec::new(),
            canon: BTreeMap::new(),
            last_rows: 0,
            traces: [Vec::new(), Vec::new()],
            cov: Coverage::default(),
            pre_take_on_new,
        };
        for kind in KINDS {
            inner.push_step(kind, "Init", -1, 0, Vec::new());
        }
        inner
    }

    fn lane_table_set(&self, lane: usize, kind: BlockKind) -> BTreeSet<usize> {
        let mut out = BTreeSet::new();
        if let Some(Some(le)) = self.lanes.get(lane) {
            for (k, s) in table_slots(&le.table) {
                if k == kind {
                    out.insert(s.get() as usize);
                }
            }
        }
        out
    }

    fn snapshot_data(&self, kind: BlockKind) -> [i32; CAPACITY] {
        let ki = kind_idx(kind);
        std::array::from_fn(|i| decode_slot_bytes(&self.mem[ki][i]))
    }

    fn push_step(
        &mut self,
        kind: BlockKind,
        action: &'static str,
        arg_lane: i32,
        arg_slot: usize,
        arg_set: Vec<usize>,
    ) {
        let ki = kind_idx(kind);
        let table = std::array::from_fn(|l| self.lane_table_set(l, kind));
        let data = self.snapshot_data(kind);
        let rec = StepRec {
            action,
            arg_lane,
            arg_slot,
            arg_set,
            table,
            data,
            truth: self.truth[ki],
            saved: self.saved[ki],
            stepping: self.stepping.clone(),
            phase: self.phase[ki],
        };
        self.traces[ki].push(rec);
    }

    fn take_slot_for_lane(&mut self, lane: usize, kind: BlockKind, bi: BlockIndex) {
        let ki = kind_idx(kind);
        let slot = self.pools.of(kind).lowest_free().expect("free slot in pool");
        let _ = self.pools.of_mut(kind).take(slot).expect("pool.take");
        let gen = self.pools.of(kind).record(slot).unwrap().generation();
        self.lanes[lane]
            .as_mut()
            .unwrap()
            .table
            .fill(self.pools.of_mut(kind), bi, slot, gen)
            .expect("table.fill");
        let idx = slot.pool_index().get() as usize;
        write_slot_int(&mut self.mem[ki][idx], 0);
        self.truth[ki][idx] = 0;
        if self.phase[ki] == "idle" {
            self.cov.take_idle += 1;
        } else {
            self.cov.take_poisoned += 1;
        }
        self.push_step(kind, "Take", lane as i32, slot.get() as usize, Vec::new());
    }

    fn share_slot_for_lane(&mut self, lane: usize, kind: BlockKind, bi: BlockIndex, e: BlockEntry) {
        let ki = kind_idx(kind);
        self.lanes[lane]
            .as_mut()
            .unwrap()
            .table
            .fill(self.pools.of_mut(kind), bi, e.slot(), e.generation())
            .expect("table.fill share");
        if self.phase[ki] == "idle" {
            self.cov.share_idle += 1;
        } else {
            self.phase[ki] = "finish";
            self.cov.share_poisoned += 1;
        }
        self.push_step(kind, "Share", lane as i32, e.slot().get() as usize, Vec::new());
    }

    fn release_slot_for_lane(&mut self, lane: usize, kind: BlockKind, bi: BlockIndex) {
        let ki = kind_idx(kind);
        let Some(old) = self.lanes[lane]
            .as_ref()
            .unwrap()
            .table
            .get(kind, bi)
            .unwrap()
        else {
            return;
        };
        let still_elsewhere = {
            let tbl = &self.lanes[lane].as_ref().unwrap().table;
            tbl.entries()
                .any(|(b, e)| !(e.kind() == kind && b == bi) && e.kind() == kind && e.slot() == old.slot())
        };
        self.lanes[lane]
            .as_mut()
            .unwrap()
            .table
            .clear_kind_entry(self.pools.of_mut(kind), bi)
            .expect("clear_kind_entry");
        if still_elsewhere {
            return;
        }
        if self.phase[ki] == "idle" {
            self.cov.release_idle += 1;
        } else {
            self.phase[ki] = "finish";
            self.cov.release_poisoned += 1;
        }
        self.push_step(kind, "Release", lane as i32, old.slot().get() as usize, Vec::new());
    }

    fn live_entry(&self, kind: BlockKind, e: BlockEntry) -> bool {
        self.pools
            .of(kind)
            .record(e.slot())
            .ok()
            .is_some_and(|r| r.generation() == e.generation() && r.sealed() && r.refs() > 0)
    }
}

struct TraceCandidate {
    inner: RefCell<Inner>,
}

impl TraceCandidate {
    fn new(pre_take_on_new: bool) -> Self {
        TraceCandidate { inner: RefCell::new(Inner::new(pre_take_on_new)) }
    }
}

impl StepPath for TraceCandidate {
    fn new_lane(&self, toks: &[u32], cap: usize) -> LaneId {
        let mut g = self.inner.borrow_mut();
        assert_eq!(g.phase, ["idle", "idle"]);
        let n_blocks = u32::try_from(cap.div_ceil(BLOCK).max(4)).unwrap();
        let lid = if let Some(reuse) = g.lanes.iter().position(|x| x.is_none()) {
            g.lanes[reuse] = Some(LaneEntry {
                table: BlockTable::new(n_blocks).unwrap(),
                tokens: toks.to_vec(),
                written: 0,
            });
            reuse
        } else {
            assert!(g.lanes.len() < LANES, "exceeded model Lanes={LANES}");
            g.lanes.push(Some(LaneEntry {
                table: BlockTable::new(n_blocks).unwrap(),
                tokens: toks.to_vec(),
                written: 0,
            }));
            g.lanes.len() - 1
        };

        // Exercise inter-step Share (start-point prefix match) or inter-step Take in `phase = "idle"`:
        if toks.len() >= BLOCK {
            let blk0 = toks[..BLOCK].to_vec();
            if let Some(&(cf, cw)) = g.canon.get(&(0, blk0)) {
                if g.live_entry(BlockKind::Full, cf) && g.live_entry(BlockKind::Window, cw) {
                    let bi = BlockIndex::new(0);
                    g.share_slot_for_lane(lid, BlockKind::Window, bi, cw);
                    g.share_slot_for_lane(lid, BlockKind::Full, bi, cf);
                    g.lanes[lid].as_mut().unwrap().written = BLOCK;
                    return lid;
                }
            }
        }
        if g.pre_take_on_new && lid == 0 {
            let bi = BlockIndex::new(0);
            if g.pools.window.lowest_free().is_some() && g.pools.full.lowest_free().is_some() {
                g.take_slot_for_lane(lid, BlockKind::Window, bi);
                g.take_slot_for_lane(lid, BlockKind::Full, bi);
            }
        }
        lid
    }

    fn set_token(&self, lane: LaneId, pos: usize, tok: u32) {
        let mut g = self.inner.borrow_mut();
        let le = g.lanes[lane].as_mut().expect("live lane");
        if pos >= le.tokens.len() {
            le.tokens.resize(pos + 1, 0);
        }
        le.tokens[pos] = tok;
    }

    fn prefill(&self, spans: &[(LaneId, usize, usize)]) {
        let mut g = self.inner.borrow_mut();
        // 1. Pre-GPU `ensure_span_blocks`: Take zeroed slots inside `phase = "poisoned"`!
        for &(l, p0, t) in spans {
            let written = g.lanes[l].as_ref().unwrap().written;
            let start = p0.max(written);
            if start >= p0 + t {
                continue;
            }
            for b in (start / BLOCK)..=((p0 + t - 1) / BLOCK) {
                let bi = BlockIndex::new(b as u32);
                for kind in KINDS {
                    if g.lanes[l].as_ref().unwrap().table.get(kind, bi).unwrap().is_none() {
                        g.take_slot_for_lane(l, kind, bi);
                    }
                }
            }
        }

        // 2. GPU execution: Write into stepping lanes' unsealed slots during `phase = "poisoned"`!
        for &(l, p0, t) in spans {
            let written = g.lanes[l].as_ref().unwrap().written;
            let start = p0.max(written);
            if start >= p0 + t {
                continue;
            }
            for b in (start / BLOCK)..=((p0 + t - 1) / BLOCK) {
                let bi = BlockIndex::new(b as u32);
                for kind in KINDS {
                    let ki = kind_idx(kind);
                    let e = g.lanes[l].as_ref().unwrap().table.get(kind, bi).unwrap().unwrap();
                    let sealed = g.pools.of(kind).record(e.slot()).unwrap().sealed();
                    let idx = e.slot().pool_index().get() as usize;
                    if !sealed && g.truth[ki][idx] < MAX_WRITE {
                        let next_v = g.truth[ki][idx] + 1;
                        write_slot_int(&mut g.mem[ki][idx], next_v);
                        g.truth[ki][idx] = next_v;
                        g.cov.write += 1;
                        g.push_step(kind, "Write", l as i32, e.slot().get() as usize, Vec::new());
                    }
                }
            }
        }

        // 3. Post-GPU `finish_prefill_span`: seal full blocks, Share canonical slots + Release replaced
        //    private slots, and prune expired window slots (`phase = "finish"` before `Restore`)!
        for &(l, p0, t) in spans {
            let end = p0 + t;
            let prev = g.lanes[l].as_ref().unwrap().written;
            if end <= prev {
                continue;
            }
            g.lanes[l].as_mut().unwrap().written = end;
            for b in (prev / BLOCK)..=((end - 1) / BLOCK) {
                let bi = BlockIndex::new(b as u32);
                let blen = (end - b * BLOCK).min(BLOCK) as u32;
                let fe = g.lanes[l].as_ref().unwrap().table.get(BlockKind::Full, bi).unwrap().unwrap();
                let we = g.lanes[l].as_ref().unwrap().table.get(BlockKind::Window, bi).unwrap().unwrap();
                for (kind, e) in [(BlockKind::Full, fe), (BlockKind::Window, we)] {
                    let rec = g.pools.of_mut(kind).record_mut(e.slot()).unwrap();
                    if !rec.sealed() {
                        rec.set_length(blen).unwrap();
                        if blen == BLOCK as u32 {
                            rec.seal();
                        }
                    }
                }
                if blen == BLOCK as u32 {
                    let key = (b, g.lanes[l].as_ref().unwrap().tokens[b * BLOCK..(b + 1) * BLOCK].to_vec());
                    if let Some(&(cf, cw)) = g.canon.get(&key) {
                        if (cf != fe || cw != we) && g.live_entry(BlockKind::Full, cf) && g.live_entry(BlockKind::Window, cw) {
                            if cw != we {
                                g.release_slot_for_lane(l, BlockKind::Window, bi);
                                g.share_slot_for_lane(l, BlockKind::Window, bi, cw);
                            }
                            if cf != fe {
                                g.release_slot_for_lane(l, BlockKind::Full, bi);
                                g.share_slot_for_lane(l, BlockKind::Full, bi, cf);
                            }
                        } else {
                            g.canon.insert(key, (fe, we));
                        }
                    } else {
                        g.canon.insert(key, (fe, we));
                    }
                }
            }
            let cur_blk = end / BLOCK;
            if cur_blk > SIM_WIN_BLOCKS {
                for old_b in 0..(cur_blk - SIM_WIN_BLOCKS) {
                    g.release_slot_for_lane(l, BlockKind::Window, BlockIndex::new(old_b as u32));
                }
            }
        }
        g.last_rows = spans.len();
    }

    fn decode(&self, spans: &[(LaneId, usize)]) {
        let mut g = self.inner.borrow_mut();
        for &(l, pos) in spans {
            let bi = BlockIndex::new((pos / BLOCK) as u32);
            for kind in KINDS {
                if g.lanes[l].as_ref().unwrap().table.get(kind, bi).unwrap().is_none() {
                    g.take_slot_for_lane(l, kind, bi);
                }
            }
        }
        for &(l, pos) in spans {
            let bi = BlockIndex::new((pos / BLOCK) as u32);
            for kind in KINDS {
                let ki = kind_idx(kind);
                let e = g.lanes[l].as_ref().unwrap().table.get(kind, bi).unwrap().unwrap();
                let idx = e.slot().pool_index().get() as usize;
                let sealed = g.pools.of(kind).record(e.slot()).unwrap().sealed();
                if !sealed && g.truth[ki][idx] < MAX_WRITE {
                    let next_v = g.truth[ki][idx] + 1;
                    write_slot_int(&mut g.mem[ki][idx], next_v);
                    g.truth[ki][idx] = next_v;
                    g.cov.write += 1;
                    g.push_step(kind, "Write", l as i32, e.slot().get() as usize, Vec::new());
                }
            }
        }
        for &(l, pos) in spans {
            let end = pos + 1;
            g.lanes[l].as_mut().unwrap().written = end;
            let b = pos / BLOCK;
            let bi = BlockIndex::new(b as u32);
            let blen = (end - b * BLOCK).min(BLOCK) as u32;
            for kind in KINDS {
                let e = g.lanes[l].as_ref().unwrap().table.get(kind, bi).unwrap().unwrap();
                let rec = g.pools.of_mut(kind).record_mut(e.slot()).unwrap();
                if !rec.sealed() {
                    rec.set_length(blen).unwrap();
                }
            }
            let cur_blk = end / BLOCK;
            if cur_blk > SIM_WIN_BLOCKS {
                for old_b in 0..(cur_blk - SIM_WIN_BLOCKS) {
                    g.release_slot_for_lane(l, BlockKind::Window, BlockIndex::new(old_b as u32));
                }
            }
        }
        g.last_rows = spans.len();
    }

    fn logits(&self, row: usize) -> Vec<f32> {
        let g = self.inner.borrow();
        assert!(row < g.last_rows);
        vec![0.0, 1.0, 0.5, -1.0]
    }

    fn end_lane(&self, lane: LaneId) {
        let mut g = self.inner.borrow_mut();
        let entries: Vec<(BlockIndex, BlockKind)> = g.lanes[lane]
            .as_ref()
            .unwrap()
            .table
            .entries()
            .map(|(b, e)| (b, e.kind()))
            .collect();
        for (bi, kind) in entries {
            g.release_slot_for_lane(lane, kind, bi);
        }
        g.lanes[lane] = None;
    }
}

impl PoisonPool for TraceCandidate {
    fn slot_count(&self, _kind: BlockKind) -> usize {
        CAPACITY
    }

    fn lane_slots(&self, lane: LaneId) -> Vec<(BlockKind, Slot)> {
        let g = self.inner.borrow();
        table_slots(&g.lanes[lane].as_ref().expect("live lane").table)
    }

    fn slot_bytes(&self, kind: BlockKind, slot: Slot) -> Option<(*mut u8, usize)> {
        let mut g = self.inner.borrow_mut();
        let ki = kind_idx(kind);
        let buf = &mut g.mem[ki][slot.pool_index().get() as usize];
        Some((buf.as_mut_ptr(), buf.len()))
    }

    fn note_applied(&self, stepping_lanes: &[LaneId], saved: &[(BlockKind, Slot, Vec<u8>)]) {
        let mut g = self.inner.borrow_mut();
        let set: BTreeSet<usize> = stepping_lanes.iter().copied().collect();
        let set_vec: Vec<usize> = set.iter().copied().collect();
        g.stepping = set;
        g.saved = [[-2; CAPACITY]; 2];
        for (kind, slot, bytes) in saved {
            let ki = kind_idx(*kind);
            g.saved[ki][slot.pool_index().get() as usize] = decode_slot_bytes(bytes);
            g.cov.saved_slots += 1;
        }
        for kind in KINDS {
            let ki = kind_idx(kind);
            g.phase[ki] = "poisoned";
            g.cov.poison_step += 1;
            g.push_step(kind, "PoisonStep", -1, 0, set_vec.clone());
        }
    }

    fn note_restored(&self) {
        let mut g = self.inner.borrow_mut();
        g.stepping.clear();
        g.saved = [[-2; CAPACITY]; 2];
        for kind in KINDS {
            let ki = kind_idx(kind);
            g.phase[ki] = "idle";
            g.cov.restore += 1;
            g.push_step(kind, "Restore", -1, 0, Vec::new());
        }
    }
}

fn format_int_set(set: &BTreeSet<usize>) -> String {
    let items: Vec<String> = set.iter().map(usize::to_string).collect();
    format!("{{{}}}", items.join(", "))
}

fn format_int_seq(seq: &[i32]) -> String {
    let items: Vec<String> = seq.iter().map(i32::to_string).collect();
    format!("<<{}>>", items.join(", "))
}

fn render_step(s: &StepRec) -> String {
    let tables: Vec<String> = s.table.iter().map(format_int_set).collect();
    let arg_set: BTreeSet<usize> = s.arg_set.iter().copied().collect();
    format!(
        "[action |-> \"{}\", argLane |-> {}, argSlot |-> {}, argSet |-> {}, table |-> <<{}>>, data |-> {}, truth |-> {}, saved |-> {}, stepping |-> {}, phase |-> \"{}\"]",
        s.action,
        s.arg_lane,
        s.arg_slot,
        format_int_set(&arg_set),
        tables.join(", "),
        format_int_seq(&s.data),
        format_int_seq(&s.truth),
        format_int_seq(&s.saved),
        format_int_set(&s.stepping),
        s.phase
    )
}

fn add_cov(dst: &mut Coverage, src: Coverage) {
    dst.poison_step += src.poison_step;
    dst.restore += src.restore;
    dst.write += src.write;
    dst.take_idle += src.take_idle;
    dst.share_idle += src.share_idle;
    dst.release_idle += src.release_idle;
    dst.take_poisoned += src.take_poisoned;
    dst.share_poisoned += src.share_poisoned;
    dst.release_poisoned += src.release_poisoned;
    dst.saved_slots += src.saved_slots;
}

fn main() {
    let base: Vec<u32> = (0..512).map(|i| (i * 7 + 3) as u32).collect();
    let mut all_traces: Vec<Vec<StepRec>> = Vec::new();
    let mut cov = Coverage::default();

    // Run 1: 3 concurrent lanes, staggered by round (`round >= i % 3`), odd lane 1 sharing
    // even lane 0's first 64-token block inside `finish_prefill_span` (while lane 0 is saved!),
    // plus multi-block prefill + decode crossing `SIM_WIN_BLOCKS` so `Release` fires inside
    // the poisoned phase AND `end_lane` releases slots in `idle`, followed by a second wave
    // on the same candidate so `new_lane` exercises `Share` in `idle`!
    {
        let cand = TraceCandidate::new(true);
        let w1 = Workload::from_tokens(&base, 2, vec![64, 64, 64], 2);
        let _ = poisoned(&cand, &w1, false);
        let w2 = Workload::from_tokens(&base, 3, vec![64, 32], 1);
        let _ = poisoned(&cand, &w2, false);
        let g = cand.inner.into_inner();
        add_cov(&mut cov, g.cov);
        for t in g.traces {
            assert!(t.len() >= 2);
            all_traces.push(t);
        }
    }

    // Run 2: partial-block prefill chunks (`32 + 32 + 64`) exercising multiple `Write`s to the
    // same unsealed slot across poisoned steps before sealing & sharing.
    {
        let cand = TraceCandidate::new(false);
        let w = Workload::from_tokens(&base[37..], 3, vec![32, 32, 64], 2);
        let _ = poisoned(&cand, &w, false);
        let g = cand.inner.into_inner();
        add_cov(&mut cov, g.cov);
        for t in g.traces {
            assert!(t.len() >= 2);
            all_traces.push(t);
        }
    }

    println!("---- MODULE PoisonGateTraceData ----");
    println!("EXTENDS Integers");
    println!("Traces == <<");
    let rendered_traces: Vec<String> = all_traces
        .iter()
        .map(|tr| {
            let steps: Vec<String> = tr.iter().map(|s| format!("  {}", render_step(s))).collect();
            format!("<<\n{}\n>>", steps.join(",\n"))
        })
        .collect();
    println!("{}\n>>\n====", rendered_traces.join(",\n"));

    let checks = [
        ("poison_step", cov.poison_step),
        ("restore", cov.restore),
        ("write", cov.write),
        ("take_idle", cov.take_idle),
        ("share_idle", cov.share_idle),
        ("release_idle", cov.release_idle),
        ("take_poisoned", cov.take_poisoned),
        ("share_poisoned", cov.share_poisoned),
        ("release_poisoned", cov.release_poisoned),
        ("saved_slots", cov.saved_slots),
    ];
    let summary: Vec<String> = checks.iter().map(|(k, v)| format!("{k}={v}")).collect();
    eprintln!("poisongate-traces: {} traces; {}", all_traces.len(), summary.join(", "));
    for (name, count) in checks {
        if count == 0 {
            eprintln!("COVERAGE GAP: poisongate-traces never reached {name}");
        }
    }
}
