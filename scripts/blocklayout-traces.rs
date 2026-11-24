// Real CPU-only block-allocator NDJSON trace emitter compiled directly against
// production `src/blocks.rs` (`rustc --edition=2021`, zero Metal/Cargo deps).
// Calls `blocks::byte_offset`, `SlotRecord`, `BlockEntry`, `BlockPools`, and
// `BlockTable` directly so mutations to `src/blocks.rs` alter the emitted trace.

#[path = "blocks.rs"]
#[allow(dead_code)]
mod blocks;

use blocks::{
    block_length, byte_offset, BlockEntry, BlockIndex, BlockKind, BlockPools, BlockTable, Slot,
    SlotRecord,
};

const LANES: usize = 2;
const LANE_NAMES: [&str; LANES] = ["a", "b"];
const FULL_CAPACITY: usize = 5;
const WINDOW_CAPACITY: usize = 5;
const B: usize = 2;
const W: usize = 2;
const MAX_POS: usize = 6;
const MAX_SPAN: usize = 2;
const MAX_GEN: u32 = 3;

fn kind_name(k: BlockKind) -> &'static str {
    match k {
        BlockKind::Full => "full",
        BlockKind::Window => "window",
    }
}

fn cap_of(k: BlockKind) -> usize {
    match k {
        BlockKind::Full => FULL_CAPACITY,
        BlockKind::Window => WINDOW_CAPACITY,
    }
}

fn span_bound_small(c: usize) -> usize {
    (W - 1 + c + B - 1) / B + 1
}

fn history_blocks(n: usize) -> Vec<u32> {
    let lo = n.saturating_sub(W);
    if n == 0 || n - 1 < lo {
        return Vec::new();
    }
    let mut out = Vec::new();
    for p in lo..n {
        let b = (p / B) as u32;
        if !out.contains(&b) {
            out.push(b);
        }
    }
    out
}

fn span_window_blocks(n: usize, c: usize) -> Vec<u32> {
    let lo = if n >= W { n - W + 1 } else { 0 };
    let hi = n + c;
    let mut out = Vec::new();
    for p in lo..hi {
        let b = (p / B) as u32;
        if !out.contains(&b) {
            out.push(b);
        }
    }
    out
}

fn full_blocks(n: usize) -> Vec<u32> {
    let mut out = Vec::new();
    for p in 0..n {
        let b = (p / B) as u32;
        if !out.contains(&b) {
            out.push(b);
        }
    }
    out
}

struct TraceAllocator {
    slot_base: u32,
    full_recs: Vec<SlotRecord>,
    win_recs: Vec<SlotRecord>,
    tables: Vec<Vec<Option<BlockEntry>>>,
    pos: [usize; LANES],
    span_start: [usize; LANES],
    target: [usize; LANES],
    pins: [Vec<(BlockKind, u32, u32)>; LANES],
    done: [bool; LANES],
    start_kept: bool,
    start_pos: usize,
    start_entries: Vec<(BlockKind, u32, BlockEntry)>,
}

impl TraceAllocator {
    fn new(slot_base: u32) -> Self {
        let n_blocks = ((MAX_POS - 1) / B) + 1;
        let _pools = BlockPools {
            window: blocks::BlockPool::new(BlockKind::Window, WINDOW_CAPACITY),
            full: blocks::BlockPool::new(BlockKind::Full, FULL_CAPACITY),
        };
        let _bt = BlockTable::new(n_blocks as u32).expect("valid BlockTable");
        Self {
            slot_base,
            full_recs: (0..FULL_CAPACITY).map(|_| SlotRecord::fresh(BlockKind::Full)).collect(),
            win_recs: (0..WINDOW_CAPACITY).map(|_| SlotRecord::fresh(BlockKind::Window)).collect(),
            tables: vec![vec![None; n_blocks * 2]; LANES],
            pos: [0; LANES],
            span_start: [0; LANES],
            target: [0; LANES],
            pins: [Vec::new(), Vec::new()],
            done: [false; LANES],
            start_kept: false,
            start_pos: 0,
            start_entries: Vec::new(),
        }
    }

    fn rec_mut(&mut self, kind: BlockKind, slot: u32) -> &mut SlotRecord {
        let i = (slot - 1) as usize;
        match kind {
            BlockKind::Full => &mut self.full_recs[i],
            BlockKind::Window => &mut self.win_recs[i],
        }
    }

    fn rec(&self, kind: BlockKind, slot: u32) -> &SlotRecord {
        let i = (slot - 1) as usize;
        match kind {
            BlockKind::Full => &self.full_recs[i],
            BlockKind::Window => &self.win_recs[i],
        }
    }

    fn row_idx(kind: BlockKind, block: u32) -> usize {
        blocks::table_index(kind, BlockIndex::new(block), ((MAX_POS - 1) / B + 1) as u32)
            .expect("valid table_index") as usize
    }

    fn is_owned(&self, kind: BlockKind, slot: u32) -> bool {
        for l in 0..LANES {
            if !self.done[l] {
                for e in self.tables[l].iter().flatten() {
                    if e.kind() == kind && e.slot().get() == slot {
                        return true;
                    }
                }
            }
            for &(pk, ps, _) in &self.pins[l] {
                if pk == kind && ps == slot {
                    return true;
                }
            }
        }
        if self.start_kept {
            for &(sk, _, se) in &self.start_entries {
                if sk == kind && se.slot().get() == slot {
                    return true;
                }
            }
        }
        false
    }

    fn prepare(&mut self, lane: usize, tokens: usize) {
        self.span_start[lane] = self.pos[lane];
        self.target[lane] = self.pos[lane] + tokens;
        let budget = span_bound_small(tokens);
        let ln = LANE_NAMES[lane];
        println!(
            "{{\"budget\":{budget},\"event\":\"prepare\",\"lane\":\"{ln}\",\"pos\":{},\"tokens\":{tokens}}}",
            self.pos[lane]
        );
    }

    fn alloc_needed(&mut self, lane: usize) {
        let c = self.target[lane] - self.span_start[lane];
        let mut needed: Vec<(BlockKind, u32)> = Vec::new();
        for b in full_blocks(self.span_start[lane] + c) {
            needed.push((BlockKind::Full, b));
        }
        for b in span_window_blocks(self.span_start[lane], c) {
            needed.push((BlockKind::Window, b));
        }
        let ln = LANE_NAMES[lane];
        for (kind, blk) in needed {
            let r_idx = Self::row_idx(kind, blk);
            if self.tables[lane][r_idx].is_some() {
                continue;
            }
            let cap = cap_of(kind) as u32;
            let chosen = (1..=cap)
                .find(|&s| !self.is_owned(kind, s) && self.rec(kind, s).generation() < MAX_GEN)
                .expect("free slot in pool");
            {
                let rec = self.rec_mut(kind, chosen);
                rec.bump_generation().expect("generation bump");
                rec.set_length(0).expect("zero length");
            }
            let gen = self.rec(kind, chosen).generation();
            let phys_slot = Slot::new(chosen + self.slot_base).expect("non-zero physical slot");
            let off = byte_offset(kind, phys_slot).expect("u64 byte_offset");
            let logical_slot = Slot::new(chosen).expect("non-zero logical slot");
            let entry = BlockEntry::new(logical_slot, gen, off, kind);
            self.tables[lane][r_idx] = Some(entry);
            let o = entry.offset();
            let hi = (o >> 32) as u32;
            let mib = ((o >> 20) & 0xfff) as u32;
            let sub = (o & 0xfffff) as u32;
            println!(
                "{{\"block\":{blk},\"event\":\"alloc\",\"generation\":{},\"kind\":\"{}\",\"lane\":\"{ln}\",\"offset\":{o},\"offset_limbs\":{{\"hi\":{hi},\"mib\":{mib},\"sub\":{sub}}},\"slot\":{}}}",
                entry.generation(),
                kind_name(entry.kind()),
                entry.slot().get(),
            );
        }
    }

    fn begin_step(&mut self, lane: usize) {
        let c = self.target[lane] - self.span_start[lane];
        let mut pins = Vec::new();
        for b in full_blocks(self.span_start[lane] + c) {
            let e = self.tables[lane][Self::row_idx(BlockKind::Full, b)].unwrap();
            pins.push((BlockKind::Full, e.slot().get(), e.generation()));
        }
        for b in span_window_blocks(self.span_start[lane], c) {
            let e = self.tables[lane][Self::row_idx(BlockKind::Window, b)].unwrap();
            if !pins.contains(&(BlockKind::Window, e.slot().get(), e.generation())) {
                pins.push((BlockKind::Window, e.slot().get(), e.generation()));
            }
        }
        let pin_count = pins.len();
        self.pins[lane] = pins;
        let ln = LANE_NAMES[lane];
        println!("{{\"event\":\"begin_step\",\"lane\":\"{ln}\",\"pin_count\":{pin_count}}}");
    }

    fn write_tokens(&mut self, lane: usize) {
        let ln = LANE_NAMES[lane];
        while self.pos[lane] < self.target[lane] {
            let blk = (self.pos[lane] / B) as u32;
            for kind in [BlockKind::Full, BlockKind::Window] {
                let s = self.tables[lane][Self::row_idx(kind, blk)].unwrap().slot().get();
                let new_len = block_length(self.rec(kind, s).length() + 1).expect("valid length");
                let rec = self.rec_mut(kind, s);
                rec.set_length(new_len).unwrap();
                if new_len as usize == B {
                    rec.seal();
                }
            }
            self.pos[lane] += 1;
            println!("{{\"event\":\"write\",\"lane\":\"{ln}\",\"pos\":{}}}", self.pos[lane]);
        }
    }

    fn end_step(&mut self, lane: usize) {
        self.pins[lane].clear();
        let ln = LANE_NAMES[lane];
        println!("{{\"event\":\"end_step\",\"lane\":\"{ln}\",\"pos\":{}}}", self.pos[lane]);
    }

    fn slide_window(&mut self, lane: usize) {
        let keep = history_blocks(self.pos[lane]);
        let n_blocks = ((MAX_POS - 1) / B + 1) as u32;
        let mut changed = false;
        for b in 0..n_blocks {
            if !keep.contains(&b) {
                let r_idx = Self::row_idx(BlockKind::Window, b);
                if self.tables[lane][r_idx].take().is_some() {
                    changed = true;
                }
            }
        }
        if changed {
            let ln = LANE_NAMES[lane];
            println!("{{\"event\":\"slide_window\",\"lane\":\"{ln}\"}}");
        }
    }

    fn keep_start(&mut self, lane: usize) {
        self.start_kept = true;
        self.start_pos = self.pos[lane];
        self.start_entries.clear();
        for b in full_blocks(self.start_pos) {
            let e = self.tables[lane][Self::row_idx(BlockKind::Full, b)].unwrap();
            self.start_entries.push((BlockKind::Full, b, e));
        }
        for b in history_blocks(self.start_pos) {
            let e = self.tables[lane][Self::row_idx(BlockKind::Window, b)].unwrap();
            self.start_entries.push((BlockKind::Window, b, e));
        }
        let ln = LANE_NAMES[lane];
        println!("{{\"event\":\"keep_start\",\"lane\":\"{ln}\",\"pos\":{}}}", self.start_pos);
    }

    fn drop_start(&mut self) {
        self.start_kept = false;
        self.start_entries.clear();
        println!("{{\"event\":\"drop_start\"}}");
    }

    fn match_start(&mut self, lane: usize) {
        for slot in &mut self.tables[lane] {
            *slot = None;
        }
        for &(kind, b, e) in &self.start_entries {
            self.tables[lane][Self::row_idx(kind, b)] = Some(e);
        }
        self.pos[lane] = self.start_pos;
        let ln = LANE_NAMES[lane];
        println!("{{\"event\":\"match\",\"lane\":\"{ln}\",\"pos\":{}}}", self.pos[lane]);
    }

    fn finish(&mut self, lane: usize) {
        self.done[lane] = true;
        for slot in &mut self.tables[lane] {
            *slot = None;
        }
        let ln = LANE_NAMES[lane];
        println!("{{\"event\":\"finish\",\"lane\":\"{ln}\"}}");
    }
}

fn main() {
    let slot_base: u32 = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("slot_base u32"))
        .unwrap_or(0);

    println!(
        "{{\"block_size\":{B},\"capacity\":{FULL_CAPACITY},\"event\":\"config\",\"full_capacity\":{FULL_CAPACITY},\"lanes\":[\"a\",\"b\"],\"max_generation\":{MAX_GEN},\"max_pos\":{MAX_POS},\"max_span\":{MAX_SPAN},\"model\":\"block_layout\",\"pool_mode\":\"per_kind\",\"slot_base\":{slot_base},\"version\":1,\"window_capacity\":{WINDOW_CAPACITY},\"window_size\":{W}}}"
    );

    let mut alloc = TraceAllocator::new(slot_base);
    // Lane "a" prefills 0..2, seals boundary 2 as a start point, and finishes
    alloc.prepare(0, 2);
    alloc.alloc_needed(0);
    alloc.begin_step(0);
    alloc.write_tokens(0);
    alloc.end_step(0);
    alloc.keep_start(0);
    alloc.finish(0);

    // Lane "b" matches start point 2, drops start point, advances 2..4 (allocating
    // full block 1 -> slot 2 and window block 1 -> slot 2 while slots 2..5 are free),
    // slides window (freeing window slot 1), then advances 4..6 reusing window slot 1
    // at generation 2!
    alloc.match_start(1);
    alloc.drop_start();
    alloc.prepare(1, 2);
    alloc.alloc_needed(1);
    alloc.begin_step(1);
    alloc.write_tokens(1);
    alloc.end_step(1);
    alloc.slide_window(1);

    alloc.prepare(1, 2);
    alloc.alloc_needed(1);
    alloc.begin_step(1);
    alloc.write_tokens(1);
    alloc.end_step(1);
    alloc.finish(1);
}
