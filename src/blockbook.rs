// Layer 3 bookkeeping for start points and unsealed tails (docs/DESIGN.md, Blocks).
//
// Which blocks a start point holds, when start points are kept and dropped, what a
// lane may read of its tail, and when a tail is dropped. Slot records, tables and
// offsets are the types in blocks.rs; the GPU side (zeroing a slot, keeping a copy
// of the recurrent state) goes through `GpuSide`, so this file runs without Metal.
//
// Bound to specs/StartTails.tla: scripts/check-blockbook-traces.sh compiles this
// file with blocks.rs, replays deterministic random walks of it through TLC, and
// requires each named mutation of this file to be rejected. Plain `//` comments:
// that script includes the file by path.
//
// The rules, with the model's names:
// - A start point at block boundary n holds a copy of the recurrent state at n,
//   every full block before n and the window blocks covering the `window`
//   positions before n (`Retained`). Each is a reference on its slot, so pruning
//   the lane's window does not free a slot a live start point holds (`StartRetained`).
// - A step that ends a prefill on a block boundary keeps a start point there. Its
//   reasons are the lane head, the prompt's last boundary, and every `every`-th
//   block. A lane has one head: the previous head loses that reason, and a point
//   left with no reason is dropped. Finishing a lane drops its head reason, so the
//   prompt-end and every-`every` points outlive the request.
// - Tail blocks are allocated by one lane, are never in a start point, and are
//   zeroed by the GPU side on every allocation, reuse included (`TailPrivateZero`).
//   Attention reads a block only up to the lane's position (`ReadsWritten`).
//   Finishing a lane drops its references; a tail has no other, so it is free.

use crate::blocks::{BlockEntry, BlockError, BlockIndex, BlockKind, BlockPools, BlockTable, Slot};
use std::fmt;

/// Kinds in the order the model and the trace list them.
pub const KINDS: [BlockKind; 2] = [BlockKind::Full, BlockKind::Window];

/// A start point is kept at the prompt's last block boundary.
pub const PROMPT: u8 = 1;
/// A start point is kept at every `every`-th block boundary of a prefill.
pub const EVERY: u8 = 2;
/// A start point is a running lane's head.
pub const HEAD: u8 = 4;

/// Handle to a kept copy of a lane's recurrent state.
pub type StateId = u64;

/// Block geometry. Production is 64 positions, a window of 1024, a start point
/// every 16 blocks; the tests and the trace check use small values.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Geometry {
    pub block: u32,
    pub window: u32,
    pub every: u32,
    /// Table rows per kind: the context in blocks.
    pub blocks: u32,
}

impl Geometry {
    pub fn production(context: u32) -> Geometry {
        Geometry { block: crate::blocks::BLOCK as u32, window: crate::blocks::WINDOW as u32, every: 16,
                   blocks: context.div_ceil(crate::blocks::BLOCK as u32) }
    }
    pub fn max_pos(&self) -> u32 { self.blocks * self.block }

    /// Blocks holding positions `lo..=hi`, empty when `hi < lo`.
    fn span(&self, lo: u32, hi: u32) -> std::ops::Range<u32> {
        if hi < lo { 0..0 } else { lo / self.block..hi / self.block + 1 }
    }
    /// Window blocks covering the `window` positions before `n`.
    pub fn history(&self, n: u32) -> std::ops::Range<u32> {
        if n == 0 { return 0..0; }
        self.span(n.saturating_sub(self.window), n - 1)
    }
    /// Full blocks before `n`.
    pub fn full(&self, n: u32) -> std::ops::Range<u32> {
        if n == 0 { 0..0 } else { self.span(0, n - 1) }
    }
    /// The blocks a start point at `n` holds.
    pub fn retained(&self, n: u32) -> impl Iterator<Item = (BlockKind, u32)> {
        self.full(n).map(|b| (BlockKind::Full, b)).chain(self.history(n).map(|b| (BlockKind::Window, b)))
    }
    /// Positions of block `b` written by a lane at `n`.
    pub fn seen(&self, n: u32, b: u32) -> u32 {
        n.saturating_sub(b * self.block).min(self.block)
    }
}

/// The GPU side of the bookkeeping. Johnny's pool owns the buffers; this is only
/// what start points and tails need from it.
pub trait GpuSide {
    /// Fill every position of `slot` with zeros. Called on every allocation.
    fn zero(&mut self, kind: BlockKind, slot: Slot);
    /// Keep a copy of `lane`'s recurrent state as it is at the end of this step.
    fn save_state(&mut self, lane: usize) -> StateId;
    /// Release a kept recurrent state.
    fn free_state(&mut self, state: StateId);
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BookError {
    Block(BlockError),
    NoLane,
    LaneBusy,
    LaneIdle,
    /// The prompt is empty or longer than the context.
    Prompt,
    /// No live start point with that id.
    NoStart,
    /// A start point at or past the prompt's end leaves nothing to prefill.
    StartPastPrompt,
    /// A pool has no free slot. Nothing was taken.
    PoolFull,
    /// The block is behind the lane, or already allocated.
    NotAhead,
    /// A span of zero tokens, past the context, or over blocks not allocated
    /// or not written up to the lane's position.
    Span,
    /// A start point would hold a missing or unsealed block.
    Unsealed,
}

impl From<BlockError> for BookError {
    fn from(e: BlockError) -> Self { BookError::Block(e) }
}

impl fmt::Display for BookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BookError::Block(e) => write!(f, "{e}"),
            BookError::NoLane => write!(f, "no such lane"),
            BookError::LaneBusy => write!(f, "lane is running a request"),
            BookError::LaneIdle => write!(f, "lane is not running a request"),
            BookError::Prompt => write!(f, "prompt is empty or longer than the context"),
            BookError::NoStart => write!(f, "no such start point"),
            BookError::StartPastPrompt => write!(f, "start point is not before the prompt's end"),
            BookError::PoolFull => write!(f, "block pool is full"),
            BookError::NotAhead => write!(f, "block is behind the lane or already allocated"),
            BookError::Span => write!(f, "span is empty, past the context or not allocated"),
            BookError::Unsealed => write!(f, "start point would hold an unsealed block"),
        }
    }
}

/// A kept start point.
#[derive(Clone, Debug)]
pub struct StartPoint {
    pos: u32,
    state: StateId,
    reasons: u8,
    lane: Option<usize>,
    table: BlockTable,
}

impl StartPoint {
    pub fn pos(&self) -> u32 { self.pos }
    pub fn state(&self) -> StateId { self.state }
    pub fn reasons(&self) -> u8 { self.reasons }
    /// The lane whose head this is, while it is one.
    pub fn lane(&self) -> Option<usize> { self.lane }
    pub fn table(&self) -> &BlockTable { &self.table }
}

#[derive(Clone, Debug)]
struct Lane {
    running: bool,
    prompt: u32,
    pos: u32,
    head: Option<usize>,
    table: BlockTable,
}

/// Start points and lanes over both block pools.
pub struct BlockBook {
    geo: Geometry,
    pools: BlockPools,
    lanes: Vec<Lane>,
    starts: Vec<Option<StartPoint>>,
}

impl BlockBook {
    /// `slots` per kind, `lanes` lanes, at most `max_starts` live start points.
    pub fn new(geo: Geometry, slots: usize, lanes: usize, max_starts: usize) -> Result<BlockBook, BookError> {
        if geo.block == 0 || geo.block as usize > crate::blocks::BLOCK || geo.window == 0 || geo.every == 0 {
            return Err(BookError::Block(BlockError::Length));
        }
        geo.blocks.checked_mul(geo.block).ok_or(BookError::Block(BlockError::Overflow))?;
        let table = BlockTable::new(geo.blocks)?;
        let lane = Lane { running: false, prompt: 0, pos: 0, head: None, table };
        Ok(BlockBook { geo, pools: BlockPools::new(slots), lanes: vec![lane; lanes], starts: vec![None; max_starts] })
    }

    pub fn geometry(&self) -> Geometry { self.geo }
    pub fn pools(&self) -> &BlockPools { &self.pools }
    pub fn lane_count(&self) -> usize { self.lanes.len() }
    pub fn running(&self, lane: usize) -> bool { self.lanes.get(lane).is_some_and(|l| l.running) }
    pub fn pos(&self, lane: usize) -> u32 { self.lanes.get(lane).map_or(0, |l| l.pos) }
    pub fn prompt(&self, lane: usize) -> u32 { self.lanes.get(lane).map_or(0, |l| l.prompt) }
    pub fn head(&self, lane: usize) -> Option<usize> { self.lanes.get(lane).and_then(|l| l.head) }
    pub fn table(&self, lane: usize) -> Option<&BlockTable> { self.lanes.get(lane).map(|l| &l.table) }
    pub fn start(&self, id: usize) -> Option<&StartPoint> { self.starts.get(id).and_then(|s| s.as_ref()) }
    pub fn start_ids(&self) -> usize { self.starts.len() }

    fn lane(&self, lane: usize) -> Result<&Lane, BookError> { self.lanes.get(lane).ok_or(BookError::NoLane) }

    fn hold(&mut self, e: BlockEntry) -> Result<(), BookError> {
        self.pools.of_mut(e.kind()).record_mut(e.slot())?.add_ref()?;
        Ok(())
    }

    fn release(&mut self, e: BlockEntry) -> Result<(), BookError> {
        self.pools.of_mut(e.kind()).record_mut(e.slot())?.drop_ref()?;
        Ok(())
    }

    fn release_table(&mut self, table: &BlockTable) -> Result<(), BookError> {
        for (_, e) in table.entries() {
            self.release(e)?;
        }
        Ok(())
    }

    /// Start a request of `prompt` positions on `lane`, cold or from start point `from`.
    pub fn begin(&mut self, lane: usize, prompt: u32, from: Option<usize>) -> Result<(), BookError> {
        if self.lane(lane)?.running { return Err(BookError::LaneBusy); }
        if prompt == 0 || prompt > self.geo.max_pos() { return Err(BookError::Prompt); }
        let (table, pos) = match from {
            None => (BlockTable::new(self.geo.blocks)?, 0),
            Some(id) => {
                let s = self.start(id).ok_or(BookError::NoStart)?;
                if s.pos >= prompt { return Err(BookError::StartPastPrompt); }
                (s.table.clone(), s.pos)
            }
        };
        for (_, e) in table.entries() {
            self.hold(e)?;
        }
        self.lanes[lane] = Lane { running: true, prompt, pos, head: None, table };
        Ok(())
    }

    /// Allocate tail block `block` of both kinds for `lane`, each in the lowest free
    /// slot of its pool, zeroed. Neither is taken unless both can be.
    pub fn alloc(&mut self, lane: usize, block: u32, gpu: &mut impl GpuSide) -> Result<(), BookError> {
        let l = self.lane(lane)?;
        if !l.running { return Err(BookError::LaneIdle); }
        if block < l.pos / self.geo.block || block >= self.geo.blocks { return Err(BookError::NotAhead); }
        let b = BlockIndex::new(block);
        for kind in KINDS {
            if l.table.get(kind, b)?.is_some() { return Err(BookError::NotAhead); }
        }
        let mut taken = [None; 2];
        for (i, kind) in KINDS.into_iter().enumerate() {
            taken[i] = Some(self.pools.of(kind).lowest_free().ok_or(BookError::PoolFull)?);
        }
        for (i, kind) in KINDS.into_iter().enumerate() {
            let slot = taken[i].ok_or(BookError::PoolFull)?;
            let pool = self.pools.of_mut(kind);
            let kvb = pool.kvb();
            let offset = pool.offset(slot)?;
            let rec = pool.record_mut(slot)?;
            rec.reissue()?;
            rec.add_ref()?;
            let entry = BlockEntry::new(slot, rec.generation(), offset, kind, kvb);
            gpu.zero(kind, slot);
            self.lanes[lane].table.set(kind, b, Some(entry))?;
        }
        Ok(())
    }

    /// Allocate every block `lane`'s next `c` positions need.
    pub fn reserve(&mut self, lane: usize, c: u32, gpu: &mut impl GpuSide) -> Result<(), BookError> {
        let l = self.lane(lane)?;
        let end = l.pos.checked_add(c).ok_or(BookError::Span)?;
        if c == 0 || end > self.geo.max_pos() { return Err(BookError::Span); }
        for block in self.geo.span(l.pos, end - 1) {
            if self.lanes[lane].table.get(BlockKind::Full, BlockIndex::new(block))?.is_none() {
                self.alloc(lane, block, gpu)?;
            }
        }
        Ok(())
    }

    /// The step wrote `c` positions of `lane` through its table. Seal what it
    /// filled, prune the window to its history, and keep a start point when the
    /// step ends a prefill on a block boundary. Returns the new start point.
    pub fn step(&mut self, lane: usize, c: u32, gpu: &mut impl GpuSide) -> Result<Option<usize>, BookError> {
        let geo = self.geo;
        let l = self.lane(lane)?;
        if !l.running { return Err(BookError::LaneIdle); }
        let pos = l.pos;
        let n = pos.checked_add(c).ok_or(BookError::Span)?;
        if c == 0 || n > geo.max_pos() { return Err(BookError::Span); }
        let mut written = Vec::new();
        for block in geo.span(pos, n - 1) {
            for kind in KINDS {
                let e = l.table.get(kind, BlockIndex::new(block))?.ok_or(BookError::Span)?;
                let rec = self.pools.of(kind).record(e.slot())?;
                if rec.length() != geo.seen(pos, block) { return Err(BookError::Span); }
                written.push((e, geo.seen(n, block)));
            }
        }
        for (e, len) in written {
            let rec = self.pools.of_mut(e.kind()).record_mut(e.slot())?;
            rec.set_length(len)?;
            if len == geo.block { rec.seal(); }
        }
        self.lanes[lane].pos = n;
        let history = geo.history(n);
        let pruned: Vec<(BlockIndex, BlockEntry)> = self.lanes[lane].table.entries()
            .filter(|(b, e)| e.kind() == BlockKind::Window && !history.contains(&b.get()) && b.get() * geo.block < n)
            .collect();
        for (b, e) in pruned { self.lanes[lane].table.set(BlockKind::Window, b, None)?; self.release(e)?; }
        self.keep_start(lane, gpu)
    }

    fn keep_start(&mut self, lane: usize, gpu: &mut impl GpuSide) -> Result<Option<usize>, BookError> {
        let geo = self.geo;
        let l = &self.lanes[lane];
        let n = l.pos;
        let last = l.prompt / geo.block * geo.block;
        if n % geo.block != 0 || n > last { return Ok(None); }
        if l.head.and_then(|h| self.start(h)).is_some_and(|h| h.pos >= n) { return Ok(None); }
        let head_dies = l.head.and_then(|h| self.start(h)).is_some_and(|h| h.reasons == HEAD);
        if !head_dies && self.starts.iter().all(|s| s.is_some()) { return Ok(None); }
        let mut table = BlockTable::new(geo.blocks)?;
        for (kind, block) in geo.retained(n) {
            let b = BlockIndex::new(block);
            let e = l.table.get(kind, b)?.ok_or(BookError::Unsealed)?;
            if !self.pools.of(kind).record(e.slot())?.sealed() { return Err(BookError::Unsealed); }
            table.set(kind, b, Some(e))?;
        }
        let mut reasons = HEAD;
        if n == last { reasons |= PROMPT; }
        if (n / geo.block) % geo.every == 0 { reasons |= EVERY; }
        if let Some(h) = self.lanes[lane].head.take() { self.drop_reason(h, gpu)?; }
        let id = self.starts.iter().position(|s| s.is_none()).ok_or(BookError::NoStart)?;
        for (_, e) in table.entries() { self.hold(e)?; }
        let state = gpu.save_state(lane);
        self.starts[id] = Some(StartPoint { pos: n, state, reasons, lane: Some(lane), table });
        self.lanes[lane].head = Some(id);
        Ok(Some(id))
    }

    /// Take the head reason off start point `id`; with no reason left, drop it.
    fn drop_reason(&mut self, id: usize, gpu: &mut impl GpuSide) -> Result<(), BookError> {
        let s = self.starts.get_mut(id).and_then(|s| s.as_mut()).ok_or(BookError::NoStart)?;
        s.reasons &= !HEAD;
        s.lane = None;
        if s.reasons == 0 { self.drop_start(id, gpu)?; }
        Ok(())
    }

    fn drop_start(&mut self, id: usize, gpu: &mut impl GpuSide) -> Result<(), BookError> {
        let s = self.starts.get_mut(id).and_then(|s| s.take()).ok_or(BookError::NoStart)?;
        self.release_table(&s.table)?;
        gpu.free_state(s.state);
        Ok(())
    }

    /// The request on `lane` ended. Its tails are dropped with its table.
    pub fn finish(&mut self, lane: usize, gpu: &mut impl GpuSide) -> Result<(), BookError> {
        if !self.lane(lane)?.running { return Err(BookError::LaneIdle); }
        let table = std::mem::replace(&mut self.lanes[lane].table, BlockTable::new(self.geo.blocks)?);
        self.release_table(&table)?;
        if let Some(h) = self.lanes[lane].head.take() { self.drop_reason(h, gpu)?; }
        let l = &mut self.lanes[lane];
        l.running = false;
        l.prompt = 0;
        l.pos = 0;
        Ok(())
    }

    /// Retire start point `id` (memory pressure; the scheduler chooses which).
    pub fn retire(&mut self, id: usize, gpu: &mut impl GpuSide) -> Result<(), BookError> {
        self.drop_start(id, gpu)?;
        for l in &mut self.lanes {
            if l.head == Some(id) { l.head = None; }
        }
        Ok(())
    }

    /// Positions of `(kind, block)` attention may read through `lane`'s table:
    /// none without an entry, otherwise only what the lane has written.
    pub fn visible(&self, lane: usize, kind: BlockKind, block: u32) -> Result<Option<u32>, BookError> {
        let l = self.lane(lane)?;
        if l.table.get(kind, BlockIndex::new(block))?.is_none() { return Ok(None); }
        let start = block * self.geo.block;
        Ok(Some(l.pos.saturating_sub(start).min(self.geo.block)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// The GPU side as data: which slots were zeroed, and live states by id.
    #[derive(Default)]
    struct Fake {
        zeroed: Vec<(BlockKind, u32)>,
        next: StateId,
        states: BTreeMap<StateId, usize>,
    }

    impl GpuSide for Fake {
        fn zero(&mut self, kind: BlockKind, slot: Slot) { self.zeroed.push((kind, slot.get())); }
        fn save_state(&mut self, lane: usize) -> StateId {
            self.next += 1;
            self.states.insert(self.next, lane);
            self.next
        }
        fn free_state(&mut self, state: StateId) {
            assert!(self.states.remove(&state).is_some(), "state {state} freed twice");
        }
    }

    fn small() -> Geometry { Geometry { block: 2, window: 2, every: 2, blocks: 4 } }

    fn rec(book: &BlockBook, kind: BlockKind, slot: u32) -> crate::blocks::SlotRecord {
        *book.pools().of(kind).record(Slot::new(slot).unwrap()).unwrap()
    }

    fn entry(book: &BlockBook, lane: usize, kind: BlockKind, block: u32) -> Option<BlockEntry> {
        book.table(lane).unwrap().get(kind, BlockIndex::new(block)).unwrap()
    }

    fn run(book: &mut BlockBook, gpu: &mut Fake, lane: usize, c: u32) -> Option<usize> {
        book.reserve(lane, c, gpu).unwrap();
        book.step(lane, c, gpu).unwrap()
    }

    #[test]
    fn a_pruned_window_block_stays_held_by_its_start_point() {
        let (mut book, mut gpu) = (BlockBook::new(small(), 2, 1, 2).unwrap(), Fake::default());
        // A prompt of 3: the point at 2 is its prompt point and outlives the lane's head.
        book.begin(0, 3, None).unwrap();
        let s = run(&mut book, &mut gpu, 0, 2).unwrap();
        assert_eq!(book.start(s).unwrap().reasons(), HEAD | PROMPT);
        let w0 = entry(&book, 0, BlockKind::Window, 0).unwrap();
        assert_eq!(run(&mut book, &mut gpu, 0, 2), None);
        // At 4 the lane's history is block 1 only: window block 0 left its table.
        assert_eq!(entry(&book, 0, BlockKind::Window, 0), None);
        assert_eq!(rec(&book, BlockKind::Window, w0.slot().get()).refs(), 1);
        // The start point still resolves window block 0 at its generation, and no
        // window slot is free for anyone.
        let kept = book.start(s).unwrap().table().get(BlockKind::Window, BlockIndex::new(0)).unwrap().unwrap();
        assert_eq!(kept, w0);
        assert_eq!(book.pools().window.lowest_free(), None);
        // Retiring the point frees it.
        book.retire(s, &mut gpu).unwrap();
        assert_eq!(book.pools().window.lowest_free(), Some(w0.slot()));
        assert_eq!(book.head(0), None);
        assert!(gpu.states.is_empty());
    }

    #[test]
    fn a_start_point_keeps_state_full_blocks_and_the_window_before_it() {
        let geo = Geometry { block: 2, window: 4, every: 8, blocks: 6 };
        let (mut book, mut gpu) = (BlockBook::new(geo, 8, 1, 2).unwrap(), Fake::default());
        book.begin(0, 12, None).unwrap();
        run(&mut book, &mut gpu, 0, 4);
        let s = run(&mut book, &mut gpu, 0, 2).unwrap();
        let p = book.start(s).unwrap();
        assert_eq!(p.pos(), 6);
        let kinds: Vec<_> = p.table().entries().map(|(b, e)| (e.kind(), b.get())).collect();
        assert_eq!(kinds, vec![(BlockKind::Full, 0), (BlockKind::Full, 1), (BlockKind::Window, 1),
                               (BlockKind::Full, 2), (BlockKind::Window, 2)]);
        assert!(p.table().entries().all(|(_, e)| rec(&book, e.kind(), e.slot().get()).sealed()));
        assert_eq!(gpu.states.get(&p.state()), Some(&0));
    }

    #[test]
    fn heads_move_and_prompt_and_every_points_outlive_the_request() {
        let geo = Geometry { block: 2, window: 2, every: 2, blocks: 6 };
        let (mut book, mut gpu) = (BlockBook::new(geo, 8, 1, 4).unwrap(), Fake::default());
        book.begin(0, 9, None).unwrap();
        let a = run(&mut book, &mut gpu, 0, 2).unwrap();
        assert_eq!(book.start(a).unwrap().reasons(), HEAD);
        let b = run(&mut book, &mut gpu, 0, 2).unwrap();
        // Head moved to 4 (block 2, an every point); the head-only point at 2 is gone.
        assert!(book.start(a).is_none() || book.start(a).unwrap().pos() == 4);
        assert_eq!(book.start(b).unwrap().reasons(), HEAD | EVERY);
        let c = run(&mut book, &mut gpu, 0, 2).unwrap();
        let d = run(&mut book, &mut gpu, 0, 2).unwrap();
        assert_eq!(book.start(d).unwrap().reasons(), HEAD | EVERY | PROMPT);
        assert_eq!(book.start(d).unwrap().pos(), 8);
        assert_eq!(c, d);
        // Decode past the prompt keeps no point.
        assert_eq!(run(&mut book, &mut gpu, 0, 1), None);
        book.finish(0, &mut gpu).unwrap();
        let live: Vec<_> = (0..book.start_ids()).filter_map(|i| book.start(i)).map(|p| (p.pos(), p.reasons(), p.lane())).collect();
        assert_eq!(live, vec![(4, EVERY, None), (8, EVERY | PROMPT, None)]);
        assert_eq!(gpu.states.len(), 2);
    }

    #[test]
    fn production_points_sit_on_every_sixteenth_block() {
        let geo = Geometry::production(4096);
        assert_eq!((geo.block, geo.window, geo.every, geo.blocks), (64, 1024, 16, 64));
        let (mut book, mut gpu) = (BlockBook::new(geo, 128, 1, 8).unwrap(), Fake::default());
        book.begin(0, 3000, None).unwrap();
        while book.pos(0) + 512 <= 2944 {
            run(&mut book, &mut gpu, 0, 512);
        }
        let mut live: Vec<_> = (0..book.start_ids()).filter_map(|i| book.start(i)).map(|p| (p.pos(), p.reasons())).collect();
        live.sort();
        assert_eq!(live, vec![(1024, EVERY), (2048, EVERY), (2560, HEAD)]);
        let p = (0..book.start_ids()).filter_map(|i| book.start(i)).find(|p| p.pos() == 2048).unwrap();
        // 32 full blocks and the 16 window blocks of positions 1024..2047.
        assert_eq!(p.table().entries().filter(|(_, e)| e.kind() == BlockKind::Full).count(), 32);
        let w: Vec<_> = p.table().entries().filter(|(_, e)| e.kind() == BlockKind::Window).map(|(b, _)| b.get()).collect();
        assert_eq!(w, (16..32).collect::<Vec<_>>());
        // The prompt's last boundary, 2944, is a prompt point.
        let last = run(&mut book, &mut gpu, 0, 384).unwrap();
        assert_eq!(book.start(last).unwrap().reasons(), HEAD | PROMPT);
    }

    #[test]
    fn a_tail_is_private_and_reads_stop_at_its_length() {
        let (mut book, mut gpu) = (BlockBook::new(small(), 4, 2, 2).unwrap(), Fake::default());
        book.begin(0, 8, None).unwrap();
        let s = run(&mut book, &mut gpu, 0, 2).unwrap();
        run(&mut book, &mut gpu, 0, 1);
        let tail = entry(&book, 0, BlockKind::Full, 1).unwrap();
        let r = rec(&book, BlockKind::Full, tail.slot().get());
        assert_eq!((r.length(), r.sealed(), r.refs()), (1, false, 1));
        assert_eq!(book.visible(0, BlockKind::Full, 1).unwrap(), Some(1));
        assert_eq!(book.visible(0, BlockKind::Full, 0).unwrap(), Some(2));
        assert_eq!(book.visible(0, BlockKind::Full, 2).unwrap(), None);
        // The start point holds no tail, and a lane resumed from it shares no tail.
        assert!(book.start(s).unwrap().table().entries().all(|(b, _)| b.get() == 0));
        book.begin(1, 4, Some(s)).unwrap();
        assert_eq!(entry(&book, 1, BlockKind::Full, 1), None);
        assert_eq!(book.pos(1), 2);
        // A block allocated ahead reads nothing until it is written.
        book.alloc(1, 1, &mut gpu).unwrap();
        assert_eq!(book.visible(1, BlockKind::Window, 1).unwrap(), Some(0));
        assert_ne!(entry(&book, 1, BlockKind::Full, 1).unwrap().slot(), tail.slot());
    }

    #[test]
    fn finishing_drops_tails_and_reuse_zeroes_the_slot() {
        let (mut book, mut gpu) = (BlockBook::new(small(), 2, 1, 1).unwrap(), Fake::default());
        book.begin(0, 1, None).unwrap();
        run(&mut book, &mut gpu, 0, 1);
        let tail = entry(&book, 0, BlockKind::Window, 0).unwrap();
        book.finish(0, &mut gpu).unwrap();
        assert_eq!(rec(&book, BlockKind::Window, tail.slot().get()).refs(), 0);
        assert_eq!(gpu.zeroed.len(), 2);
        book.begin(0, 4, None).unwrap();
        book.alloc(0, 0, &mut gpu).unwrap();
        // Same slot, next generation, nothing written, and zeroed again.
        let again = entry(&book, 0, BlockKind::Window, 0).unwrap();
        assert_eq!((again.slot(), again.generation()), (tail.slot(), tail.generation() + 1));
        let r = rec(&book, BlockKind::Window, again.slot().get());
        assert_eq!((r.length(), r.sealed()), (0, false));
        assert_eq!(gpu.zeroed, vec![(BlockKind::Full, 1), (BlockKind::Window, 1), (BlockKind::Full, 1), (BlockKind::Window, 1)]);
    }

    #[test]
    fn refusals_change_nothing() {
        let (mut book, mut gpu) = (BlockBook::new(small(), 1, 2, 1).unwrap(), Fake::default());
        assert_eq!(book.begin(2, 1, None), Err(BookError::NoLane));
        assert_eq!(book.begin(0, 0, None), Err(BookError::Prompt));
        assert_eq!(book.begin(0, 9, None), Err(BookError::Prompt));
        assert_eq!(book.begin(0, 4, Some(0)), Err(BookError::NoStart));
        assert_eq!(book.step(0, 1, &mut gpu), Err(BookError::LaneIdle));
        book.begin(0, 4, None).unwrap();
        assert_eq!(book.begin(0, 4, None), Err(BookError::LaneBusy));
        assert_eq!(book.step(0, 1, &mut gpu), Err(BookError::Span));
        assert_eq!(book.step(0, 0, &mut gpu), Err(BookError::Span));
        book.alloc(0, 0, &mut gpu).unwrap();
        assert_eq!(book.alloc(0, 0, &mut gpu), Err(BookError::NotAhead));
        // One slot per kind: the second block cannot be taken, and neither kind is.
        assert_eq!(book.alloc(0, 1, &mut gpu), Err(BookError::PoolFull));
        assert_eq!(entry(&book, 0, BlockKind::Full, 1), None);
        assert_eq!(book.step(0, 3, &mut gpu), Err(BookError::Span));
        assert_eq!(book.pos(0), 0);
        let s = book.step(0, 2, &mut gpu).unwrap().unwrap();
        assert_eq!(book.alloc(0, 0, &mut gpu), Err(BookError::NotAhead));
        // A start point at or past the prompt's end leaves nothing to prefill.
        assert_eq!(book.begin(1, 2, Some(s)), Err(BookError::StartPastPrompt));
        assert_eq!(gpu.states.len(), 1);
    }

    #[test]
    fn with_no_free_start_id_the_step_keeps_no_point_and_the_old_head() {
        let geo = Geometry { block: 2, window: 2, every: 4, blocks: 4 };
        let (mut book, mut gpu) = (BlockBook::new(geo, 8, 2, 1).unwrap(), Fake::default());
        book.begin(0, 8, None).unwrap();
        book.begin(1, 8, None).unwrap();
        let a = run(&mut book, &mut gpu, 0, 2).unwrap();
        assert_eq!(run(&mut book, &mut gpu, 1, 2), None);
        // Lane 0's head-only point is replaced in place: its id is reused.
        assert_eq!(run(&mut book, &mut gpu, 0, 2), Some(a));
        assert_eq!(book.start(a).unwrap().pos(), 4);
        assert_eq!(gpu.states.len(), 1);
        book.finish(0, &mut gpu).unwrap();
        assert!(book.start(a).is_none());
        assert!(gpu.states.is_empty());
    }
}
