// Layer 3/4 block allocator trace driver (over `src/blocks.rs` and `src/blockbook.rs`),
// dynamic-`kvb` working-set budgets, and the `CacheScheduler.tla` state machine.
// Plain `//` header comments so `scripts/check-cache-traces.sh` can include by path.
#![allow(dead_code)] // Layer 3/4 callers and standalone CI trace checkers consume this module; cargo check skips #[cfg(test)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::blockbook::Geometry;
use crate::blocks::{
    block_length, byte_offset, BlockEntry, BlockError, BlockIndex, BlockKind, BlockPools,
    BlockTable, Slot, BLOCK, WINDOW,
};
use crate::cache_trace::{
    emit_block_layout_config, emit_block_layout_event, emit_block_layout_step,
    emit_cache_scheduler_config, emit_cache_scheduler_event, emit_cache_scheduler_step, kind_name,
    BlockLayoutConfig, BlockLayoutEvent, BlockLayoutState, BlockLayoutStep, CacheBlockId,
    CacheBlockSnapshot, CacheLaneSnapshot, CacheSchedulerConfig, CacheSchedulerEvent,
    CacheSchedulerState, CacheSchedulerStep, JsonVal, OffsetLimbs, PinHandleSnapshot, StartPointSnapshot, TraceError,
    TraceSink,
};

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SchedulerError {
    Block(BlockError),
    Trace(TraceError),
    InvalidConfig(&'static str),
    Precondition(&'static str),
    InvariantViolation(String),
}

impl fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchedulerError::Block(e) => write!(f, "block error: {e}"),
            SchedulerError::Trace(e) => write!(f, "trace error: {e}"),
            SchedulerError::InvalidConfig(msg) => write!(f, "invalid scheduler config: {msg}"),
            SchedulerError::Precondition(msg) => write!(f, "scheduler precondition failed: {msg}"),
            SchedulerError::InvariantViolation(msg) => write!(f, "scheduler invariant violated: {msg}"),
        }
    }
}

impl From<BlockError> for SchedulerError {
    fn from(e: BlockError) -> Self {
        SchedulerError::Block(e)
    }
}

impl From<TraceError> for SchedulerError {
    fn from(e: TraceError) -> Self {
        SchedulerError::Trace(e)
    }
}

// ---- Checked Logical Block-Range & Worst-Case Budget Helpers --------------

/// Checked `ceil(n / d)`. Refuses `d == 0`.
pub fn ceil_div_u32(n: u32, d: u32) -> Result<u32, SchedulerError> {
    if d == 0 {
        return Err(SchedulerError::InvalidConfig("block divisor must be non-zero"));
    }
    Ok(n.div_ceil(d))
}

/// Conservative window-block pin bound `ceil((W - 1 + c) / B) + 1` matching
/// `SpanBound(c)` in `BlockLayout.tla`, `CacheScheduler.tla`, and `blocks::span_bound`.
/// Refuses zero `span_tokens`, `block_size`, or `window_size`, and checked-overflows cleanly.
pub fn span_bound_blocks(
    span_tokens: u32,
    block_size: u32,
    window_size: u32,
) -> Result<u32, SchedulerError> {
    if span_tokens == 0 || block_size == 0 || window_size == 0 {
        return Err(SchedulerError::InvalidConfig(
            "span_tokens, block_size, and window_size must be positive",
        ));
    }
    let numer = (window_size - 1)
        .checked_add(span_tokens)
        .ok_or(SchedulerError::Block(BlockError::Overflow))?;
    numer
        .div_ceil(block_size)
        .checked_add(1)
        .ok_or(SchedulerError::Block(BlockError::Overflow))
}

/// Logical block range `{p / B : p in lo..=hi}` (`{}` when `hi < lo`).
pub fn window_range(lo: u32, hi: u32, block_size: u32) -> Vec<u32> {
    if block_size == 0 || hi < lo {
        return Vec::new();
    }
    ((lo / block_size)..=(hi / block_size)).collect()
}

/// `History(n)` in `BlockLayout.tla` / `CacheScheduler.tla`, delegating to `blockbook::Geometry`.
pub fn history_blocks(pos: u32, block_size: u32, window_size: u32) -> Vec<u32> {
    let geo = Geometry { block: block_size, window: window_size, every: 1, blocks: u32::MAX / block_size.max(1) };
    geo.history(pos).collect()
}

/// `SpanWindow(n, c)` in `BlockLayout.tla` / `CacheScheduler.tla`.
pub fn span_window_blocks(pos: u32, span: u32, block_size: u32, window_size: u32) -> Vec<u32> {
    if span == 0 || block_size == 0 {
        return Vec::new();
    }
    let lo = if pos >= window_size { pos - window_size + 1 } else { 0 };
    let Some(end_incl) = pos.checked_add(span).and_then(|v| v.checked_sub(1)) else {
        return Vec::new();
    };
    window_range(lo, end_incl, block_size)
}

/// `Full(n)` in `BlockLayout.tla` / `CacheScheduler.tla`, delegating to `blockbook::Geometry`.
pub fn full_blocks(pos: u32, block_size: u32) -> Vec<u32> {
    let geo = Geometry { block: block_size, window: block_size, every: 1, blocks: u32::MAX / block_size.max(1) };
    geo.full(pos).collect()
}

/// Checked `(full_blocks, max_win_blocks, total_worst_case_blocks)` matching `WorstCaseBlocks(m)` in `CacheScheduler.tla`.
pub fn worst_case_blocks_checked(
    target_pos: u32,
    max_span: u32,
    block_size: u32,
    window_size: u32,
) -> Result<(u32, u32, u32), SchedulerError> {
    if target_pos == 0 || max_span == 0 || block_size == 0 || window_size == 0 {
        return Err(SchedulerError::InvalidConfig(
            "target_pos, max_span, block_size, and window_size must be positive",
        ));
    }
    let full = ceil_div_u32(target_pos, block_size)?;
    let bound = span_bound_blocks(max_span, block_size, window_size)?;
    let win = bound.min(full);
    let total = full
        .checked_add(win)
        .ok_or(SchedulerError::Block(BlockError::Overflow))?;
    Ok((full, win, total))
}

/// Cache memory budget derived from Metal's `recommendedMaxWorkingSetSize` at startup
/// (`DESIGN.md`: never passed by hand) and parameterized by element width `kvb` (`1` for `--kv8`, `2` for fp16).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WorkingSetBudget {
    pub recommended_working_set_bytes: u64,
    pub fixed_non_cache_bytes: u64,
    pub safety_headroom_bytes: u64,
    pub cache_budget_bytes: u64,
    pub kvb: u32,
    pub full_stride_bytes: u64,
    pub window_stride_bytes: u64,
}

impl WorkingSetBudget {
    pub fn from_recommended_working_set_bytes(
        recommended_working_set_bytes: u64,
        fixed_non_cache_bytes: u64,
        safety_headroom_bytes: u64,
        kvb: u32,
    ) -> Result<WorkingSetBudget, SchedulerError> {
        let full_stride_bytes = BlockKind::Full.stride_bytes(kvb)?;
        let window_stride_bytes = BlockKind::Window.stride_bytes(kvb)?;
        let reserved = fixed_non_cache_bytes
            .checked_add(safety_headroom_bytes)
            .ok_or(SchedulerError::InvalidConfig("working-set reservation overflow"))?;
        let cache_budget_bytes = recommended_working_set_bytes
            .checked_sub(reserved)
            .ok_or(SchedulerError::InvalidConfig(
                "recommended working set smaller than non-cache footprint",
            ))?;
        let min_pair = full_stride_bytes
            .checked_add(window_stride_bytes)
            .ok_or(SchedulerError::Block(BlockError::Overflow))?;
        if cache_budget_bytes < min_pair {
            return Err(SchedulerError::InvalidConfig(
                "remaining cache working-set budget cannot hold one full + one window block",
            ));
        }
        Ok(WorkingSetBudget {
            recommended_working_set_bytes,
            fixed_non_cache_bytes,
            safety_headroom_bytes,
            cache_budget_bytes,
            kvb,
            full_stride_bytes,
            window_stride_bytes,
        })
    }

    /// Checked worst-case GPU byte demand (`full * full_stride(kvb) + window * window_stride(kvb)`).
    /// Refuses zero `target_positions`, zero `max_span_tokens`, and any `u32`/`u64` overflow.
    pub fn worst_case_request_bytes(
        &self,
        target_positions: u32,
        max_span_tokens: u32,
    ) -> Result<u64, SchedulerError> {
        let (full, win, _) = worst_case_blocks_checked(
            target_positions,
            max_span_tokens,
            BLOCK as u32,
            WINDOW as u32,
        )?;
        let full_bytes = u64::from(full)
            .checked_mul(self.full_stride_bytes)
            .ok_or(SchedulerError::Block(BlockError::Overflow))?;
        let win_bytes = u64::from(win)
            .checked_mul(self.window_stride_bytes)
            .ok_or(SchedulerError::Block(BlockError::Overflow))?;
        full_bytes
            .checked_add(win_bytes)
            .ok_or(SchedulerError::Block(BlockError::Overflow))
    }

    /// Partition `cache_budget_bytes` into `(full_slots, window_slots)` at this pool's `kvb`.
    /// Refuses `desired_window_slots == 0` and budgets that leave zero full slots.
    pub fn partition_pools(&self, desired_window_slots: u32) -> Result<(u32, u32), SchedulerError> {
        if desired_window_slots == 0 {
            return Err(SchedulerError::InvalidConfig("desired_window_slots must be at least 1"));
        }
        let win_bytes = u64::from(desired_window_slots)
            .checked_mul(self.window_stride_bytes)
            .ok_or(SchedulerError::Block(BlockError::Overflow))?;
        if win_bytes >= self.cache_budget_bytes {
            return Err(SchedulerError::InvalidConfig("window pool exhausts entire cache budget"));
        }
        let rem = self.cache_budget_bytes - win_bytes;
        let full_slots64 = rem / self.full_stride_bytes;
        if full_slots64 == 0 {
            return Err(SchedulerError::InvalidConfig("zero full-block slots left after window pool"));
        }
        let full_slots = u32::try_from(full_slots64)
            .map_err(|_| SchedulerError::Block(BlockError::Overflow))?;
        Ok((full_slots, desired_window_slots))
    }
}

// ---- Part A: `TrackedBlockAllocator` (`specs/BlockLayout.tla` over `src/blocks.rs`) ----

#[derive(Clone, Debug)]
struct ManagedLane {
    phase: &'static str,
    pos: u32,
    target: u32,
    span_start: u32,
    budget: u32,
    table: BlockTable,
    pin: BTreeSet<PinHandleSnapshot>,
}

/// Per-kind block allocator and lane-table driver built directly on `crate::blocks`
/// (`BlockPools`, `BlockTable`, `SlotRecord::reissue`, `byte_offset`) and `blockbook::Geometry`.
/// Single source of slot ownership: `SlotRecord::refs()` in `self.pools` holds one ref
/// per live lane-table entry, one ref per in-flight step pin, and one ref per `start_table` entry.
pub struct TrackedBlockAllocator {
    cfg: BlockLayoutConfig,
    geo: Geometry,
    pools: BlockPools,
    full_meta: Vec<(u32, Vec<u8>)>,
    win_meta: Vec<(u32, Vec<u8>)>,
    lanes: BTreeMap<String, ManagedLane>,
    start_kept: bool,
    start_pos: u32,
    start_state: u32,
    start_table: BlockTable,
    seq: u64,
}

impl TrackedBlockAllocator {
    pub fn new(
        cfg: BlockLayoutConfig,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<Self, SchedulerError> {
        cfg.validate()?;
        let num_blocks = cfg.num_blocks();
        let geo = Geometry {
            block: cfg.block_size,
            window: cfg.window_size,
            every: 1,
            blocks: num_blocks,
        };
        let pools = BlockPools::with_capacities(
            cfg.full_capacity as usize,
            cfg.window_capacity as usize,
            cfg.kvb,
        )?;
        let full_meta = vec![(0u32, vec![0u8; cfg.block_size as usize]); cfg.full_capacity as usize];
        let win_meta = vec![(0u32, vec![0u8; cfg.block_size as usize]); cfg.window_capacity as usize];

        let mut lanes = BTreeMap::new();
        for l in &cfg.lanes {
            lanes.insert(
                l.clone(),
                ManagedLane {
                    phase: "idle",
                    pos: 0,
                    target: 0,
                    span_start: 0,
                    budget: 0,
                    table: BlockTable::new(num_blocks)?,
                    pin: BTreeSet::new(),
                },
            );
        }
        let alloc = TrackedBlockAllocator {
            geo,
            pools,
            full_meta,
            win_meta,
            lanes,
            start_kept: false,
            start_pos: 0,
            start_state: 0,
            start_table: BlockTable::new(num_blocks)?,
            cfg,
            seq: 0,
        };
        emit_block_layout_event(event_sink, &BlockLayoutEvent::Config(alloc.cfg.clone()))?;
        emit_block_layout_config(state_sink, &alloc.cfg)?;
        let init_step = BlockLayoutStep {
            seq: 0,
            action: "Init".into(),
            args: BTreeMap::new(),
            state: alloc.snapshot_state()?,
        };
        emit_block_layout_step(state_sink, &init_step)?;
        alloc.assert_invariants()?;
        Ok(alloc)
    }

    pub fn config(&self) -> &BlockLayoutConfig {
        &self.cfg
    }

    pub fn pools(&self) -> &BlockPools {
        &self.pools
    }

    fn meta_ref(&self, kind: BlockKind) -> &[(u32, Vec<u8>)] {
        match kind {
            BlockKind::Full => &self.full_meta,
            BlockKind::Window => &self.win_meta,
        }
    }

    fn meta_mut(&mut self, kind: BlockKind) -> &mut [(u32, Vec<u8>)] {
        match kind {
            BlockKind::Full => &mut self.full_meta,
            BlockKind::Window => &mut self.win_meta,
        }
    }

    /// Physical `blocks::Slot` passed into `crate::blocks::byte_offset`: `slot_base + local_slot`.
    fn physical_slot(&self, local_slot: Slot) -> Result<Slot, SchedulerError> {
        let idx = self
            .cfg
            .slot_base
            .checked_add(local_slot.get())
            .ok_or(BlockError::Overflow)?;
        Ok(Slot::new(idx)?)
    }

    fn hold_slot(&mut self, kind: BlockKind, slot: Slot) -> Result<(), SchedulerError> {
        self.pools.of_mut(kind).record_mut(slot)?.add_ref()?;
        Ok(())
    }

    fn release_slot(&mut self, kind: BlockKind, slot: Slot) -> Result<(), SchedulerError> {
        self.pools.of_mut(kind).record_mut(slot)?.drop_ref()?;
        Ok(())
    }

    fn needed_keys(&self, span_start: u32, span_len: u32) -> Vec<(BlockKind, u32)> {
        let mut keys = Vec::new();
        for b in self.geo.full(span_start + span_len) {
            keys.push((BlockKind::Full, b));
        }
        for b in span_window_blocks(span_start, span_len, self.cfg.block_size, self.cfg.window_size) {
            keys.push((BlockKind::Window, b));
        }
        keys
    }

    /// Lowest unreferenced (`refs() == 0`) slot whose generation is below `cfg.max_generation`.
    pub fn first_available_slot(&self, kind: BlockKind) -> Option<Slot> {
        let pool = self.pools.of(kind);
        for n in 1..=(pool.len() as u32) {
            let slot = Slot::new(n).ok()?;
            let rec = pool.record(slot).ok()?;
            if rec.refs() == 0 && rec.generation() < self.cfg.max_generation {
                return Some(slot);
            }
        }
        None
    }

    pub fn missing_needed_keys(&self, lane: &str) -> Result<Vec<(BlockKind, u32)>, SchedulerError> {
        let l = self.lanes.get(lane).ok_or(SchedulerError::Precondition("unknown lane"))?;
        if l.phase != "prepare" {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for (k, b) in self.needed_keys(l.span_start, l.target - l.span_start) {
            if l.table.get(k, BlockIndex::new(b))?.is_none() {
                out.push((k, b));
            }
        }
        Ok(out)
    }

    fn record_step(
        &mut self,
        action: &str,
        args: BTreeMap<String, JsonVal>,
        ev: BlockLayoutEvent,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(), SchedulerError> {
        self.seq += 1;
        emit_block_layout_event(event_sink, &ev)?;
        let step = BlockLayoutStep {
            seq: self.seq,
            action: action.into(),
            args,
            state: self.snapshot_state()?,
        };
        emit_block_layout_step(state_sink, &step)?;
        self.assert_invariants()
    }

    pub fn prepare(
        &mut self,
        lane: &str,
        span_tokens: u32,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(), SchedulerError> {
        if span_tokens == 0 || span_tokens > self.cfg.max_span {
            return Err(SchedulerError::Precondition("span_tokens outside 1..=max_span"));
        }
        let budget = span_bound_blocks(span_tokens, self.cfg.block_size, self.cfg.window_size)?;
        let pos_before = {
            let l = self.lanes.get_mut(lane).ok_or(SchedulerError::Precondition("unknown lane"))?;
            if l.phase != "idle" || l.pos + span_tokens > self.cfg.max_pos {
                return Err(SchedulerError::Precondition("prepare requires idle lane within max_pos"));
            }
            let p = l.pos;
            l.phase = "prepare";
            l.span_start = p;
            l.target = p + span_tokens;
            l.budget = budget;
            p
        };
        let mut args = BTreeMap::new();
        args.insert("lane".into(), JsonVal::str(lane));
        args.insert("tokens".into(), JsonVal::uint(span_tokens));
        self.record_step(
            "Prepare",
            args,
            BlockLayoutEvent::Prepare {
                lane: lane.into(),
                tokens: span_tokens,
                budget,
                pos: pos_before,
            },
            event_sink,
            state_sink,
        )
    }

    pub fn alloc_lowest_free(
        &mut self,
        lane: &str,
        kind: BlockKind,
        block: u32,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<BlockEntry, SchedulerError> {
        let (span_start, target, already) = {
            let l = self.lanes.get(lane).ok_or(SchedulerError::Precondition("unknown lane"))?;
            if l.phase != "prepare" {
                return Err(SchedulerError::Precondition("alloc requires prepare phase"));
            }
            let has = l.table.get(kind, BlockIndex::new(block))?.is_some();
            (l.span_start, l.target, has)
        };
        if already || !self.needed_keys(span_start, target - span_start).contains(&(kind, block)) {
            return Err(SchedulerError::Precondition("key not needed or already present in lane table"));
        }
        let slot = self
            .first_available_slot(kind)
            .ok_or(SchedulerError::Precondition("no free non-exhausted slot in pool"))?;

        // Call the REAL `crate::blocks::byte_offset` on the physical slot (`slot_base + slot`).
        let phys_slot = self.physical_slot(slot)?;
        let offset = byte_offset(kind, phys_slot, self.cfg.kvb)?;
        let limbs = OffsetLimbs::from_u64(offset)?;

        let kvb = self.cfg.kvb;
        let bsz = self.cfg.block_size as usize;
        let new_gen = {
            let rec = self.pools.of_mut(kind).record_mut(slot)?;
            rec.reissue()?;
            rec.set_length(block_length(0)?)?;
            rec.add_ref()?;
            rec.generation()
        };
        {
            let m = &mut self.meta_mut(kind)[slot.pool_index().get() as usize];
            m.0 = block;
            m.1.fill(0);
            debug_assert_eq!(m.1.len(), bsz);
        }

        let entry = BlockEntry::new(slot, new_gen, offset, kind, kvb);
        self.lanes
            .get_mut(lane)
            .unwrap()
            .table
            .set(kind, BlockIndex::new(block), Some(entry))?;

        let mut args = BTreeMap::new();
        args.insert("lane".into(), JsonVal::str(lane));
        args.insert("kind".into(), JsonVal::str(kind_name(kind)));
        args.insert("block".into(), JsonVal::uint(block));
        args.insert("slot".into(), JsonVal::uint(slot.get()));
        self.record_step(
            "Alloc",
            args,
            BlockLayoutEvent::Alloc {
                lane: lane.into(),
                kind,
                block,
                slot: slot.get(),
                generation: new_gen,
                offset,
                offset_limbs: limbs,
                kvb,
            },
            event_sink,
            state_sink,
        )?;
        Ok(entry)
    }

    pub fn begin_step(
        &mut self,
        lane: &str,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(), SchedulerError> {
        let pins_vec = {
            let l = self.lanes.get(lane).ok_or(SchedulerError::Precondition("unknown lane"))?;
            if l.phase != "prepare" {
                return Err(SchedulerError::Precondition("begin_step requires prepare phase"));
            }
            let mut pins = BTreeSet::new();
            for (k, b) in self.needed_keys(l.span_start, l.target - l.span_start) {
                let e = l
                    .table
                    .get(k, BlockIndex::new(b))?
                    .ok_or(SchedulerError::Precondition("missing needed block before begin_step"))?;
                pins.insert(PinHandleSnapshot {
                    kind: e.kind(),
                    slot: e.slot().get(),
                    generation: e.generation(),
                });
            }
            pins
        };
        for h in &pins_vec {
            self.hold_slot(h.kind, Slot::new(h.slot)?)?;
        }
        let pin_count = pins_vec.len() as u32;
        {
            let l = self.lanes.get_mut(lane).unwrap();
            l.phase = "step";
            l.pin = pins_vec;
        }
        let mut args = BTreeMap::new();
        args.insert("lane".into(), JsonVal::str(lane));
        self.record_step(
            "BeginStep",
            args,
            BlockLayoutEvent::BeginStep { lane: lane.into(), pin_count },
            event_sink,
            state_sink,
        )
    }

    pub fn write_one(
        &mut self,
        lane: &str,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<u32, SchedulerError> {
        let (pos_before, full_slot, win_slot) = {
            let l = self.lanes.get(lane).ok_or(SchedulerError::Precondition("unknown lane"))?;
            if l.phase != "step" || l.pos >= l.target {
                return Err(SchedulerError::Precondition("write_one requires step phase with pos < target"));
            }
            let b = BlockIndex::new(l.pos / self.cfg.block_size);
            let fe = l
                .table
                .get(BlockKind::Full, b)?
                .ok_or(SchedulerError::Precondition("missing full block entry in write"))?;
            let we = l
                .table
                .get(BlockKind::Window, b)?
                .ok_or(SchedulerError::Precondition("missing window block entry in write"))?;
            (l.pos, fe.slot(), we.slot())
        };
        let in_blk = (pos_before % self.cfg.block_size) as usize;
        let bsz = self.cfg.block_size;
        for (kind, slot) in [(BlockKind::Full, full_slot), (BlockKind::Window, win_slot)] {
            let rec = self.pools.of_mut(kind).record_mut(slot)?;
            let next_len = rec.length() + 1;
            rec.set_length(next_len)?;
            if next_len == bsz {
                rec.seal();
            }
            self.meta_mut(kind)[slot.pool_index().get() as usize].1[in_blk] = 1;
        }
        let new_pos = pos_before + 1;
        self.lanes.get_mut(lane).unwrap().pos = new_pos;

        let mut args = BTreeMap::new();
        args.insert("lane".into(), JsonVal::str(lane));
        args.insert("pos".into(), JsonVal::uint(new_pos));
        self.record_step(
            "Write",
            args,
            BlockLayoutEvent::Write { lane: lane.into(), pos: new_pos },
            event_sink,
            state_sink,
        )?;
        Ok(new_pos)
    }

    pub fn end_step(
        &mut self,
        lane: &str,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(), SchedulerError> {
        let (pos_now, released_pins) = {
            let l = self.lanes.get_mut(lane).ok_or(SchedulerError::Precondition("unknown lane"))?;
            if l.phase != "step" || l.pos != l.target {
                return Err(SchedulerError::Precondition("end_step requires step phase at pos == target"));
            }
            l.phase = "idle";
            let pins = std::mem::take(&mut l.pin);
            (l.pos, pins)
        };
        for h in released_pins {
            self.release_slot(h.kind, Slot::new(h.slot)?)?;
        }
        let mut args = BTreeMap::new();
        args.insert("lane".into(), JsonVal::str(lane));
        self.record_step(
            "EndStep",
            args,
            BlockLayoutEvent::EndStep { lane: lane.into(), pos: pos_now },
            event_sink,
            state_sink,
        )
    }

    pub fn slide_window(
        &mut self,
        lane: &str,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<bool, SchedulerError> {
        let pruned: Vec<(BlockIndex, BlockEntry)> = {
            let l = self.lanes.get(lane).ok_or(SchedulerError::Precondition("unknown lane"))?;
            if l.phase != "idle" {
                return Err(SchedulerError::Precondition("slide_window requires idle lane"));
            }
            let keep = self.geo.history(l.pos);
            l.table
                .entries()
                .filter(|(b, e)| e.kind() == BlockKind::Window && !keep.contains(&b.get()))
                .collect()
        };
        if pruned.is_empty() {
            return Ok(false);
        }
        for (b, e) in pruned {
            self.lanes.get_mut(lane).unwrap().table.set(BlockKind::Window, b, None)?;
            self.release_slot(e.kind(), e.slot())?;
        }
        let mut args = BTreeMap::new();
        args.insert("lane".into(), JsonVal::str(lane));
        self.record_step(
            "SlideWindow",
            args,
            BlockLayoutEvent::SlideWindow { lane: lane.into() },
            event_sink,
            state_sink,
        )?;
        Ok(true)
    }

    pub fn keep_start(
        &mut self,
        lane: &str,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(), SchedulerError> {
        if self.start_kept {
            return Err(SchedulerError::Precondition("start point already kept"));
        }
        let num_blocks = self.cfg.num_blocks();
        let (p, new_start_table, retained_entries) = {
            let l = self.lanes.get(lane).ok_or(SchedulerError::Precondition("unknown lane"))?;
            if l.phase != "idle" || l.pos == 0 || l.pos % self.cfg.block_size != 0 {
                return Err(SchedulerError::Precondition("keep_start requires idle lane at block boundary > 0"));
            }
            let mut st = BlockTable::new(num_blocks)?;
            let mut held = Vec::new();
            for (kind, b) in self.geo.retained(l.pos) {
                let bi = BlockIndex::new(b);
                let e = l
                    .table
                    .get(kind, bi)?
                    .ok_or(SchedulerError::Precondition("missing retained block at boundary"))?;
                st.set(kind, bi, Some(e))?;
                held.push(e);
            }
            (l.pos, st, held)
        };
        for e in retained_entries {
            self.hold_slot(e.kind(), e.slot())?;
        }
        self.start_kept = true;
        self.start_pos = p;
        self.start_state = p;
        self.start_table = new_start_table;

        let mut args = BTreeMap::new();
        args.insert("lane".into(), JsonVal::str(lane));
        args.insert("pos".into(), JsonVal::uint(p));
        self.record_step(
            "KeepStart",
            args,
            BlockLayoutEvent::KeepStart { lane: lane.into(), pos: p },
            event_sink,
            state_sink,
        )
    }

    pub fn drop_start(
        &mut self,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(), SchedulerError> {
        if !self.start_kept {
            return Err(SchedulerError::Precondition("no start point kept"));
        }
        let old = std::mem::replace(&mut self.start_table, BlockTable::new(self.cfg.num_blocks())?);
        for (_, e) in old.entries() {
            self.release_slot(e.kind(), e.slot())?;
        }
        self.start_kept = false;
        self.record_step(
            "DropStart",
            BTreeMap::new(),
            BlockLayoutEvent::DropStart,
            event_sink,
            state_sink,
        )
    }

    pub fn match_start(
        &mut self,
        lane: &str,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(), SchedulerError> {
        if !self.start_kept {
            return Err(SchedulerError::Precondition("match_start requires start_kept"));
        }
        {
            let l = self.lanes.get(lane).ok_or(SchedulerError::Precondition("unknown lane"))?;
            if l.phase != "idle" || l.pos != 0 {
                return Err(SchedulerError::Precondition("match_start requires idle lane at pos 0"));
            }
        }
        let p = self.start_pos;
        let cloned = self.start_table.clone();
        let to_hold: Vec<BlockEntry> = cloned.entries().map(|(_, e)| e).collect();
        let prev = std::mem::replace(&mut self.lanes.get_mut(lane).unwrap().table, cloned);
        self.lanes.get_mut(lane).unwrap().pos = p;
        for (_, e) in prev.entries() {
            self.release_slot(e.kind(), e.slot())?;
        }
        for e in to_hold {
            self.hold_slot(e.kind(), e.slot())?;
        }

        let mut args = BTreeMap::new();
        args.insert("lane".into(), JsonVal::str(lane));
        args.insert("pos".into(), JsonVal::uint(p));
        self.record_step(
            "Match",
            args,
            BlockLayoutEvent::Match { lane: lane.into(), pos: p },
            event_sink,
            state_sink,
        )
    }

    pub fn finish(
        &mut self,
        lane: &str,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(), SchedulerError> {
        let num_blocks = self.cfg.num_blocks();
        let released: Vec<BlockEntry> = {
            let l = self.lanes.get(lane).ok_or(SchedulerError::Precondition("unknown lane"))?;
            if l.phase != "idle" {
                return Err(SchedulerError::Precondition("finish requires idle lane"));
            }
            l.table.entries().map(|(_, e)| e).collect()
        };
        for e in released {
            self.release_slot(e.kind(), e.slot())?;
        }
        {
            let l = self.lanes.get_mut(lane).unwrap();
            l.phase = "done";
            l.table = BlockTable::new(num_blocks)?; // MUTANT_ANCHOR_CLEAR_TABLE
        }
        let mut args = BTreeMap::new();
        args.insert("lane".into(), JsonVal::str(lane));
        self.record_step(
            "Finish",
            args,
            BlockLayoutEvent::Finish { lane: lane.into() },
            event_sink,
            state_sink,
        )
    }

    pub fn snapshot_state(&self) -> Result<BlockLayoutState, TraceError> {
        let (full_pool, window_pool) =
            BlockLayoutState::snapshot_pools(&self.pools, &self.full_meta, &self.win_meta)?;
        let num_blocks = self.cfg.num_blocks();
        let mut table = BTreeMap::new();
        let mut pos = BTreeMap::new();
        let mut phase = BTreeMap::new();
        let mut target = BTreeMap::new();
        let mut span_start = BTreeMap::new();
        let mut budget = BTreeMap::new();
        let mut pin = BTreeMap::new();
        for (name, l) in &self.lanes {
            table.insert(name.clone(), BlockLayoutState::snapshot_table(&l.table, num_blocks)?);
            pos.insert(name.clone(), l.pos);
            phase.insert(name.clone(), l.phase.into());
            target.insert(name.clone(), l.target);
            span_start.insert(name.clone(), l.span_start);
            budget.insert(name.clone(), l.budget);
            pin.insert(name.clone(), l.pin.iter().cloned().collect());
        }
        Ok(BlockLayoutState {
            full_pool,
            window_pool,
            table,
            pos,
            phase,
            target,
            span_start,
            budget,
            pin,
            start_kept: self.start_kept,
            start_pos: self.start_pos,
            start_state: self.start_state,
            start_table: BlockLayoutState::snapshot_table(&self.start_table, num_blocks)?,
        })
    }

    /// Runtime mirror of all 9 invariants in `specs/BlockLayout.tla` plus `SlotRecord::refs()` consistency.
    pub fn assert_invariants(&self) -> Result<(), SchedulerError> {
        let check_entry = |kind: BlockKind, block: u32, e: &BlockEntry| -> Result<(), SchedulerError> {
            if e.kind() != kind || e.kvb() != self.cfg.kvb {
                return Err(SchedulerError::InvariantViolation("GenerationMatches: entry kind/kvb mismatch".into()));
            }
            let rec = self
                .pools
                .of(kind)
                .record(e.slot())
                .map_err(|_| SchedulerError::InvariantViolation("GenerationMatches: slot out of range".into()))?;
            if rec.generation() == 0 || rec.generation() != e.generation() {
                return Err(SchedulerError::InvariantViolation("GenerationMatches: stale generation".into()));
            }
            if self.meta_ref(kind)[e.slot().pool_index().get() as usize].0 != block {
                return Err(SchedulerError::InvariantViolation("TableIndex: slot block mismatch".into()));
            }
            let phys = self.physical_slot(e.slot())?;
            let want_off = byte_offset(kind, phys, self.cfg.kvb)?;
            if e.offset() != want_off {
                return Err(SchedulerError::InvariantViolation("Address64: entry offset mismatch".into()));
            }
            Ok(())
        };

        let mut expected_refs: BTreeMap<(u8, u32), u32> = BTreeMap::new();
        let rk = |k: BlockKind| if k == BlockKind::Full { 0u8 } else { 1u8 };

        for l in self.lanes.values() {
            for (b, e) in l.table.entries() {
                check_entry(e.kind(), b.get(), &e)?;
                if l.phase != "done" {
                    *expected_refs.entry((rk(e.kind()), e.slot().get())).or_default() += 1;
                }
            }
            for h in &l.pin {
                *expected_refs.entry((rk(h.kind), h.slot)).or_default() += 1;
            }
            if l.phase != "done" {
                for b in self.geo.full(l.pos) {
                    if l.table.get(BlockKind::Full, BlockIndex::new(b))?.is_none() {
                        return Err(SchedulerError::InvariantViolation("FullCovered violated".into()));
                    }
                }
                for b in self.geo.history(l.pos) {
                    if l.table.get(BlockKind::Window, BlockIndex::new(b))?.is_none() {
                        return Err(SchedulerError::InvariantViolation("WindowCovered violated".into()));
                    }
                }
            }
            if l.phase == "step" {
                let need_bound =
                    span_bound_blocks(l.target - l.span_start, self.cfg.block_size, self.cfg.window_size)?;
                if l.budget < need_bound {
                    return Err(SchedulerError::InvariantViolation("SpanPinned: budget too small".into()));
                }
                for (k, b) in self.needed_keys(l.span_start, l.target - l.span_start) {
                    let e = l.table.get(k, BlockIndex::new(b))?.ok_or_else(|| {
                        SchedulerError::InvariantViolation("SpanPinned: missing needed table entry".into())
                    })?;
                    let h = PinHandleSnapshot {
                        kind: k,
                        slot: e.slot().get(),
                        generation: e.generation(),
                    };
                    if !l.pin.contains(&h) {
                        return Err(SchedulerError::InvariantViolation("SpanPinned: handle not pinned".into()));
                    }
                }
                for h in &l.pin {
                    let rec = self.pools.of(h.kind).record(Slot::new(h.slot)?)?;
                    if rec.generation() != h.generation {
                        return Err(SchedulerError::InvariantViolation(
                            "SpanPinned: pinned generation mutated".into(),
                        ));
                    }
                }
            }
        }

        if self.start_kept {
            if self.start_state != self.start_pos {
                return Err(SchedulerError::InvariantViolation("StartRetained: startState != startPos".into()));
            }
            for (b, e) in self.start_table.entries() {
                *expected_refs.entry((rk(e.kind()), e.slot().get())).or_default() += 1;
                check_entry(e.kind(), b.get(), &e)?;
            }
            for (kind, b) in self.geo.retained(self.start_pos) {
                let e = self.start_table.get(kind, BlockIndex::new(b))?.ok_or_else(|| {
                    SchedulerError::InvariantViolation("StartRetained: missing retained block".into())
                })?;
                if self.pools.of(kind).record(e.slot())?.length() != self.cfg.block_size {
                    return Err(SchedulerError::InvariantViolation("StartRetained: block unsealed".into()));
                }
            }
        }

        for kind in [BlockKind::Full, BlockKind::Window] {
            let pool = self.pools.of(kind);
            let stride = kind.stride_bytes(self.cfg.kvb)?;
            let mut live_slots = BTreeSet::new();
            for s in 1..=(pool.len() as u32) {
                let slot = Slot::new(s)?;
                let rec = pool.record(slot)?;
                let want_refs = expected_refs.get(&(rk(kind), s)).copied().unwrap_or(0);
                if rec.refs() != want_refs {
                    return Err(SchedulerError::InvariantViolation(format!(
                        "SlotRecord::refs mismatch on {:?} slot {s}: got {}, want {want_refs}",
                        kind,
                        rec.refs()
                    )));
                }
                if rec.refs() > 0 {
                    live_slots.insert(s);
                }
                if rec.length() < self.cfg.block_size {
                    let owners = self
                        .lanes
                        .values()
                        .filter(|l| {
                            l.phase != "done"
                                && l.table
                                    .entries()
                                    .any(|(_, e)| e.kind() == kind && e.slot().get() == s)
                        })
                        .count();
                    if owners > 1 {
                        return Err(SchedulerError::InvariantViolation(
                            "TailPrivateZero: unsealed tail shared across multiple lanes".into(),
                        ));
                    }
                    let data = &self.meta_ref(kind)[(s - 1) as usize].1;
                    if data[rec.length() as usize..].iter().any(|&d| d != 0) {
                        return Err(SchedulerError::InvariantViolation(
                            "TailPrivateZero: non-zero byte past initialized tail length".into(),
                        ));
                    }
                }
            }
            // Pairwise byte-range non-overlap across all live slots of this kind:
            let ordered: Vec<u32> = live_slots.into_iter().collect();
            for pair in ordered.windows(2) {
                let off_a = byte_offset(kind, self.physical_slot(Slot::new(pair[0])?)?, self.cfg.kvb)?;
                let off_b = byte_offset(kind, self.physical_slot(Slot::new(pair[1])?)?, self.cfg.kvb)?;
                let end_a = off_a.checked_add(stride).ok_or(BlockError::Overflow)?;
                if end_a > off_b {
                    return Err(SchedulerError::InvariantViolation(
                        "Address64: overlapping live slot byte ranges".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

// ---- Part B: Decoupled Prefix Seam + `CacheScheduler` (`specs/CacheScheduler.tla`) ----

/// Seam allowing `src/prefix.rs` to supply deepest retained start boundaries.
pub trait PrefixBoundaryLookup {
    fn deepest_boundary(&self, prefix: &str, max_pos: u32) -> Option<u32>;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArriveOutcome {
    Queued,
    DirectAdmitted,
    Refused429 { retry_after_steps: u32 },
}

#[derive(Clone, Debug)]
struct SchedBlock {
    tier: &'static str,
    sealed: bool,
    idx: u32,
    length: u32,
    data: Vec<u8>,
    ref_count: u32,
    last_used_tick: u64,
}

impl SchedBlock {
    fn empty(block_size: u32) -> Self {
        SchedBlock {
            tier: "none",
            sealed: false,
            idx: 0,
            length: 0,
            data: vec![0; block_size as usize],
            ref_count: 0,
            last_used_tick: 0,
        }
    }
    fn in_memory(&self) -> bool {
        matches!(self.tier, "memory" | "both")
    }
    fn on_disk(&self) -> bool {
        matches!(self.tier, "disk" | "both")
    }
    fn present(&self) -> bool {
        self.tier != "none"
    }
}

#[derive(Clone, Debug)]
struct SchedStartPoint {
    kept: bool,
    state_kept: bool,
    full: BTreeSet<CacheBlockId>,
    win: BTreeSet<CacheBlockId>,
    last_used_tick: u64,
}

impl SchedStartPoint {
    fn empty() -> Self {
        SchedStartPoint {
            kept: false,
            state_kept: false,
            full: BTreeSet::new(),
            win: BTreeSet::new(),
            last_used_tick: 0,
        }
    }
}

#[derive(Clone, Debug)]
struct SchedLane {
    state: &'static str,
    prefix: String,
    target: u32,
    pos: u32,
    full: Vec<Option<CacheBlockId>>,
    win: Vec<Option<CacheBlockId>>,
    reservation: u32,
    admit_budget: u32,
    wait_ticks: u32,
}

/// Deterministic implementation of `specs/CacheScheduler.tla`:
/// sealed tree blocks (`memory | disk | both`), unsealed private tails (`memory` only,
/// zeroed past length, invisible to matching), start points, worst-case admission
/// budgets, atomic restore reservations, decode-priority step building with `MaxBypass`
/// fairness, in-step cancellation fences, and cold-only LRU SSD writeback.
pub struct CacheScheduler {
    cfg: CacheSchedulerConfig,
    blocks: BTreeMap<CacheBlockId, SchedBlock>,
    start_points: BTreeMap<(String, u32), SchedStartPoint>,
    lanes: BTreeMap<String, SchedLane>,
    queue_seq: Vec<String>,
    step_active: bool,
    step_lanes: BTreeSet<String>,
    step_span: BTreeMap<String, u32>,
    step_pins: BTreeSet<CacheBlockId>,
    step_io: BTreeSet<CacheBlockId>,
    gpu_done: bool,
    logical_tick: u64,
    seq: u64,
}

impl CacheScheduler {
    pub fn new(
        cfg: CacheSchedulerConfig,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<Self, SchedulerError> {
        cfg.validate()?;
        let mut blocks = BTreeMap::new();
        for id in cfg.all_block_ids() {
            blocks.insert(id, SchedBlock::empty(cfg.block_size));
        }
        let mut start_points = BTreeMap::new();
        for p in &cfg.prefixes {
            for b in cfg.boundaries() {
                start_points.insert((p.clone(), b), SchedStartPoint::empty());
            }
        }
        let def_pfx = cfg.default_prefix().to_owned();
        let def_tgt = cfg.default_target();
        let nblk = cfg.num_blocks() as usize;
        let mut lanes = BTreeMap::new();
        let mut step_span = BTreeMap::new();
        for l in &cfg.lanes {
            lanes.insert(
                l.clone(),
                SchedLane {
                    state: "Idle",
                    prefix: def_pfx.clone(),
                    target: def_tgt,
                    pos: 0,
                    full: vec![None; nblk],
                    win: vec![None; nblk],
                    reservation: 0,
                    admit_budget: 0,
                    wait_ticks: 0,
                },
            );
            step_span.insert(l.clone(), 0);
        }
        let sched = CacheScheduler {
            cfg,
            blocks,
            start_points,
            lanes,
            queue_seq: Vec::new(),
            step_active: false,
            step_lanes: BTreeSet::new(),
            step_span,
            step_pins: BTreeSet::new(),
            step_io: BTreeSet::new(),
            gpu_done: false,
            logical_tick: 0,
            seq: 0,
        };
        emit_cache_scheduler_event(event_sink, &CacheSchedulerEvent::Config(sched.cfg.clone()))?;
        emit_cache_scheduler_config(state_sink, &sched.cfg)?;
        let init_step = CacheSchedulerStep {
            seq: 0,
            action: "Init".into(),
            args: BTreeMap::new(),
            state: sched.snapshot_state(),
        };
        emit_cache_scheduler_step(state_sink, &init_step)?;
        sched.assert_invariants()?;
        Ok(sched)
    }

    pub fn config(&self) -> &CacheSchedulerConfig {
        &self.cfg
    }

    pub fn lane_state(&self, lane: &str) -> Option<&str> {
        self.lanes.get(lane).map(|l| l.state)
    }

    pub fn lane_pos(&self, lane: &str) -> Option<u32> {
        self.lanes.get(lane).map(|l| l.pos)
    }

    pub fn lane_wait_ticks(&self, lane: &str) -> Option<u32> {
        self.lanes.get(lane).map(|l| l.wait_ticks)
    }

    pub fn mem_used(&self) -> u32 {
        self.blocks.values().filter(|b| b.in_memory()).count() as u32
    }

    pub fn disk_used(&self) -> u32 {
        self.blocks.values().filter(|b| b.on_disk()).count() as u32
    }

    pub fn reserved_total(&self) -> u32 {
        self.lanes
            .values()
            .filter(|l| matches!(l.state, "Restoring" | "Ready" | "InStep" | "Cancelling"))
            .map(|l| l.reservation)
            .sum()
    }

    pub fn admitted_total(&self) -> u32 {
        self.lanes
            .values()
            .filter(|l| matches!(l.state, "Admitted" | "Restoring" | "Ready" | "InStep" | "Cancelling"))
            .map(|l| l.admit_budget)
            .sum()
    }

    pub fn worst_case_for(&self, target: u32) -> u32 {
        worst_case_blocks_checked(
            target,
            self.cfg.max_span,
            self.cfg.block_size,
            self.cfg.window_size,
        )
        .expect("validated CacheSchedulerConfig bounds")
        .2
    }

    fn prompt_end_of(&self, target: u32) -> u32 {
        if target > self.cfg.block_size {
            target - self.cfg.block_size
        } else {
            0
        }
    }

    fn is_prefilling(&self, l: &SchedLane) -> bool {
        l.state == "Ready" && l.pos < self.prompt_end_of(l.target)
    }

    fn is_decoding(&self, l: &SchedLane) -> bool {
        l.state == "Ready" && l.pos >= self.prompt_end_of(l.target) && l.pos < l.target
    }

    fn lane_blocks_of(l: &SchedLane) -> BTreeSet<CacheBlockId> {
        l.full
            .iter()
            .chain(l.win.iter())
            .filter_map(|x| x.clone())
            .collect()
    }

    fn lane_referrers(&self, id: &CacheBlockId) -> BTreeSet<String> {
        self.lanes
            .iter()
            .filter(|(_, l)| {
                matches!(l.state, "Admitted" | "Restoring" | "Ready" | "InStep" | "Cancelling")
                    && Self::lane_blocks_of(l).contains(id)
            })
            .map(|(name, _)| name.clone())
            .collect()
    }

    fn start_referrers_count(&self, id: &CacheBlockId) -> u32 {
        self.start_points
            .values()
            .filter(|sp| sp.kept && (sp.full.contains(id) || sp.win.contains(id)))
            .count() as u32
    }

    fn sync_ref_counts(&mut self) {
        let bsz = self.cfg.block_size;
        let ids: Vec<CacheBlockId> = self.blocks.keys().cloned().collect();
        for id in ids {
            if self.blocks[&id].tier == "none" {
                self.blocks.insert(id, SchedBlock::empty(bsz));
                continue;
            }
            let refs = (self.lane_referrers(&id).len() as u32) + self.start_referrers_count(&id);
            if !self.blocks[&id].sealed && refs == 0 {
                self.blocks.insert(id, SchedBlock::empty(bsz));
            } else {
                self.blocks.get_mut(&id).unwrap().ref_count = refs;
            }
        }
    }

    pub fn best_boundary(&self, prefix: &str, max_pos: u32) -> u32 {
        self.cfg
            .boundaries()
            .into_iter()
            .filter(|&k| {
                k <= max_pos
                    && self
                        .start_points
                        .get(&(prefix.to_owned(), k))
                        .is_some_and(|sp| sp.kept)
            })
            .max()
            .unwrap_or(0)
    }

    fn need_alloc(&self, l: &SchedLane) -> u32 {
        if l.state == "Ready" && l.pos < l.target {
            let bi = (l.pos / self.cfg.block_size) as usize;
            u32::from(l.full[bi].is_none()) + u32::from(l.win[bi].is_none())
        } else {
            0
        }
    }

    fn cold_mem_blocks(&self) -> Vec<CacheBlockId> {
        let mut v: Vec<CacheBlockId> = self
            .blocks
            .iter()
            .filter(|(id, b)| b.in_memory() && self.lane_referrers(id).is_empty() && !self.step_io.contains(id))
            .map(|(id, _)| id.clone())
            .collect();
        v.sort_by_key(|id| (self.blocks[id].last_used_tick, id.clone()));
        v
    }

    fn cold_disk_blocks(&self) -> Vec<CacheBlockId> {
        let mut v: Vec<CacheBlockId> = self
            .blocks
            .iter()
            .filter(|(id, b)| b.on_disk() && self.lane_referrers(id).is_empty() && !self.step_io.contains(id))
            .map(|(id, _)| id.clone())
            .collect();
        v.sort_by_key(|id| (self.blocks[id].last_used_tick, id.clone()));
        v
    }

    pub fn mem_pressure(&self) -> bool {
        if self.cold_mem_blocks().is_empty() {
            return false;
        }
        let restoring_blocked = self.mem_used() >= self.cfg.mem_budget
            && self.lanes.values().any(|l| {
                l.state == "Restoring"
                    && Self::lane_blocks_of(l)
                        .iter()
                        .any(|b| !self.blocks[b].in_memory())
            });
        let steppable_need: u32 = self
            .lanes
            .values()
            .filter(|l| l.state == "Ready" && l.pos < l.target)
            .map(|l| self.need_alloc(l))
            .sum();
        let has_steppable = self.lanes.values().any(|l| l.state == "Ready" && l.pos < l.target);
        restoring_blocked || (has_steppable && self.mem_used() + steppable_need > self.cfg.mem_budget)
    }

    fn can_fit_request(&self, req_len: u32) -> bool {
        self.admitted_total() + self.worst_case_for(req_len) <= self.cfg.mem_budget
    }

    fn older_queued(&self, a: &str, b: &str) -> bool {
        let ia = self.queue_seq.iter().position(|x| x == a);
        let ib = self.queue_seq.iter().position(|x| x == b);
        matches!((ia, ib), (Some(i), Some(j)) if i < j)
    }

    pub fn can_admit_now(&self, lane: &str) -> bool {
        let Some(l) = self.lanes.get(lane) else {
            return false;
        };
        if l.state != "Queued" || !self.can_fit_request(l.target) {
            return false;
        }
        for (qname, q) in &self.lanes {
            if qname != lane && q.state == "Queued" {
                if self.can_fit_request(q.target)
                    && !(l.wait_ticks > q.wait_ticks
                        || (l.wait_ticks == q.wait_ticks && self.older_queued(lane, qname)))
                {
                    return false;
                }
                if q.wait_ticks >= self.cfg.max_bypass
                    && !(self.older_queued(lane, qname) || l.wait_ticks > q.wait_ticks)
                {
                    return false;
                }
            }
        }
        true
    }

    fn actionable_waiting(&self) -> BTreeSet<String> {
        let has_cold = !self.cold_mem_blocks().is_empty();
        let mem_used = self.mem_used();
        let res_total = self.reserved_total();
        let mut set = BTreeSet::new();
        for (name, l) in &self.lanes {
            let ok = match l.state {
                "Ready" => l.pos < l.target,
                "Admitted" => res_total + self.worst_case_for(l.target) <= self.cfg.mem_budget,
                "Restoring" => {
                    Self::lane_blocks_of(l).iter().all(|b| self.blocks[b].in_memory())
                        || mem_used < self.cfg.mem_budget
                        || has_cold
                }
                "Queued" => self.can_admit_now(name),
                _ => false,
            };
            if ok {
                set.insert(name.clone());
            }
        }
        set
    }

    fn record_step(
        &mut self,
        action: &str,
        args: BTreeMap<String, JsonVal>,
        ev: CacheSchedulerEvent,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(), SchedulerError> {
        self.seq += 1;
        emit_cache_scheduler_event(event_sink, &ev)?;
        let step = CacheSchedulerStep {
            seq: self.seq,
            action: action.into(),
            args,
            state: self.snapshot_state(),
        };
        emit_cache_scheduler_step(state_sink, &step)?;
        self.assert_invariants()
    }

    pub fn arrive(
        &mut self,
        lane: &str,
        prefix: &str,
        req_len: u32,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<ArriveOutcome, SchedulerError> {
        if !self.cfg.prefixes.iter().any(|p| p == prefix) || !self.cfg.request_lengths.contains(&req_len) {
            return Err(SchedulerError::Precondition("unknown prefix or disallowed request length"));
        }
        let state_ok = self
            .lanes
            .get(lane)
            .is_some_and(|l| matches!(l.state, "Idle" | "Refused" | "Cancelled" | "Finished"));
        if !state_ok {
            return Err(SchedulerError::Precondition("arrive requires Idle/Refused/Cancelled/Finished lane"));
        }

        let wc = self.worst_case_for(req_len);
        let queued_names: Vec<String> = self
            .lanes
            .iter()
            .filter(|(_, l)| l.state == "Queued")
            .map(|(n, _)| n.clone())
            .collect();
        let q_cnt = queued_names.len() as u32;
        let can_enqueue = wc <= self.cfg.mem_budget && q_cnt < self.cfg.max_queue;
        let can_direct_admit = wc <= self.cfg.mem_budget
            && q_cnt == self.cfg.max_queue
            && self.can_fit_request(req_len)
            && queued_names.iter().all(|qn| {
                let ql = &self.lanes[qn];
                !self.can_fit_request(ql.target) && ql.wait_ticks < self.cfg.max_bypass
            });

        let nblk = self.cfg.num_blocks() as usize;
        let outcome = if can_enqueue {
            let l = self.lanes.get_mut(lane).unwrap();
            l.state = "Queued";
            l.prefix = prefix.into();
            l.target = req_len;
            l.pos = 0;
            l.full = vec![None; nblk];
            l.win = vec![None; nblk];
            l.reservation = 0;
            l.admit_budget = 0;
            l.wait_ticks = 0;
            self.queue_seq.push(lane.into());
            ArriveOutcome::Queued
        } else if can_direct_admit {
            for qn in &queued_names {
                self.lanes.get_mut(qn).unwrap().wait_ticks += 1;
            }
            let l = self.lanes.get_mut(lane).unwrap();
            l.state = "Admitted";
            l.prefix = prefix.into();
            l.target = req_len;
            l.pos = 0;
            l.full = vec![None; nblk];
            l.win = vec![None; nblk];
            l.reservation = 0;
            l.admit_budget = wc;
            l.wait_ticks = 0;
            ArriveOutcome::DirectAdmitted
        } else {
            let def_pfx = self.cfg.default_prefix().to_owned();
            let def_tgt = self.cfg.default_target();
            let l = self.lanes.get_mut(lane).unwrap();
            l.state = "Refused";
            l.prefix = def_pfx;
            l.target = def_tgt;
            l.pos = 0;
            l.full = vec![None; nblk];
            l.win = vec![None; nblk];
            l.reservation = 0;
            l.admit_budget = 0;
            l.wait_ticks = 0;
            let retry_after_steps = (q_cnt + 1).max(1);
            ArriveOutcome::Refused429 { retry_after_steps }
        };

        let mut args = BTreeMap::new();
        args.insert("lane".into(), JsonVal::str(lane));
        args.insert("prefix".into(), JsonVal::str(prefix));
        args.insert("target".into(), JsonVal::uint(req_len));
        let ev = match outcome {
            ArriveOutcome::Queued => CacheSchedulerEvent::Arrive {
                lane: lane.into(),
                prefix: prefix.into(),
                target: req_len,
                state: "Queued".into(),
            },
            ArriveOutcome::DirectAdmitted => CacheSchedulerEvent::Arrive {
                lane: lane.into(),
                prefix: prefix.into(),
                target: req_len,
                state: "Admitted".into(),
            },
            ArriveOutcome::Refused429 { .. } => CacheSchedulerEvent::Refuse429 {
                lane: lane.into(),
                prefix: prefix.into(),
                target: req_len,
            },
        };
        self.record_step("Arrive", args, ev, event_sink, state_sink)?;
        Ok(outcome)
    }

    pub fn admit(
        &mut self,
        lane: &str,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(), SchedulerError> {
        if !self.can_admit_now(lane) {
            return Err(SchedulerError::Precondition("admit precondition not satisfied"));
        }
        let target = self.lanes[lane].target;
        let wc = self.worst_case_for(target);
        let max_byp = self.cfg.max_bypass;
        let older: Vec<String> = self
            .queue_seq
            .iter()
            .take_while(|x| x.as_str() != lane)
            .cloned()
            .collect();
        for qname in &older {
            let q = self.lanes.get_mut(qname).unwrap();
            if q.state == "Queued" && q.wait_ticks < max_byp {
                q.wait_ticks += 1; // MUTANT_ANCHOR_ADMIT_BYPASS
            }
        }
        self.queue_seq.retain(|x| x != lane);
        let l = self.lanes.get_mut(lane).unwrap();
        l.state = "Admitted";
        l.admit_budget = wc;
        let wait_ticks = l.wait_ticks;

        let mut args = BTreeMap::new();
        args.insert("lane".into(), JsonVal::str(lane));
        self.record_step(
            "Admit",
            args,
            CacheSchedulerEvent::Admit { lane: lane.into(), wait_ticks },
            event_sink,
            state_sink,
        )
    }

    pub fn begin_restore(
        &mut self,
        lane: &str,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(u32, u32), SchedulerError> {
        let (prefix, target) = {
            let l = self.lanes.get(lane).ok_or(SchedulerError::Precondition("unknown lane"))?;
            if l.state != "Admitted" {
                return Err(SchedulerError::Precondition("begin_restore requires Admitted lane"));
            }
            (l.prefix.clone(), l.target)
        };
        let wc = self.worst_case_for(target);
        if self.reserved_total() + wc > self.cfg.mem_budget {
            return Err(SchedulerError::Precondition("insufficient memory budget for atomic restore reservation"));
        }
        let k = self.best_boundary(&prefix, target);
        let nblk = self.cfg.num_blocks() as usize;
        let mut nf = vec![None; nblk];
        let mut nw = vec![None; nblk];
        self.logical_tick += 1;
        let tick = self.logical_tick;
        if k > 0 {
            let sp = self.start_points.get_mut(&(prefix.clone(), k)).unwrap();
            sp.last_used_tick = tick;
            let full_ids: Vec<_> = sp.full.iter().cloned().collect();
            let win_ids: Vec<_> = sp.win.iter().cloned().collect();
            for id in full_ids {
                if let CacheBlockId::Tree { idx, .. } = &id {
                    nf[*idx as usize] = Some(id.clone());
                }
                if let Some(b) = self.blocks.get_mut(&id) {
                    b.last_used_tick = tick;
                }
            }
            for id in win_ids {
                if let CacheBlockId::Tree { idx, .. } = &id {
                    nw[*idx as usize] = Some(id.clone());
                }
                if let Some(b) = self.blocks.get_mut(&id) {
                    b.last_used_tick = tick;
                }
            }
        }
        {
            let l = self.lanes.get_mut(lane).unwrap();
            l.state = "Restoring";
            l.pos = k;
            l.full = nf;
            l.win = nw;
            l.reservation = wc;
        }
        self.sync_ref_counts();

        let mut args = BTreeMap::new();
        args.insert("lane".into(), JsonVal::str(lane));
        args.insert("pos".into(), JsonVal::uint(k));
        args.insert("reservation".into(), JsonVal::uint(wc));
        self.record_step(
            "BeginRestore",
            args,
            CacheSchedulerEvent::BeginRestore {
                lane: lane.into(),
                pos: k,
                reservation: wc,
            },
            event_sink,
            state_sink,
        )?;
        Ok((k, wc))
    }

    pub fn restore_block(
        &mut self,
        lane: &str,
        block_id: &CacheBlockId,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(), SchedulerError> {
        let my_wait = {
            let l = self.lanes.get(lane).ok_or(SchedulerError::Precondition("unknown lane"))?;
            if l.state != "Restoring" || !Self::lane_blocks_of(l).contains(block_id) {
                return Err(SchedulerError::Precondition("restore_block requires Restoring lane owning block"));
            }
            l.wait_ticks
        };
        if self.blocks.get(block_id).map(|b| b.tier) != Some("disk") || self.mem_used() >= self.cfg.mem_budget {
            return Err(SchedulerError::Precondition("block not on disk-only tier or memory full"));
        }
        for (rname, r) in &self.lanes {
            if rname != lane
                && r.state == "Restoring"
                && Self::lane_blocks_of(r).iter().any(|rb| self.blocks[rb].tier == "disk")
                && my_wait < r.wait_ticks
            {
                return Err(SchedulerError::Precondition("another restoring lane has strictly higher wait_ticks"));
            }
        }
        self.logical_tick += 1;
        let b = self.blocks.get_mut(block_id).unwrap();
        b.tier = "both";
        b.last_used_tick = self.logical_tick;

        let CacheBlockId::Tree { prefix, kind, idx } = block_id else {
            return Err(SchedulerError::Precondition("only sealed tree blocks live on disk"));
        };
        let mut args = BTreeMap::new();
        args.insert("lane".into(), JsonVal::str(lane));
        args.insert("block".into(), JsonVal::str(block_id.encode()));
        self.record_step(
            "RestoreBlock",
            args,
            CacheSchedulerEvent::RestoreBlock {
                lane: lane.into(),
                prefix: prefix.clone(),
                kind: *kind,
                block: *idx,
            },
            event_sink,
            state_sink,
        )
    }

    pub fn finish_restore(
        &mut self,
        lane: &str,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(), SchedulerError> {
        {
            let l = self.lanes.get(lane).ok_or(SchedulerError::Precondition("unknown lane"))?;
            if l.state != "Restoring"
                || !Self::lane_blocks_of(l).iter().all(|b| self.blocks[b].in_memory())
            {
                return Err(SchedulerError::Precondition("finish_restore requires all lane blocks in memory"));
            }
        }
        self.lanes.get_mut(lane).unwrap().state = "Ready";
        let mut args = BTreeMap::new();
        args.insert("lane".into(), JsonVal::str(lane));
        self.record_step(
            "FinishRestore",
            args,
            CacheSchedulerEvent::FinishRestore { lane: lane.into() },
            event_sink,
            state_sink,
        )
    }

    pub fn evict_clean_or_unref_mem(
        &mut self,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<CacheBlockId, SchedulerError> {
        if !self.mem_pressure() {
            return Err(SchedulerError::Precondition("evict_clean_or_unref_mem requires MemPressure"));
        }
        let victim = self
            .cold_mem_blocks()
            .into_iter()
            .find(|id| {
                let b = &self.blocks[id];
                b.on_disk() || b.ref_count == 0
            })
            .ok_or(SchedulerError::Precondition("no clean or unreferenced cold memory block"))?;

        let bsz = self.cfg.block_size;
        {
            let b = self.blocks.get_mut(&victim).unwrap();
            if b.ref_count == 0 && !b.on_disk() {
                *b = SchedBlock::empty(bsz);
            } else {
                b.tier = "disk";
            }
        }
        let CacheBlockId::Tree { prefix, kind, idx } = &victim else {
            return Err(SchedulerError::Precondition("unexpected private cold block"));
        };
        let mut args = BTreeMap::new();
        args.insert("block".into(), JsonVal::str(victim.encode()));
        self.record_step(
            "EvictCleanOrUnrefMem",
            args,
            CacheSchedulerEvent::EvictMem {
                prefix: prefix.clone(),
                kind: *kind,
                block: *idx,
            },
            event_sink,
            state_sink,
        )?;
        Ok(victim)
    }

    pub fn start_cold_writeback(
        &mut self,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<CacheBlockId, SchedulerError> {
        if !self.step_io.is_empty() || !self.mem_pressure() || self.disk_used() >= self.cfg.disk_budget {
            return Err(SchedulerError::Precondition("start_cold_writeback precondition failed"));
        }
        let has_clean = self.cold_mem_blocks().iter().any(|id| {
            let b = &self.blocks[id];
            b.on_disk() || b.ref_count == 0
        });
        if has_clean {
            return Err(SchedulerError::Precondition("clean/unref cold blocks must evict before dirty writeback"));
        }
        let victim = self
            .cold_mem_blocks()
            .into_iter()
            .find(|id| {
                let b = &self.blocks[id];
                b.tier == "memory" && b.sealed
            })
            .ok_or(SchedulerError::Precondition("no dirty cold sealed block available"))?;
        self.step_io.insert(victim.clone());

        let CacheBlockId::Tree { prefix, kind, idx } = &victim else {
            return Err(SchedulerError::Precondition("only tree blocks write back to SSD"));
        };
        let mut args = BTreeMap::new();
        args.insert("block".into(), JsonVal::str(victim.encode()));
        self.record_step(
            "StartColdWriteback",
            args,
            CacheSchedulerEvent::StartWriteback {
                prefix: prefix.clone(),
                kind: *kind,
                block: *idx,
            },
            event_sink,
            state_sink,
        )?;
        Ok(victim)
    }

    pub fn complete_step_io(
        &mut self,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(), SchedulerError> {
        if self.step_io.is_empty() || self.disk_used() + (self.step_io.len() as u32) > self.cfg.disk_budget {
            return Err(SchedulerError::Precondition("complete_step_io precondition failed"));
        }
        let io_ids: Vec<CacheBlockId> = self.step_io.iter().cloned().collect();
        let first = io_ids[0].clone();
        let mut resulting_tier = "disk".to_owned();
        for id in io_ids {
            let has_lane_ref = !self.lane_referrers(&id).is_empty();
            if self.blocks[&id].tier == "memory" {
                let next_t = if has_lane_ref { "both" } else { "disk" };
                self.blocks.get_mut(&id).unwrap().tier = next_t;
                resulting_tier = next_t.to_owned();
            }
        }
        self.step_io.clear();
        let CacheBlockId::Tree { prefix, kind, idx } = &first else {
            return Err(SchedulerError::Precondition("expected tree block in step_io"));
        };
        self.record_step(
            "CompleteStepIO",
            BTreeMap::new(),
            CacheSchedulerEvent::CompleteStepIo {
                prefix: prefix.clone(),
                kind: *kind,
                block: *idx,
                tier: resulting_tier,
            },
            event_sink,
            state_sink,
        )
    }

    pub fn reclaim_cold_disk(
        &mut self,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(), SchedulerError> {
        if self.disk_used() + (self.step_io.len() as u32) < self.cfg.disk_budget {
            return Err(SchedulerError::Precondition("disk budget not full"));
        }
        let cold_disk = self.cold_disk_blocks();
        let unref_disk = cold_disk.iter().find(|id| self.blocks[*id].ref_count == 0).cloned();
        if !self.mem_pressure() && unref_disk.is_none() {
            return Err(SchedulerError::Precondition("reclaim_cold_disk requires MemPressure or unref disk block"));
        }
        let bsz = self.cfg.block_size;
        if let Some(id) = unref_disk {
            let b = self.blocks.get_mut(&id).unwrap();
            if b.in_memory() {
                b.tier = "memory";
            } else {
                *b = SchedBlock::empty(bsz);
            }
        } else {
            let cold_union: BTreeSet<CacheBlockId> = cold_disk
                .into_iter()
                .chain(self.cold_mem_blocks())
                .collect();
            let victim_pk = self
                .start_points
                .iter()
                .filter(|(_, sp)| {
                    sp.kept
                        && sp.full.iter().chain(sp.win.iter()).all(|b| !self.step_io.contains(b))
                        && sp.full.iter().chain(sp.win.iter()).any(|b| cold_union.contains(b))
                })
                .min_by_key(|(pk, sp)| (sp.last_used_tick, (*pk).clone()))
                .map(|(pk, _)| pk.clone())
                .ok_or(SchedulerError::Precondition("no eligible cold start point to reclaim"))?;
            self.start_points.insert(victim_pk, SchedStartPoint::empty());
            self.sync_ref_counts();
        }
        self.record_step(
            "ReclaimColdDisk",
            BTreeMap::new(),
            CacheSchedulerEvent::ReclaimDisk,
            event_sink,
            state_sink,
        )
    }

    pub fn drop_unreferenced(
        &mut self,
        block_id: &CacheBlockId,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(), SchedulerError> {
        let ok = self.blocks.get(block_id).is_some_and(|b| b.present() && b.ref_count == 0)
            && !self.step_pins.contains(block_id)
            && !self.step_io.contains(block_id);
        if !ok {
            return Err(SchedulerError::Precondition("drop_unreferenced precondition failed"));
        }
        self.blocks.insert(block_id.clone(), SchedBlock::empty(self.cfg.block_size));
        let CacheBlockId::Tree { prefix, kind, idx } = block_id else {
            return Err(SchedulerError::Precondition("expected tree block in drop_unreferenced"));
        };
        let mut args = BTreeMap::new();
        args.insert("block".into(), JsonVal::str(block_id.encode()));
        self.record_step(
            "DropUnreferenced",
            args,
            CacheSchedulerEvent::DropUnreferenced {
                prefix: prefix.clone(),
                kind: *kind,
                block: *idx,
            },
            event_sink,
            state_sink,
        )
    }

    pub fn slide_window(
        &mut self,
        lane: &str,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<bool, SchedulerError> {
        let keep: BTreeSet<u32> = {
            let l = self.lanes.get(lane).ok_or(SchedulerError::Precondition("unknown lane"))?;
            if l.state != "Ready" {
                return Err(SchedulerError::Precondition("slide_window requires Ready lane"));
            }
            history_blocks(l.pos, self.cfg.block_size, self.cfg.window_size)
                .into_iter()
                .collect()
        };
        let mut changed = false;
        {
            let l = self.lanes.get_mut(lane).unwrap();
            for (i, slot) in l.win.iter_mut().enumerate() {
                if !keep.contains(&(i as u32)) && slot.is_some() {
                    *slot = None;
                    changed = true;
                }
            }
        }
        if !changed {
            return Ok(false);
        }
        self.sync_ref_counts();
        let mut args = BTreeMap::new();
        args.insert("lane".into(), JsonVal::str(lane));
        self.record_step(
            "SlideWindow",
            args,
            CacheSchedulerEvent::SlideWindow { lane: lane.into() },
            event_sink,
            state_sink,
        )?;
        Ok(true)
    }

    pub fn finish_ready(
        &mut self,
        lane: &str,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(), SchedulerError> {
        let def_pfx = self.cfg.default_prefix().to_owned();
        let def_tgt = self.cfg.default_target();
        let nblk = self.cfg.num_blocks() as usize;
        {
            let l = self.lanes.get_mut(lane).ok_or(SchedulerError::Precondition("unknown lane"))?;
            if l.state != "Ready" || l.pos != l.target {
                return Err(SchedulerError::Precondition("finish_ready requires Ready lane at pos == target"));
            }
            l.state = "Finished";
            l.prefix = def_pfx;
            l.target = def_tgt;
            l.pos = 0;
            l.full = vec![None; nblk];
            l.win = vec![None; nblk];
            l.reservation = 0;
            l.admit_budget = 0;
            l.wait_ticks = 0;
        }
        self.sync_ref_counts();
        let mut args = BTreeMap::new();
        args.insert("lane".into(), JsonVal::str(lane));
        self.record_step(
            "FinishReady",
            args,
            CacheSchedulerEvent::Finish { lane: lane.into() },
            event_sink,
            state_sink,
        )
    }

    pub fn build_step(
        &mut self,
        spans: &[(String, u32)],
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(), SchedulerError> {
        if self.step_active || spans.is_empty() {
            return Err(SchedulerError::Precondition("build_step requires inactive step and non-empty spans"));
        }
        let mut chosen = BTreeSet::new();
        let mut span_map: BTreeMap<String, u32> = self.cfg.lanes.iter().map(|l| (l.clone(), 0)).collect();
        let mut total_tokens = 0u32;
        for (ln, tok) in spans {
            if *tok == 0 || !chosen.insert(ln.clone()) {
                return Err(SchedulerError::Precondition("invalid or duplicate lane in spans"));
            }
            let l = self.lanes.get(ln).ok_or(SchedulerError::Precondition("unknown lane in spans"))?;
            if l.state != "Ready" || l.pos >= l.target {
                return Err(SchedulerError::Precondition("chosen lane is not SteppableReady"));
            }
            if self.is_decoding(l) {
                if *tok != 1 {
                    return Err(SchedulerError::Precondition("decoding lane span must be 1"));
                }
            } else {
                let pend = self.prompt_end_of(l.target);
                let rem_in_blk = self.cfg.block_size - (l.pos % self.cfg.block_size);
                if *tok > self.cfg.max_span || l.pos + *tok > pend || *tok > rem_in_blk {
                    return Err(SchedulerError::Precondition("prefill span violates chunk or block boundary"));
                }
            }
            span_map.insert(ln.clone(), *tok);
            total_tokens += *tok;
        }
        if total_tokens > self.cfg.step_token_budget {
            return Err(SchedulerError::Precondition("total step tokens exceed step_token_budget"));
        }

        let steppable: Vec<String> = self
            .lanes
            .iter()
            .filter(|(_, l)| l.state == "Ready" && l.pos < l.target)
            .map(|(n, _)| n.clone())
            .collect();
        let max_w = steppable.iter().map(|n| self.lanes[n].wait_ticks).max().unwrap();

        let mut new_ids = BTreeSet::new();
        for ln in &chosen {
            let l = &self.lanes[ln];
            let bi = (l.pos / self.cfg.block_size) as usize;
            if l.full[bi].is_none() {
                new_ids.insert(CacheBlockId::Priv { lane: ln.clone(), kind: BlockKind::Full });
            }
            if l.win[bi].is_none() {
                new_ids.insert(CacheBlockId::Priv { lane: ln.clone(), kind: BlockKind::Window });
            }
        }
        let alloc_cnt = new_ids.len() as u32;
        if self.mem_used() + alloc_cnt > self.cfg.mem_budget
            || new_ids.iter().any(|id| self.blocks[id].tier != "none")
        {
            return Err(SchedulerError::Precondition("insufficient memory or private tail already active"));
        }

        let mem_used_now = self.mem_used();
        let fit_unchosen_decoders: Vec<String> = steppable
            .iter()
            .filter(|n| {
                !chosen.contains(*n)
                    && self.is_decoding(&self.lanes[*n])
                    && mem_used_now + alloc_cnt + self.need_alloc(&self.lanes[*n]) <= self.cfg.mem_budget
            })
            .cloned()
            .collect();
        let has_max_decoder = steppable.iter().any(|n| {
            let l = &self.lanes[n];
            self.is_decoding(l) && l.wait_ticks == max_w && mem_used_now + self.need_alloc(l) <= self.cfg.mem_budget
        });

        let valid_leader = chosen.iter().any(|leader| {
            let ll = &self.lanes[leader];
            if ll.wait_ticks != max_w {
                return false;
            }
            if has_max_decoder && !self.is_decoding(ll) {
                return false;
            }
            if !fit_unchosen_decoders.is_empty() {
                if total_tokens != self.cfg.step_token_budget {
                    return false;
                }
                if self.is_prefilling(ll) && span_map[leader] != 1 {
                    return false;
                }
                for q in &chosen {
                    if q != leader {
                        let ql = &self.lanes[q];
                        if !self.is_decoding(ql) && ql.wait_ticks < self.cfg.max_bypass {
                            return false;
                        }
                    }
                }
            }
            true
        });
        if !valid_leader {
            return Err(SchedulerError::Precondition("decode priority or longest-waiting leader violated"));
        }

        let bypassed: BTreeSet<String> = self
            .actionable_waiting()
            .into_iter()
            .filter(|n| !chosen.contains(n))
            .collect();
        if bypassed.iter().any(|w| self.lanes[w].wait_ticks >= self.cfg.max_bypass) {
            return Err(SchedulerError::Precondition("StepBypassed lane would exceed MaxBypass"));
        }

        self.logical_tick += 1;
        let tick = self.logical_tick;
        let bsz = self.cfg.block_size;
        for id in &new_ids {
            let CacheBlockId::Priv { lane: owner, .. } = id else { unreachable!() };
            let idx = self.lanes[owner].pos / bsz;
            self.blocks.insert(
                id.clone(),
                SchedBlock {
                    tier: "memory",
                    sealed: false,
                    idx,
                    length: 0,
                    data: vec![0; bsz as usize], // MUTANT_ANCHOR_PRIV_ZERO
                    ref_count: 1,
                    last_used_tick: tick,
                },
            );
        }
        for ln in &chosen {
            let l = self.lanes.get_mut(ln).unwrap();
            let bi = (l.pos / bsz) as usize;
            if l.full[bi].is_none() {
                l.full[bi] = Some(CacheBlockId::Priv { lane: ln.clone(), kind: BlockKind::Full });
            }
            if l.win[bi].is_none() {
                l.win[bi] = Some(CacheBlockId::Priv { lane: ln.clone(), kind: BlockKind::Window });
            }
            l.state = "InStep";
        }
        self.sync_ref_counts();

        let mut all_pins = BTreeSet::new();
        for ln in &chosen {
            let l = &self.lanes[ln];
            let c = span_map[ln];
            for i in full_blocks(l.pos + c, bsz) {
                let id = l.full[i as usize]
                    .clone()
                    .ok_or(SchedulerError::Precondition("missing full block in step pins"))?;
                all_pins.insert(id);
            }
            for i in span_window_blocks(l.pos, c, bsz, self.cfg.window_size) {
                let id = l.win[i as usize]
                    .clone()
                    .ok_or(SchedulerError::Precondition("missing window block in step pins"))?;
                all_pins.insert(id);
            }
        }
        for id in &all_pins {
            if !self.blocks[id].in_memory() {
                return Err(SchedulerError::Precondition("pinned block not resident in memory"));
            }
            self.blocks.get_mut(id).unwrap().last_used_tick = tick;
        }

        let max_byp = self.cfg.max_bypass;
        for (qname, q) in &mut self.lanes {
            if chosen.contains(qname) {
                q.wait_ticks = 0;
            } else if bypassed.contains(qname) && q.wait_ticks <= max_byp {
                q.wait_ticks += 1; // MUTANT_ANCHOR_STEP_BYPASS
            }
        }

        let pin_count = all_pins.len() as u32;
        self.step_active = true;
        self.step_lanes = chosen;
        self.step_span = span_map;
        self.step_pins = all_pins;
        self.gpu_done = false;

        let ordered_spans: Vec<(String, u32)> = self
            .cfg
            .lanes
            .iter()
            .filter_map(|l| {
                let t = self.step_span[l];
                if t > 0 { Some((l.clone(), t)) } else { None }
            })
            .collect();
        self.record_step(
            "BuildStep",
            BTreeMap::new(),
            CacheSchedulerEvent::BuildStep { spans: ordered_spans, pin_count },
            event_sink,
            state_sink,
        )
    }

    pub fn plan_step(&self) -> Option<Vec<(String, u32)>> {
        if self.step_active {
            return None;
        }
        let steppable: Vec<String> = self
            .cfg
            .lanes
            .iter()
            .filter(|n| {
                let l = &self.lanes[*n];
                l.state == "Ready" && l.pos < l.target && self.mem_used() + self.need_alloc(l) <= self.cfg.mem_budget
            })
            .cloned()
            .collect();
        if steppable.is_empty() {
            return None;
        }
        for w in self.actionable_waiting() {
            if self.lanes[&w].wait_ticks >= self.cfg.max_bypass && !steppable.contains(&w) {
                return None;
            }
        }

        let all_steppable_max_w = self
            .lanes
            .values()
            .filter(|l| l.state == "Ready" && l.pos < l.target)
            .map(|l| l.wait_ticks)
            .max()?;

        let leader = steppable
            .iter()
            .filter(|n| self.lanes[*n].wait_ticks == all_steppable_max_w)
            .max_by_key(|n| u8::from(self.is_decoding(&self.lanes[*n])))?
            .clone();

        let mut chosen: BTreeMap<String, u32> = BTreeMap::new();
        let mut used_tokens = 0u32;
        let mut used_alloc = self.need_alloc(&self.lanes[&leader]);

        let max_legal_span = |l: &SchedLane, rem_budget: u32| -> u32 {
            if self.is_decoding(l) {
                1
            } else {
                let pend = self.prompt_end_of(l.target);
                let rem_blk = self.cfg.block_size - (l.pos % self.cfg.block_size);
                self.cfg
                    .max_span
                    .min(pend - l.pos)
                    .min(rem_blk)
                    .min(rem_budget)
            }
        };

        chosen.insert(leader.clone(), 1);
        used_tokens += 1;

        for n in &steppable {
            if !chosen.contains_key(n)
                && self.lanes[n].wait_ticks >= self.cfg.max_bypass
                && used_tokens < self.cfg.step_token_budget
                && self.mem_used() + used_alloc + self.need_alloc(&self.lanes[n]) <= self.cfg.mem_budget
            {
                chosen.insert(n.clone(), 1);
                used_tokens += 1;
                used_alloc += self.need_alloc(&self.lanes[n]);
            }
        }

        let mut unchosen_fit_decoder = false;
        for n in &steppable {
            if !chosen.contains_key(n) && self.is_decoding(&self.lanes[n]) {
                let na = self.need_alloc(&self.lanes[n]);
                if self.mem_used() + used_alloc + na <= self.cfg.mem_budget {
                    if used_tokens < self.cfg.step_token_budget {
                        chosen.insert(n.clone(), 1);
                        used_tokens += 1;
                        used_alloc += na;
                    } else {
                        unchosen_fit_decoder = true;
                    }
                }
            }
        }

        if !unchosen_fit_decoder {
            for n in chosen.keys().cloned().collect::<Vec<_>>() {
                let l = &self.lanes[&n];
                if self.is_prefilling(l) && used_tokens < self.cfg.step_token_budget {
                    let cur = chosen[&n];
                    let cap = max_legal_span(l, cur + (self.cfg.step_token_budget - used_tokens));
                    let add = cap.saturating_sub(cur);
                    *chosen.get_mut(&n).unwrap() += add;
                    used_tokens += add;
                }
            }
            for n in &steppable {
                if used_tokens >= self.cfg.step_token_budget {
                    break;
                }
                if !chosen.contains_key(n) && self.is_prefilling(&self.lanes[n]) {
                    let na = self.need_alloc(&self.lanes[n]);
                    if self.mem_used() + used_alloc + na <= self.cfg.mem_budget {
                        let span = max_legal_span(&self.lanes[n], self.cfg.step_token_budget - used_tokens);
                        if span > 0 {
                            chosen.insert(n.clone(), span);
                            used_tokens += span;
                            used_alloc += na;
                        }
                    }
                }
            }
        }

        for w in self.actionable_waiting() {
            if !chosen.contains_key(&w) && self.lanes[&w].wait_ticks >= self.cfg.max_bypass {
                return None;
            }
        }

        Some(
            self.cfg
                .lanes
                .iter()
                .filter_map(|l| chosen.get(l).map(|&t| (l.clone(), t)))
                .collect(),
        )
    }

    pub fn gpu_execute(
        &mut self,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(), SchedulerError> {
        if !self.step_active || self.gpu_done {
            return Err(SchedulerError::Precondition("gpu_execute requires active step with !gpu_done"));
        }
        let bsz = self.cfg.block_size;
        for ln in self.step_lanes.clone() {
            let l = &self.lanes[&ln];
            let bi = (l.pos / bsz) as usize;
            let off = (l.pos % bsz) as usize;
            let cnt = self.step_span[&ln] as usize;
            for opt_id in [&l.full[bi], &l.win[bi]] {
                if let Some(id) = opt_id {
                    let b = self.blocks.get_mut(id).unwrap();
                    if b.present() && !b.sealed {
                        b.length = (off + cnt) as u32;
                        for d in &mut b.data[off..off + cnt] {
                            *d = 1;
                        }
                    }
                }
            }
        }
        self.gpu_done = true;
        self.record_step(
            "GpuExecute",
            BTreeMap::new(),
            CacheSchedulerEvent::GpuExecute,
            event_sink,
            state_sink,
        )
    }

    pub fn finish_step(
        &mut self,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<(), SchedulerError> {
        if !self.step_active || !self.gpu_done || !self.step_io.is_empty() {
            return Err(SchedulerError::Precondition("finish_step requires step_active, gpu_done, and empty step_io"));
        }
        let bsz = self.cfg.block_size;
        let def_pfx = self.cfg.default_prefix().to_owned();
        let def_tgt = self.cfg.default_target();
        let nblk = self.cfg.num_blocks() as usize;

        let surviving: BTreeSet<String> = self
            .step_lanes
            .iter()
            .filter(|l| self.lanes[*l].state == "InStep")
            .cloned()
            .collect();
        let cancelling: BTreeSet<String> = self
            .step_lanes
            .iter()
            .filter(|l| self.lanes[*l].state == "Cancelling")
            .cloned()
            .collect();
        let boundary_lanes: BTreeSet<String> = surviving
            .iter()
            .filter(|l| (self.lanes[*l].pos + self.step_span[*l]) % bsz == 0)
            .cloned()
            .collect();

        self.logical_tick += 1;
        let tick = self.logical_tick;

        for ln in boundary_lanes.union(&cancelling) {
            for kind in [BlockKind::Full, BlockKind::Window] {
                let pid = CacheBlockId::Priv { lane: ln.clone(), kind };
                self.blocks.insert(pid, SchedBlock::empty(bsz));
            }
        }
        for ln in &boundary_lanes {
            let npos = self.lanes[ln].pos + self.step_span[ln];
            let bi = (npos / bsz) - 1;
            let pfx = self.lanes[ln].prefix.clone();
            for kind in [BlockKind::Full, BlockKind::Window] {
                let tid = CacheBlockId::Tree { prefix: pfx.clone(), kind, idx: bi };
                let next_tier = if matches!(self.blocks[&tid].tier, "disk" | "both") {
                    "both"
                } else {
                    "memory"
                };
                let prev_refs = self.blocks[&tid].ref_count;
                self.blocks.insert(
                    tid,
                    SchedBlock {
                        tier: next_tier,
                        sealed: true,
                        idx: bi,
                        length: bsz,
                        data: vec![1; bsz as usize],
                        ref_count: prev_refs,
                        last_used_tick: tick,
                    },
                );
            }
            let full_set: BTreeSet<CacheBlockId> = full_blocks(npos, bsz)
                .into_iter()
                .map(|i| CacheBlockId::Tree { prefix: pfx.clone(), kind: BlockKind::Full, idx: i })
                .collect();
            let win_set: BTreeSet<CacheBlockId> = history_blocks(npos, bsz, self.cfg.window_size)
                .into_iter()
                .map(|i| CacheBlockId::Tree { prefix: pfx.clone(), kind: BlockKind::Window, idx: i })
                .collect();
            self.start_points.insert(
                (pfx, npos),
                SchedStartPoint {
                    kept: true,
                    state_kept: true,
                    full: full_set,
                    win: win_set,
                    last_used_tick: tick,
                },
            );
        }

        let mut single_lane_pos = None;
        if surviving.len() == 1 && cancelling.is_empty() {
            let only = surviving.iter().next().unwrap().clone();
            single_lane_pos = Some((only.clone(), self.lanes[&only].pos + self.step_span[&only]));
        }

        for ln in &cancelling {
            let l = self.lanes.get_mut(ln).unwrap();
            l.state = "Cancelled";
            l.prefix = def_pfx.clone();
            l.target = def_tgt;
            l.pos = 0;
            l.full = vec![None; nblk];
            l.win = vec![None; nblk];
            l.reservation = 0;
            l.admit_budget = 0;
        }
        let wsz = self.cfg.window_size;
        for ln in &surviving {
            let span = self.step_span[ln];
            let l = self.lanes.get_mut(ln).unwrap();
            l.pos += span;
            if boundary_lanes.contains(ln) {
                let bi = ((l.pos / bsz) - 1) as usize;
                l.full[bi] = Some(CacheBlockId::Tree {
                    prefix: l.prefix.clone(),
                    kind: BlockKind::Full,
                    idx: bi as u32,
                });
                l.win[bi] = Some(CacheBlockId::Tree {
                    prefix: l.prefix.clone(),
                    kind: BlockKind::Window,
                    idx: bi as u32,
                });
            }
            let keep: BTreeSet<u32> = history_blocks(l.pos, bsz, wsz).into_iter().collect();
            for (i, slot) in l.win.iter_mut().enumerate() {
                if !keep.contains(&(i as u32)) {
                    *slot = None;
                }
            }
            l.state = "Ready";
        }

        self.step_active = false;
        self.step_lanes.clear();
        for v in self.step_span.values_mut() {
            *v = 0;
        }
        self.step_pins.clear();
        self.gpu_done = false;
        self.sync_ref_counts();

        self.record_step(
            "FinishStep",
            BTreeMap::new(),
            CacheSchedulerEvent::FinishStep { single_lane_pos },
            event_sink,
            state_sink,
        )
    }

    pub fn cancel(
        &mut self,
        lane: &str,
        event_sink: &mut dyn TraceSink,
        state_sink: &mut dyn TraceSink,
    ) -> Result<&'static str, SchedulerError> {
        let cur_state = self
            .lanes
            .get(lane)
            .ok_or(SchedulerError::Precondition("unknown lane"))?
            .state;
        if !matches!(cur_state, "Queued" | "Admitted" | "Restoring" | "Ready" | "InStep") {
            return Err(SchedulerError::Precondition("cancel requires active non-cancelling lane"));
        }
        let next_state = if cur_state == "InStep" {
            self.lanes.get_mut(lane).unwrap().state = "Cancelling";
            "Cancelling"
        } else {
            let def_pfx = self.cfg.default_prefix().to_owned();
            let def_tgt = self.cfg.default_target();
            let nblk = self.cfg.num_blocks() as usize;
            let l = self.lanes.get_mut(lane).unwrap();
            l.state = "Cancelled";
            l.prefix = def_pfx;
            l.target = def_tgt;
            l.pos = 0;
            l.full = vec![None; nblk];
            l.win = vec![None; nblk];
            l.reservation = 0;
            l.admit_budget = 0;
            l.wait_ticks = 0;
            self.queue_seq.retain(|x| x != lane);
            self.sync_ref_counts();
            "Cancelled"
        };
        let mut args = BTreeMap::new();
        args.insert("lane".into(), JsonVal::str(lane));
        args.insert("state".into(), JsonVal::str(next_state));
        self.record_step(
            "Cancel",
            args,
            CacheSchedulerEvent::Cancel {
                lane: lane.into(),
                state: next_state.into(),
            },
            event_sink,
            state_sink,
        )?;
        Ok(next_state)
    }

    pub fn snapshot_state(&self) -> CacheSchedulerState {
        let blocks = self
            .blocks
            .iter()
            .filter(|(_, b)| b.present())
            .map(|(id, b)| CacheBlockSnapshot {
                id: id.clone(),
                tier: b.tier.into(),
                sealed: b.sealed,
                idx: b.idx,
                length: b.length,
                data: b.data.clone(),
                ref_count: b.ref_count,
            })
            .collect();
        let start_points = self
            .start_points
            .iter()
            .filter(|(_, sp)| sp.kept)
            .map(|((pfx, bnd), sp)| StartPointSnapshot {
                prefix: pfx.clone(),
                boundary: *bnd,
                state_kept: sp.state_kept,
                full: sp.full.iter().cloned().collect(),
                win: sp.win.iter().cloned().collect(),
            })
            .collect();
        let lanes = self
            .cfg
            .lanes
            .iter()
            .map(|ln| {
                let l = &self.lanes[ln];
                CacheLaneSnapshot {
                    lane: ln.clone(),
                    state: l.state.into(),
                    prefix: l.prefix.clone(),
                    target: l.target,
                    pos: l.pos,
                    full: l.full.clone(),
                    win: l.win.clone(),
                    reservation: l.reservation,
                    admit_budget: l.admit_budget,
                    wait_ticks: l.wait_ticks,
                }
            })
            .collect();
        CacheSchedulerState {
            blocks,
            start_points,
            lanes,
            queue_seq: self.queue_seq.clone(),
            step_active: self.step_active,
            step_lanes: self.step_lanes.iter().cloned().collect(),
            step_span: self.step_span.clone(),
            step_pins: self.step_pins.iter().cloned().collect(),
            step_io: self.step_io.iter().cloned().collect(),
            gpu_done: self.gpu_done,
        }
    }

    pub fn assert_invariants(&self) -> Result<(), SchedulerError> {
        if self.mem_used() > self.cfg.mem_budget
            || self.disk_used() > self.cfg.disk_budget
            || self.reserved_total() > self.cfg.mem_budget
            || self.admitted_total() > self.cfg.mem_budget
        {
            return Err(SchedulerError::InvariantViolation("BudgetsAndReservations exceeded".into()));
        }
        let q_cnt = self.lanes.values().filter(|l| l.state == "Queued").count() as u32;
        if q_cnt > self.cfg.max_queue {
            return Err(SchedulerError::InvariantViolation("QueueBounded violated".into()));
        }
        for l in self.lanes.values() {
            if matches!(l.state, "Queued" | "Admitted" | "Restoring" | "Ready")
                && l.wait_ticks > self.cfg.max_bypass
            {
                return Err(SchedulerError::InvariantViolation("BypassBoundHolds violated".into()));
            }
        }
        for (id, b) in &self.blocks {
            if !b.present() {
                continue;
            }
            let expected_refs = (self.lane_referrers(id).len() as u32) + self.start_referrers_count(id);
            if b.ref_count != expected_refs {
                return Err(SchedulerError::InvariantViolation(format!(
                    "RefCountMatchesReferrers mismatch on {}: got {}, want {}",
                    id.encode(),
                    b.ref_count,
                    expected_refs
                )));
            }
            if b.sealed {
                if matches!(id, CacheBlockId::Priv { .. })
                    || b.length != self.cfg.block_size
                    || b.data.iter().any(|&d| d != 1)
                {
                    return Err(SchedulerError::InvariantViolation("SealedValid violated".into()));
                }
            } else {
                if matches!(id, CacheBlockId::Tree { .. })
                    || b.tier != "memory"
                    || b.ref_count != 1
                    || self.start_referrers_count(id) != 0
                    || b.data[b.length as usize..].iter().any(|&d| d != 0)
                {
                    return Err(SchedulerError::InvariantViolation(
                        "UnsealedPrivateInvisibleZero violated".into(),
                    ));
                }
            }
        }
        if self.step_active {
            let tok_sum: u32 = self.step_lanes.iter().map(|l| self.step_span[l]).sum();
            if tok_sum > self.cfg.step_token_budget {
                return Err(SchedulerError::InvariantViolation("StepBudgetHolds violated".into()));
            }
            for id in &self.step_pins {
                if !self.blocks[id].in_memory() {
                    return Err(SchedulerError::InvariantViolation(
                        "StepBlocksInMemoryAndOwned violated".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

// ---- Deterministic Seeded Trace Walks (for Unit Tests & CI Trace Gate) ----

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    fn pick(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 32) as usize) % n
    }
}

/// Execute a deterministic seeded `BlockLayout` walk exercising:
///   - even seeds: start-point keep/match/drop, simultaneous live slots `1` and `2`, and window sliding;
///   - odd seeds: finish + `SlotRecord::reissue` slot reuse (`generation == 2`), generation exhaustion
///     advancing allocation to slot `2`, partial unsealed tail writes, and window sliding.
pub fn walk_block_layout(
    cfg: BlockLayoutConfig,
    seed: u64,
    event_sink: &mut dyn TraceSink,
    state_sink: &mut dyn TraceSink,
) -> Result<(), SchedulerError> {
    let mut rng = Lcg::new(seed);
    let mut alloc = TrackedBlockAllocator::new(cfg.clone(), event_sink, state_sink)?;
    let lanes = cfg.lanes.clone();
    let l0 = &lanes[0];

    let step_lane = |alloc: &mut TrackedBlockAllocator,
                     lane: &str,
                     span: u32,
                     ev: &mut dyn TraceSink,
                     st: &mut dyn TraceSink|
     -> Result<bool, SchedulerError> {
        alloc.prepare(lane, span, ev, st)?;
        for (k, b) in alloc.missing_needed_keys(lane)? {
            if alloc.first_available_slot(k).is_none() {
                return Ok(false);
            }
            alloc.alloc_lowest_free(lane, k, b, ev, st)?;
        }
        alloc.begin_step(lane, ev, st)?;
        while alloc.lanes[lane].pos < alloc.lanes[lane].target {
            alloc.write_one(lane, ev, st)?;
        }
        alloc.end_step(lane, ev, st)?;
        Ok(true)
    };

    if lanes.len() >= 2 && seed % 2 == 1 {
        // Odd seeds: l0 writes block 0 (partial 1-token steps when rng chooses) then finishes;
        // l1 reuses slot 1 at generation 2, then advances into block 1 (taking slot 2 because
        // slot 1 is held / exhausted at MaxGeneration).
        let first_span = if rng.pick(2) == 0 { 1 } else { cfg.block_size.min(cfg.max_span) };
        step_lane(&mut alloc, l0, first_span, event_sink, state_sink)?;
        if alloc.lanes[l0].pos < cfg.block_size {
            let rem = cfg.block_size - alloc.lanes[l0].pos;
            step_lane(&mut alloc, l0, rem.min(cfg.max_span), event_sink, state_sink)?;
        }
        alloc.finish(l0, event_sink, state_sink)?;

        let l1 = &lanes[1];
        while alloc.lanes[l1].pos < cfg.max_pos {
            let _ = alloc.slide_window(l1, event_sink, state_sink)?;
            let rem = cfg.max_pos - alloc.lanes[l1].pos;
            let span = if rng.pick(2) == 0 { 1 } else { cfg.max_span.min(rem) };
            if !step_lane(&mut alloc, l1, span, event_sink, state_sink)? {
                break;
            }
            let _ = alloc.slide_window(l1, event_sink, state_sink)?;
        }
        if alloc.lanes[l1].phase == "idle" {
            alloc.finish(l1, event_sink, state_sink)?;
        }
    } else {
        // Even seeds: l0 reaches boundary, keeps start point, l1 matches, drops start, slides window, advances.
        step_lane(&mut alloc, l0, cfg.block_size.min(cfg.max_span), event_sink, state_sink)?;
        if alloc.lanes[l0].pos % cfg.block_size == 0 && alloc.lanes[l0].pos > 0 {
            alloc.keep_start(l0, event_sink, state_sink)?;
        }
        if lanes.len() >= 2 && alloc.start_kept {
            let l1 = &lanes[1];
            alloc.match_start(l1, event_sink, state_sink)?;
            alloc.drop_start(event_sink, state_sink)?;
            alloc.finish(l0, event_sink, state_sink)?;
            while alloc.lanes[l1].pos < cfg.max_pos {
                let _ = alloc.slide_window(l1, event_sink, state_sink)?;
                let rem = cfg.max_pos - alloc.lanes[l1].pos;
                let span = if rng.pick(2) == 0 { 1 } else { cfg.max_span.min(rem) };
                if !step_lane(&mut alloc, l1, span, event_sink, state_sink)? {
                    break;
                }
                let _ = alloc.slide_window(l1, event_sink, state_sink)?;
            }
            if alloc.lanes[l1].phase == "idle" {
                alloc.finish(l1, event_sink, state_sink)?;
            }
        } else {
            alloc.finish(l0, event_sink, state_sink)?;
        }
    }
    Ok(())
}

/// Execute a deterministic seeded `CacheScheduler` walk exercising:
///   - `MaxQueue >= 2` admit bypass (fitting smaller request bypassing larger queued request and incrementing its `wait_ticks`),
///   - `BuildStep` `StepBypassed` `wait_ticks` increment on unchosen actionable lanes,
///   - HTTP 429 refusals, start-point matching, cold SSD spills & restores, mid-step `Cancelling` fence, and completions.
pub fn walk_cache_scheduler(
    cfg: CacheSchedulerConfig,
    seed: u64,
    event_sink: &mut dyn TraceSink,
    state_sink: &mut dyn TraceSink,
) -> Result<(), SchedulerError> {
    let mut rng = Lcg::new(seed);
    let mut sched = CacheScheduler::new(cfg.clone(), event_sink, state_sink)?;
    let lanes = cfg.lanes.clone();
    let prefixes = cfg.prefixes.clone();
    let req_lens = cfg.request_lengths.clone();

    let min_len = *req_lens.iter().min().unwrap();
    let max_len = *req_lens.iter().max().unwrap();

    // Opening scenario when `max_queue >= 2` and `lanes.len() >= 3` and `max_len > min_len`:
    // deterministically exercise BOTH `admit` queue-bypass (`wait_ticks += 1`) AND `build_step` `StepBypassed` (`wait_ticks += 1`).
    if seed % 2 == 0
        && cfg.max_queue >= 2
        && lanes.len() >= 3
        && sched.worst_case_for(min_len) * 2 <= cfg.mem_budget
        && sched.worst_case_for(min_len) + sched.worst_case_for(max_len) > cfg.mem_budget
    {
        let p0 = &prefixes[0];
        let p1 = prefixes.get(1).unwrap_or(p0);
        // l0 admitted with min_len -> holds half the admission budget:
        sched.arrive(&lanes[0], p0, min_len, event_sink, state_sink)?;
        sched.admit(&lanes[0], event_sink, state_sink)?;
        sched.begin_restore(&lanes[0], event_sink, state_sink)?;
        sched.finish_restore(&lanes[0], event_sink, state_sink)?;
        // l1 queued with max_len (cannot fit while l0 is active) + l2 queued with min_len (CAN fit!):
        sched.arrive(&lanes[1], p1, max_len, event_sink, state_sink)?;
        sched.arrive(&lanes[2], p0, min_len, event_sink, state_sink)?;
        if lanes.len() >= 4 && cfg.max_queue == 2 {
            // Queue full + max_len cannot direct-admit -> 429 Refused!
            let _ = sched.arrive(&lanes[3], p1, max_len, event_sink, state_sink)?;
        }
        // Admit l2: bypasses queued l1 -> increments l1.wait_ticks!
        sched.admit(&lanes[2], event_sink, state_sink)?;
        sched.begin_restore(&lanes[2], event_sink, state_sink)?;
        sched.finish_restore(&lanes[2], event_sink, state_sink)?;
        // Step ONLY l0 (when step_token_budget == 1 or l2 is prefilling; with decode priority, if l2 is also decoding and fits step_token_budget >= 2, make l2 prefilling or step both then bypass queued l1!):
        // Wait: in BuildStep, FitUnchosenDecoders requires SumBy(chosen, spans) == StepTokenBudget!
        // So if l0 has max_len (prefilling, span = 2 == step_token_budget!) or if l2 is prefilling:
        if let Some(plan) = sched.plan_step() {
            sched.build_step(&plan, event_sink, state_sink)?;
            sched.gpu_execute(event_sink, state_sink)?;
            sched.finish_step(event_sink, state_sink)?;
        }
    }

    for round in 0..24 {
        if sched.step_active {
            if round % 5 == 2 {
                if let Some(in_step_lane) = sched
                    .step_lanes
                    .iter()
                    .find(|l| sched.lane_state(l) == Some("InStep"))
                    .cloned()
                {
                    if rng.pick(2) == 0 {
                        let _ = sched.cancel(&in_step_lane, event_sink, state_sink)?;
                    }
                }
            }
            if !sched.gpu_done {
                sched.gpu_execute(event_sink, state_sink)?;
            }
            if !sched.step_io.is_empty() {
                sched.complete_step_io(event_sink, state_sink)?;
            }
            sched.finish_step(event_sink, state_sink)?;
        }

        for ln in &lanes {
            if sched.lane_state(ln) == Some("Ready") && sched.lanes[ln].pos == sched.lanes[ln].target {
                sched.finish_ready(ln, event_sink, state_sink)?;
            }
        }

        let burst = if round == 0 { lanes.len() } else { 1 + rng.pick(2) };
        let mut submitted = 0;
        for ln in &lanes {
            if submitted >= burst {
                break;
            }
            if matches!(
                sched.lane_state(ln),
                Some("Idle" | "Refused" | "Cancelled" | "Finished")
            ) {
                let pfx = &prefixes[rng.pick(prefixes.len())];
                let tgt = req_lens[rng.pick(req_lens.len())];
                let _ = sched.arrive(ln, pfx, tgt, event_sink, state_sink)?;
                submitted += 1;
            }
        }

        loop {
            let next_q = lanes
                .iter()
                .filter(|l| sched.can_admit_now(l))
                .max_by_key(|l| sched.lanes[*l].wait_ticks)
                .cloned();
            let Some(ql) = next_q else { break };
            sched.admit(&ql, event_sink, state_sink)?;
        }

        for ln in &lanes {
            if sched.lane_state(ln) == Some("Admitted")
                && sched.reserved_total() + sched.worst_case_for(sched.lanes[ln].target) <= cfg.mem_budget
            {
                let _ = sched.begin_restore(ln, event_sink, state_sink)?;
            }
        }

        for _ in 0..4 {
            if !sched.step_io.is_empty() {
                sched.complete_step_io(event_sink, state_sink)?;
            }
            if !sched.mem_pressure() {
                break;
            }
            if sched.evict_clean_or_unref_mem(event_sink, state_sink).is_ok() {
                continue;
            }
            if sched.disk_used() >= cfg.disk_budget {
                let _ = sched.reclaim_cold_disk(event_sink, state_sink);
            }
            if sched.start_cold_writeback(event_sink, state_sink).is_ok() {
                sched.complete_step_io(event_sink, state_sink)?;
                let _ = sched.evict_clean_or_unref_mem(event_sink, state_sink);
            } else {
                break;
            }
        }

        for ln in &lanes {
            if sched.lane_state(ln) == Some("Restoring") {
                let disk_blks: Vec<CacheBlockId> = CacheScheduler::lane_blocks_of(&sched.lanes[ln])
                    .into_iter()
                    .filter(|b| sched.blocks[b].tier == "disk")
                    .collect();
                for b in disk_blks {
                    if sched.mem_used() < cfg.mem_budget {
                        let _ = sched.restore_block(ln, &b, event_sink, state_sink);
                    }
                }
                if CacheScheduler::lane_blocks_of(&sched.lanes[ln])
                    .iter()
                    .all(|b| sched.blocks[b].in_memory())
                {
                    sched.finish_restore(ln, event_sink, state_sink)?;
                }
            }
        }

        for ln in &lanes {
            if sched.lane_state(ln) == Some("Ready") && sched.lanes[ln].pos == sched.lanes[ln].target {
                sched.finish_ready(ln, event_sink, state_sink)?;
            }
        }

        if let Some(plan) = sched.plan_step() {
            sched.build_step(&plan, event_sink, state_sink)?;
        }
    }

    if sched.step_active {
        if !sched.gpu_done {
            sched.gpu_execute(event_sink, state_sink)?;
        }
        if !sched.step_io.is_empty() {
            sched.complete_step_io(event_sink, state_sink)?;
        }
        sched.finish_step(event_sink, state_sink)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::{FULL_STRIDE, WINDOW_STRIDE};
    use crate::cache_trace::{
        parse_block_layout_jsonl, parse_cache_scheduler_jsonl, JsonlSink, MemorySink, NullSink,
    };

    #[test]
    fn working_set_budget_handles_dynamic_kvb_and_refuses_zero_or_overflow() {
        let g = 1u64 << 30;
        let fp16 = WorkingSetBudget::from_recommended_working_set_bytes(64 * g, 20 * g, 2 * g, 2).unwrap();
        assert_eq!(fp16.cache_budget_bytes, 42 * g);
        assert_eq!(fp16.full_stride_bytes, FULL_STRIDE);
        assert_eq!(fp16.window_stride_bytes, WINDOW_STRIDE);
        assert_eq!(
            fp16.worst_case_request_bytes(4096, 512).unwrap(),
            64 * FULL_STRIDE + 25 * WINDOW_STRIDE
        );

        // --kv8 (`kvb = 1`) uses half the stride per block:
        let kv8 = WorkingSetBudget::from_recommended_working_set_bytes(64 * g, 20 * g, 2 * g, 1).unwrap();
        assert_eq!(kv8.full_stride_bytes, FULL_STRIDE / 2);
        assert_eq!(kv8.window_stride_bytes, WINDOW_STRIDE / 2);
        assert_eq!(
            kv8.worst_case_request_bytes(4096, 512).unwrap(),
            64 * (FULL_STRIDE / 2) + 25 * (WINDOW_STRIDE / 2)
        );
        assert!(kv8.partition_pools(256).unwrap().0 > fp16.partition_pools(256).unwrap().0);

        // Invalid kvb, underflow, overflow, and zero inputs are all refused cleanly (never panic or wrap):
        assert!(WorkingSetBudget::from_recommended_working_set_bytes(64 * g, 0, 0, 0).is_err());
        assert!(WorkingSetBudget::from_recommended_working_set_bytes(64 * g, 0, 0, 3).is_err());
        assert!(WorkingSetBudget::from_recommended_working_set_bytes(u64::MAX, u64::MAX, 1, 2).is_err());
        assert!(WorkingSetBudget::from_recommended_working_set_bytes(100, 200, 0, 2).is_err());
        assert!(WorkingSetBudget::from_recommended_working_set_bytes(12 * 1024 * 1024 - 1, 0, 0, 2).is_err());
        assert!(WorkingSetBudget::from_recommended_working_set_bytes(6 * 1024 * 1024, 0, 0, 1).is_ok());

        assert!(fp16.worst_case_request_bytes(0, 512).is_err());
        assert!(fp16.worst_case_request_bytes(4096, 0).is_err());
        assert!(fp16.worst_case_request_bytes(1 << 20, u32::MAX).is_err());
        assert!(fp16.partition_pools(0).is_err());
        assert!(span_bound_blocks(0, 64, 1024).is_err());
        assert!(span_bound_blocks(1, 0, 1024).is_err());
        assert!(span_bound_blocks(1, 64, 0).is_err());
        assert!(span_bound_blocks(u32::MAX, 64, 1024).is_err());
    }

    #[test]
    fn block_allocator_reissues_generations_exhausts_max_gen_and_uses_real_byte_offset() {
        for (kvb, slot_base, want_hi) in [(2u32, 0u32, 0u32), (2u32, 2048u32, 1u32), (1u32, 4096u32, 1u32)] {
            let cfg = BlockLayoutConfig {
                lanes: vec!["a".into(), "b".into()],
                full_capacity: 2,
                window_capacity: 2,
                block_size: 2,
                window_size: 2,
                max_pos: 4,
                max_span: 2,
                max_generation: 2,
                slot_base,
                kvb,
            };

            // Explicit reuse & MaxGeneration exhaustion verification:
            let mut ev = NullSink;
            let mut st = MemorySink::new();
            let mut alloc = TrackedBlockAllocator::new(cfg.clone(), &mut ev, &mut st).unwrap();
            alloc.prepare("a", 2, &mut ev, &mut st).unwrap();
            let e_f1 = alloc.alloc_lowest_free("a", BlockKind::Full, 0, &mut ev, &mut st).unwrap();
            let e_w1 = alloc.alloc_lowest_free("a", BlockKind::Window, 0, &mut ev, &mut st).unwrap();
            assert_eq!((e_f1.slot().get(), e_f1.generation()), (1, 1));
            assert_eq!((e_w1.slot().get(), e_w1.generation()), (1, 1));
            assert_eq!(
                e_f1.offset(),
                byte_offset(BlockKind::Full, Slot::new(slot_base + 1).unwrap(), kvb).unwrap()
            );
            assert_eq!(OffsetLimbs::from_u64(e_f1.offset()).unwrap().hi, want_hi);
            alloc.begin_step("a", &mut ev, &mut st).unwrap();
            alloc.write_one("a", &mut ev, &mut st).unwrap();
            alloc.write_one("a", &mut ev, &mut st).unwrap();
            alloc.end_step("a", &mut ev, &mut st).unwrap();
            alloc.finish("a", &mut ev, &mut st).unwrap();

            // Lane "b" reuses slot 1 at generation 2 (via SlotRecord::reissue):
            alloc.prepare("b", 1, &mut ev, &mut st).unwrap();
            let e_f2 = alloc.alloc_lowest_free("b", BlockKind::Full, 0, &mut ev, &mut st).unwrap();
            let e_w2 = alloc.alloc_lowest_free("b", BlockKind::Window, 0, &mut ev, &mut st).unwrap();
            assert_eq!((e_f2.slot().get(), e_f2.generation()), (1, 2));
            assert_eq!((e_w2.slot().get(), e_w2.generation()), (1, 2));
            alloc.begin_step("b", &mut ev, &mut st).unwrap();
            alloc.write_one("b", &mut ev, &mut st).unwrap();
            alloc.end_step("b", &mut ev, &mut st).unwrap();
            alloc.finish("b", &mut ev, &mut st).unwrap();

            // Slot 1 is now at MaxGeneration (2) -> next available slot MUST be slot 2!
            assert_eq!(alloc.first_available_slot(BlockKind::Full), Some(Slot::new(2).unwrap()));
            assert_eq!(alloc.first_available_slot(BlockKind::Window), Some(Slot::new(2).unwrap()));

            // Seeded walks must exercise generation 2, start points, and window sliding:
            let mut max_gen_seen = 0;
            let mut saw_start = false;
            let mut saw_two_slots = false;
            for seed in 0..16 {
                let mut ev_s = MemorySink::new();
                let mut st_s = MemorySink::new();
                walk_block_layout(cfg.clone(), seed, &mut ev_s, &mut st_s).unwrap();
                let parsed = parse_block_layout_jsonl(&st_s.to_jsonl()).unwrap();
                for step in &parsed.steps {
                    if step.state.start_kept {
                        saw_start = true;
                    }
                    for p in step.state.full_pool.iter().chain(&step.state.window_pool) {
                        max_gen_seen = max_gen_seen.max(p.generation);
                        if p.slot == 2 && p.generation >= 1 {
                            saw_two_slots = true;
                        }
                    }
                }
            }
            assert_eq!(max_gen_seen, 2);
            assert!(saw_start);
            assert!(saw_two_slots);
        }
    }

    #[test]
    fn cache_scheduler_admit_bypass_step_bypass_and_cold_spill_restore_hold() {
        let cfg = CacheSchedulerConfig {
            lanes: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            prefixes: vec!["p1".into(), "p2".into()],
            request_lengths: vec![2, 4],
            block_size: 2,
            window_size: 2,
            max_pos: 4,
            max_span: 2,
            step_token_budget: 2,
            mem_budget: 4,
            disk_budget: 3,
            max_queue: 2,
            max_bypass: 2,
        };
        let mut saw_admit_bypass = false;
        let mut saw_step_bypass = false;
        let mut saw_cancel = false;
        let mut saw_refuse = false;
        for seed in 0..16 {
            let mut ev_sink = MemorySink::new();
            let mut st_sink = MemorySink::new();
            walk_cache_scheduler(cfg.clone(), seed, &mut ev_sink, &mut st_sink).unwrap();
            let parsed = parse_cache_scheduler_jsonl(&st_sink.to_jsonl()).unwrap();
            for w in parsed.steps.windows(2) {
                if w[1].action == "Admit" {
                    for (l0, l1) in w[0].state.lanes.iter().zip(&w[1].state.lanes) {
                        if l0.state == "Queued" && l1.state == "Queued" && l1.wait_ticks > l0.wait_ticks {
                            saw_admit_bypass = true;
                        }
                    }
                }
                if w[1].action == "BuildStep" {
                    for (l0, l1) in w[0].state.lanes.iter().zip(&w[1].state.lanes) {
                        if l1.wait_ticks > l0.wait_ticks {
                            saw_step_bypass = true;
                        }
                    }
                }
            }
            for line in ev_sink.lines() {
                if line.contains("\"event\":\"cancel\"") {
                    saw_cancel = true;
                }
                if line.contains("\"event\":\"refuse429\"") {
                    saw_refuse = true;
                }
            }
        }
        assert!(saw_admit_bypass, "walks must exercise admit queue bypass at MaxQueue=2");
        assert!(saw_step_bypass, "walks must exercise BuildStep StepBypassed wait_ticks increment");
        assert!(saw_cancel, "walks must exercise mid-step cancellation");
        assert!(saw_refuse, "walks must exercise HTTP 429 refusal");

        // Explicit cold SSD spill + atomic restore scenario:
        let mut ev_sink = MemorySink::new();
        let mut st_sink = MemorySink::new();
        let mut sched = CacheScheduler::new(cfg, &mut ev_sink, &mut st_sink).unwrap();
        sched.arrive("a", "p1", 2, &mut ev_sink, &mut st_sink).unwrap();
        sched.admit("a", &mut ev_sink, &mut st_sink).unwrap();
        sched.begin_restore("a", &mut ev_sink, &mut st_sink).unwrap();
        sched.finish_restore("a", &mut ev_sink, &mut st_sink).unwrap();
        sched.build_step(&[("a".into(), 1)], &mut ev_sink, &mut st_sink).unwrap();
        sched.gpu_execute(&mut ev_sink, &mut st_sink).unwrap();
        sched.finish_step(&mut ev_sink, &mut st_sink).unwrap();
        sched.build_step(&[("a".into(), 1)], &mut ev_sink, &mut st_sink).unwrap();
        sched.gpu_execute(&mut ev_sink, &mut st_sink).unwrap();
        sched.finish_step(&mut ev_sink, &mut st_sink).unwrap();
        sched.finish_ready("a", &mut ev_sink, &mut st_sink).unwrap();

        sched.arrive("a", "p2", 2, &mut ev_sink, &mut st_sink).unwrap();
        sched.admit("a", &mut ev_sink, &mut st_sink).unwrap();
        sched.begin_restore("a", &mut ev_sink, &mut st_sink).unwrap();
        sched.finish_restore("a", &mut ev_sink, &mut st_sink).unwrap();
        sched.arrive("b", "p2", 2, &mut ev_sink, &mut st_sink).unwrap();
        sched.admit("b", &mut ev_sink, &mut st_sink).unwrap();
        sched.begin_restore("b", &mut ev_sink, &mut st_sink).unwrap();
        sched.finish_restore("b", &mut ev_sink, &mut st_sink).unwrap();
        assert!(sched.mem_pressure());

        let spilled1 = sched.start_cold_writeback(&mut ev_sink, &mut st_sink).unwrap();
        sched.complete_step_io(&mut ev_sink, &mut st_sink).unwrap();
        let spilled2 = sched.start_cold_writeback(&mut ev_sink, &mut st_sink).unwrap();
        sched.complete_step_io(&mut ev_sink, &mut st_sink).unwrap();
        assert_eq!(sched.disk_used(), 2);
        assert_eq!(sched.mem_used(), 0);

        sched.cancel("a", &mut ev_sink, &mut st_sink).unwrap();
        sched.cancel("b", &mut ev_sink, &mut st_sink).unwrap();
        sched.arrive("c", "p1", 2, &mut ev_sink, &mut st_sink).unwrap();
        sched.admit("c", &mut ev_sink, &mut st_sink).unwrap();
        let (matched_pos, reserved) = sched.begin_restore("c", &mut ev_sink, &mut st_sink).unwrap();
        assert_eq!((matched_pos, reserved), (2, 2));
        sched.restore_block("c", &spilled1, &mut ev_sink, &mut st_sink).unwrap();
        sched.restore_block("c", &spilled2, &mut ev_sink, &mut st_sink).unwrap();
        sched.finish_restore("c", &mut ev_sink, &mut st_sink).unwrap();
        sched.finish_ready("c", &mut ev_sink, &mut st_sink).unwrap();
    }

    #[test]
    #[ignore = "writes versioned BlockLayout NDJSON trace to INKY_BLOCK_TRACE_OUT"]
    fn dump_block_layout() {
        let path = std::env::var_os("INKY_BLOCK_TRACE_OUT").expect("set INKY_BLOCK_TRACE_OUT");
        let seed: u64 = std::env::var("INKY_BLOCK_TRACE_SEED")
            .map(|s| s.parse().expect("INKY_BLOCK_TRACE_SEED must be u64"))
            .unwrap_or(1);
        let slot_base: u32 = std::env::var("INKY_BLOCK_TRACE_SLOT_BASE")
            .map(|s| s.parse().expect("INKY_BLOCK_TRACE_SLOT_BASE must be u32"))
            .unwrap_or(0);
        let kvb: u32 = std::env::var("INKY_BLOCK_TRACE_KVB")
            .map(|s| s.parse().expect("INKY_BLOCK_TRACE_KVB must be 1 or 2"))
            .unwrap_or(2);
        let mode = std::env::var("INKY_BLOCK_TRACE_MODE").unwrap_or_else(|_| "events".into());
        let cfg = BlockLayoutConfig {
            lanes: vec!["a".into(), "b".into()],
            full_capacity: 2,
            window_capacity: 2,
            block_size: 2,
            window_size: 2,
            max_pos: 4,
            max_span: 2,
            max_generation: 2,
            slot_base,
            kvb,
        };
        let file = std::fs::File::create(path).unwrap();
        let mut writer = JsonlSink::new(std::io::BufWriter::new(file));
        let mut null = NullSink;
        if mode == "steps" {
            walk_block_layout(cfg, seed, &mut null, &mut writer).unwrap();
        } else {
            walk_block_layout(cfg, seed, &mut writer, &mut null).unwrap();
        }
        writer.flush().unwrap();
    }

    #[test]
    #[ignore = "writes versioned CacheScheduler NDJSON trace to INKY_CACHE_TRACE_OUT"]
    fn dump_cache_scheduler() {
        let path = std::env::var_os("INKY_CACHE_TRACE_OUT").expect("set INKY_CACHE_TRACE_OUT");
        let seed: u64 = std::env::var("INKY_CACHE_TRACE_SEED")
            .map(|s| s.parse().expect("INKY_CACHE_TRACE_SEED must be u64"))
            .unwrap_or(1);
        let mode = std::env::var("INKY_CACHE_TRACE_MODE").unwrap_or_else(|_| "events".into());
        let cfg = CacheSchedulerConfig {
            lanes: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            prefixes: vec!["p1".into(), "p2".into()],
            request_lengths: vec![2, 4],
            block_size: 2,
            window_size: 2,
            max_pos: 4,
            max_span: 2,
            step_token_budget: 2,
            mem_budget: 4,
            disk_budget: 3,
            max_queue: 2,
            max_bypass: 2,
        };
        let file = std::fs::File::create(path).unwrap();
        let mut writer = JsonlSink::new(std::io::BufWriter::new(file));
        let mut null = NullSink;
        if mode == "steps" {
            walk_cache_scheduler(cfg, seed, &mut null, &mut writer).unwrap();
        } else {
            walk_cache_scheduler(cfg, seed, &mut writer, &mut null).unwrap();
        }
        writer.flush().unwrap();
    }
}
