//! Layer 3 block cache, the types the rest of the layer builds on.
#![allow(dead_code)] // the pool, the table and the prefix tree are the callers; cargo check does not compile the unit tests
//!
//! The GPU buffers are not allocated here. A key read goes through the per-lane
//! table (`BlockTable::read_key`): the `u64` stored on the entry is the slot base
//! from `byte_offset`, and `key_row` adds the pass, K or V, head and position.
//! It is not a position in the layer-2 contiguous cache, and it is not
//! `base + index * (stride / 64)`.
//! A block is 64 positions. Inside it the order is `[pass][K|V][head][pos][HD]`,
//! one element `kvb` bytes (DESIGN.md). At `kvb = 2` a window block is 10 MiB and
//! a full block is 2 MiB; at `kvb = 1` (`--kv8`) each is half of that.
//! Each kind has its own pool, so an offset is a plain `u64` byte offset in that
//! pool: base 0 plus the 0-based pool index times that kind's stride at this `kvb`.
//! `BlockLayout`'s high limb of 2 is TLC's trap for a 32-bit truncation, not a base
//! this code uses. The in-block order is `InBlock.tla`.
//!
//! A generation does not wrap. The next generation after `u32::MAX` is refused.

use std::fmt;

/// Positions in one block.
pub const BLOCK: usize = 64;
/// Sliding-window positions a start point keeps.
pub const WINDOW: usize = 1024;
/// Bytes in one full-attention block at fp16 (`kvb = 2`). Eight passes, 2 MiB.
pub const FULL_STRIDE: u64 = 2 * 1024 * 1024;
/// Bytes in one sliding-window block at fp16 (`kvb = 2`). Forty passes, 10 MiB.
pub const WINDOW_STRIDE: u64 = 10 * 1024 * 1024;
/// KV heads. The same count as `engine::NKV`.
pub const NKV: usize = 8;
/// Elements in one key or value. The same count as `engine::HD`.
pub const HD: usize = 128;
/// Sliding-window layer passes stored in one window block.
pub const WINDOW_PASSES: usize = 40;
/// Full-attention layer passes stored in one full block.
pub const FULL_PASSES: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlockKind { Window, Full }

impl BlockKind {
    pub fn passes(self) -> usize {
        match self {
            BlockKind::Window => WINDOW_PASSES,
            BlockKind::Full => FULL_PASSES,
        }
    }

    /// Bytes in one slot at this element width. `kvb` is 2 for fp16 and 1 for `--kv8`.
    /// Any other width is refused. At 2 this is `FULL_STRIDE` or `WINDOW_STRIDE`.
    pub fn stride_bytes(self, kvb: u32) -> Result<u64, BlockError> {
        if kvb != 1 && kvb != 2 { return Err(BlockError::Width); }
        let passes = self.passes() as u64;
        passes.checked_mul(2)
            .and_then(|v| v.checked_mul(NKV as u64))
            .and_then(|v| v.checked_mul(BLOCK as u64))
            .and_then(|v| v.checked_mul(HD as u64))
            .and_then(|v| v.checked_mul(u64::from(kvb)))
            .ok_or(BlockError::Overflow)
    }
}

/// Which plane of a pass. Key is first, Value is second, matching layer 2's `kv_slot`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kv { Key, Value }

impl Kv {
    fn index(self) -> u32 {
        match self { Kv::Key => 0, Kv::Value => 1 }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlockError {
    /// A slot number of 0. Slots are 1-based.
    SlotZero,
    /// A checked add or multiply did not fit.
    Overflow,
    /// The generation would wrap past `u32::MAX`.
    Generation,
    /// A length past the 64 positions in a block.
    Length,
    /// The block index is outside the table this pool declared.
    OutOfRange,
    /// A reference count that is already zero cannot be dropped.
    Refs,
    /// This kind's pool has no slots. A key cannot be read from it.
    NoSlots,
    /// The block is inside the table and no entry was filled.
    NoEntry,
    /// The entry's generation is not the pool record's generation.
    Stale,
    /// Generation 0, on the argument or on the record. The slot has never been taken.
    NotTaken,
    /// `kvb` is 0, or a read asked for a different element width than the pool stores.
    Width,
}

impl fmt::Display for BlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlockError::SlotZero => write!(f, "block slot 0 is not a slot"),
            BlockError::Overflow => write!(f, "block offset overflow"),
            BlockError::Generation => write!(f, "block generation would wrap"),
            BlockError::Length => write!(f, "block length past 64"),
            BlockError::OutOfRange => write!(f, "block index outside the table"),
            BlockError::Refs => write!(f, "block reference count is already zero"),
            BlockError::NoSlots => write!(f, "pool has no slots"),
            BlockError::NoEntry => write!(f, "block table has no entry"),
            BlockError::Stale => write!(f, "block generation does not match the pool"),
            BlockError::NotTaken => write!(f, "block slot has not been taken"),
            BlockError::Width => write!(f, "block element width does not match"),
        }
    }
}

/// A 1-based slot in one kind's pool. Not a block index and not a pool index.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Slot(u32);

impl Slot {
    pub fn new(n: u32) -> Result<Slot, BlockError> {
        if n == 0 { Err(BlockError::SlotZero) } else { Ok(Slot(n)) }
    }
    pub fn get(self) -> u32 { self.0 }
    /// The 0-based index of this slot in its pool.
    pub fn pool_index(self) -> PoolIndex { PoolIndex(self.0 - 1) }
}

/// A 0-based index into one kind's pool. Not a slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PoolIndex(u32);

impl PoolIndex {
    pub fn get(self) -> u32 { self.0 }
}

/// Which 64-position block of the token stream. Not a slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlockIndex(u32);

impl BlockIndex {
    pub fn new(n: u32) -> BlockIndex { BlockIndex(n) }
    pub fn get(self) -> u32 { self.0 }
    pub fn of_pos(pos: usize) -> Result<BlockIndex, BlockError> {
        u32::try_from(pos / BLOCK).map(BlockIndex).map_err(|_| BlockError::Overflow)
    }
}

/// `index * stride`, refused when the product does not fit in `u64`.
fn mul_offset(index: u64, stride: u64) -> Result<u64, BlockError> {
    index.checked_mul(stride).ok_or(BlockError::Overflow)
}

/// Byte offset of a slot in its kind's pool at element width `kvb`.
/// Refuses a product that does not fit in `u64`, and a `kvb` other than 1 or 2.
pub fn byte_offset(kind: BlockKind, slot: Slot, kvb: u32) -> Result<u64, BlockError> {
    mul_offset(u64::from(slot.pool_index().get()), kind.stride_bytes(kvb)?)
}

/// Byte offset of one key row inside a slot.
/// Order is `[pass][K|V][head][pos in the block][HD]`, each element `kvb` bytes,
/// so one head's 64 keys are contiguous. `base` is the slot's `byte_offset`.
/// A pass, head or position outside the block is `OutOfRange`.
pub fn key_row(base: u64, kind: BlockKind, kvb: u32, pass: u32, kv: Kv, head: u32, index: u32) -> Result<u64, BlockError> {
    key_row_at(base, kind.passes() as u32, NKV as u32, BLOCK as u32, HD as u32, kvb, pass, kv.index(), head, index)
}

/// `key_row` at explicit bounds. The engine calls it with `NKV`, `BLOCK` and `HD`.
/// The InBlock trace calls it at the spec's bounds (2 passes, 3 heads, 4 positions, `HD` 2).
pub fn key_row_at(base: u64, passes: u32, heads: u32, block: u32, hd: u32, kvb: u32, pass: u32, plane: u32, head: u32, index: u32) -> Result<u64, BlockError> {
    if kvb != 1 && kvb != 2 { return Err(BlockError::Width); }
    if pass >= passes || head >= heads || index >= block {
        return Err(BlockError::OutOfRange);
    }
    let row = u64::from(hd).checked_mul(u64::from(kvb)).ok_or(BlockError::Overflow)?;
    let head_step = u64::from(block).checked_mul(row).ok_or(BlockError::Overflow)?;
    let kv_plane = u64::from(heads).checked_mul(head_step).ok_or(BlockError::Overflow)?;
    let pass_step = kv_plane.checked_mul(2).ok_or(BlockError::Overflow)?;
    let within = u64::from(pass).checked_mul(pass_step).ok_or(BlockError::Overflow)?
        .checked_add(u64::from(plane).checked_mul(kv_plane).ok_or(BlockError::Overflow)?).ok_or(BlockError::Overflow)?
        .checked_add(u64::from(head).checked_mul(head_step).ok_or(BlockError::Overflow)?).ok_or(BlockError::Overflow)?
        .checked_add(u64::from(index).checked_mul(row).ok_or(BlockError::Overflow)?).ok_or(BlockError::Overflow)?;
    base.checked_add(within).ok_or(BlockError::Overflow)
}

/// Write the InBlock trace TLC reads. The addresses are `key_row_at` at the
/// spec bounds (2 passes, 3 heads, 4 positions, HD 2), at element width 2
/// and then 1. CI runs this before TLC.
pub fn write_inblock_trace(path: &str) -> std::io::Result<()> {
    let mut out = String::from(
        "---------------------------- MODULE InBlockTraceData ------------------------\n\
         \\* Written by `inky inblock-trace` from key_row_at. TLC checks these\n\
         \\* addresses against Formula. Do not edit this file by hand.\n\
         EXTENDS Integers, Sequences\n\n",
    );
    for (name, kvb) in [("Trace2", 2u32), ("Trace1", 1u32)] {
        out.push_str(&format!("{name} == <<\n"));
        let mut first = true;
        for p in 0..2u32 {
            for k in 0..2u32 {
                for h in 0..3u32 {
                    for i in 0..4u32 {
                        let addr = key_row_at(0, 2, 3, 4, 2, kvb, p, k, h, i)
                            .expect("the spec bounds are in range");
                        if !first { out.push_str(",\n"); }
                        first = false;
                        out.push_str(&format!("  <<{p}, {k}, {h}, {i}, {addr}>>"));
                    }
                }
            }
        }
        out.push_str("\n>>\n\n");
    }
    out.push_str("=============================================================================\n");
    std::fs::write(path, out)
}

/// Flat table index for `(kind, block)` when each kind has `blocks_per_kind` rows.
/// Full is the even index, window the odd. Refuses an index past the declared rows
/// and a multiply that does not fit.
pub fn table_index(kind: BlockKind, block: BlockIndex, blocks_per_kind: u32) -> Result<u32, BlockError> {
    if block.get() >= blocks_per_kind { return Err(BlockError::OutOfRange); }
    let lane = match kind { BlockKind::Full => 0, BlockKind::Window => 1 };
    block.get().checked_mul(2).and_then(|v| v.checked_add(lane)).ok_or(BlockError::Overflow)
}

/// How many window blocks a span of `c` tokens pins. A span of 512 pins 25, not 17.
pub fn span_bound(c: usize) -> usize {
    (WINDOW - 1 + c + BLOCK - 1) / BLOCK + 1
}

/// The next generation. `u32::MAX` is the last one; the successor is refused, never 0.
pub fn next_generation(generation: u32) -> Result<u32, BlockError> {
    generation.checked_add(1).ok_or(BlockError::Generation)
}

/// A written length, in positions. Past 64 is refused.
pub fn block_length(n: u32) -> Result<u32, BlockError> {
    if n > BLOCK as u32 { Err(BlockError::Length) } else { Ok(n) }
}

/// One pool slot, the record the prefix tree, the start points and the poisoned-slot
/// gate share. `generation` 0 is a slot that has never been taken.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SlotRecord {
    generation: u32,
    kind: BlockKind,
    length: u32,
    sealed: bool,
    refs: u32,
}

impl SlotRecord {
    pub fn fresh(kind: BlockKind) -> SlotRecord {
        SlotRecord { generation: 0, kind, length: 0, sealed: false, refs: 0 }
    }

    #[cfg(test)]
    fn at_generation(kind: BlockKind, generation: u32) -> SlotRecord {
        SlotRecord { generation, kind, length: 0, sealed: false, refs: 0 }
    }

    pub fn generation(&self) -> u32 { self.generation }
    pub fn kind(&self) -> BlockKind { self.kind }
    pub fn length(&self) -> u32 { self.length }
    pub fn sealed(&self) -> bool { self.sealed }
    pub fn refs(&self) -> u32 { self.refs }

    /// The next generation. A live reference is `Refs`, and the count is left
    /// as it was. `reissue` is the route that waits for the last reference.
    pub fn bump_generation(&mut self) -> Result<(), BlockError> {
        if self.refs != 0 { return Err(BlockError::Refs); }
        self.generation = next_generation(self.generation)?;
        Ok(())
    }

    pub fn set_length(&mut self, n: u32) -> Result<(), BlockError> {
        self.length = block_length(n)?;
        Ok(())
    }

    pub fn seal(&mut self) { self.sealed = true; }

    pub fn add_ref(&mut self) -> Result<(), BlockError> {
        self.refs = self.refs.checked_add(1).ok_or(BlockError::Overflow)?;
        Ok(())
    }

    pub fn drop_ref(&mut self) -> Result<(), BlockError> {
        self.refs = self.refs.checked_sub(1).ok_or(BlockError::Refs)?;
        Ok(())
    }

    /// Take the slot again once nothing holds it. A live reference is `Refs`.
    /// Otherwise the generation bumps, the length goes to 0, and the seal clears.
    pub fn reissue(&mut self) -> Result<(), BlockError> {
        if self.refs != 0 { return Err(BlockError::Refs); }
        self.bump_generation()?;
        self.length = 0;
        self.sealed = false;
        Ok(())
    }
}

/// One table entry. The kind is on the entry, not implied by a packed index alone.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlockEntry {
    slot: Slot,
    generation: u32,
    offset: u64,
    kind: BlockKind,
    kvb: u32,
}

impl BlockEntry {
    pub fn new(slot: Slot, generation: u32, offset: u64, kind: BlockKind, kvb: u32) -> BlockEntry {
        BlockEntry { slot, generation, offset, kind, kvb }
    }
    pub fn slot(&self) -> Slot { self.slot }
    pub fn generation(&self) -> u32 { self.generation }
    pub fn offset(&self) -> u64 { self.offset }
    pub fn kind(&self) -> BlockKind { self.kind }
    pub fn kvb(&self) -> u32 { self.kvb }
}

/// Where one block sits, as read from the lane table. `offset` is the slot base
/// (`byte_offset` at fill). `index` is the position inside that 64-position block.
/// A key row is `row`: pass, K or V, and head. Neither is `pos % cap` in the
/// layer-2 contiguous cache.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct KeyAddr {
    offset: u64,
    index: u32,
    slot: Slot,
    generation: u32,
    kind: BlockKind,
    kvb: u32,
}

impl KeyAddr {
    pub fn offset(&self) -> u64 { self.offset }
    pub fn index(&self) -> u32 { self.index }
    pub fn slot(&self) -> Slot { self.slot }
    pub fn generation(&self) -> u32 { self.generation }
    pub fn kind(&self) -> BlockKind { self.kind }
    pub fn kvb(&self) -> u32 { self.kvb }

    /// Byte offset of one key row. See `key_row`.
    pub fn row(&self, pass: u32, kv: Kv, head: u32) -> Result<u64, BlockError> {
        key_row(self.offset, self.kind, self.kvb, pass, kv, head, self.index)
    }
}

/// One kind's pool. This commit keeps the records and does not allocate GPU buffers.
#[derive(Clone, Debug)]
pub struct BlockPool {
    kind: BlockKind,
    /// Element width for every slot in this pool. 2 is fp16, 1 is `--kv8`.
    kvb: u32,
    slots: Vec<SlotRecord>,
}

impl BlockPool {
    /// A pool at fp16 (`kvb` 2).
    pub fn new(kind: BlockKind, n_slots: usize) -> BlockPool {
        BlockPool::with_kvb(kind, n_slots, 2).expect("kvb 2 fits in a slot stride")
    }

    /// One element width for the whole pool. `kvb` 0 is `Width`. The stride must fit.
    pub fn with_kvb(kind: BlockKind, n_slots: usize, kvb: u32) -> Result<BlockPool, BlockError> {
        if kvb == 0 { return Err(BlockError::Width); }
        let _ = kind.stride_bytes(kvb)?;
        Ok(BlockPool { kind, kvb, slots: vec![SlotRecord::fresh(kind); n_slots] })
    }

    pub fn kind(&self) -> BlockKind { self.kind }
    pub fn kvb(&self) -> u32 { self.kvb }
    pub fn len(&self) -> usize { self.slots.len() }

    pub fn record(&self, slot: Slot) -> Result<&SlotRecord, BlockError> {
        let i = slot.pool_index().get() as usize;
        self.slots.get(i).ok_or(BlockError::OutOfRange)
    }

    pub fn record_mut(&mut self, slot: Slot) -> Result<&mut SlotRecord, BlockError> {
        let i = slot.pool_index().get() as usize;
        self.slots.get_mut(i).ok_or(BlockError::OutOfRange)
    }

    pub fn offset(&self, slot: Slot) -> Result<u64, BlockError> {
        let _ = self.record(slot)?;
        byte_offset(self.kind, slot, self.kvb)
    }
}

/// Both pools. One per kind, so their offsets are not in one address space.
#[derive(Clone, Debug)]
pub struct BlockPools {
    pub window: BlockPool,
    pub full: BlockPool,
}

impl BlockPools {
    /// No slots yet. The pool worker is the one that sizes them.
    pub fn stub() -> BlockPools {
        BlockPools { window: BlockPool::new(BlockKind::Window, 0), full: BlockPool::new(BlockKind::Full, 0) }
    }

    pub fn of(&self, kind: BlockKind) -> &BlockPool {
        match kind { BlockKind::Window => &self.window, BlockKind::Full => &self.full }
    }

    pub fn of_mut(&mut self, kind: BlockKind) -> &mut BlockPool {
        match kind { BlockKind::Window => &mut self.window, BlockKind::Full => &mut self.full }
    }
}

/// One lane's table. A missing entry inside the declared rows is no entry.
/// An index outside those rows is `OutOfRange`, not a missing entry.
#[derive(Clone, Debug)]
pub struct BlockTable {
    blocks_per_kind: u32,
    entries: Vec<Option<BlockEntry>>,
}

impl BlockTable {
    pub fn empty() -> BlockTable { BlockTable { blocks_per_kind: 0, entries: Vec::new() } }

    pub fn new(blocks_per_kind: u32) -> Result<BlockTable, BlockError> {
        let n = usize::try_from(blocks_per_kind).ok().and_then(|n| n.checked_mul(2)).ok_or(BlockError::Overflow)?;
        Ok(BlockTable { blocks_per_kind, entries: vec![None; n] })
    }

    pub fn get(&self, kind: BlockKind, block: BlockIndex) -> Result<Option<BlockEntry>, BlockError> {
        let i = table_index(kind, block, self.blocks_per_kind)? as usize;
        Ok(self.entries.get(i).copied().flatten())
    }

    /// Write one entry. The stored offset is `byte_offset` at the pool's `kvb`.
    /// Generation 0, on the argument or the record, is `NotTaken` and takes no
    /// reference. Any other mismatch is `Stale` and takes no reference.
    /// An entry already in the cell drops its reference only when that entry's
    /// generation is still the slot's generation. Every check finishes before
    /// any reference changes. An empty pool is `NoSlots`. This does not bump
    /// the generation and does not allocate a buffer.
    pub fn fill(&mut self, pool: &mut BlockPool, block: BlockIndex, slot: Slot, generation: u32) -> Result<(), BlockError> {
        if pool.len() == 0 { return Err(BlockError::NoSlots); }
        let record_gen = pool.record(slot)?.generation();
        if generation == 0 || record_gen == 0 { return Err(BlockError::NotTaken); }
        if record_gen != generation { return Err(BlockError::Stale); }
        let kvb = pool.kvb();
        let offset = byte_offset(pool.kind(), slot, kvb)?;
        let i = table_index(pool.kind(), block, self.blocks_per_kind)? as usize;
        let prev = self.entries.get(i).copied().flatten();
        let drop_prev = match prev {
            Some(prev) => pool.record(prev.slot())?.generation() == prev.generation(),
            None => false,
        };
        let same = drop_prev && prev.map(|prev| prev.slot()) == Some(slot);
        if drop_prev {
            let held = prev.ok_or(BlockError::NoEntry)?;
            if pool.record(held.slot())?.refs() == 0 { return Err(BlockError::Refs); }
        }
        if !same && pool.record(slot)?.refs() == u32::MAX { return Err(BlockError::Overflow); }
        if drop_prev && !same {
            let held = prev.ok_or(BlockError::NoEntry)?;
            pool.record_mut(held.slot())?.drop_ref()?;
        }
        if !same {
            pool.record_mut(slot)?.add_ref()?;
        }
        let cell = self.entries.get_mut(i).ok_or(BlockError::OutOfRange)?;
        *cell = Some(BlockEntry::new(slot, generation, offset, pool.kind(), kvb));
        Ok(())
    }

    /// The block at stream position `pos`. Block `pos / 64` is looked up in this table.
    /// The `u64` is that entry's slot base. A pool with no slots is `NoSlots`.
    /// A hole inside the table is `NoEntry`. An index past the table is `OutOfRange`.
    /// A generation that is not the pool record's is `Stale`. An entry stored at a
    /// different `kvb` than the pool is `Width`. A position at or past the written
    /// length is `Length`. A sealed record whose length is not the whole block is
    /// `Length` for every position.
    pub fn read_key(&self, pool: &BlockPool, pos: usize) -> Result<KeyAddr, BlockError> {
        if pool.len() == 0 { return Err(BlockError::NoSlots); }
        let block = BlockIndex::of_pos(pos)?;
        let entry = self.get(pool.kind(), block)?.ok_or(BlockError::NoEntry)?;
        if entry.kvb() != pool.kvb() { return Err(BlockError::Width); }
        let record = pool.record(entry.slot())?;
        if record.generation() != entry.generation() { return Err(BlockError::Stale); }
        if record.refs() == 0 { return Err(BlockError::Refs); }
        let index = u32::try_from(pos % BLOCK).map_err(|_| BlockError::Overflow)?;
        if record.sealed() && record.length() != BLOCK as u32 { return Err(BlockError::Length); }
        if index >= record.length() { return Err(BlockError::Length); }
        Ok(KeyAddr { offset: entry.offset(), index, slot: entry.slot(), generation: entry.generation(), kind: entry.kind(), kvb: entry.kvb() })
    }
}

impl Default for BlockTable {
    fn default() -> Self { BlockTable::empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_are_plain_bytes_per_kind() {
        let s = Slot::new(1).unwrap();
        assert_eq!(byte_offset(BlockKind::Full, s, 2).unwrap(), 0);
        assert_eq!(byte_offset(BlockKind::Window, s, 2).unwrap(), 0);
        let s3 = Slot::new(3).unwrap();
        assert_eq!(byte_offset(BlockKind::Full, s3, 2).unwrap(), 2 * FULL_STRIDE);
        assert_eq!(byte_offset(BlockKind::Window, s3, 2).unwrap(), 2 * WINDOW_STRIDE);
        assert_eq!(BlockKind::Full.stride_bytes(2).unwrap(), FULL_STRIDE);
        assert_eq!(BlockKind::Window.stride_bytes(2).unwrap(), WINDOW_STRIDE);
        assert_eq!(BlockKind::Full.stride_bytes(1).unwrap(), FULL_STRIDE / 2);
        assert_eq!(BlockKind::Window.stride_bytes(1).unwrap(), WINDOW_STRIDE / 2);
        assert_eq!(byte_offset(BlockKind::Window, s3, 1).unwrap(), 2 * (WINDOW_STRIDE / 2));
        assert_eq!(BlockKind::Full.stride_bytes(0), Err(BlockError::Width));
    }

    #[test]
    fn slot_zero_is_refused() {
        assert_eq!(Slot::new(0), Err(BlockError::SlotZero));
    }

    #[test]
    fn offset_multiply_is_refused() {
        assert_eq!(mul_offset(u64::MAX / WINDOW_STRIDE + 1, WINDOW_STRIDE), Err(BlockError::Overflow));
        let slot = Slot::new(u32::MAX).unwrap();
        let index = u64::from(slot.pool_index().get());
        assert_eq!(byte_offset(BlockKind::Window, slot, 2).unwrap(), index * WINDOW_STRIDE);
        let empty = BlockPool::new(BlockKind::Window, 0);
        assert_eq!(empty.record(Slot::new(1).unwrap()).map(|r| r.kind()), Err(BlockError::OutOfRange));
        let one = BlockPool::new(BlockKind::Full, 1);
        assert_eq!(one.kind(), BlockKind::Full);
        assert_eq!(one.len(), 1);
        assert_eq!(one.kvb(), 2);
        assert_eq!(one.offset(Slot::new(1).unwrap()).unwrap(), 0);
        assert_eq!(one.offset(Slot::new(2).unwrap()), Err(BlockError::OutOfRange));
        let narrow = BlockPool::with_kvb(BlockKind::Full, 1, 1).unwrap();
        assert_eq!(narrow.kvb(), 1);
        assert_eq!(narrow.offset(Slot::new(1).unwrap()).unwrap(), 0);
        assert_eq!(BlockPool::with_kvb(BlockKind::Window, 1, 0).err(), Some(BlockError::Width));
        assert_eq!(byte_offset(BlockKind::Window, Slot::new(1).unwrap(), 3), Err(BlockError::Width));
        assert_eq!(BlockPool::with_kvb(BlockKind::Window, 1, 3).err(), Some(BlockError::Width));
        let pools = BlockPools::stub();
        assert_eq!(pools.window.len(), 0);
        assert_eq!(pools.full.kind(), BlockKind::Full);
        assert_eq!(Slot::new(1).unwrap().get(), 1);
    }

    #[test]
    fn table_index_refuses_range_and_overflow() {
        assert_eq!(table_index(BlockKind::Full, BlockIndex::new(4), 4), Err(BlockError::OutOfRange));
        let wide = BlockIndex::new(u32::MAX / 2 + 1);
        assert_eq!(table_index(BlockKind::Window, wide, u32::MAX), Err(BlockError::Overflow));
    }

    #[test]
    fn generation_does_not_wrap_and_length_stops_at_64() {
        assert_eq!(next_generation(u32::MAX), Err(BlockError::Generation));
        assert_eq!(next_generation(0).unwrap(), 1);
        assert_eq!(block_length(64).unwrap(), 64);
        assert_eq!(block_length(65), Err(BlockError::Length));
    }

    #[test]
    fn span_of_512_pins_25_window_blocks() {
        assert_eq!(BlockIndex::of_pos(0).unwrap().get(), 0);
        assert_eq!(BlockIndex::of_pos(64).unwrap().get(), 1);
        assert_eq!(span_bound(512), 25);
    }

    #[test]
    fn a_hole_inside_the_table_is_no_entry_and_past_it_is_not() {
        let t = BlockTable::new(4).unwrap();
        assert_eq!(t.get(BlockKind::Window, BlockIndex::new(0)).unwrap(), None);
        assert_eq!(t.get(BlockKind::Full, BlockIndex::new(4)), Err(BlockError::OutOfRange));
        let mut fresh = SlotRecord::fresh(BlockKind::Full);
        assert_eq!(fresh.generation(), 0);
        assert!(!fresh.sealed());
        assert_eq!(fresh.refs(), 0);
        fresh.bump_generation().unwrap();
        assert_eq!(fresh.generation(), 1);
        assert_eq!(fresh.set_length(70), Err(BlockError::Length));
        assert_eq!(fresh.length(), 0);
        fresh.set_length(64).unwrap();
        assert_eq!(fresh.length(), 64);
        fresh.seal();
        assert!(fresh.sealed());
        assert_eq!(fresh.drop_ref(), Err(BlockError::Refs));
        fresh.add_ref().unwrap();
        fresh.drop_ref().unwrap();
        assert_eq!(fresh.refs(), 0);
        let mut last = SlotRecord::at_generation(BlockKind::Window, u32::MAX);
        assert_eq!(last.bump_generation(), Err(BlockError::Generation));
        assert_eq!(last.generation(), u32::MAX);
        let entry = BlockEntry::new(Slot::new(1).unwrap(), 1, 0, BlockKind::Full, 2);
        assert_eq!(entry.kind(), BlockKind::Full);
        assert_eq!(entry.offset(), 0);
        assert_eq!(entry.generation(), 1);
        assert_eq!(entry.slot().get(), 1);
        assert_eq!(entry.kvb(), 2);
    }

    fn take(pool: &mut BlockPool, slot: Slot) {
        pool.record_mut(slot).unwrap().bump_generation().unwrap();
    }

    fn written(pool: &mut BlockPool, slot: Slot, n: u32) {
        pool.record_mut(slot).unwrap().set_length(n).unwrap();
        if n == BLOCK as u32 {
            pool.record_mut(slot).unwrap().seal();
        }
    }

    #[test]
    fn fill_stores_byte_offset_and_a_key_read_uses_the_table() {
        let mut full = BlockPool::new(BlockKind::Full, 4);
        let mut window = BlockPool::new(BlockKind::Window, 4);
        let mut t = BlockTable::new(4).unwrap();
        let slot = Slot::new(3).unwrap();
        take(&mut full, slot);
        take(&mut window, slot);
        assert_eq!(t.fill(&mut full, BlockIndex::new(0), slot, 4), Err(BlockError::Stale));
        assert_eq!(full.record(slot).unwrap().refs(), 0);
        t.fill(&mut full, BlockIndex::new(0), slot, 1).unwrap();
        t.fill(&mut window, BlockIndex::new(0), slot, 1).unwrap();
        written(&mut full, slot, 64);
        written(&mut window, slot, 64);
        let stored = t.get(BlockKind::Full, BlockIndex::new(0)).unwrap().unwrap();
        assert_eq!(stored.offset(), byte_offset(BlockKind::Full, slot, 2).unwrap());
        assert_eq!(stored.offset(), 2 * FULL_STRIDE);
        assert_eq!(stored.generation(), 1);
        assert_eq!(stored.kvb(), 2);
        assert_eq!(full.record(slot).unwrap().generation(), 1);
        assert_eq!(full.record(slot).unwrap().refs(), 1);
        let key = t.read_key(&full, 0).unwrap();
        assert_eq!(key.offset(), stored.offset());
        assert_eq!(key.index(), 0);
        assert_eq!(key.slot(), slot);
        assert_eq!(key.generation(), 1);
        assert_eq!(key.kind(), BlockKind::Full);
        assert_eq!(key.row(0, Kv::Key, 0).unwrap(), stored.offset());
        let at63 = t.read_key(&full, 63).unwrap();
        assert_eq!(at63.offset(), stored.offset());
        assert_eq!(at63.index(), 63);
        assert_eq!(at63.row(0, Kv::Key, 0).unwrap(), stored.offset() + 63 * (HD as u64) * 2);
        let win = t.read_key(&window, 1).unwrap();
        assert_eq!(win.offset(), byte_offset(BlockKind::Window, slot, 2).unwrap());
        assert_eq!(win.offset(), 2 * WINDOW_STRIDE);
        assert_eq!(win.row(0, Kv::Key, 0).unwrap(), win.offset() + (HD as u64) * 2);
        take(&mut full, Slot::new(1).unwrap());
        t.fill(&mut full, BlockIndex::new(1), Slot::new(1).unwrap(), 1).unwrap();
        written(&mut full, Slot::new(1).unwrap(), 64);
        let next = t.read_key(&full, 64).unwrap();
        assert_eq!(next.offset(), byte_offset(BlockKind::Full, Slot::new(1).unwrap(), 2).unwrap());
        assert_eq!(next.index(), 0);
        assert_eq!(next.row(0, Kv::Key, 0).unwrap(), 0);
    }

    #[test]
    fn a_reused_slot_is_refused_at_the_old_generation() {
        let mut full = BlockPool::new(BlockKind::Full, 4);
        let mut t = BlockTable::new(4).unwrap();
        let slot = Slot::new(3).unwrap();
        take(&mut full, slot);
        t.fill(&mut full, BlockIndex::new(0), slot, 1).unwrap();
        written(&mut full, slot, 1);
        assert_eq!(t.read_key(&full, 0).unwrap().generation(), 1);
        assert_eq!(full.record_mut(slot).unwrap().bump_generation(), Err(BlockError::Refs));
        assert_eq!(full.record(slot).unwrap().generation(), 1);
        assert_eq!(full.record(slot).unwrap().refs(), 1);
        assert_eq!(t.read_key(&full, 0).unwrap().generation(), 1);
        assert_eq!(t.get(BlockKind::Full, BlockIndex::new(0)).unwrap().unwrap().slot(), slot);
    }

    #[test]
    fn a_key_row_is_pass_then_plane_then_head_then_position() {
        // Window slot 1, position 1, fp16. The old step (stride/64) is 163840,
        // which is pass 0, V, head 2, position 0. The key at head 0 is one row in.
        let base = byte_offset(BlockKind::Window, Slot::new(1).unwrap(), 2).unwrap();
        assert_eq!(base, 0);
        let old_step = WINDOW_STRIDE / BLOCK as u64;
        assert_eq!(old_step, 163840);
        assert_eq!(key_row(base, BlockKind::Window, 2, 0, Kv::Value, 2, 0).unwrap(), old_step);
        assert_eq!(key_row(base, BlockKind::Window, 2, 0, Kv::Key, 0, 1).unwrap(), (HD as u64) * 2);
        // Engine-sized addresses. The small-bounds behaviour is the InBlock trace.
        assert_eq!(key_row(0, BlockKind::Window, 2, 0, Kv::Key, 0, 0).unwrap(), 0);
        assert_eq!(key_row(0, BlockKind::Window, 2, 0, Kv::Key, 0, 1).unwrap(), 256);
        assert_eq!(key_row(0, BlockKind::Window, 2, 0, Kv::Key, 1, 0).unwrap(), 16_384);
        assert_eq!(key_row(0, BlockKind::Window, 2, 0, Kv::Value, 2, 0).unwrap(), 163_840);
        let head = (BLOCK as u64) * (HD as u64) * 2;
        assert_eq!(key_row(base, BlockKind::Full, 2, 0, Kv::Key, 1, 0).unwrap(), head);
        assert_eq!(key_row(base, BlockKind::Full, 2, 0, Kv::Key, 0, 0).unwrap(), 0);
        // kv8 is one byte per element, so the same head step is half.
        assert_eq!(key_row(base, BlockKind::Full, 1, 0, Kv::Key, 1, 0).unwrap(), head / 2);
        // pass 1, V, head 3, position 7, kv8. The number is not built from key_row's terms.
        assert_eq!(key_row(0, BlockKind::Window, 1, 1, Kv::Value, 3, 7).unwrap(), 222_080);
        assert_eq!(key_row(0, BlockKind::Window, 2, 40, Kv::Key, 0, 0), Err(BlockError::OutOfRange));
        assert_eq!(key_row(0, BlockKind::Full, 2, 8, Kv::Key, 0, 0), Err(BlockError::OutOfRange));
        assert_eq!(key_row(0, BlockKind::Full, 2, 0, Kv::Key, NKV as u32, 0), Err(BlockError::OutOfRange));
        assert_eq!(key_row(0, BlockKind::Full, 0, 0, Kv::Key, 0, 0), Err(BlockError::Width));
    }

    #[test]
    fn an_empty_pool_has_no_slots() {
        let mut pools = BlockPools::stub();
        let mut t = BlockTable::new(2).unwrap();
        let err = t.fill(pools.of_mut(BlockKind::Window), BlockIndex::new(0), Slot::new(1).unwrap(), 1).unwrap_err();
        assert_eq!(err, BlockError::NoSlots);
        assert_eq!(err.to_string(), "pool has no slots");
        let err = t.read_key(pools.of(BlockKind::Full), 0).unwrap_err();
        assert_eq!(err, BlockError::NoSlots);
        assert_eq!(err.to_string(), "pool has no slots");
        assert_eq!(pools.full.len(), 0);
        assert_eq!(t.get(BlockKind::Full, BlockIndex::new(0)).unwrap(), None);
        let one = BlockPool::new(BlockKind::Full, 1);
        let mut one = one;
        assert_eq!(t.fill(&mut one, BlockIndex::new(0), Slot::new(2).unwrap(), 1), Err(BlockError::OutOfRange));
        let empty_table = BlockTable::empty();
        assert_eq!(empty_table.read_key(&one, 0), Err(BlockError::OutOfRange));
        let sized = BlockPool::new(BlockKind::Window, 2);
        let bare = BlockTable::new(4).unwrap();
        assert_eq!(bare.read_key(&sized, 0), Err(BlockError::NoEntry));
        assert_eq!(bare.read_key(&sized, 4 * BLOCK), Err(BlockError::OutOfRange));
    }

    #[test]
    fn a_key_offset_is_a_u64_past_the_layer2_base() {
        let index = u32::MAX / (WINDOW_STRIDE as u32) + 1;
        let slot = Slot::new(index + 1).unwrap();
        let mut pool = BlockPool::new(BlockKind::Window, (index + 1) as usize);
        take(&mut pool, slot);
        let mut t = BlockTable::new(1).unwrap();
        t.fill(&mut pool, BlockIndex::new(0), slot, 1).unwrap();
        written(&mut pool, slot, 1);
        let key = t.read_key(&pool, 0).unwrap();
        let off = key.row(0, Kv::Key, 0).unwrap();
        assert_eq!(off, byte_offset(BlockKind::Window, slot, 2).unwrap());
        assert!(off > u64::from(u32::MAX));
        assert!(u32::try_from(off).is_err());
    }

    #[test]
    fn overwriting_an_entry_drops_the_previous_reference() {
        let mut pool = BlockPool::new(BlockKind::Full, 4);
        let mut t = BlockTable::new(1).unwrap();
        let a = Slot::new(1).unwrap();
        let b = Slot::new(2).unwrap();
        take(&mut pool, a);
        take(&mut pool, b);
        t.fill(&mut pool, BlockIndex::new(0), a, 1).unwrap();
        assert_eq!(pool.record(a).unwrap().refs(), 1);
        t.fill(&mut pool, BlockIndex::new(0), b, 1).unwrap();
        assert_eq!(pool.record(a).unwrap().refs(), 0);
        assert_eq!(pool.record(b).unwrap().refs(), 1);
        t.fill(&mut pool, BlockIndex::new(0), b, 1).unwrap();
        assert_eq!(pool.record(b).unwrap().refs(), 1);
    }

    #[test]
    fn a_fresh_slot_is_not_taken() {
        let mut pool = BlockPool::new(BlockKind::Full, 2);
        let mut t = BlockTable::new(1).unwrap();
        let slot = Slot::new(1).unwrap();
        assert_eq!(t.fill(&mut pool, BlockIndex::new(0), slot, 0), Err(BlockError::NotTaken));
        assert_eq!(pool.record(slot).unwrap().refs(), 0);
        assert_eq!(t.fill(&mut pool, BlockIndex::new(0), slot, 1), Err(BlockError::NotTaken));
        assert_eq!(pool.record(slot).unwrap().refs(), 0);
    }

    #[test]
    fn a_read_stops_at_the_written_length() {
        let mut pool = BlockPool::new(BlockKind::Window, 2);
        let mut t = BlockTable::new(1).unwrap();
        let slot = Slot::new(1).unwrap();
        take(&mut pool, slot);
        t.fill(&mut pool, BlockIndex::new(0), slot, 1).unwrap();
        assert_eq!(t.read_key(&pool, 0), Err(BlockError::Length));
        written(&mut pool, slot, 3);
        assert_eq!(t.read_key(&pool, 2).unwrap().index(), 2);
        assert_eq!(t.read_key(&pool, 40), Err(BlockError::Length));
        pool.record_mut(slot).unwrap().seal();
        assert_eq!(t.read_key(&pool, 2), Err(BlockError::Length));
    }

    #[test]
    fn reissue_waits_for_the_last_reference_and_resets_the_slot() {
        let mut r = SlotRecord::fresh(BlockKind::Window);
        r.reissue().unwrap();
        r.set_length(64).unwrap();
        r.seal();
        r.add_ref().unwrap();
        assert_eq!(r.reissue(), Err(BlockError::Refs));
        assert_eq!((r.generation(), r.length(), r.sealed()), (1, 64, true));
        r.drop_ref().unwrap();
        r.reissue().unwrap();
        assert_eq!((r.generation(), r.length(), r.sealed(), r.refs()), (2, 0, false, 0));
        let mut last = SlotRecord::at_generation(BlockKind::Full, u32::MAX);
        assert_eq!(last.reissue(), Err(BlockError::Generation));
        assert_eq!(last.generation(), u32::MAX);
    }

    #[test]
    fn a_fill_does_not_drop_a_reissued_slot() {
        let mut pool = BlockPool::new(BlockKind::Full, 4);
        let mut lane1 = BlockTable::new(1).unwrap();
        let mut lane2 = BlockTable::new(1).unwrap();
        let a = Slot::new(1).unwrap();
        let b = Slot::new(2).unwrap();
        take(&mut pool, a);
        lane1.fill(&mut pool, BlockIndex::new(0), a, 1).unwrap();
        pool.record_mut(a).unwrap().drop_ref().unwrap();
        pool.record_mut(a).unwrap().reissue().unwrap();
        lane2.fill(&mut pool, BlockIndex::new(0), a, 2).unwrap();
        assert_eq!(pool.record(a).unwrap().refs(), 1);
        take(&mut pool, b);
        lane1.fill(&mut pool, BlockIndex::new(0), b, 1).unwrap();
        assert_eq!(pool.record(a).unwrap().refs(), 1);
        assert_eq!(pool.record(b).unwrap().refs(), 1);
    }

    #[test]
    fn a_fill_refuses_before_it_changes_a_reference() {
        let mut pool = BlockPool::new(BlockKind::Full, 4);
        let mut t = BlockTable::new(1).unwrap();
        let a = Slot::new(1).unwrap();
        let b = Slot::new(2).unwrap();
        take(&mut pool, a);
        take(&mut pool, b);
        t.fill(&mut pool, BlockIndex::new(0), a, 1).unwrap();
        pool.record_mut(b).unwrap().refs = u32::MAX;
        assert_eq!(t.fill(&mut pool, BlockIndex::new(0), b, 1), Err(BlockError::Overflow));
        assert_eq!(pool.record(a).unwrap().refs(), 1);
        assert_eq!(t.get(BlockKind::Full, BlockIndex::new(0)).unwrap().unwrap().slot(), a);
    }

    #[test]
    fn a_read_refuses_a_slot_nothing_references() {
        let mut pool = BlockPool::new(BlockKind::Full, 2);
        let mut t = BlockTable::new(1).unwrap();
        let slot = Slot::new(1).unwrap();
        take(&mut pool, slot);
        t.fill(&mut pool, BlockIndex::new(0), slot, 1).unwrap();
        written(&mut pool, slot, 1);
        pool.record_mut(slot).unwrap().drop_ref().unwrap();
        assert_eq!(t.read_key(&pool, 0), Err(BlockError::Refs));
    }

    fn trace_rows(text: &str, name: &str) -> Vec<(u32, u32, u32, u32, u64)> {
        let start = text.find(&format!("{name} == <<")).expect(name);
        let rest = &text[start..];
        let end = rest.find("\n>>").expect("trace end");
        let mut rows = Vec::new();
        for line in rest[..end].lines() {
            let line = line.trim().trim_end_matches(',');
            if !line.starts_with("<<") { continue; }
            let inner = line.trim_start_matches("<<").trim_end_matches(">>");
            let nums: Vec<u64> = inner.split(',').map(|s| s.trim().parse().unwrap()).collect();
            rows.push((nums[0] as u32, nums[1] as u32, nums[2] as u32, nums[3] as u32, nums[4]));
        }
        rows
    }

    #[test]
    fn key_row_at_matches_the_inblock_trace() {
        let text = include_str!("../specs/InBlockTraceData.tla");
        for (kvb, name) in [(2, "Trace2"), (1, "Trace1")] {
            let rows = trace_rows(text, name);
            assert_eq!(rows.len(), 48);
            for (p, k, h, i, addr) in rows {
                assert_eq!(key_row_at(0, 2, 3, 4, 2, kvb, p, k, h, i).unwrap(), addr);
            }
        }
    }
}
