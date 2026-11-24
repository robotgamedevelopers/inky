//! Layer 3 block cache, the types the rest of the layer builds on.
//!
//! The GPU buffers and the attention read are not allocated here.
//! A block is 64 positions. A window block is 10 MiB and a full block is 2 MiB
//! (DESIGN.md). Each kind has its own pool, so an offset is a plain `u64` byte
//! offset in that pool: base 0 plus the 0-based pool index times the stride.
//! `BlockLayout`'s high limb of 2 is TLC's trap for a 32-bit truncation, not a base
//! this code uses.
//!
//! A generation does not wrap. The next generation after `u32::MAX` is refused.

use std::fmt;

/// Positions in one block.
pub const BLOCK: usize = 64;
/// Sliding-window positions a start point keeps.
pub const WINDOW: usize = 1024;
/// Bytes in one full-attention block. Eight full-attention passes, 2 MiB.
pub const FULL_STRIDE: u64 = 2 * 1024 * 1024;
/// Bytes in one sliding-window block. Forty window passes, 10 MiB.
pub const WINDOW_STRIDE: u64 = 10 * 1024 * 1024;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlockKind { Window, Full }

impl BlockKind {
    pub fn stride(self) -> u64 {
        match self {
            BlockKind::Window => WINDOW_STRIDE,
            BlockKind::Full => FULL_STRIDE,
        }
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

/// Byte offset of a slot in its kind's pool. Refuses a product that does not fit in `u64`.
pub fn byte_offset(kind: BlockKind, slot: Slot) -> Result<u64, BlockError> {
    mul_offset(u64::from(slot.pool_index().get()), kind.stride())
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

    pub fn bump_generation(&mut self) -> Result<(), BlockError> {
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

    /// Hand an unreferenced slot out again: the next generation, nothing written,
    /// unsealed. Its payload must be zeroed before the new owner's step (blockbook.rs).
    /// Refused while anything still references the slot.
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
}

impl BlockEntry {
    pub fn new(slot: Slot, generation: u32, offset: u64, kind: BlockKind) -> BlockEntry {
        BlockEntry { slot, generation, offset, kind }
    }
    pub fn slot(&self) -> Slot { self.slot }
    pub fn generation(&self) -> u32 { self.generation }
    pub fn offset(&self) -> u64 { self.offset }
    pub fn kind(&self) -> BlockKind { self.kind }
}

/// One kind's pool. This commit keeps the records and does not allocate GPU buffers.
#[derive(Clone, Debug)]
pub struct BlockPool {
    kind: BlockKind,
    slots: Vec<SlotRecord>,
}

impl BlockPool {
    pub fn new(kind: BlockKind, n_slots: usize) -> BlockPool {
        BlockPool { kind, slots: vec![SlotRecord::fresh(kind); n_slots] }
    }
    pub fn kind(&self) -> BlockKind { self.kind }
    pub fn len(&self) -> usize { self.slots.len() }

    pub fn record(&self, slot: Slot) -> Result<&SlotRecord, BlockError> {
        let i = slot.pool_index().get() as usize;
        self.slots.get(i).ok_or(BlockError::OutOfRange)
    }

    pub fn offset(&self, slot: Slot) -> Result<u64, BlockError> {
        let _ = self.record(slot)?;
        byte_offset(self.kind, slot)
    }

    pub fn record_mut(&mut self, slot: Slot) -> Result<&mut SlotRecord, BlockError> {
        let i = slot.pool_index().get() as usize;
        self.slots.get_mut(i).ok_or(BlockError::OutOfRange)
    }

    /// The lowest slot nothing references whose generation can still advance.
    pub fn lowest_free(&self) -> Option<Slot> {
        let i = self.slots.iter().position(|r| r.refs == 0 && r.generation < u32::MAX)?;
        Slot::new(u32::try_from(i + 1).ok()?).ok()
    }
}

impl BlockPools {
    /// Both pools with `n_slots` each.
    pub fn new(n_slots: usize) -> BlockPools {
        BlockPools { window: BlockPool::new(BlockKind::Window, n_slots), full: BlockPool::new(BlockKind::Full, n_slots) }
    }
    pub fn of(&self, kind: BlockKind) -> &BlockPool {
        match kind { BlockKind::Window => &self.window, BlockKind::Full => &self.full }
    }
    pub fn of_mut(&mut self, kind: BlockKind) -> &mut BlockPool {
        match kind { BlockKind::Window => &mut self.window, BlockKind::Full => &mut self.full }
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

    pub fn blocks_per_kind(&self) -> u32 { self.blocks_per_kind }

    /// Set or clear the entry for `(kind, block)`. An entry of the other kind is refused.
    pub fn set(&mut self, kind: BlockKind, block: BlockIndex, entry: Option<BlockEntry>) -> Result<(), BlockError> {
        if entry.is_some_and(|e| e.kind() != kind) { return Err(BlockError::OutOfRange); }
        let i = table_index(kind, block, self.blocks_per_kind)? as usize;
        let cell = self.entries.get_mut(i).ok_or(BlockError::OutOfRange)?;
        *cell = entry;
        Ok(())
    }

    /// Every present entry with its block, full before window within a block,
    /// blocks ascending.
    pub fn entries(&self) -> impl Iterator<Item = (BlockIndex, BlockEntry)> + '_ {
        self.entries.iter().enumerate().filter_map(|(i, e)| e.map(|e| (BlockIndex::new((i / 2) as u32), e)))
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
        assert_eq!(byte_offset(BlockKind::Full, s).unwrap(), 0);
        assert_eq!(byte_offset(BlockKind::Window, s).unwrap(), 0);
        let s3 = Slot::new(3).unwrap();
        assert_eq!(byte_offset(BlockKind::Full, s3).unwrap(), 2 * FULL_STRIDE);
        assert_eq!(byte_offset(BlockKind::Window, s3).unwrap(), 2 * WINDOW_STRIDE);
        assert_ne!(FULL_STRIDE, WINDOW_STRIDE);
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
        assert_eq!(byte_offset(BlockKind::Window, slot).unwrap(), index * WINDOW_STRIDE);
        let empty = BlockPool::new(BlockKind::Window, 0);
        assert_eq!(empty.record(Slot::new(1).unwrap()).map(|r| r.kind()), Err(BlockError::OutOfRange));
        let one = BlockPool::new(BlockKind::Full, 1);
        assert_eq!(one.kind(), BlockKind::Full);
        assert_eq!(one.len(), 1);
        assert_eq!(one.offset(Slot::new(1).unwrap()).unwrap(), 0);
        assert_eq!(one.offset(Slot::new(2).unwrap()), Err(BlockError::OutOfRange));
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
        let entry = BlockEntry::new(Slot::new(1).unwrap(), 1, 0, BlockKind::Full);
        assert_eq!(entry.kind(), BlockKind::Full);
        assert_eq!(entry.offset(), 0);
        assert_eq!(entry.generation(), 1);
        assert_eq!(entry.slot().get(), 1);
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
    }

    #[test]
    fn lowest_free_skips_referenced_and_exhausted_slots_and_set_checks_kind() {
        let mut pools = BlockPools::new(3);
        let one = Slot::new(1).unwrap();
        pools.of_mut(BlockKind::Full).record_mut(one).unwrap().add_ref().unwrap();
        pools.full.slots[1] = SlotRecord::at_generation(BlockKind::Full, u32::MAX);
        assert_eq!(pools.of(BlockKind::Full).lowest_free(), Some(Slot::new(3).unwrap()));
        assert_eq!(pools.of(BlockKind::Window).lowest_free(), Some(one));
        assert_eq!(BlockPool::new(BlockKind::Full, 0).lowest_free(), None);
        let mut t = BlockTable::new(2).unwrap();
        let e = BlockEntry::new(one, 1, 0, BlockKind::Full);
        assert_eq!(t.set(BlockKind::Window, BlockIndex::new(0), Some(e)), Err(BlockError::OutOfRange));
        assert_eq!(t.set(BlockKind::Full, BlockIndex::new(2), Some(e)), Err(BlockError::OutOfRange));
        t.set(BlockKind::Full, BlockIndex::new(1), Some(e)).unwrap();
        assert_eq!(t.entries().map(|(b, e)| (b.get(), e.kind())).collect::<Vec<_>>(), vec![(1, BlockKind::Full)]);
        t.set(BlockKind::Full, BlockIndex::new(1), None).unwrap();
        assert_eq!(t.entries().count(), 0);
        assert_eq!(t.blocks_per_kind(), 2);
    }
}
