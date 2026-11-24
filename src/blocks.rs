//! Layer 3 block cache, the types the rest of the layer builds on.
//!
//! The GPU buffers are not allocated here. A key read goes through the per-lane
//! table (`BlockTable::read_key`): the `u64` stored on the entry, which `fill`
//! took from `byte_offset`. It is not a position in the layer-2 contiguous cache.
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
    /// This kind's pool has no slots. A key cannot be read from it.
    NoSlots,
    /// The block is inside the table and no entry was filled.
    NoEntry,
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

/// Where one key sits, as read from the lane table. `offset` is the entry's `u64`
/// (from `byte_offset` at fill). `index` is the position inside that 64-key block.
/// Neither is `pos % cap` in the layer-2 contiguous cache.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct KeyAddr {
    offset: u64,
    index: u32,
    slot: Slot,
    generation: u32,
    kind: BlockKind,
}

impl KeyAddr {
    pub fn offset(&self) -> u64 { self.offset }
    pub fn index(&self) -> u32 { self.index }
    pub fn slot(&self) -> Slot { self.slot }
    pub fn generation(&self) -> u32 { self.generation }
    pub fn kind(&self) -> BlockKind { self.kind }

    /// Byte offset of this position in its kind's pool.
    /// `entry.offset() + index * (stride / 64)`. The stride is the whole block.
    pub fn position_offset(&self) -> Result<u64, BlockError> {
        let step = self.kind.stride().checked_div(BLOCK as u64).ok_or(BlockError::Overflow)?;
        self.offset.checked_add(u64::from(self.index).checked_mul(step).ok_or(BlockError::Overflow)?)
            .ok_or(BlockError::Overflow)
    }
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

    /// Write one entry. The stored offset is `byte_offset(pool.kind(), slot)`.
    /// An empty pool is `NoSlots`, not a hole and not a layer-2 address.
    /// This does not bump the slot's generation and does not allocate a buffer.
    pub fn fill(&mut self, pool: &BlockPool, block: BlockIndex, slot: Slot, generation: u32) -> Result<(), BlockError> {
        if pool.len() == 0 { return Err(BlockError::NoSlots); }
        let _ = pool.record(slot)?;
        let offset = byte_offset(pool.kind(), slot)?;
        let i = table_index(pool.kind(), block, self.blocks_per_kind)? as usize;
        let cell = self.entries.get_mut(i).ok_or(BlockError::OutOfRange)?;
        *cell = Some(BlockEntry::new(slot, generation, offset, pool.kind()));
        Ok(())
    }

    /// The key at stream position `pos`. Block `pos / 64` is looked up in this table.
    /// The `u64` is that entry's offset. A pool with no slots is `NoSlots`.
    /// A hole inside the table is `NoEntry`. An index past the table is `OutOfRange`.
    pub fn read_key(&self, pool: &BlockPool, pos: usize) -> Result<KeyAddr, BlockError> {
        if pool.len() == 0 { return Err(BlockError::NoSlots); }
        let block = BlockIndex::of_pos(pos)?;
        let entry = self.get(pool.kind(), block)?.ok_or(BlockError::NoEntry)?;
        let _ = pool.record(entry.slot())?;
        let index = u32::try_from(pos % BLOCK).map_err(|_| BlockError::Overflow)?;
        Ok(KeyAddr { offset: entry.offset(), index, slot: entry.slot(), generation: entry.generation(), kind: entry.kind() })
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
    fn fill_stores_byte_offset_and_a_key_read_uses_the_table() {
        let full = BlockPool::new(BlockKind::Full, 4);
        let window = BlockPool::new(BlockKind::Window, 4);
        let mut t = BlockTable::new(4).unwrap();
        let slot = Slot::new(3).unwrap();
        t.fill(&full, BlockIndex::new(0), slot, 4).unwrap();
        t.fill(&window, BlockIndex::new(0), slot, 9).unwrap();
        let stored = t.get(BlockKind::Full, BlockIndex::new(0)).unwrap().unwrap();
        assert_eq!(stored.offset(), byte_offset(BlockKind::Full, slot).unwrap());
        assert_eq!(stored.offset(), 2 * FULL_STRIDE);
        assert_eq!(stored.generation(), 4);
        assert_eq!(full.record(slot).unwrap().generation(), 0);
        let key = t.read_key(&full, 0).unwrap();
        assert_eq!(key.offset(), stored.offset());
        assert_eq!(key.index(), 0);
        assert_eq!(key.slot(), slot);
        assert_eq!(key.generation(), 4);
        assert_eq!(key.kind(), BlockKind::Full);
        assert_eq!(key.position_offset().unwrap(), stored.offset());
        assert_ne!(key.offset(), 0);
        let at63 = t.read_key(&full, 63).unwrap();
        let step = FULL_STRIDE / BLOCK as u64;
        assert_eq!(at63.offset(), stored.offset());
        assert_eq!(at63.index(), 63);
        assert_eq!(at63.position_offset().unwrap(), stored.offset() + 63 * step);
        assert_ne!(at63.position_offset().unwrap(), 63 * step);
        let win = t.read_key(&window, 1).unwrap();
        assert_eq!(win.offset(), byte_offset(BlockKind::Window, slot).unwrap());
        assert_eq!(win.offset(), 2 * WINDOW_STRIDE);
        assert_ne!(win.position_offset().unwrap(), key.position_offset().unwrap());
        t.fill(&full, BlockIndex::new(1), Slot::new(1).unwrap(), 1).unwrap();
        let next = t.read_key(&full, 64).unwrap();
        assert_eq!(next.offset(), byte_offset(BlockKind::Full, Slot::new(1).unwrap()).unwrap());
        assert_eq!(next.index(), 0);
        assert_ne!(next.position_offset().unwrap(), stored.offset() + FULL_STRIDE);
    }

    #[test]
    fn an_empty_pool_has_no_slots() {
        let pools = BlockPools::stub();
        let mut t = BlockTable::new(2).unwrap();
        let err = t.fill(pools.of(BlockKind::Window), BlockIndex::new(0), Slot::new(1).unwrap(), 1).unwrap_err();
        assert_eq!(err, BlockError::NoSlots);
        assert_eq!(err.to_string(), "pool has no slots");
        let err = t.read_key(pools.of(BlockKind::Full), 0).unwrap_err();
        assert_eq!(err, BlockError::NoSlots);
        assert_eq!(err.to_string(), "pool has no slots");
        assert_eq!(pools.full.len(), 0);
        assert_eq!(t.get(BlockKind::Full, BlockIndex::new(0)).unwrap(), None);
        let one = BlockPool::new(BlockKind::Full, 1);
        assert_eq!(t.fill(&one, BlockIndex::new(0), Slot::new(2).unwrap(), 1), Err(BlockError::OutOfRange));
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
        let pool = BlockPool::new(BlockKind::Window, (index + 1) as usize);
        let mut t = BlockTable::new(1).unwrap();
        t.fill(&pool, BlockIndex::new(0), slot, 1).unwrap();
        let key = t.read_key(&pool, 0).unwrap();
        let off = key.position_offset().unwrap();
        assert_eq!(off, byte_offset(BlockKind::Window, slot).unwrap());
        assert!(off > u64::from(u32::MAX));
        assert!(u32::try_from(off).is_err());
    }
}
