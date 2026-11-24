//! Layer 3 block cache, the types the rest of the layer builds on.
//!
//! A block is 64 positions. A window block is 10 MiB and a full block is 2 MiB
//! (DESIGN.md). Each kind has its own pool, so an offset is a plain `u64` byte
//! offset in that pool: base 0 plus the 0-based pool index times the stride.
//! `BlockLayout`'s high limb of 2 is TLC's trap for a 32-bit truncation, not a base
//! this code uses. A GPU pool stores that slot as its own shared Metal buffer.
//! The attention read is not here.
//!
//! A generation does not wrap. The next generation after `u32::MAX` is refused.

use std::fmt;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::Message;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

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

    /// Bytes of one position. The stride divides into 64 positions.
    pub fn position_bytes(self) -> usize {
        let stride = self.stride();
        debug_assert!(stride % BLOCK as u64 == 0);
        (stride / BLOCK as u64) as usize
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

    fn unseal(&mut self) { self.sealed = false; }

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

/// Bytes of one slot. A host pool is for tests. A GPU pool is one shared buffer per slot.
#[derive(Clone)]
enum SlotStorage {
    Host(Vec<u8>),
    Gpu(crate::metal::Buf),
}

/// One kind's pool. Slots are append-only: `grow` adds a slot at the end and does not
/// move an existing offset. `take` zero-fills a slot and bumps its generation.
pub struct BlockPool {
    kind: BlockKind,
    slots: Vec<SlotRecord>,
    bytes: Vec<SlotStorage>,
    /// Present when this pool allocates Metal buffers.
    device: Option<Retained<ProtocolObject<dyn MTLDevice>>>,
}

impl Clone for BlockPool {
    fn clone(&self) -> BlockPool {
        BlockPool {
            kind: self.kind,
            slots: self.slots.clone(),
            // A cloned GPU buffer is another retain of the same allocation.
            bytes: self.bytes.clone(),
            device: self.device.clone(),
        }
    }
}

impl fmt::Debug for BlockPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlockPool").field("kind", &self.kind).field("len", &self.slots.len()).field("gpu", &self.device.is_some()).finish()
    }
}

impl BlockPool {
    /// Host bytes, one vector per slot, zero-filled. No Metal device.
    pub fn new(kind: BlockKind, n_slots: usize) -> BlockPool {
        Self::build(kind, n_slots, None)
    }

    /// One shared Metal buffer per slot, each `kind.stride()` bytes and zero-filled.
    pub fn gpu(dev: &ProtocolObject<dyn MTLDevice>, kind: BlockKind, n_slots: usize) -> BlockPool {
        Self::build(kind, n_slots, Some(dev.retain()))
    }

    fn build(kind: BlockKind, n_slots: usize, device: Option<Retained<ProtocolObject<dyn MTLDevice>>>) -> BlockPool {
        let mut bytes = Vec::with_capacity(n_slots);
        for _ in 0..n_slots { bytes.push(Self::allocate(&device, kind)); }
        BlockPool { kind, slots: vec![SlotRecord::fresh(kind); n_slots], bytes, device }
    }

    fn allocate(device: &Option<Retained<ProtocolObject<dyn MTLDevice>>>, kind: BlockKind) -> SlotStorage {
        let n = kind.position_bytes() * BLOCK;
        match device {
            None => SlotStorage::Host(vec![0u8; n]),
            Some(dev) => {
                let buf = dev.newBufferWithLength_options(n, MTLResourceOptions::StorageModeShared).expect("block pool buffer");
                // Reuse fills this again. A fresh buffer is zero too, so a read before `take` is finite.
                unsafe { std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, n); }
                SlotStorage::Gpu(buf)
            }
        }
    }

    pub fn kind(&self) -> BlockKind { self.kind }
    pub fn len(&self) -> usize { self.slots.len() }

    fn index(&self, slot: Slot) -> Result<usize, BlockError> {
        let i = slot.pool_index().get() as usize;
        if i >= self.slots.len() { Err(BlockError::OutOfRange) } else { Ok(i) }
    }

    pub fn record(&self, slot: Slot) -> Result<&SlotRecord, BlockError> {
        Ok(&self.slots[self.index(slot)?])
    }

    pub fn record_mut(&mut self, slot: Slot) -> Result<&mut SlotRecord, BlockError> {
        let i = self.index(slot)?;
        Ok(&mut self.slots[i])
    }

    pub fn offset(&self, slot: Slot) -> Result<u64, BlockError> {
        let _ = self.record(slot)?;
        byte_offset(self.kind, slot)
    }

    /// The GPU buffer for this slot. `None` on a host pool.
    pub fn gpu_buffer(&self, slot: Slot) -> Result<Option<&crate::metal::Buf>, BlockError> {
        let i = self.index(slot)?;
        Ok(match &self.bytes[i] { SlotStorage::Gpu(b) => Some(b), SlotStorage::Host(_) => None })
    }

    /// The slot's bytes. A GPU slot is the shared buffer's contents.
    pub fn slot_bytes(&self, slot: Slot) -> Result<&[u8], BlockError> {
        let i = self.index(slot)?;
        Ok(self.bytes_of(i))
    }

    fn bytes_of(&self, i: usize) -> &[u8] {
        let n = self.kind.position_bytes() * BLOCK;
        match &self.bytes[i] {
            SlotStorage::Host(v) => v.as_slice(),
            SlotStorage::Gpu(b) => unsafe { std::slice::from_raw_parts(b.contents().as_ptr() as *const u8, n) },
        }
    }

    fn bytes_of_mut(&mut self, i: usize) -> &mut [u8] {
        let n = self.kind.position_bytes() * BLOCK;
        match &mut self.bytes[i] {
            SlotStorage::Host(v) => v.as_mut_slice(),
            // Shared storage. The CPU zero-fills and appends; no command buffer reads this yet.
            SlotStorage::Gpu(b) => unsafe { std::slice::from_raw_parts_mut(b.contents().as_ptr() as *mut u8, n) },
        }
    }

    /// Append one slot. Its offset is `len * stride`. Earlier offsets stay put.
    /// The new slot has generation 0 until `take`.
    pub fn grow(&mut self) -> Result<Slot, BlockError> {
        if self.slots.len() >= u32::MAX as usize { return Err(BlockError::Overflow); }
        let slot = Slot::new((self.slots.len() as u32).checked_add(1).ok_or(BlockError::Overflow)?)?;
        let _ = byte_offset(self.kind, slot)?;
        self.bytes.push(Self::allocate(&self.device, self.kind));
        self.slots.push(SlotRecord::fresh(self.kind));
        Ok(slot)
    }

    /// Zero-fill `slot`, clear its seal and length, and bump its generation.
    /// A live reference count is refused. Generation `u32::MAX` is refused and not cleared.
    pub fn take(&mut self, slot: Slot) -> Result<u64, BlockError> {
        let i = self.index(slot)?;
        if self.slots[i].refs() != 0 { return Err(BlockError::Refs); }
        let _ = next_generation(self.slots[i].generation())?;
        self.bytes_of_mut(i).fill(0);
        self.slots[i].bump_generation()?;
        self.slots[i].set_length(0)?;
        self.slots[i].unseal();
        byte_offset(self.kind, slot)
    }

    /// Lowest slot with a zero reference count whose generation can advance.
    /// Grows by one slot when every existing slot is live or exhausted.
    pub fn take_free(&mut self) -> Result<(Slot, u64), BlockError> {
        let n = self.slots.len();
        for i in 0..n {
            if self.slots[i].refs() == 0 && next_generation(self.slots[i].generation()).is_ok() {
                let slot = Slot::new((i as u32) + 1).map_err(|_| BlockError::Overflow)?;
                return Ok((slot, self.take(slot)?));
            }
        }
        let slot = self.grow()?;
        Ok((slot, self.take(slot)?))
    }

    /// Write the next position. `data` is one position. A sealed slot and a 65th position are refused.
    /// Bytes past the new length stay as they were (zero after `take`).
    pub fn append(&mut self, slot: Slot, data: &[u8]) -> Result<(), BlockError> {
        let width = self.kind.position_bytes();
        if data.len() != width { return Err(BlockError::Length); }
        let i = self.index(slot)?;
        if self.slots[i].generation() == 0 || self.slots[i].sealed() { return Err(BlockError::Length); }
        let len = self.slots[i].length();
        let new_len = block_length(len.checked_add(1).ok_or(BlockError::Overflow)?)?;
        let at = (len as usize).checked_mul(width).ok_or(BlockError::Overflow)?;
        self.bytes_of_mut(i)[at..at + width].copy_from_slice(data);
        self.slots[i].set_length(new_len)?;
        Ok(())
    }

    #[cfg(test)]
    fn force_generation(&mut self, slot: Slot, generation: u32) -> Result<(), BlockError> {
        self.record_mut(slot)?.generation = generation;
        Ok(())
    }

    #[cfg(test)]
    fn slot_bytes_mut_for_test(&mut self, slot: Slot) -> &mut [u8] {
        let i = self.index(slot).unwrap();
        self.bytes_of_mut(i)
    }
}

/// Both pools. One per kind, so their offsets are not in one address space.
#[derive(Clone, Debug)]
pub struct BlockPools {
    pub window: BlockPool,
    pub full: BlockPool,
}

impl BlockPools {
    /// Host records and no device. A running engine uses [`BlockPools::gpu`].
    pub fn stub() -> BlockPools {
        BlockPools { window: BlockPool::new(BlockKind::Window, 0), full: BlockPool::new(BlockKind::Full, 0) }
    }

    /// GPU pools. `n_window` and `n_full` are allocated now, one buffer per slot.
    /// Zero is a pool that still holds the device and grows through [`BlockPool::grow`].
    pub fn gpu(dev: &ProtocolObject<dyn MTLDevice>, n_window: usize, n_full: usize) -> BlockPools {
        BlockPools {
            window: BlockPool::gpu(dev, BlockKind::Window, n_window),
            full: BlockPool::gpu(dev, BlockKind::Full, n_full),
        }
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
    fn a_taken_slot_is_zero_filled_and_a_live_one_is_not() {
        for kind in [BlockKind::Full, BlockKind::Window] {
            let mut pool = BlockPool::new(kind, 1);
            let slot = Slot::new(1).unwrap();
            assert!(pool.gpu_buffer(slot).unwrap().is_none());
            assert_eq!(pool.slot_bytes(slot).unwrap().len(), kind.stride() as usize);
            pool.slot_bytes_mut_for_test(slot).fill(0xab);
            assert_eq!(pool.take(slot).unwrap(), 0);
            assert!(pool.slot_bytes(slot).unwrap().iter().all(|&b| b == 0));
            assert_eq!(pool.record(slot).unwrap().generation(), 1);
            assert_eq!(pool.record(slot).unwrap().length(), 0);
            assert!(!pool.record(slot).unwrap().sealed());
            pool.record_mut(slot).unwrap().add_ref().unwrap();
            pool.slot_bytes_mut_for_test(slot)[0] = 0xcd;
            assert_eq!(pool.take(slot), Err(BlockError::Refs));
            assert_eq!(pool.slot_bytes(slot).unwrap()[0], 0xcd);
            assert_eq!(pool.record(slot).unwrap().generation(), 1);
        }
    }

    #[test]
    fn take_does_not_wrap_and_reuse_zero_fills_again() {
        let mut pool = BlockPool::new(BlockKind::Full, 1);
        let slot = Slot::new(1).unwrap();
        pool.force_generation(slot, u32::MAX).unwrap();
        pool.slot_bytes_mut_for_test(slot)[0] = 0x11;
        assert_eq!(pool.take(slot), Err(BlockError::Generation));
        assert_eq!(pool.record(slot).unwrap().generation(), u32::MAX);
        assert_eq!(pool.slot_bytes(slot).unwrap()[0], 0x11);
        pool.force_generation(slot, 0).unwrap();
        pool.take(slot).unwrap();
        let width = BlockKind::Full.position_bytes();
        let mut pos = vec![0u8; width];
        pos[0] = 7;
        pool.append(slot, &pos).unwrap();
        pool.record_mut(slot).unwrap().seal();
        pool.take(slot).unwrap();
        assert!(!pool.record(slot).unwrap().sealed());
        assert_eq!(pool.record(slot).unwrap().generation(), 2);
        assert!(pool.slot_bytes(slot).unwrap().iter().all(|&b| b == 0));
    }

    #[test]
    fn append_writes_the_next_position_only() {
        let mut pool = BlockPool::new(BlockKind::Full, 1);
        let slot = Slot::new(1).unwrap();
        let width = BlockKind::Full.position_bytes();
        let mut pos = vec![0u8; width];
        pos[0] = 3;
        assert_eq!(pool.append(slot, &pos), Err(BlockError::Length));
        pool.take(slot).unwrap();
        assert_eq!(pool.append(slot, &[0u8; 4]), Err(BlockError::Length));
        pool.append(slot, &pos).unwrap();
        assert_eq!(pool.record(slot).unwrap().length(), 1);
        assert_eq!(pool.slot_bytes(slot).unwrap()[0], 3);
        assert_eq!(pool.slot_bytes(slot).unwrap()[width], 0);
        assert_eq!(*pool.slot_bytes(slot).unwrap().last().unwrap(), 0);
        pool.record_mut(slot).unwrap().seal();
        pos[0] = 9;
        assert_eq!(pool.append(slot, &pos), Err(BlockError::Length));
        assert_eq!(pool.record(slot).unwrap().length(), 1);
        assert_eq!(pool.slot_bytes(slot).unwrap()[0], 3);
        pool.record_mut(slot).unwrap().unseal();
        pool.record_mut(slot).unwrap().set_length(64).unwrap();
        assert_eq!(pool.append(slot, &pos), Err(BlockError::Length));
        assert_eq!(pool.record(slot).unwrap().length(), 64);
        assert!(!pool.record(slot).unwrap().sealed());
    }

    #[test]
    fn grow_appends_and_take_free_reuses_the_lowest_slot() {
        let mut pool = BlockPool::new(BlockKind::Window, 0);
        assert_eq!(pool.take(Slot::new(1).unwrap()), Err(BlockError::OutOfRange));
        let first = pool.grow().unwrap();
        assert_eq!(first.get(), 1);
        assert_eq!(pool.offset(first).unwrap(), 0);
        assert_eq!(pool.record(first).unwrap().generation(), 0);
        let (slot, off) = pool.take_free().unwrap();
        assert_eq!(slot, first);
        assert_eq!(off, 0);
        pool.record_mut(slot).unwrap().add_ref().unwrap();
        let (second, off2) = pool.take_free().unwrap();
        assert_eq!(second.get(), 2);
        assert_eq!(off2, WINDOW_STRIDE);
        assert_eq!(pool.offset(first).unwrap(), 0);
        pool.record_mut(slot).unwrap().drop_ref().unwrap();
        pool.slot_bytes_mut_for_test(slot)[0] = 0x5a;
        let (again, _) = pool.take_free().unwrap();
        assert_eq!(again, first);
        assert_eq!(pool.record(again).unwrap().generation(), 2);
        assert_eq!(pool.slot_bytes(again).unwrap()[0], 0);
        pool.force_generation(first, u32::MAX).unwrap();
        pool.force_generation(second, u32::MAX).unwrap();
        let (third, off3) = pool.take_free().unwrap();
        assert_eq!(third.get(), 3);
        assert_eq!(off3, 2 * WINDOW_STRIDE);
        assert_eq!(pool.len(), 3);
    }

    #[test]
    #[ignore = "needs a Metal device; run on mini-m5"]
    fn gpu_pool_zero_fills_a_taken_slot() {
        use objc2_metal::MTLBuffer;
        let dev = objc2_metal::MTLCreateSystemDefaultDevice().expect("metal device");
        let mut pools = BlockPools::gpu(&dev, 1, 1);
        for pool in [&mut pools.full, &mut pools.window] {
            let slot = Slot::new(1).unwrap();
            let buf = pool.gpu_buffer(slot).unwrap().expect("gpu buffer");
            assert_eq!(buf.length(), pool.kind().stride() as usize);
            pool.slot_bytes_mut_for_test(slot).fill(0xef);
            assert_eq!(pool.take(slot).unwrap(), 0);
            assert!(pool.slot_bytes(slot).unwrap().iter().all(|&b| b == 0));
            assert_eq!(pool.record(slot).unwrap().generation(), 1);
            let grown = pool.grow().unwrap();
            assert_eq!(grown.get(), 2);
            assert_eq!(pool.offset(grown).unwrap(), pool.kind().stride());
            assert!(pool.gpu_buffer(grown).unwrap().is_some());
            assert_eq!(pool.offset(slot).unwrap(), 0);
        }
        assert_ne!(pools.window.kind().stride(), pools.full.kind().stride());
    }
}
