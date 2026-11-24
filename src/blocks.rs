//! Layer 3 block cache, the types the rest of the layer builds on.
#![allow(dead_code)] // the pool, the table and the prefix tree are the callers; cargo check does not compile the unit tests
//!
//! A block is 64 positions. A window block is 10 MiB and a full block is 2 MiB
//! (DESIGN.md) at fp16 (`kvb` 2). `kvb` 1 is half of that. Each kind has its own
//! pool, so an offset is a plain `u64` byte offset in that pool: base 0 plus the
//! 0-based pool index times that stride. One contiguous allocation holds the kind.
//! When a new slot would pass the slab cap (the device `maxBufferLength` on GPU),
//! the pool opens another slab, and a fifth slab is refused. `BlockLayout`'s high
//! limb of 2 is TLC's trap for a 32-bit truncation, not a base this code uses.
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
/// fp16 element width. `--kv8` passes 1.
pub const KVB_FP16: u32 = 2;
/// A kind holds at most this many slabs. One more is [`BlockError::Slab`].
pub const MAX_SLABS: usize = 4;

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
    /// `kvb` is not 1 or 2.
    Kvb,
    /// A slot does not fit in the remaining slabs. At most [`MAX_SLABS`], each within the cap.
    Slab,
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
            BlockError::Kvb => write!(f, "block kvb must be 1 or 2"),
            BlockError::Slab => write!(f, "block slab cap exceeded"),
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

/// Byte offset of a slot in its kind's pool at the fp16 stride. Refuses a product that does not fit in `u64`.
pub fn byte_offset(kind: BlockKind, slot: Slot) -> Result<u64, BlockError> {
    mul_offset(u64::from(slot.pool_index().get()), kind.stride())
}

/// Bytes of one slot at element width `kvb`. `kvb` is 1 (`--kv8`) or 2 (fp16).
/// fp16 is [`BlockKind::stride`]. [`byte_offset`] stays on that fp16 stride.
pub fn stride_at(kind: BlockKind, kvb: u32) -> Result<u64, BlockError> {
    match kvb {
        1 | 2 => Ok(kind.stride() / 2 * u64::from(kvb)),
        _ => Err(BlockError::Kvb),
    }
}

/// How many slots fit in one slab. A stride larger than the cap is [`BlockError::Slab`].
fn slots_in_slab(stride: u64, max_slab_bytes: u64) -> Result<usize, BlockError> {
    if stride == 0 || max_slab_bytes < stride { return Err(BlockError::Slab); }
    usize::try_from(max_slab_bytes / stride).map_err(|_| BlockError::Overflow)
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

/// One slab: a contiguous allocation holding one or more slots of this kind.
/// A host slab is a `Vec`. A GPU slab is one shared Metal buffer.
#[derive(Clone)]
enum SlabStorage {
    Host(Vec<u8>),
    Gpu(crate::metal::Buf),
}

/// One kind's pool. Slots are append-only: `grow` adds a slot at the end and does not
/// move an existing offset. The bytes live in one slab until a new slot would pass
/// `maxBufferLength` (or the host cap). At most [`MAX_SLABS`]. `take` zero-fills a
/// slot and bumps its generation.
pub struct BlockPool {
    kind: BlockKind,
    kvb: u32,
    stride: u64,
    slots_per_slab: usize,
    slots: Vec<SlotRecord>,
    slabs: Vec<SlabStorage>,
    /// Present when this pool allocates Metal buffers.
    device: Option<Retained<ProtocolObject<dyn MTLDevice>>>,
}

impl Clone for BlockPool {
    fn clone(&self) -> BlockPool {
        BlockPool {
            kind: self.kind,
            kvb: self.kvb,
            stride: self.stride,
            slots_per_slab: self.slots_per_slab,
            slots: self.slots.clone(),
            // A cloned GPU buffer is another retain of the same allocation.
            slabs: self.slabs.clone(),
            device: self.device.clone(),
        }
    }
}

impl fmt::Debug for BlockPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlockPool")
            .field("kind", &self.kind)
            .field("kvb", &self.kvb)
            .field("len", &self.slots.len())
            .field("slabs", &self.slabs.len())
            .field("gpu", &self.device.is_some())
            .finish()
    }
}

impl BlockPool {
    /// Host bytes at fp16 (`kvb` 2), one contiguous slab, zero-filled. No Metal device.
    pub fn new(kind: BlockKind, n_slots: usize) -> BlockPool {
        Self::open(kind, n_slots, KVB_FP16, u64::MAX, None).expect("host block pool")
    }

    /// GPU pools for this kind. `kvb` is 1 or 2. One buffer until a slot would pass
    /// the device `maxBufferLength`, then another slab, and never a fifth.
    pub fn gpu(dev: &ProtocolObject<dyn MTLDevice>, kind: BlockKind, n_slots: usize, kvb: u32) -> BlockPool {
        let max = dev.maxBufferLength() as u64;
        Self::open(kind, n_slots, kvb, max, Some(dev.retain())).expect("gpu block pool")
    }

    fn open(
        kind: BlockKind,
        n_slots: usize,
        kvb: u32,
        max_slab_bytes: u64,
        device: Option<Retained<ProtocolObject<dyn MTLDevice>>>,
    ) -> Result<BlockPool, BlockError> {
        let stride = stride_at(kind, kvb)?;
        let slots_per_slab = slots_in_slab(stride, max_slab_bytes)?;
        let mut pool = BlockPool {
            kind, kvb, stride, slots_per_slab, slots: Vec::with_capacity(n_slots), slabs: Vec::new(), device,
        };
        for _ in 0..n_slots { pool.grow()?; }
        Ok(pool)
    }

    fn allocate_slab(&self, nbytes: u64) -> Result<SlabStorage, BlockError> {
        let n = usize::try_from(nbytes).map_err(|_| BlockError::Overflow)?;
        match &self.device {
            None => Ok(SlabStorage::Host(vec![0u8; n])),
            Some(dev) => {
                let buf = dev.newBufferWithLength_options(n, MTLResourceOptions::StorageModeShared).expect("block pool buffer");
                // Reuse fills a slot again. A fresh buffer is zero too, so a read before `take` is finite.
                unsafe { std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, n); }
                Ok(SlabStorage::Gpu(buf))
            }
        }
    }

    /// Grow the slab that owns `index`, or open the next one. Does not move a slot already placed.
    fn ensure_slot(&mut self, index: usize) -> Result<(), BlockError> {
        let slab_i = index / self.slots_per_slab;
        if slab_i >= MAX_SLABS { return Err(BlockError::Slab); }
        let within = index % self.slots_per_slab;
        let count = (within as u64).checked_add(1).ok_or(BlockError::Overflow)?;
        let need = count.checked_mul(self.stride).ok_or(BlockError::Overflow)?;
        if slab_i == self.slabs.len() {
            self.slabs.push(self.allocate_slab(need)?);
            return Ok(());
        }
        if slab_i + 1 != self.slabs.len() { return Err(BlockError::Slab); }
        self.resize_slab(slab_i, need)
    }

    fn resize_slab(&mut self, slab_i: usize, nbytes: u64) -> Result<(), BlockError> {
        let n = usize::try_from(nbytes).map_err(|_| BlockError::Overflow)?;
        if let SlabStorage::Host(v) = &mut self.slabs[slab_i] {
            if v.len() < n { v.resize(n, 0); }
            return Ok(());
        }
        let old_len = match &self.slabs[slab_i] {
            SlabStorage::Gpu(b) => b.length(),
            SlabStorage::Host(_) => return Ok(()),
        };
        if old_len >= n { return Ok(()); }
        let dev = self.device.as_ref().expect("gpu slab without a device");
        let new_buf = dev.newBufferWithLength_options(n, MTLResourceOptions::StorageModeShared).expect("block pool buffer");
        if let SlabStorage::Gpu(old) = &self.slabs[slab_i] {
            unsafe {
                std::ptr::copy_nonoverlapping(old.contents().as_ptr() as *const u8, new_buf.contents().as_ptr() as *mut u8, old_len);
                std::ptr::write_bytes((new_buf.contents().as_ptr() as *mut u8).add(old_len), 0, n - old_len);
            }
        }
        self.slabs[slab_i] = SlabStorage::Gpu(new_buf);
        Ok(())
    }

    pub fn kind(&self) -> BlockKind { self.kind }
    pub fn kvb(&self) -> u32 { self.kvb }
    pub fn stride_bytes(&self) -> u64 { self.stride }
    pub fn len(&self) -> usize { self.slots.len() }
    pub fn slab_count(&self) -> usize { self.slabs.len() }

    fn index(&self, slot: Slot) -> Result<usize, BlockError> {
        let i = slot.pool_index().get() as usize;
        if i >= self.slots.len() { Err(BlockError::OutOfRange) } else { Ok(i) }
    }

    /// Which slab holds `slot`. Slots in one slab share one allocation.
    pub fn slab_index(&self, slot: Slot) -> Result<usize, BlockError> {
        Ok(self.index(slot)? / self.slots_per_slab)
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
        mul_offset(u64::from(slot.pool_index().get()), self.stride)
    }

    /// The slab buffer that holds this slot. `None` on a host pool.
    /// Two slots in one slab return the same buffer. Its length is the bytes placed
    /// in that slab, not the stride of one slot.
    pub fn gpu_buffer(&self, slot: Slot) -> Result<Option<&crate::metal::Buf>, BlockError> {
        let i = self.index(slot)?;
        let slab = i / self.slots_per_slab;
        Ok(match &self.slabs[slab] { SlabStorage::Gpu(b) => Some(b), SlabStorage::Host(_) => None })
    }

    /// The slot's bytes: one stride inside the owning slab.
    pub fn slot_bytes(&self, slot: Slot) -> Result<&[u8], BlockError> {
        let i = self.index(slot)?;
        Ok(self.bytes_of(i))
    }

    fn slot_range(&self, i: usize) -> (usize, usize, usize) {
        let slab = i / self.slots_per_slab;
        let start = (i % self.slots_per_slab) * (self.stride as usize);
        (slab, start, self.stride as usize)
    }

    fn bytes_of(&self, i: usize) -> &[u8] {
        let (slab, start, n) = self.slot_range(i);
        match &self.slabs[slab] {
            SlabStorage::Host(v) => &v[start..start + n],
            SlabStorage::Gpu(b) => unsafe {
                std::slice::from_raw_parts((b.contents().as_ptr() as *const u8).add(start), n)
            },
        }
    }

    fn bytes_of_mut(&mut self, i: usize) -> &mut [u8] {
        let (slab, start, n) = self.slot_range(i);
        match &mut self.slabs[slab] {
            SlabStorage::Host(v) => &mut v[start..start + n],
            // Shared storage. The CPU zero-fills and appends; no command buffer reads this yet.
            SlabStorage::Gpu(b) => unsafe {
                std::slice::from_raw_parts_mut((b.contents().as_ptr() as *mut u8).add(start), n)
            },
        }
    }

    /// Append one slot. Its offset is `len * stride`. Earlier offsets stay put.
    /// The new slot has generation 0 until `take`. A fifth slab is [`BlockError::Slab`].
    pub fn grow(&mut self) -> Result<Slot, BlockError> {
        if self.slots.len() >= u32::MAX as usize { return Err(BlockError::Overflow); }
        let slot = Slot::new((self.slots.len() as u32).checked_add(1).ok_or(BlockError::Overflow)?)?;
        let _ = mul_offset(u64::from(slot.pool_index().get()), self.stride)?;
        self.ensure_slot(slot.pool_index().get() as usize)?;
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
        mul_offset(u64::from(slot.pool_index().get()), self.stride)
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
        let width = (self.stride / BLOCK as u64) as usize;
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

    /// GPU pools. `n_window` and `n_full` are allocated now. `kvb` is 1 or 2 and sizes
    /// both kinds. Zero slots still holds the device and grows through [`BlockPool::grow`].
    pub fn gpu(dev: &ProtocolObject<dyn MTLDevice>, n_window: usize, n_full: usize, kvb: u32) -> BlockPools {
        BlockPools {
            window: BlockPool::gpu(dev, BlockKind::Window, n_window, kvb),
            full: BlockPool::gpu(dev, BlockKind::Full, n_full, kvb),
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
        use objc2_metal::{MTLBuffer, MTLDevice};
        let dev = objc2_metal::MTLCreateSystemDefaultDevice().expect("metal device");
        let mut pools = BlockPools::gpu(&dev, 1, 1, KVB_FP16);
        for pool in [&mut pools.full, &mut pools.window] {
            let slot = Slot::new(1).unwrap();
            let buf = pool.gpu_buffer(slot).unwrap().expect("gpu buffer");
            assert_eq!(buf.length(), pool.stride_bytes() as usize);
            pool.slot_bytes_mut_for_test(slot).fill(0xef);
            assert_eq!(pool.take(slot).unwrap(), 0);
            assert!(pool.slot_bytes(slot).unwrap().iter().all(|&b| b == 0));
            assert_eq!(pool.record(slot).unwrap().generation(), 1);
            let grown = pool.grow().unwrap();
            assert_eq!(grown.get(), 2);
            assert_eq!(pool.offset(grown).unwrap(), pool.stride_bytes());
            assert_eq!(pool.offset(slot).unwrap(), 0);
            let share = 2 * pool.stride_bytes() <= dev.maxBufferLength() as u64;
            if share {
                assert_eq!(pool.slab_index(slot).unwrap(), pool.slab_index(grown).unwrap());
                let slab = pool.gpu_buffer(grown).unwrap().expect("slab");
                assert_eq!(slab.length(), 2 * pool.stride_bytes() as usize);
                assert_eq!(pool.gpu_buffer(slot).unwrap().unwrap().contents().as_ptr(), slab.contents().as_ptr());
            } else {
                assert_eq!(pool.slab_index(grown).unwrap(), 1);
            }
        }
        assert_ne!(pools.window.stride_bytes(), pools.full.stride_bytes());
    }

    #[test]
    fn two_slots_share_a_slab_and_a_third_does_not_alias() {
        let stride = FULL_STRIDE;
        let mut pool = BlockPool::open(BlockKind::Full, 0, KVB_FP16, stride * 2, None).unwrap();
        let a = pool.grow().unwrap();
        let b = pool.grow().unwrap();
        assert_eq!(pool.slab_count(), 1);
        assert_eq!(pool.slab_index(a).unwrap(), 0);
        assert_eq!(pool.slab_index(b).unwrap(), 0);
        pool.take(a).unwrap();
        pool.take(b).unwrap();
        pool.slot_bytes_mut_for_test(a)[0] = 0x11;
        pool.slot_bytes_mut_for_test(b)[0] = 0x22;
        assert_eq!(pool.slot_bytes(a).unwrap()[0], 0x11);
        assert_eq!(pool.slot_bytes(b).unwrap()[0], 0x22);
        let pa = pool.slot_bytes(a).unwrap().as_ptr() as usize;
        let pb = pool.slot_bytes(b).unwrap().as_ptr() as usize;
        assert_eq!(pb - pa, stride as usize);
        let c = pool.grow().unwrap();
        assert_eq!(pool.slab_index(c).unwrap(), 1);
        assert_eq!(pool.slab_count(), 2);
        assert_eq!(pool.offset(c).unwrap(), 2 * stride);
        pool.take(c).unwrap();
        pool.slot_bytes_mut_for_test(c)[0] = 0x33;
        assert_eq!(pool.slot_bytes(a).unwrap()[0], 0x11);
        assert_eq!(pool.slot_bytes(b).unwrap()[0], 0x22);
        assert_eq!(pool.slot_bytes(c).unwrap()[0], 0x33);
        assert_ne!(pool.slot_bytes(c).unwrap().as_ptr(), pool.slot_bytes(a).unwrap().as_ptr());
    }

    #[test]
    fn a_fifth_slab_is_refused() {
        let mut pool = BlockPool::open(BlockKind::Full, 0, KVB_FP16, FULL_STRIDE, None).unwrap();
        for i in 1..=4 {
            let slot = pool.grow().unwrap();
            assert_eq!(slot.get(), i);
            assert_eq!(pool.slab_index(slot).unwrap(), (i - 1) as usize);
        }
        assert_eq!(pool.slab_count(), 4);
        assert_eq!(pool.grow(), Err(BlockError::Slab));
        assert_eq!(pool.len(), 4);
        assert_eq!(BlockPool::open(BlockKind::Full, 0, KVB_FP16, FULL_STRIDE - 1, None).unwrap_err(), BlockError::Slab);
    }

    #[test]
    fn kv8_pool_uses_half_the_fp16_stride() {
        let pool = BlockPool::open(BlockKind::Full, 2, 1, u64::MAX, None).unwrap();
        assert_eq!(pool.kvb(), 1);
        assert_eq!(pool.stride_bytes(), FULL_STRIDE / 2);
        assert_eq!(pool.slab_count(), 1);
        assert_eq!(pool.offset(Slot::new(2).unwrap()).unwrap(), FULL_STRIDE / 2);
        assert_eq!(pool.slot_bytes(Slot::new(1).unwrap()).unwrap().len(), (FULL_STRIDE / 2) as usize);
        assert_eq!(stride_at(BlockKind::Window, 2).unwrap(), WINDOW_STRIDE);
        assert_eq!(stride_at(BlockKind::Window, 1).unwrap(), WINDOW_STRIDE / 2);
        assert_eq!(stride_at(BlockKind::Full, 0), Err(BlockError::Kvb));
        assert_eq!(stride_at(BlockKind::Full, 3), Err(BlockError::Kvb));
        assert!(BlockPool::open(BlockKind::Window, 1, 4, u64::MAX, None).is_err());
    }

    #[test]
    fn bump_generation_advances_and_leaves_a_live_reference() {
        let mut slot = SlotRecord::fresh(BlockKind::Full);
        slot.set_length(64).unwrap();
        slot.seal();
        slot.add_ref().unwrap();
        slot.add_ref().unwrap();
        slot.bump_generation().unwrap();
        assert_eq!((slot.generation(), slot.length(), slot.sealed(), slot.refs()), (1, 64, true, 2));
        let mut max_slot = SlotRecord::at_generation(BlockKind::Window, u32::MAX);
        max_slot.set_length(32).unwrap();
        max_slot.seal();
        max_slot.add_ref().unwrap();
        assert_eq!(max_slot.bump_generation(), Err(BlockError::Generation));
        assert_eq!((max_slot.generation(), max_slot.length(), max_slot.sealed(), max_slot.refs()), (u32::MAX, 32, true, 1));
    }
}
