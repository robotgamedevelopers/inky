//! Layer 3 block cache, the types the rest of the layer builds on.
//!
//! The GPU buffer pool and the attention read through the table are not in this file.
//! The numbers match `specs/BlockLayout.tla`: block size `B`, window `W` from the design
//! (1024 positions), slot offsets with high limb 2 and stride 4096, and the span pin bound.

/// Positions in one block. `BlockLayout`'s `B`.
pub const BLOCK: usize = 64;
/// Sliding-window positions a start point keeps. The design's window.
pub const WINDOW: usize = 1024;
/// `BlockLayout`'s `BaseHigh`. A high limb of 2 is not a 32-bit offset.
pub const OFFSET_BASE_HI: u32 = 2;
/// `BlockLayout`'s `Stride`.
pub const OFFSET_STRIDE: u32 = 4096;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlockKind { Window, Full }

/// A 64-bit pool offset, split the way `BlockLayout`'s `Offset` is split.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlockOffset { pub hi: u32, pub lo: u32 }

impl BlockOffset {
    /// Slot numbers are 1-based, as in `BlockLayout`'s `Slots`.
    pub fn of_slot(slot: u32) -> BlockOffset {
        assert!(slot >= 1, "block slots are 1-based");
        BlockOffset { hi: OFFSET_BASE_HI, lo: slot.saturating_sub(1).saturating_mul(OFFSET_STRIDE) }
    }

    /// `BlockLayout`'s `AddBytes`. The high limb is unchanged.
    pub fn add_bytes(self, n: u32) -> BlockOffset {
        BlockOffset { hi: self.hi, lo: self.lo.wrapping_add(n) }
    }
}

/// One table entry. `None` from the table is `BlockLayout`'s `NoEntry`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlockEntry {
    pub slot: u32,
    pub generation: u32,
    pub offset: BlockOffset,
}

/// Block index of a position. Position 0 is block 0.
pub fn block_index(pos: usize) -> usize { pos / BLOCK }

/// How many window blocks a span of `c` tokens pins. `BlockLayout`'s `SpanBound`.
/// A span of 512 pins 25, not 17.
pub fn span_bound(c: usize) -> usize {
    (WINDOW - 1 + c + BLOCK - 1) / BLOCK + 1
}

/// The GPU buffer pool. This commit does not allocate it.
#[derive(Clone, Debug)]
pub struct BlockPool { capacity: usize }

impl BlockPool {
    pub fn new(capacity: usize) -> BlockPool { BlockPool { capacity } }
    pub fn capacity(&self) -> usize { self.capacity }
}

/// One lane's block table. This commit does not fill it, and attention does not read it.
#[derive(Clone, Debug)]
pub struct BlockTable { entries: Vec<Option<BlockEntry>> }

impl BlockTable {
    pub fn new() -> BlockTable { BlockTable { entries: Vec::new() } }

    pub fn get(&self, kind: BlockKind, block: usize) -> Option<BlockEntry> {
        let i = match kind {
            BlockKind::Full => block * 2,
            BlockKind::Window => block * 2 + 1,
        };
        self.entries.get(i).copied().flatten()
    }
}

impl Default for BlockTable {
    fn default() -> Self { BlockTable::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_offset_matches_block_layout() {
        assert_eq!(BlockOffset::of_slot(1), BlockOffset { hi: 2, lo: 0 });
        assert_eq!(BlockOffset::of_slot(3), BlockOffset { hi: 2, lo: 8192 });
        assert_eq!(BlockOffset::of_slot(1).add_bytes(16), BlockOffset { hi: 2, lo: 16 });
    }

    #[test]
    fn span_of_512_pins_25_window_blocks() {
        assert_eq!(block_index(0), 0);
        assert_eq!(block_index(63), 0);
        assert_eq!(block_index(64), 1);
        assert_eq!(span_bound(512), 25);
        assert_ne!(span_bound(512), 17);
    }

    #[test]
    fn empty_table_is_no_entry() {
        let t = BlockTable::new();
        assert_eq!(t.get(BlockKind::Window, 0), None);
        assert_eq!(t.get(BlockKind::Full, 4), None);
        assert_eq!(BlockPool::new(4).capacity(), 4);
    }
}
