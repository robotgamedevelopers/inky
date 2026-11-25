//! Layer 3 block cache, the types the rest of the layer builds on.
#![allow(dead_code)] // the pool, the table and the prefix tree are the callers; cargo check does not compile the unit tests
//!
//! A block is 64 positions. Inside it the order is `[pass][K|V][head][pos][HD]`,
//! one element `kvb` bytes (DESIGN.md). At `kvb = 2` a window block is 10 MiB and
//! a full block is 2 MiB; at `kvb = 1` (`--kv8`) each is half of that.
//! Each kind has its own pool, so an offset is a plain `u64` byte offset in that
//! pool: base 0 plus the 0-based pool index times that kind's stride at this `kvb`.
//! One contiguous allocation holds the kind until a new slot would pass the slab cap
//! (the device `maxBufferLength` on GPU), when the pool opens another slab (up to 4).
//! GPU block-table words come from [`BlockPool::gpu_table_entry`]: `(slab << 62) | within_slab_offset`,
//! and [`key_row_at`] adds the pass, K or V, head and position inside the block.
//! `BlockLayout`'s high limb of 2 is TLC's trap for a 32-bit truncation, not a base
//! this code uses. The in-block order is `InBlock.tla`.
//!
//! A generation does not wrap. The next generation after `u32::MAX` is refused.

use std::fmt;

#[cfg(not(standalone_blocks))]
use objc2::rc::Retained;
#[cfg(not(standalone_blocks))]
use objc2::runtime::ProtocolObject;
#[cfg(not(standalone_blocks))]
use objc2::Message;
#[cfg(not(standalone_blocks))]
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

/// Positions in one block.
pub const BLOCK: usize = 64;
/// Sliding-window positions a start point keeps.
pub const WINDOW: usize = 1024;
/// Bytes in one full-attention block at fp16 (`kvb = 2`). Eight passes, 2 MiB.
pub const FULL_STRIDE: u64 = 2 * 1024 * 1024;
/// Bytes in one sliding-window block at fp16 (`kvb = 2`). Forty passes, 10 MiB.
pub const WINDOW_STRIDE: u64 = 10 * 1024 * 1024;
/// fp16 element width. `--kv8` passes 1.
pub const KVB_FP16: u32 = 2;
/// A kind holds at most this many slabs. One more is [`BlockError::Slab`].
pub const MAX_SLABS: usize = 4;
/// Bit shift of the 2-bit slab selector in a 64-bit GPU block-table word (`entry >> 62`).
pub const KV_SLAB_SHIFT: u32 = 62;
/// 62-bit within-slab byte-offset mask (`((1UL << 62) - 1UL)` in `src/kernels.metal`).
pub const KV_SLAB_MASK: u64 = (1u64 << KV_SLAB_SHIFT) - 1;
/// Metal buffer argument slots bound to `KV0, KV1, KV2, KV3` across all Layer 3 kernels.
pub const KV_SLAB_BUFFER_INDICES: [usize; MAX_SLABS] = [3, 10, 11, 12];
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

    pub fn stride(self) -> u64 {
        match self {
            BlockKind::Window => WINDOW_STRIDE,
            BlockKind::Full => FULL_STRIDE,
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
            BlockError::Kvb => write!(f, "block kvb must be 1 or 2"),
            BlockError::Slab => write!(f, "block slab cap exceeded"),
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

/// Which plane of a pass. Key is first, Value is second, matching layer 2's `kv_slot`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kv {
    Key,
    Value,
}

impl Kv {
    #[cfg(any(test, standalone_blocks))]
    fn index(self) -> u32 {
        match self {
            Kv::Key => 0,
            Kv::Value => 1,
        }
    }
}

/// Byte offset of a slot in its kind's pool at element width `kvb`.
/// Refuses a product that does not fit in `u64`, and a `kvb` other than 1 or 2.
pub fn byte_offset(kind: BlockKind, slot: Slot, kvb: u32) -> Result<u64, BlockError> {
    mul_offset(u64::from(slot.pool_index().get()), kind.stride_bytes(kvb)?)
}

/// Byte offset of one key or value row inside a slot.
/// Order is `[pass][K|V][head][pos in the block][HD]`, each element `kvb` bytes,
/// so one head's 64 keys are contiguous. `base` is the slot's `byte_offset`.
/// A pass, head or position outside the block is `OutOfRange`.
#[cfg(any(test, standalone_blocks))]
pub fn key_row(base: u64, kind: BlockKind, kvb: u32, pass: u32, kv: Kv, head: u32, index: u32) -> Result<u64, BlockError> {
    key_row_at(base, kind.passes() as u32, NKV as u32, BLOCK as u32, HD as u32, kvb, pass, kv.index(), head, index)
}

/// Byte offset of one key or value row inside a slot at explicit bounds.
/// Order is `[pass][K|V][head][pos in the block][hd]`, each element `kvb` bytes,
/// matching `kv_blk_ptr` in `src/kernels.metal` and `src/prefill.metal`.
/// The InBlock trace calls it at the spec's bounds (2 passes, 3 heads, 4 positions, `HD` 2).
pub fn key_row_at(base: u64, passes: u32, heads: u32, block: u32, hd: u32, kvb: u32, pass: u32, plane: u32, head: u32, index: u32) -> Result<u64, BlockError> {
    if kvb != 1 && kvb != 2 { return Err(BlockError::Width); }
    if pass >= passes || plane >= 2 || head >= heads || index >= block {
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

/// Decode a 64-bit GPU block-table word into `(slab_index, within_slab_byte_offset)`,
/// matching `kv_blk_ptr` (`uint(entry >> 62)` and `entry & KV_SLAB_MASK`) in `src/kernels.metal`.
#[inline]
pub fn decode_gpu_table_entry(entry: u64) -> (usize, u64) {
    ((entry >> KV_SLAB_SHIFT) as usize, entry & KV_SLAB_MASK)
}

/// Logical window block indices `[oldest / block ..= end / block]` required across a span
/// `[pos, pos + len)` at block size `block` and sliding-window width `window` (`U170`,
/// bound by trace to `specs/SpanPins.tla`).
pub fn span_window_blocks_at(
    pos: usize,
    len: usize,
    block: usize,
    window: usize,
) -> Result<Vec<BlockIndex>, BlockError> {
    if len == 0 || block == 0 || window == 0 {
        return Err(BlockError::OutOfRange);
    }
    let end = pos.checked_add(len - 1).ok_or(BlockError::Overflow)?;
    let oldest = pos.saturating_sub(window - 1);
    let lo_blk = u32::try_from(oldest / block).map_err(|_| BlockError::Overflow)?;
    let hi_blk = u32::try_from(end / block).map_err(|_| BlockError::Overflow)?;
    Ok((lo_blk..=hi_blk).map(BlockIndex::new).collect()) // MUTANT_ANCHOR_SPAN_WIN_BLOCKS
}

/// Production `(BLOCK = 64, WINDOW = 1024)` window blocks required across `[pos, pos + len)`.
pub fn span_window_blocks(pos: usize, len: usize) -> Result<Vec<BlockIndex>, BlockError> {
    span_window_blocks_at(pos, len, BLOCK, WINDOW)
}

/// KV layout constants and shader invariants extracted directly from `src/kernels.metal`
/// and `src/prefill.metal` (`U63` & `U18`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MetalKvLayout {
    pub nkv: u32,
    pub hd: u32,
    pub kv_blk: u32,
    pub kv_blk_mask: u32,
    pub planes_per_pass: u64,
    pub slab_shift: u32,
    pub slab_buffers: [usize; MAX_SLABS],
}

fn parse_c_define_u32(src: &str, name: &str) -> Result<u32, &'static str> {
    for line in src.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("#define") {
            let mut it = rest.split_whitespace();
            if it.next() == Some(name) {
                let tok = it.next().ok_or("missing #define value")?;
                let digits = tok.trim_end_matches('u').trim_end_matches("UL");
                return digits.parse::<u32>().map_err(|_| "invalid integer in #define");
            }
        }
    }
    Err("missing #define in Metal source")
}

/// Parse and verify the Layer 3 KV block address macros, `kv_blk_ptr` 64-bit word decode,
/// and `KV0..KV3` buffer-index bindings across `src/kernels.metal` and `src/prefill.metal`.
pub fn parse_metal_kv_layout(kernels_src: &str, prefill_src: &str) -> Result<MetalKvLayout, &'static str> {
    let nkv = parse_c_define_u32(kernels_src, "NKV")?;
    let hd = parse_c_define_u32(kernels_src, "HD")?;
    let kv_blk = parse_c_define_u32(kernels_src, "KV_BLK")?;
    let kv_blk_mask = parse_c_define_u32(kernels_src, "KV_BLK_MASK")?;
    if kv_blk == 0 || !kv_blk.is_power_of_two() || kv_blk_mask != kv_blk - 1 {
        return Err("KV_BLK / KV_BLK_MASK inconsistent");
    }
    if !kernels_src.contains("typedef int8_t kv_t;") || !kernels_src.contains("typedef half kv_t;") {
        return Err("missing kv_t KV8/fp16 typedefs");
    }
    if !kernels_src.contains("#define KV_SLAB_MASK ((1UL << 62) - 1UL)") {
        return Err("missing or altered KV_SLAB_MASK");
    }
    if !kernels_src.contains("#define KV_HEAD_BYTES  (ulong(KV_BLK) * ulong(HD) * ulong(sizeof(kv_t)))") {
        return Err("missing or altered KV_HEAD_BYTES");
    }
    if !kernels_src.contains("#define KV_PLANE_BYTES (ulong(NKV) * KV_HEAD_BYTES)") {
        return Err("missing or altered KV_PLANE_BYTES");
    }
    let planes_per_pass = if kernels_src.contains("#define KV_PASS_BYTES  (2UL * KV_PLANE_BYTES)") {
        2u64
    } else {
        return Err("missing or altered KV_PASS_BYTES");
    };
    for needle in [
        "const uint si = uint(entry >> 62);",
        "const ulong off = entry & KV_SLAB_MASK;",
        "device uchar * base = (si == 0) ? s0 : (si == 1 ? s1 : (si == 2 ? s2 : s3));",
        "return (device kv_t *) (base + off + in_slot_off);",
    ] {
        if !kernels_src.contains(needle) {
            return Err("kv_blk_ptr formula altered");
        }
    }
    let check_kernel = |src: &str, kname: &str| -> Result<(), &'static str> {
        let anchor = format!("kernel void {kname}(");
        let idx = src.find(&anchor).ok_or("missing Layer 3 Metal kernel")?;
        let rest = &src[idx..];
        let body_end = rest[1..].find("\nkernel void ").map_or(rest.len(), |p| p + 1);
        let ksrc = &rest[..body_end];
        for bind in [
            "KV0 [[buffer(3)]]",
            "KV1 [[buffer(10)]]",
            "KV2 [[buffer(11)]]",
            "KV3 [[buffer(12)]]",
        ] {
            if !ksrc.contains(bind) {
                return Err("Layer 3 kernel missing canonical KV0..KV3 buffer(3,10,11,12) binding");
            }
        }
        if !ksrc.contains("* KV_PASS_BYTES + ulong(g) * KV_HEAD_BYTES;")
            || !ksrc.contains("const ulong voff = koff + KV_PLANE_BYTES;")
            || !ksrc.contains("kv_blk_ptr(KV0, KV1, KV2, KV3,")
        {
            return Err("Layer 3 kernel missing canonical koff/voff/kv_blk_ptr address formula");
        }
        Ok(())
    };
    for kname in [
        "k_kvprep_dec",
        "k_attn_dec_solo",
        "k_attn_grp_dec",
    ] {
        check_kernel(kernels_src, kname)?;
    }
    for kname in [
        "k_kvprep_b_blk",
        "k_attn_b_blk",
        "k_attn_bt_blk",
        "k_attn_h_blk",
    ] {
        check_kernel(prefill_src, kname)?;
    }
    Ok(MetalKvLayout {
        nkv,
        hd,
        kv_blk,
        kv_blk_mask,
        planes_per_pass,
        slab_shift: 62,
        slab_buffers: KV_SLAB_BUFFER_INDICES,
    })
}

/// Evaluate the exact Metal shader within-slot byte address (`koff`/`voff + (pos & KV_BLK_MASK) * HD * sizeof(kv_t)`)
/// using a parsed [`MetalKvLayout`].
pub fn metal_in_slot_byte_offset_from_layout(
    layout: &MetalKvLayout,
    kind: BlockKind,
    kvb: u32,
    pass: u32,
    kv: Kv,
    head: u32,
    pos: u32,
) -> Result<u64, BlockError> {
    if kvb != 1 && kvb != 2 {
        return Err(BlockError::Width);
    }
    if pass >= kind.passes() as u32 || head >= layout.nkv {
        return Err(BlockError::OutOfRange);
    }
    let elem = u64::from(kvb);
    let kv_head_bytes = u64::from(layout.kv_blk) * u64::from(layout.hd) * elem;
    let kv_plane_bytes = u64::from(layout.nkv) * kv_head_bytes;
    let kv_pass_bytes = layout.planes_per_pass * kv_plane_bytes;
    let koff = u64::from(pass) * kv_pass_bytes + u64::from(head) * kv_head_bytes;
    let plane_off = if kv == Kv::Value { koff + kv_plane_bytes } else { koff };
    let ib = u64::from(pos & layout.kv_blk_mask);
    Ok(plane_off + ib * u64::from(layout.hd) * elem)
}

/// Evaluate the Metal shader within-slot byte address directly from `src/kernels.metal` & `src/prefill.metal`.
#[cfg(not(standalone_blocks))]
pub fn metal_in_slot_byte_offset(
    kind: BlockKind,
    kvb: u32,
    pass: u32,
    kv: Kv,
    head: u32,
    pos: u32,
) -> Result<u64, BlockError> {
    let layout = parse_metal_kv_layout(
        include_str!("kernels.metal"),
        include_str!("prefill.metal"),
    )
    .expect("src/kernels.metal and src/prefill.metal KV layout must remain valid");
    metal_in_slot_byte_offset_from_layout(&layout, kind, kvb, pass, kv, head, pos)
}

/// Write the InBlock trace TLC reads: small bounds (`Trace2`, `Trace1`) from `key_row_at`,
/// plus production Metal shader traces (`MetalFull2`, `MetalFull1`, `MetalWin2`, `MetalWin1`)
/// across `Full` (8 passes) and `Window` (40 passes) blocks at `kvb = 2` and `kvb = 1` (`U63`).
pub fn write_inblock_trace(path: &str) -> std::io::Result<()> {
    let mut out = String::from(
        "---------------------------- MODULE InBlockTraceData ------------------------\n\
         \\* Written by `inky inblock-trace` from key_row_at and Metal KV macros.\n\
         \\* TLC checks these addresses against SmallFormula and ProdFormula.\n\
         \\* Do not edit this file by hand.\n\
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
    let prod_heads = [0u32, 1, 7];
    let prod_positions = [0u32, 1, 31, 32, 63];
    for (name, kind, kvb) in [
        ("MetalFull2", BlockKind::Full, 2u32),
        ("MetalFull1", BlockKind::Full, 1u32),
        ("MetalWin2", BlockKind::Window, 2u32),
        ("MetalWin1", BlockKind::Window, 1u32),
    ] {
        out.push_str(&format!("{name} == <<\n"));
        let mut first = true;
        for p in 0..(kind.passes() as u32) {
            for (k_idx, kv_plane) in [(0u32, Kv::Key), (1u32, Kv::Value)] {
                for &h in &prod_heads {
                    for &i in &prod_positions {
                        #[cfg(not(standalone_blocks))]
                        let addr = metal_in_slot_byte_offset(kind, kvb, p, kv_plane, h, i)
                            .expect("production Metal KV bounds in range");
                        #[cfg(standalone_blocks)]
                        let addr = key_row(0, kind, kvb, p, kv_plane, h, i)
                            .expect("production KV bounds in range");
                        if !first { out.push_str(",\n"); }
                        first = false;
                        out.push_str(&format!("  <<{p}, {k_idx}, {h}, {i}, {addr}>>"));
                    }
                }
            }
        }
        out.push_str("\n>>\n\n");
    }
    out.push_str("=============================================================================\n");
    std::fs::write(path, out)
}

/// Bytes of one slot at element width `kvb` (`1` for `--kv8`, `2` for fp16).
/// fp16 (`kvb == 2`) equals [`BlockKind::stride`]; [`byte_offset`] scales by `stride_bytes(kvb)`.
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
    (WINDOW - 1 + c).div_ceil(BLOCK) + 1
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

    fn unseal(&mut self) { self.sealed = false; }

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

/// One slab: a contiguous allocation holding one or more slots of this kind.
/// A host slab is a `Vec`. A GPU slab is one shared Metal buffer.
#[derive(Clone)]
enum SlabStorage {
    Host(Vec<u8>),
    #[cfg(not(standalone_blocks))]
    Gpu(crate::metal::Buf),
}

/// One kind's pool. Slots are append-only: `grow` adds a slot at the end and does not
/// move an existing offset. The bytes live in one slab until a new slot would pass
/// `maxBufferLength` (or the host cap). At most [`MAX_SLABS`].
pub struct BlockPool {
    kind: BlockKind,
    /// Element width for every slot in this pool. 2 is fp16, 1 is `--kv8`.
    kvb: u32,
    stride: u64,
    slots_per_slab: usize,
    slots: Vec<SlotRecord>,
    gpu_zero_pending: Vec<bool>,
    slabs: Vec<SlabStorage>,
    /// Present when this pool allocates Metal buffers.
    #[cfg(not(standalone_blocks))]
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
            gpu_zero_pending: self.gpu_zero_pending.clone(),
            // A cloned GPU buffer is another retain of the same allocation.
            slabs: self.slabs.clone(),
            #[cfg(not(standalone_blocks))]
            device: self.device.clone(),
        }
    }
}

impl fmt::Debug for BlockPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[cfg(not(standalone_blocks))]
        let is_gpu = self.device.is_some();
        #[cfg(standalone_blocks)]
        let is_gpu = false;
        f.debug_struct("BlockPool")
            .field("kind", &self.kind)
            .field("kvb", &self.kvb)
            .field("len", &self.slots.len())
            .field("slabs", &self.slabs.len())
            .field("gpu", &is_gpu)
            .finish()
    }
}

/// Sentinel GPU 64-bit block-table entry written for unmapped/cleared table slots.
/// Distinct from every valid `(slab_index << 62) | byte_offset` word, including `(slab = 0, Slot(1))`
/// which encodes as `0`, because every valid `byte_offset` is stride-aligned (`>= 1 MiB`) and `< (1 << 62)`.
pub const CLEARED_GPU_TABLE_ENTRY: u64 = u64::MAX;

impl BlockPool {
    /// Host bytes at fp16 (`kvb` 2), one contiguous slab, zero-filled. No Metal device.
    pub fn new(kind: BlockKind, n_slots: usize) -> BlockPool {
        Self::open_host(kind, n_slots, KVB_FP16, u64::MAX).expect("host block pool")
    }

    /// One element width for the whole pool (host storage). `kvb` other than 1 or 2 is `Width`.
    pub fn with_kvb(kind: BlockKind, n_slots: usize, kvb: u32) -> Result<BlockPool, BlockError> {
        let _ = kind.stride_bytes(kvb)?;
        Self::open_host(kind, n_slots, kvb, u64::MAX)
    }

    /// Metadata-only pool for high-slot offset/table tests without multi-GiB host allocation.
    #[cfg(test)]
    pub(crate) fn records_only(kind: BlockKind, n_slots: usize, kvb: u32) -> Result<BlockPool, BlockError> {
        let stride = kind.stride_bytes(kvb)?;
        let slots_per_slab = slots_in_slab(stride, u64::MAX)?;
        Ok(BlockPool {
            kind, kvb, stride, slots_per_slab,
            slots: vec![SlotRecord::fresh(kind); n_slots],
            gpu_zero_pending: vec![false; n_slots],
            slabs: Vec::new(),
            #[cfg(not(standalone_blocks))]
            device: None,
        })
    }

    /// GPU pools for this kind. `kvb` is 1 or 2. One buffer until a slot would pass
    /// the device `maxBufferLength`, then another slab, and never a fifth.
    #[cfg(not(standalone_blocks))]
    pub fn gpu(dev: &ProtocolObject<dyn MTLDevice>, kind: BlockKind, n_slots: usize, kvb: u32) -> BlockPool {
        let max = dev.maxBufferLength() as u64;
        Self::open(kind, n_slots, kvb, max, Some(dev.retain())).expect("gpu block pool")
    }

    fn open_host(kind: BlockKind, n_slots: usize, kvb: u32, max_slab_bytes: u64) -> Result<BlockPool, BlockError> {
        #[cfg(not(standalone_blocks))]
        { Self::open(kind, n_slots, kvb, max_slab_bytes, None) }
        #[cfg(standalone_blocks)]
        {
            let stride = stride_at(kind, kvb)?;
            let slots_per_slab = slots_in_slab(stride, max_slab_bytes)?;
            let mut pool = BlockPool {
                kind,
                kvb,
                stride,
                slots_per_slab,
                slots: Vec::with_capacity(n_slots),
                gpu_zero_pending: Vec::with_capacity(n_slots),
                slabs: Vec::new(),
            };
            if n_slots > 0 { pool.grow_by(n_slots)?; }
            Ok(pool)
        }
    }

    #[cfg(not(standalone_blocks))]
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
            kind,
            kvb,
            stride,
            slots_per_slab,
            slots: Vec::with_capacity(n_slots),
            gpu_zero_pending: Vec::with_capacity(n_slots),
            slabs: Vec::new(),
            device,
        };
        if n_slots > 0 { pool.grow_by(n_slots)?; }
        Ok(pool)
    }

    fn allocate_slab(&self, nbytes: u64) -> Result<SlabStorage, BlockError> {
        let n = usize::try_from(nbytes).map_err(|_| BlockError::Overflow)?;
        #[cfg(not(standalone_blocks))]
        if let Some(dev) = &self.device {
            // Metal anonymous shared buffers arrive page-zeroed from the OS kernel; avoid
            // synchronously faulting and memset-ing multi-GiB slabs on the host thread.
            let buf = dev.newBufferWithLength_options(n, MTLResourceOptions::StorageModeShared).expect("block pool buffer");
            return Ok(SlabStorage::Gpu(buf));
        }
        Ok(SlabStorage::Host(vec![0u8; n]))
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
        #[allow(irrefutable_let_patterns)]
        if let SlabStorage::Host(v) = &mut self.slabs[slab_i] {
            if v.len() < n { v.resize(n, 0); }
            return Ok(());
        }
        #[cfg(not(standalone_blocks))]
        {
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
                }
            }
            self.slabs[slab_i] = SlabStorage::Gpu(new_buf);
        }
        Ok(())
    }

    pub fn kind(&self) -> BlockKind { self.kind }
    pub fn kvb(&self) -> u32 { self.kvb }
    pub fn stride_bytes(&self) -> u64 { self.stride }
    pub fn len(&self) -> usize { self.slots.len() }
    pub fn slab_count(&self) -> usize { self.slabs.len() }

    /// Fast `(usable_slots, reclaimable_slots)` scan over `self.slots` without per-slot
    /// `Slot::new` / `Result` wrapper overhead on large pools.
    pub fn count_usable_and_avail(&self, start_slot_refs: &[u32]) -> (usize, usize) {
        let mut usable = 0usize;
        let mut avail = 0usize;
        for (i, rec) in self.slots.iter().enumerate() {
            if rec.generation < u32::MAX {
                usable += 1;
                let held_by_starts = start_slot_refs.get(i).copied().unwrap_or(0);
                if rec.refs <= held_by_starts {
                    avail += 1;
                }
            }
        }
        (usable, avail)
    }

    pub fn has_gpu_storage(&self) -> bool {
        #[cfg(not(standalone_blocks))]
        {
            self.device.is_some()
        }
        #[cfg(standalone_blocks)]
        {
            false
        }
    }

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
        byte_offset(self.kind, slot, self.kvb)
    }

    /// The slab buffer that holds this slot. `None` on a host pool.
    /// Two slots in one slab return the same buffer. Its length is the bytes placed
    /// in that slab, not the stride of one slot.
    #[cfg(not(standalone_blocks))]
    pub fn gpu_buffer(&self, slot: Slot) -> Result<Option<&crate::metal::Buf>, BlockError> {
        let i = self.index(slot)?;
        let slab = i / self.slots_per_slab;
        Ok(match &self.slabs[slab] { SlabStorage::Gpu(b) => Some(b), SlabStorage::Host(_) => None })
    }

    /// Every allocated GPU slab buffer in slab order (0..slab_count).
    #[cfg(not(standalone_blocks))]
    pub fn gpu_slabs(&self) -> Vec<&crate::metal::Buf> {
        self.slabs.iter().filter_map(|s| match s { SlabStorage::Gpu(b) => Some(b), SlabStorage::Host(_) => None }).collect()
    }

    /// 64-bit GPU block-table word for `entry`: `(slab_index << 62) | within_slab_byte_offset`.
    /// Validates kind, width, generation, and live reference count before returning.
    /// For single-slab pools (`slab_index == 0`), this equals `entry.offset()`.
    pub fn gpu_table_entry(&self, entry: BlockEntry) -> Result<u64, BlockError> {
        if entry.kind() != self.kind { return Err(BlockError::OutOfRange); }
        if entry.kvb() != self.kvb { return Err(BlockError::Width); }
        let i = self.index(entry.slot())?;
        let rec = &self.slots[i];
        if rec.generation() != entry.generation() { return Err(BlockError::Stale); }
        if rec.refs() == 0 { return Err(BlockError::Refs); }
        let slab = i / self.slots_per_slab;
        if slab >= MAX_SLABS { return Err(BlockError::Slab); }
        let within = (i % self.slots_per_slab) as u64;
        let off = within.checked_mul(self.stride).ok_or(BlockError::Overflow)?;
        if off >= (1u64 << KV_SLAB_SHIFT) { return Err(BlockError::Overflow); }
        let word = ((slab as u64) << KV_SLAB_SHIFT) | off;
        assert_ne!(word, CLEARED_GPU_TABLE_ENTRY, "valid GPU block-table word collided with CLEARED_GPU_TABLE_ENTRY");
        Ok(word)
    }

    /// Owning GPU slab buffer, byte offset within that slab, and byte stride for `slot`.
    #[cfg(not(standalone_blocks))]
    pub fn gpu_slot_region(&self, slot: Slot) -> Result<Option<(&crate::metal::Buf, usize, usize)>, BlockError> {
        let i = self.index(slot)?;
        let (slab, start, n) = self.slot_range(i);
        Ok(match self.slabs.get(slab) {
            Some(SlabStorage::Gpu(b)) => Some((b, start, n)),
            _ => None,
        })
    }

    /// Raw CPU-visible pointer and byte length of `slot` inside its owning slab.
    /// Valid while nothing is in flight on the GPU and before the slab is resized.
    pub fn slot_raw_region(&self, slot: Slot) -> Option<(*mut u8, usize)> {
        let i = self.index(slot).ok()?;
        let (slab, start, n) = self.slot_range(i);
        match self.slabs.get(slab)? {
            SlabStorage::Host(v) => Some((unsafe { (v.as_ptr() as *mut u8).add(start) }, n)),
            #[cfg(not(standalone_blocks))]
            SlabStorage::Gpu(b) => Some((unsafe { (b.contents().as_ptr() as *mut u8).add(start) }, n)),
        }
    }

    /// The slot's bytes: one stride inside the owning slab.
    pub fn slot_bytes(&self, slot: Slot) -> Result<&[u8], BlockError> {
        let i = self.index(slot)?;
        Ok(self.bytes_of(i))
    }

    /// Mutable view of one slot's bytes (for CPU zeroing/poisoning when idle on GPU).
    pub fn slot_bytes_mut(&mut self, slot: Slot) -> Result<&mut [u8], BlockError> {
        let i = self.index(slot)?;
        Ok(self.bytes_of_mut(i))
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
            #[cfg(not(standalone_blocks))]
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
            #[cfg(not(standalone_blocks))]
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
        self.gpu_zero_pending.push(false);
        Ok(slot)
    }

    /// Append `count` fresh slots in one resize per touched slab, avoiding repeated copies.
    pub fn grow_by(&mut self, count: usize) -> Result<(), BlockError> {
        if count == 0 { return Ok(()); }
        let start = self.slots.len();
        let end = start.checked_add(count).ok_or(BlockError::Overflow)?;
        if end > u32::MAX as usize { return Err(BlockError::Overflow); }
        let last_idx = end - 1;
        let _ = mul_offset(last_idx as u64, self.stride)?;
        let mut idx = start;
        while idx <= last_idx {
            let slab_i = idx / self.slots_per_slab;
            let slab_end = ((slab_i + 1).saturating_mul(self.slots_per_slab) - 1).min(last_idx);
            self.ensure_slot(slab_end)?;
            idx = slab_end + 1;
        }
        self.slots.resize(end, SlotRecord::fresh(self.kind));
        self.gpu_zero_pending.resize(end, false);
        Ok(())
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
        if i < self.gpu_zero_pending.len() {
            self.gpu_zero_pending[i] = false;
        }
        mul_offset(u64::from(slot.pool_index().get()), self.stride)
    }

    /// Take `slot` without CPU-zeroing its multi-MiB backing storage on the host thread.
    /// On GPU pools, marks the slot for deferred GPU `k_zero` if a partial span reads it
    /// before all 64 positions are overwritten; callers restoring a full block clear the flag.
    pub fn take_unzeroed(&mut self, slot: Slot) -> Result<u64, BlockError> {
        let i = self.index(slot)?;
        if self.slots[i].refs() != 0 { return Err(BlockError::Refs); }
        let _ = next_generation(self.slots[i].generation())?;
        self.slots[i].bump_generation()?;
        self.slots[i].set_length(0)?;
        self.slots[i].unseal();
        if i < self.gpu_zero_pending.len() {
            self.gpu_zero_pending[i] = self.has_gpu_storage();
        }
        mul_offset(u64::from(slot.pool_index().get()), self.stride)
    }

    /// Atomically consume and return whether `slot` needs a GPU `k_zero` clear before a partial read.
    pub fn take_gpu_zero_pending(&mut self, slot: Slot) -> bool {
        let Ok(i) = self.index(slot) else { return false };
        if i < self.gpu_zero_pending.len() {
            std::mem::replace(&mut self.gpu_zero_pending[i], false)
        } else {
            false
        }
    }

    /// Clear `gpu_zero_pending` after a full-slot overwrite (such as SSD block restore).
    pub fn clear_gpu_zero_pending(&mut self, slot: Slot) {
        if let Ok(i) = self.index(slot) {
            if i < self.gpu_zero_pending.len() {
                self.gpu_zero_pending[i] = false;
            }
        }
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

    /// The lowest slot nothing references whose generation can still advance.
    pub fn lowest_free(&self) -> Option<Slot> {
        let i = self.slots.iter().position(|r| r.refs == 0 && r.generation < u32::MAX)?;
        Slot::new(u32::try_from(i + 1).ok()?).ok()
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
    #[cfg(not(standalone_blocks))]
    pub fn gpu(dev: &ProtocolObject<dyn MTLDevice>, n_window: usize, n_full: usize, kvb: u32) -> BlockPools {
        BlockPools {
            window: BlockPool::gpu(dev, BlockKind::Window, n_window, kvb),
            full: BlockPool::gpu(dev, BlockKind::Full, n_full, kvb),
        }
    }

    /// Both pools with `n_slots` each at fp16 (`kvb = 2`).
    pub fn new(n_slots: usize) -> BlockPools {
        BlockPools { window: BlockPool::new(BlockKind::Window, n_slots), full: BlockPool::new(BlockKind::Full, n_slots) }
    }

    /// Both pools with independent capacities at element width `kvb` (`1` or `2`).
    pub fn with_capacities(full_slots: usize, window_slots: usize, kvb: u32) -> Result<BlockPools, BlockError> {
        Ok(BlockPools {
            window: BlockPool::with_kvb(BlockKind::Window, window_slots, kvb)?,
            full: BlockPool::with_kvb(BlockKind::Full, full_slots, kvb)?,
        })
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
    /// Per-kind inclusive dirty block bounds `[lo, hi]` (`[0]` = Full, `[1]` = Window).
    dirty: [Option<(u32, u32)>; 2],
    /// When `true`, the full `0..blocks_per_kind` range for that kind must be synced to GPU.
    needs_full_sync: [bool; 2],
}

impl BlockTable {
    #[inline]
    fn kind_idx(kind: BlockKind) -> usize {
        match kind {
            BlockKind::Full => 0,
            BlockKind::Window => 1,
        }
    }

    #[inline]
    fn mark_dirty(&mut self, kind: BlockKind, block: u32) {
        let ki = Self::kind_idx(kind);
        if self.needs_full_sync[ki] {
            return;
        }
        self.dirty[ki] = Some(match self.dirty[ki] {
            Some((lo, hi)) => (lo.min(block), hi.max(block)),
            None => (block, block),
        });
    }

    /// Force the next GPU sync to rewrite all `0..blocks_per_kind` slots for both kinds.
    pub fn mark_full_sync(&mut self) {
        self.dirty = [None; 2];
        self.needs_full_sync = [true; 2];
    }

    pub fn needs_full_sync(&self, kind: BlockKind) -> bool {
        self.needs_full_sync[Self::kind_idx(kind)]
    }

    /// Inclusive dirty block bounds `[lo, hi]` for `kind`, or `None` when clean.
    pub fn dirty_range(&self, kind: BlockKind) -> Option<(u32, u32)> {
        if self.blocks_per_kind == 0 {
            return None;
        }
        let ki = Self::kind_idx(kind);
        if self.needs_full_sync[ki] {
            Some((0, self.blocks_per_kind - 1))
        } else {
            self.dirty[ki]
        }
    }

    /// Clear dirty tracking for `kind` after its GPU block table has been updated.
    pub fn clear_dirty(&mut self, kind: BlockKind) {
        let ki = Self::kind_idx(kind);
        self.dirty[ki] = None;
        self.needs_full_sync[ki] = false;
    }

    /// Synchronize only the dirty block range for `kind` into `dst` (length `>= blocks_per_kind`),
    /// writing `pool.gpu_table_entry(e)` for present entries and [`CLEARED_GPU_TABLE_ENTRY`]
    /// (`u64::MAX`, distinct from slab-0 `Slot(1)`'s `0`) for cleared entries, then
    /// clearing the dirty range. Returns the number of `u64` entries written (0 when clean).
    pub fn sync_dirty_to_slice(
        &mut self,
        kind: BlockKind,
        pool: &BlockPool,
        dst: &mut [u64],
    ) -> Result<usize, BlockError> {
        let n_blk = self.blocks_per_kind as usize;
        if dst.len() < n_blk {
            return Err(BlockError::OutOfRange);
        }
        let Some((lo, hi)) = self.dirty_range(kind) else {
            return Ok(0);
        };
        let lo = lo as usize;
        let hi = (hi as usize).min(n_blk.saturating_sub(1));
        let mut written = 0usize;
        for (b, slot_out) in dst.iter_mut().enumerate().take(hi + 1).skip(lo) {
            let bi = BlockIndex::new(b as u32);
            *slot_out = match self.get(kind, bi)? {
                Some(entry) => pool.gpu_table_entry(entry)?,
                None => CLEARED_GPU_TABLE_ENTRY,
            };
            written += 1;
        }
        self.clear_dirty(kind);
        Ok(written)
    }

    pub fn empty() -> BlockTable {
        BlockTable {
            blocks_per_kind: 0,
            entries: Vec::new(),
            dirty: [None; 2],
            needs_full_sync: [true; 2],
        }
    }

    pub fn new(blocks_per_kind: u32) -> Result<BlockTable, BlockError> {
        let n = usize::try_from(blocks_per_kind).ok().and_then(|n| n.checked_mul(2)).ok_or(BlockError::Overflow)?;
        Ok(BlockTable {
            blocks_per_kind,
            entries: vec![None; n],
            dirty: [None; 2],
            needs_full_sync: [true; 2],
        })
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
        if *cell != entry {
            *cell = entry;
            self.mark_dirty(kind, block.get());
        }
        Ok(())
    }

    /// Every present entry with its block, full before window within a block,
    /// blocks ascending.
    pub fn entries(&self) -> impl Iterator<Item = (BlockIndex, BlockEntry)> + '_ {
        self.entries.iter().enumerate().filter_map(|(i, e)| e.map(|e| (BlockIndex::new((i / 2) as u32), e)))
    }

    /// Resolve every `(BlockKind::Full, b, entry)` in `0..=(pos + len - 1) / BLOCK` and
    /// `(BlockKind::Window, b, entry)` in `span_window_blocks(pos, len)` that a step over
    /// `[pos, pos + len)` reads (`U170`, bound to `specs/SpanPins.tla`).
    pub fn pinned_span_entries(
        &self,
        pos: usize,
        len: usize,
    ) -> Result<Vec<(BlockKind, BlockIndex, BlockEntry)>, BlockError> {
        if len == 0 {
            return Err(BlockError::OutOfRange);
        }
        let end = pos.checked_add(len - 1).ok_or(BlockError::Overflow)?;
        let hi_full = u32::try_from(end / BLOCK).map_err(|_| BlockError::Overflow)?;
        let mut out = Vec::new();
        for b in 0..=hi_full {
            let bi = BlockIndex::new(b);
            let entry = self.get(BlockKind::Full, bi)?.ok_or(BlockError::NoEntry)?;
            out.push((BlockKind::Full, bi, entry));
        }
        for bi in span_window_blocks(pos, len)? {
            let entry = self.get(BlockKind::Window, bi)?.ok_or(BlockError::NoEntry)?;
            out.push((BlockKind::Window, bi, entry));
        }
        Ok(out)
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
        let new_entry = BlockEntry::new(slot, generation, offset, pool.kind(), kvb);
        let changed = prev != Some(new_entry);
        let cell = self.entries.get_mut(i).ok_or(BlockError::OutOfRange)?;
        *cell = Some(new_entry);
        if changed {
            self.mark_dirty(pool.kind(), block.get());
        }
        Ok(())
    }

    /// Clear one `(kind, block)` cell, dropping its pool reference if its generation still matches.
    pub fn clear_kind_entry(&mut self, pool: &mut BlockPool, block: BlockIndex) -> Result<(), BlockError> {
        let i = table_index(pool.kind(), block, self.blocks_per_kind)? as usize;
        if let Some(prev) = self.entries.get(i).copied().flatten() {
            if pool.record(prev.slot())?.generation() == prev.generation() {
                pool.record_mut(prev.slot())?.drop_ref()?;
            }
            self.entries[i] = None;
            self.mark_dirty(pool.kind(), block.get());
        }
        Ok(())
    }

    /// Drop every live reference held by this table and clear every cell to `None`.
    pub fn clear_entries(&mut self, pools: &mut BlockPools) -> Result<(), BlockError> {
        for cell in &mut self.entries {
            if let Some(prev) = cell.take() {
                let pool = pools.of_mut(prev.kind());
                if pool.record(prev.slot())?.generation() == prev.generation() {
                    pool.record_mut(prev.slot())?.drop_ref()?;
                }
            }
        }
        self.mark_full_sync();
        Ok(())
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
        pool.slot_bytes_mut_for_test(slot)[0] = 7;
        pool.record_mut(slot).unwrap().set_length(1).unwrap();
        pool.record_mut(slot).unwrap().seal();
        pool.take(slot).unwrap();
        assert!(!pool.record(slot).unwrap().sealed());
        assert_eq!(pool.record(slot).unwrap().generation(), 2);
        assert!(pool.slot_bytes(slot).unwrap().iter().all(|&b| b == 0));
    }

    #[test]
    fn take_unzeroed_updates_metadata_without_host_memset() {
        let mut pool = BlockPool::new(BlockKind::Full, 1);
        let slot = Slot::new(1).unwrap();
        pool.slot_bytes_mut_for_test(slot).fill(0x5c);
        assert_eq!(pool.take_unzeroed(slot).unwrap(), 0);
        assert_eq!(pool.record(slot).unwrap().generation(), 1);
        assert_eq!(pool.record(slot).unwrap().length(), 0);
        assert!(!pool.record(slot).unwrap().sealed());
        // Host bytes were untouched on take_unzeroed:
        assert_eq!(pool.slot_bytes(slot).unwrap()[0], 0x5c);
        assert_eq!(pool.record_mut(slot).unwrap().set_length(64), Ok(()));
        assert_eq!(pool.record_mut(slot).unwrap().set_length(65), Err(BlockError::Length));
        pool.record_mut(slot).unwrap().add_ref().unwrap();
        assert_eq!(pool.take_unzeroed(slot), Err(BlockError::Refs));
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
    #[cfg(not(standalone_blocks))]
    #[ignore = "requires a Metal GPU device; run in CI m5-gpu and nightly m5-full"]
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
    fn bump_generation_waits_for_zero_refs() {
        let mut slot = SlotRecord::fresh(BlockKind::Full);
        slot.set_length(64).unwrap();
        slot.seal();
        slot.add_ref().unwrap();
        assert_eq!(slot.bump_generation(), Err(BlockError::Refs));
        assert_eq!((slot.generation(), slot.length(), slot.sealed(), slot.refs()), (0, 64, true, 1));
        slot.drop_ref().unwrap();
        slot.bump_generation().unwrap();
        assert_eq!((slot.generation(), slot.length(), slot.sealed(), slot.refs()), (1, 64, true, 0));
        let mut max_slot = SlotRecord::at_generation(BlockKind::Window, u32::MAX);
        max_slot.set_length(32).unwrap();
        max_slot.seal();
        assert_eq!(max_slot.bump_generation(), Err(BlockError::Generation));
        assert_eq!((max_slot.generation(), max_slot.length(), max_slot.sealed(), max_slot.refs()), (u32::MAX, 32, true, 0));
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
        assert_eq!(full.gpu_table_entry(stored).unwrap(), stored.offset());
        assert_eq!(
            key_row_at(stored.offset(), FULL_PASSES as u32, NKV as u32, BLOCK as u32, HD as u32, 2, 0, 0, 0, 0).unwrap(),
            stored.offset()
        );
        assert_eq!(
            key_row_at(stored.offset(), FULL_PASSES as u32, NKV as u32, BLOCK as u32, HD as u32, 2, 0, 0, 0, 63).unwrap(),
            stored.offset() + 63 * (HD as u64) * 2
        );
        let win_stored = t.get(BlockKind::Window, BlockIndex::new(0)).unwrap().unwrap();
        assert_eq!(win_stored.offset(), byte_offset(BlockKind::Window, slot, 2).unwrap());
        assert_eq!(win_stored.offset(), 2 * WINDOW_STRIDE);
        assert_eq!(
            key_row_at(win_stored.offset(), WINDOW_PASSES as u32, NKV as u32, BLOCK as u32, HD as u32, 2, 0, 0, 0, 1).unwrap(),
            win_stored.offset() + (HD as u64) * 2
        );
        take(&mut full, Slot::new(1).unwrap());
        t.fill(&mut full, BlockIndex::new(1), Slot::new(1).unwrap(), 1).unwrap();
        written(&mut full, Slot::new(1).unwrap(), 64);
        let next = t.get(BlockKind::Full, BlockIndex::new(1)).unwrap().unwrap();
        assert_eq!(next.offset(), byte_offset(BlockKind::Full, Slot::new(1).unwrap(), 2).unwrap());
        assert_eq!(full.gpu_table_entry(next).unwrap(), 0);
    }

    #[test]
    fn a_reused_slot_is_refused_at_the_old_generation() {
        let mut full = BlockPool::new(BlockKind::Full, 4);
        let mut t = BlockTable::new(4).unwrap();
        let slot = Slot::new(3).unwrap();
        take(&mut full, slot);
        t.fill(&mut full, BlockIndex::new(0), slot, 1).unwrap();
        written(&mut full, slot, 1);
        let stored = t.get(BlockKind::Full, BlockIndex::new(0)).unwrap().unwrap();
        assert_eq!(stored.generation(), 1);
        assert_eq!(full.record_mut(slot).unwrap().bump_generation(), Err(BlockError::Refs));
        assert_eq!(full.record(slot).unwrap().generation(), 1);
        assert_eq!(full.record(slot).unwrap().refs(), 1);
        assert_eq!(full.gpu_table_entry(stored).unwrap(), stored.offset());
        assert_eq!(t.get(BlockKind::Full, BlockIndex::new(0)).unwrap().unwrap().slot(), slot);
    }

    #[test]
    fn a_key_row_is_pass_then_plane_then_head_then_position() {
        let row_at = |base, kind: BlockKind, kvb, pass, plane, head, index| {
            key_row_at(base, kind.passes() as u32, NKV as u32, BLOCK as u32, HD as u32, kvb, pass, plane, head, index)
        };
        let base = byte_offset(BlockKind::Window, Slot::new(1).unwrap(), 2).unwrap();
        assert_eq!(base, 0);
        let old_step = WINDOW_STRIDE / BLOCK as u64;
        assert_eq!(old_step, 163840);
        assert_eq!(row_at(base, BlockKind::Window, 2, 0, 1, 2, 0).unwrap(), old_step);
        assert_eq!(row_at(base, BlockKind::Window, 2, 0, 0, 0, 1).unwrap(), (HD as u64) * 2);
        assert_eq!(row_at(0, BlockKind::Window, 2, 0, 0, 0, 0).unwrap(), 0);
        assert_eq!(row_at(0, BlockKind::Window, 2, 0, 0, 0, 1).unwrap(), 256);
        assert_eq!(row_at(0, BlockKind::Window, 2, 0, 0, 1, 0).unwrap(), 16_384);
        assert_eq!(row_at(0, BlockKind::Window, 2, 0, 1, 2, 0).unwrap(), 163_840);
        let head = (BLOCK as u64) * (HD as u64) * 2;
        assert_eq!(row_at(base, BlockKind::Full, 2, 0, 0, 1, 0).unwrap(), head);
        assert_eq!(row_at(base, BlockKind::Full, 2, 0, 0, 0, 0).unwrap(), 0);
        assert_eq!(row_at(base, BlockKind::Full, 1, 0, 0, 1, 0).unwrap(), head / 2);
        assert_eq!(row_at(0, BlockKind::Window, 1, 1, 1, 3, 7).unwrap(), 222_080);
        assert_eq!(row_at(0, BlockKind::Window, 2, 40, 0, 0, 0), Err(BlockError::OutOfRange));
        assert_eq!(row_at(0, BlockKind::Full, 2, 8, 0, 0, 0), Err(BlockError::OutOfRange));
        assert_eq!(row_at(0, BlockKind::Full, 2, 0, 2, 0, 0), Err(BlockError::OutOfRange));
        assert_eq!(row_at(0, BlockKind::Full, 2, 0, 0, NKV as u32, 0), Err(BlockError::OutOfRange));
        assert_eq!(row_at(0, BlockKind::Full, 0, 0, 0, 0, 0), Err(BlockError::Width));
    }

    #[test]
    fn an_empty_pool_has_no_slots() {
        let mut pools = BlockPools::stub();
        let mut t = BlockTable::new(2).unwrap();
        let err = t.fill(pools.of_mut(BlockKind::Window), BlockIndex::new(0), Slot::new(1).unwrap(), 1).unwrap_err();
        assert_eq!(err, BlockError::NoSlots);
        assert_eq!(err.to_string(), "pool has no slots");
        assert_eq!(pools.full.len(), 0);
        assert_eq!(t.get(BlockKind::Full, BlockIndex::new(0)).unwrap(), None);
        let mut one = BlockPool::new(BlockKind::Full, 1);
        assert_eq!(t.fill(&mut one, BlockIndex::new(0), Slot::new(2).unwrap(), 1), Err(BlockError::OutOfRange));
        let empty_table = BlockTable::empty();
        assert_eq!(empty_table.get(BlockKind::Full, BlockIndex::new(0)), Err(BlockError::OutOfRange));
        let bare = BlockTable::new(4).unwrap();
        assert_eq!(bare.get(BlockKind::Window, BlockIndex::new(0)).unwrap(), None);
        assert_eq!(bare.get(BlockKind::Window, BlockIndex::new(4)), Err(BlockError::OutOfRange));
    }

    #[test]
    fn a_key_offset_is_a_u64_past_the_layer2_base() {
        let index = u32::MAX / (WINDOW_STRIDE as u32) + 1;
        let slot = Slot::new(index + 1).unwrap();
        let mut pool = BlockPool::records_only(BlockKind::Window, (index + 1) as usize, 2).unwrap();
        take(&mut pool, slot);
        let mut t = BlockTable::new(1).unwrap();
        t.fill(&mut pool, BlockIndex::new(0), slot, 1).unwrap();
        written(&mut pool, slot, 1);
        let entry = t.get(BlockKind::Window, BlockIndex::new(0)).unwrap().unwrap();
        let off = key_row_at(entry.offset(), WINDOW_PASSES as u32, NKV as u32, BLOCK as u32, HD as u32, 2, 0, 0, 0, 0).unwrap();
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
    fn block_length_bounds_and_seal_state_hold() {
        let mut pool = BlockPool::new(BlockKind::Window, 2);
        let mut t = BlockTable::new(1).unwrap();
        let slot = Slot::new(1).unwrap();
        take(&mut pool, slot);
        t.fill(&mut pool, BlockIndex::new(0), slot, 1).unwrap();
        assert_eq!(pool.record(slot).unwrap().length(), 0);
        written(&mut pool, slot, 3);
        assert_eq!(pool.record(slot).unwrap().length(), 3);
        assert!(!pool.record(slot).unwrap().sealed());
        assert_eq!(pool.record_mut(slot).unwrap().set_length(65), Err(BlockError::Length));
        written(&mut pool, slot, 64);
        assert_eq!(pool.record(slot).unwrap().length(), 64);
        assert!(pool.record(slot).unwrap().sealed());
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
    fn gpu_table_entry_refuses_a_slot_nothing_references() {
        let mut pool = BlockPool::new(BlockKind::Full, 2);
        let mut t = BlockTable::new(1).unwrap();
        let slot = Slot::new(1).unwrap();
        take(&mut pool, slot);
        t.fill(&mut pool, BlockIndex::new(0), slot, 1).unwrap();
        written(&mut pool, slot, 1);
        let entry = t.get(BlockKind::Full, BlockIndex::new(0)).unwrap().unwrap();
        pool.record_mut(slot).unwrap().drop_ref().unwrap();
        assert_eq!(pool.gpu_table_entry(entry), Err(BlockError::Refs));
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
        let e = BlockEntry::new(one, 1, 0, BlockKind::Full, 2);
        assert_eq!(t.set(BlockKind::Window, BlockIndex::new(0), Some(e)), Err(BlockError::OutOfRange));
        assert_eq!(t.set(BlockKind::Full, BlockIndex::new(2), Some(e)), Err(BlockError::OutOfRange));
        t.set(BlockKind::Full, BlockIndex::new(1), Some(e)).unwrap();
        assert_eq!(t.entries().map(|(b, e)| (b.get(), e.kind())).collect::<Vec<_>>(), vec![(1, BlockKind::Full)]);
        t.set(BlockKind::Full, BlockIndex::new(1), None).unwrap();
        assert_eq!(t.entries().count(), 0);
        assert_eq!(t.blocks_per_kind(), 2);
    }

    #[test]
    fn dirty_range_initial_full_sync_and_63_clean_decode_steps() {
        let mut pools = BlockPools::new(8);
        let mut t = BlockTable::new(32).unwrap();
        assert!(t.needs_full_sync(BlockKind::Full));
        assert!(t.needs_full_sync(BlockKind::Window));
        assert_eq!(t.dirty_range(BlockKind::Full), Some((0, 31)));
        assert_eq!(t.dirty_range(BlockKind::Window), Some((0, 31)));

        let (sf0, _) = pools.full.take_free().unwrap();
        let (sw0, _) = pools.window.take_free().unwrap();
        t.fill(&mut pools.full, BlockIndex::new(0), sf0, 1).unwrap();
        t.fill(&mut pools.window, BlockIndex::new(0), sw0, 1).unwrap();
        // Still needs full sync until first upload clears it.
        assert_eq!(t.dirty_range(BlockKind::Full), Some((0, 31)));
        assert_eq!(t.dirty_range(BlockKind::Window), Some((0, 31)));

        let mut gpu_full = vec![0xDEAD_BEEF_u64; 32];
        let mut gpu_win = vec![0xCAFE_BABE_u64; 32];
        assert_eq!(t.sync_dirty_to_slice(BlockKind::Full, &pools.full, &mut gpu_full).unwrap(), 32);
        assert_eq!(t.sync_dirty_to_slice(BlockKind::Window, &pools.window, &mut gpu_win).unwrap(), 32);
        assert!(!t.needs_full_sync(BlockKind::Full));
        assert!(!t.needs_full_sync(BlockKind::Window));
        assert_eq!(t.dirty_range(BlockKind::Full), None);
        assert_eq!(t.dirty_range(BlockKind::Window), None);
        assert_eq!(gpu_full[0], 0);
        assert_ne!(gpu_full[0], CLEARED_GPU_TABLE_ENTRY);
        assert!(gpu_full[1..].iter().all(|&w| w == CLEARED_GPU_TABLE_ENTRY));
        assert_eq!(gpu_win[0], 0);
        assert_ne!(gpu_win[0], CLEARED_GPU_TABLE_ENTRY);
        assert!(gpu_win[1..].iter().all(|&w| w == CLEARED_GPU_TABLE_ENTRY));

        // 63 steady-state decode steps inside block 0 write 0 entries to unified memory.
        // Place sentinels to prove `sync_dirty_to_slice` does not touch `dst` at all when clean.
        gpu_full[1] = 0x1111_2222;
        gpu_win[1] = 0x3333_4444;
        for _step in 1..64 {
            assert_eq!(t.dirty_range(BlockKind::Full), None);
            assert_eq!(t.dirty_range(BlockKind::Window), None);
            assert_eq!(t.sync_dirty_to_slice(BlockKind::Full, &pools.full, &mut gpu_full).unwrap(), 0);
            assert_eq!(t.sync_dirty_to_slice(BlockKind::Window, &pools.window, &mut gpu_win).unwrap(), 0);
        }
        assert_eq!(gpu_full[1], 0x1111_2222);
        assert_eq!(gpu_win[1], 0x3333_4444);
    }

    #[test]
    fn dirty_range_single_allocation_coalescing_and_window_pruning() {
        let mut pools = BlockPools::new(8);
        let mut t = BlockTable::new(32).unwrap();
        let mut gpu_full = vec![0u64; 32];
        let mut gpu_win = vec![0u64; 32];
        // Initial sync on empty table marks every cleared slot with CLEARED_GPU_TABLE_ENTRY and clears dirty flags.
        assert_eq!(t.sync_dirty_to_slice(BlockKind::Full, &pools.full, &mut gpu_full).unwrap(), 32);
        assert_eq!(t.sync_dirty_to_slice(BlockKind::Window, &pools.window, &mut gpu_win).unwrap(), 32);
        assert!(gpu_full.iter().all(|&w| w == CLEARED_GPU_TABLE_ENTRY));
        assert!(gpu_win.iter().all(|&w| w == CLEARED_GPU_TABLE_ENTRY));

        // Allocate block 0 for both kinds (e.g. token 0): Slot(1) in slab 0 encodes as 0 != CLEARED_GPU_TABLE_ENTRY.
        let (sf0, _) = pools.full.take_free().unwrap();
        let (sw0, _) = pools.window.take_free().unwrap();
        t.fill(&mut pools.full, BlockIndex::new(0), sf0, 1).unwrap();
        t.fill(&mut pools.window, BlockIndex::new(0), sw0, 1).unwrap();
        assert_eq!(t.dirty_range(BlockKind::Full), Some((0, 0)));
        assert_eq!(t.dirty_range(BlockKind::Window), Some((0, 0)));
        assert_eq!(t.sync_dirty_to_slice(BlockKind::Full, &pools.full, &mut gpu_full).unwrap(), 1);
        assert_eq!(t.sync_dirty_to_slice(BlockKind::Window, &pools.window, &mut gpu_win).unwrap(), 1);
        assert_eq!(gpu_full[0], 0);
        assert_eq!(gpu_win[0], 0);

        // Allocate single new block at index 1 (token 64).
        let (sf1, off_f1) = pools.full.take_free().unwrap();
        let (sw1, off_w1) = pools.window.take_free().unwrap();
        t.fill(&mut pools.full, BlockIndex::new(1), sf1, 1).unwrap();
        t.fill(&mut pools.window, BlockIndex::new(1), sw1, 1).unwrap();
        assert_eq!(t.dirty_range(BlockKind::Full), Some((1, 1)));
        assert_eq!(t.dirty_range(BlockKind::Window), Some((1, 1)));

        // Re-filling the exact same entry is a no-op and does not expand the dirty range.
        t.fill(&mut pools.full, BlockIndex::new(0), sf0, 1).unwrap();
        assert_eq!(t.dirty_range(BlockKind::Full), Some((1, 1)));
        // Failed fill (stale generation) does not dirty.
        assert_eq!(t.fill(&mut pools.full, BlockIndex::new(5), sf1, 99), Err(BlockError::Stale));
        assert_eq!(t.dirty_range(BlockKind::Full), Some((1, 1)));

        gpu_full[0] = 0xAAAA;
        gpu_full[2] = 0xBBBB;
        assert_eq!(t.sync_dirty_to_slice(BlockKind::Full, &pools.full, &mut gpu_full).unwrap(), 1);
        assert_eq!(t.sync_dirty_to_slice(BlockKind::Window, &pools.window, &mut gpu_win).unwrap(), 1);
        assert_eq!(gpu_full[0], 0xAAAA); // untouched
        assert_eq!(gpu_full[1], off_f1);
        assert_eq!(gpu_full[2], 0xBBBB); // untouched
        assert_eq!(gpu_win[1], off_w1);

        // Window block pruning: clearing window block 0 dirties ONLY window block 0 and writes CLEARED_GPU_TABLE_ENTRY.
        gpu_win[0] = 0x9999;
        t.clear_kind_entry(&mut pools.window, BlockIndex::new(0)).unwrap();
        assert_eq!(t.dirty_range(BlockKind::Full), None);
        assert_eq!(t.dirty_range(BlockKind::Window), Some((0, 0)));
        assert_eq!(t.sync_dirty_to_slice(BlockKind::Window, &pools.window, &mut gpu_win).unwrap(), 1);
        assert_eq!(gpu_win[0], CLEARED_GPU_TABLE_ENTRY);
        assert_eq!(gpu_win[1], off_w1);
        // Clearing an already-empty entry is a no-op (stays clean).
        t.clear_kind_entry(&mut pools.window, BlockIndex::new(0)).unwrap();
        assert_eq!(t.dirty_range(BlockKind::Window), None);

        // Coalesced range: mutate block 5 then block 2 -> [2..=5] (4 entries).
        let (sf5, off_f5) = pools.full.take_free().unwrap();
        t.fill(&mut pools.full, BlockIndex::new(5), sf5, 1).unwrap();
        let (sf2, off_f2) = pools.full.take_free().unwrap();
        t.fill(&mut pools.full, BlockIndex::new(2), sf2, 1).unwrap();
        assert_eq!(t.dirty_range(BlockKind::Full), Some((2, 5)));
        assert_eq!(t.sync_dirty_to_slice(BlockKind::Full, &pools.full, &mut gpu_full).unwrap(), 4);
        assert_eq!(gpu_full[2], off_f2);
        assert_eq!(gpu_full[3], CLEARED_GPU_TABLE_ENTRY);
        assert_eq!(gpu_full[4], CLEARED_GPU_TABLE_ENTRY);
        assert_eq!(gpu_full[5], off_f5);
    }

    #[test]
    fn dirty_range_full_reset_and_rebind() {
        let mut pools = BlockPools::new(8);
        let mut t = BlockTable::new(16).unwrap();
        let mut gpu_full = vec![0u64; 16];
        let mut gpu_win = vec![0u64; 16];

        let (sf0, _) = pools.full.take_free().unwrap();
        t.fill(&mut pools.full, BlockIndex::new(0), sf0, 1).unwrap();
        let (sf1, off_f1) = pools.full.take_free().unwrap();
        t.fill(&mut pools.full, BlockIndex::new(1), sf1, 1).unwrap();
        let (sw0, _) = pools.window.take_free().unwrap();
        t.fill(&mut pools.window, BlockIndex::new(0), sw0, 1).unwrap();
        let (sw1, off_w1) = pools.window.take_free().unwrap();
        t.fill(&mut pools.window, BlockIndex::new(1), sw1, 1).unwrap();
        assert_eq!(t.sync_dirty_to_slice(BlockKind::Full, &pools.full, &mut gpu_full).unwrap(), 16);
        assert_eq!(t.sync_dirty_to_slice(BlockKind::Window, &pools.window, &mut gpu_win).unwrap(), 16);
        assert_eq!(gpu_full[1], off_f1);
        assert_eq!(gpu_win[1], off_w1);

        // Full reset via `clear_entries` triggers full sync on both kinds.
        t.clear_entries(&mut pools).unwrap();
        assert!(t.needs_full_sync(BlockKind::Full));
        assert!(t.needs_full_sync(BlockKind::Window));
        assert_eq!(t.dirty_range(BlockKind::Full), Some((0, 15)));
        assert_eq!(t.dirty_range(BlockKind::Window), Some((0, 15)));
        assert_eq!(pools.full.record(sf0).unwrap().refs(), 0);
        assert_eq!(pools.full.record(sf1).unwrap().refs(), 0);

        // Rebind a single prefix block (block 0) before the next sync: still does full sync so stale
        // GPU slot 1 from the previous request is overwritten with CLEARED_GPU_TABLE_ENTRY!
        t.fill(&mut pools.full, BlockIndex::new(0), sf0, 1).unwrap();
        t.fill(&mut pools.window, BlockIndex::new(0), sw0, 1).unwrap();
        assert_eq!(t.sync_dirty_to_slice(BlockKind::Full, &pools.full, &mut gpu_full).unwrap(), 16);
        assert_eq!(t.sync_dirty_to_slice(BlockKind::Window, &pools.window, &mut gpu_win).unwrap(), 16);
        assert_eq!(gpu_full[0], 0);
        assert_eq!(gpu_full[1], CLEARED_GPU_TABLE_ENTRY);
        assert_eq!(gpu_win[0], 0);
        assert_eq!(gpu_win[1], CLEARED_GPU_TABLE_ENTRY);
        assert_eq!(t.dirty_range(BlockKind::Full), None);
        assert_eq!(t.dirty_range(BlockKind::Window), None);

        // Direct `set` mutation vs no-op `set`.
        let e = BlockEntry::new(sf1, 1, off_f1, BlockKind::Full, 2);
        t.set(BlockKind::Full, BlockIndex::new(3), None).unwrap(); // already None -> no-op
        assert_eq!(t.dirty_range(BlockKind::Full), None);
        t.set(BlockKind::Full, BlockIndex::new(3), Some(e)).unwrap();
        assert_eq!(t.dirty_range(BlockKind::Full), Some((3, 3)));
        t.set(BlockKind::Full, BlockIndex::new(3), Some(e)).unwrap(); // same value -> stays (3, 3)
        t.set(BlockKind::Full, BlockIndex::new(3), None).unwrap();
        assert_eq!(t.dirty_range(BlockKind::Full), Some((3, 3)));
    }

    #[test]
    #[cfg(not(standalone_blocks))]
    fn metal_in_slot_formula_matches_key_row_and_inblock_trace_across_full_and_window() {
        let kernels_src = include_str!("kernels.metal");
        let prefill_src = include_str!("prefill.metal");
        let layout = parse_metal_kv_layout(kernels_src, prefill_src).unwrap();
        assert_eq!(layout.nkv, NKV as u32);
        assert_eq!(layout.hd, HD as u32);
        assert_eq!(layout.kv_blk, BLOCK as u32);
        assert_eq!(layout.kv_blk_mask, (BLOCK - 1) as u32);
        assert_eq!(layout.planes_per_pass, 2);
        assert_eq!(layout.slab_shift, KV_SLAB_SHIFT);
        assert_eq!(layout.slab_buffers, [3, 10, 11, 12]);

        // Exhaustively check all 8,192 Full coordinates and 40,960 Window coordinates at fp16 and KV8.
        for kvb in [2u32, 1u32] {
            for kind in [BlockKind::Full, BlockKind::Window] {
                let stride = kind.stride_bytes(kvb).unwrap();
                let mut prev = None;
                for pass in 0..(kind.passes() as u32) {
                    for kv in [Kv::Key, Kv::Value] {
                        for head in 0..(NKV as u32) {
                            for blk in [0u32, 17u32] {
                                for ib in 0..(BLOCK as u32) {
                                    let stream_pos = blk * (BLOCK as u32) + ib;
                                    let got = metal_in_slot_byte_offset_from_layout(
                                        &layout, kind, kvb, pass, kv, head, stream_pos,
                                    )
                                    .unwrap();
                                    let want = key_row(0, kind, kvb, pass, kv, head, ib).unwrap();
                                    assert_eq!(got, want);
                                    assert!(got + (HD as u64) * u64::from(kvb) <= stride);
                                    if blk == 0 {
                                        if let Some(p) = prev {
                                            assert_eq!(got - p, (HD as u64) * u64::from(kvb));
                                        }
                                        prev = Some(got);
                                    }
                                }
                            }
                        }
                    }
                }
                assert_eq!(
                    prev.unwrap() + (HD as u64) * u64::from(kvb),
                    stride,
                    "last row in {kind:?} at kvb={kvb} must end exactly at slot stride"
                );
            }
        }

        // Verify `specs/InBlockTraceData.tla` matches `write_inblock_trace` and `metal_in_slot_byte_offset`.
        let trace_text = include_str!("../specs/InBlockTraceData.tla");
        for (name, kind, kvb, expected_len) in [
            ("MetalFull2", BlockKind::Full, 2u32, 240usize),
            ("MetalFull1", BlockKind::Full, 1u32, 240usize),
            ("MetalWin2", BlockKind::Window, 2u32, 1200usize),
            ("MetalWin1", BlockKind::Window, 1u32, 1200usize),
        ] {
            let rows = trace_rows(trace_text, name);
            assert_eq!(rows.len(), expected_len, "{name} row count");
            for (p, k, h, i, addr) in rows {
                let plane = if k == 0 { Kv::Key } else { Kv::Value };
                assert_eq!(metal_in_slot_byte_offset(kind, kvb, p, plane, h, i).unwrap(), addr);
                assert_eq!(key_row(0, kind, kvb, p, plane, h, i).unwrap(), addr);
            }
        }

        // Planted shader macro & buffer-binding mutations MUST fail `parse_metal_kv_layout`.
        assert!(parse_metal_kv_layout(
            &kernels_src.replacen("#define KV_PASS_BYTES  (2UL * KV_PLANE_BYTES)", "#define KV_PASS_BYTES  KV_PLANE_BYTES", 1),
            prefill_src,
        ).is_err());
        assert!(parse_metal_kv_layout(
            &kernels_src.replacen("#define KV_SLAB_MASK ((1UL << 62) - 1UL)", "#define KV_SLAB_MASK ((1UL << 60) - 1UL)", 1),
            prefill_src,
        ).is_err());
        assert!(parse_metal_kv_layout(
            kernels_src,
            &prefill_src.replacen("KV1 [[buffer(10)]]", "KV1 [[buffer(9)]]", 1),
        ).is_err());
    }

    #[test]
    fn multi_slab_gpu_table_entry_packs_all_four_slabs_decodes_shader_ptr_and_refuses_overflow() {
        // 1. Allocate across all 4 slabs (`KV0..KV3`, 2 slots per slab -> 8 slots total) at fp16 and KV8.
        for kvb in [2u32, 1u32] {
            let stride = stride_at(BlockKind::Full, kvb).unwrap();
            let mut pool = BlockPool::open_host(BlockKind::Full, 0, kvb, stride * 2).unwrap();
            let mut table = BlockTable::new(8).unwrap();
            let mut slots = Vec::new();
            for bi in 0..8u32 {
                let (slot, logical_off) = pool.take_free().unwrap();
                assert_eq!(logical_off, u64::from(bi) * stride);
                let gen = pool.record(slot).unwrap().generation();
                table.fill(&mut pool, BlockIndex::new(bi), slot, gen).unwrap();
                // Stamp a unique 8-byte tag at the start and end of each slot's backing storage.
                let tag = 0xA500_0000_0000_0000u64 | (u64::from(kvb) << 16) | u64::from(bi + 1);
                let sb = pool.slot_bytes_mut(slot).unwrap();
                sb[..8].copy_from_slice(&tag.to_le_bytes());
                let end = sb.len() - 8;
                sb[end..].copy_from_slice(&(tag ^ 0xFFFF).to_le_bytes());
                slots.push(slot);
            }
            assert_eq!(pool.slab_count(), MAX_SLABS);
            // A 9th slot requires a 5th slab (`KV4`), which MUST be refused with `BlockError::Slab`.
            assert_eq!(pool.grow(), Err(BlockError::Slab));

            // Upload all 8 entries via `sync_dirty_to_slice` and decode every word with `kv_blk_ptr` bit logic.
            let mut gpu_bt = vec![0u64; 8];
            assert_eq!(table.sync_dirty_to_slice(BlockKind::Full, &pool, &mut gpu_bt).unwrap(), 8);

            // Collect raw slab base slices (`s0, s1, s2, s3` bound at buffers `3, 10, 11, 12`).
            assert_eq!(KV_SLAB_BUFFER_INDICES, [3, 10, 11, 12]);
            for (bi, &word) in gpu_bt.iter().enumerate().take(8usize) {
                let expected_slab = bi / 2;
                let expected_off = ((bi % 2) as u64) * stride;
                assert_eq!(word, ((expected_slab as u64) << 62) | expected_off);
                let (si, off) = decode_gpu_table_entry(word);
                assert_eq!(si, expected_slab);
                assert_eq!(off, expected_off);
                assert!(off + stride <= stride * 2);

                // Verify `base + off + in_slot_off` resolves to the exact stamped slot without aliasing.
                let slab_first_slot = slots[si * 2];
                let (s_ptr, _) = pool.slot_raw_region(slab_first_slot).unwrap();
                let resolved_start = unsafe { std::slice::from_raw_parts(s_ptr.add(off as usize), 8) };
                let resolved_end = unsafe {
                    std::slice::from_raw_parts(s_ptr.add((off + stride - 8) as usize), 8)
                };
                let want_tag = 0xA500_0000_0000_0000u64 | (u64::from(kvb) << 16) | ((bi as u64) + 1);
                assert_eq!(u64::from_le_bytes(resolved_start.try_into().unwrap()), want_tag);
                assert_eq!(u64::from_le_bytes(resolved_end.try_into().unwrap()), want_tag ^ 0xFFFF);
            }
        }

        // 2. Boundary & 62-bit overflow checks on `gpu_table_entry` (`NoOffsetClobbersSlab` in `SlabWord.tla`).
        let stride = FULL_STRIDE; // 2 MiB (`1 << 21`)
        let max_valid_within = ((1u64 << KV_SLAB_SHIFT) / stride - 1) as usize; // `(1 << 41) - 1`
        let overflow_within = (1u64 << KV_SLAB_SHIFT) / stride; // `1 << 41` -> `off == 1 << 62`!
        let slots_per_slab = (overflow_within as usize) + 2;

        // Construct a lightweight metadata-only pool at the 62-bit boundary (`slab = 1` and `slab = 2`).
        let boundary_slots = vec![
            (0usize, max_valid_within),                  // slab 0, highest valid 62-bit offset `(1<<62) - 2 MiB`
            (1usize, max_valid_within),                  // slab 1, highest valid 62-bit offset
            (2usize, overflow_within as usize),          // slab 2, `off == 1<<62` -> MUST refuse `Overflow`!
        ];
        for (slab_i, within_i) in boundary_slots {
            // Shift `slots` sparse window by testing a 1-slot custom `BlockPool` via `slots_per_slab`?
            // Note `i = slot.pool_index()`, so for `slab = 0` at `max_valid_within` we can test directly
            // with a custom `stride` where `slots_per_slab = 4`:
            let _ = (slab_i, within_i, slots_per_slab);
        }
        let half_limit = 1u64 << 61; // 2 EiB stride -> within=0 is 0, within=1 is 2 EiB (< 2^62), within=2 is 4 EiB (== 2^62)!
        let mut synthetic = BlockPool {
            kind: BlockKind::Full,
            kvb: 2,
            stride: half_limit,
            slots_per_slab: 3, // within \in {0, 1, 2}, where within=2 hits `1 << 62`!
            slots: vec![SlotRecord::fresh(BlockKind::Full); 12],
            slabs: Vec::new(),
            #[cfg(not(standalone_blocks))]
            device: None,
            gpu_zero_pending: vec![false; 12],
        };
        for idx in 1..=12u32 {
            let s = Slot::new(idx).unwrap();
            synthetic.record_mut(s).unwrap().bump_generation().unwrap();
            synthetic.record_mut(s).unwrap().add_ref().unwrap();
        }
        // For every slab 0..4: `within = 0` and `within = 1` (`off = 1 << 61`) succeed and preserve `slab << 62`,
        // while `within = 2` (`off = 1 << 62`) is refused with `BlockError::Overflow`!
        for slab in 0..MAX_SLABS {
            for within in 0..3usize {
                let s = Slot::new((slab * 3 + within + 1) as u32).unwrap();
                let entry = BlockEntry::new(s, 1, 0, BlockKind::Full, 2);
                let res = synthetic.gpu_table_entry(entry);
                if within < 2 {
                    let word = res.expect("valid < 2^62 offset");
                    assert_eq!(decode_gpu_table_entry(word), (slab, (within as u64) * half_limit));
                } else {
                    assert_eq!(res, Err(BlockError::Overflow), "offset >= 2^62 must be refused");
                }
            }
        }
        // Also verify `within * stride` wrapping past `2^64` is caught via `checked_mul` -> `BlockError::Overflow`.
        synthetic.stride = (1u64 << 63) + 1024; // within=2 would wrap modulo 2^64 to `2048 < 2^62` without checked_mul!
        let wrap_slot = Slot::new(3).unwrap(); // i = 2 -> within = 2
        let wrap_entry = BlockEntry::new(wrap_slot, 1, 0, BlockKind::Full, 2);
        assert_eq!(synthetic.gpu_table_entry(wrap_entry), Err(BlockError::Overflow));
    }

    #[test]
    fn span_window_blocks_and_table_pinning_reach_25_at_worst_alignment() {
        let mut pools = BlockPools::new(32);
        let mut tbl = BlockTable::new(32).unwrap();
        for b in 0..32u32 {
            let slot = Slot::new(b + 1).unwrap();
            let _ = pools.full.take(slot).unwrap();
            let _ = pools.window.take(slot).unwrap();
            tbl.fill(&mut pools.full, BlockIndex::new(b), slot, 1).unwrap();
            tbl.fill(&mut pools.window, BlockIndex::new(b), slot, 1).unwrap();
        }
        let mut max_win = 0usize;
        for offset in 0..BLOCK {
            let pos = (WINDOW - 1) + offset;
            let win_blks = span_window_blocks(pos, 512).unwrap();
            let pinned = tbl.pinned_span_entries(pos, 512).unwrap();
            let table_win: Vec<u32> = pinned
                .iter()
                .filter(|(k, _, _)| *k == BlockKind::Window)
                .map(|(_, bi, _)| bi.get())
                .collect();
            let direct_win: Vec<u32> = win_blks.iter().map(|bi| bi.get()).collect();
            assert_eq!(table_win, direct_win);
            assert!(table_win.len() <= 25, "alignment {offset} exceeded 25 window blocks");
            max_win = max_win.max(table_win.len());
            if offset == BLOCK - 1 {
                let expected: Vec<u32> = (0..=24).collect();
                assert_eq!(table_win, expected, "offset 63 must pin blocks 0..=24 (25 blocks, not 17)");
            }
        }
        assert_eq!(max_win, 25);
        assert_eq!(span_window_blocks(0, 0), Err(BlockError::OutOfRange));
    }

    #[test]
    fn cleared_gpu_table_entry_is_distinct_from_slab_zero_first_slot() {
        // Multi-slab pool across all 4 slabs: neither slab 0 Slot(1) (`0`) nor any other slot in slabs 0..4
        // ever encodes as CLEARED_GPU_TABLE_ENTRY (`u64::MAX`).
        let mut pool = BlockPool::open(BlockKind::Full, 0, KVB_FP16, FULL_STRIDE * 2, None).unwrap();
        let mut table = BlockTable::new(8).unwrap();
        for b in 0..8u32 {
            let slot = pool.grow().unwrap();
            let off = pool.take(slot).unwrap();
            let gen = pool.record(slot).unwrap().generation();
            table.fill(&mut pool, BlockIndex::new(b), slot, gen).unwrap();
            let entry = BlockEntry::new(slot, gen, off, BlockKind::Full, KVB_FP16);
            let word = pool.gpu_table_entry(entry).unwrap();
            if slot.get() == 1 {
                assert_eq!(word, 0, "slab 0 Slot(1) encodes as 0");
            }
            assert_ne!(word, CLEARED_GPU_TABLE_ENTRY);
        }
    }
}
