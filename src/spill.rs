//! Layer 4 SSD Disk Tier: lazy LRU spill, atomic working-set reservation, and
//! crash-safe checksummed block storage (`docs/ROADMAP.md` lines 136–137, 140;
//! `docs/DESIGN.md`, Tiers & Scheduler).
//!
//! Design invariants (matching `specs/CacheScheduler.tla`):
//! 1. **Lazy LRU spill under memory pressure**: sealed blocks stay in memory
//!    (`BlockTier::Memory`) with zero SSD writes when allocated, sealed, or
//!    unpinned. Cold unreferenced blocks (`refs == 0`, not in-flight I/O) are
//!    evicted to `--spill-dir` *only* when pool free slots cannot satisfy an
//!    allocation or working-set reservation, strictly least-recently-used first.
//! 2. **Atomic working-set reservation before restore**: `reserve_working_set`
//!    and `prepare_working_set_restore` reserve every required `Full` and
//!    `Window` pool slot atomically *before* issuing any SSD read, triggering
//!    lazy LRU spills only when the entire working set can fit or returning
//!    `ReserveError::WouldBlock` with zero mutations and zero partial reads.
//! 3. **Crash-safe checksummed format (`kill -9` gate)**: every spilled block
//!    (K/V bytes + optional start-point recurrent state) carries a fixed 80-byte
//!    header (`magic`, `version`, `block_kind`, `kvb`, `block_id`, `generation`,
//!    `kv_len`, `state_len`, and SHA-256 over `header[0..48] || kv || state`).
//!    Writes go to `<name>.tmp.<pid>.<nonce>` (`0600`, `O_NOFOLLOW`), `fsync`,
//!    atomic `rename` to `<name>.blk`, and directory `fsync`. Startup sweeps
//!    orphan `.tmp.*` files; any missing, truncated, oversized, or
//!    checksum-mismatched `.blk` is treated as a clean cache miss and unlinked,
//!    never returning corrupt K/V or recurrent state bytes.
//! 4. **Clean `BlockPool` merge hook**: pool operations go through
//!    [`SpillPoolBackend`], whose slot/generation/refcount semantics match
//!    `src/blocks.rs` (`BlockPool` / `BlockPools`) 1-to-1 so `layer3` and
//!    `src/scheduler.rs` can plug in with a thin adapter.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, FileExt, MetadataExt, OpenOptionsExt, PermissionsExt};

/// Positions in one sealed block (`docs/DESIGN.md`, Blocks).
pub const BLOCK_POSITIONS: usize = 64;
/// Bytes in one full-attention fp16 block (8 full passes, 2 MiB).
pub const FULL_STRIDE: usize = 2 * 1024 * 1024;
/// Bytes in one sliding-window fp16 block (40 window passes, 10 MiB).
pub const WINDOW_STRIDE: usize = 10 * 1024 * 1024;
/// Default `--max-spill-bytes` disk budget (64 GiB).
pub const DEFAULT_MAX_SPILL_BYTES: u64 = 64 * 1024 * 1024 * 1024;
/// Maximum allowed recurrent state payload per spilled block (16 MiB;
/// Inkling 2 Mini's convolution image is ~3.66 MiB).
pub const MAX_RECURRENT_STATE_BYTES: usize = 16 * 1024 * 1024;
/// Maximum allowed K/V payload per spilled block (64 MiB defensive cap).
pub const MAX_KV_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;

/// Fixed 8-byte file magic for spilled block files.
pub const SPILL_MAGIC: [u8; 8] = *b"INKYSPIL";
/// Binary on-disk format version.
pub const SPILL_FORMAT_VERSION: u32 = 2;
/// Fixed header length in bytes:
/// magic(8) + version(4) + kind(2) + kvb(2) + block_id(8) + generation(8) + kv_len(8) + state_len(8) + sha256(32).
pub const SPILL_HEADER_LEN: usize = 80;
const HEADER_PREFIX_LEN: usize = 48;

static TEMP_NONCE: AtomicU64 = AtomicU64::new(1);

// ---------------------------------------------------------------------------
// Pure-Rust FIPS 180-4 SHA-256 (zero external crate dependencies)
// ---------------------------------------------------------------------------

/// Streaming SHA-256 hasher (FIPS 180-4).
#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_len: u64,
}

const SHA256_H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Sha256 {
    pub fn new() -> Self {
        Self {
            state: SHA256_H0,
            buf: [0u8; 64],
            buf_len: 0,
            total_len: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let block: [u8; 64] = data[..64].try_into().unwrap();
            self.compress(&block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    pub fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8);
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;
        if self.buf_len > 56 {
            self.buf[self.buf_len..64].fill(0);
            let block = self.buf;
            self.compress(&block);
            self.buf_len = 0;
        }
        self.buf[self.buf_len..56].fill(0);
        self.buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buf;
        self.compress(&block);

        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(SHA256_K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

fn ct_eq_32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

// ---------------------------------------------------------------------------
// Core types & BlockPool adapter trait
// ---------------------------------------------------------------------------

/// The two cache block kinds (1-to-1 with `blocks::BlockKind` in Layer 3).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u32)]
pub enum SpillBlockKind {
    Full = 0,
    Window = 1,
}

impl SpillBlockKind {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Full),
            1 => Some(Self::Window),
            _ => None,
        }
    }

    pub fn tag(self) -> char {
        match self {
            Self::Full => 'f',
            Self::Window => 'w',
        }
    }

    pub fn default_stride(self) -> usize {
        match self {
            Self::Full => FULL_STRIDE,
            Self::Window => WINDOW_STRIDE,
        }
    }
}

/// Unique identity of a sealed block across kinds and disk files.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct SpillKey {
    pub kind: SpillBlockKind,
    pub id: u64,
}

impl SpillKey {
    pub const fn new(kind: SpillBlockKind, id: u64) -> Self {
        Self { kind, id }
    }

    pub const fn full(id: u64) -> Self {
        Self { kind: SpillBlockKind::Full, id }
    }

    pub const fn window(id: u64) -> Self {
        Self { kind: SpillBlockKind::Window, id }
    }

    fn blk_filename(self) -> String {
        format!("blk-{}-{:016x}.blk", self.kind.tag(), self.id)
    }

    fn tmp_filename(self, pid: u32, nonce: u64) -> String {
        format!("blk-{}-{:016x}.tmp.{:08x}.{:016x}", self.kind.tag(), self.id, pid, nonce)
    }

    fn is_lower_hex(s: &str, width: usize) -> bool {
        s.len() == width && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }

    fn parse_file_stem(stem: &str) -> Option<Self> {
        let (tag, hex) = stem.strip_prefix("blk-")?.split_once('-')?;
        if !Self::is_lower_hex(hex, 16) {
            return None;
        }
        let kind = match tag {
            "f" => SpillBlockKind::Full,
            "w" => SpillBlockKind::Window,
            _ => return None,
        };
        let id = u64::from_str_radix(hex, 16).ok()?;
        Some(Self { kind, id })
    }

    fn parse_blk_filename(name: &str) -> Option<Self> {
        Self::parse_file_stem(name.strip_suffix(".blk")?)
    }

    fn parse_tmp_filename(name: &str) -> Option<Self> {
        let (stem, suffix) = name.split_once(".tmp.")?;
        let (pid, nonce) = suffix.split_once('.')?;
        if !Self::is_lower_hex(pid, 8) || !Self::is_lower_hex(nonce, 16) {
            return None;
        }
        Self::parse_file_stem(stem)
    }
}

/// Residency tier of a sealed block (matching `CacheScheduler.tla` `Tiers`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlockTier {
    None,
    Memory,
    Disk,
    Both,
}

impl BlockTier {
    pub fn in_memory(self) -> bool {
        matches!(self, Self::Memory | Self::Both)
    }

    pub fn on_disk(self) -> bool {
        matches!(self, Self::Disk | Self::Both)
    }
}

/// Handle to a resident pool slot (1-to-1 with Layer 3 `blocks::BlockEntry`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ResidentSlot {
    pub kind: SpillBlockKind,
    /// 1-based slot identifier (0 is invalid, matching `blocks::Slot`).
    pub slot: u32,
    pub generation: u32,
    pub offset: u64,
}

/// Errors from pool or disk I/O operations.
#[derive(Debug)]
pub enum SpillError {
    Io(io::Error),
    InvalidDir(String),
    InvalidKvb(u32),
    Pool(String),
    InvalidLength { expected: usize, got: usize },
    PayloadTooLarge(usize),
}

impl fmt::Display for SpillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "spill I/O error: {e}"),
            Self::InvalidDir(s) => write!(f, "invalid spill directory: {s}"),
            Self::InvalidKvb(kvb) => write!(f, "invalid kvb {kvb}: expected 1 (--kv8) or 2 (fp16)"),
            Self::Pool(s) => write!(f, "pool error: {s}"),
            Self::InvalidLength { expected, got } => {
                write!(f, "payload length mismatch: expected {expected}, got {got}")
            }
            Self::PayloadTooLarge(n) => write!(f, "payload length {n} exceeds safety cap"),
        }
    }
}

impl From<io::Error> for SpillError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Reason a spilled block read fell back to a clean cache miss.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MissReason {
    NotFound,
    NotRegularFile,
    TruncatedHeader,
    BadMagic,
    UnsupportedVersion,
    BadBlockKind,
    KvbMismatch,
    MetadataMismatch,
    LengthMismatch,
    SizeMismatch,
    PayloadTooLarge,
    ChecksumMismatch,
    IoReadError,
}

/// Outcome of writing a block to [`SpillStore`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WriteOutcome {
    /// Persisted atomically and durably (`fsync` + `rename` + dir `fsync`).
    Written { on_disk_bytes: u64 },
    /// Refused because writing would exceed `--max-spill-bytes` even after LRU disk GC.
    OverBudget,
}

/// Verified block payload read back from SSD.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RestoredBlock {
    pub key: SpillKey,
    pub generation: u64,
    pub kv_bytes: Vec<u8>,
    pub recurrent_state: Option<Vec<u8>>,
}

/// Outcome of reading a block from [`SpillStore`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ReadOutcome {
    Hit(RestoredBlock),
    Miss(MissReason),
}

/// Clean adapter trait for Layer 3's `BlockPools` (`src/blocks.rs`).
///
/// When `layer3` merges, implementing this trait for `BlockPools` binds
/// [`SpillManager`] directly to Metal GPU slabs without modifying `spill.rs`.
pub trait SpillPoolBackend {
    fn stride_bytes(&self, kind: SpillBlockKind) -> usize;
    fn capacity_slots(&self, kind: SpillBlockKind) -> usize;
    fn used_slots(&self, kind: SpillBlockKind) -> usize;
    fn free_slots(&self, kind: SpillBlockKind) -> usize;

    /// Read immutable bytes of a sealed resident slot (generation-checked).
    fn read_slot_bytes(&self, slot: ResidentSlot) -> Result<&[u8], SpillError>;

    /// Take an unoccupied slot: zero-filled, generation bumped, unsealed, `refs == 0`.
    fn take_free_slot(&mut self, kind: SpillBlockKind) -> Result<ResidentSlot, SpillError>;

    /// Release an unreferenced resident slot (`refs == 0`), zeroing its bytes
    /// and bumping generation so any stale handle is rejected.
    fn release_cold_slot(&mut self, slot: ResidentSlot) -> Result<(), SpillError>;

    /// Install verified K/V bytes into a previously taken/reserved slot and seal it.
    fn install_restored_slot(&mut self, slot: ResidentSlot, kv_bytes: &[u8]) -> Result<(), SpillError>;

    /// Generation-checked refcount operations.
    fn add_ref(&mut self, slot: ResidentSlot) -> Result<u32, SpillError>;
    fn drop_ref(&mut self, slot: ResidentSlot) -> Result<u32, SpillError>;
    fn refs(&self, slot: ResidentSlot) -> Result<u32, SpillError>;
}

// ---------------------------------------------------------------------------
// HostBlockPoolBackend: mirrors src/blocks.rs BlockPools invariants
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct HostSlot {
    generation: u32,
    occupied: bool,
    sealed: bool,
    refs: u32,
    bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
struct HostKindPool {
    kind: SpillBlockKind,
    stride: usize,
    slots: Vec<HostSlot>,
    used: usize,
}

impl HostKindPool {
    fn new(kind: SpillBlockKind, capacity: usize, stride: usize) -> Self {
        let slots = (0..capacity)
            .map(|_| HostSlot {
                generation: 0,
                occupied: false,
                sealed: false,
                refs: 0,
                bytes: vec![0u8; stride],
            })
            .collect();
        Self { kind, stride, slots, used: 0 }
    }

    fn slot_index(&self, slot: ResidentSlot) -> Result<usize, SpillError> {
        if slot.kind != self.kind {
            return Err(SpillError::Pool("block kind mismatch".into()));
        }
        if slot.slot == 0 {
            return Err(SpillError::Pool("slot 0 is invalid".into()));
        }
        let idx = (slot.slot - 1) as usize;
        if idx >= self.slots.len() {
            return Err(SpillError::Pool("slot out of range".into()));
        }
        let rec = &self.slots[idx];
        if !rec.occupied || rec.generation != slot.generation {
            return Err(SpillError::Pool("stale slot generation".into()));
        }
        Ok(idx)
    }

    fn take_free(&mut self) -> Result<ResidentSlot, SpillError> {
        for (idx, rec) in self.slots.iter_mut().enumerate() {
            if !rec.occupied && rec.refs == 0 {
                let next_gen = rec
                    .generation
                    .checked_add(1)
                    .ok_or_else(|| SpillError::Pool("generation exhausted".into()))?;
                rec.bytes.fill(0);
                rec.generation = next_gen;
                rec.occupied = true;
                rec.sealed = false;
                rec.refs = 0;
                self.used += 1;
                let offset = (idx as u64)
                    .checked_mul(self.stride as u64)
                    .ok_or_else(|| SpillError::Pool("offset overflow".into()))?;
                return Ok(ResidentSlot {
                    kind: self.kind,
                    slot: (idx as u32) + 1,
                    generation: next_gen,
                    offset,
                });
            }
        }
        Err(SpillError::Pool("no free slots".into()))
    }

    fn release(&mut self, slot: ResidentSlot) -> Result<(), SpillError> {
        let idx = self.slot_index(slot)?;
        let rec = &mut self.slots[idx];
        if rec.refs != 0 {
            return Err(SpillError::Pool("cannot release slot with live refs".into()));
        }
        let next_gen = rec
            .generation
            .checked_add(1)
            .ok_or_else(|| SpillError::Pool("generation exhausted".into()))?;
        rec.bytes.fill(0);
        rec.generation = next_gen;
        rec.occupied = false;
        rec.sealed = false;
        self.used = self.used.saturating_sub(1);
        Ok(())
    }
}

/// Host-memory [`SpillPoolBackend`] mirroring Layer 3's `BlockPools`
/// (1-based slots, non-wrapping generation counters, zero-fill on reuse,
/// `refs == 0` guard on eviction).
#[derive(Clone, Debug)]
pub struct HostBlockPoolBackend {
    full: HostKindPool,
    window: HostKindPool,
}

impl HostBlockPoolBackend {
    pub fn new(
        full_capacity: usize,
        window_capacity: usize,
        full_stride: usize,
        window_stride: usize,
    ) -> Self {
        Self {
            full: HostKindPool::new(SpillBlockKind::Full, full_capacity, full_stride),
            window: HostKindPool::new(SpillBlockKind::Window, window_capacity, window_stride),
        }
    }

    fn pool(&self, kind: SpillBlockKind) -> &HostKindPool {
        match kind {
            SpillBlockKind::Full => &self.full,
            SpillBlockKind::Window => &self.window,
        }
    }

    fn pool_mut(&mut self, kind: SpillBlockKind) -> &mut HostKindPool {
        match kind {
            SpillBlockKind::Full => &mut self.full,
            SpillBlockKind::Window => &mut self.window,
        }
    }

    /// Write raw bytes into an occupied unsealed slot and seal it (helper for
    /// tests and prefill sealing).
    pub fn write_and_seal(&mut self, slot: ResidentSlot, kv_bytes: &[u8]) -> Result<(), SpillError> {
        self.install_restored_slot(slot, kv_bytes)
    }

    /// Inspect raw bytes of any slot index (1-based) to verify zero-fill on release.
    #[cfg(test)]
    fn raw_slot_bytes(&self, kind: SpillBlockKind, slot_one_based: u32) -> &[u8] {
        &self.pool(kind).slots[(slot_one_based - 1) as usize].bytes
    }
}

impl SpillPoolBackend for HostBlockPoolBackend {
    fn stride_bytes(&self, kind: SpillBlockKind) -> usize {
        self.pool(kind).stride
    }

    fn capacity_slots(&self, kind: SpillBlockKind) -> usize {
        self.pool(kind).slots.len()
    }

    fn used_slots(&self, kind: SpillBlockKind) -> usize {
        self.pool(kind).used
    }

    fn free_slots(&self, kind: SpillBlockKind) -> usize {
        let p = self.pool(kind);
        p.slots.len().saturating_sub(p.used)
    }

    fn read_slot_bytes(&self, slot: ResidentSlot) -> Result<&[u8], SpillError> {
        let p = self.pool(slot.kind);
        let idx = p.slot_index(slot)?;
        if !p.slots[idx].sealed {
            return Err(SpillError::Pool("cannot spill unsealed slot".into()));
        }
        Ok(&p.slots[idx].bytes)
    }

    fn take_free_slot(&mut self, kind: SpillBlockKind) -> Result<ResidentSlot, SpillError> {
        self.pool_mut(kind).take_free()
    }

    fn release_cold_slot(&mut self, slot: ResidentSlot) -> Result<(), SpillError> {
        self.pool_mut(slot.kind).release(slot)
    }

    fn install_restored_slot(&mut self, slot: ResidentSlot, kv_bytes: &[u8]) -> Result<(), SpillError> {
        let p = self.pool_mut(slot.kind);
        if kv_bytes.len() != p.stride {
            return Err(SpillError::InvalidLength { expected: p.stride, got: kv_bytes.len() });
        }
        let idx = p.slot_index(slot)?;
        if p.slots[idx].sealed {
            return Err(SpillError::Pool("sealed slot is immutable".into()));
        }
        p.slots[idx].bytes.copy_from_slice(kv_bytes);
        p.slots[idx].sealed = true;
        Ok(())
    }

    fn add_ref(&mut self, slot: ResidentSlot) -> Result<u32, SpillError> {
        let p = self.pool_mut(slot.kind);
        let idx = p.slot_index(slot)?;
        p.slots[idx].refs = p.slots[idx]
            .refs
            .checked_add(1)
            .ok_or_else(|| SpillError::Pool("refcount overflow".into()))?;
        Ok(p.slots[idx].refs)
    }

    fn drop_ref(&mut self, slot: ResidentSlot) -> Result<u32, SpillError> {
        let p = self.pool_mut(slot.kind);
        let idx = p.slot_index(slot)?;
        p.slots[idx].refs = p.slots[idx]
            .refs
            .checked_sub(1)
            .ok_or_else(|| SpillError::Pool("refcount underflow".into()))?;
        Ok(p.slots[idx].refs)
    }

    fn refs(&self, slot: ResidentSlot) -> Result<u32, SpillError> {
        let p = self.pool(slot.kind);
        let idx = p.slot_index(slot)?;
        Ok(p.slots[idx].refs)
    }
}

// ---------------------------------------------------------------------------
// On-disk checksummed block store (SpillStore)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct DiskEntry {
    bytes: u64,
    lru_seq: u64,
}

/// Statistics tracked by [`SpillStore`].
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct SpillStoreStats {
    pub writes_committed: u64,
    pub bytes_written: u64,
    pub reads_hit: u64,
    pub reads_miss: u64,
    pub corrupt_files_removed: u64,
    pub disk_gc_evictions: u64,
    pub over_budget_drops: u64,
    pub orphans_swept: u64,
}

/// Crash-safe, bounded, checksummed SSD directory store (`--spill-dir`).
pub struct SpillStore {
    // Keeping the locked directory open gives its index, writers, and GC a single owner.
    directory_lock: File,
    last_allocated_id: u64,
    dir: PathBuf,
    max_bytes: u64,
    used_bytes: u64,
    lru_clock: u64,
    kvb: u32,
    entries: HashMap<SpillKey, DiskEntry>,
    stats: SpillStoreStats,
}

impl Drop for SpillStore {
    fn drop(&mut self) {
        // Release immediately even if a concurrent fork briefly inherited the descriptor.
        let _ = self.directory_lock.unlock();
    }
}

fn enable_nocache(file: &File) {
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        unsafe {
            libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = file;
    }
}

fn fsync_dir(dir: &Path) -> io::Result<()> {
    let f = File::open(dir)?;
    f.sync_all()
}

fn unlink_if_present(path: &Path) -> io::Result<bool> {
    match fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

fn pread_exact(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        while !buf.is_empty() {
            let n = file.read_at(buf, offset)?;
            if n == 0 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected EOF in pread"));
            }
            buf = &mut buf[n..];
            offset = offset.checked_add(n as u64).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "pread offset overflow")
            })?;
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = file;
        f.seek(SeekFrom::Start(offset))?;
        f.read_exact(buf)
    }
}

fn encode_header(
    key: SpillKey,
    kvb: u32,
    generation: u64,
    kv_bytes: &[u8],
    recurrent_state: Option<&[u8]>,
) -> [u8; SPILL_HEADER_LEN] {
    let state_slice = recurrent_state.unwrap_or(&[]);
    let mut header = [0u8; SPILL_HEADER_LEN];
    header[0..8].copy_from_slice(&SPILL_MAGIC);
    header[8..12].copy_from_slice(&SPILL_FORMAT_VERSION.to_le_bytes());
    header[12..14].copy_from_slice(&(key.kind as u16).to_le_bytes());
    header[14..16].copy_from_slice(&(kvb as u16).to_le_bytes());
    header[16..24].copy_from_slice(&key.id.to_le_bytes());
    header[24..32].copy_from_slice(&generation.to_le_bytes());
    header[32..40].copy_from_slice(&(kv_bytes.len() as u64).to_le_bytes());
    header[40..48].copy_from_slice(&(state_slice.len() as u64).to_le_bytes());

    let mut hasher = Sha256::new();
    hasher.update(&header[0..HEADER_PREFIX_LEN]);
    hasher.update(kv_bytes);
    if !state_slice.is_empty() {
        hasher.update(state_slice);
    }
    let digest = hasher.finalize();
    header[48..80].copy_from_slice(&digest);
    header
}

impl SpillStore {
    /// Prepare `--spill-dir` (mode `0700` on Unix), sweep orphan `.tmp.*` files,
    /// index surviving `.blk` files, and enforce `max_bytes` via LRU disk GC (default fp16 `kvb = 2`).
    pub fn open(dir: impl AsRef<Path>, max_bytes: u64) -> Result<Self, SpillError> {
        Self::open_with_kvb(dir, max_bytes, 2)
    }

    /// Prepare `--spill-dir` with an explicit KV element width `kvb` (`1` for `--kv8`, `2` for fp16).
    pub fn open_with_kvb(dir: impl AsRef<Path>, max_bytes: u64, kvb: u32) -> Result<Self, SpillError> {
        if kvb != 1 && kvb != 2 {
            return Err(SpillError::InvalidKvb(kvb));
        }
        let dir = dir.as_ref().to_path_buf();
        prepare_spill_dir(&dir)?;
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        options.custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW);
        let directory_lock = options.open(&dir)?;
        match directory_lock.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => {
                return Err(SpillError::InvalidDir(format!("{} is already in use", dir.display())));
            }
            Err(std::fs::TryLockError::Error(err)) => return Err(err.into()),
        }

        let mut store = Self {
            directory_lock,
            last_allocated_id: 1,
            dir,
            max_bytes,
            used_bytes: 0,
            lru_clock: 0,
            kvb,
            entries: HashMap::new(),
            stats: SpillStoreStats::default(),
        };
        store.scan_and_sweep()?;
        Ok(store)
    }

    pub fn kvb(&self) -> u32 {
        self.kvb
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    pub fn stats(&self) -> SpillStoreStats {
        self.stats
    }

    pub fn contains(&self, key: SpillKey) -> bool {
        self.entries.contains_key(&key)
    }

    /// Allocate an identity for a new checkpoint block, skipping surviving blocks
    /// whose content belongs to an earlier generation of the in-memory checkpoint index.
    pub fn fresh_key(&mut self, kind: SpillBlockKind) -> SpillKey {
        loop {
            self.last_allocated_id = self.last_allocated_id.wrapping_add(1).max(1);
            let key = SpillKey::new(kind, self.last_allocated_id);
            if !self.contains(key) {
                return key;
            }
        }
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn block_path(&self, key: SpillKey) -> PathBuf {
        self.dir.join(key.blk_filename())
    }

    fn next_lru(&mut self) -> u64 {
        self.lru_clock = self.lru_clock.wrapping_add(1);
        self.lru_clock
    }

    fn scan_and_sweep(&mut self) -> Result<(), SpillError> {
        let mut discovered: Vec<(SpillKey, u64, u64)> = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            let meta = match fs::symlink_metadata(&path) {
                Ok(m) => m,
                Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e.into()),
            };
            if SpillKey::parse_tmp_filename(&name).is_some() {
                if meta.is_file() && unlink_if_present(&path)? {
                    self.stats.orphans_swept += 1;
                }
                continue;
            }
            if let Some(key) = SpillKey::parse_blk_filename(&name) {
                if !meta.file_type().is_file() {
                    // Never trust symlinks or non-regular files in spill dir.
                    if unlink_if_present(&path)? {
                        self.stats.corrupt_files_removed += 1;
                    }
                    continue;
                }
                let mtime_key = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0);
                discovered.push((key, meta.len(), mtime_key));
            }
        }
        // Order surviving blocks oldest-mtime first so LRU clock preserves relative age.
        discovered.sort_by_key(|&(key, _, mtime)| (mtime, key));
        for (key, len, _) in discovered {
            let seq = self.next_lru();
            self.entries.insert(key, DiskEntry { bytes: len, lru_seq: seq });
            self.used_bytes = self.used_bytes.saturating_add(len);
        }
        if self.stats.orphans_swept > 0 {
            let _ = fsync_dir(&self.dir);
        }
        // Enforce disk budget on startup if existing files exceed max_bytes.
        let empty = HashSet::new();
        self.evict_lru_to_fit(0, &empty);
        if self.used_bytes > self.max_bytes {
            return Err(SpillError::InvalidDir(format!(
                "could not reduce spill usage of {} bytes to the {} byte budget",
                self.used_bytes, self.max_bytes
            )));
        }
        Ok(())
    }

    /// Evict least-recently-used `.blk` files not in `protected` until
    /// `self.used_bytes + additional <= self.max_bytes` (returns evicted keys).
    pub fn evict_lru_to_fit(
        &mut self,
        additional: u64,
        protected: &HashSet<SpillKey>,
    ) -> Vec<SpillKey> {
        let mut evicted = Vec::new();
        let mut failed = HashSet::new();
        while self.used_bytes.saturating_add(additional) > self.max_bytes {
            let victim = self
                .entries
                .iter()
                .filter(|(k, _)| !protected.contains(k) && !failed.contains(*k))
                .min_by_key(|(k, e)| (e.lru_seq, **k))
                .map(|(k, _)| *k);
            let Some(victim_key) = victim else {
                break;
            };
            if self.remove_block(victim_key) {
                self.stats.disk_gc_evictions += 1;
                evicted.push(victim_key);
            } else {
                failed.insert(victim_key);
            }
        }
        evicted
    }

    /// Remove a specific block from disk if present.
    pub fn remove_block(&mut self, key: SpillKey) -> bool {
        let indexed = self.entries.contains_key(&key);
        self.unlink_block(key).is_ok_and(|removed| removed || indexed)
    }

    fn forget_block(&mut self, key: SpillKey) {
        if let Some(entry) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(entry.bytes);
        }
    }

    fn unlink_block(&mut self, key: SpillKey) -> io::Result<bool> {
        let removed = unlink_if_present(&self.block_path(key))?;
        self.forget_block(key);
        Ok(removed)
    }

    /// Crash-safe atomic write of a block (K/V + optional start-point recurrent state):
    /// enforce `--max-spill-bytes` with LRU disk GC, write `<tmp>` (`0600`), `fsync`,
    /// atomic `rename` to `<id>.blk`, and `fsync` directory.
    ///
    /// Returns `(WriteOutcome, evicted_disk_keys)` so callers can update tier metadata
    /// for any cold blocks GC'd from disk.
    pub fn write_block(
        &mut self,
        key: SpillKey,
        generation: u64,
        kv_bytes: &[u8],
        recurrent_state: Option<&[u8]>,
        protected: &HashSet<SpillKey>,
    ) -> Result<(WriteOutcome, Vec<SpillKey>), SpillError> {
        if kv_bytes.len() > MAX_KV_PAYLOAD_BYTES {
            return Err(SpillError::PayloadTooLarge(kv_bytes.len()));
        }
        let state_slice = recurrent_state.unwrap_or(&[]);
        if state_slice.len() > MAX_RECURRENT_STATE_BYTES {
            return Err(SpillError::PayloadTooLarge(state_slice.len()));
        }
        let total_bytes = (SPILL_HEADER_LEN as u64)
            .checked_add(kv_bytes.len() as u64)
            .and_then(|v| v.checked_add(state_slice.len() as u64))
            .ok_or_else(|| SpillError::PayloadTooLarge(usize::MAX))?;

        if total_bytes > self.max_bytes {
            self.stats.over_budget_drops += 1;
            return Ok((WriteOutcome::OverBudget, Vec::new()));
        }

        // A case-insensitive filesystem can resolve a canonical block path to an
        // unrelated, noncanonical filename deliberately left out of the index.
        let final_path = self.block_path(key);
        let existing_bytes = match self.entries.get(&key) {
            Some(entry) => entry.bytes,
            None => match fs::symlink_metadata(&final_path) {
                Err(e) if e.kind() == io::ErrorKind::NotFound => 0,
                Err(e) => return Err(e.into()),
                Ok(_) => {
                    return Err(io::Error::new(io::ErrorKind::AlreadyExists,
                        format!("unindexed file at {}", final_path.display())).into());
                }
            },
        };
        let net_additional = total_bytes.saturating_sub(existing_bytes);
        let mut protected_with_self = protected.clone();
        protected_with_self.insert(key);
        let evicted = self.evict_lru_to_fit(net_additional, &protected_with_self);
        if self.used_bytes.saturating_sub(existing_bytes).saturating_add(total_bytes) > self.max_bytes {
            self.stats.over_budget_drops += 1;
            return Ok((WriteOutcome::OverBudget, evicted));
        }

        let header = encode_header(key, self.kvb, generation, kv_bytes, recurrent_state);
        let pid = std::process::id();
        let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let tmp_path = self.dir.join(key.tmp_filename(pid, nonce));

        let write_result = (|| -> io::Result<()> {
            let mut opts = OpenOptions::new();
            opts.write(true).create_new(true);
            #[cfg(unix)]
            {
                opts.mode(0o600).custom_flags(libc::O_NOFOLLOW);
            }
            let mut file = opts.open(&tmp_path)?;
            #[cfg(unix)]
            {
                // Enforce strict 0600 even under permissive umasks.
                file.set_permissions(fs::Permissions::from_mode(0o600))?;
            }
            enable_nocache(&file);
            file.write_all(&header)?;
            file.write_all(kv_bytes)?;
            if !state_slice.is_empty() {
                file.write_all(state_slice)?;
            }
            file.sync_all()?;
            drop(file);
            fs::rename(&tmp_path, &final_path)?;
            fsync_dir(&self.dir)?;
            Ok(())
        })();

        if let Err(e) = write_result {
            let _ = fs::remove_file(&tmp_path);
            return Err(SpillError::Io(e));
        }

        if let Some(old) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(old.bytes);
        }
        let seq = self.next_lru();
        self.entries.insert(key, DiskEntry { bytes: total_bytes, lru_seq: seq });
        self.used_bytes = self.used_bytes.saturating_add(total_bytes);
        self.stats.writes_committed += 1;
        self.stats.bytes_written = self.stats.bytes_written.saturating_add(total_bytes);

        Ok((WriteOutcome::Written { on_disk_bytes: total_bytes }, evicted))
    }

    /// Read and cryptographically verify a block from SSD via `pread`.
    ///
    /// Any missing, truncated, oversized, mismatched, or checksum-failing file
    /// returns `ReadOutcome::Miss(reason)` (and unlinks the corrupt `.blk`),
    /// NEVER exposing unverified or corrupt bytes.
    pub fn read_block(&mut self, key: SpillKey, expected_kv_len: Option<usize>) -> ReadOutcome {
        let path = self.block_path(key);
        let mut opts = OpenOptions::new();
        opts.read(true);
        #[cfg(unix)]
        {
            opts.custom_flags(libc::O_NOFOLLOW);
        }
        let file = match opts.open(&path) {
            Ok(f) => f,
            Err(e) => {
                #[cfg(unix)]
                if e.raw_os_error() == Some(libc::ELOOP) {
                    self.drop_corrupt(key);
                    return ReadOutcome::Miss(MissReason::NotRegularFile);
                }
                self.stats.reads_miss += 1;
                let reason = if e.kind() == io::ErrorKind::NotFound {
                    self.forget_block(key);
                    MissReason::NotFound
                } else {
                    MissReason::IoReadError
                };
                return ReadOutcome::Miss(reason);
            }
        };

        enable_nocache(&file);

        let meta = match file.metadata() {
            Ok(m) => m,
            Err(_) => {
                self.drop_corrupt(key);
                return ReadOutcome::Miss(MissReason::IoReadError);
            }
        };
        if !meta.is_file() {
            self.drop_corrupt(key);
            return ReadOutcome::Miss(MissReason::NotRegularFile);
        }
        let file_len = meta.len();
        // Charge the observed length even when verification fails and cleanup cannot unlink it.
        let entry = self.entries.entry(key).or_insert(DiskEntry { bytes: 0, lru_seq: 0 });
        self.used_bytes = self.used_bytes.saturating_sub(entry.bytes).saturating_add(file_len);
        entry.bytes = file_len;
        if file_len < SPILL_HEADER_LEN as u64 {
            self.drop_corrupt(key);
            return ReadOutcome::Miss(MissReason::TruncatedHeader);
        }

        let mut header = [0u8; SPILL_HEADER_LEN];
        if pread_exact(&file, &mut header, 0).is_err() {
            self.drop_corrupt(key);
            return ReadOutcome::Miss(MissReason::TruncatedHeader);
        }

        if header[0..8] != SPILL_MAGIC {
            self.drop_corrupt(key);
            return ReadOutcome::Miss(MissReason::BadMagic);
        }
        let version = u32::from_le_bytes(header[8..12].try_into().unwrap());
        if version != SPILL_FORMAT_VERSION {
            self.drop_corrupt(key);
            return ReadOutcome::Miss(MissReason::UnsupportedVersion);
        }
        let raw_kind = u32::from(u16::from_le_bytes(header[12..14].try_into().unwrap()));
        let Some(kind) = SpillBlockKind::from_u32(raw_kind) else {
            self.drop_corrupt(key);
            return ReadOutcome::Miss(MissReason::BadBlockKind);
        };
        let file_kvb = u32::from(u16::from_le_bytes(header[14..16].try_into().unwrap()));
        if file_kvb != self.kvb {
            self.drop_corrupt(key);
            return ReadOutcome::Miss(MissReason::KvbMismatch);
        }
        let block_id = u64::from_le_bytes(header[16..24].try_into().unwrap());
        if kind != key.kind || block_id != key.id {
            self.drop_corrupt(key);
            return ReadOutcome::Miss(MissReason::MetadataMismatch);
        }
        let generation = u64::from_le_bytes(header[24..32].try_into().unwrap());
        let kv_len_u64 = u64::from_le_bytes(header[32..40].try_into().unwrap());
        let state_len_u64 = u64::from_le_bytes(header[40..48].try_into().unwrap());
        let stored_digest: [u8; 32] = header[48..80].try_into().unwrap();

        let (Ok(kv_len), Ok(state_len)) = (usize::try_from(kv_len_u64), usize::try_from(state_len_u64)) else {
            self.drop_corrupt(key);
            return ReadOutcome::Miss(MissReason::PayloadTooLarge);
        };
        if kv_len > MAX_KV_PAYLOAD_BYTES || state_len > MAX_RECURRENT_STATE_BYTES {
            self.drop_corrupt(key);
            return ReadOutcome::Miss(MissReason::PayloadTooLarge);
        }
        if let Some(exp) = expected_kv_len {
            if kv_len != exp {
                self.drop_corrupt(key);
                return ReadOutcome::Miss(MissReason::LengthMismatch);
            }
        }
        let expected_file_len = (SPILL_HEADER_LEN as u64)
            .checked_add(kv_len_u64)
            .and_then(|v| v.checked_add(state_len_u64));
        if expected_file_len != Some(file_len) {
            self.drop_corrupt(key);
            return ReadOutcome::Miss(MissReason::SizeMismatch);
        }

        let mut kv_bytes = vec![0u8; kv_len];
        if kv_len > 0 && pread_exact(&file, &mut kv_bytes, SPILL_HEADER_LEN as u64).is_err() {
            self.drop_corrupt(key);
            return ReadOutcome::Miss(MissReason::IoReadError);
        }

        let recurrent_state = if state_len > 0 {
            let mut st = vec![0u8; state_len];
            let st_off = (SPILL_HEADER_LEN as u64).saturating_add(kv_len_u64);
            if pread_exact(&file, &mut st, st_off).is_err() {
                self.drop_corrupt(key);
                return ReadOutcome::Miss(MissReason::IoReadError);
            }
            Some(st)
        } else {
            None
        };

        let mut hasher = Sha256::new();
        hasher.update(&header[0..HEADER_PREFIX_LEN]);
        hasher.update(&kv_bytes);
        if let Some(st) = &recurrent_state {
            hasher.update(st);
        }
        let computed = hasher.finalize();
        if !ct_eq_32(&computed, &stored_digest) {
            self.drop_corrupt(key);
            return ReadOutcome::Miss(MissReason::ChecksumMismatch);
        }

        let seq = self.next_lru();
        self.entries.get_mut(&key).expect("indexed before verification").lru_seq = seq;
        self.stats.reads_hit += 1;

        ReadOutcome::Hit(RestoredBlock {
            key,
            generation,
            kv_bytes,
            recurrent_state,
        })
    }

    fn drop_corrupt(&mut self, key: SpillKey) {
        if matches!(self.unlink_block(key), Ok(true)) {
            self.stats.corrupt_files_removed += 1;
        }
        self.stats.reads_miss += 1;
    }
}

fn prepare_spill_dir(dir: &Path) -> Result<(), SpillError> {
    match fs::symlink_metadata(dir) {
        Ok(meta) => {
            if meta.file_type().is_symlink() {
                return Err(SpillError::InvalidDir(format!(
                    "{} must not be a symlink",
                    dir.display()
                )));
            }
            if !meta.is_dir() {
                return Err(SpillError::InvalidDir(format!(
                    "{} exists and is not a directory",
                    dir.display()
                )));
            }
            #[cfg(unix)]
            {
                let mode = meta.mode() & 0o777;
                if mode != 0o700 {
                    fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
                }
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            let mut builder = fs::DirBuilder::new();
            builder.recursive(true);
            #[cfg(unix)]
            {
                builder.mode(0o700);
            }
            builder.create(dir)?;
            #[cfg(unix)]
            {
                fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
            }
        }
        Err(e) => return Err(SpillError::Io(e)),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Lazy LRU Tier Manager + Atomic Working-Set Reservation (SpillManager)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct ManagedBlock {
    tier: BlockTier,
    resident: Option<ResidentSlot>,
    recurrent_state: Option<Vec<u8>>,
    last_used_seq: u64,
    /// Number of in-flight disk I/O operations (spills or pending restores)
    /// referencing this block; blocks with `io_pins > 0` are never evicted
    /// from memory or disk.
    io_pins: u32,
}

/// Errors from [`SpillManager::reserve_working_set`] and [`SpillManager::alloc_slot_lazy`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReserveError {
    /// Requested working set exceeds hard pool slot capacity and can never fit.
    ExceedsCapacity {
        need_full: usize,
        cap_full: usize,
        need_window: usize,
        cap_window: usize,
    },
    /// Not enough free + evictable (`refs == 0`) slots right now; lane should
    /// wait in `Restoring` without performing partial reads or thrashing.
    WouldBlock {
        need_full: usize,
        avail_full: usize,
        need_window: usize,
        avail_window: usize,
    },
    /// Underlying pool or I/O error.
    BackendFailed,
}

/// Errors from [`SpillManager::prepare_working_set_restore`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrepareRestoreError {
    /// At least one block in the requested working set is absent from both memory and disk.
    CacheMiss(SpillKey),
    /// Atomic reservation could not be satisfied right now (zero mutations occurred).
    Reserve(ReserveError),
}

/// Exclusive, atomically-reserved pool slots held by a `Restoring` lane before
/// issuing any SSD reads.
///
/// Every slot inside a `WorkingSetReservation` is already claimed (`refs == 1`)
/// so no concurrent lane can steal or evict it while SSD reads are in flight.
#[derive(Debug)]
pub struct WorkingSetReservation {
    id: u64,
    full_slots: Vec<ResidentSlot>,
    window_slots: Vec<ResidentSlot>,
}

impl WorkingSetReservation {
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn full_count(&self) -> usize {
        self.full_slots.len()
    }

    pub fn window_count(&self) -> usize {
        self.window_slots.len()
    }

    pub fn full_slots(&self) -> &[ResidentSlot] {
        &self.full_slots
    }

    pub fn window_slots(&self) -> &[ResidentSlot] {
        &self.window_slots
    }
}

/// A fully prepared restore plan: already-resident blocks are pinned so they
/// cannot be evicted, and every disk-only block has an atomically reserved slot.
#[derive(Debug)]
pub struct PreparedRestore {
    pub pinned_in_memory: Vec<(SpillKey, ResidentSlot)>,
    pub disk_full: Vec<SpillKey>,
    pub disk_window: Vec<SpillKey>,
    pub reservation: WorkingSetReservation,
}

/// Completed, checksum-verified working set ready for `Ready` / `InStep`.
#[derive(Clone, Debug)]
pub struct RestoredWorkingSet {
    pub full_slots: Vec<(SpillKey, ResidentSlot)>,
    pub window_slots: Vec<(SpillKey, ResidentSlot)>,
    pub recurrent_states: HashMap<SpillKey, Vec<u8>>,
}

/// Clean cache-miss rollback result during SSD restore execution.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RestoreWorkingSetMiss {
    pub failed_key: SpillKey,
    pub reason: MissReason,
}

/// Statistics tracked by [`SpillManager`].
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct SpillManagerStats {
    pub lazy_spills_written: u64,
    pub clean_mem_evictions: u64,
    pub over_budget_mem_drops: u64,
    pub reservations_granted: u64,
    pub reservations_would_block: u64,
    pub reservations_released: u64,
    pub blocks_restored_from_disk: u64,
    pub restore_cache_misses: u64,
}

/// Coordinates lazy LRU spilling between a [`SpillPoolBackend`] and [`SpillStore`],
/// plus atomic working-set reservation for `Restoring` lanes.
pub struct SpillManager<P: SpillPoolBackend> {
    pool: P,
    store: SpillStore,
    blocks: HashMap<SpillKey, ManagedBlock>,
    lru_clock: u64,
    next_reservation_id: u64,
    active_reservations: HashSet<u64>,
    stats: SpillManagerStats,
}

impl<P: SpillPoolBackend> SpillManager<P> {
    pub fn new(pool: P, store: SpillStore) -> Self {
        Self {
            pool,
            store,
            blocks: HashMap::new(),
            lru_clock: 0,
            next_reservation_id: 1,
            active_reservations: HashSet::new(),
            stats: SpillManagerStats::default(),
        }
    }

    pub fn pool(&self) -> &P {
        &self.pool
    }

    pub fn pool_mut(&mut self) -> &mut P {
        &mut self.pool
    }

    pub fn store(&self) -> &SpillStore {
        &self.store
    }

    pub fn stats(&self) -> SpillManagerStats {
        self.stats
    }

    pub fn active_reservation_count(&self) -> usize {
        self.active_reservations.len()
    }

    pub fn tier(&self, key: SpillKey) -> BlockTier {
        match self.blocks.get(&key) {
            Some(b) => b.tier,
            None => {
                if self.store.contains(key) {
                    BlockTier::Disk
                } else {
                    BlockTier::None
                }
            }
        }
    }

    pub fn resident_slot(&self, key: SpillKey) -> Option<ResidentSlot> {
        self.blocks.get(&key).and_then(|b| b.resident)
    }

    pub fn in_memory_recurrent_state(&self, key: SpillKey) -> Option<&[u8]> {
        self.blocks.get(&key).and_then(|b| b.recurrent_state.as_deref())
    }

    fn next_lru(&mut self) -> u64 {
        self.lru_clock = self.lru_clock.wrapping_add(1);
        self.lru_clock
    }

    /// Allocate one slot for a newly written block (e.g. a lane's private tail).
    ///
    /// Lazily spills **at most one** LRU unreferenced block of `kind` ONLY when
    /// `pool.free_slots(kind) == 0`. Never writes anything eagerly.
    pub fn alloc_slot_lazy(&mut self, kind: SpillBlockKind) -> Result<ResidentSlot, ReserveError> {
        if self.pool.capacity_slots(kind) == 0 {
            return Err(ReserveError::ExceedsCapacity {
                need_full: usize::from(kind == SpillBlockKind::Full),
                cap_full: self.pool.capacity_slots(SpillBlockKind::Full),
                need_window: usize::from(kind == SpillBlockKind::Window),
                cap_window: self.pool.capacity_slots(SpillBlockKind::Window),
            });
        }
        if self.pool.free_slots(kind) == 0 {
            if self.evictable_count(kind) == 0 {
                return Err(ReserveError::WouldBlock {
                    need_full: usize::from(kind == SpillBlockKind::Full),
                    avail_full: self.pool.free_slots(SpillBlockKind::Full)
                        + self.evictable_count(SpillBlockKind::Full),
                    need_window: usize::from(kind == SpillBlockKind::Window),
                    avail_window: self.pool.free_slots(SpillBlockKind::Window)
                        + self.evictable_count(SpillBlockKind::Window),
                });
            }
            self.evict_one_lru_cold(kind).map_err(|_| ReserveError::BackendFailed)?;
        }
        self.pool.take_free_slot(kind).map_err(|_| ReserveError::BackendFailed)
    }

    /// Register a newly sealed resident block into the tier table (`BlockTier::Memory`).
    ///
    /// **Lazy invariant**: performs ZERO disk writes.
    pub fn register_sealed_block(
        &mut self,
        key: SpillKey,
        slot: ResidentSlot,
        recurrent_state: Option<Vec<u8>>,
    ) -> Result<(), SpillError> {
        if slot.kind != key.kind {
            return Err(SpillError::Pool("block kind mismatch on register".into()));
        }
        // Verify slot is sealed and readable before admitting to the tier table.
        let _ = self.pool.read_slot_bytes(slot)?;
        let seq = self.next_lru();
        let on_disk = self.store.contains(key);
        let tier = if on_disk { BlockTier::Both } else { BlockTier::Memory };
        self.blocks.insert(
            key,
            ManagedBlock {
                tier,
                resident: Some(slot),
                recurrent_state,
                last_used_seq: seq,
                io_pins: 0,
            },
        );
        Ok(())
    }

    /// Register a pre-existing on-disk block discovered across restart so
    /// `prepare_working_set_restore` can locate it.
    pub fn index_existing_disk_blocks(&mut self) {
        let keys: Vec<SpillKey> = self.store.entries.keys().copied().collect();
        for key in keys {
            if !self.blocks.contains_key(&key) {
                let seq = self.next_lru();
                self.blocks.insert(
                    key,
                    ManagedBlock {
                        tier: BlockTier::Disk,
                        resident: None,
                        recurrent_state: None,
                        last_used_seq: seq,
                        io_pins: 0,
                    },
                );
            }
        }
    }

    /// Pin a resident block (`refs += 1`) and refresh its LRU recency.
    pub fn pin(&mut self, key: SpillKey) -> Result<ResidentSlot, SpillError> {
        let slot = self
            .blocks
            .get(&key)
            .and_then(|b| b.resident)
            .ok_or_else(|| SpillError::Pool("cannot pin non-resident block".into()))?;
        self.pool.add_ref(slot)?;
        let seq = self.next_lru();
        if let Some(b) = self.blocks.get_mut(&key) {
            b.last_used_seq = seq;
        }
        Ok(slot)
    }

    /// Unpin a resident block (`refs -= 1`) and refresh its LRU recency.
    ///
    /// **Lazy invariant**: dropping `refs` to `0` keeps the block warm in
    /// memory and performs ZERO disk writes.
    pub fn unpin(&mut self, key: SpillKey) -> Result<u32, SpillError> {
        let slot = self
            .blocks
            .get(&key)
            .and_then(|b| b.resident)
            .ok_or_else(|| SpillError::Pool("cannot unpin non-resident block".into()))?;
        let refs = self.pool.drop_ref(slot)?;
        let seq = self.next_lru();
        if let Some(b) = self.blocks.get_mut(&key) {
            b.last_used_seq = seq;
        }
        Ok(refs)
    }

    /// Refresh a block's LRU recency (e.g. on prefix cache hit).
    pub fn touch(&mut self, key: SpillKey) {
        let seq = self.next_lru();
        if let Some(b) = self.blocks.get_mut(&key) {
            b.last_used_seq = seq;
        }
    }

    /// Count resident sealed blocks of `kind` eligible for lazy LRU spill (`refs == 0`, no active I/O).
    pub fn evictable_count(&self, kind: SpillBlockKind) -> usize {
        self.blocks
            .iter()
            .filter(|(k, b)| {
                k.kind == kind
                    && b.tier.in_memory()
                    && b.io_pins == 0
                    && b.resident
                        .and_then(|s| self.pool.refs(s).ok())
                        .map(|r| r == 0)
                        .unwrap_or(false)
            })
            .count()
    }

    fn pick_lru_evictable(&self, kind: SpillBlockKind) -> Option<SpillKey> {
        self.blocks
            .iter()
            .filter(|(k, b)| {
                k.kind == kind
                    && b.tier.in_memory()
                    && b.io_pins == 0
                    && b.resident
                        .and_then(|s| self.pool.refs(s).ok())
                        .map(|r| r == 0)
                        .unwrap_or(false)
            })
            .min_by_key(|(k, b)| (b.last_used_seq, **k))
            .map(|(k, _)| *k)
    }

    /// Evict the single least-recently-used unreferenced resident block of `kind`.
    ///
    /// - If already `BlockTier::Both`, skips redundant disk write and frees memory slot.
    /// - If `BlockTier::Memory`, spills K/V + recurrent state atomically to SSD,
    ///   updates tier to `Disk` (or `None` if over `--max-spill-bytes`), and frees memory slot.
    fn evict_one_lru_cold(&mut self, kind: SpillBlockKind) -> Result<SpillKey, SpillError> {
        let key = self
            .pick_lru_evictable(kind)
            .ok_or_else(|| SpillError::Pool("no evictable cold block".into()))?;
        let (tier, slot) = {
            let b = self.blocks.get_mut(&key).unwrap();
            b.io_pins = b.io_pins.saturating_add(1);
            (b.tier, b.resident.expect("resident slot"))
        };

        if tier == BlockTier::Both && self.store.contains(key) {
            // Already cleanly persisted on disk; drop memory copy without rewriting.
            self.pool.release_cold_slot(slot)?;
            if let Some(b) = self.blocks.get_mut(&key) {
                b.tier = BlockTier::Disk;
                b.resident = None;
                b.recurrent_state = None;
                b.io_pins = b.io_pins.saturating_sub(1);
            }
            self.stats.clean_mem_evictions += 1;
            return Ok(key);
        }

        // Copy immutable slot bytes and optional recurrent state for crash-safe writeback.
        let kv_copy = self.pool.read_slot_bytes(slot)?.to_vec();
        let state_copy = self.blocks.get(&key).and_then(|b| b.recurrent_state.clone());

        // Protect any disk block currently undergoing I/O (active spill or pending restore) from disk GC.
        let protected: HashSet<SpillKey> = self
            .blocks
            .iter()
            .filter(|(_, b)| b.io_pins > 0)
            .map(|(k, _)| *k)
            .collect();

        let write_res = self.store.write_block(
            key,
            u64::from(slot.generation),
            &kv_copy,
            state_copy.as_deref(),
            &protected,
        );

        let (outcome, disk_gc_evicted) = match write_res {
            Ok(v) => v,
            Err(_) => {
                // Disk write failed: unreferenced cold block is cleanly dropped from memory
                // (loses cache, never correctness).
                self.pool.release_cold_slot(slot)?;
                self.blocks.remove(&key);
                self.stats.over_budget_mem_drops += 1;
                return Ok(key);
            }
        };

        // Update tier metadata for any cold blocks evicted by disk GC.
        for gc_key in disk_gc_evicted {
            if let Some(gb) = self.blocks.get_mut(&gc_key) {
                match gb.tier {
                    BlockTier::Disk => {
                        self.blocks.remove(&gc_key);
                    }
                    BlockTier::Both => {
                        gb.tier = BlockTier::Memory;
                    }
                    BlockTier::Memory | BlockTier::None => {}
                }
            }
        }

        // Release the memory slot now that disk writeback has finished.
        self.pool.release_cold_slot(slot)?;

        match outcome {
            WriteOutcome::Written { .. } => {
                if let Some(b) = self.blocks.get_mut(&key) {
                    b.tier = BlockTier::Disk;
                    b.resident = None;
                    b.recurrent_state = None;
                    b.io_pins = b.io_pins.saturating_sub(1);
                }
                self.stats.lazy_spills_written += 1;
            }
            WriteOutcome::OverBudget => {
                // Could not fit inside --max-spill-bytes: drop unreferenced cache block cleanly.
                self.blocks.remove(&key);
                self.stats.over_budget_mem_drops += 1;
            }
        }

        Ok(key)
    }

    /// Atomically reserve `count_full` Full slots and `count_window` Window slots
    /// before performing any SSD reads (`docs/ROADMAP.md:137`).
    ///
    /// - If either kind exceeds hard pool capacity -> `Err(ReserveError::ExceedsCapacity)` (0 mutations).
    /// - If `free + evictable` cannot cover BOTH `count_full` AND `count_window`
    ///   simultaneously -> `Err(ReserveError::WouldBlock)` with **ZERO spills and
    ///   ZERO partial allocations** so the lane waits cleanly in `Restoring`.
    /// - Otherwise lazily spills the exact shortfall in strict LRU order, takes all
    ///   required slots, and pins them (`refs == 1`) inside the returned
    ///   [`WorkingSetReservation`] so concurrent lanes cannot evict or steal them.
    pub fn reserve_working_set(
        &mut self,
        count_full: usize,
        count_window: usize,
    ) -> Result<WorkingSetReservation, ReserveError> {
        let cap_full = self.pool.capacity_slots(SpillBlockKind::Full);
        let cap_window = self.pool.capacity_slots(SpillBlockKind::Window);
        if count_full > cap_full || count_window > cap_window {
            return Err(ReserveError::ExceedsCapacity {
                need_full: count_full,
                cap_full,
                need_window: count_window,
                cap_window,
            });
        }

        let free_full = self.pool.free_slots(SpillBlockKind::Full);
        let evict_full = self.evictable_count(SpillBlockKind::Full);
        let avail_full = free_full.saturating_add(evict_full);

        let free_window = self.pool.free_slots(SpillBlockKind::Window);
        let evict_window = self.evictable_count(SpillBlockKind::Window);
        let avail_window = free_window.saturating_add(evict_window);

        if count_full > avail_full || count_window > avail_window {
            self.stats.reservations_would_block += 1;
            return Err(ReserveError::WouldBlock {
                need_full: count_full,
                avail_full,
                need_window: count_window,
                avail_window,
            });
        }

        // Both kinds are guaranteed to fit. Trigger lazy LRU spills ONLY for shortfalls.
        let mut full_slots = Vec::with_capacity(count_full);
        let mut window_slots = Vec::with_capacity(count_window);

        let claim = (|| -> Result<(), SpillError> {
            for _ in 0..count_full {
                if self.pool.free_slots(SpillBlockKind::Full) == 0 {
                    self.evict_one_lru_cold(SpillBlockKind::Full)?;
                }
                let s = self.pool.take_free_slot(SpillBlockKind::Full)?;
                self.pool.add_ref(s)?;
                full_slots.push(s);
            }
            for _ in 0..count_window {
                if self.pool.free_slots(SpillBlockKind::Window) == 0 {
                    self.evict_one_lru_cold(SpillBlockKind::Window)?;
                }
                let s = self.pool.take_free_slot(SpillBlockKind::Window)?;
                self.pool.add_ref(s)?;
                window_slots.push(s);
            }
            Ok(())
        })();

        if claim.is_err() {
            for s in full_slots.into_iter().chain(window_slots.into_iter()) {
                let _ = self.pool.drop_ref(s);
                let _ = self.pool.release_cold_slot(s);
            }
            return Err(ReserveError::BackendFailed);
        }

        let id = self.next_reservation_id;
        self.next_reservation_id = self.next_reservation_id.wrapping_add(1);
        self.active_reservations.insert(id);
        self.stats.reservations_granted += 1;

        Ok(WorkingSetReservation { id, full_slots, window_slots })
    }

    /// Release an unused [`WorkingSetReservation`] (e.g. lane cancelled during
    /// `Restoring`), returning every reserved slot to free & zero-filled state.
    pub fn release_reservation(&mut self, reservation: WorkingSetReservation) {
        self.active_reservations.remove(&reservation.id);
        for s in reservation.full_slots.into_iter().chain(reservation.window_slots.into_iter()) {
            let _ = self.pool.drop_ref(s);
            let _ = self.pool.release_cold_slot(s);
        }
        self.stats.reservations_released += 1;
    }

    /// Atomically prepare a lane's entire start-point working set (`full_keys` +
    /// preceding `window_keys`) before issuing ANY SSD read:
    /// 1. Validates all requested keys exist in memory or on disk.
    /// 2. Pins already-resident blocks FIRST so spilling for disk-only blocks
    ///    can NEVER evict this lane's own in-memory prefix blocks.
    /// 3. Calls [`Self::reserve_working_set`] atomically for the exact disk-only
    ///    shortfall; on `WouldBlock` / `ExceedsCapacity`, unpins everything it
    ///    pinned so zero state changes occur.
    pub fn prepare_working_set_restore(
        &mut self,
        full_keys: &[SpillKey],
        window_keys: &[SpillKey],
    ) -> Result<PreparedRestore, PrepareRestoreError> {
        let mut seen = HashSet::new();
        let mut in_mem_keys = Vec::new();
        let mut disk_full = Vec::new();
        let mut disk_window = Vec::new();

        for &key in full_keys {
            if key.kind != SpillBlockKind::Full {
                return Err(PrepareRestoreError::CacheMiss(key));
            }
            if !seen.insert(key) {
                continue;
            }
            match self.tier(key) {
                BlockTier::Memory | BlockTier::Both => in_mem_keys.push(key),
                BlockTier::Disk => disk_full.push(key),
                BlockTier::None => return Err(PrepareRestoreError::CacheMiss(key)),
            }
        }
        for &key in window_keys {
            if key.kind != SpillBlockKind::Window {
                return Err(PrepareRestoreError::CacheMiss(key));
            }
            if !seen.insert(key) {
                continue;
            }
            match self.tier(key) {
                BlockTier::Memory | BlockTier::Both => in_mem_keys.push(key),
                BlockTier::Disk => disk_window.push(key),
                BlockTier::None => return Err(PrepareRestoreError::CacheMiss(key)),
            }
        }

        // Pin already-resident blocks BEFORE computing evictable counts so a lane
        // can never evict its own working set while spilling for its disk blocks.
        let mut pinned_in_memory = Vec::with_capacity(in_mem_keys.len());
        for &key in &in_mem_keys {
            match self.pin(key) {
                Ok(slot) => pinned_in_memory.push((key, slot)),
                Err(_) => {
                    for (pk, _) in pinned_in_memory {
                        let _ = self.unpin(pk);
                    }
                    return Err(PrepareRestoreError::CacheMiss(key));
                }
            }
        }

        // Pin disk-only blocks against disk GC while this lane is in Restoring!
        for &key in disk_full.iter().chain(disk_window.iter()) {
            let seq = self.next_lru();
            let entry = self.blocks.entry(key).or_insert_with(|| ManagedBlock {
                tier: BlockTier::Disk,
                resident: None,
                recurrent_state: None,
                last_used_seq: seq,
                io_pins: 0,
            });
            entry.io_pins = entry.io_pins.saturating_add(1);
        }

        match self.reserve_working_set(disk_full.len(), disk_window.len()) {
            Ok(reservation) => Ok(PreparedRestore {
                pinned_in_memory,
                disk_full,
                disk_window,
                reservation,
            }),
            Err(e) => {
                // Atomic rollback: unpin disk I/O pins and every in-memory block we pinned.
                for &key in disk_full.iter().chain(disk_window.iter()) {
                    if let Some(b) = self.blocks.get_mut(&key) {
                        b.io_pins = b.io_pins.saturating_sub(1);
                    }
                }
                for (pk, _) in pinned_in_memory {
                    let _ = self.unpin(pk);
                }
                Err(PrepareRestoreError::Reserve(e))
            }
        }
    }

    /// Abort a [`PreparedRestore`] cleanly (e.g. lane cancelled before I/O).
    pub fn abort_prepared_restore(&mut self, prepared: PreparedRestore) {
        for &key in prepared.disk_full.iter().chain(prepared.disk_window.iter()) {
            if let Some(b) = self.blocks.get_mut(&key) {
                b.io_pins = b.io_pins.saturating_sub(1);
            }
        }
        self.release_reservation(prepared.reservation);
        for (key, _) in prepared.pinned_in_memory {
            let _ = self.unpin(key);
        }
    }

    /// Execute SSD reads for a [`PreparedRestore`].
    ///
    /// Every block is read and SHA-256 verified *before* installation into its
    /// pre-reserved slot. If any block is missing or corrupt, all reserved slots
    /// and in-memory pins are cleanly rolled back and `Err(RestoreWorkingSetMiss)`
    /// is returned so the lane falls back to clean prefill recomputation.
    ///
    /// If another concurrent restoring lane already installed a shared block
    /// into memory before this call ran, this method deduplicates onto the
    /// existing resident slot (`add_ref`) and releases its redundant reserved slot.
    pub fn execute_working_set_restore(
        &mut self,
        prepared: PreparedRestore,
    ) -> Result<RestoredWorkingSet, RestoreWorkingSetMiss> {
        let PreparedRestore {
            pinned_in_memory,
            disk_full,
            disk_window,
            reservation,
        } = prepared;

        let full_stride = self.pool.stride_bytes(SpillBlockKind::Full);
        let window_stride = self.pool.stride_bytes(SpillBlockKind::Window);

        let mut verified_full = Vec::with_capacity(disk_full.len());
        let mut verified_window = Vec::with_capacity(disk_window.len());

        let release_io_pins = |blocks: &mut HashMap<SpillKey, ManagedBlock>,
                               df: &[SpillKey],
                               dw: &[SpillKey]| {
            for &k in df.iter().chain(dw.iter()) {
                if let Some(b) = blocks.get_mut(&k) {
                    b.io_pins = b.io_pins.saturating_sub(1);
                }
            }
        };

        // Phase 1: read and SHA-256 verify every disk block BEFORE installing into slots.
        for &key in &disk_full {
            if self.blocks.get(&key).and_then(|b| b.resident).is_some() {
                verified_full.push(None);
                continue;
            }
            match self.store.read_block(key, Some(full_stride)) {
                ReadOutcome::Hit(blk) => verified_full.push(Some(blk)),
                ReadOutcome::Miss(reason) => {
                    release_io_pins(&mut self.blocks, &disk_full, &disk_window);
                    self.blocks.remove(&key);
                    self.release_reservation(reservation);
                    for (pk, _) in pinned_in_memory {
                        let _ = self.unpin(pk);
                    }
                    self.stats.restore_cache_misses += 1;
                    return Err(RestoreWorkingSetMiss { failed_key: key, reason });
                }
            }
        }
        for &key in &disk_window {
            if self.blocks.get(&key).and_then(|b| b.resident).is_some() {
                verified_window.push(None);
                continue;
            }
            match self.store.read_block(key, Some(window_stride)) {
                ReadOutcome::Hit(blk) => verified_window.push(Some(blk)),
                ReadOutcome::Miss(reason) => {
                    release_io_pins(&mut self.blocks, &disk_full, &disk_window);
                    self.blocks.remove(&key);
                    self.release_reservation(reservation);
                    for (pk, _) in pinned_in_memory {
                        let _ = self.unpin(pk);
                    }
                    self.stats.restore_cache_misses += 1;
                    return Err(RestoreWorkingSetMiss { failed_key: key, reason });
                }
            }
        }

        // Phase 2: all checksums passed; install verified bytes or dedup onto existing slot.
        release_io_pins(&mut self.blocks, &disk_full, &disk_window);
        self.active_reservations.remove(&reservation.id);
        let mut full_slots = Vec::new();
        let mut window_slots = Vec::new();
        let mut recurrent_states = HashMap::new();

        for (key, slot) in pinned_in_memory {
            if let Some(st) = self.in_memory_recurrent_state(key).map(|s| s.to_vec()) {
                recurrent_states.insert(key, st);
            }
            match key.kind {
                SpillBlockKind::Full => full_slots.push((key, slot)),
                SpillBlockKind::Window => window_slots.push((key, slot)),
            }
        }

        for ((&key, maybe_blk), slot) in disk_full
            .iter()
            .zip(verified_full.into_iter())
            .zip(reservation.full_slots.into_iter())
        {
            if let Some(existing) = self.blocks.get(&key).and_then(|b| b.resident) {
                // Concurrent lane already restored `key`; pin existing slot and free redundant slot.
                self.pool.add_ref(existing).expect("pin existing restored full slot");
                let _ = self.pool.drop_ref(slot);
                let _ = self.pool.release_cold_slot(slot);
                if let Some(st) = self.in_memory_recurrent_state(key).map(|s| s.to_vec()) {
                    recurrent_states.insert(key, st);
                }
                self.touch(key);
                full_slots.push((key, existing));
            } else {
                let blk = maybe_blk.expect("verified full block present");
                self.pool
                    .install_restored_slot(slot, &blk.kv_bytes)
                    .expect("verified block fits pre-reserved full slot");
                let seq = self.next_lru();
                let io_pins = self.blocks.get(&key).map(|b| b.io_pins).unwrap_or(0);
                if let Some(st) = &blk.recurrent_state {
                    recurrent_states.insert(blk.key, st.clone());
                }
                self.blocks.insert(
                    blk.key,
                    ManagedBlock {
                        tier: BlockTier::Both,
                        resident: Some(slot),
                        recurrent_state: blk.recurrent_state,
                        last_used_seq: seq,
                        io_pins,
                    },
                );
                full_slots.push((blk.key, slot));
                self.stats.blocks_restored_from_disk += 1;
            }
        }

        for ((&key, maybe_blk), slot) in disk_window
            .iter()
            .zip(verified_window.into_iter())
            .zip(reservation.window_slots.into_iter())
        {
            if let Some(existing) = self.blocks.get(&key).and_then(|b| b.resident) {
                self.pool.add_ref(existing).expect("pin existing restored window slot");
                let _ = self.pool.drop_ref(slot);
                let _ = self.pool.release_cold_slot(slot);
                if let Some(st) = self.in_memory_recurrent_state(key).map(|s| s.to_vec()) {
                    recurrent_states.insert(key, st);
                }
                self.touch(key);
                window_slots.push((key, existing));
            } else {
                let blk = maybe_blk.expect("verified window block present");
                self.pool
                    .install_restored_slot(slot, &blk.kv_bytes)
                    .expect("verified block fits pre-reserved window slot");
                let seq = self.next_lru();
                let io_pins = self.blocks.get(&key).map(|b| b.io_pins).unwrap_or(0);
                if let Some(st) = &blk.recurrent_state {
                    recurrent_states.insert(blk.key, st.clone());
                }
                self.blocks.insert(
                    blk.key,
                    ManagedBlock {
                        tier: BlockTier::Both,
                        resident: Some(slot),
                        recurrent_state: blk.recurrent_state,
                        last_used_seq: seq,
                        io_pins,
                    },
                );
                window_slots.push((blk.key, slot));
                self.stats.blocks_restored_from_disk += 1;
            }
        }

        Ok(RestoredWorkingSet {
            full_slots,
            window_slots,
            recurrent_states,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests & Real `kill -9` (SIGKILL) Gate
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    /// RAII scratch test directory created under `target/spill-tests/` (never `/tmp`).
    struct ScratchTestDir {
        path: PathBuf,
    }

    impl ScratchTestDir {
        fn new(label: &str) -> Self {
            let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("target")
                .join("spill-tests");
            fs::create_dir_all(&base).expect("create target/spill-tests");
            #[cfg(unix)]
            let _ = fs::set_permissions(&base, fs::Permissions::from_mode(0o700));
            let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
            let path = base.join(format!("{label}-{}-{nonce}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("create scratch test dir");
            #[cfg(unix)]
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700))
                .expect("chmod 0700 scratch test dir");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for ScratchTestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn hex(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            use std::fmt::Write;
            let _ = write!(&mut s, "{b:02x}");
        }
        s
    }

    #[test]
    fn fresh_checkpoint_keys_skip_existing_blocks_after_restart() {
        let scratch = ScratchTestDir::new("restart-ids");
        let mut first = SpillStore::open(scratch.path(), 1024).unwrap();
        let old = first.fresh_key(SpillBlockKind::Full);
        first.write_block(old, 1, b"old KV", None, &HashSet::new()).unwrap();
        drop(first);

        let mut next = SpillStore::open(scratch.path(), 1024).unwrap();
        let fresh = next.fresh_key(SpillBlockKind::Full);
        assert_ne!(old, fresh);
        assert!(!next.contains(fresh));
        next.write_block(fresh, 1, b"new KV", None, &HashSet::new()).unwrap();
        assert!(matches!(next.read_block(old, None), ReadOutcome::Hit(b) if b.kv_bytes == b"old KV"));
        assert!(matches!(next.read_block(fresh, None), ReadOutcome::Hit(b) if b.kv_bytes == b"new KV"));
    }

    #[test]
    fn dropping_the_store_releases_ownership_even_with_a_duplicate_fd() {
        let scratch = ScratchTestDir::new("directory-lock-lifetime");
        let owner = SpillStore::open(scratch.path(), 1024).unwrap();
        // A child between fork and exec holds another descriptor for the same open file.
        let inherited = owner.directory_lock.try_clone().unwrap();
        drop(owner);
        let successor = SpillStore::open(scratch.path(), 1024).unwrap();
        drop(inherited);
        assert!(SpillStore::open(scratch.path(), 1024).is_err());
        drop(successor);
    }

    #[test]
    fn one_process_owns_the_spill_directory_until_drop() {
        const CHILD_DIR: &str = "INKY_TEST_CONTENDED_SPILL_DIR";
        if let Some(dir) = std::env::var_os(CHILD_DIR) {
            // A second opener must fail before a zero-byte budget can evict the writer's blocks.
            let result = SpillStore::open(dir, 0);
            assert!(matches!(result, Err(SpillError::InvalidDir(ref reason)) if reason.contains("already in use")));
            return;
        }

        let scratch = ScratchTestDir::new("directory-owner");
        let dir = scratch.path();
        let mut owner = SpillStore::open(dir, 1024).unwrap();
        let key = SpillKey::full(37);
        owner.write_block(key, 1, b"retained", None, &HashSet::new()).unwrap();
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "spill::tests::one_process_owns_the_spill_directory_until_drop"])
            .env(CHILD_DIR, dir)
            .output().unwrap();
        assert!(child.status.success(), "{}", String::from_utf8_lossy(&child.stdout));
        drop(owner);

        let mut successor = SpillStore::open(dir, 1024).unwrap();
        assert!(matches!(successor.read_block(key, None), ReadOutcome::Hit(block) if block.kv_bytes == b"retained"));
    }

    /// Deterministic ground-truth generator for block K/V bytes and optional recurrent state.
    fn deterministic_kv(key: SpillKey, len: usize) -> Vec<u8> {
        let mut x = key.id.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ ((key.kind as u64 + 1) << 56) ^ 0xa076_1d64_78bd_642f;
        let mut out = vec![0u8; len];
        for chunk in out.chunks_mut(8) {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            let v = x.wrapping_mul(0x2545_f491_4f6c_dd1d).to_le_bytes();
            let n = chunk.len();
            chunk.copy_from_slice(&v[..n]);
        }
        out
    }

    fn deterministic_state(key: SpillKey, len: usize) -> Vec<u8> {
        let mut x = key.id.wrapping_mul(0xd1b5_4a32_d192_ed03) ^ 0x55aa_33cc_9966_11ee;
        let mut out = vec![0u8; len];
        for chunk in out.chunks_mut(8) {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            let v = x.wrapping_mul(0x94d0_49bb_1331_11eb).to_le_bytes();
            let n = chunk.len();
            chunk.copy_from_slice(&v[..n]);
        }
        out
    }

    #[test]
    fn sha256_matches_nist_vectors() {
        let h_empty = Sha256::new().finalize();
        assert_eq!(
            hex(&h_empty),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );

        let mut h_abc = Sha256::new();
        h_abc.update(b"abc");
        assert_eq!(
            hex(&h_abc.finalize()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );

        let mut h_448 = Sha256::new();
        h_448.update(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        assert_eq!(
            hex(&h_448.finalize()),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );

        // Streaming chunked equivalence on 128 KiB payload.
        let payload = deterministic_kv(SpillKey::full(42), 131_072);
        let mut whole = Sha256::new();
        whole.update(&payload);
        let mut chunked = Sha256::new();
        for c in payload.chunks(137) {
            chunked.update(c);
        }
        assert_eq!(whole.finalize(), chunked.finalize());
    }

    #[cfg(unix)]
    #[test]
    fn spill_dir_and_files_have_strict_unix_permissions() {
        let scratch = ScratchTestDir::new("perms");
        let dir = scratch.path().join("spill");
        let mut store = SpillStore::open(&dir, 1024 * 1024).unwrap();
        let dir_mode = fs::symlink_metadata(&dir).unwrap().mode() & 0o777;
        assert_eq!(dir_mode, 0o700, "spill dir must be mode 0700");

        let key = SpillKey::full(7);
        let kv = deterministic_kv(key, 4096);
        let (outcome, _) = store.write_block(key, 1, &kv, None, &HashSet::new()).unwrap();
        assert!(matches!(outcome, WriteOutcome::Written { .. }));
        let file_mode = fs::symlink_metadata(store.block_path(key)).unwrap().mode() & 0o777;
        assert_eq!(file_mode, 0o600, "spilled .blk files must be mode 0600");
    }

    #[test]
    fn lazy_lru_spill_never_writes_eagerly_and_evicts_strict_lru_under_pressure() {
        let scratch = ScratchTestDir::new("lazy-lru");
        let dir = scratch.path().join("spill");
        let full_stride = 4096;
        let win_stride = 8192;
        let pool = HostBlockPoolBackend::new(4, 2, full_stride, win_stride);
        let store = SpillStore::open(&dir, 64 * 1024 * 1024).unwrap();
        let mut mgr = SpillManager::new(pool, store);

        // Allocate, fill, seal, and pin 4 full blocks (fills 100% of Full pool).
        let mut slots = Vec::new();
        for id in 0..4u64 {
            let key = SpillKey::full(id);
            let slot = mgr.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
            let kv = deterministic_kv(key, full_stride);
            mgr.pool_mut().write_and_seal(slot, &kv).unwrap();
            let state = if id == 2 { Some(deterministic_state(key, 1024)) } else { None };
            mgr.register_sealed_block(key, slot, state).unwrap();
            mgr.pin(key).unwrap();
            slots.push((key, slot));
        }

        // LAZY INVARIANT: 0 disk writes while filling memory pool.
        assert_eq!(mgr.store().stats().writes_committed, 0);
        assert_eq!(mgr.store().entry_count(), 0);

        // Unpin all 4 blocks in an explicit recency order:
        // oldest -> newest: id 2, id 0, id 3, id 1.
        for id in [2u64, 0, 3, 1] {
            assert_eq!(mgr.unpin(SpillKey::full(id)).unwrap(), 0);
        }
        // Re-pin id 0 so it CANNOT be evicted even though it is older than 3 and 1.
        mgr.pin(SpillKey::full(0)).unwrap();

        // LAZY INVARIANT: dropping refs to 0 still triggers ZERO disk writes!
        assert_eq!(mgr.store().stats().writes_committed, 0);
        assert_eq!(mgr.store().entry_count(), 0);
        for id in 0..4u64 {
            assert_eq!(mgr.tier(SpillKey::full(id)), BlockTier::Memory);
        }

        // Now request a 5th full block -> memory pressure!
        // Must evict id 2 (oldest unreferenced) ONLY, writing its K/V + recurrent state.
        let k4 = SpillKey::full(4);
        let s4 = mgr.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
        mgr.pool_mut().write_and_seal(s4, &deterministic_kv(k4, full_stride)).unwrap();
        mgr.register_sealed_block(k4, s4, None).unwrap();
        mgr.pin(k4).unwrap();

        assert_eq!(mgr.stats().lazy_spills_written, 1);
        assert_eq!(mgr.tier(SpillKey::full(2)), BlockTier::Disk);
        assert_eq!(mgr.tier(SpillKey::full(0)), BlockTier::Memory); // pinned!
        assert_eq!(mgr.tier(SpillKey::full(3)), BlockTier::Memory);
        assert_eq!(mgr.tier(SpillKey::full(1)), BlockTier::Memory);

        // Request a 6th full block -> must evict id 3 (NOT pinned id 0, NOT newer id 1).
        let k5 = SpillKey::full(5);
        let s5 = mgr.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
        mgr.pool_mut().write_and_seal(s5, &deterministic_kv(k5, full_stride)).unwrap();
        mgr.register_sealed_block(k5, s5, None).unwrap();
        mgr.pin(k5).unwrap();

        assert_eq!(mgr.stats().lazy_spills_written, 2);
        assert_eq!(mgr.tier(SpillKey::full(3)), BlockTier::Disk);
        assert_eq!(mgr.tier(SpillKey::full(0)), BlockTier::Memory);
        assert_eq!(mgr.tier(SpillKey::full(1)), BlockTier::Memory);

        // Restore id 2 (a start point with recurrent state) -> must evict id 1,
        // read id 2 from SSD, and recover bit-identical K/V + recurrent state!
        let prepared = mgr.prepare_working_set_restore(&[SpillKey::full(2)], &[]).unwrap();
        let restored = mgr.execute_working_set_restore(prepared).unwrap();
        assert_eq!(restored.full_slots.len(), 1);
        let (rk, rslot) = restored.full_slots[0];
        assert_eq!(rk, SpillKey::full(2));
        assert_eq!(
            mgr.pool().read_slot_bytes(rslot).unwrap(),
            deterministic_kv(SpillKey::full(2), full_stride).as_slice()
        );
        assert_eq!(
            restored.recurrent_states.get(&SpillKey::full(2)).map(Vec::as_slice),
            Some(deterministic_state(SpillKey::full(2), 1024).as_slice())
        );
        assert_eq!(mgr.tier(SpillKey::full(2)), BlockTier::Both);
        assert_eq!(mgr.tier(SpillKey::full(1)), BlockTier::Disk);

        // Unpin id 2 and trigger one more eviction: because id 2 is already `Both`
        // (clean copy on disk), evicting it again must NOT rewrite to SSD!
        mgr.unpin(SpillKey::full(2)).unwrap();
        let writes_before = mgr.store().stats().writes_committed;
        let _s6 = mgr.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
        assert_eq!(mgr.store().stats().writes_committed, writes_before);
        assert_eq!(mgr.stats().clean_mem_evictions, 1);
        assert_eq!(mgr.tier(SpillKey::full(2)), BlockTier::Disk);
    }

    #[test]
    fn failed_unlink_preserves_budget_and_gc_skips_the_stuck_victim() {
        let scratch = ScratchTestDir::new("unlink-budget");
        let charge = (SPILL_HEADER_LEN + 4) as u64;
        let mut store = SpillStore::open(scratch.path().join("spill"), 2 * charge).unwrap();
        let keys = [1, 2, 3, 4].map(SpillKey::full);
        let unprotected = HashSet::new();
        for &key in &keys[..2] {
            store.write_block(key, 1, b"data", None, &unprotected).unwrap();
        }

        // Preserve the original bytes under a directory that remove_file cannot unlink,
        // including when this test runs with elevated filesystem permissions.
        let path = store.block_path(keys[0]);
        let saved = scratch.path().join("saved");
        fs::rename(&path, &saved).unwrap();
        fs::create_dir(&path).unwrap();
        fs::rename(&saved, path.join("original")).unwrap();
        assert!(!store.remove_block(keys[0]));
        assert!(matches!(store.read_block(keys[0], None), ReadOutcome::Miss(_)));
        assert_eq!(store.stats().corrupt_files_removed, 0);
        assert!(store.contains(keys[0]));
        assert_eq!(store.used_bytes(), 2 * charge);

        let (outcome, victims) = store.write_block(keys[2], 1, b"data", None, &unprotected).unwrap();
        assert_eq!(outcome, WriteOutcome::Written { on_disk_bytes: charge });
        assert_eq!(victims, vec![keys[1]]);
        let (outcome, victims) = store.write_block(keys[3], 1, b"data", None, &HashSet::from([keys[2]])).unwrap();
        assert_eq!(outcome, WriteOutcome::OverBudget);
        assert!(victims.is_empty());
        assert_eq!(store.stats().disk_gc_evictions, 1);
        assert!(store.contains(keys[0]));
        assert_eq!(store.used_bytes(), 2 * charge);

        fs::rename(path.join("original"), &saved).unwrap();
        fs::remove_dir(&path).unwrap();
        fs::rename(&saved, &path).unwrap();
        assert!(store.remove_block(keys[0]));
        assert_eq!(store.used_bytes(), charge);
    }

    #[test]
    fn startup_reports_canonical_block_it_cannot_remove() {
        let scratch = ScratchTestDir::new("startup-unlink");
        let dir = scratch.path().join("spill");
        fs::create_dir(&dir).unwrap();
        let stuck = dir.join(SpillKey::full(1).blk_filename());
        fs::create_dir(&stuck).unwrap();
        assert!(matches!(SpillStore::open(&dir, 0), Err(SpillError::Io(_))));
        assert!(stuck.is_dir());
        fs::remove_dir(stuck).unwrap();
        let store = SpillStore::open(&dir, 0).unwrap();
        assert_eq!(store.used_bytes(), 0);
    }

    #[test]
    fn transient_open_error_retains_charge_and_absent_files_release_it_once() {
        let scratch = ScratchTestDir::new("open-accounting");
        let dir = scratch.path().join("spill");
        let charge = (SPILL_HEADER_LEN + 4) as u64;
        let mut store = SpillStore::open(&dir, 2 * charge).unwrap();
        let keys = [1, 2].map(SpillKey::full);
        for key in keys {
            store.write_block(key, 1, b"data", None, &HashSet::new()).unwrap();
        }

        let saved = scratch.path().join("saved-dir");
        fs::rename(&dir, &saved).unwrap();
        fs::write(&dir, b"temporarily inaccessible").unwrap();
        assert_eq!(store.read_block(keys[0], None), ReadOutcome::Miss(MissReason::IoReadError));
        assert_eq!(store.used_bytes(), 2 * charge);
        assert!(store.contains(keys[0]));
        fs::remove_file(&dir).unwrap();
        fs::rename(&saved, &dir).unwrap();
        assert!(matches!(store.read_block(keys[0], None), ReadOutcome::Hit(_)));

        fs::remove_file(store.block_path(keys[0])).unwrap();
        assert_eq!(store.read_block(keys[0], None), ReadOutcome::Miss(MissReason::NotFound));
        assert!(!store.remove_block(keys[0]));
        assert_eq!(store.used_bytes(), charge);
        fs::remove_file(store.block_path(keys[1])).unwrap();
        assert!(store.remove_block(keys[1]));
        assert!(!store.remove_block(keys[1]));
        assert_eq!(store.used_bytes(), 0);
    }

    #[test]
    fn disk_budget_enforces_max_spill_bytes_with_lru_gc() {
        let scratch = ScratchTestDir::new("disk-gc");
        let dir = scratch.path().join("spill");
        let kv_len = 4096usize;
        let one_file = (SPILL_HEADER_LEN + kv_len) as u64;
        // Budget holds at most 3 spilled blocks.
        let max_bytes = one_file * 3;
        let mut store = SpillStore::open(&dir, max_bytes).unwrap();

        for id in 0..8u64 {
            let key = SpillKey::full(id);
            let kv = deterministic_kv(key, kv_len);
            let (out, _evicted) = store.write_block(key, 1, &kv, None, &HashSet::new()).unwrap();
            assert_eq!(out, WriteOutcome::Written { on_disk_bytes: one_file });
            assert!(
                store.used_bytes() <= max_bytes,
                "used_bytes {} exceeded max_bytes {}",
                store.used_bytes(),
                max_bytes
            );
            // Verify actual on-disk bytes across all files in --spill-dir also respect budget.
            let actual_bytes: u64 = fs::read_dir(&dir)
                .unwrap()
                .map(|e| e.unwrap().metadata().unwrap().len())
                .sum();
            assert!(actual_bytes <= max_bytes);
        }

        assert_eq!(store.entry_count(), 3);
        assert_eq!(store.stats().disk_gc_evictions, 5);
        // Oldest 0..5 evicted; newest 5, 6, 7 survive and verify bit-identically.
        for id in 0..5u64 {
            assert!(matches!(
                store.read_block(SpillKey::full(id), Some(kv_len)),
                ReadOutcome::Miss(MissReason::NotFound)
            ));
        }
        for id in 5..8u64 {
            let key = SpillKey::full(id);
            match store.read_block(key, Some(kv_len)) {
                ReadOutcome::Hit(blk) => assert_eq!(blk.kv_bytes, deterministic_kv(key, kv_len)),
                ReadOutcome::Miss(r) => panic!("expected surviving block {id}, got miss {r:?}"),
            }
        }

        // Touch block 5 (already touched above, now touch 6 and 7 then 5 so 6 is oldest).
        let _ = store.read_block(SpillKey::full(6), Some(kv_len));
        let _ = store.read_block(SpillKey::full(7), Some(kv_len));
        let _ = store.read_block(SpillKey::full(5), Some(kv_len));
        // Writing block 8 must evict block 6 (LRU), keeping 5, 7, 8!
        let k8 = SpillKey::full(8);
        store
            .write_block(k8, 1, &deterministic_kv(k8, kv_len), None, &HashSet::new())
            .unwrap();
        assert!(!store.contains(SpillKey::full(6)));
        assert!(store.contains(SpillKey::full(5)));
        assert!(store.contains(SpillKey::full(7)));
        assert!(store.contains(SpillKey::full(8)));

        // Zero disk budget drops cleanly without failing or writing files.
        let dir0 = scratch.path().join("spill-zero");
        let pool0 = HostBlockPoolBackend::new(1, 1, 512, 512);
        let store0 = SpillStore::open(&dir0, 0).unwrap();
        let mut mgr0 = SpillManager::new(pool0, store0);
        let s0 = mgr0.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
        mgr0.pool_mut().write_and_seal(s0, &deterministic_kv(SpillKey::full(1), 512)).unwrap();
        mgr0.register_sealed_block(SpillKey::full(1), s0, None).unwrap();
        // refs == 0 -> allocating another block drops block 1 cleanly with 0 disk usage.
        let _s1 = mgr0.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
        assert_eq!(mgr0.store().used_bytes(), 0);
        assert_eq!(mgr0.tier(SpillKey::full(1)), BlockTier::None);
        assert_eq!(mgr0.stats().over_budget_mem_drops, 1);
    }

    #[test]
    fn atomic_working_set_reservation_prevents_partial_restore_thrashing() {
        let scratch = ScratchTestDir::new("atomic-reserve");
        let dir = scratch.path().join("spill");
        let f_stride = 2048;
        let w_stride = 4096;
        // Pool holds 4 Full slots and 3 Window slots.
        let pool = HostBlockPoolBackend::new(4, 3, f_stride, w_stride);
        let mut store = SpillStore::open(&dir, 64 * 1024 * 1024).unwrap();

        // Pre-populate 6 Full blocks and 4 Window blocks on SSD (simulating spilled prefixes).
        for id in 1..=6u64 {
            let k = SpillKey::full(id);
            store.write_block(k, 1, &deterministic_kv(k, f_stride), None, &HashSet::new()).unwrap();
        }
        for id in 1..=4u64 {
            let k = SpillKey::window(id);
            store.write_block(k, 1, &deterministic_kv(k, w_stride), None, &HashSet::new()).unwrap();
        }

        let mut mgr = SpillManager::new(pool, store);
        mgr.index_existing_disk_blocks();

        // Put one cold Full block in memory (refs == 0) so we verify asymmetric WouldBlock
        // triggers ZERO spills when Window slots cannot fit!
        let cold_f = SpillKey::full(100);
        let sf = mgr.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
        mgr.pool_mut().write_and_seal(sf, &deterministic_kv(cold_f, f_stride)).unwrap();
        mgr.register_sealed_block(cold_f, sf, None).unwrap();

        // Request exceeding capacity returns ExceedsCapacity with zero mutations.
        assert!(matches!(
            mgr.reserve_working_set(5, 1),
            Err(ReserveError::ExceedsCapacity { .. })
        ));
        assert_eq!(mgr.tier(cold_f), BlockTier::Memory);

        // Lane A reserves (3 full, 2 window) atomically -> succeeds!
        let lane_a_full = [SpillKey::full(1), SpillKey::full(2), SpillKey::full(3)];
        let lane_a_win = [SpillKey::window(1), SpillKey::window(2)];
        let prep_a = mgr.prepare_working_set_restore(&lane_a_full, &lane_a_win).unwrap();
        assert_eq!(prep_a.reservation.full_count(), 3);
        assert_eq!(prep_a.reservation.window_count(), 2);
        // 3 free Full slots were enough without even spilling cold_f!
        assert_eq!(mgr.tier(cold_f), BlockTier::Memory);
        assert_eq!(mgr.stats().lazy_spills_written, 0);

        // Now Lane B wants (3 full, 2 window) while Lane A is in Restoring!
        // Only 1 Full (cold_f) and 1 Window remain -> MUST fail atomically with WouldBlock,
        // spilling ZERO blocks (cold_f stays untouched in Memory!) and NEVER evicting Lane A!
        let lane_b_full = [SpillKey::full(4), SpillKey::full(5), SpillKey::full(6)];
        let lane_b_win = [SpillKey::window(3), SpillKey::window(4)];
        let err_b = mgr.prepare_working_set_restore(&lane_b_full, &lane_b_win).unwrap_err();
        assert!(matches!(
            err_b,
            PrepareRestoreError::Reserve(ReserveError::WouldBlock {
                need_full: 3,
                avail_full: 1,
                need_window: 2,
                avail_window: 1,
            })
        ));
        assert_eq!(mgr.tier(cold_f), BlockTier::Memory, "WouldBlock must not spill cold_f");
        assert_eq!(mgr.stats().lazy_spills_written, 0);
        assert_eq!(mgr.stats().reservations_would_block, 1);

        // Asymmetric WouldBlock: Lane C wants (1 full, 2 window). Full fits (1 <= 1),
        // Window does NOT (2 > 1). MUST NOT spill cold_f!
        let err_c = mgr.reserve_working_set(1, 2).unwrap_err();
        assert!(matches!(err_c, ReserveError::WouldBlock { .. }));
        assert_eq!(mgr.tier(cold_f), BlockTier::Memory, "asymmetric WouldBlock must not spill Full");

        // Lane A completes its SSD reads with zero interference.
        let res_a = mgr.execute_working_set_restore(prep_a).unwrap();
        assert_eq!(res_a.full_slots.len(), 3);
        assert_eq!(res_a.window_slots.len(), 2);
        for &(k, s) in &res_a.full_slots {
            assert_eq!(mgr.pool().read_slot_bytes(s).unwrap(), deterministic_kv(k, f_stride).as_slice());
        }
        for &(k, s) in &res_a.window_slots {
            assert_eq!(mgr.pool().read_slot_bytes(s).unwrap(), deterministic_kv(k, w_stride).as_slice());
        }

        // Lane A finishes and unpins its blocks -> they stay hot in memory (`Both`, refs=0).
        for &(k, _) in res_a.full_slots.iter().chain(res_a.window_slots.iter()) {
            assert_eq!(mgr.unpin(k).unwrap(), 0);
        }

        // Now Lane B retries: its full working set fits via lazy eviction of cold blocks!
        let prep_b = mgr.prepare_working_set_restore(&lane_b_full, &lane_b_win).unwrap();
        let res_b = mgr.execute_working_set_restore(prep_b).unwrap();
        for &(k, s) in &res_b.full_slots {
            assert_eq!(mgr.pool().read_slot_bytes(s).unwrap(), deterministic_kv(k, f_stride).as_slice());
        }
        for &(k, s) in &res_b.window_slots {
            assert_eq!(mgr.pool().read_slot_bytes(s).unwrap(), deterministic_kv(k, w_stride).as_slice());
        }
    }

    #[test]
    fn restore_never_evicts_its_own_in_memory_working_set() {
        let scratch = ScratchTestDir::new("self-pin");
        let dir = scratch.path().join("spill");
        let stride = 1024;
        // Pool has capacity 2 Full slots.
        let pool = HostBlockPoolBackend::new(2, 1, stride, stride);
        let mut store = SpillStore::open(&dir, 1024 * 1024).unwrap();
        // Two blocks on disk: full(2), full(3).
        for id in [2u64, 3] {
            let k = SpillKey::full(id);
            store.write_block(k, 1, &deterministic_kv(k, stride), None, &HashSet::new()).unwrap();
        }
        let mut mgr = SpillManager::new(pool, store);
        mgr.index_existing_disk_blocks();

        // full(1) is in memory with refs == 0, and full(99) is in memory with refs == 1 (pinned by another lane).
        let k1 = SpillKey::full(1);
        let s1 = mgr.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
        mgr.pool_mut().write_and_seal(s1, &deterministic_kv(k1, stride)).unwrap();
        mgr.register_sealed_block(k1, s1, None).unwrap();

        let k99 = SpillKey::full(99);
        let s99 = mgr.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
        mgr.pool_mut().write_and_seal(s99, &deterministic_kv(k99, stride)).unwrap();
        mgr.register_sealed_block(k99, s99, None).unwrap();
        mgr.pin(k99).unwrap();

        // A lane wants working set [full(1) (in memory, refs=0), full(2) (on disk)].
        // Total slots needed = 2, but full(99) holds 1 slot -> MUST WouldBlock and MUST NOT
        // spill full(1) to make room for full(2)!
        let err = mgr.prepare_working_set_restore(&[k1, SpillKey::full(2)], &[]).unwrap_err();
        assert!(matches!(err, PrepareRestoreError::Reserve(ReserveError::WouldBlock { .. })));
        assert_eq!(mgr.tier(k1), BlockTier::Memory);
        assert_eq!(mgr.pool().refs(s1).unwrap(), 0, "failed prepare must unpin full(1)");

        // Once full(99) unpins, [full(1), full(2)] succeeds by spilling full(99), keeping full(1) resident!
        mgr.unpin(k99).unwrap();
        let prep = mgr.prepare_working_set_restore(&[k1, SpillKey::full(2)], &[]).unwrap();
        let res = mgr.execute_working_set_restore(prep).unwrap();
        assert_eq!(res.full_slots.len(), 2);
        assert_eq!(mgr.tier(k1), BlockTier::Memory);
        assert_eq!(mgr.tier(SpillKey::full(2)), BlockTier::Both);
        assert_eq!(mgr.tier(k99), BlockTier::Disk);
    }

    #[test]
    fn publication_preserves_an_unindexed_file_at_the_destination() {
        let scratch = ScratchTestDir::new("unindexed-destination");
        let mut store = SpillStore::open(scratch.path(), 4096).unwrap();
        let key = SpillKey::full(0xab);
        let destination = store.block_path(key);
        fs::write(&destination, b"keep").unwrap();
        let result = store.write_block(key, 1, b"new KV", None, &HashSet::new());
        assert!(matches!(result, Err(SpillError::Io(err)) if err.kind() == io::ErrorKind::AlreadyExists));
        assert_eq!(fs::read(&destination).unwrap(), b"keep");
        assert_eq!(store.entry_count(), 0);
    }

    #[test]
    fn spill_file_names_require_canonical_lowercase_hex_fields() {
        for key in [SpillKey::full(0), SpillKey::window(u64::MAX)] {
            assert_eq!(SpillKey::parse_blk_filename(&key.blk_filename()), Some(key));
            assert_eq!(SpillKey::parse_tmp_filename(&key.tmp_filename(u32::MAX, u64::MAX)), Some(key));
        }
        for name in ["blk-f-00000000000000AB.blk", "blk-f-+000000000000001.blk"] {
            assert_eq!(SpillKey::parse_blk_filename(name), None, "{name}");
        }
        for name in [
            "blk-f-0000000000000001.tmp.0000000A.0000000000000001",
            "blk-f-0000000000000001.tmp.00000001.000000000000000A",
        ] {
            assert_eq!(SpillKey::parse_tmp_filename(name), None, "{name}");
        }
    }

    #[test]
    fn corrupt_torn_bitflip_zero_and_orphan_files_cause_clean_cache_miss_never_corruption() {
        let scratch = ScratchTestDir::new("corruption");
        let dir = scratch.path().join("spill");
        let kv_len = 2048usize;
        let state_len = 512usize;

        // Only a spill writer's exact temporary names are eligible for cleanup.
        prepare_spill_dir(&dir).unwrap();
        let orphan1 = dir.join(SpillKey::full(1).tmp_filename(0x1234, 1));
        let orphan2 = dir.join(SpillKey::window(0x99).tmp_filename(0x5678, 2));
        fs::write(&orphan1, b"half-written-tmp-data").unwrap();
        fs::write(&orphan2, b"").unwrap();
        let unrelated = [
            "notes.tmp.backup",
            "blk-x-0000000000000001.tmp.00001234.0000000000000001",
            "blk-f-00000000000000AB.tmp.00001234.0000000000000001",
            "blk-f-0000000000000001.tmp.1234.0000000000000001",
            "blk-f-0000000000000001.tmp.00001234.000000000000000g",
            "blk-f-0000000000000001.tmp.00001234.0000000000000001.bak",
        ];
        for name in unrelated {
            fs::write(dir.join(name), b"keep").unwrap();
        }

        let mut store = SpillStore::open(&dir, 64 * 1024 * 1024).unwrap();
        assert_eq!(store.stats().orphans_swept, 2);
        assert!(!orphan1.exists());
        assert!(!orphan2.exists());
        for name in unrelated {
            assert_eq!(fs::read(dir.join(name)).unwrap(), b"keep", "{name}");
        }

        let write_good = |s: &mut SpillStore, id: u64| -> (SpillKey, PathBuf) {
            let key = SpillKey::full(id);
            let kv = deterministic_kv(key, kv_len);
            let st = deterministic_state(key, state_len);
            s.write_block(key, 3, &kv, Some(&st), &HashSet::new()).unwrap();
            let p = s.block_path(key);
            (key, p)
        };

        // 1. Zero-byte .blk file.
        let (k1, p1) = write_good(&mut store, 1);
        fs::write(&p1, b"").unwrap();
        assert_eq!(store.read_block(k1, Some(kv_len)), ReadOutcome::Miss(MissReason::TruncatedHeader));
        assert!(!p1.exists(), "corrupt file should be unlinked");

        // 2. Truncated header (40 bytes).
        let (k2, p2) = write_good(&mut store, 2);
        let raw = fs::read(&p2).unwrap();
        fs::write(&p2, &raw[..40]).unwrap();
        assert_eq!(store.read_block(k2, Some(kv_len)), ReadOutcome::Miss(MissReason::TruncatedHeader));

        // 3. Torn K/V payload (1 byte short & half-written).
        let (k3, p3) = write_good(&mut store, 3);
        let raw = fs::read(&p3).unwrap();
        fs::write(&p3, &raw[..SPILL_HEADER_LEN + kv_len / 2]).unwrap();
        assert_eq!(store.read_block(k3, Some(kv_len)), ReadOutcome::Miss(MissReason::SizeMismatch));

        // 4. Torn recurrent state (1 byte short).
        let (k4, p4) = write_good(&mut store, 4);
        let raw = fs::read(&p4).unwrap();
        fs::write(&p4, &raw[..raw.len() - 1]).unwrap();
        assert_eq!(store.read_block(k4, Some(kv_len)), ReadOutcome::Miss(MissReason::SizeMismatch));

        // 5. Trailing garbage appended.
        let (k5, p5) = write_good(&mut store, 5);
        let mut raw = fs::read(&p5).unwrap();
        raw.push(0xAA);
        fs::write(&p5, &raw).unwrap();
        assert_eq!(store.read_block(k5, Some(kv_len)), ReadOutcome::Miss(MissReason::SizeMismatch));

        // 6. Single-bit flips across header, digest, K/V payload, and recurrent state.
        let flip_offsets = [
            (0usize, MissReason::BadMagic),
            (8, MissReason::UnsupportedVersion),
            (12, MissReason::BadBlockKind),
            (14, MissReason::KvbMismatch),
            (16, MissReason::MetadataMismatch),
            (24, MissReason::ChecksumMismatch), // generation bit-flip
            (48, MissReason::ChecksumMismatch), // sha256 byte flip
            (SPILL_HEADER_LEN, MissReason::ChecksumMismatch), // first K/V byte
            (SPILL_HEADER_LEN + kv_len / 2, MissReason::ChecksumMismatch), // mid K/V byte
            (SPILL_HEADER_LEN + kv_len - 1, MissReason::ChecksumMismatch), // last K/V byte
            (SPILL_HEADER_LEN + kv_len + state_len / 2, MissReason::ChecksumMismatch), // recurrent state byte
        ];
        for (i, &(off, expected_reason)) in flip_offsets.iter().enumerate() {
            let (k, p) = write_good(&mut store, 100 + i as u64);
            let mut raw = fs::read(&p).unwrap();
            raw[off] ^= 0x04;
            fs::write(&p, &raw).unwrap();
            assert_eq!(
                store.read_block(k, Some(kv_len)),
                ReadOutcome::Miss(expected_reason),
                "bit flip at offset {off} must return {expected_reason:?}"
            );
            assert!(!p.exists(), "corrupt file at offset {off} must be unlinked");
        }

        // 7. File-swap attack: copy valid block 200 over block 201.
        let (k200, p200) = write_good(&mut store, 200);
        let (k201, p201) = write_good(&mut store, 201);
        fs::copy(&p200, &p201).unwrap();
        assert_eq!(store.read_block(k201, Some(kv_len)), ReadOutcome::Miss(MissReason::MetadataMismatch));
        // Original block 200 remains intact.
        match store.read_block(k200, Some(kv_len)) {
            ReadOutcome::Hit(blk) => {
                assert_eq!(blk.kv_bytes, deterministic_kv(k200, kv_len));
                assert_eq!(blk.recurrent_state, Some(deterministic_state(k200, state_len)));
            }
            ReadOutcome::Miss(r) => panic!("block 200 should be intact, got {r:?}"),
        }

        // 8. End-to-end SpillManager rollback on corrupt block in multi-block working set:
        //    pool slots must remain completely clean (zero-filled, zero leaks)!
        let pool = HostBlockPoolBackend::new(3, 1, kv_len, kv_len);
        drop(store);
        let store2 = SpillStore::open(&dir, 64 * 1024 * 1024).unwrap();
        let mut mgr = SpillManager::new(pool, store2);
        // Write two blocks via store and corrupt the second.
        let ka = SpillKey::full(300);
        let kb = SpillKey::full(301);
        mgr.store.write_block(ka, 1, &deterministic_kv(ka, kv_len), None, &HashSet::new()).unwrap();
        mgr.store.write_block(kb, 1, &deterministic_kv(kb, kv_len), None, &HashSet::new()).unwrap();
        let pb = mgr.store.block_path(kb);
        let mut raw_b = fs::read(&pb).unwrap();
        raw_b[SPILL_HEADER_LEN + 7] ^= 0x80;
        fs::write(&pb, &raw_b).unwrap();
        mgr.index_existing_disk_blocks();

        let prep = mgr.prepare_working_set_restore(&[ka, kb], &[]).unwrap();
        let miss = mgr.execute_working_set_restore(prep).unwrap_err();
        assert_eq!(miss.failed_key, kb);
        assert_eq!(miss.reason, MissReason::ChecksumMismatch);
        // Zero leaked slots, zero active reservations, and every pool slot is zeroed!
        assert_eq!(mgr.active_reservation_count(), 0);
        assert_eq!(mgr.pool().used_slots(SpillBlockKind::Full), 0);
        assert_eq!(mgr.pool().free_slots(SpillBlockKind::Full), 3);
        for slot_i in 1..=3u32 {
            assert!(
                mgr.pool().raw_slot_bytes(SpillBlockKind::Full, slot_i).iter().all(|&b| b == 0),
                "slot {slot_i} must remain zero-filled after corrupt-block rollback"
            );
        }
    }

    /// Child helper mode spawned by [`kill9_mid_write_loses_cache_never_correctness`].
    /// Marked `#[ignore]` so normal `cargo test` runs do not invoke it directly;
    /// returns immediately if `INKY_SPILL_KILL9_DIR` is unset.
    #[test]
    #[ignore = "internal child helper spawned by kill9_mid_write_loses_cache_never_correctness"]
    fn kill9_writer_helper() {
        let Ok(dir) = std::env::var("INKY_SPILL_KILL9_DIR") else {
            return;
        };
        let round: u64 = std::env::var("INKY_SPILL_KILL9_ROUND")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let kv_len: usize = std::env::var("INKY_SPILL_KILL9_KV_LEN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256 * 1024);
        let state_len: usize = std::env::var("INKY_SPILL_KILL9_STATE_LEN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32 * 1024);

        let mut store = SpillStore::open(&dir, 512 * 1024 * 1024).expect("child open spill store");
        let empty = HashSet::new();

        // Write continuously until parent sends SIGKILL.
        let mut seq = 0u64;
        loop {
            let id = (round << 32) | seq;
            let kind = if seq % 4 == 0 {
                SpillBlockKind::Window
            } else {
                SpillBlockKind::Full
            };
            let key = SpillKey::new(kind, id);
            let kv = deterministic_kv(key, kv_len);
            let st = if seq % 3 == 0 {
                Some(deterministic_state(key, state_len))
            } else {
                None
            };
            let _ = store.write_block(key, seq + 1, &kv, st.as_deref(), &empty);
            if seq == 0 {
                println!("READY");
                let _ = io::stdout().flush();
            } else {
                println!("COMMIT {seq}");
                let _ = io::stdout().flush();
            }
            seq = seq.wrapping_add(1);
            if seq >= 500 {
                break;
            }
        }
    }

    /// Layer 4 Gate (`docs/ROADMAP.md:140`):
    /// *"Gate: kill -9 in mid-write loses cache and never correctness"*
    ///
    /// Spawns real writer subprocesses that continuously spill blocks to SSD while
    /// the parent delivers `SIGKILL` mid-write (both targeted on observed `.tmp.*`
    /// files and at staggered intervals), then verifies after every crash:
    /// - 100% of orphan `.tmp.*` files are swept on recovery;
    /// - 100% of surviving `.blk` files either verify SHA-256 and match ground-truth
    ///   K/V + recurrent state bit-for-bit, or return `CacheMiss` and fall back to
    ///   ground-truth recomputation with bit-identical outputs.
    #[cfg(unix)]
    #[test]
    fn kill9_mid_write_loses_cache_never_correctness() {
        use std::os::unix::process::ExitStatusExt;

        let scratch = ScratchTestDir::new("kill9-gate");
        let exe = std::env::current_exe().expect("current test binary path");
        let kv_len = 256 * 1024usize; // 256 KiB per block so writes take real SSD I/O time
        let state_len = 32 * 1024usize;

        let mut total_hits = 0u64;
        let mut total_misses = 0u64;
        let mut total_orphans_swept = 0u64;
        let mut total_kills_verified = 0u64;

        let rounds = 10u64;
        for round in 0..rounds {
            let dir = scratch.path().join(format!("round-{round}"));
            prepare_spill_dir(&dir).unwrap();

            let mut child = Command::new(&exe)
                .args([
                    "--exact",
                    "spill::tests::kill9_writer_helper",
                    "--ignored",
                    "--nocapture",
                ])
                .env("INKY_SPILL_KILL9_DIR", &dir)
                .env("INKY_SPILL_KILL9_ROUND", round.to_string())
                .env("INKY_SPILL_KILL9_KV_LEN", kv_len.to_string())
                .env("INKY_SPILL_KILL9_STATE_LEN", state_len.to_string())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .expect("spawn kill9 writer helper");

            let child_pid = child.id() as libc::pid_t;
            let stdout = child.stdout.take().expect("child stdout");

            // Wait for child to finish block 0 and enter continuous write loop.
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            let mut saw_ready = false;
            let deadline = Instant::now() + Duration::from_secs(10);
            while Instant::now() < deadline {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if line.trim() == "READY" {
                            saw_ready = true;
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            assert!(saw_ready, "round {round}: child writer did not report READY");

            if round % 2 == 0 {
                // Active mid-write hunt: poll spill dir until a `.tmp.*` file is in flight,
                // then immediately fire SIGKILL right in mid-write!
                let hunt_deadline = Instant::now() + Duration::from_millis(250);
                while Instant::now() < hunt_deadline {
                    let has_tmp = fs::read_dir(&dir)
                        .ok()
                        .map(|rd| {
                            rd.filter_map(Result::ok)
                                .any(|e| e.file_name().to_str().and_then(SpillKey::parse_tmp_filename).is_some())
                        })
                        .unwrap_or(false);
                    if has_tmp {
                        break;
                    }
                    std::thread::yield_now();
                }
            } else {
                // Staggered asynchronous kill during active write loop.
                std::thread::sleep(Duration::from_millis(2 + (round * 3) % 17));
            }

            // Deliver genuine SIGKILL (`kill -9`)!
            let rc = unsafe { libc::kill(child_pid, libc::SIGKILL) };
            assert_eq!(rc, 0, "round {round}: SIGKILL failed");
            let status = child.wait().expect("wait for killed child");
            assert_eq!(
                status.signal(),
                Some(libc::SIGKILL),
                "round {round}: child should terminate from SIGKILL, got {status:?}"
            );
            total_kills_verified += 1;

            // Recover with a fresh SpillStore::open().
            let mut store = SpillStore::open(&dir, 512 * 1024 * 1024).expect("recover spill store");
            total_orphans_swept += store.stats().orphans_swept;

            // Assert ZERO orphan `.tmp.*` files remain after startup sweep.
            let remaining_tmps: Vec<_> = fs::read_dir(&dir)
                .unwrap()
                .filter_map(Result::ok)
                .filter(|e| e.file_name().to_str().and_then(SpillKey::parse_tmp_filename).is_some())
                .collect();
            assert!(
                remaining_tmps.is_empty(),
                "round {round}: orphan tmp files remained after open(): {remaining_tmps:?}"
            );

            // Check every block ID in the range 0..max_seen+8 (including in-flight/unwritten tail).
            let max_id = store
                .entries
                .keys()
                .map(|k| k.id & 0xffff_ffff)
                .max()
                .unwrap_or(0);
            let check_limit = (max_id + 8).min(500);

            for seq in 0..=check_limit {
                let id = (round << 32) | seq;
                let kind = if seq % 4 == 0 {
                    SpillBlockKind::Window
                } else {
                    SpillBlockKind::Full
                };
                let key = SpillKey::new(kind, id);
                let expected_kv = deterministic_kv(key, kv_len);
                let expected_state = if seq % 3 == 0 {
                    Some(deterministic_state(key, state_len))
                } else {
                    None
                };

                let (recovered_kv, recovered_state) = match store.read_block(key, Some(kv_len)) {
                    ReadOutcome::Hit(blk) => {
                        total_hits += 1;
                        (blk.kv_bytes, blk.recurrent_state)
                    }
                    ReadOutcome::Miss(_) => {
                        total_misses += 1;
                        // Clean fallback to recomputation from ground truth.
                        (deterministic_kv(key, kv_len), expected_state.clone())
                    }
                };

                assert_eq!(
                    recovered_kv, expected_kv,
                    "round {round} seq {seq}: K/V bytes must be bit-identical"
                );
                assert_eq!(
                    recovered_state, expected_state,
                    "round {round} seq {seq}: recurrent state must be bit-identical"
                );
            }
        }

        assert_eq!(total_kills_verified, rounds);
        assert!(total_hits >= rounds, "expected committed hits across SIGKILL rounds, got {total_hits}");
        assert!(total_misses > 0, "expected cache misses on interrupted tail, got {total_misses}");
        // Print concise receipt when run with --nocapture.
        eprintln!(
            "[kill9-gate] rounds={rounds} kills={total_kills_verified} hits={total_hits} misses={total_misses} orphans_swept={total_orphans_swept}"
        );
    }

    #[test]
    fn pending_restore_blocks_are_protected_from_disk_gc() {
        let scratch = ScratchTestDir::new("io-pin-gc");
        let dir = scratch.path().join("spill");
        let stride = 2048usize;
        let one_file = (SPILL_HEADER_LEN + stride) as u64;
        // Disk holds at most 2 blocks.
        let mut store = SpillStore::open(&dir, one_file * 2).unwrap();
        let k1 = SpillKey::full(1);
        let k2 = SpillKey::full(2);
        store.write_block(k1, 1, &deterministic_kv(k1, stride), None, &HashSet::new()).unwrap();
        store.write_block(k2, 1, &deterministic_kv(k2, stride), None, &HashSet::new()).unwrap();

        // Pool holds 2 Full slots.
        let pool = HostBlockPoolBackend::new(2, 1, stride, stride);
        let mut mgr = SpillManager::new(pool, store);
        mgr.index_existing_disk_blocks();

        // Put a cold block k3 in memory (refs == 0), leaving 1 free slot.
        let k3 = SpillKey::full(3);
        let s3 = mgr.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
        mgr.pool_mut().write_and_seal(s3, &deterministic_kv(k3, stride)).unwrap();
        mgr.register_sealed_block(k3, s3, None).unwrap();

        // Lane A prepares to restore k1 (the OLDEST on-disk block!).
        // k1 now has io_pins == 1 and 1 free slot is reserved.
        let prep_a = mgr.prepare_working_set_restore(&[k1], &[]).unwrap();

        // Now another allocation triggers lazy spill of k3 to disk!
        // Disk budget is 2 blocks: k1 is older than k2, BUT k1 is io-pinned by prep_a,
        // so disk GC MUST evict k2 and MUST preserve k1 on disk!
        let _s4 = mgr.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
        assert!(mgr.store().contains(k1), "io-pinned block k1 must survive disk GC");
        assert!(!mgr.store().contains(k2), "unpinned block k2 should be evicted by disk GC");
        assert!(mgr.store().contains(k3), "spilled block k3 should be on disk");

        // Lane A executes its restore of k1 cleanly with zero miss!
        let res_a = mgr.execute_working_set_restore(prep_a).unwrap();
        assert_eq!(res_a.full_slots.len(), 1);
        assert_eq!(
            mgr.pool().read_slot_bytes(res_a.full_slots[0].1).unwrap(),
            deterministic_kv(k1, stride).as_slice()
        );
    }

    #[test]
    fn concurrent_restores_of_shared_disk_block_deduplicate_without_slot_leak() {
        let scratch = ScratchTestDir::new("dedup-restore");
        let dir = scratch.path().join("spill");
        let stride = 2048usize;
        let mut store = SpillStore::open(&dir, 64 * 1024 * 1024).unwrap();
        let shared = SpillKey::full(77);
        let state = deterministic_state(shared, 512);
        store
            .write_block(shared, 1, &deterministic_kv(shared, stride), Some(&state), &HashSet::new())
            .unwrap();

        // Pool has 2 Full slots so two lanes can both reserve concurrently.
        let pool = HostBlockPoolBackend::new(2, 1, stride, stride);
        let mut mgr = SpillManager::new(pool, store);
        mgr.index_existing_disk_blocks();

        let prep_a = mgr.prepare_working_set_restore(&[shared], &[]).unwrap();
        let prep_b = mgr.prepare_working_set_restore(&[shared], &[]).unwrap();
        assert_eq!(mgr.pool().used_slots(SpillBlockKind::Full), 2);

        // Lane A finishes restore first -> installs `shared` into slot A (refs = 1).
        let res_a = mgr.execute_working_set_restore(prep_a).unwrap();
        let slot_a = res_a.full_slots[0].1;

        // Lane B finishes restore second -> MUST deduplicate onto slot A (refs = 2)
        // and release its redundant pre-reserved slot so used_slots drops back to 1!
        let res_b = mgr.execute_working_set_restore(prep_b).unwrap();
        let slot_b = res_b.full_slots[0].1;
        assert_eq!(slot_a, slot_b, "concurrent restores of same block must share one slot");
        assert_eq!(mgr.pool().refs(slot_a).unwrap(), 2);
        assert_eq!(mgr.pool().used_slots(SpillBlockKind::Full), 1);
        assert_eq!(mgr.pool().free_slots(SpillBlockKind::Full), 1);
        assert_eq!(res_b.recurrent_states.get(&shared), Some(&state));

        // Unpin both lanes cleanly.
        assert_eq!(mgr.unpin(shared).unwrap(), 1);
        assert_eq!(mgr.unpin(shared).unwrap(), 0);
    }

    #[test]
    fn full_scale_block_and_recurrent_state_roundtrip() {
        let scratch = ScratchTestDir::new("full-scale");
        let dir = scratch.path().join("spill");
        // Real Layer 3 Full block (2 MiB) + Inkling 2 Mini recurrent state (~3.66 MiB).
        let cs_bytes = crate::engine::Engine::cs_image_bytes();
        let mut store = SpillStore::open(&dir, 32 * 1024 * 1024).unwrap();
        let key = SpillKey::full(4242);
        let kv = deterministic_kv(key, FULL_STRIDE);
        let st = deterministic_state(key, cs_bytes);

        let (out, evicted) = store
            .write_block(key, 9, &kv, Some(&st), &HashSet::new())
            .unwrap();
        assert!(evicted.is_empty());
        assert_eq!(
            out,
            WriteOutcome::Written {
                on_disk_bytes: (SPILL_HEADER_LEN + FULL_STRIDE + cs_bytes) as u64
            }
        );

        match store.read_block(key, Some(FULL_STRIDE)) {
            ReadOutcome::Hit(blk) => {
                assert_eq!(blk.key, key);
                assert_eq!(blk.generation, 9);
                assert_eq!(blk.kv_bytes, kv);
                assert_eq!(blk.recurrent_state.as_deref(), Some(st.as_slice()));
            }
            ReadOutcome::Miss(r) => panic!("expected full-scale hit, got {r:?}"),
        }
    }

    #[test]
    fn kvb_mismatch_between_kv8_and_fp16_is_rejected_without_exposing_bytes() {
        let scratch = ScratchTestDir::new("kvb-mismatch");
        let dir = scratch.path().join("spill");

        // Reject invalid kvb widths at open time.
        assert!(matches!(SpillStore::open_with_kvb(&dir, 64 * 1024 * 1024, 0), Err(SpillError::InvalidKvb(0))));
        assert!(matches!(SpillStore::open_with_kvb(&dir, 64 * 1024 * 1024, 3), Err(SpillError::InvalidKvb(3))));

        let kv_len = 4096usize;
        let cs_len = 256usize;
        let k_kv8 = SpillKey::full(77);
        let k_cs_kv8 = SpillKey::full(78);

        // 1. Write K/V and recurrent-state-only blocks from a --kv8 (kvb=1) server.
        {
            let mut kv8_store = SpillStore::open_with_kvb(&dir, 64 * 1024 * 1024, 1).unwrap();
            assert_eq!(kv8_store.kvb(), 1);
            kv8_store
                .write_block(k_kv8, 1, &deterministic_kv(k_kv8, kv_len), None, &HashSet::new())
                .unwrap();
            kv8_store
                .write_block(k_cs_kv8, 1, &[], Some(&deterministic_state(k_cs_kv8, cs_len)), &HashSet::new())
                .unwrap();
            assert!(matches!(kv8_store.read_block(k_kv8, Some(kv_len)), ReadOutcome::Hit(_)));
            assert!(matches!(kv8_store.read_block(k_cs_kv8, Some(0)), ReadOutcome::Hit(_)));
        }

        // 2. Reopen as fp16 (kvb=2): must reject BOTH K/V and state-only blocks even if expected_kv_len is None/0!
        {
            let mut fp16_store = SpillStore::open_with_kvb(&dir, 64 * 1024 * 1024, 2).unwrap();
            assert_eq!(fp16_store.kvb(), 2);
            assert_eq!(fp16_store.read_block(k_kv8, None), ReadOutcome::Miss(MissReason::KvbMismatch));
            assert!(!fp16_store.block_path(k_kv8).exists());
            assert_eq!(fp16_store.read_block(k_cs_kv8, Some(0)), ReadOutcome::Miss(MissReason::KvbMismatch));
            assert!(!fp16_store.block_path(k_cs_kv8).exists());

            // Write an fp16 block (kvb=2) now.
            let k_fp16 = SpillKey::window(88);
            fp16_store
                .write_block(k_fp16, 2, &deterministic_kv(k_fp16, kv_len), None, &HashSet::new())
                .unwrap();
        }

        // 3. Reopen as --kv8 (kvb=1): must reject the fp16 block!
        let mut kv8_store = SpillStore::open_with_kvb(&dir, 64 * 1024 * 1024, 1).unwrap();
        let k_fp16 = SpillKey::window(88);
        assert_eq!(kv8_store.read_block(k_fp16, Some(kv_len)), ReadOutcome::Miss(MissReason::KvbMismatch));
        assert!(!kv8_store.block_path(k_fp16).exists());
    }
}
