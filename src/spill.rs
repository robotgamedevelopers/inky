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
#[cfg(all(unix, standalone_blocks))]
use crate::libc;

mod store;
pub use store::{read_block_from_dir, RawDiskRead, SpillStore};

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
/// Persistent start-point metadata manifest stored inside `--spill-dir`.
pub const SPILL_INDEX_FILENAME: &str = "spill-index.bin";
const SPILL_INDEX_TMP_PREFIX: &str = "spill-index.tmp.";

static TEMP_NONCE: AtomicU64 = AtomicU64::new(1);

#[cfg(test)]
type SpillReadHook = dyn Fn(&Path, SpillKey) + Send + Sync + 'static;

#[cfg(test)]
fn spill_read_hook_slot() -> &'static std::sync::Mutex<Option<std::sync::Arc<SpillReadHook>>> {
    static HOOK: std::sync::OnceLock<std::sync::Mutex<Option<std::sync::Arc<SpillReadHook>>>> =
        std::sync::OnceLock::new();
    HOOK.get_or_init(|| std::sync::Mutex::new(None))
}

#[cfg(test)]
pub fn set_test_spill_read_hook(hook: Option<std::sync::Arc<SpillReadHook>>) {
    *spill_read_hook_slot().lock().unwrap_or_else(|e| e.into_inner()) = hook;
}

#[cfg(test)]
pub(super) fn run_test_spill_read_hook(dir: &Path, key: SpillKey) {
    let hook = spill_read_hook_slot()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    if let Some(h) = hook {
        h(dir, key);
    }
}

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
        let keys: Vec<SpillKey> = self.store.keys().collect();
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
            if let Err(err) = self.pool.release_cold_slot(slot) {
                if let Some(b) = self.blocks.get_mut(&key) {
                    b.io_pins = b.io_pins.saturating_sub(1);
                }
                return Err(err);
            }
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
        let kv_copy = match self.pool.read_slot_bytes(slot) {
            Ok(bytes) => bytes.to_vec(),
            Err(err) => {
                if let Some(b) = self.blocks.get_mut(&key) {
                    b.io_pins = b.io_pins.saturating_sub(1);
                }
                return Err(err);
            }
        };
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
                if let Err(err) = self.pool.release_cold_slot(slot) {
                    if let Some(b) = self.blocks.get_mut(&key) {
                        b.io_pins = b.io_pins.saturating_sub(1);
                    }
                    return Err(err);
                }
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
        if let Err(err) = self.pool.release_cold_slot(slot) {
            if let Some(b) = self.blocks.get_mut(&key) {
                b.io_pins = b.io_pins.saturating_sub(1);
            }
            return Err(err);
        }

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
            for s in full_slots.into_iter().chain(window_slots) {
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
        for s in reservation.full_slots.into_iter().chain(reservation.window_slots) {
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
            .zip(verified_full)
            .zip(reservation.full_slots)
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
            .zip(verified_window)
            .zip(reservation.window_slots)
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
mod tests;
