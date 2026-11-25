//! Layer 3 paged KV block cache for `inky serve`:
//! - Shared GPU `BlockPools` (`Window` + `Full`) with zero-copy block sharing across lanes
//! - `EntryTree` (`PrefixTree<BlockHandles>`) + `RetainedStart` checkpoints (`EVERY` + `PROMPT`)
//! - Exact LRU eviction of retained start points on pool pressure (never grows or panics)
//! - Pool-pressure admission guard keeping GPU memory bounded inside Metal working set

use crate::blockbook::{EVERY, PROMPT};
use crate::blocks::{
    BlockEntry, BlockIndex, BlockKind, BlockPools, BlockTable, Slot, BLOCK, WINDOW,
};
use crate::engine::EXT;
use crate::prefix::{BlockHandles, EntryTree, PrefixTree, SealedId};
use crate::spill::{
    read_block_from_dir, RawDiskRead, ReadOutcome, Sha256, SpillBlockKind, SpillKey, SpillStore,
    WriteOutcome,
};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

#[cfg(standalone_blocks)]
#[path = "block_cache/trace.rs"]
pub(crate) mod trace;

pub const KINDS: [BlockKind; 2] = [BlockKind::Window, BlockKind::Full];

/// Periodic checkpoint stride for the first `EARLY_EVERY_LIMIT` tokens so shared system+tools
/// prompts (e.g. OpenCode ~8,500 tokens) are cached in large 2048-token tensor chunks.
pub const EARLY_EVERY_TOKENS: usize = 2048;
/// Above `EARLY_EVERY_LIMIT`, periodic checkpoints widen to 4096 tokens for full-chunk prefill.
pub const LATE_EVERY_TOKENS: usize = 4096;
pub const EARLY_EVERY_LIMIT: usize = 16384;

/// Maximum decode tokens reserved per request in the admission pool-pressure guard.
pub const MAX_ADMIT_DECODE_RESERVE: usize = 4096;

/// Maximum retained spilled start-point metadata entries tracked on disk.
pub const DEFAULT_MAX_SPILLED_STARTS: usize = 512;

/// Maximum concurrent host staging bytes buffered by the background SSD restore worker
/// (`512 MiB`, strictly `< 15%` of the 4 GiB safety headroom even at 128K context).
pub const MAX_INFLIGHT_RESTORE_STAGING_BYTES: usize = 512 * 1024 * 1024;
const SPILL_INDEX_MAGIC: [u8; 8] = *b"INKYSPID";
const SPILL_INDEX_VERSION: u32 = 2;
const SPILL_INDEX_HEADER_LEN: usize = 68;
const SPILL_INDEX_PREFIX_LEN: usize = 36;
const MAX_SPILL_INDEX_BYTES: usize = 64 * 1024 * 1024;

/// Default host-RAM cap for retained checkpoint copies (`cs` + `toks` + `entries`).
pub const DEFAULT_MAX_RETAINED_HOST_BYTES: u64 = 512 << 20;

/// Default wall-clock deadline for spilling one checkpoint to `--spill-dir` (aligned with
/// `DEFAULT_GPU_STEP_TIMEOUT` / `DEFAULT_SCHEDULER_STALL_THRESHOLD` so multi-block fsyncs on
/// large checkpoints or virtualized CI runners do not prematurely abort mid-spill).
pub const DEFAULT_SPILL_WRITE_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheError {
    PoolExhausted(BlockKind),
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheError::PoolExhausted(kind) => write!(f, "KV block pool exhausted ({kind:?})"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub slots_taken: u64,
    pub slots_reused: u64,
    pub blocks_sealed: u64,
    pub shared_dedups: u64,
    pub start_points_saved: u64,
    pub start_points_evicted: u64,
    pub start_matches: u64,
    pub cached_tokens_total: u64,
    pub window_pruned: u64,
    pub admit_deferred_pool_pressure: u64,
    pub step_deferred_pool_pressure: u64,
    pub singleflight_rebinds: u64,
    pub singleflight_saved_tokens: u64,
    pub starts_spilled: u64,
    pub starts_restored: u64,
    pub corrupt_fallbacks: u64,
    pub last_restore_us: u64,
    pub restore_us_total: u64,
    pub lanes_released: u64,
    pub spill_maint_passes: u64,
    pub spill_maint_evictions: u64,
    pub spill_write_timeouts: u64,
}

/// Per-lane Layer 3 tracking state (lives alongside `Lane` inside `LaneS`).
#[derive(Default, Debug)]
pub struct LaneBlockState {
    pub namespace: u64,
    pub sealed: Vec<SealedId>,
    pub prompt_len: usize,
    pub written: usize,
}

struct RetainedStart {
    namespace: u64,
    pos: usize,
    cs: Vec<u8>,
    reasons: u8,
    entries: Vec<(BlockKind, BlockIndex, BlockEntry)>,
    toks: Vec<u32>,
    last_used_tick: u64,
}

impl RetainedStart {
    fn host_bytes(&self) -> u64 {
        (self.cs.len() as u64)
            .saturating_add((self.toks.len() as u64).saturating_mul(4))
            .saturating_add(
                (self.entries.len() as u64)
                    .saturating_mul(std::mem::size_of::<(BlockKind, BlockIndex, BlockEntry)>() as u64),
            )
    }
}

struct SpilledStart {
    namespace: u64,
    pos: usize,
    reasons: u8,
    cs_key: SpillKey,
    entries: Vec<(BlockKind, BlockIndex, SpillKey)>,
    toks: Vec<u32>,
    last_used_tick: u64,
}

// Preserve the source files and shorter resident match while making room for a restore.
struct RestoreSource {
    spill_id: u64,
    keys: HashSet<SpillKey>,
    resident_start: Option<SealedId>,
}

/// Result of binding an admitted lane: either immediately ready in RAM (`Some` / `None`)
/// or asynchronously restoring a deeper spilled checkpoint from SSD (`Restoring`).
#[derive(Debug)]
pub enum AdmitBindOutcome {
    Immediate(Option<(usize, Vec<u8>)>),
    Restoring { target_pos: usize },
}

/// Completed background SSD restore ready to transition `Restoring -> Ready`.
#[derive(Debug)]
pub struct CompletedRestore {
    pub lane: usize,
    pub matched: Option<(usize, Vec<u8>)>,
    pub elapsed: Duration,
    pub corrupt_fallback: bool,
}

struct RestoreJob {
    lane: usize,
    restore_seq: u64,
    dir: PathBuf,
    kvb: u32,
    cs_key: SpillKey,
    blocks: Vec<(usize, SpillKey, usize)>,
    cancel_flag: Arc<AtomicBool>,
    began: Instant,
}

enum RestoreMsgPayload {
    RecurrentState {
        key: SpillKey,
        raw: RawDiskRead,
    },
    Block {
        idx: usize,
        key: SpillKey,
        raw: RawDiskRead,
    },
    Done {
        elapsed: Duration,
    },
    Aborted {
        key: SpillKey,
        raw: RawDiskRead,
    },
}

struct RestoreMsg {
    lane: usize,
    restore_seq: u64,
    budget_bytes: usize,
    payload: RestoreMsgPayload,
}

struct StagingBudget {
    inner: Mutex<usize>,
    cvar: Condvar,
}

impl StagingBudget {
    fn new() -> Self {
        Self {
            inner: Mutex::new(0),
            cvar: Condvar::new(),
        }
    }

    fn acquire(&self, bytes: usize, cancel: &AtomicBool) -> bool {
        if bytes == 0 {
            return !cancel.load(Ordering::Acquire);
        }
        let capped = bytes.min(MAX_INFLIGHT_RESTORE_STAGING_BYTES);
        let mut used = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if cancel.load(Ordering::Acquire) {
                return false;
            }
            if *used == 0 || used.saturating_add(capped) <= MAX_INFLIGHT_RESTORE_STAGING_BYTES {
                *used = used.saturating_add(capped);
                return true;
            }
            let (next, _) = self
                .cvar
                .wait_timeout(used, Duration::from_millis(5))
                .unwrap_or_else(|e| e.into_inner());
            used = next;
        }
    }

    fn release(&self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        let capped = bytes.min(MAX_INFLIGHT_RESTORE_STAGING_BYTES);
        let mut used = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        *used = used.saturating_sub(capped);
        self.cvar.notify_all();
    }
}

struct RestoreWake {
    seq: Mutex<u64>,
    cvar: Condvar,
}

impl RestoreWake {
    fn new() -> Self {
        Self {
            seq: Mutex::new(0),
            cvar: Condvar::new(),
        }
    }

    fn notify(&self) {
        let mut g = self.seq.lock().unwrap_or_else(|e| e.into_inner());
        *g = g.wrapping_add(1);
        self.cvar.notify_all();
    }

    fn wait_timeout(&self, last_seen: &mut u64, timeout: Duration) {
        let g = self.seq.lock().unwrap_or_else(|e| e.into_inner());
        if *g != *last_seen {
            *last_seen = *g;
            return;
        }
        let (g2, _) = self
            .cvar
            .wait_timeout(g, timeout)
            .unwrap_or_else(|e| e.into_inner());
        *last_seen = *g2;
    }
}

struct RestoreWorker {
    job_tx: Option<Sender<RestoreJob>>,
    msg_rx: Receiver<RestoreMsg>,
    budget: Arc<StagingBudget>,
    wake: Arc<RestoreWake>,
    last_wake_seq: u64,
    join: Option<std::thread::JoinHandle<()>>,
}

impl RestoreWorker {
    fn spawn() -> Self {
        let (job_tx, job_rx) = channel::<RestoreJob>();
        let (msg_tx, msg_rx) = channel::<RestoreMsg>();
        let budget = Arc::new(StagingBudget::new());
        let wake = Arc::new(RestoreWake::new());
        let budget_bg = budget.clone();
        let wake_bg = wake.clone();
        let join = std::thread::Builder::new()
            .name("inky-spill-restore".into())
            .spawn(move || {
                while let Ok(job) = job_rx.recv() {
                    if job.cancel_flag.load(Ordering::Acquire) {
                        continue;
                    }
                    let cs_est = crate::spill::MAX_RECURRENT_STATE_BYTES.min(4 * 1024 * 1024);
                    if !budget_bg.acquire(cs_est, &job.cancel_flag) {
                        continue;
                    }
                    let cs_raw = read_block_from_dir(&job.dir, job.kvb, job.cs_key, Some(0));
                    if job.cancel_flag.load(Ordering::Acquire) {
                        budget_bg.release(cs_est);
                        continue;
                    }
                    let cs_ok = matches!(
                        &cs_raw.outcome,
                        ReadOutcome::Hit(blk) if blk.recurrent_state.as_ref().is_some_and(|s| !s.is_empty())
                    );
                    if !cs_ok {
                        if msg_tx
                            .send(RestoreMsg {
                                lane: job.lane,
                                restore_seq: job.restore_seq,
                                budget_bytes: cs_est,
                                payload: RestoreMsgPayload::Aborted {
                                    key: job.cs_key,
                                    raw: cs_raw,
                                },
                            })
                            .is_err()
                        {
                            budget_bg.release(cs_est);
                            break;
                        }
                        wake_bg.notify();
                        continue;
                    }
                    if msg_tx
                        .send(RestoreMsg {
                            lane: job.lane,
                            restore_seq: job.restore_seq,
                            budget_bytes: cs_est,
                            payload: RestoreMsgPayload::RecurrentState {
                                key: job.cs_key,
                                raw: cs_raw,
                            },
                        })
                        .is_err()
                    {
                        budget_bg.release(cs_est);
                        break;
                    }
                    wake_bg.notify();

                    let mut aborted = false;
                    for (idx, key, expected_len) in job.blocks {
                        if !budget_bg.acquire(expected_len, &job.cancel_flag) {
                            aborted = true;
                            break;
                        }
                        let raw = read_block_from_dir(&job.dir, job.kvb, key, Some(expected_len));
                        if job.cancel_flag.load(Ordering::Acquire) {
                            budget_bg.release(expected_len);
                            aborted = true;
                            break;
                        }
                        let hit = matches!(&raw.outcome, ReadOutcome::Hit(_));
                        let payload = if hit {
                            RestoreMsgPayload::Block { idx, key, raw }
                        } else {
                            RestoreMsgPayload::Aborted { key, raw }
                        };
                        if msg_tx
                            .send(RestoreMsg {
                                lane: job.lane,
                                restore_seq: job.restore_seq,
                                budget_bytes: expected_len,
                                payload,
                            })
                            .is_err()
                        {
                            budget_bg.release(expected_len);
                            return;
                        }
                        wake_bg.notify();
                        if !hit {
                            aborted = true;
                            break;
                        }
                    }
                    if !aborted && !job.cancel_flag.load(Ordering::Acquire) {
                        if msg_tx
                            .send(RestoreMsg {
                                lane: job.lane,
                                restore_seq: job.restore_seq,
                                budget_bytes: 0,
                                payload: RestoreMsgPayload::Done {
                                    elapsed: job.began.elapsed(),
                                },
                            })
                            .is_err()
                        {
                            return;
                        }
                        wake_bg.notify();
                    }
                }
            })
            .expect("spawn inky-spill-restore worker");
        Self {
            job_tx: Some(job_tx),
            msg_rx,
            budget,
            wake,
            last_wake_seq: 0,
            join: Some(join),
        }
    }
}

impl Drop for RestoreWorker {
    fn drop(&mut self) {
        self.job_tx.take();
        while let Ok(msg) = self.msg_rx.try_recv() {
            self.budget.release(msg.budget_bytes);
        }
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

struct PendingRestore {
    restore_seq: u64,
    spill_id: u64,
    namespace: u64,
    pos: usize,
    reasons: u8,
    full_toks: Vec<u32>,
    protected_keys: Vec<SpillKey>,
    held_reused: Vec<(BlockKind, BlockIndex, BlockEntry)>,
    reserved_slots: Vec<(BlockKind, BlockIndex, BlockEntry)>,
    loaded_cs: Option<Vec<u8>>,
    loaded_blocks: usize,
    cancel_flag: Arc<AtomicBool>,
}

/// Only blocks leaving the sliding window during this span need their table entries cleared.
/// A cached prefix binds the still-live window at its checkpoint, so the same range applies on a hit.
fn newly_expired_window_blocks(start: usize, end: usize) -> std::ops::Range<usize> {
    let first_needed = |pos: usize| pos.saturating_sub(EXT - 1) / BLOCK;
    first_needed(start)..first_needed(end)
}

fn to_spill_kind(kind: BlockKind) -> SpillBlockKind {
    match kind {
        BlockKind::Full => SpillBlockKind::Full,
        BlockKind::Window => SpillBlockKind::Window,
    }
}

fn kind_idx(kind: BlockKind) -> usize {
    match kind {
        BlockKind::Window => 0,
        BlockKind::Full => 1,
    }
}

/// True when `end` (a multiple of `BLOCK`) is a periodic `EVERY` checkpoint boundary.
pub fn is_every_checkpoint(end: usize) -> bool {
    if end == 0 || !end.is_multiple_of(BLOCK) {
        return false;
    }
    if end <= EARLY_EVERY_LIMIT {
        end.is_multiple_of(EARLY_EVERY_TOKENS)
    } else {
        end.is_multiple_of(LATE_EVERY_TOKENS)
    }
}

/// Largest `EVERY` checkpoint boundary `<= n`, or `0` if `n < EARLY_EVERY_TOKENS`.
fn max_every_checkpoint_le(n: usize) -> usize {
    if n < EARLY_EVERY_TOKENS {
        0
    } else if n <= EARLY_EVERY_LIMIT {
        (n / EARLY_EVERY_TOKENS) * EARLY_EVERY_TOKENS
    } else {
        (n / LATE_EVERY_TOKENS) * LATE_EVERY_TOKENS
    }
}

fn count_every_checkpoints_le(limit: usize) -> usize {
    if limit < EARLY_EVERY_TOKENS {
        0
    } else if limit <= EARLY_EVERY_LIMIT {
        limit / EARLY_EVERY_TOKENS
    } else {
        EARLY_EVERY_LIMIT / EARLY_EVERY_TOKENS + (limit - EARLY_EVERY_LIMIT) / LATE_EVERY_TOKENS
    }
}

fn every_checkpoint_at(idx: usize) -> usize {
    let early_cnt = EARLY_EVERY_LIMIT / EARLY_EVERY_TOKENS;
    if idx < early_cnt {
        (idx + 1) * EARLY_EVERY_TOKENS
    } else {
        EARLY_EVERY_LIMIT + (idx + 1 - early_cnt) * LATE_EVERY_TOKENS
    }
}

/// Precompute compact cumulative 64-token block rolling hashes (`1 u64` per complete 64-token
/// block, `out.len() == toks.len() / BLOCK`). `out[b]` summarizes `toks[..(b + 1) * BLOCK]` so
/// checkpoint prefix equality reduces to `O(1)` `u64` lookup + binary search with a single
/// boundary slice verification.
pub fn compute_block_hashes(toks: &[u32]) -> Vec<u64> {
    let n_blocks = toks.len() / BLOCK;
    let mut out = Vec::with_capacity(n_blocks);
    let mut h = 0x9e37_79b9_7f4a_7c15u64;
    for b in 0..n_blocks {
        let blk = &toks[b * BLOCK..(b + 1) * BLOCK];
        let mut i = 0usize;
        while i + 2 <= BLOCK {
            let pair = u64::from(blk[i]) | (u64::from(blk[i + 1]) << 32);
            h ^= pair.wrapping_add(0x517c_c1b7_2722_0a95);
            h = h.rotate_left(29).wrapping_mul(0x94d0_49bb_1331_11eb);
            i += 2;
        }
        h ^= h >> 32;
        out.push(h);
    }
    out
}

/// Summary of an active prefilling lane for singleflight leader election and admission accounting.
#[derive(Clone, Copy, Debug)]
pub struct InflightPrefillLane<'a> {
    pub lane: usize,
    pub namespace: u64,
    pub toks: &'a [u32],
    pub block_hashes: Option<&'a [u64]>,
    pub n_past: usize,
    pub pf_end: usize,
    pub runnable: bool,
    pub eligible_follower: bool,
}

/// A producer must share the follower's cache namespace, be strictly ahead in
/// (progress, reverse lane ID), and seal the checkpoint where this follower waits.
fn can_lead_meta(me: &InflightPrefillLane<'_>, producer: &InflightPrefillLane<'_>, next: usize) -> bool {
    if me.namespace != producer.namespace
        || !me.runnable
        || !me.eligible_follower
        || next <= me.n_past
        || next >= me.pf_end
        || !producer.runnable
        || producer.n_past >= next
        || next >= producer.pf_end
    {
        return false;
    }
    let ahead = producer.n_past > me.n_past || (producer.n_past == me.n_past && producer.lane < me.lane);
    let prompt_checkpoint = (producer.pf_end - 1) / BLOCK * BLOCK;
    ahead && (is_every_checkpoint(next) || next == prompt_checkpoint)
}

/// A producer must be strictly ahead in (progress, reverse lane ID) and must seal the
/// checkpoint where this follower actually waits. Use the same rule in admission and scheduling.
fn can_lead(me: &InflightPrefillLane<'_>, producer: &InflightPrefillLane<'_>, next: usize) -> bool {
    if !can_lead_meta(me, producer, next) {
        return false;
    }
    let next_bi = next / BLOCK - 1;
    if let (Some(mh), Some(ph)) = (
        me.block_hashes.and_then(|h| h.get(next_bi)),
        producer.block_hashes.and_then(|h| h.get(next_bi)),
    ) {
        if mh != ph {
            return false;
        }
    }
    std::ptr::eq(me.toks.as_ptr(), producer.toks.as_ptr()) || producer.toks[..next] == me.toks[..next]
}

#[cfg(any(test, standalone_blocks))]
pub fn find_prefill_leader(lanes: &[InflightPrefillLane<'_>], idx: usize) -> Option<usize> {
    let me = lanes.get(idx)?;
    let next = effective_pf_target(me.n_past, me.pf_end);
    lanes.iter().find(|producer| can_lead(me, producer, next)).map(|producer| producer.lane)
}

/// Single-pass leader election across all active prefilling lanes using precomputed 64-token
/// block hashes so followers sharing a `(next, hash)` bucket verify against one canonical leader
/// without all-pairs token slice scans.
pub fn find_prefill_leaders(lanes: &[InflightPrefillLane<'_>]) -> Vec<Option<usize>> {
    let mut out = vec![None; lanes.len()];
    if lanes.len() <= 1 {
        return out;
    }
    let owned: Vec<Vec<u64>> = lanes
        .iter()
        .map(|l| {
            if l.block_hashes.is_some() {
                Vec::new()
            } else {
                compute_block_hashes(l.toks)
            }
        })
        .collect();
    let hashes_of = |i: usize| -> &[u64] {
        lanes[i].block_hashes.unwrap_or(&owned[i])
    };

    for (idx, me) in lanes.iter().enumerate() {
        let next = effective_pf_target(me.n_past, me.pf_end);
        if !me.runnable || !me.eligible_follower || next <= me.n_past || next >= me.pf_end {
            continue;
        }
        let next_bi = next / BLOCK - 1;
        let Some(&mh) = hashes_of(idx).get(next_bi) else {
            continue;
        };
        for (p_idx, producer) in lanes.iter().enumerate() {
            if !can_lead_meta(me, producer, next) {
                continue;
            }
            if hashes_of(p_idx).get(next_bi) != Some(&mh) {
                continue;
            }
            if std::ptr::eq(me.toks.as_ptr(), producer.toks.as_ptr())
                || producer.toks[..next] == me.toks[..next]
            {
                out[idx] = Some(producer.lane);
                break;
            }
        }
    }
    out
}

/// Reference exact pair shared checkpoint (used directly as collision fallback).
fn pair_shared_checkpoint_exact(
    me: &InflightPrefillLane<'_>,
    producer: &InflightPrefillLane<'_>,
    next: usize,
    final_checkpoint: usize,
) -> usize {
    if !can_lead(me, producer, next) {
        return 0;
    }
    let limit = me.pf_end.min(producer.pf_end) - 1;
    let common_final = final_checkpoint == (producer.pf_end - 1) / BLOCK * BLOCK;
    if common_final
        && (std::ptr::eq(me.toks.as_ptr(), producer.toks.as_ptr())
            || me.toks[next..final_checkpoint] == producer.toks[next..final_checkpoint])
    {
        return final_checkpoint;
    }
    let last_periodic = max_every_checkpoint_le(limit);
    if last_periodic >= next {
        if std::ptr::eq(me.toks.as_ptr(), producer.toks.as_ptr())
            || me.toks[next..last_periodic] == producer.toks[next..last_periodic]
        {
            return last_periodic;
        }
        let before = &me.toks[next..last_periodic];
        let after = &producer.toks[next..last_periodic];
        let common = next + before.iter().zip(after).take_while(|(a, b)| a == b).count();
        return max_every_checkpoint_le(common);
    }
    0
}

/// Candidate checkpoint depth between `(me, me_hashes)` and `(producer, prod_hashes)` using
/// ONLY `O(log(blocks))` `u64` block-hash lookups (zero token-slice comparisons).
fn pair_hash_candidate_checkpoint(
    me: &InflightPrefillLane<'_>,
    me_hashes: &[u64],
    producer: &InflightPrefillLane<'_>,
    prod_hashes: &[u64],
    next: usize,
    final_checkpoint: usize,
) -> usize {
    if !can_lead_meta(me, producer, next) {
        return 0;
    }
    let next_bi = next / BLOCK - 1;
    if me_hashes.get(next_bi) != prod_hashes.get(next_bi) || me_hashes.get(next_bi).is_none() {
        return 0;
    }
    let prod_final = (producer.pf_end - 1) / BLOCK * BLOCK;
    if final_checkpoint == prod_final && final_checkpoint >= next {
        let f_bi = final_checkpoint / BLOCK - 1;
        if me_hashes.get(f_bi).is_some() && me_hashes.get(f_bi) == prod_hashes.get(f_bi) {
            return final_checkpoint;
        }
    }
    let limit = me.pf_end.min(producer.pf_end) - 1;
    let cnt = count_every_checkpoints_le(limit);
    if cnt == 0 {
        return 0;
    }
    let mut lo_start = 0usize;
    while lo_start < cnt && every_checkpoint_at(lo_start) < next {
        lo_start += 1;
    }
    if lo_start >= cnt {
        return 0;
    }
    let first_bi = every_checkpoint_at(lo_start) / BLOCK - 1;
    if me_hashes.get(first_bi).is_none() || me_hashes.get(first_bi) != prod_hashes.get(first_bi) {
        return 0;
    }
    let mut lo = lo_start;
    let mut hi = cnt - 1;
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let bi = every_checkpoint_at(mid) / BLOCK - 1;
        if me_hashes.get(bi).is_some() && me_hashes.get(bi) == prod_hashes.get(bi) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    every_checkpoint_at(lo)
}

fn shared_checkpoint_against_hashed<'a>(
    me: &InflightPrefillLane<'_>,
    me_hashes: &[u64],
    producers: impl Iterator<Item = (&'a InflightPrefillLane<'a>, &'a [u64])> + Clone,
) -> usize {
    let next = effective_pf_target(me.n_past, me.pf_end);
    if !me.runnable || !me.eligible_follower || next <= me.n_past || next >= me.pf_end {
        return 0;
    }
    let final_checkpoint = me.pf_end.saturating_sub(1) / BLOCK * BLOCK;
    let mut best_ckpt = 0usize;
    let mut best_prod: Option<&InflightPrefillLane<'_>> = None;
    for (prod, ph) in producers.clone() {
        let cand = pair_hash_candidate_checkpoint(me, me_hashes, prod, ph, next, final_checkpoint);
        if cand > best_ckpt {
            best_ckpt = cand;
            best_prod = Some(prod);
            if best_ckpt == final_checkpoint {
                break;
            }
        }
    }
    let Some(winner) = best_prod else {
        return 0;
    };
    if std::ptr::eq(me.toks.as_ptr(), winner.toks.as_ptr())
        || me.toks[..best_ckpt] == winner.toks[..best_ckpt]
    {
        return best_ckpt;
    }
    // Hash-collision fallback: exact check across producers preserves 100% parity.
    let mut exact_best = 0usize;
    for (prod, _) in producers {
        exact_best = exact_best.max(pair_shared_checkpoint_exact(me, prod, next, final_checkpoint));
        if exact_best == final_checkpoint {
            break;
        }
    }
    exact_best
}

/// Furthest checkpoint these lanes will rendezvous at before their inputs diverge.
/// A producer's final, nonperiodic PROMPT checkpoint only counts if the follower also stops there.
#[cfg(any(test, standalone_blocks))]
fn inflight_shared_checkpoint(me: &InflightPrefillLane<'_>, lanes: &[InflightPrefillLane<'_>]) -> usize {
    let me_owned;
    let me_hashes: &[u64] = match me.block_hashes {
        Some(h) => h,
        None => {
            me_owned = compute_block_hashes(me.toks);
            &me_owned
        }
    };
    let owned: Vec<Vec<u64>> = lanes
        .iter()
        .map(|l| {
            if l.block_hashes.is_some() {
                Vec::new()
            } else {
                compute_block_hashes(l.toks)
            }
        })
        .collect();
    let iter = lanes
        .iter()
        .enumerate()
        .map(|(i, l)| (l, l.block_hashes.unwrap_or(&owned[i])));
    shared_checkpoint_against_hashed(me, me_hashes, iter)
}

/// Compute the capped prefill target for the current step so `finish_prefill_span` lands
/// cleanly on checkpoint boundaries (`2048`-token / `4096`-token `EVERY` boundaries and the
/// prompt's final `((pf_end - 1) / 64) * 64` `PROMPT` boundary) where `get_cs()` matches `end`.
pub fn effective_pf_target(n_past: usize, pf_end: usize) -> usize {
    if n_past >= pf_end || pf_end == 0 {
        return pf_end;
    }
    let prompt_ckpt = ((pf_end - 1) / BLOCK) * BLOCK;
    if n_past >= prompt_ckpt {
        return pf_end;
    }
    let stride = if n_past < EARLY_EVERY_LIMIT {
        EARLY_EVERY_TOKENS
    } else {
        LATE_EVERY_TOKENS
    };
    let next_every = ((n_past / stride) + 1) * stride;
    next_every.min(prompt_ckpt)
}

/// Summary of an active lane's remaining `Full` block demand for admission reservation.
#[derive(Clone, Copy, Debug)]
pub struct InflightLaneReserve<'a> {
    pub lane: usize,
    pub namespace: u64,
    pub toks: &'a [u32],
    pub block_hashes: Option<&'a [u64]>,
    pub n_past: usize,
    pub pf_end: usize,
    pub cached: usize,
    pub written: usize,
    pub gen: usize,
    pub max_gen: usize,
    pub runnable: bool,
}

impl<'a> InflightLaneReserve<'a> {
    fn reserved_end(&self) -> usize {
        let prompt_end = self.n_past.max(self.pf_end);
        // The first generated token comes from the final prompt row. For longer
        // replies, keep reserving the next bounded tranche after earlier tokens were emitted.
        prompt_end + self.max_gen.saturating_sub(self.gen.max(1)).min(MAX_ADMIT_DECODE_RESERVE)
    }

    fn as_prefill_lane(&self) -> Option<InflightPrefillLane<'a>> {
        (self.gen == 0 && self.n_past < self.pf_end).then_some(InflightPrefillLane {
            lane: self.lane,
            namespace: self.namespace,
            toks: self.toks,
            block_hashes: self.block_hashes,
            n_past: self.n_past,
            pf_end: self.pf_end,
            runnable: self.runnable,
            eligible_follower: self.n_past == self.cached && self.written == self.n_past,
        })
    }
}

struct PreparedPrefillEntry<'a> {
    pf: InflightPrefillLane<'a>,
    owned_hashes: Vec<u64>,
    written: usize,
    reserved_end_blocks: usize,
    credited: usize,
}

impl<'a> PreparedPrefillEntry<'a> {
    fn hashes(&self) -> &[u64] {
        self.pf.block_hashes.unwrap_or(&self.owned_hashes)
    }

    fn demanded_blocks(&self, credited: usize) -> usize {
        self.reserved_end_blocks - self.written.max(credited).div_ceil(BLOCK)
    }
}

type LastEvalCredited = (usize, u64, usize, usize, usize, usize);

/// Incremental admission cohort precomputed once per scheduler admission pass so each
/// candidate check runs in `O(prefilling_lanes * log(blocks))` with zero all-pairs rescans.
pub struct PreparedInflightCohort<'a> {
    len: usize,
    prefills: Vec<PreparedPrefillEntry<'a>>,
    base_reserved_full: usize,
    base_window_growth: usize,
    base_steady_window: usize,
    /// Cached `(usable_window, available_window, available_full)` headroom so bursts of
    /// candidate checks do not re-scan 10,000+ `BlockPool` records per queued job.
    cached_headroom: std::cell::Cell<Option<(usize, usize, usize)>>,
    /// Memoized `(toks_ptr, namespace, pf_end, n_past, lane, credited)` from the preceding `eval_with_candidate`
    /// so `push_admitted` commits in `O(1)` without repeating the boundary slice check.
    last_eval_credited: std::cell::Cell<Option<LastEvalCredited>>,
}

impl<'a> PreparedInflightCohort<'a> {
    pub fn empty() -> Self {
        Self {
            len: 0,
            prefills: Vec::new(),
            base_reserved_full: 0,
            base_window_growth: 0,
            base_steady_window: 0,
            cached_headroom: std::cell::Cell::new(None),
            last_eval_credited: std::cell::Cell::new(None),
        }
    }

    pub fn invalidate_headroom(&self) {
        self.cached_headroom.set(None);
    }

    #[cfg(any(test, standalone_blocks))]
    pub fn new(active: &[InflightLaneReserve<'a>]) -> Self {
        let mut cohort = Self::empty();
        for &lane in active {
            cohort.push_admitted(lane);
        }
        cohort
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[cfg(any(test, standalone_blocks))]
    pub fn reserved_full_slots(&self) -> usize {
        self.base_reserved_full
    }

    /// Evaluate total `reserved_inflight_full_slots` if `cand` were added to this cohort,
    /// touching only `O(prefilling_lanes * log(blocks))` hashes and at most one slice check.
    fn eval_with_candidate(
        &self,
        cand: &InflightLaneReserve<'_>,
        cand_hashes: &[u64],
    ) -> (usize, usize, usize) {
        let cand_end = (cand.pf_end + cand.max_gen.saturating_sub(1)).div_ceil(BLOCK);
        let win_growth = self.base_window_growth + (cand_end - cand.written.div_ceil(BLOCK));
        let steady_win = self.base_steady_window + cand_end.min(WINDOW / BLOCK + 1);

        let cand_res_blocks = cand.reserved_end().div_ceil(BLOCK);
        let Some(cand_pf) = cand.as_prefill_lane() else {
            let full = self.base_reserved_full + (cand_res_blocks - cand.written.div_ceil(BLOCK));
            return (full, win_growth, steady_win);
        };

        // 1. Best existing producer for `cand_pf`.
        let cand_credited = if cand_pf.eligible_follower {
            let iter = self.prefills.iter().map(|e| (&e.pf, e.hashes()));
            shared_checkpoint_against_hashed(&cand_pf, cand_hashes, iter)
        } else {
            0
        };
        self.last_eval_credited.set(Some((
            cand.toks.as_ptr() as usize,
            cand.namespace,
            cand.pf_end,
            cand.n_past,
            cand.lane,
            cand_credited,
        )));
        let mut full = self.base_reserved_full
            + (cand_res_blocks - cand.written.max(cand_credited).div_ceil(BLOCK));

        // 2. Check if `cand_pf` is strictly ahead of any existing eligible follower (e.g. when
        //    reusing a lower-numbered idle lane or binding a deeper cached prefix).
        if cand_pf.runnable {
            for existing in &self.prefills {
                if !existing.pf.runnable || !existing.pf.eligible_follower {
                    continue;
                }
                let ahead = cand_pf.n_past > existing.pf.n_past
                    || (cand_pf.n_past == existing.pf.n_past && cand_pf.lane < existing.pf.lane);
                if !ahead {
                    continue;
                }
                let ex_next = effective_pf_target(existing.pf.n_past, existing.pf.pf_end);
                let ex_final = existing.pf.pf_end.saturating_sub(1) / BLOCK * BLOCK;
                let cand_for_ex = pair_hash_candidate_checkpoint(
                    &existing.pf,
                    existing.hashes(),
                    &cand_pf,
                    cand_hashes,
                    ex_next,
                    ex_final,
                );
                if cand_for_ex > existing.credited {
                    let verified = if std::ptr::eq(existing.pf.toks.as_ptr(), cand_pf.toks.as_ptr())
                        || existing.pf.toks[..cand_for_ex] == cand_pf.toks[..cand_for_ex]
                    {
                        cand_for_ex
                    } else {
                        pair_shared_checkpoint_exact(&existing.pf, &cand_pf, ex_next, ex_final)
                    };
                    let new_cred = existing.credited.max(verified);
                    if new_cred > existing.credited {
                        full = full + existing.demanded_blocks(new_cred)
                            - existing.demanded_blocks(existing.credited);
                    }
                }
            }
        }

        (full, win_growth, steady_win)
    }

    /// Commit an admitted lane into this cohort in `O(prefilling_lanes * log(blocks))`.
    pub fn push_admitted(&mut self, lane: InflightLaneReserve<'a>) {
        self.len += 1;
        let end = (lane.pf_end + lane.max_gen.saturating_sub(1)).div_ceil(BLOCK);
        self.base_window_growth += end - lane.written.div_ceil(BLOCK);
        self.base_steady_window += end.min(WINDOW / BLOCK + 1);

        let res_blocks = lane.reserved_end().div_ceil(BLOCK);
        let Some(pf) = lane.as_prefill_lane() else {
            self.base_reserved_full += res_blocks - lane.written.div_ceil(BLOCK);
            return;
        };

        let owned_hashes = if pf.block_hashes.is_some() {
            Vec::new()
        } else {
            compute_block_hashes(pf.toks)
        };
        let new_hashes: &[u64] = pf.block_hashes.unwrap_or(&owned_hashes);

        let credited = if pf.eligible_follower {
            if let Some((ptr, ns, len, np, ln, memo_cred)) = self.last_eval_credited.take() {
                if ptr == pf.toks.as_ptr() as usize
                    && ns == pf.namespace
                    && len == pf.pf_end
                    && np == pf.n_past
                    && ln == pf.lane
                {
                    memo_cred
                } else {
                    let iter = self.prefills.iter().map(|e| (&e.pf, e.hashes()));
                    shared_checkpoint_against_hashed(&pf, new_hashes, iter)
                }
            } else {
                let iter = self.prefills.iter().map(|e| (&e.pf, e.hashes()));
                shared_checkpoint_against_hashed(&pf, new_hashes, iter)
            }
        } else {
            self.last_eval_credited.set(None);
            0
        };

        if pf.runnable {
            for existing in &mut self.prefills {
                if !existing.pf.runnable || !existing.pf.eligible_follower {
                    continue;
                }
                let ahead = pf.n_past > existing.pf.n_past
                    || (pf.n_past == existing.pf.n_past && pf.lane < existing.pf.lane);
                if !ahead {
                    continue;
                }
                let ex_next = effective_pf_target(existing.pf.n_past, existing.pf.pf_end);
                let ex_final = existing.pf.pf_end.saturating_sub(1) / BLOCK * BLOCK;
                let ex_h = existing.pf.block_hashes.unwrap_or(&existing.owned_hashes);
                let cand_for_ex = pair_hash_candidate_checkpoint(
                    &existing.pf,
                    ex_h,
                    &pf,
                    new_hashes,
                    ex_next,
                    ex_final,
                );
                if cand_for_ex > existing.credited {
                    let verified = if std::ptr::eq(existing.pf.toks.as_ptr(), pf.toks.as_ptr())
                        || existing.pf.toks[..cand_for_ex] == pf.toks[..cand_for_ex]
                    {
                        cand_for_ex
                    } else {
                        pair_shared_checkpoint_exact(&existing.pf, &pf, ex_next, ex_final)
                    };
                    if verified > existing.credited {
                        let old_blocks = existing.demanded_blocks(existing.credited);
                        existing.credited = verified;
                        let new_blocks = existing.demanded_blocks(existing.credited);
                        self.base_reserved_full = self.base_reserved_full + new_blocks - old_blocks;
                    }
                }
            }
        }

        let entry = PreparedPrefillEntry {
            pf,
            owned_hashes,
            written: lane.written,
            reserved_end_blocks: res_blocks,
            credited,
        };
        self.base_reserved_full += entry.demanded_blocks(credited);
        self.prefills.push(entry);
    }
}

/// Pool slots not already owned by a lane or covered by a strictly earlier producer.
/// Including the incoming lane in the same election handles reuse of a low-numbered idle lane.
#[cfg(any(test, standalone_blocks))]
pub fn reserved_inflight_full_slots(lanes: &[InflightLaneReserve<'_>]) -> usize {
    let _ = inflight_shared_checkpoint as fn(&InflightPrefillLane<'_>, &[InflightPrefillLane<'_>]) -> usize;
    PreparedInflightCohort::new(lanes).reserved_full_slots()
}

fn entry_is_live_sealed(pools: &BlockPools, kind: BlockKind, e: BlockEntry) -> bool {
    pools.of(kind).record(e.slot()).ok().is_some_and(|r| {
        r.generation() == e.generation() && r.sealed() && r.refs() > 0
    })
}

pub struct ServeBlockCache {
    tree: EntryTree,
    starts: HashMap<SealedId, RetainedStart>,
    /// Per-slot refcount contributed exclusively by live entries in `self.starts`.
    /// Maintained in lockstep with `add_ref`/`drop_ref` on `RetainedStart` insert/evict
    /// so `free_plus_evictable_slots` is exact and allocation-free.
    start_slot_refs: [Vec<u32>; 2],
    tick: u64,
    stats: CacheStats,
    spill_store: Option<SpillStore>,
    spilled_tree: PrefixTree<u64>,
    spilled_starts: HashMap<u64, SpilledStart>,
    spilled_tips: HashMap<u64, SealedId>,
    spill_key_owners: HashMap<SpillKey, Vec<u64>>,
    entry_spill_keys: HashMap<(u8, u32, u32), SpillKey>,
    spill_key_to_entry: HashMap<SpillKey, (u8, u32, u32)>,
    next_spill_id: u64,
    max_spilled_starts: usize,
    restore_worker: Option<RestoreWorker>,
    pending_restores: HashMap<usize, PendingRestore>,
    in_flight_restore_keys: HashMap<SpillKey, u32>,
    in_flight_restore_spills: HashSet<u64>,
    next_restore_seq: u64,
    retained_host_bytes: u64,
    max_retained_host_bytes: u64,
    spill_write_timeout: Duration,
}

impl Default for ServeBlockCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ServeBlockCache {
    pub fn new() -> Self {
        Self {
            tree: EntryTree::new(),
            starts: HashMap::new(),
            start_slot_refs: [Vec::new(), Vec::new()],
            tick: 0,
            stats: CacheStats::default(),
            spill_store: None,
            spilled_tree: PrefixTree::new(),
            spilled_starts: HashMap::new(),
            spilled_tips: HashMap::new(),
            spill_key_owners: HashMap::new(),
            entry_spill_keys: HashMap::new(),
            spill_key_to_entry: HashMap::new(),
            next_spill_id: 1,
            max_spilled_starts: DEFAULT_MAX_SPILLED_STARTS,
            restore_worker: None,
            pending_restores: HashMap::new(),
            in_flight_restore_keys: HashMap::new(),
            in_flight_restore_spills: HashSet::new(),
            next_restore_seq: 1,
            retained_host_bytes: 0,
            max_retained_host_bytes: DEFAULT_MAX_RETAINED_HOST_BYTES,
            spill_write_timeout: DEFAULT_SPILL_WRITE_TIMEOUT,
        }
    }

    /// Host-copy byte reserve carved out of `available_kv_budget_bytes` before sizing GPU `BlockPools`.
    pub fn recommended_host_budget_bytes(available_kv_budget_bytes: u64) -> u64 {
        if available_kv_budget_bytes == 0 {
            return 0;
        }
        (available_kv_budget_bytes / 32).clamp(16 << 20, DEFAULT_MAX_RETAINED_HOST_BYTES)
    }

    pub fn set_max_retained_host_bytes(&mut self, max_bytes: u64) {
        self.max_retained_host_bytes = max_bytes;
    }

    #[cfg(any(test, standalone_blocks))]
    pub fn retained_host_bytes(&self) -> u64 {
        self.retained_host_bytes
    }

    #[cfg(any(test, standalone_blocks))]
    pub fn set_spill_write_timeout(&mut self, timeout: Duration) {
        self.spill_write_timeout = timeout;
    }

    pub fn with_spill(store: SpillStore) -> Self {
        let mut cache = Self::new();
        cache.spill_store = Some(store);
        cache.load_spill_index_on_open();
        cache
    }

    pub fn spill_store(&self) -> Option<&SpillStore> {
        self.spill_store.as_ref()
    }

    fn pin_inflight_restore_keys(&mut self, spill_id: u64, keys: &[SpillKey]) {
        self.in_flight_restore_spills.insert(spill_id);
        for &k in keys {
            *self.in_flight_restore_keys.entry(k).or_insert(0) += 1;
        }
    }

    fn unpin_inflight_restore_keys(&mut self, spill_id: u64, keys: &[SpillKey]) {
        for &k in keys {
            if let Some(cnt) = self.in_flight_restore_keys.get_mut(&k) {
                *cnt = cnt.saturating_sub(1);
                if *cnt == 0 {
                    self.in_flight_restore_keys.remove(&k);
                }
            }
        }
        let still_active = self
            .pending_restores
            .values()
            .any(|pr| pr.spill_id == spill_id);
        if !still_active {
            self.in_flight_restore_spills.remove(&spill_id);
        }
    }

    fn protected_spill_keys(&self, extra: Option<&RestoreSource>) -> HashSet<SpillKey> {
        let mut out: HashSet<SpillKey> = self.in_flight_restore_keys.keys().copied().collect();
        if let Some(source) = extra {
            out.extend(source.keys.iter().copied());
        }
        out
    }

    /// Persist `self.spilled_starts` crash-safely to `spill-index.bin` (pure-Rust binary format
    /// protected by `Sha256`, zero external crates).
    fn persist_spill_index(&mut self) {
        let Some(store) = self.spill_store.as_ref() else {
            return;
        };
        let kvb = store.kvb();
        let mut records: Vec<(u64, &SpilledStart)> =
            self.spilled_starts.iter().map(|(&id, sp)| (id, sp)).collect();
        records.sort_by_key(|&(id, sp)| (sp.last_used_tick, id));

        let mut payload = Vec::new();
        for (id, sp) in &records {
            if sp.toks.len() != sp.pos || sp.pos == 0 || !sp.pos.is_multiple_of(BLOCK) {
                continue;
            }
            payload.extend_from_slice(&id.to_le_bytes());
            payload.extend_from_slice(&sp.namespace.to_le_bytes());
            payload.extend_from_slice(&(sp.pos as u64).to_le_bytes());
            payload.push(sp.reasons);
            payload.extend_from_slice(&sp.last_used_tick.to_le_bytes());
            payload.push(sp.cs_key.kind as u8);
            payload.extend_from_slice(&sp.cs_key.id.to_le_bytes());
            payload.extend_from_slice(&(sp.toks.len() as u32).to_le_bytes());
            for &t in &sp.toks {
                payload.extend_from_slice(&t.to_le_bytes());
            }
            payload.extend_from_slice(&(sp.entries.len() as u32).to_le_bytes());
            for &(kind, bi, sk) in &sp.entries {
                payload.push(kind_idx(kind) as u8);
                payload.extend_from_slice(&bi.get().to_le_bytes());
                payload.push(sk.kind as u8);
                payload.extend_from_slice(&sk.id.to_le_bytes());
            }
        }
        let valid_count = records
            .iter()
            .filter(|(_, sp)| sp.toks.len() == sp.pos && sp.pos > 0 && sp.pos.is_multiple_of(BLOCK))
            .count() as u32;

        let mut header = [0u8; SPILL_INDEX_HEADER_LEN];
        header[0..8].copy_from_slice(&SPILL_INDEX_MAGIC);
        header[8..12].copy_from_slice(&SPILL_INDEX_VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&kvb.to_le_bytes());
        header[16..24].copy_from_slice(&self.next_spill_id.to_le_bytes());
        header[24..32].copy_from_slice(&self.tick.to_le_bytes());
        header[32..36].copy_from_slice(&valid_count.to_le_bytes());
        let payload_len = payload.len() as u64;

        let mut hasher = Sha256::new();
        hasher.update(&header[0..SPILL_INDEX_PREFIX_LEN]);
        hasher.update(&payload_len.to_le_bytes());
        hasher.update(&payload);
        let digest = hasher.finalize();
        header[36..68].copy_from_slice(&digest);

        let mut full = Vec::with_capacity(SPILL_INDEX_HEADER_LEN + 8 + payload.len());
        full.extend_from_slice(&header[0..SPILL_INDEX_PREFIX_LEN]);
        full.extend_from_slice(&payload_len.to_le_bytes());
        full.extend_from_slice(&digest);
        full.extend_from_slice(&payload);
        let _ = store.write_index_bytes(&full);
    }

    fn load_spill_index_on_open(&mut self) {
        let Some(raw) = self
            .spill_store
            .as_mut()
            .and_then(|s| s.read_index_bytes(MAX_SPILL_INDEX_BYTES))
        else {
            if let Some(store) = self.spill_store.as_mut() {
                store.retain_keys(&HashSet::new());
            }
            return;
        };
        let store_kvb = self.spill_store.as_ref().unwrap().kvb();
        let reject = |this: &mut Self| {
            if let Some(store) = this.spill_store.as_mut() {
                store.remove_index(true);
                store.retain_keys(&HashSet::new());
            }
        };
        if raw.len() < SPILL_INDEX_PREFIX_LEN + 8 + 32 {
            reject(self);
            return;
        }
        if raw[0..8] != SPILL_INDEX_MAGIC {
            reject(self);
            return;
        }
        let version = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        let file_kvb = u32::from_le_bytes(raw[12..16].try_into().unwrap());
        if version != SPILL_INDEX_VERSION || file_kvb != store_kvb {
            reject(self);
            return;
        }
        let saved_next_spill_id = u64::from_le_bytes(raw[16..24].try_into().unwrap());
        let saved_tick = u64::from_le_bytes(raw[24..32].try_into().unwrap());
        let count = u32::from_le_bytes(raw[32..36].try_into().unwrap()) as usize;
        let payload_len = u64::from_le_bytes(raw[36..44].try_into().unwrap());
        let stored_digest: [u8; 32] = raw[44..76].try_into().unwrap();
        let payload = &raw[76..];
        if payload.len() as u64 != payload_len {
            reject(self);
            return;
        }
        let mut hasher = Sha256::new();
        hasher.update(&raw[0..44]);
        hasher.update(payload);
        if hasher.finalize() != stored_digest {
            reject(self);
            return;
        }

        let parse_records = || -> Option<Vec<(u64, SpilledStart)>> {
            let mut cur = 0usize;
            let mut out = Vec::with_capacity(count.min(self.max_spilled_starts + 16));
            for _ in 0..count {
                if cur + 46 > payload.len() {
                    return None;
                }
                let id = u64::from_le_bytes(payload[cur..cur + 8].try_into().ok()?);
                let namespace = u64::from_le_bytes(payload[cur + 8..cur + 16].try_into().ok()?);
                let pos = usize::try_from(u64::from_le_bytes(payload[cur + 16..cur + 24].try_into().ok()?)).ok()?;
                let reasons = payload[cur + 24];
                let last_used_tick = u64::from_le_bytes(payload[cur + 25..cur + 33].try_into().ok()?);
                let cs_kind = SpillBlockKind::from_u32(u32::from(payload[cur + 33]))?;
                let cs_id = u64::from_le_bytes(payload[cur + 34..cur + 42].try_into().ok()?);
                let toks_len = u32::from_le_bytes(payload[cur + 42..cur + 46].try_into().ok()?) as usize;
                cur += 46;
                if pos == 0 || !pos.is_multiple_of(BLOCK) || toks_len != pos {
                    return None;
                }
                let toks_bytes = toks_len.checked_mul(4)?;
                if cur + toks_bytes + 4 > payload.len() {
                    return None;
                }
                let mut toks = Vec::with_capacity(toks_len);
                for i in 0..toks_len {
                    let off = cur + i * 4;
                    toks.push(u32::from_le_bytes(payload[off..off + 4].try_into().ok()?));
                }
                cur += toks_bytes;
                let entries_len = u32::from_le_bytes(payload[cur..cur + 4].try_into().ok()?) as usize;
                cur += 4;
                let entries_bytes = entries_len.checked_mul(14)?;
                if cur + entries_bytes > payload.len() {
                    return None;
                }
                let mut entries = Vec::with_capacity(entries_len);
                for i in 0..entries_len {
                    let off = cur + i * 14;
                    let bk = match payload[off] {
                        0 => BlockKind::Window,
                        1 => BlockKind::Full,
                        _ => return None,
                    };
                    let bi = BlockIndex::new(u32::from_le_bytes(payload[off + 1..off + 5].try_into().ok()?));
                    let sk_kind = SpillBlockKind::from_u32(u32::from(payload[off + 5]))?;
                    let sk_id = u64::from_le_bytes(payload[off + 6..off + 14].try_into().ok()?);
                    entries.push((bk, bi, SpillKey::new(sk_kind, sk_id)));
                }
                cur += entries_bytes;
                out.push((
                    id,
                    SpilledStart {
                        namespace,
                        pos,
                        reasons,
                        cs_key: SpillKey::new(cs_kind, cs_id),
                        entries,
                        toks,
                        last_used_tick,
                    },
                ));
            }
            (cur == payload.len()).then_some(out)
        };

        let Some(mut decoded) = parse_records() else {
            reject(self);
            return;
        };
        decoded.sort_by_key(|&(id, ref sp)| (sp.last_used_tick, id));
        if decoded.len() > self.max_spilled_starts {
            let drop_n = decoded.len() - self.max_spilled_starts;
            decoded.drain(0..drop_n);
        }

        let mut max_id = saved_next_spill_id.max(1);
        let mut max_tick = saved_tick;
        let mut kept_keys = HashSet::new();
        let mut skipped_any = false;
        for (spill_id, sp) in decoded {
            let store = self.spill_store.as_ref().unwrap();
            let all_present = store.contains(sp.cs_key)
                && sp.entries.iter().all(|&(_, _, sk)| store.contains(sk));
            if !all_present {
                skipped_any = true;
                continue;
            }
            max_id = max_id.max(spill_id.saturating_add(1));
            max_tick = max_tick.max(sp.last_used_tick);
            let n_blocks = sp.pos / BLOCK;
            let mut parent = None;
            for b in 0..n_blocks {
                let id = self
                    .spilled_tree
                    .insert_in_namespace(sp.namespace, parent, b * BLOCK, &sp.toks[b * BLOCK..(b + 1) * BLOCK], spill_id)
                    .expect("insert loaded spilled block into spilled_tree");
                parent = Some(id);
            }
            let tip = parent.expect("non-empty loaded spilled blocks");
            if self.spilled_tree.start_holders(tip) > 0 {
                if let Some(old_id) = self.spilled_tree.handle(tip) {
                    self.remove_spilled_start_inner(old_id, false);
                }
                let mut p = None;
                for b in 0..n_blocks {
                    let id = self
                        .spilled_tree
                        .insert_in_namespace(sp.namespace, p, b * BLOCK, &sp.toks[b * BLOCK..(b + 1) * BLOCK], spill_id)
                        .expect("re-insert loaded spilled block into spilled_tree");
                    p = Some(id);
                }
                let new_tip = p.expect("non-empty loaded spilled blocks");
                let _ = self.spilled_tree.set_handle(new_tip, spill_id);
                self.spilled_tree.retain_start(new_tip).expect("retain loaded spilled start");
                self.spilled_tips.insert(spill_id, new_tip);
            } else {
                let _ = self.spilled_tree.set_handle(tip, spill_id);
                self.spilled_tree.retain_start(tip).expect("retain loaded spilled start");
                self.spilled_tips.insert(spill_id, tip);
            }
            kept_keys.insert(sp.cs_key);
            self.spill_key_owners.entry(sp.cs_key).or_default().push(spill_id);
            for &(_, _, sk) in &sp.entries {
                kept_keys.insert(sk);
                self.spill_key_owners.entry(sk).or_default().push(spill_id);
            }
            self.spilled_starts.insert(spill_id, sp);
        }
        self.next_spill_id = max_id;
        self.tick = self.tick.max(max_tick);
        let swept = self
            .spill_store
            .as_mut()
            .map_or(0, |s| s.retain_keys(&kept_keys));
        if skipped_any || swept > 0 {
            self.persist_spill_index();
        }
    }

    /// Bounded host/disk-only maintenance safe to overlap with an in-flight GPU step:
    /// enforces `max_spilled_starts` and `SpillStore` disk budget without touching
    /// `BlockPools`, `BlockTable`, or active RAM pins.
    pub fn maintain_spill_during_gpu(&mut self) -> usize {
        if self.spill_store.is_none() {
            return 0;
        }
        let mut evicted = 0usize;
        let mut index_dirty = false;
        while self.spilled_starts.len() > self.max_spilled_starts {
            let oldest = self
                .spilled_starts
                .iter()
                .filter(|(id, _)| !self.in_flight_restore_spills.contains(id))
                .min_by_key(|(_, sp)| (sp.last_used_tick, std::cmp::Reverse(sp.pos)))
                .map(|(&id, _)| id);
            let Some(oid) = oldest else { break };
            if self.remove_spilled_start_inner(oid, true) {
                index_dirty = true;
            }
            evicted += 1;
        }
        let protected = self.protected_spill_keys(None);
        let gc_evicted = if let Some(store) = self.spill_store.as_mut() {
            if store.used_bytes() > store.max_bytes() {
                store.evict_lru_to_fit(0, &protected)
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        if !gc_evicted.is_empty() {
            evicted += gc_evicted.len();
            self.handle_disk_gc_evicted(&gc_evicted);
            index_dirty = false; // handle_disk_gc_evicted persists index when needed
        }
        if index_dirty {
            self.persist_spill_index();
        }
        self.stats.spill_maint_passes += 1;
        self.stats.spill_maint_evictions += evicted as u64;
        evicted
    }

    pub fn spilled_start_count(&self) -> usize {
        self.spilled_starts.len()
    }

    pub fn record_step_deferred_pool_pressure(&mut self) {
        self.stats.step_deferred_pool_pressure += 1;
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    #[cfg(any(test, standalone_blocks))]
    pub fn retained_start_count(&self) -> usize {
        self.starts.len()
    }

    fn inc_start_slot_ref(&mut self, kind: BlockKind, slot: Slot) {
        let idx = slot.pool_index().get() as usize;
        let v = &mut self.start_slot_refs[kind_idx(kind)];
        if v.len() <= idx {
            v.resize(idx + 1, 0);
        }
        v[idx] = v[idx].checked_add(1).expect("start_slot_refs overflow");
    }

    fn dec_start_slot_ref(&mut self, kind: BlockKind, slot: Slot) {
        let idx = slot.pool_index().get() as usize;
        let v = &mut self.start_slot_refs[kind_idx(kind)];
        debug_assert!(idx < v.len() && v[idx] > 0, "start_slot_refs underflow");
        if idx < v.len() && v[idx] > 0 {
            v[idx] -= 1;
        }
    }

    /// Number of slots in `kind`'s pool that are currently free (`refs == 0`) OR held only by
    /// evictable `RetainedStart` entries (no active lane references them).
    pub fn free_plus_evictable_slots(&self, pools: &BlockPools, kind: BlockKind) -> usize {
        self.pool_headroom(pools, kind).1
    }

    // (non-retired capacity, slots reclaimable before any active lane is released).
    fn pool_headroom(&self, pools: &BlockPools, kind: BlockKind) -> (usize, usize) {
        pools
            .of(kind)
            .count_usable_and_avail(&self.start_slot_refs[kind_idx(kind)])
    }

    fn resident_start(&self, pools: &BlockPools, namespace: u64, toks: &[u32]) -> Option<&RetainedStart> {
        if self.starts.is_empty() {
            return None;
        }
        let prefix = toks.get(..toks.len().checked_sub(1)?)?;
        let matched = self.tree.deepest_start_in_namespace(namespace, prefix);
        let st = self.starts.get(&matched.start?)?;
        (st.namespace == namespace
            && st.entries.iter().all(|&(kind, _, e)| entry_is_live_sealed(pools, kind, e)))
        .then_some(st)
    }

    /// Length of the deepest live in-memory checkpoint strictly shorter than the prompt in default namespace `0`.
    #[cfg(any(test, standalone_blocks))]
    pub fn peek_match_len(&self, pools: &BlockPools, toks: &[u32]) -> usize {
        self.peek_match_len_in_namespace(pools, 0, toks)
    }

    /// Length of the deepest live in-memory checkpoint strictly shorter than `toks` in `namespace`.
    #[cfg(any(test, standalone_blocks))]
    pub fn peek_match_len_in_namespace(&self, pools: &BlockPools, namespace: u64, toks: &[u32]) -> usize {
        self.resident_start(pools, namespace, toks).map_or(0, |st| st.pos)
    }

    /// Check the marginal cost of binding the prospective lane in `namespace` and completing the
    /// bounded outstanding work using a prebuilt [`PreparedInflightCohort`]
    /// (`O(prefilling_lanes * log(blocks))`, zero all-pairs rescans across a burst).
    #[allow(clippy::too_many_arguments)]
    pub fn can_admit_with_cohort_in_namespace(
        &mut self,
        pools: &BlockPools,
        cohort: &PreparedInflightCohort<'_>,
        lane: usize,
        namespace: u64,
        toks: &[u32],
        cand_block_hashes: Option<&[u64]>,
        max_gen: usize,
        chunk: usize,
    ) -> bool {
        // The handler and scheduler backstop enforce each single-lane physical context bound.
        // With no competing jobs, retained-only slots are reclaimable on demand.
        if cohort.is_empty() {
            return true;
        }

        let owned_hashes;
        let cand_hashes: &[u64] = match cand_block_hashes {
            Some(h) => h,
            None => {
                owned_hashes = compute_block_hashes(toks);
                &owned_hashes
            }
        };

        let (usable_window, available_window, available_full) = match cohort.cached_headroom.get() {
            Some(h) => h,
            None => {
                let (uw, aw) = self.pool_headroom(pools, BlockKind::Window);
                let af = self.free_plus_evictable_slots(pools, BlockKind::Full);
                cohort.cached_headroom.set(Some((uw, aw, af)));
                (uw, aw, af)
            }
        };
        let mut pins = [0; 2];
        let mut cached = 0;
        if let Some(st) = self.resident_start(pools, namespace, toks) {
            cached = st.pos;
            for &(kind, _, entry) in &st.entries {
                let i = kind_idx(kind);
                let slot = entry.slot();
                let refs = self.start_slot_refs[i][slot.pool_index().get() as usize];
                if pools.of(kind).record(slot).unwrap().refs() == refs {
                    pins[i] += 1;
                }
            }
        }

        let fits = |start: usize, newly_pinned: [usize; 2]| {
            let cand = InflightLaneReserve {
                lane,
                namespace,
                toks,
                block_hashes: Some(cand_hashes),
                n_past: start,
                pf_end: toks.len(),
                cached: start,
                written: start,
                gen: 0,
                max_gen,
                runnable: true,
            };
            let (base_full, base_win_growth, steady_window) =
                cohort.eval_with_candidate(&cand, cand_hashes);
            let full = base_full + newly_pinned[kind_idx(BlockKind::Full)];
            let window_growth = base_win_growth + newly_pinned[kind_idx(BlockKind::Window)];
            // Two independent upper bounds: all remaining writes with no pruning, or the
            // whole cohort's rolling windows plus one shared step budget and per-span rounding.
            let peak_window = steady_window + chunk.div_ceil(BLOCK) + (cohort.len + 1);
            let window_ok = window_growth <= available_window || peak_window <= usable_window;
            full <= available_full && window_ok
        };

        let mut ok = fits(cached, pins);
        // A deeper SSD restore can replace the resident start during bind. Budget that outcome
        // too, including the newly pinned restored blocks, without doing disk I/O on a refused job.
        if ok && !toks.is_empty() && !self.spilled_starts.is_empty() {
            let disk = self.spilled_tree.deepest_start_in_namespace(namespace, &toks[..toks.len() - 1]).at;
            if disk > cached {
                let blocks = disk / BLOCK;
                let already_pinned_full = cached / BLOCK - pins[kind_idx(BlockKind::Full)];
                ok = fits(disk, [blocks.min(WINDOW / BLOCK), blocks - already_pinned_full]);
            }
        }
        if !ok {
            self.stats.admit_deferred_pool_pressure += 1;
        }
        ok
    }

    #[cfg(any(test, standalone_blocks))]
    #[allow(clippy::too_many_arguments)]
    pub fn can_admit_with_cohort(
        &mut self,
        pools: &BlockPools,
        cohort: &PreparedInflightCohort<'_>,
        lane: usize,
        toks: &[u32],
        cand_block_hashes: Option<&[u64]>,
        max_gen: usize,
        chunk: usize,
    ) -> bool {
        self.can_admit_with_cohort_in_namespace(pools, cohort, lane, 0, toks, cand_block_hashes, max_gen, chunk)
    }

    /// Check the marginal cost of binding the prospective lane in `namespace` and completing the bounded
    /// outstanding work. Retained-only slots stop being evictable as soon as this job binds.
    #[cfg(any(test, standalone_blocks))]
    #[allow(clippy::too_many_arguments)]
    pub fn can_admit_in_namespace(
        &mut self,
        pools: &BlockPools,
        active: &[InflightLaneReserve<'_>],
        lane: usize,
        namespace: u64,
        toks: &[u32],
        max_gen: usize,
        chunk: usize,
    ) -> bool {
        if active.is_empty() {
            return true;
        }
        let cohort = PreparedInflightCohort::new(active);
        self.can_admit_with_cohort_in_namespace(pools, &cohort, lane, namespace, toks, None, max_gen, chunk)
    }

    /// Check the marginal cost of binding the prospective lane and completing the bounded
    /// outstanding work in default namespace `0`.
    #[cfg(any(test, standalone_blocks))]
    pub fn can_admit(
        &mut self,
        pools: &BlockPools,
        active: &[InflightLaneReserve<'_>],
        lane: usize,
        toks: &[u32],
        max_gen: usize,
        chunk: usize,
    ) -> bool {
        self.can_admit_in_namespace(pools, active, lane, 0, toks, max_gen, chunk)
    }

    /// Rebind an untouched prefilling lane (`state.written == n_past` and no unsealed blocks)
    /// when `EntryTree` now holds a deeper live `RetainedStart` than `n_past` in `state.namespace`.
    /// Validates and stages the deeper start point BEFORE releasing the lane's prior bindings
    /// so a failed or stale candidate leaves `tbl` and `state` completely intact and consistent.
    pub fn rebind_if_deeper_match(
        &mut self,
        pools: &mut BlockPools,
        tbl: &mut BlockTable,
        state: &mut LaneBlockState,
        toks: &[u32],
        n_past: usize,
    ) -> Option<(usize, Vec<u8>)> {
        if state.written != n_past || state.written != state.sealed.len() * BLOCK || toks.len() <= 1 {
            return None;
        }
        let namespace = state.namespace;
        let search = &toks[..toks.len() - 1];
        let mut matched = self.tree.deepest_start_in_namespace(namespace, search);
        loop {
            if matched.at <= n_past {
                return None;
            }
            let sid = matched.start?;
            let all_live = match self.starts.get(&sid) {
                Some(st) => {
                    debug_assert_eq!(st.pos, matched.at);
                    st.namespace == namespace
                        && st.entries.iter().all(|&(kind, _, e)| entry_is_live_sealed(pools, kind, e))
                }
                None => false,
            };
            if all_live {
                break;
            }
            self.evict_start_inner(pools, sid, false, None);
            matched = self.tree.deepest_start_in_namespace(namespace, search);
        }
        let sid = matched.start.expect("validated live start id");
        let (cs, entries) = {
            let st = self.starts.get(&sid).expect("validated live start exists");
            (st.cs.clone(), st.entries.clone())
        };
        // Stage all new block-table bindings and EntryTree pins BEFORE releasing the lane's
        // prior bindings so any failure (such as table-capacity bounds or ref overflow) leaves
        // `tbl` and `state` completely untouched.
        let Ok(mut staged) = BlockTable::new(tbl.blocks_per_kind()) else {
            return None;
        };
        for &(kind, bi, e) in &entries {
            if staged.fill(pools.of_mut(kind), bi, e.slot(), e.generation()).is_err() {
                let _ = staged.clear_entries(pools);
                return None;
            }
        }
        let mut pinned_ids: Vec<SealedId> = Vec::with_capacity(matched.ids.len());
        for &id in &matched.ids {
            if self.tree.pin(id).is_err() {
                for pid in pinned_ids {
                    self.tree.unpin(pid);
                }
                let _ = staged.clear_entries(pools);
                return None;
            }
            pinned_ids.push(id);
        }
        staged.mark_full_sync();

        // Commit: release old bindings only now that the deeper target is 100% staged and pinned.
        tbl.clear_entries(pools).expect("clear lane block table for rebind");
        let old_tip = state.sealed.last().copied();
        for id in state.sealed.drain(..) {
            self.tree.unpin(id);
        }
        if let Some(old_tip) = old_tip {
            if !pinned_ids.contains(&old_tip) {
                self.tree.prune_unreferenced(old_tip);
            }
        }
        self.tick += 1;
        let now = self.tick;
        for &anc in &pinned_ids {
            if let Some(st) = self.starts.get_mut(&anc) {
                st.last_used_tick = now;
            }
        }
        *tbl = staged;
        state.prompt_len = toks.len();
        state.sealed = pinned_ids;
        state.written = matched.at;
        self.stats.start_matches += 1;
        self.stats.cached_tokens_total += matched.at as u64;
        self.stats.singleflight_rebinds += 1;
        self.stats.singleflight_saved_tokens += (matched.at - n_past) as u64;
        Some((matched.at, cs))
    }

    /// Bind `state` in `namespace` to the deepest live in-RAM or spilled SSD start point strictly
    /// shorter than `toks`.
    #[cfg(any(test, standalone_blocks))]
    pub fn match_and_bind_in_namespace(
        &mut self,
        pools: &mut BlockPools,
        tbl: &mut BlockTable,
        state: &mut LaneBlockState,
        namespace: u64,
        toks: &[u32],
    ) -> Option<(usize, Vec<u8>)> {
        state.namespace = namespace; // MUTANT_ANCHOR_BIND_NAMESPACE
        self.match_and_bind(pools, tbl, state, toks)
    }

    /// Look up the deepest live `RetainedStart` (or spilled SSD start point when deeper than any
    /// in-RAM match) strictly shorter than `toks` in `state.namespace`, bind its `Full` and `Window`
    /// entries into `tbl`, pin its `EntryTree` path on `state`, touch its LRU tick, and return
    /// `Some((matched_at, cs_bytes))`.
    pub fn match_and_bind(
        &mut self,
        pools: &mut BlockPools,
        tbl: &mut BlockTable,
        state: &mut LaneBlockState,
        toks: &[u32],
    ) -> Option<(usize, Vec<u8>)> {
        debug_assert!(state.sealed.is_empty(), "lane must be released before match_and_bind");
        let namespace = state.namespace;
        state.prompt_len = toks.len();
        state.written = 0;
        if toks.len() <= 1 {
            return None;
        }
        let search = &toks[..toks.len() - 1];
        let mut in_ram = self.tree.deepest_start_in_namespace(namespace, search); // MUTANT_ANCHOR_MATCH_IN_RAM
        if in_ram.at > 0 {
            if let Some(sid) = in_ram.start {
                let all_live = match self.starts.get(&sid) {
                    Some(st) => {
                        debug_assert_eq!(st.pos, in_ram.at);
                        st.namespace == namespace
                            && st.entries.iter().all(|&(kind, _, e)| entry_is_live_sealed(pools, kind, e))
                    }
                    None => false,
                };
                if !all_live {
                    self.evict_start_inner(pools, sid, false, None);
                    in_ram = self.tree.deepest_start_in_namespace(namespace, search);
                }
            } else {
                in_ram.at = 0;
            }
        }
        let mut restored_from_disk = false;
        if self.spill_store.is_some() && !self.spilled_starts.is_empty() {
            let on_disk = self.spilled_tree.deepest_start_in_namespace(namespace, search);
            if on_disk.at > in_ram.at {
                if let Some(&spill_id) = on_disk.blocks.last() {
                    restored_from_disk =
                        self.try_restore_spilled_start(pools, namespace, toks, spill_id, &in_ram).is_some();
                }
            }
        }
        let matched = if restored_from_disk {
            self.tree.deepest_start_in_namespace(namespace, search)
        } else {
            in_ram
        };
        self.bind_matched_resident_start(pools, tbl, state, toks.len(), matched)
    }

    fn bind_resident_start(
        &mut self,
        pools: &mut BlockPools,
        tbl: &mut BlockTable,
        state: &mut LaneBlockState,
        toks: &[u32],
    ) -> Option<(usize, Vec<u8>)> {
        state.prompt_len = toks.len();
        state.written = 0;
        if toks.len() <= 1 {
            return None;
        }
        let matched = self.tree.deepest_start_in_namespace(state.namespace, &toks[..toks.len() - 1]);
        self.bind_matched_resident_start(pools, tbl, state, toks.len(), matched)
    }

    fn bind_matched_resident_start(
        &mut self,
        pools: &mut BlockPools,
        tbl: &mut BlockTable,
        state: &mut LaneBlockState,
        prompt_len: usize,
        matched: crate::prefix::StartMatch<BlockHandles>,
    ) -> Option<(usize, Vec<u8>)> {
        let namespace = state.namespace;
        state.prompt_len = prompt_len;
        if matched.at == 0 {
            return None;
        }
        let sid = matched.start?;
        let st = self.starts.get(&sid)?;
        debug_assert_eq!(st.pos, matched.at);
        let all_live = st.namespace == namespace
            && st.entries.iter().all(|&(kind, _, e)| entry_is_live_sealed(pools, kind, e));
        if !all_live {
            self.evict_start_inner(pools, sid, false, None);
            return None;
        }
        self.tick += 1;
        let now = self.tick;
        // Touch the matched start point and any ancestor start points along the hot prefix.
        for &anc in &matched.ids {
            if let Some(st) = self.starts.get_mut(&anc) {
                st.last_used_tick = now;
            }
        }
        let (cs, entries) = {
            let st = self.starts.get(&sid).expect("checked start exists");
            (st.cs.clone(), st.entries.clone())
        };
        for &(kind, bi, e) in &entries {
            tbl.fill(pools.of_mut(kind), bi, e.slot(), e.generation())
                .expect("install start-point block");
        }
        for &id in &matched.ids {
            self.tree.pin(id).expect("pin matched prefix block");
        }
        state.sealed = matched.ids;
        state.written = matched.at;
        self.stats.start_matches += 1;
        self.stats.cached_tokens_total += matched.at as u64;
        Some((matched.at, cs))
    }

    fn alloc_spill_key(&mut self, kind: BlockKind) -> SpillKey {
        self.spill_store.as_mut().expect("spill keys require an open store").fresh_key(to_spill_kind(kind))
    }

    fn remove_spilled_start(&mut self, spill_id: u64, delete_from_disk: bool) {
        if self.remove_spilled_start_inner(spill_id, delete_from_disk) {
            self.persist_spill_index();
        }
    }

    fn remove_spilled_start_inner(&mut self, spill_id: u64, delete_from_disk: bool) -> bool {
        let Some(sp) = self.spilled_starts.remove(&spill_id) else {
            return false;
        };
        if let Some(tip) = self.spilled_tips.remove(&spill_id) {
            let _ = self.spilled_tree.release_start(tip);
            self.spilled_tree.prune_unreferenced(tip);
        }
        for sk in std::iter::once(sp.cs_key).chain(sp.entries.iter().map(|&(_, _, k)| k)) {
            let remove_key = if let Some(owners) = self.spill_key_owners.get_mut(&sk) {
                owners.retain(|&id| id != spill_id);
                owners.is_empty()
            } else {
                false
            };
            if remove_key {
                self.spill_key_owners.remove(&sk);
                if let Some(entry_key) = self.spill_key_to_entry.remove(&sk) {
                    self.entry_spill_keys.remove(&entry_key);
                }
                if delete_from_disk && !self.in_flight_restore_keys.contains_key(&sk) {
                    if let Some(store) = self.spill_store.as_mut() {
                        store.remove_block(sk);
                    }
                }
            }
        }
        true
    }

    fn handle_disk_gc_evicted(&mut self, evicted: &[SpillKey]) {
        let mut dirty = false;
        for &gk in evicted {
            if let Some(entry_key) = self.spill_key_to_entry.remove(&gk) {
                self.entry_spill_keys.remove(&entry_key);
            }
            if let Some(owners) = self.spill_key_owners.remove(&gk) {
                for oid in owners {
                    if self.remove_spilled_start_inner(oid, true) {
                        dirty = true;
                    }
                }
            }
        }
        if dirty {
            self.persist_spill_index();
        }
    }

    fn claim_spill_key(&mut self, spill_id: u64, sk: SpillKey, claimed: &mut Vec<SpillKey>) {
        let owners = self.spill_key_owners.entry(sk).or_default();
        if !owners.contains(&spill_id) {
            owners.push(spill_id);
            claimed.push(sk);
        }
    }

    fn abort_spill_claims(&mut self, spill_id: u64, claimed: &[SpillKey]) {
        for &sk in claimed {
            let remove_key = if let Some(owners) = self.spill_key_owners.get_mut(&sk) {
                owners.retain(|&id| id != spill_id);
                owners.is_empty()
            } else {
                false
            };
            if remove_key {
                self.spill_key_owners.remove(&sk);
                if let Some(entry_key) = self.spill_key_to_entry.remove(&sk) {
                    self.entry_spill_keys.remove(&entry_key);
                }
                if let Some(store) = self.spill_store.as_mut() {
                    store.remove_block(sk);
                }
            }
        }
    }

    fn spill_retained_start(&mut self, pools: &BlockPools, st: &RetainedStart, restore: Option<&RestoreSource>) {
        if self.spill_store.is_none()
            || self.max_spilled_starts == 0
            || st.pos == 0
            || !st.pos.is_multiple_of(BLOCK)
            || st.toks.len() != st.pos
            || st.entries.is_empty()
        {
            return;
        }
        if self.spill_write_timeout.is_zero() {
            self.stats.spill_write_timeouts += 1;
            return;
        }
        let deadline = Instant::now() + self.spill_write_timeout;
        for &(kind, _, e) in &st.entries {
            let pool = pools.of(kind);
            if !entry_is_live_sealed(pools, kind, e) || pool.slot_raw_region(e.slot()).is_none() {
                return;
            }
        }

        let spill_id = self.next_spill_id;
        self.next_spill_id = self.next_spill_id.wrapping_add(1).max(1);

        // Claim ownership of all reusable on-disk blocks BEFORE trimming `max_spilled_starts`,
        // writing new blocks, or replacing an existing tip, so concurrent/interleaved eviction
        // can never delete a `.blk` file that `spill_id` is reusing.
        let mut protected = self.protected_spill_keys(restore);
        let mut claimed_keys: Vec<SpillKey> = Vec::with_capacity(st.entries.len() + 1);
        let mut reusable_keys: HashSet<SpillKey> = HashSet::new();
        for &(kind, _, e) in &st.entries {
            let ek = (kind_idx(kind) as u8, e.slot().get(), e.generation());
            if let Some(&sk) = self.entry_spill_keys.get(&ek) {
                if self.spill_store.as_ref().is_some_and(|store| store.contains(sk)) {
                    protected.insert(sk);
                    reusable_keys.insert(sk);
                    self.claim_spill_key(spill_id, sk, &mut claimed_keys);
                }
            }
        }

        while self.spilled_starts.len() >= self.max_spilled_starts && self.max_spilled_starts > 0 {
            let oldest = self
                .spilled_starts
                .iter()
                .filter(|(id, _)| {
                    !restore.is_some_and(|source| source.spill_id == **id)
                        && !self.in_flight_restore_spills.contains(id)
                })
                .min_by_key(|(_, sp)| (sp.last_used_tick, std::cmp::Reverse(sp.pos)))
                .map(|(&id, _)| id);
            let Some(oid) = oldest else {
                self.abort_spill_claims(spill_id, &claimed_keys);
                return;
            };
            self.remove_spilled_start(oid, true);
        }

        let mut spilled_entries: Vec<(BlockKind, BlockIndex, SpillKey)> =
            Vec::with_capacity(st.entries.len());

        for &(kind, bi, e) in &st.entries {
            let ek = (kind_idx(kind) as u8, e.slot().get(), e.generation());
            let (sk, is_new_key) = if let Some(&existing) = self.entry_spill_keys.get(&ek) {
                (existing, false)
            } else {
                (self.alloc_spill_key(kind), true)
            };
            if !is_new_key && reusable_keys.contains(&sk) && self.spill_store.as_ref().is_some_and(|store| store.contains(sk)) {
                spilled_entries.push((kind, bi, sk));
                continue;
            }
            if Instant::now() >= deadline {
                self.stats.spill_write_timeouts += 1;
                self.abort_spill_claims(spill_id, &claimed_keys);
                return;
            }
            let kv_bytes = match pools.of(kind).slot_bytes(e.slot()) {
                Ok(b) => b,
                Err(_) => {
                    self.abort_spill_claims(spill_id, &claimed_keys);
                    return;
                }
            };
            let write_res = self
                .spill_store
                .as_mut()
                .expect("checked spill_store")
                .write_block(sk, u64::from(e.generation()), kv_bytes, None, &protected);
            match write_res {
                Ok((WriteOutcome::Written { .. }, gc_evicted)) => {
                    self.handle_disk_gc_evicted(&gc_evicted);
                    if is_new_key {
                        self.entry_spill_keys.insert(ek, sk);
                        self.spill_key_to_entry.insert(sk, ek);
                    }
                    protected.insert(sk);
                    self.claim_spill_key(spill_id, sk, &mut claimed_keys);
                    spilled_entries.push((kind, bi, sk));
                }
                Ok((WriteOutcome::OverBudget, gc_evicted)) => {
                    self.handle_disk_gc_evicted(&gc_evicted);
                    self.abort_spill_claims(spill_id, &claimed_keys);
                    return;
                }
                Err(_) => {
                    self.abort_spill_claims(spill_id, &claimed_keys);
                    return;
                }
            }
        }

        if Instant::now() >= deadline {
            self.stats.spill_write_timeouts += 1;
            self.abort_spill_claims(spill_id, &claimed_keys);
            return;
        }
        let cs_key = self.alloc_spill_key(BlockKind::Full);
        let cs_res = self
            .spill_store
            .as_mut()
            .expect("checked spill_store")
            .write_block(cs_key, 1, &[], Some(&st.cs), &protected);
        match cs_res {
            Ok((WriteOutcome::Written { .. }, gc_evicted)) => {
                self.handle_disk_gc_evicted(&gc_evicted);
                protected.insert(cs_key);
                self.claim_spill_key(spill_id, cs_key, &mut claimed_keys);
            }
            Ok((WriteOutcome::OverBudget, gc_evicted)) => {
                self.handle_disk_gc_evicted(&gc_evicted);
                self.abort_spill_claims(spill_id, &claimed_keys);
                return;
            }
            Err(_) => {
                self.abort_spill_claims(spill_id, &claimed_keys);
                return;
            }
        }

        let n_blocks = st.pos / BLOCK;
        let mut parent = None;
        for b in 0..n_blocks {
            let id = self
                .spilled_tree
                .insert_in_namespace(st.namespace, parent, b * BLOCK, &st.toks[b * BLOCK..(b + 1) * BLOCK], spill_id)
                .expect("insert spilled block into spilled_tree");
            parent = Some(id);
        }
        let tip = parent.expect("non-empty spilled blocks");
        if self.spilled_tree.start_holders(tip) > 0 {
            if let Some(old_id) = self.spilled_tree.handle(tip) {
                let _ = self.remove_spilled_start_inner(old_id, true);
            }
            // Re-insert path if pruned above so tip holds the new spill_id.
            let mut p = None;
            for b in 0..n_blocks {
                let id = self
                    .spilled_tree
                    .insert_in_namespace(st.namespace, p, b * BLOCK, &st.toks[b * BLOCK..(b + 1) * BLOCK], spill_id)
                    .expect("re-insert spilled block into spilled_tree");
                p = Some(id);
            }
            let new_tip = p.expect("non-empty spilled blocks");
            let _ = self.spilled_tree.set_handle(new_tip, spill_id);
            self.spilled_tree
                .retain_start(new_tip)
                .expect("retain spilled start");
            self.spilled_tips.insert(spill_id, new_tip);
        } else {
            let _ = self.spilled_tree.set_handle(tip, spill_id);
            self.spilled_tree
                .retain_start(tip)
                .expect("retain spilled start");
            self.spilled_tips.insert(spill_id, tip);
        }

        self.spilled_starts.insert(
            spill_id,
            SpilledStart {
                namespace: st.namespace,
                pos: st.pos,
                reasons: st.reasons,
                cs_key,
                entries: spilled_entries,
                toks: st.toks.clone(),
                last_used_tick: st.last_used_tick,
            },
        );
        self.stats.starts_spilled += 1;
        self.persist_spill_index();
    }

    fn try_restore_spilled_start(
        &mut self,
        pools: &mut BlockPools,
        namespace: u64,
        toks: &[u32],
        spill_id: u64,
        in_ram: &crate::prefix::StartMatch<BlockHandles>,
    ) -> Option<SealedId> {
        let (sp_ns, pos, reasons, cs_key, entries) = {
            let sp = self.spilled_starts.get(&spill_id)?;
            (sp.namespace, sp.pos, sp.reasons, sp.cs_key, sp.entries.clone())
        };
        if sp_ns != namespace || pos == 0 || !pos.is_multiple_of(BLOCK) || pos >= toks.len() {
            if pos == 0 || !pos.is_multiple_of(BLOCK) {
                self.remove_spilled_start(spill_id, true);
            }
            return None;
        }

        let mut reused_full: Vec<(BlockIndex, BlockEntry)> = Vec::new();
        let mut disk_entries: Vec<(BlockKind, BlockIndex, SpillKey)> = Vec::new();
        for &(kind, bi, sk) in &entries {
            if kind == BlockKind::Full {
                let b_idx = bi.get() as usize;
                if b_idx < in_ram.blocks.len() {
                    let fe = in_ram.blocks[b_idx].full();
                    if entry_is_live_sealed(pools, BlockKind::Full, fe) {
                        reused_full.push((bi, fe));
                        continue;
                    }
                }
            }
            disk_entries.push((kind, bi, sk));
        }

        let need_full = disk_entries
            .iter()
            .filter(|&&(k, _, _)| k == BlockKind::Full)
            .count();
        let need_win = disk_entries
            .iter()
            .filter(|&&(k, _, _)| k == BlockKind::Window)
            .count();

        if (need_full > 0 && pools.full.slab_count() == 0)
            || (need_win > 0 && pools.window.slab_count() == 0)
        {
            return None;
        }

        let full_srefs = &self.start_slot_refs[kind_idx(BlockKind::Full)];
        let reused_evictable = reused_full
            .iter()
            .filter(|&&(_, fe)| {
                let idx = fe.slot().pool_index().get() as usize;
                let sref = full_srefs.get(idx).copied().unwrap_or(0);
                pools
                    .full
                    .record(fe.slot())
                    .ok()
                    .is_some_and(|r| r.refs() <= sref)
            })
            .count();

        let avail_full = self
            .free_plus_evictable_slots(pools, BlockKind::Full)
            .saturating_sub(reused_evictable);
        let avail_win = self.free_plus_evictable_slots(pools, BlockKind::Window);
        if need_full > avail_full || need_win > avail_win {
            return None;
        }

        // Recurrent state is needed by every restore and has a separate bounded spill record.
        let restore_began = Instant::now();
        let cs = match self.spill_store.as_mut()?.read_block(cs_key, Some(0)) {
            ReadOutcome::Hit(blk) => match blk.recurrent_state {
                Some(cs) if !cs.is_empty() => cs,
                _ => {
                    self.stats.corrupt_fallbacks += 1;
                    self.remove_spilled_start(spill_id, true);
                    return None;
                }
            },
            ReadOutcome::Miss(_) => {
                self.stats.corrupt_fallbacks += 1;
                self.remove_spilled_start(spill_id, true);
                return None;
            }
        };

        let source = RestoreSource {
            spill_id,
            keys: entries.iter().map(|&(_, _, key)| key).chain(std::iter::once(cs_key)).collect(),
            resident_start: in_ram.start,
        };
        // Reused blocks and newly loaded blocks stay private to the restore until the
        // complete checkpoint is verified. This list owns one temporary ref per entry.
        let mut held = Vec::with_capacity(entries.len());
        for (bi, fe) in reused_full {
            pools.full.record_mut(fe.slot()).unwrap().add_ref().expect("pin reused full block");
            held.push((BlockKind::Full, bi, fe));
        }
        if self.stage_restore_blocks(pools, &source, &disk_entries, &mut held).is_none() {
            Self::release_restore_refs(pools, held);
            return None;
        }
        let elapsed_us = restore_began.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        Some(self.publish_restored_checkpoint(
            pools,
            spill_id,
            namespace,
            pos,
            reasons,
            cs,
            &toks[..pos],
            held,
            elapsed_us,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn publish_restored_checkpoint(
        &mut self,
        pools: &mut BlockPools,
        spill_id: u64,
        namespace: u64,
        pos: usize,
        reasons: u8,
        cs: Vec<u8>,
        prefix_toks: &[u32],
        held: Vec<(BlockKind, BlockIndex, BlockEntry)>,
        elapsed_us: u64,
    ) -> SealedId {
        // Publish the verified checkpoint, transferring its provisional refs to the retained start.
        let mut full_by_bi: HashMap<u32, BlockEntry> = HashMap::with_capacity(pos / BLOCK);
        let mut win_by_bi: HashMap<u32, BlockEntry> = HashMap::with_capacity(WINDOW / BLOCK);
        for &(kind, bi, e) in &held {
            match kind {
                BlockKind::Full => {
                    full_by_bi.insert(bi.get(), e);
                }
                BlockKind::Window => {
                    win_by_bi.insert(bi.get(), e);
                }
            }
        }

        let n_blocks = pos / BLOCK;
        let win_lo = pos.saturating_sub(WINDOW) / BLOCK;
        let mut parent = None;
        let mut final_entries = Vec::with_capacity(n_blocks + WINDOW / BLOCK);
        for b in 0..n_blocks {
            let bi = BlockIndex::new(b as u32);
            let full_e = *full_by_bi.get(&(b as u32)).expect("restored full entry");
            let win_e_opt = win_by_bi.get(&(b as u32)).copied();
            let handles = BlockHandles::new(full_e, win_e_opt);
            let id = self
                .tree
                .insert_in_namespace(namespace, parent, b * BLOCK, &prefix_toks[b * BLOCK..(b + 1) * BLOCK], handles)
                .expect("insert restored block into EntryTree");
            let canon = self.tree.handle(id).expect("inserted handle");
            let canon_f = canon.full();
            let best_f = if canon_f != full_e && entry_is_live_sealed(pools, BlockKind::Full, canon_f) {
                canon_f
            } else {
                full_e
            };
            let best_w = match (win_e_opt, canon.window()) {
                (Some(we), Some(cw)) if cw != we && entry_is_live_sealed(pools, BlockKind::Window, cw) => Some(cw),
                (Some(we), _) => Some(we),
                (None, Some(cw)) if entry_is_live_sealed(pools, BlockKind::Window, cw) => Some(cw),
                (None, _) => None,
            };
            if best_f != canon_f || best_w != canon.window() {
                let _ = self.tree.set_handle(id, BlockHandles::new(best_f, best_w));
            }
            final_entries.push((BlockKind::Full, bi, best_f));
            if b >= win_lo {
                if let Some(we) = best_w {
                    final_entries.push((BlockKind::Window, bi, we));
                }
            }
            parent = Some(id);
        }

        let tip = parent.expect("non-empty restored start");
        self.tick += 1;
        let now = self.tick;
        if let Some(existing) = self.starts.get_mut(&tip) {
            existing.reasons |= reasons;
            existing.last_used_tick = now;
        } else {
            for &(kind, _, e) in &final_entries {
                pools
                    .of_mut(kind)
                    .record_mut(e.slot())
                    .unwrap()
                    .add_ref()
                    .expect("retain restored start point ref");
                self.inc_start_slot_ref(kind, e.slot());
            }
            self.tree.retain_start(tip).expect("retain restored start in EntryTree");
            let st = RetainedStart {
                namespace,
                pos,
                cs,
                reasons,
                entries: final_entries,
                toks: prefix_toks.to_vec(),
                last_used_tick: now,
            };
            self.retained_host_bytes = self.retained_host_bytes.saturating_add(st.host_bytes());
            self.starts.insert(tip, st);
            self.enforce_retained_host_budget(pools, Some(tip));
        }

        Self::release_restore_refs(pools, held); // MUTANT_ANCHOR_RELEASE_RESTORE_REFS
        self.remove_spilled_start(spill_id, true);
        self.stats.starts_restored += 1;
        self.stats.last_restore_us = elapsed_us;
        self.stats.restore_us_total = self.stats.restore_us_total.saturating_add(elapsed_us);
        tip
    }

    pub fn begin_admit_bind_in_namespace(
        &mut self,
        pools: &mut BlockPools,
        lane: usize,
        tbl: &mut BlockTable,
        state: &mut LaneBlockState,
        namespace: u64,
        toks: &[u32],
    ) -> AdmitBindOutcome {
        state.namespace = namespace;
        self.begin_admit_bind(pools, lane, tbl, state, toks)
    }

    /// Begin binding an admitted lane on the scheduler thread without blocking on SSD reads:
    /// - If no deeper SSD checkpoint exists (or pool headroom cannot fit the restore), binds
    ///   any resident RAM checkpoint immediately and returns [`AdmitBindOutcome::Immediate`].
    /// - If a deeper SSD checkpoint exists and fits, atomically pre-reserves all destination
    ///   `(Full, Window)` `BlockPool` slots (`t-l4-restore`), pins source `.blk` keys against
    ///   disk GC, dispatches `pread` + SHA-256 verification to the background restore worker,
    ///   and returns [`AdmitBindOutcome::Restoring`].
    pub fn begin_admit_bind(
        &mut self,
        pools: &mut BlockPools,
        lane: usize,
        tbl: &mut BlockTable,
        state: &mut LaneBlockState,
        toks: &[u32],
    ) -> AdmitBindOutcome {
        debug_assert!(state.sealed.is_empty(), "lane must be released before begin_admit_bind");
        debug_assert!(!self.pending_restores.contains_key(&lane), "lane already restoring");
        let namespace = state.namespace;
        state.prompt_len = toks.len();
        state.written = 0;
        if toks.len() <= 1 {
            return AdmitBindOutcome::Immediate(None);
        }
        let search = &toks[..toks.len() - 1];
        let mut in_ram = self.tree.deepest_start_in_namespace(namespace, search);
        if in_ram.at > 0 {
            if let Some(sid) = in_ram.start {
                let all_live = match self.starts.get(&sid) {
                    Some(st) => {
                        debug_assert_eq!(st.pos, in_ram.at);
                        st.namespace == namespace
                            && st.entries.iter().all(|&(kind, _, e)| entry_is_live_sealed(pools, kind, e))
                    }
                    None => false,
                };
                if !all_live {
                    self.evict_start_inner(pools, sid, false, None);
                    in_ram = self.tree.deepest_start_in_namespace(namespace, search);
                }
            } else {
                in_ram.at = 0;
            }
        }

        if self.spill_store.is_none() || self.spilled_starts.is_empty() {
            return AdmitBindOutcome::Immediate(self.bind_resident_start(pools, tbl, state, toks));
        }

        let on_disk = self.spilled_tree.deepest_start_in_namespace(namespace, search);
        if on_disk.at <= in_ram.at {
            return AdmitBindOutcome::Immediate(self.bind_resident_start(pools, tbl, state, toks));
        }
        let Some(&spill_id) = on_disk.blocks.last() else {
            return AdmitBindOutcome::Immediate(self.bind_resident_start(pools, tbl, state, toks));
        };
        if self.in_flight_restore_spills.contains(&spill_id) {
            // Another lane is already restoring this exact spilled start; let singleflight /
            // rebind pick it up once published in RAM rather than double-reading from SSD.
            return AdmitBindOutcome::Immediate(self.bind_resident_start(pools, tbl, state, toks));
        }
        let Some((sp_ns, pos, reasons, cs_key, entries)) = self
            .spilled_starts
            .get(&spill_id)
            .map(|sp| (sp.namespace, sp.pos, sp.reasons, sp.cs_key, sp.entries.clone()))
        else {
            return AdmitBindOutcome::Immediate(self.bind_resident_start(pools, tbl, state, toks));
        };
        if sp_ns != namespace || pos == 0 || !pos.is_multiple_of(BLOCK) || pos >= toks.len() {
            if pos == 0 || !pos.is_multiple_of(BLOCK) {
                self.remove_spilled_start(spill_id, true);
            }
            return AdmitBindOutcome::Immediate(self.bind_resident_start(pools, tbl, state, toks));
        }

        let mut reused_full: Vec<(BlockIndex, BlockEntry)> = Vec::new();
        let mut disk_entries: Vec<(BlockKind, BlockIndex, SpillKey)> = Vec::new();
        for &(kind, bi, sk) in &entries {
            if kind == BlockKind::Full {
                let b_idx = bi.get() as usize;
                if b_idx < in_ram.blocks.len() {
                    let fe = in_ram.blocks[b_idx].full();
                    if entry_is_live_sealed(pools, BlockKind::Full, fe) {
                        reused_full.push((bi, fe));
                        continue;
                    }
                }
            }
            disk_entries.push((kind, bi, sk));
        }

        let need_full = disk_entries
            .iter()
            .filter(|&&(k, _, _)| k == BlockKind::Full)
            .count();
        let need_win = disk_entries
            .iter()
            .filter(|&&(k, _, _)| k == BlockKind::Window)
            .count();
        if (need_full > 0 && pools.full.slab_count() == 0)
            || (need_win > 0 && pools.window.slab_count() == 0)
        {
            return AdmitBindOutcome::Immediate(self.bind_resident_start(pools, tbl, state, toks));
        }

        // Pin reused_full temporarily BEFORE selecting eviction victims so select_eviction_victims
        // sees exact active+provisional references and preserves `in_ram.start`.
        let mut held_reused = Vec::with_capacity(reused_full.len());
        for (bi, fe) in reused_full {
            pools
                .full
                .record_mut(fe.slot())
                .unwrap()
                .add_ref()
                .expect("pin reused full block for async restore");
            held_reused.push((BlockKind::Full, bi, fe));
        }

        let victims = match self.select_eviction_victims(pools, need_win, need_full, in_ram.start) {
            Ok(v) => v,
            Err(_) => {
                Self::release_restore_refs(pools, held_reused);
                return AdmitBindOutcome::Immediate(self.bind_resident_start(pools, tbl, state, toks));
            }
        };

        let protected_vec: Vec<SpillKey> = entries
            .iter()
            .map(|&(_, _, k)| k)
            .chain(std::iter::once(cs_key))
            .collect();
        let source = RestoreSource {
            spill_id,
            keys: protected_vec.iter().copied().collect(),
            resident_start: in_ram.start,
        };
        self.pin_inflight_restore_keys(spill_id, &protected_vec);
        for sid in victims {
            self.evict_start_inner(pools, sid, true, Some(&source));
        }

        let mut reserved_slots = Vec::with_capacity(disk_entries.len());
        let mut blocks_for_worker = Vec::with_capacity(disk_entries.len());
        for (idx, &(kind, bi, sk)) in disk_entries.iter().enumerate() {
            let Ok((slot, generation)) = self.take_zeroed_slot_inner(pools, kind, Some(&source)) else {
                Self::release_restore_refs(pools, reserved_slots);
                Self::release_restore_refs(pools, held_reused);
                self.unpin_inflight_restore_keys(spill_id, &protected_vec);
                return AdmitBindOutcome::Immediate(self.bind_resident_start(pools, tbl, state, toks));
            };
            let pool = pools.of_mut(kind);
            let expected_len = pool.stride_bytes() as usize;
            pool.clear_gpu_zero_pending(slot);
            let rec = pool.record_mut(slot).unwrap();
            rec.set_length(BLOCK as u32).expect("set reserved restore block length");
            rec.seal();
            rec.add_ref().expect("pin reserved restore slot");
            let offset = pool.offset(slot).expect("reserved restore slot offset");
            reserved_slots.push((kind, bi, BlockEntry::new(slot, generation, offset, kind, pool.kvb())));
            blocks_for_worker.push((idx, sk, expected_len));
        }

        let (dir, kvb) = {
            let store = self.spill_store.as_ref().expect("checked spill_store");
            (store.dir().to_path_buf(), store.kvb())
        };
        let restore_seq = self.next_restore_seq;
        self.next_restore_seq = self.next_restore_seq.wrapping_add(1).max(1);
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let began = Instant::now();

        self.pending_restores.insert(
            lane,
            PendingRestore {
                restore_seq,
                spill_id,
                namespace,
                pos,
                reasons,
                full_toks: toks.to_vec(),
                protected_keys: protected_vec.clone(),
                held_reused,
                reserved_slots,
                loaded_cs: None,
                loaded_blocks: 0,
                cancel_flag: cancel_flag.clone(),
            },
        );

        let worker = self.restore_worker.get_or_insert_with(RestoreWorker::spawn);
        if worker
            .job_tx
            .as_ref()
            .and_then(|tx| {
                tx.send(RestoreJob {
                    lane,
                    restore_seq,
                    dir,
                    kvb,
                    cs_key,
                    blocks: blocks_for_worker,
                    cancel_flag,
                    began,
                })
                .ok()
            })
            .is_none()
        {
            self.cancel_restore(pools, lane);
            return AdmitBindOutcome::Immediate(self.bind_resident_start(pools, tbl, state, toks));
        }

        AdmitBindOutcome::Restoring { target_pos: pos }
    }

    pub fn has_inflight_restores(&self) -> bool {
        !self.pending_restores.is_empty()
    }

    pub fn wait_restore_activity(&mut self, timeout: Duration) {
        if let Some(worker) = self.restore_worker.as_mut() {
            let wake = worker.wake.clone();
            wake.wait_timeout(&mut worker.last_wake_seq, timeout);
        }
    }

    /// Cancel an in-flight SSD restore on `lane` (`Restoring -> Cancelled`), immediately
    /// signalling the background worker to stop and releasing all pre-reserved `BlockPool` slots
    /// with zero leaks.
    pub fn cancel_restore(&mut self, pools: &mut BlockPools, lane: usize) -> bool {
        let Some(pr) = self.pending_restores.remove(&lane) else {
            return false;
        };
        pr.cancel_flag.store(true, Ordering::Release);
        Self::release_restore_refs(pools, pr.reserved_slots);
        Self::release_restore_refs(pools, pr.held_reused);
        self.unpin_inflight_restore_keys(pr.spill_id, &pr.protected_keys);
        true
    }

    pub fn cancel_all_restores(&mut self, pools: &mut BlockPools) {
        let lanes: Vec<usize> = self.pending_restores.keys().copied().collect();
        for l in lanes {
            self.cancel_restore(pools, l);
        }
    }

    /// Drain verified SSD blocks from the background restore worker into pre-reserved `BlockPool`
    /// slots and bind any completed (or corrupt-fallback) restores (`Restoring -> Ready`).
    pub fn poll_restores<F>(
        &mut self,
        pools: &mut BlockPools,
        mut lane_tables: F,
    ) -> Vec<CompletedRestore>
    where
        F: FnMut(usize) -> Option<(&'static (), *mut BlockTable, *mut LaneBlockState)>,
    {
        let mut ready_msgs = Vec::new();
        if let Some(worker) = self.restore_worker.as_ref() {
            while let Ok(msg) = worker.msg_rx.try_recv() {
                worker.budget.release(msg.budget_bytes);
                ready_msgs.push(msg);
            }
        }

        let mut completed = Vec::new();
        for msg in ready_msgs {
            let is_current = self
                .pending_restores
                .get(&msg.lane)
                .is_some_and(|pr| pr.restore_seq == msg.restore_seq);
            if !is_current {
                continue;
            }
            match msg.payload {
                RestoreMsgPayload::RecurrentState { key, raw } => {
                    let outcome = self
                        .spill_store
                        .as_mut()
                        .map(|s| s.apply_raw_read(key, raw));
                    if let Some(ReadOutcome::Hit(blk)) = outcome {
                        if let Some(pr) = self.pending_restores.get_mut(&msg.lane) {
                            pr.loaded_cs = blk.recurrent_state;
                        }
                    }
                }
                RestoreMsgPayload::Block { idx, key, raw } => {
                    let outcome = self
                        .spill_store
                        .as_mut()
                        .map(|s| s.apply_raw_read(key, raw));
                    if let Some(ReadOutcome::Hit(blk)) = outcome {
                        if let Some(pr) = self.pending_restores.get_mut(&msg.lane) {
                            let (kind, _, entry) = pr.reserved_slots[idx];
                            pools
                                .of_mut(kind)
                                .slot_bytes_mut(entry.slot())
                                .expect("reserved restore slot bytes")
                                .copy_from_slice(&blk.kv_bytes);
                            pr.loaded_blocks += 1;
                        }
                    }
                }
                RestoreMsgPayload::Done { elapsed } => {
                    let pr = self.pending_restores.remove(&msg.lane).unwrap();
                    self.unpin_inflight_restore_keys(pr.spill_id, &pr.protected_keys);
                    let ok_complete = pr.loaded_cs.as_ref().is_some_and(|c| !c.is_empty())
                        && pr.loaded_blocks == pr.reserved_slots.len();
                    if ok_complete {
                        let mut held = pr.held_reused;
                        held.extend(pr.reserved_slots);
                        let elapsed_us = elapsed.as_micros().min(u128::from(u64::MAX)) as u64;
                        let cs = pr.loaded_cs.unwrap();
                        let _ = self.publish_restored_checkpoint(
                            pools,
                            pr.spill_id,
                            pr.namespace,
                            pr.pos,
                            pr.reasons,
                            cs,
                            &pr.full_toks[..pr.pos],
                            held,
                            elapsed_us,
                        );
                        let matched = if let Some((_, tbl_ptr, st_ptr)) = lane_tables(msg.lane) {
                            let (tbl, st) = unsafe { (&mut *tbl_ptr, &mut *st_ptr) };
                            st.namespace = pr.namespace;
                            self.bind_resident_start(pools, tbl, st, &pr.full_toks)
                        } else {
                            None
                        };
                        completed.push(CompletedRestore {
                            lane: msg.lane,
                            matched,
                            elapsed,
                            corrupt_fallback: false,
                        });
                    } else {
                        Self::release_restore_refs(pools, pr.reserved_slots);
                        Self::release_restore_refs(pools, pr.held_reused);
                        self.stats.corrupt_fallbacks += 1;
                        self.remove_spilled_start(pr.spill_id, true);
                        let matched = if let Some((_, tbl_ptr, st_ptr)) = lane_tables(msg.lane) {
                            let (tbl, st) = unsafe { (&mut *tbl_ptr, &mut *st_ptr) };
                            st.namespace = pr.namespace;
                            self.bind_resident_start(pools, tbl, st, &pr.full_toks)
                        } else {
                            None
                        };
                        completed.push(CompletedRestore {
                            lane: msg.lane,
                            matched,
                            elapsed,
                            corrupt_fallback: true,
                        });
                    }
                }
                RestoreMsgPayload::Aborted { key, raw } => {
                    if let Some(store) = self.spill_store.as_mut() {
                        let _ = store.apply_raw_read(key, raw);
                    }
                    let pr = self.pending_restores.remove(&msg.lane).unwrap();
                    pr.cancel_flag.store(true, Ordering::Release);
                    self.unpin_inflight_restore_keys(pr.spill_id, &pr.protected_keys);
                    Self::release_restore_refs(pools, pr.reserved_slots);
                    Self::release_restore_refs(pools, pr.held_reused);
                    self.stats.corrupt_fallbacks += 1;
                    self.remove_spilled_start(pr.spill_id, true);
                    let matched = if let Some((_, tbl_ptr, st_ptr)) = lane_tables(msg.lane) {
                        let (tbl, st) = unsafe { (&mut *tbl_ptr, &mut *st_ptr) };
                        st.namespace = pr.namespace;
                        self.bind_resident_start(pools, tbl, st, &pr.full_toks)
                    } else {
                        None
                    };
                    completed.push(CompletedRestore {
                        lane: msg.lane,
                        matched,
                        elapsed: Duration::ZERO,
                        corrupt_fallback: true,
                    });
                }
            }
        }
        completed
    }

    fn stage_restore_blocks(
        &mut self,
        pools: &mut BlockPools,
        source: &RestoreSource,
        disk_entries: &[(BlockKind, BlockIndex, SpillKey)],
        held: &mut Vec<(BlockKind, BlockIndex, BlockEntry)>,
    ) -> Option<()> {
        let mut verified_before_eviction = false;
        for (i, &(kind, bi, key)) in disk_entries.iter().enumerate() {
            if !verified_before_eviction && pools.of(kind).lowest_free().is_none() {
                // A bad file must not discard a healthy resident checkpoint. Only the
                // restore that needs eviction pays for this extra pass, one block at a time.
                for &(remaining_kind, _, remaining_key) in &disk_entries[i..] {
                    let len = pools.of(remaining_kind).stride_bytes() as usize;
                    self.read_restore_kv(source.spill_id, remaining_key, len)?;
                }
                verified_before_eviction = true;
            }
            let len = pools.of(kind).stride_bytes() as usize;
            let bytes = self.read_restore_kv(source.spill_id, key, len)?;
            let (slot, generation) = self.take_zeroed_slot_inner(pools, kind, Some(source)).ok()?;
            let pool = pools.of_mut(kind);
            pool.slot_bytes_mut(slot).ok()?.copy_from_slice(&bytes);
            pool.clear_gpu_zero_pending(slot);
            let rec = pool.record_mut(slot).unwrap();
            rec.set_length(BLOCK as u32).expect("set restored block length");
            rec.seal();
            rec.add_ref().expect("pin restored block");
            let offset = pool.offset(slot).expect("restored slot offset");
            held.push((kind, bi, BlockEntry::new(slot, generation, offset, kind, pool.kvb())));
        }
        Some(())
    }

    fn read_restore_kv(&mut self, spill_id: u64, key: SpillKey, expected_len: usize) -> Option<Vec<u8>> {
        match self.spill_store.as_mut()?.read_block(key, Some(expected_len)) {
            ReadOutcome::Hit(block) => Some(block.kv_bytes),
            ReadOutcome::Miss(_) => {
                self.stats.corrupt_fallbacks += 1;
                self.remove_spilled_start(spill_id, true);
                None
            }
        }
    }

    fn release_restore_refs(pools: &mut BlockPools, held: Vec<(BlockKind, BlockIndex, BlockEntry)>) {
        for (kind, _, entry) in held {
            pools.of_mut(kind).record_mut(entry.slot()).unwrap().drop_ref().expect("drop restore ref");
        }
    }

    fn free_usable_slots(pool: &crate::blocks::BlockPool) -> usize {
        let mut free = 0usize;
        for i in 0..pool.len() {
            let slot = Slot::new((i + 1) as u32).expect("valid pool slot");
            let rec = pool.record(slot).expect("pool record");
            if rec.refs() == 0 && rec.generation() < u32::MAX {
                free += 1;
            }
        }
        free
    }

    /// Exact count of usable free + evictable slots in `kind`, excluding any slots pinned by
    /// active lanes or `protected_start`.
    fn reclaimable_slots(
        &self,
        pools: &BlockPools,
        kind: BlockKind,
        protected_start: Option<SealedId>,
    ) -> usize {
        let Some(psid) = protected_start else {
            return self.free_plus_evictable_slots(pools, kind);
        };
        let Some(pst) = self.starts.get(&psid) else {
            return self.free_plus_evictable_slots(pools, kind);
        };
        let mut protected_refs: HashMap<u32, u32> = HashMap::new();
        for &(ekind, _, e) in &pst.entries {
            if ekind == kind {
                *protected_refs
                    .entry(e.slot().pool_index().get())
                    .or_insert(0) += 1;
            }
        }
        let pool = pools.of(kind);
        let srefs = &self.start_slot_refs[kind_idx(kind)];
        let mut avail = 0usize;
        for i in 0..pool.len() {
            let slot = Slot::new((i + 1) as u32).expect("valid pool slot");
            let rec = pool.record(slot).expect("pool record");
            if rec.generation() < u32::MAX {
                let held_by_starts = srefs.get(i).copied().unwrap_or(0);
                let held_by_prot = protected_refs.get(&(i as u32)).copied().unwrap_or(0);
                if rec.refs() <= held_by_starts.saturating_sub(held_by_prot) {
                    avail += 1;
                }
            }
        }
        avail
    }

    /// Upfront check + minimal LRU victim prefix selection (`U105`):
    /// - If `need_win > reclaimable(Window)` or `need_full > reclaimable(Full)`, returns
    ///   `Err(CacheError::PoolExhausted(kind))` BEFORE evicting or spilling a single start point!
    /// - Otherwise returns the minimal ordered prefix of LRU `SealedId`s whose eviction brings
    ///   `(free_win, free_full)` to at least `(need_win, need_full)`, skipping any start point
    ///   whose slots are 100% pinned by active lanes.
    fn select_eviction_victims(
        &self,
        pools: &BlockPools,
        need_win: usize,
        need_full: usize,
        protected_start: Option<SealedId>,
    ) -> Result<Vec<SealedId>, CacheError> {
        if need_win > 0 {
            let avail_win = self.reclaimable_slots(pools, BlockKind::Window, protected_start);
            if need_win > avail_win {
                return Err(CacheError::PoolExhausted(BlockKind::Window));
            }
        }
        if need_full > 0 {
            let avail_full = self.reclaimable_slots(pools, BlockKind::Full, protected_start);
            if need_full > avail_full {
                return Err(CacheError::PoolExhausted(BlockKind::Full));
            }
        }

        let mut sim_free = [
            Self::free_usable_slots(&pools.window),
            Self::free_usable_slots(&pools.full),
        ];
        if sim_free[0] >= need_win && sim_free[1] >= need_full {
            return Ok(Vec::new());
        }

        let mut protected_slot_refs: [HashMap<u32, u32>; 2] = [HashMap::new(), HashMap::new()];
        if let Some(psid) = protected_start {
            if let Some(pst) = self.starts.get(&psid) {
                for &(k, _, e) in &pst.entries {
                    *protected_slot_refs[kind_idx(k)]
                        .entry(e.slot().pool_index().get())
                        .or_insert(0) += 1;
                }
            }
        }

        let mut cands: Vec<(SealedId, u64, usize)> = self
            .starts
            .iter()
            .filter(|(&sid, _)| Some(sid) != protected_start)
            .map(|(&sid, st)| (sid, st.last_used_tick, st.pos))
            .collect();
        cands.sort_by_key(|&(_, tick, pos)| (tick, std::cmp::Reverse(pos)));

        let mut dropped: [HashMap<u32, u32>; 2] = [HashMap::new(), HashMap::new()];
        let mut victims = Vec::new();
        for (sid, _, _) in cands {
            let st = &self.starts[&sid];
            let can_free_any = st.entries.iter().any(|&(kind, _, e)| {
                let ki = kind_idx(kind);
                let idx = e.slot().pool_index().get();
                let Ok(rec) = pools.of(kind).record(e.slot()) else {
                    return false;
                };
                if rec.generation() != e.generation() || rec.generation() == u32::MAX || rec.refs() == 0 {
                    return false;
                }
                let sref = self.start_slot_refs[ki].get(idx as usize).copied().unwrap_or(0);
                let pref = protected_slot_refs[ki].get(&idx).copied().unwrap_or(0);
                rec.refs() <= sref.saturating_sub(pref)
            });
            if !can_free_any {
                continue;
            }
            victims.push(sid);
            for &(kind, _, e) in &st.entries {
                let ki = kind_idx(kind);
                let idx = e.slot().pool_index().get();
                if let Ok(rec) = pools.of(kind).record(e.slot()) {
                    if rec.generation() == e.generation() && rec.generation() < u32::MAX && rec.refs() > 0 {
                        let d = dropped[ki].entry(idx).or_insert(0);
                        *d += 1;
                        if *d == rec.refs() {
                            sim_free[ki] += 1;
                        }
                    }
                }
            }
            if sim_free[0] >= need_win && sim_free[1] >= need_full {
                break;
            }
        }

        if sim_free[0] < need_win {
            return Err(CacheError::PoolExhausted(BlockKind::Window));
        }
        if sim_free[1] < need_full {
            return Err(CacheError::PoolExhausted(BlockKind::Full));
        }
        Ok(victims)
    }

    /// Non-mutating check: returns `true` iff `ensure_span_blocks(pools, tbl, p0, t)` can
    /// currently be satisfied from free + evictable unpinned slots.
    pub fn can_satisfy_span(
        &self,
        pools: &BlockPools,
        tbl: &BlockTable,
        p0: usize,
        t: usize,
    ) -> bool {
        if t == 0 {
            return true;
        }
        let lo_blk = p0 / BLOCK;
        let hi_blk = (p0 + t - 1) / BLOCK;
        let mut need_win = 0usize;
        let mut need_full = 0usize;
        for b in lo_blk..=hi_blk {
            let bi = BlockIndex::new(b as u32);
            let Ok(w) = tbl.get(BlockKind::Window, bi) else {
                return false;
            };
            if w.is_none() {
                need_win += 1;
            }
            let Ok(f) = tbl.get(BlockKind::Full, bi) else {
                return false;
            };
            if f.is_none() {
                need_full += 1;
            }
        }
        if need_win == 0 && need_full == 0 {
            return true;
        }
        need_win <= self.reclaimable_slots(pools, BlockKind::Window, None)
            && need_full <= self.reclaimable_slots(pools, BlockKind::Full, None)
    }

    #[allow(dead_code)]
    fn evict_start(&mut self, pools: &mut BlockPools, sid: SealedId) -> bool {
        self.evict_start_inner(pools, sid, true, None)
    }

    fn evict_start_inner(&mut self, pools: &mut BlockPools, sid: SealedId, allow_spill: bool, restore: Option<&RestoreSource>) -> bool {
        let Some(st) = self.starts.remove(&sid) else {
            return false;
        };
        self.retained_host_bytes = self.retained_host_bytes.saturating_sub(st.host_bytes());
        if allow_spill && self.spill_store.is_some() {
            self.spill_retained_start(pools, &st, restore);
        }
        let _ = self.tree.release_start(sid);
        for (kind, _, e) in st.entries {
            self.dec_start_slot_ref(kind, e.slot());
            let pool = pools.of_mut(kind);
            if let Ok(rec) = pool.record(e.slot()) {
                if rec.generation() == e.generation() && rec.refs() > 0 {
                    pool.record_mut(e.slot()).unwrap().drop_ref().unwrap();
                }
            }
        }
        self.tree.prune_unreferenced(sid);
        self.stats.start_points_evicted += 1;
        true
    }

    fn enforce_retained_host_budget(&mut self, pools: &mut BlockPools, protect_sid: Option<SealedId>) {
        while self.retained_host_bytes > self.max_retained_host_bytes {
            let mut victim: Option<(SealedId, u64, usize)> = None;
            for (&sid, st) in &self.starts {
                if protect_sid == Some(sid) && self.max_retained_host_bytes > 0 {
                    continue;
                }
                let better = match victim {
                    None => true,
                    Some((_, best_tick, best_pos)) => {
                        (st.last_used_tick, std::cmp::Reverse(st.pos))
                            < (best_tick, std::cmp::Reverse(best_pos))
                    }
                };
                if better {
                    victim = Some((sid, st.last_used_tick, st.pos));
                }
            }
            let Some((sid, _, _)) = victim else {
                break;
            };
            self.evict_start_inner(pools, sid, true, None);
        }
    }

    fn evict_one_lru_start_inner(&mut self, pools: &mut BlockPools, allow_spill: bool, restore: Option<&RestoreSource>) -> bool {
        let mut victim: Option<(SealedId, u64, usize)> = None;
        for (&sid, st) in &self.starts {
            if restore.is_some_and(|source| source.resident_start == Some(sid)) { continue; }
            let better = match victim {
                None => true,
                // Evict oldest LRU tick first; on a tie, evict deeper leaf before shallower shared root.
                Some((_, best_tick, best_pos)) => {
                    (st.last_used_tick, std::cmp::Reverse(st.pos))
                        < (best_tick, std::cmp::Reverse(best_pos))
                }
            };
            if better {
                victim = Some((sid, st.last_used_tick, st.pos));
            }
        }
        let Some((sid, _, _)) = victim else {
            return false;
        };
        self.evict_start_inner(pools, sid, allow_spill, restore)
    }

    /// Take one slot of `kind` without host-side full-slot CPU memset, evicting LRU retained
    /// start points only when total free + evictable slots can satisfy the request. Never grows
    /// the pool and never empties cached start points on an unsatisfiable request.
    #[cfg(any(test, standalone_blocks))]
    pub fn take_zeroed_slot(
        &mut self,
        pools: &mut BlockPools,
        kind: BlockKind,
    ) -> Result<(Slot, u32), CacheError> {
        self.take_zeroed_slot_inner(pools, kind, None)
    }

    fn take_zeroed_slot_inner(
        &mut self,
        pools: &mut BlockPools,
        kind: BlockKind,
        restore: Option<&RestoreSource>,
    ) -> Result<(Slot, u32), CacheError> {
        if let Some(slot) = pools.of(kind).lowest_free() {
            let pool = pools.of_mut(kind);
            let _ = pool.take_unzeroed(slot).expect("take slot");
            let gen = pool.record(slot).unwrap().generation();
            self.stats.slots_taken += 1;
            if gen > 1 {
                self.stats.slots_reused += 1;
            }
            return Ok((slot, gen));
        }
        let need_win = usize::from(kind == BlockKind::Window);
        let need_full = usize::from(kind == BlockKind::Full);
        let victims = self.select_eviction_victims(
            pools,
            need_win,
            need_full,
            restore.and_then(|r| r.resident_start),
        )?;
        for sid in victims {
            self.evict_start_inner(pools, sid, true, restore);
        }
        let Some(slot) = pools.of(kind).lowest_free() else {
            return Err(CacheError::PoolExhausted(kind));
        };
        let pool = pools.of_mut(kind);
        let _ = pool.take_unzeroed(slot).expect("take slot");
        let gen = pool.record(slot).unwrap().generation();
        self.stats.slots_taken += 1;
        if gen > 1 {
            self.stats.slots_reused += 1;
        }
        Ok((slot, gen))
    }

    /// Ensure every `(Full, Window)` block covering span `[p0, p0 + t)` is allocated and unsealed
    /// in `tbl`, taking zeroed slots (with bounded LRU eviction if needed). Transactional on
    /// `PoolExhausted` and mid-span rollback: verifies upfront across BOTH `Window` and `Full`
    /// that `free + evictable >= needed` before evicting or spilling a single cached start point,
    /// evicts only the minimal LRU victim prefix needed, and leaves the cache untouched if aborted.
    pub fn ensure_span_blocks(
        &mut self,
        pools: &mut BlockPools,
        tbl: &mut BlockTable,
        p0: usize,
        t: usize,
    ) -> Result<(), CacheError> {
        self.try_ensure_span_blocks_with(pools, tbl, p0, t, |_, _| Ok(()))
    }

    pub(crate) fn try_ensure_span_blocks_with<F>(
        &mut self,
        pools: &mut BlockPools,
        tbl: &mut BlockTable,
        p0: usize,
        t: usize,
        mut pre_step: F,
    ) -> Result<(), CacheError>
    where
        F: FnMut(usize, BlockKind) -> Result<(), CacheError>,
    {
        debug_assert!(t >= 1);
        let lo_blk = p0 / BLOCK;
        let hi_blk = (p0 + t - 1) / BLOCK;
        let mut missing: Vec<(BlockKind, BlockIndex)> = Vec::new();
        let mut need_win = 0usize;
        let mut need_full = 0usize;

        for b in lo_blk..=hi_blk {
            let bi = BlockIndex::new(b as u32);
            for kind in KINDS {
                let existing = tbl.get(kind, bi).expect("block in table range");
                if let Some(entry) = existing {
                    let rec = pools.of(kind).record(entry.slot()).expect("existing slot record");
                    assert_eq!(rec.generation(), entry.generation(), "stale entry in active lane table");
                    assert!(!rec.sealed(), "attempt to write into sealed {kind:?} block {b}");
                } else {
                    missing.push((kind, bi));
                    match kind {
                        BlockKind::Window => need_win += 1,
                        BlockKind::Full => need_full += 1,
                    }
                }
            }
        }
        if missing.is_empty() {
            return Ok(());
        }

        // Phase 1: Upfront bounded capacity check + minimal victim selection BEFORE any eviction/spill.
        let victims = self.select_eviction_victims(pools, need_win, need_full, None)?;

        // Phase 2: Allocate from already-free slots first while running `pre_step`; defer victim
        // eviction until all pre-checks succeed (or take free slots up to the point where eviction
        // is required, rolling back taken free slots with zero cache evictions if `pre_step` aborts!).
        let mut newly_allocated: Vec<(BlockKind, BlockIndex)> = Vec::with_capacity(missing.len());
        let mut deferred_after_eviction: Vec<(BlockKind, BlockIndex)> = Vec::new();
        for (kind, bi) in missing {
            if let Err(err) = pre_step(bi.get() as usize, kind) {
                for (rk, rbi) in newly_allocated.into_iter().rev() {
                    tbl.clear_kind_entry(pools.of_mut(rk), rbi) // MUTANT_ANCHOR_SPAN_ROLLBACK
                        .expect("rollback newly allocated block entry");
                }
                return Err(err);
            }
            if let Some(slot) = pools.of(kind).lowest_free() {
                let pool = pools.of_mut(kind);
                let _ = pool.take_unzeroed(slot).expect("take slot");
                let gen = pool.record(slot).unwrap().generation();
                self.stats.slots_taken += 1;
                if gen > 1 {
                    self.stats.slots_reused += 1;
                }
                tbl.fill(pool, bi, slot, gen)
                    .expect("fill newly taken free slot");
                newly_allocated.push((kind, bi));
            } else {
                deferred_after_eviction.push((kind, bi));
            }
        }

        if !deferred_after_eviction.is_empty() {
            for sid in victims {
                self.evict_start_inner(pools, sid, true, None);
            }
            for (kind, bi) in deferred_after_eviction {
                let Some(slot) = pools.of(kind).lowest_free() else {
                    for (rk, rbi) in newly_allocated.into_iter().rev() {
                        tbl.clear_kind_entry(pools.of_mut(rk), rbi)
                            .expect("rollback newly allocated block entry");
                    }
                    return Err(CacheError::PoolExhausted(kind));
                };
                let pool = pools.of_mut(kind);
                let _ = pool.take_unzeroed(slot).expect("take slot");
                let gen = pool.record(slot).unwrap().generation();
                self.stats.slots_taken += 1;
                if gen > 1 {
                    self.stats.slots_reused += 1;
                }
                tbl.fill(pool, bi, slot, gen)
                    .expect("fill newly evicted slot");
                newly_allocated.push((kind, bi));
            }
        }

        Ok(())
    }

    fn prune_expired_window(&mut self, pools: &mut BlockPools, tbl: &mut BlockTable, start: usize, end: usize) {
        for b in newly_expired_window_blocks(start, end) {
            let bi = BlockIndex::new(b as u32);
            if tbl.get(BlockKind::Window, bi).expect("window block in range").is_some() {
                tbl.clear_kind_entry(&mut pools.window, bi)
                    .expect("prune expired window block");
                self.stats.window_pruned += 1;
            }
        }
    }

    /// Post-prefill bookkeeping: update lengths, seal completed 64-token blocks into `EntryTree`,
    /// deduplicate against live canonical blocks, retain `EVERY` / `PROMPT` start points, and
    /// prune expired sliding-window blocks.
    pub fn finish_prefill_span<F>(
        &mut self,
        pools: &mut BlockPools,
        tbl: &mut BlockTable,
        state: &mut LaneBlockState,
        toks: &[u32],
        p0: usize,
        t: usize,
        read_cs: F,
    ) where
        F: FnOnce() -> Vec<u8>,
    {
        let end = p0 + t;
        state.written = end;
        let lo_blk = p0 / BLOCK;
        let hi_blk = (end - 1) / BLOCK;
        for b in lo_blk..=hi_blk {
            let bi = BlockIndex::new(b as u32);
            let block_len = (end - b * BLOCK).min(BLOCK) as u32;
            let full_e = tbl.get(BlockKind::Full, bi).unwrap().expect("full entry");
            let win_e = tbl.get(BlockKind::Window, bi).unwrap().expect("window entry");
            for (kind, e) in [(BlockKind::Full, full_e), (BlockKind::Window, win_e)] {
                let rec = pools.of_mut(kind).record_mut(e.slot()).unwrap();
                rec.set_length(block_len).expect("set block length");
                if block_len == BLOCK as u32 {
                    rec.seal();
                }
            }
            if block_len == BLOCK as u32 {
                self.stats.blocks_sealed += 1;
                if b == 0 || state.sealed.len() == b {
                    let parent = if b == 0 { None } else { state.sealed.get(b - 1).copied() };
                    let blk_toks = &toks[b * BLOCK..(b + 1) * BLOCK];
                    let handles = BlockHandles::new(full_e, Some(win_e));
                    let id = self
                        .tree
                        .insert_in_namespace(state.namespace, parent, b * BLOCK, blk_toks, handles)
                        .expect("insert sealed block into EntryTree");
                    let canon = self.tree.handle(id).expect("inserted handle");
                    let canon_f = canon.full();
                    let canon_w = canon.window();
                    let mut best_f = full_e;
                    let mut best_w = win_e;
                    if canon_f != full_e && entry_is_live_sealed(pools, BlockKind::Full, canon_f) {
                        tbl.fill(&mut pools.full, bi, canon_f.slot(), canon_f.generation())
                            .expect("share canonical full block");
                        best_f = canon_f;
                        self.stats.shared_dedups += 1;
                    }
                    if let Some(canon_w) = canon_w.filter(|&w| w != win_e && entry_is_live_sealed(pools, BlockKind::Window, w)) {
                        tbl.fill(&mut pools.window, bi, canon_w.slot(), canon_w.generation())
                            .expect("share canonical window block");
                        best_w = canon_w;
                    }
                    if best_f != canon_f || Some(best_w) != canon_w {
                        let _ = self.tree.set_handle(id, BlockHandles::new(best_f, Some(best_w)));
                    }
                    self.tree.pin(id).expect("pin newly sealed block");
                    if state.sealed.len() == b {
                        state.sealed.push(id);
                    } else {
                        let old = state.sealed[b];
                        if old != id {
                            self.tree.unpin(old);
                        }
                        state.sealed[b] = id;
                    }
                }
            }
        }
        // Retain start points on 64-token boundaries for `EVERY` and `PROMPT` checkpoints.
        if end > 0 && end.is_multiple_of(BLOCK) && state.sealed.len() == end / BLOCK {
            let prompt_ckpt = ((state.prompt_len.saturating_sub(1)) / BLOCK) * BLOCK;
            let mut reasons = 0u8;
            if (prompt_ckpt > 0 && end == prompt_ckpt) || end == state.prompt_len {
                reasons |= PROMPT;
            }
            if is_every_checkpoint(end) {
                reasons |= EVERY;
            }
            if reasons != 0 {
                let tip = *state.sealed.last().unwrap();
                self.tick += 1;
                let now = self.tick;
                let spill_enabled = self.spill_store.is_some();
                let mut check_host_budget = false;
                if let Some(existing) = self.starts.get_mut(&tip) {
                    existing.reasons |= reasons;
                    existing.last_used_tick = now;
                    if spill_enabled && existing.toks.is_empty() {
                        let added = (end as u64).saturating_mul(4);
                        existing.toks = toks[..end].to_vec();
                        self.retained_host_bytes = self.retained_host_bytes.saturating_add(added);
                        check_host_budget = true;
                    }
                } else {
                    let cs = read_cs();
                    let mut entries = Vec::with_capacity(end / BLOCK + WINDOW / BLOCK);
                    for b in 0..(end / BLOCK) {
                        let bi = BlockIndex::new(b as u32);
                        let fe = tbl.get(BlockKind::Full, bi).unwrap().expect("start point full block");
                        entries.push((BlockKind::Full, bi, fe));
                    }
                    let win_lo = end.saturating_sub(WINDOW) / BLOCK;
                    for b in win_lo..(end / BLOCK) {
                        let bi = BlockIndex::new(b as u32);
                        let we = tbl.get(BlockKind::Window, bi).unwrap().expect("start point window block");
                        entries.push((BlockKind::Window, bi, we));
                    }
                    for &(kind, _, e) in &entries {
                        pools
                            .of_mut(kind)
                            .record_mut(e.slot())
                            .unwrap()
                            .add_ref()
                            .expect("retain start point ref");
                        self.inc_start_slot_ref(kind, e.slot()); // MUTANT_ANCHOR_START_SLOT_REF
                    }
                    self.tree.retain_start(tip).expect("retain start in EntryTree");
                    let st = RetainedStart {
                        namespace: state.namespace,
                        pos: end,
                        cs,
                        reasons,
                        entries,
                        toks: if spill_enabled { toks[..end].to_vec() } else { Vec::new() },
                        last_used_tick: now,
                    };
                    self.retained_host_bytes = self.retained_host_bytes.saturating_add(st.host_bytes());
                    self.starts.insert(tip, st);
                    self.stats.start_points_saved += 1;
                    check_host_budget = true;
                }
                if check_host_budget {
                    self.enforce_retained_host_budget(pools, Some(tip));
                }
            }
        }
        self.prune_expired_window(pools, tbl, p0, end);
    }

    /// Post-decode bookkeeping: update unsealed block length (never sealed into `EntryTree`)
    /// and prune expired sliding-window blocks.
    pub fn finish_decode_span(
        &mut self,
        pools: &mut BlockPools,
        tbl: &mut BlockTable,
        state: &mut LaneBlockState,
        pos: usize,
    ) {
        let end = pos + 1;
        state.written = end;
        let b = pos / BLOCK;
        let bi = BlockIndex::new(b as u32);
        let block_len = (end - b * BLOCK).min(BLOCK) as u32;
        for kind in KINDS {
            let e = tbl.get(kind, bi).unwrap().expect("decode block entry");
            // Decode-written blocks stay private to the lane and are never sealed into EntryTree.
            pools
                .of_mut(kind)
                .record_mut(e.slot())
                .unwrap()
                .set_length(block_len)
                .expect("set decode length");
        }
        self.prune_expired_window(pools, tbl, pos, end);
    }

    /// Release a lane's block-table references and unpin its `EntryTree` path so private/unsealed
    /// blocks immediately return to `refs == 0` while retained start points stay live.
    pub fn release_lane(
        &mut self,
        pools: &mut BlockPools,
        tbl: &mut BlockTable,
        state: &mut LaneBlockState,
    ) {
        tbl.clear_entries(pools).expect("clear lane block table");
        let tip = state.sealed.last().copied();
        for id in state.sealed.drain(..) {
            self.tree.unpin(id);
        }
        if let Some(tip) = tip {
            self.tree.prune_unreferenced(tip);
        }
        state.namespace = 0;
        state.prompt_len = 0;
        state.written = 0;
        self.stats.lanes_released += 1;
    }

    /// Evict every retained start point and cancel any in-flight restores so all pool
    /// references drain cleanly on shutdown.
    pub fn shutdown(&mut self, pools: &mut BlockPools) {
        self.cancel_all_restores(pools);
        while self.evict_one_lru_start_inner(pools, false, None) {}
    }
}

#[cfg(standalone_blocks)]
#[derive(Clone, Debug)]
pub struct OwnershipSnapshot {
    pub num_full: usize,
    pub num_win: usize,
    pub full_refs: Vec<u32>,
    pub win_refs: Vec<u32>,
    pub full_gen: Vec<u32>,
    pub win_gen: Vec<u32>,
    pub start_full_refs: Vec<u32>,
    pub start_win_refs: Vec<u32>,
    pub actual_start_full_refs: Vec<u32>,
    pub actual_start_win_refs: Vec<u32>,
    pub table_full_refs: Vec<u32>,
    pub table_win_refs: Vec<u32>,
    pub free_plus_evict_full: usize,
    pub free_plus_evict_win: usize,
    pub lane_full_blocks: Vec<Vec<u32>>,
    pub lane_win_blocks: Vec<Vec<u32>>,
    pub lane_win_count: Vec<usize>,
    pub all_lane_entries_live: bool,
    pub all_start_entries_live_sealed: bool,
    pub tree_start_holders_match: bool,
    pub tree_pins_match_lanes: bool,
    pub expired_win_present: bool,
    pub spill_owners_consistent: bool,
    pub spilled_tips_consistent: bool,
    pub retained_starts: usize,
    pub spilled_starts: usize,
}

#[cfg(standalone_blocks)]
impl ServeBlockCache {
    /// Compute the exact ownership, generation, EntryTree, sliding-window, and SpillStore
    /// consistency snapshot across `pools`, `self`, and all active `lanes`.
    pub fn ownership_snapshot(
        &self,
        pools: &BlockPools,
        lanes: &[(&BlockTable, &LaneBlockState)],
    ) -> OwnershipSnapshot {
        let num_full = pools.full.len();
        let num_win = pools.window.len();
        let mut full_refs = Vec::with_capacity(num_full);
        let mut full_gen = Vec::with_capacity(num_full);
        for i in 0..num_full {
            let slot = Slot::new((i + 1) as u32).unwrap();
            let r = pools.full.record(slot).unwrap();
            full_refs.push(r.refs());
            full_gen.push(r.generation());
        }
        let mut win_refs = Vec::with_capacity(num_win);
        let mut win_gen = Vec::with_capacity(num_win);
        for i in 0..num_win {
            let slot = Slot::new((i + 1) as u32).unwrap();
            let r = pools.window.record(slot).unwrap();
            win_refs.push(r.refs());
            win_gen.push(r.generation());
        }

        let mut start_full_refs = vec![0u32; num_full];
        for (i, &c) in self.start_slot_refs[kind_idx(BlockKind::Full)].iter().enumerate() {
            if i < num_full {
                start_full_refs[i] = c;
            }
        }
        let mut start_win_refs = vec![0u32; num_win];
        for (i, &c) in self.start_slot_refs[kind_idx(BlockKind::Window)].iter().enumerate() {
            if i < num_win {
                start_win_refs[i] = c;
            }
        }

        let mut actual_start_full_refs = vec![0u32; num_full];
        let mut actual_start_win_refs = vec![0u32; num_win];
        let mut all_start_entries_live_sealed = true;
        for st in self.starts.values() {
            if st.reasons == 0 {
                all_start_entries_live_sealed = false;
            }
            for &(kind, _, e) in &st.entries {
                if !entry_is_live_sealed(pools, kind, e) {
                    all_start_entries_live_sealed = false;
                }
                let idx = e.slot().pool_index().get() as usize;
                match kind {
                    BlockKind::Full => actual_start_full_refs[idx] += 1,
                    BlockKind::Window => actual_start_win_refs[idx] += 1,
                }
            }
        }

        let mut table_full_refs = vec![0u32; num_full];
        let mut table_win_refs = vec![0u32; num_win];
        let mut lane_full_blocks = Vec::with_capacity(lanes.len());
        let mut lane_win_blocks = Vec::with_capacity(lanes.len());
        let mut lane_win_count = Vec::with_capacity(lanes.len());
        let mut all_lane_entries_live = true;
        let mut expired_win_present = false;
        let mut expected_pins: HashMap<SealedId, u32> = HashMap::new();

        for &(tbl, st) in lanes {
            let mut f_blks = Vec::new();
            let mut w_blks = Vec::new();
            let min_live_win_blk = if st.written > 0 {
                st.written.saturating_sub(EXT - 1) / BLOCK
            } else {
                0
            };
            for (bi, e) in tbl.entries() {
                let pool = pools.of(e.kind());
                match pool.record(e.slot()) {
                    Ok(rec) if rec.generation() == e.generation() && rec.refs() > 0 => {}
                    _ => all_lane_entries_live = false,
                }
                let idx = e.slot().pool_index().get() as usize;
                match e.kind() {
                    BlockKind::Full => {
                        table_full_refs[idx] += 1;
                        f_blks.push(bi.get());
                    }
                    BlockKind::Window => {
                        table_win_refs[idx] += 1;
                        w_blks.push(bi.get());
                        if st.written > 0 && (bi.get() as usize) < min_live_win_blk {
                            expired_win_present = true;
                        }
                    }
                }
            }
            lane_win_count.push(w_blks.len());
            lane_full_blocks.push(f_blks);
            lane_win_blocks.push(w_blks);
            for &sid in &st.sealed {
                *expected_pins.entry(sid).or_insert(0) += 1;
            }
        }

        let mut tree_start_holders_match = true;
        let mut tree_pins_match_lanes = true;
        let tree_entries = self.tree.entries();
        let tree_ids: HashSet<SealedId> = tree_entries.iter().map(|&(id, _)| id).collect();
        for &sid in self.starts.keys() {
            if !tree_ids.contains(&sid) || self.tree.start_holders(sid) == 0 {
                tree_start_holders_match = false;
            }
        }
        for (&sid, &want) in &expected_pins {
            if !tree_ids.contains(&sid) || self.tree.pin_count(sid) != want {
                tree_pins_match_lanes = false;
            }
        }
        for (sid, _) in tree_entries {
            let want_start = u32::from(self.starts.contains_key(&sid));
            if self.tree.start_holders(sid) != want_start {
                tree_start_holders_match = false;
            }
            let want_pin = expected_pins.get(&sid).copied().unwrap_or(0);
            if self.tree.pin_count(sid) != want_pin {
                tree_pins_match_lanes = false;
            }
        }

        let mut expected_owners: HashMap<SpillKey, Vec<u64>> = HashMap::new();
        let mut spilled_tips_consistent = self.spilled_tips.len() == self.spilled_starts.len();
        for (&spill_id, sp) in &self.spilled_starts {
            if sp.reasons == 0 {
                spilled_tips_consistent = false;
            }
            match self.spilled_tips.get(&spill_id) {
                Some(&tip) => {
                    if self.spilled_tree.start_holders(tip) == 0
                        || self.spilled_tree.handle(tip) != Some(spill_id)
                    {
                        spilled_tips_consistent = false;
                    }
                }
                None => spilled_tips_consistent = false,
            }
            for sk in std::iter::once(sp.cs_key).chain(sp.entries.iter().map(|&(_, _, k)| k)) {
                expected_owners.entry(sk).or_default().push(spill_id);
            }
        }
        let mut spill_owners_consistent = expected_owners.len() == self.spill_key_owners.len();
        if spill_owners_consistent {
            for (sk, mut want) in expected_owners {
                want.sort_unstable();
                match self.spill_key_owners.get(&sk) {
                    Some(got_raw) => {
                        let mut got = got_raw.clone();
                        got.sort_unstable();
                        if got != want {
                            spill_owners_consistent = false;
                            break;
                        }
                        if let Some(store) = &self.spill_store {
                            if !store.contains(sk) {
                                spill_owners_consistent = false;
                                break;
                            }
                        }
                    }
                    None => {
                        spill_owners_consistent = false;
                        break;
                    }
                }
            }
        }

        OwnershipSnapshot {
            num_full,
            num_win,
            full_refs,
            win_refs,
            full_gen,
            win_gen,
            start_full_refs,
            start_win_refs,
            actual_start_full_refs,
            actual_start_win_refs,
            table_full_refs,
            table_win_refs,
            free_plus_evict_full: self.free_plus_evictable_slots(pools, BlockKind::Full),
            free_plus_evict_win: self.free_plus_evictable_slots(pools, BlockKind::Window),
            lane_full_blocks,
            lane_win_blocks,
            lane_win_count,
            all_lane_entries_live,
            all_start_entries_live_sealed,
            tree_start_holders_match,
            tree_pins_match_lanes,
            expired_win_present,
            spill_owners_consistent,
            spilled_tips_consistent,
            retained_starts: self.starts.len(),
            spilled_starts: self.spilled_starts.len(),
        }
    }

    /// Deepest live checkpoint across BOTH resident RAM starts and valid uncorrupted SSD spilled starts
    /// strictly shorter than `toks` in `namespace` (ground-truth oracle for `ServeOwnershipTrace`).
    pub fn expected_deepest_live_at(
        &self,
        pools: &BlockPools,
        namespace: u64,
        toks: &[u32],
    ) -> usize {
        if toks.len() <= 1 {
            return 0;
        }
        let prefix = &toks[..toks.len() - 1];
        let mut best = 0usize;
        for st in self.starts.values() {
            if st.namespace == namespace
                && st.pos <= prefix.len()
                && !st.toks.is_empty()
                && prefix.starts_with(&st.toks)
                && st.entries.iter().all(|&(k, _, e)| entry_is_live_sealed(pools, k, e))
            {
                best = best.max(st.pos);
            }
        }
        // Also scan EntryTree nodes directly in case spill_store is None (st.toks empty).
        let ram = self.peek_match_len_in_namespace(pools, namespace, toks);
        best = best.max(ram);
        for (&sid, st) in &self.starts {
            if st.namespace == namespace && st.pos > best && st.pos <= prefix.len() {
                // Verify exact prefix via tree handle chain if st.toks was not populated.
                let _ = sid;
            }
        }
        if self.spill_store.is_some() {
            let disk = self.spilled_tree.deepest_start_in_namespace(namespace, prefix).at;
            best = best.max(disk);
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod spill_model;

    fn fake_cs(tag: u8) -> Vec<u8> {
        vec![tag; 16]
    }

    #[test]
    fn window_pruning_visits_each_expired_block_once() {
        // A cold prefill or a single span that advances a restored checkpoint.
        assert_eq!(newly_expired_window_blocks(0, 2048), 0..16);
        assert_eq!(newly_expired_window_blocks(2048, 4096), 16..48);
        // At 128K context, 64 consecutive decode steps expire exactly one block.
        let pos = 131072;
        let first_live = (pos - EXT) / BLOCK;
        let visited: Vec<_> = (pos..pos + BLOCK)
            .flat_map(|p| newly_expired_window_blocks(p, p + 1))
            .collect();
        assert_eq!(visited, vec![first_live]);
        assert_eq!(newly_expired_window_blocks(pos, pos + BLOCK - 2), first_live..first_live);
        assert_eq!(newly_expired_window_blocks(pos + BLOCK - 2, pos + BLOCK - 1), first_live..first_live + 1);
    }

    #[test]
    fn effective_pf_target_stops_at_every_and_prompt_boundaries() {
        // Short prompt (< 64): single step straight to pf_end.
        assert_eq!(effective_pf_target(0, 18), 18);
        assert_eq!(effective_pf_target(0, 64), 64);

        // OpenCode-sized prompt (8266): 2048, 4096, 6144, 8192, 8256, 8266.
        let mut p = 0;
        let mut stops = Vec::new();
        while p < 8266 {
            let next = effective_pf_target(p, 8266);
            assert!(next > p && next <= 8266);
            stops.push(next);
            p = next;
        }
        assert_eq!(stops, vec![2048, 4096, 6144, 8192, 8256, 8266]);

        // Exact 64-token multiple (8256): stops at 8192 (prompt_ckpt) then finishes at 8256.
        assert_eq!(effective_pf_target(6144, 8256), 8192);
        assert_eq!(effective_pf_target(8192, 8256), 8256);

        // Above 16384: widens to 4096-token chunks.
        assert_eq!(effective_pf_target(16384, 25000), 20480);
        assert_eq!(effective_pf_target(20480, 25000), 24576);
        assert_eq!(effective_pf_target(24576, 25000), 24960);
        assert_eq!(effective_pf_target(24960, 25000), 25000);
    }

    #[test]
    fn multiturn_reuse_and_cross_card_sharing_with_zero_leaks() {
        // Host pools: 512 full slots, 256 window slots.
        let mut pools = BlockPools::with_capacities(512, 256, 2).unwrap();
        let mut cache = ServeBlockCache::new();
        let mut tbl = BlockTable::new(512).unwrap();
        let mut st = LaneBlockState::default();

        // Turn 1: 8,266 tokens (> EXT = 1024).
        let turn1: Vec<u32> = (0..8266u32).map(|i| (i * 17 + 3) % 50000).collect();
        assert!(cache.match_and_bind(&mut pools, &mut tbl, &mut st, &turn1).is_none());
        let mut p = 0;
        while p < turn1.len() {
            let next = effective_pf_target(p, turn1.len());
            cache.ensure_span_blocks(&mut pools, &mut tbl, p, next - p).unwrap();
            cache.finish_prefill_span(&mut pools, &mut tbl, &mut st, &turn1, p, next - p, || {
                fake_cs((next / BLOCK) as u8)
            });
            p = next;
        }
        // Decode 100 tokens past prompt end (must NOT invalidate retained start points!).
        for d in 0..100 {
            let pos = turn1.len() + d;
            cache.ensure_span_blocks(&mut pools, &mut tbl, pos, 1).unwrap();
            cache.finish_decode_span(&mut pools, &mut tbl, &mut st, pos);
        }
        cache.release_lane(&mut pools, &mut tbl, &mut st);
        // Should have retained EVERY starts at 2048, 4096, 6144, 8192 + PROMPT start at 8256 = 5 starts.
        assert_eq!(cache.retained_start_count(), 5);

        // Turn 2: extends Turn 1 by 140 tokens (8,406 total) -> must hit 8,256!
        let mut turn2 = turn1.clone();
        turn2.extend((0..140u32).map(|i| 60000 + i));
        let (matched, cs) = cache
            .match_and_bind(&mut pools, &mut tbl, &mut st, &turn2)
            .expect("Turn 2 must match Turn 1 prompt checkpoint");
        assert_eq!(matched, 8256);
        assert_eq!(cs, fake_cs((8256 / BLOCK) as u8));
        // The first window block restored with this checkpoint stays live up to the boundary.
        let oldest = BlockIndex::new(((matched - EXT) / BLOCK) as u32);
        let near_boundary = matched + BLOCK - 2;
        for (start, end) in [(matched, near_boundary), (near_boundary, near_boundary + 1)] {
            cache.ensure_span_blocks(&mut pools, &mut tbl, start, end - start).unwrap();
            cache.finish_prefill_span(&mut pools, &mut tbl, &mut st, &turn2, start, end - start, || fake_cs(0));
            assert_eq!(tbl.get(BlockKind::Window, oldest).unwrap().is_some(), end == near_boundary);
            assert!(tbl.get(BlockKind::Full, oldest).unwrap().is_some());
        }
        cache.release_lane(&mut pools, &mut tbl, &mut st);

        // Card 2: shares first 6,500 tokens (system+tools), diverges after -> must hit 6,144!
        let mut card2 = turn1[..6500].to_vec();
        card2.extend((0..1000u32).map(|i| 70000 + i));
        let (matched_c2, cs_c2) = cache
            .match_and_bind(&mut pools, &mut tbl, &mut st, &card2)
            .expect("Card 2 must match shared EVERY checkpoint");
        assert_eq!(matched_c2, 6144);
        assert_eq!(cs_c2, fake_cs((6144 / BLOCK) as u8));
        cache.release_lane(&mut pools, &mut tbl, &mut st);

        // Shutdown drains every ref back to 0.
        cache.shutdown(&mut pools);
        assert_eq!(cache.retained_start_count(), 0);
        assert_eq!(cache.free_plus_evictable_slots(&pools, BlockKind::Full), 512);
        assert_eq!(cache.free_plus_evictable_slots(&pools, BlockKind::Window), 256);
        for kind in KINDS {
            let pool = pools.of(kind);
            for i in 0..pool.len() {
                let s = Slot::new((i + 1) as u32).unwrap();
                assert_eq!(pool.record(s).unwrap().refs(), 0, "leaked ref in {kind:?} slot {}", s.get());
            }
        }
    }

    #[test]
    fn lru_eviction_and_pool_pressure_guard_never_panic() {
        // Tight pool: 80 full slots (5120 tokens), 48 window slots.
        let mut pools = BlockPools::with_capacities(80, 48, 2).unwrap();
        let mut cache = ServeBlockCache::new();
        let mut tbl = BlockTable::new(128).unwrap();
        let mut st = LaneBlockState::default();

        // Fill cache with 4 distinct 1024-token prompts (64 full slots + 64 window slots total -> forces LRU eviction!).
        for req in 0..4u32 {
            let toks: Vec<u32> = (0..1030u32).map(|i| req * 10000 + i).collect();
            let _ = cache.match_and_bind(&mut pools, &mut tbl, &mut st, &toks);
            let mut p = 0;
            while p < toks.len() {
                let next = effective_pf_target(p, toks.len());
                cache.ensure_span_blocks(&mut pools, &mut tbl, p, next - p).unwrap();
                cache.finish_prefill_span(&mut pools, &mut tbl, &mut st, &toks, p, next - p, || fake_cs(req as u8));
                p = next;
            }
            cache.release_lane(&mut pools, &mut tbl, &mut st);
        }
        assert!(cache.stats().start_points_evicted > 0, "tight pool must trigger LRU eviction");
        // Most recent prompt (req=3) must still be cached at 1024!
        let req3_ext: Vec<u32> = (0..1050u32).map(|i| 3 * 10000 + i).collect();
        assert_eq!(cache.peek_match_len(&pools, &req3_ext), 1024);

        // Pool-pressure admit guard: always admits when active_lanes == 0, defers when active lane fills pool.
        let big: Vec<u32> = (0..4000u32).map(|i| 90000 + i).collect();
        assert!(cache.can_admit(&pools, &[], 0, &big, 1024, 2048));
        // Requesting more window blocks in a single span than exist in the pool returns PoolExhausted without panicking.
        assert_eq!(
            cache.ensure_span_blocks(&mut pools, &mut tbl, 0, 3840),
            Err(CacheError::PoolExhausted(BlockKind::Window))
        );
        cache.release_lane(&mut pools, &mut tbl, &mut st);

        // Pin 60 full slots across chunked steps (so expired window blocks prune cleanly while full slots fill).
        let active_toks: Vec<u32> = (0..3840u32).map(|i| 50000 + i).collect();
        let mut ap = 0;
        while ap < active_toks.len() {
            let chunk_t = 1024.min(active_toks.len() - ap);
            cache.ensure_span_blocks(&mut pools, &mut tbl, ap, chunk_t).unwrap();
            cache.finish_prefill_span(&mut pools, &mut tbl, &mut st, &active_toks, ap, chunk_t, || fake_cs(9));
            ap += chunk_t;
        }
        assert!(!cache.can_admit(&pools, &[InflightLaneReserve { lane: 0, namespace: 0, toks: &active_toks, block_hashes: None, n_past: ap, pf_end: ap, cached: 0, written: ap, gen: 0, max_gen: 1, runnable: true }], 1, &big, 1024, 2048), "should defer when active lane consumes pool");
        cache.release_lane(&mut pools, &mut tbl, &mut st);
        cache.shutdown(&mut pools);
    }

    #[test]
    fn concurrent_same_prefix_dedups_and_refreshes_expired_window() {
        let mut pools = BlockPools::with_capacities(256, 128, 2).unwrap();
        let mut cache = ServeBlockCache::new();
        let mut tbl0 = BlockTable::new(256).unwrap();
        let mut tbl1 = BlockTable::new(256).unwrap();
        let mut st0 = LaneBlockState::default();
        let mut st1 = LaneBlockState::default();

        // Lane 0 prefills 2100 tokens -> saves EVERY start at 2048 (window blocks 16..32, NOT 0..16!).
        let prompt0: Vec<u32> = (0..2100u32).collect();
        let mut p = 0;
        while p < prompt0.len() {
            let next = effective_pf_target(p, prompt0.len());
            cache.ensure_span_blocks(&mut pools, &mut tbl0, p, next - p).unwrap();
            cache.finish_prefill_span(&mut pools, &mut tbl0, &mut st0, &prompt0, p, next - p, || fake_cs(1));
            p = next;
        }
        cache.release_lane(&mut pools, &mut tbl0, &mut st0);

        // Lane 1 shares blocks 0..10 (640 tokens) with prompt0, then diverges before 2048.
        // Its block 0..10 Full slots dedup against prompt0's live Full slots while Window slots
        // were expired from prompt0 -> must NOT destroy prompt0's 2048 start point!
        let mut prompt1 = prompt0[..640].to_vec();
        prompt1.extend((0..200u32).map(|i| 80000 + i));
        assert!(cache.match_and_bind(&mut pools, &mut tbl1, &mut st1, &prompt1).is_none());
        let mut p1 = 0;
        while p1 < prompt1.len() {
            let next = effective_pf_target(p1, prompt1.len());
            cache.ensure_span_blocks(&mut pools, &mut tbl1, p1, next - p1).unwrap();
            cache.finish_prefill_span(&mut pools, &mut tbl1, &mut st1, &prompt1, p1, next - p1, || fake_cs(2));
            p1 = next;
        }
        cache.release_lane(&mut pools, &mut tbl1, &mut st1);
        assert!(cache.stats().shared_dedups >= 10, "blocks 0..10 full slots should dedup");

        // Verify prompt0's 2048 start point is STILL intact and matchable!
        let mut prompt0_ext = prompt0.clone();
        prompt0_ext.push(99999);
        assert_eq!(cache.peek_match_len(&pools, &prompt0_ext), 2048);

        cache.shutdown(&mut pools);
    }

    fn assert_zero_pool_leaks(pools: &BlockPools) {
        for kind in KINDS {
            let pool = pools.of(kind);
            for i in 0..pool.len() {
                let s = Slot::new((i + 1) as u32).unwrap();
                assert_eq!(
                    pool.record(s).unwrap().refs(),
                    0,
                    "leaked ref in {kind:?} slot {}",
                    s.get()
                );
            }
        }
    }

    #[test]
    fn singleflight_22_lanes_rebind_and_prefill_only_distinct_tails() {
        use crate::scheduler::{ServeLaneInput, ServeStepPlan, ServeStepPlanner};

        // 512 Full slots (32,768 tokens) is far below the 22 * 138 = 3,036 slots needed without dedup.
        let mut pools = BlockPools::with_capacities(512, 512, 2).unwrap();
        let mut cache = ServeBlockCache::new();
        let prefix: Vec<u32> = (0..8500u32).map(|i| (i * 31 + 7) % 50000).collect();

        struct SimLane {
            toks: Vec<u32>,
            tbl: BlockTable,
            blocks: LaneBlockState,
            n_past: usize,
            cached: usize,
            prefilled: usize,
            cs: Vec<u8>,
            active: bool,
        }

        let nlanes = 22usize;
        let chunk = 4096usize;
        let mut lanes: Vec<SimLane> = (0..nlanes)
            .map(|i| {
                let mut toks = prefix.clone();
                toks.extend((0..300u32).map(|j| 60000 + (i as u32) * 1000 + j));
                SimLane {
                    toks,
                    tbl: BlockTable::new(512).unwrap(),
                    blocks: LaneBlockState::default(),
                    n_past: 0,
                    cached: 0,
                    prefilled: 0,
                    cs: Vec::new(),
                    active: false,
                }
            })
            .collect();

        // Admit all 22 simultaneous requests at t=0 via in-flight shared prefix accounting.
        let mut planner = ServeStepPlanner::new(nlanes, 16);
        for i in 0..nlanes {
            let reserves: Vec<InflightLaneReserve<'_>> = lanes
                .iter()
                .enumerate()
                .filter(|(_, l)| l.active)
                .map(|(idx, l)| InflightLaneReserve {
                    lane: idx,
                    namespace: 0,
                    toks: &l.toks,
                    block_hashes: None,
                    n_past: l.n_past,
                    pf_end: l.toks.len(),
                    cached: l.cached,
                    written: l.blocks.written,
                    gen: 0,
                    max_gen: 128,
                    runnable: true,
                })
                .collect();
            assert!(
                cache.can_admit(&pools, &reserves, i, &lanes[i].toks, 128, chunk),
                "lane {i} should admit under singleflight shared prefix accounting"
            );
            let l = &mut lanes[i];
            assert!(cache.match_and_bind(&mut pools, &mut l.tbl, &mut l.blocks, &l.toks).is_none());
            l.active = true;
            planner.on_admit(i);
        }

        let mut checkpoints_sealed = false;
        let mut leader_only_steps = 0usize;
        loop {
            if checkpoints_sealed {
                checkpoints_sealed = false;
                for l in &mut lanes {
                    if l.active && l.n_past == l.cached && l.blocks.written == l.n_past {
                        if let Some((matched_at, cs)) = cache.rebind_if_deeper_match(
                            &mut pools,
                            &mut l.tbl,
                            &mut l.blocks,
                            &l.toks,
                            l.n_past,
                        ) {
                            l.cs = cs;
                            l.n_past = matched_at;
                            l.cached = matched_at;
                        }
                    }
                }
            }
            let cands: Vec<InflightPrefillLane<'_>> = lanes
                .iter()
                .enumerate()
                .filter(|(_, l)| l.active && l.n_past < l.toks.len())
                .map(|(i, l)| InflightPrefillLane {
                    lane: i,
                    namespace: 0,
                    toks: &l.toks,
                    block_hashes: None,
                    n_past: l.n_past,
                    pf_end: l.toks.len(),
                    runnable: true,
                    eligible_follower: l.n_past == l.cached && l.blocks.written == l.n_past,
                })
                .collect();
            if cands.is_empty() {
                break;
            }
            let mut deferred = vec![false; nlanes];
            for (idx, c) in cands.iter().enumerate() {
                if find_prefill_leader(&cands, idx).is_some() {
                    deferred[c.lane] = true;
                }
            }
            let inputs: Vec<ServeLaneInput> = lanes
                .iter()
                .enumerate()
                .map(|(i, l)| ServeLaneInput {
                    runnable: l.active && l.n_past < l.toks.len() && !deferred[i],
                    n_past: l.n_past,
                    pf_end: effective_pf_target(l.n_past, l.toks.len()),
                    has_pending_decode: false,
                })
                .collect();
            let Some(ServeStepPlan::Prefill(spans)) = planner.plan_and_begin_step(&inputs, chunk, BLOCK) else {
                panic!("expected prefill plan while lanes remain unfinished");
            };
            if lanes[0].n_past < 8192 {
                // Only lane 0 runs for [0..2048..4096..6144..8192), and followers accrue 0 wait_ticks.
                assert_eq!(spans.len(), 1);
                assert_eq!(spans[0].0, 0);
                for f in 1..nlanes {
                    assert_eq!(planner.wait_ticks(f), 0);
                }
                leader_only_steps += 1;
            }
            let before_starts = cache.stats().start_points_saved;
            for (i, p0, t) in spans {
                let l = &mut lanes[i];
                cache.ensure_span_blocks(&mut pools, &mut l.tbl, p0, t).unwrap();
                cache.finish_prefill_span(&mut pools, &mut l.tbl, &mut l.blocks, &l.toks, p0, t, || {
                    fake_cs(((p0 + t) / BLOCK) as u8)
                });
                l.n_past = p0 + t;
                l.prefilled += t;
            }
            if cache.stats().start_points_saved > before_starts {
                checkpoints_sealed = true;
            }
            assert!(planner.complete_step().is_empty());
        }

        assert_eq!(leader_only_steps, 4, "lane 0 should run 4 leader chunks at 2048/4096/6144/8192");
        assert_eq!(lanes[0].cached, 0);
        assert_eq!(lanes[0].prefilled, 8800);
        for (i, lane) in lanes.iter().enumerate().take(nlanes).skip(1) {
            assert_eq!(lane.cached, 8192, "follower {i} should cache up to 8192");
            assert_eq!(lane.prefilled, 608, "follower {i} should prefill only [8192..8800)");
            assert_eq!(lane.cs, fake_cs((8192 / BLOCK) as u8));
        }
        assert!(cache.stats().singleflight_rebinds >= 21);
        assert_eq!(cache.stats().singleflight_saved_tokens, 21 * 8192);

        for l in &mut lanes {
            cache.release_lane(&mut pools, &mut l.tbl, &mut l.blocks);
        }
        assert_eq!(cache.free_plus_evictable_slots(&pools, BlockKind::Full), 512);
        assert_eq!(cache.free_plus_evictable_slots(&pools, BlockKind::Window), 512);
        cache.shutdown(&mut pools);
        assert_zero_pool_leaks(&pools);
    }

    #[test]
    fn singleflight_leader_cancellation_promotes_next_follower_without_leaks() {
        use crate::scheduler::{ServeLaneInput, ServeStepPlan, ServeStepPlanner};

        let mut pools = BlockPools::with_capacities(512, 256, 2).unwrap();
        let mut cache = ServeBlockCache::new();
        let prefix: Vec<u32> = (0..8500u32).map(|i| (i * 19 + 5) % 50000).collect();

        let nlanes = 4usize;
        let mut toks_list = Vec::with_capacity(nlanes);
        let mut tbls = Vec::with_capacity(nlanes);
        let mut blocks = Vec::with_capacity(nlanes);
        let mut n_past = vec![0usize; nlanes];
        let mut cached = vec![0usize; nlanes];
        let mut active = vec![true; nlanes];
        let mut planner = ServeStepPlanner::new(nlanes, 16);

        for i in 0..nlanes {
            let mut toks = prefix.clone();
            toks.extend((0..300u32).map(|j| 70000 + (i as u32) * 1000 + j));
            let mut tbl = BlockTable::new(512).unwrap();
            let mut st = LaneBlockState::default();
            assert!(cache.match_and_bind(&mut pools, &mut tbl, &mut st, &toks).is_none());
            toks_list.push(toks);
            tbls.push(tbl);
            blocks.push(st);
            planner.on_admit(i);
        }

        // Step 1: lane 0 leads [0..2048) and seals checkpoint 2048.
        let inputs: Vec<ServeLaneInput> = (0..nlanes)
            .map(|i| ServeLaneInput {
                runnable: i == 0,
                n_past: 0,
                pf_end: effective_pf_target(0, toks_list[i].len()),
                has_pending_decode: false,
            })
            .collect();
        let Some(ServeStepPlan::Prefill(spans)) = planner.plan_and_begin_step(&inputs, 4096, BLOCK) else {
            panic!("expected step 1 prefill");
        };
        assert_eq!(spans, vec![(0, 0, 2048)]);
        cache.ensure_span_blocks(&mut pools, &mut tbls[0], 0, 2048).unwrap();
        cache.finish_prefill_span(&mut pools, &mut tbls[0], &mut blocks[0], &toks_list[0], 0, 2048, || fake_cs(32));
        n_past[0] = 2048;
        assert!(planner.complete_step().is_empty());

        // Rebind followers 1..4 to 2048, then cancel lane 0 mid-prefill!
        for i in 1..nlanes {
            let (m, _) = cache
                .rebind_if_deeper_match(&mut pools, &mut tbls[i], &mut blocks[i], &toks_list[i], n_past[i])
                .expect("follower must rebind to 2048");
            assert_eq!(m, 2048);
            n_past[i] = m;
            cached[i] = m;
        }
        assert!(planner.request_cancel(0));
        cache.release_lane(&mut pools, &mut tbls[0], &mut blocks[0]);
        active[0] = false;

        // Next step: lane 1 must immediately promote to leader; lanes 2 and 3 must follow lane 1.
        let mut checkpoints_sealed = false;
        loop {
            if checkpoints_sealed {
                checkpoints_sealed = false;
                for i in 0..nlanes {
                    if active[i] && n_past[i] == cached[i] && blocks[i].written == n_past[i] {
                        if let Some((m, _)) = cache.rebind_if_deeper_match(
                            &mut pools,
                            &mut tbls[i],
                            &mut blocks[i],
                            &toks_list[i],
                            n_past[i],
                        ) {
                            n_past[i] = m;
                            cached[i] = m;
                        }
                    }
                }
            }
            let cands: Vec<InflightPrefillLane<'_>> = (0..nlanes)
                .filter(|&i| active[i] && n_past[i] < toks_list[i].len())
                .map(|i| InflightPrefillLane {
                    lane: i,
                    namespace: 0,
                    toks: &toks_list[i],
                    block_hashes: None,
                    n_past: n_past[i],
                    pf_end: toks_list[i].len(),
                    runnable: true,
                    eligible_follower: n_past[i] == cached[i] && blocks[i].written == n_past[i],
                })
                .collect();
            if cands.is_empty() {
                break;
            }
            let mut deferred = vec![false; nlanes];
            for (idx, c) in cands.iter().enumerate() {
                let leader = find_prefill_leader(&cands, idx);
                if c.lane == 1 {
                    assert_eq!(leader, None, "lane 1 must immediately become leader after lane 0 cancels");
                } else if n_past[c.lane] < 8192 {
                    assert_eq!(leader, Some(1), "lanes 2 and 3 must follow promoted lane 1");
                }
                if leader.is_some() {
                    deferred[c.lane] = true;
                }
            }
            let inputs: Vec<ServeLaneInput> = (0..nlanes)
                .map(|i| ServeLaneInput {
                    runnable: active[i] && n_past[i] < toks_list[i].len() && !deferred[i],
                    n_past: n_past[i],
                    pf_end: effective_pf_target(n_past[i], toks_list[i].len()),
                    has_pending_decode: false,
                })
                .collect();
            let Some(ServeStepPlan::Prefill(spans)) = planner.plan_and_begin_step(&inputs, 4096, BLOCK) else {
                panic!("expected prefill after leader promotion");
            };
            let before_starts = cache.stats().start_points_saved;
            for (i, p0, t) in spans {
                cache.ensure_span_blocks(&mut pools, &mut tbls[i], p0, t).unwrap();
                cache.finish_prefill_span(&mut pools, &mut tbls[i], &mut blocks[i], &toks_list[i], p0, t, || fake_cs(1));
                n_past[i] = p0 + t;
            }
            if cache.stats().start_points_saved > before_starts {
                checkpoints_sealed = true;
            }
            assert!(planner.complete_step().is_empty());
        }

        assert_eq!(cached[1], 2048);
        assert_eq!(cached[2], 8192);
        assert_eq!(cached[3], 8192);

        for i in 1..nlanes {
            cache.release_lane(&mut pools, &mut tbls[i], &mut blocks[i]);
        }
        cache.shutdown(&mut pools);
        assert_zero_pool_leaks(&pools);
    }

    #[test]
    fn singleflight_short_or_diverged_prompt_never_waits() {
        let leader_toks: Vec<u32> = (0..8500u32).map(|i| i + 1).collect();
        let short_toks: Vec<u32> = leader_toks[..1500].to_vec();
        let mut diverged_toks = leader_toks.clone();
        diverged_toks[100] ^= 0xffff;

        let cands = vec![
            InflightPrefillLane {
                lane: 0,
                namespace: 0,
                toks: &leader_toks,
                block_hashes: None,
                n_past: 0,
                pf_end: leader_toks.len(),
                runnable: true,
                eligible_follower: true,
            },
            InflightPrefillLane {
                lane: 1,
                namespace: 0,
                toks: &short_toks,
                block_hashes: None,
                n_past: 0,
                pf_end: short_toks.len(),
                runnable: true,
                eligible_follower: true,
            },
            InflightPrefillLane {
                lane: 2,
                namespace: 0,
                toks: &diverged_toks,
                block_hashes: None,
                n_past: 0,
                pf_end: diverged_toks.len(),
                runnable: true,
                eligible_follower: true,
            },
        ];

        assert_eq!(find_prefill_leader(&cands, 0), None);
        assert_eq!(find_prefill_leader(&cands, 1), None, "prompt < EARLY_EVERY_TOKENS must never wait");
        assert_eq!(find_prefill_leader(&cands, 2), None, "diverged prefix must never wait");

        assert_eq!(inflight_shared_checkpoint(&cands[1], &cands[..1]), 0);
        assert_eq!(inflight_shared_checkpoint(&cands[2], &cands[..1]), 0);
        let mut matching = leader_toks.clone();
        matching.extend(90000..90300);
        let longer = InflightPrefillLane { lane: 1, toks: &matching, pf_end: matching.len(), ..cands[0] };
        assert_eq!(inflight_shared_checkpoint(&longer, &cands[..1]), 8192, "only the common periodic boundary is promised");
        let equal = InflightPrefillLane { lane: 1, ..cands[0] };
        assert_eq!(inflight_shared_checkpoint(&equal, &cands[..1]), 8448);
    }

    #[test]
    fn simultaneous_burst_admission_guard_prevents_midflight_exhaustion() {
        // 130 Full slots = 8,320 tokens. Each request needs (2000 + 512).div_ceil(64) = 40 Full slots
        // Six independent requests arriving simultaneously would demand 240 Full
        // slots (> 130) if admitted together; reserved_inflight_full_slots must admit 3 and defer 3!
        let mut pools = BlockPools::with_capacities(130, 128, 1).unwrap();
        let mut cache = ServeBlockCache::new();
        let chunk = 2048usize;
        let max_gen = 512usize;

        struct BurstLane {
            toks: Vec<u32>,
            tbl: BlockTable,
            blocks: LaneBlockState,
            n_past: usize,
            gen: usize,
            active: bool,
        }

        let mut requests: Vec<Vec<u32>> = (0..6u32)
            .map(|r| (0..2000u32).map(|i| (r + 1) * 100_000 + i).collect())
            .collect();
        let mut lanes: Vec<BurstLane> = Vec::new();
        let mut queue: std::collections::VecDeque<Vec<u32>> = requests.drain(..).collect();

        // Wave 1: 6 requests arrive simultaneously at t=0.
        let mut admitted_wave1 = 0usize;
        while let Some(toks) = queue.front() {
            let reserves: Vec<InflightLaneReserve<'_>> = lanes
                .iter()
                .enumerate()
                .filter(|(_, l)| l.active)
                .map(|(idx, l)| InflightLaneReserve {
                    lane: idx,
                    namespace: 0,
                    toks: &l.toks,
                    block_hashes: None,
                    n_past: l.n_past,
                    pf_end: l.toks.len(),
                    cached: 0,
                    written: l.blocks.written,
                    gen: l.gen,
                    max_gen,
                    runnable: true,
                })
                .collect();
            if !cache.can_admit(&pools, &reserves, lanes.len(), toks, max_gen, chunk) {
                break;
            }
            let toks = queue.pop_front().unwrap();
            let mut tbl = BlockTable::new(128).unwrap();
            let mut blocks = LaneBlockState::default();
            assert!(cache.match_and_bind(&mut pools, &mut tbl, &mut blocks, &toks).is_none());
            lanes.push(BurstLane {
                toks,
                tbl,
                blocks,
                n_past: 0,
                gen: 0,
                active: true,
            });
            admitted_wave1 += 1;
        }
        assert_eq!(admitted_wave1, 3, "must admit only the 3 requests that fit inside 130 Full slots");
        assert_eq!(queue.len(), 3, "remaining 3 requests must stay queued, not admit and 500 later");
        assert_eq!(cache.stats().admit_deferred_pool_pressure, 1);

        // Run all admitted requests (prefill + full 512 decode tokens) and admit queued ones as slots free:
        // ZERO PoolExhausted errors across all 6 requests!
        loop {
            let mut progressed = false;
            for l in &mut lanes {
                if !l.active {
                    continue;
                }
                progressed = true;
                if l.n_past < l.toks.len() {
                    let next = effective_pf_target(l.n_past, l.toks.len());
                    cache
                        .ensure_span_blocks(&mut pools, &mut l.tbl, l.n_past, next - l.n_past)
                        .expect("admitted lane must never hit PoolExhausted during prefill");
                    cache.finish_prefill_span(
                        &mut pools,
                        &mut l.tbl,
                        &mut l.blocks,
                        &l.toks,
                        l.n_past,
                        next - l.n_past,
                        || fake_cs(1),
                    );
                    l.n_past = next;
                } else if l.gen < max_gen {
                    cache
                        .ensure_span_blocks(&mut pools, &mut l.tbl, l.n_past, 1)
                        .expect("admitted lane must never hit PoolExhausted during decode");
                    cache.finish_decode_span(&mut pools, &mut l.tbl, &mut l.blocks, l.n_past);
                    l.n_past += 1;
                    l.gen += 1;
                } else {
                    cache.release_lane(&mut pools, &mut l.tbl, &mut l.blocks);
                    l.active = false;
                }
            }
            while let Some(toks) = queue.front() {
                let reserves: Vec<InflightLaneReserve<'_>> = lanes
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| l.active)
                    .map(|(idx, l)| InflightLaneReserve {
                        lane: idx,
                        namespace: 0,
                        toks: &l.toks,
                        block_hashes: None,
                        n_past: l.n_past,
                        pf_end: l.toks.len(),
                        cached: 0,
                        written: l.blocks.written,
                        gen: l.gen,
                        max_gen,
                        runnable: true,
                    })
                    .collect();
                if !cache.can_admit(&pools, &reserves, lanes.len(), toks, max_gen, chunk) {
                    break;
                }
                let toks = queue.pop_front().unwrap();
                let mut tbl = BlockTable::new(128).unwrap();
                let mut blocks = LaneBlockState::default();
                let _ = cache.match_and_bind(&mut pools, &mut tbl, &mut blocks, &toks);
                lanes.push(BurstLane {
                    toks,
                    tbl,
                    blocks,
                    n_past: 0,
                    gen: 0,
                    active: true,
                });
            }
            if !progressed && queue.is_empty() {
                break;
            }
        }
        assert_eq!(lanes.len(), 6);
        assert!(lanes.iter().all(|l| !l.active && l.gen == max_gen));
        cache.shutdown(&mut pools);
        assert_zero_pool_leaks(&pools);
    }

    #[test]
    fn transactional_ensure_span_blocks_and_zero_500_deferral_drain() {
        use crate::scheduler::{LaneState, ServeLaneInput, ServeStepPlan, ServeStepPlanner};

        // 1. Transactional rollback on ensure_span_blocks: 3 Full slots available, request needs 4 blocks.
        let mut pools = BlockPools::with_capacities(3, 8, 1).unwrap();
        let mut cache = ServeBlockCache::new();
        let mut tbl = BlockTable::new(16).unwrap();
        assert_eq!(
            cache.ensure_span_blocks(&mut pools, &mut tbl, 0, 256),
            Err(CacheError::PoolExhausted(BlockKind::Full))
        );
        for b in 0..4u32 {
            let bi = BlockIndex::new(b);
            assert_eq!(tbl.get(BlockKind::Full, bi).unwrap(), None, "block {b} Full must be rolled back");
            assert_eq!(tbl.get(BlockKind::Window, bi).unwrap(), None, "block {b} Window must be rolled back");
        }
        assert_eq!(cache.free_plus_evictable_slots(&pools, BlockKind::Full), 3);
        assert_eq!(cache.free_plus_evictable_slots(&pools, BlockKind::Window), 8);
        assert_zero_pool_leaks(&pools);

        // 2. End-to-end zero-500 deferral: lane 0 is decoding (holding 6 Full slots, 2 more to finish),
        // lane 1 is prefilling (needs 8 Full slots, only 4 free). Total pool = 10 Full slots.
        // Even when planner preempts to Prefill for lane 1 and lane 1 hits PoolExhausted,
        // lane 1 defers (0 HTTP 500s, 0 leaked slots), lane 0 finishes & frees its 6 slots,
        // and lane 1 completes cleanly!
        let mut pools = BlockPools::with_capacities(10, 16, 1).unwrap();
        let mut cache = ServeBlockCache::new();
        let mut planner = ServeStepPlanner::new(2, 2);

        let toks0: Vec<u32> = (0..384u32).collect(); // 6 Full blocks
        let toks1: Vec<u32> = (1000..1512u32).collect(); // 8 Full blocks
        let mut tbls = [BlockTable::new(32).unwrap(), BlockTable::new(32).unwrap()];
        let mut blocks = [LaneBlockState::default(), LaneBlockState::default()];
        let _ = cache.match_and_bind(&mut pools, &mut tbls[0], &mut blocks[0], &toks0);
        cache.ensure_span_blocks(&mut pools, &mut tbls[0], 0, 384).unwrap();
        cache.finish_prefill_span(&mut pools, &mut tbls[0], &mut blocks[0], &toks0, 0, 384, || fake_cs(1));
        let _ = cache.match_and_bind(&mut pools, &mut tbls[1], &mut blocks[1], &toks1);

        planner.on_admit(0);
        planner.on_admit(1);

        let mut n_past = [384usize, 0usize];
        let pf_end = [384usize, 512usize];
        let mut gen = [0usize, 0usize];
        let max_gen = [4usize, 1usize];
        let mut active = [true, true];
        let mut pool_deferred = [false, false];
        let mut prev_released = cache.stats().lanes_released;

        for _step in 0..32 {
            if cache.stats().lanes_released != prev_released {
                pool_deferred.fill(false);
                prev_released = cache.stats().lanes_released;
            }
            if !active[0] && !active[1] {
                break;
            }
            let inputs: Vec<ServeLaneInput> = (0..2)
                .map(|i| ServeLaneInput {
                    runnable: active[i] && !pool_deferred[i],
                    n_past: n_past[i],
                    pf_end: if n_past[i] < pf_end[i] {
                        effective_pf_target(n_past[i], pf_end[i])
                    } else {
                        pf_end[i]
                    },
                    has_pending_decode: active[i] && n_past[i] >= pf_end[i],
                })
                .collect();
            let Some(plan) = planner.plan_and_begin_step(&inputs, 1024, BLOCK) else {
                panic!("deadlock: no runnable step while lanes remain active");
            };
            match plan {
                ServeStepPlan::Prefill(sp_list) => {
                    let mut live_sp = Vec::new();
                    for (i, p0, t) in sp_list {
                        match cache.ensure_span_blocks(&mut pools, &mut tbls[i], p0, t) {
                            Ok(()) => live_sp.push((i, p0, t)),
                            Err(CacheError::PoolExhausted(_)) => {
                                cache.record_step_deferred_pool_pressure();
                                pool_deferred[i] = true;
                            }
                        }
                    }
                    for (i, p0, t) in live_sp {
                        let toks = if i == 0 { &toks0 } else { &toks1 };
                        cache.finish_prefill_span(&mut pools, &mut tbls[i], &mut blocks[i], toks, p0, t, || fake_cs(2));
                        n_past[i] = p0 + t;
                    }
                }
                ServeStepPlan::Decode(need_dec) => {
                    for i in need_dec {
                        cache.ensure_span_blocks(&mut pools, &mut tbls[i], n_past[i], 1).unwrap();
                        cache.finish_decode_span(&mut pools, &mut tbls[i], &mut blocks[i], n_past[i]);
                        n_past[i] += 1;
                        gen[i] += 1;
                        if gen[i] >= max_gen[i] {
                            cache.release_lane(&mut pools, &mut tbls[i], &mut blocks[i]);
                            active[i] = false;
                            planner.on_finish(i);
                        }
                    }
                }
            }
            assert!(planner.complete_step().is_empty());
            for (i, &is_active) in active.iter().enumerate() {
                if is_active {
                    assert_eq!(planner.state(i), LaneState::Ready);
                }
            }
        }

        assert!(!active[0] && !active[1], "both lanes must complete without HTTP 500");
        assert!(cache.stats().step_deferred_pool_pressure >= 1);
        cache.shutdown(&mut pools);
        assert_zero_pool_leaks(&pools);
    }

    struct ScratchDir(std::path::PathBuf);
    impl ScratchDir {
        fn new(name: &str) -> Self {
            let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("target")
                .join("block-cache-spill-tests")
                .join(format!("{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for ScratchDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    // Populate a retained checkpoint at end, leaving its lane free.
    fn retain_test_prompt(
        cache: &mut ServeBlockCache,
        pools: &mut BlockPools,
        tbl: &mut BlockTable,
        state: &mut LaneBlockState,
        tokens: &[u32],
        tag: u8,
    ) -> SealedId {
        let start = cache.match_and_bind(pools, tbl, state, tokens).map_or(0, |(p, _)| p);
        let end = tokens.len() - 1;
        cache.ensure_span_blocks(pools, tbl, start, end - start).unwrap();
        cache.finish_prefill_span(pools, tbl, state, tokens, start, end - start, || fake_cs(tag));
        let tip = *state.sealed.last().unwrap();
        cache.release_lane(pools, tbl, state);
        tip
    }

    #[test]
    fn long_spill_restore_keeps_source_and_accepts_early_fork() {
        let dir = ScratchDir::new("long-fork");
        let store = SpillStore::open(&dir.0, 100 * 1024 * 1024).unwrap();
        let mut cache = ServeBlockCache::with_spill(store);
        let mut pools = BlockPools::with_capacities(18, 18, 1).unwrap();
        let mut tbl = BlockTable::new(20).unwrap();
        let mut state = LaneBlockState::default();
        let end = WINDOW + BLOCK;
        let long: Vec<u32> = (0..=end as u32).collect();
        let tip = retain_test_prompt(&mut cache, &mut pools, &mut tbl, &mut state, &long, 3);
        assert!(cache.evict_start(&mut pools, tip));
        // Two unrelated resident blocks leave only 16 free Full slots; the restore
        // needs 17. Writing these victims to the tight SSD must preserve the source.
        let unrelated: Vec<u32> = (0..=128).map(|i| 20_000 + i).collect();
        retain_test_prompt(&mut cache, &mut pools, &mut tbl, &mut state, &unrelated, 4);
        assert_eq!(cache.match_and_bind(&mut pools, &mut tbl, &mut state, &long), Some((end, fake_cs(3))));
        let root = state.sealed[0];
        let root_full = cache.tree.handle(root).unwrap().full();
        assert!(cache.tree.handle(root).unwrap().window().is_none());
        cache.release_lane(&mut pools, &mut tbl, &mut state);

        let mut fork = long[..BLOCK].to_vec();
        fork.push(99_000);
        assert!(cache.match_and_bind(&mut pools, &mut tbl, &mut state, &fork).is_none());
        cache.ensure_span_blocks(&mut pools, &mut tbl, 0, BLOCK).unwrap();
        let recomputed_window = tbl.get(BlockKind::Window, BlockIndex::new(0)).unwrap().unwrap();
        cache.finish_prefill_span(&mut pools, &mut tbl, &mut state, &fork, 0, BLOCK, || fake_cs(5));
        assert_eq!(state.sealed[0], root);
        assert_eq!(cache.tree.handle(root).unwrap(), BlockHandles::new(root_full, Some(recomputed_window)));
        assert_eq!(tbl.get(BlockKind::Full, BlockIndex::new(0)).unwrap(), Some(root_full));
        cache.ensure_span_blocks(&mut pools, &mut tbl, BLOCK, 1).unwrap();
        cache.finish_prefill_span(&mut pools, &mut tbl, &mut state, &fork, BLOCK, 1, || fake_cs(6));
        cache.release_lane(&mut pools, &mut tbl, &mut state);
        cache.shutdown(&mut pools);
        assert_zero_pool_leaks(&pools);
    }

    #[test]
    fn spill_read_failure_releases_provisional_blocks_and_preserves_resident_match() {
        let dir = ScratchDir::new("stream-rollback");
        let store = SpillStore::open(&dir.0, 64 * 1024 * 1024).unwrap();
        let mut cache = ServeBlockCache::with_spill(store);
        let mut pools = BlockPools::with_capacities(16, 16, 1).unwrap();
        let mut tbl = BlockTable::new(16).unwrap();
        let mut state = LaneBlockState::default();
        let long: Vec<u32> = (0..=6 * BLOCK as u32).collect();
        let short = &long[..BLOCK + 1];
        retain_test_prompt(&mut cache, &mut pools, &mut tbl, &mut state, short, 1);
        let tip = retain_test_prompt(&mut cache, &mut pools, &mut tbl, &mut state, &long, 2);
        assert!(cache.evict_start(&mut pools, tip));

        let checkpoint = cache.spilled_starts.values().next().unwrap();
        let (_, _, last_file) = *checkpoint.entries.last().unwrap();
        std::fs::remove_file(cache.spill_store().unwrap().block_path(last_file)).unwrap();
        let before = cache.stats().slots_taken;
        assert_eq!(cache.match_and_bind(&mut pools, &mut tbl, &mut state, &long), Some((BLOCK, fake_cs(1))));
        assert!(cache.stats().slots_taken > before, "earlier files used bounded provisional slots");
        assert_eq!(cache.stats().starts_restored, 0);
        assert_eq!(cache.retained_start_count(), 1);
        assert_eq!(cache.spilled_start_count(), 0);
        cache.release_lane(&mut pools, &mut tbl, &mut state);
        cache.shutdown(&mut pools);
        assert_zero_pool_leaks(&pools);
    }

    #[test]
    fn spill_store_eviction_restore_and_corrupt_fallback() {
        let dir = ScratchDir::new("e2e");
        let store = SpillStore::open_with_kvb(&dir.0, 256 * 1024 * 1024, 1).unwrap();
        // 24 Full slots, 16 Window slots at kvb=1 (1 MiB Full, 5 MiB Window).
        // Each 576-token prompt saves a PROMPT checkpoint at 512 (8 Full + 8 Window).
        // Three distinct prompts demand 24 Window slots (> 16), forcing LRU eviction + SSD spill of prompt 0!
        let mut pools = BlockPools::with_capacities(24, 16, 1).unwrap();
        let mut cache = ServeBlockCache::with_spill(store);
        let mut tbl = BlockTable::new(32).unwrap();
        let mut st = LaneBlockState::default();

        let prompt0: Vec<u32> = (0..576u32).map(|i| 10_000 + i).collect();
        let prompt1: Vec<u32> = (0..576u32).map(|i| 20_000 + i).collect();
        let prompt2: Vec<u32> = (0..576u32).map(|i| 30_000 + i).collect();

        let run_prompt = |cache: &mut ServeBlockCache,
                          pools: &mut BlockPools,
                          tbl: &mut BlockTable,
                          st: &mut LaneBlockState,
                          toks: &[u32],
                          tag: u8| {
            assert!(cache.match_and_bind(pools, tbl, st, toks).is_none());
            let mut p = 0;
            while p < toks.len() {
                let next = effective_pf_target(p, toks.len());
                cache.ensure_span_blocks(pools, tbl, p, next - p).unwrap();
                // Stamp recognizable sentinel pattern into each newly allocated slot before sealing.
                for b in (p / BLOCK)..=((next - 1) / BLOCK) {
                    let bi = BlockIndex::new(b as u32);
                    for kind in KINDS {
                        let e = tbl.get(kind, bi).unwrap().unwrap();
                        if !pools.of(kind).record(e.slot()).unwrap().sealed() {
                            let bytes = pools.of_mut(kind).slot_bytes_mut(e.slot()).unwrap();
                            bytes[0] = tag;
                            bytes[1] = b as u8;
                            bytes[bytes.len() - 1] = tag.wrapping_add(b as u8);
                        }
                    }
                }
                cache.finish_prefill_span(pools, tbl, st, toks, p, next - p, || fake_cs(tag));
                p = next;
            }
            cache.release_lane(pools, tbl, st);
        };

        run_prompt(&mut cache, &mut pools, &mut tbl, &mut st, &prompt0, 0xA1);
        run_prompt(&mut cache, &mut pools, &mut tbl, &mut st, &prompt1, 0xB2);
        run_prompt(&mut cache, &mut pools, &mut tbl, &mut st, &prompt2, 0xC3);

        assert!(cache.stats().starts_spilled >= 1, "LRU eviction must spill prompt0 to SpillStore");
        assert!(cache.spilled_start_count() >= 1);
        assert!(cache.spill_store().unwrap().entry_count() > 0);
        // prompt0 is no longer in RAM: peek_match_len stays in-RAM only.
        let mut prompt0_ext = prompt0[..512].to_vec();
        prompt0_ext.extend(99_000..99_040);
        assert_eq!(cache.peek_match_len(&pools, &prompt0_ext), 0);

        // match_and_bind must restore prompt0 from SSD, verify SHA-256, and return exact cs + KV bytes!
        let (matched_at, cs) = cache
            .match_and_bind(&mut pools, &mut tbl, &mut st, &prompt0_ext)
            .expect("must restore spilled prompt0 checkpoint from SSD");
        assert_eq!(matched_at, 512);
        assert_eq!(cs, fake_cs(0xA1));
        assert_eq!(cache.stats().starts_restored, 1);
        for b in 0..8usize {
            let bi = BlockIndex::new(b as u32);
            for kind in KINDS {
                let e = tbl.get(kind, bi).unwrap().unwrap();
                let bytes = pools.of(kind).slot_bytes(e.slot()).unwrap();
                assert_eq!(bytes[0], 0xA1);
                assert_eq!(bytes[1], b as u8);
                assert_eq!(bytes[bytes.len() - 1], 0xA1u8.wrapping_add(b as u8));
            }
        }
        cache.release_lane(&mut pools, &mut tbl, &mut st);

        // Corrupt-file fallback: evict prompt1 to SSD, corrupt one .blk file, verify clean cache miss!
        let prompt3: Vec<u32> = (0..576u32).map(|i| 40_000 + i).collect();
        run_prompt(&mut cache, &mut pools, &mut tbl, &mut st, &prompt3, 0xD4);
        assert!(cache.spilled_start_count() >= 1);

        let mut corrupted_one = false;
        for entry in std::fs::read_dir(&dir.0).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) == Some("blk") {
                let mut data = std::fs::read(&path).unwrap();
                let last = data.len() - 1;
                data[last] ^= 0xff;
                std::fs::write(&path, data).unwrap();
                corrupted_one = true;
                break;
            }
        }
        assert!(corrupted_one);

        let mut prompt1_ext = prompt1[..512].to_vec();
        prompt1_ext.extend(88_000..88_040);
        // Regardless of which spilled file was corrupted, trying prompt1_ext and prompt2_ext must
        // never panic, must fall back to None for the corrupted checkpoint, and must unlink the corrupt file!
        let m1 = cache.match_and_bind(&mut pools, &mut tbl, &mut st, &prompt1_ext);
        cache.release_lane(&mut pools, &mut tbl, &mut st);
        let mut prompt2_ext = prompt2[..512].to_vec();
        prompt2_ext.extend(77_000..77_040);
        let m2 = cache.match_and_bind(&mut pools, &mut tbl, &mut st, &prompt2_ext);
        cache.release_lane(&mut pools, &mut tbl, &mut st);
        assert!(m1.is_none() || m2.is_none(), "corrupted .blk must fall back to clean cache miss");
        assert!(cache.spill_store().unwrap().stats().corrupt_files_removed >= 1);

        cache.shutdown(&mut pools);
        assert_zero_pool_leaks(&pools);
    }

    #[test]
    fn maintain_spill_during_gpu_trims_excess_spilled_starts_without_touching_pools() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("block-cache-maint-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = SpillStore::open_with_kvb(&dir, 256 * 1024 * 1024, 1).unwrap();
        let mut pools = BlockPools::with_capacities(16, 16, 1).unwrap();
        let mut cache = ServeBlockCache::with_spill(store);
        let mut tbl = BlockTable::new(16).unwrap();
        let mut st = LaneBlockState::default();

        for base in [1000u32, 2000u32, 3000u32] {
            let p: Vec<u32> = (0..576u32).map(|i| base + i).collect();
            assert!(cache.match_and_bind(&mut pools, &mut tbl, &mut st, &p).is_none());
            let mut pos = 0;
            while pos < p.len() {
                let next = effective_pf_target(pos, p.len());
                cache.ensure_span_blocks(&mut pools, &mut tbl, pos, next - pos).unwrap();
                cache.finish_prefill_span(&mut pools, &mut tbl, &mut st, &p, pos, next - pos, || fake_cs((base / 1000) as u8));
                pos = next;
            }
            cache.release_lane(&mut pools, &mut tbl, &mut st);
        }
        assert!(cache.spilled_start_count() >= 1);
        let used_slots = |p: &crate::blocks::BlockPool| {
            (1..=p.len())
                .filter(|&i| p.record(Slot::new(i as u32).unwrap()).unwrap().refs() > 0)
                .count()
        };
        let used_before = used_slots(&pools.full);

        cache.max_spilled_starts = 0;
        let trimmed = cache.maintain_spill_during_gpu();
        assert!(trimmed >= 1);
        assert_eq!(cache.spilled_start_count(), 0);
        assert_eq!(cache.spill_store().unwrap().entry_count(), 0);
        assert_eq!(cache.stats().spill_maint_passes, 1);
        assert_eq!(used_slots(&pools.full), used_before, "RAM pool slots must not be mutated during GPU overlap");

        cache.shutdown(&mut pools);
        assert_zero_pool_leaks(&pools);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn doomed_allocation_and_mid_span_rollback_preserve_cached_starts_without_spill() {
        let dir = ScratchDir::new("u105-bounded-evict");
        let store = SpillStore::open_with_kvb(&dir.0, 64 * 1024 * 1024, 1).unwrap();
        // 10 Full slots, 16 Window slots.
        let mut pools = BlockPools::with_capacities(10, 16, 1).unwrap();
        let mut cache = ServeBlockCache::with_spill(store);
        let mut tbl_active = BlockTable::new(32).unwrap();
        let mut st_active = LaneBlockState::default();
        let mut tbl = BlockTable::new(32).unwrap();
        let mut st = LaneBlockState::default();

        // Populate 2 distinct cached start points of 3 blocks (192 tokens) each -> 6 evictable Full slots.
        let p_a: Vec<u32> = (0..=192u32).map(|i| 1_000 + i).collect();
        let p_b: Vec<u32> = (0..=192u32).map(|i| 2_000 + i).collect();
        retain_test_prompt(&mut cache, &mut pools, &mut tbl, &mut st, &p_a, 0x11);
        retain_test_prompt(&mut cache, &mut pools, &mut tbl, &mut st, &p_b, 0x22);
        assert_eq!(cache.retained_start_count(), 2);

        // Pin 2 Full slots on an active lane -> 2 free + 6 evictable = 8 reclaimable Full slots (K - 1 = 8).
        cache.ensure_span_blocks(&mut pools, &mut tbl_active, 0, 128).unwrap();
        assert_eq!(cache.free_plus_evictable_slots(&pools, BlockKind::Full), 8);
        assert_eq!(cache.stats().start_points_evicted, 0);
        assert_eq!(cache.stats().starts_spilled, 0);

        // (a) Request needing K = 9 Full slots when free + evictable == 8 (K - 1) MUST fail with
        // PoolExhausted(Full) while evicting and spilling ZERO cached start points!
        let mut doomed_tbl = BlockTable::new(32).unwrap();
        assert_eq!(
            cache.ensure_span_blocks(&mut pools, &mut doomed_tbl, 0, 9 * BLOCK),
            Err(CacheError::PoolExhausted(BlockKind::Full))
        );
        assert_eq!(cache.retained_start_count(), 2, "doomed span must not empty cached start points");
        assert_eq!(cache.stats().start_points_evicted, 0, "zero start points evicted on doomed span");
        assert_eq!(cache.stats().starts_spilled, 0, "zero start points spilled on doomed span");
        assert_eq!(cache.spilled_start_count(), 0);
        assert_eq!(cache.peek_match_len(&pools, &p_a), 192);
        assert_eq!(cache.peek_match_len(&pools, &p_b), 192);

        // Also test single-slot take_zeroed_slot when remaining slots are 100% pinned by active lane:
        // pin the remaining 2 free slots, plus pin p_a and p_b on active tables so free + evictable == 0.
        let mut pin_a_tbl = BlockTable::new(32).unwrap();
        let mut pin_a_st = LaneBlockState::default();
        let mut pin_b_tbl = BlockTable::new(32).unwrap();
        let mut pin_b_st = LaneBlockState::default();
        assert_eq!(cache.match_and_bind(&mut pools, &mut pin_a_tbl, &mut pin_a_st, &p_a).unwrap().0, 192);
        assert_eq!(cache.match_and_bind(&mut pools, &mut pin_b_tbl, &mut pin_b_st, &p_b).unwrap().0, 192);
        cache.ensure_span_blocks(&mut pools, &mut tbl_active, 128, 128).unwrap();
        assert_eq!(cache.free_plus_evictable_slots(&pools, BlockKind::Full), 0);
        assert_eq!(
            cache.take_zeroed_slot(&mut pools, BlockKind::Full),
            Err(CacheError::PoolExhausted(BlockKind::Full))
        );
        assert_eq!(cache.retained_start_count(), 2, "active-pinned starts must not be evicted on exhaustion");
        assert_eq!(cache.stats().start_points_evicted, 0);
        cache.release_lane(&mut pools, &mut pin_a_tbl, &mut pin_a_st);
        cache.release_lane(&mut pools, &mut pin_b_tbl, &mut pin_b_st);

        // Release 2 slots on tbl_active so 2 free + 6 evictable = 8 reclaimable Full slots again.
        cache.release_lane(&mut pools, &mut tbl_active, &mut st_active);
        cache.ensure_span_blocks(&mut pools, &mut tbl_active, 0, 128).unwrap();
        assert_eq!(cache.free_plus_evictable_slots(&pools, BlockKind::Full), 8);

        // (b1) Satisfiable request needing 4 Full slots (2 free + 2 from oldest victim p_a) evicts ONLY p_a
        // and keeps newer p_b intact in RAM!
        // First test mid-span rollback via `try_ensure_span_blocks_with` aborting at block 2 (after 2 free
        // slots were taken, right before victim eviction): cache stays at 2 retained starts and 0 leaks!
        let mut rb_tbl = BlockTable::new(32).unwrap();
        assert_eq!(
            cache.try_ensure_span_blocks_with(&mut pools, &mut rb_tbl, 0, 4 * BLOCK, |blk, kind| {
                if blk == 2 && kind == BlockKind::Full {
                    Err(CacheError::PoolExhausted(BlockKind::Full))
                } else {
                    Ok(())
                }
            }),
            Err(CacheError::PoolExhausted(BlockKind::Full))
        );
        for b in 0..4u32 {
            assert_eq!(rb_tbl.get(BlockKind::Full, BlockIndex::new(b)).unwrap(), None);
            assert_eq!(rb_tbl.get(BlockKind::Window, BlockIndex::new(b)).unwrap(), None);
        }
        assert_eq!(cache.retained_start_count(), 2, "rolled-back span must leave both cached starts intact");
        assert_eq!(cache.stats().start_points_evicted, 0);
        assert_eq!(cache.stats().starts_spilled, 0);

        // (b2) Now run the 4-block allocation to completion: only oldest victim p_a is evicted (1 eviction),
        // leaving p_b warm in RAM!
        cache.ensure_span_blocks(&mut pools, &mut rb_tbl, 0, 4 * BLOCK).unwrap();
        assert_eq!(cache.retained_start_count(), 1, "only the 1 needed LRU victim may be evicted");
        assert_eq!(cache.stats().start_points_evicted, 1);
        assert_eq!(cache.peek_match_len(&pools, &p_b), 192, "newer start p_b must remain warm in RAM");

        cache.release_lane(&mut pools, &mut rb_tbl, &mut st);
        cache.release_lane(&mut pools, &mut tbl_active, &mut st_active);
        cache.shutdown(&mut pools);
        assert_zero_pool_leaks(&pools);
    }

    #[test]
    fn failed_rebind_leaves_lane_prior_bindings_intact() {
        let mut pools = BlockPools::with_capacities(32, 32, 1).unwrap();
        let mut cache = ServeBlockCache::new();
        // `tbl` has capacity for 2 blocks (128 tokens), while `tbl2` has capacity for 8 blocks (512 tokens).
        let mut tbl = BlockTable::new(2).unwrap();
        let mut st = LaneBlockState::default();

        let full_prompt: Vec<u32> = (0..=256u32).map(|i| 5000 + i).collect();
        let short_prompt = &full_prompt[..=128]; // checkpoint at 128 (`len - 1 == 128`)
        let _tip128 = retain_test_prompt(&mut cache, &mut pools, &mut tbl, &mut st, short_prompt, 0x11);

        // Bind Lane 1 to the 128-token checkpoint first.
        let (matched, cs) = cache.match_and_bind(&mut pools, &mut tbl, &mut st, &full_prompt).unwrap();
        assert_eq!(matched, 128);
        assert_eq!(cs, fake_cs(0x11));
        let prior_b0_full = tbl.get(BlockKind::Full, BlockIndex::new(0)).unwrap().unwrap();
        let prior_b1_full = tbl.get(BlockKind::Full, BlockIndex::new(1)).unwrap().unwrap();
        let prior_sealed = st.sealed.clone();

        // Install a live deeper start point at 256 (4 blocks) on Lane 2 (`tbl2` capacity 8).
        let mut tbl2 = BlockTable::new(8).unwrap();
        let mut st2 = LaneBlockState::default();
        let tip256 = retain_test_prompt(&mut cache, &mut pools, &mut tbl2, &mut st2, &full_prompt, 0x33);

        // 1. Mid-staging failure: `tbl` (capacity 2) cannot stage block index 2 of the 256-token start!
        // `rebind_if_deeper_match` must roll back its staged entries and return `None` while keeping
        // Lane 1's existing 128-token bindings, `written == 128`, and refcounts 100% intact!
        assert_eq!(
            cache.rebind_if_deeper_match(&mut pools, &mut tbl, &mut st, &full_prompt, 128),
            None
        );
        assert_eq!(st.written, 128);
        assert_eq!(st.sealed, prior_sealed);
        assert_eq!(tbl.get(BlockKind::Full, BlockIndex::new(0)).unwrap(), Some(prior_b0_full));
        assert_eq!(tbl.get(BlockKind::Full, BlockIndex::new(1)).unwrap(), Some(prior_b1_full));

        // 2. Stale/corrupted deeper candidate in `starts` on a wide table (`tbl_wide` capacity 8):
        cache.release_lane(&mut pools, &mut tbl, &mut st);
        let mut tbl_wide = BlockTable::new(8).unwrap();
        assert!(cache.evict_start(&mut pools, tip256));
        let (matched_w, _) = cache.match_and_bind(&mut pools, &mut tbl_wide, &mut st, &full_prompt).unwrap();
        assert_eq!(matched_w, 128);
        let tip256_stale = retain_test_prompt(&mut cache, &mut pools, &mut tbl2, &mut st2, &full_prompt, 0x44);
        // Simulate a stale entry in `starts` whose pool ref was dropped and generation advanced:
        {
            let bad = cache.starts.get_mut(&tip256_stale).unwrap().entries.last_mut().unwrap();
            pools.of_mut(bad.0).record_mut(bad.2.slot()).unwrap().drop_ref().unwrap();
            bad.2 = BlockEntry::new(bad.2.slot(), bad.2.generation() + 99, bad.2.offset(), bad.0, bad.2.kvb());
        }
        assert_eq!(
            cache.rebind_if_deeper_match(&mut pools, &mut tbl_wide, &mut st, &full_prompt, 128),
            None
        );
        assert_eq!(st.written, 128);
        assert_eq!(st.sealed, prior_sealed);
        assert_eq!(tbl_wide.get(BlockKind::Full, BlockIndex::new(0)).unwrap(), Some(prior_b0_full));
        assert_eq!(tbl_wide.get(BlockKind::Full, BlockIndex::new(1)).unwrap(), Some(prior_b1_full));

        cache.release_lane(&mut pools, &mut tbl_wide, &mut st);
        cache.shutdown(&mut pools);
        assert_zero_pool_leaks(&pools);
    }

    #[test]
    fn spill_retained_start_owns_reused_disk_blocks_across_trim_and_same_tip_overwrite() {
        let dir = ScratchDir::new("reused-disk-ownership");
        let store = SpillStore::open_with_kvb(&dir.0, 256 * 1024 * 1024, 1).unwrap();
        let mut pools = BlockPools::with_capacities(32, 32, 1).unwrap();
        let mut cache = ServeBlockCache::with_spill(store);
        cache.max_spilled_starts = 1;
        let mut tbl = BlockTable::new(32).unwrap();
        let mut st = LaneBlockState::default();

        // Prompt A (checkpoint at 128 = 2 blocks) and Prompt B (shares block 0 with A, checkpoint at 192 = 3 blocks).
        let prompt_a: Vec<u32> = (0..=128u32).map(|i| 70_000 + i).collect();
        let mut prompt_b: Vec<u32> = prompt_a[..64].to_vec();
        prompt_b.extend((0..=128u32).map(|i| 80_000 + i));

        let tip_a = retain_test_prompt(&mut cache, &mut pools, &mut tbl, &mut st, &prompt_a, 0x41);
        let tip_b = retain_test_prompt(&mut cache, &mut pools, &mut tbl, &mut st, &prompt_b, 0x42);

        // Spill A first (`spilled_starts.len() == 1 == max_spilled_starts`).
        assert!(cache.evict_start(&mut pools, tip_a));
        assert_eq!(cache.spilled_start_count(), 1);

        // Now spill B: `max_spilled_starts == 1` trims A at the start of `spill_retained_start`,
        // while B reuses A's shared block-0 Full/Window `.blk` files from `entry_spill_keys`.
        // Because B claims ownership of reused keys from the start, trimming A must NOT delete block 0!
        assert!(cache.evict_start(&mut pools, tip_b));
        assert_eq!(cache.spilled_start_count(), 1);

        let (matched_b, cs_b) = cache
            .match_and_bind(&mut pools, &mut tbl, &mut st, &prompt_b)
            .expect("Prompt B must restore cleanly including shared reused block 0");
        assert_eq!(matched_b, 192);
        assert_eq!(cs_b, fake_cs(0x42));

        // Same-tip overwrite: keep `tbl` pinned to B's live blocks (0..3) while evicting the restored
        // start back to `spilled_starts`, then re-retain the same 192-token prefix on `tbl2` and spill it
        // again at the exact same `sp_tip` while `spilled_starts` ALREADY holds `sp_tip`!
        cache.max_spilled_starts = 4;
        let tip_b_restored = *st.sealed.last().unwrap();
        assert!(cache.evict_start(&mut pools, tip_b_restored));
        assert_eq!(cache.spilled_start_count(), 1);

        let mut tbl2 = BlockTable::new(32).unwrap();
        let mut st2 = LaneBlockState { namespace: 0, prompt_len: prompt_b.len(), written: 0, sealed: Vec::new() };
        cache.ensure_span_blocks(&mut pools, &mut tbl2, 0, 192).unwrap();
        cache.finish_prefill_span(&mut pools, &mut tbl2, &mut st2, &prompt_b, 0, 192, || fake_cs(0x43));
        let tip_b_dup = *st2.sealed.last().unwrap();
        cache.release_lane(&mut pools, &mut tbl2, &mut st2);
        cache.release_lane(&mut pools, &mut tbl, &mut st);

        // Evicting `tip_b_dup` overwrites the existing `sp_tip` in `tip_to_spill` while reusing all 6 `.blk` files!
        assert!(cache.evict_start(&mut pools, tip_b_dup));
        assert_eq!(cache.spilled_start_count(), 1);
        let (matched_b3, cs_b3) = cache
            .match_and_bind(&mut pools, &mut tbl, &mut st, &prompt_b)
            .expect("same-tip re-spilled start must retain its reused disk blocks");
        assert_eq!(matched_b3, 192);
        assert_eq!(cs_b3, fake_cs(0x43));
        cache.release_lane(&mut pools, &mut tbl, &mut st);

        cache.shutdown(&mut pools);
        assert_zero_pool_leaks(&pools);
    }

    #[test]
    fn spill_index_persists_across_reopen_and_async_restore_keeps_decoding_responsive() {
        let dir = ScratchDir::new("persist-and-async");
        {
            let store = SpillStore::open_with_kvb(&dir.0, 128 * 1024 * 1024, 1).unwrap();
            let mut pools = BlockPools::with_capacities(16, 16, 1).unwrap();
            let mut cache = ServeBlockCache::with_spill(store);
            let mut tbl = BlockTable::new(32).unwrap();
            let mut st = LaneBlockState::default();

            let p0: Vec<u32> = (0..=512u32).map(|i| 10_000 + i).collect();
            let tip0 = retain_test_prompt(&mut cache, &mut pools, &mut tbl, &mut st, &p0, 0x5A);
            assert!(cache.evict_start(&mut pools, tip0));
            assert_eq!(cache.spilled_start_count(), 1);
            cache.shutdown(&mut pools);
            assert_zero_pool_leaks(&pools);
        }

        // Reopen SpillStore + ServeBlockCache from the same directory (simulating process restart).
        let store = SpillStore::open_with_kvb(&dir.0, 128 * 1024 * 1024, 1).unwrap();
        let mut pools = BlockPools::with_capacities(24, 24, 1).unwrap();
        let mut cache = ServeBlockCache::with_spill(store);
        assert_eq!(cache.spilled_start_count(), 1, "spilled start index must reload across restart");

        // Install a slow disk read barrier on this spill dir so lane 1 stays in `Restoring`
        // while unrelated Ready lane 0 executes 8 responsive decode steps in < 5 ms per step!
        let target_dir = dir.0.clone();
        let entered = Arc::new((Mutex::new(false), Condvar::new()));
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let entered_bg = entered.clone();
        let release_bg = release.clone();
        struct ReadHookGuard(Vec<Arc<(Mutex<bool>, Condvar)>>);
        impl Drop for ReadHookGuard {
            fn drop(&mut self) {
                crate::spill::set_test_spill_read_hook(None);
                for b in &self.0 {
                    *b.0.lock().unwrap() = true;
                    b.1.notify_all();
                }
            }
        }
        let hold = Arc::new((Mutex::new(false), Condvar::new()));
        let _hook_guard = ReadHookGuard(vec![release.clone(), hold.clone()]);
        crate::spill::set_test_spill_read_hook(Some(Arc::new(move |d, _key| {
            if d == target_dir {
                let mut e = entered_bg.0.lock().unwrap();
                if !*e {
                    *e = true;
                    entered_bg.1.notify_all();
                    drop(e);
                    let mut r = release_bg.0.lock().unwrap();
                    while !*r {
                        r = release_bg.1.wait(r).unwrap();
                    }
                }
            }
        })));

        let mut planner = crate::scheduler::ServeStepPlanner::new(3, 16);
        // Lane 0: Ready decoding lane at pos = 64.
        let mut tbl0 = BlockTable::new(32).unwrap();
        let mut st0 = LaneBlockState::default();
        let p_dec: Vec<u32> = (0..=64u32).map(|i| 90_000 + i).collect();
        retain_test_prompt(&mut cache, &mut pools, &mut tbl0, &mut st0, &p_dec, 0x33);
        let _ = cache.match_and_bind(&mut pools, &mut tbl0, &mut st0, &p_dec);
        planner.on_admit(0);

        // Lane 1: Cold SSD restore after restart (`begin_admit_bind` -> `Restoring`).
        let mut tbl1 = BlockTable::new(32).unwrap();
        let mut st1 = LaneBlockState::default();
        let mut p0_cont: Vec<u32> = (0..=512u32).map(|i| 10_000 + i).collect();
        p0_cont.extend(70_000..70_020);
        match cache.begin_admit_bind(&mut pools, 1, &mut tbl1, &mut st1, &p0_cont) {
            AdmitBindOutcome::Restoring { target_pos } => {
                assert_eq!(target_pos, 512);
                planner.admit_queued(1, crate::scheduler::LaneState::Queued);
                planner.start_restore(1);
            }
            other => panic!("expected Restoring from reloaded spill index, got {other:?}"),
        }

        // Wait until background restore worker is actively blocked inside disk read.
        {
            let mut e = entered.0.lock().unwrap();
            while !*e {
                let (next, timeout) = entered.1.wait_timeout(e, Duration::from_secs(5)).unwrap();
                assert!(!timeout.timed_out(), "background restore worker did not enter read");
                e = next;
            }
        }

        // Run 8 decode steps on Ready lane 0 while lane 1 is blocked in cold SSD read!
        let mut max_step_overhead = Duration::ZERO;
        for dec_pos in 64usize..72 {
            let t0 = Instant::now();
            let done = cache.poll_restores(&mut pools, |_| None);
            assert!(done.is_empty(), "restore is still held on barrier");
            let inputs = [
                crate::scheduler::ServeLaneInput { runnable: true, n_past: dec_pos, pf_end: 64, has_pending_decode: true },
                crate::scheduler::ServeLaneInput { runnable: true, n_past: 0, pf_end: p0_cont.len(), has_pending_decode: false },
                crate::scheduler::ServeLaneInput { runnable: false, n_past: 0, pf_end: 0, has_pending_decode: false },
            ];
            let plan = planner.plan_and_begin_step(&inputs, 2048, BLOCK).unwrap();
            assert_eq!(plan, crate::scheduler::ServeStepPlan::Decode(vec![0]));
            cache.ensure_span_blocks(&mut pools, &mut tbl0, dec_pos, 1).unwrap();
            cache.finish_decode_span(&mut pools, &mut tbl0, &mut st0, dec_pos);
            assert!(planner.complete_step().is_empty());
            max_step_overhead = max_step_overhead.max(t0.elapsed());
        }
        assert!(
            max_step_overhead < Duration::from_millis(5),
            "scheduler decode step overhead {max_step_overhead:?} exceeded 5ms while cold SSD restore ran"
        );
        eprintln!(
            "[async-spill-progress] 8 unrelated decode steps completed while cold SSD restore held in disk read (max_step_overhead={max_step_overhead:?})"
        );

        // Release barrier and verify lane 1 completes restore cleanly (`Restoring -> Ready` at 512).
        {
            *release.0.lock().unwrap() = true;
            release.1.notify_all();
        }
        crate::spill::set_test_spill_read_hook(None);

        let deadline = Instant::now() + Duration::from_secs(5);
        let mut finished = None;
        while Instant::now() < deadline {
            let tbl1_ptr: *mut BlockTable = &mut tbl1;
            let st1_ptr: *mut LaneBlockState = &mut st1;
            let mut done = cache.poll_restores(&mut pools, |l| {
                (l == 1).then_some((&(), tbl1_ptr, st1_ptr))
            });
            if let Some(c) = done.pop() {
                finished = Some(c);
                break;
            }
            cache.wait_restore_activity(Duration::from_millis(5));
        }
        let c = finished.expect("restore must finish after barrier release");
        assert!(!c.corrupt_fallback);
        assert_eq!(c.matched, Some((512, fake_cs(0x5A))));
        planner.finish_restore(1);
        assert_eq!(planner.state(1), crate::scheduler::LaneState::Ready);
        assert_eq!(cache.stats().starts_restored, 1);

        // Also verify mid-restore client disconnect cancels cleanly with zero leaked BlockPool slots!
        let p2: Vec<u32> = (0..=256u32).map(|i| 40_000 + i).collect();
        let mut tbl2 = BlockTable::new(32).unwrap();
        let mut st2 = LaneBlockState::default();
        let tip2 = retain_test_prompt(&mut cache, &mut pools, &mut tbl2, &mut st2, &p2, 0x77);
        assert!(cache.evict_start(&mut pools, tip2));

        let hold_bg = hold.clone();
        let dir2 = dir.0.clone();
        crate::spill::set_test_spill_read_hook(Some(Arc::new(move |d, _| {
            if d == dir2 {
                let mut h = hold_bg.0.lock().unwrap();
                while !*h {
                    h = hold_bg.1.wait(h).unwrap();
                }
            }
        })));
        let mut p2_cont = p2.clone();
        p2_cont.push(999);
        assert!(matches!(
            cache.begin_admit_bind(&mut pools, 2, &mut tbl2, &mut st2, &p2_cont),
            AdmitBindOutcome::Restoring { target_pos: 256 }
        ));
        planner.admit_queued(2, crate::scheduler::LaneState::Queued);
        planner.start_restore(2);
        // Disconnect mid-restore!
        assert!(planner.request_cancel(2));
        assert!(cache.cancel_restore(&mut pools, 2));
        *hold.0.lock().unwrap() = true;
        hold.1.notify_all();
        crate::spill::set_test_spill_read_hook(None);

        cache.release_lane(&mut pools, &mut tbl0, &mut st0);
        cache.release_lane(&mut pools, &mut tbl1, &mut st1);
        cache.release_lane(&mut pools, &mut tbl2, &mut st2);
        cache.shutdown(&mut pools);
        assert_zero_pool_leaks(&pools);
    }

    #[test]
    fn retained_host_copy_budget_and_spill_deadline_bound_host_work() {
        let dir = ScratchDir::new("host-budget-and-deadline");
        let store = SpillStore::open_with_kvb(&dir.0, 128 * 1024 * 1024, 1).unwrap();
        let mut pools = BlockPools::with_capacities(32, 32, 1).unwrap();
        let mut cache = ServeBlockCache::with_spill(store);
        let mut tbl = BlockTable::new(32).unwrap();
        let mut st = LaneBlockState::default();

        // Cap retained host copies so at most 1 checkpoint fits in host RAM (`cs` 4 KiB each, cap 6 KiB).
        cache.set_max_retained_host_bytes(6 * 1024);
        let p0: Vec<u32> = (0..=64u32).map(|i| 1000 + i).collect();
        let p1: Vec<u32> = (0..=64u32).map(|i| 2000 + i).collect();
        for (p, tag) in [(&p0, 0x10u8), (&p1, 0x20u8)] {
            let _ = cache.match_and_bind(&mut pools, &mut tbl, &mut st, p);
            cache.ensure_span_blocks(&mut pools, &mut tbl, 0, 64).unwrap();
            cache.finish_prefill_span(&mut pools, &mut tbl, &mut st, p, 0, 64, || vec![tag; 4096]);
            cache.release_lane(&mut pools, &mut tbl, &mut st);
        }
        assert_eq!(cache.retained_start_count(), 1, "host byte budget must evict older retained start");
        assert!(cache.retained_host_bytes() <= 6 * 1024);
        assert_eq!(cache.spilled_start_count(), 1, "evicted checkpoint spills within deadline");

        // Zero spill timeout simulates an expired write deadline without sleeping: must abort cleanly!
        cache.set_spill_write_timeout(Duration::ZERO);
        let p2: Vec<u32> = (0..=64u32).map(|i| 3000 + i).collect();
        let _ = cache.match_and_bind(&mut pools, &mut tbl, &mut st, &p2);
        cache.ensure_span_blocks(&mut pools, &mut tbl, 0, 64).unwrap();
        cache.finish_prefill_span(&mut pools, &mut tbl, &mut st, &p2, 0, 64, || vec![0x30; 4096]);
        cache.release_lane(&mut pools, &mut tbl, &mut st);
        assert_eq!(cache.stats().spill_write_timeouts, 1);
        assert_eq!(cache.retained_start_count(), 1);

        cache.shutdown(&mut pools);
        assert_eq!(cache.retained_host_bytes(), 0);
        assert_zero_pool_leaks(&pools);
    }

    #[test]
    fn cache_namespace_isolates_ram_singleflight_and_ssd_restore() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("block-cache-ns-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = SpillStore::open_with_kvb(&dir, 128 * 1024 * 1024, 1).unwrap();
        let mut pools = BlockPools::with_capacities(6, 6, 1).unwrap();
        let mut cache = ServeBlockCache::with_spill(store);
        let mut tbl_a = BlockTable::new(16).unwrap();
        let mut st_a = LaneBlockState::default();
        let mut tbl_b = BlockTable::new(16).unwrap();
        let mut st_b = LaneBlockState::default();

        let prompt: Vec<u32> = (0..2100u32).map(|i| i * 31 + 7).collect();
        let ext_prompt: Vec<u32> = (0..2200u32).map(|i| i * 31 + 7).collect();

        // Singleflight leader election must NOT coalesce across distinct namespaces.
        let cands = [
            InflightPrefillLane {
                lane: 0,
                namespace: 0xAAAA,
                toks: &prompt,
                block_hashes: None,
                n_past: 0,
                pf_end: prompt.len(),
                runnable: true,
                eligible_follower: true,
            },
            InflightPrefillLane {
                lane: 1,
                namespace: 0xBBBB,
                toks: &prompt,
                block_hashes: None,
                n_past: 0,
                pf_end: prompt.len(),
                runnable: true,
                eligible_follower: true,
            },
            InflightPrefillLane {
                lane: 2,
                namespace: 0xAAAA,
                toks: &prompt,
                block_hashes: None,
                n_past: 0,
                pf_end: prompt.len(),
                runnable: true,
                eligible_follower: true,
            },
        ];
        assert_eq!(find_prefill_leader(&cands, 1), None, "different namespace must never singleflight-wait");
        assert_eq!(find_prefill_leader(&cands, 2), Some(0), "same namespace shares leader");

        // Principal A (0xAAAA) prefills `prompt[..135]` (PROMPT checkpoint at 128).
        let short_p = prompt[..135].to_vec();
        let short_ext = ext_prompt[..195].to_vec();
        assert!(cache.match_and_bind_in_namespace(&mut pools, &mut tbl_a, &mut st_a, 0xAAAA, &short_p).is_none());
        cache.ensure_span_blocks(&mut pools, &mut tbl_a, 0, 128).unwrap();
        cache.finish_prefill_span(&mut pools, &mut tbl_a, &mut st_a, &short_p, 0, 128, || fake_cs(0x41));
        cache.ensure_span_blocks(&mut pools, &mut tbl_a, 128, 7).unwrap();
        cache.finish_prefill_span(&mut pools, &mut tbl_a, &mut st_a, &short_p, 128, 7, || fake_cs(0x42));
        cache.release_lane(&mut pools, &mut tbl_a, &mut st_a);

        // Principal B (0xBBBB) sending the exact same extended prompt gets ZERO RAM warm hit.
        assert!(
            cache.match_and_bind_in_namespace(&mut pools, &mut tbl_b, &mut st_b, 0xBBBB, &short_ext).is_none(),
            "cross-principal RAM lookup must miss"
        );
        cache.release_lane(&mut pools, &mut tbl_b, &mut st_b);

        // Same principal A (0xAAAA) gets full 128-token warm hit!
        let (hit_a, cs_a) = cache
            .match_and_bind_in_namespace(&mut pools, &mut tbl_a, &mut st_a, 0xAAAA, &short_ext)
            .expect("same-principal warm hit");
        assert_eq!(hit_a, 128);
        assert_eq!(cs_a, fake_cs(0x41));
        cache.release_lane(&mut pools, &mut tbl_a, &mut st_a);

        // Spill Principal A's start point to SSD, then verify Principal B gets ZERO SSD restore hit
        // while Principal A restores 128 tokens cleanly from SSD.
        assert!(cache.evict_one_lru_start_inner(&mut pools, true, None));
        assert_eq!(cache.retained_start_count(), 0);
        assert_eq!(cache.spilled_start_count(), 1);

        assert!(
            cache.match_and_bind_in_namespace(&mut pools, &mut tbl_b, &mut st_b, 0xBBBB, &short_ext).is_none(),
            "cross-principal SSD lookup must miss without restoring"
        );
        assert_eq!(cache.stats().starts_restored, 0);
        assert_eq!(cache.spilled_start_count(), 1);
        cache.release_lane(&mut pools, &mut tbl_b, &mut st_b);

        let (restored_a, restored_cs) = cache
            .match_and_bind_in_namespace(&mut pools, &mut tbl_a, &mut st_a, 0xAAAA, &short_ext)
            .expect("same-principal SSD restore hit");
        assert_eq!(restored_a, 128);
        assert_eq!(restored_cs, fake_cs(0x41));
        assert_eq!(cache.stats().starts_restored, 1);
        cache.release_lane(&mut pools, &mut tbl_a, &mut st_a);
        cache.shutdown(&mut pools);
        assert_zero_pool_leaks(&pools);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod admission_tests;
