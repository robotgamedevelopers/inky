//! Layer 3 paged KV block cache for `inky serve`:
//! - Shared GPU `BlockPools` (`Window` + `Full`) with zero-copy block sharing across lanes
//! - `EntryTree` (`PrefixTree<BlockHandles>`) + `RetainedStart` checkpoints (`EVERY` + `PROMPT`)
//! - Exact LRU eviction of retained start points on pool pressure (never grows or panics)
//! - Pool-pressure admission guard keeping GPU memory bounded inside Metal working set

use crate::blockbook::{EVERY, PROMPT};
use crate::blocks::{
    span_bound, BlockEntry, BlockIndex, BlockKind, BlockPools, BlockTable, Slot, BLOCK, WINDOW,
};
use crate::engine::EXT;
use crate::prefix::{BlockHandles, EntryTree, PrefixMatch, PrefixTree, SealedId};
use crate::spill::{ReadOutcome, SpillBlockKind, SpillKey, SpillStore, WriteOutcome};
use std::collections::{HashMap, HashSet};

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
    pub lanes_released: u64,
}

/// Per-lane Layer 3 tracking state (lives alongside `Lane` inside `LaneS`).
#[derive(Default, Debug)]
pub struct LaneBlockState {
    pub sealed: Vec<SealedId>,
    pub prompt_len: usize,
    pub written: usize,
}

struct RetainedStart {
    pos: usize,
    cs: Vec<u8>,
    reasons: u8,
    entries: Vec<(BlockKind, BlockIndex, BlockEntry)>,
    toks: Vec<u32>,
    last_used_tick: u64,
}

struct SpilledStart {
    pos: usize,
    #[allow(dead_code)]
    reasons: u8,
    cs_key: SpillKey,
    entries: Vec<(BlockKind, BlockIndex, SpillKey)>,
    last_used_tick: u64,
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
    if end == 0 || end % BLOCK != 0 {
        return false;
    }
    if end <= EARLY_EVERY_LIMIT {
        end % EARLY_EVERY_TOKENS == 0
    } else {
        end % LATE_EVERY_TOKENS == 0
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

/// Summary of an active prefilling lane for singleflight leader election and admission accounting.
#[derive(Clone, Copy, Debug)]
pub struct InflightPrefillLane<'a> {
    pub lane: usize,
    pub toks: &'a [u32],
    pub n_past: usize,
    pub pf_end: usize,
    pub runnable: bool,
    pub eligible_follower: bool,
}

/// Maximum checkpoint boundary `ckpt >= EARLY_EVERY_TOKENS` strictly shorter than both `toks`
/// and an active runnable prefilling leader's `pf_end` that the leader will seal ahead of
/// `leader.n_past`.
pub fn inflight_shared_checkpoint(toks: &[u32], leaders: &[InflightPrefillLane<'_>]) -> usize {
    if toks.len() <= EARLY_EVERY_TOKENS {
        return 0;
    }
    let mut best = 0usize;
    for leader in leaders {
        if !leader.runnable || leader.n_past >= leader.pf_end || leader.pf_end <= EARLY_EVERY_TOKENS {
            continue;
        }
        let limit = (toks.len() - 1).min(leader.pf_end - 1).min(leader.toks.len() - 1);
        if limit < EARLY_EVERY_TOKENS || leader.n_past >= limit {
            continue;
        }
        if leader.toks[..EARLY_EVERY_TOKENS] != toks[..EARLY_EVERY_TOKENS] {
            continue;
        }
        let common = EARLY_EVERY_TOKENS
            + leader.toks[EARLY_EVERY_TOKENS..=limit]
                .iter()
                .zip(&toks[EARLY_EVERY_TOKENS..=limit])
                .take_while(|(a, b)| a == b)
                .count();
        let capped = common.min(limit);
        let every_ckpt = max_every_checkpoint_le(capped);
        if every_ckpt >= EARLY_EVERY_TOKENS && leader.n_past < every_ckpt {
            best = best.max(every_ckpt);
        }
        let leader_prompt_ckpt = ((leader.pf_end - 1) / BLOCK) * BLOCK;
        if leader_prompt_ckpt >= EARLY_EVERY_TOKENS
            && leader_prompt_ckpt <= capped
            && leader.n_past < leader_prompt_ckpt
        {
            best = best.max(leader_prompt_ckpt);
        }
    }
    best
}

/// Returns `Some(leader_lane)` if `lanes[idx]` is an eligible untouched prefilling follower
/// whose next checkpoint will be sealed by another active runnable prefilling lane strictly
/// ahead in `(n_past, Reverse(lane))` order, or `None` if this lane should run itself.
pub fn find_prefill_leader(lanes: &[InflightPrefillLane<'_>], idx: usize) -> Option<usize> {
    let me = lanes.get(idx)?;
    if !me.runnable || !me.eligible_follower || me.n_past + 1 >= me.pf_end {
        return None;
    }
    let next_ckpt = effective_pf_target(me.n_past, me.pf_end);
    if next_ckpt <= me.n_past || next_ckpt >= me.pf_end || next_ckpt % BLOCK != 0 {
        return None;
    }
    let every = is_every_checkpoint(next_ckpt);
    for cand in lanes {
        if cand.lane == me.lane || !cand.runnable || cand.n_past >= cand.pf_end {
            continue;
        }
        if cand.n_past >= next_ckpt || next_ckpt >= cand.pf_end || next_ckpt > cand.toks.len() {
            continue;
        }
        let leader_prompt_ckpt = ((cand.pf_end - 1) / BLOCK) * BLOCK;
        if !every && next_ckpt != leader_prompt_ckpt {
            continue;
        }
        let ahead_or_tie =
            cand.n_past > me.n_past || (cand.n_past == me.n_past && cand.lane < me.lane);
        if !ahead_or_tie {
            continue;
        }
        if cand.toks[..next_ckpt] == me.toks[..next_ckpt] {
            return Some(cand.lane);
        }
    }
    None
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
    pub toks: &'a [u32],
    pub n_past: usize,
    pub pf_end: usize,
    pub cached: usize,
    pub written: usize,
    pub gen: usize,
    pub max_gen: usize,
    pub runnable: bool,
}

/// Exact remaining `Full` slots already promised to currently active lanes (unwritten prompt suffix
/// plus remaining bounded decode reserve), avoiding double-counting shared prefixes for voluntary
/// singleflight followers waiting on a runnable prefilling leader.
pub fn reserved_inflight_full_slots(lanes: &[InflightLaneReserve<'_>]) -> usize {
    if lanes.is_empty() {
        return 0;
    }
    let prefill_cands: Vec<InflightPrefillLane<'_>> = lanes
        .iter()
        .filter(|l| l.gen == 0 && l.n_past < l.pf_end)
        .map(|l| InflightPrefillLane {
            lane: l.lane,
            toks: l.toks,
            n_past: l.n_past,
            pf_end: l.pf_end,
            runnable: l.runnable,
            eligible_follower: l.n_past == l.cached && l.written == l.n_past,
        })
        .collect();

    let mut total = 0usize;
    for l in lanes {
        let unwritten_prompt_tokens = if l.gen == 0 && l.n_past < l.pf_end {
            let mut unwritten = l.pf_end.saturating_sub(l.n_past);
            if let Some(cand_idx) = prefill_cands.iter().position(|c| c.lane == l.lane) {
                if find_prefill_leader(&prefill_cands, cand_idx).is_some() {
                    let other_leaders: Vec<InflightPrefillLane<'_>> = prefill_cands
                        .iter()
                        .copied()
                        .filter(|c| c.lane != l.lane)
                        .collect();
                    let shared_ckpt = inflight_shared_checkpoint(l.toks, &other_leaders);
                    if shared_ckpt > l.n_past {
                        unwritten = l.pf_end.saturating_sub(shared_ckpt);
                    }
                }
            }
            unwritten
        } else {
            0
        };
        let rem_dec_tokens = l.max_gen.min(MAX_ADMIT_DECODE_RESERVE).saturating_sub(l.gen);
        total += (unwritten_prompt_tokens + rem_dec_tokens).div_ceil(BLOCK);
    }
    total
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
    next_spill_block_id: u64,
    max_spilled_starts: usize,
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
            next_spill_block_id: 1,
            max_spilled_starts: DEFAULT_MAX_SPILLED_STARTS,
        }
    }

    pub fn with_spill(store: SpillStore) -> Self {
        let mut cache = Self::new();
        cache.spill_store = Some(store);
        cache
    }

    #[allow(dead_code)]
    pub fn set_spill_store(&mut self, store: SpillStore) {
        self.spill_store = Some(store);
    }

    #[allow(dead_code)]
    pub fn spill_store(&self) -> Option<&SpillStore> {
        self.spill_store.as_ref()
    }

    #[allow(dead_code)]
    pub fn spill_store_mut(&mut self) -> Option<&mut SpillStore> {
        self.spill_store.as_mut()
    }

    #[allow(dead_code)]
    pub fn spilled_start_count(&self) -> usize {
        self.spilled_starts.len()
    }

    #[allow(dead_code)]
    pub fn record_step_deferred_pool_pressure(&mut self) {
        self.stats.step_deferred_pool_pressure += 1;
    }

    #[allow(dead_code)]
    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    #[allow(dead_code)]
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
        let pool = pools.of(kind);
        let srefs = &self.start_slot_refs[kind_idx(kind)];
        let mut avail = 0usize;
        for i in 0..pool.len() {
            let slot = Slot::new((i + 1) as u32).expect("valid pool slot");
            let rec = pool.record(slot).expect("pool record");
            let held_by_starts = srefs.get(i).copied().unwrap_or(0);
            debug_assert!(
                rec.refs() >= held_by_starts,
                "{kind:?} slot {} has refs {} < start_slot_refs {}",
                slot.get(),
                rec.refs(),
                held_by_starts
            );
            if rec.generation() < u32::MAX && rec.refs() <= held_by_starts {
                avail += 1;
            }
        }
        avail
    }

    /// Peek how many leading tokens of `toks` match a live `RetainedStart` without mutating state.
    pub fn peek_match_len(&self, pools: &BlockPools, toks: &[u32]) -> usize {
        if toks.len() <= 1 {
            return 0;
        }
        let matched = self.tree.deepest_start(&toks[..toks.len() - 1]);
        if matched.at == 0 {
            return 0;
        }
        let Some(sid) = matched.start else {
            return 0;
        };
        let Some(st) = self.starts.get(&sid) else {
            return 0;
        };
        if st.entries.iter().all(|&(kind, _, e)| entry_is_live_sealed(pools, kind, e)) {
            matched.at
        } else {
            0
        }
    }

    /// Pool-pressure admission guard: returns true if `pools` has enough free + evictable slots
    /// for this request's uncached prompt suffix plus bounded decode reserve. Always returns true
    /// when `active_lanes == 0` so at least one request always makes progress and never deadlocks.
    #[allow(dead_code)]
    pub fn can_admit(
        &mut self,
        pools: &BlockPools,
        active_lanes: usize,
        toks: &[u32],
        max_gen: usize,
        chunk: usize,
    ) -> bool {
        self.can_admit_with_shared(pools, active_lanes, toks, max_gen, chunk, 0, 0)
    }

    /// Pool-pressure admission guard accounting for an in-flight shared prefix checkpoint
    /// `shared_ckpt` being produced by an active prefilling leader AND `reserved_inflight_full_slots`
    /// already promised to currently active lanes. `need_full` credits `cached.max(shared_ckpt)`
    /// while `need_win` stays conservative since each admitted lane binds its own window table entries.
    pub fn can_admit_with_shared(
        &mut self,
        pools: &BlockPools,
        active_lanes: usize,
        toks: &[u32],
        max_gen: usize,
        chunk: usize,
        shared_ckpt: usize,
        reserved_inflight_full_slots: usize,
    ) -> bool {
        if active_lanes == 0 {
            return true;
        }
        let cached = self.peek_match_len(pools, toks);
        let effective_cached = cached.max(shared_ckpt.min(toks.len().saturating_sub(1)));
        let uncached_full = toks.len().saturating_sub(effective_cached);
        let uncached_win = toks.len().saturating_sub(cached);
        let dec_reserve = max_gen.min(MAX_ADMIT_DECODE_RESERVE);
        let need_full = (uncached_full + dec_reserve).div_ceil(BLOCK) + 2;
        let need_win = ((WINDOW / BLOCK) + 2 + uncached_win.min(chunk).div_ceil(BLOCK))
            .min(span_bound(chunk) + 2);
        let avail_full = self
            .free_plus_evictable_slots(pools, BlockKind::Full)
            .saturating_sub(reserved_inflight_full_slots);
        let ok = avail_full >= need_full
            && self.free_plus_evictable_slots(pools, BlockKind::Window) >= need_win;
        if !ok {
            self.stats.admit_deferred_pool_pressure += 1;
        }
        ok
    }

    /// Rebind an untouched prefilling lane (`state.written == n_past` and no unsealed blocks)
    /// when `EntryTree` now holds a deeper live `RetainedStart` than `n_past`.
    pub fn rebind_if_deeper_match(
        &mut self,
        pools: &mut BlockPools,
        tbl: &mut BlockTable,
        state: &mut LaneBlockState,
        toks: &[u32],
        n_past: usize,
    ) -> Option<(usize, Vec<u8>)> {
        if state.written != n_past || state.written != state.sealed.len() * BLOCK {
            return None;
        }
        let deeper = self.peek_match_len(pools, toks);
        if deeper <= n_past {
            return None;
        }
        self.release_lane(pools, tbl, state);
        let matched = self.match_and_bind(pools, tbl, state, toks);
        if let Some((matched_at, _)) = &matched {
            if *matched_at > n_past {
                self.stats.singleflight_rebinds += 1;
                self.stats.singleflight_saved_tokens += (*matched_at - n_past) as u64;
            }
        }
        matched
    }

    /// Look up the deepest live `RetainedStart` (or spilled SSD start point when deeper than any
    /// in-RAM match) strictly shorter than `toks`, bind its `Full` and `Window` entries into `tbl`,
    /// pin its `EntryTree` path on `state`, touch its LRU tick, and return `Some((matched_at, cs_bytes))`.
    pub fn match_and_bind(
        &mut self,
        pools: &mut BlockPools,
        tbl: &mut BlockTable,
        state: &mut LaneBlockState,
        toks: &[u32],
    ) -> Option<(usize, Vec<u8>)> {
        debug_assert!(state.sealed.is_empty(), "lane must be released before match_and_bind");
        state.prompt_len = toks.len();
        state.written = 0;
        if toks.len() <= 1 {
            return None;
        }
        let mut in_ram = self.tree.deepest_start(&toks[..toks.len() - 1]);
        if in_ram.at > 0 {
            if let Some(sid) = in_ram.start {
                let all_live = match self.starts.get(&sid) {
                    Some(st) => {
                        debug_assert_eq!(st.pos, in_ram.at);
                        st.entries.iter().all(|&(kind, _, e)| entry_is_live_sealed(pools, kind, e))
                    }
                    None => false,
                };
                if !all_live {
                    self.evict_start_inner(pools, sid, false);
                    in_ram = self.tree.deepest_start(&toks[..toks.len() - 1]);
                }
            } else {
                in_ram.at = 0;
            }
        }
        if self.spill_store.is_some() && !self.spilled_starts.is_empty() {
            let on_disk = self.spilled_tree.deepest_start(&toks[..toks.len() - 1]);
            if on_disk.at > in_ram.at {
                if let Some(&spill_id) = on_disk.blocks.last() {
                    let _ = self.try_restore_spilled_start(pools, toks, spill_id, &in_ram);
                }
            }
        }
        let matched = self.tree.deepest_start(&toks[..toks.len() - 1]);
        if matched.at == 0 {
            return None;
        }
        let sid = matched.start?;
        let all_live = match self.starts.get(&sid) {
            Some(st) => {
                debug_assert_eq!(st.pos, matched.at);
                st.entries.iter().all(|&(kind, _, e)| entry_is_live_sealed(pools, kind, e))
            }
            None => return None,
        };
        if !all_live {
            self.evict_start_inner(pools, sid, false);
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
        self.next_spill_block_id = self.next_spill_block_id.wrapping_add(1).max(1);
        SpillKey::new(to_spill_kind(kind), self.next_spill_block_id)
    }

    fn remove_spilled_start(&mut self, spill_id: u64, delete_from_disk: bool) {
        let Some(sp) = self.spilled_starts.remove(&spill_id) else {
            return;
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
                if delete_from_disk {
                    if let Some(store) = self.spill_store.as_mut() {
                        store.remove_block(sk);
                    }
                }
            }
        }
    }

    fn handle_disk_gc_evicted(&mut self, evicted: &[SpillKey]) {
        for &gk in evicted {
            if let Some(entry_key) = self.spill_key_to_entry.remove(&gk) {
                self.entry_spill_keys.remove(&entry_key);
            }
            if let Some(owners) = self.spill_key_owners.remove(&gk) {
                for oid in owners {
                    self.remove_spilled_start(oid, true);
                }
            }
        }
    }

    fn spill_retained_start(&mut self, pools: &BlockPools, st: &RetainedStart) {
        if self.spill_store.is_none()
            || st.pos == 0
            || st.pos % BLOCK != 0
            || st.toks.len() != st.pos
            || st.entries.is_empty()
        {
            return;
        }
        for &(kind, _, e) in &st.entries {
            let pool = pools.of(kind);
            if !entry_is_live_sealed(pools, kind, e) || pool.slot_raw_region(e.slot()).is_none() {
                return;
            }
        }

        while self.spilled_starts.len() >= self.max_spilled_starts && self.max_spilled_starts > 0 {
            let oldest = self
                .spilled_starts
                .iter()
                .min_by_key(|(_, sp)| (sp.last_used_tick, std::cmp::Reverse(sp.pos)))
                .map(|(&id, _)| id);
            let Some(oid) = oldest else { break };
            self.remove_spilled_start(oid, true);
        }
        if self.max_spilled_starts == 0 {
            return;
        }

        let spill_id = self.next_spill_id;
        self.next_spill_id = self.next_spill_id.wrapping_add(1).max(1);

        let mut protected: HashSet<SpillKey> = HashSet::new();
        let mut newly_written_keys: Vec<SpillKey> = Vec::new();
        let mut spilled_entries: Vec<(BlockKind, BlockIndex, SpillKey)> =
            Vec::with_capacity(st.entries.len());

        for &(kind, bi, e) in &st.entries {
            let ek = (kind_idx(kind) as u8, e.slot().get(), e.generation());
            let (sk, is_new_key) = if let Some(&existing) = self.entry_spill_keys.get(&ek) {
                (existing, false)
            } else {
                (self.alloc_spill_key(kind), true)
            };
            let already_on_disk = self
                .spill_store
                .as_ref()
                .is_some_and(|store| store.contains(sk));
            if already_on_disk {
                protected.insert(sk);
                spilled_entries.push((kind, bi, sk));
                continue;
            }
            let kv_bytes = match pools.of(kind).slot_bytes(e.slot()) {
                Ok(b) => b,
                Err(_) => {
                    for nk in newly_written_keys {
                        if !self.spill_key_owners.contains_key(&nk) {
                            if let Some(entry_key) = self.spill_key_to_entry.remove(&nk) {
                                self.entry_spill_keys.remove(&entry_key);
                            }
                            if let Some(store) = self.spill_store.as_mut() {
                                store.remove_block(nk);
                            }
                        }
                    }
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
                    newly_written_keys.push(sk);
                    spilled_entries.push((kind, bi, sk));
                }
                Ok((WriteOutcome::OverBudget, gc_evicted)) => {
                    self.handle_disk_gc_evicted(&gc_evicted);
                    for nk in newly_written_keys {
                        if !self.spill_key_owners.contains_key(&nk) {
                            if let Some(entry_key) = self.spill_key_to_entry.remove(&nk) {
                                self.entry_spill_keys.remove(&entry_key);
                            }
                            if let Some(store) = self.spill_store.as_mut() {
                                store.remove_block(nk);
                            }
                        }
                    }
                    return;
                }
                Err(_) => {
                    for nk in newly_written_keys {
                        if !self.spill_key_owners.contains_key(&nk) {
                            if let Some(entry_key) = self.spill_key_to_entry.remove(&nk) {
                                self.entry_spill_keys.remove(&entry_key);
                            }
                            if let Some(store) = self.spill_store.as_mut() {
                                store.remove_block(nk);
                            }
                        }
                    }
                    return;
                }
            }
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
            }
            Ok((WriteOutcome::OverBudget, gc_evicted)) => {
                self.handle_disk_gc_evicted(&gc_evicted);
                for nk in newly_written_keys {
                    if !self.spill_key_owners.contains_key(&nk) {
                        if let Some(entry_key) = self.spill_key_to_entry.remove(&nk) {
                            self.entry_spill_keys.remove(&entry_key);
                        }
                        if let Some(store) = self.spill_store.as_mut() {
                            store.remove_block(nk);
                        }
                    }
                }
                return;
            }
            Err(_) => {
                for nk in newly_written_keys {
                    if !self.spill_key_owners.contains_key(&nk) {
                        if let Some(entry_key) = self.spill_key_to_entry.remove(&nk) {
                            self.entry_spill_keys.remove(&entry_key);
                        }
                        if let Some(store) = self.spill_store.as_mut() {
                            store.remove_block(nk);
                        }
                    }
                }
                return;
            }
        }

        let n_blocks = st.pos / BLOCK;
        let mut parent = None;
        for b in 0..n_blocks {
            let id = self
                .spilled_tree
                .insert(parent, b * BLOCK, &st.toks[b * BLOCK..(b + 1) * BLOCK], spill_id)
                .expect("insert spilled block into spilled_tree");
            parent = Some(id);
        }
        let tip = parent.expect("non-empty spilled blocks");
        if self.spilled_tree.start_holders(tip) > 0 {
            if let Some(old_id) = self.spilled_tree.handle(tip) {
                self.remove_spilled_start(old_id, true);
            }
            // Re-insert path if pruned above so tip holds the new spill_id.
            let mut p = None;
            for b in 0..n_blocks {
                let id = self
                    .spilled_tree
                    .insert(p, b * BLOCK, &st.toks[b * BLOCK..(b + 1) * BLOCK], spill_id)
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

        for sk in std::iter::once(cs_key).chain(spilled_entries.iter().map(|&(_, _, k)| k)) {
            self.spill_key_owners.entry(sk).or_default().push(spill_id);
        }
        self.spilled_starts.insert(
            spill_id,
            SpilledStart {
                pos: st.pos,
                reasons: st.reasons,
                cs_key,
                entries: spilled_entries,
                last_used_tick: st.last_used_tick,
            },
        );
        self.stats.starts_spilled += 1;
    }

    fn try_restore_spilled_start(
        &mut self,
        pools: &mut BlockPools,
        toks: &[u32],
        spill_id: u64,
        in_ram: &crate::prefix::StartMatch<BlockHandles>,
    ) -> Option<SealedId> {
        let (pos, reasons, cs_key, entries) = {
            let sp = self.spilled_starts.get(&spill_id)?;
            (sp.pos, sp.reasons, sp.cs_key, sp.entries.clone())
        };
        if pos == 0 || pos % BLOCK != 0 || pos >= toks.len() {
            self.remove_spilled_start(spill_id, true);
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

        // Phase 1: read and SHA-256 verify cs + every needed KV block BEFORE mutating pools or evicting RAM.
        let cs = match self.spill_store.as_mut()?.read_block(cs_key, Some(0)) {
            ReadOutcome::Hit(blk) => match blk.recurrent_state {
                Some(cs) if !cs.is_empty() => cs,
                _ => {
                    self.remove_spilled_start(spill_id, true);
                    return None;
                }
            },
            ReadOutcome::Miss(_) => {
                self.remove_spilled_start(spill_id, true);
                return None;
            }
        };

        let mut verified_kv: Vec<Vec<u8>> = Vec::with_capacity(disk_entries.len());
        for &(kind, _, sk) in &disk_entries {
            let expected_len = pools.of(kind).stride_bytes() as usize;
            match self.spill_store.as_mut()?.read_block(sk, Some(expected_len)) {
                ReadOutcome::Hit(blk) => verified_kv.push(blk.kv_bytes),
                ReadOutcome::Miss(_) => {
                    self.remove_spilled_start(spill_id, true);
                    return None;
                }
            }
        }

        // Phase 2: pin reused in-RAM entries so take_zeroed_slot cannot evict them.
        for &(_, fe) in &reused_full {
            pools
                .of_mut(BlockKind::Full)
                .record_mut(fe.slot())
                .unwrap()
                .add_ref()
                .expect("temp pin reused full block");
        }

        let mut taken: Vec<(BlockKind, BlockIndex, BlockEntry)> =
            Vec::with_capacity(disk_entries.len());
        for ((kind, bi, _), kv_bytes) in disk_entries.into_iter().zip(verified_kv.into_iter()) {
            match self.take_zeroed_slot(pools, kind) {
                Ok((slot, gen)) => {
                    let pool = pools.of_mut(kind);
                    let Ok(dst) = pool.slot_bytes_mut(slot) else {
                        for (k, _, e) in taken {
                            let _ = pools.of_mut(k).record_mut(e.slot()).and_then(|r| r.drop_ref());
                        }
                        for (_, fe) in reused_full {
                            let _ = pools.of_mut(BlockKind::Full).record_mut(fe.slot()).and_then(|r| r.drop_ref());
                        }
                        return None;
                    };
                    dst.copy_from_slice(&kv_bytes);
                    let rec = pool.record_mut(slot).unwrap();
                    rec.set_length(BLOCK as u32).expect("set restored block length");
                    rec.seal();
                    rec.add_ref().expect("temp pin restored slot");
                    let offset = pool.offset(slot).expect("restored slot offset");
                    let entry = BlockEntry::new(slot, gen, offset, kind, pool.kvb());
                    taken.push((kind, bi, entry));
                }
                Err(_) => {
                    for (k, _, e) in taken {
                        let _ = pools.of_mut(k).record_mut(e.slot()).and_then(|r| r.drop_ref());
                    }
                    for (_, fe) in reused_full {
                        let _ = pools.of_mut(BlockKind::Full).record_mut(fe.slot()).and_then(|r| r.drop_ref());
                    }
                    return None;
                }
            }
        }

        // Phase 3: assemble EntryTree nodes and RetainedStart, transferring refs cleanly.
        let mut full_by_bi: HashMap<u32, BlockEntry> = HashMap::with_capacity(pos / BLOCK);
        let mut win_by_bi: HashMap<u32, BlockEntry> = HashMap::with_capacity(WINDOW / BLOCK);
        for &(bi, fe) in &reused_full {
            full_by_bi.insert(bi.get(), fe);
        }
        for &(kind, bi, e) in &taken {
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
                .insert(parent, b * BLOCK, &toks[b * BLOCK..(b + 1) * BLOCK], handles)
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
            self.starts.insert(
                tip,
                RetainedStart {
                    pos,
                    cs,
                    reasons,
                    entries: final_entries,
                    toks: toks[..pos].to_vec(),
                    last_used_tick: now,
                },
            );
        }

        // Release temporary pins taken during Phase 2.
        for (k, _, e) in taken {
            pools
                .of_mut(k)
                .record_mut(e.slot())
                .unwrap()
                .drop_ref()
                .expect("drop temp restored slot pin");
        }
        for (_, fe) in reused_full {
            pools
                .of_mut(BlockKind::Full)
                .record_mut(fe.slot())
                .unwrap()
                .drop_ref()
                .expect("drop temp reused full pin");
        }

        self.remove_spilled_start(spill_id, true);
        self.stats.starts_restored += 1;
        Some(tip)
    }

    #[allow(dead_code)]
    fn evict_start(&mut self, pools: &mut BlockPools, sid: SealedId) -> bool {
        self.evict_start_inner(pools, sid, true)
    }

    fn evict_start_inner(&mut self, pools: &mut BlockPools, sid: SealedId, allow_spill: bool) -> bool {
        let Some(st) = self.starts.remove(&sid) else {
            return false;
        };
        if allow_spill && self.spill_store.is_some() {
            self.spill_retained_start(pools, &st);
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

    fn evict_one_lru_start_inner(&mut self, pools: &mut BlockPools, allow_spill: bool) -> bool {
        let mut victim: Option<(SealedId, u64, usize)> = None;
        for (&sid, st) in &self.starts {
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
        self.evict_start_inner(pools, sid, allow_spill)
    }

    /// Take one zeroed slot of `kind`, evicting LRU retained start points until a free slot
    /// is available. Never grows the pool; returns `Err(PoolExhausted(kind))` if every slot
    /// is pinned by active lanes.
    pub fn take_zeroed_slot(
        &mut self,
        pools: &mut BlockPools,
        kind: BlockKind,
    ) -> Result<(Slot, u32), CacheError> {
        loop {
            if let Some(slot) = pools.of(kind).lowest_free() {
                let pool = pools.of_mut(kind);
                let _ = pool.take(slot).expect("take zeroed slot");
                let gen = pool.record(slot).unwrap().generation();
                self.stats.slots_taken += 1;
                if gen > 1 {
                    self.stats.slots_reused += 1;
                }
                return Ok((slot, gen));
            }
            if !self.evict_one_lru_start_inner(pools, true) {
                return Err(CacheError::PoolExhausted(kind));
            }
        }
    }

    /// Ensure every `(Full, Window)` block covering span `[p0, p0 + t)` is allocated and unsealed
    /// in `tbl`, taking zeroed slots (with LRU eviction if needed). Transactional on `PoolExhausted`:
    /// rolls back any slots newly allocated during this call so a deferred lane never leaks or strands
    /// half-allocated blocks.
    pub fn ensure_span_blocks(
        &mut self,
        pools: &mut BlockPools,
        tbl: &mut BlockTable,
        p0: usize,
        t: usize,
    ) -> Result<(), CacheError> {
        debug_assert!(t >= 1);
        let lo_blk = p0 / BLOCK;
        let hi_blk = (p0 + t - 1) / BLOCK;
        let mut newly_allocated: Vec<(BlockKind, BlockIndex)> = Vec::new();
        for b in lo_blk..=hi_blk {
            let bi = BlockIndex::new(b as u32);
            for kind in KINDS {
                let existing = tbl.get(kind, bi).expect("block in table range");
                if let Some(entry) = existing {
                    let rec = pools.of(kind).record(entry.slot()).expect("existing slot record");
                    assert_eq!(rec.generation(), entry.generation(), "stale entry in active lane table");
                    assert!(!rec.sealed(), "attempt to write into sealed {kind:?} block {b}");
                } else {
                    match self.take_zeroed_slot(pools, kind) {
                        Ok((slot, gen)) => {
                            tbl.fill(pools.of_mut(kind), bi, slot, gen)
                                .expect("fill newly taken slot");
                            newly_allocated.push((kind, bi));
                        }
                        Err(err) => {
                            for (rk, rbi) in newly_allocated.into_iter().rev() {
                                tbl.clear_kind_entry(pools.of_mut(rk), rbi)
                                    .expect("rollback newly allocated block entry");
                            }
                            return Err(err);
                        }
                    }
                }
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
                        .insert(parent, b * BLOCK, blk_toks, handles)
                        .expect("insert sealed block into EntryTree");
                    let canon = self.tree.handle(id).expect("inserted handle");
                    let canon_f = canon.full();
                    let canon_w = canon.window().expect("window handle on sealed block");
                    let mut best_f = full_e;
                    let mut best_w = win_e;
                    if canon_f != full_e && entry_is_live_sealed(pools, BlockKind::Full, canon_f) {
                        tbl.fill(&mut pools.full, bi, canon_f.slot(), canon_f.generation())
                            .expect("share canonical full block");
                        best_f = canon_f;
                        self.stats.shared_dedups += 1;
                    }
                    if canon_w != win_e && entry_is_live_sealed(pools, BlockKind::Window, canon_w) {
                        tbl.fill(&mut pools.window, bi, canon_w.slot(), canon_w.generation())
                            .expect("share canonical window block");
                        best_w = canon_w;
                    }
                    if best_f != canon_f || best_w != canon_w {
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
        if end > 0 && end % BLOCK == 0 && state.sealed.len() == end / BLOCK {
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
                if let Some(existing) = self.starts.get_mut(&tip) {
                    existing.reasons |= reasons;
                    existing.last_used_tick = now;
                    if spill_enabled && existing.toks.is_empty() {
                        existing.toks = toks[..end].to_vec();
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
                        self.inc_start_slot_ref(kind, e.slot());
                    }
                    self.tree.retain_start(tip).expect("retain start in EntryTree");
                    self.starts.insert(
                        tip,
                        RetainedStart {
                            pos: end,
                            cs,
                            reasons,
                            entries,
                            toks: if spill_enabled { toks[..end].to_vec() } else { Vec::new() },
                            last_used_tick: now,
                        },
                    );
                    self.stats.start_points_saved += 1;
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
        state.prompt_len = 0;
        state.written = 0;
        self.stats.lanes_released += 1;
    }

    /// Evict every retained start point so all pool references drain cleanly on shutdown.
    pub fn shutdown(&mut self, pools: &mut BlockPools) {
        while self.evict_one_lru_start_inner(pools, false) {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(cache.can_admit(&pools, 0, &big, 1024, 2048));
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
        assert!(!cache.can_admit(&pools, 1, &big, 1024, 2048), "should defer when active lane consumes pool");
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
            let leaders: Vec<InflightPrefillLane<'_>> = lanes
                .iter()
                .enumerate()
                .filter(|(_, l)| l.active && l.n_past < l.toks.len())
                .map(|(idx, l)| InflightPrefillLane {
                    lane: idx,
                    toks: &l.toks,
                    n_past: l.n_past,
                    pf_end: l.toks.len(),
                    runnable: true,
                    eligible_follower: false,
                })
                .collect();
            let reserves: Vec<InflightLaneReserve<'_>> = lanes
                .iter()
                .enumerate()
                .filter(|(_, l)| l.active)
                .map(|(idx, l)| InflightLaneReserve {
                    lane: idx,
                    toks: &l.toks,
                    n_past: l.n_past,
                    pf_end: l.toks.len(),
                    cached: l.cached,
                    written: l.blocks.written,
                    gen: 0,
                    max_gen: 128,
                    runnable: true,
                })
                .collect();
            let reserved_inflight_full = reserved_inflight_full_slots(&reserves);
            let shared_ckpt = inflight_shared_checkpoint(&lanes[i].toks, &leaders);
            if i > 0 {
                assert_eq!(shared_ckpt, 8192);
            }
            assert!(
                cache.can_admit_with_shared(
                    &pools,
                    i,
                    &lanes[i].toks,
                    128,
                    chunk,
                    shared_ckpt,
                    reserved_inflight_full,
                ),
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
                    toks: &l.toks,
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
        for i in 1..nlanes {
            assert_eq!(lanes[i].cached, 8192, "follower {i} should cache up to 8192");
            assert_eq!(lanes[i].prefilled, 608, "follower {i} should prefill only [8192..8800)");
            assert_eq!(lanes[i].cs, fake_cs((8192 / BLOCK) as u8));
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
                    toks: &toks_list[i],
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
                toks: &leader_toks,
                n_past: 0,
                pf_end: leader_toks.len(),
                runnable: true,
                eligible_follower: true,
            },
            InflightPrefillLane {
                lane: 1,
                toks: &short_toks,
                n_past: 0,
                pf_end: short_toks.len(),
                runnable: true,
                eligible_follower: true,
            },
            InflightPrefillLane {
                lane: 2,
                toks: &diverged_toks,
                n_past: 0,
                pf_end: diverged_toks.len(),
                runnable: true,
                eligible_follower: true,
            },
        ];

        assert_eq!(find_prefill_leader(&cands, 0), None);
        assert_eq!(find_prefill_leader(&cands, 1), None, "prompt < EARLY_EVERY_TOKENS must never wait");
        assert_eq!(find_prefill_leader(&cands, 2), None, "diverged prefix must never wait");

        assert_eq!(inflight_shared_checkpoint(&short_toks, &cands[..1]), 0);
        assert_eq!(inflight_shared_checkpoint(&diverged_toks, &cands[..1]), 0);
        let mut matching = leader_toks.clone();
        matching.extend(90000..90300);
        assert_eq!(inflight_shared_checkpoint(&matching, &cands[..1]), 8448);
    }

    #[test]
    fn simultaneous_burst_admission_guard_prevents_midflight_exhaustion() {
        // 130 Full slots = 8,320 tokens. Each request needs (2000 + 512).div_ceil(64) = 40 Full slots
        // (+ 2 guard margin = 42). Six independent requests arriving simultaneously would demand 240 Full
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
                    toks: &l.toks,
                    n_past: l.n_past,
                    pf_end: l.toks.len(),
                    cached: 0,
                    written: l.blocks.written,
                    gen: l.gen,
                    max_gen,
                    runnable: true,
                })
                .collect();
            let reserved = reserved_inflight_full_slots(&reserves);
            let active_now = lanes.iter().filter(|l| l.active).count();
            if !cache.can_admit_with_shared(&pools, active_now, toks, max_gen, chunk, 0, reserved) {
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
                        toks: &l.toks,
                        n_past: l.n_past,
                        pf_end: l.toks.len(),
                        cached: 0,
                        written: l.blocks.written,
                        gen: l.gen,
                        max_gen,
                        runnable: true,
                    })
                    .collect();
                let reserved = reserved_inflight_full_slots(&reserves);
                let active_now = lanes.iter().filter(|l| l.active).count();
                if !cache.can_admit_with_shared(&pools, active_now, toks, max_gen, chunk, 0, reserved) {
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
            for i in 0..2 {
                if active[i] {
                    assert_eq!(planner.state(i), LaneState::Ready);
                }
            }
        }

        assert!(!active[0] && !active[1], "both lanes must complete without HTTP 500");
        assert!(cache.stats().step_deferred_pool_pressure >= 1);
        cache.shutdown(&mut pools);
        assert_zero_pool_leaks(&pools);
    }

    #[test]
    fn spill_store_eviction_restore_and_corrupt_fallback() {
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

        let dir = ScratchDir::new("e2e");
        let store = SpillStore::open(&dir.0, 256 * 1024 * 1024).unwrap();
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
}
