//! Layer 4 cache-aware lane scheduler, explicit lane lifecycle, starvation bound (`W`),
//! cancel-in-step barrier, TLA+ trace emission (`specs/CacheScheduler.tla`), and real
//! server step-planning adapter (`ServeStepPlanner`).
//!
//! Kept strictly `std`-only (zero `objc2`, `Metal`, or external crate dependencies) so it
//! compiles identically both inside `inky` and under standalone CPU-only `rustc` in
//! `scripts/check-scheduler-traces.sh`.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

mod trace;
// Standalone TLC driver and reference tests consume these exports.
#[allow(unused_imports)]
pub use trace::{render_cache_scheduler_tla_bundle, walk_scheduler_trace, SchedulerTrace, TraceStep, WalkStats};

/// Explicit Layer 4 lane lifecycle (`docs/ROADMAP.md:129`, `specs/CacheScheduler.tla`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LaneState {
    Idle,
    Queued,
    Admitted,
    Restoring,
    Ready,
    InStep,
    Cancelling,
    Cancelled,
    Finished,
    Refused,
}

impl LaneState {
    pub fn as_tla(self) -> &'static str {
        match self {
            Self::Idle => "Idle",
            Self::Queued => "Queued",
            Self::Admitted => "Admitted",
            Self::Restoring => "Restoring",
            Self::Ready => "Ready",
            Self::InStep => "InStep",
            Self::Cancelling => "Cancelling",
            Self::Cancelled => "Cancelled",
            Self::Finished => "Finished",
            Self::Refused => "Refused",
        }
    }

    pub fn is_terminal_or_idle(self) -> bool {
        matches!(self, Self::Idle | Self::Refused | Self::Cancelled | Self::Finished)
    }

    pub fn is_mem_holding(self) -> bool {
        matches!(self, Self::Restoring | Self::Ready | Self::InStep | Self::Cancelling)
    }

    pub fn is_admitted(self) -> bool {
        self == Self::Admitted || self.is_mem_holding()
    }

    pub fn is_waiting(self) -> bool {
        matches!(self, Self::Queued | Self::Admitted | Self::Restoring | Self::Ready)
    }
}

/// Cache block kind matching `specs/BlockLayout.tla` and `specs/CacheScheduler.tla`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BlockKind {
    Full,
    Window,
}

impl BlockKind {
    pub fn as_tla(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Window => "window",
        }
    }
}

/// Residence tier of a cache block across GPU memory and SSD disk tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BlockTier {
    None,
    Memory,
    Disk,
    Both,
}

impl BlockTier {
    pub fn as_tla(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Memory => "memory",
            Self::Disk => "disk",
            Self::Both => "both",
        }
    }

    pub fn in_memory(self) -> bool {
        matches!(self, Self::Memory | Self::Both)
    }

    pub fn on_disk(self) -> bool {
        matches!(self, Self::Disk | Self::Both)
    }

    pub fn is_present(self) -> bool {
        self != Self::None
    }
}

/// Canonical block identifier matching `TreeIds` and `PrivIds` in `specs/CacheScheduler.tla`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BlockId {
    Tree {
        prefix: String,
        kind: BlockKind,
        idx: u32,
    },
    Priv {
        lane: usize,
        kind: BlockKind,
    },
}

/// Metadata for a single logical cache block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockRecord {
    pub tier: BlockTier,
    pub sealed: bool,
    pub idx: u32,
    pub length: u32,
    pub data: Vec<u8>,
    pub ref_count: usize,
}

impl BlockRecord {
    fn empty(block_size: u32) -> Self {
        Self {
            tier: BlockTier::None,
            sealed: false,
            idx: 0,
            length: 0,
            data: vec![0; block_size as usize],
            ref_count: 0,
        }
    }
}

/// Retained prefix start point at a block-aligned boundary.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct StartPoint {
    pub kept: bool,
    pub state_kept: bool,
    pub full: BTreeSet<BlockId>,
    pub win: BTreeSet<BlockId>,
}

/// Per-lane scheduler state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneSlot {
    pub name: String,
    pub state: LaneState,
    pub prefix: String,
    pub target: u32,
    pub pos: u32,
    pub full: BTreeMap<u32, BlockId>,
    pub win: BTreeMap<u32, BlockId>,
    pub reservation: u32,
    pub admit_budget: u32,
    pub wait_ticks: u32,
    /// Pending async restore disk I/O block (if any). While `Some`, cancelling a
    /// `Restoring` lane enters `Cancelling` and holds slot + blocks until I/O finishes.
    pub restore_io_block: Option<BlockId>,
}

/// Configuration parameters matching `CONSTANTS` of `specs/CacheScheduler.tla`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchedulerConfig {
    pub lanes: Vec<String>,
    pub prefixes: Vec<String>,
    pub request_lengths: Vec<u32>,
    pub block_size: u32,
    pub window_size: u32,
    pub max_pos: u32,
    pub max_span: u32,
    pub step_token_budget: u32,
    pub mem_budget: u32,
    pub disk_budget: u32,
    pub max_queue: usize,
    pub max_bypass: u32,
    /// Stride in blocks for retained start points (`1` = every block boundary, matching
    /// `specs/CacheScheduler.tla`; `16` = every 16 blocks + prompt end, `docs/ROADMAP.md:111`).
    pub start_point_stride_blocks: u32,
}

impl SchedulerConfig {
    pub fn validate(&self) -> Result<(), SchedError> {
        if self.start_point_stride_blocks == 0 {
            return Err(SchedError::InvalidConfig("start_point_stride_blocks must be >= 1"));
        }
        if self.lanes.is_empty() {
            return Err(SchedError::InvalidConfig("lanes must be non-empty"));
        }
        if self.prefixes.is_empty() {
            return Err(SchedError::InvalidConfig("prefixes must be non-empty"));
        }
        if self.request_lengths.is_empty() {
            return Err(SchedError::InvalidConfig("request_lengths must be non-empty"));
        }
        if self.block_size == 0 {
            return Err(SchedError::InvalidConfig("block_size must be > 0"));
        }
        if self.window_size < self.block_size {
            return Err(SchedError::InvalidConfig("window_size must be >= block_size"));
        }
        if self.max_pos < self.block_size || self.max_pos % self.block_size != 0 {
            return Err(SchedError::InvalidConfig("max_pos must be a positive multiple of block_size"));
        }
        for &r in &self.request_lengths {
            if r == 0 || r > self.max_pos || r % self.block_size != 0 {
                return Err(SchedError::InvalidConfig("each request length must be in 1..=max_pos and block-aligned"));
            }
        }
        if self.max_span == 0 || self.max_span > self.block_size {
            return Err(SchedError::InvalidConfig("max_span must be in 1..=block_size"));
        }
        if self.step_token_budget < self.max_span {
            return Err(SchedError::InvalidConfig("step_token_budget must be >= max_span"));
        }
        if self.mem_budget < 2 || self.disk_budget < 2 {
            return Err(SchedError::InvalidConfig("mem_budget and disk_budget must be >= 2"));
        }
        if self.max_bypass < 1 {
            return Err(SchedError::InvalidConfig("max_bypass (W) must be >= 1"));
        }
        Ok(())
    }

    pub fn default_prefix(&self) -> &str {
        &self.prefixes[0]
    }

    pub fn default_target(&self) -> u32 {
        self.request_lengths[0]
    }

    pub fn num_blocks(&self) -> u32 {
        self.max_pos / self.block_size
    }

    pub fn boundaries(&self) -> Vec<u32> {
        (1..=self.num_blocks()).map(|i| i * self.block_size).collect()
    }
}

/// Outcome of submitting a request to a lane slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArriveOutcome {
    Queued,
    DirectAdmitted,
    Refused429,
}

/// Description of one scheduled execution step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuiltStep {
    pub lanes: Vec<usize>,
    pub spans: BTreeMap<usize, u32>,
    pub total_tokens: u32,
    pub pins: BTreeSet<BlockId>,
}

/// Summary returned when `complete_step()` retires an in-flight step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletedStep {
    pub survived: Vec<usize>,
    pub cancelled: Vec<usize>,
    pub sealed_boundaries: Vec<(usize, u32)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SchedError {
    InvalidConfig(&'static str),
    InvalidLane(usize),
    InvalidPrefix(String),
    InvalidTarget(u32),
    InvalidState { lane: usize, expected: &'static str, actual: LaneState },
    Precondition(&'static str),
}

impl fmt::Display for SchedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(m) => write!(f, "invalid scheduler config: {m}"),
            Self::InvalidLane(l) => write!(f, "invalid lane index {l}"),
            Self::InvalidPrefix(p) => write!(f, "unknown prefix {p:?}"),
            Self::InvalidTarget(t) => write!(f, "disallowed request target length {t}"),
            Self::InvalidState { lane, expected, actual } => {
                write!(f, "lane {lane} expected {expected}, found {actual:?}")
            }
            Self::Precondition(m) => write!(f, "scheduler precondition failed: {m}"),
        }
    }
}

impl std::error::Error for SchedError {}

// ---- Exact Logical Block-Range Helpers (`specs/BlockLayout.tla` / `CacheScheduler.tla`) ----

pub fn ceil_div(n: u32, d: u32) -> u32 {
    (n + d - 1) / d
}

pub fn span_bound(window_size: u32, block_size: u32, c: u32) -> u32 {
    ceil_div(window_size - 1 + c, block_size) + 1
}

pub fn window_indices(lo: u32, hi: u32, block_size: u32) -> BTreeSet<u32> {
    if hi < lo {
        return BTreeSet::new();
    }
    let start = lo / block_size;
    let end = hi / block_size;
    (start..=end).collect()
}

pub fn full_indices(n: u32, block_size: u32) -> BTreeSet<u32> {
    if n == 0 {
        BTreeSet::new()
    } else {
        window_indices(0, n - 1, block_size)
    }
}

pub fn history_indices(n: u32, window_size: u32, block_size: u32) -> BTreeSet<u32> {
    if n == 0 {
        return BTreeSet::new();
    }
    let lo = n.saturating_sub(window_size);
    window_indices(lo, n - 1, block_size)
}

pub fn span_window_indices(n: u32, c: u32, window_size: u32, block_size: u32) -> BTreeSet<u32> {
    if c == 0 {
        return BTreeSet::new();
    }
    let lo = if n >= window_size { n - window_size + 1 } else { 0 };
    let hi = n + c - 1;
    window_indices(lo, hi, block_size)
}

pub fn worst_case_blocks(target: u32, block_size: u32, window_size: u32, max_span: u32) -> u32 {
    let total_full = full_indices(target, block_size).len() as u32;
    let max_win = span_bound(window_size, block_size, max_span).min(total_full);
    total_full + max_win
}

/// Positions below `prompt_end(target)` are prefill spans (up to `max_span`);
/// positions at or above `prompt_end(target)` are 1-token decode steps.
pub fn prompt_end(target: u32, block_size: u32) -> u32 {
    target.saturating_sub(block_size)
}

// ---- Core Layer 4 Cache-Aware Scheduler -----------------------------------------------------

/// Deterministic Layer 4 cache-aware lane scheduler implementing:
/// - Explicit lane lifecycle (`Queued -> Admitted -> Restoring -> Ready -> InStep -> Finished`,
///   plus `Cancelling -> Cancelled`),
/// - Worst-case admission and bounded FIFO queue with 429 refusal,
/// - Starvation-bounded bypass ceiling `W` (`max_bypass`) across `Queued`, `Restoring`, and `Ready`,
/// - Deterministic step building (starvation-ceiling lanes, longest-waiting ready lane first,
///   active 1-token decoding lanes, prefill chunks up to `step_token_budget`),
/// - Cancel-in-step and cancel-in-restore barriers waiting for GPU work and disk I/O.
#[derive(Clone, Debug)]
pub struct Scheduler {
    cfg: SchedulerConfig,
    blocks: BTreeMap<BlockId, BlockRecord>,
    start_points: BTreeMap<(String, u32), StartPoint>,
    lanes: Vec<LaneSlot>,
    queue_seq: Vec<usize>,
    step_active: bool,
    step_lanes: BTreeSet<usize>,
    step_span: Vec<u32>,
    step_pins: BTreeSet<BlockId>,
    step_io: BTreeSet<BlockId>,
    gpu_done: bool,
}

impl Scheduler {
    pub fn new(cfg: SchedulerConfig) -> Result<Self, SchedError> {
        cfg.validate()?;
        let n_lanes = cfg.lanes.len();
        let default_prefix = cfg.default_prefix().to_string();
        let default_target = cfg.default_target();
        let lanes = cfg
            .lanes
            .iter()
            .map(|name| LaneSlot {
                name: name.clone(),
                state: LaneState::Idle,
                prefix: default_prefix.clone(),
                target: default_target,
                pos: 0,
                full: BTreeMap::new(),
                win: BTreeMap::new(),
                reservation: 0,
                admit_budget: 0,
                wait_ticks: 0,
                restore_io_block: None,
            })
            .collect();
        Ok(Self {
            cfg,
            blocks: BTreeMap::new(),
            start_points: BTreeMap::new(),
            lanes,
            queue_seq: Vec::new(),
            step_active: false,
            step_lanes: BTreeSet::new(),
            step_span: vec![0; n_lanes],
            step_pins: BTreeSet::new(),
            step_io: BTreeSet::new(),
            gpu_done: false,
        })
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.cfg
    }

    pub fn lanes(&self) -> &[LaneSlot] {
        &self.lanes
    }

    pub fn lane(&self, l: usize) -> Option<&LaneSlot> {
        self.lanes.get(l)
    }

    pub fn blocks(&self) -> &BTreeMap<BlockId, BlockRecord> {
        &self.blocks
    }

    pub fn start_points(&self) -> &BTreeMap<(String, u32), StartPoint> {
        &self.start_points
    }

    pub fn queue_seq(&self) -> &[usize] {
        &self.queue_seq
    }

    pub fn step_active(&self) -> bool {
        self.step_active
    }

    pub fn step_lanes(&self) -> &BTreeSet<usize> {
        &self.step_lanes
    }

    pub fn step_span(&self, lane: usize) -> u32 {
        self.step_span.get(lane).copied().unwrap_or(0)
    }

    pub fn step_pins(&self) -> &BTreeSet<BlockId> {
        &self.step_pins
    }

    pub fn step_io(&self) -> &BTreeSet<BlockId> {
        &self.step_io
    }

    pub fn gpu_done(&self) -> bool {
        self.gpu_done
    }

    // ---- Queries ----------------------------------------------------------------------------

    pub fn worst_case(&self, target: u32) -> u32 {
        worst_case_blocks(target, self.cfg.block_size, self.cfg.window_size, self.cfg.max_span)
    }

    pub fn mem_used(&self) -> u32 {
        self.blocks.values().filter(|b| b.tier.in_memory()).count() as u32
    }

    pub fn disk_used(&self) -> u32 {
        self.blocks.values().filter(|b| b.tier.on_disk()).count() as u32
    }

    pub fn reserved_total(&self) -> u32 {
        self.lanes
            .iter()
            .filter(|l| l.state.is_mem_holding())
            .map(|l| l.reservation)
            .sum()
    }

    pub fn admitted_total(&self) -> u32 {
        self.lanes
            .iter()
            .filter(|l| l.state.is_admitted())
            .map(|l| l.admit_budget)
            .sum()
    }

    pub fn lane_block_set(&self, l: usize) -> BTreeSet<BlockId> {
        let slot = &self.lanes[l];
        let mut s = BTreeSet::new();
        for b in slot.full.values().chain(slot.win.values()) {
            s.insert(b.clone());
        }
        s
    }

    fn block_tier(&self, id: &BlockId) -> BlockTier {
        self.blocks.get(id).map_or(BlockTier::None, |b| b.tier)
    }

    fn lane_referrers(&self, id: &BlockId) -> usize {
        self.lanes
            .iter()
            .filter(|l| {
                l.state.is_admitted()
                    && (l.full.values().any(|b| b == id) || l.win.values().any(|b| b == id))
            })
            .count()
    }

    fn start_referrers(&self, id: &BlockId) -> usize {
        self.start_points
            .values()
            .filter(|sp| sp.kept && (sp.full.contains(id) || sp.win.contains(id)))
            .count()
    }

    fn sync_ref_counts(&mut self) {
        let mut counts: BTreeMap<BlockId, usize> = BTreeMap::new();
        for l in &self.lanes {
            if l.state.is_admitted() {
                for b in l.full.values().chain(l.win.values()) {
                    *counts.entry(b.clone()).or_insert(0) += 1;
                }
            }
        }
        for sp in self.start_points.values() {
            if sp.kept {
                for b in sp.full.iter().chain(&sp.win) {
                    *counts.entry(b.clone()).or_insert(0) += 1;
                }
            }
        }
        self.blocks.retain(|id, rec| {
            if !rec.tier.is_present() {
                return false;
            }
            let refs = counts.get(id).copied().unwrap_or(0);
            if !rec.sealed && refs == 0 {
                return false;
            }
            rec.ref_count = refs;
            true
        });
    }

    fn queue_pos(&self, lane: usize) -> Option<usize> {
        self.queue_seq.iter().position(|&q| q == lane)
    }

    fn older_queued(&self, a: usize, b: usize) -> bool {
        match (self.queue_pos(a), self.queue_pos(b)) {
            (Some(ia), Some(ib)) => ia < ib,
            _ => false,
        }
    }

    pub fn is_prefilling(&self, l: usize) -> bool {
        let slot = &self.lanes[l];
        slot.state == LaneState::Ready && slot.pos < prompt_end(slot.target, self.cfg.block_size)
    }

    pub fn is_decoding(&self, l: usize) -> bool {
        let slot = &self.lanes[l];
        let pe = prompt_end(slot.target, self.cfg.block_size);
        slot.state == LaneState::Ready && slot.pos >= pe && slot.pos < slot.target
    }

    pub fn is_steppable_ready(&self, l: usize) -> bool {
        let slot = &self.lanes[l];
        slot.state == LaneState::Ready && slot.pos < slot.target
    }

    pub fn need_alloc(&self, l: usize) -> u32 {
        if !self.is_steppable_ready(l) {
            return 0;
        }
        let slot = &self.lanes[l];
        let bi = slot.pos / self.cfg.block_size;
        let nf = u32::from(!slot.full.contains_key(&bi));
        let nw = u32::from(!slot.win.contains_key(&bi));
        nf + nw
    }

    pub fn current_working_set(&self, l: usize) -> u32 {
        let slot = &self.lanes[l];
        let base = self.lane_block_set(l).len() as u32;
        let extra = if slot.pos < slot.target {
            let bi = slot.pos / self.cfg.block_size;
            u32::from(!slot.full.contains_key(&bi)) + u32::from(!slot.win.contains_key(&bi))
        } else {
            0
        };
        base + extra
    }

    fn all_lane_referred_blocks(&self) -> BTreeSet<BlockId> {
        let mut owned = BTreeSet::new();
        for l in &self.lanes {
            if l.state.is_admitted() {
                for b in l.full.values().chain(l.win.values()) {
                    owned.insert(b.clone());
                }
            }
        }
        owned
    }

    pub fn cold_mem_blocks(&self) -> Vec<BlockId> {
        let owned = self.all_lane_referred_blocks();
        self.blocks
            .iter()
            .filter(|(id, rec)| {
                rec.tier.in_memory() && !owned.contains(*id) && !self.step_io.contains(*id)
            })
            .map(|(id, _)| id.clone())
            .collect()
    }

    pub fn cold_disk_blocks(&self) -> Vec<BlockId> {
        let owned = self.all_lane_referred_blocks();
        self.blocks
            .iter()
            .filter(|(id, rec)| {
                rec.tier.on_disk() && !owned.contains(*id) && !self.step_io.contains(*id)
            })
            .map(|(id, _)| id.clone())
            .collect()
    }

    pub fn mem_pressure(&self) -> bool {
        if self.cold_mem_blocks().is_empty() {
            return false;
        }
        let restore_blocked = self.mem_used() >= self.cfg.mem_budget
            && (0..self.lanes.len()).any(|l| {
                self.lanes[l].state == LaneState::Restoring
                    && self
                        .lane_block_set(l)
                        .iter()
                        .any(|b| !self.block_tier(b).in_memory())
            });
        let steppable_need: u32 = (0..self.lanes.len())
            .filter(|&l| self.is_steppable_ready(l))
            .map(|l| self.need_alloc(l))
            .sum();
        let ready_blocked = (0..self.lanes.len()).any(|l| self.is_steppable_ready(l))
            && self.mem_used() + steppable_need > self.cfg.mem_budget;
        restore_blocked || ready_blocked
    }

    pub fn best_boundary(&self, prefix: &str, max_p: u32) -> u32 {
        self.start_points
            .iter()
            .filter(|((p, k), sp)| p == prefix && *k <= max_p && sp.kept)
            .map(|((_, k), _)| *k)
            .max()
            .unwrap_or(0)
    }

    pub fn can_fit_request(&self, target: u32) -> bool {
        self.admitted_total() + self.worst_case(target) <= self.cfg.mem_budget
    }

    pub fn can_fit_admit(&self, l: usize) -> bool {
        self.can_fit_request(self.lanes[l].target)
    }

    /// True iff queued lane `l` is eligible to admit now under longest-waiting-first,
    /// FIFO tie-breaking, and starvation-ceiling (`max_bypass`) anti-leapfrogging.
    pub fn can_admit_now(&self, l: usize) -> bool {
        if self.lanes[l].state != LaneState::Queued || !self.can_fit_admit(l) {
            return false;
        }
        let wl = self.lanes[l].wait_ticks;
        for q in 0..self.lanes.len() {
            if q == l || self.lanes[q].state != LaneState::Queued {
                continue;
            }
            let wq = self.lanes[q].wait_ticks;
            if self.can_fit_admit(q) && !(wl > wq || (wl == wq && self.older_queued(l, q))) {
                return false;
            }
            if wq >= self.cfg.max_bypass && !(self.older_queued(l, q) || wl > wq) {
                return false;
            }
        }
        true
    }

    pub fn can_advance_admitted(&self, l: usize) -> bool {
        self.lanes[l].state == LaneState::Admitted
            && self.reserved_total() + self.worst_case(self.lanes[l].target) <= self.cfg.mem_budget
    }

    pub fn can_advance_restoring(&self, l: usize) -> bool {
        if self.lanes[l].state != LaneState::Restoring {
            return false;
        }
        let all_in_mem = self
            .lane_block_set(l)
            .iter()
            .all(|b| self.block_tier(b).in_memory());
        all_in_mem || self.mem_used() < self.cfg.mem_budget || !self.cold_mem_blocks().is_empty()
    }

    /// All waiting lanes across `Ready`, `Restoring`, `Admitted`, and `Queued` that can
    /// advance immediately without waiting for running lanes to finish.
    pub fn actionable_waiting(&self) -> BTreeSet<usize> {
        let mut set = BTreeSet::new();
        for l in 0..self.lanes.len() {
            if self.is_steppable_ready(l)
                || self.can_advance_admitted(l)
                || self.can_advance_restoring(l)
                || self.can_admit_now(l)
            {
                set.insert(l);
            }
        }
        set
    }

    pub fn step_bypassed(&self, chosen: &BTreeSet<usize>) -> BTreeSet<usize> {
        self.actionable_waiting()
            .difference(chosen)
            .copied()
            .collect()
    }

    fn check_lane_idx(&self, l: usize) -> Result<(), SchedError> {
        if l < self.lanes.len() {
            Ok(())
        } else {
            Err(SchedError::InvalidLane(l))
        }
    }

    // ---- Transitions ------------------------------------------------------------------------

    /// Atomic request arrival: enqueue (`Queued`), direct-admit (`Admitted`) when queue is
    /// full and no older queued lane can fit or reached `W`, or refuse (`Refused`, HTTP 429).
    pub fn arrive(
        &mut self,
        lane: usize,
        prefix: &str,
        target: u32,
    ) -> Result<ArriveOutcome, SchedError> {
        self.check_lane_idx(lane)?;
        if !self.lanes[lane].state.is_terminal_or_idle() {
            return Err(SchedError::InvalidState {
                lane,
                expected: "Idle|Refused|Cancelled|Finished",
                actual: self.lanes[lane].state,
            });
        }
        if !self.cfg.prefixes.iter().any(|p| p == prefix) {
            return Err(SchedError::InvalidPrefix(prefix.to_string()));
        }
        if !self.cfg.request_lengths.contains(&target) {
            return Err(SchedError::InvalidTarget(target));
        }

        let wc = self.worst_case(target);
        let queued_cnt = self.queue_seq.len();
        let can_enqueue = wc <= self.cfg.mem_budget && queued_cnt < self.cfg.max_queue;
        let can_direct_admit = wc <= self.cfg.mem_budget
            && queued_cnt == self.cfg.max_queue
            && self.can_fit_request(target)
            && self.queue_seq.iter().all(|&q| {
                !self.can_fit_admit(q) && self.lanes[q].wait_ticks < self.cfg.max_bypass
            });

        if can_enqueue {
            let slot = &mut self.lanes[lane];
            slot.state = LaneState::Queued;
            slot.prefix = prefix.to_string();
            slot.target = target;
            slot.pos = 0;
            slot.full.clear();
            slot.win.clear();
            slot.reservation = 0;
            slot.admit_budget = 0;
            slot.wait_ticks = 0;
            slot.restore_io_block = None;
            self.queue_seq.push(lane);
            Ok(ArriveOutcome::Queued)
        } else if can_direct_admit {
            let queued_snapshot: Vec<usize> = self.queue_seq.clone();
            for q in queued_snapshot {
                self.lanes[q].wait_ticks += 1;
            }
            let slot = &mut self.lanes[lane];
            slot.state = LaneState::Admitted;
            slot.prefix = prefix.to_string();
            slot.target = target;
            slot.pos = 0;
            slot.full.clear();
            slot.win.clear();
            slot.reservation = 0;
            slot.admit_budget = wc;
            slot.wait_ticks = 0;
            slot.restore_io_block = None;
            Ok(ArriveOutcome::DirectAdmitted)
        } else {
            let default_prefix = self.cfg.default_prefix().to_string();
            let default_target = self.cfg.default_target();
            let slot = &mut self.lanes[lane];
            slot.state = LaneState::Refused;
            slot.prefix = default_prefix;
            slot.target = default_target;
            slot.pos = 0;
            slot.full.clear();
            slot.win.clear();
            slot.reservation = 0;
            slot.admit_budget = 0;
            slot.wait_ticks = 0;
            slot.restore_io_block = None;
            Ok(ArriveOutcome::Refused429)
        }
    }

    /// Admit a `Queued` lane (`Queued -> Admitted`), preserving its `wait_ticks` and
    /// incrementing `wait_ticks` on any older queued lanes it bypassed.
    pub fn admit(&mut self, lane: usize) -> Result<(), SchedError> {
        self.check_lane_idx(lane)?;
        if self.lanes[lane].state != LaneState::Queued {
            return Err(SchedError::InvalidState {
                lane,
                expected: "Queued",
                actual: self.lanes[lane].state,
            });
        }
        if !self.can_admit_now(lane) {
            return Err(SchedError::Precondition("lane is not eligible to admit now"));
        }
        let wc = self.worst_case(self.lanes[lane].target);
        let bypassed: Vec<usize> = self
            .queue_seq
            .iter()
            .copied()
            .filter(|&q| {
                q != lane
                    && self.older_queued(q, lane)
                    && self.lanes[q].wait_ticks < self.cfg.max_bypass
            })
            .collect();
        self.lanes[lane].state = LaneState::Admitted;
        self.lanes[lane].admit_budget = wc;
        for q in bypassed {
            self.lanes[q].wait_ticks += 1; // MUTANT_ANCHOR_ADMIT_BYPASS
        }
        self.queue_seq.retain(|&q| q != lane);
        Ok(())
    }

    /// Pick and admit the highest-priority eligible queued lane, if any.
    pub fn admit_next(&mut self) -> Result<Option<usize>, SchedError> {
        let candidate = self
            .queue_seq
            .iter()
            .copied()
            .find(|&q| self.can_admit_now(q));
        if let Some(l) = candidate {
            self.admit(l)?;
            Ok(Some(l))
        } else {
            Ok(None)
        }
    }

    /// Prefix match and atomic worst-case working-set reservation (`Admitted -> Restoring`).
    pub fn begin_restore(&mut self, lane: usize) -> Result<(), SchedError> {
        self.check_lane_idx(lane)?;
        if self.lanes[lane].state != LaneState::Admitted {
            return Err(SchedError::InvalidState {
                lane,
                expected: "Admitted",
                actual: self.lanes[lane].state,
            });
        }
        let prefix = self.lanes[lane].prefix.clone();
        let target = self.lanes[lane].target;
        let wc = self.worst_case(target);
        if self.reserved_total() + wc > self.cfg.mem_budget {
            return Err(SchedError::Precondition("insufficient memory budget for atomic restore reservation"));
        }
        let k = self.best_boundary(&prefix, target);
        let mut nf = BTreeMap::new();
        let mut nw = BTreeMap::new();
        if k > 0 {
            if let Some(sp) = self.start_points.get(&(prefix, k)) {
                for b in &sp.full {
                    if let BlockId::Tree { idx, .. } = b {
                        nf.insert(*idx, b.clone());
                    }
                }
                for b in &sp.win {
                    if let BlockId::Tree { idx, .. } = b {
                        nw.insert(*idx, b.clone());
                    }
                }
            }
        }
        let slot = &mut self.lanes[lane];
        slot.state = LaneState::Restoring;
        slot.pos = k;
        slot.full = nf;
        slot.win = nw;
        slot.reservation = wc;
        slot.restore_io_block = None;
        self.sync_ref_counts();
        Ok(())
    }

    /// Atomic single-block disk restore (`CacheScheduler.tla::RestoreBlock`), respecting
    /// longest-waiting restoring lane priority.
    pub fn restore_block(&mut self, lane: usize, block: &BlockId) -> Result<(), SchedError> {
        self.check_lane_idx(lane)?;
        if self.lanes[lane].state != LaneState::Restoring {
            return Err(SchedError::InvalidState {
                lane,
                expected: "Restoring",
                actual: self.lanes[lane].state,
            });
        }
        if !self.lane_block_set(lane).contains(block) {
            return Err(SchedError::Precondition("block not owned by restoring lane"));
        }
        if self.block_tier(block) != BlockTier::Disk {
            return Err(SchedError::Precondition("block is not disk-only"));
        }
        if self.mem_used() >= self.cfg.mem_budget {
            return Err(SchedError::Precondition("memory budget full during restore"));
        }
        let wl = self.lanes[lane].wait_ticks;
        let active_max = (0..self.lanes.len())
            .filter(|&r| {
                self.lanes[r].state == LaneState::Restoring
                    && self
                        .lane_block_set(r)
                        .iter()
                        .any(|rb| self.block_tier(rb) == BlockTier::Disk)
            })
            .map(|r| self.lanes[r].wait_ticks)
            .max()
            .unwrap_or(0);
        if wl < active_max {
            return Err(SchedError::Precondition("longer-waiting restoring lane must read first"));
        }
        if let Some(rec) = self.blocks.get_mut(block) {
            rec.tier = BlockTier::Both;
        }
        Ok(())
    }

    /// Begin an asynchronous disk restore read for `lane`, arming the cancel-in-restore
    /// barrier so a disconnect during disk I/O enters `Cancelling` and holds slot + blocks
    /// until `complete_restore_io(lane)` finishes (`docs/ROADMAP.md:135`).
    pub fn begin_restore_io(&mut self, lane: usize, block: &BlockId) -> Result<(), SchedError> {
        self.check_lane_idx(lane)?;
        if self.lanes[lane].state != LaneState::Restoring {
            return Err(SchedError::InvalidState {
                lane,
                expected: "Restoring",
                actual: self.lanes[lane].state,
            });
        }
        if self.lanes[lane].restore_io_block.is_some() {
            return Err(SchedError::Precondition("restore I/O already in flight for lane"));
        }
        if !self.lane_block_set(lane).contains(block) {
            return Err(SchedError::Precondition("block not owned by restoring lane"));
        }
        if self.block_tier(block) != BlockTier::Disk {
            return Err(SchedError::Precondition("block is not disk-only"));
        }
        if self.mem_used() >= self.cfg.mem_budget {
            return Err(SchedError::Precondition("memory budget full during restore"));
        }
        self.lanes[lane].restore_io_block = Some(block.clone());
        Ok(())
    }

    /// Complete an in-flight async restore read. If the lane was cancelled while reading
    /// (`Cancelling`), it now transitions cleanly to `Cancelled` and releases its slot,
    /// reservation, and block references.
    pub fn complete_restore_io(&mut self, lane: usize) -> Result<LaneState, SchedError> {
        self.check_lane_idx(lane)?;
        let Some(block) = self.lanes[lane].restore_io_block.take() else {
            return Err(SchedError::Precondition("no restore I/O in flight for lane"));
        };
        match self.lanes[lane].state {
            LaneState::Restoring => {
                if let Some(rec) = self.blocks.get_mut(&block) {
                    if rec.tier == BlockTier::Disk {
                        rec.tier = BlockTier::Both;
                    }
                }
                Ok(LaneState::Restoring)
            }
            LaneState::Cancelling => {
                self.release_lane_to(lane, LaneState::Cancelled);
                Ok(LaneState::Cancelled)
            }
            other => Err(SchedError::InvalidState {
                lane,
                expected: "Restoring|Cancelling",
                actual: other,
            }),
        }
    }

    /// Finish restoring once every owned block is resident in memory (`Restoring -> Ready`),
    /// preserving `wait_ticks` into `Ready`.
    pub fn finish_restore(&mut self, lane: usize) -> Result<(), SchedError> {
        self.check_lane_idx(lane)?;
        if self.lanes[lane].state != LaneState::Restoring {
            return Err(SchedError::InvalidState {
                lane,
                expected: "Restoring",
                actual: self.lanes[lane].state,
            });
        }
        if self.lanes[lane].restore_io_block.is_some() {
            return Err(SchedError::Precondition("restore disk I/O still in flight"));
        }
        if !self
            .lane_block_set(lane)
            .iter()
            .all(|b| self.block_tier(b).in_memory())
        {
            return Err(SchedError::Precondition("not all lane blocks are in memory"));
        }
        self.lanes[lane].state = LaneState::Ready;
        Ok(())
    }

    /// Maximum single-step prefill span allowed for a prefilling lane under `max_span`,
    /// block boundary alignment, and `prompt_end`.
    fn max_prefill_span(&self, lane: usize) -> u32 {
        let slot = &self.lanes[lane];
        let pe = prompt_end(slot.target, self.cfg.block_size);
        let to_pe = pe.saturating_sub(slot.pos);
        let to_blk = self.cfg.block_size - (slot.pos % self.cfg.block_size);
        self.cfg.max_span.min(to_pe).min(to_blk)
    }

    fn violates_bypass_ceiling(&self, bypassed: &BTreeSet<usize>) -> bool {
        bypassed.iter().any(|&w| self.lanes[w].wait_ticks >= self.cfg.max_bypass) // MUTANT_ANCHOR_BYPASS_CEILING
    }

    /// Deterministically plan and launch the next step (`Ready -> InStep`):
    /// 1. Block step launch if any non-Ready actionable waiting lane (`Queued`, `Admitted`,
    ///    `Restoring`) has reached the starvation ceiling `W` (`max_bypass`).
    /// 2. Pick the longest-waiting fitting `Ready` lane as leader (decoders win ties at `max_w`),
    ///    plus every other `Ready` lane at `wait_ticks >= W`.
    /// 3. Pack all remaining fitting 1-token decoding lanes up to `step_token_budget`.
    /// 4. Fill remaining `step_token_budget` with prefill chunks without starving fitting decoders.
    ///
    /// Returns `Ok(None)` when no valid step can start right now (e.g. step already active,
    /// no steppable ready lanes, a non-Ready lane hit `W` and must advance first, or memory
    /// reclaim is needed before allocating private tail blocks).
    pub fn build_step(&mut self) -> Result<Option<BuiltStep>, SchedError> {
        if self.step_active {
            return Ok(None);
        }
        let steppable: Vec<usize> = (0..self.lanes.len())
            .filter(|&l| self.is_steppable_ready(l))
            .collect();
        if steppable.is_empty() {
            return Ok(None);
        }

        let mem_used = self.mem_used();
        let mem_budget = self.cfg.mem_budget;
        let max_w = steppable
            .iter()
            .map(|&l| self.lanes[l].wait_ticks)
            .max()
            .unwrap_or(0);

        let has_max_decoder = steppable.iter().any(|&l| {
            self.is_decoding(l)
                && self.lanes[l].wait_ticks == max_w
                && mem_used + self.need_alloc(l) <= mem_budget
        });

        // 1. Leader: longest-waiting fitting Ready lane (decoder preferred on tie at max_w).
        let leader = steppable
            .iter()
            .copied()
            .filter(|&l| {
                self.lanes[l].wait_ticks == max_w
                    && mem_used + self.need_alloc(l) <= mem_budget
                    && (!has_max_decoder || self.is_decoding(l))
            })
            .min(); // lowest lane index tie-breaker
        let Some(leader) = leader else {
            return Ok(None);
        };

        let mut chosen_set = BTreeSet::new();
        let mut chosen_order = Vec::new();
        let mut spans: BTreeMap<usize, u32> = BTreeMap::new();
        let mut used_tokens: u32 = 0;
        let mut alloc_cnt: u32 = 0;

        // Helper closure-style additions: reserve 1 token first for leader and ceiling lanes.
        chosen_set.insert(leader);
        chosen_order.push(leader);
        spans.insert(leader, 1);
        used_tokens += 1;
        alloc_cnt += self.need_alloc(leader);

        // 2. Any other Ready lane at the starvation ceiling W MUST be included so none exceeds W.
        let mut ceiling_ready: Vec<usize> = steppable
            .iter()
            .copied()
            .filter(|&l| l != leader && self.lanes[l].wait_ticks >= self.cfg.max_bypass)
            .collect();
        ceiling_ready.sort_by_key(|&l| {
            (
                std::cmp::Reverse(self.lanes[l].wait_ticks),
                !self.is_decoding(l),
                l,
            )
        });
        for l in ceiling_ready {
            let need = self.need_alloc(l);
            if used_tokens + 1 > self.cfg.step_token_budget
                || mem_used + alloc_cnt + need > mem_budget
            {
                break;
            }
            chosen_set.insert(l);
            chosen_order.push(l);
            spans.insert(l, 1);
            used_tokens += 1;
            alloc_cnt += need;
        }

        // 3. Active 1-token decoding lanes (descending wait_ticks, then lane index).
        let mut decoders: Vec<usize> = steppable
            .iter()
            .copied()
            .filter(|&l| !chosen_set.contains(&l) && self.is_decoding(l))
            .collect();
        decoders.sort_by_key(|&l| (std::cmp::Reverse(self.lanes[l].wait_ticks), l));
        for d in decoders {
            let need = self.need_alloc(d);
            if used_tokens + 1 <= self.cfg.step_token_budget
                && mem_used + alloc_cnt + need <= mem_budget
            {
                chosen_set.insert(d);
                chosen_order.push(d);
                spans.insert(d, 1);
                used_tokens += 1;
                alloc_cnt += need;
            }
        }

        // Check if any fitting decoder remained unchosen due to token budget saturation.
        let any_fit_unchosen_decoder = steppable.iter().any(|&d| {
            !chosen_set.contains(&d)
                && self.is_decoding(d)
                && mem_used + alloc_cnt + self.need_alloc(d) <= mem_budget
        });

        // 4. Prefill chunks sliced up to step_token_budget (only when no fitting decoder is starved).
        if !any_fit_unchosen_decoder {
            // Expand already-chosen prefill lanes first (leader, then ceiling lanes).
            for &l in &chosen_order {
                if self.is_prefilling(l) && used_tokens < self.cfg.step_token_budget {
                    let rem = self.cfg.step_token_budget - used_tokens;
                    let cur = spans[&l];
                    let cap = self.max_prefill_span(l);
                    let extra = cap.saturating_sub(cur).min(rem);
                    if extra > 0 {
                        spans.insert(l, cur + extra);
                        used_tokens += extra;
                    }
                }
            }
            // Add additional ready prefill lanes up to token & memory budget.
            let mut prefillers: Vec<usize> = steppable
                .iter()
                .copied()
                .filter(|&l| !chosen_set.contains(&l) && self.is_prefilling(l))
                .collect();
            prefillers.sort_by_key(|&l| (std::cmp::Reverse(self.lanes[l].wait_ticks), l));
            for p in prefillers {
                if used_tokens >= self.cfg.step_token_budget {
                    break;
                }
                let need = self.need_alloc(p);
                if mem_used + alloc_cnt + need > mem_budget {
                    continue;
                }
                let rem = self.cfg.step_token_budget - used_tokens;
                let span = self.max_prefill_span(p).min(rem);
                if span > 0 {
                    chosen_set.insert(p);
                    chosen_order.push(p);
                    spans.insert(p, span);
                    used_tokens += span;
                    alloc_cnt += need;
                }
            }
        }

        // Enforce starvation ceiling across Queued, Admitted, Restoring, and unchosen Ready lanes.
        let bypassed = self.step_bypassed(&chosen_set);
        if self.violates_bypass_ceiling(&bypassed) {
            return Ok(None);
        }

        // Allocate private unsealed zero-padded tail blocks for chosen lanes that need them.
        let bsz = self.cfg.block_size;
        let mut new_ids: Vec<(usize, BlockId, u32)> = Vec::new();
        for &l in &chosen_set {
            let bi = self.lanes[l].pos / bsz;
            if !self.lanes[l].full.contains_key(&bi) {
                let id = BlockId::Priv { lane: l, kind: BlockKind::Full };
                if self.block_tier(&id).is_present() {
                    return Err(SchedError::Precondition("private full tail already allocated"));
                }
                new_ids.push((l, id, bi));
            }
            if !self.lanes[l].win.contains_key(&bi) {
                let id = BlockId::Priv { lane: l, kind: BlockKind::Window };
                if self.block_tier(&id).is_present() {
                    return Err(SchedError::Precondition("private window tail already allocated"));
                }
                new_ids.push((l, id, bi));
            }
        }
        for (l, id, bi) in new_ids {
            match id {
                BlockId::Priv { kind: BlockKind::Full, .. } => {
                    self.lanes[l].full.insert(bi, id.clone());
                }
                BlockId::Priv { kind: BlockKind::Window, .. } => {
                    self.lanes[l].win.insert(bi, id.clone());
                }
                _ => unreachable!(),
            }
            self.blocks.insert(
                id,
                BlockRecord {
                    tier: BlockTier::Memory,
                    sealed: false,
                    idx: bi,
                    length: 0,
                    data: vec![0; bsz as usize],
                    ref_count: 1,
                },
            );
        }

        // Transition chosen lanes Ready -> InStep and compute step pins.
        let mut all_pins = BTreeSet::new();
        for &l in &chosen_set {
            self.lanes[l].state = LaneState::InStep;
            self.lanes[l].wait_ticks = 0;
            let pos = self.lanes[l].pos;
            let sp = spans[&l];
            for bi in full_indices(pos + sp, bsz) {
                if let Some(b) = self.lanes[l].full.get(&bi) {
                    all_pins.insert(b.clone());
                }
            }
            for bi in span_window_indices(pos, sp, self.cfg.window_size, bsz) {
                if let Some(b) = self.lanes[l].win.get(&bi) {
                    all_pins.insert(b.clone());
                }
            }
        }

        // Increment wait_ticks on every bypassed actionable waiting lane.
        for w in bypassed {
            if self.lanes[w].wait_ticks <= self.cfg.max_bypass {
                self.lanes[w].wait_ticks += 1; // MUTANT_ANCHOR_STEP_BYPASS
            }
        }

        self.sync_ref_counts();
        self.step_active = true;
        self.step_lanes = chosen_set.clone();
        self.step_span = vec![0; self.lanes.len()];
        for (&l, &sp) in &spans {
            self.step_span[l] = sp;
        }
        self.step_pins = all_pins.clone();
        self.gpu_done = false;

        Ok(Some(BuiltStep {
            lanes: chosen_order,
            spans,
            total_tokens: used_tokens,
            pins: all_pins,
        }))
    }

    /// Execute the GPU work of the active step, populating `[off..off+cnt]` in each chosen
    /// lane's private tail blocks and setting `gpu_done = true`.
    pub fn gpu_execute(&mut self) -> Result<(), SchedError> {
        if !self.step_active || self.gpu_done {
            return Err(SchedError::Precondition("gpu_execute requires active step with gpu_done == false"));
        }
        let bsz = self.cfg.block_size;
        let step_lanes: Vec<usize> = self.step_lanes.iter().copied().collect();
        for l in step_lanes {
            let pos = self.lanes[l].pos;
            let bi = pos / bsz;
            let off = (pos % bsz) as usize;
            let cnt = self.step_span[l] as usize;
            for bid in [self.lanes[l].full.get(&bi).cloned(), self.lanes[l].win.get(&bi).cloned()]
                .into_iter()
                .flatten()
            {
                if let Some(rec) = self.blocks.get_mut(&bid) {
                    if rec.tier.is_present() && !rec.sealed {
                        rec.length = (off + cnt) as u32;
                        for v in &mut rec.data[off..off + cnt] {
                            *v = 1;
                        }
                    }
                }
            }
        }
        self.gpu_done = true;
        Ok(())
    }

    /// Complete all pending disk I/O in `step_io`.
    pub fn complete_step_io(&mut self) -> Result<(), SchedError> {
        if self.step_io.is_empty() {
            return Err(SchedError::Precondition("no pending step I/O"));
        }
        if self.disk_used() + self.step_io.len() as u32 > self.cfg.disk_budget {
            return Err(SchedError::Precondition("disk budget exceeded on CompleteStepIO"));
        }
        let pending: Vec<BlockId> = self.step_io.iter().cloned().collect();
        for id in pending {
            let has_lane_ref = self.lane_referrers(&id) > 0;
            if let Some(rec) = self.blocks.get_mut(&id) {
                if rec.tier == BlockTier::Memory {
                    rec.tier = if has_lane_ref {
                        BlockTier::Both
                    } else {
                        BlockTier::Disk
                    };
                }
            }
        }
        self.step_io.clear();
        Ok(())
    }

    /// Retire the completed step (`FinishStep`): requires `gpu_done && step_io.is_empty()`.
    /// Surviving `InStep` lanes advance to `Ready`, seal full blocks & start points, and
    /// prune their sliding window; `Cancelling` lanes now transition to `Cancelled` and
    /// release their slot, blocks, and reservations cleanly (`docs/ROADMAP.md:135`).
    pub fn complete_step(&mut self) -> Result<CompletedStep, SchedError> {
        if !self.step_active || !self.gpu_done || !self.step_io.is_empty() {
            return Err(SchedError::Precondition(
                "complete_step requires step_active && gpu_done && step_io.is_empty()",
            ));
        }
        let bsz = self.cfg.block_size;
        let wsz = self.cfg.window_size;
        let step_lanes: Vec<usize> = self.step_lanes.iter().copied().collect();
        let surviving: Vec<usize> = step_lanes
            .iter()
            .copied()
            .filter(|&l| self.lanes[l].state == LaneState::InStep)
            .collect();
        let cancelling: Vec<usize> = step_lanes
            .iter()
            .copied()
            .filter(|&l| self.lanes[l].state == LaneState::Cancelling)
            .collect();

        let mut boundary_lanes: Vec<(usize, u32)> = Vec::new();
        for &l in &surviving {
            let npos = self.lanes[l].pos + self.step_span[l];
            if npos % bsz == 0 {
                boundary_lanes.push((l, npos));
            }
        }

        // Publish sealed tree blocks for boundary lanes and retire private tails.
        let mut published_tree: BTreeSet<BlockId> = BTreeSet::new();
        let mut retired_priv: BTreeSet<BlockId> = BTreeSet::new();
        for &(l, npos) in &boundary_lanes {
            let bi = (npos / bsz) - 1;
            let pfx = self.lanes[l].prefix.clone();
            published_tree.insert(BlockId::Tree { prefix: pfx.clone(), kind: BlockKind::Full, idx: bi });
            published_tree.insert(BlockId::Tree { prefix: pfx, kind: BlockKind::Window, idx: bi });
            retired_priv.insert(BlockId::Priv { lane: l, kind: BlockKind::Full });
            retired_priv.insert(BlockId::Priv { lane: l, kind: BlockKind::Window });
        }
        for &l in &cancelling {
            retired_priv.insert(BlockId::Priv { lane: l, kind: BlockKind::Full });
            retired_priv.insert(BlockId::Priv { lane: l, kind: BlockKind::Window });
        }

        for id in &retired_priv {
            self.blocks.remove(id);
        }
        for id in &published_tree {
            let idx = match id {
                BlockId::Tree { idx, .. } => *idx,
                _ => unreachable!(),
            };
            let prev_tier = self.block_tier(id);
            let next_tier = if prev_tier.on_disk() {
                BlockTier::Both
            } else {
                BlockTier::Memory
            };
            self.blocks.insert(
                id.clone(),
                BlockRecord {
                    tier: next_tier,
                    sealed: true,
                    idx,
                    length: bsz,
                    data: vec![1; bsz as usize],
                    ref_count: 0,
                },
            );
        }

        // Update surviving lanes: advance position, bind sealed blocks, prune window to History(npos).
        for &l in &surviving {
            let npos = self.lanes[l].pos + self.step_span[l];
            let pfx = self.lanes[l].prefix.clone();
            self.lanes[l].pos = npos;
            if npos % bsz == 0 {
                let bi = (npos / bsz) - 1;
                self.lanes[l].full.insert(
                    bi,
                    BlockId::Tree { prefix: pfx.clone(), kind: BlockKind::Full, idx: bi },
                );
                self.lanes[l].win.insert(
                    bi,
                    BlockId::Tree { prefix: pfx.clone(), kind: BlockKind::Window, idx: bi },
                );
            }
            let keep = history_indices(npos, wsz, bsz);
            self.lanes[l].win.retain(|bi, _| keep.contains(bi));
            self.lanes[l].state = LaneState::Ready;
        }

        // Record kept start points at configured block stride and prompt ends.
        let stride = self.cfg.start_point_stride_blocks;
        for &(l, npos) in &boundary_lanes {
            let blk_cnt = npos / bsz;
            let at_prompt_end = npos == prompt_end(self.lanes[l].target, bsz) || npos == self.lanes[l].target;
            if !at_prompt_end && blk_cnt % stride != 0 {
                continue;
            }
            let pfx = self.lanes[l].prefix.clone();
            let full = full_indices(npos, bsz)
                .into_iter()
                .map(|i| BlockId::Tree { prefix: pfx.clone(), kind: BlockKind::Full, idx: i })
                .collect();
            let win = history_indices(npos, wsz, bsz)
                .into_iter()
                .map(|i| BlockId::Tree { prefix: pfx.clone(), kind: BlockKind::Window, idx: i })
                .collect();
            self.start_points.insert(
                (pfx, npos),
                StartPoint { kept: true, state_kept: true, full, win },
            );
        }

        // Release cancelling lanes now that GPU work and step I/O are complete.
        let default_prefix = self.cfg.default_prefix().to_string();
        let default_target = self.cfg.default_target();
        for &l in &cancelling {
            let slot = &mut self.lanes[l];
            slot.state = LaneState::Cancelled;
            slot.prefix = default_prefix.clone();
            slot.target = default_target;
            slot.pos = 0;
            slot.full.clear();
            slot.win.clear();
            slot.reservation = 0;
            slot.admit_budget = 0;
            slot.restore_io_block = None;
        }

        self.step_active = false;
        self.step_lanes.clear();
        self.step_span.fill(0);
        self.step_pins.clear();
        self.gpu_done = false;
        self.sync_ref_counts();

        Ok(CompletedStep {
            survived: surviving,
            cancelled: cancelling,
            sealed_boundaries: boundary_lanes,
        })
    }

    /// Cancel a lane.
    /// - If `InStep`: enters `Cancelling` and holds slot + KV blocks + pins + reservation
    ///   until `complete_step()` completes GPU work and step I/O (`docs/ROADMAP.md:135`).
    /// - If `Restoring` with active async restore disk I/O: enters `Cancelling` and holds
    ///   slot + blocks + reservation until `complete_restore_io(lane)` finishes.
    /// - Otherwise (`Queued`, `Admitted`, `Ready`, or `Restoring` with no active I/O):
    ///   transitions immediately to `Cancelled` and frees resources cleanly.
    pub fn cancel(&mut self, lane: usize) -> Result<LaneState, SchedError> {
        self.check_lane_idx(lane)?;
        match self.lanes[lane].state {
            LaneState::InStep => {
                self.lanes[lane].state = LaneState::Cancelling; // MUTANT_ANCHOR_CANCEL_IN_STEP
                Ok(LaneState::Cancelling)
            }
            LaneState::Restoring if self.lanes[lane].restore_io_block.is_some() => {
                self.lanes[lane].state = LaneState::Cancelling;
                Ok(LaneState::Cancelling)
            }
            LaneState::Queued | LaneState::Admitted | LaneState::Restoring | LaneState::Ready => {
                if self.lanes[lane].state == LaneState::Queued {
                    self.queue_seq.retain(|&q| q != lane);
                }
                self.release_lane_to(lane, LaneState::Cancelled);
                Ok(LaneState::Cancelled)
            }
            other => Err(SchedError::InvalidState {
                lane,
                expected: "Queued|Admitted|Restoring|Ready|InStep",
                actual: other,
            }),
        }
    }

    /// Finish a `Ready` lane that has reached its target length (`Ready -> Finished`).
    pub fn finish_ready(&mut self, lane: usize) -> Result<(), SchedError> {
        self.check_lane_idx(lane)?;
        if self.lanes[lane].state != LaneState::Ready {
            return Err(SchedError::InvalidState {
                lane,
                expected: "Ready",
                actual: self.lanes[lane].state,
            });
        }
        if self.lanes[lane].pos != self.lanes[lane].target {
            return Err(SchedError::Precondition("lane has not reached target length"));
        }
        self.release_lane_to(lane, LaneState::Finished);
        Ok(())
    }

    fn release_lane_to(&mut self, lane: usize, target_state: LaneState) {
        let default_prefix = self.cfg.default_prefix().to_string();
        let default_target = self.cfg.default_target();
        let slot = &mut self.lanes[lane];
        slot.state = target_state;
        slot.prefix = default_prefix;
        slot.target = default_target;
        slot.pos = 0;
        slot.full.clear();
        slot.win.clear();
        slot.reservation = 0;
        slot.admit_budget = 0;
        slot.wait_ticks = 0;
        slot.restore_io_block = None;
        self.sync_ref_counts();
    }

    // ---- Memory Pressure, Cold Spill & Window Maintenance -----------------------------------

    pub fn evict_clean_or_unref_mem(&mut self) -> bool {
        if !self.mem_pressure() {
            return false;
        }
        let victim = self.cold_mem_blocks().into_iter().find(|id| {
            let rec = &self.blocks[id];
            rec.tier.on_disk() || rec.ref_count == 0
        });
        let Some(id) = victim else {
            return false;
        };
        let rec = &self.blocks[&id];
        if rec.ref_count == 0 && !rec.tier.on_disk() {
            self.blocks.remove(&id);
        } else if let Some(r) = self.blocks.get_mut(&id) {
            r.tier = BlockTier::Disk;
        }
        true
    }

    pub fn start_cold_writeback(&mut self) -> Option<BlockId> {
        if !self.step_io.is_empty() || !self.mem_pressure() {
            return None;
        }
        let has_clean_or_unref = self.cold_mem_blocks().iter().any(|id| {
            let rec = &self.blocks[id];
            rec.tier.on_disk() || rec.ref_count == 0
        });
        if has_clean_or_unref || self.disk_used() >= self.cfg.disk_budget {
            return None;
        }
        let victim = self.cold_mem_blocks().into_iter().find(|id| {
            let rec = &self.blocks[id];
            rec.tier == BlockTier::Memory && rec.sealed
        })?;
        self.step_io.insert(victim.clone());
        Some(victim)
    }

    pub fn reclaim_cold_disk(&mut self) -> bool {
        if self.disk_used() + (self.step_io.len() as u32) < self.cfg.disk_budget {
            return false;
        }
        let cold_disk = self.cold_disk_blocks();
        let has_unref_disk = cold_disk.iter().any(|id| self.blocks[id].ref_count == 0);
        if !self.mem_pressure() && !has_unref_disk {
            return false;
        }
        if let Some(id) = cold_disk.into_iter().find(|id| self.blocks[id].ref_count == 0) {
            if self.blocks[&id].tier.in_memory() {
                self.blocks.get_mut(&id).unwrap().tier = BlockTier::Memory;
            } else {
                self.blocks.remove(&id);
            }
            return true;
        }
        let cold_mem: BTreeSet<BlockId> = self.cold_mem_blocks().into_iter().collect();
        let cold_disk_set: BTreeSet<BlockId> = self.cold_disk_blocks().into_iter().collect();
        let victim_pk = self
            .start_points
            .iter()
            .find(|(_, sp)| {
                sp.kept
                    && sp.full.iter().chain(&sp.win).all(|b| !self.step_io.contains(b))
                    && sp
                        .full
                        .iter()
                        .chain(&sp.win)
                        .any(|b| cold_disk_set.contains(b) || cold_mem.contains(b))
            })
            .map(|(pk, _)| pk.clone());
        if let Some(pk) = victim_pk {
            self.start_points.remove(&pk);
            self.sync_ref_counts();
            return true;
        }
        false
    }

    pub fn drop_unreferenced(&mut self, id: &BlockId) -> bool {
        let Some(rec) = self.blocks.get(id) else {
            return false;
        };
        if !rec.tier.is_present()
            || rec.ref_count != 0
            || self.step_pins.contains(id)
            || self.step_io.contains(id)
        {
            return false;
        }
        self.blocks.remove(id);
        true
    }

    pub fn slide_window(&mut self, lane: usize) -> bool {
        if lane >= self.lanes.len() || self.lanes[lane].state != LaneState::Ready {
            return false;
        }
        let keep = history_indices(
            self.lanes[lane].pos,
            self.cfg.window_size,
            self.cfg.block_size,
        );
        let before = self.lanes[lane].win.len();
        self.lanes[lane].win.retain(|bi, _| keep.contains(bi));
        if self.lanes[lane].win.len() != before {
            self.sync_ref_counts();
            true
        } else {
            false
        }
    }

    // ---- Runtime Safety Invariant Checker ---------------------------------------------------

    pub fn check_invariants(&self) -> Result<(), String> {
        let bsz = self.cfg.block_size;
        let wsz = self.cfg.window_size;

        // 1. BudgetsAndReservations
        if self.mem_used() > self.cfg.mem_budget {
            return Err(format!("MemUsed {} > MemBudget {}", self.mem_used(), self.cfg.mem_budget));
        }
        if self.disk_used() > self.cfg.disk_budget {
            return Err(format!("DiskUsed {} > DiskBudget {}", self.disk_used(), self.cfg.disk_budget));
        }
        if self.reserved_total() > self.cfg.mem_budget {
            return Err(format!(
                "ReservedTotal {} > MemBudget {}",
                self.reserved_total(),
                self.cfg.mem_budget
            ));
        }

        // 2. BypassBoundHolds
        for (i, slot) in self.lanes.iter().enumerate() {
            if slot.state.is_waiting() && slot.wait_ticks > self.cfg.max_bypass {
                return Err(format!(
                    "lane {i} ({:?}) wait_ticks {} > max_bypass {}",
                    slot.state, slot.wait_ticks, self.cfg.max_bypass
                ));
            }
            if matches!(slot.state, LaneState::Restoring | LaneState::Ready | LaneState::InStep) {
                let ws = self.current_working_set(i);
                if slot.reservation < ws {
                    return Err(format!(
                        "lane {i} reservation {} < current_working_set {}",
                        slot.reservation, ws
                    ));
                }
                for bi in full_indices(slot.pos, bsz) {
                    let Some(bid) = slot.full.get(&bi) else {
                        return Err(format!("lane {i} missing full block {bi}"));
                    };
                    if !self.block_tier(bid).is_present() {
                        return Err(format!("lane {i} full block {bi} not present"));
                    }
                }
                for bi in history_indices(slot.pos, wsz, bsz) {
                    let Some(bid) = slot.win.get(&bi) else {
                        return Err(format!("lane {i} missing window block {bi}"));
                    };
                    if !self.block_tier(bid).is_present() {
                        return Err(format!("lane {i} window block {bi} not present"));
                    }
                }
            }
        }

        // 3. QueueBounded
        let queued_cnt = self
            .lanes
            .iter()
            .filter(|l| l.state == LaneState::Queued)
            .count();
        if queued_cnt != self.queue_seq.len() || queued_cnt > self.cfg.max_queue {
            return Err(format!(
                "queue invariant broken: queued={queued_cnt}, seq={}, max={}",
                self.queue_seq.len(),
                self.cfg.max_queue
            ));
        }

        // 4. StepBlocksInMemoryAndOwned & StepBudgetHolds
        if self.step_active {
            let total_tokens: u32 = self.step_lanes.iter().map(|&l| self.step_span[l]).sum();
            if total_tokens > self.cfg.step_token_budget {
                return Err(format!(
                    "step tokens {} > step_token_budget {}",
                    total_tokens, self.cfg.step_token_budget
                ));
            }
            for &l in &self.step_lanes {
                let st = self.lanes[l].state;
                if !matches!(st, LaneState::InStep | LaneState::Cancelling) {
                    return Err(format!("step lane {l} in illegal state {st:?}"));
                }
                let pos = self.lanes[l].pos;
                let sp = self.step_span[l];
                if pos >= prompt_end(self.lanes[l].target, bsz) && sp != 1 {
                    return Err(format!("decoding lane {l} has span {sp} != 1"));
                }
                for bi in full_indices(pos + sp, bsz) {
                    let bid = self.lanes[l]
                        .full
                        .get(&bi)
                        .ok_or_else(|| format!("step lane {l} missing full block {bi}"))?;
                    if !self.step_pins.contains(bid) {
                        return Err(format!("step lane {l} full block {bi} not pinned"));
                    }
                }
                for bi in span_window_indices(pos, sp, wsz, bsz) {
                    let bid = self.lanes[l]
                        .win
                        .get(&bi)
                        .ok_or_else(|| format!("step lane {l} missing window block {bi}"))?;
                    if !self.step_pins.contains(bid) {
                        return Err(format!("step lane {l} window block {bi} not pinned"));
                    }
                }
            }
            for b in &self.step_pins {
                if !self.block_tier(b).in_memory() {
                    return Err(format!("pinned block {b:?} not in memory"));
                }
                if self.lane_referrers(b) == 0 {
                    return Err(format!("pinned block {b:?} has no lane referrer"));
                }
            }
        }

        // 5. RefCountMatchesReferrers & UnsealedPrivateInvisibleZero (O(N) count pass)
        let mut expected_counts: BTreeMap<BlockId, usize> = BTreeMap::new();
        for l in &self.lanes {
            if l.state.is_admitted() {
                for b in l.full.values().chain(l.win.values()) {
                    *expected_counts.entry(b.clone()).or_insert(0) += 1;
                }
            }
        }
        for sp in self.start_points.values() {
            if sp.kept {
                for b in sp.full.iter().chain(&sp.win) {
                    *expected_counts.entry(b.clone()).or_insert(0) += 1;
                }
            }
        }
        for (id, rec) in &self.blocks {
            let expected_refs = expected_counts.get(id).copied().unwrap_or(0);
            if rec.ref_count != expected_refs {
                return Err(format!(
                    "block {id:?} ref_count {} != expected {expected_refs}",
                    rec.ref_count
                ));
            }
            if !rec.sealed {
                if !matches!(id, BlockId::Priv { .. }) {
                    return Err(format!("unsealed block {id:?} is not private"));
                }
                if rec.tier != BlockTier::Memory || rec.ref_count != 1 {
                    return Err(format!("unsealed private block {id:?} has invalid tier/refs"));
                }
                for &v in &rec.data[rec.length as usize..] {
                    if v != 0 {
                        return Err(format!("unsealed private block {id:?} has non-zero tail"));
                    }
                }
            }
        }
        Ok(())
    }
}

// ---- Production Server Step-Planning Adapter (`src/serve.rs`) -------------------------------

/// Lightweight snapshot of one `serve.rs` lane for starvation-bounded step planning.
#[derive(Clone, Copy, Debug)]
pub struct ServeLaneInput {
    pub runnable: bool,
    pub n_past: usize,
    pub pf_end: usize,
    pub has_pending_decode: bool,
}

impl ServeLaneInput {
    pub fn needs_prefill(&self) -> bool {
        self.runnable && self.n_past < self.pf_end
    }

    pub fn needs_decode(&self) -> bool {
        self.runnable && self.n_past >= self.pf_end && self.has_pending_decode
    }
}

/// Planned step batch for `src/serve.rs`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServeStepPlan {
    /// Batched 1-token decode over these lane indices.
    Decode(Vec<usize>),
    /// Chunked prefill spans `(lane_idx, p0, len)` up to `step_token_budget`.
    Prefill(Vec<(usize, usize, usize)>),
}

/// Real-server lane lifecycle tracker, starvation-bounded step builder (`wait_ticks <= W`),
/// and cancel-in-step barrier used directly inside `src/serve.rs::scheduler`.
pub struct ServeStepPlanner {
    states: Vec<LaneState>,
    wait_ticks: Vec<u32>,
    in_step: Vec<usize>,
    max_bypass: u32,
}

impl ServeStepPlanner {
    pub fn new(nlanes: usize, max_bypass: u32) -> Self {
        assert!(max_bypass >= 1, "max_bypass (W) must be >= 1");
        Self {
            states: vec![LaneState::Idle; nlanes],
            wait_ticks: vec![0; nlanes],
            in_step: Vec::new(),
            max_bypass,
        }
    }

    pub fn state(&self, lane: usize) -> LaneState {
        self.states[lane]
    }

    pub fn wait_ticks(&self, lane: usize) -> u32 {
        self.wait_ticks[lane]
    }

    pub fn max_bypass(&self) -> u32 {
        self.max_bypass
    }

    pub fn on_admit(&mut self, lane: usize) {
        self.states[lane] = LaneState::Ready;
        self.wait_ticks[lane] = 0;
    }

    pub fn on_finish(&mut self, lane: usize) {
        self.states[lane] = LaneState::Finished;
        self.wait_ticks[lane] = 0;
    }

    /// Cancel a lane. If it is currently `InStep`, transition to `Cancelling` and return
    /// `false` so the caller NEVER releases the GPU lane slot until `complete_step()` runs!
    /// If not `InStep`, transition to `Cancelled` and return `true` (safe to release now).
    pub fn request_cancel(&mut self, lane: usize) -> bool {
        match self.states[lane] {
            LaneState::InStep | LaneState::Cancelling => {
                self.states[lane] = LaneState::Cancelling;
                false
            }
            _ => {
                self.states[lane] = LaneState::Cancelled;
                self.wait_ticks[lane] = 0;
                true
            }
        }
    }

    /// Deterministically plan the next GPU step (decode-first with strict starvation ceiling `W`):
    /// 1. Active 1-token decoding lanes (`need_dec`) always win whenever `!need_dec.is_empty()`,
    ///    UNLESS a runnable prefiller reached the starvation ceiling `W` (`max_pf_w >= W`)
    ///    while no runnable decoder is at/above it (`max_dec_w < max_pf_w`).
    /// 2. Otherwise prefill chunks sliced up to `step_token_budget` (aligned to 64-token
    ///    boundaries unless finishing a prompt), reserving >= 1 token per ceiling prefiller.
    ///
    /// Transitions chosen lanes `Ready -> InStep`, resets their `wait_ticks = 0`, and
    /// increments `wait_ticks` on every bypassed runnable lane (enforcing `wait_ticks <= W`).
    pub fn plan_and_begin_step(
        &mut self,
        inputs: &[ServeLaneInput],
        step_token_budget: usize,
        block_align: usize,
    ) -> Option<ServeStepPlan> {
        debug_assert!(self.in_step.is_empty(), "previous step must complete before planning next");
        let need_dec: Vec<usize> = inputs
            .iter()
            .enumerate()
            .filter(|&(_, inp)| inp.needs_decode())
            .map(|(i, _)| i)
            .collect();
        let need_pf: Vec<usize> = inputs
            .iter()
            .enumerate()
            .filter(|&(_, inp)| inp.needs_prefill())
            .map(|(i, _)| i)
            .collect();

        if need_dec.is_empty() && need_pf.is_empty() {
            return None;
        }

        let max_dec_w = need_dec.iter().map(|&i| self.wait_ticks[i]).max();
        let max_pf_w = need_pf.iter().map(|&i| self.wait_ticks[i]).max();

        // Pick decode vs prefill batch:
        // - Active decoders burst up to W steps while all prefillers remain below W.
        // - Once a prefiller reaches W (and no decoder is at/above pw), prefill preempts so wait_ticks <= W.
        let choose_decode = match (max_dec_w, max_pf_w) {
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (Some(dw), Some(pw)) => !(pw >= self.max_bypass && dw < pw),
            (None, None) => return None,
        };

        let (plan, chosen): (ServeStepPlan, Vec<usize>) = if choose_decode {
            let mut dec = need_dec.clone();
            dec.sort_by_key(|&i| (std::cmp::Reverse(self.wait_ticks[i]), i));
            let cap = step_token_budget.max(1);
            dec.truncate(cap);
            (ServeStepPlan::Decode(dec.clone()), dec)
        } else {
            let mut pf = need_pf.clone();
            // Longest-waiting (and ceiling >= W) prefill lanes first.
            pf.sort_by_key(|&i| (std::cmp::Reverse(self.wait_ticks[i]), i));
            let ceiling_cnt = pf.iter().filter(|&&i| self.wait_ticks[i] >= self.max_bypass).count();
            let mut budget = step_token_budget.max(1);
            let mut spans = Vec::new();
            let mut chosen = Vec::new();
            for (idx, &i) in pf.iter().enumerate() {
                if budget == 0 {
                    break;
                }
                let inp = inputs[i];
                let rem_ceiling_after = ceiling_cnt.saturating_sub(idx + 1);
                let max_for_this = budget.saturating_sub(rem_ceiling_after).max(1);
                let raw_t = (inp.pf_end - inp.n_past).min(max_for_this);
                // Keep 64-token start-point alignment when chunk does not finish the prompt.
                let t = if block_align > 0 && inp.n_past + raw_t < inp.pf_end && raw_t > block_align
                {
                    let aligned = raw_t - ((inp.n_past + raw_t) % block_align);
                    if aligned > 0 { aligned } else { raw_t }
                } else {
                    raw_t
                };
                spans.push((i, inp.n_past, t));
                chosen.push(i);
                budget -= t;
            }
            (ServeStepPlan::Prefill(spans), chosen)
        };

        let chosen_set: BTreeSet<usize> = chosen.iter().copied().collect();
        for &i in &chosen {
            self.states[i] = LaneState::InStep;
            self.wait_ticks[i] = 0;
        }
        for &i in need_dec.iter().chain(&need_pf) {
            if !chosen_set.contains(&i) {
                self.wait_ticks[i] = (self.wait_ticks[i] + 1).min(self.max_bypass);
            }
        }
        self.in_step = chosen;
        Some(plan)
    }

    /// Complete the in-flight GPU step: surviving `InStep` lanes return to `Ready`, while
    /// any lane marked `Cancelling` mid-step now transitions to `Cancelled` and is returned
    /// so the caller releases its GPU lane slot ONLY AFTER GPU work completed.
    pub fn complete_step(&mut self) -> Vec<usize> {
        let mut cancelled = Vec::new();
        for l in self.in_step.drain(..) {
            if self.states[l] == LaneState::Cancelling {
                self.states[l] = LaneState::Cancelled;
                self.wait_ticks[l] = 0;
                cancelled.push(l);
            } else if self.states[l] == LaneState::InStep {
                self.states[l] = LaneState::Ready;
            }
        }
        cancelled
    }
}

#[cfg(test)]
mod tests;
