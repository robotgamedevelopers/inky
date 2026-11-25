//! Reference scheduler snapshots, seeded trace walks, and TLC bundle emission.

use super::*;

// ---- TLA+ Trace Capture & Bundle Rendering (`specs/CacheScheduler.tla`) ---------------------

/// One recorded state snapshot for TLC trace conformance checking.
#[derive(Clone, Debug)]
pub struct TraceStep {
    pub action: String,
    pub blocks: BTreeMap<BlockId, BlockRecord>,
    pub start_points: BTreeMap<(String, u32), StartPoint>,
    pub lanes: Vec<LaneSlot>,
    pub queue_seq: Vec<usize>,
    pub step_active: bool,
    pub step_lanes: BTreeSet<usize>,
    pub step_span: Vec<u32>,
    pub step_pins: BTreeSet<BlockId>,
    pub step_io: BTreeSet<BlockId>,
    pub gpu_done: bool,
}

#[derive(Clone, Debug)]
pub struct SchedulerTrace {
    pub config: SchedulerConfig,
    pub steps: Vec<TraceStep>,
}

impl SchedulerTrace {
    pub fn new(sched: &Scheduler) -> Self {
        let mut t = Self {
            config: sched.cfg.clone(),
            steps: Vec::new(),
        };
        t.record("Init", sched);
        t
    }

    pub fn record(&mut self, action: &str, sched: &Scheduler) {
        self.steps.push(TraceStep {
            action: action.to_string(),
            blocks: sched.blocks.clone(),
            start_points: sched.start_points.clone(),
            lanes: sched.lanes.clone(),
            queue_seq: sched.queue_seq.clone(),
            step_active: sched.step_active,
            step_lanes: sched.step_lanes.clone(),
            step_span: sched.step_span.clone(),
            step_pins: sched.step_pins.clone(),
            step_io: sched.step_io.clone(),
            gpu_done: sched.gpu_done,
        });
    }
}

fn tla_str(s: &str) -> String {
    format!("\"{s}\"")
}

fn tla_block_id(id: &BlockId, lane_names: &[String]) -> String {
    match id {
        BlockId::Tree { prefix, kind, idx } => {
            format!("<<\"tree\", \"{}\", \"{}\", {}>>", prefix, kind.as_tla(), idx)
        }
        BlockId::Priv { lane, kind } => {
            format!("<<\"priv\", \"{}\", \"{}\">>", lane_names[*lane], kind.as_tla())
        }
    }
}

fn tla_block_set(set: &BTreeSet<BlockId>, lane_names: &[String]) -> String {
    let items: Vec<String> = set.iter().map(|b| tla_block_id(b, lane_names)).collect();
    format!("{{{}}}", items.join(", "))
}

fn tla_lane_set(set: &BTreeSet<usize>, lane_names: &[String]) -> String {
    let items: Vec<String> = set.iter().map(|&l| tla_str(&lane_names[l])).collect();
    format!("{{{}}}", items.join(", "))
}

fn tla_seq_strings(items: &[String]) -> String {
    format!("<<{}>>", items.join(", "))
}

fn render_trace_step(cfg: &SchedulerConfig, step: &TraceStep) -> String {
    let ln = &cfg.lanes;
    let n_blks = cfg.num_blocks();

    // Only serialize present blocks as an association tuple; TraceCheck hydrates the full domain.
    let mut blk_entries = Vec::new();
    for (id, rec) in &step.blocks {
        if !rec.tier.is_present() {
            continue;
        }
        let data_seq: Vec<String> = rec.data.iter().map(|v| v.to_string()).collect();
        blk_entries.push(format!(
            "[id |-> {}, tier |-> \"{}\", sealed |-> {}, idx |-> {}, length |-> {}, data |-> <<{}>>, refCount |-> {}]",
            tla_block_id(id, ln),
            rec.tier.as_tla(),
            if rec.sealed { "TRUE" } else { "FALSE" },
            rec.idx,
            rec.length,
            data_seq.join(", "),
            rec.ref_count
        ));
    }

    let mut sp_entries = Vec::new();
    for ((pfx, k), sp) in &step.start_points {
        if !sp.kept {
            continue;
        }
        sp_entries.push(format!(
            "[prefix |-> \"{}\", boundary |-> {}, stateKept |-> {}, full |-> {}, win |-> {}]",
            pfx,
            k,
            if sp.state_kept { "TRUE" } else { "FALSE" },
            tla_block_set(&sp.full, ln),
            tla_block_set(&sp.win, ln)
        ));
    }

    let mut lane_entries = Vec::new();
    for (i, slot) in step.lanes.iter().enumerate() {
        let full_seq: Vec<String> = (0..n_blks)
            .map(|bi| {
                slot.full
                    .get(&bi)
                    .map_or_else(|| "<<\"none\">>".to_string(), |b| tla_block_id(b, ln))
            })
            .collect();
        let win_seq: Vec<String> = (0..n_blks)
            .map(|bi| {
                slot.win
                    .get(&bi)
                    .map_or_else(|| "<<\"none\">>".to_string(), |b| tla_block_id(b, ln))
            })
            .collect();
        lane_entries.push(format!(
            "[state |-> \"{}\", prefix |-> \"{}\", target |-> {}, pos |-> {}, full |-> {}, win |-> {}, reservation |-> {}, admitBudget |-> {}, waitTicks |-> {}, stepSpan |-> {}]",
            slot.state.as_tla(),
            slot.prefix,
            slot.target,
            slot.pos,
            tla_seq_strings(&full_seq),
            tla_seq_strings(&win_seq),
            slot.reservation,
            slot.admit_budget,
            slot.wait_ticks,
            step.step_span[i]
        ));
    }

    let q_items: Vec<String> = step.queue_seq.iter().map(|&l| tla_str(&ln[l])).collect();

    format!(
        "[blocks |-> {{{}}}, startPoints |-> {{{}}}, lanes |-> {}, queueSeq |-> {}, stepActive |-> {}, stepLanes |-> {}, stepPins |-> {}, stepIO |-> {}, gpuDone |-> {}]",
        blk_entries.join(", "),
        sp_entries.join(", "),
        tla_seq_strings(&lane_entries),
        tla_seq_strings(&q_items),
        if step.step_active { "TRUE" } else { "FALSE" },
        tla_lane_set(&step.step_lanes, ln),
        tla_block_set(&step.step_pins, ln),
        tla_block_set(&step.step_io, ln),
        if step.gpu_done { "TRUE" } else { "FALSE" }
    )
}

/// Render `(TraceData.tla, TraceCheck.tla, TraceCheck.cfg)` verifying one or more concrete
/// `SchedulerTrace` runs directly against `specs/CacheScheduler.tla` (`Init /\ [][Next]_vars`
/// plus all 12 safety invariants and 4 action properties).
pub fn render_cache_scheduler_tla_bundle(
    traces: &[SchedulerTrace],
) -> Result<(String, String, String), SchedError> {
    let Some(first) = traces.first() else {
        return Err(SchedError::Precondition("cannot render empty trace slice"));
    };
    let cfg = &first.config;
    cfg.validate()?;
    for t in traces {
        if t.config != *cfg || t.steps.len() < 2 {
            return Err(SchedError::Precondition(
                "all traces must share config and have at least 2 steps",
            ));
        }
    }

    let mut data = String::from(
        "---------------------------- MODULE TraceData ----------------------------\n\
         EXTENDS Integers, Sequences\n\n",
    );
    let mut trace_names = Vec::new();
    for (ti, tr) in traces.iter().enumerate() {
        let name = format!("Trace{}", ti + 1);
        data.push_str(&format!("{name} == <<\n"));
        for (si, st) in tr.steps.iter().enumerate() {
            let comma = if si + 1 < tr.steps.len() { "," } else { "" };
            data.push_str("  ");
            data.push_str(&render_trace_step(cfg, st));
            data.push_str(comma);
            data.push('\n');
        }
        data.push_str(">>\n\n");
        trace_names.push(name);
    }
    data.push_str(&format!(
        "Traces == <<{}>>\n=============================================================================\n",
        trace_names.join(", ")
    ));

    let lane_seq_lit = cfg
        .lanes
        .iter()
        .map(|s| tla_str(s))
        .collect::<Vec<_>>()
        .join(", ");
    let pfx_seq_lit = cfg
        .prefixes
        .iter()
        .map(|s| tla_str(s))
        .collect::<Vec<_>>()
        .join(", ");
    let req_set_lit = cfg
        .request_lengths
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(", ");

    let check = format!(
        "---------------------------- MODULE TraceCheck ----------------------------\n\
         EXTENDS CacheScheduler, TraceData, Sequences\n\
         \n\
         VARIABLES traceIdx, stepIdx\n\
         \n\
         LaneSeq == <<{lane_seq_lit}>>\n\
         LaneIdx(l) == CHOOSE i \\in 1..Len(LaneSeq) : LaneSeq[i] = l\n\
         \n\
         HydrateBlocks(bs) ==\n\
             [b \\in BlockIds |->\n\
                 IF \\E e \\in bs : e.id = b\n\
                 THEN LET e == CHOOSE x \\in bs : x.id = b\n\
                      IN [tier |-> e.tier,\n\
                          sealed |-> e.sealed,\n\
                          idx |-> e.idx,\n\
                          length |-> e.length,\n\
                          data |-> [i \\in 0..(BlockSize - 1) |-> e.data[i + 1]],\n\
                          refCount |-> e.refCount]\n\
                 ELSE EmptyBlock]\n\
         \n\
         HydrateStartPoints(sps) ==\n\
             [pk \\in Prefixes \\X Boundaries |->\n\
                 IF \\E e \\in sps : e.prefix = pk[1] /\\ e.boundary = pk[2]\n\
                 THEN LET e == CHOOSE x \\in sps : x.prefix = pk[1] /\\ x.boundary = pk[2]\n\
                      IN [kept |-> TRUE, stateKept |-> e.stateKept, full |-> e.full, win |-> e.win]\n\
                 ELSE EmptyStart]\n\
         \n\
         ApplyStep(s) ==\n\
             /\\ blocks      = HydrateBlocks(s.blocks)\n\
             /\\ startPoints = HydrateStartPoints(s.startPoints)\n\
             /\\ laneState   = [l \\in Lanes |-> s.lanes[LaneIdx(l)].state]\n\
             /\\ lanePrefix  = [l \\in Lanes |-> s.lanes[LaneIdx(l)].prefix]\n\
             /\\ laneTarget  = [l \\in Lanes |-> s.lanes[LaneIdx(l)].target]\n\
             /\\ lanePos     = [l \\in Lanes |-> s.lanes[LaneIdx(l)].pos]\n\
             /\\ laneFull    = [l \\in Lanes |-> [i \\in BlkIdx |-> s.lanes[LaneIdx(l)].full[i + 1]]]\n\
             /\\ laneWin     = [l \\in Lanes |-> [i \\in BlkIdx |-> s.lanes[LaneIdx(l)].win[i + 1]]]\n\
             /\\ reservation = [l \\in Lanes |-> s.lanes[LaneIdx(l)].reservation]\n\
             /\\ admitBudget = [l \\in Lanes |-> s.lanes[LaneIdx(l)].admitBudget]\n\
             /\\ waitTicks   = [l \\in Lanes |-> s.lanes[LaneIdx(l)].waitTicks]\n\
             /\\ queueSeq    = s.queueSeq\n\
             /\\ stepActive  = s.stepActive\n\
             /\\ stepLanes   = s.stepLanes\n\
             /\\ stepSpan    = [l \\in Lanes |-> s.lanes[LaneIdx(l)].stepSpan]\n\
             /\\ stepPins    = s.stepPins\n\
             /\\ stepIO      = s.stepIO\n\
             /\\ gpuDone     = s.gpuDone\n\
         \n\
         ApplyNext(s) ==\n\
             /\\ blocks'      = HydrateBlocks(s.blocks)\n\
             /\\ startPoints' = HydrateStartPoints(s.startPoints)\n\
             /\\ laneState'   = [l \\in Lanes |-> s.lanes[LaneIdx(l)].state]\n\
             /\\ lanePrefix'  = [l \\in Lanes |-> s.lanes[LaneIdx(l)].prefix]\n\
             /\\ laneTarget'  = [l \\in Lanes |-> s.lanes[LaneIdx(l)].target]\n\
             /\\ lanePos'     = [l \\in Lanes |-> s.lanes[LaneIdx(l)].pos]\n\
             /\\ laneFull'    = [l \\in Lanes |-> [i \\in BlkIdx |-> s.lanes[LaneIdx(l)].full[i + 1]]]\n\
             /\\ laneWin'     = [l \\in Lanes |-> [i \\in BlkIdx |-> s.lanes[LaneIdx(l)].win[i + 1]]]\n\
             /\\ reservation' = [l \\in Lanes |-> s.lanes[LaneIdx(l)].reservation]\n\
             /\\ admitBudget' = [l \\in Lanes |-> s.lanes[LaneIdx(l)].admitBudget]\n\
             /\\ waitTicks'   = [l \\in Lanes |-> s.lanes[LaneIdx(l)].waitTicks]\n\
             /\\ queueSeq'    = s.queueSeq\n\
             /\\ stepActive'  = s.stepActive\n\
             /\\ stepLanes'   = s.stepLanes\n\
             /\\ stepSpan'    = [l \\in Lanes |-> s.lanes[LaneIdx(l)].stepSpan]\n\
             /\\ stepPins'    = s.stepPins\n\
             /\\ stepIO'      = s.stepIO\n\
             /\\ gpuDone'     = s.gpuDone\n\
         \n\
         ReplayInit ==\n\
             /\\ traceIdx \\in 1..Len(Traces)\n\
             /\\ stepIdx = 1\n\
             /\\ ApplyStep(Traces[traceIdx][1])\n\
         \n\
         ReplayNext ==\n\
             /\\ stepIdx < Len(Traces[traceIdx])\n\
             /\\ traceIdx' = traceIdx\n\
             /\\ stepIdx' = stepIdx + 1\n\
             /\\ ApplyNext(Traces[traceIdx][stepIdx + 1])\n\
         \n\
         ReplaySpec == ReplayInit /\\ [][ReplayNext]_<<vars, traceIdx, stepIdx>>\n\
         Conforms == Init /\\ [][Next]_vars\n\
         =============================================================================\n"
    );

    let cfg_file = format!(
        "SPECIFICATION ReplaySpec\n\
         CHECK_DEADLOCK FALSE\n\
         CONSTANTS\n\
          Lanes = {{{lane_seq_lit}}}\n\
          Prefixes = {{{pfx_seq_lit}}}\n\
          RequestLengths = {{{req_set_lit}}}\n\
          BlockSize = {}\n\
          WindowSize = {}\n\
          MaxPos = {}\n\
          MaxSpan = {}\n\
          StepTokenBudget = {}\n\
          MemBudget = {}\n\
          DiskBudget = {}\n\
          MaxQueue = {}\n\
          MaxBypass = {}\n\
          Bug = \"none\"\n\
         INVARIANTS\n\
          TypeOK\n\
          StepBlocksInMemoryAndOwned\n\
          SealedValid\n\
          UnsealedPrivateInvisibleZero\n\
          LaneHistoryCovered\n\
          RefCountMatchesReferrers\n\
          StartPointsKept\n\
          BudgetsAndReservations\n\
          ColdBlocksSpillable\n\
          BypassBoundHolds\n\
          StepBudgetHolds\n\
          QueueBounded\n\
         PROPERTIES\n\
          Conforms\n\
          ActionSealedImmutable\n\
          ActionNoQueuedRefusal\n\
          ActionLazyColdWriteback\n\
          ActionDecodePriority\n",
        cfg.block_size,
        cfg.window_size,
        cfg.max_pos,
        cfg.max_span,
        cfg.step_token_budget,
        cfg.mem_budget,
        cfg.disk_budget,
        cfg.max_queue,
        cfg.max_bypass
    );

    Ok((data, check, cfg_file))
}

/// Deterministic seeded PRNG (splitmix64) for reproducible walks and stress gates.
struct SplitMix(u64);
impl SplitMix {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn pick(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

/// Summary statistics for a trace walk to prove coverage of key transitions.
#[derive(Clone, Debug, Default)]
pub struct WalkStats {
    pub steps: usize,
    pub arrivals: usize,
    pub refused_429: usize,
    pub admits: usize,
    pub admit_bypasses: usize,
    pub step_bypasses: usize,
    pub cancel_in_step: usize,
    pub finished: usize,
    pub cold_writebacks: usize,
    pub disk_restores: usize,
    pub max_wait_ticks_seen: u32,
}

/// Run one deterministic multi-lane workload walk on `Scheduler`, recording every state
/// transition into a `SchedulerTrace` for TLC verification.
pub fn walk_scheduler_trace(
    cfg: SchedulerConfig,
    seed: u64,
    max_ops: usize,
) -> Result<(SchedulerTrace, WalkStats), SchedError> {
    let mut sched = Scheduler::new(cfg.clone())?;
    let mut trace = SchedulerTrace::new(&sched);
    let mut rng = SplitMix::new(seed);
    let mut stats = WalkStats::default();
    let n_lanes = cfg.lanes.len();

    for op in 0..max_ops {
        // 1. If a step is active: occasionally cancel an InStep lane BEFORE gpu_done or FinishStep
        //    to exercise the InStep -> Cancelling -> Cancelled barrier!
        if sched.step_active() {
            if !sched.gpu_done() {
                if rng.pick(5) == 0 {
                    if let Some(&victim) = sched
                        .step_lanes()
                        .iter()
                        .find(|&&l| sched.lanes()[l].state == LaneState::InStep)
                    {
                        sched.cancel(victim)?;
                        trace.record("CancelInStep", &sched);
                        stats.cancel_in_step += 1;
                        continue;
                    }
                }
                sched.gpu_execute()?;
                trace.record("GpuExecute", &sched);
                continue;
            }
            if !sched.step_io().is_empty() {
                if sched.disk_used() + sched.step_io().len() as u32 <= cfg.disk_budget {
                    sched.complete_step_io()?;
                    trace.record("CompleteStepIO", &sched);
                    continue;
                } else if sched.reclaim_cold_disk() {
                    trace.record("ReclaimColdDisk", &sched);
                    continue;
                }
            }
            // Occasionally cancel right after gpu_done before FinishStep too.
            if rng.pick(9) == 0 {
                if let Some(&victim) = sched
                    .step_lanes()
                    .iter()
                    .find(|&&l| sched.lanes()[l].state == LaneState::InStep)
                {
                    sched.cancel(victim)?;
                    trace.record("CancelInStep", &sched);
                    stats.cancel_in_step += 1;
                    continue;
                }
            }
            sched.complete_step()?;
            trace.record("FinishStep", &sched);
            continue;
        }

        // Complete any pending cold writeback I/O when no step is active.
        if !sched.step_io().is_empty() {
            if sched.disk_used() + sched.step_io().len() as u32 <= cfg.disk_budget {
                sched.complete_step_io()?;
                trace.record("CompleteStepIO", &sched);
                continue;
            } else if sched.reclaim_cold_disk() {
                trace.record("ReclaimColdDisk", &sched);
                continue;
            }
        }

        // 2. Finish any Ready lane that has reached its target.
        if let Some(l) = (0..n_lanes)
            .find(|&i| sched.lanes()[i].state == LaneState::Ready && sched.lanes()[i].pos == sched.lanes()[i].target)
        {
            sched.finish_ready(l)?;
            trace.record("FinishReady", &sched);
            stats.finished += 1;
            continue;
        }

        // 3. Relieve memory pressure lazily (evict clean/unref, cold writeback, or reclaim cold disk).
        if sched.mem_pressure() {
            if sched.evict_clean_or_unref_mem() {
                trace.record("EvictCleanOrUnrefMem", &sched);
                continue;
            }
            if sched.start_cold_writeback().is_some() {
                trace.record("StartColdWriteback", &sched);
                stats.cold_writebacks += 1;
                continue;
            }
            if sched.reclaim_cold_disk() {
                trace.record("ReclaimColdDisk", &sched);
                continue;
            }
        }

        // 4. Try building a step first when Ready lanes exist and another waiting lane is at W
        //    so `build_step` enforces the bypass ceiling itself (returning Ok(None) cleanly in
        //    production, and immediately tripping BypassBoundHolds if the ceiling check is mutated).
        let has_non_ready_ceiling = (0..n_lanes).any(|i| {
            !sched.is_steppable_ready(i)
                && sched.lanes()[i].state.is_waiting()
                && sched.lanes()[i].wait_ticks >= cfg.max_bypass
        });
        if has_non_ready_ceiling {
            let before_waits: Vec<u32> = sched.lanes().iter().map(|l| l.wait_ticks).collect();
            if sched.build_step()?.is_some() {
                for (i, l) in sched.lanes().iter().enumerate() {
                    if l.wait_ticks > before_waits[i] {
                        stats.step_bypasses += 1;
                    }
                    stats.max_wait_ticks_seen = stats.max_wait_ticks_seen.max(l.wait_ticks);
                }
                trace.record("BuildStep", &sched);
                continue;
            }
        }

        // Advance any waiting non-Ready lane that has reached the starvation ceiling W first!
        let mut advanced_ceiling = false;
        if let Some(l) = (0..n_lanes).find(|&i| {
            sched.lanes()[i].state == LaneState::Restoring
                && sched.lanes()[i].wait_ticks >= cfg.max_bypass
        }) {
            let disk_blk = sched
                .lane_block_set(l)
                .into_iter()
                .find(|b| sched.block_tier(b) == BlockTier::Disk);
            if let Some(b) = disk_blk {
                if sched.restore_block(l, &b).is_ok() {
                    trace.record("RestoreBlock", &sched);
                    stats.disk_restores += 1;
                    advanced_ceiling = true;
                }
            } else if sched.finish_restore(l).is_ok() {
                trace.record("FinishRestore", &sched);
                advanced_ceiling = true;
            }
        }
        if advanced_ceiling {
            continue;
        }
        if let Some(l) = (0..n_lanes).find(|&i| {
            sched.lanes()[i].state == LaneState::Admitted
                && sched.lanes()[i].wait_ticks >= cfg.max_bypass
                && sched.can_advance_admitted(i)
        }) {
            sched.begin_restore(l)?;
            trace.record("BeginRestore", &sched);
            continue;
        }
        if let Some(l) = sched
            .queue_seq()
            .iter()
            .copied()
            .find(|&q| sched.lanes()[q].wait_ticks >= cfg.max_bypass && sched.can_admit_now(q))
        {
            sched.admit(l)?;
            trace.record("Admit", &sched);
            stats.admits += 1;
            continue;
        }

        // 5. Mix arrivals, admissions, restores, and steps deterministically (draining near end).
        let draining = op + 36 >= max_ops;
        let roll = rng.pick(100);

        if !draining && roll < 32 {
            let idle: Vec<usize> = (0..n_lanes)
                .filter(|&i| sched.lanes()[i].state.is_terminal_or_idle())
                .collect();
            if !idle.is_empty() {
                let l = idle[rng.pick(idle.len())];
                let pfx = cfg.prefixes[rng.pick(cfg.prefixes.len())].clone();
                let len = cfg.request_lengths[rng.pick(cfg.request_lengths.len())];
                let outcome = sched.arrive(l, &pfx, len)?;
                trace.record("Arrive", &sched);
                stats.arrivals += 1;
                if outcome == ArriveOutcome::Refused429 {
                    stats.refused_429 += 1;
                }
                continue;
            }
        }

        // Prefer building a step frequently when Ready lanes exist AND other lanes are waiting
        // so bypass increments and decode priority are thoroughly exercised!
        if roll < 68 {
            let before_waits: Vec<u32> = sched.lanes().iter().map(|l| l.wait_ticks).collect();
            if sched.build_step()?.is_some() {
                for (i, l) in sched.lanes().iter().enumerate() {
                    if l.wait_ticks > before_waits[i] {
                        stats.step_bypasses += 1;
                    }
                    stats.max_wait_ticks_seen = stats.max_wait_ticks_seen.max(l.wait_ticks);
                }
                trace.record("BuildStep", &sched);
                continue;
            }
        }

        // Advance Restoring lanes (longest-waiting first).
        let mut restoring: Vec<usize> = (0..n_lanes)
            .filter(|&i| sched.lanes()[i].state == LaneState::Restoring)
            .collect();
        restoring.sort_by_key(|&i| (std::cmp::Reverse(sched.lanes()[i].wait_ticks), i));
        let mut did_restore = false;
        for l in restoring {
            let disk_blk = sched
                .lane_block_set(l)
                .into_iter()
                .find(|b| sched.block_tier(b) == BlockTier::Disk);
            if let Some(b) = disk_blk {
                if sched.restore_block(l, &b).is_ok() {
                    trace.record("RestoreBlock", &sched);
                    stats.disk_restores += 1;
                    did_restore = true;
                    break;
                }
            } else if sched.finish_restore(l).is_ok() {
                trace.record("FinishRestore", &sched);
                did_restore = true;
                break;
            }
        }
        if did_restore {
            continue;
        }

        // Advance Admitted -> Restoring.
        if let Some(l) = (0..n_lanes).find(|&i| sched.can_advance_admitted(i)) {
            sched.begin_restore(l)?;
            trace.record("BeginRestore", &sched);
            continue;
        }

        // Admit next eligible Queued lane.
        let before_waits: Vec<u32> = sched.lanes().iter().map(|l| l.wait_ticks).collect();
        if sched.admit_next()?.is_some() {
            for (i, l) in sched.lanes().iter().enumerate() {
                if l.state == LaneState::Queued && l.wait_ticks > before_waits[i] {
                    stats.admit_bypasses += 1;
                }
                stats.max_wait_ticks_seen = stats.max_wait_ticks_seen.max(l.wait_ticks);
            }
            trace.record("Admit", &sched);
            stats.admits += 1;
            continue;
        }

        // Fallback: build a step if possible.
        let before_waits: Vec<u32> = sched.lanes().iter().map(|l| l.wait_ticks).collect();
        if sched.build_step()?.is_some() {
            for (i, l) in sched.lanes().iter().enumerate() {
                if l.wait_ticks > before_waits[i] {
                    stats.step_bypasses += 1;
                }
                stats.max_wait_ticks_seen = stats.max_wait_ticks_seen.max(l.wait_ticks);
            }
            trace.record("BuildStep", &sched);
            continue;
        }

        // Otherwise arrive into an idle lane if not draining, or cancel a stuck lane.
        if !draining {
            if let Some(l) = (0..n_lanes).find(|&i| sched.lanes()[i].state.is_terminal_or_idle()) {
                let pfx = cfg.prefixes[rng.pick(cfg.prefixes.len())].clone();
                let len = cfg.request_lengths[rng.pick(cfg.request_lengths.len())];
                let outcome = sched.arrive(l, &pfx, len)?;
                trace.record("Arrive", &sched);
                stats.arrivals += 1;
                if outcome == ArriveOutcome::Refused429 {
                    stats.refused_429 += 1;
                }
                continue;
            }
        }
        break;
    }

    stats.steps = trace.steps.len();
    Ok((trace, stats))
}
