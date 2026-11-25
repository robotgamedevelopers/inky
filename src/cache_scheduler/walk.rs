//! Deterministic walks for the block layout and cache scheduler trace gates.

use super::*;

// ---- Deterministic Seeded Trace Walks (for Unit Tests & CI Trace Gate) ----

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    fn pick(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 32) as usize) % n
    }
}

/// Execute a deterministic seeded `BlockLayout` walk exercising:
///   - even seeds: start-point keep/match/drop, simultaneous live slots `1` and `2`, and window sliding;
///   - odd seeds: finish + `SlotRecord::reissue` slot reuse (`generation == 2`), generation exhaustion
///     advancing allocation to slot `2`, partial unsealed tail writes, and window sliding.
pub fn walk_block_layout(
    cfg: BlockLayoutConfig,
    seed: u64,
    event_sink: &mut dyn TraceSink,
    state_sink: &mut dyn TraceSink,
) -> Result<(), SchedulerError> {
    let mut rng = Lcg::new(seed);
    let mut alloc = TrackedBlockAllocator::new(cfg.clone(), event_sink, state_sink)?;
    let lanes = cfg.lanes.clone();
    let l0 = &lanes[0];

    let step_lane = |alloc: &mut TrackedBlockAllocator,
                     lane: &str,
                     span: u32,
                     ev: &mut dyn TraceSink,
                     st: &mut dyn TraceSink|
     -> Result<bool, SchedulerError> {
        alloc.prepare(lane, span, ev, st)?;
        for (k, b) in alloc.missing_needed_keys(lane)? {
            if alloc.first_available_slot(k).is_none() {
                return Ok(false);
            }
            alloc.alloc_lowest_free(lane, k, b, ev, st)?;
        }
        alloc.begin_step(lane, ev, st)?;
        while alloc.lanes[lane].pos < alloc.lanes[lane].target {
            alloc.write_one(lane, ev, st)?;
        }
        alloc.end_step(lane, ev, st)?;
        Ok(true)
    };

    if lanes.len() >= 2 && seed % 2 == 1 {
        // Odd seeds: l0 writes block 0 (partial 1-token steps when rng chooses) then finishes;
        // l1 reuses slot 1 at generation 2, then advances into block 1 (taking slot 2 because
        // slot 1 is held / exhausted at MaxGeneration).
        let first_span = if rng.pick(2) == 0 { 1 } else { cfg.block_size.min(cfg.max_span) };
        step_lane(&mut alloc, l0, first_span, event_sink, state_sink)?;
        if alloc.lanes[l0].pos < cfg.block_size {
            let rem = cfg.block_size - alloc.lanes[l0].pos;
            step_lane(&mut alloc, l0, rem.min(cfg.max_span), event_sink, state_sink)?;
        }
        alloc.finish(l0, event_sink, state_sink)?;

        let l1 = &lanes[1];
        while alloc.lanes[l1].pos < cfg.max_pos {
            let _ = alloc.slide_window(l1, event_sink, state_sink)?;
            let rem = cfg.max_pos - alloc.lanes[l1].pos;
            let span = if rng.pick(2) == 0 { 1 } else { cfg.max_span.min(rem) };
            if !step_lane(&mut alloc, l1, span, event_sink, state_sink)? {
                break;
            }
            let _ = alloc.slide_window(l1, event_sink, state_sink)?;
        }
        if alloc.lanes[l1].phase == "idle" {
            alloc.finish(l1, event_sink, state_sink)?;
        }
    } else {
        // Even seeds: l0 reaches boundary, keeps start point, l1 matches, drops start, slides window, advances.
        step_lane(&mut alloc, l0, cfg.block_size.min(cfg.max_span), event_sink, state_sink)?;
        if alloc.lanes[l0].pos % cfg.block_size == 0 && alloc.lanes[l0].pos > 0 {
            alloc.keep_start(l0, event_sink, state_sink)?;
        }
        if lanes.len() >= 2 && alloc.start_kept {
            let l1 = &lanes[1];
            alloc.match_start(l1, event_sink, state_sink)?;
            alloc.drop_start(event_sink, state_sink)?;
            alloc.finish(l0, event_sink, state_sink)?;
            while alloc.lanes[l1].pos < cfg.max_pos {
                let _ = alloc.slide_window(l1, event_sink, state_sink)?;
                let rem = cfg.max_pos - alloc.lanes[l1].pos;
                let span = if rng.pick(2) == 0 { 1 } else { cfg.max_span.min(rem) };
                if !step_lane(&mut alloc, l1, span, event_sink, state_sink)? {
                    break;
                }
                let _ = alloc.slide_window(l1, event_sink, state_sink)?;
            }
            if alloc.lanes[l1].phase == "idle" {
                alloc.finish(l1, event_sink, state_sink)?;
            }
        } else {
            alloc.finish(l0, event_sink, state_sink)?;
        }
    }
    Ok(())
}

/// Execute a deterministic seeded `CacheScheduler` walk exercising:
///   - `MaxQueue >= 2` admit bypass (fitting smaller request bypassing larger queued request and incrementing its `wait_ticks`),
///   - `BuildStep` `StepBypassed` `wait_ticks` increment on unchosen actionable lanes,
///   - HTTP 429 refusals, start-point matching, cold SSD spills & restores, mid-step `Cancelling` fence, and completions.
pub fn walk_cache_scheduler(
    cfg: CacheSchedulerConfig,
    seed: u64,
    event_sink: &mut dyn TraceSink,
    state_sink: &mut dyn TraceSink,
) -> Result<(), SchedulerError> {
    let mut rng = Lcg::new(seed);
    let mut sched = CacheScheduler::new(cfg.clone(), event_sink, state_sink)?;
    let lanes = cfg.lanes.clone();
    let prefixes = cfg.prefixes.clone();
    let req_lens = cfg.request_lengths.clone();

    let min_len = *req_lens.iter().min().unwrap();
    let max_len = *req_lens.iter().max().unwrap();

    // Opening scenario when `max_queue >= 2` and `lanes.len() >= 3` and `max_len > min_len`:
    // deterministically exercise BOTH `admit` queue-bypass (`wait_ticks += 1`) AND `build_step` `StepBypassed` (`wait_ticks += 1`).
    if seed.is_multiple_of(2)
        && cfg.max_queue >= 2
        && lanes.len() >= 3
        && sched.worst_case_for(min_len) * 2 <= cfg.mem_budget
        && sched.worst_case_for(min_len) + sched.worst_case_for(max_len) > cfg.mem_budget
    {
        let p0 = &prefixes[0];
        let p1 = prefixes.get(1).unwrap_or(p0);
        // l0 admitted with min_len -> holds half the admission budget:
        sched.arrive(&lanes[0], p0, min_len, event_sink, state_sink)?;
        sched.admit(&lanes[0], event_sink, state_sink)?;
        sched.begin_restore(&lanes[0], event_sink, state_sink)?;
        sched.finish_restore(&lanes[0], event_sink, state_sink)?;
        // l1 queued with max_len (cannot fit while l0 is active) + l2 queued with min_len (CAN fit!):
        sched.arrive(&lanes[1], p1, max_len, event_sink, state_sink)?;
        sched.arrive(&lanes[2], p0, min_len, event_sink, state_sink)?;
        if lanes.len() >= 4 && cfg.max_queue == 2 {
            // Queue full + max_len cannot direct-admit -> 429 Refused!
            let _ = sched.arrive(&lanes[3], p1, max_len, event_sink, state_sink)?;
        }
        // Admit l2: bypasses queued l1 -> increments l1.wait_ticks!
        sched.admit(&lanes[2], event_sink, state_sink)?;
        sched.begin_restore(&lanes[2], event_sink, state_sink)?;
        sched.finish_restore(&lanes[2], event_sink, state_sink)?;
        // Step ONLY l0 (when step_token_budget == 1 or l2 is prefilling; with decode priority, if l2 is also decoding and fits step_token_budget >= 2, make l2 prefilling or step both then bypass queued l1!):
        // Wait: in BuildStep, FitUnchosenDecoders requires SumBy(chosen, spans) == StepTokenBudget!
        // So if l0 has max_len (prefilling, span = 2 == step_token_budget!) or if l2 is prefilling:
        if let Some(plan) = sched.plan_step() {
            sched.build_step(&plan, event_sink, state_sink)?;
            sched.gpu_execute(event_sink, state_sink)?;
            sched.finish_step(event_sink, state_sink)?;
        }
    }

    for round in 0..24 {
        if sched.step_active {
            if round % 5 == 2 {
                if let Some(in_step_lane) = sched
                    .step_lanes
                    .iter()
                    .find(|l| sched.lane_state(l) == Some("InStep"))
                    .cloned()
                {
                    if rng.pick(2) == 0 {
                        let _ = sched.cancel(&in_step_lane, event_sink, state_sink)?;
                    }
                }
            }
            if !sched.gpu_done {
                sched.gpu_execute(event_sink, state_sink)?;
            }
            if !sched.step_io.is_empty() {
                sched.complete_step_io(event_sink, state_sink)?;
            }
            sched.finish_step(event_sink, state_sink)?;
        }

        for ln in &lanes {
            if sched.lane_state(ln) == Some("Ready") && sched.lanes[ln].pos == sched.lanes[ln].target {
                sched.finish_ready(ln, event_sink, state_sink)?;
            }
        }

        let burst = if round == 0 { lanes.len() } else { 1 + rng.pick(2) };
        let mut submitted = 0;
        for ln in &lanes {
            if submitted >= burst {
                break;
            }
            if matches!(
                sched.lane_state(ln),
                Some("Idle" | "Refused" | "Cancelled" | "Finished")
            ) {
                let pfx = &prefixes[rng.pick(prefixes.len())];
                let tgt = req_lens[rng.pick(req_lens.len())];
                let _ = sched.arrive(ln, pfx, tgt, event_sink, state_sink)?;
                submitted += 1;
            }
        }

        loop {
            let next_q = lanes
                .iter()
                .filter(|l| sched.can_admit_now(l))
                .max_by_key(|l| sched.lanes[*l].wait_ticks)
                .cloned();
            let Some(ql) = next_q else { break };
            sched.admit(&ql, event_sink, state_sink)?;
        }

        for ln in &lanes {
            if sched.lane_state(ln) == Some("Admitted")
                && sched.reserved_total() + sched.worst_case_for(sched.lanes[ln].target) <= cfg.mem_budget
            {
                let _ = sched.begin_restore(ln, event_sink, state_sink)?;
            }
        }

        for _ in 0..4 {
            if !sched.step_io.is_empty() {
                sched.complete_step_io(event_sink, state_sink)?;
            }
            if !sched.mem_pressure() {
                break;
            }
            if sched.evict_clean_or_unref_mem(event_sink, state_sink).is_ok() {
                continue;
            }
            if sched.disk_used() >= cfg.disk_budget {
                let _ = sched.reclaim_cold_disk(event_sink, state_sink);
            }
            if sched.start_cold_writeback(event_sink, state_sink).is_ok() {
                sched.complete_step_io(event_sink, state_sink)?;
                let _ = sched.evict_clean_or_unref_mem(event_sink, state_sink);
            } else {
                break;
            }
        }

        for ln in &lanes {
            if sched.lane_state(ln) == Some("Restoring") {
                let disk_blks: Vec<CacheBlockId> = CacheScheduler::lane_blocks_of(&sched.lanes[ln])
                    .into_iter()
                    .filter(|b| sched.blocks[b].tier == "disk")
                    .collect();
                for b in disk_blks {
                    if sched.mem_used() < cfg.mem_budget {
                        let _ = sched.restore_block(ln, &b, event_sink, state_sink);
                    }
                }
                if CacheScheduler::lane_blocks_of(&sched.lanes[ln])
                    .iter()
                    .all(|b| sched.blocks[b].in_memory())
                {
                    sched.finish_restore(ln, event_sink, state_sink)?;
                }
            }
        }

        for ln in &lanes {
            if sched.lane_state(ln) == Some("Ready") && sched.lanes[ln].pos == sched.lanes[ln].target {
                sched.finish_ready(ln, event_sink, state_sink)?;
            }
        }

        if let Some(plan) = sched.plan_step() {
            sched.build_step(&plan, event_sink, state_sink)?;
        }
    }

    if sched.step_active {
        if !sched.gpu_done {
            sched.gpu_execute(event_sink, state_sink)?;
        }
        if !sched.step_io.is_empty() {
            sched.complete_step_io(event_sink, state_sink)?;
        }
        sched.finish_step(event_sink, state_sink)?;
    }
    Ok(())
}
