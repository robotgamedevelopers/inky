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

fn allow_precond<T>(res: Result<T, SchedulerError>) -> Result<Option<T>, SchedulerError> {
    match res {
        Ok(v) => Ok(Some(v)),
        Err(SchedulerError::Precondition(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Execute a deterministic seeded `CacheScheduler` walk exercising:
///   - `MaxQueue >= 2` admit bypass (fitting smaller request bypassing larger queued request and incrementing its `wait_ticks`),
///   - `BuildStep` `StepBypassed` `wait_ticks` increment on unchosen actionable lanes,
///   - LRU cold SSD spill order (`p2` older tick evicted before `p1` newer tick),
///   - `ReclaimColdDisk`, `DropUnreferenced`, `SlideWindow`, `RestoreBlock`, HTTP 429 refusals,
///     mid-step `Cancelling` fence, and completions, propagating every `Trace`/`InvariantViolation` error (`U194`).
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

    let seal_prefix_on = |sched: &mut CacheScheduler,
                          ln: &str,
                          pfx: &str,
                          ev: &mut dyn TraceSink,
                          st: &mut dyn TraceSink|
     -> Result<(), SchedulerError> {
        sched.arrive(ln, pfx, max_len, ev, st)?;
        sched.admit(ln, ev, st)?;
        sched.begin_restore(ln, ev, st)?;
        sched.finish_restore(ln, ev, st)?;
        while sched.lanes[ln].pos < sched.prompt_end_of(max_len) {
            let _ = sched.slide_window(ln, ev, st)?;
            let Some(plan) = sched.plan_step() else { break };
            sched.build_step(&plan, ev, st)?;
            sched.gpu_execute(ev, st)?;
            sched.finish_step(ev, st)?;
        }
        Ok(())
    };

    if seed == 2
        && lanes.len() >= 2
        && prefixes.len() >= 2
        && max_len >= cfg.block_size * 2
        && sched.worst_case_for(max_len) <= cfg.mem_budget
    {
        // Seed 2 (`U193`): seal `p1` ("p2") FIRST (older `last_used_tick`) and `p0` ("p1") SECOND
        // (newer `last_used_tick`, lexicographically smaller), cancel so BOTH prefixes sit cold in
        // memory simultaneously, then trigger `StartColdWriteback` — LRU MUST pick `"p2"`!
        let p0 = &prefixes[0];
        let p1 = &prefixes[1];
        seal_prefix_on(&mut sched, &lanes[0], p1, event_sink, state_sink)?;
        sched.cancel(&lanes[0], event_sink, state_sink)?;
        seal_prefix_on(&mut sched, &lanes[0], p0, event_sink, state_sink)?;
        sched.cancel(&lanes[0], event_sink, state_sink)?;

        sched.arrive(&lanes[0], p0, min_len, event_sink, state_sink)?;
        sched.admit(&lanes[0], event_sink, state_sink)?;
        sched.begin_restore(&lanes[0], event_sink, state_sink)?;
        sched.finish_restore(&lanes[0], event_sink, state_sink)?;
        while sched.mem_pressure() {
            if allow_precond(sched.evict_clean_or_unref_mem(event_sink, state_sink))?.is_some() {
                continue;
            }
            if allow_precond(sched.start_cold_writeback(event_sink, state_sink))?.is_some() {
                sched.complete_step_io(event_sink, state_sink)?;
            } else {
                break;
            }
        }
        sched.cancel(&lanes[0], event_sink, state_sink)?;
    } else if seed.is_multiple_of(2)
        && cfg.max_queue >= 2
        && lanes.len() >= 3
        && sched.worst_case_for(min_len) * 2 <= cfg.mem_budget
        && sched.worst_case_for(min_len) + sched.worst_case_for(max_len) > cfg.mem_budget
    {
        let p0 = &prefixes[0];
        let p1 = prefixes.get(1).unwrap_or(p0);
        sched.arrive(&lanes[0], p0, min_len, event_sink, state_sink)?;
        sched.admit(&lanes[0], event_sink, state_sink)?;
        sched.begin_restore(&lanes[0], event_sink, state_sink)?;
        sched.finish_restore(&lanes[0], event_sink, state_sink)?;
        sched.arrive(&lanes[1], p1, max_len, event_sink, state_sink)?;
        sched.arrive(&lanes[2], p0, min_len, event_sink, state_sink)?;
        if lanes.len() >= 4 && cfg.max_queue == 2 {
            let _ = sched.arrive(&lanes[3], p1, max_len, event_sink, state_sink)?;
        }
        sched.admit(&lanes[2], event_sink, state_sink)?;
        sched.begin_restore(&lanes[2], event_sink, state_sink)?;
        sched.finish_restore(&lanes[2], event_sink, state_sink)?;
        if let Some(plan) = sched.plan_step() {
            sched.build_step(&plan, event_sink, state_sink)?;
            sched.gpu_execute(event_sink, state_sink)?;
            sched.finish_step(event_sink, state_sink)?;
        }
    } else if seed == 1
        && lanes.len() >= 3
        && prefixes.len() >= 2
        && max_len >= cfg.block_size * 2
        && sched.worst_case_for(max_len) <= cfg.mem_budget
    {
        // Seed 1: cold spill + `RestoreBlock` + `SlideWindow`:
        let p0 = &prefixes[0];
        let p1 = &prefixes[1];
        seal_prefix_on(&mut sched, &lanes[0], p0, event_sink, state_sink)?;
        sched.cancel(&lanes[0], event_sink, state_sink)?;
        seal_prefix_on(&mut sched, &lanes[0], p1, event_sink, state_sink)?;
        while sched.mem_pressure() {
            if allow_precond(sched.evict_clean_or_unref_mem(event_sink, state_sink))?.is_some() {
                continue;
            }
            if sched.disk_used() >= cfg.disk_budget {
                let _ = allow_precond(sched.reclaim_cold_disk(event_sink, state_sink))?;
            }
            if allow_precond(sched.start_cold_writeback(event_sink, state_sink))?.is_some() {
                sched.complete_step_io(event_sink, state_sink)?;
            } else {
                break;
            }
        }
        sched.cancel(&lanes[0], event_sink, state_sink)?;

        sched.arrive(&lanes[2], p0, max_len, event_sink, state_sink)?;
        sched.admit(&lanes[2], event_sink, state_sink)?;
        sched.begin_restore(&lanes[2], event_sink, state_sink)?;
        let disk_blks: Vec<CacheBlockId> = CacheScheduler::lane_blocks_of(&sched.lanes[&lanes[2]])
            .into_iter()
            .filter(|b| sched.blocks[b].tier == "disk")
            .collect();
        for b in disk_blks {
            if sched.mem_used() < cfg.mem_budget {
                let _ = allow_precond(sched.restore_block(&lanes[2], &b, event_sink, state_sink))?;
            }
        }
        if CacheScheduler::lane_blocks_of(&sched.lanes[&lanes[2]]).iter().all(|b| sched.blocks[b].in_memory())
            && sched.start_state_ready(&sched.lanes[&lanes[2]])
        {
            sched.finish_restore(&lanes[2], event_sink, state_sink)?;
        }
    } else if seed == 3
        && lanes.len() >= 2
        && prefixes.len() >= 2
        && max_len >= cfg.block_size * 2
        && sched.worst_case_for(max_len) <= cfg.mem_budget
        && sched.worst_case_for(min_len) * 2 <= cfg.mem_budget
    {
        // Seed 3 (`U159, U162`): fill `disk_budget`, run `ReclaimColdDisk` to retire the oldest
        // start point, and explicitly call `DropUnreferenced` on the orphaned disk blocks!
        let p0 = &prefixes[0];
        let p1 = &prefixes[1];
        seal_prefix_on(&mut sched, &lanes[0], p0, event_sink, state_sink)?;
        sched.cancel(&lanes[0], event_sink, state_sink)?;
        seal_prefix_on(&mut sched, &lanes[0], p1, event_sink, state_sink)?;
        while sched.mem_pressure() {
            if allow_precond(sched.start_cold_writeback(event_sink, state_sink))?.is_some() {
                sched.complete_step_io(event_sink, state_sink)?;
            } else {
                break;
            }
        }
        sched.cancel(&lanes[0], event_sink, state_sink)?;

        sched.arrive(&lanes[0], p0, min_len, event_sink, state_sink)?;
        sched.admit(&lanes[0], event_sink, state_sink)?;
        sched.begin_restore(&lanes[0], event_sink, state_sink)?;
        sched.finish_restore(&lanes[0], event_sink, state_sink)?;
        sched.arrive(&lanes[1], p1, min_len, event_sink, state_sink)?;
        sched.admit(&lanes[1], event_sink, state_sink)?;
        sched.begin_restore(&lanes[1], event_sink, state_sink)?;
        sched.finish_restore(&lanes[1], event_sink, state_sink)?;

        while sched.mem_pressure() {
            if sched.disk_used() >= cfg.disk_budget {
                let _ = allow_precond(sched.reclaim_cold_disk(event_sink, state_sink))?;
                let unref_ids: Vec<CacheBlockId> = sched
                    .blocks
                    .iter()
                    .filter(|(id, b)| {
                        b.present()
                            && b.ref_count == 0
                            && !sched.step_pins.contains(*id)
                            && !sched.step_io.contains(*id)
                    })
                    .map(|(id, _)| id.clone())
                    .collect();
                for id in unref_ids {
                    let _ = allow_precond(sched.drop_unreferenced(&id, event_sink, state_sink))?;
                }
            }
            if allow_precond(sched.evict_clean_or_unref_mem(event_sink, state_sink))?.is_some() {
                continue;
            }
            if allow_precond(sched.start_cold_writeback(event_sink, state_sink))?.is_some() {
                sched.complete_step_io(event_sink, state_sink)?;
            } else {
                break;
            }
        }
    }

    for round in 0..24 {
        if sched.step_active {
            if round % 5 == 2 {
                if let Some(in_step_lane) =
                    sched.step_lanes.iter().find(|l| sched.lane_state(l) == Some("InStep")).cloned()
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
            if matches!(sched.lane_state(ln), Some("Idle" | "Refused" | "Cancelled" | "Finished")) {
                let pfx = &prefixes[rng.pick(prefixes.len())];
                let tgt = req_lens[rng.pick(req_lens.len())];
                let _ = sched.arrive(ln, pfx, tgt, event_sink, state_sink)?;
                submitted += 1;
            }
        }

        loop {
            let next_q =
                lanes.iter().filter(|l| sched.can_admit_now(l)).max_by_key(|l| sched.lanes[*l].wait_ticks).cloned();
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
            let unref_ids: Vec<CacheBlockId> = sched
                .blocks
                .iter()
                .filter(|(id, b)| {
                    b.present() && b.ref_count == 0 && !sched.step_pins.contains(*id) && !sched.step_io.contains(*id)
                })
                .map(|(id, _)| id.clone())
                .collect();
            for id in unref_ids {
                let _ = allow_precond(sched.drop_unreferenced(&id, event_sink, state_sink))?;
            }
            if !sched.mem_pressure() {
                break;
            }
            if allow_precond(sched.evict_clean_or_unref_mem(event_sink, state_sink))?.is_some() {
                continue;
            }
            if sched.disk_used() >= cfg.disk_budget {
                let _ = allow_precond(sched.reclaim_cold_disk(event_sink, state_sink))?;
                continue;
            }
            if allow_precond(sched.start_cold_writeback(event_sink, state_sink))?.is_some() {
                sched.complete_step_io(event_sink, state_sink)?;
                let _ = allow_precond(sched.evict_clean_or_unref_mem(event_sink, state_sink))?;
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
                        let _ = allow_precond(sched.restore_block(ln, &b, event_sink, state_sink))?;
                    }
                }
                if !sched.start_state_ready(&sched.lanes[ln]) && sched.state_mem_used() < sched.state_mem_budget() {
                    let _ = allow_precond(sched.restore_start_state(ln, event_sink, state_sink))?;
                }
                if CacheScheduler::lane_blocks_of(&sched.lanes[ln]).iter().all(|b| sched.blocks[b].in_memory())
                    && sched.start_state_ready(&sched.lanes[ln])
                {
                    sched.finish_restore(ln, event_sink, state_sink)?;
                }
            }
        }

        for _ in 0..4 {
            if !sched.state_mem_pressure() {
                break;
            }
            if allow_precond(sched.evict_clean_start_state(event_sink, state_sink))?.is_some() {
                continue;
            }
            if sched.state_disk_used() >= sched.state_disk_budget() {
                let _ = allow_precond(sched.reclaim_cold_disk(event_sink, state_sink))?;
            }
            if allow_precond(sched.spill_start_state(event_sink, state_sink))?.is_none() {
                break;
            }
        }

        for ln in &lanes {
            if sched.lane_state(ln) == Some("Ready") {
                let _ = sched.slide_window(ln, event_sink, state_sink)?;
                if sched.lanes[ln].pos == sched.lanes[ln].target {
                    sched.finish_ready(ln, event_sink, state_sink)?;
                }
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
