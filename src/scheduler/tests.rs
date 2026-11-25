use super::*;

fn small_cfg() -> SchedulerConfig {
    SchedulerConfig {
        lanes: vec!["a".into(), "b".into(), "c".into(), "d".into()],
        prefixes: vec!["p1".into(), "p2".into()],
        request_lengths: vec![2, 4],
        block_size: 2,
        window_size: 2,
        max_pos: 4,
        max_span: 2,
        step_token_budget: 2,
        mem_budget: 6,
        disk_budget: 4,
        max_queue: 2,
        max_bypass: 2,
        start_point_stride_blocks: 1,
    }
}

#[test]
fn explicit_lane_lifecycle_and_illegal_transitions() {
    let mut s = Scheduler::new(small_cfg()).unwrap();
    assert_eq!(s.lanes()[0].state, LaneState::Idle);
    // Cannot admit before Queued.
    assert!(s.admit(0).is_err());
    assert_eq!(s.arrive(0, "p1", 4).unwrap(), ArriveOutcome::Queued);
    assert_eq!(s.lanes()[0].state, LaneState::Queued);
    // Cannot begin_restore before Admitted.
    assert!(s.begin_restore(0).is_err());
    s.admit(0).unwrap();
    assert_eq!(s.lanes()[0].state, LaneState::Admitted);
    // Cannot finish_restore before Restoring.
    assert!(s.finish_restore(0).is_err());
    s.begin_restore(0).unwrap();
    assert_eq!(s.lanes()[0].state, LaneState::Restoring);
    s.finish_restore(0).unwrap();
    assert_eq!(s.lanes()[0].state, LaneState::Ready);

    // Run through prefill (pos 0->2) and decode (pos 2->3->4) to Finished.
    let mut slid = 0usize;
    while s.lanes()[0].pos < s.lanes()[0].target {
        if s.slide_window(0) {
            slid += 1;
        }
        let step = s.build_step().unwrap().expect("step built");
        assert_eq!(step.lanes, vec![0]);
        assert_eq!(s.lanes()[0].state, LaneState::InStep);
        // complete_step must fail before gpu_execute!
        assert!(s.complete_step().is_err());
        s.gpu_execute().unwrap();
        let done = s.complete_step().unwrap();
        assert_eq!(done.survived, vec![0]);
        assert_eq!(s.lanes()[0].state, LaneState::Ready);
        s.check_invariants().unwrap();
    }
    assert!(slid >= 1, "explicit slide_window exercised during decode");
    s.finish_ready(0).unwrap();
    assert_eq!(s.lanes()[0].state, LaneState::Finished);
    assert_eq!(s.reserved_total(), 0);
    assert_eq!(s.admitted_total(), 0);
    s.check_invariants().unwrap();
}

#[test]
fn cancel_in_step_waits_for_gpu_and_io_barrier() {
    let mut s = Scheduler::new(small_cfg()).unwrap();
    s.arrive(0, "p1", 4).unwrap();
    s.admit(0).unwrap();
    s.begin_restore(0).unwrap();
    s.finish_restore(0).unwrap();

    let step = s.build_step().unwrap().unwrap();
    assert_eq!(step.lanes, vec![0]);
    assert_eq!(s.lanes()[0].state, LaneState::InStep);
    let res_before = s.lanes()[0].reservation;
    let mem_before = s.mem_used();
    assert!(res_before > 0 && mem_before > 0);

    // Client disconnects mid-step: must enter Cancelling and HOLD slot, blocks, pins, reservation!
    assert_eq!(s.cancel(0).unwrap(), LaneState::Cancelling);
    assert_eq!(s.lanes()[0].state, LaneState::Cancelling);
    assert_eq!(s.lanes()[0].reservation, res_before);
    assert_eq!(s.mem_used(), mem_before);
    assert!(!s.step_pins().is_empty());
    // Cannot reuse the slot while Cancelling!
    assert!(s.arrive(0, "p1", 2).is_err());
    // Cannot finish step until GPU completes!
    assert!(s.complete_step().is_err());
    s.check_invariants().unwrap();

    // GPU work finishes -> complete_step transitions Cancelling -> Cancelled and frees everything.
    s.gpu_execute().unwrap();
    let done = s.complete_step().unwrap();
    assert_eq!(done.cancelled, vec![0]);
    assert_eq!(s.lanes()[0].state, LaneState::Cancelled);
    assert_eq!(s.lanes()[0].reservation, 0);
    assert_eq!(s.lanes()[0].admit_budget, 0);
    assert_eq!(s.mem_used(), 0);
    assert!(s.step_pins().is_empty());
    s.check_invariants().unwrap();

    // Slot is now cleanly reusable.
    assert_eq!(s.arrive(0, "p2", 2).unwrap(), ArriveOutcome::Queued);
}

#[test]
fn cancel_in_restore_io_waits_for_disk_read_barrier() {
    let mut cfg = small_cfg();
    cfg.mem_budget = 4;
    cfg.disk_budget = 4;
    let mut s = Scheduler::new(cfg).unwrap();

    // Seed a kept prefill start point at (p1, 2) (`target = 4` -> `PromptEnd = 2`) and spill its blocks to disk.
    s.arrive(0, "p1", 4).unwrap();
    s.admit(0).unwrap();
    s.begin_restore(0).unwrap();
    s.finish_restore(0).unwrap();
    s.build_step().unwrap();
    s.gpu_execute().unwrap();
    s.complete_step().unwrap();
    s.cancel(0).unwrap();

    // Fill memory with p2 on lane 1 so p1's cold blocks spill to disk under pressure.
    s.arrive(1, "p2", 4).unwrap();
    s.admit(1).unwrap();
    s.begin_restore(1).unwrap();
    s.finish_restore(1).unwrap();
    // Trigger pressure: lane 1 needs 2 private blocks, mem_used == 2 (p1's 2 sealed blocks),
    // mem_budget == 4 -> let's push another block or spill p1 explicitly via pressure.
    s.build_step().unwrap();
    s.gpu_execute().unwrap();
    s.complete_step().unwrap();
    // Now mem_used == 4 (p1: 2 blocks, p2: 2 sealed blocks), lane 1 at pos 2 needs 2 more!
    assert!(s.mem_pressure());
    let wb1 = s.start_cold_writeback().expect("cold writeback 1");
    s.complete_step_io().unwrap();
    assert_eq!(s.block_tier(&wb1), BlockTier::Disk);

    // Cancel lane 1 to free budget, then admit lane 2 on p1 -> Restoring with disk block!
    s.cancel(1).unwrap();
    s.arrive(2, "p1", 4).unwrap();
    s.admit(2).unwrap();
    s.begin_restore(2).unwrap();
    assert_eq!(s.lanes()[2].state, LaneState::Restoring);

    // Start async disk restore read, then cancel mid-read!
    s.begin_restore_io(2, &wb1).unwrap();
    assert_eq!(s.cancel(2).unwrap(), LaneState::Cancelling);
    assert_eq!(s.lanes()[2].state, LaneState::Cancelling);
    assert!(s.lanes()[2].reservation > 0, "reservation held until disk I/O completes");
    assert!(s.arrive(2, "p1", 2).is_err(), "slot cannot be reused during restore I/O");

    // Once disk read completes, lane transitions Cancelling -> Cancelled and releases cleanly.
    assert_eq!(s.complete_restore_io(2).unwrap(), LaneState::Cancelled);
    assert_eq!(s.lanes()[2].state, LaneState::Cancelled);
    assert_eq!(s.lanes()[2].reservation, 0);
    assert_eq!(s.lanes()[2].admit_budget, 0);
    s.check_invariants().unwrap();
}

#[test]
fn starvation_bound_enforced_across_ready_and_queued_adversarially() {
    // W = 2: 1 prefiller + 3 decoders competing for StepTokenBudget = 2.
    let mut cfg = small_cfg();
    cfg.lanes = vec!["l0".into(), "l1".into(), "l2".into(), "l3".into()];
    cfg.request_lengths = vec![2, 4];
    cfg.step_token_budget = 2;
    cfg.mem_budget = 16;
    cfg.max_bypass = 2;
    let mut s = Scheduler::new(cfg).unwrap();

    // Put l0, l1, l2 into decode phase (target=2 -> prompt_end=0 -> decoding from pos 0)
    // and l3 into prefill phase (target=4 -> prompt_end=2 -> prefilling at pos 0).
    for (l, len) in [(0, 2), (1, 2), (2, 2), (3, 4)] {
        s.arrive(l, "p1", len).unwrap();
        s.admit(l).unwrap();
        s.begin_restore(l).unwrap();
        s.finish_restore(l).unwrap();
    }
    assert!(s.is_decoding(0) && s.is_decoding(1) && s.is_decoding(2));
    assert!(s.is_prefilling(3));

    // Run steps until all complete; verify l3 (prefiller) is NEVER bypassed > W=2 ticks!
    let mut max_seen = 0u32;
    for _ in 0..32 {
        for l in 0..4 {
            if s.lanes()[l].state == LaneState::Ready
                && s.lanes()[l].pos == s.lanes()[l].target
            {
                s.finish_ready(l).unwrap();
            }
        }
        if s.lanes().iter().all(|l| l.state == LaneState::Finished) {
            break;
        }
        for l in 0..4 {
            s.slide_window(l);
        }
        let _step = s.build_step().unwrap().expect("step built");
        for l in s.lanes() {
            max_seen = max_seen.max(l.wait_ticks);
            assert!(
                l.wait_ticks <= 2,
                "starvation bound violated: {} > 2",
                l.wait_ticks
            );
        }
        s.check_invariants().unwrap();
        s.gpu_execute().unwrap();
        s.complete_step().unwrap();
    }
    assert!(s.lanes().iter().all(|l| l.state == LaneState::Finished));
    assert!(max_seen >= 1 && max_seen <= 2);
}

/// Layer 4 Gate (`docs/ROADMAP.md:139`):
/// Bimodal workload mixing short (`2K` = 2048 tokens) and long (`64K` = 65536 tokens)
/// requests under a tight memory budget where `64K` requests force queueing + starvation-
/// bound promotion while clients disconnect mid-step (`InStep -> Cancelling -> Cancelled`),
/// proving zero memory/slot leak and bounded wait ticks (`<= W`).
#[test]
fn bimodal_2k_64k_tight_budget_with_mid_step_disconnects() {
    const BLK: u32 = 64;
    const WIN: u32 = 1024;
    const SHORT_2K: u32 = 2048;
    const LONG_64K: u32 = 65536;
    // Worst-case blocks:
    //   2K:  full=32, win=min(SpanBound(64)=18, 32)=18 -> 50 blocks
    //   64K: full=1024, win=18 -> 1042 blocks
    let wc_2k = worst_case_blocks(SHORT_2K, BLK, WIN, BLK);
    let wc_64k = worst_case_blocks(LONG_64K, BLK, WIN, BLK);
    assert_eq!(wc_2k, 50);
    assert_eq!(wc_64k, 1042);

    // Tight budget: 1150 blocks -> fits ONE 64K (1042) + TWO 2K (100), OR many 2K,
    // so when 3+ 2K lanes are running (150 blocks), a queued 64K lane CANNOT fit until
    // starvation-bound W blocks further 2K leapfrogging and lets 2K lanes drain!
    let mem_budget = wc_64k + 2 * wc_2k + 8; // 1150 blocks
    let w_bound = 3u32;
    let cfg = SchedulerConfig {
        lanes: (0..8).map(|i| format!("lane{i}")).collect(),
        prefixes: vec!["short".into(), "long".into()],
        request_lengths: vec![SHORT_2K, LONG_64K],
        block_size: BLK,
        window_size: WIN,
        max_pos: LONG_64K,
        max_span: BLK,
        step_token_budget: 256,
        mem_budget,
        disk_budget: 128,
        max_queue: 4,
        max_bypass: w_bound,
        start_point_stride_blocks: 16,
    };
    let mut s = Scheduler::new(cfg).unwrap();

    // Phase 1: fill 4 lanes with 2K requests (200 blocks admitted -> remaining 950 < 1042).
    for l in 0..4 {
        assert_eq!(s.arrive(l, "short", SHORT_2K).unwrap(), ArriveOutcome::Queued);
        s.admit(l).unwrap();
        s.begin_restore(l).unwrap();
        s.finish_restore(l).unwrap();
    }
    assert!(!s.can_fit_request(LONG_64K), "tight budget must force 64K to queue");

    // Phase 2: enqueue a 64K request on lane 4, plus a flood of 2K requests behind it!
    assert_eq!(s.arrive(4, "long", LONG_64K).unwrap(), ArriveOutcome::Queued);
    assert_eq!(s.arrive(5, "short", SHORT_2K).unwrap(), ArriveOutcome::Queued);
    assert_eq!(s.arrive(6, "short", SHORT_2K).unwrap(), ArriveOutcome::Queued);
    assert_eq!(s.arrive(7, "short", SHORT_2K).unwrap(), ArriveOutcome::Queued);

    // Younger 2K requests leapfrog lane 4 (64K) ONLY up to W=3 times, then lane 4 hits W
    // and blocks any further younger 2K leapfrogging!
    assert!(s.can_admit_now(5));
    s.admit(5).unwrap();
    assert_eq!(s.lanes()[4].wait_ticks, 1);
    assert!(s.can_admit_now(6));
    s.admit(6).unwrap();
    assert_eq!(s.lanes()[4].wait_ticks, 2);
    assert!(s.can_admit_now(7));
    s.admit(7).unwrap();
    assert_eq!(s.lanes()[4].wait_ticks, 3);
    assert_eq!(s.lanes()[4].wait_ticks, w_bound);

    // Advance admitted 5,6,7 to Ready.
    for l in 5..=7 {
        s.begin_restore(l).unwrap();
        s.finish_restore(l).unwrap();
    }

    // Now run steps with mid-step client disconnects (lanes 1, 5, 6 disconnect mid-step!)
    // and new 2K arrivals attempting to leapfrog lane 4 while it sits at wait_ticks == W.
    let mut mid_step_cancels = 0usize;
    let mut disconnected_once = BTreeSet::new();
    let mut completed_2k = 0usize;
    let mut completed_64k = 0usize;
    let mut max_wait_observed = 0u32;
    let mut extra_short_submitted = 0usize;
    let mut long_admitted = false;

    for tick in 0..4000 {
        // Track wait_ticks bound across EVERY waiting state.
        for l in s.lanes() {
            if l.state.is_waiting() {
                max_wait_observed = max_wait_observed.max(l.wait_ticks);
                assert!(
                    l.wait_ticks <= w_bound,
                    "tick {tick}: lane {} ({:?}) exceeded W={w_bound}: {}",
                    l.name,
                    l.state,
                    l.wait_ticks
                );
            }
        }

        // Finish any completed Ready lanes.
        for l in 0..8 {
            if s.lanes()[l].state == LaneState::Ready && s.lanes()[l].pos == s.lanes()[l].target
            {
                let was_long = s.lanes()[l].target == LONG_64K;
                s.finish_ready(l).unwrap();
                if was_long {
                    completed_64k += 1;
                } else {
                    completed_2k += 1;
                }
            }
        }

        // Submit a few extra 2K requests while lane 4 is queued at W to prove zero leapfrogging!
        if !long_admitted && extra_short_submitted < 2 {
            if let Some(idle_l) = (0..8).find(|&i| s.lanes()[i].state.is_terminal_or_idle()) {
                if s.arrive(idle_l, "short", SHORT_2K).unwrap() == ArriveOutcome::Queued {
                    extra_short_submitted += 1;
                    // Lane 4 is at W=3 -> younger queued 2K MUST NOT be allowed to admit!
                    if s.lanes()[4].state == LaneState::Queued
                        && s.lanes()[4].wait_ticks >= w_bound
                    {
                        assert!(
                            !s.can_admit_now(idle_l),
                            "younger 2K lane {idle_l} illegally leapfrogged 64K lane 4 at W"
                        );
                    }
                }
            }
        }

        // Evict cold memory blocks and start-point states lazily if under pressure.
        while s.mem_pressure() {
            if s.evict_clean_or_unref_mem() {
                continue;
            }
            if s.start_cold_writeback().is_some() {
                s.complete_step_io().unwrap();
                continue;
            }
            if s.reclaim_cold_disk() {
                continue;
            }
            break;
        }
        while s.state_mem_pressure() {
            if s.evict_clean_start_state().is_some() {
                continue;
            }
            if s.spill_start_state().is_some() {
                continue;
            }
            if s.reclaim_cold_disk() {
                continue;
            }
            break;
        }

        // Admit eligible queued lanes (lane 4 admits as soon as enough 2K lanes drain!).
        while let Some(admitted) = s.admit_next().unwrap() {
            if admitted == 4 {
                long_admitted = true;
            }
        }
        for l in 0..8 {
            if s.can_advance_admitted(l) {
                s.begin_restore(l).unwrap();
            }
            if s.lanes()[l].state == LaneState::Restoring {
                for b in s.lane_block_set(l) {
                    if s.blocks().get(&b).is_some_and(|r| r.tier == BlockTier::Disk) {
                        while s.mem_used() >= s.config().mem_budget {
                            if !s.evict_clean_or_unref_mem() {
                                if s.start_cold_writeback().is_some() {
                                    s.complete_step_io().unwrap();
                                } else if !s.reclaim_cold_disk() {
                                    break;
                                }
                            }
                        }
                        s.restore_block(l, &b).unwrap();
                    }
                }
                if !s.start_state_ready(l) {
                    while s.state_mem_used() >= s.state_mem_budget() {
                        if s.evict_clean_start_state().is_none()
                            && s.spill_start_state().is_none()
                            && !s.reclaim_cold_disk()
                        {
                            break;
                        }
                    }
                    s.restore_start_state(l).unwrap();
                }
                s.finish_restore(l).unwrap();
            }
        }

        // All drained?
        if s.lanes().iter().all(|l| l.state.is_terminal_or_idle()) {
            break;
        }
        for l in 0..8 {
            s.slide_window(l);
        }

        // Build & execute one step, injecting mid-step disconnects on lanes 1, 5, 6 and
        // on a second 64K request if desired.
        let Some(step) = s.build_step().unwrap() else {
            panic!("deadlock at tick {tick}: no lane could make progress");
        };

        // Inject mid-step disconnects once per victim while InStep (before gpu_execute!).
        for &victim in &[1usize, 5, 6] {
            if !disconnected_once.contains(&victim)
                && step.lanes.contains(&victim)
                && s.lanes()[victim].state == LaneState::InStep
                && s.lanes()[victim].pos >= 128
            {
                disconnected_once.insert(victim);
                assert_eq!(s.cancel(victim).unwrap(), LaneState::Cancelling);
                // Verify barrier: slot & reservation remain held while Cancelling!
                assert!(s.lanes()[victim].reservation > 0);
                mid_step_cancels += 1;
            }
        }

        s.gpu_execute().unwrap();
        let done = s.complete_step().unwrap();
        for c in &done.cancelled {
            assert_eq!(s.lanes()[*c].state, LaneState::Cancelled);
            assert_eq!(s.lanes()[*c].reservation, 0);
            assert_eq!(s.lanes()[*c].admit_budget, 0);
        }
        if tick % 25 == 0 {
            s.check_invariants().unwrap();
        }
    }

    s.check_invariants().unwrap();
    assert!(long_admitted, "64K request must be promoted and admitted");
    assert_eq!(completed_64k, 1, "64K request must complete cleanly");
    assert!(completed_2k >= 4, "surviving 2K requests must complete");
    assert_eq!(mid_step_cancels, 3, "all 3 mid-step disconnects exercised");
    assert!(max_wait_observed <= w_bound);
    assert_eq!(s.reserved_total(), 0, "zero reservation leak after drain");
    assert_eq!(s.admitted_total(), 0, "zero admission budget leak after drain");
    assert!(
        s.lanes().iter().all(|l| l.state.is_terminal_or_idle()),
        "all lanes drained cleanly"
    );
    // Every remaining in-memory block must be a sealed cold prefix block (zero unsealed tails!).
    assert!(
        s.blocks().values().all(|b| b.sealed && b.tier.is_present()),
        "zero leaked private tail blocks"
    );
}

#[test]
fn serve_step_planner_enforces_starvation_bound_and_cancel_barrier() {
    let mut p = ServeStepPlanner::new(3, 2);
    for l in 0..3 {
        p.on_admit(l);
    }
    // Lane 0 is decoding; lane 1 is prefilling (256 tokens); lane 2 is decoding.
    let inputs = [
        ServeLaneInput { runnable: true, n_past: 64, pf_end: 64, has_pending_decode: true },
        ServeLaneInput { runnable: true, n_past: 0, pf_end: 256, has_pending_decode: false },
        ServeLaneInput { runnable: true, n_past: 64, pf_end: 64, has_pending_decode: true },
    ];

    // Step 1: tie at wait_ticks=0 -> decoders win; prefiller wait_ticks becomes 1.
    let plan1 = p.plan_and_begin_step(&inputs, 128, 64).unwrap();
    assert_eq!(plan1, ServeStepPlan::Decode(vec![0, 2]));
    assert_eq!(p.state(0), LaneState::InStep);
    assert_eq!(p.wait_ticks(1), 1);

    // Client 2 disconnects mid-step -> Cancelling, NOT immediately freed!
    assert!(!p.request_cancel(2));
    assert_eq!(p.state(2), LaneState::Cancelling);
    let freed = p.complete_step();
    assert_eq!(freed, vec![2]);
    assert_eq!(p.state(2), LaneState::Cancelled);
    assert_eq!(p.state(0), LaneState::Ready);

    // Step 2: lane 1 wait_ticks=1 < W=2 -> decoder bursts a 2nd step; prefiller hits W=2.
    let inputs2 = [
        ServeLaneInput { runnable: true, n_past: 65, pf_end: 64, has_pending_decode: true },
        ServeLaneInput { runnable: true, n_past: 0, pf_end: 256, has_pending_decode: false },
        ServeLaneInput { runnable: false, n_past: 0, pf_end: 0, has_pending_decode: false },
    ];
    let plan2 = p.plan_and_begin_step(&inputs2, 128, 64).unwrap();
    assert_eq!(plan2, ServeStepPlan::Decode(vec![0]));
    assert_eq!(p.wait_ticks(1), 2);
    assert_eq!(p.wait_ticks(0), 0);
    assert!(p.complete_step().is_empty());

    // Step 3: lane 1 hit ceiling W=2 while lane 0 is at 0 -> prefiller preempts!
    let inputs3 = [
        ServeLaneInput { runnable: true, n_past: 66, pf_end: 64, has_pending_decode: true },
        ServeLaneInput { runnable: true, n_past: 0, pf_end: 256, has_pending_decode: false },
        ServeLaneInput { runnable: false, n_past: 0, pf_end: 0, has_pending_decode: false },
    ];
    let plan3 = p.plan_and_begin_step(&inputs3, 128, 64).unwrap();
    assert_eq!(plan3, ServeStepPlan::Prefill(vec![(1, 0, 128)]));
    assert_eq!(p.wait_ticks(1), 0);
    assert_eq!(p.wait_ticks(0), 1);
    assert!(p.complete_step().is_empty());
}

#[test]
fn serve_step_planner_bursts_w_decodes_before_yielding_to_prefill() {
    let w = 16u32;
    let mut p = ServeStepPlanner::new(3, w);
    for l in 0..3 {
        p.on_admit(l);
    }
    let mut n_past = [64usize, 64, 0];
    let pf_end = [64usize, 64, 8192];

    // Round 1: decoders burst W=16 consecutive 1-token steps while prefiller waits 1..=16.
    for step in 1..=w {
        let inputs = [
            ServeLaneInput { runnable: true, n_past: n_past[0], pf_end: pf_end[0], has_pending_decode: true },
            ServeLaneInput { runnable: true, n_past: n_past[1], pf_end: pf_end[1], has_pending_decode: true },
            ServeLaneInput { runnable: true, n_past: n_past[2], pf_end: pf_end[2], has_pending_decode: false },
        ];
        let plan = p.plan_and_begin_step(&inputs, 4096, 64).unwrap();
        assert_eq!(plan, ServeStepPlan::Decode(vec![0, 1]), "step {step} must burst decode");
        assert_eq!(p.wait_ticks(2), step);
        assert_eq!(p.wait_ticks(0), 0);
        assert_eq!(p.wait_ticks(1), 0);
        n_past[0] += 1;
        n_past[1] += 1;
        assert!(p.complete_step().is_empty());
    }

    // Step W+1: prefiller reached W=16 -> prefill preempts for 1 step.
    let inputs = [
        ServeLaneInput { runnable: true, n_past: n_past[0], pf_end: pf_end[0], has_pending_decode: true },
        ServeLaneInput { runnable: true, n_past: n_past[1], pf_end: pf_end[1], has_pending_decode: true },
        ServeLaneInput { runnable: true, n_past: n_past[2], pf_end: pf_end[2], has_pending_decode: false },
    ];
    let plan = p.plan_and_begin_step(&inputs, 4096, 64).unwrap();
    assert_eq!(plan, ServeStepPlan::Prefill(vec![(2, 0, 4096)]));
    assert_eq!(p.wait_ticks(2), 0);
    assert_eq!(p.wait_ticks(0), 1);
    assert_eq!(p.wait_ticks(1), 1);
    n_past[2] += 4096;
    assert!(p.complete_step().is_empty());

    // Round 2: decoders immediately burst another W=16 steps before the remaining prefill runs.
    for step in 1..=w {
        let inputs = [
            ServeLaneInput { runnable: true, n_past: n_past[0], pf_end: pf_end[0], has_pending_decode: true },
            ServeLaneInput { runnable: true, n_past: n_past[1], pf_end: pf_end[1], has_pending_decode: true },
            ServeLaneInput { runnable: true, n_past: n_past[2], pf_end: pf_end[2], has_pending_decode: false },
        ];
        let plan = p.plan_and_begin_step(&inputs, 4096, 64).unwrap();
        assert_eq!(plan, ServeStepPlan::Decode(vec![0, 1]), "round 2 step {step} must burst decode");
        assert_eq!(p.wait_ticks(2), step);
        n_past[0] += 1;
        n_past[1] += 1;
        assert!(p.complete_step().is_empty());
    }
    let inputs = [
        ServeLaneInput { runnable: true, n_past: n_past[0], pf_end: pf_end[0], has_pending_decode: true },
        ServeLaneInput { runnable: true, n_past: n_past[1], pf_end: pf_end[1], has_pending_decode: true },
        ServeLaneInput { runnable: true, n_past: n_past[2], pf_end: pf_end[2], has_pending_decode: false },
    ];
    let plan = p.plan_and_begin_step(&inputs, 4096, 64).unwrap();
    assert_eq!(plan, ServeStepPlan::Prefill(vec![(2, 4096, 4096)]));
    assert!(p.complete_step().is_empty());
}

#[test]
fn serve_step_planner_adversarial_multi_prefill_wait_ticks_bounded() {
    let w = 4u32;
    let nlanes = 12usize;
    let mut p = ServeStepPlanner::new(nlanes, w);
    for l in 0..nlanes {
        p.on_admit(l);
    }
    // Lanes 0..2 are active decoders; lanes 3..11 are 9 heavy prefillers (4096 tokens each)
    // that will all hit wait_ticks == W simultaneously while step_token_budget == 4096.
    let mut n_past = vec![64usize; nlanes];
    let mut pf_end = vec![64usize; nlanes];
    for l in 3..nlanes {
        n_past[l] = 0;
        pf_end[l] = 4096;
    }
    let mut decode_steps = 0usize;
    let mut prefill_steps = 0usize;
    for step in 0..80 {
        let inputs: Vec<ServeLaneInput> = (0..nlanes)
            .map(|l| {
                // Toggle lane 5 backpressure off/on on steps 7..10 to test staggered ceiling ties.
                let bp = l == 5 && (7..=10).contains(&step);
                ServeLaneInput {
                    runnable: !bp && n_past[l] < pf_end[l] + 300,
                    n_past: n_past[l],
                    pf_end: pf_end[l],
                    has_pending_decode: n_past[l] >= pf_end[l],
                }
            })
            .collect();
        let Some(plan) = p.plan_and_begin_step(&inputs, 4096, 64) else {
            break;
        };
        for l in 0..nlanes {
            assert!(p.wait_ticks(l) <= w, "step {step} lane {l} exceeded W={w}: {}", p.wait_ticks(l));
        }
        match plan {
            ServeStepPlan::Decode(lanes) => {
                decode_steps += 1;
                for l in lanes {
                    n_past[l] += 1;
                }
            }
            ServeStepPlan::Prefill(spans) => {
                prefill_steps += 1;
                let total_toks: usize = spans.iter().map(|&(_, _, t)| t).sum();
                assert!(total_toks <= 4096);
                for (l, p0, t) in spans {
                    assert_eq!(p0, n_past[l]);
                    n_past[l] += t;
                }
            }
        }
        assert!(p.complete_step().is_empty());
    }
    assert!(decode_steps > prefill_steps * 2, "decoders must burst between prefill rounds ({decode_steps} dec vs {prefill_steps} pf)");
    for l in 3..nlanes {
        assert!(n_past[l] >= pf_end[l], "prefiller {l} starved: n_past={} < {}", n_past[l], pf_end[l]);
    }
}

#[test]
fn walk_scheduler_trace_satisfies_invariants_across_seeds() {
    let cfg = small_cfg();
    let mut total_cancel = 0;
    let mut total_step_bypass = 0;
    let mut total_slide = 0;
    for seed in 0..8 {
        let (trace, stats) = walk_scheduler_trace(cfg.clone(), seed, 120).unwrap();
        assert!(trace.steps.len() >= 10);
        assert!(stats.max_wait_ticks_seen <= cfg.max_bypass);
        total_cancel += stats.cancel_in_step;
        total_step_bypass += stats.step_bypasses;
        total_slide += stats.slide_windows;
    }
    assert!(total_cancel > 0, "walks must exercise cancel-in-step");
    assert!(total_step_bypass > 0, "walks must exercise step bypass");
    assert!(total_slide > 0, "walks must exercise explicit slide_window");
    let (tr, _) = walk_scheduler_trace(cfg, 0, 60).unwrap();
    let (data, check, tlc_cfg) = render_cache_scheduler_tla_bundle(&[tr]).unwrap();
    assert!(data.contains("MODULE TraceData"));
    assert!(check.contains("InitConforms == (stepIdx = 1) => Init"));
    assert!(check.contains("StepConforms == [][Next]_vars"));
    assert!(check.contains("ActionConforms == [][StepActionCheck]_<<vars, traceIdx, stepIdx>>"));
    assert!(tlc_cfg.contains("BypassBoundHolds"));
    assert!(tlc_cfg.contains("NoDecodeBlocksSealed"));
    assert!(tlc_cfg.contains("ActionConforms"));
}

#[test]
fn multi_block_prefill_span_decode_unsealed_and_start_state_tier() {
    let cfg = SchedulerConfig {
        lanes: vec!["a".into(), "b".into(), "c".into()],
        prefixes: vec!["p1".into(), "p2".into(), "p3".into()],
        request_lengths: vec![6],
        block_size: 2,
        window_size: 2,
        max_pos: 6,
        max_span: 4, // crosses 2 blocks (4 tokens) in a single prefill step (`U56`)!
        step_token_budget: 4,
        mem_budget: 8,
        disk_budget: 6,
        max_queue: 1,
        max_bypass: 2,
        start_point_stride_blocks: 1,
    };
    let mut s = Scheduler::new(cfg).unwrap();
    s.arrive(0, "p1", 6).unwrap();
    s.admit(0).unwrap();
    s.begin_restore(0).unwrap();
    s.finish_restore(0).unwrap();

    // Single prefill step advances 0 -> 4 (`PromptEnd(6) = 4`), sealing BOTH (p1, 2) and (p1, 4)!
    let step = s.build_step().unwrap().unwrap();
    assert_eq!(step.spans[&0], 4);
    // Guard `build_step` on in-flight writeback (`U61`) once step completes.
    s.gpu_execute().unwrap();
    let done = s.complete_step().unwrap();
    assert_eq!(done.sealed_boundaries, vec![(0, 2), (0, 4)]);
    assert!(s.start_points()[&("p1".into(), 2)].kept);
    assert!(s.start_points()[&("p1".into(), 4)].kept);
    assert_eq!(s.start_points()[&("p1".into(), 4)].state_tier, BlockTier::Memory);

    // Slide window at pos 4 (`History(4) = {1}`, pruning block 0) before decode steps 4->5->6.
    assert!(s.slide_window(0));
    for expected_pos in [5u32, 6] {
        s.slide_window(0);
        let dstep = s.build_step().unwrap().unwrap();
        assert_eq!(dstep.spans[&0], 1);
        s.gpu_execute().unwrap();
        let ddone = s.complete_step().unwrap();
        assert!(ddone.sealed_boundaries.is_empty(), "decode blocks must NEVER seal (U58)");
        assert_eq!(s.lanes()[0].pos, expected_pos);
        assert!(!s.start_points().contains_key(&("p1".into(), 6)));
        assert!(matches!(s.lanes()[0].full.get(&2), Some(BlockId::Priv { lane: 0, idx: 2, .. })));
        s.check_invariants().unwrap();
    }
    s.finish_ready(0).unwrap();
    s.check_invariants().unwrap();
}

#[test]
fn serve_step_planner_skips_restoring_lanes_without_blocking_ready_decoders() {
    let mut p = ServeStepPlanner::new(3, 4);
    p.on_admit(0);
    p.admit_queued(1, LaneState::Queued);
    p.start_restore(1);
    p.admit_queued(2, LaneState::Queued);
    p.start_restore(2);
    assert_eq!(p.state(1), LaneState::Restoring);
    assert_eq!(p.state(2), LaneState::Restoring);

    let inputs = [
        ServeLaneInput { runnable: true, n_past: 64, pf_end: 64, has_pending_decode: true },
        ServeLaneInput { runnable: true, n_past: 0, pf_end: 1024, has_pending_decode: false },
        ServeLaneInput { runnable: true, n_past: 0, pf_end: 1024, has_pending_decode: false },
    ];
    for _ in 0..6 {
        let plan = p.plan_and_begin_step(&inputs, 4096, 64).unwrap();
        assert_eq!(plan, ServeStepPlan::Decode(vec![0]));
        assert_eq!(p.wait_ticks(1), 0, "Restoring lane must not accrue wait_ticks");
        assert!(p.complete_step().is_empty());
    }

    // Mid-restore cancel immediately transitions Restoring -> Cancelled.
    assert!(p.request_cancel(2));
    assert_eq!(p.state(2), LaneState::Cancelled);

    // Finishing restore transitions Restoring -> Ready so lane 1 joins step planning.
    p.finish_restore(1);
    assert_eq!(p.state(1), LaneState::Ready);
}
