#![allow(dead_code)]
// CPU-only trace generator for `src/scheduler.rs` against `specs/CacheScheduler.tla`.
// Compiled directly via `rustc --edition=2021` by `scripts/check-scheduler-traces.sh`.

mod scheduler;

use std::fs;
use std::path::Path;

use scheduler::{
    render_cache_scheduler_tla_bundle, walk_scheduler_trace_opts, ArriveOutcome, BlockTier,
    LaneState, Scheduler, SchedulerConfig, SchedulerTrace, TraceAction, WalkStats,
};

fn trace_cfg() -> SchedulerConfig {
    SchedulerConfig {
        lanes: vec!["a".into(), "b".into(), "c".into(), "d".into()],
        prefixes: vec!["p1".into(), "p2".into()],
        request_lengths: vec![2, 4],
        block_size: 2,
        window_size: 2,
        max_pos: 4,
        max_span: 2,
        step_token_budget: 2,
        mem_budget: 4,
        disk_budget: 3,
        max_queue: 2,
        max_bypass: 2,
        start_point_stride_blocks: 1,
    }
}

fn rec(t: &mut SchedulerTrace, act: TraceAction, s: &Scheduler, strict: bool) {
    t.record_action(act, s);
    if strict {
        s.check_invariants().expect("scheduler runtime invariant violated during trace");
    }
}

fn drain_to_idle(t: &mut SchedulerTrace, s: &mut Scheduler, strict: bool) {
    let n_lanes = s.lanes().len();
    for _ in 0..128 {
        if s.is_drained() {
            break;
        }
        if let Some(l) = (0..n_lanes).find(|&i| s.lanes()[i].restore_io_block.is_some()) {
            s.complete_restore_io(l).unwrap();
            rec(t, TraceAction::CompleteRestoreIO { lane: l }, s, strict);
            continue;
        }
        if s.step_active() {
            if !s.gpu_done() {
                s.gpu_execute().unwrap();
                rec(t, TraceAction::GpuExecute, s, strict);
                continue;
            }
            if !s.step_io().is_empty() {
                s.complete_step_io().unwrap();
                rec(t, TraceAction::CompleteStepIO, s, strict);
                continue;
            }
            s.complete_step().unwrap();
            rec(t, TraceAction::FinishStep, s, strict);
            continue;
        }
        if !s.step_io().is_empty() {
            s.complete_step_io().unwrap();
            rec(t, TraceAction::CompleteStepIO, s, strict);
            continue;
        }
        if let Some(l) = (0..n_lanes).find(|&i| {
            s.lanes()[i].state == LaneState::Ready && s.lanes()[i].pos == s.lanes()[i].target
        }) {
            s.finish_ready(l).unwrap();
            rec(t, TraceAction::FinishReady { lane: l }, s, strict);
            continue;
        }
        if let Some(l) =
            (0..n_lanes).find(|&i| s.lanes()[i].state == LaneState::Ready && !s.window_pruned(i))
        {
            if s.slide_window(l) {
                rec(t, TraceAction::SlideWindow { lane: l }, s, strict);
                continue;
            }
        }
        if s.mem_pressure() {
            if let Some(block) = s.evict_clean_or_unref_mem_id() {
                rec(t, TraceAction::EvictCleanOrUnrefMem { block }, s, strict);
                continue;
            }
            if let Some(block) = s.start_cold_writeback() {
                rec(t, TraceAction::StartColdWriteback { block }, s, strict);
                continue;
            }
            if s.reclaim_cold_disk() {
                rec(t, TraceAction::ReclaimColdDisk, s, strict);
                continue;
            }
        }
        if let Some(l) = (0..n_lanes).find(|&i| s.lanes()[i].state == LaneState::Restoring) {
            let disk_blk = s
                .lane_block_set(l)
                .into_iter()
                .find(|b| s.block_tier(b) == BlockTier::Disk);
            if let Some(b) = disk_blk {
                s.restore_block(l, &b).unwrap();
                rec(t, TraceAction::RestoreBlock { lane: l, block: b }, s, strict);
                continue;
            }
            if !s.start_state_ready(l) {
                s.restore_start_state(l).unwrap();
                rec(t, TraceAction::RestoreStartState { lane: l }, s, strict);
                continue;
            }
            s.finish_restore(l).unwrap();
            rec(t, TraceAction::FinishRestore { lane: l }, s, strict);
            continue;
        }
        if let Some(l) = (0..n_lanes).find(|&i| s.can_advance_admitted(i)) {
            s.begin_restore(l).unwrap();
            rec(t, TraceAction::BeginRestore { lane: l }, s, strict);
            continue;
        }
        if let Some(l) = s.admit_next().unwrap() {
            rec(t, TraceAction::Admit { lane: l }, s, strict);
            continue;
        }
        if let Some(built) = s.build_step().unwrap() {
            rec(t, TraceAction::BuildStep { spans: built.spans }, s, strict);
            continue;
        }
        break;
    }
    if strict {
        assert!(s.is_drained(), "targeted scheduler trace failed to drain completely");
    }
}

/// Deterministic targeted trace that explicitly exercises:
/// - Admit bypass (`Queued` lane 1 of len 4 bypassed by younger `Queued` lane 2 of len 2),
/// - Step bypass up to ceiling `W = 2` (and verifying `build_step` refuses to bypass past `W`),
/// - Cancel-in-step barrier (`InStep -> Cancelling -> Cancelled` across `gpu_execute` + `complete_step`),
/// - Full drain to terminal/idle state.
fn targeted_edge_trace(cfg: SchedulerConfig, strict: bool) -> SchedulerTrace {
    let mut s = Scheduler::new(cfg).expect("init scheduler");
    if strict {
        s.check_invariants().unwrap();
    }
    let mut t = SchedulerTrace::new(&s);

    assert_eq!(s.arrive(0, "p1", 2).unwrap(), ArriveOutcome::Queued);
    rec(&mut t, TraceAction::Arrive { lane: 0, prefix: "p1".into(), target: 2 }, &s, strict);
    s.admit(0).unwrap();
    rec(&mut t, TraceAction::Admit { lane: 0 }, &s, strict);
    s.begin_restore(0).unwrap();
    rec(&mut t, TraceAction::BeginRestore { lane: 0 }, &s, strict);
    s.finish_restore(0).unwrap();
    rec(&mut t, TraceAction::FinishRestore { lane: 0 }, &s, strict);

    assert_eq!(s.arrive(1, "p2", 4).unwrap(), ArriveOutcome::Queued);
    rec(&mut t, TraceAction::Arrive { lane: 1, prefix: "p2".into(), target: 4 }, &s, strict);
    assert_eq!(s.arrive(2, "p1", 2).unwrap(), ArriveOutcome::Queued);
    rec(&mut t, TraceAction::Arrive { lane: 2, prefix: "p1".into(), target: 2 }, &s, strict);
    s.admit(2).unwrap();
    rec(&mut t, TraceAction::Admit { lane: 2 }, &s, strict);

    if let Some(built) = s.build_step().unwrap() {
        rec(&mut t, TraceAction::BuildStep { spans: built.spans }, &s, strict);
        s.gpu_execute().unwrap();
        rec(&mut t, TraceAction::GpuExecute, &s, strict);
        s.complete_step().unwrap();
        rec(&mut t, TraceAction::FinishStep, &s, strict);
    }
    if s.slide_window(0) {
        rec(&mut t, TraceAction::SlideWindow { lane: 0 }, &s, strict);
    }
    if let Some(built) = s.build_step().unwrap() {
        rec(&mut t, TraceAction::BuildStep { spans: built.spans }, &s, strict);
        let _ = s.cancel(0).unwrap();
        rec(&mut t, TraceAction::Cancel { lane: 0 }, &s, strict);
        s.gpu_execute().unwrap();
        rec(&mut t, TraceAction::GpuExecute, &s, strict);
        s.complete_step().unwrap();
        rec(&mut t, TraceAction::FinishStep, &s, strict);
    }

    assert_eq!(s.arrive(0, "p1", 2).unwrap(), ArriveOutcome::Queued);
    rec(&mut t, TraceAction::Arrive { lane: 0, prefix: "p1".into(), target: 2 }, &s, strict);
    s.admit(0).unwrap();
    rec(&mut t, TraceAction::Admit { lane: 0 }, &s, strict);
    s.begin_restore(0).unwrap();
    rec(&mut t, TraceAction::BeginRestore { lane: 0 }, &s, strict);
    s.finish_restore(0).unwrap();
    rec(&mut t, TraceAction::FinishRestore { lane: 0 }, &s, strict);

    if let Some(built) = s.build_step().unwrap() {
        // Only taken if bypass-ceiling check was sabotaged!
        rec(&mut t, TraceAction::BuildStep { spans: built.spans }, &s, strict);
        s.gpu_execute().unwrap();
        rec(&mut t, TraceAction::GpuExecute, &s, strict);
        s.complete_step().unwrap();
        rec(&mut t, TraceAction::FinishStep, &s, strict);
    } else {
        s.begin_restore(2).unwrap();
        rec(&mut t, TraceAction::BeginRestore { lane: 2 }, &s, strict);
        s.finish_restore(2).unwrap();
        rec(&mut t, TraceAction::FinishRestore { lane: 2 }, &s, strict);
        drain_to_idle(&mut t, &mut s, strict);
    }
    t
}

/// Deterministic cold-spill + `SlideWindow` + cancel-during-restore (`BeginRestoreIO -> Cancel ->
/// CompleteRestoreIO`) + atomic `RestoreBlock` + full drain (`U88, U265, U266, U268`).
fn cold_spill_and_slide_trace(cfg: SchedulerConfig, strict: bool) -> SchedulerTrace {
    let mut s = Scheduler::new(cfg).expect("init scheduler");
    if strict {
        s.check_invariants().unwrap();
    }
    let mut t = SchedulerTrace::new(&s);

    assert_eq!(s.arrive(0, "p1", 4).unwrap(), ArriveOutcome::Queued);
    rec(&mut t, TraceAction::Arrive { lane: 0, prefix: "p1".into(), target: 4 }, &s, strict);
    s.admit(0).unwrap();
    rec(&mut t, TraceAction::Admit { lane: 0 }, &s, strict);
    s.begin_restore(0).unwrap();
    rec(&mut t, TraceAction::BeginRestore { lane: 0 }, &s, strict);
    s.finish_restore(0).unwrap();
    rec(&mut t, TraceAction::FinishRestore { lane: 0 }, &s, strict);

    while s.lanes()[0].pos < 4 {
        if s.slide_window(0) {
            rec(&mut t, TraceAction::SlideWindow { lane: 0 }, &s, strict);
        }
        let built = s.build_step().unwrap().expect("lane a step");
        rec(&mut t, TraceAction::BuildStep { spans: built.spans }, &s, strict);
        s.gpu_execute().unwrap();
        rec(&mut t, TraceAction::GpuExecute, &s, strict);
        s.complete_step().unwrap();
        rec(&mut t, TraceAction::FinishStep, &s, strict);
    }
    s.finish_ready(0).unwrap();
    rec(&mut t, TraceAction::FinishReady { lane: 0 }, &s, strict);

    assert_eq!(s.arrive(1, "p2", 4).unwrap(), ArriveOutcome::Queued);
    rec(&mut t, TraceAction::Arrive { lane: 1, prefix: "p2".into(), target: 4 }, &s, strict);
    s.admit(1).unwrap();
    rec(&mut t, TraceAction::Admit { lane: 1 }, &s, strict);
    s.begin_restore(1).unwrap();
    rec(&mut t, TraceAction::BeginRestore { lane: 1 }, &s, strict);
    s.finish_restore(1).unwrap();
    rec(&mut t, TraceAction::FinishRestore { lane: 1 }, &s, strict);

    let built = s.build_step().unwrap().expect("lane b prefill");
    rec(&mut t, TraceAction::BuildStep { spans: built.spans }, &s, strict);
    s.gpu_execute().unwrap();
    rec(&mut t, TraceAction::GpuExecute, &s, strict);
    s.complete_step().unwrap();
    rec(&mut t, TraceAction::FinishStep, &s, strict);

    assert!(s.mem_pressure(), "cold p1 blocks must face memory pressure before lane b decode");
    let wb = s.start_cold_writeback().expect("cold writeback");
    rec(&mut t, TraceAction::StartColdWriteback { block: wb.clone() }, &s, strict);
    s.complete_step_io().unwrap();
    rec(&mut t, TraceAction::CompleteStepIO, &s, strict);

    s.cancel(1).unwrap();
    rec(&mut t, TraceAction::Cancel { lane: 1 }, &s, strict);

    // Exercise cancel-during-restore (`BeginRestoreIO -> Cancel -> Cancelling -> CompleteRestoreIO -> Cancelled`)!
    assert_eq!(s.arrive(2, "p1", 4).unwrap(), ArriveOutcome::Queued);
    rec(&mut t, TraceAction::Arrive { lane: 2, prefix: "p1".into(), target: 4 }, &s, strict);
    s.admit(2).unwrap();
    rec(&mut t, TraceAction::Admit { lane: 2 }, &s, strict);
    s.begin_restore(2).unwrap();
    rec(&mut t, TraceAction::BeginRestore { lane: 2 }, &s, strict);
    s.begin_restore_io(2, &wb).unwrap();
    rec(&mut t, TraceAction::BeginRestoreIO { lane: 2, block: wb.clone() }, &s, strict);
    let cancel_res = s.cancel(2).unwrap();
    rec(&mut t, TraceAction::Cancel { lane: 2 }, &s, strict);
    if cancel_res == LaneState::Cancelling {
        s.complete_restore_io(2).unwrap();
        rec(&mut t, TraceAction::CompleteRestoreIO { lane: 2 }, &s, strict);
    } else {
        // Sabotaged `cancel-in-restore-early-free` mutant skipped `Cancelling` while `restore_io_block` remained set!
        return t;
    }

    // Now restore `p1` on lane 3 from disk via atomic `RestoreBlock` and drain cleanly!
    assert_eq!(s.arrive(3, "p1", 4).unwrap(), ArriveOutcome::Queued);
    rec(&mut t, TraceAction::Arrive { lane: 3, prefix: "p1".into(), target: 4 }, &s, strict);
    s.admit(3).unwrap();
    rec(&mut t, TraceAction::Admit { lane: 3 }, &s, strict);
    s.begin_restore(3).unwrap();
    rec(&mut t, TraceAction::BeginRestore { lane: 3 }, &s, strict);
    s.restore_block(3, &wb).unwrap();
    rec(&mut t, TraceAction::RestoreBlock { lane: 3, block: wb.clone() }, &s, strict);
    if s.blocks().get(&wb).is_some_and(|r| r.tier != BlockTier::Disk) {
        s.finish_restore(3).unwrap();
        rec(&mut t, TraceAction::FinishRestore { lane: 3 }, &s, strict);
        drain_to_idle(&mut t, &mut s, strict);
    }
    t
}

fn add_stats(dst: &mut WalkStats, src: &WalkStats) {
    dst.steps += src.steps;
    dst.arrivals += src.arrivals;
    dst.refused_429 += src.refused_429;
    dst.admits += src.admits;
    dst.admit_bypasses += src.admit_bypasses;
    dst.step_bypasses += src.step_bypasses;
    dst.cancel_in_step += src.cancel_in_step;
    dst.cancel_in_restore += src.cancel_in_restore;
    dst.finished += src.finished;
    dst.cold_writebacks += src.cold_writebacks;
    dst.disk_restores += src.disk_restores;
    dst.slide_windows += src.slide_windows;
    dst.max_wait_ticks_seen = dst.max_wait_ticks_seen.max(src.max_wait_ticks_seen);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_dir = Path::new(args.get(1).map(String::as_str).unwrap_or("."));
    let strict = args.get(2).map(String::as_str) != Some("--mutant");
    let cfg = trace_cfg();

    let mut traces = Vec::new();
    traces.push(targeted_edge_trace(cfg.clone(), strict));
    traces.push(cold_spill_and_slide_trace(cfg.clone(), strict));

    let mut agg = WalkStats::default();
    agg.steps = traces[0].steps.len() + traces[1].steps.len();

    // Seeded walks ONLY — never silently drop short walks (`U265`)!
    for seed in 0..16u64 {
        let (trace, stats) = walk_scheduler_trace_opts(cfg.clone(), seed, 160, strict)
            .unwrap_or_else(|e| panic!("walk seed {seed} failed: {e}"));
        assert!(
            trace.steps.len() >= 2,
            "walk seed {seed} produced fewer than 2 steps ({})",
            trace.steps.len()
        );
        add_stats(&mut agg, &stats);
        traces.push(trace);
    }

    let (data, check, tlc_cfg) =
        render_cache_scheduler_tla_bundle(&traces).expect("render TLA+ trace bundle");
    fs::create_dir_all(out_dir).expect("create trace output dir");
    fs::write(out_dir.join("TraceData.tla"), data).expect("write TraceData.tla");
    fs::write(out_dir.join("TraceCheck.tla"), check).expect("write TraceCheck.tla");
    fs::write(out_dir.join("TraceCheck.cfg"), tlc_cfg).expect("write TraceCheck.cfg");

    let required = [
        ("cold_writebacks", agg.cold_writebacks),
        ("disk_restores", agg.disk_restores),
        ("refused_429", agg.refused_429),
        ("cancel_in_step", agg.cancel_in_step),
        ("cancel_in_restore", agg.cancel_in_restore),
        ("admit_bypasses", agg.admit_bypasses),
        ("step_bypasses", agg.step_bypasses),
        ("slide_windows", agg.slide_windows),
        ("finished", agg.finished),
    ];
    let summary: Vec<String> = required.iter().map(|(k, v)| format!("{k}={v}")).collect();
    eprintln!(
        "scheduler-traces: {} traces, {} states; {}",
        traces.len(),
        agg.steps,
        summary.join(", ")
    );

    if strict {
        let mut gap = false;
        for (name, count) in required {
            if count == 0 {
                eprintln!("COVERAGE GAP: seeded scheduler walks never reached {name}");
                gap = true;
            }
        }
        if gap {
            std::process::exit(1);
        }
    }
}
