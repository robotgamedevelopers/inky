#![allow(dead_code)]
// CPU-only trace generator for `src/scheduler.rs` against `specs/CacheScheduler.tla`.
// Compiled directly via `rustc --edition=2021` by `scripts/check-scheduler-traces.sh`.

#[path = "scheduler.rs"]
mod scheduler;

use std::fs;
use std::path::Path;

use scheduler::{
    render_cache_scheduler_tla_bundle, walk_scheduler_trace, ArriveOutcome, BlockTier, LaneState,
    Scheduler, SchedulerConfig, SchedulerTrace, TraceAction,
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

/// Deterministic targeted trace that explicitly exercises:
/// - Admit bypass (`Queued` lane 0 of len 4 bypassed by younger `Queued` lane 1 of len 2),
/// - Step bypass up to ceiling `W = 2` (and verifying `build_step` refuses to bypass past `W`),
/// - Cancel-in-step barrier (`InStep -> Cancelling -> Cancelled` across `gpu_execute` + `complete_step`).
fn targeted_edge_trace(cfg: SchedulerConfig) -> SchedulerTrace {
    let mut s = Scheduler::new(cfg).expect("init scheduler");
    let mut t = SchedulerTrace::new(&s);

    // 1. Lane a arrives with len 2 (wc=2), admits, restores, becomes Ready.
    assert_eq!(s.arrive(0, "p1", 2).unwrap(), ArriveOutcome::Queued);
    t.record_action(TraceAction::Arrive { lane: 0, prefix: "p1".into(), target: 2 }, &s);
    s.admit(0).unwrap();
    t.record_action(TraceAction::Admit { lane: 0 }, &s);
    s.begin_restore(0).unwrap();
    t.record_action(TraceAction::BeginRestore { lane: 0 }, &s);
    s.finish_restore(0).unwrap();
    t.record_action(TraceAction::FinishRestore { lane: 0 }, &s);

    // 2. Lane b arrives with len 4 (wc=4 -> cannot fit while lane a holds 2 of 4 blocks),
    //    then younger lane c arrives with len 2 (wc=2 -> fits and leapfrogs lane b!).
    assert_eq!(s.arrive(1, "p2", 4).unwrap(), ArriveOutcome::Queued);
    t.record_action(TraceAction::Arrive { lane: 1, prefix: "p2".into(), target: 4 }, &s);
    assert_eq!(s.arrive(2, "p1", 2).unwrap(), ArriveOutcome::Queued);
    t.record_action(TraceAction::Arrive { lane: 2, prefix: "p1".into(), target: 2 }, &s);
    // Admit lane c -> bypasses older queued lane b (b.wait_ticks becomes 1).
    s.admit(2).unwrap();
    t.record_action(TraceAction::Admit { lane: 2 }, &s);

    // 3. Lane c is now Admitted (can_advance_admitted == true, wait_ticks == 0).
    //    Run 2 steps on lane a while lane c stays Admitted -> c.wait_ticks reaches 1, then 2 (= W)!
    if let Some(built) = s.build_step().unwrap() {
        t.record_action(TraceAction::BuildStep { spans: built.spans }, &s);
        s.gpu_execute().unwrap();
        t.record_action(TraceAction::GpuExecute, &s);
        s.complete_step().unwrap();
        t.record_action(TraceAction::FinishStep, &s);
    }
    if s.slide_window(0) {
        t.record_action(TraceAction::SlideWindow { lane: 0 }, &s);
    }
    if let Some(built) = s.build_step().unwrap() {
        t.record_action(TraceAction::BuildStep { spans: built.spans }, &s);
        // Cancel lane a mid-step before gpu_execute -> exercises InStep -> Cancelling -> Cancelled!
        let _ = s.cancel(0).unwrap();
        t.record_action(TraceAction::Cancel { lane: 0 }, &s);
        s.gpu_execute().unwrap();
        t.record_action(TraceAction::GpuExecute, &s);
        s.complete_step().unwrap();
        t.record_action(TraceAction::FinishStep, &s);
    }

    // 4. Now lane c has wait_ticks == 2 (== W). Bring lane a back Ready and attempt build_step
    //    while lane c is still Admitted at W: production build_step MUST refuse (Ok(None)),
    //    whereas bypass-ceiling mutant will bypass c to wait_ticks == 3 (> W) -> TLC violation!
    assert_eq!(s.arrive(0, "p1", 2).unwrap(), ArriveOutcome::Queued);
    t.record_action(TraceAction::Arrive { lane: 0, prefix: "p1".into(), target: 2 }, &s);
    s.admit(0).unwrap();
    t.record_action(TraceAction::Admit { lane: 0 }, &s);
    s.begin_restore(0).unwrap();
    t.record_action(TraceAction::BeginRestore { lane: 0 }, &s);
    s.finish_restore(0).unwrap();
    t.record_action(TraceAction::FinishRestore { lane: 0 }, &s);

    if let Some(built) = s.build_step().unwrap() {
        // Only taken if bypass-ceiling check was sabotaged!
        t.record_action(TraceAction::BuildStep { spans: built.spans }, &s);
        s.gpu_execute().unwrap();
        t.record_action(TraceAction::GpuExecute, &s);
        s.complete_step().unwrap();
        t.record_action(TraceAction::FinishStep, &s);
    } else {
        // Production path: advance lane c at ceiling W first.
        s.begin_restore(2).unwrap();
        t.record_action(TraceAction::BeginRestore { lane: 2 }, &s);
        s.finish_restore(2).unwrap();
        t.record_action(TraceAction::FinishRestore { lane: 2 }, &s);
        if let Some(built) = s.build_step().unwrap() {
            t.record_action(TraceAction::BuildStep { spans: built.spans }, &s);
            s.gpu_execute().unwrap();
            t.record_action(TraceAction::GpuExecute, &s);
            s.complete_step().unwrap();
            t.record_action(TraceAction::FinishStep, &s);
        }
    }

    let _ = LaneState::Cancelled;
    t
}

/// Deterministic cold-spill + explicit `SlideWindow` + disk `RestoreBlock` trace (`U68` + coverage):
/// seals `(p1, 2)` via prefill, exercises `SlideWindow` during decode, spills `(p1, 2)` to disk
/// under memory pressure on `p2`, and restores `(p1, 2)` from disk via `RestoreBlock`.
fn cold_spill_and_slide_trace(cfg: SchedulerConfig) -> SchedulerTrace {
    let mut s = Scheduler::new(cfg).expect("init scheduler");
    let mut t = SchedulerTrace::new(&s);

    // 1. Lane a runs p1 (len 4): prefill 0->2 (seals (p1, 2)), decode 2->3, SlideWindow, decode 3->4, FinishReady.
    assert_eq!(s.arrive(0, "p1", 4).unwrap(), ArriveOutcome::Queued);
    t.record_action(TraceAction::Arrive { lane: 0, prefix: "p1".into(), target: 4 }, &s);
    s.admit(0).unwrap();
    t.record_action(TraceAction::Admit { lane: 0 }, &s);
    s.begin_restore(0).unwrap();
    t.record_action(TraceAction::BeginRestore { lane: 0 }, &s);
    s.finish_restore(0).unwrap();
    t.record_action(TraceAction::FinishRestore { lane: 0 }, &s);

    while s.lanes()[0].pos < 4 {
        if s.slide_window(0) {
            t.record_action(TraceAction::SlideWindow { lane: 0 }, &s);
        }
        let built = s.build_step().unwrap().expect("lane a step");
        t.record_action(TraceAction::BuildStep { spans: built.spans }, &s);
        s.gpu_execute().unwrap();
        t.record_action(TraceAction::GpuExecute, &s);
        s.complete_step().unwrap();
        t.record_action(TraceAction::FinishStep, &s);
    }
    s.finish_ready(0).unwrap();
    t.record_action(TraceAction::FinishReady { lane: 0 }, &s);

    // 2. Lane b runs p2 (len 4): prefill 0->2 fills MemBudget=4 (2 cold p1 blocks + 2 p2 blocks).
    assert_eq!(s.arrive(1, "p2", 4).unwrap(), ArriveOutcome::Queued);
    t.record_action(TraceAction::Arrive { lane: 1, prefix: "p2".into(), target: 4 }, &s);
    s.admit(1).unwrap();
    t.record_action(TraceAction::Admit { lane: 1 }, &s);
    s.begin_restore(1).unwrap();
    t.record_action(TraceAction::BeginRestore { lane: 1 }, &s);
    s.finish_restore(1).unwrap();
    t.record_action(TraceAction::FinishRestore { lane: 1 }, &s);

    let built = s.build_step().unwrap().expect("lane b prefill");
    t.record_action(TraceAction::BuildStep { spans: built.spans }, &s);
    s.gpu_execute().unwrap();
    t.record_action(TraceAction::GpuExecute, &s);
    s.complete_step().unwrap();
    t.record_action(TraceAction::FinishStep, &s);

    assert!(s.mem_pressure(), "cold p1 blocks must face memory pressure before lane b decode");
    let wb = s.start_cold_writeback().expect("cold writeback");
    t.record_action(TraceAction::StartColdWriteback { block: wb.clone() }, &s);
    s.complete_step_io().unwrap();
    t.record_action(TraceAction::CompleteStepIO, &s);

    // Cancel lane b and restore p1 on lane c from disk via `RestoreBlock`!
    s.cancel(1).unwrap();
    t.record_action(TraceAction::Cancel { lane: 1 }, &s);
    assert_eq!(s.arrive(2, "p1", 4).unwrap(), ArriveOutcome::Queued);
    t.record_action(TraceAction::Arrive { lane: 2, prefix: "p1".into(), target: 4 }, &s);
    s.admit(2).unwrap();
    t.record_action(TraceAction::Admit { lane: 2 }, &s);
    s.begin_restore(2).unwrap();
    t.record_action(TraceAction::BeginRestore { lane: 2 }, &s);

    s.restore_block(2, &wb).unwrap();
    t.record_action(TraceAction::RestoreBlock { lane: 2, block: wb.clone() }, &s);
    // If a mutant turned `restore_block` into a silent no-op stutter, stop here so TLC rejects it via `ActionConforms`!
    if s.blocks().get(&wb).is_some_and(|r| r.tier != BlockTier::Disk) {
        s.finish_restore(2).unwrap();
        t.record_action(TraceAction::FinishRestore { lane: 2 }, &s);
    }
    t
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_dir = Path::new(args.get(1).map(String::as_str).unwrap_or("."));
    let cfg = trace_cfg();

    let mut traces = Vec::new();
    traces.push(targeted_edge_trace(cfg.clone()));
    traces.push(cold_spill_and_slide_trace(cfg.clone()));

    let mut total_steps = traces[0].steps.len() + traces[1].steps.len();
    let mut total_cancel = 0usize;
    let mut total_step_bypass = 0usize;
    let mut total_admit_bypass = 0usize;
    let mut total_refuse = 0usize;
    let mut total_finished = 0usize;
    let mut total_slide = 0usize;

    for seed in 0..12u64 {
        let (trace, stats) = walk_scheduler_trace(cfg.clone(), seed, 140)
            .unwrap_or_else(|e| panic!("walk seed {seed} failed: {e}"));
        total_steps += trace.steps.len();
        total_cancel += stats.cancel_in_step;
        total_step_bypass += stats.step_bypasses;
        total_admit_bypass += stats.admit_bypasses;
        total_refuse += stats.refused_429;
        total_finished += stats.finished;
        total_slide += stats.slide_windows;
        if trace.steps.len() >= 2 {
            traces.push(trace);
        }
    }

    let (data, check, tlc_cfg) =
        render_cache_scheduler_tla_bundle(&traces).expect("render TLA+ trace bundle");
    fs::create_dir_all(out_dir).expect("create trace output dir");
    fs::write(out_dir.join("TraceData.tla"), data).expect("write TraceData.tla");
    fs::write(out_dir.join("TraceCheck.tla"), check).expect("write TraceCheck.tla");
    fs::write(out_dir.join("TraceCheck.cfg"), tlc_cfg).expect("write TraceCheck.cfg");

    eprintln!(
        "scheduler-traces: {} traces, {} states, cancel_in_step={}, step_bypass={}, admit_bypass={}, refuse429={}, slide_window={}, finished={}",
        traces.len(),
        total_steps,
        total_cancel,
        total_step_bypass,
        total_admit_bypass,
        total_refuse,
        total_slide,
        total_finished
    );
}
