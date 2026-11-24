#![allow(dead_code)]
// CPU-only trace generator for `src/scheduler.rs` against `specs/CacheScheduler.tla`.
// Compiled directly via `rustc --edition=2021` by `scripts/check-scheduler-traces.sh`.

#[path = "scheduler.rs"]
mod scheduler;

use std::fs;
use std::path::Path;

use scheduler::{
    render_cache_scheduler_tla_bundle, walk_scheduler_trace, ArriveOutcome, LaneState, Scheduler,
    SchedulerConfig, SchedulerTrace,
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
    t.record("Arrive", &s);
    s.admit(0).unwrap();
    t.record("Admit", &s);
    s.begin_restore(0).unwrap();
    t.record("BeginRestore", &s);
    s.finish_restore(0).unwrap();
    t.record("FinishRestore", &s);

    // 2. Lane b arrives with len 4 (wc=4 -> cannot fit while lane a holds 2 of 4 blocks),
    //    then younger lane c arrives with len 2 (wc=2 -> fits and leapfrogs lane b!).
    assert_eq!(s.arrive(1, "p2", 4).unwrap(), ArriveOutcome::Queued);
    t.record("Arrive", &s);
    assert_eq!(s.arrive(2, "p1", 2).unwrap(), ArriveOutcome::Queued);
    t.record("Arrive", &s);
    // Admit lane c -> bypasses older queued lane b (b.wait_ticks becomes 1).
    s.admit(2).unwrap();
    t.record("Admit", &s);

    // 3. Lane c is now Admitted (can_advance_admitted == true, wait_ticks == 0).
    //    Run 2 steps on lane a while lane c stays Admitted -> c.wait_ticks reaches 1, then 2 (= W)!
    if s.build_step().unwrap().is_some() {
        t.record("BuildStep", &s);
        s.gpu_execute().unwrap();
        t.record("GpuExecute", &s);
        s.complete_step().unwrap();
        t.record("FinishStep", &s);
    }
    if s.build_step().unwrap().is_some() {
        t.record("BuildStep", &s);
        // Cancel lane a mid-step before gpu_execute -> exercises InStep -> Cancelling -> Cancelled!
        let _ = s.cancel(0).unwrap();
        t.record("CancelInStep", &s);
        s.gpu_execute().unwrap();
        t.record("GpuExecute", &s);
        s.complete_step().unwrap();
        t.record("FinishStep", &s);
    }

    // 4. Now lane c has wait_ticks == 2 (== W). Bring lane a back Ready and attempt build_step
    //    while lane c is still Admitted at W: production build_step MUST refuse (Ok(None)),
    //    whereas bypass-ceiling mutant will bypass c to wait_ticks == 3 (> W) -> TLC violation!
    assert_eq!(s.arrive(0, "p1", 2).unwrap(), ArriveOutcome::Queued);
    t.record("Arrive", &s);
    s.admit(0).unwrap();
    t.record("Admit", &s);
    s.begin_restore(0).unwrap();
    t.record("BeginRestore", &s);
    s.finish_restore(0).unwrap();
    t.record("FinishRestore", &s);

    if s.build_step().unwrap().is_some() {
        // Only taken if bypass-ceiling check was sabotaged!
        t.record("BuildStep", &s);
        s.gpu_execute().unwrap();
        t.record("GpuExecute", &s);
        s.complete_step().unwrap();
        t.record("FinishStep", &s);
    } else {
        // Production path: advance lane c at ceiling W first.
        s.begin_restore(2).unwrap();
        t.record("BeginRestore", &s);
        s.finish_restore(2).unwrap();
        t.record("FinishRestore", &s);
        if s.build_step().unwrap().is_some() {
            t.record("BuildStep", &s);
            s.gpu_execute().unwrap();
            t.record("GpuExecute", &s);
            s.complete_step().unwrap();
            t.record("FinishStep", &s);
        }
    }

    // Verify lane c is valid (or cancelled/finished) and return.
    let _ = LaneState::Cancelled;
    t
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_dir = Path::new(args.get(1).map(String::as_str).unwrap_or("."));
    let cfg = trace_cfg();

    let mut traces = Vec::new();
    traces.push(targeted_edge_trace(cfg.clone()));

    let mut total_steps = traces[0].steps.len();
    let mut total_cancel = 0usize;
    let mut total_step_bypass = 0usize;
    let mut total_admit_bypass = 0usize;
    let mut total_refuse = 0usize;
    let mut total_finished = 0usize;

    for seed in 0..12u64 {
        let (trace, stats) = walk_scheduler_trace(cfg.clone(), seed, 140)
            .unwrap_or_else(|e| panic!("walk seed {seed} failed: {e}"));
        total_steps += trace.steps.len();
        total_cancel += stats.cancel_in_step;
        total_step_bypass += stats.step_bypasses;
        total_admit_bypass += stats.admit_bypasses;
        total_refuse += stats.refused_429;
        total_finished += stats.finished;
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
        "scheduler-traces: {} traces, {} states, cancel_in_step={}, step_bypass={}, admit_bypass={}, refuse429={}, finished={}",
        traces.len(),
        total_steps,
        total_cancel,
        total_step_bypass,
        total_admit_bypass,
        total_refuse,
        total_finished
    );
}
