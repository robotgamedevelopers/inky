//! Test-only observation of the step selected by the serving scheduler, including its
//! runnable inputs and the ages either side of the call. The standalone TLC gate compiles
//! this module and the same production planning entry point on hosts without Metal.
use crate::scheduler::{ServeLaneInput, ServeStepPlan, ServeStepPlanner};
use std::cell::RefCell;
use std::io::{BufWriter, Write};

thread_local! {
    static FRAMES: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
}

pub(super) fn snapshot(planner: &ServeStepPlanner, lanes: usize) -> Option<Vec<u32>> {
    FRAMES.with_borrow(|frames| {
        frames
            .as_ref()
            .map(|_| (0..lanes).map(|i| planner.wait_ticks(i)).collect())
    })
}

pub(super) fn record(
    planner: &ServeStepPlanner,
    inputs: &[ServeLaneInput],
    budget: usize,
    align: usize,
    before: &[u32],
    plan: Option<&ServeStepPlan>,
) {
    let lanes = inputs.len();
    let mut spans = vec![0; lanes];
    let mut starts = vec![0; lanes];
    let mut select = |i: usize, p0: usize, len: usize| {
        assert!(
            len > 0 && spans[i] == 0,
            "invalid or duplicate span in the step"
        );
        spans[i] = len;
        starts[i] = p0;
    };
    let mode = match plan {
        None => "idle",
        Some(ServeStepPlan::Decode(chosen)) => {
            for &i in chosen {
                select(i, inputs[i].n_past, 1);
            }
            "decode"
        }
        Some(ServeStepPlan::Prefill(chosen)) => {
            for &(i, p0, len) in chosen {
                select(i, p0, len);
            }
            "prefill"
        }
    };
    let runnable: Vec<_> = inputs.iter().map(|i| i.runnable).collect();
    let pending: Vec<_> = inputs.iter().map(|i| i.has_pending_decode).collect();
    let pos: Vec<_> = inputs.iter().map(|i| i.n_past).collect();
    let end: Vec<_> = inputs.iter().map(|i| i.pf_end).collect();
    let after: Vec<_> = (0..lanes).map(|i| planner.wait_ticks(i)).collect();
    // All interpolated arrays contain integers or booleans. Keep this gate usable with rustc alone.
    let frame = format!(
        r#"{{"version":2,"runnable":{runnable:?},"pending":{pending:?},"pos":{pos:?},"endPos":{end:?},"before":{before:?},"after":{after:?},"starts":{starts:?},"spans":{spans:?},"mode":"{mode}","budget":{budget},"align":{align},"ceiling":{}}}"#,
        planner.max_bypass(),
    );
    FRAMES.with_borrow_mut(|frames| frames.as_mut().unwrap().push(frame));
}

fn capture(work: impl FnOnce()) -> Vec<String> {
    FRAMES.with_borrow_mut(|frames| assert!(frames.replace(Vec::new()).is_none()));
    work();
    let frames = FRAMES.with_borrow_mut(Option::take).unwrap();
    assert!(
        !frames.is_empty(),
        "workload did not reach the serving planner"
    );
    frames
}

fn write_trace(frames: &[String]) {
    let path = std::env::var_os("INKY_SCHED_TRACE_OUT").expect("set INKY_SCHED_TRACE_OUT");
    let mut out = BufWriter::new(std::fs::File::create(path).unwrap());
    for frame in frames {
        writeln!(out, "{frame}").unwrap();
    }
    out.flush().unwrap();
}

/// Capture the real GPU-backed scheduler only when the hardware test requests a trace.
pub(super) fn capture_if_requested(work: impl FnOnce()) {
    if std::env::var_os("INKY_SCHED_TRACE_OUT").is_some() {
        write_trace(&capture(work));
    } else {
        work();
    }
}

fn walk(mut seed: u64) -> Vec<String> {
    use super::step::{plan, SERVE_STEP_MAX_BYPASS};
    const LANES: usize = 4;
    const ALIGN: usize = 64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 32) as usize
    };
    capture(|| {
        let mut planner = ServeStepPlanner::new(LANES, SERVE_STEP_MAX_BYPASS);
        let mut inputs = [ServeLaneInput {
            runnable: true,
            n_past: 0,
            pf_end: 0,
            has_pending_decode: true,
        }; LANES];
        for (i, input) in inputs.iter_mut().enumerate() {
            planner.on_admit(i);
            if i >= 2 {
                input.pf_end = 512 + next() % 256;
            }
        }
        for tick in 0..64 {
            // A full W=16 burst followed by two urgent prefillers, then variable
            // backpressure and token budgets. Restart some lanes as new prompts.
            let random = tick > SERVE_STEP_MAX_BYPASS;
            for (i, lane) in inputs.iter_mut().enumerate() {
                lane.runnable = tick != 63 && (!random || next() % 4 != 0);
                if random && lane.n_past >= lane.pf_end && next() % 5 == 0 {
                    planner.on_finish(i);
                    planner.on_admit(i);
                    lane.n_past = next() % ALIGN;
                    lane.pf_end = lane.n_past + 1 + next() % 256;
                }
                lane.has_pending_decode =
                    lane.n_past >= lane.pf_end && (!random || next() % 4 != 0);
            }
            let budget = if random { next() % 160 } else { 128 };
            match plan(&mut planner, &inputs, budget, ALIGN) {
                Some(ServeStepPlan::Decode(chosen)) => {
                    for i in chosen {
                        inputs[i].n_past += 1;
                    }
                }
                Some(ServeStepPlan::Prefill(spans)) => {
                    for (i, _, len) in spans {
                        inputs[i].n_past += len;
                    }
                }
                None => {}
            }
            planner.complete_step();
        }
    })
}

#[test]
fn seeded_decisions_are_repeatable() {
    for seed in (0..32).chain([u64::MAX]) {
        let frames = walk(seed);
        assert_eq!(frames.len(), 64);
        let burst = super::step::SERVE_STEP_MAX_BYPASS as usize;
        assert!(frames[..burst]
            .iter()
            .all(|f| f.contains(r#""mode":"decode""#)));
        assert!(frames[burst].contains(r#""mode":"prefill""#));
        assert!(frames.last().unwrap().contains(r#""mode":"idle""#));
        assert_eq!(frames, walk(seed));
    }
}

#[test]
#[ignore = "writes serve planner observations to INKY_SCHED_TRACE_OUT"]
fn dump() {
    let out = match std::env::var("INKY_SCHED_TRACE_SEED") {
        Ok(seed) => walk(seed.parse().expect("seed must be a u64")),
        Err(_) => (0..32).chain([u64::MAX]).flat_map(walk).collect(),
    };
    write_trace(&out);
}
