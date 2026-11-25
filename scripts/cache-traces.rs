#![allow(dead_code)]
// Replay deterministic random walks of `src/cache_scheduler.rs` + `src/blocks.rs`
// through `src/cache_trace.rs` into TLA+ bundles for `BlockLayout.tla` and `CacheScheduler.tla`.
#[path = "blocks.rs"]
mod blocks;
#[path = "blockbook.rs"]
mod blockbook;
#[path = "cache_trace.rs"]
mod cache_trace;
#[path = "cache_scheduler.rs"]
mod cache_scheduler;

use std::fs;
use std::path::Path;

use cache_scheduler::{walk_block_layout, walk_cache_scheduler};
use cache_trace::{
    parse_block_layout_jsonl, parse_cache_scheduler_jsonl, render_block_layout_tla_bundle,
    render_cache_scheduler_tla_bundle, BlockLayoutConfig, CacheSchedulerConfig, MemorySink,
    NullSink,
};

fn write_bundle(dir: &Path, bundle: (String, String, String)) {
    fs::create_dir_all(dir).expect("create bundle dir");
    fs::write(dir.join("TraceData.tla"), bundle.0).expect("write TraceData.tla");
    fs::write(dir.join("TraceCheck.tla"), bundle.1).expect("write TraceCheck.tla");
    fs::write(dir.join("TraceCheck.cfg"), bundle.2).expect("write TraceCheck.cfg");
}

fn gen_bl(out_root: &Path, name: &str, slot_base: u32, kvb: u32) {
    let cfg = BlockLayoutConfig {
        lanes: vec!["a".into(), "b".into()],
        full_capacity: 2,
        window_capacity: 2,
        block_size: 2,
        window_size: 2,
        max_pos: 4,
        max_span: 2,
        max_generation: 2,
        slot_base,
        kvb,
    };
    let mut traces = Vec::new();
    let mut walk_errs = 0usize;
    let mut max_gen = 0u32;
    let mut start_steps = 0usize;
    let mut two_slots = 0usize;
    for seed in 0..16 {
        let mut ev = NullSink;
        let mut st = MemorySink::new();
        if walk_block_layout(cfg.clone(), seed, &mut ev, &mut st).is_err() {
            walk_errs += 1;
        }
        if let Ok(t) = parse_block_layout_jsonl(&st.to_jsonl()) {
            for s in &t.steps {
                if s.state.start_kept {
                    start_steps += 1;
                }
                for p in s.state.full_pool.iter().chain(&s.state.window_pool) {
                    max_gen = max_gen.max(p.generation);
                    if p.slot == 2 && p.generation >= 1 {
                        two_slots += 1;
                    }
                }
            }
            if t.steps.len() >= 2 {
                traces.push(t);
            }
        } else {
            walk_errs += 1;
        }
    }
    if walk_errs > 0 || max_gen < 2 || start_steps == 0 || two_slots == 0 {
        eprintln!(
            "COVERAGE GAP {name}: walk_errs={walk_errs} max_gen={max_gen} start_steps={start_steps} two_slots={two_slots}"
        );
    } else {
        eprintln!(
            "{name}: {} walks, max_gen={max_gen}, start_steps={start_steps}, two_slots={two_slots}",
            traces.len()
        );
    }
    if !traces.is_empty() {
        write_bundle(&out_root.join(name), render_block_layout_tla_bundle(&traces).unwrap());
    }
}

fn gen_cs(out_root: &Path, name: &str) {
    let cfg = CacheSchedulerConfig {
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
    };
    let mut traces = Vec::new();
    let mut walk_errs = 0usize;
    let mut admit_bypass = 0usize;
    let mut step_bypass = 0usize;
    let mut cancel_cnt = 0usize;
    let mut refuse_cnt = 0usize;

    for seed in 0..16 {
        let mut ev = MemorySink::new();
        let mut st = MemorySink::new();
        if walk_cache_scheduler(cfg.clone(), seed, &mut ev, &mut st).is_err() {
            walk_errs += 1;
        }
        for line in ev.lines() {
            if line.contains("\"event\":\"cancel\"") {
                cancel_cnt += 1;
            }
            if line.contains("\"event\":\"refuse429\"") {
                refuse_cnt += 1;
            }
        }
        if let Ok(t) = parse_cache_scheduler_jsonl(&st.to_jsonl()) {
            for w in t.steps.windows(2) {
                if w[1].action == "Admit" {
                    for (l0, l1) in w[0].state.lanes.iter().zip(&w[1].state.lanes) {
                        if l0.state == "Queued" && l1.state == "Queued" && l1.wait_ticks > l0.wait_ticks {
                            admit_bypass += 1;
                        }
                    }
                }
                if w[1].action == "BuildStep" {
                    for (l0, l1) in w[0].state.lanes.iter().zip(&w[1].state.lanes) {
                        if l1.wait_ticks > l0.wait_ticks {
                            step_bypass += 1;
                        }
                    }
                }
            }
            if t.steps.len() >= 2 {
                traces.push(t);
            }
        } else {
            walk_errs += 1;
        }
    }
    if walk_errs > 0 || admit_bypass == 0 || step_bypass == 0 || cancel_cnt == 0 || refuse_cnt == 0 {
        eprintln!(
            "COVERAGE GAP {name}: walk_errs={walk_errs} admit_bypass={admit_bypass} step_bypass={step_bypass} cancel={cancel_cnt} refuse429={refuse_cnt}"
        );
    } else {
        eprintln!(
            "{name}: {} walks, admit_bypass={admit_bypass}, step_bypass={step_bypass}, cancel={cancel_cnt}, refuse429={refuse_cnt}",
            traces.len()
        );
    }
    if !traces.is_empty() {
        write_bundle(&out_root.join(name), render_cache_scheduler_tla_bundle(&traces).unwrap());
    }
}

fn gen_bl_mutant(out_root: &Path, name: &str, slot_base: u32, kvb: u32) {
    let cfg = BlockLayoutConfig {
        lanes: vec!["a".into(), "b".into()],
        full_capacity: 2,
        window_capacity: 2,
        block_size: 2,
        window_size: 2,
        max_pos: 4,
        max_span: 2,
        max_generation: 2,
        slot_base,
        kvb,
    };
    let mut traces = Vec::new();
    for seed in 0..4 {
        let mut ev = NullSink;
        let mut st = MemorySink::new();
        let _ = walk_block_layout(cfg.clone(), seed, &mut ev, &mut st);
        if let Ok(t) = parse_block_layout_jsonl(&st.to_jsonl()) {
            if t.steps.len() >= 2 {
                traces.push(t);
                if traces.len() >= 2 {
                    break;
                }
            }
        }
    }
    if !traces.is_empty() {
        write_bundle(&out_root.join(name), render_block_layout_tla_bundle(&traces).unwrap());
    }
}

fn gen_cs_mutant(out_root: &Path, name: &str, target: &str) {
    let cfg = CacheSchedulerConfig {
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
    };
    let mut best_admit: Option<(usize, cache_trace::CacheSchedulerTrace)> = None;
    let mut best_step: Option<(usize, cache_trace::CacheSchedulerTrace)> = None;
    let mut best_priv: Option<(usize, cache_trace::CacheSchedulerTrace)> = None;
    let mut fallback: Option<cache_trace::CacheSchedulerTrace> = None;

    for seed in 0..16 {
        let mut ev = NullSink;
        let mut st = MemorySink::new();
        let _ = walk_cache_scheduler(cfg.clone(), seed, &mut ev, &mut st);
        let Ok(t) = parse_cache_scheduler_jsonl(&st.to_jsonl()) else {
            continue;
        };
        if t.steps.len() < 2 {
            continue;
        }
        if fallback.is_none() {
            fallback = Some(t.clone());
        }
        for (idx, step) in t.steps.iter().enumerate() {
            // Structural unsealed-private-tail condition (independent of data mutation):
            // any present private block with sealed == false and length < block_size.
            if step
                .state
                .blocks
                .iter()
                .any(|b| !b.sealed && b.tier != "none" && b.length < cfg.block_size)
                && best_priv.as_ref().map_or(true, |(best, _)| idx < *best)
            {
                let mut trimmed = t.clone();
                trimmed.steps.truncate((idx + 2).max(2).min(t.steps.len()));
                best_priv = Some((idx, trimmed));
                break;
            }
        }
        for (idx, w) in t.steps.windows(2).enumerate() {
            // Structural admit-bypass condition (independent of wait_ticks mutation):
            // a lane transitioned Queued -> Admitted while at least one queued lane ahead in queue_seq remained Queued.
            let admitted_lane = w[0]
                .state
                .lanes
                .iter()
                .zip(&w[1].state.lanes)
                .find(|(l0, l1)| l0.state == "Queued" && l1.state == "Admitted")
                .map(|(l0, _)| l0.lane.as_str());
            if let Some(adm) = admitted_lane {
                let older_queued = w[0]
                    .state
                    .queue_seq
                    .iter()
                    .take_while(|q| q.as_str() != adm)
                    .any(|q| w[1].state.lanes.iter().any(|l| l.lane == *q && l.state == "Queued"));
                if older_queued && best_admit.as_ref().map_or(true, |(best, _)| idx < *best) {
                    let mut trimmed = t.clone();
                    trimmed.steps.truncate((idx + 3).min(t.steps.len()));
                    best_admit = Some((idx, trimmed));
                }
            }
            // Structural step-bypass condition (independent of wait_ticks mutation):
            // BuildStep ran while a Ready/Restoring/Admitted lane was excluded from step_lanes.
            if w[1].action == "BuildStep"
                && w[0].state.lanes.iter().any(|l0| {
                    matches!(l0.state.as_str(), "Admitted" | "Restoring" | "Ready")
                        && !w[1].state.step_lanes.contains(&l0.lane)
                })
                && best_step.as_ref().map_or(true, |(best, _)| idx < *best)
            {
                let mut trimmed = t.clone();
                trimmed.steps.truncate((idx + 3).min(t.steps.len()));
                best_step = Some((idx, trimmed));
            }
        }
    }

    let mut traces = Vec::new();
    let chosen = match target {
        "admit-wait-dropped" => best_admit.take(),
        "step-bypass-dropped" => best_step.take(),
        "priv-tail-dirty" => best_priv.take(),
        _ => None,
    };
    if let Some((_, t)) = chosen {
        traces.push(t);
    } else {
        for candidate in [best_admit, best_step, best_priv] {
            if let Some((_, t)) = candidate {
                if traces.iter().all(|existing: &cache_trace::CacheSchedulerTrace| existing.steps != t.steps) {
                    traces.push(t);
                }
            }
        }
    }
    if traces.is_empty() {
        if let Some(mut t) = fallback {
            t.steps.truncate(t.steps.len().min(32));
            traces.push(t);
        }
    }
    if !traces.is_empty() {
        write_bundle(&out_root.join(name), render_cache_scheduler_tla_bundle(&traces).unwrap());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_root = Path::new(args.get(1).map(String::as_str).unwrap_or("."));
    let mode = args.get(2).map(String::as_str).unwrap_or("all");
    let mutant_name = args.get(3).map(String::as_str).unwrap_or("");
    match mode {
        "bl_base0" => gen_bl(out_root, "bl_base0", 0, 2),
        "bl_high_fp16" => gen_bl(out_root, "bl_high_fp16", 2048, 2),
        "bl_high_kv8" => gen_bl(out_root, "bl_high_kv8", 4096, 1),
        "bl_base0_mutant" => gen_bl_mutant(out_root, "bl_base0", 0, 2),
        "bl_high_fp16_mutant" => gen_bl_mutant(out_root, "bl_high_fp16", 2048, 2),
        "cs" => gen_cs(out_root, "cs"),
        "cs_mutant" => gen_cs_mutant(out_root, "cs", mutant_name),
        "all" => {
            gen_bl(out_root, "bl_base0", 0, 2);
            gen_bl(out_root, "bl_high_fp16", 2048, 2);
            gen_bl(out_root, "bl_high_kv8", 4096, 1);
            gen_cs(out_root, "cs");
        }
        other => panic!("unknown cache-traces mode {other:?}"),
    }
}
