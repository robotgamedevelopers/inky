#![allow(dead_code)]
// Replay deterministic random walks of `src/cache_scheduler.rs` + `src/blocks.rs`
// through `src/cache_trace.rs` into TLA+ bundles for `BlockLayout.tla` and `CacheScheduler.tla`.
#[path = "blocks.rs"]
mod blocks;
#[path = "blockbook.rs"]
mod blockbook;
#[path = "cache_trace.rs"]
mod cache_trace;
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

fn gen_bl(out_root: &Path, name: &str, slot_base: u32, kvb: u32, strict: bool) {
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
    let mut slide_steps = 0usize;
    for seed in 0..16 {
        let mut ev = NullSink;
        let mut st = MemorySink::new();
        if walk_block_layout(cfg.clone(), seed, &mut ev, &mut st).is_err() {
            walk_errs += 1;
        }
        if let Ok(t) = parse_block_layout_jsonl(&st.to_jsonl()) {
            for s in &t.steps {
                if s.action == "SlideWindow" {
                    slide_steps += 1;
                }
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
            } else {
                walk_errs += 1;
            }
        } else {
            walk_errs += 1;
        }
    }
    if walk_errs > 0 || max_gen < 2 || start_steps == 0 || two_slots == 0 || slide_steps == 0 {
        eprintln!(
            "COVERAGE GAP {name}: walk_errs={walk_errs} max_gen={max_gen} start_steps={start_steps} two_slots={two_slots} slide_steps={slide_steps}"
        );
        if strict {
            std::process::exit(1);
        }
    } else {
        eprintln!(
            "{name}: {} walks, max_gen={max_gen}, start_steps={start_steps}, two_slots={two_slots}, slide_steps={slide_steps}",
            traces.len()
        );
    }
    if !traces.is_empty() {
        write_bundle(&out_root.join(name), render_block_layout_tla_bundle(&traces).unwrap());
    }
}

fn gen_cs(out_root: &Path, name: &str, strict: bool) {
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
    let mut spill_cnt = 0usize;
    let mut restore_cnt = 0usize;
    let mut drop_cnt = 0usize;
    let mut slide_cnt = 0usize;
    let mut reclaim_cnt = 0usize;

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
                match w[1].action.as_str() {
                    "StartColdWriteback" => spill_cnt += 1,
                    "RestoreBlock" => restore_cnt += 1,
                    "DropUnreferenced" => drop_cnt += 1,
                    "SlideWindow" => slide_cnt += 1,
                    "ReclaimColdDisk" => reclaim_cnt += 1,
                    "Admit" => {
                        for (l0, l1) in w[0].state.lanes.iter().zip(&w[1].state.lanes) {
                            if l0.state == "Queued" && l1.state == "Queued" && l1.wait_ticks > l0.wait_ticks {
                                admit_bypass += 1;
                            }
                        }
                    }
                    "BuildStep" => {
                        for (l0, l1) in w[0].state.lanes.iter().zip(&w[1].state.lanes) {
                            if l1.wait_ticks > l0.wait_ticks {
                                step_bypass += 1;
                            }
                        }
                    }
                    _ => {}
                }
            }
            if t.steps.len() >= 2 {
                traces.push(t);
            } else {
                walk_errs += 1;
            }
        } else {
            walk_errs += 1;
        }
    }
    if walk_errs > 0
        || admit_bypass == 0
        || step_bypass == 0
        || cancel_cnt == 0
        || refuse_cnt == 0
        || spill_cnt == 0
        || restore_cnt == 0
        || drop_cnt == 0
        || slide_cnt == 0
        || reclaim_cnt == 0
    {
        eprintln!(
            "COVERAGE GAP {name}: walk_errs={walk_errs} admit_bypass={admit_bypass} step_bypass={step_bypass} cancel={cancel_cnt} refuse429={refuse_cnt} spill={spill_cnt} restore={restore_cnt} drop={drop_cnt} slide={slide_cnt} reclaim={reclaim_cnt}"
        );
        if strict {
            std::process::exit(1);
        }
    } else {
        eprintln!(
            "{name}: {} walks, admit_bypass={admit_bypass}, step_bypass={step_bypass}, cancel={cancel_cnt}, refuse429={refuse_cnt}, spill={spill_cnt}, restore={restore_cnt}, drop={drop_cnt}, slide={slide_cnt}, reclaim={reclaim_cnt}",
            traces.len()
        );
    }
    if !traces.is_empty() {
        write_bundle(&out_root.join(name), render_cache_scheduler_tla_bundle(&traces).unwrap());
    }
}

fn gen_sp(out_root: &Path, name: &str, strict: bool) {
    let mut pools = blocks::BlockPools::new(32);
    let mut tbl = blocks::BlockTable::new(32).unwrap();
    for b in 0..32u32 {
        let slot = blocks::Slot::new(b + 1).unwrap();
        let _ = pools.full.take(slot).unwrap();
        let _ = pools.window.take(slot).unwrap();
        tbl.fill(&mut pools.full, blocks::BlockIndex::new(b), slot, 1).unwrap();
        tbl.fill(&mut pools.window, blocks::BlockIndex::new(b), slot, 1).unwrap();
    }
    let mut max_win = 0usize;
    let mut trace_entries = Vec::new();
    for offset in 0..blocks::BLOCK {
        let pos = (blocks::WINDOW - 1) + offset;
        let pinned = tbl.pinned_span_entries(pos, 512).unwrap();
        let win_ids: Vec<String> = pinned
            .iter()
            .filter(|(k, _, _)| *k == blocks::BlockKind::Window)
            .map(|(_, bi, _)| bi.get().to_string())
            .collect();
        max_win = max_win.max(win_ids.len());
        trace_entries.push(format!(
            "  <<[offset |-> 0, selected |-> FALSE, pin |-> {{}}], [offset |-> {offset}, selected |-> TRUE, pin |-> {{{}}}]>>",
            win_ids.join(", ")
        ));
    }
    if max_win == 0 {
        eprintln!("COVERAGE GAP {name}: max_win=0");
        if strict {
            std::process::exit(1);
        }
    } else {
        eprintln!("{name}: {} alignments, max_win={max_win}", blocks::BLOCK);
    }
    let data = format!(
        "---- MODULE TraceData ----\nTraces == <<\n{}\n>>\n====\n",
        trace_entries.join(",\n")
    );
    let check = String::from(
        "---- MODULE TraceCheck ----\n\
         EXTENDS SpanPins, TraceData, Sequences\n\
         VARIABLES traceIdx, stepIdx\n\
         Apply(s) == offset = s.offset /\\ selected = s.selected /\\ pin = s.pin\n\
         ApplyNext(s) == offset' = s.offset /\\ selected' = s.selected /\\ pin' = s.pin\n\
         ReplayInit == traceIdx \\in 1..Len(Traces) /\\ stepIdx = 1 /\\ Apply(Traces[traceIdx][1])\n\
         ReplayNext == stepIdx < Len(Traces[traceIdx]) /\\ UNCHANGED traceIdx /\\ stepIdx' = stepIdx + 1 /\\ ApplyNext(Traces[traceIdx][stepIdx + 1])\n\
         ReplaySpec == ReplayInit /\\ [][ReplayNext]_<<vars, traceIdx, stepIdx>>\n\
         InitConforms == (stepIdx = 1) => Init\n\
         StepConforms == [][Next]_vars\n\
         ====\n",
    );
    let cfg = String::from(
        "SPECIFICATION ReplaySpec\n\
         CHECK_DEADLOCK FALSE\n\
         CONSTANTS\n\
          B = 64\n\
          W = 1024\n\
          C = 512\n\
          Bug = \"none\"\n\
         INVARIANTS\n\
          InitConforms\n\
          Covered\n\
          BoundHolds\n\
          MaximumIs25\n\
         PROPERTIES\n\
          StepConforms\n",
    );
    write_bundle(&out_root.join(name), (data, check, cfg));
}

fn gen_gl(out_root: &Path, name: &str, strict: bool) {
    // Replay every valid Capacity=4 `(lengths, shared, windowed)` walk from `blocks::walk_grouped_tiles`
    // against `specs/GroupedLayout.tla` (`t-later-prefix-traces` & `t-audit-pr53-port`).
    // Covers unequal positions, two distinct windows, internal tile boundaries inside 4-key splits,
    // and splits crossing the shared-to-private divergence boundary.
    let mut trace_entries = Vec::new();
    let mut shared_cross_split = 0usize;
    let mut two_windows = 0usize;
    for l1 in 1..=8usize {
        for l2 in 1..=8usize {
            let max_shared = l1.min(l2) / 2;
            for shared in 0..=max_shared {
                let mut resident = std::collections::BTreeSet::new();
                for b in 0..=((l1 - 1) / 2) {
                    resident.insert((b, 1usize));
                }
                for b in 0..=((l2 - 1) / 2) {
                    resident.insert((b, if b < shared { 1 } else { 2 }));
                }
                if resident.len() > 4 {
                    continue;
                }
                for windowed in [false, true] {
                    if shared == 1 && l1 >= 4 && l2 >= 4 {
                        shared_cross_split += 1;
                    }
                    if windowed && l1.saturating_sub(3) != l2.saturating_sub(3) {
                        two_windows += 1;
                    }
                    let states = blocks::walk_grouped_tiles([l1, l2], shared, windowed, 2, 2, 4, 3);
                    let fmt_slot = |(b, r): (usize, usize)| format!("<<{b}, {r}>>");
                    let rendered_states: Vec<String> = states
                        .iter()
                        .map(|st| {
                            let loaded_s = st
                                .loaded
                                .iter()
                                .copied()
                                .map(fmt_slot)
                                .collect::<Vec<_>>()
                                .join(", ");
                            let staged_s = st
                                .staged
                                .iter()
                                .map(|&(t, s)| format!("<<{t}, {}>>", fmt_slot(s)))
                                .collect::<Vec<_>>()
                                .join(", ");
                            let fmt_visits = |v: &[(usize, (usize, usize), usize)]| {
                                v.iter()
                                    .map(|&(p, s, sp)| format!("<<{p}, {}, {sp}>>", fmt_slot(s)))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            };
                            let fmt_partials = |p: &[(usize, usize)]| {
                                p.iter()
                                    .map(|&(sp, stop)| format!("<<{sp}, {stop}>>"))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            };
                            let win_tla = if st.windowed { "TRUE" } else { "FALSE" };
                            format!(
                                "[lengths |-> <<{}, {}>>, shared |-> {}, windowed |-> {}, tile |-> {}, loaded |-> {{{}}}, staged |-> <<{}>>, visits |-> <<<<{}>>, <<{}>>>>, partials |-> <<<<{}>>, <<{}>>>>]",
                                st.lengths[0],
                                st.lengths[1],
                                st.shared,
                                win_tla,
                                st.tile,
                                loaded_s,
                                staged_s,
                                fmt_visits(&st.visits[0]),
                                fmt_visits(&st.visits[1]),
                                fmt_partials(&st.partials[0]),
                                fmt_partials(&st.partials[1]),
                            )
                        })
                        .collect();
                    trace_entries.push(format!("  <<{}>>", rendered_states.join(", ")));
                }
            }
        }
    }
    if trace_entries.is_empty() || shared_cross_split == 0 || two_windows == 0 {
        eprintln!(
            "COVERAGE GAP {name}: traces={} shared_cross_split={shared_cross_split} two_windows={two_windows}",
            trace_entries.len()
        );
        if strict {
            std::process::exit(1);
        }
    } else {
        eprintln!(
            "{name}: {} walks, shared_cross_split={shared_cross_split}, two_windows={two_windows}",
            trace_entries.len()
        );
    }
    let data = format!(
        "---- MODULE TraceData ----\nTraces == <<\n{}\n>>\n====\n",
        trace_entries.join(",\n")
    );
    let check = String::from(
        "---- MODULE TraceCheck ----\n\
         EXTENDS GroupedLayout, TraceData, Sequences\n\
         VARIABLES traceIdx, stepIdx\n\
         Apply(s) ==\n\
           /\\ lengths = s.lengths\n\
           /\\ shared = s.shared\n\
           /\\ windowed = s.windowed\n\
           /\\ tile = s.tile\n\
           /\\ loaded = s.loaded\n\
           /\\ staged = s.staged\n\
           /\\ visits = s.visits\n\
           /\\ partials = s.partials\n\
         ApplyNext(s) ==\n\
           /\\ lengths' = s.lengths\n\
           /\\ shared' = s.shared\n\
           /\\ windowed' = s.windowed\n\
           /\\ tile' = s.tile\n\
           /\\ loaded' = s.loaded\n\
           /\\ staged' = s.staged\n\
           /\\ visits' = s.visits\n\
           /\\ partials' = s.partials\n\
         ReplayInit == traceIdx \\in 1..Len(Traces) /\\ stepIdx = 1 /\\ Apply(Traces[traceIdx][1])\n\
         ReplayNext == stepIdx < Len(Traces[traceIdx]) /\\ UNCHANGED traceIdx /\\ stepIdx' = stepIdx + 1 /\\ ApplyNext(Traces[traceIdx][stepIdx + 1])\n\
         ReplayDone == stepIdx = Len(Traces[traceIdx]) /\\ UNCHANGED <<vars, traceIdx, stepIdx>>\n\
         ReplayStep == ReplayNext \\/ ReplayDone\n\
         ReplaySpec == ReplayInit /\\ [][ReplayStep]_<<vars, traceIdx, stepIdx>>\n\
         InitConforms == (stepIdx = 1) => Init\n\
         StepConforms == [][Next]_vars\n\
         ====\n",
    );
    let cfg = String::from(
        "SPECIFICATION ReplaySpec\n\
         CHECK_DEADLOCK FALSE\n\
         CONSTANTS\n\
          Capacity = 4\n\
          Bug = \"none\"\n\
         INVARIANTS\n\
          InitConforms\n\
          StagedOnce\n\
          VisitedKeys\n\
          OrderedPartials\n\
         PROPERTIES\n\
          StepConforms\n",
    );
    write_bundle(&out_root.join(name), (data, check, cfg));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_root = Path::new(args.get(1).map(String::as_str).unwrap_or("."));
    let mode = args.get(2).map(String::as_str).unwrap_or("all");
    let strict = args.get(3).map(String::as_str) != Some("--mutant");
    match mode {
        "bl_base0" => gen_bl(out_root, "bl_base0", 0, 2, strict),
        "bl_high_fp16" => gen_bl(out_root, "bl_high_fp16", 2048, 2, strict),
        "bl_high_kv8" => gen_bl(out_root, "bl_high_kv8", 4096, 1, strict),
        "cs" => gen_cs(out_root, "cs", strict),
        "sp_512" => gen_sp(out_root, "sp_512", strict),
        "gl_4" => gen_gl(out_root, "gl_4", strict),
        "all" => {
            gen_bl(out_root, "bl_base0", 0, 2, strict);
            gen_bl(out_root, "bl_high_fp16", 2048, 2, strict);
            gen_bl(out_root, "bl_high_kv8", 4096, 1, strict);
            gen_cs(out_root, "cs", strict);
            gen_sp(out_root, "sp_512", strict);
            gen_gl(out_root, "gl_4", strict);
        }
        other => panic!("unknown cache-traces mode {other:?}"),
    }
}
