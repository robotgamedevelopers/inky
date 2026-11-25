use super::*;
use crate::blocks::{FULL_STRIDE, WINDOW_STRIDE};
use crate::cache_trace::{
    parse_block_layout_jsonl, parse_cache_scheduler_jsonl, JsonlSink, MemorySink, NullSink,
};

#[test]
fn working_set_budget_handles_dynamic_kvb_and_refuses_zero_or_overflow() {
    let g = 1u64 << 30;
    let fp16 = WorkingSetBudget::from_recommended_working_set_bytes(64 * g, 20 * g, 2 * g, 2).unwrap();
    assert_eq!(fp16.cache_budget_bytes, 42 * g);
    assert_eq!(fp16.full_stride_bytes, FULL_STRIDE);
    assert_eq!(fp16.window_stride_bytes, WINDOW_STRIDE);
    assert_eq!(
        fp16.worst_case_request_bytes(4096, 512).unwrap(),
        64 * FULL_STRIDE + 25 * WINDOW_STRIDE
    );

    // --kv8 (`kvb = 1`) uses half the stride per block:
    let kv8 = WorkingSetBudget::from_recommended_working_set_bytes(64 * g, 20 * g, 2 * g, 1).unwrap();
    assert_eq!(kv8.full_stride_bytes, FULL_STRIDE / 2);
    assert_eq!(kv8.window_stride_bytes, WINDOW_STRIDE / 2);
    assert_eq!(
        kv8.worst_case_request_bytes(4096, 512).unwrap(),
        64 * (FULL_STRIDE / 2) + 25 * (WINDOW_STRIDE / 2)
    );
    assert!(kv8.partition_pools(256).unwrap().0 > fp16.partition_pools(256).unwrap().0);

    // Invalid kvb, underflow, overflow, and zero inputs are all refused cleanly (never panic or wrap):
    assert!(WorkingSetBudget::from_recommended_working_set_bytes(64 * g, 0, 0, 0).is_err());
    assert!(WorkingSetBudget::from_recommended_working_set_bytes(64 * g, 0, 0, 3).is_err());
    assert!(WorkingSetBudget::from_recommended_working_set_bytes(u64::MAX, u64::MAX, 1, 2).is_err());
    assert!(WorkingSetBudget::from_recommended_working_set_bytes(100, 200, 0, 2).is_err());
    assert!(WorkingSetBudget::from_recommended_working_set_bytes(12 * 1024 * 1024 - 1, 0, 0, 2).is_err());
    assert!(WorkingSetBudget::from_recommended_working_set_bytes(6 * 1024 * 1024, 0, 0, 1).is_ok());

    assert!(fp16.worst_case_request_bytes(0, 512).is_err());
    assert!(fp16.worst_case_request_bytes(4096, 0).is_err());
    assert!(fp16.worst_case_request_bytes(1 << 20, u32::MAX).is_err());
    assert!(fp16.partition_pools(0).is_err());
    assert!(span_bound_blocks(0, 64, 1024).is_err());
    assert!(span_bound_blocks(1, 0, 1024).is_err());
    assert!(span_bound_blocks(1, 64, 0).is_err());
    assert!(span_bound_blocks(u32::MAX, 64, 1024).is_err());
}

#[test]
fn block_allocator_reissues_generations_exhausts_max_gen_and_uses_real_byte_offset() {
    for (kvb, slot_base, want_hi) in [(2u32, 0u32, 0u32), (2u32, 2048u32, 1u32), (1u32, 4096u32, 1u32)] {
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

        // Explicit reuse & MaxGeneration exhaustion verification:
        let mut ev = NullSink;
        let mut st = MemorySink::new();
        let mut alloc = TrackedBlockAllocator::new(cfg.clone(), &mut ev, &mut st).unwrap();
        alloc.prepare("a", 2, &mut ev, &mut st).unwrap();
        let e_f1 = alloc.alloc_lowest_free("a", BlockKind::Full, 0, &mut ev, &mut st).unwrap();
        let e_w1 = alloc.alloc_lowest_free("a", BlockKind::Window, 0, &mut ev, &mut st).unwrap();
        assert_eq!((e_f1.slot().get(), e_f1.generation()), (1, 1));
        assert_eq!((e_w1.slot().get(), e_w1.generation()), (1, 1));
        assert_eq!(
            e_f1.offset(),
            byte_offset(BlockKind::Full, Slot::new(slot_base + 1).unwrap(), kvb).unwrap()
        );
        assert_eq!(OffsetLimbs::from_u64(e_f1.offset()).unwrap().hi, want_hi);
        alloc.begin_step("a", &mut ev, &mut st).unwrap();
        alloc.write_one("a", &mut ev, &mut st).unwrap();
        alloc.write_one("a", &mut ev, &mut st).unwrap();
        alloc.end_step("a", &mut ev, &mut st).unwrap();
        alloc.finish("a", &mut ev, &mut st).unwrap();

        // Lane "b" reuses slot 1 at generation 2 (via SlotRecord::reissue):
        alloc.prepare("b", 1, &mut ev, &mut st).unwrap();
        let e_f2 = alloc.alloc_lowest_free("b", BlockKind::Full, 0, &mut ev, &mut st).unwrap();
        let e_w2 = alloc.alloc_lowest_free("b", BlockKind::Window, 0, &mut ev, &mut st).unwrap();
        assert_eq!((e_f2.slot().get(), e_f2.generation()), (1, 2));
        assert_eq!((e_w2.slot().get(), e_w2.generation()), (1, 2));
        alloc.begin_step("b", &mut ev, &mut st).unwrap();
        alloc.write_one("b", &mut ev, &mut st).unwrap();
        alloc.end_step("b", &mut ev, &mut st).unwrap();
        alloc.finish("b", &mut ev, &mut st).unwrap();

        // Slot 1 is now at MaxGeneration (2) -> next available slot MUST be slot 2!
        assert_eq!(alloc.first_available_slot(BlockKind::Full), Some(Slot::new(2).unwrap()));
        assert_eq!(alloc.first_available_slot(BlockKind::Window), Some(Slot::new(2).unwrap()));

        // Seeded walks must exercise generation 2, start points, and window sliding:
        let mut max_gen_seen = 0;
        let mut saw_start = false;
        let mut saw_two_slots = false;
        for seed in 0..16 {
            let mut ev_s = MemorySink::new();
            let mut st_s = MemorySink::new();
            walk_block_layout(cfg.clone(), seed, &mut ev_s, &mut st_s).unwrap();
            let parsed = parse_block_layout_jsonl(&st_s.to_jsonl()).unwrap();
            for step in &parsed.steps {
                if step.state.start_kept {
                    saw_start = true;
                }
                for p in step.state.full_pool.iter().chain(&step.state.window_pool) {
                    max_gen_seen = max_gen_seen.max(p.generation);
                    if p.slot == 2 && p.generation >= 1 {
                        saw_two_slots = true;
                    }
                }
            }
        }
        assert_eq!(max_gen_seen, 2);
        assert!(saw_start);
        assert!(saw_two_slots);
    }
}

#[test]
fn cache_scheduler_admit_bypass_step_bypass_and_cold_spill_restore_hold() {
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
    let mut saw_admit_bypass = false;
    let mut saw_step_bypass = false;
    let mut saw_cancel = false;
    let mut saw_refuse = false;
    for seed in 0..16 {
        let mut ev_sink = MemorySink::new();
        let mut st_sink = MemorySink::new();
        walk_cache_scheduler(cfg.clone(), seed, &mut ev_sink, &mut st_sink).unwrap();
        let parsed = parse_cache_scheduler_jsonl(&st_sink.to_jsonl()).unwrap();
        for w in parsed.steps.windows(2) {
            if w[1].action == "Admit" {
                for (l0, l1) in w[0].state.lanes.iter().zip(&w[1].state.lanes) {
                    if l0.state == "Queued" && l1.state == "Queued" && l1.wait_ticks > l0.wait_ticks {
                        saw_admit_bypass = true;
                    }
                }
            }
            if w[1].action == "BuildStep" {
                for (l0, l1) in w[0].state.lanes.iter().zip(&w[1].state.lanes) {
                    if l1.wait_ticks > l0.wait_ticks {
                        saw_step_bypass = true;
                    }
                }
            }
        }
        for line in ev_sink.lines() {
            if line.contains("\"event\":\"cancel\"") {
                saw_cancel = true;
            }
            if line.contains("\"event\":\"refuse429\"") {
                saw_refuse = true;
            }
        }
    }
    assert!(saw_admit_bypass, "walks must exercise admit queue bypass at MaxQueue=2");
    assert!(saw_step_bypass, "walks must exercise BuildStep StepBypassed wait_ticks increment");
    assert!(saw_cancel, "walks must exercise mid-step cancellation");
    assert!(saw_refuse, "walks must exercise HTTP 429 refusal");

    // Explicit cold SSD spill + atomic restore scenario:
    let mut ev_sink = MemorySink::new();
    let mut st_sink = MemorySink::new();
    let mut sched = CacheScheduler::new(cfg, &mut ev_sink, &mut st_sink).unwrap();
    sched.arrive("a", "p1", 2, &mut ev_sink, &mut st_sink).unwrap();
    sched.admit("a", &mut ev_sink, &mut st_sink).unwrap();
    sched.begin_restore("a", &mut ev_sink, &mut st_sink).unwrap();
    sched.finish_restore("a", &mut ev_sink, &mut st_sink).unwrap();
    sched.build_step(&[("a".into(), 1)], &mut ev_sink, &mut st_sink).unwrap();
    sched.gpu_execute(&mut ev_sink, &mut st_sink).unwrap();
    sched.finish_step(&mut ev_sink, &mut st_sink).unwrap();
    sched.build_step(&[("a".into(), 1)], &mut ev_sink, &mut st_sink).unwrap();
    sched.gpu_execute(&mut ev_sink, &mut st_sink).unwrap();
    sched.finish_step(&mut ev_sink, &mut st_sink).unwrap();
    sched.finish_ready("a", &mut ev_sink, &mut st_sink).unwrap();

    sched.arrive("a", "p2", 2, &mut ev_sink, &mut st_sink).unwrap();
    sched.admit("a", &mut ev_sink, &mut st_sink).unwrap();
    sched.begin_restore("a", &mut ev_sink, &mut st_sink).unwrap();
    sched.finish_restore("a", &mut ev_sink, &mut st_sink).unwrap();
    sched.arrive("b", "p2", 2, &mut ev_sink, &mut st_sink).unwrap();
    sched.admit("b", &mut ev_sink, &mut st_sink).unwrap();
    sched.begin_restore("b", &mut ev_sink, &mut st_sink).unwrap();
    sched.finish_restore("b", &mut ev_sink, &mut st_sink).unwrap();
    assert!(sched.mem_pressure());

    let spilled1 = sched.start_cold_writeback(&mut ev_sink, &mut st_sink).unwrap();
    sched.complete_step_io(&mut ev_sink, &mut st_sink).unwrap();
    let spilled2 = sched.start_cold_writeback(&mut ev_sink, &mut st_sink).unwrap();
    sched.complete_step_io(&mut ev_sink, &mut st_sink).unwrap();
    assert_eq!(sched.disk_used(), 2);
    assert_eq!(sched.mem_used(), 0);

    sched.cancel("a", &mut ev_sink, &mut st_sink).unwrap();
    sched.cancel("b", &mut ev_sink, &mut st_sink).unwrap();
    sched.arrive("c", "p1", 2, &mut ev_sink, &mut st_sink).unwrap();
    sched.admit("c", &mut ev_sink, &mut st_sink).unwrap();
    let (matched_pos, reserved) = sched.begin_restore("c", &mut ev_sink, &mut st_sink).unwrap();
    assert_eq!((matched_pos, reserved), (2, 2));
    sched.restore_block("c", &spilled1, &mut ev_sink, &mut st_sink).unwrap();
    sched.restore_block("c", &spilled2, &mut ev_sink, &mut st_sink).unwrap();
    sched.finish_restore("c", &mut ev_sink, &mut st_sink).unwrap();
    sched.finish_ready("c", &mut ev_sink, &mut st_sink).unwrap();
}

#[test]
#[ignore = "writes versioned BlockLayout NDJSON trace to INKY_BLOCK_TRACE_OUT"]
fn dump_block_layout() {
    let path = std::env::var_os("INKY_BLOCK_TRACE_OUT").expect("set INKY_BLOCK_TRACE_OUT");
    let seed: u64 = std::env::var("INKY_BLOCK_TRACE_SEED")
        .map(|s| s.parse().expect("INKY_BLOCK_TRACE_SEED must be u64"))
        .unwrap_or(1);
    let slot_base: u32 = std::env::var("INKY_BLOCK_TRACE_SLOT_BASE")
        .map(|s| s.parse().expect("INKY_BLOCK_TRACE_SLOT_BASE must be u32"))
        .unwrap_or(0);
    let kvb: u32 = std::env::var("INKY_BLOCK_TRACE_KVB")
        .map(|s| s.parse().expect("INKY_BLOCK_TRACE_KVB must be 1 or 2"))
        .unwrap_or(2);
    let mode = std::env::var("INKY_BLOCK_TRACE_MODE").unwrap_or_else(|_| "events".into());
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
    let file = std::fs::File::create(path).unwrap();
    let mut writer = JsonlSink::new(std::io::BufWriter::new(file));
    let mut null = NullSink;
    if mode == "steps" {
        walk_block_layout(cfg, seed, &mut null, &mut writer).unwrap();
    } else {
        walk_block_layout(cfg, seed, &mut writer, &mut null).unwrap();
    }
    writer.flush().unwrap();
}

#[test]
#[ignore = "writes versioned CacheScheduler NDJSON trace to INKY_CACHE_TRACE_OUT"]
fn dump_cache_scheduler() {
    let path = std::env::var_os("INKY_CACHE_TRACE_OUT").expect("set INKY_CACHE_TRACE_OUT");
    let seed: u64 = std::env::var("INKY_CACHE_TRACE_SEED")
        .map(|s| s.parse().expect("INKY_CACHE_TRACE_SEED must be u64"))
        .unwrap_or(1);
    let mode = std::env::var("INKY_CACHE_TRACE_MODE").unwrap_or_else(|_| "events".into());
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
    let file = std::fs::File::create(path).unwrap();
    let mut writer = JsonlSink::new(std::io::BufWriter::new(file));
    let mut null = NullSink;
    if mode == "steps" {
        walk_cache_scheduler(cfg, seed, &mut null, &mut writer).unwrap();
    } else {
        walk_cache_scheduler(cfg, seed, &mut writer, &mut null).unwrap();
    }
    writer.flush().unwrap();
}
