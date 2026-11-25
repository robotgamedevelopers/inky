use super::*;

fn prefill_to(
    cache: &mut ServeBlockCache,
    pools: &mut BlockPools,
    prompt: &[u32],
    end: usize,
) -> (BlockTable, LaneBlockState) {
    let mut table = BlockTable::new(256).unwrap();
    let mut state = LaneBlockState::default();
    assert!(cache
        .match_and_bind(pools, &mut table, &mut state, prompt)
        .is_none());
    cache.ensure_span_blocks(pools, &mut table, 0, end).unwrap();
    cache.finish_prefill_span(pools, &mut table, &mut state, prompt, 0, end, || {
        vec![0; 16]
    });
    (table, state)
}

fn lane<'a>(
    id: usize,
    prompt: &'a [u32],
    written: usize,
    max_gen: usize,
) -> InflightLaneReserve<'a> {
    InflightLaneReserve {
        lane: id,
        toks: prompt,
        block_hashes: None,
        n_past: written,
        pf_end: prompt.len(),
        cached: 0,
        written,
        gen: 0,
        max_gen,
        runnable: true,
    }
}

#[test]
fn matching_retained_full_blocks_reduces_admission_headroom() {
    let mut pools = BlockPools::with_capacities(24, 128, 2).unwrap();
    let mut cache = ServeBlockCache::new();
    let prior = vec![1; 12 * BLOCK + 1];
    let (mut table, mut state) = prefill_to(&mut cache, &mut pools, &prior, 12 * BLOCK);
    cache.release_lane(&mut pools, &mut table, &mut state);

    let prompt_a = vec![2; 12 * BLOCK];
    let (mut table_a, mut state_a) = prefill_to(&mut cache, &mut pools, &prompt_a, 8 * BLOCK);
    let prompt_b = vec![1; 16 * BLOCK];
    let active = [lane(0, &prompt_a, 8 * BLOCK, 2 * BLOCK)];
    assert_eq!(cache.peek_match_len(&pools, &prompt_b), 12 * BLOCK);
    assert_eq!(cache.free_plus_evictable_slots(&pools, BlockKind::Full), 16);
    // B binds 12 otherwise evictable slots. Its four available slots cannot
    // cover both lanes' remaining prompts and two blocks of decode per lane.
    assert!(!cache.can_admit(&pools, &active, 1, &prompt_b, 2 * BLOCK, 256));
    cache.release_lane(&mut pools, &mut table_a, &mut state_a);
    assert!(cache.can_admit(&pools, &[], 1, &prompt_b, 2 * BLOCK, 256));
}

#[test]
fn matching_retained_window_blocks_reduces_admission_headroom() {
    let mut pools = BlockPools::with_capacities(256, 40, 2).unwrap();
    let mut cache = ServeBlockCache::new();
    let prior = vec![3; 16 * BLOCK + 1];
    let (mut table, mut state) = prefill_to(&mut cache, &mut pools, &prior, 16 * BLOCK);
    cache.release_lane(&mut pools, &mut table, &mut state);

    let prompt_a = vec![4; 16 * BLOCK];
    let (mut table_a, mut state_a) = prefill_to(&mut cache, &mut pools, &prompt_a, 8 * BLOCK);
    let prompt_b = vec![3; 28 * BLOCK + 1];
    let active = [lane(0, &prompt_a, 8 * BLOCK, 1)];
    assert_eq!(cache.peek_match_len(&pools, &prompt_b), 16 * BLOCK);
    assert_eq!(
        cache.free_plus_evictable_slots(&pools, BlockKind::Window),
        32
    );
    // The next B span needs 12 new window blocks while its 16 history blocks
    // are pinned. A can still grow by eight window blocks.
    assert!(!cache.can_admit(&pools, &active, 1, &prompt_b, 1, 12 * BLOCK));
    cache.release_lane(&mut pools, &mut table_a, &mut state_a);
}

#[test]
fn binding_blocks_already_pinned_by_an_active_lane_costs_no_full_slots() {
    let mut pools = BlockPools::with_capacities(32, 128, 2).unwrap();
    let mut cache = ServeBlockCache::new();
    let prior = vec![5; 24 * BLOCK + 1];
    let (mut table, mut state) = prefill_to(&mut cache, &mut pools, &prior, 24 * BLOCK);
    cache.release_lane(&mut pools, &mut table, &mut state);
    let prompt = vec![5; 24 * BLOCK + 3];
    let (matched, _) = cache
        .match_and_bind(&mut pools, &mut table, &mut state, &prompt)
        .unwrap();
    let active = [InflightLaneReserve {
        cached: matched,
        ..lane(0, &prompt, matched, 1)
    }];

    assert_eq!(matched, 24 * BLOCK);
    assert_eq!(cache.free_plus_evictable_slots(&pools, BlockKind::Full), 8);
    assert!(cache.can_admit(&pools, &active, 1, &prompt, 1, BLOCK));
    cache.release_lane(&mut pools, &mut table, &mut state);
}

#[test]
fn followers_cannot_both_charge_the_same_future_checkpoint_to_each_other() {
    let mut a = vec![1; 2048];
    a.push(2);
    let mut b = vec![1; 8192];
    b.push(3);
    let mut c = vec![1; 8192];
    c.push(4);
    let active = [lane(0, &a, 0, 1), lane(1, &b, 0, 1), lane(2, &c, 0, 1)];
    let reserved = reserved_inflight_full_slots(&active);
    assert!(
        reserved >= b.len().div_ceil(BLOCK),
        "B alone needs 129 Full slots, reserved {reserved}"
    );

    let pools = BlockPools::with_capacities(180, 512, 2).unwrap();
    let mut cache = ServeBlockCache::new();
    let unrelated = vec![7; 80 * BLOCK];
    assert!(!cache.can_admit(&pools, &active, 3, &unrelated, 1, 2048));
}

#[test]
fn an_ssd_restore_cannot_spend_the_slots_reserved_for_a_cold_producer() {
    let pools = BlockPools::with_capacities(160, 512, 2).unwrap();
    let mut cache = ServeBlockCache::new();
    let prompt = vec![7; 8193];
    let active = [lane(0, &prompt, 0, 1)];

    // As a cold follower, the new lane can wait for lane 0's 8192-token checkpoint.
    assert!(cache.can_admit(&pools, &active, 1, &prompt, 1, 2048));

    // The same request could bind a deeper SSD checkpoint immediately. Reserve the
    // cost of that restore before scheduling either lane; no disk I/O is needed here.
    let spill_id = 1;
    let mut tip = None;
    for (i, tokens) in prompt.chunks_exact(BLOCK).enumerate() {
        tip = Some(
            cache
                .spilled_tree
                .insert(tip, i * BLOCK, tokens, spill_id)
                .unwrap(),
        );
    }
    cache.spilled_tree.retain_start(tip.unwrap()).unwrap();
    cache.spilled_starts.insert(
        spill_id,
        SpilledStart {
            pos: 8192,
            reasons: EVERY,
            cs_key: SpillKey::new(SpillBlockKind::Full, 1),
            entries: Vec::new(),
            toks: prompt[..8192].to_vec(),
            last_used_tick: 0,
        },
    );
    assert!(!cache.can_admit(&pools, &active, 1, &prompt, 1, 2048));
}

#[test]
fn small_window_pool_keeps_enough_history_for_each_planned_prefill_and_decode() {
    use crate::scheduler::{ServeLaneInput, ServeStepPlan, ServeStepPlanner};
    use crate::serve::{pooled_context, window_span_end};

    for (windows, prompt_len, generated) in
        [(1, 60, 5), (16, 1024, 1), (17, 2311, 138), (20, 3073, 81)]
    {
        assert!(pooled_context(8192, 128, windows) >= prompt_len + generated + 2);
        for guided in [false, true] {
            let mut pools = BlockPools::with_capacities(128, windows, 1).unwrap();
            let mut cache = ServeBlockCache::new();
            let mut table = BlockTable::new(128).unwrap();
            let mut state = LaneBlockState::default();
            let prompt = vec![1; prompt_len];
            cache.match_and_bind(&mut pools, &mut table, &mut state, &prompt);
            let mut planner = ServeStepPlanner::new(1, 16);
            planner.on_admit(0);
            let final_prefill = prompt_len - usize::from(guided);
            let mut pos = 0;
            while pos < prompt_len + generated - 1 {
                let input = ServeLaneInput {
                    runnable: true,
                    n_past: pos,
                    pf_end: effective_pf_target(pos, prompt_len)
                        .min(final_prefill)
                        .min(window_span_end(pos, windows)),
                    has_pending_decode: pos >= final_prefill,
                };
                match planner
                    .plan_and_begin_step(&[input], 4096, BLOCK)
                    .expect("single lane can progress")
                {
                    ServeStepPlan::Prefill(spans) => {
                        let (_, start, count) = spans[0];
                        assert_eq!(start, pos);
                        cache
                            .ensure_span_blocks(&mut pools, &mut table, pos, count)
                            .unwrap();
                        cache.finish_prefill_span(
                            &mut pools,
                            &mut table,
                            &mut state,
                            &prompt,
                            pos,
                            count,
                            || vec![0; 16],
                        );
                        pos += count;
                    }
                    ServeStepPlan::Decode(_) => {
                        cache
                            .ensure_span_blocks(&mut pools, &mut table, pos, 1)
                            .unwrap();
                        cache.finish_decode_span(&mut pools, &mut table, &mut state, pos);
                        pos += 1;
                    }
                }
                planner.complete_step();
            }
            cache.release_lane(&mut pools, &mut table, &mut state);
            cache.shutdown(&mut pools);
        }
    }
}

#[test]
fn long_reply_maintains_a_rolling_decode_reservation_after_4096_tokens() {
    let prompt = [42];
    let generated = 8192;
    let active = InflightLaneReserve {
        gen: generated,
        ..lane(0, &prompt, generated, generated + 2 * BLOCK)
    };
    assert_eq!(reserved_inflight_full_slots(&[active]), 2);
}

fn reference_all_pairs_reserved_full_slots(lanes: &[InflightLaneReserve<'_>]) -> usize {
    let prefills: Vec<InflightPrefillLane<'_>> = lanes
        .iter()
        .filter_map(|l| l.as_prefill_lane())
        .collect();
    lanes
        .iter()
        .map(|lane| {
            let credited = prefills
                .iter()
                .find(|f| f.lane == lane.lane && f.eligible_follower)
                .map_or(0, |me| {
                    let next = effective_pf_target(me.n_past, me.pf_end);
                    let final_ckpt = me.pf_end.saturating_sub(1) / BLOCK * BLOCK;
                    prefills
                        .iter()
                        .map(|prod| pair_shared_checkpoint_exact(me, prod, next, final_ckpt))
                        .max()
                        .unwrap_or(0)
                });
            lane.reserved_end().div_ceil(BLOCK) - lane.written.max(credited).div_ceil(BLOCK)
        })
        .sum()
}

#[test]
fn burst_32_concurrent_65k_prompts_admits_in_under_2ms_with_exact_reference_parity() {
    const N_LANES: usize = 32;
    const PROMPT_LEN: usize = 65_536;

    // Construct 32 realistic 65,536-token prompts across 4 prefix families:
    // - 0..16: share 57,344 tokens, diverge in the tail (and lanes 0..4 share all 65,472 checkpoint tokens)
    // - 16..24: share 24,576 tokens with family A, then share a second branch up to 49,152
    // - 24..28: share 8,192 tokens only
    // - 28..32: completely cold / distinct prefixes
    let base_a: Vec<u32> = (0..PROMPT_LEN as u32).map(|i| (i.wrapping_mul(2654435761)) % 60_000).collect();
    let mut prompts: Vec<Vec<u32>> = Vec::with_capacity(N_LANES);
    for i in 0..N_LANES {
        let mut p = base_a.clone();
        match i {
            0..=3 => {
                // Identical up to final_checkpoint = 65,472; differ only in final 64-token tail!
                p[PROMPT_LEN - 1] = 70_000 + i as u32;
            }
            4..=15 => {
                for (k, slot) in p[58_000..].iter_mut().enumerate() {
                    *slot = 80_000 + (i as u32) * 100 + (k as u32 % 97);
                }
            }
            16..=23 => {
                for (k, slot) in p[24_576..49_152].iter_mut().enumerate() {
                    *slot = 90_000 + (k as u32 % 251);
                }
                for (k, slot) in p[49_152..].iter_mut().enumerate() {
                    *slot = 100_000 + (i as u32) * 100 + (k as u32 % 89);
                }
            }
            24..=27 => {
                for (k, slot) in p[8_192..].iter_mut().enumerate() {
                    *slot = 110_000 + (i as u32) * 100 + (k as u32 % 113);
                }
            }
            _ => {
                for (k, slot) in p.iter_mut().enumerate() {
                    *slot = 120_000 + (i as u32) * 1000 + (k as u32 % 509);
                }
            }
        }
        prompts.push(p);
    }

    // Precompute compact per-block rolling hashes once per prompt (as Job preparation does).
    let hashes: Vec<Vec<u64>> = prompts.iter().map(|p| compute_block_hashes(p)).collect();
    for h in &hashes {
        assert_eq!(h.len(), PROMPT_LEN / BLOCK);
    }

    // Metadata-only records pool avoids multi-GiB slab allocation while exercising exact pool_headroom!
    let pools = BlockPools {
        window: crate::blocks::BlockPool::records_only(BlockKind::Window, 2_048, 1).unwrap(),
        full: crate::blocks::BlockPool::records_only(BlockKind::Full, 12_000, 1).unwrap(),
    };
    let mut cache = ServeBlockCache::new();
    let max_gen = 64usize;
    let chunk = 4096usize;

    // Time the full 32-lane admission + leader-election burst!
    let began = std::time::Instant::now();
    let mut cohort = PreparedInflightCohort::empty();
    let mut decisions = Vec::with_capacity(N_LANES);
    let mut reserves_so_far: Vec<InflightLaneReserve<'_>> = Vec::with_capacity(N_LANES);
    for i in 0..N_LANES {
        let ok = cache.can_admit_with_cohort(
            &pools,
            &cohort,
            i,
            &prompts[i],
            Some(&hashes[i]),
            max_gen,
            chunk,
        );
        decisions.push(ok);
        if ok {
            let r = InflightLaneReserve {
                lane: i,
                toks: &prompts[i],
                block_hashes: Some(&hashes[i]),
                n_past: 0,
                pf_end: PROMPT_LEN,
                cached: 0,
                written: 0,
                gen: 0,
                max_gen,
                runnable: true,
            };
            cohort.push_admitted(r);
            reserves_so_far.push(r);
        }
    }
    let pf_lanes: Vec<InflightPrefillLane<'_>> = reserves_so_far
        .iter()
        .filter_map(|r| r.as_prefill_lane())
        .collect();
    let leaders = find_prefill_leaders(&pf_lanes);
    let elapsed = began.elapsed();

    eprintln!(
        "[u331-admit-scan] 32 x 65,536-token burst admission + leader election completed in {elapsed:?}"
    );
    assert!(
        elapsed < std::time::Duration::from_millis(2),
        "32 x 65,536-token burst admission + leader election took {elapsed:?} (must be < 2ms)"
    );

    // Verify 100% exact parity at every admission step and leader election against reference all-pairs logic!
    let mut ref_cache = ServeBlockCache::new();
    let mut ref_active: Vec<InflightLaneReserve<'_>> = Vec::new();
    for i in 0..N_LANES {
        let mut trial = ref_active.clone();
        trial.push(InflightLaneReserve {
            lane: i,
            toks: &prompts[i],
            block_hashes: None,
            n_past: 0,
            pf_end: PROMPT_LEN,
            cached: 0,
            written: 0,
            gen: 0,
            max_gen,
            runnable: true,
        });
        assert_eq!(
            reserved_inflight_full_slots(&trial),
            reference_all_pairs_reserved_full_slots(&trial),
            "reserved_inflight_full_slots mismatch at lane {i}"
        );
        let ref_ok = ref_cache.can_admit(&pools, &ref_active, i, &prompts[i], max_gen, chunk);
        assert_eq!(decisions[i], ref_ok, "can_admit decision mismatch at lane {i}");
        if ref_ok {
            ref_active.push(trial.pop().unwrap());
        }
    }
    for (idx, pf) in pf_lanes.iter().enumerate() {
        let no_hash_pf: Vec<InflightPrefillLane<'_>> = pf_lanes
            .iter()
            .map(|p| InflightPrefillLane { block_hashes: None, ..*p })
            .collect();
        let ref_leader = find_prefill_leader(&no_hash_pf, idx);
        assert_eq!(leaders[idx], ref_leader, "leader mismatch for prefill idx {idx} (lane {})", pf.lane);
    }
}
