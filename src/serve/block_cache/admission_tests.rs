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
            last_used_tick: 0,
        },
    );
    assert!(!cache.can_admit(&pools, &active, 1, &prompt, 1, 2048));
}
