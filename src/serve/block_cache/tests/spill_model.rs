use super::super::*;
use super::ScratchDir;
use crate::engine::{Engine, Lane, Span};
use crate::prefix::PrefixMatch;
use crate::tokenizer::Tokenizer;
use objc2_metal::MTLCommandBuffer;

// Run a serving lane through the actual paged prefill kernels. Fixed-size chunks give
// the cold comparison the same floating-point reduction order at every shared position.
fn prefill(eng: &Engine, cache: &mut ServeBlockCache, tokens: &[u32]) -> (usize, Vec<u32>) {
    let lane = Lane::new_blocked(eng, eng.max_ctx);
    let run = eng.on(&lane);
    let mut state = LaneBlockState::default();
    let matched = cache.match_and_bind(
        &mut eng.block_pool().borrow_mut(), &mut lane.block_table().borrow_mut(), &mut state, tokens,
    );
    let start = match matched {
        Some((pos, cs)) => { run.set_cs(&cs); pos }
        None => { run.reset(); 0 }
    };
    run.set_tokens(0, tokens);
    let mut pos = start;
    while pos < tokens.len() {
        let target = effective_pf_target(pos, tokens.len());
        let count = (target - pos).min(2 * BLOCK).min(eng.chunk);
        cache.ensure_span_blocks(
            &mut eng.block_pool().borrow_mut(), &mut lane.block_table().borrow_mut(), pos, count,
        ).unwrap();
        let cb = eng.gpu.cmd();
        eng.encode_spans(&cb, &[Span { lane: &lane, p0: pos, t: count }], pos + count == tokens.len(), false);
        cb.commit();
        crate::serve::wait_committed_step(&cb, "spill model regression").unwrap();
        cache.finish_prefill_span(
            &mut eng.block_pool().borrow_mut(), &mut lane.block_table().borrow_mut(),
            &mut state, tokens, pos, count, || run.get_cs(),
        );
        pos += count;
    }
    let row: Vec<_> = eng.logits().iter().map(|x| {
        assert!(x.is_finite(), "model produced nonfinite logits at prompt length {}", tokens.len());
        x.to_bits()
    }).collect();
    cache.release_lane(&mut eng.block_pool().borrow_mut(), &mut lane.block_table().borrow_mut(), &mut state);
    (start, row)
}

fn assert_same_logits(cached: &[u32], cold: &[u32]) {
    assert!(!cold.is_empty());
    assert_eq!(cached.len(), cold.len());
    for (token, (a, b)) in cached.iter().zip(cold).enumerate() {
        assert_eq!(a, b, "different logits at vocabulary token {token}: {} vs {}", f32::from_bits(*a), f32::from_bits(*b));
    }
}

#[test]
#[ignore = "requires an M5 Metal device and local Inkling-2-Mini GGUF"]
fn restored_long_prefix_and_early_fork_match_cold_logits() {
    let path = std::env::var("INKY_TEST_MODEL").unwrap_or_else(|_| crate::default_model());
    let eng = Engine::load(&path, 2048);
    eng.ensure_batch(2 * BLOCK);
    let kvb = eng.block_pool().borrow().full.kvb();
    *eng.block_pool().borrow_mut() = BlockPools::gpu(&eng.gpu.dev, 20, 20, kvb);
    let tokenizer = Tokenizer::load(&eng.g);
    let base = tokenizer.encode(" A blue moon rises above the river.", false);
    let other = tokenizer.encode("Bitter winter winds shake these walls.", false);
    assert!(!base.is_empty() && !other.is_empty() && base[0] != other[0]);
    let boundary = WINDOW + BLOCK;
    let long: Vec<u32> = base.iter().copied().cycle().take(boundary + 1).collect();
    let filler: Vec<u32> = other.iter().copied().cycle().take(6 * BLOCK + 1).collect();
    let mut fork = long[..BLOCK].to_vec();
    fork.extend(other.iter().copied().cycle().take(BLOCK + 17));
    assert_ne!(fork[BLOCK], long[BLOCK]);

    let dir = ScratchDir::new("m5-restore-fork");
    let mut cache = ServeBlockCache::with_spill(SpillStore::open(&dir.0, 512 * 1024 * 1024).unwrap());
    let (first, cold_long) = prefill(&eng, &mut cache, &long);
    assert_eq!(first, 0);
    assert_eq!(cache.peek_match_len(&eng.block_pool().borrow(), &long), boundary);

    // The disjoint prompt exceeds the remaining Full slots, forcing the LRU start to disk.
    assert_eq!(prefill(&eng, &mut cache, &filler).0, 0);
    assert_eq!(cache.peek_match_len(&eng.block_pool().borrow(), &long), 0);
    assert!(cache.stats().starts_spilled > 0);
    let writes = cache.spill_store().unwrap().stats().writes_committed;
    assert!(writes > 0);
    let reads_before = cache.spill_store().unwrap().stats().reads_hit;
    let restores_before = cache.stats().starts_restored;
    let (start, restored) = prefill(&eng, &mut cache, &long);
    assert_eq!(start, boundary);
    assert_eq!(cache.stats().starts_restored, restores_before + 1);
    let reads = cache.spill_store().unwrap().stats().reads_hit - reads_before;
    assert!(reads >= (boundary / BLOCK + WINDOW / BLOCK + 1) as u64, "expected all Full and Window blocks and recurrent state from SSD");
    assert_same_logits(&restored, &cold_long);

    let root = cache.tree.deepest_start(&long).ids[0];
    assert!(cache.tree.handle(root).unwrap().window().is_none());
    let dedups = cache.stats().shared_dedups;
    let (start, warm_fork) = prefill(&eng, &mut cache, &fork);
    assert_eq!(start, 0);
    assert!(cache.stats().shared_dedups > dedups);
    assert!(cache.tree.handle(root).unwrap().window().is_some());
    cache.shutdown(&mut eng.block_pool().borrow_mut());

    // This fresh cache has no retained handles or spill files for the independent branch run.
    let mut cold = ServeBlockCache::new();
    let (start, cold_fork) = prefill(&eng, &mut cold, &fork);
    assert_eq!(start, 0);
    assert_same_logits(&warm_fork, &cold_fork);
    cold.shutdown(&mut eng.block_pool().borrow_mut());
    super::assert_zero_pool_leaks(&eng.block_pool().borrow());
    eprintln!("SSD restore at {boundary} tokens: {writes} committed block writes, {reads} verified reads, {} logits matched for both restored prefix and early fork", cold_fork.len());
}
