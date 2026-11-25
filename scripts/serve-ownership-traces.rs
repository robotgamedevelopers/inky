// Standalone trace generator driving the ACTUAL production `ServeBlockCache`
// (`src/serve/block_cache.rs`), `BlockPools` / `BlockTable` (`src/blocks.rs`),
// `EntryTree` (`src/prefix.rs`), and `SpillStore` (`src/spill.rs`).
// Compiled via `rustc --edition=2021 --cfg standalone_blocks -O` by
// `scripts/check-serve-ownership-traces.sh`.

#[cfg(unix)]
#[allow(non_camel_case_types, dead_code)]
pub mod libc {
    pub type c_int = i32;

    #[cfg(target_os = "macos")]
    pub const O_DIRECTORY: c_int = 0x0010_0000;
    #[cfg(target_os = "macos")]
    pub const O_NOFOLLOW: c_int = 0x0000_0100;
    #[cfg(target_os = "macos")]
    pub const ELOOP: c_int = 62;
    #[cfg(target_os = "macos")]
    pub const F_NOCACHE: c_int = 48;

    #[cfg(target_os = "linux")]
    pub const O_DIRECTORY: c_int = if cfg!(target_arch = "aarch64") {
        0x0000_4000
    } else {
        0x0001_0000
    };
    #[cfg(target_os = "linux")]
    pub const O_NOFOLLOW: c_int = if cfg!(target_arch = "aarch64") {
        0x0000_8000
    } else {
        0x0002_0000
    };
    #[cfg(target_os = "linux")]
    pub const ELOOP: c_int = 40;

    #[cfg(target_os = "macos")]
    extern "C" {
        pub fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
    }
}

pub mod engine {
    pub const EXT: usize = 1024;
}

#[path = "blocks.rs"]
pub mod blocks;
#[path = "blockbook.rs"]
pub mod blockbook;
#[path = "prefix.rs"]
pub mod prefix;
#[path = "spill.rs"]
pub mod spill;
#[path = "block_cache.rs"]
pub mod block_cache;

use block_cache::{
    effective_pf_target, CacheError, InflightLaneReserve, LaneBlockState, OwnershipSnapshot,
    ServeBlockCache,
};
use blocks::{BlockKind, BlockPools, BlockTable, BLOCK, WINDOW};
use spill::{SpillStore, SPILL_HEADER_LEN};
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

fn tla_seq_u32(xs: &[u32]) -> String {
    let body: Vec<String> = xs.iter().map(|v| v.to_string()).collect();
    format!("<<{}>>", body.join(", "))
}

fn tla_seq_usize(xs: &[usize]) -> String {
    let body: Vec<String> = xs.iter().map(|v| v.to_string()).collect();
    format!("<<{}>>", body.join(", "))
}

fn tla_seq_nested(xs: &[Vec<u32>]) -> String {
    let body: Vec<String> = xs.iter().map(|v| tla_seq_u32(v)).collect();
    format!("<<{}>>", body.join(", "))
}

fn tla_bool(b: bool) -> &'static str {
    if b { "TRUE" } else { "FALSE" }
}

struct TraceEmitter {
    rows: Vec<String>,
    max_win_per_lane: usize,
}

impl TraceEmitter {
    fn new(max_win_per_lane: usize) -> Self {
        Self {
            rows: Vec::new(),
            max_win_per_lane,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit(
        &mut self,
        op: &str,
        cache: &ServeBlockCache,
        pools: &BlockPools,
        tables: &[BlockTable],
        states: &[LaneBlockState],
        matched_at: usize,
        expected_deepest_at: usize,
        corrupt_restored: bool,
        corrupt_cleaned: bool,
    ) {
        let refs: Vec<(&BlockTable, &LaneBlockState)> =
            tables.iter().zip(states.iter()).collect();
        let snap: OwnershipSnapshot = cache.ownership_snapshot(pools, &refs);
        let row = format!(
            "[op |-> \"{op}\", numFull |-> {}, numWin |-> {}, \
             fullRefs |-> {}, winRefs |-> {}, fullGen |-> {}, winGen |-> {}, \
             startFullRefs |-> {}, startWinRefs |-> {}, \
             actualStartFullRefs |-> {}, actualStartWinRefs |-> {}, \
             tableFullRefs |-> {}, tableWinRefs |-> {}, \
             freePlusEvictFull |-> {}, freePlusEvictWin |-> {}, \
             laneFullBlocks |-> {}, laneWinBlocks |-> {}, laneWinCount |-> {}, \
             maxWinPerLane |-> {}, matchedAt |-> {matched_at}, expectedDeepestAt |-> {expected_deepest_at}, \
             allLaneEntriesLive |-> {}, allStartEntriesLiveSealed |-> {}, \
             treeStartHoldersMatch |-> {}, treePinsMatchLanes |-> {}, \
             expiredWinPresent |-> {}, spillOwnersConsistent |-> {}, spilledTipsConsistent |-> {}, \
             corruptRestored |-> {}, corruptCleaned |-> {}, \
             retainedStarts |-> {}, spilledStarts |-> {}]",
            snap.num_full,
            snap.num_win,
            tla_seq_u32(&snap.full_refs),
            tla_seq_u32(&snap.win_refs),
            tla_seq_u32(&snap.full_gen),
            tla_seq_u32(&snap.win_gen),
            tla_seq_u32(&snap.start_full_refs),
            tla_seq_u32(&snap.start_win_refs),
            tla_seq_u32(&snap.actual_start_full_refs),
            tla_seq_u32(&snap.actual_start_win_refs),
            tla_seq_u32(&snap.table_full_refs),
            tla_seq_u32(&snap.table_win_refs),
            snap.free_plus_evict_full,
            snap.free_plus_evict_win,
            tla_seq_nested(&snap.lane_full_blocks),
            tla_seq_nested(&snap.lane_win_blocks),
            tla_seq_usize(&snap.lane_win_count),
            self.max_win_per_lane,
            tla_bool(snap.all_lane_entries_live),
            tla_bool(snap.all_start_entries_live_sealed),
            tla_bool(snap.tree_start_holders_match),
            tla_bool(snap.tree_pins_match_lanes),
            tla_bool(snap.expired_win_present),
            tla_bool(snap.spill_owners_consistent),
            tla_bool(snap.spilled_tips_consistent),
            tla_bool(corrupt_restored),
            tla_bool(corrupt_cleaned),
            snap.retained_starts,
            snap.spilled_starts,
        );
        self.rows.push(row);
    }
}

fn make_tokens(prefix_tag: u32, len: usize) -> Vec<u32> {
    (0..len as u32)
        .map(|i| prefix_tag.wrapping_mul(10_007).wrapping_add(i * 13 + 5))
        .collect()
}

fn stamp_lane_kv(pools: &mut BlockPools, tbl: &BlockTable, block_idx: usize, byte_tag: u8) {
    let bi = blocks::BlockIndex::new(block_idx as u32);
    for kind in [BlockKind::Full, BlockKind::Window] {
        if let Some(e) = tbl.get(kind, bi).unwrap() {
            let bytes = pools.of_mut(kind).slot_bytes_mut(e.slot()).unwrap();
            bytes[0] = byte_tag;
            bytes[bytes.len() - 1] = byte_tag ^ 0x5A;
        }
    }
}

fn walk_tight_spill(spill_root: &Path) -> Vec<String> {
    let spill_dir = spill_root.join("tight_spill");
    let store = SpillStore::open_with_kvb(&spill_dir, 256 * 1024 * 1024, 1).expect("open SpillStore");
    // 4 Full slots and 5 Window slots: because `KINDS = [Window, Full]`, when Lanes 0 & 1 pin
    // 4 blocks total (4 Full + 4 Window), Full has 0 free slots while Window has 1 free slot, so
    // `ensure_span_blocks` on Lane 2 allocates Window block 0 first, fails on Full block 0 with
    // `PoolExhausted(Full)`, and exercises transactional rollback of `newly_allocated`!
    let mut pools = BlockPools::with_capacities(4, 5, 1).expect("host pools");
    let mut cache = ServeBlockCache::with_spill(store);
    let mut tables: Vec<BlockTable> = (0..3).map(|_| BlockTable::new(8).unwrap()).collect();
    let mut states: Vec<LaneBlockState> = (0..3).map(|_| LaneBlockState::default()).collect();
    let mut em = TraceEmitter::new(WINDOW / BLOCK + 1);
    em.emit("init", &cache, &pools, &tables, &states, 0, 0, false, false);

    let base = make_tokens(1, 256);
    let p1 = base[..65].to_vec();
    let p2 = base[..130].to_vec();
    let p3 = base[..195].to_vec();

    // 1. Lane 0 & Lane 1 start cold in default namespace 0 (so `ns-isolation-breach` collapsing
    //    `state.namespace = 0` keeps Principal 0 intact and causes Principal 99 to hit Principal 0's cache!).
    let exp0 = cache.expected_deepest_live_at(&pools, 0, &p1);
    let b0 = cache
        .match_and_bind_in_namespace(&mut pools, &mut tables[0], &mut states[0], 0, &p1)
        .map_or(0, |(at, _)| at);
    em.emit("bind", &cache, &pools, &tables, &states, b0, exp0, false, false);

    let exp1 = cache.expected_deepest_live_at(&pools, 0, &p2);
    let b1 = cache
        .match_and_bind_in_namespace(&mut pools, &mut tables[1], &mut states[1], 0, &p2)
        .map_or(0, |(at, _)| at);
    em.emit("bind", &cache, &pools, &tables, &states, b1, exp1, false, false);

    // Lane 0 prefills 0..64 -> saves RetainedStart at 64.
    cache.ensure_span_blocks(&mut pools, &mut tables[0], 0, 64).unwrap();
    stamp_lane_kv(&mut pools, &tables[0], 0, 0xA1);
    em.emit("ensure_ok", &cache, &pools, &tables, &states, 0, 0, false, false);
    cache.finish_prefill_span(&mut pools, &mut tables[0], &mut states[0], &p1, 0, 64, || vec![0x11; 32]);
    em.emit("prefill_finish", &cache, &pools, &tables, &states, 0, 0, false, false);

    // Singleflight follower rebind: Lane 1 rebinds to 64.
    let exp_rebind = cache.expected_deepest_live_at(&pools, 0, &p2);
    let reb = cache
        .rebind_if_deeper_match(&mut pools, &mut tables[1], &mut states[1], &p2, 0)
        .map_or(0, |(at, _)| at);
    em.emit("rebind", &cache, &pools, &tables, &states, reb, exp_rebind, false, false);

    // Racing publisher dedup (`BlockTable::fill` dropping overwritten prev ref):
    states[2].namespace = 0;
    states[2].prompt_len = p1.len();
    cache.ensure_span_blocks(&mut pools, &mut tables[2], 0, 64).unwrap();
    em.emit("ensure_ok", &cache, &pools, &tables, &states, 0, 0, false, false);
    cache.finish_prefill_span(&mut pools, &mut tables[2], &mut states[2], &p1, 0, 64, || vec![0x11; 32]);
    em.emit("prefill_finish", &cache, &pools, &tables, &states, 0, 0, false, false);
    cache.release_lane(&mut pools, &mut tables[2], &mut states[2]);
    em.emit("release_lane", &cache, &pools, &tables, &states, 0, 0, false, false);

    // Finish Lane 0 tail (64..65) + 1 decode step (65..66) + release.
    cache.ensure_span_blocks(&mut pools, &mut tables[0], 64, 1).unwrap();
    em.emit("ensure_ok", &cache, &pools, &tables, &states, 0, 0, false, false);
    cache.finish_prefill_span(&mut pools, &mut tables[0], &mut states[0], &p1, 64, 1, || vec![0x12; 32]);
    em.emit("prefill_finish", &cache, &pools, &tables, &states, 0, 0, false, false);
    cache.ensure_span_blocks(&mut pools, &mut tables[0], 65, 1).unwrap();
    em.emit("ensure_ok", &cache, &pools, &tables, &states, 0, 0, false, false);
    cache.finish_decode_span(&mut pools, &mut tables[0], &mut states[0], 65);
    em.emit("decode_finish", &cache, &pools, &tables, &states, 0, 0, false, false);
    cache.release_lane(&mut pools, &mut tables[0], &mut states[0]);
    em.emit("release_lane", &cache, &pools, &tables, &states, 0, 0, false, false);

    // Lane 1 advances from its bound position up to 128 -> saves second RetainedStart at 128.
    let l1_cur = states[1].written;
    if l1_cur < 128 {
        cache.ensure_span_blocks(&mut pools, &mut tables[1], l1_cur, 128 - l1_cur).unwrap();
        stamp_lane_kv(&mut pools, &tables[1], 1, 0xB2);
        em.emit("ensure_ok", &cache, &pools, &tables, &states, 0, 0, false, false);
        cache.finish_prefill_span(&mut pools, &mut tables[1], &mut states[1], &p2, l1_cur, 128 - l1_cur, || vec![0x22; 32]);
        em.emit("prefill_finish", &cache, &pools, &tables, &states, 0, 0, false, false);
    }
    cache.release_lane(&mut pools, &mut tables[1], &mut states[1]);
    em.emit("release_lane", &cache, &pools, &tables, &states, 0, 0, false, false);

    // 2. Deepest-over-shallowest check: `p3` (195 tokens) in namespace 0 MUST match 128, not 64.
    let exp_p3 = cache.expected_deepest_live_at(&pools, 0, &p3);
    let b_p3 = cache
        .match_and_bind_in_namespace(&mut pools, &mut tables[0], &mut states[0], 0, &p3)
        .map_or(0, |(at, _)| at);
    em.emit("bind", &cache, &pools, &tables, &states, b_p3, exp_p3, false, false);

    // Namespace isolation check: identical `p3` in namespace 99 MUST observe matchedAt = 0.
    let ns_hit = cache
        .match_and_bind_in_namespace(&mut pools, &mut tables[1], &mut states[1], 99, &p3)
        .map_or(0, |(at, _)| at);
    em.emit("ns_isolated", &cache, &pools, &tables, &states, ns_hit, 0, false, false);
    cache.release_lane(&mut pools, &mut tables[1], &mut states[1]);
    cache.release_lane(&mut pools, &mut tables[0], &mut states[0]);
    em.emit("release_lane", &cache, &pools, &tables, &states, 0, 0, false, false);

    // 3. Transactional `ensure_span_blocks` rollback on `PoolExhausted`:
    //    Allocating 2 blocks on Lane 0 + 2 blocks on Lane 1 evicts retained starts 64 & 128 to SSD
    //    and pins all 4 Window slots while leaving 1 Full slot free (`5 - 4 = 1`).
    let alt_a = make_tokens(10, 130);
    let alt_b = make_tokens(20, 130);
    states[0].namespace = 30;
    states[0].prompt_len = alt_a.len();
    if cache.ensure_span_blocks(&mut pools, &mut tables[0], 0, 128).is_ok() {
        em.emit("ensure_ok", &cache, &pools, &tables, &states, 0, 0, false, false);
    }

    states[1].namespace = 31;
    states[1].prompt_len = alt_b.len();
    if cache.ensure_span_blocks(&mut pools, &mut tables[1], 0, 128).is_ok() {
        em.emit("ensure_ok", &cache, &pools, &tables, &states, 0, 0, false, false);
        let active_res = [
            InflightLaneReserve {
                lane: 0,
                namespace: 30,
                toks: &alt_a,
                n_past: 0,
                pf_end: alt_a.len(),
                cached: 0,
                written: 0,
                gen: 0,
                max_gen: 16,
                runnable: true,
            },
            InflightLaneReserve {
                lane: 1,
                namespace: 31,
                toks: &alt_b,
                n_past: 0,
                pf_end: alt_b.len(),
                cached: 0,
                written: 0,
                gen: 0,
                max_gen: 16,
                runnable: true,
            },
        ];
        let _ = cache.can_admit_in_namespace(&pools, &active_res, 2, 32, &p1, 16, 2048);

        // Full has 1 free slot, Window has 0 -> `ensure_span_blocks` allocates Full block 0,
        // hits `PoolExhausted(Window)`, and must roll back Full block 0!
        let err = cache.ensure_span_blocks(&mut pools, &mut tables[2], 0, 64);
        if err.is_err() {
            em.emit("ensure_rollback", &cache, &pools, &tables, &states, 0, 0, false, false);
        }
    }

    cache.release_lane(&mut pools, &mut tables[2], &mut states[2]);
    cache.release_lane(&mut pools, &mut tables[0], &mut states[0]);
    cache.release_lane(&mut pools, &mut tables[1], &mut states[1]);
    em.emit("release_lane", &cache, &pools, &tables, &states, 0, 0, false, false);

    // 4. SSD Restore check:
    let ns_disk_hit = cache
        .match_and_bind_in_namespace(&mut pools, &mut tables[1], &mut states[1], 99, &p3)
        .map_or(0, |(at, _)| at);
    em.emit("ns_isolated", &cache, &pools, &tables, &states, ns_disk_hit, 0, false, false);
    cache.release_lane(&mut pools, &mut tables[1], &mut states[1]);

    let exp_restore = cache.expected_deepest_live_at(&pools, 0, &p3);
    let restored_at = cache
        .match_and_bind_in_namespace(&mut pools, &mut tables[0], &mut states[0], 0, &p3)
        .map_or(0, |(at, _)| at);
    if let Some(f0) = tables[0].get(BlockKind::Full, blocks::BlockIndex::new(0)).unwrap() {
        assert_eq!(pools.full.slot_bytes(f0.slot()).unwrap()[0], 0xA1);
    }
    if let Some(f1) = tables[0].get(BlockKind::Full, blocks::BlockIndex::new(1)).unwrap() {
        assert_eq!(pools.full.slot_bytes(f1.slot()).unwrap()[0], 0xB2);
    }
    em.emit("bind", &cache, &pools, &tables, &states, restored_at, exp_restore, false, false);
    cache.release_lane(&mut pools, &mut tables[0], &mut states[0]);
    em.emit("release_lane", &cache, &pools, &tables, &states, 0, 0, false, false);

    // 5. Corrupt `.blk` payload test:
    let c_toks = make_tokens(77, 70);
    let c_ext = make_tokens(77, 120);
    cache.match_and_bind_in_namespace(&mut pools, &mut tables[0], &mut states[0], 77, &c_toks);
    if cache.ensure_span_blocks(&mut pools, &mut tables[0], 0, 64).is_ok() {
        stamp_lane_kv(&mut pools, &tables[0], 0, 0xCC);
        em.emit("ensure_ok", &cache, &pools, &tables, &states, 0, 0, false, false);
        cache.finish_prefill_span(&mut pools, &mut tables[0], &mut states[0], &c_toks, 0, 64, || vec![0x77; 32]);
        em.emit("prefill_finish", &cache, &pools, &tables, &states, 0, 0, false, false);
    }
    cache.release_lane(&mut pools, &mut tables[0], &mut states[0]);
    em.emit("release_lane", &cache, &pools, &tables, &states, 0, 0, false, false);

    states[0].namespace = 88;
    if cache.ensure_span_blocks(&mut pools, &mut tables[0], 0, 256).is_ok() {
        em.emit("ensure_ok", &cache, &pools, &tables, &states, 0, 0, false, false);
    }
    cache.release_lane(&mut pools, &mut tables[0], &mut states[0]);
    em.emit("release_lane", &cache, &pools, &tables, &states, 0, 0, false, false);

    let mut corrupted_files = Vec::new();
    for ent in fs::read_dir(&spill_dir).unwrap() {
        let path = ent.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) == Some("blk") {
            let len = fs::metadata(&path).unwrap().len();
            if len > (SPILL_HEADER_LEN + 64) as u64 {
                let mut f = OpenOptions::new().read(true).write(true).open(&path).unwrap();
                f.seek(SeekFrom::Start(SPILL_HEADER_LEN as u64)).unwrap();
                let mut b = [0u8; 1];
                f.read_exact(&mut b).unwrap();
                b[0] ^= 0xFF;
                f.seek(SeekFrom::Start(SPILL_HEADER_LEN as u64)).unwrap();
                f.write_all(&b).unwrap();
                f.sync_all().unwrap();
                corrupted_files.push(path);
            }
        }
    }

    if !corrupted_files.is_empty() {
        let corrupt_match = cache.match_and_bind_in_namespace(&mut pools, &mut tables[0], &mut states[0], 77, &c_ext);
        let corrupt_restored = corrupt_match.is_some();
        let corrupt_cleaned = cache.spilled_start_count() == 0
            || corrupted_files.iter().any(|p| !p.exists());
        em.emit(
            "corrupt_restore_rejected",
            &cache,
            &pools,
            &tables,
            &states,
            corrupt_match.map_or(0, |(at, _)| at),
            0,
            corrupt_restored,
            corrupt_cleaned,
        );
        cache.release_lane(&mut pools, &mut tables[0], &mut states[0]);
    }

    cache.maintain_spill_during_gpu();
    em.emit("maintain_spill", &cache, &pools, &tables, &states, 0, 0, false, false);

    cache.shutdown(&mut pools);
    em.emit("shutdown", &cache, &pools, &tables, &states, 0, 0, false, false);
    em.rows
}

fn walk_window_slide() -> Vec<String> {
    // 20 Window slots > 19 blocks written so `window-prune-skip` never panics in Rust
    // and instead reaches TLC with `expiredWinPresent = TRUE` / `laneWinCount = 19 > 17`.
    let mut pools = BlockPools::with_capacities(22, 20, 1).expect("window slide pools");
    let mut cache = ServeBlockCache::new();
    let mut tables = vec![BlockTable::new(24).unwrap()];
    let mut states = vec![LaneBlockState::default()];
    let mut em = TraceEmitter::new(WINDOW / BLOCK + 1);
    em.emit("init", &cache, &pools, &tables, &states, 0, 0, false, false);

    let toks = make_tokens(500, 1090);
    let exp0 = cache.expected_deepest_live_at(&pools, 1, &toks);
    let m0 = cache
        .match_and_bind_in_namespace(&mut pools, &mut tables[0], &mut states[0], 1, &toks)
        .map_or(0, |(at, _)| at);
    em.emit("bind", &cache, &pools, &tables, &states, m0, exp0, false, false);

    let tgt = effective_pf_target(0, toks.len());
    assert_eq!(tgt, 1088);
    cache.ensure_span_blocks(&mut pools, &mut tables[0], 0, 1024).unwrap();
    em.emit("ensure_ok", &cache, &pools, &tables, &states, 0, 0, false, false);
    cache.finish_prefill_span(&mut pools, &mut tables[0], &mut states[0], &toks, 0, 1024, || vec![0x51; 16]);
    em.emit("prefill_finish", &cache, &pools, &tables, &states, 0, 0, false, false);

    cache.ensure_span_blocks(&mut pools, &mut tables[0], 1024, 64).unwrap();
    em.emit("ensure_ok", &cache, &pools, &tables, &states, 0, 0, false, false);
    cache.finish_prefill_span(&mut pools, &mut tables[0], &mut states[0], &toks, 1024, 64, || vec![0x52; 16]);
    em.emit("prefill_finish", &cache, &pools, &tables, &states, 0, 0, false, false);

    cache.ensure_span_blocks(&mut pools, &mut tables[0], 1088, 2).unwrap();
    cache.finish_prefill_span(&mut pools, &mut tables[0], &mut states[0], &toks, 1088, 2, || vec![0x53; 16]);
    em.emit("prefill_finish", &cache, &pools, &tables, &states, 0, 0, false, false);

    for pos in [1090usize, 1150, 1151, 1152] {
        cache.ensure_span_blocks(&mut pools, &mut tables[0], pos, 1).unwrap();
        cache.finish_decode_span(&mut pools, &mut tables[0], &mut states[0], pos);
    }
    em.emit("decode_finish", &cache, &pools, &tables, &states, 0, 0, false, false);

    cache.release_lane(&mut pools, &mut tables[0], &mut states[0]);
    em.emit("release_lane", &cache, &pools, &tables, &states, 0, 0, false, false);

    cache.shutdown(&mut pools);
    em.emit("shutdown", &cache, &pools, &tables, &states, 0, 0, false, false);
    em.rows
}

fn walk_seeded(spill_root: &Path, seed: u64) -> Vec<String> {
    let spill_dir = spill_root.join(format!("seed_{seed}"));
    let store = SpillStore::open_with_kvb(&spill_dir, 256 * 1024 * 1024, 1).expect("open seeded SpillStore");
    let mut pools = BlockPools::with_capacities(7, 6, 1).expect("seeded pools");
    let mut cache = ServeBlockCache::with_spill(store);
    let mut tables: Vec<BlockTable> = (0..3).map(|_| BlockTable::new(8).unwrap()).collect();
    let mut states: Vec<LaneBlockState> = (0..3).map(|_| LaneBlockState::default()).collect();
    let mut lane_toks: Vec<Option<Vec<u32>>> = vec![None, None, None];
    let mut em = TraceEmitter::new(WINDOW / BLOCK + 1);
    em.emit("init", &cache, &pools, &tables, &states, 0, 0, false, false);

    let mut rng = seed;
    let mut next_rand = || -> usize {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng >> 32) as usize
    };

    let families = [
        make_tokens(1, 220),
        make_tokens(2, 220),
        make_tokens(3, 220),
    ];

    for _step in 0..28 {
        let l = next_rand() % 3;
        if lane_toks[l].is_none() {
            let fam = &families[next_rand() % families.len()];
            let plen = match next_rand() % 3 {
                0 => 68,
                1 => 132,
                _ => 196,
            };
            let ns = if next_rand() % 2 == 0 { 1u64 } else { 2u64 };
            let toks = fam[..plen].to_vec();
            let exp = cache.expected_deepest_live_at(&pools, ns, &toks);
            let m = cache
                .match_and_bind_in_namespace(&mut pools, &mut tables[l], &mut states[l], ns, &toks)
                .map_or(0, |(at, _)| at);
            em.emit("bind", &cache, &pools, &tables, &states, m, exp, false, false);
            lane_toks[l] = Some(toks);
        } else {
            let toks = lane_toks[l].clone().unwrap();
            let cur = states[l].written;
            if cur < toks.len() {
                let tgt = effective_pf_target(cur, toks.len()).min(cur + 64);
                let span = tgt - cur;
                match cache.ensure_span_blocks(&mut pools, &mut tables[l], cur, span) {
                    Ok(()) => {
                        em.emit("ensure_ok", &cache, &pools, &tables, &states, 0, 0, false, false);
                        cache.finish_prefill_span(
                            &mut pools,
                            &mut tables[l],
                            &mut states[l],
                            &toks,
                            cur,
                            span,
                            || vec![(l as u8) + 1; 24],
                        );
                        em.emit("prefill_finish", &cache, &pools, &tables, &states, 0, 0, false, false);
                    }
                    Err(CacheError::PoolExhausted(_)) => {
                        em.emit("ensure_rollback", &cache, &pools, &tables, &states, 0, 0, false, false);
                        cache.release_lane(&mut pools, &mut tables[l], &mut states[l]);
                        lane_toks[l] = None;
                        em.emit("release_lane", &cache, &pools, &tables, &states, 0, 0, false, false);
                    }
                }
            } else if cur < toks.len() + 2 && next_rand() % 2 == 0 {
                match cache.ensure_span_blocks(&mut pools, &mut tables[l], cur, 1) {
                    Ok(()) => {
                        cache.finish_decode_span(&mut pools, &mut tables[l], &mut states[l], cur);
                        em.emit("decode_finish", &cache, &pools, &tables, &states, 0, 0, false, false);
                    }
                    Err(CacheError::PoolExhausted(_)) => {
                        em.emit("ensure_rollback", &cache, &pools, &tables, &states, 0, 0, false, false);
                        cache.release_lane(&mut pools, &mut tables[l], &mut states[l]);
                        lane_toks[l] = None;
                        em.emit("release_lane", &cache, &pools, &tables, &states, 0, 0, false, false);
                    }
                }
            } else {
                cache.release_lane(&mut pools, &mut tables[l], &mut states[l]);
                lane_toks[l] = None;
                em.emit("release_lane", &cache, &pools, &tables, &states, 0, 0, false, false);
            }
        }
    }

    for l in 0..3 {
        if lane_toks[l].take().is_some() {
            cache.release_lane(&mut pools, &mut tables[l], &mut states[l]);
            em.emit("release_lane", &cache, &pools, &tables, &states, 0, 0, false, false);
        }
    }
    cache.shutdown(&mut pools);
    em.emit("shutdown", &cache, &pools, &tables, &states, 0, 0, false, false);
    em.rows
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_dir = Path::new(args.get(1).expect("usage: generate <out_dir> [mode]"));
    let mode = args.get(2).map(String::as_str).unwrap_or("all");
    let bundle_dir = out_dir.join("bundle");
    let spill_root = out_dir.join("spill_work");
    fs::create_dir_all(&bundle_dir).unwrap();
    fs::create_dir_all(&spill_root).unwrap();

    let mut traces = Vec::new();
    match mode {
        "window" => {
            traces.push(walk_window_slide());
        }
        "tight" => {
            traces.push(walk_tight_spill(&spill_root));
        }
        _ => {
            traces.push(walk_tight_spill(&spill_root));
            traces.push(walk_window_slide());
            for seed in [0x1234_5678_9abc_def0u64, 0xfeed_face_cafe_beefu64, 0x0ddc_0ffe_e123_4567u64] {
                traces.push(walk_seeded(&spill_root, seed));
            }
        }
    }

    let rendered: Vec<String> = traces
        .iter()
        .map(|rows| format!("<<\n{}\n>>", rows.join(",\n")))
        .collect();
    let module = format!(
        "----------------------- MODULE ServeOwnershipTraceData -----------------------\n\
         EXTENDS Integers, Sequences\n\n\
         Traces == <<\n{}\n>>\n\n\
         =============================================================================\n",
        rendered.join(",\n")
    );
    fs::write(bundle_dir.join("ServeOwnershipTraceData.tla"), module).unwrap();
    let total_rows: usize = traces.iter().map(Vec::len).sum();
    eprintln!(
        "serve-ownership-traces: mode={mode}, traces={}, rows={total_rows}",
        traces.len()
    );
}
