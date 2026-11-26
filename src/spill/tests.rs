use super::{store::prepare_spill_dir, *};
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// RAII scratch test directory created under `target/spill-tests/` (never `/tmp`).
struct ScratchTestDir {
    path: PathBuf,
}

impl ScratchTestDir {
    fn new(label: &str) -> Self {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("spill-tests");
        fs::create_dir_all(&base).expect("create target/spill-tests");
        #[cfg(unix)]
        let _ = fs::set_permissions(&base, fs::Permissions::from_mode(0o700));
        let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let path = base.join(format!("{label}-{}-{nonce}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create scratch test dir");
        #[cfg(unix)]
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700))
            .expect("chmod 0700 scratch test dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchTestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

#[test]
fn fresh_checkpoint_keys_skip_existing_blocks_after_restart() {
    let scratch = ScratchTestDir::new("restart-ids");
    let mut first = SpillStore::open(scratch.path(), 1024).unwrap();
    let old = first.fresh_key(SpillBlockKind::Full);
    first.write_block(old, 1, b"old KV", None, &HashSet::new()).unwrap();
    drop(first);

    let mut next = SpillStore::open(scratch.path(), 1024).unwrap();
    let fresh = next.fresh_key(SpillBlockKind::Full);
    assert_ne!(old, fresh);
    assert!(!next.contains(fresh));
    next.write_block(fresh, 1, b"new KV", None, &HashSet::new()).unwrap();
    assert!(matches!(next.read_block(old, None), ReadOutcome::Hit(b) if b.kv_bytes == b"old KV"));
    assert!(matches!(next.read_block(fresh, None), ReadOutcome::Hit(b) if b.kv_bytes == b"new KV"));
}

#[test]
fn dropping_the_store_releases_ownership_even_with_a_duplicate_fd() {
    let scratch = ScratchTestDir::new("directory-lock-lifetime");
    let owner = SpillStore::open(scratch.path(), 1024).unwrap();
    // A child between fork and exec holds another descriptor for the same open file.
    let inherited = owner.clone_directory_lock().unwrap();
    drop(owner);
    let successor = SpillStore::open(scratch.path(), 1024).unwrap();
    drop(inherited);
    assert!(SpillStore::open(scratch.path(), 1024).is_err());
    drop(successor);
}

#[test]
fn one_process_owns_the_spill_directory_until_drop() {
    const CHILD_DIR: &str = "INKY_TEST_CONTENDED_SPILL_DIR";
    if let Some(dir) = std::env::var_os(CHILD_DIR) {
        // A second opener must fail before a zero-byte budget can evict the writer's blocks.
        let result = SpillStore::open(dir, 0);
        assert!(matches!(result, Err(SpillError::InvalidDir(ref reason)) if reason.contains("already in use")));
        return;
    }

    let scratch = ScratchTestDir::new("directory-owner");
    let dir = scratch.path();
    let mut owner = SpillStore::open(dir, 1024).unwrap();
    let key = SpillKey::full(37);
    owner.write_block(key, 1, b"retained", None, &HashSet::new()).unwrap();
    let child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "spill::tests::one_process_owns_the_spill_directory_until_drop"])
        .env(CHILD_DIR, dir)
        .output().unwrap();
    assert!(child.status.success(), "{}", String::from_utf8_lossy(&child.stdout));
    drop(owner);

    let mut successor = SpillStore::open(dir, 1024).unwrap();
    assert!(matches!(successor.read_block(key, None), ReadOutcome::Hit(block) if block.kv_bytes == b"retained"));
}

/// Deterministic ground-truth generator for block K/V bytes and optional recurrent state.
fn deterministic_kv(key: SpillKey, len: usize) -> Vec<u8> {
    let mut x = key.id.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ ((key.kind as u64 + 1) << 56) ^ 0xa076_1d64_78bd_642f;
    let mut out = vec![0u8; len];
    for chunk in out.chunks_mut(8) {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        let v = x.wrapping_mul(0x2545_f491_4f6c_dd1d).to_le_bytes();
        let n = chunk.len();
        chunk.copy_from_slice(&v[..n]);
    }
    out
}

fn deterministic_state(key: SpillKey, len: usize) -> Vec<u8> {
    let mut x = key.id.wrapping_mul(0xd1b5_4a32_d192_ed03) ^ 0x55aa_33cc_9966_11ee;
    let mut out = vec![0u8; len];
    for chunk in out.chunks_mut(8) {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        let v = x.wrapping_mul(0x94d0_49bb_1331_11eb).to_le_bytes();
        let n = chunk.len();
        chunk.copy_from_slice(&v[..n]);
    }
    out
}

#[test]
fn sha256_matches_nist_vectors() {
    let h_empty = Sha256::new().finalize();
    assert_eq!(
        hex(&h_empty),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );

    let mut h_abc = Sha256::new();
    h_abc.update(b"abc");
    assert_eq!(
        hex(&h_abc.finalize()),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );

    let mut h_448 = Sha256::new();
    h_448.update(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
    assert_eq!(
        hex(&h_448.finalize()),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );

    // Streaming chunked equivalence on 128 KiB payload.
    let payload = deterministic_kv(SpillKey::full(42), 131_072);
    let mut whole = Sha256::new();
    whole.update(&payload);
    let mut chunked = Sha256::new();
    for c in payload.chunks(137) {
        chunked.update(c);
    }
    assert_eq!(whole.finalize(), chunked.finalize());
}

#[cfg(unix)]
#[test]
fn spill_dir_and_files_have_strict_unix_permissions() {
    let scratch = ScratchTestDir::new("perms");
    let dir = scratch.path().join("spill");
    let mut store = SpillStore::open(&dir, 1024 * 1024).unwrap();
    let dir_mode = fs::symlink_metadata(&dir).unwrap().mode() & 0o777;
    assert_eq!(dir_mode, 0o700, "spill dir must be mode 0700");

    let key = SpillKey::full(7);
    let kv = deterministic_kv(key, 4096);
    let (outcome, _) = store.write_block(key, 1, &kv, None, &HashSet::new()).unwrap();
    assert!(matches!(outcome, WriteOutcome::Written { .. }));
    let file_mode = fs::symlink_metadata(store.block_path(key)).unwrap().mode() & 0o777;
    assert_eq!(file_mode, 0o600, "spilled .blk files must be mode 0600");
}

#[test]
fn lazy_lru_spill_never_writes_eagerly_and_evicts_strict_lru_under_pressure() {
    let scratch = ScratchTestDir::new("lazy-lru");
    let dir = scratch.path().join("spill");
    let full_stride = 4096;
    let win_stride = 8192;
    let pool = HostBlockPoolBackend::new(4, 2, full_stride, win_stride);
    let store = SpillStore::open(&dir, 64 * 1024 * 1024).unwrap();
    let mut mgr = SpillManager::new(pool, store);

    // Allocate, fill, seal, and pin 4 full blocks (fills 100% of Full pool).
    let mut slots = Vec::new();
    for id in 0..4u64 {
        let key = SpillKey::full(id);
        let slot = mgr.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
        let kv = deterministic_kv(key, full_stride);
        mgr.pool_mut().write_and_seal(slot, &kv).unwrap();
        let state = if id == 2 { Some(deterministic_state(key, 1024)) } else { None };
        mgr.register_sealed_block(key, slot, state).unwrap();
        mgr.pin(key).unwrap();
        slots.push((key, slot));
    }

    // LAZY INVARIANT: 0 disk writes while filling memory pool.
    assert_eq!(mgr.store().stats().writes_committed, 0);
    assert_eq!(mgr.store().entry_count(), 0);

    // Unpin all 4 blocks in an explicit recency order:
    // oldest -> newest: id 2, id 0, id 3, id 1.
    for id in [2u64, 0, 3, 1] {
        assert_eq!(mgr.unpin(SpillKey::full(id)).unwrap(), 0);
    }
    // Re-pin id 0 so it CANNOT be evicted even though it is older than 3 and 1.
    mgr.pin(SpillKey::full(0)).unwrap();

    // LAZY INVARIANT: dropping refs to 0 still triggers ZERO disk writes!
    assert_eq!(mgr.store().stats().writes_committed, 0);
    assert_eq!(mgr.store().entry_count(), 0);
    for id in 0..4u64 {
        assert_eq!(mgr.tier(SpillKey::full(id)), BlockTier::Memory);
    }

    // Now request a 5th full block -> memory pressure!
    // Must evict id 2 (oldest unreferenced) ONLY, writing its K/V + recurrent state.
    let k4 = SpillKey::full(4);
    let s4 = mgr.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
    mgr.pool_mut().write_and_seal(s4, &deterministic_kv(k4, full_stride)).unwrap();
    mgr.register_sealed_block(k4, s4, None).unwrap();
    mgr.pin(k4).unwrap();

    assert_eq!(mgr.stats().lazy_spills_written, 1);
    assert_eq!(mgr.tier(SpillKey::full(2)), BlockTier::Disk);
    assert_eq!(mgr.tier(SpillKey::full(0)), BlockTier::Memory); // pinned!
    assert_eq!(mgr.tier(SpillKey::full(3)), BlockTier::Memory);
    assert_eq!(mgr.tier(SpillKey::full(1)), BlockTier::Memory);

    // Request a 6th full block -> must evict id 3 (NOT pinned id 0, NOT newer id 1).
    let k5 = SpillKey::full(5);
    let s5 = mgr.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
    mgr.pool_mut().write_and_seal(s5, &deterministic_kv(k5, full_stride)).unwrap();
    mgr.register_sealed_block(k5, s5, None).unwrap();
    mgr.pin(k5).unwrap();

    assert_eq!(mgr.stats().lazy_spills_written, 2);
    assert_eq!(mgr.tier(SpillKey::full(3)), BlockTier::Disk);
    assert_eq!(mgr.tier(SpillKey::full(0)), BlockTier::Memory);
    assert_eq!(mgr.tier(SpillKey::full(1)), BlockTier::Memory);

    // Restore id 2 (a start point with recurrent state) -> must evict id 1,
    // read id 2 from SSD, and recover bit-identical K/V + recurrent state!
    let prepared = mgr.prepare_working_set_restore(&[SpillKey::full(2)], &[]).unwrap();
    let restored = mgr.execute_working_set_restore(prepared).unwrap();
    assert_eq!(restored.full_slots.len(), 1);
    let (rk, rslot) = restored.full_slots[0];
    assert_eq!(rk, SpillKey::full(2));
    assert_eq!(
        mgr.pool().read_slot_bytes(rslot).unwrap(),
        deterministic_kv(SpillKey::full(2), full_stride).as_slice()
    );
    assert_eq!(
        restored.recurrent_states.get(&SpillKey::full(2)).map(Vec::as_slice),
        Some(deterministic_state(SpillKey::full(2), 1024).as_slice())
    );
    assert_eq!(mgr.tier(SpillKey::full(2)), BlockTier::Both);
    assert_eq!(mgr.tier(SpillKey::full(1)), BlockTier::Disk);

    // Unpin id 2 and trigger one more eviction: because id 2 is already `Both`
    // (clean copy on disk), evicting it again must NOT rewrite to SSD!
    mgr.unpin(SpillKey::full(2)).unwrap();
    let writes_before = mgr.store().stats().writes_committed;
    let _s6 = mgr.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
    assert_eq!(mgr.store().stats().writes_committed, writes_before);
    assert_eq!(mgr.stats().clean_mem_evictions, 1);
    assert_eq!(mgr.tier(SpillKey::full(2)), BlockTier::Disk);
}

#[test]
fn failed_unlink_preserves_budget_and_gc_skips_the_stuck_victim() {
    let scratch = ScratchTestDir::new("unlink-budget");
    let charge = (SPILL_HEADER_LEN + 4) as u64;
    let mut store = SpillStore::open(scratch.path().join("spill"), 2 * charge).unwrap();
    let keys = [1, 2, 3, 4].map(SpillKey::full);
    let unprotected = HashSet::new();
    for &key in &keys[..2] {
        store.write_block(key, 1, b"data", None, &unprotected).unwrap();
    }

    // Preserve the original bytes under a directory that remove_file cannot unlink,
    // including when this test runs with elevated filesystem permissions.
    let path = store.block_path(keys[0]);
    let saved = scratch.path().join("saved");
    fs::rename(&path, &saved).unwrap();
    fs::create_dir(&path).unwrap();
    fs::rename(&saved, path.join("original")).unwrap();
    assert!(!store.remove_block(keys[0]));
    assert!(matches!(store.read_block(keys[0], None), ReadOutcome::Miss(_)));
    assert_eq!(store.stats().corrupt_files_removed, 0);
    assert!(store.contains(keys[0]));
    assert_eq!(store.used_bytes(), 2 * charge);

    let (outcome, victims) = store.write_block(keys[2], 1, b"data", None, &unprotected).unwrap();
    assert_eq!(outcome, WriteOutcome::Written { on_disk_bytes: charge });
    assert_eq!(victims, vec![keys[1]]);
    let (outcome, victims) = store.write_block(keys[3], 1, b"data", None, &HashSet::from([keys[2]])).unwrap();
    assert_eq!(outcome, WriteOutcome::OverBudget);
    assert!(victims.is_empty());
    assert_eq!(store.stats().disk_gc_evictions, 1);
    assert!(store.contains(keys[0]));
    assert_eq!(store.used_bytes(), 2 * charge);

    fs::rename(path.join("original"), &saved).unwrap();
    fs::remove_dir(&path).unwrap();
    fs::rename(&saved, &path).unwrap();
    assert!(store.remove_block(keys[0]));
    assert_eq!(store.used_bytes(), charge);
}

#[test]
fn startup_reports_canonical_block_it_cannot_remove() {
    let scratch = ScratchTestDir::new("startup-unlink");
    let dir = scratch.path().join("spill");
    fs::create_dir(&dir).unwrap();
    let stuck = dir.join(SpillKey::full(1).blk_filename());
    fs::create_dir(&stuck).unwrap();
    assert!(matches!(SpillStore::open(&dir, 0), Err(SpillError::Io(_))));
    assert!(stuck.is_dir());
    fs::remove_dir(stuck).unwrap();
    let store = SpillStore::open(&dir, 0).unwrap();
    assert_eq!(store.used_bytes(), 0);
}

#[test]
fn transient_open_error_retains_charge_and_absent_files_release_it_once() {
    let scratch = ScratchTestDir::new("open-accounting");
    let dir = scratch.path().join("spill");
    let charge = (SPILL_HEADER_LEN + 4) as u64;
    let mut store = SpillStore::open(&dir, 2 * charge).unwrap();
    let keys = [1, 2].map(SpillKey::full);
    for key in keys {
        store.write_block(key, 1, b"data", None, &HashSet::new()).unwrap();
    }

    let saved = scratch.path().join("saved-dir");
    fs::rename(&dir, &saved).unwrap();
    fs::write(&dir, b"temporarily inaccessible").unwrap();
    assert_eq!(store.read_block(keys[0], None), ReadOutcome::Miss(MissReason::IoReadError));
    assert_eq!(store.used_bytes(), 2 * charge);
    assert!(store.contains(keys[0]));
    fs::remove_file(&dir).unwrap();
    fs::rename(&saved, &dir).unwrap();
    assert!(matches!(store.read_block(keys[0], None), ReadOutcome::Hit(_)));

    fs::remove_file(store.block_path(keys[0])).unwrap();
    assert_eq!(store.read_block(keys[0], None), ReadOutcome::Miss(MissReason::NotFound));
    assert!(!store.remove_block(keys[0]));
    assert_eq!(store.used_bytes(), charge);
    fs::remove_file(store.block_path(keys[1])).unwrap();
    assert!(store.remove_block(keys[1]));
    assert!(!store.remove_block(keys[1]));
    assert_eq!(store.used_bytes(), 0);
}

#[test]
fn disk_budget_enforces_max_spill_bytes_with_lru_gc() {
    let scratch = ScratchTestDir::new("disk-gc");
    let dir = scratch.path().join("spill");
    let kv_len = 4096usize;
    let one_file = (SPILL_HEADER_LEN + kv_len) as u64;
    // Budget holds at most 3 spilled blocks.
    let max_bytes = one_file * 3;
    let mut store = SpillStore::open(&dir, max_bytes).unwrap();

    for id in 0..8u64 {
        let key = SpillKey::full(id);
        let kv = deterministic_kv(key, kv_len);
        let (out, _evicted) = store.write_block(key, 1, &kv, None, &HashSet::new()).unwrap();
        assert_eq!(out, WriteOutcome::Written { on_disk_bytes: one_file });
        assert!(
            store.used_bytes() <= max_bytes,
            "used_bytes {} exceeded max_bytes {}",
            store.used_bytes(),
            max_bytes
        );
        // Verify actual on-disk bytes across all files in --spill-dir also respect budget.
        let actual_bytes: u64 = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().metadata().unwrap().len())
            .sum();
        assert!(actual_bytes <= max_bytes);
    }

    assert_eq!(store.entry_count(), 3);
    assert_eq!(store.stats().disk_gc_evictions, 5);
    // Oldest 0..5 evicted; newest 5, 6, 7 survive and verify bit-identically.
    for id in 0..5u64 {
        assert!(matches!(
            store.read_block(SpillKey::full(id), Some(kv_len)),
            ReadOutcome::Miss(MissReason::NotFound)
        ));
    }
    for id in 5..8u64 {
        let key = SpillKey::full(id);
        match store.read_block(key, Some(kv_len)) {
            ReadOutcome::Hit(blk) => assert_eq!(blk.kv_bytes, deterministic_kv(key, kv_len)),
            ReadOutcome::Miss(r) => panic!("expected surviving block {id}, got miss {r:?}"),
        }
    }

    // Touch block 5 (already touched above, now touch 6 and 7 then 5 so 6 is oldest).
    let _ = store.read_block(SpillKey::full(6), Some(kv_len));
    let _ = store.read_block(SpillKey::full(7), Some(kv_len));
    let _ = store.read_block(SpillKey::full(5), Some(kv_len));
    // Writing block 8 must evict block 6 (LRU), keeping 5, 7, 8!
    let k8 = SpillKey::full(8);
    store
        .write_block(k8, 1, &deterministic_kv(k8, kv_len), None, &HashSet::new())
        .unwrap();
    assert!(!store.contains(SpillKey::full(6)));
    assert!(store.contains(SpillKey::full(5)));
    assert!(store.contains(SpillKey::full(7)));
    assert!(store.contains(SpillKey::full(8)));

    // Zero disk budget drops cleanly without failing or writing files.
    let dir0 = scratch.path().join("spill-zero");
    let pool0 = HostBlockPoolBackend::new(1, 1, 512, 512);
    let store0 = SpillStore::open(&dir0, 0).unwrap();
    let mut mgr0 = SpillManager::new(pool0, store0);
    let s0 = mgr0.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
    mgr0.pool_mut().write_and_seal(s0, &deterministic_kv(SpillKey::full(1), 512)).unwrap();
    mgr0.register_sealed_block(SpillKey::full(1), s0, None).unwrap();
    // refs == 0 -> allocating another block drops block 1 cleanly with 0 disk usage.
    let _s1 = mgr0.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
    assert_eq!(mgr0.store().used_bytes(), 0);
    assert_eq!(mgr0.tier(SpillKey::full(1)), BlockTier::None);
    assert_eq!(mgr0.stats().over_budget_mem_drops, 1);
}

#[test]
fn atomic_working_set_reservation_prevents_partial_restore_thrashing() {
    let scratch = ScratchTestDir::new("atomic-reserve");
    let dir = scratch.path().join("spill");
    let f_stride = 2048;
    let w_stride = 4096;
    // Pool holds 4 Full slots and 3 Window slots.
    let pool = HostBlockPoolBackend::new(4, 3, f_stride, w_stride);
    let mut store = SpillStore::open(&dir, 64 * 1024 * 1024).unwrap();

    // Pre-populate 6 Full blocks and 4 Window blocks on SSD (simulating spilled prefixes).
    for id in 1..=6u64 {
        let k = SpillKey::full(id);
        store.write_block(k, 1, &deterministic_kv(k, f_stride), None, &HashSet::new()).unwrap();
    }
    for id in 1..=4u64 {
        let k = SpillKey::window(id);
        store.write_block(k, 1, &deterministic_kv(k, w_stride), None, &HashSet::new()).unwrap();
    }

    let mut mgr = SpillManager::new(pool, store);
    mgr.index_existing_disk_blocks();

    // Put one cold Full block in memory (refs == 0) so we verify asymmetric WouldBlock
    // triggers ZERO spills when Window slots cannot fit!
    let cold_f = SpillKey::full(100);
    let sf = mgr.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
    mgr.pool_mut().write_and_seal(sf, &deterministic_kv(cold_f, f_stride)).unwrap();
    mgr.register_sealed_block(cold_f, sf, None).unwrap();

    // Request exceeding capacity returns ExceedsCapacity with zero mutations.
    assert!(matches!(
        mgr.reserve_working_set(5, 1),
        Err(ReserveError::ExceedsCapacity { .. })
    ));
    assert_eq!(mgr.tier(cold_f), BlockTier::Memory);

    // Lane A reserves (3 full, 2 window) atomically -> succeeds!
    let lane_a_full = [SpillKey::full(1), SpillKey::full(2), SpillKey::full(3)];
    let lane_a_win = [SpillKey::window(1), SpillKey::window(2)];
    let prep_a = mgr.prepare_working_set_restore(&lane_a_full, &lane_a_win).unwrap();
    assert_eq!(prep_a.reservation.full_count(), 3);
    assert_eq!(prep_a.reservation.window_count(), 2);
    // 3 free Full slots were enough without even spilling cold_f!
    assert_eq!(mgr.tier(cold_f), BlockTier::Memory);
    assert_eq!(mgr.stats().lazy_spills_written, 0);

    // Now Lane B wants (3 full, 2 window) while Lane A is in Restoring!
    // Only 1 Full (cold_f) and 1 Window remain -> MUST fail atomically with WouldBlock,
    // spilling ZERO blocks (cold_f stays untouched in Memory!) and NEVER evicting Lane A!
    let lane_b_full = [SpillKey::full(4), SpillKey::full(5), SpillKey::full(6)];
    let lane_b_win = [SpillKey::window(3), SpillKey::window(4)];
    let err_b = mgr.prepare_working_set_restore(&lane_b_full, &lane_b_win).unwrap_err();
    assert!(matches!(
        err_b,
        PrepareRestoreError::Reserve(ReserveError::WouldBlock {
            need_full: 3,
            avail_full: 1,
            need_window: 2,
            avail_window: 1,
        })
    ));
    assert_eq!(mgr.tier(cold_f), BlockTier::Memory, "WouldBlock must not spill cold_f");
    assert_eq!(mgr.stats().lazy_spills_written, 0);
    assert_eq!(mgr.stats().reservations_would_block, 1);

    // Asymmetric WouldBlock: Lane C wants (1 full, 2 window). Full fits (1 <= 1),
    // Window does NOT (2 > 1). MUST NOT spill cold_f!
    let err_c = mgr.reserve_working_set(1, 2).unwrap_err();
    assert!(matches!(err_c, ReserveError::WouldBlock { .. }));
    assert_eq!(mgr.tier(cold_f), BlockTier::Memory, "asymmetric WouldBlock must not spill Full");

    // Lane A completes its SSD reads with zero interference.
    let res_a = mgr.execute_working_set_restore(prep_a).unwrap();
    assert_eq!(res_a.full_slots.len(), 3);
    assert_eq!(res_a.window_slots.len(), 2);
    for &(k, s) in &res_a.full_slots {
        assert_eq!(mgr.pool().read_slot_bytes(s).unwrap(), deterministic_kv(k, f_stride).as_slice());
    }
    for &(k, s) in &res_a.window_slots {
        assert_eq!(mgr.pool().read_slot_bytes(s).unwrap(), deterministic_kv(k, w_stride).as_slice());
    }

    // Lane A finishes and unpins its blocks -> they stay hot in memory (`Both`, refs=0).
    for &(k, _) in res_a.full_slots.iter().chain(res_a.window_slots.iter()) {
        assert_eq!(mgr.unpin(k).unwrap(), 0);
    }

    // Now Lane B retries: its full working set fits via lazy eviction of cold blocks!
    let prep_b = mgr.prepare_working_set_restore(&lane_b_full, &lane_b_win).unwrap();
    let res_b = mgr.execute_working_set_restore(prep_b).unwrap();
    for &(k, s) in &res_b.full_slots {
        assert_eq!(mgr.pool().read_slot_bytes(s).unwrap(), deterministic_kv(k, f_stride).as_slice());
    }
    for &(k, s) in &res_b.window_slots {
        assert_eq!(mgr.pool().read_slot_bytes(s).unwrap(), deterministic_kv(k, w_stride).as_slice());
    }
}

#[test]
fn restore_never_evicts_its_own_in_memory_working_set() {
    let scratch = ScratchTestDir::new("self-pin");
    let dir = scratch.path().join("spill");
    let stride = 1024;
    // Pool has capacity 2 Full slots.
    let pool = HostBlockPoolBackend::new(2, 1, stride, stride);
    let mut store = SpillStore::open(&dir, 1024 * 1024).unwrap();
    // Two blocks on disk: full(2), full(3).
    for id in [2u64, 3] {
        let k = SpillKey::full(id);
        store.write_block(k, 1, &deterministic_kv(k, stride), None, &HashSet::new()).unwrap();
    }
    let mut mgr = SpillManager::new(pool, store);
    mgr.index_existing_disk_blocks();

    // full(1) is in memory with refs == 0, and full(99) is in memory with refs == 1 (pinned by another lane).
    let k1 = SpillKey::full(1);
    let s1 = mgr.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
    mgr.pool_mut().write_and_seal(s1, &deterministic_kv(k1, stride)).unwrap();
    mgr.register_sealed_block(k1, s1, None).unwrap();

    let k99 = SpillKey::full(99);
    let s99 = mgr.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
    mgr.pool_mut().write_and_seal(s99, &deterministic_kv(k99, stride)).unwrap();
    mgr.register_sealed_block(k99, s99, None).unwrap();
    mgr.pin(k99).unwrap();

    // A lane wants working set [full(1) (in memory, refs=0), full(2) (on disk)].
    // Total slots needed = 2, but full(99) holds 1 slot -> MUST WouldBlock and MUST NOT
    // spill full(1) to make room for full(2)!
    let err = mgr.prepare_working_set_restore(&[k1, SpillKey::full(2)], &[]).unwrap_err();
    assert!(matches!(err, PrepareRestoreError::Reserve(ReserveError::WouldBlock { .. })));
    assert_eq!(mgr.tier(k1), BlockTier::Memory);
    assert_eq!(mgr.pool().refs(s1).unwrap(), 0, "failed prepare must unpin full(1)");

    // Once full(99) unpins, [full(1), full(2)] succeeds by spilling full(99), keeping full(1) resident!
    mgr.unpin(k99).unwrap();
    let prep = mgr.prepare_working_set_restore(&[k1, SpillKey::full(2)], &[]).unwrap();
    let res = mgr.execute_working_set_restore(prep).unwrap();
    assert_eq!(res.full_slots.len(), 2);
    assert_eq!(mgr.tier(k1), BlockTier::Memory);
    assert_eq!(mgr.tier(SpillKey::full(2)), BlockTier::Both);
    assert_eq!(mgr.tier(k99), BlockTier::Disk);
}

#[test]
fn publication_preserves_an_unindexed_file_at_the_destination() {
    let scratch = ScratchTestDir::new("unindexed-destination");
    let mut store = SpillStore::open(scratch.path(), 4096).unwrap();
    let key = SpillKey::full(0xab);
    let destination = store.block_path(key);
    fs::write(&destination, b"keep").unwrap();
    let result = store.write_block(key, 1, b"new KV", None, &HashSet::new());
    assert!(matches!(result, Err(SpillError::Io(err)) if err.kind() == io::ErrorKind::AlreadyExists));
    assert_eq!(fs::read(&destination).unwrap(), b"keep");
    assert_eq!(store.entry_count(), 0);
}

#[test]
fn spill_file_names_require_canonical_lowercase_hex_fields() {
    for key in [SpillKey::full(0), SpillKey::window(u64::MAX)] {
        assert_eq!(SpillKey::parse_blk_filename(&key.blk_filename()), Some(key));
        assert_eq!(SpillKey::parse_tmp_filename(&key.tmp_filename(u32::MAX, u64::MAX)), Some(key));
    }
    for name in ["blk-f-00000000000000AB.blk", "blk-f-+000000000000001.blk"] {
        assert_eq!(SpillKey::parse_blk_filename(name), None, "{name}");
    }
    for name in [
        "blk-f-0000000000000001.tmp.0000000A.0000000000000001",
        "blk-f-0000000000000001.tmp.00000001.000000000000000A",
    ] {
        assert_eq!(SpillKey::parse_tmp_filename(name), None, "{name}");
    }
}

#[test]
fn corrupt_torn_bitflip_zero_and_orphan_files_cause_clean_cache_miss_never_corruption() {
    let scratch = ScratchTestDir::new("corruption");
    let dir = scratch.path().join("spill");
    let kv_len = 2048usize;
    let state_len = 512usize;

    // Only a spill writer's exact temporary names are eligible for cleanup.
    prepare_spill_dir(&dir).unwrap();
    let orphan1 = dir.join(SpillKey::full(1).tmp_filename(0x1234, 1));
    let orphan2 = dir.join(SpillKey::window(0x99).tmp_filename(0x5678, 2));
    fs::write(&orphan1, b"half-written-tmp-data").unwrap();
    fs::write(&orphan2, b"").unwrap();
    let unrelated = [
        "notes.tmp.backup",
        "blk-x-0000000000000001.tmp.00001234.0000000000000001",
        "blk-f-00000000000000AB.tmp.00001234.0000000000000001",
        "blk-f-0000000000000001.tmp.1234.0000000000000001",
        "blk-f-0000000000000001.tmp.00001234.000000000000000g",
        "blk-f-0000000000000001.tmp.00001234.0000000000000001.bak",
    ];
    for name in unrelated {
        fs::write(dir.join(name), b"keep").unwrap();
    }

    let mut store = SpillStore::open(&dir, 64 * 1024 * 1024).unwrap();
    assert_eq!(store.stats().orphans_swept, 2);
    assert!(!orphan1.exists());
    assert!(!orphan2.exists());
    for name in unrelated {
        assert_eq!(fs::read(dir.join(name)).unwrap(), b"keep", "{name}");
    }

    let write_good = |s: &mut SpillStore, id: u64| -> (SpillKey, PathBuf) {
        let key = SpillKey::full(id);
        let kv = deterministic_kv(key, kv_len);
        let st = deterministic_state(key, state_len);
        s.write_block(key, 3, &kv, Some(&st), &HashSet::new()).unwrap();
        let p = s.block_path(key);
        (key, p)
    };

    // 1. Zero-byte .blk file.
    let (k1, p1) = write_good(&mut store, 1);
    fs::write(&p1, b"").unwrap();
    assert_eq!(store.read_block(k1, Some(kv_len)), ReadOutcome::Miss(MissReason::TruncatedHeader));
    assert!(!p1.exists(), "corrupt file should be unlinked");

    // 2. Truncated header (40 bytes).
    let (k2, p2) = write_good(&mut store, 2);
    let raw = fs::read(&p2).unwrap();
    fs::write(&p2, &raw[..40]).unwrap();
    assert_eq!(store.read_block(k2, Some(kv_len)), ReadOutcome::Miss(MissReason::TruncatedHeader));

    // 3. Torn K/V payload (1 byte short & half-written).
    let (k3, p3) = write_good(&mut store, 3);
    let raw = fs::read(&p3).unwrap();
    fs::write(&p3, &raw[..SPILL_HEADER_LEN + kv_len / 2]).unwrap();
    assert_eq!(store.read_block(k3, Some(kv_len)), ReadOutcome::Miss(MissReason::SizeMismatch));

    // 4. Torn recurrent state (1 byte short).
    let (k4, p4) = write_good(&mut store, 4);
    let raw = fs::read(&p4).unwrap();
    fs::write(&p4, &raw[..raw.len() - 1]).unwrap();
    assert_eq!(store.read_block(k4, Some(kv_len)), ReadOutcome::Miss(MissReason::SizeMismatch));

    // 5. Trailing garbage appended.
    let (k5, p5) = write_good(&mut store, 5);
    let mut raw = fs::read(&p5).unwrap();
    raw.push(0xAA);
    fs::write(&p5, &raw).unwrap();
    assert_eq!(store.read_block(k5, Some(kv_len)), ReadOutcome::Miss(MissReason::SizeMismatch));

    // 6. Single-bit flips across header, digest, K/V payload, and recurrent state.
    let flip_offsets = [
        (0usize, MissReason::BadMagic),
        (8, MissReason::UnsupportedVersion),
        (12, MissReason::BadBlockKind),
        (14, MissReason::KvbMismatch),
        (16, MissReason::MetadataMismatch),
        (24, MissReason::ChecksumMismatch), // generation bit-flip
        (48, MissReason::ChecksumMismatch), // sha256 byte flip
        (SPILL_HEADER_LEN, MissReason::ChecksumMismatch), // first K/V byte
        (SPILL_HEADER_LEN + kv_len / 2, MissReason::ChecksumMismatch), // mid K/V byte
        (SPILL_HEADER_LEN + kv_len - 1, MissReason::ChecksumMismatch), // last K/V byte
        (SPILL_HEADER_LEN + kv_len + state_len / 2, MissReason::ChecksumMismatch), // recurrent state byte
    ];
    for (i, &(off, expected_reason)) in flip_offsets.iter().enumerate() {
        let (k, p) = write_good(&mut store, 100 + i as u64);
        let mut raw = fs::read(&p).unwrap();
        raw[off] ^= 0x04;
        fs::write(&p, &raw).unwrap();
        assert_eq!(
            store.read_block(k, Some(kv_len)),
            ReadOutcome::Miss(expected_reason),
            "bit flip at offset {off} must return {expected_reason:?}"
        );
        assert!(!p.exists(), "corrupt file at offset {off} must be unlinked");
    }

    // 7. File-swap attack: copy valid block 200 over block 201.
    let (k200, p200) = write_good(&mut store, 200);
    let (k201, p201) = write_good(&mut store, 201);
    fs::copy(&p200, &p201).unwrap();
    assert_eq!(store.read_block(k201, Some(kv_len)), ReadOutcome::Miss(MissReason::MetadataMismatch));
    // Original block 200 remains intact.
    match store.read_block(k200, Some(kv_len)) {
        ReadOutcome::Hit(blk) => {
            assert_eq!(blk.kv_bytes, deterministic_kv(k200, kv_len));
            assert_eq!(blk.recurrent_state, Some(deterministic_state(k200, state_len)));
        }
        ReadOutcome::Miss(r) => panic!("block 200 should be intact, got {r:?}"),
    }

    // 8. End-to-end SpillManager rollback on corrupt block in multi-block working set:
    //    pool slots must remain completely clean (zero-filled, zero leaks)!
    let pool = HostBlockPoolBackend::new(3, 1, kv_len, kv_len);
    drop(store);
    let store2 = SpillStore::open(&dir, 64 * 1024 * 1024).unwrap();
    let mut mgr = SpillManager::new(pool, store2);
    // Write two blocks via store and corrupt the second.
    let ka = SpillKey::full(300);
    let kb = SpillKey::full(301);
    mgr.store.write_block(ka, 1, &deterministic_kv(ka, kv_len), None, &HashSet::new()).unwrap();
    mgr.store.write_block(kb, 1, &deterministic_kv(kb, kv_len), None, &HashSet::new()).unwrap();
    let pb = mgr.store.block_path(kb);
    let mut raw_b = fs::read(&pb).unwrap();
    raw_b[SPILL_HEADER_LEN + 7] ^= 0x80;
    fs::write(&pb, &raw_b).unwrap();
    mgr.index_existing_disk_blocks();

    let prep = mgr.prepare_working_set_restore(&[ka, kb], &[]).unwrap();
    let miss = mgr.execute_working_set_restore(prep).unwrap_err();
    assert_eq!(miss.failed_key, kb);
    assert_eq!(miss.reason, MissReason::ChecksumMismatch);
    // Zero leaked slots, zero active reservations, and every pool slot is zeroed!
    assert_eq!(mgr.active_reservation_count(), 0);
    assert_eq!(mgr.pool().used_slots(SpillBlockKind::Full), 0);
    assert_eq!(mgr.pool().free_slots(SpillBlockKind::Full), 3);
    for slot_i in 1..=3u32 {
        assert!(
            mgr.pool().raw_slot_bytes(SpillBlockKind::Full, slot_i).iter().all(|&b| b == 0),
            "slot {slot_i} must remain zero-filled after corrupt-block rollback"
        );
    }
}

/// Child helper mode spawned by [`kill9_mid_write_loses_cache_never_correctness`].
/// Marked `#[ignore]` so normal `cargo test` runs do not invoke it directly;
/// returns immediately if `INKY_SPILL_KILL9_DIR` is unset.
#[test]
#[ignore = "internal child helper spawned by kill9_mid_write_loses_cache_never_correctness"]
fn kill9_writer_helper() {
    let Ok(dir) = std::env::var("INKY_SPILL_KILL9_DIR") else {
        return;
    };
    let round: u64 = std::env::var("INKY_SPILL_KILL9_ROUND")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let kv_len: usize = std::env::var("INKY_SPILL_KILL9_KV_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256 * 1024);
    let state_len: usize = std::env::var("INKY_SPILL_KILL9_STATE_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32 * 1024);

    let mut store = SpillStore::open(&dir, 512 * 1024 * 1024).expect("child open spill store");
    let empty = HashSet::new();

    // Write continuously until parent sends SIGKILL.
    let mut seq = 0u64;
    loop {
        let id = (round << 32) | seq;
        let kind = if seq.is_multiple_of(4) {
            SpillBlockKind::Window
        } else {
            SpillBlockKind::Full
        };
        let key = SpillKey::new(kind, id);
        let kv = deterministic_kv(key, kv_len);
        let st = if seq.is_multiple_of(3) {
            Some(deterministic_state(key, state_len))
        } else {
            None
        };
        let _ = store.write_block(key, seq + 1, &kv, st.as_deref(), &empty);
        if seq == 0 {
            println!("READY");
            let _ = io::stdout().flush();
        } else {
            println!("COMMIT {seq}");
            let _ = io::stdout().flush();
        }
        seq = seq.wrapping_add(1);
        if seq >= 500 {
            break;
        }
    }
}

/// Layer 4 Gate (`docs/ROADMAP.md:140`):
/// *"Gate: kill -9 in mid-write loses cache and never correctness"*
///
/// Spawns real writer subprocesses that continuously spill blocks to SSD while
/// the parent delivers `SIGKILL` mid-write (both targeted on observed `.tmp.*`
/// files and at staggered intervals), then verifies after every crash:
/// - 100% of orphan `.tmp.*` files are swept on recovery;
/// - 100% of surviving `.blk` files either verify SHA-256 and match ground-truth
///   K/V + recurrent state bit-for-bit, or return `CacheMiss` and fall back to
///   ground-truth recomputation with bit-identical outputs.
#[cfg(unix)]
#[test]
#[ignore = "spawns SIGKILL writer subprocesses; run explicitly in CI unit job"]
fn kill9_mid_write_loses_cache_never_correctness() {
    use std::os::unix::process::ExitStatusExt;

    let scratch = ScratchTestDir::new("kill9-gate");
    let exe = std::env::current_exe().expect("current test binary path");
    let kv_len = 256 * 1024usize; // 256 KiB per block so writes take real SSD I/O time
    let state_len = 32 * 1024usize;

    let mut total_hits = 0u64;
    let mut total_misses = 0u64;
    let mut total_orphans_swept = 0u64;
    let mut total_kills_verified = 0u64;

    let rounds = 10u64;
    for round in 0..rounds {
        let dir = scratch.path().join(format!("round-{round}"));
        prepare_spill_dir(&dir).unwrap();

        let mut child = Command::new(&exe)
            .args([
                "--exact",
                "spill::tests::kill9_writer_helper",
                "--ignored",
                "--nocapture",
            ])
            .env("INKY_SPILL_KILL9_DIR", &dir)
            .env("INKY_SPILL_KILL9_ROUND", round.to_string())
            .env("INKY_SPILL_KILL9_KV_LEN", kv_len.to_string())
            .env("INKY_SPILL_KILL9_STATE_LEN", state_len.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn kill9 writer helper");

        let child_pid = child.id() as libc::pid_t;
        let stdout = child.stdout.take().expect("child stdout");

        // Wait for child to finish block 0 and enter continuous write loop.
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        let mut saw_ready = false;
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    if line.trim() == "READY" {
                        saw_ready = true;
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        assert!(saw_ready, "round {round}: child writer did not report READY");

        if round % 2 == 0 {
            // Active mid-write hunt: poll spill dir until a `.tmp.*` file is in flight,
            // then immediately fire SIGKILL right in mid-write!
            let hunt_deadline = Instant::now() + Duration::from_millis(250);
            while Instant::now() < hunt_deadline {
                let has_tmp = fs::read_dir(&dir)
                    .ok()
                    .map(|rd| {
                        rd.filter_map(Result::ok)
                            .any(|e| e.file_name().to_str().and_then(SpillKey::parse_tmp_filename).is_some())
                    })
                    .unwrap_or(false);
                if has_tmp {
                    break;
                }
                std::thread::yield_now();
            }
        } else {
            // Staggered asynchronous kill during active write loop.
            std::thread::sleep(Duration::from_millis(2 + (round * 3) % 17));
        }

        // Deliver genuine SIGKILL (`kill -9`)!
        let rc = unsafe { libc::kill(child_pid, libc::SIGKILL) };
        assert_eq!(rc, 0, "round {round}: SIGKILL failed");
        let status = child.wait().expect("wait for killed child");
        assert_eq!(
            status.signal(),
            Some(libc::SIGKILL),
            "round {round}: child should terminate from SIGKILL, got {status:?}"
        );
        total_kills_verified += 1;

        // Recover with a fresh SpillStore::open().
        let mut store = SpillStore::open(&dir, 512 * 1024 * 1024).expect("recover spill store");
        total_orphans_swept += store.stats().orphans_swept;

        // Assert ZERO orphan `.tmp.*` files remain after startup sweep.
        let remaining_tmps: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_str().and_then(SpillKey::parse_tmp_filename).is_some())
            .collect();
        assert!(
            remaining_tmps.is_empty(),
            "round {round}: orphan tmp files remained after open(): {remaining_tmps:?}"
        );

        // Check every block ID in the range 0..max_seen+8 (including in-flight/unwritten tail).
        let max_id = store
            .keys()
            .map(|k| k.id & 0xffff_ffff)
            .max()
            .unwrap_or(0);
        let check_limit = (max_id + 8).min(500);

        for seq in 0..=check_limit {
            let id = (round << 32) | seq;
            let kind = if seq % 4 == 0 {
                SpillBlockKind::Window
            } else {
                SpillBlockKind::Full
            };
            let key = SpillKey::new(kind, id);
            let expected_kv = deterministic_kv(key, kv_len);
            let expected_state = if seq % 3 == 0 {
                Some(deterministic_state(key, state_len))
            } else {
                None
            };

            let (recovered_kv, recovered_state) = match store.read_block(key, Some(kv_len)) {
                ReadOutcome::Hit(blk) => {
                    total_hits += 1;
                    (blk.kv_bytes, blk.recurrent_state)
                }
                ReadOutcome::Miss(_) => {
                    total_misses += 1;
                    // Clean fallback to recomputation from ground truth.
                    (deterministic_kv(key, kv_len), expected_state.clone())
                }
            };

            assert_eq!(
                recovered_kv, expected_kv,
                "round {round} seq {seq}: K/V bytes must be bit-identical"
            );
            assert_eq!(
                recovered_state, expected_state,
                "round {round} seq {seq}: recurrent state must be bit-identical"
            );
        }
    }

    assert_eq!(total_kills_verified, rounds);
    assert!(total_hits >= rounds, "expected committed hits across SIGKILL rounds, got {total_hits}");
    assert!(total_misses > 0, "expected cache misses on interrupted tail, got {total_misses}");
    // Print concise receipt when run with --nocapture.
    eprintln!(
        "[kill9-gate] rounds={rounds} kills={total_kills_verified} hits={total_hits} misses={total_misses} orphans_swept={total_orphans_swept}"
    );
}

#[test]
fn pending_restore_blocks_are_protected_from_disk_gc() {
    let scratch = ScratchTestDir::new("io-pin-gc");
    let dir = scratch.path().join("spill");
    let stride = 2048usize;
    let one_file = (SPILL_HEADER_LEN + stride) as u64;
    // Disk holds at most 2 blocks.
    let mut store = SpillStore::open(&dir, one_file * 2).unwrap();
    let k1 = SpillKey::full(1);
    let k2 = SpillKey::full(2);
    store.write_block(k1, 1, &deterministic_kv(k1, stride), None, &HashSet::new()).unwrap();
    store.write_block(k2, 1, &deterministic_kv(k2, stride), None, &HashSet::new()).unwrap();

    // Pool holds 2 Full slots.
    let pool = HostBlockPoolBackend::new(2, 1, stride, stride);
    let mut mgr = SpillManager::new(pool, store);
    mgr.index_existing_disk_blocks();

    // Put a cold block k3 in memory (refs == 0), leaving 1 free slot.
    let k3 = SpillKey::full(3);
    let s3 = mgr.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
    mgr.pool_mut().write_and_seal(s3, &deterministic_kv(k3, stride)).unwrap();
    mgr.register_sealed_block(k3, s3, None).unwrap();

    // Lane A prepares to restore k1 (the OLDEST on-disk block!).
    // k1 now has io_pins == 1 and 1 free slot is reserved.
    let prep_a = mgr.prepare_working_set_restore(&[k1], &[]).unwrap();

    // Now another allocation triggers lazy spill of k3 to disk!
    // Disk budget is 2 blocks: k1 is older than k2, BUT k1 is io-pinned by prep_a,
    // so disk GC MUST evict k2 and MUST preserve k1 on disk!
    let _s4 = mgr.alloc_slot_lazy(SpillBlockKind::Full).unwrap();
    assert!(mgr.store().contains(k1), "io-pinned block k1 must survive disk GC");
    assert!(!mgr.store().contains(k2), "unpinned block k2 should be evicted by disk GC");
    assert!(mgr.store().contains(k3), "spilled block k3 should be on disk");

    // Lane A executes its restore of k1 cleanly with zero miss!
    let res_a = mgr.execute_working_set_restore(prep_a).unwrap();
    assert_eq!(res_a.full_slots.len(), 1);
    assert_eq!(
        mgr.pool().read_slot_bytes(res_a.full_slots[0].1).unwrap(),
        deterministic_kv(k1, stride).as_slice()
    );
}

#[test]
fn concurrent_restores_of_shared_disk_block_deduplicate_without_slot_leak() {
    let scratch = ScratchTestDir::new("dedup-restore");
    let dir = scratch.path().join("spill");
    let stride = 2048usize;
    let mut store = SpillStore::open(&dir, 64 * 1024 * 1024).unwrap();
    let shared = SpillKey::full(77);
    let state = deterministic_state(shared, 512);
    store
        .write_block(shared, 1, &deterministic_kv(shared, stride), Some(&state), &HashSet::new())
        .unwrap();

    // Pool has 2 Full slots so two lanes can both reserve concurrently.
    let pool = HostBlockPoolBackend::new(2, 1, stride, stride);
    let mut mgr = SpillManager::new(pool, store);
    mgr.index_existing_disk_blocks();

    let prep_a = mgr.prepare_working_set_restore(&[shared], &[]).unwrap();
    let prep_b = mgr.prepare_working_set_restore(&[shared], &[]).unwrap();
    assert_eq!(mgr.pool().used_slots(SpillBlockKind::Full), 2);

    // Lane A finishes restore first -> installs `shared` into slot A (refs = 1).
    let res_a = mgr.execute_working_set_restore(prep_a).unwrap();
    let slot_a = res_a.full_slots[0].1;

    // Lane B finishes restore second -> MUST deduplicate onto slot A (refs = 2)
    // and release its redundant pre-reserved slot so used_slots drops back to 1!
    let res_b = mgr.execute_working_set_restore(prep_b).unwrap();
    let slot_b = res_b.full_slots[0].1;
    assert_eq!(slot_a, slot_b, "concurrent restores of same block must share one slot");
    assert_eq!(mgr.pool().refs(slot_a).unwrap(), 2);
    assert_eq!(mgr.pool().used_slots(SpillBlockKind::Full), 1);
    assert_eq!(mgr.pool().free_slots(SpillBlockKind::Full), 1);
    assert_eq!(res_b.recurrent_states.get(&shared), Some(&state));

    // Unpin both lanes cleanly.
    assert_eq!(mgr.unpin(shared).unwrap(), 1);
    assert_eq!(mgr.unpin(shared).unwrap(), 0);
}

#[test]
fn full_scale_block_and_recurrent_state_roundtrip() {
    const FULL_STRIDE: usize = 2 * 1024 * 1024;
    let scratch = ScratchTestDir::new("full-scale");
    let dir = scratch.path().join("spill");
    // Real Layer 3 Full block (2 MiB) + Inkling 2 Mini recurrent state (~3.66 MiB).
    let cs_bytes = crate::engine::Engine::cs_image_bytes();
    let mut store = SpillStore::open(&dir, 32 * 1024 * 1024).unwrap();
    let key = SpillKey::full(4242);
    let kv = deterministic_kv(key, FULL_STRIDE);
    let st = deterministic_state(key, cs_bytes);

    let (out, evicted) = store
        .write_block(key, 9, &kv, Some(&st), &HashSet::new())
        .unwrap();
    assert!(evicted.is_empty());
    assert_eq!(
        out,
        WriteOutcome::Written {
            on_disk_bytes: (SPILL_HEADER_LEN + FULL_STRIDE + cs_bytes) as u64
        }
    );

    match store.read_block(key, Some(FULL_STRIDE)) {
        ReadOutcome::Hit(blk) => {
            assert_eq!(blk.key, key);
            assert_eq!(blk.generation, 9);
            assert_eq!(blk.kv_bytes, kv);
            assert_eq!(blk.recurrent_state.as_deref(), Some(st.as_slice()));
        }
        ReadOutcome::Miss(r) => panic!("expected full-scale hit, got {r:?}"),
    }
}

#[test]
fn kvb_mismatch_between_kv8_and_fp16_is_rejected_without_exposing_bytes() {
    let scratch = ScratchTestDir::new("kvb-mismatch");
    let dir = scratch.path().join("spill");

    // Reject invalid kvb widths at open time.
    assert!(matches!(SpillStore::open_with_kvb(&dir, 64 * 1024 * 1024, 0), Err(SpillError::InvalidKvb(0))));
    assert!(matches!(SpillStore::open_with_kvb(&dir, 64 * 1024 * 1024, 3), Err(SpillError::InvalidKvb(3))));

    let kv_len = 4096usize;
    let cs_len = 256usize;
    let k_kv8 = SpillKey::full(77);
    let k_cs_kv8 = SpillKey::full(78);

    // 1. Write K/V and recurrent-state-only blocks from a --kv8 (kvb=1) server.
    {
        let mut kv8_store = SpillStore::open_with_kvb(&dir, 64 * 1024 * 1024, 1).unwrap();
        assert_eq!(kv8_store.kvb(), 1);
        kv8_store
            .write_block(k_kv8, 1, &deterministic_kv(k_kv8, kv_len), None, &HashSet::new())
            .unwrap();
        kv8_store
            .write_block(k_cs_kv8, 1, &[], Some(&deterministic_state(k_cs_kv8, cs_len)), &HashSet::new())
            .unwrap();
        assert!(matches!(kv8_store.read_block(k_kv8, Some(kv_len)), ReadOutcome::Hit(_)));
        assert!(matches!(kv8_store.read_block(k_cs_kv8, Some(0)), ReadOutcome::Hit(_)));
    }

    // 2. Reopen as fp16 (kvb=2): must reject BOTH K/V and state-only blocks even if expected_kv_len is None/0!
    {
        let mut fp16_store = SpillStore::open_with_kvb(&dir, 64 * 1024 * 1024, 2).unwrap();
        assert_eq!(fp16_store.kvb(), 2);
        assert_eq!(fp16_store.read_block(k_kv8, None), ReadOutcome::Miss(MissReason::KvbMismatch));
        assert!(!fp16_store.block_path(k_kv8).exists());
        assert_eq!(fp16_store.read_block(k_cs_kv8, Some(0)), ReadOutcome::Miss(MissReason::KvbMismatch));
        assert!(!fp16_store.block_path(k_cs_kv8).exists());

        // Write an fp16 block (kvb=2) now.
        let k_fp16 = SpillKey::window(88);
        fp16_store
            .write_block(k_fp16, 2, &deterministic_kv(k_fp16, kv_len), None, &HashSet::new())
            .unwrap();
    }

    // 3. Reopen as --kv8 (kvb=1): must reject the fp16 block!
    let mut kv8_store = SpillStore::open_with_kvb(&dir, 64 * 1024 * 1024, 1).unwrap();
    let k_fp16 = SpillKey::window(88);
    assert_eq!(kv8_store.read_block(k_fp16, Some(kv_len)), ReadOutcome::Miss(MissReason::KvbMismatch));
    assert!(!kv8_store.block_path(k_fp16).exists());
}

#[test]
fn evict_one_lru_cold_releases_io_pin_when_pool_returns_error() {
    struct FailOncePool {
        inner: HostBlockPoolBackend,
        fail_read: bool,
    }
    impl SpillPoolBackend for FailOncePool {
        fn stride_bytes(&self, kind: SpillBlockKind) -> usize { self.inner.stride_bytes(kind) }
        fn capacity_slots(&self, kind: SpillBlockKind) -> usize { self.inner.capacity_slots(kind) }
        fn used_slots(&self, kind: SpillBlockKind) -> usize { self.inner.used_slots(kind) }
        fn free_slots(&self, kind: SpillBlockKind) -> usize { self.inner.free_slots(kind) }
        fn read_slot_bytes(&self, slot: ResidentSlot) -> Result<&[u8], SpillError> {
            if self.fail_read {
                return Err(SpillError::Pool("injected read failure".into()));
            }
            self.inner.read_slot_bytes(slot)
        }
        fn take_free_slot(&mut self, kind: SpillBlockKind) -> Result<ResidentSlot, SpillError> {
            self.inner.take_free_slot(kind)
        }
        fn release_cold_slot(&mut self, slot: ResidentSlot) -> Result<(), SpillError> {
            self.inner.release_cold_slot(slot)
        }
        fn install_restored_slot(&mut self, slot: ResidentSlot, kv_bytes: &[u8]) -> Result<(), SpillError> {
            self.inner.install_restored_slot(slot, kv_bytes)
        }
        fn add_ref(&mut self, slot: ResidentSlot) -> Result<u32, SpillError> { self.inner.add_ref(slot) }
        fn drop_ref(&mut self, slot: ResidentSlot) -> Result<u32, SpillError> { self.inner.drop_ref(slot) }
        fn refs(&self, slot: ResidentSlot) -> Result<u32, SpillError> { self.inner.refs(slot) }
    }

    let scratch = ScratchTestDir::new("evict-io-pin-rollback");
    let store = SpillStore::open(scratch.path(), 64 * 1024 * 1024).unwrap();
    let pool = FailOncePool { inner: HostBlockPoolBackend::new(1, 1, 64, 64), fail_read: false };
    let mut mgr = SpillManager::new(pool, store);
    let key = SpillKey::full(1);
    let slot = mgr.pool_mut().take_free_slot(SpillBlockKind::Full).unwrap();
    mgr.pool_mut().install_restored_slot(slot, &[7u8; 64]).unwrap();
    mgr.register_sealed_block(key, slot, None).unwrap();
    mgr.pool_mut().fail_read = true;

    // First eviction attempt fails during `read_slot_bytes`; `io_pins` must drop back to 0.
    assert!(mgr.evict_one_lru_cold(SpillBlockKind::Full).is_err());
    assert_eq!(mgr.blocks.get(&key).unwrap().io_pins, 0, "io_pins must not leak on pool error");

    // Once the transient failure clears, the block remains evictable and spills cleanly.
    mgr.pool_mut().fail_read = false;
    assert_eq!(mgr.evict_one_lru_cold(SpillBlockKind::Full).unwrap(), key);
    assert_eq!(mgr.blocks.get(&key).unwrap().io_pins, 0);
}
