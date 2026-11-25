//! Persistent spill files, startup recovery, and bounded SSD eviction.

use super::*;

// ---------------------------------------------------------------------------
// On-disk checksummed block store (SpillStore)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct DiskEntry {
    bytes: u64,
    lru_seq: u64,
}

/// Statistics tracked by [`SpillStore`].
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct SpillStoreStats {
    pub writes_committed: u64,
    pub bytes_written: u64,
    pub reads_hit: u64,
    pub reads_miss: u64,
    pub corrupt_files_removed: u64,
    pub disk_gc_evictions: u64,
    pub over_budget_drops: u64,
    pub orphans_swept: u64,
}

/// Crash-safe, bounded, checksummed SSD directory store (`--spill-dir`).
pub struct SpillStore {
    // Keeping the locked directory open gives its index, writers, and GC a single owner.
    directory_lock: File,
    last_allocated_id: u64,
    dir: PathBuf,
    max_bytes: u64,
    used_bytes: u64,
    lru_clock: u64,
    kvb: u32,
    entries: HashMap<SpillKey, DiskEntry>,
    stats: SpillStoreStats,
}

impl Drop for SpillStore {
    fn drop(&mut self) {
        // Release immediately even if a concurrent fork briefly inherited the descriptor.
        let _ = self.directory_lock.unlock();
    }
}

fn enable_nocache(file: &File) {
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        unsafe {
            libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = file;
    }
}

fn fsync_dir(dir: &Path) -> io::Result<()> {
    let f = File::open(dir)?;
    f.sync_all()
}

fn unlink_if_present(path: &Path) -> io::Result<bool> {
    match fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

fn pread_exact(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        while !buf.is_empty() {
            let n = file.read_at(buf, offset)?;
            if n == 0 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected EOF in pread"));
            }
            buf = &mut buf[n..];
            offset = offset.checked_add(n as u64).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "pread offset overflow")
            })?;
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = file;
        f.seek(SeekFrom::Start(offset))?;
        f.read_exact(buf)
    }
}

fn encode_header(
    key: SpillKey,
    kvb: u32,
    generation: u64,
    kv_bytes: &[u8],
    recurrent_state: Option<&[u8]>,
) -> [u8; SPILL_HEADER_LEN] {
    let state_slice = recurrent_state.unwrap_or(&[]);
    let mut header = [0u8; SPILL_HEADER_LEN];
    header[0..8].copy_from_slice(&SPILL_MAGIC);
    header[8..12].copy_from_slice(&SPILL_FORMAT_VERSION.to_le_bytes());
    header[12..14].copy_from_slice(&(key.kind as u16).to_le_bytes());
    header[14..16].copy_from_slice(&(kvb as u16).to_le_bytes());
    header[16..24].copy_from_slice(&key.id.to_le_bytes());
    header[24..32].copy_from_slice(&generation.to_le_bytes());
    header[32..40].copy_from_slice(&(kv_bytes.len() as u64).to_le_bytes());
    header[40..48].copy_from_slice(&(state_slice.len() as u64).to_le_bytes());

    let mut hasher = Sha256::new();
    hasher.update(&header[0..HEADER_PREFIX_LEN]);
    hasher.update(kv_bytes);
    if !state_slice.is_empty() {
        hasher.update(state_slice);
    }
    let digest = hasher.finalize();
    header[48..80].copy_from_slice(&digest);
    header
}

impl SpillStore {
    /// Prepare `--spill-dir` (mode `0700` on Unix), sweep orphan `.tmp.*` files,
    /// index surviving `.blk` files, and enforce `max_bytes` via LRU disk GC (default fp16 `kvb = 2`).
    pub fn open(dir: impl AsRef<Path>, max_bytes: u64) -> Result<Self, SpillError> {
        Self::open_with_kvb(dir, max_bytes, 2)
    }

    /// Prepare `--spill-dir` with an explicit KV element width `kvb` (`1` for `--kv8`, `2` for fp16).
    pub fn open_with_kvb(dir: impl AsRef<Path>, max_bytes: u64, kvb: u32) -> Result<Self, SpillError> {
        if kvb != 1 && kvb != 2 {
            return Err(SpillError::InvalidKvb(kvb));
        }
        let dir = dir.as_ref().to_path_buf();
        prepare_spill_dir(&dir)?;
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        options.custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW);
        let directory_lock = options.open(&dir)?;
        match directory_lock.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => {
                return Err(SpillError::InvalidDir(format!("{} is already in use", dir.display())));
            }
            Err(std::fs::TryLockError::Error(err)) => return Err(err.into()),
        }

        let mut store = Self {
            directory_lock,
            last_allocated_id: 1,
            dir,
            max_bytes,
            used_bytes: 0,
            lru_clock: 0,
            kvb,
            entries: HashMap::new(),
            stats: SpillStoreStats::default(),
        };
        store.scan_and_sweep()?;
        Ok(store)
    }

    pub fn kvb(&self) -> u32 {
        self.kvb
    }

    #[cfg(test)]
    pub(super) fn clone_directory_lock(&self) -> io::Result<File> {
        self.directory_lock.try_clone()
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    pub fn stats(&self) -> SpillStoreStats {
        self.stats
    }

    pub fn contains(&self, key: SpillKey) -> bool {
        self.entries.contains_key(&key)
    }

    /// Allocate an identity for a new checkpoint block, skipping surviving blocks
    /// whose content belongs to an earlier generation of the in-memory checkpoint index.
    pub fn fresh_key(&mut self, kind: SpillBlockKind) -> SpillKey {
        loop {
            self.last_allocated_id = self.last_allocated_id.wrapping_add(1).max(1);
            let key = SpillKey::new(kind, self.last_allocated_id);
            if !self.contains(key) {
                return key;
            }
        }
    }

    pub(super) fn keys(&self) -> impl Iterator<Item = SpillKey> + '_ {
        self.entries.keys().copied()
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn block_path(&self, key: SpillKey) -> PathBuf {
        self.dir.join(key.blk_filename())
    }

    fn next_lru(&mut self) -> u64 {
        self.lru_clock = self.lru_clock.wrapping_add(1);
        self.lru_clock
    }

    fn scan_and_sweep(&mut self) -> Result<(), SpillError> {
        let mut discovered: Vec<(SpillKey, u64, u64)> = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            let meta = match fs::symlink_metadata(&path) {
                Ok(m) => m,
                Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e.into()),
            };
            if SpillKey::parse_tmp_filename(&name).is_some()
                || name.starts_with(SPILL_INDEX_TMP_PREFIX)
            {
                if meta.is_file() && unlink_if_present(&path)? {
                    self.stats.orphans_swept += 1;
                }
                continue;
            }
            if let Some(key) = SpillKey::parse_blk_filename(&name) {
                if !meta.file_type().is_file() {
                    // Never trust symlinks or non-regular files in spill dir.
                    if unlink_if_present(&path)? {
                        self.stats.corrupt_files_removed += 1;
                    }
                    continue;
                }
                let mtime_key = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0);
                discovered.push((key, meta.len(), mtime_key));
            }
        }
        // Order surviving blocks oldest-mtime first so LRU clock preserves relative age.
        discovered.sort_by_key(|&(key, _, mtime)| (mtime, key));
        for (key, len, _) in discovered {
            self.last_allocated_id = self.last_allocated_id.max(key.id);
            let seq = self.next_lru();
            self.entries.insert(key, DiskEntry { bytes: len, lru_seq: seq });
            self.used_bytes = self.used_bytes.saturating_add(len);
        }
        if self.stats.orphans_swept > 0 {
            let _ = fsync_dir(&self.dir);
        }
        // Enforce disk budget on startup if existing files exceed max_bytes.
        let empty = HashSet::new();
        self.evict_lru_to_fit(0, &empty);
        if self.used_bytes > self.max_bytes {
            return Err(SpillError::InvalidDir(format!(
                "could not reduce spill usage of {} bytes to the {} byte budget",
                self.used_bytes, self.max_bytes
            )));
        }
        Ok(())
    }

    /// Evict least-recently-used `.blk` files not in `protected` until
    /// `self.used_bytes + additional <= self.max_bytes` (returns evicted keys).
    pub fn evict_lru_to_fit(
        &mut self,
        additional: u64,
        protected: &HashSet<SpillKey>,
    ) -> Vec<SpillKey> {
        let mut evicted = Vec::new();
        let mut failed = HashSet::new();
        while self.used_bytes.saturating_add(additional) > self.max_bytes {
            let victim = self
                .entries
                .iter()
                .filter(|(k, _)| !protected.contains(k) && !failed.contains(*k))
                .min_by_key(|(k, e)| (e.lru_seq, **k))
                .map(|(k, _)| *k);
            let Some(victim_key) = victim else {
                break;
            };
            if self.remove_block(victim_key) {
                self.stats.disk_gc_evictions += 1;
                evicted.push(victim_key);
            } else {
                failed.insert(victim_key);
            }
        }
        evicted
    }

    /// Remove a specific block from disk if present.
    pub fn remove_block(&mut self, key: SpillKey) -> bool {
        let indexed = self.entries.contains_key(&key);
        self.unlink_block(key).is_ok_and(|removed| removed || indexed)
    }

    fn forget_block(&mut self, key: SpillKey) {
        if let Some(entry) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(entry.bytes);
        }
    }

    fn unlink_block(&mut self, key: SpillKey) -> io::Result<bool> {
        let removed = unlink_if_present(&self.block_path(key))?;
        self.forget_block(key);
        Ok(removed)
    }

    /// Crash-safe atomic write of a block (K/V + optional start-point recurrent state):
    /// enforce `--max-spill-bytes` with LRU disk GC, write `<tmp>` (`0600`), `fsync`,
    /// atomic `rename` to `<id>.blk`, and `fsync` directory.
    ///
    /// Returns `(WriteOutcome, evicted_disk_keys)` so callers can update tier metadata
    /// for any cold blocks GC'd from disk.
    pub fn write_block(
        &mut self,
        key: SpillKey,
        generation: u64,
        kv_bytes: &[u8],
        recurrent_state: Option<&[u8]>,
        protected: &HashSet<SpillKey>,
    ) -> Result<(WriteOutcome, Vec<SpillKey>), SpillError> {
        if kv_bytes.len() > MAX_KV_PAYLOAD_BYTES {
            return Err(SpillError::PayloadTooLarge(kv_bytes.len()));
        }
        let state_slice = recurrent_state.unwrap_or(&[]);
        if state_slice.len() > MAX_RECURRENT_STATE_BYTES {
            return Err(SpillError::PayloadTooLarge(state_slice.len()));
        }
        let total_bytes = (SPILL_HEADER_LEN as u64)
            .checked_add(kv_bytes.len() as u64)
            .and_then(|v| v.checked_add(state_slice.len() as u64))
            .ok_or_else(|| SpillError::PayloadTooLarge(usize::MAX))?;

        if total_bytes > self.max_bytes {
            self.stats.over_budget_drops += 1;
            return Ok((WriteOutcome::OverBudget, Vec::new()));
        }

        // A case-insensitive filesystem can resolve a canonical block path to an
        // unrelated, noncanonical filename deliberately left out of the index.
        let final_path = self.block_path(key);
        let existing_bytes = match self.entries.get(&key) {
            Some(entry) => entry.bytes,
            None => match fs::symlink_metadata(&final_path) {
                Err(e) if e.kind() == io::ErrorKind::NotFound => 0,
                Err(e) => return Err(e.into()),
                Ok(_) => {
                    return Err(io::Error::new(io::ErrorKind::AlreadyExists,
                        format!("unindexed file at {}", final_path.display())).into());
                }
            },
        };
        let net_additional = total_bytes.saturating_sub(existing_bytes);
        let mut protected_with_self = protected.clone();
        protected_with_self.insert(key);
        let evicted = self.evict_lru_to_fit(net_additional, &protected_with_self);
        if self.used_bytes.saturating_sub(existing_bytes).saturating_add(total_bytes) > self.max_bytes {
            self.stats.over_budget_drops += 1;
            return Ok((WriteOutcome::OverBudget, evicted));
        }

        let header = encode_header(key, self.kvb, generation, kv_bytes, recurrent_state);
        let pid = std::process::id();
        let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let tmp_path = self.dir.join(key.tmp_filename(pid, nonce));

        let write_result = (|| -> io::Result<()> {
            let mut opts = OpenOptions::new();
            opts.write(true).create_new(true);
            #[cfg(unix)]
            {
                opts.mode(0o600).custom_flags(libc::O_NOFOLLOW);
            }
            let mut file = opts.open(&tmp_path)?;
            #[cfg(unix)]
            {
                // Enforce strict 0600 even under permissive umasks.
                file.set_permissions(fs::Permissions::from_mode(0o600))?;
            }
            enable_nocache(&file);
            file.write_all(&header)?;
            if std::env::var("INKY_SERVE_ALLOW_FAULT_INJECTION").as_deref() == Ok("1") {
                if let Some(marker) = std::env::var_os("INKY_TEST_SPILL_PAUSE_MID_WRITE") {
                    let marker_path = Path::new(&marker);
                    if !marker_path.exists() {
                        let _ = file.flush();
                        let _ = fs::write(marker_path, tmp_path.to_string_lossy().as_bytes());
                        std::thread::sleep(std::time::Duration::from_millis(800));
                    }
                }
            }
            file.write_all(kv_bytes)?;
            if !state_slice.is_empty() {
                file.write_all(state_slice)?;
            }
            file.sync_all()?;
            drop(file);
            fs::rename(&tmp_path, &final_path)?;
            fsync_dir(&self.dir)?;
            Ok(())
        })();

        if let Err(e) = write_result {
            let _ = fs::remove_file(&tmp_path);
            return Err(SpillError::Io(e));
        }

        if let Some(old) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(old.bytes);
        }
        let seq = self.next_lru();
        self.entries.insert(key, DiskEntry { bytes: total_bytes, lru_seq: seq });
        self.used_bytes = self.used_bytes.saturating_add(total_bytes);
        self.stats.writes_committed += 1;
        self.stats.bytes_written = self.stats.bytes_written.saturating_add(total_bytes);

        Ok((WriteOutcome::Written { on_disk_bytes: total_bytes }, evicted))
    }

    /// Read and cryptographically verify a block from SSD via `pread`.
    ///
    /// Any missing, truncated, oversized, mismatched, or checksum-failing file
    /// returns `ReadOutcome::Miss(reason)` (and unlinks the corrupt `.blk`),
    /// NEVER exposing unverified or corrupt bytes.
    pub fn read_block(&mut self, key: SpillKey, expected_kv_len: Option<usize>) -> ReadOutcome {
        let raw = read_block_from_dir(&self.dir, self.kvb, key, expected_kv_len);
        self.apply_raw_read(key, raw)
    }

    /// Apply bookkeeping (LRU touch, hit/miss/corrupt counters, and unlinking corrupt `.blk`
    /// files) for a [`RawDiskRead`] completed by a background restore worker.
    pub fn apply_raw_read(&mut self, key: SpillKey, raw: RawDiskRead) -> ReadOutcome {
        if let Some(file_len) = raw.observed_file_len {
            let entry = self.entries.entry(key).or_insert(DiskEntry { bytes: 0, lru_seq: 0 });
            self.used_bytes = self.used_bytes.saturating_sub(entry.bytes).saturating_add(file_len);
            entry.bytes = file_len;
        }
        match &raw.outcome {
            ReadOutcome::Hit(_) => {
                let seq = self.next_lru();
                let entry = self.entries.entry(key).or_insert(DiskEntry { bytes: 0, lru_seq: 0 });
                entry.lru_seq = seq;
                self.stats.reads_hit += 1;
            }
            ReadOutcome::Miss(reason) => {
                if raw.is_corrupt {
                    self.drop_corrupt(key);
                } else {
                    if *reason == MissReason::NotFound {
                        self.forget_block(key);
                    }
                    self.stats.reads_miss += 1;
                }
            }
        }
        raw.outcome
    }

    /// Remove any indexed `.blk` files not present in `keep`, returning the number removed.
    pub fn retain_keys(&mut self, keep: &HashSet<SpillKey>) -> usize {
        let stale: Vec<SpillKey> = self
            .entries
            .keys()
            .copied()
            .filter(|k| !keep.contains(k))
            .collect();
        let mut removed = 0usize;
        for k in stale {
            if self.remove_block(k) {
                removed += 1;
            }
        }
        removed
    }

    /// Crash-safe atomic write of `spill-index.bin` (`0600`, `O_NOFOLLOW`, `fsync`, `rename`, dir `fsync`).
    pub fn write_index_bytes(&self, bytes: &[u8]) -> Result<(), SpillError> {
        let final_path = self.dir.join(SPILL_INDEX_FILENAME);
        let pid = std::process::id();
        let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let tmp_path = self.dir.join(format!("{SPILL_INDEX_TMP_PREFIX}{pid:08x}.{nonce:016x}"));
        let res = (|| -> io::Result<()> {
            let mut opts = OpenOptions::new();
            opts.write(true).create_new(true);
            #[cfg(unix)]
            {
                opts.mode(0o600).custom_flags(libc::O_NOFOLLOW);
            }
            let mut file = opts.open(&tmp_path)?;
            #[cfg(unix)]
            {
                file.set_permissions(fs::Permissions::from_mode(0o600))?;
            }
            file.write_all(bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&tmp_path, &final_path)?;
            fsync_dir(&self.dir)?;
            Ok(())
        })();
        if let Err(e) = res {
            let _ = fs::remove_file(&tmp_path);
            return Err(SpillError::Io(e));
        }
        Ok(())
    }

    /// Read `spill-index.bin` if present and regular (`<= max_len` bytes).
    /// Unlinks and returns `None` if corrupt, non-regular, or oversized.
    pub fn read_index_bytes(&mut self, max_len: usize) -> Option<Vec<u8>> {
        let path = self.dir.join(SPILL_INDEX_FILENAME);
        let mut opts = OpenOptions::new();
        opts.read(true);
        #[cfg(unix)]
        {
            opts.custom_flags(libc::O_NOFOLLOW);
        }
        let file = match opts.open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return None,
            Err(_) => {
                if matches!(unlink_if_present(&path), Ok(true)) {
                    self.stats.corrupt_files_removed += 1;
                }
                return None;
            }
        };
        let meta = match file.metadata() {
            Ok(m) if m.is_file() && m.len() <= max_len as u64 => m,
            _ => {
                drop(file);
                if matches!(unlink_if_present(&path), Ok(true)) {
                    self.stats.corrupt_files_removed += 1;
                }
                return None;
            }
        };
        let len = meta.len() as usize;
        let mut buf = vec![0u8; len];
        if pread_exact(&file, &mut buf, 0).is_err() {
            drop(file);
            if matches!(unlink_if_present(&path), Ok(true)) {
                self.stats.corrupt_files_removed += 1;
            }
            return None;
        }
        Some(buf)
    }

    /// Unlink `spill-index.bin` (recording a corrupt removal if `count_corrupt` is true).
    pub fn remove_index(&mut self, count_corrupt: bool) {
        let path = self.dir.join(SPILL_INDEX_FILENAME);
        if matches!(unlink_if_present(&path), Ok(true)) && count_corrupt {
            self.stats.corrupt_files_removed += 1;
        }
    }

    fn drop_corrupt(&mut self, key: SpillKey) {
        if matches!(self.unlink_block(key), Ok(true)) {
            self.stats.corrupt_files_removed += 1;
        }
        self.stats.reads_miss += 1;
    }
}

/// Result of reading and SHA-256-verifying a single `.blk` file without holding `&mut SpillStore`.
#[derive(Debug)]
pub struct RawDiskRead {
    pub outcome: ReadOutcome,
    pub observed_file_len: Option<u64>,
    pub is_corrupt: bool,
}

/// Standalone `pread` + SHA-256 verification of a `.blk` file inside `dir`.
/// Safe to execute on a background restore worker thread without blocking the scheduler.
pub fn read_block_from_dir(
    dir: &Path,
    kvb: u32,
    key: SpillKey,
    expected_kv_len: Option<usize>,
) -> RawDiskRead {
    #[cfg(test)]
    super::run_test_spill_read_hook(dir, key);

    let path = dir.join(key.blk_filename());
    let mut opts = OpenOptions::new();
    opts.read(true);
    #[cfg(unix)]
    {
        opts.custom_flags(libc::O_NOFOLLOW);
    }
    let file = match opts.open(&path) {
        Ok(f) => f,
        Err(e) => {
            #[cfg(unix)]
            if e.raw_os_error() == Some(libc::ELOOP) {
                return RawDiskRead {
                    outcome: ReadOutcome::Miss(MissReason::NotRegularFile),
                    observed_file_len: None,
                    is_corrupt: true,
                };
            }
            let reason = if e.kind() == io::ErrorKind::NotFound {
                MissReason::NotFound
            } else {
                MissReason::IoReadError
            };
            return RawDiskRead {
                outcome: ReadOutcome::Miss(reason),
                observed_file_len: None,
                is_corrupt: false,
            };
        }
    };

    enable_nocache(&file);

    let meta = match file.metadata() {
        Ok(m) => m,
        Err(_) => {
            return RawDiskRead {
                outcome: ReadOutcome::Miss(MissReason::IoReadError),
                observed_file_len: None,
                is_corrupt: true,
            };
        }
    };
    if !meta.is_file() {
        return RawDiskRead {
            outcome: ReadOutcome::Miss(MissReason::NotRegularFile),
            observed_file_len: None,
            is_corrupt: true,
        };
    }
    let file_len = meta.len();
    let miss_corrupt = |reason: MissReason| RawDiskRead {
        outcome: ReadOutcome::Miss(reason),
        observed_file_len: Some(file_len),
        is_corrupt: true,
    };
    if file_len < SPILL_HEADER_LEN as u64 {
        return miss_corrupt(MissReason::TruncatedHeader);
    }

    let mut header = [0u8; SPILL_HEADER_LEN];
    if pread_exact(&file, &mut header, 0).is_err() {
        return miss_corrupt(MissReason::TruncatedHeader);
    }
    if header[0..8] != SPILL_MAGIC {
        return miss_corrupt(MissReason::BadMagic);
    }
    let version = u32::from_le_bytes(header[8..12].try_into().unwrap());
    if version != SPILL_FORMAT_VERSION {
        return miss_corrupt(MissReason::UnsupportedVersion);
    }
    let raw_kind = u32::from(u16::from_le_bytes(header[12..14].try_into().unwrap()));
    let Some(kind) = SpillBlockKind::from_u32(raw_kind) else {
        return miss_corrupt(MissReason::BadBlockKind);
    };
    let file_kvb = u32::from(u16::from_le_bytes(header[14..16].try_into().unwrap()));
    if file_kvb != kvb {
        return miss_corrupt(MissReason::KvbMismatch);
    }
    let block_id = u64::from_le_bytes(header[16..24].try_into().unwrap());
    if kind != key.kind || block_id != key.id {
        return miss_corrupt(MissReason::MetadataMismatch);
    }
    let generation = u64::from_le_bytes(header[24..32].try_into().unwrap());
    let kv_len_u64 = u64::from_le_bytes(header[32..40].try_into().unwrap());
    let state_len_u64 = u64::from_le_bytes(header[40..48].try_into().unwrap());
    let stored_digest: [u8; 32] = header[48..80].try_into().unwrap();

    let (Ok(kv_len), Ok(state_len)) = (usize::try_from(kv_len_u64), usize::try_from(state_len_u64)) else {
        return miss_corrupt(MissReason::PayloadTooLarge);
    };
    if kv_len > MAX_KV_PAYLOAD_BYTES || state_len > MAX_RECURRENT_STATE_BYTES {
        return miss_corrupt(MissReason::PayloadTooLarge);
    }
    if let Some(exp) = expected_kv_len {
        if kv_len != exp {
            return miss_corrupt(MissReason::LengthMismatch);
        }
    }
    let expected_file_len = (SPILL_HEADER_LEN as u64)
        .checked_add(kv_len_u64)
        .and_then(|v| v.checked_add(state_len_u64));
    if expected_file_len != Some(file_len) {
        return miss_corrupt(MissReason::SizeMismatch);
    }

    let mut kv_bytes = vec![0u8; kv_len];
    if kv_len > 0 && pread_exact(&file, &mut kv_bytes, SPILL_HEADER_LEN as u64).is_err() {
        return miss_corrupt(MissReason::IoReadError);
    }

    let recurrent_state = if state_len > 0 {
        let mut st = vec![0u8; state_len];
        let st_off = (SPILL_HEADER_LEN as u64).saturating_add(kv_len_u64);
        if pread_exact(&file, &mut st, st_off).is_err() {
            return miss_corrupt(MissReason::IoReadError);
        }
        Some(st)
    } else {
        None
    };

    let mut hasher = Sha256::new();
    hasher.update(&header[0..HEADER_PREFIX_LEN]);
    hasher.update(&kv_bytes);
    if let Some(st) = &recurrent_state {
        hasher.update(st);
    }
    let computed = hasher.finalize();
    if !ct_eq_32(&computed, &stored_digest) {
        return miss_corrupt(MissReason::ChecksumMismatch);
    }

    RawDiskRead {
        outcome: ReadOutcome::Hit(RestoredBlock {
            key,
            generation,
            kv_bytes,
            recurrent_state,
        }),
        observed_file_len: Some(file_len),
        is_corrupt: false,
    }
}

pub(super) fn prepare_spill_dir(dir: &Path) -> Result<(), SpillError> {
    match fs::symlink_metadata(dir) {
        Ok(meta) => {
            if meta.file_type().is_symlink() {
                return Err(SpillError::InvalidDir(format!(
                    "{} must not be a symlink",
                    dir.display()
                )));
            }
            if !meta.is_dir() {
                return Err(SpillError::InvalidDir(format!(
                    "{} exists and is not a directory",
                    dir.display()
                )));
            }
            #[cfg(unix)]
            {
                let mode = meta.mode() & 0o777;
                if mode != 0o700 {
                    fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
                }
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            let mut builder = fs::DirBuilder::new();
            builder.recursive(true);
            #[cfg(unix)]
            {
                builder.mode(0o700);
            }
            builder.create(dir)?;
            #[cfg(unix)]
            {
                fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
            }
        }
        Err(e) => return Err(SpillError::Io(e)),
    }
    Ok(())
}
