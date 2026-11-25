//! Inky mode: while any inky process runs, the GPU may wire the memory this machine's tuning gives it; when the last one
//! stops, the limit goes back to what it was. A small root helper (`inky memd`, installed once by `sudo inky
//! tune-memory`) owns the setting; inky processes connect to it over a Unix socket only the owner can open. A connection
//! closes on a clean exit, a crash, kill -9 or sleep alike, so the memory is always given back. specs/memmode/MemMode.tla
//! models the helper.

use std::fs::{self, Metadata, OpenOptions, Permissions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{DirBuilderExt, FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const SOCKET: &str = "/private/var/db/inky/memd.sock";
const OLD_SOCKET_DIR: &str = "/private/var/run/inky";
const OLD_SOCKET_NAME: &std::ffi::CStr = c"memd.sock";
const LIMIT: &str = "iogpu.wired_limit_mb";
/// the helper's binary: a root-owned copy in a directory only root can write
const HELPER_DIR: &str = "/Library/PrivilegedHelperTools";
const HELPER: &str = "/Library/PrivilegedHelperTools/inky-memd";
const HELPER_TEMP: &str = "/Library/PrivilegedHelperTools/inky-memd.new";
/// where an earlier version installed it; removed on install
const OLD_HELPER: &str = "/usr/local/libexec/inky-memd";
/// the limit found when the helper raised it, kept on disk while raised, so a helper that dies while raised restores it
/// when it starts again
const SAVED: &str = "/private/var/db/inky-memd.saved";
const SAVED_TEMP: &str = "/private/var/db/inky-memd.saved.tmp";
const PLIST: &str = "/Library/LaunchDaemons/com.robotgamedevelopers.inky.memd.plist";
const PLIST_TEMP: &str = "/Library/LaunchDaemons/com.robotgamedevelopers.inky.memd.plist.new";
const LABEL: &str = "com.robotgamedevelopers.inky.memd";
const LAUNCHCTL: &str = "/bin/launchctl";

/// The helper's state (MemMode.tla): the limit it found when it raised it, and how many clients are connected.
#[derive(Default)]
pub struct Mode { pub saved: Option<u64>, pub clients: usize }

impl Mode {
    /// A client connected: the first one raises the limit to `target`, remembering the limit found (and `keep` stores it
    /// before the raise). If the limit cannot be read, the saved limit cannot be written, or the raise itself fails,
    /// the client is refused. A raise that moved the limit before failing is rolled back to the limit found; the save
    /// (in memory and on disk) is forgotten once the limit is confirmed back where it was, and kept only while the
    /// limit is still moved, so the helper's exit paths can still restore it.
    pub fn connect(&mut self, target: u64, get: impl Fn() -> Option<u64>, set: impl Fn(u64) -> Result<(), String>, keep: impl Fn(Option<u64>) -> Result<(), String>) -> Result<(), String> {
        if self.clients == 0 {
            if self.saved.is_none() {
                let found = get().ok_or("the current limit cannot be read; not raising it")?;
                keep(Some(found)).map_err(|e| format!("the saved limit could not be written; not raising it: {e}"))?;
                self.saved = Some(found);
            }
            let found = self.saved.unwrap();
            if let Err(e) = set(target) {
                if get() != Some(found) { let _ = set(found); }     // the raise moved the limit before failing: put it back
                if get() == Some(found) {
                    self.saved = None;
                    if let Err(ke) = keep(None) { eprintln!("[inky memd] the unraised limit {found} could not be forgotten on disk: {ke}"); }
                }
                return Err(format!("the limit could not be raised: {e}"));
            }
        }
        self.clients += 1;
        Ok(())
    }
    /// A client's connection closed: the last one restores the limit found. The saved limit is kept while the restore
    /// is unconfirmed, so the daemon's next start (or its SIGTERM handler) attempts it again from the saved file;
    /// nothing in here retries on its own. Once the restore is confirmed by reading the limit back, the in-memory
    /// save is cleared even if forgetting the file fails — the stale file is overwritten by the next connect's
    /// fresh save — so the next session always reads the owner's current limit.
    pub fn close(&mut self, get: impl Fn() -> Option<u64>, set: impl Fn(u64) -> Result<(), String>, keep: impl Fn(Option<u64>) -> Result<(), String>) {
        assert!(self.clients > 0);
        self.clients -= 1;
        if self.clients == 0 {
            if let Some(v) = self.saved {
                match set(v) {
                    Ok(()) if get() == Some(v) => {
                        self.saved = None;
                        if let Err(e) = keep(None) { eprintln!("[inky memd] the restored limit {v} could not be forgotten on disk; the saved file is stale until the next connect overwrites it: {e}"); }
                    }
                    Ok(()) => eprintln!("[inky memd] restoring {LIMIT}={v} was not confirmed by reading it back; keeping the saved limit"),
                    Err(e) => eprintln!("[inky memd] restoring {LIMIT}={v} failed: {e}; keeping the saved limit"),
                }
            }
        }
    }
}

fn get_limit() -> Option<u64> { crate::tuning::sysctl_u64(LIMIT) }
/// Sets the limit, reporting failure. Success is confirmed by reading the limit back.
fn set_limit(v: u64) -> Result<(), String> {
    let Ok(mut x) = u32::try_from(v) else { return Err(format!("{v} does not fit {LIMIT}; not set")) };
    let n = std::ffi::CString::new(LIMIT).unwrap();
    let r = unsafe { libc::sysctlbyname(n.as_ptr(), std::ptr::null_mut(), std::ptr::null_mut(), &mut x as *mut u32 as *mut libc::c_void, 4) };
    if r != 0 { return Err(format!("setting {LIMIT}={v} failed: {}", std::io::Error::last_os_error())); }
    if get_limit() != Some(v) { return Err(format!("setting {LIMIT}={v} was not confirmed by reading it back")); }
    eprintln!("[inky memd] {LIMIT} = {v}");
    Ok(())
}

fn read_saved(path: &str) -> Result<Option<u64>, String> {
    match read_saved_record(path, None)? {
        SavedRecord::Absent | SavedRecord::StaleBoot(_, _) => Ok(None),
        SavedRecord::Current(v) => Ok(Some(v)),
    }
}

#[derive(Debug, PartialEq, Eq)]
enum SavedRecord {
    Absent,
    StaleBoot(i64, u64),
    Current(u64),
}

fn read_saved_record(path: &str, current_boot: Option<i64>) -> Result<SavedRecord, String> {
    match std::fs::symlink_metadata(path) {
        Ok(md) if md.is_file() => {}
        Ok(_) => return Err(format!("the saved limit at {path} is not a regular file")),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(SavedRecord::Absent),
        Err(e) => return Err(format!("could not inspect the saved limit at {path}: {e}")),
    }
    let value = std::fs::read_to_string(path)
        .map_err(|e| format!("could not read the saved limit at {path}: {e}"))?;
    let trimmed = value.trim();
    if let Some((boot_str, limit_str)) = trimmed.split_once(':') {
        let boot = boot_str.parse::<i64>()
            .map_err(|e| format!("could not parse the saved boot time at {path}: {e}"))?;
        let limit = limit_str.parse::<u32>()
            .map_err(|e| format!("could not parse the saved limit at {path}: {e}"))?;
        if let Some(now_boot) = current_boot {
            if boot != now_boot {
                return Ok(SavedRecord::StaleBoot(boot, u64::from(limit)));
            }
        }
        return Ok(SavedRecord::Current(u64::from(limit)));
    }
    trimmed.parse::<u32>().map(|v| SavedRecord::Current(u64::from(v)))
        .map_err(|e| format!("could not parse the saved limit at {path}: {e}"))
}

pub(super) fn restore_on_startup(saved_path: &str, set: impl Fn(u64) -> Result<(), String>, keep: impl Fn(Option<u64>) -> Result<(), String>) -> Result<Mode, String> {
    let mut seed = None;
    if let Some(v) = read_saved(saved_path)? {
        eprintln!("[inky memd] the previous helper stopped while the limit was raised: restoring {v}");
        match set(v) {
            Ok(()) => { if let Err(e) = keep(None) { eprintln!("[inky memd] {e}"); } }
            Err(e) => {
                eprintln!("[inky memd] restoring {v} failed: {e}; keeping the saved limit in {saved_path}");
                seed = Some(v);
            }
        }
    }
    Ok(Mode { saved: seed, clients: 0 })
}

fn checked_directory(path: &Path, uid: u32) -> Result<Metadata, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|e| format!("could not inspect directory {}: {e}", path.display()))?;
    if !metadata.is_dir() || metadata.uid() != uid || metadata.mode() & 0o022 != 0 {
        return Err(format!("{} must be a real directory writable only by UID {uid}", path.display()));
    }
    if uid == 0 { require_no_acl(path)?; }
    Ok(metadata)
}

fn require_no_acl(path: &Path) -> Result<(), String> {
    // Darwin's extended ACL can grant writes beyond the POSIX mode bits.
    const ACL_TYPE_EXTENDED: libc::c_uint = 0x100;
    unsafe extern "C" {
        fn acl_get_link_np(path: *const libc::c_char, acl_type: libc::c_uint) -> *mut libc::c_void;
        fn acl_free(acl: *mut libc::c_void) -> libc::c_int;
    }
    let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|e| format!("invalid directory path {}: {e}", path.display()))?;
    let acl = unsafe { acl_get_link_np(c.as_ptr(), ACL_TYPE_EXTENDED) };
    if !acl.is_null() {
        unsafe { acl_free(acl); }
        return Err(format!("{} has an extended access control list", path.display()));
    }
    let e = io::Error::last_os_error();
    // The directory was verified above. Darwin reports ENOENT when its ACL is absent.
    if e.kind() == io::ErrorKind::NotFound { return Ok(()); }
    Err(format!("could not inspect the access control list for {}: {e}", path.display()))
}

fn socket_directory(path: &Path) -> Result<&Path, String> {
    path.parent().filter(|dir| !dir.as_os_str().is_empty())
        .ok_or_else(|| format!("{} needs a socket directory", path.display()))
}

fn check_system_socket_ancestors(dir: &Path) -> Result<(), String> {
    // Inspect from the root down so each path component is protected before looking below it.
    let ancestors: Vec<_> = dir.ancestors().skip(1).collect();
    for ancestor in ancestors.into_iter().rev() {
        checked_directory(ancestor, 0)?;
    }
    Ok(())
}

fn prepare_socket_directory(dir: &Path, uid: u32, fake: bool) -> Result<(), String> {
    if fake {
        if let Some(parent) = dir.parent().filter(|path| !path.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .map_err(|e| format!("could not create directory {}: {e}", parent.display()))?;
        }
    } else {
        check_system_socket_ancestors(dir)?;
    }
    let mode = if fake { 0o700 } else { 0o755 };
    match fs::DirBuilder::new().mode(mode).create(dir) {
        Ok(()) => fs::set_permissions(dir, Permissions::from_mode(mode))
            .map_err(|e| format!("could not set permissions on {}: {e}", dir.display()))?,
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(format!("could not create socket directory {}: {e}", dir.display())),
    }
    let metadata = checked_directory(dir, uid)?;
    if !fake && metadata.mode() & 0o7777 != mode {
        return Err(format!("{} must have mode {mode:o}", dir.display()));
    }
    Ok(())
}

fn remove_file_if_present(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("could not remove {}: {e}", path.display())),
    }
}

fn remove_socket(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(md) if md.file_type().is_socket() => remove_file_if_present(path),
        Ok(_) => Err(format!("{} is not a socket", path.display())),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("could not inspect socket {}: {e}", path.display())),
    }
}

fn check_socket(path: &Path, owner: u32) -> Result<(), String> {
    let md = fs::symlink_metadata(path)
        .map_err(|e| format!("could not inspect socket {}: {e}", path.display()))?;
    if !md.file_type().is_socket() || md.uid() != owner || md.mode() & 0o7777 != 0o600 {
        return Err(format!("{} must be a socket with owner UID {owner} and mode 600", path.display()));
    }
    Ok(())
}

fn bind_socket(path: &Path, owner: u32, directory_uid: u32, fake: bool) -> Result<UnixListener, String> {
    prepare_socket_directory(socket_directory(path)?, directory_uid, fake)?;
    remove_socket(path)?;
    let old_mask = unsafe { libc::umask(0o177) };
    let listener = UnixListener::bind(path);
    unsafe { libc::umask(old_mask); }
    let listener = listener.map_err(|e| format!("could not bind {}: {e}", path.display()))?;
    // With the private mode already in place, give the account access to the socket only.
    if !fake {
        let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
        if unsafe { libc::lchown(c.as_ptr(), owner, libc::gid_t::MAX) } != 0 {
            return Err(format!("could not set the owner of {}: {}", path.display(), io::Error::last_os_error()));
        }
    }
    check_socket(path, owner)?;
    Ok(listener)
}

const MAX_MEMD_LINE: usize = 256;
const MAX_MEMD_CLIENTS: usize = 256;

fn boot_time_sec() -> Option<i64> {
    let mut mib = [libc::CTL_KERN, libc::KERN_BOOTTIME];
    let mut tv: libc::timeval = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<libc::timeval>();
    let rc = unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            mib.len() as libc::c_uint,
            &mut tv as *mut _ as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && len == std::mem::size_of::<libc::timeval>() && tv.tv_sec > 0).then_some(tv.tv_sec as i64)
}

fn write_saved_atomic(path: &str, value: u64, boot: Option<i64>) -> Result<(), String> {
    let tmp = format!("{path}.tmp");
    let body = match boot {
        Some(sec) => format!("{sec}:{value}\n"),
        None => format!("{value}\n"),
    };
    (|| -> io::Result<()> {
        let mut f = OpenOptions::new().create(true).truncate(true).write(true).mode(0o600).open(&tmp)?;
        f.write_all(body.as_bytes())?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, path)
    })()
    .map_err(|e| format!("could not record the saved limit in {path}: {e}"))
}

fn restore_on_startup_for_boot(
    saved_path: &str,
    current_boot: Option<i64>,
    set: impl Fn(u64) -> Result<(), String>,
    keep: impl Fn(Option<u64>) -> Result<(), String>,
) -> Result<Mode, String> {
    match read_saved_record(saved_path, current_boot)? {
        SavedRecord::Absent => Ok(Mode::default()),
        SavedRecord::StaleBoot(old_boot, old_val) => {
            eprintln!("[inky memd] ignoring stale saved limit {old_val} from earlier boot {old_boot}");
            if let Err(e) = keep(None) { eprintln!("[inky memd] {e}"); }
            Ok(Mode::default())
        }
        SavedRecord::Current(_) => restore_on_startup(saved_path, set, keep),
    }
}

fn read_bounded_line<R: BufRead>(r: &mut R, max_bytes: usize) -> io::Result<Option<String>> {
    let mut line = String::new();
    let n = r.by_ref().take((max_bytes + 1) as u64).read_line(&mut line)?;
    if n == 0 { return Ok(None); }
    if line.len() > max_bytes {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "memd line too long"));
    }
    Ok(Some(line))
}

struct ClientGuard(Arc<AtomicUsize>);

impl ClientGuard {
    fn try_acquire(active: &Arc<AtomicUsize>, limit: usize) -> Option<Self> {
        let mut cur = active.load(Ordering::Acquire);
        loop {
            if cur >= limit { return None; }
            match active.compare_exchange_weak(cur, cur + 1, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Some(Self(active.clone())),
                Err(next) => cur = next,
            }
        }
    }
}

impl Drop for ClientGuard {
    fn drop(&mut self) { self.0.fetch_sub(1, Ordering::AcqRel); }
}

fn peer_uid(stream: &UnixStream) -> Option<u32> {
    let (mut uid, mut gid) = (0, 0);
    (unsafe { libc::getpeereid(stream.as_raw_fd(), &mut uid, &mut gid) } == 0).then_some(uid)
}

/// `inky memd --owner UID` is the root helper. The unprivileged `--fake --socket PATH` form uses an in-memory limit.
pub fn daemon(args: &[String]) {
    if let Err(e) = run_daemon(args) {
        eprintln!("[inky memd] {e}");
        std::process::exit(1);
    }
}

fn run_daemon(args: &[String]) -> Result<(), String> {
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1));
    let owner: u32 = arg("--owner").and_then(|v| v.parse().ok())
        .filter(|&uid| uid != libc::uid_t::MAX).ok_or("memd requires --owner UID")?;
    let fake = args.iter().any(|a| a == "--fake");
    let uid = unsafe { libc::geteuid() };
    let path = if fake {
        if uid == 0 || uid != owner { return Err("fake memd requires the current unprivileged UID".into()); }
        arg("--socket").ok_or("fake memd requires --socket PATH")?.as_str()
    } else {
        if uid != 0 || args.iter().any(|a| a == "--socket") {
            return Err("system memd requires root and the default socket path".into());
        }
        check_system_socket_ancestors(socket_directory(Path::new(SOCKET))?)?;
        SOCKET
    };
    let (chip, mem) = crate::tuning::this_machine();
    let tuning = crate::tuning::select(&chip, mem);
    let target = tuning.wired_limit_mb;
    let fake_limit = Arc::new(Mutex::new(0u64));
    let (fl1, fl2) = (fake_limit.clone(), fake_limit.clone());
    let get: Arc<dyn Fn() -> Option<u64> + Send + Sync> = if fake { Arc::new(move || Some(*fl1.lock().unwrap())) } else { Arc::new(get_limit) };
    let set: Arc<dyn Fn(u64) -> Result<(), String> + Send + Sync> = if fake { Arc::new(move |v| { *fl2.lock().unwrap() = v; eprintln!("[inky memd] (fake) limit = {v}"); Ok(()) }) } else { Arc::new(set_limit) };
    // the saved limit on disk (beside the socket with --fake, so tests need no root)
    let saved_path = if fake { format!("{path}.saved") } else { SAVED.to_string() };
    let boot_time = boot_time_sec();
    let sp = saved_path.clone();
    let keep: Arc<dyn Fn(Option<u64>) -> Result<(), String> + Send + Sync> = Arc::new(move |v| match v {
        Some(v) => write_saved_atomic(&sp, v, boot_time),
        None => match std::fs::remove_file(&sp) { Ok(()) => Ok(()), Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()), Err(e) => Err(format!("could not forget the saved limit in {sp}: {e}")) },
    });
    // a helper that died while the limit was raised left the limit it found on disk: put it back first (`set`
    // confirms by reading back, so its Ok is the confirmation). If the restore fails, the file is kept and Mode is
    // seeded from it below, so the first connect restores that limit instead of reading the still-raised limit
    // as the found one and overwriting the file with it. Saved files from an earlier macOS boot are ignored.
    let mode = Arc::new(Mutex::new(restore_on_startup_for_boot(&saved_path, boot_time, |v| set(v), |v| keep(v))?));
    // SIGTERM (launchctl bootout) and SIGINT: give the memory back, then exit (`set` confirms by reading back,
    // so its Ok is the confirmation; Mode::close keeps its own read-back because its `set` is injected)
    {
        let mut sigs: libc::sigset_t = unsafe { std::mem::zeroed() };
        unsafe {
            libc::sigemptyset(&mut sigs);
            libc::sigaddset(&mut sigs, libc::SIGTERM);
            libc::sigaddset(&mut sigs, libc::SIGINT);
            let rc = libc::pthread_sigmask(libc::SIG_BLOCK, &sigs, std::ptr::null_mut());
            if rc != 0 { return Err(format!("could not block shutdown signals: {}", io::Error::from_raw_os_error(rc))); }
        }
        let (mode, set, keep) = (mode.clone(), set.clone(), keep.clone());
        std::thread::spawn(move || {
            let mut sig = 0;
            let rc = unsafe { libc::sigwait(&sigs, &mut sig) };
            if rc != 0 { eprintln!("[inky memd] could not wait for a shutdown signal: {}", io::Error::from_raw_os_error(rc)); }
            let m = mode.lock().unwrap();
            if let Some(v) = m.saved {
                if let Err(e) = set(v).and_then(|()| keep(None)) {
                    eprintln!("[inky memd] could not complete shutdown recovery for {v}: {e}");
                    std::process::exit(1);
                }
            }
            std::process::exit(i32::from(rc != 0));
        });
    }
    let listener = bind_socket(Path::new(path), owner, uid, fake)?;
    let active_clients = Arc::new(AtomicUsize::new(0));
    eprintln!("[inky memd] tuning {} ({chip}, {} GB): inky mode raises {LIMIT} to {target}; listening on {path}", tuning.name, tuning.memory_gb);
    for s in listener.incoming() {
        let Ok(s) = s else { continue };
        // The socket owner can change its mode, so also check the kernel-supplied peer UID.
        if !peer_uid(&s).is_some_and(|uid| uid == owner || uid == 0) { continue; }
        let Some(slot) = ClientGuard::try_acquire(&active_clients, MAX_MEMD_CLIENTS) else {
            let _ = (&s).write_all(b"err too many clients\n");
            continue;
        };
        let (mode, get, set, keep) = (mode.clone(), get.clone(), set.clone(), keep.clone());
        std::thread::spawn(move || {
            let _slot = slot;
            let mut r = BufReader::new(&s);
            let Ok(Some(line)) = read_bounded_line(&mut r, MAX_MEMD_LINE) else { return; };
            if !line.starts_with("enter") { return; }
            if let Err(e) = mode.lock().unwrap().connect(target, || get(), |v| set(v), |v| keep(v)) {
                eprintln!("[inky memd] refused a client: {e}");
                let _ = (&s).write_all(format!("err {e}\n").as_bytes());
                return;
            }
            let _ = (&s).write_all(format!("ok {target}\n").as_bytes());
            // hold until the client's end closes, however it closes, without allocating per byte
            let mut sink = [0u8; 256];
            while r.read(&mut sink).unwrap_or(0) > 0 {}
            mode.lock().unwrap().close(|| get(), |v| set(v), |v| keep(v));
        });
    }
    Ok(())
}

fn connect_lease(path: &str) -> Option<(UnixStream, u64)> {
    let s = UnixStream::connect(path).ok()?;
    if path == SOCKET && peer_uid(&s) != Some(0) { return None; }
    let timeout = Some(Duration::from_secs(2));
    s.set_write_timeout(timeout).ok()?;
    s.set_read_timeout(timeout).ok()?;
    (&s).write_all(format!("enter {}\n", std::process::id()).as_bytes()).ok()?;
    let mut r = BufReader::new(&s);
    let line = read_bounded_line(&mut r, MAX_MEMD_LINE).ok()??;
    let v = line.strip_prefix("ok ")?.trim().parse().ok()?;
    Some((s, v))
}

/// Keeps the reservation alive until its owner drops it, including across helper restarts.
pub struct Lease {
    pub initial_limit: u64,
    stop: UnixStream,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Lease {
    fn drop(&mut self) {
        let _ = self.stop.shutdown(std::net::Shutdown::Both);
        if let Some(worker) = self.worker.take() { let _ = worker.join(); }
    }
}

fn maintain_lease(path: String, initial: UnixStream, stop: UnixStream) {
    let mut connection = Some(initial);
    let mut retry_ms = 100;
    loop {
        // An acknowledged helper sends no further data. EOF, an error, or an unexpected
        // message ends that lease. Poll the stop socket too so idle leases need no timer.
        let mut fds = [
            libc::pollfd { fd: stop.as_raw_fd(), events: libc::POLLIN, revents: 0 },
            libc::pollfd { fd: connection.as_ref().map_or(-1, AsRawFd::as_raw_fd), events: libc::POLLIN, revents: 0 },
        ];
        let timeout = if connection.is_some() { -1 } else { retry_ms };
        if unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as _, timeout) } < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted { continue; }
            eprintln!("[inky] cannot monitor memory-helper connection: {err}");
            return;
        }
        if fds[0].revents != 0 { return; }
        if connection.take().is_some() {
            eprintln!("[inky] lost memory-helper connection; retrying");
            retry_ms = 100;
        } else if let Some((stream, limit)) = connect_lease(&path) {
            eprintln!("[inky] memory-helper connection restored; GPU wired limit {limit} MB");
            connection = Some(stream);
        } else {
            retry_ms = (retry_ms * 2).min(2_000);
        }
    }
}

pub fn enter(path: &str) -> Option<Lease> {
    let (stream, initial_limit) = connect_lease(path)?;
    let (stop, monitor) = UnixStream::pair().ok()?;
    let path = path.to_owned();
    let worker = std::thread::Builder::new().name("inky-memd-lease".into())
        .spawn(move || maintain_lease(path, stream, monitor)).ok()?;
    Some(Lease { initial_limit, stop, worker: Some(worker) })
}

/// `sudo inky tune-memory [--undo]`: install (or remove) the helper as a root LaunchDaemon. The helper runs a root-owned
/// copy of this binary, so the program running as root cannot be replaced by the user.
pub fn install(args: &[String]) {
    let undo = args.iter().any(|a| a == "--undo");
    if unsafe { libc::geteuid() } != 0 {
        let program = std::env::args().next().unwrap_or("inky".into());
        eprintln!("run it with sudo: sudo {program} tune-memory{}", if undo { " --undo" } else { "" });
        std::process::exit(1);
    }
    let result = if undo { uninstall_helper() } else { install_helper() };
    if let Err(e) = result {
        let action = if undo { "remove" } else { "install" };
        eprintln!("could not {action} inky mode: {e}");
        std::process::exit(1);
    }
}

fn service_is_loaded() -> Result<bool, String> {
    let out = Command::new(LAUNCHCTL).args(["print", &format!("system/{LABEL}")]).output()
        .map_err(|e| format!("could not run launchctl print: {e}"))?;
    match out.status.code() {
        Some(0) => Ok(true),
        Some(113) => Ok(false), // launchctl: could not find specified service
        _ => Err(format!("launchctl print failed ({}): {}", out.status, String::from_utf8_lossy(&out.stderr).trim())),
    }
}

fn require_service_absent() -> Result<(), String> {
    if service_is_loaded()? { return Err(format!("launchd still has service {LABEL}")); }
    Ok(())
}

fn stop_helper() -> Result<(), String> {
    if !service_is_loaded()? { return Ok(()); }
    let status = Command::new(LAUNCHCTL).args(["bootout", &format!("system/{LABEL}")]).status()
        .map_err(|e| format!("could not run launchctl bootout: {e}"))?;
    if !status.success() { return Err(format!("launchctl bootout failed ({status})")); }
    require_service_absent()
}

fn remove_directory_if_present(path: &Path) -> Result<(), String> {
    match fs::remove_dir(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("could not remove directory {}: {e}", path.display())),
    }
}

fn retire_legacy_socket(dir: &Path) -> Result<(), String> {
    // The previous helper gave the client ownership of this directory. Open it without following a symlink,
    // and unlink the old endpoint relative to the directory descriptor without following that entry either.
    let legacy = match OpenOptions::new().read(true).custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW).open(dir) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(format!("could not open the old socket directory {}: {e}", dir.display())),
    };
    let fd = legacy.as_raw_fd();
    let name = OLD_SOCKET_NAME.as_ptr();
    let mut socket: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstatat(fd, name, &mut socket, libc::AT_SYMLINK_NOFOLLOW) } == 0 {
        if socket.st_mode & libc::S_IFMT != libc::S_IFSOCK {
            return Err(format!("{} does not contain a legacy memd socket at memd.sock", dir.display()));
        }
        if unsafe { libc::unlinkat(fd, name, 0) } != 0 {
            let e = io::Error::last_os_error();
            if e.kind() != io::ErrorKind::NotFound {
                return Err(format!("could not remove the old socket in {}: {e}", dir.display()));
            }
        }
    } else {
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::NotFound {
            return Err(format!("could not inspect the old socket in {}: {e}", dir.display()));
        }
    }
    remove_directory_if_present(dir)
}

fn require_absent(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("could not verify removal of {}: {e}", path.display())),
        Ok(_) => Err(format!("{} still exists", path.display())),
    }
}

fn remove_legacy_helper(path: &Path) -> Result<(), String> {
    // A fresh Apple Silicon install may have an unrelated Homebrew directory here.
    // Inspect the entry for absence first, but verify the full chain before unlinking it.
    match fs::symlink_metadata(path) {
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(format!("could not inspect the old helper {}: {e}", path.display())),
    }
    let ancestors: Vec<_> = path.ancestors().skip(1).collect();
    for dir in ancestors.into_iter().rev() {
        checked_directory(dir, 0)?;
    }
    remove_file_if_present(path)?;
    require_absent(path)
}

fn uninstall_helper() -> Result<(), String> {
    stop_helper()?;
    let path = Path::new(SOCKET);
    let dir = socket_directory(path)?;
    check_system_socket_ancestors(dir)?;
    // The stopped helper may have left its recovery record if it could not confirm the restore.
    // Keep the record and installed files intact until recovery is confirmed here; discard stale
    // records left from an earlier boot.
    match read_saved_record(SAVED, boot_time_sec())? {
        SavedRecord::Current(value) => {
            set_limit(value).map_err(|e| format!("restoring {value} failed: {e}; kept {SAVED} and the helper"))?;
            remove_file_if_present(Path::new(SAVED))?;
        }
        SavedRecord::StaleBoot(_, _) => remove_file_if_present(Path::new(SAVED))?,
        SavedRecord::Absent => {}
    }
    checked_directory(Path::new("/Library"), 0)?;
    for dir in [Path::new(HELPER_DIR), Path::new(PLIST).parent().unwrap()] {
        match fs::symlink_metadata(dir) {
            Ok(_) => { checked_directory(dir, 0)?; }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("could not inspect {}: {e}", dir.display())),
        }
    }
    match fs::symlink_metadata(dir) {
        Ok(_) => {
            checked_directory(dir, 0)?;
            remove_socket(path)?;
            remove_directory_if_present(dir)?;
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("could not inspect {}: {e}", dir.display())),
    }
    retire_legacy_socket(Path::new(OLD_SOCKET_DIR))?;
    remove_legacy_helper(Path::new(OLD_HELPER))?;
    let files = [SAVED_TEMP, HELPER_TEMP, PLIST_TEMP, PLIST, HELPER];
    for file in files { remove_file_if_present(Path::new(file))?; }
    for path in [dir, Path::new(OLD_SOCKET_DIR), Path::new(SAVED)].into_iter().chain(files.into_iter().map(Path::new)) {
        require_absent(path)?;
    }
    require_service_absent()?;
    println!("inky mode removed. {LIMIT} is now {}", get_limit().map_or("unreadable".into(), |v| v.to_string()));
    Ok(())
}

fn check_installed_helper() -> Result<(), String> {
    let md = fs::symlink_metadata(HELPER).map_err(|e| format!("could not inspect {HELPER}: {e}"))?;
    if !md.is_file() || md.uid() != 0 || md.mode() & 0o7777 != 0o755 {
        return Err(format!("{HELPER} must be a root-owned regular file with mode 755"));
    }
    Ok(())
}

fn install_helper() -> Result<(), String> {
    let owner: u32 = std::env::var("SUDO_UID").ok().and_then(|v| v.parse().ok())
        .filter(|&uid| uid != libc::uid_t::MAX).ok_or("run it with sudo from the owner's account")?;
    checked_directory(Path::new("/Library"), 0)?;
    match fs::DirBuilder::new().mode(0o755).create(HELPER_DIR) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(format!("could not create {HELPER_DIR}: {e}")),
    }
    checked_directory(Path::new(HELPER_DIR), 0)?;
    checked_directory(Path::new(PLIST).parent().unwrap(), 0)?;
    let socket = Path::new(SOCKET);
    let dir = socket_directory(socket)?;
    check_system_socket_ancestors(dir)?;
    stop_helper()?;
    read_saved_record(SAVED, boot_time_sec())?;
    retire_legacy_socket(Path::new(OLD_SOCKET_DIR))?;
    prepare_socket_directory(dir, 0, false)?;
    remove_socket(socket)?;

    remove_file_if_present(Path::new(HELPER_TEMP))?;
    let copy_helper = || -> io::Result<()> {
        let mut source = fs::File::open(std::env::current_exe()?)?;
        let mut staged = OpenOptions::new().create_new(true).write(true).mode(0o600).open(HELPER_TEMP)?;
        io::copy(&mut source, &mut staged)?;
        staged.set_permissions(Permissions::from_mode(0o755))?;
        fs::rename(HELPER_TEMP, HELPER)
    };
    copy_helper().map_err(|e| format!("could not install {HELPER}: {e}"))?;
    check_installed_helper()?;
    let plist = format!(r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key><array><string>{HELPER}</string><string>memd</string><string>--owner</string><string>{owner}</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardErrorPath</key><string>/var/log/inky-memd.log</string>
</dict></plist>
"#);
    remove_file_if_present(Path::new(PLIST_TEMP))?;
    let write_plist = || -> io::Result<()> {
        let mut staged = OpenOptions::new().create_new(true).write(true).mode(0o600).open(PLIST_TEMP)?;
        staged.write_all(plist.as_bytes())?;
        staged.set_permissions(Permissions::from_mode(0o644))?;
        fs::rename(PLIST_TEMP, PLIST)
    };
    write_plist().map_err(|e| format!("could not write {PLIST}: {e}"))?;
    let status = Command::new(LAUNCHCTL).args(["bootstrap", "system", PLIST]).status()
        .map_err(|e| format!("could not run launchctl bootstrap: {e}"))?;
    if !status.success() { return Err(format!("launchctl bootstrap failed ({status})")); }
    let deadline = Instant::now() + Duration::from_secs(5);
    while let Err(e) = check_socket(socket, owner) {
        if Instant::now() >= deadline { return Err(format!("the helper did not publish its socket: {e}")); }
        std::thread::sleep(Duration::from_millis(50));
    }
    check_installed_helper()?;
    remove_legacy_helper(Path::new(OLD_HELPER))?;
    let (chip, mem) = crate::tuning::this_machine();
    let t = crate::tuning::select(&chip, mem);
    println!("inky mode installed (tuning {}): while inky runs, {LIMIT} = {}; when it stops, the limit goes back. Remove: sudo inky tune-memory --undo", t.name, t.wired_limit_mb);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    // bind_socket temporarily sets process-wide umask(0o177); serialize filesystem tests within this binary.
    static TEST_FS_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn dropping_client_lease_closes_its_connection() {
        let _fs = TEST_FS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let path = std::env::temp_dir().join(format!("inky-owned-{}.sock", std::process::id()));
        let listener = UnixListener::bind(&path).unwrap();
        let helper = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(&stream);
            let request = read_bounded_line(&mut reader, MAX_MEMD_LINE).unwrap().unwrap();
            assert_eq!(request, format!("enter {}\n", std::process::id()));
            stream.write_all(b"ok 12345\n").unwrap();
            stream
        });
        let lease = enter(path.to_str().unwrap()).unwrap();
        let mut peer = helper.join().unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(lease.initial_limit, 12345);
        peer.set_nonblocking(true).unwrap();
        let mut byte = [0; 1];
        assert_eq!(peer.read(&mut byte).unwrap_err().kind(), io::ErrorKind::WouldBlock);
        drop(lease);
        peer.set_nonblocking(false).unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        assert_eq!(peer.read(&mut byte).unwrap(), 0);
    }

    #[test] fn raised_while_connected_and_restored_after_the_last() {
        let (lim, disk) = (Cell::new(50000u64), Cell::new(None));
        let mut m = Mode::default();
        m.connect(118000, || Some(lim.get()), |v| { lim.set(v); Ok(()) }, |v| { disk.set(v); Ok(()) }).unwrap();
        assert_eq!(disk.get(), Some(50000), "the limit found is on disk before the raise");
        m.connect(118000, || Some(lim.get()), |v| { lim.set(v); Ok(()) }, |v| { disk.set(v); Ok(()) }).unwrap();
        assert_eq!(lim.get(), 118000);
        m.close(|| Some(lim.get()), |v| { lim.set(v); Ok(()) }, |v| { disk.set(v); Ok(()) });
        assert_eq!(lim.get(), 118000, "still one client: still raised");
        m.close(|| Some(lim.get()), |v| { lim.set(v); Ok(()) }, |v| { disk.set(v); Ok(()) });
        assert_eq!((lim.get(), m.saved, disk.get()), (50000, None, None), "the limit found, not the default; nothing left on disk");
    }
    #[test] fn a_second_session_saves_afresh() {
        let (lim, disk) = (Cell::new(0u64), Cell::new(None));
        let mut m = Mode::default();
        m.connect(118000, || Some(lim.get()), |v| { lim.set(v); Ok(()) }, |v| { disk.set(v); Ok(()) }).unwrap(); m.close(|| Some(lim.get()), |v| { lim.set(v); Ok(()) }, |v| { disk.set(v); Ok(()) });
        lim.set(70000);     // the owner changes it between sessions
        m.connect(118000, || Some(lim.get()), |v| { lim.set(v); Ok(()) }, |v| { disk.set(v); Ok(()) }).unwrap(); m.close(|| Some(lim.get()), |v| { lim.set(v); Ok(()) }, |v| { disk.set(v); Ok(()) });
        assert_eq!(lim.get(), 70000);
    }
    #[test] fn an_unreadable_limit_is_never_raised() {
        let (lim, disk) = (Cell::new(50000u64), Cell::new(None));
        let mut m = Mode::default();
        assert!(m.connect(118000, || None, |v| { lim.set(v); Ok(()) }, |v| { disk.set(v); Ok(()) }).is_err());
        assert_eq!((lim.get(), m.clients, disk.get()), (50000, 0, None));
    }
    #[test] fn a_failed_saved_write_refuses_the_raise() {
        let (lim, disk) = (Cell::new(50000u64), Cell::new(None::<u64>));
        let mut m = Mode::default();
        assert!(m.connect(118000, || Some(lim.get()), |v| { lim.set(v); Ok(()) }, |_| Err("the saved file cannot be written".to_string())).is_err());
        assert_eq!((lim.get(), m.clients, m.saved, disk.get()), (50000, 0, None, None), "no raise and no client when the saved limit cannot be written");
    }
    #[test] fn a_failed_raise_is_refused() {
        let (lim, disk) = (Cell::new(50000u64), Cell::new(None));
        let mut m = Mode::default();
        assert!(m.connect(118000, || Some(lim.get()), |_| Err("the limit cannot be set".to_string()), |v| { disk.set(v); Ok(()) }).is_err());
        assert_eq!((lim.get(), m.clients, m.saved, disk.get()), (50000, 0, None, None), "no client and nothing saved when the raise itself fails and the limit never moved");
    }
    #[test] fn a_raise_that_moves_the_limit_then_fails_is_rolled_back() {
        let (lim, disk, calls) = (Cell::new(50000u64), Cell::new(None), Cell::new(0u32));
        let mut m = Mode::default();
        // the first set moves the limit and then reports failure (its read-back differs); the rollback set succeeds
        let set = |v: u64| { calls.set(calls.get() + 1); lim.set(v); if calls.get() == 1 { Err("setting was not confirmed by reading it back".to_string()) } else { Ok(()) } };
        assert!(m.connect(118000, || Some(lim.get()), set, |v| { disk.set(v); Ok(()) }).is_err());
        assert_eq!((lim.get(), m.clients, m.saved, disk.get()), (50000, 0, None, None), "a refused connect leaves the limit where it found it and saves nothing");
    }
    #[test] fn a_seeded_saved_limit_is_not_overwritten_by_the_raised_limit() {
        // the daemon restarted after a failed startup restore: Mode is seeded from the saved file while the
        // limit is still raised, so the first connect must not read the raised limit as the found limit
        let (lim, disk) = (Cell::new(118000u64), Cell::new(Some(50000u64)));
        let mut m = Mode { saved: Some(50000), clients: 0 };
        m.connect(118000, || Some(lim.get()), |v| { lim.set(v); Ok(()) }, |v| { disk.set(v); Ok(()) }).unwrap();
        m.close(|| Some(lim.get()), |v| { lim.set(v); Ok(()) }, |v| { disk.set(v); Ok(()) });
        assert_eq!((lim.get(), m.saved, disk.get()), (50000, None, None), "the seeded limit, not the raised one, is restored");
    }
    #[test] fn a_confirmed_restore_with_a_failed_forget_saves_afresh_next_time() {
        let (lim, disk) = (Cell::new(50000u64), Cell::new(None));
        let mut m = Mode::default();
        // keep records the save but cannot forget it on disk
        let keep = |v: Option<u64>| if v.is_none() { Err("the saved file cannot be removed".to_string()) } else { disk.set(v); Ok(()) };
        m.connect(118000, || Some(lim.get()), |v| { lim.set(v); Ok(()) }, keep).unwrap();
        m.close(|| Some(lim.get()), |v| { lim.set(v); Ok(()) }, keep);
        assert_eq!((lim.get(), m.saved, disk.get()), (50000, None, Some(50000)), "the restore stands and the in-memory save is cleared even though the file could not be forgotten");
        lim.set(70000);     // the owner changes it between sessions
        m.connect(118000, || Some(lim.get()), |v| { lim.set(v); Ok(()) }, |v| { disk.set(v); Ok(()) }).unwrap();
        m.close(|| Some(lim.get()), |v| { lim.set(v); Ok(()) }, |v| { disk.set(v); Ok(()) });
        assert_eq!(lim.get(), 70000, "the next session saves the owner's newer limit afresh");
    }
    #[test] fn the_socket_directory_must_be_a_real_directory() {
        let _fs = TEST_FS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = std::env::current_dir().unwrap().join(format!("inky-memd-dirtest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let uid = unsafe { libc::geteuid() };
        assert!(checked_directory(&base, uid).is_ok(), "a private directory is accepted");
        let target = base.join("real");
        std::fs::create_dir(&target).unwrap();
        let link = base.join("link");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert!(checked_directory(&link, uid).is_err(), "a symlink to a directory is refused");
        let file = base.join("file");
        std::fs::write(&file, "x").unwrap();
        assert!(checked_directory(&file, uid).is_err(), "a plain file is refused");
        assert!(checked_directory(&base.join("missing"), uid).is_err(), "a missing directory is refused");
        fs::set_permissions(&target, Permissions::from_mode(0o775)).unwrap();
        assert!(checked_directory(&target, uid).is_err(), "a group-writable directory is refused");
        assert!(checked_directory(&base, uid ^ 1).is_err(), "a different owner is refused");
        let _ = std::fs::remove_dir_all(&base);
    }
    #[test]
    fn startup_refuses_an_invalid_or_unreadable_save() {
        let _fs = TEST_FS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = std::env::current_dir().unwrap().join(format!("inky-memd-save-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir(&base).unwrap();
        let path = base.join("saved");
        let name = path.to_str().unwrap();
        let set = |_| panic!("startup tried to restore an unusable saved limit");
        let keep = |_| panic!("startup tried to modify an unusable saved limit");
        assert!(restore_on_startup(name, set, keep).unwrap().saved.is_none());

        for bytes in [b"malformed".as_slice(), b"4294967296", b"\xff"] {
            fs::write(&path, bytes).unwrap();
            assert!(restore_on_startup(name, set, keep).is_err());
            assert_eq!(fs::read(&path).unwrap(), bytes);
        }
        fs::write(&path, b"50000").unwrap();
        fs::set_permissions(&path, Permissions::from_mode(0o000)).unwrap();
        if unsafe { libc::geteuid() } != 0 {
            let result = restore_on_startup(name, set, keep);
            assert!(matches!(result, Err(e) if e.contains("could not read")));
        }
        fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(base.join("missing"), &path).unwrap();
        assert!(restore_on_startup(name, set, keep).is_err());
        fs::remove_dir_all(&base).unwrap();
    }
    #[test]
    fn fake_socket_rebinds_with_private_mode_and_reports_its_peer_uid() {
        let _fs = TEST_FS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = std::env::current_dir().unwrap().join(format!("inky-memd-bind-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let path = base.join("memd.sock");
        let uid = unsafe { libc::geteuid() };
        drop(bind_socket(&path, uid, uid, true).unwrap());
        let listener = bind_socket(&path, uid, uid, true).unwrap();
        assert!(check_socket(&path, uid).is_ok());
        let client = UnixStream::connect(&path).unwrap();
        let (accepted, _) = listener.accept().unwrap();
        assert_eq!(peer_uid(&client), Some(uid));
        assert_eq!(peer_uid(&accepted), Some(uid));
        drop(listener);
        fs::remove_dir_all(&base).unwrap();
    }
    #[test]
    fn legacy_cleanup_removes_the_socket_and_retries_after_a_directory_error() {
        let _fs = TEST_FS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = std::env::current_dir().unwrap().join(format!("inky-memd-old-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir(&base).unwrap();
        let socket = base.join("memd.sock");
        drop(UnixListener::bind(&socket).unwrap());
        let extra = base.join("extra");
        fs::write(&extra, b"retained").unwrap();
        assert!(retire_legacy_socket(&base).is_err());
        assert!(!socket.exists());
        assert_eq!(fs::read(&extra).unwrap(), b"retained");
        fs::remove_file(extra).unwrap();
        retire_legacy_socket(&base).unwrap();
        assert!(!base.exists());
        retire_legacy_socket(&base).unwrap();
    }
    #[test]
    fn absent_legacy_helper_requires_no_trusted_homebrew_directory() {
        let _fs = TEST_FS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = std::env::current_dir().unwrap().join(format!("inky-memd-homebrew-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir(&base).unwrap();
        fs::set_permissions(&base, Permissions::from_mode(0o775)).unwrap();
        let path = base.join("inky-memd");
        remove_legacy_helper(&path).unwrap();
        fs::write(&path, b"retained").unwrap();
        assert!(remove_legacy_helper(&path).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"retained");
        fs::remove_dir_all(&base).unwrap();
    }
    #[test] fn an_unconfirmed_restore_keeps_the_saved_limit() {
        let (lim, disk) = (Cell::new(50000u64), Cell::new(None));
        let mut m = Mode::default();
        m.connect(118000, || Some(lim.get()), |v| { lim.set(v); Ok(()) }, |v| { disk.set(v); Ok(()) }).unwrap();
        // a set that reports failure: the restore is not confirmed, so the saved limit is kept
        m.close(|| Some(lim.get()), |_| Err("the limit cannot be set".to_string()), |v| { disk.set(v); Ok(()) });
        assert_eq!((lim.get(), m.saved, disk.get()), (118000, Some(50000), Some(50000)));
        // a set that claims success but does not move the limit: reading it back does not confirm, so it is still kept
        m.clients = 1;
        m.close(|| Some(lim.get()), |_| Ok(()), |v| { disk.set(v); Ok(()) });
        assert_eq!((lim.get(), m.saved, disk.get()), (118000, Some(50000), Some(50000)));
    }
    #[test]
    fn bounded_line_and_client_guard_enforce_limits() {
        let mut ok_reader = io::Cursor::new(b"enter 12345\n".as_slice());
        assert_eq!(read_bounded_line(&mut ok_reader, MAX_MEMD_LINE).unwrap().as_deref(), Some("enter 12345\n"));
        assert_eq!(read_bounded_line(&mut ok_reader, MAX_MEMD_LINE).unwrap(), None);

        let oversized = vec![b'x'; MAX_MEMD_LINE + 16];
        let mut bad_reader = io::Cursor::new(oversized.as_slice());
        assert_eq!(read_bounded_line(&mut bad_reader, MAX_MEMD_LINE).unwrap_err().kind(), io::ErrorKind::InvalidData);

        let active = Arc::new(AtomicUsize::new(0));
        let g1 = ClientGuard::try_acquire(&active, 2).unwrap();
        let g2 = ClientGuard::try_acquire(&active, 2).unwrap();
        assert!(ClientGuard::try_acquire(&active, 2).is_none());
        drop(g1);
        let g3 = ClientGuard::try_acquire(&active, 2).unwrap();
        drop(g2);
        drop(g3);
        assert_eq!(active.load(Ordering::Acquire), 0);
    }
    #[test]
    fn boot_tagged_saved_limit_ignores_prior_boot_and_syncs_atomically() {
        let _fs = TEST_FS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = std::env::current_dir().unwrap().join(format!("inky-memd-boot-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir(&base).unwrap();
        let path = base.join("saved");
        let name = path.to_str().unwrap();

        write_saved_atomic(name, 50000, Some(1_700_000_000)).unwrap();
        assert_eq!(read_saved_record(name, Some(1_700_000_000)), Ok(SavedRecord::Current(50000)));
        assert_eq!(read_saved_record(name, Some(1_700_009_999)), Ok(SavedRecord::StaleBoot(1_700_000_000, 50000)));

        // Stale boot record is ignored (no set() call) and cleaned up via keep(None).
        let removed = Cell::new(false);
        let mode = restore_on_startup_for_boot(
            name,
            Some(1_700_009_999),
            |_| panic!("stale pre-reboot saved limit must not be restored"),
            |v| { assert_eq!(v, None); removed.set(true); let _ = fs::remove_file(name); Ok(()) },
        ).unwrap();
        assert!(removed.get());
        assert_eq!(mode.saved, None);
        assert!(!path.exists());

        // Current boot record IS restored.
        write_saved_atomic(name, 50000, Some(1_700_009_999)).unwrap();
        let restored = Cell::new(None);
        let mode2 = restore_on_startup_for_boot(
            name,
            Some(1_700_009_999),
            |v| { restored.set(Some(v)); Ok(()) },
            |v| { assert_eq!(v, None); fs::remove_file(name).map_err(|e| e.to_string()) },
        ).unwrap();
        assert_eq!(restored.get(), Some(50000));
        assert_eq!(mode2.saved, None);
        fs::remove_dir_all(&base).unwrap();
    }
}
