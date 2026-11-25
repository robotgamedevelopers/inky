//! Inky mode: while any inky process runs, the GPU may wire the memory this machine's tuning gives it; when the last one
//! stops, the limit goes back to what it was. A small root helper (`inky memd`, installed once by `sudo inky
//! tune-memory`) owns the setting; inky processes connect to it over a Unix socket only the owner can open. A connection
//! closes on a clean exit, a crash, or kill -9 alike, so the memory is given back whenever inky stops; local Unix sockets
//! stay open across macOS sleep, so the limit remains held while inky stays connected. specs/memmode/MemMode.tla
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
pub struct Mode {
    pub saved: Option<u64>,
    pub clients: usize,
}

impl Mode {
    /// A client connected: the first one raises the limit to `target` when the current limit is below `target`,
    /// remembering the limit found (and `keep` stores it before the raise). If the owner already set a limit at or
    /// above `target`, the helper admits the client without lowering or saving it. If the limit cannot be read, the
    /// saved limit cannot be written, or the raise itself fails, the client is refused. A raise that moved the limit
    /// before failing is rolled back to the limit found; the save (in memory and on disk) is forgotten once the limit
    /// is confirmed back where it was, and kept only while the limit is still moved, so the helper's exit paths can
    /// still restore it.
    pub fn connect(
        &mut self,
        target: u64,
        get: impl Fn() -> Option<u64>,
        set: impl Fn(u64) -> Result<(), String>,
        keep: impl Fn(Option<u64>) -> Result<(), String>,
    ) -> Result<(), String> {
        if self.clients == 0 {
            if self.saved.is_none() {
                let found = get().ok_or("the current limit cannot be read; not raising it")?;
                if found >= target {
                    self.clients += 1;
                    return Ok(());
                }
                keep(Some(found)).map_err(|e| format!("the saved limit could not be written; not raising it: {e}"))?;
                self.saved = Some(found);
            }
            let found = self.saved.unwrap();
            if let Err(e) = set(target) {
                if get() != Some(found) {
                    let _ = set(found);
                } // the raise moved the limit before failing: put it back
                if get() == Some(found) {
                    self.saved = None;
                    if let Err(ke) = keep(None) {
                        eprintln!("[inky memd] the unraised limit {found} could not be forgotten on disk: {ke}");
                    }
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
    pub fn close(
        &mut self,
        get: impl Fn() -> Option<u64>,
        set: impl Fn(u64) -> Result<(), String>,
        keep: impl Fn(Option<u64>) -> Result<(), String>,
    ) {
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

fn get_limit() -> Option<u64> {
    crate::tuning::sysctl_u64(LIMIT)
}
/// Sets the limit, reporting failure. Success is confirmed by reading the limit back.
fn set_limit(v: u64) -> Result<(), String> {
    let Ok(mut x) = u32::try_from(v) else { return Err(format!("{v} does not fit {LIMIT}; not set")) };
    let n = std::ffi::CString::new(LIMIT).unwrap();
    let r = unsafe {
        libc::sysctlbyname(
            n.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut x as *mut u32 as *mut libc::c_void,
            4,
        )
    };
    if r != 0 {
        return Err(format!("setting {LIMIT}={v} failed: {}", std::io::Error::last_os_error()));
    }
    if get_limit() != Some(v) {
        return Err(format!("setting {LIMIT}={v} was not confirmed by reading it back"));
    }
    eprintln!("[inky memd] {LIMIT} = {v}");
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum SavedRecord {
    Absent,
    StaleBoot(String, u64),
    Unparsable(String),
    Current(u64),
}

fn is_valid_boot_tag(tag: &str) -> bool {
    !tag.is_empty()
        && tag.len() <= 64
        && tag != "0"
        && !tag.starts_with('-')
        && tag.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
}

pub(super) fn read_saved_record(path: &str, current_boot: Option<&str>) -> Result<SavedRecord, String> {
    match std::fs::symlink_metadata(path) {
        Ok(md) if md.is_file() => {}
        Ok(_) => return Err(format!("the saved limit at {path} is not a regular file")),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(SavedRecord::Absent),
        Err(e) => return Err(format!("could not inspect the saved limit at {path}: {e}")),
    }
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(SavedRecord::Absent),
        Err(e) => return Err(format!("could not read the saved limit at {path}: {e}")),
    };
    let Ok(value) = std::str::from_utf8(&bytes) else {
        return Ok(SavedRecord::Unparsable(format!("invalid UTF-8 in {path}")));
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(SavedRecord::Unparsable(format!("empty saved limit in {path}")));
    }
    if let Some((boot_tag, limit_str)) = trimmed.split_once(':') {
        if !is_valid_boot_tag(boot_tag) {
            return Ok(SavedRecord::Unparsable(format!("could not parse the saved boot tag at {path}: invalid tag")));
        }
        let Ok(limit) = limit_str.parse::<u32>() else {
            return Ok(SavedRecord::Unparsable(format!("could not parse the saved limit at {path}")));
        };
        if let Some(now_boot) = current_boot {
            if boot_tag != now_boot {
                return Ok(SavedRecord::StaleBoot(boot_tag.to_string(), u64::from(limit)));
            }
        }
        return Ok(SavedRecord::Current(u64::from(limit)));
    }
    match trimmed.parse::<u32>() {
        Ok(v) => Ok(SavedRecord::Current(u64::from(v))),
        Err(_) => Ok(SavedRecord::Unparsable(format!("could not parse the saved limit at {path}"))),
    }
}

pub(super) fn write_saved_atomic(path: &str, value: u64, boot: Option<&str>) -> Result<(), String> {
    use std::io::Write as _;
    use std::os::unix::fs::OpenOptionsExt as _;
    if let Some(tag) = boot {
        if !is_valid_boot_tag(tag) {
            return Err(format!("refusing to write invalid boot tag {tag:?} in {path}"));
        }
    }
    let tmp = format!("{path}.tmp");
    let body = match boot {
        Some(tag) => format!("{tag}:{value}\n"),
        None => format!("{value}\n"),
    };
    (|| -> std::io::Result<()> {
        let mut f = std::fs::OpenOptions::new().create(true).truncate(true).write(true).mode(0o600).open(&tmp)?;
        f.write_all(body.as_bytes())?;
        f.sync_all()?;
        drop(f);
        std::fs::rename(&tmp, path)
    })()
    .map_err(|e| format!("could not record the saved limit in {path}: {e}"))
}

pub(super) fn restore_on_startup(
    saved_path: &str,
    set: impl Fn(u64) -> Result<(), String>,
    keep: impl Fn(Option<u64>) -> Result<(), String>,
) -> Result<Mode, String> {
    let mut seed = None;
    match read_saved_record(saved_path, None)? {
        SavedRecord::Absent | SavedRecord::StaleBoot(_, _) => {}
        SavedRecord::Unparsable(reason) => {
            eprintln!("[inky memd] ignoring unparsable saved limit in {saved_path} ({reason}); clearing corrupt file");
            if let Err(e) = keep(None) {
                eprintln!("[inky memd] {e}");
            }
        }
        SavedRecord::Current(v) => {
            eprintln!("[inky memd] the previous helper stopped while the limit was raised: restoring {v}");
            match set(v) {
                Ok(()) => {
                    if let Err(e) = keep(None) {
                        eprintln!("[inky memd] {e}");
                    }
                }
                Err(e) => {
                    eprintln!("[inky memd] restoring {v} failed: {e}; keeping the saved limit in {saved_path}");
                    seed = Some(v);
                }
            }
        }
    }
    Ok(Mode { saved: seed, clients: 0 })
}

pub(super) fn restore_on_startup_for_boot(
    saved_path: &str,
    current_boot: Option<&str>,
    set: impl Fn(u64) -> Result<(), String>,
    keep: impl Fn(Option<u64>) -> Result<(), String>,
) -> Result<Mode, String> {
    match read_saved_record(saved_path, current_boot)? {
        SavedRecord::Absent => Ok(Mode::default()),
        SavedRecord::StaleBoot(old_boot, old_val) => {
            eprintln!("[inky memd] ignoring stale saved limit {old_val} from earlier boot {old_boot}");
            if let Err(e) = keep(None) {
                eprintln!("[inky memd] {e}");
            }
            Ok(Mode::default())
        }
        SavedRecord::Unparsable(reason) => {
            eprintln!("[inky memd] ignoring unparsable saved limit in {saved_path} ({reason}); clearing corrupt file");
            if let Err(e) = keep(None) {
                eprintln!("[inky memd] {e}");
            }
            Ok(Mode::default())
        }
        SavedRecord::Current(_) => restore_on_startup(saved_path, set, keep),
    }
}

fn checked_directory(path: &Path, uid: u32) -> Result<Metadata, String> {
    let metadata =
        fs::symlink_metadata(path).map_err(|e| format!("could not inspect directory {}: {e}", path.display()))?;
    if !metadata.is_dir() || metadata.uid() != uid || metadata.mode() & 0o022 != 0 {
        return Err(format!("{} must be a real directory writable only by UID {uid}", path.display()));
    }
    if uid == 0 {
        require_no_acl(path)?;
    }
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
        unsafe {
            acl_free(acl);
        }
        return Err(format!("{} has an extended access control list", path.display()));
    }
    let e = io::Error::last_os_error();
    // The directory was verified above. Darwin reports ENOENT when its ACL is absent.
    if e.kind() == io::ErrorKind::NotFound {
        return Ok(());
    }
    Err(format!("could not inspect the access control list for {}: {e}", path.display()))
}

fn socket_directory(path: &Path) -> Result<&Path, String> {
    path.parent()
        .filter(|dir| !dir.as_os_str().is_empty())
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
            fs::create_dir_all(parent).map_err(|e| format!("could not create directory {}: {e}", parent.display()))?;
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
    let md = fs::symlink_metadata(path).map_err(|e| format!("could not inspect socket {}: {e}", path.display()))?;
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
    unsafe {
        libc::umask(old_mask);
    }
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
/// Sits comfortably below launchd's default 256 soft descriptor limit and `PLIST_SOFT_NOFILE`.
const MAX_MEMD_CLIENTS: usize = 128;
#[cfg(test)]
const LAUNCHD_DEFAULT_SOFT_NOFILE: usize = 256;
const PLIST_SOFT_NOFILE: usize = 1024;
const PLIST_HARD_NOFILE: usize = 2048;
const TARGET_SOFT_NOFILE: u64 = 4096;
const ACCEPT_LOG_INTERVAL: Duration = Duration::from_secs(5);
const SOCKET_IO_TIMEOUT: Duration = Duration::from_secs(5);
#[cfg(test)]
const ACCEPT_ERROR_BACKOFF: Duration = Duration::from_millis(50);

fn configure_socket_timeouts(stream: &UnixStream, timeout: Duration) -> io::Result<()> {
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    Ok(())
}

#[cfg(test)]
fn accept_error_backoff(err: &io::Error) -> Option<Duration> {
    if matches!(err.kind(), io::ErrorKind::Interrupted | io::ErrorKind::ConnectionAborted) {
        None
    } else {
        Some(ACCEPT_ERROR_BACKOFF)
    }
}

fn wait_for_disconnect<R: Read>(r: &mut R) {
    let mut sink = [0u8; 256];
    loop {
        match r.read(&mut sink) {
            Ok(0) => break,
            Ok(_) => continue,
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut | io::ErrorKind::Interrupted
                ) =>
            {
                continue;
            }
            Err(_) => break,
        }
    }
}

fn sanitize_reason(msg: &str) -> String {
    let clean: String = msg.chars().map(|c| if c == '\r' || c == '\n' { ' ' } else { c }).collect();
    let trimmed = clean.trim();
    if trimmed.is_empty() {
        "request refused".to_string()
    } else {
        trimmed.to_string()
    }
}

pub enum LeaseOutcome {
    /// Helper socket is not installed or not listening; caller stays on macOS default limit.
    NotInstalled,
    /// Helper admitted this process, confirmed active wired-memory limit, and started lease monitor.
    Entered(Lease),
    /// Helper is installed/reachable and refused or failed the handshake with an explicit reason.
    Refused(String),
}

fn parse_enter_reply(line: &str) -> Result<u64, String> {
    let trimmed = line.trim_end_matches(['\r', '\n']);
    if let Some(rest) = trimmed.strip_prefix("ok ") {
        rest.trim().parse::<u64>().map_err(|_| format!("invalid limit in helper reply: {trimmed}"))
    } else if let Some(reason) = trimmed.strip_prefix("err ") {
        Err(sanitize_reason(reason))
    } else if trimmed == "err" {
        Err("helper refused request".to_string())
    } else {
        Err(format!("unexpected helper reply: {}", sanitize_reason(trimmed)))
    }
}

fn serve_client_stream<R: Read, W: Write>(
    reader: R,
    mut writer: W,
    mode: &Mutex<Mode>,
    target: u64,
    get: impl Fn() -> Option<u64>,
    set: impl Fn(u64) -> Result<(), String>,
    keep: impl Fn(Option<u64>) -> Result<(), String>,
) {
    let mut r = BufReader::new(reader);
    let line = match read_bounded_line(&mut r, MAX_MEMD_LINE) {
        Ok(Some(line)) => line,
        Ok(None) => return,
        Err(e) => {
            let reason = sanitize_reason(&format!("invalid client request: {e}"));
            let _ = writer.write_all(format!("err {reason}\n").as_bytes());
            return;
        }
    };
    let cmd = line.trim_end_matches(['\r', '\n']);
    if cmd != "enter" && !cmd.starts_with("enter ") {
        let _ = writer.write_all(b"err unknown command\n");
        return;
    }
    let active_mb = {
        let mut m = mode.lock().unwrap();
        if let Err(e) = m.connect(target, &get, &set, &keep) {
            eprintln!("[inky memd] refused a client: {e}");
            let reason = sanitize_reason(&e);
            let _ = writer.write_all(format!("err {reason}\n").as_bytes());
            return;
        }
        if m.saved.is_some() {
            target
        } else {
            get().unwrap_or(target).max(target)
        }
    };
    if writer.write_all(format!("ok {active_mb}\n").as_bytes()).is_err() {
        mode.lock().unwrap().close(&get, &set, &keep);
        return;
    }
    let _ = writer.flush();
    // Hold until the client's end closes; periodic read timeouts loop cleanly without dropping the client.
    wait_for_disconnect(&mut r);
    mode.lock().unwrap().close(&get, &set, &keep);
}

fn render_plist(owner: u32) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key><array><string>{HELPER}</string><string>memd</string><string>--owner</string><string>{owner}</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>SoftResourceLimits</key><dict><key>NumberOfFiles</key><integer>{PLIST_SOFT_NOFILE}</integer></dict>
  <key>HardResourceLimits</key><dict><key>NumberOfFiles</key><integer>{PLIST_HARD_NOFILE}</integer></dict>
  <key>StandardErrorPath</key><string>/var/log/inky-memd.log</string>
</dict></plist>
"#
    )
}

/// Immutable per-boot identifier from XNU (`kern.bootsessionuuid`), unaffected by `settimeofday`
/// or NTP clock steps.
fn boot_session_id() -> Option<String> {
    let name = c"kern.bootsessionuuid";
    let mut buf = [0u8; 64];
    let mut len = buf.len();
    let rc = unsafe {
        libc::sysctlbyname(name.as_ptr(), buf.as_mut_ptr() as *mut libc::c_void, &mut len, std::ptr::null_mut(), 0)
    };
    if rc != 0 || len == 0 || len > buf.len() {
        return None;
    }
    let end = buf[..len].iter().position(|&b| b == 0).unwrap_or(len);
    let s = std::str::from_utf8(&buf[..end]).ok()?.trim();
    is_valid_boot_tag(s).then(|| s.to_string())
}

pub(crate) fn effective_max_conns(soft_nofile: u64, desired: usize, reserve_fds: usize) -> usize {
    let cap = usize::try_from(soft_nofile).unwrap_or(usize::MAX);
    desired.min(cap.saturating_sub(reserve_fds).max(1))
}

/// Raise soft `RLIMIT_NOFILE` up toward `TARGET_SOFT_NOFILE` (respecting `rlim_max` and macOS
/// `OPEN_MAX`) and clamp `desired` connections safely below the confirmed soft descriptor limit.
pub(crate) fn raise_and_clamp_max_conns(desired: usize, reserve_fds: usize) -> usize {
    let mut rlim: libc::rlimit = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut rlim) } != 0 {
        return desired.min(192);
    }
    let target = TARGET_SOFT_NOFILE as libc::rlim_t;
    let ceiling = if rlim.rlim_max == libc::RLIM_INFINITY { target } else { rlim.rlim_max.min(target) };
    if rlim.rlim_cur < ceiling {
        let raised = libc::rlimit { rlim_cur: ceiling, rlim_max: rlim.rlim_max };
        let _ = unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &raised) };
        let _ = unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut rlim) };
    }
    effective_max_conns(rlim.rlim_cur as u64, desired, reserve_fds)
}

/// Bounded exponential accept-error backoff (`10 ms -> 20 ms -> 40 ms -> 50 ms`) with rate-limited
/// stderr diagnostics so `EMFILE` / `ENFILE` descriptor pressure never spins at 100% CPU or floods logs.
pub(crate) struct AcceptBackoff {
    label: &'static str,
    delay_ms: u64,
    last_log: Option<Instant>,
    suppressed: u64,
}

impl AcceptBackoff {
    pub(crate) fn new(label: &'static str) -> Self {
        Self { label, delay_ms: 0, last_log: None, suppressed: 0 }
    }

    pub(crate) fn on_accept_ok(&mut self) {
        self.delay_ms = 0;
    }

    pub(crate) fn step_at(&mut self, err: &io::Error, now: Instant) -> (Duration, Option<String>) {
        if err.kind() == io::ErrorKind::Interrupted {
            return (Duration::ZERO, None);
        }
        self.delay_ms = match self.delay_ms {
            0 => 10,
            d => (d.saturating_mul(2)).min(50),
        };
        let should_log = match self.last_log {
            None => true,
            Some(prev) => now.saturating_duration_since(prev) >= ACCEPT_LOG_INTERVAL,
        };
        let msg = if should_log {
            let sup = self.suppressed;
            self.suppressed = 0;
            self.last_log = Some(now);
            if sup > 0 {
                Some(format!(
                    "[{}] accept error: {err} (suppressed {sup} earlier accept errors); backing off {} ms",
                    self.label, self.delay_ms
                ))
            } else {
                Some(format!("[{}] accept error: {err}; backing off {} ms", self.label, self.delay_ms))
            }
        } else {
            self.suppressed = self.suppressed.saturating_add(1);
            None
        };
        (Duration::from_millis(self.delay_ms), msg)
    }

    pub(crate) fn on_accept_err(&mut self, err: &io::Error) {
        let (sleep_dur, log_line) = self.step_at(err, Instant::now());
        if let Some(line) = log_line {
            eprintln!("{line}");
        }
        if !sleep_dur.is_zero() {
            std::thread::sleep(sleep_dur);
        }
    }
}

fn read_bounded_line<R: BufRead>(r: &mut R, max_bytes: usize) -> io::Result<Option<String>> {
    let mut line = String::new();
    let n = r.by_ref().take((max_bytes + 1) as u64).read_line(&mut line)?;
    if n == 0 {
        return Ok(None);
    }
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
            if cur >= limit {
                return None;
            }
            match active.compare_exchange_weak(cur, cur + 1, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Some(Self(active.clone())),
                Err(next) => cur = next,
            }
        }
    }
}

impl Drop for ClientGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

fn peer_uid(stream: &UnixStream) -> Option<u32> {
    let (mut uid, mut gid) = (0, 0);
    (unsafe { libc::getpeereid(stream.as_raw_fd(), &mut uid, &mut gid) } == 0).then_some(uid)
}

/// `inky memd --owner UID` is the root helper. The unprivileged `--fake --socket PATH` form uses an in-memory limit.
/// Tests can set its simulated target explicitly with `--fake-target-mb MB`.
pub fn daemon(args: &[String]) {
    if let Err(e) = run_daemon(args) {
        eprintln!("[inky memd] {e}");
        std::process::exit(1);
    }
}

fn daemon_target(args: &[String], fake: bool, tuned_mb: u64) -> Result<u64, String> {
    let Some(index) = args.iter().position(|arg| arg == "--fake-target-mb") else {
        return Ok(tuned_mb);
    };
    if !fake {
        return Err("--fake-target-mb requires --fake".into());
    }
    args.get(index + 1)
        .and_then(|v| v.parse::<u32>().ok())
        .map(u64::from)
        .ok_or_else(|| "--fake-target-mb requires a wired limit in MB (unsigned 32-bit integer)".into())
}

fn run_daemon(args: &[String]) -> Result<(), String> {
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1));
    let owner: u32 = arg("--owner")
        .and_then(|v| v.parse().ok())
        .filter(|&uid| uid != libc::uid_t::MAX)
        .ok_or("memd requires --owner UID")?;
    let fake = args.iter().any(|a| a == "--fake");
    let uid = unsafe { libc::geteuid() };
    let path = if fake {
        if uid == 0 || uid != owner {
            return Err("fake memd requires the current unprivileged UID".into());
        }
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
    let target = daemon_target(args, fake, tuning.wired_limit_mb)?;
    let fake_limit = Arc::new(Mutex::new(0u64));
    let (fl1, fl2) = (fake_limit.clone(), fake_limit.clone());
    let get: Arc<dyn Fn() -> Option<u64> + Send + Sync> =
        if fake { Arc::new(move || Some(*fl1.lock().unwrap())) } else { Arc::new(get_limit) };
    let set: Arc<dyn Fn(u64) -> Result<(), String> + Send + Sync> = if fake {
        Arc::new(move |v| {
            *fl2.lock().unwrap() = v;
            eprintln!("[inky memd] (fake) limit = {v}");
            Ok(())
        })
    } else {
        Arc::new(set_limit)
    };
    // the saved limit on disk (beside the socket with --fake, so tests need no root)
    let saved_path = if fake { format!("{path}.saved") } else { SAVED.to_string() };
    let boot_id = boot_session_id();
    let boot_for_keep = boot_id.clone();
    let sp = saved_path.clone();
    let keep: Arc<dyn Fn(Option<u64>) -> Result<(), String> + Send + Sync> = Arc::new(move |v| match v {
        Some(v) => write_saved_atomic(&sp, v, boot_for_keep.as_deref()),
        None => match std::fs::remove_file(&sp) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(format!("could not forget the saved limit in {sp}: {e}")),
        },
    });
    // a helper that died while the limit was raised left the limit it found on disk: put it back first (`set`
    // confirms by reading back, so its Ok is the confirmation). If the restore fails, the file is kept and Mode is
    // seeded from it below, so the first connect restores that limit instead of reading the still-raised limit
    // as the found one and overwriting the file with it. Saved files from an earlier macOS boot are ignored.
    let mode =
        Arc::new(Mutex::new(restore_on_startup_for_boot(&saved_path, boot_id.as_deref(), |v| set(v), |v| keep(v))?));
    // SIGTERM (launchctl bootout) and SIGINT: give the memory back, then exit (`set` confirms by reading back,
    // so its Ok is the confirmation; Mode::close keeps its own read-back because its `set` is injected)
    {
        let mut sigs: libc::sigset_t = unsafe { std::mem::zeroed() };
        unsafe {
            libc::sigemptyset(&mut sigs);
            libc::sigaddset(&mut sigs, libc::SIGTERM);
            libc::sigaddset(&mut sigs, libc::SIGINT);
            let rc = libc::pthread_sigmask(libc::SIG_BLOCK, &sigs, std::ptr::null_mut());
            if rc != 0 {
                return Err(format!("could not block shutdown signals: {}", io::Error::from_raw_os_error(rc)));
            }
        }
        let (mode, set, keep) = (mode.clone(), set.clone(), keep.clone());
        std::thread::spawn(move || {
            let mut sig = 0;
            let rc = unsafe { libc::sigwait(&sigs, &mut sig) };
            if rc != 0 {
                eprintln!("[inky memd] could not wait for a shutdown signal: {}", io::Error::from_raw_os_error(rc));
            }
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
    let max_clients = raise_and_clamp_max_conns(MAX_MEMD_CLIENTS, 32);
    let listener = bind_socket(Path::new(path), owner, uid, fake)?;
    let active_clients = Arc::new(AtomicUsize::new(0));
    let mut backoff = AcceptBackoff::new("inky memd");
    eprintln!(
        "[inky memd] tuning {} ({chip}, {} GB): inky mode raises {LIMIT} to {target}; listening on {path}",
        tuning.name, tuning.memory_gb
    );
    for incoming in listener.incoming() {
        let s = match incoming {
            Ok(s) => {
                backoff.on_accept_ok();
                s
            }
            Err(e) => {
                backoff.on_accept_err(&e);
                continue;
            }
        };
        if let Err(e) = configure_socket_timeouts(&s, SOCKET_IO_TIMEOUT) {
            eprintln!("[inky memd] could not set socket timeouts: {e}");
            continue;
        }
        // The socket owner can change its mode, so also check the kernel-supplied peer UID.
        if !peer_uid(&s).is_some_and(|uid| uid == owner || uid == 0) {
            let _ = (&s).write_all(b"err unauthorized peer uid\n");
            continue;
        }
        let Some(slot) = ClientGuard::try_acquire(&active_clients, max_clients) else {
            let _ = (&s).write_all(b"err too many clients\n");
            continue;
        };
        let Ok(writer) = s.try_clone() else {
            let _ = (&s).write_all(b"err could not clone client socket\n");
            continue;
        };
        let (mode, get, set, keep) = (mode.clone(), get.clone(), set.clone(), keep.clone());
        std::thread::spawn(move || {
            let _slot = slot;
            serve_client_stream(s, writer, &mode, target, || get(), |v| set(v), |v| keep(v));
        });
    }
    Ok(())
}

enum ConnectErr {
    NotInstalled,
    Refused(String),
}

fn connect_lease(path: &str) -> Result<(UnixStream, u64), ConnectErr> {
    let s = match UnixStream::connect(path) {
        Ok(s) => s,
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused | io::ErrorKind::PermissionDenied
            ) && (e.kind() != io::ErrorKind::PermissionDenied || path != SOCKET) =>
        {
            if e.kind() == io::ErrorKind::PermissionDenied {
                return Err(ConnectErr::Refused(format!("permission denied opening {path}: {e}")));
            }
            return Err(ConnectErr::NotInstalled);
        }
        Err(e) if matches!(e.kind(), io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused) => {
            return Err(ConnectErr::NotInstalled);
        }
        Err(e) => return Err(ConnectErr::Refused(format!("could not connect to {path}: {e}"))),
    };
    if let Err(e) = configure_socket_timeouts(&s, SOCKET_IO_TIMEOUT) {
        return Err(ConnectErr::Refused(format!("could not configure socket timeouts on {path}: {e}")));
    }
    if path == SOCKET && peer_uid(&s) != Some(0) {
        return Err(ConnectErr::Refused(format!("{path} is not owned by root")));
    }
    if let Err(e) = (&s).write_all(format!("enter {}\n", std::process::id()).as_bytes()) {
        return Err(ConnectErr::Refused(format!("could not write handshake to {path}: {e}")));
    }
    let mut r = BufReader::new(&s);
    let line = match read_bounded_line(&mut r, MAX_MEMD_LINE) {
        Ok(Some(line)) => line,
        Ok(None) => return Err(ConnectErr::Refused("helper closed socket without replying".to_string())),
        Err(e) => return Err(ConnectErr::Refused(format!("reading helper reply failed: {e}"))),
    };
    match parse_enter_reply(&line) {
        Ok(v) => Ok((s, v)),
        Err(reason) => Err(ConnectErr::Refused(reason)),
    }
}

/// Keeps the reservation alive until its owner drops it, including across helper restarts
/// or late helper installation (`enter_persistent`).
pub struct Lease {
    pub initial_limit: Option<u64>,
    stop: UnixStream,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Lease {
    fn drop(&mut self) {
        let _ = self.stop.shutdown(std::net::Shutdown::Both);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn maintain_lease(path: String, initial: Option<UnixStream>, stop: UnixStream) {
    let mut had_ever_connected = initial.is_some();
    let mut connection = initial;
    let mut retry_ms = if connection.is_some() { 100 } else { 250 };
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
            if err.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            eprintln!("[inky] cannot monitor memory-helper connection: {err}");
            return;
        }
        if fds[0].revents != 0 {
            return;
        }
        if connection.take().is_some() {
            eprintln!("[inky] lost memory-helper connection; retrying");
            retry_ms = 100;
            // Attempt immediate reconnect in case `tune-memory` already bound the replacement socket;
            // otherwise enter bounded exponential backoff (100..2000 ms).
            if let Ok((stream, limit)) = connect_lease(&path) {
                eprintln!("[inky] memory-helper connection restored; GPU wired limit {limit} MB");
                connection = Some(stream);
                had_ever_connected = true;
            }
        } else if let Ok((stream, limit)) = connect_lease(&path) {
            if had_ever_connected {
                eprintln!("[inky] memory-helper connection restored; GPU wired limit {limit} MB");
            } else {
                eprintln!("[inky] memory-helper connection established; GPU wired limit {limit} MB");
                had_ever_connected = true;
            }
            connection = Some(stream);
            retry_ms = 100;
        } else {
            retry_ms = (retry_ms * 2).min(2_000);
        }
    }
}

/// Enter inky mode and maintain the reservation lease across helper restarts until the returned [`Lease`] drops.
pub fn enter_lease(path: &str) -> LeaseOutcome {
    let (stream, initial_limit) = match connect_lease(path) {
        Ok(pair) => pair,
        Err(ConnectErr::NotInstalled) => return LeaseOutcome::NotInstalled,
        Err(ConnectErr::Refused(reason)) => return LeaseOutcome::Refused(reason),
    };
    let (stop, monitor) = match UnixStream::pair() {
        Ok(pair) => pair,
        Err(e) => return LeaseOutcome::Refused(format!("could not create lease stop socket: {e}")),
    };
    let owned_path = path.to_owned();
    let worker = match std::thread::Builder::new()
        .name("inky-memd-lease".into())
        .spawn(move || maintain_lease(owned_path, Some(stream), monitor))
    {
        Ok(w) => w,
        Err(e) => return LeaseOutcome::Refused(format!("could not spawn lease monitor thread: {e}")),
    };
    LeaseOutcome::Entered(Lease { initial_limit: Some(initial_limit), stop, worker: Some(worker) })
}

#[cfg(test)]
pub fn enter(path: &str) -> Option<Lease> {
    match enter_lease(path) {
        LeaseOutcome::Entered(lease) => Some(lease),
        LeaseOutcome::NotInstalled | LeaseOutcome::Refused(_) => None,
    }
}

/// Persistent lease monitor for long-running `inky serve`: connects immediately if the helper
/// is already running, and otherwise polls cleanly in the background so starting or reinstalling
/// `memd` (`sudo inky tune-memory`) while `serve` is live establishes/restores the reservation.
pub fn enter_persistent(path: &str) -> Option<Lease> {
    let (stream, initial_limit) = match connect_lease(path) {
        Ok((s, limit)) => (Some(s), Some(limit)),
        Err(ConnectErr::NotInstalled) => (None, None),
        Err(ConnectErr::Refused(reason)) => {
            eprintln!("[inky] inky mode refused ({reason}); polling for valid helper");
            (None, None)
        }
    };
    let (stop, monitor) = UnixStream::pair().ok()?;
    let path = path.to_owned();
    let worker = std::thread::Builder::new()
        .name("inky-memd-lease".into())
        .spawn(move || maintain_lease(path, stream, monitor))
        .ok()?;
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
    let out = Command::new(LAUNCHCTL)
        .args(["print", &format!("system/{LABEL}")])
        .output()
        .map_err(|e| format!("could not run launchctl print: {e}"))?;
    match out.status.code() {
        Some(0) => Ok(true),
        Some(113) => Ok(false), // launchctl: could not find specified service
        _ => Err(format!("launchctl print failed ({}): {}", out.status, String::from_utf8_lossy(&out.stderr).trim())),
    }
}

fn require_service_absent() -> Result<(), String> {
    if service_is_loaded()? {
        return Err(format!("launchd still has service {LABEL}"));
    }
    Ok(())
}

fn stop_helper() -> Result<(), String> {
    if !service_is_loaded()? {
        return Ok(());
    }
    let status = Command::new(LAUNCHCTL)
        .args(["bootout", &format!("system/{LABEL}")])
        .status()
        .map_err(|e| format!("could not run launchctl bootout: {e}"))?;
    if !status.success() {
        return Err(format!("launchctl bootout failed ({status})"));
    }
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
    match read_saved_record(SAVED, boot_session_id().as_deref())? {
        SavedRecord::Current(value) => {
            set_limit(value).map_err(|e| format!("restoring {value} failed: {e}; kept {SAVED} and the helper"))?;
            remove_file_if_present(Path::new(SAVED))?;
        }
        SavedRecord::StaleBoot(_, _) | SavedRecord::Unparsable(_) => remove_file_if_present(Path::new(SAVED))?,
        SavedRecord::Absent => {}
    }
    checked_directory(Path::new("/Library"), 0)?;
    for dir in [Path::new(HELPER_DIR), Path::new(PLIST).parent().unwrap()] {
        match fs::symlink_metadata(dir) {
            Ok(_) => {
                checked_directory(dir, 0)?;
            }
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
    for file in files {
        remove_file_if_present(Path::new(file))?;
    }
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
    let owner: u32 = std::env::var("SUDO_UID")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&uid| uid != libc::uid_t::MAX)
        .ok_or("run it with sudo from the owner's account")?;
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
    if let SavedRecord::Unparsable(reason) = read_saved_record(SAVED, boot_session_id().as_deref())? {
        eprintln!("[inky memd] clearing unparsable saved limit at {SAVED} ({reason})");
        remove_file_if_present(Path::new(SAVED))?;
    }
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
    let plist = render_plist(owner);
    remove_file_if_present(Path::new(PLIST_TEMP))?;
    let write_plist = || -> io::Result<()> {
        let mut staged = OpenOptions::new().create_new(true).write(true).mode(0o600).open(PLIST_TEMP)?;
        staged.write_all(plist.as_bytes())?;
        staged.set_permissions(Permissions::from_mode(0o644))?;
        fs::rename(PLIST_TEMP, PLIST)
    };
    write_plist().map_err(|e| format!("could not write {PLIST}: {e}"))?;
    let status = Command::new(LAUNCHCTL)
        .args(["bootstrap", "system", PLIST])
        .status()
        .map_err(|e| format!("could not run launchctl bootstrap: {e}"))?;
    if !status.success() {
        return Err(format!("launchctl bootstrap failed ({status})"));
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while let Err(e) = check_socket(socket, owner) {
        if Instant::now() >= deadline {
            return Err(format!("the helper did not publish its socket: {e}"));
        }
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
    use std::path::PathBuf;
    use std::sync::atomic::AtomicU64;
    // bind_socket temporarily sets process-wide umask(0o177); serialize filesystem tests within this binary.
    static TEST_FS_LOCK: Mutex<()> = Mutex::new(());

    fn scratch_dir(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(1);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("md")
            .join(format!("{tag}-{}-{n}", std::process::id()))
    }

    #[test]
    fn explicit_target_is_only_for_the_fake_daemon() {
        assert_eq!(daemon_target(&[], true, 0).unwrap(), 0);
        assert_eq!(daemon_target(&[], true, 65536).unwrap(), 65536);
        assert_eq!(daemon_target(&[], false, 65536).unwrap(), 65536);
        let args = ["--fake-target-mb".to_owned(), "8192".to_owned()];
        assert_eq!(daemon_target(&args, true, 0).unwrap(), 8192);
        assert!(daemon_target(&args, false, 65536).is_err());
        assert!(daemon_target(&args[..1], true, 0).is_err());
        assert!(daemon_target(&[args[0].clone(), "-1".into()], true, 0).is_err());
    }

    // Darwin bounds the full pathname stored in sockaddr_un, including the scratch directory.
    struct SocketTestDir(PathBuf);

    impl SocketTestDir {
        fn new(tag: &str) -> Self {
            let dir = scratch_dir(tag);
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("creating socket test directory");
            Self(dir)
        }
    }

    impl Drop for SocketTestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn dropping_client_lease_closes_its_connection() {
        let _fs = TEST_FS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let scratch = SocketTestDir::new("lease");
        let path = scratch.0.join("owned.sock");
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
        assert_eq!(lease.initial_limit, Some(12345));
        peer.set_nonblocking(true).unwrap();
        let mut byte = [0; 1];
        assert_eq!(peer.read(&mut byte).unwrap_err().kind(), io::ErrorKind::WouldBlock);
        peer.set_nonblocking(false).unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        drop(lease);
        assert_eq!(peer.read(&mut byte).unwrap(), 0);
    }

    #[test]
    fn raised_while_connected_and_restored_after_the_last() {
        let (lim, disk) = (Cell::new(50000u64), Cell::new(None));
        let mut m = Mode::default();
        m.connect(
            118000,
            || Some(lim.get()),
            |v| {
                lim.set(v);
                Ok(())
            },
            |v| {
                disk.set(v);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(disk.get(), Some(50000), "the limit found is on disk before the raise");
        m.connect(
            118000,
            || Some(lim.get()),
            |v| {
                lim.set(v);
                Ok(())
            },
            |v| {
                disk.set(v);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(lim.get(), 118000);
        m.close(
            || Some(lim.get()),
            |v| {
                lim.set(v);
                Ok(())
            },
            |v| {
                disk.set(v);
                Ok(())
            },
        );
        assert_eq!(lim.get(), 118000, "still one client: still raised");
        m.close(
            || Some(lim.get()),
            |v| {
                lim.set(v);
                Ok(())
            },
            |v| {
                disk.set(v);
                Ok(())
            },
        );
        assert_eq!(
            (lim.get(), m.saved, disk.get()),
            (50000, None, None),
            "the limit found, not the default; nothing left on disk"
        );
    }
    #[test]
    fn owner_higher_or_equal_limit_is_never_lowered_or_saved() {
        for initial in [118_000u64, 130_000u64] {
            let lim = Cell::new(initial);
            let mut m = Mode::default();
            m.connect(
                118_000,
                || Some(lim.get()),
                |_| panic!("set must not be called when current limit {initial} >= target"),
                |_| panic!("keep must not be called when current limit {initial} >= target"),
            )
            .unwrap();
            assert_eq!((lim.get(), m.clients, m.saved), (initial, 1, None));
            m.connect(
                118_000,
                || panic!("get must not be called while clients > 0"),
                |_| panic!("set must not be called while clients > 0"),
                |_| panic!("keep must not be called while clients > 0"),
            )
            .unwrap();
            assert_eq!((lim.get(), m.clients, m.saved), (initial, 2, None));
            m.close(
                || panic!("get must not be called on non-final close"),
                |_| panic!("set must not be called on non-final close"),
                |_| panic!("keep must not be called on non-final close"),
            );
            m.close(
                || panic!("get must not be called when nothing was saved"),
                |_| panic!("set must not be called when nothing was saved"),
                |_| panic!("keep must not be called when nothing was saved"),
            );
            assert_eq!((lim.get(), m.clients, m.saved), (initial, 0, None));
        }
    }
    #[test]
    fn a_second_session_saves_afresh() {
        let (lim, disk) = (Cell::new(0u64), Cell::new(None));
        let mut m = Mode::default();
        m.connect(
            118000,
            || Some(lim.get()),
            |v| {
                lim.set(v);
                Ok(())
            },
            |v| {
                disk.set(v);
                Ok(())
            },
        )
        .unwrap();
        m.close(
            || Some(lim.get()),
            |v| {
                lim.set(v);
                Ok(())
            },
            |v| {
                disk.set(v);
                Ok(())
            },
        );
        lim.set(70000); // the owner changes it between sessions
        m.connect(
            118000,
            || Some(lim.get()),
            |v| {
                lim.set(v);
                Ok(())
            },
            |v| {
                disk.set(v);
                Ok(())
            },
        )
        .unwrap();
        m.close(
            || Some(lim.get()),
            |v| {
                lim.set(v);
                Ok(())
            },
            |v| {
                disk.set(v);
                Ok(())
            },
        );
        assert_eq!(lim.get(), 70000);
    }
    #[test]
    fn an_unreadable_limit_is_never_raised() {
        let (lim, disk) = (Cell::new(50000u64), Cell::new(None));
        let mut m = Mode::default();
        assert!(m
            .connect(
                118000,
                || None,
                |v| {
                    lim.set(v);
                    Ok(())
                },
                |v| {
                    disk.set(v);
                    Ok(())
                }
            )
            .is_err());
        assert_eq!((lim.get(), m.clients, disk.get()), (50000, 0, None));
    }
    #[test]
    fn a_failed_saved_write_refuses_the_raise() {
        let (lim, disk) = (Cell::new(50000u64), Cell::new(None::<u64>));
        let mut m = Mode::default();
        assert!(m
            .connect(
                118000,
                || Some(lim.get()),
                |v| {
                    lim.set(v);
                    Ok(())
                },
                |_| Err("the saved file cannot be written".to_string())
            )
            .is_err());
        assert_eq!(
            (lim.get(), m.clients, m.saved, disk.get()),
            (50000, 0, None, None),
            "no raise and no client when the saved limit cannot be written"
        );
    }
    #[test]
    fn a_failed_raise_is_refused() {
        let (lim, disk) = (Cell::new(50000u64), Cell::new(None));
        let mut m = Mode::default();
        assert!(m
            .connect(
                118000,
                || Some(lim.get()),
                |_| Err("the limit cannot be set".to_string()),
                |v| {
                    disk.set(v);
                    Ok(())
                }
            )
            .is_err());
        assert_eq!(
            (lim.get(), m.clients, m.saved, disk.get()),
            (50000, 0, None, None),
            "no client and nothing saved when the raise itself fails and the limit never moved"
        );
    }
    #[test]
    fn a_raise_that_moves_the_limit_then_fails_is_rolled_back() {
        let (lim, disk, calls) = (Cell::new(50000u64), Cell::new(None), Cell::new(0u32));
        let mut m = Mode::default();
        // the first set moves the limit and then reports failure (its read-back differs); the rollback set succeeds
        let set = |v: u64| {
            calls.set(calls.get() + 1);
            lim.set(v);
            if calls.get() == 1 {
                Err("setting was not confirmed by reading it back".to_string())
            } else {
                Ok(())
            }
        };
        assert!(m
            .connect(
                118000,
                || Some(lim.get()),
                set,
                |v| {
                    disk.set(v);
                    Ok(())
                }
            )
            .is_err());
        assert_eq!(
            (lim.get(), m.clients, m.saved, disk.get()),
            (50000, 0, None, None),
            "a refused connect leaves the limit where it found it and saves nothing"
        );
    }
    #[test]
    fn a_seeded_saved_limit_is_not_overwritten_by_the_raised_limit() {
        // the daemon restarted after a failed startup restore: Mode is seeded from the saved file while the
        // limit is still raised, so the first connect must not read the raised limit as the found limit
        let (lim, disk) = (Cell::new(118000u64), Cell::new(Some(50000u64)));
        let mut m = Mode { saved: Some(50000), clients: 0 };
        m.connect(
            118000,
            || Some(lim.get()),
            |v| {
                lim.set(v);
                Ok(())
            },
            |v| {
                disk.set(v);
                Ok(())
            },
        )
        .unwrap();
        m.close(
            || Some(lim.get()),
            |v| {
                lim.set(v);
                Ok(())
            },
            |v| {
                disk.set(v);
                Ok(())
            },
        );
        assert_eq!(
            (lim.get(), m.saved, disk.get()),
            (50000, None, None),
            "the seeded limit, not the raised one, is restored"
        );
    }
    #[test]
    fn a_confirmed_restore_with_a_failed_forget_saves_afresh_next_time() {
        let (lim, disk) = (Cell::new(50000u64), Cell::new(None));
        let mut m = Mode::default();
        // keep records the save but cannot forget it on disk
        let keep = |v: Option<u64>| {
            if v.is_none() {
                Err("the saved file cannot be removed".to_string())
            } else {
                disk.set(v);
                Ok(())
            }
        };
        m.connect(
            118000,
            || Some(lim.get()),
            |v| {
                lim.set(v);
                Ok(())
            },
            keep,
        )
        .unwrap();
        m.close(
            || Some(lim.get()),
            |v| {
                lim.set(v);
                Ok(())
            },
            keep,
        );
        assert_eq!(
            (lim.get(), m.saved, disk.get()),
            (50000, None, Some(50000)),
            "the restore stands and the in-memory save is cleared even though the file could not be forgotten"
        );
        lim.set(70000); // the owner changes it between sessions
        m.connect(
            118000,
            || Some(lim.get()),
            |v| {
                lim.set(v);
                Ok(())
            },
            |v| {
                disk.set(v);
                Ok(())
            },
        )
        .unwrap();
        m.close(
            || Some(lim.get()),
            |v| {
                lim.set(v);
                Ok(())
            },
            |v| {
                disk.set(v);
                Ok(())
            },
        );
        assert_eq!(lim.get(), 70000, "the next session saves the owner's newer limit afresh");
    }
    #[test]
    fn the_socket_directory_must_be_a_real_directory() {
        let _fs = TEST_FS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = scratch_dir("dirtest");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        fs::set_permissions(&base, Permissions::from_mode(0o700)).unwrap();
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
    fn startup_survives_unparsable_saves_and_refuses_symlinks_or_unreadable_files() {
        let _fs = TEST_FS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = scratch_dir("save");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let path = base.join("saved");
        let name = path.to_str().unwrap();
        let set_never = |_| panic!("startup tried to restore an unusable saved limit");
        assert!(restore_on_startup(name, set_never, |_| panic!("absent save should not call keep"))
            .unwrap()
            .saved
            .is_none());

        for bytes in
            [b"".as_slice(), b"malformed", b"4294967296", b"\xff", b"0:50000", b"-5:50000", b"1700000000:notanumber"]
        {
            fs::write(&path, bytes).unwrap();
            let cleared = Cell::new(false);
            let mode = restore_on_startup(name, set_never, |v| {
                assert_eq!(v, None);
                cleared.set(true);
                fs::remove_file(&path).map_err(|e| e.to_string())
            })
            .expect("helper startup must survive an unparsable saved file");
            assert!(mode.saved.is_none());
            assert!(cleared.get(), "corrupt saved file should be cleared via keep(None)");
            assert!(!path.exists());

            fs::write(&path, bytes).unwrap();
            let cleared_boot = Cell::new(false);
            let mode_boot =
                restore_on_startup_for_boot(name, Some("00112233-4455-6677-8899-AABBCCDDEEFF"), set_never, |v| {
                    assert_eq!(v, None);
                    cleared_boot.set(true);
                    fs::remove_file(&path).map_err(|e| e.to_string())
                })
                .expect("boot-tagged startup must survive an unparsable saved file");
            assert!(mode_boot.saved.is_none());
            assert!(cleared_boot.get());
            assert!(!path.exists());
        }
        fs::write(&path, b"50000").unwrap();
        fs::set_permissions(&path, Permissions::from_mode(0o000)).unwrap();
        if unsafe { libc::geteuid() } != 0 {
            let result = restore_on_startup(name, set_never, |_| Ok(()));
            assert!(matches!(result, Err(e) if e.contains("could not read")));
        }
        fs::set_permissions(&path, Permissions::from_mode(0o600)).unwrap();
        fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(base.join("missing"), &path).unwrap();
        assert!(restore_on_startup(name, set_never, |_| Ok(())).is_err());
        fs::remove_dir_all(&base).unwrap();
    }
    #[test]
    #[ignore = "binds real Unix domain sockets; run in CI socket integration step"]
    fn fake_socket_rebinds_with_private_mode_and_reports_its_peer_uid() {
        // bind_socket changes the process umask around bind. Run the real operation
        // in its own process so other tests can create directories concurrently.
        const CHILD: &str = "INKY_TEST_MEMD_BIND_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--ignored",
                    "--exact",
                    "memd::tests::fake_socket_rebinds_with_private_mode_and_reports_its_peer_uid",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success() && stdout.contains("1 passed;"),
                "isolated socket check failed: {stdout} {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        let _fs = TEST_FS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let scratch = SocketTestDir::new("bind");
        let path = scratch.0.join("helper").join("memd.sock");
        let uid = unsafe { libc::geteuid() };
        drop(bind_socket(&path, uid, uid, true).unwrap());
        let listener = bind_socket(&path, uid, uid, true).unwrap();
        assert!(check_socket(&path, uid).is_ok());
        let client = UnixStream::connect(&path).unwrap();
        configure_socket_timeouts(&client, SOCKET_IO_TIMEOUT).unwrap();
        assert_eq!(client.read_timeout().unwrap(), Some(SOCKET_IO_TIMEOUT));
        assert_eq!(client.write_timeout().unwrap(), Some(SOCKET_IO_TIMEOUT));
        let (accepted, _) = listener.accept().unwrap();
        configure_socket_timeouts(&accepted, SOCKET_IO_TIMEOUT).unwrap();
        assert_eq!(peer_uid(&client), Some(uid));
        assert_eq!(peer_uid(&accepted), Some(uid));
    }
    #[test]
    #[ignore = "binds real Unix domain sockets; run in CI socket integration step"]
    fn legacy_cleanup_removes_the_socket_and_retries_after_a_directory_error() {
        let _fs = TEST_FS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let scratch = SocketTestDir::new("old");
        let base = scratch.0.join("legacy");
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
        let base = scratch_dir("homebrew");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        fs::set_permissions(&base, Permissions::from_mode(0o775)).unwrap();
        let path = base.join("inky-memd");
        remove_legacy_helper(&path).unwrap();
        fs::write(&path, b"retained").unwrap();
        assert!(remove_legacy_helper(&path).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"retained");
        fs::remove_dir_all(&base).unwrap();
    }
    #[test]
    fn an_unconfirmed_restore_keeps_the_saved_limit() {
        let (lim, disk) = (Cell::new(50000u64), Cell::new(None));
        let mut m = Mode::default();
        m.connect(
            118000,
            || Some(lim.get()),
            |v| {
                lim.set(v);
                Ok(())
            },
            |v| {
                disk.set(v);
                Ok(())
            },
        )
        .unwrap();
        // a set that reports failure: the restore is not confirmed, so the saved limit is kept
        m.close(
            || Some(lim.get()),
            |_| Err("the limit cannot be set".to_string()),
            |v| {
                disk.set(v);
                Ok(())
            },
        );
        assert_eq!((lim.get(), m.saved, disk.get()), (118000, Some(50000), Some(50000)));
        // a set that claims success but does not move the limit: reading it back does not confirm, so it is still kept
        m.clients = 1;
        m.close(
            || Some(lim.get()),
            |_| Ok(()),
            |v| {
                disk.set(v);
                Ok(())
            },
        );
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
    fn protocol_stream_propagates_refusal_reasons_timeouts_and_higher_limits() {
        // 1. Keep-higher reports the owner's higher limit ("ok 130000\n") and restores nothing on EOF.
        let mode = Mutex::new(Mode::default());
        let lim = Cell::new(130_000u64);
        let mut out = Vec::new();
        serve_client_stream(
            io::Cursor::new(b"enter 99\n".as_slice()),
            &mut out,
            &mode,
            118_000,
            || Some(lim.get()),
            |_| panic!("set must not run when owner limit >= target"),
            |_| panic!("keep must not run when owner limit >= target"),
        );
        let reply = String::from_utf8(out).unwrap();
        assert_eq!(reply, "ok 130000\n");
        assert_eq!(parse_enter_reply(&reply), Ok(130_000));
        assert_eq!(mode.lock().unwrap().clients, 0);

        // 2. Helper refusal reason propagates verbatim through parse_enter_reply.
        let mut refused_out = Vec::new();
        let lim2 = Cell::new(50_000u64);
        let disk2 = Cell::new(None);
        serve_client_stream(
            io::Cursor::new(b"enter 100\n".as_slice()),
            &mut refused_out,
            &mode,
            118_000,
            || Some(lim2.get()),
            |_| Err("sysctl Operation not permitted".into()),
            |v| {
                disk2.set(v);
                Ok(())
            },
        );
        let refused_line = String::from_utf8(refused_out).unwrap();
        assert_eq!(
            parse_enter_reply(&refused_line),
            Err("the limit could not be raised: sysctl Operation not permitted".into())
        );
        assert_eq!(parse_enter_reply("err too many clients\n"), Err("too many clients".into()));

        // 3. Unknown command is rejected with an explicit reason.
        let mut bad_cmd_out = Vec::new();
        serve_client_stream(
            io::Cursor::new(b"status\n".as_slice()),
            &mut bad_cmd_out,
            &mode,
            118_000,
            || Some(50_000),
            |_| Ok(()),
            |_| Ok(()),
        );
        assert_eq!(parse_enter_reply(&String::from_utf8(bad_cmd_out).unwrap()), Err("unknown command".into()));

        // 4. Handshake read timeout aborts before connect; post-handshake read timeouts keep holding until EOF.
        struct ScriptedReader {
            steps: Vec<io::Result<&'static [u8]>>,
            idx: usize,
        }
        impl Read for ScriptedReader {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                if self.idx >= self.steps.len() {
                    return Ok(0);
                }
                let step = std::mem::replace(&mut self.steps[self.idx], Ok(&[]));
                self.idx += 1;
                match step {
                    Ok(bytes) => {
                        let n = bytes.len().min(buf.len());
                        buf[..n].copy_from_slice(&bytes[..n]);
                        Ok(n)
                    }
                    Err(e) => Err(e),
                }
            }
        }
        let mut timeout_handshake_out = Vec::new();
        serve_client_stream(
            ScriptedReader { steps: vec![Err(io::Error::new(io::ErrorKind::TimedOut, "handshake timed out"))], idx: 0 },
            &mut timeout_handshake_out,
            &mode,
            118_000,
            || panic!("connect must not run after handshake timeout"),
            |_| panic!("set must not run after handshake timeout"),
            |_| panic!("keep must not run after handshake timeout"),
        );
        assert!(parse_enter_reply(&String::from_utf8(timeout_handshake_out).unwrap())
            .unwrap_err()
            .contains("handshake timed out"));

        let lim3 = Cell::new(50_000u64);
        let disk3 = Cell::new(None);
        let timeouts_observed = Cell::new(0usize);
        struct CountingHoldReader {
            sent_handshake: bool,
            timeouts_left: usize,
            observed: *const Cell<usize>,
        }
        impl Read for CountingHoldReader {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                if !self.sent_handshake {
                    self.sent_handshake = true;
                    let msg = b"enter 7\n";
                    buf[..msg.len()].copy_from_slice(msg);
                    return Ok(msg.len());
                }
                if self.timeouts_left > 0 {
                    self.timeouts_left -= 1;
                    unsafe {
                        (*self.observed).set((*self.observed).get() + 1);
                    }
                    return Err(io::Error::new(
                        if self.timeouts_left.is_multiple_of(2) {
                            io::ErrorKind::TimedOut
                        } else {
                            io::ErrorKind::WouldBlock
                        },
                        "periodic hold timeout",
                    ));
                }
                Ok(0)
            }
        }
        let mut hold_out = Vec::new();
        serve_client_stream(
            CountingHoldReader { sent_handshake: false, timeouts_left: 3, observed: &timeouts_observed },
            &mut hold_out,
            &mode,
            118_000,
            || Some(lim3.get()),
            |v| {
                lim3.set(v);
                Ok(())
            },
            |v| {
                disk3.set(v);
                Ok(())
            },
        );
        assert_eq!(String::from_utf8(hold_out).unwrap(), "ok 118000\n");
        assert_eq!(timeouts_observed.get(), 3, "periodic read timeouts must not drop held connection early");
        assert_eq!((lim3.get(), disk3.get(), mode.lock().unwrap().clients), (50_000, None, 0));
    }
    #[test]
    fn accept_backoff_and_descriptor_headroom_invariants() {
        assert_eq!(accept_error_backoff(&io::Error::from(io::ErrorKind::Interrupted)), None);
        assert_eq!(accept_error_backoff(&io::Error::from(io::ErrorKind::ConnectionAborted)), None);
        for errno in [libc::EMFILE, libc::ENFILE, libc::ENOMEM, libc::ENOBUFS] {
            assert_eq!(
                accept_error_backoff(&io::Error::from_raw_os_error(errno)),
                Some(ACCEPT_ERROR_BACKOFF),
                "errno {errno} must back off instead of spinning"
            );
        }
        const {
            assert!(
                MAX_MEMD_CLIENTS < LAUNCHD_DEFAULT_SOFT_NOFILE && MAX_MEMD_CLIENTS + 64 <= LAUNCHD_DEFAULT_SOFT_NOFILE,
                "client cap must leave ample descriptor headroom below launchd default"
            );
            assert!(MAX_MEMD_CLIENTS * 4 <= PLIST_SOFT_NOFILE && PLIST_SOFT_NOFILE < PLIST_HARD_NOFILE);
        };
        let xml = render_plist(501);
        assert!(xml.contains(&format!(
            "<key>SoftResourceLimits</key><dict><key>NumberOfFiles</key><integer>{PLIST_SOFT_NOFILE}</integer></dict>"
        )));
        assert!(xml.contains(&format!(
            "<key>HardResourceLimits</key><dict><key>NumberOfFiles</key><integer>{PLIST_HARD_NOFILE}</integer></dict>"
        )));
    }
    #[test]
    fn boot_tagged_saved_limit_ignores_prior_boot_and_syncs_atomically() {
        let _fs = TEST_FS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = scratch_dir("boot");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let path = base.join("saved");
        let name = path.to_str().unwrap();

        let boot1 = "F9A6C86D-08AB-4F34-9755-36CD42A50743";
        let boot2 = "00112233-4455-6677-8899-AABBCCDDEEFF";
        write_saved_atomic(name, 50000, Some(boot1)).unwrap();
        assert_eq!(read_saved_record(name, Some(boot1)), Ok(SavedRecord::Current(50000)));
        assert_eq!(read_saved_record(name, Some(boot2)), Ok(SavedRecord::StaleBoot(boot1.to_string(), 50000)));

        // Stale boot record is ignored (no set() call) and cleaned up via keep(None).
        let removed = Cell::new(false);
        let mode = restore_on_startup_for_boot(
            name,
            Some(boot2),
            |_| panic!("stale pre-reboot saved limit must not be restored"),
            |v| {
                assert_eq!(v, None);
                removed.set(true);
                let _ = fs::remove_file(name);
                Ok(())
            },
        )
        .unwrap();
        assert!(removed.get());
        assert_eq!(mode.saved, None);
        assert!(!path.exists());

        // Current boot UUID record IS restored across wall-clock steps.
        write_saved_atomic(name, 50000, Some(boot2)).unwrap();
        let restored = Cell::new(None);
        let mode2 = restore_on_startup_for_boot(
            name,
            Some(boot2),
            |v| {
                restored.set(Some(v));
                Ok(())
            },
            |v| {
                assert_eq!(v, None);
                fs::remove_file(name).map_err(|e| e.to_string())
            },
        )
        .unwrap();
        assert_eq!(restored.get(), Some(50000));
        assert_eq!(mode2.saved, None);

        // Malformed boot tags are rejected as Unparsable rather than misread.
        for bad in [":50000", "bad tag:50000", "boot_underscore:50000"] {
            fs::write(&path, bad).unwrap();
            assert!(
                matches!(read_saved_record(name, Some(boot2)), Ok(SavedRecord::Unparsable(_))),
                "expected {bad:?} to be rejected as Unparsable"
            );
        }
        assert!(write_saved_atomic(name, 50000, Some("bad:tag")).is_err());

        // On macOS, kern.bootsessionuuid is readable and stable across calls.
        #[cfg(target_os = "macos")]
        {
            let live1 = boot_session_id().expect("macOS kern.bootsessionuuid must be present");
            let live2 = boot_session_id().expect("macOS kern.bootsessionuuid must remain stable");
            assert_eq!(live1, live2);
            assert!(is_valid_boot_tag(&live1));
        }
        fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn accept_backoff_and_fd_clamping_bound_emfile_pressure() {
        assert_eq!(effective_max_conns(256, 512, 64), 192);
        assert_eq!(effective_max_conns(256, 256, 32), 224);
        assert_eq!(effective_max_conns(4096, 512, 64), 512);
        assert_eq!(effective_max_conns(16, 512, 64), 1);

        let clamped = raise_and_clamp_max_conns(512, 64);
        assert!((1..=512).contains(&clamped));

        let mut bo = AcceptBackoff::new("test");
        let t0 = Instant::now();
        let eintr = io::Error::from(io::ErrorKind::Interrupted);
        let (d0, m0) = bo.step_at(&eintr, t0);
        assert_eq!(d0, Duration::ZERO);
        assert!(m0.is_none());

        let emfile = io::Error::from_raw_os_error(libc::EMFILE);
        let (d1, m1) = bo.step_at(&emfile, t0);
        assert_eq!(d1, Duration::from_millis(10));
        assert!(m1.as_deref().unwrap().contains("backing off 10 ms"));

        // Immediate repeat errors step 20 -> 40 -> 50 -> 50 ms while suppressing log spam.
        let (d2, m2) = bo.step_at(&emfile, t0 + Duration::from_millis(10));
        let (d3, m3) = bo.step_at(&emfile, t0 + Duration::from_millis(30));
        let (d4, m4) = bo.step_at(&emfile, t0 + Duration::from_millis(70));
        let (d5, m5) = bo.step_at(&emfile, t0 + Duration::from_millis(120));
        assert_eq!([d2.as_millis(), d3.as_millis(), d4.as_millis(), d5.as_millis()], [20, 40, 50, 50]);
        assert!(m2.is_none() && m3.is_none() && m4.is_none() && m5.is_none());

        // After ACCEPT_LOG_INTERVAL, emits summary with exact suppressed count.
        let (d6, m6) = bo.step_at(&emfile, t0 + ACCEPT_LOG_INTERVAL + Duration::from_millis(1));
        assert_eq!(d6, Duration::from_millis(50));
        assert!(m6.as_deref().unwrap().contains("suppressed 4 earlier accept errors"));

        // Successful accept resets delay back to initial 10 ms step.
        bo.on_accept_ok();
        let (d7, _) = bo.step_at(&emfile, t0 + ACCEPT_LOG_INTERVAL + Duration::from_secs(6));
        assert_eq!(d7, Duration::from_millis(10));
    }
}
