//! Inky mode: while any inky process runs, the GPU may wire the memory this machine's tuning gives it; when the last one
//! stops, the limit goes back to what it was. A small root helper (`inky memd`, installed once by `sudo inky
//! tune-memory`) owns the setting; inky processes connect to it over a Unix socket only the owner can open. A connection
//! closes on a clean exit, a crash, kill -9 or sleep alike, so the memory is always given back. specs/memmode/MemMode.tla
//! models the helper.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Arc, Mutex};

pub const SOCKET: &str = "/var/run/inky/memd.sock";
const LIMIT: &str = "iogpu.wired_limit_mb";
/// the helper's binary: a root-owned copy in a directory only root can write
const HELPER_DIR: &str = "/Library/PrivilegedHelperTools";
const HELPER: &str = "/Library/PrivilegedHelperTools/inky-memd";
/// where an earlier version installed it; removed on install
const OLD_HELPER: &str = "/usr/local/libexec/inky-memd";
/// the limit found when the helper raised it, kept on disk while raised, so a helper that dies while raised restores it
/// when it starts again
const SAVED: &str = "/var/db/inky-memd.saved";
const PLIST: &str = "/Library/LaunchDaemons/com.robotgamedevelopers.inky.memd.plist";
const LABEL: &str = "com.robotgamedevelopers.inky.memd";

/// The helper's state (MemMode.tla): the limit it found when it raised it, and how many clients are connected.
#[derive(Default)]
pub struct Mode { pub saved: Option<u64>, pub clients: usize }

impl Mode {
    /// A client connected: the first one raises the limit to `target`, remembering the limit found (and `keep` stores it
    /// before the raise). If the limit cannot be read, nothing is raised and the client is refused.
    pub fn connect(&mut self, target: u64, get: impl Fn() -> Option<u64>, set: impl Fn(u64), keep: impl Fn(Option<u64>)) -> Result<(), String> {
        if self.clients == 0 {
            let found = get().ok_or("the current limit cannot be read; not raising it")?;
            keep(Some(found));
            self.saved = Some(found);
            set(target);
        }
        self.clients += 1;
        Ok(())
    }
    /// A client's connection closed: the last one restores the limit found, then forgets it on disk.
    pub fn close(&mut self, set: impl Fn(u64), keep: impl Fn(Option<u64>)) {
        assert!(self.clients > 0);
        self.clients -= 1;
        if self.clients == 0 { if let Some(v) = self.saved.take() { set(v); keep(None); } }
    }
}

fn get_limit() -> Option<u64> { crate::tuning::sysctl_u64(LIMIT) }
fn set_limit(v: u64) {
    let Ok(mut x) = u32::try_from(v) else { eprintln!("[inky memd] {v} does not fit {LIMIT}; not set"); return };
    let n = std::ffi::CString::new(LIMIT).unwrap();
    let r = unsafe { libc::sysctlbyname(n.as_ptr(), std::ptr::null_mut(), std::ptr::null_mut(), &mut x as *mut u32 as *mut libc::c_void, 4) };
    if r != 0 { eprintln!("[inky memd] setting {LIMIT}={v} failed: {}", std::io::Error::last_os_error()); }
    else { eprintln!("[inky memd] {LIMIT} = {v}"); }
}

/// `inky memd --owner UID [--socket PATH] [--fake]`: the helper. `--fake` keeps the limit in memory instead of the OS
/// setting (for tests without root).
pub fn daemon(args: &[String]) {
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let owner: u32 = arg("--owner").and_then(|v| v.parse().ok()).expect("--owner UID");
    let path = arg("--socket").unwrap_or(SOCKET.into());
    let fake = args.iter().any(|a| a == "--fake");
    let (chip, mem) = crate::tuning::this_machine();
    let tuning = crate::tuning::select(&chip, mem);
    let target = tuning.wired_limit_mb;
    let fake_limit = Arc::new(Mutex::new(0u64));
    let (fl1, fl2) = (fake_limit.clone(), fake_limit.clone());
    let get: Arc<dyn Fn() -> Option<u64> + Send + Sync> = if fake { Arc::new(move || Some(*fl1.lock().unwrap())) } else { Arc::new(get_limit) };
    let set: Arc<dyn Fn(u64) + Send + Sync> = if fake { Arc::new(move |v| { *fl2.lock().unwrap() = v; eprintln!("[inky memd] (fake) limit = {v}"); }) } else { Arc::new(set_limit) };
    // the saved limit on disk (beside the socket with --fake, so tests need no root)
    let saved_path = if fake { format!("{path}.saved") } else { SAVED.to_string() };
    let sp = saved_path.clone();
    let keep: Arc<dyn Fn(Option<u64>) + Send + Sync> = Arc::new(move |v| match v {
        Some(v) => { let tmp = format!("{sp}.tmp"); if std::fs::write(&tmp, v.to_string()).and_then(|_| std::fs::rename(&tmp, &sp)).is_err() { eprintln!("[inky memd] could not record the saved limit in {sp}"); } }
        None => { let _ = std::fs::remove_file(&sp); }
    });
    // a helper that died while the limit was raised left the limit it found on disk: put it back first
    if let Some(v) = std::fs::read_to_string(&saved_path).ok().and_then(|t| t.trim().parse::<u64>().ok()) {
        eprintln!("[inky memd] the previous helper stopped while the limit was raised: restoring {v}");
        set(v); keep(None);
    }
    // the socket lives in a directory only the owner (and root) can open, created before the socket exists
    let dir = std::path::Path::new(&path).parent().expect("a socket path with a directory").to_path_buf();
    let _ = std::fs::create_dir_all(&dir);
    let cd = std::ffi::CString::new(dir.to_string_lossy().as_bytes()).unwrap();
    unsafe { libc::chown(cd.as_ptr(), owner, libc::gid_t::MAX); libc::chmod(cd.as_ptr(), 0o700); }
    let _ = std::fs::remove_file(&path);
    let old_mask = unsafe { libc::umask(0o077) };
    let l = UnixListener::bind(&path).expect("bind the memd socket");
    unsafe { libc::umask(old_mask); }
    let c = std::ffi::CString::new(path.clone()).unwrap();
    unsafe { libc::chown(c.as_ptr(), owner, libc::gid_t::MAX); libc::chmod(c.as_ptr(), 0o600); }
    eprintln!("[inky memd] tuning {} ({chip}, {} GB): inky mode raises {LIMIT} to {target}; listening on {path}", tuning.name, tuning.memory_gb);
    let mode = Arc::new(Mutex::new(Mode::default()));
    // SIGTERM (launchctl bootout) and SIGINT: give the memory back, then exit
    {
        let (mode, set, keep) = (mode.clone(), set.clone(), keep.clone());
        std::thread::spawn(move || unsafe {
            let mut sigs: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&mut sigs); libc::sigaddset(&mut sigs, libc::SIGTERM); libc::sigaddset(&mut sigs, libc::SIGINT);
            let mut sig = 0;
            libc::sigwait(&sigs, &mut sig);
            let m = mode.lock().unwrap();
            if let Some(v) = m.saved { set(v); keep(None); }
            std::process::exit(0);
        });
        unsafe {
            let mut sigs: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&mut sigs); libc::sigaddset(&mut sigs, libc::SIGTERM); libc::sigaddset(&mut sigs, libc::SIGINT);
            libc::pthread_sigmask(libc::SIG_BLOCK, &sigs, std::ptr::null_mut());
        }
    }
    for s in l.incoming() {
        let Ok(s) = s else { continue };
        let (mode, get, set, keep) = (mode.clone(), get.clone(), set.clone(), keep.clone());
        std::thread::spawn(move || {
            let mut r = BufReader::new(&s);
            let mut line = String::new();
            if r.read_line(&mut line).unwrap_or(0) == 0 || !line.starts_with("enter") { return; }
            if let Err(e) = mode.lock().unwrap().connect(target, || get(), |v| set(v), |v| keep(v)) {
                eprintln!("[inky memd] refused a client: {e}");
                let _ = (&s).write_all(format!("err {e}\n").as_bytes());
                return;
            }
            let _ = (&s).write_all(format!("ok {target}\n").as_bytes());
            // hold until the client's end closes, however it closes
            let mut sink = String::new();
            while r.read_line(&mut sink).unwrap_or(0) > 0 { sink.clear(); }
            mode.lock().unwrap().close(|v| set(v), |v| keep(v));
        });
    }
}

/// Enter inky mode for the life of this process, if the helper is installed. The connection is kept open until the
/// process ends; nothing else is needed to leave. Returns the limit the helper set.
pub fn enter(path: &str) -> Option<u64> {
    let s = UnixStream::connect(path).ok()?;
    (&s).write_all(format!("enter {}\n", std::process::id()).as_bytes()).ok()?;
    let mut line = String::new();
    BufReader::new(&s).read_line(&mut line).ok()?;
    let v = line.strip_prefix("ok ")?.trim().parse().ok()?;
    Box::leak(Box::new(s));     // held open until the process exits
    Some(v)
}

/// `sudo inky tune-memory [--undo]`: install (or remove) the helper as a root LaunchDaemon. The helper runs a root-owned
/// copy of this binary, so the program running as root cannot be replaced by the user.
pub fn install(args: &[String]) {
    if unsafe { libc::geteuid() } != 0 { eprintln!("run it with sudo: sudo {} tune-memory{}", std::env::args().next().unwrap_or("inky".into()), if args.iter().any(|a| a == "--undo") { " --undo" } else { "" }); std::process::exit(1); }
    let sh = |cmd: &[&str]| { let _ = std::process::Command::new(cmd[0]).args(&cmd[1..]).status(); };
    if args.iter().any(|a| a == "--undo") {
        sh(&["launchctl", "bootout", &format!("system/{LABEL}")]);       // SIGTERM: the helper restores what it raised
        if let Some(v) = std::fs::read_to_string(SAVED).ok().and_then(|t| t.trim().parse::<u64>().ok()) { set_limit(v); }
        for f in [PLIST, HELPER, OLD_HELPER, SOCKET, SAVED] { let _ = std::fs::remove_file(f); }
        println!("inky mode removed. {LIMIT} is now {}", get_limit().map_or("unreadable".into(), |v| v.to_string()));
        return;
    }
    let owner: u32 = std::env::var("SUDO_UID").ok().and_then(|v| v.parse().ok()).expect("run it with sudo from the owner's account");
    sh(&["launchctl", "bootout", &format!("system/{LABEL}")]);
    // the helper runs as root: its directory must be root's alone, and the copy must be root's before it is in place
    use std::os::unix::fs::MetadataExt;
    let md = std::fs::symlink_metadata(HELPER_DIR).expect("read /Library/PrivilegedHelperTools");
    if !md.is_dir() || md.uid() != 0 || md.mode() & 0o022 != 0 {
        eprintln!("refused: {HELPER_DIR} must be a root-owned directory no one else can write"); std::process::exit(1);
    }
    let tmp = format!("{HELPER}.new");
    let _ = std::fs::remove_file(&tmp);
    std::fs::copy(std::env::current_exe().unwrap(), &tmp).expect("copy the helper");
    let c = std::ffi::CString::new(tmp.clone()).unwrap();
    unsafe { libc::chown(c.as_ptr(), 0, 0); libc::chmod(c.as_ptr(), 0o755); }
    std::fs::rename(&tmp, HELPER).expect("put the helper in place");
    let _ = std::fs::remove_file(OLD_HELPER);
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
    std::fs::write(PLIST, plist).expect("write the LaunchDaemon");
    sh(&["launchctl", "bootstrap", "system", PLIST]);
    let (chip, mem) = crate::tuning::this_machine();
    let t = crate::tuning::select(&chip, mem);
    println!("inky mode installed (tuning {}): while inky runs, {LIMIT} = {}; when it stops, the limit goes back. Remove: sudo inky tune-memory --undo", t.name, t.wired_limit_mb);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    #[test] fn raised_while_connected_and_restored_after_the_last() {
        let (lim, disk) = (Cell::new(50000u64), Cell::new(None));
        let mut m = Mode::default();
        m.connect(118000, || Some(lim.get()), |v| lim.set(v), |v| disk.set(v)).unwrap();
        assert_eq!(disk.get(), Some(50000), "the limit found is on disk before the raise");
        m.connect(118000, || Some(lim.get()), |v| lim.set(v), |v| disk.set(v)).unwrap();
        assert_eq!(lim.get(), 118000);
        m.close(|v| lim.set(v), |v| disk.set(v));
        assert_eq!(lim.get(), 118000, "still one client: still raised");
        m.close(|v| lim.set(v), |v| disk.set(v));
        assert_eq!((lim.get(), m.saved, disk.get()), (50000, None, None), "the limit found, not the default; nothing left on disk");
    }
    #[test] fn a_second_session_saves_afresh() {
        let (lim, disk) = (Cell::new(0u64), Cell::new(None));
        let mut m = Mode::default();
        m.connect(118000, || Some(lim.get()), |v| lim.set(v), |v| disk.set(v)).unwrap(); m.close(|v| lim.set(v), |v| disk.set(v));
        lim.set(70000);     // the owner changes it between sessions
        m.connect(118000, || Some(lim.get()), |v| lim.set(v), |v| disk.set(v)).unwrap(); m.close(|v| lim.set(v), |v| disk.set(v));
        assert_eq!(lim.get(), 70000);
    }
    #[test] fn an_unreadable_limit_is_never_raised() {
        let (lim, disk) = (Cell::new(50000u64), Cell::new(None));
        let mut m = Mode::default();
        assert!(m.connect(118000, || None, |v| lim.set(v), |v| disk.set(v)).is_err());
        assert_eq!((lim.get(), m.clients, disk.get()), (50000, 0, None));
    }
}
