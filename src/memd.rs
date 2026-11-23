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
const HELPER: &str = "/usr/local/libexec/inky-memd";
const PLIST: &str = "/Library/LaunchDaemons/com.robotgamedevelopers.inky.memd.plist";
const LABEL: &str = "com.robotgamedevelopers.inky.memd";

/// The helper's state (MemMode.tla): the limit it found when it raised it, and how many clients are connected.
#[derive(Default)]
pub struct Mode { pub saved: Option<u64>, pub clients: usize }

impl Mode {
    /// A client connected: the first one raises the limit to `target`, remembering the limit found.
    pub fn connect(&mut self, target: u64, get: impl Fn() -> u64, set: impl Fn(u64)) {
        if self.clients == 0 { self.saved = Some(get()); set(target); }
        self.clients += 1;
    }
    /// A client's connection closed: the last one restores the limit found.
    pub fn close(&mut self, set: impl Fn(u64)) {
        assert!(self.clients > 0);
        self.clients -= 1;
        if self.clients == 0 { if let Some(v) = self.saved.take() { set(v); } }
    }
}

fn get_limit() -> u64 { crate::tuning::sysctl_u64(LIMIT).unwrap_or(0) }
fn set_limit(v: u64) {
    let n = std::ffi::CString::new(LIMIT).unwrap();
    let mut x: u32 = v as u32;
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
    let get: Arc<dyn Fn() -> u64 + Send + Sync> = if fake { Arc::new(move || *fl1.lock().unwrap()) } else { Arc::new(get_limit) };
    let set: Arc<dyn Fn(u64) + Send + Sync> = if fake { Arc::new(move |v| { *fl2.lock().unwrap() = v; eprintln!("[inky memd] (fake) limit = {v}"); }) } else { Arc::new(set_limit) };
    if let Some(dir) = std::path::Path::new(&path).parent() { let _ = std::fs::create_dir_all(dir); }
    let _ = std::fs::remove_file(&path);
    let l = UnixListener::bind(&path).expect("bind the memd socket");
    let c = std::ffi::CString::new(path.clone()).unwrap();
    unsafe { libc::chown(c.as_ptr(), owner, libc::gid_t::MAX); libc::chmod(c.as_ptr(), 0o600); }
    eprintln!("[inky memd] tuning {} ({chip}, {} GB): inky mode raises {LIMIT} to {target}; listening on {path}", tuning.name, tuning.memory_gb);
    let mode = Arc::new(Mutex::new(Mode::default()));
    // SIGTERM (launchctl bootout) and SIGINT: give the memory back, then exit
    {
        let (mode, set) = (mode.clone(), set.clone());
        std::thread::spawn(move || unsafe {
            let mut sigs: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&mut sigs); libc::sigaddset(&mut sigs, libc::SIGTERM); libc::sigaddset(&mut sigs, libc::SIGINT);
            let mut sig = 0;
            libc::sigwait(&sigs, &mut sig);
            let m = mode.lock().unwrap();
            if let Some(v) = m.saved { set(v); }
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
        let (mode, get, set) = (mode.clone(), get.clone(), set.clone());
        std::thread::spawn(move || {
            let mut r = BufReader::new(&s);
            let mut line = String::new();
            if r.read_line(&mut line).unwrap_or(0) == 0 || !line.starts_with("enter") { return; }
            mode.lock().unwrap().connect(target, || get(), |v| set(v));
            let _ = (&s).write_all(format!("ok {target}\n").as_bytes());
            // hold until the client's end closes, however it closes
            let mut sink = String::new();
            while r.read_line(&mut sink).unwrap_or(0) > 0 { sink.clear(); }
            mode.lock().unwrap().close(|v| set(v));
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
        sh(&["launchctl", "bootout", &format!("system/{LABEL}")]);
        let _ = std::fs::remove_file(PLIST); let _ = std::fs::remove_file(HELPER); let _ = std::fs::remove_file(SOCKET);
        println!("inky mode removed: the helper is gone and {LIMIT} is back to what it was");
        return;
    }
    let owner: u32 = std::env::var("SUDO_UID").ok().and_then(|v| v.parse().ok()).expect("run it with sudo from the owner's account");
    sh(&["launchctl", "bootout", &format!("system/{LABEL}")]);
    std::fs::create_dir_all("/usr/local/libexec").expect("create /usr/local/libexec");
    std::fs::copy(std::env::current_exe().unwrap(), HELPER).expect("copy the helper");
    let c = std::ffi::CString::new(HELPER).unwrap();
    unsafe { libc::chown(c.as_ptr(), 0, 0); libc::chmod(c.as_ptr(), 0o755); }
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
        let lim = Cell::new(50000u64);
        let mut m = Mode::default();
        m.connect(118000, || lim.get(), |v| lim.set(v));
        m.connect(118000, || lim.get(), |v| lim.set(v));
        assert_eq!(lim.get(), 118000);
        m.close(|v| lim.set(v));
        assert_eq!(lim.get(), 118000, "still one client: still raised");
        m.close(|v| lim.set(v));
        assert_eq!((lim.get(), m.saved), (50000, None), "the limit found, not the default");
    }
    #[test] fn a_second_session_saves_afresh() {
        let lim = Cell::new(0u64);
        let mut m = Mode::default();
        m.connect(118000, || lim.get(), |v| lim.set(v)); m.close(|v| lim.set(v));
        lim.set(70000);     // the owner changes it between sessions
        m.connect(118000, || lim.get(), |v| lim.set(v)); m.close(|v| lim.set(v));
        assert_eq!(lim.get(), 70000);
    }
}
