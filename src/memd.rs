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

/// The socket directory must be a real directory, not a symlink a caller planted for the root helper to
/// chown or chmod through.
fn check_socket_dir(dir: &std::path::Path) -> Result<(), String> {
    match std::fs::symlink_metadata(dir) {
        Ok(md) if !md.file_type().is_symlink() && md.is_dir() => Ok(()),
        Ok(_) => Err(format!("the socket directory {} is not a real directory", dir.display())),
        Err(e) => Err(format!("could not inspect the socket directory {}: {e}", dir.display())),
    }
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
    let set: Arc<dyn Fn(u64) -> Result<(), String> + Send + Sync> = if fake { Arc::new(move |v| { *fl2.lock().unwrap() = v; eprintln!("[inky memd] (fake) limit = {v}"); Ok(()) }) } else { Arc::new(set_limit) };
    // the saved limit on disk (beside the socket with --fake, so tests need no root)
    let saved_path = if fake { format!("{path}.saved") } else { SAVED.to_string() };
    let sp = saved_path.clone();
    let keep: Arc<dyn Fn(Option<u64>) -> Result<(), String> + Send + Sync> = Arc::new(move |v| match v {
        Some(v) => { let tmp = format!("{sp}.tmp"); std::fs::write(&tmp, v.to_string()).and_then(|_| std::fs::rename(&tmp, &sp)).map_err(|e| format!("could not record the saved limit in {sp}: {e}")) }
        None => match std::fs::remove_file(&sp) { Ok(()) => Ok(()), Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()), Err(e) => Err(format!("could not forget the saved limit in {sp}: {e}")) },
    });
    // a helper that died while the limit was raised left the limit it found on disk: put it back first (`set`
    // confirms by reading back, so its Ok is the confirmation). If the restore fails, the file is kept and Mode is
    // seeded from it below, so the first connect restores that limit instead of reading the still-raised limit
    // as the found one and overwriting the file with it.
    let mut seed = None;
    if let Some(v) = std::fs::read_to_string(&saved_path).ok().and_then(|t| t.trim().parse::<u64>().ok()) {
        eprintln!("[inky memd] the previous helper stopped while the limit was raised: restoring {v}");
        match set(v) {
            Ok(()) => { if let Err(e) = keep(None) { eprintln!("[inky memd] {e}"); } }
            Err(e) => { eprintln!("[inky memd] restoring {v} failed: {e}; keeping the saved limit in {saved_path}"); seed = Some(v); }
        }
    }
    // the socket lives in a directory only the owner (and root) can open, created before the socket exists. Every step
    // is checked and a failure refuses the bind; the directory is verified not to be a symlink before it is chowned or
    // chmodded, and `lchown` does not follow one either.
    let dir = std::path::Path::new(&path).parent().expect("a socket path with a directory").to_path_buf();
    if let Err(e) = std::fs::create_dir_all(&dir) { eprintln!("[inky memd] could not create the socket directory {}: {e}; not binding", dir.display()); std::process::exit(1); }
    if let Err(e) = check_socket_dir(&dir) { eprintln!("[inky memd] {e}; not binding"); std::process::exit(1); }
    let cd = std::ffi::CString::new(dir.to_string_lossy().as_bytes()).unwrap();
    if unsafe { libc::lchown(cd.as_ptr(), owner, libc::gid_t::MAX) } != 0 { eprintln!("[inky memd] could not chown the socket directory {}: {}; not binding", dir.display(), std::io::Error::last_os_error()); std::process::exit(1); }
    if unsafe { libc::chmod(cd.as_ptr(), 0o700) } != 0 { eprintln!("[inky memd] could not chmod the socket directory {}: {}; not binding", dir.display(), std::io::Error::last_os_error()); std::process::exit(1); }
    let _ = std::fs::remove_file(&path);
    let old_mask = unsafe { libc::umask(0o077) };
    let l = UnixListener::bind(&path).expect("bind the memd socket");
    unsafe { libc::umask(old_mask); }
    let c = std::ffi::CString::new(path.clone()).unwrap();
    unsafe { libc::chown(c.as_ptr(), owner, libc::gid_t::MAX); libc::chmod(c.as_ptr(), 0o600); }
    eprintln!("[inky memd] tuning {} ({chip}, {} GB): inky mode raises {LIMIT} to {target}; listening on {path}", tuning.name, tuning.memory_gb);
    let mode = Arc::new(Mutex::new(Mode { saved: seed, clients: 0 }));
    // SIGTERM (launchctl bootout) and SIGINT: give the memory back, then exit (`set` confirms by reading back,
    // so its Ok is the confirmation; Mode::close keeps its own read-back because its `set` is injected)
    {
        let (mode, set, keep) = (mode.clone(), set.clone(), keep.clone());
        std::thread::spawn(move || unsafe {
            let mut sigs: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&mut sigs); libc::sigaddset(&mut sigs, libc::SIGTERM); libc::sigaddset(&mut sigs, libc::SIGINT);
            let mut sig = 0;
            libc::sigwait(&sigs, &mut sig);
            let m = mode.lock().unwrap();
            if let Some(v) = m.saved {
                match set(v) {
                    Ok(()) => { if let Err(e) = keep(None) { eprintln!("[inky memd] {e}"); } }
                    Err(e) => eprintln!("[inky memd] restoring {v} failed: {e}; keeping the saved limit"),
                }
            }
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
            mode.lock().unwrap().close(|| get(), |v| set(v), |v| keep(v));
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
        // the saved file is kept only with a reader for it: if the restore here fails, the helper and its plist
        // stay installed, so the helper's next start reads the file and restores from it (`set_limit` confirms
        // by reading back, so its Ok is the confirmation and its Err covers the unconfirmed case)
        let mut keep_saved = false;
        if let Some(v) = std::fs::read_to_string(SAVED).ok().and_then(|t| t.trim().parse::<u64>().ok()) {
            match set_limit(v) {
                Ok(()) => {}
                Err(e) => { eprintln!("restoring {v} failed: {e}; keeping {SAVED} and the installed helper so its next start restores from the file"); keep_saved = true; }
            }
        }
        if keep_saved {
            for f in [OLD_HELPER, SOCKET] { let _ = std::fs::remove_file(f); }
            println!("inky mode not removed: {SAVED} and the helper are kept until the limit is restored. {LIMIT} is now {}", get_limit().map_or("unreadable".into(), |v| v.to_string()));
            return;
        }
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
    if unsafe { libc::chown(c.as_ptr(), 0, 0) } != 0 { eprintln!("could not chown the helper copy: {}; not installed", std::io::Error::last_os_error()); let _ = std::fs::remove_file(&tmp); std::process::exit(1); }
    if unsafe { libc::chmod(c.as_ptr(), 0o755) } != 0 { eprintln!("could not chmod the helper copy: {}; not installed", std::io::Error::last_os_error()); let _ = std::fs::remove_file(&tmp); std::process::exit(1); }
    std::fs::rename(&tmp, HELPER).expect("put the helper in place");
    // read the installed file back before claiming success: root-owned, 0755, not a symlink
    match std::fs::symlink_metadata(HELPER) {
        Ok(md) if !md.file_type().is_symlink() && md.is_file() && md.uid() == 0 && md.mode() & 0o777 == 0o755 => {}
        Ok(md) => { eprintln!("refused: the installed helper {HELPER} has owner {} and mode {:o}; not installed", md.uid(), md.mode() & 0o777); std::process::exit(1); }
        Err(e) => { eprintln!("could not read back the installed helper {HELPER}: {e}; not installed"); std::process::exit(1); }
    }
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
    match std::process::Command::new("launchctl").args(["bootstrap", "system", PLIST]).status() {
        Ok(s) if s.success() => {}
        Ok(s) => { eprintln!("launchctl bootstrap failed ({s}); not installed"); std::process::exit(1); }
        Err(e) => { eprintln!("could not run launchctl bootstrap: {e}; not installed"); std::process::exit(1); }
    }
    // the bootstrap may have replaced nothing, but re-check the installed file before printing success
    match std::fs::symlink_metadata(HELPER) {
        Ok(md) if !md.file_type().is_symlink() && md.is_file() && md.uid() == 0 && md.mode() & 0o777 == 0o755 => {}
        Ok(md) => { eprintln!("refused: the installed helper {HELPER} has owner {} and mode {:o}; not installed", md.uid(), md.mode() & 0o777); std::process::exit(1); }
        Err(e) => { eprintln!("could not read back the installed helper {HELPER}: {e}; not installed"); std::process::exit(1); }
    }
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
        let base = std::env::current_dir().unwrap().join(format!("inky-memd-dirtest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        assert!(check_socket_dir(&base).is_ok(), "a real directory is accepted");
        let target = base.join("real");
        std::fs::create_dir(&target).unwrap();
        let link = base.join("link");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert!(check_socket_dir(&link).is_err(), "a symlink to a directory is refused");
        let file = base.join("file");
        std::fs::write(&file, "x").unwrap();
        assert!(check_socket_dir(&file).is_err(), "a plain file is refused");
        assert!(check_socket_dir(&base.join("missing")).is_err(), "a missing directory is refused");
        let _ = std::fs::remove_dir_all(&base);
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
}
