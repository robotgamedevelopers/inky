use std::ffi::CString;
use std::fs;
use std::os::unix::ffi::OsStringExt;
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        // Keep socket paths well below macOS `SUN_LEN` (~104 bytes) regardless of checkout or TMPDIR length.
        let mut template = CString::new("/tmp/inky-lease-XXXXXX").unwrap().into_bytes_with_nul();
        let ptr = unsafe { libc::mkdtemp(template.as_mut_ptr().cast()) };
        if !ptr.is_null() {
            template.pop();
            return Self(PathBuf::from(std::ffi::OsString::from_vec(template)));
        }
        let fallback = std::env::temp_dir().join(format!("il-{}", std::process::id()));
        let _ = fs::remove_dir_all(&fallback);
        fs::DirBuilder::new().mode(0o700).create(&fallback).unwrap();
        Self(fallback)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Process(Child);

impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn spawn_inky(args: &[&str], socket: &Path, log: &Path) -> Process {
    Process(
        Command::new(env!("CARGO_BIN_EXE_inky"))
            .args(args)
            .env("INKY_MEMD_SOCKET", socket)
            .stdout(Stdio::null())
            .stderr(fs::File::create(log).unwrap())
            .spawn()
            .unwrap(),
    )
}

fn wait_for(log: &Path, condition: impl Fn(&str) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let output = fs::read_to_string(log).unwrap_or_default();
        if condition(&output) {
            return;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {}:\n{output}", log.display());
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn fake_limits(log: &str) -> Vec<u64> {
    log.lines()
        .filter_map(|line| line.strip_prefix("[inky memd] (fake) limit = "))
        .map(|value| value.parse().unwrap())
        .collect()
}

#[test]
fn surviving_mode_client_reacquires_lease_after_fake_helper_restart() {
    let uid = unsafe { libc::geteuid() };
    if uid == 0 {
        eprintln!("fake memd requires an unprivileged test account");
        return;
    }
    let scratch = Scratch::new();
    let dir = &scratch.0;
    let (socket, saved) = (dir.join("memd.sock"), dir.join("memd.sock.saved"));
    let (first_log, second_log, client_log) = (dir.join("first"), dir.join("second"), dir.join("client"));
    let uid = uid.to_string();
    let daemon = ["memd", "--fake", "--owner", &uid, "--socket", socket.to_str().unwrap(), "--fake-target-mb", "8192"];

    let first = spawn_inky(&daemon, &socket, &first_log);
    wait_for(&first_log, |s| s.contains("listening on"));
    let client = spawn_inky(&["mode", "120"], &socket, &client_log);
    wait_for(&client_log, |s| s.contains("[inky] inky mode:") && saved.is_file());
    let first_log = fs::read_to_string(&first_log).unwrap();
    assert_eq!(fake_limits(&first_log), [8192], "{first_log}");

    drop(first); // SIGKILL leaves the recovery record for the replacement daemon.
    assert!(saved.is_file());
    let _second = spawn_inky(&daemon, &socket, &second_log);
    wait_for(&client_log, |s| s.contains("memory-helper connection restored"));
    let replacement = fs::read_to_string(&second_log).unwrap();
    assert_eq!(fake_limits(&replacement), [0, 8192], "{replacement}");
    assert!(saved.is_file(), "replacement must keep the new lease's recovery record");

    drop(client);
    wait_for(&second_log, |_| !saved.exists());
    let final_log = fs::read_to_string(&second_log).unwrap();
    assert_eq!(fake_limits(&final_log), [0, 8192, 0], "{final_log}");
}

#[test]
fn persistent_serve_lease_establishes_after_late_helper_start_and_survives_socket_dir_reinstall() {
    let uid = unsafe { libc::geteuid() };
    if uid == 0 {
        eprintln!("fake memd requires an unprivileged test account");
        return;
    }
    let scratch = Scratch::new();
    let root = &scratch.0;
    let sock_dir = root.join("run");
    let socket = sock_dir.join("memd.sock");
    let saved = sock_dir.join("memd.sock.saved");
    let (first_log, second_log, client_log) = (root.join("first"), root.join("second"), root.join("client"));
    let uid_str = uid.to_string();

    // 1. Start persistent client (`inky serve` lease path) BEFORE `sock_dir` or `memd` exists.
    let client = spawn_inky(&["mode", "--persistent", "120"], &socket, &client_log);
    wait_for(&client_log, |s| s.contains("[inky] not in inky mode:"));

    // 2. Install/start helper #1 (`8192` MB) and verify running client establishes lease automatically.
    fs::DirBuilder::new().mode(0o700).create(&sock_dir).unwrap();
    let daemon1 =
        ["memd", "--fake", "--owner", &uid_str, "--socket", socket.to_str().unwrap(), "--fake-target-mb", "8192"];
    let first = spawn_inky(&daemon1, &socket, &first_log);
    wait_for(&first_log, |s| s.contains("listening on"));
    wait_for(&client_log, |s| {
        s.contains("memory-helper connection established; GPU wired limit 8192 MB") && saved.is_file()
    });
    assert_eq!(fake_limits(&fs::read_to_string(&first_log).unwrap()), [8192]);

    // 3. Simulate `sudo inky tune-memory` reinstall: stop daemon #1, remove and recreate the socket
    // directory (new inode, unlinked socket & saved file), and launch upgraded daemon #2 (`12288` MB).
    drop(first);
    fs::remove_dir_all(&sock_dir).unwrap();
    fs::DirBuilder::new().mode(0o700).create(&sock_dir).unwrap();
    let daemon2 =
        ["memd", "--fake", "--owner", &uid_str, "--socket", socket.to_str().unwrap(), "--fake-target-mb", "12288"];
    let _second = spawn_inky(&daemon2, &socket, &second_log);
    wait_for(&second_log, |s| s.contains("listening on"));
    wait_for(&client_log, |s| {
        s.contains("memory-helper connection restored; GPU wired limit 12288 MB") && saved.is_file()
    });
    assert_eq!(fake_limits(&fs::read_to_string(&second_log).unwrap()), [12288]);

    // 4. Stopping the client restores the limit to 0 and removes the saved record on daemon #2.
    drop(client);
    wait_for(&second_log, |_| !saved.exists());
    assert_eq!(fake_limits(&fs::read_to_string(&second_log).unwrap()), [12288, 0]);
}
