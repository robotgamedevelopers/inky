use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::Command;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "inky-fleet-serve-test-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn fleet_serve_records_only_when_opted_in_and_fails_on_chmod_error() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src_script = repo_root.join("scripts/inky-fleet-serve.sh");

    // 1. Verify .gitignore ignores /logs/cards/*.jsonl.
    let git_ignore = Command::new("git")
        .args(["check-ignore", "-q", "logs/cards/cards-20261125.jsonl"])
        .current_dir(&repo_root)
        .status()
        .expect("git check-ignore");
    assert!(git_ignore.success(), "/logs/ must be ignored by .gitignore");

    // Copy wrapper into an isolated fake repo root so tests never touch repo_root/logs.
    let sandbox = TempDir::new("optin");
    let scripts_dir = sandbox.0.join("scripts");
    fs::create_dir_all(&scripts_dir).unwrap();
    let script = scripts_dir.join("inky-fleet-serve.sh");
    fs::copy(&src_script, &script).unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();

    let fake_inky = sandbox.0.join("fake-inky");
    fs::write(&fake_inky, "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n").unwrap();
    fs::set_permissions(&fake_inky, fs::Permissions::from_mode(0o755)).unwrap();

    // 2. Default run: no --record-dir argument and no logs/ directory created.
    let out_default = Command::new("bash")
        .arg(&script)
        .arg("--kv8")
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("HOME", &sandbox.0)
        .env("INKY_BIN", &fake_inky)
        .env("INKY_MODEL", "/fake/model.gguf")
        .env("INKY_BIND", "127.0.0.1:18096")
        .env("INKY_SERVE_KEY", "unit-test-secret")
        .output()
        .expect("run default fleet-serve");
    assert!(out_default.status.success(), "stderr: {}", String::from_utf8_lossy(&out_default.stderr));
    let args_default: Vec<String> = String::from_utf8_lossy(&out_default.stdout)
        .lines()
        .map(str::to_owned)
        .collect();
    assert!(!args_default.iter().any(|a| a == "--record-dir"), "default must not enable --record-dir: {args_default:?}");
    let flag_val = |flag: &str| -> Option<&str> {
        args_default
            .windows(2)
            .find(|w| w[0] == flag)
            .map(|w| w[1].as_str())
    };
    assert_eq!(flag_val("--lanes"), Some("64"), "default lanes must match M5 Max sweep knee (64)");
    assert_eq!(flag_val("--queue"), Some("128"), "default queue must match docs/LANE-SWEEP.md (128)");
    assert_eq!(flag_val("--effort"), Some("low"), "default effort must match docs/LANE-SWEEP.md (low)");
    assert_eq!(args_default.last().map(String::as_str), Some("--kv8"));
    assert!(!sandbox.0.join("logs").exists(), "default must not create <repo>/logs");

    // 3. Opt-in via INKY_RECORD_CARDS=1: creates <repo>/logs/cards with mode 0700 and passes --record-dir.
    let out_optin = Command::new("bash")
        .arg(&script)
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("HOME", &sandbox.0)
        .env("INKY_BIN", &fake_inky)
        .env("INKY_MODEL", "/fake/model.gguf")
        .env("INKY_BIND", "127.0.0.1:18096")
        .env("INKY_SERVE_KEY", "unit-test-secret")
        .env("INKY_RECORD_CARDS", "1")
        .output()
        .expect("run INKY_RECORD_CARDS=1");
    assert!(out_optin.status.success(), "stderr: {}", String::from_utf8_lossy(&out_optin.stderr));
    let default_rec = sandbox.0.join("logs/cards");
    assert!(default_rec.is_dir());
    assert_eq!(fs::metadata(&default_rec).unwrap().permissions().mode() & 0o777, 0o700);
    let out_optin_str = String::from_utf8_lossy(&out_optin.stdout);
    assert!(out_optin_str.contains("--record-dir\n"), "expected --record-dir in {out_optin_str}");
    assert!(out_optin_str.contains(default_rec.to_str().unwrap()));

    // 4. Opt-in via explicit INKY_RECORD_DIR: creates custom dir with mode 0700.
    let custom_rec = sandbox.0.join("custom-records");
    let out_custom = Command::new("bash")
        .arg(&script)
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("HOME", &sandbox.0)
        .env("INKY_BIN", &fake_inky)
        .env("INKY_MODEL", "/fake/model.gguf")
        .env("INKY_BIND", "127.0.0.1:18096")
        .env("INKY_SERVE_KEY", "unit-test-secret")
        .env("INKY_RECORD_DIR", &custom_rec)
        .output()
        .expect("run INKY_RECORD_DIR");
    assert!(out_custom.status.success());
    assert_eq!(fs::metadata(&custom_rec).unwrap().permissions().mode() & 0o777, 0o700);

    // 5. Invalid INKY_RECORD_CARDS value fails with usage exit 2.
    let out_bad_flag = Command::new("bash")
        .arg(&script)
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("HOME", &sandbox.0)
        .env("INKY_BIN", &fake_inky)
        .env("INKY_MODEL", "/fake/model.gguf")
        .env("INKY_BIND", "127.0.0.1:18096")
        .env("INKY_SERVE_KEY", "unit-test-secret")
        .env("INKY_RECORD_CARDS", "yes")
        .output()
        .expect("run bad INKY_RECORD_CARDS");
    assert_eq!(out_bad_flag.status.code(), Some(2));

    // 6. Failing chmod is NOT masked: wrapper exits non-zero and never execs inky.
    let fake_bin_dir = sandbox.0.join("fake-bin");
    fs::create_dir_all(&fake_bin_dir).unwrap();
    let failing_chmod = fake_bin_dir.join("chmod");
    fs::write(&failing_chmod, "#!/usr/bin/env bash\necho 'injected chmod failure' >&2\nexit 1\n").unwrap();
    fs::set_permissions(&failing_chmod, fs::Permissions::from_mode(0o755)).unwrap();
    let path_with_bad_chmod = format!("{}:{}", fake_bin_dir.display(), std::env::var("PATH").unwrap_or_default());

    let out_chmod_fail = Command::new("bash")
        .arg(&script)
        .env_clear()
        .env("PATH", path_with_bad_chmod)
        .env("HOME", &sandbox.0)
        .env("INKY_BIN", &fake_inky)
        .env("INKY_MODEL", "/fake/model.gguf")
        .env("INKY_BIND", "127.0.0.1:18096")
        .env("INKY_SERVE_KEY", "unit-test-secret")
        .env("INKY_RECORD_DIR", sandbox.0.join("chmod-fail-dir"))
        .output()
        .expect("run failing chmod");
    assert!(!out_chmod_fail.status.success(), "failing chmod must abort startup");
    assert!(out_chmod_fail.stdout.is_empty(), "inky must not be exec'd when chmod fails");
}
