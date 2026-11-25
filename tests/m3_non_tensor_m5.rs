//! Hardware qualification test for `t-later-m3`:
//! Exercises the non-MPP Metal 3 simdgroup prefill and decode kernels (`--no-mpp` and
//! `INKY_DISABLE_MPP=1`, matching the `tunings/m3-ultra-*.tuning` execution profile) against the
//! default MPP tensor path and golden references on Apple Silicon Metal hardware:
//! 1. Reference logit & argmax parity (`inky check` with `INK_PF=16 INK_DEC=1` under `INKY_DISABLE_MPP=1` and `--no-mpp`).
//! 2. Batched decode (`inky dbatch --no-mpp`) and grouped shared-KV (`inky sharedkv --no-mpp`) parity.
//! 3. Live HTTP `inky serve --no-mpp` greedy generation + prefix-cache continuation parity vs MPP `inky serve`.
//! 4. Micro-benchmark throughput comparison (`inky bench --no-mpp` vs MPP `inky bench`).

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn home_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").expect("HOME environment variable must be set"))
}

fn resolve_model_path() -> PathBuf {
    std::env::var_os("INKY_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            home_dir()
                .join(".lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf")
        })
}

fn resolve_ref_dir() -> PathBuf {
    std::env::var_os("INKY_REF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join("inky-ci/ref"))
}

fn resolve_bin() -> PathBuf {
    std::env::var_os("INKY_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_BIN_EXE_inky")))
}

fn run_cli(bin: &Path, args: &[&str], extra_env: &[(&str, &str)]) -> (String, String) {
    let mut cmd = Command::new(bin);
    cmd.args(args).stdin(Stdio::null());
    for &(k, v) in extra_env {
        cmd.env(k, v);
    }
    let out = cmd.output().unwrap_or_else(|e| panic!("spawn {:?}: {e}", args));
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "command {:?} failed ({:?}):\nstdout:\n{}\nstderr:\n{}",
        args,
        out.status,
        stdout,
        stderr
    );
    (stdout, stderr)
}

struct RunningServer {
    child: Child,
    addr: SocketAddr,
    stderr_lines: Arc<Mutex<Vec<String>>>,
}

impl RunningServer {
    fn spawn(bin: &Path, model: &Path, no_mpp: bool) -> Self {
        let mut cmd = Command::new(bin);
        cmd.arg("serve")
            .arg("--bind")
            .arg("127.0.0.1:0")
            .arg("--ctx")
            .arg("2048")
            .arg("-n")
            .arg("32")
            .arg("--lanes")
            .arg("2")
            .arg("--effort")
            .arg("none")
            .arg("-m")
            .arg(model)
            .env("INKY_TEST_POOL_SLOTS", "24,30")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        if no_mpp {
            cmd.arg("--no-mpp");
        }
        let mut child = cmd.spawn().expect("spawn inky serve");
        let stderr = child.stderr.take().expect("captured stderr");
        let lines: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let lines_bg = lines.clone();
        let (addr_tx, addr_rx) = std::sync::mpsc::channel::<SocketAddr>();
        std::thread::spawn(move || {
            let mut sent = false;
            let reader = BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                if !sent {
                    if let Some(rest) = line.split("http://").nth(1) {
                        if let Some(host_port) = rest.split("/v1:").next() {
                            if let Ok(a) = host_port.trim().parse::<SocketAddr>() {
                                let _ = addr_tx.send(a);
                                sent = true;
                            }
                        }
                    }
                }
                lines_bg.lock().unwrap().push(line);
            }
        });

        let addr = addr_rx.recv_timeout(Duration::from_secs(90)).unwrap_or_else(|_| {
            let _ = child.kill();
            let _ = child.wait();
            let dump = lines.lock().unwrap().join("\n");
            panic!("inky serve did not bind within 90s; stderr:\n{dump}");
        });

        let srv = Self {
            child,
            addr,
            stderr_lines: lines,
        };
        srv.wait_healthy(Duration::from_secs(30));
        srv
    }

    fn wait_healthy(&self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Ok((200, v)) = self.http_request("GET", "/health", None) {
                if v["ok"] == json!(true) {
                    return;
                }
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("server at {} failed /health readiness", self.addr);
    }

    fn post_json(&self, path: &str, body: &Value) -> Result<(u16, Value), String> {
        self.http_request("POST", path, Some(&body.to_string()))
    }

    fn http_request(&self, method: &str, path: &str, body: Option<&str>) -> Result<(u16, Value), String> {
        let mut stream = TcpStream::connect_timeout(&self.addr, Duration::from_secs(10)).map_err(|e| e.to_string())?;
        stream.set_read_timeout(Some(Duration::from_secs(120))).map_err(|e| e.to_string())?;
        stream.set_write_timeout(Some(Duration::from_secs(30))).map_err(|e| e.to_string())?;
        let payload = body.unwrap_or("");
        write!(
            stream,
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
            self.addr,
            payload.len()
        )
        .map_err(|e| e.to_string())?;
        stream.flush().map_err(|e| e.to_string())?;

        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).map_err(|e| e.to_string())?;
        let text = String::from_utf8_lossy(&raw);
        let (head, resp_body) = text
            .split_once("\r\n\r\n")
            .ok_or_else(|| format!("malformed HTTP response: {text}"))?;
        let status_line = head.lines().next().unwrap_or("");
        let code: u16 = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| format!("bad status line: {status_line}"))?;
        let parsed: Value = serde_json::from_str(resp_body).map_err(|e| format!("JSON parse error ({e}): {resp_body}"))?;
        Ok((code, parsed))
    }

    fn stop(mut self) -> String {
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.stderr_lines.lock().unwrap().join("\n")
    }
}

impl Drop for RunningServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn greedy_chat(srv: &RunningServer, messages: Value, max_tokens: u64) -> (String, String, u64, u64) {
    let req = json!({
        "model": "inkling-2-mini",
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "seed": 42,
        "reasoning_effort": "none",
        "thinking_budget": 2,
        "max_tokens": max_tokens,
        "stream": false
    });
    let (code, body) = srv.post_json("/v1/chat/completions", &req).expect("POST /v1/chat/completions");
    assert_eq!(code, 200, "unexpected HTTP {code}: {body}");
    let choice = &body["choices"][0];
    let msg = &choice["message"];
    let content = msg["content"].as_str().unwrap_or("").to_string();
    let reasoning = msg["reasoning_content"].as_str().unwrap_or("").to_string();
    let cached = body["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0);
    let completion = body["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    (content, reasoning, cached, completion)
}

#[test]
#[ignore = "requires Apple Metal GPU and local Inkling-2-Mini GGUF"]
fn m5_non_tensor_no_mpp_matches_mpp_logits_batches_and_serve() {
    let bin = resolve_bin();
    let model = resolve_model_path();
    let ref_dir = resolve_ref_dir();
    assert!(bin.exists(), "inky binary missing at {}", bin.display());
    assert!(model.exists(), "model missing at {}", model.display());
    let short_ref = ref_dir.join("short");
    let long_tokens = ref_dir.join("long.tokens");
    assert!(ref_dir.join("short.tokens").exists(), "reference short.tokens missing");
    assert!(long_tokens.exists(), "reference long.tokens missing");

    let model_s = model.to_str().unwrap();
    let short_s = short_ref.to_str().unwrap();
    let long_s = long_tokens.to_str().unwrap();

    // 1. Reference logit & argmax verification for both MPP and non-MPP (`INKY_DISABLE_MPP=1` & `--no-mpp`)
    //    across prefill (`INK_PF=16`) + batched decode (`INK_DEC=1`) paths.
    let (chk_mpp, _) = run_cli(
        &bin,
        &["check", "-m", model_s, "--ref", short_s, "-q"],
        &[("INK_PF", "16"), ("INK_DEC", "1")],
    );
    let (chk_nompp_env, _) = run_cli(
        &bin,
        &["check", "-m", model_s, "--ref", short_s, "-q"],
        &[("INK_PF", "16"), ("INK_DEC", "1"), ("INKY_DISABLE_MPP", "1")],
    );
    let (chk_nompp_flag, _) = run_cli(
        &bin,
        &["check", "--no-mpp", "-m", model_s, "--ref", short_s, "-q"],
        &[("INK_PF", "16"), ("INK_DEC", "1")],
    );
    println!("[check MPP]        {}", chk_mpp.trim());
    println!("[check INKY_DISABLE_MPP=1] {}", chk_nompp_env.trim());
    println!("[check --no-mpp]   {}", chk_nompp_flag.trim());
    assert_eq!(chk_nompp_env.trim(), chk_nompp_flag.trim(), "--no-mpp and INKY_DISABLE_MPP=1 must match");

    // 2. Non-MPP batched decode invariance (`dbatch`) and grouped shared-KV invariance (`sharedkv`).
    let (dbatch_out, _) = run_cli(
        &bin,
        &["dbatch", "--no-mpp", "-m", model_s, "--ref", long_s],
        &[("INK_PF", "128"), ("INK_DECODE", "8"), ("INK_LANES", "1,4")],
    );
    println!("[dbatch --no-mpp]\n{}", dbatch_out.trim());

    let (sharedkv_out, _) = run_cli(
        &bin,
        &["sharedkv", "--no-mpp", "-m", model_s, "--ref", long_s],
        &[("INK_PREFIX", "256"), ("INK_DECODE", "4"), ("INK_LANES", "8")],
    );
    println!("[sharedkv --no-mpp]\n{}", sharedkv_out.trim());

    // 3. Throughput receipt: default MPP vs `--no-mpp` on the exact same prompt & hardware.
    let (bench_mpp, _) = run_cli(
        &bin,
        &["bench", "-m", model_s, "-c", "2048", "-n", "32", "--greedy", "-p", "Summarize KV cache paging cleanly."],
        &[],
    );
    let (bench_nompp, _) = run_cli(
        &bin,
        &["bench", "--no-mpp", "-m", model_s, "-c", "2048", "-n", "32", "--greedy", "-p", "Summarize KV cache paging cleanly."],
        &[],
    );
    println!("[bench MPP]\n{}", bench_mpp.trim());
    println!("[bench --no-mpp]\n{}", bench_nompp.trim());

    // 4. Live HTTP `inky serve --no-mpp` greedy generation + prefix-cache continuation parity vs MPP.
    // Use a >64-token system prompt so Turn 1 seals at least one 64-token block checkpoint and Turn 2 hits the cache.
    let shared_sys = concat!(
        "You are a concise, deterministic systems verification assistant. ",
        "Always answer arithmetic questions directly in plain text with only the final integer result, ",
        "with zero extra commentary, punctuation, or Markdown formatting. ",
        "Follow instructions strictly across every turn of the conversation without exception."
    );
    let turn1 = json!([
        {"role": "system", "content": shared_sys},
        {"role": "user", "content": "What is 17 * 23? Reply with the number."}
    ]);
    let turn2 = json!([
        {"role": "system", "content": shared_sys},
        {"role": "user", "content": "What is 17 * 23? Reply with the number."},
        {"role": "assistant", "content": "391"},
        {"role": "user", "content": "Now add 9 to 391 and reply with just the number."}
    ]);

    let srv_mpp = RunningServer::spawn(&bin, &model, false);
    let mpp_t1 = greedy_chat(&srv_mpp, turn1.clone(), 32);
    let mpp_t2 = greedy_chat(&srv_mpp, turn2.clone(), 32);
    srv_mpp.stop();

    let srv_nompp = RunningServer::spawn(&bin, &model, true);
    let nompp_t1 = greedy_chat(&srv_nompp, turn1, 32);
    let nompp_t2 = greedy_chat(&srv_nompp, turn2, 32);
    let nompp_stderr = srv_nompp.stop();

    assert!(
        nompp_stderr.contains("non-MPP Metal 3 simdgroup path active"),
        "expected non-MPP banner in serve stderr:\n{nompp_stderr}"
    );
    assert!(!nompp_t1.0.is_empty(), "expected non-empty turn1 content, got reasoning={:?}", nompp_t1.1);
    assert!(!nompp_t2.0.is_empty(), "expected non-empty turn2 content, got reasoning={:?}", nompp_t2.1);
    assert_eq!(nompp_t1.0, mpp_t1.0, "turn1 content mismatch: non-MPP vs MPP");
    assert_eq!(nompp_t1.1, mpp_t1.1, "turn1 reasoning mismatch: non-MPP vs MPP");
    assert_eq!(nompp_t2.0, mpp_t2.0, "turn2 content mismatch: non-MPP vs MPP");
    assert_eq!(nompp_t2.1, mpp_t2.1, "turn2 reasoning mismatch: non-MPP vs MPP");
    assert!(nompp_t2.2 >= 64, "expected prefix-cache hit >= 64 on turn 2 in --no-mpp serve, got {}", nompp_t2.2);
    println!(
        "[serve parity] turn1={:?} ({} toks) turn2={:?} (cached={}, {} toks)",
        nompp_t1.0, nompp_t1.3, nompp_t2.0, nompp_t2.2, nompp_t2.3
    );
}
