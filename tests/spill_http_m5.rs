//! Hardware qualification test for `--spill-dir` on Apple M5 (`t-audit-cache-spill-hardware`):
//! Drives live HTTP `inky serve` processes against `Inkling-2-Mini-Q4_K_M.gguf` under tight
//! GPU `BlockPools` capacity (`INKY_TEST_POOL_SLOTS`) so a `>= 1,088`-token prompt checkpoint
//! is genuinely evicted from RAM to SSD (`spill_starts_spilled > 0`), then verifies:
//! 1. Cold SSD restore after RAM eviction for a continuation turn (`spill_starts_restored > 0`,
//!    `cached_tokens >= 1,088`) matches a fresh cold `--spill-dir`-less server 100% token-for-token
//!    and records cold SSD restore latency (`spill_last_restore_ms`).
//! 2. Stopping and restarting a fresh `inky serve` process on the same `--spill-dir` reloads
//!    `spill-index.bin` (`spill_spilled_starts_indexed > 0`), restores from SSD (`spill_starts_restored > 0`),
//!    and matches the cold server 100% token-for-token on BOTH continuation and an early diverging
//!    branch (`branching`).
//! 3. Corrupting a spilled `.blk` file on disk triggers a clean cache miss (`spill_corrupt_fallbacks > 0`,
//!    `spill_corrupt_files_removed > 0`, `cached_tokens == 0`, HTTP 200) and produces 100% identical
//!    tokens to the cold server.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

struct RunningServer {
    child: Child,
    addr: SocketAddr,
    #[allow(dead_code)]
    stderr_lines: Arc<Mutex<Vec<String>>>,
}

impl RunningServer {
    fn spawn(bin: &Path, model: &Path, spill_dir: Option<&Path>, pool_slots: &str) -> Self {
        let mut cmd = Command::new(bin);
        cmd.arg("serve")
            .arg("--bind")
            .arg("127.0.0.1:0")
            .arg("--ctx")
            .arg("2048")
            .arg("-n")
            .arg("64")
            .arg("--lanes")
            .arg("2")
            .arg("--effort")
            .arg("none")
            .arg("-m")
            .arg(model)
            .env("INKY_TEST_POOL_SLOTS", pool_slots)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        if let Some(dir) = spill_dir {
            cmd.arg("--spill-dir").arg(dir);
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

        let addr = addr_rx
            .recv_timeout(Duration::from_secs(90))
            .unwrap_or_else(|_| {
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
            if let Ok((200, v)) = self.get_json("/health") {
                if v["ok"] == json!(true) {
                    return;
                }
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("server at {} failed /health readiness within {timeout:?}", self.addr);
    }

    fn wait_drained(&self, timeout: Duration) -> Value {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Ok((200, v)) = self.get_json("/health") {
                if v["active_lanes"].as_u64() == Some(0) && v["inflight"].as_u64() == Some(0) {
                    return v;
                }
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        panic!("server at {} did not drain within {timeout:?}", self.addr);
    }

    fn get_json(&self, path: &str) -> Result<(u16, Value), String> {
        self.http_request("GET", path, None)
    }

    fn post_json(&self, path: &str, body: &Value) -> Result<(u16, Value), String> {
        self.http_request("POST", path, Some(&body.to_string()))
    }

    fn http_request(&self, method: &str, path: &str, body: Option<&str>) -> Result<(u16, Value), String> {
        let mut stream = TcpStream::connect_timeout(&self.addr, Duration::from_secs(10))
            .map_err(|e| e.to_string())?;
        stream
            .set_read_timeout(Some(Duration::from_secs(120)))
            .map_err(|e| e.to_string())?;
        stream
            .set_write_timeout(Some(Duration::from_secs(30)))
            .map_err(|e| e.to_string())?;
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
        let parsed: Value = serde_json::from_str(resp_body)
            .map_err(|e| format!("JSON parse error ({e}) on body: {resp_body}"))?;
        Ok((code, parsed))
    }

    fn stop(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for RunningServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ChatResult {
    content: String,
    reasoning: String,
    finish_reason: String,
    prompt_tokens: u64,
    cached_tokens: u64,
    completion_tokens: u64,
}

fn send_chat_greedy(srv: &RunningServer, messages: Value, max_tokens: u64) -> ChatResult {
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
    ChatResult {
        content: msg["content"].as_str().unwrap_or("").to_string(),
        reasoning: msg["reasoning_content"].as_str().unwrap_or("").to_string(),
        finish_reason: choice["finish_reason"].as_str().unwrap_or("").to_string(),
        prompt_tokens: body["usage"]["prompt_tokens"].as_u64().unwrap(),
        cached_tokens: body["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap(),
        completion_tokens: body["usage"]["completion_tokens"].as_u64().unwrap(),
    }
}

#[test]
#[ignore = "requires Apple M5 Metal GPU and local Inkling-2-Mini GGUF"]
fn m5_http_spill_eviction_restart_branching_and_corrupt_fallback_match_cold() {
    let bin = std::env::var_os("INKY_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_BIN_EXE_inky")));
    let model = std::env::var_os("INKY_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "/Users/glenn/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf",
            )
        });
    assert!(bin.exists(), "inky binary not found at {}", bin.display());
    assert!(model.exists(), "model not found at {}", model.display());

    let spill_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("m5-http-spill-qual-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&spill_dir);
    std::fs::create_dir_all(&spill_dir).unwrap();

    // Shared system prefix (~240 tokens, > 3 x 64-token blocks) used by both main conversation
    // and early diverging branch (`branching`).
    let shared_system = concat!(
        "You are a deterministic systems engineering assistant verifying paged key-value block ",
        "caching, sliding-window attention boundaries at 1024 tokens, recurrent convolution state ",
        "snapshots, crash-safe SHA-256 SSD spill files, atomic working-set reservation, and ",
        "exact greedy token parity across RAM eviction, process restart, and early prefix branching. ",
        "Always reply directly in plain concise prose with concrete technical facts."
    );
    let long_user_chunk = concat!(
        " Subsystem specification block: each block spans 64 positions; Window blocks hold 40 ",
        "sliding-window passes over the trailing 1024 tokens while Full blocks hold 8 global ",
        "attention passes across the full prompt history; start points retain exact recurrent ",
        "state bytes at periodic and prompt boundaries."
    );
    let long_user_body = format!(
        "{}\n\nQuestion: State the two block kinds and the 64-token block size in one sentence.",
        long_user_chunk.repeat(22)
    );

    let turn1_msgs = json!([
        {"role": "system", "content": shared_system},
        {"role": "user", "content": long_user_body}
    ]);
    let branch_msgs = json!([
        {"role": "system", "content": shared_system},
        {"role": "user", "content": "Branch early after the shared system prompt: name the hash algorithm used to verify spilled block files in one short sentence."}
    ]);
    let filler_a_msgs = json!([
        {"role": "system", "content": "Unrelated workload alpha: astronomical catalog indexing and orbital ephemerides."},
        {"role": "user", "content": " Keplerian elements inclination eccentricity ascending node periapsis anomaly.".repeat(48)}
    ]);
    let filler_b_msgs = json!([
        {"role": "system", "content": "Unrelated workload beta: oceanic thermohaline circulation and abyssal salinity gradients."},
        {"role": "user", "content": " Bathymetry Rossby waves Ekman transport meridional overturning pycnocline.".repeat(48)}
    ]);

    // Pool sizing: 24 Window slots, 30 Full slots -> pooled_context = min(2048, 30*64 + 3) = 1923 tokens.
    // - Turn 1 prompt (~1,180 tokens) saves a PROMPT checkpoint >= 1,088 tokens (17..19 Full + 16 Window).
    // - Two ~650-token disjoint fillers (~11 Full + 11 Window each -> 22 slots) exceed remaining RAM
    //   headroom and deterministically evict Turn 1's >= 1,088-token checkpoint to SSD!
    let pool_slots = "24,30";

    // =========================================================================
    // Phase 1: Fresh cold `--spill-dir`-less reference server
    // =========================================================================
    let cold_srv = RunningServer::spawn(&bin, &model, None, pool_slots);
    let flush_seq = std::sync::atomic::AtomicU64::new(1);
    let flush_ram = |srv: &RunningServer| {
        for pass in 0..3 {
            let id = flush_seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let base = if pass % 2 == 0 { &filler_a_msgs } else { &filler_b_msgs };
            let mut msgs = base.as_array().unwrap().clone();
            msgs[0]["content"] = json!(format!("Unrelated flush workload #{id}-{pass}: catalog indexing."));
            msgs[1]["content"] = json!(format!(
                "Epoch {id} pass {pass}: {}",
                msgs[1]["content"].as_str().unwrap_or("")
            ));
            let _ = send_chat_greedy(srv, Value::Array(msgs), 4);
        }
        let _ = srv.wait_drained(Duration::from_secs(15));
    };

    let cold_t1 = send_chat_greedy(&cold_srv, turn1_msgs.clone(), 48);
    assert!(
        cold_t1.prompt_tokens >= 1_100,
        "Turn 1 prompt_tokens={} must be >= 1,100 so checkpoint >= 1,088",
        cold_t1.prompt_tokens
    );
    assert_eq!(cold_t1.cached_tokens, 0, "Cold Turn 1 must start with cached_tokens == 0");
    assert!(!cold_t1.content.is_empty(), "Cold Turn 1 generated empty content: {cold_t1:?}");

    let turn2_cont_msgs = json!([
        {"role": "system", "content": shared_system},
        {"role": "user", "content": long_user_body},
        {"role": "assistant", "content": cold_t1.content},
        {"role": "user", "content": "Now continue in one concise sentence explaining why expired window blocks before 1024 tokens are pruned."}
    ]);

    flush_ram(&cold_srv);
    let cold_cont = send_chat_greedy(&cold_srv, turn2_cont_msgs.clone(), 48);
    assert_eq!(cold_cont.cached_tokens, 0, "Flushed cold continuation must run from cached_tokens == 0");
    assert!(!cold_cont.content.is_empty(), "Cold continuation generated empty content: {cold_cont:?}");

    flush_ram(&cold_srv);
    let cold_branch = send_chat_greedy(&cold_srv, branch_msgs.clone(), 48);
    assert_eq!(cold_branch.cached_tokens, 0, "Flushed cold branch must run from cached_tokens == 0");
    assert!(!cold_branch.content.is_empty(), "Cold branch generated empty content: {cold_branch:?}");
    cold_srv.stop();

    // =========================================================================
    // Phase 2: `--spill-dir` Server #1 — Eviction to SSD + Continuation Restore
    // =========================================================================
    let spill_srv1 = RunningServer::spawn(&bin, &model, Some(&spill_dir), pool_slots);
    let s1_t1 = send_chat_greedy(&spill_srv1, turn1_msgs.clone(), 48);
    assert_eq!(s1_t1.cached_tokens, 0);
    assert_eq!(s1_t1, cold_t1, "Spill server Turn 1 cold run must match reference cold server");

    // Evict Turn 1's >= 1,088-token checkpoint out of RAM onto SSD.
    flush_ram(&spill_srv1);
    let h_after_evict = spill_srv1.wait_drained(Duration::from_secs(15));
    let spilled_1 = h_after_evict["spill_starts_spilled"].as_u64().unwrap_or(0);
    let indexed_1 = h_after_evict["spill_spilled_starts_indexed"].as_u64().unwrap_or(0);
    assert!(spilled_1 >= 1, "expected spill_starts_spilled >= 1 after RAM pressure, health={h_after_evict}");
    assert!(indexed_1 >= 1, "expected spill_spilled_starts_indexed >= 1, health={h_after_evict}");

    // Restore from SSD for continuation turn WHILE a concurrent unrelated short decode runs on lane 2!
    let short_unrelated = json!([
        {"role": "user", "content": "Count from one to five separated by commas."}
    ]);
    let cont_msgs_bg = turn2_cont_msgs.clone();
    let addr_s1 = spill_srv1.addr;
    let unrelated_handle = std::thread::spawn(move || {
        let began = Instant::now();
        let mut stream = TcpStream::connect_timeout(&addr_s1, Duration::from_secs(10)).unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
        let body = json!({
            "model": "inkling-2-mini",
            "messages": short_unrelated,
            "temperature": 0.0,
            "seed": 7,
            "reasoning_effort": "none",
            "thinking_budget": 2,
            "max_tokens": 16,
            "stream": false
        })
        .to_string();
        write!(
            stream,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr_s1}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        stream.flush().unwrap();
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).unwrap();
        let text = String::from_utf8_lossy(&raw).into_owned();
        assert!(text.starts_with("HTTP/1.1 200 OK"), "concurrent decode failed: {text}");
        began.elapsed()
    });

    let s1_cont = send_chat_greedy(&spill_srv1, cont_msgs_bg, 48);
    let concurrent_elapsed = unrelated_handle.join().unwrap();
    let h_after_restore1 = spill_srv1.wait_drained(Duration::from_secs(15));
    let restored_1 = h_after_restore1["spill_starts_restored"].as_u64().unwrap_or(0);
    let restore_ms_1 = h_after_restore1["spill_last_restore_ms"].as_f64().unwrap_or(0.0);

    assert!(
        s1_cont.cached_tokens >= 1_088,
        "SSD-restored continuation must cache >= 1,088 tokens, got {}",
        s1_cont.cached_tokens
    );
    assert!(restored_1 >= 1, "expected spill_starts_restored >= 1, health={h_after_restore1}");
    assert!(restore_ms_1 > 0.0, "expected positive spill_last_restore_ms, health={h_after_restore1}");
    assert_eq!(s1_cont, ChatResult { cached_tokens: s1_cont.cached_tokens, ..cold_cont.clone() }, "SSD-restored continuation after RAM eviction must match cold 100%");

    // Flush back to SSD so the checkpoint is persisted on disk in `spill-index.bin` before stopping Server #1.
    flush_ram(&spill_srv1);
    let h_before_stop = spill_srv1.wait_drained(Duration::from_secs(15));
    assert!(h_before_stop["spill_spilled_starts_indexed"].as_u64().unwrap_or(0) >= 1);
    spill_srv1.stop();

    // =========================================================================
    // Phase 3: `--spill-dir` Server #2 (Restart) — Continuation + Early Branch
    // =========================================================================
    let spill_srv2 = RunningServer::spawn(&bin, &model, Some(&spill_dir), pool_slots);
    let h_boot2 = spill_srv2.wait_drained(Duration::from_secs(15));
    let boot_indexed = h_boot2["spill_spilled_starts_indexed"].as_u64().unwrap_or(0);
    assert!(
        boot_indexed >= 1,
        "Restarted server must reload spilled starts from spill-index.bin, health={h_boot2}"
    );
    assert_eq!(h_boot2["spill_starts_restored"].as_u64(), Some(0));

    let s2_cont = send_chat_greedy(&spill_srv2, turn2_cont_msgs.clone(), 48);
    let h_after_restart_cont = spill_srv2.wait_drained(Duration::from_secs(15));
    let restart_restore_ms = h_after_restart_cont["spill_last_restore_ms"].as_f64().unwrap_or(0.0);
    assert!(
        s2_cont.cached_tokens >= 1_088,
        "Post-restart SSD restore must cache >= 1,088 tokens, got {}",
        s2_cont.cached_tokens
    );
    assert!(h_after_restart_cont["spill_starts_restored"].as_u64().unwrap_or(0) >= 1);
    assert_eq!(s2_cont, ChatResult { cached_tokens: s2_cont.cached_tokens, ..cold_cont.clone() }, "Post-restart SSD-restored continuation must match cold 100%");

    // Early diverging branch (`branching`) sharing the early blocks of the SSD-restored checkpoint
    // whose early Window blocks had expired past EXT = 1024!
    let s2_branch = send_chat_greedy(&spill_srv2, branch_msgs.clone(), 48);
    assert_eq!(s2_branch, ChatResult { cached_tokens: s2_branch.cached_tokens, ..cold_branch.clone() }, "Early diverging branch after SSD restore must match cold 100%");

    // =========================================================================
    // Phase 4: Corrupt Spilled `.blk` File -> Clean Cache Miss & 100% Cold Parity
    // =========================================================================
    flush_ram(&spill_srv2);
    let h_before_corrupt = spill_srv2.wait_drained(Duration::from_secs(15));
    assert!(h_before_corrupt["spill_spilled_starts_indexed"].as_u64().unwrap_or(0) >= 1);

    let mut corrupted_files = 0usize;
    for entry in std::fs::read_dir(&spill_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) == Some("blk") {
            let mut bytes = std::fs::read(&path).unwrap();
            if !bytes.is_empty() {
                let last = bytes.len() - 1;
                bytes[last] ^= 0xFF;
                std::fs::write(&path, &bytes).unwrap();
                corrupted_files += 1;
            }
        }
    }
    assert!(corrupted_files > 0, "expected .blk files in {}", spill_dir.display());

    let s2_corrupt = send_chat_greedy(&spill_srv2, turn2_cont_msgs, 48);
    let h_after_corrupt = spill_srv2.wait_drained(Duration::from_secs(15));
    let fallbacks = h_after_corrupt["spill_corrupt_fallbacks"].as_u64().unwrap_or(0);
    let removed = h_after_corrupt["spill_corrupt_files_removed"].as_u64().unwrap_or(0);
    assert_eq!(s2_corrupt.cached_tokens, 0, "Corrupted .blk must cleanly fall back to cold prefill (cached_tokens == 0)");
    assert!(fallbacks >= 1, "expected spill_corrupt_fallbacks >= 1, health={h_after_corrupt}");
    assert!(removed >= 1, "expected spill_corrupt_files_removed >= 1, health={h_after_corrupt}");
    assert_eq!(s2_corrupt, cold_cont, "Corrupt .blk fallback must still match cold server 100%");

    eprintln!(
        "[m5-spill-http] PASS: prompt_tokens={} cached_after_evict={} restore1_ms={:.2} restart_restore_ms={:.2} concurrent_decode_ms={:.1} corrupt_fallbacks={} corrupt_removed={}",
        cold_t1.prompt_tokens,
        s1_cont.cached_tokens,
        restore_ms_1,
        restart_restore_ms,
        concurrent_elapsed.as_secs_f64() * 1000.0,
        fallbacks,
        removed
    );

    spill_srv2.stop();
    let _ = std::fs::remove_dir_all(&spill_dir);
}
