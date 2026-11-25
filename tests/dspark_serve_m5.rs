//! Hardware integration test for opt-in DSpark speculative decoding in `inky serve` (`t-later-dspark`):
//!
//! 1. Spawns standard `inky serve` (without `--dspark`) against `Inkling-2-Mini-Q4_K_M.gguf` and records
//!    greedy (`temperature = 0.0`) reference completions on both a structured repetitive coding prompt and
//!    a general prompt, verifying `/health` keeps DSpark disabled by default.
//! 2. Stops the baseline server and spawns `inky serve --dspark <mtp.gguf> --draft-k 3` (using
//!    `INKY_DSPARK_MODEL` / pre-installed `dspark*.gguf` if present, or a generated validated `dflash`
//!    suffix GGUF), verifying:
//!    - Greedy bit-identity: exact output text and completion token count match standard `inky serve` at `temp = 0.0`.
//!    - Seeded `temp > 0` GPU rejection sampling (`temperature = 0.8, top_k = 0, top_p = 1.0`, `k_dspark_reject`)
//!      is 100% deterministic across repeated runs with the same seed.
//!    - Seeded `temp > 0` CPU top-k/top-p rejection sampling (`temperature = 0.8, top_k = 64, top_p = 0.95`)
//!      is 100% deterministic across repeated runs with the same seed.
//!    - `/health` reports non-zero `dspark_verify_steps`, `dspark_draft_tokens_proposed`,
//!      `dspark_draft_tokens_accepted`, `dspark_tokens_emitted_verify`, and `dspark_accepted_tokens_per_verify_step`.

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
    stderr_lines: Arc<Mutex<Vec<String>>>,
}

impl RunningServer {
    fn spawn(bin: &Path, model: &Path, dspark: Option<&Path>, draft_k: usize) -> Self {
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
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        if let Some(dspark_path) = dspark {
            cmd.arg("--dspark")
                .arg(dspark_path)
                .arg("--draft-k")
                .arg(draft_k.to_string());
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
        let dump = self.stderr_lines.lock().unwrap().join("\n");
        panic!(
            "server at {} failed /health readiness within {timeout:?};\n{dump}",
            self.addr
        );
    }

    fn get_json(&self, path: &str) -> Result<(u16, Value), String> {
        self.http_request("GET", path, None)
    }

    fn post_json(&self, path: &str, body: &Value) -> Result<(u16, Value), String> {
        self.http_request("POST", path, Some(&body.to_string()))
    }

    fn http_request(
        &self,
        method: &str,
        path: &str,
        body: Option<&str>,
    ) -> Result<(u16, Value), String> {
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

fn find_model() -> PathBuf {
    if let Ok(p) = std::env::var("INKY_MODEL") {
        let path = PathBuf::from(p);
        if path.exists() {
            return path;
        }
    }
    let home = std::env::var("HOME").expect("HOME set");
    let default = PathBuf::from(home)
        .join(".lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf");
    assert!(
        default.exists(),
        "Inkling-2-Mini GGUF not found at {} (set INKY_MODEL)",
        default.display()
    );
    default
}

/// Write a validated `dflash` suffix-mode GGUF file when no external trained DSpark weights exist on disk.
fn write_test_suffix_dspark_gguf(path: &Path, n_vocab: usize, n_embd: usize, max_order: usize) {
    const GGUF_MAGIC: u32 = 0x4655_4747;
    const GGUF_VERSION: u32 = 3;
    const GGUF_ALIGN: usize = 32;
    const T_F32: u32 = 0;

    let mut buf = Vec::new();
    let push_str = |b: &mut Vec<u8>, s: &str| {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    };
    let push_kv_str = |b: &mut Vec<u8>, k: &str, v: &str| {
        push_str(b, k);
        b.extend_from_slice(&8u32.to_le_bytes());
        push_str(b, v);
    };
    let push_kv_u32 = |b: &mut Vec<u8>, k: &str, v: u32| {
        push_str(b, k);
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&v.to_le_bytes());
    };

    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    buf.extend_from_slice(&6u64.to_le_bytes());
    push_kv_str(&mut buf, "general.architecture", "dflash");
    push_kv_str(&mut buf, "dflash.draft_mode", "suffix");
    push_kv_u32(&mut buf, "dflash.vocab_size", n_vocab as u32);
    push_kv_u32(&mut buf, "dflash.embedding_length", n_embd as u32);
    push_kv_u32(&mut buf, "dflash.block_size", 8);
    push_kv_u32(&mut buf, "dflash.ngram_order", max_order as u32);

    push_str(&mut buf, "dflash.suffix_prior");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&4u64.to_le_bytes());
    buf.extend_from_slice(&T_F32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    let rem = buf.len() % GGUF_ALIGN;
    if rem != 0 {
        buf.resize(buf.len() + (GGUF_ALIGN - rem), 0);
    }
    for v in [1.0f32, 0.5, 0.25, 0.125] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, &buf).expect("write test suffix dspark gguf");
}

fn find_or_create_dspark(tmp_dir: &Path) -> (PathBuf, &'static str) {
    if let Ok(p) = std::env::var("INKY_DSPARK_MODEL") {
        let path = PathBuf::from(p);
        if path.exists() {
            return (path, "env-trained");
        }
    }
    let home = std::env::var("HOME").unwrap_or_default();
    for rel in [
        ".lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf",
        ".lmstudio/models/unsloth/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf",
    ] {
        let cand = PathBuf::from(&home).join(rel);
        if cand.exists() {
            return (cand, "disk-trained");
        }
    }
    let generated = tmp_dir.join("dspark-suffix-test.gguf");
    write_test_suffix_dspark_gguf(&generated, 0, 2304, 4);
    (generated, "generated-suffix")
}

fn chat_completion(
    srv: &RunningServer,
    prompt: &str,
    max_tokens: usize,
    temp: f64,
    top_k: usize,
    top_p: f64,
    seed: u64,
) -> (String, u64) {
    let body = json!({
        "model": "inkling-2-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temp,
        "top_k": top_k,
        "top_p": top_p,
        "seed": seed,
        "stream": false
    });
    let (code, resp) = srv
        .post_json("/v1/chat/completions", &body)
        .expect("POST /v1/chat/completions");
    assert_eq!(code, 200, "unexpected status {code}: {resp}");
    let content = resp["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let gen = resp["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    (content, gen)
}

#[test]
#[ignore = "requires Apple M5 Metal GPU + Inkling-2-Mini GGUF weights"]
fn dspark_serve_greedy_identity_seeded_rejection_and_telemetry_m5() {
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_inky"));
    let model = find_model();
    let tmp_dir = std::env::temp_dir().join(format!("inky-dspark-m5-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let (dspark_path, dspark_source) = find_or_create_dspark(&tmp_dir);

    let prompt_struct = concat!(
        "Repeat the exact Rust function below verbatim without any changes or commentary:\n",
        "```rust\n",
        "pub fn clamp_window(pos: usize, limit: usize) -> usize {\n",
        "    if pos >= limit {\n",
        "        limit\n",
        "    } else {\n",
        "        pos\n",
        "    }\n",
        "}\n",
        "```"
    );
    let prompt_general = "What is 17 * 23? Answer with just the number and one short sentence.";

    // 1. Standard baseline `inky serve` (DSpark disabled by default).
    let (base_struct_txt, base_struct_gen, base_gen_txt, base_gen_toks) = {
        let base_srv = RunningServer::spawn(&bin, &model, None, 3);
        let (code, health) = base_srv.get_json("/health").expect("baseline /health");
        assert_eq!(code, 200);
        assert!(
            health.get("dspark_enabled").is_none(),
            "DSpark must be off by default: {health}"
        );

        let (s_txt, s_gen) = chat_completion(&base_srv, prompt_struct, 48, 0.0, 64, 0.95, 1);
        let (g_txt, g_gen) = chat_completion(&base_srv, prompt_general, 32, 0.0, 64, 0.95, 2);
        base_srv.stop();
        (s_txt, s_gen, g_txt, g_gen)
    };
    assert!(
        base_struct_gen > 8,
        "baseline structured prompt produced too few tokens: {base_struct_gen}"
    );
    assert!(
        base_gen_toks > 3,
        "baseline general prompt produced too few tokens: {base_gen_toks}"
    );

    // 2. Opt-in `--dspark <mtp.gguf> --draft-k 3` live `inky serve`.
    let spec_srv = RunningServer::spawn(&bin, &model, Some(&dspark_path), 3);
    let (code, h0) = spec_srv.get_json("/health").expect("dspark /health");
    assert_eq!(code, 200);
    assert_eq!(h0["dspark_enabled"], json!(true));
    assert_eq!(h0["dspark_draft_k"], json!(3));

    // (a) Greedy bit-identity on both structured (high-acceptance) and general (partial-accept/rollback) prompts.
    let (spec_struct_txt, spec_struct_gen) =
        chat_completion(&spec_srv, prompt_struct, 48, 0.0, 64, 0.95, 1);
    let (_, h_after_struct) = spec_srv.get_json("/health").expect("health after struct");
    let struct_steps = h_after_struct["dspark_verify_steps"].as_u64().unwrap();
    let struct_emitted = h_after_struct["dspark_tokens_emitted_verify"]
        .as_u64()
        .unwrap();
    let struct_acc = h_after_struct["dspark_draft_tokens_accepted"]
        .as_u64()
        .unwrap();
    let struct_tps = h_after_struct["dspark_accepted_tokens_per_verify_step"]
        .as_f64()
        .unwrap();

    let (spec_gen_txt, spec_gen_toks) =
        chat_completion(&spec_srv, prompt_general, 32, 0.0, 64, 0.95, 2);
    assert_eq!(
        spec_struct_txt, base_struct_txt,
        "greedy --dspark output must match standard serve verbatim on structured prompt"
    );
    assert_eq!(spec_struct_gen, base_struct_gen);
    assert_eq!(
        spec_gen_txt, base_gen_txt,
        "greedy --dspark output must match standard serve verbatim on general prompt"
    );
    assert_eq!(spec_gen_toks, base_gen_toks);
    assert!(
        struct_steps > 0,
        "structured prompt must execute verify steps"
    );
    assert!(
        struct_acc > 0,
        "structured prompt must accept draft tokens (got 0)"
    );
    assert!(
        struct_tps > 1.2,
        "expected > 1.2 tokens/verify-step on structured prompt, got {struct_tps:.3} ({struct_emitted}/{struct_steps})"
    );

    // (b) Seeded `temp > 0` GPU rejection sampling (`top_k = 0, top_p = 1.0`, `k_dspark_reject`) repeatability.
    let (_, h_before_gpu_stoch) = spec_srv
        .get_json("/health")
        .expect("health before gpu stoch");
    let steps_before_gpu = h_before_gpu_stoch["dspark_verify_steps"].as_u64().unwrap();
    let (gpu_s1, gpu_g1) = chat_completion(&spec_srv, prompt_struct, 24, 0.8, 0, 1.0, 20261125);
    let (gpu_s2, gpu_g2) = chat_completion(&spec_srv, prompt_struct, 24, 0.8, 0, 1.0, 20261125);
    assert!(
        gpu_g1 > 4,
        "seeded GPU temp=0.8 produced too few tokens: {gpu_g1}"
    );
    assert_eq!(
        gpu_s1, gpu_s2,
        "seeded GPU k_dspark_reject temp=0.8 must be deterministic"
    );
    assert_eq!(gpu_g1, gpu_g2);

    // (c) Seeded `temp > 0` CPU top-k/top-p rejection sampling (`top_k = 64, top_p = 0.95`) repeatability.
    let (cpu_s1, cpu_g1) = chat_completion(&spec_srv, prompt_struct, 24, 0.8, 64, 0.95, 20261126);
    let (cpu_s2, cpu_g2) = chat_completion(&spec_srv, prompt_struct, 24, 0.8, 64, 0.95, 20261126);
    assert!(
        cpu_g1 > 4,
        "seeded top-k/top-p temp=0.8 produced too few tokens: {cpu_g1}"
    );
    assert_eq!(
        cpu_s1, cpu_s2,
        "seeded top-k/top-p verify_round temp=0.8 must be deterministic"
    );
    assert_eq!(cpu_g1, cpu_g2);

    let (_, h_final) = spec_srv.get_json("/health").expect("final /health");
    let final_steps = h_final["dspark_verify_steps"].as_u64().unwrap();
    let final_proposed = h_final["dspark_draft_tokens_proposed"].as_u64().unwrap();
    let final_accepted = h_final["dspark_draft_tokens_accepted"].as_u64().unwrap();
    let final_emitted = h_final["dspark_tokens_emitted_verify"].as_u64().unwrap();
    let final_tps = h_final["dspark_accepted_tokens_per_verify_step"]
        .as_f64()
        .unwrap();
    let final_rate = h_final["dspark_draft_acceptance_rate"].as_f64().unwrap();
    let final_fallbacks = h_final["dspark_capacity_fallbacks"].as_u64().unwrap();

    assert!(
        final_steps > steps_before_gpu,
        "stochastic temp=0.8 requests must execute speculative verify steps"
    );
    eprintln!(
        "[dspark_serve_m5] source={dspark_source} struct_tps={struct_tps:.2} ({struct_emitted} tok / {struct_steps} verify) | total: steps={final_steps} proposed={final_proposed} accepted={final_accepted} emitted={final_emitted} tokens/verify={final_tps:.2} accept_rate={:.1}% fallbacks={final_fallbacks}",
        final_rate * 100.0
    );

    spec_srv.stop();
    let _ = std::fs::remove_dir_all(&tmp_dir);
}
