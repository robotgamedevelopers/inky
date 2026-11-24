//! Opt-in traffic recorder (`--record-dir`) for `inky serve`.
//!
//! Writes newline-delimited JSON (`cards-YYYYMMDD.jsonl`, mode `0600` on Unix) from a bounded
//! background channel so request/decode threads never stall on disk I/O. Auth headers
//! (`Authorization`, `Cookie`, `x-api-key`) are never recorded.

use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::Mutex;
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

pub const SCHEMA_VERSION: u32 = 1;
const DEFAULT_QUEUE_CAPACITY: usize = 1024;
const MAX_TRACKED_CARDS: usize = 16_384;
const MAX_CARD_ID_LEN: usize = 128;

#[derive(Clone, Debug, PartialEq)]
pub struct ResponseMetrics {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub ttft_ms: u64,
    pub total_ms: u64,
    pub finish_reason: String,
    pub status_code: u16,
}

impl ResponseMetrics {
    pub fn to_json(&self) -> Value {
        json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            "finish_reason": self.finish_reason,
            "status_code": self.status_code,
        })
    }

    pub fn from_json(v: &Value) -> Result<Self, String> {
        let obj = v.as_object().ok_or("response must be a JSON object")?;
        let prompt_tokens = obj
            .get("prompt_tokens")
            .and_then(|x| x.as_u64())
            .and_then(|x| usize::try_from(x).ok())
            .ok_or("response.prompt_tokens missing or invalid")?;
        let completion_tokens = obj
            .get("completion_tokens")
            .and_then(|x| x.as_u64())
            .and_then(|x| usize::try_from(x).ok())
            .ok_or("response.completion_tokens missing or invalid")?;
        let ttft_ms = obj
            .get("ttft_ms")
            .and_then(|x| x.as_u64())
            .ok_or("response.ttft_ms missing or invalid")?;
        let total_ms = obj
            .get("total_ms")
            .and_then(|x| x.as_u64())
            .ok_or("response.total_ms missing or invalid")?;
        let finish_reason = obj
            .get("finish_reason")
            .and_then(|x| x.as_str())
            .ok_or("response.finish_reason missing or invalid")?
            .to_string();
        let status_code = obj
            .get("status_code")
            .and_then(|x| x.as_u64())
            .and_then(|x| u16::try_from(x).ok())
            .ok_or("response.status_code missing or invalid")?;
        Ok(Self {
            prompt_tokens,
            completion_tokens,
            ttft_ms,
            total_ms,
            finish_reason,
            status_code,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecordEntry {
    pub schema_version: u32,
    pub recorded_at_ms: u64,
    pub card_id: String,
    pub turn_index: u64,
    pub endpoint: String,
    pub request: Value,
    pub response: ResponseMetrics,
}

impl RecordEntry {
    pub fn to_json(&self) -> Value {
        json!({
            "schema_version": self.schema_version,
            "recorded_at_ms": self.recorded_at_ms,
            "card_id": self.card_id,
            "turn_index": self.turn_index,
            "endpoint": self.endpoint,
            "request": sanitize_request_body(&self.request),
            "response": self.response.to_json(),
        })
    }

    pub fn from_json(v: &Value) -> Result<Self, String> {
        let obj = v.as_object().ok_or("record entry must be a JSON object")?;
        let schema_version = obj
            .get("schema_version")
            .and_then(|x| x.as_u64())
            .and_then(|x| u32::try_from(x).ok())
            .ok_or("schema_version missing or invalid")?;
        let recorded_at_ms = obj
            .get("recorded_at_ms")
            .and_then(|x| x.as_u64())
            .ok_or("recorded_at_ms missing or invalid")?;
        let card_id = obj
            .get("card_id")
            .and_then(|x| x.as_str())
            .ok_or("card_id missing or invalid")?
            .to_string();
        let turn_index = obj
            .get("turn_index")
            .and_then(|x| x.as_u64())
            .ok_or("turn_index missing or invalid")?;
        let endpoint = obj
            .get("endpoint")
            .and_then(|x| x.as_str())
            .unwrap_or("/v1/chat/completions")
            .to_string();
        let request = obj
            .get("request")
            .cloned()
            .ok_or("request missing")?;
        let response = ResponseMetrics::from_json(obj.get("response").ok_or("response missing")?)?;
        Ok(Self {
            schema_version,
            recorded_at_ms,
            card_id,
            turn_index,
            endpoint,
            request,
            response,
        })
    }
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Converts Unix epoch milliseconds to UTC `YYYYMMDD`.
pub fn yyyymmdd_utc(recorded_at_ms: u64) -> String {
    let secs = (recorded_at_ms / 1000) as i64;
    let z = secs.div_euclid(86_400) + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = y + if m <= 2 { 1 } else { 0 };
    format!("{y:04}{m:02}{d:02}")
}

/// Validates and normalizes `X-Inky-Card-Id`; rejects empty or control-character values.
pub fn sanitize_card_id(raw: Option<&str>) -> Option<String> {
    let trimmed = raw?.trim();
    if trimmed.is_empty() || trimmed.bytes().any(|b| b < 0x20 || b == 0x7f) {
        return None;
    }
    let mut out = String::new();
    for ch in trimmed.chars() {
        if out.len() + ch.len_utf8() > MAX_CARD_ID_LEN {
            break;
        }
        out.push(ch);
    }
    (!out.is_empty()).then_some(out)
}

/// Extracts only `X-Inky-Card-Id` from HTTP headers while strictly ignoring auth/cookie headers.
#[allow(dead_code)]
pub fn card_id_from_headers<'a>(headers: impl IntoIterator<Item = (&'a str, &'a str)>) -> Option<String> {
    for (name, value) in headers {
        if name.eq_ignore_ascii_case("x-inky-card-id") {
            if let Some(id) = sanitize_card_id(Some(value)) {
                return Some(id);
            }
        }
    }
    None
}

fn is_sensitive_key(key: &str) -> bool {
    matches!(
        key.to_ascii_lowercase().as_str(),
        "authorization" | "cookie" | "set-cookie" | "x-api-key" | "api_key" | "proxy-authorization"
    )
}

/// Returns a copy of the request JSON with any accidental auth-style top-level keys stripped.
pub fn sanitize_request_body(req: &Value) -> Value {
    match req {
        Value::Object(map) => {
            let mut clean = serde_json::Map::with_capacity(map.len());
            for (k, v) in map {
                if !is_sensitive_key(k) {
                    clean.insert(k.clone(), v.clone());
                }
            }
            Value::Object(clean)
        }
        _ => req.clone(),
    }
}

/// Parses newline-delimited JSON records, ignoring blank lines.
#[allow(dead_code)]
pub fn parse_jsonl(input: &str) -> Result<Vec<RecordEntry>, String> {
    let mut out = Vec::new();
    for (idx, line) in input.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let val: Value = serde_json::from_str(trimmed)
            .map_err(|e| format!("line {}: invalid JSON: {e}", idx + 1))?;
        let entry = RecordEntry::from_json(&val)
            .map_err(|e| format!("line {}: {e}", idx + 1))?;
        out.push(entry);
    }
    Ok(out)
}

/// Groups recorded turns by `card_id`, ordering each card's turns by `(turn_index, recorded_at_ms)`
/// and ordering cards by their earliest `(recorded_at_ms, card_id)`.
#[allow(dead_code)]
pub fn group_by_card(entries: &[RecordEntry]) -> Vec<Vec<RecordEntry>> {
    let mut grouped: HashMap<&str, Vec<RecordEntry>> = HashMap::new();
    for e in entries {
        grouped.entry(e.card_id.as_str()).or_default().push(e.clone());
    }
    let mut cards: Vec<Vec<RecordEntry>> = grouped
        .into_values()
        .map(|mut turns| {
            turns.sort_by_key(|t| (t.turn_index, t.recorded_at_ms));
            turns
        })
        .collect();
    cards.sort_by(|a, b| {
        let ka = (a[0].recorded_at_ms, a[0].card_id.as_str());
        let kb = (b[0].recorded_at_ms, b[0].card_id.as_str());
        ka.cmp(&kb)
    });
    cards
}

#[derive(Default)]
struct TurnTracker {
    next_turn: HashMap<String, u64>,
    order: VecDeque<String>,
}

impl TurnTracker {
    fn next_for(&mut self, card_id: &str) -> u64 {
        if let Some(turn) = self.next_turn.get_mut(card_id) {
            let cur = *turn;
            *turn = cur.saturating_add(1);
            return cur;
        }
        if self.next_turn.len() >= MAX_TRACKED_CARDS {
            if let Some(oldest) = self.order.pop_front() {
                self.next_turn.remove(&oldest);
            }
        }
        self.order.push_back(card_id.to_string());
        self.next_turn.insert(card_id.to_string(), 1);
        0
    }
}

struct Stats {
    written: AtomicUsize,
    dropped: AtomicUsize,
    write_errors: AtomicUsize,
}

impl Stats {
    fn new() -> Self {
        Self {
            written: AtomicUsize::new(0),
            dropped: AtomicUsize::new(0),
            write_errors: AtomicUsize::new(0),
        }
    }
}

pub struct Recorder {
    tx: Option<SyncSender<RecordEntry>>,
    worker: Option<JoinHandle<()>>,
    next_card_seq: AtomicU64,
    turns: Mutex<TurnTracker>,
    stats: std::sync::Arc<Stats>,
}

impl Recorder {
    pub fn new(dir: PathBuf) -> Self {
        Self::with_capacity(dir, DEFAULT_QUEUE_CAPACITY)
    }

    #[allow(dead_code)]
    pub fn with_capacity(dir: PathBuf, capacity: usize) -> Self {
        let (tx, rx) = sync_channel::<RecordEntry>(capacity.max(1));
        let stats = std::sync::Arc::new(Stats::new());
        let worker_stats = stats.clone();
        let worker = std::thread::spawn(move || {
            let mut dir_ready = false;
            let mut warned = false;
            while let Ok(entry) = rx.recv() {
                if !dir_ready {
                    match fs::create_dir_all(&dir) {
                        Ok(()) => dir_ready = true,
                        Err(e) => {
                            worker_stats.write_errors.fetch_add(1, Ordering::Relaxed);
                            if !warned {
                                warned = true;
                                eprintln!("[record] failed to create {}: {e}", dir.display());
                            }
                            continue;
                        }
                    }
                }
                if let Err(e) = append_entry(&dir, &entry) {
                    worker_stats.write_errors.fetch_add(1, Ordering::Relaxed);
                    if !warned {
                        warned = true;
                        eprintln!("[record] write failed in {}: {e}", dir.display());
                    }
                } else {
                    worker_stats.written.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
        Self {
            tx: Some(tx),
            worker: Some(worker),
            next_card_seq: AtomicU64::new(1),
            turns: Mutex::new(TurnTracker::default()),
            stats,
        }
    }

    /// Assigns `(card_id, turn_index)` for an incoming request.
    /// Requests carrying `X-Inky-Card-Id` share a monotonic per-card `turn_index`.
    pub fn allocate_turn(&self, header_card_id: Option<&str>) -> (String, u64) {
        if let Some(id) = sanitize_card_id(header_card_id) {
            let turn = self.turns.lock().unwrap_or_else(|p| p.into_inner()).next_for(&id);
            return (id, turn);
        }
        let seq = self.next_card_seq.fetch_add(1, Ordering::Relaxed);
        (format!("card-{seq}"), 0)
    }

    /// Enqueues an entry without blocking the caller. Returns `false` if the bounded channel was full.
    pub fn record(&self, entry: RecordEntry) -> bool {
        let Some(tx) = &self.tx else { return false };
        match tx.try_send(entry) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => {
                let n = self.stats.dropped.fetch_add(1, Ordering::Relaxed) + 1;
                if n == 1 || n.is_power_of_two() {
                    eprintln!("[record] queue full; dropped {n} record(s)");
                }
                false
            }
            Err(TrySendError::Disconnected(_)) => false,
        }
    }

    #[allow(dead_code)]
    pub fn written_count(&self) -> usize {
        self.stats.written.load(Ordering::Relaxed)
    }

    pub fn dropped_count(&self) -> usize {
        self.stats.dropped.load(Ordering::Relaxed)
    }

    #[allow(dead_code)]
    pub fn write_error_count(&self) -> usize {
        self.stats.write_errors.load(Ordering::Relaxed)
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn append_entry(dir: &Path, entry: &RecordEntry) -> std::io::Result<()> {
    let filename = format!("cards-{}.jsonl", yyyymmdd_utc(entry.recorded_at_ms));
    let path = dir.join(filename);
    let mut opts = OpenOptions::new();
    opts.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut file = opts.open(&path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    let line = serde_json::to_string(&entry.to_json())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    writeln!(file, "{line}")?;
    file.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(1);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("record-test-scratch")
            .join(format!("inky-record-{tag}-{pid}-{n}"))
    }

    #[test]
    fn yyyymmdd_utc_handles_epoch_leap_days_and_dates() {
        assert_eq!(yyyymmdd_utc(0), "19700101");
        // 2024-02-29 12:00:00 UTC (leap day)
        assert_eq!(yyyymmdd_utc(1_709_208_000_000), "20240229");
        // 2026-11-24 18:00:00 UTC
        assert_eq!(yyyymmdd_utc(1_795_543_200_000), "20261124");
    }

    #[test]
    fn jsonl_round_trip_preserves_request_and_response_metrics() {
        let entry = RecordEntry {
            schema_version: SCHEMA_VERSION,
            recorded_at_ms: 1_795_543_200_123,
            card_id: "card-abc".into(),
            turn_index: 2,
            endpoint: "/v1/chat/completions".into(),
            request: json!({
                "model": "inkling-2-mini",
                "messages": [
                    {"role": "user", "content": "Read main.rs"},
                    {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"main.rs\"}"}}]},
                    {"role": "tool", "tool_call_id": "c1", "content": "fn main() {}"}
                ],
                "tools": [{"type": "function", "function": {"name": "read"}}],
                "tool_choice": "auto",
                "temperature": 0.25,
                "max_tokens": 512,
                "thinking_budget": 128,
                "stream": true
            }),
            response: ResponseMetrics {
                prompt_tokens: 240,
                completion_tokens: 64,
                ttft_ms: 19,
                total_ms: 410,
                finish_reason: "stop".into(),
                status_code: 200,
            },
        };
        let line = serde_json::to_string(&entry.to_json()).unwrap();
        let parsed = parse_jsonl(&format!("{line}\n")).unwrap();
        assert_eq!(parsed, vec![entry]);
    }

    #[test]
    fn multi_turn_card_id_grouping_orders_turns_per_card() {
        let mk = |card: &str, turn: u64, ts: u64| RecordEntry {
            schema_version: SCHEMA_VERSION,
            recorded_at_ms: ts,
            card_id: card.into(),
            turn_index: turn,
            endpoint: "/v1/chat/completions".into(),
            request: json!({"messages": [{"role": "user", "content": format!("{card}-{turn}")}]}),
            response: ResponseMetrics {
                prompt_tokens: 10,
                completion_tokens: 5,
                ttft_ms: 1,
                total_ms: 5,
                finish_reason: "stop".into(),
                status_code: 200,
            },
        };
        let entries = vec![
            mk("card-b", 1, 2000),
            mk("card-a", 1, 1500),
            mk("card-b", 0, 1100),
            mk("card-a", 0, 1000),
        ];
        let grouped = group_by_card(&entries);
        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[0][0].card_id, "card-a");
        assert_eq!(grouped[0].iter().map(|e| e.turn_index).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(grouped[1][0].card_id, "card-b");
        assert_eq!(grouped[1].iter().map(|e| e.turn_index).collect::<Vec<_>>(), vec![0, 1]);
    }

    #[test]
    fn recorder_writes_0600_jsonl_and_tracks_turns_by_card_id() {
        let dir = temp_dir("perms");
        let _ = fs::remove_dir_all(&dir);
        let ts = 1_795_543_200_000u64; // 2026-11-24
        {
            let rec = Recorder::new(dir.clone());
            let (c1, t0) = rec.allocate_turn(Some("coding-card-99"));
            let (c2, t1) = rec.allocate_turn(Some("coding-card-99"));
            let (anon1, ta1) = rec.allocate_turn(None);
            let (anon2, ta2) = rec.allocate_turn(None);
            assert_eq!((c1.as_str(), t0), ("coding-card-99", 0));
            assert_eq!((c2.as_str(), t1), ("coding-card-99", 1));
            assert_eq!((anon1.as_str(), ta1), ("card-1", 0));
            assert_eq!((anon2.as_str(), ta2), ("card-2", 0));

            for (cid, turn) in [(c1, t0), (c2, t1)] {
                assert!(rec.record(RecordEntry {
                    schema_version: SCHEMA_VERSION,
                    recorded_at_ms: ts + turn * 100,
                    card_id: cid,
                    turn_index: turn,
                    endpoint: "/v1/chat/completions".into(),
                    request: json!({"messages": [{"role": "user", "content": format!("turn {turn}")}], "stream": false}),
                    response: ResponseMetrics {
                        prompt_tokens: 20,
                        completion_tokens: 10,
                        ttft_ms: 12,
                        total_ms: 42,
                        finish_reason: "stop".into(),
                        status_code: 200,
                    },
                }));
            }
        }
        let file = dir.join("cards-20261124.jsonl");
        let content = fs::read_to_string(&file).unwrap();
        let parsed = parse_jsonl(&content).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].turn_index, 0);
        assert_eq!(parsed[1].turn_index, 1);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&file).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "recorded JSONL must have 0600 permissions");
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn auth_headers_and_keys_are_never_recorded() {
        let headers = [
            ("Authorization", "Bearer super-secret-token-xyz"),
            ("Cookie", "session=top-secret-cookie"),
            ("X-Api-Key", "sk-secret-api-key-999"),
            ("X-Inky-Card-Id", "card-safe-1"),
        ];
        assert_eq!(card_id_from_headers(headers), Some("card-safe-1".to_string()));
        assert_eq!(
            card_id_from_headers([("Authorization", "Bearer secret"), ("Cookie", "a=b")]),
            None
        );

        let dir = temp_dir("auth");
        let _ = fs::remove_dir_all(&dir);
        let ts = 1_795_543_200_000u64;
        {
            let rec = Recorder::new(dir.clone());
            let (cid, turn) = rec.allocate_turn(card_id_from_headers(headers).as_deref());
            rec.record(RecordEntry {
                schema_version: SCHEMA_VERSION,
                recorded_at_ms: ts,
                card_id: cid,
                turn_index: turn,
                endpoint: "/v1/chat/completions".into(),
                request: json!({
                    "Authorization": "Bearer leaked-in-body",
                    "Cookie": "secret=cookie",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": false
                }),
                response: ResponseMetrics {
                    prompt_tokens: 5,
                    completion_tokens: 3,
                    ttft_ms: 4,
                    total_ms: 9,
                    finish_reason: "stop".into(),
                    status_code: 200,
                },
            });
        }
        let content = fs::read_to_string(dir.join("cards-20261124.jsonl")).unwrap();
        assert!(!content.contains("super-secret-token-xyz"));
        assert!(!content.contains("top-secret-cookie"));
        assert!(!content.contains("sk-secret-api-key-999"));
        assert!(!content.contains("leaked-in-body"));
        assert!(!content.contains("secret=cookie"));
        assert!(content.contains("card-safe-1"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_error_is_graceful_and_does_not_panic() {
        let base = temp_dir("err");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let blocker_file = base.join("not-a-directory");
        fs::write(&blocker_file, b"occupied").unwrap();

        let rec = Recorder::new(blocker_file.join("nested"));
        let ok = rec.record(RecordEntry {
            schema_version: SCHEMA_VERSION,
            recorded_at_ms: 1_795_543_200_000,
            card_id: "card-err".into(),
            turn_index: 0,
            endpoint: "/v1/chat/completions".into(),
            request: json!({"messages": [{"role": "user", "content": "hi"}]}),
            response: ResponseMetrics {
                prompt_tokens: 1,
                completion_tokens: 1,
                ttft_ms: 1,
                total_ms: 1,
                finish_reason: "stop".into(),
                status_code: 200,
            },
        });
        assert!(ok);
        let errs = {
            // Wait for background thread to process via drop? We check before/after drop via Arc stats or helper.
            // Dropping joins the worker cleanly, so check via stats handle captured first.
            let stats = rec.stats.clone();
            drop(rec);
            stats.write_errors.load(Ordering::Relaxed)
        };
        assert_eq!(errs, 1, "write error must be counted without panicking");
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn bounded_channel_drops_without_blocking_when_full() {
        let dir = temp_dir("bounded");
        let _ = fs::remove_dir_all(&dir);
        let (tx, _rx) = sync_channel::<RecordEntry>(1);
        let rec = Recorder {
            tx: Some(tx),
            worker: None,
            next_card_seq: AtomicU64::new(1),
            turns: Mutex::new(TurnTracker::default()),
            stats: std::sync::Arc::new(Stats::new()),
        };
        let mk = || RecordEntry {
            schema_version: SCHEMA_VERSION,
            recorded_at_ms: 1_795_543_200_000,
            card_id: "c".into(),
            turn_index: 0,
            endpoint: "/v1/chat/completions".into(),
            request: json!({"messages": []}),
            response: ResponseMetrics {
                prompt_tokens: 0,
                completion_tokens: 0,
                ttft_ms: 0,
                total_ms: 0,
                finish_reason: "stop".into(),
                status_code: 200,
            },
        };
        assert!(rec.record(mk()));
        assert!(!rec.record(mk()), "full bounded queue must drop non-blockingly");
        assert_eq!(rec.dropped_count(), 1);
    }
}
