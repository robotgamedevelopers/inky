//! `inky serve`: OpenAI chat completions and Anthropic Messages endpoints over many lanes. One scheduler thread owns the engine and
//! the lanes and runs one GPU step at a time: a prefill step (lanes with prompt left, up to one chunk of tokens), a verify
//! step (lanes that hold drafts; their own call, not mixed into prefill), or a decode step (one token for every generating
//! lane that has no drafts). The no-drafter path never sets drafts and stays on `encode_decode`. Connection threads parse
//! requests, render the chat template, and stream back what the scheduler sends them. A lane keeps its last prompt and its
//! convolution state at the prompt's end; a request whose tokens extend that prompt continues the lane from there, so a
//! card's next turn prefills only what is new. Only prefill-written cache is ever reused (docs/DESIGN.md, decisions 4 and 5).
//! The same kept prompt may also start a different lane: when the lane holding the longest matching prompt is busy, its
//! prefill-written prefix is copied into a free lane (`copy_lane_prefix`, then the kept convolution state) and only the
//! suffix is prefilled there. A kept prompt supplies its whole length or nothing — the convolution state exists only at its
//! end — so a partial overlap with a lane's prompt is a cold start, exactly as an equal prompt is today.

use crate::engine::{Engine, Lane, Span};
use crate::metal::CmdBuf;
use crate::sampler::{checked_argmax, Sampler};
use crate::tokenizer::Tokenizer;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::Arc;

pub struct ServeOpts { pub model: String, pub ctx: usize, pub bind: String, pub lanes: usize, pub effort: String, pub max_tokens: usize,
                      /// a request must carry this key, as `x-api-key` or `Authorization: Bearer <this>` (from the environment variable --key-env names)
                      pub key: Option<String>,
                      /// requests waiting beyond this many are refused with 429
                      pub queue: usize }

#[derive(Clone, Copy)]
struct Specials { msg_user: u32, msg_model: u32, msg_system: u32, msg_tool: u32, c_text: u32, c_think: u32, c_xml: u32, c_tool: u32, end: u32, end_sampling: u32 }

fn special(tk: &Tokenizer, s: &str) -> u32 {
    let v = tk.encode(s, true);
    assert!(v.len() == 1, "{s} is not one token");
    v[0]
}

fn effort_num(e: &str) -> String {
    let v = match e.trim() {
        "none" => 0.0, "minimal" => 0.1, "low" => 0.2, "medium" => 0.7, "high" => 0.9, "xhigh" | "max" => 0.99,
        x => x.parse::<f64>().unwrap_or(0.9).clamp(0.0, 0.99),
    };
    if v == 0.0 { "0".into() } else { format!("{v}") }
}

fn content_parts(v: &Value) -> Vec<String> {
    match v {
        Value::String(s) => vec![s.clone()],
        Value::Array(a) => a.iter().filter_map(|p| match p {
            Value::String(s) => Some(s.clone()),
            Value::Object(o) => match o.get("type").and_then(|t| t.as_str()) {
                None | Some("text") | Some("input_text") => Some(o.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string()),
                _ => None,
            },
            _ => None,
        }).collect(),
        _ => vec![],
    }
}

/// The chat template of the GGUF, rendered to tokens: tool declarations, system, user, assistant (text and tool calls),
/// tool results, the thinking-effort line. Earlier assistant turns are rendered exactly as the client sends them.
fn render(tk: &Tokenizer, sp: &Specials, req: &Value, default_effort: &str) -> Result<Vec<u32>, String> {
    let text = |t: &mut Vec<u32>, s: &str| t.extend(tk.encode(s, false));
    let mut t = Vec::new();
    let msgs = req["messages"].as_array().ok_or("messages missing")?;
    if let Some(tools) = req["tools"].as_array().filter(|a| !a.is_empty()) {
        let specs: Vec<Value> = tools.iter().map(|tool| {
            let f = tool.get("function").unwrap_or(tool);
            json!({"description": f.get("description").and_then(|d| d.as_str()).unwrap_or(""), "name": f["name"],
                   "parameters": f.get("parameters").cloned().unwrap_or(json!({})), "type": tool.get("type").and_then(|d| d.as_str()).unwrap_or("function")})
        }).collect();
        t.push(sp.msg_system); text(&mut t, "tool_declare"); t.push(sp.c_xml); text(&mut t, &Value::Array(specs).to_string()); t.push(sp.end);
    }
    let effort = req.get("reasoning_effort").and_then(|v| v.as_str()).unwrap_or(default_effort).to_string();
    let emit_effort = |t: &mut Vec<u32>| { t.push(sp.msg_system); t.push(sp.c_text); t.extend(tk.encode(&format!("Thinking effort level: {}", effort_num(&effort)), false)); t.push(sp.end); };
    let mut effort_done = false;
    let mut call_names = std::collections::HashMap::new();
    for m in msgs { for tc in m["tool_calls"].as_array().into_iter().flatten() { if let (Some(id), Some(n)) = (tc["id"].as_str(), tc["function"]["name"].as_str()) { call_names.insert(id.to_string(), n.to_string()); } } }
    for m in msgs {
        let role = m["role"].as_str().ok_or("message without role")?;
        let role = if role == "developer" { "system" } else { role };
        if !effort_done && role != "system" { emit_effort(&mut t); effort_done = true; }
        match role {
            "system" | "user" => { let h = if role == "system" { sp.msg_system } else { sp.msg_user };
                                   for p in content_parts(&m["content"]) { t.push(h); t.push(sp.c_text); text(&mut t, &p); t.push(sp.end); } }
            "tool" => {
                let name = m["name"].as_str().map(|s| s.to_string()).or_else(|| m["tool_call_id"].as_str().and_then(|id| call_names.get(id).cloned())).unwrap_or_default();
                let mut parts = content_parts(&m["content"]);
                if parts.is_empty() { parts.push(String::new()); }
                let is_error = m["is_error"].as_bool().unwrap_or(false);
                for (i, p) in parts.iter().enumerate() {
                    t.push(sp.msg_tool); text(&mut t, &name); t.push(sp.c_text);
                    if is_error && i == 0 { text(&mut t, "Error: "); }
                    text(&mut t, p); t.push(sp.end);
                }
            }
            "assistant" => {
                let content = content_parts(&m["content"]).join("");
                if let Some(r) = m["reasoning_content"].as_str().filter(|r| !r.is_empty()) { t.push(sp.msg_model); t.push(sp.c_think); text(&mut t, r); t.push(sp.end); }
                let calls: Vec<&Value> = m["tool_calls"].as_array().into_iter().flatten().collect();
                if !content.is_empty() || calls.is_empty() { t.push(sp.msg_model); t.push(sp.c_text); text(&mut t, &content); t.push(sp.end); }
                for tc in calls {
                    let name = tc["function"]["name"].as_str().ok_or("tool call without a name")?;
                    let args: Value = match &tc["function"]["arguments"] {
                        Value::String(s) if s.trim().is_empty() => json!({}),
                        Value::String(s) => serde_json::from_str(s).map_err(|e| format!("tool call arguments: {e}"))?,
                        Value::Null => json!({}), v => v.clone(),
                    };
                    t.push(sp.msg_model); text(&mut t, name); t.push(sp.c_tool);
                    text(&mut t, &format!("{{\"name\":{},\"args\":{}}}", Value::String(name.into()), args)); t.push(sp.end);
                }
                t.push(sp.end_sampling);
            }
            r => return Err(format!("unknown role {r}")),
        }
    }
    if !effort_done { emit_effort(&mut t); }
    Ok(t)
}

/// A generated reply split into thinking, text and tool calls.
struct Parsed { text: String, thinking: String, calls: Vec<(String, Value)> }
fn parse(tk: &Tokenizer, sp: &Specials, toks: &[u32]) -> Parsed {
    #[derive(PartialEq)] enum M { Header, Think, Text, Tool, None }
    let (mut mode, mut cur, mut header) = (M::None, Vec::<u8>::new(), Vec::<u8>::new());
    let mut p = Parsed { text: String::new(), thinking: String::new(), calls: vec![] };
    for &t in toks {
        if t == sp.msg_model { mode = M::Header; header.clear(); cur.clear(); continue; }
        if t == sp.c_think { mode = M::Think; cur.clear(); continue; }
        if t == sp.c_text { mode = M::Text; cur.clear(); continue; }
        if t == sp.c_tool { mode = M::Tool; cur.clear(); continue; }
        if t == sp.end || t == sp.end_sampling || t == tk.eos {
            let s = String::from_utf8_lossy(&cur).to_string();
            match mode {
                M::Think => p.thinking += &s, M::Text => p.text += &s,
                M::Tool => { let v: Value = serde_json::from_str(&s).unwrap_or(Value::Null);
                             let name = v["name"].as_str().map(|s| s.to_string()).unwrap_or_else(|| String::from_utf8_lossy(&header).to_string());
                             p.calls.push((name, if v["args"].is_object() { v["args"].clone() } else { json!({"_unparsed": s}) })); }
                _ => {}
            }
            mode = M::None; cur.clear(); continue;
        }
        let b = tk.decode_bytes(t);
        match mode { M::Header => header.extend(b), M::None => {}, _ => cur.extend(b) }
    }
    let s = String::from_utf8_lossy(&cur).to_string();
    match mode { M::Text => p.text += &s, M::Think => p.thinking += &s, _ => {} }
    p
}

// ------------------------------------------------------------------------------------------------- Anthropic Messages
// `POST /v1/messages` speaks the Anthropic Messages API over the same lanes. A request is translated
// to the OpenAI shape above and rendered by the same `render`, so an Anthropic conversation and the
// equivalent chat conversation are the same token stream and a lane's kept prompt is reused across
// either API (decision 4). Thinking has no block in either direction here: a reply's thinking is
// dropped, and thinking blocks in a sent history are dropped with it. Custom stop sequences,
// `tool_choice` and `top_k` have no counterpart in the chat path and are ignored.

/// An Anthropic request as the OpenAI request the chat path renders: `system` becomes leading
/// system messages, a `system` role inside `messages[]` (Claude Code 2.1.330) becomes a system
/// message in place so its order is kept, text blocks keep their order and their per-block
/// rendering, a user message's tool results become tool messages in place, and an assistant
/// message's text and tool uses become its content and tool calls (rendered text first, as the
/// chat path renders them).
fn anthropic_to_openai(req: &Value) -> Result<Value, String> {
    let msgs_in = req["messages"].as_array().ok_or("messages missing")?;
    if msgs_in.is_empty() { return Err("messages must not be empty".into()); }
    let mut msgs: Vec<Value> = Vec::new();
    match &req["system"] {
        Value::Null => {}
        Value::String(s) => msgs.push(json!({"role": "system", "content": s})),
        Value::Array(blocks) => {
            let parts: Vec<Value> = blocks.iter().map(|b| match b["type"].as_str() {
                Some("text") => Ok(json!({"type": "text", "text": b["text"].as_str().unwrap_or("")})),
                t => Err(format!("unsupported system block type {}", t.unwrap_or("none"))),
            }).collect::<Result<_, _>>()?;
            if !parts.is_empty() { msgs.push(json!({"role": "system", "content": parts})); }
        }
        _ => return Err("system must be a string or an array of text blocks".into()),
    }
    for m in msgs_in {
        let role = m["role"].as_str().ok_or("message without role")?;
        match (role, &m["content"]) {
            ("system", Value::String(s)) => msgs.push(json!({"role": "system", "content": s})),
            ("system", Value::Array(blocks)) => {
                let parts: Vec<Value> = blocks.iter().map(|b| match b["type"].as_str() {
                    Some("text") => Ok(json!({"type": "text", "text": b["text"].as_str().unwrap_or("")})),
                    t => Err(format!("unsupported content block type {} in a system message", t.unwrap_or("none"))),
                }).collect::<Result<_, _>>()?;
                msgs.push(json!({"role": "system", "content": parts}));
            }
            ("user", Value::String(s)) => msgs.push(json!({"role": "user", "content": s})),
            ("assistant", Value::String(s)) => msgs.push(json!({"role": "assistant", "content": s})),
            ("user", Value::Array(blocks)) => {
                let mut texts: Vec<Value> = Vec::new();
                for b in blocks {
                    match b["type"].as_str() {
                        Some("text") => texts.push(json!({"type": "text", "text": b["text"].as_str().unwrap_or("")})),
                        Some("image") => texts.push(json!({"type": "text", "text": "[image]"})),
                        Some("tool_result") => {
                            if !texts.is_empty() { msgs.push(json!({"role": "user", "content": std::mem::take(&mut texts)})); }
                            let id = b["tool_use_id"].as_str().ok_or("tool_result without tool_use_id")?;
                            let c = match &b["content"] {
                                Value::Null => json!(""),
                                Value::String(s) => json!(s),
                                Value::Array(parts) => {
                                    let ps: Vec<Value> = parts.iter().map(|p| match p["type"].as_str() {
                                        Some("text") => Ok(json!({"type": "text", "text": p["text"].as_str().unwrap_or("")})),
                                        Some("image") => Ok(json!({"type": "text", "text": "[image]"})),
                                        t => Err(format!("unsupported tool_result content type {}", t.unwrap_or("none"))),
                                    }).collect::<Result<_, _>>()?;
                                    Value::Array(ps)
                                }
                                _ => return Err("tool_result content must be a string or an array of text blocks".into()),
                            };
                            let mut tool_msg = json!({"role": "tool", "tool_call_id": id, "content": c});
                            if b["is_error"].as_bool().unwrap_or(false) { tool_msg["is_error"] = json!(true); }
                            msgs.push(tool_msg);
                        }
                        t => return Err(format!("unsupported content block type {} in a user message", t.unwrap_or("none"))),
                    }
                }
                if !texts.is_empty() { msgs.push(json!({"role": "user", "content": texts})); }
            }
            ("assistant", Value::Array(blocks)) => {
                let mut text = String::new();
                let mut calls: Vec<Value> = Vec::new();
                for b in blocks {
                    match b["type"].as_str() {
                        Some("text") => text.push_str(b["text"].as_str().unwrap_or("")),
                        Some("tool_use") => {
                            let id = b["id"].as_str().ok_or("tool_use without id")?;
                            let name = b["name"].as_str().ok_or("tool_use without a name")?;
                            let input = if b["input"].is_null() { json!({}) } else { b["input"].clone() };
                            calls.push(json!({"id": id, "type": "function", "function": {"name": name, "arguments": input}}));
                        }
                        Some("thinking") | Some("redacted_thinking") => {}
                        t => return Err(format!("unsupported content block type {} in an assistant message", t.unwrap_or("none"))),
                    }
                }
                let mut msg = json!({"role": "assistant", "content": text});
                if !calls.is_empty() { msg["tool_calls"] = json!(calls); }
                msgs.push(msg);
            }
            ("user" | "assistant" | "system", _) => return Err("message content must be a string or an array of blocks".into()),
            (r, _) => return Err(format!("unknown role {r}")),
        }
    }
    let mut out = json!({"messages": msgs});
    if let Some(tools) = req.get("tools").filter(|t| !t.is_null()) {
        let tools = tools.as_array().ok_or("tools must be an array")?;
        if !tools.is_empty() {
            let mapped: Vec<Value> = tools.iter().map(|t| {
                let name = t["name"].as_str().ok_or("tool without a name")?;
                Ok(json!({"type": "function", "function": {"name": name, "description": t["description"].as_str().unwrap_or(""),
                        "parameters": t.get("input_schema").cloned().unwrap_or(json!({}))}}))
            }).collect::<Result<_, String>>()?;
            out["tools"] = json!(mapped);
        }
    }
    Ok(out)
}

/// The reply's stop reason: a tool call outranks the length stop, as the chat path's finish reason does.
fn anthropic_stop_reason(has_calls: bool, reason: &str) -> &'static str {
    if has_calls { "tool_use" } else if reason == "length" { "max_tokens" } else { "end_turn" }
}

/// A random per-process nonce (hex), read once at startup: tool ids carry it so an id
/// issued before a restart never repeats after one. `/dev/urandom` is the source; the
/// time-and-pid scramble is only the fallback for a machine without it.
fn startup_nonce() -> String {
    let mut b = [0u8; 8];
    let have = std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut b)).is_ok();
    if !have {
        let mut x = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
            ^ (std::process::id() as u64).wrapping_mul(0x9E3779B97F4A7C15);
        for byte in b.iter_mut() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; *byte = (x >> 56) as u8; }
    }
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn tool_use_id(nonce: &str, seq: u64, i: usize) -> String { format!("toolu_{nonce}_{seq}_{i}") }

/// A parsed reply as Anthropic content blocks: the text, then each tool call.
fn anthropic_content(p: &Parsed, nonce: &str, seq: u64) -> Vec<Value> {
    let mut c = Vec::new();
    if !p.text.is_empty() { c.push(json!({"type": "text", "text": p.text})); }
    for (i, (name, args)) in p.calls.iter().enumerate() {
        c.push(json!({"type": "tool_use", "id": tool_use_id(nonce, seq, i), "name": name, "input": args}));
    }
    c
}

/// Usage in Anthropic's split: `input_tokens` is the part of the prompt no lane reused,
/// `cache_read_input_tokens` the part it did, so the two sum to the prompt.
fn anthropic_usage(prompt: usize, cached: usize, gen: usize) -> Value {
    json!({"input_tokens": prompt.saturating_sub(cached), "output_tokens": gen,
           "cache_creation_input_tokens": 0, "cache_read_input_tokens": cached})
}

/// The final streaming usage, carried by `message_delta`: the same split as a
/// non-streaming reply. `message_start` is sent before the lane's reuse is known and
/// provisionally reports the whole prompt as `input_tokens`; the delta's split
/// supersedes it, so the client's final accounting sums to the prompt, not to the
/// prompt plus its cached part a second time.
fn streaming_delta_usage(prompt: usize, cached: usize, gen: usize) -> Value { anthropic_usage(prompt, cached, gen) }

fn sse_text(event: &str, data: &Value) -> String { format!("event: {event}\ndata: {data}\n\n") }

/// A streamed reply as content blocks, fed one generated token at a time. Text streams as it
/// decodes; a tool call is buffered to its end and sent whole, since its JSON wraps the arguments
/// the block must stream on their own. Thinking tokens open no block.
struct BlockStream { nonce: String, seq: u64, next: usize, calls: usize, text_index: usize, open_text: bool,
                     mode: u8, pend: Vec<u8>, header: Vec<u8>, tool: Vec<u8>, in_header: bool }
impl BlockStream {
    fn new(nonce: &str, seq: u64) -> BlockStream {
        BlockStream { nonce: nonce.to_string(), seq, next: 0, calls: 0, text_index: 0, open_text: false, mode: 0, pend: vec![], header: vec![], tool: vec![], in_header: false }
    }
    fn flush_text(&mut self, out: &mut Vec<String>) {
        if !self.open_text { self.pend.clear(); return; }
        let valid = match std::str::from_utf8(&self.pend) { Ok(_) => self.pend.len(), Err(e) => e.valid_up_to() };
        if valid > 0 {
            let text = String::from_utf8_lossy(&self.pend[..valid]).to_string();
            self.pend.drain(..valid);
            out.push(sse_text("content_block_delta", &json!({"type": "content_block_delta", "index": self.text_index,
                "delta": {"type": "text_delta", "text": text}})));
        }
    }
    fn close_text(&mut self, out: &mut Vec<String>) {
        if self.open_text {
            self.flush_text(out);
            out.push(sse_text("content_block_stop", &json!({"type": "content_block_stop", "index": self.text_index})));
            self.open_text = false;
        }
        self.pend.clear();
    }
    fn emit_tool(&mut self, out: &mut Vec<String>) {
        let s = String::from_utf8_lossy(&self.tool).to_string();
        let v: Value = serde_json::from_str(&s).unwrap_or(Value::Null);
        let name = v["name"].as_str().map(|x| x.to_string()).unwrap_or_else(|| String::from_utf8_lossy(&self.header).to_string());
        let args = if v["args"].is_object() { v["args"].clone() } else { json!({"_unparsed": s}) };
        let index = self.next; self.next += 1;
        let id = tool_use_id(&self.nonce, self.seq, self.calls); self.calls += 1;
        out.push(sse_text("content_block_start", &json!({"type": "content_block_start", "index": index,
            "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}})));
        out.push(sse_text("content_block_delta", &json!({"type": "content_block_delta", "index": index,
            "delta": {"type": "input_json_delta", "partial_json": args.to_string()}})));
        out.push(sse_text("content_block_stop", &json!({"type": "content_block_stop", "index": index})));
        self.tool.clear();
    }
    fn on_token(&mut self, tk: &Tokenizer, sp: &Specials, t: u32) -> Vec<String> {
        let mut out = Vec::new();
        if t == sp.msg_model { self.close_text(&mut out); self.mode = 0; self.in_header = true; self.header.clear(); return out; }
        if t == sp.c_think { self.close_text(&mut out); self.mode = 1; self.in_header = false; return out; }
        if t == sp.c_text {
            if !self.open_text {
                self.text_index = self.next; self.next += 1; self.open_text = true;
                out.push(sse_text("content_block_start", &json!({"type": "content_block_start", "index": self.text_index,
                    "content_block": {"type": "text", "text": ""}})));
            }
            self.mode = 2; self.in_header = false; return out;
        }
        if t == sp.c_tool { self.close_text(&mut out); self.mode = 3; self.in_header = false; self.tool.clear(); return out; }
        if t == sp.end || t == sp.end_sampling || t == tk.eos {
            if self.mode == 3 { self.emit_tool(&mut out); }
            self.close_text(&mut out);
            self.mode = 0; self.in_header = false; return out;
        }
        let b = tk.decode_bytes(t);
        match self.mode {
            2 => { self.pend.extend(&b); self.flush_text(&mut out); }
            3 => self.tool.extend(&b),
            _ if self.in_header => self.header.extend(&b),
            _ => {}
        }
        out
    }
    /// whatever block is still open when the reply ends. A tool call still open here was
    /// cut short (a length stop sends no end token): it is dropped, never emitted as a
    /// tool_use, exactly as `parse` drops a trailing call with no end. `calls` therefore
    /// counts completed calls only, and the stop reason derives from it alone.
    fn close_all(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        if self.mode == 3 { self.tool.clear(); self.mode = 0; }
        self.close_text(&mut out);
        out
    }
}

// ------------------------------------------------------------------------------------------------- scheduler

struct Job { toks: Vec<u32>, max: usize, temp: f32, top_p: f32, seed: u64, out: Sender<Ev> }
/// to a request's connection: a token, a sign of life after each prefill step (a send that fails means the client is gone),
/// the end, or a refusal
enum Ev { Tok(u32), Alive, Done { prompt: usize, cached: usize, gen: usize, reason: &'static str }, Err(String) }

/// A lane: its stream state on the GPU, the last prompt it prefilled and its convolution state at that prompt's end.
/// `tick` is the scheduler's logical clock (not wall time): a smaller tick was freed longer ago.
struct LaneS { lane: Lane, prompt: Vec<u32>, cs_at_prompt: Option<Vec<u8>>, tick: u64, job: Option<Active> }
struct Active {
    job: Job, n_past: usize, pf_end: usize, cached: usize, gen: usize,
    sampler: Sampler,
    /// Draft stream, seeded once per request (`seed ^ DRAFT_SEED_MIX`). A verify round advances it.
    /// Greedy serve does not read it. It is not the stream `sampler.sample` draws emitted tokens from.
    #[allow(dead_code)]
    draft_sampler: Sampler,
    pending: Option<u32>,
    /// Synthetic drafts for one verify round, length k. None on the no-drafter path, which stays
    /// `encode_decode`. This step does not run a drafter.
    drafts: Option<Vec<u32>>,
    /// Convolution state from before the token a verify span re-encodes (`p0`).
    cs_before: Option<Vec<u8>>,
    /// Tokens a verify step emits, 1 to k+1. Empty on the no-drafter path.
    emit_q: VecDeque<u32>,
}

impl Active {
    fn emit_token(&mut self, tok: u32, stop: &[u32], ctx: usize) -> bool {
        let gone = self.job.out.send(Ev::Tok(tok)).is_err();
        self.gen += 1;
        let budget = reply_budget(self.pf_end, self.gen, self.job.max, ctx);
        let reason = if gone { "cancelled" } else if stop.contains(&tok) { "stop" }
                     else if budget == 0 { "length" } else { return false };
        if gone { eprintln!("[serve] client gone after {} generated tokens; lane freed", self.gen); }
        let _ = self.job.out.send(Ev::Done { prompt: self.pf_end, cached: self.cached, gen: self.gen, reason });
        true
    }
}

/// Tokens two token lists share at their start.
fn shared_prefix(a: &[u32], b: &[u32]) -> usize { a.iter().zip(b).take_while(|(x, y)| x == y).count() }

/// How much of `request` a lane's kept prompt can supply: the prompt's whole length when the lane holds the
/// convolution state at that prompt's end and the prompt is a strict prefix of the request, else nothing. A
/// partial overlap supplies nothing (the state exists only at the prompt's end), and an equal prompt supplies
/// nothing either, so the suffix to prefill is never empty — the own-lane rule, unchanged.
fn reusable_prefix(prompt: &[u32], has_state: bool, request: &[u32]) -> usize {
    let n = shared_prefix(prompt, request);
    if has_state && n == prompt.len() && n < request.len() { n } else { 0 }
}

/// One lane as admission sees it, in logical time (the unit tests drive this with synthetic ticks: no wall
/// clock, no sockets, no GPU).
#[derive(Clone, Copy)]
struct LaneCand { prefix: usize, free: bool, copyable: bool, tick: u64 }

/// Where a request starts: lane `dst` runs it, starting from `prefix` tokens supplied by lane `src`.
/// `src == dst` is the lane's own kept prompt (or a cold start when `prefix` is 0); `src != dst` is a copy.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Admission { dst: usize, src: usize, prefix: usize }

/// Pick the lane for a request and where its prefix comes from. A free lane's own kept prompt comes first
/// (longest, then least recently freed, as before); only a strictly longer prompt held by a busy but copyable
/// lane is copied, into the least recently freed lane. A busy lane is never the destination.
fn choose_lane(cands: &[LaneCand]) -> Option<Admission> {
    let own = cands.iter().enumerate().filter(|(_, c)| c.free)
        .max_by_key(|(_, c)| (c.prefix, std::cmp::Reverse(c.tick)));
    let (dst, own_prefix) = match own { Some((i, c)) => (i, c.prefix), None => return None };
    let src = cands.iter().enumerate().filter(|(_, c)| c.copyable && c.prefix > own_prefix)
        .max_by_key(|(_, c)| (c.prefix, c.tick));
    match src {
        Some((s, c)) if s != dst => {
            let lru = cands.iter().enumerate().filter(|(_, c)| c.free).min_by_key(|(_, c)| c.tick).unwrap().0;
            Some(Admission { dst: lru, src: s, prefix: c.prefix })
        }
        _ => Some(Admission { dst, src: dst, prefix: own_prefix }),
    }
}

/// Copy `src`'s first `n` prefill-written positions into `dst` (the two lanes of one `lanes` slice).
fn copy_prefix(eng: &Engine, lanes: &mut [LaneS], src: usize, dst: usize, n: usize) {
    assert!(src != dst && n > 0);
    let (s, d) = if src < dst { let (a, b) = lanes.split_at_mut(dst); (&a[src], &mut b[0]) }
                 else { let (a, b) = lanes.split_at_mut(src); (&b[0], &mut a[dst]) };
    eng.copy_lane_prefix(&s.lane, &d.lane, n);
}


/// FNV-1a over the raw logit bits of one vocabulary row.
pub(crate) fn logit_hash(row: &[f32]) -> u64 {
    let mut h = 0u64;
    for x in row { h = (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3); }
    h
}

/// One lane the verify planner can see. `lane` is the index the caller uses to map the plan back.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VerifyCand { pub lane: usize, pub p0: usize, pub k: usize, pub kv_cap: usize }

/// A span chosen for this step. `k` is already clamped. The span length is `1 + k`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VerifySpanPlan { pub lane: usize, pub p0: usize, pub k: usize }

/// `k` such that `p0 + 1 + k <= kv_cap`, or none when the committed token itself does not fit.
/// That is the clamp that keeps `encode_spans_x`'s K/V capacity assert from firing.
pub(crate) fn clamp_k(p0: usize, k: usize, kv_cap: usize) -> Option<usize> {
    let room = kv_cap.saturating_sub(p0);
    if room == 0 { return None; }
    Some(k.min(room - 1))
}

/// Pack verify spans into one step. Each position consumes one activation and one logit row.
/// A span that does not fit the remaining budget stops the pack; that lane and every lane behind it
/// wait for a later step. Spans are not split, and a span is not handed to the prefill call.
/// Lanes in one step share one `k`, because `encode_verify` takes a single `k`. Without `multi`
/// (no hgemm) only one span runs. A lane whose committed token does not fit the cache is skipped.
pub(crate) fn plan_verify(cands: &[VerifyCand], rows: usize, multi: bool) -> Vec<VerifySpanPlan> {
    let mut budget = rows;
    let mut out = Vec::new();
    if rows == 0 { return out; }
    for c in cands {
        if !multi && !out.is_empty() { break; }
        let Some(k) = clamp_k(c.p0, c.k.min(rows - 1), c.kv_cap) else { continue; };
        let t = 1 + k;
        if t > budget { break; }
        if out.first().is_some_and(|p: &VerifySpanPlan| p.k != k) { break; }
        budget -= t;
        out.push(VerifySpanPlan { lane: c.lane, p0: c.p0, k });
    }
    out
}

/// What greedy acceptance kept. `rows` are the span rows whose logits predicted an emitted token.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GreedyKeep {
    pub emitted: Vec<u32>,
    pub n_keep: usize,
    pub finished: bool,
    pub rows: Vec<usize>,
    /// Token past the accepted drafts. None when a stop token in the drafts ended the block:
    /// drafts after that stop are not emitted, and there is no bonus.
    pub bonus: Option<u32>,
}

/// Greedy verify, temperature 0. A draft is kept only when it equals that row's argmax.
/// A mismatch keeps the target's token and stops. A stop token in the middle of the drafts
/// emits only the prefix through the stop, and the lane finishes. `rows` has length `k + 1`
/// (one argmax per draft, plus the bonus row). Emits 1 to `k + 1` tokens.
pub(crate) fn accept_greedy(drafts: &[u32], rows: &[u32], stop: &[u32]) -> GreedyKeep {
    assert_eq!(rows.len(), drafts.len() + 1, "one argmax per draft, plus the bonus row");
    let mut emitted = Vec::new();
    let mut used = Vec::new();
    let mut n_acc = 0usize;
    for i in 0..drafts.len() {
        if rows[i] != drafts[i] { break; }
        emitted.push(drafts[i]);
        used.push(i);
        n_acc += 1;
        if stop.contains(&drafts[i]) {
            return GreedyKeep { emitted, n_keep: 1 + n_acc, finished: true, rows: used, bonus: None };
        }
    }
    let bonus = rows[n_acc];
    emitted.push(bonus);
    used.push(n_acc);
    GreedyKeep { emitted, n_keep: 1 + n_acc, finished: stop.contains(&bonus), rows: used, bonus: Some(bonus) }
}

/// One lane ready for `verify_step`. `cs_before` is the convolution state from before `p0`.
/// `drafts` are synthetic: this step does not draft.
pub(crate) struct VerifyIn<'a> {
    pub lane: &'a Lane,
    pub p0: usize,
    pub cs_before: &'a [u8],
    pub drafts: &'a [u32],
}

/// One lane of a verify step that ran. `None` from `verify_step` means the lane waited.
#[derive(Clone, Debug)]
pub(crate) struct VerifyKept {
    pub p0: usize,
    pub k: usize,
    pub emitted: Vec<u32>,
    pub n_keep: usize,
    pub finished: bool,
    pub hashes: Vec<u64>,
    pub bonus: Option<u32>,
}

/// One verify step. `encode_verify` (the `encode_spans` path of length `1 + k`), never
/// `encode_decode` and never `encode_prefill`. Lanes that do not fit this step are `None`.
/// The bonus token is not encoded here: its K and V land on the next step, which is
/// `encode_spans` of `t = 1` for the gate and `encode_decode` only when the lane has no
/// further drafts. Rollback leaves the lane at the accepted prefix.
pub(crate) fn verify_step(eng: &Engine, lanes: &[VerifyIn<'_>], stop: &[u32]) -> Vec<Option<Result<VerifyKept, String>>> {
    let cands: Vec<VerifyCand> = lanes.iter().enumerate().map(|(i, v)| VerifyCand {
        lane: i, p0: v.p0, k: v.drafts.len(), kv_cap: eng.on(v.lane).kv_capacity(),
    }).collect();
    let plans = plan_verify(&cands, eng.verify_row_capacity(), eng.hgemm);
    let mut out = vec![None; lanes.len()];
    if plans.is_empty() { return out; }
    let k = plans[0].k;
    let spans: Vec<Span> = plans.iter().map(|p| Span { lane: lanes[p.lane].lane, p0: p.p0, t: 1 + k }).collect();
    let cb = eng.gpu.cmd();
    // Encoding snapshots the lane before the host stages inputs in the shared buffers.
    eng.encode_verify(&cb, &spans, k);
    for p in &plans {
        let v = &lanes[p.lane];
        let run = eng.on(v.lane);
        run.set_cs(v.cs_before);
        run.set_tokens(p.p0 + 1, &v.drafts[..k]);
    }
    cb.commit();
    let gpu = wait_committed_step(&cb, "verify");
    for (si, p) in plans.iter().enumerate() {
        let result = gpu.clone().and_then(|()| {
            let base = si * (1 + k);
            let logits: Vec<_> = (0..=k).map(|r| eng.span_logits(base + r)).collect();
            let argmaxes: Result<Vec<_>, _> = logits.iter().map(|row| checked_argmax(row)).collect();
            let keep = accept_greedy(&lanes[p.lane].drafts[..k], &argmaxes?, stop);
            let hashes = keep.rows.iter().map(|&r| logit_hash(logits[r])).collect();
            Ok(VerifyKept {
                p0: p.p0, k, emitted: keep.emitted, n_keep: keep.n_keep, finished: keep.finished,
                hashes, bonus: keep.bonus,
            })
        });
        eng.rollback_verify(lanes[p.lane].lane, si, result.as_ref().map_or(0, |kept| kept.n_keep));
        out[p.lane] = Some(result);
    }
    out
}

pub(crate) fn wait_committed_step(cb: &CmdBuf, stage: &str) -> Result<(), String> {
    cb.waitUntilCompleted();
    let status = cb.status();
    if status == MTLCommandBufferStatus::Completed { return Ok(()); }
    if let Some(error) = cb.error() {
        eprintln!("[serve] Metal {stage} failed ({status:?}): {}", error.localizedDescription());
    } else {
        eprintln!("[serve] Metal {stage} failed ({status:?})");
    }
    Err(format!("Metal {stage} failed"))
}

fn reply_budget(prompt: usize, gen: usize, max: usize, ctx: usize) -> usize {
    max.saturating_sub(gen).min(ctx.saturating_sub(prompt.saturating_add(gen).saturating_add(1)))
}

fn fail_lane(eng: &Engine, lane: &mut LaneS, error: String, tick: &mut u64) {
    let a = lane.job.take().unwrap();
    let _ = a.job.out.send(Ev::Err(error));
    lane.prompt.clear();
    lane.cs_at_prompt = None;
    eng.on(&lane.lane).reset();
    lane.tick = *tick;
    *tick += 1;
}

fn scheduler(eng: &Engine, tk: &Tokenizer, sp: Specials, nlanes: usize, rx: Receiver<Job>) {
    let mut lanes: Vec<LaneS> = (0..nlanes).enumerate().map(|(i, _)| LaneS { lane: Lane::new(eng), prompt: vec![], cs_at_prompt: None, tick: i as u64, job: None }).collect();
    let mut tick_seq = nlanes as u64;
    let mut queue: VecDeque<Job> = VecDeque::new();
    let stop = [tk.eos, sp.end_sampling];
    let mut last_was_prefill = false;
    loop {
        // take new requests: block only when there is nothing to do
        let busy = lanes.iter().any(|l| l.job.is_some());
        if !busy && queue.is_empty() { match rx.recv() { Ok(j) => queue.push_back(j), Err(_) => return } }
        loop { match rx.try_recv() { Ok(j) => queue.push_back(j), Err(TryRecvError::Empty) => break, Err(TryRecvError::Disconnected) => return } }
        // admit: a free lane whose kept prompt is the longest prefix of the request, else the least recently freed
        // lane — started, when a busy lane holds a strictly longer prompt it may still copy, from that lane's
        // prefill-written prefix and its kept convolution state. A busy lane may be a source only in the moment
        // its prefill has finished and no decode step has written cache yet (`n_past == pf_end`): after that its
        // sliding-window ring holds decode-written keys over prompt positions, and a lane whose cache left memory
        // for the disk tier cannot supply its prefix either. The source lane is only read, never mutated.
        while let Some(j) = queue.front() {
            let cands: Vec<LaneCand> = lanes.iter().map(|l| LaneCand {
                prefix: reusable_prefix(&l.prompt, l.cs_at_prompt.is_some(), &j.toks),
                free: l.job.is_none(),
                copyable: match &l.job { Some(a) => a.n_past == a.pf_end && !eng.on(&l.lane).tier_on(), None => false },
                tick: l.tick,
            }).collect();
            let Some(adm) = choose_lane(&cands) else { break };
            let j = queue.pop_front().unwrap();
            if j.toks.len().saturating_add(j.max).saturating_add(2) > eng.max_ctx { let _ = j.out.send(Ev::Err(format!("context full: {} prompt tokens and {} to generate, {} per lane", j.toks.len(), j.max, eng.max_ctx))); continue; }
            let i = adm.dst;
            let start = adm.prefix;
            if start > 0 && adm.src == i {
                eng.on(&lanes[i].lane).set_cs(lanes[i].cs_at_prompt.as_ref().unwrap());
            } else if start > 0 {
                // cross-lane: forget the destination, make room, copy the source's prefill-written prefix,
                // then put back the source's kept convolution state (its own state at that prompt's end)
                let cs = lanes[adm.src].cs_at_prompt.clone().unwrap();
                let run = eng.on(&lanes[i].lane);
                run.reset();
                run.reserve(j.toks.len() + j.max + 2, true);
                copy_prefix(eng, &mut lanes, adm.src, i, start);
                eng.on(&lanes[i].lane).set_cs(&cs);
            } else {
                eng.on(&lanes[i].lane).reset();
            }
            let run = eng.on(&lanes[i].lane);
            run.reserve(j.toks.len() + j.max + 2, true);
            run.set_tokens(0, &j.toks);
            let sampler = Sampler::new(j.temp, 64, j.top_p, eng.softcap, j.seed);
            // once per request, not per round and not per emitted token
            let draft_sampler = Sampler::draft(j.temp, 64, j.top_p, eng.softcap, j.seed);
            let l = &mut lanes[i];
            l.prompt.clear(); l.cs_at_prompt = None;
            l.job = Some(Active { pf_end: j.toks.len(), n_past: start, cached: start, gen: 0, sampler, draft_sampler, pending: None, drafts: None, cs_before: None, emit_q: VecDeque::new(), job: j });
        }
        // a step: prefill when a lane has prompt left (alternating with generation), else verify
        // (lanes holding drafts; not mixed into the prefill call), else decode. No drafts: decode.
        let need_pf: Vec<usize> = (0..nlanes).filter(|&i| lanes[i].job.as_ref().is_some_and(|a| a.n_past < a.pf_end)).collect();
        let need_ver: Vec<usize> = (0..nlanes).filter(|&i| lanes[i].job.as_ref().is_some_and(|a| a.n_past >= a.pf_end && a.n_past > 0 && a.drafts.is_some() && a.cs_before.is_some())).collect();
        let need_dec: Vec<usize> = (0..nlanes).filter(|&i| lanes[i].job.as_ref().is_some_and(|a| a.n_past >= a.pf_end && a.pending.is_some() && a.drafts.is_none())).collect();
        if !need_pf.is_empty() && ((need_dec.is_empty() && need_ver.is_empty()) || !last_was_prefill) {
            last_was_prefill = true;
            let mut budget = eng.chunk;
            let mut sp_list: Vec<(usize, usize, usize)> = Vec::new();      // (lane, p0, t)
            let mut snap_last: Vec<(usize, Vec<u8>)> = Vec::new();
            for &i in &need_pf {
                if budget == 0 { break; }
                let (p0, t, snap) = {
                    let a = lanes[i].job.as_ref().unwrap();
                    let t = (a.pf_end - a.n_past).min(budget);
                    // end chunks on 64-position boundaries unless they end the prompt (start points, docs/DESIGN.md)
                    let t = if a.n_past + t < a.pf_end && t > 64 { t - (a.n_past + t) % 64 } else { t };
                    // a verify lane re-encodes the last prompt token, so that token has to be its own step
                    // and the convolution state from before it has to be kept. No-drafter lanes do not split.
                    let t = if a.drafts.is_some() && a.cs_before.is_none() && a.n_past + t == a.pf_end && t > 1 { t - 1 } else { t };
                    let snap = t == 1 && a.drafts.is_some() && a.cs_before.is_none() && a.n_past + t == a.pf_end;
                    (a.n_past, t, snap)
                };
                if snap { snap_last.push((i, eng.on(&lanes[i].lane).get_cs())); }
                sp_list.push((i, p0, t)); budget -= t;
            }
            for (i, cs) in snap_last { lanes[i].job.as_mut().unwrap().cs_before = Some(cs); }
            let spans: Vec<Span> = sp_list.iter().map(|&(i, p0, t)| Span { lane: &lanes[i].lane, p0, t }).collect();
            let cb = eng.gpu.cmd(); eng.encode_spans(&cb, &spans, true, false); cb.commit();
            if let Err(error) = wait_committed_step(&cb, "prefill") {
                for &(i, _, _) in &sp_list { fail_lane(eng, &mut lanes[i], error.clone(), &mut tick_seq); }
                continue;
            }
            let many = spans.len() > 1;
            for (k, &(i, p0, t)) in sp_list.iter().enumerate() {
                let l = &mut lanes[i];
                let a = l.job.as_mut().unwrap();
                a.n_past = p0 + t;
                if a.n_past == a.pf_end {
                    // the prompt is in: keep its state for the next turn, then sample the first token.
                    // a lane that already holds drafts samples that token inside the verify span instead.
                    l.cs_at_prompt = Some(eng.on(&l.lane).get_cs());
                    l.prompt = a.job.toks.clone();
                    if a.drafts.is_none() || a.cs_before.is_none() {
                        let lg = if many { eng.span_logits(k) } else { eng.logits() };
                        match a.sampler.sample_next(lg) {
                            Ok(tok) => a.pending = Some(tok),
                            Err(error) => fail_lane(eng, l, error.into(), &mut tick_seq),
                        }
                    }
                } else if a.job.out.send(Ev::Alive).is_err() {
                    // the client went away during its prefill: free the lane now
                    eprintln!("[serve] lane {i}: client gone during prefill ({} of {} prompt tokens); lane freed", a.n_past, a.pf_end);
                    l.job = None; l.tick = tick_seq; tick_seq += 1;
                }
            }
        } else if !need_ver.is_empty() {
            last_was_prefill = false;
            let mut owned = Vec::new();
            let rows = eng.verify_row_capacity();
            for &i in &need_ver {
                let l = &mut lanes[i];
                let cap = eng.on(&l.lane).kv_capacity();
                let a = l.job.as_ref().unwrap();
                let p0 = a.n_past - 1;
                let budget = reply_budget(a.pf_end, a.gen, a.job.max, eng.max_ctx);
                let drafts = a.drafts.as_ref().unwrap();
                let k = drafts.len().min(budget.saturating_sub(1));
                if !a.sampler.greedy() {
                    fail_lane(eng, l, "synthetic verification requires greedy sampling".into(), &mut tick_seq);
                } else if budget == 0 || rows == 0 || p0 >= cap {
                    fail_lane(eng, l, format!("verify span at {p0} exceeds the reply budget, K/V capacity ({cap}), or logit arena ({rows})"), &mut tick_seq);
                } else {
                    owned.push((i, a.cs_before.clone().unwrap(), drafts[..k].to_vec(), p0));
                }
            }
            let kept = {
                let input: Vec<VerifyIn> = owned.iter().map(|(i, cs, drafts, p0)| VerifyIn {
                    lane: &lanes[*i].lane, p0: *p0, cs_before: cs, drafts,
                }).collect();
                verify_step(eng, &input, &stop)
            };
            for (slot, kopt) in kept.into_iter().enumerate() {
                let Some(result) = kopt else { continue };
                let l = &mut lanes[owned[slot].0];
                let kpt = match result {
                    Ok(kept) => kept,
                    Err(error) => {
                        fail_lane(eng, l, error, &mut tick_seq);
                        continue;
                    }
                };
                let a = l.job.as_mut().unwrap();
                a.emit_q.extend(kpt.emitted);
                // the bonus is emitted now and encoded on the next step. With no further drafts that
                // next step is encode_decode, the no-drafter path. n_past stays on the bonus.
                a.n_past = kpt.p0 + kpt.n_keep;
                a.pending = kpt.bonus;
                a.drafts = None;
                a.cs_before = None;
            }
        } else if !need_dec.is_empty() {
            last_was_prefill = false;
            // feed each lane's pending token at its position
            for &i in &need_dec { let a = lanes[i].job.as_ref().unwrap(); eng.on(&lanes[i].lane).set_tokens(a.n_past, &[a.pending.unwrap()]); }
            let spans: Vec<Span> = need_dec.iter().map(|&i| Span { lane: &lanes[i].lane, p0: lanes[i].job.as_ref().unwrap().n_past, t: 1 }).collect();
            let cb = eng.gpu.cmd(); eng.encode_decode(&cb, &spans, true, false); cb.commit();
            if let Err(error) = wait_committed_step(&cb, "decode") {
                for &i in &need_dec { fail_lane(eng, &mut lanes[i], error.clone(), &mut tick_seq); }
                continue;
            }
            for (k, &i) in need_dec.iter().enumerate() {
                let l = &mut lanes[i];
                let a = l.job.as_mut().unwrap();
                a.n_past += 1;
                match a.sampler.sample_next(eng.span_logits(k)) {
                    Ok(tok) => a.pending = Some(tok),
                    Err(error) => fail_lane(eng, l, error.into(), &mut tick_seq),
                }
            }
        }
        // emit the tokens just sampled; finish lanes that stopped or whose client went away
        for l in lanes.iter_mut() {
            let Some(a) = l.job.as_mut() else { continue };
            // a verify step emits 1 to k+1 tokens. The no-drafter path leaves this queue empty and emits one.
            if !a.emit_q.is_empty() {
                while let Some(tok) = a.emit_q.pop_front() {
                    if a.emit_token(tok, &stop, eng.max_ctx) {
                        l.job = None; l.tick = tick_seq; tick_seq += 1;
                        break;
                    }
                }
                continue;
            }
            let Some(tok) = a.pending else { continue };
            if a.n_past < a.pf_end { continue; }
            // a token is emitted once, right after it is sampled: then n_past - pf_end == gen
            let fresh = a.n_past == a.pf_end + a.gen;
            if !fresh { continue; }
            if a.emit_token(tok, &stop, eng.max_ctx) {
                l.job = None; l.tick = tick_seq; tick_seq += 1;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------- HTTP

fn respond<W: Write + ?Sized>(s: &mut W, code: u16, body: &str) -> std::io::Result<()> {
    write!(s, "HTTP/1.1 {code} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", if code == 200 { "OK" } else { "Error" }, body.len())?;
    s.flush()
}

/// A request's line and headers; the body is read only after the caller has checked the key.
struct Head { method: String, path: String, len: usize, auth: Option<String>, api_key: Option<String>, chunked: bool }
const MAX_BODY: usize = 32 << 20;

fn read_head(r: &mut BufReader<&TcpStream>) -> std::io::Result<Head> {
    let mut line = String::new();
    (&mut *r).take(8192).read_line(&mut line)?;
    let mut it = line.split_whitespace();
    let mut h = Head { method: it.next().unwrap_or("").to_string(), path: it.next().unwrap_or("").to_string(), len: 0, auth: None, api_key: None, chunked: false };
    for _ in 0..128 {
        let mut l = String::new();
        if (&mut *r).take(16384).read_line(&mut l)? == 0 || l == "\r\n" || l == "\n" { return Ok(h); }
        if let Some((k, v)) = l.split_once(':') {
            let v = v.trim();
            if k.eq_ignore_ascii_case("content-length") { h.len = v.parse().unwrap_or(usize::MAX); }
            if k.eq_ignore_ascii_case("transfer-encoding") && v.to_ascii_lowercase().contains("chunked") { h.chunked = true; }
            if k.eq_ignore_ascii_case("authorization") {
                if let Some((scheme, tok)) = v.split_once(' ') { if scheme.eq_ignore_ascii_case("bearer") { h.auth = Some(tok.trim().to_string()); } }
            }
            if k.eq_ignore_ascii_case("x-api-key") { h.api_key = Some(v.to_string()); }
        }
    }
    Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "too many headers"))
}

/// Why a chunked body was refused: over the cap (413) or malformed framing (400).
#[derive(Debug)]
enum BodyError { TooLarge, Malformed(String) }

/// A chunked request body decoded under `max`: chunk sizes are hex with
/// optional extensions, each chunk is followed by CRLF, the zero chunk is
/// followed by header-like trailers and an empty line. Anything else fails
/// closed; the cap is checked before a chunk's bytes are read.
fn read_chunked(r: &mut impl BufRead, max: usize) -> Result<Vec<u8>, BodyError> {
    let bad = |m: &str| BodyError::Malformed(m.to_string());
    let mut body = Vec::new();
    loop {
        let mut line = String::new();
        if r.by_ref().take(8192).read_line(&mut line).map_err(|e| bad(&format!("chunk size: {e}")))? == 0 { return Err(bad("body ended before a chunk size")); }
        let size = line.trim_end_matches(['\r', '\n']).split(';').next().unwrap_or("").trim();
        let n = usize::try_from(u64::from_str_radix(size, 16).map_err(|_| bad("bad chunk size"))?).map_err(|_| bad("chunk size too large"))?;
        if n == 0 {
            loop {
                let mut t = String::new();
                if r.by_ref().take(16384).read_line(&mut t).map_err(|e| bad(&format!("trailer: {e}")))? == 0 { return Err(bad("body ended inside the trailers")); }
                if t == "\r\n" || t == "\n" { return Ok(body); }
                if !t.contains(':') { return Err(bad("bad trailer")); }
            }
        }
        if n > max - body.len().min(max) { return Err(BodyError::TooLarge); }
        let at = body.len();
        body.resize(at + n, 0);
        r.read_exact(&mut body[at..]).map_err(|e| bad(&format!("chunk body: {e}")))?;
        let mut eol = String::new();
        r.read_line(&mut eol).map_err(|e| bad(&format!("chunk end: {e}")))?;
        if eol != "\r\n" && eol != "\n" { return Err(bad("chunk not followed by CRLF")); }
    }
}

/// Decide header-only rejections before allocating or reading the request body.
/// A chunked body has no length to check here: its cap is enforced chunk by
/// chunk in `read_chunked`, after this gate and after the key check.
fn check_headers(h: &Head, path: &str, key: Option<&str>) -> Result<(), (u16, String)> {
    if let Some(k) = key {
        let authed = h.auth.as_deref().is_some_and(|a| same_key(a, k)) || h.api_key.as_deref().is_some_and(|a| same_key(a, k));
        if path != "/health" && !authed {
            return Err((401, "a valid x-api-key or Authorization: Bearer [REDACTED] is required".into()));
        }
    }
    if !h.chunked && h.len > MAX_BODY { return Err((413, format!("the body is over {} bytes", MAX_BODY))); }
    Ok(())
}

/// compare without stopping at the first differing byte
fn same_key(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

/// What a sequence of peek results means for the peer: closed (0 read) or a hard error is gone; data or
/// `WouldBlock` is present. `Interrupted` is transient: the probe is retried a few times, and if it persists
/// the client is treated as still present.
fn gone_from_peek(mut peek: impl FnMut(&mut [u8]) -> std::io::Result<usize>) -> bool {
    let mut b = [0u8; 1];
    let mut tries = 0;
    loop {
        match peek(&mut b) {
            Ok(0) => return true,
            Ok(_) => return false,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return false,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted && tries < 3 => { tries += 1; continue; }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => return false,
            Err(_) => return true,
        }
    }
}

/// The client closed its end (checked without blocking or consuming anything). A peek error other than
/// `WouldBlock` is treated as gone: failing toward cancellation is the safe direction (it frees the lane and the
/// client can retry) rather than holding a lane for a peer that may be dead. `Interrupted` is transient, so the
/// probe is retried a few times and, if it persists, the client is treated as still present.
fn peer_gone(s: &TcpStream) -> bool {
    if s.set_nonblocking(true).is_err() { return true; }
    let gone = gone_from_peek(|b| s.peek(b));
    let _ = s.set_nonblocking(false);
    gone
}

/// The startup shape of a serve run: a context to serve in, and a token budget inside it.
fn check_serve_shape(ctx: usize, max_tokens: usize) -> Result<(), String> {
    if ctx == 0 { return Err("--ctx must be greater than 0".to_string()); }
    if max_tokens == 0 || max_tokens >= ctx { return Err(format!("-n must be between 1 and {} (less than --ctx {})", ctx - 1, ctx)); }
    Ok(())
}

/// What the handlers write to: the socket in production, a buffer in tests.
/// `client_gone` is the socket's peek for a closed peer; a buffer's client never leaves,
/// so a handler driven against a buffer never takes the gone path and needs no socket or timeout.
trait ClientWrite: Write {
    fn client_gone(&self) -> bool;
}
impl ClientWrite for TcpStream {
    fn client_gone(&self) -> bool { peer_gone(self) }
}
impl ClientWrite for Vec<u8> {
    fn client_gone(&self) -> bool { false }
}

struct Ctx { tk: Arc<Tokenizer>, sp: Specials, jobs: Sender<Job>, effort: String, max_tokens: usize, seq: std::sync::atomic::AtomicU64,
             nonce: String, key: Option<String>, ctx: usize,
             /// requests admitted and not yet finished (generating, prefilling or queued for a lane); at most `limit`
             inflight: std::sync::atomic::AtomicUsize, limit: usize,
             /// connection threads alive; at most MAX_CONNS
             conns: Arc<std::sync::atomic::AtomicUsize> }
const MAX_CONNS: usize = 512;

/// Whether a request can never fit a lane: prompt + generation + the two tokens the
/// scheduler reserves past the prompt. Checked on the connection thread before any
/// header is written, so a streaming refusal is a clean HTTP error, not a raw status
/// mid-SSE; the scheduler keeps the same check as a backstop.
fn context_over(prompt: usize, max: usize, ctx: usize) -> bool {
    prompt.saturating_add(max).saturating_add(2) > ctx
}

fn context_full_message(prompt: usize, max: usize, ctx: usize) -> String {
    format!("context full: {prompt} prompt tokens and {max} to generate, {ctx} per lane")
}

/// releases a request's in-flight count however the request ends
struct InFlight<'a>(&'a std::sync::atomic::AtomicUsize);
impl Drop for InFlight<'_> { fn drop(&mut self) { self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst); } }

fn chat<W: ClientWrite + ?Sized>(cx: &Ctx, req: &Value, s: &mut W) -> std::io::Result<()> {
    let toks = match render(&cx.tk, &cx.sp, req, &cx.effort) { Ok(t) => t, Err(e) => return respond(s, 400, &json!({"error": {"message": e}}).to_string()) };
    let seq = cx.seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    let (tx, rx) = channel();
    // every client-controlled number is checked here, never in the scheduler
    let mut bad = |m: String| respond(s, 400, &json!({"error": {"message": m}}).to_string());
    let max = match req["max_tokens"].as_u64().or(req["max_completion_tokens"].as_u64()) {
        Some(v) if v == 0 || v as usize >= cx.ctx => return bad(format!("max_tokens must be between 1 and {}", cx.ctx - 1)),
        Some(v) => v as usize, None => cx.max_tokens,
    };
    let temp = req["temperature"].as_f64().unwrap_or(1.0);
    let top_p = req["top_p"].as_f64().unwrap_or(0.95);
    if !(0.0..=2.0).contains(&temp) { return bad("temperature must be between 0 and 2".into()); }
    if !(top_p > 0.0 && top_p <= 1.0) { return bad("top_p must be in (0, 1]".into()); }
    if context_over(toks.len(), max, cx.ctx) { return bad(context_full_message(toks.len(), max, cx.ctx)); }
    // the seed is the request's own: its `seed`, or a hash of what decides the output (the rendered tokens and sampling)
    let seed = req["seed"].as_u64().unwrap_or_else(|| {
        let mut h = 0xcbf29ce484222325u64;
        for x in toks.iter().map(|&t| t as u64).chain([temp.to_bits(), top_p.to_bits(), max as u64]) { h = (h ^ x).wrapping_mul(0x100000001b3); }
        h
    });
    let job = Job { toks, max, temp: temp as f32, top_p: top_p as f32, seed, out: tx };
    // admission: one atomic count of requests in flight; beyond the limit, refuse now rather than hold the client
    if cx.inflight.fetch_update(std::sync::atomic::Ordering::SeqCst, std::sync::atomic::Ordering::SeqCst, |w| if w < cx.limit { Some(w + 1) } else { None }).is_err() {
        write!(s, "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 5\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{}", json!({"error": {"message": "inky is busy: every lane is taken and the queue is full"}}))?;
        return s.flush();
    }
    let _inflight = InFlight(&cx.inflight);
    if cx.jobs.send(job).is_err() { return respond(s, 503, &json!({"error": {"message": "the scheduler stopped"}}).to_string()); }
    let stream = req["stream"].as_bool().unwrap_or(false);
    let id = format!("chatcmpl-{seq}");
    let model = req["model"].as_str().unwrap_or("inkling-2-mini").to_string();
    let chunk = |delta: Value, finish: Value| json!({"id": id, "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
    if stream {
        write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
        write!(s, "data: {}\n\n", chunk(json!({"role": "assistant", "content": ""}), Value::Null))?;
        s.flush()?;
    }
    let sp = cx.sp;
    let (mut gen, mut mode, mut pend) = (Vec::<u32>::new(), 0u8, Vec::<u8>::new());
    loop {
        // wait for the scheduler, watching the client: a client gone drops `rx`, and the scheduler's next send to this
        // request fails and frees its lane (prefill steps send Alive, decode steps a token)
        let ev = match rx.recv_timeout(std::time::Duration::from_millis(250)) {
            Ok(ev) => ev,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => { if s.client_gone() { return Ok(()); } continue; }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return respond(s, 500, &json!({"error": {"message": "the request was dropped"}}).to_string()),
        };
        match Ok::<Ev, ()>(ev) {
            Ok(Ev::Alive) => { if s.client_gone() { return Ok(()); } }
            Ok(Ev::Tok(t)) => {
                gen.push(t);
                if !stream && gen.len() % 16 == 0 && s.client_gone() { return Ok(()); }
                if !stream { continue; }
                if t == sp.c_think { mode = 1; continue; }
                if t == sp.c_text { mode = 2; continue; }
                if t == sp.c_tool { mode = 3; continue; }
                if t == sp.msg_model || t == sp.end || t == sp.end_sampling || t == cx.tk.eos { mode = 0; pend.clear(); continue; }
                if mode == 1 || mode == 2 {
                    pend.extend(cx.tk.decode_bytes(t));
                    let valid = match std::str::from_utf8(&pend) { Ok(_) => pend.len(), Err(e) => e.valid_up_to() };
                    if valid > 0 {
                        let text = String::from_utf8_lossy(&pend[..valid]).to_string();
                        pend.drain(..valid);
                        let key = if mode == 1 { "reasoning_content" } else { "content" };
                        // a failed write drops `rx`: the scheduler sees the client gone and cancels the lane
                        write!(s, "data: {}\n\n", chunk(json!({key: text}), Value::Null))?;
                        s.flush()?;
                    }
                }
            }
            Ok(Ev::Done { prompt, cached, gen: ng, reason }) => {
                let p = parse(&cx.tk, &sp, &gen);
                let finish = if !p.calls.is_empty() { "tool_calls" } else if reason == "length" { "length" } else { "stop" };
                let calls: Vec<Value> = p.calls.iter().enumerate().map(|(i, (name, args))| json!({"index": i, "id": format!("call_{seq}_{i}"), "type": "function",
                    "function": {"name": name, "arguments": args.to_string()}})).collect();
                let usage = json!({"prompt_tokens": prompt, "completion_tokens": ng, "total_tokens": prompt + ng, "prompt_tokens_details": {"cached_tokens": cached}});
                eprintln!("[serve] #{seq} prompt {prompt} (cached {cached}) gen {ng} finish {finish} calls [{}]", p.calls.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(","));
                if stream {
                    if !calls.is_empty() { write!(s, "data: {}\n\n", chunk(json!({"tool_calls": calls}), Value::Null))?; }
                    let mut last = chunk(json!({}), json!(finish)); last["usage"] = usage;
                    write!(s, "data: {last}\n\ndata: [DONE]\n\n")?;
                    return s.flush();
                }
                let mut msg = json!({"role": "assistant", "content": p.text});
                if !p.thinking.is_empty() { msg["reasoning_content"] = json!(p.thinking); }
                if !calls.is_empty() { msg["tool_calls"] = json!(calls); }
                return respond(s, 200, &json!({"id": id, "object": "chat.completion", "model": model, "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage}).to_string());
            }
            Ok(Ev::Err(e)) => return respond(s, 400, &json!({"error": {"message": e}}).to_string()),
            Err(()) => unreachable!(),
        }
    }
}

/// The route a request path names: the query string and trailing slashes are not part of it
/// (Claude Code posts to `/v1/messages?beta=true`).
fn route_path(p: &str) -> String { p.split('?').next().unwrap_or("").trim_end_matches('/').to_string() }

fn anthropic_error<W: Write + ?Sized>(s: &mut W, code: u16, kind: &str, msg: &str) -> std::io::Result<()> {
    respond(s, code, &json!({"type": "error", "error": {"type": kind, "message": msg}}).to_string())
}

fn messages<W: ClientWrite + ?Sized>(cx: &Ctx, req: &Value, s: &mut W) -> std::io::Result<()> {
    let openai = match anthropic_to_openai(req) { Ok(v) => v, Err(e) => return anthropic_error(s, 400, "invalid_request_error", &e) };
    let toks = match render(&cx.tk, &cx.sp, &openai, &cx.effort) { Ok(t) => t, Err(e) => return anthropic_error(s, 400, "invalid_request_error", &e) };
    let prompt_len = toks.len();
    let seq = cx.seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    let (tx, rx) = channel();
    // every client-controlled number is checked here, never in the scheduler (the chat path's rule)
    let max = match req["max_tokens"].as_u64() {
        Some(v) if v == 0 || v as usize >= cx.ctx => return anthropic_error(s, 400, "invalid_request_error", &format!("max_tokens must be between 1 and {}", cx.ctx - 1)),
        Some(v) => v as usize,
        None => return anthropic_error(s, 400, "invalid_request_error", "max_tokens is required"),
    };
    let temp = req["temperature"].as_f64().unwrap_or(1.0);
    let top_p = req["top_p"].as_f64().unwrap_or(0.95);
    if !(0.0..=2.0).contains(&temp) { return anthropic_error(s, 400, "invalid_request_error", "temperature must be between 0 and 2"); }
    if !(top_p > 0.0 && top_p <= 1.0) { return anthropic_error(s, 400, "invalid_request_error", "top_p must be in (0, 1]"); }
    if context_over(prompt_len, max, cx.ctx) { return anthropic_error(s, 400, "invalid_request_error", &context_full_message(prompt_len, max, cx.ctx)); }
    // the seed is the request's own: a hash of what decides the output, as on the chat path
    let seed = {
        let mut h = 0xcbf29ce484222325u64;
        for x in toks.iter().map(|&t| t as u64).chain([temp.to_bits(), top_p.to_bits(), max as u64]) { h = (h ^ x).wrapping_mul(0x100000001b3); }
        h
    };
    let job = Job { toks, max, temp: temp as f32, top_p: top_p as f32, seed, out: tx };
    if cx.inflight.fetch_update(std::sync::atomic::Ordering::SeqCst, std::sync::atomic::Ordering::SeqCst, |w| if w < cx.limit { Some(w + 1) } else { None }).is_err() {
        write!(s, "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 5\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{}", json!({"type": "error", "error": {"type": "rate_limit_error", "message": "inky is busy: every lane is taken and the queue is full"}}))?;
        return s.flush();
    }
    let _inflight = InFlight(&cx.inflight);
    if cx.jobs.send(job).is_err() { return anthropic_error(s, 503, "api_error", "the scheduler stopped"); }
    let stream = req["stream"].as_bool().unwrap_or(false);
    let id = format!("msg_{seq}");
    let model = req["model"].as_str().unwrap_or("inkling-2-mini").to_string();
    let sp = cx.sp;
    if stream {
        write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
        write!(s, "{}", sse_text("message_start", &json!({"type": "message_start", "message": {"id": id, "type": "message", "role": "assistant",
            "model": model, "content": [], "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": prompt_len, "output_tokens": 0}}})))?;
        s.flush()?;
        let mut bs = BlockStream::new(&cx.nonce, seq);
        loop {
            let ev = match rx.recv_timeout(std::time::Duration::from_millis(250)) {
                Ok(ev) => ev,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if s.client_gone() { return Ok(()); }
                    write!(s, "{}", sse_text("ping", &json!({"type": "ping"})))?; s.flush()?; continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    write!(s, "{}", sse_text("error", &json!({"type": "error", "error": {"type": "api_error", "message": "the request was dropped"}})))?;
                    return s.flush();
                }
            };
            match ev {
                Ev::Alive => { if s.client_gone() { return Ok(()); } write!(s, "{}", sse_text("ping", &json!({"type": "ping"})))?; s.flush()?; }
                Ev::Tok(t) => { for text in bs.on_token(&cx.tk, &sp, t) { write!(s, "{text}")?; } s.flush()?; }
                Ev::Done { prompt, cached, gen: ng, reason } => {
                    for text in bs.close_all() { write!(s, "{text}")?; }
                    let stop = anthropic_stop_reason(bs.calls > 0, reason);
                    eprintln!("[serve] #{seq} messages prompt {prompt} (cached {cached}) gen {ng} stop {stop} calls {calls}", calls = bs.calls);
                    write!(s, "{}", sse_text("message_delta", &json!({"type": "message_delta",
                        "delta": {"stop_reason": stop, "stop_sequence": null},
                        "usage": streaming_delta_usage(prompt, cached, ng)})))?;
                    write!(s, "{}", sse_text("message_stop", &json!({"type": "message_stop"})))?;
                    return s.flush();
                }
                Ev::Err(e) => {
                    for text in bs.close_all() { write!(s, "{text}")?; }
                    write!(s, "{}", sse_text("error", &json!({"type": "error", "error": {"type": "invalid_request_error", "message": e}})))?;
                    return s.flush();
                }
            }
        }
    }
    let mut gen = Vec::<u32>::new();
    loop {
        let ev = match rx.recv_timeout(std::time::Duration::from_millis(250)) {
            Ok(ev) => ev,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => { if s.client_gone() { return Ok(()); } continue; }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return anthropic_error(s, 500, "api_error", "the request was dropped"),
        };
        match ev {
            Ev::Alive => { if s.client_gone() { return Ok(()); } }
            Ev::Tok(t) => { gen.push(t); if gen.len() % 16 == 0 && s.client_gone() { return Ok(()); } }
            Ev::Done { prompt, cached, gen: ng, reason } => {
                let p = parse(&cx.tk, &sp, &gen);
                let stop = anthropic_stop_reason(!p.calls.is_empty(), reason);
                eprintln!("[serve] #{seq} messages prompt {prompt} (cached {cached}) gen {ng} stop {stop} calls [{}]", p.calls.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(","));
                return respond(s, 200, &json!({"id": id, "type": "message", "role": "assistant", "model": model,
                    "content": anthropic_content(&p, &cx.nonce, seq), "stop_reason": stop, "stop_sequence": null,
                    "usage": anthropic_usage(prompt, cached, ng)}).to_string());
            }
            Ev::Err(e) => return anthropic_error(s, 400, "invalid_request_error", &e),
        }
    }
}

/// The would-be prompt's token count; no lane is taken, nothing is generated.
fn count_tokens(cx: &Ctx, req: &Value, s: &mut TcpStream) -> std::io::Result<()> {
    let openai = match anthropic_to_openai(req) { Ok(v) => v, Err(e) => return anthropic_error(s, 400, "invalid_request_error", &e) };
    match render(&cx.tk, &cx.sp, &openai, &cx.effort) {
        Ok(t) => respond(s, 200, &json!({"input_tokens": t.len()}).to_string()),
        Err(e) => anthropic_error(s, 400, "invalid_request_error", &e),
    }
}

pub fn serve(o: ServeOpts) {
    if let Err(e) = check_serve_shape(o.ctx, o.max_tokens) { eprintln!("{e}"); std::process::exit(2); }
    if o.lanes == 0 || o.lanes > crate::engine::MAX_SPANS { eprintln!("--lanes must be between 1 and {}", crate::engine::MAX_SPANS); std::process::exit(2); }
    let eng = Engine::load(&o.model, o.ctx);
    let tk = Arc::new(Tokenizer::load(&eng.g));
    let sp = Specials {
        msg_user: special(&tk, "<|message_user|>"), msg_model: special(&tk, "<|message_model|>"), msg_system: special(&tk, "<|message_system|>"),
        msg_tool: special(&tk, "<|message_tool|>"), c_text: special(&tk, "<|content_text|>"), c_think: special(&tk, "<|content_thinking|>"),
        c_xml: special(&tk, "<|content_xml|>"), c_tool: special(&tk, "<|content_invoke_tool_json|>"), end: special(&tk, "<|end_message|>"),
        end_sampling: special(&tk, "<|content_model_end_sampling|>"),
    };
    eng.ensure_batch(eng.chunk);
    eng.ensure_dbatch(o.lanes);
    let (jtx, jrx) = channel::<Job>();
    let cx = Arc::new(Ctx { tk: tk.clone(), sp, jobs: jtx, effort: o.effort.clone(), max_tokens: o.max_tokens, seq: Default::default(),
                            nonce: startup_nonce(), key: o.key.clone(), ctx: o.ctx, inflight: Default::default(), limit: o.lanes + o.queue, conns: Default::default() });
    let l = TcpListener::bind(&o.bind).expect("bind");
    eprintln!("[serve] http://{}/v1: {} lanes, context {} per lane, effort {}, queue {}, {}", l.local_addr().expect("bound address"), o.lanes, o.ctx, o.effort, o.queue,
              if o.key.is_some() { "a bearer key required" } else { "no key (loopback use only)" });
    std::thread::spawn(move || {
        for c in l.incoming() {
            let Ok(mut s) = c else { continue };
            let cx = cx.clone();
            if cx.conns.fetch_add(1, std::sync::atomic::Ordering::SeqCst) >= MAX_CONNS {
                cx.conns.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                let _ = respond(&mut s, 503, &json!({"error": {"message": "too many connections"}}).to_string());
                continue;
            }
            std::thread::spawn(move || {
                struct Conn(Arc<std::sync::atomic::AtomicUsize>);
                impl Drop for Conn { fn drop(&mut self) { self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst); } }
                let _conn = Conn(cx.conns.clone());
                let _ = s.set_read_timeout(Some(std::time::Duration::from_secs(60)));
                let mut r = BufReader::new(&s);
                let Ok(h) = read_head(&mut r) else { return };
                let path = route_path(&h.path);
                let method = h.method.clone();
                // the key is checked before any body is read
                if let Err((code, message)) = check_headers(&h, &path, cx.key.as_deref()) {
                    drop(r);
                    let _ = if path.starts_with("/v1/messages") { anthropic_error(&mut s, code, if code == 401 { "authentication_error" } else { "invalid_request_error" }, &message) }
                            else { respond(&mut s, code, &json!({"error": {"message": message}}).to_string()) };
                    return;
                }
                // the body is read only now, after the header gate above
                let body = if h.chunked {
                    match read_chunked(&mut r, MAX_BODY) {
                        Ok(b) => b,
                        Err(BodyError::TooLarge) => { drop(r); let _ = if path.starts_with("/v1/messages") { anthropic_error(&mut s, 413, "request_too_large", &format!("the body is over {} bytes", MAX_BODY)) } else { respond(&mut s, 413, &json!({"error": {"message": format!("the body is over {} bytes", MAX_BODY)}}).to_string()) }; return; }
                        Err(BodyError::Malformed(m)) => { drop(r); let _ = if path.starts_with("/v1/messages") { anthropic_error(&mut s, 400, "invalid_request_error", &format!("malformed chunked body: {m}")) } else { respond(&mut s, 400, &json!({"error": {"message": format!("malformed chunked body: {m}")}}).to_string()) }; return; }
                    }
                } else {
                    let mut body = vec![0u8; h.len];
                    if r.read_exact(&mut body).is_err() { return; }
                    body
                };
                drop(r);
                let _ = s.set_read_timeout(None);
                let r = match (method.as_str(), path.as_str()) {
                    ("GET", "/v1/models") => respond(&mut s, 200, &json!({"object": "list", "data": [{"id": "inkling-2-mini", "object": "model", "owned_by": "inky"}]}).to_string()),
                    ("GET", "/health") => respond(&mut s, 200, &json!({"ok": true}).to_string()),
                    ("POST", "/v1/chat/completions") => match serde_json::from_slice::<Value>(&body) {
                        Ok(v) => chat(&cx, &v, &mut s),
                        Err(e) => respond(&mut s, 400, &json!({"error": {"message": e.to_string()}}).to_string()),
                    },
                    ("POST", "/v1/messages") => match serde_json::from_slice::<Value>(&body) {
                        Ok(v) => messages(&cx, &v, &mut s),
                        Err(e) => anthropic_error(&mut s, 400, "invalid_request_error", &e.to_string()),
                    },
                    ("POST", "/v1/messages/count_tokens") => match serde_json::from_slice::<Value>(&body) {
                        Ok(v) => count_tokens(&cx, &v, &mut s),
                        Err(e) => anthropic_error(&mut s, 400, "invalid_request_error", &e.to_string()),
                    },
                    _ => if path.starts_with("/v1/messages") { anthropic_error(&mut s, 404, "not_found_error", &format!("no route {method} {path}")) }
                         else { respond(&mut s, 404, &json!({"error": {"message": format!("no route {method} {path}")}}).to_string()) },
                };
                if let Err(e) = r { eprintln!("[serve] client: {e}"); }
            });
        }
    });
    scheduler(&eng, &tk, sp, o.lanes, jrx);
}

#[cfg(test)]
mod tests {
    use super::*;
    fn test_sp(tk: &Tokenizer) -> Specials {
        Specials {
            msg_user: special(tk, "<|message_user|>"), msg_model: special(tk, "<|message_model|>"), msg_system: special(tk, "<|message_system|>"),
            msg_tool: special(tk, "<|message_tool|>"), c_text: special(tk, "<|content_text|>"), c_think: special(tk, "<|content_thinking|>"),
            c_xml: special(tk, "<|content_xml|>"), c_tool: special(tk, "<|content_invoke_tool_json|>"), end: special(tk, "<|end_message|>"),
            end_sampling: special(tk, "<|content_model_end_sampling|>"),
        }
    }

    /// a reply token stream, as the scheduler would send it: thinking, text, then a tool call
    fn reply_stream(tk: &Tokenizer, sp: &Specials) -> Vec<u32> {
        let mut v = vec![sp.msg_model, sp.c_think];
        v.extend(tk.encode("hmm", false));
        v.push(sp.end);
        v.extend([sp.msg_model, sp.c_text]);
        v.extend(tk.encode("Hello", false));
        v.push(sp.end);
        v.push(sp.msg_model);
        v.extend(tk.encode("get_weather", false));
        v.push(sp.c_tool);
        v.extend(tk.encode("{\"name\":\"get_weather\",\"args\":{\"city\":\"SF\"}}", false));
        v.extend([sp.end, sp.end_sampling]);
        v
    }

    #[test] fn over_cap_requests_are_refused_before_any_header() {
        // exactly fitting (prompt + max + 2 == ctx) is admitted; one token over is not
        assert!(!context_over(10, 5, 17));
        assert!(context_over(11, 5, 17));
        assert!(context_over(10, 6, 17));
        assert!(context_over(0, 1, 2));
        assert!(!context_over(0, 1, 3));
        // saturating: a huge prompt can never wrap into fitting
        assert!(context_over(usize::MAX, 1, 100));
        assert_eq!(context_full_message(11, 5, 17), "context full: 11 prompt tokens and 5 to generate, 17 per lane");
    }

    #[test] fn chunked_bodies_decode_bounded_and_fail_closed() {
        // Stella's preflight (nova-sprint key proxy, Claude Code 2.1.330):
        // the forwarded /v1/messages request arrives chunked, no
        // Content-Length, and serve answered 411.
        use std::io::Cursor;
        let valid = b"4\r\nWiki\r\n5;ext=1\r\npedia\r\n0\r\n\r\n".to_vec();
        assert_eq!(read_chunked(&mut BufReader::new(Cursor::new(valid.clone())), MAX_BODY).unwrap(), b"Wikipedia");
        // a body split across one-byte reads decodes the same
        struct OneByte(Cursor<Vec<u8>>);
        impl Read for OneByte { fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> { let n = b.len().min(1); self.0.read(&mut b[..n]) } }
        impl BufRead for OneByte {
            fn fill_buf(&mut self) -> std::io::Result<&[u8]> { let p = self.0.position() as usize; let v = self.0.get_ref(); Ok(&v[p..(p + 1).min(v.len())]) }
            fn consume(&mut self, n: usize) { self.0.set_position(self.0.position() + n as u64); }
        }
        assert_eq!(read_chunked(&mut OneByte(Cursor::new(valid)), MAX_BODY).unwrap(), b"Wikipedia", "split across reads");
        // over the cap, declared or accumulated, is refused before reading on
        assert!(matches!(read_chunked(&mut BufReader::new(Cursor::new(format!("{:x}\r\n", MAX_BODY + 1).into_bytes())), MAX_BODY), Err(BodyError::TooLarge)), "a declared chunk over the cap");
        assert!(matches!(read_chunked(&mut BufReader::new(Cursor::new(b"6\r\n123456\r\n6\r\n123456\r\n0\r\n\r\n".to_vec())), 10), Err(BodyError::TooLarge)), "chunks accumulating over the cap");
        // malformed framing fails closed
        assert!(matches!(read_chunked(&mut BufReader::new(Cursor::new(b"zz\r\nabc\r\n0\r\n\r\n".to_vec())), MAX_BODY), Err(BodyError::Malformed(_))), "a non-hex chunk size");
        assert!(matches!(read_chunked(&mut BufReader::new(Cursor::new(b"0\r\nBadTrailer\r\n\r\n".to_vec())), MAX_BODY), Err(BodyError::Malformed(_))), "a trailer without a colon");
        assert!(matches!(read_chunked(&mut BufReader::new(Cursor::new(b"5\r\nab".to_vec())), MAX_BODY), Err(BodyError::Malformed(_))), "a chunk cut short");
        assert!(matches!(read_chunked(&mut BufReader::new(Cursor::new(b"3\r\nabcXX\r\n0\r\n\r\n".to_vec())), MAX_BODY), Err(BodyError::Malformed(_))), "no CRLF after the chunk data");
    }

    #[test] fn routes_ignore_query_strings_and_trailing_slashes() {
        assert_eq!(route_path("/v1/messages?beta=true"), "/v1/messages");
        assert_eq!(route_path("/v1/messages/count_tokens?beta=true"), "/v1/messages/count_tokens");
        assert_eq!(route_path("/v1/chat/completions/"), "/v1/chat/completions");
        assert_eq!(route_path("/health"), "/health");
    }

    #[test] fn anthropic_requests_translate_to_the_chat_shape() {
        let a = json!({"model": "inkling-2-mini", "max_tokens": 64,
            "system": [{"type": "text", "text": "Be terse."}, {"type": "text", "text": "Be kind."}],
            "tools": [{"name": "get_weather", "description": "Weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Weather in SF?"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "Sunny"}]},
                    {"type": "text", "text": "Thanks"}]}]});
        let o = anthropic_to_openai(&a).unwrap();
        assert_eq!(o["messages"][0], json!({"role": "system", "content": [{"type": "text", "text": "Be terse."}, {"type": "text", "text": "Be kind."}]}));
        assert_eq!(o["messages"][1], json!({"role": "user", "content": [{"type": "text", "text": "Weather in SF?"}]}));
        assert_eq!(o["messages"][2]["role"], "assistant");
        assert_eq!(o["messages"][2]["content"], "Checking.");
        assert_eq!(o["messages"][2]["tool_calls"][0]["id"], "toolu_1");
        assert_eq!(o["messages"][2]["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(o["messages"][2]["tool_calls"][0]["function"]["arguments"], json!({"city": "SF"}));
        assert_eq!(o["messages"][3], json!({"role": "tool", "tool_call_id": "toolu_1", "content": [{"type": "text", "text": "Sunny"}]}));
        assert_eq!(o["messages"][4], json!({"role": "user", "content": [{"type": "text", "text": "Thanks"}]}));
        assert_eq!(o["tools"][0]["type"], "function");
        assert_eq!(o["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(o["tools"][0]["function"]["parameters"], json!({"type": "object", "properties": {"city": {"type": "string"}}}));
        // string system and string content take the plain path
        let o = anthropic_to_openai(&json!({"system": "Sys", "messages": [{"role": "user", "content": "Hi"}]})).unwrap();
        assert_eq!(o["messages"][0], json!({"role": "system", "content": "Sys"}));
        assert_eq!(o["messages"][1], json!({"role": "user", "content": "Hi"}));
        assert!(o.get("tools").is_none(), "no tools in, no tools out");
    }

    #[test] fn system_role_messages_from_claude_code_translate_in_order() {
        // Claude Code 2.1.330 (Rowan, MacBook, PR 19 at a696454) sends a
        // role:"system" entry inside messages[] and serve refused it with
        // 400 "unknown role system" at the first turn. It must fold into
        // the system prompt in order, exactly as the chat path renders a
        // system message in place.
        let a = json!({"model": "inkling-2-mini", "max_tokens": 64,
            "system": [{"type": "text", "text": "Top system."}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
                {"role": "system", "content": [{"type": "text", "text": "First in-message system."}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
                {"role": "system", "content": "Later system note."},
                {"role": "user", "content": "Continue"}]});
        let o = anthropic_to_openai(&a).unwrap();
        assert_eq!(o["messages"][0], json!({"role": "system", "content": [{"type": "text", "text": "Top system."}]}));
        assert_eq!(o["messages"][1], json!({"role": "user", "content": [{"type": "text", "text": "Hi"}]}));
        assert_eq!(o["messages"][2], json!({"role": "system", "content": [{"type": "text", "text": "First in-message system."}]}));
        assert_eq!(o["messages"][3]["role"], "assistant");
        assert_eq!(o["messages"][4], json!({"role": "system", "content": "Later system note."}));
        assert_eq!(o["messages"][5], json!({"role": "user", "content": "Continue"}));
        // and it renders to the same tokens as the equivalent chat request
        let tk = Tokenizer::for_test();
        let sp = test_sp(&tk);
        // a chat twin: the top-level system array is one system message,
        // the in-message system entries follow it in place
        let chat = json!({"messages": [
            {"role": "system", "content": [{"type": "text", "text": "Top system."}]},
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "First in-message system."},
            {"role": "assistant", "content": "Hello"},
            {"role": "system", "content": "Later system note."},
            {"role": "user", "content": "Continue"}]});
        assert_eq!(render(&tk, &sp, &o, "medium").unwrap(), render(&tk, &sp, &chat, "medium").unwrap(),
            "a system role inside messages[] must render as the chat path renders a system message in place");
    }

    #[test] fn anthropic_translation_refuses_what_it_cannot_render() {
        assert!(anthropic_to_openai(&json!({"messages": []})).is_err(), "no messages");
        assert!(anthropic_to_openai(&json!({"messages": [{"role": "system", "content": [{"type": "document", "source": {}}]}]})).is_err(), "an unsupported system block fails closed, never dropped");
        assert!(anthropic_to_openai(&json!({"messages": [{"role": "user", "content": [{"type": "document", "source": {}}]}]})).is_err(), "an unsupported block fails closed, never dropped");
        assert!(anthropic_to_openai(&json!({"messages": [{"role": "user", "content": [{"type": "tool_result", "content": "x"}]}]})).is_err(), "a tool result without its id");
        assert!(anthropic_to_openai(&json!({"tools": [{"description": "x"}], "messages": [{"role": "user", "content": "x"}]})).is_err(), "a tool without a name");
        // a thinking block in a sent history is the one block dropped: it has no rendering in either API here
        let o = anthropic_to_openai(&json!({"messages": [{"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Done"}]}]})).unwrap();
        assert_eq!(o["messages"][0]["content"], "Done");
    }

    #[test] fn anthropic_and_chat_requests_render_the_same_tokens() {
        let tk = Tokenizer::for_test();
        let sp = test_sp(&tk);
        let anthropic = json!({"model": "inkling-2-mini", "max_tokens": 64, "system": "Be terse.",
            "tools": [{"name": "get_weather", "description": "Weather", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "Weather in SF?"},
                {"role": "assistant", "content": [{"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny"}]}]});
        // the same conversation written as a chat request (arguments as the string a chat client sends)
        let chat = json!({"tools": [{"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": {"type": "object"}}}],
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Weather in SF?"},
                {"role": "assistant", "content": "Checking.", "tool_calls": [{"id": "toolu_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}}]},
                {"role": "tool", "tool_call_id": "toolu_1", "content": "Sunny"}]});
        let a = render(&tk, &sp, &anthropic_to_openai(&anthropic).unwrap(), "medium").unwrap();
        let c = render(&tk, &sp, &chat, "medium").unwrap();
        assert_eq!(a, c, "the Anthropic request must render to the chat path's token stream, or prompt reuse misses");
        // and the next turn extends the last, which is what makes a lane's kept prompt hit
        let mut next = anthropic.clone();
        next["messages"].as_array_mut().unwrap().push(json!({"role": "assistant", "content": [{"type": "text", "text": "Sunny in SF."}]}));
        next["messages"].as_array_mut().unwrap().push(json!({"role": "user", "content": "And in LA?"}));
        let n = render(&tk, &sp, &anthropic_to_openai(&next).unwrap(), "medium").unwrap();
        assert!(n.starts_with(&a), "a following Anthropic turn must extend the previous prompt");
    }

    #[test] fn corrupted_equivalents_do_not_render_the_same_tokens() {
        // reversed witness for the equivalence above: a deliberately broken equivalent
        // must NOT render to the same tokens. If render ever dropped the system block
        // or stopped caring about message order, this test goes red.
        let tk = Tokenizer::for_test();
        let sp = test_sp(&tk);
        let anthropic = json!({"model": "inkling-2-mini", "max_tokens": 64, "system": "Be terse.",
            "tools": [{"name": "get_weather", "description": "Weather", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "Weather in SF?"},
                {"role": "assistant", "content": [{"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny"}]}]});
        let chat = json!({"tools": [{"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": {"type": "object"}}}],
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Weather in SF?"},
                {"role": "assistant", "content": "Checking.", "tool_calls": [{"id": "toolu_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}}]},
                {"role": "tool", "tool_call_id": "toolu_1", "content": "Sunny"}]});
        let a = render(&tk, &sp, &anthropic_to_openai(&anthropic).unwrap(), "medium").unwrap();
        let good = render(&tk, &sp, &chat, "medium").unwrap();
        assert_eq!(a, good, "sanity: the uncorrupted equivalent still renders the same tokens");
        // corruption 1: the system block dropped from the chat equivalent
        let mut no_system = chat.clone();
        no_system["messages"].as_array_mut().unwrap().retain(|m| m["role"] != "system");
        let dropped = render(&tk, &sp, &no_system, "medium").unwrap();
        assert_ne!(a, dropped, "dropping the system block must change the rendered token stream");
        // corruption 2: the tool result reordered ahead of the assistant call it answers
        let mut reordered = chat.clone();
        let msgs = reordered["messages"].as_array_mut().unwrap();
        let tool = msgs.remove(3);
        msgs.insert(2, tool);
        let moved = render(&tk, &sp, &reordered, "medium").unwrap();
        assert_ne!(a, moved, "reordering the tool result ahead of its call must change the rendered token stream");
    }

    #[test] fn a_user_message_image_renders_the_placeholder_not_a_refusal() {
        let tk = Tokenizer::for_test();
        let sp = test_sp(&tk);
        // an image directly in a user message (not inside a tool_result) renders the
        // same [image] placeholder as an image in a tool result, never a 400 that
        // would poison every later turn carrying the same history
        let req = json!({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "look"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}]},
            {"role": "assistant", "content": "seen"},
            {"role": "user", "content": "again?"}]});
        let o = anthropic_to_openai(&req).unwrap();
        assert_eq!(o["messages"][0]["content"][1], json!({"type": "text", "text": "[image]"}),
            "a user-message image becomes the [image] text placeholder");
        let toks = render(&tk, &sp, &o, "medium").unwrap();
        let img: Vec<u32> = tk.encode("[image]", false);
        assert!(toks.windows(img.len()).any(|w| w == img.as_slice()), "the placeholder must be in the rendered tokens");
        // history carrying the image still extends: a following turn renders on top of it
        let mut next = req.clone();
        next["messages"].as_array_mut().unwrap().push(json!({"role": "assistant", "content": [{"type": "text", "text": "ok"}]}));
        next["messages"].as_array_mut().unwrap().push(json!({"role": "user", "content": "and now?"}));
        let n = render(&tk, &sp, &anthropic_to_openai(&next).unwrap(), "medium").unwrap();
        assert!(n.starts_with(&toks), "a history with a user-message image must extend, not poison later turns");
        // an image-only user message renders the placeholder too
        let only = anthropic_to_openai(&json!({"messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]})).unwrap();
        assert_eq!(only["messages"][0]["content"][0], json!({"type": "text", "text": "[image]"}));
    }

    #[test] fn a_generated_reply_parses_to_anthropic_blocks() {
        let tk = Tokenizer::for_test();
        let sp = test_sp(&tk);
        let p = parse(&tk, &sp, &reply_stream(&tk, &sp));
        assert_eq!(p.text, "Hello");
        assert_eq!(p.thinking, "hmm");
        assert_eq!(p.calls.len(), 1);
        assert_eq!(p.calls[0], ("get_weather".to_string(), json!({"city": "SF"})));
        let content = anthropic_content(&p, "abc123", 7);
        assert_eq!(content[0], json!({"type": "text", "text": "Hello"}));
        assert_eq!(content[1], json!({"type": "tool_use", "id": "toolu_abc123_7_0", "name": "get_weather", "input": {"city": "SF"}}));
        assert_eq!(anthropic_stop_reason(true, "stop"), "tool_use");
        assert_eq!(anthropic_stop_reason(false, "length"), "max_tokens");
        assert_eq!(anthropic_stop_reason(false, "stop"), "end_turn");
        assert_eq!(anthropic_usage(100, 80, 5), json!({"input_tokens": 20, "output_tokens": 5, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 80}));
    }

    #[test] fn a_streamed_reply_is_the_claude_code_event_sequence() {
        let tk = Tokenizer::for_test();
        let sp = test_sp(&tk);
        assert_eq!(sse_text("ping", &json!({"type": "ping"})), "event: ping\ndata: {\"type\":\"ping\"}\n\n");
        let mut bs = BlockStream::new("abc123", 7);
        let mut events: Vec<Value> = Vec::new();
        for t in reply_stream(&tk, &sp) {
            for text in bs.on_token(&tk, &sp, t) {
                let data = text.strip_prefix("event: ").unwrap().split("\ndata: ").nth(1).unwrap().trim_end();
                events.push(serde_json::from_str(data).unwrap());
            }
        }
        events.extend(bs.close_all().iter().map(|text| serde_json::from_str(text.split("\ndata: ").nth(1).unwrap().trim_end()).unwrap()));
        // the thinking opens no block; the text streams (one delta per character token here), then the tool block
        assert_eq!(events[0]["type"], "content_block_start");
        assert_eq!(events[0]["content_block"], json!({"type": "text", "text": ""}));
        let text: String = events.iter().filter(|e| e["delta"]["type"] == "text_delta")
            .map(|e| e["delta"]["text"].as_str().unwrap().to_string()).collect();
        assert_eq!(text, "Hello");
        let tool_at = events.iter().position(|e| e["content_block"]["type"] == "tool_use").unwrap();
        assert!(events[..tool_at].iter().all(|e| e["type"] != "content_block_start" || e["index"] == events[0]["index"]),
            "only the text block opens before the tool block");
        assert_eq!(events[tool_at - 1]["type"], "content_block_stop", "the text block stops before the tool block starts");
        assert_eq!(events[tool_at]["content_block"]["id"], "toolu_abc123_7_0");
        assert_eq!(events[tool_at]["content_block"]["name"], "get_weather");
        assert_eq!(events[tool_at + 1]["delta"]["type"], "input_json_delta");
        assert_eq!(serde_json::from_str::<Value>(events[tool_at + 1]["delta"]["partial_json"].as_str().unwrap()).unwrap(), json!({"city": "SF"}));
        assert_eq!(events[tool_at + 2]["type"], "content_block_stop");
        assert_eq!(events[tool_at + 2]["index"], events[tool_at]["index"], "a block stops at its own index");
        assert_eq!(bs.calls, 1);
    }

    #[test] fn a_tool_call_cut_off_by_length_is_not_a_tool_use() {
        let tk = Tokenizer::for_test();
        let sp = test_sp(&tk);
        // a tool call with no end token: generation stopped (length) mid-call
        let mut bs = BlockStream::new("abc123", 7);
        let mut toks = vec![sp.msg_model];
        toks.extend(tk.encode("get_weather", false));
        toks.push(sp.c_tool);
        toks.extend(tk.encode("{\"name\":\"get_weather\",\"args\":{\"ci", false));
        let mut events: Vec<Value> = Vec::new();
        for t in toks { for text in bs.on_token(&tk, &sp, t) {
            events.push(serde_json::from_str(text.split("\ndata: ").nth(1).unwrap().trim_end()).unwrap());
        } }
        for text in bs.close_all() {
            events.push(serde_json::from_str(text.split("\ndata: ").nth(1).unwrap().trim_end()).unwrap());
        }
        assert!(events.iter().all(|e| e["content_block"]["type"] != "tool_use"),
            "a partial tool call must not stream as a tool_use block: {events:?}");
        assert_eq!(bs.calls, 0, "only completed calls count toward the stop reason");
        assert_eq!(anthropic_stop_reason(bs.calls > 0, "length"), "max_tokens");
    }

    #[test] fn error_and_empty_and_image_tool_results_render() {
        let tk = Tokenizer::for_test();
        let sp = test_sp(&tk);
        let render_one = |content: Value, is_error: bool| -> Vec<u32> {
            let mut block = json!({"type": "tool_result", "tool_use_id": "toolu_1", "content": content});
            if is_error { block["is_error"] = json!(true); }
            let a = json!({"messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}]},
                {"role": "user", "content": [block]}]});
            render(&tk, &sp, &anthropic_to_openai(&a).unwrap(), "medium").unwrap()
        };
        // an error result is marked, so the model can tell it from a success with the same text
        let ok = render_one(json!("Boom"), false);
        let err = render_one(json!("Boom"), true);
        assert_ne!(ok, err, "an error tool result must render differently from a successful one");
        let err_text: Vec<u32> = tk.encode("Error", false);
        assert!(err.windows(err_text.len()).any(|w| w == err_text.as_slice()), "the error mark must be in the rendered tokens");
        // an image inside a tool result becomes a text placeholder, never a 400
        let img = render_one(json!([{"type": "image", "source": {"type": "base64", "data": "AA=="}}]), false);
        assert!(img.contains(&sp.msg_tool), "an image tool result must still render as a tool result");
        // an empty result still renders a tool block (an empty success, not silence)
        let empty = render_one(json!([]), false);
        assert!(empty.contains(&sp.msg_tool), "an empty tool result must render a tool block");
        let empty_str = render_one(json!(""), false);
        assert_eq!(empty, empty_str, "an empty array and an empty string are the same empty result");
    }

    #[test] fn streaming_delta_usage_matches_the_non_streaming_split() {
        // the delta must carry the corrected split: input (uncached) + cache_read == prompt.
        // Before the fix the delta carried no input_tokens at all, so a client adding its
        // cache_read to message_start's whole-prompt input_tokens counted the prompt twice.
        let d = streaming_delta_usage(100, 80, 5);
        assert_eq!(d, anthropic_usage(100, 80, 5), "streaming and non-streaming usage must agree");
        assert_eq!(d["input_tokens"], 20);
        assert_eq!(d["cache_read_input_tokens"], 80);
        assert_eq!(d["input_tokens"].as_u64().unwrap() + d["cache_read_input_tokens"].as_u64().unwrap(), 100);
        assert_eq!(streaming_delta_usage(50, 0, 3)["input_tokens"], 50, "nothing cached, everything is input");
        assert_eq!(streaming_delta_usage(50, 50, 3)["input_tokens"], 0, "fully cached, no fresh input");
    }

    #[test] fn tool_ids_carry_the_startup_nonce() {
        assert_eq!(tool_use_id("abc123", 7, 0), "toolu_abc123_7_0");
        assert_ne!(tool_use_id("abc123", 7, 0), tool_use_id("def456", 7, 0),
            "the same seq and index under another process's nonce must not repeat the id");
        let n = startup_nonce();
        assert_eq!(n.len(), 16, "the nonce is eight bytes as hex: {n}");
        assert!(n.chars().all(|c| c.is_ascii_hexdigit()), "the nonce is hex: {n}");
    }

    #[test] fn over_cap_streaming_requests_are_refused_at_the_call_sites() {
        // the actual handlers, not just the context_over helper: a streamed over-cap
        // request gets a clean HTTP 400 before any SSE header, on both routes. The
        // handlers write to an in-memory buffer here: no socket, no timeout.
        let tk0 = Tokenizer::for_test();
        let (jtx, _jrx) = channel::<Job>();
        let cx = Ctx { tk: Arc::new(Tokenizer::for_test()), sp: test_sp(&tk0), jobs: jtx, effort: "medium".into(),
                       max_tokens: 8, seq: Default::default(), nonce: "test".into(), key: None, ctx: 64,
                       inflight: Default::default(), limit: 4, conns: Default::default() };
        let long = "x".repeat(200);
        let mut out: Vec<u8> = Vec::new();
        messages(&cx, &json!({"model": "m", "max_tokens": 8, "stream": true,
            "messages": [{"role": "user", "content": long}]}), &mut out).unwrap();
        let a = String::from_utf8(out).unwrap();
        assert!(a.starts_with("HTTP/1.1 400"), "messages over-cap must be an HTTP 400, got: {}", &a[..a.len().min(80)]);
        assert!(a.contains("invalid_request_error") && a.contains("context full"), "messages over-cap body: {a}");
        assert!(!a.contains("text/event-stream"), "no SSE headers may precede the refusal: {a}");
        let mut out: Vec<u8> = Vec::new();
        chat(&cx, &json!({"model": "m", "max_tokens": 8, "stream": true,
            "messages": [{"role": "user", "content": long}]}), &mut out).unwrap();
        let c = String::from_utf8(out).unwrap();
        assert!(c.starts_with("HTTP/1.1 400"), "chat over-cap must be an HTTP 400, got: {}", &c[..c.len().min(80)]);
        assert!(c.contains("context full"), "chat over-cap body: {c}");
        assert!(!c.contains("text/event-stream"), "no SSE headers may precede the refusal: {c}");
    }

    #[test] fn serve_header_policy() {
        let mut head = Head {
            method: "POST".into(), path: "/v1/chat/completions".into(),
            len: MAX_BODY + 1, auth: None, api_key: None, chunked: false,
        };
        let status = |h: &Head| check_headers(h, &h.path, Some("test")).map_err(|(code, _)| code);
        assert_eq!(status(&head), Err(401), "authentication precedes length checks");
        head.auth = Some("test".into());
        assert_eq!(status(&head), Err(413));
        head.len = MAX_BODY;
        assert_eq!(status(&head), Ok(()), "the upper limit is inclusive");
        // a chunked body passes the header gate: its cap is enforced chunk
        // by chunk in read_chunked, never a 411 (Stella's proxied requests)
        head.chunked = true;
        head.len = 0;
        assert_eq!(status(&head), Ok(()), "chunked is decoded, not refused at the headers");
        let mut by_key = Head { method: "POST".into(), path: "/v1/messages".into(), len: 1, auth: None, api_key: Some("test".into()), chunked: false };
        assert_eq!(status(&by_key), Ok(()), "x-api-key authenticates as the bearer does");
        by_key.api_key = Some("wrong".into());
        assert_eq!(status(&by_key), Err(401));
    }
    #[test] fn serve_shape_refuses_an_empty_context_and_a_budget_outside_it() {
        assert!(check_serve_shape(0, 1).is_err(), "no context to serve in");
        assert!(check_serve_shape(4096, 0).is_err(), "no token budget");
        assert!(check_serve_shape(4096, 4096).is_err(), "the budget must leave room for the prompt");
        assert!(check_serve_shape(4096, 4097).is_err());
        assert!(check_serve_shape(1, 1).is_err(), "a one-token context has no budget inside it");
        assert_eq!(check_serve_shape(4096, 4095), Ok(()));
        assert_eq!(check_serve_shape(2, 1), Ok(()));
    }
    #[test] fn a_kept_prompt_supplies_its_whole_length_only_as_a_strict_prefix() {
        let p: Vec<u32> = (0..100).collect();
        let ext: Vec<u32> = (0..150).collect();
        assert_eq!(reusable_prefix(&p, true, &ext), 100, "a strict prefix with its state supplies every token");
        assert_eq!(reusable_prefix(&p, true, &p), 0, "an equal prompt supplies nothing: the suffix is never empty");
        assert_eq!(reusable_prefix(&ext, true, &p), 0, "a request shorter than the prompt supplies nothing");
        assert_eq!(reusable_prefix(&p, false, &ext), 0, "no convolution state at the prompt's end, no reuse");
        assert_eq!(reusable_prefix(&[], true, &ext), 0, "an empty prompt supplies nothing");
        assert_eq!(reusable_prefix(&p, true, &[]), 0, "an empty request takes nothing");
        // a shared head that diverges inside the prompt: the overlap is real, the state is at the wrong end
        let mut div = ext.clone(); div[60] = 9999;
        assert_eq!(shared_prefix(&p, &div), 60, "the two lists share a 60-token head");
        assert_eq!(reusable_prefix(&p, true, &div), 0, "a partial overlap supplies nothing: its state is at token 100, not 60");
        let mut at_end = ext.clone(); at_end[100] = 9999;
        assert_eq!(reusable_prefix(&p, true, &at_end), 100, "diverging at the first suffix token still reuses the whole prompt");
    }
    #[test] fn lane_choice_prefers_the_own_prompt_then_a_longer_copyable_busy_one() {
        let cand = |prefix: usize, free: bool, copyable: bool, tick: u64| LaneCand { prefix, free, copyable, tick };
        // own lane first: a free lane's own prompt beats an equally long busy one, with no copy
        let c = [cand(100, true, false, 5), cand(100, false, true, 9), cand(0, true, false, 1)];
        assert_eq!(choose_lane(&c), Some(Admission { dst: 0, src: 0, prefix: 100 }), "the own prompt wins a tie");
        // cross-lane: a strictly longer busy prompt is copied into the least recently freed lane
        let c = [cand(10, true, false, 5), cand(500, false, true, 9), cand(0, true, false, 1)];
        assert_eq!(choose_lane(&c), Some(Admission { dst: 2, src: 1, prefix: 500 }), "the longer busy prompt is the source, the stalest free lane the destination");
        // a busy lane that may not be copied (already decoding, or tiered) is excluded as a source
        let c = [cand(10, true, false, 5), cand(500, false, false, 9), cand(0, true, false, 1)];
        assert_eq!(choose_lane(&c), Some(Admission { dst: 0, src: 0, prefix: 10 }), "an uncopyable busy lane leaves the own prompt in place");
        // a busy lane is never the destination, and no free lane means no admission
        let c = [cand(500, false, true, 9), cand(0, true, false, 1)];
        assert_eq!(choose_lane(&c), Some(Admission { dst: 1, src: 0, prefix: 500 }), "the one free lane receives the copy");
        assert_eq!(choose_lane(&[cand(500, false, true, 9)]), None, "every lane busy: the request waits");
        // cold start and ties: the least recently freed lane, as before
        let c = [cand(0, true, false, 5), cand(0, true, false, 1), cand(0, false, false, 9)];
        assert_eq!(choose_lane(&c), Some(Admission { dst: 1, src: 1, prefix: 0 }), "no prefix anywhere: least recently freed");
        let c = [cand(50, true, false, 5), cand(50, true, false, 1)];
        assert_eq!(choose_lane(&c), Some(Admission { dst: 1, src: 1, prefix: 50 }), "equal own prompts: least recently freed, as before");
    }
    #[test] fn peek_results_decide_whether_the_peer_is_gone() {
        use std::io::{Error, ErrorKind};
        let script = |v: Vec<std::io::Result<usize>>| { let mut q = VecDeque::from(v); move |_: &mut [u8]| q.pop_front().unwrap() };
        assert!(gone_from_peek(script(vec![Ok(0)])), "a closed end is gone");
        assert!(!gone_from_peek(script(vec![Ok(1)])), "data waiting is present");
        assert!(!gone_from_peek(script(vec![Err(Error::from(ErrorKind::WouldBlock))])), "nothing waiting yet is present");
        assert!(gone_from_peek(script(vec![Err(Error::from(ErrorKind::Interrupted)), Err(Error::from(ErrorKind::Interrupted)), Ok(0)])), "an interrupted probe is retried, then the close is seen");
        assert!(!gone_from_peek(script(vec![Err(Error::from(ErrorKind::Interrupted)), Err(Error::from(ErrorKind::Interrupted)), Err(Error::from(ErrorKind::Interrupted)), Err(Error::from(ErrorKind::Interrupted))])), "a probe interrupted past its retries leaves the client present");
        assert!(gone_from_peek(script(vec![Err(Error::from(ErrorKind::ConnectionReset))])), "a hard error fails toward cancellation: gone");
    }

    #[test] fn verify_budget_waits_and_clamps_to_capacity() {
        let c = |lane, p0, k, kv_cap| VerifyCand { lane, p0, k, kv_cap };
        // two spans of 4, chunk 4: the second does not fit and waits
        let p = plan_verify(&[c(0, 10, 3, 100), c(1, 10, 3, 100)], 4, true);
        assert_eq!(p, vec![VerifySpanPlan { lane: 0, p0: 10, k: 3 }]);
        // without hgemm a second lane waits even when the budget remains
        let p = plan_verify(&[c(0, 1, 1, 100), c(1, 1, 1, 100)], 100, false);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].lane, 0);
        // near the cache: p0=60, cap=64, k=7 -> k=3 because 60+1+3 == 64
        let p = plan_verify(&[c(0, 60, 7, 64)], 4096, false);
        assert_eq!(p, vec![VerifySpanPlan { lane: 0, p0: 60, k: 3 }]);
        assert!(60 + 1 + p[0].k <= 64);
        // only the committed token fits: k clamps to 0, the span is t=1, the assert stays quiet
        let p = plan_verify(&[c(0, 63, 7, 64)], 4096, true);
        assert_eq!(p[0].k, 0);
        // the committed token itself is past the cache: the lane waits, it does not call encode
        assert!(plan_verify(&[c(0, 64, 1, 64)], 4096, true).is_empty());
        // a different k is not packed into a step that already chose one
        let p = plan_verify(&[c(0, 10, 3, 100), c(1, 10, 1, 100)], 100, true);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].k, 3);
    }

    #[test] fn verify_logit_arena_can_be_smaller_than_the_prefill_chunk() {
        let cands: Vec<_> = (0..64).map(|lane| VerifyCand { lane, p0: 10, k: 3, kv_cap: 64 }).collect();
        let plans = plan_verify(&cands, crate::engine::MAX_SPANS, true);
        assert_eq!(plans.len(), 32);
        assert_eq!(plans.iter().map(|p| p.k + 1).sum::<usize>(), 128);
        let waiting = plan_verify(&cands[32..], crate::engine::MAX_SPANS, true);
        assert_eq!(waiting.first().unwrap().lane, 32);
        assert_eq!(waiting.last().unwrap().lane, 63);
        assert_eq!(plan_verify(&cands, 2, true), vec![VerifySpanPlan { lane: 0, p0: 10, k: 1 }]);
    }

    #[test] fn final_verify_block_emits_the_full_reply_budget() {
        for (max, ctx, stop, reason) in [(2, 64, vec![], "length"), (10, 63, vec![], "length"), (2, 64, vec![8], "stop")] {
            let (out, events) = channel();
            let job = Job { toks: vec![0; 60], max, temp: 0.0, top_p: 1.0, seed: 0, out };
            let mut a = Active {
                job, n_past: 62, pf_end: 60, cached: 0, gen: 0,
                sampler: Sampler::new(0.0, 64, 1.0, 0.0, 0),
                draft_sampler: Sampler::draft(0.0, 64, 1.0, 0.0, 0),
                pending: Some(8), drafts: None, cs_before: None, emit_q: VecDeque::new(),
            };
            assert_eq!(reply_budget(a.pf_end, a.gen, a.job.max, ctx), 2);
            assert!(!a.emit_token(7, &stop, ctx));
            assert!(a.emit_token(8, &stop, ctx));
            assert!(matches!(events.try_recv(), Ok(Ev::Tok(7))));
            assert!(matches!(events.try_recv(), Ok(Ev::Tok(8))));
            assert!(matches!(events.try_recv(), Ok(Ev::Done { gen: 2, reason: r, .. }) if r == reason));
            assert!(events.try_recv().is_err());
        }
    }

    #[test] fn verify_stop_in_the_middle_emits_only_the_prefix() {
        let drafts = [1u32, 7, 2];
        let rows = [1u32, 7, 2, 9];
        let g = accept_greedy(&drafts, &rows, &[7]);
        assert_eq!(g.emitted, vec![1, 7]);
        assert!(g.finished);
        assert_eq!(g.bonus, None, "drafts after the stop are not emitted, and there is no bonus");
        assert_eq!(g.n_keep, 3);
        assert!(!g.emitted.contains(&2) && !g.emitted.contains(&9));
    }

    #[test] fn verify_reject_keeps_the_target_token_and_a_full_accept_emits_the_bonus() {
        let g = accept_greedy(&[1, 99, 3], &[1, 2, 3, 4], &[]);
        assert_eq!(g.emitted, vec![1, 2], "the mismatch keeps the target token and stops");
        assert_eq!(g.bonus, Some(2));
        assert!(!g.finished);
        assert_eq!(g.n_keep, 2);
        let g = accept_greedy(&[1, 2, 3], &[1, 2, 3, 8], &[]);
        assert_eq!(g.emitted, vec![1, 2, 3, 8]);
        assert_eq!(g.bonus, Some(8));
        assert_eq!(g.n_keep, 4);
        assert_eq!(g.rows, vec![0, 1, 2, 3]);
        // the bonus itself is the stop: it is emitted, and the lane finishes
        let g = accept_greedy(&[1, 2], &[1, 2, 7], &[7]);
        assert_eq!(g.emitted, vec![1, 2, 7]);
        assert!(g.finished);
        assert_eq!(g.bonus, Some(7));
    }
}
