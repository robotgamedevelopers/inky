//! `inkling serve`: an OpenAI-compatible /v1/chat/completions endpoint on loopback, one lane.
//! Renders the GGUF chat template (tools, tool calls, tool results), keeps the K/V cache across requests when the new
//! prompt extends the old context, and turns the model's tool-call messages into OpenAI tool_calls.

use crate::engine::Engine;
use crate::sampler::Sampler;
use crate::session::Session;
use crate::tokenizer::Tokenizer;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Instant;

pub struct ServeOpts { pub model: String, pub ctx: usize, pub port: u16, pub effort: String, pub max_tokens: usize, pub log: bool }

struct Specials { msg_user: u32, msg_model: u32, msg_system: u32, msg_tool: u32, c_text: u32, c_think: u32, c_xml: u32,
                  c_tool: u32, end: u32, end_sampling: u32 }

fn special(tk: &Tokenizer, s: &str) -> u32 {
    let v = tk.encode(s, true);
    assert!(v.len() == 1, "{s} is not one token");
    v[0]
}

/// What a generated assistant message parsed to, and the canonical key the client will send back.
struct Parsed { text: String, thinking: String, calls: Vec<(String, Value)> }

fn canon_key(content: &str, calls: &[(String, Value)]) -> String {
    let c: Vec<Value> = calls.iter().map(|(n, a)| json!([n, a])).collect();
    json!([content.trim(), c]).to_string()
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

/// The template's tojson(sort_keys=true, separators=(",", ":")): serde_json's default map is sorted.
fn compact(v: &Value) -> String { v.to_string() }

struct Server<'a> {
    eng: &'a Engine, tk: &'a Tokenizer, sp: Specials, o: ServeOpts,
    sess: Session<'a>,
    /// every token logically in the context (the session's pending token included)
    ctx: Vec<u32>,
    /// generated assistant messages by canonical key: their raw tokens (thinking included), so a resent history matches
    raw: HashMap<String, Vec<u32>>,
    seq: u64,
}

impl<'a> Server<'a> {
    fn text(&self, out: &mut Vec<u32>, s: &str) { out.extend(self.tk.encode(s, false)); }

    fn render(&self, req: &Value) -> Result<Vec<u32>, String> {
        let sp = &self.sp;
        let mut t = Vec::new();
        let msgs = req["messages"].as_array().ok_or("messages missing")?;
        if let Some(tools) = req["tools"].as_array().filter(|a| !a.is_empty()) {
            let specs: Vec<Value> = tools.iter().map(|tool| {
                let f = tool.get("function").unwrap_or(tool);
                json!({"description": f.get("description").and_then(|d| d.as_str()).unwrap_or(""),
                       "name": f["name"],
                       "parameters": f.get("parameters").cloned().unwrap_or(json!({})),
                       "type": tool.get("type").and_then(|d| d.as_str()).unwrap_or("function")})
            }).collect();
            t.push(sp.msg_system); self.text(&mut t, "tool_declare"); t.push(sp.c_xml);
            self.text(&mut t, &compact(&Value::Array(specs))); t.push(sp.end);
        }
        let effort = req.get("reasoning_effort").and_then(|v| v.as_str()).unwrap_or(&self.o.effort).to_string();
        let mut effort_done = false;
        let emit_effort = |t: &mut Vec<u32>| {
            t.push(sp.msg_system); t.push(sp.c_text);
            t.extend(self.tk.encode(&format!("Thinking effort level: {}", effort_num(&effort)), false)); t.push(sp.end);
        };
        // tool_call_id -> function name, for tool result headers
        let mut call_names: HashMap<String, String> = HashMap::new();
        for m in msgs {
            if let Some(tcs) = m["tool_calls"].as_array() {
                for tc in tcs { if let (Some(id), Some(n)) = (tc["id"].as_str(), tc["function"]["name"].as_str()) { call_names.insert(id.into(), n.into()); } }
            }
        }
        for m in msgs {
            let role = m["role"].as_str().ok_or("message without role")?;
            let role = if role == "developer" { "system" } else { role };
            if !effort_done && role != "system" { emit_effort(&mut t); effort_done = true; }
            match role {
                "system" | "user" => {
                    let h = if role == "system" { sp.msg_system } else { sp.msg_user };
                    for p in content_parts(&m["content"]) { t.push(h); t.push(sp.c_text); self.text(&mut t, &p); t.push(sp.end); }
                }
                "tool" => {
                    let name = m["name"].as_str().map(|s| s.to_string())
                        .or_else(|| m["tool_call_id"].as_str().and_then(|id| call_names.get(id).cloned())).unwrap_or_default();
                    for p in content_parts(&m["content"]) {
                        t.push(sp.msg_tool); self.text(&mut t, &name); t.push(sp.c_text); self.text(&mut t, &p); t.push(sp.end);
                    }
                }
                "assistant" => {
                    let content = content_parts(&m["content"]).join("");
                    let mut calls = Vec::new();
                    for tc in m["tool_calls"].as_array().into_iter().flatten() {
                        let name = tc["function"]["name"].as_str().ok_or("tool call without a name")?.to_string();
                        let args = match &tc["function"]["arguments"] {
                            Value::String(s) if s.trim().is_empty() => json!({}),
                            Value::String(s) => serde_json::from_str(s).map_err(|e| format!("tool call arguments: {e}"))?,
                            Value::Null => json!({}),
                            v => v.clone(),
                        };
                        calls.push((name, args));
                    }
                    if let Some(raw) = self.raw.get(&canon_key(&content, &calls)) { t.extend_from_slice(raw); continue; }
                    let has_text = !content.is_empty() || calls.is_empty();
                    if has_text { t.push(sp.msg_model); t.push(sp.c_text); self.text(&mut t, &content); t.push(sp.end); }
                    for (name, args) in &calls {
                        t.push(sp.msg_model); self.text(&mut t, name); t.push(sp.c_tool);
                        // the template writes args' keys sorted, values compact
                        self.text(&mut t, &format!("{{\"name\":{},\"args\":{}}}", Value::String(name.clone()), compact(args)));
                        t.push(sp.end);
                    }
                    t.push(sp.end_sampling);
                }
                r => return Err(format!("unknown role {r}")),
            }
        }
        if !effort_done { emit_effort(&mut t); }
        Ok(t)
    }

    /// Parse generated tokens into thinking, text and tool calls.
    fn parse(&self, toks: &[u32]) -> Parsed {
        let sp = &self.sp;
        #[derive(PartialEq)] enum M { Header, Think, Text, Tool, None }
        let (mut mode, mut cur, mut header) = (M::None, Vec::<u8>::new(), Vec::<u8>::new());
        let mut p = Parsed { text: String::new(), thinking: String::new(), calls: vec![] };
        for &t in toks {
            if t == sp.msg_model { mode = M::Header; header.clear(); cur.clear(); continue; }
            if t == sp.c_think { mode = M::Think; cur.clear(); continue; }
            if t == sp.c_text { mode = M::Text; cur.clear(); continue; }
            if t == sp.c_tool { mode = M::Tool; cur.clear(); continue; }
            if t == sp.end || t == sp.end_sampling || t == self.tk.eos {
                let s = String::from_utf8_lossy(&cur).to_string();
                match mode {
                    M::Think => p.thinking += &s,
                    M::Text => p.text += &s,
                    M::Tool => {
                        let v: Value = serde_json::from_str(&s).unwrap_or(Value::Null);
                        let name = v["name"].as_str().map(|s| s.to_string()).unwrap_or_else(|| String::from_utf8_lossy(&header).to_string());
                        let args = if v["args"].is_object() { v["args"].clone() } else { json!({"_unparsed": s}) };
                        p.calls.push((name, args));
                    }
                    _ => {}
                }
                mode = M::None; cur.clear();
                continue;
            }
            let b = self.tk.decode_bytes(t);
            match mode { M::Header => header.extend(b), M::None => {}, _ => cur.extend(b) }
        }
        // a message cut by max_tokens
        let s = String::from_utf8_lossy(&cur).to_string();
        match mode { M::Text => p.text += &s, M::Think => p.thinking += &s, _ => {} }
        p
    }

    fn handle(&mut self, req: &Value, stream: &mut TcpStream) -> std::io::Result<()> {
        let toks = match self.render(req) {
            Ok(t) => t,
            Err(e) => return respond(stream, 400, &json!({"error": {"message": e}}).to_string()),
        };
        let t0 = Instant::now();
        let reuse = toks.len() > self.ctx.len() && toks.starts_with(&self.ctx);
        if !reuse { self.sess = Session::new(self.eng, self.tk); self.ctx.clear(); }
        let fresh = toks.len() - self.ctx.len();
        if toks.len() + 64 >= self.eng.max_ctx {
            return respond(stream, 400, &json!({"error": {"message": format!("context full: {} of {} tokens", toks.len(), self.eng.max_ctx)}}).to_string());
        }
        self.eng.prepare(toks.len(), fresh);
        let first = self.sess.prefill(&toks[self.ctx.len()..]);
        let prefill_s = t0.elapsed().as_secs_f64();
        self.ctx = toks;
        let max = req["max_tokens"].as_u64().or(req["max_completion_tokens"].as_u64()).map(|v| v as usize).unwrap_or(self.o.max_tokens)
            .min(self.sess.ctx_left().saturating_sub(4));
        let temp = req["temperature"].as_f64().unwrap_or(1.0) as f32;
        let top_p = req["top_p"].as_f64().unwrap_or(0.95) as f32;
        let mut sampler = Sampler::new(temp, 64, top_p, self.eng.softcap, 0x5eed ^ self.seq);
        let want_stream = req["stream"].as_bool().unwrap_or(false);
        self.seq += 1;
        let id = format!("chatcmpl-{}", self.seq);
        let model = req["model"].as_str().unwrap_or("inkling").to_string();
        if want_stream {
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
            let first_chunk = json!({"id": id, "object": "chat.completion.chunk", "model": model,
                                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]});
            write!(stream, "data: {first_chunk}\n\n")?;
        }
        // generate, streaming thinking and text deltas as they complete into valid UTF-8
        let sp_text = self.sp.c_text; let sp_think = self.sp.c_think; let sp_tool = self.sp.c_tool; let sp_model = self.sp.msg_model;
        let sp_end = self.sp.end; let sp_endsam = self.sp.end_sampling;
        let mut gen: Vec<u32> = Vec::new();
        let mut mode = 0u8; // 0 none/header, 1 thinking, 2 text, 3 tool json
        let mut pend: Vec<u8> = Vec::new();
        let mut io_err: Option<std::io::Error> = None;
        let tk = self.tk;
        let (eng_ref, sess) = (self.eng, &mut self.sess);
        let _ = eng_ref;
        let tg = Instant::now();
        let n = sess.generate(first, max, &mut sampler, |t| {
            gen.push(t);
            if t == sp_endsam { return false; }
            if t == sp_think { mode = 1; return true; }
            if t == sp_text { mode = 2; return true; }
            if t == sp_tool { mode = 3; return true; }
            if t == sp_model || t == sp_end { mode = 0; pend.clear(); return true; }
            if want_stream && (mode == 1 || mode == 2) && io_err.is_none() {
                pend.extend(tk.decode_bytes(t));
                let valid = match std::str::from_utf8(&pend) { Ok(_) => pend.len(), Err(e) => e.valid_up_to() };
                if valid > 0 {
                    let s = String::from_utf8_lossy(&pend[..valid]).to_string();
                    pend.drain(..valid);
                    let key = if mode == 1 { "reasoning_content" } else { "content" };
                    let c = json!({"id": id, "object": "chat.completion.chunk", "model": model,
                                   "choices": [{"index": 0, "delta": {key: s}, "finish_reason": null}]});
                    if let Err(e) = write!(stream, "data: {c}\n\n").and_then(|_| stream.flush()) { io_err = Some(e); }
                }
            }
            true
        });
        if n > gen.len() { gen.push(self.tk.eos); }  // stopped on eos: generate does not pass it to the callback
        let gen_s = tg.elapsed().as_secs_f64();
        self.ctx.extend_from_slice(&gen);
        let p = self.parse(&gen);
        let finish = if !p.calls.is_empty() { "tool_calls" } else if gen.last() == Some(&sp_endsam) || gen.last() == Some(&self.tk.eos) { "stop" } else { "length" };
        self.raw.insert(canon_key(&p.text, &p.calls), gen.clone());
        let calls: Vec<Value> = p.calls.iter().enumerate().map(|(i, (name, args))| json!({
            "index": i, "id": format!("call_{}_{}", self.seq, i), "type": "function",
            "function": {"name": name, "arguments": args.to_string()}})).collect();
        let usage = json!({"prompt_tokens": self.ctx.len() - gen.len(), "completion_tokens": gen.len(),
                           "total_tokens": self.ctx.len(), "prompt_tokens_details": {"cached_tokens": self.ctx.len() - gen.len() - fresh}});
        if self.o.log {
            eprintln!("[serve] #{} ctx {} fresh {} ({:.2}s{}) gen {} ({:.1} tok/s) finish {} calls [{}] text {:?}",
                      self.seq, self.ctx.len(), fresh, prefill_s, if reuse { ", cache hit" } else { "" }, gen.len(), gen.len() as f64 / gen_s, finish,
                      p.calls.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(","), p.text.chars().take(120).collect::<String>());
        }
        if let Some(e) = io_err { return Err(e); }
        if want_stream {
            if !calls.is_empty() {
                let c = json!({"id": id, "object": "chat.completion.chunk", "model": model,
                               "choices": [{"index": 0, "delta": {"tool_calls": calls}, "finish_reason": null}]});
                write!(stream, "data: {c}\n\n")?;
            }
            let c = json!({"id": id, "object": "chat.completion.chunk", "model": model,
                           "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage});
            write!(stream, "data: {c}\n\ndata: [DONE]\n\n")?;
            stream.flush()
        } else {
            let mut msg = json!({"role": "assistant", "content": p.text});
            if !p.thinking.is_empty() { msg["reasoning_content"] = json!(p.thinking); }
            if !calls.is_empty() { msg["tool_calls"] = json!(calls); }
            respond(stream, 200, &json!({"id": id, "object": "chat.completion", "model": model,
                                         "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage}).to_string())
        }
    }
}

fn respond(s: &mut TcpStream, code: u16, body: &str) -> std::io::Result<()> {
    write!(s, "HTTP/1.1 {code} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
           if code == 200 { "OK" } else { "Error" }, body.len())?;
    s.flush()
}

fn read_request(s: &TcpStream) -> std::io::Result<(String, String, Vec<u8>)> {
    let mut r = BufReader::new(s);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let mut it = line.split_whitespace();
    let (method, path) = (it.next().unwrap_or("").to_string(), it.next().unwrap_or("").to_string());
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" { break; }
        if let Some((k, v)) = h.split_once(':') { if k.eq_ignore_ascii_case("content-length") { len = v.trim().parse().unwrap_or(0); } }
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body)?;
    Ok((method, path, body))
}

pub fn serve(o: ServeOpts) {
    let eng = Engine::load(&o.model, o.ctx);
    let tk = Tokenizer::load(&eng.g);
    let sp = Specials {
        msg_user: special(&tk, "<|message_user|>"), msg_model: special(&tk, "<|message_model|>"), msg_system: special(&tk, "<|message_system|>"),
        msg_tool: special(&tk, "<|message_tool|>"), c_text: special(&tk, "<|content_text|>"), c_think: special(&tk, "<|content_thinking|>"),
        c_xml: special(&tk, "<|content_xml|>"), c_tool: special(&tk, "<|content_invoke_tool_json|>"), end: special(&tk, "<|end_message|>"),
        end_sampling: special(&tk, "<|content_model_end_sampling|>"),
    };
    let _ = sp.msg_user;
    eprintln!("[serve] eos is {:?}", String::from_utf8_lossy(&tk.decode_bytes(tk.eos)));
    let addr = format!("127.0.0.1:{}", o.port);
    let l = TcpListener::bind(&addr).expect("bind");
    eprintln!("[serve] listening on http://{addr}/v1 (one lane, context {})", o.ctx);
    let sess = Session::new(&eng, &tk);
    let mut srv = Server { eng: &eng, tk: &tk, sp, o, sess, ctx: vec![], raw: HashMap::new(), seq: 0 };
    for c in l.incoming() {
        let Ok(mut s) = c else { continue };
        let (method, path, body) = match read_request(&s) { Ok(x) => x, Err(_) => continue };
        let r = match (method.as_str(), path.trim_end_matches('/')) {
            ("GET", "/v1/models") => respond(&mut s, 200, &json!({"object": "list", "data": [{"id": "inkling", "object": "model", "owned_by": "local"}]}).to_string()),
            ("POST", "/v1/chat/completions") => match serde_json::from_slice::<Value>(&body) {
                Ok(v) => {
                    if srv.o.log { if let Ok(d) = std::env::var("INK_SERVE_DUMP") { let _ = std::fs::write(format!("{d}/req-{}.json", srv.seq + 1), &body); } }
                    srv.handle(&v, &mut s)
                }
                Err(e) => respond(&mut s, 400, &json!({"error": {"message": e.to_string()}}).to_string()),
            },
            _ => respond(&mut s, 404, &json!({"error": {"message": format!("no route {method} {path}")}}).to_string()),
        };
        if let Err(e) = r { eprintln!("[serve] client: {e}"); }
    }
}
