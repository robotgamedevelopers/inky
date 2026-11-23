//! `inky serve`: an OpenAI-compatible chat completions endpoint over many lanes. One scheduler thread owns the engine and
//! the lanes and runs one GPU step at a time: a prefill step (lanes with prompt left, up to one chunk of tokens) or a
//! decode step (one token for every lane that is generating). Connection threads parse requests, render the chat
//! template, and stream back what the scheduler sends them. A lane keeps its last prompt and its convolution state at the
//! prompt's end; a request whose tokens extend that prompt continues the lane from there, so a card's next turn prefills
//! only what is new. Only prefill-written cache is ever reused (docs/DESIGN.md, decisions 4 and 5).

use crate::engine::{Engine, Lane, Span};
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;
use objc2_metal::MTLCommandBuffer;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::time::Instant;

pub struct ServeOpts { pub model: String, pub ctx: usize, pub bind: String, pub lanes: usize, pub effort: String, pub max_tokens: usize }

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
                for p in content_parts(&m["content"]) { t.push(sp.msg_tool); text(&mut t, &name); t.push(sp.c_text); text(&mut t, &p); t.push(sp.end); }
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

// ------------------------------------------------------------------------------------------------- scheduler

struct Job { toks: Vec<u32>, max: usize, temp: f32, top_p: f32, seed: u64, out: Sender<Ev> }
enum Ev { Tok(u32), Done { prompt: usize, cached: usize, gen: usize, reason: &'static str }, Err(String) }

/// A lane: its stream state on the GPU, the last prompt it prefilled and its convolution state at that prompt's end.
struct LaneS { lane: Lane, prompt: Vec<u32>, cs_at_prompt: Option<Vec<u8>>, used: Instant, job: Option<Active> }
struct Active { job: Job, n_past: usize, pf_end: usize, cached: usize, gen: usize, sampler: Sampler, pending: Option<u32> }

fn argmax(v: &[f32]) -> u32 { v.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32 }

fn scheduler(eng: &Engine, tk: &Tokenizer, sp: Specials, nlanes: usize, rx: Receiver<Job>) {
    let mut lanes: Vec<LaneS> = (0..nlanes).map(|_| LaneS { lane: Lane::new(eng), prompt: vec![], cs_at_prompt: None, used: Instant::now(), job: None }).collect();
    let mut queue: VecDeque<Job> = VecDeque::new();
    let stop = [tk.eos, sp.end_sampling];
    let mut last_was_prefill = false;
    loop {
        // take new requests: block only when there is nothing to do
        let busy = lanes.iter().any(|l| l.job.is_some());
        if !busy && queue.is_empty() { match rx.recv() { Ok(j) => queue.push_back(j), Err(_) => return } }
        loop { match rx.try_recv() { Ok(j) => queue.push_back(j), Err(TryRecvError::Empty) => break, Err(TryRecvError::Disconnected) => return } }
        // admit: a free lane whose last prompt is the longest prefix of the request, else the least recently used free lane
        while let Some(j) = queue.front() {
            let best = lanes.iter().enumerate().filter(|(_, l)| l.job.is_none())
                .max_by_key(|(_, l)| (if l.cs_at_prompt.is_some() && l.prompt.len() < j.toks.len() && j.toks.starts_with(&l.prompt) { l.prompt.len() } else { 0 }, std::cmp::Reverse(l.used)));
            let Some((i, _)) = best else { break };
            let j = queue.pop_front().unwrap();
            if j.toks.len() + j.max + 2 > eng.max_ctx { let _ = j.out.send(Ev::Err(format!("context full: {} prompt tokens, {} max", j.toks.len(), eng.max_ctx))); continue; }
            let l = &mut lanes[i];
            let run = eng.on(&l.lane);
            let reuse = l.cs_at_prompt.is_some() && l.prompt.len() < j.toks.len() && j.toks.starts_with(&l.prompt);
            let start = if reuse { l.prompt.len() } else { 0 };
            if reuse { run.set_cs(l.cs_at_prompt.as_ref().unwrap()); } else { run.reset(); }
            run.reserve(j.toks.len() + j.max + 2, true);
            run.set_tokens(0, &j.toks);
            let sampler = Sampler::new(j.temp, 64, j.top_p, eng.softcap, j.seed);
            l.prompt.clear(); l.cs_at_prompt = None;
            l.job = Some(Active { pf_end: j.toks.len(), n_past: start, cached: start, gen: 0, sampler, pending: None, job: j });
        }
        // a step: prefill when a lane has prompt left (alternating with decode while others generate), else decode
        let need_pf: Vec<usize> = (0..nlanes).filter(|&i| lanes[i].job.as_ref().is_some_and(|a| a.n_past < a.pf_end)).collect();
        let need_dec: Vec<usize> = (0..nlanes).filter(|&i| lanes[i].job.as_ref().is_some_and(|a| a.n_past >= a.pf_end && a.pending.is_some())).collect();
        if !need_pf.is_empty() && (need_dec.is_empty() || !last_was_prefill) {
            last_was_prefill = true;
            let mut budget = eng.chunk;
            let mut sp_list: Vec<(usize, usize, usize)> = Vec::new();      // (lane, p0, t)
            for &i in &need_pf {
                if budget == 0 { break; }
                let a = lanes[i].job.as_ref().unwrap();
                let t = (a.pf_end - a.n_past).min(budget);
                // end chunks on 64-position boundaries unless they end the prompt (start points, docs/DESIGN.md)
                let t = if a.n_past + t < a.pf_end && t > 64 { t - (a.n_past + t) % 64 } else { t };
                sp_list.push((i, a.n_past, t)); budget -= t;
            }
            let spans: Vec<Span> = sp_list.iter().map(|&(i, p0, t)| Span { lane: &lanes[i].lane, p0, t }).collect();
            let cb = eng.gpu.cmd(); eng.encode_spans(&cb, &spans, true, false); cb.commit(); cb.waitUntilCompleted();
            let many = spans.len() > 1;
            for (k, &(i, p0, t)) in sp_list.iter().enumerate() {
                let l = &mut lanes[i];
                let a = l.job.as_mut().unwrap();
                a.n_past = p0 + t;
                if a.n_past == a.pf_end {
                    // the prompt is in: keep its state for the next turn, then sample the first token
                    l.cs_at_prompt = Some(eng.on(&l.lane).get_cs());
                    l.prompt = a.job.toks.clone();
                    let lg = if many { eng.span_logits(k) } else { eng.logits() };
                    let tok = if a.sampler.greedy() { argmax(lg) } else { a.sampler.sample(lg) };
                    a.pending = Some(tok);
                }
            }
        } else if !need_dec.is_empty() {
            last_was_prefill = false;
            // feed each lane's pending token at its position
            for &i in &need_dec { let a = lanes[i].job.as_ref().unwrap(); eng.on(&lanes[i].lane).set_tokens(a.n_past, &[a.pending.unwrap()]); }
            let spans: Vec<Span> = need_dec.iter().map(|&i| Span { lane: &lanes[i].lane, p0: lanes[i].job.as_ref().unwrap().n_past, t: 1 }).collect();
            let cb = eng.gpu.cmd(); eng.encode_decode(&cb, &spans, true, false); cb.commit(); cb.waitUntilCompleted();
            for (k, &i) in need_dec.iter().enumerate() {
                let a = lanes[i].job.as_mut().unwrap();
                a.n_past += 1;
                let lg = eng.span_logits(k);
                a.pending = Some(if a.sampler.greedy() { argmax(lg) } else { a.sampler.sample(lg) });
            }
        }
        // emit the tokens just sampled; finish lanes that stopped or whose client went away
        for l in lanes.iter_mut() {
            let Some(a) = l.job.as_mut() else { continue };
            let Some(tok) = a.pending else { continue };
            if a.n_past < a.pf_end { continue; }
            // a token is emitted once, right after it is sampled: then n_past - pf_end == gen
            let fresh = a.n_past == a.pf_end + a.gen;
            if !fresh { continue; }
            let gone = a.job.out.send(Ev::Tok(tok)).is_err();
            a.gen += 1;
            let reason = if gone { Some("cancelled") } else if stop.contains(&tok) { Some("stop") }
                         else if a.gen >= a.job.max || a.n_past + 2 >= eng.max_ctx { Some("length") } else { None };
            if let Some(r) = reason {
                let _ = a.job.out.send(Ev::Done { prompt: a.pf_end, cached: a.cached, gen: a.gen, reason: r });
                l.job = None; l.used = Instant::now();
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------- HTTP

fn respond(s: &mut TcpStream, code: u16, body: &str) -> std::io::Result<()> {
    write!(s, "HTTP/1.1 {code} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", if code == 200 { "OK" } else { "Error" }, body.len())?;
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

struct Ctx { tk: Arc<Tokenizer>, sp: Specials, jobs: Sender<Job>, effort: String, max_tokens: usize, seq: std::sync::atomic::AtomicU64 }

fn chat(cx: &Ctx, req: &Value, s: &mut TcpStream) -> std::io::Result<()> {
    let toks = match render(&cx.tk, &cx.sp, req, &cx.effort) { Ok(t) => t, Err(e) => return respond(s, 400, &json!({"error": {"message": e}}).to_string()) };
    let seq = cx.seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    let (tx, rx) = channel();
    let max = req["max_tokens"].as_u64().or(req["max_completion_tokens"].as_u64()).map(|v| v as usize).unwrap_or(cx.max_tokens);
    // the seed is the request's own: its `seed`, or a hash of the request
    let seed = req["seed"].as_u64().unwrap_or_else(|| req.to_string().bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3)));
    let job = Job { toks, max, temp: req["temperature"].as_f64().unwrap_or(1.0) as f32, top_p: req["top_p"].as_f64().unwrap_or(0.95) as f32, seed, out: tx };
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
        match rx.recv() {
            Ok(Ev::Tok(t)) => {
                gen.push(t);
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
            Err(_) => return respond(s, 500, &json!({"error": {"message": "the request was dropped"}}).to_string()),
        }
    }
}

pub fn serve(o: ServeOpts) {
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
    let cx = Arc::new(Ctx { tk: tk.clone(), sp, jobs: jtx, effort: o.effort.clone(), max_tokens: o.max_tokens, seq: Default::default() });
    let l = TcpListener::bind(&o.bind).expect("bind");
    eprintln!("[serve] http://{}/v1: {} lanes, context {} per lane, effort {}", o.bind, o.lanes, o.ctx, o.effort);
    std::thread::spawn(move || {
        for c in l.incoming() {
            let Ok(mut s) = c else { continue };
            let cx = cx.clone();
            std::thread::spawn(move || {
                let Ok((method, path, body)) = read_request(&s) else { return };
                let r = match (method.as_str(), path.trim_end_matches('/')) {
                    ("GET", "/v1/models") => respond(&mut s, 200, &json!({"object": "list", "data": [{"id": "inkling-2-mini", "object": "model", "owned_by": "inky"}]}).to_string()),
                    ("GET", "/health") => respond(&mut s, 200, &json!({"ok": true}).to_string()),
                    ("POST", "/v1/chat/completions") => match serde_json::from_slice::<Value>(&body) {
                        Ok(v) => chat(&cx, &v, &mut s),
                        Err(e) => respond(&mut s, 400, &json!({"error": {"message": e.to_string()}}).to_string()),
                    },
                    _ => respond(&mut s, 404, &json!({"error": {"message": format!("no route {method} {path}")}}).to_string()),
                };
                if let Err(e) = r { eprintln!("[serve] client: {e}"); }
            });
        }
    });
    scheduler(&eng, &tk, sp, o.lanes, jrx);
}
