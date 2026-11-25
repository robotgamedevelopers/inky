//! Translate Anthropic Messages requests into the shared prompt format.

use serde_json::{json, Value};

// ------------------------------------------------------------------------------------------------- Anthropic Messages
// `POST /v1/messages` speaks the Anthropic Messages API over the same lanes. A request is translated
// to the OpenAI shape above and rendered by the same `render`, so an Anthropic conversation and the
// equivalent chat conversation are the same token stream and a lane's kept prompt is reused across
// either API (decision 4). Thinking has no block in either direction here: a reply's thinking is
// dropped, and thinking blocks in a sent history are dropped with it.

/// An Anthropic request as the OpenAI request the chat path renders: `system` becomes leading
/// system messages, a `system` role inside `messages[]` (Claude Code 2.1.330) becomes a system
/// message in place so its order is kept, text blocks keep their order and their per-block
/// rendering, a user message's tool results become tool messages in place, and an assistant
/// message's tool uses and text retain their order. Chat renders text before tool calls, so a text
/// block after a tool starts a new group in the same turn. The caller suppresses the intermediate
/// end-of-sampling markers at the returned (ascending) group indices.
pub(super) fn translate_messages(req: &Value) -> Result<(Value, Vec<usize>), String> {
    let msgs_in = req["messages"].as_array().ok_or("messages missing")?;
    if msgs_in.is_empty() { return Err("messages must not be empty".into()); }
    let mut msgs: Vec<Value> = Vec::new();
    let mut continued_assistants = Vec::new();
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
                let mut texts = Vec::new();
                let mut calls: Vec<Value> = Vec::new();
                for b in blocks {
                    match b["type"].as_str() {
                        Some("text") => {
                            if !calls.is_empty() {
                                continued_assistants.push(msgs.len());
                                msgs.push(json!({"role": "assistant", "content": std::mem::take(&mut texts), "tool_calls": std::mem::take(&mut calls)}));
                            }
                            texts.push(b["text"].as_str().unwrap_or(""));
                        }
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
                let mut msg = json!({"role": "assistant", "content": texts});
                if !calls.is_empty() { msg["tool_calls"] = json!(calls); }
                msgs.push(msg);
            }
            ("user" | "assistant" | "system", _) => return Err("message content must be a string or an array of blocks".into()),
            (r, _) => return Err(format!("unknown role {r}")),
        }
    }
    let mut out = json!({"messages": msgs});
    if let Some(e) = anthropic_effort(req)? { out["reasoning_effort"] = json!(e); }
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
    Ok((out, continued_assistants))
}

/// The thinking effort a Messages request asks for, as the `reasoning_effort`
/// the chat path renders through `effort_num` — or none, when the request
/// leaves the serve default standing. An explicit effort field wins (top
/// level, `output_config`, or inside `thinking`); then `thinking` itself:
/// disabled is none, an enabled budget maps by size (1024–4095 low,
/// 4096–16383 medium, else high), and adaptive — or no thinking at all —
/// decides for itself, so it sets nothing. Anything malformed refuses
/// (the caller answers 400): no silent fallback may turn an unknown
/// effort into 0.9 or drop a budget on the floor.
fn anthropic_effort(req: &Value) -> Result<Option<String>, String> {
    // an effort field that is present must be a known level (or a number in
    // effort_num's range): an unknown string would otherwise reach effort_num
    // and silently become 0.9, and a non-string would silently vanish
    let field = |v: &Value| -> Result<String, String> {
        let s = v.as_str().ok_or("effort must be a string")?.trim();
        let known = matches!(s, "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max")
            || s.parse::<f64>().is_ok_and(|n| n.is_finite() && (0.0..=0.99).contains(&n));
        if known { Ok(s.to_string()) } else { Err(format!("unknown effort level {s:?}")) }
    };
    if let Some(v) = req.get("output_config").and_then(|o| o.get("effort")).filter(|v| !v.is_null()) { return Ok(Some(field(v)?)); }
    if let Some(v) = req.get("effort").filter(|v| !v.is_null()) { return Ok(Some(field(v)?)); }
    let Some(th) = req.get("thinking").filter(|v| !v.is_null()) else { return Ok(None) };
    if let Some(v) = th.get("effort").filter(|v| !v.is_null()) { return Ok(Some(field(v)?)); }
    match th["type"].as_str() {
        Some("disabled") => Ok(Some("none".into())),
        Some("adaptive") => Ok(None),
        Some("enabled") => {
            let b = th.get("budget_tokens").filter(|v| !v.is_null()).ok_or("thinking budget_tokens is required when thinking is enabled")?;
            let b = b.as_u64().ok_or("thinking budget_tokens must be a non-negative integer")?;
            match b {
                0 => Ok(Some("none".into())),
                1..=1023 => Err(format!("thinking budget_tokens {b} is under Anthropic's 1024 minimum")),
                1024..=4095 => Ok(Some("low".into())),
                4096..=16383 => Ok(Some("medium".into())),
                _ => Ok(Some("high".into())),
            }
        }
        t => Err(format!("unknown thinking type {}", t.unwrap_or("none"))),
    }
}
