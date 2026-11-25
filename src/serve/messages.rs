//! Translate Anthropic Messages requests into the shared prompt format.

use super::validate_effort;
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
    if !req.is_object() {
        return Err("request body must be a JSON object".into());
    }
    let msgs_in = req.get("messages").and_then(Value::as_array).ok_or("messages missing")?;
    if msgs_in.is_empty() { return Err("messages must not be empty".into()); }
    let mut msgs: Vec<Value> = Vec::new();
    let mut continued_assistants = Vec::new();
    match req.get("system") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => msgs.push(json!({"role": "system", "content": s})),
        Some(Value::Array(blocks)) => {
            let parts: Vec<Value> = blocks.iter().map(|b| match b.get("type").and_then(Value::as_str) {
                Some("text") => {
                    let txt = b.get("text").and_then(Value::as_str).ok_or("system text block requires a string text field")?;
                    Ok(json!({"type": "text", "text": txt}))
                }
                t => Err(format!("unsupported system block type {}", t.unwrap_or("none"))),
            }).collect::<Result<_, String>>()?;
            if !parts.is_empty() { msgs.push(json!({"role": "system", "content": parts})); }
        }
        Some(_) => return Err("system must be a string or an array of text blocks".into()),
    }
    for m in msgs_in {
        let m_obj = m.as_object().ok_or("each message must be an object")?;
        let role = m_obj.get("role").and_then(Value::as_str).ok_or("message without role")?;
        let content = m_obj.get("content").unwrap_or(&Value::Null);
        match (role, content) {
            ("system", Value::String(s)) => msgs.push(json!({"role": "system", "content": s})),
            ("system", Value::Array(blocks)) => {
                if blocks.is_empty() {
                    return Err("message content blocks must not be empty".into());
                }
                let parts: Vec<Value> = blocks.iter().map(|b| match b.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        let txt = b.get("text").and_then(Value::as_str).ok_or("text block requires a string text field")?;
                        Ok(json!({"type": "text", "text": txt}))
                    }
                    t => Err(format!("unsupported content block type {} in a system message", t.unwrap_or("none"))),
                }).collect::<Result<_, String>>()?;
                msgs.push(json!({"role": "system", "content": parts}));
            }
            ("user", Value::String(s)) => msgs.push(json!({"role": "user", "content": s})),
            ("assistant", Value::String(s)) => msgs.push(json!({"role": "assistant", "content": s})),
            ("user", Value::Array(blocks)) => {
                if blocks.is_empty() {
                    return Err("message content blocks must not be empty".into());
                }
                let mut texts: Vec<Value> = Vec::new();
                for b in blocks {
                    let b_obj = b.as_object().ok_or("each content block must be an object")?;
                    match b_obj.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            let txt = b_obj.get("text").and_then(Value::as_str).ok_or("text block requires a string text field")?;
                            texts.push(json!({"type": "text", "text": txt}));
                        }
                        Some("image") => texts.push(json!({"type": "text", "text": "[image]"})),
                        Some("tool_result") => {
                            if !texts.is_empty() { msgs.push(json!({"role": "user", "content": std::mem::take(&mut texts)})); }
                            let id = b_obj.get("tool_use_id").and_then(Value::as_str).ok_or("tool_result without tool_use_id")?;
                            let c = match b_obj.get("content") {
                                None | Some(Value::Null) => json!(""),
                                Some(Value::String(s)) => json!(s),
                                Some(Value::Array(parts)) => {
                                    let ps: Vec<Value> = parts.iter().map(|p| match p.get("type").and_then(Value::as_str) {
                                        Some("text") => {
                                            let txt = p.get("text").and_then(Value::as_str).ok_or("tool_result text block requires a string text field")?;
                                            Ok(json!({"type": "text", "text": txt}))
                                        }
                                        Some("image") => Ok(json!({"type": "text", "text": "[image]"})),
                                        t => Err(format!("unsupported tool_result content type {}", t.unwrap_or("none"))),
                                    }).collect::<Result<_, String>>()?;
                                    Value::Array(ps)
                                }
                                Some(_) => return Err("tool_result content must be a string or an array of text blocks".into()),
                            };
                            let is_error = match b_obj.get("is_error") {
                                None | Some(Value::Null) => false,
                                Some(Value::Bool(err)) => *err,
                                Some(_) => return Err("tool_result is_error must be a boolean".into()),
                            };
                            let mut tool_msg = json!({"role": "tool", "tool_call_id": id, "content": c});
                            if is_error { tool_msg["is_error"] = json!(true); }
                            msgs.push(tool_msg);
                        }
                        t => return Err(format!("unsupported content block type {} in a user message", t.unwrap_or("none"))),
                    }
                }
                if !texts.is_empty() { msgs.push(json!({"role": "user", "content": texts})); }
            }
            ("assistant", Value::Array(blocks)) => {
                if blocks.is_empty() {
                    return Err("message content blocks must not be empty".into());
                }
                let mut texts: Vec<Value> = Vec::new();
                let mut calls: Vec<Value> = Vec::new();
                for b in blocks {
                    let b_obj = b.as_object().ok_or("each content block must be an object")?;
                    match b_obj.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            let txt = b_obj.get("text").and_then(Value::as_str).ok_or("text block requires a string text field")?;
                            if !calls.is_empty() {
                                continued_assistants.push(msgs.len());
                                msgs.push(json!({"role": "assistant", "content": std::mem::take(&mut texts), "tool_calls": std::mem::take(&mut calls)}));
                            }
                            texts.push(json!(txt));
                        }
                        Some("tool_use") => {
                            let id = b_obj.get("id").and_then(Value::as_str).ok_or("tool_use without id")?;
                            let name = b_obj.get("name").and_then(Value::as_str).filter(|s| !s.is_empty()).ok_or("tool_use without a name")?;
                            let input = match b_obj.get("input") {
                                None | Some(Value::Null) => json!({}),
                                Some(v) => v.clone(),
                            };
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
                let t_obj = t.as_object().ok_or("each tool must be an object")?;
                let name = t_obj.get("name").and_then(Value::as_str).filter(|s| !s.is_empty()).ok_or("tool without a name")?;
                let desc = match t_obj.get("description") {
                    None | Some(Value::Null) => "",
                    Some(Value::String(d)) => d.as_str(),
                    Some(_) => return Err("tool description must be a string".into()),
                };
                Ok(json!({"type": "function", "function": {"name": name, "description": desc,
                        "parameters": t_obj.get("input_schema").cloned().unwrap_or_else(|| json!({}))}}))
            }).collect::<Result<_, String>>()?;
            out["tools"] = json!(mapped);
        }
    }
    Ok((out, continued_assistants))
}

pub(super) fn anthropic_effort(req: &Value) -> Result<Option<String>, String> {
    let field = |v: &Value| -> Result<String, String> {
        let s = v.as_str().ok_or("effort must be a string")?.trim();
        let _ = validate_effort(s)?;
        Ok(s.to_string())
    };
    if let Some(oc) = req.get("output_config").filter(|v| !v.is_null()) {
        let oc_obj = oc.as_object().ok_or("output_config must be an object")?;
        if let Some(v) = oc_obj.get("effort").filter(|v| !v.is_null()) {
            return Ok(Some(field(v)?));
        }
    }
    if let Some(v) = req.get("effort").filter(|v| !v.is_null()) { return Ok(Some(field(v)?)); }
    let Some(th) = req.get("thinking").filter(|v| !v.is_null()) else { return Ok(None) };
    let th_obj = th.as_object().ok_or("thinking must be an object")?;
    if let Some(v) = th_obj.get("effort").filter(|v| !v.is_null()) { return Ok(Some(field(v)?)); }
    match th_obj.get("type").and_then(Value::as_str) {
        Some("disabled") => Ok(Some("none".into())),
        Some("adaptive") => Ok(None),
        Some("enabled") => {
            let b = th_obj.get("budget_tokens").filter(|v| !v.is_null()).ok_or("thinking budget_tokens is required when thinking is enabled")?;
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
