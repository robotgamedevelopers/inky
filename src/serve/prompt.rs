//! Render shared chat requests into the model prompt token stream.

use super::{Specials, Tokenizer};
use serde_json::{json, Value};

fn effort_num(e: &str) -> String {
    let v = match e.trim() {
        "none" => 0.0, "minimal" => 0.1, "low" => 0.2, "medium" => 0.7, "high" => 0.9, "xhigh" | "max" => 0.99,
        x => match x.parse::<f64>() {
            Ok(f) if f.is_finite() => f.clamp(0.0, 0.99),
            _ => 0.9,
        },
    };
    if v == 0.0 { "0".into() } else { format!("{v}") }
}

fn request_effort(req: &Value, default_effort: &str) -> String {
    let raw = req.get("reasoning_effort")
        .filter(|v| !v.is_null())
        .or_else(|| req.get("chat_template_kwargs").and_then(|k| k.get("reasoning_effort")).filter(|v| !v.is_null()));
    match raw {
        Some(Value::String(s)) => effort_num(s),
        Some(Value::Number(n)) => match n.as_f64() {
            Some(f) if f.is_finite() => {
                let v = f.clamp(0.0, 0.99);
                if v == 0.0 { "0".into() } else { format!("{v}") }
            }
            _ => effort_num(default_effort),
        },
        _ => effort_num(default_effort),
    }
}

fn content_parts(v: &Value) -> Vec<&str> {
    match v {
        Value::String(s) => vec![s],
        Value::Array(a) => a.iter().filter_map(|p| match p {
            Value::String(s) => Some(s.as_str()),
            Value::Object(o) => match o.get("type").and_then(|t| t.as_str()) {
                None | Some("text") | Some("input_text") => Some(o.get("text").and_then(|t| t.as_str()).unwrap_or("")),
                _ => None,
            },
            _ => None,
        }).collect(),
        _ => vec![],
    }
}

#[derive(Debug)]
pub(super) enum RenderError {
    Invalid(String),
    TooLong,
}

impl From<String> for RenderError {
    fn from(message: String) -> Self { Self::Invalid(message) }
}

impl From<&str> for RenderError {
    fn from(message: &str) -> Self { Self::Invalid(message.into()) }
}

/// The chat template of the GGUF, rendered to tokens: tool declarations, system, user, assistant (text and tool calls),
/// tool results, the thinking-effort line. Earlier assistant turns are rendered exactly as the client sends them.
pub(super) fn render_limited(tk: &Tokenizer, sp: &Specials, req: &Value, default_effort: &str, limit: usize, continued_assistants: &[usize]) -> Result<Vec<u32>, RenderError> {
    let text = |t: &mut Vec<u32>, s: &str| {
        if tk.encode_plain_limited(s, t, limit) { Ok(()) } else { Err(RenderError::TooLong) }
    };
    let marks = |t: &mut Vec<u32>, tokens: &[u32]| {
        if tokens.len() > limit.saturating_sub(t.len()) { return Err(RenderError::TooLong); }
        t.extend_from_slice(tokens);
        Ok(())
    };
    let mut t = Vec::new();
    let msgs = req["messages"].as_array().ok_or("messages missing")?;
    if let Some(tools) = req["tools"].as_array().filter(|a| !a.is_empty()) {
        let specs: Vec<Value> = tools.iter().map(|tool| {
            let f = tool.get("function").unwrap_or(tool);
            json!({"description": f.get("description").and_then(|d| d.as_str()).unwrap_or(""), "name": f["name"],
                   "parameters": f.get("parameters").cloned().unwrap_or(json!({})), "type": tool.get("type").and_then(|d| d.as_str()).unwrap_or("function")})
        }).collect();
        marks(&mut t, &[sp.msg_system])?;
        text(&mut t, "tool_declare")?;
        marks(&mut t, &[sp.c_xml])?;
        text(&mut t, &Value::Array(specs).to_string())?;
        marks(&mut t, &[sp.end])?;
    }
    let effort = request_effort(req, default_effort);
    let emit_effort = |t: &mut Vec<u32>| {
        marks(t, &[sp.msg_system, sp.c_text])?;
        text(t, &format!("Thinking effort level: {effort}"))?;
        marks(t, &[sp.end])
    };
    let mut effort_done = false;
    let mut call_names = std::collections::HashMap::new();
    for (message_index, m) in msgs.iter().enumerate() {
        let role = m["role"].as_str().ok_or("message without role")?;
        let role = if role == "developer" { "system" } else { role };
        if !effort_done && role != "system" { emit_effort(&mut t)?; effort_done = true; }
        if role == "assistant" {
            for call in m["tool_calls"].as_array().into_iter().flatten() {
                if let (Some(id), Some(name)) = (call["id"].as_str(), call["function"]["name"].as_str()) {
                    call_names.insert(id, name);
                }
            }
        }
        match role {
            "system" | "user" => {
                let h = if role == "system" { sp.msg_system } else { sp.msg_user };
                for p in content_parts(&m["content"]) {
                    marks(&mut t, &[h, sp.c_text])?;
                    text(&mut t, p)?;
                    marks(&mut t, &[sp.end])?;
                }
            }
            "tool" => {
                let name = m["name"].as_str().or_else(|| m["tool_call_id"].as_str().and_then(|id| call_names.get(id).copied())).unwrap_or_default();
                let mut parts = content_parts(&m["content"]);
                if parts.is_empty() { parts.push(""); }
                let is_error = m["is_error"].as_bool().unwrap_or(false);
                for (i, p) in parts.iter().enumerate() {
                    marks(&mut t, &[sp.msg_tool])?;
                    text(&mut t, name)?;
                    marks(&mut t, &[sp.c_text])?;
                    if is_error && i == 0 { text(&mut t, "Error: ")?; }
                    text(&mut t, p)?;
                    marks(&mut t, &[sp.end])?;
                }
            }
            "assistant" => {
                let asst_name = m["name"].as_str().filter(|n| !n.is_empty());
                let reasoning = m["reasoning_content"].as_str().filter(|r| !r.is_empty());
                let model_part = |tokens: &mut Vec<u32>, kind, body: &str| {
                    marks(tokens, &[sp.msg_model])?;
                    if let Some(name) = asst_name { text(tokens, name)?; }
                    marks(tokens, &[kind])?;
                    text(tokens, body)?;
                    marks(tokens, &[sp.end])
                };
                if let Some(r) = reasoning { model_part(&mut t, sp.c_think, r)?; }
                let calls: Vec<&Value> = m["tool_calls"].as_array().into_iter().flatten().collect();
                match &m["content"] {
                    Value::Null => {}
                    Value::String(s) => {
                        if !s.is_empty() || (calls.is_empty() && reasoning.is_none()) {
                            model_part(&mut t, sp.c_text, s)?;
                        }
                    }
                    Value::Array(parts) => {
                        for part in parts {
                            let value = if let Some(s) = part.as_str() { s } else {
                                if let Some(kind) = part.get("type") {
                                    if !matches!(kind.as_str(), Some("text" | "input_text")) {
                                        return Err(format!("unsupported assistant content part type {kind}").into());
                                    }
                                }
                                part["text"].as_str().unwrap_or("")
                            };
                            model_part(&mut t, sp.c_text, value)?;
                        }
                    }
                    _ => return Err("assistant content must be a string, array or null".into()),
                }
                for tc in calls {
                    let name = tc["function"]["name"].as_str().ok_or("tool call without a name")?;
                    let args: Value = match &tc["function"]["arguments"] {
                        Value::String(s) if s.trim().is_empty() => json!({}),
                        Value::String(s) => serde_json::from_str(s).map_err(|e| format!("tool call arguments: {e}"))?,
                        Value::Null => json!({}), v => v.clone(),
                    };
                    marks(&mut t, &[sp.msg_model])?;
                    text(&mut t, name)?;
                    marks(&mut t, &[sp.c_tool])?;
                    text(&mut t, &format!("{{\"name\":{},\"args\":{}}}", Value::String(name.into()), args))?;
                    marks(&mut t, &[sp.end])?;
                }
                if continued_assistants.binary_search(&message_index).is_err() {
                    marks(&mut t, &[sp.end_sampling])?;
                }
            }
            r => return Err(format!("unknown role {r}").into()),
        }
    }
    if !effort_done { emit_effort(&mut t)?; }
    Ok(t)
}

#[cfg(test)]
pub(super) fn render(tk: &Tokenizer, sp: &Specials, req: &Value, default_effort: &str) -> Result<Vec<u32>, RenderError> {
    render_limited(tk, sp, req, default_effort, usize::MAX, &[])
}
