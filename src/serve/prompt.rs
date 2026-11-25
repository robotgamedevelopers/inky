//! Render shared chat requests into the model prompt token stream.

use super::{Specials, Tokenizer};
use serde_json::{json, Value};

fn format_effort_num(v: f64) -> String {
    if v == 0.0 { "0".into() } else { format!("{v}") }
}

pub fn validate_effort(e: &str) -> Result<String, String> {
    let trimmed = e.trim();
    let v = match trimmed {
        "none" => 0.0,
        "minimal" => 0.1,
        "low" => 0.2,
        "medium" => 0.7,
        "high" => 0.9,
        "xhigh" | "max" => 0.99,
        x => match x.parse::<f64>() {
            Ok(f) if f.is_finite() && (0.0..=1.0).contains(&f) => f.clamp(0.0, 0.99),
            _ => {
                return Err(format!(
                    "unknown effort level {trimmed:?} (expected none, minimal, low, medium, high, xhigh, max, or a number between 0 and 1)"
                ));
            }
        },
    };
    Ok(format_effort_num(v))
}

pub(super) fn request_effort(req: &Value, default_effort: &str) -> Result<String, String> {
    let kwargs = match req.get("chat_template_kwargs") {
        None | Some(Value::Null) => None,
        Some(Value::Object(o)) => Some(o),
        Some(_) => return Err("chat_template_kwargs must be an object".into()),
    };
    let raw = match req.get("reasoning_effort") {
        Some(v) if !v.is_null() => Some(v),
        _ => kwargs.and_then(|k| k.get("reasoning_effort")).filter(|v| !v.is_null()),
    };
    match raw {
        None => validate_effort(default_effort),
        Some(Value::String(s)) => validate_effort(s),
        Some(Value::Number(n)) => match n.as_f64() {
            Some(f) if f.is_finite() && (0.0..=0.99).contains(&f) => Ok(format_effort_num(f)),
            _ => Err("reasoning_effort must be a finite number between 0 and 0.99".into()),
        },
        Some(_) => Err("reasoning_effort must be a string or a finite number between 0 and 0.99".into()),
    }
}

fn try_content_parts(v: &Value, allow_empty_array: bool) -> Result<Vec<&str>, String> {
    match v {
        Value::String(s) => Ok(vec![s.as_str()]),
        Value::Array(a) => {
            if a.is_empty() && !allow_empty_array {
                return Err("message content array must not be empty".into());
            }
            a.iter()
                .map(|p| match p {
                    Value::String(s) => Ok(s.as_str()),
                    Value::Object(o) => {
                        match o.get("type") {
                            None => {}
                            Some(Value::String(kind)) if matches!(kind.as_str(), "text" | "input_text") => {}
                            Some(Value::String(other)) => {
                                return Err(format!("unsupported content part type {other}"));
                            }
                            Some(_) => return Err("content part type must be a string".into()),
                        }
                        o.get("text")
                            .and_then(Value::as_str)
                            .ok_or_else(|| "text content part requires a string text field".to_string())
                    }
                    _ => Err("each content part must be a string or text object".into()),
                })
                .collect()
        }
        _ => Err("message content must be a string or an array of text parts".into()),
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
    if !req.is_object() {
        return Err("request body must be a JSON object".into());
    }
    let text = |t: &mut Vec<u32>, s: &str| {
        if tk.encode_plain_limited(s, t, limit) { Ok(()) } else { Err(RenderError::TooLong) }
    };
    let marks = |t: &mut Vec<u32>, tokens: &[u32]| {
        if tokens.len() > limit.saturating_sub(t.len()) { return Err(RenderError::TooLong); }
        t.extend_from_slice(tokens);
        Ok(())
    };
    let mut t = Vec::new();
    let msgs = req.get("messages").and_then(Value::as_array).ok_or("messages missing")?;
    if msgs.is_empty() {
        return Err("messages must not be empty".into());
    }
    if let Some(tools_val) = req.get("tools").filter(|v| !v.is_null()) {
        let tools = tools_val.as_array().ok_or("tools must be an array")?;
        if !tools.is_empty() {
            let mut specs = Vec::with_capacity(tools.len());
            for tool in tools {
                let tool_obj = tool.as_object().ok_or("each tool must be an object")?;
                let f = match tool_obj.get("function") {
                    None | Some(Value::Null) => tool_obj,
                    Some(Value::Object(fo)) => fo,
                    Some(_) => return Err("tool.function must be an object".into()),
                };
                let name = f.get("name").and_then(Value::as_str).filter(|s| !s.is_empty()).ok_or("tool without a name")?;
                let desc = match f.get("description") {
                    None | Some(Value::Null) => "",
                    Some(Value::String(d)) => d.as_str(),
                    Some(_) => return Err("tool description must be a string".into()),
                };
                let tool_type = match tool_obj.get("type") {
                    None | Some(Value::Null) => "function",
                    Some(Value::String(ty)) => ty.as_str(),
                    Some(_) => return Err("tool type must be a string".into()),
                };
                let params = f.get("parameters").cloned().unwrap_or_else(|| json!({}));
                specs.push(json!({"description": desc, "name": name, "parameters": params, "type": tool_type}));
            }
            marks(&mut t, &[sp.msg_system])?;
            text(&mut t, "tool_declare")?;
            marks(&mut t, &[sp.c_xml])?;
            text(&mut t, &Value::Array(specs).to_string())?;
            marks(&mut t, &[sp.end])?;
        }
    }
    let effort = request_effort(req, default_effort)?;
    let emit_effort = |t: &mut Vec<u32>| {
        marks(t, &[sp.msg_system, sp.c_text])?;
        text(t, &format!("Thinking effort level: {effort}"))?;
        marks(t, &[sp.end])
    };
    let mut effort_done = false;
    let mut call_names = std::collections::HashMap::new();
    for (message_index, m) in msgs.iter().enumerate() {
        let m_obj = m.as_object().ok_or("each message must be an object")?;
        let role = m_obj.get("role").and_then(Value::as_str).ok_or("message without role")?;
        let role = if role == "developer" { "system" } else { role };
        if !effort_done && role != "system" { emit_effort(&mut t)?; effort_done = true; }
        if role == "assistant" {
            if let Some(tcs) = m_obj.get("tool_calls").and_then(Value::as_array) {
                for tc in tcs {
                    if let (Some(id), Some(n)) = (
                        tc.get("id").and_then(Value::as_str),
                        tc.get("function").and_then(|f| f.get("name")).and_then(Value::as_str),
                    ) {
                        call_names.insert(id.to_string(), n.to_string());
                    }
                }
            }
        }
        match role {
            "system" | "user" => {
                let h = if role == "system" { sp.msg_system } else { sp.msg_user };
                let content = m_obj.get("content").ok_or("message content missing")?;
                for p in try_content_parts(content, false)? {
                    marks(&mut t, &[h, sp.c_text])?;
                    text(&mut t, p)?;
                    marks(&mut t, &[sp.end])?;
                }
            }
            "tool" => {
                let name = match m_obj.get("name") {
                    None | Some(Value::Null) => None,
                    Some(Value::String(s)) => Some(s.clone()),
                    Some(_) => return Err("tool message name must be a string".into()),
                }
                .or_else(|| m_obj.get("tool_call_id").and_then(Value::as_str).and_then(|id| call_names.get(id).cloned()))
                .unwrap_or_default();
                let mut parts = match m_obj.get("content") {
                    None | Some(Value::Null) => vec![""],
                    Some(c) => try_content_parts(c, true)?,
                };
                if parts.is_empty() { parts.push(""); }
                let is_error = match m_obj.get("is_error") {
                    None | Some(Value::Null) => false,
                    Some(Value::Bool(b)) => *b,
                    Some(_) => return Err("tool is_error must be a boolean".into()),
                };
                for (i, p) in parts.iter().enumerate() {
                    marks(&mut t, &[sp.msg_tool])?;
                    text(&mut t, &name)?;
                    marks(&mut t, &[sp.c_text])?;
                    if is_error && i == 0 {
                        text(&mut t, &format!("Error: {p}"))?;
                    } else {
                        text(&mut t, p)?;
                    }
                    marks(&mut t, &[sp.end])?;
                }
            }
            "assistant" => {
                let asst_name = match m_obj.get("name") {
                    None | Some(Value::Null) => None,
                    Some(Value::String(n)) => (!n.is_empty()).then_some(n.as_str()),
                    Some(_) => return Err("assistant name must be a string".into()),
                };
                let reasoning = match m_obj.get("reasoning_content") {
                    None | Some(Value::Null) => None,
                    Some(Value::String(r)) => (!r.is_empty()).then_some(r.as_str()),
                    Some(_) => return Err("reasoning_content must be a string".into()),
                };
                let model_part = |tokens: &mut Vec<u32>, kind, body: &str| {
                    marks(tokens, &[sp.msg_model])?;
                    if let Some(name) = asst_name { text(tokens, name)?; }
                    marks(tokens, &[kind])?;
                    text(tokens, body)?;
                    marks(tokens, &[sp.end])
                };
                if let Some(r) = reasoning {
                    model_part(&mut t, sp.c_think, r)?;
                }
                let calls: Vec<&Value> = match m_obj.get("tool_calls") {
                    None | Some(Value::Null) => Vec::new(),
                    Some(Value::Array(arr)) => arr.iter().collect(),
                    Some(_) => return Err("tool_calls must be an array".into()),
                };
                match m_obj.get("content") {
                    None | Some(Value::Null) => {
                        if calls.is_empty() && reasoning.is_none() {
                            return Err("assistant message must include content, reasoning_content, or tool_calls".into());
                        }
                    }
                    Some(Value::String(s)) => {
                        if !s.is_empty() || (calls.is_empty() && reasoning.is_none()) {
                            model_part(&mut t, sp.c_text, s)?;
                        }
                    }
                    Some(arr @ Value::Array(_)) => {
                        let parts = try_content_parts(arr, true)?;
                        for value in parts {
                            model_part(&mut t, sp.c_text, value)?;
                        }
                    }
                    Some(_) => return Err("assistant content must be a string, array or null".into()),
                }
                for tc in calls {
                    let tc_obj = tc.as_object().ok_or("each tool call must be an object")?;
                    let fn_obj = tc_obj.get("function").and_then(Value::as_object).ok_or("tool call function must be an object")?;
                    let name = fn_obj.get("name").and_then(Value::as_str).filter(|s| !s.is_empty()).ok_or("tool call without a name")?;
                    let args: Value = match fn_obj.get("arguments") {
                        None | Some(Value::Null) => json!({}),
                        Some(Value::String(s)) if s.trim().is_empty() => json!({}),
                        Some(Value::String(s)) => serde_json::from_str(s).map_err(|e| format!("tool call arguments: {e}"))?,
                        Some(v) => v.clone(),
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
