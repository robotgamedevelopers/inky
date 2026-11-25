//! Decode model replies and build Anthropic response blocks.

use super::{Specials, Tokenizer};
use serde_json::{json, Value};

enum ReplyPart {
    Text(String),
    Thinking(String),
    Tool(String, Value),
}

fn parse_tool_call(header: &[u8], body: &[u8]) -> (String, Value) {
    let s = String::from_utf8_lossy(body);
    let v: Value = serde_json::from_str(&s).unwrap_or(Value::Null);
    let name = v["name"].as_str().map(str::to_owned).unwrap_or_else(|| String::from_utf8_lossy(header).into_owned());
    let args = if v["args"].is_object() { v["args"].clone() } else { json!({"_unparsed": s}) };
    (name, args)
}

/// Visit complete blocks in generation order. Final text is retained on length stops;
/// a truncated tool body must never become a call.
fn visit_reply(tk: &Tokenizer, sp: &Specials, toks: &[u32], mut emit: impl FnMut(ReplyPart)) {
    #[derive(PartialEq)]
    enum M {
        Header,
        Think,
        Text,
        Tool,
        None,
    }
    let (mut mode, mut cur, mut header) = (M::None, Vec::<u8>::new(), Vec::<u8>::new());
    let flush_open = |mode: &M, cur: &mut Vec<u8>, emit: &mut dyn FnMut(ReplyPart)| match mode {
        M::Think => emit(ReplyPart::Thinking(take_stream_text(cur, true))),
        M::Text => emit(ReplyPart::Text(take_stream_text(cur, true))),
        _ => cur.clear(),
    };
    for &t in toks {
        if sp.is_role_stop(t) {
            flush_open(&mode, &mut cur, &mut emit);
            mode = M::None;
            break;
        }
        let ends = t == sp.end || t == sp.end_sampling || t == tk.eos;
        if t == sp.msg_model || t == sp.c_think || t == sp.c_text || t == sp.c_tool || ends {
            if t == sp.c_text && mode == M::Text {
                continue;
            }
            match mode {
                M::Text | M::Think => flush_open(&mode, &mut cur, &mut emit),
                M::Tool if ends => {
                    let (name, args) = parse_tool_call(&header, &cur);
                    emit(ReplyPart::Tool(name, args));
                    cur.clear();
                }
                _ => cur.clear(),
            }
            mode = if t == sp.msg_model {
                header.clear();
                M::Header
            } else if t == sp.c_think {
                M::Think
            } else if t == sp.c_text {
                M::Text
            } else if t == sp.c_tool {
                M::Tool
            } else {
                M::None
            };
        } else {
            match mode {
                M::Header => header.extend(tk.decode_bytes(t)),
                M::None => {}
                _ => cur.extend(tk.decode_bytes(t)),
            }
        }
    }
    flush_open(&mode, &mut cur, &mut emit);
}

/// Chat represents text, reasoning and tool calls in separate fields.
pub(super) struct Parsed {
    pub(super) text: String,
    pub(super) thinking: String,
    pub(super) calls: Vec<(String, Value)>,
}
pub(super) fn parse(tk: &Tokenizer, sp: &Specials, toks: &[u32]) -> Parsed {
    let mut p = Parsed { text: String::new(), thinking: String::new(), calls: Vec::new() };
    visit_reply(tk, sp, toks, |part| match part {
        ReplyPart::Text(s) => p.text.push_str(&s),
        ReplyPart::Thinking(s) => p.thinking.push_str(&s),
        ReplyPart::Tool(name, args) => p.calls.push((name, args)),
    });
    p
}

/// The reply's stop reason: a tool call outranks the length stop, as the chat path's finish reason does.
pub(super) fn anthropic_stop_reason(has_calls: bool, reason: &str) -> &'static str {
    if has_calls {
        "tool_use"
    } else if reason == "length" {
        "max_tokens"
    } else {
        "end_turn"
    }
}

pub(super) fn tool_use_id(nonce: &str, seq: u64, i: usize) -> String {
    format!("toolu_{nonce}_{seq}_{i}")
}

/// Messages represents each visible text or tool block in generation order.
pub(super) fn anthropic_content(tk: &Tokenizer, sp: &Specials, toks: &[u32], nonce: &str, seq: u64) -> Vec<Value> {
    let mut content = Vec::new();
    let mut calls = 0;
    visit_reply(tk, sp, toks, |part| match part {
        ReplyPart::Text(text) => content.push(json!({"type": "text", "text": text})),
        ReplyPart::Tool(name, args) => {
            content
                .push(json!({"type": "tool_use", "id": tool_use_id(nonce, seq, calls), "name": name, "input": args}));
            calls += 1;
        }
        ReplyPart::Thinking(_) => {}
    });
    content
}

/// Usage in Anthropic's split: `input_tokens` is the part of the prompt no lane reused,
/// `cache_read_input_tokens` the part it did, so the two sum to the prompt.
pub(super) fn anthropic_usage(prompt: usize, cached: usize, gen: usize) -> Value {
    json!({"input_tokens": prompt.saturating_sub(cached), "output_tokens": gen,
           "cache_creation_input_tokens": 0, "cache_read_input_tokens": cached})
}

/// The final streaming usage, carried by `message_delta`: the same split as a
/// non-streaming reply. `message_start` is sent before the lane's reuse is known and
/// provisionally reports the whole prompt as `input_tokens`; the delta's split
/// supersedes it, so the client's final accounting sums to the prompt, not to the
/// prompt plus its cached part a second time.
pub(super) fn streaming_delta_usage(prompt: usize, cached: usize, gen: usize) -> Value {
    anthropic_usage(prompt, cached, gen)
}

pub(super) fn sse_text(event: &str, data: &Value) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

pub(super) fn take_stream_text(pend: &mut Vec<u8>, end: bool) -> String {
    let mut text = String::new();
    crate::stream_text::stream_text(pend, end, |s| text.push_str(s));
    text
}

/// A streamed reply as content blocks, fed one generated token at a time. Text streams as it
/// decodes; a tool call is buffered to its end and sent whole, since its JSON wraps the arguments
/// the block must stream on their own. Thinking tokens open no block.
pub(super) struct BlockStream {
    nonce: String,
    seq: u64,
    next: usize,
    pub(super) calls: usize,
    text_index: usize,
    open_text: bool,
    mode: u8,
    pend: Vec<u8>,
    header: Vec<u8>,
    tool: Vec<u8>,
    in_header: bool,
    stopped: bool,
}
impl BlockStream {
    pub(super) fn new(nonce: &str, seq: u64) -> BlockStream {
        BlockStream {
            nonce: nonce.to_string(),
            seq,
            next: 0,
            calls: 0,
            text_index: 0,
            open_text: false,
            mode: 0,
            pend: vec![],
            header: vec![],
            tool: vec![],
            in_header: false,
            stopped: false,
        }
    }
    fn flush_text(&mut self, out: &mut Vec<String>) {
        if !self.open_text {
            self.pend.clear();
            return;
        }
        let text = take_stream_text(&mut self.pend, false);
        if !text.is_empty() {
            out.push(sse_text(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": self.text_index,
                "delta": {"type": "text_delta", "text": text}}),
            ));
        }
    }
    fn close_text(&mut self, out: &mut Vec<String>) {
        if self.open_text {
            let text = take_stream_text(&mut self.pend, true);
            if !text.is_empty() {
                out.push(sse_text(
                    "content_block_delta",
                    &json!({"type": "content_block_delta", "index": self.text_index,
                    "delta": {"type": "text_delta", "text": text}}),
                ));
            }
            out.push(sse_text("content_block_stop", &json!({"type": "content_block_stop", "index": self.text_index})));
            self.open_text = false;
        }
        self.pend.clear();
    }
    fn emit_tool(&mut self, out: &mut Vec<String>) {
        let (name, args) = parse_tool_call(&self.header, &self.tool);
        let index = self.next;
        self.next += 1;
        let id = tool_use_id(&self.nonce, self.seq, self.calls);
        self.calls += 1;
        out.push(sse_text(
            "content_block_start",
            &json!({"type": "content_block_start", "index": index,
            "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}}),
        ));
        out.push(sse_text(
            "content_block_delta",
            &json!({"type": "content_block_delta", "index": index,
            "delta": {"type": "input_json_delta", "partial_json": args.to_string()}}),
        ));
        out.push(sse_text("content_block_stop", &json!({"type": "content_block_stop", "index": index})));
        self.tool.clear();
    }
    pub(super) fn on_token(&mut self, tk: &Tokenizer, sp: &Specials, t: u32) -> Vec<String> {
        if self.stopped {
            return Vec::new();
        }
        let mut out = Vec::new();
        if sp.is_role_stop(t) {
            if self.mode == 3 {
                self.tool.clear();
            }
            self.close_text(&mut out);
            self.mode = 0;
            self.in_header = false;
            self.stopped = true;
            return out;
        }
        if t == sp.msg_model {
            self.close_text(&mut out);
            self.mode = 0;
            self.in_header = true;
            self.header.clear();
            return out;
        }
        if t == sp.c_think {
            self.close_text(&mut out);
            self.mode = 1;
            self.in_header = false;
            return out;
        }
        if t == sp.c_text {
            if self.open_text {
                let text = take_stream_text(&mut self.pend, true);
                if !text.is_empty() {
                    out.push(sse_text(
                        "content_block_delta",
                        &json!({"type": "content_block_delta", "index": self.text_index,
                        "delta": {"type": "text_delta", "text": text}}),
                    ));
                }
            } else {
                self.text_index = self.next;
                self.next += 1;
                self.open_text = true;
                out.push(sse_text(
                    "content_block_start",
                    &json!({"type": "content_block_start", "index": self.text_index,
                    "content_block": {"type": "text", "text": ""}}),
                ));
            }
            self.mode = 2;
            self.in_header = false;
            return out;
        }
        if t == sp.c_tool {
            self.close_text(&mut out);
            self.mode = 3;
            self.in_header = false;
            self.tool.clear();
            return out;
        }
        if t == sp.end || t == sp.end_sampling || t == tk.eos {
            if self.mode == 3 {
                self.emit_tool(&mut out);
            }
            self.close_text(&mut out);
            self.mode = 0;
            self.in_header = false;
            return out;
        }
        let b = tk.decode_bytes(t);
        match self.mode {
            2 => {
                self.pend.extend(&b);
                self.flush_text(&mut out);
            }
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
    pub(super) fn close_all(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        if self.mode == 3 {
            self.tool.clear();
            self.mode = 0;
        }
        self.close_text(&mut out);
        out
    }
}

// ------------------------------------------------------------------------------------------------- scheduler

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_utf8_recovers_and_flushes_incomplete_sequences() {
        let mut pend = vec![0xff, b'h', b'i', 0xe2, 0x82];
        assert_eq!(take_stream_text(&mut pend, false), "�hi");
        assert_eq!(pend, vec![0xe2, 0x82]);
        pend.push(0xac);
        assert_eq!(take_stream_text(&mut pend, false), "€");
        pend.extend([0xe2, 0x82]);
        assert_eq!(take_stream_text(&mut pend, true), "�");
        assert!(pend.is_empty());
        let mut bs = BlockStream::new("test", 1);
        bs.open_text = true;
        bs.pend = vec![0xff, b'h', b'i', 0xe2];
        let mut events = Vec::new();
        bs.flush_text(&mut events);
        bs.close_text(&mut events);
        let text: String = events
            .iter()
            .filter_map(|event| {
                let v: Value = serde_json::from_str(event.split("\ndata: ").nth(1)?.trim_end()).ok()?;
                v["delta"]["text"].as_str().map(str::to_owned)
            })
            .collect();
        assert_eq!(text, "�hi�");
    }
}
