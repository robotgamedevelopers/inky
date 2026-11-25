// Canonical versioned JSONL trace format, emitter, fail-closed parser, and
// compact TLA+ bundle renderer for `specs/BlockLayout.tla` and `specs/CacheScheduler.tla`.
// Plain `//` comments at the top so standalone `rustc` scripts in `scripts/` can include by path.
#![allow(dead_code)] // Layer 3/4 callers and standalone CI trace checkers consume this module; cargo check skips #[cfg(test)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};
use std::io;

use crate::blocks::{BlockEntry, BlockIndex, BlockKind, BlockPools, BlockTable, Slot};

/// Schema version of the Layer 3/4 JSONL trace format (`docs/TRACE-FORMAT.md`).
pub const TRACE_SCHEMA_VERSION: u32 = 1;

/// Bytes in 1 MiB (`2^20`), the low-limb radix in `specs/BlockLayout.tla`.
pub const MIB_BYTES: u64 = 1_048_576;
/// MiB in 4 GiB (`2^12` MiB = `2^32` bytes), the middle-limb radix in `specs/BlockLayout.tla`.
pub const GIB4_MIB: u64 = 4_096;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TraceError {
    Io(String),
    JsonSyntax { line: usize, col: usize, msg: String },
    InvalidSchema(String),
    OffsetOutOfRange(String),
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceError::Io(msg) => write!(f, "trace I/O error: {msg}"),
            TraceError::JsonSyntax { line, col, msg } => {
                write!(f, "trace JSON syntax error at line {line}, col {col}: {msg}")
            }
            TraceError::InvalidSchema(msg) => write!(f, "trace schema error: {msg}"),
            TraceError::OffsetOutOfRange(msg) => write!(f, "trace offset out of range: {msg}"),
        }
    }
}

impl From<io::Error> for TraceError {
    fn from(err: io::Error) -> Self {
        TraceError::Io(err.to_string())
    }
}

/// 32-bit-safe decomposition of a `u64` byte offset matching `specs/BlockLayout.tla`:
/// `u64 == (hi << 32) | (mib << 20) | sub`, with `mib < 4096` and `sub < 1_048_576`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OffsetLimbs {
    pub hi: u32,
    pub mib: u32,
    pub sub: u32,
}

impl OffsetLimbs {
    pub const ZERO: OffsetLimbs = OffsetLimbs { hi: 0, mib: 0, sub: 0 };

    pub fn from_u64(offset: u64) -> Result<OffsetLimbs, TraceError> {
        let hi64 = offset >> 32;
        if hi64 > i32::MAX as u64 {
            return Err(TraceError::OffsetOutOfRange(format!(
                "high 32-bit limb {hi64} exceeds signed 32-bit TLC Int"
            )));
        }
        Ok(OffsetLimbs {
            hi: hi64 as u32,
            mib: ((offset >> 20) & 0xfff) as u32,
            sub: (offset & 0xfffff) as u32,
        })
    }

    pub fn to_u64(self) -> Result<u64, TraceError> {
        if self.hi > i32::MAX as u32 || u64::from(self.mib) >= GIB4_MIB || u64::from(self.sub) >= MIB_BYTES {
            return Err(TraceError::OffsetOutOfRange(format!(
                "non-canonical offset limbs hi={}, mib={}, sub={}",
                self.hi, self.mib, self.sub
            )));
        }
        Ok((u64::from(self.hi) << 32) | (u64::from(self.mib) << 20) | u64::from(self.sub))
    }

    /// Mirrors `Truncate32(o)` in `specs/BlockLayout.tla` (drops the upper 32 bits).
    pub fn truncate_32(self) -> OffsetLimbs {
        OffsetLimbs { hi: 0, mib: self.mib, sub: self.sub }
    }

    /// Mirrors `AddBytes(o, n)` in `specs/BlockLayout.tla` with checked carry.
    pub fn add_bytes(self, n: u32) -> Result<OffsetLimbs, TraceError> {
        let base = self.to_u64()?;
        let sum = base.checked_add(u64::from(n)).ok_or_else(|| {
            TraceError::OffsetOutOfRange(format!("byte offset {base} + {n} overflowed u64"))
        })?;
        OffsetLimbs::from_u64(sum)
    }
}

pub fn kind_name(kind: BlockKind) -> &'static str {
    match kind {
        BlockKind::Full => "full",
        BlockKind::Window => "window",
    }
}

pub fn parse_kind(s: &str) -> Result<BlockKind, TraceError> {
    match s {
        "full" => Ok(BlockKind::Full),
        "window" => Ok(BlockKind::Window),
        other => Err(TraceError::InvalidSchema(format!("unknown block kind {other:?}"))),
    }
}

fn kind_rank(kind: BlockKind) -> u8 {
    match kind {
        BlockKind::Full => 0,
        BlockKind::Window => 1,
    }
}

// ---- Minimal Deterministic JSON Value, Serializer & Strict Parser ---------

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum JsonVal {
    Null,
    Bool(bool),
    UInt(u64),
    Int(i64),
    Str(String),
    Arr(Vec<JsonVal>),
    Obj(BTreeMap<String, JsonVal>),
}

impl JsonVal {
    pub fn str(s: impl Into<String>) -> JsonVal {
        JsonVal::Str(s.into())
    }
    pub fn uint(n: impl Into<u64>) -> JsonVal {
        JsonVal::UInt(n.into())
    }

    pub fn to_json_line(&self) -> String {
        let mut out = String::with_capacity(256);
        self.write_to(&mut out);
        out
    }

    fn write_to(&self, out: &mut String) {
        match self {
            JsonVal::Null => out.push_str("null"),
            JsonVal::Bool(true) => out.push_str("true"),
            JsonVal::Bool(false) => out.push_str("false"),
            JsonVal::UInt(n) => {
                let _ = write!(out, "{n}");
            }
            JsonVal::Int(n) => {
                let _ = write!(out, "{n}");
            }
            JsonVal::Str(s) => write_json_str(out, s),
            JsonVal::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write_to(out);
                }
                out.push(']');
            }
            JsonVal::Obj(map) => {
                out.push('{');
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_json_str(out, k);
                    out.push(':');
                    v.write_to(out);
                }
                out.push('}');
            }
        }
    }

    fn as_obj(&self) -> Result<&BTreeMap<String, JsonVal>, TraceError> {
        match self {
            JsonVal::Obj(m) => Ok(m),
            _ => Err(TraceError::InvalidSchema("expected JSON object".into())),
        }
    }

    fn as_arr(&self) -> Result<&[JsonVal], TraceError> {
        match self {
            JsonVal::Arr(a) => Ok(a),
            _ => Err(TraceError::InvalidSchema("expected JSON array".into())),
        }
    }

    fn as_str(&self) -> Result<&str, TraceError> {
        match self {
            JsonVal::Str(s) => Ok(s),
            _ => Err(TraceError::InvalidSchema("expected JSON string".into())),
        }
    }

    fn as_bool(&self) -> Result<bool, TraceError> {
        match self {
            JsonVal::Bool(b) => Ok(*b),
            _ => Err(TraceError::InvalidSchema("expected JSON boolean".into())),
        }
    }

    fn as_u64(&self) -> Result<u64, TraceError> {
        match self {
            JsonVal::UInt(n) => Ok(*n),
            JsonVal::Int(n) if *n >= 0 => Ok(*n as u64),
            _ => Err(TraceError::InvalidSchema("expected non-negative integer".into())),
        }
    }

    fn as_u32(&self) -> Result<u32, TraceError> {
        let n = self.as_u64()?;
        u32::try_from(n).map_err(|_| TraceError::InvalidSchema(format!("integer {n} exceeds u32")))
    }
}

fn write_json_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
    line: usize,
}

impl<'a> JsonParser<'a> {
    fn new(line_no: usize, src: &'a str) -> Self {
        JsonParser { bytes: src.as_bytes(), pos: 0, line: line_no }
    }

    fn err<T>(&self, msg: impl Into<String>) -> Result<T, TraceError> {
        Err(TraceError::JsonSyntax { line: self.line, col: self.pos + 1, msg: msg.into() })
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && matches!(self.bytes[self.pos], b' ' | b'\t' | b'\r') {
            self.pos += 1;
        }
    }

    fn parse_line(line_no: usize, src: &'a str) -> Result<JsonVal, TraceError> {
        let mut p = JsonParser::new(line_no, src);
        p.skip_ws();
        let val = p.parse_val()?;
        p.skip_ws();
        if p.pos != p.bytes.len() {
            return p.err("unexpected trailing characters");
        }
        Ok(val)
    }

    fn parse_val(&mut self) -> Result<JsonVal, TraceError> {
        self.skip_ws();
        let Some(&b) = self.bytes.get(self.pos) else {
            return self.err("unexpected end of JSON input");
        };
        match b {
            b'{' => self.parse_obj(),
            b'[' => self.parse_arr(),
            b'"' => self.parse_str().map(JsonVal::Str),
            b't' => self.parse_lit(b"true", JsonVal::Bool(true)),
            b'f' => self.parse_lit(b"false", JsonVal::Bool(false)),
            b'n' => self.parse_lit(b"null", JsonVal::Null),
            b'-' | b'0'..=b'9' => self.parse_num(),
            _ => self.err(format!("unexpected byte {:?}", b as char)),
        }
    }

    fn parse_lit(&mut self, lit: &[u8], val: JsonVal) -> Result<JsonVal, TraceError> {
        if self.bytes.get(self.pos..self.pos + lit.len()) == Some(lit) {
            self.pos += lit.len();
            Ok(val)
        } else {
            self.err("invalid literal")
        }
    }

    fn parse_num(&mut self) -> Result<JsonVal, TraceError> {
        let start = self.pos;
        let neg = self.bytes[self.pos] == b'-';
        if neg {
            self.pos += 1;
        }
        if self.pos >= self.bytes.len() || !self.bytes[self.pos].is_ascii_digit() {
            return self.err("expected digit");
        }
        if self.bytes[self.pos] == b'0' && self.bytes.get(self.pos + 1).is_some_and(u8::is_ascii_digit) {
            return self.err("leading zero in integer");
        }
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos < self.bytes.len() && matches!(self.bytes[self.pos], b'.' | b'e' | b'E') {
            return self.err("floating-point numbers are not allowed in traces");
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap();
        if neg {
            let v: i64 = s.parse().map_err(|_| TraceError::JsonSyntax {
                line: self.line,
                col: start + 1,
                msg: "signed integer overflow".into(),
            })?;
            Ok(JsonVal::Int(v))
        } else {
            let v: u64 = s.parse().map_err(|_| TraceError::JsonSyntax {
                line: self.line,
                col: start + 1,
                msg: "u64 integer overflow".into(),
            })?;
            Ok(JsonVal::UInt(v))
        }
    }

    fn parse_str(&mut self) -> Result<String, TraceError> {
        self.pos += 1;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == b'"' {
                self.pos += 1;
                return Ok(out);
            }
            if b == b'\\' {
                self.pos += 1;
                let Some(&esc) = self.bytes.get(self.pos) else {
                    return self.err("unterminated escape");
                };
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\x08'),
                    b'f' => out.push('\x0c'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let Some(hex) = self.bytes.get(self.pos..self.pos + 4) else {
                            return self.err("short \\u escape");
                        };
                        let hex_str = std::str::from_utf8(hex).map_err(|_| TraceError::JsonSyntax {
                            line: self.line,
                            col: self.pos + 1,
                            msg: "invalid utf8 in \\u escape".into(),
                        })?;
                        let code = u32::from_str_radix(hex_str, 16).map_err(|_| TraceError::JsonSyntax {
                            line: self.line,
                            col: self.pos + 1,
                            msg: "invalid hex in \\u escape".into(),
                        })?;
                        let Some(ch) = char::from_u32(code) else {
                            return self.err("surrogate \\u escape unsupported");
                        };
                        self.pos += 4;
                        out.push(ch);
                    }
                    _ => return self.err("invalid escape character"),
                }
            } else {
                if b < 0x20 {
                    return self.err("unescaped control character in string");
                }
                let rest = std::str::from_utf8(&self.bytes[self.pos..]).map_err(|_| TraceError::JsonSyntax {
                    line: self.line,
                    col: self.pos + 1,
                    msg: "invalid UTF-8 in string".into(),
                })?;
                let ch = rest.chars().next().unwrap();
                self.pos += ch.len_utf8();
                out.push(ch);
            }
        }
        self.err("unterminated string")
    }

    fn parse_arr(&mut self) -> Result<JsonVal, TraceError> {
        self.pos += 1;
        self.skip_ws();
        let mut items = Vec::new();
        if self.bytes.get(self.pos) == Some(&b']') {
            self.pos += 1;
            return Ok(JsonVal::Arr(items));
        }
        loop {
            items.push(self.parse_val()?);
            self.skip_ws();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonVal::Arr(items));
                }
                _ => return self.err("expected ',' or ']' in array"),
            }
        }
    }

    fn parse_obj(&mut self) -> Result<JsonVal, TraceError> {
        self.pos += 1;
        self.skip_ws();
        let mut map = BTreeMap::new();
        if self.bytes.get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Ok(JsonVal::Obj(map));
        }
        loop {
            self.skip_ws();
            if self.bytes.get(self.pos) != Some(&b'"') {
                return self.err("expected string key in object");
            }
            let key = self.parse_str()?;
            if map.contains_key(&key) {
                return self.err(format!("duplicate object key {key:?}"));
            }
            self.skip_ws();
            if self.bytes.get(self.pos) != Some(&b':') {
                return self.err("expected ':' after object key");
            }
            self.pos += 1;
            let val = self.parse_val()?;
            map.insert(key, val);
            self.skip_ws();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonVal::Obj(map));
                }
                _ => return self.err("expected ',' or '}' in object"),
            }
        }
    }
}

fn req_field<'a>(obj: &'a BTreeMap<String, JsonVal>, key: &str) -> Result<&'a JsonVal, TraceError> {
    obj.get(key)
        .ok_or_else(|| TraceError::InvalidSchema(format!("missing required field {key:?}")))
}

// ---- Trace Sink Abstraction -----------------------------------------------

pub trait TraceSink {
    fn emit_line(&mut self, json_line: &str) -> Result<(), TraceError>;
}

#[derive(Default)]
pub struct NullSink;

impl TraceSink for NullSink {
    fn emit_line(&mut self, _json_line: &str) -> Result<(), TraceError> {
        Ok(())
    }
}

#[derive(Default, Clone, Debug)]
pub struct MemorySink {
    lines: Vec<String>,
}

impl MemorySink {
    pub fn new() -> Self {
        MemorySink { lines: Vec::new() }
    }
    pub fn lines(&self) -> &[String] {
        &self.lines
    }
    pub fn to_jsonl(&self) -> String {
        let mut out = String::new();
        for l in &self.lines {
            out.push_str(l);
            out.push('\n');
        }
        out
    }
}

impl TraceSink for MemorySink {
    fn emit_line(&mut self, json_line: &str) -> Result<(), TraceError> {
        self.lines.push(json_line.to_owned());
        Ok(())
    }
}

pub struct JsonlSink<W: io::Write> {
    writer: W,
}

impl<W: io::Write> JsonlSink<W> {
    pub fn new(writer: W) -> Self {
        JsonlSink { writer }
    }
    pub fn flush(&mut self) -> Result<(), TraceError> {
        self.writer.flush()?;
        Ok(())
    }
    pub fn into_inner(self) -> W {
        self.writer
    }
}

impl<W: io::Write> TraceSink for JsonlSink<W> {
    fn emit_line(&mut self, json_line: &str) -> Result<(), TraceError> {
        self.writer.write_all(json_line.as_bytes())?;
        self.writer.write_all(b"\n")?;
        Ok(())
    }
}

// ---- Model 1: BlockLayout Trace (`specs/BlockLayout.tla`) -----------------

pub const BLOCK_LAYOUT_ACTIONS: &[&str] = &[
    "Init",
    "Prepare",
    "Alloc",
    "BeginStep",
    "Write",
    "EndStep",
    "SlideWindow",
    "KeepStart",
    "DropStart",
    "Match",
    "Finish",
    "Stutter",
];

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BlockLayoutConfig {
    pub lanes: Vec<String>,
    pub full_capacity: u32,
    pub window_capacity: u32,
    pub block_size: u32,
    pub window_size: u32,
    pub max_pos: u32,
    pub max_span: u32,
    pub max_generation: u32,
    pub slot_base: u32,
    pub kvb: u32,
}

impl BlockLayoutConfig {
    pub fn validate(&self) -> Result<(), TraceError> {
        if self.lanes.is_empty() {
            return Err(TraceError::InvalidSchema("block_layout.lanes must be non-empty".into()));
        }
        let uniq: BTreeSet<_> = self.lanes.iter().collect();
        if uniq.len() != self.lanes.len() {
            return Err(TraceError::InvalidSchema("block_layout.lanes contains duplicates".into()));
        }
        if self.full_capacity < 2 || self.window_capacity < 2 {
            return Err(TraceError::InvalidSchema(
                "block_layout capacities must hold at least 2 full and 2 window slots".into(),
            ));
        }
        if self.block_size < 2 || self.window_size == 0 || self.max_pos <= self.block_size {
            return Err(TraceError::InvalidSchema("invalid block_size/window_size/max_pos".into()));
        }
        if self.max_span == 0 || self.max_generation <= 1 {
            return Err(TraceError::InvalidSchema("invalid max_span/max_generation".into()));
        }
        if self.kvb != 1 && self.kvb != 2 {
            return Err(TraceError::InvalidSchema("kvb must be 1 or 2".into()));
        }
        Ok(())
    }

    pub fn num_blocks(&self) -> u32 {
        ((self.max_pos - 1) / self.block_size) + 1
    }

    pub fn to_json_val(&self) -> JsonVal {
        let mut m = BTreeMap::new();
        m.insert("event".into(), JsonVal::str("config"));
        m.insert("model".into(), JsonVal::str("block_layout"));
        m.insert("version".into(), JsonVal::uint(TRACE_SCHEMA_VERSION));
        m.insert(
            "lanes".into(),
            JsonVal::Arr(self.lanes.iter().cloned().map(JsonVal::Str).collect()),
        );
        m.insert("full_capacity".into(), JsonVal::uint(self.full_capacity));
        m.insert("window_capacity".into(), JsonVal::uint(self.window_capacity));
        m.insert("block_size".into(), JsonVal::uint(self.block_size));
        m.insert("window_size".into(), JsonVal::uint(self.window_size));
        m.insert("max_pos".into(), JsonVal::uint(self.max_pos));
        m.insert("max_span".into(), JsonVal::uint(self.max_span));
        m.insert("max_generation".into(), JsonVal::uint(self.max_generation));
        m.insert("slot_base".into(), JsonVal::uint(self.slot_base));
        m.insert("kvb".into(), JsonVal::uint(self.kvb));
        JsonVal::Obj(m)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PoolSlotSnapshot {
    pub slot: u32,
    pub generation: u32,
    pub kind: BlockKind,
    pub block: u32,
    pub length: u32,
    pub data: Vec<u8>,
}

impl PoolSlotSnapshot {
    fn to_json_val(&self) -> JsonVal {
        let mut m = BTreeMap::new();
        m.insert("slot".into(), JsonVal::uint(self.slot));
        m.insert("generation".into(), JsonVal::uint(self.generation));
        m.insert("kind".into(), JsonVal::str(kind_name(self.kind)));
        m.insert("block".into(), JsonVal::uint(self.block));
        m.insert("length".into(), JsonVal::uint(self.length));
        m.insert(
            "data".into(),
            JsonVal::Arr(self.data.iter().map(|&b| JsonVal::uint(b)).collect()),
        );
        JsonVal::Obj(m)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TableEntrySnapshot {
    pub kind: BlockKind,
    pub block: u32,
    pub slot: u32,
    pub generation: u32,
    pub offset: u64,
    pub offset_limbs: OffsetLimbs,
    pub kvb: u32,
}

impl TableEntrySnapshot {
    pub fn from_block_entry(block: BlockIndex, entry: BlockEntry) -> Result<Self, TraceError> {
        let offset_limbs = OffsetLimbs::from_u64(entry.offset())?;
        Ok(TableEntrySnapshot {
            kind: entry.kind(),
            block: block.get(),
            slot: entry.slot().get(),
            generation: entry.generation(),
            offset: entry.offset(),
            offset_limbs,
            kvb: entry.kvb(),
        })
    }

    fn to_json_val(&self) -> JsonVal {
        let mut limbs = BTreeMap::new();
        limbs.insert("hi".into(), JsonVal::uint(self.offset_limbs.hi));
        limbs.insert("mib".into(), JsonVal::uint(self.offset_limbs.mib));
        limbs.insert("sub".into(), JsonVal::uint(self.offset_limbs.sub));

        let mut m = BTreeMap::new();
        m.insert("kind".into(), JsonVal::str(kind_name(self.kind)));
        m.insert("block".into(), JsonVal::uint(self.block));
        m.insert("slot".into(), JsonVal::uint(self.slot));
        m.insert("generation".into(), JsonVal::uint(self.generation));
        m.insert("offset".into(), JsonVal::UInt(self.offset));
        m.insert("offset_limbs".into(), JsonVal::Obj(limbs));
        m.insert("kvb".into(), JsonVal::uint(self.kvb));
        JsonVal::Obj(m)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PinHandleSnapshot {
    pub kind: BlockKind,
    pub slot: u32,
    pub generation: u32,
}

impl Ord for PinHandleSnapshot {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (kind_rank(self.kind), self.slot, self.generation).cmp(&(
            kind_rank(other.kind),
            other.slot,
            other.generation,
        ))
    }
}

impl PartialOrd for PinHandleSnapshot {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PinHandleSnapshot {
    fn to_json_val(&self) -> JsonVal {
        JsonVal::Arr(vec![
            JsonVal::str(kind_name(self.kind)),
            JsonVal::uint(self.slot),
            JsonVal::uint(self.generation),
        ])
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BlockLayoutState {
    pub full_pool: Vec<PoolSlotSnapshot>,
    pub window_pool: Vec<PoolSlotSnapshot>,
    pub table: BTreeMap<String, Vec<TableEntrySnapshot>>,
    pub pos: BTreeMap<String, u32>,
    pub phase: BTreeMap<String, String>,
    pub target: BTreeMap<String, u32>,
    pub span_start: BTreeMap<String, u32>,
    pub budget: BTreeMap<String, u32>,
    pub pin: BTreeMap<String, Vec<PinHandleSnapshot>>,
    pub start_kept: bool,
    pub start_pos: u32,
    pub start_state: u32,
    pub start_table: Vec<TableEntrySnapshot>,
}

impl BlockLayoutState {
    /// Snapshot a `BlockTable` (`src/blocks.rs`) across `0..blocks_per_kind`.
    pub fn snapshot_table(
        table: &BlockTable,
        blocks_per_kind: u32,
    ) -> Result<Vec<TableEntrySnapshot>, TraceError> {
        let mut out = Vec::new();
        for kind in [BlockKind::Full, BlockKind::Window] {
            for b in 0..blocks_per_kind {
                let idx = BlockIndex::new(b);
                if let Some(entry) = table
                    .get(kind, idx)
                    .map_err(|e| TraceError::InvalidSchema(e.to_string()))?
                {
                    out.push(TableEntrySnapshot::from_block_entry(idx, entry)?);
                }
            }
        }
        Ok(out)
    }

    /// Snapshot `BlockPools` (`src/blocks.rs`) paired with logical block indices and position masks.
    pub fn snapshot_pools(
        pools: &BlockPools,
        full_meta: &[(u32, Vec<u8>)],
        win_meta: &[(u32, Vec<u8>)],
    ) -> Result<(Vec<PoolSlotSnapshot>, Vec<PoolSlotSnapshot>), TraceError> {
        if full_meta.len() != pools.full.len() || win_meta.len() != pools.window.len() {
            return Err(TraceError::InvalidSchema("pool metadata length mismatch".into()));
        }
        let mut full_out = Vec::with_capacity(pools.full.len());
        for (i, (block, data)) in full_meta.iter().enumerate() {
            let slot = Slot::new((i + 1) as u32).map_err(|e| TraceError::InvalidSchema(e.to_string()))?;
            let rec = pools.full.record(slot).map_err(|e| TraceError::InvalidSchema(e.to_string()))?;
            full_out.push(PoolSlotSnapshot {
                slot: slot.get(),
                generation: rec.generation(),
                kind: BlockKind::Full,
                block: *block,
                length: rec.length(),
                data: data.clone(),
            });
        }
        let mut win_out = Vec::with_capacity(pools.window.len());
        for (i, (block, data)) in win_meta.iter().enumerate() {
            let slot = Slot::new((i + 1) as u32).map_err(|e| TraceError::InvalidSchema(e.to_string()))?;
            let rec = pools.window.record(slot).map_err(|e| TraceError::InvalidSchema(e.to_string()))?;
            win_out.push(PoolSlotSnapshot {
                slot: slot.get(),
                generation: rec.generation(),
                kind: BlockKind::Window,
                block: *block,
                length: rec.length(),
                data: data.clone(),
            });
        }
        Ok((full_out, win_out))
    }

    pub fn to_json_val(&self) -> JsonVal {
        let mut pool = BTreeMap::new();
        pool.insert(
            "full".into(),
            JsonVal::Arr(self.full_pool.iter().map(PoolSlotSnapshot::to_json_val).collect()),
        );
        pool.insert(
            "window".into(),
            JsonVal::Arr(self.window_pool.iter().map(PoolSlotSnapshot::to_json_val).collect()),
        );

        let table_obj = self
            .table
            .iter()
            .map(|(k, v)| (k.clone(), JsonVal::Arr(v.iter().map(TableEntrySnapshot::to_json_val).collect())))
            .collect();
        let pos_obj = self.pos.iter().map(|(k, &v)| (k.clone(), JsonVal::uint(v))).collect();
        let phase_obj = self.phase.iter().map(|(k, v)| (k.clone(), JsonVal::str(v.clone()))).collect();
        let target_obj = self.target.iter().map(|(k, &v)| (k.clone(), JsonVal::uint(v))).collect();
        let span_obj = self.span_start.iter().map(|(k, &v)| (k.clone(), JsonVal::uint(v))).collect();
        let budget_obj = self.budget.iter().map(|(k, &v)| (k.clone(), JsonVal::uint(v))).collect();
        let pin_obj = self
            .pin
            .iter()
            .map(|(k, v)| (k.clone(), JsonVal::Arr(v.iter().map(PinHandleSnapshot::to_json_val).collect())))
            .collect();

        let mut m = BTreeMap::new();
        m.insert("pool".into(), JsonVal::Obj(pool));
        m.insert("table".into(), JsonVal::Obj(table_obj));
        m.insert("pos".into(), JsonVal::Obj(pos_obj));
        m.insert("phase".into(), JsonVal::Obj(phase_obj));
        m.insert("target".into(), JsonVal::Obj(target_obj));
        m.insert("span_start".into(), JsonVal::Obj(span_obj));
        m.insert("budget".into(), JsonVal::Obj(budget_obj));
        m.insert("pin".into(), JsonVal::Obj(pin_obj));
        m.insert("start_kept".into(), JsonVal::Bool(self.start_kept));
        m.insert("start_pos".into(), JsonVal::uint(self.start_pos));
        m.insert("start_state".into(), JsonVal::uint(self.start_state));
        m.insert(
            "start_table".into(),
            JsonVal::Arr(self.start_table.iter().map(TableEntrySnapshot::to_json_val).collect()),
        );
        JsonVal::Obj(m)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BlockLayoutStep {
    pub seq: u64,
    pub action: String,
    pub args: BTreeMap<String, JsonVal>,
    pub state: BlockLayoutState,
}

impl BlockLayoutStep {
    pub fn to_json_val(&self) -> JsonVal {
        let mut m = BTreeMap::new();
        m.insert("event".into(), JsonVal::str("step"));
        m.insert("seq".into(), JsonVal::UInt(self.seq));
        m.insert("action".into(), JsonVal::str(self.action.clone()));
        m.insert("args".into(), JsonVal::Obj(self.args.clone()));
        m.insert("state".into(), self.state.to_json_val());
        JsonVal::Obj(m)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BlockLayoutTrace {
    pub config: BlockLayoutConfig,
    pub steps: Vec<BlockLayoutStep>,
}

pub fn emit_block_layout_config(sink: &mut dyn TraceSink, cfg: &BlockLayoutConfig) -> Result<(), TraceError> {
    cfg.validate()?;
    sink.emit_line(&cfg.to_json_val().to_json_line())
}

pub fn emit_block_layout_step(sink: &mut dyn TraceSink, step: &BlockLayoutStep) -> Result<(), TraceError> {
    sink.emit_line(&step.to_json_val().to_json_line())
}

fn parse_pool_slots(
    arr: &[JsonVal],
    expected_kind: BlockKind,
    expected_cap: u32,
    block_size: u32,
    num_blocks: u32,
) -> Result<Vec<PoolSlotSnapshot>, TraceError> {
    if arr.len() != expected_cap as usize {
        return Err(TraceError::InvalidSchema(format!(
            "{} pool length {} != expected capacity {}",
            kind_name(expected_kind),
            arr.len(),
            expected_cap
        )));
    }
    let mut out = Vec::with_capacity(arr.len());
    for (idx, item) in arr.iter().enumerate() {
        let obj = item.as_obj()?;
        let slot = req_field(obj, "slot")?.as_u32()?;
        if slot != (idx as u32) + 1 {
            return Err(TraceError::InvalidSchema(format!(
                "pool slot at index {idx} has slot={slot}, expected {}",
                idx + 1
            )));
        }
        let kind = parse_kind(req_field(obj, "kind")?.as_str()?)?;
        if kind != expected_kind {
            return Err(TraceError::InvalidSchema("pool slot kind mismatch".into()));
        }
        let generation = req_field(obj, "generation")?.as_u32()?;
        let block = req_field(obj, "block")?.as_u32()?;
        if block >= num_blocks {
            return Err(TraceError::InvalidSchema(format!("pool block {block} >= {num_blocks}")));
        }
        let length = req_field(obj, "length")?.as_u32()?;
        if length > block_size {
            return Err(TraceError::InvalidSchema(format!("pool length {length} > {block_size}")));
        }
        let data_arr = req_field(obj, "data")?.as_arr()?;
        if data_arr.len() != block_size as usize {
            return Err(TraceError::InvalidSchema("pool data length != block_size".into()));
        }
        let mut data = Vec::with_capacity(data_arr.len());
        for d in data_arr {
            let v = d.as_u32()?;
            if v > 1 {
                return Err(TraceError::InvalidSchema("pool data element must be 0 or 1".into()));
            }
            data.push(v as u8);
        }
        out.push(PoolSlotSnapshot { slot, generation, kind, block, length, data });
    }
    Ok(out)
}

fn parse_table_entries(arr: &[JsonVal], cfg: &BlockLayoutConfig) -> Result<Vec<TableEntrySnapshot>, TraceError> {
    let num_blocks = cfg.num_blocks();
    let mut seen = BTreeSet::new();
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let obj = item.as_obj()?;
        let kind = parse_kind(req_field(obj, "kind")?.as_str()?)?;
        let block = req_field(obj, "block")?.as_u32()?;
        if block >= num_blocks {
            return Err(TraceError::InvalidSchema(format!("table block {block} out of range")));
        }
        let key = (kind_name(kind), block);
        if !seen.insert(key) {
            return Err(TraceError::InvalidSchema(format!("duplicate table entry ({}, {block})", key.0)));
        }
        let slot = req_field(obj, "slot")?.as_u32()?;
        let cap = match kind {
            BlockKind::Full => cfg.full_capacity,
            BlockKind::Window => cfg.window_capacity,
        };
        if slot == 0 || slot > cap {
            return Err(TraceError::InvalidSchema(format!("table slot {slot} out of 1..={cap}")));
        }
        let generation = req_field(obj, "generation")?.as_u32()?;
        let offset = req_field(obj, "offset")?.as_u64()?;
        let limbs_obj = req_field(obj, "offset_limbs")?.as_obj()?;
        let offset_limbs = OffsetLimbs {
            hi: req_field(limbs_obj, "hi")?.as_u32()?,
            mib: req_field(limbs_obj, "mib")?.as_u32()?,
            sub: req_field(limbs_obj, "sub")?.as_u32()?,
        };
        if offset_limbs.to_u64()? != offset {
            return Err(TraceError::InvalidSchema(format!(
                "offset {offset} disagrees with offset_limbs {offset_limbs:?}"
            )));
        }
        let kvb = req_field(obj, "kvb")?.as_u32()?;
        if kvb != 1 && kvb != 2 {
            return Err(TraceError::InvalidSchema(format!("invalid entry kvb {kvb}")));
        }
        out.push(TableEntrySnapshot { kind, block, slot, generation, offset, offset_limbs, kvb });
    }
    Ok(out)
}

pub fn parse_block_layout_jsonl(input: &str) -> Result<BlockLayoutTrace, TraceError> {
    let mut non_empty = input
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty());
    let Some((first_idx, first_line)) = non_empty.next() else {
        return Err(TraceError::InvalidSchema("empty trace input".into()));
    };
    let cfg_val = JsonParser::parse_line(first_idx + 1, first_line)?;
    let cfg_obj = cfg_val.as_obj()?;
    if req_field(cfg_obj, "event")?.as_str()? != "config" {
        return Err(TraceError::InvalidSchema("first line must have event=\"config\"".into()));
    }
    if req_field(cfg_obj, "model")?.as_str()? != "block_layout" {
        return Err(TraceError::InvalidSchema("expected model=\"block_layout\"".into()));
    }
    if req_field(cfg_obj, "version")?.as_u32()? != TRACE_SCHEMA_VERSION {
        return Err(TraceError::InvalidSchema("unsupported trace schema version".into()));
    }
    let lanes: Vec<String> = req_field(cfg_obj, "lanes")?
        .as_arr()?
        .iter()
        .map(|v| v.as_str().map(str::to_owned))
        .collect::<Result<_, _>>()?;
    let config = BlockLayoutConfig {
        lanes,
        full_capacity: req_field(cfg_obj, "full_capacity")?.as_u32()?,
        window_capacity: req_field(cfg_obj, "window_capacity")?.as_u32()?,
        block_size: req_field(cfg_obj, "block_size")?.as_u32()?,
        window_size: req_field(cfg_obj, "window_size")?.as_u32()?,
        max_pos: req_field(cfg_obj, "max_pos")?.as_u32()?,
        max_span: req_field(cfg_obj, "max_span")?.as_u32()?,
        max_generation: req_field(cfg_obj, "max_generation")?.as_u32()?,
        slot_base: req_field(cfg_obj, "slot_base")?.as_u32()?,
        kvb: req_field(cfg_obj, "kvb")?.as_u32()?,
    };
    config.validate()?;
    let lane_set: BTreeSet<&str> = config.lanes.iter().map(String::as_str).collect();
    let num_blocks = config.num_blocks();

    let mut steps = Vec::new();
    for (line_idx, line) in non_empty {
        let val = JsonParser::parse_line(line_idx + 1, line)?;
        let obj = val.as_obj()?;
        if req_field(obj, "event")?.as_str()? != "step" {
            return Err(TraceError::InvalidSchema("expected event=\"step\"".into()));
        }
        let seq = req_field(obj, "seq")?.as_u64()?;
        if seq != steps.len() as u64 {
            return Err(TraceError::InvalidSchema(format!(
                "non-contiguous step seq {seq}, expected {}",
                steps.len()
            )));
        }
        let action = req_field(obj, "action")?.as_str()?.to_owned();
        if !BLOCK_LAYOUT_ACTIONS.contains(&action.as_str()) {
            return Err(TraceError::InvalidSchema(format!("unknown block_layout action {action:?}")));
        }
        if seq == 0 && action != "Init" {
            return Err(TraceError::InvalidSchema("seq 0 must have action=\"Init\"".into()));
        }
        if seq > 0 && action == "Init" {
            return Err(TraceError::InvalidSchema("action=\"Init\" only allowed at seq 0".into()));
        }
        let args = req_field(obj, "args")?.as_obj()?.clone();
        let st_obj = req_field(obj, "state")?.as_obj()?;
        let pool_obj = req_field(st_obj, "pool")?.as_obj()?;
        let full_pool = parse_pool_slots(
            req_field(pool_obj, "full")?.as_arr()?,
            BlockKind::Full,
            config.full_capacity,
            config.block_size,
            num_blocks,
        )?;
        let window_pool = parse_pool_slots(
            req_field(pool_obj, "window")?.as_arr()?,
            BlockKind::Window,
            config.window_capacity,
            config.block_size,
            num_blocks,
        )?;

        let parse_lane_u32_map = |field: &str| -> Result<BTreeMap<String, u32>, TraceError> {
            let m = req_field(st_obj, field)?.as_obj()?;
            if m.len() != lane_set.len() {
                return Err(TraceError::InvalidSchema(format!("{field} lane count mismatch")));
            }
            let mut res = BTreeMap::new();
            for (k, v) in m {
                if !lane_set.contains(k.as_str()) {
                    return Err(TraceError::InvalidSchema(format!("unknown lane {k:?} in {field}")));
                }
                res.insert(k.clone(), v.as_u32()?);
            }
            Ok(res)
        };

        let table_raw = req_field(st_obj, "table")?.as_obj()?;
        if table_raw.len() != lane_set.len() {
            return Err(TraceError::InvalidSchema("table lane count mismatch".into()));
        }
        let mut table = BTreeMap::new();
        for (k, v) in table_raw {
            if !lane_set.contains(k.as_str()) {
                return Err(TraceError::InvalidSchema(format!("unknown lane {k:?} in table")));
            }
            table.insert(k.clone(), parse_table_entries(v.as_arr()?, &config)?);
        }

        let phase_raw = req_field(st_obj, "phase")?.as_obj()?;
        if phase_raw.len() != lane_set.len() {
            return Err(TraceError::InvalidSchema("phase lane count mismatch".into()));
        }
        let mut phase = BTreeMap::new();
        for (k, v) in phase_raw {
            if !lane_set.contains(k.as_str()) {
                return Err(TraceError::InvalidSchema(format!("unknown lane {k:?} in phase")));
            }
            let ph = v.as_str()?;
            if !matches!(ph, "idle" | "prepare" | "step" | "done") {
                return Err(TraceError::InvalidSchema(format!("invalid phase {ph:?}")));
            }
            phase.insert(k.clone(), ph.to_owned());
        }

        let pin_raw = req_field(st_obj, "pin")?.as_obj()?;
        if pin_raw.len() != lane_set.len() {
            return Err(TraceError::InvalidSchema("pin lane count mismatch".into()));
        }
        let mut pin = BTreeMap::new();
        for (k, v) in pin_raw {
            if !lane_set.contains(k.as_str()) {
                return Err(TraceError::InvalidSchema(format!("unknown lane {k:?} in pin")));
            }
            let mut handles = Vec::new();
            for h in v.as_arr()? {
                let tup = h.as_arr()?;
                if tup.len() != 3 {
                    return Err(TraceError::InvalidSchema("pin handle must be [kind, slot, gen]".into()));
                }
                handles.push(PinHandleSnapshot {
                    kind: parse_kind(tup[0].as_str()?)?,
                    slot: tup[1].as_u32()?,
                    generation: tup[2].as_u32()?,
                });
            }
            pin.insert(k.clone(), handles);
        }

        let state = BlockLayoutState {
            full_pool,
            window_pool,
            table,
            pos: parse_lane_u32_map("pos")?,
            phase,
            target: parse_lane_u32_map("target")?,
            span_start: parse_lane_u32_map("span_start")?,
            budget: parse_lane_u32_map("budget")?,
            pin,
            start_kept: req_field(st_obj, "start_kept")?.as_bool()?,
            start_pos: req_field(st_obj, "start_pos")?.as_u32()?,
            start_state: req_field(st_obj, "start_state")?.as_u32()?,
            start_table: parse_table_entries(req_field(st_obj, "start_table")?.as_arr()?, &config)?,
        };
        steps.push(BlockLayoutStep { seq, action, args, state });
    }
    if steps.is_empty() {
        return Err(TraceError::InvalidSchema("trace contains zero step records".into()));
    }
    Ok(BlockLayoutTrace { config, steps })
}

/// Render `(TraceData.tla, TraceCheck.tla, TraceCheck.cfg)` for checking one or more
/// `BlockLayoutTrace` runs sharing the same `BlockLayoutConfig` against `specs/BlockLayout.tla`.
pub fn render_block_layout_tla_bundle(
    traces: &[BlockLayoutTrace],
) -> Result<(String, String, String), TraceError> {
    let Some(first) = traces.first() else {
        return Err(TraceError::InvalidSchema("cannot render empty trace slice".into()));
    };
    let cfg = &first.config;
    cfg.validate()?;
    for t in traces {
        if &t.config != cfg {
            return Err(TraceError::InvalidSchema("all traces in bundle must share identical config".into()));
        }
        if t.steps.is_empty() {
            return Err(TraceError::InvalidSchema("trace has empty steps".into()));
        }
    }

    let num_blocks = cfg.num_blocks();
    let mut data = String::from("---------------------------- MODULE TraceData ----------------------------\n");
    data.push_str("NE == <<0, 0, 0, 0, 0, 0, 0>>\n");
    data.push_str("Traces == <<\n");

    let write_dense_table = |out: &mut String, entries: &[TableEntrySnapshot]| {
        let mut grid: Vec<Vec<Option<&TableEntrySnapshot>>> =
            vec![vec![None; num_blocks as usize]; 2];
        for e in entries {
            let ki = match e.kind {
                BlockKind::Full => 0,
                BlockKind::Window => 1,
            };
            grid[ki][e.block as usize] = Some(e);
        }
        out.push_str("<<");
        for (ki, row) in grid.iter().enumerate() {
            if ki > 0 {
                out.push_str(", ");
            }
            out.push_str("<<");
            for (bi, cell) in row.iter().enumerate() {
                if bi > 0 {
                    out.push_str(", ");
                }
                if let Some(e) = cell {
                    let kcode = if e.kind == BlockKind::Full { 1 } else { 2 };
                    let _ = write!(
                        out,
                        "<<{}, {}, {}, {}, {}, {}, {}>>",
                        e.slot,
                        e.generation,
                        e.offset_limbs.hi,
                        e.offset_limbs.mib,
                        e.offset_limbs.sub,
                        kcode,
                        e.kvb
                    );
                } else {
                    out.push_str("NE");
                }
            }
            out.push_str(">>");
        }
        out.push_str(">>");
    };

    let encode_bl_act = |step: &BlockLayoutStep| -> Result<String, TraceError> {
        let req_lane = |act: &str| -> Result<&str, TraceError> {
            let l = req_field(&step.args, "lane")?.as_str()?;
            if !cfg.lanes.iter().any(|x| x == l) {
                return Err(TraceError::InvalidSchema(format!("{act} unknown lane {l:?}")));
            }
            Ok(l)
        };
        match step.action.as_str() {
            "Init" => Ok("<<\"Init\">>".into()),
            "Prepare" => {
                let l = req_lane("Prepare")?;
                let tok = req_field(&step.args, "tokens")?.as_u32()?;
                Ok(format!("<<\"Prepare\", \"{l}\", {tok}>>"))
            }
            "Alloc" => {
                let l = req_lane("Alloc")?;
                let k = parse_kind(req_field(&step.args, "kind")?.as_str()?)?;
                let blk = req_field(&step.args, "block")?.as_u32()?;
                let slot = req_field(&step.args, "slot")?.as_u32()?;
                Ok(format!("<<\"Alloc\", \"{l}\", \"{}\", {blk}, {slot}>>", kind_name(k)))
            }
            "BeginStep" | "Write" | "EndStep" | "SlideWindow" | "KeepStart" | "Match" | "Finish" => {
                let l = req_lane(&step.action)?;
                Ok(format!("<<\"{}\", \"{l}\">>", step.action))
            }
            "DropStart" => Ok("<<\"DropStart\">>".into()),
            "Stutter" => Ok("<<\"Stutter\">>".into()),
            other => Err(TraceError::InvalidSchema(format!("unknown block_layout action {other:?}"))),
        }
    };

    for (ti, tr) in traces.iter().enumerate() {
        if ti > 0 {
            data.push_str(",\n");
        }
        data.push_str("  <<\n");
        for (si, step) in tr.steps.iter().enumerate() {
            if si > 0 {
                data.push_str(",\n");
            }
            let st = &step.state;
            data.push_str("    <<");
            data.push_str("<<");
            for (pi, p) in st.full_pool.iter().enumerate() {
                if pi > 0 {
                    data.push_str(", ");
                }
                let _ = write!(data, "<<{}, {}, {}, <<", p.generation, p.block, p.length);
                for (di, &dv) in p.data.iter().enumerate() {
                    if di > 0 {
                        data.push_str(", ");
                    }
                    let _ = write!(data, "{dv}");
                }
                data.push_str(">>>>");
            }
            data.push_str(">>, <<");
            for (pi, p) in st.window_pool.iter().enumerate() {
                if pi > 0 {
                    data.push_str(", ");
                }
                let _ = write!(data, "<<{}, {}, {}, <<", p.generation, p.block, p.length);
                for (di, &dv) in p.data.iter().enumerate() {
                    if di > 0 {
                        data.push_str(", ");
                    }
                    let _ = write!(data, "{dv}");
                }
                data.push_str(">>>>");
            }
            data.push_str(">>, <<");
            for (li, lane) in cfg.lanes.iter().enumerate() {
                if li > 0 {
                    data.push_str(", ");
                }
                write_dense_table(&mut data, &st.table[lane]);
            }
            data.push_str(">>, <<");
            for (li, lane) in cfg.lanes.iter().enumerate() {
                if li > 0 {
                    data.push_str(", ");
                }
                let _ = write!(data, "{}", st.pos[lane]);
            }
            data.push_str(">>, <<");
            for (li, lane) in cfg.lanes.iter().enumerate() {
                if li > 0 {
                    data.push_str(", ");
                }
                let _ = write!(data, "\"{}\"", st.phase[lane]);
            }
            data.push_str(">>, <<");
            for (li, lane) in cfg.lanes.iter().enumerate() {
                if li > 0 {
                    data.push_str(", ");
                }
                let _ = write!(data, "{}", st.target[lane]);
            }
            data.push_str(">>, <<");
            for (li, lane) in cfg.lanes.iter().enumerate() {
                if li > 0 {
                    data.push_str(", ");
                }
                let _ = write!(data, "{}", st.span_start[lane]);
            }
            data.push_str(">>, <<");
            for (li, lane) in cfg.lanes.iter().enumerate() {
                if li > 0 {
                    data.push_str(", ");
                }
                let _ = write!(data, "{}", st.budget[lane]);
            }
            data.push_str(">>, <<");
            for (li, lane) in cfg.lanes.iter().enumerate() {
                if li > 0 {
                    data.push_str(", ");
                }
                data.push('{');
                for (hi, h) in st.pin[lane].iter().enumerate() {
                    if hi > 0 {
                        data.push_str(", ");
                    }
                    let _ = write!(data, "<<\"{}\", {}, {}>>", kind_name(h.kind), h.slot, h.generation);
                }
                data.push('}');
            }
            let kept_lit = if st.start_kept { "TRUE" } else { "FALSE" };
            let _ = write!(data, ">>, {kept_lit}, {}, {}, ", st.start_pos, st.start_state);
            write_dense_table(&mut data, &st.start_table);
            let act_lit = encode_bl_act(step)?;
            let _ = write!(data, ", {act_lit}>>");
        }
        data.push_str("\n  >>");
    }
    data.push_str("\n>>\n=============================================================================\n");

    let lane_seq_lit = cfg
        .lanes
        .iter()
        .map(|l| format!("\"{l}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let lane_set_lit = format!("{{{lane_seq_lit}}}");

    let check = format!(
        "---------------------------- MODULE TraceCheck ----------------------------
EXTENDS BlockLayout, TraceData, Sequences

LaneSeq == <<{lane_seq_lit}>>
LaneIdxMap == [l \\in Lanes |-> CHOOSE i \\in 1..Len(LaneSeq) : LaneSeq[i] = l]
LaneIdx(l) == LaneIdxMap[l]
KindIdx(k) == IF k = \"full\" THEN 1 ELSE 2
DecodeKind(c) == IF c = 1 THEN \"full\" ELSE IF c = 2 THEN \"window\" ELSE \"none\"
DecodeEntry(e) ==
  IF e[1] = 0 THEN NoEntry
  ELSE [slot |-> e[1], generation |-> e[2],
        offset |-> [hi |-> e[3], mib |-> e[4], sub |-> e[5]],
        kind |-> DecodeKind(e[6]), kvb |-> e[7]]
LiftPoolCell(kind, c) ==
  [generation |-> c[1], kind |-> kind, block |-> c[2], length |-> c[3],
   data |-> [i \\in 0..(B-1) |-> c[4][i + 1]]]
LiftTable(raw) == [k \\in Keys |-> DecodeEntry(raw[KindIdx(k[1])][k[2] + 1])]
Lift(s) == [
  pool       |-> [kind \\in Kinds |-> [slot \\in Slots(kind) |->
                   IF kind = \"full\" THEN LiftPoolCell(\"full\", s[1][slot])
                   ELSE LiftPoolCell(\"window\", s[2][slot])]],
  table      |-> [l \\in Lanes |-> LiftTable(s[3][LaneIdx(l)])],
  pos        |-> [l \\in Lanes |-> s[4][LaneIdx(l)]],
  phase      |-> [l \\in Lanes |-> s[5][LaneIdx(l)]],
  target     |-> [l \\in Lanes |-> s[6][LaneIdx(l)]],
  spanStart  |-> [l \\in Lanes |-> s[7][LaneIdx(l)]],
  budget     |-> [l \\in Lanes |-> s[8][LaneIdx(l)]],
  pin        |-> [l \\in Lanes |-> s[9][LaneIdx(l)]],
  startKept  |-> s[10],
  startPos   |-> s[11],
  startState |-> s[12],
  startTable |-> LiftTable(s[13])
]

VARIABLES traceIdx, stepIdx

Matches(st) ==
  /\\ pool = st.pool /\\ table = st.table /\\ pos = st.pos /\\ phase = st.phase
  /\\ target = st.target /\\ spanStart = st.spanStart /\\ budget = st.budget /\\ pin = st.pin
  /\\ startKept = st.startKept /\\ startPos = st.startPos
  /\\ startTable = st.startTable /\\ startState = st.startState

ReplayInit ==
  /\\ traceIdx \\in 1..Len(Traces)
  /\\ stepIdx = 1
  /\\ Matches(Lift(Traces[traceIdx][1]))

ReplayNext ==
  /\\ stepIdx < Len(Traces[traceIdx])
  /\\ UNCHANGED traceIdx
  /\\ stepIdx' = stepIdx + 1
  /\\ LET nxt == Lift(Traces[traceIdx][stepIdx + 1]) IN
       /\\ pool' = nxt.pool /\\ table' = nxt.table /\\ pos' = nxt.pos /\\ phase' = nxt.phase
       /\\ target' = nxt.target /\\ spanStart' = nxt.spanStart
       /\\ budget' = nxt.budget /\\ pin' = nxt.pin
       /\\ startKept' = nxt.startKept /\\ startPos' = nxt.startPos
       /\\ startTable' = nxt.startTable /\\ startState' = nxt.startState

ReplaySpec == ReplayInit /\\ [][ReplayNext]_<<vars, traceIdx, stepIdx>>
InitConforms == (stepIdx = 1) => Init
StepConforms == [][Next]_vars

DispatchStep(act) ==
  IF act[1] = \"Prepare\"          THEN Prepare(act[2], act[3])
  ELSE IF act[1] = \"Alloc\"       THEN Alloc(act[2], <<act[3], act[4]>>, act[5])
  ELSE IF act[1] = \"BeginStep\"   THEN BeginStep(act[2])
  ELSE IF act[1] = \"Write\"       THEN Write(act[2])
  ELSE IF act[1] = \"EndStep\"     THEN EndStep(act[2])
  ELSE IF act[1] = \"SlideWindow\" THEN SlideWindow(act[2])
  ELSE IF act[1] = \"KeepStart\"   THEN KeepStart(act[2])
  ELSE IF act[1] = \"DropStart\"   THEN DropStart
  ELSE IF act[1] = \"Match\"       THEN Match(act[2])
  ELSE IF act[1] = \"Finish\"      THEN Finish(act[2])
  ELSE IF act[1] = \"Stutter\"     THEN UNCHANGED vars
  ELSE FALSE

StepActionCheck ==
  (stepIdx' = stepIdx + 1) =>
    \\E act \\in {{Traces[traceIdx][stepIdx + 1][14]}} : DispatchStep(act)

ActionConforms == [][StepActionCheck]_<<vars, traceIdx, stepIdx>>
=============================================================================
"
    );

    let cfg_file = format!(
        "SPECIFICATION ReplaySpec
CHECK_DEADLOCK FALSE
CONSTANTS
 Lanes = {lane_set_lit}
 FullCapacity = {}
 WindowCapacity = {}
 B = {}
 W = {}
 MaxPos = {}
 MaxSpan = {}
 MaxGeneration = {}
 SlotBase = {}
 Kvb = {}
 Bug = \"none\"
INVARIANTS
 TypeOK
 InitConforms
 GenerationMatches
 WindowCovered
 FullCovered
 SpanPinned
 StartRetained
 TailPrivateZero
 TableIndex
 Address64
PROPERTIES
 StepConforms
 ActionConforms
",
        cfg.full_capacity,
        cfg.window_capacity,
        cfg.block_size,
        cfg.window_size,
        cfg.max_pos,
        cfg.max_span,
        cfg.max_generation,
        cfg.slot_base,
        cfg.kvb,
    );

    Ok((data, check, cfg_file))
}

// ---- Model 2: CacheScheduler Trace (`specs/CacheScheduler.tla`) -----------

pub const CACHE_SCHEDULER_ACTIONS: &[&str] = &[
    "Init",
    "Arrive",
    "Refuse429",
    "Admit",
    "BeginRestore",
    "RestoreBlock",
    "RestoreStartState",
    "FinishRestore",
    "EvictCleanOrUnrefMem",
    "StartColdWriteback",
    "EvictCleanStartState",
    "SpillStartState",
    "ReclaimColdDisk",
    "DropUnreferenced",
    "SlideWindow",
    "FinishReady",
    "BuildStep",
    "GpuExecute",
    "CompleteStepIO",
    "FinishStep",
    "Cancel",
    "Stutter",
];

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CacheBlockId {
    Tree { prefix: String, kind: BlockKind, idx: u32 },
    Priv { lane: String, kind: BlockKind, idx: u32 },
}

impl Ord for CacheBlockId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (
                CacheBlockId::Tree { prefix: p1, kind: k1, idx: i1 },
                CacheBlockId::Tree { prefix: p2, kind: k2, idx: i2 },
            ) => (0u8, p1, kind_rank(*k1), i1).cmp(&(0u8, p2, kind_rank(*k2), i2)),
            (CacheBlockId::Tree { .. }, CacheBlockId::Priv { .. }) => std::cmp::Ordering::Less,
            (CacheBlockId::Priv { .. }, CacheBlockId::Tree { .. }) => std::cmp::Ordering::Greater,
            (
                CacheBlockId::Priv { lane: l1, kind: k1, idx: i1 },
                CacheBlockId::Priv { lane: l2, kind: k2, idx: i2 },
            ) => (1u8, l1, kind_rank(*k1), i1).cmp(&(1u8, l2, kind_rank(*k2), i2)),
        }
    }
}

impl PartialOrd for CacheBlockId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl CacheBlockId {
    pub fn kind(&self) -> BlockKind {
        match self {
            CacheBlockId::Tree { kind, .. } | CacheBlockId::Priv { kind, .. } => *kind,
        }
    }

    pub fn encode(&self) -> String {
        match self {
            CacheBlockId::Tree { prefix, kind, idx } => format!("tree:{prefix}:{}:{idx}", kind_name(*kind)),
            CacheBlockId::Priv { lane, kind, idx } => format!("priv:{lane}:{}:{idx}", kind_name(*kind)),
        }
    }

    pub fn decode(s: &str) -> Result<CacheBlockId, TraceError> {
        let parts: Vec<&str> = s.split(':').collect();
        match parts.as_slice() {
            ["tree", p, k, idx_s] => {
                let kind = parse_kind(k)?;
                let idx = idx_s
                    .parse::<u32>()
                    .map_err(|_| TraceError::InvalidSchema(format!("bad tree block idx in {s:?}")))?;
                Ok(CacheBlockId::Tree { prefix: (*p).to_owned(), kind, idx })
            }
            ["priv", l, k, idx_s] => {
                let kind = parse_kind(k)?;
                let idx = idx_s
                    .parse::<u32>()
                    .map_err(|_| TraceError::InvalidSchema(format!("bad priv block idx in {s:?}")))?;
                Ok(CacheBlockId::Priv { lane: (*l).to_owned(), kind, idx })
            }
            _ => Err(TraceError::InvalidSchema(format!("invalid CacheBlockId {s:?}"))),
        }
    }

    fn to_tla_tuple(&self) -> String {
        match self {
            CacheBlockId::Tree { prefix, kind, idx } => {
                format!("<<\"tree\", \"{prefix}\", \"{}\", {idx}>>", kind_name(*kind))
            }
            CacheBlockId::Priv { lane, kind, idx } => {
                format!("<<\"priv\", \"{lane}\", \"{}\", {idx}>>", kind_name(*kind))
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CacheSchedulerConfig {
    pub lanes: Vec<String>,
    pub prefixes: Vec<String>,
    pub request_lengths: Vec<u32>,
    pub block_size: u32,
    pub window_size: u32,
    pub max_pos: u32,
    pub max_span: u32,
    pub step_token_budget: u32,
    pub mem_budget: u32,
    pub disk_budget: u32,
    pub max_queue: u32,
    pub max_bypass: u32,
}

impl CacheSchedulerConfig {
    pub fn validate(&self) -> Result<(), TraceError> {
        if self.lanes.is_empty() || self.prefixes.is_empty() || self.request_lengths.is_empty() {
            return Err(TraceError::InvalidSchema("lanes, prefixes, and request_lengths must be non-empty".into()));
        }
        if self.block_size == 0 || self.window_size < self.block_size {
            return Err(TraceError::InvalidSchema("invalid block_size/window_size".into()));
        }
        if self.max_pos < self.block_size || !self.max_pos.is_multiple_of(self.block_size) {
            return Err(TraceError::InvalidSchema("max_pos must be a positive multiple of block_size".into()));
        }
        for &r in &self.request_lengths {
            if r == 0 || r > self.max_pos || r % self.block_size != 0 {
                return Err(TraceError::InvalidSchema(format!("invalid request_length {r}")));
            }
        }
        if self.max_span == 0
            || self.max_span > self.max_pos
            || self.step_token_budget < self.max_span
            || self.mem_budget < 2
            || self.disk_budget < 2
            || self.max_bypass < 1
        {
            return Err(TraceError::InvalidSchema("invalid scheduler budget constants".into()));
        }
        Ok(())
    }

    pub fn num_blocks(&self) -> u32 {
        self.max_pos / self.block_size
    }

    pub fn boundaries(&self) -> Vec<u32> {
        (1..=self.num_blocks()).map(|k| k * self.block_size).collect()
    }

    pub fn default_prefix(&self) -> &str {
        self.prefixes
            .iter()
            .min_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)))
            .expect("validated non-empty prefixes")
            .as_str()
    }

    pub fn default_target(&self) -> u32 {
        *self.request_lengths.iter().min().expect("validated non-empty request_lengths")
    }

    pub fn all_block_ids(&self) -> Vec<CacheBlockId> {
        let mut out = Vec::new();
        for p in &self.prefixes {
            for kind in [BlockKind::Full, BlockKind::Window] {
                for idx in 0..self.num_blocks() {
                    out.push(CacheBlockId::Tree { prefix: p.clone(), kind, idx });
                }
            }
        }
        for l in &self.lanes {
            for kind in [BlockKind::Full, BlockKind::Window] {
                for idx in 0..self.num_blocks() {
                    out.push(CacheBlockId::Priv { lane: l.clone(), kind, idx });
                }
            }
        }
        out
    }

    pub fn to_json_val(&self) -> JsonVal {
        let mut m = BTreeMap::new();
        m.insert("event".into(), JsonVal::str("config"));
        m.insert("model".into(), JsonVal::str("cache_scheduler"));
        m.insert("version".into(), JsonVal::uint(TRACE_SCHEMA_VERSION));
        m.insert(
            "lanes".into(),
            JsonVal::Arr(self.lanes.iter().cloned().map(JsonVal::Str).collect()),
        );
        m.insert(
            "prefixes".into(),
            JsonVal::Arr(self.prefixes.iter().cloned().map(JsonVal::Str).collect()),
        );
        m.insert(
            "request_lengths".into(),
            JsonVal::Arr(self.request_lengths.iter().map(|&r| JsonVal::uint(r)).collect()),
        );
        m.insert("block_size".into(), JsonVal::uint(self.block_size));
        m.insert("window_size".into(), JsonVal::uint(self.window_size));
        m.insert("max_pos".into(), JsonVal::uint(self.max_pos));
        m.insert("max_span".into(), JsonVal::uint(self.max_span));
        m.insert("step_token_budget".into(), JsonVal::uint(self.step_token_budget));
        m.insert("mem_budget".into(), JsonVal::uint(self.mem_budget));
        m.insert("disk_budget".into(), JsonVal::uint(self.disk_budget));
        m.insert("max_queue".into(), JsonVal::uint(self.max_queue));
        m.insert("max_bypass".into(), JsonVal::uint(self.max_bypass));
        JsonVal::Obj(m)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CacheBlockSnapshot {
    pub id: CacheBlockId,
    pub tier: String,
    pub sealed: bool,
    pub idx: u32,
    pub length: u32,
    pub data: Vec<u8>,
    pub ref_count: u32,
    pub last_used_tick: u64,
}

impl CacheBlockSnapshot {
    fn to_json_val(&self) -> JsonVal {
        let mut m = BTreeMap::new();
        m.insert("id".into(), JsonVal::str(self.id.encode()));
        m.insert("tier".into(), JsonVal::str(self.tier.clone()));
        m.insert("sealed".into(), JsonVal::Bool(self.sealed));
        m.insert("idx".into(), JsonVal::uint(self.idx));
        m.insert("length".into(), JsonVal::uint(self.length));
        m.insert(
            "data".into(),
            JsonVal::Arr(self.data.iter().map(|&d| JsonVal::uint(d)).collect()),
        );
        m.insert("ref_count".into(), JsonVal::uint(self.ref_count));
        m.insert("last_used_tick".into(), JsonVal::UInt(self.last_used_tick));
        JsonVal::Obj(m)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct StartPointSnapshot {
    pub prefix: String,
    pub boundary: u32,
    pub state_kept: bool,
    pub state_tier: String,
    pub full: Vec<CacheBlockId>,
    pub win: Vec<CacheBlockId>,
    pub last_used_tick: u64,
}

impl StartPointSnapshot {
    fn to_json_val(&self) -> JsonVal {
        let mut m = BTreeMap::new();
        m.insert("prefix".into(), JsonVal::str(self.prefix.clone()));
        m.insert("boundary".into(), JsonVal::uint(self.boundary));
        m.insert("state_kept".into(), JsonVal::Bool(self.state_kept));
        m.insert("state_tier".into(), JsonVal::str(self.state_tier.clone()));
        m.insert(
            "full".into(),
            JsonVal::Arr(self.full.iter().map(|b| JsonVal::str(b.encode())).collect()),
        );
        m.insert(
            "win".into(),
            JsonVal::Arr(self.win.iter().map(|b| JsonVal::str(b.encode())).collect()),
        );
        m.insert("last_used_tick".into(), JsonVal::UInt(self.last_used_tick));
        JsonVal::Obj(m)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CacheLaneSnapshot {
    pub lane: String,
    pub state: String,
    pub prefix: String,
    pub target: u32,
    pub pos: u32,
    pub full: Vec<Option<CacheBlockId>>,
    pub win: Vec<Option<CacheBlockId>>,
    pub reservation: u32,
    pub admit_budget: u32,
    pub wait_ticks: u32,
    pub ghost_bypass: u32,
}

impl CacheLaneSnapshot {
    fn to_json_val(&self) -> JsonVal {
        let encode_opt = |opt: &Option<CacheBlockId>| match opt {
            Some(id) => JsonVal::str(id.encode()),
            None => JsonVal::Null,
        };
        let mut m = BTreeMap::new();
        m.insert("lane".into(), JsonVal::str(self.lane.clone()));
        m.insert("state".into(), JsonVal::str(self.state.clone()));
        m.insert("prefix".into(), JsonVal::str(self.prefix.clone()));
        m.insert("target".into(), JsonVal::uint(self.target));
        m.insert("pos".into(), JsonVal::uint(self.pos));
        m.insert("full".into(), JsonVal::Arr(self.full.iter().map(encode_opt).collect()));
        m.insert("win".into(), JsonVal::Arr(self.win.iter().map(encode_opt).collect()));
        m.insert("reservation".into(), JsonVal::uint(self.reservation));
        m.insert("admit_budget".into(), JsonVal::uint(self.admit_budget));
        m.insert("wait_ticks".into(), JsonVal::uint(self.wait_ticks));
        m.insert("ghost_bypass".into(), JsonVal::uint(self.ghost_bypass));
        JsonVal::Obj(m)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CacheSchedulerState {
    pub blocks: Vec<CacheBlockSnapshot>,
    pub start_points: Vec<StartPointSnapshot>,
    pub lanes: Vec<CacheLaneSnapshot>,
    pub queue_seq: Vec<String>,
    pub step_active: bool,
    pub step_lanes: Vec<String>,
    pub step_span: BTreeMap<String, u32>,
    pub step_pins: Vec<CacheBlockId>,
    pub step_io: Vec<CacheBlockId>,
    pub gpu_done: bool,
}

impl CacheSchedulerState {
    pub fn to_json_val(&self) -> JsonVal {
        let span_obj = self
            .step_span
            .iter()
            .map(|(k, &v)| (k.clone(), JsonVal::uint(v)))
            .collect();
        let mut m = BTreeMap::new();
        m.insert(
            "blocks".into(),
            JsonVal::Arr(self.blocks.iter().map(CacheBlockSnapshot::to_json_val).collect()),
        );
        m.insert(
            "start_points".into(),
            JsonVal::Arr(self.start_points.iter().map(StartPointSnapshot::to_json_val).collect()),
        );
        m.insert(
            "lanes".into(),
            JsonVal::Arr(self.lanes.iter().map(CacheLaneSnapshot::to_json_val).collect()),
        );
        m.insert(
            "queue_seq".into(),
            JsonVal::Arr(self.queue_seq.iter().cloned().map(JsonVal::Str).collect()),
        );
        m.insert("step_active".into(), JsonVal::Bool(self.step_active));
        m.insert(
            "step_lanes".into(),
            JsonVal::Arr(self.step_lanes.iter().cloned().map(JsonVal::Str).collect()),
        );
        m.insert("step_span".into(), JsonVal::Obj(span_obj));
        m.insert(
            "step_pins".into(),
            JsonVal::Arr(self.step_pins.iter().map(|b| JsonVal::str(b.encode())).collect()),
        );
        m.insert(
            "step_io".into(),
            JsonVal::Arr(self.step_io.iter().map(|b| JsonVal::str(b.encode())).collect()),
        );
        m.insert("gpu_done".into(), JsonVal::Bool(self.gpu_done));
        JsonVal::Obj(m)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CacheSchedulerStep {
    pub seq: u64,
    pub action: String,
    pub args: BTreeMap<String, JsonVal>,
    pub state: CacheSchedulerState,
}

impl CacheSchedulerStep {
    pub fn to_json_val(&self) -> JsonVal {
        let mut m = BTreeMap::new();
        m.insert("event".into(), JsonVal::str("step"));
        m.insert("seq".into(), JsonVal::UInt(self.seq));
        m.insert("action".into(), JsonVal::str(self.action.clone()));
        m.insert("args".into(), JsonVal::Obj(self.args.clone()));
        m.insert("state".into(), self.state.to_json_val());
        JsonVal::Obj(m)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CacheSchedulerTrace {
    pub config: CacheSchedulerConfig,
    pub steps: Vec<CacheSchedulerStep>,
}

pub fn emit_cache_scheduler_config(
    sink: &mut dyn TraceSink,
    cfg: &CacheSchedulerConfig,
) -> Result<(), TraceError> {
    cfg.validate()?;
    sink.emit_line(&cfg.to_json_val().to_json_line())
}

pub fn emit_cache_scheduler_step(
    sink: &mut dyn TraceSink,
    step: &CacheSchedulerStep,
) -> Result<(), TraceError> {
    sink.emit_line(&step.to_json_val().to_json_line())
}

pub fn parse_cache_scheduler_jsonl(input: &str) -> Result<CacheSchedulerTrace, TraceError> {
    let mut non_empty = input
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty());
    let Some((first_idx, first_line)) = non_empty.next() else {
        return Err(TraceError::InvalidSchema("empty trace input".into()));
    };
    let cfg_val = JsonParser::parse_line(first_idx + 1, first_line)?;
    let cfg_obj = cfg_val.as_obj()?;
    if req_field(cfg_obj, "event")?.as_str()? != "config" {
        return Err(TraceError::InvalidSchema("first line must have event=\"config\"".into()));
    }
    if req_field(cfg_obj, "model")?.as_str()? != "cache_scheduler" {
        return Err(TraceError::InvalidSchema("expected model=\"cache_scheduler\"".into()));
    }
    if req_field(cfg_obj, "version")?.as_u32()? != TRACE_SCHEMA_VERSION {
        return Err(TraceError::InvalidSchema("unsupported trace schema version".into()));
    }
    let config = CacheSchedulerConfig {
        lanes: req_field(cfg_obj, "lanes")?
            .as_arr()?
            .iter()
            .map(|v| v.as_str().map(str::to_owned))
            .collect::<Result<_, _>>()?,
        prefixes: req_field(cfg_obj, "prefixes")?
            .as_arr()?
            .iter()
            .map(|v| v.as_str().map(str::to_owned))
            .collect::<Result<_, _>>()?,
        request_lengths: req_field(cfg_obj, "request_lengths")?
            .as_arr()?
            .iter()
            .map(JsonVal::as_u32)
            .collect::<Result<_, _>>()?,
        block_size: req_field(cfg_obj, "block_size")?.as_u32()?,
        window_size: req_field(cfg_obj, "window_size")?.as_u32()?,
        max_pos: req_field(cfg_obj, "max_pos")?.as_u32()?,
        max_span: req_field(cfg_obj, "max_span")?.as_u32()?,
        step_token_budget: req_field(cfg_obj, "step_token_budget")?.as_u32()?,
        mem_budget: req_field(cfg_obj, "mem_budget")?.as_u32()?,
        disk_budget: req_field(cfg_obj, "disk_budget")?.as_u32()?,
        max_queue: req_field(cfg_obj, "max_queue")?.as_u32()?,
        max_bypass: req_field(cfg_obj, "max_bypass")?.as_u32()?,
    };
    config.validate()?;
    let valid_ids: BTreeSet<CacheBlockId> = config.all_block_ids().into_iter().collect();
    let num_blocks = config.num_blocks() as usize;

    let parse_id = |s: &str| -> Result<CacheBlockId, TraceError> {
        let id = CacheBlockId::decode(s)?;
        if !valid_ids.contains(&id) {
            return Err(TraceError::InvalidSchema(format!("block id {s:?} outside declared universe")));
        }
        Ok(id)
    };

    let mut steps = Vec::new();
    for (line_idx, line) in non_empty {
        let val = JsonParser::parse_line(line_idx + 1, line)?;
        let obj = val.as_obj()?;
        if req_field(obj, "event")?.as_str()? != "step" {
            return Err(TraceError::InvalidSchema("expected event=\"step\"".into()));
        }
        let seq = req_field(obj, "seq")?.as_u64()?;
        if seq != steps.len() as u64 {
            return Err(TraceError::InvalidSchema(format!(
                "non-contiguous step seq {seq}, expected {}",
                steps.len()
            )));
        }
        let action = req_field(obj, "action")?.as_str()?.to_owned();
        if !CACHE_SCHEDULER_ACTIONS.contains(&action.as_str()) {
            return Err(TraceError::InvalidSchema(format!("unknown cache_scheduler action {action:?}")));
        }
        if seq == 0 && action != "Init" {
            return Err(TraceError::InvalidSchema("seq 0 must have action=\"Init\"".into()));
        }
        if seq > 0 && action == "Init" {
            return Err(TraceError::InvalidSchema("action=\"Init\" only allowed at seq 0".into()));
        }
        let args = req_field(obj, "args")?.as_obj()?.clone();
        let st_obj = req_field(obj, "state")?.as_obj()?;

        let mut blocks = Vec::new();
        let mut seen_blocks = BTreeSet::new();
        for item in req_field(st_obj, "blocks")?.as_arr()? {
            let bobj = item.as_obj()?;
            let id = parse_id(req_field(bobj, "id")?.as_str()?)?;
            if !seen_blocks.insert(id.clone()) {
                return Err(TraceError::InvalidSchema("duplicate block record in step".into()));
            }
            let tier = req_field(bobj, "tier")?.as_str()?.to_owned();
            if !matches!(tier.as_str(), "none" | "memory" | "disk" | "both") {
                return Err(TraceError::InvalidSchema(format!("invalid block tier {tier:?}")));
            }
            let sealed = req_field(bobj, "sealed")?.as_bool()?;
            let idx = req_field(bobj, "idx")?.as_u32()?;
            let length = req_field(bobj, "length")?.as_u32()?;
            let data_arr = req_field(bobj, "data")?.as_arr()?;
            if data_arr.len() != config.block_size as usize {
                return Err(TraceError::InvalidSchema("block data length != block_size".into()));
            }
            let mut data = Vec::with_capacity(data_arr.len());
            for d in data_arr {
                let v = d.as_u32()?;
                if v > 1 {
                    return Err(TraceError::InvalidSchema("block data entry must be 0 or 1".into()));
                }
                data.push(v as u8);
            }
            let ref_count = req_field(bobj, "ref_count")?.as_u32()?;
            let last_used_tick = req_field(bobj, "last_used_tick")?.as_u64()?;
            blocks.push(CacheBlockSnapshot {
                id,
                tier,
                sealed,
                idx,
                length,
                data,
                ref_count,
                last_used_tick,
            });
        }

        let valid_boundaries: BTreeSet<u32> = config.boundaries().into_iter().collect();
        let mut start_points = Vec::new();
        let mut seen_starts = BTreeSet::new();
        for item in req_field(st_obj, "start_points")?.as_arr()? {
            let sobj = item.as_obj()?;
            let prefix = req_field(sobj, "prefix")?.as_str()?.to_owned();
            if !config.prefixes.contains(&prefix) {
                return Err(TraceError::InvalidSchema(format!(
                    "start_point unknown prefix {prefix:?}"
                )));
            }
            let boundary = req_field(sobj, "boundary")?.as_u32()?;
            if !valid_boundaries.contains(&boundary) {
                return Err(TraceError::InvalidSchema(format!(
                    "start_point invalid boundary {boundary}"
                )));
            }
            if !seen_starts.insert((prefix.clone(), boundary)) {
                return Err(TraceError::InvalidSchema("duplicate start_point record in step".into()));
            }
            let state_kept = req_field(sobj, "state_kept")?.as_bool()?;
            let state_tier = req_field(sobj, "state_tier")?.as_str()?.to_owned();
            if !matches!(state_tier.as_str(), "none" | "memory" | "disk" | "both") {
                return Err(TraceError::InvalidSchema(format!(
                    "invalid start_point state_tier {state_tier:?}"
                )));
            }
            let last_used_tick = req_field(sobj, "last_used_tick")?.as_u64()?;
            let full = req_field(sobj, "full")?
                .as_arr()?
                .iter()
                .map(|v| parse_id(v.as_str()?))
                .collect::<Result<Vec<_>, _>>()?;
            let win = req_field(sobj, "win")?
                .as_arr()?
                .iter()
                .map(|v| parse_id(v.as_str()?))
                .collect::<Result<Vec<_>, _>>()?;
            start_points.push(StartPointSnapshot {
                prefix,
                boundary,
                state_kept,
                state_tier,
                full,
                win,
                last_used_tick,
            });
        }

        let lanes_arr = req_field(st_obj, "lanes")?.as_arr()?;
        if lanes_arr.len() != config.lanes.len() {
            return Err(TraceError::InvalidSchema("lanes length mismatch".into()));
        }
        let mut lanes = Vec::with_capacity(lanes_arr.len());
        for (li, item) in lanes_arr.iter().enumerate() {
            let lobj = item.as_obj()?;
            let lane = req_field(lobj, "lane")?.as_str()?.to_owned();
            if lane != config.lanes[li] {
                return Err(TraceError::InvalidSchema("lanes array must follow config.lanes order".into()));
            }
            let parse_blk_map = |arr: &[JsonVal]| -> Result<Vec<Option<CacheBlockId>>, TraceError> {
                if arr.len() != num_blocks {
                    return Err(TraceError::InvalidSchema("lane block map length != num_blocks".into()));
                }
                arr.iter()
                    .map(|v| match v {
                        JsonVal::Null => Ok(None),
                        JsonVal::Str(s) if s.is_empty() => Ok(None),
                        JsonVal::Str(s) => parse_id(s).map(Some),
                        _ => Err(TraceError::InvalidSchema("lane block map entry must be null or id".into())),
                    })
                    .collect()
            };
            let state_str = req_field(lobj, "state")?.as_str()?.to_owned();
            if !matches!(
                state_str.as_str(),
                "Idle"
                    | "Refused"
                    | "Queued"
                    | "Admitted"
                    | "Restoring"
                    | "Ready"
                    | "InStep"
                    | "Cancelling"
                    | "Cancelled"
                    | "Finished"
            ) {
                return Err(TraceError::InvalidSchema(format!("invalid lane state {state_str:?}")));
            }
            let pfx_str = req_field(lobj, "prefix")?.as_str()?.to_owned();
            if !config.prefixes.contains(&pfx_str) {
                return Err(TraceError::InvalidSchema(format!("lane unknown prefix {pfx_str:?}")));
            }
            let wait_ticks = req_field(lobj, "wait_ticks")?.as_u32()?;
            let ghost_bypass = req_field(lobj, "ghost_bypass")?.as_u32()?;
            lanes.push(CacheLaneSnapshot {
                lane,
                state: state_str,
                prefix: pfx_str,
                target: req_field(lobj, "target")?.as_u32()?,
                pos: req_field(lobj, "pos")?.as_u32()?,
                full: parse_blk_map(req_field(lobj, "full")?.as_arr()?)?,
                win: parse_blk_map(req_field(lobj, "win")?.as_arr()?)?,
                reservation: req_field(lobj, "reservation")?.as_u32()?,
                admit_budget: req_field(lobj, "admit_budget")?.as_u32()?,
                wait_ticks,
                ghost_bypass,
            });
        }

        let step_lanes = req_field(st_obj, "step_lanes")?
            .as_arr()?
            .iter()
            .map(|v| {
                let s = v.as_str()?.to_owned();
                if !config.lanes.contains(&s) {
                    return Err(TraceError::InvalidSchema(format!("step_lanes unknown lane {s:?}")));
                }
                Ok(s)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let step_span_obj = req_field(st_obj, "step_span")?.as_obj()?;
        if step_span_obj.len() != config.lanes.len() {
            return Err(TraceError::InvalidSchema("step_span must contain every declared lane".into()));
        }
        let mut step_span = BTreeMap::new();
        for (k, v) in step_span_obj {
            if !config.lanes.contains(k) {
                return Err(TraceError::InvalidSchema(format!("step_span unknown lane {k:?}")));
            }
            step_span.insert(k.clone(), v.as_u32()?);
        }
        let step_pins = req_field(st_obj, "step_pins")?
            .as_arr()?
            .iter()
            .map(|v| parse_id(v.as_str()?))
            .collect::<Result<Vec<_>, _>>()?;
        let step_io = req_field(st_obj, "step_io")?
            .as_arr()?
            .iter()
            .map(|v| parse_id(v.as_str()?))
            .collect::<Result<Vec<_>, _>>()?;

        let queue_seq = req_field(st_obj, "queue_seq")?
            .as_arr()?
            .iter()
            .map(|x| {
                let s = x.as_str()?.to_owned();
                if !config.lanes.contains(&s) {
                    return Err(TraceError::InvalidSchema(format!("queue_seq unknown lane {s:?}")));
                }
                Ok(s)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let state = CacheSchedulerState {
            blocks,
            start_points,
            lanes,
            queue_seq,
            step_active: req_field(st_obj, "step_active")?.as_bool()?,
            step_lanes,
            step_span,
            step_pins,
            step_io,
            gpu_done: req_field(st_obj, "gpu_done")?.as_bool()?,
        };
        steps.push(CacheSchedulerStep { seq, action, args, state });
    }
    if steps.is_empty() {
        return Err(TraceError::InvalidSchema("trace contains zero step records".into()));
    }
    Ok(CacheSchedulerTrace { config, steps })
}

/// Render `(TraceData.tla, TraceCheck.tla, TraceCheck.cfg)` for checking one or more
/// `CacheSchedulerTrace` runs sharing the same `CacheSchedulerConfig` against `specs/CacheScheduler.tla`.
pub fn render_cache_scheduler_tla_bundle(
    traces: &[CacheSchedulerTrace],
) -> Result<(String, String, String), TraceError> {
    let Some(first) = traces.first() else {
        return Err(TraceError::InvalidSchema("cannot render empty trace slice".into()));
    };
    let cfg = &first.config;
    cfg.validate()?;
    for t in traces {
        if &t.config != cfg {
            return Err(TraceError::InvalidSchema("all traces in bundle must share identical config".into()));
        }
        if t.steps.is_empty() {
            return Err(TraceError::InvalidSchema("trace has empty steps".into()));
        }
    }

    let all_ids = cfg.all_block_ids();
    let id_to_code: BTreeMap<CacheBlockId, usize> = all_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.clone(), i + 1))
        .collect();
    let boundaries = cfg.boundaries();

    let empty_data = (0..cfg.block_size).map(|_| "0").collect::<Vec<_>>().join(", ");
    let mut data = String::from("---------------------------- MODULE TraceData ----------------------------\n");
    let _ = writeln!(data, "ED == <<{empty_data}>>");
    data.push_str("EB == <<0, FALSE, 0, 0, ED, 0, 0>>\n");
    data.push_str("ES == <<FALSE, 0, {}, {}, 0>>\n");
    data.push_str("Traces == <<\n");

    let tier_code = |t: &str| -> u8 {
        match t {
            "none" => 0,
            "memory" => 1,
            "disk" => 2,
            "both" => 3,
            _ => 0,
        }
    };

    let encode_cs_act = |step: &CacheSchedulerStep| -> Result<String, TraceError> {
        let req_lane = |act: &str| -> Result<&str, TraceError> {
            let l = req_field(&step.args, "lane")?.as_str()?;
            if !cfg.lanes.iter().any(|x| x == l) {
                return Err(TraceError::InvalidSchema(format!("{act} unknown lane {l:?}")));
            }
            Ok(l)
        };
        let req_pfx = |act: &str| -> Result<&str, TraceError> {
            let p = req_field(&step.args, "prefix")?.as_str()?;
            if !cfg.prefixes.iter().any(|x| x == p) {
                return Err(TraceError::InvalidSchema(format!("{act} unknown prefix {p:?}")));
            }
            Ok(p)
        };
        let req_bcode = |act: &str| -> Result<usize, TraceError> {
            let bs = req_field(&step.args, "block")?.as_str()?;
            let bid = CacheBlockId::decode(bs)?;
            id_to_code
                .get(&bid)
                .copied()
                .ok_or_else(|| TraceError::InvalidSchema(format!("{act} block {bs:?} outside universe")))
        };
        match step.action.as_str() {
            "Init" => Ok("<<\"Init\">>".into()),
            "Arrive" => {
                let l = req_lane("Arrive")?;
                let p = req_pfx("Arrive")?;
                let tgt = req_field(&step.args, "target")?.as_u32()?;
                if !cfg.request_lengths.contains(&tgt) {
                    return Err(TraceError::InvalidSchema(format!("Arrive invalid target {tgt}")));
                }
                Ok(format!("<<\"Arrive\", \"{l}\", \"{p}\", {tgt}>>"))
            }
            "Refuse429" | "Admit" | "BeginRestore" | "RestoreStartState" | "FinishRestore"
            | "SlideWindow" | "FinishReady" | "Cancel" => {
                let l = req_lane(&step.action)?;
                Ok(format!("<<\"{}\", \"{l}\">>", step.action))
            }
            "RestoreBlock" => {
                let l = req_lane("RestoreBlock")?;
                let bc = req_bcode("RestoreBlock")?;
                Ok(format!("<<\"RestoreBlock\", \"{l}\", {bc}>>"))
            }
            "EvictCleanOrUnrefMem" | "StartColdWriteback" | "DropUnreferenced" => {
                let bc = req_bcode(&step.action)?;
                Ok(format!("<<\"{}\", {bc}>>", step.action))
            }
            "EvictCleanStartState" | "SpillStartState" => {
                let p = req_pfx(&step.action)?;
                let bnd = req_field(&step.args, "boundary")?.as_u32()?;
                if !boundaries.contains(&bnd) {
                    return Err(TraceError::InvalidSchema(format!("{} invalid boundary {bnd}", step.action)));
                }
                Ok(format!("<<\"{}\", \"{p}\", {bnd}>>", step.action))
            }
            "BuildStep" => {
                let sp_obj = req_field(&step.args, "spans")?.as_obj()?;
                let mut chosen = Vec::new();
                let mut sp_seq = Vec::with_capacity(cfg.lanes.len());
                for ln in &cfg.lanes {
                    let tok = match sp_obj.get(ln) {
                        Some(v) => v.as_u32()?,
                        None => 0,
                    };
                    if tok > 0 {
                        chosen.push(format!("\"{ln}\""));
                    }
                    sp_seq.push(tok.to_string());
                }
                for k in sp_obj.keys() {
                    if !cfg.lanes.contains(k) {
                        return Err(TraceError::InvalidSchema(format!("BuildStep unknown lane {k:?}")));
                    }
                }
                if chosen.is_empty() {
                    return Err(TraceError::InvalidSchema("BuildStep spans must choose >= 1 lane".into()));
                }
                Ok(format!(
                    "<<\"BuildStep\", {{{}}}, <<{}>>>>",
                    chosen.join(", "),
                    sp_seq.join(", ")
                ))
            }
            "ReclaimColdDisk" | "GpuExecute" | "CompleteStepIO" | "FinishStep" | "Stutter" => {
                Ok(format!("<<\"{}\">>", step.action))
            }
            other => Err(TraceError::InvalidSchema(format!("unknown cache_scheduler action {other:?}"))),
        }
    };

    for (ti, tr) in traces.iter().enumerate() {
        if ti > 0 {
            data.push_str(",\n");
        }
        data.push_str("  <<\n");
        for (si, step) in tr.steps.iter().enumerate() {
            if si > 0 {
                data.push_str(",\n");
            }
            let st = &step.state;
            let blk_map: BTreeMap<&CacheBlockId, &CacheBlockSnapshot> =
                st.blocks.iter().map(|b| (&b.id, b)).collect();
            let sp_map: BTreeMap<(&str, u32), &StartPointSnapshot> = st
                .start_points
                .iter()
                .map(|sp| ((sp.prefix.as_str(), sp.boundary), sp))
                .collect();

            data.push_str("    <<");
            data.push_str("<<");
            for (bi, bid) in all_ids.iter().enumerate() {
                if bi > 0 {
                    data.push_str(", ");
                }
                if let Some(b) = blk_map.get(bid) {
                    if b.tier == "none" {
                        data.push_str("EB");
                    } else {
                        let sealed_lit = if b.sealed { "TRUE" } else { "FALSE" };
                        let _ = write!(
                            data,
                            "<<{}, {sealed_lit}, {}, {}, <<",
                            tier_code(&b.tier),
                            b.idx,
                            b.length
                        );
                        for (di, &dv) in b.data.iter().enumerate() {
                            if di > 0 {
                                data.push_str(", ");
                            }
                            let _ = write!(data, "{dv}");
                        }
                        let _ = write!(data, ">>, {}, {}>>", b.ref_count, b.last_used_tick);
                    }
                } else {
                    data.push_str("EB");
                }
            }
            data.push_str(">>, <<");
            for (pi, pfx) in cfg.prefixes.iter().enumerate() {
                if pi > 0 {
                    data.push_str(", ");
                }
                data.push_str("<<");
                for (ki, &bnd) in boundaries.iter().enumerate() {
                    if ki > 0 {
                        data.push_str(", ");
                    }
                    if let Some(sp) = sp_map.get(&(pfx.as_str(), bnd)) {
                        let st_code = tier_code(&sp.state_tier);
                        let f_codes = sp
                            .full
                            .iter()
                            .map(|id| id_to_code[id].to_string())
                            .collect::<Vec<_>>()
                            .join(", ");
                        let w_codes = sp
                            .win
                            .iter()
                            .map(|id| id_to_code[id].to_string())
                            .collect::<Vec<_>>()
                            .join(", ");
                        let _ = write!(
                            data,
                            "<<TRUE, {st_code}, {{{f_codes}}}, {{{w_codes}}}, {}>>",
                            sp.last_used_tick
                        );
                    } else {
                        data.push_str("ES");
                    }
                }
                data.push_str(">>");
            }
            data.push_str(">>, <<");
            for (li, l) in st.lanes.iter().enumerate() {
                if li > 0 {
                    data.push_str(", ");
                }
                let _ = write!(data, "\"{}\"", l.state);
            }
            data.push_str(">>, <<");
            for (li, l) in st.lanes.iter().enumerate() {
                if li > 0 {
                    data.push_str(", ");
                }
                let _ = write!(data, "\"{}\"", l.prefix);
            }
            data.push_str(">>, <<");
            for (li, l) in st.lanes.iter().enumerate() {
                if li > 0 {
                    data.push_str(", ");
                }
                let _ = write!(data, "{}", l.target);
            }
            data.push_str(">>, <<");
            for (li, l) in st.lanes.iter().enumerate() {
                if li > 0 {
                    data.push_str(", ");
                }
                let _ = write!(data, "{}", l.pos);
            }
            data.push_str(">>, <<");
            for (li, l) in st.lanes.iter().enumerate() {
                if li > 0 {
                    data.push_str(", ");
                }
                data.push_str("<<");
                for (bi, opt) in l.full.iter().enumerate() {
                    if bi > 0 {
                        data.push_str(", ");
                    }
                    let code = opt.as_ref().map(|id| id_to_code[id]).unwrap_or(0);
                    let _ = write!(data, "{code}");
                }
                data.push_str(">>");
            }
            data.push_str(">>, <<");
            for (li, l) in st.lanes.iter().enumerate() {
                if li > 0 {
                    data.push_str(", ");
                }
                data.push_str("<<");
                for (bi, opt) in l.win.iter().enumerate() {
                    if bi > 0 {
                        data.push_str(", ");
                    }
                    let code = opt.as_ref().map(|id| id_to_code[id]).unwrap_or(0);
                    let _ = write!(data, "{code}");
                }
                data.push_str(">>");
            }
            data.push_str(">>, <<");
            for (li, l) in st.lanes.iter().enumerate() {
                if li > 0 {
                    data.push_str(", ");
                }
                let _ = write!(data, "{}", l.reservation);
            }
            data.push_str(">>, <<");
            for (li, l) in st.lanes.iter().enumerate() {
                if li > 0 {
                    data.push_str(", ");
                }
                let _ = write!(data, "{}", l.admit_budget);
            }
            data.push_str(">>, <<");
            for (li, l) in st.lanes.iter().enumerate() {
                if li > 0 {
                    data.push_str(", ");
                }
                let _ = write!(data, "{}", l.wait_ticks);
            }
            data.push_str(">>, <<");
            for (li, l) in st.lanes.iter().enumerate() {
                if li > 0 {
                    data.push_str(", ");
                }
                let _ = write!(data, "{}", l.ghost_bypass);
            }
            let sa_lit = if st.step_active { "TRUE" } else { "FALSE" };
            let sl_lit = st
                .step_lanes
                .iter()
                .map(|l| format!("\"{l}\""))
                .collect::<Vec<_>>()
                .join(", ");
            let ss_lit = cfg
                .lanes
                .iter()
                .map(|l| st.step_span.get(l).copied().unwrap_or(0).to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let sp_lit = st
                .step_pins
                .iter()
                .map(|id| id_to_code[id].to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let sio_lit = st
                .step_io
                .iter()
                .map(|id| id_to_code[id].to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let gd_lit = if st.gpu_done { "TRUE" } else { "FALSE" };
            let qs_lit = st
                .queue_seq
                .iter()
                .map(|l| format!("\"{l}\""))
                .collect::<Vec<_>>()
                .join(", ");
            let act_lit = encode_cs_act(step)?;
            let _ = write!(
                data,
                ">>, {sa_lit}, {{{sl_lit}}}, <<{ss_lit}>>, {{{sp_lit}}}, {{{sio_lit}}}, {gd_lit}, <<{qs_lit}>>, {act_lit}>>"
            );
        }
        data.push_str("\n  >>");
    }
    data.push_str("\n>>\n=============================================================================\n");

    let lane_seq_lit = cfg
        .lanes
        .iter()
        .map(|l| format!("\"{l}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let pfx_seq_lit = cfg
        .prefixes
        .iter()
        .map(|p| format!("\"{p}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let bnd_seq_lit = boundaries.iter().map(u32::to_string).collect::<Vec<_>>().join(", ");
    let blk_seq_lit = all_ids
        .iter()
        .map(CacheBlockId::to_tla_tuple)
        .collect::<Vec<_>>()
        .join(", ");
    let req_set_lit = cfg
        .request_lengths
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(", ");

    let mut check = format!(
        "---------------------------- MODULE TraceCheck ----------------------------
EXTENDS CacheScheduler, TraceData, Sequences

LaneSeq == <<{lane_seq_lit}>>
PrefixSeq == <<{pfx_seq_lit}>>
BoundarySeq == <<{bnd_seq_lit}>>
BlockSeq == <<{blk_seq_lit}>>

LaneIdxMap == [l \\in Lanes |-> CHOOSE i \\in 1..Len(LaneSeq) : LaneSeq[i] = l]
PrefixIdxMap == [p \\in Prefixes |-> CHOOSE i \\in 1..Len(PrefixSeq) : PrefixSeq[i] = p]
BoundaryIdxMap == [k \\in Boundaries |-> CHOOSE i \\in 1..Len(BoundarySeq) : BoundarySeq[i] = k]
BlockIdxMap == [b \\in BlockIds |-> CHOOSE i \\in 1..Len(BlockSeq) : BlockSeq[i] = b]

LaneIdx(l) == LaneIdxMap[l]
PrefixIdx(p) == PrefixIdxMap[p]
BoundaryIdx(k) == BoundaryIdxMap[k]
BlockIdxOf(b) == BlockIdxMap[b]

DecodeTier(c) ==
  IF c = 1 THEN \"memory\"
  ELSE IF c = 2 THEN \"disk\"
  ELSE IF c = 3 THEN \"both\"
  ELSE \"none\"
DecodeBlockId(c) == IF c = 0 THEN NoBlock ELSE BlockSeq[c]
DecodeBlockSet(cs) == {{BlockSeq[c] : c \\in cs}}
LiftBlock(r) ==
  IF r[1] = 0 THEN EmptyBlock
  ELSE [tier     |-> DecodeTier(r[1]),
        sealed   |-> r[2],
        idx      |-> r[3],
        length   |-> r[4],
        data     |-> [i \\in 0..(BlockSize - 1) |-> r[5][i + 1]],
        refCount |-> r[6]]
LiftStart(r) ==
  IF ~r[1] THEN EmptyStart
  ELSE [kept      |-> TRUE,
        stateTier |-> DecodeTier(r[2]),
        full      |-> DecodeBlockSet(r[3]),
        win       |-> DecodeBlockSet(r[4])]

Lift(s) == [
  blocks      |-> [b \\in BlockIds |-> LiftBlock(s[1][BlockIdxOf(b)])],
  startPoints |-> [pk \\in Prefixes \\X Boundaries |->
                    LiftStart(s[2][PrefixIdx(pk[1])][BoundaryIdx(pk[2])])],
  laneState   |-> [l \\in Lanes |-> s[3][LaneIdx(l)]],
  lanePrefix  |-> [l \\in Lanes |-> s[4][LaneIdx(l)]],
  laneTarget  |-> [l \\in Lanes |-> s[5][LaneIdx(l)]],
  lanePos     |-> [l \\in Lanes |-> s[6][LaneIdx(l)]],
  laneFull    |-> [l \\in Lanes |-> [i \\in BlkIdx |-> DecodeBlockId(s[7][LaneIdx(l)][i + 1])]],
  laneWin     |-> [l \\in Lanes |-> [i \\in BlkIdx |-> DecodeBlockId(s[8][LaneIdx(l)][i + 1])]],
  reservation |-> [l \\in Lanes |-> s[9][LaneIdx(l)]],
  admitBudget |-> [l \\in Lanes |-> s[10][LaneIdx(l)]],
  waitTicks   |-> [l \\in Lanes |-> s[11][LaneIdx(l)]],
  ghostBypass |-> [l \\in Lanes |-> s[12][LaneIdx(l)]],
  stepActive  |-> s[13],
  stepLanes   |-> s[14],
  stepSpan    |-> [l \\in Lanes |-> s[15][LaneIdx(l)]],
  stepPins    |-> DecodeBlockSet(s[16]),
  stepIO      |-> DecodeBlockSet(s[17]),
  gpuDone     |-> s[18],
  queueSeq    |-> s[19],
  restoreIO   |-> [l \\in Lanes |-> NoBlock]
]

VARIABLES traceIdx, stepIdx

Matches(st) ==
  /\\ blocks = st.blocks /\\ startPoints = st.startPoints
  /\\ laneState = st.laneState /\\ lanePrefix = st.lanePrefix
  /\\ laneTarget = st.laneTarget /\\ lanePos = st.lanePos
  /\\ laneFull = st.laneFull /\\ laneWin = st.laneWin
  /\\ reservation = st.reservation /\\ admitBudget = st.admitBudget
  /\\ waitTicks = st.waitTicks /\\ ghostBypass = st.ghostBypass /\\ queueSeq = st.queueSeq
  /\\ stepActive = st.stepActive
  /\\ stepLanes = st.stepLanes /\\ stepSpan = st.stepSpan
  /\\ stepPins = st.stepPins /\\ stepIO = st.stepIO
  /\\ gpuDone = st.gpuDone /\\ restoreIO = st.restoreIO

ReplayInit ==
  /\\ traceIdx \\in 1..Len(Traces)
  /\\ stepIdx = 1
  /\\ Matches(Lift(Traces[traceIdx][1]))

ReplayNext ==
  /\\ stepIdx < Len(Traces[traceIdx])
  /\\ UNCHANGED traceIdx
  /\\ stepIdx' = stepIdx + 1
  /\\ LET nxt == Lift(Traces[traceIdx][stepIdx + 1]) IN
       /\\ blocks' = nxt.blocks /\\ startPoints' = nxt.startPoints
       /\\ laneState' = nxt.laneState /\\ lanePrefix' = nxt.lanePrefix
       /\\ laneTarget' = nxt.laneTarget /\\ lanePos' = nxt.lanePos
       /\\ laneFull' = nxt.laneFull /\\ laneWin' = nxt.laneWin
       /\\ reservation' = nxt.reservation /\\ admitBudget' = nxt.admitBudget
       /\\ waitTicks' = nxt.waitTicks /\\ ghostBypass' = nxt.ghostBypass
       /\\ queueSeq' = nxt.queueSeq /\\ stepActive' = nxt.stepActive
       /\\ stepLanes' = nxt.stepLanes /\\ stepSpan' = nxt.stepSpan
       /\\ stepPins' = nxt.stepPins /\\ stepIO' = nxt.stepIO
       /\\ gpuDone' = nxt.gpuDone /\\ restoreIO' = nxt.restoreIO

ReplaySpec == ReplayInit /\\ [][ReplayNext]_<<vars, traceIdx, stepIdx>>
"
    );
    check.push_str(
        r#"\* Exact one-point specialization of AnyBuildStep (BuildStep fixes stepActive'=TRUE,
\* stepLanes'=chosen, stepSpan'=spans), avoiding 15 * 81 = 1,215 blind enumerations per step.
TraceAnyBuildStep ==
  /\ stepActive' = TRUE
  /\ stepLanes' \in ((SUBSET Lanes) \ {})
  /\ stepSpan' \in [Lanes -> 0..MaxSpan]
  /\ BuildStep(stepLanes', stepSpan')
TraceNext ==
  \/ (\E l \in Lanes : AdvanceLane(l))
  \/ MemReclaim
  \/ TraceAnyBuildStep
  \/ StepProgress
  \/ UnfairEnvActions

InitConforms == (stepIdx = 1) => Init
StepConforms == [][TraceNext]_vars

DispatchStep(act) ==
  IF act[1] = "Arrive"                    THEN Arrive(act[2], act[3], act[4])
  ELSE IF act[1] = "Refuse429"            THEN Refuse429(act[2])
  ELSE IF act[1] = "Admit"                THEN Admit(act[2])
  ELSE IF act[1] = "BeginRestore"         THEN BeginRestore(act[2])
  ELSE IF act[1] = "RestoreBlock"         THEN RestoreBlock(act[2], DecodeBlockId(act[3]))
  ELSE IF act[1] = "RestoreStartState"    THEN RestoreStartState(act[2])
  ELSE IF act[1] = "FinishRestore"        THEN FinishRestore(act[2])
  ELSE IF act[1] = "EvictCleanOrUnrefMem" THEN (EvictCleanOrUnrefMem
                                                  /\ InMemory(DecodeBlockId(act[2]))
                                                  /\ blocks'[DecodeBlockId(act[2])].tier \notin {"memory", "both"})
  ELSE IF act[1] = "StartColdWriteback"   THEN (StartColdWriteback
                                                  /\ stepIO' = {DecodeBlockId(act[2])})
  ELSE IF act[1] = "EvictCleanStartState" THEN (EvictCleanStartState
                                                  /\ StateInMem(<<act[2], act[3]>>)
                                                  /\ startPoints'[<<act[2], act[3]>>].stateTier = "disk")
  ELSE IF act[1] = "SpillStartState"      THEN (SpillStartState
                                                  /\ startPoints[<<act[2], act[3]>>].stateTier = "memory"
                                                  /\ startPoints'[<<act[2], act[3]>>].stateTier = "disk")
  ELSE IF act[1] = "ReclaimColdDisk"      THEN ReclaimColdDisk
  ELSE IF act[1] = "DropUnreferenced"     THEN DropUnreferenced(DecodeBlockId(act[2]))
  ELSE IF act[1] = "SlideWindow"          THEN SlideWindow(act[2])
  ELSE IF act[1] = "FinishReady"          THEN FinishReady(act[2])
  ELSE IF act[1] = "BuildStep"            THEN BuildStep(act[2], [l \in Lanes |-> act[3][LaneIdx(l)]])
  ELSE IF act[1] = "GpuExecute"           THEN GpuExecute
  ELSE IF act[1] = "CompleteStepIO"       THEN CompleteStepIO
  ELSE IF act[1] = "FinishStep"           THEN FinishStep
  ELSE IF act[1] = "Cancel"               THEN Cancel(act[2])
  ELSE IF act[1] = "Stutter"              THEN UNCHANGED vars
  ELSE FALSE

StepActionCheck ==
  (stepIdx' = stepIdx + 1) =>
    \E act \in {Traces[traceIdx][stepIdx + 1][20]} : DispatchStep(act)

ActionConforms == [][StepActionCheck]_<<vars, traceIdx, stepIdx>>

BlkTick(s, b) == s[1][BlockIdxOf(b)][7]
StartTick(s, pk) == s[2][PrefixIdx(pk[1])][BoundaryIdx(pk[2])][5]

LruCheck(s, act) ==
  IF act[1] = "EvictCleanOrUnrefMem"
  THEN \A c \in {d \in ColdMemBlocks : OnDisk(d) \/ blocks[d].refCount = 0} :
         BlkTick(s, DecodeBlockId(act[2])) <= BlkTick(s, c)
  ELSE IF act[1] = "StartColdWriteback"
  THEN \A c \in {d \in ColdMemBlocks : blocks[d].tier = "memory" /\ blocks[d].sealed} :
         BlkTick(s, DecodeBlockId(act[2])) <= BlkTick(s, c)
  ELSE IF act[1] = "EvictCleanStartState"
  THEN \A c \in {x \in ColdMemStates : StateOnDisk(x)} :
         StartTick(s, <<act[2], act[3]>>) <= StartTick(s, c)
  ELSE IF act[1] = "SpillStartState"
  THEN \A c \in {x \in ColdMemStates : startPoints[x].stateTier = "memory"} :
         StartTick(s, <<act[2], act[3]>>) <= StartTick(s, c)
  ELSE TRUE

StepLruCheck ==
  (stepIdx' = stepIdx + 1) =>
    \E prev \in {Traces[traceIdx][stepIdx]}, act \in {Traces[traceIdx][stepIdx + 1][20]} :
      LruCheck(prev, act)

ActionLruOrder == [][StepLruCheck]_<<vars, traceIdx, stepIdx>>
=============================================================================
"#,
    );

    let cfg_file = format!(
        "SPECIFICATION ReplaySpec
CHECK_DEADLOCK FALSE
CONSTANTS
 Lanes = {{{lane_seq_lit}}}
 Prefixes = {{{pfx_seq_lit}}}
 RequestLengths = {{{req_set_lit}}}
 BlockSize = {}
 WindowSize = {}
 MaxPos = {}
 MaxSpan = {}
 StepTokenBudget = {}
 MemBudget = {}
 DiskBudget = {}
 MaxQueue = {}
 MaxBypass = {}
 AsyncRestore = TRUE
 Bug = \"none\"
INVARIANTS
 TypeOK
 InitConforms
 StepBlocksInMemoryAndOwned
 SealedValid
 NoDecodeBlocksSealed
 UnsealedPrivateInvisibleZero
 LaneHistoryCovered
 RefCountMatchesReferrers
 StartPointsKept
 BudgetsAndReservations
 ColdBlocksSpillable
 BypassBoundHolds
 StepBudgetHolds
 QueueBounded
PROPERTIES
 StepConforms
 ActionConforms
 ActionLruOrder
 ActionSealedImmutable
 ActionNoQueuedRefusal
 ActionLazyColdWriteback
 ActionWritebackProtected
 ActionDecodePriority
",
        cfg.block_size,
        cfg.window_size,
        cfg.max_pos,
        cfg.max_span,
        cfg.step_token_budget,
        cfg.mem_budget,
        cfg.disk_budget,
        cfg.max_queue,
        cfg.max_bypass,
    );

    Ok((data, check, cfg_file))
}

// ---- Compact Event-Stream Emitters (`INKY_BLOCK_TRACE_OUT` / `INKY_CACHE_TRACE_OUT`) ----

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum BlockLayoutEvent {
    Config(BlockLayoutConfig),
    Prepare { lane: String, tokens: u32, budget: u32, pos: u32 },
    Alloc {
        lane: String,
        kind: BlockKind,
        block: u32,
        slot: u32,
        generation: u32,
        offset: u64,
        offset_limbs: OffsetLimbs,
        kvb: u32,
    },
    BeginStep { lane: String, pin_count: u32 },
    Write { lane: String, pos: u32 },
    EndStep { lane: String, pos: u32 },
    SlideWindow { lane: String },
    KeepStart { lane: String, pos: u32 },
    DropStart,
    Match { lane: String, pos: u32 },
    Finish { lane: String },
}

impl BlockLayoutEvent {
    pub fn to_json_val(&self) -> JsonVal {
        let mut m = BTreeMap::new();
        match self {
            BlockLayoutEvent::Config(cfg) => {
                let mut obj = match cfg.to_json_val() {
                    JsonVal::Obj(o) => o,
                    _ => unreachable!(),
                };
                obj.insert("pool_mode".into(), JsonVal::str("per_kind"));
                obj.insert(
                    "capacity".into(),
                    JsonVal::uint(cfg.full_capacity.max(cfg.window_capacity)),
                );
                return JsonVal::Obj(obj);
            }
            BlockLayoutEvent::Prepare { lane, tokens, budget, pos } => {
                m.insert("event".into(), JsonVal::str("prepare"));
                m.insert("lane".into(), JsonVal::str(lane.clone()));
                m.insert("tokens".into(), JsonVal::uint(*tokens));
                m.insert("budget".into(), JsonVal::uint(*budget));
                m.insert("pos".into(), JsonVal::uint(*pos));
            }
            BlockLayoutEvent::Alloc {
                lane,
                kind,
                block,
                slot,
                generation,
                offset,
                offset_limbs,
                kvb,
            } => {
                let mut limbs = BTreeMap::new();
                limbs.insert("hi".into(), JsonVal::uint(offset_limbs.hi));
                limbs.insert("mib".into(), JsonVal::uint(offset_limbs.mib));
                limbs.insert("sub".into(), JsonVal::uint(offset_limbs.sub));
                m.insert("event".into(), JsonVal::str("alloc"));
                m.insert("lane".into(), JsonVal::str(lane.clone()));
                m.insert("kind".into(), JsonVal::str(kind_name(*kind)));
                m.insert("block".into(), JsonVal::uint(*block));
                m.insert("slot".into(), JsonVal::uint(*slot));
                m.insert("generation".into(), JsonVal::uint(*generation));
                m.insert("offset".into(), JsonVal::UInt(*offset));
                m.insert("offset_limbs".into(), JsonVal::Obj(limbs));
                m.insert("kvb".into(), JsonVal::uint(*kvb));
            }
            BlockLayoutEvent::BeginStep { lane, pin_count } => {
                m.insert("event".into(), JsonVal::str("begin_step"));
                m.insert("lane".into(), JsonVal::str(lane.clone()));
                m.insert("pin_count".into(), JsonVal::uint(*pin_count));
            }
            BlockLayoutEvent::Write { lane, pos } => {
                m.insert("event".into(), JsonVal::str("write"));
                m.insert("lane".into(), JsonVal::str(lane.clone()));
                m.insert("pos".into(), JsonVal::uint(*pos));
            }
            BlockLayoutEvent::EndStep { lane, pos } => {
                m.insert("event".into(), JsonVal::str("end_step"));
                m.insert("lane".into(), JsonVal::str(lane.clone()));
                m.insert("pos".into(), JsonVal::uint(*pos));
            }
            BlockLayoutEvent::SlideWindow { lane } => {
                m.insert("event".into(), JsonVal::str("slide_window"));
                m.insert("lane".into(), JsonVal::str(lane.clone()));
            }
            BlockLayoutEvent::KeepStart { lane, pos } => {
                m.insert("event".into(), JsonVal::str("keep_start"));
                m.insert("lane".into(), JsonVal::str(lane.clone()));
                m.insert("pos".into(), JsonVal::uint(*pos));
            }
            BlockLayoutEvent::DropStart => {
                m.insert("event".into(), JsonVal::str("drop_start"));
            }
            BlockLayoutEvent::Match { lane, pos } => {
                m.insert("event".into(), JsonVal::str("match"));
                m.insert("lane".into(), JsonVal::str(lane.clone()));
                m.insert("pos".into(), JsonVal::uint(*pos));
            }
            BlockLayoutEvent::Finish { lane } => {
                m.insert("event".into(), JsonVal::str("finish"));
                m.insert("lane".into(), JsonVal::str(lane.clone()));
            }
        }
        JsonVal::Obj(m)
    }
}

pub fn emit_block_layout_event(sink: &mut dyn TraceSink, ev: &BlockLayoutEvent) -> Result<(), TraceError> {
    if let BlockLayoutEvent::Config(cfg) = ev {
        cfg.validate()?;
    }
    sink.emit_line(&ev.to_json_val().to_json_line())
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CacheSchedulerEvent {
    Config(CacheSchedulerConfig),
    Arrive { lane: String, prefix: String, target: u32, state: String },
    Refuse429 { lane: String, prefix: String, target: u32 },
    Admit { lane: String, wait_ticks: u32 },
    BeginRestore { lane: String, pos: u32, reservation: u32 },
    RestoreBlock { lane: String, prefix: String, kind: BlockKind, block: u32 },
    RestoreStartState { lane: String, prefix: String, boundary: u32 },
    FinishRestore { lane: String },
    EvictMem { prefix: String, kind: BlockKind, block: u32 },
    StartWriteback { prefix: String, kind: BlockKind, block: u32 },
    EvictStartState { prefix: String, boundary: u32 },
    SpillStartState { prefix: String, boundary: u32 },
    CompleteStepIo { prefix: String, kind: BlockKind, block: u32, tier: String },
    ReclaimDisk,
    DropUnreferenced { prefix: String, kind: BlockKind, block: u32 },
    SlideWindow { lane: String },
    BuildStep { spans: Vec<(String, u32)>, pin_count: u32 },
    GpuExecute,
    FinishStep { single_lane_pos: Option<(String, u32)> },
    Finish { lane: String },
    Cancel { lane: String, state: String },
}

impl CacheSchedulerEvent {
    pub fn to_json_val(&self) -> JsonVal {
        let mut m = BTreeMap::new();
        match self {
            CacheSchedulerEvent::Config(cfg) => return cfg.to_json_val(),
            CacheSchedulerEvent::Arrive { lane, prefix, target, state } => {
                m.insert("event".into(), JsonVal::str("arrive"));
                m.insert("lane".into(), JsonVal::str(lane.clone()));
                m.insert("prefix".into(), JsonVal::str(prefix.clone()));
                m.insert("target".into(), JsonVal::uint(*target));
                m.insert("state".into(), JsonVal::str(state.clone()));
            }
            CacheSchedulerEvent::Refuse429 { lane, prefix, target } => {
                m.insert("event".into(), JsonVal::str("refuse429"));
                m.insert("lane".into(), JsonVal::str(lane.clone()));
                m.insert("prefix".into(), JsonVal::str(prefix.clone()));
                m.insert("target".into(), JsonVal::uint(*target));
            }
            CacheSchedulerEvent::Admit { lane, wait_ticks } => {
                m.insert("event".into(), JsonVal::str("admit"));
                m.insert("lane".into(), JsonVal::str(lane.clone()));
                m.insert("wait_ticks".into(), JsonVal::uint(*wait_ticks));
            }
            CacheSchedulerEvent::BeginRestore { lane, pos, reservation } => {
                m.insert("event".into(), JsonVal::str("begin_restore"));
                m.insert("lane".into(), JsonVal::str(lane.clone()));
                m.insert("pos".into(), JsonVal::uint(*pos));
                m.insert("reservation".into(), JsonVal::uint(*reservation));
            }
            CacheSchedulerEvent::RestoreBlock { lane, prefix, kind, block } => {
                m.insert("event".into(), JsonVal::str("restore_block"));
                m.insert("lane".into(), JsonVal::str(lane.clone()));
                m.insert("prefix".into(), JsonVal::str(prefix.clone()));
                m.insert("kind".into(), JsonVal::str(kind_name(*kind)));
                m.insert("block".into(), JsonVal::uint(*block));
            }
            CacheSchedulerEvent::RestoreStartState { lane, prefix, boundary } => {
                m.insert("event".into(), JsonVal::str("restore_start_state"));
                m.insert("lane".into(), JsonVal::str(lane.clone()));
                m.insert("prefix".into(), JsonVal::str(prefix.clone()));
                m.insert("boundary".into(), JsonVal::uint(*boundary));
            }
            CacheSchedulerEvent::FinishRestore { lane } => {
                m.insert("event".into(), JsonVal::str("finish_restore"));
                m.insert("lane".into(), JsonVal::str(lane.clone()));
            }
            CacheSchedulerEvent::EvictMem { prefix, kind, block } => {
                m.insert("event".into(), JsonVal::str("evict_mem"));
                m.insert("prefix".into(), JsonVal::str(prefix.clone()));
                m.insert("kind".into(), JsonVal::str(kind_name(*kind)));
                m.insert("block".into(), JsonVal::uint(*block));
            }
            CacheSchedulerEvent::StartWriteback { prefix, kind, block } => {
                m.insert("event".into(), JsonVal::str("start_writeback"));
                m.insert("prefix".into(), JsonVal::str(prefix.clone()));
                m.insert("kind".into(), JsonVal::str(kind_name(*kind)));
                m.insert("block".into(), JsonVal::uint(*block));
            }
            CacheSchedulerEvent::EvictStartState { prefix, boundary } => {
                m.insert("event".into(), JsonVal::str("evict_start_state"));
                m.insert("prefix".into(), JsonVal::str(prefix.clone()));
                m.insert("boundary".into(), JsonVal::uint(*boundary));
            }
            CacheSchedulerEvent::SpillStartState { prefix, boundary } => {
                m.insert("event".into(), JsonVal::str("spill_start_state"));
                m.insert("prefix".into(), JsonVal::str(prefix.clone()));
                m.insert("boundary".into(), JsonVal::uint(*boundary));
            }
            CacheSchedulerEvent::CompleteStepIo { prefix, kind, block, tier } => {
                m.insert("event".into(), JsonVal::str("complete_step_io"));
                m.insert("prefix".into(), JsonVal::str(prefix.clone()));
                m.insert("kind".into(), JsonVal::str(kind_name(*kind)));
                m.insert("block".into(), JsonVal::uint(*block));
                m.insert("tier".into(), JsonVal::str(tier.clone()));
            }
            CacheSchedulerEvent::ReclaimDisk => {
                m.insert("event".into(), JsonVal::str("reclaim_disk"));
            }
            CacheSchedulerEvent::DropUnreferenced { prefix, kind, block } => {
                m.insert("event".into(), JsonVal::str("drop_unreferenced"));
                m.insert("prefix".into(), JsonVal::str(prefix.clone()));
                m.insert("kind".into(), JsonVal::str(kind_name(*kind)));
                m.insert("block".into(), JsonVal::uint(*block));
            }
            CacheSchedulerEvent::SlideWindow { lane } => {
                m.insert("event".into(), JsonVal::str("slide_window"));
                m.insert("lane".into(), JsonVal::str(lane.clone()));
            }
            CacheSchedulerEvent::BuildStep { spans, pin_count } => {
                let arr = spans
                    .iter()
                    .map(|(l, tok)| {
                        let mut sm = BTreeMap::new();
                        sm.insert("lane".into(), JsonVal::str(l.clone()));
                        sm.insert("tokens".into(), JsonVal::uint(*tok));
                        JsonVal::Obj(sm)
                    })
                    .collect();
                m.insert("event".into(), JsonVal::str("build_step"));
                m.insert("spans".into(), JsonVal::Arr(arr));
                m.insert("pin_count".into(), JsonVal::uint(*pin_count));
            }
            CacheSchedulerEvent::GpuExecute => {
                m.insert("event".into(), JsonVal::str("gpu_execute"));
            }
            CacheSchedulerEvent::FinishStep { single_lane_pos } => {
                m.insert("event".into(), JsonVal::str("finish_step"));
                if let Some((lane, pos)) = single_lane_pos {
                    m.insert("lane".into(), JsonVal::str(lane.clone()));
                    m.insert("pos".into(), JsonVal::uint(*pos));
                }
            }
            CacheSchedulerEvent::Finish { lane } => {
                m.insert("event".into(), JsonVal::str("finish"));
                m.insert("lane".into(), JsonVal::str(lane.clone()));
            }
            CacheSchedulerEvent::Cancel { lane, state } => {
                m.insert("event".into(), JsonVal::str("cancel"));
                m.insert("lane".into(), JsonVal::str(lane.clone()));
                m.insert("state".into(), JsonVal::str(state.clone()));
            }
        }
        JsonVal::Obj(m)
    }
}

pub fn emit_cache_scheduler_event(sink: &mut dyn TraceSink, ev: &CacheSchedulerEvent) -> Result<(), TraceError> {
    if let CacheSchedulerEvent::Config(cfg) = ev {
        cfg.validate()?;
    }
    sink.emit_line(&ev.to_json_val().to_json_line())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::{byte_offset, FULL_STRIDE, WINDOW_STRIDE};

    #[test]
    fn real_byte_offset_maps_to_limbs_across_kvb_and_above_4gib() {
        let s1 = Slot::new(1).unwrap();
        for kvb in [1u32, 2u32] {
            let off_f = byte_offset(BlockKind::Full, s1, kvb).unwrap();
            let off_w = byte_offset(BlockKind::Window, s1, kvb).unwrap();
            assert_eq!(off_f, 0);
            assert_eq!(off_w, 0);
            let snap = TableEntrySnapshot::from_block_entry(
                BlockIndex::new(0),
                BlockEntry::new(s1, 1, off_f, BlockKind::Full, kvb),
            )
            .unwrap();
            assert_eq!(snap.offset_limbs, OffsetLimbs::ZERO);
            assert_eq!(snap.kvb, kvb);
        }

        let s3 = Slot::new(3).unwrap();
        assert_eq!(
            OffsetLimbs::from_u64(byte_offset(BlockKind::Full, s3, 2).unwrap()).unwrap(),
            OffsetLimbs { hi: 0, mib: 4, sub: 0 }
        );
        assert_eq!(
            OffsetLimbs::from_u64(byte_offset(BlockKind::Window, s3, 1).unwrap()).unwrap(),
            OffsetLimbs { hi: 0, mib: 10, sub: 0 }
        );

        // Real crate::blocks::byte_offset above 4 GiB at both kvb = 2 and kvb = 1:
        let hi_slot_fp16 = Slot::new(2049).unwrap();
        let off_4gib_fp16 = byte_offset(BlockKind::Full, hi_slot_fp16, 2).unwrap();
        assert_eq!(off_4gib_fp16, 2048 * FULL_STRIDE);
        assert_eq!(off_4gib_fp16, 1u64 << 32);
        let limbs_fp16 = OffsetLimbs::from_u64(off_4gib_fp16).unwrap();
        assert_eq!(limbs_fp16, OffsetLimbs { hi: 1, mib: 0, sub: 0 });
        assert_eq!(limbs_fp16.truncate_32().to_u64().unwrap(), 0);

        let hi_slot_kv8 = Slot::new(4097).unwrap();
        let off_4gib_kv8 = byte_offset(BlockKind::Full, hi_slot_kv8, 1).unwrap();
        assert_eq!(off_4gib_kv8, 4096 * (FULL_STRIDE / 2));
        assert_eq!(off_4gib_kv8, 1u64 << 32);
        let win_20gib_kv8 = byte_offset(BlockKind::Window, hi_slot_kv8, 1).unwrap();
        assert_eq!(win_20gib_kv8, 4096 * (WINDOW_STRIDE / 2));
        assert_eq!(
            OffsetLimbs::from_u64(win_20gib_kv8).unwrap(),
            OffsetLimbs { hi: 5, mib: 0, sub: 0 }
        );
    }

    #[test]
    fn parser_rejects_tampered_offset_limbs_noncanonical_limbs_and_bad_json() {
        let cfg = BlockLayoutConfig {
            lanes: vec!["a".into(), "b".into()],
            full_capacity: 2,
            window_capacity: 2,
            block_size: 2,
            window_size: 2,
            max_pos: 4,
            max_span: 2,
            max_generation: 2,
            slot_base: 2048,
            kvb: 2,
        };
        let mut sink = MemorySink::new();
        emit_block_layout_config(&mut sink, &cfg).unwrap();

        let real_off = byte_offset(BlockKind::Full, Slot::new(2049).unwrap(), 2).unwrap();
        let entry = TableEntrySnapshot::from_block_entry(
            BlockIndex::new(0),
            BlockEntry::new(Slot::new(1).unwrap(), 1, real_off, BlockKind::Full, 2),
        )
        .unwrap();
        assert_eq!(entry.offset_limbs.hi, 1);

        let st0 = BlockLayoutState {
            full_pool: (1..=2)
                .map(|s| PoolSlotSnapshot {
                    slot: s,
                    generation: 0,
                    kind: BlockKind::Full,
                    block: 0,
                    length: 0,
                    data: vec![0, 0],
                })
                .collect(),
            window_pool: (1..=2)
                .map(|s| PoolSlotSnapshot {
                    slot: s,
                    generation: 0,
                    kind: BlockKind::Window,
                    block: 0,
                    length: 0,
                    data: vec![0, 0],
                })
                .collect(),
            table: [("a".into(), vec![]), ("b".into(), vec![])].into_iter().collect(),
            pos: [("a".into(), 0), ("b".into(), 0)].into_iter().collect(),
            phase: [("a".into(), "idle".into()), ("b".into(), "idle".into())].into_iter().collect(),
            target: [("a".into(), 0), ("b".into(), 0)].into_iter().collect(),
            span_start: [("a".into(), 0), ("b".into(), 0)].into_iter().collect(),
            budget: [("a".into(), 0), ("b".into(), 0)].into_iter().collect(),
            pin: [("a".into(), vec![]), ("b".into(), vec![])].into_iter().collect(),
            start_kept: false,
            start_pos: 0,
            start_state: 0,
            start_table: vec![],
        };
        let mut st1 = st0.clone();
        st1.phase.insert("a".into(), "prepare".into());
        st1.target.insert("a".into(), 2);
        st1.budget.insert("a".into(), 3);
        st1.full_pool[0].generation = 1;
        st1.table.insert("a".into(), vec![entry]);

        emit_block_layout_step(
            &mut sink,
            &BlockLayoutStep { seq: 0, action: "Init".into(), args: BTreeMap::new(), state: st0 },
        )
        .unwrap();
        emit_block_layout_step(
            &mut sink,
            &BlockLayoutStep { seq: 1, action: "Alloc".into(), args: BTreeMap::new(), state: st1 },
        )
        .unwrap();

        let valid_jsonl = sink.to_jsonl();
        let parsed = parse_block_layout_jsonl(&valid_jsonl).unwrap();
        assert_eq!(parsed.steps[1].state.table["a"][0].offset, 1u64 << 32);

        // 1. Tamper high 32-bit limb ("hi":1 -> "hi":0) while keeping 64-bit offset -> rejected!
        let bad_hi = valid_jsonl.replace("\"hi\":1", "\"hi\":0");
        assert_ne!(bad_hi, valid_jsonl);
        assert!(parse_block_layout_jsonl(&bad_hi).is_err());

        // 2. Non-canonical mib limb ("mib":0 -> "mib":4096) -> rejected!
        let bad_mib = valid_jsonl.replace("\"mib\":0", "\"mib\":4096");
        assert!(parse_block_layout_jsonl(&bad_mib).is_err());

        // 3. Out-of-range slot ("slot":1 -> "slot":3 when full_capacity == 2) -> rejected!
        let bad_slot = valid_jsonl.replace("\"offset\":4294967296,\"offset_limbs\":{\"hi\":1,\"mib\":0,\"sub\":0},\"slot\":1",
                                           "\"offset\":4294967296,\"offset_limbs\":{\"hi\":1,\"mib\":0,\"sub\":0},\"slot\":3");
        assert_ne!(bad_slot, valid_jsonl);
        assert!(parse_block_layout_jsonl(&bad_slot).is_err());

        // 4. Malformed JSON lines rejected through the public parser API:
        assert!(parse_block_layout_jsonl("{\"event\":\"config\",\"event\":\"dup\"}\n").is_err());
        assert!(parse_block_layout_jsonl("{\"event\":\"config\",\"version\":1.5}\n").is_err());
        assert!(parse_block_layout_jsonl("{\"event\":\"config\"} trailing\n").is_err());
    }

    #[test]
    fn trace_bundles_bind_step_actions_and_refuse_missing_action_arguments() {
        let bl_cfg = BlockLayoutConfig {
            lanes: vec!["a".into()],
            full_capacity: 2,
            window_capacity: 2,
            block_size: 2,
            window_size: 2,
            max_pos: 4,
            max_span: 2,
            max_generation: 2,
            slot_base: 0,
            kvb: 2,
        };
        let st0 = BlockLayoutState {
            full_pool: (1..=2)
                .map(|s| PoolSlotSnapshot {
                    slot: s,
                    generation: 0,
                    kind: BlockKind::Full,
                    block: 0,
                    length: 0,
                    data: vec![0, 0],
                })
                .collect(),
            window_pool: (1..=2)
                .map(|s| PoolSlotSnapshot {
                    slot: s,
                    generation: 0,
                    kind: BlockKind::Window,
                    block: 0,
                    length: 0,
                    data: vec![0, 0],
                })
                .collect(),
            table: [("a".into(), vec![])].into_iter().collect(),
            pos: [("a".into(), 0)].into_iter().collect(),
            phase: [("a".into(), "idle".into())].into_iter().collect(),
            target: [("a".into(), 0)].into_iter().collect(),
            span_start: [("a".into(), 0)].into_iter().collect(),
            budget: [("a".into(), 0)].into_iter().collect(),
            pin: [("a".into(), vec![])].into_iter().collect(),
            start_kept: false,
            start_pos: 0,
            start_state: 0,
            start_table: vec![],
        };
        let mut ok_args = BTreeMap::new();
        ok_args.insert("lane".into(), JsonVal::str("a"));
        ok_args.insert("tokens".into(), JsonVal::uint(2u32));
        let ok_trace = BlockLayoutTrace {
            config: bl_cfg.clone(),
            steps: vec![
                BlockLayoutStep { seq: 0, action: "Init".into(), args: BTreeMap::new(), state: st0.clone() },
                BlockLayoutStep { seq: 1, action: "Prepare".into(), args: ok_args, state: st0.clone() },
            ],
        };
        let (bl_data, bl_check, bl_cfg_txt) = render_block_layout_tla_bundle(&[ok_trace]).unwrap();
        assert!(bl_data.contains("<<\"Prepare\", \"a\", 2>>"));
        assert!(bl_check.contains("DispatchStep(act) =="));
        assert!(bl_check.contains("ActionConforms == [][StepActionCheck]_<<vars, traceIdx, stepIdx>>"));
        assert!(bl_cfg_txt.contains("ActionConforms"));

        // Missing `tokens` on Prepare must fail closed at bundle rendering.
        let bad_trace = BlockLayoutTrace {
            config: bl_cfg,
            steps: vec![
                BlockLayoutStep { seq: 0, action: "Init".into(), args: BTreeMap::new(), state: st0.clone() },
                BlockLayoutStep { seq: 1, action: "Prepare".into(), args: BTreeMap::new(), state: st0 },
            ],
        };
        assert!(render_block_layout_tla_bundle(&[bad_trace]).is_err());
    }

    #[test]
    fn cache_scheduler_parser_rejects_every_missing_required_field_or_bad_domain() {
        let cfg = CacheSchedulerConfig {
            lanes: vec!["a".into(), "b".into()],
            prefixes: vec!["p1".into()],
            request_lengths: vec![2, 4],
            block_size: 2,
            window_size: 2,
            max_pos: 4,
            max_span: 2,
            step_token_budget: 2,
            mem_budget: 4,
            disk_budget: 3,
            max_queue: 1,
            max_bypass: 2,
        };
        let mut sink = MemorySink::new();
        emit_cache_scheduler_config(&mut sink, &cfg).unwrap();
        let bid = CacheBlockId::Tree { prefix: "p1".into(), kind: BlockKind::Full, idx: 0 };
        let wbid = CacheBlockId::Tree { prefix: "p1".into(), kind: BlockKind::Window, idx: 0 };
        let st = CacheSchedulerState {
            blocks: vec![
                CacheBlockSnapshot {
                    id: bid.clone(),
                    tier: "memory".into(),
                    sealed: true,
                    idx: 0,
                    length: 2,
                    data: vec![1, 1],
                    ref_count: 1,
                    last_used_tick: 7,
                },
                CacheBlockSnapshot {
                    id: wbid.clone(),
                    tier: "memory".into(),
                    sealed: true,
                    idx: 0,
                    length: 2,
                    data: vec![1, 1],
                    ref_count: 1,
                    last_used_tick: 7,
                },
            ],
            start_points: vec![StartPointSnapshot {
                prefix: "p1".into(),
                boundary: 2,
                state_kept: true,
                state_tier: "memory".into(),
                full: vec![bid],
                win: vec![wbid],
                last_used_tick: 7,
            }],
            lanes: vec![
                CacheLaneSnapshot {
                    lane: "a".into(),
                    state: "Idle".into(),
                    prefix: "p1".into(),
                    target: 2,
                    pos: 0,
                    full: vec![None, None],
                    win: vec![None, None],
                    reservation: 0,
                    admit_budget: 0,
                    wait_ticks: 0,
                    ghost_bypass: 0,
                },
                CacheLaneSnapshot {
                    lane: "b".into(),
                    state: "Idle".into(),
                    prefix: "p1".into(),
                    target: 2,
                    pos: 0,
                    full: vec![None, None],
                    win: vec![None, None],
                    reservation: 0,
                    admit_budget: 0,
                    wait_ticks: 0,
                    ghost_bypass: 0,
                },
            ],
            queue_seq: vec![],
            step_active: false,
            step_lanes: vec![],
            step_span: [("a".into(), 0), ("b".into(), 0)].into_iter().collect(),
            step_pins: vec![],
            step_io: vec![],
            gpu_done: false,
        };
        emit_cache_scheduler_step(
            &mut sink,
            &CacheSchedulerStep { seq: 0, action: "Init".into(), args: BTreeMap::new(), state: st },
        )
        .unwrap();
        let valid = sink.to_jsonl();
        assert!(parse_cache_scheduler_jsonl(&valid).is_ok());

        for needle in [
            "\"last_used_tick\":7,",
            "\"state_tier\":\"memory\",",
            "\"ghost_bypass\":0,",
            "\"queue_seq\":[],",
            "\"a\":0,",
        ] {
            let dropped = valid.replacen(needle, "", 1);
            assert_ne!(dropped, valid, "failed to strip {needle}");
            assert!(
                parse_cache_scheduler_jsonl(&dropped).is_err(),
                "parser silently accepted trace missing {needle}"
            );
        }

        let bad_lane_state = valid.replacen("\"state\":\"Idle\"", "\"state\":\"Bogus\"", 1);
        assert!(parse_cache_scheduler_jsonl(&bad_lane_state).is_err());
        let bad_prefix = valid.replacen("\"prefix\":\"p1\"", "\"prefix\":\"p99\"", 1);
        assert!(parse_cache_scheduler_jsonl(&bad_prefix).is_err());
    }
}
