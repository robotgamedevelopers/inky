//! Byte-level RFC 8259 JSON pushdown grammar mask for constrained tool-call sampling.
//!
//! Tracks active tool-call spans (`<|message_model|><header><|content_invoke_tool_json|> ... <|end_message|>`)
//! per serving lane and masks out any vocabulary token whose UTF-8 bytes would make the partial JSON
//! syntactically or schema-structurally invalid, or that would emit `<|end_message|>` before the root
//! JSON object `{...}` is balanced and complete.

use crate::tokenizer::Tokenizer;
use serde_json::Value;
use std::sync::Arc;

const MAX_DEPTH_CAP: usize = 32;
const MAX_SCHEMAS: usize = 64;
const MAX_FIXED_STRINGS: usize = 128;
const MAX_SCHEMA_PROPS: usize = 64;
const NO_SCHEMA: u16 = u16::MAX;
const NO_FIXED_STR: u16 = u16::MAX;

/// Configurable structural bounds so adversarial logit distributions cannot hold open
/// unbounded containers, strings, or structural whitespace runs indefinitely while real
/// tool calls retain generous limits (including multi-space pretty-printed JSON).
#[derive(Clone, Copy, Debug)]
pub struct GrammarLimits {
    pub max_depth: u8,
    pub max_container_items: u16,
    pub max_string_bytes: u32,
    pub max_ws_run: u8,
}

impl GrammarLimits {
    pub const DEFAULT: Self =
        Self { max_depth: 28, max_container_items: 1024, max_string_bytes: 256 * 1024, max_ws_run: 64 };

    #[cfg(test)]
    pub const fn adversarial_test() -> Self {
        Self { max_depth: 8, max_container_items: 8, max_string_bytes: 48, max_ws_run: 4 }
    }
}

impl Default for GrammarLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Contiguous precomputed vocabulary byte lookup table (zero per-step heap allocations).
pub struct VocabBytes {
    data: Vec<u8>,
    offsets: Vec<u32>,
    special_bits: Vec<u64>,
    vocab_len: usize,
}

impl VocabBytes {
    pub fn from_tokenizer(tk: &Tokenizer) -> Self {
        let n = tk.vocab_len();
        let mut total_bytes = 0usize;
        for id in 0..n {
            total_bytes += tk.decode_bytes(id as u32).len();
        }
        let mut data = Vec::with_capacity(total_bytes);
        let mut offsets = Vec::with_capacity(n + 1);
        let mut special_bits = vec![0u64; n.div_ceil(64)];
        for id in 0..n {
            offsets.push(data.len() as u32);
            if tk.is_special(id as u32) {
                special_bits[id / 64] |= 1u64 << (id % 64);
            } else {
                data.extend_from_slice(&tk.decode_bytes(id as u32));
            }
        }
        offsets.push(data.len() as u32);
        Self { data, offsets, special_bits, vocab_len: n }
    }

    #[inline]
    pub fn is_special(&self, id: u32) -> bool {
        let idx = id as usize;
        if idx >= self.vocab_len {
            return true;
        }
        (self.special_bits[idx / 64] & (1u64 << (idx % 64))) != 0
    }

    /// Non-special vocabulary token bytes, or `None` for special / out-of-range IDs.
    #[inline]
    pub fn non_special_bytes(&self, id: u32) -> Option<&[u8]> {
        let idx = id as usize;
        if idx >= self.vocab_len || self.is_special(id) {
            return None;
        }
        let s = self.offsets[idx] as usize;
        let e = self.offsets[idx + 1] as usize;
        Some(&self.data[s..e])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TypeMask(u8);

impl TypeMask {
    const OBJ: Self = Self(1 << 0);
    const ARR: Self = Self(1 << 1);
    const STR: Self = Self(1 << 2);
    const NUM: Self = Self(1 << 3);
    const INT: Self = Self(1 << 4);
    const BOOL: Self = Self(1 << 5);
    const NULL: Self = Self(1 << 6);
    const CONTAINERS: Self = Self((1 << 0) | (1 << 1));
    const ANY: Self = Self((1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 5) | (1 << 6));

    #[inline]
    const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    #[inline]
    const fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline]
    const fn without_containers_if_possible(self) -> Self {
        let prims = self.0 & !Self::CONTAINERS.0;
        if prims != 0 {
            Self(prims)
        } else {
            self
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PropValueSpec {
    mask: TypeMask,
    nested_schema: u16,
    fixed_str: u16,
}

impl PropValueSpec {
    const fn any() -> Self {
        Self { mask: TypeMask::ANY, nested_schema: NO_SCHEMA, fixed_str: NO_FIXED_STR }
    }

    const fn object(nested_schema: u16) -> Self {
        Self { mask: TypeMask::OBJ, nested_schema, fixed_str: NO_FIXED_STR }
    }

    const fn fixed_string(fixed_str: u16) -> Self {
        Self { mask: TypeMask::STR, nested_schema: NO_SCHEMA, fixed_str }
    }
}

#[derive(Clone, Debug)]
struct PropSpec {
    escaped: Vec<u8>,
    spec: PropValueSpec,
}

#[derive(Clone, Debug)]
struct ObjectSchema {
    props: Vec<PropSpec>,
    required_mask: u64,
    allow_additional: bool,
}

impl ObjectSchema {
    #[inline]
    fn all_props_mask(&self) -> u64 {
        let n = self.props.len().min(MAX_SCHEMA_PROPS);
        if n == 64 {
            u64::MAX
        } else {
            (1u64 << n) - 1
        }
    }

    #[inline]
    fn required_satisfied(&self, seen_mask: u64) -> bool {
        (seen_mask & self.required_mask) == self.required_mask
    }

    #[inline]
    fn unseen_mask(&self, seen_mask: u64) -> u64 {
        self.all_props_mask() & !seen_mask
    }

    #[inline]
    fn can_add_more_keys(&self, seen_mask: u64) -> bool {
        self.allow_additional || self.unseen_mask(seen_mask) != 0
    }
}

/// Immutable arena of compiled object schemas and fixed string literals for a request.
#[derive(Clone, Debug, Default)]
pub struct SchemaArena {
    schemas: Vec<ObjectSchema>,
    fixed_strings: Vec<Vec<u8>>,
}

fn json_escaped_inner(s: &str) -> Vec<u8> {
    let quoted = serde_json::to_string(s).expect("JSON string serialization");
    quoted.as_bytes()[1..quoted.len() - 1].to_vec()
}

impl SchemaArena {
    fn intern_fixed_str(&mut self, s: &str) -> u16 {
        if self.fixed_strings.len() >= MAX_FIXED_STRINGS {
            return NO_FIXED_STR;
        }
        let esc = json_escaped_inner(s);
        if let Some(pos) = self.fixed_strings.iter().position(|x| x == &esc) {
            return pos as u16;
        }
        let idx = self.fixed_strings.len() as u16;
        self.fixed_strings.push(esc);
        idx
    }

    fn compile_prop_value(&mut self, schema: &Value, depth: usize) -> PropValueSpec {
        let Some(obj) = schema.as_object() else {
            return PropValueSpec::any();
        };
        // Schemas using combinators (`anyOf`, `oneOf`, `allOf`, `$ref`, untyped `enum`) remain permissive
        // unless an explicit JSON `"type"` is declared.
        let Some(ty_val) = obj.get("type") else {
            if obj.contains_key("properties") || obj.contains_key("required") {
                let nested = if depth < 6 { self.compile_object_schema(schema, depth + 1) } else { NO_SCHEMA };
                return PropValueSpec::object(nested);
            }
            return PropValueSpec::any();
        };
        let parse_one = |name: &str| -> Option<TypeMask> {
            match name {
                "object" => Some(TypeMask::OBJ),
                "array" => Some(TypeMask::ARR),
                "string" => Some(TypeMask::STR),
                "number" => Some(TypeMask::NUM),
                "integer" => Some(TypeMask::INT),
                "boolean" => Some(TypeMask::BOOL),
                "null" => Some(TypeMask::NULL),
                _ => None,
            }
        };
        let mut mask = TypeMask(0);
        match ty_val {
            Value::String(s) => {
                let Some(m) = parse_one(s) else {
                    return PropValueSpec::any();
                };
                mask = m;
            }
            Value::Array(items) => {
                for item in items {
                    let Some(s) = item.as_str().and_then(parse_one) else {
                        return PropValueSpec::any();
                    };
                    mask = mask.bitor(s);
                }
                if mask.0 == 0 {
                    return PropValueSpec::any();
                }
            }
            _ => return PropValueSpec::any(),
        }
        let nested_schema = if mask.contains(TypeMask::OBJ) && depth < 6 {
            self.compile_object_schema(schema, depth + 1)
        } else {
            NO_SCHEMA
        };
        PropValueSpec { mask, nested_schema, fixed_str: NO_FIXED_STR }
    }

    pub fn compile_object_schema(&mut self, schema: &Value, depth: usize) -> u16 {
        if self.schemas.len() >= MAX_SCHEMAS {
            return NO_SCHEMA;
        }
        let Some(obj) = schema.as_object() else {
            return NO_SCHEMA;
        };
        let allow_additional = !matches!(obj.get("additionalProperties"), Some(Value::Bool(false)));
        let mut prop_names: Vec<String> = Vec::new();
        let mut required_names: Vec<String> = Vec::new();
        if let Some(req_arr) = obj.get("required").and_then(Value::as_array) {
            for r in req_arr {
                if let Some(name) = r.as_str() {
                    if !required_names.iter().any(|x| x == name) && required_names.len() < MAX_SCHEMA_PROPS {
                        required_names.push(name.to_owned());
                    }
                    if !prop_names.iter().any(|x| x == name) && prop_names.len() < MAX_SCHEMA_PROPS {
                        prop_names.push(name.to_owned());
                    }
                }
            }
        }
        let props_map = obj.get("properties").and_then(Value::as_object);
        if let Some(pm) = props_map {
            for k in pm.keys() {
                if !prop_names.iter().any(|x| x == k) && prop_names.len() < MAX_SCHEMA_PROPS {
                    prop_names.push(k.clone());
                }
            }
        }
        if prop_names.is_empty() && allow_additional {
            return NO_SCHEMA;
        }
        let mut props = Vec::with_capacity(prop_names.len());
        let mut required_mask = 0u64;
        for (idx, name) in prop_names.iter().enumerate() {
            let spec = props_map
                .and_then(|pm| pm.get(name))
                .map(|sub| self.compile_prop_value(sub, depth))
                .unwrap_or_else(PropValueSpec::any);
            if required_names.iter().any(|r| r == name) {
                required_mask |= 1u64 << idx;
            }
            props.push(PropSpec { escaped: json_escaped_inner(name), spec });
        }
        if self.schemas.len() >= MAX_SCHEMAS {
            return NO_SCHEMA;
        }
        let id = self.schemas.len() as u16;
        self.schemas.push(ObjectSchema { props, required_mask, allow_additional });
        id
    }

    fn build_tool_wrapper_schema(&mut self, fixed_tool_name: Option<&str>, args_schema: u16) -> u16 {
        let name_spec = match fixed_tool_name.filter(|s| !s.is_empty()) {
            Some(name) => {
                let fs = self.intern_fixed_str(name);
                if fs != NO_FIXED_STR {
                    PropValueSpec::fixed_string(fs)
                } else {
                    PropValueSpec { mask: TypeMask::STR, nested_schema: NO_SCHEMA, fixed_str: NO_FIXED_STR }
                }
            }
            None => PropValueSpec { mask: TypeMask::STR, nested_schema: NO_SCHEMA, fixed_str: NO_FIXED_STR },
        };
        let args_spec = PropValueSpec::object(args_schema);
        let props = vec![
            PropSpec { escaped: b"name".to_vec(), spec: name_spec },
            PropSpec { escaped: b"args".to_vec(), spec: args_spec },
        ];
        // `parse_tool_call` in `src/serve/reply.rs` requires `"args"` to be a JSON object and reads
        // optional `"name"` (falling back to the header name when omitted).
        let id = self.schemas.len() as u16;
        self.schemas.push(ObjectSchema { props, required_mask: 1u64 << 1, allow_additional: false });
        id
    }
}

#[derive(Clone, Debug)]
struct ToolEntry {
    name: String,
    wrapper_schema: u16,
}

/// Pre-compiled request tool catalog shared via `Arc` on `Job`.
#[derive(Clone, Debug)]
pub struct ToolCatalog {
    arena: SchemaArena,
    tools: Vec<ToolEntry>,
    fallback_wrapper_schema: u16,
}

impl ToolCatalog {
    /// Extract tool definitions from either OpenAI `/v1/chat/completions` (`tools[].function.{name, parameters}`)
    /// or Anthropic `/v1/messages` (`tools[].{name, input_schema}`). Returns `None` when no tools are present.
    pub fn from_request(req: &Value) -> Option<Arc<Self>> {
        let arr = req.get("tools")?.as_array()?;
        if arr.is_empty() {
            return None;
        }
        let mut arena = SchemaArena::default();
        let mut raw_tools: Vec<(String, Option<&Value>)> = Vec::new();
        for item in arr {
            if let Some(func) = item.get("function").and_then(Value::as_object) {
                if let Some(name) = func.get("name").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                    raw_tools.push((name.to_owned(), func.get("parameters")));
                    continue;
                }
            }
            if let Some(obj) = item.as_object() {
                if let Some(name) = obj.get("name").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                    raw_tools.push((name.to_owned(), obj.get("input_schema").or_else(|| obj.get("parameters"))));
                }
            }
        }
        let mut tools = Vec::with_capacity(raw_tools.len());
        for (name, params) in raw_tools {
            if arena.schemas.len() + 2 >= MAX_SCHEMAS {
                break;
            }
            let args_schema = params.map(|p| arena.compile_object_schema(p, 0)).unwrap_or(NO_SCHEMA);
            let wrapper_schema = arena.build_tool_wrapper_schema(Some(&name), args_schema);
            tools.push(ToolEntry { name, wrapper_schema });
        }
        let fallback_wrapper_schema =
            if tools.len() == 1 { tools[0].wrapper_schema } else { arena.build_tool_wrapper_schema(None, NO_SCHEMA) };
        Some(Arc::new(Self { arena, tools, fallback_wrapper_schema }))
    }
}

#[inline]
fn is_json_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

#[inline]
fn hex_nibble(b: u8) -> Option<u16> {
    match b {
        b'0'..=b'9' => Some((b - b'0') as u16),
        b'a'..=b'f' => Some((b - b'a' + 10) as u16),
        b'A'..=b'F' => Some((b - b'A' + 10) as u16),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EscState {
    None,
    Backslash,
    Hex { digits: u8, val: u16, after_high: bool },
    NeedLowBackslash,
    NeedLowU,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StrSubState {
    utf8_need: u8,
    utf8_first_min: u8,
    utf8_first_max: u8,
    esc: EscState,
    byte_len: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StrStep {
    Continue,
    Closed,
    Invalid,
}

impl StrSubState {
    const fn new() -> Self {
        Self { utf8_need: 0, utf8_first_min: 0x80, utf8_first_max: 0xBF, esc: EscState::None, byte_len: 0 }
    }

    #[inline]
    fn step(&mut self, b: u8, max_bytes: u32) -> StrStep {
        if self.utf8_need > 0 {
            if b < self.utf8_first_min || b > self.utf8_first_max {
                return StrStep::Invalid;
            }
            self.utf8_need -= 1;
            self.utf8_first_min = 0x80;
            self.utf8_first_max = 0xBF;
            self.byte_len = self.byte_len.saturating_add(1);
            return StrStep::Continue;
        }

        match self.esc {
            EscState::None => {
                if b == b'"' {
                    return StrStep::Closed;
                }
                if self.byte_len >= max_bytes {
                    return StrStep::Invalid;
                }
                if b == b'\\' {
                    self.esc = EscState::Backslash;
                    self.byte_len = self.byte_len.saturating_add(1);
                    return StrStep::Continue;
                }
                if b < 0x20 {
                    // Unescaped U+0000..=U+001F control characters are forbidden by RFC 8259 §7.
                    return StrStep::Invalid;
                }
                if b < 0x80 {
                    self.byte_len = self.byte_len.saturating_add(1);
                    return StrStep::Continue;
                }
                // RFC 3629 multi-byte UTF-8 lead byte validation.
                match b {
                    0xC2..=0xDF => {
                        self.utf8_need = 1;
                        self.utf8_first_min = 0x80;
                        self.utf8_first_max = 0xBF;
                    }
                    0xE0 => {
                        self.utf8_need = 2;
                        self.utf8_first_min = 0xA0;
                        self.utf8_first_max = 0xBF;
                    }
                    0xE1..=0xEC | 0xEE..=0xEF => {
                        self.utf8_need = 2;
                        self.utf8_first_min = 0x80;
                        self.utf8_first_max = 0xBF;
                    }
                    0xED => {
                        // Exclude UTF-16 surrogates U+D800..=U+DFFF (ED A0..=BF).
                        self.utf8_need = 2;
                        self.utf8_first_min = 0x80;
                        self.utf8_first_max = 0x9F;
                    }
                    0xF0 => {
                        self.utf8_need = 3;
                        self.utf8_first_min = 0x90;
                        self.utf8_first_max = 0xBF;
                    }
                    0xF1..=0xF3 => {
                        self.utf8_need = 3;
                        self.utf8_first_min = 0x80;
                        self.utf8_first_max = 0xBF;
                    }
                    0xF4 => {
                        // Exclude > U+10FFFF (F4 90..=BF).
                        self.utf8_need = 3;
                        self.utf8_first_min = 0x80;
                        self.utf8_first_max = 0x8F;
                    }
                    _ => return StrStep::Invalid,
                }
                self.byte_len = self.byte_len.saturating_add(1);
                StrStep::Continue
            }
            EscState::Backslash => {
                self.byte_len = self.byte_len.saturating_add(1);
                match b {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                        self.esc = EscState::None;
                        StrStep::Continue
                    }
                    b'u' => {
                        self.esc = EscState::Hex { digits: 0, val: 0, after_high: false };
                        StrStep::Continue
                    }
                    _ => StrStep::Invalid,
                }
            }
            EscState::Hex { digits, val, after_high } => {
                let Some(nib) = hex_nibble(b) else {
                    return StrStep::Invalid;
                };
                // Reject lone low surrogates (0xDC00..=0xDFFF when !after_high) and non-low
                // surrogates (outside 0xDC00..=0xDFFF when after_high) at the earliest hex digit
                // so no partial prefix ever reaches a dead end.
                if after_high {
                    if digits == 0 && nib != 0xD {
                        return StrStep::Invalid;
                    }
                    if digits == 1 && nib < 0xC {
                        return StrStep::Invalid;
                    }
                } else if digits == 1 && val == 0xD && nib >= 0xC {
                    return StrStep::Invalid;
                }
                let next_val = (val << 4) | nib;
                let next_digits = digits + 1;
                self.byte_len = self.byte_len.saturating_add(1);
                if next_digits == 4 {
                    if !after_high && (0xD800..=0xDBFF).contains(&next_val) {
                        self.esc = EscState::NeedLowBackslash;
                    } else {
                        self.esc = EscState::None;
                    }
                } else {
                    self.esc = EscState::Hex { digits: next_digits, val: next_val, after_high };
                }
                StrStep::Continue
            }
            EscState::NeedLowBackslash => {
                if b != b'\\' {
                    return StrStep::Invalid;
                }
                self.byte_len = self.byte_len.saturating_add(1);
                self.esc = EscState::NeedLowU;
                StrStep::Continue
            }
            EscState::NeedLowU => {
                if b != b'u' {
                    return StrStep::Invalid;
                }
                self.byte_len = self.byte_len.saturating_add(1);
                self.esc = EscState::Hex { digits: 0, val: 0, after_high: true };
                StrStep::Continue
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumStage {
    Sign,
    Zero,
    Int { digits: u8 },
    FracNeedDigit,
    Frac { digits: u8 },
    ExpSignOrDigit,
    ExpNeedDigit,
    Exp { digits: u8 },
}

impl NumStage {
    #[inline]
    fn is_complete(self) -> bool {
        matches!(self, Self::Zero | Self::Int { .. } | Self::Frac { .. } | Self::Exp { .. })
    }

    #[inline]
    fn step_digit_or_part(&mut self, b: u8, int_only: bool) -> bool {
        match *self {
            Self::Sign => match b {
                b'0' if !int_only => {
                    *self = Self::Zero;
                    true
                }
                b'1'..=b'9' => {
                    *self = Self::Int { digits: 1 };
                    true
                }
                _ => false,
            },
            Self::Zero => {
                if int_only {
                    return false;
                }
                match b {
                    b'.' => {
                        *self = Self::FracNeedDigit;
                        true
                    }
                    b'e' | b'E' => {
                        *self = Self::ExpSignOrDigit;
                        true
                    }
                    _ => false,
                }
            }
            Self::Int { digits } => match b {
                b'0'..=b'9' if digits < 18 => {
                    *self = Self::Int { digits: digits + 1 };
                    true
                }
                b'.' if !int_only => {
                    *self = Self::FracNeedDigit;
                    true
                }
                b'e' | b'E' if !int_only => {
                    *self = Self::ExpSignOrDigit;
                    true
                }
                _ => false,
            },
            Self::FracNeedDigit => {
                if b.is_ascii_digit() {
                    *self = Self::Frac { digits: 1 };
                    true
                } else {
                    false
                }
            }
            Self::Frac { digits } => match b {
                b'0'..=b'9' if digits < 16 => {
                    *self = Self::Frac { digits: digits + 1 };
                    true
                }
                b'e' | b'E' => {
                    *self = Self::ExpSignOrDigit;
                    true
                }
                _ => false,
            },
            Self::ExpSignOrDigit => match b {
                b'+' | b'-' => {
                    *self = Self::ExpNeedDigit;
                    true
                }
                b'0'..=b'9' => {
                    *self = Self::Exp { digits: 1 };
                    true
                }
                _ => false,
            },
            Self::ExpNeedDigit => {
                if b.is_ascii_digit() {
                    *self = Self::Exp { digits: 1 };
                    true
                } else {
                    false
                }
            }
            Self::Exp { digits } => {
                if b.is_ascii_digit() && digits < 2 {
                    *self = Self::Exp { digits: digits + 1 };
                    true
                } else {
                    false
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Frame {
    Empty,
    Obj { schema_idx: u16, seen_mask: u64, items: u16 },
    Arr { items: u16 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyMode {
    Restricted { schema_idx: u16, cand_mask: u64, pos: u16 },
    Free { schema_idx: u16, cand_mask: u64, pos: u16 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StrMode {
    Free,
    Fixed { fixed_str: u16, pos: u16 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    BeforeRoot,
    ObjExpectKeyOrEnd,
    ObjExpectKey,
    InKey(KeyMode),
    ObjExpectColon(PropValueSpec),
    ExpectValue(PropValueSpec),
    ArrExpectValOrEnd,
    ArrExpectVal,
    InString(StrMode),
    InNumber { stage: NumStage, int_only: bool },
    InLit { target: [u8; 5], len: u8, pos: u8 },
    AfterValue,
    Complete,
}

#[derive(Clone, Copy, Debug)]
struct AutoState {
    phase: Phase,
    stack: [Frame; MAX_DEPTH_CAP],
    depth: u8,
    str_sub: StrSubState,
    root_schema: u16,
    ws_run: u8,
}

impl AutoState {
    fn new(root_schema: u16) -> Self {
        Self {
            phase: Phase::BeforeRoot,
            stack: [Frame::Empty; MAX_DEPTH_CAP],
            depth: 0,
            str_sub: StrSubState::new(),
            root_schema,
            ws_run: 0,
        }
    }

    #[inline]
    fn accept_ws(&mut self, limits: GrammarLimits) -> bool {
        if self.ws_run >= limits.max_ws_run {
            false
        } else {
            self.ws_run += 1;
            true
        }
    }

    #[inline]
    fn is_complete(&self) -> bool {
        matches!(self.phase, Phase::Complete)
    }

    #[inline]
    fn top(&self) -> Frame {
        self.stack[(self.depth - 1) as usize]
    }

    #[inline]
    fn top_mut(&mut self) -> &mut Frame {
        &mut self.stack[(self.depth - 1) as usize]
    }

    fn push_obj(&mut self, schema_idx: u16, arena: &SchemaArena) -> bool {
        if self.depth as usize >= MAX_DEPTH_CAP {
            return false;
        }
        self.stack[self.depth as usize] = Frame::Obj { schema_idx, seen_mask: 0, items: 0 };
        self.depth += 1;
        let has_req = schema_idx != NO_SCHEMA && !arena.schemas[schema_idx as usize].required_satisfied(0);
        self.phase = if has_req { Phase::ObjExpectKey } else { Phase::ObjExpectKeyOrEnd };
        true
    }

    fn push_arr(&mut self) -> bool {
        if self.depth as usize >= MAX_DEPTH_CAP {
            return false;
        }
        self.stack[self.depth as usize] = Frame::Arr { items: 0 };
        self.depth += 1;
        self.phase = Phase::ArrExpectValOrEnd;
        true
    }

    fn pop_container(&mut self) {
        self.depth -= 1;
        self.stack[self.depth as usize] = Frame::Empty;
        if self.depth == 0 {
            self.phase = Phase::Complete;
        } else {
            self.phase = Phase::AfterValue;
        }
    }

    fn begin_key(&mut self, arena: &SchemaArena) -> bool {
        let Frame::Obj { schema_idx, seen_mask, items } = self.top() else {
            return false;
        };
        let next_items = items.saturating_add(1);
        *self.top_mut() = Frame::Obj { schema_idx, seen_mask, items: next_items };
        self.str_sub = StrSubState::new();
        if schema_idx == NO_SCHEMA {
            self.phase = Phase::InKey(KeyMode::Free { schema_idx: NO_SCHEMA, cand_mask: 0, pos: 0 });
            return true;
        }
        let schema = &arena.schemas[schema_idx as usize];
        let unseen = schema.unseen_mask(seen_mask);
        if !schema.required_satisfied(seen_mask) || !schema.allow_additional {
            if unseen == 0 {
                return false;
            }
            self.phase = Phase::InKey(KeyMode::Restricted { schema_idx, cand_mask: unseen, pos: 0 });
            true
        } else {
            self.phase = Phase::InKey(KeyMode::Free { schema_idx, cand_mask: unseen, pos: 0 });
            true
        }
    }

    fn start_value(&mut self, b: u8, mut spec: PropValueSpec, arena: &SchemaArena, limits: GrammarLimits) -> bool {
        if is_json_ws(b) {
            return self.accept_ws(limits);
        }
        self.ws_run = 0;
        if self.depth >= limits.max_depth {
            spec.mask = spec.mask.without_containers_if_possible();
        }
        match b {
            b'{' if spec.mask.contains(TypeMask::OBJ) => self.push_obj(spec.nested_schema, arena),
            b'[' if spec.mask.contains(TypeMask::ARR) => self.push_arr(),
            b'"' if spec.mask.contains(TypeMask::STR) => {
                self.str_sub = StrSubState::new();
                if spec.fixed_str != NO_FIXED_STR {
                    self.phase = Phase::InString(StrMode::Fixed { fixed_str: spec.fixed_str, pos: 0 });
                } else {
                    self.phase = Phase::InString(StrMode::Free);
                }
                true
            }
            b'-' if spec.mask.contains(TypeMask::NUM) || spec.mask.contains(TypeMask::INT) => {
                let int_only = !spec.mask.contains(TypeMask::NUM);
                self.phase = Phase::InNumber { stage: NumStage::Sign, int_only };
                true
            }
            b'0' if spec.mask.contains(TypeMask::NUM) || spec.mask.contains(TypeMask::INT) => {
                let int_only = !spec.mask.contains(TypeMask::NUM);
                self.phase = Phase::InNumber { stage: NumStage::Zero, int_only };
                true
            }
            b'1'..=b'9' if spec.mask.contains(TypeMask::NUM) || spec.mask.contains(TypeMask::INT) => {
                let int_only = !spec.mask.contains(TypeMask::NUM);
                self.phase = Phase::InNumber { stage: NumStage::Int { digits: 1 }, int_only };
                true
            }
            b't' if spec.mask.contains(TypeMask::BOOL) => {
                self.phase = Phase::InLit { target: *b"true\0", len: 4, pos: 1 };
                true
            }
            b'f' if spec.mask.contains(TypeMask::BOOL) => {
                self.phase = Phase::InLit { target: *b"false", len: 5, pos: 1 };
                true
            }
            b'n' if spec.mask.contains(TypeMask::NULL) => {
                self.phase = Phase::InLit { target: *b"null\0", len: 4, pos: 1 };
                true
            }
            _ => false,
        }
    }

    fn step_after_value(&mut self, b: u8, arena: &SchemaArena, limits: GrammarLimits) -> bool {
        if is_json_ws(b) {
            return self.accept_ws(limits);
        }
        self.ws_run = 0;
        match self.top() {
            Frame::Obj { schema_idx, seen_mask, items } => {
                let (req_ok, more_ok) = if schema_idx == NO_SCHEMA {
                    (true, items < limits.max_container_items)
                } else {
                    let s = &arena.schemas[schema_idx as usize];
                    let req_sat = s.required_satisfied(seen_mask);
                    let more = if !req_sat {
                        s.unseen_mask(seen_mask) != 0
                    } else {
                        items < limits.max_container_items && s.can_add_more_keys(seen_mask)
                    };
                    (req_sat, more)
                };
                match b {
                    b',' if more_ok => {
                        self.phase = Phase::ObjExpectKey;
                        true
                    }
                    b'}' if req_ok => {
                        self.pop_container();
                        true
                    }
                    _ => false,
                }
            }
            Frame::Arr { items } => match b {
                b',' if items < limits.max_container_items => {
                    self.phase = Phase::ArrExpectVal;
                    true
                }
                b']' => {
                    self.pop_container();
                    true
                }
                _ => false,
            },
            Frame::Empty => false,
        }
    }

    fn step_byte(&mut self, b: u8, arena: &SchemaArena, limits: GrammarLimits) -> bool {
        match self.phase {
            Phase::BeforeRoot => {
                if is_json_ws(b) {
                    return self.accept_ws(limits);
                }
                if b == b'{' {
                    self.ws_run = 0;
                    return self.push_obj(self.root_schema, arena);
                }
                false
            }
            Phase::ObjExpectKeyOrEnd => {
                if is_json_ws(b) {
                    return self.accept_ws(limits);
                }
                self.ws_run = 0;
                if b == b'}' {
                    let Frame::Obj { schema_idx, seen_mask, .. } = self.top() else {
                        return false;
                    };
                    if schema_idx != NO_SCHEMA && !arena.schemas[schema_idx as usize].required_satisfied(seen_mask) {
                        return false;
                    }
                    self.pop_container();
                    return true;
                }
                if b == b'"' {
                    let can_key = match self.top() {
                        Frame::Obj { schema_idx, seen_mask, items } => {
                            items < limits.max_container_items
                                && (schema_idx == NO_SCHEMA
                                    || arena.schemas[schema_idx as usize].can_add_more_keys(seen_mask))
                        }
                        _ => false,
                    };
                    return can_key && self.begin_key(arena);
                }
                false
            }
            Phase::ObjExpectKey => {
                if is_json_ws(b) {
                    return self.accept_ws(limits);
                }
                if b == b'"' {
                    self.ws_run = 0;
                    return self.begin_key(arena);
                }
                false
            }
            Phase::InKey(mode) => match mode {
                KeyMode::Restricted { schema_idx, cand_mask, pos } => {
                    let schema = &arena.schemas[schema_idx as usize];
                    let pos_usize = pos as usize;
                    if b == b'"' {
                        let mut matched_idx: Option<usize> = None;
                        let mut m = cand_mask;
                        while m != 0 {
                            let j = m.trailing_zeros() as usize;
                            m &= m - 1;
                            if schema.props[j].escaped.len() == pos_usize {
                                matched_idx = Some(j);
                                break;
                            }
                        }
                        let Some(j) = matched_idx else {
                            return false;
                        };
                        if let Frame::Obj { seen_mask, .. } = self.top_mut() {
                            *seen_mask |= 1u64 << j;
                        }
                        self.phase = Phase::ObjExpectColon(schema.props[j].spec);
                        true
                    } else {
                        let mut next_mask = 0u64;
                        let mut m = cand_mask;
                        while m != 0 {
                            let j = m.trailing_zeros() as usize;
                            m &= m - 1;
                            if schema.props[j].escaped.get(pos_usize) == Some(&b) {
                                next_mask |= 1u64 << j;
                            }
                        }
                        if next_mask == 0 || pos == u16::MAX {
                            return false;
                        }
                        self.phase =
                            Phase::InKey(KeyMode::Restricted { schema_idx, cand_mask: next_mask, pos: pos + 1 });
                        true
                    }
                }
                KeyMode::Free { schema_idx, cand_mask, pos } => match self.str_sub.step(b, limits.max_string_bytes) {
                    StrStep::Invalid => false,
                    StrStep::Closed => {
                        let mut val_spec = PropValueSpec::any();
                        if schema_idx != NO_SCHEMA && cand_mask != 0 {
                            let schema = &arena.schemas[schema_idx as usize];
                            let pos_usize = pos as usize;
                            let mut m = cand_mask;
                            while m != 0 {
                                let j = m.trailing_zeros() as usize;
                                m &= m - 1;
                                if schema.props[j].escaped.len() == pos_usize {
                                    if let Frame::Obj { seen_mask, .. } = self.top_mut() {
                                        *seen_mask |= 1u64 << j;
                                    }
                                    val_spec = schema.props[j].spec;
                                    break;
                                }
                            }
                        }
                        self.phase = Phase::ObjExpectColon(val_spec);
                        true
                    }
                    StrStep::Continue => {
                        let mut next_mask = 0u64;
                        if schema_idx != NO_SCHEMA && cand_mask != 0 && (pos as usize) < usize::from(u16::MAX) {
                            let schema = &arena.schemas[schema_idx as usize];
                            let pos_usize = pos as usize;
                            let mut m = cand_mask;
                            while m != 0 {
                                let j = m.trailing_zeros() as usize;
                                m &= m - 1;
                                if schema.props[j].escaped.get(pos_usize) == Some(&b) {
                                    next_mask |= 1u64 << j;
                                }
                            }
                        }
                        self.phase = Phase::InKey(KeyMode::Free {
                            schema_idx,
                            cand_mask: next_mask,
                            pos: pos.saturating_add(1),
                        });
                        true
                    }
                },
            },
            Phase::ObjExpectColon(spec) => {
                if is_json_ws(b) {
                    return self.accept_ws(limits);
                }
                if b == b':' {
                    self.ws_run = 0;
                    self.phase = Phase::ExpectValue(spec);
                    return true;
                }
                false
            }
            Phase::ExpectValue(spec) => self.start_value(b, spec, arena, limits),
            Phase::ArrExpectValOrEnd => {
                if is_json_ws(b) {
                    return self.accept_ws(limits);
                }
                self.ws_run = 0;
                if b == b']' {
                    self.pop_container();
                    return true;
                }
                if let Frame::Arr { items } = self.top_mut() {
                    if *items >= limits.max_container_items {
                        return false;
                    }
                    *items += 1;
                }
                self.start_value(b, PropValueSpec::any(), arena, limits)
            }
            Phase::ArrExpectVal => {
                if is_json_ws(b) {
                    return self.accept_ws(limits);
                }
                self.ws_run = 0;
                if let Frame::Arr { items } = self.top_mut() {
                    *items = items.saturating_add(1);
                }
                self.start_value(b, PropValueSpec::any(), arena, limits)
            }
            Phase::InString(mode) => match mode {
                StrMode::Free => match self.str_sub.step(b, limits.max_string_bytes) {
                    StrStep::Continue => true,
                    StrStep::Closed => {
                        self.phase = Phase::AfterValue;
                        true
                    }
                    StrStep::Invalid => false,
                },
                StrMode::Fixed { fixed_str, pos } => {
                    let target = &arena.fixed_strings[fixed_str as usize];
                    let pos_usize = pos as usize;
                    if pos_usize == target.len() {
                        if b == b'"' {
                            self.phase = Phase::AfterValue;
                            true
                        } else {
                            false
                        }
                    } else if target[pos_usize] == b {
                        self.phase = Phase::InString(StrMode::Fixed { fixed_str, pos: pos + 1 });
                        true
                    } else {
                        false
                    }
                }
            },
            Phase::InNumber { mut stage, int_only } => {
                if stage.step_digit_or_part(b, int_only) {
                    self.phase = Phase::InNumber { stage, int_only };
                    return true;
                }
                if !stage.is_complete() {
                    return false;
                }
                self.phase = Phase::AfterValue;
                self.step_after_value(b, arena, limits)
            }
            Phase::InLit { target, len, pos } => {
                if target[pos as usize] != b {
                    return false;
                }
                if pos + 1 == len {
                    self.phase = Phase::AfterValue;
                } else {
                    self.phase = Phase::InLit { target, len, pos: pos + 1 };
                }
                true
            }
            Phase::AfterValue => self.step_after_value(b, arena, limits),
            Phase::Complete => {
                if is_json_ws(b) {
                    self.accept_ws(limits)
                } else {
                    false
                }
            }
        }
    }
}

/// Standalone byte-level JSON object pushdown grammar (root `{ ... }` with optional schema).
#[cfg(test)]
#[derive(Clone, Debug)]
pub struct JsonObjectGrammar {
    arena: Arc<SchemaArena>,
    state: AutoState,
    limits: GrammarLimits,
}

#[cfg(test)]
impl JsonObjectGrammar {
    pub fn new() -> Self {
        Self {
            arena: Arc::new(SchemaArena::default()),
            state: AutoState::new(NO_SCHEMA),
            limits: GrammarLimits::DEFAULT,
        }
    }

    pub fn with_schema(schema: &Value) -> Self {
        let mut arena = SchemaArena::default();
        let root = arena.compile_object_schema(schema, 0);
        Self { arena: Arc::new(arena), state: AutoState::new(root), limits: GrammarLimits::DEFAULT }
    }

    pub fn with_limits(mut self, limits: GrammarLimits) -> Self {
        self.limits = limits;
        self
    }

    #[inline]
    pub fn is_complete(&self) -> bool {
        self.state.is_complete()
    }

    #[inline]
    pub fn accepts_bytes(&self, bytes: &[u8]) -> bool {
        let mut next = self.state;
        for &b in bytes {
            if !next.step_byte(b, &self.arena, self.limits) {
                return false;
            }
        }
        true
    }

    #[inline]
    pub fn advance_bytes(&mut self, bytes: &[u8]) -> bool {
        let mut next = self.state;
        for &b in bytes {
            if !next.step_byte(b, &self.arena, self.limits) {
                return false;
            }
        }
        self.state = next;
        true
    }
}

// Inline `Copy` state avoids heap allocation per candidate token during masked sampling.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, Debug)]
enum SpanMode {
    Idle,
    Header,
    Json(AutoState),
}

/// Per-lane tool-call span tracker & JSON pushdown mask.
///
/// Tracks `<|message_model|><tool_name><|content_invoke_tool_json|>` -> activates [`AutoState`]
/// over the JSON body -> permits `<|end_message|>` (`end_tok`) ONLY once the root `{...}` object
/// is balanced and complete -> returns to `Idle` after `<|end_message|>`.
#[derive(Clone, Debug)]
pub struct ToolCallGrammar {
    catalog: Arc<ToolCatalog>,
    header_buf: Vec<u8>,
    dynamic_arena: Option<SchemaArena>,
    mode: SpanMode,
    limits: GrammarLimits,
}

impl ToolCallGrammar {
    pub fn new(catalog: Arc<ToolCatalog>) -> Self {
        Self {
            catalog,
            header_buf: Vec::with_capacity(64),
            dynamic_arena: None,
            mode: SpanMode::Idle,
            limits: GrammarLimits::DEFAULT,
        }
    }

    #[cfg(test)]
    pub fn with_limits(mut self, limits: GrammarLimits) -> Self {
        self.limits = limits;
        self
    }

    /// True while the lane is actively generating tokens inside `<|content_invoke_tool_json|> ... <|end_message|>`.
    #[inline]
    pub fn is_in_json(&self) -> bool {
        matches!(self.mode, SpanMode::Json(_))
    }

    #[inline]
    fn active_arena(&self) -> &SchemaArena {
        self.dynamic_arena.as_ref().unwrap_or(&self.catalog.arena)
    }

    /// Whether `token_id` is syntactically/structurally valid at the current position.
    #[inline]
    pub fn allows_token(&self, token_id: u32, vocab: &VocabBytes, end_tok: u32) -> bool {
        let SpanMode::Json(state) = &self.mode else {
            return true;
        };
        if token_id == end_tok {
            return state.is_complete();
        }
        let Some(bytes) = vocab.non_special_bytes(token_id) else {
            return false;
        };
        if bytes.is_empty() {
            return false;
        }
        let arena = self.active_arena();
        let mut next = *state;
        for &b in bytes {
            if !next.step_byte(b, arena, self.limits) {
                return false;
            }
        }
        true
    }

    /// Advance the lane's span state with the emitted token `tok`.
    #[allow(clippy::too_many_arguments)]
    pub fn observe_token(
        &mut self,
        tok: u32,
        vocab: &VocabBytes,
        msg_model_tok: u32,
        c_think_tok: u32,
        c_text_tok: u32,
        c_tool_tok: u32,
        end_tok: u32,
        end_sampling_tok: u32,
        eos_tok: u32,
    ) {
        if tok == msg_model_tok {
            self.header_buf.clear();
            self.dynamic_arena = None;
            self.mode = SpanMode::Header;
            return;
        }
        if tok == c_think_tok || tok == c_text_tok || tok == end_tok || tok == end_sampling_tok || tok == eos_tok {
            self.header_buf.clear();
            self.dynamic_arena = None;
            self.mode = SpanMode::Idle;
            return;
        }
        if tok == c_tool_tok {
            let header = std::str::from_utf8(&self.header_buf).ok().map(str::trim).unwrap_or("");
            self.dynamic_arena = None;
            let root_schema = if let Some(entry) = self.catalog.tools.iter().find(|t| t.name == header) {
                entry.wrapper_schema
            } else if !header.is_empty() {
                let mut dyn_arena = SchemaArena::default();
                let root = dyn_arena.build_tool_wrapper_schema(Some(header), NO_SCHEMA);
                self.dynamic_arena = Some(dyn_arena);
                root
            } else {
                self.catalog.fallback_wrapper_schema
            };
            self.mode = SpanMode::Json(AutoState::new(root_schema));
            return;
        }
        match &mut self.mode {
            SpanMode::Idle => {}
            SpanMode::Header => {
                if let Some(bytes) = vocab.non_special_bytes(tok) {
                    if self.header_buf.len() + bytes.len() <= 256 {
                        self.header_buf.extend_from_slice(bytes);
                    }
                }
            }
            SpanMode::Json(state) => {
                if let Some(bytes) = vocab.non_special_bytes(tok) {
                    let arena_ptr: *const SchemaArena = self.dynamic_arena.as_ref().unwrap_or(&self.catalog.arena);
                    // SAFETY: `arena_ptr` borrows `self.dynamic_arena` or `self.catalog.arena`, neither
                    // of which is mutated while advancing `state` (`Copy`).
                    let arena = unsafe { &*arena_ptr };
                    let mut next = *state;
                    for &b in bytes {
                        let ok = next.step_byte(b, arena, self.limits);
                        debug_assert!(ok, "observed token violated active ToolCallGrammar");
                        if !ok {
                            break;
                        }
                    }
                    *state = next;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::Sampler;
    use serde_json::json;

    #[test]
    fn accepts_valid_nested_json_escapes_surrogates_and_utf8_across_splits() {
        let valid = r#" {
            "str": "hello \"world\" \\ \/ \b\f\n\r\t \u0041 \uD83D\uDE80 café 日本語",
            "nums": [0, -0, 42, -17, 3.14159, -0.001, 1e9, 2.5E-4, -9e+12],
            "lits": [true, false, null],
            "nested": {"empty_obj": {}, "empty_arr": [], "deep": [{"k": "v"}]}
        } "#;
        let bytes = valid.as_bytes();
        // Verify full-slice acceptance AND arbitrary 1..3-byte chunk splits (including mid-UTF-8!).
        for step in [1usize, 2, 3, 5, 7] {
            let mut g = JsonObjectGrammar::new();
            for chunk in bytes.chunks(step) {
                assert!(!g.is_complete() || chunk.iter().all(|&b| is_json_ws(b)));
                assert!(g.advance_bytes(chunk), "rejected valid chunk {chunk:?} (step={step})");
            }
            assert!(g.is_complete(), "expected complete JSON at step={step}");
        }
        let parsed: Value = serde_json::from_str(valid).unwrap();
        assert_eq!(parsed["nested"]["deep"][0]["k"], "v");

        let schema = json!({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]});
        let schema_g = JsonObjectGrammar::with_schema(&schema).with_limits(GrammarLimits::DEFAULT);
        assert!(schema_g.accepts_bytes(br#"{"a": 1}"#));
        assert!(!schema_g.accepts_bytes(br#"{"a": "bad"}"#));
    }

    #[test]
    fn rejects_malformed_json_and_invalid_utf8_or_surrogates() {
        let invalid_cases: &[(&str, &[u8])] = &[
            ("unquoted key", br#"{foo: 1}"#),
            ("single-quoted string", br#"{"a": 'hi'}"#),
            ("trailing comma in object", br#"{"a": 1,}"#),
            ("trailing comma in array", br#"{"a": [1, 2,]}"#),
            ("double comma", br#"{"a": [1,,2]}"#),
            ("leading zero integer", br#"{"a": 01}"#),
            ("negative leading zero", br#"{"a": -01}"#),
            ("bare decimal point", br#"{"a": .5}"#),
            ("trailing decimal point", br#"{"a": 1.}"#),
            ("incomplete exponent", br#"{"a": 1e}"#),
            ("raw newline in string", b"{\"a\": \"line1\nline2\"}"),
            ("raw nul in string", b"{\"a\": \"bad\x00\"}"),
            ("bad escape", br#"{"a": "\x41"}"#),
            ("lone low surrogate", br#"{"a": "\uDC00"}"#),
            ("high surrogate without low", br#"{"a": "\uD83Dx"}"#),
            ("high surrogate followed by high", br#"{"a": "\uD83D\uD83D"}"#),
            ("overlong utf8 C0", b"{\"a\": \"\xC0\xAF\"}"),
            ("surrogate utf8 ED A0 80", b"{\"a\": \"\xED\xA0\x80\"}"),
            ("out of range utf8 F4 90 80 80", b"{\"a\": \"\xF4\x90\x80\x80\"}"),
            ("root array instead of object", br#"[1, 2, 3]"#),
            ("extra brace after root", br#"{"a": 1}}"#),
            ("concatenated objects", br#"{"a": 1}{"b": 2}"#),
        ];

        for &(label, case) in invalid_cases {
            let mut g = JsonObjectGrammar::new();
            let accepted_all = g.advance_bytes(case);
            assert!(
                !accepted_all || !g.is_complete(),
                "grammar should reject {label}: {:?}",
                String::from_utf8_lossy(case)
            );
        }

        // Unclosed containers and truncated literals/numbers are never `is_complete()`.
        for incomplete in [
            br#"{"#.as_slice(),
            br#"{"a""#.as_slice(),
            br#"{"a":"#.as_slice(),
            br#"{"a": 1"#.as_slice(),
            br#"{"a": [1, 2"#.as_slice(),
            br#"{"a": tru"#.as_slice(),
            br#"{"a": 1."#.as_slice(),
        ] {
            let mut g = JsonObjectGrammar::new();
            assert!(g.advance_bytes(incomplete));
            assert!(!g.is_complete(), "unclosed input marked complete: {:?}", String::from_utf8_lossy(incomplete));
        }
    }

    #[test]
    fn enforces_tool_schema_required_keys_types_and_end_token_boundary() {
        let tk = Tokenizer::for_byte_test();
        let vocab = VocabBytes::from_tokenizer(&tk);
        let msg_model = tk.encode("<|message_model|>", true)[0];
        let c_think = tk.encode("<|content_thinking|>", true)[0];
        let c_text = tk.encode("<|content_text|>", true)[0];
        let c_tool = tk.encode("<|content_invoke_tool_json|>", true)[0];
        let end = tk.encode("<|end_message|>", true)[0];
        let end_sampling = tk.encode("<|content_model_end_sampling|>", true)[0];

        let req = json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "days": {"type": "integer"}
                        },
                        "required": ["city"],
                        "additionalProperties": false
                    }
                }
            }]
        });
        let catalog = ToolCatalog::from_request(&req).unwrap();
        let mut g = ToolCallGrammar::new(catalog);
        assert!(!g.is_in_json());

        let step = |g: &mut ToolCallGrammar, tok: u32| {
            g.observe_token(tok, &vocab, msg_model, c_think, c_text, c_tool, end, end_sampling, tk.eos);
        };

        step(&mut g, msg_model);
        for tok in tk.encode("get_weather", false) {
            step(&mut g, tok);
        }
        assert!(!g.is_in_json());
        step(&mut g, c_tool);
        assert!(g.is_in_json());
        assert!(!g.allows_token(end, &vocab, end), "premature <|end_message|> before JSON must be blocked");

        // Advance `{"name":"get_weather","args":{` and verify `}` is blocked before required `"city"`.
        for tok in tk.encode(r#"{"name":"get_weather","args":{"#, false) {
            assert!(g.allows_token(tok, &vocab, end));
            step(&mut g, tok);
        }
        let close_brace = tk.encode("}", false)[0];
        assert!(!g.allows_token(close_brace, &vocab, end), "closing args before required 'city' must be rejected");

        // Optional `"days"` must be an integer, not a string or float.
        for tok in tk.encode(r#""days":"#, false) {
            assert!(g.allows_token(tok, &vocab, end));
            step(&mut g, tok);
        }
        let quote_tok = tk.encode("\"", false)[0];
        assert!(!g.allows_token(quote_tok, &vocab, end), "string value for integer property 'days' must be rejected");
        for tok in tk.encode("3", false) {
            assert!(g.allows_token(tok, &vocab, end));
            step(&mut g, tok);
        }
        let dot_tok = tk.encode(".", false)[0];
        assert!(!g.allows_token(dot_tok, &vocab, end), "decimal point for integer property 'days' must be rejected");
        assert!(!g.allows_token(close_brace, &vocab, end), "still missing required 'city'");

        // Emit `,"city":"Seattle"}}` and verify `<|end_message|>` unlocks ONLY after root `}` closes.
        for tok in tk.encode(r#","city":"Seattle"}"#, false) {
            assert!(g.allows_token(tok, &vocab, end));
            step(&mut g, tok);
            assert!(!g.allows_token(end, &vocab, end), "<|end_message|> must remain blocked before outer '}}'");
        }
        assert!(g.allows_token(close_brace, &vocab, end));
        step(&mut g, close_brace);
        assert!(g.allows_token(end, &vocab, end), "<|end_message|> must unlock once outer JSON object completes");
        step(&mut g, end);
        assert!(!g.is_in_json(), "grammar must reset to idle after <|end_message|>");
    }

    #[test]
    fn hundred_of_hundred_adversarial_random_logit_tool_calls_parse_valid_json() {
        let tk = Tokenizer::for_byte_test();
        let vocab = VocabBytes::from_tokenizer(&tk);
        let msg_model = tk.encode("<|message_model|>", true)[0];
        let c_think = tk.encode("<|content_thinking|>", true)[0];
        let c_text = tk.encode("<|content_text|>", true)[0];
        let c_tool = tk.encode("<|content_invoke_tool_json|>", true)[0];
        let end = tk.encode("<|end_message|>", true)[0];
        let end_sampling = tk.encode("<|content_model_end_sampling|>", true)[0];

        let req = json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search_repo",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer"},
                            "case_sensitive": {"type": "boolean"},
                            "filters": {"type": "object"}
                        },
                        "required": ["query", "max_results"]
                    }
                }
            }]
        });
        let catalog = ToolCatalog::from_request(&req).unwrap();
        let vocab_len = tk.vocab_len();
        let mut logits = vec![0.0f32; vocab_len];

        // Tokens that frequently break naive JSON generators: boost them adversarially on odd trials!
        let traps: Vec<u32> = [
            end,
            msg_model,
            c_tool,
            tk.eos,
            tk.encode(",", false)[0],
            tk.encode("}", false)[0],
            tk.encode("]", false)[0],
            tk.encode(":", false)[0],
            tk.encode("\\", false)[0],
            tk.encode("0", false)[0],
            tk.encode(".", false)[0],
            tk.encode("\n", false)[0],
            tk.encode("\x00", false)[0],
        ]
        .into_iter()
        .collect();

        let mut rng_state = 0xD1CE_CAFE_BAAD_F00Du64;
        let mut next_f32 = || -> f32 {
            rng_state ^= rng_state >> 12;
            rng_state ^= rng_state << 25;
            rng_state ^= rng_state >> 27;
            let u = rng_state.wrapping_mul(0x2545F4914F6CDD1D) >> 40;
            (u as f32 / ((1u32 << 24) as f32)) * 8.0 - 4.0
        };

        for trial in 0..100u64 {
            let mut g = ToolCallGrammar::new(catalog.clone()).with_limits(GrammarLimits::adversarial_test());
            g.observe_token(msg_model, &vocab, msg_model, c_think, c_text, c_tool, end, end_sampling, tk.eos);
            for tok in tk.encode("search_repo", false) {
                g.observe_token(tok, &vocab, msg_model, c_think, c_text, c_tool, end, end_sampling, tk.eos);
            }
            g.observe_token(c_tool, &vocab, msg_model, c_think, c_text, c_tool, end, end_sampling, tk.eos);

            let temp = if trial % 5 == 0 { 0.0 } else { 1.1 };
            let mut sampler = Sampler::new(temp, 64, 0.98, 0.0, 10_000 + trial);
            let mut emitted_bytes = Vec::new();
            let mut finished_with_end = false;

            for step_idx in 0..2048usize {
                for x in logits.iter_mut() {
                    *x = next_f32();
                }
                if trial % 2 == 1 {
                    // Actively try to force premature <|end_message|>, trailing commas, unclosed braces,
                    // control chars, leading zeros, and bare backslashes!
                    for &trap in &traps {
                        logits[trap as usize] = 18.0 + (step_idx % 5) as f32;
                    }
                }
                let tok = sampler
                    .sample_next_filtered(&logits, None, 0.0, |id| g.allows_token(id, &vocab, end))
                    .unwrap_or_else(|e| panic!("trial {trial} step {step_idx} failed to sample: {e}"));
                g.observe_token(tok, &vocab, msg_model, c_think, c_text, c_tool, end, end_sampling, tk.eos);
                if tok == end {
                    finished_with_end = true;
                    break;
                }
                emitted_bytes.extend_from_slice(&tk.decode_bytes(tok));
            }

            assert!(
                finished_with_end,
                "trial {trial} did not reach <|end_message|>; partial={:?}",
                String::from_utf8_lossy(&emitted_bytes)
            );
            let body_str = std::str::from_utf8(&emitted_bytes)
                .unwrap_or_else(|e| panic!("trial {trial} emitted invalid UTF-8 {emitted_bytes:?}: {e}"));
            let parsed: Value = serde_json::from_str(body_str)
                .unwrap_or_else(|e| panic!("trial {trial} emitted invalid JSON {body_str:?}: {e}"));
            assert_eq!(
                parsed.get("name").and_then(Value::as_str).unwrap_or("search_repo"),
                "search_repo",
                "trial {trial}: unexpected tool name in {parsed}"
            );
            let args = parsed
                .get("args")
                .and_then(Value::as_object)
                .unwrap_or_else(|| panic!("trial {trial}: missing args object in {parsed}"));
            assert!(
                args.get("query").and_then(Value::as_str).is_some(),
                "trial {trial}: missing string 'query' in {parsed}"
            );
            assert!(
                args.get("max_results").and_then(Value::as_i64).is_some(),
                "trial {trial}: missing integer 'max_results' in {parsed}"
            );
        }
    }
}
