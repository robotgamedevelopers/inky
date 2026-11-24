//! Byte-level BPE tokenizer for Inkling 2 (GPT-4o / o200k style pre-tokenizer), built from GGUF vocab + merges.

use crate::gguf::{Gguf, Value};
use crate::unicode_tables::*;
use std::collections::HashMap;

pub struct Tokenizer {
    id_to_tok: Vec<String>,
    tok_type: Vec<i64>,
    tok_to_id: HashMap<String, u32>,
    /// (left id, right id) -> (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    specials: Vec<(String, u32)>,
    byte_to_id: [u32; 256],
    uni_to_byte: HashMap<char, u8>,
    pub eos: u32,
}

fn uflags(c: u32) -> u8 {
    let i = RUN_START.partition_point(|&s| s <= c);
    RUN_FLAGS[i - 1]
}
#[inline] fn is_l(f: u8) -> bool { f & (U_LU | U_LL | U_LT | U_LM | U_LO) != 0 }
#[inline] fn is_n(f: u8) -> bool { f & U_N != 0 }
#[inline] fn is_ws(f: u8) -> bool { f & U_WS != 0 }
#[inline] fn is_ucls(f: u8) -> bool { f & (U_LU | U_LT | U_LM | U_LO | U_M) != 0 }
#[inline] fn is_lcls(f: u8) -> bool { f & (U_LL | U_LM | U_LO | U_M) != 0 }

impl Tokenizer {
    pub fn load(g: &Gguf) -> Tokenizer {
        let id_to_tok = match &g.kv["tokenizer.ggml.tokens"] { Value::AS(v) => v.clone(), _ => panic!("no tokens") };
        let tok_type = match &g.kv["tokenizer.ggml.token_type"] { Value::AI(v) => v.clone(), _ => panic!("no token types") };
        let merges_s = match &g.kv["tokenizer.ggml.merges"] { Value::AS(v) => v, _ => panic!("no merges") };
        let mut tok_to_id = HashMap::with_capacity(id_to_tok.len() * 2);
        for (i, t) in id_to_tok.iter().enumerate() { tok_to_id.insert(t.clone(), i as u32); }
        let mut merges = HashMap::with_capacity(merges_s.len() * 2);
        for (rank, m) in merges_s.iter().enumerate() {
            let (a, b) = m.split_once(' ').expect("bad merge");
            if let (Some(&ia), Some(&ib), Some(&im)) = (tok_to_id.get(a), tok_to_id.get(b), tok_to_id.get(&format!("{a}{b}"))) {
                merges.insert((ia, ib), (rank as u32, im));
            }
        }
        let mut specials: Vec<(String, u32)> = id_to_tok.iter().enumerate()
            .filter(|(i, _)| tok_type[*i] == 3).map(|(i, t)| (t.clone(), i as u32)).collect();
        specials.sort_by_key(|s| std::cmp::Reverse(s.0.len()));
        // GPT-2 byte <-> unicode mapping
        let mut byte_to_id = [0u32; 256];
        let mut uni_to_byte = HashMap::new();
        let mut n = 0u32;
        for b in 0..256u32 {
            let keep = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
            let cp = if keep { b } else { n += 1; 255 + n };
            let ch = char::from_u32(cp).unwrap();
            byte_to_id[b as usize] = *tok_to_id.get(&ch.to_string()).expect("byte token missing");
            uni_to_byte.insert(ch, b as u8);
        }
        let eos = g.u("tokenizer.ggml.eos_token_id") as u32;
        Tokenizer { id_to_tok, tok_type, tok_to_id, merges, specials, byte_to_id, uni_to_byte, eos }
    }

    pub fn is_special(&self, id: u32) -> bool { self.tok_type.get(id as usize) == Some(&3) }

    /// Raw bytes of one token (special tokens rendered literally).
    pub fn decode_bytes(&self, id: u32) -> Vec<u8> {
        let t = &self.id_to_tok[id as usize];
        if self.is_special(id) { return t.as_bytes().to_vec(); }
        let mut out = Vec::with_capacity(t.len());
        for ch in t.chars() {
            match self.uni_to_byte.get(&ch) {
                Some(b) => out.push(*b),
                None => { let mut buf = [0u8; 4]; out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes()); }
            }
        }
        out
    }

    /// GPT-4o pre-tokenizer regex, hand-matched:
    ///  [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
    /// |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
    /// |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
    fn split(&self, s: &[char]) -> Vec<(usize, usize)> {
        let n = s.len();
        let f: Vec<u8> = s.iter().map(|&c| uflags(c as u32)).collect();
        let lower = |c: char| c.to_ascii_lowercase();
        let contraction = |e: usize| -> usize {
            if e < n && s[e] == '\'' && e + 1 < n {
                let a = lower(s[e + 1]);
                let b = if e + 2 < n { lower(s[e + 2]) } else { '\0' };
                if (a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l') { return e + 3; }
                if matches!(a, 's' | 't' | 'm' | 'd') { return e + 2; }
            }
            e
        };
        let is_p = |i: usize| i < n && s[i] != '\r' && s[i] != '\n' && !is_l(f[i]) && !is_n(f[i]);
        let body1 = |k: usize| -> Option<usize> {
            let mut u = k;
            while u < n && is_ucls(f[u]) { u += 1; }
            let mut t = u + 1;
            while t > k {
                t -= 1;
                if t < n && is_lcls(f[t]) {
                    let mut e = t;
                    while e < n && is_lcls(f[e]) { e += 1; }
                    return Some(e);
                }
            }
            None
        };
        let body2 = |k: usize| -> Option<usize> {
            let mut u = k;
            while u < n && is_ucls(f[u]) { u += 1; }
            if u == k { return None; }
            let mut e = u;
            while e < n && is_lcls(f[e]) { e += 1; }
            Some(e)
        };
        let mut spans = Vec::new();
        let mut i = 0;
        while i < n {
            let mut end: Option<usize> = None;
            for alt in 0..2 {
                let body = |k: usize| if alt == 0 { body1(k) } else { body2(k) };
                if is_p(i) {
                    if let Some(e) = body(i + 1) { end = Some(contraction(e)); break; }
                }
                if let Some(e) = body(i) { end = Some(contraction(e)); break; }
            }
            if end.is_none() && is_n(f[i]) {
                let mut e = i;
                while e < n && e < i + 3 && is_n(f[e]) { e += 1; }
                end = Some(e);
            }
            if end.is_none() {
                let cls = |j: usize| j < n && !is_ws(f[j]) && !is_l(f[j]) && !is_n(f[j]);
                let mut k = i;
                if s[k] == ' ' && cls(k + 1) { k += 1; }
                if cls(k) {
                    let mut e = k;
                    while cls(e) { e += 1; }
                    while e < n && matches!(s[e], '\r' | '\n' | '/') { e += 1; }
                    end = Some(e);
                }
            }
            if end.is_none() && is_ws(f[i]) {
                let mut w = i;
                while w < n && is_ws(f[w]) { w += 1; }
                let last = (i..w).rev().find(|&j| s[j] == '\r' || s[j] == '\n');
                end = Some(if let Some(l) = last { l + 1 } else if w == n { w } else if w - i >= 2 { w - 1 } else { w });
            }
            let e = match end { Some(e) if e > i => e, _ => i + 1 };
            spans.push((i, e));
            i = e;
        }
        spans
    }

    fn bpe(&self, piece: &str, out: &mut Vec<u32>) {
        // fast path: whole piece is a token
        let mapped: String = piece.bytes().map(|b| self.id_to_tok[self.byte_to_id[b as usize] as usize].as_str()).collect();
        if let Some(&id) = self.tok_to_id.get(&mapped) { out.push(id); return; }
        let mut sym: Vec<u32> = piece.bytes().map(|b| self.byte_to_id[b as usize]).collect();
        loop {
            let mut best: Option<(u32, usize, u32)> = None;
            for j in 0..sym.len().saturating_sub(1) {
                if let Some(&(rank, merged)) = self.merges.get(&(sym[j], sym[j + 1])) {
                    if best.map_or(true, |b| rank < b.0) { best = Some((rank, j, merged)); }
                }
            }
            match best {
                Some((_, j, m)) => { sym[j] = m; sym.remove(j + 1); }
                None => break,
            }
        }
        out.extend_from_slice(&sym);
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() { return; }
        let cps: Vec<char> = text.chars().collect();
        for (a, b) in self.split(&cps) {
            let piece: String = cps[a..b].iter().collect();
            self.bpe(&piece, out);
        }
    }

    /// Encode text; when `parse_special`, literal `<|...|>` control tokens become their ids.
    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        let mut plain_start = 0;
        let bytes = text.as_bytes();
        let mut pos = 0;
        while pos < bytes.len() {
            if parse_special && bytes[pos] == b'<' {
                if let Some((s, id)) = self.specials.iter().find(|(s, _)| text[pos..].starts_with(s.as_str())) {
                    self.encode_plain(&text[plain_start..pos], &mut out);
                    out.push(*id);
                    pos += s.len();
                    plain_start = pos;
                    continue;
                }
            }
            pos += 1;
        }
        self.encode_plain(&text[plain_start..], &mut out);
        out
    }
}

#[cfg(test)]
impl Tokenizer {
    /// A tiny tokenizer for tests that render without a GGUF: the serve specials and an EOS are
    /// single tokens, every printable ASCII character is its own token, and there are no merges,
    /// so plain text encodes one token per character. Newlines and non-ASCII bytes fall back to
    /// `?`; test strings stay plain ASCII.
    pub(crate) fn for_test() -> Tokenizer {
        let special_strs = ["<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>",
            "<|content_text|>", "<|content_thinking|>", "<|content_xml|>", "<|content_invoke_tool_json|>",
            "<|end_message|>", "<|content_model_end_sampling|>", "<|eos|>"];
        let mut id_to_tok: Vec<String> = special_strs.iter().map(|s| s.to_string()).collect();
        let mut tok_type: Vec<i64> = vec![3; special_strs.len()];
        for b in 32u8..127 { id_to_tok.push((b as char).to_string()); tok_type.push(0); }
        let mut tok_to_id = HashMap::with_capacity(id_to_tok.len() * 2);
        for (i, t) in id_to_tok.iter().enumerate() { tok_to_id.insert(t.clone(), i as u32); }
        let mut specials: Vec<(String, u32)> = special_strs.iter().enumerate().map(|(i, s)| (s.to_string(), i as u32)).collect();
        specials.sort_by_key(|s| std::cmp::Reverse(s.0.len()));
        let fallback = *tok_to_id.get("?").unwrap();
        let mut byte_to_id = [0u32; 256];
        for b in 0..256usize {
            byte_to_id[b] = if (32..127).contains(&b) { *tok_to_id.get(&(b as u8 as char).to_string()).unwrap() } else { fallback };
        }
        let mut uni_to_byte = HashMap::new();
        for b in 32u8..127 { uni_to_byte.insert(b as char, b); }
        let eos = (special_strs.len() - 1) as u32;
        Tokenizer { id_to_tok, tok_type, tok_to_id, merges: HashMap::new(), specials, byte_to_id, uni_to_byte, eos }
    }
}
