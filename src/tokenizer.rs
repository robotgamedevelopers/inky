//! Byte-level BPE tokenizer for Inkling 2 (GPT-4o / o200k style pre-tokenizer), built from GGUF vocab + merges.

use crate::gguf::{invalid_model, Gguf, Value};
use crate::unicode_tables::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::io;

pub struct Tokenizer {
    id_to_tok: Vec<String>,
    tok_type: Vec<i64>,
    tok_to_id: HashMap<String, u32>,
    /// (left id, right id) -> (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Maximum input bytes represented by one byte or merge symbol.
    max_bpe_token_bytes: usize,
    specials: Vec<(String, u32)>,
    byte_to_id: [u32; 256],
    uni_to_byte: HashMap<char, u8>,
    pub eos: u32,
}

struct BpeSymbol {
    id: u32,
    prev: usize,
    next: usize,
}

fn uflags(c: u32) -> u8 {
    let i = RUN_START.partition_point(|&s| s <= c);
    RUN_FLAGS[i.saturating_sub(1)]
}
#[inline] fn is_l(f: u8) -> bool { f & (U_LU | U_LL | U_LT | U_LM | U_LO) != 0 }
#[inline] fn is_n(f: u8) -> bool { f & U_N != 0 }
#[inline] fn is_ws(f: u8) -> bool { f & U_WS != 0 }
#[inline] fn is_ucls(f: u8) -> bool { f & (U_LU | U_LT | U_LM | U_LO | U_M) != 0 }
#[inline] fn is_lcls(f: u8) -> bool { f & (U_LL | U_LM | U_LO | U_M) != 0 }

impl Tokenizer {
    pub fn load(g: &Gguf) -> Tokenizer {
        Self::try_load(g).unwrap_or_else(|err| {
            eprintln!("[inky] {err}");
            std::process::exit(1);
        })
    }

    /// Fallibly validate and build the BPE tokenizer from GGUF metadata, returning
    /// `ErrorKind::InvalidData` on any missing, mistyped, or out-of-bounds metadata.
    pub fn try_load(g: &Gguf) -> io::Result<Tokenizer> {
        let id_to_tok = match g.kv.get("tokenizer.ggml.tokens") {
            Some(Value::AS(v)) if !v.is_empty() && u32::try_from(v.len()).is_ok() => v.clone(),
            Some(Value::AS(_)) => return Err(invalid_model("tokenizer.ggml.tokens count is empty or overflows u32")),
            _ => return Err(invalid_model("missing or invalid tokenizer.ggml.tokens")),
        };
        let tok_type = match g.kv.get("tokenizer.ggml.token_type") {
            Some(Value::AI(v)) if v.len() == id_to_tok.len() => v.clone(),
            Some(Value::AI(v)) => return Err(invalid_model(format_args!(
                "tokenizer.ggml.token_type length {} does not match {} tokens",
                v.len(),
                id_to_tok.len(),
            ))),
            _ => return Err(invalid_model("missing or invalid tokenizer.ggml.token_type")),
        };
        let merges_s = match g.kv.get("tokenizer.ggml.merges") {
            Some(Value::AS(v)) if u32::try_from(v.len()).is_ok() => v,
            Some(Value::AS(_)) => return Err(invalid_model("tokenizer.ggml.merges count overflows u32")),
            _ => return Err(invalid_model("missing or invalid tokenizer.ggml.merges")),
        };

        let mut tok_to_id = HashMap::with_capacity(id_to_tok.len().saturating_mul(2));
        for (i, t) in id_to_tok.iter().enumerate() {
            tok_to_id.insert(t.clone(), i as u32);
        }

        let mut merges = HashMap::with_capacity(merges_s.len().saturating_mul(2));
        let mut max_bpe_token_bytes = 1;
        for (rank, m) in merges_s.iter().enumerate() {
            let Some((a, b)) = m.split_once(' ') else {
                return Err(invalid_model(format_args!("malformed BPE merge at index {rank}")));
            };
            if a.is_empty() || b.is_empty() || b.contains(' ') {
                return Err(invalid_model(format_args!("malformed BPE merge at index {rank}")));
            }
            if let (Some(&ia), Some(&ib), Some(&im)) = (
                tok_to_id.get(a),
                tok_to_id.get(b),
                tok_to_id.get(&format!("{a}{b}")),
            ) {
                merges.insert((ia, ib), (rank as u32, im));
                // Every mapped character represents one original input byte.
                if let Some(merged_tok) = id_to_tok.get(im as usize) {
                    max_bpe_token_bytes = max_bpe_token_bytes.max(merged_tok.chars().count());
                }
            }
        }

        let mut specials: Vec<(String, u32)> = id_to_tok
            .iter()
            .zip(&tok_type)
            .enumerate()
            .filter(|&(_, (_, &ty))| ty == 3)
            .map(|(i, (t, _))| (t.clone(), i as u32))
            .collect();
        specials.sort_by_key(|s| Reverse(s.0.len()));

        // GPT-2 byte <-> unicode mapping
        let mut byte_to_id = [0u32; 256];
        let mut uni_to_byte = HashMap::with_capacity(512);
        let mut n = 0u32;
        for b in 0..256u32 {
            let keep = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
            let cp = if keep { b } else { n += 1; 255 + n };
            let ch = char::from_u32(cp)
                .ok_or_else(|| invalid_model(format_args!("invalid byte mapping codepoint {cp}")))?;
            let id = tok_to_id
                .get(&ch.to_string())
                .copied()
                .ok_or_else(|| invalid_model(format_args!("missing GPT-2 byte token for byte 0x{b:02x}")))?;
            byte_to_id[b as usize] = id;
            uni_to_byte.insert(ch, b as u8);
        }

        let eos_u64 = g.uint_or_bool("tokenizer.ggml.eos_token_id")?;
        let eos = u32::try_from(eos_u64)
            .ok()
            .filter(|&id| (id as usize) < id_to_tok.len())
            .ok_or_else(|| invalid_model(format_args!("tokenizer.ggml.eos_token_id {eos_u64} is out of range")))?;

        Ok(Tokenizer {
            id_to_tok,
            tok_type,
            tok_to_id,
            merges,
            max_bpe_token_bytes,
            specials,
            byte_to_id,
            uni_to_byte,
            eos,
        })
    }

    pub fn vocab_len(&self) -> usize { self.id_to_tok.len() }

    pub fn is_special(&self, id: u32) -> bool { self.tok_type.get(id as usize) == Some(&3) }

    /// Raw bytes of one token (special tokens rendered literally; out-of-range IDs return empty).
    pub fn decode_bytes(&self, id: u32) -> Vec<u8> {
        let Some(t) = self.id_to_tok.get(id as usize) else { return Vec::new(); };
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
    fn piece_end(s: &[char], f: &[u8], i: usize) -> usize {
        let n = s.len().min(f.len());
        if i >= n { return n; }
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
        for alt in 0..2 {
            let body = |k: usize| if alt == 0 { body1(k) } else { body2(k) };
            if is_p(i) {
                if let Some(e) = body(i + 1) { return contraction(e); }
            }
            if let Some(e) = body(i) { return contraction(e); }
        }
        if is_n(f[i]) {
            let mut e = i + 1;
            while e < n && e < i + 3 && is_n(f[e]) { e += 1; }
            return e;
        }
        let cls = |j: usize| j < n && !is_ws(f[j]) && !is_l(f[j]) && !is_n(f[j]);
        let mut k = i;
        if s[k] == ' ' && cls(k + 1) { k += 1; }
        if cls(k) {
            let mut e = k + 1;
            while cls(e) { e += 1; }
            while e < n && matches!(s[e], '\r' | '\n' | '/') { e += 1; }
            return e;
        }
        if is_ws(f[i]) {
            let mut w = i + 1;
            while w < n && is_ws(f[w]) { w += 1; }
            let last = (i..w).rev().find(|&j| s[j] == '\r' || s[j] == '\n');
            return if let Some(l) = last { l + 1 } else if w == n { w } else if w - i >= 2 { w - 1 } else { w };
        }
        i + 1
    }

    fn bpe(&self, piece: &str, out: &mut Vec<u32>, remaining: usize) -> bool {
        // fast path: whole piece is a token
        let mapped: String = piece.bytes()
            .map(|b| self.id_to_tok[self.byte_to_id[b as usize] as usize].as_str())
            .collect();
        if let Some(&id) = self.tok_to_id.get(&mapped) {
            if remaining == 0 { return false; }
            out.push(id);
            return true;
        }
        // A direct vocabulary match may be longer than any merge result.
        if piece.len() > remaining.saturating_mul(self.max_bpe_token_bytes) {
            return false;
        }
        // Keep the small-piece scan bounded and avoid setting up a heap for ordinary pieces.
        if piece.len() > 32 {
            return self.bpe_ranked(piece, out, remaining);
        }
        let mut sym: Vec<u32> = piece.bytes().map(|b| self.byte_to_id[b as usize]).collect();
        loop {
            let mut best: Option<(u32, usize, u32)> = None;
            for j in 0..sym.len().saturating_sub(1) {
                if let Some(&(rank, merged)) = self.merges.get(&(sym[j], sym[j + 1])) {
                    if best.is_none_or(|b| rank < b.0) { best = Some((rank, j, merged)); }
                }
            }
            match best {
                Some((_, j, m)) => { sym[j] = m; sym.remove(j + 1); }
                None => break,
            }
        }
        if sym.len() > remaining { return false; }
        out.extend_from_slice(&sym);
        true
    }

    fn bpe_ranked(&self, piece: &str, out: &mut Vec<u32>, remaining: usize) -> bool {
        // Link symbols by their original byte offsets, with `end` as the missing-link sentinel.
        let end = piece.len();
        let mut symbols: Vec<BpeSymbol> = piece.bytes().enumerate().map(|(i, b)| BpeSymbol {
            id: self.byte_to_id[b as usize],
            prev: i.checked_sub(1).unwrap_or(end),
            next: i + 1,
        }).collect();

        let candidate = |symbols: &[BpeSymbol], left: usize| {
            let right = symbols[left].next;
            if right == end { return None; }
            let &(rank, merged) = self.merges.get(&(symbols[left].id, symbols[right].id))?;
            // Byte positions keep equal ranks in left-to-right order after any merge.
            Some(Reverse((rank, left, symbols[right].next, merged)))
        };
        let mut candidates: BinaryHeap<_> = (0..end)
            .filter_map(|left| candidate(&symbols, left))
            .collect();
        let mut count = end;
        while let Some(Reverse((_, left, pair_end, merged))) = candidates.pop() {
            let right = symbols[left].next;
            // Any change to a live pair extends its end. Removed left symbols have no next link.
            if right == end || symbols[right].next != pair_end { continue; }
            symbols[left].id = merged;
            symbols[left].next = pair_end;
            symbols[right].next = end;
            if pair_end != end {
                symbols[pair_end].prev = left;
            }
            count -= 1;

            // Only the pairs that touch the merged symbol changed.
            let prev = symbols[left].prev;
            if prev != end {
                candidates.extend(candidate(&symbols, prev));
            }
            candidates.extend(candidate(&symbols, left));
        }

        if count > remaining { return false; }
        out.reserve(count);
        let mut i = 0;
        while i != end {
            out.push(symbols[i].id);
            i = symbols[i].next;
        }
        true
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        self.encode_plain_limited(text, out, usize::MAX);
    }

    /// Append plain-text pieces and return true if all fit within `max_len` total tokens.
    /// On overflow, leave only the complete pieces that fit.
    pub(crate) fn encode_plain_limited(&self, text: &str, out: &mut Vec<u32>, max_len: usize) -> bool {
        if text.is_empty() { return out.len() <= max_len; }
        if out.len() >= max_len { return false; }
        let cps: Vec<char> = text.chars().collect();
        let flags: Vec<u8> = cps.iter().map(|&c| uflags(c as u32)).collect();
        let mut start = 0;
        while start < cps.len() && out.len() < max_len {
            let end = Self::piece_end(&cps, &flags, start);
            let piece: String = cps[start..end].iter().collect();
            if !self.bpe(&piece, out, max_len - out.len()) { return false; }
            start = end;
        }
        start == cps.len()
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
        for (b, slot) in byte_to_id.iter_mut().enumerate() {
            *slot = if (32..127).contains(&b) { *tok_to_id.get(&(b as u8 as char).to_string()).unwrap() } else { fallback };
        }
        let mut uni_to_byte = HashMap::new();
        for b in 32u8..127 { uni_to_byte.insert(b as char, b); }
        let eos = (special_strs.len() - 1) as u32;
        Tokenizer { id_to_tok, tok_type, tok_to_id, merges: HashMap::new(), max_bpe_token_bytes: 1, specials, byte_to_id, uni_to_byte, eos }
    }

    pub(crate) fn for_byte_test() -> Tokenizer {
        let special_strs = ["<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>",
            "<|content_text|>", "<|content_thinking|>", "<|content_xml|>", "<|content_invoke_tool_json|>",
            "<|end_message|>", "<|content_model_end_sampling|>", "<|eos|>"];
        let mut id_to_tok: Vec<String> = special_strs.iter().map(|s| s.to_string()).collect();
        let mut tok_type: Vec<i64> = vec![3; special_strs.len()];
        let mut byte_to_id = [0u32; 256];
        let mut uni_to_byte = HashMap::new();
        let mut n = 0u32;
        for b in 0..256u32 {
            let keep = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
            let cp = if keep { b } else { n += 1; 255 + n };
            let ch = char::from_u32(cp).unwrap();
            let id = id_to_tok.len() as u32;
            id_to_tok.push(ch.to_string());
            tok_type.push(0);
            byte_to_id[b as usize] = id;
            uni_to_byte.insert(ch, b as u8);
        }
        let mut tok_to_id = HashMap::with_capacity(id_to_tok.len() * 2);
        for (i, t) in id_to_tok.iter().enumerate() { tok_to_id.insert(t.clone(), i as u32); }
        let mut specials: Vec<(String, u32)> = special_strs.iter().enumerate().map(|(i, s)| (s.to_string(), i as u32)).collect();
        specials.sort_by_key(|s| std::cmp::Reverse(s.0.len()));
        let eos = (special_strs.len() - 1) as u32;
        Tokenizer { id_to_tok, tok_type, tok_to_id, merges: HashMap::new(), max_bpe_token_bytes: 1, specials, byte_to_id, uni_to_byte, eos }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::ErrorKind;

    fn add_merge(t: &mut Tokenizer, left: &str, right: &str, rank: u32) -> u32 {
        let pair = (t.tok_to_id[left], t.tok_to_id[right]);
        let text = format!("{left}{right}");
        let id = *t.tok_to_id.entry(text.clone()).or_insert_with(|| {
            let id = t.id_to_tok.len() as u32;
            t.id_to_tok.push(text);
            t.tok_type.push(0);
            id
        });
        t.merges.insert(pair, (rank, id));
        t.max_bpe_token_bytes = t.max_bpe_token_bytes.max(t.id_to_tok[id as usize].chars().count());
        id
    }

    fn reference_piece(t: &Tokenizer, piece: &str) -> Vec<u32> {
        let mut symbols: Vec<u32> = piece.bytes().map(|b| t.byte_to_id[b as usize]).collect();
        let mapped: String = symbols.iter().map(|&id| t.id_to_tok[id as usize].as_str()).collect();
        if let Some(&id) = t.tok_to_id.get(&mapped) {
            return vec![id];
        }
        loop {
            let best = symbols.windows(2).enumerate().filter_map(|(i, pair)| {
                t.merges.get(&(pair[0], pair[1])).map(|&(rank, merged)| (rank, i, merged))
            }).min();
            let Some((_, i, merged)) = best else { return symbols };
            symbols[i] = merged;
            symbols.remove(i + 1);
        }
    }

    fn split_pieces(text: &str) -> Vec<String> {
        let cps: Vec<char> = text.chars().collect();
        let flags: Vec<u8> = cps.iter().map(|&c| uflags(c as u32)).collect();
        let mut out = Vec::new();
        let mut start = 0;
        while start < cps.len() {
            let end = Tokenizer::piece_end(&cps, &flags, start);
            assert!(end > start && end <= cps.len());
            out.push(cps[start..end].iter().collect());
            start = end;
        }
        out
    }

    #[test]
    fn bpe_matches_scan_for_overlapping_and_newly_ranked_pairs() {
        let mut t = Tokenizer::for_test();
        for (left, right, rank) in [
            ("a", "b", 5), ("b", "a", 5), ("a", "a", 6), ("b", "c", 8),
            ("ab", "a", 1), ("a", "ba", 2), ("c", "ab", 0), ("aa", "ab", 3),
            ("aba", "bc", 4), ("c", "a", 9), ("ca", "b", 7), ("?", "?", 3),
        ] {
            add_merge(&mut t, left, right, rank);
        }
        for piece in ["", "ab", "a", "cabaa", "éä", &"abababcaabb".repeat(23), &"éä".repeat(30)] {
            let mut actual = Vec::new();
            assert!(t.bpe(piece, &mut actual, usize::MAX));
            assert_eq!(actual, reference_piece(&t, piece), "{piece}");
        }
        let mut state = 17u32;
        for len in 0..160 {
            let piece: String = (0..len).map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (b'a' + ((state >> 16) % 3) as u8) as char
            }).collect();
            let mut actual = Vec::new();
            assert!(t.bpe(&piece, &mut actual, usize::MAX));
            assert_eq!(actual, reference_piece(&t, &piece), "{piece}");
        }
    }

    #[test]
    fn bpe_merges_a_long_piece_in_order() {
        let mut t = Tokenizer::for_test();
        let ab = add_merge(&mut t, "a", "b", 2);
        let abab = add_merge(&mut t, "ab", "ab", 1);
        let mut out = Vec::new();
        assert!(t.bpe(&"ab".repeat(65_537), &mut out, 32_769));
        assert_eq!(out.len(), 32_769);
        assert!(out[..32_768].iter().all(|&id| id == abab));
        assert_eq!(out[32_768], ab);
    }

    #[test]
    fn bpe_preserves_whole_piece_vocabulary_lookup() {
        let mut t = Tokenizer::for_test();
        let ab = add_merge(&mut t, "a", "b", 0);
        t.merges.clear();
        t.max_bpe_token_bytes = 1;
        let mut out = Vec::new();
        assert!(t.bpe("", &mut out, 0));
        assert!(out.is_empty());
        assert!(!t.bpe("ab", &mut out, 0));
        assert!(t.bpe("ab", &mut out, 1));
        assert_eq!(out, vec![ab]);
    }

    #[test]
    fn byte_bound_preserves_prefix_and_exact_capacity() {
        let mut t = Tokenizer::for_test();
        let pair = add_merge(&mut t, "?", "?", 0);
        let piece = "é".repeat(64);
        let text = format!("123{piece}");
        let mut out = vec![t.eos];
        assert!(!t.encode_plain_limited(&text, &mut out, 67));
        assert_eq!(out[1..], t.encode("123", false));
        out.truncate(1);
        assert!(t.encode_plain_limited(&text, &mut out, 68));
        assert_eq!(out.len(), 68);
        assert!(out[4..].iter().all(|&id| id == pair));
    }

    #[test]
    fn plain_encoding_stops_at_the_output_limit() {
        let mut t = Tokenizer::for_test();
        add_merge(&mut t, "a", "b", 0);
        let long_piece = "ab".repeat(40);
        let text = format!("abc {long_piece}");
        let expected = t.encode(&text, false);
        let mut out = vec![t.eos];
        assert!(t.encode_plain_limited("", &mut out, 1));
        assert!(!t.encode_plain_limited(&text, &mut out, 1));
        assert_eq!(out, vec![t.eos]);
        assert!(!t.encode_plain_limited(&text, &mut out, expected.len()));
        assert_eq!(out[1..], t.encode("abc", false));
        out.truncate(1);
        assert!(t.encode_plain_limited(&text, &mut out, expected.len() + 1));
        assert_eq!(out[1..], expected);
    }

    #[test]
    fn try_load_rejects_malformed_gguf_tokenizer_metadata_cleanly() {
        let seed = crate::gguf::test_builder::valid_gguf_with_tokenizer();
        let base = Gguf::open_bytes(&seed).unwrap();
        let ok_tk = Tokenizer::try_load(&base).unwrap();
        assert!(ok_tk.decode_bytes(u32::MAX).is_empty());

        let check_err = |f: fn(&mut Gguf)| {
            let mut g = Gguf::open_bytes(&seed).unwrap();
            f(&mut g);
            assert_eq!(Tokenizer::try_load(&g).err().unwrap().kind(), ErrorKind::InvalidData);
        };

        check_err(|g| { g.kv.remove("tokenizer.ggml.tokens"); });
        check_err(|g| { g.kv.insert("tokenizer.ggml.tokens".into(), Value::AS(Vec::new())); });
        check_err(|g| { g.kv.insert("tokenizer.ggml.tokens".into(), Value::U(1)); });
        check_err(|g| { g.kv.remove("tokenizer.ggml.token_type"); });
        check_err(|g| { g.kv.insert("tokenizer.ggml.token_type".into(), Value::AI(vec![1; 3])); });
        check_err(|g| { g.kv.remove("tokenizer.ggml.merges"); });
        check_err(|g| { g.kv.insert("tokenizer.ggml.merges".into(), Value::AS(vec!["nospace".into()])); });
        check_err(|g| { g.kv.insert("tokenizer.ggml.merges".into(), Value::AS(vec!["a b c".into()])); });
        check_err(|g| { g.kv.insert("tokenizer.ggml.merges".into(), Value::AS(vec![" b".into()])); });
        check_err(|g| {
            if let Some(Value::AS(toks)) = g.kv.get_mut("tokenizer.ggml.tokens") {
                toks[0] = "missing-byte-zero".into();
            }
        });
        check_err(|g| { g.kv.remove("tokenizer.ggml.eos_token_id"); });
        check_err(|g| { g.kv.insert("tokenizer.ggml.eos_token_id".into(), Value::I(-1)); });
        check_err(|g| { g.kv.insert("tokenizer.ggml.eos_token_id".into(), Value::U(999_999)); });
    }

    #[test]
    fn uflags_classifies_unicode_categories_accurately() {
        assert_ne!(uflags('A' as u32) & U_LU, 0);
        assert_ne!(uflags('z' as u32) & U_LL, 0);
        assert_ne!(uflags('ǅ' as u32) & U_LT, 0);
        assert_ne!(uflags('ʰ' as u32) & U_LM, 0);
        assert_ne!(uflags('漢' as u32) & U_LO, 0);
        assert_ne!(uflags('א' as u32) & U_LO, 0);
        assert_ne!(uflags('\u{0301}' as u32) & U_M, 0);
        assert_ne!(uflags('7' as u32) & U_N, 0);
        assert_ne!(uflags('٠' as u32) & U_N, 0);
        assert_ne!(uflags('²' as u32) & U_N, 0);
        assert_ne!(uflags(' ' as u32) & U_WS, 0);
        assert_ne!(uflags('\n' as u32) & U_WS, 0);
        assert_ne!(uflags('\u{00A0}' as u32) & U_WS, 0);
        assert_ne!(uflags('\u{3000}' as u32) & U_WS, 0);
        assert_eq!(uflags('🦀' as u32) & (U_LU | U_LL | U_LT | U_LM | U_LO | U_N | U_WS), 0);
    }

    #[test]
    fn piece_end_gpt4o_splitting_invariants() {
        // Contractions (lowercase and uppercase) stay attached to the preceding word.
        assert_eq!(
            split_pieces("don't I'M we're they'VE she'll he'd rock's"),
            vec!["don't", " I'M", " we're", " they'VE", " she'll", " he'd", " rock's"],
        );
        // GPT-4o `o200k` `body1` (`[\p{Lu}]*[\p{Ll}]+`) keeps `HTTPServer` together, splits
        // `httpServer` at the lowercase->uppercase boundary, and `body2` (`[\p{Lu}]+[\p{Ll}]*`) splits `HTTP99`.
        assert_eq!(
            split_pieces("HTTPServer httpServer HTTP99"),
            vec!["HTTPServer", " http", "Server", " HTTP", "99"],
        );
        // Numbers split into chunks of 1..=3 digits.
        assert_eq!(split_pieces("1234567890"), vec!["123", "456", "789", "0"]);
        // Punctuation absorbs optional leading space and trailing newlines/slashes.
        assert_eq!(split_pieces("a //\nnext"), vec!["a", " //\n", "next"]);
        // Indentation leaves the final space attached to the following word (`\s+(?!\S)`).
        assert_eq!(split_pieces("    let x = 42;\r\n  return x;"), vec![
            "   ", " let", " x", " =", " ", "42", ";\r\n", " ", " return", " x", ";",
        ]);
        // Combining accents and CJK characters stay grouped in word runs.
        assert_eq!(split_pieces("cafe\u{0301} 日本語123"), vec!["cafe\u{0301}", " 日本語", "123"]);
    }

    #[test]
    fn multi_kb_bpe_roundtrip_and_determinism() {
        let mut tk = Tokenizer::for_byte_test();
        let byte_str = |tk: &Tokenizer, b: u8| tk.id_to_tok[tk.byte_to_id[b as usize] as usize].clone();
        let sp = byte_str(&tk, b' ');
        let sp2_id = add_merge(&mut tk, &sp, &sp, 0);
        let sp2 = tk.id_to_tok[sp2_id as usize].clone();
        let _ = add_merge(&mut tk, &sp2, &sp2, 1);
        let f_s = byte_str(&tk, b'f');
        let n_s = byte_str(&tk, b'n');
        let _ = add_merge(&mut tk, &f_s, &n_s, 2);
        let t_s = byte_str(&tk, b't');
        let h_s = byte_str(&tk, b'h');
        let e_s = byte_str(&tk, b'e');
        let th_id = add_merge(&mut tk, &t_s, &h_s, 3);
        let th_s = tk.id_to_tok[th_id as usize].clone();
        let _ = add_merge(&mut tk, &th_s, &e_s, 4);

        let mut doc = String::with_capacity(24_000);
        for i in 0..240 {
            doc.push_str(&format!(
                "    fn step_{i}(the_value: usize) -> usize {{ /* café 日本語 🦀 {} */ the_value + {i} }}\n",
                "the".repeat(18),
            ));
        }
        assert!(doc.len() > 20_000);

        let plain_a = tk.encode(&doc, false);
        let plain_b = tk.encode(&doc, false);
        assert_eq!(plain_a, plain_b);
        let decoded_plain: Vec<u8> = plain_a.iter().flat_map(|&id| tk.decode_bytes(id)).collect();
        assert_eq!(decoded_plain, doc.as_bytes());

        let framed = format!("<|message_user|><|content_text|>{doc}<|end_message|><|eos|>");
        let spec_a = tk.encode(&framed, true);
        let spec_b = tk.encode(&framed, true);
        assert_eq!(spec_a, spec_b);
        assert!(tk.is_special(spec_a[0]));
        assert_eq!(*spec_a.last().unwrap(), tk.eos);
        let decoded_framed: Vec<u8> = spec_a.iter().flat_map(|&id| tk.decode_bytes(id)).collect();
        assert_eq!(decoded_framed, framed.as_bytes());
    }

    #[test]
    #[ignore = "requires local Inkling-2-Mini GGUF model"]
    fn real_gguf_tokenizer_74k_corpus_roundtrip_and_determinism() {
        let model_path = std::env::var("INKY_MODEL").unwrap_or_else(|_| {
            let home = std::env::var("HOME").expect("HOME");
            format!("{home}/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf")
        });
        let g = Gguf::open(&model_path).expect("open real GGUF");
        let tk = Tokenizer::try_load(&g).expect("try_load real GGUF tokenizer");

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut corpus = String::new();
        for rel in [
            "src/engine.rs",
            "src/serve.rs",
            "src/tokenizer.rs",
            "src/scheduler.rs",
            "docs/DESIGN.md",
            "docs/DSPARK.md",
        ] {
            let text = std::fs::read_to_string(root.join(rel)).expect(rel);
            corpus.push_str(&text);
            corpus.push('\n');
        }
        while corpus.len() < 320_000 {
            let snapshot = corpus.clone();
            corpus.push_str(&snapshot);
        }

        let ids_first = tk.encode(&corpus, false);
        let ids_second = tk.encode(&corpus, false);
        assert!(
            ids_first.len() >= 74_000,
            "expected >= 74,000 tokens from real GGUF corpus, got {}",
            ids_first.len(),
        );
        assert_eq!(ids_first, ids_second, "real GGUF BPE encoding must be 100% deterministic");

        let reconstructed: Vec<u8> = ids_first.iter().flat_map(|&id| tk.decode_bytes(id)).collect();
        assert_eq!(
            reconstructed,
            corpus.as_bytes(),
            "real GGUF 74K+ token corpus must roundtrip byte-for-byte",
        );
        println!(
            "real_gguf_tokenizer_74k_corpus_roundtrip_and_determinism: {} bytes -> {} tokens -> exact byte roundtrip",
            corpus.len(),
            ids_first.len(),
        );
    }

    #[test]
    fn uflags_tables_classify_ascii_and_unicode_categories() {
        for ch in ['A', 'Z', 'Ω', 'Ж'] {
            let f = uflags(ch as u32);
            assert!(is_l(f) && is_ucls(f) && !is_lcls(f) && !is_n(f) && !is_ws(f), "upper {ch:?}");
        }
        for ch in ['a', 'z', 'é', 'ω', 'ж'] {
            let f = uflags(ch as u32);
            assert!(is_l(f) && is_lcls(f) && !is_ucls(f) && !is_n(f) && !is_ws(f), "lower {ch:?}");
        }
        // Titlecase letter (ǅ): U_LT -> letter + ucls, not lcls.
        let lt = uflags('\u{01C5}' as u32);
        assert!(is_l(lt) && is_ucls(lt) && !is_lcls(lt));
        // Modifier letter (ʰ) and Lo CJK ideograph (中): letter + both ucls and lcls.
        for ch in ['\u{02B0}', '中', 'あ'] {
            let f = uflags(ch as u32);
            assert!(is_l(f) && is_ucls(f) && is_lcls(f) && !is_n(f) && !is_ws(f), "lm/lo {ch:?}");
        }
        // Combining acute accent (U+0301): mark -> both ucls and lcls, not standalone is_l.
        let mark = uflags('\u{0301}' as u32);
        assert!(!is_l(mark) && is_ucls(mark) && is_lcls(mark) && !is_n(mark) && !is_ws(mark));
        // Numbers across Nd / No scripts.
        for ch in ['0', '9', '²', '٣'] {
            let f = uflags(ch as u32);
            assert!(is_n(f) && !is_l(f) && !is_ws(f), "number {ch:?}");
        }
        // Whitespace across ASCII and Unicode Zs.
        for ch in [' ', '\t', '\r', '\n', '\u{00A0}', '\u{2003}', '\u{3000}'] {
            let f = uflags(ch as u32);
            assert!(is_ws(f) && !is_l(f) && !is_n(f), "ws {ch:?}");
        }
        // Punctuation, emoji, and supplementary-plane boundary.
        for ch in ['!', '/', '+', '🦀', '\u{10FFFF}'] {
            let f = uflags(ch as u32);
            assert!(!is_l(f) && !is_n(f) && !is_ws(f), "other {ch:?}");
        }
    }

    #[test]
    fn piece_end_matches_o200k_contractions_digits_punctuation_and_whitespace() {
        assert_eq!(
            split_pieces("don't won't I'm we're they've you'll he'd IT'S WE'RE rock'n'roll"),
            vec![
                "don't", " won't", " I'm", " we're", " they've", " you'll", " he'd",
                " IT'S", " WE'RE", " rock", "'n", "'roll",
            ]
        );
        // CamelCase / acronym transitions (body1 merges acronym+title `HTTPServer`, while lowercase ` parse` stops before uppercase `JSON`).
        assert_eq!(
            split_pieces("HTTPServer HTML5 parseJSON"),
            vec!["HTTPServer", " HTML", "5", " parse", "JSON"]
        );
        // 1-3 digit chunking across ASCII and Unicode digits.
        assert_eq!(split_pieces("1234567890"), vec!["123", "456", "789", "0"]);
        assert_eq!(split_pieces("١٢٣٤"), vec!["١٢٣", "٤"]);
        // Optional leading space on punctuation + trailing slashes/CRLF consumption.
        assert_eq!(
            split_pieces("a += 1;//\r\nb"),
            vec!["a", " +=", " ", "1", ";//\r\n", "b"]
        );
        // Whitespace lookahead: `\s+(?!\S)` leaves the final space attached to the next word,
        // while trailing whitespace at end-of-string stays one piece and `\s*[\r\n]+` splits after last newline.
        assert_eq!(split_pieces("x    y   "), vec!["x", "   ", " y", "   "]);
        assert_eq!(split_pieces("  \r\n  z"), vec!["  \r\n", " ", " z"]);
        // Multi-script words + combining marks + emoji.
        assert_eq!(
            split_pieces("cafe\u{0301} Москва 日本語1234 🦀!"),
            vec!["cafe\u{0301}", " Москва", " 日本語", "123", "4", " 🦀!"]
        );
    }

    #[test]
    fn production_byte_bpe_round_trips_multibyte_utf8_merges_and_specials() {
        let mut t = Tokenizer::for_byte_test();
        let map_bytes = |tk: &Tokenizer, s: &[u8]| -> String {
            s.iter().map(|&b| tk.id_to_tok[tk.byte_to_id[b as usize] as usize].as_str()).collect()
        };
        let add_byte_merge = |tk: &mut Tokenizer, left: &[u8], right: &[u8], rank: u32| {
            let l = map_bytes(tk, left);
            let r = map_bytes(tk, right);
            let lid = tk.tok_to_id[&l];
            let rid = tk.tok_to_id[&r];
            let merged = format!("{l}{r}");
            let mid = tk.id_to_tok.len() as u32;
            tk.max_bpe_token_bytes = tk.max_bpe_token_bytes.max(merged.chars().count());
            tk.id_to_tok.push(merged.clone());
            tk.tok_type.push(0);
            tk.tok_to_id.insert(merged, mid);
            tk.merges.insert((lid, rid), (rank, mid));
        };
        add_byte_merge(&mut t, b" ", b"t", 0);
        add_byte_merge(&mut t, b" t", b"h", 1);
        add_byte_merge(&mut t, b" th", b"e", 2);
        add_byte_merge(&mut t, b"\xc3", b"\xa9", 3); // UTF-8 'é'
        add_byte_merge(&mut t, b"c", b"a", 4);
        add_byte_merge(&mut t, b"ca", b"f", 5);
        add_byte_merge(&mut t, b"caf", "é".as_bytes(), 6);

        let the_id = t.tok_to_id[&map_bytes(&t, b" the")];
        let cafe_id = t.tok_to_id[&map_bytes(&t, "café".as_bytes())];
        assert_eq!(t.encode("café the", false), vec![cafe_id, the_id]);

        let rich = format!(
            "<|message_user|><|content_text|>café the Москва 日本語 🦀 12345 {}\r\n<|end_message|>",
            "café".repeat(20) // > 32 bytes in a single piece to exercise bpe_ranked heap path
        );
        let ids = t.encode(&rich, true);
        let mut decoded = Vec::new();
        for &id in &ids {
            decoded.extend_from_slice(&t.decode_bytes(id));
        }
        assert_eq!(decoded, rich.as_bytes());
    }

    #[test]
    fn try_load_rejects_missing_or_malformed_tokenizer_metadata() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-scratch")
            .join(format!("inky-tok-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("empty.gguf");
        let mut file = b"GGUF".to_vec();
        file.extend(3u32.to_le_bytes());
        file.extend(0u64.to_le_bytes()); // tensors
        file.extend(0u64.to_le_bytes()); // metadata entries
        file.resize(32, 0);
        std::fs::write(&path, &file).unwrap();
        let path_str = path.to_str().unwrap();

        let mut g = Gguf::open(path_str).unwrap();
        assert!(Tokenizer::try_load(&g).is_err(), "missing tokens array must return Err");

        let valid = Tokenizer::for_byte_test();
        g.kv.insert("tokenizer.ggml.tokens".into(), Value::AS(valid.id_to_tok.clone()));
        g.kv.insert("tokenizer.ggml.token_type".into(), Value::AI(vec![0; valid.id_to_tok.len() - 1]));
        g.kv.insert("tokenizer.ggml.merges".into(), Value::AS(Vec::new()));
        g.kv.insert("tokenizer.ggml.eos_token_id".into(), Value::U(valid.eos as u64));
        assert!(Tokenizer::try_load(&g).is_err(), "mismatched token_type length must return Err");

        g.kv.insert("tokenizer.ggml.token_type".into(), Value::AI(valid.tok_type.clone()));
        g.kv.insert("tokenizer.ggml.merges".into(), Value::AS(vec!["badmerge_without_space".into()]));
        assert!(Tokenizer::try_load(&g).is_err(), "malformed merge entry must return Err");

        g.kv.insert("tokenizer.ggml.merges".into(), Value::AS(Vec::new()));
        g.kv.insert("tokenizer.ggml.eos_token_id".into(), Value::U(999_999));
        assert!(Tokenizer::try_load(&g).is_err(), "out-of-range eos_token_id must return Err");

        g.kv.insert("tokenizer.ggml.eos_token_id".into(), Value::U(valid.eos as u64));
        let loaded = Tokenizer::try_load(&g).expect("complete byte-level tokenizer metadata must load");
        assert!(loaded.decode_bytes(u32::MAX).is_empty(), "out-of-bounds token ID must not panic");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
