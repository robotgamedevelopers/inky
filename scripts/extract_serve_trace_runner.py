#!/usr/bin/env python3
"""Extract a standalone CPU-only runner for `src/serve.rs` + `src/serve/scheduler_trace.rs`.

Just like `scripts/check-memmode-traces.sh` extracts `Mode` from `src/memd.rs` so
trace conformance runs anywhere with `rustc --edition=2021` (including Linux
`ubuntu-latest` / `hetzner1` without `objc2_metal` or Cargo), this script splices
the exact production `run_scheduler`, `choose_lane`, `reusable_prefix`, `try_slot`,
`InFlight`, `Sampler`, `Counters`, and `scheduler_trace::walk` into a standalone
source file with a minimal `std`-only `Value` / `json!` shim.
"""

from __future__ import annotations

import pathlib
import sys


JSON_SHIM = r'''
use std::collections::BTreeMap as JsonMap;
use std::fmt;
use std::io::Write;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Uint(u64),
    Str(String),
    Arr(Vec<Value>),
    Obj(Vec<(String, Value)>),
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        if let Value::Str(s) = self { Some(s.as_str()) } else { None }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Uint(u) => Some(*u),
            Value::Int(i) if *i >= 0 => Some(*i as u64),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&Vec<Value>> {
        if let Value::Arr(a) = self { Some(a) } else { None }
    }
}

impl std::ops::Index<&str> for Value {
    type Output = Value;
    fn index(&self, key: &str) -> &Value {
        if let Value::Obj(entries) = self {
            for (k, v) in entries {
                if k == key { return v; }
            }
        }
        &Value::Null
    }
}

impl From<&str> for Value { fn from(s: &str) -> Self { Value::Str(s.to_string()) } }
impl From<String> for Value { fn from(s: String) -> Self { Value::Str(s) } }
impl From<bool> for Value { fn from(b: bool) -> Self { Value::Bool(b) } }
impl From<u64> for Value { fn from(n: u64) -> Self { Value::Uint(n) } }
impl From<usize> for Value { fn from(n: usize) -> Self { Value::Uint(n as u64) } }
impl From<u32> for Value { fn from(n: u32) -> Self { Value::Uint(n as u64) } }
impl From<i32> for Value { fn from(n: i32) -> Self { Value::Int(n as i64) } }
impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self { Value::Arr(v.into_iter().map(Into::into).collect()) }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int(i) => write!(f, "{i}"),
            Value::Uint(u) => write!(f, "{u}"),
            Value::Str(s) => {
                write!(f, "\"")?;
                for ch in s.chars() {
                    match ch {
                        '"' => write!(f, "\\\"")?,
                        '\\' => write!(f, "\\\\")?,
                        '\n' => write!(f, "\\n")?,
                        c => write!(f, "{c}")?,
                    }
                }
                write!(f, "\"")
            }
            Value::Arr(items) => {
                write!(f, "[")?;
                for (i, it) in items.iter().enumerate() {
                    if i > 0 { write!(f, ",")?; }
                    write!(f, "{it}")?;
                }
                write!(f, "]")
            }
            Value::Obj(entries) => {
                let mut sorted: JsonMap<&str, &Value> = JsonMap::new();
                for (k, v) in entries { sorted.insert(k.as_str(), v); }
                write!(f, "{{")?;
                for (i, (k, v)) in sorted.iter().enumerate() {
                    if i > 0 { write!(f, ",")?; }
                    write!(f, "\"{k}\":{v}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

macro_rules! json {
    ({ $($k:literal : $v:expr),* $(,)? }) => {
        Value::Obj(vec![ $( ($k.to_string(), Value::from($v)) ),* ])
    };
}
'''


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    serve_override = pathlib.Path(sys.argv[2])
    out_rs = pathlib.Path(sys.argv[3])

    serve_src = serve_override.read_text()
    sched_marker = "// ------------------------------------------------------------------------------------------------- scheduler"
    http_marker = "// ------------------------------------------------------------------------------------------------- HTTP"
    sched_start = serve_src.index(sched_marker)
    http_start = serve_src.index(http_marker, sched_start)
    scheduler_section = serve_src[sched_start:http_start]

    # Also extract InFlight + try_slot from serve.rs
    inflight_start = serve_src.index("struct InFlight")
    # Find end of try_slot fn
    try_slot_idx = serve_src.index("fn try_slot", inflight_start)
    brace_depth = 0
    end_idx = try_slot_idx
    started = False
    while end_idx < len(serve_src):
        ch = serve_src[end_idx]
        if ch == "{":
            brace_depth += 1
            started = True
        elif ch == "}":
            brace_depth -= 1
            if started and brace_depth == 0:
                end_idx += 1
                break
        end_idx += 1
    inflight_section = serve_src[inflight_start:end_idx]

    backend_src = (root / "src" / "serve" / "backend.rs").read_text()
    trait_start = backend_src.index("pub(super) trait SchedulerEngine")
    impl_start = backend_src.index("impl SchedulerEngine for Engine", trait_start)
    backend_trait = backend_src[trait_start:impl_start].replace(
        "pub(super) trait SchedulerEngine", "pub trait SchedulerEngine", 1
    )

    trace_src = (root / "src" / "serve" / "scheduler_trace.rs").read_text()
    trace_body = trace_src.replace("use super::*;", "")
    test_idx = trace_body.index("#[test]\nfn seeded_decisions")
    trace_core = trace_body[:test_idx]

    combined = f"""#![allow(dead_code, unused_imports, unused_variables, unused_assignments)]
use std::collections::VecDeque;

#[path = "{(root / 'src' / 'sampler.rs').as_posix()}"]
mod sampler;
use sampler::Sampler;

#[path = "{(root / 'src' / 'serve' / 'timing.rs').as_posix()}"]
mod timing;
use timing::{{Counters, RequestTiming}};

{JSON_SHIM}

mod backend {{
    use super::*;
{backend_trait}
}}
use backend::SchedulerEngine;

{inflight_section}

{scheduler_section}

mod sched_trace {{
{trace_core}
    use super::*;
    pub fn dump_seed(seed: u64) {{
        let out = std::io::stdout();
        let mut w = std::io::BufWriter::new(out.lock());
        for ev in walk(seed) {{
            writeln!(w, "{{ev}}").unwrap();
        }}
        w.flush().unwrap();
    }}
}}

fn main() {{
    let seed: u64 = std::env::args()
        .nth(1)
        .map(|s| if s == "max" {{ u64::MAX }} else {{ s.parse().expect("u64 seed") }})
        .unwrap_or(1);
    sched_trace::dump_seed(seed);
}}
"""
    out_rs.write_text(combined)
    return 0


if __name__ == "__main__":
    sys.exit(main())
