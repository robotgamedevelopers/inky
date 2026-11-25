// Deterministic random walks of src/blockbook.rs, printed as the TLA+ module
// StartTailsTraceData for specs/StartTailsTrace.tla. Compiled by
// scripts/check-blockbook-traces.sh next to copies of src/blocks.rs and
// src/blockbook.rs; no Cargo, Metal, time, sockets or files.
//
// The fake GPU side writes positions through the lane's table the way a step does,
// zeroes on `zero`, and records the position each kept state belongs to (a
// freed state reads as -1, and `kept` counts live states). Each call
// into the book is one recorded state, so every call must be one model step (or
// change nothing). A case the model's witnesses are about that the walks never
// reached is reported as a COVERAGE GAP, which fails the unmutated run.
#[path = "blocks.rs"]
#[allow(dead_code)]
mod blocks;
#[path = "blockbook.rs"]
#[allow(dead_code)]
mod blockbook;

use blockbook::{BlockBook, GpuSide, Geometry, StateId, EVERY, KINDS, PROMPT};
use blocks::{BlockIndex, BlockKind, Slot};
use std::collections::BTreeMap;

const SEEDS: u64 = 20;
const OPS: usize = 90;
const LANES: usize = 3;
const SLOTS: usize = 5;
const STARTS: usize = 3;
const GEO: Geometry = Geometry { block: 2, window: 4, every: 2, blocks: 6 };

struct Fake {
    data: [Vec<Vec<u8>>; 2],
    next: StateId,
    states: BTreeMap<StateId, u32>,
    at: u32,
}

fn k(kind: BlockKind) -> usize { if kind == BlockKind::Full { 0 } else { 1 } }

impl GpuSide for Fake {
    fn zero(&mut self, kind: BlockKind, slot: Slot) {
        self.data[k(kind)][slot.get() as usize - 1].iter_mut().for_each(|d| *d = 0);
    }
    fn save_state(&mut self, _lane: usize) -> StateId {
        self.next += 1;
        self.states.insert(self.next, self.at);
        self.next
    }
    fn free_state(&mut self, state: StateId) {
        if self.states.remove(&state).is_none() { panic!("state {state} freed twice"); }
    }
}

struct Rng(u64);
impl Rng {
    fn draw(&mut self, n: u64) -> u64 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.0 = x;
        x % n
    }
}

// Compact tuples, decoded by StartTailsTrace.tla: a kind is 1 (full) or 2 (window).
fn table_set(t: &blocks::BlockTable) -> String {
    let es: Vec<String> = t.entries()
        .map(|(b, e)| format!("<<{}, {}, {}, {}>>", k(e.kind()) + 1, b.get(), e.slot().get(), e.generation()))
        .collect();
    format!("{{{}}}", es.join(","))
}

// Everything but the reads: <<full pool, window pool, tables, pos, prompt, running,
// head, start points, kept states>>. A slot is <<generation, length, sealed, refs,
// data...>>; a start point is <<live, pos, state, lane, reason bits, table>>.
fn snapshot(book: &BlockBook, gpu: &Fake) -> String {
    let pool: Vec<String> = KINDS.iter().map(|&kind| {
        let recs: Vec<String> = (1..=SLOTS as u32).map(|s| {
            let r = book.pools().of(kind).record(Slot::new(s).unwrap()).unwrap();
            let d: Vec<String> = gpu.data[k(kind)][s as usize - 1].iter().map(|v| v.to_string()).collect();
            format!("<<{},{},{},{},{}>>", r.generation(), r.length(), u8::from(r.sealed()), r.refs(), d.join(","))
        }).collect();
        format!("<<{}>>", recs.join(","))
    }).collect();
    let lanes = |f: &dyn Fn(usize) -> String| (0..LANES).map(f).collect::<Vec<_>>().join(",");
    let starts: Vec<String> = (0..STARTS).map(|i| match book.start(i) {
        None => "<<0,0,0,-1,0,{}>>".to_string(),
        Some(p) => format!("<<1,{},{},{},{},{}>>", p.pos(),
                           gpu.states.get(&p.state()).map_or(-1, |&n| i64::from(n)),
                           p.lane().map_or(-1, |l| l as i64), p.reasons(), table_set(p.table())),
    }).collect();
    format!("{},<<{}>>,<<{}>>,<<{}>>,<<{}>>,<<{}>>,<<{}>>,{}",
            pool.join(","),
            lanes(&|l| table_set(book.table(l).unwrap())),
            lanes(&|l| book.pos(l).to_string()),
            lanes(&|l| book.prompt(l).to_string()),
            lanes(&|l| u8::from(book.running(l)).to_string()),
            lanes(&|l| book.head(l).map_or(0, |h| h + 1).to_string()),
            starts.join(","), gpu.states.len())
}

fn with_reads(state: &str, reads: &[(usize, BlockKind, u32, u32)]) -> String {
    let rd: Vec<String> = reads.iter().map(|(l, kind, b, n)| format!("<<{l},{},{b},{n}>>", k(*kind) + 1)).collect();
    format!("<<{state},{{{}}}>>", rd.join(","))
}

fn live_starts(book: &BlockBook) -> Vec<usize> { (0..STARTS).filter(|&i| book.start(i).is_some()).collect() }

fn main() {
    // resumed, reused slot, window pruned under a live point, partial tail read,
    // retired, finished with a tail, prompt point, every point, pool full
    let mode = std::env::args().nth(1).unwrap_or_default();
    let is_mutant = !mode.is_empty() && mode != "prod";
    let mut seen = [0u64; 9];
    let mut traces = Vec::new();
    for seed in 1..=SEEDS {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut book = BlockBook::new(GEO, SLOTS, LANES, STARTS).unwrap();
        let mut gpu = Fake { data: [vec![vec![0; GEO.block as usize]; SLOTS], vec![vec![0; GEO.block as usize]; SLOTS]],
                             next: 0, states: BTreeMap::new(), at: 0 };
        let mut last = snapshot(&book, &gpu);
        let mut states = vec![with_reads(&last, &[])];
        for _ in 0..OPS {
            let lane = rng.draw(LANES as u64) as usize;
            let op = rng.draw(10);
            let mut calls: Vec<Box<dyn FnOnce(&mut BlockBook, &mut Fake, &mut [u64; 9])>> = Vec::new();
            if !book.running(lane) {
                if op < 7 {
                    let prompt = 1 + rng.draw(u64::from(GEO.max_pos())) as u32;
                    let cands: Vec<usize> = live_starts(&book).into_iter().filter(|&i| book.start(i).unwrap().pos() < prompt).collect();
                    let from = if !cands.is_empty() && rng.draw(3) > 0 { Some(cands[rng.draw(cands.len() as u64) as usize]) } else { None };
                    calls.push(Box::new(move |b, _, s| { if b.begin(lane, prompt, from).is_ok() && from.is_some() { s[0] += 1; } }));
                } else if let Some(&i) = live_starts(&book).first() {
                    calls.push(Box::new(move |b, g, s| { if b.retire(i, g).is_ok() { s[4] += 1; } }));
                }
            } else if op < 6 {
                let pos = book.pos(lane);
                let room = GEO.max_pos() - pos;
                if room > 0 {
                    let c = 1 + rng.draw(u64::from(room.min(5))) as u32;
                    for block in pos / GEO.block..=(pos + c - 1) / GEO.block {
                        if book.table(lane).unwrap().get(BlockKind::Full, BlockIndex::new(block)).unwrap().is_none() {
                            calls.push(Box::new(move |b, g, s| alloc(b, g, s, lane, block)));
                        }
                    }
                    calls.push(Box::new(move |b, g, s| step(b, g, s, lane, c)));
                }
            } else if op < 7 {
                let block = book.pos(lane) / GEO.block + rng.draw(3) as u32;
                calls.push(Box::new(move |b, g, s| alloc(b, g, s, lane, block)));
            } else if op < 8 {
                let c = 1 + rng.draw(3) as u32;
                calls.push(Box::new(move |b, g, s| step(b, g, s, lane, c)));
            } else if op < 9 {
                calls.push(Box::new(move |b, g, s| {
                    let tail = b.pos(lane) % GEO.block != 0;
                    if b.finish(lane, g).is_ok() && tail { s[5] += 1; }
                }));
            } else if let Some(&i) = live_starts(&book).last() {
                calls.push(Box::new(move |b, g, s| { if b.retire(i, g).is_ok() { s[4] += 1; } }));
            }
            for call in calls {
                call(&mut book, &mut gpu, &mut seen);
                // A call that changed nothing (a refusal) is not recorded; one that
                // changed anything is, refusal or not.
                let now = snapshot(&book, &gpu);
                if now == last { continue; }
                let mut reads = Vec::new();
                for l in 0..LANES {
                    if !book.running(l) { continue; }
                    for _ in 0..2 {
                        let kind = KINDS[rng.draw(2) as usize];
                        let block = rng.draw(u64::from(GEO.blocks)) as u32;
                        if let Some(n) = book.visible(l, kind, block).unwrap() {
                            if n > 0 && n < GEO.block { seen[3] += 1; }
                            reads.push((l, kind, block, n));
                        }
                    }
                }
                reads.sort_by_key(|r| (r.0, k(r.1), r.2));
                reads.dedup();
                states.push(with_reads(&now, &reads));
                last = now;
            }
        }
        traces.push(states);
        if is_mutant {
            let hit = match mode.as_str() {
                "unheld-window" | "pruned-under-start" => seen[2] > 0 && seen[0] > 0,
                "overread" => seen[3] > 0,
                "no-zero" => seen[1] > 0,
                "kept-tail" => seen[5] > 0,
                "every-off" => seen[7] > 0,
                _ => seen.iter().all(|&c| c > 0),
            };
            if hit {
                break;
            }
        }
    }
    println!("---- MODULE StartTailsTraceData ----\nEXTENDS Integers\nTraces == <<");
    let body: Vec<String> = traces.iter().map(|t| format!("<<\n{}\n>>", t.join(",\n"))).collect();
    println!("{}\n>>\n====", body.join(",\n"));
    let names = ["resumed from a start point", "reused slot", "window pruned under a live point",
                 "partial tail read", "retired", "finished with a tail", "prompt point", "every point", "pool full"];
    let summary: Vec<String> = names.iter().zip(seen).map(|(n, c)| format!("{n}: {c}")).collect();
    eprintln!("{SEEDS} walks x {OPS} operations; {}", summary.join(", "));
    // The script fails the healthy run on a gap; a mutated file still goes to TLC.
    for (i, _) in seen.iter().enumerate().filter(|(_, &c)| c == 0) {
        eprintln!("COVERAGE GAP: the walks never reached: {}", names[i]);
    }
}

fn alloc(b: &mut BlockBook, g: &mut Fake, s: &mut [u64; 9], lane: usize, block: u32) {
    match b.alloc(lane, block, g) {
        Ok(()) => {
            let e = b.table(lane).unwrap().get(BlockKind::Full, BlockIndex::new(block)).unwrap().unwrap();
            if e.generation() > 1 { s[1] += 1; }
        }
        Err(blockbook::BookError::PoolFull) => s[8] += 1,
        Err(_) => {}
    }
}

fn step(b: &mut BlockBook, g: &mut Fake, s: &mut [u64; 9], lane: usize, c: u32) {
    let pos = b.pos(lane);
    let mut writes = Vec::new();
    for p in pos..pos + c {
        for kind in KINDS {
            if let Ok(Some(e)) = b.table(lane).unwrap().get(kind, BlockIndex::new(p / GEO.block)) {
                writes.push((kind, e.slot().get(), p % GEO.block));
            }
        }
    }
    let before: Vec<_> = b.table(lane).unwrap().entries().filter(|(_, e)| e.kind() == BlockKind::Window).collect();
    g.at = pos + c;
    if let Ok(kept) = b.step(lane, c, g) {
        for (kind, slot, i) in writes {
            g.data[k(kind)][slot as usize - 1][i as usize] = 1;
        }
        for (blk, e) in before {
            let gone = b.table(lane).unwrap().get(BlockKind::Window, blk).unwrap().is_none();
            let held = live_starts(b).iter().any(|&i| b.start(i).unwrap().table().get(BlockKind::Window, blk).unwrap() == Some(e));
            if gone && held { s[2] += 1; }
        }
        if let Some(id) = kept {
            let r = b.start(id).unwrap().reasons();
            if r & PROMPT != 0 { s[6] += 1; }
            if r & EVERY != 0 { s[7] += 1; }
        }
    }
}
