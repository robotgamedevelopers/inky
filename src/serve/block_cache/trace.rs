//! CPU-only trace projection of the actual serving cache. Check claims about
//! these observations in specs/ServeCacheTrace.tla.

use super::*;
use std::fmt::Write as _;

const TABLE_BLOCKS: u32 = 36;

fn tuple(items: impl IntoIterator<Item = impl ToString>) -> String {
    let items: Vec<_> = items.into_iter().map(|item| item.to_string()).collect();
    format!("<<{}>>", items.join(", "))
}

fn set(items: impl IntoIterator<Item = String>) -> String {
    let mut items: Vec<_> = items.into_iter().collect();
    items.sort_unstable();
    format!("{{{}}}", items.join(", "))
}

fn id_number(id: SealedId) -> u64 {
    id.index() + 1
}

fn kind_number(kind: BlockKind) -> u8 {
    match kind {
        BlockKind::Full => 1,
        BlockKind::Window => 2,
    }
}

fn handle(entry: BlockEntry) -> String {
    tuple([entry.slot().get(), entry.generation()])
}

fn entry_row(index: BlockIndex, entry: BlockEntry) -> String {
    tuple([
        kind_number(entry.kind()).to_string(),
        index.get().to_string(),
        handle(entry),
    ])
}

#[derive(Default)]
struct Lane {
    state: LaneBlockState,
    prompt: Vec<u32>,
}

struct Trace {
    cache: ServeBlockCache,
    pools: BlockPools,
    tables: [BlockTable; 2],
    lanes: [Lane; 2],
    rows: Vec<String>,
}

impl Trace {
    fn new() -> Self {
        Self {
            cache: ServeBlockCache::new(),
            pools: BlockPools::with_capacities(44, 40, 2).unwrap(),
            tables: std::array::from_fn(|_| BlockTable::new(TABLE_BLOCKS).unwrap()),
            lanes: std::array::from_fn(|_| Lane::default()),
            rows: Vec::new(),
        }
    }

    fn row(&mut self, op: &str, lane: usize, arg: u64, cs: usize) {
        let pools = tuple([BlockKind::Full, BlockKind::Window].map(|kind| {
            let pool = self.pools.of(kind);
            tuple((1..=pool.len()).map(|slot| {
                let rec = pool.record(Slot::new(slot as u32).unwrap()).unwrap();
                tuple([
                    rec.generation(),
                    rec.length(),
                    rec.refs(),
                    if rec.sealed() { 1 } else { 0 },
                ])
            }))
        }));
        let lanes = tuple(self.lanes.iter().zip(&self.tables).map(|(lane, table)| {
            tuple([
                lane.state.written.to_string(),
                lane.state.prompt_len.to_string(),
                tuple(
                    lane.prompt
                        .chunks_exact(BLOCK)
                        .map(|block| block[BLOCK - 1]),
                ),
                tuple(lane.state.sealed.iter().copied().map(id_number)),
                set(table
                    .entries()
                    .map(|(index, entry)| entry_row(index, entry))),
            ])
        }));
        let tree = set(self.cache.tree.entries().into_iter().map(|(id, handles)| {
            let (parent, tokens) = self.cache.tree.trace_block(id).unwrap();
            tuple([
                id_number(id).to_string(),
                parent.map_or(0, id_number).to_string(),
                tokens[BLOCK - 1].to_string(),
                self.cache.tree.pin_count(id).to_string(),
                self.cache.tree.start_holders(id).to_string(),
                handle(handles.full()),
                handles.window().map_or("<<0, 0>>".to_string(), handle),
            ])
        }));
        let starts = set(self.cache.starts.iter().map(|(&id, start)| {
            tuple([
                id_number(id).to_string(),
                start.pos.to_string(),
                start.reasons.to_string(),
                decode_cs(&start.cs).to_string(),
                set(start
                    .entries
                    .iter()
                    .map(|&(_, index, entry)| entry_row(index, entry))),
            ])
        }));
        self.rows.push(format!(
            "[op |-> \"{op}\", lane |-> {}, arg |-> {arg}, cs |-> {cs}, pools |-> {pools}, lanes |-> {lanes}, tree |-> {tree}, starts |-> {starts}]",
            lane + 1,
        ));
    }

    fn bind(&mut self, i: usize, tags: &[u32], tail: usize) {
        let lane = &mut self.lanes[i];
        // Every block shares its first 63 tokens. A first-token-only walk
        // would match the wrong history at the very last token of a block.
        lane.prompt = tags
            .iter()
            .flat_map(|&tag| (0..63).chain(std::iter::once(tag)))
            .chain(std::iter::repeat_n(999, tail))
            .collect();
        let hit = self.cache.match_and_bind(
            &mut self.pools,
            &mut self.tables[i],
            &mut lane.state,
            &lane.prompt,
        );
        let (pos, cs) = hit.map_or((0, 0), |(pos, cs)| (pos, decode_cs(&cs)));
        self.row("bind", i, pos as u64, cs);
    }

    fn prefill(&mut self, i: usize, end: usize) {
        let lane = &mut self.lanes[i];
        let start = lane.state.written;
        self.cache
            .ensure_span_blocks(&mut self.pools, &mut self.tables[i], start, end - start)
            .unwrap();
        self.cache.finish_prefill_span(
            &mut self.pools,
            &mut self.tables[i],
            &mut lane.state,
            &lane.prompt,
            start,
            end - start,
            || (end as u32).to_le_bytes().to_vec(),
        );
        self.row("prefill", i, end as u64, 0);
    }

    fn decode(&mut self, i: usize) {
        let lane = &mut self.lanes[i];
        let pos = lane.state.written;
        self.cache
            .ensure_span_blocks(&mut self.pools, &mut self.tables[i], pos, 1)
            .unwrap();
        self.cache
            .finish_decode_span(&mut self.pools, &mut self.tables[i], &mut lane.state, pos);
        self.row("decode", i, (pos + 1) as u64, 0);
    }

    fn release(&mut self, i: usize) {
        let lane = &mut self.lanes[i];
        self.cache
            .release_lane(&mut self.pools, &mut self.tables[i], &mut lane.state);
        lane.prompt.clear();
        self.row("release", i, 0, 0);
    }

    fn evict(&mut self, id: SealedId) {
        assert!(self.cache.evict_start(&mut self.pools, id));
        self.row("evict", 0, id_number(id), 0);
    }
}

fn decode_cs(cs: &[u8]) -> usize {
    u32::from_le_bytes(cs.try_into().unwrap()) as usize
}

pub(crate) fn generate(path: &std::path::Path) {
    let long: Vec<u32> = (100..133).collect();
    let fork = [long[0], 700];
    let mut t = Trace::new();
    t.row("init", 0, 0, 0);

    t.bind(0, &long, 62);
    t.prefill(0, 2048);
    t.prefill(0, 2112);
    t.prefill(0, 2174);
    t.decode(0); // With one new token, the oldest block crosses the window boundary.
    let every = t.lanes[0].state.sealed[31];
    let deep = t.lanes[0].state.sealed[32];
    t.release(0);

    let mut wrong_history = long.clone();
    wrong_history[1] = fork[1];
    t.bind(1, &wrong_history, 62); // A same-length prompt with the wrong history must miss.
    t.release(1);

    // Serving a divergent prompt must replace an expired early window handle
    // while sharing the still-live full handle of the common first block.
    t.bind(1, &fork, 1);
    t.prefill(1, 128);
    let fork_tip = t.lanes[1].state.sealed[1];
    t.release(1);
    t.bind(0, &fork, 1);
    t.evict(fork_tip); // The live lane still pins this branch.
    t.release(0);
    t.bind(0, &fork, 1); // The evicted start must miss.
    t.release(0);

    t.bind(0, &long, 62);
    t.bind(1, &long, 0); // Exact checkpoint must leave the final prompt token unwritten.
    t.release(1);
    t.evict(deep);
    t.release(0); // Removes the deep tip, leaving the EVERY prefix.
    t.bind(1, &long, 62);
    t.release(1);
    t.evict(every); // Final eviction prunes the surviving ancestor chain.

    let mut out = String::from("---- MODULE ServeCacheTraceData ----\nTrace == <<\n");
    writeln!(out, "{}\n>>\n====", t.rows.join(",\n")).unwrap();
    std::fs::write(path, out).unwrap();
}
