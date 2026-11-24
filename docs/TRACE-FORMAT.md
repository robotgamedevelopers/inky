# Layer 3 & 4 TLA+ Trace Format (`version: 1`)

This document specifies the canonical JSON Lines (`.jsonl`) trace format shared across Layer 3 (per-kind block pools, per-lane block tables, prefix tree, start points, and unsealed tails) and Layer 4 (cache-aware scheduler, admission budget, and cold SSD tier), implemented in [`src/cache_trace.rs`](../src/cache_trace.rs), emitted by [`src/cache_scheduler.rs`](../src/cache_scheduler.rs), and checked in CI by [`scripts/check-cache-traces.sh`](../scripts/check-cache-traces.sh).

Every trace proves that a concrete Rust execution is a valid finite behaviour (`Init /\ [][Next]_vars`) of its TLA+ specification:

- `model: "block_layout"` -> [`specs/BlockLayout.tla`](../specs/BlockLayout.tla) (per-kind physical pools starting at byte offset `0` at element width `kvb` (`2` for fp16 -> `2 MiB` full / `10 MiB` window stride; `1` for `--kv8` -> `1 MiB` full / `5 MiB` window stride), `1`-based slots, `SlotRecord::reissue` non-wrapping generations, pairwise non-overlapping 64-bit slot byte ranges, sliding-window coverage, `SpanBound(c)` step pins, retained start-point handles, and zero-padded private tails).
- `model: "cache_scheduler"` -> [`specs/CacheScheduler.tla`](../specs/CacheScheduler.tla) (sealed prefix-tree blocks vs unsealed private tails, start points, worst-case admission budgets, `MaxQueue` admit-bypass `wait_ticks`, atomic restore working-set reservations, decode-priority step building, in-step cancellation fences, and cold-only LRU SSD writeback).

---

## 1. CI Gate & Trace Dump Entry Points

`scripts/check-tla.sh` runs [`scripts/check-cache-traces.sh`](../scripts/check-cache-traces.sh) on CPU-only runners (`rustc --edition=2021` + pinned `tla2tools.jar`, zero Cargo/Metal/network), replaying multi-seed walks at `slot_base = 0`, `slot_base = 2048` (`kvb = 2`, `>= 4 GiB`), `slot_base = 4096` (`kvb = 1`, `>= 4 GiB`), and `max_queue = 2` through TLC and requiring 7 single-line mutations of `src/blocks.rs` and `src/cache_scheduler.rs` to fail inside TLC with a counterexample.

Mirroring `serve::scheduler_trace::dump` (`INKY_SCHED_TRACE_OUT`), interactive `.jsonl` trace dumps are also available via `cargo test`:

```sh
INKY_BLOCK_TRACE_SEED=1 INKY_BLOCK_TRACE_OUT=/tmp/block-seed-1.jsonl \
  cargo test --locked --bins cache_scheduler::tests::dump_block_layout -- --ignored --exact

INKY_CACHE_TRACE_SEED=1 INKY_CACHE_TRACE_OUT=/tmp/cache-seed-1.jsonl \
  cargo test --locked --bins cache_scheduler::tests::dump_cache_scheduler -- --ignored --exact
```

- `INKY_BLOCK_TRACE_SLOT_BASE` (default `0`; set `2048` at `kvb=2` or `4096` at `kvb=1` to exercise `>= 4 GiB` offsets through `blocks::byte_offset`).
- `INKY_BLOCK_TRACE_KVB` (`2` for fp16, `1` for `--kv8`).
- `INKY_BLOCK_TRACE_MODE` / `INKY_CACHE_TRACE_MODE`: `events` (default compact transition stream) or `steps` (full post-step state snapshots).

---

## 2. Stream Envelope & Fail-Closed Rules

1. **Encoding**: UTF-8 JSON Lines (`.jsonl`), one JSON object per line, `\n` terminated, no floating-point numbers, no duplicate keys.
2. **Line 1 (`config`)**: Must be `"event": "config"` with `"version": 1` and `"model": "block_layout" | "cache_scheduler"`, declaring all TLA+ `CONSTANTS` for the run.
3. **Determinism**: Zero wall-clock timestamps (`Instant::now()`, `SystemTime`), OS thread IDs, pointer addresses, or randomized hash iteration order appear anywhere in a trace.
4. **Plain `u64` Offsets & 32-Bit-Safe TLC Limbs**:
   Every allocated entry calls `crate::blocks::byte_offset(kind, physical_slot, kvb)` directly (`physical_slot = slot_base + slot`), storing both `"offset"` (`u64`) and `"offset_limbs": {"hi": u32, "mib": u32, "sub": u32}` where `offset == (hi << 32) | (mib << 20) | sub` (`mib < 4096`, `sub < 1_048_576`). `parse_block_layout_jsonl` rejects any mismatch between `offset` and `offset_limbs`, non-canonical limbs, invalid `kvb`, or out-of-range slots.
