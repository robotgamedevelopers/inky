# TLA+ specifications, witnesses, and trace bindings

Read against [docs/DESIGN.md](../docs/DESIGN.md) and [CACHE_SCHEDULER.md](CACHE_SCHEDULER.md).
These are bounded safety, action-property, and fairness/liveness models paired with
reversed witnesses and deterministic Rust trace conformance checks—not an unbounded
proof of GPU arithmetic or wall-clock timing.

Run on a designated Linux or Apple Silicon M5 bench, never a laptop.
Every check, `.cfg` file, trace script, and mutation is verified by
`scripts/check-tla-tiers.sh` (and `scripts/check-tla-tiers.sh --self-test`) to run in
**exactly one** CI tier (`143` checks and mutants: `107` per pull request, `36` nightly):

- **Pull-request tier (`.github/workflows/ci.yml`, `tla` matrix)**:
  - `scripts/check-tla.sh memmode` — `specs/memmode/MemMode.tla` (`1` positive + `9` reversed witnesses, including temporal liveness `UnfairRestart`), plus `scripts/check-tla-tiers.sh` and `--self-test`.
  - `scripts/check-tla.sh blocks` — `specs/run_tlc.zsh` non-grouped PR checks (`66` runs across `PrefixTree`, `PrefixTreeTrace`, `BlockLayout`, `SlabWord`, `BlockedSlab`, `VecAlloc`, `VerifyRollback`, `KvDiskTier`, `SwaRing`, `SpanPins`, `StartTails`, `InBlock`, `InBlockTrace`).
  - `scripts/check-tla.sh grouped` — `specs/run_tlc.zsh` grouped PR checks (`19` runs across `GroupedLayout`, `PoisonGate`, `ScratchTail`).
  - `scripts/check-tla.sh traces` — all seven CPU-only trace-conformance scripts in parallel (`check-memmode-traces.sh`, `check-blockbook-traces.sh`, `check-cache-traces.sh`, `check-scheduler-traces.sh`, `check-poisongate-traces.sh`, `check-serve-step-traces.sh`, `check-serve-cache-traces.sh`), each pinning every single-line Rust mutation to the exact TLA+ invariant or action property it violates.
- **Nightly tier (`.github/workflows/nightly.yml`, `TLA_NIGHTLY=1`)**:
  - `scripts/check-tla.sh layout-nightly` — exhaustive `PrefixTreeNightly` (`2` slots, up to `3` sealed nodes), simulated 21-node `PrefixTree` (`256` traces x `128` steps), three `src/prefix.rs` trace mutants (`no-evict`, `full-match`, `ref-leak`), and `StartTails3` (`Capacity = 3` per kind).
  - `scripts/check-tla.sh scheduler` (`specs/run_cache_scheduler.sh`) — `30` exhaustive `CacheScheduler` runs (`11` positive safety + liveness + action-property configs and `19` reversed witnesses; `-Xmx8g` on Linux).

With Java 21 and the pinned `v1.8.0` `tla2tools.jar`
(`fdffef247f563ef6edbfd44958d2a82db79e7564e07cb673262d896353fa6713`):

```sh
TLA_JAR_CACHE=~/.cache/inky-tla sh scripts/check-tla.sh
TLC_OUTPUT=/var/tmp/block-layout-results bash specs/run_tlc.zsh
TLA_NIGHTLY=1 TLC_WORKERS=8 TLA_JAR_CACHE=~/.cache/inky-tla sh scripts/check-tla.sh scheduler
```

Results and timings are recorded in [RESULTS.md](RESULTS.md) and [CACHE_SCHEDULER.md](CACHE_SCHEDULER.md).

## Wired-memory helper (`MemMode`)

`specs/memmode/MemMode.tla` models `src/memd.rs`: the helper daemon that raises
`iogpu.wired_limit_mb` while any `inky` client is connected, persists the owner's
previous limit on disk before raising, rolls back refused raises, and restores the
saved limit when the last connection closes or after a daemon restart.

- **Checked properties**: `TypeOK`, `RaisedWhileConnected`, `RestoredWhenIdle`,
  `SavedIsOwner`, `SavedIsOwed`, `RefusedRollbackSafe`, `RollbackAttempted`, and
  liveness `EventuallyRestored` (under weak fairness of `Restart` and `RetryRestore`).
- **Reversed witnesses (`9`)**: `RestoreDefault` (`RestoredWhenIdle`), `ReleaseEarly`
  (`RaisedWhileConnected`), `NoPersist` (`RestoredWhenIdle`), `LostSeed` (`SavedIsOwner`),
  `StaleSaved` (`RestoredWhenIdle`), `StaleSavedOwed` (`SavedIsOwed`), `NoRollback`
  (`RefusedRollbackSafe`), `NoRollbackAttempted` (`RollbackAttempted`), and temporal
  `UnfairRestart` (`SpecUnfair`, `EventuallyRestored`).
- **Trace binding**: `scripts/check-memmode-traces.sh` replays 32 deterministic random
  walks of 192 operations over `src/memd.rs` through `specs/memmode/TraceCheck.tla`
  (`InitConforms` + `StepConforms`) and pins 5 mutations (`early-release`, `no-rollback`,
  `no-readback`, `lost-seed`, `skip-keep`) to their exact TLC violation strings.

## Prefix tree (`PrefixTree` and `PrefixTreeTrace`)

`specs/PrefixTree.tla` models `src/prefix.rs` (`PrefixTree` and `LiveTree`): matching
a request to the deepest live retained start point in a tree of sealed blocks. Blocks
carry token sequences (`BlockLen = 2` over `Alphabet`), Full slot/generation (`slotOf`,
`genOf`), optional Window slot/generation (`winSlotOf`, `winGenOf`), current pool
generations (`poolGen`), pin counts (`refs`, where `refs = 0` is cached/evictable and
`refs = 1` holds a pool reference), and start-point holder counts (`starts`). `Seek`
evicts every stale block (and its subtree) before matching and steps back one block on
an exact full-request match so at least one token remains to prefill (`ExpectedResult`).

- **Positive configs**: `PrefixTreeFast` and `PrefixTreeEvictFast` (`Bug = "evict"`,
  lifting the `Reuse` guard to exercise stale eviction under `NoStaleAfterSeek`) run per
  pull request; `PrefixTreeNightly` (`2` slots, up to `3` sealed nodes) and simulated 21-node
  `PrefixTree` (`256` traces x `128` steps; exhaustive under `INKY_TLC_NIGHTLY=1`) run nightly
  in `layout-nightly`.
- **Reversed witnesses (`8`)**: `PrefixTreeShallow` (`DeepestStart`), `PrefixTreeStale`
  (`ResultLive`), `PrefixTreeFirstToken` (`MatchExact`), `PrefixTreeAnyParent`
  (`ResultIsStart`), `PrefixTreePartial` (`StartOnBoundary`), `PrefixTreeLeak`
  (`RefsSettledAfterSeek`), `PrefixTreeReuseHeld` (`HeldNeverStale`), and
  `PrefixTreeStaleInsert` (`NoBadInsert`).
- **Trace binding**: `PrefixTreeTrace.tla` replays `prefix::tests::prefix_trace_dump`
  (`specs/PrefixTreeTraceData.tla`) step-by-step against `StepOp(row)` and `PoolBooks`
  (slot reference counts equal live pinned Full + Window blocks), and `layout-nightly`
  checks three `src/prefix.rs` mutations (`no-evict`, `full-match`, `ref-leak`).

## Block layout, span pins, and GPU slab addressing

`BlockLayout` models two lanes with **one pool per kind** (`FullCapacity >= 2` and
`WindowCapacity >= 2` on every configuration so both pools allocate multiple blocks and
the sliding window genuinely slides), blocks of `B = 2` positions, window `W = 2`,
spans up to `MaxSpan` (`1` in `BlockLayout3`, `2` in `BlockLayout4`, `BlockLayoutHigh`,
and `BlockLayoutKv8`), `MaxPos = 4`, element width `Kvb \in {1, 2}`, and `2` generations
per slot. Plain `u64` byte offsets `(SlotBase + (slot - 1)) * stride_bytes(kind, Kvb)`
are represented losslessly in 32-bit-safe limbs `[hi, mib, sub]`; `BlockLayoutHigh`
(`SlotBase = 2048, Kvb = 2`) and `BlockLayoutKv8` (`SlotBase = 4096, Kvb = 1`) cross
`4 GiB` in both pools.

`SpanPinned` checks each active span against an independent per-query-token ground truth
(`QueryWinStart(q)..q` for every query position `q` in the span, plus `<= budget[l]`
pinned window handles), rather than comparing `SpanWindow` with itself.

| Configuration | Expected failure | Counterexample |
| --- | --- | --- |
| `StaleEntry` | `GenerationMatches` | Finish leaves a table entry behind; another lane reuses its slot at generation 2 while the entry still holds generation 1. |
| `WindowOffByOne` | `WindowCovered` | Sliding drops an active window block still covered by `History(pos)`. |
| `FullDropped` | `FullCovered` | Sliding drops a full-attention block `< pos`. |
| `SpanTooSmall` | `SpanPinned` | At `FullCapacity = 3, WindowCapacity = 3, MaxPos = 5, MaxSpan = 3`, a 3-token span pins 3 window blocks while reserving `budget = 2`. |
| `StartReleased` | `StartRetained` | Finish releases a kept start point's window slot so another lane reuses it at generation 2. |
| `TailNotZeroed` | `TailPrivateZero` | Reusing a previously written slot without zeroing leaves non-zero tail elements past `length = 0`. |
| `WrongTableIndex` | `TableIndex` | Allocation records logical index `b + 1` in the pool cell for table entry `b`. |
| `Truncate32` | `Address64` | Dropping the upper 32-bit limb (`hi = 0`) at `SlotBase = 2048` aliases a `>= 4 GiB` slot onto low memory. |
| `SlotOverlap` | `Address64` | Collapsing slot stride (`slot - 1 -> 0`) aliases distinct live slots onto byte offset 0. |

### `SpanPins` (`B = 64, W = 1024, C = 512`)

`SpanPins` checks production span-pinning dimensions across all `64` block alignments:
every query position's sliding-window history is covered (`Covered`), the conservative
reservation `Bound = 25` is never exceeded (`BoundHolds`), and worst-case alignment `63`
requires `25` window blocks (`MaximumIs25`). Witness configs `SpanPins17` (`Covered`),
`SpanPinsOverBound` (`BoundHolds`), and `SpanPinsMax24` (`MaximumIs25`) each fail on
their named invariant. `src/blocks.rs` (`span_window_blocks` and
`BlockTable::pinned_span_entries`) is bound to `SpanPins.tla` by `sp_512` and pinned
mutant `cap17-span-pins` in `scripts/check-cache-traces.sh`.

### Supporting engine and KV layout models (`blocks` PR job)

- **`InBlock` and `InBlockTrace`** (`specs/InBlock.tla`, `specs/InBlockTrace.tla`,
  `src/blocks.rs`, `src/kernels.metal`, `src/prefill.metal`): checks the in-slot byte
  order `[pass][K|V][head][position][HD]` (`Ordered`, `Inside`, `NoCollision`) at
  `kvb = 2` (`InBlock`) and `kvb = 1` (`InBlockKv8`), with witnesses `InBlockMajor`
  (`Ordered`), `InBlockMajorCollide` (`NoCollision`), and `InBlockAddrCol`
  (`NoCollision`). `inky inblock-trace` generates `1,488` coordinates across `Full` and
  `Window` blocks at `fp16` and `KV8`, verified by `InBlockTrace` and `InBlockTraceKv8`.
- **`SlabWord`** (`specs/SlabWord.tla`, `src/blocks.rs`): models the 64-bit packed GPU
  block-table word `(slab << 62) | byte_offset` across up to 4 slabs (`KV0..KV3`), with
  `SlabWord` (`kvb = 2`), `SlabWordKv8` (`kvb = 1`), and witnesses `SlabWordUnmasked`
  (`ShaderPtrResolves`), `SlabWordWrongBind` (`ShaderPtrResolves`), `SlabWordOverflow`
  (`NoOffsetClobbersSlab`), and `SlabWordFifthSlab` (`SlabCountBounded`).
- **`BlockedSlab`** (`specs/BlockedSlab.tla`, `src/engine.rs`): models slot allocation,
  free-list reuse, `(slot * 2 + swa) * stride` non-aliasing, stride growth/re-layout,
  and incremental `[dirty_lo..=dirty_hi]` GPU table upload (`BlockedSlab3`,
  `BlockedSlab4`, plus witnesses `BlockedSlabFlatGrow`, `BlockedSlabNoReuseZero`,
  `BlockedSlabMissedDirty`).
- **`VecAlloc`** (`specs/VecAlloc.tla`, `src/engine.rs`): models pinned activation banks
  and LoopCD dry-weak pass vector liveness (`VecAlloc10`, `VecAlloc7`, plus witnesses
  `VecAlloc6` -> `NoPoolExhaustion` and `VecAllocUnpinnedWeak` -> `NeverOverwritesLive`).
- **`VerifyRollback`** (`specs/VerifyRollback.tla`, `src/engine.rs`): models speculative
  `encode_verify` snapshot and `rollback_verify` `n_keep` convolution-state and KV
  restoration without sealing rejected tails (`VerifyRollback`, plus witnesses
  `VerifyOffByOneCs`, `VerifyKeepRejectedKv`, `VerifySealRejected`).
- **`KvDiskTier` and `SwaRing`** (`specs/KvDiskTier.tla`, `specs/SwaRing.tla`,
  `src/engine.rs`): model `--kv-disk` hot-ring split alignment and double-buffered
  staging (`KvDiskTier`, `KvDiskNoOverlap`, `KvDiskStagingRace`) and sliding-window ring
  writeback (`SwaRing`, `SwaRingShortWriteback`).

## Grouped tile traversal (`GroupedLayout`)

`GroupedLayout` checks the layout part of [the shared-prefix model gate](../docs/SHARED-PREFIX.md#gates-in-landing-order)
for a single attention pass over two pinned lane tables (`GroupedLayout3`,
`GroupedLayout4`). A shared block occupies one slot across both rows; after divergence,
each row needs a separate slot for each block, including its tail. Under weak fairness
of the enabled traversal action, each snapshot finishes (`TraversalCompletes`).

| Witness / Control | Checked property | What changes |
| --- | --- | --- |
| `LocalMatch` | `VisitedKeys` | Two complete blocks share local tokens after their parents diverged; row 2 reads row 1's slot. |
| `DonorWindow` | `VisitedKeys` | Row 2 applies row 1's window start and visits the wrong keys. |
| `TilePartial` | `OrderedPartials` | Emitting at the first tile end produces an extra partial inside a split. |
| `Restage` | `StagedOnce` | A physical slot is loaded twice while visited keys and partials stay unchanged. |
| `StalledTraversal` | `TraversalCompletes` | Both rows finish their loads, yet traversal stays on tile zero forever. |
| `AdvanceUnfair` | `TraversalCompletes` | Weak fairness covers staging only; `Advance` stays enabled and stutters forever. |
| `RestageOutputs` | PASS (negative control) | Runs the redundant-load mutant against `VisitedKeys` and `OrderedPartials` only, proving only `StagedOnce` detects it. |

## Start points and unsealed tails (`StartTails`)

`StartTails` checks `src/blockbook.rs`: which blocks a start point holds (`Retained(n)`),
when start points are kept and dropped (`head`, `prompt`, and `every`), what attention
may read of an unsealed tail (`Visible(n, b)`), and zeroing on reuse. Decode steps past
`Floor(prompt)` never keep a start point (Decision 4).

Bounds (`StartTails2` in PR CI, `StartTails3` nightly): two symmetric lanes
(`Lanes = {a, b}`, `SYMMETRY SymLanes`), **two slots per kind** in CI and **three per
kind** nightly, `B = 2, W = 2, Every = 2`, **`MaxPos = 6`**, **`MaxStarts = 2`**, and
`MaxGeneration = 2`. With `MaxPos = 6` and `MaxStarts = 2`, two start points are live
simultaneously and an `"every"` point (`pos = 4` on `prompt = 6`, reasons
`{"head", "every"}`) is distinct from a `"prompt"` point (`pos = 6`, reasons
`{"head", "prompt"}`).

| Configuration | Expected failure | Counterexample |
| --- | --- | --- |
| `LaneStaleGen` | `LaneCovered` | `Finish` leaves a lane's table populated while `running = FALSE`. |
| `UnheldWindow` | `StartRetained` | Start point holds only its full blocks; another lane reuses its window slot at generation 2. |
| `PrunedUnderStart` | `StartRetained` | Window pruning releases a window slot outright underneath a live start point. |
| `TailOverread` | `ReadsWritten` | Attention reads one position past `Seen(n, b)` on an unsealed block. |
| `ReuseNotZeroed` | `TailPrivateZero` | Reusing a dropped tail slot at generation 2 leaves stale data past `length = 0`. |
| `DanglingHead` | `HeadsConsistent` | Retiring a start point leaves a running lane's `head[l]` pointing to a dead start id. |

**Trace binding**: `scripts/check-blockbook-traces.sh` compiles `scripts/blockbook-traces.rs`
against `src/blocks.rs` and `src/blockbook.rs`, replays 20 walks of 90 operations through
`specs/StartTailsTrace.tla` (`InitConforms`, `StepConforms`, `StatesMatch`, `RefsMatch`,
`ReadsMatch`, plus all `StartTails` invariants), verifies coverage of all 8 scenarios,
and pins 6 mutations (`unheld-window`, `pruned-under-start`, `overread`, `no-zero`,
`kept-tail`, `every-off`) to their exact TLC violations.

## Serve cache traces (`ServeCacheTrace`)

The PR `traces` tier runs `scripts/check-serve-cache-traces.sh`. Its CPU harness calls
`ServeBlockCache` with the actual `EntryTree`, host-backed `BlockPools`, and lane tables,
checking the `StartTails` and `PrefixTree` contracts at serving operation boundaries via
`specs/ServeCacheTrace.tla`: prompt and periodic retention, live windows in each start and
lane, pool reference ownership, deepest-prefix matching with a seed token left over,
refreshed window handles, and pruning after the last lane or start releases a path, with
source mutations required to produce TLC counterexamples.

## Serve scheduler traces (`ServeStepTrace`)

`scripts/check-serve-step-traces.sh` replays decisions taken by `ServeStepPlanner` against
`specs/ServeStepTrace.tla` across 33 deterministic workloads (and can replay recorded JSONL
from `serve::scheduler_trace::dump` or Metal serving runs via `--replay`), pinning source
mutations to TLC counterexamples.

```sh
INKY_SCHED_TRACE_SEED=1 INKY_SCHED_TRACE_OUT=/tmp/serve-seed-1.jsonl \
  cargo test --locked serve::scheduler_trace::dump -- --ignored --exact
bash scripts/check-serve-step-traces.sh --replay /tmp/serve-seed-1.jsonl
```

## `CacheScheduler`, trace checkers, and the HTTP 429 rule

See [CACHE_SCHEDULER.md](CACHE_SCHEDULER.md) for the full `CacheScheduler.tla` model
(`11` positive configs and `19` reversed witnesses in `specs/run_cache_scheduler.sh`).

- **HTTP 429 capacity refusal vs 60-second queue-wait timeout**:
  1. *Capacity / queue-limit refusal*: `CacheScheduler.tla` models atomic arrival refusal
     in `Arrive` (`WorstCaseBlocks(reqLen) > MemBudget`, or `Cardinality(QueuedLanes) = MaxQueue`
     without direct-admit headroom -> `"Refused"`). Inside that untimed model,
     `ActionNoQueuedRefusal` proves capacity accounting never retroactively evicts an
     accepted `"Queued"` request to `"Refused"`.
  2. *Wall-clock queue timeout in `src/serve.rs`*: `AdmissionGate::acquire_with_cancel`
     (`QUEUE_WAIT_TIMEOUT = 60s`) returns HTTP 429 immediately on `QueueFull` **and**
     returns HTTP 429 (`QUEUE_TIMED_OUT`) if a queued HTTP waiter's 60-second wall-clock
     deadline expires before a lane permit opens. Real-time HTTP deadlines sit outside
     the untimed `CacheScheduler` specification and remove the waiter from the queue cleanly
     just like client `Cancel`.
- **Cache and layout trace conformance (`scripts/check-cache-traces.sh`)**: compiles
  `scripts/cache-traces.rs` (`src/blocks.rs`, `src/cache_scheduler.rs`, `src/cache_trace.rs`)
  with `rustc`, requires non-zero coverage of `spill`, `restore`, `drop` (`drop_unreferenced`),
  `slide` (`slide_window`), `reclaim`, `admit_bypass`, `step_bypass`, `cancel`, and
  `refuse429`, exposes `last_used_tick` so `ActionLruOrder` rejects out-of-LRU eviction,
  checks all 5 bundles (`bl_base0`, `bl_high_fp16`, `bl_high_kv8`, `cs`, `sp_512`) in TLC,
  and pins 10 production mutations (`trunc-real-byte-offset`, `off1-byte-offset`,
  `reissue-gen-reset`, `stale-finish`, `admit-wait-dropped`, `step-bypass-dropped`,
  `priv-tail-dirty`, `noop-restore-block`, `non-lru-writeback`, `cap17-span-pins`).
- **Scheduler twin trace conformance (`scripts/check-scheduler-traces.sh`)**: compiles
  `scripts/scheduler-traces.rs` + `src/scheduler.rs`, checks `check_invariants()` after
  every step and `is_drained()` at completion across 18 traces (never dropping short
  walks), requires non-zero coverage of `cold_writebacks`, `disk_restores`, `refused_429`,
  `cancel_in_step`, `cancel_in_restore` (`BeginRestoreIO -> Cancel -> Cancelling -> CompleteRestoreIO`),
  `admit_bypasses`, `step_bypasses`, `slide_windows`, and `finished`, and pins 6
  mutations (`cancel-in-step-early-free`, `cancel-in-restore-early-free`,
  `bypass-bound-exceeded`, `drop-step-bypass-tick`, `drop-admit-bypass-tick`,
  `noop-restore-stutter`).
- **In-process `serve` harness (`src/serve/scheduler_trace.rs`)**: on macOS,
  `INKY_SCHED_TRACE_SEED=1 INKY_SCHED_TRACE_OUT=/tmp/serve-seed-1.jsonl cargo test --locked serve::scheduler_trace::dump -- --ignored --exact`
  drives `src/serve.rs` over a fake engine (`SCHEMA_VERSION = 2`: `arrive`, `refuse429`,
  `admit`, `refuse_ctx`, `build_step`, `finish_step`, `disconnect`, `cancel`, `finish`)
  and verifies drain across 33 seeds in unit tests.

## Poisoned-slot gate (`PoisonGate` and `PoisonGateTrace`)

`PoisonGate` models `src/poolgate.rs` (`poison_plan`, `apply`, `restore`, `poisoned`):
before each step every slot of a kind's pool that no stepping lane references is
poisoned, non-stepping live lanes' poisoned slots are saved first and restored after the
step, and lanes `Take` (zeroed), `Share`, and `Release` slots both between steps
(`phase = "idle"`) and inside a step's poisoned phase (`phase = "poisoned"` for pre-GPU
`ensure_span_blocks` `Take` and GPU `Write`; `phase = "finish"` for post-GPU canonical
`Share` and duplicate/window `Release` before `Restore`). `PoisonGate4` uses
`SYMMETRY SymLanes` over `3` lanes and `4` slots.

- **Invariants**: `NoFalsePoison`, `Coverage`, `LiveIntact`.
- **Reversed witnesses (`4`)**: `PoisonUnguarded` (`NoFalsePoison`), `PoisonNoSave`
  (`LiveIntact`), `PoisonOwnedOnly` (`Coverage`), and `PoisonNoZero` (`NoFalsePoison`,
  when a slot taken during the poisoned phase is not zeroed).
- **Trace binding**: `scripts/check-poisongate-traces.sh` compiles
  `scripts/poisongate-traces.rs` against `src/poolgate.rs` and `src/blocks.rs`, replays
  `poolgate::poisoned` walks through `specs/PoisonGateTrace.tla` (`InitConforms` +
  `ActionConforms` + `NoFalsePoison` + `Coverage` + `LiveIntact`), requires non-zero
  idle and in-phase `take`/`share`/`release` plus `saved_slots`, and pins 5 mutations
  (`unguarded-stepping`, `no-save-live`, `owned-only-poison`, `skip-restore-copy`,
  `empty-stepping-action`).

## Sliding-window scratch tail (`ScratchTail`)

`ScratchTail` models the row layout of sliding-window K/V scratch across multi-span
prefill steps (`src/engine.rs` `layer_b`; `k_zero`, `k_kvcopy`, `k_kvprep_b`,
`k_attn_h`), checking `NoForeignRead` and `InBounds` across `ScratchTail3` and
`ScratchTail4`, witness `ScratchTailNoZero` (`NoForeignRead`), and negative control
`ScratchTailOneSpan` (`Bug = "notail"` with one span per step, proving tail zeroing is
needed specifically when multiple spans share a step).
