# Block layout models

Read against the **Blocks** and **Models** sections of design revision PR #7
(`f46a55d`). These are bounded safety models, not a proof of the engine or of
scheduler liveness. `CacheScheduler` owns admission, cancellation fences,
parking, restore reservations, sealed-write immutability and liveness.
BlockLayout does not assert that sealed payloads are never written; that
transition property belongs to CacheScheduler.

Run on the designated **mini-m5 bench**, never the engine's MacBook. With Java and
[tla2tools.jar](https://github.com/tlaplus/tlaplus/releases) installed:

```sh
TLC_OUTPUT=/tmp/block-layout-results zsh specs/run_tlc.zsh
```

The runner defaults to `~/sdk/bin/java` and `~/sdk/tla/tla2tools.jar`;
set `TLC_JAVA` or `TLC_JAR` to override those paths.
The runner requires complete successful output for positive runs and the exact
expected invariant failure for each witness. It keeps full TLC output, including
counterexamples, in the output directory. For an individual run, from `specs/`:

```sh
~/sdk/bin/java -XX:+UseParallelGC -Xmx4g -cp ~/sdk/tla/tla2tools.jar tlc2.TLC \
  -workers 1 -config BlockLayout3.cfg BlockLayout.tla
```

`InBlock` is the order inside one slot: `[pass][K|V][head][position][HD]`, 2 passes, 3 heads, 4 positions, `HD = 2`. The action walks the coordinates in that order and inlines the address, then appends it to `seen`. `Ordered` compares the stored address to `Formula`, which the action does not call. `Inside` says one row stays inside the slot. `NoCollision` says the addresses in `seen` are unique. `InBlock` uses `kvb = 2` and `InBlockKv8` uses `kvb = 1`. `InBlockMajor` must violate `Ordered`. `InBlockMajorCollide` and `InBlockAddrCol` must violate `NoCollision` on a reached state. `InBlockTraceData` is written by `inky inblock-trace` from `key_row_at`, and `InBlockTrace` checks that behaviour at `kvb` 2 and at `kvb` 1.

## Bounds and vocabulary

`BlockLayout` has two lanes, **one pool per kind** (`FullCapacity=2` with `WindowCapacity=1` or `2`, giving **3 or 4 slots total**), blocks of
**2 positions**, a window of **2 positions**, spans of **1 or 2 tokens**, maximum
position **4**, element width **`Kvb` (`1` or `2`)**, and **2 generations per slot**. Matching `src/blocks.rs` (PR #49 / #54), each kind owns a separate pool starting at byte offset 0 (`Kvb` MiB full stride, `5 * Kvb` MiB window stride), and each table entry carries `(slot, generation, offset, kind, kvb)`.
The next allocation chooses the
lowest available non-exhausted slot; we explore all action interleavings, not
all allocator placement policies. Generation exhaustion disables further reuse;
this is not a generation-wrap proof. Finite completed/exhausted states are
allowed, so deadlock checking is disabled. No state constraint prunes the search. The four-slot run reaches a window
straddling two blocks while retaining two full blocks; the three-slot run cannot
fit that state, so the off-by-one witness uses four slots.

- A **block** is `(kind, slot, generation)`. Pool entries also record logical
  block index, initialized length and data. A length of `B` means sealed.
- A lane's **table** maps `(kind, logical block index)` to a slot, generation and
  offset. `TableIndex` checks the logical index independently of slot selection;
  `GenerationMatches` checks every stored entry, including stale entries.
- A **span** prepares a target, allocates its missing blocks, pins all needed
  handles, writes tokens in order, then releases its pins. `budget` records the
  conservative window reservation `ceil((W-1+C)/B)+1`. The reservation can exceed
  the number of actual handles at early positions or favorable alignments.
- A **pin** prevents reuse until `EndStep`. It is a use of the lane's ownership,
  not a second writer/referrer to a private tail. GPU/I/O ordering belongs to
  `CacheScheduler`.
- A **start point** retains a state marker, all full blocks, and the window blocks
  before its boundary. Matching shares only those sealed blocks. The single
  start-point record abstracts a cache with one retained start point; its state
  marker abstracts recurrent contents, not their numerical correctness.
- An unsealed **tail** has at most one lane owner (exactly one while referenced),
  cannot be matched, and is zero past its initialized length. Finishing clears
  all lane references; an unreferenced pool cell may retain data until allocation
  zeroes it, including reuse of a previously written slot.

`History(n)` covers the last `W` written positions. A span starting at `n` reads
from `max(0,n-W+1)` through `n+C-1`, inclusively. Allocation can prepare several
blocks ahead; these remain private to the writing lane. Sliding is a separate
idle-lane action, so both delayed and immediate reclamation are explored.

TLC uses signed 32-bit integers. Plain `u64` byte offsets from base 0 (`(SlotBase + (slot - 1)) * stride_bytes(kind, Kvb)`) are represented losslessly in 32-bit-safe limbs `[hi |-> u64 >> 32, mib |-> (u64 >> 20) % 4096, sub |-> u64 % 2^20]`. `BlockLayout3` and `BlockLayout4` run at `SlotBase=0, Kvb=2`; `BlockLayoutHigh` runs at `SlotBase=2048, Kvb=2` (`>= 4 GiB` in both pools); `BlockLayoutKv8` runs at `SlotBase=4096, Kvb=1` (`>= 4 GiB` in both `--kv8` pools); and `Truncate32` drops `hi` at `SlotBase=2048` to prove accidental 32-bit truncation violates `Address64`. `Address64` checks pool bounds, non-zero high limbs above 4 GiB, strict within-block `AddBytes` progression, and pairwise byte-range non-overlap across distinct live slots.

`SpanPins` separately checks the **production dimensions** `B=64, W=1024, C=512`
at every one of the 64 possible start alignments. The maximum is exactly 25
window blocks. This avoids pretending that a tiny pool can hold a production
span. It checks logical pin coverage; the bounded pool model checks pin lifetime.

## Reversed witnesses

Each configuration changes only the named defect (the full-scale pin witness
also selects the worst alignment). Healthy configurations enable every invariant.

| Configuration | Expected failure | Counterexample |
| --- | --- | --- |
| `StaleEntry` | `GenerationMatches` | Finish leaves a table entry behind; another lane reuses its slot at generation 2 while the entry still holds generation 1. |
| `WindowOffByOne` | `WindowCovered` | At position 3, sliding drops logical block 0, although the last two positions are 1 and 2 and still cross blocks 0 and 1. |
| `SpanTooSmall` | `SpanPinned` | A two-token span reserves the single-token budget 2, below the specified bound 3. This is a reservation-contract failure, not a claim that this tiny alignment reads three blocks. |
| `StartReleased` | `StartRetained` | Keep a boundary-2 start point, finish its lane, then reuse its window slot; its full block remains retained but the window generation changes. |
| `Truncate32` | `Address64` | At `SlotBase = 2048` (`>= 4 GiB` in both pools), dropping the upper 32 bits (`hi = 0`) aliases a high-pool slot onto offset 0 and violates `Address64`. |
| `SpanPins17` | `Covered` | At offset 63, a 512-token span reads logical blocks 0 through 24; pinning 0 through 16 leaves eight blocks unpinned. |

Results and timings from mini-m5 are recorded in [RESULTS.md](RESULTS.md).

## Grouped tile traversal

`GroupedLayout` checks the layout part of [the shared-prefix model gate](../docs/SHARED-PREFIX.md#gates-in-landing-order)
for a single attention pass over two pinned lane tables. It enumerates all
positive row lengths and complete shared-prefix lengths that fit in three or
four physical slots. Each block holds two keys and each split holds four keys.
A tile is one block in this model. Production uses 64-key blocks, and the
[private fixed-split kernel](https://github.com/robotgamedevelopers/inky/pull/37)
uses 256-key full splits and 64-key window splits. A grouped kernel may stage
several tiles per block. Its tile width and block crossings need their own
kernel and trace checks. TLC requires that each modeled split contains whole
tiles.

A shared block occupies one slot across both rows; after divergence, each row
needs a separate slot for each block, including its tail.

For each tile, TLC explores staging the distinct physical slots in either order.
A staged slot serves the rows that reference it. Each row applies its own causal
end and either full attention or a three-key sliding window. At completion TLC
compares the sequence of key positions, physical identities and split indices
with an independent scalar enumeration. It also compares the sequence of
emitted partials, including the last key of each visible split. A split can
start on a shared tile and finish on a private tile. A staging trace records
the physical slot and tile of every load. Each pair must occur once, and the
completed trace must cover every pair expected by the private rows.

Under weak fairness of the enabled traversal action, each snapshot must finish.
The liveness witness blocks the first advance after its tile's loads finish.
That configuration turns off TLC's earlier deadlock diagnostic so the checker
reports the infinite stutter that violates traversal completion.

| Witness | Checked property | What changes |
| --- | --- | --- |
| `LocalMatch` | `VisitedKeys` | Two complete blocks have the same local tokens after their parents diverged. Row 2 reads row 1's slot. |
| `DonorWindow` | `VisitedKeys` | Row 2 applies row 1's window start and visits the wrong keys. |
| `TilePartial` | `OrderedPartials` | Emitting at the first tile end produces an extra partial inside a split. |
| `Restage` | `StagedOnce` | A physical slot is loaded twice. The row's visited keys and emitted partials stay unchanged. |
| `StalledTraversal` | `TraversalCompletes` | Both rows can finish their loads, yet the traversal stays on tile zero forever. |

The ordered keys represent logical position and mask membership. SIMD lane
assignment, floating-point operations and Metal execution require the kernel's
bitwise tests. Block allocation and lifetime are checked in their own models.

## Start points and unsealed tails

`StartTails` checks the bookkeeping in `src/blockbook.rs`: which blocks a start
point holds, when start points are kept and dropped, what attention may read of a
tail, and zeroing on reuse. As in `src/blocks.rs`, each kind has its own pool of
1-based slots, and a slot record holds a generation, a length and a sealed flag.
Reference counts are derived from the tables, never stored. The step in flight and
its pins stay in `BlockLayout`; tiers, budgets and the scheduler stay in
`CacheScheduler`. Its start-point record (state, full blocks, window blocks of a
boundary) is the same record as `CacheScheduler.startPoints`.

- A **start point** at boundary `n` holds the recurrent state at `n`, every full
  block before `n` and the window blocks of `History(n)`, each as a reference on
  its slot. Pruning the lane's window drops only the lane's reference.
- A **step** writes its positions through the lane's table, seals what it filled,
  prunes the window to `History`, and, when it ends a prefill on a block boundary,
  keeps a start point. The point's reasons are `head`, `prompt` at the prompt's
  last block boundary, and `every` at every `Every`-th block (16 in production).
  A lane has one head: the previous head loses that reason first, and a point left
  without one is dropped. `Finish` drops the head reason, so prompt and every
  points outlive the request. Decode never keeps a point (decision 4).
- **Tails** are allocated a block at a time, full and window together, each in
  the lowest usable slot of its pool, at the next generation, zeroed. Neither is
  taken unless both can be. A tail is referenced by its lane alone, is never in a
  start point, and is free once its lane finishes. Attention reads block `b` of a
  lane at `n` only up to `Seen(n, b)`.

Bounds: two lanes, **two slots per kind** in CI (four in all) and three per kind
nightly, blocks and windows of 2 positions, maximum position 4, spans up to 2,
start points every 2 blocks, one start-point id (a new head can reuse its
predecessor's id) and 2 generations per slot.

| Configuration | Expected failure | Counterexample |
| --- | --- | --- |
| `UnheldWindow` | `StartRetained` | A start point holds only its full blocks. Lane 1 keeps a prompt point at 2 and finishes; lane 0 takes window slot 1 at generation 2 while the point still holds generation 1. |
| `PrunedUnderStart` | `StartRetained` | Pruning releases a window slot outright. Lane 1 keeps a prompt point at 2, steps to 4 and prunes window block 0; lane 0 takes that slot at generation 2 under the live point. |
| `TailOverread` | `ReadsWritten` | Attention reads one position past the lane: block 0 just allocated at position 0, read length 1, written length 0. |
| `ReuseNotZeroed` | `TailPrivateZero` | Lane 0 writes position 0 and finishes; the next request takes the same slot at generation 2 with length 0 and position 0 still set. |

**Binding the code.** `scripts/check-blockbook-traces.sh` compiles
`scripts/blockbook-traces.rs` with the production `src/blocks.rs` and
`src/blockbook.rs` (rustc only, no Cargo or Metal). Twenty deterministic random
walks of 90 operations drive three lanes over five slots per kind, with a fake GPU
side that writes through the lane's table, zeroes on `zero` and records kept
states. Each call is one recorded state. `StartTailsTrace` replays them: the first
state must be `Init` and each call must be a `StartTails` step or change nothing
(`Conforms`), with every model invariant, plus `RefsMatch` (each slot record's
count equals its lane and start-point referrers), `ReadsMatch` (each read the
code reports equals `Visible`) and `StatesMatch` (each kept state belongs to a
live point). The walks must reach every witnessed case (a resumed lane, a reused
slot, a window pruned under a live point, a partial tail read, a retired point,
a finished tail, prompt and every points, a full pool) or the check fails. Six
one-line mutations of `src/blockbook.rs` must each be rejected by TLC:
`unheld-window` and `pruned-under-start` (a), `overread` (b), `no-zero` (c),
`kept-tail` and `every-off`.

## Serve scheduler traces

On a Mac with the Rust toolchain, write one deterministic trace from the serve
scheduler. The test uses a fake inference engine and an in-process client
workload. It opens no sockets and loads no model.

```sh
INKY_SCHED_TRACE_SEED=1 INKY_SCHED_TRACE_OUT=/tmp/serve-seed-1.jsonl \
  cargo test --locked serve::scheduler_trace::dump -- --ignored --exact
```

The first JSON line records the schema version, seed, lane count, queue limit,
prefill chunk budget and context. Each subsequent line records one event in
execution order. Request IDs (`job`) identify arrivals through termination.
Lane indices and token positions start at zero.

| Event | Observation |
| --- | --- |
| `arrive`, `refuse429` | The client submits a request with prompt length and generation limit. A refusal is the outcome of the same atomic capacity claim used by both HTTP handlers. |
| `admit` | The scheduler assigns a request to a free lane and records its cached prefix length. |
| `build_step` | The scheduler submits a prefill or decode batch. Each span names its lane, request, starting position and token count. |
| `finish_step` | The submitted batch has completed. This is the engine fence for that step. |
| `disconnect`, `cancel` | The simulated client drops its receiver; the scheduler observes delivery failure and releases the lane. These can occur on different ticks. |
| `finish` | The request reaches its stop or length condition. The event includes the generated token count. |

The fake supplies cache state and logits at the engine boundary. The production
scheduler still chooses lanes and builds batches, samples tokens, sends events
and observes disconnects. The trace covers the current serving decisions.
The default test repeats 33 seeds with two lanes and one queued request and
checks that the trace drains. These workloads use cold prompts and a constant
winning token, so normal generation ends at the requested length. Cache hits,
model output and GPU execution have separate tests. A TLC replay against
CacheScheduler is a separate verification step.

## Poisoned-slot gate

`PoisonGate` models the slot arithmetic of layer 3's output gate
(`src/poolgate.rs`, `inky poolgate`): before each step every slot of a kind's
pool that no stepping lane references is poisoned, the poisoned slots a live
lane references are saved first and restored after the step, and free slots
stay poisoned until the pool zeroes them on reuse. One pool, 3 or 4 slots, two
or three lanes that take, share and release slots between steps, one write per
slot. The code runs the plan once per kind; the kinds share no slots.

- `NoFalsePoison`: no slot a stepping lane reads holds poison.
- `Coverage`: every slot the step must not read holds poison.
- `LiveIntact`: between steps every referenced slot holds what its lanes wrote.

Reversed witnesses (`Bug`): `unguarded` (the first stepping lane is not
protected; the same switch as `INK_POOLGATE_SABOTAGE`), `nosave` and
`ownedonly`. The unit tests in `src/poolgate.rs` check `poison_plan` against
the same three properties on every two-lane, three-slot table assignment.

## Private sliding-window prefill scratch

`ScratchTail` models the private prefill path's shared K/V scratch across spans
and sliding-window layers. Each span copies its history below scratch row zero
and writes its current keys starting at zero. Attention loads whole tiles.
Rows past the span's end in its final tile must remain zero because a masked
nonfinite value can still affect the sum. For multi-span steps the encoder
clears those rows before attention. The single-span path relies on the initial
scratch clear and the same span length in each layer.

`InBounds` checks that all tile reads fit in the allocated scratch.
`NoForeignRead` checks that each row read belongs to the current span or is
zero. `ScratchTailNoZero` requires a counterexample when a multi-span step
omits the clear. `ScratchTailOneSpan` checks the single-span case with that
clear disabled.
