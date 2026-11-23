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

## Bounds and vocabulary

`BlockLayout` has two lanes, **3 or 4 slots total**, blocks of
**2 positions**, a window of **2 positions**, spans of **1 or 2 tokens**, maximum
position **4**, and **2 generations per slot**. Full and window blocks share a tagged pool; payload sizes are abstracted away.
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

TLC uses signed 32-bit integers. Offsets therefore use `(hi,lo)` 32-bit limbs:
`hi=2`, `lo=(slot-1)*4096`, representing addresses at or above 8 GiB. Within-block reads
preserve the high limb and add two bytes per position. The bounded additions
never carry out of the low limb: `Address64` explicitly requires a nonnegative
low limb and room for the last within-block byte addition below 2^31. This
stronger bounded condition avoids TLC integer overflow and proves no 32-bit
low-limb carry for these offsets. This checks high-bit preservation and table
indexing, **not general 64-bit carry/overflow arithmetic or Metal kernels**.

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
| `SpanPins17` | `Covered` | At offset 63, a 512-token span reads logical blocks 0 through 24; pinning 0 through 16 leaves eight blocks unpinned. |

Results and timings from mini-m5 are recorded in [RESULTS.md](RESULTS.md).
