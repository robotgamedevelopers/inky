# CacheScheduler model and TLC results

Read against the **Blocks**, **Scheduler**, and **Models** sections of
[docs/DESIGN.md](../docs/DESIGN.md), alongside `specs/BlockLayout.tla` (PR #9).
Where `BlockLayout` models physical pool slot reuse, generation counters, and
64-bit table offsets at capacity 3 or 4, `CacheScheduler` models the coupled
cache-tier, start-point, step-fence, admission, restore-reservation, and
starvation/liveness contract across blocks and lanes.

Run on a **Linux bench**, never the engine's MacBook. With Java and
[tla2tools.jar](https://github.com/tlaplus/tlaplus/releases) installed:

```sh
TLC_OUTPUT=/var/tmp/cache-scheduler-results sh specs/run_cache_scheduler.sh
```

The runner honors `TLC_JAR`, `TLC_JAVA`, `TLC_OUTPUT`, and `TLC_WORKERS`
(matching `specs/run_tlc.zsh` in PR #9, while auto-detecting `/opt/tla/tla2tools.jar`
or `~/sdk/tla/tla2tools.jar`), checks every positive configuration to exhaustion
(requiring zero queued states left and `Model checking completed. No error has been found.`),
and requires each reversed witness to fail on its intended invariant or temporal
property.

## Shared vocabulary and mapping to `BlockLayout`

- **Block kinds (`Kinds == {"full", "window"}`)**: `BlockSize` corresponds to
  `B` in `BlockLayout` (64 positions in production, 2 here) and `WindowSize`
  corresponds to `W` in `BlockLayout` (1024 positions in production, 2 here).
  The logical range operators `Ceil`, `SpanBound`, `Window`, `History`,
  `SpanWindow`, and `Full` are identical between both specs.
- **Sealed tree blocks vs unsealed tails**: `TreeIds` hold sealed, immutable
  prefix-tree blocks (`<<"tree", prefix, kind, idx>>`) across tiers `"memory"`,
  `"disk"`, and `"both"` (`"none"` when absent or dropped). `PrivIds` hold a
  lane's private unsealed tail (`<<"priv", lane, kind>>`) in `"memory"`, zeroed
  past `length`, invisible to prefix matching, with reference count 1; on a
  step-ending block boundary (`FinishStep`), completed blocks seal into
  `TreeIds`, and on cancellation unsealed tails are dropped.
- **Start points (`startPoints[prefix, boundary]`)**: created only when a step
  ends on a block boundary (`pos % BlockSize = 0`, where recurrent convolution
  state exists), retaining the recurrent state marker, `Full(boundary)`, and
  `History(boundary)` until retired under disk/memory pressure.
- **Lanes (`Queued`, `Admitted`, `Restoring`, `Ready`, `InStep`, `Cancelling`,
  `Cancelled`, `Finished`, plus `Idle` / `Refused` slot states)**: `Admit`
  enforces worst-case memory admission (`MemBudget`) with queue cap `MaxQueue`
  (answering `Refused` / HTTP 429 beyond the limit). `BeginRestore` matches the
  deepest kept start point and reserves the lane's working set atomically in
  `MemBudget` before restoring disk-only blocks.
- **One step in flight (`stepActive`, `stepLanes`, `stepSpan`, `stepPins`,
  `stepIO`, `gpuDone`)**: `BuildStep` enforces the aggregate step-token budget
  (`StepTokenBudget`) with longest-waiting Ready lane first, then Ready decoding
  lanes (`1`-token spans), then prefill spans (up to `MaxSpan`) to fill the
  remaining token budget without stalling Ready decoders that fit in memory. It
  allocates any missing tail blocks in `"memory"` and pins every `Full(pos + c)`
  and `SpanWindow(pos, c)` block (plus any pending background writeback `stepIO`).
  Cancelling a lane in `InStep` transitions it to `Cancelling` and holds its
  references until both `gpuDone = TRUE` and `stepIO = {}` at `FinishStep`, which
  also prunes each surviving lane's `laneWin` table to `History(npos)` at fence
  completion so expired window blocks never accumulate past the lane's reservation.
- **Stage-consistent age & bypass bound (`MaxBypass`, the scheduler's `W`)**:
  `waitTicks` tracks consecutive bypasses within each waiting stage (`Queued` in
  `Admit`/`Refuse429`, `ActiveRestoring` in `RestoreBlock`, and `SteppableReady`
  in `BuildStep`), serving longest-waiting peers first, shedding youngest arrivals
  first on 429 queue overflow, and locking out younger arrivals once any waiting
  peer hits `MaxBypass`.

## Checked properties and bounded scope

Safety invariants checked on healthy runs:

- `TypeOK`: structural types and bounded ranges of all 17 state variables.
- `StepBlocksInMemoryAndOwned`: while `stepActive` holds, every block in
  `stepPins \cup stepIO` is in `"memory"` (or `"both"`) and owned by a lane in
  `InStep` or `Cancelling`, and every span's `Full` and `SpanWindow` blocks are
  pinned.
- `SealedValid` and temporal `ActionSealedImmutable` (`[][SealedNeverWritten]_vars`):
  sealed blocks have `length = BlockSize`, full data, and never change contents
  or unseal while present.
- `UnsealedPrivateInvisibleZero`: unsealed blocks live only in `"memory"`, have
  exact reference count 1 (a single lane referrer, zero start-point referrers),
  are invisible to `BestBoundary` matching, and have `data[i] = 0` for all
  `i >= length`.
- `RefCountMatchesReferrers`: every block's `refCount` equals the cardinality of
  its live lane and start-point referrers, and `refCount > 0 => tier # "none"`.
- `StartPointsKept` and `LaneHistoryCovered`: every kept start point retains its
  state marker, sealed `Full(k)` blocks, and sealed `History(k)` window blocks;
  every `Restoring`, `Ready`, and `InStep` lane covers `Full(pos)` and
  `History(pos)`.
- `BudgetsAndReservations` and `ColdBlocksSpillable`: `MemUsed <= MemBudget`,
  `DiskUsed <= DiskBudget`, `ReservedTotal <= MemBudget`, each active lane's
  reservation covers its working set, and every memory block unreferenced by any
  lane (and outside `stepIO`) is clean and spillable/reclaimable.
- `BypassBoundHolds` and `StepBudgetHolds`: `waitTicks[l] <= MaxBypass` across
  `Queued`, `Restoring`, and `Ready`, aggregate step tokens stay within
  `StepTokenBudget`, and decoding lanes take `1`-token spans.

Liveness checked under weak fairness (`WF_vars`) of scheduler, tier-reclamation,
and step-completion actions (with client `Arrive`, `Cancel`, `SlideWindow`, and
`DropUnreferenced` left unfair):

- `LiveRequestResolves`: every queued or admitted request on symmetric
  `ProbeLane` leads to `Refused`, `Cancelled`, or `Finished`. Because `"Queued"`
  exits only to `{"Admitted", "Refused", "Cancelled"}` and `AdmittedStates`
  never reach `"Refused"` without first entering `{"Finished", "Cancelled"}`,
  this single 2-state Büchi obligation holds if and only if both (1) every
  queued request is admitted, refused, or cancelled and (2) every admitted lane
  finishes or is cancelled.

Bounded scope: these runs check small instances (`BlockSize = 2`,
`WindowSize = 2`, `MaxPos \in {2, 4, 8}`, `StepTokenBudget = 2`, `2` or `3`
reusable lane slots, `1` or `2` prefixes, `MemBudget \in {2, 4, 7}`,
`DiskBudget \in {3, 4}`). Slot reuse generations and 64-bit byte offsets belong
to `BlockLayout.tla`; engine trace conformance belongs to the Layer 3/4 CI gates.

## Bench receipts (`space`)

Validated on Linux bench `space` (32-core x86\_64, 125 GiB RAM), 2026-11-23,
with TLC 2.19 (revision `5a47802`, `/opt/tla/tla2tools.jar`), Eclipse Temurin
OpenJDK 21.0.12.1, `-workers 8`, `-Xmx8g`, `-XX:+UseParallelGC`. Wall times
include JVM startup (`44.019 s` total across all 8 runs via `specs/run_cache_scheduler.sh`):

| Run | Bounds (`Lanes`, `Prefixes`, `MaxPos`, `StepTokenBudget`, `Mem`/`Disk`, `MaxQueue`, `W`) | Generated | Distinct | Wall seconds | Result |
| --- | --- | ---: | ---: | ---: | --- |
| `CacheScheduler` | `{a,b}`, `{p1}`, `4` (`{2,4}`), `2`, `4`/`3`, `1`, `1` | 6,525 | 1,881 | 2.924 | PASS (exhaustive safety + liveness, decode priority) |
| `CacheSchedulerQueue` | `{a,b,c}`, `{p1}`, `4` (`{2,4}`), `2`, `4`/`3`, `1`, `1` | 126,532 | 23,964 | 18.766 | PASS (exhaustive safety + liveness, 429 refusal + bimodal `W`) |
| `CacheSchedulerTwoPrefixes` | `{a,b}`, `{p1,p2}`, `2` (`{2}`), `2`, `2`/`4`, `1`, `1` | 92,245 | 16,880 | 8.719 | PASS (exhaustive safety + liveness, cross-prefix disk spill/restore) |
| `CacheSchedulerMaxPos8` | `{a,b}`, `{p1}`, `8` (`{8}`), `2`, `7`/`4`, `0`, `1` | 24,737 | 6,971 | 5.446 | PASS (exhaustive safety + liveness, multi-chunk prefill + fence window pruning) |
| `CancelInStep` | `{a,b}`, `{p1,p2}`, `4` (`{2,4}`), `2`, `4`/`4`, `0`, `1`, `Bug="cancel_in_step"` | 787 | 295 | 1.056 | `StepBlocksInMemoryAndOwned` violated |
| `RestoreSplitBudget` | `{a,b}`, `{p1,p2}`, `2` (`{2}`), `2`, `2`/`4`, `1`, `1`, `Bug="restore_split"` | 46,805 | 12,234 | 4.102 | `LiveRequestResolves` violated (temporal stuttering) |
| `BoundaryWindowDropped` | `{a,b}`, `{p1}`, `4` (`{2,4}`), `2`, `4`/`4`, `1`, `1`, `Bug="boundary_window_dropped"` | 2,296 | 799 | 1.391 | `LaneHistoryCovered` violated |
| `UnprunedWindow` | `{a,b}`, `{p1}`, `8` (`{8}`), `2`, `7`/`4`, `0`, `1`, `Bug="unpruned_window"` | 1,944 | 731 | 1.615 | `BudgetsAndReservations` violated |

## Inspected reversed-witness counterexamples

- **`CancelInStep` (`StepBlocksInMemoryAndOwned`, state 7)**: in state 6
  (`BuildStep`), lane `a` enters `InStep` (`stepActive = TRUE`, `gpuDone = FALSE`)
  with `stepPins = {<<"priv", a, "full">>, <<"priv", a, "window">>}`. In state 7
  (`Cancel`), bypassing `Cancelling` sends `a` straight to `Cancelled` and
  clears its block table while the GPU step is still in flight.
- **`RestoreSplitBudget` (`LiveRequestResolves`, stuttering)**: cold
  start points `<<p1, 2>>` and `<<p2, 2>>` each hold 2 blocks on disk under
  `MemBudget = 2`. Without atomic working-set reservation in `BeginRestore`,
  lanes `a` (`p1`) and `b` (`p2`) both enter `Restoring`, restore
  one block each into `"memory"` (`MemUsed = 2 = MemBudget`), and
  stutter forever in `laneState = (a :> "Restoring" @@ b :> "Restoring")` because
  all memory blocks are actively referenced and neither lane can load its second
  block.
- **`BoundaryWindowDropped` (`LaneHistoryCovered`, state 13)**: lane `a` steps
  to boundary `pos = 2` and seals `<<p1, 2>>`, then finishes. Without start-point
  retention on `History(2)`, `DropUnreferenced` evicts `<<"tree", p1, "window", 0>>`
  while `<<"tree", p1, "full", 0>>` stays kept. On `BeginRestore(b)`, lane `b`
  matches boundary `pos = 2` with its required window block missing (`tier = "none"`).
- **`UnprunedWindow` (`BudgetsAndReservations`, state 16)**: with `W = B = 2`,
  `RequestLengths = {8}`, and `MemBudget = 7`, lane `a` reserves `WorstCaseBlocks(8) = 7`
  blocks (`4` full + `3` window). Without pruning `laneWin` to `History(npos)` at
  `FinishStep` fence completion, three two-token prefill steps (`0 -> 2 -> 4 -> 6`)
  without an optional `SlideWindow` retain window blocks `0, 1, 2` alongside full
  blocks `0, 1, 2` (`6` blocks) at `lanePos[a] = 6`, giving `CurrentWorkingSet(a) = 8 > 7 = reservation[a]`.
