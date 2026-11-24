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
- **Continuous waiting age & bypass bound (`MaxBypass`, the scheduler's `W`)**:
  `waitTicks` carries continuously across `Queued -> Admitted -> Restoring -> Ready`
  (preserved by `Admit`, `BeginRestore`, uncontested `RestoreBlock`, and
  `FinishRestore`, so a request that waited in `Queued` is never reset behind
  fresher arrivals upon entering `Restoring` or `Ready`), serving longest-waiting
  peers first, shedding youngest arrivals first on 429 queue overflow, and
  incrementing bypassed peers only when an equal-or-younger peer precedes them.

## Checked properties and bounded scope

Safety and action properties checked on healthy runs:

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
- `BypassBoundHolds`, `StepBudgetHolds`, and `ActionDecodePriority`
  (`[ ][DecodePriorityStep]_vars`): `waitTicks[l] <= MaxBypass` across `Queued`,
  `Admitted`, `Restoring`, and `Ready`, aggregate step tokens stay within
  `StepTokenBudget`, decoding lanes take `1`-token spans, and after the single
  longest-waiting leader (with `maxW` ties broken toward Ready decoders), no
  non-leader prefill lane (`waitTicks < MaxBypass`) or `> 1`-token prefill span
  leaves out a Ready decoder that fits in memory.

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
reusable lane slots, `1` or `2` prefixes, `MemBudget \in {2, 4, 7, 10}`,
`DiskBudget \in {2, 3, 4}`). Slot reuse generations and 64-bit byte offsets belong
to `BlockLayout.tla`; engine trace conformance belongs to the Layer 3/4 CI gates.

## Bench receipts (`space`)

Validated on Linux bench `space` (32-core x86\_64, 125 GiB RAM), 2026-11-23,
with pinned TLA+ `v1.8.0` (`tla2tools.jar` SHA-256 `fdffef247f56...`) as well as
TLC 2.19 (`5a47802`), Eclipse Temurin OpenJDK 21.0.12.1, `-Xmx8g`,
`-XX:+UseParallelGC` (`TLC_WORKERS=8` wall times below; also verified 10/10 at
CI's `TLC_WORKERS=2`, and `CacheSchedulerDecodePriority` also verified under full
`Spec` `LiveRequestResolves` over all `417,054` distinct states):

| Run | Bounds (`Lanes`, `Prefixes`, `MaxPos`, `StepTokenBudget`, `Mem`/`Disk`, `MaxQueue`, `W`) | Generated | Distinct | Wall seconds | Result |
| --- | --- | ---: | ---: | ---: | --- |
| `CacheScheduler` | `{a,b}`, `{p1}`, `4` (`{2,4}`), `2`, `4`/`3`, `1`, `1` | 8,529 | 2,419 | 2.081 | PASS (exhaustive safety + liveness + `ActionDecodePriority`) |
| `CacheSchedulerQueue` | `{a,b,c}`, `{p1}`, `4` (`{2,4}`), `2`, `4`/`3`, `1`, `1` | 190,942 | 35,859 | 12.628 | PASS (exhaustive safety + liveness, 429 refusal + continuous `waitTicks`) |
| `CacheSchedulerTwoPrefixes` | `{a,b}`, `{p1,p2}`, `2` (`{2}`), `2`, `2`/`4`, `1`, `1` | 122,465 | 22,352 | 6.610 | PASS (exhaustive safety + liveness, cross-prefix disk spill/restore) |
| `CacheSchedulerMaxPos8` | `{a,b}`, `{p1}`, `8` (`{8}`), `2`, `7`/`4`, `0`, `1` | 29,429 | 8,171 | 5.352 | PASS (exhaustive safety + liveness, multi-chunk prefill + fence window pruning) |
| `CacheSchedulerDecodePriority` | `{a,b,c}`, `{p1}`, `4` (`{2,4}`), `2`, `10`/`2`, `0`, `1` | 2,210,338 | 417,054 | 32.096 | PASS (exhaustive mixed-phase 3-lane `ActionDecodePriority` + safety) |
| `CancelInStep` | `{a,b}`, `{p1,p2}`, `4` (`{2,4}`), `2`, `4`/`4`, `0`, `1`, `Bug="cancel_in_step"` | 761 | 306 | 0.833 | `StepBlocksInMemoryAndOwned` violated |
| `RestoreSplitBudget` | `{a,b}`, `{p1,p2}`, `2` (`{2}`), `2`, `2`/`4`, `1`, `1`, `Bug="restore_split"` | 44,351 | 12,361 | 4.078 | `LiveRequestResolves` violated (temporal stuttering) |
| `BoundaryWindowDropped` | `{a,b}`, `{p1}`, `4` (`{2,4}`), `2`, `4`/`4`, `1`, `1`, `Bug="boundary_window_dropped"` | 2,671 | 931 | 1.338 | `LaneHistoryCovered` violated |
| `UnprunedWindow` | `{a,b}`, `{p1}`, `8` (`{8}`), `2`, `7`/`4`, `0`, `1`, `Bug="unpruned_window"` | 2,446 | 889 | 1.533 | `BudgetsAndReservations` violated |
| `DecodeStalledByPrefill` | `{a,b,c}`, `{p1}`, `4` (`{2,4}`), `2`, `10`/`2`, `0`, `1`, `Bug="no_decode_priority"` | 78,199 | 19,137 | 3.102 | `ActionDecodePriority` violated |

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
- **`DecodeStalledByPrefill` (`ActionDecodePriority`, state 14)**: with
  `MemBudget = 10`, `StepTokenBudget = 2`, and three equally aged Ready lanes
  (`waitTicks = 0` on `a, b, c`) holding mixed decode (`target = 2`) and prefill
  (`target = 4`) phases, checking only `waitTicks[p] < maxW` fails to separate a
  single longest-waiting leader from non-leader prefillers tied at `maxW`, allowing
  a prefill span to take a token in `stepLanes` while a Ready decoder that fits in
  memory is left out.
