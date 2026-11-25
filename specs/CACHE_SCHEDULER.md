# CacheScheduler model and TLC results

The reference `Scheduler` in `src/scheduler.rs` generates traces for this model.
The live server uses `ServeStepPlanner` to choose homogeneous batches from its
runnable lanes. Its decode priority, preemption at 16 bypasses, and exact token
spans are checked in `ServeStepTrace.tla`. See [the replay commands](README.md#serve-scheduler-traces).


Read against the **Blocks**, **Scheduler**, and **Models** sections of
[docs/DESIGN.md](../docs/DESIGN.md), alongside `specs/BlockLayout.tla` (PR #9).
Where `BlockLayout` models physical pool slot reuse, generation counters, and
64-bit table offsets at capacity 3 or 4, `CacheScheduler` models the coupled
cache-tier, start-point, step-fence, admission, restore-reservation, lazy
cold-block SSD writeback, and starvation/liveness contract across blocks and
lanes.

Run on a **Linux bench**, never the engine's MacBook. With Java and
[tla2tools.jar](https://github.com/tlaplus/tlaplus/releases) installed:

```sh
TLC_OUTPUT=/var/tmp/cache-scheduler-results sh specs/run_cache_scheduler.sh
```

The runner honors `TLC_JAR`, `TLC_JAVA`, `TLC_OUTPUT`, and `TLC_WORKERS`
(matching `specs/run_tlc.zsh` in PR #9, while auto-detecting `/opt/tla/tla2tools.jar`
or `~/sdk/tla/tla2tools.jar`), checks every positive configuration to exhaustion
(requiring zero queued states left and `Model checking completed. No error has been found.`),
and requires each reversed witness to fail on its intended invariant, action
property, or temporal property.

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
  `History(boundary)` until retired under disk/memory pressure (never while any
  of its blocks has in-flight `stepIO`).
- **Atomic arrival capacity check, bounded queue, and 429 refusal (`Arrive`, `Admit`)**:
  matching `serve.rs`, `Arrive` checks capacity atomically on submission. A
  request within `MemBudget` enqueues into `"Queued"` when `Cardinality(QueuedLanes) < MaxQueue`
  (or admits directly when `Cardinality(QueuedLanes) = MaxQueue` and it fits
  without violating any queued lane's `MaxBypass` bound); otherwise it transitions
  immediately to `"Refused"` (HTTP 429) without ever entering `"Queued"`. No
  accepted `"Queued"` request is ever retroactively 429-refused (`QueueBounded`
  and `ActionNoQueuedRefusal`). `BeginRestore` matches the deepest kept start
  point and reserves the lane's working set atomically in `MemBudget` before
  restoring disk-only blocks.
- **Lazy cold-only SSD writeback (`StartColdWriteback`, `CompleteStepIO`, `stepIO`)**:
  blocks referenced by no lane (`ColdMemBlocks`, `LaneReferrers(b) = {}`) are
  written to SSD lazily **only** under `MemPressure` (`StartColdWriteback` sets
  `stepIO = {b}` while keeping `b` in `"memory"` and owned; `CompleteStepIO`
  writes `b` to SSD and drops its memory copy to `"disk"`, or `"both"` only if a
  new lane matched `b` mid-writeback), enforced by `ActionLazyColdWriteback`.
  **Intentional scope narrowing (LRU victim policy)**: among multiple cold blocks
  eligible under `MemPressure`, victim selection uses nondeterministic `\E` choice
  rather than per-block recency timestamps, soundly over-approximating production
  LRU order (every LRU behavior is a behavior of the model) without timestamp
  state explosion.
- **One step in flight (`stepActive`, `stepLanes`, `stepSpan`, `stepPins`,
  `stepIO`, `gpuDone`)**: `BuildStep` enforces the aggregate step-token budget
  (`StepTokenBudget`) with longest-waiting Ready lane first, then Ready decoding
  lanes (`1`-token spans), then prefill spans (up to `MaxSpan`) to fill the
  remaining token budget without stalling Ready decoders that fit in memory. It
  allocates any missing tail blocks in `"memory"` and pins every `Full(pos + c)`
  and `SpanWindow(pos, c)` block. Cancelling a lane in `InStep` transitions it to
  `Cancelling` and holds its references until both `gpuDone = TRUE` and
  `stepIO = {}` at `FinishStep`, which also prunes each surviving lane's `laneWin`
  table to `History(npos)` at fence completion so expired window blocks never
  accumulate past the lane's reservation.
- **Cross-stage step-bypass bound (`MaxBypass`, the scheduler's `W`)**:
  `waitTicks` counts consecutive step ticks (`BuildStep`s, plus bimodal queue
  bypasses) that pass over any actionable waiting lane across `Queued`
  (`CanFitAdmit`), `Admitted`, `Restoring`, and `SteppableReady`
  (`StepBypassed(chosen)`), carrying intact through `Admit`, `BeginRestore`,
  `RestoreBlock`, and `FinishRestore` until the lane is chosen in `BuildStep`.
  `BuildStep` forbids passing over any lane in `StepBypassed(chosen)` once its
  `waitTicks` reaches `MaxBypass` (`BypassBoundHolds`).

## Checked properties and bounded scope

Safety invariants and action properties checked on healthy runs:

- `TypeOK`: structural types and bounded ranges of all 17 state variables.
- `StepBlocksInMemoryAndOwned`: while `stepActive` holds, every block in
  `stepPins` is in `"memory"` (or `"both"`) and owned by a lane in `InStep` or
  `Cancelling`, every span's `Full` and `SpanWindow` blocks are pinned, and at
  all times every block in `stepIO` stays in `"memory"` and owned (`refCount >= 1`).
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
- `QueueBounded` and `ActionNoQueuedRefusal` (`[][QueuedNeverRefused]_vars`):
  `Cardinality(QueuedLanes) <= MaxQueue`, every queued request fits `MemBudget`
  in the empty-machine worst case, and no accepted `"Queued"` request ever
  transitions to `"Refused"`.
- `ActionLazyColdWriteback` (`[][LazyColdWritebackStep]_vars`): SSD writeback
  starts only under `MemPressure` on sealed memory-only cold blocks
  (`LaneReferrers(b) = {}`) and drops the memory copy to `"disk"` on completion
  whenever no lane re-referenced `b` mid-writeback.
- `BypassBoundHolds`, `StepBudgetHolds`, and `ActionDecodePriority`
  (`[ ][DecodePriorityStep]_vars`): `waitTicks[l] <= MaxBypass` across `Queued`,
  `Admitted`, `Restoring`, and `Ready`, every `BuildStep` increments `waitTicks`
  on all bypassed actionable waiting lanes across all four waiting stages,
  aggregate step tokens stay within `StepTokenBudget`, decoding lanes take
  `1`-token spans, and after the single longest-waiting leader (with `maxW` ties
  broken toward Ready decoders), no non-leader prefill lane (`waitTicks < MaxBypass`)
  or `> 1`-token prefill span leaves out a Ready decoder that fits in memory.

Liveness checked under weak fairness (`WF_vars`) of scheduler, tier-reclamation,
and step-completion actions (with client `Arrive`, `Cancel`, `SlideWindow`, and
`DropUnreferenced` left unfair):

- `LiveRequestResolves`: every queued or admitted request on symmetric
  `ProbeLane` leads to `Refused`, `Cancelled`, or `Finished` (and combined with
  `ActionNoQueuedRefusal`, every accepted `"Queued"` or admitted request reaches
  `Cancelled` or `Finished`).

Bounded scope: these runs check small instances (`BlockSize = 2`,
`WindowSize = 2`, `MaxPos \in {2, 4, 8}`, `StepTokenBudget = 2`, `2` or `3`
reusable lane slots, `1` or `2` prefixes, `MemBudget \in {2, 4, 7, 10}`,
`DiskBudget \in {2, 3, 4}`). Cold victim selection abstracts LRU to
nondeterministic `\E` choice; slot reuse generations and 64-bit byte offsets
belong to `BlockLayout.tla`; engine trace conformance belongs to the Layer 3/4 CI
gates.

## Bench receipts (`b1`)

Validated on Linux bench `b1` (16-core x86\_64), 2026-11-24, with pinned TLA+
`v1.8.0` (`tla2tools.jar` SHA-256 `fdffef247f563ef6edbfd44958d2a82db79e7564e07cb673262d896353fa6713`),
Eclipse Temurin OpenJDK 21.0.12.1, `-Xmx8g`, `-XX:+UseParallelGC`
(`TLC_WORKERS=8` wall times below, `31.993 s` total across all 14 runs; also
verified 14/14 at CI's `TLC_WORKERS=2` in `58.934 s` total, and
`CacheSchedulerDecodePriority` additionally verified under full `Spec`
`LiveRequestResolves` over all `102,327` distinct states in `39 s`):

| Run | Bounds (`Lanes`, `Prefixes`, `MaxPos`, `StepTokenBudget`, `Mem`/`Disk`, `MaxQueue`, `W`) | Generated | Distinct | Wall seconds | Result |
| --- | --- | ---: | ---: | ---: | --- |
| `CacheScheduler` | `{a,b}`, `{p1}`, `4` (`{2,4}`), `2`, `4`/`3`, `1`, `1` | 3,997 | 1,148 | 1.186 | PASS (exhaustive safety + liveness + action properties) |
| `CacheSchedulerQueue` | `{a,b,c}`, `{p1}`, `4` (`{2,4}`), `2`, `4`/`3`, `1`, `1` | 71,698 | 13,860 | 5.412 | PASS (exhaustive safety + liveness, atomic 429 + bimodal `W`) |
| `CacheSchedulerTwoPrefixes` | `{a,b}`, `{p1,p2}`, `2` (`{2}`), `2`, `2`/`4`, `1`, `1` | 113,661 | 19,203 | 5.533 | PASS (exhaustive safety + liveness, lazy cold `stepIO` spill/restore) |
| `CacheSchedulerMaxPos8` | `{a,b}`, `{p1}`, `8` (`{8}`), `2`, `7`/`4`, `0`, `1` | 1,635 | 599 | 0.338 | PASS (exhaustive safety + liveness, multi-chunk prefill + fence window pruning) |
| `CacheSchedulerDecodePriority` | `{a,b,c}`, `{p1}`, `4` (`{2,4}`), `2`, `10`/`2`, `0`, `1` | 513,469 | 102,327 | 8.167 | PASS (exhaustive mixed-phase 3-lane `ActionDecodePriority` + safety) |
| `CacheSchedulerQueuedBypass` | `{a,b}`, `{p1}`, `4` (`{2,4}`), `2`, `10`/`2`, `1`, `1` | 14,469 | 4,171 | 2.061 | PASS (exhaustive safety + liveness, `W=1` Ready + admissible Queued bound) |
| `CancelInStep` | `{a,b}`, `{p1,p2}`, `4` (`{2,4}`), `2`, `4`/`4`, `0`, `1`, `Bug="cancel_in_step"` | 463 | 142 | 0.586 | `StepBlocksInMemoryAndOwned` violated |
| `RestoreSplitBudget` | `{a,b}`, `{p1,p2}`, `2` (`{2}`), `2`, `2`/`4`, `1`, `1`, `Bug="restore_split"` | 75,697 | 16,976 | 3.735 | `LiveRequestResolves` violated (temporal stuttering) |
| `BoundaryWindowDropped` | `{a,b}`, `{p1}`, `4` (`{2,4}`), `2`, `4`/`4`, `1`, `1`, `Bug="boundary_window_dropped"` | 2,171 | 734 | 0.801 | `LaneHistoryCovered` violated |
| `UnprunedWindow` | `{a,b}`, `{p1}`, `8` (`{8}`), `2`, `7`/`4`, `0`, `1`, `Bug="unpruned_window"` | 806 | 338 | 0.697 | `BudgetsAndReservations` violated |
| `DecodeStalledByPrefill` | `{a,b,c}`, `{p1}`, `4` (`{2,4}`), `2`, `10`/`2`, `0`, `1`, `Bug="no_decode_priority"` | 17,061 | 5,625 | 1.476 | `ActionDecodePriority` violated |
| `QueuedBypassedByStep` | `{a,b}`, `{p1}`, `4` (`{2,4}`), `2`, `10`/`2`, `1`, `1`, `Bug="queued_bypass"` | 1,271 | 532 | 0.689 | `BypassBoundHolds` violated |
| `DeferredTied429` | `{a,b,c}`, `{p1}`, `4` (`{2,4}`), `2`, `4`/`3`, `1`, `1`, `Bug="deferred_tied_429"` | 667 | 351 | 0.632 | `ActionNoQueuedRefusal` violated |
| `EagerHotWriteback` | `{a,b}`, `{p1}`, `4` (`{2,4}`), `2`, `10`/`2`, `1`, `1`, `Bug="eager_hot_writeback"` | 1,146 | 475 | 0.681 | `ActionLazyColdWriteback` violated |

Runner negative controls: both `TLC_JAVA=/usr/bin/true` (exit 0 with empty log)
and `TLC_JAVA=/usr/bin/false` (exit 1) were rejected by `specs/run_cache_scheduler.sh`.

## Inspected reversed-witness counterexamples

- **`CancelInStep` (`StepBlocksInMemoryAndOwned`, state 5)**: lane `a` enters
  `InStep` (`stepActive = TRUE`, `gpuDone = FALSE`) with
  `stepPins = {<<"priv", a, "full">>, <<"priv", a, "window">>}`; bypassing
  `Cancelling` sends `a` straight to `Cancelled` and clears its block table
  while the GPU step is still in flight.
- **`RestoreSplitBudget` (`LiveRequestResolves`, stuttering)**: cold
  start points `<<p1, 2>>` and `<<p2, 2>>` each hold 2 blocks on disk under
  `MemBudget = 2`. Without atomic working-set reservation in `BeginRestore`,
  lanes `a` (`p1`) and `b` (`p2`) both enter `Restoring`, restore one block each
  into `"memory"` (`MemUsed = 2 = MemBudget`), and stutter forever in
  `laneState = (a :> "Restoring" @@ b :> "Restoring")`.
- **`BoundaryWindowDropped` (`LaneHistoryCovered`, state 13)**: lane `a` steps
  to boundary `pos = 2` and seals `<<p1, 2>>`, then finishes. Without start-point
  retention on `History(2)`, `DropUnreferenced` evicts `<<"tree", p1, "window", 0>>`
  while `<<"tree", p1, "full", 0>>` stays kept; `BeginRestore(b)` then matches
  boundary `pos = 2` with its required window block missing (`tier = "none"`).
- **`UnprunedWindow` (`BudgetsAndReservations`, state 15)**: with `W = B = 2`,
  `RequestLengths = {8}`, and `MemBudget = 7`, lane `a` reserves `WorstCaseBlocks(8) = 7`
  blocks. Without pruning `laneWin` to `History(npos)` at `FinishStep` fence
  completion, three two-token prefill steps (`0 -> 2 -> 4 -> 6`) without an
  optional `SlideWindow` retain window blocks `0, 1, 2` alongside full blocks
  `0, 1, 2`, giving `CurrentWorkingSet(a) = 8 > 7 = reservation[a]`.
- **`DecodeStalledByPrefill` (`ActionDecodePriority`, state 12)**: with
  `MemBudget = 10`, `StepTokenBudget = 2`, and three equally aged Ready lanes
  (`waitTicks = 0` on `a, b, c`) holding mixed decode (`target = 2`) and prefill
  (`target = 4`) phases, checking only `waitTicks[p] < maxW` allows a non-leader
  prefill span tied at `maxW` to take a token in `stepLanes` while a Ready
  decoder that fits in memory is left out.
- **`QueuedBypassedByStep` (`BypassBoundHolds`, state 10)**: at `W = 1`
  (`MaxBypass = 1`) and `MemBudget = 10`, Ready lane `a` and admissible `Queued`
  lane `b` (`CanFitAdmit(b) = TRUE`) reach `BuildStep` #1 (`waitTicks[b] = 1`);
  guarding only `SteppableReady \ chosen` in `BuildStep` allows `a` to take a
  second consecutive `BuildStep` while `b` remains queued (`waitTicks[b] = 2 > 1`).
- **`DeferredTied429` (`ActionNoQueuedRefusal`, state 6)**: deferring capacity
  refusal past `Arrive` lets overflow requests enter `"Queued"` (`Cardinality(QueuedLanes) > MaxQueue`)
  and subsequently transitions a `"Queued"` request to `"Refused"` via `Refuse429`.
- **`EagerHotWriteback` (`ActionLazyColdWriteback`, state 9)**: eagerly starting
  `stepIO` on pinned hot blocks in `BuildStep` writes `<<"tree", p1, "full", 0>>`
  while `MemUsed = 2 < MemBudget = 10` (`~MemPressure`) and while lane `a`
  actively references the block (`LaneReferrers # {}`).
