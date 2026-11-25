# CacheScheduler model and TLC results

Read against the **Blocks**, **Scheduler**, and **Models** sections of
[docs/DESIGN.md](../docs/DESIGN.md), alongside [README.md](README.md) and `specs/BlockLayout.tla`.
Where `BlockLayout` models physical pool slot reuse, generation counters, and
64-bit table offsets at capacity 3 or 4, `CacheScheduler` models the coupled
cache-tier, start-point (including recurrent state tier), step-fence, admission,
restore-reservation, async cancel-during-restore fence, lazy cold-block SSD
writeback, decode-tail non-sealing, and starvation/liveness contract across blocks
and lanes.

Run on a **Linux bench** (`-Xmx8g`, `.github/workflows/nightly.yml` `tla-scheduler`),
never a laptop:

```sh
TLC_OUTPUT=/var/tmp/cache-scheduler-results sh specs/run_cache_scheduler.sh
```

The runner honors `TLC_JAR`, `TLC_JAVA`, `TLC_OUTPUT`, and `TLC_WORKERS`, checks
every positive configuration to exhaustion (`0 states left on queue.` and
`Model checking completed. No error has been found.`), and requires each reversed
witness to fail on its exact named invariant, action property, or temporal property
(with every witness `.cfg` keeping all 13 safety invariants enabled).

## Shared vocabulary and key rules

- **Block kinds (`Kinds == {"full", "window"}`)**: `BlockSize` corresponds to `B`
  in `BlockLayout` (`64` in production; `2` in standard configs and `1` in
  `CacheSchedulerOversizeWindow`, where `WindowSize = 2 > BlockSize = 1` exercises
  multi-block windows).
- **Sealed prefill blocks vs unsealed private/decode tails**: `TreeIds` hold sealed,
  immutable prefix-tree blocks (`<<"tree", prefix, kind, idx>>`) across tiers
  `"memory"`, `"disk"`, and `"both"` (`"none"` when absent or dropped). `PrivIds`
  hold a lane's private unsealed tail (`<<"priv", lane, kind, idx>>`) in `"memory"`,
  zeroed past `length`, invisible to prefix matching, with reference count 1.
  Matching Decision 4 of `docs/DESIGN.md`, **only prefill steps** (`pos <= PromptEnd(target)`)
  seal full blocks into `TreeIds` or keep start points (`NoDecodeBlocksSealed`);
  decode-written blocks stay private and unsealed until the lane finishes or cancels.
- **Start points and recurrent state tiers (`startPoints[prefix, boundary]`)**:
  created when a prefill step crosses or ends on a block boundary, retaining sealed
  `Full(boundary)`, sealed `History(boundary)`, and a recurrent state marker with its
  own `stateTier \in {"none", "memory", "disk", "both"}` (`SpillStartState`,
  `EvictCleanStartState`, `RestoreStartState`) verified by `StartPointsKept`.
- **Arrival capacity check, bounded FIFO queue, and the HTTP 429 rule (`Arrive`, `Admit`)**:
  1. *Untimed capacity & queue-limit refusal (`CacheScheduler.tla`)*: `Arrive` checks
     capacity atomically on submission. Oversize requests (`WorstCaseBlocks(reqLen) > MemBudget`,
     exercised in `CacheSchedulerOversizeWindow`) or arrivals when `Cardinality(QueuedLanes) = MaxQueue`
     without direct-admit headroom transition immediately to `"Refused"` (HTTP 429).
     Inside this untimed scheduler model, `QueueBounded` and `ActionNoQueuedRefusal`
     prove that capacity accounting never retroactively evicts an accepted `"Queued"`
     request to `"Refused"`.
  2. *60-second wall-clock queue timeout in `src/serve.rs`*: in production HTTP
     serving (`AdmissionGate::acquire_with_cancel`, `QUEUE_WAIT_TIMEOUT = 60s`), a
     request is refused with HTTP 429 both immediately when `waiters.len() >= max_queue`
     (`QUEUE_FULL`) **and** if its 60-second wall-clock wait in the queue expires before
     a lane permit opens (`QUEUE_TIMED_OUT`). Real-time HTTP deadlines live outside
     this untimed TLA+ specification and drop the waiter from the queue cleanly just like
     client `Cancel`.
- **Atomic working-set reservation and async cancel-during-restore (`BeginRestore`, `restoreIO`)**:
  `BeginRestore` matches `BestBoundary(prefix, MatchCeiling(target))` and reserves the
  lane's worst-case working set atomically in `MemBudget`. Under `AsyncRestore = TRUE`
  (`CacheSchedulerCancelRestore`, `CancelInRestore`, and trace replay), `BeginRestoreIO`
  records an in-flight disk read in `restoreIO[l]`; cancelling a `Restoring` lane while
  `restoreIO[l] # NoBlock` transitions it to `"Cancelling"` and retains `restoreIO[l]`
  ownership until `CompleteRestoreIO` finishes (`StepBlocksInMemoryAndOwned`).
- **Lazy cold-only SSD writeback and LRU trace binding (`StartColdWriteback`, `stepIO`)**:
  cold memory blocks (`LaneReferrers(b) = {}`) spill lazily **only** under `MemPressure`
  (`ActionLazyColdWriteback`), and `BuildStep` / `ActionWritebackProtected` protect
  in-flight `stepIO` blocks from mutation or premature drop. In the exhaustive model,
  cold victim choice uses nondeterministic `\E` over-approximation to avoid timestamp
  state explosion; in `scripts/check-cache-traces.sh`, `src/cache_scheduler.rs` exports
  per-block and per-start-point `last_used_tick` and `src/cache_trace.rs` checks
  `ActionLruOrder` so any out-of-LRU eviction fails TLC (`non-lru-writeback`).
- **Cross-stage step-bypass bound (`MaxBypass`, ghost `ghostBypass`)**: `waitTicks`
  and independent ghost counter `ghostBypass` track consecutive passes over actionable
  waiting lanes across `Queued` (`CanFitAdmit`), `Admitted`, `Restoring`, and
  `SteppableReady` (`BypassBoundHolds`, checked at both `MaxBypass = 1` and
  `MaxBypass = 2`).

## Checked properties (`20` state variables, `13` invariants, `6` properties)

- **Safety invariants (`13`)**: `TypeOK`, `StepBlocksInMemoryAndOwned` (including
  `stepIO` and `restoreIO` ownership), `SealedValid`, `NoDecodeBlocksSealed`,
  `UnsealedPrivateInvisibleZero`, `LaneHistoryCovered`, `RefCountMatchesReferrers`,
  `StartPointsKept`, `BudgetsAndReservations`, `ColdBlocksSpillable` (requiring clean
  cold memory blocks to be reclaimable or spillable under `DiskBudget > 0 /\ SpillEnabled`),
  `BypassBoundHolds`, `StepBudgetHolds`, and `QueueBounded`.
- **Action properties (`5`)**: `ActionSealedImmutable`, `ActionNoQueuedRefusal`,
  `ActionLazyColdWriteback`, `ActionWritebackProtected`, and `ActionDecodePriority`.
- **Temporal liveness (`1`)**: `LiveRequestResolves` under weak fairness (`WF_vars`)
  of scheduler, restore, tier-reclamation, and step-completion actions.

## Bench receipts (2026-11-25)

Validated on a 64-core `x86_64` Linux bench with pinned TLA+ `v1.8.0` (`tla2tools.jar`
SHA-256 `fdffef247f563ef6edbfd44958d2a82db79e7564e07cb673262d896353fa6713`), Eclipse
Adoptium OpenJDK `21.0.12.1`, `-Xmx8g`, `-XX:+UseParallelGC`, `TLC_WORKERS=8`:

| Run | Key bounds / witness `Bug` | Generated | Distinct | Wall seconds | Result |
| --- | --- | ---: | ---: | ---: | --- |
| `CacheScheduler` | `{a,b}`, `{p1}`, `req={2,4}`, `B=2, W=2, mem=4, disk=3, Q=1, MaxBypass=1` | 20,763 | 5,695 | 22.505 | PASS |
| `CacheSchedulerQueue` | `{a,b,c}`, `{p1}`, `req={2,4}`, `Q=1, MaxBypass=1` | 438,574 | 81,960 | 63.338 | PASS |
| `CacheSchedulerQueue2` | `{a,b,c}`, `{p1}`, `req={2,4}`, `Q=2, MaxBypass=1` (FIFO `queueSeq`) | 486,682 | 93,909 | 71.929 | PASS |
| `CacheSchedulerTwoPrefixes` | `{a,b}`, `{p1,p2}`, `req={4}`, `mem=4, disk=4, Q=1` | 287,549 | 51,144 | 31.025 | PASS |
| `CacheSchedulerMaxPos8` | `{a,b}`, `{p1}`, `req={8}`, `MaxPos=8, mem=7, disk=4` | 3,597 | 1,277 | 4.371 | PASS |
| `CacheSchedulerDecodePriority` | `{a,b,c}`, `{p1}`, `req={2,4}`, `mem=10, disk=2` | 414,565 | 85,936 | 19.286 | PASS |
| `CacheSchedulerQueuedBypass` | `{a,b}`, `{p1}`, `req={2,4}`, `mem=10, Q=1, MaxBypass=1` | 11,917 | 3,569 | 5.546 | PASS |
| `CacheSchedulerBypass2` | `{a,b}`, `{p1}`, `req={2,4}`, `mem=10, Q=1, MaxBypass=2` | 16,043 | 4,725 | 5.537 | PASS |
| `CacheSchedulerMultiBlockSpan` | `{a,b}`, `{p1}`, `req={6}`, `B=2, MaxSpan=4, StepTokenBudget=4` | 567 | 209 | 2.559 | PASS |
| `CacheSchedulerOversizeWindow` | `{a,b}`, `{p1}`, `B=1, W=2` (2-block window), `req={3,4}`, `mem=6`, `MaxBypass=2` | 959 | 262 | 2.898 | PASS |
| `CacheSchedulerCancelRestore` | `{a,b}`, `{p1,p2}`, `req={4}`, `AsyncRestore=TRUE` | 211,309 | 36,172 | 27.000 | PASS |
| `CancelInStep` | `Bug="cancel_in_step"` | 337 | 106 | 1.421 | `StepBlocksInMemoryAndOwned` violated |
| `CancelInRestore` | `AsyncRestore=TRUE, Bug="cancel_in_restore"` | 4,762 | 1,385 | 3.000 | `StepBlocksInMemoryAndOwned` violated |
| `RestoreSplitBudget` | `Bug="restore_split"` | 311,733 | 76,639 | 64.868 | `LiveRequestResolves` violated |
| `BoundaryWindowDropped` | `Bug="boundary_window_dropped"` | 1,979 | 653 | 2.265 | `LaneHistoryCovered` violated |
| `UnprunedWindow` | `Bug="unpruned_window"` | 1,060 | 447 | 2.700 | `BudgetsAndReservations` violated |
| `DecodeStalledByPrefill` | `Bug="no_decode_priority"` | 16,652 | 5,500 | 3.569 | `ActionDecodePriority` violated |
| `QueuedBypassedByStep` | `Bug="queued_bypass"` | 1,158 | 486 | 3.007 | `BypassBoundHolds` violated |
| `DeferredTied429` | `Bug="deferred_tied_429"` | 185 | 111 | 1.435 | `ActionNoQueuedRefusal` violated |
| `EagerHotWriteback` | `Bug="eager_hot_writeback"` | 1,140 | 470 | 2.082 | `ActionLazyColdWriteback` violated |
| `StartStateDropped` | `Bug="start_state_dropped"` | 587 | 244 | 2.612 | `StartPointsKept` violated |
| `DecodeBlockSealed` | `Bug="seal_decode"` | 784 | 282 | 1.946 | `NoDecodeBlocksSealed` violated |
| `WritebackClobbered` | `Bug="clobber_writeback"` | 25,432 | 6,635 | 5.307 | `ActionWritebackProtected` violated |
| `SpillDisabled` | `Bug="spill_disabled"` | 1,556 | 484 | 2.155 | `ColdBlocksSpillable` violated |
| `SealedShortLength` | `Bug="sealed_short_length"` | 199 | 95 | 1.336 | `SealedValid` violated |
| `DirtyPrivTail` | `Bug="dirty_priv_tail"` | 61 | 33 | 1.210 | `UnsealedPrivateInvisibleZero` violated |
| `StaleRefCount` | `Bug="stale_refcount"` | 197 | 95 | 1.347 | `RefCountMatchesReferrers` violated |
| `StepOverBudget` | `Bug="step_over_budget"` | 274 | 116 | 1.495 | `StepBudgetHolds` violated |
| `QueueOverflow` | `Bug="queue_overflow"` | 39 | 38 | 1.067 | `QueueBounded` violated |
| `SealedMutated` | `Bug="mutate_sealed"` | 311 | 135 | 1.649 | `ActionSealedImmutable` violated |
