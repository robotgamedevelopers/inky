# TLC and trace-conformance results

Measured on a 64-core `x86_64` Linux bench (Eclipse Adoptium OpenJDK
`21.0.12.1`, pinned TLA+ `v1.8.0` `tla2tools.jar` SHA-256
`fdffef247f563ef6edbfd44958d2a82db79e7564e07cb673262d896353fa6713`, TLC `2026.11.19`
rev `bc1b506`), 2026-11-25 UTC. Wall seconds include JVM startup. Positive runs
exhausted their reachable state graphs (`0 states left on queue`, `No error has been found.`);
reversed witnesses stopped on their exact expected invariant, action-property, or
temporal violation.

## CI tier placement (`scripts/check-tla-tiers.sh`)

`scripts/check-tla-tiers.sh` and `scripts/check-tla-tiers.sh --self-test` verify that
every `.cfg` file, runner entry, trace script, and trace mutation sits in **exactly one**
CI tier:

- **Total**: `137` checks and mutants (`105` per pull request in `.github/workflows/ci.yml`,
  `32` nightly in `.github/workflows/nightly.yml`), `0` gaps, `0` duplicates.
- **Self-test**: all `7` planted faults (`PoisonGate4` dropped, `StaleSaved` dropped,
  `skip-keep` mutant dropped, `memmode` part unscheduled, `memmode` scheduled in both
  tiers, unwired `check-planted-traces.sh`, and `traces` dropped from `ci.yml`) failed
  red with their expected `TIER LOST` / `TIER GAP` / `TIER DUP` diagnostics.

## `memmode` PR part (`scripts/check-tla.sh memmode`)

10 runs in parallel (`-Xmx2g`, 1 worker per JVM):

| Run | Expected result | Verdict |
| --- | --- | --- |
| `MemMode` | Exhaustive safety + `EventuallyRestored` | PASS |
| `RestoreDefault` | `Invariant RestoredWhenIdle is violated.` | Expected violation |
| `ReleaseEarly` | `Invariant RaisedWhileConnected is violated.` | Expected violation |
| `NoPersist` | `Invariant RestoredWhenIdle is violated.` | Expected violation |
| `LostSeed` | `Invariant SavedIsOwner is violated.` | Expected violation |
| `StaleSaved` | `Invariant RestoredWhenIdle is violated.` | Expected violation |
| `StaleSavedOwed` | `Invariant SavedIsOwed is violated.` | Expected violation |
| `NoRollback` | `Invariant RefusedRollbackSafe is violated.` | Expected violation |
| `NoRollbackAttempted` | `Invariant RollbackAttempted is violated.` | Expected violation |
| `UnfairRestart` | `Temporal property EventuallyRestored was violated.` | Expected temporal violation |

## `blocks` PR part (`scripts/check-tla.sh blocks`)

66 runs via `specs/run_tlc.zsh` (`16` single-worker JVMs at a time, `48.8 s` total wall):

| Run | Generated states | Distinct states | Wall seconds | Result |
| --- | ---: | ---: | ---: | --- |
| `PrefixTreeFast` | 232,383 | 10,588 | 48.203 | PASS |
| `PrefixTreeEvictFast` | 295,009 | 12,916 | 48.452 | PASS |
| `PrefixTreeShallow` | 6,370 | 3,944 | 7.447 | `DeepestStart` violated |
| `PrefixTreeStale` | 251,406 | 54,986 | 38.644 | `ResultLive` violated |
| `PrefixTreeFirstToken` | 3,787 | 2,464 | 3.143 | `MatchExact` violated |
| `PrefixTreeAnyParent` | 126 | 121 | 2.152 | `ResultIsStart` violated |
| `PrefixTreePartial` | 28 | 28 | 2.290 | `StartOnBoundary` violated |
| `PrefixTreeLeak` | 152,792 | 39,822 | 30.696 | `RefsSettledAfterSeek` violated |
| `PrefixTreeReuseHeld` | 3,874 | 2,459 | 3.182 | `HeldNeverStale` violated |
| `PrefixTreeStaleInsert` | 26 | 26 | 1.801 | `NoBadInsert` violated |
| `PrefixTreeTrace` | 67 | 62 | 3.163 | PASS |
| `BlockLayout3` | 11,106 | 5,788 | 13.754 | PASS |
| `BlockLayout4` | 33,876 | 16,544 | 23.042 | PASS |
| `BlockLayoutHigh` | 33,876 | 16,544 | 21.255 | PASS |
| `BlockLayoutKv8` | 33,876 | 16,544 | 23.299 | PASS |
| `StaleEntry` | 741 | 426 | 4.294 | `GenerationMatches` violated |
| `WindowOffByOne` | 605 | 347 | 4.922 | `WindowCovered` violated |
| `FullDropped` | 4,955 | 2,552 | 12.565 | `FullCovered` violated |
| `SpanTooSmall` | 317 | 221 | 7.995 | `SpanPinned` violated |
| `StartReleased` | 1,862 | 1,097 | 7.748 | `StartRetained` violated |
| `TailNotZeroed` | 741 | 426 | 9.430 | `TailPrivateZero` violated |
| `WrongTableIndex` | 510 | 293 | 7.767 | `TableIndex` violated |
| `Truncate32` | 8 | 8 | 8.339 | `Address64` violated |
| `SlotOverlap` | 95 | 57 | 4.808 | `Address64` violated |
| `SlabWord` | 8,330 | 570 | 4.939 | PASS |
| `SlabWordKv8` | 8,330 | 570 | 4.982 | PASS |
| `SlabWordUnmasked` | 13 | 13 | 5.469 | `ShaderPtrResolves` violated |
| `SlabWordWrongBind` | 14 | 14 | 4.647 | `ShaderPtrResolves` violated |
| `SlabWordOverflow` | 6 | 6 | 3.702 | `NoOffsetClobbersSlab` violated |
| `SlabWordFifthSlab` | 81 | 23 | 4.384 | `SlabCountBounded` violated |
| `BlockedSlab3` | 42,319 | 10,431 | 15.115 | PASS |
| `BlockedSlab4` | 41,333 | 10,416 | 13.770 | PASS |
| `BlockedSlabFlatGrow` | 63 | 38 | 3.793 | `SyncedMatchesCpu` violated |
| `BlockedSlabNoReuseZero` | 196 | 103 | 4.747 | `CleanSlotZeroed` violated |
| `BlockedSlabMissedDirty` | 69 | 29 | 4.224 | `SyncedMatchesCpu` violated |
| `VecAlloc10` | 110 | 109 | 4.279 | PASS |
| `VecAlloc7` | 110 | 109 | 3.540 | PASS |
| `VecAlloc6` | 63 | 63 | 3.521 | `NoPoolExhaustion` violated |
| `VecAllocUnpinnedWeak` | 51 | 51 | 3.193 | `NeverOverwritesLive` violated |
| `VerifyRollback` | 10,915 | 3,819 | 3.812 | PASS |
| `VerifyOffByOneCs` | 44 | 44 | 1.842 | `ConvMatchesSequential` violated |
| `VerifyKeepRejectedKv` | 44 | 44 | 1.800 | `RejectedTailRestored` violated |
| `VerifySealRejected` | 4 | 4 | 1.751 | `NoRejectedSeal` violated |
| `KvDiskTier` | 82 | 43 | 1.674 | PASS |
| `KvDiskNoOverlap` | 3 | 3 | 1.516 | `SplitBoundaryAligned` violated |
| `KvDiskStagingRace` | 24 | 18 | 1.593 | `StagingHalfSafe` violated |
| `SwaRing` | 96 | 30 | 1.452 | PASS |
| `SwaRingShortWriteback` | 3 | 3 | 1.223 | `RingHoldsExactWindow` violated |
| `SpanPins512` | 66 | 65 | 1.382 | PASS |
| `SpanPins17` | 2 | 2 | 1.301 | `Covered` violated |
| `SpanPinsOverBound` | 2 | 2 | 1.203 | `BoundHolds` violated |
| `SpanPinsMax24` | 65 | 65 | 1.546 | `MaximumIs25` violated |
| `StartTails2` | 135,945 | 28,946 | 26.030 | PASS (`SYMMETRY SymLanes`, `MaxPos=6, MaxStarts=2`) |
| `LaneStaleGen` | 148 | 64 | 1.860 | `LaneCovered` violated |
| `UnheldWindow` | 4,179 | 1,578 | 4.729 | `StartRetained` violated |
| `PrunedUnderStart` | 21,185 | 7,100 | 6.803 | `StartRetained` violated |
| `TailOverread` | 14 | 14 | 1.930 | `ReadsWritten` violated |
| `ReuseNotZeroed` | 3,147 | 1,242 | 4.433 | `TailPrivateZero` violated |
| `DanglingHead` | 724 | 230 | 2.832 | `HeadsConsistent` violated |
| `InBlock` | 49 | 48 | 1.758 | PASS |
| `InBlockKv8` | 49 | 48 | 1.833 | PASS |
| `InBlockMajor` | 2 | 2 | 1.883 | `Ordered` violated |
| `InBlockMajorCollide` | 5 | 5 | 1.511 | `NoCollision` violated |
| `InBlockAddrCol` | 5 | 5 | 1.223 | `NoCollision` violated |
| `InBlockTrace` | 1,489 | 1,488 | 2.031 | PASS |
| `InBlockTraceKv8` | 1,489 | 1,488 | 2.085 | PASS |

## `grouped` PR part (`scripts/check-tla.sh grouped`)

19 runs via `specs/run_tlc.zsh` (`8` single-worker JVMs at a time, `57.5 s` total wall):

| Run | Generated states | Distinct states | Wall seconds | Result |
| --- | ---: | ---: | ---: | --- |
| `GroupedLayout3` | 579 | 485 | 8.420 | PASS |
| `GroupedLayout4` | 1,553 | 1,329 | 7.918 | PASS |
| `LocalMatch` | 1,007 | 981 | 4.952 | `VisitedKeys` violated |
| `DonorWindow` | 361 | 343 | 5.223 | `VisitedKeys` violated |
| `TilePartial` | 357 | 339 | 4.793 | `OrderedPartials` violated |
| `Restage` | 149 | 149 | 3.671 | `StagedOnce` violated |
| `StalledTraversal` | 195 | 195 | 5.362 | `TraversalCompletes` violated |
| `AdvanceUnfair` | 270 | 268 | 6.895 | `TraversalCompletes` violated |
| `RestageOutputs` | 18,231 | 13,611 | 11.972 | PASS (expected negative control) |
| `PoisonGate3` | 964,473 | 173,840 | 22.086 | PASS |
| `PoisonGate4` | 5,246,785 | 800,160 | 52.072 | PASS (`SYMMETRY SymLanes`) |
| `PoisonUnguarded` | 17 | 17 | 6.968 | `NoFalsePoison` violated |
| `PoisonNoSave` | 142 | 81 | 6.182 | `LiveIntact` violated |
| `PoisonOwnedOnly` | 86 | 51 | 5.645 | `Coverage` violated |
| `PoisonNoZero` | 65 | 47 | 3.210 | `NoFalsePoison` violated |
| `ScratchTail3` | 469,184 | 435,360 | 19.048 | PASS |
| `ScratchTail4` | 2,359,830 | 2,190,375 | 44.697 | PASS |
| `ScratchTailOneSpan` | 832 | 728 | 5.246 | PASS (`notail`, one span) |
| `ScratchTailNoZero` | 1,429 | 1,429 | 4.886 | `NoForeignRead` violated |

## Nightly layout and `CacheScheduler` tiers (`TLA_NIGHTLY=1`)

- **`layout-nightly`** (`.github/workflows/nightly.yml` `tla-layout-nightly`):
  - `StartTails3` (`Capacity = 3, B = 2, W = 2, Every = 2, MaxPos = 6, MaxSpan = 1, MaxStarts = 2, SYMMETRY SymLanes`, 1 worker): `6,533,679` generated states, `1,341,035` distinct states, `485.7 s` wall — **PASS**.
  - `PrefixTree` (`MaxGen = 2`) + three `src/prefix.rs` trace mutants (`no-evict`, `full-match`, `ref-leak` on Darwin).
- **`scheduler`** (`.github/workflows/nightly.yml` `tla-scheduler`, `specs/run_cache_scheduler.sh`):
  all `30` runs (`11` exhaustive safety + liveness + action-property configs and `19` reversed witnesses) — **30/30 PASS**. Full per-run table and counterexamples are recorded in [CACHE_SCHEDULER.md](CACHE_SCHEDULER.md).

## `traces` PR part (`scripts/check-tla.sh traces`)

All five trace-conformance gates run in parallel (`rustc` + pinned `tla2tools.jar`, zero network/GPU), and every mutation is pinned to the exact `Error: ...` line TLC must report:

1. **`scripts/check-memmode-traces.sh`** (`src/memd.rs` -> `specs/memmode/TraceCheck.tla`):
   - Production: `32` walks x `192` ops (`6,208` generated, `6,176` distinct states) — **PASS**.
   - Pinned mutations (`5/5` rejected): `early-release` (`StepConforms`), `no-rollback` (`RollbackAttempted`), `no-readback` (`StepConforms`), `lost-seed` (`StepConforms`), `skip-keep` (`SavedIsOwner`).
2. **`scripts/check-blockbook-traces.sh`** (`src/blockbook.rs` -> `specs/StartTailsTrace.tla`):
   - Production: `20` walks x `90` ops (`1,026` generated, `1,006` distinct states; `resumed=19, reused=204, pruned_under_start=15, partial_tail=137, retired=76, finished_tail=29, prompt=24, every=36, pool_full=1,214`) — **PASS**.
   - Pinned mutations (`6/6` rejected): `unheld-window` (`RefsMatch`), `pruned-under-start` (`RefsMatch`), `overread` (`ReadsMatch`), `no-zero` (`TailPrivateZero`), `kept-tail` (`RefsMatch`), `every-off` (`StepConforms`).
3. **`scripts/check-cache-traces.sh`** (`src/blocks.rs` + `src/cache_scheduler.rs` + `src/cache_trace.rs` -> `BlockLayout`, `CacheScheduler`, `SpanPins`):
   - Production (`5/5` bundles **PASS**): `bl_base0` (`395` states, `slide_steps=16`), `bl_high_fp16` (`395` states, `>= 4 GiB`), `bl_high_kv8` (`395` states, `>= 4 GiB`), `cs` (`3,024` states; `admit_bypass=39, step_bypass=39, cancel=39, refuse429=317, spill=68, restore=40, drop=44, slide=76, reclaim=22`), `sp_512` (`128` states, `64` alignments, `max_win=25`).
   - Pinned mutations (`10/10` rejected): `trunc-real-byte-offset` (`Address64`), `off1-byte-offset` (`Address64`), `reissue-gen-reset` (`StepConforms`), `stale-finish` (`StepConforms`), `admit-wait-dropped` (`BypassBoundHolds`), `step-bypass-dropped` (`BypassBoundHolds`), `priv-tail-dirty` (`UnsealedPrivateInvisibleZero`), `noop-restore-block` (`ActionConforms`), `non-lru-writeback` (`ActionLruOrder`), `cap17-span-pins` (`Covered`).
4. **`scripts/check-scheduler-traces.sh`** (`src/scheduler.rs` + `scripts/scheduler-traces.rs` -> `specs/CacheScheduler.tla`):
   - Production: `18` drained, invariant-checked traces (`3,044` states; `cold_writebacks=47, disk_restores=31, refused_429=230, cancel_in_step=103, cancel_in_restore=9, admit_bypasses=19, step_bypasses=78, slide_windows=69, finished=133`) — **PASS**.
   - Pinned mutations (`6/6` rejected): `cancel-in-step-early-free` (`StepBlocksInMemoryAndOwned`), `cancel-in-restore-early-free` (`StepBlocksInMemoryAndOwned`), `bypass-bound-exceeded` (`BypassBoundHolds`), `drop-step-bypass-tick` (`BypassBoundHolds`), `drop-admit-bypass-tick` (`BypassBoundHolds`), `noop-restore-stutter` (`ActionConforms`).
5. **`scripts/check-poisongate-traces.sh`** (`src/poolgate.rs` + `src/blocks.rs` -> `specs/PoisonGateTrace.tla`):
   - Production: `4` traces (`256` states; `poison_step=46, restore=46, write=64, take_idle=6, share_idle=4, release_idle=44, take_poisoned=36, share_poisoned=2, release_poisoned=4, saved_slots=25`) — **PASS**.
   - Pinned mutations (`5/5` rejected): `unguarded-stepping` (`NoFalsePoison`), `no-save-live` (`ActionConforms`), `owned-only-poison` (`Coverage`), `skip-restore-copy` (`LiveIntact`), `empty-stepping-action` (`ActionConforms`).
