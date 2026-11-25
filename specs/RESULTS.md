# TLC results

Run on the designated `mini-m5` bench, 2026-11-23 16:29 UTC, with TLC 2.19
(revision `5a47802`), Eclipse Temurin Java 21.0.12.1, one TLC worker, a 4 GiB maximum heap
and Parallel GC. Times below are wall time including JVM startup, measured by
`run_tlc.zsh`. The healthy state spaces were exhausted; negative runs stopped at
the first specified invariant violation. Counts for negative runs can vary with
TLC traversal order.

| Run | Generated states | Distinct states | Wall seconds | Result |
| --- | ---: | ---: | ---: | --- |
| BlockLayout3 | 2,879 | 1,612 | 0.661 | PASS |
| BlockLayout4 | 30,908 | 15,182 | 1.730 | PASS |
| BlockLayoutHigh | 30,908 | 15,182 | 1.709 | PASS |
| BlockLayoutKv8 | 2,879 | 1,612 | 0.551 | PASS |
| StaleEntry | 423 | 248 | 0.444 | GenerationMatches violated |
| WindowOffByOne | 4,955 | 2,552 | 0.721 | WindowCovered violated |
| SpanTooSmall | 116 | 67 | 0.381 | SpanPinned violated |
| StartReleased | 795 | 492 | 0.482 | StartRetained violated |
| Truncate32 | 8 | 8 | 0.368 | Address64 violated |
| SpanPins512 | 66 | 65 | 0.300 | PASS |
| SpanPins17 | 2 | 2 | 0.271 | Covered violated |

Runner negative controls: an empty successful process (`TLC_JAVA=/usr/bin/true`)
and a failed process (`TLC_JAVA=/usr/bin/false`) were both rejected.

`InBlock` on mini-m5, 2026-11-24T05:03Z, by `run_tlc.zsh`: TLC 2.19 (revision `5a47802`), Eclipse Adoptium 21.0.12.1, one worker, a 4 GiB maximum heap (the log reports 3641MB). Wall seconds include JVM startup. The machine clock on the log lines is 2026-11-24 00:03. The action walks the 48 coordinates in order and records each stored address, so the healthy runs are one path of 48 states.

| Run | Generated states | Distinct states | Wall seconds | Result |
| --- | ---: | ---: | ---: | --- |
| InBlock | 49 | 48 | 0.717 | PASS |
| InBlockKv8 | 49 | 48 | 0.712 | PASS |
| InBlockMajor | 2 | 2 | 0.685 | Ordered violated |
| InBlockMajorCollide | 5 | 5 | 0.684 | NoCollision violated |
| InBlockAddrCol | 5 | 5 | 0.658 | NoCollision violated |
| InBlockTrace | 49 | 48 | 0.669 | PASS |
| InBlockTraceKv8 | 49 | 48 | 0.695 | PASS |

InBlockMajor state 2 is `p = 0`, `k = 0`, `h = 0`, `i = 1`, `addr = 48`, `seen = <<0, 48>>`. The position-major step is `index * (stride / B)`, which is 48 there, and `Formula` is 4. `Ordered` compares the stored address to `Formula`, which the action does not call. InBlockMajorCollide state 5 is `p = 0`, `k = 0`, `h = 1`, `i = 0`, `addr = 0`, `seen = <<0, 48, 96, 144, 0>>`. InBlockAddrCol drops the head term and reaches the same coordinate with `seen = <<0, 4, 8, 12, 0>>`. Both are reached states, each log has two `Error:` lines, and `NoCollision` reads `seen`. The trace files were regenerated from `key_row_at` before this run. The earlier head-term witness is retired.

Inspected counterexamples:

- **StaleEntry, state 10:** lane `a` is done with window table entry
  `(slot=1,generation=1)`; lane `b` allocates window slot 1 at generation 2.
- **WindowOffByOne, state 15:** lane `a` is idle at position 3. Logical window
  block 0 has been cleared from its table, although written position 1 lies in it.
- **SpanTooSmall, state 5:** lane `a` starts a two-token span with budget 2;
  the specified conservative bound is 3.
- **StartReleased, state 12:** start point 2 still refers to window slot 1 at
  generation 1; the pool slot has generation 2 and length 0 after lane `a`
  allocates it. The start point's full slot remains generation 1 and length 2.
- **SpanPins17, state 2:** offset 63, pinned blocks `{0,...,16}`, required blocks
  `{0,...,24}`. The healthy run covers all 64 offsets and checks maximum 25.

These runs establish only the bounds and abstractions in [README.md](README.md).
They do not establish engine trace conformance, general integer-overflow safety,
allocator-policy independence, or scheduler liveness.

## Grouped layout

On mini-m5, the CI entry point `scripts/check-tla.sh` ran the grouped traversal
with the pinned v1.8.0 tools jar and one TLC worker. The three-slot run covers
62 initial snapshots and the four-slot run covers 136. Both exhaust the graph
with deadlock checking enabled. Times include JVM startup.

| Run | Generated states | Distinct states | Wall seconds | Result |
| --- | ---: | ---: | ---: | --- |
| GroupedLayout3 | 579 | 485 | 0.425 | PASS |
| GroupedLayout4 | 1,553 | 1,329 | 0.509 | PASS |
| LocalMatch | 1,007 | 981 | 0.392 | VisitedKeys violated |
| DonorWindow | 361 | 343 | 0.336 | VisitedKeys violated |
| TilePartial | 357 | 339 | 0.325 | OrderedPartials violated |
| Restage | 149 | 149 | 0.335 | StagedOnce violated |
| StalledTraversal | 195 | 195 | 0.397 | TraversalCompletes violated |

A test copy with a three-key split and a two-key tile was rejected by the
divisibility assumption before exploring any states.

### Fairness witness and negative control in CI

Run on the `hetzner1` bench (AMD Ryzen 7 PRO 8700GE), 2026-11-24 01:58 UTC, through
`specs/run_tlc.zsh` with the pinned v1.8.0 tools jar (TLC reports
2026.11.19, rev `bc1b506`), OpenJDK 21.0.12.1, one TLC worker. Times include JVM
startup. The whole `run_tlc.zsh` list, all 17 runs, took 27.0 seconds.

| Run | Generated states | Distinct states | Wall seconds | Result |
| --- | ---: | ---: | ---: | --- |
| GroupedLayout3 | 579 | 485 | 1.157 | PASS |
| GroupedLayout4 | 1,553 | 1,329 | 1.676 | PASS |
| StalledTraversal | 195 | 195 | 1.098 | TraversalCompletes violated |
| AdvanceUnfair | 579 | 485 | 1.144 | TraversalCompletes violated |
| RestageOutputs | 18,231 | 13,611 | 1.592 | PASS (expected) |

`StalledTraversal` disables Advance, so the stalled state has no successor and
weak fairness holds trivially. It cannot tell whether the spec has fairness at
all. It stays because it is the witness for a lost enabling condition, which no
fairness can repair.

`AdvanceUnfair` keeps every action and keeps deadlock checking on, so TLC confirms
that a step exists in every reachable state. Weak fairness covers staging only.
Its counterexample, inspected: lengths `<<1, 1>>`, nothing shared, no window, one
tile. State 3 has staged both private slots, so `loaded = Requests` and
`tile = 0 < TileCount = 1`: Advance is enabled. State 4 stutters forever.

Deleting `WF_vars(FairWork)` from `Spec` in a scratch copy made the healthy
`GroupedLayout3` run fail with `TraversalCompletes` violated (485 distinct states,
stuttering at state 4). The healthy runs check the property, so CI catches that
deletion.

`RestageOutputs` is the negative control that was run by hand before. It runs the
redundant-load mutant against `VisitedKeys` and `OrderedPartials` only, at four
slots, and the runner requires a clean exhaustive pass. `Restage` in the same list
shows that the mutant is live. At three slots the same configuration covers
2,887 distinct states, the count of the earlier hand run.

Runner checks, by scratch copies: with `RestageOutputs` set to the `localMatch`
mutant, and with `AdvanceUnfair` set to `Bug = "none"`, `run_tlc.zsh` stopped
with "Unexpected result" for that run.

### Time budget of the CI TLA step

`scripts/check-tla.sh` on hetzner1, 2026-11-24, with every runner: 355 seconds.
The CacheScheduler runner took about 305 of them; its DecodePriority run alone
took 177 seconds (2,210,338 states). On the hosted macOS runner the same step
took 4 min 58 s for pull request #43. That breaks the two-minute budget, so
pull request CI no longer runs the CacheScheduler runner. The nightly workflow
`.github/workflows/nightly.yml` runs everything with `TLA_NIGHTLY=1`. No
configuration, bound or check was changed. The pull request tier (MemMode, the
production traces, and all 17 `run_tlc.zsh` runs) took 49.5 seconds on hetzner1.

## Start points and tails

On hetzner1 (16-core x86_64, Linux), 2026-11-24, with the pinned v1.8.0 tools jar
(SHA-256 `fdffef24…6713`), Eclipse Temurin Java 21.0.12.1, one TLC worker, a
4 GiB maximum heap and Parallel GC. Times are wall time including JVM startup.

| Run | Generated states | Distinct states | Wall seconds | Result |
| --- | ---: | ---: | ---: | --- |
| StartTails2 | 58,565 | 12,907 | 4.5 | PASS |
| UnheldWindow | 2,023 | 780 | 2.2 | StartRetained violated |
| PrunedUnderStart | 9,522 | 3,145 | 2.8 | StartRetained violated |
| TailOverread | 10 | 10 | 0.9 | ReadsWritten violated |
| ReuseNotZeroed | 1,449 | 584 | 1.9 | TailPrivateZero violated |
| StartTails3 (nightly, `TLC_NIGHTLY=1`) | 1,800,308 | 372,626 | 61.2 | PASS |

StartTails3, three slots per kind, takes about a minute on one worker, so the
runner checks it only with `TLC_NIGHTLY=1`, which the nightly workflow's `tla` job
sets; per push, StartTails2 covers two slots per kind.

Inspected counterexamples:

- **UnheldWindow, state 7:** lane 1 (prompt 2) steps to 2 and keeps a prompt
  point holding full and window slot 1 at generation 1, then finishes. The
  point does not hold its window slot, so lane 0 takes window slot 1 at
  generation 2 for its block 0.
- **PrunedUnderStart, state 9:** lane 1 keeps the same prompt point at 2, steps
  to 4 and prunes window block 0, which releases window slot 1 outright
  (`freed = {<<"window", 1>>}`). After lane 1 finishes, lane 0 takes window slot
  1 at generation 2 while the live point holds generation 1.
- **TailOverread, state 3:** lane 0 at position 0 allocates block 0; attention
  would read one position of it, its length is 0.
- **ReuseNotZeroed, state 7:** lane 0 writes position 0 and finishes, dropping
  its tail; the next request takes the same slot at generation 2 with length 0
  and position 0 still 1.

`scripts/check-blockbook-traces.sh` (CI, after the memmode traces): the
unmodified `src/blockbook.rs` records 20 walks, 1,006 distinct replayed states,
reaching every case listed in README.md (19 resumed lanes, 204 reused slots, 15
windows pruned under a live point, 137 partial tail reads, 76 retired points, 29
finished tails, 24 prompt and 36 every points, 1,214 refused allocations), and
TLC finds no error. Each mutation is rejected: `unheld-window`,
`pruned-under-start` and `kept-tail` by `RefsMatch`, `overread` by `ReadsMatch`,
`no-zero` by `TailPrivateZero`, `every-off` by `Conforms`. The seven runs go in
parallel: 32 seconds on a loaded hetzner1. Negative controls
`TLC_JAVA=/usr/bin/true` and `TLC_JAVA=/usr/bin/false` were both rejected.

## Poisoned-slot gate

Run on `mini-m5`, 2026-11-24 02:30 UTC, by `run_tlc.zsh` with the same JVM and TLC.

| Run | Generated states | Distinct states | Wall seconds | Result |
| --- | ---: | ---: | ---: | --- |
| PoisonGate3 | 158,849 | 34,504 | 1.104 | PASS |
| PoisonGate4 | 3,239,729 | 563,728 | 8.197 | PASS |
| PoisonUnguarded | 17 | 17 | 0.677 | NoFalsePoison violated |
| PoisonNoSave | 124 | 79 | 0.673 | LiveIntact violated |
| PoisonOwnedOnly | 74 | 51 | 0.695 | Coverage violated |

Inspected counterexamples, each checked against `poison_plan` by hand:

- **PoisonUnguarded, state 3:** lane `a` takes slot 1 and steps alone; slot 1
  is poisoned in its own step (`poison_plan(.., guard = false)`, the
  `INK_POOLGATE_SABOTAGE` run on the GPU).
- **PoisonNoSave, state 4:** lane `a` owns slot 1, lane `b` steps; slot 1 is
  poisoned, not saved, and still poison after the restore.
- **PoisonOwnedOnly, state 4:** lane `a` takes slot 1 (zeroed) and releases it;
  the next step leaves the free slot 1 unpoisoned.

## Sliding-window scratch tail

Run on `mini-m5`, 2026-11-24 02:45 UTC, by `run_tlc.zsh` with the same JVM and TLC.

| Run | Generated states | Distinct states | Wall seconds | Result |
| --- | ---: | ---: | ---: | --- |
| ScratchTail3 | 469,184 | 435,360 | 3.608 | PASS |
| ScratchTail4 | 2,359,830 | 2,190,375 | 16.883 | PASS |
| ScratchTailOneSpan | 832 | 728 | 0.864 | PASS (`notail`, one span) |
| ScratchTailNoZero | 1,429 | 1,429 | 0.923 | NoForeignRead violated |

Inspected counterexample, checked against `layer_b` by hand:

- **ScratchTailNoZero, state 4:** spans `<<[p0 0, t 2], [p0 0, t 1]>>`; span 1
  writes rows 0 and 1, span 2 writes row 0 and its tile (AK = 2) reads row 1,
  span 1's key. On the GPU (`INK_POOLGATE_SABOTAGE`): lane 0 (t 23, NaN after
  the sabotage) then lane 1 (p0 31, t 1, its tile reaching position 63) in one
  step; lane 1 read lane 0's NaN keys in rows 1 .. 22 and diverged, then lane 2
  the same way a step later.

## BlockLayout and CacheScheduler trace conformance

`scripts/check-cache-traces.sh` (CI, CPU-only via `rustc` + pinned `tla2tools.jar` v1.8.0):
replays 16 seeded walks per configuration (`bl_base0`, `bl_high_fp16` above 4 GiB at `kvb=2`,
`bl_high_kv8` above 4 GiB at `kvb=1`, and `cs` at `MaxQueue=2`) into generated `TraceData.tla` /
`TraceCheck.tla` bundles (`395` distinct states per `BlockLayout` bundle with `max_gen=2`,
start-point steps and dual-slot coverage; `2,445` distinct states for `CacheScheduler` covering
`admit_bypass=91`, `step_bypass=20`, `cancel=18`, `refuse429=331`) and checks all four bundles
plus seven single-line production mutations (`trunc-real-byte-offset`, `off1-byte-offset`,
`reissue-gen-reset`, `stale-finish`, `admit-wait-dropped`, `step-bypass-dropped`,
`priv-tail-dirty`) in parallel, requiring every mutation to fail inside TLC with an invariant or
action counterexample.
