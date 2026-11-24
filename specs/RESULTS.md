# TLC results

Run on the designated `mini-m5` bench, 2026-11-23 16:29 UTC, with TLC 2.19
(revision `5a47802`), Eclipse Temurin Java 21.0.12.1, one TLC worker, a 4 GiB maximum heap
and Parallel GC. Times below are wall time including JVM startup, measured by
`run_tlc.zsh`. The healthy state spaces were exhausted; negative runs stopped at
the first specified invariant violation. Counts for negative runs can vary with
TLC traversal order.

| Run | Generated states | Distinct states | Wall seconds | Result |
| --- | ---: | ---: | ---: | --- |
| BlockLayout3 | 13,049 | 6,668 | 1.299 | PASS |
| BlockLayout4 | 144,089 | 64,418 | 5.013 | PASS |
| StaleEntry | 985 | 703 | 1.096 | GenerationMatches violated |
| WindowOffByOne | 18,393 | 10,576 | 1.765 | WindowCovered violated |
| SpanTooSmall | 124 | 85 | 0.916 | SpanPinned violated |
| StartReleased | 2,284 | 1,629 | 1.072 | StartRetained violated |
| SpanPins512 | 66 | 65 | 0.825 | PASS |
| SpanPins17 | 2 | 2 | 0.755 | Covered violated |

Runner negative controls: an empty successful process (`TLC_JAVA=/usr/bin/true`)
and a failed process (`TLC_JAVA=/usr/bin/false`) were both rejected.

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
production traces, and all 17 `run_tlc.zsh` runs of that date) took 49.5 seconds on hetzner1; the list now has 22 runs.

## Fixed-split partials arena

On mini-m5, 2026-11-23, `specs/run_tlc.zsh` with one TLC worker. The healthy
runs exhaust the graph and check `Finishes`. The SplitArena6 row was run later on hetzner1, 2026-11-24, in the full `run_tlc.zsh` list (its neighbours there: SplitArena3 1.427 s, SplitArena4 1.931 s). Times include JVM startup.

| Run | Generated states | Distinct states | Wall seconds | Result |
| --- | ---: | ---: | ---: | --- |
| SplitArena3 | 1,006 | 766 | 0.952 | PASS |
| SplitArena4 | 1,046 | 806 | 0.977 | PASS |
| SplitArena6 (hetzner1) | 1,046 | 806 | 1.846 | PASS |
| SplitEdgeDropped | 375 | 375 | 0.713 | CombineReadsOwn violated |
| SplitRowsUnbounded | 251 | 251 | 0.681 | WritesInArena violated |

Inspected counterexamples:

- **SplitEdgeDropped, state 2:** window layer, span `p0=0, t=5`, rows from 4.
  The row ends at 5 and its window starts at 1, so it needs splits 0, 1 and 2;
  two are reserved. The arena holds `0 :> (4,0,0)`, `1 :> (4,0,1)`,
  `2 :> (4,1,0)`, and head 0's third read is slot 2.
- **SplitRowsUnbounded, state 2:** full layer, span `p0=0, t=5`, `nsplit=2`,
  four rows in one batch. Row 3, head 1 writes slot 14; the arena ends at 8.

`SplitArena6` (arena floor 6, so two rows per batch where a window span has three
reserved splits) reaches the case floors 3 and 4 never do: a batch of two or more
rows on a window layer whose span passes the window. Checked by a scratch invariant
that forbids that case: it holds at floor 4 (no error) and is violated by the
initial state at floor 6. Its state count equals floor 4's by coincidence of the
reachable batch starts; the case is the point, not the count. The PR-tier TLA step
(`scripts/check-tla.sh`, all runners but CacheScheduler) took 54.5 seconds on
hetzner1, inside the two-minute budget.

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
