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

`InBlock` on mini-m5, 2026-11-24T03:44Z, by `run_tlc.zsh`: TLC 2.19 (revision `5a47802`), Eclipse Adoptium 21.0.12.1, one worker, a 4 GiB maximum heap (the log reports 3641MB). Wall seconds include JVM startup. The machine clock on the log lines is 2026-11-23 22:44.

| Run | Generated states | Distinct states | Wall seconds | Result |
| --- | ---: | ---: | ---: | --- |
| InBlock | 2,257 | 48 | 0.834 | PASS |
| InBlockKv8 | 2,257 | 48 | 0.823 | PASS |
| InBlockMajor | 2 | 2 | 0.656 | Ordered violated |
| InBlockMajorCollide | — | — | 0.660 | NoCollision violated by the initial state |
| InBlockTrace | 49 | 48 | 0.709 | PASS |
| InBlockTraceKv8 | 49 | 48 | 0.673 | PASS |

InBlockMajor state 2 is `p = 1`, `k = 0`, `h = 0`, `i = 0`, `addr = 0`. The position-major step is `index * (stride / B)`, which is 0 at position 0, so the pass change does not move the address. `Ordered` compares that address to `Formula`, which the action does not call. `Inside` holds on that state. `NoCollision` is checked on its own config and fails while TLC is still computing the initial state, because every index-0 coordinate stores address 0. That log has one `Error:` line and no state count. The earlier head-term witness is retired.

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

As a separate negative-control check, the redundant-load mutant passed both
output invariants across all 2,887 distinct states when `StagedOnce` was
omitted. A test copy with a three-key split and a two-key tile was rejected by
the divisibility assumption before exploring any states.

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
