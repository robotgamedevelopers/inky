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

As a separate negative-control check, the redundant-load mutant passed both
output invariants across all 2,887 distinct states when `StagedOnce` was
omitted. A test copy with a three-key split and a two-key tile was rejected by
the divisibility assumption before exploring any states.

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
