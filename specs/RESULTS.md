# TLC results

Run on Linux bench `amd-1` (`hostname: b1`), 2026-11-23, with TLC 2.19
(revision `5a47802`), Eclipse Temurin Java 21.0.12.1, one TLC worker, 4 GiB heap
and Parallel GC. Times below are wall time including JVM startup, measured by
`run_tlc.py`. The healthy state spaces were exhausted; negative runs stopped at
the first specified invariant violation. Counts for negative runs can vary with
TLC traversal order.

| Run | Generated states | Distinct states | Wall seconds | Result |
| --- | ---: | ---: | ---: | --- |
| BlockLayout3 | 13,049 | 6,668 | 1.298 | PASS |
| BlockLayout4 | 144,089 | 64,418 | 4.603 | PASS |
| StaleEntry | 985 | 703 | 0.888 | GenerationMatches violated |
| WindowOffByOne | 18,393 | 10,576 | 1.641 | WindowCovered violated |
| SpanTooSmall | 124 | 85 | 0.711 | SpanPinned violated |
| StartReleased | 2,284 | 1,629 | 1.013 | StartRetained violated |
| SpanPins512 | 66 | 65 | 0.731 | PASS |
| SpanPins17 | 2 | 2 | 0.652 | Covered violated |

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
