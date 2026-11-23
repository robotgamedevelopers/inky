# inky roadmap

inky serves Inkling 2 Mini to many coding-agent tasks at once from one process on one Apple silicon machine. The design is in [DESIGN.md](DESIGN.md). The work is built bottom up, one layer at a time: a layer's gates are green before the next layer starts, and nothing in "After the first build" starts before layer 8. At Glenn's direction (2026-11-23) the server (layer 5) was built ahead of the block cache and scheduler, so inky can serve as a provider now; layers 3 and 4 follow underneath it.

<!-- The tables between the nova-work markers are generated from roadmap.sexp by nova-roadmap: edit the data and run `nova-roadmap render`. Everything else on this page is written by hand. -->

## Evaluation

What was measured and tried before any design, and the design itself.

<!-- nova-work:evaluation:start -->

| feature | done |
|---|:---:|
| Four real cards at once through inky serve, each landing the merged fix | ✅ |
| Max-context eval: one stream and memory-bounded lanes from 16K to 256K on real code | ✅ |
| Single-stream speed and memory bandwidth measured on an M5 Max | ✅ |
| Separate processes measured as a baseline: 1.35x at best | ✅ |
| Grouping lanes by predicted experts simulated on the routing dumps: about 20% fewer weight bytes | ✅ |
| One real coding-agent card lands through a one-lane OpenAI endpoint | ✅ |
| The batched path calibrated: 58 ms + 0.21 ms per token | ✅ |
| llama.cpp's parallel slots measured as a baseline: flat near 200 tok/s | ✅ |
| The lanes design written | ✅ |
| The batched path shown batch-invariant (inky binv); a one-token batch bug found and fixed | ✅ |
| The design cold-read by a second reader, findings folded in | ✅ |
| complete | 100% |

[Source data](roadmap.sexp)

<!-- nova-work:evaluation:end -->

## Design revision

The cold read's findings, folded into the design before layer 2 starts.

<!-- nova-work:revision:start -->

| feature | done |
|---|:---:|
| The lane table corrected: the cache is read at about 210 GB/s and a lane reads all of it | ✅ |
| The four invariance and attention claims from the cold read measured | ✅ |
| Blocks: a prefix keeps the window blocks and recurrent state that make it a start point | ✅ |
| Blocks: recurrent state kept at prompt ends, every 16 blocks and lane heads, not at every boundary | ✅ |
| Blocks: 64-bit cache addressing through the block table | ✅ |
| Admission: the worst-case memory bound sets the lane limit | ✅ |
| Scheduler: cancel during a step waits for the step | ✅ |
| Tiers: lazy spill of unreferenced blocks, an atomic reservation to restore, a bound for every waiting lane | ✅ |
| Models: one coupled CacheScheduler model, exhaustive in TLC | ✅ |
| Server: earlier turns rendered exactly as sent; no content-keyed map | ✅ |
| Layers 4 and 5 merged; a performance and memory gate inside layer 2 | ✅ |
| Every layer's gate strengthened as the cold read lists | ✅ |
| Decode spans use split-K attention per lane; a many-lane attention kernel is the first follow-on | ✅ |
| complete | 100% |

[Source data](roadmap.sexp)

<!-- nova-work:revision:end -->

## Layer 1: lane state out of the engine

A refactor with no change in behaviour: everything that belongs to one stream moves out of the engine.

<!-- nova-work:layer-1:start -->

| feature | done |
|---|:---:|
| Tests and CI on a Mac runner: the llama.cpp reference-logits check runs on every change | ✅ |
| A lane owns its state (cache, convolution rows, loop-controller state, position); the engine owns only the weights | ✅ |
| Gate: one lane's logit hash is identical to the unmodified engine | ✅ |
| complete | 100% |

[Source data](roadmap.sexp)

<!-- nova-work:layer-1:end -->

## Layer 2: batched step over spans

Many lanes advance in one step, and each expert is read once for all of them. The layer ends with a performance and memory gate, before the cache is built on it.

<!-- nova-work:layer-2:start -->

| feature | done |
|---|:---:|
| Profile of where the batched path's 58 ms fixed cost goes | ✅ |
| One kind of step: it advances a set of (lane, tokens) spans through the batched projections, router, experts and controllers | ✅ |
| Attention per span in two arithmetics — prefill rows and decode steps — each batch-invariant, so prefill chunking never changes a token's numbers | ✅ |
| Each lane's convolution state and cache written by its own span only | ✅ |
| Per-lane sampling with one step in flight, seeded by the request | ✅ |
| Gate: bit-identical cache, state and multi-step logits for a lane alone and in batches of 1, 2, 64, 65 and 129, with prime-sized prefill splits | ✅ |
| Gate: agreement with llama.cpp as CI checks it | ✅ |
| Gate: measured step time and GPU memory at 2K, 20K and 42K context for 1 to 64 lanes, against the lane table | ✅ |
| complete | 100% |

[Source data](roadmap.sexp)

<!-- nova-work:layer-2:end -->

## Layer 3: block cache

The cache becomes a prefix tree of append-only blocks, so identical prefixes are stored and prefilled once.

<!-- nova-work:layer-3:start -->

| feature | done |
|---|:---:|
| CacheScheduler.tla: blocks of both kinds, start points, lanes, the step in flight, budgets; exhaustive in TLC, safety and liveness | ❌ |
| BlockLayout.tla: slot reuse, window sliding and 64-bit table indexing at capacity 3 or 4, with reversed witnesses | ❌ |
| Append-only blocks of 64 positions, window and full kinds, in a pool of GPU buffers | ❌ |
| Attention reads keys through a per-lane block table of 64-bit offsets | ❌ |
| A request matches the deepest start point in a prefix tree of sealed blocks | ❌ |
| Start points: state, full blocks and the window blocks before them, kept at prompt ends, every 16 blocks and lane heads | ❌ |
| Unsealed tails private, zero past their length, dropped at the end; pool slots zeroed on reuse | ❌ |
| Gate: the engine's traces are behaviours of both models, checked in CI | ❌ |
| Gate: outputs identical to layer 2 with poisoned pool slots | ❌ |
| complete | 0% |

[Source data](roadmap.sexp)

<!-- nova-work:layer-3:end -->

## Layer 4: scheduler, admission and disk tier

Which lanes share each step, which requests are let in, and which cold blocks leave memory.

<!-- nova-work:layer-4:start -->

| feature | done |
|---|:---:|
| Lane lifecycle: Queued, Admitted, Restoring, Ready, InStep, Finished; Cancelling and Cancelled | ❌ |
| Step building: the longest-waiting lane first, decoding lanes, then prefill spans to the token budget | ❌ |
| No Ready, Restoring or Queued lane is passed over more than W ticks | ❌ |
| Admission by worst-case memory against the measured budget, a bounded queue, then 429 with Retry-After | ❌ |
| Cancel in a step waits for the step's GPU work and disk I/O | ❌ |
| Unreferenced blocks spilled to the SSD lazily under memory pressure, least recently used first | ❌ |
| Restoring reserves the whole working set atomically before reading | ❌ |
| Gate: scheduler traces are behaviours of the model, checked in CI | ❌ |
| Gate: a bimodal workload of 2K and 64K requests under a tight budget; disconnects in mid-step | ❌ |
| Gate: kill -9 in mid-write loses cache and never correctness | ❌ |
| Gate: temperature 1 with fixed seeds gives the same outputs with and without the cache | ❌ |
| complete | 0% |

[Source data](roadmap.sexp)

<!-- nova-work:layer-4:end -->

## Layer 5: server

The endpoint a coding-agent harness talks to.

<!-- nova-work:layer-6:start -->

| feature | done |
|---|:---:|
| OpenAI-compatible chat completions, streaming and not, over the lanes | ✅ |
| The chat template rendered exactly, tested against the GGUF's own template | ✅ |
| Tool declarations, tool calls and tool results translated both ways | ✅ |
| Earlier turns rendered exactly as the client sends them | ✅ |
| A client that disconnects cancels its lane | ✅ |
| A health endpoint, and binding to loopback or a private network only | ✅ |
| Gate: the tracer card and a recorded replay set land through it | ✅ |
| Gate: concurrent cards issuing identical tool calls; disconnects in mid-thinking stop generation | ❌ |
| Gate: memd traces are behaviours of MemMode.tla, checked in CI | ❌ |
| complete | 77% |

[Source data](roadmap.sexp)

<!-- nova-work:layer-6:end -->

## Layer 6: the lane-count sweep

Measure the curve and pick the lane count where adding lanes stops paying.

<!-- nova-work:layer-7:start -->

| feature | done |
|---|:---:|
| Real card traffic recorded as a replay set | ❌ |
| Replay at 8, 16, 32, 48, 64, 96 and 128 lanes: total and per-lane tok/s, memory, cards per hour | ❌ |
| Each candidate machine measured: memory bandwidth, single-stream decode, prefill | ❌ |
| The knee chosen and written down as the default lane count | ❌ |
| complete | 0% |

[Source data](roadmap.sexp)

<!-- nova-work:layer-7:end -->

## Layer 7: fleet wiring

Real work arrives from a fleet of agents.

<!-- nova-work:layer-8:start -->

| feature | done |
|---|:---:|
| A provider entry for an agent fleet: local billing, concurrency equal to the lane count, a priority ceiling | ❌ |
| Harness and tools run on other machines and reach the endpoint over a private network | ❌ |
| Gate: a real low-priority card lands from the fleet | ❌ |
| complete | 0% |

[Source data](roadmap.sexp)

<!-- nova-work:layer-8:end -->

## Layer 8: soak

Run it for days.

<!-- nova-work:layer-9:start -->

| feature | done |
|---|:---:|
| Days under real load with no growth in blocks, memory or disk | ❌ |
| Kills, a full disk and client disconnect storms leave the server correct | ❌ |
| complete | 0% |

[Source data](roadmap.sexp)

<!-- nova-work:layer-9:end -->

## After the first build

Improvements that wait for the layers above. Those that change the model's numbers also wait for the quality replay.

<!-- nova-work:later:start -->

| feature | done |
|---|:---:|
| The M3 Ultra tuning: decode steps on the non-tensor kernels (later) | ❌ |
| Speculative decoding with Inkling's own DSpark drafter, exact verification: more tokens per cache read at long context (later) | ❌ |
| One attention kernel for every span, split over keys at positions fixed by the cache: decode spans near bus speed, bit-identical to prefill (later) | ❌ |
| A kernel for decode-size batches that reads each touched expert once at full bandwidth: up to 2x at 16 to 32 lanes (later) | ❌ |
| A shared prefix is read once per step for all lanes' queries (later) | ❌ |
| A quality replay: landed cards run again and compared with a reference route (later) | ❌ |
| The lowest thinking effort that still lands cards, per kind of card (later) | ❌ |
| The 8-bit cache, behind the quality replay (later) | ❌ |
| Far-key pruning on the full-attention passes, behind the quality replay (later) | ❌ |
| Smaller weight quantizations, behind the quality replay (later) | ❌ |
| Tool calls constrained to valid JSON while sampling (later) | ❌ |
| Sampling on the GPU (later) | ❌ |
| General fixes sent upstream to erisinky (later) | ❌ |
| complete | 0% |

[Source data](roadmap.sexp)

<!-- nova-work:later:end -->
