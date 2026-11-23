# inky roadmap

inky serves Inkling 2 Mini to many coding-agent tasks at once from one process on one Apple silicon machine. The design is in [DESIGN.md](DESIGN.md). The work is built bottom up, one layer at a time: a layer's gates are green before the next layer starts, and nothing in "After the first build" starts before layer 9.

<!-- The tables between the nova-work markers are generated from roadmap.sexp by nova-roadmap: edit the data and run `nova-roadmap render`. Everything else on this page is written by hand. -->

## Evaluation

What was measured and tried before any design, and the design itself.

<!-- nova-work:evaluation:start -->

| feature | done |
|---|:---:|
| Single-stream speed and memory bandwidth measured on an M5 Max | ✅ |
| Separate processes measured as a baseline: 1.35x at best | ✅ |
| Grouping lanes by predicted experts simulated on the routing dumps: about 20% fewer weight bytes | ✅ |
| One real coding-agent card lands through a one-lane OpenAI endpoint | ✅ |
| The batched path calibrated: 58 ms + 0.21 ms per token | ✅ |
| llama.cpp's parallel slots measured as a baseline: flat near 200 tok/s | ✅ |
| The lanes design written | ✅ |
| The batched path shown batch-invariant (inky binv); a one-token batch bug found and fixed | ✅ |
| The design cold-read by a second reader, findings folded in | ❌ |
| complete | 88% |

[Source data](roadmap.sexp)

<!-- nova-work:evaluation:end -->

## Design revision

The cold read's findings, folded into the design before layer 2 starts. Layer 1 does not depend on them.

<!-- nova-work:revision:start -->

| feature | done |
|---|:---:|
| The lane table corrected: the cache is read at about 210 GB/s and a lane reads all of it | ✅ |
| The four invariance and attention claims from the cold read measured | ❌ |
| Blocks: a prefix keeps the window blocks and recurrent state that make it a start point | ❌ |
| Blocks: recurrent state kept at lane heads, turn ends and a sparse stride, not at every boundary | ❌ |
| Blocks: 64-bit or per-block cache addressing | ❌ |
| Admission: the worst-case memory bound sets the lane limit | ❌ |
| Scheduler: cancel during a step waits for the step | ❌ |
| Parking: lazy under memory pressure, atomic reservation to restore, a bound for parked and queued lanes | ❌ |
| Models: one coupled cache and scheduler model for the cross-cutting properties | ❌ |
| Server: earlier turns' thinking tokens round-trip without a content-keyed map | ❌ |
| Layers 4 and 5 merged; a performance and memory gate at 20K and 42K context after layer 2 | ❌ |
| Every layer's gate strengthened as the cold read lists | ❌ |
| An attention kernel for many lanes named as the first kernel to write | ❌ |
| complete | 7% |

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

Many lanes advance in one step, and each expert is read once for all of them.

<!-- nova-work:layer-2:start -->

| feature | done |
|---|:---:|
| Profile of where the batched path's 58 ms fixed cost goes | ✅ |
| One kind of step: it advances a set of (lane, tokens) spans, so decode and prefill are the same operation | ❌ |
| Each lane's attention reads only its own cache | ❌ |
| Per-lane sampling with one step in flight | ❌ |
| Gate: a lane's logits are bit-identical alone and in any batch | ❌ |
| Gate: batched logits stay within today's tolerance of the llama.cpp reference | ❌ |
| complete | 16% |

[Source data](roadmap.sexp)

<!-- nova-work:layer-2:end -->

## Layer 3: block cache in memory

The cache becomes a prefix tree of append-only blocks, so identical prefixes are stored and prefilled once.

<!-- nova-work:layer-3:start -->

| feature | done |
|---|:---:|
| BlockCache.tla: references, sealing, tiers and budgets, checked with TLC | ❌ |
| BlockLayout.tla: slot reuse, window sliding and table indexing at capacity 3 or 4, with reversed witnesses | ❌ |
| Append-only blocks of 64 positions, window and full kinds, with recurrent state at each boundary | ❌ |
| Attention reads keys through a per-lane block table | ❌ |
| A request matches a prefix tree of sealed blocks; shared prefixes are shared by reference count | ❌ |
| Window blocks are released as a lane moves past them | ❌ |
| Gate: the engine's traces are behaviours of both models, checked in CI | ❌ |
| Gate: outputs identical to layer 2 | ❌ |
| complete | 0% |

[Source data](roadmap.sexp)

<!-- nova-work:layer-3:end -->

## Layer 4: scheduler and admission

Which lanes share each step, and which requests are let in.

<!-- nova-work:layer-4:start -->

| feature | done |
|---|:---:|
| LaneScheduler.tla: lane states, the step in flight and wait counters, checked with TLC | ❌ |
| Lane lifecycle, with cancel reachable from every state | ❌ |
| Step building: the longest-waiting lane first, then lanes adding the fewest new experts | ❌ |
| Prefill spans fill the rest of each step's token budget | ❌ |
| No ready lane is passed over more than W steps in a row | ❌ |
| Admission by worst-case memory, a bounded queue, then 429 with Retry-After | ❌ |
| Gate: scheduler traces are behaviours of the model, checked in CI | ❌ |
| Gate: no lane starves in a soak | ❌ |
| complete | 0% |

[Source data](roadmap.sexp)

<!-- nova-work:layer-4:end -->

## Layer 5: disk tier and parking

Lanes that are waiting leave memory.

<!-- nova-work:layer-5:start -->

| feature | done |
|---|:---:|
| A waiting lane's private blocks are written to the SSD and dropped from memory | ❌ |
| Blocks are read back before a lane's next step | ❌ |
| Unreferenced blocks are dropped least-recently-used, memory first, then disk | ❌ |
| Gate: outputs are identical after park and restore | ❌ |
| Gate: kill -9 loses cache and never correctness | ❌ |
| complete | 0% |

[Source data](roadmap.sexp)

<!-- nova-work:layer-5:end -->

## Layer 6: server

The endpoint a coding-agent harness talks to.

<!-- nova-work:layer-6:start -->

| feature | done |
|---|:---:|
| OpenAI-compatible chat completions, streaming and not, over the lanes | ❌ |
| The chat template rendered exactly, tested against the GGUF's own template | ❌ |
| Tool declarations, tool calls and tool results translated both ways | ❌ |
| A resent assistant message maps back to the tokens that were generated | ❌ |
| A health endpoint, and binding to loopback or a private network only | ❌ |
| Gate: the tracer card and a recorded replay set land through it | ❌ |
| complete | 0% |

[Source data](roadmap.sexp)

<!-- nova-work:layer-6:end -->

## Layer 7: the lane-count sweep

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

## Layer 8: fleet wiring

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

## Layer 9: soak

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
| A kernel for decode-size batches that reads each touched expert once at full bandwidth: up to 2x at 16 to 32 lanes (later) | ❌ |
| A shared prefix is read once per step for all lanes' queries (later) | ❌ |
| A quality replay: landed cards run again and compared with a reference route (later) | ❌ |
| The lowest thinking effort that still lands cards, per kind of card (later) | ❌ |
| The 8-bit cache, behind the quality replay (later) | ❌ |
| Far-key pruning on the full-attention passes, behind the quality replay (later) | ❌ |
| Smaller weight quantizations, behind the quality replay (later) | ❌ |
| Tool calls constrained to valid JSON while sampling (later) | ❌ |
| Sampling on the GPU (later) | ❌ |
| General fixes sent upstream to inkling (later) | ❌ |
| complete | 0% |

[Source data](roadmap.sexp)

<!-- nova-work:later:end -->
