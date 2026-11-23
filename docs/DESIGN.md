# Lanes: design

Status: design, 2026-11-22. Nothing here is built. The work is tracked layer by layer in [ROADMAP.md](ROADMAP.md).

One process on one Apple silicon machine serves Inkling 2 Mini to many low-priority coding-agent tasks (cards) at once. It starts from the erisinky engine (GPL-3.0-or-later). This repository is the fork: `main` is upstream with its licence, and the branch `tracer-serve` holds the one-lane endpoint the tracer card ran on. The aim is the most cards landed per hour, not the fastest single card.

## What is measured (M5 Max, 128 GB, 2026-11-22)

| | |
|---|---|
| Weight reads | 565 GB/s |
| One stream, decode | 106 tok/s short context, 96 at 2.4K, about 60 at 40K |
| Prefill | 4531 tok/s at 2.4K tokens |
| N separate processes | 1.35x at best (4 copies), collapse at 16 |
| The engine's batched path, T tokens in one pass | 34: 64 ms. 82: 78 ms. 145: 95 ms. 389: 143 ms. 1024: 264 ms |
| One real card (a two-line fix from a private repository, replayed through the opencode harness) | landed, the same fix as the merged change |

The card took 62 turns and 7 min 38 s: 345 s generating (20,118 tokens), 63 s prefill, about 50 s in tools. The context reached 64K. Six turns missed the cache and prefilled again from the start, which was 60 of the 63 prefill seconds. The harness prefix (system prompt and tool declarations) is about 9,300 tokens and is the same for every card.

The batched path fits **58 ms + 0.21 ms per token**. A decode step for B lanes is that pass with one token per lane, plus each lane's own cache read. A lane reads its whole cache for every token: 160 MiB of window keys once it is past 1024 tokens, plus 32 KiB per token of context. Single-stream decode slows from 9.4 ms to 13.3 ms between a short context and 20K, so today's attention reads the cache at about 210 GB/s, not at the bus's 565. The calibrated step time is

    step(B) = 58 ms + B x (0.21 ms + lane cache bytes / 210 GB/s)

| active lanes | total tok/s at 2K context | at 20K | at 42K |
|---:|---:|---:|---:|
| 1 (today's decode path) | 96 | 75 | 58 |
| 16 | 200 | 130 | 90 |
| 32 | 320 | 170 | 107 |
| 64 | 450 | 200 | 119 |
| 128 | 560 | 220 | 125 |

The tracer card ran at about 42K on average. So lanes pay well at short context and little at long context: the per-lane cache read is the limit, not the weights. A first version of this table assumed the cache was read at 565 GB/s and left the shared prefix out of each lane's reads; the cold read caught both. The table still leaves out the output layer and sampling for B tokens, which the calibration ran for one.

What moves the long-context numbers, in order: reading a shared prefix once per step, an attention kernel for many lanes that reads the cache near bus speed, the 8-bit cache, far-key pruning, and keeping card contexts short. With the first two, 64 lanes at 42K come to about 290 tok/s; with the 8-bit cache as well, about 450.

### Baseline: llama.cpp's own batching

llama-server (master 44d5087, Metal, 64 parallel slots, short prompts, 256 tokens per stream) on the same machine:

| streams | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| total tok/s | 41 | 62 | 100 | 142 | 155 | 194 | 215 |
| per stream | 41 | 34 | 27 | 19 | 10 | 6.4 | 3.6 |

It flattens near 200 tok/s, about twice one inky stream. inky's batched path, at 58 ms + 0.21 ms per token, would give about 240, 430 and 710 tok/s at 16, 32 and 64 lanes at the same short context. That is an estimate from timing one sequence's tokens, not a built decode step. So llama.cpp's batching is not close enough to build on; the engine work is needed.

### Where the batched path's time goes

`inky prof_pf` at 32 tokens in one batch (52 ms):

| part | time | note |
|---|---:|---|
| Experts (up and down projections) | 31 ms | about 46 of 128 experts per layer are touched; they are read at about 330 GB/s, 58% of the memory bus, in 16-token tiles that are 35% full |
| Dense projections, router, loop controllers | 16 ms | about 1.3 GB of weights, which the single-token decode path reads in 2 to 3 ms |
| Everything else | 5 ms | mixes, attention, gathers |

So the 58 ms is not a fixed overhead. It is the experts a batch touches, read by a path tuned for large batches. At full bandwidth the same 32 tokens would take about 25 ms. From 32 to 128 tokens the expert time grows only from 31 to 41 ms.

### Batch invariance

`inky binv` prefills the same tokens split at different points and hashes the final logits. A split moves every later token to a different batch, tile slot and set of neighbours. On two prompts (2403 and 61 tokens), every split gives bit-identical logits, at chunk sizes from 64 to 4096. The check found one bug: a batch of exactly one token went to the decode kernel with batch-layout arguments and gave wrong, non-repeatable logits (fixed in `mix_t`). Upstream never sends a one-token batch; a lane decoding alone would.

### What CI checks

CI checks agreement with llama.cpp on a short prompt's decode path and a long prompt's prefill path. It does not check the long prompt's decode path position by position: near the end of that prompt some positions are ill-conditioned in the model itself, and llama.cpp disagrees with its own batched run on 16 of the last 64 (inky decoding token by token agrees with llama.cpp's batched run on 47, and with its token-by-token run on 50).

## The cold read, and what it changed

A second reader read this design and the code cold on 2026-11-22 (verdict: revise before layer 2). The sections below include its accepted findings. Its claims about the engine, measured on 2026-11-23:

- **The batched path is repeatable and batch-invariant.** 16 prefills in a row give identical bits, and splits at prime offsets and on both sides of 64, 512, 1024 and 2048 (`inky binv`, which takes several boundaries per split) give identical logits. The atomic packing of tokens by expert has not shown up in results.
- **One-token batches** were a real bug, fixed before layer 1 (`mix_t`).
- **Masked rows reaching the tensor multiply** were real: the window K/V scratch is now zeroed at every prefill (`k_zero`), found by `inky lanes`.
- **The convolution state exists only at the end of a step.** Inside a prefill chunk it is never kept, so a block boundary is a start point only where a step ended there (Blocks, below).
- **Batched attention does not suit decode spans.** At 3K context it takes about 5 ms for a tile of 32 queries, where the decode path's split-K attention takes about 1.3 ms for one token; a lane's single token pays for the whole tile. The decode path's kernel cannot simply be used for decode spans: it gives different bits from the prefill path, which would make the cache change answers (Decisions, 2). The fix is one attention kernel for all spans (Follow-ons).
- **The cache is read at about 210 GB/s and every lane reads all of its own:** the lane table above.

## Decisions

1. **One process, one copy of the weights.** Separate processes cannot share weight reads.
2. **One kind of step, one computation per token.** A step advances a set of spans, each (lane, tokens). A decoding lane brings one token, a prefilling lane up to a chunk. The projections, router, experts and controllers run once over every span's tokens; attention and the convolutions run per span. Every span, whatever its length, goes through the same kernels, so a token's numbers depend neither on what else shares its step nor on how its own lane's work was chunked (`inky spans`, `inky binv`). That second property is what decision 4 rests on: a cached token was computed in one chunking and a cold run computes it in another. So the single-stream decode path, whose fused attention gives different bits, is never used for a lane; a faster attention for decode spans must give the same bits as the prefill path (Follow-ons).
3. **The server is stateless; the cache is a tree of blocks.** A request is a token list, matched against a prefix tree of sealed cache blocks. A lane lives for one request: it starts at the deepest start point the tree holds for its tokens, and when the request ends its blocks stay in the tree. There are no sessions. A shared prefix, a resumed conversation, a harness side request and a rollback are all the same case.
4. **The cache never changes an answer.** A request's output is a function of the request and its seed; the cache only changes how fast it comes. This rests on batch invariance and on start points being exact. The seed is the request's own (its `seed`, or a hash of the request), never arrival order.
5. **Earlier turns are rendered exactly as the client sends them.** A client that drops the reasoning of earlier turns gets them rendered without it, as the chat template does. The tokens a lane generated are therefore reused only up to where its turn began; the turn itself is prefilled again with the next request (its visible text and tool calls: short). No map from visible text back to generated tokens exists.
6. **Exact by default.** fp16 cache, no pruning. The 8-bit cache and far-key pruning are flags that go on only if the quality replay passes.
7. **One step in flight.** The CPU picks every step's lanes. This costs about 3% against GPU-chained decode and removes the hardest ordering cases.
8. **The Mac only generates tokens.** Harness and tools run on other machines and reach it over a private network.

## Blocks

A block holds the cache for 64 consecutive positions. Two kinds, because the model has two kinds of attention:

- **window block**: the 40 sliding-window layer passes, 10 MiB.
- **full block**: the 8 full-attention layer passes, 2 MiB.

A block is written by exactly one lane, then sealed, then never written again. Sharing is a reference count; because blocks are append-only there is no copy-on-write. The cache for position p depends only on tokens 0..p, so two requests with the same first N tokens share the same blocks exactly.

**Start points.** The recurrent state (convolution rows, 3.66 MiB) exists only at the end of a step. A position k is a start point when the tree keeps all three of: the state at k, the full blocks before k, and the window blocks covering the 1024 positions before k. A start point references those window blocks, so they are not released while it is kept. Start points are kept at the end of every request's prompt, at every 16th block boundary a prefill crosses, and at a running lane's head. Prefill chunks end on block boundaries, so a chunk's end can be a start point. A request starts at the deepest start point its tokens match and prefills from there: at most 1023 positions more than the deepest block it matches.

**Unsealed tails.** A lane's last, partly written block is private, zero-filled past its length on allocation, never matched, and dropped when the lane ends. Pool slots are zero-filled on reuse, so a masked read never meets a non-finite value.

**A span pins** the window blocks its attention reads: up to ceil((1023 + C) / 64) + 1 for a span of C tokens, and every full block of its lane.

**Addressing.** Blocks live in a pool of GPU buffers. Attention reads keys through a per-lane block table whose entries hold 64-bit offsets, so no buffer or offset limits the cache to 8 GiB.

**Tiers.** A block is in memory, on disk, or both. Blocks no lane references (the tree's cold blocks) are written to the SSD lazily, only under memory pressure, least recently used first, and dropped from memory; they are dropped from disk under disk pressure. A request whose blocks were dropped prefills again: slower, never wrong.

## Scheduler

Lane states: `Queued -> Admitted -> Restoring -> Ready <-> InStep -> Finished`, with `Cancelling` from `InStep` and `Cancelled` from every other state.

- **Admission.** A lane's worst case is its window blocks, its full blocks at the request's maximum context, its state and its tail. A request is admitted only when the worst cases of all admitted lanes fit the memory budget: the GPU's working-set limit less the weights, scratch and a margin, measured on the machine. At 64K context a lane's worst case is about 1.9 GiB, so 128 GB admits about 42 to 52 lanes. Otherwise it queues; beyond the queue limit the server answers 429 with Retry-After.
- **Restoring** reserves the lane's whole working set in memory, atomically, before it reads any block back from disk.
- **Each tick builds one step:** the longest-waiting Ready lane first, then other Ready decoding lanes, then prefill spans to fill the step's token budget, so a large tool result is taken over several steps and never stalls the decoders. No Ready, Restoring or Queued lane is passed over more than W ticks in a row.
- **Cancel during a step waits for the step.** Command buffers do not retain their buffers, so a lane cancelled while `InStep` goes to `Cancelling` and releases its references only when the GPU and any disk I/O of that step have finished. A client that disconnects cancels its lane.

## Models (TLA+, before code; TLC on a bench)

**CacheScheduler.** One model, because the dangerous properties cross between the cache and the scheduler. State: blocks of both kinds with reference counts, tier and sealed flag; start points; lanes with their state, block lists and reservations; the step in flight with the blocks it pins and its pending disk I/O; the memory and disk budgets. Checked exhaustively by TLC at small bounds, for safety and liveness. Invariants:

- a step reads only blocks in memory, and every block a step or its I/O uses stays in memory and owned until the step completes;
- a sealed block is never written; an unsealed block has one referrer, is invisible to matching, and is zero past its length;
- a referenced block is never dropped, and a count equals its referrers;
- a start point's state, full blocks and window blocks are all kept;
- the reservations of Restoring, Ready and InStep lanes fit the memory budget, and every block no lane references can be spilled.

Liveness: every admitted lane finishes or is cancelled; every queued request is admitted or refused.

**BlockLayout.** The pool and table arithmetic at capacity 3 or 4: slot reuse with generation counters, the window's sliding over the block list, the table's indexing with 64-bit offsets. One reversed witness per bug class: a reused slot read through a stale entry; a window off by one block; a span of 512 tokens pinning 17 window blocks instead of 25; a start point whose window blocks were released.

The engine's random walk emits traces, and TLC checks in CI that each is a behaviour of the model: that binds the code to the safety properties. Liveness is the model's, checked by TLC.

## Layers, bottom up

Each is secured before the next starts.

| | layer | gate |
|---|---|---|
| 1 | Lane state out of the engine (done) | logits identical to the unmodified engine; two interleaved lanes with poisoned scratch give each lane its own logits |
| 2 | Batched step over spans, on today's per-lane cache buffers | bit-identical cache, state and multi-step logits for a lane alone and in batches of 1, 2, 64, 65 and 129, with prime-sized prefill splits; agreement with llama.cpp; then measured step time and GPU memory at 2K, 20K and 42K context for 1 to 64 lanes, against the lane table |
| 3 | Block cache: pool, table, tree, start points, unsealed tails | CacheScheduler and BlockLayout hold in TLC with their reversed witnesses; traces checked in CI; outputs identical to layer 2 with poisoned pool slots |
| 4 | Scheduler, admission, disk tier | traces checked; a bimodal workload of 2K and 64K requests under a tight budget; disconnects in mid-step; kill -9 in mid-write loses cache and never correctness; temperature 1 with fixed seeds gives the same outputs with and without the cache |
| 5 | Server: the OpenAI endpoint, template, tool calls, 429, health | the tracer card and a replay set land; concurrent cards issuing identical tool calls; disconnects in mid-thinking stop generation |
| 6 | Sweep: replay recorded card traffic at 8 to 128 lanes | the knee becomes the route's concurrency |
| 7 | Fleet wiring: a provider entry, concurrency, a priority ceiling | a real low-priority card lands from the fleet |
| 8 | Soak under real load | days, not hours |

## Follow-ons, not in the first build

- **One attention kernel for every span, split over keys at positions fixed by the cache, not by the batch.** Each split's partial softmax is computed by the same arithmetic for one query or thirty-two, and splits merge in a fixed order, so a decode span and a prefill row give identical bits. It runs over every lane's span in one dispatch, groups the three query heads of each key head, and reads the cache through the block table near bus speed. At long context this is the largest gain: today a decode span runs the prefill tile over its lane's keys with one threadgroup per head, about 4x the decode path's split-K.
- **A kernel for decode-size batches.** Up to 2x at 16 to 32 lanes of short context, less as more lanes touch more experts. Read each touched expert's quantized rows once at full bandwidth and apply them to every lane routed to it, and the same for the dense projections (see "Where the batched path's time goes").
- **Read a shared prefix once per step** for all lanes' queries, then each lane's own keys, merged with the online softmax the engine already has.
- **Grouping lanes by predicted experts.** It saves nothing while every Ready lane steps; it pays only when more lanes are ready than a step takes.
- **8-bit cache and far-key pruning**, behind the quality replay.
- **Quality replay**: N landed low-priority cards against the current flash route, comparing how many land.

## Settled, 2026-11-22

- **Machine:** an M5 Max with 128 GB for now. An M5 Pro has about half the memory bandwidth and should run near half these rates. Each candidate machine is measured with `inky bw` and `inky bench`.
- **Per-card speed:** no floor. The lane count is one route setting, started at the knee the sweep finds.
- **Stay on Mac:** the engine is Metal only and needs an M5 or later.

## Open questions

1. The licence of our own parts of the fork: GPL as upstream, or AGPL.
