# Lanes: design

Status (2026-11-25): Layers 1–7 are built and running in CI — the multi-lane Metal engine (Layers 1–2), paged 64-token K/V block pools, prefix tree and start-point restore (Layer 3), starvation-bounded scheduler, admission guard and SSD spill tier (Layer 4), stateless OpenAI Chat Completions + Anthropic Messages HTTP server (Layer 5), recorded card traffic replay sweep (Layer 6), and private-network fleet wiring (Layer 7); Layer 8 multi-day live soak continues under production load. The work is tracked layer by layer in [ROADMAP.md](ROADMAP.md).

One process on one Apple silicon machine serves Inkling 2 Mini to many low-priority coding-agent tasks (cards) at once. It starts from inkling, Eris's engine ([github.com/Eriskii/erisinky](https://github.com/Eriskii/erisinky), GPL-3.0-or-later). This repository is the fork: `main` builds and serves the multi-lane engine (`inky serve`), while the historical branch `tracer-serve` held the initial one-lane endpoint the tracer card ran on. The aim is the most cards landed per hour, not the fastest single card.

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

CI checks agreement with llama.cpp across three `inky check` runs: a 58-position short-prompt decode run (`encode_step`), a 58-position short-prompt batched decode-step run (`INK_DEC=1`, `encode_decode`), and a 24-position long-prompt run (`INK_PF=2380` on the 2,403-token reference, comparing position `2379` at the end of the 2,380-token prefill plus the 23 post-prefill decode-path positions `2380..=2402`). It does not run cold token-by-token decode from position 0 across the whole 2,403-token prompt: near the end of that prompt some positions are ill-conditioned in the model itself, and llama.cpp disagrees with its own batched run on 16 of the last 64 (inky decoding token by token from position 0 agrees with llama.cpp's batched run on 47, and with its token-by-token run on 50).

## The cold read, and what it changed

A second reader read this design and the code cold on 2026-11-22 (verdict: revise before layer 2). The sections below include its accepted findings. Its claims about the engine, measured on 2026-11-23:

- **The batched path is repeatable and batch-invariant.** 16 prefills in a row give identical bits, and splits at prime offsets and on both sides of 64, 512, 1024 and 2048 (`inky binv`, which takes several boundaries per split) give identical logits. The atomic packing of tokens by expert has not shown up in results.
- **One-token batches** were a real bug, fixed before layer 1 (`mix_t`).
- **Masked rows reaching the tensor multiply** were real: the window K/V scratch is now zeroed at every prefill (`k_zero`), found by `inky lanes`.
- **The convolution state exists only at the end of a step.** Inside a prefill chunk it is never kept, so a block boundary is a start point only where a step ended there (Blocks, below).
- **Batched attention does not suit decode spans.** At 3K context it takes about 5 ms for a tile of 32 queries, where the decode path's split-K attention takes about 1.3 ms for one token; a lane's single token pays for the whole tile. The decode path's kernel cannot simply be used for decode spans: it gives different bits from the prefill path, which would make the cache change answers (Decisions, 2). The fix is one attention kernel for all spans (Follow-ons).
- **The cache is read at about 210 GB/s and every lane reads all of its own:** the lane table above.

## Measured: lanes on the M5 Max (2026-11-23)

Decode steps (`encode_decode`): row-wise work through the tensor path, each lane's attention through the decode path's fused split-K kernel, lanes' independent dispatches overlapped, the output layer one GEMM for all lanes. Every lane is bit-identical to itself decoding alone (`inky dbatch`) and agrees with llama.cpp (short reference: 58/58, worst KL 0.0022).

| context | one stream (decode path) | lanes | total | gain |
|---:|---:|---:|---:|---:|
| 2K | 93 tok/s | 64 | 578 tok/s | 6.2x |
| 16K | 79 | 64 | 389 | 4.9x |
| 64K | 53 | 27 | 150 | 2.9x |
| 128K | 33 | 14 | 78 | 2.4x |
| ≈256K | 21 | 7 | 40 | 1.9x |

(16K and up from `inky maxctx` on real code, 60 GB for lanes. The built-in long checkpoint is 262,000 tokens and the current context limit is 262,080.) The gain falls with context because each lane reads its own cache for every token; at 42K a lane already costs about 80% of that bandwidth floor. Prefill at 128K to about 256K runs at 875 tok/s: 214 s to the first token at about 256K.

**Four cards at once through `inky serve`** (8 lanes, 131K context per lane), the tracer card four times: all four landed the same two-line fix as the merged change. 208 requests, 10.6 M prompt tokens of which 97% came from the lanes' kept prompts, 314 K prefilled, 53 K generated, largest prompt 97 K.

Tried and reverted: decode-step experts with the decode path's f32 dot products (exact, KL 0.0004, but compute-bound: 262 tok/s at 64 lanes against 578).

## Decisions

1. **One process, one copy of the weights.** Separate processes cannot share weight reads.
2. **One kind of step, two arithmetics.** A step advances a set of spans, each (lane, tokens). A decoding lane brings one token, a prefilling lane up to a chunk. The projections, router, experts and controllers run once over every span's tokens; attention and the convolutions run per span. The two kinds of span do not share an arithmetic: a prefill row's attention is the batched path's, a decode step's attention is the decode path's fused split-K kernel, and the same token comes out with different bits from each. Each arithmetic is batch-invariant on its own — a token's numbers depend neither on what else shares its step nor, for a prefill row, on how its own lane's work was chunked (`inky spans` for prefill rows, `inky dbatch` for decode steps, `inky binv` for prefill splits) — but neither invariance crosses from one arithmetic to the other. So the decode path is used for a lane, as its decode step; it just never stands in for a prefill row. One attention kernel for every span would merge the two arithmetics (Follow-ons); until then both stand, and decision 4 is what makes that safe.
3. **The server is stateless; the cache is a tree of blocks.** A request is a token list, matched against a prefix tree of sealed cache blocks. A lane lives for one request: it starts at the deepest start point the tree holds for its tokens, and when the request ends its blocks stay in the tree. There are no sessions. A shared prefix, a resumed conversation, a harness side request and a rollback are all the same case.
4. **The cache never changes an answer.** A request's output is a function of the request and its seed; the cache only changes how fast it comes. That is well-defined because the boundary between the two arithmetics is the end of the prompt, which the request itself fixes: the prompt is prefilled and the reply is decoded, whatever the cache held, so the same request always splits at the same place. Only what the prefill wrote is reused across a card's turns: the kept prompt and the convolution state at its end. Cache written while decoding a reply is never read as history; the next turn prefills the reply again exactly as the client rendered it (decision 5), over it. The seed is the request's own (its `seed`, or a hash of the rendered tokens and the sampling settings), never arrival order.
5. **Earlier turns are rendered exactly as the client sends them.** A client that drops the reasoning of earlier turns gets them rendered without it, as the chat template does. The tokens a lane generated are therefore reused only up to where its turn began; the turn itself is prefilled again with the next request (its visible text and tool calls: short). No map from visible text back to generated tokens exists.
6. **Exact by default.** fp16 cache, no pruning. The 8-bit cache and far-key pruning are flags that go on only if the quality replay passes.
7. **One step in flight.** The CPU picks every step's lanes. This costs about 3% against GPU-chained decode and removes the hardest ordering cases.
8. **The Mac only generates tokens.** Harness and tools run on other machines and reach it over a private network.
9. **Release builds abort on panic (`[profile.release] panic = "abort"`).** Raw-pointer GPU arenas cannot safely continue or unwind after a thread panic in release builds; recoverable parse and input boundaries (such as GGUF loading and CLI validation) return `Result` errors before touching GPU state.

## Blocks

A block holds the cache for 64 consecutive positions. Two kinds, because the model has two kinds of attention:

- **window block**: the 40 sliding-window layer passes, 10 MiB at fp16.
- **full block**: the 8 full-attention layer passes, 2 MiB at fp16.

**Inside a block.** The bytes are `[pass][K|V][head][position in the block][HD]`, each element `kvb` bytes: 2 for the fp16 cache, 1 for `--kv8`. A head's 64 keys are contiguous. At `kvb = 2` those sizes are the 10 MiB and 2 MiB above; at `kvb = 1` each block is half that. The address takes the pass, K or V, and the head. It is not the slot base plus the position times `stride / 64`. That step is position-major across every pass, K, V and head, and for window slot 1 at position 1 it lands on pass 0, V, head 2, position 0.

A block is written by exactly one lane, then sealed, then never written again. Sharing is a reference count; because blocks are append-only there is no copy-on-write. The cache for position p depends only on tokens 0..p, so two requests with the same first N tokens share the same blocks exactly.

**Start points.** The recurrent state (convolution rows, 3.66 MiB) exists only at the end of a step. A position k is a start point when the tree keeps all three of: the state at k, the full blocks before k, and the window blocks covering the 1024 positions before k. A start point references those window blocks, so they are not released while it is kept. In `inky serve` (`ServeBlockCache`), start points are kept at block-aligned prompt ends and at periodic `EVERY` boundaries — every 2,048 tokens (32 blocks, `EARLY_EVERY_TOKENS`) up to 16,384 tokens, widening to every 4,096 tokens (64 blocks, `LATE_EVERY_TOKENS`) above 16,384 tokens — and never at a decoding lane's head, since decode writes use decode arithmetic and are never reused as prompt history (Decisions 2, 4, and 5; the standalone `BlockBook` model harness additionally tests 16-block strides and running heads). Prefill chunks align to 64-token block boundaries (except a prompt's final unaligned tail), so periodic boundaries land at chunk ends. A request starts at the deepest retained start point its prompt tokens match and prefills the suffix from there.

**Unsealed tails.** A lane's last, partly written block is private, zero-filled past its length on allocation, never matched, and dropped when the lane ends. Pool slots are zero-filled on reuse, so a masked read never meets a non-finite value.

**A span pins** the window blocks its attention reads: up to ceil((1023 + C) / 64) + 1 for a span of C tokens, and every full block of its lane.

**Addressing.** Blocks live in a pool of GPU buffers. Attention reads keys through a per-lane block table whose entries hold 64-bit offsets, so no buffer or offset limits the cache to 8 GiB.

**Tiers.** A block is in memory, on disk, or both. Blocks no lane references (the tree's cold blocks) are written to the SSD lazily, only under memory pressure, least recently used first, and dropped from memory; they are dropped from disk under disk pressure. A request whose blocks were dropped prefills again: slower, never wrong.

## Scheduler

Lane states: `Queued -> Admitted -> Restoring -> Ready <-> InStep -> Finished`, with `Cancelling` from `InStep` and `Cancelled` from every other state.

- **Admission.** The server sizes its pools from the GPU working-set limit after subtracting weights, scratch and a margin. It checks the physical prompt and reply limits before opening a response. When admitting concurrent lanes, it accounts for their pinned blocks, unwritten prompts, rolling windows and up to 4,096 future decode tokens per lane. A longer reply extends that reservation as it advances. Requests wait when capacity is busy. A full queue returns HTTP 429 with `Retry-After`. If every runnable lane needs another block and none can make progress, the scheduler fails one request to release its blocks. An open stream receives an error event.
- **Restoring** reserves the lane's whole working set in memory, atomically, before it reads any block back from disk.
- **Each tick builds one homogeneous step (`ServeStepPlanner`):** decode steps (`ServeStepPlan::Decode`, 1 token per ready decoding lane via `encode_decode`) and prefill steps (`ServeStepPlan::Prefill`, 64-token-aligned spans sliced up to the step's `chunk` token budget via `encode_spans`) run in separate command buffers and are never mixed in one step. Ready decoding lanes burst first so multi-chunk prefills do not stall active streams, up to the starvation bound `W = 16` (`SERVE_STARVATION_W`): once any runnable prefill lane has been bypassed `W` ticks (`wait_ticks >= W`, with no runnable decoder above it), a prefill step preempts, reserving span budget for ceiling prefillers first so no runnable lane exceeds `W` bypassed ticks. Admission from the wait queue is likewise bounded against indefinite bypass.
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

**InBlock.** The bytes inside one slot, at 3 heads and 4 positions: `[pass][K|V][head][position][HD]`, with the element width a constant (`kvb` 2, and again at 1). The action walks those coordinates and stores an address by inlining the arithmetic, then records that address. `Ordered` compares it to a formula the action does not call. `Inside` says the row stays in the slot. `NoCollision` says the recorded addresses are unique. `InBlockTrace` is the behaviour `inky inblock-trace` writes from `key_row_at` at those bounds, and CI regenerates it before TLC.

The engine's random walk emits traces, and TLC checks in CI that each is a behaviour of the model: that binds the code to the safety properties. Liveness is the model's, checked by TLC.

## Inky mode: memory in, memory out

The Mac goes in and out of inky mode. Starting inky gives the GPU the memory it needs; stopping inky gives it back.

- **A one-time install, by the owner:** `sudo inky tune-memory` installs a small root LaunchDaemon. It does two things only: raise the GPU's wired-memory limit (`iogpu.wired_limit_mb`) and put it back. It accepts no other setting and runs nothing else. `sudo inky tune-memory --undo` removes it.
- **Upgrading an existing helper:** run `sudo inky tune-memory` using the upgraded binary, then restart your Inky processes with that binary. The installer retires the old socket in `/var/run/inky`. The upgraded binary uses `/private/var/db/inky/memd.sock`, so the helper and clients need to be upgraded together.
- **Entering:** inky connects to the helper over a Unix socket only the owner's user can open. The helper records the limit it finds and raises it to a value sized from the Mac's memory (118000 MB on 128 GB, leaving about 13 GB for macOS).
- **Leaving:** when inky's connection closes, the helper restores exactly the limit it found. A clean exit, a crash, `kill -9` or a sleep all close the connection, so the memory is always given back. With several inky processes, the limit is held while any is connected.
- **In the sprint,** nova-worker drives it: it starts inky when it brings local lanes up, and stops it when it stops serving them, so the laptop goes into inky mode for sprint work and out of it afterwards.
- **Inside inky mode,** inky asks Metal for the recommended working set, subtracts the weights, scratch and a margin, and sizes lanes and admission from that. Nobody passes a memory budget by hand. Without the helper, inky runs inside the default limit and says, once, what installing it would give.

## Tunings

A tuning is one machine's settings, in a file under `tunings/`, matched by chip and memory: the GPU wired-memory limit in inky mode, the margin kept free of lane caches, and in time the kernel choices that suit that GPU. inky picks the tuning that matches the machine it runs on; a machine without one runs on defaults derived from its memory. Tunings are how inky reaches other machines: anyone with a Mac it does not know yet can measure, write a tuning and contribute it back.

inky is tuned 100% for the first one, `m5-max-128gb`: the Apple M5 Max MacBook Pro with 128 GB. The next are an M3 Ultra Mac Studio with 512 GB and, when there is one, an M5 Ultra Mac Studio.

## Layers, bottom up

Each is secured before the next starts.

| | layer | gate |
|---|---|---|
| 1 | Lane state out of the engine (done) | logits identical to the unmodified engine; two interleaved lanes with poisoned scratch give each lane its own logits |
| 2 | Batched step over spans, on per-lane cache buffers (done) | bit-identical cache, state and multi-step logits for a lane alone and in batches of 1, 2, 64, 65 and 129, with prime-sized prefill splits; agreement with llama.cpp; then measured step time and GPU memory at 2K, 20K and 42K context for 1 to 64 lanes, against the lane table |
| 3 | Block cache: pool, table, tree, start points, unsealed tails (done) | CacheScheduler and BlockLayout hold in TLC with their reversed witnesses; traces checked in CI; outputs identical to layer 2 with poisoned pool slots |
| 4 | Scheduler, admission, disk tier (done) | traces checked; a bimodal workload of 2K and 64K requests under a tight budget; disconnects in mid-step; kill -9 in mid-write loses cache and never correctness; temperature 1 with fixed seeds gives the same outputs with and without the cache |
| 5 | Server: OpenAI + Anthropic endpoints, template, tool calls, 429, health (done) | the tracer card and a replay set land; concurrent cards issuing identical tool calls; disconnects in mid-thinking stop generation |
| 6 | Sweep: replay recorded card traffic at 8 to 128 lanes (done) | the knee becomes the route's concurrency |
| 7 | Fleet wiring: a provider entry, concurrency, a priority ceiling (done) | a real low-priority card lands from the fleet |
| 8 | Soak under real load (in progress) | days, not hours |

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
