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

The batched path fits **58 ms + 0.21 ms per token**. A decode step for B lanes is that pass with one token per lane, plus each lane's own cache read. So the calibrated step time is

    step(B) = 58 ms + B x (0.21 ms + lane cache bytes / 565 GB/s)

| active lanes | total tok/s at 20K context | per lane | with 8-bit cache | per lane |
|---:|---:|---:|---:|---:|
| 1 (today's decode path) | 75 | 75 | | |
| 16 | 210 | 13 | 240 | 15 |
| 32 | 340 | 10.7 | 400 | 12.6 |
| 64 | 500 | 7.8 | 640 | 10 |
| 128 | 640 | 5.0 | 900 | 7 |
| limit | 910 | | 1500 | |

These replace the earlier bandwidth-only estimates, which were too high below 64 lanes. About half of the 58 ms can be removed (see "Where the batched path's time goes"). The knee is near 64 active lanes.

### Baseline: llama.cpp's own batching

llama-server (master 44d5087, Metal, 64 parallel slots, short prompts, 256 tokens per stream) on the same machine:

| streams | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| total tok/s | 41 | 62 | 100 | 142 | 155 | 194 | 215 |
| per stream | 41 | 34 | 27 | 19 | 10 | 6.4 | 3.6 |

It flattens near 200 tok/s, about twice one inky stream. inky's batched path, at 58 ms + 0.21 ms per token, would give about 260, 490 and 900 tok/s at 16, 32 and 64 lanes at the same short context. That is an estimate from timing one sequence's tokens, not a built decode step. So llama.cpp's batching is not close enough to build on; the engine work is needed.

### Where the batched path's time goes

`inkling prof_pf` at 32 tokens in one batch (52 ms):

| part | time | note |
|---|---:|---|
| Experts (up and down projections) | 31 ms | about 46 of 128 experts per layer are touched; they are read at about 330 GB/s, 58% of the memory bus, in 16-token tiles that are 35% full |
| Dense projections, router, loop controllers | 16 ms | about 1.3 GB of weights, which the single-token decode path reads in 2 to 3 ms |
| Everything else | 5 ms | mixes, attention, gathers |

So the 58 ms is not a fixed overhead. It is the experts a batch touches, read by a path tuned for large batches. At full bandwidth the same 32 tokens would take about 25 ms. From 32 to 128 tokens the expert time grows only from 31 to 41 ms.

### Batch invariance

`inkling binv` prefills the same tokens split at different points and hashes the final logits. A split moves every later token to a different batch, tile slot and set of neighbours. On two prompts (2403 and 61 tokens), every split gives bit-identical logits, at chunk sizes from 64 to 4096. The check found one bug: a batch of exactly one token went to the decode kernel with batch-layout arguments and gave wrong, non-repeatable logits (fixed in `mix_t`). Upstream never sends a one-token batch; a lane decoding alone would.

## Decisions

1. **One process, one copy of the weights.** Separate processes cannot share weight reads.
2. **One kind of step.** A step advances a set of spans, each (lane, tokens). A decoding lane brings one token; a prefilling lane brings up to a chunk. Decode and prefill are the same operation, so there is one path to make correct and one to tune. Today's single-stream decode path stays only as the reference the batched step is checked against.
3. **The server is stateless; the cache is a tree of blocks.** A request is a token list. It is matched against a prefix tree of sealed cache blocks, a lane starts at the deepest match, and when the lane finishes its blocks stay in the tree. There are no sessions. A shared prefix, a resumed conversation, a harness side request (the title turn that reset the tracer's cache) and a rollback are all the same case: a match at a block boundary.
4. **Exact by default.** fp16 cache, no pruning. The 8-bit cache and far-key pruning are flags that go on only if the quality replay passes.
5. **One step in flight.** The CPU picks every step's lanes. This costs about 3% against GPU-chained decode and removes the hardest ordering cases.
6. **The Mac only generates tokens.** Harness and tools run on other machines and reach it over a private network. 64 cards' builds do not fit beside 64 lanes.

## Blocks

A block holds the cache for 64 consecutive positions. Two kinds, because the model has two kinds of attention:

- **window block**: the 40 sliding-window layer passes, 10 MiB. A lane references only the blocks covering its last 1024 positions (17 at most).
- **full block**: the 8 full-attention layer passes, 2 MiB. A lane references all of them back to position 0.

At each block boundary the block also stores the small recurrent state (convolution rows, loop-controller state), so any boundary is a point a lane can start from.

A block is written by exactly one lane, then sealed, then never written again. Sharing is a reference count. Because blocks are append-only there is no copy-on-write. The cache for position p depends only on tokens 0..p, so two requests with the same first N tokens share the same blocks exactly. This needs the step to be batch-invariant (Gate 2).

Each block is in memory, on disk, or both. A lane that is waiting has its private blocks written to the SSD and dropped from memory. Blocks no lane references are kept as cache and dropped least-recently-used, memory first, then disk. A request whose blocks were dropped prefills again: slower, never wrong.

Attention reads keys through a per-lane block table, 64 keys per entry, in place of one contiguous buffer.

## Scheduler

Lane states: `Matching -> Restoring -> Ready -> InStep -> Ready ... -> Finished`, with `Cancelled` reachable from every state and `Parked` for a lane that must wait for memory.

Each tick builds one step:

1. The longest-waiting Ready lane goes in first.
2. Decoding lanes are added by fewest new experts, predicted from each lane's previous token (48 x 128 bits per lane). Simulated on the engine's routing dumps this saves about 20% of weight bytes.
3. Prefill spans fill what is left of the step's token budget, so a large tool result is taken in over several steps and never stalls the decoders.
4. No Ready lane is passed over more than W steps in a row.

Admission: a request is admitted only if its worst case fits in memory with every admitted lane's. Otherwise it queues, and beyond the queue limit the server answers 429 with Retry-After.

## Models (TLA+, before code; TLC on a bench)

**BlockCache.** State: each block's reference count, tier, sealed flag and parent; each lane's block lists; the memory and disk budgets. Actions: Match, AllocTail, Seal, SlideWindow, Finish, Park, Restore, DropUnreferenced. Invariants: a step reads only blocks in memory; a sealed block is never written; a referenced block is never dropped; a count equals its referrers; a child's tokens extend its parent's; memory stays within budget. Liveness: an admitted lane eventually steps.

**BlockLayout.** The same cache with the real arithmetic at capacity 3 or 4: slot reuse in the block pool with generation counters, the window's sliding over the block list, the block table's indexing. One reversed witness per bug class (a reused slot read through a stale table entry; a window off by one block).

**LaneScheduler.** State: lane states, the step in flight, wait counters. Invariants: a lane is in at most one step; a lane's tokens advance in order with no gap or repeat; the bypass bound W holds; a cancelled lane releases every reference. Liveness: every Ready lane steps; every request ends Finished, Cancelled or refused.

The engine's random walk emits traces of all three and TLC checks them in CI.

## Layers, bottom up

Each is secured before the next starts.

| | layer | gate |
|---|---|---|
| 1 | Lane state out of the engine: one lane behaves exactly as today | logit hash identical to the unmodified engine |
| 2 | Batched step over spans | a lane's logits are bit-identical alone and in any batch; within today's tolerance of the llama.cpp reference |
| 3 | Block cache in memory: tree, references, window sliding | BlockCache and BlockLayout hold on traces; outputs identical to layer 2 |
| 4 | Scheduler and admission | LaneScheduler holds on traces; no lane starves in a soak |
| 5 | Disk tier and parking | identical outputs after park and restore; kill -9 loses cache, never correctness |
| 6 | Server: the OpenAI endpoint, template, tool calls, 429, health | the tracer card and a replay set land through it |
| 7 | Sweep: replay recorded card traffic at 8 to 128 lanes | the knee becomes the route's concurrency |
| 8 | Fleet wiring: a provider entry, concurrency, a priority ceiling | a real low-priority card lands from the fleet |
| 9 | Soak under real load | days, not hours |

## Follow-ons, not in the first build

- **A kernel for decode-size batches.** The largest known gain: about 2x at 16 to 64 lanes. Read each touched expert's quantized rows once at full bandwidth and apply them to every lane routed to it, and give the dense projections the same treatment (see "Where the batched path's time goes"). The first build uses the existing batched path, which is correct and batch-invariant today.
- **Read a shared prefix once per step** for all lanes' queries, then each lane's own keys, merged with the online softmax the engine already has. Version one shares the memory and skips the prefill; this shares the reads too.
- **8-bit cache and far-key pruning**, behind the quality replay.
- **Quality replay**: N landed low-priority cards against the current flash route, comparing how many land.

## Settled, 2026-11-22

- **Machine:** an M5 Max with 128 GB for now. An M5 Pro has about half the memory bandwidth and should run near half these rates. Each candidate machine is measured with `inkling bw` and `inkling bench`.
- **Per-card speed:** no floor. The lane count is one route setting, started at the knee the sweep finds.
- **Stay on Mac:** the engine is Metal only and needs an M5 or later.

## Open questions

1. The licence of our own parts of the fork: GPL as upstream, or AGPL.
