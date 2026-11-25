# Shared-prefix attention

Several decoding lanes can have the same opening prompt. Store each sealed 64-token prefix block once. When two or more query rows need the same physical K/V tile in a decode step, stage the tile once for that group and use it for every row. Each row still computes its own attention scores and output. The first cut uses resident fp16 blocks. Storage can land on layer 3 independently. The shared-read kernel depends on the cache-fixed split arithmetic described below.

## How much is shared?

Rowan measured 204 recorded opencode requests from four concurrent cards in one job. This is a favorable case for cross-card sharing. The opening ~8.2K tokens were common, about 15% at a median context of 54K. The first difference was the card-specific working directory in the system prompt. Moving those lines to the end raised the measured overlap to ~10K (21% in the replay). The first replies diverged within 70 characters. Even identical repository files later in the conversation appeared at different positions. The common head stayed roughly fixed as context grew.

At median context, decode begins around position 54K. Its 1024-position sliding window has already passed the 8.2K common head, so **none of the 40 window passes reads the cross-card head during decode**. The opportunity is in the 8 full-attention passes. The useful window case here is a same-card fork that diverged recently enough for both rows still to see the same positions.

Rowan estimated roughly 15–20% of full-attention reads were candidates for sharing. At 54K a lane reads about 1.85 GiB of K/V per decode step (160 MiB window and 1.69 GiB full). About 270 MiB of shared full K/V is roughly 14% of those bytes per additional reader. Four lanes still read the first copy, so the ideal group saving is about 10–11%: three avoided reads of the head out of four whole lane reads. For bandwidth-bound KV time with the same effective bandwidth, time falls in that proportion. For example, if KV reads occupy half the original step, that is about 5% of whole-step time before grouping losses or new overhead. Measure step latency on replay; the byte ratio alone gives no measured speedup. Storing the head once also frees roughly 270 MiB of physical memory per additional sharer. The worst-case admission policy reserves enough room for every lane to diverge, so the admitted lane count stays the same.

Within one card Rowan measured ~98% reuse across requests. Existing prompt reuse already skips much of that prefill. Claude Code cards may have a longer common head from system and tool definitions. Measure their tokenized requests separately. Lanes at different stages of the same card can also benefit whenever both are decoding and their attention sees the same blocks.

[The cross-lane prefix-copy proposal](https://github.com/robotgamedevelopers/inky/pull/22) copies a kept prompt's cache and convolution state from a donor into a free lane, then prefills the suffix. That saves prefill even when the lanes decode at different times. This proposal additionally keeps one physical copy and amortizes its decode reads across simultaneously active lanes.

## Find and retain a block

Use the block pool, start points and pin lifetime in [the layer design](DESIGN.md#blocks). The index is scoped to one engine's weights, KV format, positional settings and prefill arithmetic. For each 64-position block, look up its kind, absolute start position, parent's canonical prefix identity and hash of the ordered token IDs. Validate parent, position, kind and **all token IDs** on a hit before returning the canonical handle. Equal local tokens reached through different histories can have different K/V. A digest collision must take the miss path. Only publish complete blocks from prefill, after their writes have finished. Decode-generated history stays private across requests under the prompt-cache policy (layer design, decisions 4 and 5).

On a match, retain the generation-tagged handle and install its 64-bit offset in the lane table before releasing the old reference. Deduplicate racing publishers only after exact comparison. Restore a lane only at a start point with the corresponding recurrent state, full-block prefix and live preceding window. A token match alone cannot reconstruct the state. Charge one allocation per physical block and retain the existing worst-case admission bound, so every admitted lane can diverge.

Only a retained start point (a prompt end, a checkpoint at every 16th block boundary, or a lane head) can restore the convolution state. Within a retained checkpoint interval, divergence can require prefilling up to 1023 positions to rebuild that state. If a checkpoint has been evicted, use the earlier available start point and account for the extra work. Direct every replay write into a private scratch block or private tail. The worst-case reservation covers one private K/V history per lane. It does not by itself cover up to 1023 positions of duplicate replay scratch alongside retained blocks. Before dispatch, reserve scratch for the chosen replay chunk, including blocks pinned through its fence. Recycle it after completion. Shorten the chunk or defer replay if the pool cannot cover the peak; never spend capacity already promised to another lane. For a completed replay block whose canonical match was already retained, keep the sealed handle and discard the private duplicate only after the step finishes. At the first differing block, continue in private zeroed storage. This is the layer-3 append-only rule: a sealed shared block has no writable alias. A partly written tail belongs to one writer and has zeroes beyond its length.

Publish only after the prefill write fence. Release lane, start-point and index ownership under the pool's eviction policy. A step pin lasts until the GPU completes even if its lane cancels. Reuse a slot only when the last owner and pin are gone, then increment its generation.

## Fix the reduction boundaries first

The original decode kernel selected its split width from each lane's visible key count. With the default eight target splits it used `32 × ceil(nkeys / 256)`. At 54,000 keys the width was 6752; at 54,100 it was 6784. Tile [6720, 6784) therefore crossed a split boundary in just one lane. Window attention also moved its origin with the first visible key.

The private kernel anchors splits at multiples of 64 positions for window attention and 256 for full attention, starting at absolute position zero. Each row intersects the grid with its own visible interval. SIMD key assignment stays anchored to that grid within partially visible tiles. Empty SIMD groups contribute no softmax state. The combine pass visits the row's visible splits in increasing position order, including the sink exactly once. Each lane still reads its own K/V.

For an engine configured with context limit `C`, the tensor path's partials arena holds `max(256, ceil(C / 256) + 1)` split slots per query head (`specs/SplitArena.tla`). The extra slot accommodates a clipped boundary in the experimental pruning path. Each slot stores a maximum, a denominator and 128 weighted value components in fp32. Prefill batches as many query rows as fit in this arena, reusing it after each combine. A row's split widths are independent of both the configured limit and the number of rows in the batch.

Review the one-time chip golden changes against llama.cpp (all checked argmaxes agree, worst KL ≤ 0.01) and the invariance gates below. `inky crosspath` compares each logit bit and a hash of the K/V and convolution state after the same token is processed across multi-row and split prefill chunkings, straddling 64, 256, 1024 and 2048 tokens. Freeze the reviewed private-path goldens before enabling shared reads.

## Stage one K/V tile for several queries

Group simultaneous decode rows by pass, KV head and absolute fixed split. Keep each row’s accumulator live for that split. Within the split, stage a tile once for rows whose handles resolve to the same physical tile and generation. Synchronize before those rows consume it with their own masks and queries. Cap query rows per group by register and threadgroup memory limits; extra groups may reload the tile. A suffix or singleton uses private reads with the same fixed-split arithmetic. Pin all participating blocks through GPU completion.

If a fixed split crosses the divergence, carry each row's accumulator through the shared and private portions in baseline key order. Emit a partial only at that row's fixed split end clipped to its visible range. A tile boundary or the end of the shared head does not create a reduction boundary. The implementation must match the re-baselined private kernel bit for bit, including its treatment of the current token, masks, precision, and final merge order.

## Gates in landing order

1. **Models.** Extend BlockLayout at 3 and 4 slots with two histories and per-row visible ranges. Model the split grid and tile ranges separately (scale both down and include a tile boundary strictly inside a split). Compare each row's visited keys, split membership and ordered list of emitted partials with its private reference. Cover unequal positions, two windows and a split that crosses divergence. CacheScheduler owns retain/release, the actual refcount versus enumerated owners, immutable sealed blocks, step pins through completion, and window references held by retained start points. Exhaust both bounded models in TLC and run each reversed witness against its named invariant.

   | Model | Reversed action | Required failure |
   | --- | --- | --- |
   | BlockLayout | Match equal local tokens and position under different parent histories | A lane resolves to the wrong prefix. |
   | BlockLayout | Use the donor's window for both rows | The other row's visible keys differ from its reference. |
   | BlockLayout | Emit a partial at each tile end | One split produces an extra partial before its true end. |
   | CacheScheduler | Release the same owner's final reference twice | The count goes below zero. |
   | CacheScheduler | Publish an unsealed tail | A second lane sees a still-writable block. |
   | CacheScheduler | Keep a start point while releasing its window blocks | Restore lacks the required window prefix. |
   | CacheScheduler | Write replay K/V into a sealed block | The other lane's retained content changes. |
   | CacheScheduler | Reuse a slot while a cancelled step still has a pin | The in-flight reader's generation is stale. |

2. **Layer-3 storage.** Use the existing block-pool and lane-table roadmap rows. With the old attention arithmetic still active, compare K/V, recurrent state and logits byte for byte against private storage on poisoned and reused slots. Force a digest collision, fork within a block and at a block boundary, re-prefill from a retained checkpoint, drop the donor, cancel in flight and then allocate again. Before layer 3 locks, CI must replay storage events against the ownership model, including publish, retain, release, restore, pin, completion and slot reuse. Require the real trace to pass and a trace with a corrupted release to fail. Keep the existing goldens unchanged.
3. **Private fixed-split kernel.** Pass the reference and one-time golden review above. Run `binv`, `lanes`, `spans` and `dbatch` alongside the prefill/decode comparison. Repeat with different group sizes, contexts and windows. Group size, lane position relative to other lanes and prefill chunking must leave a row's result unchanged.
4. **Grouped reads.** Compare shared and private fixed-split K/V, state and logits at every step on the same machine and tuning. Require unchanged fixed-split goldens and all four invariance gates. Test ends at 63/64/65, 1023/1024/1025 and a split boundary ±1. Inject an extra partial at an internal tile edge and require a mismatch on a fixture sensitive to the changed order. In CI, replay each row’s range, tile visits, split index and partial emissions against the layout model. Require the real grouping trace to pass and a trace with a corrupted partial boundary to fail.
5. **Replay.** Run the same recorded requests with grouped reads off and on at fixed seeds, lane counts, context lengths and admission budget. Record eligible and actually shared full/window bytes, peak physical KV allocation, loads per query group, step latency and aggregate decode tokens/s. Include no overlap, a singleton, staggered same-card lanes and concurrent shared heads. Compare hardware counters where available with logical tile counts. Enable by default only after the bitwise gates pass and replay shows repeatable benefit without material regression for singleton or unrelated lanes.

## Apple M5 Pro Replay Measurements (`inky prefixbench` & `inky sharedkv`)

Measured on Apple M5 Pro (64 GiB unified memory, `Inkling-2-Mini-Q4_K_M.gguf`, fp16 resident Layer 3 `BlockPools`, `-c 10240`, `8` decode steps per case) via `inky prefixbench --ref ref/long.tokens -c 10240`. Every case compares unshared private execution (`Layer3Pool::new` with `INK_NO_GRP_DEC=1`) against shared-prefix execution (`Layer3Pool::with_persistent_starts`, `PrefixTree` block deduplication, and grouped `k_attn_grp_dec` `G_MAX = 8`), verifying bit-for-bit identical vocabulary logits across every lane and step:

| Case | Lanes | Private KV (MiB) | Shared KV (MiB) | Peak Metal Alloc (MiB) | Private E2E (ms) | Shared E2E (ms) | Decode Median `priv -> shrd` (ms/step) | Shared Aggregate Decode (tok/s) | Logit Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `singleton-1x8k` | 1 | 420.0 | 600.0 | 19,602.1 | 10,128.0 | 4,501.7 | `39.1 -> 39.2` | 25.5 | exact |
| `unrelated-4x2k` | 4 | 912.0 | 912.0 | 19,651.2 | 5,396.1 | 5,098.7 | `67.0 -> 67.2` | 59.6 | exact |
| `staggered-card-4x2k` | 4 | 964.0 | 320.0 | 19,715.2 | 5,809.6 | 2,507.0 | `64.4 -> 63.1` | 63.4 | exact |
| `shared-head-4x8k` | 4 | 1,716.0 | 648.0 | 20,897.2 | 50,142.2 | 5,561.2 | `68.2 -> 65.2` | 61.4 | exact |
| `shared-head-8x8k` | 8 | 3,430.0 | 768.0 | 22,628.0 | 133,627.6 | 7,305.1 | `96.4 -> 92.9` | 86.2 | exact |
| `shared-head-16x8k` | 16 | 6,878.0 | 1,008.0 | 26,112.7 | 477,457.3 | 9,632.3 | `139.2 -> 157.4` | 101.7 | exact |

Notes on the measured behavior:

- **Physical K/V footprint**: On `singleton-1x8k`, persistent start points retain periodic 1,024-token (`EVERY@16`) sliding-window checkpoints (`+180 MiB` over pruning every expired window block). Across multi-lane shared heads, deduplicating the 8,192-token prefix saves `1,068 MiB` at 4 lanes (`1,716 -> 648 MiB`), `2,662 MiB` at 8 lanes (`3,430 -> 768 MiB`), and `5,870 MiB` (`5.73 GiB`, `6.8x` reduction) at 16 lanes (`6,878 -> 1,008 MiB`), while `staggered-card-4x2k` drops from `964 MiB` to `320 MiB`.
- **Singleton & unrelated controls**: `singleton-1x8k` (`39.1 -> 39.2 ms/step`) and `unrelated-4x2k` (`67.0 -> 67.2 ms/step`, identical `912 MiB` K/V pool) show no material decode regression when `no_grp_dec = false` partitions unshared splits onto `k_attn_dec_solo`.
- **Boundary & bitwise verification**: `inky sharedkv` additionally checks ends `[63, 64, 65, 255, 256, 257, 1023, 1024, 1025]` against both `INK_NO_GRP_DEC=1` and private Layer 2 fixed-split `encode_decode`, with `INK_SHAREDKV_SABOTAGE=1` enforcing failure detection in `tests/ci.sh`.
