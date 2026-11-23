# Shared-prefix attention

Several decoding lanes can have the same opening prompt. Store each sealed 64-token prefix block once. When two or more query rows need the same physical K/V tile in a decode step, stage the tile once for that group and use it for every row. Each row still computes its own attention scores and output. Start with resident fp16 blocks and the existing decode arithmetic.

## How much is shared?

Rowan measured 204 recorded opencode requests from four concurrent cards in one job. This is a favorable case for cross-card sharing. The opening ~8.2K tokens were common, about 15% at a median context of 54K. The first difference was the card-specific working directory in the system prompt. Moving those lines to the end raised the measured overlap to ~10K (21% in the replay). The first replies diverged within 70 characters. Even identical repository files later in the conversation appeared at different positions. The common head stayed roughly fixed as context grew.

On that workload Rowan estimated 15–20% of per-lane full-attention reads were opportunities for sharing and roughly 270 MiB could be avoided per additional lane with the same prefix. For scale, 128 full blocks cover 8192 positions and occupy 256 MiB. The first copy still has to be stored and read. At a common-prefix fraction f, a group of L lanes with equal visible lengths ideally avoids (L−1) × f lane-equivalents of full-cache reads per step. These are byte estimates. Throughput depends on the rest of the step and on how many rows can consume one tile.

Within one card Rowan measured ~98% reuse across requests. Existing prompt reuse already skips much of that prefill. Claude Code cards may have a longer common head from system and tool definitions. Measure their tokenized requests separately. Lanes at different stages of the same card can also benefit whenever both are decoding and their attention sees the same blocks.

[The cross-lane prefix-copy proposal](https://github.com/robotgamedevelopers/inky/pull/22) copies a kept prompt's cache and convolution state from a donor into a free lane, then prefills the suffix. That saves prefill even when the lanes decode at different times. This proposal additionally keeps one physical copy and amortizes its decode reads across simultaneously active lanes.

## Find and retain a block

Use the block pool, start points and pin lifetime in [the layer design](DESIGN.md#blocks). The index is scoped to one engine's weights, KV format, positional settings and prefill arithmetic. For each 64-position block, look up its kind, absolute start position, parent's canonical prefix identity and hash of the ordered token IDs. Validate parent, position, kind and **all token IDs** on a hit before returning the canonical handle. Equal local tokens reached through different histories can have different K/V. A digest collision must take the miss path. Only publish complete blocks from prefill, after their writes have finished. Decode-generated history stays private across requests, as required by the two arithmetics.

On a match, retain the generation-tagged handle and install its 64-bit offset in the lane table before releasing the old reference. Deduplicate racing publishers only after exact comparison. Restore a lane only at a start point with the corresponding recurrent state, full-block prefix and live preceding window. A token match alone cannot reconstruct the state. Charge one allocation per physical block and retain the existing worst-case admission bound, so every admitted lane can diverge.

At the first differing block, allocate a private, zeroed tail for each writer. Keep the complete matching blocks shared. For a divergence inside a block, resume at a valid start point and privately prefill the matching remainder and suffix. This is copy-on-write at block granularity: the sealed shared block is immutable and the first write goes to the new tail. This first cut never lends a partly written tail to another lane. Release lane, start-point and index ownership under the pool's eviction policy. A step pin lasts until the GPU completes even when its lane is cancelled. Reuse the slot only when all owners and pins are gone, incrementing its generation.

## Read a tile for several queries

The 8 full-attention passes can use the common prefix throughout a long decode. In each of the 40 sliding-window passes, intersect the block with **each query's** causal 1024-position window. Once a lane passes the common head by a full window it reads none of that head on its window passes. Different lane positions can yield different masks on the same physical tile.

Group eligible decode rows by pass, KV head, physical block and generation. Within a threadgroup, load one K/V tile into shared storage, synchronize, and consume it with several queries. Bound query rows per group by register and threadgroup memory limits. Extra groups may reload the tile. Use the existing path for singleton groups, ineligible blocks and prefill. Lane-specific suffix tiles can follow the ordinary decode path. Hold the full step's pins across both paths.

The private path is the numerical reference. Use each row's original first key, split width (`Run::split_kps`), key order, masks, per-split online softmax operations, precision and final merge order. Sharing changes who loads a tile. It cannot insert a new reduction boundary where the common head ends. For a split that straddles the divergence, carry its row-local accumulator through both halves in the original order; emit a partial only at the original split end. Group only compatible work or fall back for that tile. Independently normalizing prefix and suffix and combining the answers can change bits even with mathematically exact cascade-attention formulas.

## Model and acceptance

Extend BlockLayout on its 3- and 4-slot bounds with two distinguishable histories, canonical sealed blocks, logical table owners, start-point ownership, and in-flight step pins. Exercise two lanes joining, diverging within and at a block boundary, sliding different windows, and dropping or cancelling in either order. On every transition check that the physical refcount agrees with explicitly enumerated owners under the chosen pin representation, every handle resolves at its recorded generation, and the referenced K/V identity matches its entire logical prefix. Check sealed content immutability, private zero tails and exact per-row visible key positions. Run reversed witnesses separately:

| Deliberate error | Expected counterexample |
| --- | --- |
| Match only local tokens and position | Two different parent histories alias the same block. |
| Append into a shared block at divergence | The other lane's sealed prefix changes. |
| Drop the donor or recycle on cancellation before GPU completion | A retained reader or step pin resolves to a reused slot. |
| Apply the donor's window to all query rows | A lane misses a visible key or reads one outside its window. |

Compare shared and private decode at every step on the same machine and tuning. Require byte-identical K/V, recurrent state and logits, unchanged golden hashes, and passing `binv`, `lanes`, `spans` and `dbatch`. Include prefixes ending at positions 63/64/65, 1023/1024/1025 and a split boundary ±1, unequal lane positions, eviction/reuse and cancellation. Force a digest collision in the lookup test. Include a changed reduction order as a negative control for the bitwise comparison. Model checking covers the finite ownership and indexing rules; the Metal comparisons cover arithmetic.

Replay the same recorded requests with sharing disabled and enabled under identical seeds, lane counts and context lengths. Record eligible and actually shared full/window bytes, peak physical KV allocation, tile loads per query group, step latency, and aggregate decode tokens/s. Compare no overlap, one lane, staggered lanes and concurrent shared heads. Report hardware counters if available alongside logical load counts. Enable by default only after the bitwise gates pass and the replay shows a repeatable benefit without material regression for singleton or unrelated lanes.
