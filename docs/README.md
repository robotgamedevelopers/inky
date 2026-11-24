# inky

> The engine's full guide, written by its original author, Eris, for [inkling](https://github.com/Eriskii/erisinky), with the program renamed inky. The design for many lanes is [DESIGN.md](DESIGN.md) and the work in progress is [ROADMAP.md](ROADMAP.md).

## Target hardware and tuning

**The target is the Apple M5 Max MacBook Pro with 128 GB of memory**: inky is tuned and measured on it. The next target is an M5 Ultra Mac Studio. inky needs an M5 or later (Metal 4 and the GPU neural accelerators).

**The model runs exactly as published, lossless.** No change to Inkling 2 Mini: no quantization beyond the GGUF's own weights, no 8-bit cache, no key pruning, no int8 activations. The speed comes from two things only: many lanes in one step, which turns many small operations into a few wide ones, and using everything the hardware offers.

**Give the GPU the memory.** macOS lets the GPU wire about 107.5 GiB of a 128 GB Mac by default. More memory is more lanes at long context, so raise the limit:

```sh
sysctl iogpu.wired_limit_mb                  # 0 = the macOS default
sudo sysctl iogpu.wired_limit_mb=118000      # about 115 GiB for the GPU, about 13 GB left for macOS
sudo sysctl iogpu.wired_limit_mb=0           # back to the default
```

The setting lasts until the next reboot. With it, Metal's recommended working set on the 128 GB MacBook Pro is 115.2 GiB. inky does this itself in inky mode (docs/DESIGN.md). Install the helper once:

```sh
sudo inky tune-memory           # starting inky now raises the limit; stopping it restores the limit it found
sudo inky tune-memory --undo    # remove it
```

The value comes from this machine's tuning (`tunings/`; `m5-max-128gb` sets 118000).

**Memory per lane.** A lane's cache is 160 MiB for the sliding-window layers plus 32 KiB per token of context for the full-attention layers. After the 17.5 GB of weights and a few GB of scratch, about 90 GiB is left for lanes:

| context | cache per lane | lanes in about 90 GiB |
|---:|---:|---:|
| 16K | 0.7 GiB | 64 or more |
| 64K | 2.2 GiB | about 40 |
| 256K | 8.2 GiB | about 10 |
| 1M | 32 GiB | 2 |

**Watch for swapping.** Swapping ruins every measurement and slows the whole Mac. Check `vm_stat | grep Swapouts` before and after a run; the count must not grow.

## Serving: inky as a provider

Run `inky serve` on an Apple Silicon Mac with Metal 4 and an Inkling 2 Mini GGUF. Claude Code 2.1.330 uses the Anthropic Messages API through the nova-sprint v2.0.58 worker key proxy. OpenAI clients can use Chat Completions on the same server. Both APIs share the inference lanes and prompt cache.

### Build and choose a tuning

Build the release binary from the inky checkout.

```sh
cargo build --locked --release
```

The default model file is `~/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf`. Pass `-m /path/to/model.gguf` to use another location.

Inky selects a compiled-in [tuning](../tunings/) by the Mac's chip and unified memory. The shipped M5 Max 128 GB tuning sets the inky-mode wired-memory limit to 118000 MB. Other Macs use a memory-derived default that leaves the greater of 8 GiB or 10% of RAM for macOS. Size lanes and context for the host and its other workloads. Each active lane needs its own KV cache.

To enable inky mode, the Mac's owner installs the memory helper once from the checkout.

```sh
sudo ./target/release/inky tune-memory
```

The helper runs as a LaunchDaemon and raises the GPU wired-memory limit while inky is connected. It restores the prior limit after the last inky process disconnects. Inky runs under the owner's account and logs whether it entered inky mode. Without the helper, the server uses the macOS default limit. To remove it, run `sudo ./target/release/inky tune-memory --undo`.

### Start the server

Make `INKY_SERVE_KEY` available to the server and worker through their nova-secrets seats. Use the same value for both processes. Start the server on the Mac that runs the worker.

```sh
nova-secrets exec --only INKY_SERVE_KEY -- \
  ./target/release/inky serve --bind 127.0.0.1:8096 \
  --key-env INKY_SERVE_KEY --lanes 4 -c 65536 --queue 64
```

| Flag | Default | What it controls |
|---|---|---|
| `-m PATH` | LM Studio path above | Model GGUF |
| `--bind ADDR` | `127.0.0.1:8090` | Listen address. This walkthrough uses port 8096. |
| `--key-env NAME` | unset | Reads the key from the named environment variable. Accepts Bearer authorization or `x-api-key`. |
| `--lanes N` | 16 | Concurrent requests |
| `-c CTX` | 65536 | Context per lane, rounded up to a multiple of 64 tokens |
| `--queue N` | 64 | Waiting requests beyond the active lanes. A full queue returns HTTP 429 with `Retry-After: 5`. |
| `--effort E` | `medium` | Default model thinking effort |
| `-n MAX` | 16384 | Default reply budget for Chat Completions. Messages requires `max_tokens` in the request. |

Endpoints: `POST /v1/chat/completions`, `GET /v1/models` (model `inkling-2-mini`), `GET /health`. A card's next turn reuses its lane's previous prompt (`usage.prompt_tokens_details.cached_tokens`), so it prefills only what is new. That kept prompt may also start another lane: when the lane holding it is busy, a free lane copies its prefill-written prefix and the prompt-end convolution state and prefills only the suffix (`inky xcopy` gates that the copy is bit-identical to a cold prefill; a partial overlap with a kept prompt still starts cold, because the state exists only at a prompt's end).

`GET /health` is available without a key. `GET /v1/models` lists `inkling-2-mini`. `POST /v1/messages` supports streaming and tool turns, and `POST /v1/messages/count_tokens` counts the rendered input without generating a reply. OpenAI-compatible clients can use `http://127.0.0.1:8096/v1` as their base URL, with model `inkling-2-mini` and the server key. On a cache hit, Messages reports reused prompt tokens in `cache_read_input_tokens` and Chat Completions reports them in `usage.prompt_tokens_details.cached_tokens`.

Each completed request logs its queue wait (`queue_ms`) and the number of occupied lanes at admission, including its own (`lanes_busy_start`). Prefill and generation times sum the wall time of the engine batches that served the lane. Each participating lane receives the full batch duration. Prefill tokens/s counts newly evaluated prompt tokens. Generation tokens/s counts decode steps; the first output token comes from prefill, so a one-token response has zero decode time and rate.

Once a minute, including idle periods, the summary reports the time-weighted mean occupied lanes, newly prefilled and sampled output tokens per second over the whole interval, and the number of requests currently waiting for a lane. Use these interval-wide rates to distinguish slow inference from idle time between tool turns.

### Route Claude Code through nova-sprint

On the sprint's configuration host, register the provider with concurrency matching the server's lane count. Use the URL the worker can reach. Here the worker runs on the server Mac.

```sh
nova-config provider add inky --url http://127.0.0.1:8096 \
  --api anthropic --key INKY_SERVE_KEY --concurrency 4
nova-config model add inky/inkling-2-mini --harness claude
```

Put Claude Code on the worker's PATH. Set `NOVA_PG_DSN` and `NOVA_BUS_REDIS` for the sprint. In the active sprint policy, list `inky/inkling-2-mini` as a bench model for the tiers this worker should serve. The worker probes its configured models before it hooks in.

Start the v2.0.58 worker for its assigned row, replacing `<row>` with that row's name. Include `NOVA_BUS_PASSWORD` in `--only` and set `NOVA_BUS_USER` if the sprint bus requires authentication.

```sh
nova-secrets exec --only INKY_SERVE_KEY,NOVA_SPRINT_PG_PASSWORD,GH_GATE_TOKEN -- \
  nova-sprint worker --as '<row>' --harness claude \
  --providers inky=http://127.0.0.1:8096@INKY_SERVE_KEY
```

Deal the row work using model `inky/inkling-2-mini`. The worker gives Claude Code the local proxy URL, the bare model ID `inkling-2-mini`, and a per-job placeholder token. The proxy forwards the request to inky with `INKY_SERVE_KEY` as a Bearer token. Use the server origin in the provider URL. Claude Code supplies the `/v1/messages` path. Keep the secret name in the configuration and command arguments. Nova-secrets supplies the value to the server and worker.

### Troubleshoot

- **Connection refused.** Run `curl --fail http://127.0.0.1:8096/health` on the worker Mac. Inky loads the model before listening. Check its startup log for the model path and bind address.
- **HTTP 404 on Messages.** Check the inky revision for Messages support and verify the worker route points to the Anthropic provider's server origin.
- **HTTP 401 from inky or HTTP 503 with “missing API key” from the proxy.** Check that both processes started through `nova-secrets exec` with the same `INKY_SERVE_KEY`. Verify the secret name in `--key-env` and `--providers`.
- **HTTP 400 for context.** The rendered prompt, requested output budget, and two reserved tokens must fit the lane's context. Shorten the prompt or output budget, or raise `-c` if the Mac has room.
- **Worker probe times out.** The v2.0.58 worker holds the row if its Claude Code startup probe exceeds the launch timeout. Inspect the worker's probe log. On a shared M5 Pro, the probe reached Inky and generated a tool call before hitting the 20-second limit. A healthy server can still have a held worker.
- **HTTP 429 or memory pressure.** For sustained 429s, lower the worker load or add lanes if memory permits. For memory pressure, reduce lanes or context and check the inky-mode line in the startup log.

A from-scratch inference engine for **Thinking Machines' Inkling 2 Mini** (`inkling2_loop`, 26.4B total / 3.4B active MoE)
on Apple Silicon: Rust host + hand-fused Metal kernels, reading the LM Studio GGUF directly (zero-copy mmap).

## Results (Apple M5 Pro, 48 GB, `Inkling-2-Mini-Q4_K_M.gguf`, greedy)

| | LM Studio runtime (llama.cpp 2.80.1) | llama.cpp master | **inky** | speedup vs LM Studio |
|---|---:|---:|---:|---:|
| decode, short context | 33.1 tok/s | 32.7 tok/s | **63.0 tok/s** | **1.90x** |
| decode @ 2.4K context | 31.7 tok/s | – | **58.2 tok/s** | **1.84x** |
| prefill, 2394-token prompt | 381 tok/s | 409 tok/s (pp2048) | **2200 tok/s** | **5.8x** |

LM Studio numbers are from its own bundled `llama-server` (same machine, same file). Sampling mode (`--temp > 0`)
costs ~3% vs greedy (one CPU round trip per token).

**Correctness.** Logits are checked position-by-position against llama.cpp (`tools/ref_logits.cpp` dumps the reference):
61/61 argmax agreement on the decode path (max logit diff ~0.005, worst KL 0.0026), 25/25 after a batched
2379-token prefill (worst KL 0.0069; identical results whether the prompt is one chunk or several), 41/41 after a short one
(worst KL 0.0001). The tokenizer matches `llama-tokenize` token-for-token on 74K tokens of code/markdown.

## Usage

```sh
cargo build --release
./target/release/inky chat                       # interactive, multi-turn (KV cache reused across turns)
./target/release/inky run -p "Explain RoPE in one paragraph" --effort low
./target/release/inky run -f prompt.txt --greedy  # greedy = fully GPU-pipelined, fastest
./target/release/inky bench -f prompt.txt -n 256  # prefill + decode throughput
```
Options: `--effort none|minimal|low|medium|high|xhigh`, `--temp/--top-p/--top-k/--seed`, `--system`, `-c CTX`,
`--hide-thinking`, `--raw`, `--kv-disk` (half of the exact K/V cache in memory, the rest streamed from SSD: slower
generation, bit-identical output), `--kv8` (half-size 8-bit K/V cache, not exact - both in the limits section),
`-m model.gguf` (default: LM Studio's Inkling-2-Mini Q4_K_M). Requires macOS with Metal 4
(M5 neural accelerators are used for prefill GEMMs and prefill attention).

## The model (what had to be implemented)

Inkling 2 Mini is not a standard transformer. Per token: 24 physical layers, of which the middle 12 run **3 times**
(48 layer executions, 48 KV/conv cache slots), and every layer has:
- **block attention residuals**: layer inputs are a learned softmax mix (over the RMS-normalized outputs of earlier blocks);
- **no RoPE**: a learned relative-position bias `r·B[d]` (16-dim query-dependent projection, clip-normalized), plus
  per-head **attention sinks** projected from the input, QK norm *clipping*, log-scaled temperature for long contexts;
- **causal short convolutions** (width 4) on K, V (input-dependent taps on full-attention layers) and on both residual
  branch outputs, with recurrent state;
- sigmoid **output gate** on attention, sliding window (1024) on 5 of every 6 layers;
- 128-expert top-8 **MoE** with sigmoid routing + selection bias and the **SiTU** activation
  (`β·tanh(g/β)·σ(g) · λ·tanh(u/λ)`);
- in loop passes 2 and 3, **low-rank loop controllers** gate the inputs with "messages" mixed from the previous pass.

## How it is fast

Decode is memory-bound: ~3.2 GB of weights per token (top-8 experts x 48 layer executions + a 380 MB lm_head), so the
floor at the measured ~290 GB/s is ~11 ms/token. The engine runs at ~15.9 ms; llama.cpp at ~30 ms.

- **Everything stays on the GPU.** One command buffer per token (~530 dependent dispatches), no CPU round trips: routing/top-k,
  attention split-combine, norms and argmax run on the GPU; greedy decoding chains tokens GPU-to-GPU with two steps in
  flight (exact: the in-flight step always consumes a real token, so stopping never corrupts the recurrent state).
- **Fusion where the model's op soup lived.** llama.cpp builds thousands of tiny graph nodes per token for the mixes,
  clips, convolutions and controllers. Here: one depth-mix kernel (all states, all queries, optional norm, normalized
  copies for the controllers), one segmented matvec for q/gate/k/v/rel/sink/conv-taps, one attention kernel that also does
  the K/V convolution + state update + clipping + norms + relative bias + sinks, matvec epilogues that apply the post
  convolutions and residuals, MoE kernels with the activation and weighted expert sum fused in.
- **Bandwidth-tuned quantized matvecs** (Q4_K/Q6_K/Q8_0/Q5_0/bf16): x read straight from cache (no threadgroup staging,
  no barriers), tuned rows-per-simdgroup: ~285 GB/s in isolation (98% of measured peak). Dense f32 router/controller
  matrices are stored as bf16 — bit-exact, since they are bf16-representable — halving their traffic.
- **Barrier-free decode attention**: each simdgroup runs its own online softmax over 8-key blocks (4 lanes per key,
  shuffle reductions), merged once per threadgroup; adaptive split-K across threadgroups.
- **GPU page pre-wiring**: a one-time GPU pass over the mmapped weights at load. Without it, the first use of every
  rarely-used expert page takes a GPU page fault mid-generation (measured: MoE kernel 122-206 GB/s cold vs 283 warm).
- **Memory sized by need.** Metal commits a buffer in full the first time it is used (and that first use costs ~20 ms per GiB),
  so every byte of capacity is real memory. Besides the mmapped weights the engine now holds:
  - *the K/V cache* (fp16, not quantized), sized to the context. The 40 sliding-window layer executions keep exactly their
    window: a ring of 1024 keys each (160 MiB once the context reaches 1024 tokens; the ring is linear until then, so it grows
    like everything else). A prefill batch attends through a scratch copy [window history | batch keys] and leaves the last
    window in the ring. The 8 full-attention executions keep every key: 32 KiB per token. Both buffers grow in small steps
    (at most 6% headroom, in steps of 64 positions): 24 MiB for a short chat (it was 36 with a 64-position minimum headroom), 240 MiB at 2.4K tokens, ~300 MiB at 4K, ~430 MiB at 8K, 672 MiB at 16K. It was
    a fixed 832 MiB. Growing in the middle of a pipelined generation is exact (the GPU is drained first).
  - *the prefill arena* (activations of a batch, ~0.4 MiB per token) only while a prefill runs; it is released when generation
    starts. It used to be 2.8 GiB allocated at load. One-shot runs allocate and map it while they still know nothing but the
    prompt length (`Engine::prepare`), so the first-use cost stays out of the prefill; in a chat, a large paste pays it (~2%).
  Process footprint beyond the weights: ~0.38 GB for a short chat (was 3.6 GB), 1.6 GB at the peak of a 2.4K-token prefill.
- **Prefill on the M5 neural accelerators** (MetalPerformancePrimitives `matmul2d`, the whole prompt as one batch up to 4096
  tokens). What made it 2x faster again (1085 -> 2200 tok/s), roughly in order of size:
  - *One tile shape for every GEMM, one tensor op per simdgroup.* A threadgroup stages a 64-row x 32-k weight tile per k-step
    (dequantized to fp16, double-buffered) and each of its simdgroups runs its own `matmul2d` on its own 16 tokens, activations
    streamed as fp16 from device memory, accumulators in registers: ~30 TFLOPs of the ~33 the hardware does with no dequant at
    all. Simdgroups whose tokens do not exist skip the matmul, so the ragged per-expert token counts of the MoE cost
    16-token granularity instead of whole padded tiles (routing on the GPU: counting sort + tile lists + indirect dispatch).
  - *Dispatch order is a cache policy.* Metal sweeps the grid's y axis fastest. Tiles that share activations must be
    neighbours in y, or every threadgroup re-reads its input from DRAM: with the axes the wrong way round, a GEMM whose
    activations outgrow the cache drops from 28 to 16 TFLOPs (bound by memory bandwidth, not by the tensor units), which is
    what had made batches larger than 1024 tokens slower instead of faster.
  - *Whole-prompt batches.* The MoE is the bulk of the work and its efficiency is tokens-per-expert: 2394 tokens give each
    expert ~150 instead of ~64 at the old 1024-token chunks.
  - *Short-latency weight staging.* Every simdgroup of a tile waits for the staging of the next weight tile, so what matters
    is its dependency chain, not its op count: the Q4_K block header is kept in registers and a k-step is one 16-byte load.
  - *Tensor results belong in registers.* A `matmul2d` that writes to threadgroup memory is ~2x slower than the same op into
    register accumulators followed by a store; attention went from 88 to 53 ms per 1024 tokens with that alone. The
    relative-position bias is one more small tensor op (R.B^T for all distances a tile can see) and a skewed lookup.
  - *Epilogues in registers.* The accumulator layout is fixed (verified at load), so SiTU x routing weight, the half
    conversions, the attention gate and the loop controllers' gain/forget apply all happen on the accumulators before
    anything is written; the mix and controller kernels emit their fp16 outputs directly (no conversion passes).
  - *The last layer only matters for the last tokens.* Nothing downstream reads it except the final logits and 3 rows of
    conv state, so its queries, attention, output projection and FFN run for 8 tokens instead of all (K/V still for all).
  Probabilities are carried in fp16 pre-scaled by e^8 — without that, small probabilities underflow fp16 and the first-token
  KL vs llama.cpp was 0.04; with it, 0.0001.

### Things tried that did not help (kept out of the hot path)
- **Megakernel-style fusion** (inspired by [mvm](https://github.com/Bobliuuu/mvm)): "last threadgroup finishes the job"
  (atomic completion counter + device-scope fences; verified coherent on M5 over 20K dispatches) to fold the attention
  combine, router top-k and controller norms into their producers. Correct, but no measurable gain: on Metal,
  dependent dispatches in one concurrent encoder cost ~1.5 us; the remaining small-kernel time is genuine serial
  latency, which fusion cannot remove. (mvm's own README reports the same for decode: 0.97x vs CUDA graphs.)
- **Cross-phase weight prefetch into the system-level cache** (it is ~24-32 MB and serves 330-370 GB/s when warm):
  concurrent "touch" kernels steal bandwidth and the barriers wait for them; every variant was slower.
- **Aligned repack of the Q8_0/Q5_0 expert down-projections** (lossless, on the GPU at load): ~1.5% slower — the
  odd-sized blocks were never the bottleneck.

## Where the limits are (measured)

- **Decode floor.** ~3.2 GB of weights must be read per token; the M5 Pro reads ~290 GB/s, so exact single-stream
  decode cannot go below ~11 ms/token (~90 tok/s). We are at ~15.9 ms: ~12.5 ms of weight streaming plus ~3 ms of
  inherently serial small-kernel latency (depth mixes, controllers, attention at short context).
- **Speculative decoding** is the only way past that floor, and this MoE makes it expensive: routing is strongly
  correlated between neighbouring tokens (a 4-token verification pass touches 19.5 experts/layer vs 29 if random), but a
  4-token pass still costs ~1.85x a single step in bytes. Free prompt-lookup drafts accept too rarely on this model's
  reasoning text (simulated best case 1.02-1.12x); even perfect drafts cap at ~2.2x. A trained draft model (EAGLE-style)
  would be needed — not done here.
- **Prefill.** The GEMMs are 8.9 TFLOP per 1024 tokens (20.8 TFLOP for the 2394-token prompt). The tensor units do
  ~30 TFLOPs of fp16 with no weight staging at all and one op call per tile (`inky kpeak`), 33 with the weights in
  threadgroup memory: so the GEMMs alone cannot take less than ~270 ms per 1024 tokens (~3800 tok/s with everything else
  free). The engine takes ~455 ms: dense GEMM tiles at 29-30 TFLOPs (at the ceiling), the ragged MoE at 24-26 (one weight
  staging per expert tile, partially filled last tile), attention ~37 ms, and ~70 ms of memory-bound glue (depth mixes,
  gathers, expert sum, convolutions). There is no faster exact mode: `relaxed_precision` changes neither speed nor
  results, fp16 accumulators are slower and 0.1% off (`inky kmode`).
  Things that measured well in isolation and lost in the kernel:
  - *the tensor units reading the quantized weights directly* (fp16 x packed 4-bit / int8 straight from the GGUF: 32 TFLOPs
    at any tile size, exact). Q4_K scales change every 32 weights per row, so every k-step needs its partial products pulled
    out of the accumulators and rescaled; that extraction costs more than dequantizing (23 vs 30 TFLOPs);
  - int8 x int8 (57-61 TOPS raw): needs quantized activations (inexact) and the same per-block rescale;
  - wider tiles (256+ tokens per staging), larger k-steps, static extents, spreading the dequantization over more
    threads, pre-touching weight rows, reading MoE inputs through the routing permutation instead of a gathered copy.
- **K/V cache.** The payload cannot shrink without changing its numbers: the cached fp16 keys and values carry 13.7 bits of
  entropy per 16-bit value (mantissas are uniform noise, there are no zeros; xz saves 10%; `inky kvdump` writes the payload
  out). The floor for this model is min(context, 1024) x 160 KiB + context x 32 KiB, and the cache is now within 6% of it.
  A buffer cannot be left partly unbacked either: Metal makes a no-copy buffer over untouched anonymous memory resident
  in full the first time the GPU uses it (`inky lazytest`: 1 GiB resident after a one-word write), and the process
  footprint does not count those pages, so footprint numbers for such a buffer would look smaller than the memory used.
  Nothing exact gets near half of that: split into bytes, the low byte of every cached value (8 mantissa bits, exactly half
  the cache) does not compress at all (xz: 100.0%), nor do the 2 mantissa bits above it, so 10 of every 16 bits are
  irreducible and any exact store is above 62.5%; sign and exponent bring the practical floor to 85%. Every key dimension is
  read by a query head (`inky qclip`: no zero query scales), so no dimension can be left out. Also measured and ruled out: the three loop passes' K/V differ by 50-106% (no sharing);
  K and V are full rank (a rank-64 subspace loses 5-20% of the energy); the full-attention layers really use their history
  (bounding it to the last 2048 / 1024 / 512 keys leaves 22 / 18 / 17 of 25 next-token choices intact); attention there is
  diffuse (`inky kvscreen`: 69% of all keys score within 12 logits of the best one, 96% within 17), so no exact scheme can
  skip keys either.
- **Dropping keys instead of bits (measured, not in the engine yet).** `inky kvprune -f prompt.txt` prefills a prompt,
  decodes with the whole cache, then decodes the same tokens again with only a share of the full-attention layers' far keys
  (older than one window; they carry no position information, so they are an unordered set): the ones that received the
  most attention from a sample of the prompt's own queries. At 14.5K tokens, keeping half of them takes the cache from 614 to
  404 MiB with KL 0.003 mean / 0.016 worst and 64/64 next-token choices unchanged; a quarter takes it to 298 MiB (KL 0.009 /
  0.071, 63/64). On a 16.2K-token source file half costs 3-4 of 64 next-token choices (KL 0.002 / 0.03). Decode gets 3-8%
  faster (fewer keys). Random eviction is 10x worse, scores from distant queries matter most, and nothing is saved below
  ~1K tokens, where every key is inside the window.
- **`--kv-disk` halves the cache's memory without touching a number, and pays for it in generation speed.** Since the
  values cannot be made smaller, half of them are moved out of memory: while generating, each of the 48 cache slots keeps
  only its newest keys in a ring, and every older key a layer still attends to lives in an unlinked temporary file (4 KiB
  per position and slot, uncached I/O) and is read back for every token into a small staging buffer, two halves in turn so
  the GPU works on one while the next reads land in the other. Memory = hot rings + staging <= half of the in-memory cache
  at every context from 512 tokens up (process footprint while generating at 15K tokens: 952 -> 622 MB). The attention
  keeps the split boundaries of the in-memory path (the file runs one split ahead of the ring so a boundary always lies
  where both have the keys), so every partial sum covers the same keys in the same order: logits are bit-identical
  (`inky check` prints a hash of all logit bits), and greedy, sampled and multi-turn outputs are byte-identical to the
  in-memory engine. The cost is the SSD: half the cache is read per token at 7-9 GiB/s.

  | context | in memory | `--kv-disk` memory | decode in memory | decode `--kv-disk` |
  |---|---|---|---|---|
  | 2.5K tokens | 246 MiB | 118 MiB | 58.7 tok/s | 36.3 tok/s |
  | 5.5K tokens | 336 MiB | 164 MiB | 57.4 tok/s | 30.8 tok/s |
  | 15K tokens | 642 MiB | 314 MiB | 53.2 tok/s | 17.8 tok/s |

  Not covered: a prefill brings the whole cache back into memory for its duration (it sits next to the 1-1.6 GB prefill
  scratch then) and prefill speed is unchanged; every generated token writes its 192 KiB of K/V to the SSD once.
- **`--kv8`, the half-size cache, is a quantized cache.** It stores one signed byte per value and a per-vector scale that takes
  no space (its 8 bits ride in the lowest bits of the vector's first 8 bytes), so it is exactly half at every context length
  (120 instead of 240 MiB at 2.4K tokens, 336 instead of 672 MiB at 16K) at unchanged speed (the tensor units take int8
  operands natively). It is off by default because it is not the model's numbers any more. Against the llama.cpp reference
  logits: KL 0.0013 mean / 0.012 worst after a short prompt (fp16: 0.00001 / 0.00014), 0.0030 / 0.023 after a 2.4K-token
  prompt (fp16: 0.0008 / 0.0069); all 66 checked next-token choices agree; a greedy generation departs from the fp16 one
  after about 20 tokens and stays coherent. This model reacts to cache precision more than the usual 8-bit-cache experience
  suggests: the same check run token by token flips one prediction inside the prompt (KL 2.9), and in simulation
  (`INK_KVQ=kbits,vbits inky check`, which rounds the fp16 cache in place) even 12 bits per value flip two of 61.
- Multi-sequence batching would roughly double aggregate decode throughput at ~16 concurrent streams (not implemented:
  this engine targets single-stream latency).

## Layout

```
src/main.rs           CLI (chat / run / bench / check + dev tools: prof, prof_pf, kbench, kmoe, bw, debug)
src/session.rs        context management, chunked prefill, pipelined greedy decode, sampling, output rendering
src/engine.rs         weight binding, GPU memory layout, per-token decode schedule and batched prefill schedule
src/kernels.metal     decode kernels
src/prefill.metal     batched / tensor-core prefill kernels
src/bench_kernels.metal  matvec variants for micro-benchmarks
src/tokenizer.rs      byte-level BPE + hand-written GPT-4o pre-tokenizer
src/gguf.rs           zero-copy GGUF reader
tools/ref_logits.cpp  llama.cpp reference-logit dumper (validation oracle)
```

`inky prof -n POS` / `inky prof_pf -f prompt.txt` attribute GPU time per kernel class (skip-mask method);
`inky kmoepf -f prompt.txt` times the MoE GEMMs alone on the prompt's real routing, `inky ksg` the tile kernel vs
threadgroup size and fill, `inky kpeak` / `kmode` the tensor units' ceiling and precision modes, `inky kvdump` dumps the K/V payload, `inky kvscreen` measures how
concentrated full attention is over its keys, `INK_KVQ=kbits,vbits inky check` simulates a coarser cache; `INK_TIER_MIN=n` / `INK_TIER_HOT=n` make `--kv-disk` start at a smaller context / with tiny rings for tests. `INK_CHUNK=N` caps the prefill batch (default 4096 tokens). If the tensor ops' accumulator layout is not the one the register epilogues expect, the engine
falls back to a slower tensor path (`INK_NO_HGEMM=1` forces it); `INK_NO_TENSOR=1` uses simdgroup matrices instead of the
neural accelerators.

## Releasing

A pushed tag vX.Y.Z releases inky (`.github/workflows/release.yml`). The workflow checks that the tag, less its v, equals the Cargo.toml version and that `docs/releases/<tag>.md` exists, then builds `inky_<tag>_darwin_arm64` on a GitHub-hosted Apple silicon runner, packs `tunings/` as `inky_<tag>_tunings.tar.gz`, writes `SHA256SUMS` and creates the release as a prerelease with that file as its notes. It then runs `cargo publish` for the crate `inky-engine` (installed with `cargo install inky-engine`; the binary is `inky`) once the `CRATES_IO_TOKEN` Actions secret exists, and skips that step with a log line until it does.

1. Bump `version` in Cargo.toml.
2. Add `docs/releases/<tag>.md`: what ships, what you get, how to start.
3. Merge to main.
4. Push the tag from main: `git tag vX.Y.Z && git push origin vX.Y.Z`.
5. After a real run on the released binary, mark it latest: `gh release edit vX.Y.Z --prerelease=false --latest`.

## License

Copyright (C) 2026 Eris

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See [LICENSE](../LICENSE) for the full text.

The license covers this engine's source code only. The Inkling 2 Mini weights are not part of this repository and
remain under their own terms.
