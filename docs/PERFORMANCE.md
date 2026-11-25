# Performance (`Apple M5 Max MacBook Pro, 128 GB`)

All hardware measurements in this document were collected on a single target system: **Apple M5 Max MacBook Pro (18 CPU cores, 128.0 GiB unified memory)** running `Inkling-2-Mini-Q4_K_M.gguf` (`16.3 GiB` mapped weights, `73.2 MiB` FP32/BF16 parameters).

> **Calibration & Integrity Note:** Every throughput, latency, and memory figure below is a direct hardware measurement from `inky` (`v0.2.15` integration receipts and `DESIGN.md` / `LANE-SWEEP.md` M5 Max baselines) or `llama-server` on the same machine. Opt-in non-bit-exact modes (`--kv8`, `--full-attn-budget`, sub-4-bit weights) and synthetic benchmark fixtures (`tests/replay/recorded_cards.jsonl`) are explicitly labeled where used.

---

## 1. Executive Summary & Comparison vs. `llama.cpp`

`llama.cpp` (`llama-server` master `44d5087`, Metal, 64 parallel slots, short prompts, 256 tokens/stream) flattens near `~200 tok/s` on the Apple M5 Max (128 GB):

| Streams / Slots | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llama-server` total `tok/s` | 41 | 62 | 100 | 142 | 155 | 194 | **215** |
| `llama-server` per stream (`tok/s`) | 41 | 34 | 27 | 19 | 10 | 6.4 | 3.6 |

`inky` keeps weights mapped once in unified memory, coalesces active MoE experts across lanes via `DEC_M=16, DEC_K=128` tensor-core staging, deduplicates shared prompt-prefix blocks in Layer 3 `BlockPools`, and keeps all routing/sampling on GPU:

| Metric | `llama-server` / Ungrouped | **inky** (Apple M5 Max, 128 GB) | Speedup / Savings |
|---|---:|---:|---:|
| 1-lane decode (short context) | `41 tok/s` (`llama-server`) | **`95.3–106 tok/s`** (`inky bench`) | **`2.33x – 2.59x`** |
| 1-lane prefill (2.4K–2.7K prompt) | — (not timed on M5 Max) | **`4,186–4,531 tok/s`** (`0.65 s` @ 2.7K) | **`4,186–4,531 tok/s`** (`0.65 s` @ 2.7K) |
| 64-lane pure decode (`2K` context) | `215 tok/s` (`llama-server`, `3.6 tok/s/lane`) | **`529.7–578 tok/s`** (`541.5 tok/s` `--kv8`) | **`2.46x – 2.69x`** |
| 64-lane `inky serve` (mixed 16K tool cards, streamed) | *(below `215 tok/s` short-prompt ceiling)* | **`540.8 tok/s`** (`7,132 cards/hr`, `2.26 s` p95 TTFT) | **`2.52x`** vs. pure-decode ceiling |
| Expert-grouped MoE decode (`64` lanes A/B) | `153.8 tok/s` (`379.9 ms` GPU) | **`435.4 tok/s`** (`104.8 ms` GPU) | **`2.83x` wall (`3.63x` GPU)**, `0`-bit diff |
| Shared prompt-prefix KV reuse (`16 × 8K` lanes) | `6,878 MiB` private KV (`430.4 s` E2E) | **`1,008 MiB` shared KV** (`4.21 s` E2E, `265.7 tok/s`) | **`6.82x` lower KV** (`5.73 GiB` saved, `102.2x` E2E) |

**Correctness parity:** `inky check` enforces exact argmax agreement against `llama.cpp` with worst KL `<= 0.01` across CI's reference positions (`58/58` short-prompt decode, `24/24` after a `2,380`-token prefill, and `58/58` short-prompt `encode_decode` steps).

![Aggregate Throughput vs. Parallel Lanes on Apple M5 Max (128 GB)](assets/perf-vs-llama-lanes.svg)

---

## 2. Hardware Profile & Single-Stream Bus Ceilings (`inky bw` & `inky bench`)

### System Configuration
- **Host profile**: Apple M5 Max MacBook Pro (`18` CPU cores, `128.0 GiB` = `137,438,953,472` bytes unified RAM)
- **GPU wired working set**: `107.52 GiB` (`115,448,725,504` bytes) under macOS default (`iogpu.wired_limit_mb = 0`), or **`115.23 GiB`** (`123,731,968,000` bytes) when `sudo inky tune-memory` raises `iogpu.wired_limit_mb` to `118000`
- **Raw weight-read bandwidth (`./inky bw`)**:
  - `v0.2.15` isolated run: **`555.2 GB/s`** (`1 GB`), **`537.6 GB/s`** (`2 GB`), **`544.1 GB/s`** (`4 GB`), **`517.0 GB/s`** (`8 GB`)
  - Peak observed (`DESIGN.md` / `LANE-SWEEP.md`): **`565–574 GB/s`** (`574.0 / 570.2 / 522.9 / 565.8 GB/s`)

### Single-Stream Prefill & 128-Token Greedy Decode (`./inky bench -n 128`)

| Workload | Prompt Tokens | Prefill Latency | Prefill `tok/s` | Decode (128 tok) | Decode `tok/s` | KV Cache | Prefill Scratch | Decode Scratch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Default short prompt | 23 | `62.4 ms` | 368.6 | `1,342.7 ms` | **95.33** | `36 MiB` | `83 MiB` | `4.2 MiB` |
| `tools/replay/knee.go` | 2,721 | `650.0 ms` | **4,186.0** | `1,450.8 ms` | **88.23** | `252 MiB` | `1,186 MiB` | `4.2 MiB` |
| `src/dspark.rs` | 10,076 | `2,460.8 ms` | **4,094.7** | `1,510.9 ms` | **84.72** | `496 MiB` | `1,713 MiB` | `4.2 MiB` |
| `src/sampler.rs` | 14,355 | `4,198.5 ms` | **3,419.1** | `1,581.8 ms` | **80.92** | `638 MiB` | `1,713 MiB` | `4.2 MiB` |
| `src/tokenizer.rs` | 16,094 | `3,950.7 ms` | **4,073.7** | `1,568.2 ms` | **81.62** | `672 MiB` | `1,713 MiB` | `4.2 MiB` |
| `src/record.rs` (`-c 32768`) | 20,945 | `5,800.4 ms` | **3,610.9** | `1,612.0 ms` | **79.41** | `856 MiB` | `1,713 MiB` | `4.2 MiB` |

*(In `DESIGN.md` on the same machine, short-context pipelined decode reached `106 tok/s` and 2.4K-token prefill reached `4,531 tok/s`.)*

---

## 3. Multi-Lane Scaling Across Context Sizes (`2K` to `~256K`)

### Why Context Length Shapes the Curve (KV Bandwidth Physics)
Every token in Inkling 2 Mini executes 48 layer passes: **40 sliding-window passes** (capped at a 1,024-token ring = **`160 MiB` per lane** past 1,024 tokens in FP16) plus **8 full-attention passes** (**`32 KiB` per token per lane** in FP16).

At short context (`2K`), each lane's KV cache is only `~224 MiB`, so batching 64 lanes amortizes the `~3.2 GB` of active MoE/dense weight reads per step and scales throughput `5.6x–6.2x`. At long context (`20K–42K+`), each lane reads `0.8–1.5 GiB` of private KV on every decode step, so per-lane KV bandwidth—not weight streaming—becomes the dominant term.

![Multi-Lane Throughput Scaling Across Context Sizes on Apple M5 Max (128 GB)](assets/perf-context-scaling.svg)

### 3A. FP16 KV Cache (`./inky perfgate --ref ref/long.tokens`, `v0.2.15`)

| Context | Lanes | Step Time (`ms/step`) | Total `tok/s` | Per-Lane `tok/s` | Metal `alloc MiB` (`GiB`) | Notes |
|---:|---|---:|---:|---:|---:|---|
| 2,048 | `1 (solo)` | 17.91* | 55.8* | 55.8* | `19,248.5` (`18.80 GiB`) | *Unpipelined 6-step `logits=true` sync (`bench` pipelined = **`10.49 ms / 95.33 tok/s`**) |
| 2,048 | 1 | 25.51 | 39.2 | 39.2 | `19,248.5` (`18.80 GiB`) | Batched kernel path |
| 2,048 | 8 | 48.14 | **166.2** | 20.8 | `20,857.3` (`20.37 GiB`) | |
| 2,048 | 16 | 65.48 | **244.3** | 15.3 | `22,695.9` (`22.16 GiB`) | |
| 2,048 | 32 | 85.77 | **373.1** | 11.7 | `26,373.2` (`25.76 GiB`) | |
| 2,048 | 64 | 120.81 | **529.7** | 8.3 | `33,497.9` (`32.71 GiB`) | |
| 20,480 | `1 (solo)` | 16.53 | 60.5 | 60.5 | `20,400.5` (`19.92 GiB`) | (`bench` pipelined @ 21K = **`12.59 ms / 79.41 tok/s`**) |
| 20,480 | 1 | 24.88 | 40.2 | 40.2 | `20,400.5` (`19.92 GiB`) | |
| 20,480 | 8 | 53.02 | **150.9** | 18.9 | `26,041.3` (`25.43 GiB`) | |
| 20,480 | 16 | 80.27 | **199.3** | 12.5 | `32,487.9` (`31.73 GiB`) | |
| 20,480 | 32 | 112.32 | **284.9** | 8.9 | `45,381.2` (`44.32 GiB`) | |
| 20,480 | 64 | 198.37 | **322.6** | 5.0 | `70,361.9` (`68.71 GiB`) | |
| 43,008 | `1 (solo)` | 20.37 | 49.1 | 49.1 | `21,808.5` (`21.30 GiB`) | |
| 43,008 | 1 | 26.92 | 37.1 | 37.1 | `21,808.5` (`21.30 GiB`) | |
| 43,008 | 8 | 72.58 | **110.2** | 13.8 | `32,377.3` (`31.62 GiB`) | |
| 43,008 | 16 | 152.84 | **104.7** | 6.5 | `44,455.9` (`43.41 GiB`) | |
| 43,008 | 32 | 166.81 | **191.8** | 6.0 | `67,103.4` (`65.53 GiB`) | Max power-of-2 rung under `107.5 GiB` default limit (`fitting_max = 59`) |
| 43,008 | 48 | 217.38 | **220.8** | 4.6 | `92,770.4` (`90.60 GiB`) | Fits `107.5 GiB` default wired limit (`64L` = `114.1 GiB` needs `inky tune-memory` or `--kv8`) |

### 3B. Q8_0 KV Cache (`./inky perfgate --kv8 --ref ref/long.tokens`, `v0.2.15`)

`--kv8` is an **opt-in, non-bit-exact** 8-bit K/V cache that halves both KV memory and KV read traffic per decode step. At `20K` / 64 lanes it lifts throughput **+34.5%** (`433.8 tok/s`) while saving **25.1 GiB**; at `42K` it enables all **64 lanes** inside **`65.63 GiB`** (`322.4 tok/s`):

| Context | Lanes | Step Time (`ms/step`) | Total `tok/s` | Per-Lane `tok/s` | Metal `alloc MiB` (`GiB`) | vs. FP16 `tok/s` / Memory |
|---:|---|---:|---:|---:|---:|---|
| 2,048 | `1 (solo)` | 20.50 | 48.8 | 48.8 | `19,006.3` (`18.56 GiB`) | |
| 2,048 | 1 | 27.46 | 36.4 | 36.4 | `19,006.3` (`18.56 GiB`) | |
| 2,048 | 8 | 47.85 | **167.2** | 20.9 | `19,824.1` (`19.36 GiB`) | `+0.6%` `tok/s`, `-1.0 GiB` |
| 2,048 | 16 | 62.63 | **255.4** | 16.0 | `20,758.7` (`20.27 GiB`) | `+4.5%` `tok/s`, `-1.9 GiB` |
| 2,048 | 32 | 86.09 | **371.7** | 11.6 | `22,627.9` (`22.10 GiB`) | `-0.4%` `tok/s`, `-3.7 GiB` |
| 2,048 | 64 | 118.18 | **541.5** | 8.5 | `26,249.6` (`25.63 GiB`) | `+2.2%` `tok/s`, **`-7.1 GiB`** |
| 20,480 | `1 (solo)` | 21.46 | 46.6 | 46.6 | `19,582.3` (`19.12 GiB`) | |
| 20,480 | 1 | 30.20 | 33.1 | 33.1 | `19,582.3` (`19.12 GiB`) | |
| 20,480 | 8 | 49.40 | **162.0** | 20.3 | `22,416.1` (`21.89 GiB`) | `+7.4%` `tok/s`, `-3.5 GiB` |
| 20,480 | 16 | 66.93 | **239.1** | 14.9 | `25,654.7` (`25.05 GiB`) | `+20.0%` `tok/s`, `-6.7 GiB` |
| 20,480 | 32 | 101.02 | **316.8** | 9.9 | `32,131.9` (`31.38 GiB`) | `+11.2%` `tok/s`, `-12.9 GiB` |
| 20,480 | 64 | 147.54 | **433.8** | 6.8 | `44,681.6` (`43.63 GiB`) | **`+34.5%` `tok/s`**, **`-25.1 GiB`** |
| 43,008 | `1 (solo)` | 21.21 | 47.1 | 47.1 | `20,286.3` (`19.81 GiB`) | |
| 43,008 | 1 | 28.52 | 35.1 | 35.1 | `20,286.3` (`19.81 GiB`) | |
| 43,008 | 8 | 58.75 | **136.2** | 17.0 | `25,584.1` (`24.98 GiB`) | `+23.6%` `tok/s`, `-6.6 GiB` |
| 43,008 | 16 | 79.60 | **201.0** | 12.6 | `31,638.7` (`30.90 GiB`) | **`+92.0%` `tok/s`**, `-12.5 GiB` |
| 43,008 | 32 | 126.51 | **252.9** | 7.9 | `43,747.9` (`42.72 GiB`) | **`+31.9%` `tok/s`**, **`-22.8 GiB`** |
| 43,008 | 64 | 198.50 | **322.4** | 5.0 | `67,209.6` (`65.63 GiB`) | Fits all **64 × 42K** lanes inside **`65.63 GiB`** |

### 3C. Extreme Context Scaling (`2K` to `~256K`, `inky maxctx` / `DESIGN.md`)

Measured on Apple M5 Max (128 GB) with `60 GB` allocated for lanes (`encode_decode`, bit-identical to single-stream decode):

| Context | One Stream (decode path) | Max Fitting Lanes | Aggregate Decode | Multi-Lane Gain |
|---:|---:|---:|---:|---:|
| `2K` | `93 tok/s` | 64 | **578 tok/s** | **6.2x** |
| `16K` | `79 tok/s` | 64 | **389 tok/s** | **4.9x** |
| `64K` | `53 tok/s` | 27 | **150 tok/s** | **2.9x** |
| `128K` | `33 tok/s` | 14 | **78 tok/s** | **2.4x** |
| `~256K` (`262,080` cap) | `21 tok/s` | 7 | **40 tok/s** | **1.9x** |

Prefill at `128K` to `~256K` runs at **`875 tok/s`** (`214 s` to the first token at `~256K`).

---

## 4. Expert-Grouped MoE Decode & Shared Prompt-Prefix Reuse

![Expert-Grouped MoE Decode and Shared-Prefix KV Reuse on Apple M5 Max (128 GB)](assets/perf-moe-and-prefix.svg)

### 4A. Expert-Grouped MoE Decode A/B (`dbatch`, 512-Token Context, 8 Decode Steps)

For decode-size batches (`1..=128` rows), `inky` routes active `(token, expert_slot)` assignments through `k_moe_topk_route_dec` and stages `16 × 128` coalesced quantized weight tiles (`k_moe_dec_sg4` / `k_moe_dec_sg8`), reading each touched expert **once** across all routed lanes instead of reloading weights per lane. Accumulation remains **bit-identical (`0`-bit diff)** to ungrouped execution (`INK_NO_GRP_MOE=1`):

| Lanes | `no_grp` Wall (`ms/step`) | `grp_moe` Wall (`ms/step`) | `no_grp` GPU (`ms`) | `grp_moe` GPU (`ms`) | `no_grp` `tok/s` | `grp_moe` `tok/s` | Speedup | Parity |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `19.93` | `20.13` | `18.99` | `19.15` | 50.2 | 49.7 | `0.99x` | `exact` |
| 8 | `58.36` | `44.27` | `55.25` | `41.10` | 137.1 | **180.7** | **1.32x** | `exact` |
| 16 | `109.35` | `71.30` | `104.18` | `62.62` | 146.3 | **224.4** | **1.53x** | `exact` |
| 32 | `222.48` | `102.97` | `207.98` | `78.59` | 143.8 | **310.8** | **2.16x** | `exact` |
| 64 | `416.07` | `147.00` | `379.89` | `104.77` | 153.8 | **435.4** | **2.83x** (GPU **3.63x**) | `exact` |

### 4B. Layer 3 Shared-Prefix KV Reuse (`./inky prefixbench --ref ref/long.tokens -c 10240`)

Layer 3 `ServeBlockCache` deduplicates sealed 64-token prefix blocks across lanes and shares read staging (`k_attn_grp_dec`, `G_MAX = 8`) with **100% bit-identical** vocabulary logits:

| Case | Lanes | Private KV (`MiB`) | Shared KV (`MiB`) | KV Savings | Peak GPU (`MiB`) | Private E2E (`ms`) | Shared E2E (`ms`) | Decode `ms/step` (`priv -> shrd`) | Shared Decode (`tok/s`) | Parity |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| `singleton-1x8k` | 1 | `420.0` | `600.0` | `0.70x` | `19,602.1` | `5,483.6` | `2,227.4` | `19.3 -> 19.4` | `51.6` | `exact` |
| `unrelated-4x2k` | 4 | `912.0` | `912.0` | `1.00x` | `19,651.2` | `2,735.4` | `2,565.0` | `32.4 -> 39.3` | `101.7` | `exact` |
| `staggered-card-4x2k` | 4 | `964.0` | **`320.0`** | **3.01x** | `19,715.2` | `3,002.5` | **`1,329.9`** | `32.9 -> 32.2` | **124.3** | `exact` |
| `shared-head-4x8k` | 4 | `1,716.0` | **`648.0`** | **2.65x** | `20,897.2` | `28,054.3` | **`2,810.2`** | `33.4 -> 32.0` | **124.9** | `exact` |
| `shared-head-8x8k` | 8 | `3,430.0` | **`768.0`** | **4.47x** | `22,628.0` | `205,101.1` | **`3,125.5`** | `46.3 -> 43.7` | **183.1** | `exact` |
| `shared-head-16x8k` | 16 | `6,878.0` | **`1,008.0`** | **6.82x** (`5.73 GiB` saved) | `26,112.7` | `430,382.4` | **`4,210.3`** (**102.2x** E2E) | `65.6 -> 60.2` | **265.7** | `exact` |

### 4C. Large-Context Shared-KV Decode (`inky sharedkv --ref ref/long.tokens`, `INK_DECODE=4`)

Measured on Apple M5 Max (128 GB) in [`docs/releases/v0.2.3.md`](releases/v0.2.3.md), comparing `count=1` private split reads against `G_MAX=8` grouped shared-KV tile staging at `8K` to `64K` shared prefixes:

| Shared Prefix | Mode | Lanes | `count=1` GPU Step (Attn) | `G_MAX=8` GPU Step (Attn) | Step Speedup | Attn Speedup | Parity |
|---:|---|---:|---:|---:|---:|---:|---|
| `8,192` | `fp16` | 16 | `60.91 ms` (`2.53 ms`) | `61.68 ms` (`3.30 ms`) | `0.99x` | `0.77x` | `100% bit-identical` |
| `8,192` | `fp16` | 32 | `79.47 ms` (`9.83 ms`) | `79.28 ms` (`9.64 ms`) | `1.00x` | `1.02x` | `100% bit-identical` |
| `16,384` | `fp16` | 16 | `68.56 ms` (`4.55 ms`) | `65.40 ms` (`1.39 ms`) | `1.05x` | `3.27x` | `100% bit-identical` |
| `16,384` | `fp16` | 32 | `95.63 ms` (`16.24 ms`) | `87.42 ms` (`8.02 ms`) | `1.09x` | `2.02x` | `100% bit-identical` |
| `32,768` | `fp16` | 16 | `87.15 ms` (`19.14 ms`) | `73.84 ms` (`5.83 ms`) | `1.18x` | **`3.28x`** | `100% bit-identical` |
| `32,768` | `fp16` | 32 | `129.81 ms` (`48.00 ms`) | `101.25 ms` (`19.43 ms`) | `1.28x` | `2.47x` | `100% bit-identical` |
| `65,536` | `fp16` | 16 | `118.45 ms` (`50.02 ms`) | `88.70 ms` (`20.27 ms`) | `1.34x` | `2.47x` | `100% bit-identical` |
| `65,536` | `fp16` | 32 | `188.55 ms` (`107.88 ms`) | `127.59 ms` (`46.91 ms`) | **`1.48x`** | `2.30x` | `100% bit-identical` |
| `32,768` | `--kv8` | 16 | `75.36 ms` (`9.08 ms`) | `71.27 ms` (`4.99 ms`) | `1.06x` | `1.82x` | `100% bit-identical` |
| `32,768` | `--kv8` | 32 | `108.15 ms` (`29.00 ms`) | `98.08 ms` (`18.93 ms`) | `1.10x` | `1.53x` | `100% bit-identical` |

---

## 5. End-to-End `inky serve` Multi-Turn Replay (`tools/replay`)

> **Synthetic Fixture Qualification (`tests/replay/recorded_cards.jsonl`):** `recorded_cards.jsonl` is a **synthetic scripted multi-turn coding-card fixture** (7 multi-turn cards, 22 turns with `read_file` / `edit_file` / `run_test` tool schemas, `12,636` prompt tokens and `1,920` completion tokens in metadata), not captured production fleet traffic.

### 5A. `v0.2.15` Streamed Multi-Turn Sweep (`1..64` Lanes, Apple M5 Max 128 GB)

Server configuration: `./inky serve -c 16384 -n 4096 --lanes 64 --queue 128 --effort low` (`107.52 GiB` working set, `model+scratch = 18.08 GiB`, Layer 3 paged pool `3,890` window slots / `37.99 GiB` + `23,782` full slots / `46.45 GiB`). Sweep run with `--force-stream --measure-headroom --duration 15s`:

| Lanes | Total `tok/s` | Per-Lane `tok/s` | `ttft_p50` (`ms`) | `ttft_p95` (`ms`) | Cards Completed | Equivalent `cards/hour` | Errors / `429` | TTFT Samples | Process Footprint (`GiB`) | Peak RSS (`GiB`) | Min Host Headroom (`GiB`) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 31.08 | 31.08 | `204.6` | `3,862.2` | 2 | 464.2 | `0 / 0` | 6 | `87.00` | `2.15` | `9.11` |
| 8 | **128.54** | 16.07 | `803.3` | `1,717.2` | 11 | 1,723.7 | `0 / 0` | 34 | `87.05` | `0.37` | `9.02` |
| 16 | **193.50** | 12.09 | `996.1` | `1,178.8` | 21 | 2,535.8 | `0 / 0` | 66 | `87.06` | `0.39` | `9.59` |
| 32 | **295.45** | 9.23 | `1,152.8` | `1,476.9` | 37 | 3,897.6 | `0 / 0` | 116 | `87.09` | `0.43` | `8.98` |
| **64** | **540.75** | **8.45** | `1,888.8` | **`2,259.1`** | **64** | **7,131.5** | **`0 / 0`** | **201** | **`87.13`** | `0.49` | **`9.39`** |

Throughput scales cleanly through **`64` lanes (`540.75 tok/s`, `7,131.5 cards/hr`, `2.26 s` p95 TTFT, `0` errors, `0` `429` retries)**, matching the M5 Max default `--lanes 64` profile (`src/tuning.rs`). Pure per-lane decode rate at 40+ active concurrent lanes in server logs is `16.9–17.5 tok/s`.

### 5B. Historical `8..128` Lane Sweep (`LANE-SWEEP.md`, `-c 16384 -n 8192`)

Earlier `8..128` lane exploration on Apple M5 Max (`115.23 GiB` working set under `inky tune-memory`, `model_bytes = 18.02 GiB`, `per_lane_worst_case_bytes = 0.66 GiB`; *note: this historical run mixed buffered and streamed latency and reported static admission budget estimates rather than sampled OS RSS/headroom*):

| Lanes | Total `tok/s` | Per-Lane `tok/s` | Mixed Latency p50 (`ms`) | Mixed Latency p95 (`ms`) | Peak Budget Estimate (`GiB`) | Cards Completed | Equivalent `cards/hour` | Errors / `429` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 62.17 | 7.77 | `2,080.8` | `9,987.0` | `23.34 GiB` | 8 | 854.7 | `0 / 0` |
| 16 | 123.32 | 7.71 | `3,135.9` | `11,125.5` | `28.65 GiB` | 16 | 1,638.9 | `0 / 0` |
| 32 | 180.28 | 5.63 | `3,829.2` | `14,831.3` | `39.27 GiB` | 32 | 2,395.2 | `0 / 0` |
| 48 | 232.94 | 4.85 | `4,528.9` | `18,385.0` | `49.90 GiB` | 48 | 3,058.9 | `0 / 0` |
| **64** | **322.66** | **5.04** | `4,639.6` | `16,331.1` | `60.52 GiB` | 64 | **4,259.0** | `0 / 0` |
| 96 | 368.61 | 3.84 | `5,858.4` | `20,689.1` | `81.77 GiB` | 96 | 4,848.9 | `0 / 0` |
| 128 | 401.96 | 3.14 | `7,020.8` | `25,447.6` | `103.02 GiB` | 128 | 5,283.9 | `0 / 0` |

*(With `-c 65536` per lane, `MemoryBudget` computes `2.16 GiB/lane` and automatically bounds `max_admitted_lanes` to `43` on 128 GB M5 Max.)*

---

## 6. Memory, Compression, Speculative Decode & Effort Options

By default, `inky` runs **exact FP16 KV cache** with **zero key pruning** and **zero extra weight quantization** beyond the GGUF. Several opt-in modes trade exactness or I/O for capacity, bandwidth, or latency:

- **`--kv8` (opt-in, non-bit-exact 8-bit K/V cache)**: Stores 1 byte/element instead of 2, cutting per-lane KV footprint and KV read bandwidth in half. On Apple M5 Max (`perfgate --kv8`, §3B above), `--kv8` delivers **541.5 tok/s** at `2K/64L` (`-7.1 GiB`), **433.8 tok/s** at `20K/64L` (`+34.5%`, `-25.1 GiB`), and unlocks **64 concurrent `42K` lanes** at **`322.4 tok/s`** inside **`65.63 GiB`**. See [QUALITY-REPLAY.md](QUALITY-REPLAY.md) for deterministic coding/diff/tool-call parity evaluation.
- **`--kv-disk` (CLI exact bit-identical SSD KV tier)**: Keeps exact FP16 K/V precision while holding only the newest hot rings in memory and streaming older keys from an unlinked `0600` temp file, cutting resident KV memory by half above the one-split floor at SSD-read latency cost (incompatible with `--kv8` and `--loopcd`).
- **`--full-attn-budget <tokens>` (opt-in, non-bit-exact far-key pruning, `>= 128`)**: Retains 64 sink tokens plus `(budget - 64)` recent keys across the 8 full-attention passes during prefill and Layer 2/3 decode while leaving requests shorter than `<tokens>` 100% bit-identical to unpruned default (`None`). See [releases/v0.2.14.md](releases/v0.2.14.md).
- **Sub-4-bit weight quantizations (`Q3_K`, `Q2_K`, `Q3_0`)**: Opt-in non-bit-exact GGML formats supported across `tools/inkconvert`, `src/gguf.rs`, and Metal decode/prefill/MoE kernels (`Q3_K` `3.4375 bpw` / `-23.6%` vs. `Q4_K`; `Q2_K` `2.625 bpw` / `-41.7%` vs. `Q4_K`; compact `Q3_0` `3.50 bpw` / `-36.4%` vs. `Q5_0` for `K % 32 == 0` tensors such as `ffn_down_exps`). See [releases/v0.2.14.md](releases/v0.2.14.md).
- **`--dspark <mtp.gguf> --draft-k <1..=4>` (opt-in DSpark speculative decoding)**: Verifies up to `k` draft tokens per step via GPU `k_dspark_reject` with exact Layer 3 paged rollback and automatic fallback to 1-token decode whenever `spans * (1 + k) > 128`. See [DSPARK.md](DSPARK.md) and [releases/v0.2.12.md](releases/v0.2.12.md).
- **`--effort low` (`reasoning_effort=low`)**: Default reasoning effort in `scripts/inky-fleet-serve.sh` and `tools/replay`, minimizing generated thinking tokens per card without sacrificing tool-call/diff pass rate (see [QUALITY-REPLAY.md](QUALITY-REPLAY.md)).

---

## 7. Reproducing the Benchmarks

Build the release binary and (optionally) install `inky tune-memory` once to raise `iogpu.wired_limit_mb` to `118000` (`115.23 GiB` GPU working set) while `inky` is running:

```sh
cargo build --locked --release
sudo ./target/release/inky tune-memory   # optional; restores macOS default on exit
```

Run the suites individually:

```sh
# 1. Raw weight-read memory bandwidth
./target/release/inky bw

# 2. Single-stream prefill + 128-token greedy decode across prompt sizes
./target/release/inky bench -n 128
./target/release/inky bench -f tools/replay/knee.go -n 128
./target/release/inky bench -f src/record.rs -c 32768 -n 128

# 3. Expert-grouped MoE decode A/B (0-bit diff check + wall/GPU timing)
INK_DB_AB=1 INK_PF=512 INK_DECODE=8 INK_LANES=1,8,16,32,64 \
  ./target/release/inky dbatch --ref ref/long.tokens -c 4096

# 4. Multi-lane x context-length decode & memory envelope (FP16 and --kv8)
./target/release/inky perfgate --ref ref/long.tokens
./target/release/inky perfgate --kv8 --ref ref/long.tokens

# 5. Layer 3 shared prompt-prefix KV savings & decode parity
./target/release/inky prefixbench --ref ref/long.tokens -c 10240
./target/release/inky sharedkv --ref ref/long.tokens

# 6. End-to-end streamed multi-turn serving sweep (synthetic fixture)
INKY_SERVE_KEY="$YOUR_SECRET" ./target/release/inky serve \
  -c 16384 -n 4096 --lanes 64 --queue 128 --effort low \
  --bind 127.0.0.1:18931 --key-env INKY_SERVE_KEY

cd tools/replay && INKY_SERVE_KEY="$YOUR_SECRET" go run . \
  --url http://127.0.0.1:18931 --input ../../tests/replay/recorded_cards.jsonl \
  --force-stream --server-pid <serve-pid> --measure-headroom \
  --lanes 1,8,16,32,64 --duration 15s --key-env INKY_SERVE_KEY
```
