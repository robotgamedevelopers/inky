[![ci](https://github.com/robotgamedevelopers/inky/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/robotgamedevelopers/inky/actions/workflows/ci.yml)

<img width="2172" height="724" alt="image" src="https://github.com/user-attachments/assets/1ae045b5-9adc-4590-8746-bd55b8bedad8" />

Fast local inference for [**Inkling 2 Mini**](https://thinkingmachines.ai/news/inkling-2-mini/) on Apple silicon.

With **inky**, you can:

* Run Thinking Machines' Inkling 2 Mini on an M5 Mac, from the LM Studio GGUF (`16.3 GiB` mapped)
* Decode **`95.3–106 tok/s`** single-stream (`2.33x–2.59x` faster than `llama-server` on Apple M5 Max) and prefill at **`4,186–4,531 tok/s`** (`~5.4x–5.8x` faster than `llama.cpp` / LM Studio)
* Scale to **64 parallel lanes at `529.7–578 tok/s`** pure decode (`541.5 tok/s` with `--kv8`) and **`540.8 tok/s` (`7,132 cards/hr`)** in streamed multi-turn `inky serve`
* Match llama.cpp argmaxes (worst KL <= 0.01) across CI's checked reference positions: 58/58 on short-prompt decode, 24/24 after a 2380-token long-prompt prefill, and 58/58 on short-prompt decode steps
* Run many coding-agent tasks served at once from one process, each its own lane. The [performance benchmarks](docs/PERFORMANCE.md), [design](docs/DESIGN.md), and [roadmap](docs/ROADMAP.md) say how.

## Measured Performance (`Apple M5 Max MacBook Pro, 128 GB`, `Inkling-2-Mini-Q4_K_M.gguf`)

| Workload | Baseline (`llama.cpp` / unoptimized) | **inky** | Speedup / Savings |
|---|---:|---:|---:|
| 1-lane decode (short context) | `41 tok/s` (`llama-server` master) | **`95.3–106 tok/s`** (`inky bench`) | **`2.33x – 2.59x`** |
| 1-lane prefill (2.4K–2.7K prompt) | *~5.4x–5.8x slower in `llama.cpp` / LM Studio* | **`4,186–4,531 tok/s`** (`0.65 s` @ 2.7K) | **`4.2K–4.5K tok/s`** |
| 64 parallel lanes pure decode (`2K` context) | `215 tok/s` (`3.6 tok/s/lane`, `llama-server` 64 slots) | **`529.7–578 tok/s`** (`541.5 tok/s` with `--kv8`) | **`2.46x – 2.69x`** |
| 64 lanes `inky serve` (mixed 16K tool cards, streamed) | *(below `215 tok/s` short-prompt ceiling)* | **`540.8 tok/s`** (`7,132 cards/hr`, `2.26 s` p95 TTFT, `0` errors) | **`2.52x`** vs. short-prompt decode ceiling |
| Expert-grouped MoE decode (`64` lanes A/B) | `153.8 tok/s` (`379.9 ms` GPU, ungrouped) | **`435.4 tok/s`** (`104.8 ms` GPU) | **`2.83x` wall (`3.63x` GPU)**, `0`-bit diff |
| Shared prompt-prefix KV reuse (`16 × 8K` lanes) | `6,878 MiB` private KV (`430.4 s` E2E) | **`1,008 MiB` shared KV** (`4.21 s` E2E, `265.7 tok/s`) | **`6.82x` lower KV** (`5.73 GiB` saved, `102.2x` E2E) |

See [**docs/PERFORMANCE.md**](docs/PERFORMANCE.md) for full multi-lane scaling curves, `2K` to `~256K` context-length behavior, FP16 vs. `--kv8` memory footprints, and benchmark reproduction commands.

This is a fork of [erisinky](https://github.com/Eriskii/erisinky) by Eris.

Built by Glenn Fiedler ([yojimbo](https://github.com/mas-bandwidth/yojimbo), [netcode](https://github.com/mas-bandwidth/netcode)), Rowan Claude and Eris.

For more information, please ask your AI to [read the docs](docs/README.md), or see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for building and running the validation suite with or without an M5 Mac.

## Licensing (`GPL-3.0-or-later`)

`inky` (and the `inky-engine` crate) is free software distributed under the [GNU General Public License v3.0 or later](LICENSE), inherited from upstream `erisinky`. Linking, embedding, or redistributing `inky` or `inky-engine` in a binary or library requires distributing the combined work's complete corresponding source code under `GPL-3.0-or-later`-compatible terms. Calling `inky serve` as a separate process over its HTTP APIs (`/v1/chat/completions`, `/v1/messages`, `/health`) sends prompts and receives generated tokens over standard network protocols without linking `inky` into the client program, though any redistributed copy or modified version of `inky` itself remains covered by `GPL-3.0-or-later`.
