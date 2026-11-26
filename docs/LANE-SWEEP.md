# Layer 6: Candidate Machine Measurements, Replay Sweep & Default Knee

Measured on **2026-11-24** and **2026-11-25** against `Inkling-2-Mini-Q4_K_M.gguf` (`16.3 GiB` mapped, `73.2 MiB` params) on both candidate Apple Silicon hosts, covering both the synthetic scripted multi-turn benchmark fixture [`tests/replay/recorded_cards.jsonl`](../tests/replay/recorded_cards.jsonl) (§§2–5) and an authorized 2-day production capture of **`1,242` real fleet turns across `1,239` cards** replayed via [`tools/replay`](../tools/replay) on **Apple M5 Max (128 GB)** (§6).

## 1. Candidate Machine Measurements

| Profile | Chip | Unified RAM | `iogpu.wired_limit_mb` | Metal Working Set | Weights Read (`inky bw`, 1 / 2 / 4 / 8 GB) |
|---|---|---:|---:|---:|---|
| M5 Max (128 GB) | Apple M5 Max | 128 GB | `118000` | `115.23 GiB` | `574.0` / `570.2` / `522.9` / `565.8 GB/s` |
| M5 Pro (64 GB) | Apple M5 Pro | 64 GB | `0` (macOS default) | `51.84 GiB` | `289.3` / `290.4` / `287.6` / `273.0 GB/s` |

Single-stream prefill & decode (`./target/release/inky bench ... -n 128`, greedy):

| Profile | Prompt | Prompt Tokens | Prefill Time | Prefill Rate | Decode (`128 tok`) | Single-Stream Decode |
|---|---|---:|---:|---:|---:|---:|
| M5 Max (128 GB) | default (`23 tok`) | 23 | `61.5 ms` | `373.9 tok/s` | `1395.9 ms` | `91.70 tok/s` |
| M5 Max (128 GB) | [`src/tuning.rs`](../src/tuning.rs) | 1,466 | `378.3 ms` | `3874.8 tok/s` | `1312.1 ms` | `97.55 tok/s` |
| M5 Max (128 GB) | [`tools/replay/knee.go`](../tools/replay/knee.go) | 2,502 | `596.3 ms` | `4196.2 tok/s` | `1333.0 ms` | `96.02 tok/s` |
| M5 Max (128 GB) | [`src/record.rs`](../src/record.rs) | 9,174 | `3684.6 ms` | `2489.8 tok/s` | `1597.0 ms` | `80.15 tok/s` |
| M5 Pro (64 GB) | default (`23 tok`) | 23 | `101.0 ms` | `227.8 tok/s` | `1929.1 ms` | `66.35 tok/s` |
| M5 Pro (64 GB) | [`tools/replay/knee.go`](../tools/replay/knee.go) | 2,502 | `1178.6 ms` | `2122.8 tok/s` | `2077.1 ms` | `61.62 tok/s` |
| M5 Pro (64 GB) | [`src/record.rs`](../src/record.rs) | 9,174 | `4475.5 ms` | `2049.8 tok/s` | `2195.3 ms` | `58.31 tok/s` |

The M5 Max (`574.0 GB/s` peak weight read) runs ~1.98× the memory bandwidth of M5 Pro (`290.4 GB/s`), giving ~1.46× single-stream decode (`96.02` vs `61.62 tok/s` at 2.5K) and ~1.98× prefill (`4196.2` vs `2122.8 tok/s` at 2.5K).

## 2. Synthetic Scripted Benchmark Fixture (Not Fleet Traffic)

[`tests/replay/recorded_cards.jsonl`](../tests/replay/recorded_cards.jsonl) is a **synthetic scripted benchmark fixture** containing handcrafted prompts and canned tool results representing **7 multi-turn coding-agent cards** (**22 synthetic turns**, all with `read_file` / `edit_file` / `run_test` tool declarations and multi-turn `tool_calls` + `tool` result follow-ups; `12,636` prompt tokens and `1,920` completion tokens in fixture metadata). It is **not** recorded production fleet traffic; real fleet traffic requires authorized captures from `inky serve --record-dir` / [`cardproxy`](../tools/replay/cmd/cardproxy/main.go).

> **Qualification:** Sections 3–5 below record the synthetic benchmark fixture sweeps. Section 6 records the completed single-tenant Apple M5 Max (128 GB) sweep over **`1,242` authorized real production fleet turns** (`--force-stream`, `/health` footprint, `ps` RSS, and `vm_stat`/`sysctl` OS headroom).

## 3. Historical Synthetic Lane-Count Sweep (`8..128` Lanes on M5 Max 128 GB)

Server configuration:

```sh
./target/release/inky serve -c 16384 -n 8192 --lanes 128 --queue 128 --effort low --bind 127.0.0.1:18096 --key-env INKY_SERVE_KEY
```

`/health` admission telemetry at startup:
- `recommended_working_set_bytes`: `123,731,968,000` (`115.23 GiB`)
- `model_bytes` (weights + scratch): `19,352,764,416` (`18.02 GiB`)
- `per_lane_worst_case_bytes` (at `-c 16384`): `712,976,672` (`0.66 GiB`)
- `max_admitted_lanes`: `128` (note: at `-c 65536` in FP16, `MemoryBudget` computes `2.16 GiB/lane` and caps `max_admitted_lanes` to `43` on 128 GB M5 Max; with `--kv8`, per-lane 64K reserve drops to `0.586 GiB/lane` and admits `96+` lanes inside `107.52 GiB`)

Sweep command (`--duration 20s` per rung):

```sh
cd tools/replay
go run . --url http://127.0.0.1:18096 --input ../../tests/replay/recorded_cards.jsonl \
  --lanes 8,16,32,48,64,96,128 --duration 20s --key-env INKY_SERVE_KEY
```

The November 24 runner combined the time to first visible streamed output with the time to receive a complete buffered response. Seven of the fixture's 22 turns were buffered. For example, the fixture metadata on turn 2 of `card-slice-bounds` lists 168 ms TTFT and 5,979 ms total. The mixed latency percentiles below preserve what the historical runner reported. The memory column projects the admission budget from `model_bytes + active_lanes × per_lane_worst_case_bytes`.

| lanes | total_tok_s | per_lane_tok_s | mixed_latency_p50_ms | mixed_latency_p95_ms | peak_budget_estimate_bytes | cards_completed | cards_per_hour | errors | retries_429 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 62.17 | 7.77 | 2080.8 | 9987.0 | 25,056,577,792 (23.34 GiB) | 8 | 854.7 | 0 | 0 |
| 16 | 123.32 | 7.71 | 3135.9 | 11125.5 | 30,760,391,168 (28.65 GiB) | 16 | 1638.9 | 0 | 0 |
| 32 | 180.28 | 5.63 | 3829.2 | 14831.3 | 42,168,017,920 (39.27 GiB) | 32 | 2395.2 | 0 | 0 |
| 48 | 232.94 | 4.85 | 4528.9 | 18385.0 | 53,575,644,672 (49.90 GiB) | 48 | 3058.9 | 0 | 0 |
| **64** | **322.66** | **5.04** | **4639.6** | **16331.1** | **64,983,271,424 (60.52 GiB)** | **64** | **4259.0** | **0** | **0** |
| 96 | 368.61 | 3.84 | 5858.4 | 20689.1 | 87,798,524,928 (81.77 GiB) | 96 | 4848.9 | 0 | 0 |
| 128 | 401.96 | 3.14 | 7020.8 | 25447.6 | 110,613,778,432 (103.02 GiB) | 128 | 5283.9 | 0 | 0 |

## 4. Streamed Qualification on M5 Pro 64 GB & M5 Max 128 GB (Synthetic Fixture)

Measured on **2026-11-25** on an Apple M5 Pro 64 GB host (`hw.memsize = 68,719,476,736`, Metal working set `51.84 GiB`) against `Inkling-2-Mini-Q4_K_M.gguf` (`--lanes 16 -c 16384 -n 2048 --queue 64 --effort low`, `/health` `model_bytes = 19,378,225,152`, `per_lane_worst_case_bytes = 712,995,904`), replaying [`tests/replay/recorded_cards.jsonl`](../tests/replay/recorded_cards.jsonl) with `--force-stream --duration 15s --server-pid <pid> --measure-headroom`:

| lanes | total_tok_s | per_lane_tok_s | ttft_p50_ms | ttft_p95_ms | ttft_samples | peak_memory_bytes (footprint) | peak_rss_bytes (`ps`) | min_headroom_bytes (`vm_stat`) | peak_budget_estimate_bytes | cards_completed | cards_per_hour | errors | retries_429 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22.75 | 22.75 | 363.3 | 1031.4 | 6 | 26,813,632,256 (24.97 GiB) | 2,139,488,256 (1.99 GiB) | 16,100,753,408 (14.99 GiB) | 20,091,221,056 (18.71 GiB) | 2 | 331.6 | 0 | 0 |
| 4 | 41.48 | 10.37 | 1331.8 | 2219.0 | 15 | 26,839,158,552 (25.00 GiB) | 2,167,046,144 (2.02 GiB) | 15,915,466,752 (14.82 GiB) | 22,230,208,768 (20.70 GiB) | 5 | 550.6 | 0 | 0 |
| 8 | 64.99 | 8.12 | 1608.1 | 3249.8 | 25 | 26,869,436,256 (25.02 GiB) | 2,198,274,048 (2.05 GiB) | 10,472,865,792 (9.75 GiB) | 25,082,192,384 (23.36 GiB) | 8 | 877.5 | 0 | 0 |
| 16 | 126.40 | 7.90 | 1645.2 | 2177.5 | 50 | 26,885,181,328 (25.04 GiB) | 2,210,332,672 (2.06 GiB) | 10,412,605,440 (9.70 GiB) | 30,786,159,616 (28.67 GiB) | 16 | 1650.9 | 0 | 0 |

On **Apple M5 Max 128 GB** (`v0.2.15`, `-c 16384 -n 4096 --lanes 64 --queue 128 --effort low`), the corrected `--force-stream --measure-headroom --duration 15s` sweep over `recorded_cards.jsonl` (`1..64` lanes; full table in [PERFORMANCE.md §5B](PERFORMANCE.md)) reached **`540.75 tok/s` (`8.45 tok/s/lane`, `7,131.5 cards/hr`, `2,259.1 ms` p95 TTFT across `201` streamed samples, `87.13 GiB` footprint, `9.39 GiB` min OS headroom, `0` errors)** at `64` lanes.

## 5. Historical Selection & Measurement Separation

`tools/replay` separately reports:
- `ttft_p50_ms`, `ttft_p95_ms`, and `ttft_samples` from streamed turns with observable output only (with `--force-stream` available when every turn in a mixed fixture should be measured as a stream),
- `peak_memory_bytes` from `/health` (`process_footprint_bytes`),
- `peak_rss_bytes` from sampled OS process RSS (`--server-pid` / `ps -o rss= -p <pid>`),
- `min_headroom_bytes` and `host_memory_bytes` from independently sampled OS memory telemetry (`--measure-headroom` via `vm_stat` + `sysctl hw.memsize` on macOS, or `/proc/meminfo` on Linux), and
- `peak_budget_estimate_bytes` as the admission-budget upper bound. Sampling can miss peaks between polls.

When `inky serve` starts without `--lanes`, it uses the machine profile. M5 Max 128 GB requests 64 lanes, M5 Pro 64 GB requests 32, and unmatched machines request 64. An explicit `--lanes` overrides the profile. The memory budget can reduce the effective lane count. [Machine profiles](../src/tuning.rs)

---

## 6. Authorized Real Fleet Production Replay & 64K `--kv8` Lane-Count Knee (M5 Max 128 GB)

Measured on **2026-11-25** inside a dedicated single-tenant hardware window (`0` concurrent `llama-server` or CI processes) on **Apple M5 Max MacBook Pro (`18` CPU cores, `128.0 GiB` = `137,438,953,472` B unified memory)** using `inky v0.2.15` + `tools/replay`. Raw recorded JSONL files remained strictly on the benchmark host and temporary filtered files/logs were deleted immediately after sweep completion; only anonymized aggregate token statistics and numeric `tools/replay` telemetry are retained here.

### 6A. Anonymized Real Fleet Dataset Statistics (2-Day Production Capture)

- **Raw JSONL lines**: `1,339` (`259` Day 1 + `1,080` Day 2; all `POST /v1/chat/completions`)
  - **`HTTP 200` completed turns**: **`1,264`** (`232` Day 1 + `1,032` Day 2) across **`1,261` cards**
  - **`HTTP 429` / `HTTP 499` non-completed entries**: `75` (`29` `HTTP 429` + `46` `HTTP 499`, `prompt_tokens = 0`, excluded from replay)
- **64K Context Admission Filter (`max_turn_prompt_tokens <= 61,440` for `-c 65536 -n 4096`)**:
  - **Included**: **`1,239 / 1,261` cards (`98.26%`)** and **`1,242 / 1,264` turns (`98.26%`)**
  - **Excluded** (`prompt_tokens > 61,440`): `22 / 1,261` cards (`1.74%`) and `22 / 1,264` turns (`1.74%`)
  - **Recorded `finish_reason` mix (`1,242` included turns)**: `tool_calls` `1,124` (`90.50%`), `stop` `116` (`9.34%`), `length` `2` (`0.16%`)

| Dataset Slice | Metric | Count | Min | p25 | Median | Mean | p75 | p90 | p95 | p99 | Max | Sum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **All Completed (`HTTP 200`)** | `prompt_tokens` | 1,264 | 18 | 13,374.2 | **21,271.0** | **23,560.9** | 30,261.2 | 41,523.9 | 50,226.3 | 71,571.5 | **90,190** | 29,780,917 |
| **All Completed (`HTTP 200`)** | `completion_tokens` | 1,264 | 5 | 70.8 | **108.0** | **196.3** | 210.2 | 399.0 | 593.8 | 1,419.3 | **3,822** | 248,067 |
| **Filtered 64K Set (`<= 61,440`)** | `prompt_tokens` | 1,242 | 18 | 13,326.0 | **20,910.5** | **22,648.7** | 29,665.2 | 39,984.1 | 46,979.9 | 55,412.7 | **61,327** | 28,129,639 |
| **Filtered 64K Set (`<= 61,440`)** | `completion_tokens` | 1,242 | 5 | 71.0 | **108.0** | **194.8** | 209.0 | 395.0 | 582.9 | 1,383.9 | **3,822** | 241,990 |
| **Filtered 64K Set (`<= 61,440`)** | `turns_per_card` | 1,239 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 4 | 1,242 |
| **Chronological Prefix (`0..79`)** | `prompt_tokens` | 80 | 18 | 6,964.5 | 10,390.0 | 9,141.2 | 12,212.2 | 13,951.6 | 14,669.6 | 16,050.9 | 16,656 | 731,293 |
| **Chronological Prefix (`0..79`)** | `completion_tokens` | 80 | 11 | 51.0 | 75.0 | 93.0 | 112.5 | 178.4 | 247.0 | 268.8 | 287 | 7,438 |

> **Key Workload Characteristic**: Real autonomous coding fleet traffic is **heavily prefill-dominated** (`116.3:1` prompt-to-completion token ratio across the 64K set: `22,648.7` mean prompt tokens vs. `194.8` mean completion tokens, with `90.5%` ending in `tool_calls`).

### 6B. Startup `/health` & Layer 3 Paged `--kv8` Pool Telemetry (`-c 65536 --lanes 96`)

```sh
inky serve --kv8 -c 65536 -n 4096 --lanes 96 --queue 128 --effort low --bind 127.0.0.1:18961 --key-env INKY_SERVE_KEY
```

```text
[serve] Metal working set 107.52 GiB (model+scratch 18.09 GiB, headroom 4.00 GiB, avail 85.43 GiB, per-lane worst-case 0.59 GiB, max_admitted_lanes 96)
[serve] Layer 3 paged KV pool (kvb=1, --kv8): 7756 window slots (37.87 GiB) + 47401 full slots (46.29 GiB)
```

| `/health` Field | Initial Idle Value | Human-Readable | Post-Sweep Final Value |
|---|---:|---|---:|
| `ok` | `true` | Healthy | `true` |
| `kv8` / `kv_bytes_per_elem` | `true` / `1` | `Q8_0` KV (`1 B/elem`) | `true` / `1` |
| `recommended_working_set_bytes` | `115,448,725,504` | `107.52 GiB` (standard macOS cap) | `115,448,725,504` |
| `model_bytes` | `19,425,509,376` | `18.09 GiB` | `19,425,509,376` |
| `per_lane_worst_case_bytes` | `629,331,520` | **`600.18 MiB` (`0.586 GiB`) / 64K lane** | `629,331,520` |
| `max_admitted_lanes` | **`96`** | **`96 / 96` lanes admitted at 64K** | **`96`** |
| `process_footprint_bytes` | `2,155,073,112` | `2.01 GiB` | `93,987,017,624` (`87.53 GiB`) |
| `starts_spilled` | `0` | `0` disk spills | **`0`** |

Because Inkling 2 Mini uses 40 sliding-window (`1,024`-token) layers and only 8 full-attention layers, `--kv8` shrinks worst-case 64K (`65,536`-token) KV reservation to **`600.18 MiB` (`0.586 GiB`) per lane**, admitting **all `96` concurrent 64K lanes** inside macOS's standard `107.52 GiB` Metal working set without requiring elevated `iogpu.wired_limit_mb`.

### 6C. `tools/replay` Sweep Results (`8, 16, 32, 48, 64, 80, 96` Lanes, `20s` per Rung)

```sh
replay --url http://127.0.0.1:18961 --key-env INKY_SERVE_KEY \
  --input <filtered-64k-fleet.jsonl> --lanes 8,16,32,48,64,80,96 --duration 20s \
  --force-stream --server-pid $SERVER_PID --measure-headroom \
  --max-ttft-p95-ms 20000 --min-per-lane-tok-s 4.0
```

| lanes | total_tok_s | per_lane_tok_s | ttft_p50_ms | ttft_p95_ms | peak_memory_bytes | cards_completed | cards_per_hour | errors | retries_429 | ttft_samples | peak_budget_estimate_bytes | peak_rss_bytes | min_headroom_bytes | host_memory_bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 21.35 | 2.67 | 26601.8 | 28045.0 | 93594734696 | 9 | 967.3 | 0 | 0 | 9 | 24460161536 | 2517286912 | 9813000192 | 137438953472 |
| 16 | 22.15 | 1.38 | 38586.3 | 77003.6 | 93846393200 | 21 | 805.1 | 0 | 0 | 21 | 29494813696 | 2697445376 | 9448194048 | 137438953472 |
| 32 | 58.60 | 1.83 | 20089.5 | 55055.1 | 93949317680 | 33 | 1741.8 | 0 | 0 | 33 | 39564118016 | 2786721792 | 9500704768 | 137438953472 |
| 48 | 51.48 | 1.07 | 41121.7 | 100776.4 | 94172435184 | 49 | 1541.1 | 0 | 0 | 49 | 49633422336 | 2907668480 | 8839659520 | 137438953472 |
| **64** | **71.56** | **1.12** | **27714.2** | **74531.7** | **94278144800** | **66** | **2392.8** | **0** | **0** | **66** | **59702726656** | **2905604096** | **9509781504** | **137438953472** |
| 80 | 61.43 | 0.77 | 61218.6 | 119110.4 | 94266299240 | 81 | 2100.5 | 0 | 0 | 81 | 69772030976 | 3010379776 | 8959213568 | 137438953472 |
| 96 | 41.00 | 0.43 | 69022.3 | 201755.7 | 94501245968 | 97 | 1497.1 | 0 | 0 | 97 | 79841335296 | 3321217024 | 8816689152 | 137438953472 |

Human-readable summary (`GiB` & seconds):

| Lanes | Completion `tok/s` | Effective `tok/s/lane` | TTFT `p50` (`s`) | TTFT `p95` (`s`) | Cards Done | **Cards / Hour** | Errors / `429`s | Worst-Case KV Budget | Peak Footprint | Min OS Headroom |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `8` | `21.35` | `2.67` | `26.60` | `28.05` | `9` | `967.3` | `0 / 0` | `22.78 GiB` | `87.17 GiB` | `9.14 GiB` |
| `16` | `22.15` | `1.38` | `38.59` | `77.00` | `21` | `805.1` | `0 / 0` | `27.47 GiB` | `87.40 GiB` | `8.80 GiB` |
| `32` | `58.60` | `1.83` | `20.09` | `55.06` | `33` | `1,741.8` | `0 / 0` | `36.85 GiB` | `87.50 GiB` | `8.85 GiB` |
| `48` | `51.48` | `1.07` | `41.12` | `100.78` | `49` | `1,541.1` | `0 / 0` | `46.22 GiB` | `87.70 GiB` | `8.23 GiB` |
| **`64`** | **`71.56`** | **`1.12`** | **`27.71`** | **`74.53`** | **`66`** | **`2,392.8`** | **`0 / 0`** | **`55.60 GiB`** | **`87.80 GiB`** | **`8.86 GiB`** |
| `80` | `61.43` | `0.77` | `61.22` | `119.11` | `81` | `2,100.5` | `0 / 0` | `64.98 GiB` | `87.79 GiB` | `8.34 GiB` |
| `96` | `41.00` | `0.43` | `69.02` | `201.76` | `97` | `1,497.1` | `0 / 0` | `74.36 GiB` | `88.01 GiB` | `8.21 GiB` |

### 6D. `PickKnee` Cold-Prefill Burst Verdict & Batch Throughput Knee

1. **Strict Interactive Cold-Burst Gate (`--max-ttft-p95-ms 20000 --min-per-lane-tok-s 4.0`)**:
   - `tools/replay` exited `1` at the `8`-lane floor with:
     `no qualifying lane count before 8 lanes (p95 TTFT 28045.0ms > 20000.0ms ceiling)`
   - **Physical Cause**: `tools/replay` fires all `N` workers simultaneously at `t = 0` and computes `per_lane_tok_s = (completion_tokens / total_wall_s) / lanes` (amortizing full prompt prefill across short ~93-token tool-call completions). When `8` cold `~10K–13K`-token prompts (`~85K–95K` prompt tokens) arrive simultaneously at `t = 0.0 s`, M5 Max prefill (`~3.6K–4.0K tok/s`) requires `26.6 s` (`p50`) / `28.0 s` (`p95`) before the 8th lane begins decoding. Keeping cold simultaneous-burst `p95` TTFT below `20 s` on `10K–21K` prompts physically bounds uncached simultaneous prefill concurrency to **`<= 4` lanes**.
2. **Autonomous Fleet Batch Throughput Knee (`8..96` Lanes): `64` Lanes (`2,392.8 cards/hr`)**:
   - **Empirical Peak Knee**: **`64` lanes** delivers **`2,392.8 cards/hour`** (`71.56 completion tok/s` — **`2.47×` cards/hour** and **`3.35×` completion `tok/s`** over `8` lanes), with **`0` errors, `0` `429` retries, `0` spills (`starts_spilled: 0`), `55.60 GiB` worst-case 64K KV budget, `87.80 GiB` peak process footprint, and `8.86 GiB` minimum OS available headroom**.
   - **Clean Rollover Above `64` Lanes**: Stepping past `64` lanes to `80` (`2,100.5 cards/hr`, `-12.2%`, `p95 TTFT = 119.1 s`) and `96` (`1,497.1 cards/hr`, `-37.4%`, `p95 TTFT = 201.8 s`) reduces card throughput while doubling tail TTFT, though all `96` 64K `--kv8` lanes still run with `0` errors and `8.21 GiB` OS headroom.
   - **Prefix Cache Warming Dynamics**: Replaying `cards[0..K]` across consecutive rungs on one live `inky serve` process progressively warms Layer 3 shared-prefix KV checkpoints (matching production multi-worker sprints sharing system/tool schemas and repository context), driving the step-ups at `32` (`1,741.8 cards/hr`) and `64` (`2,392.8 cards/hr`) before scheduler/bandwidth saturation dominates at `80–96` lanes.

The fleet wrapper [`scripts/inky-fleet-serve.sh`](../scripts/inky-fleet-serve.sh) passes `--kv8` (`INKY_KV8=1`, `INKY_KV8=0` for exact FP16), `--lanes 64` (`INKY_LANES=32` on M5 Pro 64 GB), `--queue 128`, and `--effort low` by default; for full 64K working-context autonomous sprints on M5 Max 128 GB, set `INKY_CTX=65536 INKY_MAX_TOKENS=4096`.
