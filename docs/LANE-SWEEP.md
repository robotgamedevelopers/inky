# Layer 6: Candidate Machine Measurements, Replay Sweep & Default Knee

Measured on **2026-11-24** against `Inkling-2-Mini-Q4_K_M.gguf` (`16.3 GiB` mapped, `73.2 MiB` params) on both candidate Apple Silicon hosts, using the synthetic scripted multi-turn benchmark fixture [`tests/replay/recorded_cards.jsonl`](../tests/replay/recorded_cards.jsonl) and replayed via [`tools/replay`](../tools/replay).

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

> **Qualification:** This fixture is synthetic benchmark traffic, not verified real fleet card traffic. The November 24 historical replay mixed streamed and buffered latency, and the reported memory values were static admission estimates rather than sampled footprint/RSS or measured OS headroom. Those historical numbers do not establish a fleet-derived lane knee or verified latency and memory limits. Repeat the M5 Max 64-lane sweep using authorized real fleet traffic and corrected measurements in a dedicated offline M5 Max window before locking fleet regression thresholds.

## 3. Lane-Count Sweep (`8..128` Lanes on M5 Max 128 GB)

Server configuration:

```sh
./target/release/inky serve -c 16384 -n 8192 --lanes 128 --queue 128 --effort low --bind 127.0.0.1:18096 --key-env INKY_SERVE_KEY
```

`/health` admission telemetry at startup:
- `recommended_working_set_bytes`: `123,731,968,000` (`115.23 GiB`)
- `model_bytes` (weights + scratch): `19,352,764,416` (`18.02 GiB`)
- `per_lane_worst_case_bytes` (at `-c 16384`): `712,976,672` (`0.66 GiB`)
- `max_admitted_lanes`: `128` (note: at `-c 65536`, `MemoryBudget` computes `2.16 GiB/lane` and caps `max_admitted_lanes` to `43` on 128 GB M5 Max)

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

## 4. Streamed Qualification on M5 Pro 64 GB (`1..16` Lanes, Synthetic Fixture)

Measured on **2026-11-25** on an Apple M5 Pro 64 GB host (`hw.memsize = 68,719,476,736`, Metal working set `51.84 GiB`) against `Inkling-2-Mini-Q4_K_M.gguf` (`--lanes 16 -c 16384 -n 2048 --queue 64 --effort low`, `/health` `model_bytes = 19,378,225,152`, `per_lane_worst_case_bytes = 712,995,904`), replaying [`tests/replay/recorded_cards.jsonl`](../tests/replay/recorded_cards.jsonl) with `--force-stream --duration 15s --server-pid <pid> --measure-headroom`:

| lanes | total_tok_s | per_lane_tok_s | ttft_p50_ms | ttft_p95_ms | ttft_samples | peak_memory_bytes (footprint) | peak_rss_bytes (`ps`) | min_headroom_bytes (`vm_stat`) | peak_budget_estimate_bytes | cards_completed | cards_per_hour | errors | retries_429 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22.75 | 22.75 | 363.3 | 1031.4 | 6 | 26,813,632,256 (24.97 GiB) | 2,139,488,256 (1.99 GiB) | 16,100,753,408 (14.99 GiB) | 20,091,221,056 (18.71 GiB) | 2 | 331.6 | 0 | 0 |
| 4 | 41.48 | 10.37 | 1331.8 | 2219.0 | 15 | 26,839,158,552 (25.00 GiB) | 2,167,046,144 (2.02 GiB) | 15,915,466,752 (14.82 GiB) | 22,230,208,768 (20.70 GiB) | 5 | 550.6 | 0 | 0 |
| 8 | 64.99 | 8.12 | 1608.1 | 3249.8 | 25 | 26,869,436,256 (25.02 GiB) | 2,198,274,048 (2.05 GiB) | 10,472,865,792 (9.75 GiB) | 25,082,192,384 (23.36 GiB) | 8 | 877.5 | 0 | 0 |
| 16 | 126.40 | 7.90 | 1645.2 | 2177.5 | 50 | 26,885,181,328 (25.04 GiB) | 2,210,332,672 (2.06 GiB) | 10,412,605,440 (9.70 GiB) | 30,786,159,616 (28.67 GiB) | 16 | 1650.9 | 0 | 0 |

Across `1..16` lanes on M5 Pro 64 GB, all `6 / 15 / 25 / 50` streamed TTFT samples stayed well below the 20 s ceiling (`2,177.5 ms` p95 at 16 lanes), `errors = 0`, `retries_429 = 0`, sampled `process_footprint_bytes` peaked at `25.04 GiB` (`peak_rss_bytes` at `2.06 GiB`), and independently measured `vm_stat` + `sysctl` headroom stayed at least `9.70 GiB` out of `64.00 GiB`. Because this run uses the synthetic `recorded_cards.jsonl` fixture up to 16 lanes on M5 Pro, the 64-lane M5 Max rerun (`t-audit-replay-capacity-rerun`) and a fleet-derived knee from authorized recorded fleet traffic (`t-au-replay-real`) still require a dedicated offline M5 Max window with real fleet captures.

## 5. Historical Selection & Default Lane Count

The following selection describes the historical fixture run and is not a verified operational knee.

**64 lanes was the highest tested concurrency above the 4.0 tok/s per-lane floor.** The M5 Max delivered 322.66 tok/s in aggregate, 5.04 tok/s per lane, and 4,259 cards/hour at that setting. Aggregate throughput continued rising to 401.96 tok/s at 128 lanes, while per-lane throughput fell to 3.84 tok/s at 96 lanes and 3.14 tok/s at 128.

The historical aggregates leave the 20-second p95 TTFT target unverified. On a new sweep, `ttft_samples` counts successful streaming turns with observable output. Only these turns contribute to `ttft_p50_ms` and `ttft_p95_ms`. Buffered turns still count toward completed cards and throughput.

At 64 lanes the budget projection was 60.52 GiB. Actual process footprint, OS RSS, and available system memory were unmeasured in that November 24 run. `tools/replay` now separately reports:
- `ttft_p50_ms`, `ttft_p95_ms`, and `ttft_samples` from streamed turns with observable output only (with `--force-stream` available when every turn in a mixed fixture should be measured as a stream),
- `peak_memory_bytes` from `/health` (`process_footprint_bytes`),
- `peak_rss_bytes` from sampled OS process RSS (`--server-pid` / `ps -o rss= -p <pid>`),
- `min_headroom_bytes` and `host_memory_bytes` from independently sampled OS memory telemetry (`--measure-headroom` via `vm_stat` + `sysctl hw.memsize` on macOS, or `/proc/meminfo` on Linux), and
- `peak_budget_estimate_bytes` as the admission-budget upper bound. Sampling can miss peaks between polls.

When `inky serve` starts without `--lanes`, it uses the machine profile. M5 Max 128 GB requests 64 lanes, M5 Pro 64 GB requests 32, and unmatched machines request 64. An explicit `--lanes` overrides the profile. The memory budget can reduce the effective lane count. [Machine profiles](../src/tuning.rs)

- **Unconstrained scaling (`PickKnee` with `--min-gain 0.05`)**: Total throughput continues scaling cleanly at short-to-medium prompt sizes through `128` lanes (`401.96 tok/s`, `5,283.9 cards/hr`, `+21.8%/doubling` from `96` to `128`), because per-lane KV cache traffic at `< 2K` tokens is small relative to weight bandwidth.
- **Selected Operational Knee (`PickKnee` with `--max-ttft-p95-ms 20000 --min-per-lane-tok-s 4.0`): `64` lanes**:
  1. **Marginal gain inflection**: `48 -> 64` jumps `+38.5%` (`232.94 -> 322.66 tok/s`, `+92.8%/doubling`), whereas `64 -> 96` drops to `+14.2%` (`+24.3%/doubling`) and `96 -> 128` drops to `+9.0%` (`+21.8%/doubling`).
  2. **Worker probe & latency SLO**: At `64` lanes, `ttft_p95_ms = 16,331.1 ms` stays under the `20,000 ms` (`20 s`) worker startup probe ceiling (`docs/README.md`) and `per_lane_tok_s = 5.04` stays above `4.0 tok/s`; `96` (`20,689.1 ms`, `3.84 tok/s`) and `128` (`25,447.6 ms`, `3.14 tok/s`) cross both thresholds.
  3. **Working-set headroom**: At `64` lanes (`60.52 GiB` peak worst-case at `-c 16384`), `54.7 GiB` of working set remains free for macOS/spill/longer turns, whereas `128` lanes consumes `103.02 GiB` (`89%` of the `115.23 GiB` working set). When running at higher per-lane context (`-c 65536`), `MemoryBudget` automatically caps admission to `43` lanes on M5 Max 128 GB.

The fleet wrapper passes `--lanes` explicitly. Its default of 64 matches the M5 Max profile. Set `INKY_LANES=32` to use the M5 Pro 64 GB lane count with this wrapper. [Fleet launcher](../scripts/inky-fleet-serve.sh)
