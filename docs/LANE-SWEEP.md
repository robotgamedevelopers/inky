# Layer 6: Candidate Machine Measurements, Replay Sweep & Default Knee

Measured on **2026-11-24** against `Inkling-2-Mini-Q4_K_M.gguf` (`16.3 GiB` mapped, `73.2 MiB` params) on both candidate Apple Silicon hosts, using real multi-turn coding-agent card traffic recorded through `inky serve --record-dir` into [`tests/replay/recorded_cards.jsonl`](../tests/replay/recorded_cards.jsonl) and replayed via [`tools/replay`](../tools/replay).

## 1. Candidate Machine Measurements

| Host | Chip | Unified RAM | `iogpu.wired_limit_mb` | Metal Working Set | Weights Read (`inky bw`, 1 / 2 / 4 / 8 GB) |
|---|---|---:|---:|---:|---|
| `macbook` | Apple M5 Max | 128 GB | `118000` | `115.23 GiB` | `574.0` / `570.2` / `522.9` / `565.8 GB/s` |
| `mini-m5` | Apple M5 Pro | 64 GB | `0` (macOS default) | `51.84 GiB` | `289.3` / `290.4` / `287.6` / `273.0 GB/s` |

Single-stream prefill & decode (`./target/release/inky bench ... -n 128`, greedy):

| Host | Prompt | Prompt Tokens | Prefill Time | Prefill Rate | Decode (`128 tok`) | Single-Stream Decode |
|---|---|---:|---:|---:|---:|---:|
| `macbook` (M5 Max) | default (`23 tok`) | 23 | `61.5 ms` | `373.9 tok/s` | `1395.9 ms` | `91.70 tok/s` |
| `macbook` (M5 Max) | [`src/tuning.rs`](../src/tuning.rs) | 1,466 | `378.3 ms` | `3874.8 tok/s` | `1312.1 ms` | `97.55 tok/s` |
| `macbook` (M5 Max) | [`tools/replay/knee.go`](../tools/replay/knee.go) | 2,502 | `596.3 ms` | `4196.2 tok/s` | `1333.0 ms` | `96.02 tok/s` |
| `macbook` (M5 Max) | [`src/record.rs`](../src/record.rs) | 9,174 | `3684.6 ms` | `2489.8 tok/s` | `1597.0 ms` | `80.15 tok/s` |
| `mini-m5` (M5 Pro) | default (`23 tok`) | 23 | `101.0 ms` | `227.8 tok/s` | `1929.1 ms` | `66.35 tok/s` |
| `mini-m5` (M5 Pro) | [`tools/replay/knee.go`](../tools/replay/knee.go) | 2,502 | `1178.6 ms` | `2122.8 tok/s` | `2077.1 ms` | `61.62 tok/s` |
| `mini-m5` (M5 Pro) | [`src/record.rs`](../src/record.rs) | 9,174 | `4475.5 ms` | `2049.8 tok/s` | `2195.3 ms` | `58.31 tok/s` |

The M5 Max (`574.0 GB/s` peak weight read) runs ~1.98× the memory bandwidth of M5 Pro (`290.4 GB/s`), giving ~1.46× single-stream decode (`96.02` vs `61.62 tok/s` at 2.5K) and ~1.98× prefill (`4196.2` vs `2122.8 tok/s` at 2.5K).

## 2. Recorded Multi-Turn Replay Set

[`tests/replay/recorded_cards.jsonl`](../tests/replay/recorded_cards.jsonl) was captured from live `inky serve --record-dir` (`cards-20261124.jsonl`) across **7 multi-turn coding-agent cards** (**22 recorded turns**, all with `read_file` / `edit_file` / `run_test` tool declarations and multi-turn `tool_calls` + `tool` result follow-ups; `12,636` prompt tokens and `1,920` completion tokens recorded).

## 3. Lane-Count Sweep (`8..128` Lanes on `macbook`, M5 Max 128 GB)

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

The November 24 runner combined the time to first visible streamed output with the time to receive a complete buffered response. Seven of the fixture's 22 turns were buffered. For example, the original recording of turn 2 of `card-slice-bounds` shows 168 ms to first token and 5,979 ms to completion. The mixed latency percentiles below preserve what the runner reported. The memory column projects the admission budget from `model_bytes + active_lanes × per_lane_worst_case_bytes`.

| lanes | total_tok_s | per_lane_tok_s | mixed_latency_p50_ms | mixed_latency_p95_ms | peak_budget_estimate_bytes | cards_completed | cards_per_hour | errors | retries_429 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 62.17 | 7.77 | 2080.8 | 9987.0 | 25,056,577,792 (23.34 GiB) | 8 | 854.7 | 0 | 0 |
| 16 | 123.32 | 7.71 | 3135.9 | 11125.5 | 30,760,391,168 (28.65 GiB) | 16 | 1638.9 | 0 | 0 |
| 32 | 180.28 | 5.63 | 3829.2 | 14831.3 | 42,168,017,920 (39.27 GiB) | 32 | 2395.2 | 0 | 0 |
| 48 | 232.94 | 4.85 | 4528.9 | 18385.0 | 53,575,644,672 (49.90 GiB) | 48 | 3058.9 | 0 | 0 |
| **64** | **322.66** | **5.04** | **4639.6** | **16331.1** | **64,983,271,424 (60.52 GiB)** | **64** | **4259.0** | **0** | **0** |
| 96 | 368.61 | 3.84 | 5858.4 | 20689.1 | 87,798,524,928 (81.77 GiB) | 96 | 4848.9 | 0 | 0 |
| 128 | 401.96 | 3.14 | 7020.8 | 25447.6 | 110,613,778,432 (103.02 GiB) | 128 | 5283.9 | 0 | 0 |

## 4. Knee Selection & Default Lane Count

**64 lanes was the highest tested concurrency above the 4.0 tok/s per-lane floor.** The M5 Max delivered 322.66 tok/s in aggregate, 5.04 tok/s per lane, and 4,259 cards/hour at that setting. Aggregate throughput continued rising to 401.96 tok/s at 128 lanes, while per-lane throughput fell to 3.84 tok/s at 96 lanes and 3.14 tok/s at 128.

The historical aggregates leave the 20-second p95 TTFT target unverified. On a new sweep, `ttft_samples` counts successful streaming turns with observable output. Only these turns contribute to `ttft_p50_ms` and `ttft_p95_ms`. Buffered turns still count toward completed cards and throughput.

At 64 lanes the budget projection was 60.52 GiB. Actual process footprint and available system memory were unmeasured in this run. With a server exposing `process_footprint_bytes` through `/health`, replay reports the largest sampled process footprint in `peak_memory_bytes` and the largest admission-budget estimate in `peak_budget_estimate_bytes`. Sampling can miss peaks between polls.

When `inky serve` starts without `--lanes`, it uses the machine profile. M5 Max 128 GB requests 64 lanes, M5 Pro 64 GB requests 32, and unmatched machines request 64. An explicit `--lanes` overrides the profile. The memory budget can reduce the effective lane count. [Machine profiles](../src/tuning.rs)

The fleet wrapper passes `--lanes` explicitly. Its default of 64 matches the M5 Max profile. Set `INKY_LANES=32` to use the M5 Pro 64 GB lane count with this wrapper. [Fleet launcher](../scripts/inky-fleet-serve.sh)
