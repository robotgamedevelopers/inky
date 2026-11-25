# Layer 8: Hardware Spill, Disconnect Storm & Crash Qualification

Measured on **2026-11-25** on an **Apple M5 Pro** (`64 GiB` unified memory, `51.84 GiB` Metal recommended working set, macOS `27.3` build `26C5035r`, `rustc 1.98.1`, `go1.27.4`) against `Inkling-2-Mini-Q4_K_M.gguf` (`16.3 GiB` mapped weights, `73.2 MiB` parameter buffer) using the corrected qualification harnesses in [`tests/soak.sh`](../tests/soak.sh), [`tests/soak_faults.sh`](../tests/soak_faults.sh), [`tests/soak_qualification.zsh`](../tests/soak_qualification.zsh), and [`tests/soak_client.go`](../tests/soak_client.go).

## 1. What Changed Before Hardware Re-Qualification

1. **Server & Monitor Spill Forwarding (`t-audit-soak-spill`)**:
   - [`tests/soak_qualification.zsh`](../tests/soak_qualification.zsh) (`add_spill_args`) forwards `--spill-dir <dir> --max-spill-bytes <budget>` to `inky serve` and `-spill-dir <dir> -spill-max-total-bytes <budget>` to `soak_client`.
   - `inky serve` (`GET /health` in [`src/serve.rs`](../src/serve.rs)) publishes cumulative `starts_spilled` from `ServeBlockCache::stats()`.
   - [`tests/soak_client.go`](../tests/soak_client.go) requires readable directory walks, `>= 3` post-warmup drained epochs when `--spill-dir` is enabled, `starts_spilled` strictly increasing over the initial drained snapshot, and `spill_dir_bytes_max <= spill_max_bytes` without monotonic creep across steady epochs.
   - `withID` in [`tests/soak_client.go`](../tests/soak_client.go) places its request tag inside block 0 (preserving exact `64 B` / `1024 B` / `4096 B` prompt byte lengths) rather than in the trailing `< 64`-token unsealed tail, so tagged `medium` (`4` sealed blocks) and `long` (`15` sealed blocks) prompts create distinct Layer 3 prefix checkpoints that exercise real Metal-to-SSD eviction and LRU disk garbage collection.
2. **Observed Stream Milestones in Disconnect Storm (`t-audit-soak-storm`)**:
   - `readStreamUntil` in [`tests/soak_client.go`](../tests/soak_client.go) parses SSE JSON chunks and credits `ttft` only on non-empty `content` or `reasoning_content`, `mid-thinking` only on non-empty `reasoning_content`, `mid-decode` only after `>= 2` non-empty `content` chunks, and `complete` only after both `finish_reason` and `data: [DONE]`.
   - Every target stage (`0ms`, `header-done`, `ttft`, `mid-thinking`, `mid-decode`, `complete`) must record `>= 1` qualified observation, with `0` HTTP `5xx`/`401`/unexpected statuses and `0` transport/stream/timeout errors.
3. **Active-Work Observation at `SIGKILL` (`t-audit-soak-crash`)**:
   - `crash_active_server` in [`tests/soak_qualification.zsh`](../tests/soak_qualification.zsh) extracts integer `active_lanes >= 1` from `/health` via `plutil` and sends `kill -KILL` immediately with no intervening sleep, then [`tests/soak_faults.sh`](../tests/soak_faults.sh) verifies exit status `137` (`128 + SIGKILL`).
4. **Probe Validity & Effective Token-Mix Receipts (`t-audit-soak-probes`, `t-audit-soak-token-mix`)**:
   - `runSoak` fails closed on non-positive or errored RSS samples, missing `--record-dir` / `--spill-dir` paths, or insufficient steady-state samples, and prints the effective parsed `max_tokens` list (`16,32,64` in `soak_faults.sh`, `16,64,256` default in `soak.sh`).

## 2. Live Metal-to-SSD Spill & LRU Disk Budget Qualification (`tests/soak.sh`)

Command (sanitized paths):

```sh
./tests/soak.sh \
  --inky ./target/release/inky \
  --bind 127.0.0.1:18290 \
  --duration 120s \
  --epoch 8s \
  --warmup-epochs 2 \
  --lanes 2 \
  --clients 4 \
  --ctx 2048 \
  --queue 16 \
  --seed 20261125 \
  --disconnect-pct 20 \
  --record-dir ./target/qual-record-dir \
  --record-max-bytes 4096 \
  --spill-dir ./target/qual-spill-dir \
  --spill-max-bytes 54525952
```

Server pool geometry at startup (`kvb = 2` fp16): `86` window slots (`0.84 GiB`) + `527` full slots (`1.03 GiB`), `--max-spill-bytes 54525952` (`52.00 MiB`, sizing the SSD tier to hold one `15`-block `43.03 MiB` long checkpoint or up to three `4`-block `14.16 MiB` medium checkpoints before LRU eviction).

Per-epoch drained trajectory (`active_lanes = 0`, `inflight = 0`, `queued = 0` at every boundary):

| Epoch | Role | Elapsed (`s`) | Server RSS (`bytes`) | `--record-dir` (`bytes`) | Files |
|---:|---|---:|---:|---:|---:|
| 0 | warmup | `16.3` | `3,404,316,672` (`3.17 GiB`) | `8,527` | `2` |
| 1 | warmup | `38.8` | `3,421,192,192` (`3.19 GiB`) | `7,514` | `2` |
| 2 | steady 1 | `68.2` | `3,429,089,280` (`3.19 GiB`) | `5,595` | `2` |
| 3 | steady 2 | `85.8` | `3,432,988,672` (`3.20 GiB`) | `5,984` | `2` |
| 4 | steady 3 | `108.5` | `3,445,686,272` (`3.21 GiB`) | `6,542` | `2` |
| 5 | steady 4 | `126.1` | `3,445,719,040` (`3.21 GiB`) | `6,549` | `2` |

Retained final receipt:

```text
seed=20261125
wired_limit_before=0
start=2026-11-25T10:52:33Z
end=2026-11-25T10:54:40Z
client_exit=0
wired_limit_after=0
check_recovery_wave=1 completed=2 requested=2 server_active_lanes_end=0 server_inflight_end=0 server_queued_end=0 source=server_health_and_client_outstanding
check_client=1 workload_completed=33 client_exit=0
check_drift=1 epochs_total=6 epochs_steady=4 rss_steady_first=3429089280 rss_steady_last=3445719040 rss_drift_bytes=16629760 record_dir_bytes_max=8527
starts_spilled=7 spill_dir_bytes_max=54166224 spill_max_bytes=54525952
check_wired_limit=1 before=0 after=0
prompt_lengths=short:64B medium:1024B long:4096B
max_tokens=16,64,256
requests=40 http_200=40 http_429=0 disconnected=7 inflight_max=4
effort_default=medium temperature=0 stream=true
exit=0
```

Key verified invariants:
- **Active SSD spilling & bounded LRU GC**: `starts_spilled = 7` checkpoints were evicted from Metal RAM to SSD, with `spill_dir_bytes_max = 54,166,224` (`51.66 MiB <= 54,525,952` cap) across `4` steady-state drained epochs (`>= 3` required).
- **Steady-state RSS & record bounds**: `rss_drift_bytes = +16,629,760` (`+15.86 MiB`, `+0.48%` over `3.19 GiB` initial steady RSS, plateauing to `+32 KiB` between epochs 4 and 5), `record_dir_bytes_max = 8,527` (`<= 73,728` cap).
- **Full drain & recovery wave**: `2/2` post-soak recovery streams succeeded with `active_lanes = 0`, `inflight = 0`, `queued_requests = 0`, and `iogpu.wired_limit_mb` unchanged (`0 -> 0`).

## 3. Fault Storm, `ENOSPC` & Active-Lane `SIGKILL` Qualification (`tests/soak_faults.sh`)

Command:

```sh
./tests/soak_faults.sh --inky ./target/release/inky --seed 20261125
```

### Phase `[1/4]` — Longitudinal Zero-Drift Soak & Bounded `--record-dir` Rotation

`4` lanes, `-c 4096`, `--record-max-bytes 4096`, `max_tokens=16,32,64`, `25%` random mid-stream disconnects:

```text
seed=20261125
duration_s=12
clients=4
lanes=4
queue=32
disconnect_pct=25
prompt_short_bytes=64
prompt_medium_bytes=1024
prompt_long_bytes=4096
max_tokens=16,32,64
requests=19
http_200=19
http_4xx=0
http_401=0
http_429=0
http_5xx=0
disconnected=3
transport_errors=0
timeouts=0
inflight_end=0
inflight_max=4
server_active_lanes_end=0
server_inflight_end=0
server_queued_end=0
server_health_ok=1
workload_completed=16
unexpected_status=0
stream_errors=0
recovery_completed=4
recovery_requested=4
epochs_total=3
epochs_steady=2
rss_warmup_bytes=6148898816
rss_steady_first_bytes=6156681216
rss_steady_last_bytes=6164496384
rss_steady_max_bytes=6164496384
rss_drift_bytes=7815168
record_dir_bytes_max=6030
spill_dir_bytes_max=0
drift_ok=1
inflight_source=server_health_and_client_outstanding
```

Retained segments in `--record-dir`: `cards-20261125.jsonl` (`3,471 B`, mode `0600`) and `cards-20261125.jsonl.1` (`4,551 B`, mode `0600`), bounded to active + `.1` (`record_dir_bytes_max = 6,030 <= 16,384`).

### Phase `[2/4]` — 72-Request Multi-Stage Disconnect Storm

`16` concurrent clients against `4` lanes (`72` overlapping requests across all `6` disconnect/completion stages):

```text
storm_requests=72
storm_stage_0ms=15
storm_stage_header_done=15
storm_stage_ttft=14
storm_stage_mid_thinking=14
storm_stage_mid_decode=7
storm_stage_complete=7
storm_http_200=42
storm_http_429=0
storm_http_5xx=0
storm_disconnected=65
storm_completed=7
storm_transport_errors=0
storm_stream_errors=0
drain_ok=1
server_active_lanes_end=0
server_inflight_end=0
server_queued_end=0
ref_match=1
recovery_completed=4
recovery_requested=4
```

All `72/72` storm requests reached their assigned stage (`15` `0ms` TCP closes, `15` `header-done` partial bodies, `14` `ttft` first-token disconnects, `14` `mid-thinking` reasoning-chunk disconnects, `7` `mid-decode` `>= 2`-chunk visible-text disconnects, and `7` `complete` end-to-end SSE controls), with `0` `5xx`, `0` transport errors, `0` stream errors, clean drain (`0` active / `0` inflight / `0` queued), exact seeded solo reference agreement (`ref_match = 1`), and `4/4` post-storm recovery completions.

### Phase `[3/4]` — Full-Disk (`ENOSPC`) Recorder Fault Isolation

`2 MiB` HFS+ disk image pre-filled to `0` free blocks (`ENOSPC`):

```text
requests=10
http_200=10
http_429=0
http_5xx=0
workload_completed=10
stream_errors=0
transport_errors=0
timeouts=0
drain_ok=1
record_written=0
record_write_errors=10
```

All `10` streamed chat completions succeeded (`http_200 = 10`, `http_5xx = 0`, `drain_ok = 1`) while `/health` reported `record_write_errors = 10` and `inky serve` logged `[record]` write-failure diagnostics without aborting or stalling inference.

### Phase `[4/4]` — Active-Lane `SIGKILL` (`kill -9`), Helper Recovery & Restart Determinism

```text
reference_saved=1 finish_reason=length completion_tokens=32
crash_active_lanes=1
reference_match=1 lanes_verified=4 finish_reason=length completion_tokens=32
ALL FAULT STORMS & LONGITUDINAL SOAK GATES PASSED
```

- **Observed active work & signal exit**: `crash_active_server` observed `crash_active_lanes = 1` on `/health` immediately before `kill -KILL`, and `wait` confirmed exit code `137` (`SIGKILL`).
- **Post-crash restart determinism**: Restarting `inky serve` on the same `--record-dir` matched the pre-crash seeded solo reply bit-for-bit across both a solo check and `4` concurrent lanes (`reference_match = 1`, `lanes_verified = 4`, `finish_reason = length`, `completion_tokens = 32`).
- **Memory-helper backend note (`t-audit-soak-hardware`)**: Phase `[4/4]` ran against `inky memd --fake --owner <uid> --socket <workdir>/memd.sock` via `INKY_MEMD_SOCKET` (verifying that `SIGKILL` socket close removes `<workdir>/memd.sock.saved` and logs `limit = 0`, and that server restart re-acquires the saved lease). No privileged `/Library/PrivilegedHelperTools/inky-memd` LaunchDaemon or `/private/var/db/inky/memd.sock` was installed on this M5 Pro machine (`iogpu.wired_limit_mb` remained at the macOS default `0` before and after `tests/soak.sh`); privileged `launchd` installation, root-owned kernel `iogpu.wired_limit_mb` elevation, and `sudo inky untune-memory` removal remain separate from fake-helper coverage.
