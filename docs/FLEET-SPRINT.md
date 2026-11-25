# Layer 7: Private-Network Fleet Serving & Harness Gate

`inky serve` runs on Apple Silicon hosts and serves OpenAI Chat Completions (`/v1/chat/completions`) and Anthropic Messages (`/v1/messages`) over loopback or authenticated private networks (RFC 1918, RFC 6598 CGNAT, and RFC 4193 IPv6 ULA, validated by `serve::is_private_ip` in [`src/serve.rs`](../src/serve.rs)).

## 1. Service Wrapper (`scripts/inky-fleet-serve.sh`)

[`scripts/inky-fleet-serve.sh`](../scripts/inky-fleet-serve.sh) launches `inky serve` bound to `INKY_BIND` (default `127.0.0.1:8096`, or any private-network `IP:PORT` supplied by the operator) with `--key-env INKY_SERVE_KEY`, `--lanes 64` (the measured M5 Max knee from [`docs/LANE-SWEEP.md`](LANE-SWEEP.md)), `-c 16384`, `-n 8192`, and `--record-dir`:

```sh
INKY_BIND=<inky-host>:8096 INKY_SERVE_KEY="$YOUR_SECRET" \
  scripts/inky-fleet-serve.sh
```

Never hard-code bearer keys in scripts or command lines; pass only the environment variable name to `--key-env` and inject the value via the service supervisor's environment.

## 2. Client & Harness Configuration

Both OpenAI-compatible and Anthropic-compatible coding-agent harnesses share the same server origin and lane pool:

1. **OpenAI Chat Completions (`/v1/chat/completions`, e.g. `opencode`)**:
   - Base URL: `http://<inky-host>:8096/v1`
   - Model ID: `inkling-2-mini` (or `inky/inkling-2-mini`)
   - Auth header: `Authorization: Bearer <key>`
   - Recommended concurrency: match `--lanes` (`64` on M5 Max 128 GB, `32` on M5 Pro 64 GB)

2. **Anthropic Messages (`/v1/messages` & `/v1/messages/count_tokens`, e.g. `claude`)**:
   - Base URL (`ANTHROPIC_BASE_URL`): `http://<inky-host>:8096`
   - Model ID: `inkling-2-mini`
   - Auth header: `Authorization: Bearer <key>` or `x-api-key: <key>`

## 3. Remote Harness Gate (`tests/fleet_gate.sh` & `tests/fleetcheck`)

[`tests/fleet_gate.sh`](../tests/fleet_gate.sh) runs offline Go `-race` sandbox and tool-execution tests (`--offline`) and, when given `INKY_FLEET_URL` (or `--url`), executes a live multi-turn coding-agent card (`read_file` -> `edit_file` -> `run_test`) against the server and verifies `go test -v ./...` passes inside a Bubblewrap sandbox:

```sh
# Offline unit and Bubblewrap sandbox checks:
./tests/fleet_gate.sh --offline

# Live multi-turn coding-agent card verification:
INKY_FLEET_URL=http://<inky-host>:8096 INKY_SERVE_KEY="$YOUR_SECRET" ./tests/fleet_gate.sh
```

Each fleet card runs inside a fresh private fixture directory (`fleetcheck -workdir DIR` overrides the parent directory). The model inspects the failing acceptance test and edits `Clamp` in `mathutil.go`. Every `run_test` verification executes inside a fresh `bwrap` sandbox with a private network namespace, empty initial environment, disposable scratch directory, and a 90-second deadline.

### Qualification Summary

- **Multi-turn `fleetcheck` card (`fleet-card`)**:
  - `GET /health` -> `200 OK` (`ok: true`)
  - 4-turn coding-agent card (`read_file` -> `edit_file` -> `run_test` -> summary): `FLEET_CARD_PASS`, `prompt_tokens=3325`, `cached_tokens=2301`, `completion_tokens=281`, post-card `go test -v ./...` `PASS`.
- **`claude` CLI (`/v1/messages`) & `opencode` CLI (`/v1/chat/completions`)**:
  - End-to-end multi-turn repository repair (`clamp.go` / `clamp_test.go`, `go test -count=1 ./...`) verified in both bare (`1.3K–1.9K` prompt tokens/turn) and full tool-prompt (`8.3K–13.5K` prompt tokens/turn) configurations.
