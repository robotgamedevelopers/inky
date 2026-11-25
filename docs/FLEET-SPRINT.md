# Layer 7: Private-Network Fleet Serving & Harness Gate

`inky serve` runs on Apple Silicon hosts and serves OpenAI Chat Completions (`/v1/chat/completions`) and Anthropic Messages (`/v1/messages`) over loopback or authenticated private networks (RFC 1918, RFC 6598 CGNAT, and RFC 4193 IPv6 ULA, validated by `serve::is_private_ip` in [`src/serve.rs`](../src/serve.rs)).

## 1. Service Wrapper (`scripts/inky-fleet-serve.sh`)

[`scripts/inky-fleet-serve.sh`](../scripts/inky-fleet-serve.sh) launches `inky serve` bound to `INKY_BIND` (default `127.0.0.1:8096`, or any private-network `IP:PORT` supplied by the operator) with `--key-env INKY_SERVE_KEY`, `--lanes 64` (the M5 Max profile default, configurable with `INKY_LANES`), `-c 16384`, `-n 8192`, and `--record-dir`:

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

Each fleet card runs inside a fresh private fixture directory (`fleetcheck -workdir DIR` overrides the parent directory). The model inspects the failing acceptance test and edits `Clamp` in `mathutil.go`. Every `run_test` verification executes inside a fresh `bwrap` sandbox with a private network namespace, empty initial environment, disposable scratch directory, and a 90-second deadline. A sandbox startup error fails the card.

### Qualification Summary

- **Multi-turn `fleetcheck` card (`fleet-card`)**:
  - `GET /health` -> `200 OK` (`ok: true`)
  - 4-turn coding-agent card (`read_file` -> `edit_file` -> `run_test` -> summary): `FLEET_CARD_PASS`, `prompt_tokens=3325`, `cached_tokens=2301`, `completion_tokens=281`, post-card `go test -v ./...` `PASS`.
- **`claude` CLI (`/v1/messages`) & `opencode` CLI (`/v1/chat/completions`)**:
  - End-to-end multi-turn repository repair (`clamp.go` / `clamp_test.go`, `go test -count=1 ./...`) verified in both bare (`1.3K–1.9K` prompt tokens/turn) and full tool-prompt (`8.3K–13.5K` prompt tokens/turn) configurations.

### Recording an individual OpenCode run for replay

`tools/replay/cmd/cardproxy` is an opt-in loopback gateway for one agent run. It forwards the client's authorization unchanged to the selected Inky origin and overwrites `X-Inky-Card-Id` with the run's supplied ID, so each turn is grouped together. Use a fresh unique, non-sensitive ID and a separate gateway for every run, including retries. Every client using this loopback listener receives the same label. Use it only for the assigned run. This is a direct OpenCode route; it does not add identity propagation to a shared multi-worker key proxy.

Build the gateway on the Linux worker from `tools/replay` (`go build -o cardproxy ./cmd/cardproxy`). Start it with `./cardproxy --upstream http://<inky-host>:8096 --card-id <unique-run-id>`, and use the printed loopback origin plus `/v1` as the `baseURL` of the `@ai-sdk/openai-compatible` Inky provider in `OPENCODE_CONFIG_CONTENT`. Supply the server key through an environment variable or secret store, never in a command line or a checked-in configuration. Run OpenCode on an isolated checkout of a public Inky task. Stop the gateway after that run. The gateway does not itself provide authentication or a sandbox; the local client and Inky server retain their own controls.

Enable `--record-dir` on the Inky server in a private directory. The recorder captures request bodies, including prompts and tool results, so treat its files as sensitive even though it omits authentication headers. Preserve the server revision, configuration, unique run ID, task, agent exit and independently checked result alongside the private recording. Check for dropped or failed recorder writes and verify the IDs and turn order before replay. Do not publish raw recordings without an appropriate disclosure review. A real run is not necessarily a successful card, and one card does not establish a representative fleet workload or a lane knee.
