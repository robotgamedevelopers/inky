# Layer 7: Private-Network Fleet Wiring (`nova-config` & `nova-sprint`)

`inky serve` runs on Apple Silicon hosts and serves OpenAI Chat Completions (`/v1/chat/completions`) and Anthropic Messages (`/v1/messages`) over the Tailscale private network (`100.64.0.0/10` RFC 6598 CGNAT and `fc00::/7` ULA, validated by `serve::is_private_ip` in [`src/serve.rs`](../src/serve.rs)), while coding-agent harnesses and tools run on Linux fleet benches (`amd-3`, `b1`, `space`, `vision`).

## 1. Host Service Wrapper (`scripts/inky-fleet-serve.sh`)

[`scripts/inky-fleet-serve.sh`](../scripts/inky-fleet-serve.sh) launches `inky serve` bound to the host's Tailscale IPv4 address (`tailscale ip -4`, port `18096` by default so existing port `8096` services stay untouched) with `--key-env INKY_SERVE_KEY`, `--lanes 64` (the M5 Max profile default, configurable with `INKY_LANES`), `-c 16384`, `-n 8192`, and `--record-dir` enabled:

```sh
nova-secrets exec --only INKY_SERVE_KEY -- \
  /Users/glenn/inky/scripts/inky-fleet-serve.sh
```

> [!IMPORTANT]
> On macOS hosts with Application Firewall (`socketfilterfw`) enabled (such as `macbook`, `State = 1`), `target/release/inky` must be allowlisted by the host owner (`sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /path/to/inky --unblockapp /path/to/inky`) so incoming TCP connections on Tailscale `100.64.0.0/10` are not dropped after the handshake.

## 2. `nova-config` Recipes (`--dry-run` Verified Syntax)

Do not store raw API keys in `nova-config`; reference secret name `INKY_SERVE_KEY` and inject it via `nova-secrets exec`. Run with `--dry-run` first to review before committing to PostgreSQL:

```sh
# 1. Host service on macbook (Tailscale 100.117.59.68:18096, 64 lanes)
nova-config service add inky-fleet \
  --machine macbook \
  --argv '["/Users/glenn/inky/scripts/inky-fleet-serve.sh"]' \
  --keys INKY_SERVE_KEY \
  --keepalive true \
  --enabled true \
  --actor emma \
  --reason "Register M5 Max Tailscale inky serve fleet service (64 lanes)" \
  --dry-run

# 2. Provider entry: concurrency equals the configured M5 Max lane count (64)
nova-config provider add inky \
  --url http://100.117.59.68:18096 \
  --api openai \
  --key INKY_SERVE_KEY \
  --models_url http://100.117.59.68:18096/v1/models \
  --concurrency 64 \
  --note "M5 Max 128GB Tailscale inky serve (64 lanes, c=16384)" \
  --enabled true \
  --actor emma \
  --reason "Wire inky private-network fleet provider at the 64-lane M5 Max setting" \
  --dry-run

# 3. Model entry: zero API billing (price_in=0, price_out=0, price_cache_read=0)
nova-config model add inky/inkling-2-mini \
  --harness opencode \
  --price_in 0 \
  --price_out 0 \
  --price_cache_read 0 \
  --machine macbook \
  --note "Local Apple Silicon inference, zero external API spend" \
  --enabled true \
  --actor emma \
  --reason "Register inky/inkling-2-mini with local zero-cost billing" \
  --dry-run

# 4. Priority ceiling: restrict inky worker rows to flash (low-priority) tier only
nova-config friend set <inky-row> \
  --harness opencode \
  --model inky/inkling-2-mini \
  --tiers flash \
  --actor emma \
  --reason "Enforce flash-only priority ceiling for local inky worker row" \
  --dry-run
```

If routing `claude` harness workers via Anthropic Messages (`/v1/messages`), register `--api anthropic` and `--harness claude` against the same `http://100.117.59.68:18096` origin (`inky serve` serves both APIs on the shared lane pool).

## 3. Remote Harness Gate (`tests/fleet_gate.sh` & `tests/fleetcheck`)

[`tests/fleet_gate.sh`](../tests/fleet_gate.sh) verifies Go `-race` tool-execution tests offline (`--offline`) and, given `INKY_URL`, executes a live multi-turn coding-agent card (`read_file` -> `edit_file` -> `run_test`) in an isolated Go module and asserts `go test -v ./...` passes after the model's fix:

```sh
# From Linux bench (e.g. amd-3) over Tailscale:
INKY_URL=http://100.65.89.96:18096 INKY_SERVE_KEY="$KEY" ./tests/fleet_gate.sh
```

### Live Cross-Machine Proof Receipt (2026-11-24, `amd-3` -> `mini-m5`)

- **Client**: `amd-3` (`100.108.127.47`, Linux x86_64)
- **Server**: `mini-m5` (`100.65.89.96:18096`, Apple M5 Pro, Tailscale `utun`, `Firewall State = 0`)
- **Result**:
  - `GET http://100.65.89.96:18096/health` -> `200 OK` (`ok: true`, `max_admitted_lanes: 16`)
  - 4-turn coding-agent card (`fleet-card-amd3-tailscale`):
    - Turn 0: `read_file("mathutil.go")`, `read_file("mathutil_test.go")`
    - Turn 1: `edit_file("mathutil.go")` fixing `if v > hi { return lo }` -> `return hi`
    - Turn 2: `run_test("go test ./...")` -> `PASS`
    - Turn 3: final summary (`finish_reason: stop`)
  - `FLEET_CARD_PASS card_id=fleet-card-amd3-tailscale turns=4 tools=read_file,edit_file,run_test prompt_tokens=3325 cached_tokens=2301 completion_tokens=281`
  - Post-card `go test -v ./...`: `PASS ok fleetcard`
- **Operational note on `macbook` (`100.117.59.68`)**: `macbook` currently has macOS Application Firewall enabled (`socketfilterfw State = 1`), dropping non-allowlisted adhoc-signed binaries on non-loopback sockets; adding `target/release/inky` to `socketfilterfw` enables the identical `amd-3 -> macbook` Tailscale route.

### Real `claude` CLI & `opencode` CLI Sprint Harness Qualification (2026-11-24)

Both production `nova-sprint` coding-agent harnesses were qualified end-to-end against `inky serve` on `mini-m5` (`100.65.89.96:18096`) in isolated Git repositories (`DONE-WHEN: go test ./...`):

1. **`claude` CLI (`2.1.311`, Anthropic `/v1/messages`)**:
   - Bare mode (`claude --bare -p --model inkling-2-mini`): 5 turns (`1,294`–`1,923` prompt tokens/turn), read `clamp.go` & `clamp_test.go`, edited `clamp.go`, ran `go test ./...` -> `PASS ok clampcard`, `CLAUDE_EXIT=0` (`25.7s`).
   - Full mode (`claude -p --model inkling-2-mini` with isolated `CLAUDE_CONFIG_DIR`): 6 turns (`12,800`–`13,507` prompt tokens/turn), fixed `clamp.go:9`, ran `go test ./... -count=1` -> `PASS ok clampcard`, `CLAUDE_FULL_EXIT=0` (`78.1s`).
2. **`opencode` CLI (`1.18.20`, OpenAI `/v1/chat/completions`, `amd-3` -> `mini-m5` over Tailscale)**:
   - Using `nova-sprint`'s exact `OPENCODE_CONFIG_CONTENT` (`@ai-sdk/openai-compatible`, `"models": {"inkling-2-mini": {}}`): surfaced `opencode`'s default `max_tokens: 128000` payload, fixed via `clamped_chat_max` in `src/serve.rs`.
   - Re-run over Tailscale (`amd-3` -> `100.65.89.96:18096/v1`): concurrent title + build agent turns across 2 lanes (`8,278`–`8,997` prompt tokens/turn), read `clamp.go` & `clamp_test.go`, applied fix, verified `go test -count=1 ./...` -> `PASS ok clampcard 0.002s`, `OPENCODE_EXACT_EXIT=0`.
