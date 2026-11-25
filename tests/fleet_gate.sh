#!/usr/bin/env bash
# tests/fleet_gate.sh — Private-network fleet wiring & remote coding-agent card gate.
#
# Usage:
#   # Run unit tests + live multi-turn card against a running loopback or private-network endpoint:
#   INKY_FLEET_URL=http://127.0.0.1:8096 INKY_SERVE_KEY=... tests/fleet_gate.sh
#
#   # Or run offline self-checks (Go -race unit tests + script validation) without a live server:
#   tests/fleet_gate.sh --offline
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

offline=0
url=${INKY_FLEET_URL:-${INKY_URL:-}}
model_name=${INKY_MODEL_NAME:-inkling-2-mini}
key_env=${INKY_KEY_ENV:-INKY_SERVE_KEY}
card_id=${INKY_CARD_ID:-fleet-gate-card-1}

while (( $# )); do
    case "$1" in
        --offline) offline=1; shift ;;
        --url) url=$2; shift 2 ;;
        --model) model_name=$2; shift 2 ;;
        --key-env) key_env=$2; shift 2 ;;
        --card-id) card_id=$2; shift 2 ;;
        -h|--help)
            printf '%s\n' "usage: tests/fleet_gate.sh [--offline] [--url URL] [--model NAME] [--key-env ENV] [--card-id ID]"
            exit 0
            ;;
        *)
            printf '%s\n' "unknown option: $1" >&2
            exit 2
            ;;
    esac
done

if (( offline == 0 )) && [[ -z "$url" ]]; then
    echo "[fleet_gate] live verification requires --url or INKY_URL / INKY_FLEET_URL (use --offline for unit checks)" >&2
    exit 2
fi

echo "[fleet_gate] running Go -race tests for tests/fleetcheck..."
(cd tests && go test -race -count=1 ./fleetcheck)

if [[ -f scripts/inky-fleet-serve.sh ]]; then
    bash -n scripts/inky-fleet-serve.sh
    echo "[fleet_gate] scripts/inky-fleet-serve.sh syntax OK"
fi

if (( offline == 1 )); then
    exit 0
fi

echo "[fleet_gate] checking /health at $url/health..."
curl -fsS "$url/health"
echo ""

echo "[fleet_gate] running multi-turn coding-agent card ($card_id, model=$model_name) against $url..."
(cd tests && go run ./fleetcheck -mode=fleet-card -url "$url" -model "$model_name" -key-env "$key_env" -card-id "$card_id")
