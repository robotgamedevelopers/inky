#!/usr/bin/env bash
# tests/fleet_gate.sh — Layer 7 private-network fleet wiring & remote coding-agent card gate.
#
# Usage:
#   # Run unit tests + remote live card against an already-running private-network endpoint:
#   INKY_URL=http://100.117.59.68:18096 INKY_SERVE_KEY=... tests/fleet_gate.sh
#
#   # Or run offline self-checks (Go -race unit tests + script validation) without live server:
#   tests/fleet_gate.sh --offline
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

offline=0
url=${INKY_URL:-}
key_env=${INKY_KEY_ENV:-INKY_SERVE_KEY}
card_id=${INKY_CARD_ID:-fleet-gate-card-1}

while (( $# )); do
    case "$1" in
        --offline) offline=1; shift ;;
        --url) url=$2; shift 2 ;;
        --key-env) key_env=$2; shift 2 ;;
        --card-id) card_id=$2; shift 2 ;;
        -h|--help)
            printf '%s\n' "usage: tests/fleet_gate.sh [--offline] [--url URL] [--key-env ENV] [--card-id ID]"
            exit 0
            ;;
        *)
            printf '%s\n' "unknown option: $1" >&2
            exit 2
            ;;
    esac
done

if (( offline == 0 )) && [[ -z "$url" ]]; then
    echo "[fleet_gate] live verification requires --url or INKY_URL (use --offline for unit checks)" >&2
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

echo "[fleet_gate] running multi-turn coding-agent card ($card_id) against $url..."
(cd tests && go run ./fleetcheck -mode=fleet-card -url "$url" -key-env "$key_env" -card-id "$card_id")
