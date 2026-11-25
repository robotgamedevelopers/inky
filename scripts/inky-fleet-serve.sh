#!/usr/bin/env bash
# scripts/inky-fleet-serve.sh — Launch wrapper for `nova-config service` running `inky serve`
# bound to the host's Tailscale private network IP with opt-in card traffic recording.
#
# Usage:
#   nova-secrets exec --only INKY_SERVE_KEY -- scripts/inky-fleet-serve.sh [extra inky serve flags...]
#
# Environment overrides:
#   INKY_BIN           Path to inky binary (default: <repo>/target/release/inky)
#   INKY_MODEL         Path to GGUF (default: ~/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf)
#   INKY_BIND          Listen address (default: <tailscale-ipv4>:18096)
#   INKY_LANES         Requested lane count (default: 64, the M5 Max profile default)
#   INKY_CTX           Context per lane (default: 16384)
#   INKY_QUEUE         Bounded admission queue depth (default: 128, matching docs/LANE-SWEEP.md)
#   INKY_EFFORT        Default reasoning effort (default: low, matching docs/LANE-SWEEP.md)
#   INKY_MAX_TOKENS    Default reply token budget (default: 8192, must be < INKY_CTX)
#   INKY_RECORD_CARDS  Set to 1 to record card traffic into <repo>/logs/cards (default: 0, disabled)
#   INKY_RECORD_DIR    Explicit directory for cards-YYYYMMDD.jsonl (enables recording when non-empty)
#   INKY_KEY_ENV       Env var carrying bearer API key (default: INKY_SERVE_KEY)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INKY_BIN=${INKY_BIN:-$REPO_ROOT/target/release/inky}
INKY_MODEL=${INKY_MODEL:-$HOME/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf}
INKY_LANES=${INKY_LANES:-64}
INKY_CTX=${INKY_CTX:-16384}
INKY_QUEUE=${INKY_QUEUE:-128}
INKY_EFFORT=${INKY_EFFORT:-low}
INKY_MAX_TOKENS=${INKY_MAX_TOKENS:-8192}
INKY_KEY_ENV=${INKY_KEY_ENV:-INKY_SERVE_KEY}

if [[ ! -x "$INKY_BIN" ]]; then
    printf '%s\n' "[fleet-serve] missing executable inky binary at $INKY_BIN" >&2
    exit 2
fi

if [[ -z "${INKY_BIND:-}" ]]; then
    if command -v tailscale >/dev/null 2>&1; then
        TS_IP=$(tailscale ip -4 2>/dev/null | head -n 1 || true)
    else
        TS_IP=""
    fi
    if [[ -z "$TS_IP" ]]; then
        printf '%s\n' "[fleet-serve] INKY_BIND is unset and 'tailscale ip -4' returned no IPv4 address" >&2
        exit 2
    fi
    INKY_BIND="${TS_IP}:18096"
fi

if [[ -z "${!INKY_KEY_ENV:-}" ]]; then
    printf '%s\n' "[fleet-serve] required key env var $INKY_KEY_ENV is unset or empty (run under nova-secrets exec)" >&2
    exit 2
fi

RECORD_DIR=""
if [[ -n "${INKY_RECORD_DIR:-}" ]]; then
    RECORD_DIR="$INKY_RECORD_DIR"
else
    case "${INKY_RECORD_CARDS:-0}" in
        0) RECORD_DIR="" ;;
        1) RECORD_DIR="$REPO_ROOT/logs/cards" ;;
        *)
            printf '%s\n' "[fleet-serve] INKY_RECORD_CARDS must be 0 or 1 (got ${INKY_RECORD_CARDS})" >&2
            exit 2
            ;;
    esac
fi

RECORD_ARGS=()
if [[ -n "$RECORD_DIR" ]]; then
    mkdir -p "$RECORD_DIR"
    chmod 700 "$RECORD_DIR"
    RECORD_ARGS=(--record-dir "$RECORD_DIR")
fi

exec "$INKY_BIN" serve \
    -m "$INKY_MODEL" \
    -c "$INKY_CTX" \
    -n "$INKY_MAX_TOKENS" \
    --lanes "$INKY_LANES" \
    --queue "$INKY_QUEUE" \
    --effort "$INKY_EFFORT" \
    --bind "$INKY_BIND" \
    --key-env "$INKY_KEY_ENV" \
    ${RECORD_ARGS[@]+"${RECORD_ARGS[@]}"} \
    "$@"
