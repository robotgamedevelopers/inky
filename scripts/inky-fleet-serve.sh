#!/usr/bin/env bash
# scripts/inky-fleet-serve.sh — Launch wrapper for `inky serve` bound to loopback or an
# operator-specified private-network address with bearer authentication and opt-in recording.
#
# Usage:
#   INKY_BIND=127.0.0.1:8096 INKY_SERVE_KEY=... scripts/inky-fleet-serve.sh [extra inky serve flags...]
#
# Environment overrides:
#   INKY_BIN          Path to inky binary (default: <repo>/target/release/inky)
#   INKY_MODEL        Path to GGUF (default: ~/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf)
#   INKY_BIND         Listen address (default: 127.0.0.1:8096)
#   INKY_LANES        Active lane count (default: 64 — measured M5 Max knee)
#   INKY_CTX          Context per lane (default: 16384)
#   INKY_QUEUE        Bounded admission queue depth (default: 64)
#   INKY_EFFORT       Default reasoning effort (default: medium)
#   INKY_MAX_TOKENS   Default reply token budget (default: 8192, must be < INKY_CTX)
#   INKY_RECORD_DIR   Directory for cards-YYYYMMDD.jsonl (default: <repo>/logs/cards)
#   INKY_KEY_ENV      Env var carrying bearer API key (default: INKY_SERVE_KEY)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INKY_BIN=${INKY_BIN:-$REPO_ROOT/target/release/inky}
INKY_MODEL=${INKY_MODEL:-$HOME/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf}
INKY_BIND=${INKY_BIND:-127.0.0.1:8096}
INKY_LANES=${INKY_LANES:-64}
INKY_CTX=${INKY_CTX:-16384}
INKY_QUEUE=${INKY_QUEUE:-64}
INKY_EFFORT=${INKY_EFFORT:-medium}
INKY_MAX_TOKENS=${INKY_MAX_TOKENS:-8192}
INKY_RECORD_DIR=${INKY_RECORD_DIR:-$REPO_ROOT/logs/cards}
INKY_KEY_ENV=${INKY_KEY_ENV:-INKY_SERVE_KEY}

if [[ ! -x "$INKY_BIN" ]]; then
    printf '%s\n' "[fleet-serve] missing executable inky binary at $INKY_BIN" >&2
    exit 2
fi

if [[ -z "${!INKY_KEY_ENV:-}" ]]; then
    printf '%s\n' "[fleet-serve] required key env var $INKY_KEY_ENV is unset or empty" >&2
    exit 2
fi

mkdir -p "$INKY_RECORD_DIR"
chmod 700 "$INKY_RECORD_DIR" || true

exec "$INKY_BIN" serve \
    -m "$INKY_MODEL" \
    -c "$INKY_CTX" \
    -n "$INKY_MAX_TOKENS" \
    --lanes "$INKY_LANES" \
    --queue "$INKY_QUEUE" \
    --effort "$INKY_EFFORT" \
    --bind "$INKY_BIND" \
    --key-env "$INKY_KEY_ENV" \
    --record-dir "$INKY_RECORD_DIR" \
    "$@"
