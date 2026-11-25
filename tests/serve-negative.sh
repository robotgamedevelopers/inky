#!/bin/sh
# Run all four live server-fault negative controls; accept only a completed suite with the intended failures.
set -eu
log=$(mktemp)
trap 'rm -f "$log"' EXIT
if SERVECHECK_SABOTAGE=all "$@" >"$log" 2>&1; then rc=0; else rc=$?; fi
cat "$log"
test "$rc" -eq 1
test "$(grep -c '^FAIL ' "$log")" -eq 4
grep -Fxq 'FAIL bearer authentication and public health: got HTTP 503, want 200' "$log"
grep -Fxq 'FAIL health reports Metal memory budget and admission limits: missing or non-positive recommended_working_set_bytes' "$log"
grep -Fxq 'FAIL disconnect in mid-thinking stops generation: /health did not drain within 3s: active_lanes=1 inflight=1 queued=0' "$log"
grep -Fxq 'FAIL concurrent identical tool calls: stream drain: /health did not drain within 5s: active_lanes=0 inflight=1 queued=0' "$log"
grep -Fxq 'one or more serve checks failed' "$log"
