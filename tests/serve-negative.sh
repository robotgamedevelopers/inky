#!/bin/sh
# Run all four live controls; accept only a completed suite with the intended failures.
set -eu
log=$(mktemp)
trap 'rm -f "$log"' EXIT
if SERVECHECK_SABOTAGE=all "$@" >"$log" 2>&1; then rc=0; else rc=$?; fi
cat "$log"
test "$rc" -eq 1
test "$(grep -c '^FAIL ' "$log")" -eq 4
grep -Fxq 'FAIL bearer authentication and public health: sabotage health-status: got HTTP 200, want 418' "$log"
grep -Fxq 'FAIL health reports Metal memory budget and admission limits: sabotage memory-budget: missing or non-positive recommended_working_set_bytes' "$log"
grep -Fxq 'FAIL disconnect in mid-thinking stops generation: sabotage disconnect-mid-thinking: late decode cancellation: 512/512 tokens' "$log"
grep -Eq '^FAIL concurrent identical tool calls: sabotage concurrent-tool-calls: reply 1: duplicate tool_calls\[0\]\.id "call_[^"]+" across concurrent cards$' "$log"
grep -Fxq 'one or more serve checks failed' "$log"
