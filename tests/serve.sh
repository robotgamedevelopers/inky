#!/bin/zsh
# Usage: tests/serve.sh [release-binary [model.gguf]]
# SERVECHECK_SABOTAGE supports negative-control probes:
#   health-status           - expects HTTP 418 from /health
#   memory-budget           - simulates zero Metal working-set reporting on /health
#   concurrent-tool-calls   - injects duplicate tool_calls[0].id and argument bleed
#   disconnect-mid-thinking - simulates uncancelled generation reaching max_tokens
# CI runs normal success first, then requires each negative-control failure.
set -eu
cd "${0:A:h}/.."
bin=${1:-./target/release/inky}
model=${2:-$HOME/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf}
bin_abs=${bin:A}
model_abs=${model:A}
# Exercise header rejection order on parsed metadata, entirely in-process.
out=$(cargo test --locked --offline serve_header_policy) || { print -ru2 -- "$out"; exit 1; }
[[ "$out" == *'1 passed; 0 failed;'* ]] || { print -ru2 -- "$out"; exit 1; }
cd tests
exec go run ./servecheck "$bin_abs" "$model_abs"
