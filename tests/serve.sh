#!/bin/zsh
# Usage: tests/serve.sh [release-binary [model.gguf]]
set -eu
cd "${0:A:h}/.."
bin=${1:-./target/release/inky}
model=${2:-$HOME/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf}
# Exercise header rejection order on parsed metadata, entirely in-process.
out=$(cargo test --locked --offline serve_header_policy) || { print -ru2 -- "$out"; exit 1; }
[[ "$out" == *'1 passed; 0 failed;'* ]] || { print -ru2 -- "$out"; exit 1; }
exec go run tests/servecheck/*.go "$bin" "$model"
