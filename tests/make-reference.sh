#!/bin/zsh
# tests/make-reference.sh LLAMA_CPP_DIR: build tools/ref_logits against a built llama.cpp checkout and write the reference
# logits tests/ci.sh compares against into $INKY_REF (default ~/inky-ci/ref). Measured with llama.cpp 44d5087.
set -e
cd "${0:A:h}/.."
L=${1:?usage: tests/make-reference.sh LLAMA_CPP_DIR}
REF=${INKY_REF:-$HOME/inky-ci/ref}; mkdir -p $REF
M=$HOME/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf
clang++ -std=c++17 -O2 -I $L/include -I $L/ggml/include tools/ref_logits.cpp -L $L/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin -o $REF/ref_logits
$REF/ref_logits $M "Write a function in C that reverses a singly linked list, and explain it." 40 $REF/short
$REF/ref_logits $M "$(cat ref/long_prompt.txt)" 24 $REF/long 1
# keep the last 64 positions of the long reference (check aligns a trimmed dump to the end of the tokens)
tail -c $((64 * 200058 * 4)) $REF/long.logits > $REF/long.t && mv $REF/long.t $REF/long.logits
