#!/bin/zsh
# tests/nightly.sh: the long GPU gates, too slow for a pull request; .github/workflows/nightly.yml runs them on the
# M5 Mac after `cargo build --release`. Each line is one gate; every one runs, and any failure fails the night.
#  - inky poolgate, long: 17 lanes, prompts of 618 tokens in chunks of 1 to 256 (across many 64-position blocks), 64
#    greedy decode steps, every pool slot outside the step poisoned; must give layer 2's outputs bit for bit
#    (about 140 s on an M5 Pro).
set -e
cd "${0:A:h}/.."
B=./target/release/inky
fail=0
INK_POOLGATE_LANES=17 INK_POOLGATE_CHUNKS=31,1,23,7,17,1,29,64,127,61,1,256 INK_POOLGATE_DECODE=64 \
    $B poolgate --ref ref/long.tokens -c 4096 2>&1 | tail -2 | tee /dev/stderr | grep -q '^pool gate identical' || { echo "FAIL: poolgate long"; fail=1; }
exit $fail
