#!/bin/zsh
# tests/nightly.sh: the long GPU gates, too slow for a pull request; .github/workflows/nightly.yml runs them on the
# M5 Mac after `cargo build --release`. Each line is one gate; every one runs, and any failure fails the night.
#  - inky poolgate, long: 17 lanes, prompts of 618 tokens in chunks of 1 to 256 (across many 64-position blocks), 64
#    greedy decode steps, every pool slot outside the step poisoned; must give layer 2's outputs bit for bit
#    (measured 194 s wall, 3:14, on an M5 Pro, 2026-11-23).
set -e
cd "${0:A:h}/.."
B=./target/release/inky
fail=0
INK_POOLGATE_LANES=17 INK_POOLGATE_CHUNKS=31,1,23,7,17,1,29,64,127,61,1,256 INK_POOLGATE_DECODE=64 \
    $B poolgate --ref ref/long.tokens -c 4096 2>&1 | tail -2 | tee /dev/stderr | grep -q '^pool gate identical' || { echo "FAIL: poolgate long"; fail=1; }
# Verify a real SSD restore and early prefix fork against both sets of cold model logits.
spill_test=serve::block_cache::tests::spill_model::restored_long_prefix_and_early_fork_match_cold_logits
if spill_result=$(cargo test --locked --release --bin inky "$spill_test" -- --ignored --exact --nocapture 2>&1); then
    grep -Fq 'test result: ok. 1 passed;' <<< "$spill_result" || fail=1
else
    fail=1
fi
print -r -- "$spill_result"
exit $fail
