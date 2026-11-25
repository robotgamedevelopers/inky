#!/bin/zsh
# tests/nightly.sh: the long GPU gates, too slow for a pull request; .github/workflows/nightly.yml runs them on the
# M5 Mac after `cargo build --release`. Each line is one gate; every one runs, and any failure fails the night.
#  - inky poolgate, long: 17 lanes over ServeBlockCache, prompts of 1067 tokens (> 1024 sliding window) in chunks of
#    1 to 256 across 64-position blocks, 64 greedy decode steps, every pool slot outside the step poisoned; must give
#    layer 2's outputs bit for bit while exercising sharing, reuse, and sliding-window pruning.
set -e
cd "${0:A:h}/.."
B=./target/release/inky
fail=0
INK_POOLGATE_LANES=17 INK_POOLGATE_CHUNKS=31,1,23,7,17,1,29,64,127,61,1,256,256,193 INK_POOLGATE_DECODE=64 \
    $B poolgate --ref ref/long.tokens -c 4096 2>&1 | tail -2 | tee /dev/stderr | grep -q '^pool gate identical' || { echo "FAIL: poolgate long"; fail=1; }
# A fresh path makes a skipped test or a disconnected trace hook fail the replay.
trace_work=$(mktemp -d "${TMPDIR:-/tmp}/inky-nightly-serve.XXXXXX")
trap 'rm -rf "$trace_work"' EXIT
if trace_result=$(INKY_SCHED_TRACE_OUT="$trace_work/serve.jsonl" cargo test --locked --release --bin inky \
        serve::tests::serve_layer3_three_turn_10k_cached_matches_cold_solo -- --ignored --exact --nocapture 2>&1); then
    print -r -- "$trace_result"
    grep -Eq '^test result: ok\. 1 passed; 0 failed;' <<< "$trace_result" || { echo "FAIL: serve_layer3_three_turn_10k_cached_matches_cold_solo did not match 1 passed; 0 failed" >&2; fail=1; }
    scripts/check-tla.sh --serve-trace "$trace_work/serve.jsonl" || { echo "FAIL: three-turn serve trace"; fail=1; }
else
    print -r -- "$trace_result"
    echo "FAIL: three-turn serve trace"
    fail=1
fi
# Verify a real SSD restore and early prefix fork against both sets of cold model logits.
spill_test=serve::block_cache::tests::spill_model::restored_long_prefix_and_early_fork_match_cold_logits
if spill_result=$(cargo test --locked --release --bin inky "$spill_test" -- --ignored --exact --nocapture 2>&1); then
    grep -Eq '^test result: ok\. 1 passed; 0 failed;' <<< "$spill_result" || { echo "FAIL: $spill_test did not match 1 passed; 0 failed" >&2; fail=1; }
else
    fail=1
fi
print -r -- "$spill_result"
exit $fail
