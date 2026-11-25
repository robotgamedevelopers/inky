#!/bin/zsh
# Require the exact Metal fixtures and verify a planted weak-pass state write.
set -eu
cd "${0:A:h}/.."
unset INKY_TEST_LOOPCD_WRITE_WEAK_STATE
kernel=engine::tests::loopcd_mixed_batch_preserves_disabled_rows
cli=loopcd_tests::loopcd_cli_first_token
cached=loopcd_tests::loopcd_cached_prompt_first_token_and_mixed_tie_isolation
model=loopcd_tests::loopcd_paged_mixed_lane_state
fail=0
report='test result: ok. 1 passed; 0 failed;'
for fixture in "$kernel" "$cli" "$cached" "$model"; do
    if out=$(cargo test --locked --release --bin inky "$fixture" -- --ignored --exact 2>&1) &&
       [[ "$out" == *"$report"* ]]; then
        print -r -- "$out"
    else
        print -r -- "$out"
        echo "FAIL: exact LoopCD test $fixture"
        fail=1
    fi
done

rc=0
out=$(INKY_TEST_LOOPCD_WRITE_WEAK_STATE=1 cargo test --locked --release --bin inky "$model" \
    -- --ignored --exact --nocapture 2>&1) || rc=$?
print -r -- "$out"
if [[ "$rc" != 101 || "$out" != *'test result: FAILED. 0 passed; 1 failed;'* ||
      "$out" != *'LoopCD paged weak pass changed strong logits or state'* ]]; then
    echo "FAIL: LoopCD weak-pass state-write control (exit $rc)"
    fail=1
fi
exit "$fail"
