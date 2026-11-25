#!/bin/zsh
# tests/nightly.sh: the long GPU gates, too slow for a pull request; .github/workflows/nightly.yml runs them on the
# M5 Mac after `cargo build --release`. Each line is one gate; every one runs, and any failure fails the night.
#  - planted-bug negative controls for check, binv, lanes, and dbatch (each must exit 1 with its exact failure line;
#    a crash or any other exit code fails the gate).
#  - inky poolgate, long: 17 lanes, prompts of 618 tokens in chunks of 1 to 256 (across many 64-position blocks), 64
#    greedy decode steps, every pool slot outside the step poisoned; must give layer 2's outputs bit for bit
#    (measured 194 s wall, 3:14, on an M5 Pro, 2026-11-23).
#  - SSD spill/restore model logit exactness test (`restored_long_prefix_and_early_fork_match_cold_logits`).
set -e
cd "${0:A:h}/.."
B=${INKY_BIN:-./target/release/inky}
REF=${INKY_REF:-$HOME/inky-ci/ref}
fail=0

chk_out=$(INK_CHECK_SABOTAGE=1 "$B" check --ref "$REF/short" -c 4096 2>&1) && chk_rc=0 || chk_rc=$?
if [ "$chk_rc" = 1 ] && print -r -- "$chk_out" | grep -q 'CHECK AGREEMENT FAILED'; then echo "nightly check sabotage: detected, as required"; else echo "FAIL: nightly check sabotage not detected (exit $chk_rc)"; fail=1; fi

binv_out=$(INK_BINV_SABOTAGE=1 INK_SPLITS=0,64 "$B" binv --ref ref/long.tokens -c 4096 2>&1) && binv_rc=0 || binv_rc=$?
if [ "$binv_rc" = 1 ] && print -r -- "$binv_out" | grep -q '^NOT batch invariant$'; then echo "nightly binv sabotage: detected, as required"; else echo "FAIL: nightly binv sabotage not detected (exit $binv_rc)"; fail=1; fi

lanes_out=$(INK_LANES_SABOTAGE=1 INK_PF=2380 "$B" lanes --ref ref/long.tokens -c 4096 2>&1) && lanes_rc=0 || lanes_rc=$?
if [ "$lanes_rc" = 1 ] && print -r -- "$lanes_out" | grep -q '^LANES NOT INDEPENDENT$'; then echo "nightly lanes sabotage: detected, as required"; else echo "FAIL: nightly lanes sabotage not detected (exit $lanes_rc)"; fail=1; fi

db_out=$(INK_DBATCH_SABOTAGE=1 INK_LANES=2 INK_PF=128 INK_DECODE=2 "$B" dbatch --ref ref/long.tokens -c 4096 2>&1) && db_rc=0 || db_rc=$?
if [ "$db_rc" = 1 ] && print -r -- "$db_out" | grep -q '^DBATCH NOT EXACT$'; then echo "nightly dbatch sabotage: detected, as required"; else echo "FAIL: nightly dbatch sabotage not detected (exit $db_rc)"; fail=1; fi

INK_POOLGATE_LANES=17 INK_POOLGATE_CHUNKS=31,1,23,7,17,1,29,64,127,61,1,256 INK_POOLGATE_DECODE=64 \
    "$B" poolgate --ref ref/long.tokens -c 4096 2>&1 | tail -2 | tee /dev/stderr | grep -q '^pool gate identical' || { echo "FAIL: poolgate long"; fail=1; }
# A fresh path makes a skipped test or a disconnected trace hook fail the replay.
trace_work=$(mktemp -d "${TMPDIR:-/tmp}/inky-nightly-serve.XXXXXX")
trap 'rm -rf "$trace_work"' EXIT
if ! INKY_SCHED_TRACE_OUT="$trace_work/serve.jsonl" cargo test --locked --release \
        serve::tests::serve_layer3_three_turn_10k_cached_matches_cold_solo -- --ignored --exact ||
   ! scripts/check-tla.sh --serve-trace "$trace_work/serve.jsonl"; then
    echo "FAIL: three-turn serve trace"
    fail=1
fi
# Verify a real SSD restore and early prefix fork against both sets of cold model logits.
spill_test=serve::block_cache::tests::spill_model::restored_long_prefix_and_early_fork_match_cold_logits
if spill_result=$(cargo test --locked --release --bin inky "$spill_test" -- --ignored --exact --nocapture 2>&1); then
    grep -Fq 'test result: ok. 1 passed;' <<< "$spill_result" || fail=1
else
    fail=1
fi
print -r -- "$spill_result"
exit $fail
