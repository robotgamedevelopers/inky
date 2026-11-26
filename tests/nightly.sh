#!/bin/zsh
# tests/nightly.sh: the long GPU gates, too slow for a pull request; .github/workflows/nightly.yml runs them on the
# M5 Mac after `cargo build --release`. Each line is one gate; every one runs, and any failure fails the night.
#  - planted-bug negative controls for check, binv, lanes, and dbatch (each must exit 1 with its exact failure line;
#    a crash or any other exit code fails the gate).
#  - inky poolgate, long: 17 lanes over ServeBlockCache, prompts of 1067 tokens in chunks of 1 to 256 (across many
#    64-position blocks and past the 1024-token sliding window), 64 greedy decode steps (1131 positions total,
#    exercising sharing, reuse, and live window pruning), every pool slot outside the step poisoned; must give layer 2's
#    outputs bit for bit.
#  - SSD spill/restore model logit exactness test (`restored_long_prefix_and_early_fork_match_cold_logits`).
set -e
setopt pipefail
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

INK_POOLGATE_LANES=17 INK_POOLGATE_CHUNKS=31,1,23,7,17,1,29,64,127,61,1,256,256,193 INK_POOLGATE_DECODE=64 \
    "$B" poolgate --ref ref/long.tokens -c 4096 2>&1 | tail -2 | tee /dev/stderr | grep '^pool gate identical' || { echo "FAIL: poolgate long"; fail=1; }
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
INK_LANES=24 INK_PF=1152 INK_DECODE=8 INK_FULL_SPAN=4096 \
    $B blockedinv --ref ref/long.tokens -c 4160 2>&1 | tail -1 | tee /dev/stderr | grep -q '^blockedinv exact' || { echo "FAIL: blockedinv long"; fail=1; }
bi_out=$(INK_BLOCKEDINV_SABOTAGE=1 $B blockedinv --ref ref/long.tokens -c 1024 2>&1) && bi_rc=0 || bi_rc=$?
if [ "$bi_rc" = 1 ] && print -r -- "$bi_out" | grep -q 'BLOCKED NOT INVARIANT'; then echo "blockedinv sabotage: detected, as required"; else print -r -- "$bi_out"; echo "FAIL: nightly blockedinv sabotage not detected (exit $bi_rc)"; fail=1; fi
# Expert-grouped decode MoE A/B parity gate across 1..64 lanes (`0` bit difference vs `INK_NO_GRP_MOE=1`).
INK_DB_AB=1 INK_PF=256 INK_DECODE=4 INK_LANES=1,8,16,32,64 "$B" dbatch --ref ref/long.tokens -c 4096 2>&1 | tee /dev/stderr | grep -q '^dbatch exact: grouped MoE decode matches INK_NO_GRP_MOE=1 bit for bit$' || { echo "FAIL: dbatch grouped MoE A/B parity"; fail=1; }
# Layer 2 step-time and GPU-memory envelope gate at 2K, 20K, and 42K context (1..=64 fitting lanes) + sabotage negative control.
"$B" perfgate --ref ref/long.tokens -c 43136 2>&1 | tee /dev/stderr | grep -q '^perfgate ok:' || { echo "FAIL: perfgate 2K/20K/42K"; fail=1; }
pg_out=$(INK_PERFGATE_SABOTAGE=1 INK_CTXS=2048 INK_LANES=1,8 INK_STEPS=2 "$B" perfgate --ref ref/long.tokens -c 4096 2>&1) && pg_rc=0 || pg_rc=$?
if [ "$pg_rc" = 1 ] && print -r -- "$pg_out" | grep -q '^PERFGATE FAILED$'; then echo "nightly perfgate sabotage: detected, as required"; else print -r -- "$pg_out"; echo "FAIL: nightly perfgate sabotage not detected (exit $pg_rc)"; fail=1; fi
# Verify a real SSD restore and early prefix fork against both sets of cold model logits.
spill_test=serve::block_cache::tests::spill_model::restored_long_prefix_and_early_fork_match_cold_logits
if spill_result=$(cargo test --locked --release --bin inky "$spill_test" -- --ignored --exact --nocapture 2>&1); then
    grep -Eq '^test result: ok\. 1 passed; 0 failed;' <<< "$spill_result" || { echo "FAIL: $spill_test did not match 1 passed; 0 failed" >&2; fail=1; }
else
    fail=1
fi
print -r -- "$spill_result"
# Verify live `inky serve --spill-dir` SIGKILL mid-spill-write recovery and SSD HTTP parity.
if spill_kill_out=$(cargo test --locked --release --test spill_http_m5 m5_http_spill_sigkill_mid_write_restart_preserves_exact_outputs -- --ignored --exact --nocapture 2>&1); then
    grep -Fq 'test result: ok. 1 passed;' <<< "$spill_kill_out" || fail=1
else
    fail=1
fi
print -r -- "$spill_kill_out"
# Verify M5 native Q8 debug decode, 32K disk-vs-resident split-resize parity, and CLI session error recovery.
for m5_test in \
    engine::tests::debug_single_token_decode_native_checkpoint \
    engine::tests::disk_tier_matches_resident_logits_across_32k_staging_resize_transition \
    engine::tests::full_attn_budget_default_bit_identical_and_pruned_4k_coherent_on_m5 \
    engine::tests::sub4bit_quant_metal_kernels_match_cpu_and_save_bytes \
    engine::tests::sub4bit_quant_real_model_lmhead_logits_cosine_and_top1_on_m5 \
    session::tests::cli_session_recovers_from_injected_prefill_and_decode_failures; do
    if m5_out=$(cargo test --locked --release --bin inky "$m5_test" -- --ignored --exact --nocapture 2>&1); then
        grep -Eq '^test result: ok\. 1 passed; 0 failed;' <<< "$m5_out" || { echo "FAIL: $m5_test did not match 1 passed; 0 failed" >&2; fail=1; }
    else
        echo "FAIL: $m5_test" >&2
        fail=1
    fi
    print -r -- "$m5_out"
done
exit $fail
