#!/bin/zsh
# tests/ci.sh: what CI runs on an M5 Mac. Needs the model at its default path and the llama.cpp reference logits in
# $INKY_REF (default ~/inky-ci/ref; tests/make-reference.sh writes them).
#  1. agreement with llama.cpp: every checked position's argmax agrees and the worst KL is at most 0.01, on the decode
#     path (a short prompt) and the prefill path (a 2394-token prompt).
#  2. batch invariance (inky binv), lane independence (inky lanes), steps over spans (inky spans), decode steps over
#     lanes (inky dbatch) and the cross-lane prefix copy (inky xcopy): every lane identical to itself alone; decode
#     steps also agree with llama.cpp. The sabotaged spans and xcopy runs are required to fail.
#     inky poolgate: 3 lanes with every pool slot outside the step poisoned give layer 2's outputs; its sabotaged run
#     must fail, and only in the lane it poisoned. The long poolgate runs in tests/nightly.sh.
#  3. tests/golden.sh is byte-identical to tests/golden-<chip>.txt: the logits did not change. A change that changes them
#     on purpose updates that file in the same commit and says why.
# Usage: tests/ci.sh [full|pr]. full (the default; the nightly workflow) runs everything above. pr (every pull request
# and push, which must finish in about a minute) runs the fast representative set: the three llama.cpp agreements and
# the xcopy and poolgate pairs with their negative controls (about 40 s on an M5). INKY_BIN overrides the binary (default
# ./target/release/inky).
set -e
set -o pipefail
cd "${0:A:h}/.."
B=${INKY_BIN:-./target/release/inky}
REF=${INKY_REF:-$HOME/inky-ci/ref}
mode=${1:-full}
case $mode in
    full|pr) ;;
    *) echo "usage: tests/ci.sh [full|pr]" >&2; exit 2 ;;
esac
[ -x "$B" ] || { echo "FAIL: no inky binary at $B" >&2; exit 1; }
fail=0
tests/golden-status.sh || { echo "FAIL: golden producer status"; fail=1; }
agree() {  # name, ref, expected positions, env...
    local name=$1 ref=$2 expected=$3
    shift 3
    local out
    if ! out=$(env -u INK_PF -u INK_DEC "$@" "$B" check --ref "$REF/$ref" -c 4096 2>&1); then
        print -r -- "$out"
        echo "FAIL: $name"
        fail=1
        return
    fi
    print -r -- "$out" | awk -v name="$name" -v expected="$expected" '
        /^argmax agreement / {
            print name ": " $0
            seen++
            split($3, a, "/")
            if (NF != 6 || $3 !~ /^[0-9]+\/[0-9]+$/ || a[1]+0 != expected || a[2]+0 != expected ||
                $4 != "worst" || $5 != "KL" || $6 !~ /^[0-9]+([.][0-9]+)?$/ || $6+0 > 0.01) bad=1
        }
        END { exit (seen != 1 || bad) }
    ' || { echo "FAIL: $name (expected $expected agreeing positions)"; fail=1; }
}
agree "decode, short prompt vs llama.cpp" short 58
agree "prefill, long prompt vs llama.cpp" long 24 INK_PF=2380
agree "decode steps, short prompt vs llama.cpp" short 58 INK_DEC=1
$B xcopy --ref ref/long.tokens -c 4096 2>&1 | tail -1 | tee /dev/stderr | grep -q '^xcopy lossless' || { echo "FAIL: xcopy"; fail=1; }
# negative control: a copy one token short of the head it claims must exit 1 AND report XCOPY NOT LOSSLESS (src/main.rs); a crash or any other status does not count
xc_out=$(INK_XCOPY_SABOTAGE=1 $B xcopy --ref ref/long.tokens -c 4096 2>&1) && xc_rc=0 || xc_rc=$?
if [ "$xc_rc" = 1 ] && print -r -- "$xc_out" | grep -q 'XCOPY NOT LOSSLESS'; then echo "xcopy sabotage: detected, as required"; else print -r -- "$xc_out"; echo "FAIL: xcopy sabotage not detected (exit $xc_rc)"; fail=1; fi
$B poolgate --ref ref/long.tokens -c 4096 2>&1 | tail -1 | tee /dev/stderr | grep -q '^pool gate identical' || { echo "FAIL: poolgate"; fail=1; }
# negative control: a step whose own slots are poisoned must exit 1 AND print POOL GATE DIFFERS (src/main.rs); a crash
# (any other status, a panic) does not count, nor does the text without status 1. And only the poisoned lane may differ:
# another lane that differs read the poison through state the lanes share.
pg_out=$(INK_POOLGATE_SABOTAGE=1 $B poolgate --ref ref/long.tokens -c 4096 2>&1) && pg_rc=0 || pg_rc=$?
if [ "$pg_rc" = 1 ] && print -r -- "$pg_out" | grep -q 'POOL GATE DIFFERS'; then echo "poolgate sabotage: detected, as required"; else echo "FAIL: poolgate sabotage not detected (exit $pg_rc)"; fail=1; fi
if print -r -- "$pg_out" | grep -q '^INK_POOLGATE_SABOTAGE: no other lane differs'; then echo "poolgate sabotage: no other lane differs, as required"; else print -r -- "$pg_out" | grep 'LANE LEAK' || true; echo "FAIL: poolgate sabotage leaked across lanes"; fail=1; fi
$B loopcdgate --ref ref/long.tokens -c 256 2>&1 | tail -1 | tee /dev/stderr | grep -q '^loopcd gate exact' || { echo "FAIL: loopcdgate"; fail=1; }
# negative control: perturbing one lane's blended LoopCD logit must exit 1 AND report LOOPCD GATE DIFFERS
lc_out=$(INK_LOOPCDGATE_SABOTAGE=1 $B loopcdgate --ref ref/long.tokens -c 256 2>&1) && lc_rc=0 || lc_rc=$?
if [ "$lc_rc" = 1 ] && print -r -- "$lc_out" | grep -q 'LOOPCD GATE DIFFERS'; then echo "loopcdgate sabotage: detected, as required"; else print -r -- "$lc_out"; echo "FAIL: loopcdgate sabotage not detected (exit $lc_rc)"; fail=1; fi
if [ "$mode" = pr ]; then
    echo "pr set done; binv, lanes, spans (and its sabotage), dbatch, blockedinv (and its sabotage), cli checks and golden run in the nightly workflow (tests/ci.sh full)"
    exit $fail
fi
$B binv --ref ref/long.tokens -c 4096 2>&1 | tail -1 | tee /dev/stderr | grep -q '^batch invariant' || { echo "FAIL: binv"; fail=1; }
$B lanes --ref ref/long.tokens -c 4096 2>&1 | tail -1 | tee /dev/stderr | grep -q '^lanes independent' || { echo "FAIL: lanes"; fail=1; }
INK_LANES=1,2,17,65 $B spans --ref ref/long.tokens -c 4096 2>&1 | tail -1 | tee /dev/stderr | grep -q '^spans invariant' || { echo "FAIL: spans"; fail=1; }
# A shorter private lane must clear the masked tail of its last K/V tile.
private_test=engine::tests::private_prefill_clears_the_last_window_tile_between_lanes
if private_out=$(cargo test --locked --release --bin inky "$private_test" -- --ignored --exact 2>&1) &&
   [[ "$private_out" == *"test $private_test ... ok"* ]]; then
    echo "private prefill scratch: masked rows cleared"
else
    print -r -- "$private_out"
    echo "FAIL: private prefill scratch"
    fail=1
fi
# negative control: the sabotaged spans run (one span bound to the wrong lane's state) must exit 1 AND report SPANS NOT INVARIANT
# (the exact line `inky spans` prints when its lanes differ, src/main.rs); a crash or any other status does not count
sp_out=$(INK_SPANS_SABOTAGE=1 INK_LANES=2 $B spans --ref ref/long.tokens -c 4096 2>&1) && sp_rc=0 || sp_rc=$?
if [ "$sp_rc" = 1 ] && print -r -- "$sp_out" | grep -q 'SPANS NOT INVARIANT'; then echo "spans sabotage: detected, as required"; else print -r -- "$sp_out"; echo "FAIL: spans sabotage not detected (exit $sp_rc)"; fail=1; fi
INK_LANES=1,8,33 INK_DECODE=8 $B dbatch --ref ref/long.tokens -c 4096 2>&1 | tail -1 | tee /dev/stderr | grep -q '^dbatch exact' || { echo "FAIL: dbatch"; fail=1; }
$B blockedinv --ref ref/long.tokens -c 4160 2>&1 | tail -1 | tee /dev/stderr | grep -q '^blockedinv exact' || { echo "FAIL: blockedinv"; fail=1; }
bi_out=$(INK_BLOCKEDINV_SABOTAGE=1 $B blockedinv --ref ref/long.tokens -c 1024 2>&1) && bi_rc=0 || bi_rc=$?
if [ "$bi_rc" = 1 ] && print -r -- "$bi_out" | grep -q 'BLOCKED NOT INVARIANT'; then echo "blockedinv sabotage: detected, as required"; else print -r -- "$bi_out"; echo "FAIL: blockedinv sabotage not detected (exit $bi_rc)"; fail=1; fi
zero_out=$(mktemp)
if zero_stats=$($B run -c 128 --raw -n 0 -p 'Once upon a time' 2>&1 > "$zero_out") &&
   [[ ! -s "$zero_out" && "$zero_stats" == *'generated 0 tok @ 0.00 tok/s'* ]]; then
    echo "zero generation budget: no tokens emitted or counted"
else
    print -r -- "$zero_stats"
    echo "FAIL: zero generation budget"
    fail=1
fi
rm -f "$zero_out"

chat_args=(chat -c 128 -n 1 --greedy --effort xhigh --system 'Answer concisely in Spanish.')
short='What color is the sky?'
long=$(printf 'oversized %.0s' {1..256})
prompt_count() { sed -nE 's/.*\[([0-9]+) prompt tok @.*/\1/p'; }
if fresh=$(printf '%s\n' "$short" | $B "${chat_args[@]}" 2>&1) &&
   retried=$(printf '%s\n' "$long" "$short" | $B "${chat_args[@]}" 2>&1) &&
   [[ "$retried" == *'[context full: /reset]'* ]] &&
   [[ -n $(print -r -- "$fresh" | prompt_count) &&
      $(print -r -- "$fresh" | prompt_count) == $(print -r -- "$retried" | prompt_count) ]]; then
    echo "first chat turn: oversized input preserves the retry's system prompt and effort"
else
    print -r -- "$fresh" "$retried"
    echo "FAIL: first chat turn after oversized input"
    fail=1
fi

chip=$(sysctl -n machdep.cpu.brand_string | sed 's/Apple //; s/ /-/g' | tr A-Z a-z)
if [ -f tests/golden-$chip.txt ]; then
    tests/golden.sh $B > /tmp/inky-golden.$$ 2>&1
    if diff tests/golden-$chip.txt /tmp/inky-golden.$$; then echo "golden: identical to tests/golden-$chip.txt"; else echo "FAIL: golden hashes changed"; fail=1; fi
    rm -f /tmp/inky-golden.$$
else
    echo "FAIL: missing tests/golden-$chip.txt for this chip"
    fail=1
fi
exit $fail
