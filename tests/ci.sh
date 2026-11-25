#!/bin/zsh
# tests/ci.sh: what CI runs on an M5 Mac. Needs the model at its default path and the llama.cpp reference logits in
# $INKY_REF (default ~/inky-ci/ref; tests/make-reference.sh writes them).
#  1. agreement with llama.cpp: every checked position's argmax agrees and the worst KL is at most 0.01, on the decode
#     path (a 58-position short prompt, both single-stream and INK_DEC=1 decode steps) and on a 2,403-token long prompt
#     with INK_PF=2380 (comparing position 2379 at the end of the 2,380-token prefill plus the 23 post-prefill decode
#     positions 2380..=2402). Its sabotaged run (INK_CHECK_SABOTAGE=1) must exit 1 with CHECK AGREEMENT FAILED.
#  2. batch invariance (inky binv), lane independence (inky lanes), steps over spans (inky spans), decode steps over
#     lanes (inky dbatch) and the cross-lane prefix copy (inky xcopy): every lane identical to itself alone; decode
#     steps also agree with llama.cpp. Every GPU gate's planted-bug sabotage run (INK_BINV_SABOTAGE=1,
#     INK_LANES_SABOTAGE=1, INK_SPANS_SABOTAGE=1, INK_DBATCH_SABOTAGE=1, INK_XCOPY_SABOTAGE=1) must exit 1 with its
#     exact failure line; a crash or any other exit code fails CI.
#     inky poolgate: 3 lanes with every pool slot outside the step poisoned give layer 2's outputs; its sabotaged run
#     must exit 1 and differ only in the lane it poisoned. The long poolgate runs in tests/nightly.sh.
#  3. tests/golden.sh is byte-identical to tests/golden-<chip>.txt (plus a mutated-hash negative control proving diff
#     catches any hash drift). A change that alters logits on purpose updates that file in the same commit and says why.
# Usage: tests/ci.sh [full|pr]. full (the default; the nightly workflow) runs everything above. pr (every pull request
# and push, which must finish in about a minute) runs the fast representative set: the three llama.cpp agreements +
# check sabotage, and the xcopy and poolgate pairs with their negative controls. INKY_BIN overrides the binary (default
# ./target/release/inky).
set -e
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
agree "prefill+decode (1 prefill end + 23 decode rows), long prompt vs llama.cpp" long 24 INK_PF=2380
agree "decode steps, short prompt vs llama.cpp" short 58 INK_DEC=1
# negative control: INK_CHECK_SABOTAGE=1 perturbs one compared logit row and must exit 1 with CHECK AGREEMENT FAILED
chk_out=$(INK_CHECK_SABOTAGE=1 "$B" check --ref "$REF/short" -c 4096 2>&1) && chk_rc=0 || chk_rc=$?
if [ "$chk_rc" = 1 ] && print -r -- "$chk_out" | grep -q 'CHECK AGREEMENT FAILED'; then echo "check sabotage: detected, as required"; else echo "FAIL: check sabotage not detected (exit $chk_rc)"; fail=1; fi
$B xcopy --ref ref/long.tokens -c 4096 2>&1 | tail -1 | tee /dev/stderr | grep -q '^xcopy lossless' || { echo "FAIL: xcopy"; fail=1; }
# negative control: a copy one token short of the head it claims must exit 1 and report XCOPY NOT LOSSLESS (src/main.rs); a crash or any other failure does not count
xc_out=$(INK_XCOPY_SABOTAGE=1 $B xcopy --ref ref/long.tokens -c 4096 2>&1) && xc_rc=0 || xc_rc=$?
if [ "$xc_rc" = 1 ] && print -r -- "$xc_out" | grep -q 'XCOPY NOT LOSSLESS'; then echo "xcopy sabotage: detected, as required"; else echo "FAIL: xcopy sabotage not detected (exit $xc_rc)"; fail=1; fi
$B poolgate --ref ref/long.tokens -c 4096 2>&1 | tail -1 | tee /dev/stderr | grep -q '^pool gate identical' || { echo "FAIL: poolgate"; fail=1; }
# negative control: a step whose own slots are poisoned must exit 1 AND print POOL GATE DIFFERS (src/main.rs); a crash
# (any other status, a panic) does not count, nor does the text without status 1. And only the poisoned lane may differ:
# another lane that differs read the poison through state the lanes share.
pg_out=$(INK_POOLGATE_SABOTAGE=1 $B poolgate --ref ref/long.tokens -c 4096 2>&1) && pg_rc=0 || pg_rc=$?
if [ "$pg_rc" = 1 ] && print -r -- "$pg_out" | grep -q 'POOL GATE DIFFERS'; then echo "poolgate sabotage: detected, as required"; else echo "FAIL: poolgate sabotage not detected (exit $pg_rc)"; fail=1; fi
if print -r -- "$pg_out" | grep -q '^INK_POOLGATE_SABOTAGE: no other lane differs'; then echo "poolgate sabotage: no other lane differs, as required"; else print -r -- "$pg_out" | grep 'LANE LEAK' || true; echo "FAIL: poolgate sabotage leaked across lanes"; fail=1; fi
if [ "$mode" = pr ]; then
    echo "pr set done; binv, lanes, spans, dbatch (each with sabotage), cli checks and golden run in the nightly workflow (tests/ci.sh full)"
    exit $fail
fi
$B binv --ref ref/long.tokens -c 4096 2>&1 | tail -1 | tee /dev/stderr | grep -q '^batch invariant' || { echo "FAIL: binv"; fail=1; }
binv_out=$(INK_BINV_SABOTAGE=1 INK_SPLITS=0,64 $B binv --ref ref/long.tokens -c 4096 2>&1) && binv_rc=0 || binv_rc=$?
if [ "$binv_rc" = 1 ] && print -r -- "$binv_out" | grep -q '^NOT batch invariant$'; then echo "binv sabotage: detected, as required"; else echo "FAIL: binv sabotage not detected (exit $binv_rc)"; fail=1; fi
$B lanes --ref ref/long.tokens -c 4096 2>&1 | tail -1 | tee /dev/stderr | grep -q '^lanes independent' || { echo "FAIL: lanes"; fail=1; }
lanes_out=$(INK_LANES_SABOTAGE=1 INK_PF=2380 $B lanes --ref ref/long.tokens -c 4096 2>&1) && lanes_rc=0 || lanes_rc=$?
if [ "$lanes_rc" = 1 ] && print -r -- "$lanes_out" | grep -q '^LANES NOT INDEPENDENT$'; then echo "lanes sabotage: detected, as required"; else echo "FAIL: lanes sabotage not detected (exit $lanes_rc)"; fail=1; fi
INK_LANES=1,2,17,65 $B spans --ref ref/long.tokens -c 4096 2>&1 | tail -1 | tee /dev/stderr | grep -q '^spans invariant' || { echo "FAIL: spans"; fail=1; }
# negative control: the sabotaged spans run (one span bound to the wrong lane's state) must exit 1 and report SPANS NOT INVARIANT
spans_out=$(INK_SPANS_SABOTAGE=1 INK_LANES=2 $B spans --ref ref/long.tokens -c 4096 2>&1) && spans_rc=0 || spans_rc=$?
if [ "$spans_rc" = 1 ] && print -r -- "$spans_out" | grep -q 'SPANS NOT INVARIANT'; then echo "spans sabotage: detected, as required"; else echo "FAIL: spans sabotage not detected (exit $spans_rc)"; fail=1; fi
INK_LANES=1,8,33 INK_DECODE=8 $B dbatch --ref ref/long.tokens -c 4096 2>&1 | tail -1 | tee /dev/stderr | grep -q '^dbatch exact' || { echo "FAIL: dbatch"; fail=1; }
db_out=$(INK_DBATCH_SABOTAGE=1 INK_LANES=2 INK_PF=128 INK_DECODE=2 $B dbatch --ref ref/long.tokens -c 4096 2>&1) && db_rc=0 || db_rc=$?
if [ "$db_rc" = 1 ] && print -r -- "$db_out" | grep -q '^DBATCH NOT EXACT$'; then echo "dbatch sabotage: detected, as required"; else echo "FAIL: dbatch sabotage not detected (exit $db_rc)"; fail=1; fi
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
    # negative control: mutating one hex nibble in the golden output must fail diff
    awk 'NR==1 { sub(/[0-9a-f]$/, "X") } { print }' /tmp/inky-golden.$$ > /tmp/inky-golden.$$.sab
    if diff -q tests/golden-$chip.txt /tmp/inky-golden.$$.sab >/dev/null 2>&1; then
        echo "FAIL: golden diff negative control did not detect mutated hash"
        fail=1
    else
        echo "golden sabotage: mutated hash detected, as required"
    fi
    rm -f /tmp/inky-golden.$$ /tmp/inky-golden.$$.sab
else
    echo "FAIL: missing tests/golden-$chip.txt for this chip"
    fail=1
fi
exit $fail
