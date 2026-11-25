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
if [ ! -f "$REF/long.tokens" ] || ! cmp -s "$REF/long.tokens" ref/long.tokens; then
    echo "FAIL: $REF/long.tokens is missing or differs from committed ref/long.tokens"
    fail=1
else
    echo "reference tokens: $REF/long.tokens matches committed ref/long.tokens"
fi
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
# Keep command status and the exact verdict together. A crash after printing a verdict fails.
verdict() {
    local name=$1 expected_status=$2 expected_line=$3
    shift 3
    local out rc=0
    out=$("$@" 2>&1) || rc=$?
    if [[ "$rc" == "$expected_status" ]] && grep -Eq "$expected_line" <<< "$out"; then
        print -r -- "$name: $(grep -E "$expected_line" <<< "$out")"
    else
        print -r -- "$out"
        echo "FAIL: $name (exit $rc, expected $expected_status)"
        fail=1
    fi
}

agree "decode, short prompt vs llama.cpp" short 58
agree "prefill+decode (1 prefill end + 23 decode rows), long prompt vs llama.cpp" long 24 INK_PF=2380
agree "decode steps, short prompt vs llama.cpp" short 58 INK_DEC=1
# negative control: INK_CHECK_SABOTAGE=1 perturbs one compared logit row and must exit 1 with CHECK AGREEMENT FAILED
chk_out=$(INK_CHECK_SABOTAGE=1 "$B" check --ref "$REF/short" -c 4096 2>&1) && chk_rc=0 || chk_rc=$?
if [ "$chk_rc" = 1 ] && print -r -- "$chk_out" | grep -q 'CHECK AGREEMENT FAILED'; then echo "check sabotage: detected, as required"; else echo "FAIL: check sabotage not detected (exit $chk_rc)"; fail=1; fi
kvd_mem_out=$(env -u INK_DEC -u INK_TIER_HOT INK_PF=1500 "$B" check --allow-mismatch --ref "$REF/long" -c 4096 2>&1) || { print -r -- "$kvd_mem_out"; echo "FAIL: in-memory baseline for --kv-disk"; fail=1; }
kvd_dsk_out=$(env -u INK_DEC -u INK_TIER_HOT INK_PF=1500 "$B" check --allow-mismatch --kv-disk --ref "$REF/long" -c 4096 2>&1) || { print -r -- "$kvd_dsk_out"; echo "FAIL: aligned --kv-disk check"; fail=1; }
kvd_mem_hash=$(print -r -- "$kvd_mem_out" | sed -nE 's/^logit bits hash ([0-9a-f]+)$/\1/p')
kvd_dsk_hash=$(print -r -- "$kvd_dsk_out" | sed -nE 's/^logit bits hash ([0-9a-f]+)$/\1/p')
if [ -n "$kvd_mem_hash" ] && [ "$kvd_mem_hash" = "$kvd_dsk_hash" ] && print -r -- "$kvd_dsk_out" | grep -q '^disk tier:' && ! print -r -- "$kvd_dsk_out" | grep -q 'hot ring shorter than a split'; then
    echo "aligned --kv-disk vs in-memory: logit bits hash $kvd_dsk_hash identical"
else
    print -r -- "$kvd_dsk_out"
    echo "FAIL: aligned --kv-disk hash ($kvd_dsk_hash) != in-memory ($kvd_mem_hash)"
    fail=1
fi
verdict xcopy 0 '^xcopy lossless' "$B" xcopy --ref ref/long.tokens -c 4096
# negative control: a copy one token short of the head it claims must exit 1 AND report XCOPY NOT LOSSLESS (src/main.rs); a crash or any other status does not count
verdict "xcopy sabotage" 1 '^XCOPY NOT LOSSLESS$' env INK_XCOPY_SABOTAGE=1 "$B" xcopy --ref ref/long.tokens -c 4096
verdict poolgate 0 '^pool gate identical' "$B" poolgate --ref ref/long.tokens -c 4096
# negative control: a step whose own slots are poisoned must exit 1 AND print POOL GATE DIFFERS (src/main.rs); a crash
# (any other status, a panic) does not count, nor does the text without status 1. And only the poisoned lane may differ:
# another lane that differs read the poison through state the lanes share.
pg_out=$(INK_POOLGATE_SABOTAGE=1 $B poolgate --ref ref/long.tokens -c 4096 2>&1) && pg_rc=0 || pg_rc=$?
if [ "$pg_rc" = 1 ] && print -r -- "$pg_out" | grep -q 'POOL GATE DIFFERS'; then echo "poolgate sabotage: detected, as required"; else echo "FAIL: poolgate sabotage not detected (exit $pg_rc)"; fail=1; fi
if print -r -- "$pg_out" | grep -q '^INK_POOLGATE_SABOTAGE: no other lane differs'; then echo "poolgate sabotage: no other lane differs, as required"; else print -r -- "$pg_out" | grep 'LANE LEAK' || true; echo "FAIL: poolgate sabotage leaked across lanes"; fail=1; fi
pgs_out=$(INK_POOLGATE_START_SABOTAGE=1 $B poolgate --ref ref/long.tokens -c 4096 2>&1) && pgs_rc=0 || pgs_rc=$?
if [ "$pgs_rc" = 1 ] && print -r -- "$pgs_out" | grep -q 'PREFIX START-POINT GATE DIFFERS'; then echo "poolgate start-point sabotage: detected, as required"; else echo "FAIL: poolgate start-point sabotage not detected (exit $pgs_rc)"; fail=1; fi
$B loopcdgate --ref ref/long.tokens -c 256 2>&1 | tail -1 | tee /dev/stderr | grep -q '^loopcd gate exact' || { echo "FAIL: loopcdgate"; fail=1; }
# negative control: perturbing one lane's blended LoopCD logit must exit 1 AND report LOOPCD GATE DIFFERS
lc_out=$(INK_LOOPCDGATE_SABOTAGE=1 $B loopcdgate --ref ref/long.tokens -c 256 2>&1) && lc_rc=0 || lc_rc=$?
if [ "$lc_rc" = 1 ] && print -r -- "$lc_out" | grep -q 'LOOPCD GATE DIFFERS'; then echo "loopcdgate sabotage: detected, as required"; else print -r -- "$lc_out"; echo "FAIL: loopcdgate sabotage not detected (exit $lc_rc)"; fail=1; fi
if [ "$mode" = pr ]; then
    echo "pr set done; binv, lanes, spans, dbatch, blockedinv (each with sabotage), cli checks and golden run in the nightly workflow (tests/ci.sh full)"
    exit $fail
fi
verdict binv 0 '^batch invariant' "$B" binv --ref ref/long.tokens -c 4096
verdict "binv sabotage" 1 '^NOT batch invariant$' env INK_BINV_SABOTAGE=1 INK_SPLITS=0,64 "$B" binv --ref ref/long.tokens -c 4096
verdict lanes 0 '^lanes independent' "$B" lanes --ref ref/long.tokens -c 4096
verdict "lanes sabotage" 1 '^LANES NOT INDEPENDENT$' env INK_LANES_SABOTAGE=1 INK_PF=2380 "$B" lanes --ref ref/long.tokens -c 4096
verdict spans 0 '^spans invariant' env INK_LANES=1,2,17,65 "$B" spans --ref ref/long.tokens -c 4096
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
verdict "spans sabotage" 1 '^SPANS NOT INVARIANT$' env INK_SPANS_SABOTAGE=1 INK_LANES=2 "$B" spans --ref ref/long.tokens -c 4096
verdict crosspath 0 '^crosspath exact:' "$B" crosspath --ref ref/long.tokens -c 4096
verdict "crosspath sabotage" 1 '^CROSSPATH MISMATCH$' env INK_CROSSPATH_SABOTAGE=1 "$B" crosspath --ref ref/long.tokens -c 4096
verdict dbatch 0 '^dbatch exact' env INK_LANES=1,8,33,65 INK_DECODE=8 "$B" dbatch --ref ref/long.tokens -c 4096
verdict "dbatch sabotage" 1 '^DBATCH NOT EXACT$' env INK_DBATCH_SABOTAGE=1 INK_LANES=2 INK_PF=128 INK_DECODE=2 "$B" dbatch --ref ref/long.tokens -c 4096
verdict blockedinv 0 '^blockedinv exact' "$B" blockedinv --ref ref/long.tokens -c 4160
verdict "blockedinv sabotage" 1 '^BLOCKED NOT INVARIANT' env INK_BLOCKEDINV_SABOTAGE=1 "$B" blockedinv --ref ref/long.tokens -c 1024
verdict temp1 0 '^temp1 equivalent:' env -u INK_TEMP1_SABOTAGE "$B" temp1 --ref ref/long.tokens -c 4096
verdict "temp1 sabotage" 1 '^TEMP1 NOT EQUIVALENT$' env INK_TEMP1_SABOTAGE=1 "$B" temp1 --ref ref/long.tokens -c 4096
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
