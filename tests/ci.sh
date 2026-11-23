#!/bin/zsh
# tests/ci.sh: what CI runs on an M5 Mac. Needs the model at its default path and the llama.cpp reference logits in
# $INKY_REF (default ~/inky-ci/ref; tests/make-reference.sh writes them).
#  1. agreement with llama.cpp: every checked position's argmax agrees and the worst KL is at most 0.01, on the decode
#     path (a short prompt) and the prefill path (a 2394-token prompt).
#  2. batch invariance (inky binv), lane independence (inky lanes), steps over spans (inky spans) and decode steps over
#     lanes (inky dbatch): every lane identical to itself alone; decode steps also agree with llama.cpp.
#  3. tests/golden.sh is byte-identical to tests/golden-<chip>.txt: the logits did not change. A change that changes them
#     on purpose updates that file in the same commit and says why.
set -e
cd "${0:A:h}/.."
B=./target/release/inky
REF=${INKY_REF:-$HOME/inky-ci/ref}
fail=0
agree() {  # name, env, ref
    local out
    if ! out=$(env "$2" "$B" check --ref "$REF/$3" -c 4096 2>&1); then
        print -r -- "$out"
        echo "FAIL: $1"
        fail=1
        return
    fi
    print -r -- "$out" | awk -v name="$1" '
        /^argmax agreement / {
            print name ": " $0
            seen++
            split($3, a, "/")
            if (NF != 6 || $3 !~ /^[0-9]+\/[0-9]+$/ || a[2]+0 == 0 || a[1]+0 != a[2]+0 ||
                $4 != "worst" || $5 != "KL" || $6 !~ /^[0-9]+([.][0-9]+)?$/ || $6+0 > 0.01) bad=1
        }
        END { exit (seen != 1 || bad) }
    ' || { echo "FAIL: $1"; fail=1; }
}
agree "decode, short prompt vs llama.cpp " "INK_PF=0" short
agree "prefill, long prompt vs llama.cpp " "INK_PF=2380" long
agree "decode steps, short prompt vs llama.cpp " "INK_DEC=1" short
$B binv --ref ref/long.tokens -c 4096 2>&1 | tail -1 | tee /dev/stderr | grep -q '^batch invariant' || { echo "FAIL: binv"; fail=1; }
$B lanes --ref ref/long.tokens -c 4096 2>&1 | tail -1 | tee /dev/stderr | grep -q '^lanes independent' || { echo "FAIL: lanes"; fail=1; }
INK_LANES=1,2,17,65 $B spans --ref ref/long.tokens -c 4096 2>&1 | tail -1 | tee /dev/stderr | grep -q '^spans invariant' || { echo "FAIL: spans"; fail=1; }
# negative control: the sabotaged spans run (one span bound to the wrong lane's state) must FAIL; if it passes, the gate above proves nothing
if INK_SPANS_SABOTAGE=1 INK_LANES=2 $B spans --ref ref/long.tokens -c 4096; then echo "FAIL: spans sabotage not detected"; fail=1; else echo "spans sabotage: detected, as required"; fi
INK_LANES=1,8,33 INK_DECODE=8 $B dbatch --ref ref/long.tokens -c 4096 2>&1 | tail -1 | tee /dev/stderr | grep -q '^dbatch exact' || { echo "FAIL: dbatch"; fail=1; }
chip=$(sysctl -n machdep.cpu.brand_string | sed 's/Apple //; s/ /-/g' | tr A-Z a-z)
if [ -f tests/golden-$chip.txt ]; then
    tests/golden.sh $B > /tmp/inky-golden.$$ 2>&1
    if diff tests/golden-$chip.txt /tmp/inky-golden.$$; then echo "golden: identical to tests/golden-$chip.txt"; else echo "FAIL: golden hashes changed"; fail=1; fi
    rm -f /tmp/inky-golden.$$
else
    echo "golden: no tests/golden-$chip.txt for this chip, skipped"
fi
exit $fail
