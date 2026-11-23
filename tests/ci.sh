#!/bin/zsh
# tests/ci.sh: what CI runs on an M5 Mac. Needs the model at its default path and the llama.cpp reference logits in
# $INKY_REF (default ~/inky-ci/ref; tests/make-reference.sh writes them).
#  1. agreement with llama.cpp: every checked position's argmax agrees and the worst KL is at most 0.01, on the decode
#     path (a short prompt) and the prefill path (a 2394-token prompt).
#  2. batch invariance (inky binv) and lane independence (inky lanes).
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
$B binv --ref ref/long.tokens -c 4096 2>&1 | tail -1 | tee /dev/stderr | grep -q '^batch invariant' || { echo "FAIL: binv"; fail=1; }
$B lanes --ref ref/long.tokens -c 4096 2>&1 | tail -1 | tee /dev/stderr | grep -q '^lanes independent' || { echo "FAIL: lanes"; fail=1; }
chip=$(sysctl -n machdep.cpu.brand_string | sed 's/Apple //; s/ /-/g' | tr A-Z a-z)
if [ -f tests/golden-$chip.txt ]; then
    tests/golden.sh $B > /tmp/inky-golden.$$ 2>&1
    if diff tests/golden-$chip.txt /tmp/inky-golden.$$; then echo "golden: identical to tests/golden-$chip.txt"; else echo "FAIL: golden hashes changed"; fail=1; fi
    rm -f /tmp/inky-golden.$$
else
    echo "golden: no tests/golden-$chip.txt for this chip, skipped"
fi
exit $fail
