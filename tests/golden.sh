#!/bin/zsh
# tests/golden.sh [inky binary]: hashes of every logit the engine produces in a set of fixed scenarios, one line each.
# A refactor that must not change behaviour passes when its output is byte-identical to the output before it, on the
# same machine (different chips may sum in a different order). Needs the model at its default path and a Metal GPU.
set -e
B=${1:-./target/release/inky}
D=$(mktemp -d); trap 'rm -rf $D' EXIT
nv=200058
# a reference of one zero logit row: check then hashes every step and compares only the last position (ignored here)
export INK_ALLOW_MISMATCH=1
for f in haiku long; do cp ref/$f.tokens $D/$f.tokens; head -c $((nv * 4)) /dev/zero > $D/$f.logits; done
h() { grep -o 'hash [0-9a-f]*' | sed 's/hash //'; }
echo "decode haiku         $($B check --ref $D/haiku -c 4096 2>&1 | h)"
echo "decode long          $($B check --ref $D/long -c 4096 2>&1 | h)"
echo "prefill+decode long  $(INK_PF=1500 $B check --ref $D/long -c 4096 2>&1 | h)"
echo "kv8 decode haiku     $($B check --kv8 --ref $D/haiku -c 4096 2>&1 | h)"
echo "kv-disk decode long  $(INK_TIER_HOT=256 $B check --kv-disk --ref $D/long -c 4096 2>&1 | h)"
echo "binv long            $($B binv --ref ref/long.tokens -c 4096 2>&1 | grep -c same) splits same"
echo "run greedy           $($B run -p 'Write a function in C that reverses a singly linked list.' --greedy -n 300 --effort low 2>/dev/null | shasum | cut -c1-16)"
echo "run sampled          $($B run -p 'Name three prime numbers.' --temp 1 --seed 7 -n 200 --effort low 2>/dev/null | shasum | cut -c1-16)"
echo "chat two turns       $(printf 'Say hi.\nNow say bye.\n' | $B chat --greedy -n 120 --effort none 2>/dev/null | shasum | cut -c1-16)"
echo "lanes long           $($B lanes --ref ref/long.tokens -c 4096 2>&1 | tail -1)"
