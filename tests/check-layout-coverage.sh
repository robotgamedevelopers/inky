#!/bin/sh
# Verify the output of this CI part, independently of the list the runner selected.
set -eu
output=$1
part=$2
case $part in
    blocks) set -- BlockLayout3 PrefixTreeFast PrefixTreeTrace ;;
    grouped) set -- GroupedLayout3 RestageOutputs PoisonGate3 ScratchTail3 ;;
    layout-nightly) set -- StartTails3 PrefixTreeNightly ;;
    *) echo "Unknown layout part: $part" >&2; exit 2 ;;
esac
for config do
    if ! grep -Fxq 'Model checking completed. No error has been found.' "$output/$config.log"; then
        echo "Missing successful TLC result for $config" >&2
        exit 1
    fi
done
if [ "$part" = layout-nightly ]; then
    grep -Eq '^Progress: .*256 traces generated \(trace length: mean=128,' "$output/PrefixTree.log" ||
        { echo 'Missing completed 21-node PrefixTree simulation' >&2; exit 1; }
    for name in mutant-no-evict mutant-full-match mutant-ref-leak; do
        test -s "$output/$name.result" || { echo "Missing checked witness for $name" >&2; exit 1; }
    done
fi
