#!/bin/sh
# Check the TLC output from this run. A skipped runner or wrong nightly selector
# used to give a successful exit without checking any of the expected states.
set -eu
output=$1
set -- BlockLayout3 PrefixTreeFast
if [ -n "${TLC_NIGHTLY:-}" ]; then
    set -- "$@" StartTails3 PrefixTreeNightly
fi
for config do
    if ! grep -Fxq 'Model checking completed. No error has been found.' "$output/$config.log"; then
        echo "Missing successful TLC result for $config" >&2
        exit 1
    fi
done
if [ -n "${TLC_NIGHTLY:-}" ] && ! grep -Eq '^Progress: .*256 traces generated \(trace length: mean=128,' "$output/PrefixTreeSimulation.log"; then
    echo "Missing completed 21-node PrefixTree simulation" >&2
    exit 1
fi
