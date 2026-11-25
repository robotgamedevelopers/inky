#!/bin/sh
# Verify the output of this CI part, independently of the list the runner selected.
# `tests/check-layout-coverage.sh --self-test` also proves that `specs/run_tlc.zsh` lists the
# canonical checks when `zsh` is absent from PATH and that this verifier fails closed on
# empty/missing model logs or missing witness results (`t-audit-cache-layout-runner` & `t-audit-cache-prefix-nightly`).
set -eu

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)

if [ "${1:-}" = "--self-test" ]; then
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/inky-layout-cov.XXXXXX")
    trap 'rm -rf "$tmp"' EXIT HUP INT TERM
    mkdir -p "$tmp/bin"
    for cmd in bash sh env dirname pwd basename tr sed awk grep wc mktemp rm cat mkdir; do
        p=$(command -v "$cmd" 2>/dev/null || true)
        [ -n "$p" ] && ln -sf "$p" "$tmp/bin/$cmd"
    done
    if PATH="$tmp/bin" command -v zsh >/dev/null 2>&1; then
        echo "self-test setup failed: zsh still in stripped PATH" >&2
        exit 1
    fi
    list_pr=$(env PATH="$tmp/bin" TLC_NIGHTLY= TLA_NIGHTLY=0 sh "$root/specs/run_tlc.zsh" --list)
    n_pr=$(printf '%s\n' "$list_pr" | grep -c . || true)
    [ "$n_pr" -ge 50 ] || { echo "self-test: no-zsh PR check list too short ($n_pr)" >&2; exit 1; }
    list_ni=$(env PATH="$tmp/bin" TLA_NIGHTLY=1 TLC_NIGHTLY_ONLY=1 sh "$root/specs/run_tlc.zsh" --list)
    for req in PrefixTreeNightly PrefixTree StartTails3 mutant-no-evict; do
        printf '%s\n' "$list_ni" | grep -Fxq "$req" || { echo "self-test: layout-nightly missing $req" >&2; exit 1; }
    done
    # Empty directory must fail closed on every part.
    mkdir -p "$tmp/empty"
    for part in blocks grouped layout-nightly; do
        if sh "$0" "$tmp/empty" "$part" >/dev/null 2>&1; then
            echo "self-test: $part succeeded with zero logs" >&2
            exit 1
        fi
    done
    echo "layout-coverage self-test: no-zsh list ($n_pr PR checks) and fail-closed log checks ok"
    exit 0
fi

output=$1
part=$2
case $part in
    blocks) set -- BlockLayout3 PrefixTreeFast PrefixTreeTrace ;;
    grouped) set -- GroupedLayout3 RestageOutputs PoisonGate3 ScratchTail3 SplitArena3 ;;
    layout-nightly) set -- StartTails3 PrefixTreeNightly ;;
    *) echo "Unknown layout part: $part" >&2; exit 2 ;;
esac
for config do
    if [ ! -s "$output/$config.log" ] || ! grep -Fxq 'Model checking completed. No error has been found.' "$output/$config.log"; then
        echo "Missing successful TLC result for $config" >&2
        exit 1
    fi
done
if [ "$part" = layout-nightly ]; then
    [ -s "$output/PrefixTree.log" ] &&
    grep -Eq '^Progress: .*256 traces generated \(trace length: mean=128,' "$output/PrefixTree.log" ||
        { echo 'Missing completed 21-node PrefixTree simulation' >&2; exit 1; }
    for name in mutant-no-evict mutant-full-match mutant-ref-leak; do
        test -s "$output/$name.result" || { echo "Missing checked witness for $name" >&2; exit 1; }
    done
    echo "layout-nightly status: PrefixTreeNightly=PASS (exhaustive), PrefixTree=PASS (21-node simulation, 256 traces), StartTails3=PASS, 3 trace mutants=PASS"
fi
