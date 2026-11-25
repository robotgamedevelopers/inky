#!/usr/bin/env bash
# CPU-only: no Cargo, Metal, root, sysctl or network access.
# The unmutated walk and the named mutations are independent: each gets its own directory, binary and TLC, and they
# run in parallel. `check-memmode-traces.sh --list` prints the run names (production and each mutant) and runs nothing;
# scripts/check-tla-tiers.sh reads it.
set -euo pipefail
mutants=(early-release no-rollback no-readback lost-seed skip-keep)
case ${1:-} in
    '') ;;
    --list) printf '%s\n' production "${mutants[@]}"; exit 0 ;;
    *) echo "usage: check-memmode-traces.sh [--list]" >&2; exit 2 ;;
esac
root=$(cd "$(dirname "$0")/.." && pwd)
: "${TLA2TOOLS_JAR:?set TLA2TOOLS_JAR to the pinned tla2tools.jar}"
jar=$(cd "$(dirname "$TLA2TOOLS_JAR")" && pwd)/$(basename "$TLA2TOOLS_JAR")
work=$(mktemp -d "${TMPDIR:-/tmp}/inky-memmode-traces.XXXXXX")
trap 'rm -rf "$work"' EXIT
cp "$root"/specs/memmode/{MemMode.tla,TraceCheck.tla,TraceCheck.cfg,TraceProtocol.tla} "$work/"
cp "$root/scripts/memmode-traces.rs" "$work/main.rs"
python3 - "$root/src/memd.rs" "$work/mode.rs" <<'PY'
import pathlib, sys
source = pathlib.Path(sys.argv[1]).read_text()
start = source.index('#[derive(Default)]\npub struct Mode')
end = source.index('\nfn get_limit()', start)
pathlib.Path(sys.argv[2]).write_text(source[start:end])
start = source.index('\nfn read_saved(')
end = source.index('\nfn checked_directory(', start)
pathlib.Path(sys.argv[2]).with_name('startup.rs').write_text(source[start:end])
PY
# run_trace DIR [MUTANT]: compile the walk against DIR's extract, record it, replay it through TLC.
run_trace() {
    local dir=$1
    : > "$dir/tlc.log"
    rustc --edition=2021 "$dir/main.rs" -o "$dir/generate" 2> "$dir/rustc.log" || { cat "$dir/rustc.log" >&2; return 1; }
    "$dir/generate" "$dir/saved" "${2:-positive}" > "$dir/TraceData.tla" 2> "$dir/faults.log" || { cat "$dir/faults.log" >&2; return 1; }
    (cd "$dir" && "${TLC_JAVA:-java}" -Xmx1g -XX:+UseParallelGC -cp "$jar" tlc2.TLC -noGenerateSpecTE -workers 1 -metadir "$dir/states" \
        -config TraceCheck.cfg TraceCheck.tla) > "$dir/tlc.log" 2>&1
}
# setup NAME: a private copy of the models, the walk and the unmutated extract.
setup() {
    mkdir "$work/$1"
    cp "$work"/{MemMode.tla,TraceCheck.tla,TraceCheck.cfg,TraceProtocol.tla,main.rs,mode.rs,startup.rs} "$work/$1/"
}
check_production() {
    local dir=$work/production
    setup production
    if ! run_trace "$dir"; then cat "$dir/tlc.log" >&2; return 1; fi
    grep -q 'Model checking completed. No error has been found.' "$dir/tlc.log" || { cat "$dir/tlc.log" >&2; return 1; }
    { cat "$dir/tlc.log"; tail -1 "$dir/faults.log"; } > "$dir/result"
}
# Each named mutation changes the extracted production source, never the oracle.
check_mutant() {
    local mutant=$1 dir=$work/$1 status
    setup "$mutant"
    python3 "$root/scripts/mutate-memmode.py" "$mutant" "$dir"
    if cmp -s "$dir/mode.rs" "$work/mode.rs" && cmp -s "$dir/startup.rs" "$work/startup.rs"; then
        echo "ERROR: $mutant changed nothing" >&2; return 1
    fi
    if run_trace "$dir" "$mutant"; then echo "ERROR: $mutant passed trace checking" >&2; return 1; else status=$?; fi
    if [[ $status != 12 && $status != 13 ]] || ! grep -Eq 'Action property .* is violated|Invariant .* is violated' "$dir/tlc.log"; then
        cat "$dir/tlc.log" >&2; echo "ERROR: $mutant failed without a conformance counterexample (exit $status)" >&2; return 1
    fi
    echo "$mutant rejected (TLC exit $status)" > "$dir/result"
}
names=() pids=()
check_production & names+=(production) pids+=($!)
for mutant in "${mutants[@]}"; do
    check_mutant "$mutant" & names+=("$mutant") pids+=($!)
done
failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then cat "$work/${names[$i]}/result"; else echo "FAILED: ${names[$i]}" >&2; failed=1; fi
done
[[ $failed = 0 ]] || exit 1
echo "32 deterministic random walks x 192 operations passed; all ${#mutants[@]} Rust mutations rejected."
