#!/usr/bin/env bash
# CPU-only: no Cargo, Metal, root, sysctl or network access.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
: "${TLA2TOOLS_JAR:?set TLA2TOOLS_JAR to the pinned tla2tools.jar}"
jar=$(cd "$(dirname "$TLA2TOOLS_JAR")" && pwd)/$(basename "$TLA2TOOLS_JAR")
work=$(mktemp -d)
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
run_trace() {
    : > "$work/tlc.log"
    rustc --edition=2021 "$work/main.rs" -o "$work/generate" || return 1
    "$work/generate" "$work/saved" "${1:-positive}" > "$work/TraceData.tla" 2> "$work/faults.log" || { cat "$work/faults.log" >&2; return 1; }
    (cd "$work" && "${TLC_JAVA:-java}" -Xmx1g -XX:+UseParallelGC -cp "$jar" tlc2.TLC -workers 1 -config TraceCheck.cfg TraceCheck.tla) > "$work/tlc.log" 2>&1
}
if ! run_trace; then cat "$work/tlc.log"; exit 1; fi
grep -q 'Model checking completed. No error has been found.' "$work/tlc.log"
cat "$work/tlc.log"
tail -1 "$work/faults.log"
# Each named mutation changes the extracted production source, never the oracle.
cp "$work/mode.rs" "$work/mode.original"
cp "$work/startup.rs" "$work/startup.original"
for mutant in early-release no-rollback no-readback lost-seed skip-keep; do
    cp "$work/mode.original" "$work/mode.rs"
    cp "$work/startup.original" "$work/startup.rs"
    python3 "$root/scripts/mutate-memmode.py" "$mutant" "$work"
    if run_trace "$mutant"; then echo "ERROR: $mutant passed trace checking" >&2; exit 1; else status=$?; fi
    if [[ $status != 12 && $status != 13 ]] || ! grep -Eq 'Action property .* is violated|Invariant .* is violated' "$work/tlc.log"; then
        cat "$work/tlc.log"; echo "ERROR: $mutant failed without a conformance counterexample (exit $status)" >&2; exit 1
    fi
    echo "$mutant rejected (TLC exit $status)"
done
echo '32 deterministic random walks x 192 operations passed; all five Rust mutations rejected.'
