#!/usr/bin/env bash
# Bind src/blockbook.rs to specs/StartTails.tla: replay deterministic random walks
# of the production file through TLC, and require each named mutation of that
# file to be rejected. CPU-only: rustc and Java, no Cargo, Metal or network.
# The unmutated run and the mutations are independent and run in parallel.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
: "${TLA2TOOLS_JAR:?set TLA2TOOLS_JAR to the pinned tla2tools.jar}"
jar=$(cd "$(dirname "$TLA2TOOLS_JAR")" && pwd)/$(basename "$TLA2TOOLS_JAR")
work=$(mktemp -d "${TMPDIR:-/tmp}/inky-blockbook.XXXXXX")
trap 'rm -rf "$work"' EXIT
original=$work/blockbook.original
cp "$root/src/blockbook.rs" "$original"

# run_trace DIR: compile the walk against DIR/blockbook.rs, record it, replay it.
run_trace() {
    local dir=$1
    cp "$root"/specs/{StartTails.tla,StartTailsTrace.tla,StartTailsTrace.cfg} "$root/src/blocks.rs" "$dir/"
    cp "$root/scripts/blockbook-traces.rs" "$dir/main.rs"
    : > "$dir/tlc.log"
    rustc --edition=2021 --cfg standalone_blocks "$dir/main.rs" -o "$dir/generate" 2> "$dir/rustc.log" || { cat "$dir/rustc.log" >&2; return 1; }
    "$dir/generate" > "$dir/StartTailsTraceData.tla" 2> "$dir/walk.log" || { cat "$dir/walk.log" >&2; return 1; }
    (cd "$dir" && "${TLC_JAVA:-java}" -Xmx2g -XX:+UseParallelGC -cp "$jar" tlc2.TLC \
        -noGenerateSpecTE -workers 1 -metadir "$dir/states" \
        -config StartTailsTrace.cfg StartTailsTrace.tla) > "$dir/tlc.log" 2>&1
}

# check_mutant NAME OLD NEW: replace one line of the production file, never the
# oracle, and require TLC to reject the walk with a counterexample.
check_mutant() {
    local name=$1 old=$2 new=$3 dir=$work/$1 src status
    mkdir "$dir"
    python3 -c 'import pathlib, sys
orig, dst, old, new = sys.argv[1:5]
s = pathlib.Path(orig).read_text()
if s.count(old) != 1:
    sys.exit(1)
pathlib.Path(dst).write_text(s.replace(old, new, 1))
' "$original" "$dir/blockbook.rs" "$old" "$new" || { echo "ERROR: $name target is not one line of src/blockbook.rs" >&2; return 1; }
    if cmp -s "$dir/blockbook.rs" "$original"; then echo "ERROR: $name changed nothing" >&2; return 1; fi
    if run_trace "$dir"; then echo "ERROR: $name passed trace checking" >&2; return 1; else status=$?; fi
    if [[ $status != 12 && $status != 13 ]] || ! grep -Eq 'Action property .* is violated|Temporal properties were violated|Invariant .* is violated' "$dir/tlc.log"; then
        cat "$dir/tlc.log" "$dir/walk.log" >&2; echo "ERROR: $name failed without a conformance counterexample (exit $status)" >&2; return 1
    fi
    echo "$name rejected: $(grep -Em1 'is violated|were violated' "$dir/tlc.log")" > "$dir/result"
}

check_production() {
    local dir=$work/production
    mkdir "$dir"
    cp "$original" "$dir/blockbook.rs"
    if ! run_trace "$dir"; then cat "$dir/tlc.log" >&2; return 1; fi
    grep -Fq 'Model checking completed. No error has been found.' "$dir/tlc.log" || { cat "$dir/tlc.log" >&2; return 1; }
    if grep -q 'COVERAGE GAP' "$dir/walk.log"; then cat "$dir/walk.log" >&2; return 1; fi
    { grep -E '^[0-9,]+ states generated' "$dir/tlc.log"; cat "$dir/walk.log"; } > "$dir/result"
}

names=() pids=()
start() { names+=("$1"); "${@:2}" & pids+=($!); }
start production check_production
# (a) a start point that does not hold its window blocks
start unheld-window check_mutant unheld-window 'for (_, e) in table.entries() { self.hold(e)?; }' \
    'for (_, e) in table.entries() { if e.kind() == BlockKind::Full { self.hold(e)?; } }'
# (a) pruning that releases a window slot outright, under a live start point
start pruned-under-start check_mutant pruned-under-start 'for (b, e) in pruned { self.lanes[lane].table.set(BlockKind::Window, b, None)?; self.release(e)?; }' \
    'for (b, e) in pruned { self.lanes[lane].table.set(BlockKind::Window, b, None)?; while self.pools.of(e.kind()).record(e.slot())?.refs() > 0 { self.release(e)?; } }'
# (b) a read one position past what the lane wrote
start overread check_mutant overread 'Ok(Some(l.pos.saturating_sub(start).min(self.geo.block)))' \
    'Ok(Some((l.pos.saturating_sub(start) + 1).min(self.geo.block)))'
# (c) a reused slot handed out without zeroing
start no-zero check_mutant no-zero 'gpu.zero(kind, slot);' '{}'
# a finished lane that keeps its tails
start kept-tail check_mutant kept-tail '        self.release_table(&table)?;' \
    '        for (_, e) in table.entries() { if self.pools.of(e.kind()).record(e.slot())?.sealed() { self.release(e)?; } }'
# every-16 points on the wrong boundaries
start every-off check_mutant every-off 'if (n / geo.block) % geo.every == 0 { reasons |= EVERY; }' \
    'if (n / geo.block) % geo.every == 1 { reasons |= EVERY; }'

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then cat "$work/${names[$i]}/result"; else echo "FAILED: ${names[$i]}" >&2; failed=1; fi
done
[[ $failed = 0 ]] || exit 1
echo "blockbook traces conform to StartTails; all $(( ${#names[@]} - 1 )) mutations of src/blockbook.rs rejected."
