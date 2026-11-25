#!/usr/bin/env bash
# Compile the reference scheduler and run its CacheScheduler traces and source mutations
# in TLC. The traces tier separately runs check-serve-step-traces.sh for the live planner.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
: "${TLA2TOOLS_JAR:?set TLA2TOOLS_JAR to the pinned tla2tools.jar}"
jar=$(cd "$(dirname "$TLA2TOOLS_JAR")" && pwd)/$(basename "$TLA2TOOLS_JAR")
java_bin=${TLC_JAVA:-java}
spec="$root/specs/CacheScheduler.tla"
orig_sched="$root/src/scheduler.rs"
driver="$root/scripts/scheduler-traces.rs"

[[ -r "$spec" ]] || { echo "ERROR: missing $spec" >&2; exit 1; }
[[ -r "$orig_sched" ]] || { echo "ERROR: missing $orig_sched" >&2; exit 1; }
[[ -r "$driver" ]] || { echo "ERROR: missing $driver" >&2; exit 1; }

work=$(mktemp -d "${TMPDIR:-/tmp}/inky-sched-traces.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

JOBS=${INKY_TRACE_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}
if [[ "$JOBS" -lt 1 ]]; then JOBS=1; fi

run_bundle() {
    local dir=$1 workers=${2:-1} mode=${3:-fast}
    cp "$spec" "$dir/CacheScheduler.tla"
    if [[ "$mode" == "c2" ]]; then
        (cd "$dir" && "$java_bin" -Xmx768m -XX:+UseParallelGC -XX:CICompilerCount=2 -cp "$jar" tlc2.TLC \
            -noGenerateSpecTE -workers "$workers" -metadir "$dir/states" \
            -config TraceCheck.cfg TraceCheck.tla) > "$dir/tlc.log" 2>&1
    else
        (cd "$dir" && "$java_bin" -Xmx512m -XX:+UseSerialGC -XX:TieredStopAtLevel=1 -cp "$jar" tlc2.TLC \
            -noGenerateSpecTE -workers 1 -metadir "$dir/states" \
            -config TraceCheck.cfg TraceCheck.tla) > "$dir/tlc.log" 2>&1
    fi
}

check_production() {
    local dir=$work/production
    mkdir -p "$dir"
    cp "$orig_sched" "$dir/scheduler.rs"
    mkdir -p "$dir/scheduler"
    cp "$root/src/scheduler/trace.rs" "$dir/scheduler/"
    cp "$driver" "$dir/main.rs"
    rustc --edition=2021 -C debug-assertions=on -C debuginfo=0 "$dir/main.rs" -o "$dir/generate"
    "$dir/generate" "$dir/bundle" 2> "$dir/walk.log"
    if ! run_bundle "$dir/bundle" 2 c2; then
        cat "$dir/walk.log" "$dir/bundle/tlc.log" >&2
        return 1
    fi
    if ! grep -Fq 'Model checking completed. No error has been found.' "$dir/bundle/tlc.log" || \
        grep -q '^Error:' "$dir/bundle/tlc.log"; then
        cat "$dir/bundle/tlc.log" >&2
        return 1
    fi
    {
        cat "$dir/walk.log"
        echo "production: $(grep -Em1 '^[0-9,]+ states generated' "$dir/bundle/tlc.log")"
    } > "$dir/result"
}

check_mutant() {
    local name=$1 old=$2 new=$3
    local dir=$work/$name status
    mkdir -p "$dir"
    mkdir -p "$dir/scheduler"
    cp "$root/src/scheduler/trace.rs" "$dir/scheduler/"
    cp "$driver" "$dir/main.rs"
    python3 - "$orig_sched" "$dir/scheduler.rs" "$old" "$new" <<'PY'
import pathlib, sys
orig, dst, old, new = sys.argv[1:5]
s = pathlib.Path(orig).read_text()
if s.count(old) != 1:
    sys.stderr.write(f"ERROR: mutant target not unique ({s.count(old)} matches)\n")
    sys.exit(1)
pathlib.Path(dst).write_text(s.replace(old, new, 1))
PY
    rustc --edition=2021 -C debug-assertions=on -C debuginfo=0 "$dir/main.rs" -o "$dir/generate"
    "$dir/generate" "$dir/bundle" mutant 2> "$dir/walk.log"
    if run_bundle "$dir/bundle"; then
        echo "ERROR: $name unexpectedly passed TLC trace checking" >&2
        cat "$dir/bundle/tlc.log" >&2
        return 1
    else
        status=$?
    fi
    if [[ $status != 12 && $status != 13 ]] || \
        ! grep -Eq 'Action property .* is violated|Temporal properties were violated|Invariant .* is violated' "$dir/bundle/tlc.log"; then
        cat "$dir/walk.log" "$dir/bundle/tlc.log" >&2
        echo "ERROR: $name failed without a TLC counterexample (exit $status)" >&2
        return 1
    fi
    echo "$name rejected (TLC exit $status): $(grep -Em1 'is violated|were violated' "$dir/bundle/tlc.log")" > "$dir/result"
}

names=() pids=()
has_slot() {
    local n=0 pid st
    for pid in "${pids[@]:-}"; do
        [[ -n "$pid" ]] || continue
        if [[ -r "/proc/$pid/stat" ]]; then
            read -r _ _ st _ < "/proc/$pid/stat" 2>/dev/null || continue
            [[ "$st" != "Z" ]] && n=$((n + 1))
        elif kill -0 "$pid" 2>/dev/null; then
            n=$((n + 1))
        fi
    done
    [[ "$n" -lt "$JOBS" ]]
}
start() {
    while ! has_slot; do
        sleep 0.05
    done
    names+=("$1")
    "${@:2}" &
    pids+=($!)
}

start production check_production

# 1. Reversed witness: freeing an InStep lane immediately on cancel instead of Cancelling barrier -> StepBlocksInMemoryAndOwned
start cancel-in-step-early-free check_mutant cancel-in-step-early-free \
    'self.lanes[lane].state = LaneState::Cancelling; // MUTANT_ANCHOR_CANCEL_IN_STEP' \
    'self.release_lane_to(lane, LaneState::Cancelled); return Ok(LaneState::Cancelled);'

# 2. Reversed witness: bypassing a waiting lane past ceiling W -> BypassBoundHolds
start bypass-bound-exceeded check_mutant bypass-bound-exceeded \
    'bypassed.iter().any(|&w| self.lanes[w].wait_ticks >= self.cfg.max_bypass) // MUTANT_ANCHOR_BYPASS_CEILING' \
    'false'

# 3. Reversed witness: dropping wait_ticks increment on bypassed waiting lanes in build_step -> Conforms
start drop-step-bypass-tick check_mutant drop-step-bypass-tick \
    'self.lanes[w].wait_ticks += 1; // MUTANT_ANCHOR_STEP_BYPASS' \
    'let _ = w;'

# 4. Reversed witness: dropping wait_ticks increment on bypassed older queued lanes in admit -> Conforms
start drop-admit-bypass-tick check_mutant drop-admit-bypass-tick \
    'self.lanes[q].wait_ticks += 1; // MUTANT_ANCHOR_ADMIT_BYPASS' \
    'let _ = q;'

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        cat "$work/${names[$i]}/result"
    else
        echo "FAILED: ${names[$i]}" >&2
        failed=1
    fi
done
[[ $failed = 0 ]] || exit 1
echo "scheduler traces conform to CacheScheduler.tla; all $(( ${#names[@]} - 1 )) reversed-witness mutations rejected."
