#!/usr/bin/env bash
# CPU-only Layer 4 scheduler trace conformance gate against specs/CacheScheduler.tla:
# zero Cargo, Metal, root, or network access. Compiles real src/scheduler.rs directly
# with rustc, checks multi-seed + edge-case traces in TLC (requiring complete drain,
# runtime invariant checks, and seeded disk writeback/restore/cancel-in-restore coverage),
# and pins each single-line production mutation to the exact TLA+ property it violates.
# The traces tier separately runs check-serve-step-traces.sh for the live planner.
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

run_bundle() {
    local dir=$1
    cp "$spec" "$dir/CacheScheduler.tla"
    (cd "$dir" && "$java_bin" -Xmx4g -XX:+UseParallelGC -cp "$jar" tlc2.TLC \
        -noGenerateSpecTE -workers 1 -metadir "$dir/states" \
        -config TraceCheck.cfg TraceCheck.tla) > "$dir/tlc.log" 2>&1
}

check_production() {
    local dir=$work/production
    mkdir -p "$dir"
    cp "$orig_sched" "$dir/scheduler.rs"
    mkdir -p "$dir/scheduler"
    cp "$root/src/scheduler/trace.rs" "$dir/scheduler/"
    cp "$driver" "$dir/main.rs"
    rustc --edition=2021 -D warnings -O "$dir/main.rs" -o "$dir/generate" 2> "$dir/rustc.log" || {
        cat "$dir/rustc.log" >&2
        return 1
    }
    "$dir/generate" "$dir/bundle" 2> "$dir/walk.log" || {
        cat "$dir/walk.log" >&2
        return 1
    }
    if grep -q 'COVERAGE GAP' "$dir/walk.log"; then
        cat "$dir/walk.log" >&2
        return 1
    fi
    if ! run_bundle "$dir/bundle"; then
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
    local name=$1 expected=$2 old=$3 new=$4
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
    if cmp -s "$dir/scheduler.rs" "$orig_sched"; then
        echo "ERROR: $name changed nothing" >&2
        return 1
    fi
    rustc --edition=2021 -O "$dir/main.rs" -o "$dir/generate" 2> "$dir/rustc.log" || {
        cat "$dir/rustc.log" >&2
        return 1
    }
    "$dir/generate" "$dir/bundle" --mutant 2> "$dir/walk.log" || {
        cat "$dir/walk.log" >&2
        return 1
    }
    if run_bundle "$dir/bundle"; then
        echo "ERROR: $name unexpectedly passed TLC trace checking" >&2
        cat "$dir/bundle/tlc.log" >&2
        return 1
    else
        status=$?
    fi
    if [[ $status != 12 && $status != 13 ]] || ! grep -Fxq "Error: $expected" "$dir/bundle/tlc.log"; then
        cat "$dir/walk.log" "$dir/bundle/tlc.log" >&2
        echo "ERROR: $name did not fail with pinned 'Error: $expected' (exit $status)" >&2
        return 1
    fi
    echo "$name rejected (TLC exit $status): Error: $expected" > "$dir/result"
}

names=() pids=()
start() { names+=("$1"); "${@:2}" & pids+=($!); }

start production check_production

# 1. Freeing an InStep lane immediately on cancel instead of Cancelling barrier -> StepBlocksInMemoryAndOwned
start cancel-in-step-early-free check_mutant cancel-in-step-early-free \
    'Invariant StepBlocksInMemoryAndOwned is violated.' \
    'self.lanes[lane].state = LaneState::Cancelling; // MUTANT_ANCHOR_CANCEL_IN_STEP' \
    'if true { self.release_lane_to(lane, LaneState::Cancelled); return Ok(LaneState::Cancelled); }'

# 2. Freeing a Restoring lane with active disk I/O immediately on cancel (`U88`) -> StepBlocksInMemoryAndOwned
start cancel-in-restore-early-free check_mutant cancel-in-restore-early-free \
    'Invariant StepBlocksInMemoryAndOwned is violated.' \
    'self.lanes[lane].state = LaneState::Cancelling; // MUTANT_ANCHOR_CANCEL_IN_RESTORE' \
    'if true { let io = self.lanes[lane].restore_io_block.clone(); self.release_lane_to(lane, LaneState::Cancelled); self.lanes[lane].restore_io_block = io; return Ok(LaneState::Cancelled); }'

# 3. Bypassing a waiting lane past ceiling W -> BypassBoundHolds
start bypass-bound-exceeded check_mutant bypass-bound-exceeded \
    'Invariant BypassBoundHolds is violated.' \
    'bypassed.iter().any(|&w| self.lanes[w].wait_ticks >= self.cfg.max_bypass) // MUTANT_ANCHOR_BYPASS_CEILING' \
    '{ let _ = bypassed; false }'

# 4. Dropping wait_ticks increment on bypassed waiting lanes in build_step -> BypassBoundHolds
start drop-step-bypass-tick check_mutant drop-step-bypass-tick \
    'Invariant BypassBoundHolds is violated.' \
    'self.lanes[w].wait_ticks += 1; // MUTANT_ANCHOR_STEP_BYPASS' \
    'let _ = w;'

# 5. Dropping wait_ticks increment on bypassed older queued lanes in admit -> BypassBoundHolds
start drop-admit-bypass-tick check_mutant drop-admit-bypass-tick \
    'Invariant BypassBoundHolds is violated.' \
    'self.lanes[q].wait_ticks += 1; // MUTANT_ANCHOR_ADMIT_BYPASS' \
    'let _ = q;'

# 6. Silent no-op stutter on RestoreBlock (`U68`) -> ActionConforms
start noop-restore-stutter check_mutant noop-restore-stutter \
    'Action property ActionConforms is violated.' \
    'rec.tier = BlockTier::Both; // MUTANT_ANCHOR_SCHED_RESTORE_BLOCK' \
    'let _ = rec;'

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
echo "scheduler traces conform to CacheScheduler.tla; all $(( ${#names[@]} - 1 )) pinned reversed-witness mutations rejected."
