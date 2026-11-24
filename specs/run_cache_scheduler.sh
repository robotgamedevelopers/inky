#!/bin/sh
# Run CacheScheduler safety + liveness checks on a bench or CI runner.
# Usage: specs/run_cache_scheduler.sh [full|pr|nightly-only]
#   - full (default, or CACHE_SCHEDULER_TIER=full|nightly): all 14 configurations
#   - pr (or CACHE_SCHEDULER_TIER=pr): fast <10s parallel per-PR set (3 exhaustive
#     positive models + all 8 reversed witnesses)
#   - nightly-only: the 3 larger state-space positive sweeps (Queue, TwoPrefixes,
#     DecodePriority)
set -eu

export PATH="/opt/nova/sdk/bin:$HOME/sdk/bin:$PATH"
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
tier=${1:-${CACHE_SCHEDULER_TIER:-full}}
case "$tier" in
    full|nightly|pr|nightly-only) ;;
    *) echo "usage: $0 [full|pr|nightly-only]" >&2; exit 2 ;;
esac

if [ -n "${TLC_JAR:-}" ]; then
    jar=$TLC_JAR
elif [ -n "${TLA2TOOLS_JAR:-}" ]; then
    jar=$TLA2TOOLS_JAR
elif [ -r "/opt/tla/tla2tools.jar" ]; then
    jar="/opt/tla/tla2tools.jar"
else
    jar="$HOME/sdk/tla/tla2tools.jar"
fi

if [ -n "${TLC_JAVA:-}" ]; then
    java_bin=$TLC_JAVA
elif [ -x "$HOME/sdk/bin/java" ]; then
    java_bin="$HOME/sdk/bin/java"
else
    java_bin="java"
fi

output=${TLC_OUTPUT:-"$PWD/tlc-cache-scheduler-results"}
workers=${TLC_WORKERS:-8}
mkdir -p "$output"
output=$(CDPATH= cd -- "$output" && pwd)

[ -r "$jar" ] || { echo "Missing TLC jar: $jar" >&2; exit 1; }

now_ms() {
    if date +%s%3N 2>/dev/null | grep -Eq '^[0-9]+$'; then
        date +%s%3N
    else
        perl -MTime::HiRes=time -e 'printf "%d\n", int(time()*1000)'
    fi
}

exec_tlc() {
    name=$1
    module=$2
    run_workers=$3
    heap=$4
    log="$output/$name.log"
    time_file="$output/$name.ms"
    dir="$output/$name-work"
    mkdir -p "$dir"
    cp "$root/$module.tla" "$root/$name.cfg" "$dir/"
    t0=$(now_ms)
    if (cd "$dir" && "$java_bin" -XX:+UseParallelGC "-Xmx$heap" -cp "$jar" tlc2.TLC \
        -noGenerateSpecTE -workers "$run_workers" -metadir "$dir/states" \
        -config "$name.cfg" "$module.tla") >"$log" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    t1=$(now_ms)
    printf "%s %s\n" "$rc" "$(( t1 - t0 ))" >"$time_file"
}

verify_run() {
    name=$1
    kind=${2:-pass}
    expected=${3:-}
    log="$output/$name.log"
    read -r rc elapsed_ms <"$output/$name.ms"
    elapsed_s=$(awk -v ms="$elapsed_ms" 'BEGIN { printf "%.3f", ms / 1000.0 }')

    ok=true
    grep -Eq "^[0-9,]+ states generated, [0-9,]+ distinct states found" "$log" || ok=false
    grep -q "^Finished in " "$log" || ok=false

    case "$kind" in
        pass)
            [ "$rc" -eq 0 ] || ok=false
            if grep -q "^Error:" "$log"; then ok=false; fi
            grep -Fxq "Model checking completed. No error has been found." "$log" || ok=false
            grep -q "0 states left on queue\." "$log" || ok=false
            label="PASS"
            ;;
        invariant)
            [ "$rc" -ne 0 ] || ok=false
            [ "$(grep -c "^Error: " "$log" || true)" = "2" ] || ok=false
            grep -Fxq "Error: Invariant $expected is violated." "$log" || ok=false
            grep -Fxq "Error: The behavior up to this point is:" "$log" || ok=false
            grep -q "^State 2:" "$log" || ok=false
            label="$expected violated"
            ;;
        action)
            [ "$rc" -ne 0 ] || ok=false
            [ "$(grep -c "^Error: " "$log" || true)" = "2" ] || ok=false
            grep -Fxq "Error: Action property $expected is violated." "$log" || ok=false
            grep -Fxq "Error: The behavior up to this point is:" "$log" || ok=false
            grep -q "^State 2:" "$log" || ok=false
            label="$expected violated"
            ;;
        temporal)
            [ "$rc" -ne 0 ] || ok=false
            grep -Eq "^Error: Temporal propert(ies were|y [A-Za-z0-9_]+ was) violated\.$" "$log" || ok=false
            grep -Fq "$expected" "$log" || ok=false
            label="LiveRequestResolves violated"
            ;;
        *)
            ok=false
            label="unknown"
            ;;
    esac

    if [ "$ok" != "true" ]; then
        echo "Unexpected result for $name (exit $rc); inspect $log" >&2
        exit 1
    fi

    counts=$(awk '/^[0-9,]+ states generated, [0-9,]+ distinct states found/ { print $1 " | " $4; exit }' "$log")
    printf "%s | %s | %s | %s\n" "$name" "$counts" "$elapsed_s" "$label"
}

echo "run | generated | distinct | wall seconds | result"

if [ "$tier" = "pr" ]; then
    # Launch PR configs in bounded parallel batches (longest liveness/witness runs first)
    # so 4-vCPU CI runners stay uncontended and finish in ~10s.
    max_jobs=${CACHE_SCHEDULER_JOBS:-4}
    active=0
    for cfg in \
        RestoreSplitBudget DecodeStalledByPrefill CacheSchedulerQueuedBypass \
        CacheScheduler CacheSchedulerMaxPos8 BoundaryWindowDropped \
        CancelInStep UnprunedWindow QueuedBypassedByStep DeferredTied429 EagerHotWriteback
    do
        exec_tlc "$cfg" CacheScheduler 1 1g &
        active=$((active + 1))
        if [ "$active" -ge "$max_jobs" ]; then
            wait
            active=0
        fi
    done
    wait

    verify_run CacheScheduler pass
    verify_run CacheSchedulerMaxPos8 pass
    verify_run CacheSchedulerQueuedBypass pass
    verify_run CancelInStep invariant StepBlocksInMemoryAndOwned
    verify_run RestoreSplitBudget temporal 'a :> "Restoring" @@ b :> "Restoring"'
    verify_run BoundaryWindowDropped invariant LaneHistoryCovered
    verify_run UnprunedWindow invariant BudgetsAndReservations
    verify_run DecodeStalledByPrefill action ActionDecodePriority
    verify_run QueuedBypassedByStep invariant BypassBoundHolds
    verify_run DeferredTied429 action ActionNoQueuedRefusal
    verify_run EagerHotWriteback action ActionLazyColdWriteback
    exit 0
fi

run_seq() {
    name=$1
    kind=$2
    expected=${3:-}
    exec_tlc "$name" CacheScheduler "$workers" 8g
    verify_run "$name" "$kind" "$expected"
}

if [ "$tier" != "nightly-only" ]; then
    run_seq CacheScheduler pass
    run_seq CacheSchedulerMaxPos8 pass
    run_seq CacheSchedulerQueuedBypass pass
fi

run_seq CacheSchedulerQueue pass
run_seq CacheSchedulerTwoPrefixes pass
run_seq CacheSchedulerDecodePriority pass

if [ "$tier" != "nightly-only" ]; then
    run_seq CancelInStep invariant StepBlocksInMemoryAndOwned
    run_seq RestoreSplitBudget temporal 'a :> "Restoring" @@ b :> "Restoring"'
    run_seq BoundaryWindowDropped invariant LaneHistoryCovered
    run_seq UnprunedWindow invariant BudgetsAndReservations
    run_seq DecodeStalledByPrefill action ActionDecodePriority
    run_seq QueuedBypassedByStep invariant BypassBoundHolds
    run_seq DeferredTied429 action ActionNoQueuedRefusal
    run_seq EagerHotWriteback action ActionLazyColdWriteback
fi
