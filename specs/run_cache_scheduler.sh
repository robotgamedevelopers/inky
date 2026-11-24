#!/bin/sh
# Run exhaustive CacheScheduler safety + liveness checks on a bench;
# every positive run must exhaust with 0 errors and each reversed witness must fail for its own property.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ -n "${TLC_JAR:-}" ]; then
    jar=$TLC_JAR
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

check_run() {
    name=$1
    module=$2
    kind=${3:-pass}
    expected=${4:-}
    log="$output/$name.log"
    t0=$(now_ms)

    if (cd "$output" && "$java_bin" -XX:+UseParallelGC -Xmx8g -cp "$jar" tlc2.TLC \
        -noGenerateSpecTE -workers "$workers" -metadir "$output/$name-states" \
        -config "$root/$name.cfg" "$root/$module.tla") >"$log" 2>&1; then
        rc=0
    else
        rc=$?
    fi

    t1=$(now_ms)
    elapsed_ms=$(( t1 - t0 ))
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
check_run CacheScheduler CacheScheduler pass
check_run CacheSchedulerQueue CacheScheduler pass
check_run CacheSchedulerTwoPrefixes CacheScheduler pass
check_run CacheSchedulerMaxPos8 CacheScheduler pass
check_run CacheSchedulerDecodePriority CacheScheduler pass
check_run CacheSchedulerQueuedBypass CacheScheduler pass
check_run CancelInStep CacheScheduler invariant StepBlocksInMemoryAndOwned
check_run RestoreSplitBudget CacheScheduler temporal 'a :> "Restoring" @@ b :> "Restoring"'
check_run BoundaryWindowDropped CacheScheduler invariant LaneHistoryCovered
check_run UnprunedWindow CacheScheduler invariant BudgetsAndReservations
check_run DecodeStalledByPrefill CacheScheduler action ActionDecodePriority
check_run QueuedBypassedByStep CacheScheduler invariant BypassBoundHolds
check_run DeferredTied429 CacheScheduler action ActionNoQueuedRefusal
check_run EagerHotWriteback CacheScheduler action ActionLazyColdWriteback
