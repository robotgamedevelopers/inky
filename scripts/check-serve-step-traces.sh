#!/usr/bin/env bash
# Model-check decisions recorded at the serving scheduler's production planning entry point.
# With --replay, check a v2 JSONL capture from the GPU-backed scheduler test.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
: "${TLA2TOOLS_JAR:?set TLA2TOOLS_JAR to the pinned tla2tools.jar}"
jar=$(cd "$(dirname "$TLA2TOOLS_JAR")" && pwd)/$(basename "$TLA2TOOLS_JAR")
work=$(mktemp -d "${TMPDIR:-/tmp}/inky-serve-step.XXXXXX")
trap 'rm -rf "$work"' EXIT

cp "$root"/specs/ServeStepTrace.{tla,cfg} "$work/"
cat > "$work/ServeStepTraceData.tla" <<'TLA'
---- MODULE ServeStepTraceData ----
LOCAL INSTANCE Json
Frames == ndJsonDeserialize("frames.jsonl")
====
TLA

record_count() { awk 'END {print NR}' "$1"; }

# Expect TLC to explore one distinct state per record and exhaust the queue.
check_trace() {
    local name=$1 expected=$2 witness=${3:-} status=0
    local log="$work/tlc.log"
    (cd "$work" && "${TLC_JAVA:-java}" -Xmx1g -XX:+UseParallelGC -cp "$jar" tlc2.TLC \
        -noGenerateSpecTE -workers 1 -metadir "$work/$name-states" ServeStepTrace) > "$log" 2>&1 || status=$?
    if [[ -n "$witness" ]]; then
        if [[ "$status" = 12 ]] &&
            grep -Eq '^Error: Invariant AgreesWithPlannerPolicy is violated(\.| by the initial state:)$' "$log" &&
            grep -Fxq "frame = $witness" "$log"; then
            echo "$name: TLC rejected the mutated planner at decision $witness"
            return
        fi
    elif [[ "$status" = 0 ]] && grep -Fq 'Model checking completed. No error has been found.' "$log" &&
        tr -d ',' < "$log" | grep -E "^[0-9]+ states generated $expected distinct states found 0 states left on queue\.$" >/dev/null; then
        echo "$name: TLC checked all $expected planner decisions"
        return
    fi
    cat "$log" >&2
    echo "$name: unexpected TLC result (exit $status, $expected input frames)" >&2
    local frame
    frame=$(awk '/^frame = [0-9]+$/ {n=$3} END {print n}' "$log")
    if [[ -n "$frame" ]]; then
        sed -n "${frame}p" "$work/frames.jsonl" >&2
    fi
    return 1
}

if [[ $# != 0 ]]; then
    [[ $# = 2 && "$1" = --replay ]] || { echo "usage: $0 [--replay TRACE.jsonl]" >&2; exit 2; }
    cp -- "$2" "$work/frames.jsonl"
    count=$(record_count "$work/frames.jsonl")
    [[ "$count" -gt 0 ]] || { echo "empty trace: $2" >&2; exit 1; }
    check_trace replay "$count"
    exit 0
fi

# Mirror the production module hierarchy so --test enables the very same recorder.
mkdir "$work/serve"
cp "$root"/src/serve/{step,scheduler_trace}.rs "$work/serve/"
cat > "$work/driver.rs" <<'RUST'
mod scheduler;
mod serve {
    mod step;
    mod scheduler_trace;
}
RUST

compile() { rustc --edition 2021 --test -A dead_code "$work/driver.rs" -o "$work/driver"; }
dump_seed() {
    local seed=$1
    INKY_SCHED_TRACE_OUT="$work/seed.jsonl" INKY_SCHED_TRACE_SEED="$seed" \
        "$work/driver" --exact serve::scheduler_trace::dump --ignored > "$work/driver.log" || {
            cat "$work/driver.log" >&2; return 1;
        }
    [[ $(record_count "$work/seed.jsonl") = 64 ]] || {
        echo "seed $seed: expected 64 decisions" >&2; return 1;
    }
    cat "$work/seed.jsonl" >> "$work/frames.jsonl"
    rm "$work/seed.jsonl"
}

cp "$root/src/scheduler.rs" "$work/scheduler.rs"
compile
"$work/driver" --exact serve::scheduler_trace::seeded_decisions_are_repeatable | tee "$work/driver.log"
grep -Fq "1 passed; 0 failed" "$work/driver.log"
for seed in {0..31} 18446744073709551615; do dump_seed "$seed"; done
check_trace production 2112

# Change only the scheduler source in the temporary build; keep the input generator,
# capture, and TLC specification identical to the green run.
check_mutant() {
    local name=$1 before=$2 after=$3 witness=$4
    python3 - "$root/src/scheduler.rs" "$work/scheduler.rs" "$before" "$after" <<'PY'
from pathlib import Path
import sys

src, dst, before, after = sys.argv[1:]
source = Path(src).read_text()
if source.count(before) != 1:
    sys.exit(f"mutation target must occur exactly once in {src}: {before}")
Path(dst).write_text(source.replace(before, after))
PY
    compile
    : > "$work/frames.jsonl"
    dump_seed 0
    check_trace "$name" 64 "$witness"
}
check_mutant never-preempt '(Some(dw), Some(pw)) => !(pw >= self.max_bypass && dw < pw)' '(Some(_), Some(_)) => true' 17
check_mutant frozen-age '(self.wait_ticks[i] + 1).min(self.max_bypass)' 'self.wait_ticks[i]' 1
