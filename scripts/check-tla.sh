#!/bin/sh
# Check every available inky TLA+ model and its reversed witnesses.
# Usage: scripts/check-tla.sh [all|memmode|layout|scheduler|traces]. all (the default; the nightly workflow) runs every
# part. The pull-request workflow runs memmode, layout and traces as parallel jobs; the CacheScheduler models
# (scheduler) are exhaustive and slow, so they run in the nightly workflow.
set -eu

section=${1:-all}
case $section in
    all|memmode|layout|scheduler|traces) ;;
    *) echo "usage: scripts/check-tla.sh [all|memmode|layout|scheduler|traces]" >&2; exit 2 ;;
esac
want() { [ "$section" = all ] || [ "$section" = "$1" ]; }
# A part named on its own must have something to check; an empty pass is a failure.
case $section in
    layout) [ -f "$(dirname -- "$0")/../specs/BlockLayout.tla" ] || { echo 'layout: specs/BlockLayout.tla missing' >&2; exit 1; } ;;
    scheduler) [ -f "$(dirname -- "$0")/../specs/CacheScheduler.tla" ] || { echo 'scheduler: specs/CacheScheduler.tla missing' >&2; exit 1; } ;;
esac

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
jar_url=https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar
jar_sha256=fdffef247f563ef6edbfd44958d2a82db79e7564e07cb673262d896353fa6713
work=$(mktemp -d "${TMPDIR:-/tmp}/inky-tla.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

if [ -n "${TLA2TOOLS_JAR:-}" ]; then
    jar=$TLA2TOOLS_JAR
else
    jar=$work/tla2tools.jar
    curl --fail --location --silent --show-error --retry 3 --output "$jar" "$jar_url"
fi

if command -v shasum >/dev/null 2>&1; then
    actual_sha256=$(shasum -a 256 "$jar" | awk '{print $1}')
else
    actual_sha256=$(sha256sum "$jar" | awk '{print $1}')
fi
if [ "$actual_sha256" != "$jar_sha256" ]; then
    echo "tla2tools.jar SHA-256 mismatch: $actual_sha256" >&2
    exit 1
fi

jar=$(CDPATH='' cd -- "$(dirname -- "$jar")" && pwd)/$(basename -- "$jar")
java_bin=${TLC_JAVA:-java}
export TLA2TOOLS_JAR="$jar" TLC_JAR="$jar" TLC_JAVA="$java_bin"
export TLC_WORKERS="${TLC_WORKERS:-2}"

memmode=$root/specs/memmode
check_memmode() {
    name=$1
    expected=${2:-}
    log=$work/memmode-$name.log
    if (cd "$work" && "$java_bin" -Xmx2g -cp "$jar" tlc2.TLC \
        -noGenerateSpecTE -workers 1 -metadir "$work/memmode-$name-states" \
        -config "$memmode/$name.cfg" "$memmode/MemMode.tla") >"$log" 2>&1; then
        rc=0
    else
        rc=$?
    fi

    if [ -z "$expected" ]; then
        if [ "$rc" -ne 0 ] || ! grep -Fq 'Model checking completed. No error has been found.' "$log" || \
            ! grep -Fq '0 states left on queue.' "$log" || grep -q '^Error:' "$log"; then
            cat "$log" >&2
            echo "MemMode $name did not complete cleanly" >&2
            exit 1
        fi
        echo "MemMode $name: PASS"
    else
        if [ "$rc" -eq 0 ] || ! grep -Fxq "Error: Invariant $expected is violated." "$log" || \
            ! grep -Fxq 'Error: The behavior up to this point is:' "$log" || \
            ! grep -q '^State 2:' "$log" || \
            ! grep -Eq '^[0-9,]+ states generated, [0-9,]+ distinct states found' "$log"; then
            cat "$log" >&2
            echo "MemMode $name did not show the $expected witness" >&2
            exit 1
        fi
        echo "MemMode $name: expected $expected violation"
    fi
}

if want memmode; then
check_memmode MemMode
check_memmode RestoreDefault RestoredWhenIdle
check_memmode ReleaseEarly RaisedWhileConnected
check_memmode NoPersist RestoredWhenIdle
check_memmode LostSeed SavedIsOwner
check_memmode StaleSaved RestoredWhenIdle
check_memmode NoRollback RefusedRollbackSafe
fi

# The layer 3 models land separately. A model without its runner is an error.
if want layout && [ -f "$root/specs/BlockLayout.tla" ]; then
    [ -f "$root/specs/run_tlc.zsh" ] || { echo 'BlockLayout runner missing' >&2; exit 1; }
    TLC_OUTPUT=$work/block-layout zsh "$root/specs/run_tlc.zsh" || {
        find "$work/block-layout" -name '*.log' -exec tail -n 60 {} \; >&2
        exit 1
    }
fi
if want scheduler && [ -f "$root/specs/CacheScheduler.tla" ]; then
    [ -f "$root/specs/run_cache_scheduler.sh" ] || { echo 'CacheScheduler runner missing' >&2; exit 1; }
    TLC_OUTPUT=$work/cache-scheduler sh "$root/specs/run_cache_scheduler.sh" || {
        find "$work/cache-scheduler" -name '*.log' -exec tail -n 60 {} \; >&2
        exit 1
    }
fi

# Required since PR #23: a missing checker must fail this gate.
if want traces; then
    bash "$root/scripts/check-memmode-traces.sh"
fi
