#!/bin/sh
# Check every available inky TLA+ model and its reversed witnesses.
# Usage: scripts/check-tla.sh [all|memmode|layout|blocks|grouped|layout-nightly|scheduler|traces].
# all (the default) runs every PR part, plus layout-nightly and scheduler when TLA_NIGHTLY=1.
# layout is blocks + grouped: grouped is the GroupedLayout, PoisonGate and ScratchTail checks of
# specs/run_tlc.zsh, blocks is every other PR check there, so a new model lands in blocks.
# layout-nightly runs the slow layout configs (PrefixTree, StartTails3) and PrefixTree trace mutants
# (requires TLA_NIGHTLY=1). The pull-request workflow runs memmode, blocks, grouped and traces as
# parallel jobs; layout-nightly and the CacheScheduler models (scheduler, which also requires
# TLA_NIGHTLY=1) are exhaustive and slow, so they run nightly.
# traces runs every trace-conformance script named in trace_scripts below, in parallel.
#
# scripts/check-tla.sh --list PART prints, one per line, every check and mutant PART runs, as RUNNER:NAME, and runs
# nothing. scripts/check-tla-tiers.sh reads it to prove each check sits in exactly one CI tier.
#
# The pinned tla2tools.jar comes from TLA2TOOLS_JAR when set; otherwise from TLA_JAR_CACHE (a directory, kept across
# CI runs by actions/cache) when set, downloading it there once; otherwise it is downloaded for this run. Its SHA-256
# is checked every time.
set -eu

parts='memmode blocks grouped layout-nightly scheduler traces'
# The trace-conformance scripts, scripts/check-<name>-traces.sh: memmode, blockbook, cache and scheduler
# all live on main and run in parallel. A check-*-traces.sh that is not named here fails the traces part
# and the tier check.
trace_scripts='memmode blockbook cache scheduler'
grouped_modules=GroupedLayout,PoisonGate,ScratchTail

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)

listing=
if [ "${1:-}" = --list ]; then
    listing=1
    shift
fi
section=${1:-all}
case $section in
    all|memmode|layout|blocks|grouped|layout-nightly|scheduler|traces) ;;
    *) echo "usage: scripts/check-tla.sh [--list] [all|memmode|layout|blocks|grouped|layout-nightly|scheduler|traces]" >&2; exit 2 ;;
esac
want() { [ "$section" = all ] || [ "$section" = "$1" ] || { [ "$section" = layout ] && { [ "$1" = blocks ] || [ "$1" = grouped ]; }; }; }

# The memmode model checks: config name, then the invariant its witness must violate (none for a passing run).
memmode_checks='MemMode
RestoreDefault RestoredWhenIdle
ReleaseEarly RaisedWhileConnected
NoPersist RestoredWhenIdle
LostSeed SavedIsOwner
StaleSaved RestoredWhenIdle
NoRollback RefusedRollbackSafe
RestoreStale RestoredWhenIdle
StaleBoot RestoredWhenIdle'

layout_sel() {
    case $1 in
        blocks) echo "TLC_NIGHTLY= INKY_TLC_NIGHTLY= TLC_NIGHTLY_ONLY= TLC_EXCEPT=$grouped_modules" ;;
        grouped) echo "TLC_NIGHTLY= INKY_TLC_NIGHTLY= TLC_NIGHTLY_ONLY= TLC_ONLY=$grouped_modules" ;;
        layout-nightly) echo "TLC_NIGHTLY_ONLY=1 TLC_ONLY= TLC_EXCEPT=" ;;
    esac
}

if [ -n "$listing" ]; then
    for p in $parts; do
        want "$p" || continue
        case $p in
            memmode) echo "$memmode_checks" | while read -r name _; do echo "memmode:$name"; done ;;
            blocks|grouped|layout-nightly)
                [ -f "$root/specs/BlockLayout.tla" ] || continue
                env $(layout_sel "$p") zsh "$root/specs/run_tlc.zsh" --list | sed 's/^/layout:/' ;;
            scheduler)
                [ -f "$root/specs/CacheScheduler.tla" ] || continue
                sh "$root/specs/run_cache_scheduler.sh" --list | sed 's/^/scheduler:/' ;;
            traces)
                bash "$root/scripts/check-memmode-traces.sh" --list | sed 's/^/memmode-traces:/'
                for t in $trace_scripts; do
                    [ "$t" = memmode ] || [ ! -f "$root/scripts/check-$t-traces.sh" ] || echo "trace-script:check-$t-traces.sh"
                done ;;
        esac
    done
    exit 0
fi

# A part named on its own must have something to check; an empty pass is a failure.
case $section in
    layout|blocks|grouped) [ -f "$root/specs/BlockLayout.tla" ] || { echo 'layout: specs/BlockLayout.tla missing' >&2; exit 1; } ;;
    layout-nightly)
        [ -f "$root/specs/BlockLayout.tla" ] || { echo 'layout-nightly: specs/BlockLayout.tla missing' >&2; exit 1; }
        [ "${TLA_NIGHTLY:-0}" = 1 ] || { echo 'layout-nightly: refused without TLA_NIGHTLY=1 (the nightly workflow sets it)' >&2; exit 1; }
        ;;
    scheduler)
        [ -f "$root/specs/CacheScheduler.tla" ] || { echo 'scheduler: specs/CacheScheduler.tla missing' >&2; exit 1; }
        # The scheduler part exists to run the CacheScheduler models; without TLA_NIGHTLY=1 it would check nothing.
        [ "${TLA_NIGHTLY:-0}" = 1 ] || { echo 'scheduler: refused without TLA_NIGHTLY=1 (the nightly workflow sets it)' >&2; exit 1; }
        ;;
esac

jar_url=https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar
jar_sha256=fdffef247f563ef6edbfd44958d2a82db79e7564e07cb673262d896353fa6713
work=$(mktemp -d "${TMPDIR:-/tmp}/inky-tla.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

sha256() {
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'; else sha256sum "$1" | awk '{print $1}'; fi
}
if [ -n "${TLA2TOOLS_JAR:-}" ]; then
    jar=$TLA2TOOLS_JAR
elif [ -n "${TLA_JAR_CACHE:-}" ]; then
    mkdir -p "$TLA_JAR_CACHE"
    jar=$TLA_JAR_CACHE/tla2tools-$jar_sha256.jar
    if [ -f "$jar" ] && [ "$(sha256 "$jar")" = "$jar_sha256" ]; then
        echo "tla2tools.jar: cached"
    else
        curl --fail --location --silent --show-error --retry 3 --output "$jar.part" "$jar_url"
        mv "$jar.part" "$jar"
        echo "tla2tools.jar: downloaded into the cache"
    fi
else
    jar=$work/tla2tools.jar
    curl --fail --location --silent --show-error --retry 3 --output "$jar" "$jar_url"
fi

actual_sha256=$(sha256 "$jar")
if [ "$actual_sha256" != "$jar_sha256" ]; then
    echo "tla2tools.jar SHA-256 mismatch: $actual_sha256" >&2
    exit 1
fi

jar=$(CDPATH='' cd -- "$(dirname -- "$jar")" && pwd)/$(basename -- "$jar")
java_bin=${TLC_JAVA:-java}
export TLA2TOOLS_JAR="$jar" TLC_JAR="$jar" TLC_JAVA="$java_bin"
export TLC_WORKERS="${TLC_WORKERS:-2}"

memmode=$root/specs/memmode
# check_memmode NAME [INVARIANT]: one TLC run of specs/memmode/NAME.cfg; its line goes to $work/memmode-NAME.result.
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
        echo "MemMode $name: PASS" > "$work/memmode-$name.result"
    else
        if [ "$rc" -eq 0 ] || ! grep -Fxq "Error: Invariant $expected is violated." "$log" || \
            ! grep -Fxq 'Error: The behavior up to this point is:' "$log" || \
            ! grep -q '^State 2:' "$log" || \
            ! grep -Eq '^[0-9,]+ states generated, [0-9,]+ distinct states found' "$log"; then
            cat "$log" >&2
            echo "MemMode $name did not show the $expected witness" >&2
            exit 1
        fi
        echo "MemMode $name: expected $expected violation" > "$work/memmode-$name.result"
    fi
}

if want memmode; then
    # The seven runs are independent and small: all at once, then the results in order.
    pids=
    while read -r name expected; do
        check_memmode "$name" "$expected" &
        pids="$pids $!"
    done <<EOF
$memmode_checks
EOF
    failed=0
    for pid in $pids; do wait "$pid" || failed=1; done
    echo "$memmode_checks" | while read -r name _; do cat "$work/memmode-$name.result" 2>/dev/null || echo "MemMode $name: FAILED"; done
    [ "$failed" = 0 ] || exit 1
fi

# The layer 3 models land separately. A model without its runner is an error.
for p in blocks grouped; do
    want "$p" && [ -f "$root/specs/BlockLayout.tla" ] || continue
    [ -f "$root/specs/run_tlc.zsh" ] || { echo 'BlockLayout runner missing' >&2; exit 1; }
    env $(layout_sel "$p") TLC_OUTPUT="$work/block-layout-$p" zsh "$root/specs/run_tlc.zsh" || {
        find "$work/block-layout-$p" -name '*.log' -exec tail -n 60 {} \; >&2
        exit 1
    }
done

if want layout-nightly && [ -f "$root/specs/BlockLayout.tla" ]; then
    [ -f "$root/specs/run_tlc.zsh" ] || { echo 'BlockLayout runner missing' >&2; exit 1; }
    case "${TLA_NIGHTLY:-0}" in
        1)
            env $(layout_sel layout-nightly) TLC_OUTPUT="$work/block-layout-nightly" zsh "$root/specs/run_tlc.zsh" || {
                find "$work/block-layout-nightly" -name '*.log' -exec tail -n 60 {} \; >&2
                exit 1
            }
            ;;
        0)
            echo 'layout-nightly (PrefixTree, StartTails3): not run here; nightly only (.github/workflows/nightly.yml)'
            ;;
        *)
            echo "TLA_NIGHTLY must be 0 or 1, not $TLA_NIGHTLY" >&2
            exit 1
            ;;
    esac
fi

# The CacheScheduler runner takes more than two minutes (its DecodePriority run
# alone took 159 s on hetzner1, and the whole cpu TLA step 4 min 58 s on the
# hosted runner, 2026-11-24). Pull request CI therefore leaves it to the nightly
# workflow, which sets TLA_NIGHTLY=1. The runner must still exist. Asked for by
# name (the scheduler part) without TLA_NIGHTLY=1, this script refuses above.
if want scheduler && [ -f "$root/specs/CacheScheduler.tla" ]; then
    [ -f "$root/specs/run_cache_scheduler.sh" ] || { echo 'CacheScheduler runner missing' >&2; exit 1; }
    case "${TLA_NIGHTLY:-0}" in
        1)
            TLC_OUTPUT=$work/cache-scheduler sh "$root/specs/run_cache_scheduler.sh" || {
                find "$work/cache-scheduler" -name '*.log' -exec tail -n 60 {} \; >&2
                exit 1
            }
            ;;
        0)
            echo 'CacheScheduler: not run here; nightly only (.github/workflows/nightly.yml)'
            ;;
        *)
            echo "TLA_NIGHTLY must be 0 or 1, not $TLA_NIGHTLY" >&2
            exit 1
            ;;
    esac
fi

# Required since PR #23: a missing checker must fail this gate. Every trace script runs, all at once; a
# check-*-traces.sh that trace_scripts does not name fails here rather than going unrun.
if want traces; then
    [ -f "$root/scripts/check-memmode-traces.sh" ] || { echo 'memmode trace checker missing' >&2; exit 1; }
    for f in "$root"/scripts/check-*-traces.sh; do
        t=${f##*/check-}; t=${t%-traces.sh}
        case " $trace_scripts " in
            *" $t "*) ;;
            *) echo "traces: scripts/check-$t-traces.sh is not in trace_scripts in scripts/check-tla.sh; wire it" >&2; exit 1 ;;
        esac
    done
    pids= names=
    for t in $trace_scripts; do
        if [ ! -f "$root/scripts/check-$t-traces.sh" ]; then
            echo "traces: scripts/check-$t-traces.sh missing" >&2
            exit 1
        fi
        bash "$root/scripts/check-$t-traces.sh" >"$work/traces-$t.log" 2>&1 &
        pids="$pids $!" names="$names $t"
    done
    failed=0
    set -- $names
    for pid in $pids; do
        if wait "$pid"; then
            echo "== check-$1-traces.sh: PASS"
        else
            echo "== check-$1-traces.sh: FAILED" >&2
            failed=1
        fi
        cat "$work/traces-$1.log"
        shift
    done
    [ "$failed" = 0 ] || exit 1
fi
