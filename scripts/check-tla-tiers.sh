#!/usr/bin/env bash
# Every TLA+ check and trace mutant runs in exactly one CI tier: per pull request (.github/workflows/ci.yml) or
# nightly (.github/workflows/nightly.yml). Nothing is copied here: the checks come from the runners' own lists
# (scripts/check-tla.sh --list, which asks specs/run_tlc.zsh, specs/run_cache_scheduler.sh and
# scripts/check-memmode-traces.sh), and the tiers from the parts each workflow passes to scripts/check-tla.sh.
# It fails, naming each offender, when
#   a check or mutant runs in no tier, or in more than one tier or part;
#   a model config (specs/*.cfg, specs/memmode/*.cfg) is run by no runner and named by no trace script, which is
#   what deleting a check line looks like;
#   a scripts/check-*-traces.sh is not wired into a tier.
# Usage: scripts/check-tla-tiers.sh [ROOT]; scripts/check-tla-tiers.sh --self-test plants each failure in a scratch
# copy and requires the red line for it. CPU only, no Java, no network.
set -euo pipefail

check() {
    local root=$1 out
    local wf=$root/.github/workflows tla=$root/scripts/check-tla.sh
    out=$(mktemp -d "${TMPDIR:-/tmp}/inky-tiers.XXXXXX")
    : > "$out/placed"
    for tier in ci nightly; do
        [ -f "$wf/$tier.yml" ] || { echo "tier check: $wf/$tier.yml missing" >&2; return 1; }
        ruby "$root/scripts/tla-workflow-parts.rb" "$wf/$tier.yml" > "$out/parts" || return 1
        while IFS= read -r p; do
            sh "$tla" --list "$p" | sed "s|^|$tier $p |" >> "$out/placed" || { echo "tier check: check-tla.sh --list $p failed" >&2; return 1; }
        done < "$out/parts"
    done
    sh "$tla" --list all > "$out/known"
    # A config no runner lists and no trace script names has lost its check.
    local lost=0 cfg name
    for cfg in "$root"/specs/*.cfg "$root"/specs/memmode/*.cfg; do
        [ -f "$cfg" ] || continue
        name=$(basename "$cfg" .cfg)
        if grep -Eq "^(layout|scheduler|memmode):$name\$" "$out/known"; then continue; fi
        if grep -lFq "$name.cfg" "$root"/scripts/check-*-traces.sh 2>/dev/null; then continue; fi
        echo "TIER LOST: ${cfg#$root/} is run by no runner and named by no trace script"
        lost=1
    done
    # A mutation scripts/mutate-memmode.py defines that the trace script does not run is lost the same way.
    local m
    for m in $(grep -oE '^ *"[a-z0-9-]+": \(' "$root/scripts/mutate-memmode.py" | tr -d ' "():'); do
        grep -Fxq "memmode-traces:$m" "$out/known" && continue
        echo "TIER LOST: mutation $m in scripts/mutate-memmode.py is run by no trace check"
        lost=1
    done
    # Every trace script on disk is a check of its own.
    local f
    for f in "$root"/scripts/check-*-traces.sh; do
        [ -f "$f" ] || continue
        case ${f##*/} in check-memmode-traces.sh) continue ;; esac
        echo "trace-script:${f##*/}" >> "$out/known"
    done
    sort -u "$out/known" -o "$out/known"
    local bad=0 item where n
    while read -r item; do
        n=$(awk -v i="$item" '$3 == i' "$out/placed" | wc -l | tr -d ' ')
        where=$(awk -v i="$item" '$3 == i { printf "%s%s (%s)", sep, $1, $2; sep=", " }' "$out/placed")
        if [ "$n" = 0 ]; then
            echo "TIER GAP: $item runs in no CI tier"; bad=1
        elif [ "$n" != 1 ]; then
            echo "TIER DUP: $item runs $n times: $where"; bad=1
        fi
    done < "$out/known"
    local total pr nightly
    total=$(wc -l < "$out/known" | tr -d ' ')
    pr=$(awk '$1 == "ci"' "$out/placed" | wc -l | tr -d ' ')
    nightly=$(awk '$1 == "nightly"' "$out/placed" | wc -l | tr -d ' ')
    rm -rf "$out"
    if [ "$bad" = 1 ] || [ "$lost" = 1 ]; then
        echo "tier check: FAILED" >&2
        return 1
    fi
    [ "$total" -gt 0 ] || { echo "tier check: no checks found" >&2; return 1; }
    echo "tier check: $total checks and mutants, each in exactly one tier ($pr per pull request, $nightly nightly)"
}

# plant NAME WANT EDIT...: copy the tree, apply the edit, and require a failing check that prints WANT.
plant() {
    local name=$1 want=$2 src=$3 copy log
    shift 3
    copy=$(mktemp -d "${TMPDIR:-/tmp}/inky-tiers-plant.XXXXXX")
    mkdir -p "$copy/.github"
    cp -R "$src/specs" "$src/scripts" "$copy/"
    cp -R "$src/.github/workflows" "$copy/.github/"
    (cd "$copy" && "$@")
    log=$copy/log
    if check "$copy" > "$log" 2>&1; then
        cat "$log"; echo "self-test: planted $name passed the tier check" >&2; rm -rf "$copy"; return 1
    fi
    if ! grep -Fxq "$want" "$log"; then
        cat "$log"; echo "self-test: planted $name failed without: $want" >&2; rm -rf "$copy"; return 1
    fi
    echo "self-test: $name -> $want"
    rm -rf "$copy"
}
drop_line() { local f=$1 pat=$2; grep -v -- "$pat" "$f" > "$f.new"; cmp -s "$f" "$f.new" && { echo "plant matched nothing: $pat" >&2; return 1; }; mv "$f.new" "$f"; }
add_line() { printf '%s\n' "$2" >> "$1"; }
sub_line() { local f=$1; sed -E "$2" "$f" > "$f.new"; cmp -s "$f" "$f.new" && { echo "plant matched nothing: $2" >&2; return 1; }; mv "$f.new" "$f"; }

if [ "${1:-}" = --self-test ]; then
    root=$(cd "$(dirname "$0")/.." && pwd)
    check "$root" > /dev/null || { echo "self-test: the unplanted tree fails the tier check" >&2; exit 1; }
    plant 'a dropped check line' 'TIER LOST: specs/PoisonGate4.cfg is run by no runner and named by no trace script' "$root" \
        drop_line specs/run_tlc.zsh '^check PoisonGate4 '
    plant 'a dropped memmode check' 'TIER LOST: specs/memmode/StaleSaved.cfg is run by no runner and named by no trace script' "$root" \
        drop_line scripts/check-tla.sh '^StaleSaved RestoredWhenIdle$'
    plant 'a dropped mutant' 'TIER LOST: mutation skip-keep in scripts/mutate-memmode.py is run by no trace check' "$root" \
        sub_line scripts/check-memmode-traces.sh 's/ skip-keep\)/)/'
    plant 'a declared matrix with no runner' 'TIER GAP: memmode:MemMode runs in no CI tier' "$root" \
        sub_line .github/workflows/ci.yml 's|^        run: TLA_JAR_CACHE=.*check-tla.sh.*$|        run: echo skipped|'
    plant 'a disabled TLA job' 'TLA tier coverage requires an unconditional, required runner' "$root" \
        sub_line .github/workflows/ci.yml 's/^  tla:$/  tla:\
    if: false/'
    plant 'a part no workflow runs' 'TIER GAP: memmode:MemMode runs in no CI tier' "$root" \
        sub_line .github/workflows/ci.yml 's/part: \[memmode, /part: [/'
    plant 'a part in both tiers' 'TIER DUP: memmode:MemMode runs 2 times: ci (memmode), nightly (memmode)' "$root" \
        add_line .github/workflows/nightly.yml '      - run: scripts/check-tla.sh memmode'
    plant 'an unwired trace script' 'TIER GAP: trace-script:check-planted-traces.sh runs in no CI tier' "$root" \
        add_line scripts/check-planted-traces.sh 'exit 0'
    plant 'the traces part dropped' 'TIER GAP: memmode-traces:skip-keep runs in no CI tier' "$root" \
        sub_line .github/workflows/ci.yml 's/, traces\]/]/'
    echo "self-test: every planted tier fault was caught"
    exit 0
fi
check "${1:-$(cd "$(dirname "$0")/.." && pwd)}"
