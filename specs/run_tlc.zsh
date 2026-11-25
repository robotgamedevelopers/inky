#!/bin/zsh
# Run on the designated bench; negative controls must fail for the right reason.
# Compatible with both zsh (macOS) and bash 5+ (Linux benches).
# The `check` lines below queue their runs; the queue then runs TLC_JOBS at a time (default: one per available core), each a
# single-worker TLC in its own process, and the table prints in file order once every run has finished.
# `run_tlc.zsh --list` prints the names of the selected checks and runs nothing: scripts/check-tla-tiers.sh reads it.
set -eu
if [ -n "${ZSH_VERSION:-}" ]; then
    zmodload zsh/datetime
    runner_sh=zsh
else
    runner_sh=bash
fi
root=$(cd "$(dirname "$0")" && pwd)
self="$root/$(basename "$0")"
jar=${TLC_JAR:-$HOME/sdk/tla/tla2tools.jar}
java=${TLC_JAVA:-$HOME/sdk/bin/java}
mode=run
case ${1:-} in
    '') ;;
    --list) mode=list ;;
    --one) mode=one ;;  # internal: one queued run, started by xargs below
    *) printf '%s\n' "usage: run_tlc.zsh [--list]" >&2; exit 2 ;;
esac
if [[ $mode != list ]]; then
    output=${TLC_OUTPUT:-$PWD/tlc-results}
    mkdir -p "$output"
    output=$(cd "$output" && pwd)
    [[ -r "$jar" ]] || { printf '%s\n' "Missing TLC jar: $jar" >&2; exit 1; }
fi

# TLC_ONLY=<modules> runs only the checks of those modules; TLC_EXCEPT=<modules> runs every other check (comma lists).
# TLC_NIGHTLY_ONLY=1 selects only the slow nightly checks (PrefixTree, StartTails3) and PrefixTree trace mutants.
# scripts/check-tla.sh splits its layout part this way into parallel CI jobs. A selection that runs nothing, or names
# a module with no check, fails.
only=${TLC_ONLY:-} except=${TLC_EXCEPT:-} nightly_only=${TLC_NIGHTLY_ONLY:-} ran=0
typeset -A seen
queue=()

# run_one NAME MODULE EXPECTED KIND: one TLC run; EXPECTED is - for a run that must pass. Its table row goes to
# $output/NAME.result, which exists only when the run gave exactly the expected result.
run_one() {
    local name=$1 module=$2 expected=$3 kind=$4 log="$output/$1.log"
    [[ $expected = - ]] && expected=
    local started=$EPOCHREALTIME elapsed rc
    if "$java" -XX:+UseParallelGC -Xmx4g -cp "$jar" tlc2.TLC -noGenerateSpecTE \
        -workers 1 -metadir "$output/$name-states" \
        -config "$root/$name.cfg" "$root/$module.tla" >"$log" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    elapsed=$(awk -v s="$started" -v e="$EPOCHREALTIME" 'BEGIN { printf "%.3f", e - s }')
    local ok=true
    # An initial-state violation stops before TLC prints a state count.
    if [[ $kind != initial ]]; then
        grep -Eq "^[0-9,]+ states generated, [0-9,]+ distinct states found" "$log" || ok=false
    fi
    grep -q "^Finished in " "$log" || ok=false
    if [[ -n "$expected" ]]; then
        [[ $rc -ne 0 ]] || ok=false
        # An initial-state violation is one Error line. A reached state adds
        # "The behavior up to this point is:" so the count is two.
        local want_errors=2
        if [[ $kind = initial ]]; then
            want_errors=1
        fi
        [[ $(grep -c "^Error: " "$log" || true) = $want_errors ]] || ok=false
        if [[ $kind = temporal ]]; then
            # Older TLC reports an unnamed temporal violation. The witness
            # config must name precisely the requested property in that case.
            if ! grep -Fxq "Error: Temporal property $expected was violated." "$log"; then
                grep -Fxq "Error: Temporal properties were violated." "$log" || ok=false
                [[ $(awk '/^PROPERTY / {print substr($0,10)}' "$root/$name.cfg") = "$expected" ]] || ok=false
            fi
            grep -Fxq "Error: The following behavior constitutes a counter-example:" "$log" || ok=false
            grep -Eq '^State [0-9]+: Stuttering$' "$log" || ok=false
        elif [[ $kind = initial ]]; then
            grep -Fxq "Error: Invariant $expected is violated by the initial state:" "$log" || ok=false
        else
            grep -Fxq "Error: Invariant $expected is violated." "$log" || ok=false
            grep -Fxq "Error: The behavior up to this point is:" "$log" || ok=false
        fi
        if [[ $kind != initial ]]; then
            grep -q "^State 2:" "$log" || ok=false
        fi
    else
        [[ $rc -eq 0 ]] || ok=false
        if grep -q "^Error:" "$log"; then ok=false; fi
        grep -Fxq "Model checking completed. No error has been found." "$log" || ok=false
    fi
    if [[ $ok != true ]]; then
        printf '%s\n' "Unexpected result for $name (exit $rc); inspect $log" >&2
        exit 1
    fi
    local counts
    if [[ $kind = initial ]]; then
        counts="- | -"
    else
        counts=$(awk "/^[0-9,]+ states generated, [0-9,]+ distinct states found/ { print \$1 \" | \" \$4 }" "$log")
    fi
    printf "%s | %s | %s | %s\n" "$name" "$counts" "$elapsed" "${expected:-PASS}" > "$output/$name.result"
}
if [[ $mode = one ]]; then
    IFS=: read -r qn qm qe qk <<EOF
$2
EOF
    run_one "$qn" "$qm" "$qe" "$qk"
    exit 0
fi

select_check() {
    local name=$1 module=$2 expected=${3:-} kind=${4:-invariant}
    if [[ -n $only && ",$only," != *",$module,"* ]] || [[ -n $except && ",$except," == *",$module,"* ]]; then
        return 0
    fi
    ran=$(( ran + 1 )); seen[$module]=1
    queue+=("$name:$module:${expected:--}:$kind")
}

check() {
    [[ -z $nightly_only ]] || return 0
    select_check "$@"
}

check_nightly() {
    if [[ -n $nightly_only || -n ${TLC_NIGHTLY:-} || -n ${INKY_TLC_NIGHTLY:-} ]]; then
        select_check "$@"
    fi
}

check PrefixTreeFast PrefixTree
check PrefixTreeEvictFast PrefixTree
check PrefixTreeShallow PrefixTree DeepestStart
check PrefixTreeStale PrefixTree ResultLive
check PrefixTreeFirstToken PrefixTree MatchExact
check PrefixTreeAnyParent PrefixTree ResultIsStart
check PrefixTreePartial PrefixTree StartOnBoundary
check PrefixTreeLeak PrefixTree RefsSettledAfterSeek
check PrefixTreeReuseHeld PrefixTree HeldNeverStale
check PrefixTreeStaleInsert PrefixTree NoBadInsert
check PrefixTreeTrace PrefixTreeTrace
check_nightly PrefixTree PrefixTree
check BlockLayout3 BlockLayout
check BlockLayout4 BlockLayout
check BlockLayoutHigh BlockLayout
check BlockLayoutKv8 BlockLayout
check StaleEntry BlockLayout GenerationMatches
check WindowOffByOne BlockLayout WindowCovered
check FullDropped BlockLayout FullCovered
check SpanTooSmall BlockLayout SpanPinned
check StartReleased BlockLayout StartRetained
check TailNotZeroed BlockLayout TailPrivateZero
check WrongTableIndex BlockLayout TableIndex
check Truncate32 BlockLayout Address64
check SlotOverlap BlockLayout Address64
check SlabWord SlabWord
check SlabWordKv8 SlabWord
check SlabWordUnmasked SlabWord ShaderPtrResolves
check SlabWordWrongBind SlabWord ShaderPtrResolves
check SlabWordOverflow SlabWord NoOffsetClobbersSlab
check SlabWordFifthSlab SlabWord SlabCountBounded
check BlockedSlab3 BlockedSlab
check BlockedSlab4 BlockedSlab
check BlockedSlabFlatGrow BlockedSlab SyncedMatchesCpu
check BlockedSlabNoReuseZero BlockedSlab CleanSlotZeroed
check BlockedSlabMissedDirty BlockedSlab SyncedMatchesCpu
check VecAlloc10 VecAlloc
check VecAlloc7 VecAlloc
check VecAlloc6 VecAlloc NoPoolExhaustion
check VecAllocUnpinnedWeak VecAlloc NeverOverwritesLive
check VerifyRollback VerifyRollback
check VerifyOffByOneCs VerifyRollback ConvMatchesSequential
check VerifyKeepRejectedKv VerifyRollback RejectedTailRestored
check VerifySealRejected VerifyRollback NoRejectedSeal
check KvDiskTier KvDiskTier
check KvDiskNoOverlap KvDiskTier SplitBoundaryAligned
check KvDiskStagingRace KvDiskTier StagingHalfSafe
check SwaRing SwaRing
check SwaRingShortWriteback SwaRing RingHoldsExactWindow
check SpanPins512 SpanPins
check SpanPins17 SpanPins Covered
check SpanPinsOverBound SpanPins BoundHolds
check SpanPinsMax24 SpanPins MaximumIs25
check GroupedLayout3 GroupedLayout
check GroupedLayout4 GroupedLayout
check LocalMatch GroupedLayout VisitedKeys
check DonorWindow GroupedLayout VisitedKeys
check TilePartial GroupedLayout OrderedPartials
check Restage GroupedLayout StagedOnce
check StalledTraversal GroupedLayout TraversalCompletes temporal
check AdvanceUnfair GroupedLayout TraversalCompletes temporal
# Negative control: the redundant-load mutant must keep both output invariants,
# so only StagedOnce can see it. Restage above proves the same mutant is live.
check RestageOutputs GroupedLayout
check StartTails2 StartTails
check LaneStaleGen StartTails LaneCovered
check UnheldWindow StartTails StartRetained
check PrunedUnderStart StartTails StartRetained
check TailOverread StartTails ReadsWritten
check ReuseNotZeroed StartTails TailPrivateZero
check DanglingHead StartTails HeadsConsistent
# Three slots per kind takes about a minute on one worker: the nightly workflow runs it via layout-nightly.
check_nightly StartTails3 StartTails
check PoisonGate3 PoisonGate
check PoisonGate4 PoisonGate
check PoisonUnguarded PoisonGate NoFalsePoison
check PoisonNoSave PoisonGate LiveIntact
check PoisonOwnedOnly PoisonGate Coverage
check PoisonNoZero PoisonGate NoFalsePoison
check ScratchTail3 ScratchTail
check ScratchTail4 ScratchTail
check ScratchTailOneSpan ScratchTail
check ScratchTailNoZero ScratchTail NoForeignRead
check InBlock InBlock
check InBlockKv8 InBlock
check InBlockMajor InBlock Ordered
check InBlockMajorCollide InBlock NoCollision
check InBlockAddrCol InBlock NoCollision
check InBlockTrace InBlockTrace
check InBlockTraceKv8 InBlockTrace

(( ran > 0 )) || { printf '%s\n' "run_tlc.zsh: no check selected (TLC_ONLY=$only TLC_EXCEPT=$except TLC_NIGHTLY_ONLY=$nightly_only)" >&2; exit 1; }
if [[ -n $only ]]; then
    while IFS= read -r m; do
        [[ -z $m ]] && continue
        [[ -n ${seen[$m]:-} ]] || { printf '%s\n' "run_tlc.zsh: TLC_ONLY names $m, which has no check" >&2; exit 1; }
    done <<EOF
$(printf '%s\n' "$only" | tr ',' '\n')
EOF
fi
if [[ $mode = list ]]; then
    for item in "${queue[@]}"; do
        printf '%s\n' "${item%%:*}"
    done
    exit 0
fi

# Mutants of the PrefixTree trace binding (Cargo + Metal-crate build on macOS):
# run in the nightly layout-nightly tier (no PR timeout), sharing one cargo target dir.
mutant() {
    local name=$1 from=$2 to=$3 expected=$4 log="$output/mutant-$1.log" rc
    local tmp=$(mktemp -d)
    mkdir "$tmp/repo"
    (cd "$root/.." && tar --exclude=target --exclude=.git --exclude=tlc-results -cf - .) \
        | (cd "$tmp/repo" && tar -xf -) || { printf '%s\n' "mutant $name: copy failed" >&2; exit 1; }
    MUT_FROM="$from" MUT_TO="$to" perl -0pi -e \
        's/\Q$ENV{MUT_FROM}\E/$ENV{MUT_TO}/ or die "plant not applied\n"' \
        "$tmp/repo/src/prefix.rs" \
        || { printf '%s\n' "mutant $name: plant not applied" >&2; rm -rf "$tmp"; exit 1; }
    (cd "$tmp/repo" && CARGO_TARGET_DIR="$output/mutant-target" \
        INKY_PREFIX_TRACE_OUT="$tmp/repo/specs/PrefixTreeTraceData.tla" \
        cargo test --locked prefix::tests::prefix_trace_dump -- --ignored --exact >/dev/null 2>&1) \
        || { printf '%s\n' "mutant $name: the plant broke the trace walk itself" >&2; rm -rf "$tmp"; exit 1; }
    if "$java" -XX:+UseParallelGC -Xmx4g -cp "$jar" tlc2.TLC -noGenerateSpecTE \
        -workers 1 -metadir "$output/mutant-$name-states" \
        -config "$tmp/repo/specs/PrefixTreeTrace.cfg" "$tmp/repo/specs/PrefixTreeTrace.tla" \
        >"$log" 2>&1; then rc=0; else rc=$?; fi
    rm -rf "$tmp"
    if [[ $rc -eq 0 ]] || ! grep -Fxq "Error: $expected." "$log"; then
        printf '%s\n' "mutant $name did not go red as $expected; inspect $log" >&2
        exit 1
    fi
    printf "mutant-%s | red as required (%s)\n" "$name" "$expected"
}

njobs=${TLC_JOBS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)}  # nproc honours a CPU reservation
[[ $njobs =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' "run_tlc.zsh: TLC_JOBS must be a positive integer, not $njobs" >&2; exit 1; }
for item in "${queue[@]}"; do
    rm -f "$output/${item%%:*}.result"
done
export TLC_JAR=$jar TLC_JAVA=$java TLC_OUTPUT=$output
started=$EPOCHREALTIME xrc=0
printf '%s\0' "${queue[@]}" | xargs -0 -n 1 -P "$njobs" "$runner_sh" "$self" --one || xrc=$?
printf '%s\n' "run | generated | distinct | wall seconds | result"
failed=()
for item in "${queue[@]}"; do
    qname=${item%%:*}
    if [[ -f $output/$qname.result ]]; then cat "$output/$qname.result"; else failed+=("$qname"); fi
done
wall_s=$(awk -v s="$started" -v e="$EPOCHREALTIME" 'BEGIN { printf "%.3f", e - s }')
printf "%d runs, %d at a time: %s s wall\n" "${#queue[@]}" "$njobs" "$wall_s"
if (( ${#failed[@]} || xrc )); then
    printf '%s\n' "run_tlc.zsh: failed: ${failed[*]:-none missing, but xargs exited $xrc}" >&2
    exit 1
fi

if [[ -n $nightly_only && $(uname -s) = Darwin ]]; then
    (cd "$root/.." && INKY_PREFIX_TRACE_OUT="$root/PrefixTreeTraceData.tla" \
        cargo test --locked prefix::tests::prefix_trace_dump -- --ignored --exact >/dev/null) \
        || { printf '%s\n' "prefix trace dump failed" >&2; exit 1; }
    mutant no-evict \
        '.filter(|(_, h)| h.entries().any(|e| self.records.generation(e.slot(), e.kind()) != Some(e.generation())))' \
        '.filter(|(_, _h)| false)' \
        'Deadlock reached'
    mutant full-match \
        '!tokens.is_empty() && m.at == tokens.len()' \
        'false && m.at == tokens.len()' \
        'Deadlock reached'
    mutant ref-leak \
        'self.records.drop_ref(e.slot(), e.kind(), e.generation())' \
        'Ok::<(), BlockError>(())' \
        'Invariant PoolBooks is violated'
fi
