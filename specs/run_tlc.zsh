#!/bin/zsh
# Run on the designated bench; negative controls must fail for the right reason.
# The `check` lines below queue their runs; the queue then runs TLC_JOBS at a time (default: one per available core), each a
# single-worker TLC in its own process, and the table prints in file order once every run has finished.
# `run_tlc.zsh --list` prints the names of the selected checks and runs nothing: scripts/check-tla-tiers.sh reads it.
set -eu
zmodload zsh/datetime
root=${0:A:h} self=${0:A}
jar=${TLC_JAR:-$HOME/sdk/tla/tla2tools.jar}
java=${TLC_JAVA:-$HOME/sdk/bin/java}
mode=run
case ${1:-} in
    '') ;;
    --list) mode=list ;;
    --one) mode=one ;;  # internal: one queued run, started by xargs below
    *) print -u2 "usage: run_tlc.zsh [--list]"; exit 2 ;;
esac
if [[ $mode != list ]]; then
    output=${TLC_OUTPUT:-$PWD/tlc-results}
    mkdir -p "$output"
    output=${output:A}
    [[ -r "$jar" ]] || { print -u2 "Missing TLC jar: $jar"; exit 1; }
fi
# TLC_ONLY=<modules> runs only the checks of those modules; TLC_EXCEPT=<modules> runs every other check (comma lists).
# scripts/check-tla.sh splits its layout part this way into parallel CI jobs. A selection that runs nothing, or names
# a module with no check, fails.
only=${TLC_ONLY:-} except=${TLC_EXCEPT:-} ran=0
typeset -A seen
typeset -a queue

# run_one NAME MODULE EXPECTED KIND: one TLC run; EXPECTED is - for a run that must pass. Its table row goes to
# $output/NAME.result, which exists only when the run gave exactly the expected result.
run_one() {
    local name=$1 module=$2 expected=${3:#-} kind=$4 log="$output/$1.log"
    local started=$EPOCHREALTIME elapsed rc
    if "$java" -XX:+UseParallelGC -Xmx4g -cp "$jar" tlc2.TLC -noGenerateSpecTE \
        -workers 1 -metadir "$output/$name-states" \
        -config "$root/$name.cfg" "$root/$module.tla" >"$log" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    elapsed=$(( EPOCHREALTIME - started ))
    local ok=true
    grep -Eq "^[0-9,]+ states generated, [0-9,]+ distinct states found" "$log" || ok=false
    grep -q "^Finished in " "$log" || ok=false
    if [[ -n "$expected" ]]; then
        [[ $rc -ne 0 ]] || ok=false
        [[ $(grep -c "^Error: " "$log" || true) = 2 ]] || ok=false
        if [[ $kind = temporal ]]; then
            # Older TLC reports an unnamed temporal violation. The witness
            # config must name precisely the requested property in that case.
            if ! grep -Fxq "Error: Temporal property $expected was violated." "$log"; then
                grep -Fxq "Error: Temporal properties were violated." "$log" || ok=false
                [[ $(awk '/^PROPERTY / {print substr($0,10)}' "$root/$name.cfg") = "$expected" ]] || ok=false
            fi
            grep -Fxq "Error: The following behavior constitutes a counter-example:" "$log" || ok=false
            grep -Eq '^State [0-9]+: Stuttering$' "$log" || ok=false
        else
            grep -Fxq "Error: Invariant $expected is violated." "$log" || ok=false
            grep -Fxq "Error: The behavior up to this point is:" "$log" || ok=false
        fi
        grep -q "^State 2:" "$log" || ok=false
    else
        [[ $rc -eq 0 ]] || ok=false
        if grep -q "^Error:" "$log"; then ok=false; fi
        grep -Fxq "Model checking completed. No error has been found." "$log" || ok=false
    fi
    if [[ $ok != true ]]; then
        print -u2 "Unexpected result for $name (exit $rc); inspect $log"
        exit 1
    fi
    local counts=$(awk "/^[0-9,]+ states generated, [0-9,]+ distinct states found/ { print \$1 \" | \" \$4 }" "$log")
    printf "%s | %s | %.3f | %s\n" "$name" "$counts" "$elapsed" "${expected:-PASS}" > "$output/$name.result"
}
if [[ $mode = one ]]; then run_one "${@:2}"; exit 0; fi

check() {
    local name=$1 module=$2 expected=${3:-} kind=${4:-invariant}
    if [[ -n $only && ",$only," != *",$module,"* ]] || [[ -n $except && ",$except," == *",$module,"* ]]; then
        return 0
    fi
    ran=$(( ran + 1 )); seen[$module]=1
    queue+=("$name" "$module" "${expected:--}" "$kind")
}
check BlockLayout3 BlockLayout
check BlockLayout4 BlockLayout
check StaleEntry BlockLayout GenerationMatches
check WindowOffByOne BlockLayout WindowCovered
check SpanTooSmall BlockLayout SpanPinned
check StartReleased BlockLayout StartRetained
check SpanPins512 SpanPins
check SpanPins17 SpanPins Covered
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
check SplitArena3 SplitArena
check SplitArena4 SplitArena
check SplitArena6 SplitArena
check SplitEdgeDropped SplitArena CombineReadsOwn
check SplitRowsUnbounded SplitArena WritesInArena
check PoisonGate3 PoisonGate
check PoisonGate4 PoisonGate
check PoisonUnguarded PoisonGate NoFalsePoison
check PoisonNoSave PoisonGate LiveIntact
check PoisonOwnedOnly PoisonGate Coverage
(( ran > 0 )) || { print -u2 "run_tlc.zsh: no check selected (TLC_ONLY=$only TLC_EXCEPT=$except)"; exit 1; }
for m in ${(s:,:)only}; do
    [[ -n ${seen[$m]:-} ]] || { print -u2 "run_tlc.zsh: TLC_ONLY names $m, which has no check"; exit 1; }
done
if [[ $mode = list ]]; then
    for (( i = 1; i <= $#queue; i += 4 )); do print -r -- $queue[i]; done
    exit 0
fi
njobs=${TLC_JOBS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)}  # nproc honours a CPU reservation
[[ $njobs = <1-> ]] || { print -u2 "run_tlc.zsh: TLC_JOBS must be a positive integer, not $njobs"; exit 1; }
for (( i = 1; i <= $#queue; i += 4 )); do rm -f "$output/$queue[i].result"; done
export TLC_JAR=$jar TLC_JAVA=$java TLC_OUTPUT=$output
started=$EPOCHREALTIME xrc=0
print -rN -- "${queue[@]}" | xargs -0 -n 4 -P "$njobs" zsh "$self" --one || xrc=$?
print "run | generated | distinct | wall seconds | result"
typeset -a failed
for (( i = 1; i <= $#queue; i += 4 )); do
    if [[ -f $output/$queue[i].result ]]; then cat "$output/$queue[i].result"; else failed+=($queue[i]); fi
done
printf "%d runs, %d at a time: %.3f s wall\n" $(( $#queue / 4 )) $njobs $(( EPOCHREALTIME - started ))
if (( $#failed || xrc )); then
    print -u2 "run_tlc.zsh: failed: ${failed[*]:-none missing, but xargs exited $xrc}"
    exit 1
fi
