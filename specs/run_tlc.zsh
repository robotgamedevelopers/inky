#!/bin/zsh
# Run on the designated bench; negative controls must fail for the right reason.
set -eu
zmodload zsh/datetime
root=${0:A:h}
jar=${TLC_JAR:-$HOME/sdk/tla/tla2tools.jar}
java=${TLC_JAVA:-$HOME/sdk/bin/java}
output=${TLC_OUTPUT:-$PWD/tlc-results}
mkdir -p "$output"
output=${output:A}
[[ -r "$jar" ]] || { print -u2 "Missing TLC jar: $jar"; exit 1; }

check() {
    local name=$1 module=$2 expected=${3:-} kind=${4:-invariant} log="$output/$1.log"
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
    printf "%s | %s | %.3f | %s\n" "$name" "$counts" "$elapsed" "${expected:-PASS}"
}
print "run | generated | distinct | wall seconds | result"
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
