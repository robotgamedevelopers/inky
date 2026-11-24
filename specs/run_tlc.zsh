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
    if "$java" -XX:+UseParallelGC -Xmx4g -cp "$jar" tlc2.TLC \
        -workers 1 -metadir "$output/$name-states" \
        -config "$root/$name.cfg" "$root/$module.tla" >"$log" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    elapsed=$(( EPOCHREALTIME - started ))
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
        print -u2 "Unexpected result for $name (exit $rc); inspect $log"
        exit 1
    fi
    local counts
    if [[ $kind = initial ]]; then
        counts="- | -"
    else
        counts=$(awk "/^[0-9,]+ states generated, [0-9,]+ distinct states found/ { print \$1 \" | \" \$4 }" "$log")
    fi
    printf "%s | %s | %.3f | %s\n" "$name" "$counts" "$elapsed" "${expected:-PASS}"
}
print "run | generated | distinct | wall seconds | result"
check BlockLayout3 BlockLayout
check BlockLayout4 BlockLayout
check BlockLayoutHigh BlockLayout
check BlockLayoutKv8 BlockLayout
check StaleEntry BlockLayout GenerationMatches
check WindowOffByOne BlockLayout WindowCovered
check SpanTooSmall BlockLayout SpanPinned
check StartReleased BlockLayout StartRetained
check Truncate32 BlockLayout Address64
check SpanPins512 SpanPins
check SpanPins17 SpanPins Covered
check GroupedLayout3 GroupedLayout
check GroupedLayout4 GroupedLayout
check LocalMatch GroupedLayout VisitedKeys
check DonorWindow GroupedLayout VisitedKeys
check TilePartial GroupedLayout OrderedPartials
check Restage GroupedLayout StagedOnce
check StalledTraversal GroupedLayout TraversalCompletes temporal
check StartTails2 StartTails
check UnheldWindow StartTails StartRetained
check PrunedUnderStart StartTails StartRetained
check TailOverread StartTails ReadsWritten
check ReuseNotZeroed StartTails TailPrivateZero
# Three slots per kind takes about a minute on one worker: the nightly workflow's tla job sets TLC_NIGHTLY.
if [[ -n ${TLC_NIGHTLY:-} ]]; then
    check StartTails3 StartTails
fi
check PoisonGate3 PoisonGate
check PoisonGate4 PoisonGate
check PoisonUnguarded PoisonGate NoFalsePoison
check PoisonNoSave PoisonGate LiveIntact
check PoisonOwnedOnly PoisonGate Coverage
check InBlock InBlock
check InBlockKv8 InBlock
check InBlockMajor InBlock Ordered
check InBlockMajorCollide InBlock NoCollision initial
check InBlockTrace InBlockTrace
check InBlockTraceKv8 InBlockTrace
