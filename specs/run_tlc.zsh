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
# The trace binding: regenerate the walk trace from the code under
# test, then check every emitted state is a behaviour of the spec.
(cd "$root/.." && INKY_PREFIX_TRACE_OUT="$root/PrefixTreeTraceData.tla" \
    cargo test --locked prefix::tests::prefix_trace_dump -- --ignored --exact >/dev/null) \
    || { print -u2 "prefix trace dump failed"; exit 1; }
check PrefixTreeTrace PrefixTreeTrace

# Mutants of the trace binding: each plant below must turn
# PrefixTreeTrace red for its named reason (a row the model cannot
# step to is a stalled trace and deadlocks; a leaked ref breaks
# PoolBooks). Red for any other reason — a plant that broke trace
# parsing, say — is the gate failing, not the mutant caught. All
# three share one cargo target, so the dependencies build once and
# only the planted crate rebuilds.
mutant() {
    local name=$1 from=$2 to=$3 expected=$4 log="$output/mutant-$1.log" rc
    local tmp=$(mktemp -d)
    mkdir "$tmp/repo"
    (cd "$root/.." && tar --exclude=target --exclude=.git --exclude=tlc-results -cf - .) \
        | (cd "$tmp/repo" && tar -xf -) || { print -u2 "mutant $name: copy failed"; exit 1; }
    MUT_FROM="$from" MUT_TO="$to" perl -0pi -e \
        's/\Q$ENV{MUT_FROM}\E/$ENV{MUT_TO}/ or die "plant not applied\n"' \
        "$tmp/repo/src/prefix.rs" \
        || { print -u2 "mutant $name: plant not applied"; rm -rf "$tmp"; exit 1; }
    (cd "$tmp/repo" && CARGO_TARGET_DIR="$output/mutant-target" \
        INKY_PREFIX_TRACE_OUT="$tmp/repo/specs/PrefixTreeTraceData.tla" \
        cargo test --locked prefix::tests::prefix_trace_dump -- --ignored --exact >/dev/null 2>&1) \
        || { print -u2 "mutant $name: the plant broke the trace walk itself"; rm -rf "$tmp"; exit 1; }
    if "$java" -XX:+UseParallelGC -Xmx4g -cp "$jar" tlc2.TLC \
        -workers 1 -metadir "$output/mutant-$name-states" \
        -config "$tmp/repo/specs/PrefixTreeTrace.cfg" "$tmp/repo/specs/PrefixTreeTrace.tla" \
        >"$log" 2>&1; then rc=0; else rc=$?; fi
    rm -rf "$tmp"
    if [[ $rc -eq 0 ]] || ! grep -Fxq "Error: $expected." "$log"; then
        print -u2 "mutant $name did not go red as $expected; inspect $log"
        exit 1
    fi
    printf "mutant-%s | red as required (%s)\n" "$name" "$expected"
}
mutant no-evict \
    '.filter(|(_, h)| h.entries().any(|e| self.records.generation(e.slot(), e.kind()) != Some(e.generation())))' \
    '.filter(|(_, _h)| false)' \
    'Deadlock reached'
mutant shallowest \
    'if n.starts > 0 {
                best = StartMatch' \
    'if best.at == 0 && n.starts > 0 {
                best = StartMatch' \
    'Deadlock reached'
mutant ref-leak \
    'self.records.drop_ref(e.slot(), e.kind(), e.generation())' \
    'Ok::<(), BlockError>(())' \
    'Invariant PoolBooks is violated'
if [[ -n "${INKY_TLC_NIGHTLY:-}" ]]; then
check PrefixTree PrefixTree
fi
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
check PoisonGate3 PoisonGate
check PoisonGate4 PoisonGate
check PoisonUnguarded PoisonGate NoFalsePoison
check PoisonNoSave PoisonGate LiveIntact
check PoisonOwnedOnly PoisonGate Coverage
