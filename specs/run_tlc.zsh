#!/usr/bin/env bash
# zsh specs/run_tlc.zsh remains a supported entry point. The queue uses Bash 3+.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
self=$root/run_tlc.zsh
jar=${TLC_JAR:-$HOME/sdk/tla/tla2tools.jar}
java=${TLC_JAVA:-$HOME/sdk/bin/java}
mode=run
case ${1:-} in
    '') ;;
    --list) mode=list ;;
    --one) mode=one ;;  # internal: one queued run, started by xargs below
    *) echo >&2 "usage: run_tlc.zsh [--list]"; exit 2 ;;
esac
if [[ $mode != list ]]; then
    output=${TLC_OUTPUT:-$PWD/tlc-results}
    mkdir -p "$output"
    output=$(cd -- "$output" && pwd)
    [[ -r "$jar" ]] || { echo >&2 "Missing TLC jar: $jar"; exit 1; }
fi

# TLC_ONLY=<modules> runs only the checks of those modules; TLC_EXCEPT=<modules> runs every other check (comma lists).
# TLC_NIGHTLY_ONLY=1 selects only the slow nightly checks (PrefixTree, StartTails3) and PrefixTree trace mutants.
# scripts/check-tla.sh splits its layout part this way into parallel CI jobs. A selection that runs nothing, or names
# a module with no check, fails.
only=${TLC_ONLY:-} except=${TLC_EXCEPT:-} nightly_only=${TLC_NIGHTLY_ONLY:-} ran=0
seen=,
queue=()

now() { perl -MTime::HiRes=time -e 'print time'; }
elapsed_since() { perl -MTime::HiRes=time -e 'printf "%.3f", time - $ARGV[0]' "$1"; }

# Compile the production prefix walk with the host slot records. Each case gets
# its own model and generated trace, so concurrent runs never modify the checkout.
prepare_prefix_trace() {
    local name=$1 dir=$2 from to
    cp "$root"/{PrefixTree.tla,PrefixTreeTrace.tla,PrefixTreeTrace.cfg} "$dir/"
    cp "$root/../src/prefix.rs" "$dir/"
    case $name in
        mutant-no-evict)
            from='.filter(|(_, h)| h.entries().any(|e| self.records.generation(e.slot(), e.kind()) != Some(e.generation())))'
            to='.filter(|(_, _h)| false)' ;;
        mutant-full-match)
            from='!tokens.is_empty() && m.at == tokens.len()'
            to='false && m.at == tokens.len()' ;;
        mutant-ref-leak)
            from='self.records.drop_ref(e.slot(), e.kind(), e.generation())'
            to='Ok::<(), BlockError>(())' ;;
    esac
    if [[ -n ${from:-} ]]; then
        MUT_FROM="$from" MUT_TO="$to" perl -0pi -e \
            's/\Q$ENV{MUT_FROM}\E/$ENV{MUT_TO}/ or die "plant not applied\n"' "$dir/prefix.rs"
    fi
    printf '%s\n' 'extern crate blocks;' 'mod prefix;' > "$dir/main.rs"
    rustc --edition=2021 --extern blocks="$dir/libblocks.rlib" -A dead_code --test "$dir/main.rs" -o "$dir/walk"
    rm -f "$dir/PrefixTreeTraceData.tla"
    INKY_PREFIX_TRACE_OUT="$dir/PrefixTreeTraceData.tla" "$dir/walk" \
        --exact prefix::tests::prefix_trace_dump --ignored > "$dir/walk.log"
    grep -Fq 'test result: ok. 1 passed;' "$dir/walk.log" && [[ -s "$dir/PrefixTreeTraceData.tla" ]] || { echo >&2 "prefix trace dump failed: $name"; exit 1; }
}

# Use the same address generator as "inky inblock-trace", without linking Metal.
prepare_inblock_trace() {
    local name=$1 dir=$2
    cp "$root/InBlockTrace.tla" "$root/$name.cfg" "$dir/"
    printf '%s\n' 'fn main() -> std::io::Result<()> {' \
        '    blocks::write_inblock_trace(&std::env::args().nth(1).unwrap())' '}' > "$dir/main.rs"
    rustc --edition=2021 --extern blocks="$dir/libblocks.rlib" "$dir/main.rs" -o "$dir/generate"
    rm -f "$dir/InBlockTraceData.tla"
    "$dir/generate" "$dir/InBlockTraceData.tla"
}

# run_one NAME MODULE EXPECTED KIND: one TLC run; EXPECTED is - for a run that must pass. Its table row goes to
# $output/NAME.result, which exists only when the run gave exactly the expected result.
run_one() {
    local name=$1 module=$2 expected=$3 kind=$4 log="$output/$1.log"
    local specdir=$root config=$name workers=1
    if [[ $name = PrefixTreeNightly ]]; then workers=${TLC_WORKERS:-1}; fi
    if [[ $expected = - ]]; then expected=; fi
    if [[ $module = PrefixTreeTrace || $module = InBlockTrace ]]; then
        specdir=$output/$name-trace
        mkdir -p "$specdir"
        rustc --edition=2021 --cfg standalone_blocks --crate-type rlib \
            "$root/../src/blocks.rs" -o "$specdir/libblocks.rlib"
        if [[ $module = PrefixTreeTrace ]]; then
            config=PrefixTreeTrace
            prepare_prefix_trace "$name" "$specdir"
        else
            prepare_inblock_trace "$name" "$specdir"
        fi
    fi
    local started elapsed rc
    started=$(now)
    if [[ $kind = simulation ]]; then
        "$java" -XX:+UseParallelGC -Xmx4g -cp "$jar" tlc2.TLC -noGenerateSpecTE \
            -workers 1 -metadir "$output/$name-states" -simulate num=256 -depth 128 -seed 431273 \
            -config "$root/$name.cfg" "$root/$module.tla" >"$log" 2>&1
        grep '^Progress: .*256 traces generated (trace length: mean=128,' "$log" > "$output/$name.result"
        return
    fi
    if "$java" -XX:+UseParallelGC -Xmx4g -cp "$jar" tlc2.TLC -noGenerateSpecTE \
        -workers "$workers" -metadir "$output/$name-states" \
        -config "$specdir/$config.cfg" "$specdir/$module.tla" >"$log" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    elapsed=$(elapsed_since "$started")
    if [[ $kind = trace-mutant ]]; then
        if [[ $rc -eq 0 ]] || ! grep -Fxq "Error: $expected." "$log"; then
            echo >&2 "$name failed to produce its $expected witness; inspect $log"
            exit 1
        fi
        printf "%s | red as required (%s)\n" "$name" "$expected" > "$output/$name.result"
        return
    fi
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
        if [[ $module = InBlockTrace ]]; then
            # Both element widths enumerate 2 passes x 2 planes x 3 heads x 4 positions.
            grep -Eq '^[0-9,]+ states generated, 48 distinct states found, 0 states left on queue\.' "$log" || ok=false
        fi
    fi
    if [[ $ok != true ]]; then
        echo >&2 "Unexpected result for $name (exit $rc); inspect $log"
        exit 1
    fi
    if [[ $module = InBlockTrace ]] && ! cmp -s "$specdir/InBlockTraceData.tla" "$root/InBlockTraceData.tla"; then
        echo >&2 "InBlockTraceData.tla is stale; regenerate it with inky inblock-trace. Fresh trace: $specdir/InBlockTraceData.tla"
        exit 1
    fi
    local counts
    if [[ $kind = initial ]]; then
        counts="- | -"
    else
        counts=$(awk "/^[0-9,]+ states generated, [0-9,]+ distinct states found/ { print \$1 \" | \" \$4 }" "$log")
    fi
    printf "%s | %s | %.3f | %s\n" "$name" "$counts" "$elapsed" "${expected:-PASS}" > "$output/$name.result"
}
if [[ $mode = one ]]; then run_one "${@:2}"; exit 0; fi

select_check() {
    local name=$1 module=$2 expected=${3:-} kind=${4:-invariant}
    if [[ -n $only && ",$only," != *",$module,"* ]] || [[ -n $except && ",$except," == *",$module,"* ]]; then
        return 0
    fi
    ran=$(( ran + 1 )); seen="$seen$module,"
    queue+=("$name" "$module" "${expected:--}" "$kind")
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
# The 21-node full config has at least 6^20 slot assignments. Nightly CI
# exhausts two slots with up to three sealed nodes and simulates the full config.
check_nightly PrefixTreeNightly PrefixTree
if [[ -n ${INKY_TLC_NIGHTLY:-} && -z $nightly_only ]]; then
    check_nightly PrefixTree PrefixTree
else
    check_nightly PrefixTree PrefixTree - simulation
fi
check_nightly mutant-no-evict PrefixTreeTrace 'Deadlock reached' trace-mutant
check_nightly mutant-full-match PrefixTreeTrace 'Deadlock reached' trace-mutant
check_nightly mutant-ref-leak PrefixTreeTrace 'Invariant PoolBooks is violated' trace-mutant
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
check AdvanceUnfair GroupedLayout TraversalCompletes temporal
# Negative control: the redundant-load mutant must keep both output invariants,
# so only StagedOnce can see it. Restage above proves the same mutant is live.
check RestageOutputs GroupedLayout
check StartTails2 StartTails
check UnheldWindow StartTails StartRetained
check PrunedUnderStart StartTails StartRetained
check TailOverread StartTails ReadsWritten
check ReuseNotZeroed StartTails TailPrivateZero
# Three slots per kind takes about a minute on one worker: the nightly workflow runs it via layout-nightly.
check_nightly StartTails3 StartTails
check PoisonGate3 PoisonGate
check PoisonGate4 PoisonGate
check PoisonUnguarded PoisonGate NoFalsePoison
check PoisonNoSave PoisonGate LiveIntact
check PoisonOwnedOnly PoisonGate Coverage
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

(( ran > 0 )) || { echo >&2 "run_tlc.zsh: no check selected (TLC_ONLY=$only TLC_EXCEPT=$except TLC_NIGHTLY_ONLY=$nightly_only)"; exit 1; }
if [[ -n $only ]]; then
    IFS=, read -r -a requested <<< "$only"
    for module in "${requested[@]}"; do
        [[ $seen = *",$module,"* ]] || { echo >&2 "run_tlc.zsh: TLC_ONLY names $module, which has no check"; exit 1; }
    done
fi
names=()
for (( i=0; i<${#queue[@]}; i+=4 )); do names+=("${queue[i]}"); done
if [[ $mode = list ]]; then
    printf '%s\n' "${names[@]}"
    exit 0
fi

njobs=${TLC_JOBS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)}
[[ $njobs =~ ^[1-9][0-9]*$ ]] || { echo >&2 "run_tlc.zsh: TLC_JOBS must be a positive integer"; exit 1; }
for name in "${names[@]}"; do rm -f "$output/$name.result"; done
export TLC_JAR=$jar TLC_JAVA=$java TLC_OUTPUT=$output
started=$(now)
xrc=0
printf '%s\0' "${queue[@]}" | xargs -0 -n 4 -P "$njobs" bash "$self" --one || xrc=$?
echo "run | generated | distinct | wall seconds | result"
failed=
for name in "${names[@]}"; do
    if [[ -s $output/$name.result ]]; then cat "$output/$name.result"; else failed="$failed $name"; fi
done
printf "%d runs, %d at a time: %s s wall\n" "${#names[@]}" "$njobs" "$(elapsed_since "$started")"
if [[ -n $failed || $xrc != 0 ]]; then
    echo >&2 "run_tlc.zsh: failed:$failed (xargs exit $xrc)"
    exit 1
fi
