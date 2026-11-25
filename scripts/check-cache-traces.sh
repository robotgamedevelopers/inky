#!/usr/bin/env bash
# Bind `src/blocks.rs`, `src/cache_trace.rs`, and `src/cache_scheduler.rs` to
# `specs/BlockLayout.tla` and `specs/CacheScheduler.tla`: replay deterministic
# walks through pinned TLC, and require single-line mutations of the production
# files to be rejected by TLC with a counterexample. CPU-only: rustc and Java.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
: "${TLA2TOOLS_JAR:?set TLA2TOOLS_JAR to the pinned tla2tools.jar}"
jar=$(cd "$(dirname "$TLA2TOOLS_JAR")" && pwd)/$(basename "$TLA2TOOLS_JAR")
work=$(mktemp -d "${TMPDIR:-/tmp}/inky-cache-traces.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

JOBS=${INKY_TRACE_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}
if [[ "$JOBS" -lt 1 ]]; then JOBS=1; fi

orig_blocks=$work/blocks.original
orig_sched=$work/cache_scheduler.original
cp "$root/src/blocks.rs" "$orig_blocks"
cp "$root/src/cache_scheduler.rs" "$orig_sched"

compile_gen() {
    local dir=$1 warn_flag=${2:-}
    mkdir -p "$dir/cache_scheduler"
    cp "$root/src/cache_scheduler/walk.rs" "$dir/cache_scheduler/"
    cp "$root/src/blockbook.rs" "$root/src/cache_trace.rs" "$dir/"
    cp "$root/scripts/cache-traces.rs" "$dir/main.rs"
    # Unoptimized codegen compiles ~4x faster on 4-core CI runners while the deterministic
    # walk itself finishes in < 50 ms; -C debug-assertions=on preserves all debug checks.
    rustc --edition=2021 --cfg standalone_blocks -C debug-assertions=on -C debuginfo=0 $warn_flag "$dir/main.rs" -o "$dir/generate" 2> "$dir/rustc.log" || {
        cat "$dir/rustc.log" >&2
        return 1
    }
}

run_tlc_bundle() {
    local bdir=$1 spec_file=$2 workers=${3:-1} mode=${4:-fast}
    cp "$spec_file" "$bdir/"
    if [[ "$mode" == "c2" ]]; then
        (cd "$bdir" && "${TLC_JAVA:-java}" -Xmx768m -XX:+UseParallelGC -XX:CICompilerCount=2 -cp "$jar" tlc2.TLC \
            -noGenerateSpecTE -workers "$workers" -metadir "$bdir/states" \
            -config TraceCheck.cfg TraceCheck.tla) > "$bdir/tlc.log" 2>&1
    else
        (cd "$bdir" && "${TLC_JAVA:-java}" -Xmx512m -XX:+UseSerialGC -XX:TieredStopAtLevel=1 -cp "$jar" tlc2.TLC \
            -noGenerateSpecTE -workers 1 -metadir "$bdir/states" \
            -config TraceCheck.cfg TraceCheck.tla) > "$bdir/tlc.log" 2>&1
    fi
}

check_production() {
    local dir=$work/production
    mkdir -p "$dir"
    cp "$orig_blocks" "$dir/blocks.rs"
    cp "$orig_sched" "$dir/cache_scheduler.rs"
    compile_gen "$dir" "-D warnings" || return 1
    "$dir/generate" "$dir" all 2> "$dir/walk.log" || { cat "$dir/walk.log" >&2; return 1; }
    if grep -q 'COVERAGE GAP' "$dir/walk.log"; then
        cat "$dir/walk.log" >&2
        return 1
    fi
    local bpids=() bnames=(cs bl_base0 bl_high_fp16 bl_high_kv8)
    run_tlc_bundle "$dir/cs" "$root/specs/CacheScheduler.tla" 2 c2 &
    bpids+=($!)
    for b in bl_base0 bl_high_fp16 bl_high_kv8; do
        run_tlc_bundle "$dir/$b" "$root/specs/BlockLayout.tla" 1 fast &
        bpids+=($!)
    done

    local brc=0
    for i in "${!bpids[@]}"; do
        local b=${bnames[$i]}
        if ! wait "${bpids[$i]}"; then
            cat "$dir/$b/tlc.log" >&2
            brc=1
        elif ! grep -Fq 'Model checking completed. No error has been found.' "$dir/$b/tlc.log"; then
            cat "$dir/$b/tlc.log" >&2
            brc=1
        fi
    done
    [[ $brc = 0 ]] || return 1

    {
        for b in bl_base0 bl_high_fp16 bl_high_kv8 cs; do
            echo "$b: $(grep -Em1 '^[0-9,]+ states generated' "$dir/$b/tlc.log")"
        done
        cat "$dir/walk.log"
    } > "$dir/result"
}

check_mutant() {
    local name=$1 target_file=$2 bundle=$3 spec_name=$4 old=$5 new=$6
    local dir=$work/$name status orig
    mkdir -p "$dir"
    cp "$orig_blocks" "$dir/blocks.rs"
    cp "$orig_sched" "$dir/cache_scheduler.rs"
    if [[ "$target_file" == "blocks.rs" ]]; then
        orig=$orig_blocks
    else
        orig=$orig_sched
    fi
    python3 -c 'import pathlib, sys
orig, dst, old, new = sys.argv[1:5]
s = pathlib.Path(orig).read_text()
if s.count(old) != 1:
    sys.stderr.write(f"ERROR: target not unique ({s.count(old)} matches)\n")
    sys.exit(1)
pathlib.Path(dst).write_text(s.replace(old, new, 1))
' "$orig" "$dir/$target_file" "$old" "$new" || {
        echo "ERROR: $name target is not unique in src/$target_file" >&2
        return 1
    }
    if cmp -s "$dir/$target_file" "$orig"; then
        echo "ERROR: $name changed nothing" >&2
        return 1
    fi
    compile_gen "$dir" || return 1
    local gen_mode="${bundle}_mutant"
    "$dir/generate" "$dir" "$gen_mode" "$name" 2> "$dir/walk.log" || { cat "$dir/walk.log" >&2; return 1; }
    [[ -f "$dir/$bundle/TraceCheck.tla" ]] || {
        cat "$dir/walk.log" >&2
        echo "ERROR: $name produced no TraceCheck.tla" >&2
        return 1
    }
    if run_tlc_bundle "$dir/$bundle" "$root/specs/$spec_name"; then
        echo "ERROR: $name passed TLC trace checking" >&2
        return 1
    else
        status=$?
    fi
    if [[ $status != 12 && $status != 13 ]] || \
        ! grep -Eq 'Action property .* is violated|Temporal properties were violated|Invariant .* is violated' "$dir/$bundle/tlc.log"; then
        cat "$dir/$bundle/tlc.log" "$dir/walk.log" >&2
        echo "ERROR: $name failed without a TLC conformance counterexample (exit $status)" >&2
        return 1
    fi
    echo "$name rejected: $(grep -Em1 'is violated|were violated' "$dir/$bundle/tlc.log")" > "$dir/result"
}

names=() pids=()
has_slot() {
    local n=0 pid st
    for pid in "${pids[@]:-}"; do
        [[ -n "$pid" ]] || continue
        if [[ -r "/proc/$pid/stat" ]]; then
            read -r _ _ st _ < "/proc/$pid/stat" 2>/dev/null || continue
            [[ "$st" != "Z" ]] && n=$((n + 1))
        elif kill -0 "$pid" 2>/dev/null; then
            n=$((n + 1))
        fi
    done
    [[ "$n" -lt "$JOBS" ]]
}
start() {
    while ! has_slot; do
        sleep 0.05
    done
    names+=("$1")
    "${@:2}" &
    pids+=($!)
}

start production check_production

# 1. Real blocks::byte_offset 32-bit truncation on >= 4 GiB slot -> rejected by Address64
start trunc-real-byte-offset check_mutant trunc-real-byte-offset blocks.rs bl_high_fp16 BlockLayout.tla \
    'mul_offset(u64::from(slot.pool_index().get()), kind.stride_bytes(kvb)?)' \
    'mul_offset(u64::from(slot.pool_index().get()), kind.stride_bytes(kvb)?).map(|o| o & 0xffff_ffff)'

# 2. Real blocks::byte_offset 1-based off-by-one (slot.get() instead of pool_index()) -> rejected by Address64
start off1-byte-offset check_mutant off1-byte-offset blocks.rs bl_base0 BlockLayout.tla \
    'mul_offset(u64::from(slot.pool_index().get()), kind.stride_bytes(kvb)?)' \
    'mul_offset(u64::from(slot.get()), kind.stride_bytes(kvb)?)'

# 3. SlotRecord::reissue resetting generation to 1 instead of bumping -> rejected by Conforms
start reissue-gen-reset check_mutant reissue-gen-reset blocks.rs bl_base0 BlockLayout.tla \
    '        self.bump_generation()?;' \
    '        self.generation = 1;'

# 4. Finish leaving stale entries in done lane's table while releasing refs -> rejected by GenerationMatches
start stale-finish check_mutant stale-finish cache_scheduler.rs bl_base0 BlockLayout.tla \
    '            l.table = BlockTable::new(num_blocks)?; // MUTANT_ANCHOR_CLEAR_TABLE' \
    '            let _ = num_blocks;'

# 5. Admit dropping wait_ticks increment on bypassed queued lane (MaxQueue=2) -> rejected by Conforms
start admit-wait-dropped check_mutant admit-wait-dropped cache_scheduler.rs cs CacheScheduler.tla \
    '                q.wait_ticks += 1; // MUTANT_ANCHOR_ADMIT_BYPASS' \
    '                let _ = 0;'

# 6. BuildStep dropping wait_ticks increment on StepBypassed lane -> rejected by Conforms
start step-bypass-dropped check_mutant step-bypass-dropped cache_scheduler.rs cs CacheScheduler.tla \
    '                q.wait_ticks += 1; // MUTANT_ANCHOR_STEP_BYPASS' \
    '                let _ = 0;'

# 7. Allocating unsealed private tail with non-zero payload -> rejected by UnsealedPrivateInvisibleZero
start priv-tail-dirty check_mutant priv-tail-dirty cache_scheduler.rs cs CacheScheduler.tla \
    '                    data: vec![0; bsz as usize], // MUTANT_ANCHOR_PRIV_ZERO' \
    '                    data: vec![1; bsz as usize],'

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        cat "$work/${names[$i]}/result"
    else
        echo "FAILED: ${names[$i]}" >&2
        failed=1
    fi
done
[[ $failed = 0 ]] || exit 1
echo "cache traces conform to BlockLayout and CacheScheduler; all $(( ${#names[@]} - 1 )) production mutations rejected."
