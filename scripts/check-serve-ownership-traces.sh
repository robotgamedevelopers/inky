#!/usr/bin/env bash
# Bind production `ServeBlockCache` (`src/serve/block_cache.rs`), `BlockPools` /
# `BlockTable` (`src/blocks.rs`), `EntryTree` (`src/prefix.rs`), and `SpillStore`
# (`src/spill.rs`) to `specs/ServeOwnershipTrace.tla` (`ServeOwnershipTrace.cfg`),
# and require 8 real single-line mutations across those live serving files to be
# rejected by TLC with an invariant/action counterexample. CPU-only: rustc + Java.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
: "${TLA2TOOLS_JAR:?set TLA2TOOLS_JAR to the pinned tla2tools.jar}"
jar=$(cd "$(dirname "$TLA2TOOLS_JAR")" && pwd)/$(basename "$TLA2TOOLS_JAR")
work=$(mktemp -d "${TMPDIR:-/tmp}/inky-serve-ownership.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

orig_blocks=$work/blocks.rs.orig
orig_book=$work/blockbook.rs.orig
orig_prefix=$work/prefix.rs.orig
orig_spill=$work/spill.rs.orig
orig_cache=$work/block_cache.rs.orig
cp "$root/src/blocks.rs" "$orig_blocks"
cp "$root/src/blockbook.rs" "$orig_book"
cp "$root/src/prefix.rs" "$orig_prefix"
cp "$root/src/spill.rs" "$orig_spill"
cp "$root/src/serve/block_cache.rs" "$orig_cache"

stage_sources() {
    local dir=$1
    cp "$orig_blocks" "$dir/blocks.rs"
    cp "$orig_book" "$dir/blockbook.rs"
    cp "$orig_prefix" "$dir/prefix.rs"
    cp "$orig_spill" "$dir/spill.rs"
    cp "$orig_cache" "$dir/block_cache.rs"
    cp "$root/scripts/serve-ownership-traces.rs" "$dir/main.rs"
}

compile_gen() {
    local dir=$1 warn_flag=${2:-}
    rustc --edition=2021 --cfg standalone_blocks $warn_flag -O "$dir/main.rs" -o "$dir/generate" 2> "$dir/rustc.log" || {
        cat "$dir/rustc.log" >&2
        return 1
    }
}

run_tlc_bundle() {
    local bdir=$1
    cp "$root/specs/ServeOwnershipTrace.tla" "$bdir/"
    cp "$root/specs/ServeOwnershipTrace.cfg" "$bdir/"
    (cd "$bdir" && "${TLC_JAVA:-java}" -Xmx2g -XX:+UseParallelGC -cp "$jar" tlc2.TLC \
        -noGenerateSpecTE -workers 1 -metadir "$bdir/states" \
        -config ServeOwnershipTrace.cfg ServeOwnershipTrace.tla) > "$bdir/tlc.log" 2>&1
}

check_production() {
    local dir=$work/production
    mkdir -p "$dir"
    stage_sources "$dir"
    compile_gen "$dir" "-D warnings" || return 1
    "$dir/generate" "$dir" all 2> "$dir/walk.log" || { cat "$dir/walk.log" >&2; return 1; }
    run_tlc_bundle "$dir/bundle" || { cat "$dir/bundle/tlc.log" >&2; return 1; }
    grep -Fq 'Model checking completed. No error has been found.' "$dir/bundle/tlc.log" || {
        cat "$dir/bundle/tlc.log" >&2
        return 1
    }
    {
        echo "production: $(grep -Em1 '^[0-9,]+ states generated' "$dir/bundle/tlc.log")"
        cat "$dir/walk.log"
    } > "$dir/result"
}

check_mutant() {
    local name=$1 target_file=$2 mode=$3 want_inv=$4 old=$5 new=$6
    local dir=$work/$name status
    mkdir -p "$dir"
    stage_sources "$dir"
    python3 -c 'import pathlib, sys
dst, old, new = sys.argv[1:4]
s = pathlib.Path(dst).read_text()
if s.count(old) != 1:
    sys.stderr.write(f"ERROR: target not unique ({s.count(old)} matches)\n")
    sys.exit(1)
pathlib.Path(dst).write_text(s.replace(old, new, 1))
' "$dir/$target_file" "$old" "$new" || {
        echo "ERROR: $name target is not unique in $target_file" >&2
        return 1
    }
    compile_gen "$dir" || return 1
    "$dir/generate" "$dir" "$mode" 2> "$dir/walk.log" || { cat "$dir/walk.log" >&2; return 1; }
    [[ -f "$dir/bundle/ServeOwnershipTraceData.tla" ]] || {
        cat "$dir/walk.log" >&2
        echo "ERROR: $name produced no ServeOwnershipTraceData.tla" >&2
        return 1
    }
    if run_tlc_bundle "$dir/bundle"; then
        echo "ERROR: $name passed TLC trace checking" >&2
        return 1
    else
        status=$?
    fi
    if [[ $status != 12 && $status != 13 ]] || \
        ! grep -Eq "$want_inv" "$dir/bundle/tlc.log"; then
        cat "$dir/bundle/tlc.log" "$dir/walk.log" >&2
        echo "ERROR: $name failed without expected TLC counterexample '$want_inv' (exit $status)" >&2
        return 1
    fi
    echo "$name rejected: $(grep -Em1 'is violated|were violated|Deadlock reached' "$dir/bundle/tlc.log")" > "$dir/result"
}

names=() pids=()
start() { names+=("$1"); "${@:2}" & pids+=($!); }

start production check_production

# 1. ServeBlockCache::finish_prefill_span omitting start_slot_refs increment -> StartRefsConsistent violated
start start-ref-accounting check_mutant start-ref-accounting block_cache.rs tight \
    'Invariant StartRefsConsistent is violated' \
    'self.inc_start_slot_ref(kind, e.slot()); // MUTANT_ANCHOR_START_SLOT_REF' \
    'let _ = (kind, e);'

# 2. ServeBlockCache::ensure_span_blocks skipping rollback on PoolExhausted -> PoolRefConservation / RollbackAtomic violated
start rollback-leak check_mutant rollback-leak block_cache.rs tight \
    'Invariant (PoolRefConservation|RollbackAtomic) is violated|Deadlock reached' \
    'tbl.clear_kind_entry(pools.of_mut(rk), rbi)' \
    'Ok::<(), crate::blocks::BlockError>(())'

# 3. ServeBlockCache::try_restore_spilled_start leaking provisional restore refs -> PoolRefConservation violated
start restore-ref-leak check_mutant restore-ref-leak block_cache.rs tight \
    'Invariant PoolRefConservation is violated' \
    'Self::release_restore_refs(pools, held); // MUTANT_ANCHOR_RELEASE_RESTORE_REFS' \
    'let _ = held;'

# 4. ServeBlockCache::prune_expired_window skipping clear_kind_entry -> WindowSpanBounded violated
start window-prune-skip check_mutant window-prune-skip block_cache.rs window \
    'Invariant WindowSpanBounded is violated' \
    'tbl.clear_kind_entry(&mut pools.window, bi)' \
    'Ok::<(), crate::blocks::BlockError>(())'

# 5. BlockTable::fill omitting drop_ref when overwriting a previous live entry -> PoolRefConservation violated
start table-overwrite-leak check_mutant table-overwrite-leak blocks.rs tight \
    'Invariant PoolRefConservation is violated' \
    'if drop_prev && !same {' \
    'if false && drop_prev && !same {'

# 6. PrefixTree::deepest_start_in_namespace stopping at shallowest held start -> DeepestStartBound violated
start shallow-prefix-match check_mutant shallow-prefix-match prefix.rs tight \
    'Invariant DeepestStartBound is violated' \
    'if n.starts > 0 { // MUTANT_ANCHOR_DEEPEST_NS' \
    'if n.starts > 0 && best.at == 0 {'

# 7. SpillStore::read_block ignoring SHA-256 checksum mismatch -> CorruptSpillNeverRestored violated
start corrupt-spill-accepted check_mutant corrupt-spill-accepted spill.rs tight \
    'Invariant CorruptSpillNeverRestored is violated' \
    'if !ct_eq_32(&computed, &stored_digest) {' \
    'if false && !ct_eq_32(&computed, &stored_digest) {'

# 8. ServeBlockCache::match_and_bind_in_namespace collapsing caller namespace -> NamespaceIsolated violated
start ns-isolation-breach check_mutant ns-isolation-breach block_cache.rs tight \
    'Invariant NamespaceIsolated is violated' \
    'state.namespace = namespace; // MUTANT_ANCHOR_BIND_NAMESPACE' \
    'state.namespace = 0; let _ = namespace;'

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
echo "serve ownership traces conform to ServeOwnershipTrace; all $(( ${#names[@]} - 1 )) serving mutations rejected."
