#!/usr/bin/env bash
# Bind `src/poolgate.rs` + `src/blocks.rs` to `specs/PoisonGate.tla` (`U64, U179`):
# replay `poolgate::poisoned` walks (including `Take`, `Share`, and `Release` both
# between steps and inside the poisoned phase) through `specs/PoisonGateTrace.tla`,
# and pin each single-line mutation of `src/poolgate.rs` to the exact TLA+ property
# it violates (`U158`). CPU-only: rustc and Java, no Cargo, Metal, or network.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
: "${TLA2TOOLS_JAR:?set TLA2TOOLS_JAR to the pinned tla2tools.jar}"
jar=$(cd "$(dirname "$TLA2TOOLS_JAR")" && pwd)/$(basename "$TLA2TOOLS_JAR")
work=$(mktemp -d "${TMPDIR:-/tmp}/inky-poisongate.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

orig_poolgate=$work/poolgate.original
cp "$root/src/poolgate.rs" "$orig_poolgate"

run_trace() {
    local dir=$1 warn_flag=${2:-}
    cp "$root"/specs/{PoisonGate.tla,PoisonGateTrace.tla,PoisonGateTrace.cfg} "$root/src/blocks.rs" "$dir/"
    cp "$root/scripts/poisongate-traces.rs" "$dir/main.rs"
    : > "$dir/tlc.log"
    : > "$dir/walk.log"
    rustc --edition=2021 --cfg standalone_blocks $warn_flag -O \
        "$dir/main.rs" -o "$dir/generate" 2> "$dir/rustc.log" || { cat "$dir/rustc.log" >&2; return 1; }
    "$dir/generate" > "$dir/PoisonGateTraceData.tla" 2> "$dir/walk.log" || { cat "$dir/walk.log" >&2; return 1; }
    (cd "$dir" && "${TLC_JAVA:-java}" -Xmx2g -XX:+UseParallelGC -cp "$jar" tlc2.TLC \
        -noGenerateSpecTE -workers 1 -metadir "$dir/states" \
        -config PoisonGateTrace.cfg PoisonGateTrace.tla) > "$dir/tlc.log" 2>&1
}

check_production() {
    local dir=$work/production
    mkdir "$dir"
    cp "$orig_poolgate" "$dir/poolgate.rs"
    if ! run_trace "$dir" "-D warnings"; then cat "$dir/tlc.log" "$dir/walk.log" >&2; return 1; fi
    grep -Fq 'Model checking completed. No error has been found.' "$dir/tlc.log" || { cat "$dir/tlc.log" >&2; return 1; }
    if grep -q 'COVERAGE GAP' "$dir/walk.log"; then cat "$dir/walk.log" >&2; return 1; fi
    { grep -Em1 '^[0-9,]+ states generated' "$dir/tlc.log"; cat "$dir/walk.log"; } > "$dir/result"
}

check_mutant() {
    local name=$1 expected=$2 old=$3 new=$4 dir=$work/$1 status
    mkdir "$dir"
    python3 - "$orig_poolgate" "$dir/poolgate.rs" "$old" "$new" <<'PY'
import pathlib, sys
orig, dst, old, new = sys.argv[1:5]
s = pathlib.Path(orig).read_text()
if s.count(old) != 1:
    sys.stderr.write(f"ERROR: mutant target not unique ({s.count(old)} matches)\n")
    sys.exit(1)
pathlib.Path(dst).write_text(s.replace(old, new, 1))
PY
    if cmp -s "$dir/poolgate.rs" "$orig_poolgate"; then
        echo "ERROR: $name changed nothing" >&2
        return 1
    fi
    if run_trace "$dir"; then
        echo "ERROR: $name unexpectedly passed TLC trace checking" >&2
        return 1
    else
        status=$?
    fi
    if [[ $status != 12 && $status != 13 ]] || ! grep -Fxq "Error: $expected" "$dir/tlc.log"; then
        cat "$dir/tlc.log" "$dir/walk.log" >&2
        echo "ERROR: $name did not fail with pinned 'Error: $expected' (exit $status)" >&2
        return 1
    fi
    echo "$name rejected as required: Error: $expected" > "$dir/result"
}

names=() pids=()
start() { names+=("$1"); "${@:2}" & pids+=($!); }

start production check_production

# 1. Dropping protection of the first stepping lane (`Bug = "unguarded"`) -> NoFalsePoison
start unguarded-stepping check_mutant unguarded-stepping \
    'Invariant NoFalsePoison is violated.' \
    'let steps = stepping.contains(&i) && (guard || Some(&i) != stepping.first());' \
    'let _ = guard; let steps = stepping.contains(&i) && Some(&i) != stepping.first();'

# 2. Omitting `save` of non-stepping live lanes' poisoned slots in `poison_plan` -> ActionConforms on PoisonStep (`saved' # SaveSet`)
start no-save-live check_mutant no-save-live \
    'Action property ActionConforms is violated.' \
    'let save = poison.iter().copied().filter(|&s| owned[s]).collect();' \
    'let save = Vec::new();'

# 3. Poisoning only referenced slots and skipping free slots (`Bug = "ownedonly"`) -> Coverage
start owned-only-poison check_mutant owned-only-poison \
    'Invariant Coverage is violated.' \
    'let poison: Vec<usize> = (0..n_slots).filter(|&s| !protect[s]).collect();' \
    'let poison: Vec<usize> = (0..n_slots).filter(|&s| owned[s] && !protect[s]).collect();'

# 4. Omitting byte copy in `restore` -> LiveIntact
start skip-restore-copy check_mutant skip-restore-copy \
    'Invariant LiveIntact is violated.' \
    'unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), p, n); }' \
    '{ let _ = (&v, p, n); }'

# 5. Stuttering on `Restore` (`note_restored` fired without `restore` running while saved is empty,
#    or `note_applied` passing empty stepping set) -> ActionConforms
start empty-stepping-action check_mutant empty-stepping-action \
    'Action property ActionConforms is violated.' \
    'cand.note_applied(&stepping_ids, &saved);' \
    'cand.note_applied(&[], &saved);'

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
echo "poolgate traces conform to PoisonGate.tla; all $(( ${#names[@]} - 1 )) pinned mutations of src/poolgate.rs rejected."
