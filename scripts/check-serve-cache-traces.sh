#!/usr/bin/env bash
# Replay actual inky serve cache state on the CPU, with no model weights or Metal.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
: "${TLA2TOOLS_JAR:?set TLA2TOOLS_JAR to the pinned tla2tools.jar}"
jar=$(cd "$(dirname "$TLA2TOOLS_JAR")" && pwd)/$(basename "$TLA2TOOLS_JAR")
work=$(mktemp -d "${TMPDIR:-/tmp}/inky-serve-cache.XXXXXX")
trap 'rm -rf "$work"' EXIT

cp "$root"/src/{blockbook,blocks,prefix,spill}.rs "$work/"
cp "$root/src/serve/block_cache.rs" "$work/"
cp -R "$root/src/serve/block_cache" "$work/"
cp "$root/scripts/serve-cache-traces.rs" "$work/main.rs"
cp "$root"/specs/ServeCacheTrace.{tla,cfg} "$work/"
grep '^pub const EXT:' "$root/src/engine.rs" > "$work/extent.rs"

# Link the same pinned libc as Inky; BlockPools uses host buffers in this build.
libc_lock=$(awk '/^name = "libc"$/{found=1; print "[[package]]"} found && NF {print} found && !NF {exit}' "$root/Cargo.lock")
cat > "$work/Cargo.toml" <<'TOML'
[package]
name = "serve-cache-traces"
version = "0.0.0"
edition = "2021"

[[bin]]
name = "serve-cache-traces"
path = "main.rs"

[dependencies]
libc = "0.2"
TOML
cat > "$work/Cargo.lock" <<LOCK
version = 4

$libc_lock

[[package]]
name = "serve-cache-traces"
version = "0.0.0"
dependencies = ["libc"]
LOCK

check() {
    local label=$1 expected=${2:-} status=0
    (
        cd "$work"
        RUSTFLAGS="${RUSTFLAGS:-} --cfg standalone_blocks --check-cfg=cfg(standalone_blocks)" \
            cargo run --quiet --locked -- ServeCacheTraceData.tla
    ) || return
    (
        cd "$work"
        "${TLC_JAVA:-java}" -Xmx1g -cp "$jar" tlc2.TLC -noGenerateSpecTE -workers 1 \
            -metadir "$work/states-$label" -config ServeCacheTrace.cfg ServeCacheTrace.tla
    ) >"$work/tlc.log" 2>&1 || status=$?

    if [[ -z "$expected" ]]; then
        if [[ $status == 0 ]] &&
            grep -Fq 'Model checking completed. No error has been found.' "$work/tlc.log" &&
            grep -Fq '0 states left on queue.' "$work/tlc.log"; then
            echo "$label: $(grep -m1 'states generated' "$work/tlc.log")"
            return
        fi
    elif [[ $status == 12 ]] &&
        grep -Fxq "Error: Invariant $expected is violated." "$work/tlc.log" &&
        grep -q '^State 2:' "$work/tlc.log"; then
        echo "$label: rejected by $expected"
        return
    fi

    cat "$work/tlc.log" >&2
    echo "$label: unexpected TLC outcome (exit $status)" >&2
    return 1
}

mutant() {
    local label=$1 expected=$2 source=$3 old=$4 new=$5
    python3 - "$root/src/$source" "$work/${source##*/}" "$old" "$new" <<'PY'
from pathlib import Path
import sys

source, target, old, new = sys.argv[1:]
text = Path(source).read_text()
if text.count(old) != 1:
    raise SystemExit(f"mutant target must occur exactly once: {old}")
Path(target).write_text(text.replace(old, new, 1))
PY
    check "$label" "$expected"
    cp "$root/src/$source" "$work/${source##*/}"
}

check production
mutant short-start-window RetainedWindows serve/block_cache.rs \
    'for b in win_lo..(end / BLOCK) {' 'for b in (win_lo + 1)..(end / BLOCK) {'
mutant skip-decode-prune Tables serve/block_cache.rs \
    'self.prune_expired_window(pools, tbl, pos, end);' 'let _ = (pools, tbl, pos, end);'
mutant skip-window-refresh HandleRefresh serve/block_cache.rs \
    'let _ = self.tree.set_handle(id, BlockHandles::new(best_f, Some(best_w)));' \
    'let _ = (id, best_f, best_w);'
mutant skip-release-prune TreePinsAndPruning serve/block_cache.rs \
    'self.tree.prune_unreferenced(tip);' 'let _ = tip;'
mutant allow-complete-prompt Lookup serve/block_cache.rs \
    'let matched = self.tree.deepest_start(&toks[..toks.len() - 1]);' \
    'let matched = self.tree.deepest_start(toks);'
mutant prefer-shallow-checkpoint Lookup prefix.rs \
    'if n.starts > 0 {' 'if n.starts > 0 && best.at == 0 {'
