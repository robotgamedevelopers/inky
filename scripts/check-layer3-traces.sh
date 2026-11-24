#!/usr/bin/env bash
# Layer 3 + Serve trace-conformance gate (pinned TLA+ v1.8.0, parallel JVMs):
#   1. Compiles real production Rust code (`src/blocks.rs` and `src/serve.rs` +
#      `src/serve/scheduler_trace.rs`, plus `src/cache_scheduler.rs` +
#      `src/cache_trace.rs` whenever present) and emits live JSONL traces.
#   2. Checks standalone `specs/PerKindBlockLayout.tla` (`2 MiB` full / `10 MiB`
#      window, plain `u64` offsets from base `0`, `>= 4 GiB` high-slot config,
#      and reversed witnesses `PerKindWrongStride`, `PerKindTrunc32`,
#      `PerKindStaleEntry`).
#   3. Replays all live emitted Rust traces (and any `--block-trace`,
#      `--scheduler-trace`, or `--cache-scheduler-trace` files) with zero slot
#      remapping and zero Python offset arithmetic.
#   4. Runs fail-closed negative controls built from real emitted traces and
#      real single-line Rust source mutations, proving:
#        - unknown events (`bogus_op_xyz`) and empty traces hard-fail (`exit 2`);
#        - truncated 32-bit `u64` offsets and wrong kind strides in `src/blocks.rs`
#          violate `Invariant Address64` (`exit 12`);
#        - stale reused slot generations violate `Invariant GenerationMatches` (`exit 12`);
#        - moving window block 1 to slot 5 while slots 2..4 are free violates
#          `Action property Conforms` inside `PerKindBlockLayout!Alloc` (`exit 12`);
#        - missing 429 refusal violates `Invariant InFlightBounded` (`exit 12`);
#        - over-chunk prefill violates `Invariant StepBudgetAndModeValid` (`exit 12`);
#        - removing `!last_was_prefill` alternation in `src/serve.rs` violates
#          `Action property Conforms` (`exit 12`).
#
# Usage:
#   scripts/check-layer3-traces.sh [--block-trace FILE]... [--scheduler-trace FILE]...
#                                  [--cache-scheduler-trace FILE]... [--no-self-test]
set -euo pipefail

export PATH="/opt/nova/sdk/bin:$HOME/sdk/bin:$HOME/.cargo/bin:$PATH"

root=$(cd "$(dirname "$0")/.." && pwd)
jar_url=https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar
jar_sha256=fdffef247f563ef6edbfd44958d2a82db79e7564e07cb673262d896353fa6713
java_bin=${TLC_JAVA:-java}
rustc_bin=${RUSTC_BIN:-rustc}
work=$(mktemp -d "${TMPDIR:-/tmp}/inky-layer3-traces.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

extra_block=()
extra_sched=()
extra_cache=()
run_self_test=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --block-trace)
            [[ $# -ge 2 ]] || { echo "missing path after --block-trace" >&2; exit 2; }
            extra_block+=("$2")
            shift 2
            ;;
        --scheduler-trace)
            [[ $# -ge 2 ]] || { echo "missing path after --scheduler-trace" >&2; exit 2; }
            extra_sched+=("$2")
            shift 2
            ;;
        --cache-scheduler-trace)
            [[ $# -ge 2 ]] || { echo "missing path after --cache-scheduler-trace" >&2; exit 2; }
            extra_cache+=("$2")
            shift 2
            ;;
        --no-self-test)
            run_self_test=0
            shift
            ;;
        *)
            echo "usage: $0 [--block-trace FILE]... [--scheduler-trace FILE]... [--cache-scheduler-trace FILE]... [--no-self-test]" >&2
            exit 2
            ;;
    esac
done

if [[ -n "${TLA2TOOLS_JAR:-}" ]]; then
    jar=$TLA2TOOLS_JAR
elif [[ -n "${TLC_JAR:-}" ]]; then
    jar=$TLC_JAR
elif [[ -r "/opt/tla/tla2tools.jar" ]]; then
    jar=/opt/tla/tla2tools.jar
elif [[ -r "$HOME/sdk/tla/tla2tools.jar" ]]; then
    jar="$HOME/sdk/tla/tla2tools.jar"
else
    jar=$work/tla2tools.jar
    curl --fail --location --silent --show-error --retry 3 --output "$jar" "$jar_url"
fi

if command -v shasum >/dev/null 2>&1; then
    actual_sha=$(shasum -a 256 "$jar" | awk '{print $1}')
else
    actual_sha=$(sha256sum "$jar" | awk '{print $1}')
fi
if [[ "$actual_sha" != "$jar_sha256" ]]; then
    echo "tla2tools.jar SHA-256 mismatch: got $actual_sha, want $jar_sha256" >&2
    exit 1
fi
jar=$(cd "$(dirname "$jar")" && pwd)/$(basename "$jar")

if [[ -n "${INKY_LAYER3_TRACE_CMD:-}" ]]; then
    (cd "$root" && eval "$INKY_LAYER3_TRACE_CMD")
fi

for env_pair in "INKY_BLOCK_TRACES:block" "INKY_SCHED_TRACES:sched" "INKY_CACHE_SCHEDULER_TRACES:cache"; do
    var_name=${env_pair%%:*}
    kind_tag=${env_pair##*:}
    val=${!var_name:-}
    if [[ -n "$val" ]]; then
        IFS=':' read -r -a parts <<< "$val"
        for p in "${parts[@]}"; do
            [[ -z "$p" ]] && continue
            case "$kind_tag" in
                block) extra_block+=("$p") ;;
                sched) extra_sched+=("$p") ;;
                cache) extra_cache+=("$p") ;;
            esac
        done
    fi
done

# ---------------------------------------------------------------------------
# Stage 1: Compile real Rust emitters (and negative-control Rust mutants) in parallel
# ---------------------------------------------------------------------------
mkdir -p "$work/bin_blocks" "$work/bin_serve" "$work/traces"
cp "$root/src/blocks.rs" "$work/bin_blocks/blocks.rs"
cp "$root/scripts/blocklayout-traces.rs" "$work/bin_blocks/main.rs"
python3 "$root/scripts/extract_serve_trace_runner.py" \
    "$root" "$root/src/serve.rs" "$work/bin_serve/main.rs"

compile_pids=()
"$rustc_bin" --edition=2021 -O "$work/bin_blocks/main.rs" -o "$work/bin_blocks/emit" &
compile_pids+=($!)
"$rustc_bin" --edition=2021 -O "$work/bin_serve/main.rs" -o "$work/bin_serve/emit" &
compile_pids+=($!)

has_w2_cache=0
if [[ -f "$root/src/cache_scheduler.rs" && -f "$root/src/cache_trace.rs" ]]; then
    has_w2_cache=1
    mkdir -p "$work/bin_w2"
    cp "$root/src/blocks.rs" "$work/bin_w2/blocks.rs"
    cp "$root/src/cache_trace.rs" "$work/bin_w2/cache_trace.rs"
    cp "$root/src/cache_scheduler.rs" "$work/bin_w2/cache_scheduler.rs"
    cat > "$work/bin_w2/main.rs" <<'EOF'
mod blocks;
mod cache_trace;
mod cache_scheduler;
fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    let arg: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let out = std::io::stdout();
    let mut sink = cache_trace::JsonlSink::new(std::io::BufWriter::new(out.lock()));
    let mut null = cache_trace::NullSink;
    if mode == "block" {
        let cfg = cache_trace::BlockLayoutConfig {
            lanes: vec!["a".into(), "b".into()],
            full_capacity: 2,
            window_capacity: 2,
            block_size: 2,
            window_size: 2,
            max_pos: 4,
            max_span: 2,
            max_generation: 2,
            slot_base: arg,
        };
        cache_scheduler::walk_block_layout(cfg, 1, &mut sink, &mut null).unwrap();
    } else {
        let cfg = cache_trace::CacheSchedulerConfig {
            lanes: vec!["a".into(), "b".into(), "c".into()],
            prefixes: vec!["p1".into(), "p2".into()],
            request_lengths: vec![2, 4],
            block_size: 2,
            window_size: 2,
            max_pos: 4,
            max_span: 2,
            step_token_budget: 2,
            mem_budget: 4,
            disk_budget: 3,
            max_queue: 1,
            max_bypass: 2,
        };
        cache_scheduler::walk_cache_scheduler(cfg, u64::from(arg.max(1)), &mut sink, &mut null).unwrap();
    }
    sink.flush().unwrap();
}
EOF
    "$rustc_bin" --edition=2021 -O "$work/bin_w2/main.rs" -o "$work/bin_w2/emit" &
    compile_pids+=($!)
fi

if [[ "$run_self_test" -eq 1 ]]; then
    mkdir -p "$work/mut_trunc32" "$work/mut_stride" "$work/mut_serve"
    cp "$root/scripts/blocklayout-traces.rs" "$work/mut_trunc32/main.rs"
    python3 - "$root/src/blocks.rs" "$work/mut_trunc32/blocks.rs" <<'PY'
import pathlib, sys
src = pathlib.Path(sys.argv[1]).read_text()
old = "index.checked_mul(stride).ok_or(BlockError::Overflow)"
assert old in src, "expected checked_mul in mul_offset in src/blocks.rs"
pathlib.Path(sys.argv[2]).write_text(src.replace(old, "index.checked_mul(stride).map(|v| (v as u32) as u64).ok_or(BlockError::Overflow)", 1))
PY
    "$rustc_bin" --edition=2021 -O "$work/mut_trunc32/main.rs" -o "$work/mut_trunc32/emit" &
    compile_pids+=($!)

    cp "$root/scripts/blocklayout-traces.rs" "$work/mut_stride/main.rs"
    python3 - "$root/src/blocks.rs" "$work/mut_stride/blocks.rs" <<'PY'
import pathlib, sys
src = pathlib.Path(sys.argv[1]).read_text()
old = "BlockKind::Window => WINDOW_STRIDE,"
assert old in src, "expected BlockKind::Window => WINDOW_STRIDE in src/blocks.rs"
pathlib.Path(sys.argv[2]).write_text(src.replace(old, "BlockKind::Window => FULL_STRIDE,", 1))
PY
    "$rustc_bin" --edition=2021 -O "$work/mut_stride/main.rs" -o "$work/mut_stride/emit" &
    compile_pids+=($!)

    python3 - "$root/src/serve.rs" "$work/mut_serve/serve.rs" <<'PY'
import pathlib, sys
src = pathlib.Path(sys.argv[1]).read_text()
old = "if !need_pf.is_empty() && (need_dec.is_empty() || !last_was_prefill)"
assert old in src, "expected prefill/decode alternation guard in src/serve.rs"
pathlib.Path(sys.argv[2]).write_text(src.replace(old, "if !need_pf.is_empty()", 1))
PY
    python3 "$root/scripts/extract_serve_trace_runner.py" \
        "$root" "$work/mut_serve/serve.rs" "$work/mut_serve/main.rs"
    "$rustc_bin" --edition=2021 -O "$work/mut_serve/main.rs" -o "$work/mut_serve/emit" &
    compile_pids+=($!)
fi

for pid in "${compile_pids[@]}"; do
    wait "$pid"
done

"$work/bin_blocks/emit" 0    > "$work/traces/real_block_base0.jsonl"
"$work/bin_blocks/emit" 2048 > "$work/traces/real_block_high4g.jsonl"
"$work/bin_serve/emit" 1     > "$work/traces/real_serve_seed1.jsonl"   2>"$work/traces/serve1.err"
"$work/bin_serve/emit" 2     > "$work/traces/real_serve_seed2.jsonl"   2>"$work/traces/serve2.err"
"$work/bin_serve/emit" max   > "$work/traces/real_serve_seedmax.jsonl" 2>"$work/traces/servemax.err"

if [[ "$has_w2_cache" -eq 1 ]]; then
    "$work/bin_w2/emit" block 0    > "$work/traces/w2_block_base0.jsonl"
    "$work/bin_w2/emit" block 2048 > "$work/traces/w2_block_high4g.jsonl"
    "$work/bin_w2/emit" cache 1    > "$work/traces/w2_cache_seed1.jsonl"
fi

if [[ "$run_self_test" -eq 1 ]]; then
    "$work/mut_trunc32/emit" 2048 > "$work/traces/mut_block_trunc32.jsonl"
    "$work/mut_stride/emit" 0     > "$work/traces/mut_block_stride.jsonl"
    "$work/mut_serve/emit" 1      > "$work/traces/mut_serve_no_alt.jsonl" 2>"$work/traces/mut_serve.err"
fi

# ---------------------------------------------------------------------------
# Stage 2: Converter hard-failure negative controls (unknown events & empty trace)
# ---------------------------------------------------------------------------
converter_rejected=0
check_converter_reject() {
    local label=$1 model=$2 trace_file=$3 sabotage=$4 pattern=$5
    local dir="$work/$label"
    mkdir -p "$dir"
    local rc=0
    if python3 "$root/scripts/layer3_trace_to_tla.py" \
        --model "$model" --trace "$trace_file" --out-dir "$dir" --sabotage "$sabotage" \
        >"$dir/conv.out" 2>"$dir/conv.err"; then
        rc=0
    else
        rc=$?
    fi
    if [[ "$rc" -ne 2 ]] || ! grep -Eq "$pattern" "$dir/conv.err"; then
        echo "FAIL [$label]: expected converter hard-failure (exit 2, /$pattern/), got exit $rc:" >&2
        cat "$dir/conv.err" >&2
        exit 1
    fi
    echo "REJECTED (as required) [$label]: $(head -n 1 "$dir/conv.err")"
    converter_rejected=$((converter_rejected + 1))
}

if [[ "$run_self_test" -eq 1 ]]; then
    check_converter_reject "neg-unknown-event-serve" "auto" \
        "$work/traces/real_serve_seed1.jsonl" "unknown-event" \
        "unknown serve_scheduler event 'bogus_op_xyz'"
    check_converter_reject "neg-unknown-event-block" "auto" \
        "$work/traces/real_block_base0.jsonl" "unknown-event" \
        "unknown per_kind_block_layout event 'bogus_op_xyz'"
    printf '{"event":"config","model":"block_layout","pool_mode":"per_kind"}\n' > "$work/traces/empty.jsonl"
    check_converter_reject "neg-empty-trace" "auto" \
        "$work/traces/empty.jsonl" "none" \
        "trace contains zero events"
fi

# ---------------------------------------------------------------------------
# Stage 3: Launch ALL standalone model + trace-conformance JVMs in parallel
# ---------------------------------------------------------------------------
job_labels=()
job_kinds=()
job_expected=()
job_pids=()
max_jobs=${TLC_JOBS:-8}
running_jobs=0
oldest_idx=0

throttle_jobs() {
    if [[ "$running_jobs" -ge "$max_jobs" ]]; then
        wait "${job_pids[$oldest_idx]}" 2>/dev/null || true
        oldest_idx=$((oldest_idx + 1))
        running_jobs=$((running_jobs - 1))
    fi
}

copy_spec_files() {
    local model=$1 dir=$2
    case "$model" in
        per_kind_block_layout)
            cp "$root/specs/PerKindBlockLayout.tla" "$root/specs/trace/PerKindBlockLayoutTraceCheck.tla" "$dir/"
            ;;
        shared_block_layout)
            cp "$root/specs/BlockLayout.tla" "$root/specs/trace/BlockLayoutTraceCheck.tla" "$dir/"
            ;;
        serve_scheduler)
            cp "$root/specs/ServeScheduler.tla" "$root/specs/trace/ServeSchedulerTraceCheck.tla" "$dir/"
            ;;
        cache_scheduler)
            cp "$root/specs/CacheScheduler.tla" "$root/specs/trace/CacheSchedulerTraceCheck.tla" "$dir/"
            ;;
        *)
            echo "internal error: unknown compiled model $model" >&2
            exit 1
            ;;
    esac
}

queue_trace_job() {
    local label=$1 model_arg=$2 trace_file=$3 sabotage=$4 expect_kind=$5 expect_name=${6:-}
    local dir="$work/$label"
    mkdir -p "$dir"
    local meta
    meta=$(python3 "$root/scripts/layer3_trace_to_tla.py" \
        --model "$model_arg" --trace "$trace_file" --out-dir "$dir" --sabotage "$sabotage")
    local resolved_model spec_mod cfg_file step_count
    read -r resolved_model spec_mod cfg_file step_count <<< "$meta"
    [[ "$step_count" -ge 1 ]] || { echo "ERROR: $label produced 0 steps" >&2; exit 1; }
    copy_spec_files "$resolved_model" "$dir"
    printf "%s %s\n" "$resolved_model" "$step_count" > "$dir/meta.txt"

    throttle_jobs
    (
        local rc=0
        if (cd "$dir" && "$java_bin" -Xmx384m -XX:+UseParallelGC -XX:TieredStopAtLevel=1 \
            -cp "$jar" tlc2.TLC -noGenerateSpecTE -workers 1 \
            -metadir "$dir/states" -config "$cfg_file" "$spec_mod.tla") >"$dir/tlc.log" 2>&1; then
            rc=0
        else
            rc=$?
        fi
        echo "$rc" > "$dir/rc.txt"
    ) &
    job_labels+=("$label")
    job_kinds+=("$expect_kind")
    job_expected+=("$expect_name")
    job_pids+=($!)
    running_jobs=$((running_jobs + 1))
}

queue_model_job() {
    local label=$1 cfg_name=$2 expect_kind=$3 expect_name=${4:-}
    local dir="$work/$label"
    mkdir -p "$dir"
    cp "$root/specs/PerKindBlockLayout.tla" "$root/specs/$cfg_name.cfg" "$dir/"
    printf "PerKindBlockLayout 0\n" > "$dir/meta.txt"

    throttle_jobs
    (
        local rc=0
        if (cd "$dir" && "$java_bin" -Xmx384m -XX:+UseParallelGC -XX:TieredStopAtLevel=1 \
            -cp "$jar" tlc2.TLC -noGenerateSpecTE -workers 1 \
            -metadir "$dir/states" -config "$cfg_name.cfg" PerKindBlockLayout.tla) >"$dir/tlc.log" 2>&1; then
            rc=0
        else
            rc=$?
        fi
        echo "$rc" > "$dir/rc.txt"
    ) &
    job_labels+=("$label")
    job_kinds+=("$expect_kind")
    job_expected+=("$expect_name")
    job_pids+=($!)
    running_jobs=$((running_jobs + 1))
}

# 3a. Per-kind BlockLayout standalone model + reversed witnesses
queue_model_job "model-PerKindBlockLayout3"  "PerKindBlockLayout3"  "pass"
queue_model_job "model-PerKindHighSlot"      "PerKindHighSlot"      "pass"
queue_model_job "witness-PerKindWrongStride" "PerKindWrongStride"   "invariant" "Address64"
queue_model_job "witness-PerKindTrunc32"     "PerKindTrunc32"       "invariant" "Address64"
queue_model_job "witness-PerKindStaleEntry"  "PerKindStaleEntry"    "invariant" "GenerationMatches"

# 3b. Positive live-emitted Rust traces
queue_trace_job "pos-real-block-base0"   "per_kind_block_layout" "$work/traces/real_block_base0.jsonl"   "none" "pass"
queue_trace_job "pos-real-block-high4g"  "per_kind_block_layout" "$work/traces/real_block_high4g.jsonl"  "none" "pass"
queue_trace_job "pos-real-serve-seed1"   "serve_scheduler"       "$work/traces/real_serve_seed1.jsonl"   "none" "pass"
queue_trace_job "pos-real-serve-seed2"   "serve_scheduler"       "$work/traces/real_serve_seed2.jsonl"   "none" "pass"
queue_trace_job "pos-real-serve-seedmax" "serve_scheduler"       "$work/traces/real_serve_seedmax.jsonl" "none" "pass"
queue_trace_job "pos-cache-scheduler"    "cache_scheduler"       "$root/specs/trace/fixtures/cache_scheduler_walk.jsonl" "none" "pass"

if [[ -f "$root/specs/BlockLayout.tla" ]] && ! grep -q 'FullCapacity' "$root/specs/BlockLayout.tla"; then
    queue_trace_job "pos-shared-block-layout" "shared_block_layout" "$root/specs/trace/fixtures/block_layout_shared.jsonl" "none" "pass"
fi

if [[ "$has_w2_cache" -eq 1 ]]; then
    queue_trace_job "pos-w2-block-base0"  "auto" "$work/traces/w2_block_base0.jsonl"  "none" "pass"
    queue_trace_job "pos-w2-block-high4g" "auto" "$work/traces/w2_block_high4g.jsonl" "none" "pass"
    queue_trace_job "pos-w2-cache-seed1"  "auto" "$work/traces/w2_cache_seed1.jsonl"  "none" "pass"
fi

idx=0
if [[ ${#extra_block[@]} -gt 0 ]]; then
    for bt in "${extra_block[@]}"; do
        idx=$((idx + 1))
        queue_trace_job "pos-extra-block-$idx" "auto" "$bt" "none" "pass"
    done
fi
if [[ ${#extra_sched[@]} -gt 0 ]]; then
    for st in "${extra_sched[@]}"; do
        idx=$((idx + 1))
        queue_trace_job "pos-extra-sched-$idx" "auto" "$st" "none" "pass"
    done
fi
if [[ ${#extra_cache[@]} -gt 0 ]]; then
    for ct in "${extra_cache[@]}"; do
        idx=$((idx + 1))
        queue_trace_job "pos-extra-cache-$idx" "cache_scheduler" "$ct" "none" "pass"
    done
fi

# 3c. Negative TLC controls built from real emitted traces & real Rust source mutations
if [[ "$run_self_test" -eq 1 ]]; then
    queue_trace_job "neg-rust-blocks-trunc32"      "per_kind_block_layout" "$work/traces/mut_block_trunc32.jsonl"  "none"                  "invariant" "Address64"
    queue_trace_job "neg-rust-blocks-wrong-stride" "per_kind_block_layout" "$work/traces/mut_block_stride.jsonl"   "none"                  "invariant" "Address64"
    queue_trace_job "neg-real-block-nonlowest-slot" "per_kind_block_layout" "$work/traces/real_block_base0.jsonl"  "block-nonlowest-slot"  "action"    "Conforms"
    queue_trace_job "neg-real-block-stale-gen"     "per_kind_block_layout" "$work/traces/real_block_base0.jsonl"   "block-stale-generation" "invariant" "GenerationMatches"
    queue_trace_job "neg-real-serve-over-inflight" "serve_scheduler"       "$work/traces/real_serve_seed1.jsonl"   "serve-over-inflight"   "invariant" "InFlightBounded"
    queue_trace_job "neg-real-serve-over-chunk"    "serve_scheduler"       "$work/traces/real_serve_seed1.jsonl"   "serve-over-chunk"      "invariant" "StepBudgetAndModeValid"
    queue_trace_job "neg-rust-serve-no-alternation" "serve_scheduler"      "$work/traces/mut_serve_no_alt.jsonl"   "none"                  "action"    "Conforms"
fi

for pid in "${job_pids[@]}"; do
    wait "$pid"
done

pass_count=0
tlc_rejected=0

for i in "${!job_labels[@]}"; do
    label=${job_labels[$i]}
    kind=${job_kinds[$i]}
    expected=${job_expected[$i]}
    dir="$work/$label"
    log="$dir/tlc.log"
    rc=$(cat "$dir/rc.txt")
    read -r resolved_model step_count < "$dir/meta.txt"

    case "$kind" in
        pass)
            if [[ "$rc" -ne 0 ]] || \
               ! grep -Fxq 'Model checking completed. No error has been found.' "$log" || \
               ! grep -Fq '0 states left on queue.' "$log" || \
               grep -q '^Error:' "$log"; then
                cat "$log" >&2
                echo "FAIL [$label]: expected PASS (exit 0), got exit $rc" >&2
                exit 1
            fi
            counts=$(awk '/^[0-9,]+ states generated, [0-9,]+ distinct states found/ { print $1 " generated, " $4 " distinct"; exit }' "$log")
            if [[ "$step_count" -gt 0 ]]; then
                want_depth=$((step_count + 1))
                if ! grep -Fxq "The depth of the complete state graph search is ${want_depth}." "$log"; then
                    cat "$log" >&2
                    echo "FAIL [$label]: trace truncated before end (expected BFS depth $want_depth)" >&2
                    exit 1
                fi
                echo "PASS [$label]: $resolved_model ($step_count events; $counts)"
            else
                echo "PASS [$label]: $resolved_model ($counts)"
            fi
            pass_count=$((pass_count + 1))
            ;;
        invariant)
            if [[ "$rc" -ne 12 ]] || ! grep -Fxq "Error: Invariant $expected is violated." "$log"; then
                cat "$log" >&2
                echo "FAIL [$label]: expected TLC exit 12 with 'Error: Invariant $expected is violated.', got exit $rc" >&2
                exit 1
            fi
            echo "REJECTED (as required) [$label]: exit $rc — Error: Invariant $expected is violated."
            tlc_rejected=$((tlc_rejected + 1))
            ;;
        action)
            if { [[ "$rc" -ne 12 ]] && [[ "$rc" -ne 13 ]]; } || ! grep -Fxq "Error: Action property $expected is violated." "$log"; then
                cat "$log" >&2
                echo "FAIL [$label]: expected TLC exit 12/13 with 'Error: Action property $expected is violated.', got exit $rc" >&2
                exit 1
            fi
            echo "REJECTED (as required) [$label]: exit $rc — Error: Action property $expected is violated."
            tlc_rejected=$((tlc_rejected + 1))
            ;;
        *)
            echo "internal error: unknown kind $kind" >&2
            exit 1
            ;;
    esac
done

total_rejected=$((converter_rejected + tlc_rejected))
echo "layer3 trace gate ok: $pass_count positive checks PASS, $total_rejected negative controls REJECTED ($converter_rejected converter + $tlc_rejected TLC exit-12/13 invariant/action witnesses)."
