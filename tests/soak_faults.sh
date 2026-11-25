#!/bin/zsh
# tests/soak_faults.sh — Layer 8 live GPU chaos & longitudinal verification gate:
#   1. Longitudinal zero-drift soak + bounded --record-dir rotation (cards-*.jsonl -> .1)
#   2. Client disconnect storm (72 overlapping requests across 0ms/header-done/ttft/mid-thinking/mid-decode)
#   3. Full disk (hdiutil ENOSPC mount) --record-dir write-error isolation without 5xx/panic/corruption
#   4. SIGKILL (kill -9) crash mid-flight + memd socket-close cleanup + restart + exact seeded output
#
# Never uses /tmp. Always cleans up spawned processes, mounts, and temp dirs via trap EXIT.
set -eu
setopt pipefail

cd "${0:A:h}/.."

inky=
model=
bind=127.0.0.1:18190
lanes=4
ctx=4096
queue=32
seed=20261124
soak_duration=12s
soak_epoch=3s

usage() {
    print -r -- "usage: tests/soak_faults.sh [--inky PATH] [--model PATH] [--bind ADDR] [--lanes N] [--ctx N] [--soak-duration DUR] [--soak-epoch DUR] [--seed N]"
}

while (( $# )); do
    case $1 in
        --inky) inky=$2; shift 2 ;;
        --model) model=$2; shift 2 ;;
        --bind) bind=$2; shift 2 ;;
        --lanes) lanes=$2; shift 2 ;;
        --ctx) ctx=$2; shift 2 ;;
        --soak-duration) soak_duration=$2; shift 2 ;;
        --soak-epoch) soak_epoch=$2; shift 2 ;;
        --seed) seed=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) print -r -- "unknown argument: $1"; usage; exit 2 ;;
    esac
done

if [[ -z $inky ]]; then
    if [[ -x ./target/release/inky ]]; then
        inky=$PWD/target/release/inky
    elif command -v inky >/dev/null 2>&1; then
        inky=$(command -v inky)
    else
        print -r -- "inky binary not found (build target/release/inky first)"
        exit 4
    fi
fi
[[ -x $inky ]] || { print -r -- "inky binary is not executable: $inky"; exit 4 }

if [[ -z $model ]]; then
    model=$HOME/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf
fi
[[ -f $model ]] || { print -r -- "model file not found: $model"; exit 4 }

command -v go >/dev/null || { print -r -- "go not found"; exit 4 }
command -v curl >/dev/null || { print -r -- "curl not found"; exit 4 }
command -v hdiutil >/dev/null || { print -r -- "hdiutil not found (macOS required for ENOSPC disk image)"; exit 4 }

work_root=${INKY_FAULT_WORKDIR:-$PWD/target/soak-faults-scratch}
mkdir -p "$work_root"
work=$(mktemp -d "${work_root%/}/run.XXXXXX")

serve_pid=0
memd_pid=0
bg_pid=0
mounted_dir=

stop_pid() {
    local pid=$1 sig=${2:-TERM}
    if (( pid > 0 )) && kill -0 "$pid" 2>/dev/null; then
        kill "-$sig" "$pid" 2>/dev/null || true
        local i
        for i in {1..15}; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.5
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
    fi
}

cleanup() {
    set +e
    if (( bg_pid > 0 )); then
        stop_pid "$bg_pid" KILL
        bg_pid=0
    fi
    if (( serve_pid > 0 )); then
        stop_pid "$serve_pid" TERM
        serve_pid=0
    fi
    if (( memd_pid > 0 )); then
        stop_pid "$memd_pid" TERM
        memd_pid=0
    fi
    if [[ -n ${mounted_dir:-} ]] && mount | grep -F "$mounted_dir" >/dev/null 2>&1; then
        hdiutil detach "$mounted_dir" >/dev/null 2>&1 || hdiutil detach -force "$mounted_dir" >/dev/null 2>&1 || true
    fi
    if [[ -n ${work:-} && -d $work ]]; then
        chmod -R u+rwX "$work" 2>/dev/null || true
        rm -rf -- "$work"
    fi
}
trap cleanup EXIT

# Generate an ephemeral test bearer in memory only (never passed on argv or logged).
export INKY_SERVE_KEY
INKY_SERVE_KEY=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')

mkdir -p "$work/client"
cp tests/soak_client.go "$work/client/main.go"
(
    cd "$work/client"
    go mod init inky-soak-client >/dev/null 2>&1
    GOPROXY=off go build -mod=readonly -o "$work/soak-client" .
)
client="$work/soak-client"

start_serve() {
    local rec_dir=$1 rec_max=$2 log_file=$3 memd_sock=${4:-}
    local -a env_prefix=()
    if [[ -n $memd_sock ]]; then
        env_prefix=( env "INKY_MEMD_SOCKET=$memd_sock" )
    fi
    "${env_prefix[@]}" "$inky" serve \
        --key-env INKY_SERVE_KEY \
        --lanes "$lanes" \
        --bind "$bind" \
        -c "$ctx" \
        --queue "$queue" \
        --record-dir "$rec_dir" \
        --record-max-bytes "$rec_max" \
        -n 512 \
        -m "$model" >"$log_file" 2>&1 &
    serve_pid=$!

    local i
    for i in {1..180}; do
        if ! kill -0 "$serve_pid" 2>/dev/null; then
            print -r -- "FAIL: inky serve exited before /health ($log_file)"
            tail -n 40 "$log_file" 2>/dev/null || true
            return 1
        fi
        if curl -sf -m 2 "http://$bind/health" >/dev/null; then
            return 0
        fi
        sleep 1
    done
    print -r -- "FAIL: inky serve timed out waiting for /health"
    tail -n 40 "$log_file" 2>/dev/null || true
    return 1
}

stop_serve() {
    if (( serve_pid > 0 )); then
        stop_pid "$serve_pid" TERM
        serve_pid=0
    fi
}

verify_record_dir_intact() {
    local rec_dir=$1 label=$2
    python3 - "$rec_dir" "$label" <<'PY'
import json, pathlib, sys

rec_dir = pathlib.Path(sys.argv[1])
label = sys.argv[2]
if not rec_dir.exists():
    sys.exit(0)
files = sorted(
    p for p in rec_dir.iterdir()
    if p.is_file() and p.name.startswith("cards-") and (p.name.endswith(".jsonl") or p.name.endswith(".jsonl.1"))
)
required = {"schema_version", "recorded_at_ms", "card_id", "turn_index", "request", "response"}
checked_lines = 0
for path in files:
    raw = path.read_bytes()
    if not raw:
        continue
    if not raw.endswith(b"\n"):
        print(f"FAIL [{label}]: torn trailing record (missing newline) in {path}", file=sys.stderr)
        sys.exit(1)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        print(f"FAIL [{label}]: invalid UTF-8 in {path}: {exc}", file=sys.stderr)
        sys.exit(1)
    for idx, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            print(f"FAIL [{label}]: unexpected blank line {idx} in {path}", file=sys.stderr)
            sys.exit(1)
        try:
            obj = json.loads(line)
        except Exception as exc:
            print(f"FAIL [{label}]: invalid JSON at {path}:{idx}: {exc}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(obj, dict) or not required.issubset(obj.keys()):
            print(f"FAIL [{label}]: incomplete record schema at {path}:{idx}: keys={list(obj) if isinstance(obj, dict) else type(obj)}", file=sys.stderr)
            sys.exit(1)
        checked_lines += 1
print(f"record_integrity_ok=1 label={label} files={len(files)} lines={checked_lines}")
PY
}

print -r -- "=== [1/4] Longitudinal Zero-Drift Soak & Bounded Record Rotation ==="
rec1="$work/rec-soak"
log1="$work/serve-1.log"
mkdir -p "$rec1"
start_serve "$rec1" 4096 "$log1"
"$client" \
    -mode soak \
    -url "http://$bind/v1/chat/completions" \
    -duration "$soak_duration" \
    -epoch "$soak_epoch" \
    -min-epochs 3 \
    -warmup-epochs 1 \
    -max-tokens 16,32,64 \
    -clients 4 \
    -lanes "$lanes" \
    -queue "$queue" \
    -seed "$seed" \
    -disconnect-pct 25 \
    -server-pid "$serve_pid" \
    -record-dir "$rec1" \
    -record-max-total-bytes 16384 \
    -max-rss-drift-pct 15 | tee "$work/soak1.out"
grep -q '^drift_ok=1$' "$work/soak1.out" || { print -r -- "FAIL: longitudinal drift check failed"; exit 1 }
epochs_steady=$(awk -F= '/^epochs_steady=/{print $2}' "$work/soak1.out")
(( ${epochs_steady:-0} >= 2 )) || { print -r -- "FAIL: expected >= 2 steady-state epochs, got ${epochs_steady:-0}"; exit 1 }
stop_serve
ls -la "$rec1"
# Ensure rotation kept disk bounded and created at most active + .1
rot_files=( "$rec1"/cards-*.jsonl.1(N) )
(( ${#rot_files} >= 1 )) || { print -r -- "FAIL: expected bounded rotation .1 file in $rec1"; exit 1 }
verify_record_dir_intact "$rec1" "step1-rotation"

print -r -- "=== [2/4] Client Disconnect Storm (72 overlapping requests) ==="
rec2="$work/rec-storm"
log2="$work/serve-2.log"
mkdir -p "$rec2"
start_serve "$rec2" 1048576 "$log2"
"$client" \
    -mode disconnect-storm \
    -url "http://$bind/v1/chat/completions" \
    -clients 16 \
    -lanes "$lanes" \
    -storm-requests 72 \
    -seed "$seed" | tee "$work/storm.out"
grep -q '^drain_ok=1$' "$work/storm.out" && grep -q '^ref_match=1$' "$work/storm.out" || {
    print -r -- "FAIL: disconnect storm failed drain or seeded reference check"
    exit 1
}
stop_serve
verify_record_dir_intact "$rec2" "step2-disconnect-storm"

print -r -- "=== [3/4] Full-Disk (ENOSPC) Record Fault Isolation ==="
mounted_dir="$work/mnt-full"
dmg="$work/full.dmg"
mkdir -p "$mounted_dir"
hdiutil create -size 2m -fs HFS+ -volname inkyfull "$dmg" >/dev/null
hdiutil attach -nobrowse -mountpoint "$mounted_dir" "$dmg" >/dev/null
rec3="$mounted_dir/records"
mkdir -p "$rec3"
# Fill remaining space completely so recorder hits ENOSPC immediately.
dd if=/dev/zero of="$mounted_dir/filler" bs=4096 count=1024 >/dev/null 2>&1 || true
log3="$work/serve-3.log"
start_serve "$rec3" 1048576 "$log3"
"$client" \
    -mode fault-check \
    -url "http://$bind/v1/chat/completions" \
    -duration 8s \
    -clients 8 \
    -lanes "$lanes" \
    -seed "$seed" \
    -expect-record-write-errors | tee "$work/full-disk.out"
grep -q '^drain_ok=1$' "$work/full-disk.out" || { print -r -- "FAIL: full-disk drain failed"; exit 1 }
stop_serve
grep -q '\[record\]' "$log3" || { print -r -- "FAIL: expected [record] error log under ENOSPC"; exit 1 }
verify_record_dir_intact "$rec3" "step3-enospc"
hdiutil detach "$mounted_dir" >/dev/null 2>&1 || hdiutil detach -force "$mounted_dir" >/dev/null 2>&1
mounted_dir=

print -r -- "=== [4/4] SIGKILL (kill -9) Crash, memd Recovery, and Restart Reference Check ==="
memd_sock="$work/memd.sock"
memd_log="$work/memd.log"
crash_rec="$work/rec-crash"
mkdir -p "$crash_rec"
"$inky" memd --fake --owner "$(id -u)" --socket "$memd_sock" >"$memd_log" 2>&1 &
memd_pid=$!
for i in {1..50}; do
    [[ -S $memd_sock ]] && break
    sleep 0.1
done
[[ -S $memd_sock ]] || { print -r -- "FAIL: fake memd socket did not appear"; cat "$memd_log"; exit 1 }

log4a="$work/serve-4a.log"
start_serve "$crash_rec" 1048576 "$log4a" "$memd_sock"
[[ -f "${memd_sock}.saved" ]] || { print -r -- "FAIL: memd saved state file missing while serve running"; cat "$memd_log"; exit 1 }

# Save pre-crash deterministic seeded reference.
"$client" \
    -mode reference-check \
    -url "http://$bind/v1/chat/completions" \
    -lanes "$lanes" \
    -seed "$seed" \
    -reference-file "$work/ref.json"

# Drive concurrent load so multiple lanes are active, then SIGKILL mid-flight.
"$client" \
    -mode soak \
    -url "http://$bind/v1/chat/completions" \
    -duration 30s \
    -clients 8 \
    -lanes "$lanes" \
    -seed "$seed" \
    -disconnect-pct 0 >"$work/crash-load.out" 2>&1 &
bg_pid=$!

for i in {1..100}; do
    h=$(curl -sf -m 1 "http://$bind/health" || true)
    if [[ $h == *'"active_lanes":'[1-9]* ]]; then
        break
    fi
    sleep 0.05
done
sleep 0.2
kill -KILL "$serve_pid"
wait "$serve_pid" 2>/dev/null || true
serve_pid=0
stop_pid "$bg_pid" KILL
bg_pid=0
verify_record_dir_intact "$crash_rec" "step4-post-sigkill"

# Verify MemMode.tla socket-close cleanup restored limit=0 and removed .saved file.
restored=0
for i in {1..50}; do
    if [[ ! -f "${memd_sock}.saved" ]] && grep -q 'limit = 0' "$memd_log"; then
        restored=1
        break
    fi
    sleep 0.1
done
(( restored == 1 )) || { print -r -- "FAIL: memd did not restore limit on SIGKILL socket close"; cat "$memd_log"; exit 1 }

# Restart on the same record-dir and verify exact seeded multi-lane output.
log4b="$work/serve-4b.log"
start_serve "$crash_rec" 1048576 "$log4b" "$memd_sock"
[[ -f "${memd_sock}.saved" ]] || { print -r -- "FAIL: memd did not re-raise limit after restart"; cat "$memd_log"; exit 1 }

"$client" \
    -mode reference-check \
    -url "http://$bind/v1/chat/completions" \
    -lanes "$lanes" \
    -seed "$seed" \
    -reference-file "$work/ref.json" | tee "$work/restart-ref.out"
grep -q '^reference_match=1 ' "$work/restart-ref.out" || { print -r -- "FAIL: post-crash seeded reference mismatch"; exit 1 }

stop_serve
verify_record_dir_intact "$crash_rec" "step4-post-restart"
stop_pid "$memd_pid" TERM
memd_pid=0

print -r -- "ALL FAULT STORMS & LONGITUDINAL SOAK GATES PASSED"
