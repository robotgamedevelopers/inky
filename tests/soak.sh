#!/bin/zsh
# tests/soak.sh — N clients against `inky serve` for --duration (receipt run: 3600 or 72h).
# Mixed prompt lengths, mixed max_tokens, random mid-response disconnects, longitudinal
# post-drain RSS/lane/block/disk drift monitoring, and bounded --record-dir rotation.
#
# The bearer is INKY_SERVE_KEY, injected by nova-secrets exec into serve and the
# client (or already present in INKY_SERVE_KEY). This script never expands that
# variable on stdout/argv. Do not run it under xtrace.
set -eu
setopt pipefail

cd "${0:A:h}/.."
source tests/soak_qualification.zsh

duration=3600
epoch=0
warmup_epochs=1
lanes=16
clients=24
bind=127.0.0.1:18090
ctx=131072
queue=64
seed=
inky=
model=
disconnect=30
record_dir=
record_max_bytes=1048576
spill_dir=
spill_max_bytes=
max_rss_drift_pct=15
allow_existing=0
effort_note=medium

usage() {
    print -r -- "usage: tests/soak.sh [--duration DUR] [--epoch DUR] [--warmup-epochs N] [--lanes N] [--clients N] [--bind ADDR] [--ctx N] [--queue N] [--seed N] [--inky PATH] [--model PATH] [--disconnect-pct N] [--record-dir DIR] [--record-max-bytes N] [--spill-dir DIR [--spill-max-bytes N]] [--allow-existing-inky]"
    print -r -- "receipt run: tests/soak.sh --duration 3600 --lanes 16"
    print -r -- "multi-day:   tests/soak.sh --duration 72h --lanes 16"
}

while (( $# )); do
    case $1 in
        --duration) duration=$2; shift 2 ;;
        --epoch) epoch=$2; shift 2 ;;
        --warmup-epochs) warmup_epochs=$2; shift 2 ;;
        --lanes) lanes=$2; shift 2 ;;
        --clients) clients=$2; shift 2 ;;
        --bind) bind=$2; shift 2 ;;
        --ctx) ctx=$2; shift 2 ;;
        --queue) queue=$2; shift 2 ;;
        --seed) seed=$2; shift 2 ;;
        --inky) inky=$2; shift 2 ;;
        --model) model=$2; shift 2 ;;
        --disconnect-pct) disconnect=$2; shift 2 ;;
        --record-dir) record_dir=$2; shift 2 ;;
        --record-max-bytes) record_max_bytes=$2; shift 2 ;;
        --spill-dir) spill_dir=$2; shift 2 ;;
        --spill-max-bytes) spill_max_bytes=$2; shift 2 ;;
        --max-rss-drift-pct) max_rss_drift_pct=$2; shift 2 ;;
        --allow-existing-inky) allow_existing=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) print -r -- "unknown argument: $1"; usage; exit 2 ;;
    esac
done

integer_ok() { [[ $1 == <-> ]] }
duration_ok() {
    [[ $1 == <-> ]] && (( $1 >= 1 )) && return 0
    [[ $1 =~ ^[0-9]+(h|m|s)+([0-9]+(h|m|s))*$ ]] && return 0
    return 1
}

duration_ok "$duration" || { print -r -- "bad --duration: $duration"; exit 2 }
[[ $epoch == 0 ]] || duration_ok "$epoch" || { print -r -- "bad --epoch: $epoch"; exit 2 }
integer_ok "$warmup_epochs" || { print -r -- "bad --warmup-epochs"; exit 2 }
integer_ok "$lanes" && (( lanes >= 1 )) || { print -r -- "bad --lanes"; exit 2 }
integer_ok "$clients" && (( clients >= 1 )) || { print -r -- "bad --clients"; exit 2 }
integer_ok "$ctx" && (( ctx >= 64 )) || { print -r -- "bad --ctx"; exit 2 }
integer_ok "$queue" && (( queue >= 0 )) || { print -r -- "bad --queue"; exit 2 }
integer_ok "$disconnect" && (( disconnect >= 0 && disconnect <= 100 )) || { print -r -- "bad --disconnect-pct"; exit 2 }
integer_ok "$record_max_bytes" && (( record_max_bytes >= 1 )) || { print -r -- "bad --record-max-bytes"; exit 2 }
if [[ -n $spill_max_bytes ]]; then
    [[ -n $spill_dir ]] || { print -r -- "--spill-max-bytes requires --spill-dir"; exit 2 }
    integer_ok "$spill_max_bytes" && (( spill_max_bytes > 0 )) || { print -r -- "bad --spill-max-bytes"; exit 2 }
else
    spill_max_bytes=$(( 64 * 1024 * 1024 * 1024 )) # server's default spill budget
fi
if [[ -z $seed ]]; then
    seed=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')
fi
integer_ok "$seed" || { print -r -- "bad --seed"; exit 2 }

if [[ -z $inky ]]; then
    if command -v inky >/dev/null 2>&1; then
        inky=$(command -v inky)
    elif [[ -x ./target/release/inky ]]; then
        inky=$PWD/target/release/inky
    else
        print -r -- "inky binary not found (not on PATH, no ./target/release/inky). Not compiling."
        exit 4
    fi
fi
[[ -x $inky ]] || { print -r -- "inky binary is not executable: $inky"; exit 4 }

if [[ -z $model ]]; then
    model=$HOME/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf
fi
[[ -f $model ]] || { print -r -- "model file not found: $model"; exit 4 }

use_env_key=0
if [[ -n ${INKY_SERVE_KEY:-} ]]; then
    use_env_key=1
else
    secrets=${NOVA_SECRETS:-}
    if [[ -z $secrets ]]; then
        secrets=$(command -v nova-secrets || true)
    fi
    if [[ -z $secrets ]]; then
        secrets=$HOME/.local/bin/nova-secrets
    fi
    [[ -x $secrets ]] || { print -r -- "nova-secrets not found"; exit 4 }
    [[ -x /opt/homebrew/bin/sops ]] || { print -r -- "sops not found at /opt/homebrew/bin/sops"; exit 4 }
    host_id=$(hostname -s)
    if [[ -f "$HOME/.config/nova-secrets/${host_id}.key" ]]; then
        secrets_as=$host_id
        secrets_key="$HOME/.config/nova-secrets/${host_id}.key"
    else
        secrets_as=mini-m5
        secrets_key="$HOME/.config/nova-secrets/mini-m5.key"
    fi
    [[ -f $secrets_key ]] || { print -r -- "nova-secrets key not found: $secrets_key"; exit 4 }
fi

command -v curl >/dev/null || { print -r -- "curl not found"; exit 4 }
command -v go >/dev/null || { print -r -- "go not found (needed to build tests/soak_client.go)"; exit 4 }

if (( allow_existing == 0 )) && ps -ax -o pid=,command= | awk -v self="$$" '$1 != self { $1=""; print substr($0, 2) }' | grep -E '(^|[ /])[i]nky( |$)|[c]argo (build|test|check|run)' >/dev/null; then
    print -r -- "refusing: inky or a cargo build is already running on this machine; not starting a second server"
    exit 3
fi

print -r -- "seed=$seed"
load=$(sysctl -n vm.loadavg)
print -r -- "load=$load"

wired() { sysctl -n iogpu.wired_limit_mb }
wired_before=$(wired) || { print -r -- "cannot read iogpu.wired_limit_mb"; exit 1 }
print -r -- "wired_limit_before=$wired_before"

serve_pid=0
work=
log=
client_out=
cleanup() {
    if (( serve_pid > 0 )) && kill -0 "$serve_pid" 2>/dev/null; then
        kill -TERM "$serve_pid" 2>/dev/null || true
        local i
        for i in {1..20}; do
            kill -0 "$serve_pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$serve_pid" 2>/dev/null; then
            kill -KILL "$serve_pid" 2>/dev/null || true
        fi
        wait "$serve_pid" 2>/dev/null || true
        serve_pid=0
    fi
    if [[ -n ${work:-} && -d $work ]]; then
        rm -rf -- "$work"
        work=
    fi
}
trap cleanup EXIT

scratch_base=${TMPDIR:-$PWD/target/soak-scratch}
mkdir -p "$scratch_base"
work=$(mktemp -d "${scratch_base%/}/inky-soak-client.XXXXXX")
[[ -n $work && -d $work ]] || { print -r -- "mktemp failed"; exit 1 }
cp tests/soak_client.go "$work/main.go"
(
    cd "$work"
    go mod init inky-soak-client
    GOPROXY=off go build -mod=readonly -o "$work/soak-client" .
)

log=$(mktemp "${scratch_base%/}/inky-soak-serve.XXXXXX")
client_out=$(mktemp "${scratch_base%/}/inky-soak-client-out.XXXXXX")
if [[ -z $record_dir ]]; then
    record_dir="$work/records"
fi
mkdir -p "$record_dir"
# Two generations (.jsonl + .jsonl.1) plus single-line framing margin.
record_max_total_bytes=$(( record_max_bytes * 2 + 65536 ))

serve_args=(
    "$inky" serve
    --key-env INKY_SERVE_KEY
    --lanes "$lanes"
    --bind "$bind"
    -c "$ctx"
    --queue "$queue"
    --record-dir "$record_dir"
    --record-max-bytes "$record_max_bytes"
    -n 512
    -m "$model"
)
client_args=(
    "$work/soak-client"
    -url "http://$bind/v1/chat/completions"
    -duration "$duration"
    -epoch "$epoch"
    -warmup-epochs "$warmup_epochs"
    -clients "$clients"
    -lanes "$lanes"
    -queue "$queue"
    -seed "$seed"
    -disconnect-pct "$disconnect"
    -record-dir "$record_dir"
    -record-max-total-bytes "$record_max_total_bytes"
    -max-rss-drift-pct "$max_rss_drift_pct"
)
add_spill_args "$spill_dir" "$spill_max_bytes"

if (( use_env_key == 1 )); then
    serve_cmd=( "${serve_args[@]}" )
    client_cmd_prefix=( )
else
    serve_cmd=(
        "$secrets" exec
        --store "$HOME/.local/share/nova/secrets"
        --as "$secrets_as"
        --key "$secrets_key"
        --sops /opt/homebrew/bin/sops
        --only INKY_SERVE_KEY
        --
        "${serve_args[@]}"
    )
    client_cmd_prefix=(
        "$secrets" exec
        --store "$HOME/.local/share/nova/secrets"
        --as "$secrets_as"
        --key "$secrets_key"
        --sops /opt/homebrew/bin/sops
        --only INKY_SERVE_KEY
        --
    )
fi

print -r -- "command: ${serve_cmd[*]}"

"${serve_cmd[@]}" >"$log" 2>&1 &
serve_pid=$!

health_ok=0
for _ in {1..450}; do
    if ! kill -0 "$serve_pid" 2>/dev/null; then
        print -r -- "serve exited before /health"
        tail -n 40 "$log" 2>/dev/null || true
        exit 1
    fi
    if curl -sf -m 2 "http://$bind/health" >/dev/null; then
        health_ok=1
        break
    fi
    sleep 2
done
if (( health_ok == 0 )); then
    print -r -- "serve did not answer /health"
    tail -n 40 "$log" 2>/dev/null || true
    exit 1
fi

client_cmd=( "${client_cmd_prefix[@]}" "${client_args[@]}" -server-pid "$serve_pid" )
print -r -- "clients_command: ${client_cmd[*]}"

start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
print -r -- "start=$start"
client_ec=0
"${client_cmd[@]}" >"$client_out" || client_ec=$?
end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
print -r -- "end=$end"
print -r -- "client_exit=$client_ec"

# Stop serve before reading the wired limit back. The helper, if installed,
# restores the limit when this process exits. This script does not install it.
if (( serve_pid > 0 )) && kill -0 "$serve_pid" 2>/dev/null; then
    kill -TERM "$serve_pid" 2>/dev/null || true
    for _ in {1..20}; do
        kill -0 "$serve_pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$serve_pid" 2>/dev/null; then
        kill -KILL "$serve_pid" 2>/dev/null || true
    fi
    wait "$serve_pid" 2>/dev/null || true
    serve_pid=0
fi

wired_after=
wired_restored=0
for _ in {1..15}; do
    wired_after=$(wired) || wired_after=
    if [[ -n $wired_after && $wired_after == $wired_before ]]; then
        wired_restored=1
        break
    fi
    sleep 1
done
print -r -- "wired_limit_after=${wired_after:-unread}"

typeset -A rec
if [[ -s $client_out ]]; then
    while IFS='=' read -r k v; do
        [[ -z $k || $k == \#* ]] && continue
        rec[$k]=$v
    done <"$client_out"
fi

inflight_end=${rec[inflight_end]:-}
server_active_lanes_end=${rec[server_active_lanes_end]:-}
server_inflight_end=${rec[server_inflight_end]:-}
server_queued_end=${rec[server_queued_end]:-}
server_health_ok=${rec[server_health_ok]:-}
recovery_completed=${rec[recovery_completed]:-}
recovery_requested=${rec[recovery_requested]:-}
workload_completed=${rec[workload_completed]:-}
drift_ok=${rec[drift_ok]:-}

recovery_ok=0
if [[ $inflight_end == 0 && $server_health_ok == 1 && $server_active_lanes_end == 0 && $server_inflight_end == 0 && ${server_queued_end:-0} == 0 && $recovery_completed == "$lanes" && $recovery_requested == "$lanes" ]]; then
    recovery_ok=1
fi
client_ok=0
if (( client_ec == 0 )); then
    client_ok=1
fi
print -r -- "check_recovery_wave=$recovery_ok completed=${recovery_completed:-missing} requested=${recovery_requested:-missing} server_active_lanes_end=${server_active_lanes_end:-missing} server_inflight_end=${server_inflight_end:-missing} server_queued_end=${server_queued_end:-missing} source=server_health_and_client_outstanding"
print -r -- "check_client=$client_ok workload_completed=${workload_completed:-missing} client_exit=$client_ec"
print -r -- "check_drift=${drift_ok:-0} epochs_total=${rec[epochs_total]:-0} epochs_steady=${rec[epochs_steady]:-0} rss_steady_first=${rec[rss_steady_first_bytes]:-0} rss_steady_last=${rec[rss_steady_last_bytes]:-0} rss_drift_bytes=${rec[rss_drift_bytes]:-0} record_dir_bytes_max=${rec[record_dir_bytes_max]:-0}"
if [[ -n $spill_dir ]]; then
    print -r -- "starts_spilled=${rec[starts_spilled]:-missing} spill_dir_bytes_max=${rec[spill_dir_bytes_max]:-missing} spill_max_bytes=$spill_max_bytes"
fi
print -r -- "check_wired_limit=$wired_restored before=$wired_before after=${wired_after:-unread}"
print -r -- "prompt_lengths=short:${rec[prompt_short_bytes]:-64}B medium:${rec[prompt_medium_bytes]:-1024}B long:${rec[prompt_long_bytes]:-4096}B"
print -r -- "max_tokens=${rec[max_tokens]:-missing}"
print -r -- "requests=${rec[requests]:-0} http_200=${rec[http_200]:-0} http_429=${rec[http_429]:-0} disconnected=${rec[disconnected]:-0} inflight_max=${rec[inflight_max]:-0}"
print -r -- "effort_default=$effort_note temperature=0 stream=true"

exit_code=0
if (( recovery_ok == 0 || client_ok == 0 || wired_restored == 0 )) || [[ ${drift_ok:-0} != 1 ]]; then
    exit_code=1
    print -r -- "serve_log=$log"
else
    rm -f -- "$log"
    log=
fi
print -r -- "exit=$exit_code"
exit "$exit_code"
