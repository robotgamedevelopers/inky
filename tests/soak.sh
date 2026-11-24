#!/bin/zsh
# tests/soak.sh — N clients against `inky serve` for --duration seconds (receipt run: 3600).
# Mixed prompt lengths, mixed max_tokens, random mid-response disconnects.
# Checks: client in-flight back to 0 (the server has no stats route), no 5xx, and
# iogpu.wired_limit_mb back to the value read before serve. Reads the sysctl only.
# Does not sudo and does not install inky mode.
#
# The bearer is INKY_SERVE_KEY, injected by nova-secrets exec into serve and the
# client. This script never expands that variable. Do not run it under xtrace.
set -eu
setopt pipefail

cd "${0:A:h}/.."

duration=3600
lanes=16
clients=24
bind=127.0.0.1:18090
ctx=131072
queue=64
seed=
inky=
model=
disconnect=30
effort_note=medium

usage() {
    print -r -- "usage: tests/soak.sh [--duration SEC] [--lanes N] [--clients N] [--bind ADDR] [--ctx N] [--queue N] [--seed N] [--inky PATH] [--model PATH] [--disconnect-pct N]"
    print -r -- "receipt run: tests/soak.sh --duration 3600 --lanes 16"
}

while (( $# )); do
    case $1 in
        --duration) duration=$2; shift 2 ;;
        --lanes) lanes=$2; shift 2 ;;
        --clients) clients=$2; shift 2 ;;
        --bind) bind=$2; shift 2 ;;
        --ctx) ctx=$2; shift 2 ;;
        --queue) queue=$2; shift 2 ;;
        --seed) seed=$2; shift 2 ;;
        --inky) inky=$2; shift 2 ;;
        --model) model=$2; shift 2 ;;
        --disconnect-pct) disconnect=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) print -r -- "unknown argument: $1"; usage; exit 2 ;;
    esac
done

integer_ok() { [[ $1 == <-> ]] }
integer_ok "$duration" && (( duration >= 1 )) || { print -r -- "bad --duration"; exit 2 }
integer_ok "$lanes" && (( lanes >= 1 )) || { print -r -- "bad --lanes"; exit 2 }
integer_ok "$clients" && (( clients >= 1 )) || { print -r -- "bad --clients"; exit 2 }
integer_ok "$ctx" && (( ctx >= 64 )) || { print -r -- "bad --ctx"; exit 2 }
integer_ok "$queue" && (( queue >= 0 )) || { print -r -- "bad --queue"; exit 2 }
integer_ok "$disconnect" && (( disconnect >= 0 && disconnect <= 100 )) || { print -r -- "bad --disconnect-pct"; exit 2 }
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

secrets=${NOVA_SECRETS:-}
if [[ -z $secrets ]]; then
    secrets=$(command -v nova-secrets || true)
fi
if [[ -z $secrets ]]; then
    secrets=$HOME/.local/bin/nova-secrets
fi
[[ -x $secrets ]] || { print -r -- "nova-secrets not found"; exit 4 }
[[ -x /opt/homebrew/bin/sops ]] || { print -r -- "sops not found at /opt/homebrew/bin/sops"; exit 4 }
command -v curl >/dev/null || { print -r -- "curl not found"; exit 4 }
command -v go >/dev/null || { print -r -- "go not found (needed to build tests/soak_client.go)"; exit 4 }

if ps -ax -o command= | grep -E '(^|[ /])[i]nky( |$)|[c]argo (build|test|check|run)' >/dev/null; then
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

work=$(mktemp -d "${TMPDIR:-/tmp}/inky-soak-client.XXXXXX")
[[ -n $work && -d $work ]] || { print -r -- "mktemp failed"; exit 1 }
cp tests/soak_client.go "$work/main.go"
(
    cd "$work"
    go mod init inky-soak-client
    GOPROXY=off go build -mod=readonly -o "$work/soak-client" .
)

log=$(mktemp "${TMPDIR:-/tmp}/inky-soak-serve.XXXXXX")
client_out=$(mktemp "${TMPDIR:-/tmp}/inky-soak-client-out.XXXXXX")

# argv only. The key stays in the environment nova-secrets exec becomes.
serve_cmd=(
    "$secrets" exec
    --store "$HOME/.local/share/nova/secrets"
    --as mini-m5
    --key "$HOME/.config/nova-secrets/mini-m5.key"
    --sops /opt/homebrew/bin/sops
    --only INKY_SERVE_KEY
    --
    "$inky" serve
    --key-env INKY_SERVE_KEY
    --lanes "$lanes"
    --bind "$bind"
    -c "$ctx"
    --queue "$queue"
    -n 512
    -m "$model"
)
client_cmd=(
    "$secrets" exec
    --store "$HOME/.local/share/nova/secrets"
    --as mini-m5
    --key "$HOME/.config/nova-secrets/mini-m5.key"
    --sops /opt/homebrew/bin/sops
    --only INKY_SERVE_KEY
    --
    "$work/soak-client"
    -url "http://$bind/v1/chat/completions"
    -duration "$duration"
    -clients "$clients"
    -lanes "$lanes"
    -queue "$queue"
    -seed "$seed"
    -disconnect-pct "$disconnect"
)

print -r -- "command: ${serve_cmd[*]}"
print -r -- "clients_command: ${client_cmd[*]}"

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

http_5xx=${rec[http_5xx]:-}
inflight_end=${rec[inflight_end]:-}
probe_status=${rec[probe_status]:-}
http_401=${rec[http_401]:-}
transport_errors=${rec[transport_errors]:-}
timeouts=${rec[timeouts]:-}

lane_ok=0
if [[ $inflight_end == 0 && $probe_status == 200 ]]; then
    lane_ok=1
fi
five_ok=0
if [[ -s $client_out && $http_5xx == 0 && $http_401 == 0 && $transport_errors == 0 && $timeouts == 0 ]]; then
    five_ok=1
fi

print -r -- "check_lane_leak=$lane_ok inflight_end=${inflight_end:-missing} probe_status=${probe_status:-missing} source=client_outstanding_no_server_stats_route"
print -r -- "check_no_5xx=$five_ok http_5xx=${http_5xx:-missing} http_401=${http_401:-missing} transport_errors=${transport_errors:-missing} timeouts=${timeouts:-missing}"
print -r -- "check_wired_limit=$wired_restored before=$wired_before after=${wired_after:-unread}"
print -r -- "prompt_lengths=short:${rec[prompt_short_bytes]:-64}B medium:${rec[prompt_medium_bytes]:-1024}B long:${rec[prompt_long_bytes]:-4096}B"
print -r -- "max_tokens=${rec[max_tokens]:-16,64,256}"
print -r -- "requests=${rec[requests]:-0} http_200=${rec[http_200]:-0} http_429=${rec[http_429]:-0} disconnected=${rec[disconnected]:-0} inflight_max=${rec[inflight_max]:-0}"
print -r -- "effort_default=$effort_note temperature=0 stream=true"

exit_code=0
if (( lane_ok == 0 || five_ok == 0 || wired_restored == 0 )); then
    exit_code=1
    print -r -- "serve_log=$log"
else
    rm -f -- "$log"
    log=
fi
print -r -- "exit=$exit_code"
exit "$exit_code"
