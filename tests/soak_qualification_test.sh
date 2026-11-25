#!/bin/zsh
# Exercise the arguments and crash signal decision without starting a model or contacting a server.
set -eu
setopt pipefail
tests=${0:A:h}
source "$tests/soak_qualification.zsh"

fail() { print -u2 -r -- "$*"; exit 1; }

serve_args=(serve)
client_args=(soak)
add_spill_args "" 128
(( ${#serve_args} == 1 && ${#client_args} == 1 )) || fail "spill enabled without a directory"
dir="spill directory"
add_spill_args "$dir" 128
[[ ${#serve_args} == 5 && $serve_args[2] == --spill-dir && $serve_args[3] == "$dir" &&
    $serve_args[4] == --max-spill-bytes && $serve_args[5] == 128 ]] || fail "wrong server spill settings"
[[ ${#client_args} == 5 && $client_args[2] == -spill-dir && $client_args[3] == "$dir" &&
    $client_args[4] == -spill-max-total-bytes && $client_args[5] == 128 ]] || fail "wrong monitor spill settings"

rc=0
zsh "$tests/soak.sh" --inky /nonexistent --spill-max-bytes 128 >/dev/null || rc=$?
(( rc == 2 )) || fail "accepted a spill budget without a directory"

# A sleep after the positive observation, or a signal after the final idle sample, fails the test.
health_reply=
signals=0
sleeps=0
curl() { print -r -- "$health_reply"; }
sleep() { (( ++sleeps )); }
kill() {
    [[ $1 == -KILL && $2 == 47 ]] || fail "unexpected signal"
    (( ++signals ))
}
for health_reply in '{"active_lanes":0}' '{"inflight":2}' '{"active_lanes":"1"}' 'malformed'; do
    if crash_active_server 47 http://fixture/health 1 2>/dev/null; then
        fail "qualified idle or invalid health response"
    fi
    (( signals == 0 )) || fail "signaled without an active lane"
done
sleeps=0
health_reply='{"active_lanes": 12}'
crash_active_server 47 http://fixture/health 1
(( signals == 1 && sleeps == 0 )) || fail "signal did not immediately follow active work"
print -r -- "soak qualification shell tests passed"
