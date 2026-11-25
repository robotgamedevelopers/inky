# Append to the command arrays used to launch the server and its monitor.
add_spill_args() {
    local dir=$1 budget=$2
    [[ -n $dir ]] || return 0
    serve_args+=( --spill-dir "$dir" --max-spill-bytes "$budget" )
    client_args+=( -spill-dir "$dir" -spill-max-total-bytes "$budget" )
}

# Require observed active work immediately before signaling the process started by the fault gate.
crash_active_server() {
    local pid=$1 url=$2 attempts=$3 active i
    for (( i=0; i<attempts; i++ )); do
        if active=$(curl -sf -m 1 "$url" | plutil -extract active_lanes raw -expect integer -o - - 2>/dev/null) &&
           [[ $active == <1-> ]]; then
            kill -KILL "$pid" || return 1
            print -r -- "crash_active_lanes=$active"
            return 0
        fi
        sleep 0.05
    done
    print -u2 -r -- "FAIL: no active lane observed before the crash deadline"
    return 1
}
