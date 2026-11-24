#!/bin/sh
# Run PerKindBlockLayout (2 MiB full / 10 MiB window, plain u64 offsets from base 0)
# and its reversed witnesses in parallel.
set -eu

export PATH="/opt/nova/sdk/bin:$HOME/sdk/bin:$PATH"
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ -n "${TLC_JAR:-}" ]; then
    jar=$TLC_JAR
elif [ -n "${TLA2TOOLS_JAR:-}" ]; then
    jar=$TLA2TOOLS_JAR
elif [ -r "/opt/tla/tla2tools.jar" ]; then
    jar="/opt/tla/tla2tools.jar"
else
    jar="$HOME/sdk/tla/tla2tools.jar"
fi

if [ -n "${TLC_JAVA:-}" ]; then
    java_bin=$TLC_JAVA
elif [ -x "$HOME/sdk/bin/java" ]; then
    java_bin="$HOME/sdk/bin/java"
else
    java_bin="java"
fi

if [ -n "${TLC_OUTPUT:-}" ]; then
    output=$TLC_OUTPUT
    mkdir -p "$output"
else
    output=$(mktemp -d "${TMPDIR:-/tmp}/tlc-per-kind-block-layout.XXXXXX")
    trap 'rm -rf "$output"' EXIT HUP INT TERM
fi
output=$(CDPATH= cd -- "$output" && pwd)

[ -r "$jar" ] || { echo "Missing TLC jar: $jar" >&2; exit 1; }

start_cfg() {
    name=$1
    dir="$output/$name-work"
    mkdir -p "$dir"
    cp "$root/PerKindBlockLayout.tla" "$root/$name.cfg" "$dir/"
    (
        cd "$dir" && "$java_bin" -XX:+UseSerialGC -XX:TieredStopAtLevel=1 -Xmx1g \
            -cp "$jar" tlc2.TLC -noGenerateSpecTE -workers 1 \
            -metadir "$dir/states" -config "$name.cfg" PerKindBlockLayout.tla
    ) >"$output/$name.log" 2>&1 || true
}

for cfg in PerKindBlockLayout3 PerKindHighSlot PerKindWrongStride PerKindTrunc32 PerKindStaleEntry; do
    start_cfg "$cfg" &
done
wait

verify_pass() {
    name=$1
    log="$output/$name.log"
    if ! grep -Fxq "Model checking completed. No error has been found." "$log" || \
       ! grep -Fq "0 states left on queue." "$log" || grep -q "^Error:" "$log"; then
        cat "$log" >&2
        echo "PerKindBlockLayout $name did not complete cleanly" >&2
        exit 1
    fi
    counts=$(awk '/^[0-9,]+ states generated, [0-9,]+ distinct states found/ { print $1 " generated, " $4 " distinct"; exit }' "$log")
    echo "PerKindBlockLayout $name: PASS ($counts)"
}

verify_inv() {
    name=$1
    inv=$2
    log="$output/$name.log"
    if ! grep -Fxq "Error: Invariant $inv is violated." "$log" || \
       ! grep -Fxq "Error: The behavior up to this point is:" "$log"; then
        cat "$log" >&2
        echo "PerKindBlockLayout $name did not show $inv violation" >&2
        exit 1
    fi
    echo "PerKindBlockLayout $name: expected $inv violation"
}

verify_pass PerKindBlockLayout3
verify_pass PerKindHighSlot
verify_inv PerKindWrongStride Address64
verify_inv PerKindTrunc32 Address64
verify_inv PerKindStaleEntry GenerationMatches
