#!/bin/sh
# Verify the audit gate rejects a known vulnerable dependency, not just a broken invocation.
set -eu
output=$(cargo audit --no-fetch --file tests/fixtures/audit-vulnerable.lock 2>&1) && {
    echo 'cargo audit unexpectedly accepted the vulnerable fixture' >&2
    exit 1
}
printf '%s\n' "$output"
printf '%s\n' "$output" | grep -q 'RUSTSEC-2020-0071' || {
    echo 'cargo audit failed without detecting the expected advisory' >&2
    exit 1
}
