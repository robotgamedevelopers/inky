#!/bin/zsh
# Verify that a golden producer cannot hide a nonzero exit behind its output filter.
set -eu
cd "${0:A:h}/.."
D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
cat > "$D/fake" <<'FAKE'
#!/bin/zsh
n=$(cat "$INK_GOLDEN_COUNT")
n=$((n + 1))
print -r -- "$n" > "$INK_GOLDEN_COUNT"
case "$1" in
  check) print 'hash abcdef' ;;
  binv) print 'same' ;;
  chat) cat > /dev/null; print 'text' ;;
  lanes) print 'lanes independent' ;;
  *) print 'text' ;;
esac
[[ $n -ne ${INK_GOLDEN_FAIL:-0} ]] || exit 7
FAKE
chmod +x "$D/fake"
export INK_GOLDEN_COUNT="$D/count"
for failure in {0..10}; do
    print 0 > "$INK_GOLDEN_COUNT"
    export INK_GOLDEN_FAIL=$failure
    if tests/golden.sh "$D/fake" > "$D/out" 2>&1; then
        if (( failure != 0 )); then print -u2 "golden accepted failed producer $failure"; exit 1; fi
        [[ $(wc -l < "$D/out") -eq 10 ]] || { print -u2 'golden omitted successful rows'; exit 1; }
    else
        if (( failure == 0 )); then print -u2 'golden rejected successful producers'; cat "$D/out"; exit 1; fi
        [[ $(cat "$D/count") -eq $failure ]] || { print -u2 "golden did not stop at failed producer $failure"; exit 1; }
    fi
done
print 'golden producer status: all ten failure cases rejected'
