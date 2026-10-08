#!/usr/bin/env bash
# test_mem_gate.sh — Shell-level tests for engine/mem-gate.sh.
#
# Exercises the threshold logic against a fake meminfo: the zero-cost pass
# when already above the threshold, the wait-and-recover path, the hard
# failure on timeout, and rejection of a non-numeric threshold.
#
# Run from the repo root:
#   bash tests/test_mem_gate.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
MEMINFO="$WORK/meminfo"
write_mem() { printf 'MemTotal:       127000000 kB\nMemAvailable:   %s kB\n' "$1" > "$MEMINFO"; }

GATE=(bash engine/mem-gate.sh)

PASS=0
run() {
    local name="$1"
    shift
    if timeout 20 "$@"; then
        PASS=$((PASS + 1))
        echo "ok - $name"
    else
        echo "FAIL/Timeout - $name (exit $?)"
        exit 1
    fi
}

echo "== mem-gate.sh shell tests =="

# 1. Already above the threshold: immediate pass, no waiting.
write_mem 110000000   # ~105 GiB
run "above threshold exits immediately" \
    env LAUNCH_MEMINFO="$MEMINFO" LAUNCH_MIN_MEM_AVAIL_GIB=100 "${GATE[@]}"

# 2. Below the threshold, then memory settles: waits, then passes.
write_mem 90000000    # ~86 GiB
( sleep 2; write_mem 110000000 ) &
run "waits until the threshold is reached" \
    env LAUNCH_MEMINFO="$MEMINFO" LAUNCH_MIN_MEM_AVAIL_GIB=100 \
        LAUNCH_MEM_POLL_S=1 LAUNCH_MEM_WAIT_TIMEOUT_S=10 "${GATE[@]}"

# 3. Stays below the threshold past the deadline: hard failure.
write_mem 90000000
run "timeout is a hard failure (non-zero exit)" bash -c "
    set +e
    LAUNCH_MEMINFO=$MEMINFO LAUNCH_MIN_MEM_AVAIL_GIB=100 \
        LAUNCH_MEM_POLL_S=1 LAUNCH_MEM_WAIT_TIMEOUT_S=2 \
        bash engine/mem-gate.sh >/dev/null 2>&1
    rc=\$?
    set -e
    [[ \$rc -ne 0 ]]
"

# 4. A non-numeric threshold is rejected, not silently miscompared.
run "non-numeric threshold rejected (non-zero exit)" bash -c "
    set +e
    LAUNCH_MEMINFO=$MEMINFO LAUNCH_MIN_MEM_AVAIL_GIB=abc \
        bash engine/mem-gate.sh >/dev/null 2>&1
    rc=\$?
    set -e
    [[ \$rc -ne 0 ]]
"

echo ""
echo "All $PASS shell tests passed."
