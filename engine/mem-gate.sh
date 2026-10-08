#!/usr/bin/env bash
# ============================================================================
# engine/mem-gate.sh — Pre-launch memory gate.
#
# After a vLLM container dies, the Spark shows phantom CUDA OOM for 30-60s
# while memory settles; relaunching too fast kills the boot (myllmbox
# single-Spark recipe). Wait until MemAvailable >= LAUNCH_MIN_MEM_AVAIL_GIB
# (default 100 GiB on this 121 GiB box) before the container starts. Already
# above the threshold = zero cost; otherwise poll /proc/meminfo every 5s and
# hard-fail after LAUNCH_MEM_WAIT_TIMEOUT_S (default 600s).
#
# Env overrides:
#   LAUNCH_MIN_MEM_AVAIL_GIB   threshold in GiB (default 100)
#   LAUNCH_MEM_WAIT_TIMEOUT_S  give-up deadline in seconds (default 600)
#   LAUNCH_MEM_POLL_S          poll interval in seconds (default 5)
#   LAUNCH_MEMINFO             meminfo path (default /proc/meminfo; tests only)
# ============================================================================
set -euo pipefail

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

MIN_GIB="${LAUNCH_MIN_MEM_AVAIL_GIB:-100}"
TIMEOUT_S="${LAUNCH_MEM_WAIT_TIMEOUT_S:-600}"
POLL_S="${LAUNCH_MEM_POLL_S:-5}"
MEMINFO="${LAUNCH_MEMINFO:-/proc/meminfo}"
[[ "$MIN_GIB" =~ ^[0-9]+$ ]] || err "LAUNCH_MIN_MEM_AVAIL_GIB must be a non-negative integer (got: '$MIN_GIB')"

avail_gib() { awk '/^MemAvailable:/ {print int($2 / 1024 / 1024)}' "$MEMINFO"; }

have=$(avail_gib)
if (( have >= MIN_GIB )); then
    ok "MemAvailable ${have} GiB >= ${MIN_GIB} GiB — launching"
    exit 0
fi

info "Waiting for ${MIN_GIB} GiB MemAvailable before launch (have ${have} GiB, timeout ${TIMEOUT_S}s)..."
deadline=$(( $(date +%s) + TIMEOUT_S ))
while (( have < MIN_GIB )); do
    if (( $(date +%s) > deadline )); then
        err "MemAvailable still ${have} GiB (< ${MIN_GIB} GiB) after ${TIMEOUT_S}s — memory did not settle, not launching. Retry later or lower LAUNCH_MIN_MEM_AVAIL_GIB."
    fi
    sleep "$POLL_S"
    have=$(avail_gib)
    info "  MemAvailable ${have} GiB / ${MIN_GIB} GiB"
done
ok "MemAvailable settled at ${have} GiB — launching"
