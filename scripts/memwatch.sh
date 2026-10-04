#!/usr/bin/env bash
# memwatch.sh [--dry-run] [container]   (default container: vllm-fn)
#
# Host-memory watchdog for the single-Spark deployment, adapted from the
# vendor single-Spark recipe's debounced version. On unified memory an
# exhausted pool hangs the kernel instead of raising an OOM, so this stops
# the container when the host runs out of margin. It is a second line of
# defence behind the container's cgroup cap; a userspace poller cannot catch
# a GiB/s collapse alone.
#
# Two independent triggers, each debounced over MEMWATCH_CONSEC consecutive
# 1s samples (a lone excursion is logged and resets the counter — the
# un-debounced version of this script killed healthy servers on transient
# dips):
#   * MemAvailable < MEMWATCH_MIN_GIB   (default 6)  -- page cache gone.
#   * MemFree < MEMWATCH_MIN_FREE_GIB   (default 2)  -- the NVIDIA driver
#     starts refusing allocations (NV_ERR_NO_MEMORY in `journalctl -k`) with
#     MemFree around 3 GiB while MemAvailable still reads 6+ GiB, so the
#     MemAvailable floor alone reacts late. Counted ONLY while MemAvailable
#     is also under MEMWATCH_FREE_GATE_GIB (default 10): with the stock
#     kernel watermarks MemFree legitimately falls near the low watermark
#     whenever the page cache is full of reclaimable data. Free pages backed
#     by reclaimable cache are not what the driver runs out of; free pages
#     with no cache left to reclaim are.
#
# Before stopping the container it archives `docker logs --tail 3000` to
# logs/archive/, then `docker stop -t $MEMWATCH_GRACE` (SIGTERM; a SIGKILL
# leaks the container's POSIX shm onto the host's /dev/shm under --ipc host),
# falling back to docker kill. Also logs a memory timeline every 5 s for
# post-mortems, and appends a WATCHDOG EMERGENCY STOP marker to its own log
# so an emergency stop is distinguishable from a clean stop.sh.
#
# Env: MEMWATCH_MIN_GIB (6), MEMWATCH_MIN_FREE_GIB (2), MEMWATCH_FREE_GATE_GIB (10),
# MEMWATCH_CONSEC (5), MEMWATCH_GRACE (30), MEMWATCH_LOG (this script's own
# log; default logs/memwatch-<container>.log), MEMWATCH_ARCHIVE_DIR (logs/archive).
set -euo pipefail

DRY_RUN=false
CONTAINER="vllm-fn"
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [container]   (default container: vllm-fn)"
            echo "Env: MEMWATCH_MIN_GIB (6) MEMWATCH_MIN_FREE_GIB (2) MEMWATCH_FREE_GATE_GIB (10)"
            echo "     MEMWATCH_CONSEC (5) MEMWATCH_GRACE (30) MEMWATCH_LOG MEMWATCH_ARCHIVE_DIR"
            exit 0
            ;;
        *) CONTAINER="$arg" ;;
    esac
done

MIN_GIB="${MEMWATCH_MIN_GIB:-6}"
MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB:-2}"
FREE_GATE_GIB="${MEMWATCH_FREE_GATE_GIB:-10}"
CONSEC="${MEMWATCH_CONSEC:-5}"
GRACE="${MEMWATCH_GRACE:-30}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
OWN_LOG="${MEMWATCH_LOG:-$REPO_DIR/logs/memwatch-${CONTAINER}.log}"
ARCHIVE_DIR="${MEMWATCH_ARCHIVE_DIR:-$REPO_DIR/logs/archive}"

MIN_KB=$(( MIN_GIB * 1048576 ))
MIN_FREE_KB=$(( MIN_FREE_GIB * 1048576 ))
FREE_GATE_KB=$(( FREE_GATE_GIB * 1048576 ))

if $DRY_RUN; then
    echo "container=$CONTAINER floors: MemAvailable<${MIN_GIB}GiB, MemFree<${MIN_FREE_GIB}GiB (while MemAvailable<${FREE_GATE_GIB}GiB); trigger=${CONSEC} consecutive samples; grace=${GRACE}s"
    echo "log=$OWN_LOG archive=$ARCHIVE_DIR"
    awk '/^(MemFree|MemAvailable|SwapFree):/' /proc/meminfo
    exit 0
fi

echo "$(date '+%F %T') watchdog start: container=$CONTAINER" \
     "floors: MemAvailable<${MIN_GIB}GiB, MemFree<${MIN_FREE_GIB}GiB (while MemAvailable<${FREE_GATE_GIB}GiB);" \
     "trigger=${CONSEC} consecutive samples; grace=${GRACE}s; archive=$ARCHIVE_DIR"

stop_container() {  # <reason>
    local ts; ts=$(date '+%Y%m%dT%H%M%S')
    echo "$(date '+%F %T') $1 -> stopping $CONTAINER"
    mkdir -p "$ARCHIVE_DIR"
    docker logs --tail 3000 "$CONTAINER" > "$ARCHIVE_DIR/${CONTAINER}-$ts-container.log" 2>&1 || true
    docker stop -t "$GRACE" "$CONTAINER" >/dev/null 2>&1 \
        || docker kill "$CONTAINER" >/dev/null 2>&1
    # The marker line is how a later log read tells an emergency stop from a
    # clean stop.sh: only this path emits it.
    echo "WATCHDOG EMERGENCY STOP $1" >> "$OWN_LOG"
    echo "$(date '+%F %T') stopped; container log archived to $ARCHIVE_DIR/${CONTAINER}-$ts-container.log"
    exit 2
}

tick=0
below_avail=0
below_free=0
cg_path=""
while docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}\$"; do
    eval "$(awk '/^(MemFree|MemAvailable|SwapFree):/ { sub(":", "", $1); print "m_" $1 "=" $2 }' /proc/meminfo)"
    avail=$m_MemAvailable; free=$m_MemFree
    if [[ -z "$cg_path" || ! -f "$cg_path" ]]; then
        cg_path="/sys/fs/cgroup/system.slice/docker-$(docker inspect -f '{{.Id}}' "$CONTAINER" 2>/dev/null).scope/memory.current"
    fi
    cg=$(cat "$cg_path" 2>/dev/null || echo 0)

    if (( avail < MIN_KB )); then
        below_avail=$(( below_avail + 1 ))
        echo "$(date '+%F %T') below MemAvailable floor ${below_avail}/${CONSEC}: MemAvailable=$((avail/1024)) MiB MemFree=$((free/1024)) MiB"
        (( below_avail >= CONSEC )) && stop_container "MemAvailable under ${MIN_GIB} GiB for ${CONSEC} samples"
    else
        (( below_avail > 0 )) && echo "$(date '+%T') recovered after ${below_avail} sub-floor MemAvailable sample(s): MemAvailable=$((avail/1024)) MiB"
        below_avail=0
    fi
    if (( free < MIN_FREE_KB && avail < FREE_GATE_KB )); then
        below_free=$(( below_free + 1 ))
        echo "$(date '+%F %T') below MemFree floor ${below_free}/${CONSEC}: MemFree=$((free/1024)) MiB MemAvailable=$((avail/1024)) MiB"
        (( below_free >= CONSEC )) && stop_container "MemFree under ${MIN_FREE_GIB} GiB for ${CONSEC} samples"
    else
        (( below_free > 0 )) && echo "$(date '+%T') recovered after ${below_free} sub-floor MemFree sample(s): MemFree=$((free/1024)) MiB"
        below_free=0
    fi

    if (( tick % 5 == 0 )); then
        echo "$(date '+%T') avail=$((avail/1024))MiB free=$((free/1024))MiB swapfree=$((m_SwapFree/1024))MiB container=$((cg/1048576))MiB"
    fi
    tick=$((tick+1))
    sleep 1
done
echo "$(date '+%F %T') container gone; watchdog exit"
