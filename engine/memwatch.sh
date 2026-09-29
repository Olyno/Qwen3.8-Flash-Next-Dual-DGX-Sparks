#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# engine/memwatch.sh — watchdog STARTER, ported from the single-spark sister
# repo's scripts/start-memwatch.sh:1-31 so start.sh and any future supervisor
# cannot drift on the arguments.
#
# The watchdog BODY is files/memwatch.sh (177-line poll loop; kept as a runtime
# dependency file, not split — it is a single conceptual unit, see
# files/NOTES.md). Its policy, ported verbatim from files/memwatch.sh:6-44:
#   * On unified memory an exhausted pool hangs the kernel instead of raising
#     an OOM, so this is the second line of defence behind HOST_RESERVE_GIB.
#   * Triggers, each debounced over CONSEC samples: MemAvailable <
#     MEMWATCH_MIN_GIB (page cache gone, PLE lookups about to hit NVMe) and
#     MemFree < MEMWATCH_MIN_FREE_GIB counted ONLY while MemAvailable is also
#     under MEMWATCH_FREE_GATE_GIB (the NVIDIA driver starts refusing
#     allocations with MemFree ~3 GiB while MemAvailable still reads 6+).
#   * Before stopping the container it archives `docker logs --tail 3000`,
#     then `docker stop -t $GRACE` (SIGTERM; a SIGKILL leaks the container's
#     POSIX shm onto the host /dev/shm because of --ipc host).

memwatch_start() {  # consumes CONTAINER, MEMWATCH_*; logs under logs/
    local repo_dir memwatch_log
    repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    memwatch_log="${MEMWATCH_LOG:-$repo_dir/logs/memwatch-${CONTAINER}.log}"
    mkdir -p "$repo_dir/logs/archive"

    # Anchor to the memwatch binary path so we kill a running memwatch but never
    # our own argv (which contains "memwatch.sh <container>"); the [m] class
    # stops the pattern from matching the very pkill/-f command line we run
    # (start-memwatch.sh:22-25).
    pkill -f "[m]emwatch.sh $CONTAINER" 2>/dev/null || true

    [[ -f "$repo_dir/files/memwatch.sh" ]] || {
        warn "files/memwatch.sh absent — watchdog NOT started."
        warn "     Copy it from the single-spark sister repo before serving on a Spark;"
        warn "     HOST_RESERVE_GIB alone cannot catch a GiB/s pool collapse."
        return 0
    }

    # Knobs pass EXPLICITLY (start-memwatch.sh:27-28): recipes/peers set them as
    # shell vars, which nohup would not hand to the child's environment.
    MEMWATCH_MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB:-2}" \
    MEMWATCH_FREE_GATE_GIB="${MEMWATCH_FREE_GATE_GIB:-10}" \
    MEMWATCH_GRACE="${MEMWATCH_GRACE:-30}" MEMWATCH_LOG="$memwatch_log" \
    nohup bash "$repo_dir/files/memwatch.sh" "$CONTAINER" "${MEMWATCH_MIN_GIB:-6}" \
        > "$memwatch_log" 2>&1 &
    ok "Watchdog running (stops $CONTAINER if MemAvailable < ${MEMWATCH_MIN_GIB:-6} GiB): $memwatch_log"
}
