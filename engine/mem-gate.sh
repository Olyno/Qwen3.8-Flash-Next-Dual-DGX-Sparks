#!/usr/bin/env bash
# ============================================================================
# engine/mem-gate.sh — Pre-launch memory gate.
#
# After a vLLM container dies, the Spark shows phantom CUDA OOM for 30-60s
# while memory settles; relaunching too fast kills the boot (myllmbox
# single-Spark recipe). Wait until the effective memory guard (see below) is
# at least LAUNCH_MIN_MEM_AVAIL_GIB
# (default 97 GiB on this 121 GiB box — a settled idle host tops out at
# ~97-98 GiB MemAvailable once baseline services creep in; 100 never clears
# again) before the container starts. Already
# above the threshold = zero cost; otherwise poll /proc/meminfo every 5s and
# hard-fail after LAUNCH_MEM_WAIT_TIMEOUT_S (default 600s).
#
# The threshold is checked against the EFFECTIVE memory guard, not raw
# MemAvailable: min(MemAvailable, cgroup v2 headroom), where headroom =
# memory.max - memory.current + inactive_file (reclaimable page cache counts
# toward the cgroup's usable memory the same way it counts toward
# MemAvailable host-wide). Inside a capped container MemAvailable lies — it
# reports the host's slack, not the slice the launch actually has. Shape:
# fodelf/YoungAi (MIT). One line is printed at startup:
#   memory guard: <N> GiB (source: meminfo|cgroup|min)
# "meminfo" = no cgroup v2 limit applies to this process; "cgroup" = only the
# cgroup numbers were readable; "min" = both, the smaller one gates.
#
# Env overrides:
#   LAUNCH_MIN_MEM_AVAIL_GIB   threshold in GiB (default 97)
#   LAUNCH_MEM_WAIT_TIMEOUT_S  give-up deadline in seconds (default 600)
#   LAUNCH_MEM_POLL_S          poll interval in seconds (default 5)
#   LAUNCH_MEMINFO             meminfo path (default /proc/meminfo; tests only)
#   LAUNCH_CGROUP_DIR          cgroup v2 dir (default: this process's; tests only)
# ============================================================================
set -euo pipefail

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

MIN_GIB="${LAUNCH_MIN_MEM_AVAIL_GIB:-97}"
TIMEOUT_S="${LAUNCH_MEM_WAIT_TIMEOUT_S:-600}"
POLL_S="${LAUNCH_MEM_POLL_S:-5}"
MEMINFO="${LAUNCH_MEMINFO:-/proc/meminfo}"
[[ "$MIN_GIB" =~ ^[0-9]+$ ]] || err "LAUNCH_MIN_MEM_AVAIL_GIB must be a non-negative integer (got: '$MIN_GIB')"

avail_gib() { awk '/^MemAvailable:/ {print int($2 / 1024 / 1024)}' "$MEMINFO"; }

# Headroom of a cgroup v2 limit that applies to this process, in GiB (>=0);
# returns 1 when no hard limit is in effect or the files can't be read.
cgroup_headroom_gib() {
    local dir="$CGROUP_DIR" max cur ifile v
    [[ -n "$dir" && -d "$dir" ]] || return 1
    max=$(cat "$dir/memory.max" 2>/dev/null) || return 1
    [[ "$max" =~ ^[0-9]+$ ]] || return 1   # "max" = unlimited
    cur=$(cat "$dir/memory.current" 2>/dev/null) || return 1
    [[ "$cur" =~ ^[0-9]+$ ]] || return 1
    ifile=$(awk '$1=="inactive_file" {print $2; exit}' "$dir/memory.stat" 2>/dev/null)
    [[ "$ifile" =~ ^[0-9]+$ ]] || ifile=0
    v=$(( (max - cur + ifile) / 1073741824 ))
    (( v < 0 )) && v=0
    echo "$v"
}

# Effective guard into GUARD_GIB / GUARD_SRC.
guard_gib() {
    local av cg
    av=$(avail_gib); cg=$(cgroup_headroom_gib) || cg=""
    if [[ -z "$cg" ]]; then GUARD_GIB=$av;  GUARD_SRC=meminfo
    elif [[ -z "$av" ]];   then GUARD_GIB=$cg; GUARD_SRC=cgroup
    else GUARD_GIB=$(( av < cg ? av : cg )); GUARD_SRC=min
    fi
}

# cgroup v2 dir of the caller (start.sh), unless overridden for tests.
CGROUP_DIR="${LAUNCH_CGROUP_DIR:-}"
if [[ -z "$CGROUP_DIR" ]]; then
    cg_self=$(awk -F'::' '$1=="0" {print $2; exit}' /proc/self/cgroup 2>/dev/null)
    [[ "$cg_self" == "/" ]] && cg_self=""
    CGROUP_DIR="/sys/fs/cgroup${cg_self}"
fi

guard_gib; have=$GUARD_GIB
ok "memory guard: ${have} GiB (source: $GUARD_SRC)"
if (( have >= MIN_GIB )); then
    ok "memory guard ${have} GiB >= ${MIN_GIB} GiB — launching"
    exit 0
fi

info "Waiting for ${MIN_GIB} GiB memory guard before launch (have ${have} GiB, timeout ${TIMEOUT_S}s)..."
deadline=$(( $(date +%s) + TIMEOUT_S ))
while (( have < MIN_GIB )); do
    if (( $(date +%s) > deadline )); then
        err "memory guard still ${have} GiB (< ${MIN_GIB} GiB) after ${TIMEOUT_S}s — memory did not settle, not launching. Retry later or lower LAUNCH_MIN_MEM_AVAIL_GIB."
    fi
    sleep "$POLL_S"
    guard_gib; have=$GUARD_GIB
    info "  memory guard ${have} GiB / ${MIN_GIB} GiB"
done
ok "memory guard settled at ${have} GiB — launching"
