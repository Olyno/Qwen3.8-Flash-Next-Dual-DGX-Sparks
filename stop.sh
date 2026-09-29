#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# stop.sh — the ONE teardown. Zero flags required. On a DUAL pair (pair .env
# present) it stops the WORKER first, then the head — same ergonomics as the
# old dual stop.sh. On a single box it stops the local containers. Stops
# whatever container the active recipe names (env RECIPE / recipes state),
# memwatch, and evicts the checkpoint's clean page cache so the next boot
# doesn't CUDA-OOM on an "idle" box (files/evict_page_cache.py; lane #35/#61).
#
# Graceful by default: vLLM gets SIGTERM + a chance to unlink the POSIX shm
# segments the PLE offload handshake allocates. With --ipc host, a SIGKILL
# leaks them onto /dev/shm until reboot. --force skips.
RUN_WORKER="${RUN_WORKER:-1}"   # 0 = local-only even with a pair .env
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }

FORCE=false; [[ "${1:-}" == --force ]] && FORCE=true
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"   # docker escalates to SIGKILL after this
[[ "$STOP_TIMEOUT" =~ ^[0-9]+$ ]] || { echo "STOP_TIMEOUT must be an integer (got '$STOP_TIMEOUT')"; exit 1; }

# .env carries the pair identity (WORKER_IP/WORKER_USER) + container naming.
if [[ -f .env ]]; then set -a; source .env; set +a; fi
CONTAINER="${CONTAINER:-vllm-fn}"   # the historical pair name; prod default in recipes

# Which containers: the recipe(s)' CONTAINER names, the pair default, always
# including prod.
RECIPE="${RECIPE:-prod}"
containers=("$CONTAINER")
for r in "recipes/$RECIPE.conf" recipes/prod.conf recipes/peers.conf; do
    [[ -f "$r" ]] || continue
    c=$(sed -n 's/^CONTAINER="\{0,1\}\([A-Za-z0-9_.-]*\)"\{0,1\}.*/\1/p' "$r" | tail -1)
    [[ -n "$c" ]] && containers+=("$c")
done
# Dedupe.
if (( ${#containers[@]} )); then
    mapfile -t containers < <(printf '%s\n' "${containers[@]}" | sort -u)
else
    containers=(vllm-fn)   # recipes vanished: stop the pair default name
fi

stop_local() {
    local CONTAINER="$1"
    # Watchdog first so it cannot race a slow graceful stop into a kill.
    if pkill -f "[m]emwatch.sh $CONTAINER" 2>/dev/null; then info "$CONTAINER: watchdog stopped"; fi
    if [[ -n "$(docker ps -aq -f "name=^${CONTAINER}$" 2>/dev/null)" ]]; then
        if $FORCE; then
            docker kill "$CONTAINER" >/dev/null 2>&1 || true; ok "$CONTAINER: killed (--force)"
        else
            docker stop -t "$STOP_TIMEOUT" "$CONTAINER" >/dev/null 2>&1 \
                || docker kill "$CONTAINER" >/dev/null 2>&1 || true
            ok "$CONTAINER: stopped (graceful, ${STOP_TIMEOUT}s)"
        fi
        docker rm "$CONTAINER" >/dev/null 2>&1 || true
    else
        info "$CONTAINER: not present."
    fi
}

stop_remote() {  # same teardown on the worker box, over ssh
    local CONTAINER="$1"
    ssh -o BatchMode=yes -o ConnectTimeout=8 "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}" \
        "pkill -f '[m]emwatch.sh $CONTAINER' 2>/dev/null; docker rm -f $CONTAINER 2>/dev/null >/dev/null" \
        && ok "$CONTAINER: stopped on worker ($WORKER_IP)" \
        || warn "$CONTAINER: worker ssh/docker rm failed (run ./stop.sh there yourself, or the container was already gone)"
}

# DUAL: worker first, then head — mirrors the old dual stop.sh order.
if [[ -n "${WORKER_IP:-}" && "$RUN_WORKER" == 1 ]]; then
    for CONTAINER in "${containers[@]}"; do stop_remote "$CONTAINER"; done
fi
for CONTAINER in "${containers[@]}"; do stop_local "$CONTAINER"; done

# Report leaked shm, never delete it: other --ipc host containers live here too
# (single-spark stop.sh:92-94).
leaked=$(find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'sem.mp-*' \) 2>/dev/null | wc -l)
(( leaked > 0 )) && warn "$leaked psm_*/sem.mp-* segments still in /dev/shm (another --ipc host container, or a kill before unlink)."

# Release the checkpoint's clean pages for the NEXT boot. Needs no root
# (posix_fadvise DONTNEED, read-only opens); best-effort by design — the
# script itself never fails over cache eviction.
EVICT="$SCRIPT_DIR/files/evict_page_cache.py"
if [[ -f "$EVICT" ]]; then
    HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME:-$HOME/.cache/huggingface}}"
    # Evict (a) every cached HF snapshot tree and (b) every local MODEL_PATH a
    # recipe names (bench arms live outside the hub cache). Harmless no-op when
    # nothing is loaded.
    evict_dirs=()
    while read -r p; do [[ -n "$p" && -d "$p" ]] && evict_dirs+=("$p"); done \
        < <(sed -n 's/^MODEL_PATH="\{0,1\}\$HOME\([^"#]*\)"\{0,1\}.*/'"$HOME"'\1/p; s/^MODEL_PATH="\{0,1\}\(\/[^"#]*\)"\{0,1\}.*/\1/p' recipes/*.conf 2>/dev/null | sort -u)
    mapfile -t -O ${#evict_dirs[@]} _ckpt < <(find "$HF_CACHE_DIR/hub" -maxdepth 1 -type d -name 'models--*' 2>/dev/null)
    for d in "${evict_dirs[@]}" "${_ckpt[@]:-}"; do
        [[ -n "$d" ]] && python3 "$EVICT" "$d" >/dev/null 2>&1 || true
    done
    info "page cache evicted for checkpoints + HF cache."
else
    warn "files/evict_page_cache.py absent — skipped (copy it; see files/NOTES.md)."
fi
ok "done.$( [[ -n "${WORKER_IP:-}" && "$RUN_WORKER" == 1 ]] && echo " (worker $WORKER_IP stopped too)" )"
