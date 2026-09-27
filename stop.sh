#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# stop.sh — the ONE teardown. Zero flags required. Stops whatever container the
# active recipe names (env RECIPE / recipes state), its memwatch, and evicts
# the checkpoint's clean page cache so the next boot doesn't CUDA-OOM on an
# "idle" box (files/evict_page_cache.py; lane finding #35/#61).
#
# Graceful by default (single-spark stop.sh:6-9): vLLM gets SIGTERM + a chance
# to unlink the POSIX shm segments the PLE offload handshake allocates. With
# --ipc host, a SIGKILL leaks them onto /dev/shm until reboot. --force skips.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }

FORCE=false; [[ "${1:-}" == --force ]] && FORCE=true
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"   # docker escalates to SIGKILL after this
[[ "$STOP_TIMEOUT" =~ ^[0-9]+$ ]] || { echo "STOP_TIMEOUT must be an integer (got '$STOP_TIMEOUT')"; exit 1; }

# Which containers: the recipe(s)' CONTAINER names, always including prod.
RECIPE="${RECIPE:-prod}"
containers=()
for r in "recipes/$RECIPE.conf" recipes/prod.conf recipes/peers.conf; do
    [[ -f "$r" ]] || continue
    c=$(sed -n 's/^CONTAINER="\{0,1\}\([A-Za-z0-9_.-]*\)"\{0,1\}.*/\1/p' "$r" | tail -1)
    [[ -n "$c" ]] && containers+=("$c")
done
# Dedupe.
if (( ${#containers[@]} )); then
    mapfile -t containers < <(printf '%s\n' "${containers[@]}" | sort -u)
else
    containers=(vllm-fn-prod)   # recipes vanished: stop the name we launched
fi

for CONTAINER in "${containers[@]}"; do
    # Watchdog first so it cannot race a slow graceful stop into a kill
    # (single-spark stop.sh:48-53). [m]emwatch anchor never matches our argv.
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
done

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
    # Evict every cached snapshot tree; harmless no-op when nothing is loaded.
    mapfile -t _ckpt < <(find "$HF_CACHE_DIR/hub" -maxdepth 1 -type d -name 'models--*' 2>/dev/null)
    for d in "${_ckpt[@]:-}"; do
        [[ -n "$d" ]] && python3 "$EVICT" "$d" >/dev/null 2>&1 || true
    done
    info "page cache evicted for the HF checkpoints."
else
    warn "files/evict_page_cache.py absent — skipped (copy it; see files/NOTES.md)."
fi
ok "done. On the peer node of a dual pair, run ./stop.sh there too."
