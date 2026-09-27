#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# start.sh — the ONE launcher. Zero flags required. On a single DGX Spark it
# serves TP=1; run the same file on both nodes of a dual-Spark pair and it
# starts the TP2+EP job (rank 0 = .1 = serves HTTP, rank 1 = .2 = headless).
# Topology comes from engine/detect.sh: a 192.168.100.0/30 address on the 200G
# link = dual. Recipe = recipes/prod.conf unless RECIPE=x. PLE defaults to the
# v0.30 mmap overlay (engine/ple.sh); pinned offload is REFUSED on <=128G pools.
#
# Usage:  ./start.sh                                  # prod, auto-detect
#         RECIPE=arm-a4 ./start.sh                    # bench arm
#         GPU_MEMORY_UTILIZATION=0.7 ./start.sh       # pin the budget (warned)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*" >&2; exit 1; }

# shellcheck source=engine/detect.sh
source engine/detect.sh
detect_topology                                   # TOPO_MODE NODE_RANK HEAD_IP NODE_IP MEM_*
TP_SIZE=1; [[ "$TOPO_MODE" == dual ]] && TP_SIZE=2

# --- recipe: env > recipes/$RECIPE.conf > recipes/peers.conf > engine default --
RECIPE="${RECIPE:-prod}"
_rc="recipes/$RECIPE.conf"; [[ -f "$_rc" ]] || err "recipe not found: $_rc"
_peers=(); if [[ -f recipes/peers.conf ]]; then _peers=(recipes/peers.conf); fi
_keys=$(sed -n 's/^\([A-Z][A-Z0-9_]*\)=.*/\1/p' "$_rc" ${_peers[@]+"${_peers[@]}"} | sort -u)
declare -A _env=(); while read -r k; do if [[ -n "${!k:-}" ]]; then _env[$k]="${!k}"; fi; done <<<"$_keys"
# shellcheck disable=SC1090
source "$_rc"; if [[ -f recipes/peers.conf ]]; then source recipes/peers.conf; fi
for k in "${!_env[@]}"; do export "$k=${_env[$k]}"; done
: "${MODEL_ID:?recipe sets MODEL_ID}"; : "${IMAGE:?recipe sets IMAGE (@sha256-pinned)}"
CONTAINER="${CONTAINER:-vllm-fn-$RECIPE}"; PORT="${PORT:-8888}"; MASTER_PORT="${MASTER_PORT:-50000}"

# --- resolve checkpoint (MODEL_PATH=local dir wins; else HF cache by ID) -----
# Our bench arms are LOCAL checkpoints (~/models/q38-lean-hyb), not hub repos:
# set MODEL_PATH in the recipe. HF-hub resolution stays for the published
# nvidia/ checkpoints (prod on a fresh box).
HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME:-$HOME/.cache/huggingface}}"
if [[ -n "${MODEL_PATH:-}" ]]; then
    [[ -f "$MODEL_PATH/config.json" ]] || err "MODEL_PATH=$MODEL_PATH has no config.json"
    MODEL_SNAPSHOT="$(cd "$MODEL_PATH" && pwd)"
else
    ORG="${MODEL_ID%%/*}"; NAME="${MODEL_ID##*/}"
    MODEL_ROOT="$HF_CACHE_DIR/hub/models--${ORG}--${NAME}"
    [[ -d "$MODEL_ROOT" ]] || err "checkpoint not in cache: $MODEL_ROOT — ./download.sh $MODEL_ID first, or set MODEL_PATH."
    if [[ -f "$MODEL_ROOT/refs/main" ]]; then MODEL_SNAPSHOT="$MODEL_ROOT/snapshots/$(cat "$MODEL_ROOT/refs/main")"
    else MODEL_SNAPSHOT="$(ls -1dt "$MODEL_ROOT"/snapshots/*/ | head -1)"; fi
    [[ -f "$MODEL_SNAPSHOT/config.json" ]] || err "no config.json under $MODEL_SNAPSHOT"
fi

# --- preflight: disk headroom, ports free, no stale container -----------------
_free_gib=$(df -BG --output=avail "$HOME/.cache/vllm" 2>/dev/null | tail -1 | tr -dc '0-9')
[[ "${_free_gib:-0}" -ge 60 ]] || err "only ${_free_gib:-?} GiB free under ~/.cache/vllm; the mmap PLE table needs ~50 + engine cache."
for _p in "$PORT" "$MASTER_PORT"; do
    ss -ltn "sport = :$_p" 2>/dev/null | grep -q LISTEN \
        && err "port $_p is already taken (ss -ltnp shows who). Stop the old job: ./stop.sh"
done
docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER" \
    && err "container $CONTAINER is running; ./stop.sh first (or RECIPE=y for another arm)."

# --- engine -------------------------------------------------------------------
# shellcheck source=/dev/null
source engine/ple.sh; source engine/budget.sh; source engine/quant-detect.sh; source engine/memwatch.sh
info "=== PLE policy: $TOPO_MODE / ${PLE_MODE:-mmap} ==="
ple_policy                                # PLE_OFFLOAD_ENV; mmap also sets PLE_MMAP_HOST
# The mmap table builds lazily INSIDE the engine on first boot (msync +
# fingerprint; later boots map it straight in) — no build step, no cross-node
# race. What must be prepared per node is only the overlay patch:
if [[ "$PLE_MODE" == mmap ]]; then
    mkdir -p "$PLE_MMAP_HOST"
    ple_prepare_overlay                          # extract pristine ngram_embedding.py + patch
    PLE_MOUNT=(${PLE_OVERLAY[@]+"${PLE_OVERLAY[@]}"})
else
    PLE_MOUNT=()
fi
info "=== Memory budget ($TOPO_MODE, TP=$TP_SIZE) ==="
compute_budget                                     # engine/budget.sh; monolith Step 2
quant_preflight

# fp8dense overlay: hybrid/lean-hyb checkpoints declare FP8_PER_CHANNEL_PER_TOKEN
# on dense+HC layers; v0.30 stock code mishandles the MTP HC mixer + final
# quant dispatch. Recipe FP8DENSE=1 mounts our 3 patched sources over the
# package (stock nvidia/ ckpts: FP8DENSE unset = no mounts).
FP8DENSE_MOUNT=()
if [[ "${FP8DENSE:-0}" == 1 ]]; then
    _pkg=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
    for f in model.py mtp.py hyperconnection.py; do
        [[ -f "$SCRIPT_DIR/files/$f" ]] || err "FP8DENSE=1 needs files/$f (see files/NOTES.md)"
        FP8DENSE_MOUNT+=(-v "$SCRIPT_DIR/files/$f:$_pkg/$f:ro")
    done
fi


# fp8 KV on the QSA path needs the PR #55557 backport (missed the v0.30 cut):
# patch pristine image qsa.py + ops/qsa.py, bind over the package. Without
# this wiring, KV_CACHE_DTYPE=fp8_e4m3 is silently unsupported on qwen4_exp.
QSA_MOUNT=()
if [[ "${KV_CACHE_DTYPE:-auto}" == fp8* ]]; then
    [[ -f files/patch_qsa_fp8_kv_v030.py ]] || err "KV_CACHE_DTYPE=fp8* needs files/patch_qsa_fp8_kv_v030.py"
    QO=files/v030_fp8kv/orig; QP=files/v030_fp8kv
    if [[ ! -f $QO/qsa.py || ! -f $QO/ops/qsa.py ]]; then
        info "Extracting pristine qsa.py + ops/qsa.py from $IMAGE ..."
        CID=$(docker create "$IMAGE" /bin/true)
        mkdir -p "$QO/ops"
        docker cp "$CID:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/qsa.py" "$QO/qsa.py"
        docker cp "$CID:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/qsa.py" "$QO/ops/qsa.py"
        docker rm "$CID" >/dev/null
    fi
    python3 files/patch_qsa_fp8_kv_v030.py "$QO" "$QP" >/dev/null || err "qsa #55557 does not apply (image drift)"
    _pkg=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
    QSA_MOUNT=(-v "$SCRIPT_DIR/$QP/qsa.py:$_pkg/qsa.py:ro" -v "$SCRIPT_DIR/$QP/ops/qsa.py:$_pkg/ops/qsa.py:ro")
fi
# --- docker run ----------------------------------------------------------------
read -r -a _vllm <<<"$VLLM_ARGS"
# Caches mount to /root (container runs as root, monolith :900-901).
# ~/.cache/vllm rw wholesale: the mmap table dir + engine cache ride it.
args=(docker run -d --name "$CONTAINER" --gpus all --network host --ipc host
      --cap-add SYS_NICE --cap-add SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864
      --memory "${CONTAINER_MEM_GIB}g" --memory-swap "${CONTAINER_MEM_GIB}g"
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/root/.cache/huggingface
      -v "$HF_CACHE_DIR:/root/.cache/huggingface" -v "$HOME/.cache/vllm:/root/.cache/vllm"
      -e VLLM_HOST_IP="${NODE_IP:-127.0.0.1}" "${PLE_OFFLOAD_ENV[@]}" "${PLE_MOUNT[@]}" "${QSA_MOUNT[@]}")
if [[ "$TOPO_MODE" == dual ]]; then
    # NCCL/RoCE over the 200G link: iface from detection; HCA/GID per the dual
    # repo's .env.sample convention (=rocep1s0f0, GID 3), recipe-overridable.
    args+=(-e GLOO_SOCKET_IFNAME="${FABRIC_IFNAME:-$FABRIC_IFACE}"
           -e NCCL_SOCKET_IFNAME="${FABRIC_IFNAME:-$FABRIC_IFACE}"
           -e TP_SOCKET_IFNAME="${FABRIC_IFNAME:-$FABRIC_IFACE}"
           -e NCCL_IB_DISABLE=0 -e "NCCL_IB_HCA=${IB_HCA:-=rocep1s0f0}"
           -e "NCCL_IB_GID_INDEX=${IB_GID_INDEX:-3}" -e NCCL_IB_AUTO_DETECT=0
           -e NCCL_DEBUG=WARN --device /dev/infiniband:/dev/infiniband)
    _vllm+=(--tensor-parallel-size 2 --nnodes 2 --master-addr "$HEAD_IP" --master-port "$MASTER_PORT" --enable-expert-parallel)
    if [[ "$NODE_RANK" == 1 ]]; then _vllm+=(--node-rank 1 --headless)
    else _vllm+=(--node-rank 0 --host 0.0.0.0 --port "$PORT"); fi
else
    _vllm+=(--tensor-parallel-size 1 --host 0.0.0.0 --port "$PORT")
fi
_vllm+=(--kv-cache-dtype "${KV_CACHE_DTYPE:-auto}")
[[ -n "${SPECULATIVE_CONFIG:-}" ]] && _vllm+=(--speculative-config "$SPECULATIVE_CONFIG")
_vllm+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
[[ -n "${COMPILATION_CONFIG:-}" ]] && _vllm+=(--compilation-config "$COMPILATION_CONFIG")
[[ -n "${HF_OVERRIDES:-}"       ]] && _vllm+=(--hf-overrides       "$HF_OVERRIDES")
[[ -n "${MAX_MODEL_LEN:-}"      ]] && _vllm+=(--max-model-len      "$MAX_MODEL_LEN")
# shellcheck disable=SC2206
[[ -n "${DOCKER_ARGS_EXTRA:-}"  ]] && args+=(${DOCKER_ARGS_EXTRA})
info "=== Launch: $CONTAINER (recipe $RECIPE, $( if [[ "$TOPO_MODE" == dual ]]; then echo "rank $NODE_RANK"; else echo single; fi )) ==="
# shellcheck disable=SC2086
"${args[@]}" "${FP8DENSE_MOUNT[@]}" "$IMAGE" "$MODEL_SNAPSHOT" "${_vllm[@]}" ${EXTRA_VLLM_ARGS:-}
ok "container started."

memwatch_start
[[ "$TOPO_MODE" == dual && "$NODE_RANK" == 1 ]] && { info "rank 1 is headless; the API lives on $HEAD_IP:$PORT. Done."; exit 0; }
info "following logs until /health answers on :$PORT (Ctrl-C detaches; the container keeps running)"
docker logs -f "$CONTAINER" & _lp=$!
while sleep 10; do
    docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" || err "container exited; check: docker logs $CONTAINER"
    [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null)" == 200 ]] && break
done
kill "$_lp" 2>/dev/null || true
ok "serving http://${HEAD_IP:-localhost}:$PORT/v1 (model: ${SERVED_MODEL_NAME:-qwen3.8-flash-next}); stop: ./stop.sh"
