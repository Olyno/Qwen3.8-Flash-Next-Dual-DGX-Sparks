#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# start.sh — the ONE launcher. Zero flags required. On a single DGX Spark it
# serves TP=1; run the same file on both nodes of a dual-Spark pair and it
# starts the TP2+EP job (rank 0 = .1 = serves HTTP, rank 1 = .2 = headless).
# Topology comes from engine/detect.sh: a 192.168.100.0/30 address on the 200G
# link = dual. Recipe = recipes/prod.conf unless RECIPE=x. PLE defaults to the
# v0.30 mmap overlay (engine/ple.sh); pinned offload is REFUSED on <=128G pools.
#
# DUAL PAIR: run this SAME script on the head only; it SSHes to WORKER_IP
# (.env or peers.conf) and launches the worker container itself, then serves
# rank 0. The worker needs nothing but ssh + docker + the checkpoint copy.
# Order: worker up first (15 s init), then head; single box: just the head.
# Patches (mmap-PLE/fp8-KV/draft-vocab/FP8DENSE) are PREPARED on the head and
# rsync'd to the worker's $SCRIPT_DIR so both ranks bind identical patched
# sources — the same directory layout on both nodes is the contract.
#
# Usage:  ./start.sh                                  # prod; pair config in .env
#         RECIPE=arm-a4 ./start.sh                    # bench arm
#         GPU_MEMORY_UTILIZATION=0.7 ./start.sh       # pin the budget (warned)
#         RUN_WORKER=0 ./start.sh                     # head-only, worker by hand
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*" >&2; exit 1; }
# --- .env FIRST: it carries the pair identity (HEAD_IP/WORKER_IP/IFACE/...) --
if [[ -f .env ]]; then
    # shellcheck disable=SC1091
    set -a; source .env; set +a
fi

# shellcheck source=engine/detect.sh
source engine/detect.sh
detect_topology                       # TOPO_MODE NODE_RANK HEAD_IP NODE_IP MEM_*
# head-orchestrated re-invocation: trust the passed rank (pair .env is
# rsync'd, so detection should agree; this pins it deterministically).
[[ -n "${NODE_RANK_OVERRIDE:-}" ]] && { TOPO_MODE=dual; NODE_RANK="$NODE_RANK_OVERRIDE"; }
if [[ "${TOPO_MODE:-}" == dual && "$NODE_RANK" == 1 && -z "${NODE_IP:-}" ]]; then
    # belt for the head-orchestrated re-invocation: .env OR the exported
    # HEAD_IP/WORKER_IP (launch_worker passes both) make detect path-1 set
    # NODE_IP; if neither reached us, take our fabric address from the route
    # to the head rather than launching rank 1 on 127.0.0.1.
    _r=$($DETECT_IP_CMD -4 route get "$HEAD_IP" 2>/dev/null | head -1)
    NODE_IP=$(sed -n 's/.* src \([0-9.]*\).*/\1/p' <<<"$_r")
    FABRIC_IFACE=${FABRIC_IFACE:-$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$_r")}
    [[ -n "$NODE_IP" ]] || err "rank 1 with no fabric address and no route to $HEAD_IP — check the 200G link."
fi
TP_SIZE=1; [[ "$TOPO_MODE" == dual ]] && TP_SIZE=2
# --- recipe: env > recipes/$RECIPE.conf > recipes/peers.conf > engine default --
_rc="recipes/$RECIPE.conf"; [[ -f "$_rc" ]] || err "recipe not found: $_rc"
_peers=(); if [[ -f recipes/peers.conf ]]; then _peers=(recipes/peers.conf); fi
_keys=$(sed -n 's/^\([A-Z][A-Z0-9_]*\)=.*/\1/p' "$_rc" ${_peers[@]+"${_peers[@]}"} | sort -u)
declare -A _env=(); while read -r k; do if [[ -n "${!k:-}" ]]; then _env[$k]="${!k}"; fi; done <<<"$_keys"
# shellcheck disable=SC1090
source "$_rc"; if [[ -f recipes/peers.conf ]]; then source recipes/peers.conf; fi
for k in "${!_env[@]}"; do export "$k=${_env[$k]}"; done
: "${MODEL_ID:?recipe sets MODEL_ID}"; : "${IMAGE:?recipe sets IMAGE (@sha256-pinned)}"
CONTAINER="${CONTAINER:-vllm-fn}"; PORT="${PORT:-8888}"; MASTER_PORT="${MASTER_PORT:-50000}"

# --- YaRN (long-context rope scaling) — merged into HF_OVERRIDES text_config --
# The monolith (:845-866) merged rope_parameters INSIDE text_config because
# vLLM's _apply_dict_overrides only recurses into nested config keys: a
# top-level rope_parameters was silently a no-op (issue class preserved here).
# Force-disabled at <=262144 (native: YaRN would degrade quality for nothing).
if [[ "${YARN:-0}" == 1 ]]; then
    if [[ "${MAX_MODEL_LEN:-262144}" -le 262144 ]]; then
        warn "YARN=1 but MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144} <= native 262144 — force-disabled."
    else
        _YF="${YARN_FACTOR:?YARN=1 needs YARN_FACTOR (e.g. 4.0 for 262144x4=1M)}"
        HF_OVERRIDES=$(YARN_F="$_YF" python3 -c "
import json, os
ov = json.loads(os.environ.get('HF_OVERRIDES') or '{}')
tc = ov.setdefault('text_config', {})
tc['rope_parameters'] = {'rope_type': 'yarn', 'factor': float(os.environ['YARN_F']),
                         'original_max_position_embeddings': 262144}
print(json.dumps(ov, separators=(',', ':')))")
        VLLM_ALLOW_LONG_MAX_MODEL_LEN=1   # required above 262144
        info "YaRN: factor $_YF -> ${MAX_MODEL_LEN} ctx (rope_parameters merged into text_config)"
    fi
fi


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
[[ -n "${MODEL_SNAPSHOT_OVERRIDE:-}" ]] && MODEL_SNAPSHOT="$MODEL_SNAPSHOT_OVERRIDE"

# --- preflight: disk headroom, ports free (head-side only), stale container --
_free_gib=$(df -BG --output=avail "$HOME/.cache/vllm" 2>/dev/null | tail -1 | tr -dc '0-9')
[[ "${_free_gib:-0}" -ge 60 ]] || err "only ${_free_gib:-?} GiB free under ~/.cache/vllm; the mmap PLE table needs ~50 + engine cache."
if [[ "$TOPO_MODE" == single || "$NODE_RANK" == 0 ]]; then
    for _p in "$PORT" "$MASTER_PORT"; do
        ss -ltn "sport = :$_p" 2>/dev/null | grep -q LISTEN \
            && err "port $_p is already taken (ss -ltnp shows who). Stop the old job: ./stop.sh"
    done
    docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER" \
        && err "container $CONTAINER is running; ./stop.sh first (or RECIPE=y for another arm)."
fi

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
# --- dual pair: head orchestrates the worker (old start.sh ergonomics) -------
# RUN_WORKER=1 (dual default, head rank): prove ssh + docker + image on the
# worker, sync THIS repo dir (same absolute $SCRIPT_DIR is the contract — the
# worker re-invocation sources the same recipes/), sync the model snapshot at
# the SAME absolute path (parent dir created on the worker, snapshot copied
# with hardlinks preserved: one rsync handles both local MODEL_PATH dirs and
# HF hub snapshot trees), then re-invoke start.sh remotely with the resolved
# launch state exported + RUN_WORKER=0 NODE_RANK_OVERRIDE=1. The worker re-
launch_worker() {
    WORKER_SSH="${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}"
    info "=== worker $WORKER_SSH: preflight ==="
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$WORKER_SSH" true 2>/dev/null \
        || err "ssh to $WORKER_SSH failed (non-interactive). Fix key auth (ssh-copy-id) or set WORKER_USER; head-only: RUN_WORKER=0 and run ./start.sh on the worker yourself."
    docker image inspect "$IMAGE" >/dev/null 2>&1 \
        || err "image $IMAGE not on the head. Single-node pull first: docker pull ${IMAGE_HINT:-$IMAGE}"
    ssh "$WORKER_SSH" "docker image inspect '$IMAGE' >/dev/null 2>&1" \
        || { info "  image missing on worker — docker save | ssh load (~2 min at 200G, one-time)"; \
             docker save "$IMAGE" | ssh "$WORKER_SSH" docker load || err "image ship failed"; }
    if ! ssh "$WORKER_SSH" "test -f '$MODEL_SNAPSHOT/config.json'"; then
        if [[ "${SYNC_WEIGHTS:-1}" == 1 ]]; then
            info "  model missing on worker — rsyncing $MODEL_SNAPSHOT (hardlinks preserved)..."
            ssh "$WORKER_SSH" "mkdir -p '$(dirname "$MODEL_SNAPSHOT")'" \
                || err "cannot create parent dir on worker"
            rsync -aH "$MODEL_SNAPSHOT/" "${WORKER_SSH}:$MODEL_SNAPSHOT/" \
                || err "weight rsync failed; worker disk: ssh $WORKER_SSH df -h"
        else
            err "$MODEL_SNAPSHOT has no config.json on $WORKER_IP (SYNC_WEIGHTS=0). Put it there yourself, same path."
        fi
    fi
    info "  syncing repo dir to $SCRIPT_DIR ..."
    ssh "$WORKER_SSH" "mkdir -p '$SCRIPT_DIR'"
    rsync -aic --exclude 'logs/' --exclude 'v30_bench/' --exclude '*.pyc' --exclude '.git/' \
        "$SCRIPT_DIR/" "${WORKER_SSH}:$SCRIPT_DIR/" >/dev/null || err "repo rsync failed"
    # mmap-PLE table: per-rank files under $PLE_MMAP_HOST (etp<rank> suffix) —
    # pre-create the dir on the worker so the two builders cannot race the
    # fingerprint sidecars; each rank builds lazily on first boot.
    [[ "${PLE_MODE:-mmap}" == mmap ]] && ssh "$WORKER_SSH" "mkdir -p '$PLE_MMAP_HOST'" || true
    if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
        err "container $CONTAINER is running on the head; ./stop.sh first (it stops both)."
    fi
    info "=== launching worker container (rank 1) on $WORKER_IP ==="
    # BUG CLASS (gx10 field failure 09-28): bare `ssh host env VAR=val ...`
    # hands the remote shell a re-tokenized command line — any value with a
    # space (VLLM_ARGS, SPECULATIVE_CONFIG/HF_OVERRIDES JSON) split `env` off
    # its assignments and it tried to EXECUTE the next word. Fix: build the
    # whole worker launch as a %q-quoted bash script HERE, ship it base64
    # (alphabet has no shell metachars), and pipe it into the remote bash.
    _wscript=$(
        printf 'export RECIPE=%q RUN_WORKER=0 NODE_RANK_OVERRIDE=1\n' "$RECIPE"
        printf 'export CONTAINER=%q PORT=%q MASTER_PORT=%q\n' "$CONTAINER" "$PORT" "$MASTER_PORT"
        printf 'export MODEL_PATH=%q MODEL_ID=%q HF_CACHE_DIR=%q IMAGE=%q\n' "${MODEL_PATH:-}" "$MODEL_ID" "$HF_CACHE_DIR" "$IMAGE"
        printf 'export KV_CACHE_DTYPE=%q MAX_MODEL_LEN=%q\n' "${KV_CACHE_DTYPE:-auto}" "${MAX_MODEL_LEN:-}"
        printf 'export MTP_NUM_SPECULATIVE_TOKENS=%q SPECULATIVE_CONFIG=%q\n' "${MTP_NUM_SPECULATIVE_TOKENS:-0}" "${SPECULATIVE_CONFIG:-}"
        printf 'export GPU_MEMORY_UTILIZATION=%q CONTAINER_MEM_GIB=%q\n' "$GPU_MEMORY_UTILIZATION" "$CONTAINER_MEM_GIB"
        printf 'export PLE_MODE=%q FP8DENSE=%q\n' "${PLE_MODE:-mmap}" "${FP8DENSE:-0}"
        printf 'export FABRIC_IFNAME=%q IB_HCA=%q IB_GID_INDEX=%q HEAD_IP=%q WORKER_IP=%q\n' "${FABRIC_IFNAME:-}" "${IB_HCA:-}" "${IB_GID_INDEX:-}" "$HEAD_IP" "$WORKER_IP"
        printf 'export VLLM_ALLOW_LONG_MAX_MODEL_LEN=%q HF_OVERRIDES=%q COMPILATION_CONFIG=%q\n' "${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-}" "${HF_OVERRIDES:-}" "${COMPILATION_CONFIG:-}"
        printf 'export YARN=%q YARN_FACTOR=%q\n' "${YARN:-0}" "${YARN_FACTOR:-}"
        printf 'export VLLM_ARGS=%q EXTRA_VLLM_ARGS=%q DOCKER_ARGS_EXTRA=%q SERVED_MODEL_NAME=%q\n' "$VLLM_ARGS" "${EXTRA_VLLM_ARGS:-}" "${DOCKER_ARGS_EXTRA:-}" "${SERVED_MODEL_NAME:-}"
        printf 'export MEMWATCH_MIN_GIB=%q MEMWATCH_MIN_FREE_GIB=%q\n' "${MEMWATCH_MIN_GIB:-}" "${MEMWATCH_MIN_FREE_GIB:-}"
        printf 'cd %q && exec bash ./start.sh\n' "$SCRIPT_DIR"
    )
    _wb64=$(printf '%s' "$_wscript" | base64 -w0)
    # shellcheck disable=SC2029
    ssh "$WORKER_SSH" "printf %s '$_wb64' | base64 -d | bash" \
        || err "worker launch failed — inspect: ssh $WORKER_SSH docker logs $CONTAINER (if the container never appeared, the worker's start.sh printed the reason to this terminal's ssh stream above)"
    ok "worker container up; waiting 15 s to initialize..."
    sleep 15
}
# image present locally (the docker run below binds $IMAGE)
docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || err "image $IMAGE not on this host. Single: docker pull ${IMAGE_HINT:-$IMAGE}; pair: the head ships it to the worker itself (this run will, but the head needs it first)"

# RUN_WORKER=0 is the worker role: orchestration + head-side staleness skip.
if [[ "$TOPO_MODE" == dual && "${RUN_WORKER:-1}" == 1 && "$NODE_RANK" == 0 ]]; then
    launch_worker
fi

# --- docker run ----------------------------------------------------------------
read -r -a _vllm <<<"$VLLM_ARGS"
# Caches mount to /root (container runs as root, monolith :900-901).
# ~/.cache/vllm rw wholesale: the mmap table dir + engine cache ride it.
args=(docker run -d --name "$CONTAINER" --gpus all --network host --ipc host
      --cap-add SYS_NICE --cap-add SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/root/.cache/huggingface
      -v "$HF_CACHE_DIR:/root/.cache/huggingface" -v "$HOME/.cache/vllm:/root/.cache/vllm"
      # The model MUST exist INSIDE the container at the same absolute path
      # (monolith launch_v30.sh :20 -v "$MODEL:$MODEL:ro"). Without it vLLM
      # treats the nonexistent path as an HF repo id -> HFValidationError.
      -v "$MODEL_SNAPSHOT:$MODEL_SNAPSHOT:ro"
      -e VLLM_HOST_IP="${NODE_IP:-127.0.0.1}" "${PLE_OFFLOAD_ENV[@]}" "${PLE_MOUNT[@]}" "${QSA_MOUNT[@]}")
[[ -n "${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-}" ]] && args+=(-e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=$VLLM_ALLOW_LONG_MAX_MODEL_LEN")
if [[ "$TOPO_MODE" == dual ]]; then
    # NCCL/RoCE over the fabric iface (detection: .env pair members, else /30).
    # HCA/GID per the dual repo .env convention (=rocep1s0f0, GID 3), override.
    args+=(-e GLOO_SOCKET_IFNAME="${FABRIC_IFNAME:-${IFACE:-$FABRIC_IFACE}}"
           -e NCCL_SOCKET_IFNAME="${FABRIC_IFNAME:-${IFACE:-$FABRIC_IFACE}}"
           -e TP_SOCKET_IFNAME="${FABRIC_IFNAME:-${IFACE:-$FABRIC_IFACE}}"
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
