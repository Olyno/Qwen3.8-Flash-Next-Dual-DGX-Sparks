# ---------------------------------------------------------------------------
# engine/config.sh — .env (machine truth) + recipes/<RECIPE>.yaml (serving
# truth) + CLI flags -> the environment every later step consumes.
#
# .env carries ONLY cluster identity and secrets (HEAD_IP, WORKER_IP, IFACE,
# IB_HCA, IB_GID_INDEX, HF_TOKEN, ABLIT, HF_HOME, ...). Serving config comes
# from the recipe, and a recipe wins over .env for every key it sets, so a
# stale .env value can never silently take effect. The shell environment wins
# over both only for ABLIT and HF_TOKEN.
# ---------------------------------------------------------------------------
_CLI_ABLIT="${ABLIT:-}"
_CLI_HF_TOKEN="${HF_TOKEN:-}"
if [[ ! -f .env ]]; then
    err ".env not found. Copy .env.example to .env and edit it.
  cp .env.example .env"
fi

# shellcheck source=.env
source .env

# Recipe: serving config. Default boot is prod (the local checkpoint).
RECIPE="${RECIPE:-prod}"
RECIPE_ENV="$(python3 "$SCRIPT_DIR/engine/recipe.py" "$SCRIPT_DIR/recipes" "$RECIPE")" \
    || err "failed to load recipe '$RECIPE'"
eval "$RECIPE_ENV"
info "Recipe: recipes/$RECIPE.yaml"

[[ -n "$_CLI_ABLIT" ]] && ABLIT="$_CLI_ABLIT"
ABLIT="${ABLIT:-0}"
[[ "$ABLIT" == "0" || "$ABLIT" == "1" ]] || err "ABLIT must be 0 or 1 (got: '$ABLIT')"
[[ -n "$_CLI_HF_TOKEN" ]] && HF_TOKEN="$_CLI_HF_TOKEN"
HF_TOKEN="${HF_TOKEN:-}"
[[ -n "$HF_TOKEN" ]] && export HF_TOKEN
ABLIT_MODEL_ID="drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-dual-ablit-house-qsa-L3-47"
ABLIT_PAGE="https://huggingface.co/${ABLIT_MODEL_ID}"

# Topology: probe the worker — reachable -> dual-node (TP2 across two Sparks),
# unset or unreachable -> single node. Everything downstream keys off NNODES;
# TP follows it (one GPU per Spark) unless a recipe pins tensor_parallel_size.
WORKER_IP="${WORKER_IP:-}"
WORKER_USER="${WORKER_USER:-}"
NNODES=1
if [[ -n "$WORKER_IP" ]]; then
    if ssh -o BatchMode=yes -o ConnectTimeout=3 -o StrictHostKeyChecking=no \
        "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}" true 2>/dev/null; then
        NNODES=2
    else
        warn "WORKER_IP=$WORKER_IP is set but unreachable — booting single-node."
    fi
fi

# Validate required variables (recipe or .env must provide them)
for var in HEAD_IP MAX_MODEL_LEN GPU_MEMORY_UTILIZATION MAX_NUM_SEQS \
           MAX_NUM_BATCHED_TOKENS PORT IMAGE \
           MASTER_PORT; do
    if [[ -z "${!var:-}" ]]; then
        err "Required variable $var is not set (checked recipes/$RECIPE.yaml and .env)"
    fi
done
if [[ "$NNODES" -eq 2 ]]; then
    for var in IFACE IB_HCA IB_GID_INDEX; do
        if [[ -z "${!var:-}" ]]; then
            err "Required variable $var is not set (needed for dual-node; checked .env)"
        fi
    done
fi

# Numeric sanity: the YaRN guard below does an arithmetic comparison on MAX_MODEL_LEN
[[ "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]] || err "MAX_MODEL_LEN must be a positive integer (got: '$MAX_MODEL_LEN')"
# Per-node overrides — the two nodes may be cross-wired (head port f1 ↔ worker port f0),
# so the connected interface/HCA can have different names on each node.
WORKER_IFACE="${WORKER_IFACE:-${IFACE:-}}"
WORKER_IB_HCA="${WORKER_IB_HCA:-${IB_HCA:-}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$NNODES}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-true}"
MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-3}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"   # fp8 needs patches/patch_qsa_fp8_kv.py, applied automatically in step 4f; auto = bf16
# dtype of the GDN recurrent (SSM) state. The checkpoint asks for float32; the
# fused GDN kernel also accepts bfloat16 (FUSED_GDN_STATE_DTYPES in
# qwen_gdn_linear_attn.py). BF16 halves the ~0.23 GB per sequence the state
# costs to read and write every step, and halves the mamba page, which lets
# vLLM pick a smaller attention block. Empty keeps the checkpoint's float32.
MAMBA_SSM_CACHE_DTYPE="${MAMBA_SSM_CACHE_DTYPE:-}"
PLE_OFFLOAD="${PLE_OFFLOAD:-false}"
# Vision MLP intermediate_size=4304 is not divisible by 16 after TP split (4304/2=2152).
# NVFP4 kernels require input features % 16 == 0, so replicate the encoder on each GPU.
MM_ENCODER_TP_MODE="${MM_ENCODER_TP_MODE:-data}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
# Weight distribution. false (default) = each node keeps its own copy of the
# checkpoint, worker seeded by rsync from the head. true = head exports its
# cache over NFS on ConnectX and the worker mounts it read-only (HF models only).
NFS_SHARE="${NFS_SHARE:-false}"
# Optional: head ConnectX address used as the NFS server (auto-detected from IFACE).
NFS_SERVER_IP="${NFS_SERVER_IP:-}"
V030="${V030:-false}"
SKIP_PLE_PATCH="${SKIP_PLE_PATCH:-false}"
# FP8-dense hybrid checkpoint (NVFP4 experts + FP8 per-channel dense projections,
# built by scripts/fp8dense/make_fp8_dense_checkpoint.py). Needs the vLLM overlay
# patches in overlays/fp8dense (bind-mounted, no image rebuild).
FP8_DENSE="${FP8_DENSE:-false}"
FP8_DENSE_MODEL_ID="${FP8_DENSE_MODEL_ID:-MiaAI-Lab/Qwen3.8-Flash-Next-NVFP4-FP8dense}"
if [[ "$FP8_DENSE" == "true" ]]; then
    MODEL_ID="$FP8_DENSE_MODEL_ID"
    DO_DOWNLOAD_DEFAULT=false   # local-only checkpoint, never on the Hub
fi

# Local checkpoint (recipe model_path): served via bind-mount at /model,
# never downloaded. MODEL_ID becomes a display name only.
MODEL_PATH="${MODEL_PATH:-}"
if [[ -n "$MODEL_PATH" ]]; then
    [[ "$NFS_SHARE" == "true" ]] && err "model_path and nfs_share are mutually exclusive (NFS shares an HF cache)"
    [[ "$ABLIT" == "1" ]] && err "model_path and ABLIT=1 are mutually exclusive"
    [[ "$FP8_DENSE" == "true" ]] && err "model_path and fp8_dense are mutually exclusive"
    MODEL_PATH="${MODEL_PATH/#\~/$HOME}"
    [[ -d "$MODEL_PATH" ]] || err "model_path not found: $MODEL_PATH"
    MODEL_ID="${MODEL_ID:-local/$(basename "$MODEL_PATH")}"
    DO_DOWNLOAD_DEFAULT=false
fi

# ABLIT=1 serves the gated Keys house QSA o_proj checkpoint (L3–47).
if [[ "$ABLIT" == "1" ]]; then
    if [[ "$FP8_DENSE" == "true" ]]; then
        warn "ABLIT=1 ignored for checkpoint selection: FP8_DENSE=true (MODEL_ID=$MODEL_ID)"
    else
        MODEL_ID="$ABLIT_MODEL_ID"
    fi
fi
MODEL_ID="${MODEL_ID:-}"
[[ -n "$MODEL_ID" ]] || err "MODEL_ID is not set (checked recipes/$RECIPE.yaml and .env)"
# What vLLM gets as its model argument.
MODEL_ARG="$MODEL_ID"
if [[ -n "$MODEL_PATH" ]]; then
    MODEL_ARG="/model"
fi

# Reduced-vocabulary MTP drafting: path to a token-id list from
# scripts/build_draft_vocab.py, or empty to draft over the full 248,320 vocabulary.
# Relative paths resolve against the repo root: the overlay mount below needs an
# absolute host path for docker -v, and `cd $SCRIPT_DIR` alone is not enough for
# callers that pass a path from a different working directory.
MTP_DRAFT_VOCAB="${MTP_DRAFT_VOCAB:-}"
if [[ -n "$MTP_DRAFT_VOCAB" && "$MTP_DRAFT_VOCAB" != /* ]]; then
    MTP_DRAFT_VOCAB="$SCRIPT_DIR/$MTP_DRAFT_VOCAB"
fi
# Custom chat template (e.g. chat_template_cod.jinja for the CoD prior).
# Resolved like MTP_DRAFT_VOCAB; bind-mounted and passed via --chat-template.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
if [[ -n "$CHAT_TEMPLATE" ]]; then
    [[ "$CHAT_TEMPLATE" != /* ]] && CHAT_TEMPLATE="$SCRIPT_DIR/$CHAT_TEMPLATE"
    [[ -f "$CHAT_TEMPLATE" ]] || err "chat_template not found: $CHAT_TEMPLATE"
fi
# QSA Triton launch profile: stock | gb10 | path to JSON from overlays/qsa_gb10/bench_qsa_kernels.py
QSA_PROFILE="${QSA_PROFILE:-stock}"
MTP_DISABLE_BLOCK_DROP="${MTP_DISABLE_BLOCK_DROP:-0}"
MTP_INDEX_SHARE="${MTP_INDEX_SHARE:-false}"
VLLM_QSA_DET_TOPK="${VLLM_QSA_DET_TOPK:-}"
VLLM_MOE_DET_FINALIZE="${VLLM_MOE_DET_FINALIZE:-}"
VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-}"
# Refuse to launch when another process already holds the GPU (both nodes).
REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-true}"

# YaRN only makes sense ABOVE the native 262144 context. At or below native,
# rope scaling degrades quality for zero benefit — force it off.
YARN_ENABLE="${YARN_ENABLE:-false}"
if [[ "$YARN_ENABLE" == "true" && "$MAX_MODEL_LEN" -le 262144 ]]; then
    echo "NOTE: MAX_MODEL_LEN=$MAX_MODEL_LEN <= native 262144 — YaRN force-disabled."
    YARN_ENABLE=false
fi

# ---------------------------------------------------------------------------
# Parse CLI flags
# ---------------------------------------------------------------------------
DO_DOWNLOAD="${DO_DOWNLOAD_DEFAULT:-true}"
DO_LAUNCH=true
DO_SYNC=true

for arg in "$@"; do
    case "$arg" in
        --no-download)  DO_DOWNLOAD=false ;;
        --no-launch)    DO_LAUNCH=false ;;
        --launch)       DO_DOWNLOAD=false; DO_SYNC=false ;;
        --nfs)          NFS_SHARE=true ;;
        --no-nfs)       NFS_SHARE=false ;;
        -h|--help)
            echo "Usage: $0 [--no-download] [--no-launch] [--launch] [--nfs|--no-nfs]"
            echo ""
            echo "  (default)      Download weights on head (HF recipes), rsync to worker, launch"
            echo "  --no-download  Skip HF download (weights already cached on head)"
            echo "  --no-launch    Download + sync weights only, don't start vLLM"
            echo "  --launch       Skip download + sync; apply patch and launch"
            echo "  --nfs          Share the head cache over NFS instead of rsync (no worker copy)"
            echo "  --no-nfs       Force rsync distribution even if nfs_share is set"
            echo "  RECIPE=<name>  Boot recipes/<name>.yaml (default: prod; mia = vendor reference)"
            echo "  ABLIT=1        Serve the gated Keys house QSA L3-47 checkpoint"
            echo "                 (accept the Hugging Face terms, then ABLIT=1 ./download.sh)"
            exit 0
            ;;
        *)
            err "Unknown argument: $arg (try --help)"
            ;;
    esac
done

if [[ -n "$MODEL_PATH" && "$NFS_SHARE" == "true" ]]; then
    err "model_path and --nfs are mutually exclusive (NFS shares an HF cache)"
fi

if [[ "$ABLIT" == "1" && "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
    warn "ABLIT=1: serving gated Keys checkpoint ($ABLIT_MODEL_ID)."
    warn "     Safety refusals are removed. MTP, PLE, experts and the chat template stay stock."
    warn "     Compatible ONLY with the nvidia dual-Spark NVFP4 layout (this recipe)."
fi
