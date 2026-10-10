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

# Topology: NODES in .env picks single vs dual. Everything downstream keys off
# NNODES; TP follows it (one GPU per Spark) unless a recipe pins
# tensor_parallel_size.
NODES="${NODES:-1}"
[[ "$NODES" == "1" || "$NODES" == "2" ]] || err "NODES must be 1 or 2 (got: '$NODES')"
NNODES=$NODES
WORKER_IP="${WORKER_IP:-}"
WORKER_USER="${WORKER_USER:-}"
if [[ "$NNODES" -eq 2 && -z "$WORKER_IP" ]]; then
    err "NODES=2 but WORKER_IP is not set in .env"
fi

# Topology-derived serving defaults — a recipe can still pin any of these.
# Dual-node halves per-rank weights/KV/state: the full checkpoint fits on-GPU
# (no PLE host table in the decode loop), the native 262144 context fits in
# the doubled KV pool, and 8 seqs keeps the MTP verify batch tiny (8*(1+k)
# << max_num_batched_tokens; mtp_block.py re-validates).
if [[ -z "${PLE_OFFLOAD:-}" ]]; then
    [[ "$NNODES" == 2 ]] && PLE_OFFLOAD=false || PLE_OFFLOAD=true
fi
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$([[ "$NNODES" == 2 ]] && echo 262144 || echo 65536)}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$([[ "$NNODES" == 2 ]] && echo 8 || echo 4)}"

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
[[ "$TENSOR_PARALLEL_SIZE" == "$NNODES" ]] || err "tensor_parallel_size=$TENSOR_PARALLEL_SIZE but NODES=$NNODES (one GPU per Spark — fix the recipe pin or NODES)"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-true}"
MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-3}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"   # fp8 needs patches/patch_qsa_fp8_kv.py, applied automatically in step 4f; auto = bf16
# dtype of the GDN recurrent (SSM) state. The checkpoint asks for float32; the
# fused GDN kernel also accepts bfloat16 (FUSED_GDN_STATE_DTYPES in
# qwen_gdn_linear_attn.py). BF16 halves the ~0.23 GB per sequence the state
# costs to read and write every step, and halves the mamba page, which lets
# vLLM pick a smaller attention block. Empty keeps the checkpoint's float32.
MAMBA_SSM_CACHE_DTYPE="${MAMBA_SSM_CACHE_DTYPE:-}"
# PLE_OFFLOAD default is topology-derived in the NODES block above.

# SM12x plan table covers TP=1 and TP=2 shapes (vllm#59753 + vllm#59632); at
# other TP sizes no plan matches and the standard linear path is kept
# (patches/gb10_skinny_gemm), so default it on only where it can engage.
# Recipe true/false overrides.
if [[ -z "${SKINNY_GEMM:-}" ]]; then
    [[ "$TENSOR_PARALLEL_SIZE" == "1" || "$TENSOR_PARALLEL_SIZE" == "2" ]] && SKINNY_GEMM=true || SKINNY_GEMM=false
fi
# Vision MLP intermediate_size=4304 is not divisible by 16 after TP split (4304/2=2152).
# NVFP4 kernels require input features % 16 == 0, so replicate the encoder on each GPU.
MM_ENCODER_TP_MODE="${MM_ENCODER_TP_MODE:-data}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
# MoE backend overrides (v0.30 kernel config). Empty = vLLM auto per quant type.
# flashinfer_b12x only supports NVFP4, so with our mixed checkpoint the FP8
# block-scale draft experts must be pinned separately (DRAFT_MOE_BACKEND=auto).
MOE_BACKEND="${MOE_BACKEND:-}"
DRAFT_MOE_BACKEND="${DRAFT_MOE_BACKEND:-}"
# Optional --gdn-prefill-backend (v0.30, flag shipped by vllm#55715):
# flashinfer pins the FlashInfer CuTe-DSL chunk_gated_delta_rule GDN prefill
# kernel (its gate was widened to SM12x), triton forces the stock FLA path.
# Empty = vLLM's own resolution (auto), i.e. stock behavior.
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-}"
[[ -z "$GDN_PREFILL_BACKEND" || "$GDN_PREFILL_BACKEND" =~ ^(flashinfer|triton|cutedsl)$ ]] || err "gdn_prefill_backend must be flashinfer, triton or cutedsl (got: '$GDN_PREFILL_BACKEND')"
# Optional docker --cpuset-cpus for the vLLM container (both ranks). The GB10
# mixes ten 3.9 GHz Cortex-X925 cores (5-9,15-19) with ten 2.8 GHz A725 cores
# (0-4,10-14); pinning to the X925 set measured +2-3 % at every concurrency in
# the vendor single-Spark recipe. Pure scheduling, no output change. Empty =
# all cores. Verified against /sys cpufreq on the target host.
CPUSET="${CPUSET:-}"
[[ -z "$CPUSET" || "$CPUSET" =~ ^[0-9,-]+$ ]] || err "cpuset must be a CPU list like 5-9,15-19 (got: '$CPUSET')"
# Weight distribution. false (default) = each node keeps its own copy of the
# checkpoint, worker seeded by rsync from the head. true = head exports its
# cache over NFS on ConnectX and the worker mounts it read-only (HF models only).
NFS_SHARE="${NFS_SHARE:-false}"
# Optional: head ConnectX address used as the NFS server (auto-detected from IFACE).
NFS_SERVER_IP="${NFS_SERVER_IP:-}"
V030="${V030:-false}"
[[ "$SKINNY_GEMM" != "true" || "$V030" == "true" ]] || warn "skinny_gemm ignored: the SM12x plan overlay only exists on the v030 lane (v030: false)."
SKIP_PLE_PATCH="${SKIP_PLE_PATCH:-false}"
# K3 lazy GDN state commit for MTP verify (v0.30 lane only,
# patches/patch_gdn_lazy_v030.py): the verify kernel commits the fp32 GDN state
# once per step instead of after every token. Needs speculative decoding
# (k=1..7) and an fp32 GDN state; a load-time self-test keeps the stock kernel
# unless the Triton reimplementation is bitwise-identical on this GPU.
LAZY_GDN="${LAZY_GDN:-false}"
# ReplaySSM-GDN spec decode (v0.30 lane only, patches/replayssm_gdn/): port of
# vllm#47576's GDN variant. MTP verify reconstructs each window from an fp32
# checkpoint + a circular d/k/g ring instead of writing the full GDN state per
# accepted token; the checkpoint is rewritten only when the ring fills
# (REPLAYSSM_GDN_BUFFER_LEN + 1 + k tokens). Needs speculative decoding;
# mutually exclusive with lazy_gdn. Buffer len must be >= 1 + k.
REPLAYSSM_GDN="${REPLAYSSM_GDN:-false}"
REPLAYSSM_GDN_BUFFER_LEN="${REPLAYSSM_GDN_BUFFER_LEN:-16}"
# Draft-only FP8 (E4M3, per-row scale) copy of the MTP drafter's lm_head rows
# (v0.30 lane only, patches/patch_mtp_draft_vocab_v030.py --fp8): quantizes the
# reduced draft-vocab slice when MTP_DRAFT_VOCAB is set, else the full shard.
# The target keeps its BF16 head for verify, so emitted tokens are unchanged.
FP8_DRAFT_HEAD="${FP8_DRAFT_HEAD:-false}"
# Fused multi-step MTP draft metadata for QSA (v0.30 lane only,
# patches/patch_qsa_fused_draft_v030.py): the QSA metadata builder opts into
# the speculator's in-place update between draft steps instead of a full
# attention-metadata rebuild per step (vllm#58449 port). Needs k > 1.
QSA_FUSED_DRAFT="${QSA_FUSED_DRAFT:-false}"
# Clamp QSA pre-indexer RoPE positions into the cos/sin table (v0.30 lane
# only, patches/patch_qsa_rope_clamp_v030.py): stock reads cos_sin[pos]
# unchecked, and CUDA-graph warmup dummy positions can index past the table
# (IMA on SM121/GB10).
QSA_ROPE_CLAMP="${QSA_ROPE_CLAMP:-false}"
# Adaptive MTP draft depth (v0.30 lane only,
# patches/patch_mtp_adaptive_depth.py): the V2 speculator truncates the
# per-step draft chain once the draft head's own survival product (running
# product of per-step top-token probs, batch mean) drops below
# MTP_ADAPTIVE_DEPTH_THRESHOLD. k is per-step uniform, target verification is
# unchanged. Needs k > 1.
MTP_ADAPTIVE_DEPTH="${MTP_ADAPTIVE_DEPTH:-false}"
MTP_ADAPTIVE_DEPTH_THRESHOLD="${MTP_ADAPTIVE_DEPTH_THRESHOLD:-0.5}"
# VLLM_MTP_ADAPTIVE_DEBUG=1 in the container: rate-limited logging of the
# recorded per-step probs / survival products / widths (see the patcher).
MTP_ADAPTIVE_DEBUG="${MTP_ADAPTIVE_DEBUG:-false}"
# posix_fadvise(DONTNEED) on each checkpoint shard right after the loader
# consumes it (v0.30 lane only, patches/patch_load_drop_cache_v030.py): keeps
# the page cache one shard deep during the load on unified memory.
LOAD_DROP_CACHE="${LOAD_DROP_CACHE:-false}"
[[ "$LAZY_GDN" != "true" || "$V030" == "true" ]] || err "lazy_gdn is only supported on the vLLM 0.30 lane (v030: true)."
[[ "$REPLAYSSM_GDN" != "true" || "$V030" == "true" ]] || err "replayssm_gdn is only supported on the vLLM 0.30 lane (v030: true)."
[[ "$FP8_DRAFT_HEAD" != "true" || "$V030" == "true" ]] || err "fp8_draft_head is only supported on the vLLM 0.30 lane (v030: true)."
[[ "$QSA_FUSED_DRAFT" != "true" || "$V030" == "true" ]] || err "qsa_fused_draft is only supported on the vLLM 0.30 lane (v030: true)."
[[ "$QSA_ROPE_CLAMP" != "true" || "$V030" == "true" ]] || err "qsa_rope_clamp is only supported on the vLLM 0.30 lane (v030: true)."
[[ "$LOAD_DROP_CACHE" != "true" || "$V030" == "true" ]] || err "load_drop_cache is only supported on the vLLM 0.30 lane (v030: true)."
[[ "$MTP_ADAPTIVE_DEPTH" != "true" || "$V030" == "true" ]] || err "mtp_adaptive_depth is only supported on the vLLM 0.30 lane (v030: true)."
[[ "$MTP_ADAPTIVE_DEPTH" != "true" || "$MTP_NUM_SPECULATIVE_TOKENS" -gt 1 ]] || err "mtp_adaptive_depth needs mtp_num_speculative_tokens > 1 (k <= 1 has no chain to truncate)."
[[ "$MTP_ADAPTIVE_DEPTH_THRESHOLD" =~ ^0(\.[0-9]+)?$|^1(\.0+)?$ ]] || err "mtp_adaptive_depth_threshold must be in [0, 1] (got: '$MTP_ADAPTIVE_DEPTH_THRESHOLD')"
[[ -z "$GDN_PREFILL_BACKEND" || "$V030" == "true" ]] || err "gdn_prefill_backend is only supported on the vLLM 0.30 lane (v030: true)."
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
# Explicit --async-scheduling. vLLM 0.30 already resolves async scheduling ON
# for MTP + the mp executor (config/vllm.py enables it unless incompatible;
# mtp is in EagleModelTypes and the mp executor supports it), so false here
# keeps the resolved default — it does NOT turn async scheduling off. true
# pins the flag explicitly against future resolution changes.
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-false}"
# Optional spec-config sampling overrides (empty = vLLM defaults: greedy draft,
# standard rejection). Both are lossless w.r.t. the target distribution.
MTP_DRAFT_SAMPLE_METHOD="${MTP_DRAFT_SAMPLE_METHOD:-}"
MTP_REJECTION_SAMPLE_METHOD="${MTP_REJECTION_SAMPLE_METHOD:-}"
# Optional --long-prefill-token-threshold (empty = vLLM default 0/off).
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-}"
VLLM_QSA_DET_TOPK="${VLLM_QSA_DET_TOPK:-}"
VLLM_MOE_DET_FINALIZE="${VLLM_MOE_DET_FINALIZE:-}"
VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-}"
# Marlin MoE atomic-add (envs.py; marlin_utils.py:610). Only meaningful with
# moe_backend: marlin; empty = container default (0).
VLLM_MARLIN_USE_ATOMIC_ADD="${VLLM_MARLIN_USE_ATOMIC_ADD:-}"
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
