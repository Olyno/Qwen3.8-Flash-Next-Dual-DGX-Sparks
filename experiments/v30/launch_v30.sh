#!/bin/bash
# vLLM v0.30.0 upgrade-test launcher. Same server flags as ps_launch.sh on the
# new image, with the ported [fp8dense overlay] (model/mtp/hyperconnection)
# bind-mounted.
#
# PLE: v0.30 native offload (VLLM_PLE_CPU_OFFLOAD=1) pins the whole 47.7 GiB
# table as NON-evictable anonymous RAM -> unified-pool D-state hang (crashes
# #3-#8). Set PLE_MMAP=<dir> for the PROVEN fix (patch_ple_mmap_v030.py from
# the single-spark lane): ATS row reads over a shared file map, no pinning;
# the table builds once under the dir, later boots skip the copy entirely.
# PLE_OFFLOAD=0 is a last resort (table on GPU -> driver OOM at 131k ctx).
#
# Optional, each inert without its flag:
#   DRAFT_VOCAB=<ids.txt>  reduced drafter lm_head (sister patcher trio)
#   KV_FP8=1               vllm#55557 QSA fp8_e4m3 KV backport
#   VLLM_TORCH_PROFILER_DIR=<dir>  torch profiler capture (P1)
#
# Usage: K=6 PLE_MMAP=$HOME/.cache/vllm/ple_mmap_v030 ./launch_v30.sh <container> <port> [model-dir] [extra vllm args...]
set -euo pipefail
NAME=$1; PORT=$2; K=${K:-6}
MODEL=${3:-$HOME/models/Qwen3.8-Flash-Next-NVFP4-wk1}
OV=${OV:-$HOME/upgrade/v30/overlay}
# Boot-robustness overrides (hangs #3-5: v0.30 compile-warm died under stock
# AND reduced-profile watermarks; PLE_OFFLOAD=0 tests the pinned-table path).
GPU_UTIL=${GPU_UTIL:-}; BATCHED=${BATCHED:-8192}; MAXLEN=${MAXLEN:-131072}
PLE_OFFLOAD=${PLE_OFFLOAD:-1}; LOAD_STRAT=${LOAD_STRAT:-lazy}
# GPU_UTIL empty => DERIVE like single-spark start.sh Step 2: budget =
# min(weights+overhead5.6+mtp1.49+max(kv_need,12G), MemTotal - HOST_RESERVE26).
if [[ -z "$GPU_UTIL" ]]; then
    GPU_UTIL=$(python3 - "$MODEL" "$MAXLEN" "$PLE_OFFLOAD" "${PLE_MMAP:-}" <<'PY'
import json, os, sys, math
model, maxlen, ple_off, mmap_ = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
idx = os.path.join(model, "model.safetensors.index.json")
wtb = json.load(open(idx))["metadata"]["total_size"] if os.path.exists(idx) else 105e9
# PLE lives on HOST whenever offload=1 (pinned or mmap — GPU budget is the
# same; the difference is host-side evictability, which HOST_RESERVE covers
# only for the evictable mmap table).
ple = 47.68 * 2**30 if ple_off == "1" else 0.0
w = max(wtb - ple, 0) / 2**30
kv_need = maxlen * 29482 * 0.58 / 2**30           # fp8 KV mult
fixed = w + 5.6 + 1.49
mt = int(open("/proc/meminfo").readline().split()[1]) / 2**20
budget = min(fixed + max(kv_need, 12.0), mt - 26.0)
print(f"{math.floor(budget / mt * 1000) / 1000:.3f}")
PY
    )
    echo "launch_v30: derived GPU_UTIL=$GPU_UTIL" >&2
fi
PROF_ARGS=()
if [[ -n "${VLLM_TORCH_PROFILER_DIR:-}" ]]; then
    PROF_ARGS=(-e VLLM_TORCH_PROFILER_DIR=/prof -v "$VLLM_TORCH_PROFILER_DIR":/prof)
fi
PT_ENV=()
NG_EMB="$OV/ngram_embedding.py"
if [[ -n "${PLE_MMAP:-}" ]]; then
    # Proven v0.30 path (single-spark lane): pristine image ngram + patcher.
    # Table dir lives INSIDE the already bind-mounted ~/.cache/vllm (rw), so
    # no extra mount; first boot builds it from shard copies through the
    # normal loader, commits via msync+fingerprint; later boots skip the
    # 47.7 GiB copy entirely. Lookups read rows over ATS, no cudaHostRegister.
    ORIG="$OV/ngram_embedding.orig.py"
    if [[ ! -f "$ORIG" ]]; then
        CID=$(docker create vllm/vllm-openai:v0.30.0 /bin/true)
        docker cp "$CID:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ngram_embedding.py" "$ORIG"
        docker rm "$CID" >/dev/null
    fi
    python3 "$OV/patch_ple_mmap_v030.py" "$ORIG" "$NG_EMB" || { echo "launch_v30: mmap patch anchor drift" >&2; exit 1; }
    PT_ENV=(-e "VLLM_PLE_MMAP_DIR=/root/.cache/vllm/ple_mmap_v030" -e VLLM_PLE_MMAP_ADVICE=1)
    mkdir -p "$HOME/.cache/vllm/ple_mmap_v030"
fi

# --- optional proven sister-lane patches (each inert without its flag) -----
# DRAFT_VOCAB=<ids.txt>: reduced drafter lm_head (their patch_mtp_draft_vocab
# trio; needs files/patch_mtp_draft_vocab.py NEXT TO the v030 patcher, both in
# $OV). Applies onto OUR ported mtp.py so the fp8dense HC-mixer splice stays.
# KV_FP8=1: vllm#55557 fp8-e4m3 QSA KV backport (patch_qsa_fp8_kv_v030.py);
# pristine image qsa files extracted, patched, bind-ro over the package.
MTP_PY="$OV/mtp.py"
if [[ -n "${DRAFT_VOCAB:-}" ]]; then
    # v030 patcher resolves its inputs relative to ITS own directory: stage a
    # private copy dir with (v030-patcher, base patcher, our mtp as .orig).
    DVD="$OV/dvdraft"
    mkdir -p "$DVD"
    cp "$OV/patch_mtp_draft_vocab_v030.py" "$OV/patch_mtp_draft_vocab.py" "$DVD/"
    cp "$OV/mtp.py" "$DVD/mtp_v030_patched.py.orig"
    ( cd "$DVD" && python3 patch_mtp_draft_vocab_v030.py ) || { echo "launch_v30: draft-vocab anchor drift" >&2; exit 1; }
    MTP_PY="$DVD/mtp_v030_patched.py"
    DV_ARGS=(-v "$DRAFT_VOCAB:/root/draft_vocab.txt:ro" -e VLLM_MTP_DRAFT_VOCAB=/root/draft_vocab.txt)
else
    DV_ARGS=()
fi
KV_ARGS=()
QSA_MOUNTS=()
if [[ "${KV_FP8:-}" == 1 ]]; then
    QO="$OV/qsa_orig"; QP="$OV/qsa_patch"
    if [[ ! -f "$QO/qsa.py" || ! -f "$QO/ops/qsa.py" ]]; then
        CID=$(docker create vllm/vllm-openai:v0.30.0 /bin/true)
        mkdir -p "$QO/ops"
        docker cp "$CID:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/qsa.py" "$QO/qsa.py"
        docker cp "$CID:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/qsa.py" "$QO/ops/qsa.py"
        docker rm "$CID" >/dev/null
    fi
    python3 "$OV/patch_qsa_fp8_kv_v030.py" "$QO" "$QP" || { echo "launch_v30: qsa#55557 does not apply" >&2; exit 1; }
    QSA_MOUNTS=(-v "$QP/qsa.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/qsa.py:ro" \
                -v "$QP/ops/qsa.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/qsa.py:ro")
    KV_DTYPE=${KV_DTYPE:-fp8_e4m3}
fi
# Page-cache release before launch (GB10 unified pool; lane finding: weight
# loading can CUDA-OOM on an "idle" box without it — files/evict_page_cache.py).
python3 "$HOME/Qwen3.8-Flash-Next-Dual-DGX-Sparks/files/evict_page_cache.py" \
    "$MODEL" >/dev/null 2>&1 || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run \
    -d --name "$NAME" \
    --gpus all --network host --ipc host \
    --cap-add SYS_NICE --cap-add SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e "VLLM_PLE_CPU_OFFLOAD=$PLE_OFFLOAD" \
    -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
    -e HF_HOME=/root/.cache/huggingface \
    "${PT_ENV[@]}" \
    "${PROF_ARGS[@]}" \
    "${DV_ARGS[@]}" \
    "${QSA_MOUNTS[@]}" \
    -v $OV/model.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model.py:ro \
    -v "$MTP_PY":/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/mtp.py:ro \
    -v $OV/hyperconnection.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/hyperconnection.py:ro \
    -v "$NG_EMB":/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ngram_embedding.py:ro \
    -v /home/olyno/.cache/huggingface:/root/.cache/huggingface \
    -v /home/olyno/.cache/vllm:/root/.cache/vllm \
    -v "$MODEL:$MODEL:ro" \
    vllm/vllm-openai:v0.30.0 \
    "$MODEL" \
    --hf-overrides "{\"text_config\": {\"ple_embedding_dtype\": \"float8_e4m3fn\", \"num_experts_per_tok\": $K}}" \
    --served-model-name qwen3.8-flash-next \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs 8 \
    --max-num-batched-tokens "$BATCHED" \
    --max-model-len "$MAXLEN" \
    --kv-cache-dtype "${KV_DTYPE:-auto}" \
    --safetensors-load-strategy "$LOAD_STRAT" \
    --enable-chunked-prefill \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --distributed-executor-backend mp \
    --compilation-config "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}" \
    --quantization modelopt \
    --host 0.0.0.0 \
    --port "$PORT" \
    "${@:4}"
