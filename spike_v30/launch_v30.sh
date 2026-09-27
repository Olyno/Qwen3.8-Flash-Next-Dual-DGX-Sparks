#!/bin/bash
# vLLM v0.30.0 upgrade-test launcher. Mirrors ps_launch.sh (same server flags)
# on the new image with the ported [fp8dense overlay] bind-mounted. PLE CPU
# offload is native now (EngramConfig, VLLM_PLE_CPU_OFFLOAD defaults to 1 in
# v0.30 — see lane PR #68: set PLE_OFFLOAD=false / VLLM_PLE_CPU_OFFLOAD=0 to
# disable). The packed-table mmap path is OPT-IN via VLLM_PLE_PACKED_TABLE_DIR:
# it crashed the first boot (see README diagnosis — UVA read of the whole
# 47.7 GiB map during profiling inside a 100g cgroup on unified memory).
# Usage: K=6 [VLLM_TORCH_PROFILER_DIR=...] [VLLM_PLE_PACKED_TABLE_DIR=...] ./launch_v30.sh <container> <port> [model-dir] [extra vllm args...]
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
    GPU_UTIL=$(python3 - "$MODEL" "$MAXLEN" "$PLE_OFFLOAD" <<'PY'
import json, os, sys, math
model, maxlen, ple_off = sys.argv[1], int(sys.argv[2]), sys.argv[3]
idx = os.path.join(model, "model.safetensors.index.json")
wtb = json.load(open(idx))["metadata"]["total_size"] if os.path.exists(idx) else 105e9
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
PT_ARGS=()
if [[ -n "${VLLM_PLE_PACKED_TABLE_DIR:-}" ]]; then
    PT_ARGS=(-e "VLLM_PLE_PACKED_TABLE_DIR=$VLLM_PLE_PACKED_TABLE_DIR")
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
    "${PT_ARGS[@]}" \
    "${PROF_ARGS[@]}" \
    -v $OV/model.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model.py:ro \
    -v $OV/mtp.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/mtp.py:ro \
    -v $OV/hyperconnection.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/hyperconnection.py:ro \
    -v $OV/ngram_embedding.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ngram_embedding.py:ro \
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
    --kv-cache-dtype auto \
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
