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
    -e VLLM_PLE_CPU_OFFLOAD=1 \
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
    --gpu-memory-utilization 0.735 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --max-model-len 131072 \
    --kv-cache-dtype auto \
    --load-format safetensors \
    --safetensors-load-strategy lazy \
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
