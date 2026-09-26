#!/bin/bash
# vLLM v0.30.0 upgrade-test launcher. Mirrors ps_launch.sh (same server flags)
# on the new image with the ported [fp8dense overlay] bind-mounted. PLE CPU
# offload is native now (EngramConfig, VLLM_PLE_CPU_OFFLOAD default on); the
# packed-table mmap overlay is selected via VLLM_PLE_PACKED_TABLE_DIR.
# Usage: K=10 ./launch_v30.sh <container> <port> [model-dir] [extra vllm args...]
set -euo pipefail
NAME=$1; PORT=$2; K=${K:-10}
MODEL=${3:-$HOME/models/Qwen3.8-Flash-Next-NVFP4-wk1}
OV=/home/olyno/upgrade/v30/overlay
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run \
    -d --name "$NAME" \
    --gpus all --network host --ipc host \
    --cap-add SYS_NICE --cap-add SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 \
    --memory 100g --memory-swap 100g \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_PLE_CPU_OFFLOAD=1 \
    -e VLLM_PLE_PACKED_TABLE_DIR=/root/.cache/vllm/ple_cache/nvidia--Qwen3.8-Flash-Next-NVFP4 \
    -e HF_HOME=/root/.cache/huggingface \
    -v $OV/model.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model.py:ro \
    -v $OV/mtp.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/mtp.py:ro \
    -v $OV/hyperconnection.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/hyperconnection.py:ro \
    -v $OV/ngram_embedding.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ngram_embedding.py:ro \
    -v /home/olyno/.cache/huggingface:/root/.cache/huggingface \
    -v /home/olyno/.cache/vllm:/root/.cache/vllm \
    -v "$MODEL:$MODEL:ro" \
    vllm/vllm-openai:v0.30.0 \
    nvidia/Qwen3.8-Flash-Next-NVFP4 \
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
