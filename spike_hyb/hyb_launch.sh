#!/bin/bash
# Hybrid-FP8 spike server: nvidia-layout checkpoint with dense projections in
# FP8 per-channel. Same TP1 recipe as the ProbSparse control (ps_launch.sh)
# plus the four fp8dense overlay mounts (modelopt_hybrid carries BOTH the
# FP8_BLOCK MoE fix and the per-channel dispatch).
# Usage: ./hyb_launch.sh <container> <port> [model_dir]
set -euo pipefail
NAME=$1; PORT=$2; MODEL=${3:-$HOME/models/q38-hyb}
SERVE=$HOME/Qwen38-overthinking-lab/serve/files
HYB=$HOME/hyb_spike
OV=$HYB/overlay
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
    -e VLLM_PLE_OFFLOAD_STEP_TIMEOUT=300 \
    -e HF_HOME=/root/.cache/huggingface \
    -v $SERVE/ple_layer_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro \
    -v $OV/modelopt_hybrid.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py:ro \
    -v $OV/hyperconnection.py.patched:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/hyperconnection.py:ro \
    -v $OV/model.py.patched:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/model.py:ro \
    -v $OV/mtp.py.patched:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/mtp.py:ro \
    -v $SERVE/ple_offload/ple_offload_layer.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/ple_offload_layer.py:ro \
    -v $SERVE/ple_offload/connector.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/connector.py:ro \
    -v $SERVE/ple_offload/worker.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/worker.py:ro \
    -v $SERVE/ple_offload/protocol.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/protocol.py:ro \
    -v /home/olyno/.cache/huggingface:/root/.cache/huggingface \
    -v /home/olyno/.cache/vllm:/root/.cache/vllm \
    -v "$MODEL:$MODEL:ro" \
    vllm/vllm-openai:qwen38-flash-next \
    "$MODEL" \
    --hf-overrides "{\"text_config\": {\"ple_embedding_dtype\": \"float8_e4m3fn\"}}" \
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
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --quantization modelopt \
    --host 0.0.0.0 \
    --port "$PORT"
