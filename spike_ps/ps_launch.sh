#!/bin/bash
# ProbSparse spike: day-1 ng_launch recipe (stock nvidia checkpoint, PLE offload,
# fp8 PLE override), drafter removed, expert-count-per-token injected via
# --hf-overrides. No checkpoint copy needed: the override replaces the config
# field at load time.
# Usage: K=8 ./ps_launch.sh <container> <port>
set -euo pipefail
NAME=$1; PORT=$2; K=${K:?set K=<experts per token>}
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
    -v /home/olyno/Qwen38-overthinking-lab/serve/fork-lean/files/ple_layer_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro \
    -v /home/olyno/Qwen38-overthinking-lab/serve/fork-lean/files/modelopt_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py:ro \
    -v /home/olyno/Qwen38-overthinking-lab/serve/fork-lean/files/ple_offload/ple_offload_layer.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/ple_offload_layer.py:ro \
    -v /home/olyno/Qwen38-overthinking-lab/serve/fork-lean/files/ple_offload/connector.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/connector.py:ro \
    -v /home/olyno/Qwen38-overthinking-lab/serve/fork-lean/files/ple_offload/worker.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/worker.py:ro \
    -v /home/olyno/Qwen38-overthinking-lab/serve/fork-lean/files/ple_offload/protocol.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/protocol.py:ro \
    -v /home/olyno/.cache/huggingface:/root/.cache/huggingface \
    -v /home/olyno/.cache/vllm:/root/.cache/vllm \
    vllm/vllm-openai:qwen38-flash-next \
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
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --quantization modelopt \
    --host 0.0.0.0 \
    --port "$PORT"
