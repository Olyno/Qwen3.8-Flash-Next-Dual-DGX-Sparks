#!/bin/bash
# ProbSparse spike launcher. Exact day-1 ngram-recipe server (ng_launch.sh):
# wk1 working copy served by the qwen38-flash-next image with the fork-lean
# PLE/modelopt patches, GMU 0.78, ctx 262144, no drafter. The only difference
# from that baseline run is the expert-count override below.
# Usage: K=8 ./ps_launch.sh <container> <port>
#   K=10 reproduces the day-1 baseline exactly (override equals checkpoint value).
set -euo pipefail
NAME=$1; PORT=$2; K=${K:?set K=<experts per token>}
MODEL=${MODEL:-$HOME/models/Qwen3.8-Flash-Next-NVFP4-wk1}
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
    -v /home/olyno/Qwen38-overthinking-lab/serve/files/ple_layer_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro \
    -v /home/olyno/Qwen38-overthinking-lab/serve/files/modelopt_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py:ro \
    -v /home/olyno/Qwen38-overthinking-lab/serve/files/ple_offload/ple_offload_layer.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/ple_offload_layer.py:ro \
    -v /home/olyno/Qwen38-overthinking-lab/serve/files/ple_offload/connector.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/connector.py:ro \
    -v /home/olyno/Qwen38-overthinking-lab/serve/files/ple_offload/worker.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/worker.py:ro \
    -v /home/olyno/Qwen38-overthinking-lab/serve/files/ple_offload/protocol.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/protocol.py:ro \
    -v /home/olyno/.cache/huggingface:/root/.cache/huggingface \
    -v /home/olyno/.cache/vllm:/root/.cache/vllm \
    -v "$MODEL:$MODEL:ro" \
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
