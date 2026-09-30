#!/bin/bash
# R5: QSA kernel-geometry family match — _is_sm120() exact-tuple test excludes
# GB10 (12,1), so prod silently uses the GB300 launch table while the sm120
# table ships unused in the same file. Arm: family match (major==12).
set -uo pipefail
R=$HOME/v30_bench; OV=$HOME/upgrade/v30/overlay; MODEL=$HOME/models/q38-lean-hyb
NAME=v30r5; PORT=8901; LOG=$R/ride_r5.log
QP=$R/r5_qsa_patch
rm -rf $QP; mkdir -p $QP/orig $QP/ops
cp $OV/qsa_patch/qsa.py $QP/ 2>/dev/null
sed 's/== (12, 0)/.major == 12/' $OV/qsa_patch/ops/qsa.py > $QP/ops/qsa.py
grep -n "major == 12" $QP/ops/qsa.py || { echo "R5 SKIP: sed no-op" >> $LOG; exit 4; }
python3 -m py_compile $QP/ops/qsa.py || { echo "R5 SKIP: compile" >> $LOG; exit 4; }
docker rm -f $NAME >/dev/null 2>&1
spec='{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic","rejection_sample_method":"block","disable_eagle_block_drop":true}'
ho='{"text_config": {"ple_embedding_dtype": "float8_e4m3fn", "num_experts_per_tok": 6}}'
python3 ~/fork/files/evict_page_cache.py "$MODEL" >/dev/null 2>&1 || true  # the boot that follows a 120G cold read MUST release stale pages first
docker run -d --name $NAME --gpus all --network host --ipc host \
  --cap-add SYS_NICE --cap-add SYS_PTRACE --ulimit memlock=-1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/root/.cache/huggingface \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/vllm:/root/.cache/vllm \
  -v $HOME/.cache/vllm/triton_home:/root/.triton \
  -v $MODEL:$MODEL:ro \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/root/.cache/vllm/fi_autotune \
  -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_PLE_MMAP_DIR=/root/.cache/vllm/ple_mmap_v030 -e VLLM_PLE_MMAP_ADVICE=1 \
  -v $OV/ngram_embedding.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ngram_embedding.py:ro \
  -v $OV/model.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model.py:ro \
  -v $OV/mtp.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/mtp.py:ro \
  -v $OV/hyperconnection.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/hyperconnection.py:ro \
  -v $QP/qsa.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/qsa.py:ro \
  -v $QP/ops/qsa.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/qsa.py:ro \
  vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90 \
  $MODEL --served-model-name qwen3.8-flash-next --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --safetensors-load-strategy lazy --enable-chunked-prefill --reasoning-parser qwen3 \
  --quantization modelopt --kv-cache-dtype fp8_e4m3 --max-model-len 131072 \
  --speculative-config "$spec" --hf-overrides "$ho" \
  --gpu-memory-utilization 0.748 >> $LOG 2>&1 || { echo "R5 RUN-FAIL" >> $LOG; exit 2; }
ok=0
for i in $(seq 1 180); do sleep 30; curl -s -m 5 localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; docker ps --format '{{.Names}}' | grep -qx $NAME || break; done
[ $ok -eq 1 ] || { docker logs $NAME 2>&1 | tail -30 >> $LOG; echo "R5 BOOT-FAIL $(date +%H:%M)" >> $LOG; docker rm -f $NAME >/dev/null 2>&1; exit 2; }
echo "R5 HEALTHY $(date +%H:%M)" >> $LOG
BENCH_PORT=$PORT python3 $HOME/fork/bench/decodebench.py --decode 600 --contexts 1000,100000 --temps 0.6 > $R/r5_pass1.txt 2>&1
curl -s -m 10 localhost:$PORT/metrics > $R/r5_metrics.txt 2>&1
docker rm -f $NAME >/dev/null 2>&1
echo "R5 DONE $(date +%H:%M)" >> $LOG
