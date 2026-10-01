#!/bin/bash
# R2: fused multi-step MTP draft (PR #58449 provider port) A/B.
# Needs files/v030_fused/qsa_cache.py staged (FusedPort58449). Arm OFF = A4
# banked config; boot with the patched common/qsa_cache.py mounted and assert
# the boot log DOES NOT print the "falling back to rebuilding" sentence.
# Compare decodebench rows + per-position acceptance (r2_metrics) vs r1_pass1.
set -uo pipefail
R=$HOME/v30_bench; mkdir -p $R
LOG=$R/ride_r2.log; : > "$LOG"
MODEL=$HOME/models/q38-lean-hyb
OV=$HOME/upgrade/v30/overlay
QSAF=$HOME/wt-reactivity/files/v030_fused/qsa_cache.py
NAME=v30r2; PORT=8898
[ -f "$QSAF" ] || { echo "R2 SKIP: no fused port at $QSAF" >>$LOG; exit 4; }
docker rm -f $NAME >/dev/null 2>&1
mkdir -p $HOME/.cache/vllm/triton_home $HOME/.cache/vllm/fi_autotune
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
  -v $HOME/upgrade/v30/overlay/qsa_patch/qsa.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/qsa.py:ro \
  -v $HOME/upgrade/v30/overlay/qsa_patch/ops/qsa.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/qsa.py:ro \
  -v $QSAF:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/common/qsa_cache.py:ro \
  vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90 \
  $MODEL --served-model-name qwen3.8-flash-next --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --safetensors-load-strategy lazy --enable-chunked-prefill --reasoning-parser qwen3 \
  --quantization modelopt --kv-cache-dtype fp8_e4m3 --max-model-len 131072 \
  --speculative-config "$spec" --hf-overrides "$ho" --port $PORT \
  --gpu-memory-utilization 0.748 >>"$LOG" 2>&1 || { echo "R2 RUN-FAIL" >>$LOG; exit 2; }
t0=$(date +%s); ok=0
while [ $(($(date +%s)-t0)) -lt 5400 ]; do
  curl -s -m 5 localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }
  docker ps --format '{{.Names}}' | grep -qx $NAME || break
  sleep 15
done
[ $ok -eq 1 ] || { echo "R2 BOOT-FAIL $(date +%H:%M)" >>$LOG; docker logs $NAME 2>&1 | tail -40 >>$LOG; docker rm -f $NAME >/dev/null 2>&1; exit 2; }
echo "R2 HEALTHY after $((($(date +%s)-t0)/60)) min $(date +%H:%M)" >>$LOG
docker logs $NAME 2>&1 | grep -ciE "falling back to rebuilding attention metadata" > $R/r2_fusedflag.txt
# ^ expect 0 (= fused ACTIVE); 1 = port did not flip the gate
BENCH_PORT=$PORT python3 $HOME/fork/bench/decodebench.py --decode 600 --contexts 1000,100000 --temps 0.6 > $R/r2_pass1.txt 2>&1
python3 $HOME/fork/experiments/v30/concbench.py --port $PORT --levels 1,4,8 --decode 300 > $R/r2_conc.txt 2>&1
curl -s -m 10 localhost:$PORT/metrics > $R/r2_metrics.txt 2>&1
docker rm -f $NAME >/dev/null 2>&1
echo "R2 DONE $(date +%H:%M) fused_fallback_count=$(cat $R/r2_fusedflag.txt)" >>$LOG
