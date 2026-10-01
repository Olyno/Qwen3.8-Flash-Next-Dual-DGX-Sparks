#!/bin/bash
# R1: reactivity-fix evidence — prod stack boot twice.
#  boot1 (cold triton cache): count JIT-during-inference warnings, time-to-health.
#  boot2 (warm): expect ~zero JIT warnings, faster time-to-health.
# Then decodebench prose/code @1k on boot2 -> R1 speed row vs banked A4 (30.5/35.5).
# Runs AFTER the A3 gate chain frees the GPU. No set -e: every step records and moves on.
set -uo pipefail
R=$HOME/v30_bench; mkdir -p $R
LOG=$R/ride_r1.log; : > "$LOG"
MODEL=$HOME/models/q38-lean-hyb
OV=$HOME/upgrade/v30/overlay
NAME=v30r1; PORT=8895
TLIVE=0
jit_count() { docker logs $NAME 2>&1 | grep -c "jit_monitor.py:141"; }
health_t0() {
    local t0=$(date +%s)
    while ! curl -s -m 2 localhost:$PORT/health >/dev/null 2>&1; do
        sleep 5; [ $(($(date +%s)-t0)) -gt 2400 ] && { echo "BOOT-TIMEOUT $(date +%H:%M)" >>$LOG; return 1; }
    done
    echo "HEALTHY-AFTER $(($(date +%s)-t0))s $(date +%H:%M)"
}
boot() { # $1 = label
    docker rm -f $NAME >/dev/null 2>&1
    # cold = wipe the persisted jit cache dir first; warm = keep it
    [ "$1" = cold ] && rm -rf $HOME/.cache/vllm/triton_home/*
    mkdir -p $HOME/.cache/vllm/triton_home $HOME/.cache/vllm/fi_autotune
    local spec='{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic","rejection_sample_method":"block","disable_eagle_block_drop":true}'
    local ho='{"text_config": {"ple_embedding_dtype": "float8_e4m3fn", "num_experts_per_tok": 6}}'
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
      vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90 \
      $MODEL --served-model-name qwen3.8-flash-next --max-num-seqs 8 --max-num-batched-tokens 2048 \
      --safetensors-load-strategy lazy --enable-chunked-prefill --reasoning-parser qwen3 \
      --quantization modelopt --kv-cache-dtype fp8_e4m3 --max-model-len 131072 \
      --speculative-config "$spec" --hf-overrides "$ho" --port $PORT \
      --gpu-memory-utilization 0.748 >>"$LOG" 2>&1 || { echo "RUN-FAIL $1" >>$LOG; return 1; }
    health_t0 $1 | tee -a "$LOG"
    echo "JIT-WARNINGS-AFTER-BOOT $1=$(jit_count) $(date +%H:%M)" >>"$LOG"
}
echo "R1 START $(date +%H:%M)" >>"$LOG"
boot cold || exit 2
python3 $HOME/fork/experiments/v30/concbench.py --port $PORT --levels 1 --decode 300 > $R/r1_cold_sanity.txt 2>&1
# one warm-up turn per kernel family so the cold boot populates the cache (the
# bench itself exercises conv/PLE/topp paths; add a sampling temp-1.0 ping)
curl -s -m 600 localhost:$PORT/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"List three ideas."}],"temperature":1.0,"max_tokens":400}' >/dev/null 2>&1
echo "COLD-JIT-TOTAL $(jit_count) $(date +%H:%M)" >>"$LOG"
boot warm || exit 2
JW=$(jit_count); echo "WARM-JIT-TOTAL $JW $(date +%H:%M)" >>"$LOG"
BENCH_PORT=$PORT python3 $HOME/fork/bench/decodebench.py --decode 600 --contexts 1000,100000 --temps 0.6 > $R/r1_pass1.txt 2>&1
curl -s -m 10 localhost:$PORT/metrics > $R/r1_metrics.txt 2>&1
python3 $HOME/fork/experiments/v30/concbench.py --port $PORT --levels 4,8,16,32 --decode 300 > $R/r1_conc_census.txt 2>&1
python3 $HOME/fork/experiments/v30/concbench.py --port $PORT --levels 4,8,16 --decode 300 --min-tokens 280 > $R/r1_conc_mintok.txt 2>&1
BENCH_PORT=$PORT python3 $HOME/fork/bench/longctx.py --target 250000 --max-tokens 512 > $R/r1_needle262k.txt 2>&1
docker rm -f $NAME >/dev/null 2>&1
echo "R1 DONE $(date +%H:%M) warm_jit=$JW" >>"$LOG"
