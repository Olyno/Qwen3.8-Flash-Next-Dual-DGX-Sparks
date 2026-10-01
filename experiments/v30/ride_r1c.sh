#!/bin/bash
# R1c — the K-retune arm: r1b's suite at k=3 (t1b3 EV: k=3 beats k=4 at c=1, +8.7%); the product row for the proposed locked-constant change. Suite identical, only SPEC depth + names differ.
# its whole bench suite: BOTH engines died within a minute of /health. Root
# cause of the cold boot's death is in the log: the cold-cache wipe `rm -rf
# ~/.cache/vllm/triton_home/*` ran as the USER against a cache half-owned by
# root (containers write as root) — permission denied mid-sweep left the JIT
# cache in a broken half-state (metadata vs .so disagree = the TritonBundler
# emit/reload mismatch the README warns about): first request after boot
# crashed the engine, both times, and the driver rm'd the container (logs).
# The cold-vs-warm reactivity question was already answered 09-28 (persistent
# cache shipped; gx10 prod shows zero mid-turn JIT). What R1 still owes:
# the prod-stack 2048-batch speed row (now directly A4-comparable), the
# concurrency census 4/8/16/32 (feeds the R7 band), min-tokens, and the
# 262k needle — with one change: needle target 130000, NOT 250000, because
# 200k+ chunked prefill on v0.30 is a MEASURED death (pool bleed, guard-fired
# 13:24; upstream #56457/#57105 fix is 0.31-only). Deep-needle evidence now
# comes from the ctx arm's cells, not from a suicide run.
# Differences from ride_r1.sh: (1) NO cache wipe (warm cache only — the
# honest surviving claim), (2) docker logs archived after EVERY phase (no
# more evidence loss on death), (3) needle at 130k, (4) decodebench 1k+100k.
set -uo pipefail
R=$HOME/v30_bench; mkdir -p $R
LOG=$R/ride_r1c.log; : > "$LOG"
MODEL=$HOME/models/q38-lean-hyb
OV=$HOME/upgrade/v30/overlay
NAME=v30r1c; PORT=8896
ARCH=~/v30_bench/r1c_logs
mkdir -p $ARCH
jit_count() { docker logs $NAME 2>&1 | grep -c "jit_monitor.py:141"; }
archive_logs() { docker logs $NAME > $ARCH/$1.txt 2>&1 || true; }
health_t0() {
    local t0=$(date +%s)
    while ! curl -s -m 2 localhost:$PORT/health >/dev/null 2>&1; do
        sleep 5; [ $(($(date +%s)-t0)) -gt 3600 ] && { echo "BOOT-TIMEOUT $(date +%H:%M)" >>$LOG; return 1; }
    done
    echo "HEALTHY-AFTER $(($(date +%s)-t0))s $(date +%H:%M)"
}
echo "R1c START $(date +%H:%M)" >>"$LOG"
docker rm -f $NAME >/dev/null 2>&1
mkdir -p $HOME/.cache/vllm/triton_home $HOME/.cache/vllm/fi_autotune
spec="{\"method\":\"mtp\",\"num_speculative_tokens\":${KSPECS:-3},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"disable_eagle_block_drop\":true}"
ho='{"text_config": {"ple_embedding_dtype": "float8_e4m3fn", "num_experts_per_tok": 6}}'
python3 ~/fork/files/evict_page_cache.py "$MODEL" >/dev/null 2>&1 || true
docker run -d --name $NAME --gpus all --network host --ipc host \
  --cap-add SYS_NICE --cap-add SYS_PTRACE --ulimit memlock=-1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/root/.cache/huggingface \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/vllm:/root/.cache/vllm \
  -v $HOME/.cache/vllm/triton_home:/root/.triton \
  -v $MODEL:$MODEL:ro \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 \
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
  --gpu-memory-utilization 0.748 >>"$LOG" 2>&1 || { echo "RUN-FAIL" >>$LOG; archive_logs runfail; docker rm -f $NAME >/dev/null 2>&1; exit 2; }
health_t0 | tee -a "$LOG" || { archive_logs boottimeout; docker rm -f $NAME >/dev/null 2>&1; exit 2; }
echo "JIT-WARNINGS-AFTER-BOOT $(jit_count) $(date +%H:%M)" >>"$LOG"
BENCH_PORT=$PORT timeout 900 python3 $HOME/fork/experiments/v30/concbench.py --port $PORT --levels 1 --decode 300 > $R/r1c_sanity.txt 2>&1; archive_logs after_sanity
BENCH_PORT=$PORT timeout 3600 python3 $HOME/fork/bench/decodebench.py --decode 600 --contexts 1000,100000 --temps 0.6 --tasks prose,code > $R/r1c_pass1.txt 2>&1; archive_logs after_pass1
curl -s -m 10 localhost:$PORT/metrics > $R/r1c_metrics.txt 2>&1
BENCH_PORT=$PORT timeout 3600 python3 $HOME/fork/experiments/v30/concbench.py --port $PORT --levels 4,8,16,32 --decode 300 > $R/r1c_conc_census.txt 2>&1; archive_logs after_census
BENCH_PORT=$PORT timeout 3600 python3 $HOME/fork/experiments/v30/concbench.py --port $PORT --levels 4,8,16 --decode 300 --min-tokens 280 > $R/r1c_conc_mintok.txt 2>&1; archive_logs after_mintok
BENCH_PORT=$PORT timeout 3000 python3 $HOME/fork/bench/longctx.py --target 130000 --max-tokens 512 > $R/r1c_needle131k.txt 2>&1; archive_logs after_needle
# alive-proof: every phase file must have table lines, not a traceback tail
for f in r1c_sanity r1c_pass1 r1c_conc_census r1c_conc_mintok r1c_needle131k; do
    grep -qiE "URLError|Connection refused" $R/$f.txt && { echo "R1c-DEATH after $f (logs archived)" >>"$LOG"; break; }; true
done
docker rm -f $NAME >/dev/null 2>&1
echo "R1b DONE $(date +%H:%M)" >>"$LOG"
