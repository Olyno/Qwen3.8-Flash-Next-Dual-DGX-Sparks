#!/bin/bash
# CTX4 — the ship-gate cell for the 262K prod bump. ctx2 proved 200k prefill
# survives on v0.30 with #57105 + cap-64 — but at BATCHED=2048. Prod serves
# BATCHED=8192, and the allocator churn this fix addresses scales with the
# chunk size. This arm boots the EXACT intended prod config (lean-hyb, fp8-KV,
# k=4, mmap PLE, QSA_RESERVE, cap-64, 8192 batch, MAXLEN 262144) and runs one
# cell: 200k needle + 200k decode + pool guard. PASS -> recipes bump 262144
# (plus the KV_TARGET math below); FAIL -> prod stays 131072 and the verdict
# file gets the chunk-size footnote. One boot, ~40 min, decisive.
set -uo pipefail
R=$HOME/v30_bench
LOG=$R/ride_ctx4.log; : > "$LOG"
MODEL=$HOME/models/q38-lean-hyb
OV=$HOME/upgrade/v30/overlay
NAME=v30ctx4; PORT=8901
BASE_HO='"ple_embedding_dtype": "float8_e4m3fn", "num_experts_per_tok": 6'
NAT_HO="{\"text_config\": {$BASE_HO}}"
SPEC='{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic","rejection_sample_method":"block","disable_eagle_block_drop":true}'

if [ -z "${CHAIN_HELD:-}" ]; then
    while ! mkdir $R/gpu.lock 2>/dev/null; do sleep 60; done
    echo "PID=$$" > $R/gpu.lock/owner; trap 'rm -rf $R/gpu.lock' EXIT
fi

docker rm -f $NAME >/dev/null 2>&1
export K=6 MAXLEN=262144 KV_FP8=1 PLE_MMAP=$HOME/.cache/vllm/ple_mmap_v030 OV BATCHED=8192 ALLOW_LONG=1 SPARSE_MAX_LOGITS_MB=64 QSA_RESERVE=1
bash $HOME/fork/experiments/v30/launch_v30.sh $NAME $PORT $MODEL \
    --speculative-config "$SPEC" --hf-overrides "$NAT_HO" >>"$LOG" 2>&1 \
    || { echo "RUN-FAIL $(date +%H:%M)" >>$LOG; docker logs $NAME 2>&1 | tail -30 > $R/ctx_c4_FAIL.txt; docker rm -f $NAME >/dev/null 2>&1; exit 1; }
t0=$(date +%s)
until curl -s -m 2 localhost:$PORT/health >/dev/null 2>&1; do
    sleep 10
    docker ps -q -f name="^$NAME$" | grep -q . || { echo "DIED $(date +%H:%M)" >>$LOG; docker logs $NAME 2>&1 | tail -30 > $R/ctx_c4_FAIL.txt; docker rm -f $NAME >/dev/null 2>&1; exit 1; }
    [ $(($(date +%s)-t0)) -gt 5400 ] && { echo "BOOT-TIMEOUT $(date +%H:%M)" >>$LOG; docker logs $NAME 2>&1 | tail -30 > $R/ctx_c4_FAIL.txt; docker rm -f $NAME >/dev/null 2>&1; exit 1; }
done
echo "HEALTHY-AFTER $(($(date +%s)-t0))s $(date +%H:%M)" >>"$LOG"
BENCH_PORT=$PORT timeout 3600 python3 $HOME/fork/bench/longctx.py --target 200000 --max-tokens 512 > $R/ctx_c4_needle_200000.txt 2>&1
BENCH_PORT=$PORT timeout 1800 python3 $HOME/fork/bench/decodebench.py --decode 300 --contexts 200000 --temps 0.6 --tasks prose > $R/ctx_c4_dec_200000.txt 2>&1
grep -oE "KV cache size: [0-9,]+ tokens" "$LOG" | tail -1 >> $R/ctx_c4_needle_200000.txt
echo "c4 done $(date +%H:%M)" >>"$LOG"
docker rm -f $NAME >/dev/null 2>&1
echo "CTX4 DONE $(date)" >>"$LOG"
