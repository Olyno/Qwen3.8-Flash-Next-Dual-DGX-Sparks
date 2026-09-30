#!/bin/bash
# CTX3 — the YaRN-1M arm of the context A/B, alone. The native arm banked
# its verdict tonight (ctx2: 200k PASSES with the #57105 backport, see
# docs/verdicts/V30-LANE.md). The yarn arm died twice for HARNESS reasons,
# never measurement ones: (1) missing VLLM_ALLOW_LONG_MAX_MODEL_LEN (fixed
# in ride_ctx2), (2) the launch_v30 PT_ENV clobber just fixed (6c15de3).
# This re-run uses the live fork launcher, so both fixes are in force.
set -uo pipefail
R=$HOME/v30_bench
LOG=$R/ride_ctx3.log; : > "$LOG"
MODEL=$HOME/models/q38-lean-hyb
OV=$HOME/upgrade/v30/overlay
NAME=v30ctx; PORT=8899
BASE_HO='"ple_embedding_dtype": "float8_e4m3fn", "num_experts_per_tok": 6'
YARN_HO="{\"text_config\": {$BASE_HO, \"rope_parameters\": {\"rope_type\": \"yarn\", \"factor\": 4.0, \"original_max_position_embeddings\": 262144}}}"
SPEC='{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic","rejection_sample_method":"block","disable_eagle_block_drop":true}'

if [ -z "${CHAIN_HELD:-}" ]; then
    while ! mkdir $R/gpu.lock 2>/dev/null; do sleep 60; done
    echo "PID=$$" > $R/gpu.lock/owner; trap 'rm -rf $R/gpu.lock' EXIT
fi

docker rm -f $NAME >/dev/null 2>&1
export K=6 MAXLEN=1048576 KV_FP8=1 PLE_MMAP=$HOME/.cache/vllm/ple_mmap_v030 OV BATCHED=2048 ALLOW_LONG=1 SPARSE_MAX_LOGITS_MB=64 QSA_RESERVE=1
bash $HOME/fork/experiments/v30/launch_v30.sh $NAME $PORT $MODEL \
    --speculative-config "$SPEC" --hf-overrides "$YARN_HO" >>"$LOG" 2>&1 \
    || { echo "RUN-FAIL yarn $(date +%H:%M)" >>$LOG; docker logs $NAME 2>&1 | tail -30 > $R/ctx_yarn_FAIL.txt; docker rm -f $NAME >/dev/null 2>&1; exit 1; }
t0=$(date +%s)
until curl -s -m 2 localhost:$PORT/health >/dev/null 2>&1; do
    sleep 10
    docker ps -q -f name="^$NAME$" | grep -q . || { echo "DIED yarn $(date +%H:%M)" >>$LOG; docker logs $NAME 2>&1 | tail -30 > $R/ctx_yarn_FAIL.txt; docker rm -f $NAME >/dev/null 2>&1; exit 1; }
    [ $(($(date +%s)-t0)) -gt 5400 ] && { echo "BOOT-TIMEOUT yarn $(date +%H:%M)" >>$LOG; docker logs $NAME 2>&1 | tail -30 > $R/ctx_yarn_FAIL.txt; docker rm -f $NAME >/dev/null 2>&1; exit 1; }
done
echo "HEALTHY-AFTER $(($(date +%s)-t0))s $(date +%H:%M)" >>"$LOG"
for T in 60000 200000 500000 950000; do
    BENCH_PORT=$PORT timeout 3000 python3 $HOME/fork/bench/longctx.py --target $T --max-tokens 512 > $R/ctx_yarn_needle_$T.txt 2>&1
    BENCH_PORT=$PORT timeout 1800 python3 $HOME/fork/bench/decodebench.py --decode 300 --contexts $T --temps 0.6 --tasks prose > $R/ctx_yarn_dec_$T.txt 2>&1
    curl -s -m 10 localhost:$PORT/metrics > $R/ctx_yarn_metrics_$T.txt 2>&1
    echo "yarn $T done $(date +%H:%M)" >>"$LOG"
done
docker rm -f $NAME >/dev/null 2>&1
echo "CTX3 DONE $(date)" >>"$LOG"
