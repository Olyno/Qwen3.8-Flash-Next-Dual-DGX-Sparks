#!/bin/bash
# CONTEXT-QUALITY A/B (user priority 09-30): settle 1M-YaRN vs 262K-native for
# real, on the production stack. Every prior claim about 1M was inference
# (acceptance tail + KV math), never a measured needle-vs-context curve.
#
# Arms on q38-lean-hyb + v0.30 + fp8-KV + k=4 via launch_v30.sh (extra argv
# lands AFTER the model path and after the baked --hf-overrides; vLLM's CLI is
# last-wins on duplicate flags, so passing the full merged blob overrides it):
#   native : MAXLEN=262144, no rope override. 500k/950k cells then FAIL at the
#            API — that refusal is part of the answer ("best" = what the box
#            serves well, not what it technically accepts).
#   yarn   : MAXLEN=1048576 + YaRN factor 4.0 (the retired 09-27 prod config).
# Measures at 60k/200k/500k/950k: needle retrieval (longctx.py, 3 needles at
# 5/50/95% depth + TTFT) + prose decode (decodebench) + /metrics acceptance.
# yarn boots LAST (rope re-init + huge-KV boot can take >30 min; the native
# arm is already banked if it dies).
#
# Locks the chain GPU lock unless CHAIN_HELD is inherited (chain injects it).
set -uo pipefail
R=$HOME/v30_bench
LOG=$R/ride_ctx.log; : > "$LOG"
MODEL=$HOME/models/q38-lean-hyb
OV=$HOME/upgrade/v30/overlay
NAME=v30ctx; PORT=8899
BASE_HO='"ple_embedding_dtype": "float8_e4m3fn", "num_experts_per_tok": 6'
NAT_HO="{\"text_config\": {$BASE_HO}}"
YARN_HO="{\"text_config\": {$BASE_HO, \"rope_parameters\": {\"rope_type\": \"yarn\", \"factor\": 4.0, \"original_max_position_embeddings\": 262144}}}"
SPEC='{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic","rejection_sample_method":"block","disable_eagle_block_drop":true}'

if [ -z "${CHAIN_HELD:-}" ]; then
    while ! mkdir $R/gpu.lock 2>/dev/null; do sleep 60; done
    echo "PID=$$" > $R/gpu.lock/owner; trap 'rm -rf $R/gpu.lock' EXIT
fi

health() {
    local t0=$(date +%s)
    while ! curl -s -m 2 localhost:$PORT/health >/dev/null 2>&1; do
        sleep 10
        [ $(($(date +%s)-t0)) -gt 3600 ] && { echo "BOOT-TIMEOUT $(date +%H:%M)" >>$LOG; return 1; }
    done
    echo "HEALTHY-AFTER $(($(date +%s)-t0))s $(date +%H:%M)" >>$LOG
}

arm() { # $1 = native|yarn, $2 = maxlen, $3 = hf-overrides blob
    docker rm -f $NAME >/dev/null 2>&1
    export K=6 MAXLEN=$2 KV_FP8=1 PLE_MMAP=$HOME/.cache/vllm/ple_mmap_v030 OV BATCHED=2048 SPARSE_MAX_LOGITS_MB=256   # 8192-boot freezes msi (3/3 today); 2048 = banked-reference-comparable + survives
    bash $HOME/fork/experiments/v30/launch_v30.sh $NAME $PORT $MODEL \
        --speculative-config "$SPEC" --hf-overrides "$3" >>"$LOG" 2>&1 \
        || { echo "RUN-FAIL $1 $(date +%H:%M)" >>$LOG; docker logs $NAME 2>&1 | tail -30 > $R/ctx_${1}_FAIL.txt; docker rm -f $NAME >/dev/null 2>&1; return 1; }
    if ! health; then
        docker logs $NAME 2>&1 | tail -30 > $R/ctx_${1}_FAIL.txt
        docker rm -f $NAME >/dev/null 2>&1; return 1
    fi
    for T in 60000 200000 500000 950000; do
        if [ "$1" = native ] && [ "$T" -gt 262144 ]; then
            echo "$1 $T REFUSED-NATIVE-CTX $(date +%H:%M)" | tee -a $R/ctx_verdict.txt
            continue
        fi
        BENCH_PORT=$PORT timeout 3000 python3 $HOME/fork/bench/longctx.py --target $T --max-tokens 512 > $R/ctx_${1}_needle_$T.txt 2>&1
        BENCH_PORT=$PORT timeout 1800 python3 $HOME/fork/bench/decodebench.py --decode 300 --contexts $T --temps 0.6 --tasks prose > $R/ctx_${1}_dec_$T.txt 2>&1
        curl -s -m 10 localhost:$PORT/metrics > $R/ctx_${1}_metrics_$T.txt 2>&1
        echo "$1 $T done $(date +%H:%M)" >>"$LOG"
    done
    docker rm -f $NAME >/dev/null 2>&1
}

echo "CTX AB START $(date)" >>"$LOG"
arm native 262144  "$NAT_HO"
arm yarn   1048576 "$YARN_HO"
echo "CTX AB DONE $(date)" >>"$LOG"
