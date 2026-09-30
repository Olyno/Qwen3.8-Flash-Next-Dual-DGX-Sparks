#!/bin/bash
# R7 — Dynamic Speculative Decoding (DSD) under concurrency, on OUR image.
# Image-verified 09-30: config/speculative.py:479 carries
# num_speculative_tokens_per_batch_size; v1/core/sched/scheduler.py builds
# dynamic_sd_lookup; worker/gpu/cudagraph_utils.py sizes graph capture per K.
# Upstream docs (2026-07-07) list tested = Eagle/Eagle3/DFlash — MTP is
# "may or may not work out of the box": a boot/validation FAIL is itself the
# answer (R4 precedent), success means the concurrency lever already shipped.
#
# Two boots, SAME shape (max_num_seqs 32 — required to fill the table ranges;
# all banked rows ran 8, so neither side inherits a baseline):
#   static: k=4 fixed                       -> r7s_*
#   dynamic: [1-8]=4, [9-16]=2, [17-32]=0   -> r7d_*
# Pre-registered band: ADOPT dynamic if aggregate tok/s at c>=9 beats static
# by >=5 % AND c=1 prose loses <=2 % (c=1 runs the same K=4 path; it is the
# regression canary). Quality structurally safe: lossless rejection unchanged.
set -uo pipefail
R=$HOME/v30_bench; mkdir -p $R
LOG=$R/ride_r7.log; : > "$LOG"
MODEL=$HOME/models/q38-lean-hyb
NAME=v30r7; PORT=8897
SPEC_BASE='"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic","rejection_sample_method":"block","disable_eagle_block_drop":true'
SPEC_STATIC="{$SPEC_BASE}"
SPEC_DYN="{$SPEC_BASE,\"num_speculative_tokens_per_batch_size\":[[1,8,4],[9,16,2],[17,32,0]]}"

if [ -z "${CHAIN_HELD:-}" ]; then
    while ! mkdir $R/gpu.lock 2>/dev/null; do sleep 60; done
    echo "PID=$$" > $R/gpu.lock/owner; trap 'rm -rf $R/gpu.lock' EXIT
fi

boot() { # $1 = s|d, $2 = spec json
    docker rm -f $NAME >/dev/null 2>&1
    export K=6 MAXLEN=131072 KV_FP8=1 BATCHED=8192 PLE_MMAP=$HOME/.cache/vllm/ple_mmap_v030 OV=$HOME/upgrade/v30/overlay
    bash $HOME/fork/experiments/v30/launch_v30.sh $NAME $PORT $MODEL \
        --max-num-seqs 32 --speculative-config "$2" >>"$LOG" 2>&1 || {
        echo "RUN-FAIL $1 $(date +%H:%M)" >>"$LOG"
        docker logs $NAME 2>&1 | tail -40 > $R/r7_${1}_FAIL.txt; docker rm -f $NAME >/dev/null 2>&1; return 1; }
    local t0=$(date +%s)
    until curl -s -m 2 localhost:$PORT/health >/dev/null 2>&1; do
        sleep 10
        [ $(($(date +%s)-t0)) -gt 3600 ] && { echo "BOOT-TIMEOUT $1 $(date +%H:%M)" >>"$LOG"; docker logs $NAME 2>&1 | tail -40 > $R/r7_${1}_FAIL.txt; docker rm -f $NAME >/dev/null 2>&1; return 1; }
    done
    echo "HEALTHY $1 $(($(date +%s)-t0))s $(date +%H:%M)" >>"$LOG"
    if [ "$1" = d ]; then
        grep -m2 -iE "dynamic" <(docker logs $NAME 2>&1) >>"$LOG" \
            || echo "NO-DYNAMIC-LOG-EVIDENCE (config possibly ignored)" >>"$LOG"
    fi
}

echo "R7 START $(date +%H:%M)" >>"$LOG"
# dynamic FIRST: if validation refuses MTP, the arm is decided early and we
# still run static-only (which doubles as the missing max_num_seqs=32 row).
if boot d "$SPEC_DYN"; then
    python3 $HOME/fork/experiments/v30/concbench.py --port $PORT --levels 1,4,8,16,32 --decode 300 > $R/r7d_conc.txt 2>&1
    BENCH_PORT=$PORT python3 $HOME/fork/bench/decodebench.py --decode 600 --contexts 1000 --temps 0.6 --tasks prose,code > $R/r7d_pass1.txt 2>&1
    curl -s -m 10 localhost:$PORT/metrics > $R/r7d_metrics.txt 2>&1
    docker rm -f $NAME >/dev/null 2>&1
fi
boot s "$SPEC_STATIC" || { echo "R7 BOTH-FAIL $(date +%H:%M)" >>"$LOG"; exit 2; }
python3 $HOME/fork/experiments/v30/concbench.py --port $PORT --levels 1,4,8,16,32 --decode 300 > $R/r7s_conc.txt 2>&1
BENCH_PORT=$PORT python3 $HOME/fork/bench/decodebench.py --decode 600 --contexts 1000 --temps 0.6 --tasks prose,code > $R/r7s_pass1.txt 2>&1
curl -s -m 10 localhost:$PORT/metrics > $R/r7s_metrics.txt 2>&1
docker rm -f $NAME >/dev/null 2>&1
echo "R7 DONE $(date +%H:%M)" >>"$LOG"
