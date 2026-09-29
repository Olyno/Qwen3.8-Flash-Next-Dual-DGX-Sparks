#!/bin/bash
# R3: ITL-isolation arm. Same A4 prod stack, plus --long-prefill-token-threshold
# 1024 + --max-num-scheduled-tokens 8192 (knobs exist in image, config/scheduler.py:56,70).
# Hypothesis (research #8): c=1 tok/s ~unchanged; multi-stream inter-token
# spikes shrink (prefill no longer rides inside decode steps). Measure both.
set -uo pipefail
R=$HOME/v30_bench; OV=$HOME/upgrade/v30/overlay; MODEL=$HOME/models/q38-lean-hyb
NAME=v30r3; PORT=8897
LOG=$R/ride_r3.log
docker rm -f $NAME >/dev/null 2>&1
K=6 bash $OV/launch_v30.sh "$NAME" "$PORT" "$MODEL" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic","rejection_sample_method":"block","disable_eagle_block_drop":true}' \
  --long-prefill-token-threshold 1024 >> $LOG 2>&1 &
LP=$!
for i in $(seq 1 40); do sleep 60; curl -s -m 5 localhost:$PORT/health >/dev/null 2>&1 && break; kill -0 $LP 2>/dev/null || break; done
if ! curl -s -m 5 localhost:$PORT/health >/dev/null 2>&1; then echo "R3 BOOT-FAIL $(date +%H:%M)" >> $LOG; docker logs $NAME 2>&1 | tail -30 >> $LOG; exit 2; fi
echo "R3 HEALTHY $(date +%H:%M)" >> $LOG
BENCH_PORT=$PORT python3 $HOME/fork/bench/decodebench.py --decode 600 --contexts 1000,100000 --temps 0.6 > $R/r3_pass1.txt 2>&1
python3 $HOME/fork/experiments/v30/concbench.py --port $PORT --levels 1,4,8 --decode 300 > $R/r3_conc.txt 2>&1
curl -s -m 10 localhost:$PORT/metrics > $R/r3_metrics.txt 2>&1
docker rm -f $NAME >/dev/null 2>&1
echo "R3 DONE $(date +%H:%M)" >> $LOG
