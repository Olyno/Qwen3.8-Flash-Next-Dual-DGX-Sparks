#!/bin/bash
# A6: prod-stack K retune. CONDITIONAL — boot only if T1b EV math says
# argmax_k A(k)/C(k) != 4 for prose (banked wk1 ladder peaked K2 23.9 vs
# K4 20.2; two independent boxes report "speculate less for prose").
# Usage: K=2 ./ride_a6_k.sh   (writes a6_k${K}_pass1.txt for r_analyze pairing)
set -uo pipefail
R=$HOME/v30_bench; OV=$HOME/upgrade/v30/overlay; MODEL=$HOME/models/q38-lean-hyb
K=${K:?set K}; NAME=v30a6k$K; PORT=891$((K%10)); LOG=$R/ride_a6_k$K.log
[ -f $R/a6_lock ] && { echo "A6 already ran this K" >> $LOG; exit 5; }
docker rm -f $NAME >/dev/null 2>&1
KV_FP8=1 MAXLEN=131072 BATCHED=2048 PLE_MMAP=$HOME/.cache/vllm/ple_mmap_v030 bash $OV/launch_v30.sh "$NAME" "$PORT" "$MODEL" \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$K,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"disable_eagle_block_drop\":true}" >> $LOG 2>&1 &
ok=0
for i in $(seq 1 40); do sleep 60; curl -s -m 5 localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; done
[ $ok -eq 1 ] || { docker logs $NAME 2>&1 | tail -30 >> $LOG; echo "A6 BOOT-FAIL K=$K" >> $LOG; docker rm -f $NAME >/dev/null 2>&1; exit 2; }
echo "A6 HEALTHY K=$K $(date +%H:%M)" >> $LOG
BENCH_PORT=$PORT python3 $HOME/fork/bench/decodebench.py --decode 600 --contexts 1000,100000 --temps 0.6 > $R/a6_k${K}_pass1.txt 2>&1
curl -s -m 10 localhost:$PORT/metrics > $R/a6_k${K}_metrics.txt 2>&1
docker rm -f $NAME >/dev/null 2>&1; touch $R/a6_lock
echo "A6 DONE K=$K $(date +%H:%M)" >> $LOG
