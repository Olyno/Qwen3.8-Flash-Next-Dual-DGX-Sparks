#!/bin/bash
# R4: NVFP4 MoE b12x backend opt-in (image note: excluded from auto on SM121
# pending CUTLASS MMA guard). Single flag; boot-crash is itself evidence.
set -uo pipefail
R=$HOME/v30_bench; OV=$HOME/upgrade/v30/overlay; MODEL=$HOME/models/q38-lean-hyb
NAME=v30r4; PORT=8899; LOG=$R/ride_r4.log
docker rm -f $NAME >/dev/null 2>&1
KV_FP8=1 MAXLEN=131072 BATCHED=2048 K=6 bash $OV/launch_v30.sh "$NAME" "$PORT" "$MODEL" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic","rejection_sample_method":"block","disable_eagle_block_drop":true}' \
  --moe-backend flashinfer_b12x > $LOG 2>&1 &
LP=$!
ok=0
for i in $(seq 1 40); do sleep 60; kill -0 $LP 2>/dev/null || break; curl -s -m 5 localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; done
kill -0 $LP 2>/dev/null || { wait $LP; RC=$?; docker logs $NAME > $R/r4_dockerlogs_FAIL.txt 2>&1; echo "R4 BOOT-FAIL rc=$RC $(date +%H:%M)" >> $LOG; grep -iE "b12x|mma|guard|assert|error" $R/r4_dockerlogs_FAIL.txt | head -8 >> $LOG; docker rm -f $NAME >/dev/null 2>&1; exit 2; }
[ $ok -eq 1 ] || { docker logs $NAME > $R/r4_dockerlogs_FAIL.txt 2>&1; echo "R4 NO-HEALTH $(date +%H:%M)" >> $LOG; docker rm -f $NAME >/dev/null 2>&1; exit 2; }
echo "R4 HEALTHY $(date +%H:%M)" >> $LOG
docker logs $NAME 2>&1 | grep -iE "b12x|Selected kernels|fused_moe" | head -6 > $R/r4_backend_proof.txt
BENCH_PORT=$PORT python3 $HOME/fork/bench/decodebench.py --decode 600 --contexts 1000,100000 --temps 0.6 > $R/r4_pass1.txt 2>&1
curl -s -m 10 localhost:$PORT/metrics > $R/r4_metrics.txt 2>&1
docker rm -f $NAME >/dev/null 2>&1
echo "R4 DONE $(date +%H:%M)" >> $LOG
