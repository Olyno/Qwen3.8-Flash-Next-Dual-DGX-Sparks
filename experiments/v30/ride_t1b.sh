#!/bin/bash
# ride_t1b.sh — recover the T1 acceptance telemetry that ride_t1 lost to the
# $KK_metrics unbound-var bug (speed/conc rows are ALREADY banked; this dumps
# ONLY spec_decode counters). One boot per K, 600-tok prose@1k as the draft
# generator (short, deterministic workload), curl /metrics, kill. ~12 min/arm.
set -uo pipefail
R=$HOME/v30_bench; OV=$HOME/upgrade/v30/overlay
REPO=$HOME/Qwen3.8-Flash-Next-Dual-DGX-Sparks
NAME_BASE=v30tb; PORT=8894; MODEL=$HOME/models/Qwen3.8-Flash-Next-NVFP4-wk1
export KV_FP8=1 MAXLEN=114688
LOG=$R/ride_T1b.log; exec >>"$LOG" 2>&1
echo "=== ride T1b start $(date) ==="
for KK in 1 2 3 4 9; do
  NAME=$NAME_BASE$KK
  echo "--- K=$KK boot $(date +%H:%M) ---"
  K=6 bash $OV/launch_v30.sh "$NAME" "$PORT" "$MODEL" --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$KK,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"disable_eagle_block_drop\":true}"
  RC=1
  for _ in $(seq 1 480); do
    curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { RC=0; break; }
    docker ps -q -f name="^$NAME$" | grep -q . || { RC=2; break; }
    sleep 30
  done
  if [ $RC -ne 0 ]; then echo "T1b K=$KK BOOT-FAIL rc=$RC"; docker rm -f "$NAME" >/dev/null 2>&1; continue; fi
  BENCH_PORT=$PORT python3 "$REPO/bench/decodebench.py" --decode 600 --contexts 1000 --temps 0.6 --only prose > "$R/t1b_k${KK}_probe.txt" 2>&1 || true
  sleep 20
  curl -s localhost:$PORT/metrics > "$R/t1_k${KK}_metrics.txt"
  grep -q spec_decode_num_drafts "$R/t1_k${KK}_metrics.txt" || echo "T1B-WARN K=$KK no spec counters in dump"
  docker rm -f "$NAME" >/dev/null 2>&1
  echo "K=$KK metrics dumped $(date +%H:%M) lines=$(wc -l < "$R/t1_k${KK}_metrics.txt")"
done
T1_KS=1,2,3,4,9 python3 $HOME/_speedrepo/spike_v30/t1_analyze_v2.py "$R" > "$R/t1_verdict.txt" 2>&1 && echo T1b-VERDICT-OK || echo T1b-ANALYZE-FAILED >> "$R/t1_verdict.txt"
echo "=== ride T1b complete $(date) ==="
