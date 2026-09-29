#!/bin/bash
# R6 queue-slot: after chain_r completes AND the A3 gap-fill container is gone,
# run the DeepGEMM correctness A/B. One heavy job per box, strict order.
set -uo pipefail
R=$HOME/v30_bench
while true; do
  grep -q "chain_r complete" $R/chainr.log 2>/dev/null || { sleep 300; continue; }
  ! docker ps -q -f name=v30a3r | grep -q . && ! pgrep -f "[r]ide_a3retry" >/dev/null && break
  sleep 300
done
echo "R6 QUEUE-OPEN $(date +%H:%M)" >> $R/chainr.log
bash $HOME/ride_r6.sh; echo "R6 rc=$? $(date +%H:%M)" >> $R/chainr.log
