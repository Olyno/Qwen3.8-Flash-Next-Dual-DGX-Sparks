#!/bin/bash
# Standby: after the running chain_r2 completes (CTX/R1..R5/A3gap/R7/R6), fire
# the corrected T1b telemetry rerun. Mirrors the a6 standby pattern. Never
# edits the live chain script (bash reads it by file offset — in-place edits
# corrupt it mid-run, the known field lesson).
R=$HOME/v30_bench
exec >>$R/chain_t1b2.log 2>&1
echo "=== t1b2 standby start $(date) ==="
while ! grep -q "chain_r2 complete" $R/chainr2.log 2>/dev/null; do sleep 120; done
echo "chain_r2 finished $(date +%H:%M) — firing t1b2"
CHAIN_HELD=0 bash $HOME/ride_t1b2.sh
echo "=== t1b2 standby done $(date) ==="
