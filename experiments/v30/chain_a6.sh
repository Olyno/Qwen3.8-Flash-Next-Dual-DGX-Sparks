#!/bin/bash
# Standby A6 runner: never edits the live chain_r. Fires the K-retune arm only
# if a verdict-review writes ~/v30_bench/a6_request containing "K=<n>".
set -uo pipefail
R=$HOME/v30_bench
while ! grep -q "chain_r complete" $R/chainr.log 2>/dev/null; do sleep 300; done
[ -f $R/a6_request ] || { echo "A6 no-request $(date +%H:%M)" >> $R/chainr.log; exit 0; }
K=$(sed -n 's/^K=\([0-9]*\).*/\1/p' $R/a6_request)
echo "A6 START K=$K $(date +%H:%M)" >> $R/chainr.log
K=$K bash $HOME/ride_a6_k.sh; echo "A6 rc=$? $(date +%H:%M)" >> $R/chainr.log
