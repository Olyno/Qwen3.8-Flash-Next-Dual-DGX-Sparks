#!/bin/bash
# T1b re-run slot: fires after chain_r completes AND the GPU lock frees
# (a3retry/R6 take priority; lock is first-come after release).
set -uo pipefail
R=$HOME/v30_bench
while ! grep -q "chain_r complete" $R/chainr.log 2>/dev/null; do sleep 300; done
bash $HOME/ride_t1b.sh; echo "T1b-rerun rc=$? $(date +%H:%M)" >> $R/chainr.log
