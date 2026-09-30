#!/bin/bash
# Standby: after chain_r2 completes, serialize the corrected reruns:
# (1) T1b telemetry sweep (probe-flag fix), (2) context arm v2 (YaRN gets the
# rope-ceiling override this time). Both wait on the same completion marker,
# so they MUST be serial here — never two chains grabbing the GPU lock blind.
R=$HOME/v30_bench
exec >>$R/chain_t1b2.log 2>&1
echo "=== standby start $(date) ==="
while ! grep -q "chain_r2 complete" $R/chainr2.log 2>/dev/null; do sleep 120; done
echo "chain_r2 finished $(date +%H:%M) — firing t1b2 then ctx2"
CHAIN_HELD=0 bash $HOME/ride_t1b2.sh
CHAIN_HELD=0 bash $HOME/ride_ctx2.sh
CHAIN_HELD=0 bash $HOME/ride_r1b.sh
echo "=== standby done $(date) ==="
