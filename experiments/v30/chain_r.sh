#!/bin/bash
# Post-A3 queue: T1b -> R1 (JIT evidence+speed+census+needle) -> R2 (fused
# draft, self-skips if port unstage) -> R3 (ITL isolation) -> R4 (NVFP4 MoE
# b12x opt-in) -> P1b2 (profiler). One heavy job per box.
set -uo pipefail
R=$HOME/v30_bench
while ! grep -q "chain_p1b_a3 complete" $R/chain2.log 2>/dev/null; do sleep 120; done
echo "CHAIN-R START $(date +%H:%M)" >> $R/chainr.log
bash $HOME/ride_t1b.sh >> $R/chainr.log 2>&1; echo "T1b rc=$? $(date +%H:%M)" >> $R/chainr.log
bash $HOME/ride_r1.sh  >> $R/chainr.log 2>&1; echo "R1  rc=$? $(date +%H:%M)" >> $R/chainr.log
bash $HOME/ride_r2.sh  >> $R/chainr.log 2>&1; echo "R2  rc=$? $(date +%H:%M)" >> $R/chainr.log
bash $HOME/ride_r3.sh  >> $R/chainr.log 2>&1; echo "R3  rc=$? $(date +%H:%M)" >> $R/chainr.log
bash $HOME/ride_r4.sh  >> $R/chainr.log 2>&1; echo "R4  rc=$? $(date +%H:%M)" >> $R/chainr.log
if grep -q "R1  rc=0" $R/chainr.log 2>/dev/null; then
  bash $HOME/ride_p1b2.sh >> $R/chainr.log 2>&1; echo "P1b2 rc=$? $(date +%H:%M)" >> $R/chainr.log
fi
echo "chain_r complete $(date +%H:%M)" >> $R/chainr.log
