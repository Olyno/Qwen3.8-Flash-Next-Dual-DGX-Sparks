#!/bin/bash
# Standby (post-chain_r3): what the live queue structurally cannot run.
# (1) t1b2 — fixed telemetry probe; (2) ctx2 — YaRN arm with the rope-ceiling
# override; (3) r1b — product-row rerun (warm cache, 130k needle).
# Then a self-audit: the live queue booted the drivers that existed at 17:13;
# three of them carry bugs fixed later on main (R3/R4 wrapper-pid health loop;
# R2's 40-min window can't survive the fused arm's extra JIT compile). For
# those names ONLY, if chainr3.log recorded rc=2, re-run with the current
# driver. rc=0/4 (ran fine / deliberate skip) arms are never re-run.
R=$HOME/v30_bench
exec >>$R/chain_t1b2.log 2>&1
echo "=== standby start $(date) ==="
while ! grep -q "chain_r3 complete" $R/chainr3.log 2>/dev/null; do sleep 120; done
echo "queue finished $(date +%H:%M) — firing corrected reruns"
CHAIN_HELD=0 bash $HOME/ride_t1b2.sh
CHAIN_HELD=0 bash $HOME/ride_ctx2.sh
CHAIN_HELD=0 bash $HOME/ride_r1b.sh
CHAIN_HELD=0 bash $HOME/ride_r8.sh   # decode-step anatomy (settles the kernel-lever shortlist)
cd ~/fork && git fetch -q origin main && git reset -q --hard origin/main
for n in R2 R3 R4 R5 R6; do
    if grep -q "^$n rc=2" $R/chainr3.log 2>/dev/null; then
        f=$(echo $n | tr A-Z a-z)
        echo "--- standby redo $n (failed rc=2 in the live queue; driver since fixed) $(date +%H:%M) ---"
        cp --remove-destination ~/fork/experiments/v30/ride_$f.sh ~/
        CHAIN_HELD=1 bash $HOME/ride_$f.sh
        echo "$n redo rc=$? $(date +%H:%M)"
    fi
done
echo "=== standby done $(date) ==="
