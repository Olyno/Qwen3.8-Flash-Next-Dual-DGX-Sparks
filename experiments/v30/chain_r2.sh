#!/bin/bash
# chain_r2 — re-run of the R queue after the 09-29 pool-release cascade + the
# 16:50 box reboot killed everything (R1 never produced a valid row; all files
# were connection-refused tracebacks). Difference from chain_r:
#   * preflight between arms: no v30* containers + >=105 GiB GPU-free AND
#     >=80 GiB host MemAvailable (the unified-pool release lesson, twice
#     paid for); 30-min ceiling then proceed anyway with a loud WARN.
#   * gpu.lock held across the whole chain (a6/standalone arms respect it).
#   * health window 60 min per boot (8192-batch boot on this box is slow).
set -uo pipefail
R=$HOME/v30_bench; LOG=$R/chainr2.log
exec >>"$LOG" 2>&1
echo "=== chain_r2 start $(date) ==="
mkdir -p $R/gpu.lock && echo "PID=$$" > $R/gpu.lock/owner
trap 'rm -rf $R/gpu.lock' EXIT
preflight() {
  local t0=$(date +%s)
  while true; do
    local n=$(docker ps -q --filter "name=v30" | wc -l)
    local g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [[ "$g" =~ ^[0-9]+$ ]] || g=0   # GB10 unified pool reports N/A; host MemAvailable is the real gate
    local m=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    [ "$n" -eq 0 ] && [ "${g:-99999}" -le 16384 ] && [ "${m:-0}" -ge 80 ] && { echo "PREFLIGHT-OK gpu-used=${g}MiB avail=${m}G $(date +%H:%M)"; return 0; }
    if [ $(( $(date +%s) - t0 )) -gt 1800 ]; then echo "PREFLIGHT-WARN timeout gpu-used=${g}MiB avail=${m}G — proceeding $(date +%H:%M)"; return 0; fi
    sleep 60
  done
}
run() { local name=$1; shift; preflight; echo "--- $name $(date +%H:%M) ---"; CHAIN_HELD=1 bash "$@"; echo "$name rc=$? $(date +%H:%M)"; }
run T1b   $HOME/ride_t1b.sh
run CTX   $HOME/ride_ctx.sh
run R1    $HOME/ride_r1.sh
run R2    $HOME/ride_r2.sh
run R3    $HOME/ride_r3.sh
run R4    $HOME/ride_r4.sh
run R5    $HOME/ride_r5.sh
run A3gap $HOME/ride_a3retry.sh
run R6    $HOME/ride_r6.sh
[ -f $R/a6_request ] && K=$(sed -n 's/^K=\([0-9]*\).*/\1/p' $R/a6_request) bash $HOME/ride_a6_k.sh
echo "=== chain_r2 complete $(date) ==="
