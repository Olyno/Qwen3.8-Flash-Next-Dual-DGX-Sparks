#!/bin/bash
# chain_r3 — the remaining queue after 09-30's partial harvest. Skips what the
# live box already banked (depth-sweep speed ladder, context native arm);
# everything else re-runs clean. Preflight + chain-held lock + 60-min windows
# like chain_r2. Started ONLY by hand after a collision/root-cause audit.
set -uo pipefail
R=$HOME/v30_bench; LOG=$R/chainr3.log
exec >>"$LOG" 2>&1
echo "=== chain_r3 start $(date) ==="
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
    if [ $(( $(date +%s) - t0 )) -gt 1800 ]; then echo "PREFLIGHT-WARN timeout — proceeding $(date +%H:%M)"; return 0; fi
    sleep 60
  done
}
run() { local name=$1; shift; preflight; echo "--- $name $(date +%H:%M) ---"; CHAIN_HELD=1 bash "$@"; echo "$name rc=$? $(date +%H:%M)"; }
run R2    $HOME/ride_r2.sh
run R3    $HOME/ride_r3.sh
run R4    $HOME/ride_r4.sh
run R5    $HOME/ride_r5.sh
run A3gap $HOME/ride_a3retry.sh
run R7    $HOME/ride_r7.sh
run R6    $HOME/ride_r6.sh
echo "=== chain_r3 complete $(date) ==="
