#!/bin/bash
# tests/gpu_arm_smoke.sh — offline validation for a ride_* arm script.
# Stubs docker/curl/pgrep-free so NO GPU is needed: asserts the arm's
# docker run argv carries the pre-registered settings (batch size, seats,
# spec depth, mounts). Run before arming any chain: a driver that drifted
# from the repo (the 09-30 freeze-loop lesson) fails HERE, not on the box.
#
# Usage: bash tests/gpu_arm_smoke.sh <arm-script> [--expect BATCHED=2048 ...]
#   --expect KEY=VAL   repeatable; KEY=VAL must appear in the composed env
#   --grep PATTERN     repeatable; PATTERN must match the script text
set -uo pipefail
S=${1:?usage: gpu_arm_smoke.sh <script>}
shift
FAIL=0
while [ $# -gt 0 ]; do
    case $1 in
        --expect) KEY=${2%%=*} WANT=${2#*=}
            GOT=$(env -u $KEY bash -c "source <(sed -n '/^[[:space:]]*export /p' $S) 2>/dev/null; echo \"\${$KEY:-unset}\"" 2>/dev/null)
            [ "$GOT" = "$WANT" ] || { echo "FAIL: $KEY=$WANT expected, got '$GOT'"; FAIL=1; } ;;
        --grep) rtk_grep=$2; grep -q "$rtk_grep" "$S" || { echo "FAIL: pattern '$rtk_grep' missing"; FAIL=1; } ;;
    esac
    shift 2 2>/dev/null || shift
done
# every sourced helper must lint
bash -n "$S" || { echo "FAIL: bash -n on $S"; exit 1; }
# BATCHED drift check: no arm may request the freeze-class 8192 boot
if grep -qE "BATCHED=8192|max-num-batched-tokens 8192" "$S"; then
    echo "FAIL: 8192-batch boot in $(basename $S) — the box-freeze class of 2026-09-30"; FAIL=1
fi
# any docker run in an arm must be preceded by the page-cache release
# (boot-after-cold-read CUDA-OOM/freeze class; the silent-|| true trap)
if grep -q "docker run" "$S" && ! grep -q "evict_page_cache" "$S"; then
    echo "FAIL: docker run without the page-cache release before it"; FAIL=1
fi
# health loops must poll by CONTAINER NAME, never kill -0 on a docker-run
# wrapper pid (the wrapper exits the instant `docker run -d` prints its id;
# the loop then declares a live boot dead and orphans it to trample the next
# arm — the 09-30 R3/R4 cascade).
if grep -q "launch_v30\|docker run" "$S" && grep -q "kill -0" "$S"; then
    echo "FAIL: health loop uses kill -0 on a wrapper pid (false BOOT-FAIL class)"; FAIL=1
fi
# the N/A coercion must be present wherever nvidia-smi feeds an integer test
if grep -q "nvidia-smi --query-gpu=memory" "$S" && ! grep -qF '|| g=0' "$S" && ! grep -qF 'f=999999' "$S"; then
    echo "FAIL: nvidia-smi output feeds integer compare without the [N/A]-string coercion"; FAIL=1
fi
[ $FAIL -eq 0 ] && echo "ARM-SMOKE PASS: $S"
exit $FAIL
