#!/usr/bin/env bash
# ps_bench.sh <port> <tag>  — decode speed for one expert-count config (TP1, no drafter).
# Single pass, same protocol as the day-1 spike (greedy-adjacent temp 0.6, 600 tokens).
set -euo pipefail
PORT=$1; TAG=$2
REPO=$HOME/Qwen3.8-Flash-Next-Dual-DGX-Sparks
OUT=$HOME/ps_spike/results
mkdir -p "$OUT"
BENCH_PORT=$PORT python3 "$REPO/bench/decodebench.py" \
  --decode 600 --contexts 1000,100000 --temps 0.6 \
  | tee "$OUT/${TAG}_pass1.txt"
