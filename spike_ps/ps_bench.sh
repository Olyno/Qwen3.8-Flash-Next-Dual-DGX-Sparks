#!/usr/bin/env bash
# ps_bench.sh <port> <tag>  — decode speed for one expert-count config (TP1, no drafter).
set -euo pipefail
PORT=$1; TAG=$2
REPO=$HOME/Qwen3.8-Flash-Next-Dual-DGX-Sparks
OUT=$HOME/ps_spike/results
mkdir -p "$OUT"
# two repeats, short warmup pass discarded by decodebench's median of repeats?
# decodebench prints per-cell tok/s; run it twice at ctx 1k/100k, prose/code/entropy
for pass in 1 2; do
  BENCH_PORT=$PORT python3 "$REPO/bench/decodebench.py" \
    --decode 600 --contexts 1000,100000 --temps 0.6 \
    | tee "$OUT/${TAG}_pass${pass}.txt"
done
