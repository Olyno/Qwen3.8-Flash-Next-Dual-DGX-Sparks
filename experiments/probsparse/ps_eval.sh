#!/usr/bin/env bash
# ps_eval.sh <port> <tag>  — quality gate: GPQA-198 + MATH-500 + GSM8K-100, bake-matrix protocol
# (temp 0.6 top-p 0.95 seed 1337, same runner/score scripts as the lean study => comparable rows).
set -euo pipefail
PORT=$1; TAG=$2
LAB=$HOME/Qwen38-overthinking-lab
OUT=$HOME/ps_spike/results
mkdir -p "$OUT"
cd "$LAB"
for ds in gpqa200 math500 gsm8k100; do
  python3 scripts/bench_runner.py --port "$PORT" --dataset datasets/$ds.jsonl \
    --out "$OUT/${ds}__${TAG}.jsonl" --arm "$TAG" --concurrency 8
  python3 scripts/score.py --results "$OUT/${ds}__${TAG}.jsonl" \
    --dataset datasets/$ds.jsonl --out "$OUT/${ds}__${TAG}.scored.jsonl"
done
