#!/bin/bash
# A3 gap-fill: same engine params as ride_a3gate.sh (114688/2048/KV_FP8/k4),
# append-only runner skips done ids and re-attempts the 27 error rows
# (error ids are NOT in the skip set). Score at the end, fresh engine.
set -uo pipefail
R=$HOME/v30_bench
OV=$HOME/upgrade/v30/overlay
REPO=$HOME/Qwen3.8-Flash-Next-Dual-DGX-Sparks
LAB=$HOME/Qwen38-overthinking-lab
OUT=$HOME/ps_spike/results
NAME=v30a3r; PORT=8903; MODEL=$HOME/models/q38-hyb
export KV_FP8=1 MAXLEN=114688 BATCHED=2048
export PLE_MMAP=$HOME/.cache/v30/ple_mmap_v030
[ -d "$PLE_MMAP" ] || export PLE_MMAP=$HOME/.cache/vllm/ple_mmap_v030
SPEC='{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic","rejection_sample_method":"block","disable_eagle_block_drop":true}'
LOG=$R/ride_a3retry.log; exec >>"$LOG" 2>&1
echo "=== A3 retry start $(date) ==="
while ! mkdir $R/gpu.lock 2>/dev/null; do
  [ -f $R/gpu.lock/owner ] || { sleep 60; continue; }
  . $R/gpu.lock/owner 2>/dev/null || true
  if [ -n "${PID:-}" ] && ! kill -0 $PID 2>/dev/null; then rm -rf $R/gpu.lock; fi
  sleep 60
done
echo "PID=$$" > $R/gpu.lock/owner
grep -q "chain_r complete" $R/chainr.log 2>/dev/null || { echo "guard: chain_r not complete — refusing"; exit 3; }
docker rm -f $NAME >/dev/null 2>&1
bash $OV/launch_v30.sh "$NAME" "$PORT" "$MODEL" --speculative-config "$SPEC"
rc=1
for i in $(seq 1 30); do sleep 60; curl -sf localhost:$PORT/health >/dev/null 2>&1 && { rc=0; break; }; done
[ $rc -eq 0 ] || { echo "BOOT-FAIL $(date +%H:%M)"; docker logs $NAME 2>&1 | tail -30; exit 2; }
echo "HEALTHY $(date +%H:%M)"
cd "$LAB"
python3 scripts/bench_runner.py --port $PORT --dataset datasets/gpqa200.jsonl \
  --out "$OUT/gpqa200__a3-gate.jsonl" --arm "a3-gate" --concurrency 8
python3 scripts/score.py --results "$OUT/gpqa200__a3-gate.jsonl" \
  --dataset datasets/gpqa200.jsonl --out "$OUT/gpqa200__a3-gate.scored.jsonl"
echo "A3-RETRY DONE $(date) rc=$?"
docker rm -f $NAME >/dev/null 2>&1
