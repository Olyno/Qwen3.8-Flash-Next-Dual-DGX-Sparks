#!/bin/bash
# R8 — decode-step anatomy (op-level): which kernel runs each skinny GEMM on
# GB10, and what fraction of the step is cuBLAS fallback vs CuTe vs Triton.
# Settles the Hunt-6 kernel levers (SM121 plan tables / FP8 indexer dot / QSA
# prepare fusion) with ONE measurement instead of three guesses.
# Mechanism: launch_v30's VLLM_TORCH_PROFILER_DIR passthrough (mounts /prof);
# boot = prod stack exactly (fp8-KV, mmap PLE, k=4, 2048-batch, FULL graphs);
# drive ~300 decode tokens at c=1 (the product path), dump via /start_profile
# +st /stop_profile, then aggregate host-side with key_averages().
# Adopt-bands pre-registered: a lever is ARM-WORTHY only if its op class is
# >=8 % of the GPU step (below that, upstream kernel wins cannot close the
# gap to 300 tok/s anyway — bandwidth says so: 30.5 tok/s already moves
# ~27 G/token through LPDDR5X).
set -uo pipefail
R=$HOME/v30_bench; mkdir -p $R
LOG=$R/ride_r8.log; : > "$LOG"
MODEL=$HOME/models/q38-lean-hyb
NAME=v30r8; PORT=8905
PROF=$R/r8_prof; rm -rf $PROF; mkdir -p $PROF
if [ -z "${CHAIN_HELD:-}" ]; then
    while ! mkdir $R/gpu.lock 2>/dev/null; do sleep 60; done
    echo "PID=$$" > $R/gpu.lock/owner; trap 'rm -rf $R/gpu.lock' EXIT
fi
echo "R8 START $(date +%H:%M)" >>"$LOG"
docker rm -f $NAME >/dev/null 2>&1
python3 ~/fork/files/evict_page_cache.py "$MODEL" >/dev/null 2>&1 || true
export K=6 MAXLEN=131072 KV_FP8=1 BATCHED=2048 PLE_MMAP=$HOME/.cache/vllm/ple_mmap_v030 OV=$HOME/upgrade/v30/overlay VLLM_TORCH_PROFILER_DIR=$PROF
bash $HOME/fork/experiments/v30/launch_v30.sh $NAME $PORT $MODEL \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic","rejection_sample_method":"block","disable_eagle_block_drop":true}' >>"$LOG" 2>&1 || {
    echo "RUN-FAIL $(date +%H:%M)" >>"$LOG"; docker logs $NAME 2>&1 | tail -30 > $R/r8_FAIL.txt; docker rm -f $NAME >/dev/null 2>&1; exit 2; }
ok=0
for i in $(seq 1 60); do sleep 30; curl -s -m 5 localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; docker ps --format '{{.Names}}' | grep -qx $NAME || break; done
[ $ok -eq 1 ] || { echo "BOOT-FAIL $(date +%H:%M)" >>"$LOG"; docker logs $NAME 2>&1 | tail -40 > $R/r8_FAIL.txt; docker rm -f $NAME >/dev/null 2>&1; exit 2; }
echo "HEALTHY $(date +%H:%M)" >>"$LOG"
# warm one turn (JIT + autotune out of the capture), then profile a clean c=1 decode
curl -s -m 300 localhost:$PORT/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Brief warmup sentence."}],"max_tokens":50}' >/dev/null
curl -s -m 20 localhost:$PORT/start_profile >/dev/null
BENCH_PORT=$PORT timeout 900 python3 $HOME/fork/bench/decodebench.py --decode 300 --contexts 1000 --temps 0.6 --tasks prose > $R/r8_bench_under_profile.txt 2>&1 || echo "PROBE-FAIL" >>"$LOG"
curl -s -m 60 localhost:$PORT/stop_profile >/dev/null
sleep 15
docker cp $NAME:/prof $R/r8_trace 2>/dev/null || cp -r $PROF $R/r8_trace 2>/dev/null
find $R/r8_trace -type f 2>/dev/null | head -5 > $R/r8_trace_files.txt
python3 $HOME/fork/experiments/v30/r8_analyze.py $R/r8_trace > $R/r8_verdict.txt 2>&1 || echo "ANALYZE-PENDING (writer in tree; trace files above)" >> $R/r8_verdict.txt
docker rm -f $NAME >/dev/null 2>&1
echo "R8 DONE $(date +%H:%M)" >>"$LOG"
