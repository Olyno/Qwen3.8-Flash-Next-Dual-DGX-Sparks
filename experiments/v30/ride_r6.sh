#!/bin/bash
# R6: DeepGEMM correctness A/B on GB10. Boot log proves "Using DEEPGEMM Fp8
# MoE backend" for the FP8_BLOCK_SCALES MTP experts; DeepGEMM#417 reports the
# pure-fp8 SM12x kernels were removed (fp8xfp8 misreads weights, silent).
# If corrupt, disabling DeepGEMM changes outputs -> correctness fix + the
# gate's -1.5pp finding may partly live here.
set -uo pipefail
R=$HOME/v30_bench; OV=$HOME/upgrade/v30/overlay; MODEL=$HOME/models/q38-lean-hyb
PORT=8904
LOG=$R/ride_r6.log; exec >>"$LOG" 2>&1
echo "=== R6 start $(date) ==="
# CHAIN_HELD=1 is exported by chain_r2 run(); standalone invocations must
# refuse while any chain process is alive (children run AS the chain).
[ "${CHAIN_HELD:-0}" = 1 ] || { pgrep -f "[c]hain_r" >/dev/null && { echo "guard: chain busy"; exit 3; }; }
run() {  # $1=env assignment (may be empty), $2=tag
  local NAME=v30r6$2 DG_OPT=()
  docker rm -f $NAME >/dev/null 2>&1
  if [ -n "$1" ]; then DG_OPT=(-e "$1"); fi
  env KV_FP8=1 MAXLEN=131072 BATCHED=2048 PLE_MMAP=$HOME/.cache/vllm/ple_mmap_v030 DG_EXTRA="${1:-}" bash -c '
    if [ -n "$DG_EXTRA" ]; then export $DG_EXTRA; fi
    KV_FP8=1 MAXLEN=131072 BATCHED=2048 K=6 bash '"$OV"'/launch_v30.sh '"$NAME"' '"$PORT"' '"$MODEL"' --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":4,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"disable_eagle_block_drop\":true}"' &
  local LP=$! ok=0
  for i in $(seq 1 60); do sleep 60; curl -s -m 5 localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; done
  if [ $ok -ne 1 ]; then
    docker logs $NAME 2>&1 | tail -20; docker rm -f $NAME >/dev/null 2>&1
    kill $LP 2>/dev/null; return 2
  fi
  docker logs $NAME 2>&1 | grep -iE "DEEPGEMM|deep_gemm|Fp8 MoE" | head -4
  python3 $HOME/fork/experiments/v30/r6_probe.py $PORT $R/r6_$2.jsonl || return 3
  docker rm -f $NAME >/dev/null 2>&1
}
run "VLLM_USE_DEEP_GEMM=1" dg1
RC1=$?
run "VLLM_USE_DEEP_GEMM=0" dg0
RC0=$?
echo "R6 boots: dg1 rc=$RC1 dg0 rc=$RC0"
if [ $RC1 -eq 0 ] && [ $RC0 -eq 0 ]; then
  python3 - "$R" <<'VPEOF'
import json, sys
R = sys.argv[1]
a = [json.loads(l) for l in open(f"{R}/r6_dg1.jsonl")]
b = [json.loads(l) for l in open(f"{R}/r6_dg0.jsonl")]
same = sum(1 for x, y in zip(a, b) if x["text"] == y["text"])
dl = [abs(x["lp"] - y["lp"]) for x, y in zip(a, b)
      if x.get("lp") is not None and y.get("lp") is not None]
v = f"R6-VERDICT: text-identical {same}/{len(a)}; logprob max|Δ| {max(dl) if dl else 'na'}"
print(v)
open(f"{R}/r6_dg_verdict.txt", "w").write(v + "\n")
VPEOF
fi
echo "R6 DONE $(date)"
