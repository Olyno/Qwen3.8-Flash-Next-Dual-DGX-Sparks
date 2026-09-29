#!/bin/bash
# P1 step-floor profile attempt #2. Fixes over #1 (rc=2 WorkerProc, trace lost):
#  - profiler mounts via VLLM_TORCH_PROFILER_DIR (launch_v30 PROF_ARGS: -e + -v $DIR:/prof)
#  - full `docker logs` archived to file the moment boot fails (auto-remove ate them)
#  - rc=0: probe /start_profile existence before trusting an empty trace
set -uo pipefail
R=$HOME/v30_bench; OV=$HOME/upgrade/v30/overlay; MODEL=$HOME/models/q38-lean-hyb
NAME=v30p1b; PORT=8896; PROF=$R/p1b2_prof; mkdir -p $PROF
echo "P1B2 boot $(date +%H:%M)" >> $R/p1b2_boot.txt
VLLM_TORCH_PROFILER_DIR=$PROF K=6 bash $OV/launch_v30.sh "$NAME" "$PORT" "$MODEL" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic","rejection_sample_method":"block","disable_eagle_block_drop":true}' \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/prof"}' >> $R/p1b2_boot.txt 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  docker logs $NAME > $R/p1b2_dockerlogs_FAIL.txt 2>&1 || echo "container gone, no logs" >> $R/p1b2_boot.txt
  echo "P1B2 BOOT-FAIL rc=$RC $(date +%H:%M)" >> $R/p1b2_boot.txt; exit 2
fi
for i in $(seq 1 40); do
  sleep 60
  curl -s -m 5 localhost:$PORT/health >/dev/null 2>&1 && { echo "HEALTHY after ${i} min" >> $R/p1b2_boot.txt; break; }
  docker ps --format '{{.Names}}' | grep -qx $NAME || { docker logs $NAME > $R/p1b2_dockerlogs_FAIL.txt 2>&1; echo "P1B2 DIED ${i} min rc-boot0 $(date +%H:%M)" >> $R/p1b2_boot.txt; exit 3; }
done
curl -s -m 10 -o $R/p1b2_startprof.json -w '%{http_code}' -X POST localhost:$PORT/start_profile >> $R/p1b2_boot.txt 2>&1
echo " start_profile $(date +%H:%M)" >> $R/p1b2_boot.txt
python3 $HOME/fork/bench/decodebench.py --decode 300 --contexts 1000 --temps 0.6 > $R/p1b2_bench.txt 2>&1
curl -s -m 10 -X POST localhost:$PORT/stop_profile >/dev/null 2>&1; sleep 25
echo "prof dir:" >> $R/p1b2_boot.txt; du -sh $PROF >> $R/p1b2_boot.txt 2>&1
find $PROF -name '*.trace.json.gz' -o -name '*.pt.trace.json' | head -4 >> $R/p1b2_boot.txt
python3 $HOME/fork/experiments/v30/p1_analyze.py $PROF > $R/p1_verdict.txt 2>&1
docker rm -f $NAME >/dev/null 2>&1
echo "P1B2 DONE $(date +%H:%M)" >> $R/p1b2_boot.txt
