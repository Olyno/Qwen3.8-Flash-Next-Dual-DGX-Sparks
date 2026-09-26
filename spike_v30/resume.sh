#!/bin/bash
# resume.sh <phase> — msi recovery driver, ONE phase per invocation (post-mortem
# lesson 2026-09-26: no self-chaining poison boots; first boot of each new recipe
# is supervised, identical-failure signature stops the line).
# Phases:
#   A1  v0.30 stock K10 boot + decode bench   (the boot that died 4x; fix set applied)
#   A2  A1 + MTP3             bench row: first-ever MTP speed on this box, v0.30
#   A3  v0.30 + hybrid ckpt   bench row: v0.30-native modelopt (no stacker)
#   B1  old image + hybrid    QUALITY GATE (GPQA/MATH/GSM8K, ~9h) — old-image boot
#       is proven-good (hyb_old speed run completed on it), so quality may run
#       independently of A1..A3; it is the top-priority phase for the verdict.
# Heartbeat every 5 min; exit codes: 0 done, 2 OOM-signature ABORT, 3 timeout-no-boot.
set -uo pipefail
R=$HOME/v30_bench; mkdir -p "$R"
LOG=$R/resume_$1.log; exec >>"$LOG" 2>&1
OV=$HOME/upgrade/v30/overlay
LAB=$HOME/Qwen38-overthinking-lab
REPO=$HOME/Qwen3.8-Flash-Next-Dual-DGX-Sparks
PH=$1
echo "=== resume phase $PH start $(date) ==="

evict() { python3 "$REPO/files/evict_page_cache.py" "$1" >/dev/null 2>&1 || true; }

hb() { # name — heartbeat + death watch (signals via marker file; a subshell
    # exit cannot stop the main script, and the lesson of today is that
    # polling a dead boot for an hour is exactly what must not happen again)
    local name=$1 n=0
    rm -f "$R/DEAD_$PH"
    while sleep 120; do n=$((n+1))
        echo "hb $name +$((n*2))m $(date +%H:%M) $(docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' "$name" 2>/dev/null || echo gone)"
        if ! docker ps -q -f name="^$name$" | grep -q .; then
            echo "!! container $name exited — dumping root cause"
            docker logs --tail 80 "$name" 2>&1 | grep -iE "error|kill|oom|137|exception|CUDA" | tail -20
            echo "!! free:"; free -g | head -2
            touch "$R/DEAD_$PH"; exit 2
        fi
    done
}

wait_health() { # 90-min window; abort early on DEAD marker
    for _ in $(seq 360); do
        [ -f "$R/DEAD_$PH" ] && return 2
        curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 15
    done
    return 1
}

bench() { BENCH_PORT=$PORT python3 "$REPO/bench/decodebench.py" --decode 600 --contexts 1000,100000 --temps 0.6 > "$R/$1_pass1.txt" 2>&1; }
conc() { python3 $R/concbench.py --port $PORT --levels 1,4,8,16,32 --decode 400 > "$R/$1_conc.txt" 2>&1; }

case "$PH" in
A1) NAME=v30a1; PORT=8892; MODEL=$HOME/models/Qwen3.8-Flash-Next-NVFP4-wk1
    evict "$MODEL"
    K=6 bash $OV/launch_v30.sh "$NAME" $PORT "$MODEL"
    hb "$NAME" & HP=$!
    wait_health; RC=$?
    kill $HP 2>/dev/null
    if [ $RC -eq 2 ]; then echo "ABORT: boot died (see root cause above) $(date +%H:%M)"; exit 2; fi
    if [ $RC -ne 0 ]; then echo "FAIL: boot timeout"; docker logs --tail 40 "$NAME" 2>&1; exit 3; fi
    echo "HEALTHY $(date +%H:%M)"; bench v30_k6; conc v30_k6; echo "BENCH-OK"
    docker rm -f "$NAME" ;;
A2) NAME=v30a2; PORT=8892; MODEL=$HOME/models/Qwen3.8-Flash-Next-NVFP4-wk1
    evict "$MODEL"
    K=6 bash $OV/launch_v30.sh "$NAME" $PORT "$MODEL" --speculative-config '{"method":"mtp","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","disable_eagle_block_drop":true}'
    hb "$NAME" & HP=$!
    wait_health; RC=$?
    kill $HP 2>/dev/null
    if [ $RC -eq 2 ]; then echo "ABORT: boot died (see root cause above) $(date +%H:%M)"; exit 2; fi
    if [ $RC -ne 0 ]; then echo "FAIL: boot timeout"; docker logs --tail 40 "$NAME" 2>&1; exit 3; fi
    echo "HEALTHY $(date +%H:%M)"; bench v30_k6_mtp5pb; conc v30_k6_mtp5pb; echo "BENCH-OK"
    docker rm -f "$NAME" ;;
A3) NAME=v30a3; PORT=8892; MODEL=$HOME/models/q38-hyb
    evict "$MODEL"
    K=6 bash $OV/launch_v30.sh "$NAME" $PORT "$MODEL" --speculative-config '{"method":"mtp","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","disable_eagle_block_drop":true}'
    hb "$NAME" & HP=$!
    wait_health; RC=$?
    kill $HP 2>/dev/null
    if [ $RC -eq 2 ]; then echo "ABORT: boot died (see root cause above) $(date +%H:%M)"; exit 2; fi
    if [ $RC -ne 0 ]; then echo "FAIL: boot timeout"; docker logs --tail 40 "$NAME" 2>&1; exit 3; fi
    echo "HEALTHY $(date +%H:%M)"; bench hyb_v30_mtp5pb; conc hyb_v30_mtp5pb; echo "BENCH-OK"
    docker rm -f "$NAME" ;;
P1) NAME=v30p1; PORT=8892; MODEL=$HOME/models/Qwen3.8-Flash-Next-NVFP4-wk1
    PROF=$HOME/v30_bench/prof; mkdir -p "$PROF"; rm -f "$PROF"/* 2>/dev/null
    evict "$MODEL"
    VLLM_TORCH_PROFILER_DIR="$PROF" K=6 bash $OV/launch_v30.sh "$NAME" $PORT "$MODEL"
    hb "$NAME" & HP=$!
    wait_health; RC=$?
    kill $HP 2>/dev/null
    if [ $RC -ne 0 ]; then echo "P1 BOOT-FAIL rc=$RC"; exit 2; fi
    echo "P1 healthy $(date +%H:%M) — warm then profile"
    BENCH_PORT=$PORT timeout 300 python3 "$REPO/bench/decodebench.py" --decode 120 --contexts 1000 --temps 0.6 >/dev/null 2>&1
    curl -s -X POST localhost:$PORT/start_profile
    BENCH_PORT=$PORT timeout 300 python3 "$REPO/bench/decodebench.py" --decode 200 --contexts 1000 --temps 0.6 > "$R/p1_profiled.txt" 2>&1
    curl -s -X POST localhost:$PORT/stop_profile
    sleep 25
    docker rm -f "$NAME" >/dev/null 2>&1
    du -sh "$PROF"; ls "$PROF" | head -4
    tar -C "$PROF" -czf "$HOME/v30_bench/p1_trace.tgz" . && echo P1-TAR-OK ;;
T1) # fixed-K sweep + telemetry: ONE boot per K, kill-test data for the whole
    # MoE-cost-aware family (Limits 2609.22156), per-position acceptance for
    # EVICT's rule replay, and state-page usage for the spec-tax question.
    for KK in 1 2 3 5 7; do
      NAME=v30t$KK; PORT=8892; MODEL=$HOME/models/Qwen3.8-Flash-Next-NVFP4-wk1
      evict "$MODEL"
      K=6 bash $OV/launch_v30.sh "$NAME" $PORT "$MODEL" --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$KK,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"disable_eagle_block_drop\":true}" --enable-return-routed-experts
      hb "$NAME" & HP=$!
      wait_health; RC=$?
      kill $HP 2>/dev/null
      if [ $RC -ne 0 ]; then echo "T1 K=$KK BOOT-FAIL rc=$RC — stopping sweep"; exit 2; fi
      echo "HEALTHY K=$KK $(date +%H:%M)"; bench "t1_k$KK"; conc "t1_k$KK"
      curl -s localhost:$PORT/metrics > "$R/t1_k$KK_metrics.txt" 2>/dev/null
      docker rm -f "$NAME" >/dev/null 2>&1
    done
    python3 "$REPO/spike_v30/t1_analyze.py" "$R" > "$R/t1_verdict.txt" 2>&1 || echo T1-ANALYZE-FAILED >> "$R/t1_verdict.txt"; echo T1-DONE ;;
B1) NAME=hybq; PORT=8891; MODEL=$HOME/models/q38-hyb
    evict "$MODEL"
    bash ~/hyb_spike/hyb_launch.sh "$NAME" $PORT "$MODEL"
    hb "$NAME" & HP=$!
    wait_health; RC=$?
    if [ $RC -eq 2 ]; then kill $HP 2>/dev/null; echo "ABORT: boot died" ; exit 2; fi
    if [ $RC -ne 0 ]; then kill $HP 2>/dev/null; echo "FAIL: boot timeout"; docker logs --tail 40 "$NAME" 2>&1; exit 3; fi
    echo "HEALTHY $(date +%H:%M), starting quality gate (~9h, server stays up = hb keeps watching)"
    bash ~/ps_spike/ps_eval.sh $PORT hyb
    RC=$?; kill $HP 2>/dev/null; echo "QUALITY-EXIT $RC"; [ $RC -eq 0 ] && docker rm -f "$NAME" ;;
*) echo "unknown phase"; exit 1 ;;
esac
echo "=== resume phase $PH complete $(date) ==="
