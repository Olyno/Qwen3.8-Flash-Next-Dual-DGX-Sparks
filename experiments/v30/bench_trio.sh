#!/bin/bash
# The deck-triage top-3 (tasks/BENCHTRIAGE.md): API-only public benchmarks,
# pointed at our OpenAI endpoint. No docker, no judge model (τ² bm25 = offline
# retrieval; AutomationBench scores by assertions; MultiChallenge's judge is
# the one gap — we grade with our own big box later if the run produces
# responses; here we capture responses + the harness's own scoring where it
# exists). Runs DURING the cyber engine boot (c<=2 extra load; scores are
# content-based, timing irrelevant).
set -uo pipefail
API=${API:-http://127.0.0.1:8888/v1}
OUT=$HOME/v30_bench/trio; mkdir -p $OUT
M=qwen3.8-flash-next

echo "=== trio start $(date) ==="
# 1. τ²-Banking (25 tasks, both sides our model — user-sim same model is the
# documented cheap config; bm25 = no embeddings API needed)
cd $HOME/agentbench/tau2-bench
printf 'OPENAI_API_KEY=dummy\nOPENAI_API_BASE=%s\n' "$API" > .env
PATH=$HOME/.local/bin:$PATH timeout 7200 uv run tau2 run \
    --domain banking_knowledge --retrieval-config bm25 \
    --agent-llm openai/$M --user-llm openai/$M \
    --num-trials 1 --num-tasks 25 --max-concurrency 2 \
    > $OUT/tau2.log 2>&1
echo "tau2 rc=$? $(date +%H:%M)"
grep -E "average|reward|pass" $OUT/tau2.log | tail -3 | tee -a $OUT/summary.txt

# 2. AutomationBench public (finance domain, 25 examples, assertion-scored)
cd $HOME/agentbench/AutomationBench
PATH=$HOME/.local/bin:$PATH timeout 7200 uv run auto-bench \
    --model $M --base-url $API --api-key dummy \
    --domains finance --num-examples 25 --max-concurrent 2 --max-steps 50 \
    --export-json $OUT/autobench_f25.json > $OUT/autobench.log 2>&1
echo "autobench rc=$? $(date +%H:%M)"
python3 -c "import json;d=json.load(open('$OUT/autobench_f25.json'));r=[x for x in (d if isinstance(d,list) else d.get('results',[]))];print('autobench pass:',sum(1 for x in r if x.get('passed')),'/',len(r))" 2>&1 | tee -a $OUT/summary.txt

# 3. MultiChallenge via AgentSuite (captures conversations; official judge is
# GPT-based — we run the harness WITHOUT scores, store responses; grading
# later on gx10 by the slow role or an off-the-shelf judge). 25 conversations.
cd $HOME/agentbench/AgentSuite
printf 'API_KEY=dummy\nBASE_URL=%s\n' "$API" > .env
timeout 5400 python run_benchmarks.py $M --benchmark multichallenge --proc-num 2 \
    > $OUT/multichallenge.log 2>&1 || echo "multichallenge rc=$? (likely judge-key needed; responses kept in ./outputs)"
echo "=== trio done $(date) ==="
