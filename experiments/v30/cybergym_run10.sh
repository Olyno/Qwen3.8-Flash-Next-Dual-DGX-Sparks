#!/bin/bash
# CyberGym 10-task official subset eval for our endpoint. Pre-gen (CPU) runs
# NOW so GPU release only pays the agent loop. Server: msi 172.17.0.1:8666
# (binary-only mode; started + E2E-proven 00:4x, see tasks/CYBERGYM.md).
# The model endpoint is passed in (default: prod stack 8888 once the queue
# hands over the GPU).
set -uo pipefail
CG=$HOME/agentbench/cybergym
VENV=$HOME/agentbench/cgvenv/bin/python
API=${API:-http://127.0.0.1:8888}
OUT=$HOME/v30_bench/cybergym
mkdir -p $OUT/tasks $OUT/runs
TASKS="arvo:47101 arvo:3938 arvo:24993 arvo:1065 arvo:10400 arvo:368 oss-fuzz:42535201 oss-fuzz:42535468 oss-fuzz:370689421 oss-fuzz:385167047"
for t in $TASKS; do
    d=$OUT/tasks/${t/:/_}
    [ -d "$d" ] || (cd $CG && $VENV -m cybergym.task.gen_task --task-id "$t" \
        --out-dir "$d" --data-dir $CG/cybergym_data/data \
        --server http://172.17.0.1:8666 --mask-map mask_map.json --difficulty level1) \
        && echo "GEN $t ok"
done
for t in $TASKS; do
    d=$OUT/tasks/${t/:/_}
    echo "--- $t $(date +%H:%M) ---"
    timeout 3000 $VENV $HOME/fork/experiments/v30/cybergym_agent.py \
        --task-dir "$d" --base-url "$API" --model qwen3.8-flash-next \
        --iters 12 --out $OUT/runs/${t/:/_}.json 2>&1 | tail -1
done
# Each gen_task assigns its own agent_id; score by walking the db's distinct
# agents through the official verifier (vul-crashes & fix-doesn't = solved).
python3 -c "import sqlite3; db=sqlite3.connect('$CG/server_poc/poc.db'); [print(r[0]) for r in db.execute('SELECT DISTINCT agent_id FROM submissions')]" > $OUT/runs/agents.txt 2>/dev/null
while read -r AID; do
    $VENV $CG/scripts/verify_agent_result.py --server http://172.17.0.1:8666 --pocdb_path $CG/server_poc/poc.db --agent_id $AID
done < $OUT/runs/agents.txt > $OUT/runs/verify.txt 2>&1
echo "verified rows: $(grep -c poc_id $OUT/runs/verify.txt)"; echo "CYBERGYM10 DONE $(date)"
