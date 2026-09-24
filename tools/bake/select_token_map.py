import json, os, statistics, sys
# choose TOKEN_MAP for the bake: refined vs original marker set at chosen penalty strength.
# Pre-registered rule: among {original=pen<lam>, refined=p1ref} runs, qualify = mean
# dacc>=-1.0 pp and worst>=-3.0 pp vs base; among qualifiers pick greater mean token
# reduction; tie -> refined (fewer neutral rows).
# Usage: OVERTHINKING_LAB_DIR=/path/to/results python3 select_token_map.py
# (lab root contains results/chosen_lambda.json and <dataset>__<run>.scored.jsonl files)
L = os.environ.get("OVERTHINKING_LAB_DIR") or sys.exit(
    "set OVERTHINKING_LAB_DIR to the lab root (contains results/)")
def agg(ds, run):
    f = f"{L}/results/arms/{ds}__{run}.scored.jsonl"
    if not os.path.exists(f): return None
    rows = [json.loads(l) for l in open(f) if l.strip()]
    ok = [r for r in rows if r.get("error") is None]
    c = sum(1 for r in ok if r.get("correct"))
    t = [r["completion_tokens"] for r in ok if r.get("completion_tokens")]
    return dict(acc=100*c/max(len(ok),1), tok=statistics.mean(t) if t else 0)
lam = float(json.load(open(f"{L}/results/chosen_lambda.json"))["lambda"])
cands = {}
for name, run in (("original", f"pen{lam}"), ("refined", "p1ref")):
    per = {}
    for ds in ("math500", "gsm8k100", "gpqa200"):
        b, a = agg(ds, "base"), agg(ds, run)
        if not b or not a: per = None; break
        per[ds] = (a["acc"]-b["acc"], 100*(a["tok"]-b["tok"])/max(b["tok"],1))
    cands[name] = per
print(json.dumps(cands, indent=1))
def stats(per):
    daccs = [v[0] for v in per.values()]; dl = [v[1] for v in per.values()]
    return statistics.mean(daccs), min(daccs), statistics.mean(dl)
def qualifies(per):
    m, w, _ = stats(per); return m >= -1.0 and w >= -3.0
scored = {k: v for k, v in cands.items() if v and qualifies(v)}
if not scored:
    print("MAP-CHOICE=original (neither qualified; conservative default)")
else:
    best = min(scored, key=lambda k: stats(scored[k])[2])
    m = stats(scored[best])
    print(f"MAP-CHOICE={best} (mean dacc {m[0]:+.2f}, dtok {m[2]:+.1f}%)")
