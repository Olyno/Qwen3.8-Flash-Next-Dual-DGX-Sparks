#!/usr/bin/env python3
"""gate_reanchor.py — the GPQA gate verdict recomputed on the common-answer
anchor set, robust to reruns. Reads scored JSONLs (one per arm), dedupes by
question id keeping the LAST non-error row (append-only gap-fill contract),
intersects ids answered by ALL named arms, and reports per-arm accuracy on
the common set + the raw (deduped-valid) rate + a McNemar discordance line
for the first two arms. Banked reference (09-29 13:30): anchor n=163,
baked 143, stock-engine 138, lean-engine 138 => engine shift, not bake.

Usage: python3 tools/gate_reanchor.py A.jsonl B.jsonl [C.jsonl ...]
Names for the report fall back to basename. Requires the `correct` field
produced by the lab scorer (bool) and `error` absent/null for a valid row.
"""
import json, os, sys
from collections import OrderedDict

def load(path):
    by_id = OrderedDict()
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        qid = r.get("id", r.get("question_id"))
        if qid is None:
            continue
        err = r.get("error")
        if err:
            by_id.setdefault(qid, None)          # remember it existed
            continue
        by_id[qid] = bool(r.get("correct"))      # later valid rows overwrite
    return {k: v for k, v in by_id.items() if v is not None}

def main(paths):
    arms = [(os.path.basename(p).split("__")[-1].split(".")[0], load(p)) for p in paths]
    common = set(arms[0][1])
    for _, d in arms[1:]:
        common &= set(d)
    print(f"arms: {[a for a,_ in arms]}  common-answer n={len(common)}")
    for name, d in arms:
        c = sum(1 for i in common if d[i])
        print(f"  {name:>12}: {c}/{len(common)} = {100*c/len(common):.2f} %   (own valid n={len(d)})")
    if len(arms) >= 2:
        (na, da), (nb, db) = arms[0], arms[1]
        b_only = sum(1 for i in common if da[i] and not db[i])
        a_only = sum(1 for i in common if db[i] and not da[i])
        print(f"  McNemar {na}-vs-{nb}: {b_only} vs {a_only} discordant "
              f"(two-sided binomial p approx: {binom_p(b_only + a_only, max(b_only, a_only)):.3f})")

def binom_p(n, k):
    # exact two-sided sign-test p for the discordant split
    if n == 0:
        return 1.0
    from math import comb
    tail = sum(comb(n, i) for i in range(0, n - k + 1)) / 2**n  # P(X <= n-k) = P(X >= k)
    return min(1.0, 2 * tail)

if __name__ == "__main__":
    main(sys.argv[1:])
