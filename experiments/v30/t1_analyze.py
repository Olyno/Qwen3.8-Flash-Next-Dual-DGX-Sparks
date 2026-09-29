#!/usr/bin/env python3
"""t1_analyze.py — verdict math for the T1 fixed-K sweep (spike_v30/resume.sh).

Inputs (written by T1 in $R = ~/v30_bench):
  t1_k<K>_pass1.txt        decodebench output per arm (tok/s rows)
  t1_k<K>_conc.txt         concbench JSONL per arm
  t1_k<K>_metrics.txt      /metrics dump DURING that arm's bench (cumulative —
                           we use per-position acceptance RATIOS only; ratios
                           over one arm are dominated by that arm's traffic)

Computes, per arm K:
  q_i        = per-position acceptance prob (accepted_per_pos[i]/drafts)
  tau        = 1 + sum_i q_i            (accepted tokens per verify step)
  t_step     = tau / tps                (prose tok/s at ctx 1k)
Then:
  1) monotonicity kill-gate: is tps(K) increasing through K=7? If yes the
     MoE verify-cost premise (Limits 2609.22156) does not bind on GB10 ->
     kill the whole cost-aware family.
  2) EVICT offline replay (Variant A, zero vLLM edit): static rule
     k* = argmax_k [1 + sum_{i<=k} cumprod(q)] / t_step(k) using the K=7
     arm's q_i as the shared acceptance model; predicted tok/s vs the K=5
     fixed arm. Kill-gate: uplift < 3 %.
stdlib only. Usage: t1_analyze.py <R>
"""
import json
import re
import statistics
import sys

R = sys.argv[1].rstrip("/")
KS = [1, 2, 3, 5, 7]


def decode_prose(path):
    """decodebench prose tok/s at ctx 1k (first matching row)."""
    try:
        for line in open(path):
            m = re.search(r"1,000\s+0\.6\s+prose.*?([\d.]+)\s*$", line.strip())
            if m:
                return float(m.group(1))
    except FileNotFoundError:
        pass
    return None


def metrics_parse(path):
    """accepted_per_pos buckets, drafts, accepted total."""
    drafts = 0.0
    acc_tot = 0.0
    per_pos = {}
    try:
        for line in open(path):
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if line.startswith("vllm:spec_decode_num_drafts_total"):
                drafts += float(line.rsplit(" ", 1)[-1])
            elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
                acc_tot += float(line.rsplit(" ", 1)[-1])
            elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
                m = re.search(r'position="(\d+)"', line)
                if m:
                    per_pos[int(m.group(1))] = per_pos.get(int(m.group(1)), 0.0) \
                        + float(line.rsplit(" ", 1)[-1])
    except FileNotFoundError:
        pass
    return drafts, acc_tot, per_pos


def main():
    rows = {}
    for k in KS:
        tps = decode_prose(f"{R}/t1_k{k}_pass1.txt")
        drafts, acc_tot, per_pos = metrics_parse(f"{R}/t1_k{k}_metrics.txt")
        conc = None
        try:
            for line in open(f"{R}/t1_k{k}_conc.txt"):
                d = json.loads(line)
                if d.get("C") == 1:
                    conc = d
                    break
        except FileNotFoundError:
            pass
        if tps and drafts and per_pos:
            q = [per_pos.get(i, 0.0) / drafts for i in sorted(per_pos)]
            tau = 1.0 + sum(q)
            rows[k] = {"tps": tps, "q": q, "tau": tau,
                       "t_step": tau / tps, "agg_c1": (conc or {}).get("aggregate")}
        else:
            print(f"WARN arm K={k} incomplete (tps={tps} drafts={drafts} "
                  f"per_pos={len(per_pos)})")
    if not rows:
        sys.exit("NO-USABLE-ARMS")

    ks = sorted(rows)
    print("arm  tps   tau  t_step_ms  q_per_pos")
    for k in ks:
        r = rows[k]
        print(f"K={k:<2} {r['tps']:5.1f} {r['tau']:5.2f} {1000*r['t_step']:7.2f}   "
              + " ".join(f"{x:.2f}" for x in r["q"]))

    # 1) monotonicity kill-gate
    tps_seq = [rows[k]["tps"] for k in ks]
    monotone = all(b >= a * 0.98 for a, b in zip(tps_seq, tps_seq[1:]))
    best_k = max(rows, key=lambda k: rows[k]["tps"])
    print(f"MONOTONE_THROUGH_K7: {monotone} (best fixed K={best_k} @ "
          f"{rows[best_k]['tps']:.1f} tok/s)")
    if monotone:
        print("VERDICT-LIMITS: tok/s rises through K=7 -> MoE verify-cost "
              "premise does not bind on GB10. Kill EVICT/Limits/EcoSpec family.")

    # 2) EVICT static-rule replay from the widest arm's acceptance model
    kmax = ks[-1]
    q = rows[kmax]["q"]
    surv = 1.0
    ea = []          # E[accepted | budget k] = 1 + sum_{i<=k} prod_{j<i} q_j
    cum = 0.0
    for i, qi in enumerate(q):
        cum += surv
        surv *= qi
        ea.append(1.0 + cum)
    print("k    E[A(k)]  t_step(k)ms  rule=EA/tstep (tok/s-equiv)")
    rule_best, rule_val = None, 0.0
    for i, e in enumerate(ea, start=1):
        # cost of budget i: measured arm if available, else interpolate t_step
        arm = min(rows, key=lambda k: abs(k - i))
        ts = rows[arm]["t_step"]
        val = e / ts
        marker = ""
        if val > rule_val:
            rule_val, rule_best = val, i
            marker = " <-max"
        print(f"i={i:<2} {e:6.2f}  {1000*ts:8.2f}      {val:6.1f}{marker}")
    ref = rows.get(5) or rows[best_k]
    uplift = (rule_val - ref["tps"]) / ref["tps"] * 100
    print(f"REPLAY: k*={rule_best} predicted {rule_val:.1f} tok/s vs "
          f"K=5-arm {ref['tps']:.1f} -> uplift {uplift:+.1f} %")
    print("VERDICT-EVICT:", "KILL (<3 %)" if uplift < 3.0 else "PROCEED to Variant B")


if __name__ == "__main__":
    main()
