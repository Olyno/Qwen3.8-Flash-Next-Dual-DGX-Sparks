#!/usr/bin/env python3
"""r_analyze.py — verdict math for the post-gate R arms (R1..R5, P1b2).

Reads ~/v30_bench (pass1 tables, metrics dumps, census JSONL, ride logs) and
prints one line per arm with the pre-registered decision band applied.
Banks the A1/A4 reference rows so the comparison is one command.

Usage: python3 experiments/v30/r_analyze.py [R]     (R default $HOME/v30_bench)
"""
import json
import os
import re
import sys

R = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/v30_bench")

# banked references (docs/verdicts/V30-LANE.md; A4 at BATCHED=2048!)
A4 = {"prose": 30.5, "code": 35.5}   # prod stack, k=4, 2048-budget rows
A1 = {"prose": 17.1, "code": 17.5}   # v0.30 stock floor

ROW = re.compile(r"^\s+([\d,]+)\s+[\d.]+\s+(\w+)\s+[\d,]+\s+[\d,]+\s+([\d.]+)\s+([\d.]+)")


def pass1(path):
    """decodebench table -> {(ctx,class): tps}."""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        m = ROW.match(line)
        if m:
            ctx, klass, ttft, tps = m.groups()
            out[(ctx.replace(",", ""), klass)] = (float(ttft), float(tps))
    return out


def metrics_pos(path):
    """/metrics dump -> per-position acceptance ratios + mean al."""
    if not os.path.exists(path):
        return None
    acc = {}
    fam = {
        "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
        "drafted": "vllm:spec_decode_num_drafts_total",
        "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    }
    pos = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        m = re.match(r'vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"\}\s+([\d.e+]+)', line)
        if m:
            pos[int(m.group(1))] = float(m.group(2))
        for k, name in fam.items():
            if line.startswith(name + "{"):
                acc[k] = acc.get(k, 0) + float(line.rsplit(" ", 1)[1])
    if not pos:
        return None
    drafts = acc.get("drafted", 0) or 1
    q = [pos.get(i, 0.0) / drafts for i in range(4)]
    tau = 1 + sum(q)
    return {"q": [round(x, 3) for x in q], "tau": round(tau, 2),
            "accept_rate": round(acc.get("accepted_tokens", 0) / max(acc.get("draft_tokens", 1), 1), 3)}


def census(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def main():
    print("== R1: reactivity evidence (JIT cold vs warm) ==")
    log = os.path.join(R, "ride_r1.log")
    if os.path.exists(log):
        for pat in ("JIT-WARNINGS-AFTER-BOOT", "COLD-JIT-TOTAL", "WARM-JIT-TOTAL", "HEALTHY-AFTER"):
            for line in open(log):
                if pat in line:
                    print("  " + line.strip())
    else:
        print("  pending (ride_r1.log absent)")

    for arm, desc, band in [
        ("r1", "prod-config 8192 baseline row (vs A4@2048): expect parity within ±5 %", None),
        ("r2", "fused draft #58449: expect prose +1.2 % (PR c=1), acceptance ±2 pp", 1.2),
        ("r3", "ITL isolation: expect c=1 delta ~0; read census ttft_med/p90 improvement", None),
        ("r4", "MoE flashinfer_b12x: band open; boot-fail = evidence", None),
        ("r5", "QSA geometry family-match: adopt >±3 % acceptance-flat", 3.0),
    ]:
        print(f"== {arm.upper()}: {desc} ==")
        p = os.path.join(R, f"{arm}_pass1.txt")
        rows = pass1(p)
        if not rows:
            print("  pending")
        for (ctx, klass), (ttft, tps) in sorted(rows.items()):
            if klass in A4:
                d = 100 * (tps / A4[klass] - 1)
                flag = ""
                if band and d >= band:
                    flag = " -> ADOPT-band"
                elif band:
                    flag = f" -> below {band} % band"
                print(f"  {klass:>5}@{ctx}: {tps:6.1f} tok/s (vs A4 {A4[klass]:5.1f}: {d:+5.1f} %) TTFT {ttft:.2f}{flag}")
        m = metrics_pos(os.path.join(R, f"{arm}_metrics.txt"))
        if m:
            print(f"  acceptance: q={m['q']} tau={m['tau']} rate={m['accept_rate']}")
        for f in (f"{arm}_conc_census.txt", f"{arm}_conc_mintok.txt", f"{arm}_conc.txt"):
            c = census(os.path.join(R, f))
            for row in c:
                print(f"  {f}: {row}")
    print("== T1b verdict ==")
    tv = os.path.join(R, "t1_verdict.txt")
    if os.path.exists(tv):
        print("".join("  " + l for l in open(tv).readlines()[-30:]))
    else:
        print("  pending")


if __name__ == "__main__":
    main()
