#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Draft-acceptance sweep: batch-1 decode runs, one metrics delta per cell.

Answers the drafter-selection question directly — tokens per verification
step (tau = 1 + accepted/drafts) on genuine prose, code, and high-entropy
reasoning-style text, at greedy temperature with ignore_eos so every cell
decodes exactly the same number of tokens:

  python3 bench/acceptance_sweep.py --tag mtp3 --repeats 3 --out logs/acc_mtp3.jsonl

Complements sweep.py (sparkDash-driven, aggregate tok/s): this script owns
the per-task acceptance fraction, which is what a kill-gate decision turns
on. Requires an idle server on :8888 (same counter-hygiene rule as sweep.py).
"""
import argparse, json, re, time, urllib.request

METRICS = "http://localhost:8888/metrics"
CHAT = "http://localhost:8888/v1/chat/completions"
WANTED = ("vllm:spec_decode_num_drafts_total",
          "vllm:spec_decode_num_accepted_tokens_total",
          "vllm:spec_decode_num_draft_tokens_total")
LINE = re.compile(r"^([a-z_:]+)\{([^}]*)\}\s+([0-9.eE+-]+)$")

# Same content types as decodebench.py (prose/code/entropy) plus the two
# reasoning shapes the target traffic actually has.
TASKS = {
    "prose":   "Write a flowing, continuous essay about the history of maritime "
               "navigation. Use ordinary narrative prose, no lists, no headings.",
    "code":    "Write a complete, heavily-commented Python implementation of a "
               "red-black tree with insert, delete and search.",
    "reason":  "A farmer has 17 sheep. All but 9 die. He buys goats equal to twice "
               "the surviving sheep, then sells one third of all his animals. Reason "
               "step by step about what the wording means and compute the result.",
    "entropy": "Output a long list of random 12-character uppercase alphanumeric "
               "license keys, one per line, all different, no commentary.",
}


def scrape():
    body = urllib.request.urlopen(METRICS, timeout=20).read().decode()
    out = {}
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        m = LINE.match(line.strip())
        if m and m.group(1) in WANTED:
            pos = re.search(r'position="(\d+)"', m.group(2))
            key = f"{m.group(1)}[{pos.group(1)}]" if pos else m.group(1)
            out[key] = out.get(key, 0.0) + float(m.group(3))
    return out


def decode(prompt, n, model):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": n, "min_tokens": n, "ignore_eos": True,
               "temperature": 0.0, "stream": True,
               "stream_options": {"include_usage": True}}
    req = urllib.request.Request(CHAT, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = t_last = usage = None
    for raw in urllib.request.urlopen(req, timeout=3600):
        s = raw.decode().strip()
        if not s.startswith("data: "):
            continue
        d = s[6:]
        if d == "[DONE]":
            break
        o = json.loads(d)
        if o.get("usage"):
            usage = o["usage"]
        for ch in o.get("choices", []):
            if ch.get("delta") is None:
                continue
            if ttft is None:
                ttft = time.time() - t0
            t_last = time.time()
    ctok = (usage or {}).get("completion_tokens", n)
    win = (t_last - (t0 + ttft)) if (t_last and ttft) else None
    return ctok, ((ctok - 1) / win if win and win > 0 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="configuration label, e.g. mtp3")
    ap.add_argument("--tasks", default="prose,code,reason,entropy")
    ap.add_argument("--decode", type=int, default=600)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--out", default="logs/acceptance.jsonl")
    a = ap.parse_args()
    rows = []
    for task in a.tasks.split(","):
        for rep in range(a.repeats):
            before = scrape()
            ctok, tps = decode(TASKS[task], a.decode, a.model)
            after = scrape()
            d = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
            drafts = d.get("vllm:spec_decode_num_drafts_total", 0.0)
            acc = d.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
            dtok = d.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
            row = {"tag": a.tag, "task": task, "rep": rep, "tok_s": round(tps, 2),
                   "decode_tokens": ctok, "drafts": drafts,
                   "tau": round(1 + acc / drafts, 3) if drafts else None,
                   "acceptance": round(acc / dtok, 4) if dtok else None}
            print(f"{a.tag:>10} {task:<8} rep{rep}  {tps:6.1f} tok/s  "
                  f"tau={row['tau']}  acc={row['acceptance']}", flush=True)
            rows.append(row)
            time.sleep(2)
    with open(a.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


main()
