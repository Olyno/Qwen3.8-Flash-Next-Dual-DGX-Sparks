#!/usr/bin/env python3
"""Decode-speed benchmark: separates CONTENT TYPE from CONTEXT LENGTH.

Uses ignore_eos so every run decodes exactly N tokens.

  python3 bench/decodebench.py --decode 600 --contexts 1000,48000 --temps 0.0

Every run also decomposes decode speed into its two factors:

  decode tok/s = engine steps/s x tokens/step

tokens/step is 1 + accepted/drafts and steps/s is drafts / decode window,
from vLLM's spec-decode counters snapshotted around the run (same source as
bench/mtp_accept.py). The server must be otherwise idle or the counter deltas
pick up foreign traffic; without the counters the extra columns read "nan".

--conditions myllmbox pins the myllmbox-comparable setup -- thinking off
(chat_template_kwargs enable_thinking=false), temperature 0, and three fixed
prompts (short code-gen, long prose, mixed) -- so numbers line up with
myllmbox's 64.7/86.3/49.8 claims. --dry-run prints the plan without a server.
Endpoint/model come from BENCH_BASE/BENCH_MODEL when set (prod-recipe
defaults); --jsonl appends machine-readable rows for bench/gate_report.sh.

Findings that motivated the shape of this script:
  * decode is dominated by MTP acceptance, not context length
    (1k -> 600k costs only ~2-8%)
  * "copy from context" is the best case (~70 tok/s) because the MTP draft
    head predicts quoted text almost perfectly; genuine prose is ~40 tok/s.
    Do NOT quote a copy-heavy number as typical decode speed.
  * the "entropy" task at temperature 0 degenerates into repetition, which
    MTP then predicts easily -- read it only at temp 0.8.
"""
import json, os, sys, time, argparse, urllib.request

import mtp_accept

BASE = os.environ.get("BENCH_BASE", "http://localhost:8888")
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.8-Flash-Next-NVFP4")
DRAFTS = "vllm:spec_decode_num_drafts_total"
DRAFT_TOK = "vllm:spec_decode_num_draft_tokens_total"
ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
FILLER = ("Entry {i:06d}: the quarterly logistics audit recorded a routine "
          "variance in the northbound depot inventory.\n")
TASKS = {
 "prose":   "Write a flowing, continuous essay about the history of maritime "
            "navigation. Use ordinary narrative prose, no lists, no headings.",
 "code":    "Write a complete, heavily-commented Python implementation of a "
            "red-black tree with insert, delete and search.",
 "entropy": "Output a long list of random 12-character uppercase alphanumeric "
            "license keys, one per line, all different, no commentary.",
 "copy":    "Reproduce verbatim, in order, entries 000005 through 000034 from "
            "the log above. Output the lines exactly as they appear.",
}
# myllmbox generates short, thinking-off, temperature-0 answers; these three
# fixed prompts mirror its short code-gen / long-prose / mixed mix.
MYLLMBOX = {
 "short_code": "Write a Python function that checks whether a string is a palindrome.",
 "long_prose": "Write a detailed, flowing essay about the history of the printing press.",
 "mixed":      "Explain how a hash map resolves collisions, then show a small "
               "Python implementation.",
}

def post(path, payload, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def build_ctx(t):
    return "".join(FILLER.format(i=i) for i in range(max(1, int(t / 25))))

def snapshot():
    try:
        return mtp_accept.scrape(BASE + "/metrics")
    except Exception:
        return None

def run(ctx, task, n, temp, tkwargs=None):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": ctx + "\n\n" + task}],
               "max_tokens": n, "min_tokens": n, "ignore_eos": True,
               "temperature": temp, "stream": True,
               "stream_options": {"include_usage": True}}
    if tkwargs:
        payload["chat_template_kwargs"] = tkwargs
    before = snapshot()
    t0 = time.time(); resp = post("/v1/chat/completions", payload)
    ttft = t_last = usage = None
    for raw in resp:
        s = raw.decode().strip()
        if not s.startswith("data: "): continue
        d = s[6:]
        if d == "[DONE]": break
        o = json.loads(d)
        if o.get("usage"): usage = o["usage"]
        for ch in o.get("choices", []):
            if ch.get("delta") is None: continue
            if ttft is None: ttft = time.time() - t0
            t_last = time.time()
    after = snapshot()
    ctok = (usage or {}).get("completion_tokens", n)
    ptok = (usage or {}).get("prompt_tokens", 0)
    win = (t_last - (t0 + ttft)) if (t_last and ttft) else None
    dec = (ctok - 1) / win if win and win > 0 else float("nan")
    sps = tps = acc = float("nan")
    if before and after and win and win > 0:
        drafts = after.get(DRAFTS, 0) - before.get(DRAFTS, 0)
        atok = after.get(ACCEPTED, 0) - before.get(ACCEPTED, 0)
        dtok = after.get(DRAFT_TOK, 0) - before.get(DRAFT_TOK, 0)
        if drafts > 0:
            sps, tps = drafts / win, 1 + atok / drafts
            acc = 100.0 * atok / dtok if dtok > 0 else float("nan")
    return ptok, ctok, ttft, dec, sps, tps, acc

def max_served_len():
    """max_model_len from /tokenize (falls back to /v1/models), or None."""
    try:
        body = {"model": MODEL, "prompt": "x"}
        req = urllib.request.Request(BASE + "/tokenize", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            got = json.load(r).get("max_model_len")
        if got:
            return int(got)
    except Exception:
        pass
    try:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as r:
            data = json.load(r).get("data") or []
        return int(data[0]["max_model_len"]) if data and data[0].get("max_model_len") else None
    except Exception:
        return None

def count_tokens(text):
    """Token count of `text` from the server, or None when /tokenize is absent."""
    try:
        body = {"model": MODEL, "prompt": text}
        req = urllib.request.Request(BASE + "/tokenize", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(json.load(r)["count"])
    except Exception:
        return None

def shrink_ctx(c, n_tokens, decode, max_len):
    """New nominal context for a prompt that tokenized to `n_tokens`: the
    builders emit ~1.07 tokens per nominal unit, so back off by the overshoot
    with margin and retry."""
    if not max_len or n_tokens + decode <= max_len:
        return c
    over = n_tokens + decode - max_len
    return max(1, c - int(over * 1.1) - 16)

def fit_ctx(c, decode, task, max_len):
    """Build a context for nominal size `c` whose tokens fit `max_len`."""
    if not max_len:
        return c, build_ctx(c)
    ctx = build_ctx(c)
    for _ in range(4):
        n = count_tokens(ctx + "\n\n" + task)
        if n is None:
            break
        nxt = shrink_ctx(c, n, decode, max_len)
        if nxt == c:
            break
        c = nxt
        ctx = build_ctx(c)
    return c, ctx

def clamp_ctx(ctx, decode, max_len, margin=64):
    """Largest prompt that leaves room for `decode` tokens (and a margin) in
    max_model_len; a request over the served limit is a 400, not a slow run."""
    if not max_len:
        return ctx
    return min(ctx, max(1, max_len - decode - margin))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", type=int, default=600)
    ap.add_argument("--contexts")
    ap.add_argument("--temps")
    ap.add_argument("--conditions", choices=["default", "myllmbox"], default="default")
    ap.add_argument("--jsonl", help="append each row as a JSON line (for bench/gate_report.sh)")
    ap.add_argument("--dry-run", action="store_true", help="print the run plan and exit")
    a = ap.parse_args()
    myll = a.conditions == "myllmbox"
    contexts = [int(x) for x in (a.contexts or ("256" if myll else "1000,48000")).split(",")]
    limit = max_served_len()
    clamped = [clamp_ctx(c, a.decode, limit) for c in contexts]
    for c, n in zip(contexts, clamped):
        if n != c:
            print(f"note: context {c} clamped to {n} "
                  f"(max_model_len {limit}, decode {a.decode})", file=sys.stderr)
    contexts = clamped
    temps = [float(x) for x in (a.temps or ("0.0" if myll else "0.0,0.8")).split(",")]
    tasks = MYLLMBOX if myll else TASKS
    tkwargs = {"enable_thinking": False} if myll else None
    if a.dry_run:
        print(f"conditions={a.conditions} decode={a.decode} chat_template_kwargs={tkwargs}")
        for c in contexts:
            for t in temps:
                for name in tasks:
                    print(f"  ctx={c:>9,} temp={t:.1f} task={name}")
        return
    print(f"{'context':>9} {'temp':>5} {'content':<10} {'ptok':>9} {'ctok':>6} {'TTFT s':>9} "
          f"{'dec tok/s':>10} {'steps/s':>8} {'tok/step':>8} {'acc%':>6}")
    print("-" * 89)
    for c in contexts:
        used = None
        for t in temps:
            for name, task in tasks.items():
                fit, ctx = fit_ctx(c, a.decode, task, limit)
                if fit != c and used != fit:
                    print(f"note: context {c} fitted to {fit} "
                          f"(max_model_len {limit}, decode {a.decode})",
                          file=sys.stderr)
                    used = fit
                p, ct, tt, dec, sps, tps, acc = run(ctx, task, a.decode, t, tkwargs)
                print(f"{c:>9,} {t:>5.1f} {name:<10} {p:>9,} {ct:>6,} {tt:>9.2f} {dec:>10.1f} "
                      f"{sps:>8.1f} {tps:>8.2f} {acc:>6.1f}", flush=True)
                if a.jsonl:
                    with open(a.jsonl, "a") as jf:
                        jf.write(json.dumps({"context": c, "temp": t, "task": name,
                                             "ptok": p, "ctok": ct, "ttft_s": tt,
                                             "decode_tps": dec, "steps_ps": sps,
                                             "tok_per_step": tps, "accept_pct": acc}) + "\n")

if __name__ == "__main__":
    main()
