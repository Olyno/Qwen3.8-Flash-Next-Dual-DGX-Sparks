#!/usr/bin/env python3
"""Bit-exact output gate for speed patches.

Speed work (overlays, kernel swaps, sampling keys) is only shippable if not a
single token changes. This runs a fixed greedy prompt set, sha256s each
completion, and either records the hashes as a baseline or compares against a
recorded one:

    python3 bench/output_gate.py --record bench/baseline_prod.json
    # land a patch, restart the server, then:
    python3 bench/output_gate.py --compare bench/baseline_prod.json

Exit 1 on any mismatch. Greedy with fixed max_tokens so reruns are comparable.
The baseline records model and git HEAD so a stale baseline is obvious. For
*quality* (not identity) checks, use bench/reasoning_check.py instead.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request

PROMPTS = [
    ("prose", "Write one paragraph about the history of the lighthouse."),
    ("code", "Write a Python function that checks whether a string is a palindrome. Code only."),
    ("math", "Compute step by step: 17 * 23 + 149. Give the final number last."),
    ("french", "Explique en deux phrases pourquoi le ciel est bleu."),
    ("json", 'Reply with only a JSON object: {"city": "Paris", "country": "France", "eu": true}'),
    ("long", "Tell me a short story about a robot learning to paint."),
]


def chat(url, model, prompt, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        msg = json.load(resp)["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")
    return hashlib.sha256(text.encode()).hexdigest()


def git_head():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def run(args):
    hashes = {}
    for name, prompt in PROMPTS:
        hashes[name] = chat(args.url, args.model, prompt, args.max_tokens)
        print(f"  {name}: {hashes[name][:16]}…")
    return hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8888")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--max-tokens", type=int, default=128)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", metavar="FILE", help="record hashes to FILE")
    mode.add_argument("--compare", metavar="FILE", help="compare against FILE")
    args = ap.parse_args()

    hashes = run(args)

    if args.record:
        with open(args.record, "w") as f:
            json.dump({"model": args.model, "git": git_head(),
                       "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "hashes": hashes}, f, indent=2)
        print(f"recorded {len(hashes)} hashes -> {args.record}")
        return 0

    with open(args.compare) as f:
        baseline = json.load(f)
    if baseline.get("model") != args.model:
        print(f"WARN: baseline model is {baseline.get('model')}, comparing against {args.model}")
    base_hashes = baseline["hashes"]
    bad = 0
    for name, h in hashes.items():
        b = base_hashes.get(name)
        if b is None:
            print(f"  {name}: MISSING from baseline")
            bad += 1
        elif b != h:
            print(f"  {name}: CHANGED ({b[:16]}… -> {h[:16]}…)")
            bad += 1
    for name in base_hashes:
        if name not in hashes:
            print(f"  {name}: only in baseline (prompt removed?)")
    if bad:
        print(f"FAIL: {bad}/{len(hashes)} completions differ from {args.compare}")
        return 1
    print(f"PASS: all {len(hashes)} completions bit-identical to {args.compare}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
