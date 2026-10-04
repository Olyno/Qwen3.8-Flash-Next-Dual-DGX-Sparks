#!/usr/bin/env python3
"""Answer-correctness gate for speed patches.

Bitwise gating is impossible on this stack: GDN attention has no
batch-invariant mode in vLLM 0.30 and greedy completions drift run to run.
So instead of hashing completions, this checks each prompt's *answer* against
ground truth, N times per prompt, and fails if any run is wrong:

    python3 bench/output_gate.py                 # against localhost:8888
    python3 bench/output_gate.py --runs 5

Exit 1 on any wrong answer. A patch that preserves quality passes regardless
of wording drift; a patch that breaks the model fails within a run or two.
"""
import argparse
import json
import re
import sys
import urllib.request

CHECKS = [
    ("math",
     "Compute step by step: 17 * 23 + 149. Give the final number last.",
     lambda t: "540" in t),
    ("json",
     'Reply with only a JSON object: {"city": "Paris", "country": "France", "eu": true}',
     lambda t: _is_paris_json(t)),
    ("code",
     "Write a Python function that checks whether a string is a palindrome. Code only.",
     lambda t: _palindrome_works(t)),
    ("french",
     "Explique en deux phrases pourquoi le ciel est bleu.",
     lambda t: "bleu" in t.lower() and ("diffus" in t.lower() or "rayleigh" in t.lower())),
    ("prose",
     "Write one paragraph about the history of the lighthouse.",
     lambda t: len(t) > 200),
]


def _is_paris_json(t):
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return False
    return obj.get("city") == "Paris" and obj.get("country") == "France" and obj.get("eu") is True


def _palindrome_works(t):
    m = re.search(r"```(?:python)?\n(.*?)```", t, re.S)
    code = m.group(1) if m else t
    ns = {}
    try:
        exec(code, ns)  # noqa: S102 — model output, sandboxed only by trust in the gate
    except Exception:
        return False
    fn = next((v for k, v in ns.items() if callable(v) and "palindrome" in k.lower()), None)
    if fn is None:
        return False
    try:
        return fn("racecar") is True and fn("hello") is False and fn("A man a plan a canal Panama".replace(" ", "").lower()) is True
    except Exception:
        return False


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
    return (msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8888")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    failures = 0
    for name, prompt, check in CHECKS:
        ok = 0
        for _ in range(args.runs):
            try:
                if check(chat(args.url, args.model, prompt, args.max_tokens)):
                    ok += 1
            except Exception as exc:
                print(f"  {name}: request error: {exc}")
        status = "ok" if ok == args.runs else "FAIL"
        if ok != args.runs:
            failures += 1
        print(f"  {name}: {ok}/{args.runs} {status}")

    if failures:
        print(f"FAIL: {failures}/{len(CHECKS)} checks below {args.runs}/{args.runs}")
        return 1
    print(f"PASS: all {len(CHECKS)} checks correct in {args.runs}/{args.runs} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
