#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Shared quality gate: fixed parity suite vs a recorded baseline.

Any opt-in/quality-gated change must pass this before landing. One suite,
three sections: a slice of the AA-Omniscience questions (same loader/judge as
bench/omniscience.py), a small code-gen parity set, and a short prose parity
set. Deterministic: temperature 0, fixed seed, fixed max_tokens per prompt.

    python3 bench/quality_gate.py record              # write bench/data/gate_baseline.json
    python3 bench/quality_gate.py check               # exit 1 on regression

`record` runs the suite against a known-good server and saves per-section
scores (plus raw outputs, the parity reference). `check` runs it against a
possibly modified server and fails if any section or the overall score drops
more than the threshold (GATE_MAX_DROP env or --max-drop, default 1.0 point).
Parity sections are scored against the baseline outputs, not a reference
answer — this is a parity gate, not an accuracy benchmark.
"""
import argparse
import difflib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import omniscience

DATA = Path(__file__).resolve().parent / "data"
MAX_DROP = float(os.environ.get("GATE_MAX_DROP", "1.0"))
SEED = 0

# Short, low-drift prompts: (id, prompt, max_tokens).
CODE_PROMPTS = [
    ("fizzbuzz",
     "Write a Python function fizzbuzz(n) that returns 'FizzBuzz', 'Fizz', "
     "'Buzz', or str(n) for an integer n. Code only, no explanation.", 512),
    ("revwords",
     "Write a Python function rev_words(s) that reverses the order of the "
     "words in a string. Code only, no explanation.", 384),
    ("dedup",
     "Write a Python function dedup(xs) that removes duplicates from a list "
     "while preserving order. Code only, no explanation.", 384),
]
PROSE_PROMPTS = [
    ("lighthouse",
     "Explain in exactly two sentences how a lighthouse helps ships at night.", 256),
    ("rain",
     "Explain in exactly two sentences why rain falls from clouds.", 256),
    ("tea",
     "Describe in exactly two sentences how to brew a cup of green tea.", 256),
]


def similarity(a, b):
    """Token-level parity of two outputs, 0-100. Exact (normalized) match = 100."""
    ta, tb = " ".join(a.split()), " ".join(b.split())
    if ta == tb:
        return 100.0
    return 100.0 * difflib.SequenceMatcher(None, ta.split(), tb.split()).ratio()


def run_omniscience(url, model, csv_path, limit, concurrency, max_tokens):
    questions = omniscience.load_questions(csv_path, limit)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(
            lambda q: omniscience.run_question(url, model, q, max_tokens, seed=SEED),
            questions))
    s = omniscience.score(rows)
    return {"score": 100.0 * s["accuracy"], "n": s["n"],
            "accuracy": s["accuracy"],
            "non_hallucination_rate": s["non_hallucination_rate"]}


def run_parity(url, model, prompts, reference=None):
    """Generate outputs for the fixed prompts; score them against the baseline
    outputs in `reference` (items with id+output), or 100 when recording."""
    ref = {r["id"]: r["output"] for r in reference} if reference else {}
    items = []
    for pid, prompt, max_tokens in prompts:
        output, _ = omniscience.chat(
            url, model, [{"role": "user", "content": prompt}], max_tokens, seed=SEED)
        score = 100.0 if not reference else similarity(ref.get(pid, ""), output)
        items.append({"id": pid, "output": output, "score": score})
    return {"score": sum(i["score"] for i in items) / len(items), "items": items}


def run_suite(url, model, csv_path, omni_limit, concurrency, max_tokens,
              reference=None):
    ref_sections = (reference or {}).get("sections", {})
    sections = {}
    if omni_limit:
        sections["omniscience"] = run_omniscience(
            url, model, csv_path, omni_limit, concurrency, max_tokens)
    sections["code"] = run_parity(
        url, model, CODE_PROMPTS, ref_sections.get("code", {}).get("items"))
    sections["prose"] = run_parity(
        url, model, PROSE_PROMPTS, ref_sections.get("prose", {}).get("items"))
    overall = sum(s["score"] for s in sections.values()) / len(sections)
    return {"version": 1, "model": model, "sections": sections, "overall": overall}


def compare(baseline, candidate, max_drop):
    """Per-section and overall regression vs baseline. A drop of more than
    max_drop points fails; a drop of exactly max_drop passes."""
    rows = []
    for name, base in baseline["sections"].items():
        cand = candidate["sections"].get(name)
        if cand is None:
            rows.append({"section": name, "baseline": base["score"],
                         "candidate": None, "drop": None, "ok": False})
            continue
        drop = base["score"] - cand["score"]
        rows.append({"section": name, "baseline": base["score"],
                     "candidate": cand["score"], "drop": drop,
                     "ok": drop <= max_drop})
    drop = baseline["overall"] - candidate["overall"]
    rows.append({"section": "overall", "baseline": baseline["overall"],
                 "candidate": candidate["overall"], "drop": drop,
                 "ok": drop <= max_drop})
    return {"ok": all(r["ok"] for r in rows), "max_drop": max_drop, "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("record", "check"))
    ap.add_argument("--url", default=os.environ.get("BENCH_BASE", "http://localhost:8888"))
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "Qwen3.8-Flash-Next-NVFP4"))
    ap.add_argument("--csv", default=str(DATA / "aa_omniscience_public.csv"))
    ap.add_argument("--baseline", default=str(DATA / "gate_baseline.json"))
    ap.add_argument("--omni-limit", type=int, default=24,
                    help="omniscience questions in the suite slice (0 = parity only)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-drop", type=float, default=MAX_DROP,
                    help="max allowed score regression in points (env GATE_MAX_DROP)")
    a = ap.parse_args(argv)

    baseline = None
    if a.mode == "check":
        with open(a.baseline) as f:
            baseline = json.load(f)

    result = run_suite(a.url, a.model, a.csv, a.omni_limit, a.concurrency,
                       a.max_tokens, reference=baseline)

    if a.mode == "record":
        with open(a.baseline, "w") as f:
            json.dump(result, f, indent=2)
        for name, s in result["sections"].items():
            print(f"  {name}: {s['score']:.2f}")
        print(f"baseline written to {a.baseline}: overall {result['overall']:.2f}")
        return 0

    report = compare(baseline, result, a.max_drop)
    for r in report["rows"]:
        cand = "missing" if r["candidate"] is None else f"{r['candidate']:.2f}"
        drop = "n/a" if r["drop"] is None else f"{r['drop']:+.2f}"
        print(f"  {r['section']}: {r['baseline']:.2f} -> {cand} (drop {drop}) "
              f"{'ok' if r['ok'] else 'FAIL'}")
    if not report["ok"]:
        print(f"FAIL: regression beyond {a.max_drop} point(s) vs {a.baseline}")
        return 1
    print(f"PASS: all sections within {a.max_drop} point(s) of baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
