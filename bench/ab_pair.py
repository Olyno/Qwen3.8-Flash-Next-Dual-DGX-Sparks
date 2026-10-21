#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Teacher-forced A/B ("spec-follow"): both arms process the SAME token sequence.

Free-running A/B under speculation is confounded: one accepted token early
sends arm B down a different continuation, and the latency delta you report is
then partly a text delta. This removes the confound by replaying a captured
completion (jsonl of prompt+completion) through both arms as fixed text:
each record is cut into chunks at newline boundaries; a chunk's text is
teacher-forced into the prompt, and the arm decodes the next chunk with
ignore_eos at a fixed token length, so both arms always see an identical
prefix and decode an identical-length window. Divergence is measured (follow%)
instead of silently corrupting the timing.

    python3 bench/ab_pair.py --corpus bench/data/omni_prod.jsonl \
        --prompt-field question --completion-field response \
        --arm-a http://localhost:8888 --arm-b http://localhost:8888

Reporting conventions (codified in docs/bench-gates.md, followed by this tool):
  * A/B under speculation reports **ms/step**, never tok/s — acceptance moves
    with the text, so tok/s compares texts, not configurations. ms/step is
    decode-window / engine-steps, steps = spec_decode drafts (draft-free arm:
    generation tokens, one step per token), from vLLM /metrics deltas
    (bench/mtp_accept.py's counters).
  * If |delta ms/step| < 2%, the verdict is not a win: this tool immediately
    re-runs the baseline arm and prints the baseline-vs-itself drift; the
    standing rule is to re-run the baseline within the hour before quoting
    any sub-2% delta (server state, thermals and cache warmth drift).

Ideas credited (methodology only, no code copied):
  Strata (MIT)         — fixed-sequence replay so both arms score identical text
  XeStrata (LGPL, idea-only) — chunked teacher-forced continuation under spec decode
  flashrt (Apache-2.0) — never judge a delta inside the baseline's own re-run noise
  ds4-dfm-rs (MIT)     — "follow" as the name for replaying captured completions

Records too short to chunk fall back to fixed-tail decode on the whole text
(both arms decode --tail tokens on the identical prefix).
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

import mtp_accept

DRAFTS = "vllm:spec_decode_num_drafts_total"
DRAFT_TOK = "vllm:spec_decode_num_draft_tokens_total"
ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
GEN = "vllm:generation_tokens_total"
FIELDS = [("prompt", "completion"), ("question", "response")]


def load_records(path, prompt_field=None, completion_field=None, limit=0):
    """One record per line; fields auto-detected (prompt/completion, then
    question/response = bench/data/omni_prod.jsonl) unless given."""
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pf, cf = prompt_field, completion_field
            if pf is None or cf is None:
                for pf, cf in FIELDS:
                    if row.get(pf) and row.get(cf):
                        break
            text = (row.get(pf) or "") + "\n\n" + (row.get(cf) or "")
            if text.strip():
                out.append(text)
            if limit and len(out) >= limit:
                break
    return out


def split_chunks(text, chunk_chars):
    """Boundary offsets into `text`, cut at newlines, ~chunk_chars apart.
    Returns ascending offsets with the last one == len(text); the caller
    teacher-forces text[:b_i] and decodes toward b_{i+1}."""
    bounds, pos = [], 0
    while pos < len(text):
        want = min(pos + chunk_chars, len(text))
        if want >= len(text):
            cut = len(text)
        else:
            cut = text.rfind("\n", min(pos + 1, want), want + 1)
            cut = want if cut <= pos else cut + 1
        if cut == pos:  # no newline in reach (single huge line): hard-cut
            cut = want
        bounds.append(cut)
        pos = cut
    return bounds


def follow_ratio(got, want):
    """Fraction of `want`'s characters reproduced as a prefix of `got` — the
    strict divergence measure: one wrong character ends the streak."""
    if not want:
        return 1.0
    n = 0
    for a, b in zip(got, want):
        if a != b:
            break
        n += 1
    return min(1.0, n / len(want))


def arm_counters(url):
    return mtp_accept.scrape(url.rstrip("/") + "/metrics")


def delta_ms_per_step(rows, cur, base):
    """ms/step for one arm pass: summed decode windows / summed engine steps.
    steps = drafts under speculation; generation tokens on a draft-free arm."""
    win = sum(r["decode_s"] for r in rows)
    d = cur.get(DRAFTS, 0.0) - base.get(DRAFTS, 0.0)
    g = cur.get(GEN, 0.0) - base.get(GEN, 0.0)
    steps = d if d > 0 else g
    tps = 1.0 + (cur.get(ACCEPTED, 0.0) - base.get(ACCEPTED, 0.0)) / d if d > 0 else 1.0
    return (win / steps * 1000.0 if steps > 0 else float("nan")), steps, tps


def post_stream(url, payload, timeout=600):
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    t_first = t_last = usage = None
    parts = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
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
                piece = ch.get("text")
                if piece:
                    if t_first is None:
                        t_first = time.time() - t0
                    t_last = time.time()
                    parts.append(piece)
    got = "".join(parts)
    ptok = (usage or {}).get("prompt_tokens", 0)
    ctok = (usage or {}).get("completion_tokens", len(parts))
    ttfb = t_first if t_first is not None else time.time() - t0
    decode_s = (t_last - (t0 + ttfb)) if (t_last and ctok > 1) else 0.0
    return {"got": got, "ptok": ptok, "ctok": ctok, "ttfb": ttfb,
            "decode_s": max(decode_s, 0.0)}


def tokenize_len(url, text, model):
    req = urllib.request.Request(url.rstrip("/") + "/tokenize",
                                 data=json.dumps({"model": model, "prompt": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)["count"]
    except Exception:
        return max(8, len(text) // 4)  # char/4 heuristic if /tokenize unavailable


def run_arm(url, model, records, a, tag, rep):
    """One arm over the whole corpus; returns per-chunk rows."""
    rows = []
    for ri, text in enumerate(records):
        bounds = split_chunks(text, a.chunk_chars)
        tasks = []
        if len(bounds) >= 2:  # decode toward each following boundary
            for i in range(len(bounds) - 1):
                target = text[bounds[i]:bounds[i + 1]]
                ntok = min(tokenize_len(url, target, model) + 8, a.max_decode)
                tasks.append((text[:bounds[i]], target, ntok))
        if not tasks:  # too short to chunk: fixed-tail decode on full text
            tasks.append((text, "", a.tail))
        for ci, (prefix, target, ntok) in enumerate(tasks):
            if not prefix.strip():
                prefix = text[:max(64, len(text) // 10)]
            payload = {"model": model, "prompt": prefix, "max_tokens": ntok,
                       "min_tokens": ntok, "ignore_eos": True, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True}}
            m = post_stream(url, payload)
            rows.append({"rep": rep, "arm": tag, "url": url, "record": ri,
                         "chunk": ci, "ptok": m["ptok"], "ctok": m["ctok"],
                         "ttfb_s": m["ttfb"], "decode_s": m["decode_s"],
                         "follow": follow_ratio(m["got"], target)})
    return rows


def summarize(rows, cur, base, tag):
    msps, steps, tps = delta_ms_per_step(rows, cur, base)
    ptok = sum(r["ptok"] for r in rows)
    ttfb = sum(r["ttfb_s"] for r in rows)
    follow = sum(r["follow"] for r in rows) / len(rows)
    return {"arm": tag, "ms_per_step": msps, "tok_per_step": tps,
            "steps": steps, "follow": follow,
            "prefill_tps": ptok / ttfb if ttfb > 0 else float("nan"),
            "ptok": ptok, "ctok": sum(r["ctok"] for r in rows)}


def verdict(a_sum, b_sum):
    d = a_sum["ms_per_step"], b_sum["ms_per_step"]
    if any(x != x for x in d):  # NaN
        return None
    return 100.0 * (d[1] - d[0]) / d[0]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", help="jsonl of prompt+completion records")
    ap.add_argument("--prompt-field")
    ap.add_argument("--completion-field")
    ap.add_argument("--records", type=int, default=4, help="max records per pass")
    ap.add_argument("--arm-a", default=os.environ.get("BENCH_BASE", "http://localhost:8888"),
                    help="baseline endpoint")
    ap.add_argument("--arm-b", required=False, help="candidate endpoint")
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "Qwen3.8-Flash-Next-NVFP4"))
    ap.add_argument("--chunk-chars", type=int, default=3000)
    ap.add_argument("--max-decode", type=int, default=2048,
                    help="cap on forced decode tokens per chunk window")
    ap.add_argument("--tail", type=int, default=64, help="decode length for short records")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", default="logs/ab_pair.jsonl")
    ap.add_argument("--rerun-pct", type=float, default=2.0,
                    help="re-run baseline arm when |delta ms/step| < this percent")
    ap.add_argument("--no-rerun", action="store_true")
    ap.add_argument("--self-check", action="store_true",
                    help="run the offline self-check and exit")
    a = ap.parse_args(argv)
    if a.self_check:
        return _self_check()
    if not a.corpus or not a.arm_b:
        ap.error("--corpus and --arm-b are required (or use --self-check)")

    records = load_records(a.corpus, a.prompt_field, a.completion_field, a.records)
    print(f"spec-follow: {len(records)} records, chunk={a.chunk_chars} chars, "
          f"A={a.arm_a} B={a.arm_b}", flush=True)
    arms = {"A": a.arm_a, "B": a.arm_b}
    order = ["A", "B"]
    for rep in range(a.reps):
        if rep % 2:
            order = order[::-1]  # alternate order: thermal/cache drift hits both
        sums = {}
        for tag in order:
            base = arm_counters(arms[tag])
            rows = run_arm(arms[tag], a.model, records, a, tag, rep)
            cur = arm_counters(arms[tag])
            sums[tag] = summarize(rows, cur, base, tag)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            with open(a.out, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            s = sums[tag]
            print(f"  rep{rep} {tag}: ms/step={s['ms_per_step']:.2f} "
                  f"tok/step={s['tok_per_step']:.3f} steps={s['steps']:,.0f} "
                  f"follow={100*s['follow']:.1f}% prefill={s['prefill_tps']:,.0f} tok/s",
                  flush=True)
        d = verdict(sums["A"], sums["B"])
        if d is None:
            print("  verdict: no engine steps counted on one arm (drafts/gen tokens)")
            continue
        print(f"  rep{rep} delta ms/step (B vs A): {d:+.2f}%  "
              f"[report ms/step, not tok/s — acceptance moves with the text]")
        if abs(d) < a.rerun_pct and not a.no_rerun:
            print(f"  |delta| < {a.rerun_pct}% -> re-running baseline arm (sub-threshold "
                  f"deltas are not wins; re-baseline within the hour before quoting)",
                  flush=True)
            base = arm_counters(a.arm_a)
            rows = run_arm(a.arm_a, a.model, records, a, "A-rerun", rep)
            cur = arm_counters(a.arm_a)
            rer = summarize(rows, cur, base, "A-rerun")
            drift = 100.0 * (rer["ms_per_step"] - sums["A"]["ms_per_step"]) / sums["A"]["ms_per_step"]
            call = "within baseline noise — inconclusive" if abs(drift) >= abs(d) \
                else "delta survives baseline re-run"
            print(f"  baseline re-run ms/step={rer['ms_per_step']:.2f} "
                  f"(self-drift {drift:+.2f}%) -> {call}")
            with open(a.out, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        print(f"ab arm_a_ms_per_step={sums['A']['ms_per_step']:.4f} "
              f"arm_b_ms_per_step={sums['B']['ms_per_step']:.4f} "
              f"delta_pct={d:+.3f} follow_a={sums['A']['follow']:.4f} "
              f"follow_b={sums['B']['follow']:.4f} rep={rep}", flush=True)
    return 0


def _self_check():
    # chunker: monotonic, covers the text, cuts at newlines when available
    t = ("line\n" * 500)
    b = split_chunks(t, 300)
    assert b == sorted(set(b)) and b[-1] == len(t)
    assert all(t[i - 1] == "\n" for i in b[:-1]), "cuts at newlines"
    b2 = split_chunks("x" * 1000, 300)
    assert b2 == [300, 600, 900, 1000], b2  # hard cut without newlines
    assert split_chunks("short", 300) == [5]
    # follow: exact, diverging, empty
    assert follow_ratio("abc", "abc") == 1.0
    assert abs(follow_ratio("abXdef", "abcdef") - 2 / 6) < 1e-9
    assert follow_ratio("", "abc") == 0.0
    assert follow_ratio("abc", "") == 1.0
    # steps fallback: draft-free arm counts generation tokens as steps
    rows = [{"decode_s": 1.0}, {"decode_s": 1.0}]
    msps, steps, tps = delta_ms_per_step(rows, {GEN: 20.0, DRAFTS: 0.0},
                                         {GEN: 0.0, DRAFTS: 0.0})
    assert msps == 100.0 and steps == 20.0 and tps == 1.0
    msps, steps, tps = delta_ms_per_step(rows, {DRAFTS: 8.0, ACCEPTED: 12.0, GEN: 20.0}, {})
    assert abs(msps - 250.0) < 1e-9 and steps == 8.0 and abs(tps - 2.5) < 1e-9
    # record field autodetection
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"question": "q1", "response": "r1"}) + "\n")
        f.write(json.dumps({"prompt": "p2", "completion": "c2"}) + "\n")
        path = f.name
    recs = load_records(path)
    assert recs[0].startswith("q1") and recs[1].startswith("p2")
    os.unlink(path)
    print("self-check OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
