#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""p99-KL quality yardstick: teacher-forced distribution comparison, two arms.

Generation-based quality gates (output_gate, quality_gate) judge sampled text,
which on this stack drifts run to run. This measures the model's *distribution*
instead: five byte-frozen texts (bench/data/kl_corpus.jsonl — code, English
prose, German, agent-transcript, math) are sent to /v1/completions with
prompt_logprobs=64 and return_token_ids, so every arm is scored on exactly the
same tokenized positions; nothing is sampled.

    python3 bench/kl_yardstick.py compare --url http://localhost:8888 \
        --candidate http://localhost:8889
    python3 bench/kl_yardstick.py compare --url http://localhost:8888 \
        --capture base.json                        # snapshot baseline + floor
    python3 bench/kl_yardstick.py compare --replay base.json \
        --candidate http://localhost:8889          # score a later candidate
    python3 bench/kl_yardstick.py spine            # rebuild positions file
    python3 bench/kl_yardstick.py freeze --url ..  # pin token ids (live host)

Per position (the top-64 sets, renormalized over their union; ids missing from
one arm get its 64th-best logprob minus 23 nats — a hard-zero stand-in that
never produces inf): top-1 agreement, KL(baseline||candidate), and each arm's
logprob of the actual token (perplexities). Reported: top-1 agreement, mean KL,
**p99 KL** (nearest-rank), perplexity delta — per text and pooled.

Optional extras, all off by default: `--decision-ids` adds a diagnostic row for
stop-decision token ids (n, mean KL, geometric-mean probability ratio, ppl per
arm) without moving the verdict; `--batch` sends requests concurrently and
`--shape-floor` folds the resulting batch-composition noise into the floor; and
a corpus item's `"scored_token_spans"` restricts pooling to its generated
spans while the full text is still scored.

The floor comes first, always (flashrt discipline): the baseline is measured
twice and scored against itself, and the candidate is only judged relative to
that self-noise. A delta inside your own floor is not a difference. Default
gates: p99 KL <= 2x floor p99 KL, top-1 agreement >= 99.5% (thresholds are
flags; see docs/bench-gates.md). Exit 1 on FAIL.

Position identity across hosts: the corpus texts are byte-frozen (sha256 in
bench/data/kl_corpus_positions.json, together with the whitespace-segment
spine), the two arms' prompt_token_ids are compared and must match exactly,
and once `freeze` has pinned the tokenized position list against a healthy
endpoint every later run diffs the live tokenization against the pin and
refuses to score on mismatch.

Single-GPU capture/replay: `--capture FILE` writes the baseline arm's
per-position top-k distributions, its logprob of each actual token, the
tokenized spine, the corpus sha256, the positions-pin hash, and the self-noise
floor measured in that same run. `--replay FILE` then substitutes that file for
the baseline endpoint and keeps the candidate live; the captured floor is used
verbatim, and replay refuses to score if the live corpus hash, position pin, or
candidate tokenization disagrees with the capture. `--url` and `--replay` are
mutually exclusive. With neither flag the behaviour is unchanged.

Ideas credited (methodology only, no code copied):
  Strata (MIT)         — fixed corpus, identical scoring positions
  XeStrata (LGPL, idea-only) — per-position top-k renormalized KL, p99 tail metric
  flashrt (Apache-2.0) — self-noise floor printed and enforced before any delta
  ds4-dfm-rs (MIT)   — teacher-forced scoring via prompt_logprobs
"""
import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CORPUS = os.path.join(DATA, "kl_corpus.jsonl")
POSITIONS = os.path.join(DATA, "kl_corpus_positions.json")
MISS_MARGIN = 23.0  # nats below the k-th logprob: stand-in for a truncated id


# ---------- math (pure, unit-tested) ----------

def logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def kl_renorm(p, q, margin=MISS_MARGIN):
    """KL(P||Q) over the union of two top-k logprob dicts, each renormalized
    over the union; an id missing from an arm gets that arm's worst shown
    logprob - margin (truncation is symmetric enough at k=64)."""
    ids = p.keys() | q.keys()
    fp = min(p.values()) - margin
    fq = min(q.values()) - margin
    lp = [p.get(t, fp) for t in ids]
    lq = [q.get(t, fq) for t in ids]
    sp, sq = logsumexp(lp), logsumexp(lq)
    kl = 0.0
    for a, b in zip(lp, lq):
        pa = a - sp
        kl += math.exp(pa) * (pa - (b - sq))
    return max(kl, 0.0)


def top1(p):
    return max(p, key=p.get)


def percentile_nearest(vals, q):
    """Nearest-rank percentile: index ceil(q*n)-1 of the sorted list."""
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[max(0, math.ceil(q * len(s)) - 1)]


def norm_dist(elem):
    """vLLM prompt_logprobs element -> {token_id:int -> logprob:float}."""
    if not elem:
        return None
    out = {}
    for k, v in elem.items():
        try:
            tid = int(k)
        except ValueError:
            continue
        lp = v.get("logprob") if isinstance(v, dict) else None
        if lp is not None:
            out[tid] = float(lp)
    return out or None


def span_mask(n, spans):
    """Boolean mask of length n, True inside any of `spans`; each span is
    [start, end) in token indices and is clamped to [0, n). Overlaps union."""
    mask = [False] * n
    for start, end in spans or ():
        for i in range(max(0, start), min(n, end)):
            mask[i] = True
    return mask


def decision_stats(rows):
    """Pool decision-position samples across texts: position count, mean KL,
    geometric-mean probability ratio of the actual stop token (candidate over
    baseline; below 1 means the candidate is less likely to stop), and each
    arm's perplexity over just those positions."""
    kl = [v for r in rows for v in r["decision"]["kl"]]
    lp_p = [v for r in rows for v in r["decision"]["lp_p"]]
    lp_c = [v for r in rows for v in r["decision"]["lp_c"]]
    n = len(kl)
    if not n:
        return {"n": 0, "mean_kl": float("nan"), "geo_ratio": float("nan"),
                "ppl_p": float("nan"), "ppl_c": float("nan")}
    mp, mc = sum(lp_p) / n, sum(lp_c) / n
    return {"n": n, "mean_kl": sum(kl) / n, "geo_ratio": math.exp(mp - mc),
            "ppl_p": math.exp(-mp), "ppl_c": math.exp(-mc)}


def effective_floor(floor, shape_floor):
    """Conservative verdict floor: the larger p99 KL of the measured passes."""
    vals = [f["p99_kl"] for f in (floor, shape_floor) if f]
    return max(vals) if vals else None


def score_pair(tp, tc, ids_p, ids_c, spans=None, decision_ids=()):
    """Score one text, baseline dists tp vs candidate dists tc (index i is the
    distribution over token i; position 0 is None on both). Returns per-text
    stats; raises if the two arms' tokenizations differ. `spans` limits pooled
    positions to those [start, end) token ranges; `decision_ids` additionally
    tracks stop-decision positions for the diagnostic row."""
    if ids_p != ids_c:
        raise SystemExit(
            f"tokenization differs between arms ({len(ids_p)} vs {len(ids_c)} "
            f"tokens or id mismatch) — positions are not identical, refusing "
            f"to score; run `freeze` against a healthy endpoint to pin them")
    mask = span_mask(len(ids_p), spans) if spans else None
    kls, hits, nll_p, nll_c, flips = [], 0, [], [], []
    dkl, dlp_p, dlp_c = [], [], []
    for i in range(1, len(ids_p)):
        if mask is not None and not mask[i]:
            continue
        dp, dc = norm_dist(tp[i]), norm_dist(tc[i])
        if not dp or not dc:
            continue
        kls.append(kl_renorm(dp, dc))
        lp_p = dp.get(ids_p[i], min(dp.values()) - MISS_MARGIN)
        lp_c = dc.get(ids_c[i], min(dc.values()) - MISS_MARGIN)
        hits += top1(dp) == top1(dc)
        nll_p.append(-lp_p)
        nll_c.append(-lp_c)
        flips.append((i, ids_p[i], top1(dp), top1(dc)))
        if decision_ids and ids_p[i] in decision_ids:
            dkl.append(kls[-1])
            dlp_p.append(lp_p)
            dlp_c.append(lp_c)
    n = len(kls)
    if not n:
        raise SystemExit("no scorable positions (server returned no prompt_logprobs, "
                         "or scored_token_spans excludes every position?)")
    return {"n": n, "kl": kls, "top1": hits / n,
            "mean_kl": sum(kls) / n, "p99_kl": percentile_nearest(kls, 0.99),
            "ppl_p": math.exp(sum(nll_p) / n), "ppl_c": math.exp(sum(nll_c) / n),
            "decision": {"kl": dkl, "lp_p": dlp_p, "lp_c": dlp_c},
            "flips": [f for f in flips if f[2] != f[3]]}


def pool(rows):
    kl = [v for r in rows for v in r["kl"]]
    n = sum(r["n"] for r in rows)
    return {"n": n, "top1": sum(r["top1"] * r["n"] for r in rows) / n,
            "mean_kl": sum(r["mean_kl"] * r["n"] for r in rows) / n,
            "p99_kl": percentile_nearest(kl, 0.99),
            "ppl_p": math.exp(sum(math.log(r["ppl_p"]) * r["n"] for r in rows) / n),
            "ppl_c": math.exp(sum(math.log(r["ppl_c"]) * r["n"] for r in rows) / n)}


# ---------- transport ----------

def completions_logprobs(url, model, text, k, gen_tokens, timeout):
    payload = {"model": model, "prompt": text, "max_tokens": gen_tokens,
               "temperature": 0, "prompt_logprobs": k, "return_token_ids": True}
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                choice = json.load(r)["choices"][0]
            ids = choice.get("prompt_token_ids")
            plist = choice.get("prompt_logprobs")
            if ids is None or plist is None:
                raise SystemExit("server did not return prompt_token_ids/prompt_logprobs "
                                 "(needs /v1/completions with prompt_logprobs + "
                                 "return_token_ids support)")
            return ids, plist
        except SystemExit:
            raise
        except Exception as e:
            last = e
            time.sleep(10 * (attempt + 1))
    raise last


def run_pass(url, model, texts, a, batch=None):
    """Score every text, up to `batch` requests in flight (default a.batch);
    results stay in input order regardless of completion order."""
    def one(t):
        return completions_logprobs(url, model, t["text"], a.topk,
                                    a.gen_tokens, a.timeout)
    workers = max(1, a.batch if batch is None else batch)
    if workers == 1:
        return [one(t) for t in texts]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, texts))


# ---------- corpus spine ----------

def corpus_sha256(texts):
    """Hash the corpus content: sha256 over the ordered per-text sha256 hex
    strings. Host-independent and byte-frozen (same recipe as `spine`)."""
    blob = hashlib.sha256()
    for t in texts:
        blob.update(hashlib.sha256(t["text"].encode()).hexdigest().encode())
    return blob.hexdigest()


def spine_build(texts):
    """Offline position spine: per text — sha256, char count, whitespace-segment
    [start,end] spans. Host-independent; the pin that makes scoring
    position-identical once `freeze` adds live token ids."""
    doc = {"version": 1, "corpus_sha256": None, "texts": {}}
    for t in texts:
        segs, i, n = [], 0, len(t["text"])
        while i < n:
            while i < n and t["text"][i].isspace():
                i += 1
            j = i
            while j < n and not t["text"][j].isspace():
                j += 1
            if j > i:
                segs.append([i, j])
            i = j
        sha = hashlib.sha256(t["text"].encode()).hexdigest()
        doc["texts"][t["id"]] = {"sha256": sha, "n_chars": n,
                                 "segments": segs, "token_ids": None,
                                 "frozen_url": None, "frozen_at": None}
    doc["corpus_sha256"] = corpus_sha256(texts)
    return doc


def spine_dump(doc, path):
    """Compact serialization: one line per text (segment spans would explode
    into thousands of indented lines otherwise)."""
    items = list(doc["texts"].items())
    with open(path, "w") as f:
        f.write('{\n "version": %d,\n "corpus_sha256": %s,\n "texts": {\n'
                % (doc["version"], json.dumps(doc["corpus_sha256"])))
        for i, (k, v) in enumerate(items):
            f.write("  %s: %s%s\n" % (json.dumps(k),
                                      json.dumps(v, separators=(",", ":")),
                                      "," if i < len(items) - 1 else ""))
        f.write(" }\n}\n")


def load_corpus(path=CORPUS):
    texts = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                texts.append({"id": r["id"], "text": r["text"],
                              "spans": r.get("scored_token_spans")})
    return texts


def load_positions(path=POSITIONS, corpus_sha=None):
    with open(path) as f:
        doc = json.load(f)
    if corpus_sha and doc.get("corpus_sha256") != corpus_sha:
        raise SystemExit(f"{path} is stale for {CORPUS} — rerun `spine`")
    return doc


# ---------- capture / replay (single-GPU yardstick) ----------

CAPTURE_VERSION = 1


def dist_pairs(elem):
    """vLLM prompt_logprobs element -> [[token_id, logprob], ...] or None.
    Canonical compact form written to a capture and rebuilt on replay."""
    d = norm_dist(elem)
    if not d:
        return None
    return [[tid, lp] for tid, lp in sorted(d.items())]


def pairs_elem(pairs):
    """[[token_id, logprob], ...] -> the vLLM element shape norm_dist expects."""
    if pairs is None:
        return None
    return {str(tid): {"logprob": float(lp)} for tid, lp in pairs}


def stat_summary(st):
    """Scalar stats safe to serialize: score_pair also carries per-position
    lists, but a capture only needs the summary for display and gating."""
    return {k: st[k] for k in
            ("n", "top1", "mean_kl", "p99_kl", "ppl_p", "ppl_c")}


def positions_pin_hash(doc):
    """Hash of the pinned tokenized spine, or None when nothing is pinned.
    Metadata (frozen_url/frozen_at) is excluded: only position identity
    matters for a replay."""
    if not doc:
        return None
    pins = {k: v["token_ids"] for k, v in doc.get("texts", {}).items()
            if v.get("token_ids") is not None}
    if not pins:
        return None
    return hashlib.sha256(json.dumps(pins, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def capture_build(texts, base, a, floor=None, floor_rows=None,
                  shape_floor=None, shape_rows=None, pos=None):
    """Everything a later candidate needs to be scored against this baseline:
    per-position top-k pairs, the baseline's logprob of each actual token, the
    tokenized spine, corpus/positions hashes, and the measured floor."""
    doc = {
        "version": CAPTURE_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_url": a.url,
        "model": a.model,
        "topk": a.topk,
        "gen_tokens": a.gen_tokens,
        "corpus_sha256": corpus_sha256(texts),
        "positions_pin_sha256": positions_pin_hash(pos),
        "floor": stat_summary(floor) if floor else None,
        "floor_rows": ({t["id"]: stat_summary(st)
                        for t, st in zip(texts, floor_rows)}
                       if floor_rows is not None else None),
        "shape_floor": stat_summary(shape_floor) if shape_floor else None,
        "shape_floor_rows": ({t["id"]: stat_summary(st)
                              for t, st in zip(texts, shape_rows)}
                             if shape_rows is not None else None),
        "shape_floor_batch": a.shape_floor if shape_floor else None,
        "texts": {},
    }
    for t, (ids, plist) in zip(texts, base):
        if len(plist) < len(ids):
            raise SystemExit(f"capture: {t['id']} has {len(ids)} token ids but "
                             f"only {len(plist)} logprob elements — refusing to write")
        dists, actual = [], []
        for i, tid in enumerate(ids):
            d = norm_dist(plist[i])
            dists.append(dist_pairs(plist[i]))
            if i == 0 or not d:
                actual.append(None)
            else:
                actual.append(d.get(tid, min(d.values()) - MISS_MARGIN))
        doc["texts"][t["id"]] = {
            "sha256": hashlib.sha256(t["text"].encode()).hexdigest(),
            "n_positions": len(ids),
            "token_ids": ids,
            "dists": dists,
            "token_logprob": actual,
        }
    return doc


def capture_write(path, doc):
    try:
        with open(path, "w") as f:
            json.dump(doc, f, separators=(",", ":"))
            f.write("\n")
    except OSError as e:
        raise SystemExit(f"capture: cannot write {path}: {e}")


def capture_load(path):
    try:
        with open(path) as f:
            doc = json.load(f)
    except OSError as e:
        raise SystemExit(f"replay: cannot read capture {path}: {e}")
    except ValueError as e:
        raise SystemExit(f"replay: {path} is not valid JSON: {e}")
    if not isinstance(doc, dict) or doc.get("version") != CAPTURE_VERSION:
        got = doc.get("version") if isinstance(doc, dict) else "?"
        raise SystemExit(f"replay: {path} is not a version-{CAPTURE_VERSION} "
                         f"capture (got version={got})")
    if not isinstance(doc.get("texts"), dict) or not doc.get("corpus_sha256"):
        raise SystemExit(f"replay: {path} is missing corpus_sha256/texts")
    return doc


def replay_baseline(capture, texts, pos=None):
    """Reconstruct the baseline pass from a capture, refusing loudly when the
    live corpus, position pin, or captured spine no longer matches."""
    live_sha = corpus_sha256(texts)
    if capture["corpus_sha256"] != live_sha:
        raise SystemExit(
            f"replay refused: capture corpus_sha256={capture['corpus_sha256']} "
            f"but the live corpus is {live_sha} — the capture was made from "
            f"different text, refusing to score")
    cap_pin = capture.get("positions_pin_sha256")
    live_pin = positions_pin_hash(pos)
    if cap_pin and live_pin and cap_pin != live_pin:
        raise SystemExit("replay refused: the positions pin changed since the "
                         "capture — re-capture or re-freeze, refusing to score")
    out = []
    for t in texts:
        cap = capture["texts"].get(t["id"])
        if not cap:
            raise SystemExit(f"replay refused: capture has no text {t['id']!r}")
        sha = hashlib.sha256(t["text"].encode()).hexdigest()
        if cap.get("sha256") != sha:
            raise SystemExit(f"replay refused: corpus text {t['id']} differs "
                             f"from the capture — refusing to score")
        ids = cap["token_ids"]
        if pos:
            frozen = pos.get("texts", {}).get(t["id"], {}).get("token_ids")
            if frozen is not None and frozen != ids:
                raise SystemExit(f"replay refused: positions pin for {t['id']} "
                                 f"differs from the capture — refusing to score")
        pairs = cap["dists"]
        if len(pairs) != len(ids):
            raise SystemExit(f"replay refused: capture for {t['id']} has "
                             f"{len(pairs)} distributions for {len(ids)} positions")
        out.append((ids, [pairs_elem(p) for p in pairs]))
    return out


def check_replay_tokenization(capture, texts, cand):
    """The live candidate must be scored on exactly the captured positions."""
    for t, (ids_c, _) in zip(texts, cand):
        cap = capture["texts"].get(t["id"])
        if not cap:
            raise SystemExit(f"replay refused: capture has no text {t['id']!r}")
        if ids_c != cap["token_ids"]:
            raise SystemExit(
                f"replay refused: live tokenization of {t['id']} differs from "
                f"the capture ({len(ids_c)} vs {len(cap['token_ids'])} tokens or "
                f"id mismatch) — positions are not identical, refusing to score")


# ---------- modes ----------

def fmt_row(name, st):
    return (f"{name:<26}{st['n']:>7,}{100*st['top1']:>9.2f}{st['mean_kl']:>10.5f}"
            f"{st['p99_kl']:>10.5f}{st['ppl_p']:>10.2f}{st['ppl_c']:>10.2f}")


def print_header(title):
    print(f"\n{title}")
    print(f"{'comparison':<26}{'pos':>7}{'top1 %':>9}{'mean KL':>10}"
          f"{'p99 KL':>10}{'ppl base':>10}{'ppl cand':>10}")
    print("-" * 82)


def fmt_decision(st):
    name = "decision positions"
    if not st["n"]:
        return f"{name:<26}{0:>7,}{'-':>10}{'-':>13}{'-':>10}{'-':>10}"
    return (f"{name:<26}{st['n']:>7,}{st['mean_kl']:>10.5f}"
            f"{st['geo_ratio']:>13.4f}{st['ppl_p']:>10.2f}{st['ppl_c']:>10.2f}")


def score_all(texts, a, left, right):
    """Score one pass against another, text by text, order preserved."""
    return [score_pair(left[i][1], right[i][1], left[i][0], right[i][0],
                       spans=texts[i].get("spans"),
                       decision_ids=a.decision_ids)
            for i in range(len(texts))]


def replay_capture_floor(capture, texts):
    """Print a captured floor before any verdict (floor first, as in a live
    run); returns the pooled floor and shape floor for the gate."""
    floor = capture.get("floor")
    floor_rows = capture.get("floor_rows") or {}
    if floor:
        print_header("SELF-NOISE FLOOR (baseline vs itself) — judge nothing inside this")
        for t in texts:
            st = floor_rows.get(t["id"])
            if st:
                print(fmt_row(f"floor {t['id']}", st))
        print(fmt_row("floor POOLED", floor))
    shape_floor = capture.get("shape_floor")
    shape_rows = capture.get("shape_floor_rows") or {}
    if shape_floor:
        batch = capture.get("shape_floor_batch")
        batch = "?" if batch is None else batch
        print_header(f"BATCH-SHAPE FLOOR (baseline vs itself, "
                     f"batch={batch}) — folded into the gate floor")
        for t in texts:
            st = shape_rows.get(t["id"])
            if st:
                print(fmt_row(f"floor(shape) {t['id']}", st))
        print(fmt_row("floor(shape) POOLED", shape_floor))
    return floor, shape_floor


def cmd_compare(a):
    texts = load_corpus(a.corpus)
    pos = load_positions(a.positions) if os.path.isfile(a.positions) else None
    if pos:
        for t in texts:
            pin = pos["texts"].get(t["id"])
            if not pin or hashlib.sha256(t["text"].encode()).hexdigest() != pin["sha256"]:
                raise SystemExit(f"corpus text {t['id']} differs from {a.positions} "
                                 f"— rerun `spine` (corpus is byte-frozen)")
    capture = capture_load(a.replay) if a.replay else None
    if capture is not None and capture.get("topk") is not None \
            and a.topk != capture["topk"]:
        raise SystemExit(f"replay refused: --topk={a.topk} but the capture was "
                         f"made with topk={capture['topk']} — refusing to score")
    baseline = f"replay:{a.replay}" if capture is not None else a.url
    hdr = f"baseline={baseline} candidate={a.candidate or '(self-noise only)'} topk={a.topk}"
    print(f"p99-KL yardstick — {hdr}")
    floor = shape_floor = None
    floor_rows = shape_rows = None
    if capture is not None:
        base1 = replay_baseline(capture, texts, pos)
        print("  [pass 1/2 baseline replay] done", flush=True)
        if a.shape_floor > 0 and not capture.get("shape_floor"):
            raise SystemExit("--shape-floor was requested but the capture has no "
                             "shape floor — re-capture with --shape-floor")
        floor, shape_floor = replay_capture_floor(capture, texts)
    else:
        base1 = run_pass(a.url, a.model, texts, a)
        print("  [pass 1/3 baseline] done", flush=True)
        if pos:  # diff live tokenization against the frozen pin where pinned
            for t, (ids, _) in zip(texts, base1):
                frozen = pos["texts"][t["id"]]["token_ids"]
                if frozen is not None and ids != frozen:
                    raise SystemExit(f"tokenization of {t['id']} differs from the frozen "
                                     f"pin in {a.positions} — positions moved; do not judge "
                                     f"this delta, re-freeze on a verified host")
        if not a.no_floor:
            base2 = run_pass(a.url, a.model, texts, a)
            print("  [pass 2/3 baseline re-run] done", flush=True)
            floor_rows = score_all(texts, a, base1, base2)
            floor = pool(floor_rows)
            print_header("SELF-NOISE FLOOR (baseline vs itself) — judge nothing inside this")
            for t, st in zip(texts, floor_rows):
                print(fmt_row(f"floor {t['id']}", st))
            print(fmt_row("floor POOLED", floor))
            if a.shape_floor > 0:
                base3 = run_pass(a.url, a.model, texts, a, batch=a.shape_floor)
                print(f"  [pass 2b baseline re-run at batch={a.shape_floor}] done",
                      flush=True)
                shape_rows = score_all(texts, a, base1, base3)
                shape_floor = pool(shape_rows)
                print_header(f"BATCH-SHAPE FLOOR (baseline vs itself, "
                             f"batch={a.shape_floor}) — folded into the gate floor")
                for t, st in zip(texts, shape_rows):
                    print(fmt_row(f"floor(shape) {t['id']}", st))
                print(fmt_row("floor(shape) POOLED", shape_floor))

    if a.capture:
        capture_write(a.capture, capture_build(
            texts, base1, a, floor, floor_rows, shape_floor, shape_rows, pos))
        print(f"  [capture] baseline -> {a.capture}", flush=True)

    rows = None
    if a.candidate:
        cand = run_pass(a.candidate, a.model, texts, a)
        if capture is not None:
            check_replay_tokenization(capture, texts, cand)
        total = 2 if capture is not None else 3
        print(f"  [pass {total}/{total} candidate] done", flush=True)
        rows = score_all(texts, a, base1, cand)
        print_header("BASELINE vs CANDIDATE")
        for t, st in zip(texts, rows):
            print(fmt_row(t["id"], st))
        cand_st = pool(rows)
        print(fmt_row("POOLED", cand_st))
        if a.decision_ids is not None:
            ids = ",".join(str(i) for i in sorted(a.decision_ids)) or "(none)"
            print(f"\nDECISION POSITIONS  ids={ids}")
            print(f"{'comparison':<26}{'pos':>7}{'mean KL':>10}{'geo ratio':>13}"
                  f"{'ppl base':>10}{'ppl cand':>10}")
            print(fmt_decision(decision_stats(rows)))
    else:
        cand_st = None

    # machine-readable lines, floor first (docs/bench-gates.md)
    ok = True
    eff = effective_floor(floor, shape_floor)
    if floor:
        print(f"kl metric=floor_p99_kl value={floor['p99_kl']:.6f} threshold=- INFO")
        print(f"kl metric=floor_top1 value={floor['top1']:.6f} threshold=- INFO")
    if shape_floor:
        print(f"kl metric=floor_shape_p99_kl value={shape_floor['p99_kl']:.6f} "
              f"threshold=- INFO")
        print(f"kl metric=floor_shape_top1 value={shape_floor['top1']:.6f} "
              f"threshold=- INFO")
    if cand_st:
        p99_thr = a.mult * eff if eff is not None else a.max_p99
        p99_ok = p99_thr is not None and cand_st["p99_kl"] <= p99_thr
        if p99_thr is None:
            print(f"kl metric=p99_kl value={cand_st['p99_kl']:.6f} "
                  f"threshold=UNSET(no floor run, no --max-p99) FAIL")
        else:
            print(f"kl metric=p99_kl value={cand_st['p99_kl']:.6f} "
                  f"threshold={p99_thr:.6f} {'PASS' if p99_ok else 'FAIL'}")
        t1_ok = cand_st["top1"] >= a.min_top1
        print(f"kl metric=top1_agreement value={cand_st['top1']:.6f} "
              f"threshold={a.min_top1:.6f} {'PASS' if t1_ok else 'FAIL'}")
        print(f"kl metric=mean_kl value={cand_st['mean_kl']:.6f} threshold=- INFO")
        print(f"kl metric=ppl_delta value={cand_st['ppl_c']-cand_st['ppl_p']:+.6f} "
              f"threshold=- INFO")
        ok = p99_ok and t1_ok
    print(f"kl overall {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_spine(a):
    spine_dump(spine_build(load_corpus(a.corpus)), a.positions)
    doc = load_positions(a.positions)
    seg = sum(len(v["segments"]) for v in doc["texts"].values())
    print(f"spine -> {a.positions}: {len(doc['texts'])} texts, {seg} segments, "
          f"corpus_sha256={doc['corpus_sha256'][:16]}…")
    return 0


def cmd_freeze(a):
    """Pin the tokenized position list from a healthy endpoint into the
    positions file (one-time per corpus; compare-mode enforces it after)."""
    texts = load_corpus(a.corpus)
    doc = load_positions(a.positions) if os.path.isfile(a.positions) \
        else spine_build(texts)
    for t in texts:
        ids, _ = completions_logprobs(a.url, a.model, t["text"], 1, 1, a.timeout)
        doc["texts"][t["id"]]["token_ids"] = ids
        doc["texts"][t["id"]]["frozen_url"] = a.url
        doc["texts"][t["id"]]["frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"  {t['id']}: pinned {len(ids)} token positions", flush=True)
    spine_dump(doc, a.positions)
    print(f"positions frozen -> {a.positions}")
    return 0


def _self_check():
    p = {1: -0.1, 2: -2.5, 3: -7.0}
    q = {1: -0.1, 2: -2.5, 3: -7.0}
    assert kl_renorm(p, q) < 1e-9
    q2 = {1: -0.4, 2: -2.2, 3: -7.0}
    assert 0.01 < kl_renorm(p, q2) < 0.2
    # disjoint top-k: after union renorm the stand-in gap dominates -> KL ~= margin
    big = {i: -math.log(50) for i in range(50)}
    other = {i + 100: -math.log(50) for i in range(50)}
    assert 20.0 < kl_renorm(big, other) < 26.0
    assert top1(p) == 1
    vals = list(range(1, 101))
    assert percentile_nearest(vals, 0.99) == 99
    assert percentile_nearest([5], 0.99) == 5
    # score_pair: 3 tokens, one flip, positions must match (vLLM element shape)
    ids = [7, 8, 9]
    dp = [None, {"8": {"logprob": -0.05}, "1": {"logprob": -3.0},
                 "2": {"logprob": -5.0}},
          {"9": {"logprob": -0.2}, "3": {"logprob": -4.0}}]
    dc = [None, {"8": {"logprob": -0.06}, "1": {"logprob": -3.1},
                 "2": {"logprob": -5.1}},
          {"4": {"logprob": -0.1}, "9": {"logprob": -1.0},
           "3": {"logprob": -4.1}}]
    st = score_pair(dp, dc, ids, list(ids))
    assert st["n"] == 2 and st["top1"] == 0.5 and st["mean_kl"] > 0
    assert st["flips"][0][0] == 2
    try:
        score_pair(dp, dc, ids, [7, 8, 10])
        raise AssertionError("tokenization mismatch must abort")
    except SystemExit:
        pass
    # spine determinism
    t = [{"id": "x", "text": "a bb\n\n  ccc "}
         ]
    d = spine_build(t)
    assert d["texts"]["x"]["segments"] == [[0, 1], [2, 4], [8, 11]]
    assert spine_build(t)["corpus_sha256"] == d["corpus_sha256"]

    # capture/replay: capture a fixed baseline, replay it as the baseline and
    # score the same fixed pass as the candidate -- no server involved.
    class _CaptureArgs:
        url = "self-check"
        model = "self-check"
        topk = 64
        gen_tokens = 1
        batch = 1
        shape_floor = 0
        decision_ids = None

    cap_args = _CaptureArgs()
    fixed = [(ids, dp)]
    fixed_floor_rows = score_all(t, cap_args, fixed, fixed)
    fixed_floor = pool(fixed_floor_rows)
    cap = capture_build(t, fixed, cap_args, fixed_floor, fixed_floor_rows)
    assert cap["positions_pin_sha256"] is None
    assert cap["texts"]["x"]["token_ids"] == ids
    assert cap["texts"]["x"]["token_logprob"][1] == dp[1]["8"]["logprob"]
    with tempfile.TemporaryDirectory() as tmp:
        cap_path = os.path.join(tmp, "capture.json")
        capture_write(cap_path, cap)
        loaded = capture_load(cap_path)
        replay = replay_baseline(loaded, t)
        assert replay[0][0] == ids
        replay_st = pool(score_all(t, cap_args, replay, fixed))
        assert replay_st["top1"] == 1.0
        assert replay_st["mean_kl"] < 1e-12
        assert replay_st["p99_kl"] < 1e-12
        assert replay_st["ppl_p"] == replay_st["ppl_c"]
        # a live corpus that disagrees with the capture is refused
        try:
            replay_baseline(loaded, [{"id": "x", "text": "a different corpus"}])
            raise AssertionError("mismatched corpus hash must abort replay")
        except SystemExit as e:
            assert "corpus" in str(e)
        # a live candidate on different token positions is refused
        try:
            check_replay_tokenization(loaded, t, [(ids[:2], dp[:2])])
            raise AssertionError("mismatched candidate tokenization must abort")
        except SystemExit as e:
            assert "tokenization" in str(e)
        # the CLI path: --replay as baseline, a (fake) live candidate
        corpus_path = os.path.join(tmp, "corpus.jsonl")
        with open(corpus_path, "w") as f:
            f.write(json.dumps({"id": "x", "kind": "prose",
                                "text": t[0]["text"]}) + "\n")
        positions_path = os.path.join(tmp, "no-positions.json")
        calls = []

        def fake_pass(url, model, texts, args, batch=None):
            calls.append(url)
            return fixed

        orig = globals()["run_pass"]
        globals()["run_pass"] = fake_pass
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["compare", "--replay", cap_path,
                             "--candidate", "http://self-check",
                             "--corpus", corpus_path,
                             "--positions", positions_path])
            text = out.getvalue()
            assert code == 0, text
            assert calls == ["http://self-check"], calls
            assert "SELF-NOISE FLOOR" in text
            assert "kl metric=p99_kl value=0.000000 threshold=0.000000 PASS" in text
            assert "kl metric=top1_agreement value=1.000000" in text
            assert "kl overall PASS" in text
            # mismatched corpus is refused loudly through the CLI too
            bad_path = os.path.join(tmp, "bad.jsonl")
            with open(bad_path, "w") as f:
                f.write(json.dumps({"id": "x", "kind": "prose",
                                    "text": "different text"}) + "\n")
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    main(["compare", "--replay", cap_path,
                          "--candidate", "http://self-check",
                          "--corpus", bad_path,
                          "--positions", positions_path])
                raise AssertionError("mismatched corpus must abort the CLI")
            except SystemExit as e:
                assert "corpus" in str(e)
            # --url and --replay are mutually exclusive
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    main(["compare", "--url", "http://self-check",
                          "--replay", cap_path])
                raise AssertionError("--url and --replay must be exclusive")
            except SystemExit:
                pass
        finally:
            globals()["run_pass"] = orig
    print("self-check OK (capture/replay: top1=100% KL=0; corpus mismatch refused)")
    return 0


def parse_id_set(s):
    """Comma-separated token ids -> set of ints; an empty string stays empty."""
    return {int(x) for x in s.split(",") if x.strip()}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("compare", "spine", "freeze"), nargs="?")
    ap.add_argument("--url", default=None, help="baseline endpoint")
    ap.add_argument("--candidate", help="candidate endpoint (omit for floor-only run)")
    ap.add_argument("--capture", metavar="FILE",
                    help="write the baseline arm's per-position data and the "
                         "measured floor to FILE for a later --replay")
    ap.add_argument("--replay", metavar="FILE",
                    help="use a --capture file as the baseline arm (the "
                         "candidate stays live); mutually exclusive with --url")
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "Qwen3.8-Flash-Next-NVFP4"))
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--positions", default=POSITIONS)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--gen-tokens", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--no-floor", action="store_true",
                    help="skip the baseline re-run (p99 gate then needs --max-p99)")
    ap.add_argument("--mult", type=float, default=2.0,
                    help="max allowed p99 KL as a multiple of the floor")
    ap.add_argument("--min-top1", type=float, default=0.995,
                    help="min top-1 agreement (fraction)")
    ap.add_argument("--max-p99", type=float, default=None,
                    help="absolute p99 KL ceiling (used when no floor run)")
    ap.add_argument("--batch", type=int, default=1, metavar="N",
                    help="max concurrent completion requests per pass (default 1)")
    ap.add_argument("--shape-floor", type=int, default=0, metavar="N",
                    help="re-run the baseline with N concurrent requests and fold "
                         "that p99 KL into the floor (0 = off)")
    ap.add_argument("--decision-ids", type=parse_id_set, default=None, metavar="IDS",
                    help="comma-separated token ids scored as stop decisions; adds "
                         "a diagnostic row, never changes the verdict")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args(argv)
    if a.batch < 1:
        ap.error("--batch must be >= 1")
    if a.shape_floor < 0:
        ap.error("--shape-floor must be >= 0")
    if a.self_check:
        return _self_check()
    if a.mode is None:
        ap.error("a mode is required (compare, spine, or freeze)")
    if a.mode != "compare" and (a.capture or a.replay):
        ap.error("--capture/--replay are only valid with `compare`")
    if a.capture and a.replay:
        ap.error("--capture and --replay cannot be combined")
    if a.replay and a.url is not None:
        ap.error("--url and --replay are mutually exclusive")
    if a.replay and a.no_floor:
        ap.error("--no-floor cannot be combined with --replay "
                 "(the capture carries its floor)")
    if a.url is None:
        a.url = os.environ.get("BENCH_BASE", "http://localhost:8888")
    if a.mode == "spine":
        return cmd_spine(a)
    if a.mode == "freeze":
        return cmd_freeze(a)
    return cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
