#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Thinking-compliance measurement for an OpenAI-compatible chat endpoint.

Disabling a thinking mode does not reliably remove visible reasoning, so three
metrics are measured per arm (thinking on, thinking off):

  empty_thinking     the pre-answer trace is empty or shorter than
                     --min-trace-chars (answer-only compliance)
  relevance          cosine between question and trace embeddings via
                     --embed-url; skipped when absent or the trace is empty
  explicit_inference share of traces an LLM judge marks as containing
                     explicit step-by-step inference (--judge)

The served template asks for a minimum draft per thinking step and a final
answer after a ``####`` separator (see chat_template_cod.jinja). Everything
before the last separator is the trace, the rest is the answer; when no
separator is present the text is split on the last ``\\boxed{``, and when
neither is present the whole text is the answer and the trace is empty.

    python3 bench/thinking_compliance.py --url http://localhost:8888 \\
        --model MODEL --arm both --embed-url http://localhost:8888 \\
        --out thinking-report.json

The nothink arm sends ``chat_template_kwargs={"enable_thinking": false}``;
the thinking arm sends no ``chat_template_kwargs`` at all. Exit status is 0
unless --fail-empty-below RATE is given and an arm's empty-thinking rate
falls below it.
"""
import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CORPUS = os.path.join(DATA, "thinking_prompts.jsonl")
TRACE_START = "<<<TRACE>>>"
TRACE_END = "<<<END TRACE>>>"
JUDGE_INSTRUCTION = (
    "Does the text between the markers contain explicit step-by-step "
    "inference or deductions, rather than only restating the question or "
    "giving the answer? Reply YES or NO.")


# ---------- response parsing and metrics (pure, unit-tested) ----------

def normalize_ws(text):
    return " ".join((text or "").split())


def split_trace(text):
    """Split a response into the pre-answer trace and the final answer.

    The last ``####`` wins; without one, the last ``\\boxed{`` starts the
    answer; without either, the whole text is the answer and the trace is
    empty. Both halves are whitespace-normalized.
    """
    text = text or ""
    if "####" in text:
        trace, answer = text.rsplit("####", 1)
    elif "\\boxed{" in text:
        cut = text.rfind("\\boxed{")
        trace, answer = text[:cut], text[cut:]
    else:
        trace, answer = "", text
    return {"trace": normalize_ws(trace), "answer": normalize_ws(answer)}


def parse_judge(text):
    """Return the first YES/NO token in the judge reply, or None.

    Tokens are matched case-insensitively, so leading prose ("The answer is
    YES.") still counts while YES- or NO-like words that are not standalone
    tokens do not.
    """
    for token in re.findall(r"[A-Za-z]+", (text or "").upper()):
        if token in ("YES", "NO"):
            return token
    return None


def cosine(a, b):
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return None
    return dot / (na * nb)


def aggregate(rows, judge_enabled=None, relevance_enabled=None):
    """Per-arm counts and rates. Rows carry empty_thinking, trace_chars,
    relevance, judge_verdict ("YES"/"NO"/None) and judge_raw (None when the
    judge was not called)."""
    n = len(rows)
    empty = sum(1 for r in rows if r["empty_thinking"])
    traces = sum(1 for r in rows if r.get("trace_chars", 0) > 0)
    rel = [r["relevance"] for r in rows if r.get("relevance") is not None]
    verdicts = [r.get("judge_verdict") for r in rows
                if r.get("judge_verdict") in ("YES", "NO")]
    unparsed = sum(1 for r in rows
                   if r.get("judge_raw") is not None
                   and r.get("judge_verdict") not in ("YES", "NO"))
    if judge_enabled is None:
        judge_enabled = any(r.get("judge_raw") is not None for r in rows)
    if relevance_enabled is None:
        relevance_enabled = bool(rel)
    st = {"n": n,
          "empty_thinking": empty,
          "empty_thinking_rate": empty / n if n else None,
          "traces": traces}
    st["relevance"] = ({"n": len(rel),
                        "mean": sum(rel) / len(rel) if rel else None}
                       if relevance_enabled else None)
    if judge_enabled:
        yes = sum(1 for v in verdicts if v == "YES")
        st["explicit_inference"] = {
            "judged": len(verdicts), "yes": yes,
            "yes_rate": yes / len(verdicts) if verdicts else None,
            "unparsed": unparsed}
    else:
        st["explicit_inference"] = None
    return st


# ---------- transport ----------

def endpoint(base, path):
    """Join a base URL with an API path, tolerating a trailing /v1."""
    base = base.rstrip("/")
    if base.endswith("/v1"):
        return base + path
    return base + "/v1" + path


def post_json(url, payload, timeout=300.0, retries=3, retry_sleep=5.0):
    """POST JSON and return the decoded response. Retries transient failures
    with linear backoff; a 4xx response is not retried."""
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            last = RuntimeError(f"HTTP {e.code} from {url}: {detail}")
            if 400 <= e.code < 500:
                break
        except Exception as e:  # connection refused, timeout, bad JSON
            last = e
        if attempt + 1 < retries:
            time.sleep(retry_sleep * (attempt + 1))
    raise last


def chat_payload(model, question, max_tokens, thinking):
    payload = {"model": model,
               "messages": [{"role": "user", "content": question}],
               "temperature": 0,
               "max_tokens": max_tokens}
    if not thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def chat_text(url, model, question, max_tokens, timeout, thinking=True):
    data = post_json(endpoint(url, "/chat/completions"),
                     chat_payload(model, question, max_tokens, thinking),
                     timeout=timeout)
    try:
        return data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        raise RuntimeError("malformed chat response: "
                           + json.dumps(data)[:300])


def embed_vector(embed_url, model, text, timeout):
    data = post_json(endpoint(embed_url, "/embeddings"),
                     {"model": model, "input": [text]}, timeout=timeout)
    try:
        items = sorted(data["data"], key=lambda d: d.get("index", 0))
        return [float(x) for x in items[0]["embedding"]]
    except (KeyError, IndexError, TypeError, ValueError):
        raise RuntimeError("malformed embeddings response: "
                           + json.dumps(data)[:300])


def judge_prompt(question, trace):
    return (JUDGE_INSTRUCTION + "\n\nQuestion: " + question
            + "\n\n" + TRACE_START + "\n" + trace + "\n" + TRACE_END)


# ---------- measurement ----------

def load_prompts(path=CORPUS):
    prompts, seen = [], set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("id", "question"):
                if not row.get(key):
                    raise SystemExit(f"{path}: row missing {key!r}: {line.strip()}")
            if row["id"] in seen:
                raise SystemExit(f"{path}: duplicate prompt id {row['id']!r}")
            seen.add(row["id"])
            prompts.append({"id": str(row["id"]),
                            "kind": str(row.get("kind", "")),
                            "question": row["question"]})
    if not prompts:
        raise SystemExit(f"{path}: no prompts")
    return prompts


def evaluate_prompt(a, prompt, arm):
    """One generation plus the optional embedding and judge calls."""
    thinking = arm == "thinking"
    response = chat_text(a.url, a.model, prompt["question"], a.max_tokens,
                         a.timeout, thinking)
    parts = split_trace(response)
    trace, answer = parts["trace"], parts["answer"]
    trace_chars = len(trace)
    empty = trace_chars == 0 or trace_chars < a.min_trace_chars
    relevance = None
    if a.embed_url and trace_chars:
        qvec = embed_vector(a.embed_url, a.embed_model or a.model,
                            prompt["question"], a.timeout)
        tvec = embed_vector(a.embed_url, a.embed_model or a.model,
                            trace, a.timeout)
        relevance = cosine(qvec, tvec)
    judge_raw = judge_verdict = None
    if a.judge and trace_chars:
        judge_raw = chat_text(a.url, a.model,
                              judge_prompt(prompt["question"], trace),
                              a.judge_max_tokens, a.timeout, thinking=False)
        judge_verdict = parse_judge(judge_raw)
    return {"id": prompt["id"], "kind": prompt["kind"],
            "question": prompt["question"], "response": response,
            "trace": trace, "answer": answer, "trace_chars": trace_chars,
            "empty_thinking": empty, "relevance": relevance,
            "judge_raw": judge_raw, "judge_verdict": judge_verdict}


def run_arm(a, prompts, arm):
    print(f"\narm={arm}")
    print(f"  {'id':<24}{'kind':<16}{'empty':>6}{'chars':>7}"
          f"{'rel':>8}{'judge':>7}")
    rows = []
    for prompt in prompts:
        row = evaluate_prompt(a, prompt, arm)
        rows.append(row)
        rel = "-" if row["relevance"] is None else f"{row['relevance']:.3f}"
        verdict = row["judge_verdict"] or (
            "?" if row["judge_raw"] is not None else "-")
        print(f"  {row['id']:<24}{row['kind']:<16}"
              f"{'yes' if row['empty_thinking'] else 'no':>6}"
              f"{row['trace_chars']:>7}{rel:>8}{verdict:>7}", flush=True)
    return rows


def fmt_summary(name, st):
    rate = st["empty_thinking_rate"]
    rel = st["relevance"]
    judge = st["explicit_inference"]
    mean_rel = "-" if not rel or rel["mean"] is None else f"{rel['mean']:.4f}"
    judged = "-" if not judge else str(judge["judged"])
    yes = "-" if not judge else str(judge["yes"])
    yes_rate = ("-" if not judge or judge["yes_rate"] is None
                else f"{100 * judge['yes_rate']:.1f}")
    empty_rate = "-" if rate is None else f"{100 * rate:.1f}"
    return (f"{name:<10}{st['n']:>4}{st['empty_thinking']:>7}{empty_rate:>8}"
            f"{st['traces']:>7}{rel['n'] if rel else '-':>7}{mean_rel:>10}"
            f"{judged:>7}{yes:>5}{yes_rate:>7}")


def write_report(path, report):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True,
                    help="OpenAI-compatible base URL, e.g. http://localhost:8888")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default=CORPUS,
                    help=f"JSONL corpus (default: {CORPUS})")
    ap.add_argument("--arm", choices=("thinking", "nothink", "both"),
                    default="both")
    judge = ap.add_mutually_exclusive_group()
    judge.add_argument("--judge", dest="judge", action="store_true",
                       default=True, help="run the explicit-inference judge "
                                          "(default)")
    judge.add_argument("--no-judge", dest="judge", action="store_false")
    ap.add_argument("--embed-url",
                    help="base URL for /v1/embeddings; relevance is skipped "
                         "without it")
    ap.add_argument("--embed-model", help="embedding model (default: --model)")
    ap.add_argument("--out", help="write the JSON report here")
    ap.add_argument("--min-trace-chars", type=int, default=15,
                    help="traces shorter than this count as empty thinking")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--judge-max-tokens", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--fail-empty-below", type=float, default=None, metavar="RATE",
                    help="exit 1 when any measured arm's empty-thinking rate "
                         "is below RATE (0..1); off by default")
    a = ap.parse_args(argv)
    if a.min_trace_chars < 0:
        ap.error("--min-trace-chars must be >= 0")
    if a.max_tokens < 1 or a.judge_max_tokens < 1:
        ap.error("--max-tokens and --judge-max-tokens must be >= 1")
    if a.fail_empty_below is not None and not 0.0 <= a.fail_empty_below <= 1.0:
        ap.error("--fail-empty-below must be within 0..1")

    prompts = load_prompts(a.prompts)
    arms = ("thinking", "nothink") if a.arm == "both" else (a.arm,)
    out_arms = {}
    print(f"thinking compliance — {len(prompts)} prompts, "
          f"model={a.model}, arms={','.join(arms)}")
    for arm in arms:
        rows = run_arm(a, prompts, arm)
        st = aggregate(rows, judge_enabled=a.judge,
                       relevance_enabled=bool(a.embed_url))
        out_arms[arm] = dict(st)
        out_arms[arm]["chat_template_kwargs"] = (
            {} if arm == "thinking" else {"enable_thinking": False})
        out_arms[arm]["rows"] = rows

    print("\nsummary")
    print(f"{'arm':<10}{'n':>4}{'empty':>7}{'empty %':>8}{'traces':>7}"
          f"{'rel n':>7}{'mean rel':>10}{'judged':>7}{'yes':>5}{'yes %':>7}")
    print("-" * 72)
    for arm in arms:
        print(fmt_summary(arm, out_arms[arm]))

    report = {"url": a.url, "model": a.model, "prompts": a.prompts,
              "arms": out_arms,
              "config": {"min_trace_chars": a.min_trace_chars,
                         "max_tokens": a.max_tokens,
                         "temperature": 0,
                         "judge": a.judge,
                         "judge_max_tokens": a.judge_max_tokens,
                         "embed_url": a.embed_url,
                         "embed_model": a.embed_model or a.model}}
    if a.out:
        write_report(a.out, report)
        print(f"\nreport -> {a.out}")

    failed = False
    if a.fail_empty_below is not None:
        print("")
        for arm in arms:
            rate = out_arms[arm]["empty_thinking_rate"]
            ok = rate is not None and rate >= a.fail_empty_below
            failed = failed or not ok
            shown = "-" if rate is None else f"{rate:.6f}"
            print(f"gate arm={arm} empty_thinking_rate={shown} "
                  f"threshold={a.fail_empty_below:.6f} "
                  f"{'PASS' if ok else 'FAIL'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
