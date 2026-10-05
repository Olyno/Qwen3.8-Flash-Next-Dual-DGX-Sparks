"""bench/omniscience.py — AA-Omniscience runner + same-server judge.

Public subset: https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public
(600 of the paper's 6,000 questions). Two passes against an OpenAI-compatible
server: answer generation (temp 0), then a judge pass classifying each answer
as CORRECT / INCORRECT / PARTIAL_ANSWER / NOT_ATTEMPTED (AA's categories).
Reports Accuracy and Non-Hallucination Rate — the two Intelligence Index
components. The official grader is GPT-5.6 Luna; a same-server judge makes
scores directional, not leaderboard-comparable, but the bias is identical on
both A/B arms.

Resumable: question_ids already present in --out are skipped on both passes.

  python3 bench/omniscience.py --url http://localhost:8888 --model Qwen3.8-Flash-Next-NVFP4 \
      --csv bench/data/aa_omniscience_public.csv --out bench/data/omni_prod.jsonl
"""

import argparse
import csv
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

LABELS = ("CORRECT", "INCORRECT", "PARTIAL_ANSWER", "NOT_ATTEMPTED")

JUDGE_PROMPT = """You are grading a factual answer against ground truth.

Question: {question}
Ground truth: {gold}
Model answer: {response}

Grade the model answer with exactly one label:
- CORRECT: contains the ground-truth fact, no significant wrong claims
- PARTIAL_ANSWER: on-topic and partly right but missing the key fact
- INCORRECT: asserts wrong information (a hallucinated guess)
- NOT_ATTEMPTED: declines, says it does not know, or gives no answer

Reply with the label only."""


def chat(url, model, messages, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        choice = json.load(r)["choices"][0]
    return choice["message"].get("content") or "", choice.get("finish_reason")


def classify(text):
    for label in LABELS:
        if label in text.upper():
            return label
    return "NOT_ATTEMPTED"  # ponytail: unparseable judge output ≈ no grade; ~0 in practice, revisit if it skews


def score(rows):
    n = len(rows)
    counts = {label: sum(1 for r in rows if r["verdict"] == label) for label in LABELS}
    attempted = n - counts["NOT_ATTEMPTED"]
    return {
        "n": n,
        **counts,
        "accuracy": counts["CORRECT"] / n if n else 0,
        "hallucination_rate": counts["INCORRECT"] / attempted if attempted else 0,
        "non_hallucination_rate": 1 - (counts["INCORRECT"] / attempted) if attempted else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8888")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke)")
    a = ap.parse_args()

    with open(a.csv, newline="") as f:
        questions = list(csv.DictReader(f))
    if a.limit:
        questions = questions[: a.limit]

    done = set()
    try:
        with open(a.out) as f:
            for line in f:
                done.add(json.loads(line)["question_id"])
    except FileNotFoundError:
        pass
    todo = [q for q in questions if q["question_id"] not in done]
    print(f"{len(questions)} questions, {len(done)} done, {len(todo)} to run", flush=True)

    out = open(a.out, "a")

    def run_one(q):
        msgs = [{"role": "user", "content": q["question"]}]
        response, finish = chat(a.url, a.model, msgs, a.max_tokens)
        if not response and finish == "length":
            # reasoning ate the budget — retry once with 2x
            response, finish = chat(a.url, a.model, msgs, a.max_tokens * 2)
        verdict = classify(chat(a.url, a.model, [
            {"role": "user", "content": JUDGE_PROMPT.format(
                question=q["question"], gold=q["answer"], response=response)}],
            256)[0])
        return {
            "question_id": q["question_id"], "domain": q["domain"],
            "topic": q["topic"], "question": q["question"], "gold": q["answer"],
            "response": response, "finish_reason": finish, "verdict": verdict,
        }

    rows = []
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        for i, row in enumerate(pool.map(run_one, todo), 1):
            out.write(json.dumps(row) + "\n")
            out.flush()
            rows.append(row)
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    out.close()

    with open(a.out) as f:
        all_rows = [json.loads(line) for line in f]
    print(json.dumps(score(all_rows), indent=2))


def _self_check():
    rows = [{"verdict": v} for v in
            ("CORRECT", "CORRECT", "INCORRECT", "PARTIAL_ANSWER", "NOT_ATTEMPTED")]
    s = score(rows)
    assert s["accuracy"] == 0.4
    assert abs(s["hallucination_rate"] - 0.25) < 1e-9
    assert classify("correct") == "CORRECT"
    assert classify("I would say: PARTIAL_ANSWER.") == "PARTIAL_ANSWER"
    assert classify("garbage") == "NOT_ATTEMPTED"
    print("self-check OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _self_check()
    else:
        main()
