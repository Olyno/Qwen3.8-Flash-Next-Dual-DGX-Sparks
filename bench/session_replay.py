#!/usr/bin/env python3
"""Replay the prompts of a recorded agent-harness session against a live endpoint.

Purpose: one paired instrument for the reasoning-effort question. The knob is the
chat template's `reasoning_effort` (xhigh default, medium, low) — prompt text, not a
hard token cap. The session file
already holds every request's message array, so the same prompts can be sent to the
same server with different budgets and compared on tokens and latency, without
executing any tool.

    session_replay.py extract SESSION.jsonl --out /tmp/replay.jsonl
    session_replay.py stats /tmp/replay.jsonl
    session_replay.py send /tmp/replay.jsonl --url http://localhost:8888 \\
        --model Qwen3.8-Flash-Next-NVFP4 --reasoning-effort low --out /tmp/armlow.jsonl
    session_replay.py compare /tmp/arm512.jsonl /tmp/armnone.jsonl
    session_replay.py --selftest

Limitation, stated up front: the recording does not contain the client's own system
prompt (only a digest of it), so the replay is not byte-identical to the original
request. It is identical *across arms*, which is what a paired comparison needs; the
absolute numbers are a lower bound on the real prompt.
"""
import argparse
import json
import sys
import time
import urllib.request


def ascii(s):
    return s.encode("ascii", "replace").decode("ascii") if isinstance(s, str) else s


def extract(path):
    """One request per recorded assistant turn: the messages that preceded it."""
    lines = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    history, out = [], []
    for rec in lines:
        if rec.get("type") != "message":
            continue
        msg = rec.get("message") or {}
        role = msg.get("role")
        if role == "assistant":
            if history:
                out.append({"turn": len(out) + 1,
                            "messages": json.loads(json.dumps(history))})
            history.append({"role": "assistant",
                            "content": msg.get("content") or "",
                            **({"tool_calls": msg["tool_calls"]}
                               if msg.get("tool_calls") else {})})
        elif role in ("user", "system"):
            content = msg.get("content")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content
                                   if isinstance(p, dict))
            history.append({"role": role, "content": content or ""})
        elif role == "toolResult":
            content = msg.get("content")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content
                                   if isinstance(p, dict))
            history.append({"role": "tool",
                            "tool_call_id": msg.get("toolCallId") or "",
                            "content": content or ""})
    return out


def trim(messages, budget_chars):
    """Keep the newest whole turns under a char budget, never splitting a
    tool call from its results. The client compacts too; this reproduction is
    an approximation and is honest about being one."""
    def size(m):
        return len(json.dumps(m))
    kept, used = [], 0
    i = len(messages) - 1
    while i >= 0:
        m = messages[i]
        group = [m]
        if m.get("role") == "tool":
            j = i - 1
            while j >= 0 and (messages[j].get("role") in ("tool", "assistant")):
                group.insert(0, messages[j])
                if messages[j].get("role") == "assistant":
                    break
                j -= 1
            i = j
        cost = sum(size(x) for x in group)
        if used + cost > budget_chars and kept:
            break
        kept = group + kept
        used += cost
        i -= 1
    return kept


def chars(requests):
    return sum(len(json.dumps(r["messages"])) for r in requests)


def cmd_extract(args):
    reqs = extract(args.session)
    if args.window_tokens:
        budget = int(args.window_tokens * 4)
        reqs = [{"turn": r["turn"], "messages": trim(r["messages"], budget)}
                for r in reqs]
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in reqs:
            fh.write(json.dumps(r) + "\n")
    print(f"extracted {len(reqs)} requests, {chars(reqs) / 4:.0f} approx prompt tokens "
          f"-> {args.out}")
    return 0


def cmd_stats(args):
    reqs = [json.loads(l) for l in open(args.path, encoding="utf-8") if l.strip()]
    if not reqs:
        print("empty")
        return 1
    sizes = [len(json.dumps(r["messages"])) / 4 for r in reqs]
    print(f"requests {len(reqs)}  prompt tokens approx: "
          f"min {min(sizes):.0f} p50 {sorted(sizes)[len(sizes) // 2]:.0f} "
          f"max {max(sizes):.0f}  total {sum(sizes):.0f}")
    return 0


def post(url, payload, timeout):
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    started = time.time()
    first = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.load(resp)
    elapsed = time.time() - started
    return body, elapsed, first


def cmd_send(args):
    reqs = [json.loads(l) for l in open(args.path, encoding="utf-8") if l.strip()]
    rows, started_all = [], time.time()
    for req in reqs:
        payload = {"model": args.model, "messages": req["messages"],
                   "max_tokens": args.max_tokens, "temperature": 0}
        if args.reasoning_effort:
            payload["chat_template_kwargs"] = {
                "reasoning_effort": args.reasoning_effort}
        body, elapsed, _ = post(args.url, payload, args.timeout)
        usage = body.get("usage") or {}
        choice = (body.get("choices") or [{}])[0].get("message") or {}
        rows.append({"turn": req["turn"], "seconds": round(elapsed, 2),
                     "prompt": usage.get("prompt_tokens"),
                     "completion": usage.get("completion_tokens"),
                     "reasoning": (usage.get("completion_tokens_details") or {})
                     .get("reasoning_tokens"),
                     "tool_calls": [tc.get("function", {}).get("name")
                                    for tc in (choice.get("tool_calls") or [])],
                     "reply_head": (choice.get("content") or "")[:160]})
        print(f"  turn {req['turn']:3d}  {elapsed:6.1f}s  "
              f"out {rows[-1]['completion']}  think {rows[-1]['reasoning']}")
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    wall = time.time() - started_all
    print(f"{len(rows)} turns, {wall:.0f}s wall, "
          f"{sum(r['completion'] or 0 for r in rows)} completion tokens, "
          f"{sum(r['reasoning'] or 0 for r in rows)} reasoning")
    return 0


def cmd_compare(a_path, b_path):
    a = [json.loads(l) for l in open(a_path, encoding="utf-8") if l.strip()]
    b = [json.loads(l) for l in open(b_path, encoding="utf-8") if l.strip()]
    n = min(len(a), len(b))
    ta = sum(r["completion"] or 0 for r in a[:n])
    tb = sum(r["completion"] or 0 for r in b[:n])
    ra = sum(r["reasoning"] or 0 for r in a[:n])
    rb = sum(r["reasoning"] or 0 for r in b[:n])
    sa = sum(r["seconds"] for r in a[:n])
    sb = sum(r["seconds"] for r in b[:n])
    print(f"turns {n}")
    print(f"  completion {ta} -> {tb}  ({100 * (tb - ta) / max(ta, 1):+.1f}%)")
    print(f"  reasoning  {ra} -> {rb}  ({100 * (rb - ra) / max(ra, 1):+.1f}%)")
    print(f"  wall       {sa:.0f}s -> {sb:.0f}s  ({100 * (sb - sa) / max(sa, 1):+.1f}%)")
    agree = sum(1 for i in range(n)
                if a[i].get("tool_calls") == b[i].get("tool_calls"))
    print(f"  tool-call agreement {agree}/{n} ({100 * agree / max(n, 1):.0f}%)")
    return 0


def cmd_turns(args):
    """Turn-level shape of a recorded session: what the contract is meant to change."""
    lines = [json.loads(l) for l in open(args.path, encoding="utf-8") if l.strip()]
    turns, calls, tool_secs, reasoning, completion, prompt = [], 0, 0.0, 0, 0, 0
    for rec in lines:
        if rec.get("type") != "message":
            continue
        msg = rec.get("message") or {}
        if msg.get("role") == "assistant":
            parts = msg.get("content") or []
            if isinstance(parts, str):
                parts = [{"type": "text"}]
            n = sum(1 for p in parts
                    if isinstance(p, dict)
                    and p.get("type") in ("toolCall", "tool-call", "tool_use"))
            turns.append(n)
            calls += n
            usage = rec.get("usage") or msg.get("usage") or {}
            completion += usage.get("output") or usage.get("completion_tokens") or 0
            reasoning += (usage.get("reasoningTokens") or usage.get("reasoning") or
                          (usage.get("completion_tokens_details") or {})
                          .get("reasoning_tokens") or 0)
            prompt += usage.get("input") or usage.get("prompt_tokens") or 0
    multi = sum(1 for n in turns if n > 1)
    print(f"turns {len(turns)}  tool calls {calls}  "
          f"single-call turns {turns.count(1)}  multi-call turns {multi}")
    print(f"completion {completion}  reasoning {reasoning}  prompt {prompt}")
    return 0


def selftest():
    import tempfile
    session = [
        {"type": "message", "message": {"role": "user", "content": "do it"}},
        {"type": "message", "message": {"role": "assistant", "content": "ok",
                                        "tool_calls": [{"id": "c1"}]}},
        {"type": "message", "message": {"role": "toolResult", "toolCallId": "c1",
                                        "content": "result text"}},
        {"type": "message", "message": {"role": "assistant", "content": "done"}},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        for rec in session:
            fh.write(json.dumps(rec) + "\n")
        path = fh.name
    reqs = extract(path)
    assert len(reqs) == 2, reqs
    assert reqs[0]["messages"] == [{"role": "user", "content": "do it"}], reqs[0]
    assert reqs[1]["messages"][1] == {"role": "assistant", "content": "ok",
                                      "tool_calls": [{"id": "c1"}]}, reqs[1]
    assert reqs[1]["messages"][2]["role"] == "tool"
    print("SELFTEST PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    e = sub.add_parser("extract")
    e.add_argument("session")
    e.add_argument("--out", required=True)
    e.add_argument("--window-tokens", type=int, default=12000,
                   help="cap the reconstructed prompt (0 = no cap); the real client "
                        "compacts, this is the closest reproduction")
    s = sub.add_parser("stats")
    s.add_argument("path")
    d = sub.add_parser("send")
    d.add_argument("path")
    d.add_argument("--url", default="http://localhost:8888")
    d.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    d.add_argument("--max-tokens", type=int, default=2048)
    d.add_argument("--timeout", type=int, default=600)
    d.add_argument("--reasoning-effort", default=None,
                   choices=["xhigh", "medium", "low"],
                   help="sent as chat_template_kwargs.reasoning_effort; the chat "
                        "template default is xhigh and it is prompt text, not a "
                        "hard token cap")
    d.add_argument("--out", required=True)
    t = sub.add_parser("turns")
    t.add_argument("path")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.cmd == "extract":
        return cmd_extract(args)
    if args.cmd == "stats":
        return cmd_stats(args)
    if args.cmd == "send":
        return cmd_send(args)
    if args.cmd == "turns":
        return cmd_turns(args)
    if args.cmd == "compare":
        return cmd_compare(args.a, args.b)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
