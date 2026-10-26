#!/usr/bin/env python3
"""Replay the prompts of a recorded agent session against a live endpoint.

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
import urllib.error
import urllib.request


def ascii(s):
    return s.encode("ascii", "replace").decode("ascii") if isinstance(s, str) else s


def _text(parts):
    if isinstance(parts, str):
        return parts
    out = []
    for part in parts or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            out.append(part.get("text") or "")
        elif part.get("type") == "thinking":
            continue  # the client's internal thinking parts are not valid input
    return "\n".join(t for t in out if t)


def _tool_calls(parts):
    calls = []
    for part in parts or []:
        if isinstance(part, dict) and part.get("type") == "toolCall":
            args = part.get("partialArgs")
            if not args:
                args = json.dumps(part.get("arguments") or {})
            calls.append({"id": part.get("id") or "",
                          "type": "function",
                          "function": {"name": part.get("name") or "",
                                       "arguments": args}})
    return calls


def extract(path):
    """One request per recorded assistant turn, in OpenAI message shape.

    The session format stores content as parts (thinking / text / toolCall) and tool results as
    their own records; the endpoint wants plain assistant messages plus
    tool_calls and tool_call_id. Thinking parts are dropped: they are not valid
    input, and the point is to measure how much the model *chooses* to think.
    """
    lines = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    history, out = [], []
    for rec in lines:
        if rec.get("type") != "message":
            continue
        msg = rec.get("message") or {}
        role = msg.get("role")
        if role == "assistant":
            if history:
                # Shallow list copy: every entry is a fresh dict whose payload
                # is never mutated afterwards, so this is the whole snapshot the
                # window needs. Serialising the history per turn instead made
                # extraction quadratic — a 22 MB transcript spent minutes in
                # json.dumps and produced nothing.
                out.append({"turn": len(out) + 1,
                            "messages": list(history)})
            entry = {"role": "assistant", "content": _text(msg.get("content"))}
            calls = _tool_calls(msg.get("content"))
            if calls:
                entry["tool_calls"] = calls
            history.append(entry)
        elif role in ("user", "system"):
            history.append({"role": role,
                            "content": _text(msg.get("content"))})
        elif role == "toolResult":
            history.append({
                "role": "tool",
                "tool_call_id": msg.get("toolCallId") or msg.get("tool_call_id") or "",
                "content": _text(msg.get("content")) or "(no output)"})
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
    if not any(m.get("role") == "user" for m in kept):
        for m in messages:  # the task itself must survive any window
            if m.get("role") == "user":
                kept = [m] + kept
                break
    return kept


def chars(requests):
    return sum(len(json.dumps(r["messages"])) for r in requests)


def sample_evenly(reqs, k):
    """Keep k requests spread across the session, endpoints included.

    A replay of a whole session is hours of prefill; the point of the gate is a
    realistic mix of prompt sizes, and that needs the start, the middle and the
    end rather than the first k turns, whose prompts all look alike.
    """
    if k <= 0 or k >= len(reqs):
        return reqs
    if k == 1:
        return [reqs[-1]]
    step = (len(reqs) - 1) / (k - 1)
    return [reqs[round(i * step)] for i in range(k)]


def cmd_extract(args):
    reqs = extract(args.session)
    if args.window_tokens:
        budget = int(args.window_tokens * 4)
        reqs = [{"turn": r["turn"], "messages": trim(r["messages"], budget)}
                for r in reqs]
    total = len(reqs)
    reqs = sample_evenly(reqs, args.sample)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in reqs:
            fh.write(json.dumps(r) + "\n")
    print(f"extracted {len(reqs)} requests"
          + (f" sampled from {total}" if len(reqs) != total else "")
          + f", {chars(reqs) / 4:.0f} approx prompt tokens -> {args.out}")
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


def _norm_args(raw):
    """Canonical form of a tool call's arguments: key-sorted JSON, so two arms
    can be compared on what they asked for, not just which tool they picked."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return raw
    try:
        return json.dumps(raw, sort_keys=True)
    except TypeError:
        return str(raw)


def _sse_rows(lines):
    """Parse a server-sent-event stream into the JSON payloads it carries."""
    for raw in lines:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        line = raw.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _merge_tool_call(acc, delta):
    """A streamed tool call arrives as fragments keyed by index."""
    slot = acc.setdefault(delta.get("index", 0), {"name": None, "args": ""})
    fn = delta.get("function") or {}
    if fn.get("name"):
        slot["name"] = fn["name"]
    if fn.get("arguments"):
        slot["args"] += fn["arguments"]
    return acc


def stream_row(url, payload, timeout, clock=time.monotonic):
    """One streamed request, split where the engine splits it.

    Time to the first token is the prefill (plus any queueing); the remainder is
    decode. `cached_tokens` says how much of the prompt the prefix cache served,
    which is the difference between a 13k prompt costing 5 s and costing 0.5 s.
    """
    payload = dict(payload, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    started = clock()
    ttft, content, calls, usage = None, [], {}, {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for event in _sse_rows(resp):
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta and ttft is None:
                        ttft = clock() - started
                    if delta.get("content"):
                        content.append(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        _merge_tool_call(calls, tc)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from None
    elapsed = clock() - started
    decode_s = elapsed - (ttft or 0.0)
    completion = usage.get("completion_tokens")
    return {
        "seconds": round(elapsed, 2),
        "ttft": None if ttft is None else round(ttft, 2),
        "decode_s": round(decode_s, 2),
        "prompt": usage.get("prompt_tokens"),
        "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "completion": completion,
        "reasoning": (usage.get("completion_tokens_details") or {})
        .get("reasoning_tokens"),
        "decode_tok_s": (round(completion / decode_s, 1)
                         if completion and decode_s > 0 else None),
        "tool_calls": [{"name": v["name"], "args": _norm_args(v["args"])}
                       for _, v in sorted(calls.items())],
        "reply_head": "".join(content)[:160],
    }


def summarise(rows):
    """Turn-level totals, with the prefill/decode split the goal is about."""
    turns = len(rows)
    if not turns:
        return {"turns": 0, "wall": 0.0, "wall_mean": 0.0, "ttft_mean": 0.0,
                "decode_mean": 0.0, "prompt": 0, "cached": 0, "cached_share": 0.0,
                "completion": 0, "decode_tok_s": 0.0}
    wall = sum(r.get("seconds") or 0.0 for r in rows)
    ttfts = [r["ttft"] for r in rows if r.get("ttft") is not None]
    decodes = [r["decode_s"] for r in rows if r.get("decode_s")]
    prompt = sum(r.get("prompt") or 0 for r in rows)
    cached = sum(r.get("cached") or 0 for r in rows)
    completion = sum(r.get("completion") or 0 for r in rows)
    dsum = sum(decodes)
    return {"turns": turns,
            "wall": round(wall, 1),
            "wall_mean": round(wall / turns, 1),
            "ttft_mean": round(sum(ttfts) / len(ttfts), 2) if ttfts else 0.0,
            "decode_mean": round(dsum / len(decodes), 2) if decodes else 0.0,
            "prompt": prompt,
            "cached": cached,
            "cached_share": round(cached / prompt, 3) if prompt else 0.0,
            "completion": completion,
            "decode_tok_s": round(completion / dsum, 1) if dsum else 0.0}


def cmd_send(args):
    reqs = [json.loads(l) for l in open(args.path, encoding="utf-8") if l.strip()]
    rows = []
    for req in reqs:
        payload = {"model": args.model, "messages": req["messages"],
                   "max_tokens": args.max_tokens, "temperature": 0}
        if args.reasoning_effort:
            payload["chat_template_kwargs"] = {
                "reasoning_effort": args.reasoning_effort}
        row = {"turn": req["turn"]}
        row.update(stream_row(args.url, payload, args.timeout))
        rows.append(row)
        print(f"  turn {req['turn']:3d}  {row['seconds']:6.1f}s  "
              f"prefill {row['ttft'] or 0:5.1f}  decode {row['decode_s']:5.1f}  "
              f"cached {row['cached']}/{row['prompt']}  out {row['completion']}")
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    s = summarise(rows)
    print(f"{s['turns']} turns, {s['wall']:.1f}s wall, "
          f"{s['wall_mean']:.1f}s per turn "
          f"(prefill {s['ttft_mean']:.1f}s, decode {s['decode_mean']:.1f}s)")
    print(f"  prompt {s['prompt']} tokens, cached {s['cached']} "
          f"({s['cached_share']:.0%}), completion {s['completion']}, "
          f"decode {s['decode_tok_s']:.1f} tok/s")
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
    def calls(row):
        return row.get("tool_calls") or []

    name_agree = arg_agree = 0
    for i in range(n):
        ca, cb = calls(a[i]), calls(b[i])
        if [c.get("name") for c in ca] == [c.get("name") for c in cb]:
            name_agree += 1
            if [c.get("args") for c in ca] == [c.get("args") for c in cb]:
                arg_agree += 1
    print(f"  tool-name agreement {name_agree}/{n} ({100 * name_agree / max(n, 1):.0f}%)")
    print(f"  tool-arg  agreement {arg_agree}/{n} ({100 * arg_agree / max(n, 1):.0f}%)")
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
        {"type": "message", "message": {"role": "user", "content": [
            {"type": "text", "text": "do it"}]}},
        {"type": "message", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "ok"},
            {"type": "toolCall", "id": "c1", "name": "bash",
             "arguments": {"cmd": "ls"}}]}},
        {"type": "message", "message": {"role": "toolResult", "toolCallId": "c1",
                                        "content": [{"type": "text",
                                                     "text": "result text"}]}},
        {"type": "message", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "done"}]}},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        for rec in session:
            fh.write(json.dumps(rec) + "\n")
        path = fh.name
    reqs = extract(path)
    assert len(reqs) == 2, reqs
    assert reqs[0]["messages"] == [{"role": "user", "content": "do it"}], reqs[0]
    second = reqs[1]["messages"]
    assert second[1]["role"] == "assistant" and second[1]["content"] == "ok", second[1]
    assert second[1]["tool_calls"][0]["function"]["name"] == "bash", second[1]
    assert second[2]["role"] == "tool" and second[2]["tool_call_id"] == "c1", second[2]
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
    e.add_argument("--sample", type=int, default=0,
                   help="keep N requests spread across the session instead of all of "
                        "them; a full session is hours of prefill to replay")
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
