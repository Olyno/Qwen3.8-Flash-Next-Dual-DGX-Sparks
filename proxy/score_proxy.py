#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""proxy/score_proxy.py — thin OpenAI-compatible decision proxy for vLLM.

Passthrough reverse proxy (SSE streams through chunk-by-chunk) plus two
decision features — see docs/score-proxy.md:

  POST /v1/score              score N options in one batched prefill
  POST /v1/chat/completions   confidence-gated thinking escalation
                              ("escalate": true or PROXY_ESCALATE=1);
                              streaming requests get a degenerate-repetition
                              loop guard (PROXY_LOOP_GUARD=0 disables).
                              Opt-in cache care: PROXY_CANONICALIZE=1
                              hoists system blocks ahead of the first user
                              message (reorder only; PROXY_CANONICALIZE_STRIP_JITTER=1
                              also strips trailing 'Current time:'/'Date:'
                              lines from system messages);
                              PROXY_CACHE_KEEPALIVE_S=N re-touches APC
                              recency for clients idle past N seconds.

Env: PROXY_UPSTREAM (default http://localhost:8888), PROXY_PORT (8889),
PROXY_ESCALATE, PROXY_ESCALATE_THRESHOLD (-0.35),
PROXY_LOOP_GUARD (1), PROXY_LOOP_GUARD_REPEAT (3). Stdlib only.
"""
import http.client
import json
import math
import os
import re
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP_BY_HOP = frozenset(("connection", "keep-alive", "proxy-authenticate",
                        "proxy-authorization", "te", "trailer",
                        "transfer-encoding", "upgrade"))


def softmax(scores):
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def build_prefix(question, options):
    # JEV prompt rules: one multi-option question, use-when descriptions inline.
    lines = [question.strip(), "", "Options:"]
    for i, opt in enumerate(options, 1):
        line = f"{i}. {opt['label']}"
        if opt.get("use_when"):
            line += f" — use when: {opt['use_when']}"
        lines.append(line)
    lines += ["", "Answer: "]
    return "\n".join(lines)


MIN_PHRASE = 20   # shortest repeated phrase (normalized chars) the guard sees
WINDOW = 2048     # rolling normalized text kept for detection
PAIR_RUN = 16     # 'ab' x PAIR_RUN (2-char unit) also counts as a loop


def _degenerate_tail(text, repeat):
    """True if the normalized text ends in the same >=MIN_PHRASE phrase
    repeated `repeat` times, or in a 2-char unit repeated PAIR_RUN times.
    Tail-anchored on purpose: legit repetition that already ended (code,
    tables) must not trigger — only a loop still running at the frontier."""
    n = len(text)
    if n >= 2 * PAIR_RUN and text.endswith(text[n - 2:] * PAIR_RUN):
        return True
    for p in range(MIN_PHRASE, n // repeat + 1):
        if text.endswith(text[n - p:] * repeat):
            return True
    return False


class LoopGuard:
    """Streaming repetition detector for SSE chat responses. Raw chunks are
    relayed untouched; only delta text is kept (rolling, normalized) for
    detection. ponytail: tail-anchored check + 20-char minimum can still
    false-positive on pathological legit output (e.g. 3+ identical long JSON
    rows back-to-back, or 32+ dashes of table separator at the frontier) —
    PROXY_LOOP_GUARD=0 is the escape hatch."""

    def __init__(self, repeat):
        self.repeat = repeat
        self._pending = b""  # bytes of an SSE event split across chunks
        self._text = ""      # rolling normalized window

    def feed(self, chunk):
        """Inspect one raw upstream chunk; True => degenerate loop detected."""
        *events, self._pending = (self._pending + chunk).split(b"\n\n")
        for event in events:
            for line in event.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == b"[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue  # unparseable event: relay it, just don't score it
                delta = ((obj.get("choices") or [{}])[0].get("delta") or {})
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    self._text = (self._text
                                  + " ".join(piece.split()).lower())[-WINDOW:]
                    if _degenerate_tail(self._text, self.repeat):
                        return True
        return False

    @staticmethod
    def final_chunk():
        # finish_reason 'stop' (not 'content_filter'/'length': neither is what
        # happened — the proxy cut a loop). The cut stays machine-detectable
        # via a top-level extension field; delta text is left unpolluted and
        # SSE framing stays valid.
        note = json.dumps({"choices": [{"index": 0, "delta": {},
                                        "finish_reason": "stop"}],
                           "x_proxy": {"loop_guard": "cut"}})
        return f"data: {note}\n\ndata: [DONE]\n\n".encode()


def post_json(upstream, path, body):
    conn = http.client.HTTPConnection(upstream[0], upstream[1], timeout=300)
    try:
        conn.request("POST", path, body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


def score_order(upstream, model, prefix, labels):
    """One batched echo-logprobs call: prompt = prefix + label per option;
    sum each label's token logprobs past the shared prefix, softmax."""
    body = json.dumps({
        "model": model,
        "prompt": [prefix + label for label in labels],
        "echo": True, "logprobs": 0, "max_tokens": 1, "temperature": 0,
    }).encode()
    status, resp = post_json(upstream, "/v1/completions", body)
    if status != 200:
        raise ValueError(f"upstream /v1/completions returned {status}")
    scores = []
    for choice in sorted(resp["choices"], key=lambda c: c["index"]):
        lp = choice["logprobs"]
        scores.append(sum(t for t, off in zip(lp["token_logprobs"], lp["text_offset"])
                          if off >= len(prefix) and t is not None))
    return softmax(scores)


JITTER_RE = re.compile(
    r"(?:\s*\n\s*(?:Current (?:date|time|datetime)|Date|Time)\s*:[^\n]*)+\s*$")


def canonicalize_messages(messages, strip_jitter=False):
    """Normalize the messages array to protect the prefix cache ('Don't
    Break the Cache', arXiv 2601.06007): system messages sitting before the
    first user message are hoisted to the front in stable order, so static
    blocks stay ahead of the cache boundary. Content is never edited unless
    strip_jitter=True, and then only trailing 'Current time:'/'Date:'-style
    lines inside system messages are dropped. Returns (messages, changed)."""
    head_end = next((i for i, m in enumerate(messages)
                     if isinstance(m, dict) and m.get("role") == "user"),
                    len(messages))
    head, tail = messages[:head_end], messages[head_end:]
    is_system = lambda m: isinstance(m, dict) and m.get("role") == "system"
    out = [m for m in head if is_system(m)] + \
          [m for m in head if not is_system(m)] + tail
    if strip_jitter:
        stripped = []
        for m in out:
            if is_system(m) and isinstance(m.get("content"), str):
                content = JITTER_RE.sub("", m["content"])
                if content != m["content"]:
                    m = {**m, "content": content}
            stripped.append(m)
        out = stripped
    return out, out != messages


class CacheKeepalive(threading.Thread):
    """Re-touch APC recency during idle gaps (arXiv 2607.19214): when a
    client has been silent past the interval AND the proxy saw zero traffic
    in that window, replay its first system+user prefix with max_tokens=1.
    ponytail: one dict guarded only by the GIL; a lost update just means a
    ping fires one tick late."""

    def __init__(self, server, interval):
        super().__init__(daemon=True)
        self.server = server
        self.interval = interval
        self.clients = {}        # client ip -> last prefix + touch time
        self.last_traffic = 0.0

    def note_traffic(self):
        self.last_traffic = time.monotonic()

    def note_prefix(self, client, model, messages):
        prefix = [m for m in messages
                  if isinstance(m, dict) and m.get("role") == "system"][:1]
        prefix += [m for m in messages
                   if isinstance(m, dict) and m.get("role") == "user"][:1]
        if prefix:
            self.clients[client] = {"t": time.monotonic(), "model": model,
                                    "messages": prefix}

    def run(self):
        while True:
            time.sleep(max(0.05, min(self.interval / 4, 60)))
            now = time.monotonic()
            if now - self.last_traffic < self.interval:
                continue  # mid-request flood: don't add to it
            for c in list(self.clients.values()):
                if now - c["t"] < self.interval or not c["model"]:
                    continue
                body = json.dumps({"model": c["model"],
                                   "messages": c["messages"],
                                   "max_tokens": 1}).encode()
                try:
                    post_json(self.server.upstream, "/v1/chat/completions", body)
                except Exception:
                    pass  # upstream down: retry next interval
                c["t"] = now


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()

    def do_PUT(self):
        self._handle()

    def do_PATCH(self):
        self._handle()

    def do_DELETE(self):
        self._handle()

    def _handle(self):
        if self.server.keepalive is not None:
            self.server.keepalive.note_traffic()
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        try:
            if self.command == "POST" and self.path == "/v1/score":
                self._score(body)
            elif self.command == "POST" and self.path == "/v1/chat/completions":
                self._chat(body)
            else:
                self._passthrough(body)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            self._send_json(400, {"error": str(e)})

    def _passthrough(self, body, guard=None):
        conn = http.client.HTTPConnection(*self.server.upstream, timeout=600)
        try:
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in HOP_BY_HOP | {"host", "content-length"}}
            conn.request(self.command, self.path, body=body, headers=headers)
            self._relay(conn.getresponse(), guard)
        except (OSError, http.client.HTTPException):
            self._send_json(502, {"error": "upstream unreachable"})
        finally:
            conn.close()  # on a guard cut this aborts upstream generation

    def _relay(self, resp, guard=None):
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in HOP_BY_HOP:
                self.send_header(k, v)
        self.end_headers()
        if resp.getheader("Content-Length") is None:
            # streamed (e.g. SSE): nothing to frame against, delimit by close
            self.close_connection = True
        while chunk := resp.read1(65536):
            self.wfile.write(chunk)
            self.wfile.flush()
            if guard is not None and guard.feed(chunk):
                self.wfile.write(guard.final_chunk())
                self.wfile.flush()
                break

    def _score(self, body):
        req = json.loads(body)
        model, question = req["model"], req["question"]
        options = [{"label": o} if isinstance(o, str)
                   else {"label": o["label"], "use_when": o.get("use_when")}
                   for o in req["options"]]
        if not options:
            raise ValueError("options must be a non-empty list")
        labels = [o["label"] for o in options]
        probs = score_order(self.server.upstream, model,
                            build_prefix(question, options), labels)
        averaged = False
        if req.get("average_orders", True) and len(options) > 1:
            # JEV position-bias mitigation: rescore with the list reversed
            # and average the two distributions.
            rev = score_order(self.server.upstream, model,
                              build_prefix(question, list(reversed(options))), labels)
            probs = [(a + b) / 2 for a, b in zip(probs, rev)]
            averaged = True
        ranked = sorted(zip(labels, probs), key=lambda x: -x[1])
        self._send_json(200, {
            "ranked": [{"option": label, "prob": p} for label, p in ranked],
            "orders_averaged": averaged,
        })

    def _chat(self, body):
        req = json.loads(body)
        if (self.server.canonicalize
                and isinstance(req.get("messages"), list)):
            messages, changed = canonicalize_messages(
                req["messages"], self.server.strip_jitter)
            if changed:
                print(f"score proxy: canonicalized chat messages "
                      f"({len(messages)} messages)", file=sys.stderr)
                req["messages"] = messages
                body = json.dumps(req).encode()
        if self.server.keepalive is not None:
            self.server.keepalive.note_prefix(
                self.client_address[0], req.get("model"),
                req.get("messages") or [])
        escalate = self.server.escalate_env or req.get("escalate") is True
        if req.get("stream"):
            # escalation needs the full response; streams pass through with
            # only the loop guard watching the deltas
            guard = (LoopGuard(self.server.loop_repeat)
                     if self.server.loop_guard else None)
            return self._passthrough(body, guard)
        if not escalate:
            return self._passthrough(body)
        # pass 1: thinking off, logprobs on
        p1 = dict(req)
        kwargs = dict(req.get("chat_template_kwargs") or {})
        kwargs["enable_thinking"] = False
        p1["chat_template_kwargs"] = kwargs
        p1["logprobs"] = True
        p1["stream"] = False
        status, resp = post_json(self.server.upstream, "/v1/chat/completions",
                                 json.dumps(p1).encode())
        if status != 200:
            return self._send_json(status, resp)
        tokens = ((resp.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or []
        vals = [t["logprob"] for t in tokens if t.get("logprob") is not None]
        mean = sum(vals) / len(vals) if vals else 0.0
        if mean >= self.server.threshold:
            resp["x_proxy"] = {"escalated": False, "mean_logprob": mean}
            return self._send_json(status, resp)
        # low confidence: re-issue the original request unchanged (thinking on)
        status, resp = post_json(self.server.upstream, "/v1/chat/completions", body)
        resp["x_proxy"] = {"escalated": True, "mean_logprob": mean}
        self._send_json(status, resp)

    def _send_json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def make_server(port, upstream_url, escalate_env=False, threshold=-0.35,
                loop_guard=True, loop_repeat=3, canonicalize=False,
                strip_jitter=False, keepalive_s=0):
    u = urllib.parse.urlparse(upstream_url)
    srv = ThreadingHTTPServer(("", port), Handler)
    srv.upstream = (u.hostname, u.port or 80)
    srv.escalate_env = escalate_env
    srv.threshold = threshold
    srv.loop_guard = loop_guard
    srv.loop_repeat = loop_repeat
    srv.canonicalize = canonicalize
    srv.strip_jitter = strip_jitter
    srv.keepalive = None
    if keepalive_s > 0:
        srv.keepalive = CacheKeepalive(srv, keepalive_s)
        srv.keepalive.start()
    return srv


def main():
    srv = make_server(
        int(os.environ.get("PROXY_PORT", 8889)),
        os.environ.get("PROXY_UPSTREAM", "http://localhost:8888"),
        os.environ.get("PROXY_ESCALATE") == "1",
        # ponytail: -0.35 is a heuristic default — calibrate per workload
        # on real traffic before trusting the escalation gate.
        float(os.environ.get("PROXY_ESCALATE_THRESHOLD", -0.35)),
        os.environ.get("PROXY_LOOP_GUARD", "1") != "0",
        int(os.environ.get("PROXY_LOOP_GUARD_REPEAT", 3)),
        os.environ.get("PROXY_CANONICALIZE") == "1",
        os.environ.get("PROXY_CANONICALIZE_STRIP_JITTER") == "1",
        float(os.environ.get("PROXY_CACHE_KEEPALIVE_S", 0) or 0))
    print(f"score proxy on :{srv.server_port} -> "
          f"http://{srv.upstream[0]}:{srv.upstream[1]}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
