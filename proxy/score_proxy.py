#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""proxy/score_proxy.py — thin OpenAI-compatible decision proxy for vLLM.

Passthrough reverse proxy (SSE streams through chunk-by-chunk) plus two
decision features — see docs/score-proxy.md:

  POST /v1/score              score N options in one batched prefill
  POST /v1/chat/completions   confidence-gated thinking escalation
                              ("escalate": true or PROXY_ESCALATE=1)

Env: PROXY_UPSTREAM (default http://localhost:8888), PROXY_PORT (8889),
PROXY_ESCALATE, PROXY_ESCALATE_THRESHOLD (-0.35). Stdlib only.
"""
import http.client
import json
import math
import os
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

    def _passthrough(self, body):
        conn = http.client.HTTPConnection(*self.server.upstream, timeout=600)
        try:
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in HOP_BY_HOP | {"host", "content-length"}}
            conn.request(self.command, self.path, body=body, headers=headers)
            self._relay(conn.getresponse())
        except (OSError, http.client.HTTPException):
            self._send_json(502, {"error": "upstream unreachable"})
        finally:
            conn.close()

    def _relay(self, resp):
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
        escalate = self.server.escalate_env or req.get("escalate") is True
        if req.get("stream") or not escalate:
            # streaming requests always pass through untouched
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


def make_server(port, upstream_url, escalate_env=False, threshold=-0.35):
    u = urllib.parse.urlparse(upstream_url)
    srv = ThreadingHTTPServer(("", port), Handler)
    srv.upstream = (u.hostname, u.port or 80)
    srv.escalate_env = escalate_env
    srv.threshold = threshold
    return srv


def main():
    srv = make_server(
        int(os.environ.get("PROXY_PORT", 8889)),
        os.environ.get("PROXY_UPSTREAM", "http://localhost:8888"),
        os.environ.get("PROXY_ESCALATE") == "1",
        # ponytail: -0.35 is a heuristic default — calibrate per workload
        # on real traffic before trusting the escalation gate.
        float(os.environ.get("PROXY_ESCALATE_THRESHOLD", -0.35)))
    print(f"score proxy on :{srv.server_port} -> "
          f"http://{srv.upstream[0]}:{srv.upstream[1]}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
