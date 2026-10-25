import contextlib
import copy
import io
import json
import math
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "proxy"))
import score_proxy

sp = score_proxy

BASE_LP = {"red": -0.2, "green": -2.0, "blue": -1.0}
POSITION_BONUS = 0.6  # fake position bias: the label listed first scores higher

SSE_CHUNKS = ('data: {"delta": "a"}\n\n', 'data: {"delta": "b"}\n\n', 'data: [DONE]\n\n')

USAGE_STREAM_CHUNKS = (
    'data: {"choices": [{"index": 0, "delta": {"content": "hi"}}]}\n\n',
    'data: {"choices": [], "usage": {"prompt_tokens": 321, '
    '"completion_tokens": 9, "total_tokens": 330, '
    '"prompt_tokens_details": {"cached_tokens": 300}, '
    '"completion_tokens_details": {"reasoning_tokens": 2}}}\n\n',
    'data: [DONE]\n\n',
)

LOOP_PIECE = "the same phrase again and again. "
LOOP_CHUNKS = tuple(
    f'data: {{"choices": [{{"index": 0, "delta": {{"content": {json.dumps(LOOP_PIECE)}}}}}], "finish_reason": null}}\n\n'
    for _ in range(60)) + ('data: [DONE]\n\n',)

HARD_KEYWORDS = ("prove", "design", "debug", "analy", "parser",
                 "trade-off", "refactor", "hard")

REPAIR_TOOLS = [{"type": "function", "function": {
    "name": "run",
    "parameters": {"type": "object", "properties": {
        "count": {"type": "integer"},
        "ratio": {"type": "number"},
        "verbose": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "items": {"type": "array", "items": {"type": "integer"}},
        "opts": {"type": "object", "properties": {"n": {"type": "integer"}}},
        "query": {"type": "string"},
    }},
}}]

MALFORMED_CALL = (
    '```json\n{"name": "run", "arguments": {"count": "5", '
    '"verbose": "true"}}\n```')
WELLFORMED_CALL = {
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "run", "arguments": '{"count": 5}'}}]},
        "finish_reason": "tool_calls",
    }],
}


def fake_embed(text):
    """2-d fake embedding: hard-keyword texts -> [0,1], 'ambiguous' -> [.5,.5],
    everything else -> [1,0]. Routes the proxy's built-in prototypes correctly."""
    t = text.lower()
    if "ambiguous" in t:
        return [0.5, 0.5]
    if any(k in t for k in HARD_KEYWORDS):
        return [0.0, 1.0]
    return [1.0, 0.0]


def softmax(xs):
    top = max(xs)
    exps = [math.exp(x - top) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


class FakeUpstream(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self._json(200, {"data": [{"id": "fake-model"}]})
        elif self.path == "/teapot":
            self._json(418, {"error": "I'm a teapot"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        self.server.requests.append((self.path, raw))
        self.server.last_headers = dict(self.headers)
        req = json.loads(raw)
        if self.path == "/v1/completions":
            self._completions(req)
        elif self.path == "/v1/embeddings":
            texts = req["input"] if isinstance(req["input"], list) else [req["input"]]
            self._json(200, {"data": [{"index": i, "embedding": fake_embed(t)}
                                      for i, t in enumerate(texts)]})
        elif self.path == "/v1/chat/completions":
            self._sse(req) if req.get("stream") else self._chat(req)
        else:
            self._json(404, {"error": "not found"})

    def _completions(self, req):
        choices = []
        for i, prompt in enumerate(req["prompt"]):
            idx = prompt.rfind("Answer: ")
            prefix = prompt[:idx + len("Answer: ")]
            label = prompt[idx + len("Answer: "):]
            position = next(int(line.split(".")[0])
                            for line in prefix.splitlines()
                            if line.split(". ", 1)[-1].split(" — ")[0] == label)
            lp = BASE_LP[label] + (POSITION_BONUS if position == 1 else 0.0)
            choices.append({
                "index": i, "text": prompt,
                "logprobs": {"tokens": ["prefix", label],
                             "token_logprobs": [None, lp],
                             "text_offset": [0, len(prefix)]},
            })
        self._json(200, {"choices": choices})

    def _chat(self, req):
        text = json.dumps(req["messages"])
        if "usage report" in text:
            self._json(200, {
                "choices": [{"index": 0,
                             "message": {"role": "assistant",
                                         "content": "usage answer"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1234, "completion_tokens": 56,
                          "total_tokens": 1290,
                          "prompt_tokens_details": {"cached_tokens": 1000},
                          "completion_tokens_details": {"reasoning_tokens": 7}}})
            return
        if "forever think" in text:
            self._json(200, {"choices": [{"index": 0,
                "message": {"role": "assistant", "content": "",
                            "reasoning_content": "still thinking"},
                "finish_reason": "length"}]})
            return
        if "truncated think" in text:
            if req.get("continue_final_message"):
                self._json(200, {"choices": [{"index": 0,
                    "message": {"role": "assistant",
                                "content": "recovered answer"},
                    "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 3, "total_tokens": 13}})
            else:
                self._json(200, {"choices": [{"index": 0,
                    "message": {"role": "assistant", "content": "",
                                "reasoning_content": "still thinking"},
                    "finish_reason": "length"}],
                    "usage": {"completion_tokens": 10, "total_tokens": 20}})
            return
        if "malformed call" in text:
            self._json(200, {"choices": [{"index": 0,
                "message": {"role": "assistant", "content": MALFORMED_CALL},
                "finish_reason": "stop"}]})
            return
        if "wellformed call" in text:
            self._json(200, WELLFORMED_CALL)
            return
        if "plain answer" in text:
            self._json(200, {"choices": [{"index": 0,
                "message": {"role": "assistant", "content": "plain answer"},
                "finish_reason": "stop"}]})
            return
        kwargs = req.get("chat_template_kwargs") or {}
        if kwargs.get("enable_thinking") is False:
            lp = -0.01 if "confident" in json.dumps(req["messages"]) else -1.0
            self._json(200, {"choices": [{"index": 0,
                "message": {"role": "assistant", "content": "no-think answer"},
                "logprobs": {"content": [{"token": "no", "logprob": lp},
                                         {"token": "-think", "logprob": lp}]}}]})
        else:
            self._json(200, {"choices": [{"index": 0,
                "message": {"role": "assistant", "content": "thinking answer"}}]})

    def _sse(self, req):
        text = json.dumps(req["messages"])
        loop = "loop me" in text
        chunks = (USAGE_STREAM_CHUNKS if "usage stream" in text
                  else LOOP_CHUNKS if loop else SSE_CHUNKS)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            try:
                self.wfile.write(chunk.encode())
                self.wfile.flush()
                # The loop-guard assertion counts events that reach the client,
                # which is whatever one read returns. Loopback coalesces
                # back-to-back writes, so space the loop frames out and let the
                # guard decide on one frame at a time.
                if loop:
                    time.sleep(0.002)
            except (BrokenPipeError, ConnectionResetError):
                break  # proxy loop guard cut the stream mid-write


class LoopGuardUnitTest(unittest.TestCase):
    def test_phrase_repetition(self):
        text = ("the same phrase again and again. " * 4).strip()
        self.assertTrue(score_proxy._degenerate_tail(text, 3))
        self.assertFalse(score_proxy._degenerate_tail(
            "the same phrase again and again. " * 2, 3))

    def test_two_char_run(self):
        self.assertTrue(score_proxy._degenerate_tail("abc" + "ab" * 16, 3))
        self.assertFalse(score_proxy._degenerate_tail("abc" + "ab" * 8, 3))

    def test_normal_text_untouched(self):
        self.assertFalse(score_proxy._degenerate_tail(
            "def f(x):\n    return x + 1  # short, ordinary code", 3))


class ScoreProxyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
        cls.upstream.requests = []
        cls.proxy = score_proxy.make_server(
            0, f"http://127.0.0.1:{cls.upstream.server_port}")
        for srv in (cls.upstream, cls.proxy):
            threading.Thread(target=srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.proxy.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.proxy.shutdown()
        cls.upstream.shutdown()
        cls.proxy.server_close()
        cls.upstream.server_close()

    def post(self, path, obj):
        req = urllib.request.Request(self.url + path, data=json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req))

    def test_passthrough_preserves_body_and_status(self):
        with urllib.request.urlopen(self.url + "/v1/models") as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(json.load(resp), {"data": [{"id": "fake-model"}]})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.url + "/teapot")
        self.assertEqual(ctx.exception.code, 418)
        with ctx.exception as err:
            self.assertEqual(json.load(err), {"error": "I'm a teapot"})

    def test_score_softmax_ranking(self):
        resp = self.post("/v1/score", {"model": "m", "question": "Pick a color.",
                                       "options": ["red", "green", "blue"],
                                       "average_orders": False})
        self.assertFalse(resp["orders_averaged"])
        # forward order: red is listed first and gets the position bonus
        expected = softmax([BASE_LP["red"] + POSITION_BONUS,
                            BASE_LP["green"], BASE_LP["blue"]])
        got = {r["option"]: r["prob"] for r in resp["ranked"]}
        for label, prob in zip(("red", "green", "blue"), expected):
            self.assertAlmostEqual(got[label], prob, places=6)
        self.assertEqual([r["option"] for r in resp["ranked"]], ["red", "blue", "green"])

    def test_score_order_averaging(self):
        resp = self.post("/v1/score", {"model": "m", "question": "Pick a color.",
                                       "options": [
                                           {"label": "red", "use_when": "warm"},
                                           {"label": "green"},
                                           {"label": "blue"}]})
        self.assertTrue(resp["orders_averaged"])
        fwd = dict(zip(("red", "green", "blue"),
                       softmax([BASE_LP["red"] + POSITION_BONUS,
                                BASE_LP["green"], BASE_LP["blue"]])))
        rev = dict(zip(("red", "green", "blue"),
                       softmax([BASE_LP["red"], BASE_LP["green"],
                                BASE_LP["blue"] + POSITION_BONUS])))
        got = {r["option"]: r["prob"] for r in resp["ranked"]}
        for label in ("red", "green", "blue"):
            self.assertAlmostEqual(got[label], (fwd[label] + rev[label]) / 2, places=6)
        self.assertAlmostEqual(sum(got.values()), 1.0, places=6)

    def test_escalation_below_threshold(self):
        resp = self.post("/v1/chat/completions",
                         {"model": "m", "escalate": True,
                          "messages": [{"role": "user", "content": "hard question"}]})
        self.assertEqual(resp["x_proxy"], {"escalated": True, "mean_logprob": -1.0})
        self.assertEqual(resp["choices"][0]["message"]["content"], "thinking answer")

    def test_no_escalation_above_threshold(self):
        resp = self.post("/v1/chat/completions",
                         {"model": "m", "escalate": True,
                          "messages": [{"role": "user", "content": "confident question"}]})
        self.assertEqual(resp["x_proxy"], {"escalated": False, "mean_logprob": -0.01})
        self.assertEqual(resp["choices"][0]["message"]["content"], "no-think answer")

    def test_streaming_passthrough_untouched(self):
        req = urllib.request.Request(self.url + "/v1/chat/completions",
                                     data=json.dumps({"model": "m", "stream": True,
                                                      "escalate": True,
                                                      "messages": []}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.headers["Content-Type"], "text/event-stream")
            self.assertEqual(resp.read().decode(), "".join(SSE_CHUNKS))

    def test_streaming_loop_guard_cuts(self):
        req = urllib.request.Request(self.url + "/v1/chat/completions",
                                     data=json.dumps({"model": "m", "stream": True,
                                                      "messages": [{"role": "user",
                                                                    "content": "loop me"}]}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode()
        events = [e for e in body.split("\n\n") if e.startswith("data: ")]
        # guard cut early: far fewer loop chunks than the upstream's 60
        self.assertLess(len(events), 10)
        cut = json.loads(events[-2][len("data: "):])
        self.assertEqual(cut["choices"][0]["finish_reason"], "stop")
        self.assertEqual(cut["choices"][0]["delta"], {})
        self.assertEqual(cut["x_proxy"], {"loop_guard": "cut"})
        self.assertEqual(events[-1], "data: [DONE]")


class ProxyTestBase(unittest.TestCase):
    @classmethod
    def make_stack(cls, **proxy_kwargs):
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
        cls.upstream.requests = []
        if proxy_kwargs.get("router_url") == "upstream":
            proxy_kwargs["router_url"] = \
                f"http://127.0.0.1:{cls.upstream.server_port}"
        cls.proxy = score_proxy.make_server(
            0, f"http://127.0.0.1:{cls.upstream.server_port}", **proxy_kwargs)
        for srv in (cls.upstream, cls.proxy):
            threading.Thread(target=srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.proxy.server_port}"

    @classmethod
    def stop_stack(cls):
        cls.proxy.shutdown()
        cls.upstream.shutdown()
        cls.proxy.server_close()
        cls.upstream.server_close()

    def setUp(self):
        self.upstream.requests.clear()

    def post_chat(self, obj, url=None):
        body = json.dumps(obj).encode()
        req = urllib.request.Request((url or self.url) + "/v1/chat/completions",
                                     data=body,
                                     headers={"Content-Type": "application/json"})
        return body, json.load(urllib.request.urlopen(req))


class CanonicalizeTest(ProxyTestBase):
    @classmethod
    def setUpClass(cls):
        cls.make_stack(canonicalize=True)
        cls.proxy_strip = score_proxy.make_server(
            0, f"http://127.0.0.1:{cls.upstream.server_port}",
            canonicalize=True, strip_jitter=True)
        threading.Thread(target=cls.proxy_strip.serve_forever, daemon=True).start()
        cls.url_strip = f"http://127.0.0.1:{cls.proxy_strip.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.proxy_strip.shutdown()
        cls.proxy_strip.server_close()
        cls.stop_stack()

    def upstream_messages(self):
        path, raw = self.upstream.requests[-1]
        self.assertEqual(path, "/v1/chat/completions")
        return json.loads(raw)["messages"]

    def test_reorders_system_before_first_user(self):
        self.post_chat({"model": "m", "messages": [
            {"role": "system", "content": "A"},
            {"role": "assistant", "content": "primer"},
            {"role": "system", "content": "B"},
            {"role": "user", "content": "U"}]})
        got = self.upstream_messages()
        self.assertEqual([m["role"] for m in got],
                         ["system", "system", "assistant", "user"])
        self.assertEqual([m["content"] for m in got], ["A", "B", "primer", "U"])

    def test_wellformed_request_byte_identical(self):
        body, _ = self.post_chat({"model": "m", "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"}]})
        self.assertEqual(self.upstream.requests[-1][1], body)

    def test_jitter_strip_off_by_default(self):
        jittered = "You are helpful.\nCurrent time: 2026-10-09 12:00:00"
        self.post_chat({"model": "m", "messages": [
            {"role": "system", "content": jittered},
            {"role": "user", "content": "U"}]})
        self.assertEqual(self.upstream_messages()[0]["content"], jittered)

    def test_jitter_strip_opt_in(self):
        self.post_chat({"model": "m", "messages": [
            {"role": "system",
             "content": "You are helpful.\nDate: 2026-10-09\nCurrent time: 12:00"},
            {"role": "user", "content": "U"}]}, url=self.url_strip)
        self.assertEqual(self.upstream_messages()[0]["content"],
                         "You are helpful.")


class PromptLayoutHttpTest(unittest.TestCase):
    """Proxy-injected context trails the conversation: a changed injection
    must never move the cached prompt head."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        roots = [os.path.join(cls.tmp.name, n) for n in ("base", "extra")]
        for root, extra in zip(roots, (False, True)):
            cls.write_skill(root, "alpha", "Ship services safely.")
            cls.write_skill(root, "beta", "Trace failures to a root cause.")
            if extra:
                # fake_embed sends this one far from the query; the other
                # two stay in the selected set.
                cls.write_skill(root, "design", "Paint watercolor landscapes.")
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
        cls.upstream.requests = []
        threading.Thread(target=cls.upstream.serve_forever, daemon=True).start()
        upstream_url = f"http://127.0.0.1:{cls.upstream.server_port}"
        cls.proxies = []
        for root in roots:
            proxy = score_proxy.make_server(
                0, upstream_url, router_url=upstream_url, skills_dir=root)
            threading.Thread(target=proxy.serve_forever, daemon=True).start()
            cls.proxies.append(proxy)

    @staticmethod
    def write_skill(root, name, description):
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, name + ".md"), "w", encoding="utf-8") as f:
            f.write(f"---\ndescription: {description}\n---\n")

    @classmethod
    def tearDownClass(cls):
        for proxy in cls.proxies:
            proxy.shutdown()
            proxy.server_close()
        cls.upstream.shutdown()
        cls.upstream.server_close()
        cls.tmp.cleanup()

    def test_changed_skills_keep_the_head_byte_identical(self):
        payload = {"model": "m", "route": False, "messages": [
            {"role": "system", "content": "policy: be brief"},
            {"role": "assistant", "content": "ready"},
            {"role": "user", "content": "how do I fix the deploy pipeline?"},
        ]}
        self.upstream.requests.clear()
        for proxy in self.proxies:
            req = urllib.request.Request(
                f"http://127.0.0.1:{proxy.server_port}/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req).read()
        bodies = [json.loads(raw)["messages"]
                  for path, raw in self.upstream.requests
                  if path == "/v1/chat/completions"]
        self.assertEqual(len(bodies), 2)
        head = len(payload["messages"])
        for messages in bodies:
            self.assertEqual(messages[:head], payload["messages"])
            self.assertEqual(messages[-1]["role"], "user")
            # the chat template refuses a system message that is not first
            self.assertEqual([i for i, m in enumerate(messages)
                              if m["role"] == "system"], [0])
            self.assertIn(sp.SKILLS_MARKER, messages[-1]["content"])
        self.assertEqual(
            json.dumps(bodies[0][:head]).encode(),
            json.dumps(bodies[1][:head]).encode())
        self.assertIn("alpha", bodies[0][-1]["content"])
        self.assertIn("beta", bodies[0][-1]["content"])
        self.assertNotIn("design", bodies[1][-1]["content"])


class KeepaliveTest(ProxyTestBase):
    INTERVAL = 0.3

    @classmethod
    def setUpClass(cls):
        cls.make_stack(keepalive_s=cls.INTERVAL)

    @classmethod
    def tearDownClass(cls):
        cls.stop_stack()

    def pings(self):
        return [json.loads(raw) for path, raw in self.upstream.requests
                if path == "/v1/chat/completions"
                and json.loads(raw).get("max_tokens") == 1]

    def wait_for_ping(self, timeout=3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.pings():
                return True
            time.sleep(0.02)
        return False

    def test_keepalive_fires_when_idle(self):
        msgs = [{"role": "system", "content": "S"},
                {"role": "user", "content": "U"}]
        self.post_chat({"model": "m", "messages": msgs})
        self.assertTrue(self.wait_for_ping())
        ping = self.pings()[0]
        self.assertEqual(ping["model"], "m")
        self.assertEqual(ping["messages"], msgs)

    def test_keepalive_quiet_during_traffic(self):
        end = time.time() + self.INTERVAL * 3
        while time.time() < end:
            self.post_chat({"model": "m",
                            "messages": [{"role": "user", "content": "still here"}]})
            time.sleep(self.INTERVAL / 3)
        self.assertEqual(self.pings(), [])


class RouterTest(ProxyTestBase):
    @classmethod
    def setUpClass(cls):
        cls.make_stack(router_url="upstream", router_model="fake-embed",
                       router_threshold=0.15, router_max_tokens=77)

    @classmethod
    def tearDownClass(cls):
        cls.stop_stack()

    def upstream_body(self):
        path, raw = self.upstream.requests[-1]
        self.assertEqual(path, "/v1/chat/completions")
        return raw

    def test_trivial_routed_no_thinking_and_capped(self):
        _, resp = self.post_chat({"model": "m", "max_tokens": 5000,
                                  "messages": [{"role": "user",
                                                "content": "a simple thing"}]})
        req = json.loads(self.upstream_body())
        self.assertIs(req["chat_template_kwargs"]["enable_thinking"], False)
        self.assertEqual(req["max_tokens"], 77)
        self.assertEqual(resp["x_proxy"]["route"], "trivial")
        self.assertEqual(resp["choices"][0]["message"]["content"],
                         "no-think answer")

    def test_trivial_without_max_tokens_gets_cap(self):
        self.post_chat({"model": "m",
                        "messages": [{"role": "user", "content": "hi there"}]})
        self.assertEqual(json.loads(self.upstream_body())["max_tokens"], 77)

    def test_hard_passthrough_byte_identical(self):
        body, _ = self.post_chat({"model": "m", "messages": [
            {"role": "user", "content": "prove this lemma"}]})
        self.assertEqual(self.upstream_body(), body)

    def test_below_margin_passthrough_byte_identical(self):
        body, _ = self.post_chat({"model": "m", "messages": [
            {"role": "user", "content": "an ambiguous request"}]})
        self.assertEqual(self.upstream_body(), body)

    def test_route_false_escape_hatch(self):
        body, _ = self.post_chat({"model": "m", "route": False, "messages": [
            {"role": "user", "content": "a simple thing"}]})
        self.assertEqual(self.upstream_body(), body)

    def test_router_precedes_escalation(self):
        _, resp = self.post_chat({"model": "m", "escalate": True,
                                  "messages": [{"role": "user",
                                                "content": "a simple thing"}]})
        self.assertEqual(resp["x_proxy"]["route"], "trivial")
        self.assertNotIn("escalated", resp["x_proxy"])
        _, resp = self.post_chat({"model": "m", "escalate": True,
                                  "messages": [{"role": "user",
                                                "content": "prove this"}]})
        self.assertIn("escalated", resp["x_proxy"])


class RouterDownTest(ProxyTestBase):
    @classmethod
    def setUpClass(cls):
        cls.make_stack(router_url="http://127.0.0.1:1",  # nothing listening
                       router_timeout=0.5)

    @classmethod
    def tearDownClass(cls):
        cls.stop_stack()

    def test_embeddings_down_fails_open(self):
        body, resp = self.post_chat({"model": "m", "messages": [
            {"role": "user", "content": "a simple thing"}]})
        self.assertEqual(self.upstream.requests[-1][1], body)
        self.assertEqual(resp["choices"][0]["message"]["content"],
                         "thinking answer")


class ToolCallRepairTest(unittest.TestCase):
    def repair(self, content, reasoning=None, finish="stop"):
        message = {"role": "assistant", "content": content}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        resp = {"choices": [{"index": 0, "message": message,
                             "finish_reason": finish}]}
        req = {"model": "m", "messages": [{"role": "user", "content": "go"}],
               "tools": REPAIR_TOOLS}
        return resp, sp.repair_reply(resp, req)

    def assert_call(self, resp, expected):
        choice = resp["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        calls = choice["message"]["tool_calls"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["type"], "function")
        self.assertEqual(calls[0]["function"]["name"], "run")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), expected)

    def test_malformed_forms_normalize_to_tool_calls(self):
        cases = [
            ("fenced",
             '```json\n{"name": "run", "arguments": {"count": 5}}\n```',
             None, {"count": 5}),
            ("fence left open",
             '```json\n{"name": "run", "arguments": {"count": 8}',
             None, {"count": 8}),
            ("trailing commas",
             '{"name": "run", "arguments": {"count": 5,}}',
             None, {"count": 5}),
            ("single-quoted keys",
             "{'name': 'run', 'arguments': {'verbose': 'true'}}",
             None, {"verbose": True}),
            ("tool_call wrapper",
             '{"tool_call": {"name": "run", "arguments": {"count": 6}}}',
             None, {"count": 6}),
            ("wrapper without arguments",
             '{"tool_call": {"name": "run"}}',
             None, {}),
            ("arguments as JSON string",
             '{"function_call": {"name": "run", '
             '"arguments": "{\\"ratio\\": \\"3.5\\"}"}}',
             None, {"ratio": 3.5}),
            ("function as a name",
             '{"function": "run", "arguments": {"count": 7}}',
             None, {"count": 7}),
            ("unclosed JSON tail",
             '{"name": "run", "arguments": {"count": 5, "query": ',
             None, {"count": 5}),
            ("unclosed string",
             '{"name": "run", "arguments": {"query": "half a sent',
             None, {"query": "half a sent"}),
            ("unclosed array",
             '{"name": "run", "arguments": {"tags": ["a", "b"',
             None, {"tags": ["a", "b"]}),
            ("xml tool_call",
             "<tool_call>\n<function=run>\n<parameter=count>\n5 ms\n"
             "</parameter>\n</function>\n</tool_call>",
             None, {"count": 5}),
            ("xml wrapper left open",
             "<tool_call>\n<function=run>\n<parameter=tags>\nalpha\n"
             "</parameter>",
             None, {"tags": ["alpha"]}),
            ("invoke wrapper",
             '<invoke name="run"><parameter name="count">7</parameter></invoke>',
             None, {"count": 7}),
            ("call inside reasoning",
             "",
             '<tool_call><function=run><parameter=opts>{"n": "2"}'
             "</parameter></function></tool_call>",
             {"opts": {"n": 2}}),
        ]
        for name, content, reasoning, expected in cases:
            with self.subTest(name):
                resp, (changed, retry) = self.repair(content, reasoning)
                self.assertTrue(changed)
                self.assertIsNone(retry)
                self.assert_call(resp, expected)

    def test_unclosed_tail_at_the_token_cap_is_repaired(self):
        resp, (changed, retry) = self.repair(
            '{"name": "run", "arguments": {"count": 9',
            finish="length")
        self.assertTrue(changed)
        self.assertIsNone(retry)
        self.assert_call(resp, {"count": 9})

    def test_quoted_call_inside_an_answer_is_not_repaired(self):
        resp, (changed, retry) = self.repair(
            "The harness sends "
            '{"name": "run", "arguments": {"count": 5}} '
            "and then waits for the result.")
        self.assertFalse(changed)
        self.assertIsNone(retry)
        self.assertFalse(resp["choices"][0]["message"].get("tool_calls"))

    def test_reasoning_call_ignored_once_an_answer_exists(self):
        resp, (changed, retry) = self.repair(
            "the answer",
            '<tool_call><function=run><parameter=count>5</parameter>'
            "</function></tool_call>")
        self.assertFalse(changed)
        self.assertIsNone(retry)
        self.assertNotIn("tool_calls", resp["choices"][0]["message"])

    def test_undeclared_tool_name_is_not_emitted(self):
        resp, (changed, retry) = self.repair(
            '{"name": "nope", "arguments": {"count": 5}}')
        self.assertFalse(changed)
        self.assertIsNone(retry)
        self.assertNotIn("tool_calls", resp["choices"][0]["message"])

    def test_wellformed_reply_is_not_touched(self):
        resp = copy.deepcopy(WELLFORMED_CALL)
        changed, retry = sp.repair_reply(
            resp, {"model": "m", "messages": [], "tools": REPAIR_TOOLS})
        self.assertFalse(changed)
        self.assertIsNone(retry)
        self.assertEqual(resp, WELLFORMED_CALL)


class ThinkingRepairTest(unittest.TestCase):
    def reply(self, content="", reasoning=None, finish="length", usage=None):
        message = {"role": "assistant", "content": content}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        resp = {"choices": [{"index": 0, "message": message,
                             "finish_reason": finish}]}
        if usage is not None:
            resp["usage"] = usage
        return resp

    def request(self, max_tokens=8192):
        req = {"model": "m", "messages": [{"role": "user", "content": "q"}]}
        if max_tokens is not None:
            req["max_tokens"] = max_tokens
        return req

    def test_stops_inside_reasoning_schedules_one_retry(self):
        resp = self.reply(reasoning="still thinking",
                          usage={"completion_tokens": 2048})
        changed, retry = sp.repair_reply(resp, self.request())
        self.assertFalse(changed)
        self.assertIsNotNone(retry)
        body = retry["request"]
        self.assertTrue(body["continue_final_message"])
        self.assertFalse(body["add_generation_prompt"])
        self.assertEqual(body["messages"][-1]["reasoning_content"],
                         "still thinking")
        self.assertEqual(body["max_tokens"], sp.THINK_RETRY_MAX)
        self.assertGreaterEqual(body["max_tokens"], sp.ANSWER_RESERVE_MIN)

    def test_wellformed_reasoning_and_answer_untouched(self):
        resp = self.reply(content="the answer", reasoning="complete thought",
                          finish="stop")
        before = copy.deepcopy(resp)
        changed, retry = sp.repair_reply(resp, self.request())
        self.assertFalse(changed)
        self.assertIsNone(retry)
        self.assertEqual(resp, before)

    def test_answer_in_content_blocks_retry(self):
        resp = self.reply(content="the answer", reasoning="thinking",
                          finish="stop")
        changed, retry = sp.repair_reply(resp, self.request())
        self.assertFalse(changed)
        self.assertIsNone(retry)

    def test_closed_thinking_blocks_retry(self):
        resp = self.reply(content="<think>done</think>", finish="stop")
        _, retry = sp.repair_reply(resp, self.request())
        self.assertIsNone(retry)

    def test_finish_reason_must_be_length_or_stop(self):
        resp = self.reply(reasoning="thinking", finish="tool_calls")
        _, retry = sp.repair_reply(resp, self.request())
        self.assertIsNone(retry)

    def test_allow_retry_false_never_schedules(self):
        resp = self.reply(reasoning="thinking")
        _, retry = sp.repair_reply(resp, self.request(), allow_retry=False)
        self.assertIsNone(retry)

    def test_retry_budget_reserves_answer_room(self):
        cases = [
            (None, sp.THINK_RETRY_MAX),
            (512, 512),
            (1024, 512),
            (4096, sp.THINK_RETRY_MAX),
            (100000, sp.THINK_RETRY_MAX),
        ]
        for cap, expected in cases:
            with self.subTest(cap):
                self.assertEqual(sp.answer_room(cap), expected)
                self.assertLessEqual(sp.answer_room(cap), sp.THINK_RETRY_MAX)

    def test_retry_budget_clamped_to_remaining_context(self):
        resp = self.reply(reasoning="thinking",
                          usage={"completion_tokens": 10})
        req = {"model": "m", "max_tokens": 8192,
               "messages": [{"role": "user", "content": "x" * 4000}]}
        _, retry = sp.repair_reply(resp, req, context_tokens=1200)
        budget = retry["request"]["max_tokens"]
        self.assertGreaterEqual(budget, 1)
        self.assertLess(budget, sp.THINK_RETRY_MAX)

    def test_merge_closes_content_thinking_and_appends_answer(self):
        resp = self.reply(content="<think>partial", finish="stop")
        _, retry = sp.repair_reply(resp, self.request())
        retried = {"choices": [{"index": 0,
                                "message": {"role": "assistant",
                                            "content": "42"},
                                "finish_reason": "stop"}],
                   "usage": {"completion_tokens": 1, "total_tokens": 1}}
        self.assertTrue(sp.apply_thinking_retry(resp, retried, retry))
        message = resp["choices"][0]["message"]
        self.assertEqual(message["content"],
                         "<think>partial\n\n</think>\n\n42")
        self.assertEqual(resp["choices"][0]["finish_reason"], "stop")
        self.assertEqual(resp["x_proxy"], {"thinking_retry": "closed"})

    def test_merge_keeps_reasoning_channel_and_uses_answer(self):
        resp = self.reply(reasoning="partial thought")
        _, retry = sp.repair_reply(resp, self.request())
        retried = {"choices": [{"index": 0,
                                "message": {"role": "assistant",
                                            "content": "the answer"},
                                "finish_reason": "stop"}]}
        self.assertTrue(sp.apply_thinking_retry(resp, retried, retry))
        message = resp["choices"][0]["message"]
        self.assertEqual(message["content"], "the answer")
        self.assertEqual(message["reasoning_content"], "partial thought")

    def test_retry_merges_into_the_truncated_choice(self):
        resp = {"choices": [
            {"index": 0, "message": {"role": "assistant",
                                     "content": "first answer"},
             "finish_reason": "stop"},
            {"index": 1, "message": {"role": "assistant", "content": "",
                                     "reasoning_content": "partial"},
             "finish_reason": "length"}]}
        changed, retry = sp.repair_reply(resp, self.request())
        self.assertFalse(changed)
        self.assertEqual(retry["index"], 1)
        retried = {"choices": [{"index": 0,
                                "message": {"role": "assistant",
                                            "content": "second answer"},
                                "finish_reason": "stop"}]}
        self.assertTrue(sp.apply_thinking_retry(resp, retried, retry))
        self.assertEqual(resp["choices"][0]["message"]["content"],
                         "first answer")
        self.assertEqual(resp["choices"][1]["message"]["content"],
                         "second answer")

    def test_empty_continuation_leaves_original_alone(self):
        resp = self.reply(reasoning="partial thought")
        before = copy.deepcopy(resp)
        _, retry = sp.repair_reply(resp, self.request())
        retried = {"choices": [{"index": 0,
                                "message": {"role": "assistant",
                                            "content": ""},
                                "finish_reason": "length"}]}
        self.assertFalse(sp.apply_thinking_retry(resp, retried, retry))
        self.assertEqual(resp, before)


class MaxTokensClampTest(unittest.TestCase):
    def test_clamps_only_when_the_explicit_cap_exceeds_room(self):
        req = {"max_tokens": 5000,
               "messages": [{"role": "user", "content": "x" * 400}]}
        self.assertTrue(sp.clamp_max_tokens(req, 1024))
        self.assertLess(req["max_tokens"], 5000)
        self.assertGreaterEqual(req["max_tokens"], 1)
        fits = {"max_tokens": 10,
                "messages": [{"role": "user", "content": "x" * 400}]}
        self.assertFalse(sp.clamp_max_tokens(fits, 1024))
        self.assertEqual(fits["max_tokens"], 10)

    def test_disabled_without_context_or_cap(self):
        req = {"max_tokens": 5000,
               "messages": [{"role": "user", "content": "x" * 400}]}
        self.assertFalse(sp.clamp_max_tokens(req, 0))
        no_cap = {"messages": [{"role": "user", "content": "x" * 400}]}
        self.assertFalse(sp.clamp_max_tokens(no_cap, 1024))
        self.assertNotIn("max_tokens", no_cap)

    def test_tools_count_toward_the_prompt(self):
        messages = [{"role": "user", "content": "x" * 200}]
        without = {"max_tokens": 5000, "messages": messages}
        with_tools = {"max_tokens": 5000, "messages": messages,
                      "tools": REPAIR_TOOLS}
        sp.clamp_max_tokens(without, 1024)
        sp.clamp_max_tokens(with_tools, 1024)
        self.assertLess(with_tools["max_tokens"], without["max_tokens"])

    def test_fits_already_keeps_the_request_byte_identical(self):
        cases = [
            ("cap inside the room",
             {"max_tokens": 10,
              "messages": [{"role": "user", "content": "x" * 400}]}, 1024),
            ("no context configured",
             {"max_tokens": 5000,
              "messages": [{"role": "user", "content": "x" * 400}]}, 0),
            ("no cap to lower",
             {"messages": [{"role": "user", "content": "x" * 400}]}, 1024),
        ]
        for name, req, context in cases:
            with self.subTest(name):
                before = json.dumps(req)
                self.assertFalse(sp.clamp_max_tokens(req, context))
                self.assertEqual(json.dumps(req), before)

    def test_prompt_alone_over_the_context_is_left_alone(self):
        req = {"max_tokens": 5000,
               "messages": [{"role": "user", "content": "x" * 4000}]}
        before = json.dumps(req)
        self.assertFalse(sp.clamp_max_tokens(req, 100))
        self.assertEqual(json.dumps(req), before)


class RepairHttpTest(ProxyTestBase):
    @classmethod
    def setUpClass(cls):
        cls.make_stack()

    @classmethod
    def tearDownClass(cls):
        cls.stop_stack()

    def test_client_headers_forwarded_on_buffered_path(self):
        body = json.dumps({
            "model": "m",
            "messages": [{"role": "user", "content": "plain answer"}],
        }).encode()
        req = urllib.request.Request(
            self.url + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer test-token"})
        urllib.request.urlopen(req).read()
        self.assertEqual(self.upstream.last_headers.get("Authorization"),
                         "Bearer test-token")

    def test_malformed_call_repaired_end_to_end(self):
        _, resp = self.post_chat({
            "model": "m", "tools": REPAIR_TOOLS,
            "messages": [{"role": "user", "content": "malformed call"}]})
        choice = resp["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        call = choice["message"]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "run")
        self.assertEqual(json.loads(call["function"]["arguments"]),
                         {"count": 5, "verbose": True})

    def test_wellformed_tool_call_bytes_untouched(self):
        body = json.dumps({
            "model": "m", "tools": REPAIR_TOOLS,
            "messages": [{"role": "user", "content": "wellformed call"}],
        }).encode()
        req = urllib.request.Request(
            self.url + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            received = resp.read()
        self.assertEqual(received, json.dumps(WELLFORMED_CALL).encode())

    def test_wellformed_answer_without_tools_bytes_untouched(self):
        body = json.dumps({
            "model": "m",
            "messages": [{"role": "user", "content": "plain answer"}],
        }).encode()
        req = urllib.request.Request(
            self.url + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            received = resp.read()
        expected = {"choices": [{"index": 0,
                                 "message": {"role": "assistant",
                                             "content": "plain answer"},
                                 "finish_reason": "stop"}]}
        self.assertEqual(received, json.dumps(expected).encode())

    def test_truncated_thinking_retried_once(self):
        _, resp = self.post_chat({
            "model": "m", "max_tokens": 8192,
            "messages": [{"role": "user", "content": "truncated think"}]})
        choice = resp["choices"][0]
        self.assertEqual(choice["message"]["content"], "recovered answer")
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(resp["x_proxy"], {"thinking_retry": "closed"})
        sent = [json.loads(raw) for _, raw in self.upstream.requests]
        self.assertEqual(len(sent), 2)
        self.assertTrue(sent[1]["continue_final_message"])
        self.assertEqual(sent[1]["messages"][-1]["reasoning_content"],
                         "still thinking")
        self.assertLessEqual(sent[1]["max_tokens"], sp.THINK_RETRY_MAX)
        self.assertGreaterEqual(sent[1]["max_tokens"], 1)

    def test_failed_continuation_not_retried_twice(self):
        _, resp = self.post_chat({
            "model": "m",
            "messages": [{"role": "user", "content": "forever think"}]})
        self.assertEqual(len(self.upstream.requests), 2)
        self.assertEqual(resp["choices"][0]["message"]["content"], "")
        self.assertEqual(resp["choices"][0]["finish_reason"], "length")


class MaxTokensClampHttpTest(ProxyTestBase):
    @classmethod
    def setUpClass(cls):
        cls.make_stack(context_tokens=600)

    @classmethod
    def tearDownClass(cls):
        cls.stop_stack()

    def test_max_tokens_clamped_before_the_request(self):
        self.post_chat({
            "model": "m", "max_tokens": 100000,
            "messages": [{"role": "user", "content": "x" * 400}]})
        sent = json.loads(self.upstream.requests[-1][1])
        self.assertLess(sent["max_tokens"], 100000)
        self.assertGreaterEqual(sent["max_tokens"], 1)

    def test_fits_already_keeps_the_body_byte_identical(self):
        body, _ = self.post_chat({
            "model": "m", "max_tokens": 10,
            "messages": [{"role": "user", "content": "x" * 400}]})
        self.assertEqual(self.upstream.requests[-1][1], body)

    def test_prompt_alone_over_the_context_is_passed_through(self):
        body, _ = self.post_chat({
            "model": "m", "max_tokens": 5000,
            "messages": [{"role": "user", "content": "x" * 4000}]})
        self.assertEqual(self.upstream.requests[-1][1], body)


class TokenAccountingTest(ProxyTestBase):
    @classmethod
    def setUpClass(cls):
        cls.make_stack()

    @classmethod
    def tearDownClass(cls):
        cls.stop_stack()

    def accounting_lines(self, payload):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            req = urllib.request.Request(
                self.url + "/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req).read()
        return [line for line in buf.getvalue().splitlines()
                if line.startswith("score proxy: usage")]

    def test_one_line_per_completed_request(self):
        cases = [
            ("non-streaming",
             {"model": "m",
              "messages": [{"role": "user", "content": "usage report"}]},
             "score proxy: usage prompt_tokens=1234 cached_tokens=1000 "
             "completion_tokens=56 reasoning_tokens=7"),
            ("streaming usage chunk",
             {"model": "m", "stream": True,
              "messages": [{"role": "user", "content": "usage stream"}]},
             "score proxy: usage prompt_tokens=321 cached_tokens=300 "
             "completion_tokens=9 reasoning_tokens=2"),
            ("no usage reported",
             {"model": "m",
              "messages": [{"role": "user", "content": "plain answer"}]},
             "score proxy: usage unavailable"),
        ]
        for name, payload, expected in cases:
            with self.subTest(name):
                self.assertEqual(self.accounting_lines(payload), [expected])


def tool_message(content, call_id="call_1"):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def call_message(call_id, name="read_file", arguments='{"path": "a.log"}'):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name,
                                         "arguments": arguments}}]}


class ToolOutputDedupeTest(unittest.TestCase):
    LONG = "tool output line\n" * 60

    def messages(self):
        return [
            {"role": "user", "content": "read it, then read it again"},
            call_message("call_1"), tool_message(self.LONG, "call_1"),
            call_message("call_2"), tool_message(self.LONG, "call_2"),
            # same bytes, different path: a different call, so it must survive
            call_message("call_3", arguments='{"path": "b.log"}'),
            tool_message(self.LONG, "call_3"),
        ]

    def test_collapses_a_repeated_identical_call(self):
        req = {"messages": self.messages()}
        size = len(self.LONG.encode())
        self.assertEqual(sp.collapse_duplicate_tool_output(req), 1)
        messages = req["messages"]
        self.assertEqual(messages[2]["content"], self.LONG)   # first result
        self.assertEqual(
            messages[4]["content"],
            f'[identical to the read_file({{"path": "a.log"}}) result at '
            f'message 3; {size} bytes]')
        self.assertEqual(messages[6]["content"], self.LONG)

    def test_updated_file_is_not_collapsed(self):
        req = {"messages": [
            call_message("call_1"), tool_message(self.LONG, "call_1"),
            call_message("call_2"),
            tool_message(self.LONG + "a new line\n", "call_2")]}
        self.assertEqual(sp.collapse_duplicate_tool_output(req), 0)

    def test_unresolvable_call_is_not_collapsed(self):
        req = {"messages": [tool_message(self.LONG, "ghost_1"),
                            tool_message(self.LONG, "ghost_2")]}
        before = copy.deepcopy(req)
        self.assertEqual(sp.collapse_duplicate_tool_output(req), 0)
        self.assertEqual(req, before)

    def test_short_and_non_string_content_untouched(self):
        small = "y" * (sp.DEDUPE_MIN_BYTES - 1)
        blocks = [{"type": "text", "text": "z" * 1024}]
        req = {"messages": [
            call_message("c1"), tool_message(small, "c1"),
            call_message("c2"), tool_message(small, "c2"),
            call_message("c3"), tool_message(blocks, "c3"),
            call_message("c4"), tool_message(blocks, "c4")]}
        before = copy.deepcopy(req)
        self.assertEqual(sp.collapse_duplicate_tool_output(req), 0)
        self.assertEqual(req, before)

    def test_second_pass_is_a_no_op(self):
        req = {"messages": self.messages()}
        self.assertEqual(sp.collapse_duplicate_tool_output(req), 1)
        self.assertEqual(sp.collapse_duplicate_tool_output(req), 0)


class ToolOutputDedupeHttpTest(ProxyTestBase):
    @classmethod
    def setUpClass(cls):
        cls.make_stack(dedupe_tool_output=True)
        cls.proxy_off = score_proxy.make_server(
            0, f"http://127.0.0.1:{cls.upstream.server_port}")
        threading.Thread(target=cls.proxy_off.serve_forever,
                         daemon=True).start()
        cls.url_off = f"http://127.0.0.1:{cls.proxy_off.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.proxy_off.shutdown()
        cls.proxy_off.server_close()
        cls.stop_stack()

    def payload(self):
        big = "tool output line\n" * 60
        return {"model": "m", "messages": [
            {"role": "user", "content": "plain answer"},
            call_message("call_1"), tool_message(big, "call_1"),
            call_message("call_2"), tool_message(big, "call_2")]}

    def test_gate_off_is_byte_identical(self):
        body, _ = self.post_chat(self.payload(), url=self.url_off)
        self.assertEqual(self.upstream.requests[-1][1], body)

    def test_gate_on_collapses_the_later_copy(self):
        payload = self.payload()
        big = payload["messages"][2]["content"]
        body, _ = self.post_chat(payload)
        sent = json.loads(self.upstream.requests[-1][1])
        self.assertNotEqual(self.upstream.requests[-1][1], body)
        self.assertEqual(sent["messages"][2]["content"], big)
        self.assertEqual(
            sent["messages"][4]["content"],
            f'[identical to the read_file({{"path": "a.log"}}) result at '
            f'message 3; {len(big.encode())} bytes]')


if __name__ == "__main__":
    unittest.main()
