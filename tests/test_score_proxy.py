import json
import math
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "proxy"))
import score_proxy

BASE_LP = {"red": -0.2, "green": -2.0, "blue": -1.0}
POSITION_BONUS = 0.6  # fake position bias: the label listed first scores higher

SSE_CHUNKS = ('data: {"delta": "a"}\n\n', 'data: {"delta": "b"}\n\n', 'data: [DONE]\n\n')

LOOP_PIECE = "the same phrase again and again. "
LOOP_CHUNKS = tuple(
    f'data: {{"choices": [{{"index": 0, "delta": {{"content": {json.dumps(LOOP_PIECE)}}}}}], "finish_reason": null}}\n\n'
    for _ in range(60)) + ('data: [DONE]\n\n',)


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
        req = json.loads(self.rfile.read(length))
        if self.path == "/v1/completions":
            self._completions(req)
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
        chunks = (LOOP_CHUNKS if "loop me" in json.dumps(req["messages"])
                  else SSE_CHUNKS)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            try:
                self.wfile.write(chunk.encode())
                self.wfile.flush()
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


if __name__ == "__main__":
    unittest.main()
