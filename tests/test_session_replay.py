"""Tests for bench/session_replay.py extraction.

The extractor used to deep-copy the growing history at every assistant turn
(`json.loads(json.dumps(history))`), which is quadratic: a 22 MB transcript ran
for minutes without producing anything. Both the cost and the aliasing contract
that makes the cheap copy safe are pinned here.
"""

import importlib.util
import http.server
import json
import os
import pathlib
import tempfile
import threading
import time
import unittest

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT = HERE.parent / "bench" / "session_replay.py"

spec = importlib.util.spec_from_file_location("session_replay", SCRIPT)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def rec(role, text, cid=None):
    body = {"role": role, "content": [{"type": "text", "text": text}]}
    if cid:
        body["toolCallId"] = cid
    return {"type": "message", "message": body}


class ExtractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for line in (rec("user", "task"),
                     rec("assistant", "first"),
                     rec("toolResult", "output", cid="c1"),
                     rec("assistant", "second")):
            self.tmp.write(json.dumps(line) + "\n")
        self.tmp.close()
        self.turns = replay.extract(self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_one_request_per_assistant_turn(self):
        # A request is the history *before* each assistant turn, so two
        # assistant records give two windows: [user] and [user, assistant, tool].
        self.assertEqual(len(self.turns), 2)
        self.assertEqual([m["role"] for m in self.turns[0]["messages"]], ["user"])

    def test_tool_result_becomes_tool_message(self):
        msgs = self.turns[1]["messages"]
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "tool"])
        self.assertEqual(msgs[2]["tool_call_id"], "c1")
        self.assertEqual(msgs[2]["content"], "output")

    def test_turns_do_not_alias_each_other(self):
        # The cheap snapshot is a shallow list copy, so the per-turn lists must
        # stay independent even though they share the message dicts.
        first = replay.extract(self.tmp.name)[1]["messages"]
        first.append({"role": "user", "content": "injected"})
        again = replay.extract(self.tmp.name)[1]["messages"]
        self.assertEqual(len(again), 3)

    def test_no_quadratic_serialisation(self):
        src = SCRIPT.read_text()
        self.assertNotIn("json.loads(json.dumps(history))", src)


class SampleTest(unittest.TestCase):
    def test_keeps_endpoints_and_count(self):
        reqs = list(range(100))
        picked = replay.sample_evenly(reqs, 24)
        self.assertEqual(len(picked), 24)
        self.assertEqual(picked[0], 0)
        self.assertEqual(picked[-1], 99)
        self.assertEqual(picked, sorted(set(picked)))

    def test_no_sample_is_identity(self):
        reqs = list(range(5))
        self.assertEqual(replay.sample_evenly(reqs, 0), reqs)
        self.assertEqual(replay.sample_evenly(reqs, 9), reqs)

    def test_single_turn_takes_the_last(self):
        self.assertEqual(replay.sample_evenly(list(range(7)), 1), [6])


class SseTest(unittest.TestCase):
    def test_parses_payloads_and_ignores_noise(self):
        lines = [b": keepalive\n", b"\n", b'data: {"a": 1}\n',
                 b"data: [DONE]\n", b'data: {"b": 2}\n']
        self.assertEqual(list(replay._sse_rows(lines)), [{"a": 1}, {"b": 2}])

    def test_tool_call_fragments_join_by_index(self):
        acc = {}
        replay._merge_tool_call(acc, {"index": 0, "function": {"name": "read"}})
        replay._merge_tool_call(acc, {"index": 0, "function": {"arguments": '{"p"'}})
        replay._merge_tool_call(acc, {"index": 0, "function": {"arguments": ': 1}'}})
        self.assertEqual(acc[0]["name"], "read")
        self.assertEqual(acc[0]["args"], '{"p": 1}')


class StreamRowTest(unittest.TestCase):
    """The split has to come from the wire, not from arithmetic on totals."""

    @classmethod
    def setUpClass(cls):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                assert self.headers["Content-Type"] == "application/json"
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for piece in ("hi ", "there"):
                    chunk = {"choices": [{"delta": {"content": piece}}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.05)
                tail = {"choices": [], "usage": {
                    "prompt_tokens": 20, "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 12}}}
                self.wfile.write(f"data: {json.dumps(tail)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            def log_message(self, *args):
                pass

        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_splits_prefill_from_decode(self):
        row = replay.stream_row(self.url, {"model": "m", "messages": []}, timeout=10)
        self.assertEqual(row["prompt"], 20)
        self.assertEqual(row["cached"], 12)
        self.assertEqual(row["completion"], 2)
        self.assertEqual(row["tool_calls"], [])
        self.assertEqual(row["reply_head"], "hi there")
        self.assertIsNotNone(row["ttft"])
        self.assertGreater(row["seconds"], row["ttft"])
        self.assertAlmostEqual(row["seconds"], row["ttft"] + row["decode_s"], places=1)

    def test_decodes_faster_than_the_whole_turn(self):
        row = replay.stream_row(self.url, {"model": "m", "messages": []}, timeout=10)
        self.assertGreater(row["decode_tok_s"], 0)


class SummariseTest(unittest.TestCase):
    def test_totals_and_cache_share(self):
        rows = [{"seconds": 10.0, "ttft": 4.0, "decode_s": 6.0, "prompt": 100,
                 "cached": 50, "completion": 20},
                {"seconds": 20.0, "ttft": 6.0, "decode_s": 14.0, "prompt": 100,
                 "cached": 70, "completion": 30}]
        s = replay.summarise(rows)
        self.assertEqual(s["wall"], 30.0)
        self.assertEqual(s["wall_mean"], 15.0)
        self.assertEqual(s["ttft_mean"], 5.0)
        self.assertEqual(s["decode_mean"], 10.0)
        self.assertEqual(s["prompt"], 200)
        self.assertEqual(s["cached"], 120)
        self.assertEqual(s["cached_share"], 0.6)
        self.assertEqual(s["decode_tok_s"], 2.5)

    def test_empty_rows_do_not_divide_by_zero(self):
        self.assertEqual(replay.summarise([])["turns"], 0)


if __name__ == "__main__":
    unittest.main()
