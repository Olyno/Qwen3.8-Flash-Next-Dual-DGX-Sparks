"""Tests for bench/session_replay.py extraction.

The extractor used to deep-copy the growing history at every assistant turn
(`json.loads(json.dumps(history))`), which is quadratic: a 22 MB transcript ran
for minutes without producing anything. Both the cost and the aliasing contract
that makes the cheap copy safe are pinned here.
"""

import importlib.util
import json
import os
import pathlib
import tempfile
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


if __name__ == "__main__":
    unittest.main()
