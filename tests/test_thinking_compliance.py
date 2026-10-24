import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "bench" / "thinking_compliance.py"
_spec = importlib.util.spec_from_file_location("thinking_compliance", MODULE_PATH)
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)

PROMPTS = [
    {"id": "b1", "kind": "boolean",
     "question": "Is the sky blue? Answer yes or no."},
    {"id": "o1", "kind": "open-ended", "question": "Why is the sky blue?"},
]


class SplitTraceTest(unittest.TestCase):
    def test_last_separator_splits_trace_and_answer(self):
        self.assertEqual(tc.split_trace("step one\nstep two #### final answer"),
                         {"trace": "step one step two",
                          "answer": "final answer"})

    def test_multiple_separators_use_the_last(self):
        self.assertEqual(tc.split_trace("a #### b #### c"),
                         {"trace": "a #### b", "answer": "c"})

    def test_boxed_fallback_splits_before_last_box(self):
        self.assertEqual(tc.split_trace("because 2+2=4 \\boxed{4}"),
                         {"trace": "because 2+2=4", "answer": "\\boxed{4}"})

    def test_no_separator_means_answer_only(self):
        self.assertEqual(tc.split_trace("42"), {"trace": "", "answer": "42"})

    def test_leading_separator_has_empty_trace(self):
        self.assertEqual(tc.split_trace("#### answer"),
                         {"trace": "", "answer": "answer"})

    def test_whitespace_is_normalized(self):
        self.assertEqual(tc.split_trace("  one   two \n####\n  three  four "),
                         {"trace": "one two", "answer": "three four"})

    def test_empty_response(self):
        self.assertEqual(tc.split_trace(""), {"trace": "", "answer": ""})


class JudgeParseTest(unittest.TestCase):
    def test_bare_verdicts(self):
        self.assertEqual(tc.parse_judge("YES"), "YES")
        self.assertEqual(tc.parse_judge("no"), "NO")

    def test_verbose_replies(self):
        self.assertEqual(tc.parse_judge("The answer is YES."), "YES")
        self.assertEqual(tc.parse_judge("No, it does not."), "NO")
        self.assertEqual(
            tc.parse_judge("Based on the trace, yes, it contains deductions."),
            "YES")

    def test_first_verdict_token_wins(self):
        self.assertEqual(tc.parse_judge("NO. Actually YES."), "NO")

    def test_malformed_replies(self):
        self.assertIsNone(tc.parse_judge("I cannot tell."))
        self.assertIsNone(tc.parse_judge("YESNO"))
        self.assertIsNone(tc.parse_judge(""))
        self.assertIsNone(tc.parse_judge(None))


class AggregateTest(unittest.TestCase):
    def rows(self):
        return [
            {"empty_thinking": True, "trace_chars": 0, "relevance": None,
             "judge_verdict": None, "judge_raw": None},
            {"empty_thinking": False, "trace_chars": 20, "relevance": 0.5,
             "judge_verdict": "YES", "judge_raw": "YES"},
            {"empty_thinking": False, "trace_chars": 30, "relevance": 0.7,
             "judge_verdict": "NO", "judge_raw": "No."},
            {"empty_thinking": False, "trace_chars": 16, "relevance": None,
             "judge_verdict": None, "judge_raw": "maybe"},
        ]

    def test_counts_and_rates(self):
        st = tc.aggregate(self.rows(), judge_enabled=True,
                          relevance_enabled=True)
        self.assertEqual(st["n"], 4)
        self.assertEqual(st["empty_thinking"], 1)
        self.assertAlmostEqual(st["empty_thinking_rate"], 0.25)
        self.assertEqual(st["traces"], 3)
        self.assertEqual(st["relevance"], {"n": 2, "mean": 0.6})
        self.assertEqual(st["explicit_inference"],
                         {"judged": 2, "yes": 1, "yes_rate": 0.5,
                          "unparsed": 1})

    def test_disabled_metrics_are_none(self):
        st = tc.aggregate(self.rows(), judge_enabled=False,
                          relevance_enabled=False)
        self.assertIsNone(st["relevance"])
        self.assertIsNone(st["explicit_inference"])

    def test_empty_arm_has_no_rates(self):
        st = tc.aggregate([], judge_enabled=True, relevance_enabled=True)
        self.assertEqual(st["n"], 0)
        self.assertIsNone(st["empty_thinking_rate"])
        self.assertEqual(st["explicit_inference"]["judged"], 0)
        self.assertIsNone(st["explicit_inference"]["yes_rate"])
        self.assertIsNone(st["relevance"]["mean"])


class CorpusTest(unittest.TestCase):
    def test_default_corpus_shape(self):
        with open(tc.CORPUS) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({r["id"] for r in rows}), 12)
        self.assertEqual({r["kind"] for r in rows},
                         {"boolean", "multiple-choice", "open-ended"})
        for kind in ("boolean", "multiple-choice", "open-ended"):
            self.assertGreaterEqual(sum(r["kind"] == kind for r in rows), 3)
        for r in rows:
            self.assertTrue(r["question"].strip())
            if r["kind"] == "multiple-choice":
                self.assertIn("Candidates:", r["question"])


class FakeEndpoint:
    """Deterministic stand-in for post_json; records every payload."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, payload, timeout=None, retries=None,
                 retry_sleep=None):
        self.calls.append((url, payload))
        if "input" in payload:
            text = payload["input"][0]
            return {"data": [{"index": 0,
                              "embedding": [float(len(text)), 1.0, 2.0]}]}
        content = payload["messages"][-1]["content"]
        if tc.TRACE_START in content:
            verdict = "YES" if "because" in content else "NO"
            return {"choices": [{"message": {"content": verdict}}]}
        if payload.get("chat_template_kwargs", {}).get(
                "enable_thinking") is False:
            return {"choices": [{"message": {"content": "#### 42"}}]}
        return {"choices": [{"message": {
            "content": "The sky is blue because air scatters short "
                       "wavelengths. #### Blue."}}]}


def generation_payloads(fake):
    out = []
    for _, payload in fake.calls:
        if "messages" not in payload:
            continue
        if tc.TRACE_START in payload["messages"][-1]["content"]:
            continue
        out.append(payload)
    return out


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.corpus = Path(self.tmp.name) / "prompts.jsonl"
        self.corpus.write_text(
            "".join(json.dumps(row) + "\n" for row in PROMPTS))
        self.out = Path(self.tmp.name) / "report.json"
        self.fake = FakeEndpoint()
        real = tc.post_json
        tc.post_json = self.fake
        self.addCleanup(setattr, tc, "post_json", real)

    def run_main(self, extra):
        buf = io.StringIO()
        argv = ["--url", "http://endpoint:8888", "--model", "test-model",
                "--prompts", str(self.corpus), "--max-tokens", "32"] + extra
        with contextlib.redirect_stdout(buf):
            code = tc.main(argv)
        return code, buf.getvalue()

    def test_both_arms_payloads_and_report(self):
        code, out = self.run_main(
            ["--arm", "both", "--embed-url", "http://endpoint:8888",
             "--out", str(self.out)])
        self.assertEqual(code, 0)

        gens = generation_payloads(self.fake)
        self.assertEqual(len(gens), 4)  # two prompts, two arms
        nothink = [p for p in gens if "chat_template_kwargs" in p]
        thinking = [p for p in gens if "chat_template_kwargs" not in p]
        self.assertEqual(len(nothink), 2)
        self.assertEqual(len(thinking), 2)
        for p in nothink:
            self.assertEqual(p["chat_template_kwargs"],
                             {"enable_thinking": False})
        for p in thinking:
            self.assertNotIn("chat_template_kwargs", p)
        for p in gens:
            self.assertEqual(p["temperature"], 0)
            self.assertEqual(p["max_tokens"], 32)

        judged = [p for _, p in self.fake.calls
                  if "messages" in p
                  and tc.TRACE_START in p["messages"][-1]["content"]]
        self.assertEqual(len(judged), 2)  # only the thinking arm has traces
        for p in judged:
            self.assertEqual(p["chat_template_kwargs"],
                             {"enable_thinking": False})

        report = json.loads(self.out.read_text())
        self.assertEqual(set(report["arms"]), {"thinking", "nothink"})
        self.assertAlmostEqual(
            report["arms"]["nothink"]["empty_thinking_rate"], 1.0)
        self.assertAlmostEqual(
            report["arms"]["thinking"]["empty_thinking_rate"], 0.0)
        self.assertEqual(
            report["arms"]["nothink"]["explicit_inference"]["judged"], 0)
        self.assertEqual(
            report["arms"]["thinking"]["explicit_inference"]["yes"], 2)
        self.assertAlmostEqual(
            report["arms"]["thinking"]["explicit_inference"]["yes_rate"], 1.0)
        self.assertEqual(report["arms"]["thinking"]["relevance"]["n"], 2)
        self.assertEqual(report["arms"]["nothink"]["relevance"]["n"], 0)
        self.assertEqual(len(report["arms"]["thinking"]["rows"]), 2)
        self.assertIn("summary", out)
        self.assertIn("empty %", out)

    def test_empty_rate_gate_is_opt_in(self):
        code, _ = self.run_main(["--arm", "nothink", "--no-judge"])
        self.assertEqual(code, 0)

        code, out = self.run_main(
            ["--arm", "nothink", "--no-judge",
             "--fail-empty-below", "0.5"])
        self.assertEqual(code, 0)
        self.assertIn("gate arm=nothink", out)
        self.assertIn("PASS", out)

        code, out = self.run_main(
            ["--arm", "thinking", "--no-judge",
             "--fail-empty-below", "0.5"])
        self.assertEqual(code, 1)
        self.assertIn("gate arm=thinking", out)
        self.assertIn("FAIL", out)

    def test_endpoint_join_tolerates_v1(self):
        self.assertEqual(tc.endpoint("http://h:1", "/chat/completions"),
                         "http://h:1/v1/chat/completions")
        self.assertEqual(tc.endpoint("http://h:1/v1", "/chat/completions"),
                         "http://h:1/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
