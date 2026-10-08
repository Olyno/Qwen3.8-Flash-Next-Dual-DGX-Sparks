import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import quality_gate


def _result(sections):
    overall = sum(s["score"] for s in sections.values()) / len(sections)
    return {"version": 1, "model": "m", "sections": sections, "overall": overall}


BASELINE = _result({
    "omniscience": {"score": 62.5},
    "code": {"score": 100.0, "items": [{"id": "x", "output": "a b c", "score": 100.0}]},
    "prose": {"score": 100.0, "items": [{"id": "y", "output": "d e f", "score": 100.0}]},
})


class SimilarityTest(unittest.TestCase):
    def test_exact_match_is_100_despite_whitespace(self):
        self.assertEqual(quality_gate.similarity("a b  c\n", " a b c"), 100.0)

    def test_partial_overlap_between_0_and_100(self):
        s = quality_gate.similarity("the cat sat on the mat",
                                    "the cat ran off the mat")
        self.assertGreater(s, 0.0)
        self.assertLess(s, 100.0)

    def test_disjoint_is_low(self):
        self.assertLess(quality_gate.similarity("a b c d", "w x y z"), 20.0)


class CompareTest(unittest.TestCase):
    def test_no_regression_passes(self):
        candidate = _result({"omniscience": {"score": 62.5},
                             "code": {"score": 100.0}, "prose": {"score": 100.0}})
        report = quality_gate.compare(BASELINE, candidate, 1.0)
        self.assertTrue(report["ok"])
        self.assertEqual([r["section"] for r in report["rows"]],
                         ["omniscience", "code", "prose", "overall"])

    def test_improvement_passes(self):
        candidate = _result({"omniscience": {"score": 80.0},
                             "code": {"score": 100.0}, "prose": {"score": 100.0}})
        self.assertTrue(quality_gate.compare(BASELINE, candidate, 1.0)["ok"])

    def test_section_drop_over_threshold_fails(self):
        candidate = _result({"omniscience": {"score": 60.0},  # drop 2.5
                             "code": {"score": 100.0}, "prose": {"score": 100.0}})
        report = quality_gate.compare(BASELINE, candidate, 1.0)
        self.assertFalse(report["ok"])
        failed = [r["section"] for r in report["rows"] if not r["ok"]]
        # overall drop is 2.5/3 ≈ 0.83, within threshold — only the section fails
        self.assertEqual(failed, ["omniscience"])

    def test_drop_exactly_at_threshold_passes(self):
        candidate = _result({"omniscience": {"score": 61.5},  # drop exactly 1.0
                             "code": {"score": 100.0}, "prose": {"score": 100.0}})
        self.assertTrue(quality_gate.compare(BASELINE, candidate, 1.0)["ok"])

    def test_drop_just_over_threshold_fails(self):
        candidate = _result({"omniscience": {"score": 61.5 - 1e-9},
                             "code": {"score": 100.0}, "prose": {"score": 100.0}})
        self.assertFalse(quality_gate.compare(BASELINE, candidate, 1.0)["ok"])

    def test_missing_section_fails(self):
        candidate = _result({"code": {"score": 100.0}, "prose": {"score": 100.0}})
        report = quality_gate.compare(BASELINE, candidate, 1.0)
        self.assertFalse(report["ok"])
        omni = next(r for r in report["rows"] if r["section"] == "omniscience")
        self.assertIsNone(omni["candidate"])
        self.assertFalse(omni["ok"])


class FakeServer(BaseHTTPRequestHandler):
    text = "the cat sat on the mat"

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        body = json.dumps({"choices": [{"message": {"content": self.text},
                                        "finish_reason": "stop"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class RecordCheckRoundtripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.tmp = tempfile.TemporaryDirectory()
        cls.baseline = str(Path(cls.tmp.name) / "gate_baseline.json")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def _run(self, mode):
        return quality_gate.main([mode, "--url", self.url, "--model", "m",
                                  "--baseline", self.baseline, "--omni-limit", "0"])

    def test_record_then_identical_candidate_passes(self):
        self.assertEqual(self._run("record"), 0)
        with open(self.baseline) as f:
            baseline = json.load(f)
        self.assertEqual(set(baseline["sections"]), {"code", "prose"})
        self.assertEqual(baseline["overall"], 100.0)
        self.assertEqual(self._run("check"), 0)

    def test_degraded_candidate_fails(self):
        self.assertEqual(self._run("record"), 0)
        FakeServer.text = "completely different words showing up everywhere"
        try:
            self.assertEqual(self._run("check"), 1)
        finally:
            FakeServer.text = "the cat sat on the mat"


if __name__ == "__main__":
    unittest.main()
