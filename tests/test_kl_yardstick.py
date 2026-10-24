import hashlib
import io
import json
import sys
import tempfile
import threading
import unittest
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import kl_yardstick


def dist(true_id, extra=None, second=-3.1):
    """One vLLM prompt_logprobs element: true token on top, two fillers."""
    e = {str(true_id): {"logprob": -0.05},
         str(true_id + 7): {"logprob": second},
         str(true_id + 13): {"logprob": -5.2}}
    if extra:
        e[str(extra)] = {"logprob": -6.0}
    return e


class MathTest(unittest.TestCase):
    def test_identical_distributions_have_zero_kl(self):
        p = {1: -0.2, 2: -1.5, 9: -6.0}
        self.assertLess(kl_yardstick.kl_renorm(p, dict(p)), 1e-9)

    def test_kl_matches_closed_form_on_two_events(self):
        # P=(0.7,0.3) Q=(0.5,0.5): KL = .7ln(.7/.5)+.3ln(.3/.5) = 0.08225
        import math
        kl = kl_yardstick.kl_renorm({1: math.log(0.7), 2: math.log(0.3)},
                                    {1: math.log(0.5), 2: math.log(0.5)})
        self.assertAlmostEqual(kl, 0.08225, places=4)

    def test_disjoint_topk_kl_equals_the_margin(self):
        p = {i: -4.0 for i in range(20)}
        q = {i + 500: -4.0 for i in range(20)}
        self.assertAlmostEqual(kl_yardstick.kl_renorm(p, q),
                               kl_yardstick.MISS_MARGIN, places=6)

    def test_missing_id_renormalized_not_inf(self):
        p = {1: -0.1, 2: -9.0}
        q = {1: -0.1, 2: -9.0, 3: -9.5}
        kl = kl_yardstick.kl_renorm(p, q)
        self.assertTrue(kl == kl)  # not NaN
        self.assertLess(kl, 1.0)

    def test_percentile_nearest_rank(self):
        self.assertEqual(kl_yardstick.percentile_nearest(list(range(1, 101)), 0.99), 99)
        self.assertEqual(kl_yardstick.percentile_nearest([7], 0.99), 7)
        self.assertEqual(kl_yardstick.percentile_nearest([2, 1], 0.5), 1)

    def test_norm_dist_parses_vllm_shape(self):
        elem = {"42": {"logprob": -0.5, "rank": 1, "decoded_token": "x"},
                "meta": {"logprob": -1.0}, "7": {"note": "no logprob"}}
        self.assertEqual(kl_yardstick.norm_dist(elem), {42: -0.5})
        self.assertIsNone(kl_yardstick.norm_dist(None))
        self.assertIsNone(kl_yardstick.norm_dist({}))


class ScorePairTest(unittest.TestCase):
    def test_flips_and_token_mismatch(self):
        ids = [7, 8, 9]
        dp = [None, {"8": {"logprob": -0.05}, "1": {"logprob": -3.0}},
              {"9": {"logprob": -0.2}, "3": {"logprob": -4.0}}]
        dc = [None, {"8": {"logprob": -0.06}, "1": {"logprob": -3.1}},
              {"4": {"logprob": -0.1}, "9": {"logprob": -1.0}}]
        st = kl_yardstick.score_pair(dp, dc, ids, list(ids))
        self.assertEqual(st["n"], 2)
        self.assertEqual(st["top1"], 0.5)
        self.assertGreater(st["mean_kl"], 0.0)
        self.assertGreater(st["ppl_p"], 1.0)
        self.assertEqual([f[0] for f in st["flips"]], [2])

    def test_divergent_tokenization_refuses_to_score(self):
        with self.assertRaises(SystemExit):
            kl_yardstick.score_pair([None, dist(8)], [None, dist(8)],
                                    [7, 8], [7, 8, 9])


class SpineTest(unittest.TestCase):
    def test_segments_are_whitespace_spans(self):
        d = kl_yardstick.spine_build([{"id": "x", "text": "a bb\n\n  ccc "}])
        self.assertEqual(d["texts"]["x"]["segments"], [[0, 1], [2, 4], [8, 11]])
        self.assertEqual(d["texts"]["x"]["n_chars"], 12)

    def test_dump_load_roundtrip_and_staleness(self):
        texts = [{"id": "t1", "text": "hello frozen world"}]
        doc = kl_yardstick.spine_build(texts)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "pos.json")
            kl_yardstick.spine_dump(doc, path)
            back = kl_yardstick.load_positions(path, doc["corpus_sha256"])
            self.assertEqual(back["texts"]["t1"]["sha256"], doc["texts"]["t1"]["sha256"])
            with self.assertRaises(SystemExit):
                kl_yardstick.load_positions(path, "deadbeef")

    def test_committed_corpus_matches_committed_spine(self):
        texts = kl_yardstick.load_corpus()
        doc = kl_yardstick.load_positions()
        self.assertEqual(len(texts), 5)
        kinds = {json.loads(l)["kind"] for l in
                 open(Path(__file__).resolve().parent.parent / "bench/data/kl_corpus.jsonl")}
        self.assertEqual(kinds, {"code", "prose", "de", "agent", "math"})
        blob = hashlib.sha256()
        for t in texts:
            pin = doc["texts"][t["id"]]
            self.assertEqual(pin["sha256"],
                             hashlib.sha256(t["text"].encode()).hexdigest())
            blob.update(pin["sha256"].encode())
        self.assertEqual(doc["corpus_sha256"], blob.hexdigest())


class FakeCompletions(BaseHTTPRequestHandler):
    """Serves deterministic teacher-forced logprobs for the pinned corpus;
    FLIP=1 moves top-1 on every 8th scored position."""
    FLIP = False

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        words = req.get("prompt", "").split()
        ids = [int(hashlib.sha256(w.encode()).hexdigest(), 16) % 49000 + 1000
               for w in words]
        plist = [None]
        for i, tid in enumerate(ids):
            if self.FLIP and i % 8 == 0:
                plist.append({str(tid + 7): {"logprob": -0.05},
                              str(tid): {"logprob": -0.9},
                              str(tid + 13): {"logprob": -5.2}})
            else:
                plist.append(dist(tid))
        body = json.dumps({"choices": [{
            "index": 0, "text": "", "finish_reason": "length",
            "prompt_token_ids": ids, "prompt_logprobs": plist}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FlippedCompletions(FakeCompletions):
    FLIP = True


class CompareEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeCompletions)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.flip_server = ThreadingHTTPServer(("127.0.0.1", 0), FlippedCompletions)
        threading.Thread(target=cls.flip_server.serve_forever, daemon=True).start()
        cls.flip_url = f"http://127.0.0.1:{cls.flip_server.server_port}"
        # The repo pins carry token ids from a real tokenizer; the fake server
        # tokenizes differently, so this class scores against an unpinned copy.
        cls._tmp = tempfile.TemporaryDirectory()
        src = Path(kl_yardstick.POSITIONS)
        doc = json.loads(src.read_text())
        for v in doc["texts"].values():
            v["token_ids"] = None
        cls.positions = str(Path(cls._tmp.name) / "positions.json")
        Path(cls.positions).write_text(json.dumps(doc))

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.flip_server.shutdown()
        cls.flip_server.server_close()
        cls._tmp.cleanup()

    def _run(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = kl_yardstick.main(["compare", "--url", self.url,
                                      "--positions", self.positions] + argv)
        return code, buf.getvalue()

    def test_identical_arm_with_floor_passes(self):
        code, out = self._run(["--candidate", self.url])
        self.assertEqual(code, 0)
        self.assertIn("SELF-NOISE FLOOR", out)
        self.assertLess(out.find("SELF-NOISE FLOOR"), out.find("BASELINE vs CANDIDATE"),
                        "the floor must print before any verdict")
        self.assertIn("kl metric=floor_p99_kl", out)
        self.assertIn("kl metric=p99_kl value=0.000000 threshold=0.000000 PASS", out)
        self.assertIn("kl metric=top1_agreement value=1.000000", out)
        self.assertIn("kl overall PASS", out)

    def test_flipped_candidate_fails_and_reports_top1(self):
        code, out = self._run(["--candidate", self.flip_url])
        self.assertEqual(code, 1)
        self.assertIn("kl metric=top1_agreement value=0.", out)  # ~1/8 flipped
        self.assertIn("kl overall FAIL", out)

    def test_no_floor_requires_absolute_ceiling(self):
        code, out = self._run(["--candidate", self.url, "--no-floor",
                              "--max-p99", "0.01"])
        self.assertEqual(code, 0)
        self.assertNotIn("SELF-NOISE FLOOR", out)
        code, out = self._run(["--candidate", self.url, "--no-floor"])
        self.assertEqual(code, 1)  # fail-closed: no floor, no ceiling -> FAIL
        self.assertIn("threshold=UNSET", out)

    def test_corpus_tampering_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "corpus.jsonl"
            bad.write_text(json.dumps({"id": "code_py", "kind": "code",
                                       "text": "tampered text here\n"}))
            with self.assertRaises(SystemExit) as cm:
                self._run(["--corpus", str(bad)])
            self.assertIn("byte-frozen", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
