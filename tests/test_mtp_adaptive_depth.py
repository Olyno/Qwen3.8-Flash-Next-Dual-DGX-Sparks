import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
import unittest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "patch_mtp_adaptive_depth.py"

spec = importlib.util.spec_from_file_location("patch_mtp_adaptive_depth", SCRIPT)
pad = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pad)


class AdaptiveDraftLength(unittest.TestCase):
    def test_truncates_when_survival_drops(self):
        # 0.9 -> 0.63 -> 0.252 < 0.5: keep the token that crossed, stop.
        self.assertEqual(pad.adaptive_draft_length([0.9, 0.7, 0.4, 0.9], 0.5), 3)

    def test_floor_of_one(self):
        # First token already below threshold: still keep exactly one.
        self.assertEqual(pad.adaptive_draft_length([0.2, 0.9, 0.9], 0.5), 1)

    def test_confident_chain_runs_full_length(self):
        self.assertEqual(pad.adaptive_draft_length([0.9, 0.9, 0.9, 0.9], 0.5), 4)

    def test_threshold_edge_is_strict(self):
        # prod == threshold keeps drafting; the next step crosses.
        self.assertEqual(pad.adaptive_draft_length([0.5, 0.5], 0.5), 2)
        self.assertEqual(pad.adaptive_draft_length([0.5, 0.9], 0.5), 2)

    def test_threshold_zero_never_cuts(self):
        self.assertEqual(pad.adaptive_draft_length([0.01, 0.01, 0.01], 0.0), 3)

    def test_threshold_one(self):
        # p == 1.0 is not < 1.0; the first sub-certain token stops the chain.
        self.assertEqual(pad.adaptive_draft_length([1.0, 0.9, 0.9], 1.0), 2)
        self.assertEqual(pad.adaptive_draft_length([0.9, 0.9], 1.0), 1)

    def test_single_token(self):
        self.assertEqual(pad.adaptive_draft_length([0.3], 0.5), 1)

    def test_matches_incremental_loop_rule(self):
        # The proposer loop breaks before drafting token n+1 when the running
        # product after token n crosses the threshold; the kept count must
        # match adaptive_draft_length on the same sequence.
        for probs in (
            [0.9, 0.7, 0.4, 0.9],
            [0.2, 0.9, 0.9],
            [0.9, 0.9, 0.9, 0.9],
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0, 1.0],
        ):
            for threshold in (0.0, 0.3, 0.5, 0.9, 1.0):
                prod = 1.0
                keep = 0
                for p in probs:
                    if keep >= 1 and pad._adaptive_cut(prod, threshold):
                        break
                    prod *= p
                    keep += 1
                self.assertEqual(
                    keep, pad.adaptive_draft_length(probs, threshold),
                    f"probs={probs} threshold={threshold}",
                )

    def test_injected_block_is_the_tested_source(self):
        # The overlay embeds these very functions via inspect.getsource; pin
        # that so the offline tests cover the code that runs in the container.
        self.assertIn(inspect.getsource(pad._adaptive_cut), pad.PURE_BLOCK)
        self.assertIn(inspect.getsource(pad.adaptive_draft_length), pad.PURE_BLOCK)
        self.assertIn(pad.PURE_BLOCK, pad.MODULE_NEW)


class PatchAnchors(unittest.TestCase):
    def test_missing_orig_fails(self):
        import tempfile
        import shutil
        tmp = Path(tempfile.mkdtemp())
        try:
            r = subprocess.run(
                [sys.executable, str(SCRIPT), str(tmp), str(tmp)],
                capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("missing", r.stderr)
        finally:
            shutil.rmtree(tmp)

    def test_patches_real_v030_source(self):
        # orig extracted from the v0.30.0 image at boot; skipped until then.
        orig = REPO / "patches" / "mtp_adaptive_depth" / "orig" / "llm_base_proposer.py"
        if not orig.is_file():
            self.skipTest("no extracted v0.30 llm_base_proposer.py; boot the prod recipe once")
        import tempfile
        import shutil
        tmp = Path(tempfile.mkdtemp())
        try:
            r = subprocess.run(
                [sys.executable, str(SCRIPT), str(orig.parent), str(tmp)],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            out = (tmp / "llm_base_proposer_v030.py").read_text()
            import ast
            ast.parse(out)
            self.assertEqual(out.count("def _adaptive_draft_sample"), 1)
            self.assertIn("_adaptive_cut(", out)
            self.assertIn('os.environ.get("VLLM_MTP_ADAPTIVE_DEPTH", "0")', out)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
