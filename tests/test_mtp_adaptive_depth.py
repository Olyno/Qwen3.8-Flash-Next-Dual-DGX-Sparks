import ast
import importlib.util
import inspect
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "patch_mtp_adaptive_depth.py"

spec = importlib.util.spec_from_file_location("patch_mtp_adaptive_depth", SCRIPT)
pad = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pad)


def _fake_speculator_src() -> str:
    """Syntactically valid stand-in carrying every speculator anchor once."""
    return (
        pad.SP_IMPORTS_OLD
        + pad.SP_DIST_OLD
        + "from vllm.logger import init_logger\n"
        + pad.SP_MODULE_OLD
        + "\n\nclass AutoRegressiveSpeculator:\n"
        + "    def __init__(self):\n"
        + pad.SP_INIT_OLD
        + "\n"
        + pad.SP_METHODS_OLD
        + "\n"
        + "    def propose(self, num_reqs):\n"
        + pad.SP_PROPOSE_MID_OLD
        + "        ...\n"
        + pad.SP_PROPOSE_TAIL_OLD
        + "\n"
        + "    def _multi_step_decode(self, num_reqs):\n"
        + pad.SP_MSD_LOOP_OLD
        + "            ...\n"
        + "\n"
        + "    def _generate_fused_drafts(self, num_reqs):\n"
        + pad.SP_FUSED_LOOP_OLD
        + "                num_reqs,\n"
        + "            )\n"
    )


def _fake_runner_src() -> str:
    """Syntactically valid stand-in carrying the model_runner anchor once."""
    return "class Runner:\n    def step(self, input_batch):\n" + pad.MR_HANDOFF_OLD


def _stage_origs(speculator_src=None, runner_src=None):
    tmp = Path(tempfile.mkdtemp())
    if speculator_src is not None:
        (tmp / "speculator.py").write_text(speculator_src)
    if runner_src is not None:
        (tmp / "model_runner.py").write_text(runner_src)
    return tmp


def _run(orig_dir, out_dir):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(orig_dir), str(out_dir)],
        capture_output=True, text=True)


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


class BatchRule(unittest.TestCase):
    def test_running_means(self):
        # req A: 0.9, 0.4; req B: 1.0, 1.0 -> means [0.95, (0.36+1.0)/2].
        self.assertEqual(
            pad.adaptive_running_survival_means([[0.9, 0.4], [1.0, 1.0]]),
            [0.95, 0.68],
        )

    def test_batch_keep_uses_mean_of_products(self):
        # Per-step means would be (0.925, 0.7) — no cut. The mean of
        # per-request products is 0.45 < 0.5 — cut after column 2.
        rows = [[0.9, 0.4], [0.95, 1.0]]
        self.assertEqual(pad.adaptive_batch_keep(rows, 0.5), 2)
        self.assertEqual(pad.adaptive_batch_keep(rows, 0.4), 2)  # 0.45 >= 0.4

    def test_batch_keep_floor_of_one(self):
        self.assertEqual(pad.adaptive_batch_keep([[0.2, 0.9], [0.1, 0.9]], 0.5), 1)

    def test_batch_keep_empty(self):
        self.assertEqual(pad.adaptive_batch_keep([], 0.5), 1)

    def test_single_request_matches_adaptive_draft_length(self):
        for probs in (
            [0.9, 0.7, 0.4, 0.9],
            [0.2, 0.9, 0.9],
            [0.9, 0.9, 0.9, 0.9],
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0, 1.0],
        ):
            for threshold in (0.0, 0.3, 0.5, 0.9, 1.0):
                self.assertEqual(
                    pad.adaptive_batch_keep([probs], threshold),
                    pad.adaptive_draft_length(probs, threshold),
                    f"probs={probs} threshold={threshold}",
                )

    def test_loop_break_matches_finalize(self):
        # Simulate the patched control flow: the host-driven loops break
        # before drafting column `step` when the running-mean after column
        # step-1 crosses (_adaptive_should_cut sets produced = step);
        # propose() then keeps adaptive_batch_keep over the produced columns
        # (_adaptive_finalize). The two must agree on the width, and the
        # width must equal the post-hoc rule over the full chain.
        batches = (
            [[0.9, 0.7, 0.4, 0.9]],
            [[0.9, 0.9, 0.9, 0.9]],
            [[0.9, 0.4], [0.95, 1.0]],
            [[0.2, 0.9, 0.9], [0.8, 0.8, 0.8]],
            [[1.0, 1.0, 1.0, 1.0], [0.6, 0.9, 0.9, 0.9]],
        )
        for rows in batches:
            k = len(rows[0])
            for threshold in (0.0, 0.3, 0.5, 0.9, 1.0):
                produced = k
                for step in range(1, k):
                    means = pad.adaptive_running_survival_means(
                        [r[:step] for r in rows]
                    )
                    if pad._adaptive_cut(means[-1], threshold):
                        produced = step
                        break
                keep = pad.adaptive_batch_keep(
                    [r[:produced] for r in rows], threshold
                )
                self.assertEqual(
                    keep, produced, f"rows={rows} threshold={threshold}"
                )
                self.assertEqual(
                    keep,
                    pad.adaptive_batch_keep(rows, threshold),
                    f"rows={rows} threshold={threshold}",
                )

    def test_injected_block_is_the_tested_source(self):
        # The overlay embeds these very functions via inspect.getsource; pin
        # that so the offline tests cover the code that runs in the container.
        for fn in (
            pad._adaptive_cut,
            pad.adaptive_draft_length,
            pad.adaptive_running_survival_means,
            pad.adaptive_batch_keep,
        ):
            self.assertIn(inspect.getsource(fn), pad.PURE_BLOCK)
        self.assertIn(pad.PURE_BLOCK, pad.SP_MODULE_NEW)


class PatchAnchors(unittest.TestCase):
    def test_missing_orig_fails(self):
        tmp = _stage_origs()
        try:
            r = _run(tmp, tmp)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("missing", r.stderr)
            self.assertIn("speculator.py", r.stderr)
        finally:
            shutil.rmtree(tmp)

    def test_missing_runner_orig_fails(self):
        tmp = _stage_origs(speculator_src=_fake_speculator_src())
        out = tmp / "out"
        try:
            r = _run(tmp, out)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("model_runner.py", r.stderr)
            # Fail closed: no partial output pair.
            self.assertFalse((out / "model_runner_v030.py").exists())
        finally:
            shutil.rmtree(tmp)

    def test_applies_to_fake_origs(self):
        tmp = _stage_origs(_fake_speculator_src(), _fake_runner_src())
        out = tmp / "out"
        try:
            r = _run(tmp, out)
            self.assertEqual(r.returncode, 0, r.stderr)
            spec_out = (out / "speculator_v030.py").read_text()
            ast.parse(spec_out)
            self.assertEqual(spec_out.count("def _adaptive_should_cut"), 1)
            self.assertEqual(spec_out.count("def _adaptive_finalize"), 1)
            self.assertEqual(spec_out.count("def sample_draft"), 1)
            self.assertEqual(spec_out.count("_adaptive_should_cut(num_reqs, step)"), 2)
            self.assertIn('os.environ.get("VLLM_MTP_ADAPTIVE_DEPTH", "0")', spec_out)
            self.assertIn("_adaptive_finalize(num_reqs)", spec_out)
            self.assertIn("adaptive_batch_keep", spec_out)
            runner_out = (out / "model_runner_v030.py").read_text()
            ast.parse(runner_out)
            self.assertIn("adaptive_num_draft_tokens", runner_out)
            self.assertIn("draft_tokens_for_handler", runner_out)
        finally:
            shutil.rmtree(tmp)

    def test_already_patched_fails(self):
        tmp = _stage_origs(_fake_speculator_src(), _fake_runner_src())
        out = tmp / "out"
        try:
            self.assertEqual(_run(tmp, out).returncode, 0)
            r = _run(out, tmp / "out2")  # outputs are named *_v030.py, so...
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("missing", r.stderr)
            # ...feed the patched files back under the orig names.
            repatched = _stage_origs(
                (out / "speculator_v030.py").read_text(),
                (out / "model_runner_v030.py").read_text(),
            )
            try:
                r = _run(repatched, tmp / "out3")
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("already patched", r.stderr)
            finally:
                shutil.rmtree(repatched)
        finally:
            shutil.rmtree(tmp)

    def test_anchor_drift_fails_closed(self):
        drifted = _fake_speculator_src().replace(
            "self.use_fused_multi_step_decode = False",
            "self.use_fused_multi_step_decode = None",
        )
        tmp = _stage_origs(drifted, _fake_runner_src())
        try:
            r = _run(tmp, tmp / "out")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("anchor", r.stderr)
            self.assertIn("speculator.py", r.stderr)
        finally:
            shutil.rmtree(tmp)

    def test_patches_real_v030_source(self):
        # origs extracted from the v0.30.0 image at boot; skipped until then.
        orig_dir = REPO / "patches" / "mtp_adaptive_depth" / "orig"
        if not (orig_dir / "speculator.py").is_file():
            self.skipTest("no extracted v0.30 speculator.py; boot the prod recipe once")
        tmp = Path(tempfile.mkdtemp())
        try:
            r = _run(orig_dir, tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            spec_out = (tmp / "speculator_v030.py").read_text()
            ast.parse(spec_out)
            self.assertEqual(spec_out.count("def _adaptive_should_cut"), 1)
            self.assertIn('os.environ.get("VLLM_MTP_ADAPTIVE_DEPTH", "0")', spec_out)
            runner_out = (tmp / "model_runner_v030.py").read_text()
            ast.parse(runner_out)
            self.assertIn("adaptive_num_draft_tokens", runner_out)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
