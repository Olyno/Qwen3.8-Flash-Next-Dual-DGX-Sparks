import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "hc_down_silu" / "apply_patch.py"
ORIG_DIR = REPO / "patches" / "hc_down_silu" / "orig"
ORIGS = ("hyperconnection.py", "model.py")
OUTS = ("hyperconnection_v030.py", "model_v030.py")


@unittest.skipUnless(
    all((ORIG_DIR / o).is_file() for o in ORIGS),
    "no extracted v0.30 origs; boot once on the v030 lane (engine/patches.sh "
    "extracts them) or stage patches/hc_down_silu/orig/ from the image",
)
class HCDownSiluV030(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.orig = self.tmp / "orig"
        self.out = self.tmp / "out"
        self.orig.mkdir()
        self.out.mkdir()
        for name in ORIGS:
            shutil.copy(ORIG_DIR / name, self.orig / name)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_patch(self):
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.orig), str(self.out)],
            capture_output=True, text=True)

    def test_all_outputs_parse_and_carry_markers(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        for name in OUTS:
            ast.parse((self.out / name).read_text())
        hc = (self.out / "hyperconnection_v030.py").read_text()
        self.assertIn("def _down_and_inject", hc)
        self.assertEqual(hc.count("self._down_and_inject(xn)"), 2)
        self.assertIn("hc_down_silu(", hc)
        self.assertIn("_use_hc_down_silu", hc)
        self.assertIn("MAX_FUSED_M", hc)
        self.assertIn("from .ops.cute_dsl.hc_down_silu import", hc)
        mod = (self.out / "model_v030.py").read_text()
        self.assertIn("request_hc_down_silu_warmup(", mod)
        # replayssm_gdn chains off this model.py output: its anchors must
        # survive untouched.
        self.assertIn(
            "from .qsa import Qwen4ExpQSAAttention\n\n\ndef without_modelopt_fp4(",
            mod)

    def test_rerun_gives_same_output(self):
        self.assertEqual(self.run_patch().returncode, 0)
        first = {n: (self.out / n).read_text() for n in OUTS}
        self.assertEqual(self.run_patch().returncode, 0)
        for n in OUTS:
            self.assertEqual((self.out / n).read_text(), first[n])

    def test_already_patched_input_fails(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        # Feed the patcher its own hyperconnection output: the marker check
        # must refuse rather than double-apply.
        shutil.copy(self.out / "hyperconnection_v030.py",
                    self.orig / "hyperconnection.py")
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already patched", r.stderr)

    def test_moved_anchor_fails_without_writing(self):
        orig = self.orig / "hyperconnection.py"
        orig.write_text(orig.read_text().replace(
            "from vllm.model_executor.models.utils import maybe_prefix",
            "from vllm.model_executor.models.utils import  maybe_prefix"))
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.out / "hyperconnection_v030.py").exists())


if __name__ == "__main__":
    unittest.main()
