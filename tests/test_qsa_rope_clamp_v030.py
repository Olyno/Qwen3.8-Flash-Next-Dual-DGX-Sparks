import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "patch_qsa_rope_clamp_v030.py"
ORIG = REPO / "patches" / "v030_qsa_rope" / "orig" / "qsa_prepare.py"


@unittest.skipUnless(ORIG.is_file(), "no extracted v0.32 qsa_prepare.py; boot the prod recipe once (RECIPE=prod ./start.sh)")
class QsaRopeClampV030(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.orig = self.tmp / "orig"
        self.out = self.tmp / "out"
        self.orig.mkdir()
        shutil.copy(ORIG, self.orig / "qsa_prepare.py")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_patch(self):
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.orig), str(self.out)],
            capture_output=True, text=True)

    def test_clamps_rope_positions(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        src = (self.out / "qsa_prepare_v032.py").read_text()
        ast.parse(src)
        # _norm_rope and the fused kernel signature each take the constexpr pair
        self.assertEqual(src.count("CLAMP_POS: tl.constexpr"), 2)
        # one clamp block in _norm_rope, one in the inline main-attention section
        self.assertEqual(src.count("if CLAMP_POS:"), 2)
        self.assertIn("pos_t = tl.minimum(tl.maximum(pos_t, 0), MAX_POS)", src)
        self.assertIn("pos = tl.minimum(tl.maximum(pos, 0), MAX_POS)", src)
        self.assertIn('CLAMP_POS=os.environ.get("VLLM_QSA_ROPE_CLAMP", "0") == "1"', src)
        self.assertIn("MAX_POS=max(int(cos_sin_cache.shape[0]) - 1, 0)", src)

    def test_rerun_gives_same_output(self):
        self.run_patch()
        first = (self.out / "qsa_prepare_v032.py").read_text()
        self.assertEqual(self.run_patch().returncode, 0)
        self.assertEqual((self.out / "qsa_prepare_v032.py").read_text(), first)

    def test_moved_anchor_fails_without_writing(self):
        orig = self.orig / "qsa_prepare.py"
        orig.write_text(orig.read_text().replace("MROPE_W=section[2],", "MROPE_W=section[2],  "))
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.out / "qsa_prepare_v032.py").exists())


if __name__ == "__main__":
    unittest.main()
