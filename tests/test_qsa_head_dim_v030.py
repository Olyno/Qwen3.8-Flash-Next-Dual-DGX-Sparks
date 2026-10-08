import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "qsa_head_dim" / "apply_patch.py"
ORIG_DIR = REPO / "patches" / "qsa_head_dim" / "orig"
ORIGS = ("qsa.py",)
OUTS = ("qsa_v030.py",)


@unittest.skipUnless(
    all((ORIG_DIR / o).is_file() for o in ORIGS),
    "no extracted v0.30 origs; boot once on the v030 lane (engine/patches.sh "
    "extracts them) or stage patches/qsa_head_dim/orig/ from the image",
)
class QSAHeadDimV030(unittest.TestCase):
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

    def test_output_parses_and_fixes_the_fallback(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        src = (self.out / "qsa_v030.py").read_text()
        ast.parse(src)
        self.assertIn(
            "self.head_dim = int(config.head_dim or self.hidden_size // self.total_num_heads)",
            src)
        self.assertNotIn(
            "self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)",
            src)
        # Only the fallback expression changed; the sharded head count stays.
        self.assertIn("self.num_heads = self.total_num_heads // tp_size", src)
        self.assertIn("self.q_size = self.num_heads * self.head_dim", src)

    def test_rerun_gives_same_output(self):
        self.assertEqual(self.run_patch().returncode, 0)
        first = (self.out / "qsa_v030.py").read_text()
        self.assertEqual(self.run_patch().returncode, 0)
        self.assertEqual((self.out / "qsa_v030.py").read_text(), first)

    def test_already_patched_input_fails(self):
        self.assertEqual(self.run_patch().returncode, 0)
        shutil.copy(self.out / "qsa_v030.py", self.orig / "qsa.py")
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already patched", r.stderr)

    def test_moved_anchor_fails_without_writing(self):
        orig = self.orig / "qsa.py"
        orig.write_text(orig.read_text().replace(
            "self.hidden_size // self.num_heads",
            "self.hidden_size //  self.num_heads"))
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.out / "qsa_v030.py").exists())


if __name__ == "__main__":
    unittest.main()
