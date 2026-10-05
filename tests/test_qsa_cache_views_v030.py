import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "qsa_cache_views" / "apply_patch.py"
ORIG_DIR = REPO / "patches" / "qsa_cache_views" / "orig"
ORIGS = ("qsa_cache.py",)
OUTS = ("qsa_cache_v030.py",)


@unittest.skipUnless(
    all((ORIG_DIR / o).is_file() for o in ORIGS),
    "no extracted v0.30 origs; boot once on the v030 lane (engine/patches.sh "
    "extracts them) or stage patches/qsa_cache_views/orig/ from the image",
)
class QSACacheViewsV030(unittest.TestCase):
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

    def test_output_parses_and_carries_markers(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        src = (self.out / "qsa_cache_v030.py").read_text()
        ast.parse(src)
        self.assertIn("Derived on access", src)
        self.assertEqual(src.count("@property"), 2)
        self.assertIn("def key_cache(self) -> torch.Tensor:", src)
        self.assertIn("def rope_position_cache(self) -> torch.Tensor | None:", src)
        self.assertIn("return self.kv_cache[..., : self.key_head_size]", src)
        self.assertIn(
            "return self.kv_cache[..., self.rope_position_offset :].view(torch.int64)",
            src,
        )
        # The persistent views are gone; bind_kv_cache is no longer overridden.
        self.assertNotIn("self.key_cache = qsa_cache", src)
        self.assertNotIn("self.rope_position_cache = position_tail", src)
        cls = src[src.index("class QSAKeyStateCache("):]
        cls = cls[: cls.index("class QSACompressedKeyCache(")]
        self.assertNotIn("def bind_kv_cache", cls)

    def test_rerun_gives_same_output(self):
        self.assertEqual(self.run_patch().returncode, 0)
        first = (self.out / "qsa_cache_v030.py").read_text()
        self.assertEqual(self.run_patch().returncode, 0)
        self.assertEqual((self.out / "qsa_cache_v030.py").read_text(), first)

    def test_already_patched_input_fails(self):
        self.assertEqual(self.run_patch().returncode, 0)
        shutil.copy(self.out / "qsa_cache_v030.py", self.orig / "qsa_cache.py")
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already patched", r.stderr)

    def test_moved_anchor_fails_without_writing(self):
        orig = self.orig / "qsa_cache.py"
        orig.write_text(orig.read_text().replace(
            "self.key_cache = qsa_cache[..., : self.key_head_size]",
            "self.key_cache =  qsa_cache[..., : self.key_head_size]"))
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.out / "qsa_cache_v030.py").exists())


if __name__ == "__main__":
    unittest.main()
