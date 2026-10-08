import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "qsa_meta_clamp" / "apply_patch.py"
FUSED_SCRIPT = REPO / "patches" / "patch_qsa_fused_draft_v030.py"
VIEWS_SCRIPT = REPO / "patches" / "qsa_cache_views" / "apply_patch.py"
ORIG_DIR = REPO / "patches" / "qsa_meta_clamp" / "orig"
ORIGS = ("qsa_cache.py",)
CLAMP = "min(\n        int(common_attn_metadata.query_start_loc_cpu[-1]), num_tokens\n    )"


@unittest.skipUnless(
    all((ORIG_DIR / o).is_file() for o in ORIGS),
    "no extracted v0.30 origs; boot once on the v030 lane (engine/patches.sh "
    "extracts them) or stage patches/qsa_meta_clamp/orig/ from the image",
)
class QSAMetaClampV030(unittest.TestCase):
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

    def run_patch(self, orig=None, out=None):
        return subprocess.run(
            [sys.executable, str(SCRIPT),
             str(orig or self.orig), str(out or self.out)],
            capture_output=True, text=True)

    def test_stock_input_parses_and_clamps_both_builders(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        src = (self.out / "qsa_cache_v030.py").read_text()
        ast.parse(src)
        # Triton builder + torch fallback builder, upstream comment on both.
        self.assertEqual(
            src.count("# Graph padding can make the final query offset exceed "
                      "the real token count."), 2)
        self.assertEqual(src.count(CLAMP), 2)
        self.assertNotIn(
            "num_mapped_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])",
            src)

    def test_fused_draft_views_input_clamps_all_sites(self):
        # Chain the way engine/patches.sh does with QSA_FUSED_DRAFT=true:
        # stock -> fused-draft overlay -> views overlay -> this clamp.
        fused_out = self.tmp / "fused"
        views_orig = self.tmp / "views_orig"
        views_out = self.tmp / "views"
        clamp_out = self.tmp / "clamp"
        for d in (fused_out, views_orig, views_out, clamp_out):
            d.mkdir()
        r = subprocess.run(
            [sys.executable, str(FUSED_SCRIPT), str(self.orig), str(fused_out)],
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        shutil.copy(fused_out / "qsa_cache_v030.py", views_orig / "qsa_cache.py")
        r = subprocess.run(
            [sys.executable, str(VIEWS_SCRIPT), str(views_orig), str(views_out)],
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        clamp_orig = self.tmp / "clamp_orig"
        clamp_orig.mkdir()
        shutil.copy(views_out / "qsa_cache_v030.py", clamp_orig / "qsa_cache.py")
        r = self.run_patch(orig=clamp_orig, out=clamp_out)
        self.assertEqual(r.returncode, 0, r.stderr)
        src = (clamp_out / "qsa_cache_v030.py").read_text()
        ast.parse(src)
        # torch fallback + inline kernel-launch arg + recorded metadata field
        self.assertEqual(
            src.count("# Graph padding can make the final query offset"), 3)
        self.assertNotIn(
            "num_mapped_tokens=int(common_attn_metadata.query_start_loc_cpu[-1])",
            src)
        self.assertNotIn(
            "num_mapped_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])",
            src)

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
            "del request_capacity\n",
            "del  request_capacity\n"))
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.out / "qsa_cache_v030.py").exists())


if __name__ == "__main__":
    unittest.main()
