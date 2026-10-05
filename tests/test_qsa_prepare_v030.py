import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "qsa_prepare" / "apply_patch.py"
ORIG_DIR = REPO / "patches" / "qsa_prepare" / "orig"
ORIGS = ("qsa.py", "indexer_qsa.py", "qsa_pre_indexer.py")
OUTS = ("qsa_v030.py", "indexer_qsa_v030.py", "qsa_prepare_v030.py")


@unittest.skipUnless(
    all((ORIG_DIR / o).is_file() for o in ORIGS),
    "no extracted v0.30 origs; boot once on the v030 lane (engine/patches.sh "
    "extracts them) or stage patches/qsa_prepare/orig/ from the image",
)
class QSAPrepareFusionV030(unittest.TestCase):
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
        qsa = (self.out / "qsa_v030.py").read_text()
        self.assertIn("self.use_fused_qsa_prepare = (", qsa)
        self.assertIn("selected, main_outputs = self.indexer(", qsa)
        self.assertIn("if main_outputs is None:", qsa)
        self.assertIn("attn_output = qkv.new_empty(", qsa)
        idx = (self.out / "indexer_qsa_v030.py").read_text()
        self.assertIn("from .ops.qsa_prepare import qsa_prepare", idx)
        self.assertIn('attn: "Qwen4ExpQSAAttention",', idx)
        self.assertIn("main_kv_cache = attn.kv_cache.transpose(1, 2)", idx)
        self.assertIn("main_k_scale=attn._k_scale_float,", idx)
        self.assertEqual(idx.count("return out, main_outputs"), 2)
        self.assertIn("return out, None", idx)
        prep = (self.out / "qsa_prepare_v030.py").read_text()
        self.assertIn("def qsa_prepare(", prep)
        self.assertEqual(prep.count("_qsa_prepare_kernel["), 1)
        self.assertIn("num_k_work + num_q_work + num_main_work", prep)
        self.assertIn("def _store_rotated(", prep)
        self.assertIn("main_qkv_ptr=main_qkv,", prep)
        self.assertIn('__all__ = ["qsa_prepare"]', prep)
        # Folded-in RoPE clamp (was patches/patch_qsa_rope_clamp_v030.py):
        # _norm_rope, the new main-attention section, and the launch site.
        self.assertGreaterEqual(prep.count("CLAMP_POS"), 6)
        self.assertIn('os.environ.get("VLLM_QSA_ROPE_CLAMP", "0") == "1"', prep)
        self.assertIn("pos = tl.minimum(tl.maximum(pos, 0), MAX_POS)", prep)

    def test_rerun_gives_same_output(self):
        self.assertEqual(self.run_patch().returncode, 0)
        first = {n: (self.out / n).read_text() for n in OUTS}
        self.assertEqual(self.run_patch().returncode, 0)
        for n in OUTS:
            self.assertEqual((self.out / n).read_text(), first[n])

    def test_already_patched_input_fails(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        # Feed the patcher its own indexer output: the marker check must
        # refuse rather than double-apply.
        shutil.copy(self.out / "indexer_qsa_v030.py", self.orig / "indexer_qsa.py")
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already patched", r.stderr)

    def test_moved_anchor_fails_without_writing(self):
        orig = self.orig / "indexer_qsa.py"
        orig.write_text(orig.read_text().replace(
            "from .ops.qsa_pre_indexer import qsa_pre_indexer",
            "from .ops.qsa_pre_indexer import  qsa_pre_indexer"))
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.out / "indexer_qsa_v030.py").exists())


if __name__ == "__main__":
    unittest.main()
