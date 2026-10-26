import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
FILES = REPO / "patches"
ORIG = FILES / "mtp_v030_patched.py.orig"
SCRIPT = REPO / "patches" / "patch_mtp_draft_vocab_v030.py"


class Fp8DraftHeadAnchors(unittest.TestCase):
    """Source-level pins for the FP8 draft-head block (no extracted orig needed)."""

    def setUp(self):
        self.src = SCRIPT.read_text()

    def test_slice_guard_fails_closed(self):
        self.assertIn('rows = getattr(model, "_draft_lm_head_weight", None)', self.src)
        self.assertIn("if rows is None:", self.src)
        self.assertIn("no draft-vocab slice", self.src)

    def test_no_full_vocab_fallback(self):
        self.assertNotIn("_draft_head_full", self.src)
        self.assertNotIn("full vocab", self.src)
        self.assertNotIn("shard_indices", self.src)

    def test_empty_cache_after_build(self):
        self.assertIn("torch.cuda.empty_cache()", self.src)

    def test_memory_instrumentation(self):
        self.assertIn("torch.cuda.memory_allocated()", self.src)
        self.assertIn("torch.cuda.memory_reserved()", self.src)

    def test_attach_reports_draft_vocab_only(self):
        self.assertIn('logger.info(\n        "MTP FP8 draft head (draft vocab):', self.src)


class ScaleAwareSlice(unittest.TestCase):
    """The 47k slice must carry a quantized head's per-row scale with it."""

    def setUp(self):
        self.block = (FILES / "patch_mtp_draft_vocab.py").read_text()

    def test_float8_counts_as_quantized(self):
        # float8 dtypes report is_floating_point() True, which silently made the
        # slice keep raw FP8 rows with no scale attached.
        self.assertIn('startswith("torch.float8")', self.block)
        self.assertNotIn("not weight.is_floating_point()", self.block)

    def test_scale_sliced_with_the_same_rows(self):
        self.assertIn("wscale.data.index_select(0, rows)", self.block)
        self.assertIn("wscale.shape[0] != weight.shape[0]", self.block)

    def test_quantized_head_without_scale_is_refused(self):
        self.assertIn("without a per-row", self.block)
        self.assertIn("skipping.", self.block)

    def test_slice_materialised_in_bfloat16(self):
        self.assertIn("slice_weight.to(torch.bfloat16)", self.block)
        self.assertIn("unsqueeze(1)", self.block)


@unittest.skipUnless(ORIG.is_file(), "no extracted v0.32 mtp.py; boot the prod recipe once (RECIPE=prod ./start.sh)")
class DraftVocabV030(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        for name in ("patch_mtp_draft_vocab_v030.py", "patch_mtp_draft_vocab.py", "mtp_v030_patched.py.orig"):
            shutil.copy(FILES / name, self.tmp / name)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_patch(self):
        return subprocess.run([sys.executable, str(self.tmp / "patch_mtp_draft_vocab_v030.py")], capture_output=True, text=True)

    def test_patches_get_top_tokens_and_attach(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        src = (self.tmp / "mtp_v030_patched.py").read_text()
        ast.parse(src)
        self.assertIn("def get_top_tokens(self, hidden_states", src)
        self.assertIn("_attach_draft_vocab(self)", src)
        self.assertEqual(src.count("logger = init_logger(__name__)"), 1)

    def test_rerun_gives_same_output(self):
        self.run_patch()
        first = (self.tmp / "mtp_v030_patched.py").read_text()
        self.assertEqual(self.run_patch().returncode, 0)
        self.assertEqual((self.tmp / "mtp_v030_patched.py").read_text(), first)

    def test_moved_anchor_fails_without_writing(self):
        orig = self.tmp / "mtp_v030_patched.py.orig"
        orig.write_text(orig.read_text().replace("return loader.load_weights(", "return  loader.load_weights("))
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.tmp / "mtp_v030_patched.py").exists())


if __name__ == "__main__":
    unittest.main()
