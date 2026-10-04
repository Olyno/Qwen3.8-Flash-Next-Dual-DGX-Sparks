import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "patch_qsa_fused_draft_v030.py"
ORIG = REPO / "patches" / "v030_qsa_fused" / "orig" / "qsa_cache.py"


@unittest.skipUnless(ORIG.is_file(), "no extracted v0.30 qsa_cache.py; boot the prod recipe once (RECIPE=prod ./start.sh)")
class QsaFusedDraftV030(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.orig = self.tmp / "orig"
        self.out = self.tmp / "out"
        self.orig.mkdir()
        shutil.copy(ORIG, self.orig / "qsa_cache.py")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_patch(self):
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.orig), str(self.out)],
            capture_output=True, text=True)

    def test_opts_builder_into_fused_draft(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        src = (self.out / "qsa_cache_v030.py").read_text()
        ast.parse(src)
        self.assertIn("def update_draft_decode_metadata(self, metadata: QSAForwardMetadata)", src)
        self.assertIn("supports_draft_decode_metadata_update = HAS_TRITON", src)
        self.assertIn('os.environ.get("VLLM_QSA_FUSED_DRAFT", "0") == "1"', src)
        self.assertEqual(src.count("def _launch_qsa_metadata_kernel"), 1)
        self.assertIn("    common_slot_mapping: torch.Tensor\n    num_mapped_tokens: int\n", src)

    def test_rerun_gives_same_output(self):
        self.run_patch()
        first = (self.out / "qsa_cache_v030.py").read_text()
        self.assertEqual(self.run_patch().returncode, 0)
        self.assertEqual((self.out / "qsa_cache_v030.py").read_text(), first)

    def test_moved_anchor_fails_without_writing(self):
        orig = self.orig / "qsa_cache.py"
        orig.write_text(orig.read_text().replace("num_reqs = common_attn_metadata.query_start_loc.shape[0] - 1", "num_reqs  = 1"))
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.out / "qsa_cache_v030.py").exists())


if __name__ == "__main__":
    unittest.main()
