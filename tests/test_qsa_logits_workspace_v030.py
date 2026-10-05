import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "qsa_logits_workspace" / "apply_patch.py"
ORIG_DIR = REPO / "patches" / "qsa_logits_workspace" / "orig"
ORIGS = ("qsa_indexer.py",)
OUTS = ("qsa_indexer_v030.py",)


@unittest.skipUnless(
    all((ORIG_DIR / o).is_file() for o in ORIGS),
    "no extracted v0.30 origs; boot once on the v030 lane (engine/patches.sh "
    "extracts them) or stage patches/qsa_logits_workspace/orig/ from the image",
)
class QSAPrefillLogitsWorkspaceV030(unittest.TestCase):
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
        src = (self.out / "qsa_indexer_v030.py").read_text()
        ast.parse(src)
        self.assertIn("logits_workspace: torch.Tensor,", src)
        self.assertIn("assert logits_workspace.is_contiguous()", src)
        self.assertIn("logits_workspace = q.new_empty(budget_bytes // 4", src)
        self.assertIn("logits = logits_workspace[: num_queries * logits_width].view(", src)
        self.assertIn("budget_bytes = max(max_logits_bytes,", src)
        # The per-chunk torch.empty inside _prefill_logits is gone (the decode
        # path's own allocation is untouched); the call site threads the
        # workspace through.
        prefill = src[src.index("def _prefill_logits("):]
        prefill = prefill[: prefill.index("def expand_qsa_block_indices(")]
        self.assertNotIn("torch.empty(", prefill)
        call = src[src.index("logits = _prefill_logits("):]
        self.assertIn("visible_blocks,\n            logits_workspace,", call)
        self.assertEqual(src.count("def _prefill_logits("), 1)

    def test_rerun_gives_same_output(self):
        self.assertEqual(self.run_patch().returncode, 0)
        first = (self.out / "qsa_indexer_v030.py").read_text()
        self.assertEqual(self.run_patch().returncode, 0)
        self.assertEqual((self.out / "qsa_indexer_v030.py").read_text(), first)

    def test_already_patched_input_fails(self):
        self.assertEqual(self.run_patch().returncode, 0)
        shutil.copy(self.out / "qsa_indexer_v030.py", self.orig / "qsa_indexer.py")
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already patched", r.stderr)

    def test_moved_anchor_fails_without_writing(self):
        orig = self.orig / "qsa_indexer.py"
        orig.write_text(orig.read_text().replace(
            "rows_per_chunk = max(1, max_logits_bytes // (logits_width * 4))",
            "rows_per_chunk = max(1,  max_logits_bytes // (logits_width * 4))"))
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.out / "qsa_indexer_v030.py").exists())


if __name__ == "__main__":
    unittest.main()
