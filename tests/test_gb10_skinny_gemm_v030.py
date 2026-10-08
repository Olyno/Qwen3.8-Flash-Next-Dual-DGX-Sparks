import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "gb10_skinny_gemm" / "apply_patch.py"
ORIGS = ("low_latency_gemm.py.orig", "skinny_gemm.py.orig")
OUTS = ("low_latency_gemm.py", "skinny_gemm.py")


@unittest.skipUnless(
    all((SCRIPT.parent / o).is_file() for o in ORIGS),
    "no extracted v0.30 origs; boot once with skinny_gemm on (engine/patches.sh "
    "extracts them) or stage patches/gb10_skinny_gemm/*.orig from the image",
)
class GB10SkinnyGemmV030(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.orig = self.tmp / "orig"
        self.out = self.tmp / "out"
        self.orig.mkdir()
        self.out.mkdir()
        for name in ORIGS:
            shutil.copy(SCRIPT.parent / name, self.orig / name)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_patch(self):
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.orig), str(self.out)],
            capture_output=True, text=True)

    def test_output_parses_and_carries_markers(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        gemm = (self.out / "low_latency_gemm.py").read_text()
        ast.parse(gemm)
        self.assertIn("QWEN4_EXP_SM121_GEMM_PLANS", gemm)
        self.assertIn("(10240, 320):", gemm)
        # vllm#60027: dispatch accepts a row-major column slice (the fused
        # HC-down output) instead of requiring a packed row-major input.
        self.assertIn("    row_stride_ok,\n", gemm)
        self.assertIn(
            "def _runtime_ok(\n"
            "    x: torch.Tensor, weight: torch.Tensor, config: SkinnyGemmConfig\n"
            ") -> bool:",
            gemm,
        )
        self.assertIn("and x.dim() == 2", gemm)
        self.assertIn("and row_stride_ok(x, config)", gemm)
        self.assertIn("_runtime_ok(x, weight, config)", gemm)
        self.assertNotIn("and _is_packed_row_major(x)", gemm)
        kern = (self.out / "skinny_gemm.py").read_text()
        ast.parse(kern)
        self.assertIn("def row_stride_ok(", kern)
        self.assertIn('raise ValueError("b must be contiguous")', kern)
        self.assertIn(
            'raise ValueError("a must be row-major with vector_width-aligned rows")',
            kern,
        )
        self.assertNotIn("a and b must be contiguous", kern)

    def test_rerun_gives_same_output(self):
        self.assertEqual(self.run_patch().returncode, 0)
        first = {o: (self.out / o).read_text() for o in OUTS}
        self.assertEqual(self.run_patch().returncode, 0)
        for o in OUTS:
            self.assertEqual((self.out / o).read_text(), first[o])

    def test_moved_anchor_fails_without_writing(self):
        orig = self.orig / "low_latency_gemm.py.orig"
        orig.write_text(orig.read_text().replace(
            "and _is_packed_row_major(x)",
            "and  _is_packed_row_major(x)"))
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        for o in OUTS:
            self.assertFalse((self.out / o).exists())


if __name__ == "__main__":
    unittest.main()
