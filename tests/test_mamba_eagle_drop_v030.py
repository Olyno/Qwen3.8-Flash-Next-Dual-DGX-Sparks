import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "mamba_eagle_drop" / "apply_patch.py"
ORIG_DIR = REPO / "patches" / "mamba_eagle_drop" / "orig"
ORIGS = ("single_type_kv_cache_manager.py", "kv_cache_coordinator.py")
OUTS = ("single_type_kv_cache_manager_v030.py", "kv_cache_coordinator_v030.py")


@unittest.skipUnless(
    all((ORIG_DIR / o).is_file() for o in ORIGS),
    "no extracted v0.30 origs; boot once on the v030 lane (engine/patches.sh "
    "extracts them) or stage patches/mamba_eagle_drop/orig/ from the image",
)
class MambaEagleDropV030(unittest.TestCase):
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

    def test_outputs_parse_and_carry_markers(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        mgr = (self.out / OUTS[0]).read_text()
        ast.parse(mgr)
        # drop honored in both branches: 2 assignments + 2 checks + 2 clears
        self.assertEqual(mgr.count("skip_next_hit"), 6)
        self.assertIn("skip only the first real match we find", mgr)
        self.assertIn("keep scanning for the next", mgr)
        # The drop sits before the hit commit in both branches.
        fine = mgr[mgr.index("max_num_partial_units = min("):]
        fine = fine[: fine.index("return computed_blocks, hit_length")]
        self.assertLess(fine.index("skip_next_hit = drop_eagle_block"),
                        fine.index("hit_length = num_tokens"))
        coarse = mgr[mgr.index("max_num_blocks = max_length // block_size"):]
        self.assertLess(coarse.index("if skip_next_hit:"),
                        coarse.index("hit_length = (i + 1) * block_size"))
        coord = (self.out / OUTS[1]).read_text()
        ast.parse(coord)
        self.assertIn("never needs to search *beyond* the candidate length", coord)
        self.assertNotIn("its finder never drops", coord)
        # Logic untouched: the MambaSpec margin skip stays as-is.
        self.assertIn("if drop_eagle_block and not isinstance(spec, MambaSpec):",
                      coord)

    def test_rerun_gives_same_output(self):
        self.assertEqual(self.run_patch().returncode, 0)
        first = {n: (self.out / n).read_text() for n in OUTS}
        self.assertEqual(self.run_patch().returncode, 0)
        for n in OUTS:
            self.assertEqual((self.out / n).read_text(), first[n])

    def test_already_patched_input_fails(self):
        self.assertEqual(self.run_patch().returncode, 0)
        for n_in, n_out in zip(ORIGS, OUTS):
            shutil.copy(self.out / n_out, self.orig / n_in)
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already patched", r.stderr)

    def test_moved_anchor_fails_without_writing(self):
        orig = self.orig / ORIGS[0]
        orig.write_text(orig.read_text().replace(
            "block_idx = fine_idx // scale_factor",
            "block_idx =  fine_idx // scale_factor"))
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.out / OUTS[0]).exists())


if __name__ == "__main__":
    unittest.main()
