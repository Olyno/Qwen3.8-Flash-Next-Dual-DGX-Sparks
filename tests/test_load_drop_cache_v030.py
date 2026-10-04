import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "patch_load_drop_cache_v030.py"
ORIG = REPO / "patches" / "v030_dropcache" / "orig" / "weight_utils.py"


@unittest.skipUnless(ORIG.is_file(), "no extracted v0.30 weight_utils.py; boot the prod recipe once (RECIPE=prod ./start.sh)")
class LoadDropCacheV030(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.orig = self.tmp / "orig"
        self.out = self.tmp / "out"
        self.orig.mkdir()
        shutil.copy(ORIG, self.orig / "weight_utils.py")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_patch(self):
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.orig), str(self.out)],
            capture_output=True, text=True)

    def test_drops_each_consumed_shard(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        src = (self.out / "weight_utils_v030.py").read_text()
        ast.parse(src)
        self.assertEqual(src.count("def _fadvise_dontneed"), 1)
        self.assertEqual(src.count("_fadvise_dontneed(st_file)"), 3)
        self.assertIn("os.POSIX_FADV_DONTNEED", src)
        self.assertIn('os.environ.get("VLLM_LOAD_DROP_CACHE", "0") != "1"', src)

    def test_rerun_gives_same_output(self):
        self.run_patch()
        first = (self.out / "weight_utils_v030.py").read_text()
        self.assertEqual(self.run_patch().returncode, 0)
        self.assertEqual((self.out / "weight_utils_v030.py").read_text(), first)

    def test_moved_anchor_fails_without_writing(self):
        orig = self.orig / "weight_utils.py"
        orig.write_text(orig.read_text().replace("yield from unflattened_state_dict.items()", "yield from  unflattened_state_dict.items()"))
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.out / "weight_utils_v030.py").exists())


if __name__ == "__main__":
    unittest.main()
