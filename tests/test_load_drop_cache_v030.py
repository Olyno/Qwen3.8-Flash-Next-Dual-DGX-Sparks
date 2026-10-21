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


@unittest.skipUnless(ORIG.is_file(), "no extracted v0.32 weight_utils.py; boot the prod recipe once (RECIPE=prod ./start.sh)")
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
        self.assertEqual(src.count("def _fadvise_dontneed("), 1)
        # 3 per-shard calls in the iterator + 1 inside the load-end sweep
        self.assertEqual(src.count("_fadvise_dontneed(st_file)"), 4)
        self.assertIn("os.POSIX_FADV_DONTNEED", src)
        self.assertIn('os.environ.get("VLLM_LOAD_DROP_CACHE", "0") != "1"', src)

    def test_end_of_load_sweep_after_the_shard_loop(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        src = (self.out / "weight_utils_v030.py").read_text()
        self.assertEqual(src.count("def _fadvise_dontneed_load_end("), 1)
        self.assertEqual(src.count("_fadvise_dontneed_load_end(hf_weights_files)"), 1)
        # the sweep runs at function level, after the last per-shard drop
        it = src[src.index("def safetensors_weights_iterator"):]
        self.assertLess(it.rfind("_fadvise_dontneed(st_file)"),
                        it.index("_fadvise_dontneed_load_end(hf_weights_files)"))
        for name in ("chat_template.jinja", "model.safetensors.index.json",
                     "hf_quant_config.json"):
            self.assertIn(name, src)

    def test_load_end_sweep_is_gated_and_covers_shards_plus_sidecars(self):
        import os
        import tempfile
        import unittest.mock as mock
        self.assertEqual(self.run_patch().returncode, 0)
        src = (self.out / "weight_utils_v030.py").read_text()
        tree = ast.parse(src)
        ns = {"os": os}
        wanted = {"_fadvise_dontneed", "_fadvise_dontneed_load_end",
                  "_END_SWEEP_SIDECARS"}
        for node in tree.body:
            name = getattr(node, "name", None) or \
                getattr(getattr(node, "targets", [None])[0], "id", "")
            if name in wanted:
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             "<extract>", "exec"), ns)
        snap = tempfile.mkdtemp()
        shards = [os.path.join(snap, "model-00001-of-00002.safetensors"),
                  os.path.join(snap, "model-00002-of-00002.safetensors")]
        with mock.patch.object(os, "open", return_value=3), \
                mock.patch.object(os, "close"), \
                mock.patch.object(os, "posix_fadvise") as fadv, \
                mock.patch.dict(os.environ, {"VLLM_LOAD_DROP_CACHE": "1"}):
            ns["_fadvise_dontneed_load_end"](shards)
        dropped = [c.args for c in fadv.call_args_list]
        self.assertEqual(len(dropped), len(shards) + len(ns["_END_SWEEP_SIDECARS"]))
        self.assertTrue(all(a[3] == os.POSIX_FADV_DONTNEED for a in dropped))
        self.assertIn((3, 0, 0, os.POSIX_FADV_DONTNEED), dropped)
        with mock.patch.object(os, "open", return_value=3), \
                mock.patch.object(os, "close"), \
                mock.patch.object(os, "posix_fadvise") as fadv_off, \
                mock.patch.dict(os.environ, {"VLLM_LOAD_DROP_CACHE": "0"}):
            ns["_fadvise_dontneed_load_end"](shards)
            ns["_fadvise_dontneed"](shards[0])
        fadv_off.assert_not_called()

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
