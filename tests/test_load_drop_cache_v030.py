import ast
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

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


@unittest.skipUnless(ORIG.is_file(), "no extracted v0.32 weight_utils.py")
class RunaiLoadDropCache(unittest.TestCase):
    """Exercise the patched RunAI consumer without GPU or optional imports."""

    def setUp(self):
        spec = importlib.util.spec_from_file_location("load_drop_cache_patch", SCRIPT)
        self.patch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.patch)
        self.events = []
        self.shards = ["/model/one.safetensors", "/model/two.safetensors"]

        events = self.events

        class Tensor:
            def clone(self):
                events.append("clone")
                return "cloned tensor"

        class Streamer:
            def __enter__(self):
                events.append("enter")
                return self

            files_to_tensors_metadata = {"fixture": ["weight"]}

            def stream_files(self, files, **kwargs):
                pass
            def __exit__(self, *args):
                events.append("exit")

            def get_tensors(self):
                yield "weight", Tensor()

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(ORIG.parent), self.tmp.name],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        tree = ast.parse((Path(self.tmp.name) / "weight_utils_v030.py").read_text())
        wanted = {
            "_fadvise_dontneed", "_fadvise_dontneed_load_end",
            "_END_SWEEP_SIDECARS", "runai_safetensors_weights_iterator",
        }
        nodes = [
            node for node in tree.body
            if (getattr(node, "name", None) or
                getattr(getattr(node, "targets", [None])[0], "id", "")) in wanted
        ]
        self.ns = {
            "os": os, "SafetensorsStreamer": Streamer,
            "current_platform": SimpleNamespace(is_cuda_alike=lambda: False),
            "tqdm": lambda iterable, **kwargs: iterable,
            "enable_tqdm": lambda value: value, "_BAR_FORMAT": "",
        }
        source = "from __future__ import annotations\n" + ast.unparse(
            ast.Module(body=nodes, type_ignores=[]))
        exec(compile(source, "<patched RunAI iterator>", "exec"), self.ns)

    def consume(self, gate):
        with mock.patch.dict(os.environ, {"VLLM_LOAD_DROP_CACHE": gate}), \
                mock.patch.object(os, "open", return_value=3) as opened, \
                mock.patch.object(os, "close"), \
                mock.patch.object(os, "posix_fadvise") as advised:
            advised.side_effect = lambda *args: self.events.append("drop")
            iterator = self.ns["runai_safetensors_weights_iterator"](self.shards, False)
            self.assertEqual(next(iterator), ("weight", "cloned tensor"))
            advised.assert_not_called()
            self.assertEqual(list(iterator), [])
        return opened, advised

    def test_runai_sweeps_shards_and_sidecars_after_streamer_release(self):
        opened, advised = self.consume("1")
        expected = self.shards + [
            "/model/" + name for name in self.ns["_END_SWEEP_SIDECARS"]
        ]
        self.assertEqual([call.args[0] for call in opened.call_args_list], expected)
        self.assertEqual(advised.call_count, len(expected))
        self.assertTrue(all(
            call.args == (3, 0, 0, os.POSIX_FADV_DONTNEED)
            for call in advised.call_args_list
        ))
        self.assertEqual(self.events[:3], ["enter", "clone", "exit"])
        self.assertEqual(self.events[3:], ["drop"] * len(expected))

    def test_runai_gate_off_preserves_stream_without_advice(self):
        opened, advised = self.consume("0")
        opened.assert_not_called()
        advised.assert_not_called()
        self.assertEqual(self.events, ["enter", "clone", "exit"])

    def test_runai_anchor_drift_fails_closed(self):
        source = self.patch.RUNAI_OLD.replace("tensor.clone()", "tensor.detach()")
        with self.assertRaises(SystemExit):
            self.patch._apply(
                source, ((self.patch.RUNAI_OLD, self.patch.RUNAI_NEW),),
                "weight_utils.py",
            )


if __name__ == "__main__":
    unittest.main()
