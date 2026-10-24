#!/usr/bin/env python3
"""Offline tests for the staged PLE lookup in the v0.32 mmap overlay.

The overlay imports torch and vLLM, so the pure pieces are loaded from the
generated source with ast/exec instead of imported: the host gather runs
against a synthetic table file, and the replay hook installs into a fake
CudaGraphManager. No GPU, no torch, no vLLM.
"""
import ast
import functools
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
COMMON = REPO / "patches" / "v030_ple" / "ple_common_v032.py"
NVIDIA = REPO / "patches" / "v030_ple" / "ple_nvidia_v032.py"
PATCHER = REPO / "patches" / "patch_ple_mmap_v030.py"
ORIG_DIR = REPO / "patches" / "v030_ple" / "orig"
ORIGS = ("ngram_common.py", "ngram_nvidia.py")

ROW_BYTES = 160  # 16 n-gram heads x 10-byte FP8 rows, packed per token
ROWS = 64

_STATE = {}


def _generate_overlays():
    tmp = Path(tempfile.mkdtemp(prefix="ple-staged-"))
    (tmp / "orig").mkdir()
    for name in ORIGS:
        shutil.copy(ORIG_DIR / name, tmp / "orig" / name)
    run = subprocess.run(
        [sys.executable, str(PATCHER), str(tmp)],
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        shutil.rmtree(tmp, True)
        raise AssertionError(f"patcher failed: {run.stderr}")
    return tmp


def setUpModule():
    """Test the patcher's current output, not a stale boot artifact."""
    if all((ORIG_DIR / name).is_file() for name in ORIGS):
        tmp = _generate_overlays()
        _STATE["tmp"] = tmp
        _STATE["common"] = tmp / "ple_common_v032.py"
        _STATE["nvidia"] = tmp / "ple_nvidia_v032.py"
    elif COMMON.is_file() and NVIDIA.is_file():
        _STATE["common"] = COMMON
        _STATE["nvidia"] = NVIDIA
    else:
        raise unittest.SkipTest(
            "no v0.32 ngram_embedding origs; boot once on the v030 lane"
        )


def tearDownModule():
    tmp = _STATE.pop("tmp", None)
    if tmp is not None:
        shutil.rmtree(tmp, True)


def common_path():
    return _STATE["common"]


def nvidia_path():
    return _STATE["nvidia"]

COMMON_NAMES = {
    "_ple_gather_plan",
    "_ple_staged_lookup_enabled",
    "_ple_byte_view",
    "_ple_scatter_read",
    "_ple_stage_rows",
    "_PLE_STAGE_READ_MAX",
    "_PLE_LOOKUP_PATHS_LOGGED",
}
NVIDIA_NAMES = {
    "_install_staged_lookup_hook",
    "_register_staged_layer",
    "_STAGED_LAYERS",
    "_STAGED_HOOK_INSTALLED",
}


class _CountingOs:
    """os proxy that records every preadv range the gather issues."""

    def __init__(self):
        self.reads = []

    def __getattr__(self, name):
        return getattr(os, name)

    def preadv(self, fd, buffers, offset):
        self.reads.append((offset, len(buffers[0])))
        return os.preadv(fd, buffers, offset)


class _Logger:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(("info", msg % args if args else msg))

    def warning(self, msg, *args):
        self.lines.append(("warning", msg % args if args else msg))


def execute(path, names, extra=None):
    """Exec the named top-level defs/assignments from a generated overlay."""
    tree = ast.parse(path.read_text())
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            keep.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if any(
                isinstance(t, ast.Name) and t.id in names for t in targets
            ):
                keep.append(node)
    namespace = dict(extra or {})
    exec(
        compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


def find_method(tree, class_name, method_name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found")


def calls_named(node, name):
    return [
        sub
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Subscript)
        and isinstance(sub.func.value, ast.Name)
        and sub.func.value.id == name
    ]


class StagedHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def load(self, with_os=None):
        return execute(common_path(), COMMON_NAMES, {"os": with_os or os})

    def table(self, rows=ROWS, row_bytes=ROW_BYTES):
        data = bytes(
            (i * 7 + j) % 251 for i in range(rows) for j in range(row_bytes)
        )
        path = self.tmp / "table.bin"
        path.write_bytes(data)
        return path, data

    def naive_rows(self, data, ids, row_bytes):
        return b"".join(
            data[i * row_bytes : (i + 1) * row_bytes] for i in ids
        )

    def test_gather_matches_direct_row_reads(self):
        ns = self.load()
        path, data = self.table()
        fd = os.open(path, os.O_RDONLY)
        self.addCleanup(os.close, fd)
        ids = [5, 0, 63, 5, 17, 31, 32]
        stage = bytearray(len(ids) * ROW_BYTES)
        scratch = bytearray(1 << 20)
        got = ns["_ple_stage_rows"](
            fd, ids, ROW_BYTES, 0, ROWS, stage, scratch
        )
        self.assertEqual(got, len(ids))
        self.assertEqual(bytes(stage), self.naive_rows(data, ids, ROW_BYTES))

    def test_out_of_shard_rows_are_zero_filled(self):
        ns = self.load()
        path, data = self.table()
        fd = os.open(path, os.O_RDONLY)
        self.addCleanup(os.close, fd)
        lo, hi = 4, 60
        ids = [3, 4, 60, 59, -1, 10**9, 10]
        stage = bytearray(b"\xaa" * (len(ids) * ROW_BYTES))
        ns["_ple_stage_rows"](
            fd, ids, ROW_BYTES, lo, hi, stage, bytearray(1 << 20)
        )
        zero = bytes(ROW_BYTES)
        for slot, row_id in enumerate(ids):
            chunk = bytes(stage[slot * ROW_BYTES : (slot + 1) * ROW_BYTES])
            if lo <= row_id < hi:
                local = row_id - lo
                self.assertEqual(
                    chunk, data[local * ROW_BYTES : (local + 1) * ROW_BYTES]
                )
            else:
                self.assertEqual(chunk, zero)

    def test_coalesced_plan_equals_naive_per_row_reads(self):
        counter = _CountingOs()
        ns = self.load(counter)
        path, data = self.table()
        fd = os.open(path, os.O_RDONLY)
        self.addCleanup(os.close, fd)
        ids = list(range(37))
        stage = bytearray(len(ids) * ROW_BYTES)
        ns["_ple_stage_rows"](fd, ids, ROW_BYTES, 0, ROWS, stage, bytearray(1 << 20))
        self.assertEqual(bytes(stage), self.naive_rows(data, ids, ROW_BYTES))
        # Contiguous rows fold into one pread, not len(ids).
        self.assertEqual(len(counter.reads), 1)
        self.assertEqual(counter.reads[0][0], 0)

    def test_unsorted_ids_keep_one_coalesced_read(self):
        counter = _CountingOs()
        ns = self.load(counter)
        path, data = self.table()
        fd = os.open(path, os.O_RDONLY)
        self.addCleanup(os.close, fd)
        ids = [36, 3, 20, 1, 35, 2]
        stage = bytearray(len(ids) * ROW_BYTES)
        ns["_ple_stage_rows"](fd, ids, ROW_BYTES, 0, ROWS, stage, bytearray(1 << 20))
        self.assertEqual(bytes(stage), self.naive_rows(data, ids, ROW_BYTES))
        self.assertEqual(len(counter.reads), 1)

    def test_scratch_bound_splits_reads_without_changing_bytes(self):
        counter = _CountingOs()
        ns = self.load(counter)
        path, data = self.table()
        fd = os.open(path, os.O_RDONLY)
        self.addCleanup(os.close, fd)
        ids = list(range(37))
        stage = bytearray(len(ids) * ROW_BYTES)
        ns["_ple_stage_rows"](
            fd, ids, ROW_BYTES, 0, ROWS, stage, bytearray(3 * ROW_BYTES)
        )
        self.assertEqual(bytes(stage), self.naive_rows(data, ids, ROW_BYTES))
        self.assertGreater(len(counter.reads), 1)
        for offset, length in counter.reads:
            self.assertLessEqual(length, 3 * ROW_BYTES)
            self.assertGreaterEqual(offset, 0)

    def test_read_bound_below_one_row_still_reads_whole_rows(self):
        ns = self.load()
        path, data = self.table()
        fd = os.open(path, os.O_RDONLY)
        self.addCleanup(os.close, fd)
        ids = [9, 4, 20]
        stage = bytearray(len(ids) * ROW_BYTES)
        ns["_ple_stage_rows"](
            fd, ids, ROW_BYTES, 0, ROWS, stage, bytearray(1 << 16), read_max=1
        )
        self.assertEqual(bytes(stage), self.naive_rows(data, ids, ROW_BYTES))

    def test_short_read_fails_closed(self):
        ns = self.load()
        path, _data = self.table()
        fd = os.open(path, os.O_RDONLY)
        self.addCleanup(os.close, fd)
        with self.assertRaises(OSError):
            ns["_ple_stage_rows"](
                fd, [ROWS + 5], ROW_BYTES, 0, ROWS + 100, bytearray(ROW_BYTES),
                bytearray(1 << 16),
            )

    def test_gate_defaults_off(self):
        saved = os.environ.pop("VLLM_PLE_STAGED_LOOKUP", None)
        try:
            ns = self.load()
            self.assertFalse(ns["_ple_staged_lookup_enabled"]())
            for value in ("1", "true", "TRUE", "yes", "on"):
                os.environ["VLLM_PLE_STAGED_LOOKUP"] = value
                self.assertTrue(ns["_ple_staged_lookup_enabled"](), value)
            for value in ("", "0", "false", "no", "off"):
                os.environ["VLLM_PLE_STAGED_LOOKUP"] = value
                self.assertFalse(ns["_ple_staged_lookup_enabled"](), value)
        finally:
            os.environ.pop("VLLM_PLE_STAGED_LOOKUP", None)
            if saved is not None:
                os.environ["VLLM_PLE_STAGED_LOOKUP"] = saved


class ReplayHook(unittest.TestCase):
    MODULES = (
        "vllm",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu",
        "vllm.v1.worker.gpu.cudagraph_utils",
    )

    def install_fake(self, manager_cls, target_cls=None):
        saved = {name: sys.modules.get(name) for name in self.MODULES}
        self.addCleanup(self.restore, saved)
        for name in self.MODULES[:-1]:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        module = types.ModuleType(self.MODULES[-1])
        module.CudaGraphManager = manager_cls
        if target_cls is not None:
            module.ModelCudaGraphManager = target_cls
        sys.modules[self.MODULES[-1]] = module

    def restore(self, saved):
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def load(self):
        common = execute(common_path(), COMMON_NAMES, {"os": os})
        namespace = execute(
            nvidia_path(),
            NVIDIA_NAMES,
            {
                "functools": functools,
                "weakref": weakref,
                "logger": _Logger(),
                "_ple_staged_lookup_enabled": common[
                    "_ple_staged_lookup_enabled"
                ],
            },
        )
        return namespace

    def test_install_is_idempotent_and_stages_before_original(self):
        events = []

        class Manager:
            def run_fullgraph(self, desc):
                events.append(("replay", desc.num_tokens))
                return "replay"

        class Layer:
            def stage_registered(self, num_tokens):
                events.append(("stage", num_tokens))

        self.install_fake(Manager)
        os.environ["VLLM_PLE_STAGED_LOOKUP"] = "1"
        try:
            ns = self.load()
            self.assertTrue(ns["_install_staged_lookup_hook"]())
            wrapped = Manager.run_fullgraph
            self.assertTrue(ns["_install_staged_lookup_hook"]())
            self.assertIs(Manager.run_fullgraph, wrapped)

            layer = Layer()
            ns["_STAGED_LAYERS"].append(weakref.ref(layer))
            self.assertEqual(
                Manager().run_fullgraph(SimpleNamespace(num_tokens=12)), "replay"
            )
            self.assertEqual(events, [("stage", 12), ("replay", 12)])
        finally:
            os.environ.pop("VLLM_PLE_STAGED_LOOKUP", None)

    def test_speculator_manager_is_not_staged(self):
        events = []

        class Manager:
            def run_fullgraph(self, desc):
                events.append(("replay", type(self).__name__, desc.num_tokens))
                return "replay"

        class Target(Manager):
            pass

        class Drafter(Manager):
            pass

        class Layer:
            def stage_registered(self, num_tokens):
                events.append(("stage", num_tokens))

        self.install_fake(Manager, Target)
        os.environ["VLLM_PLE_STAGED_LOOKUP"] = "1"
        try:
            ns = self.load()
            self.assertTrue(ns["_install_staged_lookup_hook"]())
            layer = Layer()
            ns["_STAGED_LAYERS"].append(weakref.ref(layer))
            desc = SimpleNamespace(num_tokens=8)
            self.assertEqual(Target().run_fullgraph(desc), "replay")
            self.assertEqual(Drafter().run_fullgraph(desc), "replay")
            self.assertEqual(
                events,
                [("stage", 8), ("replay", "Target", 8), ("replay", "Drafter", 8)],
            )
        finally:
            os.environ.pop("VLLM_PLE_STAGED_LOOKUP", None)

    def test_gate_off_leaves_the_class_untouched(self):
        class Manager:
            def run_fullgraph(self, desc):
                return "stock"

        self.install_fake(Manager)
        saved = os.environ.pop("VLLM_PLE_STAGED_LOOKUP", None)
        try:
            ns = self.load()
            original = Manager.run_fullgraph
            self.assertFalse(ns["_install_staged_lookup_hook"]())
            self.assertIs(Manager.run_fullgraph, original)
        finally:
            if saved is not None:
                os.environ["VLLM_PLE_STAGED_LOOKUP"] = saved

    def test_missing_anchor_fails_closed(self):
        class Manager:
            pass

        self.install_fake(Manager)
        os.environ["VLLM_PLE_STAGED_LOOKUP"] = "1"
        try:
            ns = self.load()
            self.assertFalse(ns["_install_staged_lookup_hook"]())
            self.assertFalse(ns["_STAGED_HOOK_INSTALLED"])
        finally:
            os.environ.pop("VLLM_PLE_STAGED_LOOKUP", None)


class LookupPathPin(unittest.TestCase):
    """The staged branch must sit in front of an otherwise intact kernel path."""

    def test_default_off_keeps_the_host_addr_kernel(self):
        tree = ast.parse(common_path().read_text())
        lookup = find_method(tree, "Qwen4ExpPLEPinnedHostEmbedding", "_lookup")
        kernel_calls = calls_named(lookup, "_lookup_ple_rows_from_host_addr_kernel")
        self.assertEqual(len(kernel_calls), 1)

        staged = None
        uva_branch = None
        for node in ast.walk(lookup):
            if not isinstance(node, ast.If):
                continue
            test = ast.unparse(node.test)
            if test == "self._staged_lookup":
                staged = node
            elif test == "self._uva_weight is None":
                uva_branch = node
        self.assertIsNotNone(staged)
        self.assertIsNotNone(uva_branch)
        self.assertTrue(any(node is staged for node in ast.walk(uva_branch)))
        self.assertLess(staged.lineno, kernel_calls[0].lineno)

        body = ast.unparse(staged)
        self.assertIn("self._upload_staged(", body)
        self.assertIn("return output", body)
        # The kernel call stays under the same `_uva_weight is None` branch.
        self.assertGreater(
            kernel_calls[0].col_offset, uva_branch.col_offset
        )

    def test_lookup_kernel_is_only_reached_with_the_gate_off(self):
        source = common_path().read_text()
        # Exactly one host-address launch site and one staged upload site.
        self.assertEqual(source.count("_lookup_ple_rows_from_host_addr_kernel[("), 1)
        self.assertEqual(source.count("self._upload_staged(flat_ids.numel(), output)"), 1)

    def test_capture_guard_precedes_the_host_read(self):
        tree = ast.parse(common_path().read_text())
        warm = find_method(
            tree, "Qwen4ExpPLEPinnedHostEmbedding", "_ple_mmap_warm"
        )
        guards = [
            node
            for node in ast.walk(warm)
            if isinstance(node, ast.If)
            and "is_current_stream_capturing" in ast.unparse(node.test)
        ]
        self.assertEqual(len(guards), 1)
        host_reads = [
            node
            for node in ast.walk(warm)
            if isinstance(node, ast.Call)
            and ".to(" in ast.unparse(node.func)
        ]
        self.assertTrue(host_reads)
        self.assertLess(guards[0].lineno, min(n.lineno for n in host_reads))


class PrefetchPathPin(unittest.TestCase):
    """The capture path must not do host work; replay must recompute ids."""

    def test_capture_branch_returns_before_ids_work(self):
        tree = ast.parse(nvidia_path().read_text())
        start = find_method(tree, "Qwen4ExpNGramEmbedding", "start_prefetch")
        capture = [
            node
            for node in ast.walk(start)
            if isinstance(node, ast.If)
            and "is_current_stream_capturing" in ast.unparse(node.test)
        ]
        self.assertEqual(len(capture), 1)
        ids_calls = [
            node
            for node in ast.walk(start)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "compute_ngram_ids"
        ]
        self.assertTrue(ids_calls)
        self.assertLess(capture[0].lineno, min(c.lineno for c in ids_calls))
        self.assertIn("return", ast.unparse(capture[0]))
        self.assertIn("disable_staged_lookup", ast.unparse(capture[0]))

    def test_stage_registered_recomputes_from_captured_views(self):
        tree = ast.parse(nvidia_path().read_text())
        stage = find_method(tree, "Qwen4ExpNGramEmbedding", "stage_registered")
        text = ast.unparse(stage)
        self.assertIn("input_ids[:num_tokens]", text)
        self.assertIn("self.compute_ngram_ids", text)
        self.assertIn("embedding.stage_rows_to_device(ids)", text)
        self.assertIn("num_tokens not in self._stage_sizes", text)
        self.assertIn("raise RuntimeError", text)
        ids_line = min(
            node.lineno
            for node in ast.walk(stage)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "compute_ngram_ids"
        )
        upload_line = min(
            node.lineno
            for node in ast.walk(stage)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "stage_rows_to_device"
        )
        self.assertLess(ids_line, upload_line)


class GeneratedOverlayDrift(unittest.TestCase):
    """The boot regenerates the overlays; a stale on-disk copy must not ship."""

    def test_on_disk_overlays_match_the_patcher(self):
        if not (COMMON.is_file() and NVIDIA.is_file()):
            self.skipTest("no boot-generated overlays on this tree")
        for name, on_disk in (
            ("ple_common_v032.py", COMMON),
            ("ple_nvidia_v032.py", NVIDIA),
        ):
            self.assertEqual(
                (common_path().parent / name).read_text(),
                on_disk.read_text(),
                f"{name} is stale: rerun {PATCHER.name}",
            )


if __name__ == "__main__":
    unittest.main()
