#!/usr/bin/env python3
"""Offline tests for the prefill-only PLE row prefetch in the v0.32 overlay.

The overlay imports torch and vLLM, so the prefetch helper block and the
_ple_prefill_touch method are exec'd straight from the patcher's source. A
small file-backed table -- mapped with the stdlib mmap, the same shape the
overlay's torch.from_file mapping takes -- stands in for the 51 GB PLE table,
and an os proxy records every pread range the readers issue. No GPU, no torch,
no vLLM: the point is to pin *which rows* the prefetch touches and that every
non-prefill or non-mmap path stays a no-op.
"""
import ast
import ctypes
import importlib.util
import inspect
import mmap
import os
import shutil
import tempfile
import textwrap
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
PATCHER = REPO / "patches" / "patch_ple_mmap_v030.py"
COMMON = REPO / "patches" / "v030_ple" / "ple_common_v032.py"

ROW_BYTES = 160  # 16 n-gram heads x 10-byte FP8 rows, packed per token
ROWS = 64
PAGE = 4096


def load_patcher():
    spec = importlib.util.spec_from_file_location("ple_mmap_patcher", PATCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PATCHER_MODULE = load_patcher()


class _CountingOs:
    """os proxy that records every range the prefetch readers issue."""

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

    def warnings(self):
        return [line for level, line in self.lines if level == "warning"]


def helper_namespace(counting=None, forward_context=None, logger=None):
    """Exec the patcher's gather-plan + prefetch helpers into a namespace."""
    source = (
        inspect.getsource(_PATCHER_MODULE._ple_gather_plan)
        + "\n\n"
        + _PATCHER_MODULE.PREFETCH_HELPERS
    )
    namespace = {
        "os": counting or os,
        "mmap": mmap,
        "ctypes": ctypes,
        "threading": threading,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "logger": logger or _Logger(),
        "get_forward_context": forward_context
        or (lambda: SimpleNamespace(attn_metadata=None)),
    }
    exec(compile(source, str(PATCHER), "exec"), namespace)
    return namespace


def touch_method(namespace):
    """Exec _ple_prefill_touch standalone so a fake self can call it."""
    tree = ast.parse(textwrap.dedent(_PATCHER_MODULE.PREFETCH_METHODS))
    exec(
        compile(ast.Module(body=tree.body[:1], type_ignores=[]), str(PATCHER), "exec"),
        namespace,
    )
    return namespace["_ple_prefill_touch"]


class Table:
    """A small file-backed PLE table: file, read fd, and a live mapping."""

    def __init__(self, rows=ROWS, row_bytes=ROW_BYTES):
        self.dir = Path(tempfile.mkdtemp(prefix="ple-prefetch-"))
        self.path = self.dir / "table.bin"
        self.row_bytes = row_bytes
        self.rows = rows
        self.data = bytes(
            (i * 7 + j) % 251 for i in range(rows) for j in range(row_bytes)
        )
        self.path.write_bytes(self.data)
        self.fd = os.open(self.path, os.O_RDWR)
        self.mapping = mmap.mmap(self.fd, len(self.data))

    def addr(self):
        return ctypes.addressof(ctypes.c_char.from_buffer(self.mapping))

    def close(self):
        self.mapping.close()
        os.close(self.fd)
        shutil.rmtree(self.dir, True)


def page_range(start, length):
    return range(start // PAGE, (start + length - 1) // PAGE + 1)


class Rig:
    """One prefetch-enabled embedding stand-in plus its recording namespace."""

    def __init__(self, prefill=1, capturing=False):
        self.counting = _CountingOs()
        self.logger = _Logger()
        self.capturing = capturing

        def forward_context():
            if prefill is None:
                return SimpleNamespace(attn_metadata=None)
            return SimpleNamespace(
                attn_metadata={"layer": SimpleNamespace(num_prefills=prefill)}
            )

        self.ns = helper_namespace(self.counting, forward_context, self.logger)
        self.ns["torch"] = SimpleNamespace(
            cuda=SimpleNamespace(
                is_current_stream_capturing=lambda: self.capturing,
            )
        )
        self.touch = touch_method(self.ns)
        self.ns["_ple_resident_pages"] = lambda addr, length: False

    def layer(self, table, on=True, fd=True, rows=None):
        return SimpleNamespace(
            _prefill_prefetch_on=on,
            _mmap_fd=table.fd if fd else None,
            _mmap_path=str(table.path),
            _row_bytes=table.row_bytes,
            weight=SimpleNamespace(data_ptr=table.addr),
            shard_indices=SimpleNamespace(
                org_vocab_start_index=0,
                org_vocab_end_index=table.rows if rows is None else rows,
            ),
        )

    def run(self, table, ids, on=True, fd=True):
        return self.touch(self.layer(table, on=on, fd=fd), list(ids))

    def pages(self):
        return {
            page
            for start, length in self.counting.reads
            for page in page_range(start, length)
        }


class PrefetchPlan(unittest.TestCase):
    def setUp(self):
        self.table = Table()
        self.addCleanup(self.table.close)

    def test_rows_are_distinct_and_shard_bounded(self):
        ns = helper_namespace()
        ids = [5, 5, 3, -1, 64, 63, 64, 0]
        self.assertEqual(ns["_ple_prefetch_rows"](ids, 0, 64), [0, 3, 5, 63])

    def test_split_ranges_covers_every_range_once(self):
        ns = helper_namespace()
        ranges = [(index * PAGE, PAGE) for index in range(10)]
        parts = ns["_ple_split_ranges"](ranges, 4)
        self.assertEqual(len(parts), 4)
        self.assertEqual([item for part in parts for item in part], ranges)
        self.assertEqual(ns["_ple_split_ranges"](ranges, 0), [ranges])
        self.assertEqual(ns["_ple_split_ranges"]([], 8), [])

    def test_plan_reads_the_referenced_rows_through_the_pool(self):
        rig = Rig()
        rig.ns["_ple_resident_pages"] = lambda addr, length: False
        ids = [0, 5, 30, 31, 32, 63, 5]
        self.assertTrue(rig.run(self.table, ids))
        rows = sorted({0, 5, 30, 31, 32, 63})
        referenced = {
            page
            for row in rows
            for page in page_range(row * ROW_BYTES, ROW_BYTES)
        }
        self.assertEqual(rig.pages(), referenced)
        self.assertEqual(rig.logger.warnings(), [])

    def test_resident_rows_are_skipped(self):
        rig = Rig()
        rig.ns["_ple_resident_pages"] = lambda addr, length: True
        self.assertTrue(rig.run(self.table, [0, 5, 30, 63]))
        self.assertEqual(rig.counting.reads, [])

    def test_resident_probe_sees_a_page_just_read(self):
        ns = helper_namespace()
        addr = self.table.addr()
        self.assertTrue(
            ns["_ple_resident_pages"](addr + 5 * ROW_BYTES, ROW_BYTES)
        )
        # An unmapped address is the probe's own failure mode, read as cold.
        self.assertFalse(ns["_ple_resident_pages"](0, ROW_BYTES))


class PrefillGate(unittest.TestCase):
    def setUp(self):
        self.table = Table()
        self.addCleanup(self.table.close)

    def test_knob_defaults_off(self):
        ns = helper_namespace()
        saved = os.environ.pop("VLLM_PLE_PREFILL_PREFETCH", None)
        try:
            self.assertFalse(ns["_ple_prefill_prefetch_enabled"]())
            for value in ("1", "true", "TRUE", "yes", "on"):
                os.environ["VLLM_PLE_PREFILL_PREFETCH"] = value
                self.assertTrue(ns["_ple_prefill_prefetch_enabled"](), value)
            for value in ("", "0", "false", "no", "off"):
                os.environ["VLLM_PLE_PREFILL_PREFETCH"] = value
                self.assertFalse(ns["_ple_prefill_prefetch_enabled"](), value)
        finally:
            os.environ.pop("VLLM_PLE_PREFILL_PREFETCH", None)
            if saved is not None:
                os.environ["VLLM_PLE_PREFILL_PREFETCH"] = saved

    def test_threads_are_clamped_to_the_measured_range(self):
        ns = helper_namespace()
        saved = os.environ.pop("VLLM_PLE_PREFETCH_THREADS", None)
        try:
            self.assertEqual(ns["_ple_prefetch_threads"](), 16)
            for value, expected in (
                ("1", 1), ("8", 8), ("32", 32), ("64", 32), ("0", 1),
                ("", 16), ("many", 16),
            ):
                os.environ["VLLM_PLE_PREFETCH_THREADS"] = value
                self.assertEqual(ns["_ple_prefetch_threads"](), expected, value)
        finally:
            os.environ.pop("VLLM_PLE_PREFETCH_THREADS", None)
            if saved is not None:
                os.environ["VLLM_PLE_PREFETCH_THREADS"] = saved

    def test_pool_honours_the_thread_knob(self):
        ns = helper_namespace()
        os.environ["VLLM_PLE_PREFETCH_THREADS"] = "8"
        try:
            pool, threads = ns["_ple_prefetch_pool"]()
            self.assertEqual(threads, 8)
            self.assertEqual(pool._max_workers, 8)
            pool.shutdown(wait=False)
        finally:
            os.environ.pop("VLLM_PLE_PREFETCH_THREADS", None)

    def test_prefill_step_reads_the_attn_metadata(self):
        cases = (
            (None, False),
            ({"layer": SimpleNamespace(num_prefills=2)}, True),
            ({"layer": SimpleNamespace(num_prefills=0, num_decodes=4)}, False),
            ({"layer": SimpleNamespace(num_decodes=4)}, False),
            ([{"a": SimpleNamespace(num_prefills=0)},
              {"b": SimpleNamespace(num_prefills=1)}], True),
            ([], False),
            ("not metadata", False),
        )
        for attn_metadata, expected in cases:
            ns = helper_namespace(
                forward_context=lambda m=attn_metadata: SimpleNamespace(
                    attn_metadata=m
                )
            )
            self.assertEqual(ns["_ple_prefill_step"](), expected, attn_metadata)

    def test_unreadable_forward_context_reads_as_decode(self):
        def boom():
            raise AssertionError("no forward context")

        ns = helper_namespace(forward_context=boom)
        self.assertFalse(ns["_ple_prefill_step"]())

    def test_knob_off_is_a_no_op(self):
        rig = Rig()
        self.assertFalse(rig.run(self.table, [0, 5, 30, 63], on=False))
        self.assertEqual(rig.counting.reads, [])
        self.assertEqual(rig.logger.lines, [])

    def test_capture_is_a_no_op(self):
        rig = Rig(capturing=True)
        self.assertFalse(rig.run(self.table, [0, 5, 30, 63]))
        self.assertEqual(rig.counting.reads, [])
        self.assertEqual(rig.logger.lines, [])

    def test_decode_batch_is_a_no_op(self):
        rig = Rig(prefill=0)
        self.assertFalse(rig.run(self.table, [0, 5, 30, 63]))
        self.assertEqual(rig.counting.reads, [])
        self.assertEqual(rig.logger.lines, [])

    def test_table_without_a_mapping_falls_back_silently(self):
        rig = Rig()
        self.assertFalse(rig.run(self.table, [0, 5, 30, 63], fd=False))
        self.assertEqual(rig.counting.reads, [])
        self.assertEqual(rig.logger.lines, [])

    def test_prefill_engagement_logs_once(self):
        rig = Rig()
        # Rows 0 and 30 are more than a page apart, so two ranges.
        self.assertTrue(rig.run(self.table, [0, 30]))
        self.assertTrue(rig.run(self.table, [1, 31]))
        infos = [line for level, line in rig.logger.lines if level == "info"]
        self.assertEqual(len(infos), 1)
        self.assertIn("prefill prefetch touched 2 rows in 2 ranges", infos[0])
        self.assertIn(str(self.table.path), infos[0])

    def test_reader_failure_warns_once_and_falls_back(self):
        rig = Rig()
        good = rig.layer(self.table)
        good._mmap_fd = 1 << 30  # not an open fd: preadv fails with EBADF
        self.assertFalse(rig.touch(good, [0, 5]))
        self.assertFalse(rig.touch(good, [0, 5]))
        warnings = rig.logger.warnings()
        self.assertEqual(len(warnings), 1)
        self.assertIn("prefill prefetch fell back", warnings[0])


def _class_functions(source):
    """Map method name -> FunctionDef for a class-body source block."""
    body = textwrap.indent(textwrap.dedent(source), "    ")
    found = {}
    for node in ast.walk(ast.parse("class _Block:\n" + body)):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    found[item.name] = item
    return found


class Wiring(unittest.TestCase):
    """The call must sit in the prefill warm path, ahead of the GPU gather.

    The patcher's own injected blocks are always checked; the boot-generated
    overlay adds the checks that need the vendored file around them.
    """

    def warm(self):
        return _class_functions(_PATCHER_MODULE.STAGED_METHODS)["_ple_mmap_warm"]

    def touch(self):
        return _class_functions(_PATCHER_MODULE.PREFETCH_METHODS)[
            "_ple_prefill_touch"
        ]

    def init(self):
        return _class_functions(_PATCHER_MODULE.STAGED_INIT_TAIL)

    def test_touch_precedes_the_serial_warm_and_follows_the_gates(self):
        warm = self.warm()
        capture = min(
            node.lineno
            for node in ast.walk(warm)
            if isinstance(node, ast.If)
            and "is_current_stream_capturing" in ast.unparse(node.test)
        )
        staged = min(
            node.lineno
            for node in ast.walk(warm)
            if isinstance(node, ast.If)
            and "self._staged_lookup" in ast.unparse(node.test)
        )
        touch = min(
            node.lineno
            for node in ast.walk(warm)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "self._ple_prefill_touch"
        )
        serial = min(
            node.lineno
            for node in ast.walk(warm)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "os.pread"
        )
        self.assertLess(capture, touch)
        self.assertLess(staged, touch)
        self.assertLess(touch, serial)
        self.assertIn("if self._ple_prefill_touch(values):", ast.unparse(warm))

    def test_only_the_warm_path_calls_the_touch(self):
        callers = [
            name
            for name, node in _class_functions(
                _PATCHER_MODULE.STAGED_METHODS + _PATCHER_MODULE.PREFETCH_METHODS
            ).items()
            if any(
                isinstance(sub, ast.Call)
                and ast.unparse(sub.func) == "self._ple_prefill_touch"
                for sub in ast.walk(node)
            )
        ]
        self.assertEqual(callers, ["_ple_mmap_warm"])

    def test_flag_is_armed_from_the_mmap_path(self):
        tree = ast.parse(textwrap.dedent(_PATCHER_MODULE.STAGED_INIT_TAIL))
        assignments = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "_prefill_prefetch_on"
                for target in node.targets
            )
        ]
        self.assertEqual(len(assignments), 1)
        self.assertIn("_mmap_path is not None", assignments[0])
        self.assertIn("_ple_prefill_prefetch_enabled()", assignments[0])

    def test_warm_hook_rides_the_advice_switch(self):
        # The prefetch hangs off the eager warm hook, which start_prefetch
        # arms on the VLLM_PLE_MMAP_ADVICE set; that coupling is documented
        # and must not silently resolve to an always-on hook.
        calls = [
            new
            for _old, new in _PATCHER_MODULE.EDITS_COMMON
            if "self._ple_mmap_warm(gathered_ids)" in new
        ]
        self.assertEqual(len(calls), 1)
        self.assertIn("_ple_advice_value()", calls[0])

    def test_touch_method_adds_no_stream_no_pinned_no_device_sync(self):
        source = _PATCHER_MODULE.PREFETCH_METHODS + _PATCHER_MODULE.PREFETCH_HELPERS
        for banned in (
            "Stream(",
            "pin_memory",
            "synchronize",
            "record_stream",
            ".cuda(",
            'to("cuda")',
        ):
            self.assertNotIn(banned, source)

    def test_generated_overlay_keeps_lookup_untouched(self):
        if not COMMON.is_file():
            self.skipTest("no boot-generated overlay on this tree")
        text = COMMON.read_text()
        for name, node in _class_functions(text).items():
            if name == "_lookup":
                self.assertNotIn("_ple_prefill_touch", ast.unparse(node))
        self.assertIn("self._ple_prefill_touch(values)", text)


if __name__ == "__main__":
    unittest.main()
