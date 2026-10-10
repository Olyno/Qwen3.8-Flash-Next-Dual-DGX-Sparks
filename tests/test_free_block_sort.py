import ast
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "free_block_sort" / "apply_patch.py"

FAKE_ORIG = '''\
class FreeKVCacheBlockQueue:
    def append_n(self, blocks):
        """Put a list of blocks back into the free list"""
        if len(blocks) == 0:
            return

        last_block = self.fake_free_list_tail.prev_free_block
        assert last_block is not None, (
            "prev_free_block of fake_free_list_tail should always exist"
        )
        # Add inter-connections between consecutive blocks
        for block in blocks:
            block.prev_free_block = last_block
            last_block.next_free_block = block
            last_block = block

        # Connect the last block of <blocks> to the fake tail
        last_block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = last_block

        self.num_free_blocks += len(blocks)
'''


def _run(orig_dir, out_dir):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(orig_dir), str(out_dir)],
        capture_output=True, text=True)


def _stage(src=FAKE_ORIG):
    tmp = Path(tempfile.mkdtemp())
    (tmp / "kv_cache_utils.py").write_text(src)
    return tmp


def _patched_append_ns(src):
    """exec the patched append_n and return (callable, namespace)."""
    m = re.search(r"    def append_n\(self.*?\n(?=    def |\Z)", src, re.S)
    assert m, "append_n not found in patched source"
    ns = {}
    exec(textwrap.dedent(m.group(0)), ns)
    return ns["append_n"]


def _walk_ids_tail_first(self):
    ids = []
    block = self.fake_free_list_tail.prev_free_block
    while block is not self.head_sentinel:
        ids.append(block.block_id)
        block = block.prev_free_block
    return ids


class AppendNSort(unittest.TestCase):
    def test_freed_blocks_come_out_block_id_ordered(self):
        tmp = _stage()
        try:
            r = _run(tmp, tmp / "out")
            self.assertEqual(r.returncode, 0, r.stderr)
            src = (tmp / "out" / "kv_cache_utils_v030.py").read_text()
            ast.parse(src)
            append_n = _patched_append_ns(src)
            self_ns = SimpleNamespace()
            self_ns.head_sentinel = SimpleNamespace(block_id=-1)
            self_ns.fake_free_list_tail = SimpleNamespace(
                prev_free_block=self_ns.head_sentinel)
            self_ns.num_free_blocks = 0
            blocks = [SimpleNamespace(block_id=b) for b in (7, 2, 9, 2 + 1, 0)]
            append_n(self_ns, blocks)
            # Tail-first walk of an ascending insertion reads descending.
            self.assertEqual(_walk_ids_tail_first(self_ns), [9, 7, 3, 2, 0])
            self.assertEqual(self_ns.num_free_blocks, 5)
        finally:
            shutil.rmtree(tmp)

    def test_empty_batch_is_a_noop(self):
        append_n = _patched_append_ns(
            FAKE_ORIG.replace(
                "        last_block = self.fake_free_list_tail.prev_free_block",
                "        blocks.sort(key=lambda x: x.block_id)\n"
                "        last_block = self.fake_free_list_tail.prev_free_block"))
        self_ns = SimpleNamespace(num_free_blocks=0)
        append_n(self_ns, [])
        self.assertEqual(self_ns.num_free_blocks, 0)


class PatchAnchors(unittest.TestCase):
    def test_missing_orig_fails(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            r = _run(tmp, tmp)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("missing", r.stderr)
            self.assertIn("kv_cache_utils.py", r.stderr)
        finally:
            shutil.rmtree(tmp)

    def test_applies_to_fake_orig(self):
        tmp = _stage()
        out = tmp / "out"
        try:
            r = _run(tmp, out)
            self.assertEqual(r.returncode, 0, r.stderr)
            src = (out / "kv_cache_utils_v030.py").read_text()
            ast.parse(src)
            self.assertEqual(src.count("blocks.sort(key=lambda x: x.block_id)"), 1)
        finally:
            shutil.rmtree(tmp)

    def test_already_patched_fails(self):
        tmp = _stage()
        try:
            self.assertEqual(_run(tmp, tmp / "out").returncode, 0)
            repatched = _stage((tmp / "out" / "kv_cache_utils_v030.py").read_text())
            try:
                r = _run(repatched, tmp / "out2")
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("already patched", r.stderr)
            finally:
                shutil.rmtree(repatched)
        finally:
            shutil.rmtree(tmp)

    def test_anchor_drift_fails_closed(self):
        drifted = FAKE_ORIG.replace(
            "        last_block = self.fake_free_list_tail.prev_free_block",
            "        last_block = self.fake_free_list_tail.prev")
        tmp = _stage(drifted)
        out = tmp / "out"
        try:
            r = _run(tmp, out)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("anchor", r.stderr)
            self.assertFalse((out / "kv_cache_utils_v030.py").exists())
        finally:
            shutil.rmtree(tmp)

    def test_patches_real_v030_source(self):
        # orig extracted from the v0.30.0 image at boot; skipped until then.
        orig_dir = REPO / "patches" / "free_block_sort" / "orig"
        if not (orig_dir / "kv_cache_utils.py").is_file():
            self.skipTest("no extracted v0.30 kv_cache_utils.py; boot once with free_block_sort")
        tmp = Path(tempfile.mkdtemp())
        try:
            r = _run(orig_dir, tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            ast.parse((tmp / "kv_cache_utils_v030.py").read_text())
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
