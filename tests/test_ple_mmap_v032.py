#!/usr/bin/env python3
"""PLE mmap overlay re-anchored for v0.32: the real class lives in
models/qwen4_exp/common/ngram_embedding.py (thin nvidia wrapper on top), so
the patcher now consumes two extracted originals and emits two overlay
files. Runs against the extracted origs once present (boot stages them);
the fail-closed check uses a copy so the origs stay pristine."""
import ast
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "patch_ple_mmap_v030.py"
ORIG_DIR = REPO / "patches" / "v030_ple" / "orig"
ORIGS = ("ngram_common.py", "ngram_nvidia.py")


@unittest.skipUnless(
    all((ORIG_DIR / o).is_file() for o in ORIGS),
    "no extracted v0.32 ngram_embedding origs; boot once on the v030 lane "
    "(engine/patches.sh extracts them) or stage patches/v030_ple/orig/ "
    "from the image",
)
class PleMmapV032(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / "orig").mkdir()
        for name in ORIGS:
            shutil.copy(ORIG_DIR / name, self.tmp / "orig" / name)

    def run_patch(self):
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.tmp)],
            capture_output=True, text=True)

    def test_both_outputs_patch_parse_and_carry_mmap_marks(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        common = (self.tmp / "ple_common_v032.py").read_text()
        nvidia = (self.tmp / "ple_nvidia_v032.py").read_text()
        ast.parse(common)
        ast.parse(nvidia)
        self.assertIn("def commit_mmap_table", common)
        self.assertIn("_lookup_ple_rows_from_host_addr_kernel", common)
        self.assertIn("file-backed table", common)
        # unified-memory hygiene set on the VLLM_PLE_MMAP_ADVICE switch; a
        # reused table is deliberately NOT pre-faulted at boot (a full-table
        # populate re-reads ~48 GiB and stalls restarts)
        self.assertIn("os.POSIX_FADV_RANDOM", common)
        self.assertNotIn("MADV_POPULATE_READ", common)
        # staged buffers are pinned CPU tensors even though construction runs
        # inside a CUDA device context (pin_memory on a CUDA tensor raises)
        self.assertGreaterEqual(common.count("pin_memory=True"), 2)
        for alloc in re.finditer(
                r"torch\.empty\([^)]*pin_memory=True[^)]*\)", common, re.S):
            self.assertIn('device="cpu"', alloc.group(0))
        self.assertIn("self._mmap_fd: int | None = None", common)
        self.assertIn("weight, self._mmap_ready, self._mmap_fd = _ple_mmap_open(",
                      common)
        # offset-sorted coalesced warm-gather, mirrored from the patcher
        self.assertIn("def _ple_gather_plan(offsets, row_bytes, page_size=4096):",
                      common)
        self.assertIn("def _ple_mmap_warm(self, ids: torch.Tensor) -> None:",
                      common)
        self.assertIn("self._ple_mmap_warm(gathered_ids)", common)
        # native prefetch machinery kept; its lazy stream/buffer bind to the
        # mmap-safe device selector instead of the (None) UVA view, and the
        # staged path allocates its own pair up front (4 total)
        self.assertEqual(common.count("device=self._lookup_device"), 4)
        self.assertIn("supports_prefetch", common)
        self.assertIn("_mmap_expected_shards", nvidia)

    def test_rerun_on_orig_is_deterministic(self):
        self.assertEqual(self.run_patch().returncode, 0)
        first = (self.tmp / "ple_common_v032.py").read_text()
        self.assertEqual(self.run_patch().returncode, 0)
        self.assertEqual((self.tmp / "ple_common_v032.py").read_text(), first)

    def test_already_patched_input_fails(self):
        self.assertEqual(self.run_patch().returncode, 0)
        shutil.copy(self.tmp / "ple_common_v032.py", self.tmp / "orig" / "ngram_common.py")
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already patched", r.stderr)

    def test_moved_anchor_fails_closed(self):
        common = self.tmp / "orig" / "ngram_common.py"
        text = common.read_text().replace(
            "        self._prefetch_stream: torch.cuda.Stream | None = None\n",
            "        self._prefetch_stream: torch.cuda.Stream | None = None  \n")
        common.write_text(text.replace(
            "        self._row_bytes = self.embedding_dim",
            "        self._row_bytes  = self.embedding_dim"))
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("anchor", r.stderr)


if __name__ == "__main__":
    unittest.main()
