#!/usr/bin/env python3
"""Reliability batch assertions:
  * scripts/memwatch.sh gained a PSI memory-full trigger (debounced) without
    losing the MemAvailable/MemFree triggers;
  * engine/mem-gate.sh gates on min(MemAvailable, cgroup v2 headroom) and
    prints the honest "memory guard: N GiB (source: ...)" line — checked
    behaviorally against fake meminfo/cgroup files;
  * engine/launch.sh fails a launch whose log shows no PLE table engagement
    when the recipe asked for PLE offload;
  * none of the touched files leak the banned host token."""
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


class TestMemwatchPsi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = (REPO / "scripts" / "memwatch.sh").read_text()

    def test_psi_file_and_avg10_parsed(self):
        self.assertIn("/proc/pressure/memory", self.src)
        self.assertIn("avg10", self.src)

    def test_psi_trigger_debounced_on_consec(self):
        self.assertIn("MEMWATCH_PSI_FULL", self.src)
        self.assertRegex(self.src, r"below_psi >= CONSEC")

    def test_psi_stop_uses_existing_action(self):
        self.assertRegex(self.src, r'stop_container "PSI memory full')

    def test_memavail_and_memfree_triggers_intact(self):
        for token in ("MEMWATCH_MIN_GIB", "MEMWATCH_MIN_FREE_GIB",
                      "below_avail >= CONSEC", "below_free >= CONSEC"):
            self.assertIn(token, self.src)


class TestCgroupGuard(unittest.TestCase):
    GIB = 1073741824

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        self.meminfo = self.tmp / "meminfo"
        # ~100 GiB MemAvailable
        self.meminfo.write_text("MemTotal: 127000000 kB\nMemAvailable:   104857600 kB\n")
        self.cgroup = self.tmp / "cg"
        self.cgroup.mkdir()

    def run_gate(self, min_gib, max_text, current, inactive_file, timeout=8):
        (self.cgroup / "memory.max").write_text(max_text)
        (self.cgroup / "memory.current").write_text(str(current))
        (self.cgroup / "memory.stat").write_text(
            f"anon 123\ninactive_file {inactive_file}\nfile_dirty 0\n")
        env = dict(os.environ,
                   LAUNCH_MEMINFO=str(self.meminfo),
                   LAUNCH_CGROUP_DIR=str(self.cgroup),
                   LAUNCH_MIN_MEM_AVAIL_GIB=str(min_gib),
                   LAUNCH_MEM_POLL_S="1",
                   LAUNCH_MEM_WAIT_TIMEOUT_S="2")
        return subprocess.run(["bash", str(REPO / "engine" / "mem-gate.sh")],
                              capture_output=True, text=True, env=env,
                              cwd=REPO, timeout=timeout + 5)

    def test_cgroup_headroom_becomes_the_guard(self):
        # max 60 GiB, current 20 GiB, 3 GiB reclaimable -> 43 GiB headroom.
        out = self.run_gate(40, str(60 * self.GIB), 20 * self.GIB, 3 * self.GIB)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertRegex(out.stdout, r"memory guard: 43 GiB \(source: min\)")

    def test_cgroup_below_threshold_hard_fails(self):
        # 43 GiB effective guard < 50 GiB threshold: hard failure even though
        # MemAvailable alone (100 GiB) would have passed.
        out = self.run_gate(50, str(60 * self.GIB), 20 * self.GIB, 3 * self.GIB)
        self.assertNotEqual(out.returncode, 0)

    def test_unlimited_cgroup_falls_back_to_meminfo(self):
        out = self.run_gate(90, "max", 20 * self.GIB, 0)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertRegex(out.stdout, r"memory guard: 100 GiB \(source: meminfo\)")

    def test_missing_memory_max_falls_back_to_meminfo(self):
        (self.cgroup / "memory.max").write_text("max")  # baseline, then remove
        (self.cgroup / "memory.max").unlink()
        env = dict(os.environ,
                   LAUNCH_MEMINFO=str(self.meminfo),
                   LAUNCH_CGROUP_DIR=str(self.cgroup),
                   LAUNCH_MIN_MEM_AVAIL_GIB="90")
        out = subprocess.run(["bash", str(REPO / "engine" / "mem-gate.sh")],
                             capture_output=True, text=True, env=env,
                             cwd=REPO, timeout=20)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertRegex(out.stdout, r"memory guard: 100 GiB \(source: meminfo\)")


class TestPleAssertion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = (REPO / "engine" / "launch.sh").read_text()

    def test_engagement_markers_cover_both_patcher_lanes(self):
        for token in ("using packed mmap table", "mmap table attached",
                      "file-backed table"):
            self.assertIn(token, self.src)

    def test_assertion_gated_on_v030_ple_offload(self):
        self.assertRegex(
            self.src,
            r'\[\[ "\$V030" == "true" && "\$PLE_OFFLOAD" == "true" \]\]')

    def test_assertion_after_readiness_and_fails_loudly(self):
        ready = self.src.index("vLLM is ready and serving")
        assert_pos = self.src.index("no PLE table engagement line")
        self.assertGreater(assert_pos, ready)


class TestNoBannedHostToken(unittest.TestCase):
    def test_touched_files_clean(self):
        for rel in ("scripts/memwatch.sh", "engine/mem-gate.sh",
                    "engine/launch.sh", "tests/test_mem_gate.sh",
                    "tests/test_memory_guard.py"):
            text = (REPO / rel).read_text()
            self.assertIsNone(re.search(r"\bmsi\b", text, re.IGNORECASE), rel)


if __name__ == "__main__":
    unittest.main()
