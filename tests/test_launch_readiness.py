#!/usr/bin/env python3
"""CPU-only launcher readiness and archived-diagnostics regressions."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


class TestLaunchReadiness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for rel in ("engine/mem-gate.sh", "scripts/start-memwatch.sh",
                    "scripts/stop-memwatch.sh"):
            self.executable(self.root / rel, "#!/bin/bash\nexit 0\n")
        self.executable(self.bin / "docker", f"#!{sys.executable}\n" + '''
import os
import sys
from pathlib import Path

command = sys.argv[1]
if command == "ps":
    if os.environ["FAKE_RUNNING"] == "true":
        print("vllm-fn")
elif command == "logs":
    if "-f" not in sys.argv:
        stream = sys.stderr.buffer if os.environ["FAKE_LOG_STDERR"] == "true" else sys.stdout.buffer
        stream.write(Path(os.environ["FAKE_LOG"]).read_bytes())
        stream.flush()
        sys.exit(int(os.environ["FAKE_LOG_STATUS"]))
elif command == "run":
    print("fake-container-id")
elif command != "rm":
    sys.exit(99)
''')
        self.log = self.root / "container.log"
        self.env = dict(os.environ,
                        PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        HOME=str(self.root), SCRIPT_DIR=str(self.root),
                        LAUNCH_SOURCE=str(REPO / "engine" / "launch.sh"),
                        MODEL_PATH="", CHAT_TEMPLATE="", CPUSET="",
                        NNODES="1", V030="true", PLE_OFFLOAD="true",
                        EXTRA_VLLM_ARGS="--api-key test-key", HEAD_IP="127.0.0.1",
                        HEAD_PLE_MOUNT="", HEAD_MODELOPT_MOUNT="",
                        OVERLAY_ENV_STR="", HF_CACHE_DIR=str(self.root / "hf"),
                        IMAGE="fake-image", MODEL_ARG="fake-model", VLLM_ARGS_STR="",
                        PORT="8000", SERVED_MODEL_NAME="fake-model", READY_TIMEOUT_S="60",
                        FAKE_RUNNING="true", FAKE_LOG=str(self.log),
                        FAKE_LOG_STATUS="0", FAKE_LOG_STDERR="false")

    @staticmethod
    def executable(path, source):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        path.chmod(0o700)

    def launch(self, log, **env):
        self.log.write_text(log)
        return subprocess.run(
            ["bash", "-c", '''
set -euo pipefail
info() { printf '%s\\n' "$*"; }
warn() { printf '%s\\n' "$*" >&2; }
ok() { printf '%s\\n' "$*"; }
err() { printf '%s\\n' "$*" >&2; exit 1; }
sleep() { :; }
curl() { printf '200'; }
source "$LAUNCH_SOURCE"
printf 'LAUNCH_COMPLETE\\n'
'''], env=dict(self.env, **env), cwd=self.root,
            capture_output=True, text=True, timeout=15)

    def assert_archived(self, log):
        archives = list((self.root / "logs").glob("vllm-fn-*.log"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(archives[0].read_text(), log)

    def test_oversized_healthy_logs_pass_for_all_engagement_markers(self):
        # An early match followed by several MiB forces the producer to keep
        # writing after grep has seen the marker, beyond any pipe buffer.
        for marker in ("using packed mmap table", "mmap table attached",
                       "file-backed table"):
            with self.subTest(marker=marker):
                log = marker + "\n" + "INFO loading shard contents\n" * 131072
                out = self.launch(log)
                self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
                self.assertIn("LAUNCH_COMPLETE", out.stdout)
                self.assertFalse((self.root / "logs").exists())

    def test_missing_marker_fails_and_preserves_benign_log(self):
        log = "INFO model loaded\nINFO server ready\n"
        out = self.launch(log)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("ple_offload is on", out.stderr)
        self.assertNotIn("LAUNCH_COMPLETE", out.stdout)
        self.assert_archived(log)

    def test_log_producer_failure_fails_even_after_marker(self):
        log = "mmap table attached\nINFO server ready\n"
        out = self.launch(log, FAKE_LOG_STATUS="7")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("ple_offload is on", out.stderr)
        self.assertNotIn("LAUNCH_COMPLETE", out.stdout)
        self.assert_archived(log)

    def test_engagement_marker_on_stderr_passes(self):
        out = self.launch("file-backed table\n", FAKE_LOG_STDERR="true")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("LAUNCH_COMPLETE", out.stdout)

    def test_ple_assertion_only_applies_to_requested_lane(self):
        for v030, ple_offload in (("false", "true"), ("true", "false"),
                                  ("false", "false")):
            with self.subTest(v030=v030, ple_offload=ple_offload):
                out = self.launch("INFO server ready\n", V030=v030,
                                  PLE_OFFLOAD=ple_offload)
                self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
                self.assertIn("LAUNCH_COMPLETE", out.stdout)

    def test_benign_archive_does_not_hide_original_container_failure(self):
        log = "INFO server shutdown complete\n"
        out = self.launch(log, FAKE_RUNNING="false")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("Container vllm-fn exited unexpectedly", out.stderr)
        self.assertNotIn("LAUNCH_COMPLETE", out.stdout)
        self.assert_archived(log)


if __name__ == "__main__":
    unittest.main()
