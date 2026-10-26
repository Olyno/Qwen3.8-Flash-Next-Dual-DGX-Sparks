"""Tests for scripts/ab_proxy_arm.sh.

The point of the harness is that an arm which cannot prove its configuration
fails loudly, so the negative path is the one that matters: a stale or
wrongly-started proxy must exit 2, not be published as a null result. These
tests stand up a plain listener with a known environment and check both
directions, without a model server.
"""

import os
import shutil
import socket
import subprocess
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "scripts", "ab_proxy_arm.sh")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(port, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.05)
    return False


@unittest.skipUnless(shutil.which("ss"), "needs ss to resolve the listener pid")
class ArmProofTest(unittest.TestCase):
    port = None
    listener = None

    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        env = dict(os.environ, PROXY_EDIT_GUARD="1", PROXY_REASONING_EFFORT="low")
        cls.listener = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(cls.port), "--bind", "127.0.0.1"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not wait_for_port(cls.port):
            cls.listener.kill()
            raise unittest.SkipTest("listener never bound")

    @classmethod
    def tearDownClass(cls):
        if cls.listener and cls.listener.poll() is None:
            cls.listener.terminate()
            cls.listener.wait(timeout=10)

    def run_arm(self, arm, *envs):
        cmd = ["bash", SCRIPT, arm, "--port", str(self.port), *envs, "--assert-only"]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                              env=dict(os.environ, AB_LOG_DIR="/tmp/ab-test"))

    def test_matching_environment_passes(self):
        out = self.run_arm("ok", "PROXY_EDIT_GUARD=1")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("ARM OK", out.stdout)
        log = open("/tmp/ab-test/ok.log", encoding="utf-8").read()
        self.assertIn("PROXY_EDIT_GUARD=1", log)
        self.assertIn("ARM OK", log)

    def test_unset_key_fails(self):
        out = self.run_arm("unset", "PROXY_DIFF_TOOL_OUTPUT=1")
        self.assertEqual(out.returncode, 2, out.stderr)
        self.assertIn("PROXY_DIFF_TOOL_OUTPUT=<unset>", out.stderr)

    def test_different_value_fails(self):
        out = self.run_arm("diff", "PROXY_REASONING_EFFORT=medium")
        self.assertEqual(out.returncode, 2, out.stderr)
        self.assertIn("PROXY_REASONING_EFFORT=low", out.stderr)

    def test_log_written_before_assertion(self):
        self.run_arm("prelog", "PROXY_EDIT_GUARD=1")
        log = open("/tmp/ab-test/prelog.log", encoding="utf-8").read()
        head = log.splitlines()[:2]
        self.assertTrue(head[0].startswith("# arm=prelog at "), head)
        self.assertIn("requested: PROXY_EDIT_GUARD=1", head[1])


if __name__ == "__main__":
    unittest.main()
