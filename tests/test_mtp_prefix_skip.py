"""Source pins + application tests for patches/patch_mtp_prefix_skip.py.

The real-orig class needs the v0.30 image's nvidia/mtp.py extracted by a prod
boot (same file the draft-vocab v0.30 tests use); it skips until then. The
fake-orig class embeds a module whose string literals carry the three patch
anchors verbatim, so the anchor/gate logic runs anywhere.
"""
import ast
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PATCHER = REPO / "patches" / "patch_mtp_prefix_skip.py"
ORIG = REPO / "patches" / "mtp_v030_patched.py.orig"

# The anchors, verbatim from the image file (indentation is part of them).
INIT_ANCHOR = """                for idx in range(self.num_mtp_layers)
            )
"""
MIXER_ANCHOR = """        multi_hidden, sample_hidden_states, _ = (
            self.hyper_connection_mixer.combine_and_mix(
                hidden_states, block_output, injection
            )
        )
"""
CLASS_ANCHOR = "class Qwen4ExpMTP(nn.Module, SupportsPP, Qwen4ExpMixtureOfExperts):\n"


def fake_mtp_src():
    """Minimal parseable module carrying all three anchors in string literals."""
    return (
        "import torch\nfrom torch import nn\n\n"
        + CLASS_ANCHOR
        + "    def __init__(self):\n"
        + "        self.layers = nn.ModuleList()\n"
        + "        anchors = '''\n"
        + INIT_ANCHOR
        + "\n"
        + MIXER_ANCHOR
        + "'''\n"
    )


def stage(tmp, src):
    out = tmp / "mtp_prefix_skip"
    (out / "orig").mkdir(parents=True)
    (out / "orig" / "mtp.py").write_text(src)
    return out


def run_patcher(out):
    return subprocess.run(
        [sys.executable, str(PATCHER), str(out)], capture_output=True, text=True
    )


class PatcherSource(unittest.TestCase):
    """Credit and fail-closed pins that hold without an extracted orig."""

    def setUp(self):
        self.src = PATCHER.read_text()
        self.doc = ast.get_docstring(ast.parse(self.src)) or ""

    def test_credits_in_docstring(self):
        self.assertIn("ds4-dfm-rs", self.doc)
        self.assertIn("flashrt", self.doc)
        self.assertIn("docs/qwen38-perf-2026-09-14.md", self.doc)
        self.assertIn("docs/ds41-qwen38-serve-lessons.md", self.doc)
        self.assertIn("docs/improvement-plan.md", self.doc)
        self.assertIn("no code copied", self.doc)
        self.assertIn("MIT", self.doc)
        self.assertIn("Apache-2.0", self.doc)

    def test_no_host_identifiers_in_committed_files(self):
        for path in (PATCHER, REPO / "engine" / "patches.sh",
                     REPO / "engine" / "config.sh", REPO / "recipes" / "prod.yaml"):
            self.assertIsNone(
                re.search(r"\bmsi\b", path.read_text()),
                f"{path.name} mentions the bench host",
            )

    def test_runtime_gates_present(self):
        self.assertIn("VLLM_MTP_SKIP_PREFIX", self.src)
        self.assertIn("is_current_stream_capturing", self.src)
        self.assertIn("num_tokens_across_dp", self.src)
        self.assertIn("data_parallel_size", self.src)
        self.assertIn("pipeline_parallel_size", self.src)
        self.assertIn("CUDAGraphMode.FULL_DECODE_ONLY", self.src)
        self.assertIn("_arm_prefix_skip", self.src)


class AppliesToFakeOrig(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_applies_and_parses(self):
        out = stage(self.tmp, fake_mtp_src())
        r = run_patcher(out)
        self.assertEqual(r.returncode, 0, r.stderr)
        patched = (out / "mtp_v030.py").read_text()
        ast.parse(patched)
        self.assertIn("_arm_prefix_skip(self, vllm_config)", patched)
        self.assertIn("tail = _prefill_tail_rows(hidden_states)", patched)
        self.assertIn("_mixer_tail_combine(", patched)
        self.assertEqual(patched.count("def _arm_prefix_skip"), 1)

    def test_already_patched_fails(self):
        out = stage(self.tmp, fake_mtp_src())
        self.assertEqual(run_patcher(out).returncode, 0)
        repatched = stage(self.tmp / "again", (out / "mtp_v030.py").read_text())
        r = run_patcher(repatched)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already patched", r.stderr)

    def test_anchor_drift_fails_closed(self):
        drifted = fake_mtp_src().replace(
            "hidden_states, block_output, injection",
            "hidden_states, block_output,injection",
        )
        out = stage(self.tmp, drifted)
        r = run_patcher(out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("anchor", r.stderr)
        self.assertFalse((out / "mtp_v030.py").exists())

    def test_missing_orig_fails(self):
        r = run_patcher(self.tmp / "nothing")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("missing", r.stderr)


@unittest.skipUnless(ORIG.is_file(), "no extracted v0.30 mtp.py; boot the prod recipe once (RECIPE=prod ./start.sh)")
class AppliesToRealOrig(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        shutil.copy(ORIG, self.tmp / "mtp_v030_patched.py.orig")
        for name in ("patch_mtp_draft_vocab_v030.py", "patch_mtp_draft_vocab.py"):
            shutil.copy(REPO / "patches" / name, self.tmp / name)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_applies_and_parses(self):
        out = stage(self.tmp, ORIG.read_text())
        r = run_patcher(out)
        self.assertEqual(r.returncode, 0, r.stderr)
        patched = (out / "mtp_v030.py").read_text()
        ast.parse(patched)
        self.assertIn("_arm_prefix_skip(self, vllm_config)", patched)
        self.assertIn("tail = _prefill_tail_rows(hidden_states)", patched)
        self.assertIn("_mixer_tail_combine(", patched)
        self.assertEqual(patched.count("def _arm_prefix_skip"), 1)

    def test_rerun_gives_same_output(self):
        out = stage(self.tmp, ORIG.read_text())
        self.assertEqual(run_patcher(out).returncode, 0)
        first = (out / "mtp_v030.py").read_text()
        self.assertEqual(run_patcher(out).returncode, 0)
        self.assertEqual((out / "mtp_v030.py").read_text(), first)

    def test_composes_after_draft_vocab(self):
        # The chain engine/patches.sh runs when mtp_draft_vocab is set:
        # draft-vocab first, then the prefix skip on its output.
        dv = subprocess.run(
            [sys.executable, str(self.tmp / "patch_mtp_draft_vocab_v030.py")],
            capture_output=True, text=True, cwd=self.tmp,
        )
        self.assertEqual(dv.returncode, 0, dv.stderr)
        chained = (self.tmp / "mtp_v030_patched.py").read_text()
        out = stage(self.tmp, chained)
        self.assertEqual(run_patcher(out).returncode, 0, run_patcher(out).stderr)
        patched = (out / "mtp_v030.py").read_text()
        ast.parse(patched)
        self.assertIn("_attach_draft_vocab(self)", patched)
        self.assertIn("def get_top_tokens(self, hidden_states", patched)
        self.assertIn("_arm_prefix_skip(self, vllm_config)", patched)
        self.assertEqual(patched.count("logger = init_logger(__name__)"), 1)


if __name__ == "__main__":
    unittest.main()
