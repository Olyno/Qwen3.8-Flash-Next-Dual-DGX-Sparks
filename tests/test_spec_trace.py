"""Source pins + application tests for patches/patch_spec_trace.py.

The fake-orig class embeds a module whose string literals carry the patch
anchors verbatim, so the anchor/apply logic runs anywhere. The helper class
executes the exact module block the patcher injects, with a fake batch and a
fake rejected-count tensor, pinning the JSONL line format and the
alloc/accepted arithmetic. The real-orig class runs only when the v0.32
model_runner.py has been extracted by a boot.
"""
import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
PATCHER = REPO / "patches" / "patch_spec_trace.py"
ORIG = REPO / "patches" / "spec_trace" / "orig" / "model_runner.py"


def fake_runner_src():
    """Minimal parseable module carrying all four anchors in code."""
    return '''\
import functools
import gc
import time

from vllm.logger import init_logger

logger = init_logger(__name__)


class GPUModelRunner:
    def sample_tokens(self, grammar_output):
        input_batch = self.execute_model_state.input_batch
        self.execute_model_state = None

        # Last rank: sample tokens
        assert hidden_states is not None
        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )
        self.postprocess_sampled(
            input_batch.idx_mapping, num_sampled, num_rejected
        )
        model_runner_output.kv_connector_output = kv_connector_output
        model_runner_output.ec_connector_output = ec_connector_output

        return async_output
'''


def stage(tmp, src):
    out = tmp / "spec_trace"
    (out / "orig").mkdir(parents=True)
    (out / "orig" / "model_runner.py").write_text(src)
    return out


def run_patcher(out):
    return subprocess.run(
        [sys.executable, str(PATCHER), str(out)], capture_output=True, text=True
    )


def load_patcher():
    spec = importlib.util.spec_from_file_location("patch_spec_trace", PATCHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeTensor:
    def __init__(self, values):
        self._values = list(values)

    def __getitem__(self, item):
        return FakeTensor(self._values[item])

    def tolist(self):
        return list(self._values)


class FakeBatch:
    num_reqs = 3
    num_draft_tokens = 9
    num_draft_tokens_per_req = [3, 3, 3]


class FakeLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, *args):
        self.warnings.append(args)


class PatcherSource(unittest.TestCase):
    """Behavior pins that hold without an extracted orig."""

    def setUp(self):
        self.src = PATCHER.read_text()
        self.doc = ast.get_docstring(ast.parse(self.src)) or ""

    def test_docstring_states_contract(self):
        self.assertIn("Measurement only", self.doc)
        self.assertIn("VLLM_SPEC_TRACE", self.doc)
        self.assertIn("InputBatch.num_draft_tokens", self.doc)
        self.assertIn("not the draft graph width", self.doc)
        self.assertIn("flushed per line", self.doc)
        self.assertIn("one D2H read", self.doc)

    def test_default_off_is_a_string_check(self):
        self.assertIn('os.environ.get("VLLM_SPEC_TRACE", "")', self.src)
        self.assertIn("if _SPEC_TRACE_PATH:", self.src)

    def test_round_data_sources(self):
        self.assertIn("input_batch.num_draft_tokens", self.src)
        self.assertIn("input_batch.num_draft_tokens_per_req", self.src)
        self.assertIn("num_rejected", self.src)
        for key in ('"round"', '"ms"', '"alloc"', '"accepted"'):
            self.assertIn(key, self.src)

    def test_fail_closed_messages(self):
        self.assertIn("already patched", self.src)
        self.assertIn("anchor {i} not unique/missing", self.src)
        self.assertIn("does not parse", self.src)
        self.assertIn("ERROR: missing", self.src)

    def test_no_host_identifiers_in_committed_files(self):
        for path in (PATCHER, REPO / "engine" / "patches.sh",
                     REPO / "engine" / "config.sh", REPO / "recipes" / "prod.yaml"):
            self.assertIsNone(
                re.search(r"\bmsi\b", path.read_text()),
                f"{path.name} mentions the bench host",
            )

    def test_engine_wiring(self):
        self.assertIn("SPEC_TRACE", (REPO / "engine" / "config.sh").read_text())
        self.assertIn(
            'OVERLAY_ENV+=("-e VLLM_SPEC_TRACE=',
            (REPO / "engine" / "patches.sh").read_text(),
        )
        self.assertIn("spec_trace: false",
                      (REPO / "recipes" / "prod.yaml").read_text())
        self.assertIn("SPEC_TRACE",
                      (REPO / "engine" / "recipe.py").read_text())


class InjectedHelper(unittest.TestCase):
    """Runs the exact module block the patcher inserts."""

    def setUp(self):
        self.patcher = load_patcher()
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def exec_block(self, trace_path):
        ns = {"os": os, "logger": FakeLogger()}
        env = {} if trace_path is None else {"VLLM_SPEC_TRACE": str(trace_path)}
        with mock.patch.dict(os.environ, env, clear=True):
            exec(compile(self.patcher.MODULE_BLOCK, "<spec-trace>", "exec"), ns)
        return ns

    def test_default_off_opens_nothing(self):
        ns = self.exec_block(None)
        self.assertEqual(ns["_SPEC_TRACE_PATH"], "")
        ns["_spec_trace_mark"]()
        ns["_spec_trace_emit"](FakeBatch(), FakeTensor([1, 0, 2]))
        self.assertIsNone(ns["_SPEC_TRACE_STATE"][2])
        self.assertFalse(ns["_SPEC_TRACE_STATE"][4])

    def test_line_format_and_accepted_math(self):
        path = self.tmp / "trace.jsonl"
        ns = self.exec_block(path)
        # Same order as the patched call site: emit, then boundary mark.
        ns["_spec_trace_emit"](FakeBatch(), FakeTensor([1, 0, 2]))
        ns["_spec_trace_mark"]()
        ns["_spec_trace_emit"](FakeBatch(), FakeTensor([1, 0, 2]))
        ns["_spec_trace_mark"]()
        ns["_spec_trace_emit"](FakeBatch(), FakeTensor([3, 3, 3]))
        recs = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([r["round"] for r in recs], [1, 2, 3])
        first = recs[0]
        self.assertIsNone(first["ms"])
        self.assertEqual(first["alloc"], 9)
        self.assertEqual(first["accepted"], 6)
        self.assertEqual(first["reqs"], 3)
        self.assertEqual(first["alloc_req"], [3, 3, 3])
        self.assertEqual(first["accepted_req"], [2, 3, 1])
        self.assertEqual(first["rank"], 0)
        self.assertIsInstance(recs[1]["ms"], float)
        self.assertEqual(recs[2]["accepted"], 0)

    def test_none_rejected_is_a_noop(self):
        path = self.tmp / "trace.jsonl"
        ns = self.exec_block(path)
        ns["_spec_trace_mark"]()
        ns["_spec_trace_emit"](FakeBatch(), None)
        self.assertFalse(path.exists())

    def test_failed_open_disables_after_one_warning(self):
        path = self.tmp / "missing-dir" / "trace.jsonl"
        ns = self.exec_block(path)
        ns["_spec_trace_mark"]()
        ns["_spec_trace_emit"](FakeBatch(), FakeTensor([1, 0, 2]))
        ns["_spec_trace_emit"](FakeBatch(), FakeTensor([1, 0, 2]))
        self.assertTrue(ns["_SPEC_TRACE_STATE"][4])
        self.assertEqual(len(ns["logger"].warnings), 1)
        self.assertFalse(path.exists())


class AppliesToFakeOrig(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_applies_and_parses(self):
        out = stage(self.tmp, fake_runner_src())
        r = run_patcher(out)
        self.assertEqual(r.returncode, 0, r.stderr)
        patched = (out / "model_runner_v030.py").read_text()
        ast.parse(patched)
        self.assertIn("import os", patched)
        self.assertIn("VLLM_SPEC_TRACE", patched)
        self.assertEqual(patched.count("def _spec_trace_emit"), 1)
        self.assertEqual(
            patched.count("            _spec_trace_emit(input_batch, num_rejected)"),
            1,
        )
        self.assertEqual(patched.count("            _spec_trace_mark()"), 1)
        self.assertIn(
            "if _SPEC_TRACE_PATH and input_batch.num_draft_tokens:", patched
        )

    def test_already_patched_fails(self):
        out = stage(self.tmp, fake_runner_src())
        self.assertEqual(run_patcher(out).returncode, 0)
        repatched = stage(self.tmp / "again",
                          (out / "model_runner_v030.py").read_text())
        r = run_patcher(repatched)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already patched", r.stderr)

    def test_anchor_drift_fails_closed(self):
        drifted = fake_runner_src().replace(
            "        model_runner_output.ec_connector_output = ec_connector_output\n"
            "\n"
            "        return async_output",
            "        model_runner_output.ec_connector_output = ec_connector_output\n"
            "        return async_output",
        )
        out = stage(self.tmp, drifted)
        r = run_patcher(out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("anchor", r.stderr)
        self.assertFalse((out / "model_runner_v030.py").exists())

    def test_missing_orig_fails(self):
        r = run_patcher(self.tmp / "nothing")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("missing", r.stderr)


@unittest.skipUnless(
    ORIG.is_file(),
    "no extracted v0.32 model_runner.py; boot the prod recipe once",
)
class AppliesToRealOrig(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_applies_and_parses(self):
        out = stage(self.tmp, ORIG.read_text())
        r = run_patcher(out)
        self.assertEqual(r.returncode, 0, r.stderr)
        patched = (out / "model_runner_v030.py").read_text()
        ast.parse(patched)
        self.assertEqual(patched.count("def _spec_trace_emit"), 1)
        self.assertEqual(
            patched.count("            _spec_trace_emit(input_batch, num_rejected)"),
            1,
        )

    def test_rerun_refuses(self):
        out = stage(self.tmp, ORIG.read_text())
        self.assertEqual(run_patcher(out).returncode, 0)
        repatched = stage(self.tmp / "again",
                          (out / "model_runner_v030.py").read_text())
        r = run_patcher(repatched)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already patched", r.stderr)


if __name__ == "__main__":
    unittest.main()
