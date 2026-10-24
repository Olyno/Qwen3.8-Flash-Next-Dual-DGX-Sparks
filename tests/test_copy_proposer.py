"""Source pins, application tests and decision tests for
patches/patch_copy_proposer.py.

The anchors below are copied verbatim from the pinned v0.32.0 model_runner.py,
so the fake-orig class runs the anchor/count/parse logic anywhere. The
decision tests are pure python (no torch, no image): copy_match_continuation
is the shipped function, and kernel_pick re-implements the stock n-gram scan's
packed (n, pos) score in python so the two can be compared on the same input.
The real-orig and chained-real classes need the extracted image files and skip
until engine/patches.sh has run once.
"""
import ast
import importlib.util
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
PATCHER = REPO / "patches" / "patch_copy_proposer.py"
TRACE_PATCHER = REPO / "patches" / "patch_spec_trace.py"
ORIG = REPO / "patches" / "copy_proposer" / "orig" / "model_runner.py"
TRACE_ORIG = REPO / "patches" / "spec_trace" / "orig" / "model_runner.py"
ENV = "VLLM_COPY_PROPOSER"
TOUCHED = (
    PATCHER,
    REPO / "engine" / "patches.sh",
    REPO / "engine" / "config.sh",
    REPO / "engine" / "recipe.py",
    REPO / "recipes" / "prod.yaml",
    REPO / ".gitignore",
)

# Verbatim from the image's vllm/v1/worker/gpu/model_runner.py.
DENSE_ANCHOR = """\
        use_dense_all_token_ids = (
            self.speculative_config is not None and self.speculative_config.use_ngram()
        )
"""
STATE_ANCHOR = """\
        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
            num_prefill_lookahead=num_prefill_lookahead,
            use_dense_all_token_ids=use_dense_all_token_ids,
        )
"""
HANDOFF_ANCHOR = """\
            if num_spec_tokens < self.num_speculative_steps:
                draft_tokens[:, num_spec_tokens:] = -1
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens
"""
# The spec_trace patcher's anchors, needed by the chained test.
TRACE_EMIT_ANCHOR = """\
        model_runner_output.kv_connector_output = kv_connector_output
        model_runner_output.ec_connector_output = ec_connector_output

        return async_output
"""


def load_patcher(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # inspect.getsource() on the dispatcher class needs the module findable.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PATCH = load_patcher(PATCHER, "copy_proposer_patch")
TRACE_PATCH = load_patcher(TRACE_PATCHER, "spec_trace_patch_for_chain")


def fake_runner_src():
    """Minimal parseable module carrying both patchers' anchors in code."""
    return (
        "import functools\n"
        "import gc\n"
        "import time\n\n"
        "from vllm.logger import init_logger\n\n"
        "logger = init_logger(__name__)\n\n\n"
        "class RequestState:\n"
        "    pass\n\n\n"
        "class GPUModelRunner:\n"
        "    def __init__(self, vllm_config, device):\n"
        "        self.vllm_config = vllm_config\n"
        "        self.max_num_reqs = 4\n"
        "        self.max_model_len = 4096\n"
        "        self.num_speculative_steps = 4\n"
        "        self.is_last_pp_rank = True\n"
        "        self.speculative_config = None\n"
        + DENSE_ANCHOR
        + STATE_ANCHOR
        + "\n"
        "    def sample_tokens(self, grammar_output):\n"
        "        input_batch = self.execute_model_state.input_batch\n"
        "        sampler_output, num_sampled, num_rejected = self.sample(\n"
        "            hidden_states, input_batch, grammar_output\n"
        "        )\n"
        "        if self.speculator is not None:\n"
        "            draft_tokens = self.speculator.propose(input_batch)\n"
        + HANDOFF_ANCHOR
        + "\n"
        "        model_runner_output = ModelRunnerOutput()\n"
        + TRACE_EMIT_ANCHOR
    )


def stage(tmp, src, name="copy_proposer"):
    out = tmp / name
    (out / "orig").mkdir(parents=True)
    (out / "orig" / "model_runner.py").write_text(src)
    return out


def run_patcher(out):
    return subprocess.run(
        [sys.executable, str(PATCHER), str(out)], capture_output=True, text=True
    )


def kernel_pick(tokens, min_match=8, max_match=32, cap=None):
    """The stock scan's decision, written as the kernel packs it.

    _ngram_scan_kernel scans every n in [MIN_N, MAX_N] and every start pos in
    [0, seq_len - n - 1], scoring n * (L + 1) + pos + 1; one max picks the
    longest n, then the rightmost pos. _ngram_finalize_kernel gathers from
    best_pos + best_n. L is any value above every position.
    """
    seq_len = len(tokens)
    if seq_len < min_match:
        return None
    lp1 = 1 << 20
    best = 0
    for n in range(min_match, max_match + 1):
        suffix = tokens[seq_len - n:]
        for pos in range(0, seq_len - n):
            if tokens[pos:pos + n] == suffix:
                best = max(best, n * lp1 + pos + 1)
    if best == 0:
        return None
    start = best - 1
    n = start // lp1
    pos = start - n * lp1
    start = pos + n
    end = seq_len if cap is None else start + cap
    return list(tokens[start:end])


class MatchDecision(unittest.TestCase):
    """copy_match_continuation: the pure decision the kernel mirrors."""

    def test_no_match(self):
        self.assertIsNone(PATCH.copy_match_continuation(list(range(64))))
        self.assertIsNone(PATCH.copy_match_continuation([]))
        # Nothing can match when the buffer cannot hold a match plus one token.
        self.assertIsNone(PATCH.copy_match_continuation(list(range(8)), 8, 32))
        self.assertIsNone(PATCH.copy_match_continuation([7] * 7, 8, 32))

    def test_match_at_suffix(self):
        pat = [1, 2, 3, 4, 5, 6, 7, 8]
        tokens = pat + [99] + pat
        self.assertEqual(
            PATCH.copy_match_continuation(tokens, 8, 32, 4), [99, 1, 2, 3]
        )
        self.assertEqual(
            PATCH.copy_match_continuation(tokens, 8, 32),
            tokens[8:],
        )

    def test_rightmost_of_equal_length(self):
        pat = [1, 2, 3, 4, 5, 6, 7, 8]
        tokens = pat + [50] + pat + [60] + pat
        # Both earlier occurrences are candidates; the rightmost one wins, so
        # the continuation starts with 60, not 50.
        self.assertEqual(
            PATCH.copy_match_continuation(tokens, 8, 32, 2), [60, 1]
        )

    def test_longest_beats_rightmost(self):
        pat9 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # n=9 matches at pos 0 (continuation 70). The shorter n=8 suffix
        # [2..9] also matches later, at the 80 block, but longest wins.
        tokens = pat9 + [70] + [2, 3, 4, 5, 6, 7, 8, 9, 80] + pat9
        self.assertEqual(
            PATCH.copy_match_continuation(tokens, 8, 32, 2), [70, 2]
        )

    def test_min_max_bounds(self):
        pat7 = [1, 2, 3, 4, 5, 6, 7]
        tokens = pat7 + [99] + pat7
        self.assertIsNone(PATCH.copy_match_continuation(tokens, 8, 32))
        self.assertEqual(
            PATCH.copy_match_continuation(tokens, 7, 32, 2), [99, 1]
        )

        long = list(range(33))
        shorter = long[1:]
        # n=33 matches at pos 0 (continuation starts at 100); the nested n=32
        # suffix also matches at pos 34 (continuation starts at 200). A max of
        # 32 must not consider the 33-token match.
        tokens = long + [100] + shorter + [200] + long
        self.assertEqual(
            PATCH.copy_match_continuation(tokens, 8, 33, 1), [100]
        )
        self.assertEqual(
            PATCH.copy_match_continuation(tokens, 8, 32, 1), [200]
        )

    def test_cap(self):
        pat = [1, 2, 3, 4, 5, 6, 7, 8]
        tokens = pat + [10, 11, 12, 13] + pat
        self.assertEqual(PATCH.copy_match_continuation(tokens, 8, 32, 3),
                         [10, 11, 12])
        self.assertEqual(PATCH.copy_match_continuation(tokens, 8, 32, 0), [])
        self.assertEqual(
            PATCH.copy_match_continuation(tokens, 8, 32, 99), tokens[8:]
        )
        # A match at the latest possible start leaves one continuation token.
        self.assertEqual(
            PATCH.copy_match_continuation([5] * 40, 8, 32, 99), [5]
        )

    def test_empty_buffer(self):
        for cap in (None, 0, 1, 4):
            self.assertIsNone(PATCH.copy_match_continuation([], 8, 32, cap))

    def test_kernel_packing_agrees_on_random_buffers(self):
        import random

        rng = random.Random(20261024)
        for _ in range(400):
            alphabet = rng.choice((2, 3, 4, 8, 32))
            n_tokens = rng.randrange(0, 90)
            tokens = [rng.randrange(alphabet) for _ in range(n_tokens)]
            min_match = rng.choice((1, 2, 4, 8))
            max_match = min_match + rng.choice((0, 1, 4, 24))
            cap = rng.choice((None, 0, 1, 3))
            self.assertEqual(
                PATCH.copy_match_continuation(tokens, min_match, max_match, cap),
                kernel_pick(tokens, min_match, max_match, cap),
                f"tokens={tokens} min={min_match} max={max_match} cap={cap}",
            )


class HelperBlock(unittest.TestCase):
    """The generated module block parses and the gate is literal."""

    def test_gate_reads_the_env(self):
        for value, expected in (
            (None, False),
            ("0", False),
            ("1", True),
            (" 1 ", True),
            ("true", False),
        ):
            env = {} if value is None else {ENV: value}
            with mock.patch.dict(os.environ, env, clear=True):
                ns = {}
                exec(compile(PATCH.helper_block(), "<copy-proposer>", "exec"), ns)
            self.assertIs(ns["_COPY_PROPOSER_ON"], expected, value)


class PatcherSource(unittest.TestCase):
    """Docstring and wiring pins that hold without an extracted orig."""

    def setUp(self):
        self.src = PATCHER.read_text()
        self.doc = ast.get_docstring(ast.parse(self.src)) or ""

    def test_docstring_states_the_contract(self):
        for needle in (
            "vllm/v1/worker/gpu/model_runner.py",
            "VLLM_COPY_PROPOSER",
            "8 and 32",
            "rightmost longest",
            "Lossless",
            "exact rejection sampling",
            "no D2H read",
            "eager",
            "patch_spec_trace.py",
            "use_ngram()",
        ):
            self.assertIn(needle, self.doc)

    def test_docstring_states_the_kernel_launches(self):
        for needle in (
            "_ngram_scan_kernel[(num_reqs, n_blocks)]",
            "_ngram_finalize_kernel[(num_reqs,)]",
            "num_warps=4",
            "num_stages=1",
            "K_PO2",
            "N_BLOCKS_PO2",
            "spec_decode.ngram.speculator",
        ):
            self.assertIn(needle, self.doc)

    def test_gate_default_off(self):
        self.assertIn('os.environ.get("VLLM_COPY_PROPOSER", "")', self.src)
        self.assertIn('os.environ.get("VLLM_COPY_PROPOSER_MIN_MATCH", "8")',
                      self.src)
        self.assertIn('os.environ.get("VLLM_COPY_PROPOSER_MAX_MATCH", "32")',
                      self.src)

    def test_fail_closed_messages(self):
        for needle in (
            "already patched",
            "anchor {i} not unique/missing",
            "does not parse",
            "ERROR: missing",
            "count != 1",
            "ast.parse",
        ):
            self.assertIn(needle, self.src)

    def test_no_device_sync_in_the_dispatcher(self):
        block = PATCH.helper_block()
        for call in (".item()", ".cpu()", ".tolist()", ".numpy()"):
            self.assertNotIn(call, block)

    def test_stock_kernels_are_imported_not_copied(self):
        block = PATCH.helper_block()
        self.assertIn(
            "from vllm.v1.worker.gpu.spec_decode.ngram.speculator import", block
        )
        for kernel in ("_ngram_scan_kernel", "_ngram_finalize_kernel"):
            self.assertEqual(block.count(f"def {kernel}("), 0,
                             f"{kernel} must be reused, not redefined")
            self.assertGreaterEqual(self.src.count(kernel), 2)

    def test_launch_argument_order_matches_the_stock_signature(self):
        tree = ast.parse(PATCH.helper_block())
        calls = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Subscript):
                func = func.value
            if isinstance(func, ast.Attribute) and func.attr in (
                "_scan", "_finalize"
            ):
                calls[func.attr] = node
        self.assertEqual(sorted(calls), ["_finalize", "_scan"])
        expected = {
            "_scan": [
                "token_ids",
                "token_ids.stride(0)",
                "idx_mapping",
                "req_states.total_len.gpu",
                "num_sampled",
                "self.scratch",
                "self.scratch.stride(0)",
                "self.max_model_len",
                "self.min_match",
                "self.max_match",
                "self.max_match_po2",
                "self.block_l",
            ],
            "_finalize": [
                "token_ids",
                "token_ids.stride(0)",
                "idx_mapping",
                "req_states.total_len.gpu",
                "num_sampled",
                "req_states.last_sampled_tokens.view(-1)",
                "self.scratch",
                "self.scratch.stride(0)",
                "self.drafts",
                "self.max_model_len",
                "self.n_blocks",
                "self.num_speculative_steps",
                "self.k_po2",
                "self.n_blocks_po2",
            ],
        }
        for name, call in calls.items():
            self.assertEqual(
                [ast.unparse(arg) for arg in call.args], expected[name], name
            )
        self.assertEqual(
            {kw.arg: ast.literal_eval(kw.value) for kw in calls["_scan"].keywords},
            {"num_warps": 4, "num_stages": 2},
        )
        self.assertEqual(
            {kw.arg: ast.literal_eval(kw.value)
             for kw in calls["_finalize"].keywords},
            {"num_warps": 2, "num_stages": 1},
        )

    def test_anchors_are_the_stock_lines(self):
        # The patcher's anchors must be the verbatim stock fragments the fake
        # module embeds; a drift here would make the fake test meaningless.
        self.assertIn(PATCH.DENSE_OLD, DENSE_ANCHOR)
        self.assertIn(PATCH.STATE_OLD, STATE_ANCHOR)
        self.assertIn(PATCH.HANDOFF_OLD, HANDOFF_ANCHOR)

    def test_wiring_pins(self):
        patches = (REPO / "engine" / "patches.sh").read_text()
        self.assertIn("patch_copy_proposer.py", patches)
        self.assertIn('-e VLLM_COPY_PROPOSER=1', patches)
        self.assertIn("$CP/model_runner_v030.py", patches)
        self.assertIn('COPY_PROPOSER" != "true"', patches)
        config = (REPO / "engine" / "config.sh").read_text()
        self.assertIn('COPY_PROPOSER="${COPY_PROPOSER:-false}"', config)
        self.assertIn("copy_proposer is only supported", config)
        self.assertIn("copy_proposer needs speculative decoding", config)
        self.assertIn(
            "copy_proposer and mtp_adaptive_depth both overlay", config
        )
        self.assertIn("COPY_PROPOSER", (REPO / "engine" / "recipe.py").read_text())
        self.assertIn("patches/copy_proposer/",
                      (REPO / ".gitignore").read_text())

    def test_recipe_key_defaults_off(self):
        out = subprocess.run(
            [sys.executable, str(REPO / "engine" / "recipe.py"),
             str(REPO / "recipes"), "prod"],
            capture_output=True, text=True, check=True,
        ).stdout
        self.assertIn("export COPY_PROPOSER='false'", out)
        yaml = (REPO / "recipes" / "prod.yaml").read_text()
        self.assertIn("copy_proposer: false", yaml)
        self.assertIn("lossless", yaml.lower())

    def test_no_host_identifiers(self):
        # Built at runtime so the name itself is never committed.
        host = "m" + "si"
        for path in TOUCHED:
            self.assertIsNone(
                re.search(rf"\b{host}\b", path.read_text(), re.IGNORECASE),
                f"{path.name} mentions the bench host",
            )


class AppliesToFakeOrig(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_applies_and_parses(self):
        out = stage(self.tmp, fake_runner_src())
        r = run_patcher(out)
        self.assertEqual(r.returncode, 0, r.stderr)
        patched = (out / "model_runner_v030.py").read_text()
        ast.parse(patched)
        self.assertEqual(patched.count("def copy_match_continuation("), 1)
        self.assertEqual(patched.count("class _CopyProposer"), 1)
        self.assertEqual(patched.count("def merge_drafts("), 1)
        self.assertEqual(patched.count("_COPY_PROPOSER_ON = ("), 1)
        self.assertEqual(patched.count("if _COPY_PROPOSER_ON:"), 1)
        self.assertEqual(
            patched.count("use_dense_all_token_ids = True"), 1
        )
        self.assertEqual(patched.count("self._copy_proposer = None"), 1)
        self.assertEqual(
            patched.count("self._copy_proposer.merge_drafts("), 1
        )
        self.assertEqual(patched.count("import os"), 1)
        # The three MTP draft rows and the handoff write are still there.
        self.assertIn("draft_tokens[:, num_spec_tokens:] = -1", patched)
        self.assertIn(
            "self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens",
            patched,
        )

    def test_already_patched_fails(self):
        out = stage(self.tmp, fake_runner_src())
        self.assertEqual(run_patcher(out).returncode, 0)
        again = stage(self.tmp / "again",
                      (out / "model_runner_v030.py").read_text())
        r = run_patcher(again)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already patched", r.stderr)

    def test_anchor_drift_fails_closed(self):
        drifted = fake_runner_src().replace(
            "            use_dense_all_token_ids=use_dense_all_token_ids,\n",
            "            use_dense_all_token_ids=use_dense_all_token_ids ,\n",
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

    def test_applies_when_os_is_already_imported(self):
        # The trace patcher adds `import os` at the top; the copy block adds
        # its own, so the chained file parses with two imports and one copy
        # hook.
        traced = TRACE_PATCH.patch(fake_runner_src())
        out = stage(self.tmp, traced)
        r = run_patcher(out)
        self.assertEqual(r.returncode, 0, r.stderr)
        patched = (out / "model_runner_v030.py").read_text()
        ast.parse(patched)
        self.assertEqual(patched.count("import os"), 2)


class ChainedWithTrace(unittest.TestCase):
    """Both keys on: trace first, copy second, both hooks exactly once."""

    def check(self, src):
        traced = TRACE_PATCH.patch(src)
        chained = PATCH.patch(traced)
        ast.parse(chained)
        self.assertEqual(chained.count("def _spec_trace_emit("), 1)
        self.assertEqual(
            chained.count("            _spec_trace_emit(input_batch, num_rejected)"),
            1,
        )
        self.assertEqual(chained.count("def copy_match_continuation("), 1)
        self.assertEqual(chained.count("class _CopyProposer"), 1)
        self.assertEqual(
            chained.count("self._copy_proposer.merge_drafts("), 1
        )
        self.assertEqual(chained.count("if _COPY_PROPOSER_ON:"), 1)
        self.assertEqual(
            chained.count("use_dense_all_token_ids = True"), 1
        )
        return chained

    def test_fake_orig(self):
        chained = self.check(fake_runner_src())
        self.assertEqual(chained.count("import os"), 2)

    @unittest.skipUnless(
        TRACE_ORIG.is_file(),
        "no extracted v0.32 model_runner.py; boot the prod recipe once",
    )
    def test_real_orig(self):
        self.check(TRACE_ORIG.read_text())


class AppliesToRealOrig(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    @unittest.skipUnless(
        ORIG.is_file(),
        "no extracted v0.32 model_runner.py; extract it with docker cp or a launch",
    )
    def test_applies_and_parses(self):
        out = stage(self.tmp, ORIG.read_text())
        r = run_patcher(out)
        self.assertEqual(r.returncode, 0, r.stderr)
        patched = (out / "model_runner_v030.py").read_text()
        ast.parse(patched)
        self.assertEqual(patched.count("def copy_match_continuation("), 1)
        self.assertEqual(patched.count("def merge_drafts("), 1)
        self.assertEqual(
            patched.count("self._copy_proposer.merge_drafts("), 1
        )

    @unittest.skipUnless(
        ORIG.is_file(),
        "no extracted v0.32 model_runner.py; extract it with docker cp or a launch",
    )
    def test_rerun_fails_closed(self):
        out = stage(self.tmp, ORIG.read_text())
        self.assertEqual(run_patcher(out).returncode, 0)
        again = stage(self.tmp / "again",
                      (out / "model_runner_v030.py").read_text())
        r = run_patcher(again)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already patched", r.stderr)


if __name__ == "__main__":
    unittest.main()
