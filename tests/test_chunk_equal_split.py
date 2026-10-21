"""Chunk-policy tests for patches/patch_chunk_equal_split.py.

The policy classes are pure python — no engine, no image, no torch. The apply
class needs the v0.32 image's v1/core/sched/scheduler.py, which
engine/patches.sh extracts into patches/chunk_equal_split/orig on a prod boot;
it skips until then.
"""
import ast
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PATCHER = REPO / "patches" / "patch_chunk_equal_split.py"
ORIG = REPO / "patches" / "chunk_equal_split" / "orig" / "scheduler.py"
TOUCHED = (
    PATCHER,
    REPO / "engine" / "patches.sh",
    REPO / "engine" / "config.sh",
    REPO / "recipes" / "prod.yaml",
)


def load_patcher():
    spec = importlib.util.spec_from_file_location("chunk_eq_patch", PATCHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


POLICY = load_patcher()
CAP = 4800


def stock_plan(total, cap):
    """What vLLM v0.32 does today: take the cap while it is under what is
    left, run the remainder as the last chunk."""
    plan = []
    left = total
    while left > cap:
        plan.append(cap)
        left -= cap
    plan.append(left)
    return plan


class Policy(unittest.TestCase):
    """chunk_rows / plan_chunks: the policy itself, no image involved."""

    def test_fits_one_chunk(self):
        self.assertEqual(POLICY.chunk_rows(1, CAP), 1)
        self.assertEqual(POLICY.chunk_rows(4799, CAP), 4799)
        self.assertEqual(POLICY.chunk_rows(CAP, CAP), CAP, "exact cap is one chunk")
        self.assertEqual(POLICY.plan_chunks(500, CAP), [500])

    def test_stub_elimination_5000_over_4800(self):
        self.assertEqual(POLICY.chunk_rows(5000, CAP), 2500)
        self.assertEqual(POLICY.plan_chunks(5000, CAP), [2500, 2500])
        self.assertEqual(stock_plan(5000, CAP), [4800, 200])  # the stub we remove

    def test_exact_multiple_has_no_split_leftover(self):
        self.assertEqual(POLICY.plan_chunks(2 * CAP, CAP), [CAP, CAP])
        self.assertEqual(POLICY.plan_chunks(3 * CAP, CAP), [CAP, CAP, CAP])

    def test_three_chunks_stay_equal(self):
        self.assertEqual(POLICY.plan_chunks(9700, CAP), [3234, 3233, 3233])

    def test_grid_no_stub_never_above_cap_step_count_unchanged(self):
        for cap in (1, 2, 7, 128, CAP, 8192):
            for total in range(1, 4 * cap + 4):
                plan = POLICY.plan_chunks(total, cap)
                self.assertEqual(sum(plan), total, (total, cap, plan))
                self.assertEqual(len(plan), -(-total // cap),
                                 f"step count grew: {total} {cap} {plan}")
                self.assertLessEqual(max(plan), cap, (total, cap, plan))
                first = POLICY.chunk_rows(total, cap)
                self.assertGreaterEqual(min(plan), first - 1,
                                        f"stub tail: {total} {cap} {plan}")

    def test_tail_chunk_never_near_empty(self):
        """The whole point: a split prompt's last chunk is a full share, not a
        stub. Every 2-chunk prompt ends at >= half the cap."""
        for cap in (7, 128, CAP):
            for total in range(cap + 1, 2 * cap + 1):
                plan = POLICY.plan_chunks(total, cap)
                self.assertEqual(len(plan), 2, (total, cap, plan))
                self.assertGreaterEqual(plan[-1], (total + 1) // 2 - 1, (total, plan))
                self.assertGreaterEqual(plan[-1], cap // 2, (total, cap, plan))

    def test_generated_helper_agrees_with_policy(self):
        """The source shipped into scheduler.py is the same policy."""
        ns = {}
        exec(compile(ast.parse(POLICY.helper_block()), "<gate>", "exec"), {}, ns)
        self.assertFalse(ns["_CHUNK_EQUAL_SPLIT"], "the gate defaults off")
        for cap in (1, 128, CAP):
            for total in (1, 4799, CAP, 5000, 9600, 9700, 20000):
                self.assertEqual(ns["_chunk_equal_split"](total, cap),
                                 POLICY.chunk_rows(total, cap), (total, cap))


class PatcherSource(unittest.TestCase):
    """Credit and fail-closed pins that hold without an extracted orig."""

    def setUp(self):
        self.src = PATCHER.read_text()
        self.doc = ast.get_docstring(ast.parse(self.src)) or ""

    def test_credits_in_docstring(self):
        self.assertIn("Niko1221/Strata", self.doc)
        self.assertIn("MIT", self.doc)
        self.assertIn("no code copied", self.doc)
        self.assertIn("0.1.38", self.doc)
        self.assertIn("#41", self.doc)
        self.assertIn("C++", self.doc)

    def test_gate_default_off(self):
        self.assertIn("VLLM_CHUNK_EQUAL_SPLIT", self.src)
        self.assertIn('else long_prefill_token_threshold', self.src)

    def test_self_check_exits_zero(self):
        r = subprocess.run([sys.executable, str(PATCHER), "--self-check"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("self-check ok", r.stdout)

    def test_no_host_identifiers_in_committed_files(self):
        import re
        for path in TOUCHED:
            self.assertIsNone(
                re.search(r"\bmsi\b", path.read_text(), re.IGNORECASE),
                f"{path.name} mentions the bench host",
            )

    def test_prod_recipe_key_defaults_off(self):
        out = subprocess.run(
            [sys.executable, str(REPO / "engine" / "recipe.py"),
             str(REPO / "recipes"), "prod"],
            capture_output=True, text=True, check=True).stdout
        self.assertIn("export CHUNK_EQUAL_SPLIT='false'", out)


class AppliesToFakeOrig(unittest.TestCase):
    """Anchor/gate logic on a synthetic module carrying the three anchors."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def fake_src(self):
        # Both clamp sites at their real indentation (the running-request one
        # sits in the schedule loop, the waiting one two levels deeper).
        return (
            "import time\n\nlogger = init_logger(__name__)\n\n\n"
            "class Scheduler:\n"
            "    def schedule(self):\n"
            "        num_new_tokens = 0\n"
            "        long_prefill_token_threshold = 0\n"
            "        while True:\n"
            "            if 0 < long_prefill_token_threshold < num_new_tokens:\n"
            "                num_new_tokens = long_prefill_token_threshold\n"
            "            if True:\n"
            "                if True:\n"
            "                    if 0 < long_prefill_token_threshold < num_new_tokens:\n"
            "                        num_new_tokens = long_prefill_token_threshold\n"
            "        return num_new_tokens\n"
        )

    def stage(self, src):
        out = self.tmp / "ces"
        (out / "orig").mkdir(parents=True, exist_ok=True)
        (out / "orig" / "scheduler.py").write_text(src)
        return out

    def run_patcher(self, out):
        return subprocess.run([sys.executable, str(PATCHER), str(out)],
                              capture_output=True, text=True)

    def test_applies_and_parses(self):
        out = self.stage(self.fake_src())
        r = self.run_patcher(out)
        self.assertEqual(r.returncode, 0, r.stderr)
        patched = (out / "scheduler_v032.py").read_text()
        ast.parse(patched)
        self.assertEqual(patched.count("def _chunk_equal_split("), 1)
        self.assertEqual(
            patched.count("if _CHUNK_EQUAL_SPLIT\n"), 2, "both clamp sites")

    def test_already_patched_fails(self):
        out = self.stage(self.fake_src())
        self.assertEqual(self.run_patcher(out).returncode, 0)
        again = self.tmp / "again"
        (again / "orig").mkdir(parents=True)
        shutil.copy(out / "scheduler_v032.py", again / "orig" / "scheduler.py")
        r = self.run_patcher(again)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already patched", r.stderr)

    def test_anchor_drift_fails_closed(self):
        drifted = self.fake_src().replace(
            "num_new_tokens = long_prefill_token_threshold",
            "num_new_tokens =  long_prefill_token_threshold")
        out = self.stage(drifted)
        r = self.run_patcher(out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("anchor", r.stderr)
        self.assertFalse((out / "scheduler_v032.py").exists())

    def test_missing_orig_fails(self):
        r = self.run_patcher(self.tmp / "nothing")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("missing", r.stderr)


@unittest.skipUnless(
    ORIG.is_file(),
    "no extracted v0.32 scheduler.py; boot the prod recipe once "
    "(RECIPE=prod ./start.sh) or extract it with docker cp")
class AppliesToRealOrig(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.out = self.tmp / "ces"
        (self.out / "orig").mkdir(parents=True)
        shutil.copy(ORIG, self.out / "orig" / "scheduler.py")

    def run_patcher(self):
        return subprocess.run([sys.executable, str(PATCHER), str(self.out)],
                              capture_output=True, text=True)

    def test_applies_both_sites_and_parses(self):
        r = self.run_patcher()
        self.assertEqual(r.returncode, 0, r.stderr)
        patched = (self.out / "scheduler_v032.py").read_text()
        ast.parse(patched)
        self.assertEqual(patched.count("def _chunk_equal_split("), 1)
        self.assertEqual(patched.count("if _CHUNK_EQUAL_SPLIT\n"), 2)
        self.assertEqual(patched.count("import os"), 1, "no duplicate import")
        # stock behaviour survives the gate: both sites keep the plain cap
        self.assertEqual(patched.count("else long_prefill_token_threshold"), 2)

    def test_only_the_gate_and_two_sites_change(self):
        self.assertEqual(self.run_patcher().returncode, 0)
        orig_lines = ORIG.read_text().splitlines()
        patched_lines = (self.out / "scheduler_v032.py").read_text().splitlines()
        kept = set(patched_lines)
        removed = [ln.strip() for ln in orig_lines if ln not in kept]
        self.assertEqual(
            removed,
            ["num_new_tokens = long_prefill_token_threshold"] * 2,
            "the patch must only add, besides the two clamp lines it rewrites")

    def test_rerun_gives_same_output(self):
        self.assertEqual(self.run_patcher().returncode, 0)
        first = (self.out / "scheduler_v032.py").read_text()
        self.assertEqual(self.run_patcher().returncode, 0)
        self.assertEqual((self.out / "scheduler_v032.py").read_text(), first)


if __name__ == "__main__":
    unittest.main()
