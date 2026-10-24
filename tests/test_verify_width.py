"""Source pins, anchor application and policy tests for
patches/patch_verify_width.py.

The policy functions are pure python: no engine, no image, no torch. The
fake-orig class embeds a module whose anchor text is copied verbatim from the
v0.32 scheduler, so the anchor/count/parse logic runs anywhere. The
real-orig class needs the image's v1/core/sched/scheduler.py, which
engine/patches.sh extracts on a launch; it skips until then.
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
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
PATCHER = REPO / "patches" / "patch_verify_width.py"
ORIG = REPO / "patches" / "verify_width" / "orig" / "scheduler.py"
ENV = "VLLM_VERIFY_WIDTH_ADAPTIVE"
TOUCHED = (
    PATCHER,
    REPO / "engine" / "patches.sh",
    REPO / "engine" / "config.sh",
    REPO / "engine" / "recipe.py",
    REPO / "recipes" / "prod.yaml",
    REPO / ".gitignore",
)


def load_patcher():
    spec = importlib.util.spec_from_file_location("verify_width_patch", PATCHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PATCH = load_patcher()

# The anchors, verbatim from the pinned v0.32.0 scheduler.py.
EAGLE_ANCHOR = """\
            if self.use_eagle and not self.use_eagle_block_drop:
                logger.warning(
                    "EAGLE trailing prefix-cache block dropping is disabled. "
                    "This is experimental and may affect speculative-token "
                    "acceptance rates."
                )
"""
INIT_ANCHOR = """\
        # Create the KV cache manager.
        if hash_block_size is None:
            hash_block_size = block_size
"""
SCHEDULE_ANCHOR = """\
    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.current_step += 1
"""
PAD_ANCHOR = """\
                        padded_num_tokens = 1 + self.num_spec_tokens
"""
MAMBA_ANCHOR = """\
                        if (
                            pad_spec_decode
                            and num_new_tokens != 1 + self.num_spec_tokens
                        ):
"""
FILL_ANCHOR = """\
                if pad_spec_decode:
                    assert num_new_tokens == 1 + self.num_spec_tokens
                    scheduled_spec_decode_tokens[request_id] = [
                        -1
                    ] * self.num_spec_tokens
"""
PUBLISH_ANCHOR = """\
        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
"""
ACCEPT_ANCHOR = """\
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                num_rejected = num_draft_tokens - num_accepted
"""
END_ANCHOR = """\
            eco.scheduler_stats = stats

        return engine_core_outputs
"""

ANCHOR_TEXT = (
    EAGLE_ANCHOR
    + INIT_ANCHOR
    + SCHEDULE_ANCHOR
    + PAD_ANCHOR
    + MAMBA_ANCHOR
    + FILL_ANCHOR
    + PUBLISH_ANCHOR
    + ACCEPT_ANCHOR
    + END_ANCHOR
)


def fake_scheduler_src():
    """Minimal parseable module carrying every anchor verbatim."""
    return (
        "import itertools\n"
        "import time\n\n\n"
        "ANCHOR_TEXT = '''\n"
        + ANCHOR_TEXT
        + "'''\n\n\n"
        "class SchedulerInterface:\n"
        "    pass\n\n\n"
        "class Scheduler(SchedulerInterface):\n"
        "    def __init__(\n"
        "        self,\n"
        "    ) -> None:\n"
        "        pass\n"
    )


def prepare(tmp, src):
    out = tmp / "verify_width"
    (out / "orig").mkdir(parents=True)
    (out / "orig" / "scheduler.py").write_text(src)
    return out


def run_patcher(out):
    return subprocess.run(
        [sys.executable, str(PATCHER), str(out)], capture_output=True, text=True
    )


def load_helpers(gate):
    """Exec the shipped helper block with the env gate as it would boot."""
    ns = {"logger": mock.Mock()}
    old = os.environ.pop(ENV, None)
    try:
        if gate is not None:
            os.environ[ENV] = gate
        exec(compile(ast.parse(PATCH.helper_block()), "<helper>", "exec"), ns)
    finally:
        os.environ.pop(ENV, None)
        if old is not None:
            os.environ[ENV] = old
    return ns


def fake_sched(config_k=4, dp=1, table=None):
    return SimpleNamespace(
        num_spec_tokens=config_k,
        parallel_config=SimpleNamespace(data_parallel_size=dp),
        dynamic_sd_lookup=table,
    )


class Policy(unittest.TestCase):
    """EMA, thresholds, dwell and cap: the policy itself, no image involved."""

    def test_ema_first_round_is_the_ratio(self):
        self.assertEqual(PATCH._verify_width_update_ema(None, 3, 4), 0.75)
        self.assertEqual(PATCH._verify_width_update_ema(None, 1, 3), 1.0 / 3.0)
        self.assertEqual(PATCH._verify_width_update_ema(None, 0, 5), 0.0)

    def test_ema_recursive_math(self):
        # alpha 0.25: new = 0.25 * ratio + 0.75 * previous
        self.assertAlmostEqual(
            PATCH._verify_width_update_ema(0.5, 1, 4), 0.25 * 0.25 + 0.75 * 0.5
        )
        self.assertAlmostEqual(
            PATCH._verify_width_update_ema(0.5, 4, 4), 0.25 * 1.0 + 0.75 * 0.5
        )
        self.assertAlmostEqual(
            PATCH._verify_width_update_ema(0.75, 0, 2), 0.75 * 0.75
        )

    def test_ema_zero_draft_round_uses_the_max_one_guard(self):
        # accepted / max(1, drafted), literally: 0 / 1 == 0.
        self.assertAlmostEqual(
            PATCH._verify_width_update_ema(0.5, 0, 0), 0.75 * 0.5
        )

    def test_ema_is_aggregate_not_mean_of_ratios(self):
        # Draft rows are summed, so 1/2 + 3/4 gives 4/6, not the 0.625 mean.
        self.assertAlmostEqual(
            PATCH._verify_width_update_ema(None, 4, 6), 4.0 / 6.0
        )
        self.assertNotAlmostEqual(
            PATCH._verify_width_update_ema(None, 4, 6),
            (1 / 2 + 3 / 4) / 2,
        )

    def test_thresholds(self):
        target = PATCH._verify_width_target
        self.assertEqual(target(0.0, 4), 1)
        self.assertEqual(target(0.3499, 4), 1)
        self.assertEqual(target(0.35, 4), 2)
        self.assertEqual(target(0.5999, 4), 2)
        self.assertEqual(target(0.60, 4), 4)
        self.assertEqual(target(1.0, 4), 4)
        self.assertEqual(target(None, 4), 4, "no signal yet: config k")

    def test_cap_at_config(self):
        target = PATCH._verify_width_target
        self.assertEqual(target(0.5, 1), 1, "config 1 cannot go to 2")
        self.assertEqual(target(0.5, 2), 2)
        self.assertEqual(target(0.9, 2), 2)
        self.assertEqual(target(0.2, 1), 1)
        self.assertEqual(target(0.9, 0), 0, "no drafting: no width")

    def test_dwell_holds_the_current_width(self):
        nxt = PATCH._verify_width_next
        self.assertEqual(nxt(0.0, 4, 4, 0), 4)
        self.assertEqual(nxt(0.0, 4, 4, 1), 4)
        self.assertEqual(nxt(0.0, 4, 4, 3), 4)
        self.assertEqual(nxt(0.0, 4, 4, 4), 1)
        self.assertEqual(nxt(0.5, 4, 1, 3), 1)
        self.assertEqual(nxt(0.5, 4, 1, 4), 2)
        self.assertEqual(nxt(0.9, 4, 2, 3), 2)
        self.assertEqual(nxt(0.9, 4, 2, 4), 4)

    def test_custom_dwell(self):
        self.assertEqual(PATCH._verify_width_next(0.0, 4, 4, 1, dwell=2), 4)
        self.assertEqual(PATCH._verify_width_next(0.0, 4, 4, 2, dwell=2), 1)

    def test_generated_helper_agrees_with_policy(self):
        ns = load_helpers("1")
        for ema, accepted, drafted in (
            (None, 3, 4), (None, 0, 0), (0.5, 1, 4), (0.9, 4, 4)
        ):
            self.assertEqual(
                ns["_verify_width_update_ema"](ema, accepted, drafted),
                PATCH._verify_width_update_ema(ema, accepted, drafted),
            )
        for ema, config_k in ((None, 4), (0.2, 4), (0.5, 4), (0.7, 4), (0.5, 1)):
            self.assertEqual(
                ns["_verify_width_target"](ema, config_k),
                PATCH._verify_width_target(ema, config_k),
            )


class StateHelpers(unittest.TestCase):
    """_verify_width_init / _step / _finish_round against a stub scheduler."""

    def test_env_off_is_stock_and_does_not_track(self):
        for gate in (None, "0", "true"):
            ns = load_helpers(gate)
            sched = fake_sched()
            ns["_verify_width_init"](sched)
            self.assertFalse(sched._vw_active, gate)
            self.assertEqual(sched._vw_off_reason, "env off")
            self.assertEqual(ns["_verify_width_step"](sched), 4)
            sched._vw_accepted, sched._vw_drafted = 0, 4
            ns["_verify_width_finish_round"](sched)
            self.assertIsNone(sched._vw_ema)

    def test_data_parallel_forces_config_width(self):
        ns = load_helpers("1")
        sched = fake_sched(dp=2)
        ns["_verify_width_init"](sched)
        self.assertFalse(sched._vw_active)
        self.assertIn("data parallel", sched._vw_off_reason)
        sched._vw_ema = 0.0
        self.assertEqual(ns["_verify_width_step"](sched), 4)
        self.assertEqual(ns["_verify_width_step"](sched), 4)

    def test_stock_per_batch_size_table_forces_config_width(self):
        ns = load_helpers("1")
        sched = fake_sched(table=[4, 3, 2, 1])
        ns["_verify_width_init"](sched)
        self.assertFalse(sched._vw_active)
        self.assertIn(
            "num_speculative_tokens_per_batch_size", sched._vw_off_reason
        )
        sched._vw_ema = 0.0
        self.assertEqual(ns["_verify_width_step"](sched), 4)

    def test_no_speculative_tokens_is_inactive(self):
        ns = load_helpers("1")
        sched = fake_sched(config_k=0)
        ns["_verify_width_init"](sched)
        self.assertFalse(sched._vw_active)

    def test_starts_at_config_then_changes_on_the_fourth_call(self):
        ns = load_helpers("1")
        sched = fake_sched()
        ns["_verify_width_init"](sched)
        self.assertTrue(sched._vw_active)
        widths = []
        for _ in range(4):
            sched._vw_accepted, sched._vw_drafted = 0, 4
            ns["_verify_width_finish_round"](sched)
            widths.append(ns["_verify_width_step"](sched))
        self.assertEqual(widths, [4, 4, 4, 1], "dwell holds, then the EMA wins")

    def test_recovers_to_config_k_after_another_dwell(self):
        ns = load_helpers("1")
        sched = fake_sched()
        ns["_verify_width_init"](sched)
        for _ in range(4):
            sched._vw_accepted, sched._vw_drafted = 0, 4
            ns["_verify_width_finish_round"](sched)
            ns["_verify_width_step"](sched)
        widths = []
        for _ in range(4):
            sched._vw_accepted, sched._vw_drafted = 4, 4
            ns["_verify_width_finish_round"](sched)
            widths.append(ns["_verify_width_step"](sched))
        self.assertEqual(widths, [1, 1, 1, 4])

    def test_round_without_drafts_is_skipped(self):
        ns = load_helpers("1")
        sched = fake_sched()
        ns["_verify_width_init"](sched)
        ns["_verify_width_finish_round"](sched)
        self.assertIsNone(sched._vw_ema)
        self.assertEqual(sched._vw_accepted, 0)
        self.assertEqual(sched._vw_drafted, 0)


class PatcherSource(unittest.TestCase):
    """Docstring statements and fail-closed pins that hold without an orig."""

    def setUp(self):
        self.src = PATCHER.read_text()
        self.doc = ast.get_docstring(ast.parse(self.src)) or ""

    def test_docstring_states_the_policy(self):
        self.assertIn("vllm/v1/core/sched/scheduler.py", self.doc)
        self.assertIn("num_spec_tokens_to_schedule", self.doc)
        self.assertIn("ratio < 0.35", self.doc)
        self.assertIn("ratio < 0.60", self.doc)
        self.assertIn("alpha 0.25", self.doc)
        self.assertIn("4 schedule() calls", self.doc)
        self.assertIn("config k", self.doc)

    def test_docstring_states_the_guards(self):
        self.assertIn("VLLM_VERIFY_WIDTH_ADAPTIVE", self.doc)
        self.assertIn("data parallelism > 1", self.doc)
        self.assertIn("num_speculative_tokens_per_batch_size", self.doc)
        self.assertIn("one step later", self.doc)
        self.assertIn("Lossless", self.doc)
        self.assertIn("target", self.doc)

    def test_fail_closed_messages(self):
        self.assertIn("already patched", self.src)
        self.assertIn("not unique/missing", self.src)
        self.assertIn("does not parse", self.src)
        self.assertIn("missing", self.src)
        self.assertIn("count != 1", self.src)
        self.assertIn("ast.parse", self.src)

    def test_runtime_uses_the_same_scalar_everywhere(self):
        for marker in (
            "padded_num_tokens = 1 + _verify_width_k",
            "num_new_tokens != 1 + _verify_width_k",
            "assert num_new_tokens == 1 + _verify_width_k",
            "] * _verify_width_k",
            "num_spec_tokens_to_schedule = _verify_width_k",
        ):
            self.assertIn(marker, self.src)

    def test_wiring_pins(self):
        patches = (REPO / "engine" / "patches.sh").read_text()
        self.assertIn("patch_verify_width.py", patches)
        self.assertIn('-e VLLM_VERIFY_WIDTH_ADAPTIVE=1', patches)
        self.assertIn("$VW/scheduler_v030.py", patches)
        config = (REPO / "engine" / "config.sh").read_text()
        self.assertIn("VERIFY_WIDTH_ADAPTIVE=", config)
        self.assertIn("verify_width_adaptive is only supported", config)
        self.assertIn("verify_width_adaptive needs speculative decoding", config)
        self.assertIn("mtp_adaptive_depth is inert", config)
        self.assertIn(
            "both overlay v1/core/sched/scheduler.py", config
        )
        self.assertIn("patches/verify_width/", (REPO / ".gitignore").read_text())

    def test_recipe_key_matches_the_measured_default(self):
        out = subprocess.run(
            [sys.executable, str(REPO / "engine" / "recipe.py"),
             str(REPO / "recipes"), "prod"],
            capture_output=True, text=True, check=True,
        ).stdout
        # enabled after the gate A/B (decode -3.1..-4.1% ms/step)
        self.assertIn("export VERIFY_WIDTH_ADAPTIVE='true'", out)
        yaml = (REPO / "recipes" / "prod.yaml").read_text()
        self.assertIn("verify_width_adaptive: true", yaml)
        self.assertIn("Lossless:", yaml)
        self.assertIn("target verification unchanged", yaml)

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
        out = prepare(self.tmp, fake_scheduler_src())
        r = run_patcher(out)
        self.assertEqual(r.returncode, 0, r.stderr)
        patched = (out / "scheduler_v030.py").read_text()
        ast.parse(patched)
        self.assertEqual(patched.count("def _verify_width_init("), 1)
        self.assertEqual(patched.count("_verify_width_init(self)"), 1)
        self.assertEqual(
            patched.count("_verify_width_k = _verify_width_step(self)"), 1
        )
        self.assertEqual(patched.count("1 + _verify_width_k"), 3)
        self.assertEqual(patched.count("] * _verify_width_k"), 1)
        self.assertEqual(
            patched.count("num_spec_tokens_to_schedule = _verify_width_k"), 1
        )
        self.assertEqual(patched.count("self._vw_accepted += num_accepted"), 1)
        self.assertEqual(patched.count("self._vw_drafted += num_draft_tokens"), 1)
        self.assertEqual(patched.count("_verify_width_finish_round(self)"), 1)
        self.assertEqual(patched.count("import os"), 1)

    def test_already_patched_fails(self):
        out = prepare(self.tmp, fake_scheduler_src())
        self.assertEqual(run_patcher(out).returncode, 0)
        again = prepare(self.tmp / "again", (out / "scheduler_v030.py").read_text())
        r = run_patcher(again)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already patched", r.stderr)

    def test_anchor_drift_fails_closed(self):
        drifted = fake_scheduler_src().replace(
            "padded_num_tokens = 1 + self.num_spec_tokens",
            "padded_num_tokens = 1 +  self.num_spec_tokens",
        )
        out = prepare(self.tmp, drifted)
        r = run_patcher(out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("anchor", r.stderr)
        self.assertFalse((out / "scheduler_v030.py").exists())

    def test_missing_orig_fails(self):
        r = run_patcher(self.tmp / "nothing")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("missing", r.stderr)


@unittest.skipUnless(
    ORIG.is_file(),
    "no extracted v0.32 scheduler.py; extract it with docker cp or a launch",
)
class AppliesToRealOrig(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.out = prepare(self.tmp, ORIG.read_text())

    def test_applies_and_parses(self):
        r = run_patcher(self.out)
        self.assertEqual(r.returncode, 0, r.stderr)
        patched = (self.out / "scheduler_v030.py").read_text()
        ast.parse(patched)
        self.assertEqual(patched.count("def _verify_width_init("), 1)
        self.assertEqual(patched.count("1 + _verify_width_k"), 3)
        self.assertEqual(
            patched.count("num_spec_tokens_to_schedule = _verify_width_k"), 1
        )
        self.assertEqual(patched.count("import os"), 1)

    def test_rerun_fails_closed(self):
        self.assertEqual(run_patcher(self.out).returncode, 0)
        again = prepare(
            self.tmp / "again", (self.out / "scheduler_v030.py").read_text()
        )
        r = run_patcher(again)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already patched", r.stderr)

    def test_width_local_is_assigned_before_every_use_in_schedule(self):
        self.assertEqual(run_patcher(self.out).returncode, 0)
        tree = ast.parse((self.out / "scheduler_v030.py").read_text())
        sched = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "schedule"
        )
        stores = [
            node for node in ast.walk(sched)
            if isinstance(node, ast.Name) and node.id == "_verify_width_k"
            and isinstance(node.ctx, ast.Store)
        ]
        loads = [
            node for node in ast.walk(sched)
            if isinstance(node, ast.Name) and node.id == "_verify_width_k"
            and isinstance(node.ctx, ast.Load)
        ]
        self.assertEqual(len(stores), 1)
        self.assertEqual(len(loads), 5)
        self.assertLess(stores[0].lineno, min(node.lineno for node in loads))


if __name__ == "__main__":
    unittest.main()
