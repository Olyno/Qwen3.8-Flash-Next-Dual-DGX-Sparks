#!/usr/bin/env python3
"""Exercise recipe/config resolution and the actual Bash-rendered vLLM argv."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
FLAGS = ("--async-scheduling", "--no-async-scheduling")
BASE_RECIPE = """model_id: test/model
image: vllm/vllm-openai:qwen38-flash-next
gpu_memory_utilization: 0.5
max_model_len: 4096
max_num_seqs: 1
max_num_batched_tokens: 4096
port: 8000
master_port: 29500
mtp_num_speculative_tokens: 0
"""
BASH = r'''
set -euo pipefail
info() { :; }
warn() { :; }
err() { printf '%s\n' "$*" >&2; exit 1; }
SCRIPT_DIR="$1"
# start.sh consumes its CLI arguments, not the fixture root.
set --
source "$SCRIPT_DIR/engine/config.sh"
OVERLAY_ENV=()
PLE_EMBEDDING_DTYPE=""
source "$SCRIPT_DIR/engine/args.sh"
# launch.sh embeds this rendered string in both head and worker shell scripts.
eval "set -- $VLLM_ARGS_STR"
printf '%s\0' "$@"
'''


class TestAsyncScheduling(unittest.TestCase):
    def render(self, recipe="", env_values=None, valid=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "engine").symlink_to(REPO / "engine", target_is_directory=True)
            (root / "recipes").mkdir()
            (root / "recipes" / "test.yaml").write_text(BASE_RECIPE + recipe)
            values = {"HEAD_IP": "127.0.0.1", **(env_values or {})}
            (root / ".env").write_text("".join(
                f"{key}={shlex.quote(value)}\n" for key, value in values.items()))
            result = subprocess.run(
                ["bash", "-c", BASH, "bash", str(root)], cwd=root,
                env={"PATH": os.environ["PATH"], "HOME": directory,
                     "RECIPE": "test"}, capture_output=True)
        if not valid:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"async_scheduling must be true or false", result.stderr)
            return None
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertTrue(result.stdout.endswith(b"\0"))
        return result.stdout.decode().removesuffix("\0").split("\0")

    def scheduling_flags(self, argv):
        return [arg for arg in argv if arg in FLAGS]

    def test_true_and_false_are_explicit_on_both_model_lanes(self):
        for lane in ("true", "false"):
            for value, flag in (("true", FLAGS[0]), ("false", FLAGS[1])):
                with self.subTest(v030=lane, value=value):
                    argv = self.render(f"v030: {lane}\nasync_scheduling: {value}\n")
                    self.assertEqual(self.scheduling_flags(argv), [flag])

    def test_unset_defaults_to_explicit_disable(self):
        self.assertEqual(self.scheduling_flags(self.render()), [FLAGS[1]])

    def test_boolean_recipe_aliases_reach_bash(self):
        for value, flag in (("yes", FLAGS[0]), ("1", FLAGS[0]),
                            ("no", FLAGS[1]), ("0", FLAGS[1])):
            with self.subTest(value=value):
                self.assertEqual(self.scheduling_flags(self.render(
                    f"async_scheduling: {value}\n")), [flag])

    def test_recipe_wins_over_stale_env(self):
        for value, stale, flag in (("true", "false", FLAGS[0]),
                                   ("false", "true", FLAGS[1])):
            with self.subTest(value=value):
                argv = self.render(f"async_scheduling: {value}\n",
                                   {"ASYNC_SCHEDULING": stale})
                self.assertEqual(self.scheduling_flags(argv), [flag])

    def test_env_value_applies_when_recipe_omits_it(self):
        self.assertEqual(self.scheduling_flags(self.render(
            env_values={"ASYNC_SCHEDULING": "true"})), [FLAGS[0]])

    def test_extra_flags_remain_last_and_override_recipe_in_either_direction(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction)
        for value, generated, extra, effective in (
            ("true", FLAGS[0], FLAGS[1], False),
            ("false", FLAGS[1], FLAGS[0], True),
            ("true", FLAGS[0], f"{FLAGS[0]} {FLAGS[1]}", False),
        ):
            with self.subTest(value=value, extra=extra):
                argv = self.render(f"async_scheduling: {value}\n",
                                   {"EXTRA_VLLM_ARGS": extra})
                self.assertEqual(self.scheduling_flags(argv),
                                 [generated, *shlex.split(extra)])
                self.assertEqual(argv[-len(shlex.split(extra)):], shlex.split(extra))
                parsed, _ = parser.parse_known_args(argv)
                self.assertIs(parsed.async_scheduling, effective)

    def test_shipped_prod_preserves_enabled_intent(self):
        result = subprocess.run(
            ["python3", str(REPO / "engine" / "recipe.py"),
             str(REPO / "recipes"), "prod"], capture_output=True, text=True,
            check=True)
        exports = dict(shlex.split(line.removeprefix("export "))[0].split("=", 1)
                       for line in result.stdout.splitlines())
        self.assertEqual(exports["ASYNC_SCHEDULING"], "true")
        self.assertEqual(self.scheduling_flags(self.render(
            f"async_scheduling: {exports['ASYNC_SCHEDULING']}\n")), [FLAGS[0]])

    def test_auto_is_not_silently_treated_as_false(self):
        self.render("async_scheduling: auto\n", valid=False)
        self.render(env_values={"ASYNC_SCHEDULING": "auto"}, valid=False)


if __name__ == "__main__":
    unittest.main()
