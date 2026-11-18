#!/usr/bin/env python3
"""The shipped recipes must resolve to their contracts: prod (the default
boot) is the local checkpoint on the v030 lane (vLLM v0.32.0) with no
download; mia is the vendor day-0 reference (stock HF checkpoint, stock
image)."""
import subprocess, unittest
import os
import shlex
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load(name):
    out = subprocess.run(
        ["python3", str(REPO / "engine" / "recipe.py"), str(REPO / "recipes"), name],
        capture_output=True, text=True, check=True).stdout
    env = {}
    for line in out.splitlines():
        k, v = line.removeprefix("export ").split("=", 1)
        env[k] = v.strip("'")
    return env


class TestRecipes(unittest.TestCase):
    def test_prod_is_lean_local_v030(self):
        env = load("prod")
        self.assertTrue(env["IMAGE"].startswith("vllm/vllm-openai:v0.32.0@sha256:"))
        self.assertEqual(env["V030"], "true")
        self.assertIn("Qwen3.8-Flash-Next-NVFP4-lean", env["MODEL_PATH"])
        self.assertEqual(env["DO_DOWNLOAD_DEFAULT"], "false")
        self.assertEqual(env["MTP_NUM_SPECULATIVE_TOKENS"], "4")
        # opt-in flags stay off by default; prod must not pin them
        self.assertNotIn("GDN_PREFILL_BACKEND", env)

    def test_mia_is_vendor_reference(self):
        env = load("mia")
        self.assertEqual(env["IMAGE"], "vllm/vllm-openai:qwen38-flash-next")
        self.assertEqual(env["MODEL_ID"], "nvidia/Qwen3.8-Flash-Next-NVFP4")
        self.assertNotIn("V030", env)
        self.assertNotIn("MODEL_PATH", env)


class TestPrefetchTopology(unittest.TestCase):
    def resolve(self, nodes, *, offload=None, prefetch=None, lane=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "engine").symlink_to(REPO / "engine", target_is_directory=True)
            (root / "chat_template_cod.jinja").symlink_to(REPO / "chat_template_cod.jinja")
            (root / "models/Qwen3.8-Flash-Next-NVFP4-lean").mkdir(parents=True)
            (root / "recipes").mkdir()
            recipe = (REPO / "recipes/prod.yaml").read_text()
            if prefetch is not None:
                recipe = "\n".join(
                    line for line in recipe.splitlines()
                    if not line.startswith("ple_prefill_prefetch:")
                ) + f"\nple_prefill_prefetch: {prefetch}\n"
            if lane is not None:
                # A minimal legacy recipe avoids unrelated v0.32-only flags.
                recipe = f"""model_id: test/model
image: test/image
gpu_memory_utilization: 0.5
max_num_batched_tokens: 4096
port: 8000
master_port: 29500
v030: {lane}
ple_prefill_prefetch: {prefetch}
"""
            (root / "recipes/prod.yaml").write_text(recipe)
            values = dict(NODES=str(nodes), HEAD_IP="127.0.0.1",
                          WORKER_IP="127.0.0.2", IFACE="eth0",
                          IB_HCA="mlx5_0", IB_GID_INDEX="0")
            if offload is not None:
                values["PLE_OFFLOAD"] = offload
            (root / ".env").write_text("".join(
                f"{key}={shlex.quote(value)}\n" for key, value in values.items()))
            return subprocess.run(
                ["bash", "-c", '''
set -euo pipefail
info() { :; }
warn() { :; }
err() { printf '%s\\n' "$*" >&2; exit 1; }
SCRIPT_DIR="$1"
set --
source "$SCRIPT_DIR/engine/config.sh"
printf '%s %s\\n' "$PLE_OFFLOAD" "$PLE_PREFILL_PREFETCH"
''', "bash", str(root)], cwd=root, capture_output=True, text=True,
                env={"PATH": os.environ["PATH"], "HOME": directory})

    def assert_resolved(self, nodes, expected, **settings):
        result = self.resolve(nodes, **settings)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), expected)

    def test_shipped_production_follows_topology(self):
        self.assert_resolved(1, "true true")
        self.assert_resolved(2, "false false")

    def test_auto_follows_explicit_offload_in_either_topology(self):
        self.assert_resolved(1, "false false", offload="false")
        self.assert_resolved(2, "true true", offload="true")

    def test_explicit_prefetch_false_is_preserved(self):
        self.assert_resolved(1, "true false", prefetch="false")

    def test_explicit_true_without_offload_remains_an_error(self):
        result = self.resolve(2, prefetch="true")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ple_prefill_prefetch needs ple_offload", result.stderr)

    def test_auto_is_inert_on_the_legacy_lane(self):
        self.assert_resolved(1, "true false", prefetch="auto", lane="false")

    def test_invalid_prefetch_mode_is_rejected(self):
        result = self.resolve(1, prefetch="typo")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ple_prefill_prefetch must be", result.stderr)


if __name__ == "__main__":
    unittest.main()
