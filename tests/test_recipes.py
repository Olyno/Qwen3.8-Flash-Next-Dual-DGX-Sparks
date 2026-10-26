#!/usr/bin/env python3
"""The shipped recipes must resolve to their contracts: prod (the default
boot) is the local checkpoint on the v030 lane (vLLM v0.32.0) with no
download; mia is the vendor day-0 reference (stock HF checkpoint, stock
image)."""
import subprocess, unittest
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

    def test_ple_prefill_prefetch_is_off_for_every_recipe(self):
        # The knob is derived in config.sh and no shipped recipe turns it on:
        # the overlay's prefetch stays a deliberate arm, not a default.
        config = (REPO / "engine" / "config.sh").read_text()
        self.assertIn(
            'PLE_PREFILL_PREFETCH="${PLE_PREFILL_PREFETCH:-false}"', config
        )
        for path in sorted((REPO / "recipes").glob("*.yaml")):
            env = load(path.stem)
            self.assertEqual(
                env.get("PLE_PREFILL_PREFETCH", "false"),
                "false",
                f"{path.name} must not enable the prefill PLE prefetch",
            )


if __name__ == "__main__":
    unittest.main()
