#!/usr/bin/env python3
"""recipes/fp8.yaml wiring: fp8_dense with a local directory must route through
the model_path bind-mount machinery (config.sh), not the HF-download path."""
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


class TestFp8DenseLocal(unittest.TestCase):
    def test_recipe_emits_fp8_keys(self):
        out = subprocess.run(
            ["python3", str(REPO / "engine" / "recipe.py"),
             str(REPO / "recipes"), "fp8"],
            capture_output=True, text=True, check=True).stdout
        env = dict((k, v.strip("'")) for k, v in
                   (line.removeprefix("export ").split("=", 1)
                    for line in out.splitlines()))
        self.assertEqual(env["FP8_DENSE"], "true")
        self.assertEqual(env["FP8_DENSE_MODEL_ID"], "~/models/q38-lean-hyb")
        self.assertNotIn("MODEL_ID", env)         # no Hub id — local dir
        self.assertNotIn("MTP_DRAFT_VOCAB", env)  # conflicts with fp8_dense on mtp.py
        self.assertNotIn("V030", env)             # legacy lane only

    def test_config_routes_local_fp8_through_model_path(self):
        sh = (REPO / "engine" / "config.sh").read_text()
        self.assertIn('MODEL_PATH="${FP8_DENSE_MODEL_ID/#\\~/$HOME}"', sh)
        self.assertIn('[[ "$FP8_DENSE" == "true" && -z "${FP8_DENSE_LOCAL:-}" ]]', sh)

    def test_overlay_applies_committed_diffs(self):
        sh = (REPO / "engine" / "patches.sh").read_text()
        self.assertNotIn("apply_patches.py", sh)  # deleted in the reorg
        self.assertIn("git apply -p2", sh)
        for name in ("modelopt", "model", "hyperconnection", "mtp"):
            self.assertTrue((REPO / "overlays" / "fp8dense" / f"{name}.diff").is_file(), name)
            self.assertIn(f'"{name}.py ', sh)


if __name__ == "__main__":
    unittest.main()
