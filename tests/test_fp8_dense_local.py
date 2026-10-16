#!/usr/bin/env python3
"""recipes/fp8.yaml wiring: fp8_dense with a local directory must route through
the model_path bind-mount machinery (config.sh), and the v030 overlay must
chain off the hc_down_silu outputs instead of double-mounting those paths."""
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
        self.assertNotIn("MODEL_ID", env)  # no Hub id — local dir
        self.assertEqual(env["V030"], "true")               # vLLM 0.30 lane
        self.assertEqual(env["TENSOR_PARALLEL_SIZE"], "1")  # single-node
        # the reduced draft head assumes a BF16 draft lm_head; this
        # checkpoint's draft head is FP8 — the combination is refused.
        self.assertNotIn("MTP_DRAFT_VOCAB", env)

    def test_config_routes_local_fp8_through_model_path(self):
        sh = (REPO / "engine" / "config.sh").read_text()
        self.assertIn('MODEL_PATH="${FP8_DENSE_MODEL_ID/#\\~/$HOME}"', sh)
        self.assertIn('[[ "$FP8_DENSE" == "true" && -z "${FP8_DENSE_LOCAL:-}" ]]', sh)

    def test_legacy_overlay_stays_on_legacy_lane(self):
        sh = (REPO / "engine" / "patches.sh").read_text()
        self.assertNotIn("apply_patches.py", sh)  # deleted in the reorg
        self.assertNotIn("FP8_DENSE is not supported", sh)  # v030 hard-block lifted
        self.assertIn('[[ "$FP8_DENSE" == "true" && "$V030" != "true" ]]', sh)
        for name in ("modelopt", "model", "hyperconnection", "mtp"):
            self.assertTrue((REPO / "overlays" / "fp8dense" / f"{name}.diff").is_file(), name)
            self.assertIn(f'"{name}.py ', sh)

    def test_v030_overlay_diffs_present_and_chained(self):
        sh = (REPO / "engine" / "patches.sh").read_text()
        for name in ("modelopt", "model", "hyperconnection", "mtp"):
            self.assertTrue((REPO / "overlays" / "fp8dense_v030" / f"{name}.diff").is_file(), name)
        self.assertIn('FD="$SCRIPT_DIR/overlays/fp8dense_v030"', sh)
        # model.py / hyperconnection.py chain off the hc_down_silu outputs,
        # which skip their own mounts when fp8_dense claims those paths.
        self.assertIn('cp "$HC/model_v030.py" "$FD/model.py"', sh)
        self.assertIn('cp "$HC/hyperconnection_v030.py" "$FD/hyperconnection.py"', sh)
        self.assertIn('if [[ "$FP8_DENSE" != "true" ]]; then', sh)

    def test_v030_mtp_draft_head_combo_refused(self):
        sh = (REPO / "engine" / "patches.sh").read_text()
        # the draft-vocab/FP8 draft-head block must fail fast with fp8_dense
        self.assertIn('[[ "$FP8_DENSE" != "true" ]] || err "MTP_DRAFT_VOCAB/FP8_DRAFT_HEAD', sh)
        # and the fp8dense block mounts its own mtp.py unconditionally
        self.assertIn('add_overlay "$FD/mtp.py" "$NVIDIA_V030/mtp.py"', sh)


if __name__ == "__main__":
    unittest.main()
