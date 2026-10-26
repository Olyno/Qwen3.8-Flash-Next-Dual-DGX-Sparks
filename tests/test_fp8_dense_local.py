#!/usr/bin/env python3
"""recipes/fp8.yaml wiring: fp8_dense with a local directory must route through
the model_path bind-mount machinery (config.sh), and the v030 overlay must
apply its diffs to stock-extracted files (the hc_down_silu chaining went away
when that patch shipped upstream)."""
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
        self.assertEqual(env["V030"], "true")  # v030 lane (vLLM v0.32.0)
        # no TP pin: config.sh derives tensor_parallel_size from NODES
        self.assertNotIn("TENSOR_PARALLEL_SIZE", env)
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

    def test_v030_overlay_diffs_present_and_standalone(self):
        sh = (REPO / "engine" / "patches.sh").read_text()
        for name in ("modelopt", "model", "hyperconnection", "mtp"):
            self.assertTrue((REPO / "overlays" / "fp8dense_v030" / f"{name}.diff").is_file(), name)
        self.assertIn('FD="$SCRIPT_DIR/overlays/fp8dense_v030"', sh)
        # All four diffs apply to the stock files extracted from the image;
        # the hc_down_silu chaining is gone (patch shipped upstream).
        self.assertIn('for f in modelopt model hyperconnection mtp; do', sh)
        self.assertIn('cp "$FD/orig/$f.py" "$FD/$f.py"', sh)
        self.assertNotIn('cp "$HC/model_v030.py"', sh)
        self.assertNotIn('$HC/hyperconnection_v030.py', sh)

    def test_diffs_apply_clean_to_extracted_stock(self):
        # git apply --check against the stock extracts once the boot has
        # pulled them (all four apply to plain v0.32.0 stock — the
        # hc_down_silu pre-pass is no longer needed).
        import shutil
        import subprocess
        import tempfile
        from pathlib import Path
        if not shutil.which("git"):
            self.skipTest("git not available")
        fd = REPO / "overlays" / "fp8dense_v030" / "orig"
        missing = [n for n in ("modelopt", "model", "hyperconnection", "mtp")
                   if not (fd / f"{n}.py").is_file()]
        if missing:
            self.skipTest(f"no stock extracts yet ({', '.join(missing)}); boot the fp8 recipe once")
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        for n in ("modelopt", "model", "hyperconnection", "mtp"):
            (tmp / f"{n}.py").write_text((fd / f"{n}.py").read_text())
            r = subprocess.run(["git", "apply", "-p2", "--check",
                                str(REPO / "overlays" / "fp8dense_v030" / f"{n}.diff")],
                               cwd=tmp, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{n}: {r.stderr}")

    def test_v030_composes_draft_vocab_but_refuses_fp8_draft_head(self):
        sh = (REPO / "engine" / "patches.sh").read_text()
        # fp8_dense + mtp_draft_vocab compose: the slice reads the fp8dense
        # mtp.py (its draft head is already quantized) and mounts its output.
        self.assertIn('DRAFT_IN="$SCRIPT_DIR/overlays/fp8dense_v030/mtp.py"', sh)
        self.assertIn(
            '--input "$DRAFT_IN" --output "$SCRIPT_DIR/patches/mtp_v030_patched.py"',
            sh,
        )
        # the self-quantizing FP8 draft head still cannot stack on fp8_dense,
        # and the refusal names the one supported order
        self.assertIn('FP8_DRAFT_HEAD and FP8_DENSE both quantize the draft lm_head', sh)
        self.assertIn("the supported order is fp8_dense (quantized head) then mtp_draft_vocab (slice)", sh)
        # and the fp8dense block still mounts its other three files from the loop
        self.assertIn('add_overlay "$FD/$f.py" "$fsrc"', sh)


if __name__ == "__main__":
    unittest.main()
