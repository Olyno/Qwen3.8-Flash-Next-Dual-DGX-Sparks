#!/usr/bin/env python3
"""Regression guards for the patch families dropped because vLLM v0.32.0
delivers them natively (vllm#55557 FP8 KV, #57097 fused QSA prepare,
#59533 merged QKV/indexer projections, #59945 head_dim, #57105 indexer
workspace, #58957 HC down+SiLU, #58114 PLE metadata syncs, #58961 key-cache
views, #59632/#59753/#60027 skinny plans) plus the retired ReplaySSM-GDN and
K3 lazy-GDN ports. The engine must not wire any of them again: the recipes
boot the merged projection and the rest as plain stock code."""
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DEAD_DIRS = (
    "patches/v030_fp8kv", "patches/qsa_prepare", "patches/qsa_head_dim",
    "patches/qsa_qkvg_merge", "patches/qsa_logits_workspace",
    "patches/hc_down_silu", "patches/ple_metadata_syncs",
    "patches/qsa_cache_views", "patches/gb10_skinny_gemm",
    "patches/replayssm_gdn",
)
DEAD_SCRIPTS = (
    "patches/patch_qsa_fp8_kv_v030.py", "patches/patch_gdn_lazy_v030.py",
    "patches/gdn_lazy_k3.py",
)
DEAD_TOKENS = (
    "patches/qsa_prepare", "patches/qsa_head_dim", "patches/qsa_qkvg_merge",
    "patches/qsa_logits_workspace", "patches/hc_down_silu",
    "patches/ple_metadata_syncs", "patches/qsa_cache_views",
    "patches/gb10_skinny_gemm", "patches/replayssm_gdn",
    "patches/patch_gdn_lazy", "patches/v030_fp8kv",
    "SKINNY_GEMM", "LAZY_GDN", "REPLAYSSM", "QSA_QKVG_MERGE", "58489",
)


class TestDroppedOverlayFamilies(unittest.TestCase):
    def test_directories_gone(self):
        for d in DEAD_DIRS:
            self.assertFalse((REPO / d).exists(), d)

    def test_standalone_scripts_gone(self):
        for s in DEAD_SCRIPTS:
            self.assertFalse((REPO / s).exists(), s)

    def test_engine_does_not_wire_them(self):
        for sh in ("engine/patches.sh", "engine/config.sh", "engine/launch.sh"):
            text = (REPO / sh).read_text()
            for tok in DEAD_TOKENS:
                self.assertNotIn(tok, text, f"{tok} still in {sh}")

    def test_recipes_boot_without_the_dropped_overlays(self):
        # The merged QKV/indexer projection (and everything else upstreamed)
        # must be reached as native code: no shipped recipe may re-arm a
        # dropped toggle.
        for name in ("prod", "fp8", "mia"):
            out = subprocess.run(
                ["python3", str(REPO / "engine" / "recipe.py"),
                 str(REPO / "recipes"), name],
                capture_output=True, text=True, check=True).stdout
            for tok in ("REPLAYSSM_GDN", "QSA_QKVG_MERGE", "LAZY_GDN",
                        "SKINNY_GEMM"):
                self.assertNotIn(f"export {tok}=", out, f"{name}: {tok}")

    def test_recipe_bool_keys_dropped(self):
        src = (REPO / "engine" / "recipe.py").read_text()
        self.assertNotIn("REPLAYSSM_GDN", src)

    def test_breakable_cudagraph_warn_guard_dropped(self):
        # vllm#58489 ships in the target image; the boot-time warning that
        # refused the combo must be gone, the env passthrough stays.
        sh = (REPO / "engine" / "patches.sh").read_text()
        self.assertNotIn("silent nondeterminism", sh)
        self.assertIn("-e VLLM_USE_BREAKABLE_CUDAGRAPH=", sh)


if __name__ == "__main__":
    unittest.main()
