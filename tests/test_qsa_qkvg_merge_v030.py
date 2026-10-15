import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "qsa_qkvg_merge" / "apply_patch.py"
PREPARE = REPO / "patches" / "qsa_prepare" / "apply_patch.py"
HEAD_DIM = REPO / "patches" / "qsa_head_dim" / "apply_patch.py"
FP8KV = REPO / "patches" / "patch_qsa_fp8_kv_v030.py"
PREPARE_ORIG = REPO / "patches" / "qsa_prepare" / "orig"
FP8KV_ORIG = REPO / "patches" / "v030_fp8kv" / "orig"


def run(script, orig, out):
    return subprocess.run(
        [sys.executable, str(script), str(orig), str(out)],
        capture_output=True, text=True)


@unittest.skipUnless(
    all((PREPARE_ORIG / o).is_file()
        for o in ("qsa.py", "indexer_qsa.py", "qsa_pre_indexer.py")),
    "no extracted v0.30 origs; boot once on the v030 lane (engine/patches.sh "
    "extracts them) or stage patches/qsa_prepare/orig/ from the image",
)
class QSAQKVGMergeV030(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # Chain: stock -> qsa_prepare -> qsa_head_dim -> orig for the merge.
        prep_orig = self.tmp / "prep_orig"
        prep_out = self.tmp / "prep_out"
        shutil.copytree(PREPARE_ORIG, prep_orig)
        prep_out.mkdir()
        r = run(PREPARE, prep_orig, prep_out)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.orig = self.tmp / "orig"
        self.out = self.tmp / "out"
        self.orig.mkdir()
        self.out.mkdir()
        hd_orig = self.tmp / "hd_orig"
        hd_out = self.tmp / "hd_out"
        hd_orig.mkdir()
        hd_out.mkdir()
        shutil.copy(prep_out / "qsa_v030.py", hd_orig / "qsa.py")
        r = run(HEAD_DIM, hd_orig, hd_out)
        self.assertEqual(r.returncode, 0, r.stderr)
        shutil.copy(hd_out / "qsa_v030.py", self.orig / "qsa.py")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_patch(self):
        return run(SCRIPT, self.orig, self.out)

    def test_output_parses_and_carries_markers(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        src = (self.out / "qsa_v030.py").read_text()
        ast.parse(src)
        # The merged projection class and its imports.
        self.assertIn(
            "class Qwen4ExpQSAQKVIndexerLinear(MergedColumnParallelLinear):", src)
        self.assertIn("from collections.abc import Callable, Iterable", src)
        self.assertIn("    MergedColumnParallelLinear,\n", src)
        self.assertIn(
            "from vllm.model_executor.models.utils import "
            "AutoWeightsLoader, WeightsMapper",
            src)
        # Upstream's LoRA gate and the indexer-projection merge in __init__.
        self.assertIn(
            "self.fuse_indexer_projection = vllm_config.lora_config is None", src)
        self.assertIn("self.index_qk_size = self.indexer.index_qk_proj.output_size",
                      src)
        self.assertIn("del self.indexer.index_qk_proj", src)
        # The attention-level weight remap (replaces the PR's model.py hunks).
        self.assertIn('"indexer.index_qk_proj.": ("qkv_proj.", 3)', src)
        self.assertIn(
            "return AutoWeightsLoader(self).load_weights(weights, mapper=mapper)",
            src)
        # The forward split / unfused fallback.
        self.assertIn(
            "qkv, projected_qk = qkv.split(\n"
            "                [2 * self.q_size + 2 * self.kv_size, self.index_qk_size],"
            " dim=-1\n            )",
            src)
        self.assertIn(
            "projected_qk, _ = self.indexer.index_qk_proj(hidden_states)", src)
        # The split feeds the fused prepare: qkv no longer carries the indexer
        # columns past this point.
        self.assertEqual(src.count("self.indexer.index_qk_proj(hidden_states)"), 1)

    def test_rerun_gives_same_output(self):
        self.assertEqual(self.run_patch().returncode, 0)
        first = (self.out / "qsa_v030.py").read_text()
        self.assertEqual(self.run_patch().returncode, 0)
        self.assertEqual((self.out / "qsa_v030.py").read_text(), first)

    def test_already_patched_input_fails(self):
        self.assertEqual(self.run_patch().returncode, 0)
        shutil.copy(self.out / "qsa_v030.py", self.orig / "qsa.py")
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already patched", r.stderr)

    def test_moved_anchor_fails_without_writing(self):
        orig = self.orig / "qsa.py"
        orig.write_text(orig.read_text().replace(
            "from vllm.model_executor.layers.linear import QKVParallelLinear,"
            " RowParallelLinear",
            "from vllm.model_executor.layers.linear import QKVParallelLinear,"
            "  RowParallelLinear"))
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.out / "qsa_v030.py").exists())

    @unittest.skipUnless(
        (FP8KV_ORIG / "qsa.py").is_file()
        and (FP8KV_ORIG / "ops" / "qsa.py").is_file(),
        "no extracted v0.30 fp8-kv origs (patches/v030_fp8kv/orig/)",
    )
    def test_full_chain_with_fp8_kv(self):
        # The launch-time chain under kv_cache_dtype: fp8 — stock -> fp8_kv ->
        # qsa_prepare -> qsa_head_dim -> qsa_qkvg_merge — must apply cleanly.
        fp8_orig = self.tmp / "fp8_orig"
        fp8_out = self.tmp / "fp8_out"
        shutil.copytree(FP8KV_ORIG, fp8_orig)
        fp8_out.mkdir()
        r = run(FP8KV, fp8_orig, fp8_out)
        self.assertEqual(r.returncode, 0, r.stderr)
        prep_orig = self.tmp / "fp8_prep_orig"
        prep_out = self.tmp / "fp8_prep_out"
        shutil.copytree(PREPARE_ORIG, prep_orig)
        shutil.copy(fp8_out / "qsa.py", prep_orig / "qsa.py")
        prep_out.mkdir()
        r = run(PREPARE, prep_orig, prep_out)
        self.assertEqual(r.returncode, 0, r.stderr)
        hd_orig = self.tmp / "fp8_hd_orig"
        hd_out = self.tmp / "fp8_hd_out"
        hd_orig.mkdir()
        hd_out.mkdir()
        shutil.copy(prep_out / "qsa_v030.py", hd_orig / "qsa.py")
        r = run(HEAD_DIM, hd_orig, hd_out)
        self.assertEqual(r.returncode, 0, r.stderr)
        qm_orig = self.tmp / "fp8_qm_orig"
        qm_orig.mkdir()
        shutil.copy(hd_out / "qsa_v030.py", qm_orig / "qsa.py")
        r = run(SCRIPT, qm_orig, self.out)
        self.assertEqual(r.returncode, 0, r.stderr)
        src = (self.out / "qsa_v030.py").read_text()
        ast.parse(src)
        self.assertIn("class Qwen4ExpQSAQKVIndexerLinear(", src)
        # The fp8-kv overlay's marker survives the whole chain.
        self.assertIn('"fp8_e4m3"', src)


if __name__ == "__main__":
    unittest.main()
