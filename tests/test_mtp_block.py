"""CPU check for scripts/mtp_block.py, the pre-launch MTP legality guard.

The expected blocks are engine log values ("Setting attention block size to")
with the prod dtypes (mamba_ssm_cache_dtype=bfloat16, kv_cache_dtype=fp8):
1664 at k=3, 1680 at k=4 and 1728 at k=6.

    python3 tests/test_mtp_block.py
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "scripts" / "mtp_block.py"
spec = importlib.util.spec_from_file_location("mtp_block", SRC)
mtp_block = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mtp_block)

# Checkpoint values (Mia-AiLab/Qwen3.8-Flash-Next-NVFP4 config.json).
CFG = {"text_config": {
    "linear_key_head_dim": 128, "linear_num_key_heads": 16,
    "linear_value_head_dim": 128, "linear_num_value_heads": 48,
    "linear_conv_kernel_dim": 4, "hidden_size": 2560, "hc_count": 4,
    "ple_conv_kernel_size": 4, "ngram_size": 3, "num_key_value_heads": 2,
    "head_dim": 256, "indexer_compress_ratio": 4}}


def legal(k, block):
    return block % mtp_block.ring_capacity(k, 4) == 0


def run_cli(cfg_path, *args):
    return subprocess.run([sys.executable, SRC, cfg_path, *map(str, args)],
                          capture_output=True, text=True)


class DerivedBlock(unittest.TestCase):
    def test_engine_log_values(self):
        for k, want in ((3, 1664), (4, 1680), (6, 1728)):
            self.assertEqual(mtp_block.derived_block(CFG, k, "bfloat16", "fp8"), want, k)

    def test_k6_is_legal_and_k5_is_not(self):
        blocks = {k: mtp_block.derived_block(CFG, k, "bfloat16", "fp8") for k in range(17)}
        legal_ks = [k for k, b in blocks.items() if legal(k, b)]
        self.assertEqual(legal_ks, [0, 1, 2, 3, 4, 6, 9, 10, 11, 12, 16])

    def test_block_grows_with_k(self):
        blocks = [mtp_block.derived_block(CFG, k, "bfloat16", "fp8") for k in range(17)]
        self.assertEqual(blocks, sorted(blocks))
        self.assertLess(blocks[0], blocks[16])

    def test_fallback_block_is_one_k(self):
        # 848 is the block of k=4 at SSM bfloat16, KV auto.
        self.assertEqual(mtp_block.derived_block(CFG, 4, "bfloat16", "auto"), 848)
        self.assertNotEqual(mtp_block.derived_block(CFG, 6, "bfloat16", "auto"), 848)

    def test_dtypes_change_the_block(self):
        fp8 = mtp_block.derived_block(CFG, 4, "bfloat16", "fp8")
        self.assertLess(mtp_block.derived_block(CFG, 4, "bfloat16", "auto"), fp8)
        self.assertGreater(mtp_block.derived_block(CFG, 4, "", "fp8"), fp8)


class GuardCli(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = Path(self._tmp.name, "config.json")
        self.cfg.write_text(json.dumps(CFG))

    def tearDown(self):
        self._tmp.cleanup()

    def test_prod_combo_passes(self):
        # recipes/prod.yaml: k=4, max_num_seqs=4, max_num_batched_tokens=8192,
        # SSM bfloat16, KV fp8.
        rc = run_cli(self.cfg, 4, 4, 8192, "bfloat16", "fp8")
        self.assertEqual(rc.returncode, 0, rc.stderr)
        self.assertEqual(rc.stdout.strip(), "1680 4")

    def test_illegal_k_fails(self):
        # k=5: ring capacity 12 does not divide the derived block 1696.
        rc = run_cli(self.cfg, 5, 4, 8192, "bfloat16", "fp8")
        self.assertEqual(rc.returncode, 1)
        self.assertIn("illegal", rc.stderr)

    def test_k1_is_dominated(self):
        rc = run_cli(self.cfg, 1, 4, 8192, "bfloat16", "fp8")
        self.assertEqual(rc.returncode, 1)
        self.assertIn("dominated", rc.stderr)

    def test_widest_verify_batch_exceeds_budget(self):
        # (1+4)*4 = 20 > 16.
        rc = run_cli(self.cfg, 4, 4, 16, "bfloat16", "fp8")
        self.assertEqual(rc.returncode, 1)
        self.assertIn("max_num_batched_tokens", rc.stderr)

    def test_missing_keys_fail(self):
        self.cfg.write_text(json.dumps({"hidden_size": 2560}))
        rc = run_cli(self.cfg, 4, 4, 8192)
        self.assertEqual(rc.returncode, 1)
        self.assertEqual(rc.stdout, "")


if __name__ == "__main__":
    unittest.main()
