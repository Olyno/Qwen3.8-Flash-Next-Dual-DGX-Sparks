import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "ple_metadata_syncs" / "apply_patch.py"
ORIG_DIR = REPO / "patches" / "ple_metadata_syncs" / "orig"
ORIGS = ("mamba_attn.py", "short_conv_attn.py", "ple_layer.py")
OUTS = ("mamba_attn_v030.py", "short_conv_attn_v030.py", "ple_layer_v030.py")


@unittest.skipUnless(
    all((ORIG_DIR / o).is_file() for o in ORIGS),
    "no extracted v0.30 origs; boot once on the v030 lane (engine/patches.sh "
    "extracts them) or stage patches/ple_metadata_syncs/orig/ from the image",
)
class PleMetadataSyncsV030(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.orig = self.tmp / "orig"
        self.out = self.tmp / "out"
        self.orig.mkdir()
        self.out.mkdir()
        for name in ORIGS:
            shutil.copy(ORIG_DIR / name, self.orig / name)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_patch(self):
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.orig), str(self.out)],
            capture_output=True, text=True)

    def test_all_outputs_parse_and_carry_markers(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        for name in OUTS:
            ast.parse((self.out / name).read_text())
        mamba = (self.out / "mamba_attn_v030.py").read_text()
        self.assertIn("needs_causal_conv1d_metadata: bool = True", mamba)
        self.assertIn("if self.needs_causal_conv1d_metadata:", mamba)
        sc = (self.out / "short_conv_attn_v030.py").read_text()
        self.assertIn("needs_causal_conv1d_metadata = False", sc)
        self.assertNotIn("compute_causal_conv1d_metadata", sc)
        self.assertNotIn("non_spec_query_start_loc_cpu", sc)
        self.assertIn("spec_state_indices_tensor = "
                      "block_table_tensor[:num_spec_decodes, 0]", sc)
        self.assertIn("spec_token_indx = None", sc)
        self.assertIn("num_accepted_tokens = "
                      "num_accepted_tokens[:num_spec_decodes]", sc)
        self.assertIn("num_accepted_tokens = "
                      "num_accepted_tokens[spec_req_idx]", sc)
        self.assertIn("req_group.index_fill_(0, spec_req_idx, 0)", sc)
        self.assertIn("output_size=int(query_start_loc_cpu[-1])", sc)
        ple = (self.out / "ple_layer_v030.py").read_text()
        self.assertIn("query_start_loc = metadata.query_start_loc_p", ple)
        self.assertNotIn("metadata.non_spec_query_start_loc", ple)

    def test_rerun_gives_same_output(self):
        self.assertEqual(self.run_patch().returncode, 0)
        first = {n: (self.out / n).read_text() for n in OUTS}
        self.assertEqual(self.run_patch().returncode, 0)
        for n in OUTS:
            self.assertEqual((self.out / n).read_text(), first[n])

    def test_already_patched_input_fails(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        # Feed the patcher its own short_conv_attn output: the marker check
        # must refuse rather than double-apply.
        shutil.copy(self.out / "short_conv_attn_v030.py",
                    self.orig / "short_conv_attn.py")
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already patched", r.stderr)

    def test_moved_anchor_fails_without_writing(self):
        orig = self.orig / "short_conv_attn.py"
        orig.write_text(orig.read_text().replace(
            "    supports_update_block_table = False",
            "    supports_update_block_table =  False"))
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.out / "short_conv_attn_v030.py").exists())


if __name__ == "__main__":
    unittest.main()
