import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "replayssm_gdn" / "apply_patch.py"
ORIG_DIR = REPO / "patches" / "replayssm_gdn" / "orig"
ORIGS = (
    "mamba_utils.py",
    "base.py",
    "abstract.py",
    "qwen_gdn_linear_attn.py",
    "gdn_attn.py",
    "model.py",
    "replayssm_config.py",
)
OUTS = (
    "mamba_utils_v030.py",
    "gdn_base_v030.py",
    "mamba_abstract_v030.py",
    "qwen_gdn_linear_attn_v030.py",
    "gdn_attn_v030.py",
    "qwen4_exp_model_v030.py",
)


@unittest.skipUnless(
    all((ORIG_DIR / o).is_file() for o in ORIGS),
    "no extracted v0.30 origs; boot once with REPLAYSSM_GDN=true (recipe replayssm_gdn: true)",
)
class ReplaySSMGdnV030(unittest.TestCase):
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
            src = (self.out / name).read_text()
            ast.parse(src)
            self.assertIn("replayssm", src.lower(), name)
        lin = (self.out / "qwen_gdn_linear_attn_v030.py").read_text()
        self.assertEqual(lin.count("def _replayssm_spec_ssm"), 1)
        # module path + import + call site
        self.assertGreaterEqual(lin.count("gdn_replayssm_spec_decode"), 3)
        self.assertIn("self.get_state_dtype()[:2]", lin)
        self.assertIn("state_dtype = self.get_state_dtype()[1]", lin)
        self.assertIn("not self.use_replayssm_spec", lin)
        att = (self.out / "gdn_attn_v030.py").read_text()
        self.assertIn("commit_gdn_replayssm_spec(", att)
        self.assertIn("reset_gdn_replayssm_spec_cursors(", att)
        self.assertIn("m.rswa_prefix_lens", att)
        self.assertIn("treat_short_extends_as_decodes=not self.use_replayssm_spec", att)
        mu = (self.out / "mamba_utils_v030.py").read_text()
        self.assertEqual(mu.count("gated_delta_net_replayssm_spec_state_dtype"), 1)
        self.assertEqual(mu.count("gated_delta_net_replayssm_spec_state_shape"), 1)

    def test_rerun_gives_same_output(self):
        self.assertEqual(self.run_patch().returncode, 0)
        first = {n: (self.out / n).read_text() for n in OUTS}
        self.assertEqual(self.run_patch().returncode, 0)
        for n in OUTS:
            self.assertEqual((self.out / n).read_text(), first[n])

    def test_moved_anchor_fails_without_writing(self):
        orig = self.orig / "gdn_attn.py"
        orig.write_text(orig.read_text().replace(
            "num_accepted_tokens=num_accepted_tokens,",
            "num_accepted_tokens = num_accepted_tokens,"))
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.out / "gdn_attn_v030.py").exists())


if __name__ == "__main__":
    unittest.main()
