import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "patch_nvfp4_act_hooks.py"
ORIG = REPO / "patches" / "v030_nvfp4hooks" / "orig" / "nvfp4_emulation_moe.py"


@unittest.skipUnless(ORIG.is_file(), "no extracted v0.30 nvfp4_emulation_moe.py; boot once with nvfp4_act_hooks: true")
class Nvfp4ActHooksV030(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.orig = self.tmp / "orig"
        self.out = self.tmp / "out"
        self.orig.mkdir()
        shutil.copy(ORIG, self.orig / "nvfp4_emulation_moe.py")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_patch(self):
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.orig), str(self.out)],
            capture_output=True, text=True)

    def test_adds_fail_closed_capture(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        src = (self.out / "nvfp4_emulation_moe_v030.py").read_text()
        ast.parse(src)
        self.assertIn('_ACT_HOOK_ARM = "/tmp/nvfp4_act_hooks.arm"', src)
        self.assertIn('os.environ.get("VLLM_NVFP4_ACT_HOOKS", "0") == "1"', src)
        self.assertIn("_act_hook_capture(self._act_hook_idx, hidden_states,", src)
        self.assertIn("gate_up_proj.input", src)
        self.assertIn("down_proj.input", src)
        self.assertEqual(src.count("_ACT_HOOK_LAYERS.append(self)"), 1)

    def test_rerun_gives_same_output(self):
        self.run_patch()
        first = (self.out / "nvfp4_emulation_moe_v030.py").read_text()
        self.assertEqual(self.run_patch().returncode, 0)
        self.assertEqual((self.out / "nvfp4_emulation_moe_v030.py").read_text(), first)

    def test_moved_anchor_fails_without_writing(self):
        orig = self.orig / "nvfp4_emulation_moe.py"
        orig.write_text(orig.read_text().replace(
            "self.quantization_emulation = True",
            "self.quantization_emulation = True  "))
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.out / "nvfp4_emulation_moe_v030.py").exists())


if __name__ == "__main__":
    unittest.main()
