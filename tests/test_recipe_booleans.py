#!/usr/bin/env python3
"""engine/recipe.py boolean keys must emit the strings the shell consumers
compare against (`[[ "$V030" == "true" ]]`) and execute (`if $DO_DOWNLOAD`).
Emitting 1/0 instead silently disengaged every boolean a YAML recipe set."""
import subprocess, unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


class TestRecipeBooleans(unittest.TestCase):
    def emit(self, yaml_text):
        recipes = self.tmp / "recipes"
        recipes.mkdir()
        (recipes / "t.yaml").write_text(yaml_text)
        out = subprocess.run(
            ["python3", str(REPO / "engine" / "recipe.py"), str(recipes), "t"],
            capture_output=True, text=True, check=True).stdout
        env = {}
        for line in out.splitlines():
            k, v = line.removeprefix("export ").split("=", 1)
            env[k] = v.strip("'")
        return env

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)

    def test_booleans_emit_shell_true_false(self):
        env = self.emit('v030: true\nskip_ple_patch: true\nnfs_share: false\n')
        self.assertEqual(env["V030"], "true")
        self.assertEqual(env["SKIP_PLE_PATCH"], "true")
        self.assertEqual(env["NFS_SHARE"], "false")

    def test_numeric_aliases_still_accepted(self):
        env = self.emit("v030: 1\n")
        self.assertEqual(env["V030"], "true")


if __name__ == "__main__":
    unittest.main()
