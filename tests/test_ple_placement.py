"""Resolve launcher arguments through the extracted v0.32 PLE selector."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import test_model_source as fixtures

REPO = Path(__file__).resolve().parents[1]
ORIG = REPO / "patches/v030_ple/orig/ngram_nvidia.py"


@unittest.skipUnless(ORIG.is_file(), "requires extracted pinned-image ngram_nvidia.py")
class TestPlePlacement(unittest.TestCase):
    def test_effective_offload_selects_the_requested_table_backend(self):
        # Execute the real model's class-selection expression, not an asserted
        # copy of the launcher string or a fake Docker echo.
        tree = ast.parse(ORIG.read_text())
        selectors = [node for node in ast.walk(tree)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "embedding_cls"
                             for target in node.targets)]
        self.assertEqual(len(selectors), 1)
        selector = compile(ast.Module(body=selectors, type_ignores=[]), str(ORIG), "exec")
        pinned, device = object(), object()
        for nodes, override, expected in ((1, None, pinned), (2, None, device),
                                          (1, "false", device), (2, "true", pinned)):
            with self.subTest(nodes=nodes, offload=override):
                fixture = fixtures.TestModelSource()
                fixture.setUp()
                try:
                    fixture.values.update(NODES=str(nodes), WORKER_IP="127.0.0.2",
                                          IFACE="eth0", IB_HCA="mlx5_0", IB_GID_INDEX="0")
                    if override is not None:
                        fixture.values["PLE_OFFLOAD"] = override
                    result = fixture.run_shell('''
MTP_NUM_SPECULATIVE_TOKENS=0
PLE_EMBEDDING_DTYPE=""
source "$SCRIPT_DIR/engine/args.sh"
# Reparse the rendered launch string exactly as the launch-script shell does.
bash -c "printf '%s\\0' $VLLM_ARGS_STR"
''')
                    self.assertEqual(result.returncode, 0, result.stderr)
                    argv = result.stdout.rstrip("\0").split("\0")
                    parser = argparse.ArgumentParser()
                    parser.add_argument("--engram-config", type=json.loads, default={})
                    args, _ = parser.parse_known_args(argv)
                    # The inspected EngramConfig defaults cpu_offload to True.
                    config = {"cpu_offload": True, **args.engram_config}
                    namespace = {
                        "engram_config": SimpleNamespace(**config),
                        "Qwen4ExpPLEPinnedHostEmbedding": pinned,
                        "Qwen4ExpPLEDeviceEmbedding": device,
                    }
                    exec(selector, namespace)
                    self.assertIs(namespace["embedding_cls"], expected)
                finally:
                    fixture.doCleanups()


if __name__ == "__main__":
    unittest.main()
