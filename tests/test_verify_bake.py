"""Verify allowed bake edits and reject independently re-hashed corruption."""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts" / "verify_bake.py"
SHARD = "model-00009-of-00010.safetensors"
OTHER = "model-00001-of-00010.safetensors"


def tensor_file(path, tensors):
    header = {}
    payload = bytearray()
    for name, shape, data in tensors:
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": "BF16", "shape": shape,
                        "data_offsets": [start, len(payload)]}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class TestVerifyBake(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.baked = self.root / "source", self.root / "baked"
        self.source.mkdir()
        tensor_file(self.source / SHARD, [
            ("lm_head.weight", [3, 2], bytes(range(12))),
            ("embed_tokens.weight", [2, 2], bytes(range(20, 28))),
        ])
        tensor_file(self.source / OTHER, [("layer.weight", [2, 2], b"abcdefgh")])
        (self.source / "config.json").write_text('{"model_type":"example"}')
        (self.source / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"lm_head.weight": SHARD, "embed_tokens.weight": SHARD,
                           "layer.weight": OTHER}}))
        (self.source / "generation_config.json").write_text("{}")
        shutil.copytree(self.source, self.baked)
        self.mutate(SHARD, 4)
        (self.baked / "generation_config.json").write_text('{"logit_bias":{"1":-2.0}}')
        (self.baked / "overthinking_bake.json").write_text('{"lambda":2.0}')
        self.mapping = self.root / "map.json"
        self.mapping.write_text('{"ids":{"marker":1}}')
        self.refresh_manifest()

    def refresh_manifest(self):
        # Re-sign even the corrupt cases: the source comparison must catch them.
        manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in self.baked.iterdir() if p.name != "SHA256MANIFEST.json"}
        (self.baked / "SHA256MANIFEST.json").write_text(json.dumps(manifest))

    def mutate(self, name, offset):
        path = self.baked / name
        raw = bytearray(path.read_bytes())
        base = 8 + struct.unpack("<Q", raw[:8])[0]
        raw[base + offset] ^= 0xFF
        path.write_bytes(raw)

    def run_verifier(self):
        return subprocess.run([sys.executable, str(VERIFIER), "--src", str(self.source),
                               "--baked", str(self.baked), "--token-map", str(self.mapping)],
                              capture_output=True, text=True, timeout=15)

    def assert_rejected(self, message):
        result = self.run_verifier()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(message, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_declared_marker_change_passes_cli(self):
        result = self.run_verifier()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("only 1 declared lm_head rows changed", result.stdout)

    def test_non_marker_head_corruption_rejected_despite_matching_manifest(self):
        self.mutate(SHARD, 0)
        self.refresh_manifest()
        self.assert_rejected("undeclared payload change")

    def test_other_tensor_in_edited_shard_rejected_despite_matching_manifest(self):
        self.mutate(SHARD, 12)
        self.refresh_manifest()
        self.assert_rejected("undeclared payload change")

    def test_untouched_shard_payload_rejected_despite_matching_manifest(self):
        self.mutate(OTHER, 0)
        self.refresh_manifest()
        self.assert_rejected("undeclared payload change")

    def test_unchanged_declared_marker_rejected(self):
        self.mutate(SHARD, 4)
        self.refresh_manifest()
        self.assert_rejected("declared marker row did not change")

    def test_rehashed_model_config_change_rejected(self):
        (self.baked / "config.json").write_text('{"model_type":"changed"}')
        self.refresh_manifest()
        self.assert_rejected("undeclared payload change")

    def test_invalid_marker_ids_rejected(self):
        for value in (-1, 3, True, "1"):
            with self.subTest(value=value):
                self.mapping.write_text(json.dumps({"ids": {"marker": value}}))
                self.assert_rejected("IDs")

    def test_manifest_cannot_omit_baked_changes(self):
        path = self.baked / "SHA256MANIFEST.json"
        manifest = json.loads(path.read_text())
        del manifest[SHARD]
        path.write_text(json.dumps(manifest))
        self.assert_rejected("manifest must cover")

    def test_manifest_path_traversal_is_rejected(self):
        path = self.baked / "SHA256MANIFEST.json"
        manifest = json.loads(path.read_text())
        manifest["../outside"] = "a" * 64
        path.write_text(json.dumps(manifest))
        self.assert_rejected("invalid checkpoint filename")

    def test_truncated_tensor_file_rejected_even_if_manifest_refreshed(self):
        path = self.baked / OTHER
        path.write_bytes(path.read_bytes()[:-1])
        self.refresh_manifest()
        self.assert_rejected("invalid tensor offsets")

    def test_incorrect_bias_rejected_even_if_manifest_refreshed(self):
        (self.baked / "generation_config.json").write_text('{"logit_bias":{"1":-4.0}}')
        self.refresh_manifest()
        self.assert_rejected("declared penalty")


if __name__ == "__main__":
    unittest.main()
