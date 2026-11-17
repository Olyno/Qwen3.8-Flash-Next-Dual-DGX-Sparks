#!/usr/bin/env python3
"""Execute generated PLE provenance helpers on tiny safetensors, without torch."""
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPT = Path(__file__).resolve().parent.parent / "patches" / "patch_ple_mmap_v030.py"


class PleContentIdentity(unittest.TestCase):
    prefix = "model.layers.0.ple.ple_embedding.ngram_embedding"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.model = self.tmp / "checkpoint"
        self.model.mkdir()
        self.config = SimpleNamespace(
            model_config=SimpleNamespace(
                model=str(self.model), model_weights=None, revision="main"
            ),
            load_config=SimpleNamespace(download_dir=str(self.tmp / "hf")),
        )
        self.group = SimpleNamespace(world_size=1)
        self.logger = mock.Mock()
        spec = importlib.util.spec_from_file_location("ple_patcher", SCRIPT)
        self.patcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.patcher)
        # Exercise helper insertion as well as execution without importing vLLM.
        fixture = self.patcher.patch(
            "from abc import ABC, abstractmethod\n"
            "class Qwen4ExpPLEPinnedHostEmbedding(Qwen4ExpPLEEmbedding):\n"
            "    pass\n",
            [
                self.patcher.EDITS_COMMON[0],
                next(edit for edit in self.patcher.EDITS_COMMON
                     if edit[0].startswith("class Qwen4ExpPLEPinnedHostEmbedding(")),
            ],
            "CPU fixture",
        )
        tree = ast.parse(fixture)
        names = {
            "_ple_json_object", "_ple_checkpoint_root", "_ple_checkpoint_identity",
            "_ple_mmap_location", "_ple_mmap_fingerprint", "_ple_mmap_sidecar",
            "_ple_mmap_matches",
        }
        nodes = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
            or isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_PLE_MMAP_FORMAT"
                    for t in node.targets)
        ]
        self.ns = {
            "hashlib": hashlib, "json": json, "os": os, "logger": self.logger,
            "get_current_vllm_config": lambda: self.config,
            "get_etp_group": lambda: self.group,
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "fixture", "exec"),
             self.ns)
        parent = self.prefix.rsplit(".", 1)[0]
        self.tensors = {
            "model.layers.0.mlp.experts.0.weight": ("U8", [13], b"expert-values"),
            self.prefix + ".shard_0.weight": ("U8", [2, 2], b"abcd"),
            self.prefix + ".weight_scale": ("F32", [], b"\x00\x00\x80\x3f"),
            parent + ".layer_multipliers": ("I64", [1], b"\x01" + b"\x00" * 7),
            parent + ".ngram_heads_offsets": ("I64", [1], b"\x00" * 8),
            parent + ".ngram_heads_vocab_sizes": ("I64", [1], b"\x02" + b"\x00" * 7),
        }
        self.write_checkpoint()
        env = mock.patch.dict(os.environ, {"VLLM_PLE_MMAP_DIR": str(self.tmp / "cache")})
        env.start()
        self.addCleanup(env.stop)

    def write_checkpoint(self):
        header, payload = {}, bytearray()
        for name, (dtype, shape, data) in self.tensors.items():
            start = len(payload)
            payload.extend(data)
            header[name] = {
                "dtype": dtype, "shape": shape,
                "data_offsets": [start, len(payload)],
            }
        raw = json.dumps(header).encode()
        self.shard = self.model / "model.safetensors"
        self.shard.write_bytes(len(raw).to_bytes(8, "little") + raw + payload)
        self.index = self.model / "model.safetensors.index.json"
        self.index.write_text(json.dumps({
            "weight_map": {name: self.shard.name for name in self.tensors},
        }))

    def fingerprint(self, rank=0):
        path, identity = self.ns["_ple_mmap_location"](self.prefix, rank)
        return path, self.ns["_ple_mmap_fingerprint"](
            identity, (2, 2), "torch.uint8"
        )

    def persist_cache(self):
        path, fingerprint = self.fingerprint()
        Path(path).write_bytes(b"abcd")
        Path(self.ns["_ple_mmap_sidecar"](path)).write_text(json.dumps(fingerprint))
        return path, fingerprint

    def matches(self, path, fingerprint):
        return self.ns["_ple_mmap_matches"](path, fingerprint, 4)

    def load_fixture(self, ready):
        source = (
            "def load(self, weights):\n"
            "    embedding = self.ngram_embedding\n"
            "    shard_size = 2\n"
            "    for shard_index, loaded_weight in weights:\n"
            "        checkpoint_start = shard_index * shard_size\n"
            "        if True:\n"
            "                embedding.weight.weight_loader(\n"
            "                    embedding.weight,\n"
            "                    loaded_weight,\n"
            "                    checkpoint_start=checkpoint_start,\n"
            "                )\n"
        )
        edit = next(edit for edit in self.patcher.EDITS_NVIDIA
                    if "checkpoint_start=checkpoint_start" in edit[0])
        namespace = {}
        exec(self.patcher.patch(source, [edit], "loader fixture"), namespace)
        loader = mock.Mock()
        embedding = SimpleNamespace(
            weight=SimpleNamespace(weight_loader=loader), org_vocab_size=4,
            _mmap_path="table.bin", _mmap_ready=ready, _mmap_shards=set(),
        )
        namespace["load"](SimpleNamespace(ngram_embedding=embedding), [(0, b"wxyz")])
        return loader

    def test_same_path_index_shape_size_changed_payload_rejects(self):
        path, first = self.persist_cache()
        index, size = self.index.read_bytes(), self.shard.stat().st_size
        self.tensors[self.prefix + ".shard_0.weight"] = ("U8", [2, 2], b"wxyz")
        self.write_checkpoint()
        current_path, current = self.fingerprint()
        self.assertEqual(current_path, path)
        self.assertEqual(self.index.read_bytes(), index)
        self.assertEqual(self.shard.stat().st_size, size)
        self.assertNotEqual(first, current)
        self.assertFalse(self.matches(path, current))
        self.assertEqual(self.load_fixture(self.matches(path, current)).call_count, 1)

    def test_unchanged_payload_reuses_despite_metadata_and_expert_changes(self):
        path, first = self.persist_cache()
        self.tensors["model.layers.0.mlp.experts.0.weight"] = ("U8", [13], b"EXPERT-VALUES")
        self.write_checkpoint()
        (self.model / "config.json").write_text('{"changed": true}')
        self.index.write_text(json.dumps(json.loads(self.index.read_text()), indent=2))
        current_path, current = self.fingerprint()
        self.assertEqual((current_path, current), (path, first))
        self.assertTrue(self.matches(path, current))
        self.assertEqual(self.load_fixture(self.matches(path, current)).call_count, 0)

    def test_scales_and_hash_buffers_are_content_bound(self):
        _, original = self.fingerprint()
        for name in self.tensors:
            if name.startswith("model.layers.0.mlp.") or name.endswith(".weight"):
                continue
            with self.subTest(tensor=name):
                dtype, shape, payload = self.tensors[name]
                self.tensors[name] = (dtype, shape, bytes([payload[0] ^ 1]) + payload[1:])
                self.write_checkpoint()
                self.assertNotEqual(self.fingerprint()[1], original)
                self.tensors[name] = (dtype, shape, payload)
        self.write_checkpoint()

    def test_rank_and_world_size_are_bound(self):
        _, one = self.fingerprint()
        self.group.world_size = 2
        path, zero = self.fingerprint(0)
        other_path, other = self.fingerprint(1)
        self.assertNotEqual(one, zero)
        self.assertNotEqual(zero, other)
        self.assertNotEqual(path, other_path)

    def test_model_weights_directory_takes_precedence(self):
        original = self.fingerprint()[1]
        self.config.model_config.model_weights = str(self.model)
        self.config.model_config.model = "not/the-checkpoint"
        self.assertEqual(self.fingerprint()[1], original)

    def test_hf_revision_uses_snapshot_not_blob_directory(self):
        original = self.fingerprint()[1]
        blobs = self.tmp / "blobs"
        blobs.mkdir()
        self.shard.rename(blobs / "payload")
        self.index.rename(blobs / "index")
        self.shard.symlink_to(blobs / "payload")
        self.index.symlink_to(blobs / "index")
        hub = SimpleNamespace(try_to_load_from_cache=mock.Mock(return_value=str(self.index)))
        self.config.model_config.model = "org/model"
        self.config.model_config.revision = "commit-id"
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
            self.assertEqual(self.fingerprint()[1], original)
        hub.try_to_load_from_cache.assert_called_once_with(
            "org/model", "model.safetensors.index.json", revision="commit-id",
            cache_dir=self.config.load_config.download_dir,
        )

    def test_hf_missing_cached_payload_fails_loudly(self):
        self.config.model_config.model = "org/model"
        hub = SimpleNamespace(try_to_load_from_cache=mock.Mock(return_value=None))
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
            with self.assertRaisesRegex(RuntimeError, "refusing mmap reuse"):
                self.fingerprint()

    def test_unindexed_checkpoint_uses_actual_payload(self):
        original = self.fingerprint()[1]
        self.index.unlink()
        self.assertEqual(self.fingerprint()[1], original)

    def test_old_metadata_only_cache_is_never_reused(self):
        path, fingerprint = self.persist_cache()
        old = {
            "format": "ple-mmap-v1", "model": str(self.model), "revision": "main",
            "snapshot": self.model.name, "shape": [2, 2], "dtype": "torch.uint8",
        }
        Path(self.ns["_ple_mmap_sidecar"](path)).write_text(json.dumps(old))
        self.assertFalse(self.matches(path, fingerprint))

    def test_malformed_and_truncated_cache_metadata_rebuilds_loudly(self):
        path, fingerprint = self.persist_cache()
        sidecar = Path(self.ns["_ple_mmap_sidecar"](path))
        for content in ('{"payload_sha256":', "not-json"):
            with self.subTest(content=content):
                sidecar.write_text(content)
                self.assertFalse(self.matches(path, fingerprint))
        self.assertEqual(self.logger.warning.call_count, 2)
        sidecar.write_text(json.dumps(fingerprint))
        Path(path).write_bytes(b"a")
        self.assertFalse(self.matches(path, fingerprint))

    def test_malformed_checkpoint_provenance_is_fatal(self):
        for content in (
            '{"weight_map":',
            '{"weight_map": {}}',
            '{"weight_map": {}, "weight_map": {}}',
            json.dumps({"weight_map": {self.prefix + ".shard_0.weight": "../bad.safetensors"}}),
        ):
            with self.subTest(index=content):
                self.index.write_text(content)
                with self.assertRaisesRegex(RuntimeError, "cannot verify checkpoint content"):
                    self.fingerprint()

    def test_truncated_checkpoint_payload_is_fatal_even_with_ready_cache(self):
        self.persist_cache()
        self.shard.write_bytes(self.shard.read_bytes()[:-1])
        with self.assertRaisesRegex(RuntimeError, "truncated PLE payload"):
            self.fingerprint()

    def test_malformed_and_truncated_safetensors_headers_are_fatal(self):
        for raw in (b"short", (1000).to_bytes(8, "little") + b"{}", b"\x02" + b"\x00" * 7 + b"[]"):
            with self.subTest(header=raw):
                self.shard.write_bytes(raw)
                with self.assertRaisesRegex(RuntimeError, "cannot verify checkpoint content"):
                    self.fingerprint()

    def test_missing_indexed_tensor_is_fatal(self):
        weight_map = json.loads(self.index.read_text())
        weight_map["weight_map"][self.prefix + ".shard_1.weight"] = self.shard.name
        self.index.write_text(json.dumps(weight_map))
        with self.assertRaisesRegex(RuntimeError, "index/header mismatch"):
            self.fingerprint()

    def test_claimed_hash_does_not_override_actual_bytes(self):
        path, first = self.persist_cache()
        (self.model / "manifest.json").write_text(json.dumps(first))
        self.tensors[self.prefix + ".shard_0.weight"] = ("U8", [2, 2], b"wxyz")
        self.write_checkpoint()
        self.assertFalse(self.matches(path, self.fingerprint()[1]))

    def test_streaming_reads_drop_source_pages_and_skip_expert_payloads(self):
        large = b"x" * ((8 << 20) + 17)
        self.tensors[self.prefix + ".shard_0.weight"] = ("U8", [len(large), 1], large)
        self.write_checkpoint()
        with mock.patch.object(os, "posix_fadvise", wraps=os.posix_fadvise) as advice:
            self.fingerprint()
        dropped = [
            call.args for call in advice.call_args_list
            if call.args[3] == os.POSIX_FADV_DONTNEED
        ]
        expected = sum(len(data) for name, (_, _, data) in self.tensors.items()
                       if not name.startswith("model.layers.0.mlp."))
        self.assertEqual(sum(call[2] for call in dropped), expected)
        self.assertTrue(all(0 < call[2] <= 8 << 20 for call in dropped))


if __name__ == "__main__":
    unittest.main()
