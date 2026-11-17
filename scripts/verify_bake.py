#!/usr/bin/env python3
"""Verify that a Lean bake changes only the declared BF16 lm_head rows.

The source snapshot must be independently trusted. The bake's SHA256MANIFEST.json
checks its declared files, not its authenticity. All other source model bytes
are compared directly, including untouched tensors in the edited shard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

CHUNK = 1 << 20
EDITED_JSON = {"generation_config.json", "overthinking_bake.json"}
MANIFEST = "SHA256MANIFEST.json"
DOCUMENTS = {"README.md", "REPRODUCTION.md"}


class VerificationError(ValueError):
    pass


def read_json(path):
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise VerificationError(f"expected a JSON object: {path.name}")
    return value


def filename(name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise VerificationError(f"invalid checkpoint filename: {name!r}")
    return name


def read_header(handle):
    prefix = handle.read(8)
    if len(prefix) != 8:
        raise VerificationError("truncated safetensors length")
    size = struct.unpack("<Q", prefix)[0]
    remaining = handle.seek(0, 2) - 8
    if not 0 < size <= min(remaining, 100_000_000):
        raise VerificationError("invalid safetensors header length")
    handle.seek(8)
    raw = handle.read(size)
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise VerificationError("invalid safetensors header")
    tensors = header
    tensors.pop("__metadata__", None)
    for name, info in tensors.items():
        if (not isinstance(info, dict) or not isinstance(info.get("data_offsets"), list)
                or len(info["data_offsets"]) != 2):
            raise VerificationError(f"invalid tensor entry: {name}")
    end = 0
    for name, info in sorted(tensors.items(), key=lambda item: item[1]["data_offsets"][0]):
        start, stop = info["data_offsets"]
        if (type(start) is not int or type(stop) is not int or start != end
                or stop < start or stop > remaining - size):
            raise VerificationError(f"invalid tensor offsets: {name}")
        end = stop
    if end != remaining - size:
        raise VerificationError("safetensors payload is truncated or has unindexed bytes")
    return tensors, prefix + raw, end


def marker_ids(token_map):
    ids = token_map.get("ids")
    if not isinstance(ids, dict) or not ids:
        raise VerificationError("token map needs a nonempty ids object")
    if any(type(value) is not int for value in ids.values()):
        raise VerificationError("token map IDs must be integers")
    return sorted(set(ids.values()))


def compare_file(source, baked, *, edited=False, ids=(), hash_output=True):
    """Compare bytes once, hashing the baked stream only when declared."""
    digest = hashlib.sha256() if hash_output else None
    with source.open("rb") as left, baked.open("rb") as right:
        if source.suffix == ".safetensors":
            before, _, payload_size = read_header(left)
            after, raw, after_size = read_header(right)
            if before != after or payload_size != after_size:
                raise VerificationError(f"tensor layout changed: {source.name}")
            if digest is not None:
                digest.update(raw)
            spans = []
            if edited:
                head = before.get("lm_head.weight", {})
                shape = head.get("shape", [])
                if head.get("dtype") != "BF16" or len(shape) != 2:
                    raise VerificationError("edited lm_head must be a two-dimensional BF16 tensor")
                rows, width = shape
                if (type(rows) is not int or type(width) is not int or rows <= 0 or width <= 0
                        or not ids or ids[0] < 0 or ids[-1] >= rows):
                    raise VerificationError("marker IDs or lm_head dimensions are out of bounds")
                start, stop = head["data_offsets"]
                row_bytes = width * 2
                if stop - start != rows * row_bytes:
                    raise VerificationError("lm_head shape does not match payload size")
                spans = [(start + row * row_bytes, start + (row + 1) * row_bytes) for row in ids]
        else:
            if source.stat().st_size != baked.stat().st_size:
                raise VerificationError(f"file size changed: {source.name}")
            payload_size = source.stat().st_size
            spans = []

        position = 0

        def consume(end, allow_change=False):
            nonlocal position
            changed = False
            while position < end:
                size = min(CHUNK, end - position)
                a, b = left.read(size), right.read(size)
                if len(a) != size or len(b) != size:
                    raise VerificationError(f"truncated payload: {source.name}")
                if digest is not None:
                    digest.update(b)
                if a != b:
                    if not allow_change:
                        raise VerificationError(f"undeclared payload change: {source.name} at data offset {position}")
                    changed = True
                position += size
            if allow_change and not changed:
                raise VerificationError(f"declared marker row did not change: {source.name} at {end}")

        for start, stop in spans:
            consume(start)
            consume(stop, allow_change=True)
        consume(payload_size)
        if left.read(1) or right.read(1):
            raise VerificationError(f"unexpected trailing bytes: {source.name}")
    return digest.hexdigest() if digest is not None else None


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(source, baked, token_map, shard):
    source_files = {p.name for p in source.iterdir() if p.is_file()}
    baked_files = {p.name for p in baked.iterdir() if p.is_file()}
    if source_files - baked_files:
        raise VerificationError(f"missing source files: {sorted(source_files - baked_files)}")
    extra = baked_files - source_files - EDITED_JSON - DOCUMENTS - {MANIFEST}
    if extra:
        raise VerificationError(f"unexpected baked files: {sorted(extra)}")
    shard = filename(shard)
    if shard not in source_files:
        raise VerificationError(f"edited shard is absent from source: {shard}")
    index = "model.safetensors.index.json"
    source_index = read_json(source / index).get("weight_map")
    baked_index = read_json(baked / index).get("weight_map")
    if not isinstance(source_index, dict) or source_index != baked_index:
        raise VerificationError("source and baked weight maps differ")
    for name in source_index.values():
        filename(name)
        if name not in source_files or name not in baked_files:
            raise VerificationError(f"indexed shard is missing: {name}")
    if source_index.get("lm_head.weight") != shard:
        raise VerificationError("edited shard does not contain the indexed lm_head.weight")

    ids = marker_ids(read_json(token_map))
    metadata = read_json(baked / "overthinking_bake.json")
    penalty = metadata.get("lambda")
    if type(penalty) not in (int, float) or not math.isfinite(penalty) or penalty <= 0:
        raise VerificationError("bake lambda must be finite and positive")
    bias = read_json(baked / "generation_config.json").get("logit_bias")
    if not isinstance(bias, dict) or set(bias) != {str(row) for row in ids}:
        raise VerificationError("logit_bias IDs differ from the token map")
    if any(type(value) not in (int, float) or not math.isfinite(value)
           or not math.isclose(value, -penalty, rel_tol=0, abs_tol=1e-9) for value in bias.values()):
        raise VerificationError("logit_bias values differ from the declared penalty")

    manifest = read_json(baked / MANIFEST)
    if not (EDITED_JSON | {shard}) <= set(manifest):
        raise VerificationError("manifest must cover the edited shard and bake metadata")
    for name, wanted in manifest.items():
        filename(name)
        if name == MANIFEST or name not in baked_files:
            raise VerificationError(f"invalid manifest member: {name}")
        if (not isinstance(wanted, str) or len(wanted) != 64
                or any(c not in "0123456789abcdefABCDEF" for c in wanted)):
            raise VerificationError(f"invalid SHA-256 digest: {name}")
    digests = {}
    compared = 0
    for name in sorted(source_files - EDITED_JSON - DOCUMENTS - {MANIFEST}):
        digests[name] = compare_file(source / name, baked / name, edited=name == shard,
                                     ids=ids, hash_output=name in manifest)
        compared += 1
    for name, wanted in manifest.items():
        actual = digests.get(name)
        if actual is None:
            actual = file_digest(baked / name)
        if actual.lower() != wanted.lower():
            raise VerificationError(f"manifest SHA-256 mismatch: {name}")
    return compared, len(ids)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--baked", required=True, type=Path)
    parser.add_argument("--token-map", required=True, type=Path)
    parser.add_argument("--shard", default="model-00009-of-00010.safetensors")
    args = parser.parse_args(argv)
    try:
        files, rows = verify(args.src, args.baked, args.token_map, args.shard)
    except (OSError, ValueError, TypeError, KeyError, struct.error) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: {files} source files verified; only {rows} declared lm_head rows changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
