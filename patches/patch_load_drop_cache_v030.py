#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Loader page-cache drop for the vLLM 0.30.0 lane (opt-in,
LOAD_DROP_CACHE=true).

Ports myllmbox/vllm@354bfc281aab ("Loader: drop each safetensors shard from
the page cache once consumed") onto the stock v0.30.0 file:

    vllm/model_executor/model_loader/weight_utils.py
    - safetensors_weights_iterator: posix_fadvise(POSIX_FADV_DONTNEED) on each
      shard file right after its tensors have been consumed (all three load
      strategies: eager, torchao, and the default lazy safe_open path).

On unified memory (GB10) the page cache and the GPU share one pool, and the
driver wants free pages, not reclaimable ones: a cache that grows while the
weights are copied can stall a host->GPU copy indefinitely. Dropping each
shard as it is consumed keeps the cache about one shard deep during the load
itself; engine/ step 4b-2 (scripts/evict_page_cache.py) already evicts the
checkpoint from the host side before launch, this covers the load window.

Changes vs the vendor commit: the helper is renamed and the gate is
VLLM_LOAD_DROP_CACHE, default OFF (fail closed; engine/patches.sh passes
-e VLLM_LOAD_DROP_CACHE=1 via OVERLAY_ENV when it mounts the file).

Inputs:  patches/v030_dropcache/orig/weight_utils.py (from the image)
Outputs: patches/v030_dropcache/weight_utils_v030.py
argv[0]/argv[1] override the orig/output directories (used by the test).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

HELPER_OLD = """\
def safetensors_weights_iterator(
    hf_weights_files: list[str],
"""
HELPER_NEW = """\
def _fadvise_dontneed(path: str) -> None:
    \"\"\"Drop one consumed checkpoint shard from the page cache.

    On unified memory (GB10) the page cache and the GPU share one pool;
    shards left cached can stall the next driver allocation. No-op unless
    VLLM_LOAD_DROP_CACHE=1.
    \"\"\"
    if os.environ.get("VLLM_LOAD_DROP_CACHE", "0") != "1":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def safetensors_weights_iterator(
    hf_weights_files: list[str],
"""
EAGER_OLD = """\
            for name, param in state_dict.items():
                if not should_skip_weight(name, local_expert_ids):
                    yield name, param
        elif safetensors_load_strategy == "torchao":
"""
EAGER_NEW = """\
            for name, param in state_dict.items():
                if not should_skip_weight(name, local_expert_ids):
                    yield name, param
            _fadvise_dontneed(st_file)
        elif safetensors_load_strategy == "torchao":
"""
TORCHAO_OLD = """\
            yield from unflattened_state_dict.items()
        else:
"""
TORCHAO_NEW = """\
            yield from unflattened_state_dict.items()
            _fadvise_dontneed(st_file)
        else:
"""
LAZY_OLD = """\
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    param = f.get_tensor(name)
                    yield name, param


def multi_thread_safetensors_weights_iterator(
"""
LAZY_NEW = """\
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    param = f.get_tensor(name)
                    yield name, param
            _fadvise_dontneed(st_file)


def multi_thread_safetensors_weights_iterator(
"""
HUNKS = ((HELPER_OLD, HELPER_NEW), (EAGER_OLD, EAGER_NEW),
         (TORCHAO_OLD, TORCHAO_NEW), (LAZY_OLD, LAZY_NEW))


def _apply(src, hunks, name):
    for i, (old, new) in enumerate(hunks):
        count = src.count(old)
        if count != 1:
            sys.exit(f"load_drop_cache_v030: anchor {i} in {name} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    return src


def main(argv) -> int:
    orig_dir = argv[0] if argv else os.path.join(HERE, "v030_dropcache", "orig")
    out_dir = argv[1] if len(argv) > 1 else os.path.join(HERE, "v030_dropcache")
    orig = os.path.join(orig_dir, "weight_utils.py")
    if not os.path.isfile(orig):
        sys.exit(f"ERROR: missing {orig} (start.sh extracts it from the image)")
    src = open(orig).read()
    if "_fadvise_dontneed" in src:
        sys.exit("ERROR: v030_dropcache orig is already patched")
    src = _apply(src, HUNKS, "weight_utils.py")
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"load_drop_cache_v030: patched weight_utils_v030.py does not parse: {exc}")
    out = os.path.join(out_dir, "weight_utils_v030.py")
    os.makedirs(out_dir, exist_ok=True)
    with open(out + ".tmp", "w") as f:
        f.write(src)
    os.replace(out + ".tmp", out)
    print(f"patched {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
