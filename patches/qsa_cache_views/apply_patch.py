#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""QSA key-cache view lifetime for the vLLM 0.30.0 lane (unconditional).

Backports vllm-project/vllm#58961 onto the stock v0.30.0 file:

    vllm/models/qwen4_exp/common/qsa_cache.py
    - QSAKeyStateCache.bind_kv_cache stored key_cache / rope_position_cache
      as persistent views of the bound KV storage. clear_layer_kv_caches
      (v1/worker/utils.py, called on model.modules() after CUDA-graph memory
      profiling and on teardown) resets only layer.kv_cache, so those views
      kept the whole block referenced and the freed storage stayed pinned in
      the allocator — real memory on a unified-memory box. The views are now
      derived on access via @property, leaving kv_cache the only reference.

Quality-neutral (same tensors, same values; only the binding lifetime
changes), so no toggle. Nothing in the tree assigns either attribute outside
bind_kv_cache and no parent defines them, so the read-only properties are
safe. Composes with the fused-draft overlay
(patches/patch_qsa_fused_draft_v030.py, vllm#58449): that patch's hunks sit
in the metadata builder, far from bind_kv_cache, so engine/patches.sh feeds
its output as this patcher's input when QSA_FUSED_DRAFT is on.

Inputs:  patches/qsa_cache_views/orig/qsa_cache.py (extracted from the image,
         or the fused-draft overlay's output)
Outputs: patches/qsa_cache_views/qsa_cache_v030.py
argv[0]/argv[1] override the orig/output directories (used by the test).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

BIND_OLD = """\
    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        super().bind_kv_cache(kv_cache)
        qsa_cache = self.kv_cache
        self.key_cache = qsa_cache[..., : self.key_head_size]
        if self.cache_rope_positions:
            position_tail = qsa_cache[..., self.rope_position_offset :]
            self.rope_position_cache = position_tail.view(torch.int64)
        else:
            self.rope_position_cache = None
"""
BIND_NEW = """\
    # Derived on access so `kv_cache` stays the only reference to the bound
    # storage: clearing it (e.g. after CUDA graph memory profiling) must free
    # the cache, or the freed block stays pinned in the allocator.
    @property
    def key_cache(self) -> torch.Tensor:
        return self.kv_cache[..., : self.key_head_size]

    @property
    def rope_position_cache(self) -> torch.Tensor | None:
        if not self.cache_rope_positions:
            return None
        return self.kv_cache[..., self.rope_position_offset :].view(torch.int64)
"""
HUNKS = ((BIND_OLD, BIND_NEW),)


def main(argv) -> int:
    orig_dir = argv[0] if argv else os.path.join(HERE, "orig")
    out_dir = argv[1] if len(argv) > 1 else HERE
    orig_path = os.path.join(orig_dir, "qsa_cache.py")
    if not os.path.isfile(orig_path):
        sys.exit(f"ERROR: missing {orig_path} "
                 "(start.sh extracts it from the image)")
    src = open(orig_path).read()
    if "Derived on access" in src:
        sys.exit("ERROR: qsa_cache_views orig qsa_cache.py is already patched")
    for i, (old, new) in enumerate(HUNKS):
        count = src.count(old)
        if count != 1:
            sys.exit(f"qsa_cache_views: anchor {i} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"qsa_cache_views: patched qsa_cache_v030.py does not parse: {exc}")
    out = os.path.join(out_dir, "qsa_cache_v030.py")
    open(out, "w").write(src)
    print(f"patched {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
