#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Free-list block-id sort for the vLLM 0.30.0 lane (opt-in, FREE_BLOCK_SORT).

Proposed upstream as vllm-project/vllm#31371 (pegaflow, RFC), Apache-2.0.

    vllm/v1/core/kv_cache_utils.py
    - FreeKVCacheBlockQueue.append_n appended freed blocks in eviction order,
      scattering their block ids across the free list. Freed blocks are now
      sorted by block_id before linking, so pops from the free list hand out
      address-sequential blocks: contiguous ids coalesce into fewer, larger
      DMA bursts on KV offload/prefetch.

Trade: the free list no longer preserves strict LRU eviction order within a
freed batch (prefix-cache eviction may pick a warmer block first). Structural
change, no numeric effect on emitted tokens; gated by the decode bench +
quality gate, not an exact-parity argument.

Inputs:  patches/free_block_sort/orig/kv_cache_utils.py
         (start.sh extracts it from the image)
Outputs: patches/free_block_sort/kv_cache_utils_v030.py
argv[0]/argv[1] override the orig/output directories (used by the test).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

APPEND_N_OLD = """\
        if len(blocks) == 0:
            return

        last_block = self.fake_free_list_tail.prev_free_block
"""
APPEND_N_NEW = """\
        if len(blocks) == 0:
            return

        # Address-sequential free list (vllm#31371): contiguous block ids
        # coalesce into larger DMA bursts on KV offload/prefetch.
        blocks.sort(key=lambda x: x.block_id)

        last_block = self.fake_free_list_tail.prev_free_block
"""

PATCH_MARK = "Address-sequential free list (vllm#31371)"


def main(argv) -> int:
    orig_dir = argv[0] if argv else os.path.join(HERE, "orig")
    out_dir = argv[1] if len(argv) > 1 else HERE
    orig_path = os.path.join(orig_dir, "kv_cache_utils.py")
    if not os.path.isfile(orig_path):
        sys.exit(f"ERROR: missing {orig_path} "
                 "(start.sh extracts it from the image)")
    src = open(orig_path).read()
    if PATCH_MARK in src:
        sys.exit("ERROR: free_block_sort orig kv_cache_utils.py is already patched")
    count = src.count(APPEND_N_OLD)
    if count != 1:
        sys.exit(f"free_block_sort: append_n anchor not unique/missing "
                 f"(count={count})")
    src = src.replace(APPEND_N_OLD, APPEND_N_NEW)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"free_block_sort: patched kv_cache_utils_v030.py does not "
                 f"parse: {exc}")
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "kv_cache_utils_v030.py"), "w").write(src)
    print("patched kv_cache_utils_v030.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
