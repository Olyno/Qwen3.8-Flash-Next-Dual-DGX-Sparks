#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""QSA metadata graph-padding clamp for the vLLM 0.30.0 lane (unconditional).

Backports vllm-project/vllm#58040 onto the stock v0.30.0 file:

Upstream: https://github.com/vllm-project/vllm/pull/58040 (shaopeng-666,
unmerged), Apache-2.0.

    vllm/models/qwen4_exp/common/qsa_cache.py
    - build_qsa_metadata_triton and _build_qsa_metadata_torch derived
      num_mapped_tokens from query_start_loc_cpu[-1] unclamped. CUDA-graph
      capture pads the final query offset past num_actual_tokens, while the
      QSA metadata buffers hold only real tokens, so a padded decode step
      read/wrote past those buffers. Both builders now clamp to
      num_actual_tokens.

Correctness fix, so no toggle. Composes with the fused-draft overlay
(patches/patch_qsa_fused_draft_v030.py, vllm#58449): that patcher rewrites
build_qsa_metadata_triton and drops the local num_mapped_tokens variable,
so this patcher accepts both input flavors — the stock triton builder head
(local variable) or the fused-draft one (inline call arg plus the recorded
QSAForwardMetadata.num_mapped_tokens field) — and clamps whichever it
finds, failing closed if neither matches. engine/patches.sh feeds the
qsa_cache_views output as this patcher's input and mounts the result.

Inputs:  patches/qsa_meta_clamp/orig/qsa_cache.py (the qsa_cache_views
         overlay's output; the stock file works too — the clamped regions
         are identical)
Outputs: patches/qsa_meta_clamp/qsa_cache_v030.py
argv[0]/argv[1] override the orig/output directories (used by the test).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Torch fallback builder: present in every input flavor.
TORCH_OLD = """\
    del request_capacity
    num_tokens = common_attn_metadata.num_actual_tokens
    num_mapped_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
"""
TORCH_NEW = """\
    del request_capacity
    num_tokens = common_attn_metadata.num_actual_tokens
    # Graph padding can make the final query offset exceed the real token count.
    num_mapped_tokens = min(
        int(common_attn_metadata.query_start_loc_cpu[-1]), num_tokens
    )
"""

# Stock triton builder head (local variable).
TRITON_STOCK_OLD = """\
    \"\"\"Build QSA side-cache and optional pre-indexer work metadata.\"\"\"
    num_tokens = common_attn_metadata.num_actual_tokens
    num_mapped_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
"""
TRITON_STOCK_NEW = """\
    \"\"\"Build QSA side-cache and optional pre-indexer work metadata.\"\"\"
    num_tokens = common_attn_metadata.num_actual_tokens
    # Graph padding can make the final query offset exceed the real token count.
    num_mapped_tokens = min(
        int(common_attn_metadata.query_start_loc_cpu[-1]), num_tokens
    )
"""

# Fused-draft triton builder (vllm#58449 overlay output): the count is passed
# inline to _launch_qsa_metadata_kernel and recorded on QSAForwardMetadata.
FUSED_CALL_OLD = """\
        num_mapped_tokens=int(common_attn_metadata.query_start_loc_cpu[-1]),
        storage_block_size=storage_block_size,
"""
FUSED_CALL_NEW = """\
        # Graph padding can make the final query offset exceed the real token
        # count.
        num_mapped_tokens=min(
            int(common_attn_metadata.query_start_loc_cpu[-1]), num_tokens
        ),
        storage_block_size=storage_block_size,
"""
FUSED_FIELD_OLD = """\
            num_mapped_tokens=int(common_attn_metadata.query_start_loc_cpu[-1]),
            num_actual_tokens=num_tokens,
"""
FUSED_FIELD_NEW = """\
            # Graph padding can make the final query offset exceed the real
            # token count.
            num_mapped_tokens=min(
                int(common_attn_metadata.query_start_loc_cpu[-1]), num_tokens
            ),
            num_actual_tokens=num_tokens,
"""


def _apply(src, hunks, name):
    for i, (old, new) in enumerate(hunks):
        count = src.count(old)
        if count != 1:
            sys.exit(f"qsa_meta_clamp: anchor {i} in {name} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    return src


def main(argv) -> int:
    orig_dir = argv[0] if argv else os.path.join(HERE, "orig")
    out_dir = argv[1] if len(argv) > 1 else HERE
    orig_path = os.path.join(orig_dir, "qsa_cache.py")
    if not os.path.isfile(orig_path):
        sys.exit(f"ERROR: missing {orig_path} "
                 "(start.sh extracts it from the image)")
    src = open(orig_path).read()
    if "Graph padding can make the final query offset" in src:
        sys.exit("ERROR: qsa_meta_clamp orig qsa_cache.py is already patched")
    src = _apply(src, ((TORCH_OLD, TORCH_NEW),), "qsa_cache.py")
    if src.count(TRITON_STOCK_OLD) == 1:
        src = _apply(src, ((TRITON_STOCK_OLD, TRITON_STOCK_NEW),), "qsa_cache.py")
    elif src.count(FUSED_CALL_OLD) == 1 and src.count(FUSED_FIELD_OLD) == 1:
        # Fused-draft overlay output (vllm#58449): no local variable, the
        # unclamped count flows through the inline call arg and the recorded
        # metadata field instead.
        src = _apply(
            src,
            ((FUSED_CALL_OLD, FUSED_CALL_NEW), (FUSED_FIELD_OLD, FUSED_FIELD_NEW)),
            "qsa_cache.py",
        )
    else:
        sys.exit("qsa_meta_clamp: neither the stock nor the fused-draft "
                 "triton builder head matched; refusing to patch")
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"qsa_meta_clamp: patched qsa_cache_v030.py does not parse: {exc}")
    open(os.path.join(out_dir, "qsa_cache_v030.py"), "w").write(src)
    print("patched qsa_cache_v030.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
