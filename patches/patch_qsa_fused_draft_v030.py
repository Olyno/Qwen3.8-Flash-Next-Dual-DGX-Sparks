#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Fused multi-step MTP draft metadata for QSA on the vLLM 0.30.0 lane
(opt-in, QSA_FUSED_DRAFT=true).

Ports myllmbox/vllm@c3f56fe3b412 ("Qwen4Exp: QSA fused multi-step draft
metadata", the code proposed upstream as vllm-project/vllm#58449) onto the
stock v0.30.0 file:

Upstream: https://github.com/vllm-project/vllm/pull/58449 (bilikaz,
unmerged), Apache-2.0; ported from the myllmbox/vllm fork
(https://github.com/myllmbox/qwen38-flash-next-recipe, MIT for kit scripts
and image patches — see licenses/myllmbox-MIT.LICENSE).

    vllm/models/qwen4_exp/common/qsa_cache.py
    - build_qsa_metadata_triton: the kernel launch is factored into
      _launch_qsa_metadata_kernel (same grid, same arguments, in place on the
      builder's persistent buffers).
    - QSAForwardMetadata: common_slot_mapping + num_mapped_tokens fields so
      the update path never reads query_start_loc_cpu.
    - QSAMetadataBuilder: opts into supports_draft_decode_metadata_update and
      implements update_draft_decode_metadata, re-launching the metadata
      kernel between draft steps (the DeepseekSparseSWA pattern the v0.30
      speculator already calls at
      v1/worker/gpu/spec_decode/autoregressive/speculator.py).

Stock v0.30 rebuilds attention metadata for the whole model between MTP draft
steps because the QSA builder never opted in; the speculator logs "Fused
multi-step draft decode is not supported by attention backend(s) ...". With
the overlay the speculator takes its fused path for QSA too.

Two changes vs the vendor commit: the A/B gate is renamed MBX_FUSED_DRAFT ->
VLLM_QSA_FUSED_DRAFT and defaults OFF (fail closed; engine/patches.sh passes
-e VLLM_QSA_FUSED_DRAFT=1 via OVERLAY_ENV when it mounts the file).

Inputs:  patches/v030_qsa_fused/orig/qsa_cache.py (extracted from the image)
Outputs: patches/v030_qsa_fused/qsa_cache_v030.py
argv[0]/argv[1] override the orig/output directories (used by the test).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------- qsa_cache.py
IMPORT_OLD = """\
import math
from dataclasses import dataclass
"""
IMPORT_NEW = """\
import math
import os
from dataclasses import dataclass
"""
BUILD_HEAD_OLD = """\
    \"\"\"Build QSA side-cache and optional pre-indexer work metadata.\"\"\"
    num_tokens = common_attn_metadata.num_actual_tokens
    num_mapped_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
    token_to_req = token_to_req_buffer[:num_tokens]
    logical_positions = logical_positions_buffer[:num_tokens]
    visible_blocks = visible_blocks_buffer[:num_tokens]
    slot_mapping = slot_mapping_buffer[:num_tokens]
    num_reqs = common_attn_metadata.query_start_loc.shape[0] - 1
    assert num_reqs > 0
"""
BUILD_HEAD_NEW = """\
    \"\"\"Build QSA side-cache and optional pre-indexer work metadata.\"\"\"
    num_tokens = common_attn_metadata.num_actual_tokens
    token_to_req = token_to_req_buffer[:num_tokens]
    logical_positions = logical_positions_buffer[:num_tokens]
    visible_blocks = visible_blocks_buffer[:num_tokens]
    slot_mapping = slot_mapping_buffer[:num_tokens]
    if num_tokens == 0 and k_work_metadata_buffer is None:
        return token_to_req, logical_positions, visible_blocks, slot_mapping
    _launch_qsa_metadata_kernel(
        common_attn_metadata.query_start_loc,
        common_attn_metadata.seq_lens,
        common_attn_metadata.slot_mapping,
        common_attn_metadata.block_table_tensor,
        token_to_req,
        logical_positions,
        visible_blocks,
        slot_mapping,
        k_work_metadata_buffer,
        num_mapped_tokens=int(common_attn_metadata.query_start_loc_cpu[-1]),
        storage_block_size=storage_block_size,
        compress_ratio=compress_ratio,
        circular_buffer_size=circular_buffer_size,
        request_capacity=request_capacity,
    )
    if circular_buffer_size == 0 and compress_ratio == 1:
        slot_mapping = common_attn_metadata.slot_mapping[:num_tokens]
    return token_to_req, logical_positions, visible_blocks, slot_mapping


def _launch_qsa_metadata_kernel(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    common_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    visible_blocks: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_work_metadata_buffer: torch.Tensor | None,
    *,
    num_mapped_tokens: int,
    storage_block_size: int,
    compress_ratio: int,
    circular_buffer_size: int,
    request_capacity: int | None,
) -> None:
    \"\"\"Fill QSA metadata in place; capture-safe given fixed shapes and scalars.\"\"\"
    num_tokens = token_to_req.shape[0]
    num_reqs = query_start_loc.shape[0] - 1
    assert num_reqs > 0
"""
EARLY_RET_OLD = """\
    if num_tokens == 0 and k_work_metadata_buffer is None:
        return token_to_req, logical_positions, visible_blocks, slot_mapping

    block_table = common_attn_metadata.block_table_tensor
    num_search_steps = int(math.ceil(math.log2(num_reqs)))
"""
EARLY_RET_NEW = """\
    num_search_steps = int(math.ceil(math.log2(num_reqs)))
"""
KARGS_OLD = """\
    _build_qsa_metadata_kernel[(max(num_token_blocks, num_work_blocks, 1),)](
        common_attn_metadata.query_start_loc,
        common_attn_metadata.seq_lens,
        common_attn_metadata.slot_mapping,
        block_table,
"""
KARGS_NEW = """\
    _build_qsa_metadata_kernel[(max(num_token_blocks, num_work_blocks, 1),)](
        query_start_loc,
        seq_lens,
        common_slot_mapping,
        block_table,
"""
TAIL_OLD = """\
        WORK_BLOCK_SIZE=256,
        num_warps=4,
    )
    if circular_buffer_size == 0 and compress_ratio == 1:
        slot_mapping = common_attn_metadata.slot_mapping[:num_tokens]
    return token_to_req, logical_positions, visible_blocks, slot_mapping
"""
TAIL_NEW = """\
        WORK_BLOCK_SIZE=256,
        num_warps=4,
    )
"""
FIELDS_OLD = """\
    k_work_metadata: torch.Tensor
    num_actual_tokens: int
"""
FIELDS_NEW = """\
    k_work_metadata: torch.Tensor
    common_slot_mapping: torch.Tensor
    num_mapped_tokens: int
    num_actual_tokens: int
"""
OPTIN_OLD = """\
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
"""
OPTIN_NEW = """\
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    # Fused multi-step draft metadata (vllm#58449 port): re-launch the QSA
    # metadata kernel in place between draft steps instead of a full rebuild.
    supports_draft_decode_metadata_update = HAS_TRITON and (
        os.environ.get("VLLM_QSA_FUSED_DRAFT", "0") == "1"
    )

    def __init__(
"""
RETURN_OLD = """\
            k_work_metadata=k_work_metadata,
            num_actual_tokens=num_tokens,
"""
RETURN_NEW = """\
            k_work_metadata=k_work_metadata,
            common_slot_mapping=common_attn_metadata.slot_mapping,
            num_mapped_tokens=int(common_attn_metadata.query_start_loc_cpu[-1]),
            num_actual_tokens=num_tokens,
"""
UPDATE_OLD = """\
            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
        )


class QSAStateBackend(AttentionBackend):
"""
UPDATE_NEW = """\
            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
        )

    def update_draft_decode_metadata(self, metadata: QSAForwardMetadata) -> None:
        if metadata.num_actual_tokens == 0:
            return
        build_k_work = not self.is_circular_buffer and self.compress_ratio != 1
        _launch_qsa_metadata_kernel(
            metadata.query_start_loc,
            metadata.seq_lens,
            metadata.common_slot_mapping,
            metadata.block_table,
            metadata.token_to_req,
            metadata.logical_positions,
            metadata.visible_blocks,
            metadata.slot_mapping,
            metadata.k_work_metadata if build_k_work else None,
            num_mapped_tokens=metadata.num_mapped_tokens,
            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
            circular_buffer_size=(
                self.kv_cache_spec.block_size if self.is_circular_buffer else 0
            ),
            request_capacity=self.request_capacity if build_k_work else None,
        )


class QSAStateBackend(AttentionBackend):
"""
HUNKS = ((IMPORT_OLD, IMPORT_NEW), (BUILD_HEAD_OLD, BUILD_HEAD_NEW),
         (EARLY_RET_OLD, EARLY_RET_NEW), (KARGS_OLD, KARGS_NEW),
         (TAIL_OLD, TAIL_NEW), (FIELDS_OLD, FIELDS_NEW),
         (OPTIN_OLD, OPTIN_NEW), (RETURN_OLD, RETURN_NEW),
         (UPDATE_OLD, UPDATE_NEW))


def _apply(src, hunks, name):
    for i, (old, new) in enumerate(hunks):
        count = src.count(old)
        if count != 1:
            sys.exit(f"qsa_fused_draft_v030: anchor {i} in {name} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    return src


def main(argv) -> int:
    orig_dir = argv[0] if argv else os.path.join(HERE, "v030_qsa_fused", "orig")
    out_dir = argv[1] if len(argv) > 1 else os.path.join(HERE, "v030_qsa_fused")
    orig = os.path.join(orig_dir, "qsa_cache.py")
    if not os.path.isfile(orig):
        sys.exit(f"ERROR: missing {orig} (start.sh extracts it from the image)")
    src = open(orig).read()
    if "update_draft_decode_metadata" in src:
        sys.exit("ERROR: v030_qsa_fused orig is already patched")
    src = _apply(src, HUNKS, "qsa_cache.py")
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"qsa_fused_draft_v030: patched qsa_cache_v030.py does not parse: {exc}")
    out = os.path.join(out_dir, "qsa_cache_v030.py")
    os.makedirs(out_dir, exist_ok=True)
    with open(out + ".tmp", "w") as f:
        f.write(src)
    os.replace(out + ".tmp", out)
    print(f"patched {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
