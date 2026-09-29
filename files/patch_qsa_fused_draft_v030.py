#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Backport vllm-project/vllm#58449 (QSA builder implements
update_draft_decode_metadata -> fused multi-step MTP draft decode) onto the
pristine vLLM 0.30.0 models/qwen4_exp/common/qsa_cache.py.

The PR extracts _launch_qsa_metadata_kernel from build_qsa_metadata_triton,
carries common_slot_mapping + num_mapped_tokens on QSAForwardMetadata, sets
supports_draft_decode_metadata_update = HAS_TRITON on QSAMetadataBuilder, and
implements the in-place update so the autoregressive speculator runs its fused
draft-decode loop without rebuilding QSA metadata between draft steps.

Usage: patch_qsa_fused_draft_v030.py <pristine_dir_or_file> <out_dir_or_file>
Anchor-exact: refuses loudly on drift, and refuses a second application (MARK).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RELNAME = "qsa_cache.py"
MARK = "# vllm#58449 fused-multi-step-draft-decode backport (v0.30 port)"

CANDIDATES = [
    "qsa_cache.py",
    os.path.join("common", "qsa_cache.py"),
    os.path.join("models", "qwen4_exp", "common", "qsa_cache.py"),
    os.path.join("vllm", "models", "qwen4_exp", "common", "qsa_cache.py"),
]

# Vendored from the PR's qsa_cache.py hunks, byte-exact (anchor, replacement).
EDITS = [
    (
        (
            ') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:\n'
            '    """Build QSA side-cache and optional pre-indexer work metadata."""\n'
            '    num_tokens = common_attn_metadata.num_actual_tokens\n'
            '    num_mapped_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])\n'
            '    token_to_req = token_to_req_buffer[:num_tokens]\n'
            '    logical_positions = logical_positions_buffer[:num_tokens]\n'
            '    visible_blocks = visible_blocks_buffer[:num_tokens]\n'
            '    slot_mapping = slot_mapping_buffer[:num_tokens]\n'
            '    num_reqs = common_attn_metadata.query_start_loc.shape[0] - 1\n'
            '    assert num_reqs > 0\n'
            '\n'
            '    if k_work_metadata_buffer is not None:\n'
        ),
        (
            ') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:\n'
            '    """Build QSA side-cache and optional pre-indexer work metadata."""\n'
            '    num_tokens = common_attn_metadata.num_actual_tokens\n'
            '    token_to_req = token_to_req_buffer[:num_tokens]\n'
            '    logical_positions = logical_positions_buffer[:num_tokens]\n'
            '    visible_blocks = visible_blocks_buffer[:num_tokens]\n'
            '    slot_mapping = slot_mapping_buffer[:num_tokens]\n'
            '    if num_tokens == 0 and k_work_metadata_buffer is None:\n'
            '        return token_to_req, logical_positions, visible_blocks, slot_mapping\n'
            '    _launch_qsa_metadata_kernel(\n'
            '        common_attn_metadata.query_start_loc,\n'
            '        common_attn_metadata.seq_lens,\n'
            '        common_attn_metadata.slot_mapping,\n'
            '        common_attn_metadata.block_table_tensor,\n'
            '        token_to_req,\n'
            '        logical_positions,\n'
            '        visible_blocks,\n'
            '        slot_mapping,\n'
            '        k_work_metadata_buffer,\n'
            '        num_mapped_tokens=int(common_attn_metadata.query_start_loc_cpu[-1]),\n'
            '        storage_block_size=storage_block_size,\n'
            '        compress_ratio=compress_ratio,\n'
            '        circular_buffer_size=circular_buffer_size,\n'
            '        request_capacity=request_capacity,\n'
            '    )\n'
            '    if circular_buffer_size == 0 and compress_ratio == 1:\n'
            '        slot_mapping = common_attn_metadata.slot_mapping[:num_tokens]\n'
            '    return token_to_req, logical_positions, visible_blocks, slot_mapping\n'
            '\n'
            '\n'
            'def _launch_qsa_metadata_kernel(\n'
            '    query_start_loc: torch.Tensor,\n'
            '    seq_lens: torch.Tensor,\n'
            '    common_slot_mapping: torch.Tensor,\n'
            '    block_table: torch.Tensor,\n'
            '    token_to_req: torch.Tensor,\n'
            '    logical_positions: torch.Tensor,\n'
            '    visible_blocks: torch.Tensor,\n'
            '    slot_mapping: torch.Tensor,\n'
            '    k_work_metadata_buffer: torch.Tensor | None,\n'
            '    *,\n'
            '    num_mapped_tokens: int,\n'
            '    storage_block_size: int,\n'
            '    compress_ratio: int,\n'
            '    circular_buffer_size: int,\n'
            '    request_capacity: int | None,\n'
            ') -> None:\n'
            '    """Fill QSA metadata in place; capture-safe given fixed shapes and scalars."""\n'
            '    num_tokens = token_to_req.shape[0]\n'
            '    num_reqs = query_start_loc.shape[0] - 1\n'
            '    assert num_reqs > 0\n'
            '\n'
            '    if k_work_metadata_buffer is not None:\n'
        ),
    ),
    (
        (
            '        request_scan_size = 1\n'
            '        max_num_work = 0\n'
            '\n'
            '    if num_tokens == 0 and k_work_metadata_buffer is None:\n'
            '        return token_to_req, logical_positions, visible_blocks, slot_mapping\n'
            '\n'
            '    block_table = common_attn_metadata.block_table_tensor\n'
            '    num_search_steps = int(math.ceil(math.log2(num_reqs)))\n'
            '    work_search_steps = int(math.ceil(math.log2(num_reqs)))\n'
            '    # The same grid covers token tiles and, for the compressed cache, work tiles.\n'
        ),
        (
            '        request_scan_size = 1\n'
            '        max_num_work = 0\n'
            '\n'
            '    num_search_steps = int(math.ceil(math.log2(num_reqs)))\n'
            '    work_search_steps = int(math.ceil(math.log2(num_reqs)))\n'
            '    # The same grid covers token tiles and, for the compressed cache, work tiles.\n'
        ),
    ),
    (
        (
            '        cdiv(max_num_work, 256) if k_work_metadata_buffer is not None else 0\n'
            '    )\n'
            '    _build_qsa_metadata_kernel[(max(num_token_blocks, num_work_blocks, 1),)](\n'
            '        common_attn_metadata.query_start_loc,\n'
            '        common_attn_metadata.seq_lens,\n'
            '        common_attn_metadata.slot_mapping,\n'
            '        block_table,\n'
            '        token_to_req,\n'
            '        logical_positions,\n'
        ),
        (
            '        cdiv(max_num_work, 256) if k_work_metadata_buffer is not None else 0\n'
            '    )\n'
            '    _build_qsa_metadata_kernel[(max(num_token_blocks, num_work_blocks, 1),)](\n'
            '        query_start_loc,\n'
            '        seq_lens,\n'
            '        common_slot_mapping,\n'
            '        block_table,\n'
            '        token_to_req,\n'
            '        logical_positions,\n'
        ),
    ),
    (
        (
            '        WORK_BLOCK_SIZE=256,\n'
            '        num_warps=4,\n'
            '    )\n'
            '    if circular_buffer_size == 0 and compress_ratio == 1:\n'
            '        slot_mapping = common_attn_metadata.slot_mapping[:num_tokens]\n'
            '    return token_to_req, logical_positions, visible_blocks, slot_mapping\n'
            '\n'
            '\n'
            'def _build_qsa_metadata_torch(\n'
        ),
        (
            '        WORK_BLOCK_SIZE=256,\n'
            '        num_warps=4,\n'
            '    )\n'
            '\n'
            '\n'
            'def _build_qsa_metadata_torch(\n'
        ),
    ),
    (
        (
            '    logical_positions: torch.Tensor\n'
            '    visible_blocks: torch.Tensor\n'
            '    k_work_metadata: torch.Tensor\n'
            '    num_actual_tokens: int\n'
            '    num_decodes: int\n'
            '    num_decode_tokens: int\n'
        ),
        (
            '    logical_positions: torch.Tensor\n'
            '    visible_blocks: torch.Tensor\n'
            '    k_work_metadata: torch.Tensor\n'
            '    common_slot_mapping: torch.Tensor\n'
            '    num_mapped_tokens: int\n'
            '    num_actual_tokens: int\n'
            '    num_decodes: int\n'
            '    num_decode_tokens: int\n'
        ),
    ),
    (
        (
            '    """Build QSA metadata from vLLM\'s cache-group-specific common metadata."""\n'
            '\n'
            '    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH\n'
            '\n'
            '    def __init__(\n'
            '        self,\n'
        ),
        (
            '    """Build QSA metadata from vLLM\'s cache-group-specific common metadata."""\n'
            '\n'
            '    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH\n'
            '    # vllm#58449 fused-multi-step-draft-decode backport (v0.30 port)\n'
            '    supports_draft_decode_metadata_update = HAS_TRITON\n'
            '\n'
            '    def __init__(\n'
            '        self,\n'
        ),
    ),
    (
        (
            '            logical_positions=logical_positions,\n'
            '            visible_blocks=visible_blocks,\n'
            '            k_work_metadata=k_work_metadata,\n'
            '            num_actual_tokens=num_tokens,\n'
            '            num_decodes=num_decodes,\n'
            '            num_decode_tokens=num_decode_tokens,\n'
        ),
        (
            '            logical_positions=logical_positions,\n'
            '            visible_blocks=visible_blocks,\n'
            '            k_work_metadata=k_work_metadata,\n'
            '            common_slot_mapping=common_attn_metadata.slot_mapping,\n'
            '            num_mapped_tokens=int(common_attn_metadata.query_start_loc_cpu[-1]),\n'
            '            num_actual_tokens=num_tokens,\n'
            '            num_decodes=num_decodes,\n'
            '            num_decode_tokens=num_decode_tokens,\n'
        ),
    ),
    (
        (
            '            compress_ratio=self.compress_ratio,\n'
            '        )\n'
            '\n'
            '\n'
            'class QSAStateBackend(AttentionBackend):\n'
            '    """Key-only dummy backend for out-of-band QSA side-cache operations."""\n'
        ),
        (
            '            compress_ratio=self.compress_ratio,\n'
            '        )\n'
            '\n'
            '    def update_draft_decode_metadata(self, metadata: QSAForwardMetadata) -> None:\n'
            '        if metadata.num_actual_tokens == 0:\n'
            '            return\n'
            '        build_k_work = not self.is_circular_buffer and self.compress_ratio != 1\n'
            '        _launch_qsa_metadata_kernel(\n'
            '            metadata.query_start_loc,\n'
            '            metadata.seq_lens,\n'
            '            metadata.common_slot_mapping,\n'
            '            metadata.block_table,\n'
            '            metadata.token_to_req,\n'
            '            metadata.logical_positions,\n'
            '            metadata.visible_blocks,\n'
            '            metadata.slot_mapping,\n'
            '            metadata.k_work_metadata if build_k_work else None,\n'
            '            num_mapped_tokens=metadata.num_mapped_tokens,\n'
            '            storage_block_size=self.storage_block_size,\n'
            '            compress_ratio=self.compress_ratio,\n'
            '            circular_buffer_size=(\n'
            '                self.kv_cache_spec.block_size if self.is_circular_buffer else 0\n'
            '            ),\n'
            '            request_capacity=self.request_capacity if build_k_work else None,\n'
            '        )\n'
            '\n'
            '\n'
            'class QSAStateBackend(AttentionBackend):\n'
            '    """Key-only dummy backend for out-of-band QSA side-cache operations."""\n'
        ),
    ),
]


def find_input(orig: str) -> str:
    if os.path.isfile(orig):
        return orig
    for rel in CANDIDATES:
        p = os.path.join(orig, rel)
        if os.path.isfile(p):
            return p
    sys.exit(f"qsa_cache.py not found under {orig} (tried: {', '.join(CANDIDATES)})")


def patch(src: str) -> str:
    marks = src.count(MARK)
    if marks > 0:
        sys.exit(
            f"qsa_cache.py already carries the vllm#58449 port "
            f"(MARK x{marks}); refusing to double-apply"
        )
    for i, (old, new) in enumerate(EDITS):
        count = src.count(old)
        if count != 1:
            first = old.splitlines()[0] if old else ""
            sys.exit(
                "DRIFT: vllm#58449 anchor "
                f"{i} not unique/missing (count={count}) in the pristine "
                f"source; anchor starts: {first!r} - nothing written"
            )
        src = src.replace(old, new)
    if src.count(MARK) != 1:
        sys.exit(f"internal error: MARK count {src.count(MARK)} != 1")
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"qsa_cache.py: patched source does not parse: {exc}")
    return src


def main(argv: list[str]) -> int:
    orig = find_input(argv[0] if argv else os.path.join(HERE, "v030_fused_draft", "orig"))
    out = argv[1] if len(argv) > 1 else os.path.join(HERE, "v030_fused_draft")
    with open(orig, encoding="utf-8") as handle:
        src = handle.read()
    patched = patch(src)
    if os.path.isdir(out) or out.endswith(os.sep):
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, RELNAME)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        path = out
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        handle.write(patched)
    os.replace(path + ".tmp", path)
    print(f"[ok] patched, ast.parse OK, MARK once -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
