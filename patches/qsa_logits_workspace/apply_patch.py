#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""QSA prefill logits workspace for the vLLM 0.30.0 lane (unconditional).

Backports vllm-project/vllm#57105 onto the stock v0.30.0 file:

    vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py
    - qsa_select_paged_prefill chunked its scoring loop to keep the temporary
      fp32 logits under VLLM_SPARSE_INDEXER_MAX_LOGITS_MB, but allocated a
      fresh torch.empty per chunk: on long prefills the allocator sees a
      grow/shrink sequence of large transient blocks and fragments the
      unified-memory pool. The wrapper now allocates the worst-case workspace
      once (max of the chunk budget and one full-width row set) and
      _prefill_logits slices per-chunk views out of it.

Quality-neutral (same kernel, same math, same chunking; only the allocation
site changes), so no toggle. _prefill_logits's only caller is
qsa_select_paged_prefill, so the signature change is contained. Composes with
the QSA prepare fusion overlay (patches/qsa_prepare, vllm#57097): that patch
renames ops/qsa_pre_indexer.py, a different file.

Inputs:  patches/qsa_logits_workspace/orig/qsa_indexer.py (from the image)
Outputs: patches/qsa_logits_workspace/qsa_indexer_v030.py
argv[0]/argv[1] override the orig/output directories (used by the test).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

SIG_OLD = """\
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    max_query_len: int,
    logits_width: int,
    query_offset: int,
    num_queries: int,
) -> torch.Tensor:
    assert query_start_loc.shape == (page_table.shape[0] + 1,)
    assert visible_blocks.shape == (q.shape[0],)
    assert 0 <= query_offset <= query_offset + num_queries <= q.shape[0]
    assert 0 < logits_width <= page_table.shape[1] * k_cache.shape[1]

    logits = torch.empty(
        (num_queries, logits_width), dtype=torch.float32, device=q.device
    )
"""
SIG_NEW = """\
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    logits_workspace: torch.Tensor,
    max_query_len: int,
    logits_width: int,
    query_offset: int,
    num_queries: int,
) -> torch.Tensor:
    assert query_start_loc.shape == (page_table.shape[0] + 1,)
    assert visible_blocks.shape == (q.shape[0],)
    assert 0 <= query_offset <= query_offset + num_queries <= q.shape[0]
    assert 0 < logits_width <= page_table.shape[1] * k_cache.shape[1]
    assert logits_workspace.is_contiguous()
    assert logits_workspace.numel() >= num_queries * logits_width

    # slice from pre-allocated workspace
    logits = logits_workspace[: num_queries * logits_width].view(
        num_queries, logits_width
    )
"""
ALLOC_OLD = """\
    # chunk the inputs to keep temp logits below VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
    max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    rows_per_chunk = max(1, max_logits_bytes // (logits_width * 4))
    topk_workspace = torch.empty(
"""
ALLOC_NEW = """\
    # chunk the inputs to keep temp logits below VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
    # always allocate the worst case to avoid memory fragmentation.
    max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    rows_per_chunk = max(1, max_logits_bytes // (logits_width * 4))

    budget_bytes = max(max_logits_bytes, page_table.shape[1] * k_cache.shape[1] * 4)
    logits_workspace = q.new_empty(budget_bytes // 4, dtype=torch.float32)

    topk_workspace = torch.empty(
"""
CALL_OLD = """\
        logits = _prefill_logits(
            q,
            k_cache,
            page_table,
            query_start_loc,
            visible_blocks,
            max_query_len,
            logits_width,
            query_offset=query_start,
            num_queries=query_end - query_start,
        )
"""
CALL_NEW = """\
        logits = _prefill_logits(
            q,
            k_cache,
            page_table,
            query_start_loc,
            visible_blocks,
            logits_workspace,
            max_query_len,
            logits_width,
            query_offset=query_start,
            num_queries=query_end - query_start,
        )
"""
HUNKS = ((SIG_OLD, SIG_NEW), (ALLOC_OLD, ALLOC_NEW), (CALL_OLD, CALL_NEW))


def main(argv) -> int:
    orig_dir = argv[0] if argv else os.path.join(HERE, "orig")
    out_dir = argv[1] if len(argv) > 1 else HERE
    orig_path = os.path.join(orig_dir, "qsa_indexer.py")
    if not os.path.isfile(orig_path):
        sys.exit(f"ERROR: missing {orig_path} "
                 "(start.sh extracts it from the image)")
    src = open(orig_path).read()
    if "logits_workspace" in src:
        sys.exit("ERROR: qsa_logits_workspace orig qsa_indexer.py is already patched")
    for i, (old, new) in enumerate(HUNKS):
        count = src.count(old)
        if count != 1:
            sys.exit(f"qsa_logits_workspace: anchor {i} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"qsa_logits_workspace: patched qsa_indexer_v030.py does not parse: {exc}")
    out = os.path.join(out_dir, "qsa_indexer_v030.py")
    open(out, "w").write(src)
    print(f"patched {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
