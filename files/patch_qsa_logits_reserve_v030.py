#!/usr/bin/env python3
"""Backport vllm#57105 (merged 2026-09-27, after v0.30.0) onto the v0.30
qsa_indexer: qsa_select_paged_prefill allocated a fresh (num_queries,
logits_width) fp32 logits tensor per chunk, so long prefills churn the
caching allocator and fragment the pool — the measured death path at
200k+ prefill on this box (31 GiB free -> 0 in one guard tick).
The fix reserves the worst-case workspace once per call and slices it.
Pure allocation change: no numerics touched. Usage:
    patch_qsa_logits_reserve_v030.py ORIG_DIR OUT_DIR
where ORIG_DIR holds qsa_indexer.py (pristine v0.30)."""
import ast
import os
import sys

ORIG, OUT = sys.argv[1], sys.argv[2]
SRC = os.path.join(ORIG, "qsa_indexer.py")

edits = [
    # 1. private prefill signature: workspace arrives after visible_blocks
    ("""    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    max_query_len: int,
    logits_width: int,
    query_offset: int,
    num_queries: int,
) -> torch.Tensor:""",
     """    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    logits_workspace: torch.Tensor,
    max_query_len: int,
    logits_width: int,
    query_offset: int,
    num_queries: int,
) -> torch.Tensor:"""),
    # 2. per-chunk allocation -> slice of the pre-reserved workspace
    ("""    assert 0 < logits_width <= page_table.shape[1] * k_cache.shape[1]

    logits = torch.empty(
        (num_queries, logits_width), dtype=torch.float32, device=q.device
    )
""",
     """    assert 0 < logits_width <= page_table.shape[1] * k_cache.shape[1]
    assert logits_workspace.is_contiguous()
    assert logits_workspace.numel() >= num_queries * logits_width

    logits = logits_workspace[: num_queries * logits_width].view(
        num_queries, logits_width
    )
"""),
    # 3. reserve worst case once per call (upstream comment kept verbatim)
    ("""    # chunk the inputs to keep temp logits below VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
    max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    rows_per_chunk = max(1, max_logits_bytes // (logits_width * 4))
    topk_workspace = torch.empty(""",
     """    # chunk the inputs to keep temp logits below VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
    # always allocate the worst case to avoid memory fragmentation.
    max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    rows_per_chunk = max(1, max_logits_bytes // (logits_width * 4))

    budget_bytes = max(max_logits_bytes, page_table.shape[1] * k_cache.shape[1] * 4)
    logits_workspace = q.new_empty(budget_bytes // 4, dtype=torch.float32)

    topk_workspace = torch.empty("""),
    # 4. the single call site inside qsa_select_paged_prefill
    ("""        query_start_loc,
        visible_blocks,
        max_query_len,
        logits_width,
        query_offset=query_start,""",
     """        query_start_loc,
        visible_blocks,
        logits_workspace,
        max_query_len,
        logits_width,
        query_offset=query_start,"""),
]

s = open(SRC).read()
for old, new in edits:
    n = s.count(old)
    assert n == 1, f"anchor count {n} (expected 1) for: {old[:60]!r}"
    s = s.replace(old, new)
ast.parse(s)  # refuse to emit a non-parsing file
os.makedirs(OUT, exist_ok=True)
dst = os.path.join(OUT, "qsa_indexer.py")
open(dst, "w").write(s)
print(f"patched {dst} (vllm#57105 backport)")
