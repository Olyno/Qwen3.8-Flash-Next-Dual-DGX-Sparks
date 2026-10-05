#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""PLE short-conv metadata sync removal for the vLLM 0.30.0 lane (unconditional).

Backports vllm-project/vllm#58114 onto the stock v0.30.0 files:

    vllm/v1/attention/backends/mamba_attn.py
    - BaseMambaAttentionMetadataBuilder gains needs_causal_conv1d_metadata
      (default True); the query_start_loc_p_cpu read +
      compute_causal_conv1d_metadata call in _compute_common_metadata only
      run when it is set.

    vllm/v1/attention/backends/short_conv_attn.py
    - PleShortConvAttentionMetadataBuilder sets the flag False and stops
      building the causal_conv1d Triton metadata (nums_dict / batch_ptr /
      token_chunk_offset_ptr) and the CPU query-loc mirrors entirely: the
      non_spec_query_start_loc_cpu twin, the unconditional query_lens diff,
      and the pure-spec branch's arange/empty token indices all go away.
      Pure-spec batches take block_table_tensor[:num_spec_decodes, 0]
      directly and spec_token_indx/non_spec_token_indx stay None;
      num_accepted_tokens is sliced per branch instead of globally.

    vllm/models/qwen4_exp/nvidia/ple_layer.py
    - _short_conv_dilated_dispatch reads metadata.query_start_loc_p (already
      populated by the builder) instead of re-slicing
      metadata.non_spec_query_start_loc.

Every removed line existed only to feed GPU->CPU syncs or CPU->GPU plumbing
that no PLE consumer reads (nums_dict/batch_ptr/token_chunk_offset_ptr are
None-defaulted fields on BaseMambaAttentionMetadata), so this is
behavior-neutral and unconditional. The amd/ple_layer.py twin and the PR's
test-file hunks are not ported (amd lane unused here).

Inputs:  patches/ple_metadata_syncs/orig/{mamba_attn,short_conv_attn,ple_layer}.py
         (extracted from the image)
Outputs: patches/ple_metadata_syncs/{mamba_attn,short_conv_attn,ple_layer}_v030.py
argv[0]/argv[1] override the orig/output directories (used by the test).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# vllm/v1/attention/backends/mamba_attn.py
# ---------------------------------------------------------------------------
MAMBA_ATTR_OLD = """\
    # Will be disabled if speculative decoding is used
    supports_update_block_table: bool = True

    def __init__(
"""
MAMBA_ATTR_NEW = """\
    # Will be disabled if speculative decoding is used
    supports_update_block_table: bool = True
    needs_causal_conv1d_metadata: bool = True

    def __init__(
"""
MAMBA_CPU_OLD = """\
            query_start_loc_p_cpu = (
                common_attn_metadata.query_start_loc_cpu[-num_prefills - 1 :]
                - num_decode_tokens
            )
            query_start_loc_p = (
"""
MAMBA_CPU_NEW = """\
            query_start_loc_p = (
"""
MAMBA_GATE_OLD = """\
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    query_start_loc_p_cpu,
                    device=common_attn_metadata.query_start_loc.device,
                )
            )
"""
MAMBA_GATE_NEW = """\
            if self.needs_causal_conv1d_metadata:
                query_start_loc_p_cpu = (
                    common_attn_metadata.query_start_loc_cpu[-num_prefills - 1 :]
                    - num_decode_tokens
                )
                nums_dict, batch_ptr, token_chunk_offset_ptr = (
                    compute_causal_conv1d_metadata(
                        query_start_loc_p_cpu,
                        device=common_attn_metadata.query_start_loc.device,
                    )
                )
"""
MAMBA_HUNKS = (
    (MAMBA_ATTR_OLD, MAMBA_ATTR_NEW),
    (MAMBA_CPU_OLD, MAMBA_CPU_NEW),
    (MAMBA_GATE_OLD, MAMBA_GATE_NEW),
)

# ---------------------------------------------------------------------------
# vllm/v1/attention/backends/short_conv_attn.py
# ---------------------------------------------------------------------------
SC_IMPORT_OLD = """\
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
)
"""
SC_IMPORT_NEW = """\
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    mamba_get_block_table_tensor,
)
"""
SC_ATTR_OLD = """\
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold: int = 1
    supports_update_block_table = False

    def __init__(
"""
SC_ATTR_NEW = """\
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold: int = 1
    supports_update_block_table = False
    needs_causal_conv1d_metadata = False

    def __init__(
"""
SC_PROLOGUE_OLD = """\
        # For causal_conv1d (non-spec prefill Triton kernel metadata).
        nums_dict = None
        batch_ptr = None
        token_chunk_offset_ptr = None
        has_initial_states_p = None
        has_initial_states_d = None
        num_computed_tokens_p = None
        # Original request indices of the non-spec requests, ordered
        # [decodes, prefills]. Used to gather per-request data consistently.
        non_spec_req_idx_cpu: torch.Tensor | None = None

        query_lens = torch.diff(query_start_loc)
        # Per-request classification by mask, NOT by position. With
"""
SC_PROLOGUE_NEW = """\
        has_initial_states_p = None
        has_initial_states_d = None
        num_computed_tokens_p = None

        # Per-request classification by mask, NOT by position. With
"""
SC_GROUP_OLD = """\
        # Original request indices grouped as
        # [spec | non-spec decode | non-spec prefill]; each group keeps the
        # original (already reordered) relative order via a stable nonzero.
        spec_req_idx_cpu = spec_sequence_masks_cpu.nonzero(as_tuple=True)[0]
        decode_req_idx_cpu = decode_mask_cpu.nonzero(as_tuple=True)[0]
        prefill_req_idx_cpu = prefill_mask_cpu.nonzero(as_tuple=True)[0]
        non_spec_req_idx_cpu = torch.cat((decode_req_idx_cpu, prefill_req_idx_cpu))
        spec_req_idx = async_tensor_h2d(spec_req_idx_cpu, device=query_start_loc.device)
        non_spec_req_idx: torch.Tensor | None = None

        if num_decodes == 0 and num_prefills == 0:
            # Pure speculative-decode batch: all real tokens are spec tokens.
            spec_token_indx = torch.arange(
                num_spec_decode_tokens,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            non_spec_token_indx = torch.empty(
                0, dtype=torch.int32, device=query_start_loc.device
            )
            spec_state_indices_tensor = block_table_tensor[spec_req_idx, 0]
            non_spec_state_indices_tensor = None
            spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
            non_spec_query_start_loc = None
            non_spec_query_start_loc_cpu = None
        else:
            # Mixed batch: build a per-token group key consistent with the
            # request grouping above (spec=0 | decode=1 | prefill=2) and a
            # stable sort, so tokens of each request stay contiguous and in
            # request order. This yields spec tokens first, then the non-spec
            # [decode, prefill] tokens.
            non_spec_req_idx = async_tensor_h2d(
                non_spec_req_idx_cpu, device=query_start_loc.device
            )
"""
SC_GROUP_NEW = """\
        assert num_accepted_tokens is not None
        non_spec_req_idx: torch.Tensor | None = None

        if num_decodes == 0 and num_prefills == 0:
            # All real requests are speculative; zero-length padding is trailing.
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = block_table_tensor[:num_spec_decodes, 0]
            num_accepted_tokens = num_accepted_tokens[:num_spec_decodes]
            non_spec_state_indices_tensor = None
            spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
            non_spec_query_start_loc = None
        else:
            # Mixed batch: build a per-token group key consistent with the
            # request grouping above (spec=0 | decode=1 | prefill=2) and a
            # stable sort, so tokens of each request stay contiguous and in
            # request order. This yields spec tokens first, then the non-spec
            # [decode, prefill] tokens.
            query_lens = torch.diff(query_start_loc)
            spec_req_idx_cpu = spec_sequence_masks_cpu.nonzero(as_tuple=True)[0]
            decode_req_idx_cpu = decode_mask_cpu.nonzero(as_tuple=True)[0]
            prefill_req_idx_cpu = prefill_mask_cpu.nonzero(as_tuple=True)[0]
            non_spec_req_idx_cpu = torch.cat((decode_req_idx_cpu, prefill_req_idx_cpu))
            spec_req_idx = async_tensor_h2d(
                spec_req_idx_cpu, device=query_start_loc.device
            )
            non_spec_req_idx = async_tensor_h2d(
                non_spec_req_idx_cpu, device=query_start_loc.device
            )
"""
SC_REQGROUP_OLD = """\
            req_group[spec_req_idx] = 0
            req_group[decode_req_idx] = 1
            token_group = torch.repeat_interleave(req_group, query_lens)
"""
SC_REQGROUP_NEW = """\
            req_group.index_fill_(0, spec_req_idx, 0)
            req_group.index_fill_(0, decode_req_idx, 1)
            token_group = torch.repeat_interleave(
                req_group, query_lens, output_size=int(query_start_loc_cpu[-1])
            )
"""
SC_ACCEPTED_OLD = """\
            spec_state_indices_tensor = block_table_tensor[spec_req_idx, 0]
            non_spec_state_indices_tensor = block_table_tensor[non_spec_req_idx, 0]
"""
SC_ACCEPTED_NEW = """\
            spec_state_indices_tensor = block_table_tensor[spec_req_idx, 0]
            num_accepted_tokens = num_accepted_tokens[spec_req_idx]
            non_spec_state_indices_tensor = block_table_tensor[non_spec_req_idx, 0]
"""
SC_CPUTWIN_OLD = """\
            non_spec_query_start_loc_cpu = torch.zeros(
                num_decodes + num_prefills + 1, dtype=torch.int32
            )
            torch.cumsum(
                query_lens_cpu[non_spec_req_idx_cpu],
                dim=0,
                out=non_spec_query_start_loc_cpu[1:],
            )

        assert num_accepted_tokens is not None
        # Accepted-token counts must follow the same request order as the
        # speculative state indices.
        num_accepted_tokens = num_accepted_tokens[spec_req_idx]

        # Compute the conv-state slots for the non-spec decode/prefill split,
        # plus the initial-state masks and Triton causal_conv1d metadata.
        if non_spec_state_indices_tensor is None:
"""
SC_CPUTWIN_NEW = """\

        # Compute the conv-state slots for the non-spec decode/prefill split.
        if non_spec_state_indices_tensor is None:
"""
SC_PREFILL_OLD = """\
                has_initial_states_p = num_computed_tokens_p > 0
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                query_start_loc_p = (
                    non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                )
                query_start_loc_p_cpu = (
                    non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
                if query_start_loc.device.type != "cpu":
                    nums_dict, batch_ptr, token_chunk_offset_ptr = (
                        compute_causal_conv1d_metadata(
                            query_start_loc_p_cpu,
                            device=query_start_loc.device,
                        )
                    )
"""
SC_PREFILL_NEW = """\
                has_initial_states_p = num_computed_tokens_p > 0
                assert non_spec_query_start_loc is not None
                query_start_loc_p = (
                    non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                )
"""
SC_CTOR_OLD = """\
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            query_start_loc_p=query_start_loc_p,
"""
SC_CTOR_NEW = """\
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            query_start_loc_p=query_start_loc_p,
"""
SC_HUNKS = (
    (SC_IMPORT_OLD, SC_IMPORT_NEW),
    (SC_ATTR_OLD, SC_ATTR_NEW),
    (SC_PROLOGUE_OLD, SC_PROLOGUE_NEW),
    (SC_GROUP_OLD, SC_GROUP_NEW),
    (SC_REQGROUP_OLD, SC_REQGROUP_NEW),
    (SC_ACCEPTED_OLD, SC_ACCEPTED_NEW),
    (SC_CPUTWIN_OLD, SC_CPUTWIN_NEW),
    (SC_PREFILL_OLD, SC_PREFILL_NEW),
    (SC_CTOR_OLD, SC_CTOR_NEW),
)

# ---------------------------------------------------------------------------
# vllm/models/qwen4_exp/nvidia/ple_layer.py
# ---------------------------------------------------------------------------
PLE_OLD = """\
            query_start_loc = metadata.non_spec_query_start_loc
            if query_start_loc is None:
                raise ValueError("query_start_loc is required for prefill short-conv")
            query_start_loc = query_start_loc[-num_prefills - 1 :] - num_decode_tokens
"""
PLE_NEW = """\
            query_start_loc = metadata.query_start_loc_p
            if query_start_loc is None:
                raise ValueError("query_start_loc is required for prefill short-conv")
"""
PLE_HUNKS = ((PLE_OLD, PLE_NEW),)

FILES = (
    ("mamba_attn.py", "mamba_attn_v030.py", MAMBA_HUNKS,
     "needs_causal_conv1d_metadata"),
    ("short_conv_attn.py", "short_conv_attn_v030.py", SC_HUNKS,
     "needs_causal_conv1d_metadata"),
    ("ple_layer.py", "ple_layer_v030.py", PLE_HUNKS,
     "metadata.query_start_loc_p"),
)


def main(argv) -> int:
    orig_dir = argv[0] if argv else os.path.join(HERE, "orig")
    out_dir = argv[1] if len(argv) > 1 else HERE
    for name, out_name, hunks, marker in FILES:
        orig_path = os.path.join(orig_dir, name)
        if not os.path.isfile(orig_path):
            sys.exit(f"ERROR: missing {orig_path} "
                     "(start.sh extracts it from the image)")
        src = open(orig_path).read()
        if marker in src:
            sys.exit(f"ERROR: ple_metadata_syncs orig {name} is already patched")
        for i, (old, new) in enumerate(hunks):
            count = src.count(old)
            if count != 1:
                sys.exit(f"ple_metadata_syncs: anchor {i} in {name} not "
                         f"unique/missing (count={count}):\n{old[:200]}")
            src = src.replace(old, new)
        try:
            ast.parse(src)
        except SyntaxError as exc:
            sys.exit(f"ple_metadata_syncs: patched {out_name} does not parse: {exc}")
        out = os.path.join(out_dir, out_name)
        open(out, "w").write(src)
        print(f"patched {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
