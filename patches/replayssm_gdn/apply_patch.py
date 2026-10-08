#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""ReplaySSM-GDN spec decode for the vLLM 0.30.0 lane (opt-in,
REPLAYSSM_GDN=true).

Ports the GDN half of vllm-project/vllm#47576 (draft, Johnny-Liou) onto the
stock v0.30.0 files.

Upstream: https://github.com/vllm-project/vllm/pull/47576 (Johnny-Liou,
unmerged draft), Apache-2.0. The two vendored Triton statics
(gdn_replayssm_spec_decode.py, replayssm_config.py) come verbatim from the PR
and keep their upstream Apache-2.0/vLLM SPDX headers. Spec-decode-only port: the baseline decode path
(fused_recurrent_replayssm.py, write_pos machinery, V1 runner plumbing) is
dropped because this lane always runs MTP. Under the V2 model runner the
builder already receives num_accepted_tokens / num_decode_draft_tokens_cpu /
is_prefilling / rswa_prefix_lens on every non-capture step, so none of the
PR's gpu_model_runner/backend/ubatch_utils hunks are needed. First-decode
detection uses rswa_prefix_lens (prompt lens) vs compute_num_computed_tokens()
instead of the PR's num_prompt_tokens_cpu (not exposed by the V2 runner).

How it works: the GDN page grows from (conv, ssm) to (conv, fp32 checkpoint,
d_cache, k_cache, g_cache); the spec verify kernel
(gdn_replayssm_spec_decode.py, vendored verbatim from the PR) reconstructs
each verify window from the checkpoint + a circular ring of per-token d/k/g
vectors instead of reading/writing the full [HV, V, K] state per token, and
rewrites the checkpoint only when the ring fills (L = buffer_len + 1 + k).
Block-keyed cursors live in fixed-address buffers owned by the metadata
builder, committed once per step from the previous step's num_accepted and
reset on each request's first decode. num_speculative_blocks is forced to 0
(same trick as the in-image KDA RecoverSSM): the spec window positions are
ring offsets, not extra state slots.

Runtime gate: VLLM_REPLAYSSM_GDN=1 plus VLLM_REPLAYSSM_GDN_BUFFER_LEN
(default 16, must be >= 1 + k), both passed via OVERLAY_ENV. Requirements
(enforced in engine/patches.sh): speculative decoding on, LAZY_GDN off
(mutually exclusive, same target files).

Inputs:  patches/replayssm_gdn/orig/*.py (extracted from the image)
Outputs: patches/replayssm_gdn/*_v030.py (mounted by engine/patches.sh)
         plus the two vendored statics gdn_replayssm_spec_decode.py and
         replayssm_config.py, validated here and mounted verbatim.
argv[0]/argv[1] override the orig/output directories (used by the test).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

ENV_HEADER = (
    "# ReplaySSM-GDN spec decode (env-gated overlay; patches/replayssm_gdn).\n"
    '_REPLAYSSM_GDN = os.environ.get("VLLM_REPLAYSSM_GDN", "0") == "1"\n'
    '_REPLAYSSM_GDN_BUFFER_LEN = int(os.environ.get("VLLM_REPLAYSSM_GDN_BUFFER_LEN", "16"))\n'
)

# ---------------------------------------------------------------------------
# vllm/model_executor/layers/mamba/mamba_utils.py
# ---------------------------------------------------------------------------
MU_DTYPE_OLD = """\
        return cls._mamba_state_dtype(
            model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype
        )

    @classmethod
    def kda_state_dtype(
"""
MU_DTYPE_NEW = """\
        return cls._mamba_state_dtype(
            model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype
        )

    @classmethod
    def gated_delta_net_replayssm_spec_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, ...]:
        \"\"\"GDN ReplaySSM state dtypes for the SPECULATIVE-decode kernel.

        The ``ssm`` checkpoint is forced to ``float32``; the ``d``/``k`` ring
        caches use fp16 for bf16 activations (same rule as the non-spec path).
        Call only when use_replayssm_spec is on.
        \"\"\"
        conv_dtype, ssm_dtype = cls._mamba_state_dtype(
            model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype
        )
        activation_dtype = get_kv_cache_torch_dtype("auto", model_dtype)
        cache_dtype = (
            torch.float16 if activation_dtype == torch.bfloat16 else activation_dtype
        )
        return (
            conv_dtype,
            torch.float32,  # fp32 checkpoint
            cache_dtype,  # d_cache
            cache_dtype,  # k_cache
            torch.float32,  # g_cache
        )

    @classmethod
    def kda_state_dtype(
"""
MU_SHAPE_OLD = """\
        return conv_state_shape, temporal_state_shape

    @classmethod
    def kda_state_shape(
"""
MU_SHAPE_NEW = """\
        return conv_state_shape, temporal_state_shape

    @classmethod
    def gated_delta_net_replayssm_spec_state_shape(
        cls,
        tp_world_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        replayssm_buffer_len: int,
        num_spec: int = 0,
    ) -> tuple[tuple[int, ...], ...]:
        \"\"\"GDN ReplaySSM state shapes for the SPECULATIVE-decode kernel.

        The circular ``d_cache``/``k_cache``/``g_cache`` use the L = B + max_spec_len
        history window: a power-of-two buffer ``next_pow2(replayssm_buffer_len + 1 +
        num_spec)``. Call only when use_replayssm_spec is on. The block-keyed
        cursors live in the GDN metadata builder, not the page.
        \"\"\"
        conv_state_shape, temporal_state_shape = cls.gated_delta_net_state_shape(
            tp_world_size,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size,
            num_spec,
        )
        cache_buf_len = 1 << (replayssm_buffer_len + num_spec).bit_length()
        local_v_heads = divide(num_v_heads, tp_world_size)
        local_k_heads = divide(num_k_heads, tp_world_size)
        d_cache_shape = (local_v_heads, cache_buf_len, head_v_dim)
        k_cache_shape = (local_k_heads, cache_buf_len, head_k_dim)
        g_cache_shape = (local_v_heads, cache_buf_len)
        return (
            conv_state_shape,
            temporal_state_shape,
            d_cache_shape,
            k_cache_shape,
            g_cache_shape,
        )

    @classmethod
    def kda_state_shape(
"""
MU_HUNKS = ((MU_DTYPE_OLD, MU_DTYPE_NEW), (MU_SHAPE_OLD, MU_SHAPE_NEW))

# ---------------------------------------------------------------------------
# vllm/model_executor/layers/mamba/gdn/base.py
# ---------------------------------------------------------------------------
BASE_IMPORT_OLD = """\
import torch
from transformers import PretrainedConfig
"""
BASE_IMPORT_NEW = """\
import os

import torch
from transformers import PretrainedConfig
"""
BASE_CONST_OLD = """\
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


class GatedDeltaNetAttention(PluggableLayer, MambaBase):
"""
BASE_CONST_NEW = """\
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

""" + ENV_HEADER + """\

class GatedDeltaNetAttention(PluggableLayer, MambaBase):
"""
BASE_DTYPE_OLD = """\
    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
"""
BASE_DTYPE_NEW = """\
    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        if _REPLAYSSM_GDN and self.num_spec > 0:
            return MambaStateDtypeCalculator.gated_delta_net_replayssm_spec_state_dtype(
                self.model_config.dtype,
                self.cache_config.mamba_cache_dtype,
                self.cache_config.mamba_ssm_cache_dtype,
            )
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
"""
BASE_HUNKS = (
    (BASE_IMPORT_OLD, BASE_IMPORT_NEW),
    (BASE_CONST_OLD, BASE_CONST_NEW),
    (BASE_DTYPE_OLD, BASE_DTYPE_NEW),
)

# ---------------------------------------------------------------------------
# vllm/model_executor/layers/mamba/abstract.py
# ---------------------------------------------------------------------------
ABS_IMPORT_OLD = """\
from abc import abstractmethod
from collections.abc import Iterable
from math import prod

import torch
"""
ABS_IMPORT_NEW = """\
import os
from abc import abstractmethod
from collections.abc import Iterable
from math import prod

import torch
"""
ABS_CONST_OLD = """\
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


class MambaBase(AttentionLayerBase):
"""
ABS_CONST_NEW = """\
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec

# ReplaySSM-GDN spec decode (env-gated overlay; patches/replayssm_gdn).
_REPLAYSSM_GDN = os.environ.get("VLLM_REPLAYSSM_GDN", "0") == "1"


class MambaBase(AttentionLayerBase):
"""
ABS_BLOCKS_OLD = """\
            # RecoverSSM verifies the whole window off one checkpoint, so it
            # never writes the baseline's per-draft-token state slots.
            num_speculative_blocks=(
                0
                if vllm_config.cache_config.use_kda_recoverssm
                else vllm_config.num_speculative_tokens
            ),
"""
ABS_BLOCKS_NEW = """\
            # RecoverSSM verifies the whole window off one checkpoint, so it
            # never writes the baseline's per-draft-token state slots. The
            # ReplaySSM-GDN spec kernel does the same off its fp32 checkpoint.
            num_speculative_blocks=(
                0
                if (
                    vllm_config.cache_config.use_kda_recoverssm
                    or (_REPLAYSSM_GDN and vllm_config.num_speculative_tokens > 0)
                )
                else vllm_config.num_speculative_tokens
            ),
"""
ABS_HUNKS = (
    (ABS_IMPORT_OLD, ABS_IMPORT_NEW),
    (ABS_CONST_OLD, ABS_CONST_NEW),
    (ABS_BLOCKS_OLD, ABS_BLOCKS_NEW),
)

# ---------------------------------------------------------------------------
# vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
# ---------------------------------------------------------------------------
LIN_CONST_OLD = """\
MAX_FUSED_GDN_MTP_TOKENS = 8
FUSED_GDN_STATE_DTYPES = (torch.float32, torch.bfloat16)
"""
LIN_CONST_NEW = """\
MAX_FUSED_GDN_MTP_TOKENS = 8
FUSED_GDN_STATE_DTYPES = (torch.float32, torch.bfloat16)

""" + ENV_HEADER
LIN_SHAPE_OLD = """\
    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
"""
LIN_SHAPE_NEW = """\
    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], ...]:
        if _REPLAYSSM_GDN and self.num_spec > 0:
            return MambaStateShapeCalculator.gated_delta_net_replayssm_spec_state_shape(
                self.tp_size,
                self.num_k_heads,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                self.conv_kernel_size,
                _REPLAYSSM_GDN_BUFFER_LEN,
                self.num_spec,
            )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
"""
LIN_INIT_OLD = """\
        self.enable_packed_recurrent_decode = (
            envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE
        )
        self.gdn_decode_kernel = envs.VLLM_GDN_DECODE_KERNEL.strip().lower()
"""
LIN_INIT_NEW = """\
        self.enable_packed_recurrent_decode = (
            envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE
        )
        # ReplaySSM-GDN spec decode: the page grows to the 5-tuple
        # (conv, fp32 checkpoint, d/k/g rings) and spec verify is diverted.
        self.use_replayssm_spec = _REPLAYSSM_GDN and self.num_spec > 0
        self.replayssm_buffer_len = _REPLAYSSM_GDN_BUFFER_LEN
        self.max_spec_len = 1 + self.num_spec
        self.gdn_decode_kernel = envs.VLLM_GDN_DECODE_KERNEL.strip().lower()
"""
# get_state_dtype() grows to a 5-tuple under ReplaySSM; both stock unpack
# sites only need the (conv, ssm) pair.
LIN_UNPACK1_OLD = """\
        conv_state_dtype, recurrent_state_dtype = self.get_state_dtype()
"""
LIN_UNPACK1_NEW = """\
        conv_state_dtype, recurrent_state_dtype = self.get_state_dtype()[:2]
"""
LIN_UNPACK2_OLD = """\
        _, state_dtype = self.get_state_dtype()
"""
LIN_UNPACK2_NEW = """\
        # get_state_dtype() is (conv, ssm[, d, k, g]); we only need the ssm dtype.
        state_dtype = self.get_state_dtype()[1]
"""
LIN_CONV_OLD = """\
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
"""
LIN_CONV_NEW = """\
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                # Spec verify window = 1 + num_spec. Use the constant rather
                # than the block-table width so the ReplaySSM path can run
                # with num_speculative_blocks=0.
                max_query_len=(
                    self.max_spec_len
                    if self.use_replayssm_spec
                    else spec_state_indices_tensor.size(-1)
                ),
                validate_data=False,
"""
LIN_REARRANGE_OLD = """\
        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)

        # Split mixed non-spec-decode+prefill to process independently
"""
LIN_REARRANGE_NEW = """\
        # The ReplaySSM spec kernel consumes the post-conv packed
        # mixed_qkv_spec directly, so the q/k/v rearrange is waste on that
        # path (rearrange_mixed_qkv(None) is a no-op for the stock call).
        if spec_sequence_masks is not None and self.use_replayssm_spec:
            query_spec, key_spec, value_spec = None, None, None
        else:
            query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(
                mixed_qkv_spec
            )

        # Split mixed non-spec-decode+prefill to process independently
"""
LIN_SPEC_OLD = """\
        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
"""
LIN_SPEC_NEW = """\
        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None and self.use_replayssm_spec:
            # ReplaySSM spec verify: reconstruct the window from the fp32
            # checkpoint + d/k/g rings instead of the per-token state slots.
            assert mixed_qkv_spec is not None
            cs_out = torch.empty(
                (
                    mixed_qkv_spec.shape[0],
                    self.num_v_heads // self.tp_size,
                    self.head_v_dim,
                ),
                dtype=mixed_qkv_spec.dtype,
                device=mixed_qkv_spec.device,
            )
            self._replayssm_spec_ssm(
                mixed_qkv_spec, a_spec, b_spec, cs_out, attn_metadata
            )
            core_attn_out_spec = cs_out.unsqueeze(0)
            last_recurrent_state = None
        elif spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
"""
LIN_METHOD_OLD = """\
            output_gate_activation=self.norm.activation,
        )

    def _forward_core_fused_norm_packed(
"""
LIN_METHOD_NEW = """\
            output_gate_activation=self.norm.activation,
        )

    def _replayssm_spec_ssm(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> None:
        \"\"\"ReplaySSM spec verify on the circular d/k/g caches + fp32 checkpoint.

        ``mixed_qkv`` is the post-conv packed (q|k|v) of the spec rows;
        ``a``/``b`` are the compacted spec-row gating vectors. The block-keyed
        cursors are committed once per step by the GDN metadata builder.
        \"\"\"
        from vllm.model_executor.layers.mamba.ops.gdn_replayssm_spec_decode import (
            gdn_replayssm_spec_decode,
        )

        num_spec_decodes = attn_metadata.num_spec_decodes
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
        assert spec_query_start_loc is not None
        assert spec_state_indices_tensor is not None
        assert attn_metadata.spec_write_pos_d is not None
        gdn_replayssm_spec_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            checkpoint_state=self.kv_cache[1],
            d_cache=self.kv_cache[2],
            k_cache=self.kv_cache[3],
            g_cache=self.kv_cache[4],
            out=out,
            query_start_loc=spec_query_start_loc[: num_spec_decodes + 1],
            ssm_state_indices=spec_state_indices_tensor[:num_spec_decodes, 0],
            write_pos=attn_metadata.spec_write_pos_d,
            cache_base=attn_metadata.spec_cache_base_d,
            is_flush=attn_metadata.spec_is_flush_d,
            max_cache_len=self.replayssm_buffer_len + self.max_spec_len,
            max_spec_len=self.max_spec_len,
            scale=self.head_k_dim**-0.5,
            use_qk_l2norm_in_kernel=True,
        )

    def _forward_core_fused_norm_packed(
"""
LIN_DIVERT_OLD = """\
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        if (
            self._can_use_fused_gdn_mtp_decode(attn_metadata)
            and attn_metadata.num_prefills == 0
        ):
"""
LIN_DIVERT_NEW = """\
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        if (
            # ReplaySSM spec verify runs through _forward_core (checkpoint +
            # rings), never the fused MTP kernel on the live state slots.
            not self.use_replayssm_spec
            and self._can_use_fused_gdn_mtp_decode(attn_metadata)
            and attn_metadata.num_prefills == 0
        ):
"""
LIN_HUNKS = (
    (LIN_CONST_OLD, LIN_CONST_NEW),
    (LIN_SHAPE_OLD, LIN_SHAPE_NEW),
    (LIN_INIT_OLD, LIN_INIT_NEW),
    (LIN_UNPACK1_OLD, LIN_UNPACK1_NEW),
    (LIN_UNPACK2_OLD, LIN_UNPACK2_NEW),
    (LIN_CONV_OLD, LIN_CONV_NEW),
    (LIN_REARRANGE_OLD, LIN_REARRANGE_NEW),
    (LIN_SPEC_OLD, LIN_SPEC_NEW),
    (LIN_METHOD_OLD, LIN_METHOD_NEW),
    (LIN_DIVERT_OLD, LIN_DIVERT_NEW),
)

# ---------------------------------------------------------------------------
# vllm/v1/attention/backends/gdn_attn.py
# ---------------------------------------------------------------------------
ATT_IMPORT_OLD = """\
from dataclasses import dataclass
from typing import Literal

import torch
"""
ATT_IMPORT_NEW = """\
import os
from dataclasses import dataclass
from typing import Literal

import torch
"""
ATT_CONST_OLD = """\
from vllm.v1.kv_cache_interface import MambaSpec


class GDNAttentionBackend(AttentionBackend):
"""
ATT_CONST_NEW = """\
from vllm.v1.kv_cache_interface import MambaSpec

""" + ENV_HEADER + """\

class GDNAttentionBackend(AttentionBackend):
"""
ATT_FIELDS_OLD = """\
    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
"""
ATT_FIELDS_NEW = """\
    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # ReplaySSM-GDN spec decode cursors: persistent, block-keyed (full
    # (num_gpu_blocks,) fixed-address buffers indexed by
    # spec_state_indices_tensor[:, 0]), advanced once per step by
    # commit_gdn_replayssm_spec. None unless VLLM_REPLAYSSM_GDN is on.
    spec_write_pos_d: torch.Tensor | None = None
    spec_cache_base_d: torch.Tensor | None = None
    spec_is_flush_d: torch.Tensor | None = None

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
"""
ATT_INIT_OLD = """\
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    def _build_chunk_metadata(
"""
ATT_INIT_NEW = """\
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

        # ReplaySSM-GDN spec decode: block-keyed cursors (sized
        # num_gpu_blocks), allocated lazily on first build (num_gpu_blocks is
        # unknown here), advanced once per step by commit_gdn_replayssm_spec.
        self.use_replayssm_spec: bool = _REPLAYSSM_GDN and self.num_spec > 0
        self.replayssm_buffer_len: int = _REPLAYSSM_GDN_BUFFER_LEN
        self.max_spec_len: int = 1 + self.num_spec
        # L = B + max_spec_len history window; physical pow2 ring next_pow2(L).
        self.spec_flush_threshold = self.replayssm_buffer_len + self.max_spec_len
        self.spec_cache_buf_len = 1 << (self.spec_flush_threshold - 1).bit_length()
        self.cursor_device = device
        self.spec_write_pos: torch.Tensor | None = None
        self.spec_cache_base: torch.Tensor | None = None
        self.spec_is_flush: torch.Tensor | None = None

    def _build_chunk_metadata(
"""
ATT_MASK_OLD = """\
        spec_sequence_masks_cpu: torch.Tensor | None = None
        if not self.use_spec_decode or num_decode_draft_tokens_cpu is None:
"""
ATT_MASK_NEW = """\
        spec_sequence_masks_cpu: torch.Tensor | None = None
        if self.use_replayssm_spec and num_accepted_tokens is not None:
            # ReplaySSM spec: every post-prefill row must run through the spec
            # kernel (a draft-less row is a T=1 window). The baseline decode /
            # prefill paths read the checkpoint page, which lags the committed
            # ring history, so routing any decode row there corrupts the state.
            # num_decode_draft_tokens_cpu cannot drive this mask: it is stale
            # on draft-less steps and -1 for decode rows whose drafts were
            # dropped. Zero-query padded rows are excluded by the query-len test.
            is_prefilling_cpu = m.is_prefilling
            assert is_prefilling_cpu is not None
            query_lens_cpu_all = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            spec_sequence_masks_cpu = (
                ~is_prefilling_cpu[: query_lens_cpu_all.shape[0]]
            ) & (query_lens_cpu_all > 0)
            num_spec_decodes = int(spec_sequence_masks_cpu.sum().item())
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                assert (
                    int(query_lens_cpu_all[spec_sequence_masks_cpu].max().item())
                    <= self.num_spec + 1
                ), "ReplaySSM-spec decode row wider than the spec window"
                spec_sequence_masks = async_tensor_h2d(
                    spec_sequence_masks_cpu, device=query_start_loc.device
                )
        elif not self.use_spec_decode or num_decode_draft_tokens_cpu is None:
"""
ATT_SPLIT_OLD = """\
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
"""
ATT_SPLIT_NEW = """\
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                # ReplaySSM routes single-token prefill-as-decode rows to
                # prefill: a fresh block must not be read as decode state.
                split_decodes_and_prefills(
                    m,
                    decode_threshold=1,
                    treat_short_extends_as_decodes=not self.use_replayssm_spec,
                )
            )
"""
ATT_COMMIT_OLD = """\
        # Prepare per-request tensors for cudagraph. m.num_actual_tokens is
"""
ATT_COMMIT_NEW = """\
        # ReplaySSM-GDN spec decode: advance the block-keyed cursors once per
        # step (commit-at-start, using the previous step's num_accepted), then
        # reset first-decode rows. Runs on the UNPADDED spec tensors (the
        # cursors are block-keyed and the kernels skip null blocks, so the
        # cudagraph padding below is fine). The commit/reset launches run in
        # build() (eager, outside the captured region); the cursors are full
        # (num_gpu_blocks,) fixed-address buffers read by the captured kernel.
        spec_write_pos_d = None
        spec_cache_base_d = None
        spec_is_flush_d = None
        if self.use_replayssm_spec and num_spec_decodes > 0:
            from vllm.model_executor.layers.mamba.ops.gdn_replayssm_spec_decode import (
                commit_gdn_replayssm_spec,
                reset_gdn_replayssm_spec_cursors,
            )

            assert spec_state_indices_tensor is not None
            assert num_accepted_tokens is not None
            # non-None whenever num_spec_decodes > 0 (set together above)
            assert spec_sequence_masks_cpu is not None
            if self.spec_write_pos is None:
                n_blocks = self.vllm_config.cache_config.num_gpu_blocks
                assert n_blocks is not None and n_blocks > 0, (
                    "VLLM_REPLAYSSM_GDN needs num_gpu_blocks at build time to "
                    "size the block-keyed cursor buffers"
                )
                self.spec_write_pos = torch.zeros(
                    n_blocks, dtype=torch.int32, device=self.cursor_device
                )
                self.spec_cache_base = torch.zeros(
                    n_blocks, dtype=torch.int32, device=self.cursor_device
                )
                self.spec_is_flush = torch.zeros(
                    n_blocks, dtype=torch.int8, device=self.cursor_device
                )
            sbi = spec_state_indices_tensor[:, 0]
            commit_gdn_replayssm_spec(
                self.spec_write_pos,
                self.spec_cache_base,
                self.spec_is_flush,
                num_accepted_tokens.to(torch.int32),
                sbi,
                max_cache_len=self.spec_flush_threshold,
                max_spec_len=self.max_spec_len,
                cache_buf_len=self.spec_cache_buf_len,
            )
            # Prefill->decode reset for first-decode rows (cursors only; the
            # conv context lives in conv_state). A request's first spec verify
            # has num_computed_tokens == prompt len; that resets its (possibly
            # recycled) block's cursors to write_pos=0.
            prompt_lens_cpu = m.rswa_prefix_lens
            if prompt_lens_cpu is not None:
                context_lens_tensor = m.compute_num_computed_tokens()
                first_decode_full = (
                    context_lens_tensor
                    == prompt_lens_cpu.to(
                        context_lens_tensor.device, non_blocking=True
                    )[: context_lens_tensor.shape[0]]
                ).to(torch.int8)
                spec_row_idx = spec_sequence_masks_cpu.nonzero(as_tuple=True)[0].to(
                    query_start_loc.device, non_blocking=True
                )
                first_decode_d = first_decode_full.index_select(0, spec_row_idx)
                reset_gdn_replayssm_spec_cursors(
                    self.spec_write_pos,
                    self.spec_cache_base,
                    self.spec_is_flush,
                    first_decode_d,
                    sbi,
                    max_cache_len=self.spec_flush_threshold,
                    max_spec_len=self.max_spec_len,
                )
            spec_write_pos_d = self.spec_write_pos
            spec_cache_base_d = self.spec_cache_base
            spec_is_flush_d = self.spec_is_flush

        # Prepare per-request tensors for cudagraph. m.num_actual_tokens is
"""
ATT_CTOR_OLD = """\
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
"""
ATT_CTOR_NEW = """\
            num_accepted_tokens=num_accepted_tokens,
            spec_write_pos_d=spec_write_pos_d,
            spec_cache_base_d=spec_cache_base_d,
            spec_is_flush_d=spec_is_flush_d,
            nums_dict=nums_dict,
"""
ATT_HUNKS = (
    (ATT_IMPORT_OLD, ATT_IMPORT_NEW),
    (ATT_CONST_OLD, ATT_CONST_NEW),
    (ATT_FIELDS_OLD, ATT_FIELDS_NEW),
    (ATT_INIT_OLD, ATT_INIT_NEW),
    (ATT_MASK_OLD, ATT_MASK_NEW),
    (ATT_SPLIT_OLD, ATT_SPLIT_NEW),
    (ATT_COMMIT_OLD, ATT_COMMIT_NEW),
    (ATT_CTOR_OLD, ATT_CTOR_NEW),
)

# ---------------------------------------------------------------------------
# vllm/model_executor/models/qwen4_exp/nvidia/model.py
# ---------------------------------------------------------------------------
MOD_IMPORT_OLD = """\
from collections.abc import Iterable
from itertools import islice

import torch
"""
MOD_IMPORT_NEW = """\
import os
from collections.abc import Iterable
from itertools import islice

import torch
"""
MOD_CONST_OLD = """\
from .qsa import Qwen4ExpQSAAttention


def without_modelopt_fp4(
"""
MOD_CONST_NEW = """\
from .qsa import Qwen4ExpQSAAttention

""" + ENV_HEADER + """\

def without_modelopt_fp4(
"""
MOD_DTYPE_OLD = """\
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )
"""
MOD_DTYPE_NEW = """\
        if _REPLAYSSM_GDN and vllm_config.num_speculative_tokens > 0:
            return MambaStateDtypeCalculator.gated_delta_net_replayssm_spec_state_dtype(
                vllm_config.model_config.dtype,
                vllm_config.cache_config.mamba_cache_dtype,
                vllm_config.cache_config.mamba_ssm_cache_dtype,
            )
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )
"""
MOD_SHAPE_OLD = """\
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )
"""
MOD_SHAPE_NEW = """\
        if _REPLAYSSM_GDN and num_spec > 0:
            return MambaStateShapeCalculator.gated_delta_net_replayssm_spec_state_shape(
                tp_size,
                hf_config.linear_num_key_heads,
                hf_config.linear_num_value_heads,
                hf_config.linear_key_head_dim,
                hf_config.linear_value_head_dim,
                hf_config.linear_conv_kernel_dim,
                _REPLAYSSM_GDN_BUFFER_LEN,
                num_spec,
            )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )
"""
MOD_HUNKS = (
    (MOD_IMPORT_OLD, MOD_IMPORT_NEW),
    (MOD_CONST_OLD, MOD_CONST_NEW),
    (MOD_DTYPE_OLD, MOD_DTYPE_NEW),
    (MOD_SHAPE_OLD, MOD_SHAPE_NEW),
)

# (orig name, hunks, output name)
TARGETS = (
    ("mamba_utils.py", MU_HUNKS, "mamba_utils_v030.py"),
    ("base.py", BASE_HUNKS, "gdn_base_v030.py"),
    ("abstract.py", ABS_HUNKS, "mamba_abstract_v030.py"),
    ("qwen_gdn_linear_attn.py", LIN_HUNKS, "qwen_gdn_linear_attn_v030.py"),
    ("gdn_attn.py", ATT_HUNKS, "gdn_attn_v030.py"),
    ("model.py", MOD_HUNKS, "qwen4_exp_model_v030.py"),
)


def _apply(src, hunks, name):
    for i, (old, new) in enumerate(hunks):
        count = src.count(old)
        if count != 1:
            sys.exit(f"replayssm_gdn: anchor {i} in {name} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    return src


def _validate_vendored(orig_dir: str) -> None:
    """Fail closed if the image's replayssm_config.py is not the one the
    vendored superset was built from, or if the vendored files drifted."""
    orig_path = os.path.join(orig_dir, "replayssm_config.py")
    if not os.path.isfile(orig_path):
        sys.exit(f"ERROR: missing {orig_path} (start.sh extracts it from the image)")
    orig = open(orig_path).read()
    for mark in ("def _mamba2_output_only", "unknown ReplaySSM kernel config key"):
        if mark not in orig:
            sys.exit(f"replayssm_gdn: orig replayssm_config.py lacks {mark!r}; "
                     "the vendored superset may not apply")
    vendored = open(os.path.join(HERE, "replayssm_config.py")).read()
    # The mamba2 config must stay bitwise identical: the in-image Mamba2
    # ReplaySSM consumer (selective_state_update_replayssm_output_only.py)
    # depends on it.
    m2_orig = orig[orig.index("def _mamba2_output_only"):]
    m2_orig = m2_orig[:m2_orig.index("\n\n")].rstrip()
    if m2_orig not in vendored:
        sys.exit("replayssm_gdn: vendored replayssm_config.py drifted from the "
                 "image's _mamba2_output_only config")
    kernels = open(os.path.join(HERE, "gdn_replayssm_spec_decode.py")).read()
    for name, src in (("replayssm_config.py", vendored),
                      ("gdn_replayssm_spec_decode.py", kernels)):
        try:
            ast.parse(src)
        except SyntaxError as exc:
            sys.exit(f"replayssm_gdn: vendored {name} does not parse: {exc}")
    for mark in ("def gdn_replayssm_spec_decode", "def commit_gdn_replayssm_spec",
                 "def reset_gdn_replayssm_spec_cursors"):
        if mark not in kernels:
            sys.exit(f"replayssm_gdn: vendored kernel file lacks {mark!r}")


def main(argv) -> int:
    orig_dir = argv[0] if argv else os.path.join(HERE, "orig")
    out_dir = argv[1] if len(argv) > 1 else HERE
    _validate_vendored(orig_dir)
    for orig_name, hunks, out_name in TARGETS:
        orig_path = os.path.join(orig_dir, orig_name)
        if not os.path.isfile(orig_path):
            sys.exit(f"ERROR: missing {orig_path} "
                     "(start.sh extracts it from the image)")
        src = open(orig_path).read()
        if "_REPLAYSSM_GDN" in src:
            sys.exit(f"ERROR: replayssm_gdn orig {orig_name} is already patched")
        src = _apply(src, hunks, orig_name)
        try:
            ast.parse(src)
        except SyntaxError as exc:
            sys.exit(f"replayssm_gdn: patched {out_name} does not parse: {exc}")
        open(os.path.join(out_dir, out_name), "w").write(src)
        print(f"patched {out_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
