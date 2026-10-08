#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""QSA prepare fusion for the vLLM 0.30.0 lane (unconditional).

Backports vllm-project/vllm#57097 ("Qwen4Exp: fuse the main-attention prepare
into the QSA pre-indexer launch") onto the stock v0.30.0 files:

Upstream: https://github.com/vllm-project/vllm/pull/57097 (ShuoleiWang),
Apache-2.0.

    vllm/models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py
        Renamed to ops/qsa_prepare.py (mounted at the new path; the stock file
        stays in the image but nothing imports it anymore). The kernel grows a
        main-attention section: after the indexer K/Q work, one program per
        (token, Q or KV head) does the main QK-norm/RoPE, the gate copy and
        the main K/V cache write (bf16 or fp8-e4m3 with per-tensor scales),
        returning the main Q and gate. qsa_pre_indexer -> qsa_prepare.
    vllm/models/qwen4_exp/nvidia/indexer_qsa.py
        QSAIndexer.forward takes attn/qkv/slot_mapping and returns
        (selection, main_outputs); with attn.use_fused_qsa_prepare the same
        launch also prepares the main attention.
    vllm/models/qwen4_exp/nvidia/qsa.py
        use_fused_qsa_prepare = use_fused_qk_norm_rope_gate and
        indexer.use_fused_pre_indexer. In fused mode _run_qsa skips
        _project_qkv_gate and impl.do_kv_cache_update (the Triton kernel did
        them) and forwards the kernel's Q/gate to impl.forward_qsa.

The only QSAIndexer caller is Qwen4ExpQSAAttention._run_qsa (the MTP drafter
only touches indexer.skip_topk / topk_indices_buffer), so the signature
change is contained. The patch composes with the FP8-KV overlay
(patches/patch_qsa_fp8_kv_v030.py, #55557): that patch's qsa.py hunks sit in
regions this one does not touch, so engine/patches.sh feeds its output as
qsa.py input when KV_CACHE_DTYPE is fp8, and the fused kernel writes the e4m3
cache with the same per-tensor scales the unfused path used.

This overlay also carries the QSA_ROPE_CLAMP deltas (myllmbox/vllm@9ff17c0,
formerly patches/patch_qsa_rope_clamp_v030.py on qsa_pre_indexer.py): the
CLAMP_POS/MAX_POS constexpr pair gates a position clamp in _norm_rope AND in
the new main-attention section (same unchecked cos_sin[pos] load, same SM121
IMA during CUDA-graph warmup). constexpr-off compiles to the bit-exact
upstream kernel; engine/patches.sh passes VLLM_QSA_ROPE_CLAMP=1 when
qsa_rope_clamp: true.

Inputs:  patches/qsa_prepare/orig/{qsa.py,indexer_qsa.py,qsa_pre_indexer.py}
         (extracted from the image; qsa.py may be the FP8-KV overlay output)
Outputs: patches/qsa_prepare/{qsa_v030.py,indexer_qsa_v030.py,qsa_prepare_v030.py}
argv[0]/argv[1] override the orig/output directories (used by the test).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# vllm/models/qwen4_exp/nvidia/qsa.py
# ---------------------------------------------------------------------------
QSA_INIT_OLD = """\
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
"""
QSA_INIT_NEW = """\
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        # One launch does the indexer prepare, the main QK-norm/RoPE/gate and
        # the main K/V cache write (see QSAIndexer.forward); otherwise all of
        # them take the separate kernels.
        self.use_fused_qsa_prepare = (
            self.use_fused_qk_norm_rope_gate and self.indexer.use_fused_pre_indexer
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
"""
QSA_RUNSIG_OLD = """\
    def _run_qsa(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        output_gate: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata
"""
QSA_RUNSIG_NEW = """\
    def _run_qsa(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor | None,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        output: torch.Tensor,
        output_gate: torch.Tensor | None,
        qkv: torch.Tensor,
    ) -> None:
        # query/key/value/output_gate are None when the fused prepare runs
        # inside the indexer launch.
        metadata = get_forward_context().attn_metadata
"""
QSA_CALL_OLD = """\
        selected = self.indexer(
            projected_qk,
            positions,
            self.topk_indices_buffer[:num_tokens],
        )
"""
QSA_CALL_NEW = """\
        selected, main_outputs = self.indexer(
            projected_qk,
            positions,
            self.topk_indices_buffer[:num_tokens],
            attn=self,
            qkv=qkv,
            slot_mapping=main_metadata.slot_mapping,
        )
"""
QSA_KV_OLD = """\
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key,
            value,
            self.kv_cache,
            main_metadata.slot_mapping,
        )
        impl.forward_qsa(
"""
QSA_KV_NEW = """\
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        if main_outputs is None:
            assert key is not None and value is not None
            impl.do_kv_cache_update(
                self,
                key,
                value,
                self.kv_cache,
                main_metadata.slot_mapping,
            )
        else:
            query, output_gate = main_outputs
        assert query is not None and output_gate is not None
        impl.forward_qsa(
"""
QSA_FWD_OLD = """\
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        assert gate is not None
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        # Keep the index projection outside the eager break.
        projected_qk, _ = self.indexer.index_qk_proj(hidden_states)
        self._run_qsa(
            projected_qk,
            positions,
            query,
            key,
            value,
            attn_output,
            gate,
        )
"""
QSA_FWD_NEW = """\
        qkv, _ = self.qkv_proj(hidden_states)
        num_tokens = hidden_states.shape[0]
        if not self.use_fused_qsa_prepare:
            q, k, v, gate = self._project_qkv_gate(qkv, positions)
            assert gate is not None
            query = q.view(num_tokens, self.num_heads, self.head_dim)
            key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
            value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        else:
            # Norm/RoPE/gate and the K/V cache write happen inside _run_qsa.
            query = key = value = gate = None
        attn_output = qkv.new_empty(num_tokens, self.num_heads, self.head_dim)
        # Keep the index projection outside the eager break.
        projected_qk, _ = self.indexer.index_qk_proj(hidden_states)
        self._run_qsa(
            projected_qk,
            positions,
            query,
            key,
            value,
            attn_output,
            gate,
            qkv,
        )
"""
QSA_HUNKS = (
    (QSA_INIT_OLD, QSA_INIT_NEW),
    (QSA_RUNSIG_OLD, QSA_RUNSIG_NEW),
    (QSA_CALL_OLD, QSA_CALL_NEW),
    (QSA_KV_OLD, QSA_KV_NEW),
    (QSA_FWD_OLD, QSA_FWD_NEW),
)

# ---------------------------------------------------------------------------
# vllm/models/qwen4_exp/nvidia/indexer_qsa.py
# ---------------------------------------------------------------------------
IDX_IMPORT_OLD = """\
from typing import cast
"""
IDX_IMPORT_NEW = """\
from typing import TYPE_CHECKING, cast
"""
IDX_IMPORT2_OLD = """\
from .ops.qsa_pre_indexer import qsa_pre_indexer
"""
IDX_IMPORT2_NEW = """\
from .ops.qsa_prepare import qsa_prepare

if TYPE_CHECKING:
    from .qsa import Qwen4ExpQSAAttention
"""
IDX_SIG_OLD = """\
    def forward(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        \"\"\"Update side caches and select token indices from pre-projected Q/K.

        Returns the packed buffer of shape [num_tokens, output_width + 1]:
        the leading ``output_width`` columns are ``-1``-padded
        request-relative token indices, and the trailing column is the row's
        valid-entry count (the attention kernel's loop bound, never a token
        index).
        \"\"\"
"""
IDX_SIG_NEW = """\
    def forward(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        attn: "Qwen4ExpQSAAttention",
        qkv: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        \"\"\"Update side caches and select token indices from pre-projected Q/K.

        Returns the packed buffer of shape [num_tokens, output_width + 1]:
        the leading ``output_width`` columns are ``-1``-padded
        request-relative token indices, and the trailing column is the row's
        valid-entry count (the attention kernel's loop bound, never a token
        index).

        With ``attn.use_fused_qsa_prepare``, the same launch also writes
        ``attn``'s K/V into ``attn.kv_cache`` at ``slot_mapping`` and prepares
        its Q and gate from ``qkv``, returned as the second element (None
        otherwise). ``qkv`` and ``slot_mapping`` are only read in that mode.
        \"\"\"
"""
IDX_META_OLD = """\
        metadata = self._metadata()
        if metadata is None:
            # Preserve step-0 indices when later MTP steps reuse the buffer.
            if self.skip_topk and out is not None:
                return out
            result = torch.full(
                (projected_qk.shape[0], self.packed_output_width),
                -1,
                dtype=torch.int32,
                device=projected_qk.device,
            )
            # Inert rows carry a zero valid count (empty loop bound), not -1.
            result[:, -1] = 0
            if out is not None:
                out.copy_(result)
                return out
            return result
"""
IDX_META_NEW = """\
        metadata = self._metadata()
        if metadata is None:
            # Preserve step-0 indices when later MTP steps reuse the buffer.
            if self.skip_topk and out is not None:
                return out, None
            result = torch.full(
                (projected_qk.shape[0], self.packed_output_width),
                -1,
                dtype=torch.int32,
                device=projected_qk.device,
            )
            # Inert rows carry a zero valid count (empty loop bound), not -1.
            result[:, -1] = 0
            if out is not None:
                out.copy_(result)
                return out, None
            return result, None
"""
IDX_FUSED_OLD = """\
        if self.use_fused_pre_indexer:
            q = projected_q.new_empty(
                num_tokens,
                self.index_n_heads,
                self.index_head_dim,
                dtype=self.indexer_dtype,
            )
            qsa_pre_indexer(
"""
IDX_FUSED_NEW = """\
        main_outputs: tuple[torch.Tensor, torch.Tensor] | None = None
        if attn.use_fused_qsa_prepare:
            if qkv is None or slot_mapping is None:
                raise ValueError("fused QSA prepare requires qkv and slot_mapping")
            q = projected_q.new_empty(
                num_tokens,
                self.index_n_heads,
                self.index_head_dim,
                dtype=self.indexer_dtype,
            )
            main_kv_cache = attn.kv_cache.transpose(1, 2)
            if attn.kv_cache_dtype in ("fp8", "fp8_e4m3"):
                main_kv_cache = main_kv_cache.view(torch.float8_e4m3fn)
            main_outputs = qsa_prepare(
"""
IDX_KWARGS_OLD = """\
                rope_pos_offset=(
                    raw_key_state_cache.rope_position_offset
                    if raw_key_state_cache.rope_position_cache is not None
                    else None
                ),
            )
        else:
            # Unfused reference path
"""
IDX_KWARGS_NEW = """\
                rope_pos_offset=(
                    raw_key_state_cache.rope_position_offset
                    if raw_key_state_cache.rope_position_cache is not None
                    else None
                ),
                main_qkv=qkv[:num_tokens],
                main_q_norm_weight=attn.q_norm.weight,
                main_k_norm_weight=attn.k_norm.weight,
                main_eps=attn.q_norm.variance_epsilon,
                main_kv_cache=main_kv_cache,
                main_slot_mapping=slot_mapping[:num_tokens],
                main_k_scale=attn._k_scale_float,
                main_v_scale=attn._v_scale_float,
            )
        else:
            # Unfused reference path
"""
IDX_SKIP_OLD = """\
        if self.skip_topk:
            if out is None:
                raise RuntimeError("QSA top-k reuse requires an output buffer")
            return out
"""
IDX_SKIP_NEW = """\
        if self.skip_topk:
            if out is None:
                raise RuntimeError("QSA top-k reuse requires an output buffer")
            return out, main_outputs
"""
IDX_RET_OLD = """\
        expand_qsa_block_indices(
            block_indices,
            compressed_metadata.logical_positions[:num_tokens],
            visible_blocks,
            self.compress_ratio,
            self.token_topk,
            out,
        )
        return out
"""
IDX_RET_NEW = """\
        expand_qsa_block_indices(
            block_indices,
            compressed_metadata.logical_positions[:num_tokens],
            visible_blocks,
            self.compress_ratio,
            self.token_topk,
            out,
        )
        return out, main_outputs
"""
IDX_HUNKS = (
    (IDX_IMPORT_OLD, IDX_IMPORT_NEW),
    (IDX_IMPORT2_OLD, IDX_IMPORT2_NEW),
    (IDX_SIG_OLD, IDX_SIG_NEW),
    (IDX_META_OLD, IDX_META_NEW),
    (IDX_FUSED_OLD, IDX_FUSED_NEW),
    (IDX_KWARGS_OLD, IDX_KWARGS_NEW),
    (IDX_SKIP_OLD, IDX_SKIP_NEW),
    (IDX_RET_OLD, IDX_RET_NEW),
)

# ---------------------------------------------------------------------------
# vllm/models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py -> ops/qsa_prepare.py
# (vllm#57097 hunks plus the CLAMP_POS deltas, env-gated bit-exact when off)
# ---------------------------------------------------------------------------
PREP_DOC_OLD = """\
\"\"\"Fused QSA pre-indexer kernel for Qwen4Exp.\"\"\"

import torch

from vllm.triton_utils import tl, triton
"""
PREP_DOC_NEW = """\
\"\"\"Fused QSA prepare kernel for Qwen4Exp.\"\"\"

import os

import torch

from vllm.triton_utils import tl, triton
"""
PREP_HELPERS_OLD = """\
    return tl.reshape(result, (TILE_T, TILE_H, D))


@triton.jit(
    do_not_specialize=[
"""
PREP_HELPERS_NEW = """\
    return tl.reshape(result, (TILE_T, TILE_H, D))


@triton.jit
def _to_dst_dtype(x, dst, scale):
    \"\"\"Round to BF16 like the unfused path, then scale for an FP8 destination.\"\"\"
    out_ty = dst.dtype.element_ty
    x = x.to(tl.bfloat16)
    if out_ty == tl.float8e4nv:
        x = x.to(tl.float32) / scale
    return x.to(out_ty)


@triton.jit
def _store_rotated(dst, y, o1, o2, scale):
    \"\"\"Store a normalized head whose first ``2 * len(o1)`` dims are rotated.\"\"\"
    HALF: tl.constexpr = o1.shape[0]
    dims = tl.arange(0, y.shape[0])
    rot = tl.arange(0, HALF)
    tl.store(dst + dims, _to_dst_dtype(y, dst, scale), mask=dims >= 2 * HALF)
    tl.store(dst + rot, _to_dst_dtype(o1, dst, scale))
    tl.store(dst + HALF + rot, _to_dst_dtype(o2, dst, scale))


@triton.jit(
    do_not_specialize=[
"""
PREP_KNAME_OLD = """\
def _qsa_pre_indexer_kernel(
"""
PREP_KNAME_NEW = """\
def _qsa_prepare_kernel(
"""
PREP_KSIG_OLD = """\
    CACHE_HAS_ROPE_POS: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
):
    pid = tl.program_id(0)
    # K work occupies the first programs; the remaining programs tile Q. This
"""
PREP_KSIG_NEW = """\
    CACHE_HAS_ROPE_POS: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
    main_qkv_ptr,
    main_q_norm_weight_ptr,
    main_k_norm_weight_ptr,
    main_eps,
    main_q_out_ptr,
    main_gate_out_ptr,
    main_cache_ptr,
    main_cache_stride_block,
    main_cache_stride_token,
    main_cache_stride_head,
    main_slots_ptr,
    main_k_scale,
    main_v_scale,
    MAIN_HQ: tl.constexpr,
    MAIN_HK: tl.constexpr,
    MAIN_D: tl.constexpr,
    MAIN_PAGE_SIZE: tl.constexpr,
    CLAMP_POS: tl.constexpr,
    MAX_POS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_index_work = num_k_work + tl.cdiv(num_tokens, TILE_T_Q) * tl.cdiv(HQ, TILE_H_Q)
    if pid >= num_index_work:
        # Main attention: one program per (token, Q or KV head) after the
        # indexer work. RoPE covers the first D // 2 dims, the width of the
        # shared cos/sin table.
        main_pid = pid - num_index_work
        token = main_pid // (MAIN_HQ + MAIN_HK)
        head = main_pid % (MAIN_HQ + MAIN_HK)
        HALF: tl.constexpr = D // 4
        dims = tl.arange(0, MAIN_D)
        rot = tl.arange(0, HALF)
        row = main_qkv_ptr + token * (2 * (MAIN_HQ + MAIN_HK) * MAIN_D)
        is_k = head >= MAIN_HQ
        kv_head = head - MAIN_HQ
        if is_k:
            src = row + (2 * MAIN_HQ + kv_head) * MAIN_D
            weight_ptr = main_k_norm_weight_ptr
        else:
            src = row + 2 * head * MAIN_D
            weight_ptr = main_q_norm_weight_ptr
        x = tl.load(src + dims).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / MAIN_D + main_eps)
        w = tl.load(weight_ptr + dims).to(tl.float32) + 1.0
        y = x * inv_rms * w
        x1 = tl.load(src + rot).to(tl.float32)
        x2 = tl.load(src + HALF + rot).to(tl.float32)
        w1 = tl.load(weight_ptr + rot).to(tl.float32) + 1.0
        w2 = tl.load(weight_ptr + HALF + rot).to(tl.float32) + 1.0
        x1 = (x1 * inv_rms * w1).to(tl.bfloat16).to(tl.float32)
        x2 = (x2 * inv_rms * w2).to(tl.bfloat16).to(tl.float32)
        pos = tl.load(pos_ptr + token * pos_stride_token).to(tl.int64)
        if IS_2D_POSITIONS:
            pos_h = tl.load(pos_ptr + pos_stride_axis + token * pos_stride_token)
            pos_w = tl.load(pos_ptr + 2 * pos_stride_axis + token * pos_stride_token)
            is_h = (rot % 3 == 1) & (rot < 3 * MROPE_H)
            is_w = (rot % 3 == 2) & (rot < 3 * MROPE_W)
            pos = tl.where(
                is_h, pos_h.to(tl.int64), tl.where(is_w, pos_w.to(tl.int64), pos)
            )
        if CLAMP_POS:
            # CUDA-graph warmup feeds dummy positions beyond the cos/sin
            # table; stock loads cos_sin[pos] unchecked, an IMA on SM121.
            pos = tl.minimum(tl.maximum(pos, 0), MAX_POS)
        cos = tl.load(cos_sin_ptr + pos * (D // 2) + rot).to(tl.float32)
        sin = tl.load(cos_sin_ptr + pos * (D // 2) + HALF + rot).to(tl.float32)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        if is_k:
            slot = tl.load(main_slots_ptr + token).to(tl.int64)
            if slot >= 0:
                dst = (
                    main_cache_ptr
                    + (slot // MAIN_PAGE_SIZE) * main_cache_stride_block
                    + (slot % MAIN_PAGE_SIZE) * main_cache_stride_token
                    + kv_head * main_cache_stride_head
                )
                _store_rotated(dst, y, o1, o2, main_k_scale)
                v = tl.load(src + MAIN_HK * MAIN_D + dims)
                tl.store(dst + MAIN_D + dims, _to_dst_dtype(v, dst, main_v_scale))
        else:
            out = (token * MAIN_HQ + head) * MAIN_D
            _store_rotated(main_q_out_ptr + out, y, o1, o2, None)
            gate = tl.load(src + MAIN_D + dims)
            tl.store(main_gate_out_ptr + out + dims, gate)
        return
    # K work occupies the first programs; the remaining programs tile Q. This
"""
PREP_NORMROPE_OLD = """\
    IS_MROPE: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
):
    \"\"\"Apply Gemma RMSNorm and selected-axis NeoX RoPE to register rows.\"\"\"
    TILE_T: tl.constexpr = x.shape[0]
"""
PREP_NORMROPE_NEW = """\
    IS_MROPE: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
    CLAMP_POS: tl.constexpr,
    MAX_POS: tl.constexpr,
):
    \"\"\"Apply Gemma RMSNorm and selected-axis NeoX RoPE to register rows.\"\"\"
    if CLAMP_POS:
        # CUDA-graph warmup feeds dummy positions beyond the cos/sin table;
        # stock loads cos_sin[pos] unchecked, an IMA on SM121 (GB10).
        pos_t = tl.minimum(tl.maximum(pos_t, 0), MAX_POS)
        pos_h = tl.minimum(tl.maximum(pos_h, 0), MAX_POS)
        pos_w = tl.minimum(tl.maximum(pos_w, 0), MAX_POS)
    TILE_T: tl.constexpr = x.shape[0]
"""
PREP_QCALL_OLD = """\
            IS_2D_POSITIONS,
            MROPE_H,
            MROPE_W,
        )
"""
PREP_QCALL_NEW = """\
            IS_2D_POSITIONS,
            MROPE_H,
            MROPE_W,
            CLAMP_POS,
            MAX_POS,
        )
"""
PREP_KCALL_OLD = """\
                IS_K_MROPE,
                MROPE_H,
                MROPE_W,
            )
"""
PREP_KCALL_NEW = """\
                IS_K_MROPE,
                MROPE_H,
                MROPE_W,
                CLAMP_POS,
                MAX_POS,
            )
"""
PREP_WRAPPER_OLD = """\
def qsa_pre_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    eps: float,
    q_out: torch.Tensor,
    state_cache: torch.Tensor,
    state_slots: torch.Tensor,
    state_block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_slots: torch.Tensor,
    k_work_metadata: torch.Tensor,
    *,
    compress_ratio: int,
    mrope_section: tuple[int, int, int] | None,
    rope_pos_offset: int | None,
) -> None:
    \"\"\"Normalize Q, compress K, then update the circular raw state.\"\"\"
    num_tokens = q.shape[0]
    if num_tokens == 0:
        return
"""
PREP_WRAPPER_NEW = """\
def qsa_prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    eps: float,
    q_out: torch.Tensor,
    state_cache: torch.Tensor,
    state_slots: torch.Tensor,
    state_block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_slots: torch.Tensor,
    k_work_metadata: torch.Tensor,
    *,
    compress_ratio: int,
    mrope_section: tuple[int, int, int] | None,
    rope_pos_offset: int | None,
    main_qkv: torch.Tensor,
    main_q_norm_weight: torch.Tensor,
    main_k_norm_weight: torch.Tensor,
    main_eps: float,
    main_kv_cache: torch.Tensor,
    main_slot_mapping: torch.Tensor,
    main_k_scale: float,
    main_v_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    \"\"\"Normalize Q, compress K, then update the circular raw state.

    Also prepares the main attention (QK-norm/RoPE, gate copy, K/V cache write)
    and returns its Q and gate.
    \"\"\"
    num_tokens = q.shape[0]
    main_head_dim = main_kv_cache.shape[-1] // 2
    num_main_kv_heads = main_kv_cache.shape[2]
    num_main_q_heads = main_qkv.shape[1] // (2 * main_head_dim) - num_main_kv_heads
    main_q_out = main_qkv.new_empty(num_tokens, num_main_q_heads, main_head_dim)
    main_gate_out = torch.empty_like(main_q_out)
    if num_tokens == 0:
        return main_q_out, main_gate_out
"""
PREP_ASSERTS_OLD = """\
    section = mrope_section if mrope_section is not None else (0, 0, 0)
    assert len(section) == 3
"""
PREP_ASSERTS_NEW = """\
    section = mrope_section if mrope_section is not None else (0, 0, 0)
    assert len(section) == 3
    qkv_width = 2 * (num_main_q_heads + num_main_kv_heads) * main_head_dim
    assert main_qkv.shape == (num_tokens, qkv_width) and main_qkv.is_contiguous()
    assert main_slot_mapping.shape == (num_tokens,)
"""
PREP_LAUNCH_OLD = """\
    num_k_work = k_work_metadata.shape[0]
    num_q_work = triton.cdiv(num_tokens, TILE_T_Q) * triton.cdiv(num_q_heads, TILE_H_Q)
    _qsa_pre_indexer_kernel[(num_k_work + num_q_work,)](
"""
PREP_LAUNCH_NEW = """\
    num_k_work = k_work_metadata.shape[0]
    num_q_work = triton.cdiv(num_tokens, TILE_T_Q) * triton.cdiv(num_q_heads, TILE_H_Q)
    num_main_work = num_tokens * (num_main_q_heads + num_main_kv_heads)
    _qsa_prepare_kernel[(num_k_work + num_q_work + num_main_work,)](
"""
PREP_KWARGS_OLD = """\
        CACHE_HAS_ROPE_POS=cache_has_rope_pos,
        MROPE_H=section[1],
        MROPE_W=section[2],
        num_warps=1,
    )


__all__ = ["qsa_pre_indexer"]
"""
PREP_KWARGS_NEW = """\
        CACHE_HAS_ROPE_POS=cache_has_rope_pos,
        MROPE_H=section[1],
        MROPE_W=section[2],
        main_qkv_ptr=main_qkv,
        main_q_norm_weight_ptr=main_q_norm_weight,
        main_k_norm_weight_ptr=main_k_norm_weight,
        main_eps=main_eps,
        main_q_out_ptr=main_q_out,
        main_gate_out_ptr=main_gate_out,
        main_cache_ptr=main_kv_cache,
        main_cache_stride_block=main_kv_cache.stride(0),
        main_cache_stride_token=main_kv_cache.stride(1),
        main_cache_stride_head=main_kv_cache.stride(2),
        main_slots_ptr=main_slot_mapping,
        main_k_scale=main_k_scale,
        main_v_scale=main_v_scale,
        MAIN_HQ=num_main_q_heads,
        MAIN_HK=num_main_kv_heads,
        MAIN_D=main_head_dim,
        MAIN_PAGE_SIZE=main_kv_cache.shape[1],
        CLAMP_POS=os.environ.get("VLLM_QSA_ROPE_CLAMP", "0") == "1",
        MAX_POS=max(int(cos_sin_cache.shape[0]) - 1, 0),
        num_warps=1,
    )
    return main_q_out, main_gate_out


__all__ = ["qsa_prepare"]
"""
PREP_HUNKS = (
    (PREP_DOC_OLD, PREP_DOC_NEW),
    (PREP_NORMROPE_OLD, PREP_NORMROPE_NEW),
    (PREP_HELPERS_OLD, PREP_HELPERS_NEW),
    (PREP_KNAME_OLD, PREP_KNAME_NEW),
    (PREP_KSIG_OLD, PREP_KSIG_NEW),
    (PREP_QCALL_OLD, PREP_QCALL_NEW),
    (PREP_KCALL_OLD, PREP_KCALL_NEW),
    (PREP_WRAPPER_OLD, PREP_WRAPPER_NEW),
    (PREP_ASSERTS_OLD, PREP_ASSERTS_NEW),
    (PREP_LAUNCH_OLD, PREP_LAUNCH_NEW),
    (PREP_KWARGS_OLD, PREP_KWARGS_NEW),
)

# (orig name, hunks, output name, already-patched marker)
TARGETS = (
    ("qsa.py", QSA_HUNKS, "qsa_v030.py", "use_fused_qsa_prepare"),
    ("indexer_qsa.py", IDX_HUNKS, "indexer_qsa_v030.py", "qsa_prepare"),
    ("qsa_pre_indexer.py", PREP_HUNKS, "qsa_prepare_v030.py", "qsa_prepare"),
)


def _apply(src, hunks, name):
    for i, (old, new) in enumerate(hunks):
        count = src.count(old)
        if count != 1:
            sys.exit(f"qsa_prepare: anchor {i} in {name} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    return src


def main(argv) -> int:
    orig_dir = argv[0] if argv else os.path.join(HERE, "orig")
    out_dir = argv[1] if len(argv) > 1 else HERE
    for orig_name, hunks, out_name, marker in TARGETS:
        orig_path = os.path.join(orig_dir, orig_name)
        if not os.path.isfile(orig_path):
            sys.exit(f"ERROR: missing {orig_path} "
                     "(start.sh extracts it from the image)")
        src = open(orig_path).read()
        if marker in src:
            sys.exit(f"ERROR: qsa_prepare orig {orig_name} is already patched")
        src = _apply(src, hunks, orig_name)
        try:
            ast.parse(src)
        except SyntaxError as exc:
            sys.exit(f"qsa_prepare: patched {out_name} does not parse: {exc}")
        open(os.path.join(out_dir, out_name), "w").write(src)
        print(f"patched {out_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
