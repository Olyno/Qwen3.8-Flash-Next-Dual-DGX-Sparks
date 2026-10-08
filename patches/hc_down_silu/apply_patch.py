#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Fused HC down+SiLU GEMM for the vLLM 0.30.0 lane (unconditional).

Backports vllm-project/vllm#58957 onto the stock v0.30.0 files:

Upstream: https://github.com/vllm-project/vllm/pull/58957 (gau-nernst),
Apache-2.0. The four CuTe-DSL statics are vendored verbatim from the PR and
keep their upstream Apache-2.0/vLLM SPDX headers.

    vllm/models/qwen4_exp/nvidia/ops/cute_dsl/{__init__,hc_down_silu,
    _hc_down_silu_fma,_hc_down_silu_mma}.py
        Vendored verbatim from the PR (new files; mounted alongside the stock
        tree). CuTe-DSL GEMMs with the mHC SiLU epilogue fused into the final
        store: an FMA backend for M <= 4 and a clustered split-K MMA backend
        for M in (4, 48], both bf16 with the production rounding boundary.
    vllm/models/qwen4_exp/nvidia/hyperconnection.py
        GatedResidual._down_and_inject: for 1 <= M <= MAX_FUSED_M (48) the
        merged down+inject projection and the hc_silu run as one GEMM via
        hc_down_silu instead of the Linear module + a separate SiLU kernel;
        larger M keeps the stock path. Eligibility (bf16 weight, K % 8 == 0,
        SM90+) is decided once in __init__; VLLM_BATCH_INVARIANT forces the
        stock path.
    vllm/models/qwen4_exp/nvidia/model.py
        request_hc_down_silu_warmup for every CUDA-graph capture size in the
        fused range, so no CuTe-DSL JIT happens during graph capture
        (v0.30's kernel_warmup calls cutedsl_warmup() at startup when
        kernel_config.enable_cutedsl_warmup is set; otherwise the kernels
        compile lazily on first eager use, same as upstream).

Quality-neutral (the epilogue keeps the production bf16 rounding boundary;
upstream tests pin it against the unfused ll_bf16 + hc_silu reference), so no
toggle. Composes with the ReplaySSM-GDN overlay (patches/replayssm_gdn): its
model.py hunks sit in different regions, so engine/patches.sh feeds this
output as the replayssm patcher's model.py input when REPLAYSSM_GDN is on.

Inputs:  patches/hc_down_silu/orig/{hyperconnection.py,model.py}
Outputs: patches/hc_down_silu/{hyperconnection_v030.py,model_v030.py}
         plus the four vendored statics, validated here and mounted verbatim.
argv[0]/argv[1] override the orig/output directories (used by the test).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# vllm/models/qwen4_exp/nvidia/hyperconnection.py
# ---------------------------------------------------------------------------
HC_IMPORT_OLD = """\
import torch
from torch import nn

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.models.utils import maybe_prefix

from ..common.hyperconnection import (
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from .ops.hc import (
"""
HC_IMPORT_NEW = """\
import torch
from torch import nn

import vllm.envs as envs
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform

from ..common.hyperconnection import (
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from .ops.cute_dsl.hc_down_silu import MAX_FUSED_M, hc_down_silu
from .ops.hc import (
"""
HC_DOC_OLD = """\
    Weights: the norm owns the grouped GemmaRMSNorm affine; the projections
    are vLLM Linear modules (merged replicated linear for down+inject), so
    GEMM dispatch (e.g. the low-latency skinny GEMM) applies through the
    standard quant_method mechanism.
    \"\"\"
"""
HC_DOC_NEW = """\
    Weights: the norm owns the grouped GemmaRMSNorm affine; the projections
    are vLLM Linear modules (merged replicated linear for down+inject).
    Eligible down+inject projections use the fused SiLU GEMM; other projections
    use their Linear module's GEMM dispatch.
    \"\"\"
"""
HC_INIT_OLD = """\
                return_bias=False,
                disable_tp=True,
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
"""
HC_INIT_NEW = """\
                return_bias=False,
                disable_tp=True,
            )
            weight = self.input_mix_weight_down_block_inject.weight
            self._use_hc_down_silu = (
                weight.shape[1] % 8 == 0
                and weight.dtype == torch.bfloat16
                and current_platform.has_device_capability(90)
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
"""
HC_METHOD_OLD = """\
        self.input_mix_weight_up = ReplicatedLinear(
            self.lora_rank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
        )

    def mix(
"""
HC_METHOD_NEW = """\
        self.input_mix_weight_up = ReplicatedLinear(
            self.lora_rank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
        )

    def _down_and_inject(
        self, xn: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        \"\"\"Down projection + SiLU; also returns injection logits if combined.\"\"\"
        if not self.use_combine:
            return hc_silu(self.input_mix_weight_down(xn), self.hc_count), None

        use_fused = (
            self._use_hc_down_silu
            and not envs.VLLM_BATCH_INVARIANT
            and 1 <= xn.shape[0] <= MAX_FUSED_M
        )
        if use_fused:
            return hc_down_silu(
                xn,
                self.input_mix_weight_down_block_inject.weight,
                self.lora_rank,
                self.hc_count,
            )
        down_and_injection = self.input_mix_weight_down_block_inject(xn)
        split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
        lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        return hc_silu(lora, self.hc_count), injection

    def mix(
"""
HC_MIX_OLD = """\
        xn = grouped_gemma_rmsnorm(
            hidden_states,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [M, D]
"""
HC_MIX_NEW = """\
        xn = grouped_gemma_rmsnorm(
            hidden_states,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        lora, injection = self._down_and_inject(xn)
        gate = self.input_mix_weight_up(lora)  # [M, D]
"""
HC_COMBINE_OLD = """\
        hidden_states, xn = hc_combine_norm(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [M, D]
"""
HC_COMBINE_NEW = """\
        hidden_states, xn = hc_combine_norm(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        lora, injection = self._down_and_inject(xn)
        gate = self.input_mix_weight_up(lora)  # [M, D]
"""
HC_HUNKS = (
    (HC_IMPORT_OLD, HC_IMPORT_NEW),
    (HC_DOC_OLD, HC_DOC_NEW),
    (HC_INIT_OLD, HC_INIT_NEW),
    (HC_METHOD_OLD, HC_METHOD_NEW),
    (HC_MIX_OLD, HC_MIX_NEW),
    (HC_COMBINE_OLD, HC_COMBINE_NEW),
)

# ---------------------------------------------------------------------------
# vllm/models/qwen4_exp/nvidia/model.py
# ---------------------------------------------------------------------------
MOD_IMPORT_OLD = """\
from .low_latency_gemm import enable_qwen4_exp_low_latency_gemm
from .ple_layer import Qwen4ExpPLELayer
"""
MOD_IMPORT_NEW = """\
from .low_latency_gemm import enable_qwen4_exp_low_latency_gemm
from .ops.cute_dsl.hc_down_silu import request_hc_down_silu_warmup
from .ple_layer import Qwen4ExpPLELayer
"""
MOD_WARMUP_OLD = """\
        self.set_moe_parameters(self.model.layers)
        enable_qwen4_exp_low_latency_gemm(self, self.model_config.dtype)
"""
MOD_WARMUP_NEW = """\
        self.set_moe_parameters(self.model.layers)
        enable_qwen4_exp_low_latency_gemm(self, self.model_config.dtype)
        if self.model_config.dtype == torch.bfloat16:
            # Precompile the fused HC down+SiLU kernels for every CUDA-graph
            # capture size in the fused dispatch range, so no CuTe-DSL JIT
            # happens during graph capture.
            request_hc_down_silu_warmup(
                vllm_config.compilation_config.cudagraph_capture_sizes or (),
                self.config.hc_lowrank,
                self.config.hc_count,
                self.config.hidden_size * self.config.hc_count,
            )
"""
MOD_HUNKS = ((MOD_IMPORT_OLD, MOD_IMPORT_NEW), (MOD_WARMUP_OLD, MOD_WARMUP_NEW))

# (orig name, hunks, output name, already-patched marker)
TARGETS = (
    ("hyperconnection.py", HC_HUNKS, "hyperconnection_v030.py", "hc_down_silu"),
    ("model.py", MOD_HUNKS, "model_v030.py", "hc_down_silu"),
)

VENDORED = (
    "__init__.py",
    "hc_down_silu.py",
    "_hc_down_silu_fma.py",
    "_hc_down_silu_mma.py",
)


def _apply(src, hunks, name):
    for i, (old, new) in enumerate(hunks):
        count = src.count(old)
        if count != 1:
            sys.exit(f"hc_down_silu: anchor {i} in {name} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    return src


def _validate_vendored() -> None:
    """Fail closed if the vendored CuTe-DSL files drifted or no longer parse."""
    main = open(os.path.join(HERE, "hc_down_silu.py")).read()
    for name in VENDORED:
        src = open(os.path.join(HERE, name)).read()
        try:
            ast.parse(src)
        except SyntaxError as exc:
            sys.exit(f"hc_down_silu: vendored {name} does not parse: {exc}")
    for mark in ("def hc_down_silu", "def request_hc_down_silu_warmup",
                 "MAX_FUSED_M = 48", "register_cutedsl_warmup_provider",
                 "make_fake_tensor"):
        if mark not in main:
            sys.exit(f"hc_down_silu: vendored hc_down_silu.py lacks {mark!r}")


def main(argv) -> int:
    orig_dir = argv[0] if argv else os.path.join(HERE, "orig")
    out_dir = argv[1] if len(argv) > 1 else HERE
    _validate_vendored()
    for orig_name, hunks, out_name, marker in TARGETS:
        orig_path = os.path.join(orig_dir, orig_name)
        if not os.path.isfile(orig_path):
            sys.exit(f"ERROR: missing {orig_path} "
                     "(start.sh extracts it from the image)")
        src = open(orig_path).read()
        if marker in src:
            sys.exit(f"ERROR: hc_down_silu orig {orig_name} is already patched")
        src = _apply(src, hunks, orig_name)
        try:
            ast.parse(src)
        except SyntaxError as exc:
            sys.exit(f"hc_down_silu: patched {out_name} does not parse: {exc}")
        open(os.path.join(out_dir, out_name), "w").write(src)
        print(f"patched {out_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
