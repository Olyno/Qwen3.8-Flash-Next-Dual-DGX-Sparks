#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""QSA QKV/gate + indexer projection merge for the vLLM 0.30.0 lane (opt-in:
qsa_qkvg_merge / QSA_QKVG_MERGE, TP=1 only).

Backports vllm-project/vllm#59533 ("Qwen4Exp: merge the QSA QKV/gate and
indexer projections") onto the qsa_head_dim overlay's output:

Upstream: https://github.com/vllm-project/vllm/pull/59533, Apache-2.0.

    vllm/models/qwen4_exp/nvidia/qsa.py
    - Qwen4ExpQSAQKVIndexerLinear (MergedColumnParallelLinear): packs the
      sharded Q/gate, K, V and the replicated indexer Q/K into one GEMM; its
      weight_loader remaps the replicated indexer shard (id 3) to rank 0 on
      every TP rank.
    - Qwen4ExpQSAAttention.__init__ replaces qkv_proj with the merged module
      and drops indexer.index_qk_proj when lora_config is None (upstream's
      LoRA gate; under LoRA the projections stay separate).
    - A load_weights with WeightsMapper(orig_to_new_stacked=
      {"indexer.index_qk_proj.": ("qkv_proj.", 3)}) stacks the checkpoint's
      indexer weight into the merged projection, so the PR's model.py
      packed_modules_mapping hunks are not needed.
    - forward splits the fused output back into qkv and projected_qk; the
      indexer slice is a row-strided column view, which the qsa_prepare
      overlay's main_qkv_stride_token fold-in (vllm#59533 prepare side)
      accepts.

Pruned from the PR: the SM90/SM100/SM103 TP=4 plan-table edits and the
QWEN4_EXP_GEMM_PLANS_BY_CAPABILITY dispatch refactor (this lane's plan table
is patches/gb10_skinny_gemm's vendored SM12x one, which carries the merged
(13952, 2560) rows separately), the model.py hunks (see above), and the
upstream tests. The merged shape exists at TP=1 only, so engine/config.sh
hard-gates QSA_QKVG_MERGE on tensor_parallel_size == 1: at TP=2 the merged
GEMM has no skinny plan and falls back to cuBLAS, a likely regression.

Delete this file, its test and its overlay block in engine/patches.sh once
the image ships vllm#59533 natively (the qsa_prepare fold-in goes with it).

Inputs:  patches/qsa_qkvg_merge/orig/qsa.py (the qsa_head_dim overlay's
         output; engine/patches.sh chains it)
Outputs: patches/qsa_qkvg_merge/qsa_v030.py
argv[0]/argv[1] override the orig/output directories (used by the test).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

IMPORT_ABC_OLD = """\
from __future__ import annotations

from typing import ClassVar, cast
"""
IMPORT_ABC_NEW = """\
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import ClassVar, cast
"""
IMPORT_LINEAR_OLD = """\
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
"""
IMPORT_LINEAR_NEW = """\
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
"""
IMPORT_UTILS_OLD = """\
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
"""
IMPORT_UTILS_NEW = """\
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
"""
CLASS_OLD = """\
from .indexer_qsa import QSAIndexer


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
"""
CLASS_NEW = """\
from .indexer_qsa import QSAIndexer


class Qwen4ExpQSAQKVIndexerLinear(MergedColumnParallelLinear):
    \"\"\"Pack sharded Q/gate, K, V and replicated indexer Q/K in one GEMM.\"\"\"

    def __init__(
        self,
        qkv_proj: QKVParallelLinear,
        index_size: int,
        quant_config: QuantizationConfig | None,
    ) -> None:
        self.num_kv_head_replicas = qkv_proj.num_kv_head_replicas
        super().__init__(
            input_size=qkv_proj.input_size,
            output_sizes=[*qkv_proj.output_sizes, index_size * qkv_proj.tp_size],
            bias=False,
            params_dtype=qkv_proj.params_dtype,
            quant_config=quant_config,
            prefix=qkv_proj.prefix,
        )

    def _load_shard(
        self,
        loader: Callable[..., None],
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | tuple[int, ...] | None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id == 3:
            shard_rank = 0
        elif loaded_shard_id in (1, 2):
            shard_rank = tp_rank // self.num_kv_head_replicas
        else:
            shard_rank = tp_rank
        self.tp_rank = shard_rank
        if param_tp_rank is not None:
            param.tp_rank = shard_rank
        try:
            loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None) -> None:
        self._load_shard(super().weight_loader, param, loaded_weight, loaded_shard_id)

    def weight_loader_v2(self, param, loaded_weight, loaded_shard_id=None) -> None:
        self._load_shard(
            super().weight_loader_v2, param, loaded_weight, loaded_shard_id
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        def remap_shards():
            for name, weight in weights:
                shard_id = getattr(weight, "shard_id", None)
                if isinstance(shard_id, str):
                    weight = weight.detach()
                    weight.shard_id = {"q": 0, "k": 1, "v": 2}[shard_id]
                yield name, weight

        return super().load_weights(remap_shards())


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
"""
INIT_OLD = """\
        self.use_fused_qsa_prepare = (
            self.use_fused_qk_norm_rope_gate and self.indexer.use_fused_pre_indexer
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
"""
INIT_NEW = """\
        self.use_fused_qsa_prepare = (
            self.use_fused_qk_norm_rope_gate and self.indexer.use_fused_pre_indexer
        )
        self.fuse_indexer_projection = vllm_config.lora_config is None
        if self.fuse_indexer_projection:
            self.index_qk_size = self.indexer.index_qk_proj.output_size
            self.qkv_proj = Qwen4ExpQSAQKVIndexerLinear(
                self.qkv_proj,
                self.index_qk_size,
                model.without_modelopt_fp4(quant_config),
            )
            del self.indexer.index_qk_proj

        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
"""
LOAD_WEIGHTS_OLD = """\
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
"""
LOAD_WEIGHTS_NEW = """\
        static_context[self.layer_name] = self

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = None
        if self.fuse_indexer_projection:
            mapper = WeightsMapper(
                orig_to_new_stacked={
                    "indexer.index_qk_proj.": ("qkv_proj.", 3),
                }
            )
        return AutoWeightsLoader(self).load_weights(weights, mapper=mapper)

    def get_attn_backend(self) -> type[AttentionBackend]:
"""
FWD_SPLIT_OLD = """\
        qkv, _ = self.qkv_proj(hidden_states)
        num_tokens = hidden_states.shape[0]
        if not self.use_fused_qsa_prepare:
"""
FWD_SPLIT_NEW = """\
        qkv, _ = self.qkv_proj(hidden_states)
        if self.fuse_indexer_projection:
            qkv, projected_qk = qkv.split(
                [2 * self.q_size + 2 * self.kv_size, self.index_qk_size], dim=-1
            )
        else:
            projected_qk, _ = self.indexer.index_qk_proj(hidden_states)
        num_tokens = hidden_states.shape[0]
        if not self.use_fused_qsa_prepare:
"""
FWD_PROJ_OLD = """\
        attn_output = qkv.new_empty(num_tokens, self.num_heads, self.head_dim)
        # Keep the index projection outside the eager break.
        projected_qk, _ = self.indexer.index_qk_proj(hidden_states)
        self._run_qsa(
"""
FWD_PROJ_NEW = """\
        attn_output = qkv.new_empty(num_tokens, self.num_heads, self.head_dim)
        self._run_qsa(
"""
HUNKS = (
    (IMPORT_ABC_OLD, IMPORT_ABC_NEW),
    (IMPORT_LINEAR_OLD, IMPORT_LINEAR_NEW),
    (IMPORT_UTILS_OLD, IMPORT_UTILS_NEW),
    (CLASS_OLD, CLASS_NEW),
    (INIT_OLD, INIT_NEW),
    (LOAD_WEIGHTS_OLD, LOAD_WEIGHTS_NEW),
    (FWD_SPLIT_OLD, FWD_SPLIT_NEW),
    (FWD_PROJ_OLD, FWD_PROJ_NEW),
)


def main(argv) -> int:
    orig_dir = argv[0] if argv else os.path.join(HERE, "orig")
    out_dir = argv[1] if len(argv) > 1 else HERE
    orig_path = os.path.join(orig_dir, "qsa.py")
    if not os.path.isfile(orig_path):
        sys.exit(f"ERROR: missing {orig_path} "
                 "(engine/patches.sh chains the qsa_head_dim output here)")
    src = open(orig_path).read()
    if "fuse_indexer_projection" in src:
        sys.exit("ERROR: qsa_qkvg_merge orig qsa.py is already patched")
    for i, (old, new) in enumerate(HUNKS):
        count = src.count(old)
        if count != 1:
            sys.exit(f"qsa_qkvg_merge: anchor {i} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"qsa_qkvg_merge: patched qsa_v030.py does not parse: {exc}")
    open(os.path.join(out_dir, "qsa_v030.py"), "w").write(src)
    print("patched qsa_v030.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
