#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""GB10 (SM12x) plan table for the Qwen4Exp skinny decode GEMM, vendored from
myllmbox's patches/gb10-skinny-gemm.patch (TP=2 shapes) and upstream
vllm#59753 / vllm#59632 (TP=1 shapes, plus upstream-measured TP=2 entries),
plus the vllm#60027 strided-A dispatch fix.

Upstream: https://github.com/vllm-project/vllm/pull/59753 (stecasta) and
https://github.com/vllm-project/vllm/pull/59632 (sudhanshu112233shukla),
Apache-2.0. The myllmbox-only rows come from
https://github.com/myllmbox/qwen38-flash-next-recipe (MIT, see
licenses/myllmbox-MIT.LICENSE), whose table was first tuned by
@sethforprivacy (https://github.com/vllm-project/vllm/issues/59605).

vLLM 0.30's models/qwen4_exp/nvidia/low_latency_gemm.py (already shipped and
already wired into model.py and mtp.py by the image) carries plans only for
SM103 and SM90 at TP=4, so on the GB10 every decode-sized BF16 projection
falls back to cuBLAS (SM80 WMMA 16x16 kernels). This adds an SM12x plan table
keyed by local (N, K) shape and exact token count M, covering TP=1 and TP=2
local shapes; a missing shape or M keeps the standard linear path, so other
GPUs and TP sizes are unchanged. MBX_SKINNY_GEMM_SM12X=0 in the container
restores the standard linear path.

vllm#59632 (merged 2026-10-05) upstreamed the TP=2-local BF16 shapes
(48, 2560), (2560, 3072), (6656, 2560), (8192, 2560) and (124160, 2560).
The rows vendored here from the myllmbox table are config-identical to the
merged PR (verified row-by-row against the PR diff), so the backport adds no
table delta; tests/test_gb10_skinny_gemm_v030.py now pins the
upstream-measured rows so a future table edit cannot silently drift from
upstream.

vllm#60027 (merged 2026-10-05) is folded in on both sides, matching upstream:
the SM121 (10240, 320) HC-up plan never fired because the fused HC-down
output reaches the projection as a column slice of a wider buffer and
_runtime_ok only accepted packed row-major inputs, so every call fell back
to cuBLAS. low_latency_gemm._runtime_ok now accepts any 2D row-major A whose
row stride keeps the vectorized loads aligned (row_stride_ok), and the
kernel wrapper (model_executor/kernels/linear/cute_dsl/skinny_gemm.py, a new
overlay target) drops its blanket contiguity requirement on A for the same
check. The PR base matches v0.30 in every touched region (verified against
the pinned v0.30.0 image), so the hunks are verbatim.

Delete this file and its overlay in engine/patches.sh once the image ships
SM12x plans natively.
"""
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = {
    "models/qwen4_exp/nvidia/low_latency_gemm.py": "low_latency_gemm.py",
    "model_executor/kernels/linear/cute_dsl/skinny_gemm.py": "skinny_gemm.py",
}
HUNK_HEADER = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")

UPSTREAM_DIFF = r'''
GB10 (SM12x) plans for the Qwen4Exp skinny decode GEMM.
- vLLM 0.30's qwen4_exp_low_latency_gemm (a CuTe-DSL skinny GEMM) carries plans only for SM103 and SM90 TP=4 shapes,
  so on the GB10 every decode-sized BF16 projection stays on cuBLAS, which picks SM80 WMMA 16x16 kernels (one warp
  per CTA, 125-225 GB/s); the SM12x plans run at 240-255 GB/s, 1.4-1.5x at M=1 (the draft passes) and up to 7x on
  the latency-bound shapes (timed in CUDA graphs with the weights rotated past L2).
- TP=1 shapes are vendored verbatim from upstream vllm#59753 (merged 2026-10-02, post-v0.30; measured on a DGX
  Spark GB10 with the NVFP4 checkpoint: -9.5 % ITL / +10 % tok/s at concurrency 1, the (248320, 2560) LM head
  alone -8.3 % ITL). TP=2-only shapes come from upstream vllm#59632 where it measured them, otherwise from
  myllmbox's gb10-skinny-gemm patch (contributed by @sethforprivacy, timed 2026-10-01 on two GB10 pairs; each
  kept entry beat cuBLAS F.linear by >= 3 %). The M 6 / 12 rows (K=5 verify) exist only in the myllmbox table
  and are merged in alongside the upstream rows.
- On hibrid48 the table covers the BF16 layers: QSA Q/gate/K/V and output, shared expert gate/up/down and its
  sigmoid gate, router, GDN B/A and fused QKVZ, the hyper-connection mixes, the MTP fc projections and the LM
  head, at M 1-16 plus the K=5 verify rows 6 / 12. Plans are keyed by local (N, K) shape and exact token count M,
  so one table serves TP=1 and TP=2: sharded projections differ in N per TP size and get their own entries;
  shapes shared between TP sizes (replicated projections, or a TP=2 shard coinciding with a TP=1 shape) carry
  the upstream TP=1 config - the GEMM is identical either way. A missing shape or M, other GPUs and other TP
  sizes keep the standard linear path. MBX_SKINNY_GEMM_SM12X=0 turns it off.
- Measured on 2x GB10 (TP=2, image v6, hibrid48, K=5): 1-stream decode step 47.6 -> 46.1 ms (with five vLLM
  backports in the same arm), NLL unchanged; on a BF16-dense Qwen4Exp export the plans alone took dense GEMM time
  from 31.4 to 27.9 ms per step.
- Startup logs which local shapes took the skinny path ("Qwen4Exp low-latency GEMM: N modules on skinny plans ...").

--- a/models/qwen4_exp/nvidia/low_latency_gemm.py
+++ b/models/qwen4_exp/nvidia/low_latency_gemm.py
@@ -13,6 +13,7 @@
 import vllm.envs as envs
 from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
     SkinnyGemmConfig,
+    row_stride_ok,
     shape_dynamic_skinny_gemm,
 )
 from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
@@ -140,7 +140,186 @@
         2: SkinnyGemmConfig(2, 128, 1, k_unroll=10),
         4: SkinnyGemmConfig(4, 128, 1, k_unroll=10),
         8: SkinnyGemmConfig(8, 128, 1, k_unroll=10),
+    },
+}
+
+
+# DGX Spark (GB10, SM121) plans for TP=1 and TP=2 local shapes. TP=1 entries and the TP=2 entries for
+# (48, 2560), (2560, 3072), (6656, 2560), (8192, 2560) and (124160, 2560) are the upstream measurements
+# (vllm#59753 / vllm#59632); the remaining TP=2-only shapes ((2560, 320), (320, 10240)) and every M 6 / 12
+# row (K=5 verify) are from myllmbox's gb10-skinny-gemm table, timed 2026-10-01 on two GB10 pairs with
+# CUDA-graph replay and the weights rotated past L2 (each kept entry beat cuBLAS F.linear by >= 3 %;
+# cuBLAS picks SM80 WMMA 16x16 kernels for these on GB10). Shapes shared between TP=1 and TP=2 carry the
+# upstream TP=1 config - the GEMM is identical either way. A missing shape or M (e.g. 16 for K=320) keeps
+# the standard linear path. (8192, 2560) and (124160, 2560) are BF16 only in BF16-dense Qwen4Exp exports
+# (hibrid48 serves both as W4A16).
+QWEN4_EXP_SM121_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
+    # Shared-expert sigmoid gate (replicated; TP=1 and TP=2). vllm#59753, M 6/12 myllmbox.
+    (1, 2560): {
+        1: SkinnyGemmConfig(1, 64, 1, k_unroll=2, static_k=2560),
+        2: SkinnyGemmConfig(2, 64, 1, k_unroll=2, vector_width=4, static_k=2560),
+        4: SkinnyGemmConfig(4, 128, 1, vector_width=2, static_k=2560),
+        6: SkinnyGemmConfig(6, 128, 1, vector_width=4, static_k=2560),
+        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
+        12: SkinnyGemmConfig(12, 128, 1, vector_width=4, static_k=2560),
+        16: SkinnyGemmConfig(16, 128, 1, vector_width=4, static_k=2560),
+    },
+    # GDN fused B/A, TP=2. vllm#59632, M 6/12 myllmbox.
+    (48, 2560): {
+        1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
+        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4, static_k=2560),
+        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
+        6: SkinnyGemmConfig(6, 128, 1, k_unroll=4, vector_width=4),
+        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
+        12: SkinnyGemmConfig(12, 64, 1, k_unroll=4),
+        16: SkinnyGemmConfig(16, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
+    },
+    # GDN fused B/A, TP=1. vllm#59753.
+    (96, 2560): {
+        1: SkinnyGemmConfig(1, 128, 4, k_unroll=2, vector_width=2, static_k=2560),
+        2: SkinnyGemmConfig(2, 64, 4, vector_width=2, static_k=2560),
+        4: SkinnyGemmConfig(4, 128, 4, vector_width=2, static_k=2560),
+        8: SkinnyGemmConfig(8, 128, 4, k_unroll=2, vector_width=4, static_k=2560),
+        16: SkinnyGemmConfig(16, 64, 1),
+    },
+    # Final HC down (replicated). myllmbox TP=2 table; no upstream entry.
+    (320, 10240): {
+        1: SkinnyGemmConfig(1, 64, 1, static_k=10240),
+        2: SkinnyGemmConfig(2, 64, 1, static_k=10240),
+        4: SkinnyGemmConfig(4, 64, 1, static_k=10240),
+        6: SkinnyGemmConfig(6, 64, 1, static_k=10240),
+        8: SkinnyGemmConfig(8, 64, 1, static_k=10240),
+        12: SkinnyGemmConfig(12, 64, 2, static_k=10240),
+        16: SkinnyGemmConfig(16, 64, 2, static_k=10240),
+    },
+    # HC merged down/injection (replicated; TP=1 and TP=2). vllm#59753, M 6/12 myllmbox.
+    (336, 10240): {
+        1: SkinnyGemmConfig(1, 256, 1, vector_width=4, static_k=10240),
+        2: SkinnyGemmConfig(2, 256, 1, k_unroll=4, vector_width=4, static_k=10240),
+        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, static_k=10240),
+        6: SkinnyGemmConfig(6, 64, 1, static_k=10240),
+        8: SkinnyGemmConfig(8, 128, 4, k_unroll=4, static_k=10240),
+        12: SkinnyGemmConfig(12, 64, 2, static_k=10240),
+        16: SkinnyGemmConfig(16, 128, 4, k_unroll=4, static_k=10240),
+    },
+    # Router gate (replicated; TP=1 and TP=2). vllm#59753, M 6/12 myllmbox.
+    (512, 2560): {
+        1: SkinnyGemmConfig(1, 256, 1, k_unroll=4, vector_width=2, static_k=2560),
+        2: SkinnyGemmConfig(2, 256, 1, k_unroll=4, vector_width=2, static_k=2560),
+        4: SkinnyGemmConfig(4, 64, 1, k_unroll=2, static_k=2560),
+        6: SkinnyGemmConfig(6, 64, 1),
+        8: SkinnyGemmConfig(8, 64, 2, static_k=2560),
+        12: SkinnyGemmConfig(12, 64, 1),
+        16: SkinnyGemmConfig(16, 64, 2, k_unroll=2, static_k=2560),
+    },
+    # QSA indexer Q/K (replicated) and the TP=2 shared-expert gate/up shard. vllm#59753, M 6/12 myllmbox.
+    (640, 2560): {
+        1: SkinnyGemmConfig(1, 256, 1, vector_width=2, static_k=2560),
+        2: SkinnyGemmConfig(2, 256, 1, vector_width=2, static_k=2560),
+        4: SkinnyGemmConfig(4, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
+        6: SkinnyGemmConfig(6, 32, 1, static_k=2560),
+        8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2560),
+        12: SkinnyGemmConfig(12, 64, 1),
+        16: SkinnyGemmConfig(16, 64, 1, static_k=2560),
+    },
+    # Shared-expert fused gate/up (TP=1) and MTP fc_embedding / fc_hidden. vllm#59753, M 6/12 myllmbox.
+    (1280, 2560): {
+        1: SkinnyGemmConfig(1, 32, 1, k_unroll=2, static_k=2560),
+        2: SkinnyGemmConfig(2, 64, 1, vector_width=4, static_k=2560),
+        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, vector_width=2, static_k=2560),
+        6: SkinnyGemmConfig(6, 64, 1, k_unroll=4),
+        8: SkinnyGemmConfig(8, 128, 1, k_unroll=2, vector_width=2, static_k=2560),
+        12: SkinnyGemmConfig(12, 32, 1, static_k=2560),
+        16: SkinnyGemmConfig(16, 32, 1, k_unroll=2, static_k=2560),
+    },
+    # Shared-expert down, TP=2. myllmbox only.
+    (2560, 320): {
+        1: SkinnyGemmConfig(1, 32, 1, vector_width=2, static_k=320),
+        2: SkinnyGemmConfig(2, 32, 1, vector_width=2, static_k=320),
+        4: SkinnyGemmConfig(4, 32, 1, k_unroll=4, vector_width=2),
+        6: SkinnyGemmConfig(6, 32, 1, vector_width=2, static_k=320),
+        8: SkinnyGemmConfig(8, 32, 1, vector_width=2, static_k=320),
+    },
+    # Shared-expert down, TP=1. vllm#59753.
+    (2560, 640): {
+        1: SkinnyGemmConfig(1, 64, 1, k_unroll=4, vector_width=2, static_k=640),
+        2: SkinnyGemmConfig(2, 64, 1, vector_width=2, static_k=640),
+        4: SkinnyGemmConfig(4, 64, 1, vector_width=2, static_k=640),
+        8: SkinnyGemmConfig(8, 32, 1, k_unroll=4, vector_width=4, static_k=640),
+        16: SkinnyGemmConfig(16, 32, 1, k_unroll=2, vector_width=4, static_k=640),
+    },
+    # GDN and QSA output projections, TP=2. vllm#59632 (M 1-4), M 8/16 myllmbox.
+    (2560, 3072): {
+        1: SkinnyGemmConfig(1, 128, 2, k_unroll=2, vector_width=4, static_k=3072),
+        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2, static_k=3072),
+        4: SkinnyGemmConfig(4, 64, 2, k_unroll=2, static_k=3072),
+        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4),
+        16: SkinnyGemmConfig(16, 32, 1, static_k=3072),
+    },
+    # GDN and QSA output projections, TP=1. vllm#59753.
+    (2560, 6144): {
+        1: SkinnyGemmConfig(1, 64, 2, vector_width=4, static_k=6144),
+        2: SkinnyGemmConfig(2, 64, 1, k_unroll=4, vector_width=4, static_k=6144),
+        4: SkinnyGemmConfig(4, 64, 1, k_unroll=4, vector_width=4, static_k=6144),
+        8: SkinnyGemmConfig(8, 128, 1, vector_width=2, static_k=6144),
+        16: SkinnyGemmConfig(16, 32, 1, static_k=6144),
+    },
+    # QSA fused Q/gate/K/V, TP=2. vllm#59632 (M 1-4), M 8/16 myllmbox.
+    (6656, 2560): {
+        1: SkinnyGemmConfig(1, 128, 4, k_unroll=2, vector_width=4, static_k=2560),
+        2: SkinnyGemmConfig(2, 128, 4, k_unroll=2, vector_width=4, static_k=2560),
+        4: SkinnyGemmConfig(4, 64, 2, k_unroll=4, vector_width=4, static_k=2560),
+        8: SkinnyGemmConfig(8, 32, 1, static_k=2560),
+        16: SkinnyGemmConfig(16, 32, 1, static_k=2560),
     },
+    # GDN fused QKVZ, TP=2. vllm#59632 (M 1-4), M 8/16 myllmbox.
+    (8192, 2560): {
+        1: SkinnyGemmConfig(1, 128, 2, vector_width=4, static_k=2560),
+        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2, static_k=2560),
+        4: SkinnyGemmConfig(4, 64, 2, k_unroll=4, vector_width=4, static_k=2560),
+        8: SkinnyGemmConfig(8, 32, 1, static_k=2560),
+        16: SkinnyGemmConfig(16, 32, 1, static_k=2560),
+    },
+    # HC up (replicated; TP=1 and TP=2). vllm#59753 (M 16 keeps the cuBLAS fallback), M 6 myllmbox.
+    (10240, 320): {
+        1: SkinnyGemmConfig(1, 32, 1, k_unroll=2, vector_width=2, static_k=320),
+        2: SkinnyGemmConfig(2, 32, 1, vector_width=2, static_k=320),
+        4: SkinnyGemmConfig(4, 32, 1, k_unroll=2, vector_width=2, static_k=320),
+        6: SkinnyGemmConfig(6, 32, 1, vector_width=2, static_k=320),
+        8: SkinnyGemmConfig(8, 32, 1, vector_width=2, static_k=320),
+    },
+    # QSA fused QKV/gate, TP=1. vllm#59753.
+    (13312, 2560): {
+        1: SkinnyGemmConfig(1, 32, 1, vector_width=4, static_k=2560),
+        2: SkinnyGemmConfig(2, 32, 1, vector_width=4, static_k=2560),
+        4: SkinnyGemmConfig(4, 32, 1, vector_width=4, static_k=2560),
+        8: SkinnyGemmConfig(8, 64, 1, vector_width=2, static_k=2560),
+        16: SkinnyGemmConfig(16, 32, 1, vector_width=4, static_k=2560),
+    },
+    # GDN fused QKVZ, TP=1. vllm#59753.
+    (16384, 2560): {
+        1: SkinnyGemmConfig(1, 32, 1, k_unroll=4, vector_width=2, static_k=2560),
+        2: SkinnyGemmConfig(2, 32, 1, vector_width=2, static_k=2560),
+        4: SkinnyGemmConfig(4, 32, 1, k_unroll=2, vector_width=2, static_k=2560),
+        8: SkinnyGemmConfig(8, 64, 1, vector_width=2, static_k=2560),
+        16: SkinnyGemmConfig(16, 32, 1, k_unroll=2, vector_width=2, static_k=2560),
+    },
+    # LM head, TP=2. vllm#59632 (M 1-2), M 4/8/16 myllmbox.
+    (124160, 2560): {
+        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
+        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2),
+        4: SkinnyGemmConfig(4, 256, 1, vector_width=2, static_k=2560),
+        8: SkinnyGemmConfig(8, 256, 1, vector_width=2, static_k=2560),
+        16: SkinnyGemmConfig(16, 32, 1, static_k=2560),
+    },
+    # LM head, TP=1. vllm#59753 (this shape alone is -8.3 % ITL).
+    (248320, 2560): {
+        1: SkinnyGemmConfig(1, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
+        2: SkinnyGemmConfig(2, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
+        4: SkinnyGemmConfig(4, 32, 1, vector_width=4, static_k=2560),
+        8: SkinnyGemmConfig(8, 64, 1, k_unroll=2, vector_width=4, static_k=2560),
+        16: SkinnyGemmConfig(16, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
+    },
 }
 
 
@@ -152,11 +331,18 @@
     return current_platform.is_device_capability((9, 0))
 
 
+def _is_sm12x() -> bool:
+    return current_platform.is_device_capability_family(120)
+
+
 def _gemm_plans() -> dict[tuple[int, int], dict[int, SkinnyGemmConfig]]:
     if _is_sm103():
         return QWEN4_EXP_GEMM_PLANS
     if _is_sm90():
         return QWEN4_EXP_SM90_GEMM_PLANS
+    # GB10 (SM12x) skinny-GEMM plans; MBX_SKINNY_GEMM_SM12X=0 restores the standard linear path.
+    if _is_sm12x() and __import__("os").environ.get("MBX_SKINNY_GEMM_SM12X", "1") == "1":
+        return QWEN4_EXP_SM121_GEMM_PLANS
     return {}
 
 
@@ -167,10 +167,13 @@
     return tensor.dim() == 2 and tensor.stride() == (tensor.shape[1], 1)
 
 
-def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
+def _runtime_ok(
+    x: torch.Tensor, weight: torch.Tensor, config: SkinnyGemmConfig
+) -> bool:
     return (
         not envs.VLLM_BATCH_INVARIANT
-        and _is_packed_row_major(x)
+        and x.dim() == 2
+        and row_stride_ok(x, config)
         and _is_packed_row_major(weight)
         and x.dtype == torch.bfloat16
         and weight.dtype == torch.bfloat16
@@ -205,7 +208,7 @@
     config = None if plan is None else plan.get(x.shape[0])
     if (
         config is not None
-        and _runtime_ok(x, weight)
+        and _runtime_ok(x, weight, config)
         and shape_dynamic_skinny_gemm.is_available()
     ):
         return shape_dynamic_skinny_gemm(x, weight, config)
@@ -261,3 +447,13 @@
 
     if warmup_configs:
         shape_dynamic_skinny_gemm.request_warmup_configs(dtype, warmup_configs)
+    # report which local shapes took the skinny path (once per model: target and MTP)
+    from collections import Counter as _Counter
+    _hit = _Counter(
+        tuple(c.weight.shape) for c in module.modules()
+        if isinstance(getattr(c, "quant_method", None), _Qwen4ExpLowLatencyApply)
+        and getattr(c, "weight", None) is not None
+    )
+    from vllm.logger import init_logger as _il
+    _il(__name__).info("Qwen4Exp low-latency GEMM: %d modules on skinny plans %s (%d configs)",
+                       sum(_hit.values()), dict(_hit), len(warmup_configs))

--- a/model_executor/kernels/linear/cute_dsl/skinny_gemm.py
+++ b/model_executor/kernels/linear/cute_dsl/skinny_gemm.py
@@ -26,6 +26,20 @@
     static_k: int | None = None
 
 
+def row_stride_ok(a: torch.Tensor, config: SkinnyGemmConfig) -> bool:
+    """Whether the vectorized A loads stay aligned for a row-major ``a``.
+
+    The kernel reads A through its layout, so column slices of a wider buffer
+    are supported without a copy.
+    """
+    vector_bytes = config.vector_width * a.element_size()
+    return (
+        a.stride(1) == 1
+        and a.stride(0) % config.vector_width == 0
+        and a.data_ptr() % vector_bytes == 0
+    )
+
+
 class ShapeDynamicSkinnyGemm:
     def __init__(self) -> None:
         self._compiled: dict[tuple[torch.dtype, SkinnyGemmConfig, bool], Any] = {}
@@ -209,8 +223,8 @@
             raise ValueError("a and b must have the same BF16 or FP16 dtype")
         if not a.is_cuda or not b.is_cuda or a.device != b.device:
             raise ValueError("a and b must be CUDA tensors on the same device")
-        if not a.is_contiguous() or not b.is_contiguous():
-            raise ValueError("a and b must be contiguous")
+        if not b.is_contiguous():
+            raise ValueError("b must be contiguous")
         if a.shape[1] != b.shape[1]:
             raise ValueError("a and b must have matching K dimensions")
         if not 1 <= a.shape[0] <= 16:
@@ -236,6 +250,8 @@
             )
         if config.static_k is not None and a.shape[1] != config.static_k:
             raise ValueError("input K must match config static_k")
+        if not row_stride_ok(a, config):
+            raise ValueError("a must be row-major with vector_width-aligned rows")
         has_residual = residual is not None
         cache_key = (a.dtype, config, has_residual)
         if cache_key not in self._compiled:
'''


def parse_diff(text: str) -> dict[str, list[tuple[list[str], list[str]]]]:
    files: dict[str, list[tuple[list[str], list[str]]]] = {}
    hunks: list[tuple[list[str], list[str]]] = []
    old_left = new_left = 0
    for line in text.splitlines(keepends=True):
        if old_left or new_left:
            if line[0] != "+":
                hunks[-1][0].append(line[1:])
                old_left -= 1
            if line[0] != "-":
                hunks[-1][1].append(line[1:])
                new_left -= 1
        elif line.startswith("+++ b/"):
            hunks = files.setdefault(line[len("+++ b/"):].rstrip("\n"), [])
        elif line.startswith("@@"):
            old_left, new_left = (int(n or 1) for n in HUNK_HEADER.match(line).groups())
            hunks.append(([], []))
    if old_left or new_left:
        raise SystemExit("vendored gb10-skinny-gemm diff is truncated")
    return files


def find(lines: list[str], block: list[str], start: int) -> int:
    for i in range(start, len(lines) - len(block) + 1):
        if lines[i : i + len(block)] == block:
            return i
    return -1


def apply(lines: list[str], hunks: list[tuple[list[str], list[str]]]) -> list[str] | None:
    out: list[str] = []
    cursor = 0
    for old, new in hunks:
        at = find(lines, old, cursor)
        if at < 0:
            return None
        out += lines[cursor:at] + new
        cursor = at + len(old)
    return out + lines[cursor:]


def already_applied(lines: list[str], hunks: list[tuple[list[str], list[str]]]) -> bool:
    cursor = 0
    for _, new in hunks:
        at = find(lines, new, cursor)
        if at < 0:
            return False
        cursor = at + len(new)
    return True


def main(argv: list[str]) -> int:
    if argv[:1] == ["--list"]:
        print("\n".join(TARGETS))
        return 0
    orig_dir = argv[0] if argv else HERE
    out_dir = argv[1] if len(argv) > 1 else HERE
    diff = parse_diff(UPSTREAM_DIFF)
    sources = {}
    for rel, local in TARGETS.items():
        with open(os.path.join(orig_dir, local + ".orig"), encoding="utf-8") as f:
            sources[rel] = f.read().splitlines(keepends=True)
    if all(already_applied(sources[rel], diff[rel]) for rel in TARGETS):
        print("gb10-skinny-gemm already present in the original, copying it unchanged")
        results = sources
    else:
        results = {}
        for rel in TARGETS:
            patched = apply(sources[rel], diff[rel])
            if patched is None:
                print(f"gb10-skinny-gemm does not apply to {rel}, nothing written",
                      file=sys.stderr)
                return 1
            results[rel] = patched
    texts = {rel: "".join(lines) for rel, lines in results.items()}
    for rel, text in texts.items():
        ast.parse(text, filename=rel)
    for rel, local in TARGETS.items():
        path = os.path.join(out_dir, local)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            f.write(texts[rel])
        os.replace(path + ".tmp", path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
