#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""GB10 (SM12x) plan table for the Qwen4Exp skinny decode GEMM, vendored from
myllmbox's patches/gb10-skinny-gemm.patch.

vLLM 0.30's models/qwen4_exp/nvidia/low_latency_gemm.py (already shipped and
already wired into model.py and mtp.py by the image) carries plans only for
SM103 and SM90 at TP=4, so on the GB10 every decode-sized BF16 projection
falls back to cuBLAS (SM80 WMMA 16x16 kernels). This adds a TP=2 SM12x plan
table keyed by local (N, K) shape and exact token count M; a missing shape or
M keeps the standard linear path, so other GPUs and TP sizes are unchanged
(at TP=1 only the replicated projections match the table).
MBX_SKINNY_GEMM_SM12X=0 in the container restores the standard linear path.

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
}
HUNK_HEADER = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")

UPSTREAM_DIFF = r'''
GB10 (SM12x) plans for the Qwen4Exp skinny decode GEMM (contributed by @sethforprivacy).
- vLLM 0.30's qwen4_exp_low_latency_gemm (a CuTe-DSL skinny GEMM) carries plans only for SM103 and SM90 TP=4 shapes,
  so on the GB10 every decode-sized BF16 projection stays on cuBLAS, which picks SM80 WMMA 16x16 kernels (one warp
  per CTA, 125-225 GB/s). This adds a TP=2 plan table for SM12x: 240-255 GB/s, 1.4-1.5x at M=1 (the draft passes)
  and up to 7x on the latency-bound shapes (timed in CUDA graphs with the weights rotated past L2).
- On hibrid48 it covers the BF16 layers: QSA Q/gate/K/V and output, shared expert gate/up/down and its sigmoid gate,
  router, GDN B/A, the hyper-connection mixes and the MTP fc projections, at M 1-16 plus the K=5 verify rows 6 / 12.
  An M without a plan, other GPUs and other TP sizes keep the standard linear path. MBX_SKINNY_GEMM_SM12X=0 turns it off.
- Measured on 2x GB10 (TP=2, image v6, hibrid48, K=5): 1-stream decode step 47.6 -> 46.1 ms (with five vLLM
  backports in the same arm), NLL unchanged; on a BF16-dense Qwen4Exp export the plans alone took dense GEMM time
  from 31.4 to 27.9 ms per step.
- Startup logs which local shapes took the skinny path ("Qwen4Exp low-latency GEMM: N modules on skinny plans ...").

--- a/models/qwen4_exp/nvidia/low_latency_gemm.py
+++ b/models/qwen4_exp/nvidia/low_latency_gemm.py
@@ -140,7 +140,132 @@
         2: SkinnyGemmConfig(2, 128, 1, k_unroll=10),
         4: SkinnyGemmConfig(4, 128, 1, k_unroll=10),
         8: SkinnyGemmConfig(8, 128, 1, k_unroll=10),
+    },
+}
+
+
+# GB10 (SM121) plans at TP=2, timed 2026-10-01 on two GB10 pairs (M 1-16; M 6/12 = K=5 verify rows) with
+# CUDA-graph replay and the weights rotated past L2: each entry beat cuBLAS F.linear by >= 3 % (cuBLAS picks SM80
+# WMMA 16x16 kernels for these on GB10). A missing M (e.g. 16 for K=320) keeps the standard linear path.
+# (8192, 2560) and (124160, 2560) are BF16 only in BF16-dense Qwen4Exp exports (hibrid48 serves both as W4A16).
+QWEN4_EXP_SM121_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
+    # GDN fused QKVZ, TP=2.
+    (8192, 2560): {
+        1: SkinnyGemmConfig(1, 256, 1, k_unroll=4, vector_width=2),
+        2: SkinnyGemmConfig(2, 256, 1, k_unroll=2, vector_width=2),
+        4: SkinnyGemmConfig(4, 32, 1, static_k=2560),
+        8: SkinnyGemmConfig(8, 32, 1, static_k=2560),
+        16: SkinnyGemmConfig(16, 32, 1, static_k=2560),
+    },
+    # GDN and QSA output projections, TP=2.
+    (2560, 3072): {
+        1: SkinnyGemmConfig(1, 256, 1, vector_width=4),
+        2: SkinnyGemmConfig(2, 256, 1, k_unroll=4, vector_width=4),
+        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4),
+        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4),
+        16: SkinnyGemmConfig(16, 32, 1, static_k=3072),
+    },
+    # HC up (replicated), TP=2.
+    (10240, 320): {
+        1: SkinnyGemmConfig(1, 32, 1, vector_width=2, static_k=320),
+        2: SkinnyGemmConfig(2, 32, 1, vector_width=2, static_k=320),
+        4: SkinnyGemmConfig(4, 32, 1, vector_width=2, static_k=320),
+        6: SkinnyGemmConfig(6, 32, 1, vector_width=2, static_k=320),
+        8: SkinnyGemmConfig(8, 32, 1, vector_width=2, static_k=320),
+    },
+    # router gate (replicated), TP=2.
+    (512, 2560): {
+        1: SkinnyGemmConfig(1, 128, 1, vector_width=4),
+        2: SkinnyGemmConfig(2, 128, 1, vector_width=4),
+        4: SkinnyGemmConfig(4, 64, 1),
+        6: SkinnyGemmConfig(6, 64, 1),
+        8: SkinnyGemmConfig(8, 64, 1),
+        12: SkinnyGemmConfig(12, 64, 1),
+        16: SkinnyGemmConfig(16, 64, 2),
+    },
+    # GDN fused B/A, TP=2.
+    (48, 2560): {
+        1: SkinnyGemmConfig(1, 256, 1, k_unroll=4, vector_width=2),
+        2: SkinnyGemmConfig(2, 256, 1, k_unroll=4, vector_width=2),
+        4: SkinnyGemmConfig(4, 256, 1, k_unroll=4, vector_width=2),
+        6: SkinnyGemmConfig(6, 128, 1, k_unroll=4, vector_width=4),
+        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4),
+        12: SkinnyGemmConfig(12, 64, 1, k_unroll=4),
+        16: SkinnyGemmConfig(16, 128, 1, k_unroll=4, vector_width=4),
+    },
+    # LM head, TP=2.
+    (124160, 2560): {
+        1: SkinnyGemmConfig(1, 256, 1, vector_width=2, static_k=2560),
+        2: SkinnyGemmConfig(2, 256, 1, vector_width=2, static_k=2560),
+        4: SkinnyGemmConfig(4, 256, 1, vector_width=2, static_k=2560),
+        8: SkinnyGemmConfig(8, 256, 1, vector_width=2, static_k=2560),
+        16: SkinnyGemmConfig(16, 32, 1, static_k=2560),
+    },
+    # MTP fc_embedding / fc_hidden, TP=2.
+    (1280, 2560): {
+        1: SkinnyGemmConfig(1, 32, 1, static_k=2560),
+        2: SkinnyGemmConfig(2, 256, 1, k_unroll=4, vector_width=2),
+        4: SkinnyGemmConfig(4, 32, 1, static_k=2560),
+        6: SkinnyGemmConfig(6, 64, 1, k_unroll=4),
+        8: SkinnyGemmConfig(8, 32, 1, static_k=2560),
+        12: SkinnyGemmConfig(12, 32, 1, static_k=2560),
+        16: SkinnyGemmConfig(16, 32, 1, static_k=2560),
+    },
+    # QSA fused Q/gate/K/V, TP=2.
+    (6656, 2560): {
+        1: SkinnyGemmConfig(1, 256, 1, k_unroll=4, vector_width=2),
+        2: SkinnyGemmConfig(2, 256, 1, k_unroll=4, vector_width=2),
+        4: SkinnyGemmConfig(4, 256, 1, k_unroll=4, vector_width=2),
+        8: SkinnyGemmConfig(8, 32, 1, static_k=2560),
+        16: SkinnyGemmConfig(16, 32, 1, static_k=2560),
     },
+    # HC merged down/injection (replicated), TP=2.
+    (336, 10240): {
+        1: SkinnyGemmConfig(1, 64, 2, static_k=10240),
+        2: SkinnyGemmConfig(2, 64, 1, static_k=10240),
+        4: SkinnyGemmConfig(4, 64, 2, static_k=10240),
+        6: SkinnyGemmConfig(6, 64, 1, static_k=10240),
+        8: SkinnyGemmConfig(8, 64, 1, static_k=10240),
+        12: SkinnyGemmConfig(12, 64, 2, static_k=10240),
+        16: SkinnyGemmConfig(16, 64, 2, static_k=10240),
+    },
+    # shared-expert gate/up and QSA indexer Q/K, TP=2.
+    (640, 2560): {
+        1: SkinnyGemmConfig(1, 64, 1, k_unroll=2),
+        2: SkinnyGemmConfig(2, 64, 1),
+        4: SkinnyGemmConfig(4, 64, 1, k_unroll=2),
+        6: SkinnyGemmConfig(6, 32, 1, static_k=2560),
+        8: SkinnyGemmConfig(8, 64, 1),
+        12: SkinnyGemmConfig(12, 64, 1),
+        16: SkinnyGemmConfig(16, 64, 1),
+    },
+    # shared-expert down, TP=2.
+    (2560, 320): {
+        1: SkinnyGemmConfig(1, 32, 1, vector_width=2, static_k=320),
+        2: SkinnyGemmConfig(2, 32, 1, vector_width=2, static_k=320),
+        4: SkinnyGemmConfig(4, 32, 1, k_unroll=4, vector_width=2),
+        6: SkinnyGemmConfig(6, 32, 1, vector_width=2, static_k=320),
+        8: SkinnyGemmConfig(8, 32, 1, vector_width=2, static_k=320),
+    },
+    # shared-expert sigmoid gate, TP=2.
+    (1, 2560): {
+        1: SkinnyGemmConfig(1, 128, 1, vector_width=4, static_k=2560),
+        2: SkinnyGemmConfig(2, 128, 1, vector_width=4, static_k=2560),
+        4: SkinnyGemmConfig(4, 128, 1, vector_width=4, static_k=2560),
+        6: SkinnyGemmConfig(6, 128, 1, vector_width=4, static_k=2560),
+        8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2560),
+        12: SkinnyGemmConfig(12, 128, 1, vector_width=4, static_k=2560),
+    },
+    # final HC down (replicated), TP=2.
+    (320, 10240): {
+        1: SkinnyGemmConfig(1, 64, 1, static_k=10240),
+        2: SkinnyGemmConfig(2, 64, 1, static_k=10240),
+        4: SkinnyGemmConfig(4, 64, 1, static_k=10240),
+        6: SkinnyGemmConfig(6, 64, 1, static_k=10240),
+        8: SkinnyGemmConfig(8, 64, 1, static_k=10240),
+        12: SkinnyGemmConfig(12, 64, 2, static_k=10240),
+        16: SkinnyGemmConfig(16, 64, 2, static_k=10240),
+    },
 }
 
 
@@ -152,11 +277,18 @@
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
 
 
@@ -261,3 +393,13 @@
 
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
        elif line.startswith("+++ b/models/"):
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
