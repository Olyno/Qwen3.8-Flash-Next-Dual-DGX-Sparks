#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""QSA pre-indexer RoPE position clamp for the vLLM 0.30.0 lane (opt-in,
QSA_ROPE_CLAMP=true).

Ports myllmbox/vllm@9ff17c016e68 ("Qwen4Exp: clamp QSA pre-indexer RoPE
positions (CUDA-graph warmup IMA on SM121)") onto the stock v0.30.0 file:

Upstream: https://github.com/myllmbox/qwen38-flash-next-recipe (MIT for kit
scripts and image patches — see licenses/myllmbox-MIT.LICENSE); the
myllmbox/vllm fork commit is an Apache-2.0 vLLM derivative.

    vllm/models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py
    - _norm_rope loads cos_sin[pos] with no bounds check; CUDA-graph capture
      feeds dummy positions beyond the cos/sin table, which is an illegal
      memory access on SM121 (GB10). The overlay clamps pos_t/pos_h/pos_w
      into [0, cos_sin_rows) inside _norm_rope so both the Q-tile and the
      K-pool call sites are covered.

Change vs the vendor commit: the vendor clamps unconditionally; here the
clamp is behind a CLAMP_POS constexpr read from VLLM_QSA_ROPE_CLAMP (default
OFF, fail closed — the constexpr-false branch compiles to the bit-exact
stock kernel). engine/patches.sh passes -e VLLM_QSA_ROPE_CLAMP=1 via
OVERLAY_ENV when it mounts the file.

Inputs:  patches/v030_qsa_rope/orig/qsa_pre_indexer.py (from the image)
Outputs: patches/v030_qsa_rope/qsa_pre_indexer_v030.py
argv[0]/argv[1] override the orig/output directories (used by the test).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

IMPORT_OLD = """\
import torch

from vllm.triton_utils import tl, triton
"""
IMPORT_NEW = """\
import os

import torch

from vllm.triton_utils import tl, triton
"""
NORM_ROPE_OLD = """\
    IS_MROPE: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
):
    \"\"\"Apply Gemma RMSNorm and selected-axis NeoX RoPE to register rows.\"\"\"
    TILE_T: tl.constexpr = x.shape[0]
"""
NORM_ROPE_NEW = """\
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
KERNEL_SIG_OLD = """\
    CACHE_HAS_ROPE_POS: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
):
    pid = tl.program_id(0)
"""
KERNEL_SIG_NEW = """\
    CACHE_HAS_ROPE_POS: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
    CLAMP_POS: tl.constexpr,
    MAX_POS: tl.constexpr,
):
    pid = tl.program_id(0)
"""
Q_CALL_OLD = """\
            IS_2D_POSITIONS,
            MROPE_H,
            MROPE_W,
        )
"""
Q_CALL_NEW = """\
            IS_2D_POSITIONS,
            MROPE_H,
            MROPE_W,
            CLAMP_POS,
            MAX_POS,
        )
"""
K_CALL_OLD = """\
                IS_K_MROPE,
                MROPE_H,
                MROPE_W,
            )
"""
K_CALL_NEW = """\
                IS_K_MROPE,
                MROPE_H,
                MROPE_W,
                CLAMP_POS,
                MAX_POS,
            )
"""
LAUNCH_OLD = """\
        CACHE_HAS_ROPE_POS=cache_has_rope_pos,
        MROPE_H=section[1],
        MROPE_W=section[2],
        num_warps=1,
    )
"""
LAUNCH_NEW = """\
        CACHE_HAS_ROPE_POS=cache_has_rope_pos,
        MROPE_H=section[1],
        MROPE_W=section[2],
        CLAMP_POS=os.environ.get("VLLM_QSA_ROPE_CLAMP", "0") == "1",
        MAX_POS=max(int(cos_sin_cache.shape[0]) - 1, 0),
        num_warps=1,
    )
"""
HUNKS = ((IMPORT_OLD, IMPORT_NEW), (NORM_ROPE_OLD, NORM_ROPE_NEW),
         (KERNEL_SIG_OLD, KERNEL_SIG_NEW), (Q_CALL_OLD, Q_CALL_NEW),
         (K_CALL_OLD, K_CALL_NEW), (LAUNCH_OLD, LAUNCH_NEW))


def _apply(src, hunks, name):
    for i, (old, new) in enumerate(hunks):
        count = src.count(old)
        if count != 1:
            sys.exit(f"qsa_rope_clamp_v030: anchor {i} in {name} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    return src


def main(argv) -> int:
    orig_dir = argv[0] if argv else os.path.join(HERE, "v030_qsa_rope", "orig")
    out_dir = argv[1] if len(argv) > 1 else os.path.join(HERE, "v030_qsa_rope")
    orig = os.path.join(orig_dir, "qsa_pre_indexer.py")
    if not os.path.isfile(orig):
        sys.exit(f"ERROR: missing {orig} (start.sh extracts it from the image)")
    src = open(orig).read()
    if "CLAMP_POS" in src:
        sys.exit("ERROR: v030_qsa_rope orig is already patched")
    src = _apply(src, HUNKS, "qsa_pre_indexer.py")
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"qsa_rope_clamp_v030: patched qsa_pre_indexer_v030.py does not parse: {exc}")
    out = os.path.join(out_dir, "qsa_pre_indexer_v030.py")
    os.makedirs(out_dir, exist_ok=True)
    with open(out + ".tmp", "w") as f:
        f.write(src)
    os.replace(out + ".tmp", out)
    print(f"patched {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
