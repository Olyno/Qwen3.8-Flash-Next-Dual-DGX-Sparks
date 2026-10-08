#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""QSA head_dim unsharded fallback fix for the vLLM 0.30.0 lane
(unconditional).

Backports vllm-project/vllm#59945 onto the stock v0.30.0 file:

Upstream: https://github.com/vllm-project/vllm/pull/59945 (lesj0610, merged
upstream 2026-10-07), Apache-2.0.

    vllm/models/qwen4_exp/nvidia/qsa.py
    - The fallback head_dim divided hidden_size by self.num_heads, which is
      already TP-sharded, so at TP > 1 the fallback came out tp_size times
      too large (q_size/kv_size then mis-size the QKV projection and the
      checkpoint fails to load with a shape mismatch). Now divides by
      self.total_num_heads. Latent on every real checkpoint (head_dim is
      always set), but the arithmetic is now right on any TP size.

The PR's identical amd/qsa.py hunk is dropped: this lane only ever loads
the nvidia model package, so an amd overlay would be dead weight.
Correctness fix, so no toggle. Composes with the QSA-prepare overlay
(patches/qsa_prepare/, vllm#57097) and the FP8-KV overlay
(patches/patch_qsa_fp8_kv_v030.py, vllm#55557): neither touches the
__init__ head-count block, so engine/patches.sh feeds the qsa_prepare
output as this patcher's input and mounts the result.

Inputs:  patches/qsa_head_dim/orig/qsa.py (the qsa_prepare overlay's
         output; the stock nvidia/qsa.py works too — the anchor region is
         identical)
Outputs: patches/qsa_head_dim/qsa_v030.py
argv[0]/argv[1] override the orig/output directories (used by the test).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

HEAD_DIM_OLD = """\
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
"""
HEAD_DIM_NEW = """\
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.total_num_heads)
"""
HUNKS = ((HEAD_DIM_OLD, HEAD_DIM_NEW),)


def main(argv) -> int:
    orig_dir = argv[0] if argv else os.path.join(HERE, "orig")
    out_dir = argv[1] if len(argv) > 1 else HERE
    orig_path = os.path.join(orig_dir, "qsa.py")
    if not os.path.isfile(orig_path):
        sys.exit(f"ERROR: missing {orig_path} "
                 "(start.sh extracts it from the image)")
    src = open(orig_path).read()
    if "self.hidden_size // self.total_num_heads" in src:
        sys.exit("ERROR: qsa_head_dim orig qsa.py is already patched")
    for i, (old, new) in enumerate(HUNKS):
        count = src.count(old)
        if count != 1:
            sys.exit(f"qsa_head_dim: anchor {i} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"qsa_head_dim: patched qsa_v030.py does not parse: {exc}")
    open(os.path.join(out_dir, "qsa_v030.py"), "w").write(src)
    print("patched qsa_v030.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
