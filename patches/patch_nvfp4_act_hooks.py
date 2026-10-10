#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""NVFP4 per-channel activation hooks for the vLLM 0.30.0 lane (opt-in debug
tooling, nvfp4_act_hooks: true).

Answers the question scripts/check_nvfp4_activations.py cannot answer from
the checkpoint alone: do the calibration-time input_scale outliers (e.g. the
last decoder layer's expert down_proj at ~10x the family median) correspond
to real massive activation channels at runtime? The production NVFP4 MoE
backends (flashinfer cutlass/trtllm) run a single fused kernel and never
materialize the post-SwiGLU intermediate, so this capture rides the
moe_backend: emulation path (Nvfp4QuantizationEmulationTritonExperts), which
holds the down_proj input in intermediate_cache2 between its two GEMMs.

    vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py
    - adds a fail-closed capture block: with VLLM_NVFP4_ACT_HOOKS=1
      (engine/patches.sh passes it via OVERLAY_ENV when it mounts the file)
      each emulated experts instance registers its construction index (==
      decoder layer index; only the NVFP4 target experts use this class, the
      FP8 draft experts resolve through the fp8 oracle) and, once the
      /tmp/nvfp4_act_hooks.arm sentinel exists (create it after boot so
      CUDA-graph capture dummy runs stay out of the sample), accumulates
      per-channel absmax of the MoE input (gate_up_proj input) and the
      pre-quantize down_proj input. After VLLM_NVFP4_ACT_HOOKS_STEPS eager
      calls per layer (default 200) it writes {tensor: [channel absmax]} to
      VLLM_NVFP4_ACT_HOOKS_OUT (default /tmp/nvfp4_act_hooks.json) and
      disarms. Feed the dump to check_nvfp4_activations.py --channel-stats.
      Decode steps captured into FULL cudagraphs never call apply(), so the
      sample is prefill/chunked-prefill traffic only.
      Without the env var the file is bit-exact stock behavior.

Inputs:  patches/v030_nvfp4hooks/orig/nvfp4_emulation_moe.py (from the image)
Outputs: patches/v030_nvfp4hooks/nvfp4_emulation_moe_v030.py
argv[0]/argv[1] override the orig/output directories (used by the test).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

IMPORT_OLD = """\
from typing import Any

import torch
"""
IMPORT_NEW = """\
import json
import os
from typing import Any

import torch
"""
CLASS_OLD = """\
class Nvfp4QuantizationEmulationTritonExperts(TritonExperts):
"""
CLASS_NEW = '''\
# --- activation channel-stat hooks (debug, fail-closed) -------------------
# Armed by the /tmp/nvfp4_act_hooks.arm sentinel so boot-time CUDA-graph
# capture dummy runs stay out of the sample. Registry index == decoder layer
# index: the MoE blocks are constructed in layer order and only the NVFP4
# target experts use this class (the FP8 draft experts use the fp8 oracle).
_ACT_HOOK_ARM = "/tmp/nvfp4_act_hooks.arm"
_ACT_HOOK_LAYERS: list = []
_ACT_HOOK_ACC: dict = {}


def _act_hook_armed() -> bool:
    return os.environ.get("VLLM_NVFP4_ACT_HOOKS", "0") == "1" \\
        and os.path.exists(_ACT_HOOK_ARM)


def _act_hook_capture(layer_idx: int, hidden_states, down_proj_input) -> None:
    acc = _ACT_HOOK_ACC
    with torch.no_grad():
        for suffix, tensor in (("gate_up_proj.input", hidden_states),
                               ("down_proj.input", down_proj_input)):
            name = f"model.layers.{layer_idx}.mlp.experts.{suffix}"
            flat = tensor.detach().reshape(-1, tensor.shape[-1]).float()
            flat = flat.abs().amax(dim=0)
            prev = acc.get(name)
            acc[name] = (flat if prev is None else torch.maximum(prev, flat))


def _act_hook_flush() -> None:
    out = os.environ.get("VLLM_NVFP4_ACT_HOOKS_OUT", "/tmp/nvfp4_act_hooks.json")
    payload = {name: [float(v) for v in vec.cpu()]
               for name, vec in _ACT_HOOK_ACC.items()}
    with open(out + ".tmp", "w") as fh:
        json.dump(payload, fh)
    os.replace(out + ".tmp", out)
    try:
        os.remove(_ACT_HOOK_ARM)
    except OSError:
        pass
    logger.info("nvfp4 act hooks: wrote %d tensors to %s", len(payload), out)


class Nvfp4QuantizationEmulationTritonExperts(TritonExperts):
'''
INIT_OLD = """\
        self.quantization_emulation = True
"""
INIT_NEW = """\
        self.quantization_emulation = True

        self._act_hook_idx = len(_ACT_HOOK_LAYERS)
        _ACT_HOOK_LAYERS.append(self)
        self._act_hook_calls = 0
"""
ACT_OLD = """\
        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )
"""
ACT_NEW = """\
        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        if _act_hook_armed():
            # intermediate_cache2 is the pre-quantize down_proj input,
            # (tokens*top_k, intermediate); hidden_states the gate_up input.
            _act_hook_capture(self._act_hook_idx, hidden_states,
                              intermediate_cache2)
            self._act_hook_calls += 1
            if self._act_hook_calls >= int(
                    os.environ.get("VLLM_NVFP4_ACT_HOOKS_STEPS", "200")):
                _act_hook_flush()
"""
HUNKS = ((IMPORT_OLD, IMPORT_NEW), (CLASS_OLD, CLASS_NEW),
         (INIT_OLD, INIT_NEW), (ACT_OLD, ACT_NEW))


def _apply(src, hunks, name):
    for i, (old, new) in enumerate(hunks):
        count = src.count(old)
        if count != 1:
            sys.exit(f"nvfp4_act_hooks: anchor {i} in {name} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    return src


def main(argv) -> int:
    orig_dir = argv[0] if argv else os.path.join(HERE, "v030_nvfp4hooks", "orig")
    out_dir = argv[1] if len(argv) > 1 else os.path.join(HERE, "v030_nvfp4hooks")
    orig = os.path.join(orig_dir, "nvfp4_emulation_moe.py")
    if not os.path.isfile(orig):
        sys.exit(f"ERROR: missing {orig} (start.sh extracts it from the image)")
    src = open(orig).read()
    if "_ACT_HOOK_ARM" in src:
        sys.exit("ERROR: v030_nvfp4hooks orig is already patched")
    src = _apply(src, HUNKS, "nvfp4_emulation_moe.py")
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"nvfp4_act_hooks: patched nvfp4_emulation_moe_v030.py does not parse: {exc}")
    out = os.path.join(out_dir, "nvfp4_emulation_moe_v030.py")
    os.makedirs(out_dir, exist_ok=True)
    with open(out + ".tmp", "w") as f:
        f.write(src)
    os.replace(out + ".tmp", out)
    print(f"patched {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
