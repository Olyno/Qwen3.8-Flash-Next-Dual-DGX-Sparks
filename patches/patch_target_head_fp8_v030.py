#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Patch the v030-lane qwen4_exp/nvidia/model.py: opt-in FP8 target lm_head.

VLLM_TARGET_HEAD_FP8=1 (or marlin | w8a8) makes Qwen4ExpForCausalLM replace
its lm_head with an E4M3 copy (per-row symmetric scale) once the weights are
loaded: the full 248,320-row vocabulary, each rank keeping its own TP shard,
no slicing. The BF16 rows are dropped and LogitsProcessor keeps its stock
path -- it calls lm_head.quant_method.apply(), which becomes the FP8 GEMM,
then all-gathers and trims the vocabulary exactly as before. With the env
unset the patched file is inert.

E4M3 with a per-row scale is lossy in the last logits bit, and this head feeds
sampling and target verification, so -- unlike the draft-head variant -- it
can change emitted tokens. Gate it on the KL yardstick, then decodebench;
until that run passes, no speedup is claimed.

The quantization and kernels are the block the MTP draft-head patch injects
(patches/patch_mtp_draft_vocab_v030.py, FP8_HEAD_KERNELS, imported here); this
file only adds the target-side attach and a quant-method shim.

Inputs:  <dir>/orig/model.py   (nvidia/model.py from the pinned image)
Outputs: <dir>/model_v030.py
"""
import argparse
import ast
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, "v030_target_head")
ORIG_NAME = "model.py"
OUT_NAME = "model_v030.py"
MARKER = "_attach_target_head_fp8"


def _shared_kernels() -> str:
    spec = importlib.util.spec_from_file_location(
        "patch_mtp_draft_vocab_v030",
        os.path.join(HERE, "patch_mtp_draft_vocab_v030.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FP8_HEAD_KERNELS


FP8_HEAD_KERNELS = _shared_kernels()


TARGET_BLOCK = '''

# Opt-in FP8 copy of the target lm_head (VLLM_TARGET_HEAD_FP8). The kernels
# and quantization above are shared with the MTP draft-head patch; this part
# is the target-side attach.
_TARGET_HEAD_FP8_ENV = "VLLM_TARGET_HEAD_FP8"


class _Fp8HeadQuantMethod:
    """lm_head.quant_method stand-in: LogitsProcessor calls apply().

    The stock LogitsProcessor all-gathers this rank's rows across TP and
    trims the vocabulary afterwards, so the projection only covers the local
    shard -- exactly like the BF16 head it replaces.
    """

    def __init__(self, head) -> None:
        self.head = head

    def apply(self, layer, x, bias=None):
        out = self.head.logits(x)
        return out if bias is None else out + bias


def _attach_target_head_fp8(model: nn.Module) -> None:
    """Quantize the target lm_head in place (VLLM_TARGET_HEAD_FP8).

    Full vocabulary, no slicing: each rank quantizes its own TP shard of
    rows. Fail closed -- no kernel, LoRA, pipeline or batch-invariant mode,
    or a head_dtype override leaves the BF16 head in place, agreed across TP
    ranks so every rank serves the same path.
    """
    try:
        kernels = _fp8_head_kernels(
            os.environ.get(_TARGET_HEAD_FP8_ENV, ""), _TARGET_HEAD_FP8_ENV
        )
    except ValueError as exc:
        logger.error("target FP8 lm_head: %s; serving the BF16 head.", exc)
        return
    if not kernels:
        return
    if getattr(model, "_target_head_fp8", None) is not None:
        return  # load_weights ran twice; keep the head already built
    if getattr(getattr(model, "vllm_config", None), "lora_config", None) is not None:
        logger.warning("target FP8 lm_head: LoRA unsupported; disabled.")
        return
    if get_pp_group().world_size != 1:
        logger.warning("target FP8 lm_head: pipeline parallel unsupported; disabled.")
        return
    if getattr(envs, "VLLM_BATCH_INVARIANT", False):
        logger.warning(
            "target FP8 lm_head: batch-invariant decoding unsupported; disabled."
        )
        return
    rows = getattr(getattr(model, "lm_head", None), "weight", None)
    if rows is None:
        logger.warning("target FP8 lm_head: no lm_head weight; disabled.")
        return
    head_dtype = getattr(getattr(model, "logits_processor", None), "head_dtype", None)
    if head_dtype is not None and head_dtype != rows.dtype:
        logger.warning(
            "target FP8 lm_head: head_dtype=%s unsupported; disabled.", head_dtype
        )
        return
    alloc0 = torch.cuda.memory_allocated()
    resv0 = torch.cuda.memory_reserved()
    head, err = None, ""
    try:
        head, err = _build_fp8_head(rows, kernels)
    except Exception as exc:  # noqa: BLE001 - fail closed to BF16
        head, err = None, f"{type(exc).__name__}: {exc}"
    torch.cuda.empty_cache()
    logger.info(
        "target FP8 lm_head attach: cuda %.1f -> %.1f MiB allocated, "
        "%.1f -> %.1f MiB reserved.",
        alloc0 / 2**20, torch.cuda.memory_allocated() / 2**20,
        resv0 / 2**20, torch.cuda.memory_reserved() / 2**20,
    )
    ok, err = _tp_agree(head is not None, err)
    if not ok:
        logger.warning(
            "target FP8 lm_head disabled (%s); serving the BF16 head.",
            err or "not requested on every rank",
        )
        return
    bf16_mib = head.n * head.k * rows.element_size() / 2**20
    logger.info(
        "target FP8 lm_head: %d rows on this rank x %d, %s, %.1f MiB; "
        "%.1f MiB BF16 dropped (full vocabulary, no slice).",
        head.n, head.k, head.kernel, head.nbytes / 2**20, bf16_mib,
    )
    model._target_head_fp8 = head
    model.lm_head.quant_method = _Fp8HeadQuantMethod(head)
    model.lm_head.weight.data = torch.empty(0, dtype=head.dtype, device=rows.device)

'''

LOAD_WEIGHTS = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={"mtp.": None}
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=mapper)
'''


def edits(fp8_block: str) -> list[tuple[str, str]]:
    return [
        (
            "import torch\nfrom torch import nn\n",
            "import os\n\nimport torch\nfrom torch import nn\n",
        ),
        (
            "from vllm.distributed import get_pp_group\n",
            "from vllm.distributed import (\n"
            "    get_pp_group,\n"
            "    get_tensor_model_parallel_world_size,\n"
            "    get_tp_group,\n"
            ")\n"
            "from vllm.logger import init_logger\n",
        ),
        (
            "\nclass Qwen4ExpForCausalLM(\n",
            "\nlogger = init_logger(__name__)\n" + fp8_block + "\nclass Qwen4ExpForCausalLM(\n",
        ),
        (
            LOAD_WEIGHTS,
            LOAD_WEIGHTS.replace(
                "        return loader.load_weights(weights, mapper=mapper)\n",
                "        loaded = loader.load_weights(weights, mapper=mapper)\n"
                f"        {MARKER}(self)\n"
                "        return loaded\n",
            ),
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", nargs="?", default=DEFAULT_DIR,
                    help=f"patch dir with orig/{ORIG_NAME} (default: {DEFAULT_DIR})")
    args = ap.parse_args()
    orig = os.path.join(args.dir, "orig", ORIG_NAME)
    out = os.path.join(args.dir, OUT_NAME)
    if not os.path.isfile(orig):
        sys.exit(f"ERROR: missing {orig} (engine/patches.sh extracts it from the image)")
    src = open(orig).read()
    if MARKER in src:
        sys.exit(f"ERROR: {orig} is already patched")
    block = FP8_HEAD_KERNELS + TARGET_BLOCK
    for i, (old, new) in enumerate(edits(block)):
        count = src.count(old)
        if count != 1:
            sys.exit(f"target_head_fp8: anchor {i} not unique/missing (count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"target_head_fp8: patched source does not parse: {exc}")
    open(out, "w").write(src)
    print(f"patched {out}")


if __name__ == "__main__":
    main()
