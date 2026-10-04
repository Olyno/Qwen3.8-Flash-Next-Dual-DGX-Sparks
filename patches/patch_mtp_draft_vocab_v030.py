#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Patch v0.30 qwen4_exp/nvidia/mtp.py: reduced-vocab drafting + FP8 draft head.

Without flags this applies the draft-vocab delta only (VLLM_MTP_DRAFT_VOCAB),
exactly as before. With --fp8 it additionally applies the L1b' FP8 draft-head
delta (ported from the sfxnz 2x-DGX-Spark recipe's apply_mtp_overlay.py, MIT):
VLLM_MTP_DRAFT_HEAD_FP8=1 (or marlin | w8a8) makes the drafter quantize its own
lm_head rows to E4M3 with a per-row scale at load -- the reduced draft-vocab
slice when _attach_draft_vocab built one (the BF16 slice is then dropped), else
this rank's full shard. The target keeps its BF16 lm_head for verification, so
emitted tokens are unchanged; only the draft-step head bandwidth drops (~2x).
Kernel: Marlin W8A16 first, CUTLASS W8A8 fallback, BF16 rows if neither builds
(fail closed, agreed across TP ranks).

Each delta is gated by its own env at runtime, so the two compose
independently: draft vocab only, FP8 only, or both.

Inputs:  patches/mtp_v030_patched.py.orig   (nvidia/mtp.py from the image)
Outputs: patches/mtp_v030_patched.py
"""
import argparse
import ast
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "mtp_v030_patched.py.orig")
OUT = os.path.join(HERE, "mtp_v030_patched.py")

FP8_BLOCK = '''

# L1b' draft-only FP8 lm_head copy (ported from the sfxnz 2x-DGX-Spark
# recipe's apply_mtp_overlay.py, MIT). Opt-in via VLLM_MTP_DRAFT_HEAD_FP8.
_DRAFT_HEAD_FP8_ENV = "VLLM_MTP_DRAFT_HEAD_FP8"
_FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max


def _draft_head_fp8_kernels(value: str) -> tuple:
    """FP8 draft-head kernels to try, in order, for a VLLM_MTP_DRAFT_HEAD_FP8 value."""
    v = value.strip().lower()
    if v in ("", "0", "off", "false", "no"):
        return ()
    if v in ("1", "on", "true", "yes", "auto"):
        return ("marlin", "w8a8")
    if v in ("marlin", "w8a16"):
        return ("marlin",)
    if v in ("w8a8", "cutlass"):
        return ("w8a8",)
    raise ValueError(f"{_DRAFT_HEAD_FP8_ENV}={value!r}: use 1|auto, marlin or w8a8")


def _quantize_rows_fp8(w: torch.Tensor, chunk: int = 8192) -> tuple:
    """Per-row symmetric E4M3: w[i] ~= q[i] * scale[i] (scale fp32, [N]).

    The scale is rounded to w.dtype first, so Marlin (which keeps scales in
    the activation dtype) and CUTLASS (fp32 scales) dequantize identically.
    Chunked to bound the fp32 transient.
    """
    q = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device)
    scale = torch.empty(w.shape[0], dtype=torch.float32, device=w.device)
    for s in range(0, w.shape[0], chunk):
        blk = w[s : s + chunk].float()
        sc = (blk.abs().amax(dim=1) / _FP8_MAX).to(w.dtype).float()
        sc = torch.where(sc > 0, sc, torch.ones_like(sc))  # all-zero rows
        q[s : s + chunk] = (
            (blk / sc[:, None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        )
        scale[s : s + chunk] = sc
    return q, scale


class _Fp8DraftHead:
    """Draft-only FP8 copy of lm_head rows [N, K]; logits(h) -> [M, N].

    kernel "marlin": W8A16, FP8 weights dequantized in the Marlin GEMM,
    activations stay BF16. kernel "w8a8": per-token dynamic FP8 activations
    + CUTLASS scaled_mm with the per-row weight scale.
    """

    def __init__(self, rows: torch.Tensor, kernel: str) -> None:
        q, scale = _quantize_rows_fp8(rows)
        self.kernel = kernel
        self.n, self.k = rows.shape
        self.dtype = rows.dtype
        self.nbytes = q.numel() * q.element_size() + scale.numel() * 4
        if kernel == "marlin":
            from vllm.model_executor.layers.quantization.utils import (
                marlin_utils_fp8 as mu,
            )

            layer = nn.Module()
            layer.output_size_per_partition = self.n
            layer.input_size_per_partition = self.k
            layer.orig_dtype = rows.dtype
            layer.weight = nn.Parameter(q, requires_grad=False)
            layer.weight_scale = nn.Parameter(scale, requires_grad=False)
            mu.prepare_fp8_layer_for_marlin(layer, size_k_first=False)
            self.weight = layer.weight.data
            self.scale = layer.weight_scale.data
            self.workspace = layer.workspace
            self._marlin = mu.apply_fp8_marlin_linear
        elif kernel == "w8a8":
            from vllm import _custom_ops as ops

            self._ops = ops
            self.weight = q.t()  # [K, N] column-major, as cutlass_scaled_mm wants
            self.scale = scale.view(-1, 1)
        else:
            raise ValueError(f"unknown FP8 draft-head kernel {kernel!r}")

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        if self.kernel == "marlin":
            return self._marlin(
                h, self.weight, self.scale, self.workspace, self.n, self.k, None
            )
        hq, hs = self._ops.scaled_fp8_quant(h, use_per_token_if_dynamic=True)
        return self._ops.cutlass_scaled_mm(hq, self.weight, hs, self.scale, self.dtype)

    def check(self, rows: torch.Tensor) -> float:
        """Raise unless logits() matches the BF16 rows to FP8 accuracy."""
        h = torch.randn(4, self.k, dtype=rows.dtype, device=rows.device)
        ref = torch.nn.functional.linear(h, rows).float()
        got = self.logits(h).float()
        if got.shape != ref.shape or not bool(torch.isfinite(got).all()):
            raise RuntimeError(f"bad output {tuple(got.shape)}")
        err = float((got - ref).abs().max() / ref.abs().max().clamp(min=1e-6))
        if err > 0.1:
            raise RuntimeError(f"max rel. error {err:.3f} > 0.1 vs BF16")
        return err


def _build_fp8_draft_head(rows: torch.Tensor, kernels: tuple) -> tuple:
    """First kernel that builds and passes its self-check, else (None, why)."""
    errs = []
    for kernel in kernels:
        try:
            head = _Fp8DraftHead(rows, kernel)
            head.check(rows)
            return head, ""
        except Exception as e:  # noqa: BLE001 - try the next kernel
            errs.append(f"{kernel}: {type(e).__name__}: {e}")
    return None, "; ".join(errs)


def _tp_agree(ok: bool, err: str) -> tuple:
    """MIN over TP ranks on the CPU group; every rank must call it."""
    if get_tensor_model_parallel_world_size() > 1:
        flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
        torch.distributed.all_reduce(
            flag,
            op=torch.distributed.ReduceOp.MIN,
            group=get_tp_group().cpu_group,
        )
        if ok and int(flag.item()) == 0:
            return False, "another TP rank could not build it"
    return ok, err


def _attach_fp8_draft_head(model: nn.Module) -> None:
    """Quantize the drafter's lm_head rows to FP8 (VLLM_MTP_DRAFT_HEAD_FP8).

    Rows are the reduced draft-vocab slice when _attach_draft_vocab built one
    (the BF16 slice is then dropped), else this rank's full lm_head shard.
    Fail closed: any problem logs a warning and drafting stays on the BF16
    rows or the stock head, agreed across TP ranks so no rank diverges.
    """
    try:
        kernels = _draft_head_fp8_kernels(os.environ.get(_DRAFT_HEAD_FP8_ENV, ""))
    except ValueError as exc:
        logger.warning("MTP FP8 draft head: %s; disabled.", exc)
        kernels = ()
    if not kernels:
        return
    if get_pp_group().world_size != 1:
        logger.warning("MTP FP8 draft head: pipeline parallel unsupported; disabled.")
        return
    rows = getattr(model, "_draft_lm_head_weight", None)
    full = rows is None
    head, err = None, ""
    try:
        if full:
            shard = model.lm_head.shard_indices
            w = model.lm_head.weight.data
            if shard.num_org_vocab_padding or shard.num_added_elements_padded:
                raise ValueError("padded or added-vocab lm_head shard")
            if w.dtype not in (torch.bfloat16, torch.float16):
                raise ValueError(f"lm_head dtype {w.dtype} is not BF16/FP16")
            if w.shape[0] != shard.num_org_elements:
                raise ValueError(f"lm_head shard has {w.shape[0]} rows")
            rows = w
        head, err = _build_fp8_draft_head(rows, kernels)
    except Exception as exc:  # noqa: BLE001 - fail closed to BF16
        head, err = None, f"{type(exc).__name__}: {exc}"
    ok, err = _tp_agree(head is not None, err)
    if not ok:
        logger.warning(
            "MTP FP8 draft head disabled (%s); drafting on the BF16 rows.",
            err or "not requested on every rank",
        )
        return
    if full:
        start = int(model.lm_head.shard_indices.org_vocab_start_index)
        model.register_buffer(
            "_draft_id_to_target_id",
            torch.arange(start, start + rows.shape[0], dtype=torch.long,
                         device=rows.device),
            persistent=False,
        )
        model._draft_head_full = True
    else:
        model._draft_lm_head_weight = None  # the FP8 copy replaces the BF16 slice
    model._draft_head_fp8 = head
    logger.info(
        "MTP FP8 draft head (%s): %d rows on this rank, %s, %.1f MiB; "
        "the target keeps its BF16 lm_head for verify.",
        "full vocab" if full else "draft vocab",
        head.n, head.kernel, head.nbytes / 2**20,
    )

'''

# get_top_tokens prologue + branch structure with an FP8-first path; replaces
# the same region of the shared GET_TOP_TOKENS block when --fp8 is on.
GTT_BF16_ONLY = """\
        weight = getattr(self, "_draft_lm_head_weight", None)
        if weight is None:
            return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

        num_tokens = hidden_states.shape[0]
        if weight.shape[0] == 0:
"""
GTT_FP8 = """\
        fp8 = getattr(self, "_draft_head_fp8", None)
        weight = getattr(self, "_draft_lm_head_weight", None)
        if fp8 is None and weight is None:
            return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

        num_tokens = hidden_states.shape[0]
        if fp8 is not None:
            logits = fp8.logits(hidden_states)
            local_max_vals, local_max_indices = logits.max(dim=-1)
            global_indices = self._draft_id_to_target_id[local_max_indices]
        elif weight.shape[0] == 0:
"""

LOGITS_STOCK_LINE = "        return self.logits_processor(self.lm_head, hidden_states)\n"
LOGITS_FP8 = """\
        fp8_head = getattr(self, "_draft_head_fp8", None)
        if fp8_head is not None and getattr(self, "_draft_head_full", False):
            # FP8-only full-vocab draft logits (no reduced vocab): gathered
            # like the stock head, for greedy drafting without local argmax
            # and for probabilistic drafting.
            logits = fp8_head.logits(hidden_states)
            if get_tensor_model_parallel_world_size() > 1:
                logits = self.logits_processor._gather_logits(logits)
            if logits is not None:
                logits = logits[..., : self.logits_processor.org_vocab_size]
            return logits
        return self.logits_processor(self.lm_head, hidden_states)
"""


def _blocks():
    spec = importlib.util.spec_from_file_location("patch_mtp_draft_vocab", os.path.join(HERE, "patch_mtp_draft_vocab.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DRAFT_VOCAB_BLOCK, module.GET_TOP_TOKENS


def _fp8_get_top_tokens(get_top_tokens: str) -> str:
    if get_top_tokens.count(GTT_BF16_ONLY) != 1:
        sys.exit("mtp_v030_patched: shared GET_TOP_TOKENS drifted; "
                 "rebase the FP8 branch against patch_mtp_draft_vocab.py")
    return get_top_tokens.replace(GTT_BF16_ONLY, GTT_FP8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fp8", action="store_true",
                    help="also apply the FP8 draft-head delta (VLLM_MTP_DRAFT_HEAD_FP8)")
    args = ap.parse_args()
    if not os.path.isfile(ORIG):
        sys.exit(f"ERROR: missing {ORIG} (start.sh extracts it from the image)")
    src = open(ORIG).read()
    if "_attach_draft_vocab" in src:
        sys.exit(f"ERROR: {ORIG} is already patched")
    draft_vocab_block, get_top_tokens = _blocks()
    if args.fp8:
        get_top_tokens = _fp8_get_top_tokens(get_top_tokens)
    edits = [
        ("import torch\nfrom torch import nn\n", "import os\n\nimport torch\nfrom torch import nn\n"),
        ("from vllm.distributed import get_pp_group\n",
         "from vllm.distributed import get_pp_group\n"
         + ("from vllm.distributed import (\n"
            "    get_tensor_model_parallel_world_size,\n"
            "    get_tp_group,\n"
            ")\n" if args.fp8 else "")
         + "from vllm.distributed.communication_op import tensor_model_parallel_all_gather\n"
         "from vllm.logger import init_logger\n"),
        ("\nclass Qwen4ExpMultiTokenPredictor(nn.Module):\n",
         "\nlogger = init_logger(__name__)\n" + draft_vocab_block
         + (FP8_BLOCK if args.fp8 else "")
         + "\nclass Qwen4ExpMultiTokenPredictor(nn.Module):\n"),
        (LOGITS_STOCK_LINE,
         (LOGITS_FP8 if args.fp8 else LOGITS_STOCK_LINE) + get_top_tokens),
        ("        return loader.load_weights(remap_weight_names(), mapper=mapper)\n",
         "        loaded = loader.load_weights(remap_weight_names(), mapper=mapper)\n"
         "        _attach_draft_vocab(self)\n"
         + ("        _attach_fp8_draft_head(self)\n" if args.fp8 else "")
         + "        return loaded\n"),
    ]
    for i, (old, new) in enumerate(edits):
        count = src.count(old)
        if count != 1:
            sys.exit(f"mtp_v030_patched: anchor {i} not unique/missing (count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"mtp_v030_patched: patched source does not parse: {exc}")
    open(OUT, "w").write(src)
    print("patched mtp_v030_patched.py" + (" (draft vocab + FP8 head)" if args.fp8 else ""))


if __name__ == "__main__":
    main()
