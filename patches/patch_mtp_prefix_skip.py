#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Patch the v030-lane qwen4_exp/nvidia/mtp.py: skip the MTP drafter's discarded
prefill computation (opt-in: mtp_skip_prefix / VLLM_MTP_SKIP_PREFIX).

Idea-level credit, no code copied: ds4-dfm-rs (MIT),
docs/qwen38-perf-2026-09-14.md and docs/ds41-qwen38-serve-lessons.md (Q-2);
independently flashrt (Apache-2.0), docs/improvement-plan.md:212-217. Both
measured the same mechanism on their engines: during prefill the MTP block's
output is consumed only for the last prompt row of each sequence, and
skipping the discarded sub-computation collapsed the draft-prefill kernel
stream (flashrt: 78 kernels / 136 ms -> 27 / 22 ms) with frontier logits
byte-identical.

The consumption site for this lane is the V2 speculator's _prefill()
(vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py): it runs the
draft model over the full prefill batch and keeps exactly
last_hidden_states[last_token_indices] and the same rows of the multi
stream; every other row the forward produced is dropped. What the discarded
rows still own is the drafter's own KV cache: the pre-attention K/V they
write during draft prefill is read back by every later decode draft step,
so attention (and everything feeding it: the embedding branch, fc_embedding
/ fc_hidden, and inside Qwen4ExpDecoderLayer the qkv GEMM, q-norm/RoPE, the
QSA indexer, the attention kernel itself, o_proj and the intra-layer HC
combines) must stay full width. The layer call in this file is a single
monolithic call, so a patch scoped to mtp.py can only reach the two
post-attention per-row stages, and that is what this skips:

  - the layer's sparse-MoE block: the module instance's `forward` attribute
    is swapped (the module object and its parameter names stay intact for
    AutoWeightsLoader, EPLB and the skinny-GEMM registrant) to compute only
    the per-sequence tail rows and scatter them into zeros elsewhere;
  - the final hyper_connection_mixer.combine_and_mix, sliced to the same
    rows and scattered (its outputs feed the draft logits and the
    next-step stream only at those rows).

The zeroed rows are provably never consumed: within one MTP layer
everything downstream of the attention kernel is per-row, the attention K/V
is written before the MoE, and the speculator indexes only
last_token_indices (== query_start_loc[1:] - 1) out of the returned
tensors. Target verification is untouched, so even a miscomputed draft
could only cost acceptance, never correctness — but the skip is held to
bit-parity anyway:

Fail-closed layers, outermost first:
  - VLLM_MTP_SKIP_PREFIX unset -> nothing is swapped, zero overhead;
  - arming refuses under data_parallel_size > 1 (the MoE token count must
    match across DP ranks), pipeline_parallel_size > 1, or a cudagraph mode
    outside the verified set (NONE / PIECEWISE / FULL_DECODE_ONLY; anything
    that can capture a T > P draft prefill is refused with a warning);
  - no query_start_loc in the forward context (warmup, profile run), DP
    sync active, non-2D input, or a uniform-decode batch (every draft
    decode step and every FULL-graph replay) -> full width for that call;
  - parity self-check before anything is trusted: the first eager draft
    prefill computes BOTH variants of each stage and keeps the skip only if
    the tail rows match bit-exactly; a mismatch warns and pins that stage
    to stock forever. This is this lane's port of flashrt's pre-accept
    parity gate, run in place on real batch data instead of as a patch-time
    script (the patcher cannot boot the model). Boot verification: enable
    the key, look for the two "parity" INFO lines during the first
    prefill, then measure TTFT with bench/longctx.py.

Composes with patches/patch_mtp_draft_vocab_v030.py (disjoint regions: that
patcher touches imports, compute_logits and load_weights; this one touches
__init__, forward and the class prelude). engine/patches.sh chains this
patcher on the draft-vocab output when that block is active and mounts the
chained file itself, so the draft-vocab block skips its own mount.

Inputs:  argv[1] (default patches/mtp_prefix_skip) holds orig/mtp.py — the
         image's mtp.py, or patches/mtp_v030_patched.py when the draft-vocab
         patcher ran first (engine/patches.sh chains it)
Outputs: argv[1]/mtp_v030.py
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

HELPERS = '''
# --- prefill tail skip (VLLM_MTP_SKIP_PREFIX; patches/patch_mtp_prefix_skip.py)
# During draft prefill the speculator consumes the per-sequence last row only
# (last_hidden_states[last_token_indices]); the sparse-MoE block and the final
# HC mixer over the other prompt rows are computed and thrown away. Attention
# stays full width — its K/V for every prompt row populates the drafter's own
# KV cache that the decode draft steps read. Both skipped stages must pass a
# bit-parity self-check on the first eager prefill before being trusted; a
# mismatch pins that stage to stock.

_MTP_SKIP_PREFIX_ENV = "VLLM_MTP_SKIP_PREFIX"
_PREFIX_SKIP = {
    "armed": False,
    "moe_checked": False,
    "moe_ok": False,
    "mixer_checked": False,
    "mixer_ok": False,
}
_PREFIX_SKIP_LOGGER = None


def _skip_log():
    global _PREFIX_SKIP_LOGGER
    if _PREFIX_SKIP_LOGGER is None:
        from vllm.logger import init_logger

        _PREFIX_SKIP_LOGGER = init_logger(__name__)
    return _PREFIX_SKIP_LOGGER


def _arm_prefix_skip(model, vllm_config) -> None:
    """Swap the MTP layers' MoE forward for the tail-windowed variant."""
    import os

    if os.environ.get(_MTP_SKIP_PREFIX_ENV, "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    from vllm.config.compilation import CUDAGraphMode

    mode = vllm_config.compilation_config.cudagraph_mode
    if mode not in (
        CUDAGraphMode.NONE,
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL_DECODE_ONLY,
    ):
        _skip_log().warning(
            "MTP prefix skip: cudagraph mode %s can capture a wide draft "
            "prefill graph; not armed, drafting stock.",
            mode,
        )
        return
    if vllm_config.parallel_config.data_parallel_size > 1:
        _skip_log().warning("MTP prefix skip: DP>1 unsupported; not armed.")
        return
    if vllm_config.parallel_config.pipeline_parallel_size > 1:
        _skip_log().warning("MTP prefix skip: PP>1 unsupported; not armed.")
        return
    armed = False
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, Qwen4ExpSparseMoeBlock):
            _arm_moe_tail_skip(mlp)
            armed = True
    if not armed:
        return
    _PREFIX_SKIP["armed"] = True
    _skip_log().info(
        "MTP prefix skip armed: MoE and final mixer run on the per-sequence "
        "tail rows during draft prefill; parity self-check pending the first "
        "eager prefill.",
    )


def _arm_moe_tail_skip(mlp) -> None:
    inner = mlp.forward  # bound before the swap; the module object itself (and
    # its parameter names for the loader, EPLB and the skinny-GEMM registrant)
    # stays untouched, only the instance-level forward is replaced.

    def forward(hidden_states, *args, **kwargs):
        if args or kwargs:
            return inner(hidden_states, *args, **kwargs)
        tail = _prefill_tail_rows(hidden_states)
        if tail is None:
            return inner(hidden_states)
        return _tail_apply(tail, hidden_states, inner, "moe")

    mlp.forward = forward


def _prefill_tail_rows(hidden_states):
    """Int64 CUDA indices of each sequence's last row, or None for full width.

    Padded (zero-query-length) slots duplicate an existing row; recomputing a
    duplicate and scattering it with the same value is idempotent.
    """
    if not _PREFIX_SKIP["armed"] or hidden_states.dim() != 2:
        return None
    from vllm.forward_context import get_forward_context

    try:
        ctx = get_forward_context()
    except Exception:  # outside a forward context (warmup)
        return None
    if ctx is None or getattr(ctx, "num_tokens_across_dp", None) is not None:
        return None  # DP: MoE token counts must match across ranks
    metadata = getattr(ctx, "attn_metadata", None)
    qsl = None
    if isinstance(metadata, dict):
        for value in metadata.values():
            cand = getattr(value, "query_start_loc", None)
            if isinstance(cand, torch.Tensor) and cand.numel() >= 2:
                qsl = cand
                break
    if qsl is None:
        return None
    num_rows = hidden_states.shape[0]
    if qsl.numel() - 1 >= num_rows:
        return None  # uniform decode (draft decode steps, FULL graph replays)
    return (qsl[1:] - 1).clamp_(0, num_rows - 1).to(torch.long)


def _scatter_tail(sliced, tail, num_rows):
    out = sliced.new_zeros((num_rows, *sliced.shape[1:]))
    return out.index_copy_(0, tail, sliced)


def _parity(which, full_tail_rows, sliced) -> bool:
    _PREFIX_SKIP[f"{which}_checked"] = True
    if torch.equal(full_tail_rows, sliced):
        _PREFIX_SKIP[f"{which}_ok"] = True
        _skip_log().info(
            "MTP prefix skip %s parity: bit-identical on the tail rows; "
            "enabled.",
            which,
        )
        return True
    rel = float(
        (
            (full_tail_rows.float() - sliced.float()).abs().max()
            / full_tail_rows.float().abs().max().clamp(min=1.0)
        ).item()
    )
    _PREFIX_SKIP[f"{which}_ok"] = False
    _skip_log().warning(
        "MTP prefix skip %s parity FAILED (max rel err %.3g); "
        "stage stays stock.",
        which,
        rel,
    )
    return False


def _tail_apply(tail, hidden_states, fn, which):
    """fn over the tail rows only, scattered to the batch, once parity holds."""
    if _PREFIX_SKIP[f"{which}_checked"] and not _PREFIX_SKIP[f"{which}_ok"]:
        return fn(hidden_states)
    if not _PREFIX_SKIP[f"{which}_checked"]:
        if torch.cuda.is_current_stream_capturing():
            return fn(hidden_states)  # no sync during capture; defer the check
        full = fn(hidden_states)
        sliced = fn(hidden_states.index_select(0, tail))
        if not _parity(which, full.index_select(0, tail), sliced):
            return full
        return _scatter_tail(sliced, tail, hidden_states.shape[0])
    return _scatter_tail(
        fn(hidden_states.index_select(0, tail)), tail, hidden_states.shape[0]
    )


def _mixer_tail_combine(tail, mixer, hidden_states, block_output, injection):
    """combine_and_mix over the tail rows, scattered; parity-checked like MoE."""
    inputs = (hidden_states, block_output, injection)
    if _PREFIX_SKIP["mixer_checked"] and not _PREFIX_SKIP["mixer_ok"]:
        return mixer.combine_and_mix(*inputs)
    if not _PREFIX_SKIP["mixer_checked"]:
        if torch.cuda.is_current_stream_capturing():
            return mixer.combine_and_mix(*inputs)
        full = mixer.combine_and_mix(*inputs)
        sliced = mixer.combine_and_mix(
            *(t.index_select(0, tail) for t in inputs)
        )
        # The third stream is discarded by the caller; only compare what is
        # consumed (the LM-head stream and the next-step multi stream).
        ok = all(
            torch.equal(a.index_select(0, tail), b)
            for a, b in zip(full[:2], sliced[:2])
        )
        _PREFIX_SKIP["mixer_checked"] = True
        if ok:
            _PREFIX_SKIP["mixer_ok"] = True
            _skip_log().info(
                "MTP prefix skip mixer parity: bit-identical on the tail "
                "rows; enabled.",
            )
        else:
            _PREFIX_SKIP["mixer_ok"] = False
            _skip_log().warning(
                "MTP prefix skip mixer parity FAILED; final mixer stays stock.",
            )
            return full
    else:
        sliced = mixer.combine_and_mix(*(t.index_select(0, tail) for t in inputs))
    return (
        _scatter_tail(sliced[0], tail, hidden_states.shape[0]),
        _scatter_tail(sliced[1], tail, hidden_states.shape[0]),
        sliced[2],
    )


'''

INIT_ARM = """\
                for idx in range(self.num_mtp_layers)
            )
        _arm_prefix_skip(self, vllm_config)
"""

FWD_MIXER_OLD = """\
        multi_hidden, sample_hidden_states, _ = (
            self.hyper_connection_mixer.combine_and_mix(
                hidden_states, block_output, injection
            )
        )
"""
FWD_MIXER_NEW = """\
        tail = _prefill_tail_rows(hidden_states)
        if tail is None:
            multi_hidden, sample_hidden_states, _ = (
                self.hyper_connection_mixer.combine_and_mix(
                    hidden_states, block_output, injection
                )
            )
        else:
            multi_hidden, sample_hidden_states, _ = _mixer_tail_combine(
                tail,
                self.hyper_connection_mixer,
                hidden_states,
                block_output,
                injection,
            )
"""

# (anchor, replacement) pairs. Class-prelude and __init__ anchors are regions
# the draft-vocab patcher never touches, so this composes with
# patch_mtp_draft_vocab_v030.py in either chain order.
EDITS = [
    (
        "class Qwen4ExpMTP(nn.Module, SupportsHCSP, SupportsPP, Qwen4ExpMixtureOfExperts):\n",
        HELPERS + "class Qwen4ExpMTP(nn.Module, SupportsHCSP, SupportsPP, Qwen4ExpMixtureOfExperts):\n",
    ),
    (
        """\
                for idx in range(self.num_mtp_layers)
            )
""",
        INIT_ARM,
    ),
    (FWD_MIXER_OLD, FWD_MIXER_NEW),
]


def main(argv) -> int:
    out_dir = argv[0] if argv else os.path.join(HERE, "mtp_prefix_skip")
    orig_path = os.path.join(out_dir, "orig", "mtp.py")
    if not os.path.isfile(orig_path):
        sys.exit(f"ERROR: missing {orig_path} "
                 "(engine/patches.sh extracts the image file, or chains the "
                 "draft-vocab output here)")
    src = open(orig_path).read()
    if "_arm_prefix_skip" in src:
        sys.exit("ERROR: mtp_prefix_skip orig mtp.py is already patched")
    for i, (old, new) in enumerate(EDITS):
        count = src.count(old)
        if count != 1:
            sys.exit(f"mtp_prefix_skip: anchor {i} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"mtp_prefix_skip: patched mtp_v030.py does not parse: {exc}")
    open(os.path.join(out_dir, "mtp_v030.py"), "w").write(src)
    print("patched mtp_v030.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
