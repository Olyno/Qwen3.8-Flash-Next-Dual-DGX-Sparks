#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Direct GDN prefill output for the vLLM v0.32.0 lane
(opt-in: gdn_direct_out / VLLM_GDN_DIRECT_OUT=1).

Target: vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py.

Stock prefill: ChunkGatedDeltaRule.forward_cuda calls
fi_chunk_gated_delta_rule, which calls flashinfer's
chunk_gated_delta_rule without its `output=` argument. The kernel then
allocates its own [tokens, v_heads, head_dim] tensor, and the caller
copies the result into the attention layer's preallocated core_attn_out
buffer (a full-tensor copy_ in forward_cuda, then an assignment at the
merge site that consumes the prefill output unchanged).

This overlay passes the destination slice core_attn_out[
:num_actual_tokens] to the same flashinfer kernel as `output=` and
skips both copies. Same kernel, same inputs, same math; only the
destination buffer differs, so bit-parity is expected. Prefill only:
the decode paths, the initial buffer allocation and every state update
are untouched.

Eligibility is a pure runtime predicate, _gdn_direct_out_target: the
destination must be a 3-D tensor with the exact kernel output shape,
the flashinfer output dtype and device, contiguous storage, and 16-byte
aligned. Any mismatch keeps the stock allocate+copy path and logs one
warning; it never raises. The call site additionally takes the direct
path only when the prefill result is merged unchanged (prefill-only
step: no speculative masks, no peeled decode rows), because only then
does the merge target alias the destination slice. Otherwise the stock
result and the stock merge copy stay in place.

Patch time is fail-closed: every anchor must match exactly once, the
result must parse, and nothing is written on a mismatch.

Inputs:  <out_dir>/orig/qwen_gdn_linear_attn.py (extracted from the
         pinned image by engine/patches.sh)
Outputs: <out_dir>/qwen_gdn_linear_attn_v030.py
argv[0] overrides the output directory (used by the test).
"""
import ast
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH_MARK = "_GDN_DIRECT_OUT_ON"


def _gdn_direct_out_target(dest, shape, dtype, device):
    """The destination if the GDN prefill kernel may write it directly.

    Mirrors flashinfer's non-CP output contract: rank 3, exact
    [tokens, heads, head_dim] shape, the output dtype and device,
    contiguous, 16-byte aligned. Returns None (stock allocate+copy) on
    any mismatch, including a non-tensor or a failing attribute access.
    """
    if dest is None:
        return None
    try:
        if dest.ndim != 3:
            return None
        if tuple(dest.shape) != tuple(shape):
            return None
        if dest.dtype != dtype or dest.device != device:
            return None
        if not dest.is_contiguous():
            return None
        if int(dest.data_ptr()) % 16:
            return None
    except (AttributeError, TypeError, RuntimeError):
        return None
    return dest


_MODULE_COMMENT = '''\
# --- direct GDN prefill output (VLLM_GDN_DIRECT_OUT) ------------------------
# patches/patch_gdn_direct_out.py. Prefill only: when
# core_attn_out[:num_actual_tokens] is a valid flashinfer destination
# (rank 3, exact shape, output dtype/device, contiguous, 16-byte aligned),
# the kernel writes it directly and the allocate+copy is skipped. Same
# kernel and inputs -> bit parity expected. Inert unless armed.
# ---------------------------------------------------------------------------
'''


def helper_block() -> str:
    """Gate and eligibility predicate, inserted after the module logger."""
    return (
        _MODULE_COMMENT
        + "_GDN_DIRECT_OUT_ON = (\n"
        '    os.environ.get("VLLM_GDN_DIRECT_OUT", "").strip() == "1"\n'
        ")\n\n\n"
        + inspect.getsource(_gdn_direct_out_target).strip("\n")
        + "\n\n"
    )


MODULE_OLD = "logger = init_logger(__name__)\n"
MODULE_NEW = MODULE_OLD + "\n\n" + helper_block()

FI_SIG_OLD = """\
def fi_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
):
"""
FI_SIG_NEW = """\
def fi_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    output: torch.Tensor | None = None,
):
"""

FI_CALL_OLD = """\
    result = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=torch.exp(fi_g),
        beta=fi_beta,
        initial_state=fi_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        backend="flashinfer",
    )
"""
FI_CALL_NEW = """\
    result = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=torch.exp(fi_g),
        beta=fi_beta,
        initial_state=fi_state,
        output_final_state=output_final_state,
        # The caller's buffer when it is a valid kernel destination; the
        # kernel then writes it in place instead of allocating its own.
        output=output,
        cu_seqlens=cu_seqlens,
        backend="flashinfer",
    )
"""

CUDA_OLD = """\
        o, final_state = fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        if core_attn_out is not None:
            o_flat = o.squeeze(0).reshape(-1)
            co_flat = core_attn_out.reshape(-1)
            co_flat[: o_flat.numel()].copy_(o_flat)
        return o, final_state
"""
CUDA_NEW = """\
        # Write the kernel output straight into the caller's buffer when it
        # meets the kernel's output contract; otherwise allocate and copy.
        direct_out = _gdn_direct_out_target(
            core_attn_out,
            (
                q.shape[-3],
                max(q.shape[-2], v.shape[-2]),
                q.shape[-1],
            ),
            q.dtype,
            q.device,
        )
        o, final_state = fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            output=direct_out,
        )
        if core_attn_out is not None and direct_out is None:
            logger.warning_once(
                "GDN direct output destination is not eligible "
                "(shape/dtype/device/layout); allocating and copying instead."
            )
            o_flat = o.squeeze(0).reshape(-1)
            co_flat = core_attn_out.reshape(-1)
            co_flat[: o_flat.numel()].copy_(o_flat)
        return o, final_state
"""

INIT_OLD = """\
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens
"""
INIT_NEW = """\
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        prefill_out_direct = None
"""

PREFILL_CALL_OLD = """\
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = self.chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=attn_metadata.prefill_query_start_loc,
                chunk_indices=attn_metadata.chunk_indices,
                chunk_offsets=attn_metadata.chunk_offsets,
                use_qk_l2norm_in_kernel=False,
                aiter_prefill_metadata=attn_metadata.aiter_prefill_metadata,
            )
"""
PREFILL_CALL_NEW = """\
            # The prefill result is merged unchanged into the head of
            # core_attn_out only on a prefill-only step (no spec masks, no
            # peeled decode rows); hand that slice to the kernel then.
            if (
                _GDN_DIRECT_OUT_ON
                and self.gdn_prefill_backend == "flashinfer"
                and spec_sequence_masks is None
                and not split_non_spec
            ):
                prefill_out_direct = _gdn_direct_out_target(
                    core_attn_out[:num_actual_tokens],
                    (
                        query_non_spec.shape[-3],
                        max(
                            query_non_spec.shape[-2],
                            value_non_spec.shape[-2],
                        ),
                        query_non_spec.shape[-1],
                    ),
                    query_non_spec.dtype,
                    query_non_spec.device,
                )
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = self.chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=attn_metadata.prefill_query_start_loc,
                chunk_indices=attn_metadata.chunk_indices,
                chunk_offsets=attn_metadata.chunk_offsets,
                use_qk_l2norm_in_kernel=False,
                core_attn_out=prefill_out_direct,
                aiter_prefill_metadata=attn_metadata.aiter_prefill_metadata,
            )
"""

MERGE_OLD = """\
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
"""
MERGE_NEW = """\
        elif prefill_out_direct is None:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
"""

EDITS = (
    (MODULE_OLD, MODULE_NEW),
    (FI_SIG_OLD, FI_SIG_NEW),
    (FI_CALL_OLD, FI_CALL_NEW),
    (CUDA_OLD, CUDA_NEW),
    (INIT_OLD, INIT_NEW),
    (PREFILL_CALL_OLD, PREFILL_CALL_NEW),
    (MERGE_OLD, MERGE_NEW),
)


def patch(src: str) -> str:
    if PATCH_MARK in src:
        sys.exit("ERROR: gdn_direct_out orig qwen_gdn_linear_attn.py "
                 "is already patched")
    for i, (old, new) in enumerate(EDITS):
        count = src.count(old)
        if count != 1:
            sys.exit(f"gdn_direct_out: anchor {i} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit("gdn_direct_out: patched qwen_gdn_linear_attn_v030.py "
                 f"does not parse: {exc}")
    return src


def main(argv) -> int:
    out_dir = argv[0] if argv else os.path.join(HERE, "gdn_direct_out")
    orig_path = os.path.join(out_dir, "orig", "qwen_gdn_linear_attn.py")
    if not os.path.isfile(orig_path):
        sys.exit(f"ERROR: missing {orig_path} "
                 "(engine/patches.sh extracts it from the pinned image)")
    src = patch(open(orig_path).read())
    out_path = os.path.join(out_dir, "qwen_gdn_linear_attn_v030.py")
    with open(out_path + ".tmp", "w") as f:
        f.write(src)
    os.replace(out_path + ".tmp", out_path)
    print("patched qwen_gdn_linear_attn_v030.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
