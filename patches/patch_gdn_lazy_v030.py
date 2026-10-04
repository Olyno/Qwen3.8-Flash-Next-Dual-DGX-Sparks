#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""K3 lazy GDN state commit for the vLLM 0.30.0 lane (opt-in, LAZY_GDN=true).

Ports the sfxnz 2x-DGX-Spark recipe's K3 overlay (MIT,
docker/v030/apply_gdn_lazy_overlay.py + gdn_lazy.py; design in its
docker/v030/K3.md) onto the two stock v0.30.0 files:

1. vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
   - __init__: gdl_layer_init(self) self-tests the Triton kernel bitwise
     against the stock CUDA kernel at layer construction (fail closed).
   - _forward_core_decode_spec_post_conv_fused_norm: gdl_decode() replaces
     ops.fused_gdn_decode_post_conv_mtp for pure spec batches.
   - _forward_core: gdl_fixup() materializes pending rings into the stock
     layout before any stock reader.
   - patches/gdn_lazy_k3.py appended verbatim (the kernels).
2. vllm/v1/attention/backends/gdn_attn.py
   - the S1.5 fresh-prefill hunk (a fresh 1-token prompt is a prefill, not a
     decode on a stale state slot; rides along with the K3 metadata overlay);
   - GDNAttentionMetadata.lazy_* fields + the builder's header table.

The stock kernel writes the 64 KiB fp32 GDN state after EVERY verify token;
K3 commits it once per step and replays the exact token inputs from a small
ring instead (~283 MB -> ~116 MB per request per step at k=3 over 36
layers). Lazy and eager call the same jit functions, so outputs and the
committed state are bitwise the stock layout; the load-time self-test
requires bitwise equality on the GPU or K3 stays off.

Requirements (enforced in engine/patches.sh and again fail-closed at
runtime): speculative decoding on (W = k+1 in 2..8, i.e. k in 1..7), an
fp32 GDN state (MAMBA_SSM_CACHE_DTYPE empty/float32), mamba_cache_mode
none/align. Runtime gate: VLLM_QWEN38_GDN_LAZY=1 (passed via OVERLAY_ENV).

Inputs:  patches/v030_gdn/orig/qwen_gdn_linear_attn.py
         patches/v030_gdn/orig/gdn_attn.py   (extracted from the image)
Outputs: patches/v030_gdn/gdn_lazy_linear_attn_v030.py
         patches/v030_gdn/gdn_lazy_attn_v030.py
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG_DIR = os.path.join(HERE, "v030_gdn", "orig")
OUT_DIR = os.path.join(HERE, "v030_gdn")
KERNELS = os.path.join(HERE, "gdn_lazy_k3.py")
EMBED_MARK = "# ---- K3 embedded from patches/gdn_lazy_k3.py (do not hand-edit) ----\n"

# ---------------------------------------------------- qwen_gdn_linear_attn.py
LIN_INIT_OLD = """\
        logger.info_once("GDN decode kernel: %s", self.gdn_decode_kernel)
"""
LIN_INIT_NEW = """\
        logger.info_once("GDN decode kernel: %s", self.gdn_decode_kernel)
        gdl_layer_init(self)  # K3: self-test before any cudagraph capture
"""
LIN_DEC_OLD = """\
        num_requests = attn_metadata.num_spec_decodes
        ops.fused_gdn_decode_post_conv_mtp(
"""
LIN_DEC_NEW = """\
        num_requests = attn_metadata.num_spec_decodes
        if gdl_decode(self, mixed_qkv, a, b, output_gate, core_attn_out, attn_metadata):
            return  # K3 lazy commit (patches/gdn_lazy_k3.py)
        ops.fused_gdn_decode_post_conv_mtp(
"""
LIN_FIX_OLD = """\
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        if (
            self.enable_packed_recurrent_decode
"""
LIN_FIX_NEW = """\
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        gdl_fixup(self, attn_metadata)  # K3: pending rings -> stock layout

        if (
            self.enable_packed_recurrent_decode
"""
LIN_HUNKS = ((LIN_INIT_OLD, LIN_INIT_NEW), (LIN_DEC_OLD, LIN_DEC_NEW),
             (LIN_FIX_OLD, LIN_FIX_NEW))

# ------------------------------------------------------------------ gdn_attn.py
ATT_FRESH_OLD = """\
                split_decodes_and_prefills(m, decode_threshold=1)
"""
ATT_FRESH_NEW = """\
                # S1.5: a fresh 1-token prompt is a prefill (zeroed initial state),
                # not a decode that reads a stale state slot.
                split_decodes_and_prefills(
                    m,
                    decode_threshold=1,
                    treat_short_extends_as_decodes=m.is_prefilling is None,
                )
"""
ATT_FIELDS_OLD = """\
    token_chunk_offset_ptr: torch.Tensor | None = None


class GDNAttentionMetadataBuilder"""
ATT_FIELDS_NEW = """\
    token_chunk_offset_ptr: torch.Tensor | None = None

    # K3 lazy GDN commit (patches/gdn_lazy_k3.py). Unset when
    # VLLM_QWEN38_GDN_LAZY is off.
    lazy_hdr: dict | None = None  # layer name -> int32 [num_blocks]
    lazy_seq_lens: torch.Tensor | None = None  # [batch] num_computed + query_len
    lazy_block_size: int = -1  # align block size; 0 = no align copies
    lazy_ns_state_indices: torch.Tensor | None = None  # [non-spec rows, 1+k]
    lazy_ns_prefilling: torch.Tensor | None = None  # [non-spec rows] bool


class GDNAttentionMetadataBuilder"""
ATT_INIT_OLD = """\
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    def _build_chunk_metadata(
"""
ATT_INIT_NEW = """\
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

        # K3: one int32 header per state block per layer (pending ring length
        # of the committed state in that block; 0 = stock layout).
        import os as _os

        self.lazy_hdr: dict | None = None
        self.lazy_block_size = -1
        _mode = vllm_config.cache_config.mamba_cache_mode
        _nblk = vllm_config.cache_config.num_gpu_blocks
        if (
            _os.environ.get("VLLM_QWEN38_GDN_LAZY", "0").strip().lower() in ("1", "force")
            and self.num_spec > 0
            and _mode in ("none", "align")
            and _nblk
        ):
            self.lazy_hdr = {
                name: torch.zeros(int(_nblk), dtype=torch.int32, device=device)
                for name in layer_names
            }
            self.lazy_block_size = kv_cache_spec.block_size if _mode == "align" else 0

    def _build_chunk_metadata(
"""
ATT_BUILD_OLD = """\
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return attn_metadata

    def build_for_cudagraph_capture(
"""
ATT_BUILD_NEW = """\
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        if self.lazy_hdr is not None:
            # K3: m.seq_lens is the runner's persistent buffer (graph-safe).
            attn_metadata.lazy_hdr = self.lazy_hdr
            attn_metadata.lazy_block_size = self.lazy_block_size
            attn_metadata.lazy_seq_lens = m.seq_lens
            if (num_prefills > 0 or num_decodes > 0) and m.is_prefilling is not None:
                _n = m.num_reqs
                if spec_sequence_masks_cpu is None:
                    _ns = torch.ones(_n, dtype=torch.bool)
                else:
                    _ns = ~spec_sequence_masks_cpu[:_n]
                attn_metadata.lazy_ns_state_indices = block_table_tensor[:_n][
                    _ns, : self.num_spec + 1
                ]
                attn_metadata.lazy_ns_prefilling = async_tensor_h2d(
                    m.is_prefilling[:_n][_ns], device=query_start_loc.device
                )
        return attn_metadata

    def build_for_cudagraph_capture(
"""
ATT_HUNKS = ((ATT_FRESH_OLD, ATT_FRESH_NEW), (ATT_FIELDS_OLD, ATT_FIELDS_NEW),
             (ATT_INIT_OLD, ATT_INIT_NEW), (ATT_BUILD_OLD, ATT_BUILD_NEW))


def _apply(src, hunks, name):
    for i, (old, new) in enumerate(hunks):
        count = src.count(old)
        if count != 1:
            sys.exit(f"gdn_lazy_v030: anchor {i} in {name} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    return src


def main() -> None:
    lin_orig = os.path.join(ORIG_DIR, "qwen_gdn_linear_attn.py")
    att_orig = os.path.join(ORIG_DIR, "gdn_attn.py")
    for orig in (lin_orig, att_orig):
        if not os.path.isfile(orig):
            sys.exit(f"ERROR: missing {orig} (start.sh extracts it from the image)")
    lin = open(lin_orig).read()
    att = open(att_orig).read()
    if "gdl_decode" in lin or "lazy_hdr" in att:
        sys.exit("ERROR: v030_gdn origs are already patched")
    kernels = open(KERNELS).read()
    if "from __future__" in kernels:
        sys.exit("ERROR: gdn_lazy_k3.py must not use __future__ imports "
                 "(it is embedded verbatim at the end of the overlay)")
    lin = _apply(lin, LIN_HUNKS, "qwen_gdn_linear_attn.py")
    lin = lin + "\n\n" + EMBED_MARK + kernels
    att = _apply(att, ATT_HUNKS, "gdn_attn.py")
    for name, src in (("gdn_lazy_linear_attn_v030.py", lin),
                      ("gdn_lazy_attn_v030.py", att)):
        try:
            ast.parse(src)
        except SyntaxError as exc:
            sys.exit(f"gdn_lazy_v030: patched {name} does not parse: {exc}")
        open(os.path.join(OUT_DIR, name), "w").write(src)
        print(f"patched {name}")


if __name__ == "__main__":
    main()
