#!/usr/bin/env python3
"""Refuse illegal MTP configs before weight loading.

The engine sizes the attention block so that one block holds one mamba page.
The GDN conv state in that page has (conv_kernel - 1 + k) rows, so the block
grows with k. The formula follows platforms/interface.py:849-921 of the pinned
image (mamba_cache_mode "align"). It gives the engine log values
"Setting attention block size to 1664 / 1680 / 1728" at k = 3 / 4 / 6 with
SSM bfloat16 + KV fp8.

A k is legal iff the QSA ring capacity for that k divides the block:

    capacity = compress_ratio * ceil((compress_ratio + k) / compress_ratio)

An illegal k hard-fails the engine at config validation — after minutes of
weight loading. k=1 is rejected separately: strictly dominated (same fixed
cache-block cost as k=2, half the decode gain). The widest decode step also
needs max_num_seqs * (1 + k) <= max_num_batched_tokens.

    python3 mtp_block.py <config.json> <k> <max_num_seqs> <max_num_batched_tokens> [ssm_dtype] [kv_dtype] [forced_block]

A 7th argument validates a user-forced --block-size instead of the derived
one: it must stay kernel-aligned, at least the derived size (the engine only
ever raises the block, never lowers it), and divisible by the ring capacity.

Prints "<block> <compress_ratio>" and exits 0 when the combo is legal; prints
the reason to stderr and exits 1 when it is not (including a config.json that
does not have the keys of a Qwen3.8-Flash-Next checkpoint).
"""
import json
import sys

# Bytes per element. An empty SSM dtype means the checkpoint state (float32).
# KV "auto" means the model dtype (bfloat16).
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "half": 2, "float32": 4,
               "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1}
KERNEL_BLOCK_ALIGN = 16
CONV_BYTES = 2  # the conv states use the model dtype (bfloat16)


def ring_capacity(k, compress_ratio):
    """QSA ring rows (models/qwen3_8_flash_next/common/qsa_cache.py:773-785)."""
    return compress_ratio * -(-(compress_ratio + k) // compress_ratio)


def derived_block(cfg, k, ssm_dtype="", kv_dtype="auto"):
    t = cfg.get("text_config", cfg)
    ssm = DTYPE_BYTES.get(ssm_dtype, 4) if ssm_dtype else 4
    kv = 2 if kv_dtype in ("", "auto") else DTYPE_BYTES[kv_dtype]
    kd, nk = t["linear_key_head_dim"], t["linear_num_key_heads"]
    vd, nv = t["linear_value_head_dim"], t["linear_num_value_heads"]
    gdn_conv_dim = kd * nk * 2 + vd * nv
    gdn_page = (gdn_conv_dim * (t["linear_conv_kernel_dim"] - 1 + k) * CONV_BYTES
                + nv * vd * kd * ssm)
    ple_conv_dim = t["hidden_size"] * t.get("hc_count", 4)
    ple_state_len = (t.get("ple_conv_kernel_size", 4) - 1) * t.get("ngram_size", 3)
    ple_page = ple_conv_dim * (ple_state_len + k) * CONV_BYTES
    attn_bytes_per_token = 2 * t["num_key_value_heads"] * t["head_dim"] * kv
    page = max(gdn_page, ple_page)
    return KERNEL_BLOCK_ALIGN * -(-page // (KERNEL_BLOCK_ALIGN * attn_bytes_per_token))


def main(argv):
    if len(argv) < 5:
        sys.exit(__doc__)
    with open(argv[1]) as f:
        cfg = json.load(f)
    k = int(argv[2])
    max_num_seqs = int(argv[3])
    max_num_batched = int(argv[4])
    ssm = argv[5] if len(argv) > 5 else ""
    kv = argv[6] if len(argv) > 6 else "auto"
    forced = int(argv[7]) if len(argv) > 7 else None
    try:
        block = derived_block(cfg, k, ssm, kv)
    except KeyError as e:
        print(f"config.json has no {e}: not a Qwen3.8-Flash-Next checkpoint, "
              "cannot validate MTP", file=sys.stderr)
        return 1
    if forced is not None:
        if forced % KERNEL_BLOCK_ALIGN != 0:
            print(f"forced block {forced} is not a multiple of the kernel "
                  f"block alignment {KERNEL_BLOCK_ALIGN}.", file=sys.stderr)
            return 1
        if forced < block:
            print(f"forced block {forced} is below the derived block {block}: "
                  "the engine raises a smaller --block-size back to the "
                  "derived value, so the forced one would not hold.",
                  file=sys.stderr)
            return 1
        block = forced
    ratio = int(cfg.get("text_config", cfg).get("indexer_compress_ratio", 4))
    if k == 1:
        print("k=1 is strictly dominated: same fixed cache-block cost as k=2, "
              "half the decode gain. Use 2, 3 or 4.", file=sys.stderr)
        return 1
    cap = ring_capacity(k, ratio)
    if block % cap != 0:
        print(f"k={k} is illegal for block {block}, compress ratio {ratio} "
              f"(ring capacity {cap} does not divide the block): illegal k "
              "hard-fails the engine at config validation.", file=sys.stderr)
        return 1
    widest = max_num_seqs * (1 + k)
    if widest > max_num_batched:
        print(f"widest verify batch max_num_seqs*(1+k)={widest} exceeds "
              f"max_num_batched_tokens={max_num_batched}.", file=sys.stderr)
        return 1
    print(f"{block} {ratio}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
