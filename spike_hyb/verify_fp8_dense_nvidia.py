#!/usr/bin/env python3
"""Verify the nvidia-layout hybrid FP8-dense snapshot without a GPU.

Port of fp8dense/verify_fp8_dense_checkpoint.py. Differences: no
`model-bf16-*` shard naming exists here, so "rewritten" shards are derived
from the quantized_layers map itself (any shard holding an
FP8_PER_CHANNEL_PER_TOKEN tensor), and foreign tensors in a rewritten shard
must keep the source dtype/shape (the nvidia shards mix NVFP4/FP8 payloads
inside them).
"""
import argparse
import json
import os
import struct
import sys

import torch


def header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--samples", type=int, default=6)
    a = ap.parse_args()
    src_idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    dst_idx = json.load(open(os.path.join(a.dst, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(a.dst, "config.json")))["quantization_config"]
    ql = cfg["quantized_layers"]
    fp8_layers = {k for k, v in ql.items() if v["quant_algo"] == "FP8_PER_CHANNEL_PER_TOKEN"}
    problems = 0

    # 1. every source tensor still exists (or was replaced by weight+scale)
    for name in src_idx:
        if name not in dst_idx:
            print("MISSING", name); problems += 1
    for lay in fp8_layers:
        for suffix in (".weight", ".weight_scale"):
            if lay + suffix not in dst_idx:
                print("MISSING", lay + suffix); problems += 1
    # 2. shards that hold quantized tensors were rewritten; all others are the
    #    same inode (hard link)
    rewritten = {dst_idx[l + ".weight"] for l in fp8_layers}
    src_files = set(src_idx.values())
    for f in sorted(src_files):
        s, d = os.path.join(a.src, f), os.path.join(a.dst, f)
        if f in rewritten:
            if os.path.exists(d) and os.stat(s).st_ino == os.stat(d).st_ino:
                print("REWRITTEN SHARD IS STILL THE LINK", f); problems += 1
            continue
        if os.path.exists(d) and os.stat(s).st_ino == os.stat(d).st_ino:
            pass
        else:
            print("NOT HARDLINKED", f); problems += 1
    print(f"rewritten shards: {sorted(rewritten)}")

    # 3. dtype/shape checks + dequant error samples in the rewritten shards
    checked = 0
    samples = 0
    for f in sorted(rewritten):
        hdr, data_start = header(os.path.join(a.dst, f))
        src_hdr, src_start = header(os.path.join(a.src, f))
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            base = name[: -len(".weight")] if name.endswith(".weight") else None
            if base in fp8_layers and name.endswith(".weight"):
                sc = hdr.get(base + ".weight_scale")
                if info["dtype"] != "F8_E4M3" or sc is None or sc["dtype"] != "F32" \
                        or sc["shape"] != [info["shape"][0]]:
                    print("BAD QUANT ENTRY", name, info, sc); problems += 1
                checked += 1
                if samples < a.samples and checked % 97 == 1:
                    s0, s1 = info["data_offsets"]; c0, c1 = sc["data_offsets"]
                    with open(os.path.join(a.dst, f), "rb") as fh:
                        fh.seek(data_start + s0)
                        q = torch.frombuffer(bytearray(fh.read(s1 - s0)),
                                             dtype=torch.float8_e4m3fn).view(info["shape"])
                        fh.seek(data_start + c0)
                        scale = torch.frombuffer(bytearray(fh.read(c1 - c0)),
                                                 dtype=torch.float32)
                    o0, o1 = src_hdr[name]["data_offsets"]
                    with open(os.path.join(a.src, f), "rb") as fh:
                        fh.seek(src_start + o0)
                        x = torch.frombuffer(bytearray(fh.read(o1 - o0)),
                                             dtype=torch.bfloat16).view(info["shape"]).float()
                    deq = q.float() * scale[:, None]
                    rel = ((deq - x).norm() / x.norm()).item()
                    print(f"sample {name}: rel_err={rel:.4f} shape={info['shape']}")
                    samples += 1
                    if rel > 0.06:
                        print("HIGH ERROR", name); problems += 1
            elif name in src_hdr:
                if (info["dtype"], info["shape"]) != (src_hdr[name]["dtype"],
                                                      src_hdr[name]["shape"]):
                    print("CHANGED UNTOUCHED TENSOR", name); problems += 1
    print(f"fp8 tensors checked: {checked} (expected {len(fp8_layers)})")
    if checked != len(fp8_layers):
        problems += 1
    print("PROBLEMS:", problems)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
