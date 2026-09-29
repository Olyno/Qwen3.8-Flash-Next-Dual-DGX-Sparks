#!/usr/bin/env python3
"""Build the hybrid FP8-dense checkpoint from the NVIDIA Qwen3.8-Flash-Next-NVFP4 layout.

Port of files/fp8dense/make_fp8_dense_checkpoint.py (written for the RadixArk
layout, which keeps dense bf16 weights in separate `model-bf16-*` shards).
The nvidia layout packs bf16 + FP8 + NVFP4 tensors into one 10-shard set plus a
dedicated FP8 MTP/PLE shard, so instead of rewriting only the bf16 shards this
converter rewrites every shard that actually contains a quantizable dense
tensor, copies foreign tensors verbatim, and merges per-tensor FP8 entries
into the shipped ModelOpt MIXED_PRECISION `quantized_layers` (the shipped
config's 48 NVFP4 expert blocks, FP8 PLE and FP8_BLOCK group survive).

The shipped `exclude_modules` list, by contrast, is REPLACED, not merged:
vLLM's is_layer_excluded() runs BEFORE the quantized_layers lookup and the
nvidia checkpoint excludes exactly the modules we quantize
(`model.language_model.layers.N.linear_attn*`, `.mlp.shared_expert*`,
hyper-connections, `lm_head`) because they ship as bf16. Anything not listed
in quantized_layers falls through to UnquantizedLinearMethod anyway, so the
list only needs the keep-bf16 essentials (same as the RadixArk converter).

Weights quantized: QUANT_PATTERNS (GDN in/out, attention q/k/v/o,
HyperConnection projections, shared expert, lm_head) to FP8 E4M3 with
per-output-channel weight scales + dynamic per-token activation quantization
at runtime (no calibration data needed). Streaming, memory-frugal (row
chunks); run inside the serving image with CUDA_VISIBLE_DEVICES emptied.
Never writes to --src.

Usage (inside the vLLM image, CPU only):
  python3 make_fp8_dense_nvidia.py --src <snapshot dir> --dst <out dir> [--dry-run]
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import struct
import sys
import time

import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8).max)  # 448.0
ROW_CHUNK_BYTES = 16 * 1024 * 1024  # bf16 bytes per processed chunk

# Dense projections that become FP8_PER_CHANNEL_PER_TOKEN (same list as the
# RadixArk converter; the tensor namespace is identical in the nvidia build).
QUANT_PATTERNS = [
    "model.language_model.layers.*.linear_attn.in_proj_qkv.weight",
    "model.language_model.layers.*.linear_attn.in_proj_z.weight",
    "model.language_model.layers.*.linear_attn.out_proj.weight",
    "model.language_model.layers.*.self_attn.q_proj.weight",
    "model.language_model.layers.*.self_attn.k_proj.weight",
    "model.language_model.layers.*.self_attn.v_proj.weight",
    "model.language_model.layers.*.self_attn.o_proj.weight",
    "model.language_model.layers.*.attn_hyper_connection.input_mix_weight_down.weight",
    "model.language_model.layers.*.attn_hyper_connection.block_inject_weight.weight",
    "model.language_model.layers.*.attn_hyper_connection.input_mix_weight_up.weight",
    "model.language_model.layers.*.mlp_hyper_connection.input_mix_weight_down.weight",
    "model.language_model.layers.*.mlp_hyper_connection.block_inject_weight.weight",
    "model.language_model.layers.*.mlp_hyper_connection.input_mix_weight_up.weight",
    "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
    "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
    "model.language_model.layers.*.mlp.shared_expert.gate_proj.weight",
    "model.language_model.layers.*.mlp.shared_expert.up_proj.weight",
    "model.language_model.layers.*.mlp.shared_expert.down_proj.weight",
    "lm_head.weight",
]

# Modules that stay unquantized (verbatim replacement of the shipped list;
# identical semantics to the RadixArk converter). Sibling tensors not named
# in quantized_layers fall through to unquantized without this list.
EXCLUDE_MODULES = [
    "model.language_model.embed_tokens",
    "model.embed_tokens",
    "mtp.*",
    "model.mtp.*",
    "model.visual.*",
    "*.ple.*",
    "*.mlp.gate",
    "*.mlp.shared_expert_gate",
    "*.self_attn.indexer.*",
    "*.self_attn.q_norm",
    "*.self_attn.k_norm",
    "*.linear_attn.in_proj_a",
    "*.linear_attn.in_proj_b",
    "*.linear_attn.in_proj_ba",
    "*.linear_attn.conv1d",
    "*.linear_attn.norm",
    "*.hc_norm",
    "model.language_model.hyper_connection_mixer.block_inject_weight",
]

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "I64": 8,
               "I32": 4, "BOOL": 1, "F64": 8}


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    return hdr, 8 + n


def matches_quant(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in QUANT_PATTERNS)


class SafetensorsWriter:
    """Streaming safetensors writer: header first (sizes known up-front), then data."""

    def __init__(self, path: str, entries: list[tuple[str, str, list[int]]],
                 metadata: dict | None):
        self.path = path
        header: dict = {}
        offset = 0
        self.offsets: dict[str, tuple[int, int]] = {}
        for name, dtype, shape in entries:
            nbytes = DTYPE_BYTES[dtype]
            for d in shape:
                nbytes *= d
            header[name] = {"dtype": dtype, "shape": shape,
                            "data_offsets": [offset, offset + nbytes]}
            self.offsets[name] = (offset, offset + nbytes)
            offset += nbytes
        if metadata:
            header["__metadata__"] = metadata
        hbytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
        pad = (8 - len(hbytes) % 8) % 8
        hbytes += b" " * pad
        self.fh = open(path + ".tmp", "wb")
        self.fh.write(struct.pack("<Q", len(hbytes)))
        self.fh.write(hbytes)
        self.data_start = 8 + len(hbytes)
        self.total = offset
        self.written: dict[str, int] = {}

    def write(self, name: str, chunk: torch.Tensor) -> None:
        start, end = self.offsets[name]
        pos = self.written.get(name, 0)
        # numpy cannot carry fp8/bf16 dtypes; reinterpret contiguous bytes as u8.
        raw = chunk.contiguous()
        buf = (raw if raw.dtype == torch.uint8
               else raw.view(torch.uint8)).numpy().tobytes()
        assert start + pos + len(buf) <= end, name
        self.fh.seek(self.data_start + start + pos)
        self.fh.write(buf)
        self.written[name] = pos + len(buf)

    def close(self) -> None:
        for name, (start, end) in self.offsets.items():
            assert self.written.get(name, 0) == end - start, f"short write for {name}"
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.fh.close()
        os.replace(self.path + ".tmp", self.path)


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric FP8 E4M3: w = round(x / s), s = amax/448."""
    xf = x.float()
    amax = xf.abs().amax(dim=1)
    scale = (amax / FP8_MAX).clamp_min(1e-12)
    q = (xf / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q, scale


def convert_shard(src_path: str, dst_path: str, stats: dict,
                  dry_run: bool) -> list[tuple[str, str]]:
    """Rewrite one shard: quantize matching bf16 tensors, byte-copy the rest."""
    hdr, data_start = read_header(src_path)
    meta = hdr.pop("__metadata__", None)
    entries: list[tuple[str, str, list[int]]] = []
    plan: list[tuple[str, dict, bool]] = []
    for name, info in hdr.items():
        q = matches_quant(name)
        if q:
            assert info["dtype"] == "BF16" and len(info["shape"]) == 2, (name, info)
            entries.append((name, "F8_E4M3", info["shape"]))
            entries.append((name[: -len(".weight")] + ".weight_scale",
                            "F32", [info["shape"][0]]))
        else:
            entries.append((name, info["dtype"], info["shape"]))
        plan.append((name, info, q))
    if dry_run:
        return [(n, d) for n, d, _ in entries]

    writer = SafetensorsWriter(dst_path, entries, meta)
    with open(src_path, "rb") as fh:
        for name, info, q in plan:
            s, e = info["data_offsets"]
            shape = info["shape"]
            if not q:
                fh.seek(data_start + s)
                remaining = e - s
                while remaining:
                    piece = fh.read(min(remaining, 64 << 20))
                    writer.write(name, torch.frombuffer(bytearray(piece),
                                                        dtype=torch.uint8))
                    remaining -= len(piece)
                continue
            rows, cols = shape
            row_bytes = cols * 2
            rows_per_chunk = max(1, ROW_CHUNK_BYTES // row_bytes)
            scale_name = name[: -len(".weight")] + ".weight_scale"
            err_num = 0.0
            err_den = 0.0
            fh.seek(data_start + s)
            for r0 in range(0, rows, rows_per_chunk):
                r1 = min(rows, r0 + rows_per_chunk)
                raw = fh.read((r1 - r0) * row_bytes)
                x = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16) \
                        .view(r1 - r0, cols)
                qv, sc = quantize_rows(x)
                writer.write(name, qv)
                writer.write(scale_name, sc)
                deq = qv.float() * sc[:, None]
                err_num += (deq - x.float()).pow(2).sum().item()
                err_den += x.float().pow(2).sum().item()
            stats[name] = {"rows": rows, "cols": cols,
                           "rel_rmse": (err_num / max(err_den, 1e-30)) ** 0.5}
    writer.close()
    return [(n, d) for n, d, _ in entries]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="nvidia-layout snapshot dir "
                    "(a real directory, e.g. the stock copy — not an HF "
                    "snapshot symlink farm)")
    ap.add_argument("--dst", required=True, help="output directory (created)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="keep already-written output shards")
    args = ap.parse_args()
    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    os.makedirs(dst, exist_ok=True)

    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    files = sorted(set(index["weight_map"].values()))

    # 1. Which shards contain a tensor we would quantize? Those are rewritten;
    #    every other file (incl. the FP8 MTP/PLE shard and pure-NVFP4 expert
    #    payloads) is hard-linked into the output.
    quant_shards: set[str] = set()
    for name, f in index["weight_map"].items():
        if matches_quant(name):
            quant_shards.add(f)
    print(f"{len(quant_shards)} of {len(files)} shards contain quantizable "
          f"dense weights: {sorted(quant_shards)}", flush=True)

    linked = 0
    link_names = [f for f in files if f not in quant_shards]
    link_names += [x for x in os.listdir(src)
                   if not x.endswith(".safetensors")
                   and x not in ("config.json", "hf_quant_config.json",
                                 "model.safetensors.index.json")]
    for f in link_names:
        s, d = os.path.join(src, f), os.path.join(dst, f)
        if os.path.isdir(s):
            continue
        target = os.path.realpath(s)  # HF snapshots are symlinks into blobs/
        if os.path.lexists(d):
            if os.path.exists(d) and os.path.samefile(d, target):
                linked += 1
                continue
            if not args.dry_run:
                os.unlink(d)
        if not args.dry_run:
            try:
                os.link(target, d)
            except OSError as exc:
                # EXDEV (separate /src vs /dst mounts, e.g. build container
                # bind roots). A symlink back to the host path is NOT a safe
                # fallback: the checkpoint must stay self-contained when
                # served from a container that mounts only the dst dir.
                print(f"  hardlink failed for {f} ({exc}); copying", flush=True)
                shutil.copyfile(target, d)
            if not os.path.exists(d):
                raise RuntimeError(f"could not link {f} into {dst}")
        linked += 1
    print(f"linked {linked} untouched files", flush=True)

    # 2. Rewrite the quantized shards, keeping the same shard filenames.
    new_weight_map = {k: v for k, v in index["weight_map"].items()
                      if v not in quant_shards}
    stats: dict = {}
    fp8_layers: dict[str, dict] = {}
    t0 = time.time()
    for f in sorted(quant_shards):
        dst_shard = os.path.join(dst, f)
        if args.resume and os.path.exists(dst_shard):
            print(f"resume: keeping existing {f}", flush=True)
            entries = convert_shard(os.path.join(src, f), dst_shard, stats,
                                    dry_run=True)
        else:
            print(f"converting {f} ...", flush=True)
            entries = convert_shard(os.path.join(src, f), dst_shard, stats,
                                    args.dry_run)
        for name, dtype in entries:
            new_weight_map[name] = f
            if dtype == "F8_E4M3" and matches_quant(name):
                fp8_layers[name[: -len(".weight")]] = {
                    "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"}
        print(f"  done ({time.time() - t0:.0f}s elapsed)", flush=True)

    # 3. Config: merge quantized_layers with the shipped ones, replace
    #    exclude_modules with the canonical keep-list (see module docstring).
    shipped_q = json.load(open(os.path.join(src, "hf_quant_config.json")))
    qbase = dict(shipped_q["quantization"])
    merged_ql = dict(qbase.get("quantized_layers", {}))
    merged_ql.update(fp8_layers)
    qbase["quantized_layers"] = merged_ql
    qbase["exclude_modules"] = list(EXCLUDE_MODULES)
    hfq = {"producer": shipped_q.get("producer"), "quantization": qbase}

    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["quantization_config"] = {
        "quant_method": "modelopt",
        **qbase,
        "comment": "nvidia-layout Qwen3.8-Flash-Next-NVFP4 with dense "
                   "projections requantized to FP8 E4M3 per-output-channel "
                   "(dynamic per-token activations) by hyb_spike/"
                   "make_fp8_dense_nvidia.py; experts, PLE, MTP and "
                   "embeddings shipped as-is.",
    }

    if not args.dry_run:
        json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
        json.dump(hfq, open(os.path.join(dst, "hf_quant_config.json"), "w"),
                  indent=2)
        total = 0
        for f in sorted(set(new_weight_map.values())):
            total += os.path.getsize(os.path.join(dst, f))
        json.dump({"metadata": {"total_size": total},
                   "weight_map": dict(sorted(new_weight_map.items()))},
                  open(os.path.join(dst, "model.safetensors.index.json"), "w"),
                  indent=2)
        json.dump(stats, open(os.path.join(dst, "fp8_dense_quant_stats.json"),
                              "w"), indent=1)
    print(f"quantized {len(fp8_layers)} dense linears to FP8 per-channel; "
          f"expert/PLE/MTP shards linked untouched; "
          f"shipped entries preserved: "
          f"{len(merged_ql) - len(fp8_layers)}")
    if stats:
        worst = sorted(stats.items(), key=lambda kv: -kv[1]["rel_rmse"])[:8]
        print("worst relative RMSE (dequant vs bf16):")
        for n, s in worst:
            print(f"  {s['rel_rmse']:.4f}  {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
