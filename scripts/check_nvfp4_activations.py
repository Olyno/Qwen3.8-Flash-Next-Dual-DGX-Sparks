#!/usr/bin/env python3
"""Flag NVFP4-quantized tensors whose activation-outlier profile makes FP4 risky.

Massive activations (a few channels orders of magnitude above the rest) break
the small-dynamic-range assumption of 4-bit quantization. The checkpoint does
not record per-channel activation stats -- those need runtime hooks, a GPU-host
follow-up -- but every NVFP4 linear carries two calibration-time scalars that
proxy it: `input_scale` (activation amax scale, F32) and `weight_scale_2`
(global weight scale, F32). Both are readable with the stdlib alone by parsing
safetensors headers and seeking to the 4 data bytes, so the whole checkpoint
is surveyed in seconds without torch and without loading any weights.

  python3 scripts/check_nvfp4_activations.py ~/models/Qwen3.8-Flash-Next-NVFP4-lean
  python3 scripts/check_nvfp4_activations.py --channel-stats hooks.json   # GPU-host hook dump
  python3 scripts/check_nvfp4_activations.py --self-check

Tensors are grouped into families (layers.N / experts.N collapsed) and each
family's input_scale spread is compared against the thresholds below; tensors
past FAIL are listed individually. --channel-stats consumes a JSON mapping of
tensor name -> per-channel activation absmax (from a future runtime hook) and
applies the per-channel thresholds. Exit status: 0 = PASS/WARN, 1 = any FAIL.

Thresholds are heuristics, named so a report line can say exactly which one
fired; tune them against measured acceptance if they prove jumpy.
"""
import json
import math
import os
import re
import statistics
import struct
import sys

# Family-level: max/median input_scale across the tensors of one family.
INPUT_SCALE_SPREAD_WARN = 5.0
INPUT_SCALE_SPREAD_FAIL = 25.0
# Per-channel: channel absmax vs the tensor's median channel absmax. LLM.int8()
# used ~6 as the outlier gate for 8-bit; FP4 has far less headroom, so the warn
# line is tighter and 100x is treated as disqualifying.
CHANNEL_OUTLIER_RATIO_WARN = 20.0
CHANNEL_OUTLIER_RATIO_FAIL = 100.0
# ...but only if at least this fraction of channels crosses the warn ratio.
CHANNEL_OUTLIER_FRAC_WARN = 0.001

_HEADER_STRUCT = struct.Struct("<Q")
_SCALAR_SIZE = {"F64": ("<d", 8), "F32": ("<f", 4), "F16": ("<e", 2),
                "I64": ("<q", 8), "I32": ("<i", 4), "U8": ("<B", 1)}


def _bf16_to_float(raw: bytes) -> float:
    return struct.unpack("<f", b"\x00\x00" + raw)[0]


def read_header(path: str) -> tuple[dict, int]:
    """Return (tensor_name -> safetensors metadata, data-buffer offset)."""
    with open(path, "rb") as handle:
        (size,) = _HEADER_STRUCT.unpack(handle.read(8))
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    return header, 8 + size


def read_scalar(path: str, meta: dict, data_offset: int) -> float:
    """Read a 0-d/1-element tensor's value without loading anything else."""
    start, end = meta["data_offsets"]
    with open(path, "rb") as handle:
        handle.seek(data_offset + start)
        raw = handle.read(end - start)
    dtype = meta["dtype"]
    if dtype == "BF16":
        return _bf16_to_float(raw)
    fmt, size = _SCALAR_SIZE[dtype]
    assert len(raw) == size, f"{dtype} scalar is {len(raw)} bytes, want {size}"
    return struct.unpack(fmt, raw)[0]


def family_of(tensor_name: str) -> str:
    """Collapse layer/expert indices so e.g. all experts' down_proj share a family."""
    return re.sub(r"\.\d+\.", ".*.", tensor_name)


def shard_map(model_dir: str) -> dict:
    """tensor name -> shard path, from the index json or the shards themselves."""
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as handle:
            weight_map = json.load(handle)["weight_map"]
        return {name: os.path.join(model_dir, shard)
                for name, shard in weight_map.items()}
    out = {}
    for name in sorted(os.listdir(model_dir)):
        if name.endswith(".safetensors"):
            header, _ = read_header(os.path.join(model_dir, name))
            for tensor in header:
                out[tensor] = os.path.join(model_dir, name)
    return out


def collect_scales(model_dir: str) -> dict:
    """input_scale/weight_scale_2 per NVFP4-quantized linear in the checkpoint."""
    tensors = shard_map(model_dir)
    quantized = sorted(name[:-len(".input_scale")] for name in tensors
                       if name.endswith(".input_scale"))
    scales = {}
    headers: dict[str, tuple] = {}
    for base in quantized:
        row = {}
        for suffix in ("input_scale", "weight_scale_2"):
            name = f"{base}.{suffix}"
            if name not in tensors:
                continue
            shard = tensors[name]
            if shard not in headers:
                headers[shard] = read_header(shard)
            header, data_offset = headers[shard]
            row[suffix] = read_scalar(shard, header[name], data_offset)
        scales[base] = row
    return scales


def outlier_stats(values: list) -> dict:
    """max/median spread and outlier counts for one family or channel set."""
    med = statistics.median(values)
    peak = max(values)
    ratio = peak / med if med > 0 else math.inf
    n_out = sum(1 for v in values if med > 0 and v > CHANNEL_OUTLIER_RATIO_WARN * med)
    return {"n": len(values), "median": med, "max": peak, "ratio": ratio,
            "n_outliers": n_out, "frac_outliers": n_out / len(values) if values else 0.0}


def verdict_family(stats: dict) -> str:
    if stats["ratio"] >= INPUT_SCALE_SPREAD_FAIL:
        return "FAIL"
    if stats["ratio"] >= INPUT_SCALE_SPREAD_WARN:
        return "WARN"
    return "PASS"


def verdict_channels(stats: dict) -> str:
    if stats["ratio"] >= CHANNEL_OUTLIER_RATIO_FAIL:
        return "FAIL"
    if (stats["ratio"] >= CHANNEL_OUTLIER_RATIO_WARN
            and stats["frac_outliers"] >= CHANNEL_OUTLIER_FRAC_WARN):
        return "WARN"
    return "PASS"


def report_scales(scales: dict) -> int:
    families: dict[str, list] = {}
    for base, row in scales.items():
        if "input_scale" in row:
            families.setdefault(family_of(base), []).append((base, row["input_scale"]))
    print(f"{len(scales)} NVFP4-quantized linears in {len(families)} families")
    print(f"thresholds: input_scale spread warn>={INPUT_SCALE_SPREAD_WARN:g}x "
          f"fail>={INPUT_SCALE_SPREAD_FAIL:g}x family median")
    worst = 0
    flagged = []
    for fam in sorted(families):
        rows = families[fam]
        stats = outlier_stats([v for _, v in rows])
        verdict = verdict_family(stats)
        if verdict != "PASS":
            flagged.append((fam, rows, stats, verdict))
            worst = max(worst, 2 if verdict == "FAIL" else 1)
        print(f"{verdict:4} {fam:60} n={stats['n']:<5} input_scale "
              f"median={stats['median']:.4g} max={stats['max']:.4g} "
              f"spread={stats['ratio']:.2f}x")
    for fam, rows, stats, verdict in sorted(flagged, key=lambda f: -f[2]["ratio"]):
        top = sorted(rows, key=lambda r: -r[1])[:5]
        for name, value in top:
            print(f"     {verdict:4} {name} input_scale={value:.4g}")
    print("overall:", "FAIL" if worst == 2 else "WARN" if worst else "PASS")
    return 1 if worst == 2 else 0


def report_channel_stats(path: str) -> int:
    with open(path) as handle:
        dump = json.load(handle)
    print(f"thresholds: channel outlier warn>={CHANNEL_OUTLIER_RATIO_WARN:g}x "
          f"fail>={CHANNEL_OUTLIER_RATIO_FAIL:g}x tensor median, "
          f"frac>={CHANNEL_OUTLIER_FRAC_WARN:g}")
    worst = 0
    for name in sorted(dump):
        stats = outlier_stats([abs(v) for v in dump[name]])
        verdict = verdict_channels(stats)
        worst = max(worst, 2 if verdict == "FAIL" else 1 if verdict == "WARN" else 0)
        print(f"{verdict:4} {name:60} channels={stats['n']:<6} "
              f"median={stats['median']:.4g} max={stats['max']:.4g} "
              f"ratio={stats['ratio']:.2f}x outliers={stats['n_outliers']}")
    print("overall:", "FAIL" if worst == 2 else "WARN" if worst else "PASS")
    return 1 if worst == 2 else 0


def _write_synthetic_shard(path: str, tensors: dict) -> None:
    """Minimal safetensors file: name -> (dtype, shape, packed bytes)."""
    header, blob, offset = {}, b"", 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + len(raw)]}
        blob += raw
        offset += len(raw)
    encoded = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(_HEADER_STRUCT.pack(len(encoded)))
        handle.write(encoded)
        handle.write(blob)


def _self_check() -> None:
    import tempfile
    # Channel analysis: a flat tensor passes, one massive channel fails.
    flat = outlier_stats([1.0] * 100)
    assert verdict_channels(flat) == "PASS", flat
    spiked = outlier_stats([1.0] * 99 + [500.0])
    assert spiked["n_outliers"] == 1 and spiked["ratio"] >= 100
    assert verdict_channels(spiked) == "FAIL", spiked
    mild = outlier_stats([1.0] * 99 + [30.0])
    assert verdict_channels(mild) == "WARN", mild
    # Family verdicts.
    assert verdict_family({"ratio": 1.0}) == "PASS"
    assert verdict_family({"ratio": INPUT_SCALE_SPREAD_WARN}) == "WARN"
    assert verdict_family({"ratio": INPUT_SCALE_SPREAD_FAIL}) == "FAIL"
    # Reader: synthetic 2-shard checkpoint with a planted outlier family.
    f32 = lambda v: struct.pack("<f", v)
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "sub"), exist_ok=True)
        shard = os.path.join(tmp, "model.safetensors")
        tensors = {}
        for layer in range(3):
            for expert in range(4):
                base = f"model.layers.{layer}.mlp.experts.{expert}.down_proj"
                scale = 1.0 + 0.1 * expert
                if layer == 2 and expert == 3:
                    scale = 200.0  # planted massive-activation proxy
                tensors[f"{base}.input_scale"] = ("F32", [], f32(scale))
                tensors[f"{base}.weight_scale_2"] = ("F32", [], f32(0.5))
                tensors[f"{base}.weight"] = ("U8", [8], b"\x00" * 8)
        tensors["model.embed_tokens.weight"] = ("BF16", [4], b"\x00" * 8)
        _write_synthetic_shard(shard, tensors)
        scales = collect_scales(tmp)
        assert len(scales) == 12, scales.keys()
        fam = "model.layers.*.mlp.experts.*.down_proj"
        stats = outlier_stats([r["input_scale"] for r in scales.values()])
        assert stats["ratio"] >= INPUT_SCALE_SPREAD_FAIL, stats
        assert verdict_family(stats) == "FAIL"
        assert family_of("model.layers.2.mlp.experts.3.down_proj.input_scale") \
            == f"{fam}.input_scale"
        header, data_offset = read_header(shard)
        meta = header["model.layers.2.mlp.experts.3.down_proj.input_scale"]
        assert read_scalar(shard, meta, data_offset) == 200.0
    print("self-check OK")


def main(argv: list) -> int:
    if "--self-check" in argv or not argv:
        _self_check()
        return 0 if "--self-check" in argv else print(__doc__) or 0
    if argv[0] == "--channel-stats":
        return report_channel_stats(argv[1])
    return report_scales(collect_scales(argv[0]))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
