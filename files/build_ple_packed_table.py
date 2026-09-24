#!/usr/bin/env python3
"""Build a packed PLE table file for memory-mapped CPU offload (GB10/tp1).

Auto-detects the checkpoint's PLE quant format from the shard tensors:

  NVFP4  (local-inference-lab / RadixArk PLE=fp4): shards carry `.weight` U8
    codes [rows, head_dim/2] AND `.weight_scale` F8_E4M3 [rows, head_dim/16].
    Packed row = cat(codes, scales-as-u8) -> width head_dim/2 + head_dim/16.

  FP8    (nvidia/Qwen3.8-Flash-Next-NVFP4): shards carry ONLY `.weight`
    F8_E4M3 [rows, head_dim]. A byte is a byte: packed row = raw weight bytes,
    width = head_dim. The CPU worker's F8_E4M3 embedding returns exactly these
    bytes (view uint8), and the GPU side applies the single global scale.

Layout: one flat file <layer>.ngram_embedding.packed_u8 = [total_rows, width]
uint8, row r of shard i at index i*rows_per_shard + r; sidecar .json mirrors
what patch_ple_offload._attach_packed_table checks (total_rows, row_width).
Streams shard-by-shard with memmaps; peak RAM << 1 GiB.

Usage: build_ple_packed_table.py <snapshot_dir> <out_dir>
"""
import json, os, struct, sys, time
import numpy as np

snap, out_dir = sys.argv[1], sys.argv[2]
idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
prefix_of = {}
for k in idx:
    if ".ngram_embedding.shard_0.weight" in k and not k.endswith("weight_scale"):
        prefix_of[k[: k.index(".shard_0.weight")]] = True
if not prefix_of:
    sys.exit("no PLE ngram_embedding shards in index")

headers = {}
def header(fname):
    if fname not in headers:
        with open(os.path.join(snap, fname), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            headers[fname] = (json.loads(f.read(n)), 8 + n)
    return headers[fname]

def view(name):
    fname = idx[name]
    h, base = header(fname)
    meta = h[name]
    start, end = meta["data_offsets"]
    mm = np.memmap(os.path.join(snap, fname), dtype=np.uint8, mode="r",
                   offset=base + start, shape=(end - start,))
    return mm.reshape(meta["shape"]), meta["dtype"]

os.makedirs(out_dir, exist_ok=True)
for prefix in prefix_of:
    # vLLM name: strip leading "model." and map language_model -> language_model.model
    vname = prefix
    if vname.startswith("model.language_model."):
        vname = "language_model.model." + vname[len("model.language_model."):]
    shards = sorted({int(k[len(prefix) + len(".shard_"):].split(".")[0])
                     for k in idx if k.startswith(prefix + ".shard_")})
    assert shards == list(range(len(shards))), shards
    w0, wdt = view(f"{prefix}.shard_0.weight")
    have_scales = f"{prefix}.shard_0.weight_scale" in idx
    if have_scales:
        assert wdt == "U8", f"NVFP4 path expects U8 codes, got {wdt}"
        s0, sdt = view(f"{prefix}.shard_0.weight_scale")
        assert sdt == "F8_E4M3", sdt
        rows, cw = w0.shape; sw = s0.shape[1]; width = cw + sw
        kind = "nvfp4"
    elif wdt == "F8_E4M3":
        rows, width = w0.shape; cw = width; sw = 0
        kind = "fp8"
    else:
        sys.exit(f"unsupported PLE shard dtype {wdt} (no weight_scale tensors)")
    out_name = os.path.join(out_dir, vname + ".packed_u8")
    meta = {"kind": kind, "rows_per_shard": rows, "num_shards": len(shards),
            "row_width": width, "codes_width": cw, "scales_width": sw,
            "total_rows": rows * len(shards),
            "snapshot": os.path.basename(os.path.normpath(snap))}
    if os.path.exists(out_name) and os.path.getsize(out_name) == rows * len(shards) * width:
        print("exists:", out_name); continue
    print(f"building {out_name} [{kind}]: {len(shards)} shards x {rows} rows x {width} B = "
          f"{rows*len(shards)*width/2**30:.2f} GiB", flush=True)
    t0 = time.time()
    tmp = out_name + ".tmp"
    CH = 1 << 19
    with open(tmp, "wb") as out:
        for i in shards:
            w, _ = view(f"{prefix}.shard_{i}.weight")
            if have_scales:
                s, _ = view(f"{prefix}.shard_{i}.weight_scale")
                assert w.shape == (rows, cw) and s.shape == (rows, sw), (i, w.shape, s.shape)
                for c in range(0, rows, CH):
                    np.concatenate([w[c:c + CH], s[c:c + CH]], axis=1).tofile(out)
            else:
                assert w.shape == (rows, width), (i, w.shape)
                for c in range(0, rows, CH):
                    w[c:c + CH].tofile(out)
            if i % 16 == 0:
                print(f"  shard {i}/{len(shards)} {time.time()-t0:.0f}s", flush=True)
    got = os.path.getsize(tmp)
    assert got == rows * len(shards) * width, (got, rows * len(shards) * width)
    os.rename(tmp, out_name)
    json.dump(meta, open(out_name + ".json", "w"), indent=1)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
