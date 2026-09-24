#!/usr/bin/env python3
"""bake_lm_head.py — bake the arXiv 2606.00206 overthinking logit penalty into lm_head.weight.

Run INSIDE the vllm image container (torch needed):
  docker run --rm -v ~/Qwen38-overthinking-lab:/lab \
    -v ~/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/<sha>:/src:ro \
    --entrypoint python3 vllm/vllm-openai:qwen38-flash-next /lab/scripts/bake_lm_head.py \
    --snapshot /src --hs-dir /lab/calib/hs --token-map /lab/artifacts/overthinking_token_map.json \
    --lam 2.0 --out /lab/artifacts/baked_model

Method: vLLM ParallelLMHead here has bias=False (verified), so z_v = h . w_v. The paper's
penalty subtracts lambda from marker logits at every step. Minimum-norm row delta with exact
mean penalty over real decode hidden states h:
    d = -lambda * (G + t*eps*I)^{-1} m,  G = E[h h^T], m = E[h],  t = mean(diag G)
Shard surgery: header bytes copied VERBATIM; payload copied VERBATIM except the 49 marker
rows of lm_head.weight, which are rewritten (W+ d in f32, cast back to bf16). Nothing else
in the file moves. generation_config.json gains logit_bias {id: -lam} (exact for HF
Transformers.generate; vLLM ignores that key — verified whitelist — no double penalty).
"""
import argparse, hashlib, json, os, struct, sys

G = 2**30

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""): h.update(c)
    return h.hexdigest()

def read_st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hb = f.read(n)
    return json.loads(hb), 8 + n, hb

def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, help="source checkpoint dir (resolved snapshot)")
    ap.add_argument("--hs-dir", required=True, help="dir of harvested decode hidden-state .pt shards")
    ap.add_argument("--token-map", required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default=None, help="override lm_head shard filename (default: resolve from index)")
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--max-rows", type=int, default=800000)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tm = json.load(open(a.token_map))
    ids = sorted({int(v) for v in tm["ids"].values()})
    snap = os.path.abspath(a.snapshot)
    ipath = os.path.join(snap, "model.safetensors.index.json")
    if a.shard:
        shard = a.shard
    elif os.path.exists(ipath):
        shard = json.load(open(ipath))["weight_map"]["lm_head.weight"]
    else:
        import struct as _st
        shard = None
        for fn in sorted(os.listdir(snap)):
            if not fn.endswith(".safetensors"): continue
            with open(os.path.realpath(os.path.join(snap, fn)), "rb") as fh:
                n = _st.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            if "lm_head.weight" in hdr:
                shard = fn; break
        if shard is None:
            sys.exit("no index file and no shard contains lm_head.weight")
    src = os.path.realpath(os.path.join(snap, shard))
    hdr, data_off, hdr_bytes = read_st_header(src)
    ent = hdr["lm_head.weight"]
    V, H = ent["shape"]
    assert ent["dtype"] == "BF16", ent["dtype"]
    lo, hi = ent["data_offsets"]
    ROW = H * 2
    assert (hi - lo) == V * ROW
    print(f"shard {shard}: lm_head [{V},{H}] bf16 at payload [{lo},{hi}), file {os.path.getsize(src)/G:.2f} GiB")

    # ---- calibration stats (streamed)
    files = sorted(f for f in os.listdir(a.hs_dir) if f.endswith(".pt"))
    if not files: sys.exit("no .pt calibration shards in --hs-dir")
    m = torch.zeros(H, dtype=torch.float64)
    Gm = torch.zeros(H, H, dtype=torch.float64)
    n = 0
    for fn in files:
        t = torch.load(os.path.join(a.hs_dir, fn), map_location="cpu", weights_only=True)
        t = t.to(torch.float64).reshape(-1, H)
        m += t.sum(0); Gm += t.T @ t; n += t.shape[0]
        print(f"  hs {fn}: +{t.shape[0]} rows (total {n})")
        if n >= a.max_rows: break
    m /= n; Gm /= n
    t_reg = float(torch.diagonal(Gm).mean()) * a.ridge
    Gm += torch.eye(H, dtype=torch.float64) * t_reg
    Gi_m = torch.linalg.solve(Gm, m)
    denom = float(m @ Gi_m)
    d = (-a.lam / denom) * Gi_m
    print(f"calib n={n}  m.Gi.m={denom:.4f}  ||d||={float(d.norm()):.3e}")
    # achieved penalty on f64 d, and its spread over calibration rows
    pens = []
    for fn in files:
        t = torch.load(os.path.join(a.hs_dir, fn), map_location="cpu", weights_only=True)
        pens.append(t.to(torch.float64).reshape(-1, H) @ d)
        if sum(p.shape[0] for p in pens) >= n: break
    pen = torch.cat(pens, 0)
    print(f"penalty(f64 d): mean {pen.mean():+.4f} (target {-a.lam}) std {pen.std():.4f}")
    if a.dry_run:
        print("dry-run — nothing written"); return

    # ---- byte surgery on the edited shard
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, shard)
    idset = set(ids)
    d_f32 = d.to(torch.float32)
    n_changed = 0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        head = fi.read(data_off)          # 8B length + header bytes, verbatim
        assert head[8:] == hdr_bytes
        fo.write(head)
        # payload before lm_head
        pre = data_off + lo
        fi.seek(data_off)
        rest = pre - data_off
        while rest > 0:
            b = fi.read(min(rest, 1 << 22)); fo.write(b); rest -= len(b)
        # lm_head rows
        for r in range(V):
            raw = fi.read(ROW)
            if r in idset:
                w = torch.frombuffer(bytearray(raw), dtype=torch.uint16).view(torch.bfloat16).float()
                nb = (w + d_f32).to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
                assert len(nb) == ROW
                if nb != raw: n_changed += 1
                fo.write(nb)
            else:
                fo.write(raw)
        # tail after lm_head
        while True:
            b = fi.read(1 << 22)
            if not b: break
            fo.write(b)
    assert os.path.getsize(dst) == os.path.getsize(src), "size drift"
    assert n_changed == len(ids), f"only {n_changed}/{len(ids)} marker rows changed"
    print(f"wrote {dst}: {n_changed} marker rows mutated, {os.path.getsize(dst)/G:.2f} GiB")

    # ---- achieved penalty on the BF16-truthful baked rows (re-measure against real deltas)
    with open(dst, "rb") as f:
        f.seek(data_off + lo + ids[0] * ROW)
        r0 = torch.frombuffer(bytearray(f.read(ROW)), dtype=torch.uint16).view(torch.bfloat16).float()
    with open(src, "rb") as f:
        f.seek(data_off + lo + ids[0] * ROW)
        o0 = torch.frombuffer(bytearray(f.read(ROW)), dtype=torch.uint16).view(torch.bfloat16).float()
    db = (r0 - o0).double()
    hs0 = torch.load(os.path.join(a.hs_dir, files[0]), map_location="cpu", weights_only=True)
    hs0 = hs0.to(torch.float64).reshape(-1, H)[:5000]
    real = hs0 @ db
    print(f"bf16-baked penalty (first-shard 5k-row sample): mean {real.mean():+.4f} std {real.std():.4f}")

    # ---- copy everything else verbatim
    manifest = {shard: sha256(dst)}
    edited_json = {"generation_config.json"}
    for fn in sorted(os.listdir(snap)):
        p = os.path.realpath(os.path.join(snap, fn))
        if not os.path.isfile(p): continue
        if fn == shard or fn in edited_json: continue
        dpath = os.path.join(a.out, fn)
        with open(p, "rb") as fi, open(dpath, "wb") as fo:
            for c in iter(lambda: fi.read(1 << 22), b""): fo.write(c)
        manifest[fn] = sha256(dpath)
        print(f"copied {fn} ({os.path.getsize(dpath)/G:.2f} GiB)")

    # ---- generation_config.json: exact logit_bias for HF stacks (vLLM ignores the key)
    gc = json.load(open(os.path.join(snap, "generation_config.json")))
    gc["logit_bias"] = {str(v): -a.lam for v in ids}
    gc["baked_overthinking_penalty"] = {
        "lambda": a.lam, "paper": "arXiv:2606.00206",
        "method": "min-norm ridge LS weight-bake of decode hidden states",
        "n_markers": len(ids), "calib_rows": n, "ridge_rel": a.ridge,
        "penalty_std_over_calib": round(float(pen.std()), 4),
        "source_snapshot": os.path.basename(snap)}
    gcp = os.path.join(a.out, "generation_config.json")
    json.dump(gc, open(gcp, "w"), indent=2)
    manifest["generation_config.json"] = sha256(gcp)
    json.dump(manifest, open(os.path.join(a.out, "SHA256MANIFEST.json"), "w"), indent=2)
    # sidecar for verify + provenance
    json.dump({"lambda": a.lam, "ids": ids, "shard": shard, "calib_rows": n,
               "penalty_std_bf16": round(float(real.std()), 4)},
              open(os.path.join(a.out, "overthinking_bake.json"), "w"), indent=2)
    print("DONE ->", a.out)

if __name__ == "__main__":
    main()
