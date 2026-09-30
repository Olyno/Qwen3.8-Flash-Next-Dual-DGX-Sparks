#!/usr/bin/env python3
"""r8_analyze.py — decode-step anatomy verdict from a torch-profiler chrome
trace dir. Pure python (no torch on the host): parses trace_event JSON files,
keeps GPU-kernel events (cat "kernel"/"gpu_memcpy"/"gpu_user_annotation"),
aggregates self-time by name, classifies into the lever buckets, and applies
the pre-registered 8 % arm-worthiness bar.

Usage: python3 r8_analyze.py <trace_dir>   (prints verdict lines)
"""
import json, os, re, sys, collections

BUCKETS = {
    "cublas-fallback": re.compile(r"gemv|gemm|cutlass.*gemm|Matmul|matmul|nvjet|cublas", re.I),
    "sparse-indexer":  re.compile(r"indexer|qsa_index|logits", re.I),
    "qsa-attention":   re.compile(r"flash_attn|flashattn|attn_fwd|paged|qsa", re.I),
    "moe":             re.compile(r"moe|expert|grouped", re.I),
    "mtp-draft":       re.compile(r"mtp|draft|propose", re.I),
    "ple":             re.compile(r"ngram|ple|embedding", re.I),
    "hc-mixer":        re.compile(r"hyper|gated_residual|silu|mixer", re.I),
    "copy-memset":     re.compile(r"memcpy|memset|copy_", re.I),
    "graphs":          re.compile(r"cudaGraph", re.I),
}

def main(d):
    files = []
    for root, _, fs in os.walk(d):
        for f in fs:
            if f.endswith((".json", ".json.gz", ".pt.trace.json")):
                files.append(os.path.join(root, f))
    if not files:
        print(f"NO TRACE FILES under {d} — arm failed to capture (see r8_FAIL/r8_trace_files)")
        return 1
    tot = collections.defaultdict(float)
    gpu_ms = 0.0
    n_ev = 0
    for path in files:
        try:
            raw = open(path).read()
            data = json.loads(raw)
        except Exception as e:
            print(f"skip {os.path.basename(path)}: {e}")
            continue
        evs = data.get("traceEvents", data if isinstance(data, list) else [])
        for ev in evs:
            if not isinstance(ev, dict): continue
            cat = ev.get("cat", "")
            if not re.search(r"kernel|gpu_memcpy|gpu_user_annotation|cuda_runtime", str(cat)):
                continue
            # prefer kernel-level only when both exist
            if cat == "cuda_runtime": continue
            dur = float(ev.get("dur", 0) or 0)
            if dur <= 0: continue
            name = str(ev.get("name", "?"))[:120]
            tot[name] += dur
            gpu_ms += dur / 1e3
            n_ev += 1
    print(f"gpu events={n_ev}  total gpu time={gpu_ms:.1f} ms")
    if gpu_ms <= 0:
        print("NO GPU TIME in trace (profiler captured CPU-only?)")
        return 2
    ranked = sorted(tot.items(), key=lambda kv: -kv[1])
    print("top-15 kernels:")
    for name, dur in ranked[:15]:
        print(f"  {100*dur/1e3/gpu_ms:5.1f}%  {dur/1e3:8.2f} ms  {name}")
    print("bucket shares (arm-worthiness bar: >=8%):")
    for b, rx in BUCKETS.items():
        share = sum(d for n, d in tot.items() if rx.search(n))
        pct = 100 * share / sum(tot.values())
        flag = "  << ARM-WORTHY" if pct >= 8 else ""
        print(f"  {b:>14}: {pct:5.1f}%{flag}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
