#!/usr/bin/env python3
"""p1_analyze.py — step-floor decomposition of the P1 torch-profiler trace.

Input: chrome-trace JSON files in a dir (VLLM_TORCH_PROFILER_DIR dump, .json /
.json.gz). Answers the G2 question: of one decode step's wall time, how much is
GPU kernels, how much memcpy/memset, and how much is HOST-BOUND GAP (dispatch
latency, python, sync) — i.e. where the floor per token actually is, and how
many CUDA graph launches vs eager kernels we pay.

Usage: p1_analyze.py <prof_dir> [top_n]
Emits a fixed-shape report (grep-stable): FLOOR-PerStep, GPU-Busy %,
TOPKERNEL rows, GRAPH-LAUNCHES, GAP-MASS, VERDICT line.
"""
import gzip
import json
import sys
from collections import defaultdict

PROF = sys.argv[1].rstrip("/")
TOPN = int(sys.argv[2]) if len(sys.argv) > 2 else 15


def load_events(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    return d["traceEvents"] if isinstance(d, dict) else d


def main():
    import glob
    import os
    files = []
    for pat in ("**/*.json", "**/*.json.gz"):
        files += [p for p in glob.glob(os.path.join(PROF, pat), recursive=True)
                  if "metadata" not in p]
    if not files:
        sys.exit(f"NO-TRACE-FILES under {PROF}")
    ev = []
    for p in files:
        try:
            ev += load_events(p)
        except Exception as e:
            print(f"WARN skip {p}: {type(e).__name__} {e}")
    # phases: cat=cpu/gpu kernel rows have dur; markers are instant
    kernels = [e for e in ev if e.get("ph") == "X" and e.get("dur")]
    gpu = [e for e in kernels if e.get("cat") in ("kernel", "gpu_kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation")]
    cpu = [e for e in kernels if e.get("cat") in ("cuda_runtime", "cuda_driver", "cpu_op", "user_annotation", "overhead")]
    t0 = min((e["ts"] for e in kernels), default=0)
    t1 = max((e["ts"] + e.get("dur", 0) for e in kernels), default=0)
    span_ms = (t1 - t0) / 1000.0
    if span_ms <= 0:
        sys.exit("EMPTY-TRACE")

    # decode-step count: FULL graphs = one cudaGraphLaunch per decode step
    # (most reliable); explicit step annotations next; sampler ops only as a
    # last resort (they can fire several times per step when graphs partial).
    def count(name_sub, pool):
        return sum(1 for e in pool if name_sub in e.get("name", "").lower())
    graph_launches = count("cudagraphlaunch", cpu) + count("graph replay", cpu)
    step_markers = count("step", [e for e in ev if e.get("cat") == "user_annotation" or e.get("ph") == "i"])
    sampler_ops = count("argmax", cpu) + count("sample", cpu)  # host-side only:
    # in-graph sampler kernels named argmax MUST NOT count (they'd 34x the step count)
    n_steps = graph_launches or step_markers or sampler_ops or 1

    kname = defaultdict(lambda: [0, 0.0])
    memcpy_ms = memset_ms = 0.0
    for e in gpu:
        c = e.get("cat", "")
        if "memcpy" in c:
            memcpy_ms += e["dur"] / 1000.0
            continue
        if "memset" in c:
            memset_ms += e["dur"] / 1000.0
            continue
        k = kname[e["name"][:90]]
        k[0] += 1
        k[1] += e["dur"] / 1000.0
    busy_ms = sum(v[1] for v in kname.values()) + memcpy_ms + memset_ms

    # host-bound gap mass: merge GPU intervals, diff against span
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in gpu)
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    gap_ms = span_ms - sum(b - a for a, b in merged) / 1000.0

    print(f"P1 span_ms {span_ms:.0f}  steps_est {n_steps}  graph_launches {graph_launches}")
    print(f"FLOOR-PerStep ms {1000*span_ms/n_steps/1000:.3f}".replace("ms ms", "ms"))
    print(f"GPU-Busy % {100*busy_ms/span_ms:.1f}  (kernels {busy_ms-memcpy_ms-memset_ms:.1f} ms, "
          f"memcpy {memcpy_ms:.2f}, memset {memset_ms:.2f})")
    print(f"GAP-MASS ms {gap_ms:.1f} = host-bound/launch-bound "
          f"({100*gap_ms/span_ms:.1f} % of span)")
    print(f"TOPKERNEL rows (name | n | ms | ms/step):")
    for name, (n, ms) in sorted(kname.items(), key=lambda kv: -kv[1][1])[:TOPN]:
        print(f"  {ms:8.2f} {n:7d} {ms/n_steps:6.3f}  {name}")
    # verdict heuristic: which lever moves the floor
    kern_ms = busy_ms - memcpy_ms - memset_ms
    if gap_ms > 0.5 * span_ms:
        v = ("HOST-BOUND: >50% of step is not GPU work -> CUDA-graph coverage / "
             "launch-count reduction (cudagraph_mode, fewer eager kernels) is "
             "the dominant lever; kernel speedups cannot fix it")
    elif kern_ms / max(n_steps, 1) > 6.0:
        v = ("GPU-BOUND: step kernel time >6 ms -> weight-read floor; levers are "
             "acceptance (spec depth = free tokens/step) and quant/dense "
             "bandwidth (hybrid class), not scheduling")
    else:
        v = ("BALANCED near floor: per-step GPU time already low; remaining "
             "ceiling is bytes-per-token; check TOPKERNEL for unexpected mass")
    print("VERDICT:", v)


if __name__ == "__main__":
    main()
