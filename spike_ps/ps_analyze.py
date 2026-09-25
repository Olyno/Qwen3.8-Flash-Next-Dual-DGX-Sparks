#!/usr/bin/env python3
"""ps_analyze.py — fold ps_bench pass files into one speed table (tok/s per config).

Usage: python3 ps_analyze.py results_dir   (reads *_pass1.txt *_pass2.txt)
Rows = (context, content). Columns = tags (k10, k8, k6). Values = mean of passes.
"""
import sys, re, glob, os
from collections import defaultdict

def parse(path):
    cell = {}
    for line in open(path):
        m = re.match(r"\s*([\d,]+)\s+([\d.]+)\s+(\w+)\s+([\d,]+)\s+([\d,]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            ctx = int(m.group(1).replace(",", ""))
            task = m.group(3)
            dec = float(m.group(7))
            cell[(ctx, task)] = dec
    return cell

def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/ps_spike/results")
    vals = defaultdict(list)
    tags = set()
    for f in sorted(glob.glob(os.path.join(d, "*_pass*.txt"))):
        base = os.path.basename(f)
        tag = base.rsplit("_pass", 1)[0]
        tags.add(tag)
        for k, v in parse(f).items():
            vals[(tag, k)].append(v)
    tags = sorted(tags)
    keys = sorted({k for (_, k) in vals})
    w = max(len(t) for t in tags) + 2
    print(f"{'ctx':>8} {'content':<8}", " ".join(f"{t:>{w}}" for t in tags))
    med = {}
    for k in keys:
        row = []
        for t in tags:
            vs = vals.get((t, k), [])
            row.append(sum(vs) / len(vs) if vs else float("nan"))
        med[k] = row
        print(f"{k[0]:>8,} {k[1]:<8}", " ".join(f"{v:>{w}.1f}" for v in row))
    # gain vs reference tag (first sorted: k10 expected)
    ref = tags.index(sorted(tags)[0])
    print("\nrelative to", sorted(tags)[0] + ":")
    for k in keys:
        base = med[k][ref]
        print(f"{k[0]:>8,} {k[1]:<8}", " ".join(f"{(v/base-1)*100:>+{w}.1f}%" if v == v else " " * w for v in med[k]))

main()
