#!/usr/bin/env python3
"""patch_ple_offload_fp8_packed.py — post-process patch_ple_offload.py for FP8 PLE tables.

The root patch_ple_offload.py attaches the packed mmap as uint8, which is exactly the
NVFP4 row layout (codes + fp8 scales as bytes). For the nvidia checkpoint the PLE is
FP8 (float8_e4m3fn, width == embedding head dim), and the worker's offload lookup does
    torch.index_select(packed_table, 0, ids, out=fp8_buffer)
in the embedding's own dtype. A raw uint8 table would raise a dtype mismatch.

Fix applied here: after building the memmap, view the table into the weight's dtype when
the weight is float8_e4m3fn (zero-copy, same itemsize). Anchors are the exact string
lines emitted by the root patch script — if upstream moves them, this script fails loudly.
"""
import sys

P = sys.argv[1] if len(sys.argv) > 1 else \
   "/home/olyno/Qwen38-overthinking-lab/serve/files/patch_ple_offload.py"
src = open(P).read()

OLD = ('"        mm = np.memmap(path, dtype=np.uint8, mode=\\"r\\", shape=(rows, width))\\n"\n'
       '        "        table = torch.from_numpy(mm)  # zero-copy, file-backed, evictable\\n"')
NEW = ('"        mm = np.memmap(path, dtype=np.uint8, mode=\\"r\\", shape=(rows, width))\\n"\n'
       '        "        table = torch.from_numpy(mm)  # zero-copy, file-backed, evictable\\n"\n'
       '        "        wdt = getattr(emb.weight, \\"dtype\\", None)\\n"\n'
       '        "        if wdt == torch.float8_e4m3fn and table.dtype == torch.uint8:\\n"\n'
       '        "            table = table.view(torch.float8_e4m3fn)  # FP8 PLE (nvidia)\\n"')

if NEW.split('"        wdt')[1][:60] in src:
    print("already patched"); sys.exit(0)
n = src.count(OLD)
if n != 1:
    sys.exit(f"anchor count {n} != 1 — upstream changed, refusing")
src = src.replace(OLD, NEW, 1)
open(P, "w").write(src)
print("patched patch_ple_offload.py: FP8 packed table dtype view")
