#!/usr/bin/env python3
"""engine/recipe.py — recipes/<name>.yaml (strict 2-space subset) -> export lines.

Keys are the engine's own UPPER_SNAKE variable names, lowercased (model_id ->
MODEL_ID). Nesting: one level only.
"""
import os, sys

BOOL10 = {"true":"true","false":"false","yes":"true","no":"false","1":"true","0":"false"}
BOOL_KEYS = {"YARN_ENABLE","ENABLE_EXPERT_PARALLEL","FP8_DENSE","PLE_OFFLOAD",
             "REQUIRE_IDLE_GPU","EVICT_PAGE_CACHE","NFS_SHARE","V030",
             "SKIP_PLE_PATCH","DO_DOWNLOAD_DEFAULT","ASYNC_SCHEDULING"}

def die(msg): print(f"recipe error: {msg}", file=sys.stderr); sys.exit(2)

def parse(path):
    out, stack = {}, [(-1, {})]
    for ln, raw in enumerate(open(path), 1):
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        if "\t" in raw: die(f"{path}:{ln}: tabs")
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if line.startswith("- "): die(f"{path}:{ln}: lists not supported")
        if ": " not in line and not line.endswith(":"):
            die(f"{path}:{ln}: expected 'key: value'")
        while indent <= stack[-1][0] and len(stack) > 1: stack.pop()
        if line.endswith(":"):
            key = line[:-1].strip()
            stack.append((indent, {}, key)); out.setdefault("__parent__", []).append(key)
            continue
        k, v = line.split(": ", 1)
        parent = stack[-1][2] + "_" if len(stack) > 1 else ""
        key = (parent + k.strip()).upper()
        v = v.split(" #")[0].strip().strip('"').strip("'")
        if key in out: die(f"{path}:{ln}: duplicate key {key}")
        out[key] = v
    return out

def main():
    if len(sys.argv) != 3: die("usage: recipe.py <recipes-dir> <name>")
    d, name = sys.argv[1], sys.argv[2]
    path = os.path.join(d, name + ".yaml")
    if not os.path.isfile(path): die(f"no such recipe: {path}")
    for k, v in parse(path).items():
        if k in ("NAME", "__PARENT__"): continue
        if k in BOOL_KEYS:
            v = BOOL10.get(v.lower()) if v.lower() in BOOL10 else v
            if v is None: die(f"{k}: expected bool")
        if not k.replace("_", "").isalnum() or not k[0].isalpha():
            die(f"bad key: {k}")
        print(f"export {k}={v!r}")

main()
