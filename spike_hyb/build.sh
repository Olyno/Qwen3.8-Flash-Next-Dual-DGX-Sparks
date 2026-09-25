#!/usr/bin/env bash
# Hybrid-FP8 spike build (CPU-only; runs next to the live eval server).
# stock copy -> FP8-dense hybrid checkpoint in ~/models/q38-hyb
set -euo pipefail
SRC=$HOME/models/q38-stock-real
DST=$HOME/models/q38-hyb
IMG=vllm/vllm-openai:qwen38-flash-next
[[ -f $SRC/model.safetensors.index.json ]] || { echo "stock copy not ready"; exit 1; }
grep -q COPY-DONE $HOME/hyb_spike/copy.log 2>/dev/null || { echo "copy still running"; exit 1; }
rm -rf "$DST"; mkdir -p "$DST"
docker run --rm --memory=1500m --memory-swap=1500m --cpus=4 --entrypoint python3 \
  -e CUDA_VISIBLE_DEVICES="" \
  -v $HOME/hyb_spike:/work:ro \
  -v $SRC:/src:ro \
  -v $DST:/dst \
  $IMG /work/make_fp8_dense_nvidia.py --src /src --dst /dst 2>&1 | tee $HOME/hyb_spike/build.log | tail -6
python3 - <<EOF
import json
q=json.load(open("$DST/hf_quant_config.json"))["quantization"]
import collections
print("merged algos:", dict(collections.Counter(v["quant_algo"] for v in q["quantized_layers"].values())))
print("exclude count:", len(q["exclude_modules"]))
EOF
