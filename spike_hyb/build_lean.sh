#!/usr/bin/env bash
# A4: lean checkpoint -> FP8-dense hybrid combo checkpoint (CPU-only).
# Input: q38-lean-real = symlink FARM over the lean bake artifacts dir
# (path-preserving docker mounts -v P:P, because the converter resolves
# realpath and HF-farm symlinks break under plain /src mounts; config.json
# is copied from q38-stock-real - the bake artifacts dir lost that file,
# the bake never touches architecture, so stock's config is correct).
# NEVER run this while a vLLM server is booting (GB10 unified memory: two
# bulk jobs = hard hang, see spike_v30/README.md "Second crash").
set -euo pipefail
SRC=$HOME/models/q38-lean-real
LEANREAL=$HOME/Qwen38-overthinking-lab/artifacts/Qwen3.8-Flash-Next-NVFP4-lean
DST=$HOME/models/q38-lean-hyb
IMG=vllm/vllm-openai:qwen38-flash-next
[[ -f $SRC/model.safetensors.index.json ]] || { echo "lean farm not ready"; exit 1; }
rm -rf "$DST"; mkdir -p "$DST"
docker run --rm --memory=1500m --memory-swap=1500m --cpus=4 --entrypoint python3 \
  -e CUDA_VISIBLE_DEVICES="" \
  -v $HOME/hyb_spike:/work:ro \
  -v $SRC:$SRC:ro \
  -v "$LEANREAL":"$LEANREAL":ro \
  -v $DST:/dst \
  $IMG /work/make_fp8_dense_nvidia.py --src "$SRC" --dst /dst 2>&1 | tee $HOME/hyb_spike/build_lean.log | tail -8
python3 - <<"PY"
import json, collections
q = json.load(open("/home/olyno/models/q38-lean-hyb/hf_quant_config.json"))["quantization"]
c = collections.Counter(v.get("strategy") for v in q.values())
print("VERIFY-COUNTS", dict(c))
assert c.get("FP8_PER_CHANNEL_PER_TOKEN", 0) >= 500, "conversion did not apply"
print("A4-BUILD-OK")
PY
