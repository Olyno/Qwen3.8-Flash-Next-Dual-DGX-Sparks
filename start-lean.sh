#!/usr/bin/env bash
# ============================================================================
# start-lean.sh — serve Qwen3.8-Flash-Next-NVFP4-lean: the overthinking-baked
#                 checkpoint (marker-token penalty baked into lm_head rows).
#
# Drop-in: plain `vllm serve <dir>`, no logit_bias, no special sampler — the
# leaner reasoning comes out of the weights.
#
# Usage:
#   ./start-lean.sh                      # default baked path (below)
#   MODEL_SOURCE=/path/to/ckpt ./start-lean.sh
# All start-tp1.sh flags/env still apply (PORT=..., --no-launch, ...).
#
# Defaults below are the values validated during the lean bake benchmarks.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_SOURCE="${MODEL_SOURCE:-$HOME/Qwen38-overthinking-lab/artifacts/Qwen3.8-Flash-Next-NVFP4-lean}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next-lean}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
[[ -f "$MODEL_SOURCE/config.json" ]] || { echo "model dir not found: $MODEL_SOURCE (pass MODEL_SOURCE=...)"; exit 1; }
exec "$HERE/start-tp1.sh" "$@"
