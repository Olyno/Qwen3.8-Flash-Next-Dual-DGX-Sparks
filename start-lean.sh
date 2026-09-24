#!/usr/bin/env bash
# ============================================================================
# start-lean.sh — Serve Qwen3.8-Flash-Next-NVFP4-lean on the 2-node recipe
#                 from start.sh. The overthinking-marker penalty is baked
#                 into this checkpoint's lm_head weights, so leaner reasoning
#                 comes from the weights themselves: plain `vllm serve`, no
#                 runtime logit_bias or special sampler.
#
# The quant layout matches the stock NVFP4 checkpoint (NVFP4 experts + FP8
# PLE table), so start.sh's standard PLE detection and patch path applies
# unchanged. KV cache dtype stays whatever .env owns.
#
# The checkpoint lives on the head node at
#   $HOME/models/Qwen3.8-Flash-Next-NVFP4-lean   (LEAN_MODEL_DIR overrides)
# and is never published to the Hub. This script seeds it into the head's
# HuggingFace cache as a local pseudo-repo (models--local--...) so the
# resolve/verify/worker-sync steps treat it like any cached checkpoint, and
# skips the download step.
#
# Context matches this cluster's stock serving: YaRN 4x scaled ~1M, the same
# runtime recipe as the base checkpoint (the bake only touches lm_head rows;
# context handling is identical and untested-vs-262K at 1M). For the native,
# measurement-verified window: OVERRIDE_MAX_MODEL_LEN=262144
# OVERRIDE_YARN_ENABLE=false.
#
# Thinking level is per-request via the checkpoint's chat template:
# chat_template_kwargs.reasoning_effort = "low" | "medium" | "xhigh"
# (default xhigh) and enable_thinking=true|false. See README.
#
# Single-Spark experiments use ./start-tp1.sh (separate recipe).
#
# Usage: same flags as start.sh
#   ./start-lean.sh              # seed cache, sync worker, patch, launch
#   ./start-lean.sh --no-launch  # seed + sync weights only, don't start vLLM
#   ./start-lean.sh --launch     # skip sync (worker cache current), patch, launch
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NAME="Qwen3.8-Flash-Next-NVFP4-lean"
SRC="${LEAN_MODEL_DIR:-$HOME/models/$NAME}"
REPO="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--local--$NAME"
[[ -f "$SRC/config.json" ]] || { echo "lean checkpoint not found: $SRC (set LEAN_MODEL_DIR=/path/to/ckpt)" >&2; exit 1; }
# Hardlink the tree into the cache as snapshot "lean" (same filesystem, no
# extra disk) and point refs/main at it. Re-stage only when the cache is
# missing or incomplete. refs/main is written WITHOUT a trailing newline:
# the offline resolver reads it raw (issue #36).
if ! { python3 "$SCRIPT_DIR/files/resolve_snapshot.py" "$REPO" >/dev/null 2>&1 && [[ -f "$REPO/refs/main" ]]; }; then
    mkdir -p "$REPO/snapshots/lean" "$REPO/refs"
    cp -alf "$SRC/." "$REPO/snapshots/lean/"
    printf 'lean' > "$REPO/refs/main"
fi
python3 "$SCRIPT_DIR/files/resolve_snapshot.py" "$REPO" >/dev/null \
    || { echo "staged cache incomplete: $REPO (source $SRC)" >&2; exit 1; }

# Overrides consumed by start.sh (applied after .env, like start-fp8.sh).
export OVERRIDE_MODEL_ID="${OVERRIDE_MODEL_ID:-local/$NAME}"
# Register under the stock model id: clients key capability profiles
# (e.g. the reasoning_effort picker) off the model name, so a "-lean"
# suffix hides the thinking-level options even though the checkpoint
# supports them. Override back to a distinct name if you serve both at
# once and need to tell them apart.
export OVERRIDE_SERVED_MODEL_NAME="${OVERRIDE_SERVED_MODEL_NAME:-qwen3.8-flash-next}"
# YaRN 4x scaled context — identical runtime recipe to the stock cluster
# config; weights are context-length agnostic.
export OVERRIDE_MAX_MODEL_LEN="${OVERRIDE_MAX_MODEL_LEN:-1048576}"
export OVERRIDE_YARN_ENABLE="${OVERRIDE_YARN_ENABLE:-true}"
# Pseudo-repo only ever exists on the head — never call download.sh at the Hub.
export DO_DOWNLOAD_DEFAULT=false
exec "$SCRIPT_DIR/start.sh" "$@"
