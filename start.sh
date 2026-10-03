#!/usr/bin/env bash
# ============================================================================
# start.sh — Serve Qwen3.8-Flash-Next across a 2-node DGX Spark cluster
#             with vLLM TP2+EP+MTP.
#
# Based on: https://github.com/getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark
#
# Config is split in two:
#   .env                  machine truth only — IPs, interface, IB, secrets
#   recipes/<RECIPE>.yaml serving truth — model, image, KV dtype, MTP, ports
# Default boot is RECIPE=prod (the lean local checkpoint on the vLLM 0.30
# lane, no download). RECIPE=mia boots the vendor reference recipe (stock
# nvidia checkpoint from the HF cache).
#
# Usage:
#   ./start.sh                # sync weights if needed → patch → launch
#   ./start.sh --no-download  # skip HF download (weights already cached on head)
#   ./start.sh --no-launch    # download + sync only, don't start server
#   ./start.sh --launch       # skip download/sync; apply patch + launch
#   ./start.sh --nfs          # distribute weights over NFS instead of rsync
#   ./start.sh --no-nfs       # force rsync distribution (overrides nfs_share)
#   RECIPE=mia ./start.sh     # vendor reference recipe
#   ABLIT=1 ./start.sh        # gated Keys house QSA L3-47 checkpoint (download first)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

# shellcheck source=engine/config.sh
source "$SCRIPT_DIR/engine/config.sh"
# shellcheck source=scripts/nfs-share.sh
source "$SCRIPT_DIR/scripts/nfs-share.sh"

ssh_worker() {
    local user_prefix=""
    if [[ -n "$WORKER_USER" ]]; then
        user_prefix="${WORKER_USER}@"
    fi
    ssh -o StrictHostKeyChecking=no "${user_prefix}${WORKER_IP}" "$@"
}

# 1. Download the model weights (head node) — HF recipes only; local
#    model_path recipes set do_download_default: "false".
if $DO_DOWNLOAD; then
    "$SCRIPT_DIR/download.sh" "$MODEL_ID"
fi

# shellcheck source=engine/cache.sh
source "$SCRIPT_DIR/engine/cache.sh"
# shellcheck source=engine/pair.sh
source "$SCRIPT_DIR/engine/pair.sh"
# shellcheck source=engine/verify.sh
source "$SCRIPT_DIR/engine/verify.sh"
# shellcheck source=engine/patches.sh
source "$SCRIPT_DIR/engine/patches.sh"
if $DO_LAUNCH; then
    # shellcheck source=engine/prepare.sh
    source "$SCRIPT_DIR/engine/prepare.sh"
    # shellcheck source=engine/args.sh
    source "$SCRIPT_DIR/engine/args.sh"
    # shellcheck source=engine/launch.sh
    source "$SCRIPT_DIR/engine/launch.sh"
fi

ok "Done."
