#!/usr/bin/env bash
# tests/recipe_lint.sh — validate recipes/*.conf as DATA: known keys only,
# hard pins present, no logic. Exits non-zero with every violation listed.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"

# Keys the engine/start.sh actually consume. A typo'd key would silently do
# nothing at launch — that is exactly the bug class this lint kills.
ALLOWED="IMAGE IMAGE_HINT MODEL_ID CKPT_SHA256 CONTAINER PORT MASTER_PORT
PLE_GIB MTP_WEIGHTS_GIB KV_CACHE_DTYPE MAX_MODEL_LEN MTP_NUM_SPECULATIVE_TOKENS
PLE_MODE MEMWATCH_MIN_GIB MEMWATCH_MIN_FREE_GIB MEMWATCH_FREE_GATE_GIB
MEMWATCH_GRACE VLLM_ARGS HF_OVERRIDES COMPILATION_CONFIG SPECULATIVE_CONFIG
NUM_EXPERTS_PER_TOK PLE_EMBEDDING_DTYPE EXTRA_VLLM_ARGS DOCKER_ARGS_EXTRA
RECIPE SERVED_MODEL_NAME PLE_CACHE_ID FABRIC_IFNAME IB_HCA IB_GID_INDEX
HOST_RESERVE_GIB KV_TARGET_GIB OVERHEAD_GIB HOST_SLACK_GIB OS_RESERVE_GIB
GPU_MEMORY_UTILIZATION HF_CACHE_DIR QUANT_PREFLIGHT NATIVE_MAX_MODEL_LEN
MODEL_PATH FP8DENSE HEAD_IP WORKER_IP WORKER_USER RUN_WORKER SYNC_WEIGHTS
IFACE VLLM_ALLOW_LONG_MAX_MODEL_LEN YARN YARN_FACTOR MTP_DRAFT_VOCAB VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"

fail=0; nviol=0
report() { echo "FAIL: $*"; fail=1; nviol=$((nviol + 1)); }
# ALLOWED is written multi-line for readability; membership tests need spaces.
ALLOWED=" ${ALLOWED//$'\n'/ } "
GETVAL() { sed -n "s/^$1=\"\{0,1\}\([^\" #]*\).*/\1/p" "$2" | tail -1; }

shopt -s nullglob
for f in "$REPO"/recipes/*.conf; do
    base="$(basename "$f")"
    # Data-only: no command substitution, no backticks, no assignment operators
    # beyond plain =.
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
        [[ "$line" =~ \$\( || "$line" == *'`'* ]] && report "$base: command substitution in data: $line"
        [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*[+?:]?= ]] || report "$base: not a KEY=VALUE line: $line"
        key="${line%%=*}"
        [[ " $ALLOWED " == *" $key "* ]] || report "$base: unknown key '$key'"
    done < <(grep -v '^\s*:' "$f" 2>/dev/null || true)

    # Required pins. GETVAL stops at a quote, space, or inline comment.
    image=$(GETVAL IMAGE "$f")
    sha=$(GETVAL CKPT_SHA256 "$f")
    ctr=$(GETVAL CONTAINER "$f")
    [[ "$image" =~ ^[A-Za-z0-9._/-]+@sha256:[0-9a-f]{64}$ ]] \
        || report "$base: IMAGE must be name@sha256:<64 hex> (floating tags hid five crash triages); got '$image'"
    [[ "$sha" =~ ^[0-9a-f]{64}$ ]] \
        || report "$base: CKPT_SHA256 must be 64 hex; got '${sha:-<missing>}'"
    [[ "$ctr" =~ ^vllm-fn(-[A-Za-z0-9_.-]+)?$ ]] \
        || report "$base: CONTAINER must be vllm-fn[-<suffix>]; got '${ctr:-<missing>}'"
    # PLE_MODE values engine/ple.sh understands (packed is DEAD: hang #8).
    pm=$(GETVAL PLE_MODE "$f")
    [[ -z "$pm" || "$pm" == mmap || "$pm" == pinned || "$pm" == off ]] \
        || report "$base: PLE_MODE '$pm' not in mmap|pinned|off"
done

# peers.conf.example is documentation (all-commented); it must exist.
[[ -f "$REPO/recipes/peers.conf.example" ]] || report "recipes/peers.conf.example missing"

if (( fail )); then
    echo "recipe_lint: $nviol violation(s) above"; exit 1
fi
echo "recipe_lint: PASS ($(ls "$REPO"/recipes/*.conf | wc -l) recipes pinned)"
