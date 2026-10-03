# ---------------------------------------------------------------------------
# 2. Resolve local cache path
# ---------------------------------------------------------------------------
info "=== Step 2: Resolve cache path ==="

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
HUB_PATH="$HF_CACHE_DIR/hub"

if [[ -n "$MODEL_PATH" ]]; then
    # Local checkpoint: no HF repo, no download. Verified directly and served
    # via bind-mount (see launch.sh).
    HEAD_MODEL_PATH="$MODEL_PATH"
    SNAP_RC=0
    python3 "$SCRIPT_DIR/scripts/resolve_snapshot.py" "$HEAD_MODEL_PATH" >/dev/null || SNAP_RC=$?
    case "$SNAP_RC" in
        0) ;;
        1) err "model_path checkpoint is incomplete (missing indexed weight shards): $HEAD_MODEL_PATH" ;;
        *) err "model_path is not a checkpoint dir (no model.safetensors.index.json): $HEAD_MODEL_PATH" ;;
    esac
    [[ -f "$HEAD_MODEL_PATH/config.json" ]] || err "model_path has no config.json: $HEAD_MODEL_PATH"
    SNAP="local"
    MODEL_DIR="$HEAD_MODEL_PATH"
    PLE_CONFIG_DIR="$HEAD_MODEL_PATH"
    ok "Model path: $HEAD_MODEL_PATH (local checkpoint, no download)"
else
ORG="${MODEL_ID%%/*}"
NAME="${MODEL_ID##*/}"
# Hub repo dir (blobs/refs/snapshots). Prefer this over `hf path`, which is not
# a real CLI command and (when it exists) often returns a snapshot subdir.
MODEL_DIR="$HUB_PATH/models--${ORG}--${NAME}"

if [[ ! -d "$MODEL_DIR" ]]; then
    local_guess=""
    for tool_cmd in "uvx hf path" "hf path" "huggingface-cli path"; do
        first_word="${tool_cmd%% *}"
        if command -v "$first_word" &>/dev/null; then
            local_guess=$($tool_cmd "$MODEL_ID" 2>/dev/null || true)
            [[ -n "$local_guess" && -d "$local_guess" ]] && break
            local_guess=""
        fi
    done
    if [[ -n "$local_guess" && -d "$local_guess" ]]; then
        # Walk up to models--org--name if the CLI pointed at a snapshot.
        case "$local_guess" in
            *"/models--${ORG}--${NAME}"*)
                MODEL_DIR="${local_guess%%/models--${ORG}--${NAME}*}/models--${ORG}--${NAME}"
                ;;
            *)
                MODEL_DIR="$local_guess"
                ;;
        esac
    fi
fi

if [[ -z "$MODEL_DIR" || ! -d "$MODEL_DIR" ]]; then
    if [[ "$ABLIT" == "1" && "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
        err "Could not resolve local cache path for $MODEL_ID under $HUB_PATH
       Fetch it first (HF_TOKEN required):
         1. Set HF_TOKEN in .env (or: export HF_TOKEN=hf_...)
         2. Open $ABLIT_PAGE
         3. Accept the terms on that page
         4. ABLIT=1 ./download.sh"
    fi
    err "Could not resolve local cache path for $MODEL_ID under $HUB_PATH"
fi
case "$MODEL_DIR" in
    "$HF_CACHE_DIR"|"$HF_CACHE_DIR"/*) ;;
    *) err "snapshot ${MODEL_DIR} is not under HF_HOME=${HF_CACHE_DIR}" ;;
esac

ORG="${MODEL_ID%%/*}"
NAME="${MODEL_ID##*/}"
HEAD_MODEL_PATH="$HUB_PATH/models--${ORG}--${NAME}"
if [[ ! -d "$HEAD_MODEL_PATH" ]]; then
    if [[ "$ABLIT" == "1" && "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
        err "Could not find HF repo cache at $HEAD_MODEL_PATH
       Fetch it first (HF_TOKEN required):
         1. Set HF_TOKEN in .env (or: export HF_TOKEN=hf_...)
         2. Open $ABLIT_PAGE
         3. Accept the terms on that page
         4. ABLIT=1 ./download.sh"
    fi
    err "Could not find HF repo cache at $HEAD_MODEL_PATH (resolved snapshot: ${MODEL_DIR:-none})"
fi

# This image ships huggingface_hub 1.28.0, whose offline branch reads
# refs/main with a bare f.read() and no .strip(). A hand-staged ref written the
# obvious way (`echo $SHA > refs/main`) carries a trailing newline, so offline
# resolution builds `snapshots/<sha>\n`, os.path.exists() fails, and vLLM dies
# at arg-parse with a repo-not-found error that names neither the file nor the
# newline. hf download writes the ref clean, so this only bites the staging
# flows this recipe advertises: NFS staging, rsync-your-own-hub-dir,
# --no-download. Normalise it here instead. Rewrites only when the bytes
# actually differ, so it is a no-op on hub-downloaded caches. Issue #36.
normalize_ref_main() {
    local ref="$1/refs/main"
    [[ -f "$ref" ]] || return 0
    local raw stripped
    raw=$(cat "$ref"; printf x); raw="${raw%x}"      # preserve trailing bytes
    stripped=$(printf '%s' "$raw" | tr -d '[:space:]')
    [[ -n "$stripped" && "$raw" != "$stripped" ]] || return 0
    printf '%s' "$stripped" > "$ref" || return 0
    warn "Normalised trailing whitespace in $ref (hf_hub 1.28 offline resolution
     reads this file without .strip(); see issue #36)"
}
normalize_ref_main "$HEAD_MODEL_PATH"

# Every shard named by the safetensors index must exist. A hub dir from an
# interrupted download is not enough — rsync would copy the hole to the worker
# and vLLM would die minutes into load.
SNAP=""
SNAP_RC=0
SNAP="$(python3 "$SCRIPT_DIR/scripts/resolve_snapshot.py" "$HEAD_MODEL_PATH")" && SNAP_RC=0 || SNAP_RC=$?
[[ -n "$SNAP" ]] || err "No snapshot under $HEAD_MODEL_PATH/snapshots"
if [[ "$SNAP_RC" -ne 0 ]]; then
    if [[ "$ABLIT" == "1" && "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
        err "Checkpoint snapshot is incomplete (missing indexed weight shards).
       Resume with (HF_TOKEN required):  ABLIT=1 ./download.sh
       Accept the terms on $ABLIT_PAGE first."
    fi
    err "Checkpoint snapshot is incomplete (missing indexed weight shards).
       Resume with:  ./download.sh $MODEL_ID"
fi
ok "Model cache: $HEAD_MODEL_PATH  (snapshot $SNAP)"

# PLE config must come from the snapshot the engine will load. refs/main
# names that revision. Directory order does not. Never guess by ls order.
# SNAP was already resolved by scripts/resolve_snapshot.py (complete shards,
# refs/main preferred).
PLE_CONFIG_DIR="$HEAD_MODEL_PATH/snapshots/$SNAP"
if [[ ! -f "$PLE_CONFIG_DIR/config.json" && -f "$MODEL_DIR/config.json" ]]; then
    PLE_CONFIG_DIR="$MODEL_DIR"
fi
if [[ ! -f "$PLE_CONFIG_DIR/config.json" ]]; then
    err "snapshot $SNAP has no config.json (partial download). Delete $PLE_CONFIG_DIR and re-run ./download.sh $MODEL_ID."
fi
fi

# Checkpoints disagree about declaring text_config.ple_embedding_dtype, which is
# what the patched ple_layer.py dispatches on (nvidia/... omits it and declares
# the FP8 PLE table only in quantization_config.config_groups). Recover it from
# the checkpoint and feed it back via --hf-overrides below. Empty = already
# declared, or no quantized PLE table.
PLE_EMBEDDING_DTYPE="${PLE_EMBEDDING_DTYPE:-}"
if [[ -z "$PLE_EMBEDDING_DTYPE" && -f "$PLE_CONFIG_DIR/config.json" ]]; then
    PLE_EMBEDDING_DTYPE=$(python3 "$SCRIPT_DIR/scripts/detect_ple_dtype.py" "$PLE_CONFIG_DIR")
fi
if [[ -n "$PLE_EMBEDDING_DTYPE" ]]; then
    ok "PLE table dtype not declared by checkpoint — overriding to $PLE_EMBEDDING_DTYPE"
fi
SNAPSHOT_SHA=$(basename "${PLE_CONFIG_DIR%/}")

# Resolve the worker's HF cache. It mirrors the head's absolute path unless that
# path lives under $HOME (then the prefix is rewritten to the worker's $HOME), or
# WORKER_HF_HOME overrides it outright.
REMOTE_HOME=$(ssh_worker "echo \"\$HOME\"")
[[ -n "$REMOTE_HOME" ]] || err "Could not resolve \$HOME on worker ($WORKER_IP). Check SSH / WORKER_USER."
if [[ -n "${WORKER_HF_HOME:-}" ]]; then
    REMOTE_HF="$WORKER_HF_HOME"
elif [[ "$HF_CACHE_DIR" == "$HOME" || "$HF_CACHE_DIR" == "$HOME/"* ]]; then
    REMOTE_HF="${REMOTE_HOME}${HF_CACHE_DIR#"$HOME"}"
else
    REMOTE_HF="$HF_CACHE_DIR"
fi
if [[ -n "$MODEL_PATH" ]]; then
    # The worker's copy of a local checkpoint always lives under ~/models/<name>.
    WORKER_MODEL_PATH="$REMOTE_HOME/models/$(basename "$HEAD_MODEL_PATH")"
    info "Worker copy: $WORKER_MODEL_PATH"
else
    info "Head HF cache:   $HF_CACHE_DIR"
    info "Worker HF cache: $REMOTE_HF"
fi
