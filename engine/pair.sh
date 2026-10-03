# ---------------------------------------------------------------------------
# 3. Distribute weights to the worker.
#    Default: rsync into the worker's own storage (worker keeps a local copy).
#    NFS_SHARE=true: export the head cache over NFS (ConnectX); no worker copy.
# ---------------------------------------------------------------------------
REMOTE_HUB="${REMOTE_HF}/hub"

if [[ -n "$MODEL_PATH" ]]; then
    info "=== Step 3: Sync weights to worker ($WORKER_IP) ==="
    if ! $DO_SYNC; then
        info "  --launch: skipping sync (assuming worker copy is current)"
    else
        WORKER_SNAP_RC=2
        if ssh_worker "test -d '$WORKER_MODEL_PATH'" 2>/dev/null; then
            set +e
            ssh_worker python3 - "$WORKER_MODEL_PATH" \
                < "$SCRIPT_DIR/scripts/resolve_snapshot.py" >/dev/null
            WORKER_SNAP_RC=$?
            set -e
        fi
        if [[ "$WORKER_SNAP_RC" -eq 0 ]]; then
            ok "Worker already has a complete copy of $(basename "$HEAD_MODEL_PATH") — skipping rsync."
            info "  (delete $WORKER_MODEL_PATH on the worker to force a re-sync)"
        else
            if [[ "$WORKER_SNAP_RC" -eq 1 ]]; then
                warn "Worker copy is incomplete — re-syncing."
            fi
            info "  Worker path: $WORKER_MODEL_PATH"
            warn "  This copies the full checkpoint over the network and needs the same"
            warn "  free space on the worker."
            ssh_worker "mkdir -p '$(dirname "$WORKER_MODEL_PATH")'"
            rsync -av --progress --partial \
                "${HEAD_MODEL_PATH}/" \
                "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:${WORKER_MODEL_PATH}/"
            ok "Rsync complete."
        fi
    fi
elif [[ "$NFS_SHARE" == "true" ]]; then
    info "=== Step 3: NFS-share weights from head ==="
    nfs_ensure_server
else
    info "=== Step 3: Sync weights to worker ($WORKER_IP) ==="
    if ! $DO_SYNC; then
        info "  --launch: skipping sync (assuming worker cache is current)"
    else
        WORKER_SNAP_RC=2
        if ssh_worker "test -d '$REMOTE_HUB/models--${ORG}--${NAME}'" 2>/dev/null; then
            set +e
            ssh_worker python3 - "$REMOTE_HUB/models--${ORG}--${NAME}" \
                < "$SCRIPT_DIR/scripts/resolve_snapshot.py" >/dev/null
            WORKER_SNAP_RC=$?
            set -e
        fi
        if [[ "$WORKER_SNAP_RC" -eq 0 ]]; then
            ok "Worker already has a complete snapshot of models--${ORG}--${NAME} — skipping rsync."
            info "  (delete $REMOTE_HUB/models--${ORG}--${NAME} on the worker to force a re-sync)"
        else
            [[ -d "$HEAD_MODEL_PATH" ]] || err "HEAD ($HEAD_IP): $HEAD_MODEL_PATH — NOT FOUND. Nothing to sync; run ./download.sh first."
            if [[ "$WORKER_SNAP_RC" -eq 1 ]]; then
                warn "Worker snapshot is incomplete — re-syncing."
            fi
            info "  Worker cache: $REMOTE_HUB"
            warn "  This copies the full checkpoint over the network and needs the same"
            warn "  free space on the worker. NFS_SHARE=true avoids both — see the README."
            ssh_worker "mkdir -p '$REMOTE_HUB'"
            rsync -av --progress --partial \
                "${HEAD_MODEL_PATH}/" \
                "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:${REMOTE_HUB}/models--${ORG}--${NAME}/"
            ok "Rsync complete."
        fi
    fi
    # Same hf_hub 1.28 refs/main hazard as the head (issue #36). Only the rsync
    # path needs this: under NFS_SHARE the worker mounts the head's cache, so
    # normalising the head above already covers it. Runs whether or not the
    # sync ran, since a worker cache staged by hand is exactly the exposed case.
    # Compare byte count to stripped length. Do NOT compare against
    # "$(cat "$ref")": command substitution strips trailing newlines, so the
    # comparison is blind to exactly the byte this is meant to catch.
    # NOTE: keep this as an if-statement, not `... && warn` — as the last
    # command of a sourced file, a false `&&` list makes `source` return 1 and
    # `set -e` in start.sh kills the script silently right after Step 3.
    if ssh_worker "ref='$REMOTE_HUB/models--${ORG}--${NAME}/refs/main'
        if [ -f \"\$ref\" ]; then
            s=\$(tr -d '[:space:]' < \"\$ref\")
            n=\$(wc -c < \"\$ref\")
            if [ -n \"\$s\" ] && [ \"\$n\" -ne \"\${#s}\" ]; then
                printf '%s' \"\$s\" > \"\$ref\" && echo normalized
            fi
        fi" 2>/dev/null | grep -q normalized; then
        warn "Normalised trailing whitespace in the worker's refs/main (issue #36)"
    fi
fi
