# ---------------------------------------------------------------------------
# 4. Verify weights on head (the worker is checked once its copy is in place)
# ---------------------------------------------------------------------------
info "=== Step 4: Verify weights on head ==="
if [[ -d "$HEAD_MODEL_PATH" ]]; then
    HEAD_SIZE=$(du -sh "$HEAD_MODEL_PATH" 2>/dev/null | cut -f1)
    ok "HEAD  ($HEAD_IP): $HEAD_MODEL_PATH ($HEAD_SIZE)"
else
    err "HEAD  ($HEAD_IP): $HEAD_MODEL_PATH — NOT FOUND"
fi

if [[ "$NNODES" -eq 2 ]]; then
    ssh_worker true || err "passwordless ssh to $WORKER_IP failed — set up keys first (WORKER_USER=${WORKER_USER:-$USER})"
fi

if ! $DO_LAUNCH && [[ "$NNODES" -eq 2 ]]; then
    if [[ -n "$MODEL_PATH" ]]; then
        if ssh_worker "test -d '$WORKER_MODEL_PATH'" 2>/dev/null; then
            WORKER_SIZE=$(ssh_worker "du -sh '$WORKER_MODEL_PATH' 2>/dev/null | cut -f1" || true)
            ok "WORKER ($WORKER_IP): $WORKER_MODEL_PATH (${WORKER_SIZE:-?})"
        else
            err "WORKER ($WORKER_IP): $WORKER_MODEL_PATH — NOT FOUND. Re-run without --launch to sync."
        fi
    elif [[ "$NFS_SHARE" == "true" ]]; then
        nfs_ensure_worker_volume
        if nfs_worker_has_model "hub/models--${ORG}--${NAME}"; then
            ok "WORKER ($WORKER_IP): nfs $NFS_VOLUME → hub/models--${ORG}--${NAME}"
        else
            err "WORKER cannot see hub/models--${ORG}--${NAME} over NFS. Check: docker logs $NFS_CONTAINER"
        fi
    elif ssh_worker "test -d '$REMOTE_HUB/models--${ORG}--${NAME}'" 2>/dev/null; then
        WORKER_SIZE=$(ssh_worker "du -sh '$REMOTE_HUB/models--${ORG}--${NAME}' 2>/dev/null | cut -f1" || true)
        ok "WORKER ($WORKER_IP): $REMOTE_HUB/models--${ORG}--${NAME} (${WORKER_SIZE:-?})"
    else
        err "WORKER ($WORKER_IP): $REMOTE_HUB/models--${ORG}--${NAME} — NOT FOUND. Re-run without --launch to sync."
    fi
fi

# ---------------------------------------------------------------------------
# 4b-0. Dual-node network preflight: the RoCE link interface must exist with
#       a sane MTU on both nodes before NCCL discovery depends on it.
# ---------------------------------------------------------------------------
if $DO_LAUNCH && [[ "$NNODES" -eq 2 ]]; then
    info "=== Step 4b-0: Network preflight ($IFACE ↔ $WORKER_IFACE) ==="
    ip -o link show "$IFACE" >/dev/null 2>&1 \
        || err "IFACE=$IFACE not found on head — check .env (see: ip -o link)"
    ssh_worker "ip -o link show '$WORKER_IFACE'" >/dev/null 2>&1 \
        || err "WORKER_IFACE=$WORKER_IFACE not found on $WORKER_IP — set WORKER_IFACE in .env if the nodes are cross-wired"
    HEAD_MTU=$(ip -o link show "$IFACE" | sed -n 's/.*mtu \([0-9]*\).*/\1/p')
    WORKER_MTU=$(ssh_worker "ip -o link show '$WORKER_IFACE'" | sed -n 's/.*mtu \([0-9]*\).*/\1/p')
    if [[ "$HEAD_MTU" != "$WORKER_MTU" ]]; then
        warn "MTU mismatch: head $IFACE=$HEAD_MTU vs worker $WORKER_IFACE=$WORKER_MTU — RoCE needs both ends equal."
    fi
    # RoCE over ConnectX at the stock 1500 measurably hurts; 9000 is the
    # jumbo-frame default for this link.
    if [[ "${HEAD_MTU:-0}" -lt 9000 ]]; then
        warn "MTU $HEAD_MTU < 9000 hurts RoCE on ConnectX — run on BOTH nodes: sudo ip link set <iface> mtu 9000"
    fi
    ok "Interfaces present (MTU head=$HEAD_MTU worker=$WORKER_MTU)."
fi

# ---------------------------------------------------------------------------
# 4b. Preflight: both GPUs must be free (another vLLM/SGLang tenant would OOM us
#     ten minutes into weight loading).
# ---------------------------------------------------------------------------
gpu_tenants() {  # prints "pid,name,mem" lines for compute apps, empty if idle
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed '/^$/d'
}
if $DO_LAUNCH && [[ "$REQUIRE_IDLE_GPU" == "true" ]]; then
    info "=== Step 4b: GPU preflight ==="
    HEAD_TENANTS=$(gpu_tenants || true)
    WORKER_TENANTS=""
    if [[ "$NNODES" -eq 2 ]]; then
        WORKER_TENANTS=$(ssh_worker "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed '/^\$/d'" || true)
    fi
    if [[ -n "$HEAD_TENANTS" || -n "$WORKER_TENANTS" ]]; then
        echo "HEAD:   ${HEAD_TENANTS:-idle}"
        [[ "$NNODES" -eq 2 ]] && echo "WORKER: ${WORKER_TENANTS:-idle}"
        err "GPU is in use (set REQUIRE_IDLE_GPU=false to override)."
    fi
    ok "GPU idle."
fi

# ---------------------------------------------------------------------------
# 4b-2. Release the checkpoint's own page cache (issue #35)
#
# On GB10 the page cache shares one unified-memory pool with the model. A
# checkpoint that was just written -- hf download, the rsync above, or an NFS
# read on a previous launch -- leaves its bytes resident as clean pages, and
# weight loading can then die partway with CUDA OOM on an otherwise idle box.
#
# `echo 3 > /proc/sys/vm/drop_caches` needs root, which these nodes do not have
# passwordless; that is why the README tells you to do it yourself rather than
# claiming start.sh does it. posix_fadvise(POSIX_FADV_DONTNEED) drops clean
# pages of files we can open, needs no privileges, and touches only this
# checkpoint instead of the whole system's cache.
#
# Best-effort by construction: evicting the cache is an optimisation, so a
# failure here must never block a launch that would otherwise work.
# EVICT_PAGE_CACHE=false opts out.
EVICT_PAGE_CACHE="${EVICT_PAGE_CACHE:-true}"
if $DO_LAUNCH && [[ "$EVICT_PAGE_CACHE" == "true" ]]; then
    info "=== Step 4b-2: Release checkpoint page cache ==="
    EVICT_PY="$SCRIPT_DIR/scripts/evict_page_cache.py"
    python3 "$EVICT_PY" "$HEAD_MODEL_PATH" 2>&1 | sed 's/^/  HEAD   /' || true

    # Both nodes hold their own copy, but in different places:
    #
    #   rsync mode : the worker has its own copy on local disk, so fadvise
    #                runs over ssh against that path.
    #   NFS mode   : the worker has NO local copy. Its resident pages are NFS
    #                client cache, reachable only through the mount, and that
    #                mount exists only inside a container. So the pass runs in
    #                a throwaway container holding the same volume, the way
    #                nfs_worker_has_model() already probes it. Verified on
    #                NFS 4.2: fadvise evicts there exactly as it does locally
    #                (3.00 GiB -> 0.00 GiB resident, measured with mincore).
    #
    # One copy of the script on the worker serves both modes. Skipped entirely
    # on a single node.
    if [[ "$NNODES" -eq 2 ]] && scp -q "$EVICT_PY" \
        "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:/tmp/evict_page_cache.py" 2>/dev/null
    then
        if [[ -n "$MODEL_PATH" ]]; then
            ssh_worker "python3 /tmp/evict_page_cache.py \
                '$WORKER_MODEL_PATH'" 2>&1 \
                | sed 's/^/  WORKER /' || true
        elif [[ "$NFS_SHARE" == "true" ]]; then
            ssh_worker "docker run --name vllm-fn-evict-\$\$ \
                -v '${NFS_VOLUME}:/hf:ro' \
                -v /tmp/evict_page_cache.py:/evict.py:ro \
                --entrypoint python3 '$IMAGE' \
                /evict.py '/hf/hub/models--${ORG}--${NAME}'; \
                rc=\$?; docker rm -f vllm-fn-evict-\$\$ >/dev/null 2>&1; exit \$rc" 2>&1 \
                | sed 's/^/  WORKER /' || true
        else
            ssh_worker "python3 /tmp/evict_page_cache.py \
                '$REMOTE_HUB/models--${ORG}--${NAME}'" 2>&1 \
                | sed 's/^/  WORKER /' || true
        fi
    elif [[ "$NNODES" -eq 2 ]]; then
        warn "  Could not copy the evictor to the worker; skipping its pass."
    fi
fi
