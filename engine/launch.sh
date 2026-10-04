    # Deadline for the /health poll at the end of this file; overridable via env.
    READY_TIMEOUT_S="${READY_TIMEOUT_S:-3600}"

    # Head-side mounts (the worker block below computes its own variants over
    # ssh; on a single node these are the only ones that exist).
    HEAD_MODEL_MOUNT=""
    [[ -n "$MODEL_PATH" ]] && HEAD_MODEL_MOUNT="-v $HEAD_MODEL_PATH:/model:ro"
    HEAD_OVERLAY_MOUNTS="${OVERLAY_MOUNTS[*]:-}"
    HEAD_CHAT_MOUNT=""
    [[ -n "$CHAT_TEMPLATE" ]] && HEAD_CHAT_MOUNT="-v $CHAT_TEMPLATE:/chat_template.jinja:ro"

    # PLE offload env. v0.30 defaults VLLM_PLE_CPU_OFFLOAD to True, which parks
    # the table in pinned anonymous host memory (~32 GiB/node) — so the flag is
    # always explicit, never inherited. On the v030 lane with ple_offload the
    # mmap overlay (patches/v030_ple) reroutes that class to a file-backed
    # table via VLLM_PLE_MMAP_DIR in OVERLAY_ENV; leave the flag at its default
    # there so the pinned-host class the patch intercepts stays selected.
    if [[ "$V030" == "true" && "$PLE_OFFLOAD" == "true" ]]; then
        PLE_OFFLOAD_ENV=""
    elif [[ "$PLE_OFFLOAD" == "true" ]]; then
        PLE_OFFLOAD_ENV="-e VLLM_PLE_CPU_OFFLOAD=1"
    else
        PLE_OFFLOAD_ENV="-e VLLM_PLE_CPU_OFFLOAD=0"
    fi

    # Inter-node NCCL/IB env for the head container. Single node is TP=1: no
    # cross-node traffic, so none of this is needed (and IFACE may be unset).
    # NCCL_MAX_NCHANNELS=4: with no GPUDirect RDMA on the GB10 every large
    # message is copied through host memory; 64 channels (the default) only add
    # host copies — 4 measured +10 % tok/s at 32 streams, +7 % at 64 in the
    # vendor dual-Spark recipe. Dual-node only by construction.
    HEAD_NET_ENV=""
    if [[ "$NNODES" -eq 2 ]]; then
        HEAD_NET_ENV="-e GLOO_SOCKET_IFNAME=$IFACE -e NCCL_SOCKET_IFNAME=$IFACE -e TP_SOCKET_IFNAME=$IFACE -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=$IB_HCA -e NCCL_IB_GID_INDEX=$IB_GID_INDEX -e NCCL_IB_AUTO_DETECT=0 -e NCCL_MAX_NCHANNELS=4"
    fi

    # Optional cpuset pinning (recipe key cpuset); same layout on both nodes.
    CPUSET_ARG=""
    [[ -n "$CPUSET" ]] && CPUSET_ARG="--cpuset-cpus $CPUSET"

    # Allocator tuning, quality-neutral and vendor-validated on this hardware.
    # expandable_segments cuts CUDA-allocator fragmentation near the pool cap
    # (sfxnz dual-Spark recipe, every evidence boot). The glibc thresholds make
    # the loader return big freed blocks to the OS at once instead of keeping
    # them resident — and eventually swapped (myllmbox single-Spark recipe).
    ALLOC_ENV="-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e MALLOC_MMAP_THRESHOLD_=65536 -e MALLOC_TRIM_THRESHOLD_=131072"

    # ---- Worker (rank 1) ----
    if [[ "$NNODES" -eq 2 ]]; then
    info "--- Launching worker (rank 1) on $WORKER_IP ---"
    ssh_worker "docker rm -f vllm-fn >/dev/null 2>&1 || true"
    ssh_worker "mkdir -p '$REMOTE_HF' ~/.cache/vllm"
    WORKER_MODEL_MOUNT=""
    # --launch skips the sync, so test -d is not enough: probe completeness
    # (indexed shards present) exactly like engine/pair.sh does before syncing.
    worker_snapshot_ok() {  # worker_snapshot_ok <remote checkpoint dir>
        local rc=2
        if ssh_worker "test -d '$1'" 2>/dev/null; then
            set +e
            ssh_worker python3 - "$1" \
                < "$SCRIPT_DIR/scripts/resolve_snapshot.py" >/dev/null
            rc=$?
            set -e
        fi
        return "$rc"
    }
    if [[ -n "$MODEL_PATH" ]]; then
        if ! worker_snapshot_ok "$WORKER_MODEL_PATH"; then
            err "WORKER copy of $WORKER_MODEL_PATH is missing or incomplete. Re-run ./start.sh without --launch to sync."
        fi
        ok "Worker has a complete checkpoint copy"
        WORKER_HF_MOUNT="-v $REMOTE_HF:/root/.cache/huggingface"
        WORKER_MODEL_MOUNT="-v $WORKER_MODEL_PATH:/model:ro"
    elif [[ "$NFS_SHARE" == "true" ]]; then
        nfs_ensure_worker_volume recreate
        if nfs_worker_has_model "hub/models--${ORG}--${NAME}"; then
            ok "Worker sees checkpoint over NFS"
        else
            err "WORKER cannot see hub/models--${ORG}--${NAME} over NFS. Check: docker logs $NFS_CONTAINER"
        fi
        WORKER_HF_MOUNT="-v $NFS_VOLUME:/root/.cache/huggingface:ro"
    else
        if ! worker_snapshot_ok "$REMOTE_HUB/models--${ORG}--${NAME}"; then
            err "WORKER snapshot $REMOTE_HUB/models--${ORG}--${NAME} is missing or incomplete. Re-run ./start.sh without --launch to sync, or use --nfs."
        fi
        ok "Worker has a complete checkpoint copy"
        WORKER_HF_MOUNT="-v $REMOTE_HF:/root/.cache/huggingface"
    fi

    # Worker can't mount head's filesystem — copy the patched file over
    if [[ -n "$WORKER_PLE_MOUNT" ]]; then
        info "  Copying PLE patch to worker..."
        PLE_DEST="/tmp/ple_layer_patched.py"
        scp -q "$PATCHED_PLE" "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:${PLE_DEST}"
    fi
    if [[ -n "$WORKER_MODELOPT_MOUNT" ]]; then
        info "  Copying MXFP8 fallback patch to worker..."
        scp -q "$PATCHED_MODELOPT" "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:/tmp/modelopt_patched.py"
    fi
    # Overlay files likewise: copy to /tmp/vllm-overlay on the worker
    WORKER_OVERLAY_MOUNTS=""
    if [[ ${#OVERLAY_FILES[@]} -gt 0 ]]; then
        info "  Copying ${#OVERLAY_FILES[@]} overlay file(s) to worker..."
        ssh_worker "mkdir -p /tmp/vllm-overlay"
        for entry in "${OVERLAY_FILES[@]}"; do
            host_file="${entry%%|*}"; container_path="${entry##*|}"
            scp -q "$host_file" "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:/tmp/vllm-overlay/$(basename "$host_file")"
            WORKER_OVERLAY_MOUNTS+=" -v /tmp/vllm-overlay/$(basename "$host_file"):$container_path:ro"
        done
    fi

    # Chat template: the worker runs the same vllm CLI and validates
    # --chat-template even when --headless, so copy it over like the overlays.
    WORKER_CHAT_MOUNT=""
    if [[ -n "$CHAT_TEMPLATE" ]]; then
        scp -q "$CHAT_TEMPLATE" "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:/tmp/chat_template.jinja"
        WORKER_CHAT_MOUNT="-v /tmp/chat_template.jinja:/chat_template.jinja:ro"
    fi

    # Write worker launch script to a temp file and scp it (avoids SSH JSON quoting issues)
    WORKER_SCRIPT=$(mktemp /tmp/vllm_worker_XXXXXX.sh)
    cat > "$WORKER_SCRIPT" <<LAUNCH_EOF
#!/bin/bash
docker run \
    -d --name vllm-fn \
    --gpus all --network host --ipc host \
    --log-opt max-size=50m --log-opt max-file=3 \
    --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 \
    --device /dev/infiniband:/dev/infiniband \
    $CPUSET_ARG \
    $ALLOC_ENV \
    -e GLOO_SOCKET_IFNAME=$WORKER_IFACE \
    -e NCCL_SOCKET_IFNAME=$WORKER_IFACE \
    -e TP_SOCKET_IFNAME=$WORKER_IFACE \
    -e NCCL_IB_DISABLE=0 \
    -e NCCL_IB_HCA=$WORKER_IB_HCA \
    -e NCCL_IB_GID_INDEX=$IB_GID_INDEX \
    -e NCCL_IB_AUTO_DETECT=0 \
    -e NCCL_MAX_NCHANNELS=4 \
    -e NCCL_DEBUG=WARN \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_HOST_IP=$WORKER_IP \
    ${VLLM_ALLOW_LONG_MAX_MODEL_LEN:+-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=$VLLM_ALLOW_LONG_MAX_MODEL_LEN} \
    $PLE_OFFLOAD_ENV \
    -e HF_HOME=/root/.cache/huggingface \
    $WORKER_PLE_MOUNT \
    $WORKER_MODELOPT_MOUNT \
    $WORKER_OVERLAY_MOUNTS \
    $WORKER_CHAT_MOUNT \
    $OVERLAY_ENV_STR \
    $WORKER_HF_MOUNT \
    $WORKER_MODEL_MOUNT \
    -v $REMOTE_HOME/.cache/vllm:/root/.cache/vllm \
    $IMAGE \
    $MODEL_ARG \
    $VLLM_ARGS_STR \
    --node-rank 1 \
    --headless
LAUNCH_EOF
    # No chmod here on purpose: mktemp already creates the file 0600. What used
    # to be `chmod +x` *loosened* that to 0711 under every umask below 077, and
    # the +x bit was never needed — the script is run as `bash <file>`, which
    # works fine on mode 0600. Matters as soon as EXTRA_VLLM_ARGS carries
    # something like `--api-key <key>` and the rendered script holds a secret.
    #
    # Feed it to the worker on stdin instead of scp'ing it to the fixed path
    # /tmp/vllm_worker_launch.sh: nothing is left on the worker to leak or to
    # clean up, there is no predictable /tmp name to pre-create as a symlink,
    # and ssh still reports the remote exit status, so `set -e` fails fast when
    # the worker's `docker run` fails instead of hanging in the health loop.
    info "  (starting worker container...)"
    worker_rc=0
    ssh_worker "bash -s" < "$WORKER_SCRIPT" || worker_rc=$?
    rm -f "$WORKER_SCRIPT"
    if (( worker_rc != 0 )); then
        err "Worker container failed to start (exit $worker_rc) — not launching the head."
    fi
    ok "Worker container started."
    info "  Waiting 15s for worker to initialize..."
    sleep 15
    # `docker run -d` succeeding only means the container spawned — a worker
    # that crashes in its first seconds (bad mount, missing checkpoint) would
    # otherwise surface as a head-side NCCL hang.
    if ! ssh_worker "docker ps --format '{{.Names}}'" 2>/dev/null | grep -q '^vllm-fn$'; then
        ssh_worker "docker logs --tail 50 vllm-fn" 2>&1 || true
        err "Worker container exited within 15s of start — see its log above; not launching the head."
    fi
    fi

    # ---- Head (rank 0) ----
    if [[ "$EXTRA_VLLM_ARGS" != *--api-key* ]]; then
        warn "Serving on 0.0.0.0 with no API key — anyone on the network can use it. Pass --api-key via EXTRA_VLLM_ARGS to require one."
    fi
    info "--- Launching head (rank 0) on $HEAD_IP ---"
    docker rm -f vllm-fn >/dev/null 2>&1 || true
    mkdir -p "$HOME/.cache/vllm"

    # Write head launch script (same approach as worker — avoids eval JSON issues)
    HEAD_SCRIPT=$(mktemp /tmp/vllm_head_XXXXXX.sh)
    cat > "$HEAD_SCRIPT" <<LAUNCH_EOF
#!/bin/bash
docker run \
    -d --name vllm-fn \
    --gpus all --network host --ipc host \
    --log-opt max-size=50m --log-opt max-file=3 \
    --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 \
    --device /dev/infiniband:/dev/infiniband \
    $CPUSET_ARG \
    $ALLOC_ENV \
    $HEAD_NET_ENV \
    -e NCCL_DEBUG=WARN \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_HOST_IP=$HEAD_IP \
    ${VLLM_ALLOW_LONG_MAX_MODEL_LEN:+-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=$VLLM_ALLOW_LONG_MAX_MODEL_LEN} \
    ${VLLM_BATCH_INVARIANT:+-e VLLM_BATCH_INVARIANT=$VLLM_BATCH_INVARIANT} \
    $PLE_OFFLOAD_ENV \
    -e HF_HOME=/root/.cache/huggingface \
    $HEAD_PLE_MOUNT \
    $HEAD_MODELOPT_MOUNT \
    $HEAD_OVERLAY_MOUNTS \
    $HEAD_CHAT_MOUNT \
    $OVERLAY_ENV_STR \
    -v $HF_CACHE_DIR:/root/.cache/huggingface \
    $HEAD_MODEL_MOUNT \
    -v $HOME/.cache/vllm:/root/.cache/vllm \
    $IMAGE \
    $MODEL_ARG \
    $VLLM_ARGS_STR \
    --node-rank 0 \
    --host 0.0.0.0 \
    --port $PORT
LAUNCH_EOF
    # Same reasoning as the worker script above: mktemp's 0600 is already right,
    # so no chmod. The inspection copy below does need one — `cp` onto an
    # existing .last_head_launch.sh keeps that file's old mode, so on any tree
    # that ever ran the `chmod +x` version above it stays 0711 (verified).
    cp "$HEAD_SCRIPT" "$SCRIPT_DIR/.last_head_launch.sh"   # for inspection (gitignored)
    chmod 600 "$SCRIPT_DIR/.last_head_launch.sh"           # may contain --api-key values

    info "  (starting head container...)"
    bash "$HEAD_SCRIPT"
    rm -f "$HEAD_SCRIPT"
    ok "Head container started."
    info ""
    info "vLLM is loading (~6-7 min). Following logs until ready..."
    info ""

    # Follow logs in background, poll /health until 200, then return to shell
    docker logs -f vllm-fn &
    LOGPID=$!
    READY_DEADLINE=$(( $(date +%s) + READY_TIMEOUT_S ))

    # On failure keep the evidence: archive the full container log under logs/
    # and echo its error lines, so a crash during load stays diagnosable.
    dump_failure_logs() {
        kill $LOGPID 2>/dev/null || true
        mkdir -p "$SCRIPT_DIR/logs"
        local ts log
        ts=$(date +%s)
        log="$SCRIPT_DIR/logs/vllm-fn-$ts.log"
        docker logs vllm-fn > "$log" 2>&1 || true
        warn "Container log archived to $log"
        grep -E 'ERROR|Traceback|Error' "$log" | tail -15
        # The head log alone hides the usual dual-node failure: the worker died
        # first and the head blocked on NCCL. Archive the worker side too.
        if [[ "$NNODES" -eq 2 ]]; then
            local wlog="$SCRIPT_DIR/logs/vllm-fn-worker-$ts.log"
            ssh_worker "docker logs --tail 3000 vllm-fn" > "$wlog" 2>&1 || true
            warn "Worker container log archived to $wlog"
        fi
    }

    info "Waiting for /health to return 200 (deadline ${READY_TIMEOUT_S}s)..."
    while true; do
        sleep 10
        # Check if container is still running
        if ! docker ps --format '{{.Names}}' | grep -q '^vllm-fn$'; then
            dump_failure_logs
            err "Container vllm-fn exited unexpectedly. See the archived log above."
        fi
        # A dead worker otherwise costs the full timeout: the head just blocks
        # in NCCL init and stays "running" until the deadline.
        if [[ "$NNODES" -eq 2 ]] && ! ssh_worker "docker ps --format '{{.Names}}'" 2>/dev/null | grep -q '^vllm-fn$'; then
            warn "Worker container vllm-fn is not running on $WORKER_IP — last 100 log lines:"
            ssh_worker "docker logs --tail 100 vllm-fn" 2>&1 || true
            dump_failure_logs
            err "Worker container exited during boot. See its log above and the archived logs."
        fi
        # Give up once the readiness deadline passes
        if (( $(date +%s) > READY_DEADLINE )); then
            dump_failure_logs
            err "vLLM not ready after ${READY_TIMEOUT_S}s. See the archived log above."
        fi
        # Check health endpoint
        HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null || echo "000")
        if [[ "$HTTP_CODE" == "200" ]]; then
            kill $LOGPID 2>/dev/null || true
            echo ""
            ok "vLLM is ready and serving on port $PORT!"
            # Host-memory watchdog, now that the load-time memory ramp is over.
            "$SCRIPT_DIR/scripts/start-memwatch.sh" vllm-fn >/dev/null
            info "Memory watchdog running (logs/memwatch-vllm-fn.log)"
            info ""
            info "Test with:"
            info "  curl http://localhost:$PORT/v1/chat/completions \\"
            info "    -H 'Content-Type: application/json' \\"
            info "    -d '{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
            info ""
            info "View logs: docker logs -f vllm-fn"
            info "Stop:      ./stop.sh"
            break
        fi
    done
