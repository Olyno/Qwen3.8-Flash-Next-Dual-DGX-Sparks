    info "=== Step 5: Docker image '$IMAGE' ==="

    if ! docker image inspect "$IMAGE" &>/dev/null; then
        info "Pulling $IMAGE ..."
        docker pull "$IMAGE"
    fi
    ok "Image ready on head."

    LOCAL_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || echo "")
    REMOTE_ID=$(ssh_worker "docker image inspect --format '{{.Id}}' '$IMAGE' 2>/dev/null" || echo "")

    if [[ "$LOCAL_ID" != "$REMOTE_ID" ]]; then
        info "Pulling image on worker..."
        ssh_worker "docker pull '$IMAGE'"
        ok "Image ready on worker."
    else
        ok "Image already on worker."
    fi

    # ---------------------------------------------------------------------------
    # 6. Prepare the PLE patch
    #    Mixed-quant NVFP4 checkpoints declare ple_embedding_dtype in config:
    #      nvfp4          -> packed uint8 PLE table (this checkpoint)
    #      float8_e4m3fn  -> FP8 PLE excluded from the parent ModelOpt config
    #    patches/patch_ple_layer.py adds NVFP4 + mixed dispatch (vLLM PR #53899 logic).
    # ---------------------------------------------------------------------------
    PATCHED_PLE="$SCRIPT_DIR/patches/ple_layer_patched.py"
    PLE_ORIG="$SCRIPT_DIR/patches/ple_layer_patched.py.orig"
    HEAD_PLE_MOUNT=""
    WORKER_PLE_MOUNT=""

    if [[ "$SKIP_PLE_PATCH" == "true" ]]; then
        info "=== Step 6: PLE patch skipped (native FP8 checkpoint) ==="
    else
    info "=== Step 6: Prepare PLE patch ==="

    if [[ ! -f "$PLE_ORIG" ]]; then
        info "Extracting ple_layer.py from image..."
        tmp_container=$(docker create "$IMAGE" /bin/true)
        docker cp "$tmp_container:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py" "$PLE_ORIG"
        docker rm "$tmp_container" >/dev/null 2>&1
        [[ -f "$PLE_ORIG" ]] || err "Failed to extract ple_layer.py from image. Is the image pulled?"
    fi

    python3 "$SCRIPT_DIR/patches/patch_ple_layer.py"
    [[ -f "$PATCHED_PLE" ]] || err "PLE patch file not found after patch_ple_layer.py"

    ok "PLE patch ready: $PATCHED_PLE"
    HEAD_PLE_MOUNT="-v $PATCHED_PLE:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro"
    WORKER_PLE_MOUNT="-v /tmp/ple_layer_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro"
    fi

    # ---------------------------------------------------------------------------
    # 6b. MXFP8 kernel-fallback patch.
    #     FlashInfer mm_mxfp8 needs N,K >= 128 and both divisible by 32. Two
    #     shapes in this checkpoint miss that — linear_attn.in_proj_a/b [48,2560]
    #     (fatal at engine start) and the vision MLP fc1 [4304,1152] — so those
    #     layers are routed to the BF16 emulation kernel. visual.* stays fully
    #     emulated (verified multimodal path; global dequant would OOM).
    # ---------------------------------------------------------------------------
    PATCHED_MODELOPT="$SCRIPT_DIR/patches/modelopt_patched.py"
    MODELOPT_ORIG="$SCRIPT_DIR/patches/modelopt_patched.py.orig"
    HEAD_MODELOPT_MOUNT=""
    WORKER_MODELOPT_MOUNT=""
    MODEL_OPT_PKG="$VLLM_PKG/model_executor/layers/quantization/modelopt.py"

    if [[ "$V030" == "true" ]]; then
    info "=== Step 6b: MXFP8 kernel-fallback patch skipped (vLLM 0.30) ==="
    else
    info "=== Step 6b: Prepare MXFP8 kernel-fallback patch ==="
    if [[ ! -f "$MODELOPT_ORIG" ]]; then
        info "Extracting modelopt.py from image..."
        tmp_container=$(docker create "$IMAGE" /bin/true)
        docker cp "$tmp_container:$MODEL_OPT_PKG" "$MODELOPT_ORIG"
        docker rm "$tmp_container" >/dev/null 2>&1
        [[ -f "$MODELOPT_ORIG" ]] || err "Failed to extract modelopt.py from image."
    fi
    python3 "$SCRIPT_DIR/patches/patch_modelopt_mxfp8.py"
    [[ -f "$PATCHED_MODELOPT" ]] || err "modelopt patch missing after patch_modelopt_mxfp8.py"
    # Stacks on top: adds the FP8_BLOCK_SCALES routed-expert branch that neither
    # this image nor upstream vLLM has, which is what MTP needs on this checkpoint.
    python3 "$SCRIPT_DIR/patches/patch_modelopt_fp8_block_moe.py"
    ok "MXFP8 fallback patch ready: $PATCHED_MODELOPT"
    HEAD_MODELOPT_MOUNT="-v $PATCHED_MODELOPT:$MODEL_OPT_PKG:ro"
    WORKER_MODELOPT_MOUNT="-v /tmp/modelopt_patched.py:$MODEL_OPT_PKG:ro"
    fi
