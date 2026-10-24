# ---------------------------------------------------------------------------
# 4c. vLLM overlay patches (bind-mounted files, no image rebuild).
#     overlays/fp8dense/*.py   -> FP8-dense loader support (only with FP8_DENSE=true)
#     overlays/qsa_gb10/qsa.py -> QSA launch-profile override (only with QSA_PROFILE != stock)
# ---------------------------------------------------------------------------
VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm
OVERLAY_MOUNTS=()        # head-side "-v host:container:ro"
OVERLAY_FILES=()         # host files to scp to the worker (/tmp/vllm-overlay/<name>)
OVERLAY_ENV=()
add_overlay() {          # add_overlay <host file> <container path>
    [[ -f "$1" ]] || err "overlay file missing: $1"
    # Two overlays on one container path would silently race in docker run, and
    # the worker copies land in a flat /tmp/vllm-overlay keyed by basename, so a
    # basename clash would have one file quietly overwrite the other there.
    for existing in "${OVERLAY_FILES[@]:-}"; do
        if [[ "${existing#*|}" == "$2" ]]; then
            err "overlay conflict on $2: already claimed by ${existing%%|*}, now $1"
        fi
        if [[ "$(basename "${existing%%|*}")" == "$(basename "$1")" ]]; then
            err "overlay basename clash on $(basename "$1"): ${existing%%|*} vs $1
       (worker overlays share a flat /tmp/vllm-overlay directory)"
        fi
    done
    OVERLAY_MOUNTS+=("-v $1:$2:ro")
    OVERLAY_FILES+=("$1|$2")
}
extract_from_image() {   # extract_from_image <container path> <host dest>
    # The cache is keyed by image id: a stale extraction from a previous image
    # must never be patched and mounted over the new one.
    local stamp="$2.image-id" img_id=""
    img_id=$(docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null || true)
    if [[ -f "$2" && -n "$img_id" && "$(cat "$stamp" 2>/dev/null)" == "$img_id" ]]; then
        return 0
    fi
    info "  Extracting $(basename "$1") from image..."
    local c; c=$(docker create "$IMAGE" /bin/true)
    docker cp "$c:$1" "$2" >/dev/null
    docker rm "$c" >/dev/null 2>&1
    [[ -f "$2" ]] || err "Failed to extract $1 from image."
    [[ -n "$img_id" ]] && printf '%s' "$img_id" > "$stamp"
}
if $DO_LAUNCH && [[ "$FP8_DENSE" == "true" && "$V030" != "true" ]]; then
    info "=== Step 4c: FP8-dense overlay ==="
    OV="$SCRIPT_DIR/overlays/fp8dense"
    NVIDIA="$VLLM_PKG/models/qwen3_8_flash_next/nvidia"
    for spec in \
        "modelopt.py $VLLM_PKG/model_executor/layers/quantization/modelopt.py" \
        "model.py $NVIDIA/model.py" \
        "hyperconnection.py $NVIDIA/hyperconnection.py" \
        "mtp.py $NVIDIA/mtp.py"; do
        read -r f src <<< "$spec"
        if [[ ! -f "$OV/$f" ]]; then
            extract_from_image "$src" "$OV/$f"
            (cd "$OV" && git apply -p2 "${f%.py}.diff") \
                || err "fp8dense: git apply ${f%.py}.diff failed (image layout changed?)"
        fi
    done
    add_overlay "$OV/modelopt.py"        "$VLLM_PKG/model_executor/layers/quantization/modelopt.py"
    add_overlay "$OV/model.py"           "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/model.py"
    add_overlay "$OV/hyperconnection.py" "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/hyperconnection.py"
    add_overlay "$OV/mtp.py"             "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/mtp.py"
    ok "FP8-dense overlay: 4 files"
fi
if $DO_LAUNCH && [[ "$QSA_PROFILE" != "stock" ]]; then
    info "=== Step 4d: QSA profile overlay ($QSA_PROFILE) ==="
    QO="$SCRIPT_DIR/overlays/qsa_gb10"
    [[ -f "$QO/qsa.py" ]] || python3 "$QO/apply_patch.py"
    add_overlay "$QO/qsa.py" "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
    if [[ -f "$QSA_PROFILE" ]]; then
        add_overlay "$QSA_PROFILE" "/etc/vllm-qsa-profile.json"
        OVERLAY_ENV+=("-e VLLM_QSA_PROFILE_JSON=/etc/vllm-qsa-profile.json")
    else
        OVERLAY_ENV+=("-e VLLM_QSA_PROFILE=$QSA_PROFILE")
    fi
fi

# ---------------------------------------------------------------------------
# 4d. Reduced-vocabulary MTP drafting (FR-Spec style).
#     The drafter owns a full 248,320-row BF16 lm_head that is read once per
#     draft step; slicing it to a frequency-ranked subset is the single largest
#     bandwidth lever in a decode step. Output-safe: a draft outside the subset
#     is rejected at verification, never emitted. See patches/patch_mtp_draft_vocab.py.
# ---------------------------------------------------------------------------
if $DO_LAUNCH && [[ "$V030" == "true" ]]; then
    [[ "$QSA_PROFILE" == "stock" ]] || err "V030: QSA_PROFILE=$QSA_PROFILE is not supported on the vLLM 0.32 lane."
    if [[ "$FP8_DENSE" == "true" ]]; then
        info "=== Step 4c: FP8-dense overlay (vLLM 0.32) ==="
        # Port of overlays/fp8dense onto the qwen4_exp tree: GatedResidual and
        # the two ParallelLMHeads get quant_config so the MIXED_PRECISION
        # config's per-prefix resolution reaches the dense HC and head
        # projections. v0.32's modelopt.py already dispatches
        # FP8_PER_CHANNEL_PER_TOKEN / FP8_PB_WO natively (LINEAR_ALGOS +
        # resolve()), so its only delta is the "model." prefix candidate the
        # text-only entry point needs. All four diffs apply to the stock
        # extracted files. MTP_DRAFT_VOCAB / FP8_DRAFT_HEAD are refused in
        # Step 4e: their reduced head assumes a BF16 draft lm_head and this
        # checkpoint's draft head is FP8.
        FD="$SCRIPT_DIR/overlays/fp8dense_v030"
        NVIDIA_V030="$VLLM_PKG/models/qwen4_exp/nvidia"
        mkdir -p "$FD/orig"
        for f in modelopt model hyperconnection mtp; do
            case "$f" in
                modelopt) fsrc="$VLLM_PKG/model_executor/layers/quantization/modelopt.py" ;;
                *)        fsrc="$NVIDIA_V030/$f.py" ;;
            esac
            extract_from_image "$fsrc" "$FD/orig/$f.py"
            cp "$FD/orig/$f.py" "$FD/$f.py"
            (cd "$FD" && git apply -p2 "$f.diff") \
                || err "fp8dense v030: git apply $f.diff failed (image layout changed?)"
            add_overlay "$FD/$f.py" "$fsrc"
        done
        ok "FP8-dense overlay (v030): 4 files"
    fi
    info "=== Step 4c: Mamba drop_eagle_block cache-hit fix (vllm#57128, vLLM 0.32) ==="
    # Backport of vllm#57128 (patches/mamba_eagle_drop/): MambaManager's
    # find_longest_cache_hit accepted drop_eagle_block but never acted on it,
    # so with GDN + MTP + prefix caching the most recent matched block (state
    # possibly written over rejected draft positions) stayed reachable through
    # the prefix cache — silent wrong outputs. Both search branches now skip
    # the first (most recent) checkpoint found and keep scanning for an older,
    # committed one; kv_cache_coordinator gets the matching comment fix.
    # Correctness fix, so no toggle.
    MD="$SCRIPT_DIR/patches/mamba_eagle_drop"
    mkdir -p "$MD/orig"
    extract_from_image "$VLLM_PKG/v1/core/single_type_kv_cache_manager.py" \
                       "$MD/orig/single_type_kv_cache_manager.py"
    extract_from_image "$VLLM_PKG/v1/core/kv_cache_coordinator.py" \
                       "$MD/orig/kv_cache_coordinator.py"
    python3 "$MD/apply_patch.py" || err "mamba_eagle_drop apply_patch.py failed"
    add_overlay "$MD/single_type_kv_cache_manager_v030.py" \
                "$VLLM_PKG/v1/core/single_type_kv_cache_manager.py"
    add_overlay "$MD/kv_cache_coordinator_v030.py" \
                "$VLLM_PKG/v1/core/kv_cache_coordinator.py"
    [[ "$VLLM_QSA_DET_TOPK" == "1" || "$VLLM_MOE_DET_FINALIZE" == "1" ]] && err "V030: the determinism knobs are not ported to vLLM 0.32."
    if [[ "$PLE_OFFLOAD" == "true" ]]; then
        info "=== Step 4c: PLE mmap offload (vLLM 0.32) ==="
        # v0.32's PLE CPU offload parks the table in pinned anonymous host
        # memory (~32 GiB/node). The mmap overlay reroutes it to a file-backed
        # table read over ATS (VLLM_PLE_MMAP_DIR), persisted across launches
        # under ~/.cache/vllm. Single-node Spark needs the offload to fit the
        # model at TP=1; the mmap form avoids the pin on any topology. The
        # implementation lives in models/qwen4_exp/common/ngram_embedding.py
        # (v0.32 split it out of the nvidia wrapper), so the patcher rewrites
        # both files and both are mounted; the patched subclass inherits the
        # native supports_prefetch side-stream for free.
        PL="$SCRIPT_DIR/patches/v030_ple"
        mkdir -p "$PL/orig"
        extract_from_image "$VLLM_PKG/models/qwen4_exp/common/ngram_embedding.py" \
                           "$PL/orig/ngram_common.py"
        extract_from_image "$VLLM_PKG/models/qwen4_exp/nvidia/ngram_embedding.py" \
                           "$PL/orig/ngram_nvidia.py"
        python3 "$SCRIPT_DIR/patches/patch_ple_mmap_v030.py" "$PL" \
            || err "patch_ple_mmap_v030.py failed"
        add_overlay "$PL/ple_common_v032.py" \
                    "$VLLM_PKG/models/qwen4_exp/common/ngram_embedding.py"
        add_overlay "$PL/ple_nvidia_v032.py" \
                    "$VLLM_PKG/models/qwen4_exp/nvidia/ngram_embedding.py"
        OVERLAY_ENV+=("-e VLLM_PLE_MMAP_DIR=/root/.cache/vllm/ple_mmap_v030")
        # VLLM_PLE_MMAP_ADVICE is also the unified-memory page-cache hygiene
        # kill-switch: nonzero (=1) additionally sets FADV_RANDOM on the table
        # fd, pre-faults a ready table with MADV_POPULATE_READ, and warms each
        # eager PLE batch's row pages with coalesced preads; 0 disables the
        # whole set. See patches/patch_ple_mmap_v030.py's docstring.
        OVERLAY_ENV+=("-e VLLM_PLE_MMAP_ADVICE=1")
        if [[ "$PLE_STAGED_LOOKUP" == "true" ]]; then
            # Host gather through a pinned staging buffer, H2D before each
            # graph replay, so the GPU never reads the file-backed mapping.
            # The staged path also installs the per-replay run_fullgraph hook.
            OVERLAY_ENV+=("-e VLLM_PLE_STAGED_LOOKUP=1")
        fi
    fi
    if [[ "$QSA_FUSED_DRAFT" == "true" ]]; then
        info "=== Step 4c: QSA fused multi-step draft metadata (vLLM 0.32) ==="
        # Port of myllmbox/vllm@c3f56fe (the code proposed upstream as
        # vllm#58449, patches/patch_qsa_fused_draft_v030.py): stock v0.32
        # rebuilds attention metadata for the whole model between MTP draft
        # steps because the QSA builder never opted into the speculator's
        # in-place update; the overlay re-launches the builder's own metadata
        # kernel on its persistent buffers instead.
        [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 1 ]] || err "qsa_fused_draft needs MTP_NUM_SPECULATIVE_TOKENS > 1 (with k <= 1 there is no inter-draft-step rebuild to fuse)."
        mkdir -p "$SCRIPT_DIR/patches/v030_qsa_fused/orig"
        extract_from_image "$VLLM_PKG/models/qwen4_exp/common/qsa_cache.py" \
                           "$SCRIPT_DIR/patches/v030_qsa_fused/orig/qsa_cache.py"
        python3 "$SCRIPT_DIR/patches/patch_qsa_fused_draft_v030.py" || err "patch_qsa_fused_draft_v030.py failed"
        # qsa_cache.py is NOT mounted here: the qsa_meta_clamp block below
        # re-patches this output with the vllm#58040 graph-padding clamp and
        # mounts the result (the two patches touch non-overlapping regions).
        OVERLAY_ENV+=("-e VLLM_QSA_FUSED_DRAFT=1")
    fi
    info "=== Step 4c: QSA metadata graph-padding clamp (vllm#58040, vLLM 0.32) ==="
    # Backport of vllm#58040 (patches/qsa_meta_clamp/): CUDA-graph capture
    # pads query_start_loc_cpu[-1] past num_actual_tokens while the QSA
    # metadata buffers hold only real tokens, so a padded decode step
    # read/wrote past those buffers. Both metadata builders (and, with
    # QSA_FUSED_DRAFT=true, the fused-draft inline call arg + recorded
    # metadata field) now clamp the mapped count to num_actual_tokens.
    # Correctness fix, so no toggle. With QSA_FUSED_DRAFT=true the input is
    # the fused-draft overlay output above; otherwise the stock file (the
    # clamped regions are identical in both flavors). Mounts the result.
    QC="$SCRIPT_DIR/patches/qsa_meta_clamp"
    mkdir -p "$QC/orig"
    extract_from_image "$VLLM_PKG/models/qwen4_exp/common/qsa_cache.py" "$QC/orig/qsa_cache_stock.py"
    if [[ "$QSA_FUSED_DRAFT" == "true" ]]; then
        cp "$SCRIPT_DIR/patches/v030_qsa_fused/qsa_cache_v030.py" "$QC/orig/qsa_cache.py"
    else
        cp "$QC/orig/qsa_cache_stock.py" "$QC/orig/qsa_cache.py"
    fi
    python3 "$QC/apply_patch.py" || err "qsa_meta_clamp apply_patch.py failed"
    add_overlay "$QC/qsa_cache_v030.py" "$VLLM_PKG/models/qwen4_exp/common/qsa_cache.py"
    if [[ "$QSA_ROPE_CLAMP" == "true" ]]; then
        info "=== Step 4c: QSA prepare RoPE clamp (vLLM 0.32) ==="
        # The clamp (myllmbox/vllm@9ff17c0) retargeted to v0.32's fused
        # prepare kernel: ops/qsa_prepare.py (native since vllm#57097) loads
        # cos_sin[pos] unchecked both inside _norm_rope and in its inline
        # main-attention RoPE, and CUDA-graph warmup feeds dummy positions
        # past the cos/sin table (IMA on SM121/GB10). patches/
        # patch_qsa_rope_clamp_v030.py clamps all of them behind a CLAMP_POS
        # constexpr (default OFF, fail closed — the constexpr-false branch
        # compiles to the bit-exact stock kernel); mount the patched file and
        # arm it via the env flag.
        mkdir -p "$SCRIPT_DIR/patches/v030_qsa_rope/orig"
        extract_from_image "$VLLM_PKG/models/qwen4_exp/nvidia/ops/qsa_prepare.py" \
                           "$SCRIPT_DIR/patches/v030_qsa_rope/orig/qsa_prepare.py"
        python3 "$SCRIPT_DIR/patches/patch_qsa_rope_clamp_v030.py" \
            || err "patch_qsa_rope_clamp_v030.py failed"
        add_overlay "$SCRIPT_DIR/patches/v030_qsa_rope/qsa_prepare_v032.py" \
                    "$VLLM_PKG/models/qwen4_exp/nvidia/ops/qsa_prepare.py"
        OVERLAY_ENV+=("-e VLLM_QSA_ROPE_CLAMP=1")
    fi
    if [[ "$LOAD_DROP_CACHE" == "true" ]]; then
        info "=== Step 4c: loader page-cache drop (vLLM 0.32) ==="
        # Port of myllmbox/vllm@354bfc2 (patches/patch_load_drop_cache_v030.py):
        # posix_fadvise(DONTNEED) on each safetensors shard right after the
        # loader consumes it. Step 4b-2 evicts the checkpoint from the host
        # side before launch; this keeps the cache one shard deep during the
        # load itself, on the same unified-memory pool the GPU allocates from.
        mkdir -p "$SCRIPT_DIR/patches/v030_dropcache/orig"
        extract_from_image "$VLLM_PKG/model_executor/model_loader/weight_utils.py" \
                           "$SCRIPT_DIR/patches/v030_dropcache/orig/weight_utils.py"
        python3 "$SCRIPT_DIR/patches/patch_load_drop_cache_v030.py" || err "patch_load_drop_cache_v030.py failed"
        add_overlay "$SCRIPT_DIR/patches/v030_dropcache/weight_utils_v030.py" \
                    "$VLLM_PKG/model_executor/model_loader/weight_utils.py"
        OVERLAY_ENV+=("-e VLLM_LOAD_DROP_CACHE=1")
    fi
    if [[ "$FREE_BLOCK_SORT" == "true" ]]; then
        info "=== Step 4c: free-list block-id sort (vllm#31371, vLLM 0.32) ==="
        # RFC overlay (patches/free_block_sort/): FreeKVCacheBlockQueue.append_n
        # sorts freed blocks by block_id before linking, so pops hand out
        # address-sequential blocks — contiguous ids coalesce into larger DMA
        # bursts on KV offload/prefetch. Trades strict LRU order within a
        # freed batch; no numeric effect on emitted tokens.
        FS="$SCRIPT_DIR/patches/free_block_sort"
        mkdir -p "$FS/orig"
        extract_from_image "$VLLM_PKG/v1/core/kv_cache_utils.py" \
                           "$FS/orig/kv_cache_utils.py"
        python3 "$FS/apply_patch.py" || err "free_block_sort apply_patch.py failed"
        add_overlay "$FS/kv_cache_utils_v030.py" \
                    "$VLLM_PKG/v1/core/kv_cache_utils.py"
    fi
    if [[ "$CHUNK_EQUAL_SPLIT" == "true" ]]; then
        info "=== Step 4c: equal-width chunked prefill (vLLM 0.32) ==="
        # patches/patch_chunk_equal_split.py (idea credit: Niko1221/Strata,
        # MIT, chunk policy #41): stock caps a prefill chunk at
        # long_prefill_token_threshold, so a prompt just past it lands on a
        # stub tail — 5,000 rows at the prod cap of 4,800 splits 4800+200 and
        # spends a whole step (metadata build, KV alloc, GDN state checkpoint)
        # on 200 rows. The overlay keeps the cap and the step count and cuts
        # ceil(N/cap) equal chunks instead: 2500+2500. Both clamp sites in
        # v1/core/sched/scheduler.py take it; nothing else overlays that file
        # on this lane.
        CES="$SCRIPT_DIR/patches/chunk_equal_split"
        mkdir -p "$CES/orig"
        extract_from_image "$VLLM_PKG/v1/core/sched/scheduler.py" \
                           "$CES/orig/scheduler.py"
        python3 "$SCRIPT_DIR/patches/patch_chunk_equal_split.py" "$CES" \
            || err "patch_chunk_equal_split.py failed"
        add_overlay "$CES/scheduler_v032.py" "$VLLM_PKG/v1/core/sched/scheduler.py"
        OVERLAY_ENV+=("-e VLLM_CHUNK_EQUAL_SPLIT=1")
    fi
    if [[ "$GDN_DIRECT_OUT" == "true" ]]; then
        info "=== Step 4c: GDN prefill direct output (vLLM 0.32) ==="
        # patches/patch_gdn_direct_out.py: the stock flashinfer GDN prefill
        # call omits its output= parameter, so the kernel allocates a
        # [tokens, v_heads, head_dim] tensor that is then copied into the
        # layer's preallocated core_attn_out. On prefill-only steps the
        # overlay passes that destination slice as output= and skips both
        # copies. Same kernel and inputs, different destination -> bit
        # parity expected; ineligible buffers keep the stock path.
        GD="$SCRIPT_DIR/patches/gdn_direct_out"
        mkdir -p "$GD/orig"
        extract_from_image "$VLLM_PKG/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py" \
                           "$GD/orig/qwen_gdn_linear_attn.py"
        python3 "$SCRIPT_DIR/patches/patch_gdn_direct_out.py" "$GD" \
            || err "patch_gdn_direct_out.py failed"
        add_overlay "$GD/qwen_gdn_linear_attn_v030.py" \
                    "$VLLM_PKG/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
        OVERLAY_ENV+=("-e VLLM_GDN_DIRECT_OUT=1")
    fi
    if [[ "$NVFP4_ACT_HOOKS" == "true" ]]; then
        info "=== Step 4c: NVFP4 activation channel-stat hooks (debug) ==="
        # Debug overlay (patches/patch_nvfp4_act_hooks.py): per-channel absmax
        # capture on the emulated NVFP4 MoE path for
        # check_nvfp4_activations.py --channel-stats. Only meaningful with
        # moe_backend: emulation (the fused backends never materialize the
        # down_proj input). Arms on /tmp/nvfp4_act_hooks.arm inside the
        # container, dumps to /tmp/nvfp4_act_hooks.json, then disarms.
        HK="$SCRIPT_DIR/patches/v030_nvfp4hooks"
        mkdir -p "$HK/orig"
        extract_from_image "$VLLM_PKG/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py" \
                           "$HK/orig/nvfp4_emulation_moe.py"
        python3 "$SCRIPT_DIR/patches/patch_nvfp4_act_hooks.py" || err "patch_nvfp4_act_hooks.py failed"
        add_overlay "$HK/nvfp4_emulation_moe_v030.py" \
                    "$VLLM_PKG/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py"
        OVERLAY_ENV+=("-e VLLM_NVFP4_ACT_HOOKS=1")
    fi
    OVERLAY_ENV+=("-e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/tmp/fi_autotune")
    OVERLAY_ENV+=("-e VLLM_USE_BREAKABLE_CUDAGRAPH=${V030_BREAKABLE_CUDAGRAPH:-0}")
fi
if $DO_LAUNCH && [[ "$V030" == "true" ]] && { [[ -n "$MTP_DRAFT_VOCAB" ]] || [[ "$FP8_DRAFT_HEAD" == "true" ]]; }; then
    info "=== Step 4e: MTP draft head (vLLM 0.32) ==="
    [[ "$MTP_NUM_SPECULATIVE_TOKENS" == "0" ]] && err "MTP_DRAFT_VOCAB/FP8_DRAFT_HEAD is set but MTP_NUM_SPECULATIVE_TOKENS=0 - nothing drafts."
    # The fp8dense checkpoint's draft lm_head is FP8; the reduced/FP8 draft
    # head paths here assume BF16 rows (a bare F.linear on the slice dies with
    # "addmm_cuda not implemented for Float8_e4m3fn"). Refuse the combination.
    [[ "$FP8_DENSE" != "true" ]] || err "MTP_DRAFT_VOCAB/FP8_DRAFT_HEAD and FP8_DENSE both touch the qwen4_exp draft head, and the reduced/FP8 draft head assumes a BF16 lm_head - pick one."
    if [[ -n "$MTP_DRAFT_VOCAB" ]]; then
        [[ -f "$MTP_DRAFT_VOCAB" ]] || err "MTP_DRAFT_VOCAB file not found: $MTP_DRAFT_VOCAB"
    fi
    extract_from_image "$VLLM_PKG/models/qwen4_exp/nvidia/mtp.py" \
                       "$SCRIPT_DIR/patches/mtp_v030_patched.py.orig"
    if [[ "$FP8_DRAFT_HEAD" == "true" ]]; then
        # One patched mtp.py carries both deltas, each gated by its own env
        # (VLLM_MTP_DRAFT_VOCAB / VLLM_MTP_DRAFT_HEAD_FP8) so they compose
        # independently. With no draft vocab the FP8 head covers the full
        # shard and hooks compute_logits; with one it quantizes the slice.
        python3 "$SCRIPT_DIR/patches/patch_mtp_draft_vocab_v030.py" --fp8 \
            || err "patch_mtp_draft_vocab_v030.py --fp8 failed"
        OVERLAY_ENV+=("-e VLLM_MTP_DRAFT_HEAD_FP8=1")
    else
        python3 "$SCRIPT_DIR/patches/patch_mtp_draft_vocab_v030.py" || err "patch_mtp_draft_vocab_v030.py failed"
    fi
    # With MTP_SKIP_PREFIX the prefix-skip block below chains off this output
    # and mounts the chained file itself (two overlays on one container path
    # trip add_overlay's conflict guard).
    if [[ "$MTP_SKIP_PREFIX" != "true" ]]; then
        add_overlay "$SCRIPT_DIR/patches/mtp_v030_patched.py" \
                    "$VLLM_PKG/models/qwen4_exp/nvidia/mtp.py"
    fi
    if [[ -n "$MTP_DRAFT_VOCAB" ]]; then
        add_overlay "$MTP_DRAFT_VOCAB" "/etc/vllm-draft-vocab.txt"
        OVERLAY_ENV+=("-e VLLM_MTP_DRAFT_VOCAB=/etc/vllm-draft-vocab.txt")
        ok "Draft vocab: $(wc -l < "$MTP_DRAFT_VOCAB") ids from $MTP_DRAFT_VOCAB"
    fi
elif $DO_LAUNCH && [[ -n "$MTP_DRAFT_VOCAB" ]]; then
    info "=== Step 4e: MTP reduced draft vocabulary ==="
    if [[ "$MTP_NUM_SPECULATIVE_TOKENS" == "0" ]]; then
        err "MTP_DRAFT_VOCAB is set but MTP_NUM_SPECULATIVE_TOKENS=0 - nothing drafts."
    fi
    [[ -f "$MTP_DRAFT_VOCAB" ]] || err "MTP_DRAFT_VOCAB file not found: $MTP_DRAFT_VOCAB
       Build one with: python3 scripts/build_draft_vocab.py <corpus.jsonl> --out draft_vocab.txt --size 65536"
    if [[ "$FP8_DENSE" == "true" ]]; then
        err "MTP_DRAFT_VOCAB and FP8_DENSE both overlay nvidia/mtp.py - pick one."
    fi
    extract_from_image "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/mtp.py" \
                       "$SCRIPT_DIR/patches/mtp_patched.py.orig"
    python3 "$SCRIPT_DIR/patches/patch_mtp_draft_vocab.py"
    add_overlay "$SCRIPT_DIR/patches/mtp_patched.py" \
                "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/mtp.py"
    add_overlay "$MTP_DRAFT_VOCAB" "/etc/vllm-draft-vocab.txt"
    OVERLAY_ENV+=("-e VLLM_MTP_DRAFT_VOCAB=/etc/vllm-draft-vocab.txt")
    ok "Draft vocab: $(wc -l < "$MTP_DRAFT_VOCAB") ids from $MTP_DRAFT_VOCAB"
elif $DO_LAUNCH && [[ "$MTP_NUM_SPECULATIVE_TOKENS" != "0" ]]; then
    warn "MTP=$MTP_NUM_SPECULATIVE_TOKENS is drafting over the FULL 248,320-token head"
    warn "     (0.59 GiB/rank at TP=2, read once per draft step). Setting MTP_DRAFT_VOCAB"
    warn "     to vocab/draft_vocab_en_code_47k.txt cuts that ~5x; see the recipes."
fi

if $DO_LAUNCH && [[ "$V030" == "true" && "$MTP_SKIP_PREFIX" == "true" ]]; then
    info "=== Step 4e: MTP prefill prefix skip (vLLM 0.32) ==="
    [[ "$MTP_NUM_SPECULATIVE_TOKENS" == "0" ]] && err "MTP_SKIP_PREFIX is on but MTP_NUM_SPECULATIVE_TOKENS=0 - nothing drafts."
    # patches/patch_mtp_prefix_skip.py: the draft prefill forward is consumed
    # at the per-sequence last rows only, so the drafter's sparse-MoE block
    # and final HC mixer run on just those tail rows (attention stays full
    # width — the drafter's own KV cache needs every prompt row's K/V). Each
    # stage self-checks bit-parity on the first eager prefill and pins itself
    # to stock on a mismatch. Idea credit: ds4-dfm-rs + flashrt, see the
    # patcher docstring. Chains the draft-vocab output when that block is
    # active (disjoint regions; it skips its own mount for ours).
    PS="$SCRIPT_DIR/patches/mtp_prefix_skip"
    mkdir -p "$PS/orig"
    if [[ -n "$MTP_DRAFT_VOCAB" || "$FP8_DRAFT_HEAD" == "true" ]]; then
        cp "$SCRIPT_DIR/patches/mtp_v030_patched.py" "$PS/orig/mtp.py"
    else
        extract_from_image "$VLLM_PKG/models/qwen4_exp/nvidia/mtp.py" \
                           "$PS/orig/mtp.py"
    fi
    python3 "$SCRIPT_DIR/patches/patch_mtp_prefix_skip.py" "$PS" \
        || err "patch_mtp_prefix_skip.py failed"
    add_overlay "$PS/mtp_v030.py" "$VLLM_PKG/models/qwen4_exp/nvidia/mtp.py"
    OVERLAY_ENV+=("-e VLLM_MTP_SKIP_PREFIX=1")
    ok "MTP prefix skip armed (parity self-check lands with the first prefill)"
fi

if $DO_LAUNCH && [[ "$V030" == "true" && "$MTP_ADAPTIVE_DEPTH" == "true" ]]; then
    info "=== Step 4e: adaptive MTP draft depth (vLLM 0.32) ==="
    # patches/patch_mtp_adaptive_depth.py: the V2 speculator (the code the
    # gpu_worker V2 runner actually executes) stops the draft chain early
    # once the draft head's own survival product (running product of per-step
    # top-token probs, batch mean) drops below
    # VLLM_MTP_ADAPTIVE_DEPTH_THRESHOLD. Per-step uniform k: propose()
    # publishes the truncated width and the model runner hands the narrowed
    # slice to the DraftTokensHandler (variable chain widths are a stock
    # scheduler path). The host-driven draft loops also break at the cut;
    # under the fused-FULL single-graph loop (qsa_fused_draft) the chain runs
    # full length and the width is trimmed post-hoc. Target verification is
    # unchanged. Composes with MTP_DRAFT_VOCAB / FP8_DRAFT_HEAD (the
    # confidence is read off the reduced/FP8 head) and QSA_FUSED_DRAFT
    # (different file: qsa_cache.py, not the speculator).
    AD="$SCRIPT_DIR/patches/mtp_adaptive_depth"
    mkdir -p "$AD/orig"
    extract_from_image "$VLLM_PKG/v1/worker/gpu/spec_decode/autoregressive/speculator.py" \
                       "$AD/orig/speculator.py"
    extract_from_image "$VLLM_PKG/v1/worker/gpu/model_runner.py" \
                       "$AD/orig/model_runner.py"
    python3 "$SCRIPT_DIR/patches/patch_mtp_adaptive_depth.py" || err "patch_mtp_adaptive_depth.py failed"
    add_overlay "$AD/speculator_v030.py" \
                "$VLLM_PKG/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
    add_overlay "$AD/model_runner_v030.py" \
                "$VLLM_PKG/v1/worker/gpu/model_runner.py"
    OVERLAY_ENV+=("-e VLLM_MTP_ADAPTIVE_DEPTH=1" "-e VLLM_MTP_ADAPTIVE_DEPTH_THRESHOLD=$MTP_ADAPTIVE_DEPTH_THRESHOLD")
    [[ "$MTP_ADAPTIVE_DEBUG" == "true" ]] && OVERLAY_ENV+=("-e VLLM_MTP_ADAPTIVE_DEBUG=1")
fi

if $DO_LAUNCH && [[ "$V030" == "true" && "$SPEC_TRACE" == "true" ]]; then
    info "=== Step 4e: speculative-decode round trace (vLLM 0.32) ==="
    # Measurement-only overlay (patches/patch_spec_trace.py): appends one
    # JSONL line per draft/verify round (scheduled draft positions, accepted,
    # wall ms) host-side at the end of GPUModelRunner.sample_tokens. Off = no
    # file, no timings, no syncs. The container writes to VLLM_SPEC_TRACE
    # (default /tmp/vllm-spec-trace.jsonl); docker cp it out afterwards.
    [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]] \
        || err "spec_trace needs speculative decoding (mtp_num_speculative_tokens > 0)."
    [[ "$MTP_ADAPTIVE_DEPTH" != "true" ]] \
        || err "spec_trace and mtp_adaptive_depth both overlay v1/worker/gpu/model_runner.py - pick one."
    ST="$SCRIPT_DIR/patches/spec_trace"
    mkdir -p "$ST/orig"
    extract_from_image "$VLLM_PKG/v1/worker/gpu/model_runner.py" \
                       "$ST/orig/model_runner.py"
    python3 "$SCRIPT_DIR/patches/patch_spec_trace.py" "$ST" \
        || err "patch_spec_trace.py failed"
    # With COPY_PROPOSER the copy patch chains off this output and mounts the
    # chained file itself (two overlays on one container path trip
    # add_overlay's conflict guard).
    if [[ "$COPY_PROPOSER" != "true" ]]; then
        add_overlay "$ST/model_runner_v030.py" \
                    "$VLLM_PKG/v1/worker/gpu/model_runner.py"
    fi
    OVERLAY_ENV+=("-e VLLM_SPEC_TRACE=${VLLM_SPEC_TRACE:-/tmp/vllm-spec-trace.jsonl}")
fi

if $DO_LAUNCH && [[ "$V030" == "true" && "$COPY_PROPOSER" == "true" ]]; then
    info "=== Step 4e: copy-based draft proposer (vLLM 0.32) ==="
    # patches/patch_copy_proposer.py: the stock n-gram scan/finalize kernels
    # (imported from the ngram_gpu speculator module, not overlaid) turn each
    # request's rightmost longest suffix match into its next draft chain when
    # one exists; otherwise the MTP drafts stay. Host-side dispatch, eager,
    # no device syncs. Target verification is exact, so outputs are unchanged.
    [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]] \
        || err "copy_proposer needs speculative decoding (mtp_num_speculative_tokens > 0)."
    [[ "$MTP_ADAPTIVE_DEPTH" != "true" ]] \
        || err "copy_proposer and mtp_adaptive_depth both overlay v1/worker/gpu/model_runner.py - pick one."
    CP="$SCRIPT_DIR/patches/copy_proposer"
    mkdir -p "$CP/orig"
    if [[ "$SPEC_TRACE" == "true" ]]; then
        # Chain the trace overlay: its output is this patch's input. Drop any
        # stamp so a later run without spec_trace re-extracts stock.
        cp "$ST/model_runner_v030.py" "$CP/orig/model_runner.py"
        rm -f "$CP/orig/model_runner.py.image-id"
    else
        extract_from_image "$VLLM_PKG/v1/worker/gpu/model_runner.py" \
                           "$CP/orig/model_runner.py"
    fi
    python3 "$SCRIPT_DIR/patches/patch_copy_proposer.py" "$CP" \
        || err "patch_copy_proposer.py failed"
    add_overlay "$CP/model_runner_v030.py" \
                "$VLLM_PKG/v1/worker/gpu/model_runner.py"
    OVERLAY_ENV+=("-e VLLM_COPY_PROPOSER=1")
fi

if $DO_LAUNCH && [[ "$V030" == "true" && "$VERIFY_WIDTH_ADAPTIVE" == "true" ]]; then
    info "=== Step 4c: adaptive speculative-verify width (vLLM 0.32) ==="
    # patches/patch_verify_width.py: the scheduler publishes one verify width
    # per step, picked from its rolling acceptance EMA (1 / 2 / config k,
    # four-step dwell) and capped at config k. Only the number of verify rows
    # changes; target verification is untouched. DP>1 and the stock
    # per-batch-size table stand the policy down at scheduler construction.
    VW="$SCRIPT_DIR/patches/verify_width"
    mkdir -p "$VW/orig"
    extract_from_image "$VLLM_PKG/v1/core/sched/scheduler.py" \
                       "$VW/orig/scheduler.py"
    python3 "$SCRIPT_DIR/patches/patch_verify_width.py" "$VW" \
        || err "patch_verify_width.py failed"
    add_overlay "$VW/scheduler_v030.py" "$VLLM_PKG/v1/core/sched/scheduler.py"
    OVERLAY_ENV+=("-e VLLM_VERIFY_WIDTH_ADAPTIVE=1")
fi

# ---------------------------------------------------------------------------
# 4e. FP8 KV cache. The stock QSA kernels hard-refuse anything but BF16 KV
#     (supported_kv_cache_dtypes = ["auto","bfloat16"]); this teaches them to
#     read an FP8-e4m3 cache with per-tensor scales applied after the dots.
#     A capacity trade, not a free win - see the README before enabling.
# ---------------------------------------------------------------------------
if $DO_LAUNCH && [[ "$KV_CACHE_DTYPE" == fp8* && "$V030" != "true" ]]; then
    info "=== Step 4f: FP8 KV cache patch ($KV_CACHE_DTYPE) ==="
    if [[ "$QSA_PROFILE" != "stock" ]]; then
        err "KV_CACHE_DTYPE=$KV_CACHE_DTYPE and QSA_PROFILE=$QSA_PROFILE both overlay ops/qsa.py - pick one."
    fi
    extract_from_image "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ops/qsa.py" \
                       "$SCRIPT_DIR/patches/qsa_ops_patched.py.orig"
    extract_from_image "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/qsa.py" \
                       "$SCRIPT_DIR/patches/qsa_nvidia_patched.py.orig"
    python3 "$SCRIPT_DIR/patches/patch_qsa_fp8_kv.py"
    add_overlay "$SCRIPT_DIR/patches/qsa_ops_patched.py" \
                "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
    add_overlay "$SCRIPT_DIR/patches/qsa_nvidia_patched.py" \
                "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/qsa.py"
    warn "FP8 KV is a quality trade on sparse attention - validate reasoning on your workload."
fi

# ---------------------------------------------------------------------------
# 4d. MTP layer-index alias overlay.
#     vLLM builds the MTP draft layer at the absolute index that continues the
#     main stack (mtp.layers.48 for num_hidden_layers=48) and matches that
#     prefix against quantization_config.quantized_layers by exact string.
#     nvidia/... records only mtp.layers.0, so the lookup misses, the MTP MoE is
#     built unquantized, and its FP8 weight_scale_inv tensors fail to load. We
#     bind-mount a config.json carrying both names (what the known-good
#     local-inference-lab checkpoint ships) — the HF cache is left untouched.
# ---------------------------------------------------------------------------
if $DO_LAUNCH && [[ -n "${SNAPSHOT_SHA:-}" && "$V030" != "true" ]]; then
    info "=== Step 4g: MTP layer-index alias ==="
    if [[ -n "$MODEL_PATH" ]]; then
        CONTAINER_SNAPSHOT="/model"
    else
        CONTAINER_SNAPSHOT="/root/.cache/huggingface/hub/models--${ORG}--${NAME}/snapshots/${SNAPSHOT_SHA}"
    fi
    rm -f "$SCRIPT_DIR/patches/config_patched.json" \
          "$SCRIPT_DIR/patches/hf_quant_config_patched.json"
    PATCHED_FILES=$(python3 "$SCRIPT_DIR/patches/patch_checkpoint_config.py" \
        "$PLE_CONFIG_DIR" "$SCRIPT_DIR/patches")
    if [[ -z "$PATCHED_FILES" ]]; then
        ok "Checkpoint already declares absolute MTP layer indices"
    else
        # quantized_layers lives in BOTH config.json and the legacy
        # hf_quant_config.json, and the two can disagree: nvidia/... rev
        # fc694b54 says FP8_PB_WO in config.json and FP8_BLOCK_SCALES in the
        # sidecar. Runtime evidence (issue #38) shows the MoE dispatch
        # consumes config.json, so that mount is the one that must be right.
        # The sidecar is mounted too, for consistency, not because it wins.
        for cfg_name in $PATCHED_FILES; do
            case "$cfg_name" in
                config.json)         host_file="$SCRIPT_DIR/patches/config_patched.json" ;;
                hf_quant_config.json) host_file="$SCRIPT_DIR/patches/hf_quant_config_patched.json" ;;
                *) err "unexpected patched config: $cfg_name" ;;
            esac
            add_overlay "$host_file" "$CONTAINER_SNAPSHOT/$cfg_name"
        done
        ok "MTP experts alias added to: $PATCHED_FILES"
    fi

    # Speculative decoding needs the MTP routed experts to be built with a
    # quantization method the mixed-precision dispatch actually implements.
    # ModelOptMixedPrecisionConfig.get_quant_method covers FP8 / NVFP4 /
    # W4A16_NVFP4 / MXFP8 for RoutedExperts and returns None for anything else,
    # which yields a silently *unquantized* MoE that then dies ~7 min into the
    # load. Fail fast here instead.
    if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]]; then
        MTP_ALGO=$(python3 "$SCRIPT_DIR/patches/patch_checkpoint_config.py" \
            --mtp-moe-algo "$PLE_CONFIG_DIR") && MTP_RC=0 || MTP_RC=$?
        if [[ "$MTP_RC" -eq 3 ]]; then
            err "MTP experts are ${MTP_ALGO}, which this image's mixed-precision MoE
       dispatch cannot build (supports FP8 / NVFP4 / W4A16_NVFP4 / MXFP8 /
       FP8_BLOCK_SCALES; FP8_PB_WO with group_size 128 is treated as
       FP8_BLOCK_SCALES). Set MTP_NUM_SPECULATIVE_TOKENS=0 to serve
       without speculative decoding, or use a checkpoint whose MTP experts
       are NVFP4."
        fi
        ok "MTP experts quantization: ${MTP_ALGO:-unquantized} (supported)"
    fi
fi

if $DO_LAUNCH && [[ "$MTP_DISABLE_BLOCK_DROP" == "1" && "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 && "$V030" != "true" ]]; then
    info "=== Step 4h: vllm#53388 block-drop backport ==="
    BD="$SCRIPT_DIR/patches/block_drop"
    for f in $(python3 "$SCRIPT_DIR/patches/patch_block_drop.py" --list); do
        mkdir -p "$(dirname "$BD/orig/$f")"
        extract_from_image "$VLLM_PKG/$f" "$BD/orig/$f"
    done
    python3 "$SCRIPT_DIR/patches/patch_block_drop.py" || err "patch_block_drop.py failed"
    for f in config/speculative.py v1/core/kv_cache_utils.py v1/core/sched/scheduler.py; do
        [[ -f "$BD/$f" ]] && add_overlay "$BD/$f" "$VLLM_PKG/$f"
    done
fi
if $DO_LAUNCH && [[ "$VLLM_QSA_DET_TOPK" == "1" || "$VLLM_MOE_DET_FINALIZE" == "1" ]]; then
    info "=== Step 4i: reproducible greedy decoding ==="
    if [[ "$KV_CACHE_DTYPE" != fp8* ]]; then
        [[ "$QSA_PROFILE" == "stock" ]] || err "VLLM_QSA_DET_TOPK and QSA_PROFILE=$QSA_PROFILE both overlay ops/qsa.py - pick one."
        extract_from_image "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ops/qsa.py" \
                           "$SCRIPT_DIR/patches/qsa_ops_patched.py.orig"
        extract_from_image "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/qsa.py" \
                           "$SCRIPT_DIR/patches/qsa_nvidia_patched.py.orig"
        python3 "$SCRIPT_DIR/patches/patch_qsa_fp8_kv.py"
        add_overlay "$SCRIPT_DIR/patches/qsa_ops_patched.py" \
                    "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
        add_overlay "$SCRIPT_DIR/patches/qsa_nvidia_patched.py" \
                    "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/qsa.py"
    fi
    MOE_CUTLASS="model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py"
    mkdir -p "$SCRIPT_DIR/patches/determinism/orig"
    extract_from_image "$VLLM_PKG/$MOE_CUTLASS" "$SCRIPT_DIR/patches/determinism/orig/flashinfer_cutlass_moe.py"
    python3 "$SCRIPT_DIR/patches/patch_determinism.py" || err "patch_determinism.py failed"
    add_overlay "$SCRIPT_DIR/patches/determinism/flashinfer_cutlass_moe.py" "$VLLM_PKG/$MOE_CUTLASS"
    [[ "$VLLM_QSA_DET_TOPK" == "1" ]] && OVERLAY_ENV+=("-e VLLM_QSA_DET_TOPK=1")
    if [[ "$VLLM_MOE_DET_FINALIZE" == "1" ]]; then
        OVERLAY_ENV+=("-e VLLM_MOE_DET_FINALIZE=1" "-e VLLM_FLASHINFER_MOE_FUSED_FINALIZE=0")
        OVERLAY_ENV+=("-e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/root/.cache/vllm/flashinfer_autotune_cache_unfused")
    fi
fi
