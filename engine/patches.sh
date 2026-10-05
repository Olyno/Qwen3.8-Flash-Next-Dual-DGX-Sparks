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
    [[ -f "$2" ]] && return 0
    info "  Extracting $(basename "$1") from image..."
    local c; c=$(docker create "$IMAGE" /bin/true)
    docker cp "$c:$1" "$2" >/dev/null
    docker rm "$c" >/dev/null 2>&1
    [[ -f "$2" ]] || err "Failed to extract $1 from image."
}
if $DO_LAUNCH && [[ "$FP8_DENSE" == "true" ]]; then
    info "=== Step 4c: FP8-dense overlay ==="
    OV="$SCRIPT_DIR/overlays/fp8dense"
    [[ -f "$OV/modelopt.py" ]] || python3 "$OV/apply_patches.py"
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
    [[ "$FP8_DENSE" == "true" ]] && err "V030: FP8_DENSE is not supported on the vLLM 0.30 lane."
    [[ "$QSA_PROFILE" == "stock" ]] || err "V030: QSA_PROFILE=$QSA_PROFILE is not supported on the vLLM 0.30 lane."
    if [[ "$KV_CACHE_DTYPE" == fp8* ]]; then
        mkdir -p "$SCRIPT_DIR/patches/v030_fp8kv/orig/ops"
        extract_from_image "$VLLM_PKG/models/qwen4_exp/nvidia/qsa.py" "$SCRIPT_DIR/patches/v030_fp8kv/orig/qsa.py"
        extract_from_image "$VLLM_PKG/models/qwen4_exp/nvidia/ops/qsa.py" "$SCRIPT_DIR/patches/v030_fp8kv/orig/ops/qsa.py"
        python3 "$SCRIPT_DIR/patches/patch_qsa_fp8_kv_v030.py" || err "patch_qsa_fp8_kv_v030.py failed"
        cp "$SCRIPT_DIR/patches/v030_fp8kv/qsa.py" "$SCRIPT_DIR/patches/v030_fp8kv/qsa_nvidia_v030.py"
        cp "$SCRIPT_DIR/patches/v030_fp8kv/ops/qsa.py" "$SCRIPT_DIR/patches/v030_fp8kv/qsa_ops_v030.py"
        # nvidia/qsa.py is NOT mounted here: the QSA-prepare block below
        # re-patches this output with the vllm#57097 fusion and mounts the
        # result (the two patches touch non-overlapping regions).
        add_overlay "$SCRIPT_DIR/patches/v030_fp8kv/qsa_ops_v030.py" "$VLLM_PKG/models/qwen4_exp/nvidia/ops/qsa.py"
    fi
    info "=== Step 4c: QSA prepare fusion (vllm#57097 backport, vLLM 0.30) ==="
    # Backport of vllm#57097 (patches/qsa_prepare/): the indexer prepare
    # launch also does the main attention's QK-norm/RoPE/gate and the main
    # K/V cache write (ops/qsa_pre_indexer.py becomes ops/qsa_prepare.py), so
    # _project_qkv_gate and do_kv_cache_update drop out of the hot path.
    # Quality-neutral (same math, one launch instead of three), so no toggle:
    # fused mode self-gates on use_fused_qk_norm_rope_gate and
    # indexer.use_fused_pre_indexer and otherwise runs the stock paths. With
    # KV_CACHE_DTYPE=fp8 the qsa.py input is the FP8-KV overlay output above;
    # the fused kernel writes the e4m3 cache with the same per-tensor scales.
    QP="$SCRIPT_DIR/patches/qsa_prepare"
    mkdir -p "$QP/orig"
    extract_from_image "$VLLM_PKG/models/qwen4_exp/nvidia/qsa.py" "$QP/orig/qsa_stock.py"
    extract_from_image "$VLLM_PKG/models/qwen4_exp/nvidia/indexer_qsa.py" "$QP/orig/indexer_qsa.py"
    extract_from_image "$VLLM_PKG/models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py" "$QP/orig/qsa_pre_indexer.py"
    if [[ "$KV_CACHE_DTYPE" == fp8* ]]; then
        cp "$SCRIPT_DIR/patches/v030_fp8kv/qsa_nvidia_v030.py" "$QP/orig/qsa.py"
    else
        cp "$QP/orig/qsa_stock.py" "$QP/orig/qsa.py"
    fi
    python3 "$QP/apply_patch.py" || err "qsa_prepare apply_patch.py failed"
    add_overlay "$QP/qsa_v030.py" "$VLLM_PKG/models/qwen4_exp/nvidia/qsa.py"
    add_overlay "$QP/indexer_qsa_v030.py" "$VLLM_PKG/models/qwen4_exp/nvidia/indexer_qsa.py"
    add_overlay "$QP/qsa_prepare_v030.py" "$VLLM_PKG/models/qwen4_exp/nvidia/ops/qsa_prepare.py"
    [[ "$VLLM_QSA_DET_TOPK" == "1" || "$VLLM_MOE_DET_FINALIZE" == "1" ]] && err "V030: the determinism knobs are not ported to vLLM 0.30."
    if [[ "$PLE_OFFLOAD" == "true" ]]; then
        info "=== Step 4c: PLE mmap offload (vLLM 0.30) ==="
        # v0.30's PLE CPU offload parks the table in pinned anonymous host
        # memory (~32 GiB/node). The mmap overlay reroutes it to a file-backed
        # table read over ATS (VLLM_PLE_MMAP_DIR), persisted across launches
        # under ~/.cache/vllm. Single-node Spark needs the offload to fit the
        # model at TP=1; the mmap form avoids the pin on any topology.
        mkdir -p "$SCRIPT_DIR/patches/v030_ple/orig"
        extract_from_image "$VLLM_PKG/models/qwen4_exp/nvidia/ngram_embedding.py" \
                           "$SCRIPT_DIR/patches/v030_ple/orig/ngram_embedding.py"
        python3 "$SCRIPT_DIR/patches/patch_ple_mmap_v030.py" || err "patch_ple_mmap_v030.py failed"
        add_overlay "$SCRIPT_DIR/patches/v030_ple/ngram_embedding.py" \
                    "$VLLM_PKG/models/qwen4_exp/nvidia/ngram_embedding.py"
        OVERLAY_ENV+=("-e VLLM_PLE_MMAP_DIR=/root/.cache/vllm/ple_mmap_v030")
        OVERLAY_ENV+=("-e VLLM_PLE_MMAP_ADVICE=1")
    fi
    if [[ "$SKINNY_GEMM" == "true" ]]; then
        info "=== Step 4c: GB10 skinny-GEMM plans (SM12x) ==="
        # v0.30 already ships low_latency_gemm.py, wired into model.py and
        # mtp.py, but its plan tables cover only SM103/SM90 TP=4 shapes, so on
        # the GB10 the decode-sized BF16 projections stay on cuBLAS SM80 WMMA
        # kernels. The overlay adds the SM12x table (myllmbox gb10-skinny-gemm,
        # timed at TP=2). Plans are keyed by local (N, K) shape and exact
        # token count M; a miss keeps the standard linear path, so any TP is
        # safe — at TP!=2 only the replicated projections match.
        SG="$SCRIPT_DIR/patches/gb10_skinny_gemm"
        extract_from_image "$VLLM_PKG/models/qwen4_exp/nvidia/low_latency_gemm.py" \
                           "$SG/low_latency_gemm.py.orig"
        python3 "$SG/apply_patch.py" || err "apply_patch.py (gb10_skinny_gemm) failed"
        add_overlay "$SG/low_latency_gemm.py" \
                    "$VLLM_PKG/models/qwen4_exp/nvidia/low_latency_gemm.py"
        [[ "$TENSOR_PARALLEL_SIZE" == "2" ]] || warn "SKINNY_GEMM: plans are TP=2 shapes; at TP=$TENSOR_PARALLEL_SIZE only replicated projections take the skinny path."
    fi
    if [[ "$LAZY_GDN" == "true" ]]; then
        info "=== Step 4c: K3 lazy GDN state commit (vLLM 0.30) ==="
        # Port of the sfxnz recipe's K3 overlay (patches/patch_gdn_lazy_v030.py):
        # the stock GDN verify kernel writes the 64 KiB fp32 state after every
        # token; K3 commits once per step and replays exact token inputs from a
        # ring. Bitwise-pinned to stock by a load-time self-test (fail closed).
        [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]] || err "lazy_gdn needs speculative decoding (MTP_NUM_SPECULATIVE_TOKENS > 0)."
        [[ "$MTP_NUM_SPECULATIVE_TOKENS" -le 7 ]] || err "lazy_gdn supports k <= 7 speculative tokens (W=k+1 <= 8); got $MTP_NUM_SPECULATIVE_TOKENS."
        [[ -z "$MAMBA_SSM_CACHE_DTYPE" || "$MAMBA_SSM_CACHE_DTYPE" == "float32" ]] || err "lazy_gdn requires an fp32 GDN state: MAMBA_SSM_CACHE_DTYPE=$MAMBA_SSM_CACHE_DTYPE
       is incompatible (the K3 kernel is bitwise-pinned to the stock fp32 kernel and
       would stay off anyway). Drop mamba_ssm_cache_dtype or disable lazy_gdn."
        mkdir -p "$SCRIPT_DIR/patches/v030_gdn/orig"
        extract_from_image "$VLLM_PKG/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py" \
                           "$SCRIPT_DIR/patches/v030_gdn/orig/qwen_gdn_linear_attn.py"
        extract_from_image "$VLLM_PKG/v1/attention/backends/gdn_attn.py" \
                           "$SCRIPT_DIR/patches/v030_gdn/orig/gdn_attn.py"
        python3 "$SCRIPT_DIR/patches/patch_gdn_lazy_v030.py" || err "patch_gdn_lazy_v030.py failed"
        add_overlay "$SCRIPT_DIR/patches/v030_gdn/gdn_lazy_linear_attn_v030.py" \
                    "$VLLM_PKG/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
        add_overlay "$SCRIPT_DIR/patches/v030_gdn/gdn_lazy_attn_v030.py" \
                    "$VLLM_PKG/v1/attention/backends/gdn_attn.py"
        OVERLAY_ENV+=("-e VLLM_QWEN38_GDN_LAZY=1")
    fi
    if [[ "$REPLAYSSM_GDN" == "true" ]]; then
        info "=== Step 4c: ReplaySSM-GDN spec decode (vLLM 0.30) ==="
        # Port of vllm#47576's GDN variant (patches/replayssm_gdn/): spec
        # verify runs a Triton kernel over an fp32 checkpoint + circular d/k/g
        # rings instead of the per-token state slots; the block-keyed cursors
        # live in the GDN metadata builder. python-only overlay (2 vendored
        # Triton files + 6 patched stock files), env-gated, spec-only.
        [[ "$LAZY_GDN" != "true" ]] || err "replayssm_gdn and lazy_gdn are mutually exclusive (same target files)."
        [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]] || err "replayssm_gdn needs speculative decoding (MTP_NUM_SPECULATIVE_TOKENS > 0)."
        [[ "$REPLAYSSM_GDN_BUFFER_LEN" =~ ^[0-9]+$ ]] || err "replayssm_gdn_buffer_len must be an integer (got: '$REPLAYSSM_GDN_BUFFER_LEN')"
        [[ "$REPLAYSSM_GDN_BUFFER_LEN" -gt "$MTP_NUM_SPECULATIVE_TOKENS" ]] || err "replayssm_gdn_buffer_len must be >= 1 + MTP_NUM_SPECULATIVE_TOKENS ($MTP_NUM_SPECULATIVE_TOKENS); got $REPLAYSSM_GDN_BUFFER_LEN."
        if [[ -n "$MAMBA_SSM_CACHE_DTYPE" && "$MAMBA_SSM_CACHE_DTYPE" != "float32" ]]; then
            warn "replayssm_gdn forces an fp32 GDN checkpoint regardless of MAMBA_SSM_CACHE_DTYPE=$MAMBA_SSM_CACHE_DTYPE (numerically >= the bf16 baseline)."
        fi
        RG="$SCRIPT_DIR/patches/replayssm_gdn"
        mkdir -p "$RG/orig"
        extract_from_image "$VLLM_PKG/model_executor/layers/mamba/mamba_utils.py" \
                           "$RG/orig/mamba_utils.py"
        extract_from_image "$VLLM_PKG/model_executor/layers/mamba/gdn/base.py" \
                           "$RG/orig/base.py"
        extract_from_image "$VLLM_PKG/model_executor/layers/mamba/abstract.py" \
                           "$RG/orig/abstract.py"
        extract_from_image "$VLLM_PKG/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py" \
                           "$RG/orig/qwen_gdn_linear_attn.py"
        extract_from_image "$VLLM_PKG/v1/attention/backends/gdn_attn.py" \
                           "$RG/orig/gdn_attn.py"
        extract_from_image "$VLLM_PKG/models/qwen4_exp/nvidia/model.py" \
                           "$RG/orig/model.py"
        extract_from_image "$VLLM_PKG/model_executor/layers/mamba/ops/replayssm_config.py" \
                           "$RG/orig/replayssm_config.py"
        python3 "$RG/apply_patch.py" || err "replayssm_gdn apply_patch.py failed"
        add_overlay "$RG/mamba_utils_v030.py" \
                    "$VLLM_PKG/model_executor/layers/mamba/mamba_utils.py"
        add_overlay "$RG/gdn_base_v030.py" \
                    "$VLLM_PKG/model_executor/layers/mamba/gdn/base.py"
        add_overlay "$RG/mamba_abstract_v030.py" \
                    "$VLLM_PKG/model_executor/layers/mamba/abstract.py"
        add_overlay "$RG/qwen_gdn_linear_attn_v030.py" \
                    "$VLLM_PKG/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
        add_overlay "$RG/gdn_attn_v030.py" \
                    "$VLLM_PKG/v1/attention/backends/gdn_attn.py"
        add_overlay "$RG/qwen4_exp_model_v030.py" \
                    "$VLLM_PKG/models/qwen4_exp/nvidia/model.py"
        add_overlay "$RG/gdn_replayssm_spec_decode.py" \
                    "$VLLM_PKG/model_executor/layers/mamba/ops/gdn_replayssm_spec_decode.py"
        add_overlay "$RG/replayssm_config.py" \
                    "$VLLM_PKG/model_executor/layers/mamba/ops/replayssm_config.py"
        OVERLAY_ENV+=("-e VLLM_REPLAYSSM_GDN=1")
        OVERLAY_ENV+=("-e VLLM_REPLAYSSM_GDN_BUFFER_LEN=$REPLAYSSM_GDN_BUFFER_LEN")
    fi
    if [[ "$QSA_FUSED_DRAFT" == "true" ]]; then
        info "=== Step 4c: QSA fused multi-step draft metadata (vLLM 0.30) ==="
        # Port of myllmbox/vllm@c3f56fe (the code proposed upstream as
        # vllm#58449, patches/patch_qsa_fused_draft_v030.py): stock v0.30
        # rebuilds attention metadata for the whole model between MTP draft
        # steps because the QSA builder never opted into the speculator's
        # in-place update; the overlay re-launches the builder's own metadata
        # kernel on its persistent buffers instead.
        [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 1 ]] || err "qsa_fused_draft needs MTP_NUM_SPECULATIVE_TOKENS > 1 (with k <= 1 there is no inter-draft-step rebuild to fuse)."
        mkdir -p "$SCRIPT_DIR/patches/v030_qsa_fused/orig"
        extract_from_image "$VLLM_PKG/models/qwen4_exp/common/qsa_cache.py" \
                           "$SCRIPT_DIR/patches/v030_qsa_fused/orig/qsa_cache.py"
        python3 "$SCRIPT_DIR/patches/patch_qsa_fused_draft_v030.py" || err "patch_qsa_fused_draft_v030.py failed"
        add_overlay "$SCRIPT_DIR/patches/v030_qsa_fused/qsa_cache_v030.py" \
                    "$VLLM_PKG/models/qwen4_exp/common/qsa_cache.py"
        OVERLAY_ENV+=("-e VLLM_QSA_FUSED_DRAFT=1")
    fi
    if [[ "$QSA_ROPE_CLAMP" == "true" ]]; then
        info "=== Step 4c: QSA pre-indexer RoPE clamp (vLLM 0.30) ==="
        # The clamp (myllmbox/vllm@9ff17c0) now lives inside the qsa_prepare
        # overlay above: the vllm#57097 backport replaces qsa_pre_indexer.py
        # outright, so patches/patch_qsa_rope_clamp_v030.py's CLAMP_POS/
        # MAX_POS deltas are folded into patches/qsa_prepare (main-attention
        # section included), behind the same VLLM_QSA_ROPE_CLAMP constexpr
        # gate — bit-exact stock kernel when off. Only the env flag is set
        # here; the standalone patcher is kept for reference but no longer
        # wired.
        OVERLAY_ENV+=("-e VLLM_QSA_ROPE_CLAMP=1")
    fi
    if [[ "$LOAD_DROP_CACHE" == "true" ]]; then
        info "=== Step 4c: loader page-cache drop (vLLM 0.30) ==="
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
    OVERLAY_ENV+=("-e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/tmp/fi_autotune")
    OVERLAY_ENV+=("-e VLLM_USE_BREAKABLE_CUDAGRAPH=${V030_BREAKABLE_CUDAGRAPH:-0}")
fi
if $DO_LAUNCH && [[ "$V030" == "true" ]] && { [[ -n "$MTP_DRAFT_VOCAB" ]] || [[ "$FP8_DRAFT_HEAD" == "true" ]]; }; then
    info "=== Step 4e: MTP draft head (vLLM 0.30) ==="
    [[ "$MTP_NUM_SPECULATIVE_TOKENS" == "0" ]] && err "MTP_DRAFT_VOCAB/FP8_DRAFT_HEAD is set but MTP_NUM_SPECULATIVE_TOKENS=0 - nothing drafts."
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
    add_overlay "$SCRIPT_DIR/patches/mtp_v030_patched.py" \
                "$VLLM_PKG/models/qwen4_exp/nvidia/mtp.py"
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
