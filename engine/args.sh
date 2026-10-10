    # ---------------------------------------------------------------------------
    # 7. Build vLLM args (shared between head and worker)
    # ---------------------------------------------------------------------------
    info "=== Step 7: Launch vLLM ==="

    VLLM_ARGS=()
    VLLM_ARGS+=("--enable-prompt-tokens-details")
    VLLM_ARGS+=("--served-model-name" "$SERVED_MODEL_NAME")
    [[ -n "$CHAT_TEMPLATE" ]] && VLLM_ARGS+=("--chat-template" "/chat_template.jinja")
    VLLM_ARGS+=("--tensor-parallel-size" "$TENSOR_PARALLEL_SIZE")
    VLLM_ARGS+=("--gpu-memory-utilization" "$GPU_MEMORY_UTILIZATION")
    VLLM_ARGS+=("--max-num-seqs" "$MAX_NUM_SEQS")
    VLLM_ARGS+=("--max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS")
    # Chunks long prefills below MAX_NUM_BATCHED_TOKENS: a full-budget 48K
    # prefill spikes ~1.3 GiB in one step and trips the memwatch floor (sfxnz
    # runs 4800 on this hardware).
    [[ -n "$LONG_PREFILL_TOKEN_THRESHOLD" ]] && VLLM_ARGS+=("--long-prefill-token-threshold" "$LONG_PREFILL_TOKEN_THRESHOLD")
    VLLM_ARGS+=("--max-model-len" "$MAX_MODEL_LEN")
    VLLM_ARGS+=("--kv-cache-dtype" "$KV_CACHE_DTYPE")
    # Forced attention block size (recipe key attention_block_size): the engine
    # only ever raises a smaller value back to the derived one; mtp_block.py
    # above validates the ring-capacity legality of the forced size.
    [[ -n "$ATTENTION_BLOCK_SIZE" ]] && VLLM_ARGS+=("--block-size" "$ATTENTION_BLOCK_SIZE")
    [[ -n "$MAMBA_SSM_CACHE_DTYPE" ]] && VLLM_ARGS+=("--mamba-ssm-cache-dtype" "$MAMBA_SSM_CACHE_DTYPE")
    [[ -n "$GDN_PREFILL_BACKEND" ]] && VLLM_ARGS+=("--gdn-prefill-backend" "$GDN_PREFILL_BACKEND")
    VLLM_ARGS+=("--load-format" "safetensors")
    VLLM_ARGS+=("--safetensors-load-strategy" "lazy")
    VLLM_ARGS+=("--enable-chunked-prefill")
    VLLM_ARGS+=("--reasoning-parser" "qwen3")
    VLLM_ARGS+=("--enable-auto-tool-choice")
    VLLM_ARGS+=("--tool-call-parser" "qwen3_coder")
    VLLM_ARGS+=("--distributed-executor-backend" "mp")
    VLLM_ARGS+=("--mm-encoder-tp-mode" "$MM_ENCODER_TP_MODE")
    VLLM_ARGS+=("--nnodes" "$NNODES")
    VLLM_ARGS+=("--master-addr" "$HEAD_IP")
    VLLM_ARGS+=("--master-port" "$MASTER_PORT")

    if [[ "$ENABLE_EXPERT_PARALLEL" == "true" ]]; then
        VLLM_ARGS+=("--enable-expert-parallel")
        VLLM_ARGS+=("--all2all-backend" "allgather_reducescatter")
    fi

    # Refuse illegal MTP configs before weight loading (scripts/mtp_block.py:
    # ring-capacity legality of k against the block the engine derives for
    # this k + dtypes, k=1 dominated, widest verify batch vs token budget).
    if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]]; then
        python3 "$SCRIPT_DIR/scripts/mtp_block.py" "$PLE_CONFIG_DIR/config.json" \
            "$MTP_NUM_SPECULATIVE_TOKENS" "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS" \
            "$MAMBA_SSM_CACHE_DTYPE" "$KV_CACHE_DTYPE" $ATTENTION_BLOCK_SIZE || err "illegal MTP config (see above)"
    fi

    # JSON args: use printf to build properly quoted strings for the heredoc
    if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]]; then
        _SPEC_EXTRA=""
        [[ "$MTP_DISABLE_BLOCK_DROP" == "1" ]] && _SPEC_EXTRA+=',"disable_eagle_block_drop":true'
        [[ "$MTP_INDEX_SHARE" == "true" ]] && _SPEC_EXTRA+=',"index_share_for_mtp_iteration":true'
        [[ -n "$MTP_REJECTION_SAMPLE_METHOD" ]] && _SPEC_EXTRA+=",\"rejection_sample_method\":\"$MTP_REJECTION_SAMPLE_METHOD\""
        if [[ -n "$MTP_DRAFT_SAMPLE_METHOD" ]]; then
            # vLLM rejects probabilistic drafting with use_local_argmax_reduction,
            # which the reduced draft vocab relies on.
            [[ -z "$MTP_DRAFT_VOCAB" ]] || err "mtp_draft_sample_method requires mtp_draft_vocab to be unset"
            _SPEC_EXTRA+=",\"draft_sample_method\":\"$MTP_DRAFT_SAMPLE_METHOD\""
        fi
        [[ -n "$DRAFT_MOE_BACKEND" ]] && _SPEC_EXTRA+=",\"moe_backend\":\"$DRAFT_MOE_BACKEND\""
        if [[ -n "$MTP_DRAFT_VOCAB" ]]; then
            # get_top_tokens (added by patch_mtp_draft_vocab.py) is only reached
            # through this flag; it also cuts the draft all-gather from
            # O(vocab_size) to O(2*tp_size) per token.
            VLLM_ARGS+=("--speculative-config" "$(printf "'{\"method\":\"mtp\",\"num_speculative_tokens\":%s,\"use_local_argmax_reduction\":true%s}'" "$MTP_NUM_SPECULATIVE_TOKENS" "$_SPEC_EXTRA")")
        else
            VLLM_ARGS+=("--speculative-config" "$(printf "'{\"method\":\"mtp\",\"num_speculative_tokens\":%s%s}'" "$MTP_NUM_SPECULATIVE_TOKENS" "$_SPEC_EXTRA")")
        fi
    fi

    VLLM_ARGS+=("--compilation-config" "$(printf "'{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'")")
    [[ -n "$MOE_BACKEND" ]] && VLLM_ARGS+=("--kernel-config" "$(printf "'{\"moe_backend\":\"%s\"}'" "$MOE_BACKEND")")
    # Known cross-node hazard (vllm#46253): CUDA-graph capture can die with an
    # illegal memory access at capture_end on multi-node. If boot crashes
    # there, rerun with EXTRA_VLLM_ARGS="--enforce-eager".
    [[ "$NNODES" -eq 2 ]] && warn "dual-node: if boot dies at CUDA-graph capture (IMA at capture_end, vllm#46253), rerun with EXTRA_VLLM_ARGS=\"--enforce-eager\""
    [[ "$ASYNC_SCHEDULING" == "true" ]] && VLLM_ARGS+=("--async-scheduling")

    # hf-overrides: ONE merged payload, nested under "text_config".
    # vLLM's ModelConfig._apply_dict_overrides only recurses into keys that are
    # themselves nested configs. For qwen4_exp the parent config also exposes a
    # plain `rope_parameters` dict, so a top-level {"rope_parameters":...} is
    # setattr'd onto the parent and NEVER reaches text_config -- i.e. YaRN was
    # silently a no-op. Everything the model reads lives under text_config, so
    # nest both the rope override and the PLE dtype there.
    [[ "$V030" == "true" ]] && PLE_EMBEDDING_DTYPE=""
    HF_OVERRIDES_JSON=$(
        PLE_DTYPE="$PLE_EMBEDDING_DTYPE" \
        YARN="$YARN_ENABLE" YARN_FACTOR="${YARN_FACTOR:-}" \
        python3 -c '
import json, os
tc = {}
if os.environ.get("PLE_DTYPE"):
    tc["ple_embedding_dtype"] = os.environ["PLE_DTYPE"]
if os.environ.get("YARN") == "true":
    tc["rope_parameters"] = {
        "rope_type": "yarn",
        "factor": float(os.environ["YARN_FACTOR"]),
        "original_max_position_embeddings": 262144,
    }
print(json.dumps({"text_config": tc}, separators=(",", ":")) if tc else "")
'
    )
    if [[ -n "$HF_OVERRIDES_JSON" ]]; then
        VLLM_ARGS+=("--hf-overrides" "'$HF_OVERRIDES_JSON'")
    fi

    # EXTRA_VLLM_ARGS is appended LAST and verbatim (quote JSON values with single
    # quotes exactly as you would on a shell command line).
    if [[ -n "$EXTRA_VLLM_ARGS" ]]; then
        VLLM_ARGS+=("$EXTRA_VLLM_ARGS")
    fi
    # One rendered string shared by the worker and head launch scripts. JSON
    # values already carry their own single quotes (see printf above).
    VLLM_ARGS_STR="${VLLM_ARGS[*]}"
    OVERLAY_ENV_STR="${OVERLAY_ENV[*]:-}"

    # -----------------------------------------------------------------------
    # Launch: worker (rank 1) first, then head (rank 0).
    # The head node serves the API; the worker runs headless.
    # -----------------------------------------------------------------------

    info ""
    info "Config:"
    info "  Model:      $MODEL_ID"
    if [[ -n "$MODEL_PATH" ]]; then
        info "  Model path: $MODEL_PATH (bind-mounted at /model)"
    fi
    if [[ "$ABLIT" == "1" && "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
        info "  ABLIT:      1 (gated Keys house QSA L3-47)"
    else
        info "  ABLIT:      $ABLIT"
    fi
    if [[ "$NFS_SHARE" == "true" ]]; then
        info "  Weights:    NFS from $NFS_SERVER_IP (head cache, no worker copy)"
    elif [[ "$NNODES" -eq 2 ]]; then
        info "  Weights:    local copy on each node (worker copy synced from head)"
    else
        info "  Weights:    local (single node)"
    fi
    info "  Image:      $IMAGE"
    if [[ "$NNODES" -eq 2 ]]; then
        info "  Nodes:      $HEAD_IP (head, rank 0) + $WORKER_IP (worker, rank 1)"
    else
        info "  Nodes:      $HEAD_IP (single node)"
    fi
    info "  TP=$TENSOR_PARALLEL_SIZE  EP=$( [[ "$ENABLE_EXPERT_PARALLEL" == "true" ]] && echo on || echo off )  MTP=$MTP_NUM_SPECULATIVE_TOKENS"
    info "  Context:    $MAX_MODEL_LEN tokens"
    info "  GMU:        $GPU_MEMORY_UTILIZATION"
    info "  Max seqs:   $MAX_NUM_SEQS"
    info "  Port:       $PORT"
    if [[ "$NNODES" -eq 2 ]]; then
        info "  IFACE:      $IFACE"
        info "  IB_HCA:     $IB_HCA"
    fi
    info "  FP8 dense:  $FP8_DENSE   QSA profile: $QSA_PROFILE"
    info "  KV dtype:   $KV_CACHE_DTYPE   Draft vocab: ${MTP_DRAFT_VOCAB:-full}"
    info "  SSM state:  ${MAMBA_SSM_CACHE_DTYPE:-float32 (checkpoint)}"
    info "  MM encoder: $MM_ENCODER_TP_MODE tp-mode"
    [[ -n "$EXTRA_VLLM_ARGS" ]] && info "  Extra args: $EXTRA_VLLM_ARGS"
    info ""
