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
    VLLM_ARGS+=("--max-model-len" "$MAX_MODEL_LEN")
    VLLM_ARGS+=("--kv-cache-dtype" "$KV_CACHE_DTYPE")
    [[ -n "$MAMBA_SSM_CACHE_DTYPE" ]] && VLLM_ARGS+=("--mamba-ssm-cache-dtype" "$MAMBA_SSM_CACHE_DTYPE")
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

    # JSON args: use printf to build properly quoted strings for the heredoc
    if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]]; then
        _SPEC_EXTRA=""
        [[ "$MTP_DISABLE_BLOCK_DROP" == "1" ]] && _SPEC_EXTRA+=',"disable_eagle_block_drop":true'
        [[ "$MTP_INDEX_SHARE" == "true" ]] && _SPEC_EXTRA+=',"index_share_for_mtp_iteration":true'
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
    info "  IFACE:      $IFACE"
    info "  IB_HCA:     $IB_HCA"
    info "  FP8 dense:  $FP8_DENSE   QSA profile: $QSA_PROFILE"
    info "  KV dtype:   $KV_CACHE_DTYPE   Draft vocab: ${MTP_DRAFT_VOCAB:-full}"
    info "  SSM state:  ${MAMBA_SSM_CACHE_DTYPE:-float32 (checkpoint)}"
    info "  MM encoder: $MM_ENCODER_TP_MODE tp-mode"
    [[ -n "$EXTRA_VLLM_ARGS" ]] && info "  Extra args: $EXTRA_VLLM_ARGS"
    info ""
