#!/bin/bash
# Proves: (1) OLD bare-words form breaks under ssh re-tokenization with a
# space-bearing VLLM_ARGS; (2) NEW base64 form survives, env intact.
set -uo pipefail
WORKER_SSH=fakehost
VLLM_ARGS='--gpu-memory-utilization 0.47 --speculative-config {"method":"mtp","num_speculative_tokens":4} --hf-overrides {"text_config":{"rope_parameters":{"rope_type":"yarn"}}}'
SERVED_MODEL_NAME="Qwen3.8-Flash-Next-NVFP4"
SCRIPT_DIR=/tmp/fk3/worker-repo
RECIPE=lean-stock; CONTAINER=vllm-fn; PORT=8888; MASTER_PORT=50000
MODEL_PATH=/tmp/fk3/worker-home/models/lean-stock; MODEL_ID=x/y; HF_CACHE_DIR=/h; IMAGE=img:latest

# fake ssh models the remote shell exactly
ssh() { local h="$1"; shift; bash -c "$*"; }
# worker-side start.sh stand-in: dump the env we care about
mkdir -p $SCRIPT_DIR
cat > $SCRIPT_DIR/start.sh <<'W'
#!/bin/bash
[[ "$RECIPE" == lean-stock && "$NODE_RANK_OVERRIDE" == 1 && "$SERVED_MODEL_NAME" == Qwen3.8-Flash-Next-NVFP4 ]] || { echo "WORKER-GOT: RECIPE=$RECIPE NRO=${NODE_RANK_OVERRIDE:-} SMN=${SERVED_MODEL_NAME:-}"; exit 3; }
[[ "$VLLM_ARGS" == '--gpu-memory-utilization 0.47 --speculative-config {"method":"mtp","num_speculative_tokens":4} --hf-overrides {"text_config":{"rope_parameters":{"rope_type":"yarn"}}}' ]] || { echo "WORKER-VLLMARGS-MANGLED: [$VLLM_ARGS]"; exit 4; }
echo WORKER-ENV-INTACT
W
# --- OLD form (the field failure): expect env to receive a bare word ---
echo "== OLD =="
ssh fakehost env RECIPE="$RECIPE" VLLM_ARGS="$VLLM_ARGS" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" bash "'$SCRIPT_DIR/start.sh'" 2>&1 | tail -2
echo "old survived? rc=$?"
# --- NEW form (verbatim from start.sh) ---
echo "== NEW =="
_wscript=$(
    printf 'export RECIPE=%q RUN_WORKER=0 NODE_RANK_OVERRIDE=1\n' "$RECIPE"
    printf 'export CONTAINER=%q PORT=%q MASTER_PORT=%q\n' "$CONTAINER" "$PORT" "$MASTER_PORT"
    printf 'export MODEL_PATH=%q MODEL_ID=%q HF_CACHE_DIR=%q IMAGE=%q\n' "${MODEL_PATH:-}" "$MODEL_ID" "$HF_CACHE_DIR" "$IMAGE"
    printf 'export VLLM_ALLOW_LONG_MAX_MODEL_LEN=%q HF_OVERRIDES=%q COMPILATION_CONFIG=%q\n' "${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-}" "${HF_OVERRIDES:-}" "${COMPILATION_CONFIG:-}"
    printf 'export VLLM_ARGS=%q EXTRA_VLLM_ARGS=%q DOCKER_ARGS_EXTRA=%q SERVED_MODEL_NAME=%q\n' "$VLLM_ARGS" "${EXTRA_VLLM_ARGS:-}" "${DOCKER_ARGS_EXTRA:-}" "${SERVED_MODEL_NAME:-}"
    printf 'cd %q && exec bash ./start.sh\n' "$SCRIPT_DIR"
)
_wb64=$(printf '%s' "$_wscript" | base64 -w0)
ssh fakehost "printf %s '$_wb64' | base64 -d | bash"; echo "new-rc=$?"
