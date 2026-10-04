#!/usr/bin/env bash
# smoke-test.sh — one short chat completion against the running server; fails
# when the response does not parse or carries no content.
#
# Usage:
#   ./scripts/smoke-test.sh                      # localhost:8888, no auth
#   PORT=9000 API_KEY=xyz ./scripts/smoke-test.sh
#
# Reads .env (repo-relative) for PORT/API_KEY when the caller did not set
# them; environment wins, matching start.sh's precedence.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
_CLI_PORT="${PORT:-}"
_CLI_API_KEY="${API_KEY:-}"
if [[ -f "$REPO_DIR/.env" ]]; then
    # shellcheck source=.env
    source "$REPO_DIR/.env"
fi
[[ -n "$_CLI_PORT" ]] && PORT="$_CLI_PORT"
[[ -n "$_CLI_API_KEY" ]] && API_KEY="$_CLI_API_KEY"

PORT="${PORT:-8888}"
MODEL="${SERVED_MODEL_NAME:-Qwen3.8-Flash-Next-NVFP4}"
BASE="http://localhost:$PORT"
AUTH=(); [[ -n "${API_KEY:-}" ]] && AUTH=(-H "Authorization: Bearer $API_KEY")

RESP=$(curl -s -m 120 "${AUTH[@]}" -H 'Content-Type: application/json' \
    "$BASE/v1/chat/completions" -d "{
  \"model\": \"$MODEL\", \"temperature\": 0, \"max_tokens\": 32,
  \"messages\": [{\"role\":\"user\",\"content\":\"Say hello in one word.\"}]}")
# The model reasons first in a `reasoning` field; content may lag behind, so
# either field counts as a non-empty answer.
echo "$RESP" | python3 -c 'import json,sys
m = json.load(sys.stdin)["choices"][0]["message"]
assert (m.get("content") or m.get("reasoning")), "empty content"' 2>/dev/null \
    && echo "PASS: chat completion on $BASE returned content" \
    || { echo "FAIL: no usable response from $BASE/v1/chat/completions" >&2
         echo "${RESP:0:300}" >&2; exit 1; }
