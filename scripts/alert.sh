#!/usr/bin/env bash
# alert.sh <message> — POST one alert to ALERT_WEBHOOK_URL (.env). A generic
# webhook: any URL that accepts {hostname, timestamp, message, container,
# mem_available}.
#
# No-op with a note on stderr when ALERT_WEBHOOK_URL is unset (out-of-the-box
# stays silent-safe). Identical messages collapse to one per 15 min (state
# file), so a breaker-open loop cannot spam. An alerting failure NEVER
# changes control flow: log and return 0.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
MESSAGE="${1:-no message}"

if [[ -f "$REPO_DIR/.env" ]]; then
    # shellcheck source=.env
    source "$REPO_DIR/.env"
fi
ALERT_WEBHOOK_URL="${ALERT_WEBHOOK_URL:-}"
RATE_STATE="${ALERT_STATE_FILE:-${TMPDIR:-/tmp}/qwen38-flash-alert.state}"
RATE_SECS="${ALERT_RATE_SECS:-900}"

if [[ -z "$ALERT_WEBHOOK_URL" ]]; then
    echo "[alert] ALERT_WEBHOOK_URL unset; not sending: $MESSAGE" >&2
    exit 0
fi

# Rate-limit: identical messages collapse to one per window. The state file
# holds "<hash> <ts>"; a different message is never blocked by an earlier one.
_now=$(date +%s)
_msg_hash=$(printf '%s' "$MESSAGE" | sha256sum | cut -c1-16)
_last_hash=""
_last_ts=""
if [[ -s "$RATE_STATE" ]]; then
    read -r _last_hash _last_ts < "$RATE_STATE" 2>/dev/null || true
fi
if [[ -n "$_last_hash" && "$_last_hash" == "$_msg_hash" && -n "$_last_ts" ]] \
        && (( _now - _last_ts < RATE_SECS )); then
    exit 0
fi

_container=""
if command -v docker >/dev/null 2>&1; then
    _container=$(docker ps --format '{{.Names}}' 2>/dev/null | tr '\n' ' ')
fi
_mem=""
if [[ -r /proc/meminfo ]]; then
    _mem=$(grep MemAvailable /proc/meminfo | awk '{printf "%.1f GiB", $2/1048576}')
fi

_payload=$(python3 -c '
import json, sys
print(json.dumps({
    "hostname": __import__("socket").gethostname(),
    "timestamp": __import__("datetime").datetime.now().isoformat(),
    "message": sys.argv[1],
    "container": sys.argv[2],
    "mem_available": sys.argv[3],
}))
' "$MESSAGE" "$_container" "$_mem")

if curl -s -m 10 -H 'Content-Type: application/json' \
        -d "$_payload" "$ALERT_WEBHOOK_URL" >/dev/null 2>&1; then
    # Record the rate-limit state only after a successful delivery: a webhook
    # outage must not mark the message as sent (an identical retry can then
    # fire once the endpoint recovers).
    printf '%s %s\n' "$_msg_hash" "$_now" > "$RATE_STATE" 2>/dev/null || true
else
    echo "[alert] webhook POST failed: $MESSAGE" >&2
fi
exit 0
