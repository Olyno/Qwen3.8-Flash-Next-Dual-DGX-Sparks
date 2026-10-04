#!/usr/bin/env bash
# start-memwatch.sh [container] — start scripts/memwatch.sh in the background
# (nohup), record its pid in logs/memwatch-<container>.pid for stop.sh, and
# print the log path. MEMWATCH_* env knobs are inherited by the child.
set -euo pipefail

CONTAINER="${1:-vllm-fn}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
MEMWATCH_LOG="${MEMWATCH_LOG:-$REPO_DIR/logs/memwatch-${CONTAINER}.log}"
PIDFILE="$REPO_DIR/logs/memwatch-${CONTAINER}.pid"

mkdir -p "$REPO_DIR/logs/archive"
# Replace a still-running watchdog, never error on a stale pidfile.
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
fi

MEMWATCH_LOG="$MEMWATCH_LOG" \
    nohup bash "$SCRIPT_DIR/memwatch.sh" "$CONTAINER" \
    > "$MEMWATCH_LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "$MEMWATCH_LOG"
