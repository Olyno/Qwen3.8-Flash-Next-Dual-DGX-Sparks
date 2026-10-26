#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
#
# Start (or check) one A/B arm and prove which proxy the arm uses.
#
# An A/B that cannot name its arm's configuration is not an A/B: a stale proxy
# on the port, or a restart that silently kept the previous environment, gives
# two identical arms and a null result that looks like evidence. This script
# makes that failure loud.
#
#   ./scripts/ab_proxy_arm.sh guard PROXY_EDIT_GUARD=1 PROXY_REASONING_EFFORT=medium \
#       -- python3 bench/tool_tasks.py --repeat 2
#   ./scripts/ab_proxy_arm.sh guard PROXY_EDIT_GUARD=1 --assert-only
#
# Before the run starts, the requested environment is written to
# /tmp/ab/<arm>.log. After the proxy is up, the arm passes only if every
# requested KEY=VALUE appears verbatim in the live process's environ
# (/proc/<pid>/environ) — read from the process that actually owns the port,
# not from what we believe we passed. Exit codes: 0 arm verified, 2 arm
# mismatch, 3 proxy never became ready.
set -euo pipefail

ARM="${1:-}"
[[ -n "$ARM" ]] || { echo "usage: $0 <arm-name> [ENV=VAL ...] [--port N] [--assert-only] [-- cmd...]" >&2; exit 64; }
shift

PORT="${PROXY_PORT:-8889}"
ASSERT_ONLY=0
REQUESTED=()
CMD=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --assert-only) ASSERT_ONLY=1; shift ;;
        --) shift; CMD=("$@"); break ;;
        *=*) REQUESTED+=("$1"); shift ;;
        *) echo "unexpected argument: $1" >&2; exit 64 ;;
    esac
done

LOG_DIR="${AB_LOG_DIR:-/tmp/ab}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$ARM.log"
: > "$LOG"
{
    echo "# arm=$ARM at $(date -u +%Y-%m-%dT%H:%M:%SZ) port=$PORT"
    echo "# requested: ${REQUESTED[*]:-(none)}"
} >> "$LOG"

listener_pid() {
    local pid
    pid="$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)"
    [[ -n "$pid" ]] && echo "$pid"
}

read_environ() {
    local pid="$1"
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null || true
}

assert_arm() {
    local pid="$1" kv key val missing=0
    local env_dump
    env_dump="$(read_environ "$pid")"
    if [[ -z "$env_dump" ]]; then
        echo "ARM FAILED: cannot read /proc/$pid/environ" | tee -a "$LOG" >&2
        return 2
    fi
    for kv in "${REQUESTED[@]}"; do
        key="${kv%%=*}"
        val="${kv#*=}"
        if ! grep -qx -- "$key=$val" <<<"$env_dump"; then
            local actual
            actual="$(grep -m1 "^$key=" <<<"$env_dump" || echo "$key=<unset>")"
            echo "ARM FAILED: wanted $key=$val, live process has $actual" | tee -a "$LOG" >&2
            missing=1
        fi
    done
    if [[ "$missing" -eq 0 ]]; then
        {
            echo "# live pid=$pid cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-200)"
            grep -E '^(PROXY_[A-Z_]+)=' <<<"$env_dump" | sort | sed 's/^/# live /' || true
            echo "ARM OK: $ARM verified on pid $pid"
        } >> "$LOG"
        echo "ARM OK: $ARM verified on pid $pid (log $LOG)"
        return 0
    fi
    return 2
}

pid="$(listener_pid || true)"

if [[ "$ASSERT_ONLY" -eq 1 ]]; then
    [[ -n "$pid" ]] || { echo "no listener on port $PORT" | tee -a "$LOG" >&2; exit 2; }
    assert_arm "$pid"
    exit 0
fi

if [[ -n "$pid" ]]; then
    echo "# replacing listener pid=$pid on port $PORT" >> "$LOG"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 50); do
        [[ -z "$(listener_pid || true)" ]] && break
        sleep 0.1
    done
    pid="$(listener_pid || true)"
    if [[ -n "$pid" ]]; then
        echo "ARM FAILED: port $PORT still held by pid $pid" | tee -a "$LOG" >&2
        exit 2
    fi
fi

# The proxy takes its knobs from the environment; start it with exactly the
# arm's environment plus the port, and stream its log into the arm file.
env "${REQUESTED[@]}" PROXY_PORT="$PORT" \
    python3 "$(dirname "$0")/../proxy/score_proxy.py" >>"$LOG" 2>&1 &
child=$!
echo "# started pid=$child" >> "$LOG"

for _ in $(seq 1 300); do
    if curl -sf -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then break; fi
    if ! kill -0 "$child" 2>/dev/null; then
        echo "ARM FAILED: proxy exited before becoming ready (log $LOG)" >&2
        exit 3
    fi
    sleep 0.2
done
if ! curl -sf -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then
    echo "ARM FAILED: proxy not ready on port $PORT (log $LOG)" >&2
    exit 3
fi

assert_arm "$child"

if [[ "${#CMD[@]}" -gt 0 ]]; then
    BENCH_BASE="${BENCH_BASE:-http://127.0.0.1:$PORT}" "${CMD[@]}"
fi
