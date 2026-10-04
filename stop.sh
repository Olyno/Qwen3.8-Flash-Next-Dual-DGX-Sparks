#!/usr/bin/env bash
# stop.sh — Stop the vLLM container on both head and worker nodes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f .env ]]; then
    echo "ERROR: .env not found."
    exit 1
fi

source .env

WORKER_USER="${WORKER_USER:-}"
WORKER_IP="${WORKER_IP:-}"
NODES="${NODES:-1}"
HAS_WORKER=false
[[ "$NODES" == "2" && -n "$WORKER_IP" ]] && HAS_WORKER=true
CONTAINER_NAME="vllm-fn"
NFS_CONTAINER="${NFS_CONTAINER:-vllm-fn-nfs}"
NFS_VOLUME="${NFS_VOLUME:-vllm-fn-hf}"
STOP_NFS=false

for arg in "$@"; do
    case "$arg" in
        --nfs|--all) STOP_NFS=true ;;
        -h|--help)
            echo "Usage: $0 [--nfs]"
            echo "  (default)  Stop vLLM on worker then head"
            echo "  --nfs      Also stop the head NFS share and remove the worker volume"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg (try --help)"
            exit 1
            ;;
    esac
done

ssh_cmd() {
    local user_prefix=""
    [[ -n "$WORKER_USER" ]] && user_prefix="${WORKER_USER}@"
    ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no "${user_prefix}$WORKER_IP" "$@"
}

# Stop the memory watchdog first, so it cannot emergency-stop the container
# while we are gracefully stopping it ourselves.
MEMWATCH_PIDFILE="logs/memwatch-$CONTAINER_NAME.pid"
if [[ -f "$MEMWATCH_PIDFILE" ]]; then
    kill "$(cat "$MEMWATCH_PIDFILE")" 2>/dev/null || true
    rm -f "$MEMWATCH_PIDFILE"
fi

# SIGTERM with a 30s grace, then rm: `docker rm -f` SIGKILLs, and under
# --ipc host that leaks the container's POSIX shm segments onto the host.
if $HAS_WORKER; then
    echo "Stopping $CONTAINER_NAME on worker ($WORKER_IP)..."
    ssh_cmd "if docker stop -t 30 $CONTAINER_NAME >/dev/null 2>&1; then docker rm $CONTAINER_NAME >/dev/null 2>&1; echo '  Worker: stopped.'; else echo '  Worker: not running.'; fi"
fi

echo "Stopping $CONTAINER_NAME on head..."
if docker stop -t 30 "$CONTAINER_NAME" >/dev/null 2>&1; then
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1
    echo "  Head: stopped."
else
    echo "  Head: not running."
fi

if $STOP_NFS; then
    echo "Stopping NFS share ($NFS_CONTAINER) on head..."
    echo "  (kernel NFS in Docker can ignore SIGKILL if rpcbind is in D-state; Ctrl-C and reboot if this hangs)"
    if timeout 15 docker rm -f "$NFS_CONTAINER" >/dev/null 2>&1; then
        echo "  NFS server: stopped."
    else
        echo "  NFS server: still running (could not kill). Leave it — start.sh will reuse it."
    fi
    if $HAS_WORKER; then
        echo "Removing worker NFS volume ($NFS_VOLUME)..."
        ssh_cmd "docker volume rm $NFS_VOLUME 2>/dev/null && echo '  Worker volume: removed.' || echo '  Worker volume: not present.'"
    fi
fi

echo "Done."
