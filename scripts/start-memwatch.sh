#!/usr/bin/env bash
# start-memwatch.sh [--stop] [container] — idempotent host watchdog lifecycle.
# MEMWATCH_* env knobs are inherited by the child. Requires Linux pidfds so a
# stale state file can never redirect a stop signal to a reused PID.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 - "$SCRIPT_DIR" "$@" <<'PY'
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile

scripts = Path(sys.argv[1])
args = sys.argv[2:]
stopping = bool(args and args[0] == "--stop")
if stopping:
    args.pop(0)
if len(args) > 1:
    sys.exit("Usage: start-memwatch.sh [--stop] [container]")
if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
    sys.exit("Watchdog lifecycle requires Python with Linux pidfd support")
container = args[0] if args else "vllm-fn"
if not container or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in container):
    sys.exit("Invalid container name")
os.umask(0o077)
logs = scripts.parent / "logs"
logs.mkdir(parents=True, exist_ok=True)
state_file = logs / f"memwatch-{container}.pid"
log_file = os.environ.get("MEMWATCH_LOG", str(logs / f"memwatch-{container}.log"))
script = scripts / "memwatch.sh"
boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def start_time(pid):
    return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[19]


def open_monitor(state):
    # Open the pidfd before checking identity: later signals address this exact
    # process even if it exits and the numeric PID is reused during the check.
    fd = None
    try:
        fd = os.pidfd_open(state["pid"])
        cmd = Path(f'/proc/{state["pid"]}/cmdline').read_bytes().split(b"\0")[:-1]
        legacy = set(state) == {"pid"}
        if ((legacy or (state["boot"] == boot and state["start"] == start_time(state["pid"])))
                and cmd == [b"bash", os.fsencode(script), os.fsencode(container)]):
            return fd
    except (OSError, KeyError, TypeError, ValueError, IndexError):
        pass
    if fd is not None:
        os.close(fd)
    return None


with (logs / f"memwatch-{container}.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        state = json.loads(state_file.read_text())
        if isinstance(state, int):
            state = {"pid": state}
        elif not isinstance(state, dict):
            state = {}
    except (FileNotFoundError, ValueError):
        state = {}
    fd = open_monitor(state)
    container_id = ""
    if not stopping:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.Id}} {{.State.Running}}", container],
            capture_output=True, text=True, check=True)
        container_id, running = result.stdout.strip().split()
        if running != "true":
            sys.exit(f"Cannot monitor stopped container {container}")
        if fd is not None and state.get("container_id") == container_id:
            os.close(fd)
            print(log_file)
            sys.exit(0)
    if fd is not None:
        try:
            signal.pidfd_send_signal(fd, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if not select.select([fd], [], [], int(state.get("grace", 30)) + 5)[0]:
            sys.exit("Watchdog did not stop; refusing to start a duplicate")
        os.close(fd)
    state_file.unlink(missing_ok=True)
    if stopping:
        sys.exit(0)
    env = dict(os.environ, MEMWATCH_LOG=log_file, MEMWATCH_CONTAINER_ID=container_id)
    with open(log_file, "a") as log:
        child = subprocess.Popen(["bash", str(script), container], env=env,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 start_new_session=True)
    state = dict(pid=child.pid, start=start_time(child.pid), boot=boot,
                 container_id=container_id, grace=int(env.get("MEMWATCH_GRACE", "30")))
    with tempfile.NamedTemporaryFile(mode="w", dir=logs, delete=False) as tmp:
        json.dump(state, tmp)
    os.replace(tmp.name, state_file)
    print(log_file)
PY
