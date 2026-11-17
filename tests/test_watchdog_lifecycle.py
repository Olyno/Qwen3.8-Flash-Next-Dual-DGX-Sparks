#!/usr/bin/env python3
"""CPU-only watchdog lifecycle and launcher ordering regressions."""
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

REPO = Path(__file__).resolve().parent.parent


def executable(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    path.chmod(0o700)


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Watchdog lifecycle did not settle")


@unittest.skipUnless(hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"),
                     "Linux pidfds required")
class TestWatchdogLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scripts = self.root / "scripts"
        self.scripts.mkdir()
        for name in ("start-memwatch.sh", "memwatch.sh"):
            shutil.copy2(REPO / "scripts" / name, self.scripts / name)
        shutil.copy2(REPO / "stop.sh", self.root / "stop.sh")
        (self.root / ".env").write_text("NODES=1\n")
        self.state = self.root / "container.json"
        self.state.write_text(json.dumps(dict(id="container-one", running=True)))
        self.events = self.root / "events"
        self.events.touch()
        self.pidfile = self.root / "logs/memwatch-vllm-fn.pid"
        self.env = dict(os.environ, PATH=str(self.root / "bin") + ":" + os.environ["PATH"],
                        FAKE_STATE=str(self.state), EVENTS=str(self.events),
                        WATCH_STATE=str(self.pidfile), MEMWATCH_MIN_GIB="0",
                        MEMWATCH_MIN_FREE_GIB="0", MEMWATCH_PSI_FULL="100",
                        MEMWATCH_LOG=str(self.root / "watch.log"))
        executable(self.root / "bin/docker", f"#!{sys.executable}\n" + '''
import json
import os
from pathlib import Path
import select
import sys
state = json.loads(Path(os.environ["FAKE_STATE"]).read_text())
args = sys.argv[1:]
if args[0] == "inspect":
    target = args[-1]
    if target not in ("vllm-fn", state["id"]):
        sys.exit(1)
    if args[2] == "{{.Id}} {{.State.Running}}":
        print(state["id"], str(state["running"]).lower())
    elif args[2] == "{{.Id}}":
        print(state["id"])
    else:
        print(str(state["running"]).lower())
else:
    with open(os.environ["EVENTS"], "a") as events:
        events.write(" ".join(args) + "\\n")
        if args[0] == "stop" and Path(os.environ["WATCH_STATE"]).exists():
            saved = json.loads(Path(os.environ["WATCH_STATE"]).read_text())
            try:
                fd = os.pidfd_open(saved["pid"])
                if not select.select([fd], [], [], 0)[0]:
                    events.write("monitor-alive-at-container-stop\\n")
                os.close(fd)
            except ProcessLookupError:
                pass
''')
        self.addCleanup(self.helper, "--stop")

    def helper(self, *args, **env):
        return subprocess.run(["bash", str(self.scripts / "start-memwatch.sh"), *args],
                              env=dict(self.env, **env), capture_output=True,
                              text=True, timeout=10)

    def start(self, **env):
        out = self.helper(**env)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        state = json.loads(self.pidfile.read_text())
        wait_until(lambda: "watchdog start:" in (self.root / "watch.log").read_text())
        return state

    def alive(self, pid):
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            return False
        try:
            return not select.select([fd], [], [], 0)[0]
        finally:
            os.close(fd)

    def test_repeated_and_concurrent_starts_keep_one_monitor(self):
        first = self.start()
        jobs = [subprocess.Popen(["bash", str(self.scripts / "start-memwatch.sh")],
                                 env=self.env, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True) for _ in range(3)]
        for job in jobs:
            stdout, stderr = job.communicate(timeout=10)
            self.assertEqual(job.returncode, 0, stdout + stderr)
        self.assertEqual(json.loads(self.pidfile.read_text())["pid"], first["pid"])
        self.assertEqual((self.root / "watch.log").read_text().count("watchdog start:"), 1)

    def test_stop_sh_stops_monitor_before_graceful_container_stop(self):
        state = self.start()
        out = subprocess.run(["bash", str(self.root / "stop.sh")], env=self.env,
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertFalse(self.alive(state["pid"]))
        self.assertFalse(self.pidfile.exists())
        events = self.events.read_text()
        self.assertIn("stop -t 30 vllm-fn", events)
        self.assertNotIn("monitor-alive-at-container-stop", events)
        self.assertNotIn("WATCHDOG EMERGENCY STOP", (self.root / "watch.log").read_text())

    def test_legacy_stale_pid_and_reused_identity_never_kill_unrelated_process(self):
        other = subprocess.Popen(["sleep", "60"])
        self.addCleanup(other.wait)
        self.addCleanup(other.terminate)
        self.pidfile.parent.mkdir(parents=True)
        for state in (str(other.pid), json.dumps(dict(pid=other.pid, boot="old-boot",
                                                     start="0", container_id="container-one")),
                      "corrupt state"):
            with self.subTest(state=state):
                self.pidfile.write_text(state)
                out = self.helper("--stop")
                self.assertEqual(out.returncode, 0, out.stderr)
                self.assertIsNone(other.poll())
                self.assertFalse(self.pidfile.exists())

    def test_start_time_mismatch_does_not_signal_live_monitor(self):
        state = self.start()
        self.pidfile.write_text(json.dumps(dict(state, start="0")))
        out = self.helper("--stop")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertTrue(self.alive(state["pid"]))
        self.pidfile.write_text(json.dumps(state))

    def test_verified_legacy_monitor_is_replaced_without_duplicate(self):
        state = self.start()
        self.pidfile.write_text(str(state["pid"]))
        new = self.start()
        self.assertNotEqual(new["pid"], state["pid"])
        self.assertFalse(self.alive(state["pid"]))

    def test_stopped_or_missing_container_does_not_start_monitor(self):
        self.state.write_text(json.dumps(dict(id="container-one", running=False)))
        out = self.helper()
        self.assertNotEqual(out.returncode, 0)
        self.assertFalse(self.pidfile.exists())
        self.state.write_text(json.dumps(dict(id="other-name", running=True)))
        out = self.helper("missing-container")
        self.assertNotEqual(out.returncode, 0)
        self.assertFalse((self.root / "logs/memwatch-missing-container.pid").exists())

    def test_monitor_exits_when_container_id_is_replaced(self):
        old = self.start()
        self.state.write_text(json.dumps(dict(id="container-two", running=True)))
        wait_until(lambda: not self.alive(old["pid"]))
        new = self.start()
        self.assertEqual(new["container_id"], "container-two")
        self.assertNotEqual(new["pid"], old["pid"])
        self.assertNotIn("stop ", self.events.read_text())

    def test_emergency_stop_targets_pinned_id_and_keeps_grace(self):
        self.start(MEMWATCH_MIN_GIB="999999", MEMWATCH_CONSEC="1")
        wait_until(lambda: "stop -t 30 container-one" in self.events.read_text())
        events = self.events.read_text()
        self.assertIn("logs --tail 3000 container-one", events)
        self.assertNotIn("stop -t 30 vllm-fn", events)

    def test_dual_node_stop_uses_worker_state_without_a_remote_checkout(self):
        head = self.start()
        worker_home = self.root / "worker"
        worker_scripts = worker_home / ".local/state/vllm-memwatch/scripts"
        worker_scripts.mkdir(parents=True)
        for name in ("start-memwatch.sh", "memwatch.sh"):
            shutil.copy2(self.scripts / name, worker_scripts / name)
        worker_pidfile = worker_scripts.parent / "logs/memwatch-vllm-fn.pid"
        worker_env = dict(self.env, HOME=str(worker_home),
                          WATCH_STATE=str(worker_pidfile),
                          MEMWATCH_LOG=str(worker_home / "watch.log"))
        worker_command = ["bash", str(worker_scripts / "start-memwatch.sh")]
        self.addCleanup(subprocess.run, [*worker_command, "--stop"], env=worker_env,
                        capture_output=True, timeout=10)
        started = subprocess.run(worker_command, env=worker_env, capture_output=True,
                                 text=True, timeout=10)
        self.assertEqual(started.returncode, 0, started.stdout + started.stderr)
        worker = json.loads(worker_pidfile.read_text())
        (self.root / ".env").write_text("NODES=2\nWORKER_IP=fake-worker\n")
        executable(self.root / "bin/ssh", f"#!{sys.executable}\n" + '''
import os
import subprocess
import sys
args = sys.argv[1:]
while args and args[0] == "-o":
    args = args[2:]
args.pop(0)
env = dict(os.environ, HOME=os.environ["WORKER_HOME"], WATCH_STATE=os.environ["WORKER_STATE"])
sys.exit(subprocess.run(["bash", "-c", " ".join(args)], env=env).returncode)
''')
        out = subprocess.run(["bash", str(self.root / "stop.sh")],
                             env=dict(self.env, WORKER_HOME=str(worker_home),
                                      WORKER_STATE=str(worker_pidfile)),
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertFalse(self.alive(head["pid"]))
        self.assertFalse(self.alive(worker["pid"]))
        self.assertFalse(worker_pidfile.exists())
        self.assertEqual(self.events.read_text().count("stop -t 30 vllm-fn"), 2)
        self.assertNotIn("monitor-alive-at-container-stop", self.events.read_text())


class TestLaunchWatchdogOrdering(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.events = self.root / "events"
        self.events.touch()
        self.worker = self.root / "worker"
        self.worker.mkdir()
        executable(self.root / "engine/mem-gate.sh", "#!/bin/bash\nexit 0\n")
        executable(self.root / "scripts/start-memwatch.sh", '''#!/bin/bash
if [[ "${1:-}" == "--stop" ]]; then action=stop-watch; else action=watch; fi
printf '%s:%s:%s\\n' "$TEST_HOST" "$action" "${MEMWATCH_CONSEC:-5}" >> "$EVENTS"
''')
        executable(self.root / "scripts/memwatch.sh", "#!/bin/bash\nexit 0\n")
        executable(self.root / "bin/docker", f"#!{sys.executable}\n" + '''
import os
import sys
command = sys.argv[1]
host = os.environ["TEST_HOST"]
with open(os.environ["EVENTS"], "a") as events:
    events.write(host + ":docker-" + command + "\\n")
if command == "run":
    sys.exit(7 if os.environ.get("FAIL_HOST") == host else 0)
if command == "ps":
    print("vllm-fn")
''')
        executable(self.root / "bin/scp", f"#!{sys.executable}\n" + '''
from pathlib import Path
import shutil
import sys
args = [arg for arg in sys.argv[1:] if arg != "-q"]
target = Path(args[-1].split(":", 1)[1])
for source in args[:-1]:
    shutil.copy2(source, target / Path(source).name)
''')
        self.env = dict(os.environ, PATH=str(self.root / "bin") + ":" + os.environ["PATH"],
                        HOME=str(self.root), SCRIPT_DIR=str(self.root), TEST_HOST="head",
                        EVENTS=str(self.events), REMOTE_HOME=str(self.worker),
                        LAUNCH_SOURCE=str(REPO / "engine/launch.sh"),
                        MODEL_PATH="", CHAT_TEMPLATE="", CPUSET="", NNODES="1",
                        V030="false", PLE_OFFLOAD="false", EXTRA_VLLM_ARGS="--api-key test",
                        HEAD_IP="127.0.0.1", HEAD_PLE_MOUNT="", HEAD_MODELOPT_MOUNT="",
                        OVERLAY_ENV_STR="", HF_CACHE_DIR=str(self.root / "hf"),
                        IMAGE="fake", MODEL_ARG="fake", VLLM_ARGS_STR="", PORT="8000",
                        SERVED_MODEL_NAME="fake", READY_TIMEOUT_S="60", WORKER_IP="worker",
                        WORKER_USER="", WORKER_IFACE="eth0", IFACE="eth0", IB_HCA="mlx5",
                        WORKER_IB_HCA="mlx5", IB_GID_INDEX="0", NFS_SHARE="true",
                        NFS_VOLUME="fake-hf", REMOTE_HF=str(self.worker / "hf"),
                        ORG="org", NAME="model", WORKER_PLE_MOUNT="", WORKER_MODELOPT_MOUNT="",
                        MEMWATCH_CONSEC="7")

    def launch(self, **env):
        out = subprocess.run(["bash", "-c", '''
set -euo pipefail
info() { :; }; warn() { :; }; ok() { :; }
err() { echo "$*" >&2; exit 1; }
ssh_worker() { TEST_HOST=worker HOME="$REMOTE_HOME" bash -c "$*"; }
nfs_ensure_worker_volume() { :; }; nfs_worker_has_model() { return 0; }
sleep() { printf '%s:sleep:%s\\n' "$TEST_HOST" "$1" >> "$EVENTS"; }
curl() { echo head:health >> "$EVENTS"; printf 200; }
OVERLAY_FILES=()
source "$LAUNCH_SOURCE"
'''], env=dict(self.env, **env), cwd=self.root,
                             capture_output=True, text=True, timeout=15)
        return out, self.events.read_text().splitlines()

    def test_single_node_monitors_before_health_without_ready_duplicate(self):
        out, events = self.launch()
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertLess(events.index("head:docker-run"), events.index("head:watch:7"))
        self.assertLess(events.index("head:watch:7"), events.index("head:health"))
        self.assertEqual(events.count("head:watch:7"), 1)
        self.assertFalse(any(event.startswith("worker:") for event in events))

    def test_worker_scripts_are_transferred_and_monitor_precedes_initialization(self):
        out, events = self.launch(NNODES="2")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertTrue((self.worker / ".local/state/vllm-memwatch/scripts/memwatch.sh").is_file())
        self.assertLess(events.index("worker:docker-run"), events.index("worker:watch:7"))
        self.assertLess(events.index("worker:watch:7"), events.index("head:sleep:15"))
        self.assertLess(events.index("worker:watch:7"), events.index("head:docker-run"))
        for host in ("worker", "head"):
            self.assertEqual(events.count(f"{host}:watch:7"), 1)
            self.assertLess(events.index(f"{host}:watch:7"), events.index("head:health"))

    def test_failed_head_start_never_starts_monitor_or_polls_health(self):
        out, events = self.launch(FAIL_HOST="head")
        self.assertNotEqual(out.returncode, 0)
        self.assertNotIn("head:watch:7", events)
        self.assertNotIn("head:health", events)

    def test_failed_worker_start_never_starts_either_monitor(self):
        out, events = self.launch(NNODES="2", FAIL_HOST="worker")
        self.assertNotEqual(out.returncode, 0)
        self.assertNotIn("worker:watch:7", events)
        self.assertNotIn("head:watch:7", events)
        self.assertNotIn("head:docker-run", events)


if __name__ == "__main__":
    unittest.main()
