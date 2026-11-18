"""Exercise model selection and HF-cache consumers without network or inference."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
MODEL = "Olyno/Qwen3.8-Flash-Next-NVFP4-lean"
PIN = "a4d1e6082aa54e64188ea0cd3fca644e0447bc8a"
OTHER = "b" * 40


class TestModelSource(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("engine", "recipes", "scripts", "chat_template_cod.jinja"):
            (self.root / name).symlink_to(REPO / name)
        (self.root / "download.sh").write_bytes((REPO / "download.sh").read_bytes())
        self.cache = self.root / ".cache/huggingface"
        self.values = {"NODES": "1", "HEAD_IP": "127.0.0.1"}

    def repo(self, cache=None, model=MODEL):
        return (cache or self.cache) / "hub" / ("models--" + model.replace("/", "--"))

    def snapshot(self, revision, *, cache=None, model=MODEL, complete=True):
        repo = self.repo(cache, model)
        snap = repo / "snapshots" / revision
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "config.json").write_text(json.dumps({"fixture_revision": revision}))
        (snap / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"fixture.weight": "weights.safetensors"}}))
        if complete:
            (snap / "weights.safetensors").write_bytes(b"local fixture, not model weights")
        (repo / "refs").mkdir(exist_ok=True)
        (repo / "refs/main").write_text(revision)
        return snap

    def write_env(self):
        (self.root / ".env").write_text("".join(
            f"{key}={shlex.quote(value)}\n" for key, value in self.values.items()))

    def run_shell(self, body):
        self.write_env()
        return subprocess.run(["bash", "-c", '''
set -euo pipefail
info() { :; }; ok() { :; }; warn() { :; }
err() { printf '%s\\n' "$*" >&2; exit 1; }
SCRIPT_DIR="$PWD"
source "$SCRIPT_DIR/engine/config.sh"
''' + body], cwd=self.root, text=True, capture_output=True,
            env={"PATH": os.environ["PATH"], "HOME": str(self.root)})

    def resolve(self):
        return self.run_shell('''
source "$SCRIPT_DIR/engine/cache.sh"
printf '%s\\n' "$MODEL_ID" "$MODEL_REVISION" "$MODEL_PATH" "$DO_DOWNLOAD" "$PLE_CONFIG_DIR"
''')

    def test_default_uses_pinned_hf_snapshot_not_main(self):
        pinned = self.snapshot(PIN)
        self.snapshot(OTHER)
        result = self.resolve()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [MODEL, PIN, "", "true", str(pinned)])

    def test_missing_or_incomplete_pin_cannot_fall_back_to_complete_main(self):
        for incomplete in (False, True):
            with self.subTest(incomplete=incomplete):
                if incomplete:
                    self.snapshot(PIN, complete=False)
                self.snapshot(OTHER)
                result = self.resolve()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("snapshot", result.stderr.lower())

    def test_env_cache_location_and_revision_override(self):
        custom = self.root / "custom cache"
        selected = self.snapshot(OTHER, cache=custom)
        self.values.update(HF_HOME=str(custom), MODEL_REVISION=OTHER)
        result = self.resolve()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[-1], str(selected))

    def test_env_repository_change_does_not_inherit_lean_pin(self):
        model = "test/other-model"
        selected = self.snapshot(OTHER, model=model)
        self.values["MODEL_ID"] = model
        result = self.resolve()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [model, "", "", "true", str(selected)])

    def test_env_can_explicitly_follow_main(self):
        self.snapshot(PIN)
        selected = self.snapshot(OTHER)
        self.values["MODEL_REVISION"] = ""
        result = self.resolve()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[1], "")
        self.assertEqual(result.stdout.splitlines()[-1], str(selected))

    def test_direct_path_override_disables_download_and_revision(self):
        selected = self.snapshot(OTHER)
        self.values["MODEL_PATH"] = str(selected)
        result = self.resolve()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[1:], ["", str(selected), "false", str(selected)])

    def test_worker_with_only_another_revision_is_resynchronized(self):
        self.snapshot(PIN)
        worker_cache = self.root / "worker/cache"
        self.snapshot(OTHER, cache=worker_cache)
        self.values.update(NODES="2", WORKER_IP="127.0.0.2", IFACE="eth0",
                           IB_HCA="mlx5_0", IB_GID_INDEX="0", WORKER_HF_HOME=str(worker_cache))
        result = self.run_shell('''
# These are local fixture operations, never SSH or network rsync.
ssh_worker() {
    if (( $# > 1 )); then "$@"; else bash -c "$1"; fi
}
rsync() {
    local src="${@: -2:1}" dst="${@: -1}"
    dst="${dst#*:}"
    mkdir -p "$dst"
    cp -a "$src/." "$dst/"
}
source "$SCRIPT_DIR/engine/cache.sh"
source "$SCRIPT_DIR/engine/pair.sh"
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        snapshot = self.repo(worker_cache) / "snapshots" / PIN
        self.assertEqual(json.loads((snapshot / "config.json").read_text())["fixture_revision"], PIN)
        self.assertTrue((snapshot / "weights.safetensors").is_file())

    def test_downloader_fetches_pin_even_when_main_is_complete(self):
        self.snapshot(OTHER)
        self.values["MODEL_REVISION"] = PIN
        self.write_env()
        tools = self.root / "bin"
        tools.mkdir()
        # A local HF CLI fixture writes whichever revision the real downloader
        # requests. The real post-download completeness check must accept it.
        cli = tools / "uvx"
        cli.write_text('''#!/usr/bin/python3
import json, pathlib, sys
args = sys.argv[1:]
repo = pathlib.Path(args[args.index('--cache-dir') + 1]) / ('models--' + args[2].replace('/', '--'))
revision = args[args.index('--revision') + 1] if '--revision' in args else 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
snapshot = repo / 'snapshots' / revision
snapshot.mkdir(parents=True, exist_ok=True)
(snapshot / 'config.json').write_text(json.dumps({'fixture_revision': revision}))
(snapshot / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w': 'weights.safetensors'}}))
(snapshot / 'weights.safetensors').write_bytes(b'local fixture')
''')
        cli.chmod(0o755)
        result = subprocess.run(["bash", str(self.root / "download.sh"), MODEL],
            cwd=self.root, text=True, capture_output=True,
            env={"PATH": str(tools) + ":" + os.environ["PATH"], "HOME": str(self.root)})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.repo() / "snapshots" / PIN / "weights.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()
