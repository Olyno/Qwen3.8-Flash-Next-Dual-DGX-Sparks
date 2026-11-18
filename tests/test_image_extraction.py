"""Image extraction must survive Docker's implicit first pull under errexit."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


class TestImageExtraction(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        tools = self.root / "bin"
        tools.mkdir()
        docker = tools / "docker"
        docker.write_text('''#!/usr/bin/python3
import json, os, pathlib, sys
root = pathlib.Path(os.environ['FIXTURE_ROOT'])
args = sys.argv[1:]
with (root / 'commands.jsonl').open('a') as log:
    log.write(json.dumps(args) + '\\n')
state = root / 'image'
mode = os.environ.get('DOCKER_FAILURE', '')
if args[:2] == ['image', 'inspect']:
    if not state.exists():
        sys.exit(1)
    print(state.read_text())
elif args[0] == 'create':
    if mode == 'pull':
        sys.exit(9)
    if not state.exists():
        print('Pull complete', file=sys.stderr)
        state.write_text('sha256:first')
    print('fixture-container')
elif args[0] == 'inspect':
    print(state.read_text())
elif args[0] == 'cp':
    if mode == 'copy':
        sys.exit(8)
    pathlib.Path(args[-1]).write_text('payload from ' + state.read_text())
elif args[0] != 'rm':
    sys.exit(99)
''')
        docker.chmod(0o755)
        self.env = {"PATH": str(tools) + ":" + os.environ["PATH"],
                    "HOME": str(self.root), "FIXTURE_ROOT": str(self.root)}

    def extract(self, tail=""):
        return subprocess.run(["bash", "-c", '''
set -euo pipefail
info() { :; }
err() { printf '%s\\n' "$*" >&2; exit 1; }
IMAGE=fixture/vllm:latest
DO_LAUNCH=false
source ''' + shlex.quote(str(REPO / "engine/patches.sh")) + '''
extract_from_image /vllm/source.py "$FIXTURE_ROOT/source.py"
printf 'continued\\n'
''' + tail], cwd=self.root, text=True, capture_output=True, env=self.env)

    def commands(self):
        return [json.loads(line) for line in (self.root / "commands.jsonl").read_text().splitlines()]

    def test_first_pull_continues_and_caches_the_extracted_image_identity(self):
        result = self.extract('extract_from_image /vllm/source.py "$FIXTURE_ROOT/source.py"\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("continued", result.stdout)
        self.assertEqual((self.root / "source.py").read_text(), "payload from sha256:first")
        self.assertEqual((self.root / "source.py.image-id").read_text(), "sha256:first")
        self.assertEqual(sum(args[0] == "create" for args in self.commands()), 1)

    def test_changed_image_replaces_stale_extraction(self):
        (self.root / "image").write_text("sha256:new")
        (self.root / "source.py").write_text("old image payload")
        (self.root / "source.py.image-id").write_text("sha256:old")
        result = self.extract()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "source.py").read_text(), "payload from sha256:new")
        self.assertEqual((self.root / "source.py.image-id").read_text(), "sha256:new")

    def test_pull_failure_stops_before_continuation_or_cache_stamp(self):
        self.env["DOCKER_FAILURE"] = "pull"
        result = self.extract()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("continued", result.stdout)
        self.assertFalse((self.root / "source.py.image-id").exists())
        self.assertFalse((self.root / "source.py").exists())

    def test_copy_failure_does_not_mark_a_stale_payload_as_current(self):
        (self.root / "image").write_text("sha256:new")
        (self.root / "source.py").write_text("old image payload")
        (self.root / "source.py.image-id").write_text("sha256:old")
        self.env["DOCKER_FAILURE"] = "copy"
        result = self.extract()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("continued", result.stdout)
        self.assertEqual((self.root / "source.py.image-id").read_text(), "sha256:old")
        self.assertEqual((self.root / "source.py").read_text(), "old image payload")


if __name__ == "__main__":
    unittest.main()
