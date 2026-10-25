#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Tool-use tasks: finish small agent jobs through the real tool-call loop.

Every other gate in bench/ scores an answer. This one scores the loop: the
model is offered tools, its calls are executed against a scratch directory,
results go back as `role: "tool"` messages, and the run ends on a final answer
or a turn cap. It exists so changes to tool routing, tool-call repair, skill
nudges and output compaction can be measured by completed tasks, turns and
tokens instead of unit tests alone.

    python3 bench/tool_tasks.py --selftest
    BENCH_BASE=http://localhost:8888 BENCH_MODEL=... python3 bench/tool_tasks.py
    BENCH_BASE=http://localhost:8888 BENCH_MODEL=... python3 bench/tool_tasks.py --repeat 2
    python3 bench/tool_tasks.py --json run.json
    python3 bench/tool_tasks.py --compare run.json --fail-on-regression

Greedy (temperature 0). Endpoint/model come from BENCH_BASE/BENCH_MODEL.
The outcome is code, never a judge: a task passes when its checker sees the
expected final answer, file content or program output within its turn bound.
`no_tool_call` means a task that needs a tool was answered without one;
`tool_called` means a task that needs no tool used one; `turn_cap` means the
loop never reached a final answer.

`--repeat N` runs each task N times in fresh scratch directories. A task that
passes some runs but not all is reported as `FLAKY` with `passes/N`, never
rounded to a pass or a fail; `passes` and `repeats` carry the same count in
the JSON result.

`run_python` executes model-authored code with the invoking user's privileges.
Point this at a trusted endpoint and run it on a machine you are willing to
expose to that code.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

import reasoning_check

BASE = os.environ.get("BENCH_BASE", "http://localhost:8888")
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.8-Flash-Next-NVFP4")

MAX_OUTPUT_CHARS = 40000
MAX_SEARCH_MATCHES = 100
RUN_TIMEOUT = 20


def endpoint(base):
    """Accept a server base URL or a full /v1/chat/completions endpoint."""
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _tool(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }}


TOOLS = [
    _tool(
        "list_dir",
        "List the entries in a directory. Directories end with '/'.",
        {"path": {"type": "string",
                  "description": "Directory path relative to the scratch "
                                 "directory, e.g. '.' or 'docs'."}},
        ["path"],
    ),
    _tool(
        "read_file",
        "Read a UTF-8 text file and return its contents.",
        {"path": {"type": "string",
                  "description": "File path relative to the scratch directory."}},
        ["path"],
    ),
    _tool(
        "write_file",
        "Write text to a file, creating parent directories. Overwrites an "
        "existing file.",
        {"path": {"type": "string",
                  "description": "File path relative to the scratch directory."},
         "content": {"type": "string", "description": "Exact text to write."}},
        ["path", "content"],
    ),
    _tool(
        "search",
        "Search a file or directory for a literal, case-sensitive substring. "
        "Returns matching lines as path:line: text.",
        {"pattern": {"type": "string", "description": "Literal substring."},
         "path": {"type": "string",
                  "description": "File or directory path relative to the "
                                 "scratch directory. Defaults to '.'."}},
        ["pattern"],
    ),
    _tool(
        "run_python",
        "Run a Python file from the scratch directory with the current "
        "interpreter. Returns the exit code, stdout and stderr.",
        {"path": {"type": "string",
                  "description": "File path relative to the scratch directory."}},
        ["path"],
    ),
]
TOOL_NAMES = tuple(t["function"]["name"] for t in TOOLS)


class ToolError(Exception):
    pass


def _str_arg(arguments, key):
    value = arguments.get(key)
    if not isinstance(value, str) or not value:
        raise ToolError(f"{key!r} must be a non-empty string")
    return value


def _text_arg(arguments, key):
    value = arguments.get(key)
    if not isinstance(value, str):
        raise ToolError(f"{key!r} must be a string")
    return value


def _safe_path(root, rel):
    if not isinstance(rel, str) or not rel.strip():
        raise ToolError("path must be a non-empty string")
    full = os.path.realpath(os.path.join(root, rel))
    if os.path.commonpath([root, full]) != root:
        raise ToolError(f"path escapes the scratch directory: {rel}")
    return full


def tool_list_dir(root, arguments):
    rel = _str_arg(arguments, "path")
    path = _safe_path(root, rel)
    if not os.path.exists(path):
        raise ToolError(f"no such path: {rel}")
    if not os.path.isdir(path):
        raise ToolError(f"not a directory: {rel}")
    names = sorted(os.listdir(path))
    if not names:
        return "(empty directory)"
    return "\n".join(name + "/" if os.path.isdir(os.path.join(path, name))
                     else name for name in names)


def tool_read_file(root, arguments):
    rel = _str_arg(arguments, "path")
    path = _safe_path(root, rel)
    if not os.path.exists(path):
        raise ToolError(f"no such file: {rel}")
    if not os.path.isfile(path):
        raise ToolError(f"not a file: {rel}")
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read(MAX_OUTPUT_CHARS + 1)
    if len(text) > MAX_OUTPUT_CHARS:
        text = (text[:MAX_OUTPUT_CHARS]
                + f"\n[truncated at {MAX_OUTPUT_CHARS} characters]")
    return text if text else "(empty file)"


def tool_write_file(root, arguments):
    rel = _str_arg(arguments, "path")
    content = _text_arg(arguments, "content")
    path = _safe_path(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"wrote {len(content.encode('utf-8'))} bytes to {rel}"


def _rel(root, path):
    return os.path.relpath(path, root).replace(os.sep, "/")


def tool_search(root, arguments):
    pattern = _str_arg(arguments, "pattern")
    rel = arguments.get("path", ".")
    if not isinstance(rel, str) or not rel:
        raise ToolError("'path' must be a non-empty string")
    base = _safe_path(root, rel)
    if not os.path.exists(base):
        raise ToolError(f"no such path: {rel}")
    if os.path.isfile(base):
        files = [base]
    elif os.path.isdir(base):
        files = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames.sort()
            files.extend(os.path.join(dirpath, name)
                         for name in sorted(filenames))
    else:
        raise ToolError(f"not a file or directory: {rel}")

    hits = []
    for path in files:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    if pattern in line:
                        shown = line.rstrip()[:500]
                        hits.append(f"{_rel(root, path)}:{lineno}: {shown}")
                        if len(hits) >= MAX_SEARCH_MATCHES:
                            break
        except OSError:
            continue
        if len(hits) >= MAX_SEARCH_MATCHES:
            break

    if not hits:
        return f"no matches for {pattern!r}"
    head = f"{len(hits)} match(es) for {pattern!r}"
    if len(hits) >= MAX_SEARCH_MATCHES:
        head += f" (stopped at {MAX_SEARCH_MATCHES})"
    return head + "\n" + "\n".join(hits)


def tool_run_python(root, arguments):
    rel = _str_arg(arguments, "path")
    path = _safe_path(root, rel)
    if not os.path.isfile(path):
        raise ToolError(f"no such file: {rel}")
    try:
        cp = subprocess.run([sys.executable, path], cwd=root,
                            capture_output=True, text=True, timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise ToolError(f"run_python timed out after {RUN_TIMEOUT}s")
    parts = [f"exit code: {cp.returncode}",
             "stdout:\n" + (cp.stdout or "(empty)")]
    if cp.stderr:
        parts.append("stderr:\n" + cp.stderr)
    output = "\n".join(parts)
    if len(output) > MAX_OUTPUT_CHARS:
        output = (output[:MAX_OUTPUT_CHARS]
                  + f"\n[truncated at {MAX_OUTPUT_CHARS} characters]")
    return output


TOOL_FUNCS = {
    "list_dir": tool_list_dir,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "search": tool_search,
    "run_python": tool_run_python,
}


def invoke_tool(name, arguments, root):
    """Return (ok, text). Tool errors are text the model can read and retry."""
    if name not in TOOL_FUNCS:
        return False, (f"error: unknown tool {name!r}; available: "
                       f"{', '.join(TOOL_NAMES)}")
    try:
        return True, TOOL_FUNCS[name](root, arguments)
    except ToolError as exc:
        return False, f"error: {exc}"
    except Exception as exc:                            # noqa: BLE001
        return False, f"error: {type(exc).__name__}: {exc}"


def normalize_call(tool_call, index):
    if not isinstance(tool_call, dict):
        tool_call = {}
    function = tool_call.get("function")
    if not isinstance(function, dict):
        function = {}
    call_id = tool_call.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"call_{index}"
    name = function.get("name")
    if not isinstance(name, str):
        name = ""
    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    elif not isinstance(arguments, str):
        arguments = "" if arguments is None else str(arguments)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def parse_arguments(raw):
    if not raw.strip():
        return {}
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolError(f"arguments are not valid JSON: {exc}")
    if not isinstance(arguments, dict):
        raise ToolError("arguments must be a JSON object")
    return arguments


SYSTEM = (
    "You are a coding agent working in an isolated scratch directory. Every "
    "path in a tool call is relative to that directory; use '.' for the top "
    "level. Use the provided tools to inspect and modify files. Do not guess "
    "file contents. When the task is complete, reply with a short final "
    "answer and no tool call."
)

CONFIG_BEFORE = "[service]\nretries=2\ntimeout=30\n"
CONFIG_AFTER = "[service]\nretries=5\ntimeout=30\n"

APP_LOG = "\n".join([
    "2026-10-01 09:00:01 INFO  service started",
    "2026-10-01 09:00:02 INFO  loaded 12 workers",
    "2026-10-01 09:00:05 INFO  connected to queue",
    "2026-10-01 09:01:10 INFO  job 4101 accepted",
    "2026-10-01 09:01:14 INFO  job 4101 completed in 4.1s",
    "2026-10-01 09:02:03 INFO  job 4102 accepted",
    "2026-10-01 09:02:09 INFO  job 4102 completed in 6.0s",
    "2026-10-01 09:03:21 WARN  queue depth above 20",
    "2026-10-01 09:03:40 INFO  queue depth back to 3",
    "2026-10-01 09:04:12 INFO  job 4103 accepted",
    "2026-10-01 09:04:18 INFO  job 4103 completed in 6.2s",
    "2026-10-01 09:05:00 INFO  cache warm",
    "2026-10-01 09:06:31 WARN  retrying job 4104",
    "2026-10-01 09:06:44 INFO  job 4104 completed on retry",
    "2026-10-01 09:08:02 INFO  metrics flushed",
    "2026-10-01 09:09:15 INFO  job 4105 accepted",
    "2026-10-01 09:09:20 INFO  job 4105 completed in 5.4s",
    "2026-10-01 09:11:48 WARN  disk usage above 80 percent",
    "2026-10-01 09:14:22 ERROR code=QW-7781 disk quota exceeded",
    "2026-10-01 09:14:23 INFO  worker 7 paused",
    "2026-10-01 09:15:02 INFO  worker 8 picked up remaining jobs",
    "2026-10-01 09:20:00 INFO  shutdown requested",
]) + "\n"

INVENTORY = "\n".join([
    "sku,status",
    "A-100,status=ready",
    "A-101,status=pending",
    "A-102,status=ready",
    "A-103,status=ready",
    "A-104,status=failed",
    "A-105,status=pending",
    "A-106,status=ready",
    "A-107,status=failed",
    "A-108,status=ready",
    "A-109,status=pending",
]) + "\n"

BUGGY_TOTAL = (
    "def total(nums):\n"
    "    return sum(nums) - 1\n"
    "\n"
    "\n"
    "if __name__ == \"__main__\":\n"
    "    print(total([1, 2, 3]))\n"
)

SETTINGS_BEFORE = "mode=draft\nlevel=3\n"
SETTINGS_AFTER = "mode=final\nlevel=3\n"

EVENTS_LOG = "\n".join([
    "2026-10-02 10:00:00 INFO  service boot",
    "2026-10-02 10:00:01 INFO  retries=7",
    "2026-10-02 10:00:02 INFO  ready",
]) + "\n"

HELPER_PY = (
    "# Prints a value that does not come from events.log.\n"
    "print(\"retries=99\")\n"
)

NO_TOOL_NOTES = "build: 9999\n"

SEARCH_LINE = ("2026-10-02 11:22:33 ERROR incident=ZX-3107 quota exceeded "
               "for tenant 42")

TOKEN_BYTES = "seed=314159\n"

LIST_CANDIDATES = {
    "project/src/main.py": True,
    "project/src/config.py": False,
    "project/src/nested/deep.txt": True,
    "project/docs/notes.md": True,
    "project/tests/test_util.py": True,
    "project/src/helpers.py": False,
}

LIST_TREE = {
    "project/README.md": "# Demo project\n",
    "project/src/main.py": "print('main')\n",
    "project/src/util.py": "def helper():\n    return 1\n",
    "project/src/nested/deep.txt": "deep\n",
    "project/docs/notes.md": "# Notes\n",
    "project/tests/test_util.py": "def test_util():\n    assert True\n",
}

LIST_DIRS = ("project", "project/src", "project/src/nested",
             "project/docs", "project/tests")


def make_big_log(root):
    lines = [f"entry {i:04d}: routine inventory reconciliation nominal "
             f"for depot north" for i in range(400)]
    lines.append("checksum: ZX-8842")
    with open(os.path.join(root, "big.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def make_search_log(root):
    lines = [f"entry {i:05d}: routine inventory reconciliation nominal "
             f"for depot north" for i in range(8000)]
    lines[6000] = SEARCH_LINE
    with open(os.path.join(root, "big.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _read_text(root, rel):
    try:
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _norm(rel):
    rel = (rel or "").replace("\\", "/")
    rel = os.path.normpath(rel)
    return "" if rel == "." else rel.lstrip("./")


def _call_index(calls, name, path=None, ok=None, after=-1, pattern=None):
    for i, call in enumerate(calls):
        if i <= after or call["name"] != name:
            continue
        if path is not None and _norm(call["arguments"].get("path")) != path:
            continue
        if ok is not None and call["ok"] != ok:
            continue
        if pattern is not None and pattern not in str(
                call["arguments"].get("pattern", "")):
            continue
        return i
    return -1


def _last_index(calls, name, path, ok=True):
    found = -1
    for i, call in enumerate(calls):
        if (call["name"] == name and call["ok"] == ok
                and _norm(call["arguments"].get("path")) == path):
            found = i
    return found


def _run_file(root, rel):
    try:
        return subprocess.run([sys.executable, os.path.join(root, rel)],
                              cwd=root, capture_output=True, text=True,
                              timeout=RUN_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _run_python_code(root, code):
    try:
        return subprocess.run([sys.executable, "-c", code], cwd=root,
                              capture_output=True, text=True,
                              timeout=RUN_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        return None


def check_list_then_read(root, final, calls):
    listed = _call_index(calls, "list_dir", "docs", ok=True)
    if listed < 0:
        return False, "missing_step", "docs was not listed"
    if _call_index(calls, "read_file", "docs/alpha.txt", ok=True,
                   after=listed) < 0:
        return False, "missing_step", "the first guide was not read after listing"
    return True, None, ""


def check_search_quote(root, final, calls):
    if _call_index(calls, "search", ok=True, pattern="QW-7781") < 0:
        return False, "missing_step", "search was not used for QW-7781"
    return True, None, ""


def check_edit(root, final, calls):
    got = _read_text(root, "config.ini")
    if got != CONFIG_AFTER:
        return False, "wrong_file", f"config.ini is {got!r}"
    wrote = _last_index(calls, "write_file", "config.ini")
    if wrote < 0:
        return False, "missing_step", "config.ini was not written"
    if _call_index(calls, "read_file", "config.ini", ok=True,
                   after=wrote) < 0:
        return False, "missing_step", "config.ini was not read back after the edit"
    return True, None, ""


def check_fix_bug(root, final, calls):
    wrote = _last_index(calls, "write_file", "total.py")
    if wrote < 0:
        return False, "missing_step", "total.py was not edited"
    if _call_index(calls, "run_python", "total.py", ok=True,
                   after=wrote) < 0:
        return False, "missing_step", "total.py was not run after the edit"
    cp = _run_file(root, "total.py")
    if cp is None or cp.returncode != 0:
        return False, "wrong_run", "total.py did not exit 0"
    if cp.stdout.strip() != "6":
        return False, "wrong_run", (f"total.py printed {cp.stdout.strip()!r}, "
                                    f"expected '6'")
    probe = _run_python_code(
        root,
        "import sys\n"
        "sys.path.insert(0, '.')\n"
        "import total\n"
        "print(total.total([2, 3]))\n")
    if probe is None or probe.returncode != 0:
        return False, "wrong_run", "could not import total.py to test total([2, 3])"
    if probe.stdout.strip() != "5":
        return False, "wrong_run", (f"total([2, 3]) printed "
                                    f"{probe.stdout.strip()!r}, expected '5'")
    return True, None, ""


def check_recover(root, final, calls):
    failed = _call_index(calls, "read_file", "data/secrets.txt", ok=False)
    if failed < 0:
        return False, "missing_step", "data/secrets.txt was not tried"
    if _call_index(calls, "read_file", "data/actual.txt", ok=True,
                   after=failed) < 0:
        return False, "missing_step", ("no successful read of data/actual.txt "
                                       "after the failed read")
    return True, None, ""


def check_large_output(root, final, calls):
    if _call_index(calls, "read_file", "big.log", ok=True) < 0:
        return False, "missing_step", "big.log was not read"
    return True, None, ""


def check_absent_info(root, final, calls):
    if _call_index(calls, "read_file", "release_notes.md", ok=True) < 0:
        return False, "missing_step", "release_notes.md was not read"
    return True, None, ""


def check_write_new(root, final, calls):
    got = _read_text(root, "out/hello.txt")
    if got != "hello tool\n":
        return False, "wrong_file", f"out/hello.txt is {got!r}"
    wrote = _last_index(calls, "write_file", "out/hello.txt")
    if wrote < 0:
        return False, "missing_step", "out/hello.txt was not written"
    if _call_index(calls, "read_file", "out/hello.txt", ok=True,
                   after=wrote) < 0:
        return False, "missing_step", "out/hello.txt was not read back"
    return True, None, ""


def check_run_calc(root, final, calls):
    wrote = _last_index(calls, "write_file", "calc.py")
    if wrote < 0:
        return False, "missing_step", "calc.py was not written"
    if _call_index(calls, "run_python", "calc.py", ok=True,
                   after=wrote) < 0:
        return False, "missing_step", "calc.py was not run after writing"
    cp = _run_file(root, "calc.py")
    if cp is None or cp.returncode != 0:
        return False, "wrong_run", "calc.py did not exit 0"
    if cp.stdout.strip() != "385":
        return False, "wrong_run", (f"calc.py printed {cp.stdout.strip()!r}, "
                                    f"expected '385'")
    return True, None, ""


def check_multi_file(root, final, calls):
    for rel in ("a.txt", "b.txt"):
        if _call_index(calls, "read_file", rel, ok=True) < 0:
            return False, "missing_step", f"{rel} was not read"
    return True, None, ""


def check_updated_file(root, final, calls):
    got = _read_text(root, "settings.ini")
    if (got or "").rstrip("\n") != SETTINGS_AFTER.rstrip("\n"):
        return False, "wrong_file", f"settings.ini is {got!r}"
    wrote = _last_index(calls, "write_file", "settings.ini")
    if wrote < 0:
        return False, "missing_step", "settings.ini was not written"
    reread = _call_index(calls, "read_file", "settings.ini", ok=True,
                         after=wrote)
    if reread < 0:
        return False, "not_reread", "settings.ini was not read after the write"
    if "mode=final" not in calls[reread]["result"]:
        return False, "stale_read", ("the re-read returned stale content: "
                                     f"{calls[reread]['result'][:120]!r}")
    if re.search(r"ANSWER:\s*mode=final\b", final):
        return True, None, ""
    if "mode=draft" in final:
        return False, "stale_value", "the final answer reported the old value"
    return False, "wrong_value", f"unexpected final answer: {final[-120:]!r}"


def check_pick_read_over_run(root, final, calls):
    executed = next((call for call in calls if call["name"] == "run_python"),
                    None)
    if executed is not None:
        path = executed["arguments"].get("path")
        return False, "wrong_tool", f"ran {path!r} instead of reading the log"
    if _call_index(calls, "read_file", "events.log", ok=True) < 0:
        return False, "missing_read", "events.log was not read"
    if not re.search(r"ANSWER:\s*retries=7\b", final):
        return False, "wrong_value", f"unexpected final answer: {final[-120:]!r}"
    return True, None, ""


def check_pick_search(root, final, calls):
    searched = _call_index(calls, "search", ok=True, pattern="ZX-3107")
    if searched < 0:
        return False, "missing_search", "no successful search for ZX-3107"
    if SEARCH_LINE not in calls[searched]["result"]:
        return False, "stale_search", "the search result did not contain the line"
    if SEARCH_LINE not in final:
        return False, "wrong_line", (f"final answer did not quote "
                                     f"{SEARCH_LINE!r}")
    return True, None, ""


def _path_claims(final):
    claims = {}
    for line in final.splitlines():
        match = re.match(r"\s*(EXISTS|MISSING)\s*:\s*(.+?)\s*$", line, re.I)
        if not match:
            continue
        raw = match.group(2).strip().strip("`*_ ").rstrip(".,;:")
        claims[_norm(raw)] = match.group(1).upper()
    return claims


def check_pick_list(root, final, calls):
    missing = [rel for rel in LIST_DIRS
               if _call_index(calls, "list_dir", rel, ok=True) < 0]
    if missing:
        return False, "never_listed", f"not listed: {', '.join(missing)}"
    claims = _path_claims(final)
    actual = {rel for rel, exists in LIST_CANDIDATES.items() if exists}
    invented = sorted(rel for rel, kind in claims.items()
                      if kind == "EXISTS" and rel in LIST_CANDIDATES
                      and rel not in actual)
    if invented:
        return False, "invented", ("reported as existing but absent: "
                                   + ", ".join(invented))
    omitted = sorted(rel for rel in actual if claims.get(rel) != "EXISTS")
    if omitted:
        return False, "omitted", ("existing paths not reported: "
                                  + ", ".join(omitted))
    return True, None, ""


def check_no_tool_needed(root, final, calls):
    if calls:
        return False, "tool_called", (f"{calls[0]['name']} was called, but "
                                      "the question needs no tool")
    if not re.search(r"ANSWER:\s*4312\b", final):
        return False, "wrong_value", f"unexpected final answer: {final[-120:]!r}"
    return True, None, ""


def check_write_then_verify(root, final, calls):
    got = _read_text(root, "out/token.txt")
    if got != TOKEN_BYTES:
        return False, "wrong_file", f"out/token.txt is {got!r}"
    wrote = _last_index(calls, "write_file", "out/token.txt")
    if wrote < 0:
        return False, "missing_step", "out/token.txt was not written"
    reread = _call_index(calls, "read_file", "out/token.txt", ok=True,
                         after=wrote)
    if reread < 0:
        return False, "not_reread", "out/token.txt was not read after the write"
    if "seed=314159" not in calls[reread]["result"]:
        return False, "stale_read", ("the re-read returned stale content: "
                                     f"{calls[reread]['result'][:120]!r}")
    if not re.search(r"ANSWER:\s*11\b", final):
        return False, "wrong_value", f"unexpected final answer: {final[-120:]!r}"
    return True, None, ""


TASKS = [
    {
        "name": "read_fact",
        "files": {"notes.txt": "release: 4.12.0\nbuild: 4312\n"
                               "channel: stable\n"},
        "prompt": "Read notes.txt and report the build number. "
                  "End with `ANSWER: <build>`.",
        "answer": r"ANSWER:\s*4312\b",
        "max_turns": 4,
    },
    {
        "name": "list_then_read",
        "files": {
            "docs/README.md": "Index of guides.\n",
            "docs/alpha.txt": "setting: teal\n",
            "docs/beta.txt": "setting: amber\n",
        },
        "prompt": "List the docs directory. Then read the non-README .txt "
                  "guide whose filename comes first alphabetically, and "
                  "report its setting. End with `ANSWER: <setting>`.",
        "answer": r"ANSWER:\s*teal\b",
        "check": check_list_then_read,
        "max_turns": 5,
    },
    {
        "name": "search_quote",
        "files": {"app.log": APP_LOG},
        "prompt": "Use the search tool to find the line in app.log that "
                  "contains QW-7781, and quote that exact line. "
                  "End with `ANSWER: <line>`.",
        "answer": r"ANSWER:.*ERROR code=QW-7781 disk quota exceeded",
        "check": check_search_quote,
        "max_turns": 4,
    },
    {
        "name": "edit_value",
        "files": {"config.ini": CONFIG_BEFORE},
        "prompt": "Use write_file to change retries from 2 to 5 in config.ini. "
                  "Leave every other line exactly as it is. Read the file back "
                  "to confirm. End with `ANSWER: retries=<value>`.",
        "answer": r"ANSWER:\s*retries=5\b",
        "check": check_edit,
        "max_turns": 6,
    },
    {
        "name": "fix_bug",
        "files": {"total.py": BUGGY_TOTAL},
        "prompt": "total.py should print 6 but prints 5. Fix the bug with "
                  "write_file, run total.py with run_python to prove the fix, "
                  "and report the printed value. End with `ANSWER: <value>`.",
        "answer": r"ANSWER:\s*6\b",
        "check": check_fix_bug,
        "max_turns": 7,
    },
    {
        "name": "recover_missing",
        "files": {
            "data/actual.txt": "password: ORCHID-5521\n",
            "data/notes.txt": "password: obsolete\n",
        },
        "prompt": "The password is meant to be in data/secrets.txt. Start by "
                  "trying to read that file. If the read fails, list data/, "
                  "find the real password file, read it, and report the "
                  "password. Do not guess. End with `ANSWER: <password>`.",
        "answer": r"ANSWER:\s*ORCHID-5521\b",
        "check": check_recover,
        "max_turns": 7,
    },
    {
        "name": "large_output",
        "make": make_big_log,
        "prompt": "Read big.log and report the checksum from its last line. "
                  "End with `ANSWER: <checksum>`.",
        "answer": r"ANSWER:\s*ZX-8842\b",
        "check": check_large_output,
        "max_turns": 4,
    },
    {
        "name": "absent_info",
        "files": {"release_notes.md": "# Release notes\n\n"
                                      "- Faster startup on cold caches.\n"
                                      "- Fixed a cache invalidation bug in the "
                                      "worker pool.\n"
                                      "- Added filters to the dashboard.\n"},
        "prompt": "Read release_notes.md and report the release date. If the "
                  "file does not state a date, answer NOT FOUND instead of "
                  "guessing. End with `ANSWER: <date or NOT FOUND>`.",
        "answer": r"ANSWER:\s*NOT FOUND\b",
        "check": check_absent_info,
        "max_turns": 4,
    },
    {
        "name": "count_matches",
        "files": {"inventory.csv": INVENTORY},
        "prompt": "Count the lines in inventory.csv that contain the literal "
                  "substring status=ready. End with `ANSWER: <count>`.",
        "answer": r"ANSWER:\s*5\b",
        "max_turns": 5,
    },
    {
        "name": "write_new_file",
        "prompt": "Use write_file to create out/hello.txt whose exact content "
                  "is the single line hello tool followed by a newline. Read "
                  "it back to confirm. End with `ANSWER: written`.",
        "answer": r"ANSWER:\s*written\b",
        "check": check_write_new,
        "max_turns": 5,
    },
    {
        "name": "run_computation",
        "prompt": "Use write_file to create calc.py that prints the sum of the "
                  "squares of the integers 1 through 10 inclusive. Run it with "
                  "run_python and report the printed value. "
                  "End with `ANSWER: <value>`.",
        "answer": r"ANSWER:\s*385\b",
        "check": check_run_calc,
        "max_turns": 6,
    },
    {
        "name": "multi_file",
        "files": {"a.txt": "left=17\n", "b.txt": "right=25\n"},
        "prompt": "Read a.txt and b.txt, add the two integers they contain, "
                  "and report the sum. End with `ANSWER: <sum>`.",
        "answer": r"ANSWER:\s*42\b",
        "check": check_multi_file,
        "max_turns": 5,
    },
    {
        "name": "updated_file",
        "files": {"settings.ini": SETTINGS_BEFORE},
        "prompt": "Use write_file to change mode=draft to mode=final in "
                  "settings.ini, leaving level=3 unchanged. Read the file "
                  "again after writing it, then report the new mode. "
                  "End with `ANSWER: mode=<value>`.",
        "answer": r"ANSWER:\s*mode=final\b",
        "check": check_updated_file,
        "max_turns": 5,
    },
    {
        "name": "pick_read_over_run",
        "files": {"events.log": EVENTS_LOG, "helper.py": HELPER_PY},
        "prompt": "events.log records the retry setting. Read the value with "
                  "read_file; do not execute anything. End with "
                  "`ANSWER: retries=<value>`.",
        "answer": r"ANSWER:\s*retries=7\b",
        "check": check_pick_read_over_run,
        "max_turns": 4,
    },
    {
        "name": "pick_search_over_read",
        "make": make_search_log,
        "prompt": "big.log is too large to read in full. Use search to find "
                  "the single line containing incident=ZX-3107 and quote that "
                  "exact line. End with `ANSWER: <line>`.",
        "check": check_pick_search,
        "max_turns": 4,
    },
    {
        "name": "pick_list_over_guess",
        "files": LIST_TREE,
        "prompt": "List the project tree as needed, then report for each path "
                  "below whether it exists, one per line as "
                  "`EXISTS: <path>` or `MISSING: <path>`. Do not guess. "
                  "Paths: " + ", ".join(LIST_CANDIDATES) + ".",
        "check": check_pick_list,
        "max_turns": 8,
    },
    {
        "name": "no_tool_needed",
        "files": {"notes.txt": NO_TOOL_NOTES},
        "prompt": "The release version is 4.12.0 and the build number is 4312. "
                  "Answer from this message alone, without calling any tool. "
                  "End with `ANSWER: <build>`.",
        "answer": r"ANSWER:\s*4312\b",
        "check": check_no_tool_needed,
        "no_tools": True,
        "max_turns": 3,
    },
    {
        "name": "write_then_verify",
        "prompt": "Use write_file to create out/token.txt whose exact content "
                  "is the single line seed=314159 followed by a newline. Read "
                  "it back and report the number of characters before the "
                  "newline. End with `ANSWER: <count>`.",
        "answer": r"ANSWER:\s*11\b",
        "check": check_write_then_verify,
        "max_turns": 5,
    },
]

TASK_BY_NAME = {task["name"]: task for task in TASKS}


def setup_task(task, root):
    os.makedirs(root, exist_ok=True)
    for rel, content in task.get("files", {}).items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    maker = task.get("make")
    if maker:
        maker(root)


def grade(task, root, final, calls):
    checker = task.get("check")
    if checker:
        ok, reason, detail = checker(root, final, calls)
        if not ok:
            return ok, reason, detail
    pattern = task.get("answer")
    if pattern and not re.search(pattern, final, re.S):
        return (False, "wrong_answer",
                f"final answer did not match {pattern!r}: "
                f"{final.strip()[-160:]!r}")
    return True, None, ""


def sum_usage(usages):
    rows = {}
    for i, usage in enumerate(usages):
        rows[str(i)] = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (
                usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"),
        }
    return reasoning_check.token_totals(rows)


def run_task(task, root, chat, max_turns):
    setup_task(task, root)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task["prompt"]},
    ]
    calls, usages = [], []
    final = ""
    hit_cap = False
    error = None
    turns = 0
    started = time.time()
    # Wall-clock split: the model call is the engine's share, the tool loop is
    # local work (file reads, python). Without this the harness can only say
    # what a task cost, not whether the engine or the harness spent it.
    model_seconds = 0.0
    tool_seconds = 0.0

    while turns < max_turns:
        turns += 1
        t_model = time.time()
        try:
            reply = chat(messages)
        except Exception as exc:                        # noqa: BLE001
            model_seconds += time.time() - t_model
            error = f"{type(exc).__name__}: {exc}"
            break
        model_seconds += time.time() - t_model
        usage = reply.get("usage")
        usages.append(usage if isinstance(usage, dict) else {})
        message = reply.get("message")
        if not isinstance(message, dict):
            message = {}
        content = message.get("content") or ""
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raw_calls = []

        if not raw_calls:
            final = reasoning_check.strip_reasoning(content)
            break

        normalized = [normalize_call(call, i)
                      for i, call in enumerate(raw_calls)]
        messages.append({"role": "assistant",
                         "content": content or None,
                         "tool_calls": normalized})
        for call in normalized:
            name = call["function"]["name"]
            t_tool = time.time()
            try:
                arguments = parse_arguments(call["function"]["arguments"])
                ok_call, output = invoke_tool(name, arguments, root)
            except ToolError as exc:
                arguments, ok_call, output = {}, False, f"error: {exc}"
            tool_seconds += time.time() - t_tool
            calls.append({"name": name, "arguments": arguments,
                          "ok": ok_call, "result": output,
                          "seconds": round(time.time() - t_tool, 2)})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": output})
    else:
        hit_cap = True

    if error:
        ok, reason, detail = False, "error", error
    elif hit_cap and task.get("no_tools") and calls:
        ok, reason, detail = (False, "tool_called",
                              f"{calls[0]['name']} was called, but the "
                              "question needs no tool")
    elif hit_cap:
        ok, reason, detail = (False, "turn_cap",
                              f"no final answer in {max_turns} turns")
    elif not calls and not task.get("no_tools"):
        ok, reason, detail = (False, "no_tool_call",
                              "answered without calling a tool")
    else:
        ok, reason, detail = grade(task, root, final, calls)

    result = {
        "ok": ok,
        "reason": reason,
        "detail": detail,
        "turns": turns,
        "tool_calls": len(calls),
        "seconds": round(time.time() - started, 1),
        "model_seconds": round(model_seconds, 1),
        "tool_seconds": round(tool_seconds, 1),
        "final_tail": final.strip()[-200:],
        # What the model actually did, so a failure can be diagnosed without
        # re-running the task. Arguments and results are clipped: enough to
        # see the shape of the call, not a second copy of the transcript.
        "calls": [{"name": call["name"],
                   "arguments": json.dumps(call["arguments"],
                                           ensure_ascii=False)[:160],
                   "ok": call["ok"],
                   "result": str(call["result"])[:160],
                   "seconds": call["seconds"]}
                  for call in calls],
    }
    result.update(sum_usage(usages))
    return result


def aggregate_runs(runs):
    """Collapse one task's repeated runs into one result.

    Existing fields stay in place so a run file still compares: turns, calls,
    seconds and tokens are summed across runs. `passes`/`repeats` and `flaky`
    keep a task that only sometimes passes visible as such.
    """
    passes = sum(1 for run in runs if run["ok"])
    repeats = len(runs)
    if repeats == 1:
        result = dict(runs[0])
        result["passes"] = passes
        result["repeats"] = 1
        result["flaky"] = False
        return result

    failures = [run for run in runs if not run["ok"]]
    shown = failures[0] if failures else runs[-1]
    result = {
        "ok": passes == repeats,
        "passes": passes,
        "repeats": repeats,
        "flaky": 0 < passes < repeats,
        "turns": sum(run["turns"] for run in runs),
        "tool_calls": sum(run["tool_calls"] for run in runs),
        "seconds": round(sum(run["seconds"] for run in runs), 1),
        "model_seconds": round(sum(run.get("model_seconds", 0)
                                   for run in runs), 1),
        "tool_seconds": round(sum(run.get("tool_seconds", 0)
                                  for run in runs), 1),
        "final_tail": shown["final_tail"],
        "calls": shown["calls"],
    }
    if failures:
        result["reason"] = "flaky" if passes else failures[0]["reason"]
        result["detail"] = (f"{passes}/{repeats} runs passed; "
                            + "; ".join(f"run {i + 1}: {run['reason']}: "
                                        f"{run['detail']}"
                                        for i, run in enumerate(runs)
                                        if not run["ok"]))
    else:
        result["reason"] = None
        result["detail"] = ""
    result.update(reasoning_check.token_totals(
        {str(i): run for i, run in enumerate(runs)}))
    return result


def chat_live(url, model, messages, max_tokens, timeout):
    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("response has no choices")
    usage = data.get("usage")
    return {"message": choices[0].get("message") or {},
            "usage": usage if isinstance(usage, dict) else {}}


SELFTEST_TASK = {
    "name": "selftest",
    "files": {"value.txt": "42\n"},
    "prompt": "Read value.txt and report the number. End with `ANSWER: <number>`.",
    "answer": r"ANSWER:\s*42\b",
    "max_turns": 4,
}


def _reply(content=None, tool_calls=None, usage=None):
    message = {"content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"message": message, "usage": usage or {}}


def _call(name, arguments, call_id="call_1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


LIST_ANSWER = "\n".join(
    f"{'EXISTS' if exists else 'MISSING'}: {rel}"
    for rel, exists in LIST_CANDIDATES.items())
LIST_INVENTED = LIST_ANSWER.replace("MISSING: project/src/config.py",
                                    "EXISTS: project/src/config.py")
LIST_OMITTED = "\n".join(line for line in LIST_ANSWER.splitlines()
                         if "project/docs/notes.md" not in line)

LIST_SELFTEST_CALLS = [
    _call("list_dir", {"path": rel}, f"list_{i}")
    for i, rel in enumerate(LIST_DIRS)
]

SELFTEST_CASES = [
    {
        "name": "pass",
        "replies": [
            _reply(None, [
                _call("read_file", {"path": "value.txt"}, "call_1"),
                _call("write_file", {"path": "proof.txt", "content": "seen"},
                      "call_2"),
            ], {"prompt_tokens": 100, "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 2}}),
            _reply("ANSWER: 42", usage={"prompt_tokens": 120,
                                        "completion_tokens": 5}),
        ],
        "want": {"ok": True, "reason": None, "turns": 2, "tool_calls": 2,
                 "prompt_tokens": 220, "cached_tokens": 0,
                 "completion_tokens": 15, "reasoning_tokens": 2},
        "expect_file": ("proof.txt", "seen"),
    },
    {
        "name": "no_tool_call",
        "replies": [_reply("ANSWER: 42")],
        "want": {"ok": False, "reason": "no_tool_call", "turns": 1,
                 "tool_calls": 0},
    },
    {
        "name": "turn_cap",
        "max_turns": 2,
        "replies": [
            _reply(None, [_call("list_dir", {"path": "."})]),
            _reply(None, [_call("list_dir", {"path": "."})]),
            _reply("ANSWER: 42"),
        ],
        "want": {"ok": False, "reason": "turn_cap", "turns": 2,
                 "tool_calls": 2},
    },
    {
        "name": "wrong_answer",
        "replies": [
            _reply(None, [_call("read_file", {"path": "value.txt"})]),
            _reply("ANSWER: 41"),
        ],
        "want": {"ok": False, "reason": "wrong_answer", "turns": 2,
                 "tool_calls": 1},
    },
    {
        "name": "updated_pass",
        "task": "updated_file",
        "replies": [
            _reply(None, [_call("read_file", {"path": "settings.ini"}, "u1")]),
            _reply(None, [_call("write_file",
                                {"path": "settings.ini",
                                 "content": SETTINGS_AFTER}, "u2")]),
            _reply(None, [_call("read_file", {"path": "settings.ini"}, "u3")]),
            _reply("ANSWER: mode=final"),
        ],
        "want": {"ok": True, "reason": None, "turns": 4, "tool_calls": 3},
    },
    {
        "name": "updated_stale_answer",
        "task": "updated_file",
        "replies": [
            _reply(None, [_call("read_file", {"path": "settings.ini"}, "u4")]),
            _reply(None, [_call("write_file",
                                {"path": "settings.ini",
                                 "content": SETTINGS_AFTER}, "u5")]),
            _reply("ANSWER: mode=draft"),
        ],
        "want": {"ok": False, "reason": "not_reread", "turns": 3,
                 "tool_calls": 2},
    },
    {
        "name": "updated_stale_value",
        "task": "updated_file",
        "replies": [
            _reply(None, [_call("read_file", {"path": "settings.ini"}, "u6")]),
            _reply(None, [_call("write_file",
                                {"path": "settings.ini",
                                 "content": SETTINGS_AFTER}, "u7")]),
            _reply(None, [_call("read_file", {"path": "settings.ini"}, "u8")]),
            _reply("ANSWER: mode=draft"),
        ],
        "want": {"ok": False, "reason": "stale_value", "turns": 4,
                 "tool_calls": 3},
    },
    {
        "name": "read_not_run_pass",
        "task": "pick_read_over_run",
        "replies": [
            _reply(None, [_call("read_file", {"path": "events.log"}, "r1")]),
            _reply("ANSWER: retries=7"),
        ],
        "want": {"ok": True, "reason": None, "turns": 2, "tool_calls": 1},
    },
    {
        "name": "read_not_run_fail",
        "task": "pick_read_over_run",
        "replies": [
            _reply(None, [_call("run_python", {"path": "helper.py"}, "r2")]),
            _reply("ANSWER: retries=99"),
        ],
        "want": {"ok": False, "reason": "wrong_tool", "turns": 2,
                 "tool_calls": 1},
    },
    {
        "name": "search_line_pass",
        "task": "pick_search_over_read",
        "replies": [
            _reply(None, [_call("search", {"pattern": "ZX-3107",
                                           "path": "big.log"}, "s1")]),
            _reply("ANSWER: " + SEARCH_LINE),
        ],
        "want": {"ok": True, "reason": None, "turns": 2, "tool_calls": 1},
    },
    {
        "name": "search_line_read_fail",
        "task": "pick_search_over_read",
        "replies": [
            _reply(None, [_call("read_file", {"path": "big.log"}, "s2")]),
            _reply("ANSWER: not found"),
        ],
        "want": {"ok": False, "reason": "missing_search", "turns": 2,
                 "tool_calls": 1},
    },
    {
        "name": "search_line_wrong_fail",
        "task": "pick_search_over_read",
        "replies": [
            _reply(None, [_call("search", {"pattern": "ZX-3107",
                                           "path": "big.log"}, "s3")]),
            _reply("ANSWER: ZX-3107"),
        ],
        "want": {"ok": False, "reason": "wrong_line", "turns": 2,
                 "tool_calls": 1},
    },
    {
        "name": "list_guess_pass",
        "task": "pick_list_over_guess",
        "replies": [
            _reply(None, LIST_SELFTEST_CALLS),
            _reply(LIST_ANSWER),
        ],
        "want": {"ok": True, "reason": None, "turns": 2, "tool_calls": 5},
    },
    {
        "name": "list_guess_invented",
        "task": "pick_list_over_guess",
        "replies": [
            _reply(None, LIST_SELFTEST_CALLS),
            _reply(LIST_INVENTED),
        ],
        "want": {"ok": False, "reason": "invented", "turns": 2,
                 "tool_calls": 5},
    },
    {
        "name": "list_guess_omitted",
        "task": "pick_list_over_guess",
        "replies": [
            _reply(None, LIST_SELFTEST_CALLS),
            _reply(LIST_OMITTED),
        ],
        "want": {"ok": False, "reason": "omitted", "turns": 2,
                 "tool_calls": 5},
    },
    {
        "name": "no_tool_pass",
        "task": "no_tool_needed",
        "replies": [_reply("ANSWER: 4312")],
        "want": {"ok": True, "reason": None, "turns": 1, "tool_calls": 0},
    },
    {
        "name": "no_tool_called",
        "task": "no_tool_needed",
        "replies": [
            _reply(None, [_call("read_file", {"path": "notes.txt"}, "n1")]),
            _reply("ANSWER: 4312"),
        ],
        "want": {"ok": False, "reason": "tool_called", "turns": 2,
                 "tool_calls": 1},
    },
    {
        "name": "write_verify_pass",
        "task": "write_then_verify",
        "replies": [
            _reply(None, [_call("write_file",
                                {"path": "out/token.txt",
                                 "content": TOKEN_BYTES}, "w1")]),
            _reply(None, [_call("read_file", {"path": "out/token.txt"},
                                "w2")]),
            _reply("ANSWER: 11"),
        ],
        "want": {"ok": True, "reason": None, "turns": 3, "tool_calls": 2},
    },
    {
        "name": "write_verify_bytes",
        "task": "write_then_verify",
        "replies": [
            _reply(None, [_call("write_file",
                                {"path": "out/token.txt",
                                 "content": "seed=314159"}, "w3")]),
            _reply(None, [_call("read_file", {"path": "out/token.txt"},
                                "w4")]),
            _reply("ANSWER: 11"),
        ],
        "want": {"ok": False, "reason": "wrong_file", "turns": 3,
                 "tool_calls": 2},
    },
    {
        "name": "write_verify_reread",
        "task": "write_then_verify",
        "replies": [
            _reply(None, [_call("write_file",
                                {"path": "out/token.txt",
                                 "content": TOKEN_BYTES}, "w5")]),
            _reply("ANSWER: 11"),
        ],
        "want": {"ok": False, "reason": "not_reread", "turns": 2,
                 "tool_calls": 1},
    },
    {
        "name": "write_verify_value",
        "task": "write_then_verify",
        "replies": [
            _reply(None, [_call("write_file",
                                {"path": "out/token.txt",
                                 "content": TOKEN_BYTES}, "w6")]),
            _reply(None, [_call("read_file", {"path": "out/token.txt"},
                                "w7")]),
            _reply("ANSWER: 12"),
        ],
        "want": {"ok": False, "reason": "wrong_value", "turns": 3,
                 "tool_calls": 2},
    },
]


def selftest():
    print("selftest: canned transcripts through the production agent loop")
    failures = []
    with tempfile.TemporaryDirectory(prefix="tooltasks-selftest-") as tmp:
        for case in SELFTEST_CASES:
            task = TASK_BY_NAME.get(case.get("task"), SELFTEST_TASK)
            root = os.path.join(tmp, case["name"])
            replies = list(case["replies"])

            def chat(messages, replies=replies):
                if not replies:
                    return {"message": {"content": ""}, "usage": {}}
                return replies.pop(0)

            result = run_task(task, root, chat,
                              case.get("max_turns", task["max_turns"]))
            problems = []
            for field, expected in case["want"].items():
                if result.get(field) != expected:
                    problems.append(f"{field}={result.get(field)!r}, "
                                    f"expected {expected!r}")
            expect_file = case.get("expect_file")
            if expect_file:
                rel, content = expect_file
                if _read_text(root, rel) != content:
                    problems.append(f"{rel} was not written by the tool call")

            verdict = "ok" if not problems else "FAIL"
            state = "PASS" if result["ok"] else "FAIL"
            reason = result["reason"] or ""
            print(f"  {case['name']:<24} {task['name']:<22} {verdict:<4} "
                  f"{state:<4} {reason:<16} turns {result['turns']}  "
                  f"calls {result['tool_calls']}  {_tokens_text(result)}")
            for problem in problems:
                failures.append(f"{case['name']}: {problem}")

        probe_root = os.path.join(tmp, "stale-read-probe")
        os.makedirs(probe_root)
        with open(os.path.join(probe_root, "settings.ini"), "w",
                  encoding="utf-8") as f:
            f.write(SETTINGS_AFTER)
        stale_pass, stale_reason, _ = check_updated_file(
            probe_root, "",
            [{"name": "write_file",
              "arguments": {"path": "settings.ini"}, "ok": True,
              "result": "wrote settings.ini"},
             {"name": "read_file",
              "arguments": {"path": "settings.ini"}, "ok": True,
              "result": "[duplicate tool output collapsed]"}])
        stale_ok = not stale_pass and stale_reason == "stale_read"

    flaky = aggregate_runs([
        {"ok": True, "reason": None, "detail": "", "turns": 2,
         "tool_calls": 1, "seconds": 0.1, "final_tail": "", "calls": [],
         "prompt_tokens": 10, "cached_tokens": 0,
         "completion_tokens": 5, "reasoning_tokens": 0},
        {"ok": False, "reason": "wrong_answer", "detail": "bad", "turns": 3,
         "tool_calls": 2, "seconds": 0.2, "final_tail": "", "calls": [],
         "prompt_tokens": 20, "cached_tokens": 0,
         "completion_tokens": 6, "reasoning_tokens": 0},
    ])
    aggregate_ok = (not flaky["ok"] and flaky["flaky"]
                    and flaky["passes"] == 1 and flaky["repeats"] == 2
                    and flaky["reason"] == "flaky" and flaky["turns"] == 5
                    and flaky["tool_calls"] == 3
                    and flaky["prompt_tokens"] == 30)
    if not stale_ok:
        failures.append("stale-read: a collapsed re-read was not rejected")
    print(f"  {'stale-read':<24} {'ok' if stale_ok else 'FAIL'}")
    if not aggregate_ok:
        failures.append("repeat-aggregate: a 1/2 run was not reported as flaky")
    print(f"  {'repeat-aggregate':<24} {'ok' if aggregate_ok else 'FAIL'}")

    if failures:
        for line in failures:
            print(f"SELFTEST FAIL: {line}")
        return 1
    print("SELFTEST PASS")
    return 0


def _tokens_text(row):
    if (row.get("prompt_tokens") is None
            and row.get("completion_tokens") is None):
        return "tokens n/a"
    return reasoning_check.token_line(row)


def print_result(name, result):
    repeats = result.get("repeats", 1)
    passes = result.get("passes", 1 if result["ok"] else 0)
    if repeats > 1:
        if passes == repeats:
            state = "PASS"
        elif passes:
            state = "FLAKY"
        else:
            state = "FAIL"
        print(f"  {name:<18} {state:<5} {passes}/{repeats}  "
              f"turns {result['turns']}  calls {result['tool_calls']}  "
              f"{_tokens_text(result)}")
    else:
        state = "PASS" if result["ok"] else "FAIL"
        print(f"  {name:<18} {state}  turns {result['turns']}  "
              f"calls {result['tool_calls']}  {_tokens_text(result)}")
    if "model_seconds" in result:
        print(f"       time {result['seconds']}s = model "
              f"{result['model_seconds']}s + tools {result['tool_seconds']}s")
    if not result["ok"]:
        print(f"       {result['reason']}: {result['detail']}")


def _status(result):
    repeats = result.get("repeats") or 1
    passes = result.get("passes")
    if passes is None or repeats <= 1:
        return "PASS" if result.get("ok") else "FAIL"
    if passes == repeats:
        return f"PASS {passes}/{repeats}"
    if not passes:
        return f"FAIL 0/{repeats}"
    return f"FLAKY {passes}/{repeats}"


def _delta(current, baseline):
    if current is None or baseline is None:
        return "n/a"
    return f"{current - baseline:+d}"


def compare_run(path, baseline, run, filtered=False):
    print(f"\nbaseline {path}: {baseline.get('passed')}/"
          f"{baseline.get('total')}")
    old = baseline.get("results") or {}
    new = run["results"]
    regressions, fixes = [], []

    for name, current in new.items():
        was = old.get(name)
        if was is None:
            print(f"  {name:<18} new -> {_status(current)}")
            continue
        if was.get("ok") and not current.get("ok"):
            regressions.append(name)
        elif not was.get("ok") and current.get("ok"):
            fixes.append(name)
        prompt_delta = _delta(current.get("prompt_tokens"),
                              was.get("prompt_tokens"))
        print(f"  {name:<18} {_status(was)} -> {_status(current)}  "
              f"turns {was.get('turns')} -> {current.get('turns')} "
              f"({_delta(current.get('turns'), was.get('turns'))})  "
              f"calls {was.get('tool_calls')} -> {current.get('tool_calls')} "
              f"({_delta(current.get('tool_calls'), was.get('tool_calls'))})  "
              f"prompt {was.get('prompt_tokens')} -> "
              f"{current.get('prompt_tokens')} ({prompt_delta})")

    if not filtered:
        for name, was in old.items():
            if name in new:
                continue
            print(f"  {name:<18} {_status(was)} -> missing")
            if was.get("ok"):
                regressions.append(name)

    print(f"  total: {baseline.get('passed')}/{baseline.get('total')} -> "
          f"{run['passed']}/{run['total']} "
          f"({_delta(run['passed'], baseline.get('passed'))})")
    print(f"  turns: {baseline.get('turns')} -> {run['turns']} "
          f"({_delta(run['turns'], baseline.get('turns'))})")
    print(f"  tool calls: {baseline.get('tool_calls')} -> "
          f"{run['tool_calls']} "
          f"({_delta(run['tool_calls'], baseline.get('tool_calls'))})")

    now_tokens = run.get("tokens") or {}
    old_tokens = baseline.get("tokens") or {}
    changed = []
    for field in reasoning_check.TOKEN_FIELDS:
        current, was = now_tokens.get(field), old_tokens.get(field)
        if current is not None and was is not None and current != was:
            changed.append(f"{field} {current - was:+d}")
    print(f"  tokens: {', '.join(changed) or 'unchanged'}")
    print(f"regressions: {regressions or 'none'}")
    print(f"newly passing: {fixes or 'none'}")
    return regressions


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="score canned transcripts through the same loop; "
                         "no model needed")
    ap.add_argument("--base", default=BASE,
                    help="OpenAI-compatible base URL or /v1/chat/completions "
                         "endpoint (env BENCH_BASE)")
    ap.add_argument("--model", default=MODEL,
                    help="model name (env BENCH_MODEL)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--repeat", type=int, default=1,
                    help="run each task N times and report passes per task "
                         "(default 1)")
    ap.add_argument("--only", help="comma-separated task names")
    ap.add_argument("--json", metavar="FILE", help="write the run to FILE")
    ap.add_argument("--compare", metavar="FILE",
                    help="print per-task and total deltas against an earlier run")
    ap.add_argument("--fail-on-regression", action="store_true",
                    help="exit 1 when a task that passed in --compare now fails")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.fail_on_regression and not args.compare:
        ap.error("--fail-on-regression requires --compare")
    if args.repeat < 1:
        ap.error("--repeat must be at least 1")

    baseline = None
    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            baseline = json.load(f)

    wanted = None
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        unknown = wanted - {task["name"] for task in TASKS}
        if unknown:
            ap.error(f"unknown task(s): {', '.join(sorted(unknown))}")
    tasks = [task for task in TASKS
             if wanted is None or task["name"] in wanted]

    url = endpoint(args.base)

    def chat(messages):
        return chat_live(url, args.model, messages, args.max_tokens,
                         args.timeout)

    results = {}
    print(f"endpoint {url}")
    print(f"model {args.model}")
    if args.repeat > 1:
        print(f"repeat {args.repeat}")
    with tempfile.TemporaryDirectory(prefix="tooltasks-") as tmp:
        for task in tasks:
            runs = []
            for i in range(args.repeat):
                root = os.path.join(tmp, f"{task['name']}-{i + 1}")
                runs.append(run_task(task, root, chat, task["max_turns"]))
            result = aggregate_runs(runs)
            results[task["name"]] = result
            print_result(task["name"], result)

    passed = sum(1 for result in results.values() if result["ok"])
    task_passes = sum(result["passes"] for result in results.values())
    task_runs = sum(result["repeats"] for result in results.values())
    turns = sum(result["turns"] for result in results.values())
    calls = sum(result["tool_calls"] for result in results.values())
    totals = reasoning_check.token_totals(results)
    print(f"\nscore: {passed}/{len(results)}")
    if task_runs != len(results):
        print(f"passes: {task_passes}/{task_runs} task runs")
    print(f"turns: {turns}  tool calls: {calls}")
    print(f"tokens: {_tokens_text(totals)}")
    wall_total = round(sum(r.get("seconds", 0) for r in results.values()), 1)
    model_total = round(sum(r.get("model_seconds", 0)
                            for r in results.values()), 1)
    tool_total = round(sum(r.get("tool_seconds", 0)
                           for r in results.values()), 1)
    print(f"time: {wall_total}s = model {model_total}s + tools {tool_total}s"
          f" + harness {round(wall_total - model_total - tool_total, 1)}s")

    run = {
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "base": args.base,
        "repeat": args.repeat,
        "passed": passed,
        "total": len(results),
        "task_passes": task_passes,
        "task_runs": task_runs,
        "turns": turns,
        "tool_calls": calls,
        "seconds": wall_total,
        "model_seconds": model_total,
        "tool_seconds": tool_total,
        "tokens": totals,
        "results": results,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(run, f, indent=2)
        print(f"wrote {args.json}")

    if baseline is not None:
        regressions = compare_run(args.compare, baseline, run,
                                  filtered=args.only is not None)
        if regressions and args.fail_on_regression:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
