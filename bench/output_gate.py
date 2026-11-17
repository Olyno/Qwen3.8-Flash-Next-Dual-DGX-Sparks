#!/usr/bin/env python3
"""Answer-correctness gate for speed patches.

Bitwise gating is impossible on this stack: GDN attention has no
batch-invariant mode in vLLM 0.30 and greedy completions drift run to run.
So instead of hashing completions, this checks each prompt's *answer* against
ground truth, N times per prompt, and fails if any run is wrong:

    python3 bench/output_gate.py                 # against localhost:8888
    python3 bench/output_gate.py --runs 5

Exit 1 on any wrong or incomplete answer, including any failed repeat. The
default command also runs a twelve-batch stateful scratch-file tool task.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tool_tasks

CHECKS = [
    ("math",
     "Compute step by step: 17 * 23 + 149. Give the final number last.",
     lambda t: _math_works(t)),
    ("json",
     'Reply with only a JSON object: {"city": "Paris", "country": "France", "eu": true}',
     lambda t: _is_paris_json(t)),
    ("code",
     "Write a Python function that checks whether a string is a palindrome. Code only.",
     lambda t: _palindrome_works(t)),
    ("french",
     "Explique en deux phrases pourquoi le ciel est bleu.",
     lambda t: _french_works(t)),
    ("prose",
     "Write one paragraph of 60 to 180 words about lighthouse history. Include "
     "the ancient Pharos at Alexandria, Augustin Fresnel's lens introduced in "
     "1822, and the later use of electric lights.",
     lambda t: _lighthouse_works(t)),
]


def _math_works(text):
    # This is a short arithmetic task, not a free-form essay.
    if len(text.split()) > 200:
        return False
    return bool(re.search(r"(?<![\w.,])540(?:\.0+)?\s*[}*$\\)\].]*\s*$", text))


def _french_works(text):
    lower = text.lower()
    return (
        15 <= len(text.split()) <= 100
        and re.search(r"\bciel\b", lower) is not None
        and re.search(r"\bbleu[eës]?\b", lower) is not None
        and re.search(r"diffus|rayleigh", lower) is not None
        and re.search(r"\bair\b|atmosph[eè]re|mol[eé]cule", lower) is not None
        and re.search(r"lumi[eè]re|soleil", lower) is not None
        and re.search(r"courte|plus|davantage|mieux", lower) is not None
    )


def _lighthouse_works(text):
    words = re.findall(r"\b[\w'-]+\b", text)
    lower = text.lower()
    return (
        60 <= len(words) <= 180
        and "pharos" in lower
        and "alexandria" in lower
        and re.search(r"\bancient\b|\bantiquity\b|\bthird century\b|\b3rd century\b",
                      lower) is not None
        and "fresnel" in lower
        and re.search(r"\blens(?:es)?\b", lower) is not None
        and re.search(r"\b1822\b", lower) is not None
        and re.search(r"\belectric(?:al|ity)?\b", lower) is not None
    )


def _is_paris_json(t):
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        return False
    return (isinstance(obj, dict) and obj.get("city") == "Paris"
            and obj.get("country") == "France" and obj.get("eu") is True)


CODE_TIMEOUT = 5
PALINDROME_RUNNER = r"""
import resource
import sys

resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)
resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
namespace = {}
try:
    exec(sys.stdin.read(), namespace)
    function = next((value for name, value in namespace.items()
                     if callable(value) and "palindrome" in name.lower()), None)
    good = function is not None and all(
        function(text) is expected for text, expected in (
            ("racecar", True), ("hello", False),
            ("amanaplanacanalpanama", True)))
except BaseException:
    good = False
sys.exit(0 if good else 1)
"""


def _palindrome_works(t):
    m = re.search(r"```(?:python)?\n(.*?)```", t, re.S)
    code = m.group(1) if m else t
    # Resource-bounded trusted execution, NOT a filesystem/network sandbox.
    try:
        with tempfile.TemporaryDirectory(prefix="palindrome-gate-") as root:
            with subprocess.Popen(
                [sys.executable, "-I", "-c", PALINDROME_RUNNER],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, text=True, cwd=root,
                start_new_session=True,
            ) as process:
                try:
                    process.communicate(code, timeout=CODE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.communicate()
                    return False
                return process.returncode == 0
    except (OSError, ValueError):
        return False


def chat(url, model, prompt, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
    choices = data.get("choices") or []
    if len(choices) != 1:
        raise ValueError("expected exactly one completion")
    return {**choices[0], "usage": data.get("usage") or {}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8888")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args(argv)
    if args.runs < 1 or args.max_tokens < 1:
        ap.error("--runs and --max-tokens must be at least 1")

    failures = 0
    for name, prompt, check in CHECKS:
        ok = 0
        for _ in range(args.runs):
            try:
                reply = chat(args.url, args.model, prompt, args.max_tokens)
                error = tool_tasks.completion_error(reply, allow_tools=False)
                if error:
                    print(f"  {name}: incomplete: {error}; "
                          f"finish_reason={reply.get('finish_reason')!r}, "
                          f"stop_reason={reply.get('stop_reason')!r}")
                elif check(tool_tasks.final_content(reply["message"])):
                    ok += 1
            except Exception as exc:
                print(f"  {name}: request error: {exc}")
        status = "ok" if ok == args.runs else "FAIL"
        if ok != args.runs:
            failures += 1
        print(f"  {name}: {ok}/{args.runs} {status}")

    task = tool_tasks.SUSTAINED_TASK
    tools = [tool for tool in tool_tasks.TOOLS
             if tool["function"]["name"] in task["tools"]]

    def tool_chat(messages):
        return tool_tasks.chat_live(tool_tasks.endpoint(args.url), args.model,
                                    messages, args.max_tokens, 600, tools=tools)

    runs = []
    with tempfile.TemporaryDirectory(prefix="output-gate-") as root:
        for i in range(args.runs):
            runs.append(tool_tasks.run_task(
                task, os.path.join(root, str(i)), tool_chat, task["max_turns"]))
    result = tool_tasks.aggregate_runs(runs)
    tool_tasks.print_result(task["name"], result)
    if not result["ok"]:
        failures += 1
    total = len(CHECKS) + 1
    if failures:
        print(f"FAIL: {failures}/{total} checks failed at least one repeat")
        return 1
    print(f"PASS: all {total} checks correct in every one of {args.runs} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
