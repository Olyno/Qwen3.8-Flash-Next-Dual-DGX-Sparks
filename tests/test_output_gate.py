import contextlib
import copy
import importlib.util
import io
import json
import runpy
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import output_gate
import tool_tasks


PROSE = (
    "Ancient sailors used coastal fires to find safe harbors, and the Pharos "
    "at Alexandria became one of antiquity's most celebrated lighthouses. "
    "Later towers helped ships navigate dangerous shores and busy ports. "
    "In 1822 Augustin Fresnel introduced a lens that concentrated light into "
    "a powerful beam while using less fuel. Electric lights subsequently "
    "replaced many older lamps, and automation reduced the need for resident "
    "keepers. Today these historic structures remain useful navigation aids "
    "as well as reminders of maritime engineering."
)
ANSWERS = {
    "math": "17 * 23 = 391; 391 + 149 = 540.",
    "json": '{"country":"France","eu":true,"city":"Paris"}',
    "code": "def is_palindrome(s):\n    return s == s[::-1]\n",
    "french": "Le ciel est bleu car les molécules de l'air diffusent la lumière "
              "du Soleil. Les courtes longueurs d'onde bleues sont davantage "
              "diffusées que les rouges, selon la diffusion de Rayleigh.",
    "prose": PROSE,
}


def reply(content=None, calls=None, finish=None, **message_fields):
    message = {"role": "assistant", "content": content, **message_fields}
    if calls is not None:
        message["tool_calls"] = calls
    return {"message": message,
            "finish_reason": finish or ("tool_calls" if calls else "stop"),
            "stop_reason": None, "usage": {"completion_tokens": 1}}


def call(name, **arguments):
    return tool_tasks._call(name, arguments)


class Reconciler:
    """Deterministic API peer; the production runner executes every file tool."""
    def __init__(self, wrong_batch=None, finish="stop", reasoning_field=None):
        self.wrong_batch = wrong_batch
        self.finish = finish
        self.reasoning_field = reasoning_field
        self.seen = []

    def __call__(self, messages):
        self.seen.append(copy.deepcopy(messages))
        step = sum(message["role"] == "tool" for message in messages)
        batch, phase = divmod(step, 3)
        if batch == 12:
            stock = json.loads(messages[-1]["content"])["stock"]
            return reply(f"ANSWER: {stock}", finish=self.finish)
        number = batch + 1
        if phase == 0:
            tool = call("read_file", path=f"batches/{number:02d}.json")
        elif phase == 1:
            incoming = json.loads(messages[-1]["content"])
            stock = 100 if batch == 0 else json.loads(messages[-3]["content"])["stock"]
            stock += incoming["received"] - incoming["shipped"]
            if self.wrong_batch == number:
                stock += 1
            tool = call("write_file", path="inventory.json",
                        content=json.dumps({"batch": number, "stock": stock}))
        else:
            tool = call("read_file", path="inventory.json")
        fields = {self.reasoning_field: "inspect the returned state"} if self.reasoning_field else {}
        return reply(calls=[tool], **fields)


class AnswerChecksTest(unittest.TestCase):
    def test_exact_garbage_fails(self):
        checks = {name: check for name, _, check in output_gate.CHECKS}
        self.assertFalse(checks["prose"]("RAM" * 1000))
        self.assertFalse(checks["math"]("540" * 1000))
        self.assertFalse(checks["math"]("540 " * 1000))
        self.assertFalse(checks["math"]("540 is an intermediate value; final answer 541"))
        self.assertFalse(checks["prose"]("lighthouse " * 1000))
        self.assertFalse(checks["french"]("bleu diffusion " * 1000))
        self.assertFalse(checks["french"]("Le ciel est bleu. La diffusion existe."))

    def test_correct_alternatives_and_legitimate_repetition(self):
        for name, _, check in output_gate.CHECKS:
            with self.subTest(name=name):
                self.assertTrue(check(ANSWERS[name]))
        for text in ("540", "The result is **540**.", r"\boxed{540}",
                     "391 + 149 = 540\nFinal answer: 540.0"):
            with self.subTest(text=text):
                self.assertTrue(output_gate._math_works(text))
        self.assertTrue(output_gate._lighthouse_works(PROSE + " Lighthouses guide ships; lighthouses warn ships."))
        for french in (
            "La lumière solaire est diffusée par l'atmosphère, ce qui rend le "
            "ciel bleu. Le bleu est plus diffusé que le rouge.",
            "Le ciel paraît bleu parce que les molécules de l'air diffusent "
            "davantage la lumière bleue que la rouge. Il s'agit de la diffusion "
            "de Rayleigh.",
        ):
            with self.subTest(french=french):
                self.assertTrue(output_gate._french_works(french))

    def test_reasoning_is_never_an_answer(self):
        for field in ("reasoning", "reasoning_content"):
            with self.subTest(field=field):
                good = reply("540", **{field: "RAM" * 1000})
                self.assertIsNone(tool_tasks.completion_error(good, False))
                self.assertTrue(output_gate._math_works(tool_tasks.final_content(good["message"])))
                bad = reply("541", **{field: "The answer is 540"})
                self.assertFalse(output_gate._math_works(tool_tasks.final_content(bad["message"])))
                empty = reply(None, **{field: "540"})
                self.assertIsNotNone(tool_tasks.completion_error(empty, False))
        self.assertEqual(tool_tasks.final_content(reply("<think>541</think>540")["message"]), "540")
        self.assertIsNotNone(tool_tasks.completion_error(reply("<think>540"), False))

    def test_completion_contract(self):
        bad = [reply("540", finish=finish) for finish in
               ("length", "tool_calls", "content_filter", "error")]
        bad += [reply(""), reply(" \n "), reply("540", calls=[call("read_file", path="x")]),
                reply("540", role="tool"), reply("540", function_call={"name": "x"}),
                {"message": {"content": "540"}}, reply(["540"]), reply("540", calls={"bad": 1})]
        for completion in bad:
            with self.subTest(completion=completion):
                self.assertIsNotNone(tool_tasks.completion_error(completion, False))

    def test_chat_preserves_finish_metadata_and_both_reasoning_fields(self):
        choice = reply("540", reasoning="new channel", reasoning_content="legacy")
        choice["stop_reason"] = 151645
        body = {"choices": [choice], "usage": {"completion_tokens": 9}}
        with patch.object(output_gate.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(body).encode())):
            actual = output_gate.chat("http://example", "m", "math", 100)
        self.assertEqual(actual["finish_reason"], "stop")
        self.assertEqual(actual["stop_reason"], 151645)
        self.assertEqual(actual["message"]["reasoning"], "new channel")
        self.assertEqual(actual["message"]["reasoning_content"], "legacy")
        with patch.object(output_gate.urllib.request, "urlopen", return_value=io.BytesIO(b'{"choices":')):
            with self.assertRaises(json.JSONDecodeError):
                output_gate.chat("http://example", "m", "math", 100)


class StatefulTaskTest(unittest.TestCase):
    def run_task(self, peer, max_turns=50):
        with tempfile.TemporaryDirectory() as root:
            result = tool_tasks.run_task(tool_tasks.SUSTAINED_TASK, root, peer, max_turns)
            self.inventory = json.loads(Path(root, "inventory.json").read_text())
        return result

    def test_real_file_loop_and_ground_truth(self):
        result = self.run_task(Reconciler())
        self.assertTrue(result["ok"], result["detail"])
        self.assertEqual(result["turns"], 37)
        self.assertEqual(result["tool_calls"], 36)
        stock = 100 + sum(batch["received"] - batch["shipped"] for batch in tool_tasks.RECONCILE_BATCHES)
        self.assertEqual(self.inventory, {"batch": 12, "stock": stock})
        self.assertEqual(result["final_tail"], f"ANSWER: {stock}")
        self.assertEqual(result["completions"][-1]["finish_reason"], "stop")

    def test_wrong_intermediate_state_fails_even_with_plausible_final(self):
        result = self.run_task(Reconciler(wrong_batch=7))
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "wrong_state")

    def test_turn_cap_and_incomplete_final_fail(self):
        self.assertEqual(self.run_task(Reconciler(), 20)["reason"], "turn_cap")
        result = self.run_task(Reconciler(finish="length"))
        self.assertFalse(result["ok"])
        self.assertEqual(result["completions"][-1]["finish_reason"], "length")

    def test_prefetched_batch_and_unverified_write_fail(self):
        replies = iter([
            reply(calls=[call("read_file", path=f"batches/{i:02d}.json") for i in range(1, 13)]),
            reply(calls=[call("write_file", path="inventory.json", content='{"batch":12,"stock":444}')]),
            reply("ANSWER: 444"),
        ])
        result = self.run_task(lambda messages: next(replies))
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "wrong_state")

    def test_dependent_read_and_write_in_one_turn_fail(self):
        peer = Reconciler()

        def combined(messages):
            if len(messages) == 2:
                return reply(calls=[
                    call("read_file", path="batches/01.json"),
                    call("write_file", path="inventory.json",
                         content='{"batch":1,"stock":113}'),
                ])
            return peer(messages)

        result = self.run_task(combined)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "missing_step")

    def test_stale_verification_fails(self):
        invoke = tool_tasks.invoke_tool

        def stale(name, arguments, root):
            ok, text = invoke(name, arguments, root)
            if name == "read_file" and arguments["path"] == "inventory.json":
                state = json.loads(text)
                if state["batch"] == 12:
                    text = json.dumps({"batch": 11, "stock": state["stock"]})
            return ok, text

        with patch.object(tool_tasks, "invoke_tool", side_effect=stale):
            result = self.run_task(Reconciler())
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "stale_read")

    def test_correct_tools_do_not_excuse_wrong_final_answer(self):
        peer = Reconciler()

        def wrong_final(messages):
            completion = peer(messages)
            if completion["finish_reason"] == "stop":
                completion["message"]["content"] = "ANSWER: 0"
            return completion

        result = self.run_task(wrong_final)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "wrong_answer")

    def test_reasoning_history_uses_v032_field(self):
        for field in ("reasoning", "reasoning_content"):
            with self.subTest(field=field):
                peer = Reconciler(reasoning_field=field)
                result = self.run_task(peer)
                self.assertTrue(result["ok"], result["detail"])
                assistant = peer.seen[1][2]
                self.assertEqual(assistant["reasoning"], "inspect the returned state")
                self.assertNotIn("reasoning_content", assistant)

    def test_only_file_tools_can_execute(self):
        replies = iter([reply(calls=[call("run_python", path="x.py")]), reply("ANSWER: 444")])
        with patch.object(tool_tasks, "invoke_tool", side_effect=AssertionError("must not execute")):
            result = self.run_task(lambda messages: next(replies))
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "wrong_tool")


class ModuleLoadingTest(unittest.TestCase):
    def test_repo_level_importlib_and_runpy_load_without_bench_path(self):
        source = Path(output_gate.__file__).resolve()
        bench = str(source.parent)
        for loader in ("importlib", "runpy"):
            with self.subTest(loader=loader), patch.object(
                sys, "path", [entry for entry in sys.path if entry != bench]
            ), patch.dict(sys.modules):
                sys.modules.pop("tool_tasks", None)
                sys.modules.pop("reasoning_check", None)
                if loader == "runpy":
                    module = runpy.run_path(str(source))
                    self.assertIn("main", module)
                else:
                    spec = importlib.util.spec_from_file_location("loaded_gate", source)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.assertTrue(callable(module.main))


class GateCommandTest(unittest.TestCase):
    def run_gate(self, failed_math_runs=(), tool_failure=False):
        counts = {name: 0 for name, _, _ in output_gate.CHECKS}
        tool_peers = []

        def request(req, timeout):
            payload = json.loads(req.data)
            if "tools" in payload:
                self.assertEqual({tool["function"]["name"] for tool in payload["tools"]},
                                 {"read_file", "write_file"})
                if len(payload["messages"]) == 2:
                    tool_peers.append(Reconciler(wrong_batch=7 if tool_failure else None))
                completion = tool_peers[-1](payload["messages"])
            else:
                name = next(name for name, prompt, _ in output_gate.CHECKS
                            if prompt == payload["messages"][0]["content"])
                run = counts[name]
                counts[name] += 1
                completion = reply("541" if name == "math" and run in failed_math_runs else ANSWERS[name])
            return io.BytesIO(json.dumps({"choices": [completion]}).encode())

        with patch.object(output_gate.urllib.request, "urlopen", side_effect=request), contextlib.redirect_stdout(io.StringIO()) as log:
            code = output_gate.main(["--url", "http://fake", "--model", "m", "--runs", "5"])
        return code, log.getvalue(), tool_peers

    def test_default_gate_command_exercises_sustained_task(self):
        code, log, peers = self.run_gate()
        self.assertEqual(code, 0, log)
        self.assertEqual(len(peers), 5)
        self.assertTrue(all(len(peer.seen) == 37 for peer in peers))
        self.assertIn("all 6 checks correct", log)

    def test_two_failed_repeats_are_not_majority_pass(self):
        code, log, _ = self.run_gate(failed_math_runs=(1, 4))
        self.assertEqual(code, 1, log)
        self.assertIn("math: 3/5 FAIL", log)

    def test_tool_state_failure_fails_gate_command(self):
        code, log, _ = self.run_gate(tool_failure=True)
        self.assertEqual(code, 1, log)
        self.assertIn("reconcile_inventory", log)
        self.assertIn("wrong_state", log)

    def test_repeated_incomplete_completions_fail(self):
        for completion in (reply("540", finish="length"), reply(None, reasoning="540"),
                           reply("540", calls=[call("read_file", path="x")])):
            with self.subTest(completion=completion), patch.object(output_gate, "CHECKS", [("math", "p", output_gate._math_works)]), patch.object(output_gate, "chat", return_value=completion), patch.object(tool_tasks, "chat_live", side_effect=Reconciler()), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(output_gate.main(["--runs", "1"]), 1)

    def test_repeated_french_garbage_fails_consumer(self):
        with patch.object(output_gate, "CHECKS", [
            ("french", "p", output_gate._french_works)
        ]), patch.object(
            output_gate, "chat", return_value=reply("bleu diffusion " * 1000)
        ), patch.object(
            tool_tasks, "chat_live", side_effect=Reconciler()
        ), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(output_gate.main(["--runs", "1"]), 1)

    def test_nonterminating_code_fails_consumer_with_bounded_execution(self):
        for code in (
            "while True:\\n    pass",
            "def palindrome(text):\\n    while True:\\n        pass",
        ):
            with self.subTest(code=code), patch.object(
                output_gate, "CHECKS", [("code", "p", output_gate._palindrome_works)]
            ), patch.object(
                output_gate, "chat", return_value=reply(code)
            ), patch.object(
                output_gate, "CODE_TIMEOUT", 0.2
            ), patch.object(
                tool_tasks, "chat_live", side_effect=Reconciler()
            ), contextlib.redirect_stdout(io.StringIO()):
                started = time.monotonic()
                self.assertEqual(output_gate.main(["--runs", "1"]), 1)
                self.assertLess(time.monotonic() - started, 10)


if __name__ == "__main__":
    unittest.main()
