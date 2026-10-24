import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "score_proxy", os.path.join(ROOT, "proxy", "score_proxy.py"))
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)


class FakeScorer:
    """Returns a scripted probability for 'use_tool' on each call; the proxy
    calls it twice (forward and reversed option order)."""

    def __init__(self, forward, reversed_):
        self.calls = [forward, reversed_]

    def __call__(self, model, prefix, labels):
        p = self.calls.pop(0)
        return [p, 1.0 - p] if labels[0] == "use_tool" else [1.0 - p, p]


def req_with(tools=True, system=None, **extra):
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": "fix the failing test"})
    req = {"model": "m", "messages": messages}
    if tools:
        req["tools"] = [{"type": "function", "function": {"name": "run"}}]
    req.update(extra)
    return req


class ToolRouterTest(unittest.TestCase):
    def test_orders_averaged(self):
        tr = sp.ToolRouter(FakeScorer(0.9, 0.7), threshold=0.5)
        self.assertAlmostEqual(tr.needs_tool(req_with()["messages"], "m"), 0.8)

    def test_below_threshold_passthrough(self):
        tr = sp.ToolRouter(FakeScorer(0.3, 0.3), threshold=0.5)
        req = req_with(system="base")
        changed, prob = tr.apply(req, "m")
        self.assertFalse(changed)
        self.assertEqual(req["messages"][0]["content"], "base")
        self.assertNotIn("tool_choice", req)

    def test_nudge_inserted_without_system(self):
        tr = sp.ToolRouter(FakeScorer(0.9, 0.9))
        req = req_with()
        changed, prob = tr.apply(req, "m")
        self.assertTrue(changed)
        self.assertEqual(req["messages"][0]["role"], "system")
        self.assertIn(sp.TOOL_NUDGE, req["messages"][0]["content"])

    def test_nudge_appended_to_system_and_idempotent(self):
        tr = sp.ToolRouter(FakeScorer(0.9, 0.9))
        req = req_with(system="base")
        changed, _ = tr.apply(req, "m")
        self.assertTrue(changed)
        self.assertTrue(req["messages"][0]["content"].startswith("base"))
        self.assertIn(sp.TOOL_NUDGE, req["messages"][0]["content"])
        changed, _ = tr.apply(req, "m")
        self.assertFalse(changed)

    def test_force_sets_tool_choice_only(self):
        tr = sp.ToolRouter(FakeScorer(0.9, 0.9), force=True)
        req = req_with(system="base")
        changed, _ = tr.apply(req, "m")
        self.assertTrue(changed)
        self.assertEqual(req["tool_choice"], "required")
        self.assertEqual(req["messages"][0]["content"], "base")

    def test_force_respects_client_tool_choice(self):
        tr = sp.ToolRouter(FakeScorer(0.9, 0.9), force=True)
        req = req_with(tool_choice="auto")
        changed, _ = tr.apply(req, "m")
        self.assertFalse(changed)
        self.assertEqual(req["tool_choice"], "auto")

    def test_no_tools_noop(self):
        tr = sp.ToolRouter(FakeScorer(0.9, 0.9))
        req = req_with(tools=False)
        changed, prob = tr.apply(req, "m")
        self.assertFalse(changed)
        self.assertEqual(prob, 0.0)
        self.assertEqual(len(req["messages"]), 1)

    def test_scorer_failure_fails_open(self):
        def boom(model, prefix, labels):
            raise OSError("down")
        tr = sp.ToolRouter(boom)
        req = req_with()
        changed, prob = tr.apply(req, "m")
        self.assertFalse(changed)
        self.assertEqual(prob, 0.0)


if __name__ == "__main__":
    unittest.main()
