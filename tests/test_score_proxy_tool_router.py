import copy
import importlib.util
import json
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

    def test_nudge_is_a_trailing_user_message(self):
        tr = sp.ToolRouter(FakeScorer(0.9, 0.9))
        req = req_with()
        changed, prob = tr.apply(req, "m")
        self.assertTrue(changed)
        self.assertEqual(req["messages"][0]["content"],
                         "fix the failing test")
        self.assertEqual(req["messages"][-1]["role"], "user")
        self.assertIn(sp.TOOL_NUDGE, req["messages"][-1]["content"])

    def test_nudge_leaves_the_head_and_is_idempotent(self):
        tr = sp.ToolRouter(FakeScorer(0.9, 0.9))
        req = req_with(system="base")
        head = copy.deepcopy(req["messages"])
        changed, _ = tr.apply(req, "m")
        self.assertTrue(changed)
        self.assertEqual(req["messages"][:len(head)], head)
        self.assertEqual(req["messages"][-1]["role"], "user")
        self.assertIn(sp.TOOL_NUDGE, req["messages"][-1]["content"])
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


def tools(n):
    return [{"type": "function", "function": {"name": f"t{i}",
                                              "description": f"tool {i}"}}
            for i in range(n)]


def embed_ranked(texts):
    # query = [1, 0]; tool i = [1, i] -> cosine falls with i
    return [[1.0, 0.0]] + [[1.0, float(i)] for i in range(len(texts) - 1)]


class ToolPruneTest(unittest.TestCase):
    def test_prunes_to_top_k_and_keeps_always(self):
        tr = sp.ToolRouter(FakeScorer(0.1, 0.1), embed_fn=embed_ranked,
                           top_k=3, always=("t9",))
        req = req_with()
        req["tools"] = tools(10)
        changed, _ = tr.apply(req, "m")
        self.assertTrue(changed)
        names = [t["function"]["name"] for t in req["tools"]]
        self.assertEqual(names, ["t0", "t1", "t2", "t9"])

    def test_no_prune_under_top_k(self):
        tr = sp.ToolRouter(FakeScorer(0.1, 0.1), embed_fn=embed_ranked, top_k=8)
        req = req_with()
        req["tools"] = tools(5)
        changed, _ = tr.apply(req, "m")
        self.assertFalse(changed)
        self.assertEqual(len(req["tools"]), 5)

    def test_no_prune_without_embed_fn(self):
        tr = sp.ToolRouter(FakeScorer(0.1, 0.1), top_k=3)
        req = req_with()
        req["tools"] = tools(10)
        changed, _ = tr.apply(req, "m")
        self.assertFalse(changed)
        self.assertEqual(len(req["tools"]), 10)

    def test_embed_failure_keeps_all_tools(self):
        def boom(texts):
            raise OSError("down")
        tr = sp.ToolRouter(FakeScorer(0.1, 0.1), embed_fn=boom, top_k=3)
        req = req_with()
        req["tools"] = tools(10)
        changed, _ = tr.apply(req, "m")
        self.assertFalse(changed)
        self.assertEqual(len(req["tools"]), 10)


ARG_SCHEMA = {"type": "object", "properties": {
    "count": {"type": "integer"},
    "ratio": {"type": "number"},
    "verbose": {"type": "boolean"},
    "tags": {"type": "array", "items": {"type": "string"}},
    "items": {"type": "array", "items": {"type": "integer"}},
    "opts": {"type": "object",
             "properties": {"n": {"type": "integer"}}},
    "query": {"type": "string"},
    "nullable": {"type": ["integer", "null"]},
    "untyped": {"description": "no declared type"},
}}


class ToolArgumentCoercionTest(unittest.TestCase):
    def test_declared_types_are_coerced(self):
        cases = [
            ({"count": "5"}, {"count": 5}),
            ({"count": " 5 "}, {"count": 5}),
            ({"count": "5ms"}, {"count": 5}),
            ({"count": "5.0"}, {"count": 5}),
            ({"count": "five"}, {"count": "five"}),
            ({"count": True}, {"count": True}),
            ({"ratio": "3.5"}, {"ratio": 3.5}),
            ({"ratio": "2"}, {"ratio": 2}),
            ({"verbose": "true"}, {"verbose": True}),
            ({"verbose": " FALSE "}, {"verbose": False}),
            ({"tags": "alpha"}, {"tags": ["alpha"]}),
            ({"tags": '["a", "b"]'}, {"tags": ["a", "b"]}),
            ({"items": "7"}, {"items": [7]}),
            ({"opts": '{"n": "2"}'}, {"opts": {"n": 2}}),
            ({"opts": {"n": "2"}}, {"opts": {"n": 2}}),
            ({"nullable": "7"}, {"nullable": 7}),
        ]
        for given, expected in cases:
            with self.subTest(given):
                self.assertEqual(sp.coerce_arguments(given, ARG_SCHEMA),
                                 expected)

    def test_undeclared_and_string_values_stay_verbatim(self):
        given = {"query": "5", "extra": "5", "untyped": "5"}
        self.assertEqual(sp.coerce_arguments(given, ARG_SCHEMA), given)

    def test_absent_arguments_are_not_invented(self):
        given = {"count": 5}
        self.assertEqual(sp.coerce_arguments(given, ARG_SCHEMA), {"count": 5})

    def test_input_is_not_mutated(self):
        given = {"count": "5"}
        sp.coerce_arguments(given, ARG_SCHEMA)
        self.assertEqual(given, {"count": "5"})

    def test_wellformed_values_are_byte_equivalent_after_json_roundtrip(self):
        given = {"count": 5, "ratio": 3.5, "verbose": True,
                 "tags": ["a"], "items": [1, 2], "opts": {"n": 2},
                 "nullable": None, "query": "5"}
        coerced = sp.coerce_arguments(given, ARG_SCHEMA)
        self.assertEqual(json.dumps(coerced), json.dumps(given))


if __name__ == "__main__":
    unittest.main()
