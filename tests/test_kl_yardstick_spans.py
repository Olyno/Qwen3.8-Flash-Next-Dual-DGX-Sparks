import importlib.util
import math
import os
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "kl_yardstick", os.path.join(ROOT, "bench", "kl_yardstick.py"))
ky = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ky)


class SpanMaskTest(unittest.TestCase):
    def test_empty_or_missing_spans_select_nothing(self):
        self.assertEqual(ky.span_mask(4, []), [False] * 4)
        self.assertEqual(ky.span_mask(4, None), [False] * 4)

    def test_start_inclusive_end_exclusive(self):
        self.assertEqual(ky.span_mask(3, [[1, 2]]), [False, True, False])
        self.assertEqual(ky.span_mask(2, [[0, 2]]), [True, True])

    def test_overlapping_spans_union(self):
        self.assertEqual(ky.span_mask(6, [[0, 3], [2, 5]]),
                         [True, True, True, True, True, False])
        self.assertEqual(ky.span_mask(4, [[1, 3], [1, 2], [2, 4]]),
                         [False, True, True, True])

    def test_out_of_range_is_clamped(self):
        self.assertEqual(ky.span_mask(5, [[-3, 2], [4, 99]]),
                         [True, True, False, False, True])
        self.assertEqual(ky.span_mask(3, [[3, 9], [-5, -1], [2, 1]]),
                         [False, False, False])


def elem(spec):
    return {str(k): {"logprob": v} for k, v in spec.items()}


class DecisionStatsTest(unittest.TestCase):
    """Two fake top-k arms over four tokens. Positions 1 and 3 carry the
    actual token and are the decision positions; position 2 is scored but is
    not a stop decision."""

    ids = [0, 10, 11, 12]
    tp = [None,
          elem({10: -0.5, 99: -2.0}),
          elem({11: -1.0, 99: -2.5}),
          elem({12: -0.25, 98: -3.0})]
    tc = [None,
          elem({10: -0.7, 99: -2.2}),
          elem({11: -0.9, 99: -2.4}),
          elem({12: -0.3, 98: -3.1})]

    def test_ratio_and_perplexity_over_decision_positions(self):
        st = ky.score_pair(self.tp, self.tc, self.ids, self.ids,
                           decision_ids={10, 12})
        dt = ky.decision_stats([st])
        self.assertEqual(dt["n"], 2)
        # mean log ratio = ((0.7-0.5) + (0.3-0.25)) / 2 = 0.125
        self.assertAlmostEqual(dt["geo_ratio"], math.exp(0.125))
        self.assertAlmostEqual(dt["ppl_p"], math.exp((0.5 + 0.25) / 2))
        self.assertAlmostEqual(dt["ppl_c"], math.exp((0.7 + 0.3) / 2))
        self.assertAlmostEqual(dt["mean_kl"], sum(st["decision"]["kl"]) / 2)

    def test_no_matching_decision_ids_is_n_zero(self):
        st = ky.score_pair(self.tp, self.tc, self.ids, self.ids,
                           decision_ids={12345})
        dt = ky.decision_stats([st])
        self.assertEqual(dt["n"], 0)
        self.assertTrue(math.isnan(dt["mean_kl"]))
        self.assertTrue(math.isnan(dt["geo_ratio"]))

    def test_spans_and_decision_ids_compose(self):
        st = ky.score_pair(self.tp, self.tc, self.ids, self.ids,
                           spans=[[1, 2]], decision_ids={10, 12})
        self.assertEqual(st["n"], 1)
        self.assertEqual(len(st["decision"]["kl"]), 1)
        self.assertEqual(st["decision"]["lp_p"], [-0.5])
        self.assertEqual(st["decision"]["lp_c"], [-0.7])


class EffectiveFloorTest(unittest.TestCase):
    def test_larger_p99_wins_in_either_order(self):
        low, high = {"p99_kl": 0.4}, {"p99_kl": 0.9}
        self.assertEqual(ky.effective_floor(low, high), 0.9)
        self.assertEqual(ky.effective_floor(high, low), 0.9)

    def test_single_floor_or_none(self):
        self.assertEqual(ky.effective_floor({"p99_kl": 0.3}, None), 0.3)
        self.assertEqual(ky.effective_floor(None, {"p99_kl": 0.2}), 0.2)
        self.assertIsNone(ky.effective_floor(None, None))


class BatchPassTest(unittest.TestCase):
    def test_concurrent_pass_preserves_input_order(self):
        seen = []

        def fake(url, model, text, k, gen, timeout):
            time.sleep(0.02 * (3 - int(text)))  # later inputs finish first
            seen.append(int(text))
            return [int(text)], []

        class Args:
            topk, gen_tokens, timeout, batch = 1, 1, 1, 3

        orig = ky.completions_logprobs
        ky.completions_logprobs = fake
        try:
            texts = [{"id": str(i), "text": str(i)} for i in (1, 2, 3)]
            out = ky.run_pass("url", "model", texts, Args())
        finally:
            ky.completions_logprobs = orig
        self.assertEqual([ids[0] for ids, _ in out], [1, 2, 3])
        self.assertEqual(sorted(seen), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
