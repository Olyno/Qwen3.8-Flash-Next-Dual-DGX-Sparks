#!/usr/bin/env python3
"""Pure-python unit tests for the PLE mmap coalesced-gather plan function.

_ple_gather_plan lives at module level in patches/patch_ple_mmap_v030.py
and is mirrored verbatim into the overlay (inspect.getsource), so it can be
imported and tested without vLLM, torch, or a GPU. Case list covers the
reader-test situations: ordered, unordered, duplicate, same-page, gap at
and under one page, page-boundary-straddling rows, far-apart rows, and the
structural invariants a scatter relies on.
"""
import importlib.util
import random
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "patches" / "patch_ple_mmap_v030.py"

_spec = importlib.util.spec_from_file_location("patch_ple_mmap_v030", SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
plan = _mod._ple_gather_plan

RB = 90        # packed NVFP4 row width (codes + scales), rows/4KiB page = 45
PAGE = 4096


def rows_of(plan_out):
    return [r for _s, _l, rows in plan_out for r in rows]


class TestPleGatherPlan(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(plan([], RB), [])

    def test_single_row_exact_range(self):
        self.assertEqual(plan([500], RB), [(500, RB, [0])])

    def test_ordered_adjacent_rows_coalesce(self):
        self.assertEqual(plan([0, RB], RB), [(0, 2 * RB, [0, 1])])

    def test_duplicate_offsets_fold_into_one_range(self):
        p = plan([700, 700], RB)
        self.assertEqual(p, [(700, RB, [0, 1])])

    def test_unsorted_input_returns_sorted_plan(self):
        # 100/200 fold (gap < page), 5000 starts a new range (gap 4710 > page)
        self.assertEqual(plan([5000, 100, 200], RB),
                         [(100, 290 - 100, [1, 2]), (5000, RB, [0])])

    def test_same_page_rows_merge_despite_large_gap(self):
        # 4000 + 90 stays inside page 0, so the two rows share a page
        self.assertEqual(plan([0, 4000], RB), [(0, 4000 + RB, [0, 1])])

    def test_gap_of_exactly_one_page_splits(self):
        # next start 4186 = RB + PAGE: gap == page is NOT < one page, and
        # the last byte (89) and the new start sit on different pages
        self.assertEqual(plan([0, RB + PAGE], RB), [(0, RB, [0]), (RB + PAGE, RB, [1])])

    def test_gap_of_one_page_minus_one_merges(self):
        self.assertEqual(plan([0, RB + PAGE - 1], RB), [(0, RB + PAGE - 1 + RB, [0, 1])])

    def test_boundary_straddling_row_widens_the_range(self):
        # row at 4060 straddles pages 0/1 (4060+90 = 4150); the merge test
        # must use the last byte of the range, not its start
        p = plan([4060, 8190], RB)
        self.assertEqual(p, [(4060, 8190 + RB - 4060, [0, 1])])
        self.assertGreaterEqual(p[0][0] + p[0][1], 4150)

    def test_far_apart_rows_stay_split(self):
        offs = [0, 10_000_000, 500_000_000]
        self.assertEqual(plan(offs, RB), [(o, RB, [i]) for i, o in enumerate(offs)])

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            plan([0], 0)
        with self.assertRaises(ValueError):
            plan([0], RB, 0)

    def test_invariants_on_seeded_random_offsets(self):
        rnd = random.Random(20261021)
        offs = rnd.sample(range(0, 1 << 30, RB), 200)
        offs += offs[:20]  # duplicates on purpose
        p = plan(offs, RB)
        # every input row appears exactly once
        seen = sorted(rows_of(p))
        self.assertEqual(seen, list(range(len(offs))))
        # ranges ascend, never overlap, and are exactly tight
        prev_end = -1
        for start, length, rows in p:
            self.assertGreaterEqual(start, prev_end)
            ends = [offs[r] + RB for r in rows]
            self.assertEqual(min(offs[r] for r in rows), start)
            self.assertEqual(max(ends) - start, length)
            prev_end = start + length
            # each row is covered by its range
            for r in rows:
                self.assertTrue(start <= offs[r] and offs[r] + RB <= start + length)

    def test_custom_page_size(self):
        # tiny page (8): the 10-byte gap is now >= one page -> split...
        self.assertEqual(plan([0, 100], RB, page_size=8),
                         [(0, RB, [0]), (100, RB, [1])])
        # bigger page (64): same gap folds again
        self.assertEqual(plan([0, 100], RB, page_size=64), [(0, 100 + RB, [0, 1])])


if __name__ == "__main__":
    unittest.main()
