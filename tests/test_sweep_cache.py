#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""bench/sweep.py persists measured configurations to a JSON cache so re-runs
skip them; the key covers the swept parameters plus a serving-config
fingerprint that must track the active recipe file."""
import os, shutil, sys, tempfile, unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
import sweep


class Args:
    tag, max_tokens, note = "K3", 600, ""


class TestSweepCache(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_cache_round_trip(self):
        path = self.tmp / "sweep_cache.json"
        cache = {"K3|prose|2|0|600||abc": {"S": 2, "ms_per_step": 12.3}}
        sweep.save_cache(str(path), cache)
        self.assertEqual(sweep.load_cache(str(path)), cache)

    def test_load_missing_returns_empty(self):
        self.assertEqual(sweep.load_cache(str(self.tmp / "nope.json")), {})

    def test_key_covers_swept_params_and_fingerprint(self):
        key = sweep.cache_key(Args(), "prose", 2, 0, "fp1")
        self.assertNotEqual(key, sweep.cache_key(Args(), "code", 2, 0, "fp1"))
        self.assertNotEqual(key, sweep.cache_key(Args(), "prose", 4, 0, "fp1"))
        self.assertNotEqual(key, sweep.cache_key(Args(), "prose", 2, 1, "fp1"))
        self.assertNotEqual(key, sweep.cache_key(Args(), "prose", 2, 0, "fp2"))

    def test_fingerprint_tracks_recipe(self):
        old = os.environ.pop("RECIPE", None)
        if old is not None:
            self.addCleanup(os.environ.__setitem__, "RECIPE", old)
        (self.tmp / "recipes").mkdir()
        (self.tmp / "recipes" / "prod.yaml").write_text("a: 1\n")
        fp1 = sweep.fingerprint(str(self.tmp))
        (self.tmp / "recipes" / "prod.yaml").write_text("a: 2\n")
        self.assertNotEqual(fp1, sweep.fingerprint(str(self.tmp)))


if __name__ == "__main__":
    unittest.main()
