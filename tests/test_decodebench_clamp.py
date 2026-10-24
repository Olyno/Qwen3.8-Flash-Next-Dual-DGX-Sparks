import importlib.util
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
spec = importlib.util.spec_from_file_location(
    "decodebench", os.path.join(ROOT, "bench", "decodebench.py"))
db = importlib.util.module_from_spec(spec)
spec.loader.exec_module(db)


class ClampCtxTest(unittest.TestCase):
    def test_fits_untouched(self):
        self.assertEqual(db.clamp_ctx(1000, 600, 65536), 1000)

    def test_clamps_to_leave_decode_room(self):
        self.assertEqual(db.clamp_ctx(65000, 600, 65536), 65536 - 600 - 64)

    def test_unknown_limit_is_passthrough(self):
        self.assertEqual(db.clamp_ctx(65000, 600, None), 65000)

    def test_tiny_limit_never_zero(self):
        self.assertEqual(db.clamp_ctx(1000, 600, 512), 1)

    def test_margin_override(self):
        self.assertEqual(db.clamp_ctx(10000, 100, 5000, margin=0), 4900)


class ShrinkCtxTest(unittest.TestCase):
    def test_no_overshoot_keeps_context(self):
        self.assertEqual(db.shrink_ctx(1000, 1100, 600, 65536), 1000)

    def test_overshoot_backs_off_with_margin(self):
        new = db.shrink_ctx(65000, 70000, 600, 65536)
        self.assertLess(new, 65000)
        self.assertGreater(new, 50000)

    def test_never_below_one(self):
        self.assertEqual(db.shrink_ctx(10, 99999, 600, 65536), 1)


if __name__ == "__main__":
    unittest.main()
