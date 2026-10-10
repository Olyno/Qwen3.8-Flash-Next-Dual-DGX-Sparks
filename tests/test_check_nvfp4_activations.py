"""Test for scripts/check_nvfp4_activations.py: runs its synthetic self-check."""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import check_nvfp4_activations


class CheckNvfp4Activations(unittest.TestCase):
    def test_self_check(self):
        check_nvfp4_activations._self_check()


if __name__ == "__main__":
    unittest.main()
