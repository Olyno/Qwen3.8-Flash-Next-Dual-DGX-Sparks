import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "omniscience", Path(__file__).resolve().parent.parent / "bench" / "omniscience.py"
)
omniscience = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(omniscience)


class Classify(unittest.TestCase):
    def test_incorrect_wins_over_correct_substring(self):
        self.assertEqual(omniscience.classify("INCORRECT"), "INCORRECT")
        self.assertEqual(omniscience.classify("The answer is incorrect."), "INCORRECT")

    def test_correct_still_parses(self):
        self.assertEqual(omniscience.classify("CORRECT"), "CORRECT")
        self.assertEqual(omniscience.classify("correct."), "CORRECT")

    def test_partial_and_not_attempted(self):
        self.assertEqual(omniscience.classify("PARTIAL_ANSWER"), "PARTIAL_ANSWER")
        self.assertEqual(omniscience.classify("NOT_ATTEMPTED"), "NOT_ATTEMPTED")

    def test_unparseable_is_not_attempted(self):
        self.assertEqual(omniscience.classify("I cannot grade this."), "NOT_ATTEMPTED")


if __name__ == "__main__":
    unittest.main()
