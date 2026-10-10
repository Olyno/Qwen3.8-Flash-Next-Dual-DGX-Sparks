"""Format check for the shipped vocab/draft_vocab_*.txt files.

patches/patch_mtp_draft_vocab.py's _attach_draft_vocab expects one integer
token id per line; scripts/build_draft_vocab.py writes them sorted, unique,
and below the checkpoint's vocab_size (248320). Stdlib only.
"""
import unittest
from pathlib import Path

VOCAB = Path(__file__).resolve().parent.parent / "vocab"
VOCAB_SIZE = 248320  # checkpoint config.json vocab_size


class DraftVocabFiles(unittest.TestCase):
    def test_files_well_formed(self):
        files = sorted(VOCAB.glob("draft_vocab_*.txt"))
        self.assertTrue(files, "no draft vocab files found")
        for path in files:
            with self.subTest(file=path.name):
                ids = [int(line) for line in path.read_text().splitlines() if line.strip()]
                self.assertTrue(ids)
                self.assertEqual(ids, sorted(ids), "not sorted")
                self.assertEqual(len(ids), len(set(ids)), "duplicate ids")
                self.assertGreaterEqual(min(ids), 0)
                self.assertLess(max(ids), VOCAB_SIZE)

    def test_language_variants_extend_en_47k(self):
        base = set((VOCAB / "draft_vocab_en_code_47k.txt").read_text().split())
        self.assertEqual(len(base), 47149)
        for path in VOCAB.glob("draft_vocab_*_65k.txt"):
            with self.subTest(file=path.name):
                self.assertTrue(base <= set(path.read_text().split()))


if __name__ == "__main__":
    unittest.main()
