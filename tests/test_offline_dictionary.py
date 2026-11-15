"""Unit test suite for OfflineDictionary.

Covers all 18 mandatory test cases from Phase 3.10.2 specification:
1. Prefix A
2. Prefix AP
3. Prefix APP
4. Prefix B
5. Prefix HEL
6. Case insensitivity (lowercase vs uppercase vs mixed)
7. Non-existent prefix
8. Empty / whitespace prefix
9. Limit top 3
10. Limit top 5
11. Duplicate words
12. Empty words handling
13. Missing data file error
14. Malformed data handling
15. Equal frequency tie-breaking
16. Stable result order across consecutive queries
17. contains() check
18. get_frequency() check
Plus strict prefix adherence verification.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.suggestion.dictionary import OfflineDictionary

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_dictionary.txt"


class TestOfflineDictionary(unittest.TestCase):
    """Test suite strictly verifying OfflineDictionary specification."""

    @classmethod
    def setUpClass(cls) -> None:
        if not FIXTURE_PATH.is_file():
            raise FileNotFoundError(f"Test fixture not found: {FIXTURE_PATH}")
        cls.dict = OfflineDictionary.load(FIXTURE_PATH)

    # 1. Prefix A
    def test_01_prefix_a(self) -> None:
        results = self.dict.search("A", limit=5)
        self.assertGreaterEqual(len(results), 3)
        for word in results:
            self.assertTrue(word.startswith("A"), f"Word '{word}' does not start with 'A'")
        # High frequency words in A: AND (15000), APPLE (10000), ABOUT (9500), AFTER (9000), ABLE (8500)
        self.assertIn("AND", results)
        self.assertIn("APPLE", results)
        self.assertIn("ABOUT", results)

    # 2. Prefix AP
    def test_02_prefix_ap(self) -> None:
        results = self.dict.search("AP", limit=5)
        self.assertGreaterEqual(len(results), 3)
        for word in results:
            self.assertTrue(word.startswith("AP"), f"Word '{word}' does not start with 'AP'")
        # Must contain words starting with AP, MUST NOT contain AND or ABLE
        self.assertIn("APPLE", results)
        self.assertIn("APPLY", results)
        self.assertNotIn("AND", results)
        self.assertNotIn("ABLE", results)

    # 3. Prefix APP
    def test_03_prefix_app(self) -> None:
        results = self.dict.search("APP", limit=5)
        self.assertGreaterEqual(len(results), 2)
        for word in results:
            self.assertTrue(word.startswith("APP"), f"Word '{word}' does not start with 'APP'")
        self.assertIn("APPLE", results)
        self.assertIn("APPLY", results)
        self.assertNotIn("APART", results)
        self.assertNotIn("APRIL", results)

    # 4. Prefix B
    def test_04_prefix_b(self) -> None:
        results = self.dict.search("B", limit=5)
        self.assertGreaterEqual(len(results), 3)
        for word in results:
            self.assertTrue(word.startswith("B"), f"Word '{word}' does not start with 'B'")
        # High frequency in B: BOOK (8000), BOY (7500), BABY (7000), BALL (6000)
        self.assertIn("BOOK", results)
        self.assertIn("BOY", results)

    # 5. Prefix HEL
    def test_05_prefix_hel(self) -> None:
        results = self.dict.search("HEL", limit=5)
        self.assertGreaterEqual(len(results), 3)
        for word in results:
            self.assertTrue(word.startswith("HEL"), f"Word '{word}' does not start with 'HEL'")
        # HELP (9000) should be before HELLO (6000)
        self.assertEqual(results[0], "HELP")
        self.assertEqual(results[1], "HELLO")

    # 6. Prefix lowercase vs uppercase vs mixed
    def test_06_case_insensitivity(self) -> None:
        lower = self.dict.search("ap", limit=5)
        upper = self.dict.search("AP", limit=5)
        mixed = self.dict.search("aP", limit=5)
        self.assertEqual(lower, upper)
        self.assertEqual(upper, mixed)

    # 7. Non-existent prefix
    def test_07_nonexistent_prefix(self) -> None:
        results = self.dict.search("XYZQNOTEXIST")
        self.assertEqual(results, [])

    # 8. Empty prefix
    def test_08_empty_prefix(self) -> None:
        self.assertEqual(self.dict.search(""), [])
        self.assertEqual(self.dict.search("   "), [])
        self.assertEqual(self.dict.search(None), [])

    # 9. Limit top 3
    def test_09_limit_top_3(self) -> None:
        results = self.dict.search("A", limit=3)
        self.assertEqual(len(results), 3)

    # 10. Limit top 5
    def test_10_limit_top_5(self) -> None:
        results = self.dict.search("A", limit=5)
        self.assertEqual(len(results), 5)

    # 11. Duplicate words
    def test_11_duplicate_words(self) -> None:
        d = OfflineDictionary()
        d.add_word("APPLE", 1000)
        d.add_word("apple", 2000)  # duplicate case
        d.add_word("APPLE", 3000)  # duplicate uppercase
        results = d.search("AP", limit=10)
        self.assertEqual(results, ["APPLE"])
        self.assertEqual(d.get_frequency("APPLE"), 3000)
        self.assertEqual(len(d), 1)

    # 12. Empty words handling
    def test_12_empty_words(self) -> None:
        d = OfflineDictionary()
        d.add_word("", 100)
        d.add_word("   ", 200)
        d.add_word("123", 300)
        self.assertEqual(len(d), 0)
        self.assertFalse(d.contains(""))

    # 13. Missing data file
    def test_13_missing_data_file(self) -> None:
        nonexistent = Path("C:/nonexistent_dir_12345/missing_dict.txt")
        with self.assertRaises(FileNotFoundError):
            OfflineDictionary.load(nonexistent)

    # 14. Malformed data handling
    def test_14_malformed_data(self) -> None:
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as tmp:
            tmp.write("# valid line\n")
            tmp.write("GOOD\t5000\n")
            tmp.write("BADFREQ\tnot_an_int\n")
            tmp.write("SPECIAL@CHAR\t1000\n")
            tmp.write("ONLYWORD\n")
            tmp.write("   \n")
            tmp_path = Path(tmp.name)

        try:
            d = OfflineDictionary.load(tmp_path)
            self.assertTrue(d.contains("GOOD"))
            self.assertEqual(d.get_frequency("GOOD"), 5000)
            self.assertTrue(d.contains("ONLYWORD"))
            self.assertEqual(d.get_frequency("ONLYWORD"), 1)
            # Malformed frequency still loads word with default 1
            self.assertTrue(d.contains("BADFREQ"))
            self.assertEqual(d.get_frequency("BADFREQ"), 1)
            # Special chars rejected
            self.assertFalse(d.contains("SPECIAL@CHAR"))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # 15. Equal frequency tie-breaking (alphabetical order)
    def test_15_equal_frequency(self) -> None:
        results = self.dict.search("CA", limit=5)
        # CAN (3000), CAR (3000), CAT (3000)
        self.assertEqual(results, ["CAN", "CAR", "CAT"])

    # 16. Stable order across consecutive queries
    def test_16_stable_order(self) -> None:
        query1 = self.dict.search("A", limit=5)
        query2 = self.dict.search("A", limit=5)
        query3 = self.dict.search("A", limit=5)
        self.assertEqual(query1, query2)
        self.assertEqual(query2, query3)

    # 17. contains() check
    def test_17_contains(self) -> None:
        self.assertTrue(self.dict.contains("APPLE"))
        self.assertTrue(self.dict.contains("apple"))
        self.assertTrue(self.dict.contains("ApPlE"))
        self.assertFalse(self.dict.contains("NOT_IN_DICTIONARY"))
        self.assertFalse(self.dict.contains(""))

    # 18. get_frequency() check
    def test_18_get_frequency(self) -> None:
        self.assertEqual(self.dict.get_frequency("APPLE"), 10000)
        self.assertEqual(self.dict.get_frequency("apple"), 10000)
        self.assertEqual(self.dict.get_frequency("NOT_IN_DICTIONARY"), 0)
        self.assertEqual(self.dict.get_frequency(""), 0)

    # Strict prefix matching adherence across extensive prefixes
    def test_19_strict_prefix_adherence(self) -> None:
        test_prefixes = ["A", "AP", "APP", "B", "BO", "C", "CA", "D", "E", "F", "G", "H", "HEL"]
        for p in test_prefixes:
            matches = self.dict.search(p, limit=20)
            for m in matches:
                self.assertTrue(
                    m.startswith(p),
                    f"Candidate '{m}' does not strictly match prefix '{p}'",
                )


if __name__ == "__main__":
    unittest.main()
