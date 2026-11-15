"""Unit tests for SuggestionContext parser.
"""
from __future__ import annotations

import unittest
from app.suggestion.context import ParsedContext, SuggestionContext


class TestSuggestionContext(unittest.TestCase):
    """Test suite for SuggestionContext parser."""

    def test_empty_and_none(self):
        ctx1 = SuggestionContext.parse("")
        self.assertEqual(ctx1.raw_text, "")
        self.assertEqual(ctx1.committed_text, "")
        self.assertEqual(ctx1.current_prefix, "")
        self.assertEqual(ctx1.last_committed_word, "")
        self.assertEqual(ctx1.recent_words, [])
        self.assertFalse(ctx1.has_prefix)
        self.assertFalse(ctx1.has_context)

        ctx2 = SuggestionContext.parse(None)
        self.assertEqual(ctx2.current_prefix, "")
        self.assertEqual(ctx2.committed_text, "")

    def test_single_character(self):
        ctx = SuggestionContext.parse("A")
        self.assertEqual(ctx.current_prefix, "A")
        self.assertEqual(ctx.committed_text, "")
        self.assertEqual(ctx.recent_words, [])
        self.assertTrue(ctx.has_prefix)
        self.assertFalse(ctx.has_context)

    def test_multiple_characters_no_space(self):
        ctx = SuggestionContext.parse("APP")
        self.assertEqual(ctx.current_prefix, "APP")
        self.assertEqual(ctx.committed_text, "")
        self.assertEqual(ctx.recent_words, [])

    def test_single_word_with_trailing_space(self):
        ctx = SuggestionContext.parse("APPLE ")
        self.assertEqual(ctx.current_prefix, "")
        self.assertEqual(ctx.committed_text, "APPLE")
        self.assertEqual(ctx.last_committed_word, "APPLE")
        self.assertEqual(ctx.recent_words, ["APPLE"])
        self.assertFalse(ctx.has_prefix)
        self.assertTrue(ctx.has_context)

    def test_multiple_words_with_prefix(self):
        ctx = SuggestionContext.parse("I WANT TO LE")
        self.assertEqual(ctx.current_prefix, "LE")
        self.assertEqual(ctx.committed_text, "I WANT TO")
        self.assertEqual(ctx.last_committed_word, "TO")
        self.assertEqual(ctx.recent_words, ["I", "WANT", "TO"])
        self.assertTrue(ctx.has_prefix)
        self.assertTrue(ctx.has_context)

    def test_multiple_consecutive_spaces(self):
        ctx = SuggestionContext.parse("HELLO    WORLD    AP")
        self.assertEqual(ctx.current_prefix, "AP")
        self.assertEqual(ctx.committed_text, "HELLO WORLD")
        self.assertEqual(ctx.last_committed_word, "WORLD")
        self.assertEqual(ctx.recent_words, ["HELLO", "WORLD"])

    def test_lowercase_normalized_to_uppercase(self):
        ctx = SuggestionContext.parse("she ate an ap")
        self.assertEqual(ctx.current_prefix, "AP")
        self.assertEqual(ctx.committed_text, "SHE ATE AN")
        self.assertEqual(ctx.last_committed_word, "AN")

    def test_trailing_multiple_spaces(self):
        ctx = SuggestionContext.parse("HELLO WORLD   ")
        self.assertEqual(ctx.current_prefix, "")
        self.assertEqual(ctx.committed_text, "HELLO WORLD")
        self.assertEqual(ctx.last_committed_word, "WORLD")

    def test_spaces_only(self):
        ctx = SuggestionContext.parse("     ")
        self.assertEqual(ctx.current_prefix, "")
        self.assertEqual(ctx.committed_text, "")

    def test_sentinels_filtered(self):
        ctx = SuggestionContext.parse("UNCERTAIN")
        self.assertEqual(ctx.current_prefix, "")

        ctx2 = SuggestionContext.parse("HELLO UNCERTAIN AP")
        self.assertEqual(ctx2.current_prefix, "AP")
        self.assertEqual(ctx2.committed_text, "HELLO")
        self.assertEqual(ctx2.last_committed_word, "HELLO")

    def test_non_alphabetic_in_prefix_stripped(self):
        ctx = SuggestionContext.parse("HELLO AP123")
        self.assertEqual(ctx.current_prefix, "AP")
        self.assertEqual(ctx.committed_text, "HELLO")


if __name__ == "__main__":
    unittest.main()
