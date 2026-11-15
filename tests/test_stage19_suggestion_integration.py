"""Stage 19 — Word and Language Suggestion Integration Tests.

Verifies:
1. Commit character -> updates word suggestions immediately.
2. Commit next character -> updates prefix and candidate suggestions.
3. SPACE -> transitions to word boundary (clears word suggestions, commits word).
4. Select suggestion -> commits chosen word with exactly one trailing space once.
5. Ignore suggestion -> text is never auto-replaced; clicking outside changes nothing.
6. Ollama offline / unreachable -> character recognition and Tier 1 Trie suggestions unaffected.
7. Sentence suggestions update when eligible (word boundary with committed context).
8. Stale suggestions from older generations are strictly discarded.
"""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path

from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from app.inference.unified_abc123_jz_auto_experiment import (
    compute_sentence_hitboxes,
    compute_suggestion_hitboxes,
    render,
)
from app.inference.unified_static_coordinator import TextComposer
from app.suggestion.context import SuggestionContext
from app.suggestion.dictionary import OfflineDictionary
from app.suggestion.manager import SuggestionManager
from app.suggestion.qwen_ranker import QwenRanker, RankResult
from app.suggestion.sentence_generator import SentenceGenerator, SentenceResult
from src.recognition.alphabet import AlphabetRecognizer
from tests.test_unified_abc123_jz_auto import DynamicModel
from tests.test_unified_camera_experiment import FakePredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DICT_PATH = PROJECT_ROOT / "data/dictionary/english_words_25k.tsv"


class MockFailingRanker:
    """Mock ranker simulating Ollama service completely offline."""

    def __init__(self):
        self.call_count = 0

    def rank(self, context: str, prefix: str, candidates: list[str]) -> RankResult:
        self.call_count += 1
        return RankResult(
            success=False,
            ranked_words=[],
            error_type="CONNECTION_ERROR",
            error_message="Ollama offline",
            latency_ms=0.5,
        )


class MockFailingSentenceGenerator:
    """Mock sentence generator simulating Ollama offline."""

    def __init__(self):
        self.call_count = 0

    def generate(self, context: str) -> SentenceResult:
        self.call_count += 1
        return SentenceResult(
            success=False,
            suggestions=[],
            error_type="CONNECTION_ERROR",
            error_message="Ollama offline",
            latency_ms=0.5,
        )


class MockDelayedRanker:
    """Mock ranker that adds a controllable delay to test stale-result dropping."""

    def __init__(self, delay_s: float = 0.15):
        self.delay_s = delay_s
        self.call_count = 0

    def rank(self, context: str, prefix: str, candidates: list[str]) -> RankResult:
        self.call_count += 1
        time.sleep(self.delay_s)
        # Reverse order to verify Qwen re-ranking order
        return RankResult(
            success=True,
            ranked_words=list(reversed(candidates)),
            latency_ms=self.delay_s * 1000.0,
        )


class MockWorkingSentenceGenerator:
    """Mock sentence generator returning realistic continuations."""

    def __init__(self, continuations: list[str] | None = None):
        self.continuations = continuations or ["DOING TODAY?", "FEELING NOW?", "ALL RIGHT?"]
        self.call_count = 0

    def generate(self, context: str) -> SentenceResult:
        self.call_count += 1
        return SentenceResult(
            success=True,
            suggestions=list(self.continuations),
            latency_ms=10.0,
        )


class Stage19SuggestionIntegrationTests(unittest.TestCase):
    """Integration test suite verifying the 8 Stage 19 criteria."""

    @classmethod
    def setUpClass(cls):
        if DICT_PATH.is_file():
            cls.dict = OfflineDictionary.load(DICT_PATH)
        else:
            cls.dict = OfflineDictionary()
            for w, f in [
                ("HE", 1000), ("HAVE", 900), ("HERE", 800), ("HIM", 700), ("HOW", 600),
                ("HER", 500), ("HEY", 400), ("HELP", 300), ("HELLO", 200), ("HELL", 100),
                ("WITH", 500), ("WHAT", 400), ("WHEN", 300), ("WHERE", 200), ("WHO", 100),
                ("APPLE", 300), ("APPLICATION", 200), ("APPLY", 100),
            ]:
                cls.dict.add_word(w, f)

    def _create_coordinator(
        self,
        ranker=None,
        sentence_generator=None,
    ) -> AlphabetRecognizer:
        sm = SuggestionManager(
            dictionary=self.dict,
            ranker=ranker,
            sentence_generator=sentence_generator,
            debounce_ms=50.0,
        )
        coord = AlphabetRecognizer(
            FakePredictor("A", 0.0),
            FakePredictor("0", 0.5),
            DynamicModel("Z"),
            enable_experimental_jz=True,
            suggestion_manager=sm,
        )
        return coord

    def test_01_commit_char_updates_word_suggestion(self):
        """1. Commit character -> updates word suggestions immediately."""
        coord = self._create_coordinator()
        try:
            self.assertEqual(coord.composer.text, "")
            self.assertEqual(coord.suggestions, [])

            # Commit 'H'
            coord.composer.append("H")
            self.assertEqual(coord.composer.text, "H")
            self.assertEqual(coord.composer.current_prefix, "H")

            # Verify suggestions updated immediately with words starting with 'H'
            words = coord.suggestions
            self.assertTrue(len(words) > 0, "Suggestions should not be empty after committing 'H'")
            for w in words:
                self.assertTrue(w.startswith("H"), f"Suggestion '{w}' does not start with 'H'")
            self.assertIn("HE", words)
        finally:
            coord.close()

    def test_02_commit_next_char_updates_prefix_and_suggestions(self):
        """2. Commit next character -> updates prefix and candidate suggestions."""
        coord = self._create_coordinator()
        try:
            # Commit 'H' -> 'HE' -> 'HEL'
            coord.composer.append("H")
            words_h = coord.suggestions

            coord.composer.append("E")
            self.assertEqual(coord.composer.text, "HE")
            self.assertEqual(coord.composer.current_prefix, "HE")
            words_he = coord.suggestions
            self.assertTrue(len(words_he) > 0)
            for w in words_he:
                self.assertTrue(w.startswith("HE"), f"Suggestion '{w}' does not start with 'HE'")
            self.assertIn("HELP", words_he)

            coord.composer.append("L")
            self.assertEqual(coord.composer.text, "HEL")
            self.assertEqual(coord.composer.current_prefix, "HEL")
            words_hel = coord.suggestions
            self.assertTrue(len(words_hel) > 0)
            for w in words_hel:
                self.assertTrue(w.startswith("HEL"), f"Suggestion '{w}' does not start with 'HEL'")
            self.assertIn("HELLO", words_hel)
        finally:
            coord.close()

    def test_03_space_transitions_to_new_word(self):
        """3. SPACE -> transitions to word boundary (clears word suggestions, ready for next word)."""
        coord = self._create_coordinator()
        try:
            # Type 'HEL'
            coord.composer.append("H")
            coord.composer.append("E")
            coord.composer.append("L")
            self.assertTrue(len(coord.suggestions) > 0)

            # Append SPACE
            coord.composer.append(" ")
            self.assertEqual(coord.composer.text, "HEL ")
            self.assertEqual(coord.composer.current_prefix, "")
            self.assertEqual(coord.composer.committed_context, "HEL")
            # Word suggestions should be cleared on word boundary
            self.assertEqual(coord.suggestions, [])

            # Begin next word with 'W'
            coord.composer.append("W")
            self.assertEqual(coord.composer.text, "HEL W")
            self.assertEqual(coord.composer.current_prefix, "W")
            self.assertEqual(coord.composer.committed_context, "HEL")
            words_w = coord.suggestions
            self.assertTrue(len(words_w) > 0)
            for w in words_w:
                self.assertTrue(w.startswith("W"), f"Suggestion '{w}' does not start with 'W'")
        finally:
            coord.close()

    def test_04_select_suggestion_updates_text_exactly_once(self):
        """4. Select suggestion -> commits chosen word with exactly one trailing space once."""
        coord = self._create_coordinator()
        try:
            coord.composer.append("H")
            coord.composer.append("E")
            coord.composer.append("L")

            valid_suggestions = coord.suggestions
            self.assertIn("HELLO", valid_suggestions)
            idx = valid_suggestions.index("HELLO") + 1

            # Select HELLO (1-based index)
            selected = coord.suggestion_manager.select_suggestion(idx, coord.composer)
            self.assertEqual(selected, "HELLO")
            # Must replace active prefix and append exactly one space
            self.assertEqual(coord.composer.text, "HELLO ")
            self.assertEqual(coord.composer.current_prefix, "")
            self.assertEqual(coord.composer.committed_context, "HELLO")

            # Word suggestions cleared
            self.assertEqual(coord.suggestions, [])
        finally:
            coord.close()

    def test_05_ignore_suggestions_leaves_text_unchanged(self):
        """5. Ignore suggestion -> text is never auto-replaced; typing continues naturally."""
        coord = self._create_coordinator()
        try:
            coord.composer.append("H")
            coord.composer.append("E")
            # Suggestions are available
            self.assertTrue(len(coord.suggestions) > 0)

            # User ignores suggestions and types 'L' then 'P'
            coord.composer.append("L")
            coord.composer.append("P")
            self.assertEqual(coord.composer.text, "HELP")
            self.assertEqual(coord.composer.current_prefix, "HELP")

            # Attempting to select with invalid index leaves text unchanged
            coord.suggestion_manager.select_suggestion(99, coord.composer)
            self.assertEqual(coord.composer.text, "HELP")
        finally:
            coord.close()

    def test_06_ollama_offline_does_not_affect_recognition_or_tier1(self):
        """6. Ollama offline / unreachable -> character recognition and Tier 1 Trie suggestions unaffected."""
        failing_ranker = MockFailingRanker()
        failing_sentence_gen = MockFailingSentenceGenerator()
        coord = self._create_coordinator(ranker=failing_ranker, sentence_generator=failing_sentence_gen)
        try:
            # Commit characters while Ollama is completely failing
            coord.composer.append("H")
            coord.composer.append("E")

            # Wait briefly for background worker to attempt ranking and fail
            time.sleep(0.15)

            # Tier 1 word suggestions must STILL be fully available!
            suggestions = coord.suggestions
            self.assertTrue(len(suggestions) > 0, "Tier 1 dictionary suggestions must work when Ollama is offline")
            for w in suggestions:
                self.assertTrue(w.startswith("HE"))

            # Fallback metric should have incremented
            self.assertTrue(coord.suggestion_manager.metric_fallbacks > 0)

            # Character recognition operates normally with no exceptions
            self.assertEqual(coord.composer.text, "HE")
        finally:
            coord.close()

    def test_07_sentence_suggestion_updates_when_eligible(self):
        """7. Sentence suggestions update when eligible (word boundary with committed context)."""
        working_sentence_gen = MockWorkingSentenceGenerator(["DOING TODAY?", "FEELING GOOD?"])
        coord = self._create_coordinator(sentence_generator=working_sentence_gen)
        try:
            # Type complete word + SPACE
            coord.composer.append("H")
            coord.composer.append("E")
            coord.composer.append("L")
            coord.composer.append("P")
            coord.composer.append(" ")

            self.assertEqual(coord.composer.current_prefix, "")
            self.assertEqual(coord.composer.committed_context, "HELP")

            # Wait for background worker to complete sentence generation
            time.sleep(0.15)

            sentences = coord.sentence_suggestions
            self.assertTrue(len(sentences) > 0, "Sentence suggestions should populate on word boundary")
            self.assertIn("DOING TODAY?", sentences)

            # Verify sentence hitboxes can be computed
            hitboxes = compute_sentence_hitboxes(640, sentences)
            self.assertEqual(len(hitboxes), len(sentences))
        finally:
            coord.close()

    def test_08_no_stale_suggestions_displayed(self):
        """8. Stale suggestions from older generations are strictly discarded."""
        delayed_ranker = MockDelayedRanker(delay_s=0.20)
        coord = self._create_coordinator(ranker=delayed_ranker)
        try:
            # Type 'HE' -> triggers background task for 'HE'
            coord.composer.append("H")
            coord.composer.append("E")

            # Wait for debounce (50ms) so worker picks up 'HE' task and begins ranking
            time.sleep(0.08)
            # Now quickly type 'L' while 'HE' is still ranking in worker -> invalidates generation
            coord.composer.append("L")

            # Wait for delayed ranker to finish the first task
            time.sleep(0.25)

            # Active suggestions MUST start with 'HEL', not old 'HE'
            words = coord.suggestions
            self.assertTrue(len(words) > 0)
            for w in words:
                self.assertTrue(w.startswith("HEL"), f"Stale suggestion '{w}' detected for prefix 'HEL'")
            self.assertTrue(coord.suggestion_manager.metric_stale_dropped > 0)
        finally:
            coord.close()


if __name__ == "__main__":
    unittest.main()
