"""Unit tests for SuggestionManager.

Validates:
- Two-tier suggestions (Tier 1 instant Trie + Tier 2 async Qwen).
- Latest-only scheduling & Stale drop.
- Fallback preservation of dictionary suggestions.
- Concurrency and graceful shutdown.
"""
from __future__ import annotations

import time
import unittest
from typing import Any, Dict, List, Optional

from app.suggestion.dictionary import OfflineDictionary
from app.suggestion.manager import SuggestionManager
from app.suggestion.qwen_ranker import QwenRanker, RankResult


class MockComposer:
    """Mock TextComposer with commit_word interface."""
    def __init__(self, initial_text: str = ""):
        self.text = initial_text

    def append(self, char: str) -> None:
        self.text += char

    def commit_word(self, word: str) -> None:
        last_space = self.text.rfind(" ")
        if last_space == -1:
            self.text = word + " "
        else:
            self.text = self.text[:last_space + 1] + word + " "


class TestSuggestionManager(unittest.TestCase):
    """Test suite for SuggestionManager."""

    def setUp(self):
        # Create a populated dictionary for testing
        self.dictionary = OfflineDictionary()
        words = [
            ("APPLE", 1000),
            ("APPLICATION", 800),
            ("APPOINTMENT", 600),
            ("APPROACH", 500),
            ("APPEAR", 400),
            ("APARTMENT", 300),
            ("ABOUT", 950),
            ("AFTER", 900),
            ("AGAIN", 850),
            ("ALL", 800),
            ("ALONE", 750),
            ("BALL", 1000),
            ("BACK", 900),
            ("BAG", 800),
            ("BANK", 700),
            ("BASE", 600),
            ("HELP", 1000),
            ("HELLO", 900),
            ("HELD", 800),
            ("HELICOPTER", 700),
            ("HELIUM", 600),
        ]
        for w, f in words:
            self.dictionary.add_word(w, f)

    def tearDown(self):
        # Ensure any managers are closed
        pass

    def test_tier1_instant_lookup_synchronous(self):
        manager = SuggestionManager(dictionary=self.dictionary, ranker=None)
        try:
            # Type 'A'
            sugg_A = manager.update_text("A")
            self.assertEqual(len(sugg_A), 5)
            self.assertTrue(all(w.startswith("A") for w in sugg_A))

            # Type 'AP'
            sugg_AP = manager.update_text("AP")
            self.assertEqual(len(sugg_AP), 5)
            self.assertTrue(all(w.startswith("AP") for w in sugg_AP))

            # Type 'APP'
            sugg_APP = manager.update_text("APP")
            self.assertEqual(len(sugg_APP), 5)
            self.assertTrue(all(w.startswith("APP") for w in sugg_APP))
            self.assertEqual(sugg_APP[0], "APPLE")  # highest frequency
        finally:
            manager.close()

    def test_tier2_async_reranking_updates_suggestions(self):
        updated_events = []

        def fake_transport(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
            # Prioritize APPROACH over APPLE
            return {
                "message": {
                    "role": "assistant",
                    "content": '{"ranked_words": ["APPROACH", "APPLE", "APPLICATION"]}'
                }
            }

        ranker = QwenRanker(transport_fn=fake_transport)
        manager = SuggestionManager(
            dictionary=self.dictionary,
            ranker=ranker,
            debounce_ms=50.0,
            on_update_callback=lambda words: updated_events.append(list(words))
        )
        try:
            # 1. Immediately returns Tier 1 (APPLE is #1 by frequency)
            t0 = time.perf_counter()
            instant = manager.update_text("I NEED A NEW APP")
            elapsed = (time.perf_counter() - t0) * 1000
            self.assertLess(elapsed, 10.0)  # Must be < 10ms
            self.assertEqual(instant[0], "APPLE")

            # 2. Wait for Tier 2 worker (debounce 50ms + instant fake transport)
            time.sleep(0.15)
            refined = manager.get_suggestions()
            self.assertEqual(refined[0], "APPROACH")
            self.assertGreaterEqual(len(updated_events), 1)
        finally:
            manager.close()

    def test_stale_drop_fast_typing(self):
        """When user types 'A' then immediately 'AP', the older 'A' task must be dropped."""
        def slow_transport(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
            # Simulate a 100ms slow response
            time.sleep(0.10)
            return {
                "message": {
                    "role": "assistant",
                    "content": '{"ranked_words": ["ABOUT", "AFTER"]}'
                }
            }

        ranker = QwenRanker(transport_fn=slow_transport)
        manager = SuggestionManager(
            dictionary=self.dictionary,
            ranker=ranker,
            debounce_ms=10.0
        )
        try:
            # User types 'A'
            manager.update_text("A")
            # Rapidly types 'P' (total: 'AP')
            time.sleep(0.02)
            manager.update_text("AP")

            # Wait for all background tasks to complete
            time.sleep(0.25)

            # Final suggestions MUST be for 'AP', not overwritten by 'A'
            final_sugg = manager.get_suggestions()
            for w in final_sugg:
                self.assertTrue(w.startswith("AP"), f"Expected word starting with AP, got {w}")
            self.assertGreater(manager.metric_stale_dropped, 0)
        finally:
            manager.close()

    def test_fallback_on_qwen_timeout(self):
        def timeout_transport(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
            raise TimeoutError("Simulated Ollama Timeout")

        ranker = QwenRanker(transport_fn=timeout_transport)
        manager = SuggestionManager(
            dictionary=self.dictionary,
            ranker=ranker,
            debounce_ms=20.0
        )
        try:
            instant = manager.update_text("APP")
            self.assertEqual(instant[0], "APPLE")

            # Wait for timeout
            time.sleep(0.1)

            # Verify Tier 1 suggestions are preserved
            current = manager.get_suggestions()
            self.assertEqual(current[0], "APPLE")
            self.assertEqual(manager.metric_fallbacks, 1)
        finally:
            manager.close()

    def test_fallback_on_ollama_offline(self):
        def offline_transport(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
            raise ConnectionError("Ollama offline")

        ranker = QwenRanker(transport_fn=offline_transport)
        manager = SuggestionManager(
            dictionary=self.dictionary,
            ranker=ranker,
            debounce_ms=20.0
        )
        try:
            instant = manager.update_text("HEL")
            self.assertEqual(instant[0], "HELP")

            time.sleep(0.1)
            current = manager.get_suggestions()
            self.assertEqual(current[0], "HELP")
            self.assertEqual(manager.metric_fallbacks, 1)
        finally:
            manager.close()

    def test_fallback_on_malformed_json(self):
        def bad_json_transport(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
            return {"message": {"content": "INVALID_JSON_PAYLOAD"}}

        ranker = QwenRanker(transport_fn=bad_json_transport)
        manager = SuggestionManager(
            dictionary=self.dictionary,
            ranker=ranker,
            debounce_ms=20.0
        )
        try:
            instant = manager.update_text("BA")
            self.assertEqual(instant[0], "BALL")

            time.sleep(0.1)
            current = manager.get_suggestions()
            self.assertEqual(current[0], "BALL")
            self.assertEqual(manager.metric_fallbacks, 1)
        finally:
            manager.close()

    def test_whitelist_sanitizer_in_manager(self):
        def hallucinating_transport(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
            return {
                "message": {
                    "role": "assistant",
                    "content": '{"ranked_words": ["FAKE_WORD", "APPLE"]}'
                }
            }

        ranker = QwenRanker(transport_fn=hallucinating_transport)
        manager = SuggestionManager(
            dictionary=self.dictionary,
            ranker=ranker,
            debounce_ms=20.0
        )
        try:
            manager.update_text("APP")
            time.sleep(0.1)
            current = manager.get_suggestions()
            self.assertNotIn("FAKE_WORD", current)
            self.assertIn("APPLE", current)
        finally:
            manager.close()

    def test_space_clears_suggestions(self):
        manager = SuggestionManager(dictionary=self.dictionary, ranker=None)
        try:
            manager.update_text("APP")
            self.assertGreater(len(manager.get_suggestions()), 0)

            # Space committed
            manager.update_text("APP ")
            self.assertEqual(manager.get_suggestions(), [])
        finally:
            manager.close()

    def test_mode_switch_suppression(self):
        manager = SuggestionManager(dictionary=self.dictionary, ranker=None)
        try:
            manager.update_text("APP")
            self.assertGreater(len(manager.get_suggestions()), 0)

            # Switch to digits mode '123'
            manager.set_mode("123")
            self.assertEqual(manager.get_suggestions(), [])

            # Typing in 123 returns empty
            sugg = manager.update_text("123")
            self.assertEqual(sugg, [])

            # Switch back to ABC
            manager.set_mode("ABC")
            sugg_abc = manager.update_text("APP")
            self.assertGreater(len(sugg_abc), 0)
        finally:
            manager.close()

    def test_select_suggestion_and_composer_commit(self):
        composer = MockComposer("I WANT APP")
        manager = SuggestionManager(dictionary=self.dictionary, ranker=None)
        try:
            manager.update_text(composer.text)
            self.assertEqual(manager.get_suggestions()[0], "APPLE")

            # User selects suggestion #1
            chosen = manager.select_suggestion(1, composer=composer)
            self.assertEqual(chosen, "APPLE")
            # Composer text updated with chosen word + single space
            self.assertEqual(composer.text, "I WANT APPLE ")
            # Suggestions reset
            self.assertEqual(manager.get_suggestions(), [])
        finally:
            manager.close()

    def test_no_automatic_selection(self):
        """Suggestions should remain active without modifying text until user selects."""
        composer = MockComposer("I WANT APP")
        manager = SuggestionManager(dictionary=self.dictionary, ranker=None)
        try:
            manager.update_text(composer.text)
            time.sleep(0.05)
            # Text must not change on its own!
            self.assertEqual(composer.text, "I WANT APP")
            self.assertGreater(len(manager.get_suggestions()), 0)
        finally:
            manager.close()

    def test_graceful_shutdown(self):
        manager = SuggestionManager(dictionary=self.dictionary, ranker=None)
        self.assertTrue(manager._worker_thread.is_alive())
        manager.shutdown(timeout_s=0.5)
        self.assertFalse(manager._worker_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
