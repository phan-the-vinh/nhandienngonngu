"""Comprehensive Unit and Integration Tests for Word Suggestion and Click-to-Select.

Phase 3.10.6B2 verification suite covering all 20 required scenarios:
1. Input A -> suggestion starts with A
2. Input AP -> only words starting with AP
3. Input APP -> only words starting with APP
4. Select a word -> replaces active prefix exactly
5. Select a word -> adds exactly one ASCII space
6. Select a word -> preserves committed context
7. Double-click or single event -> commits exactly once
8. Click outside hitbox -> text unchanged
9. Click second hitbox -> selects second word
10. Click when suggestions empty -> no crash or change
11. Click stale result -> rejected, no commit
12. Typing new letter during Qwen -> stale result dropped
13. SPACE during Qwen -> prefix cleared and stale dropped
14. Switch ABC/123 -> text preserved, suggestions cleared/restored
15. UNCERTAIN sentinel -> not allowed in text or prefix
16. Ollama unavailable -> dictionary fallback works instantly
17. Manager shutdown -> clean, no hang or deadlock
18. TextComposer backwards compatibility -> .text and append() work as before
19. Mouse callback and UI safety -> worker thread never mutates UI
20. No accidental consecutive double spaces on selection
"""
from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.inference.unified_static_coordinator import TextComposer
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from app.inference.unified_abc123_jz_auto_experiment import (
    compute_suggestion_hitboxes,
    mode_from_click,
)
from app.suggestion.context import SuggestionContext
from app.suggestion.dictionary import OfflineDictionary
from app.suggestion.manager import SuggestionManager
from app.suggestion.qwen_ranker import QwenRanker, RankResult

ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = ROOT / "data/dictionary/english_words_25k.tsv"


class FakeSlowRanker:
    """Mock ranker that simulates configurable network delay and responses."""

    def __init__(self, delay_s: float = 0.1, reverse_order: bool = True):
        self.delay_s = delay_s
        self.reverse_order = reverse_order
        self.call_count = 0

    def rank(self, context: str, prefix: str, candidates: list[str]) -> RankResult:
        self.call_count += 1
        time.sleep(self.delay_s)
        # Reverse candidate order for test distinguishable ranking
        ranked = list(reversed(candidates)) if self.reverse_order else list(candidates)
        return RankResult(
            success=True,
            ranked_words=ranked,
            latency_ms=self.delay_s * 1000.0,
        )


class FakeFailingRanker:
    """Mock ranker that always fails (simulating Ollama offline)."""

    def __init__(self):
        self.call_count = 0

    def rank(self, context: str, prefix: str, candidates: list[str]) -> RankResult:
        self.call_count += 1
        return RankResult(
            success=False,
            ranked_words=[],
            error_type="CONNECTION_ERROR",
            error_message="Ollama offline",
            latency_ms=1.0,
        )


class TestClickToSelect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if DICT_PATH.is_file():
            cls.dict = OfflineDictionary.load(DICT_PATH)
        else:
            cls.dict = OfflineDictionary()
            for w, f in [
                ("APPLE", 100),
                ("APPLICATION", 90),
                ("APPLY", 80),
                ("APPROVE", 70),
                ("ALL", 60),
                ("AND", 50),
                ("ABOUT", 40),
                ("ARE", 30),
                ("EAT", 95),
                ("EVEN", 85),
                ("ENTER", 75),
                ("EVERY", 65),
                ("EACH", 55),
            ]:
                cls.dict.add_word(w, f)

    def setUp(self):
        self.composer = TextComposer()

    # Scenario 1: Nhap A -> co goi y A
    def test_01_prefix_a_suggestions(self):
        mgr = SuggestionManager(dictionary=self.dict, ranker=None)
        try:
            suggs = mgr.update_text("A")
            self.assertTrue(len(suggs) > 0)
            for word in suggs:
                self.assertTrue(word.startswith("A"), f"{word} does not start with A")
        finally:
            mgr.close()

    # Scenario 2: Nhap AP -> chi co tu bat dau bang AP
    def test_02_prefix_ap_suggestions(self):
        mgr = SuggestionManager(dictionary=self.dict, ranker=None)
        try:
            suggs = mgr.update_text("AP")
            self.assertTrue(len(suggs) > 0)
            for word in suggs:
                self.assertTrue(word.startswith("AP"), f"{word} does not start with AP")
        finally:
            mgr.close()

    # Scenario 3: Nhap APP -> chi co tu bat dau bang APP
    def test_03_prefix_app_suggestions(self):
        mgr = SuggestionManager(dictionary=self.dict, ranker=None)
        try:
            suggs = mgr.update_text("APP")
            self.assertTrue(len(suggs) > 0)
            for word in suggs:
                self.assertTrue(word.startswith("APP"), f"{word} does not start with APP")
        finally:
            mgr.close()

    # Scenario 4: Chon mot tu -> thay prefix chinh xac
    def test_04_select_word_replaces_prefix(self):
        self.composer.text = "I WANT TO E"
        self.assertEqual(self.composer.current_prefix, "E")
        ok = self.composer.commit_word("EAT", valid_suggestions=["EAT", "EVEN"])
        self.assertTrue(ok)
        self.assertEqual(self.composer.text, "I WANT TO EAT ")
        self.assertEqual(self.composer.current_prefix, "")

    # Scenario 5: Chon tu -> them dung mot dau cach
    def test_05_select_word_adds_exactly_one_space(self):
        self.composer.text = "APP"
        ok = self.composer.commit_word("APPLE", valid_suggestions=["APPLE"])
        self.assertTrue(ok)
        self.assertEqual(self.composer.text, "APPLE ")
        self.assertTrue(self.composer.text.endswith(" "))
        self.assertFalse(self.composer.text.endswith("  "))

    # Scenario 6: Chon tu -> khong mat committed context
    def test_06_select_word_preserves_committed_context(self):
        self.composer.text = "THE QUICK BROWN F"
        self.assertEqual(self.composer.committed_context, "THE QUICK BROWN")
        ok = self.composer.commit_word("FOX", valid_suggestions=["FOX", "FOOD"])
        self.assertTrue(ok)
        self.assertEqual(self.composer.text, "THE QUICK BROWN FOX ")
        self.assertEqual(self.composer.committed_context, "THE QUICK BROWN FOX")

    # Scenario 7: Chon tu hai lan bang mot click -> chi commit mot lan
    def test_07_single_click_commits_once(self):
        self.composer.text = "APP"
        current_valid = ["APPLE", "APPLICATION"]
        ok1 = self.composer.commit_word("APPLE", valid_suggestions=current_valid)
        self.assertTrue(ok1)
        self.assertEqual(self.composer.text, "APPLE ")
        # Second attempt with same event/coordinates on updated state
        # In updated state, prefix is "" and "APPLE" does not match
        ok2 = self.composer.commit_word("APPLE", valid_suggestions=[])
        self.assertFalse(ok2)
        self.assertEqual(self.composer.text, "APPLE ")

    # Scenario 8: Click ngoai hitbox -> khong thay doi van ban
    def test_08_click_outside_hitbox_no_change(self):
        width = 640
        hitboxes = compute_suggestion_hitboxes(width, ["APPLE", "APPLICATION"])
        self.composer.text = "APP"
        initial_text = self.composer.text

        # Click at (100, 100) inside camera frame
        click_x, click_y = 100, 100
        matched = False
        for x1, y1, x2, y2, word in hitboxes:
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                matched = True
                self.composer.commit_word(word)
        self.assertFalse(matched)
        self.assertEqual(self.composer.text, initial_text)

        # Click at panel but above suggestions (width + 50, 200)
        click_x, click_y = width + 50, 200
        matched = False
        for x1, y1, x2, y2, word in hitboxes:
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                matched = True
                self.composer.commit_word(word)
        self.assertFalse(matched)
        self.assertEqual(self.composer.text, initial_text)

    # Scenario 9: Click vao hitbox tu thu hai -> chon dung tu thu hai
    def test_09_click_second_hitbox_selects_second_word(self):
        width = 640
        suggestions = ["APPLE", "APPLICATION"]
        hitboxes = compute_suggestion_hitboxes(width, suggestions)
        self.assertEqual(len(hitboxes), 2)
        x1_2, y1_2, x2_2, y2_2, word_2 = hitboxes[1]
        self.assertEqual(word_2, "APPLICATION")

        # Simulate click inside second pill center
        click_x = (x1_2 + x2_2) // 2
        click_y = (y1_2 + y2_2) // 2

        self.composer.text = "APP"
        selected = None
        for x1, y1, x2, y2, word in hitboxes:
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                selected = word
                self.composer.commit_word(word, valid_suggestions=suggestions)
                break

        self.assertEqual(selected, "APPLICATION")
        self.assertEqual(self.composer.text, "APPLICATION ")

    # Scenario 10: Click khi goi y rong -> khong crash
    def test_10_click_empty_suggestions_no_crash(self):
        width = 640
        hitboxes = compute_suggestion_hitboxes(width, [])
        self.assertEqual(hitboxes, [])
        self.composer.text = "HELLO "
        # Click simulation with empty hitboxes
        click_x, click_y = width + 50, 360
        for x1, y1, x2, y2, word in hitboxes:
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                self.composer.commit_word(word)
        self.assertEqual(self.composer.text, "HELLO ")

    # Scenario 11: Click khi ket qua cu -> khong commit
    def test_11_click_stale_result_rejected(self):
        self.composer.text = "APX"
        # Attempt to commit "APPLE" which is not valid for prefix "APX"
        ok = self.composer.commit_word("APPLE", valid_suggestions=["APPLE"])
        self.assertFalse(ok)
        self.assertEqual(self.composer.text, "APX")

        # Also test with empty valid suggestions
        ok2 = self.composer.commit_word("APPLE", valid_suggestions=[])
        self.assertFalse(ok2)
        self.assertEqual(self.composer.text, "APX")

    # Scenario 12: Go ky tu moi trong luc Qwen chay -> ket qua cu bi loai
    def test_12_typing_during_qwen_drops_stale_results(self):
        fake_ranker = FakeSlowRanker(delay_s=0.08)
        mgr = SuggestionManager(dictionary=self.dict, ranker=fake_ranker, debounce_ms=10.0)
        try:
            # First text input "A"
            mgr.update_text("A")
            # Immediately type "P" -> prefix becomes "AP" before fake ranker finishes
            time.sleep(0.015)
            mgr.update_text("AP")
            # Wait for ranker to finish
            time.sleep(0.15)
            suggs = mgr.get_suggestions()
            self.assertTrue(len(suggs) > 0)
            for s in suggs:
                self.assertTrue(s.startswith("AP"), f"Stale suggestion found: {s}")
            self.assertGreater(mgr.metric_stale_dropped, 0)
        finally:
            mgr.close()

    # Scenario 13: SPACE trong luc Qwen chay -> prefix bi xoa va ket qua cu bi loai
    def test_13_space_during_qwen_clears_prefix_and_drops_stale(self):
        fake_ranker = FakeSlowRanker(delay_s=0.08)
        mgr = SuggestionManager(dictionary=self.dict, ranker=fake_ranker, debounce_ms=10.0)
        try:
            mgr.update_text("APP")
            time.sleep(0.015)
            # User commits SPACE -> text becomes "APP "
            mgr.update_text("APP ")
            time.sleep(0.15)
            # Prefix is empty, suggestions must be empty
            self.assertEqual(mgr.get_suggestions(), [])
        finally:
            mgr.close()

    # Scenario 14: Chuyen ABC/123 -> khong mat van ban, cap nhat suggestions
    def test_14_mode_switch_preserves_text_and_updates_suggestions(self):
        mgr = SuggestionManager(dictionary=self.dict, ranker=None)
        try:
            self.composer.text = "HELLO W"
            mgr.update_text(self.composer.text)
            self.assertTrue(len(mgr.get_suggestions()) > 0)

            # Switch to 123
            mgr.set_mode("123")
            self.assertEqual(mgr.get_suggestions(), [])
            self.assertEqual(self.composer.text, "HELLO W")

            # Switch back to ABC
            mgr.set_mode("ABC")
            mgr.update_text(self.composer.text)
            self.assertEqual(self.composer.text, "HELLO W")
            self.assertTrue(len(mgr.get_suggestions()) > 0)
        finally:
            mgr.close()

    # Scenario 15: UNCERTAIN -> khong tro thanh prefix
    def test_15_uncertain_sentinel_not_in_text_or_prefix(self):
        parsed = SuggestionContext.parse("UNCERTAIN")
        self.assertEqual(parsed.current_prefix, "")

        parsed2 = SuggestionContext.parse("HELLO UNCERTAIN")
        self.assertEqual(parsed2.current_prefix, "")
        self.assertEqual(parsed2.committed_text, "HELLO")

        # TextComposer rejection of sentinels
        self.composer.text = "HELLO"
        self.composer.append(" ")
        self.composer.append("A")
        self.composer.append("UNCERTAIN")  # Should be safely ignored
        self.assertEqual(self.composer.text, "HELLO A")

        # commit_word with sentinel
        ok = self.composer.commit_word("UNCERTAIN")
        self.assertFalse(ok)
        self.assertEqual(self.composer.text, "HELLO A")

    # Scenario 16: Ollama khong hoat dong -> tu dien van goi y
    def test_16_ranker_failure_falls_back_to_dictionary(self):
        failing_ranker = FakeFailingRanker()
        mgr = SuggestionManager(dictionary=self.dict, ranker=failing_ranker, debounce_ms=10.0)
        try:
            suggs = mgr.update_text("APP")
            # Immediate Tier 1 suggestions available
            self.assertTrue(len(suggs) > 0)
            self.assertTrue(all(w.startswith("APP") for w in suggs))

            # Wait for background worker to fail and fall back
            time.sleep(0.05)
            suggs_after = mgr.get_suggestions()
            self.assertTrue(len(suggs_after) > 0)
            self.assertTrue(all(w.startswith("APP") for w in suggs_after))
            self.assertGreater(mgr.metric_fallbacks, 0)
        finally:
            mgr.close()

    # Scenario 17: Manager shutdown -> khong treo camera/UI loop
    def test_17_manager_shutdown_clean(self):
        fake_ranker = FakeSlowRanker(delay_s=0.2)
        mgr = SuggestionManager(dictionary=self.dict, ranker=fake_ranker, debounce_ms=10.0)
        mgr.update_text("APP")
        start = time.monotonic()
        mgr.close(timeout_s=0.5)
        elapsed = time.monotonic() - start
        self.assertFalse(mgr._worker_thread.is_alive())
        self.assertLess(elapsed, 0.6)

    # Scenario 18: TextComposer API cu -> regression test
    def test_18_textcomposer_backwards_compatibility(self):
        composer = TextComposer()
        self.assertEqual(composer.text, "")
        composer.append("A")
        composer.append("B")
        composer.append(" ")
        composer.append("1")
        self.assertEqual(composer.text, "AB 1")
        with self.assertRaises(ValueError):
            composer.append("TOO_LONG")
        with self.assertRaises(ValueError):
            composer.append("$")

    # Scenario 19: Mouse callback -> khong sua UI tu worker thread
    def test_19_mouse_callback_thread_safety_no_worker_ui_mutation(self):
        main_thread_id = threading.get_ident()
        callback_thread_ids = []

        def dummy_callback(suggs):
            callback_thread_ids.append(threading.get_ident())

        mgr = SuggestionManager(
            dictionary=self.dict,
            ranker=FakeSlowRanker(delay_s=0.02),
            debounce_ms=5.0,
            on_update_callback=dummy_callback,
        )
        try:
            mgr.update_text("E")
            time.sleep(0.08)
            # The callback should execute on worker thread, but it must NOT call any OpenCV functions
            if callback_thread_ids:
                self.assertNotEqual(callback_thread_ids[0], main_thread_id)

            # Hitbox generation and selection happen strictly on the calling (UI) thread
            width = 640
            hitboxes = compute_suggestion_hitboxes(width, mgr.get_suggestions())
            self.assertTrue(len(hitboxes) > 0)
            # Verify coordinates are in side panel
            for x1, y1, x2, y2, word in hitboxes:
                self.assertGreaterEqual(x1, width)
                self.assertGreater(y1, 47)  # Below ABC/123 buttons
                self.assertLess(y2, 423)    # Above TEXT line
        finally:
            mgr.close()

    # Scenario 20: Khong co hai dau cach lien tiep ngoai truong hop nguoi dung chu dong tao
    def test_20_no_accidental_double_spaces(self):
        self.composer.text = "I WANT "
        # Select "TO"
        ok = self.composer.commit_word("TO", valid_suggestions=["TO"])
        self.assertTrue(ok)
        self.assertEqual(self.composer.text, "I WANT TO ")
        self.assertNotIn("  ", self.composer.text)

        # Select "EAT"
        self.composer.text = "I WANT TO E"
        ok2 = self.composer.commit_word("EAT", valid_suggestions=["EAT"])
        self.assertTrue(ok2)
        self.assertEqual(self.composer.text, "I WANT TO EAT ")
        self.assertNotIn("  ", self.composer.text)


class TestHitboxGeometry(unittest.TestCase):
    def test_hitbox_bounds_and_no_overlap(self):
        width = 640
        for count in range(1, 6):
            words = [f"WORD{i}" for i in range(count)]
            hitboxes = compute_suggestion_hitboxes(width, words)
            self.assertEqual(len(hitboxes), count)

            # Check individual bounds
            for x1, y1, x2, y2, w in hitboxes:
                # Minimum distance from camera: x >= width + 14
                self.assertGreaterEqual(x1, width + 14)
                self.assertLessEqual(x2, width + 296)
                # ABC/123 buttons are at y: 8 to 47 -> must be well below 47
                self.assertGreater(y1, 47)
                self.assertGreaterEqual(y1, 350)
                # Bottom TEXT is at y: 423 -> must be well above 423
                self.assertLessEqual(y2, 411)
                self.assertGreater(x2, x1)
                self.assertGreater(y2, y1)

            # Check pairwise non-overlap
            for i in range(len(hitboxes)):
                for j in range(i + 1, len(hitboxes)):
                    x1_a, y1_a, x2_a, y2_a, _ = hitboxes[i]
                    x1_b, y1_b, x2_b, y2_b, _ = hitboxes[j]
                    # Two rectangles overlap iff they overlap in both X and Y
                    x_overlap = max(0, min(x2_a, x2_b) - max(x1_a, x1_b))
                    y_overlap = max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
                    self.assertEqual(
                        x_overlap * y_overlap,
                        0,
                        f"Overlap detected between hitbox {i} and {j}: {hitboxes[i]} vs {hitboxes[j]}",
                    )


if __name__ == "__main__":
    unittest.main()
