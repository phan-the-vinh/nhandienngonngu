"""Comprehensive Unit and Integration Tests for Sentence Suggestion UI & Click-to-Select.

Phase 3.10.7-B4 verification suite covering all required scenarios:
Group A. Geometry (7 scenarios)
Group B. Mouse Interaction (10 scenarios)
Group C. Text Integrity (6 scenarios)
"""
from __future__ import annotations

import cv2
import numpy as np
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence
from unittest.mock import MagicMock

from app.inference.unified_static_coordinator import TextComposer
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from app.inference.unified_abc123_jz_auto_experiment import (
    compute_sentence_hitboxes,
    compute_suggestion_hitboxes,
    mode_from_click,
)
from app.suggestion.context import SuggestionContext
from app.suggestion.dictionary import OfflineDictionary
from app.suggestion.manager import SuggestionManager
from app.suggestion.sentence_generator import SentenceGenerator, SentenceResult


# ==============================================================================
# GROUP A: GEOMETRY TESTS (7 scenarios)
# ==============================================================================
class TestSentenceHitboxGeometry(unittest.TestCase):
    def setUp(self):
        self.width = 640
        self.sentences = [
            "EAT SOMETHING.",
            "GO TO THE STORE AND BUY SOME APPLES.",
            "LEARN ENGLISH VERY WELL.",
        ]

    # 1. Hitbox khop chinh xac vung ve
    def test_a01_hitbox_bounds_match_drawing_area(self):
        hitboxes = compute_sentence_hitboxes(self.width, self.sentences)
        self.assertEqual(len(hitboxes), 3)

        # Container bounding area is (width + 8, 330) to (width + 302, 416)
        container_x1 = self.width + 8
        container_y1 = 330
        container_x2 = self.width + 302
        container_y2 = 416

        for x1, y1, x2, y2, s in hitboxes:
            self.assertGreaterEqual(x1, container_x1)
            self.assertLessEqual(x2, container_x2)
            self.assertGreater(y1, container_y1)
            self.assertLessEqual(y2, container_y2)
            self.assertGreater(x2, x1)
            self.assertGreater(y2, y1)

    # 2. Ba hitbox khong chong lan doi mot
    def test_a02_three_hitboxes_strictly_non_overlapping(self):
        hitboxes = compute_sentence_hitboxes(self.width, self.sentences)
        self.assertEqual(len(hitboxes), 3)

        for i in range(len(hitboxes)):
            for j in range(i + 1, len(hitboxes)):
                x1_a, y1_a, x2_a, y2_a, _ = hitboxes[i]
                x1_b, y1_b, x2_b, y2_b, _ = hitboxes[j]
                x_overlap = max(0, min(x2_a, x2_b) - max(x1_a, x1_b))
                y_overlap = max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
                self.assertEqual(
                    x_overlap * y_overlap,
                    0,
                    f"Overlap between hitbox {i} ({hitboxes[i]}) and {j} ({hitboxes[j]})",
                )

    # 3. Khong de nut ABC/123 (y: 8 to 47)
    def test_a03_does_not_overlap_abc123_buttons(self):
        for n in [1, 2, 3]:
            hitboxes = compute_sentence_hitboxes(self.width, self.sentences[:n])
            for x1, y1, x2, y2, _ in hitboxes:
                # ABC/123 buttons span y = 8 to 47
                self.assertGreater(y1, 47, f"y1 ({y1}) must be well below 47")

    # 4. Khong de vung camera (x < width)
    def test_a04_does_not_overlap_camera_view(self):
        for n in [1, 2, 3]:
            hitboxes = compute_sentence_hitboxes(self.width, self.sentences[:n])
            for x1, y1, x2, y2, _ in hitboxes:
                # Camera view spans x = 0 to width
                self.assertGreaterEqual(x1, self.width, f"x1 ({x1}) must not encroach on camera {self.width}")
                self.assertGreaterEqual(x1, self.width + 14)

    # 5. Khong de vung TEXT (y >= 423)
    def test_a05_does_not_overlap_text_line(self):
        for n in [1, 2, 3]:
            hitboxes = compute_sentence_hitboxes(self.width, self.sentences[:n])
            for x1, y1, x2, y2, _ in hitboxes:
                # TEXT line starts at y = 423 (height - 57)
                self.assertLess(y2, 423, f"y2 ({y2}) must be strictly above 423")

    # 6. Ho tro cau nhieu dong khong tran pill
    def test_a06_multiline_wrapping_support(self):
        long_sentence = "PLEASE GO TO THE SUPERMARKET AND BUY SOME FRESH MILK AND BREAD FOR BREAKFAST."
        hitboxes = compute_sentence_hitboxes(self.width, [long_sentence])
        self.assertEqual(len(hitboxes), 1)
        x1, y1, x2, y2, text = hitboxes[0]

        pill_w = x2 - x1
        pill_h = y2 - y1
        self.assertGreaterEqual(pill_w, 260)
        self.assertGreaterEqual(pill_h, 20)

        # Verify OpenCV rendering of wrapped text on dummy image does not crash
        dummy = np.full((480, 950, 3), 245, dtype=np.uint8)
        font_scale = 0.27
        words = text.split()
        lines = []
        curr = []
        for w in words:
            cand = " ".join(curr + [w])
            (tw, _), _ = cv2.getTextSize(cand, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            if tw <= pill_w - 14:
                curr.append(w)
            else:
                if curr:
                    lines.append(" ".join(curr))
                    curr = [w]
                else:
                    lines.append(w)
                    curr = []
        if curr:
            lines.append(" ".join(curr))

        self.assertGreaterEqual(len(lines), 2, "Long sentence should wrap into at least 2 lines")
        for line in lines:
            (lw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            self.assertLessEqual(lw, pill_w - 10, f"Line '{line}' exceeds pill width")

    # 7. Khong vuot panel ben phai (x2 <= width + 310)
    def test_a07_does_not_exceed_panel_width(self):
        panel_max_x = self.width + 310
        for n in [1, 2, 3]:
            hitboxes = compute_sentence_hitboxes(self.width, self.sentences[:n])
            for x1, y1, x2, y2, _ in hitboxes:
                self.assertLessEqual(x2, panel_max_x, f"x2 ({x2}) exceeds panel width ({panel_max_x})")
                self.assertLessEqual(x2, self.width + 296)


# ==============================================================================
# GROUP B: MOUSE INTERACTION TESTS (10 scenarios)
# ==============================================================================
class TestSentenceMouseInteraction(unittest.TestCase):
    def setUp(self):
        self.width = 640
        self.composer = TextComposer()
        self.composer.text = "I WANT TO "
        self.sentences = ["EAT SOMETHING.", "GO HOME.", "LEARN ENGLISH."]
        self.hitboxes = compute_sentence_hitboxes(self.width, self.sentences)

        self.mock_manager = MagicMock()
        self.mock_manager.current_sentence_context = "I WANT TO"
        self.coordinator = SimpleNamespace(
            mode="ABC",
            composer=self.composer,
            suggestions=[],
            sentence_suggestions=list(self.sentences),
            suggestion_hitboxes=[],
            sentence_hitboxes=list(self.hitboxes),
            suggestion_manager=self.mock_manager,
            hold=SimpleNamespace(text=self.composer.text),
        )

    def _simulate_click(self, event, x, y):
        """Simulate mouse callback logic on coordinator."""
        if event == cv2.EVENT_LBUTTONDOWN:
            sentence_hitboxes = getattr(self.coordinator, "sentence_hitboxes", [])
            if sentence_hitboxes:
                for x1, y1, x2, y2, sentence in list(sentence_hitboxes):
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        current_valid = getattr(self.coordinator, "sentence_suggestions", [])
                        if self.coordinator.suggestion_manager is not None and hasattr(self.coordinator.suggestion_manager, "current_sentence_context"):
                            expected_ctx = self.coordinator.suggestion_manager.current_sentence_context
                        else:
                            expected_ctx = self.coordinator.composer.committed_context
                        if sentence in current_valid:
                            committed = self.coordinator.composer.commit_sentence(
                                sentence,
                                valid_suggestions=current_valid,
                                expected_context=expected_ctx,
                            )
                            if committed:
                                if self.coordinator.suggestion_manager is not None:
                                    self.coordinator.suggestion_manager.update_text(self.coordinator.composer.text)
                                self.coordinator.hold.text = self.coordinator.composer.text
                                self.coordinator.sentence_hitboxes = []
                        break

    # 1. Click cau thu nhat
    def test_b01_click_first_sentence(self):
        x1, y1, x2, y2, _ = self.hitboxes[0]
        click_x = (x1 + x2) // 2
        click_y = (y1 + y2) // 2
        self._simulate_click(cv2.EVENT_LBUTTONDOWN, click_x, click_y)
        self.assertEqual(self.composer.text, "I WANT TO EAT SOMETHING.")
        self.assertEqual(self.coordinator.sentence_hitboxes, [])

    # 2. Click cau thu hai
    def test_b02_click_second_sentence(self):
        x1, y1, x2, y2, _ = self.hitboxes[1]
        click_x = (x1 + x2) // 2
        click_y = (y1 + y2) // 2
        self._simulate_click(cv2.EVENT_LBUTTONDOWN, click_x, click_y)
        self.assertEqual(self.composer.text, "I WANT TO GO HOME.")
        self.assertEqual(self.coordinator.sentence_hitboxes, [])

    # 3. Click cau thu ba
    def test_b03_click_third_sentence(self):
        x1, y1, x2, y2, _ = self.hitboxes[2]
        click_x = (x1 + x2) // 2
        click_y = (y1 + y2) // 2
        self._simulate_click(cv2.EVENT_LBUTTONDOWN, click_x, click_y)
        self.assertEqual(self.composer.text, "I WANT TO LEARN ENGLISH.")
        self.assertEqual(self.coordinator.sentence_hitboxes, [])

    # 4. Click ngoai hitbox
    def test_b04_click_outside_hitboxes(self):
        # Click in camera region
        self._simulate_click(cv2.EVENT_LBUTTONDOWN, 200, 200)
        self.assertEqual(self.composer.text, "I WANT TO ")
        self.assertEqual(len(self.coordinator.sentence_hitboxes), 3)

        # Click in gap between pill 0 and pill 1
        gap_y = (self.hitboxes[0][3] + self.hitboxes[1][1]) // 2
        click_x = self.hitboxes[0][0] + 50
        self._simulate_click(cv2.EVENT_LBUTTONDOWN, click_x, gap_y)
        self.assertEqual(self.composer.text, "I WANT TO ")
        self.assertEqual(len(self.coordinator.sentence_hitboxes), 3)

    # 5. Click chuot phai (RBUTTONDOWN) khong commit
    def test_b05_right_click_ignored(self):
        x1, y1, x2, y2, _ = self.hitboxes[0]
        click_x = (x1 + x2) // 2
        click_y = (y1 + y2) // 2
        self._simulate_click(cv2.EVENT_RBUTTONDOWN, click_x, click_y)
        self.assertEqual(self.composer.text, "I WANT TO ")
        self.assertEqual(len(self.coordinator.sentence_hitboxes), 3)

    # 6. Double click chi commit dung mot lan
    def test_b06_double_click_commits_only_once(self):
        x1, y1, x2, y2, _ = self.hitboxes[0]
        click_x = (x1 + x2) // 2
        click_y = (y1 + y2) // 2

        # First click commits
        self._simulate_click(cv2.EVENT_LBUTTONDOWN, click_x, click_y)
        self.assertEqual(self.composer.text, "I WANT TO EAT SOMETHING.")
        self.assertEqual(self.coordinator.sentence_hitboxes, [])

        # Immediate second click (before re-render) -> hitboxes is empty, rejected
        self._simulate_click(cv2.EVENT_LBUTTONDOWN, click_x, click_y)
        self.assertEqual(self.composer.text, "I WANT TO EAT SOMETHING.")

    # 7. Click stale hitbox bi tu choi
    def test_b07_click_stale_hitbox_rejected(self):
        # User typed new letter before click arrived
        self.composer.text = "I WANT TO E"
        x1, y1, x2, y2, _ = self.hitboxes[0]
        self._simulate_click(cv2.EVENT_LBUTTONDOWN, (x1 + x2) // 2, (y1 + y2) // 2)
        # Commit should be rejected because current_prefix is non-empty
        self.assertEqual(self.composer.text, "I WANT TO E")

    # 8. Click khi context da doi bi tu choi
    def test_b08_click_when_context_changed_rejected(self):
        # Context changed to completely different phrase
        self.composer.text = "SHE DECIDED TO "
        x1, y1, x2, y2, _ = self.hitboxes[0]
        self._simulate_click(cv2.EVENT_LBUTTONDOWN, (x1 + x2) // 2, (y1 + y2) // 2)
        self.assertEqual(self.composer.text, "SHE DECIDED TO ")

    # 9. Click khi generation da doi bi tu choi
    def test_b09_click_when_generation_advanced(self):
        # Simulate generation invalidation clearing sentence suggestions
        self.coordinator.sentence_suggestions = []
        x1, y1, x2, y2, _ = self.hitboxes[0]
        self._simulate_click(cv2.EVENT_LBUTTONDOWN, (x1 + x2) // 2, (y1 + y2) // 2)
        self.assertEqual(self.composer.text, "I WANT TO ")

    # 10. Click cau khong thuoc whitelist bi tu choi
    def test_b10_click_unwhitelisted_sentence_rejected(self):
        # Manipulate hitbox to point to unapproved sentence
        stale_hitboxes = [(self.hitboxes[0][0], self.hitboxes[0][1], self.hitboxes[0][2], self.hitboxes[0][3], "HACKED SENTENCE.")]
        self.coordinator.sentence_hitboxes = stale_hitboxes
        x1, y1, x2, y2, _ = stale_hitboxes[0]
        self._simulate_click(cv2.EVENT_LBUTTONDOWN, (x1 + x2) // 2, (y1 + y2) // 2)
        self.assertEqual(self.composer.text, "I WANT TO ")


# ==============================================================================
# GROUP C: TEXT INTEGRITY TESTS (6 scenarios)
# ==============================================================================
class TestSentenceTextIntegrity(unittest.TestCase):
    def setUp(self):
        self.composer = TextComposer()

    # 1. Continuation commit chinh xac
    def test_c01_continuation_commit_exact(self):
        self.composer.text = "I WANT TO "
        ok = self.composer.commit_sentence("EAT SOMETHING.", valid_suggestions=["EAT SOMETHING."])
        self.assertTrue(ok)
        self.assertEqual(self.composer.text, "I WANT TO EAT SOMETHING.")

    # 2. Full sentence khong lap context
    def test_c02_full_sentence_no_context_repetition(self):
        self.composer.text = "I WANT TO "
        # Model produced continuation repeating prompt context
        ok = self.composer.commit_sentence("I WANT TO EAT SOMETHING.", valid_suggestions=["I WANT TO EAT SOMETHING."])
        self.assertTrue(ok)
        self.assertEqual(self.composer.text, "I WANT TO EAT SOMETHING.")
        self.assertNotIn("I WANT TO I WANT TO", self.composer.text)

    # 3. Khong co double spaces
    def test_c03_no_double_spaces(self):
        self.composer.text = "I WANT TO   "
        ok = self.composer.commit_sentence("GO HOME.", valid_suggestions=["GO HOME."])
        self.assertTrue(ok)
        self.assertEqual(self.composer.text, "I WANT TO GO HOME.")
        self.assertNotIn("  ", self.composer.text)

    # 4. Khong mat punctuation
    def test_c04_preserve_punctuation(self):
        self.composer.text = "HELLO, HOW ARE YOU? "
        ok = self.composer.commit_sentence("I AM DOING GREAT!", valid_suggestions=["I AM DOING GREAT!"])
        self.assertTrue(ok)
        self.assertEqual(self.composer.text, "HELLO, HOW ARE YOU? I AM DOING GREAT!")
        self.assertIn("?", self.composer.text)
        self.assertIn(",", self.composer.text)
        self.assertIn("!", self.composer.text)

    # 5. Context cu duoc bao toan 100%
    def test_c05_prior_context_fully_preserved(self):
        prior = "WE HAVE STUDIED ENGLISH FOR THREE YEARS "
        self.composer.text = prior
        ok = self.composer.commit_sentence("AND WE ENJOY IT.", valid_suggestions=["AND WE ENJOY IT."])
        self.assertTrue(ok)
        self.assertEqual(self.composer.text, "WE HAVE STUDIED ENGLISH FOR THREE YEARS AND WE ENJOY IT.")
        self.assertTrue(self.composer.text.startswith(prior.rstrip()))

    # 6. Tu choi commit lap lai tren cung mot trang thai
    def test_c06_reject_repeated_commit(self):
        self.composer.text = "I WANT TO "
        ok1 = self.composer.commit_sentence(
            "EAT SOMETHING.",
            valid_suggestions=["EAT SOMETHING."],
            expected_context="I WANT TO"
        )
        self.assertTrue(ok1)
        self.assertEqual(self.composer.text, "I WANT TO EAT SOMETHING.")

        # Second commit with old expected context should fail
        ok2 = self.composer.commit_sentence(
            "EAT SOMETHING.",
            valid_suggestions=["EAT SOMETHING."],
            expected_context="I WANT TO"
        )
        self.assertFalse(ok2)
        self.assertEqual(self.composer.text, "I WANT TO EAT SOMETHING.")


if __name__ == "__main__":
    unittest.main()
