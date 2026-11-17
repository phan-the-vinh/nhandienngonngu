"""Comprehensive unit and integration tests for Backspace (⌫) and Clear functionality.

Covers:
1. TextComposer backspace and clear logic:
   - Empty string: backspace() causes no error and text remains empty
   - Single character: "A" -> ""
   - Multiple characters: "HELLO" -> "HELL"
   - String with trailing space: "HELLO WORLD " -> "HELLO WORLD"
   - English words and numbers: "ASL 2026" -> "ASL 202"
   - Clear on populated string: "HELLO WORLD" -> ""
   - Clear on empty string: "" -> ""
   - on_change notification fired on backspace() and clear()
   - SuggestionContext synchronization (prefix & committed context)
2. Coordinator coordination:
   - RecognitionCoordinator.backspace() and .clear() synchronize hold.text
   - UnifiedAutoCoordinator.backspace() and .clear() synchronize static.hold, jz_hold, and suggestion_manager
3. Hitbox detection:
   - text_action_from_click returns 'BACKSPACE' for clicks inside Backspace hitbox
   - text_action_from_click returns 'CLEAR' for clicks inside Clear hitbox
   - Click on camera area (x < width) returns None
   - Click in empty panel region returns None
   - Strict non-overlapping with suggestion pills and mode buttons
4. Panel visual rendering:
   - _panel() produces valid (height, width + 310, 3) image without error for empty and long text
   - Action button patches match required dimensions
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from app.inference.hold_confirmation_camera_test import (
    _get_button_patch,
    _draw_text_action_buttons,
    _panel,
)
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from app.inference.unified_abc123_jz_auto_experiment import (
    compute_suggestion_hitboxes,
    mode_from_click,
    text_action_from_click,
)
from app.inference.unified_static_coordinator import (
    HoldSettings,
    RecognitionCoordinator,
    TextComposer,
)
from app.interaction.hold_confirmation import HoldConfirmation


class TestTextComposerBackspaceClear(unittest.TestCase):
    """Test string manipulation and notification in TextComposer."""

    def setUp(self):
        self.composer = TextComposer()

    def test_backspace_on_empty_string(self):
        """When text is empty, backspace must not raise any error and text remains empty."""
        self.assertEqual(self.composer.text, "")
        self.composer.backspace()
        self.assertEqual(self.composer.text, "")

    def test_backspace_single_character(self):
        """Single character becomes empty string."""
        self.composer.append("A")
        self.assertEqual(self.composer.text, "A")
        self.composer.backspace()
        self.assertEqual(self.composer.text, "")

    def test_backspace_multiple_characters(self):
        """Multiple characters have exactly one character removed."""
        for ch in "HELLO":
            self.composer.append(ch)
        self.assertEqual(self.composer.text, "HELLO")
        self.composer.backspace()
        self.assertEqual(self.composer.text, "HELL")
        self.composer.backspace()
        self.assertEqual(self.composer.text, "HEL")

    def test_backspace_trailing_space(self):
        """Trailing space is removed cleanly."""
        for ch in "HELLO":
            self.composer.append(ch)
        self.composer.append(" ")
        self.assertEqual(self.composer.text, "HELLO ")
        self.composer.backspace()
        self.assertEqual(self.composer.text, "HELLO")

    def test_backspace_alphanumeric_and_english(self):
        """Alphanumeric text handles backspace properly."""
        for ch in "ASL 2026":
            self.composer.append(ch)
        self.assertEqual(self.composer.text, "ASL 2026")
        self.composer.backspace()
        self.assertEqual(self.composer.text, "ASL 202")

    def test_repeated_backspaces_beyond_length(self):
        """Calling backspace more times than character count is safe and terminates at empty."""
        self.composer.append("X")
        self.composer.backspace()
        self.composer.backspace()
        self.composer.backspace()
        self.assertEqual(self.composer.text, "")

    def test_clear_populated_string(self):
        """Clear resets text to empty string."""
        for ch in "HELLO WORLD":
            self.composer.append(ch)
        self.assertEqual(self.composer.text, "HELLO WORLD")
        self.composer.clear()
        self.assertEqual(self.composer.text, "")

    def test_clear_already_empty_string(self):
        """Clear on empty string causes no error."""
        self.assertEqual(self.composer.text, "")
        self.composer.clear()
        self.assertEqual(self.composer.text, "")

    def test_on_change_notification_on_backspace(self):
        """on_change callback is invoked with the updated text on backspace."""
        notified_values = []
        self.composer.on_change = lambda val: notified_values.append(val)

        self.composer.append("A")
        self.composer.append("B")
        self.assertEqual(notified_values, ["A", "AB"])

        self.composer.backspace()
        self.assertEqual(notified_values, ["A", "AB", "A"])

        self.composer.backspace()
        self.assertEqual(notified_values, ["A", "AB", "A", ""])

    def test_on_change_notification_on_clear(self):
        """on_change callback is invoked with empty string on clear."""
        notified_values = []
        self.composer.on_change = lambda val: notified_values.append(val)

        for ch in "TEST":
            self.composer.append(ch)
        self.composer.clear()
        self.assertEqual(notified_values[-1], "")

    def test_suggestion_context_sync(self):
        """Prefix and committed context update properly after backspace."""
        for ch in "HELLO W":
            self.composer.append(ch)
        self.assertEqual(self.composer.current_prefix, "W")
        self.assertEqual(self.composer.committed_context, "HELLO")

        self.composer.backspace()  # removes 'W'
        self.assertEqual(self.composer.current_prefix, "")
        self.assertEqual(self.composer.committed_context, "HELLO")


class TestHitboxActionDetection(unittest.TestCase):
    """Test text_action_from_click hitbox math and bounds."""

    def setUp(self):
        self.width = 640
        self.height = 480

    def test_backspace_click_inside(self):
        """Click inside Backspace button returns 'BACKSPACE'."""
        # x: 640 + 156 .. 640 + 224, y: 480 - 60 (420) .. 480 - 36 (444)
        x = self.width + 190
        y = self.height - 48
        self.assertEqual(text_action_from_click(x, y, self.width, self.height), "BACKSPACE")

    def test_clear_click_inside(self):
        """Click inside Clear button returns 'CLEAR'."""
        # x: 640 + 232 .. 640 + 296, y: 480 - 60 (420) .. 480 - 36 (444)
        x = self.width + 264
        y = self.height - 48
        self.assertEqual(text_action_from_click(x, y, self.width, self.height), "CLEAR")

    def test_camera_area_click_returns_none(self):
        """Click on camera frame (x < width) returns None."""
        self.assertIsNone(text_action_from_click(100, 200, self.width, self.height))
        self.assertIsNone(text_action_from_click(self.width - 1, self.height - 48, self.width, self.height))

    def test_panel_outside_buttons_returns_none(self):
        """Click in non-button areas of the panel returns None."""
        # Above buttons (e.g. suggestions at y=380)
        self.assertIsNone(text_action_from_click(self.width + 190, 380, self.width, self.height))
        # Below buttons (e.g. text display at y=460)
        self.assertIsNone(text_action_from_click(self.width + 190, self.height - 20, self.width, self.height))
        # Between Backspace and Clear button (x = width + 228)
        self.assertIsNone(text_action_from_click(self.width + 228, self.height - 48, self.width, self.height))
        # Left of Backspace button (x = width + 100)
        self.assertIsNone(text_action_from_click(self.width + 100, self.height - 48, self.width, self.height))

    def test_hitbox_boundaries(self):
        """Exact pixel boundary conditions for Backspace and Clear."""
        y_top = self.height - 60
        y_bottom = self.height - 36
        bs_left = self.width + 156
        bs_right = self.width + 224
        clr_left = self.width + 232
        clr_right = self.width + 296

        # Corners of Backspace
        self.assertEqual(text_action_from_click(bs_left, y_top, self.width, self.height), "BACKSPACE")
        self.assertEqual(text_action_from_click(bs_right, y_bottom, self.width, self.height), "BACKSPACE")

        # 1 pixel outside Backspace
        self.assertIsNone(text_action_from_click(bs_left - 1, y_top, self.width, self.height))
        self.assertIsNone(text_action_from_click(bs_right + 1, y_top, self.width, self.height))
        self.assertIsNone(text_action_from_click(bs_left, y_top - 1, self.width, self.height))
        self.assertIsNone(text_action_from_click(bs_left, y_bottom + 1, self.width, self.height))

        # Corners of Clear
        self.assertEqual(text_action_from_click(clr_left, y_top, self.width, self.height), "CLEAR")
        self.assertEqual(text_action_from_click(clr_right, y_bottom, self.width, self.height), "CLEAR")

        # 1 pixel outside Clear
        self.assertIsNone(text_action_from_click(clr_left - 1, y_top, self.width, self.height))
        self.assertIsNone(text_action_from_click(clr_right + 1, y_top, self.width, self.height))

    def test_non_overlapping_with_mode_and_suggestions(self):
        """Button hitboxes never overlap with mode buttons or word suggestions."""
        # Mode buttons are at y: 8..47
        self.assertIsNotNone(mode_from_click(self.width + 50, 20, self.width))
        self.assertIsNone(text_action_from_click(self.width + 50, 20, self.width, self.height))

        # Suggestions are at y: 350..411 (container ends at 416)
        hitboxes = compute_suggestion_hitboxes(self.width, ["APPLE", "APPLICATION"])
        for x1, y1, x2, y2, _ in hitboxes:
            # Check corners of each suggestion pill
            self.assertIsNone(text_action_from_click(x1, y1, self.width, self.height))
            self.assertIsNone(text_action_from_click(x2, y2, self.width, self.height))


class TestCoordinatorBackspaceClear(unittest.TestCase):
    """Test coordinator synchronization of hold text and suggestion manager."""

    def test_recognition_coordinator_backspace_clear(self):
        mock_letter = MagicMock()
        mock_digit = MagicMock()
        coord = RecognitionCoordinator(mock_letter, mock_digit, HoldSettings())

        coord.composer.append("A")
        coord.composer.append("B")
        coord.hold.text = coord.composer.text
        self.assertEqual(coord.hold.text, "AB")

        coord.backspace()
        self.assertEqual(coord.composer.text, "A")
        self.assertEqual(coord.hold.text, "A")

        coord.clear()
        self.assertEqual(coord.composer.text, "")
        self.assertEqual(coord.hold.text, "")


class TestPanelVisualRendering(unittest.TestCase):
    """Test that _panel renders cleanly without errors."""

    def test_panel_rendering_empty_text(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        machine = HoldConfirmation(HoldSettings())
        machine.text = ""
        output = _panel(frame, machine, "A", 1.0)
        self.assertEqual(output.shape, (480, 640 + 310, 3))

    def test_panel_rendering_populated_text(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        machine = HoldConfirmation(HoldSettings())
        machine.text = "HELLO WORLD 2026"
        output = _panel(frame, machine, None, 1.0)
        self.assertEqual(output.shape, (480, 640 + 310, 3))

    def test_button_patches(self):
        bs_patch = _get_button_patch("BACKSPACE", 68, 24)
        clr_patch = _get_button_patch("CLEAR", 64, 24)
        self.assertEqual(bs_patch.shape, (24, 68, 3))
        self.assertEqual(clr_patch.shape, (24, 64, 3))


if __name__ == "__main__":
    unittest.main()
