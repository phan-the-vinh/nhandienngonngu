"""Phase 1: Test-only suite for SPACE integration into NhanDienCuChi.

Tests cover:
- Section A: TextComposer behavior with letters, digits, literal space ' ', and rejection of token 'SPACE'.
- Section B: RecognitionCoordinator translation of label 'SPACE' to literal ' ' before calling TextComposer.append().
- Section C: HoldConfirmation single-commit, release re-arm, and multi-commit lifecycle with 'SPACE'.
- Section D: Text synchronization between TextComposer, HoldConfirmation.text, and UI-facing text state.
- Section E: Regression protection for existing letters, digits, mode-switching, and tracking loss.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from app.inference.letter_inference import Prediction
from app.inference.unified_static_coordinator import (
    RecognitionCoordinator,
    TextComposer,
)
from app.interaction.hold_confirmation import HoldConfirmation, HoldSettings
from tests.test_hold_confirmation import hand as make_hold_hand
from tests.test_unified_camera_experiment import FakePredictor, synthetic_hand


class TextComposerSpaceTests(unittest.TestCase):
    """Section A: TextComposer requirements."""

    def setUp(self):
        self.composer = TextComposer()

    def test_accepts_single_letter(self):
        """A single letter (e.g. 'A', 'b') must be accepted."""
        self.composer.append("A")
        self.assertEqual(self.composer.text, "A")
        self.composer.append("b")
        self.assertEqual(self.composer.text, "Ab")

    def test_accepts_single_digit(self):
        """A single digit (e.g. '0', '9') must be accepted."""
        self.composer.append("1")
        self.assertEqual(self.composer.text, "1")
        self.composer.append("0")
        self.assertEqual(self.composer.text, "10")

    def test_accepts_literal_ascii_space(self):
        """A literal ASCII space ' ' must be accepted as one character."""
        self.composer.append("A")
        self.composer.append(" ")
        self.composer.append("B")
        self.assertEqual(self.composer.text, "A B")

    def test_rejects_string_token_space(self):
        """The 5-character token 'SPACE' must be rejected as an input character."""
        with self.assertRaises(ValueError):
            self.composer.append("SPACE")
        self.assertEqual(self.composer.text, "")

    def test_preserves_existing_invalid_input_rejections(self):
        """Empty strings, multi-character strings, and punctuation must be rejected."""
        for invalid in ("", "AB", "  ", ".", "!", "\n", "\t"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.composer.append(invalid)
        self.assertEqual(self.composer.text, "")


class RecognitionCoordinatorSpaceTests(unittest.TestCase):
    """Section B: RecognitionCoordinator requirements."""

    def setUp(self):
        self.letters = FakePredictor("A", 0.0)
        self.digits = FakePredictor("0", 0.5)
        self.coordinator = RecognitionCoordinator(self.letters, self.digits, HoldSettings())
        self.hand = synthetic_hand()

    def feed(self, moment: float):
        return self.coordinator.process_hand(self.hand, (480, 640, 3), moment)[1]

    def confirm_pose(self, label: str, shape: float, start: float):
        self.letters.label = label
        self.letters.shape = shape
        for moment in (start, start + 0.1, start + 0.2):
            self.assertIsNone(self.feed(moment))
        self.assertEqual(self.coordinator.hold.state, "HOLDING")
        return self.feed(start + 3.0)

    def test_converts_space_label_to_literal_space_for_composer(self):
        """When label 'SPACE' is committed, composer must receive literal ' ', not 'SPACE'."""
        self.assertEqual(self.confirm_pose("A", 0.0, 0.0), "A")
        self.coordinator.missing_hand(3.1)
        self.coordinator.missing_hand(3.8)
        self.assertEqual(self.coordinator.hold.state, "IDLE")

        self.confirm_pose("SPACE", 0.2, 3.9)
        self.assertEqual(self.coordinator.composer.text, "A ")
        self.assertNotIn("SPACE", self.coordinator.composer.text)

    def test_other_labels_retain_existing_behavior(self):
        """Non-SPACE labels (letters, digits) retain their literal string behavior."""
        self.assertEqual(self.confirm_pose("A", 0.0, 0.0), "A")
        self.assertEqual(self.coordinator.composer.text, "A")
        self.coordinator.missing_hand(3.1)
        self.coordinator.missing_hand(3.8)
        self.assertEqual(self.confirm_pose("B", 0.1, 3.9), "B")
        self.assertEqual(self.coordinator.composer.text, "AB")

    def test_composer_receives_literal_space_between_words(self):
        """Sequencing A -> SPACE -> B produces 'A B' in TextComposer."""
        self.confirm_pose("A", 0.0, 0.0)
        self.coordinator.missing_hand(3.1)
        self.coordinator.missing_hand(3.8)
        self.confirm_pose("SPACE", 0.2, 3.9)
        self.coordinator.missing_hand(7.0)
        self.coordinator.missing_hand(7.7)
        self.confirm_pose("B", 0.3, 7.8)
        self.assertEqual(self.coordinator.composer.text, "A B")


class HoldConfirmationSpaceTests(unittest.TestCase):
    """Section C: HoldConfirmation requirements."""

    def setUp(self):
        self.machine = HoldConfirmation(HoldSettings())

    def begin_hold(self, label="SPACE", *, shape=0.2, start=0.0):
        for offset in (0.0, 0.1, 0.2):
            self.assertIsNone(
                self.machine.update(make_hold_hand(label, shape=shape), timestamp=start + offset)
            )
        self.assertEqual(self.machine.state, "HOLDING")
        self.assertEqual(self.machine.hold_started_at, start)

    def test_continuous_space_hold_commits_only_once(self):
        """A continuous SPACE hold must commit once at 3.0s and never repeat while held."""
        self.begin_hold("SPACE", shape=0.2, start=0.0)
        self.assertIsNone(self.machine.update(make_hold_hand("SPACE", shape=0.2), timestamp=2.99))
        committed = self.machine.update(make_hold_hand("SPACE", shape=0.2), timestamp=3.0)
        self.assertEqual(committed, "SPACE")
        self.assertEqual(self.machine.state, "CONFIRMED")

        # Next frame with continuous holding transitions to WAIT_RELEASE and must not commit again
        self.assertIsNone(self.machine.update(make_hold_hand("SPACE", shape=0.2), timestamp=3.1))
        self.assertEqual(self.machine.state, "WAIT_RELEASE")

        for moment in (4.0, 5.0, 6.0, 9.0):
            self.assertIsNone(
                self.machine.update(make_hold_hand("SPACE", shape=0.2), timestamp=moment)
            )
        self.assertEqual(self.machine.state, "WAIT_RELEASE")
        self.assertEqual(self.machine.text, "SPACE")

    def test_releasing_hand_rearms_confirmation_after_space(self):
        """Releasing hand for release_seconds re-arms the machine from WAIT_RELEASE to IDLE."""
        self.begin_hold("SPACE", shape=0.2, start=0.0)
        self.assertEqual(
            self.machine.update(make_hold_hand("SPACE", shape=0.2), timestamp=3.0),
            "SPACE",
        )
        self.assertEqual(self.machine.state, "CONFIRMED")

        # Hand missing at 3.1 starts release timer and enters WAIT_RELEASE
        self.machine.update(None, timestamp=3.1)
        self.assertEqual(self.machine.state, "WAIT_RELEASE")
        self.machine.update(None, timestamp=3.4)
        self.assertEqual(self.machine.state, "WAIT_RELEASE")
        # At 3.7 (missing for 0.6s >= release_seconds 0.5s), re-arms to IDLE
        self.machine.update(None, timestamp=3.7)
        self.assertEqual(self.machine.state, "IDLE")
        self.assertTrue(self.machine.rearmed)

    def test_subsequent_valid_hold_can_commit_another_character(self):
        """After release and rearm, a subsequent hold (SPACE or letter) can commit."""
        self.begin_hold("SPACE", shape=0.2, start=0.0)
        self.machine.update(make_hold_hand("SPACE", shape=0.2), timestamp=3.0)
        self.machine.update(None, timestamp=3.1)
        self.machine.update(None, timestamp=3.8)
        self.assertEqual(self.machine.state, "IDLE")

        # Subsequent hold with letter 'A'
        self.begin_hold("A", shape=0.0, start=3.9)
        committed = self.machine.update(make_hold_hand("A", shape=0.0), timestamp=6.9)
        self.assertEqual(committed, "A")

    def test_space_recognition_does_not_cause_duplicate_commits(self):
        """Jitter during SPACE hold does not produce double commits."""
        self.begin_hold("SPACE", shape=0.2, start=0.0)
        self.machine.update(make_hold_hand("SPACE", shape=0.2), timestamp=3.0)
        for moment in (3.1, 3.2, 3.3, 3.4, 3.5):
            self.machine.update(
                make_hold_hand("SPACE", x=102.0, y=103.0, scale=101.0, shape=0.21),
                timestamp=moment,
            )
        self.assertEqual(self.machine.state, "WAIT_RELEASE")


class TextSynchronizationSpaceTests(unittest.TestCase):
    """Section D: Text synchronization requirements."""

    def setUp(self):
        self.letters = FakePredictor("A", 0.0)
        self.digits = FakePredictor("0", 0.5)
        self.coordinator = RecognitionCoordinator(self.letters, self.digits, HoldSettings())
        self.hand = synthetic_hand()

    def feed(self, moment: float):
        return self.coordinator.process_hand(self.hand, (480, 640, 3), moment)[1]

    def confirm_pose(self, label: str, shape: float, start: float):
        self.letters.label = label
        self.letters.shape = shape
        for moment in (start, start + 0.1, start + 0.2):
            self.assertIsNone(self.feed(moment))
        return self.feed(start + 3.0)

    def test_composer_and_hold_text_remain_synchronized_for_space(self):
        """Committed text must remain consistent between composer and hold-confirmation state."""
        self.confirm_pose("A", 0.0, 0.0)
        self.coordinator.missing_hand(3.1)
        self.coordinator.missing_hand(3.8)

        self.confirm_pose("SPACE", 0.2, 3.9)
        self.assertEqual(self.coordinator.composer.text, "A ")
        # Hold state text must match composer text (single space, not 5-letter 'SPACE')
        self.assertEqual(self.coordinator.hold.text, self.coordinator.composer.text)

    def test_mode_switch_preserves_space_text_synchronization(self):
        """Switching mode to 123 preserves space in text for the new mode's hold instance."""
        self.confirm_pose("A", 0.0, 0.0)
        self.coordinator.missing_hand(3.1)
        self.coordinator.missing_hand(3.8)
        self.confirm_pose("SPACE", 0.2, 3.9)
        self.assertEqual(self.coordinator.composer.text, "A ")

        # Switch mode to 123
        self.coordinator.switch_mode("123")
        self.assertEqual(self.coordinator.composer.text, "A ")
        self.assertEqual(self.coordinator.hold.text, "A ")


class RegressionProtectionTests(unittest.TestCase):
    """Section E: Regression protection for existing behavior."""

    def setUp(self):
        self.letters = FakePredictor("A", 0.0)
        self.digits = FakePredictor("0", 0.5)
        self.coordinator = RecognitionCoordinator(self.letters, self.digits, HoldSettings())
        self.hand = synthetic_hand()

    def feed(self, moment: float):
        return self.coordinator.process_hand(self.hand, (480, 640, 3), moment)[1]

    def confirm_pose(self, label: str, shape: float, start: float):
        self.letters.label = label
        self.letters.shape = shape
        for moment in (start, start + 0.1, start + 0.2):
            self.assertIsNone(self.feed(moment))
        return self.feed(start + 3.0)

    def test_existing_letters_flow_unchanged(self):
        """Existing letter sequence 'OK' works as before."""
        self.confirm_pose("O", 0.0, 0.0)
        self.coordinator.missing_hand(3.1)
        self.coordinator.missing_hand(3.8)
        self.confirm_pose("K", 0.4, 3.9)
        self.assertEqual(self.coordinator.composer.text, "OK")
        self.assertEqual(self.coordinator.hold.text, "OK")

    def test_existing_digits_flow_unchanged(self):
        """Existing digit sequence in 123 mode works as before."""
        self.coordinator.switch_mode("123")
        self.digits.label = "7"
        for moment in (0.0, 0.1, 0.2):
            self.assertIsNone(self.feed(moment))
        self.assertEqual(self.feed(3.0), "7")
        self.assertEqual(self.coordinator.composer.text, "7")
        self.assertEqual(self.coordinator.hold.text, "7")

    def test_tracking_loss_resets_candidate_without_corrupting_text(self):
        """Losing hand tracking during stabilization resets candidate without altering text."""
        self.confirm_pose("A", 0.0, 0.0)
        self.coordinator.missing_hand(3.1)
        self.coordinator.missing_hand(3.8)

        # Start stabilizing SPACE, but lose hand before hold completes
        self.letters.label = "SPACE"
        self.letters.shape = 0.2
        self.feed(3.9)
        self.coordinator.missing_hand(4.0)
        self.assertEqual(self.coordinator.hold.state, "IDLE")
        self.assertEqual(self.coordinator.composer.text, "A")


if __name__ == "__main__":
    unittest.main()
