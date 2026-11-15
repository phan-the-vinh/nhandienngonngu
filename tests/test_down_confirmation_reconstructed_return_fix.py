"""Offline regression checks for the isolated reconstructed camera return fix."""

from __future__ import annotations

import unittest

from app.inference.letter_inference import TemporalSmoother
from app.interaction.down_confirmation import DownConfirmation, DownSettings
from app.interaction.down_confirmation_reconstructed_return_fix import DownConfirmationReturnFix
from app.inference.push_confirmation_camera_reconstructed_return_fix import HandObservation


def hand(label="A", y=100.0, *, x=100.0, scale=100.0, wrist_y=None, shape=0.0):
    return HandObservation(
        label=label, confidence=.95, scale_px=scale, width_px=80.0,
        wrist_px=(x, y - 10.0 if wrist_y is None else wrist_y),
        shape=(shape,) * 63, palm_center_px=(x, y),
    )


class VotePipeline:
    """Same seven-frame vote behavior, with no post-confirmation reset."""

    labels = ("A", "P", "L", "E")

    def __init__(self):
        self.smoother = TemporalSmoother(class_count=len(self.labels))

    def observation(self, raw_label, y, *, x=100.0, scale=100.0):
        raw_index = self.labels.index(raw_label)
        probabilities = [float(index == raw_index) for index in range(len(self.labels))]
        stable_index, confidence = self.smoother.update(raw_index, probabilities)
        stable = None if stable_index is None else self.labels[stable_index]
        eligible = raw_label if stable == raw_label else None
        return hand(eligible, y, x=x, scale=scale, shape=0.0)


class ReconstructedReturnFixTests(unittest.TestCase):
    def setUp(self):
        self.settings = DownSettings()
        self.machine = DownConfirmationReturnFix(self.settings)

    def ready(self, label="A", y=100.0, **geometry):
        for _ in range(self.settings.hold_frames):
            self.assertIsNone(self.machine.update(hand(label, y, **geometry)))
        self.assertEqual(self.machine.state, "READY")

    def confirm(self, label="A", y=100.0, **geometry):
        self.ready(label, y, **geometry)
        self.assertIsNone(self.machine.update(hand(label, y + 45, **geometry)))
        self.assertEqual(self.machine.update(hand(label, y + 45, **geometry)), label)

    def test_old_return_sticks_after_valid_lift_with_new_pose_and_scale(self):
        old = DownConfirmation(self.settings)
        for _ in range(7):
            old.update(hand())
        old.update(hand(y=145))
        self.assertEqual(old.update(hand(y=145)), "A")
        for _ in range(8):
            old.update(hand("P", 124, x=160, scale=120, shape=.1))
        self.assertEqual(old.state, "WAIT_RETURN")
        self.assertEqual(old._return_count, 0)

    def test_lift_unlocks_with_new_letter_size_and_horizontal_position(self):
        self.confirm()
        for _ in range(4):
            self.assertIsNone(self.machine.update(hand("P", 124, x=160, scale=120, shape=.1)))
        self.assertEqual(self.machine.state, "IDLE")
        self.ready("P", 124, x=160, scale=120, shape=.1)
        self.assertIsNone(self.machine.update(hand("P", 169, x=160, scale=120, shape=.1)))
        self.assertEqual(self.machine.update(hand("P", 169, x=160, scale=120, shape=.1)), "P")
        self.assertEqual(self.machine.text, "AP")

    def test_held_low_missing_hand_and_pose_only_change_do_not_unlock(self):
        self.confirm()
        for _ in range(8):
            self.assertIsNone(self.machine.update(hand("P", 145, x=160, scale=120)))
        self.assertEqual(self.machine.state, "WAIT_RETURN")
        self.assertEqual(self.machine.text, "A")
        self.machine.update(None)
        self.assertEqual(self.machine._return_count, 0)
        # Palm center moves, but the wrist does not: no whole-hand return.
        for _ in range(6):
            self.machine.update(hand("P", 124, x=160, scale=120, wrist_y=135))
        self.assertEqual(self.machine.state, "WAIT_RETURN")

    def test_partial_return_never_unlocks_or_repeats(self):
        self.confirm()
        for _ in range(12):
            self.machine.update(hand(y=145))
        for _ in range(6):
            self.machine.update(hand(y=137))
        self.assertEqual(self.machine.state, "WAIT_RETURN")
        self.assertEqual(self.machine.text, "A")

    def test_apple_with_unreset_prediction_votes_and_repeated_p(self):
        votes = VotePipeline()
        expected = ""
        for label in "APPLE":
            for _ in range(15):
                self.machine.update(votes.observation(label, 100))
            self.assertEqual(self.machine.state, "READY")
            self.assertEqual(self.machine.pending, label)
            self.assertIsNone(self.machine.update(votes.observation(label, 145)))
            self.assertEqual(self.machine.update(votes.observation(label, 145)), label)
            expected += label
            self.assertEqual(self.machine.text, expected)
            for _ in range(8):
                self.assertIsNone(self.machine.update(votes.observation(label, 145)))
            self.assertEqual(self.machine.state, "WAIT_RETURN")
            self.assertEqual(self.machine.text, expected)
            for _ in range(4):
                self.assertIsNone(self.machine.update(votes.observation(label, 100)))
            self.assertEqual(self.machine.state, "IDLE")
        self.assertEqual(self.machine.text, "APPLE")


if __name__ == "__main__":
    unittest.main()
