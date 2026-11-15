"""Synthetic HOLD_CONFIRM tests; never open a camera."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.interaction.hold_confirmation import HoldConfirmation, HoldSettings
from app.inference import hold_confirmation_camera_test as camera_test


def hand(label="A", *, x=100.0, y=100.0, scale=100.0, shape=0.0, confidence=.95):
    return SimpleNamespace(label=label, confidence=confidence, scale_px=scale,
                           palm_center_px=(x, y), shape=(shape,) * 63)


class HoldConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.machine = HoldConfirmation(HoldSettings())

    def begin(self, label="A", *, shape=0.0, start=0.0):
        for offset in (0.0, .1, .2):
            self.assertIsNone(self.machine.update(hand(label, shape=shape),
                                                  timestamp=start + offset))
        self.assertEqual(self.machine.state, "HOLDING")
        self.assertEqual(self.machine.hold_started_at, start)

    def test_three_seconds_confirms_once_and_six_seconds_does_not_repeat(self):
        self.begin()
        self.assertIsNone(self.machine.update(hand(), timestamp=2.99))
        self.assertAlmostEqual(self.machine.remaining_seconds, .01)
        self.assertEqual(self.machine.update(hand(), timestamp=3.0), "A")
        for moment in (3.1, 4.0, 6.0, 9.0):
            self.assertIsNone(self.machine.update(hand(), timestamp=moment))
        self.assertEqual(self.machine.text, "A")
        self.assertEqual(self.machine.state, "WAIT_RELEASE")

    def test_under_three_seconds_never_confirms(self):
        self.begin()
        self.machine.update(hand(), timestamp=2.9)
        self.assertEqual(self.machine.text, "")

    def test_change_a_to_b_cancels_old_timer(self):
        self.begin()
        self.machine.update(hand(), timestamp=2.0)
        self.machine.update(hand("B", shape=.30), timestamp=2.1)
        self.assertEqual(self.machine.state, "STABILIZING")
        self.assertEqual(self.machine.pending, "B")
        self.assertEqual(self.machine.progress, 0.0)
        self.machine.update(hand("B", shape=.30), timestamp=2.2)
        self.machine.update(hand("B", shape=.30), timestamp=2.3)
        self.assertEqual(self.machine.hold_started_at, 2.1)
        self.assertIsNone(self.machine.update(hand("B", shape=.30), timestamp=3.0))
        self.assertEqual(self.machine.update(hand("B", shape=.30), timestamp=5.1), "B")
        self.assertEqual(self.machine.text, "B")

    def test_a_then_p_can_confirm_without_no_hand_release(self):
        self.begin()
        self.assertEqual(self.machine.update(hand(), timestamp=3.0), "A")
        self.machine.update(hand(), timestamp=3.1)
        self.machine.update(hand("P", shape=.30), timestamp=3.2)
        self.assertEqual(self.machine.state, "STABILIZING")
        self.machine.update(hand("P", shape=.30), timestamp=3.3)
        self.machine.update(hand("P", shape=.30), timestamp=3.4)
        self.assertEqual(self.machine.state, "HOLDING")
        self.assertEqual(self.machine.hold_started_at, 3.2)
        self.assertEqual(self.machine.update(hand("P", shape=.30), timestamp=6.2), "P")
        self.assertEqual(self.machine.text, "AP")

    def test_repeated_p_requires_real_release(self):
        self.begin("P")
        self.assertEqual(self.machine.update(hand("P"), timestamp=3.0), "P")
        self.machine.update(None, timestamp=3.1)
        self.machine.update(None, timestamp=3.3)
        self.machine.update(hand("P"), timestamp=3.4)
        self.assertEqual(self.machine.state, "WAIT_RELEASE")
        self.machine.update(None, timestamp=4.0)
        self.machine.update(None, timestamp=4.61)
        self.assertEqual(self.machine.state, "IDLE")
        self.assertTrue(self.machine.rearmed)
        for moment in (4.7, 4.8, 4.9):
            self.machine.update(hand("P"), timestamp=moment)
        self.assertEqual(self.machine.state, "HOLDING")
        self.assertEqual(self.machine.update(hand("P"), timestamp=7.7), "P")
        self.assertEqual(self.machine.text, "PP")

    def test_same_pose_label_flicker_does_not_rearm(self):
        self.begin()
        self.machine.update(hand(), timestamp=3.0)
        for moment in (3.1, 3.2, 3.3, 3.4, 4.0):
            self.machine.update(hand("P", shape=.01), timestamp=moment)
        self.assertEqual(self.machine.state, "WAIT_RELEASE")
        self.assertEqual(self.machine.text, "A")

    def test_tracking_loss_jitter_and_unstable_prediction(self):
        self.begin()
        self.machine.update(hand(x=103, y=104, scale=102, shape=.02), timestamp=1.0)
        self.assertEqual(self.machine.state, "HOLDING")
        self.machine.update(None, timestamp=1.1)
        self.assertEqual(self.machine.state, "IDLE")
        self.assertEqual(self.machine.progress, 0.0)
        self.machine.update(hand(), timestamp=1.2)
        self.machine.update(hand(label=None), timestamp=1.3)
        self.assertEqual(self.machine.state, "IDLE")
        self.machine.update(hand(), timestamp=1.4)
        self.machine.update(hand(x=150), timestamp=1.5)
        self.assertEqual(self.machine.state, "STABILIZING")
        self.assertEqual(self.machine.stable_count, 1)
        self.assertEqual(self.machine.text, "")

    def test_elapsed_time_not_frame_count(self):
        self.begin()
        self.assertIsNone(self.machine.update(hand(), timestamp=.21))
        self.assertIsNone(self.machine.update(hand(), timestamp=2.8))
        self.assertEqual(self.machine.update(hand(), timestamp=3.0), "A")

    def test_check_only_never_opens_camera(self):
        with patch.object(camera_test.cv2, "VideoCapture", side_effect=AssertionError("camera opened")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(camera_test.main(["--check-only"]), 0)
        checked = json.loads(output.getvalue())
        self.assertFalse(checked["camera_opened"])
        self.assertEqual(checked["settings"]["hold_seconds"], 3.0)


if __name__ == "__main__":
    unittest.main()
