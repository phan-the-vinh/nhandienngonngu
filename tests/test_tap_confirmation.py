"""Offline TAP tests; camera access is forbidden."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.interaction.tap_confirmation import TapConfirmation, TapSettings


def hand(y=100.0, *, x=100.0, scale=100.0, label="A", confidence=.95,
         shape=None):
    return SimpleNamespace(
        label=label, confidence=confidence, scale_px=scale,
        wrist_px=(x, y), palm_center_px=(x, y),
        shape=tuple([0.0] * 63 if shape is None else shape))


class TapConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.machine = TapConfirmation(TapSettings(
            hold_frames=3, down_frames=2, return_frames=2,
            reset_frames=3, max_duration_ms=500))

    def ready(self):
        for stamp in (0, 30, 60):
            self.assertIsNone(self.machine.update(hand(), timestamp_ms=stamp))
        self.assertEqual(self.machine.state, "READY")

    def test_complete_down_return_commits_once_and_resets(self):
        self.ready()
        self.machine.update(hand(140), timestamp_ms=90)
        self.machine.update(hand(140), timestamp_ms=120)
        self.assertEqual(self.machine.state, "TAP_DOWN")
        self.machine.update(hand(125), timestamp_ms=150)
        self.assertEqual(self.machine.state, "TAP_RETURN")
        self.assertIsNone(self.machine.update(hand(104), timestamp_ms=180))
        self.assertEqual(self.machine.update(hand(104), timestamp_ms=210), "A")
        self.assertEqual(self.machine.state, "CONFIRMED")
        for stamp in (240, 270, 300, 330, 360, 390):
            self.assertIsNone(self.machine.update(hand(104), timestamp_ms=stamp))
        self.assertEqual(self.machine.text, "A")
        self.assertEqual(self.machine.state, "READY")
        self.machine.update(hand(140), timestamp_ms=420)
        self.machine.update(hand(140), timestamp_ms=450)
        self.machine.update(hand(125), timestamp_ms=480)
        self.machine.update(hand(104), timestamp_ms=510)
        self.assertEqual(self.machine.update(hand(104), timestamp_ms=540), "A")
        self.assertEqual(self.machine.text, "AA")

    def test_down_without_return_times_out_without_confirmation(self):
        self.ready()
        self.machine.update(hand(140), timestamp_ms=90)
        self.machine.update(hand(140), timestamp_ms=120)
        self.assertEqual(self.machine.state, "TAP_DOWN")
        self.machine.update(hand(140), timestamp_ms=700)
        self.assertEqual(self.machine.state, "IDLE")
        self.assertEqual(self.machine.text, "")

    def test_idle_hand_and_small_jitter_do_not_confirm(self):
        self.ready()
        for stamp, y in enumerate([100, 103, 98, 108, 105, 99], start=1):
            self.machine.update(hand(y), timestamp_ms=100 + 30 * stamp)
        self.assertEqual(self.machine.state, "READY")
        self.assertEqual(self.machine.text, "")

    def test_sideways_motion_and_scale_change_cancel_tap(self):
        self.ready()
        self.machine.update(hand(140, x=170), timestamp_ms=90)
        self.assertEqual(self.machine.state, "IDLE")
        self.ready()
        self.machine.update(hand(140, scale=125), timestamp_ms=90)
        self.assertEqual(self.machine.state, "IDLE")
        self.assertEqual(self.machine.text, "")

    def test_tracking_loss_during_tap_cancels_without_commit(self):
        self.ready()
        self.machine.update(hand(140), timestamp_ms=90)
        self.machine.update(hand(140), timestamp_ms=120)
        self.machine.update(None, timestamp_ms=150)
        self.assertEqual(self.machine.state, "IDLE")
        for stamp in (180, 210, 240):
            self.machine.update(hand(100), timestamp_ms=stamp)
        self.assertEqual(self.machine.text, "")

    def test_shape_or_label_instability_and_dynamic_guard_cancel(self):
        self.ready()
        self.machine.update(hand(140, shape=[1.0] * 63), timestamp_ms=90)
        self.assertEqual(self.machine.state, "IDLE")
        self.ready()
        self.machine.update(hand(140, label="B"), timestamp_ms=90)
        self.assertEqual(self.machine.state, "IDLE")
        self.ready()
        self.machine.update(hand(140), timestamp_ms=90, dynamic_active=True)
        self.assertEqual(self.machine.state, "IDLE")
        self.assertEqual(self.machine.text, "")

    def test_post_confirm_latch_ignores_missing_hand_and_recalibrate(self):
        self.ready()
        for stamp, y in ((90, 140), (120, 140), (150, 125),
                         (180, 104), (210, 104)):
            self.machine.update(hand(y), timestamp_ms=stamp)
        self.machine.update(None, timestamp_ms=240)
        self.machine.recalibrate()
        self.assertEqual(self.machine.state, "WAIT_RESET")
        self.assertEqual(self.machine.text, "A")

    def test_backup_state_machine_is_independent_of_push_camera(self):
        machine = TapConfirmation(TapSettings(hold_frames=1))
        self.assertIsNone(machine.update(hand(shape=[0.0] * 5), timestamp_ms=0))
        self.assertEqual(machine.state, "READY")


if __name__ == "__main__":
    unittest.main()
