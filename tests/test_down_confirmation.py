"""Synthetic DOWN_CONFIRM tests; never open a camera or decode video."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.inference import push_confirmation_camera_test as camera_test
from app.inference.letter_inference import TemporalSmoother
from app.interaction.down_confirmation import DownConfirmation, DownSettings


def hand(y=100.0, *, x=100.0, scale=100.0, label="A", confidence=.95,
         shape=None):
    return SimpleNamespace(
        label=label, confidence=confidence, scale_px=scale,
        palm_center_px=(x, y), shape=tuple([0.0] * 63 if shape is None else shape))


class VotePipeline:
    """Offline stand-in for the real seven-frame letter vote buffer."""

    labels = ("A", "P", "L")

    def __init__(self):
        self.smoother = TemporalSmoother(class_count=3)
        self.reset_count = 0

    def reset(self):
        self.smoother.reset()
        self.reset_count += 1

    def observation(self, raw_label, y=100.0, *, shape=None):
        raw_index = self.labels.index(raw_label)
        probabilities = [float(index == raw_index) for index in range(3)]
        stable_index, confidence = self.smoother.update(raw_index, probabilities)
        stable_label = None if stable_index is None else self.labels[stable_index]
        accepted_label = raw_label if stable_label == raw_label else None
        return hand(y, label=accepted_label, confidence=confidence or 0.0, shape=shape)


class DownConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.machine = DownConfirmation(DownSettings(
            hold_frames=3, down_frames=2, return_frames=3))

    def ready(self, label="A"):
        for _ in range(3):
            self.assertIsNone(self.machine.update(hand(label=label)))
        self.assertEqual(self.machine.state, "READY")
        self.assertEqual(self.machine.pending, label)

    def test_slow_down_confirms_without_waiting_for_return(self):
        self.ready()
        commits = [self.machine.update(hand(y)) for y in (108, 116, 124, 132, 140)]
        self.assertEqual(commits.count("A"), 1)
        self.assertEqual(self.machine.state, "CONFIRMED")
        self.assertEqual(self.machine.text, "A")

    def test_faster_defaults_keep_two_frame_confirmation(self):
        defaults = DownSettings()
        self.assertEqual(defaults.hold_frames, 7)
        self.assertEqual(defaults.down_ratio, .25)
        self.assertEqual(defaults.down_frames, 2)
        self.assertEqual(defaults.smoothing_alpha, .70)
        self.assertEqual(camera_test.parse_args([]).down_hold_frames, 7)
        self.assertEqual(camera_test.parse_args([]).down_down_ratio, .25)

    def test_timed_fast_pull_confirms_on_second_threshold_frame(self):
        machine = DownConfirmation(DownSettings())
        for index in range(7):
            machine.update(hand(), timestamp_ms=index * 33.0)
        self.assertEqual(machine.state, "READY")
        self.assertEqual(machine.last_ready_latency_ms, 198.0)
        self.assertIsNone(machine.update(hand(145), timestamp_ms=231.0))
        self.assertEqual(machine.update(hand(145), timestamp_ms=264.0), "A")
        self.assertEqual(machine.last_confirm_latency_ms, 33.0)

    def test_timed_slow_pull_and_repeated_pull_after_return(self):
        machine = DownConfirmation(DownSettings())
        for index in range(7):
            machine.update(hand(), timestamp_ms=index * 33.0)
        commits = [machine.update(hand(y), timestamp_ms=231.0 + index * 33.0)
                   for index, y in enumerate((105, 110, 115, 120, 125, 130, 135))]
        self.assertEqual(commits.count("A"), 1)
        self.assertGreater(machine.last_confirm_latency_ms, 33.0)
        for _ in range(6):
            self.assertIsNone(machine.update(hand(140)))
        self.assertEqual(machine.text, "A")
        for _ in range(5):
            machine.update(hand(100))
        self.assertEqual(machine.state, "IDLE")
        for _ in range(7):
            machine.update(hand(label="B"))
        self.assertIsNone(machine.update(hand(145, label="B")))
        self.assertEqual(machine.update(hand(145, label="B")), "B")
        self.assertEqual(machine.text, "AB")

    def test_sustained_small_motion_or_stationary_hand_does_not_confirm(self):
        self.ready()
        for y in (100, 102, 98, 108, 120, 124, 123, 100, 100, 100):
            self.assertIsNone(self.machine.update(hand(y)))
        self.assertEqual(self.machine.text, "")
        self.assertEqual(self.machine.state, "READY")

    def test_one_frame_down_spike_does_not_confirm(self):
        self.ready()
        self.assertIsNone(self.machine.update(hand(145)))
        self.assertEqual(self.machine.state, "DOWN_DETECTED")
        for _ in range(5):
            self.assertIsNone(self.machine.update(hand(100)))
        self.assertEqual(self.machine.text, "")

    def test_fast_down_confirms_frozen_ready_label(self):
        self.ready()
        self.assertIsNone(self.machine.update(hand(155, label=None)))
        self.assertEqual(self.machine.state, "DOWN_DETECTED")
        self.assertEqual(self.machine.update(hand(155, label="B")), "A")
        self.assertEqual(self.machine.text, "A")

    def test_no_repeat_while_low_and_next_pull_after_return(self):
        self.ready()
        self.machine.update(hand(155))
        self.assertEqual(self.machine.update(hand(155)), "A")
        for _ in range(12):
            self.assertIsNone(self.machine.update(hand(155)))
        self.assertEqual(self.machine.state, "WAIT_RETURN")
        self.assertEqual(self.machine.text, "A")
        self.machine.update(None)
        self.assertEqual(self.machine.state, "WAIT_RETURN")
        for _ in range(4):
            self.machine.update(hand(100))
        self.assertEqual(self.machine.state, "IDLE")
        self.ready(label="B")
        self.machine.update(hand(155, label="B"))
        self.assertEqual(self.machine.update(hand(155, label="B")), "B")
        self.assertEqual(self.machine.text, "AB")

    def test_upward_jitter_stationary_and_sideways_do_not_confirm(self):
        self.ready()
        for y in (100, 103, 97, 105, 95, 60, 75, 100):
            self.assertIsNone(self.machine.update(hand(y)))
        self.assertEqual(self.machine.text, "")
        self.assertIsNone(self.machine.update(hand(155, x=170)))
        self.assertEqual(self.machine.state, "IDLE")
        self.assertEqual(self.machine.text, "")

    def test_scale_or_shape_change_and_tracking_loss_cancel_pull(self):
        self.ready()
        self.machine.update(hand(155, scale=125))
        self.assertEqual(self.machine.state, "IDLE")
        self.ready()
        self.machine.update(hand(155, shape=[1.0] * 63))
        self.assertEqual(self.machine.state, "IDLE")
        self.ready()
        self.machine.update(hand(155))
        self.machine.update(None)
        self.assertEqual(self.machine.state, "IDLE")
        self.assertEqual(self.machine.text, "")

    def test_dynamic_guard_cancels_pending_without_confirmation(self):
        self.ready()
        self.machine.update(hand(155), dynamic_active=True)
        self.assertEqual(self.machine.state, "IDLE")
        self.assertEqual(self.machine.text, "")

    def test_app_then_l_commits_appl_with_fresh_prediction_votes(self):
        pipeline = VotePipeline()
        machine = DownConfirmation(DownSettings(hold_frames=3, return_frames=3))

        def feed(raw_label, y=100.0):
            observation = pipeline.observation(raw_label, y)
            return camera_test._update_down_and_clear_confirmed_votes(pipeline, machine, observation)

        for expected_text, label in (("A", "A"), ("AP", "P"), ("APP", "P"), ("APPL", "L")):
            for _ in range(6):
                self.assertIsNone(feed(label))
            self.assertEqual(machine.state, "READY")
            self.assertEqual(machine.pending, label)
            self.assertIsNone(feed(label, 155))
            self.assertEqual(feed(label, 155), label)
            self.assertEqual(machine.text, expected_text)
            self.assertEqual(len(pipeline.smoother._history), 0)
            self.assertIsNone(feed(label, 155))
            self.assertEqual(machine.state, "WAIT_RETURN")
            for _ in range(4):
                self.assertIsNone(feed(label, 100))
            self.assertEqual(machine.state, "IDLE")
            self.assertIsNone(machine.pending)
            self.assertEqual(len(pipeline.smoother._history), 0)
        self.assertEqual(pipeline.reset_count, 8)  # commit and return for each character

    def test_locked_l_survives_raw_p_during_pull(self):
        pipeline = VotePipeline()
        machine = DownConfirmation(DownSettings(hold_frames=3))
        for _ in range(6):
            camera_test._update_down_and_clear_confirmed_votes(
                pipeline, machine, pipeline.observation("L"))
        self.assertEqual(machine.pending, "L")
        self.assertIsNone(camera_test._update_down_and_clear_confirmed_votes(
            pipeline, machine, pipeline.observation("P", 155)))
        self.assertEqual(camera_test._update_down_and_clear_confirmed_votes(
            pipeline, machine, pipeline.observation("P", 155)), "L")
        self.assertEqual(machine.text, "L")

    def test_pull_before_ready_cannot_commit(self):
        pipeline = VotePipeline()
        machine = DownConfirmation(DownSettings(hold_frames=3))
        for _ in range(3):
            self.assertIsNone(camera_test._update_down_and_clear_confirmed_votes(
                pipeline, machine, pipeline.observation("P", 155)))
        self.assertEqual(machine.state, "IDLE")
        self.assertEqual(machine.text, "")

    def test_camera_observation_uses_pixel_palm_center(self):
        points = [SimpleNamespace(x=.5, y=.5, z=0.0) for _ in range(21)]
        points[5] = SimpleNamespace(x=.6, y=.5, z=0.0)
        points[9] = SimpleNamespace(x=.5, y=.6, z=0.0)
        points[17] = SimpleNamespace(x=.4, y=.5, z=0.0)
        prediction = SimpleNamespace(instant_label="A", stable_label="A",
                                     stable_confidence=.95, features=[0.0] * 63)
        observed = camera_test.observe_hand(points, prediction, (480, 640, 3))
        self.assertEqual(observed.palm_center_px, (320.0, 252.0))

    def test_cli_default_and_push_backup_check_only_never_open_camera(self):
        self.assertEqual(camera_test.parse_args([]).confirmation_mode, "down")
        self.assertEqual(camera_test.parse_args(["--confirmation-mode", "push"]).confirmation_mode, "push")
        adjusted = camera_test.parse_args(["--down-ratio", "0.40", "--down-smoothing-alpha", "0.80"])
        self.assertEqual(adjusted.down_down_ratio, 0.40)
        self.assertEqual(adjusted.down_smoothing_alpha, 0.80)
        with patch.object(camera_test.cv2, "VideoCapture", side_effect=AssertionError("camera opened")):
            for mode in ("down", "push"):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(camera_test.main(["--confirmation-mode", mode, "--check-only"]), 0)
                self.assertEqual(json.loads(output.getvalue())["confirmation_mode"], mode)


if __name__ == "__main__":
    unittest.main()
