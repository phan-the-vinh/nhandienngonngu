"""Offline routing and HOLD tests; VideoCapture must never be opened here."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from app.inference.letter_inference import Prediction
from app.inference.unified_camera_experiment import (
    main, mode_from_click, prepare_inference_frame, render_mirrored_frame, run_camera,
)
from app.inference.unified_static_coordinator import (
    DigitPredictor, RecognitionCoordinator, load_digit_model,
)
from app.interaction.hold_confirmation import HoldSettings


def synthetic_hand():
    points = [SimpleNamespace(x=.5, y=.5, z=.0) for _ in range(21)]
    points[0] = SimpleNamespace(x=.5, y=.8, z=.0)
    points[5] = SimpleNamespace(x=.4, y=.6, z=.0)
    points[9] = SimpleNamespace(x=.5, y=.55, z=.0)
    points[17] = SimpleNamespace(x=.6, y=.6, z=.0)
    return points


class FakePredictor:
    def __init__(self, label: str, shape: float):
        self.label = label
        self.shape = shape
        self.calls = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def predict_landmarks(self, _hand):
        self.calls += 1
        return Prediction(self.label, .95, self.label, .95,
                          (self.shape,) * 63, (.95, .05))


class UnifiedCameraTests(unittest.TestCase):
    def setUp(self):
        self.letters = FakePredictor("A", 0.0)
        self.digits = FakePredictor("0", .5)
        self.coordinator = RecognitionCoordinator(self.letters, self.digits, HoldSettings())
        self.hand = synthetic_hand()

    def feed(self, moment: float):
        return self.coordinator.process_hand(self.hand, (480, 640, 3), moment)[1]

    def confirm(self, label: str, shape: float, start: float):
        self.letters.label, self.letters.shape = label, shape
        for moment in (start, start + .1, start + .2):
            self.assertIsNone(self.feed(moment))
        self.assertEqual(self.coordinator.hold.state, "HOLDING")
        return self.feed(start + 3.0)

    def test_letter_a_and_no_repeat_after_six_seconds(self):
        self.assertEqual(self.confirm("A", 0.0, 0.0), "A")
        self.assertIsNone(self.feed(6.1))
        self.assertEqual(self.coordinator.composer.text, "A")
        self.assertEqual(self.coordinator.hold.state, "WAIT_RELEASE")

    def test_apple_including_repeated_p(self):
        self.assertEqual(self.confirm("A", 0.0, 0.0), "A")
        self.assertEqual(self.confirm("P", .3, 3.1), "P")
        self.coordinator.missing_hand(6.2)
        self.coordinator.missing_hand(6.9)
        self.assertEqual(self.coordinator.hold.state, "IDLE")
        self.assertEqual(self.confirm("P", .3, 7.0), "P")
        self.assertEqual(self.confirm("L", .6, 10.1), "L")
        self.assertEqual(self.confirm("E", .9, 13.2), "E")
        self.assertEqual(self.coordinator.composer.text, "APPLE")
        self.assertEqual(self.coordinator.hold.text, "APPLE")
        self.assertGreaterEqual(self.letters.resets, 5)

    def test_mode_switch_cancels_hold_and_keeps_one_text_stream(self):
        self.feed(0.0)
        self.feed(.1)
        self.feed(.2)
        self.feed(2.0)
        self.coordinator.switch_mode("123")
        self.assertEqual(self.coordinator.hold.state, "IDLE")
        self.assertEqual(self.coordinator.composer.text, "")
        for moment in (2.1, 2.2, 2.3):
            self.feed(moment)
        self.assertIsNone(self.feed(3.0))
        self.assertEqual(self.feed(5.1), "0")
        self.assertEqual(self.coordinator.composer.text, "0")
        self.assertEqual(self.letters.calls, 4)
        self.assertEqual(self.digits.calls, 5)

    def test_o_and_zero_are_selected_by_mode_not_cross_model_confidence(self):
        self.letters.label = "O"
        self.feed(0.0)
        self.assertEqual(self.letters.calls, 1)
        self.assertEqual(self.digits.calls, 0)
        self.coordinator.switch_mode("123")
        self.feed(.1)
        self.assertEqual(self.letters.calls, 1)
        self.assertEqual(self.digits.calls, 1)
        self.assertEqual(mode_from_click(640 + 60, 30, 640), "ABC")
        self.assertEqual(mode_from_click(640 + 210, 30, 640), "123")
        self.assertEqual(mode_from_click(640 + 210, 65, 640), "JZ_PAUSED")
        self.assertIsNone(mode_from_click(640 + 210, 90, 640))

    def test_jz_paused_mode_never_runs_static_inference_or_hold(self):
        self.coordinator.switch_mode("JZ_PAUSED")
        for moment in (0.0, 1.0, 3.0, 7.0):
            self.coordinator.missing_hand(moment)
        self.assertEqual(self.coordinator.composer.text, "")
        self.assertEqual(self.coordinator.hold.state, "IDLE")
        self.assertEqual(self.letters.calls, 0)
        self.assertEqual(self.digits.calls, 0)
        with self.assertRaises(ValueError):
            self.feed(8.0)

    def test_production_digit_pipeline_accepts_canonical_63_features(self):
        model = load_digit_model()
        predictor = DigitPredictor(model)
        prediction = predictor.predict_landmarks(self.hand)
        self.assertEqual(len(prediction.features), 63)
        self.assertEqual(len(prediction.probabilities), 10)
        self.assertAlmostEqual(sum(prediction.probabilities), 1.0)
        self.assertIn(prediction.instant_label, tuple(str(i) for i in range(10)))

    def test_check_only_validates_all_models_without_camera(self):
        with patch("app.inference.unified_camera_experiment.cv2.VideoCapture",
                   side_effect=AssertionError("camera opened")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["--check-only"]), 0)
        report = json.loads(output.getvalue())
        self.assertFalse(report["camera_opened"])
        self.assertEqual(report["dynamic_feature_shape"], [16, 69])
        self.assertIn("DISABLED", report["dynamic_live_status"])

    def test_display_stays_mirrored_while_inference_orientation_is_preserved(self):
        raw = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        original = raw.copy()
        for mode in ("ABC", "123", "ABC", "JZ_PAUSED"):
            with self.subTest(mode=mode):
                inference = prepare_inference_frame(raw, mode)
                expected_input = raw[:, ::-1] if mode == "ABC" else raw
                np.testing.assert_array_equal(inference, expected_input)
                displayed = render_mirrored_frame(inference, mode)
                np.testing.assert_array_equal(displayed, raw[:, ::-1])
                displayed[:] = 0  # Rendering must not alias raw or inference.
                np.testing.assert_array_equal(inference, expected_input)
                np.testing.assert_array_equal(raw, original)

    def test_landmark_overlay_is_mirrored_without_changing_inference_or_hand(self):
        raw = np.zeros((80, 100, 3), dtype=np.uint8)
        hand = [SimpleNamespace(x=.2, y=.4, z=.0) for _ in range(21)]
        for mode, display_x in (("ABC", 20), ("123", 79)):
            with self.subTest(mode=mode):
                inference = prepare_inference_frame(raw, mode)
                before = inference.copy()
                displayed = render_mirrored_frame(inference, mode, hand)
                np.testing.assert_array_equal(displayed[32, display_x], [0, 0, 255])
                np.testing.assert_array_equal(inference, before)
                self.assertTrue(all(point.x == .2 for point in hand))
        self.assertFalse(raw.any())

    def test_camera_loop_mode_switch_preserves_inputs_and_mirrored_display(self):
        # Fake camera/task only: exercise the real loop and mouse routing offline.
        raw = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        camera = MagicMock()
        camera.read.side_effect = [(True, raw.copy()) for _ in range(3)]
        task = MagicMock()
        task.__enter__.return_value = task
        seen_inputs, seen_displays, callbacks = [], [], []

        def detect(image, _timestamp):
            seen_inputs.append(image.numpy_view().copy())
            return SimpleNamespace(hand_landmarks=[])

        def next_key(_delay):
            if len(seen_inputs) == 1:
                callbacks[0](cv2.EVENT_LBUTTONDOWN, 8 + 210, 30, 0, None)
            elif len(seen_inputs) == 2:
                callbacks[0](cv2.EVENT_LBUTTONDOWN, 8 + 60, 30, 0, None)
            return ord("q") if len(seen_inputs) == 3 else 0

        task.detect_for_video.side_effect = detect
        prefix = "app.inference.unified_camera_experiment."
        with (patch(prefix + "cv2.VideoCapture", return_value=camera) as capture,
              patch(prefix + "create_hand_landmarker", return_value=task) as create_task,
              patch(prefix + "cv2.namedWindow"),
              patch(prefix + "cv2.setMouseCallback", side_effect=lambda _w, cb: callbacks.append(cb)),
              patch(prefix + "cv2.imshow", side_effect=lambda _w, frame: seen_displays.append(frame.copy())),
              patch(prefix + "cv2.waitKey", side_effect=next_key),
              patch(prefix + "cv2.destroyAllWindows"),
              patch(prefix + "_panel", side_effect=lambda frame, *_args: frame),
              patch(prefix + "_draw_mode_controls")):
            self.assertEqual(run_camera(self.coordinator, 0), 0)
        capture.assert_called_once_with(0)
        create_task.assert_called_once()
        camera.release.assert_called_once()
        for actual, mode in zip(seen_inputs, ("ABC", "123", "ABC")):
            expected = raw[:, ::-1] if mode == "ABC" else raw
            np.testing.assert_array_equal(actual, cv2.cvtColor(expected, cv2.COLOR_BGR2RGB))
        for frame in seen_displays:
            np.testing.assert_array_equal(frame, raw[:, ::-1])


if __name__ == "__main__":
    unittest.main()
