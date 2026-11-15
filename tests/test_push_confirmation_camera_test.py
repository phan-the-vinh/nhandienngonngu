"""Pure state-machine tests; no camera or training."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.inference import push_confirmation_camera_test as push


def observation(scale=100.0, width=70.0, label="A", wrist=(100.0, 100.0),
                confidence=.95, shape=None):
    return push.HandObservation(label, confidence, scale, width, wrist,
                                tuple([0.0] * 63 if shape is None else shape))


class PushConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.machine = push.PushConfirmation(push.PushSettings(
            hold_frames=3, push_frames=2, return_frames=2))

    def ready(self):
        for _ in range(3):
            self.assertIsNone(self.machine.update(observation()))
        self.assertEqual(self.machine.state, "READY")

    def test_one_character_per_push_and_return_required(self):
        self.ready()
        self.assertIsNone(self.machine.update(observation(125, 86)))
        self.assertEqual(self.machine.state, "READY")
        self.assertIsNone(self.machine.update(observation(125, 86)))
        self.assertEqual(self.machine.state, "PUSH_DETECTED")
        self.assertEqual(self.machine.update(observation(125, 86)), "A")
        self.assertEqual(self.machine.state, "CONFIRMED")
        for _ in range(10):
            self.assertIsNone(self.machine.update(observation(125, 86)))
        self.assertEqual(self.machine.text, "A")
        self.assertEqual(self.machine.state, "WAIT_RETURN")
        self.machine.update(None)
        self.assertEqual(self.machine.state, "WAIT_RETURN")
        self.machine.update(observation(103, 71))
        self.machine.update(observation(103, 71))
        self.assertEqual(self.machine.state, "IDLE")
        self.ready()
        for _ in range(2):
            self.machine.update(observation(125, 86))
        self.assertEqual(self.machine.update(observation(125, 86)), "A")
        self.assertEqual(self.machine.text, "AA")

    def test_no_confirm_without_stable_letter_or_with_lateral_motion(self):
        for _ in range(4):
            self.machine.update(observation(label=None, confidence=0))
        self.assertEqual(self.machine.state, "IDLE")
        self.ready()
        for _ in range(4):
            self.machine.update(observation(130, 90, wrist=(300, 100)))
        self.assertEqual(self.machine.text, "")

    def test_shape_change_and_label_change_cancel_ready(self):
        self.ready()
        self.machine.update(observation(125, 86, shape=[1.0] * 63))
        self.assertEqual(self.machine.state, "IDLE")
        self.ready()
        self.machine.update(observation(125, 86, label="B"))
        self.assertEqual(self.machine.state, "IDLE")

    def test_lateral_gesture_cannot_arm(self):
        self.machine.update(observation(wrist=(100, 100)))
        self.machine.update(observation(wrist=(150, 100)))
        self.machine.update(observation(wrist=(200, 100)))
        self.assertEqual(self.machine.state, "IDLE")
        self.assertEqual(self.machine.text, "")

    def test_geometry_uses_aspect_corrected_xy_not_absolute_z(self):
        points = [SimpleNamespace(x=.5, y=.5, z=999.0) for _ in range(21)]
        points[5] = SimpleNamespace(x=.6, y=.5, z=-999.0)
        points[9] = SimpleNamespace(x=.5, y=.6, z=0.0)
        points[17] = SimpleNamespace(x=.4, y=.5, z=0.0)
        prediction = SimpleNamespace(instant_label="A", stable_label="A",
                                     stable_confidence=.95, features=[0.0] * 63)
        result = push.observe_hand(points, prediction, (480, 640, 3))
        self.assertAlmostEqual(result.scale_px, 64.0)
        self.assertAlmostEqual(result.width_px, 128.0)

    def test_dynamic_guard_never_commits_and_cannot_unlock_wait_return(self):
        self.ready()
        self.machine.update(observation(125, 86), dynamic_active=True)
        self.assertEqual(self.machine.state, "IDLE")
        self.assertEqual(self.machine.text, "")
        self.ready()
        for _ in range(3):
            self.machine.update(observation(125, 86))
        self.machine.recalibrate()
        self.machine.update(observation(125, 86), dynamic_active=True)
        self.assertEqual(self.machine.state, "WAIT_RETURN")
        self.assertEqual(self.machine.text, "A")

    def test_check_only_never_opens_camera(self):
        with patch.object(push.cv2, "VideoCapture", side_effect=AssertionError("camera opened")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(push.main(["--check-only"]), 0)
        result = json.loads(output.getvalue())
        self.assertFalse(result["camera_opened"])
        self.assertEqual(result["features"], 63)


if __name__ == "__main__":
    unittest.main()
