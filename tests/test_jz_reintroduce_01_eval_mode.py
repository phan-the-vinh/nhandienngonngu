"""Offline tests for --jz-evaluation-mode in PHASE JZ-REINTRODUCE-01.

Uses synthetic fixtures/fake predictors ONLY to verify telemetry, UI overlay,
and commit-safety contracts; never opens the webcam or claims real ASL accuracy.
"""
import contextlib
import io
import json
import unittest
from types import SimpleNamespace

import numpy as np

from app.inference import unified_abc123_jz_auto_experiment as camera
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from tests.test_dynamic_jz_motion_gate import curved_hand
from tests.test_unified_camera_experiment import FakePredictor, synthetic_hand


class FakeDynamicModel:
    classes_ = np.asarray(["J", "Z"])

    def __init__(self, label="J", prob=0.96):
        self.label = label
        self.prob = prob

    def predict_proba(self, _x):
        if self.label == "J":
            return np.asarray([[self.prob, 1.0 - self.prob]])
        return np.asarray([[1.0 - self.prob, self.prob]])


def mirrored(hand):
    return [SimpleNamespace(x=1.0 - p.x, y=p.y, z=p.z) for p in hand]


class JzEvaluationModeTests(unittest.TestCase):
    def test_check_only_flag_surfaces_jz_evaluation_mode(self):
        stdout_default = io.StringIO()
        with contextlib.redirect_stdout(stdout_default):
            code = camera.main(["--check-only"])
        self.assertEqual(code, 0)
        payload_default = json.loads(stdout_default.getvalue())
        self.assertEqual(payload_default["status"], "PASS")
        self.assertFalse(payload_default["jz_evaluation_mode"])

        stdout_eval = io.StringIO()
        with contextlib.redirect_stdout(stdout_eval):
            code = camera.main(["--check-only", "--jz-evaluation-mode"])
        self.assertEqual(code, 0)
        payload_eval = json.loads(stdout_eval.getvalue())
        self.assertEqual(payload_eval["status"], "PASS")
        self.assertTrue(payload_eval["jz_evaluation_mode"])

    def test_rejected_raw_prediction_preserved_and_never_committed(self):
        letters = FakePredictor("A", 0.0)
        digits = FakePredictor("0", 0.5)
        model = FakeDynamicModel(label="Z", prob=0.95)
        events = []
        coord = UnifiedAutoCoordinator(
            letters, digits, model,
            enable_experimental_jz=True,
            jz_evaluation_mode=True,
            event_sink=events.append,
        )
        # Feed a J-shaped curve while model predicts Z so the validator rejects Z.
        for idx in range(70):
            hand = curved_hand(min(idx, 60), "J")
            coord.update(idx, mirrored(hand), hand, (480, 640, 3), idx / 30.0, raw_hand_count=1)
        still = curved_hand(60, "J")
        for idx in range(70, 95):
            coord.update(idx, mirrored(still), still, (480, 640, 3), idx / 30.0, raw_hand_count=1)

        rec = coord.jz_evaluation_record
        self.assertIsNotNone(rec)
        self.assertEqual(rec["raw_prediction"], "Z")
        self.assertAlmostEqual(rec["probability_Z"], 0.95, places=3)
        self.assertEqual(rec["validator_status"], "REJECTED")
        self.assertNotIn(rec["reason"], ("", "NONE", None))
        self.assertEqual(rec["final_output"], "NONE")
        self.assertFalse(rec["committed"])
        self.assertEqual(coord.composer.text, "")
        self.assertTrue(any(e.get("event") == "JZ_EVALUATION_RECORD" for e in events))

        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        rendered = camera.render(blank, "ABC", still, blank, still, coord, 85 / 30.0)
        self.assertEqual(rendered.shape[0], 480)
        self.assertGreater(rendered.shape[1], 640)
        coord.close()

    def test_valid_j_tracks_accepted_and_commits_in_evaluation_mode(self):
        letters = FakePredictor("A", 0.0)
        digits = FakePredictor("0", 0.5)
        model = FakeDynamicModel(label="J", prob=0.97)
        coord = UnifiedAutoCoordinator(
            letters, digits, model,
            enable_experimental_jz=True,
            jz_evaluation_mode=True,
        )
        idx = 0
        for k in range(80):
            hand = curved_hand(min(k, 60), "J")
            coord.update(idx, mirrored(hand), hand, (480, 640, 3), idx / 30.0, raw_hand_count=1)
            idx += 1
        terminal = curved_hand(60, "J")
        for _ in range(110):
            coord.update(idx, mirrored(terminal), terminal, (480, 640, 3), idx / 30.0, raw_hand_count=1)
            idx += 1
            if coord.composer.text == "J":
                break

        self.assertEqual(coord.composer.text, "J")
        rec = coord.jz_evaluation_record
        self.assertIsNotNone(rec)
        self.assertEqual(rec["raw_prediction"], "J")
        self.assertEqual(rec["validator_status"], "ACCEPTED")
        self.assertTrue(rec["committed"])
        self.assertEqual(rec["final_output"], "J")
        coord.close()

    def test_static_hold_unchanged_with_jz_evaluation_mode(self):
        letters = FakePredictor("A", 0.0)
        digits = FakePredictor("0", 0.5)
        model = FakeDynamicModel(label="J", prob=0.97)
        coord = UnifiedAutoCoordinator(
            letters, digits, model,
            enable_experimental_jz=True,
            jz_evaluation_mode=True,
        )
        hand = synthetic_hand()
        for idx, t in enumerate((0.0, 0.1, 0.2, 3.05)):
            coord.update(idx, hand, hand, (480, 640, 3), t, raw_hand_count=1)
        self.assertEqual(coord.composer.text, "A")
        coord.close()


if __name__ == "__main__":
    unittest.main()
