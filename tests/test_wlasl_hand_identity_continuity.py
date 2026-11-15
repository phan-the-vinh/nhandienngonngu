"""Synthetic, no-MediaPipe tests for WLASL hand-continuity audit policy."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

import numpy as np


TOOL = Path(__file__).resolve().parents[1] / "tools" / "audit_wlasl_hand_identity_continuity.py"
SPEC = importlib.util.spec_from_file_location("wlasl_identity_audit", TOOL)
identity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(identity)


def hand(x: float, handedness: str = "Right") -> dict:
    points = np.zeros((21, 2), dtype=np.float64)
    for index in range(21):
        points[index] = (x + index % 5 * 5, index // 5 * 5)
    return {"points_xy": points, "scale": identity.palm_scale(points),
            "handedness": handedness}


def frame(index: int, hands: list[dict] | None = None, raw_count: int | None = None) -> dict:
    hands = hands or []
    return {"index": index, "hands": hands,
            "raw_hand_count": len(hands) if raw_count is None else raw_count,
            "invalid_hands": 0, "detection_error": ""}


class WlaslHandContinuityTests(unittest.TestCase):
    def test_check_only_validates_103_without_creating_outputs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(identity.main(["--check-only"]), 0)
        self.assertFalse(identity.REPORT.exists())
        self.assertFalse(identity.FRAMES.exists())
        self.assertFalse(identity.SUMMARY.exists())

    def test_pass_samples_exact_jz_valid_frame_policy(self):
        frames = [frame(i, [hand(i * 0.1)]) for i in range(20)]
        result = identity.classify_frames(frames, prior_valid=20)
        self.assertEqual(result["continuity_decision"], "PASS")
        self.assertEqual(result["proposed_original_frame_indices"],
                         identity.sample_valid_frame_indices(list(range(20))))
        self.assertEqual(len(result["proposed_original_frame_indices"]), 16)
        self.assertTrue(result["identity_not_proven"])

    def test_two_hands_are_reviewed_and_never_selected(self):
        frames = [frame(i, [hand(i * 0.1)]) for i in range(20)]
        frames[9] = frame(9, [hand(1.0), hand(30.0)])
        result = identity.classify_frames(frames, prior_valid=20)
        self.assertEqual(result["continuity_decision"], "REVIEW")
        self.assertIn("TWO_HAND_FRAMES", result["decision_reason"])
        self.assertEqual(result["proposed_original_frame_indices"], [])

    def test_internal_missing_detection_is_review(self):
        frames = [frame(i, [hand(i * 0.1)]) for i in range(21)]
        frames[10] = frame(10)
        result = identity.classify_frames(frames, prior_valid=20)
        self.assertEqual(result["continuity_decision"], "REVIEW")
        self.assertEqual(result["internal_detection_gaps"], 1)

    def test_handedness_alone_does_not_establish_identity(self):
        frames = [frame(i, [hand(i * 0.1)]) for i in range(20)]
        frames[12] = frame(12, [hand(1.2, "Left")])
        result = identity.classify_frames(frames, prior_valid=20)
        self.assertEqual(result["continuity_decision"], "REVIEW")
        self.assertGreater(result["single_hand_handedness_changes"], 0)

    def test_spatial_landmark_jump_is_review_even_with_constant_handedness(self):
        frames = [frame(i, [hand(i * 0.1)]) for i in range(20)]
        frames[12] = frame(12, [hand(200)])
        result = identity.classify_frames(frames, prior_valid=20)
        self.assertEqual(result["continuity_decision"], "REVIEW")
        self.assertIn("ABRUPT_POSITION", result["decision_reason"])

    def test_fewer_than_16_valid_frames_is_fail(self):
        frames = [frame(i, [hand(i * 0.1)]) if i < 14 else frame(i)
                  for i in range(20)]
        result = identity.classify_frames(frames, prior_valid=14)
        self.assertEqual(result["continuity_decision"], "FAIL")
        self.assertFalse(result["can_sample_same_hand_16"])

    def test_no_hand_before_and_after_one_run_is_allowed(self):
        frames = [frame(i) for i in range(3)] + [frame(i, [hand(i * 0.1)]) for i in range(3, 19)]
        frames += [frame(i) for i in range(19, 21)]
        result = identity.classify_frames(frames, prior_valid=16)
        self.assertEqual(result["continuity_decision"], "PASS")
        self.assertEqual(result["proposed_original_frame_indices"], list(range(3, 19)))


if __name__ == "__main__":
    unittest.main()
