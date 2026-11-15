"""Synthetic-only tests: no dataset videos, MediaPipe task, or camera opened."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tools import audit_dynamic_landmark_coverage as coverage


def valid_hand():
    return [SimpleNamespace(x=0.2, y=0.3, z=0.0) for _ in range(21)]


class FakeCapture:
    def __init__(self, frame_count):
        self.frame_count = frame_count
        self.read_count = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.read_count >= self.frame_count:
            return False, None
        self.read_count += 1
        return True, np.zeros((4, 5, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakeLandmarker:
    def __init__(self, missing_indices=(), error_indices=()):
        self.calls = 0
        self.missing_indices = set(missing_indices)
        self.error_indices = set(error_indices)

    def detect(self, _image):
        index = self.calls
        self.calls += 1
        if index in self.error_indices:
            raise RuntimeError("synthetic detection failure")
        return SimpleNamespace(hand_landmarks=[] if index in self.missing_indices else [valid_hand()])


class DynamicLandmarkCoverageTests(unittest.TestCase):
    def test_every_decoded_frame_is_detected_and_missing_frames_counted(self):
        import cv2
        import mediapipe as mp

        capture = FakeCapture(20)
        landmarker = FakeLandmarker({3, 9, 12})
        path = coverage.ROOT / "J" / "fake.avi"
        with patch.object(cv2, "VideoCapture", return_value=capture), \
                patch.object(mp, "Image", return_value=object()):
            row = coverage.audit_video(path, "J", landmarker)
        self.assertEqual(landmarker.calls, 20)
        self.assertTrue(capture.released)
        self.assertEqual(row["decoded_frame_count"], 20)
        self.assertEqual(row["valid_landmark_frames"], 17)
        self.assertEqual(row["invalid_landmark_frames"], 3)
        self.assertAlmostEqual(row["landmark_coverage_ratio"], 0.85)
        self.assertTrue(row["has_16_valid_frames"])
        self.assertEqual(row["audit_status"], "COMPLETE")

    def test_discontinuous_valid_frames_are_accepted(self):
        import cv2
        import mediapipe as mp

        capture = FakeCapture(32)
        landmarker = FakeLandmarker(set(range(0, 32, 2)))
        path = coverage.ROOT / "Z" / "fake.avi"
        with patch.object(cv2, "VideoCapture", return_value=capture), \
                patch.object(mp, "Image", return_value=object()):
            row = coverage.audit_video(path, "Z", landmarker)
        self.assertEqual(row["valid_landmark_frames"], 16)
        self.assertTrue(row["has_16_valid_frames"])

    def test_fewer_than_sixteen_valid_frames_are_flagged(self):
        import cv2
        import mediapipe as mp

        capture = FakeCapture(20)
        landmarker = FakeLandmarker(missing_indices={1, 3, 5, 7}, error_indices={9})
        path = coverage.ROOT / "J" / "fake.avi"
        with patch.object(cv2, "VideoCapture", return_value=capture), \
                patch.object(mp, "Image", return_value=object()):
            row = coverage.audit_video(path, "J", landmarker)
        self.assertEqual(row["valid_landmark_frames"], 15)
        self.assertFalse(row["has_16_valid_frames"])
        self.assertEqual(row["invalid_landmark_frames"], 5)
        self.assertEqual(row["audit_status"], "REVIEW_REQUIRED")
        self.assertIn("DETECTION_ERROR:RuntimeError", row["invalid_reason_counts"])

    def test_invalid_count_and_summary_boundary(self):
        rows = [
            {"label": "J", "landmark_coverage_ratio": 0.8,
             "has_16_valid_frames": True, "audit_status": "COMPLETE"},
            {"label": "J", "landmark_coverage_ratio": 0.5,
             "has_16_valid_frames": False, "audit_status": "COMPLETE"},
            {"label": "Z", "landmark_coverage_ratio": "",
             "has_16_valid_frames": False, "audit_status": "ERROR"},
        ]
        summary = coverage.summarize(rows)
        self.assertEqual(summary["status"], "REVIEW_REQUIRED")
        self.assertEqual(summary["per_label"]["J"]["videos_at_least_16_valid_frames"], 1)
        self.assertEqual(summary["per_label"]["J"]["videos_fewer_than_16_valid_frames"], 1)
        self.assertEqual(summary["per_label"]["J"]["coverage_ratio"]["median"], 0.65)
        self.assertEqual(summary["per_label"]["Z"]["video_audit_errors"], 1)
        self.assertIsNone(summary["per_label"]["Z"]["coverage_ratio"]["min"])


if __name__ == "__main__":
    unittest.main()
