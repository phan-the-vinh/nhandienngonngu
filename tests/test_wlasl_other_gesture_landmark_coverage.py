"""Offline tests for WLASL OTHER_GESTURE coverage preparation."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


TOOL = Path(__file__).resolve().parents[1] / "tools" / "audit_wlasl_other_gesture_landmark_coverage.py"
SPEC = importlib.util.spec_from_file_location("wlasl_coverage", TOOL)
coverage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coverage)


class WlaslOtherGestureCoverageTests(unittest.TestCase):
    def test_real_preflight_accepts_only_103_reviewed_segments(self):
        plan = coverage.preflight()
        self.assertEqual(len(plan), 103)
        self.assertEqual(len({item["review"]["video_id"] for item in plan}), 103)
        self.assertTrue(all(item["review"]["review_status"] == "ACCEPT_OTHER_GESTURE" for item in plan))
        self.assertTrue(all(item["end"] - item["start"] + 1 >= 16 for item in plan))

    def test_source_rights_inventory_matches_accepted_provenance(self):
        rights = json.loads((coverage.EXPERIMENT / "metadata" /
                             "wlasl_other_gesture_source_rights_audit_v1.json").read_text(encoding="utf-8"))
        self.assertFalse(rights["training_eligible_from_rights_audit"])
        self.assertEqual(
            {item["source"]: item["accepted_videos"] for item in rights["sources"]},
            coverage.EXPECTED_SOURCES,
        )
        self.assertTrue(all(item["source_rights_status"] == "UNVERIFIED" for item in rights["sources"]))

    def test_check_only_creates_no_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "coverage.csv"
            summary = Path(directory) / "summary.json"
            with patch.object(coverage, "REPORT", report), patch.object(coverage, "SUMMARY", summary):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(coverage.main(["--check-only"]), 0)
            self.assertFalse(report.exists())
            self.assertFalse(summary.exists())

    def test_hand_identity_cannot_be_auto_confirmed(self):
        self.assertEqual(coverage.hand_identity_status(0, 0, 0), "HAND_IDENTITY_UNVERIFIED")
        self.assertEqual(coverage.hand_identity_status(1, 0, 0), "REVIEW_REQUIRED_HAND_IDENTITY")
        self.assertEqual(coverage.hand_identity_status(0, 1, 0), "REVIEW_REQUIRED_HAND_IDENTITY")
        self.assertEqual(coverage.hand_identity_status(0, 0, 1), "REVIEW_REQUIRED_HAND_IDENTITY")

    def test_audit_video_only_detects_within_reviewed_segment_and_flags_two_hands(self):
        import cv2
        import numpy as np

        class FakeCapture:
            def __init__(self):
                self.index = 0
                self.released = False

            def isOpened(self):
                return True

            def read(self):
                self.index += 1
                return True, np.zeros((8, 8, 3), dtype=np.uint8)

            def release(self):
                self.released = True

        class FakeLandmarker:
            def __init__(self):
                self.calls = 0

            def detect(self, _image):
                self.calls += 1
                hand = [SimpleNamespace(x=0.1, y=0.2, z=0.0) for _ in range(21)]
                category = SimpleNamespace(category_name="Right")
                return SimpleNamespace(
                    hand_landmarks=[hand, hand],
                    handedness=[[category], [category]],
                )

        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "WLASL_01206.mp4"
            video.write_bytes(b"synthetic")
            tech = {
                "actual_sha256": coverage.sha256(video), "url": "https://example.org/video",
                "split": "train", "decoded_frame_count": "4",
            }
            review = {
                "video_id": "01206", "gloss": "OTHER", "signer_id": "1", "source": "source",
                "local_path": "synthetic", "review_status": "ACCEPT_OTHER_GESTURE",
                "manual_reason": "human", "preview_mode": "METADATA_RANGE_PREVIEW",
                "metadata_frame_start": "2", "metadata_frame_end": "3",
            }
            capture = FakeCapture()
            model = FakeLandmarker()
            with patch.object(cv2, "VideoCapture", return_value=capture):
                row = coverage.audit_video(
                    {"review": review, "technical": tech, "path": video, "start": 1, "end": 2},
                    model,
                )
            self.assertEqual(model.calls, 2)
            self.assertEqual(row["segment_decoded_frames"], 2)
            self.assertEqual(row["valid_21_landmark_frames"], 2)
            self.assertEqual(row["multiple_hand_frames"], 2)
            self.assertEqual(row["single_valid_hand_frames"], 0)
            self.assertEqual(row["hand_identity_status"], "REVIEW_REQUIRED_HAND_IDENTITY")
            self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
