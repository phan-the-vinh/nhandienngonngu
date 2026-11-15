"""Synthetic tests only: no real pilot AVI is decoded."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tools import audit_ipn_pilot_videos as pilot


class FakeCapture:
    def __init__(self, count):
        self.count = count
        self.position = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return self.count
        if property_id == cv2.CAP_PROP_FPS:
            return 30.0
        if property_id == cv2.CAP_PROP_POS_FRAMES:
            return self.position
        return 0.0

    def read(self):
        if self.position >= self.count:
            return False, None
        self.position += 1
        return True, np.zeros((4, 5, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class IpNPilotAuditTests(unittest.TestCase):
    def test_one_based_inclusive_to_zero_based(self):
        self.assertEqual(pilot.frame_range_0based(1, 16, 16), (0, 15))
        self.assertEqual(pilot.frame_range_0based(1544, 1818, 275), (1543, 1817))

    def test_invalid_annotation_range_rejected(self):
        for values in ((0, 16, 17), (3, 2, 0), (1, 16, 15)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                pilot.frame_range_0based(*values)

    def test_preflight_is_metadata_only(self):
        with patch.object(pilot, "decode_video", side_effect=AssertionError("video was opened")):
            checked = pilot.preflight()
        self.assertEqual(len(checked["videos"]), 8)
        self.assertEqual(len(checked["segments"]), 24)

    def test_segment_in_bounds_pass(self):
        segment = {"video_id": "a", "label": "D0X", "start": 1, "end": 16,
                   "frames": 16, "start_zero": 0, "end_zero": 15}
        video = {"decoded_frames": 16, "video_status": "PASS"}
        result = pilot.segment_result(segment, video)
        self.assertEqual(result["segment_status"], "PASS")
        self.assertEqual(result["opencv_end_0based"], 15)

    def test_out_of_bounds_fails_and_review_propagates(self):
        segment = {"video_id": "a", "label": "D0X", "start": 1, "end": 16,
                   "frames": 16, "start_zero": 0, "end_zero": 15}
        self.assertEqual(pilot.segment_result(segment, {"decoded_frames": 15,
                                                        "video_status": "REVIEW"})["segment_status"], "FAIL")
        self.assertEqual(pilot.segment_result(segment, {"decoded_frames": 16,
                                                        "video_status": "REVIEW"})["segment_status"], "REVIEW")

    def test_synthetic_full_decode_passes(self):
        capture = FakeCapture(16)
        path = pilot.VIDEOS / "4CM11_1_R_#13.avi"
        with patch.object(cv2, "VideoCapture", return_value=capture):
            row = pilot.decode_video(path, 16)
        self.assertTrue(capture.released)
        self.assertEqual(row["decoded_frames"], 16)
        self.assertEqual(row["video_status"], "PASS")

    def test_synthetic_early_stop_requires_review(self):
        capture = FakeCapture(15)
        path = pilot.VIDEOS / "4CM11_5_L_#10.avi"
        with patch.object(cv2, "VideoCapture", return_value=capture):
            row = pilot.decode_video(path, 16)
        self.assertTrue(capture.released)
        self.assertTrue(row["special_large_video"])
        self.assertEqual(row["video_status"], "REVIEW")


if __name__ == "__main__":
    unittest.main()
