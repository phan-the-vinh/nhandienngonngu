"""Synthetic/read-only checks for WLASL contact-sheet range policy."""

import contextlib
import io
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np

from tools import preview_wlasl_other_gesture as preview


class WlaslPreviewTests(unittest.TestCase):
    def test_25fps_metadata_indices_are_one_based_and_inclusive(self):
        start, end, mode, review, reason = preview.choose_range(11, 50, 100, 25.0, 4.0)
        self.assertEqual((start, end), (10, 49))
        self.assertEqual(mode, "METADATA_RANGE_PREVIEW")
        self.assertFalse(review)
        self.assertIn("METADATA_RANGE", reason)

    def test_different_download_fps_is_time_mapped(self):
        start, end, mode, review, reason = preview.choose_range(
            3121, 3246, 9806, 29.97002997002997, 327.2,
        )
        self.assertNotEqual(start, 3120)
        self.assertEqual((start, end), (3740, 3890))
        self.assertEqual(mode, "METADATA_RANGE_PREVIEW")
        self.assertFalse(review)
        self.assertIn("VERIFY_FPS_MAPPING", reason)

    def test_unknown_end_short_and_long_are_distinct(self):
        short = preview.choose_range(1, -1, 90, 30.0, 3.0)
        self.assertEqual(short[:4], (0, 89, "FULL_CLIP_PREVIEW", False))
        long = preview.choose_range(1, -1, 2000, 30.0, 66.7)
        self.assertEqual(long[:4], (0, 1999, "FULL_CLIP_NAVIGATION_ONLY", True))
        self.assertIn("RANGE_REVIEW_REQUIRED", long[4])

    def test_sample_sixteen_ordered_original_indices(self):
        indices = preview.evenly_spaced_indices(10, 49)
        self.assertEqual(len(indices), 16)
        self.assertEqual((indices[0], indices[-1]), (10, 49))
        self.assertEqual(indices, sorted(set(indices)))

    def test_plan_covers_all_downloaded_videos_without_decoding(self):
        with patch.object(preview.cv2, "VideoCapture", side_effect=AssertionError("decode")):
            plan = preview.load_plan()
        self.assertEqual(len(plan), 121)
        self.assertEqual(sum(item["preview_mode"] == "METADATA_RANGE_PREVIEW" for item in plan), 9)
        self.assertEqual(sum(item["preview_mode"] == "FULL_CLIP_PREVIEW" for item in plan), 112)
        self.assertTrue(all(len(item["indices"]) == 16 for item in plan))

    def test_check_only_creates_no_review_output(self):
        output = io.StringIO()
        with (patch.object(sys, "argv", ["preview_wlasl_other_gesture.py", "--check-only"]),
              patch.object(preview, "write_preview", side_effect=AssertionError("preview")),
              patch("builtins.input", side_effect=AssertionError("prompt")),
              contextlib.redirect_stdout(output)):
            preview.main()
        self.assertEqual(json.loads(output.getvalue())["videos"], 121)
        self.assertFalse(preview.OUTPUT.exists())

    def test_synthetic_contact_sheet_has_sixteen_tiles(self):
        item = {
            "video_id": "SYNTH", "gloss": "example", "signer_id": "1",
            "source": "synthetic", "metadata_frame_start": 1,
            "metadata_frame_end": -1, "preview_start": 0,
            "preview_end": 29, "preview_mode": "FULL_CLIP_PREVIEW",
            "duration_seconds": 1.0, "local_fps": 30.0,
            "review_reason": "FULL_CLIP_PREVIEW",
        }
        frames = [(index, np.zeros((40, 60, 3), dtype=np.uint8)) for index in range(16)]
        sheet = preview.make_sheet(item, frames)
        self.assertEqual(sheet.shape, (966, 1280, 3))


if __name__ == "__main__":
    unittest.main()
