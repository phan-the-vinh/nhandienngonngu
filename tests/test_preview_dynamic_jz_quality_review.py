"""Metadata-only tests; never decode J/Z video or run MediaPipe."""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from tools import preview_dynamic_jz_quality_review as preview


class QualityPreviewTests(unittest.TestCase):
    def test_exact_nine_saved_sequences(self):
        plan = preview.preflight()
        self.assertEqual([item["relative"] for item in plan], list(preview.TARGETS))
        self.assertEqual(len(plan), 9)
        self.assertTrue(all(len(item["indices"]) == 16 for item in plan))
        self.assertEqual(plan[0]["indices"],
                         [45, 46, 47, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 86, 87, 88])
        self.assertEqual(plan[3]["fps"], 20.0)
        self.assertTrue(all(item["no_hand_intervals"] is None for item in plan))

    def test_indices_reject_invalid_order_and_bounds(self):
        with self.assertRaises(ValueError):
            preview.validate_indices([0] * 16, 89)
        with self.assertRaises(ValueError):
            preview.validate_indices(list(range(15)) + [90], 89)

    def test_footer_does_not_modify_source_pixels(self):
        item = preview.preflight()[0]
        frame = np.arange(60 * 120 * 3, dtype=np.uint8).reshape(60, 120, 3)
        result = preview.render_frame(frame, item["indices"][0], item)
        self.assertEqual(result.shape, (60 + preview.FOOTER_HEIGHT, 120, 3))
        np.testing.assert_array_equal(result[:60], frame)

    def test_check_only_opens_no_video_and_writes_nothing(self):
        with patch("cv2.VideoCapture", side_effect=AssertionError("video opened")):
            with patch.object(preview, "write_one", side_effect=AssertionError("video processed")):
                with patch.object(preview, "h264_encoder", return_value=("ffmpeg", "libx264 available")):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(preview.main(["--check-only"]), 0)
        self.assertFalse(preview.OUTPUT_DIR.exists())


if __name__ == "__main__":
    unittest.main()
