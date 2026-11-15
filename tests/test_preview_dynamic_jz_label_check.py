"""No-video, no-MediaPipe tests for four J/Z full-video preview plans."""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from tools import preview_dynamic_jz_label_check as preview


class DynamicJzLabelPreviewTests(unittest.TestCase):
    def test_exact_targets_and_manifest_bounds(self):
        plan = preview.preflight()
        self.assertEqual([item["relative"] for item in plan], list(preview.TARGETS))
        self.assertEqual([item["frame_count"] for item in plan], [89, 90, 89, 89])
        self.assertEqual([item["status"] for item in plan],
                         ["OK", "REVIEW_REQUIRED", "OK", "REVIEW_REQUIRED"])
        self.assertEqual(plan[0]["bounds"], (6, 19, 3, 22))
        self.assertEqual(plan[2]["bounds"], (11, 30, 8, 33))
        self.assertEqual(plan[1]["bounds"], (None,) * 4)
        self.assertEqual(plan[3]["bounds"], (None,) * 4)
        self.assertTrue(all(item["fps"] == 30.0 for item in plan))

    def test_region_marker_respects_manifest_only(self):
        ok, review = preview.preflight()[:2]
        self.assertEqual(preview.frame_region(2, ok), "OUTSIDE_PROPOSAL")
        self.assertEqual(preview.frame_region(3, ok), "MARGIN")
        self.assertEqual(preview.frame_region(6, ok), "ACTIVE_PROPOSAL")
        self.assertEqual(preview.frame_region(19, ok), "ACTIVE_PROPOSAL")
        self.assertEqual(preview.frame_region(22, ok), "MARGIN")
        self.assertEqual(preview.frame_region(23, ok), "OUTSIDE_PROPOSAL")
        self.assertEqual(preview.frame_region(10, review), "REVIEW_REQUIRED")

    def test_footer_preserves_source_pixels(self):
        item = preview.preflight()[0]
        frame = np.arange(60 * 100 * 3, dtype=np.uint8).reshape(60, 100, 3)
        rendered = preview.render_frame(frame, 6, item)
        self.assertEqual(rendered.shape, (60 + preview.FOOTER_HEIGHT, 100, 3))
        np.testing.assert_array_equal(rendered[:60], frame)

    def test_check_only_never_opens_video_or_writes(self):
        with patch("cv2.VideoCapture", side_effect=AssertionError("video opened")):
            with patch.object(preview, "write_video", side_effect=AssertionError("video written")):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(preview.main(["--check-only"]), 0)
        self.assertFalse(preview.OUTPUT_DIR.exists())


if __name__ == "__main__":
    unittest.main()
