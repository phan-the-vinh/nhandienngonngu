"""J review preflight tests: metadata only, no video or MediaPipe processing."""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from tools import preview_dynamic_j_remaining_review as preview


class RemainingJReviewTests(unittest.TestCase):
    def test_exact_groups_and_saved_sequence_membership(self):
        groups, records = preview.review_groups()
        self.assertEqual([len(group) for group in groups], [10, 10, 10, 8])
        names = [row["video"] for group in groups for row in group]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(name in records for name in names[:30]))
        self.assertTrue(all(name not in records for name in names[30:]))
        self.assertEqual(names[:3], ["J/281.avi", "J/103.avi", "J/299.avi"])

    def test_preflight_distinguishes_extracted_and_preview_only(self):
        extracted, _ = preview.preflight(1)
        unsequenced, _ = preview.preflight(4)
        self.assertTrue(all(item["selection_kind"] == "EXTRACTED_16" for item in extracted))
        self.assertTrue(all(item["selection_kind"] == "PREVIEW_ONLY_NOT_EXTRACTED"
                            for item in unsequenced))
        self.assertTrue(all(len(item["indices"]) == 16 for item in extracted + unsequenced))
        self.assertTrue(all(item["motif_to_compare"] in {"ARM_SWEEP_U", "AIR_DRAWN_J"}
                            for item in extracted))

    def test_check_only_never_opens_video_or_writes(self):
        with patch("cv2.VideoCapture", side_effect=AssertionError("video opened")):
            with patch.object(preview, "write_one", side_effect=AssertionError("video processed")):
                with patch.object(preview, "h264_encoder", return_value=("ffmpeg", "libx264 available")):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(preview.main(["--check-only", "--batch", "1"]), 0)
        self.assertFalse((preview.OUTPUT_ROOT / "batch_01").exists())


if __name__ == "__main__":
    unittest.main()
