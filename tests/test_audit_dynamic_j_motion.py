"""J-only saved-data audit and no-video preview-plan checks."""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from tools import audit_dynamic_j_motion as audit
from tools import preview_dynamic_j_exclusion_candidates as preview


class DynamicJMotionTests(unittest.TestCase):
    def test_inventory_and_user_decisions(self):
        extraction, sequences, quality, coverage = audit.load_j()
        self.assertEqual((len(extraction), len(sequences), len(quality), len(coverage)),
                         (316, 308, 308, 316))
        rows, summary = audit.analyze()
        self.assertEqual(len(rows), 316)
        self.assertEqual(summary["classification_counts"], {
            "EXCLUDE_CONFIRMED": 4, "EXCLUDE_CANDIDATE": 36,
            "NEEDS_VISUAL_REVIEW": 38, "NO_MATCH_FOUND": 238,
        })
        by_video = {row["video"]: row for row in rows}
        self.assertTrue(all(by_video[name]["classification"] == "EXCLUDE_CONFIRMED"
                            for name in audit.CONFIRMED))
        self.assertEqual(by_video["J/209.avi"]["classification"], "EXCLUDE_CANDIDATE")
        self.assertEqual(by_video["J/124.avi"]["classification"], "NEEDS_VISUAL_REVIEW")
        self.assertEqual(by_video["J/229.avi"]["classification"], "NO_MATCH_FOUND")
        self.assertEqual(by_video["J/241.avi"]["classification"], "NO_MATCH_FOUND")

    def test_check_only_never_processes_video_or_writes(self):
        with patch("cv2.VideoCapture", side_effect=AssertionError("video opened")):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(["--check-only"]), 0)

    def test_candidate_preview_batch_uses_saved_indices(self):
        plan, output = preview.preflight(1)
        self.assertEqual(len(plan), 9)
        self.assertEqual(plan[0]["relative"], "J/209.avi")
        self.assertTrue(all(len(item["indices"]) == 16 for item in plan))
        self.assertFalse(output.exists())
        with patch("cv2.VideoCapture", side_effect=AssertionError("video opened")):
            with patch.object(preview, "h264_encoder", return_value=("ffmpeg", "libx264 available")):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(preview.main(["--check-only", "--batch", "1"]), 0)
        self.assertFalse(output.exists())

    def test_all_candidate_batches_and_out_of_range(self):
        self.assertEqual([len(preview.preflight(batch)[0]) for batch in (1, 2, 3, 4)],
                         [9, 9, 9, 9])
        with self.assertRaises(ValueError):
            preview.preflight(5)


if __name__ == "__main__":
    unittest.main()
