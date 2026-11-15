"""No-video, no-MediaPipe tests for the IPN recovery-preview tool."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

TOOL = Path(__file__).resolve().parents[1] / "tools/preview_ipn_recovery_segments.py"
SPEC = importlib.util.spec_from_file_location("ipn_recovery_preview", TOOL)
preview = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preview)


class IpnRecoveryPreviewTests(unittest.TestCase):
    def test_preflight_selects_exact_three_review_segments(self):
        plan = preview.preflight()
        self.assertEqual(len(plan), 3)
        self.assertEqual([(p["video_id"], p["label"], p["start_1"], p["end_1"])
                          for p in plan], list(preview.TARGETS))
        self.assertEqual([len(p["frames"]) for p in plan], [46, 71, 62])
        self.assertTrue(all(p["audit"]["technical_status"] == "REVIEW" for p in plan))

    def test_known_warning_frames_and_candidate_ranges(self):
        plan = preview.preflight()
        first, second, third = [p["details"] for p in plan]
        self.assertEqual(first["missing_hand_ranges"], [])
        self.assertEqual([x["frame_0based"] for x in first["handedness_changes"]], [2702, 2703])
        self.assertEqual([(x["start_0based"], x["end_0based"]) for x in
                          first["candidate_contiguous_ranges"]], [(2683, 2701), (2703, 2728)])
        self.assertEqual(second["missing_hand_ranges"],
                         [{"start_0based": 3217, "end_0based": 3217, "frame_count": 1}])
        self.assertEqual([(x["start_0based"], x["end_0based"]) for x in
                          second["candidate_contiguous_ranges"]], [(3166, 3216), (3218, 3236)])
        self.assertEqual(third["missing_hand_ranges"],
                         [{"start_0based": 3357, "end_0based": 3360, "frame_count": 4}])
        self.assertEqual([(x["start_0based"], x["end_0based"]) for x in
                          third["candidate_contiguous_ranges"]], [(3324, 3356), (3361, 3385)])

    def test_check_only_never_opens_video_or_writes(self):
        before = [(p, p.exists()) for p in (preview.OUTPUT_METADATA,
                                             *(preview.OUTPUT_DIR.glob("*.avi") if preview.OUTPUT_DIR.exists() else []))]
        with patch("cv2.VideoCapture", side_effect=AssertionError("video opened")):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(preview.main(["--check-only"]), 0)
        self.assertEqual(before, [(p, p.exists()) for p, _ in before])

    def test_footer_does_not_modify_source_frame_region(self):
        frame = np.arange(60 * 100 * 3, dtype=np.uint8).reshape(60, 100, 3)
        details = {"candidate_contiguous_ranges": [
            {"start_0based": 5, "end_0based": 25, "frame_count": 21}],
            "handedness_changes": []}
        diagnostic = {"raw_hand_count": 1}
        result = preview.render_frame(frame, 10, details, diagnostic)
        self.assertEqual(result.shape, (60 + preview.FOOTER_HEIGHT, 100, 3))
        np.testing.assert_array_equal(result[:60], frame)

    def test_no_gap_joined_into_candidate(self):
        def frame(index, count, label="Right"):
            return {"original_frame_index": index, "raw_hand_count": count,
                    "valid_hand_count": count, "handedness_observed": [label] if count else []}
        frames = [frame(i, 1) for i in range(20)] + [frame(20, 0)] + [frame(i, 1) for i in range(21, 41)]
        result = preview.diagnostic_plan(frames)
        self.assertEqual([(r["start_0based"], r["end_0based"]) for r in
                          result["candidate_contiguous_ranges"]], [(0, 19), (21, 40)])


if __name__ == "__main__":
    unittest.main()
