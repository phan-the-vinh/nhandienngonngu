"""Metadata and synthetic-only tests; no video decode or MediaPipe run."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

TOOL = Path(__file__).resolve().parents[1] / "tools/audit_ipn_handedness_event_2702.py"
SPEC = importlib.util.spec_from_file_location("ipn_event_2702", TOOL)
event = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(event)


class HandednessEventTests(unittest.TestCase):
    def test_preflight_checks_exact_historical_event(self):
        plan = event.preflight()
        self.assertEqual(len(plan["old_frames"]), 46)
        self.assertEqual([plan["old_by_index"][i]["handedness_observed"][0]
                          for i in (2701, 2702, 2703)], ["Right", "Left", "Right"])
        self.assertEqual((plan["width"], plan["height"]), (640, 480))

    def test_check_only_never_opens_video_or_mediapipe(self):
        before = [(path, path.exists()) for path in event.output_paths()]
        with patch("cv2.VideoCapture", side_effect=AssertionError("video opened")):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(event.main(["--check-only"]), 0)
        self.assertEqual(before, [(path, path.exists()) for path, _ in before])

    def test_handedness_score_is_optional_and_not_invented(self):
        got = event.categories_for_hand([SimpleNamespace(category_name="Left", score=0.62)])
        self.assertEqual(got, [{"label": "Left", "score": 0.62}])
        self.assertEqual(event.categories_for_hand([SimpleNamespace(category_name="Right")]),
                         [{"label": "Right", "score": None}])

    def test_adjacent_shape_formula_is_reused(self):
        points = np.array([[i % 5 * 10.0, i // 5 * 8.0] for i in range(21)])
        shifted = points + np.array([2.0, 3.0])
        scale = event.prior_audit.continuity.palm_scale(points)
        records = [{"original_frame_index": 2701, "valid_hand_count": 1, "raw_hand_count": 1,
                    "_valid_hands": [{"points_xy": points, "scale": scale}]},
                   {"original_frame_index": 2702, "valid_hand_count": 1, "raw_hand_count": 1,
                    "_valid_hands": [{"points_xy": shifted, "scale": scale}]}]
        metrics = event.adjacent_metrics(records)
        self.assertEqual(len(metrics), 1)
        self.assertGreater(metrics[0]["wrist_step_palm_widths"], 0)
        self.assertAlmostEqual(metrics[0]["shape_step_palm_widths"], 0.0)

    def test_prior_comparison_reports_changed_label(self):
        old = {"raw_hand_count": 1, "valid_hand_count": 1,
               "handedness_observed": ["Left"], "wrist_pixel_xy": [[10.0, 20.0]],
               "palm_scales_px": [50.0]}
        new = {"raw_hand_count": 1, "valid_hand_count": 1,
               "handedness_observed": ["Right"], "wrist_pixel_xy": [[10.0, 20.0]],
               "palm_scales_px": [50.0]}
        result = event.compare_to_old(new, old)
        self.assertIn("HANDEDNESS_LABEL_CHANGED", result["replay_difference_flags"])
        self.assertEqual(result["wrist_delta_px"], 0.0)

    def test_only_nine_overlay_files_are_planned(self):
        names = [path.name for path in event.output_paths() if path.parent == event.OVERLAY_DIR]
        self.assertEqual(names, [f"frame_{i:04d}.png" for i in range(2698, 2707)])

    def test_overlay_uses_copy_and_draws_21_indexed_points(self):
        source = np.zeros((100, 160, 3), dtype=np.uint8)
        record = {"original_frame_index": 2702, "raw_hand_count": 1, "valid_hand_count": 1,
                  "hands": [{"valid": True, "handedness_label": "Left",
                             "handedness_confidence": 0.51,
                             "landmarks_xy_pixels": [[20 + i * 3, 40] for i in range(21)]}]}
        overlay = event.make_overlay(source, record)
        self.assertEqual(overlay.shape, (156, 160, 3))
        self.assertTrue(np.any(overlay[56:] != 0))
        self.assertFalse(np.any(source != 0))


if __name__ == "__main__":
    unittest.main()
