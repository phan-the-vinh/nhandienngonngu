"""Synthetic-only checks for the J/Z baseline-B feature contract."""

import contextlib
import io
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ai.feature_extraction.dynamic_sequence import (
    DynamicSequenceError,
    build_sequence_features,
    detect_active_interval,
    sample_frame_indices,
    sample_valid_frame_indices,
)
from ai.preprocessing.landmark_utils import normalize_landmarks
from tools import extract_dynamic


MOTION_PARAMS = {
    "median_window": 5,
    "mean_window": 3,
    "min_valid_fraction": 0.9,
    "min_peak_score": 0.04,
    "min_peak_noise_ratio": 2.5,
    "noise_mad_multiplier": 3.0,
    "relative_peak_fraction": 0.25,
    "min_active_edges": 3,
    "max_gap_frames": 6,
    "secondary_mass_ratio": 0.35,
    "margin_frames": 3,
}


def motion_points(offset=0.0):
    points = np.full((21, 2), (100.0, 100.0), dtype=np.float64)
    points[5] = (115.0, 140.0)
    points[9] = (100.0, 145.0)
    points[17] = (85.0, 140.0)
    points[8] = (130.0, 65.0)
    points[20] = (70.0, 65.0)
    points[:, 0] += offset
    return points


def hand_at(dx=0.0, dy=0.0, index_extra=0.0, pinky_extra=0.0):
    points = [SimpleNamespace(x=0.3 + dx, y=0.4 + dy, z=0.0) for _ in range(21)]
    points[5] = SimpleNamespace(x=0.35 + dx, y=0.6 + dy, z=0.01)
    points[9] = SimpleNamespace(x=0.3 + dx, y=0.65 + dy, z=0.02)
    points[17] = SimpleNamespace(x=0.25 + dx, y=0.6 + dy, z=-0.01)
    points[8] = SimpleNamespace(x=0.4 + dx + index_extra, y=0.25 + dy, z=0.01)
    points[20] = SimpleNamespace(x=0.2 + dx, y=0.25 + dy + pinky_extra, z=-0.01)
    return points


class DynamicSequenceTests(unittest.TestCase):
    def test_temporal_sampling_spans_full_video_without_padding(self):
        indices = sample_frame_indices(26)
        self.assertEqual(len(indices), 16)
        self.assertEqual((indices[0], indices[-1]), (0, 25))
        self.assertEqual(len(set(indices)), 16)
        self.assertEqual(indices, sorted(indices))
        self.assertNotEqual(indices, list(range(16)))
        with self.assertRaisesRegex(DynamicSequenceError, "TOO_FEW_DECODED_FRAMES"):
            sample_frame_indices(15)

    def test_official_sampling_uses_only_ordered_valid_original_indices(self):
        valid = list(range(1, 50, 2))  # 25 valid frames among 50 decoded frames.
        indices = sample_valid_frame_indices(valid)
        self.assertEqual(len(indices), 16)
        self.assertEqual(indices[0], 1)
        self.assertEqual(indices[-1], 49)
        self.assertEqual(indices, sorted(set(indices)))
        self.assertTrue(set(indices).issubset(valid))
        self.assertEqual(sample_valid_frame_indices(valid[:16]), valid[:16])
        with self.assertRaisesRegex(DynamicSequenceError, "INSUFFICIENT_VALID_LANDMARK_FRAMES"):
            sample_valid_frame_indices(valid[:15])
        with self.assertRaisesRegex(DynamicSequenceError, "INVALID_VALID_FRAME_INDICES"):
            sample_valid_frame_indices([1, 3, 3] + list(range(5, 31, 2)))

    def test_shape_and_aspect_correct_trajectory(self):
        hands = [hand_at(dx=0.05 * t / 15, dy=0.05 * t / 15) for t in range(16)]
        values, scale = build_sequence_features(hands, [(200, 100)] * 16)
        self.assertEqual(values.shape, (16, 69))
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(values[0, :63], normalize_landmarks(hands[0]))
        np.testing.assert_allclose(values[0, 63:], 0.0)
        # The same image-normalized XY shift is 10 px horizontally, 5 px vertically.
        np.testing.assert_allclose(values[-1, 63:65], [10.0 / scale, 5.0 / scale])
        np.testing.assert_allclose(values[-1, 65:67], [10.0 / scale, 5.0 / scale])
        np.testing.assert_allclose(values[-1, 67:69], [10.0 / scale, 5.0 / scale])
        self.assertGreater(scale, 0)

    def test_index_and_pinky_motion_remain_distinct_from_wrist(self):
        hands = [hand_at(index_extra=0.02 * t / 15, pinky_extra=0.03 * t / 15)
                 for t in range(16)]
        values, scale = build_sequence_features(hands, [(200, 100)] * 16)
        np.testing.assert_allclose(values[-1, 63:65], [0, 0])
        np.testing.assert_allclose(values[-1, 65:67], [4.0 / scale, 0])
        np.testing.assert_allclose(values[-1, 67:69], [0, 3.0 / scale])

    def test_rejects_missing_or_invalid_landmarks_and_degenerate_scale(self):
        hands = [hand_at() for _ in range(16)]
        hands[2] = hands[2][:-1]
        with self.assertRaisesRegex(DynamicSequenceError, "INVALID_LANDMARK_COUNT"):
            build_sequence_features(hands, [(200, 100)] * 16)
        hands[2] = hand_at()
        with self.assertRaisesRegex(DynamicSequenceError, "INCONSISTENT_FRAME_SIZE"):
            build_sequence_features(hands, [(200, 100)] * 15 + [(100, 100)])
        flat = [[SimpleNamespace(x=0.3, y=0.4, z=0.0) for _ in range(21)]
                for _ in range(16)]
        with self.assertRaisesRegex(DynamicSequenceError, "DEGENERATE_HAND_SCALE"):
            build_sequence_features(flat, [(200, 100)] * 16)
        hands[2][8].x = float("nan")
        with self.assertRaisesRegex(DynamicSequenceError, "NONFINITE_LANDMARK"):
            build_sequence_features(hands, [(200, 100)] * 16)

    def test_fewer_than_sixteen_valid_frames_requires_review(self):
        source = extract_dynamic.ROOT / "data/raw/dynamic/external/signnteam_jz/SigNN Video Data/Z/example.avi"
        entries = [(index, hand_at(), (200, 100)) for index in range(1, 30, 2)]
        with patch.object(extract_dynamic, "_all_valid_landmarks", return_value=(30, entries, "")):
            row, features = extract_dynamic.extract_one(
                source, "Z", object(), {"decoded_frame_count": 30, "valid_landmark_frames": 15})
        self.assertIsNone(features)
        self.assertEqual(row["extraction_status"], "REVIEW_REQUIRED")
        self.assertEqual(row["failure_reason"], "INSUFFICIENT_VALID_LANDMARK_FRAMES")
        self.assertEqual(row["z_variant"], "UNKNOWN")
        self.assertEqual(row["decoded_frame_count"], 30)
        self.assertEqual(row["valid_landmark_frames"], 15)
        self.assertEqual((row["sequence_length"], row["feature_dim"]), (16, 69))

    def test_official_extract_one_preserves_indices_and_first_selected_trajectory_origin(self):
        source = extract_dynamic.ROOT / "data/raw/dynamic/external/signnteam_jz/SigNN Video Data/J/example.avi"
        entries = [(index, hand_at(dx=index * 0.001), (200, 100))
                   for index in range(1, 50, 2)]
        with patch.object(extract_dynamic, "_all_valid_landmarks", return_value=(50, entries, "")):
            row, features = extract_dynamic.extract_one(
                source, "J", object(), {"decoded_frame_count": 50, "valid_landmark_frames": 25})
        self.assertEqual(row["extraction_status"], "SUCCESS")
        self.assertEqual(row["z_variant"], "NOT_APPLICABLE")
        self.assertEqual(row["selected_original_frame_indices"], sample_valid_frame_indices(
            [entry[0] for entry in entries]))
        self.assertEqual(np.asarray(features).shape, (16, 69))
        np.testing.assert_allclose(np.asarray(features)[0, 63:], 0.0)
        self.assertTrue(np.isfinite(features).all())

    def test_coverage_drift_prevents_training_sample(self):
        source = extract_dynamic.ROOT / "data/raw/dynamic/external/signnteam_jz/SigNN Video Data/J/example.avi"
        entries = [(index, hand_at(), (200, 100)) for index in range(20)]
        with patch.object(extract_dynamic, "_all_valid_landmarks", return_value=(30, entries, "")):
            row, features = extract_dynamic.extract_one(
                source, "J", object(), {"decoded_frame_count": 30, "valid_landmark_frames": 21})
        self.assertIsNone(features)
        self.assertEqual(row["failure_reason"], "LANDMARK_COVERAGE_DIFFERS_FROM_AUDIT")

    def test_preview_selection_defaults_to_five_per_label_and_accepts_specific_video(self):
        root = extract_dynamic.ROOT / "data/raw/dynamic/external/signnteam_jz/SigNN Video Data"
        videos = [(root / label / f"{index}.avi", label)
                  for label in ("J", "Z") for index in range(20)]
        selected = extract_dynamic.select_preview_videos(videos, root, [], 5)
        self.assertEqual(Counter(label for _, label in selected), {"J": 5, "Z": 5})
        self.assertEqual([path.name for path, _ in selected[:5]],
                         ["0.avi", "5.avi", "10.avi", "14.avi", "19.avi"])
        chosen = extract_dynamic.select_preview_videos(videos, root, ["Z/3.avi"], 5)
        self.assertEqual(chosen, [(root / "Z/3.avi", "Z")])
        with self.assertRaisesRegex(ValueError, "Unknown or out-of-root"):
            extract_dynamic.select_preview_videos(videos, root, ["../outside.avi"], 5)

    def test_preview_sheet_has_sixteen_ordered_index_captions(self):
        import cv2

        frames = [np.full((20, 40, 3), index, dtype=np.uint8) for index in range(16)]
        indices = sample_frame_indices(26)
        with patch.object(cv2, "putText", wraps=cv2.putText) as draw_text:
            sheet = extract_dynamic.make_preview_sheet(frames, indices, "J/1.avi")
        self.assertEqual(sheet.shape, (1130, 1280, 3))
        captions = [call.args[1] for call in draw_text.call_args_list]
        self.assertEqual(len(captions), 17)  # one title plus 16 indexed tiles
        self.assertIn("01/16  frame 0", captions)
        self.assertIn("16/16  frame 25", captions)

    def test_motion_interval_crops_idle_and_preserves_sequence_order(self):
        frames = [motion_points(2.0 * max(0, min(t - 30, 30))) for t in range(90)]
        result = detect_active_interval(frames, MOTION_PARAMS)
        self.assertEqual(result["status"], "OK", result["failure_reason"])
        self.assertLess(result["margin_start"], 30)
        self.assertGreater(result["margin_end"], 60)
        self.assertLess(result["margin_end"] - result["margin_start"] + 1, 90)
        indices = result["sampled_frame_indices"]
        self.assertEqual(len(indices), 16)
        self.assertEqual(indices[0], result["margin_start"])
        self.assertEqual(indices[-1], result["margin_end"])
        self.assertEqual(indices, sorted(indices))

    def test_motion_review_policy_for_idle_missing_and_disjoint_bouts(self):
        still = [motion_points() for _ in range(90)]
        self.assertEqual(detect_active_interval(still, MOTION_PARAMS)["status"], "REVIEW_REQUIRED")
        moving = [motion_points(2.0 * max(0, min(t - 30, 30))) for t in range(90)]
        moving[45] = None
        result = detect_active_interval(moving, MOTION_PARAMS)
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertEqual(result["failure_reason"], "MISSING_LANDMARK_IN_PROPOSED_INTERVAL")
        two_bouts = [motion_points(2.0 * max(0, min(t - 20, 15)) +
                                    2.0 * max(0, min(t - 55, 15))) for t in range(90)]
        result = detect_active_interval(two_bouts, MOTION_PARAMS)
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertEqual(result["failure_reason"], "MULTIPLE_SEPARATED_MOTION_CLUSTERS")

    def test_official_extraction_cli_requires_confirmation(self):
        root = extract_dynamic.ROOT
        fake_config = {
            "labels": ["J", "Z", "NONE"],
            "paths": {"source_root": root, "output_jsonl": root / "unused.jsonl",
                      "audit_csv": root / "unused.csv"},
            "spec": {"sampling_policy_status": "VALID_LANDMARK_FRAMES_V1"},
        }
        source = root / "J/1.avi"
        coverage = {source.relative_to(root).as_posix(): {
            "decoded_frame_count": 20, "valid_landmark_frames": 20}}
        with patch.object(extract_dynamic, "load_config", return_value=fake_config), \
                patch.object(extract_dynamic, "discover_videos",
                             return_value=[(source, "J")]), \
                patch.object(extract_dynamic, "load_coverage_contract", return_value=coverage), \
                patch.object(extract_dynamic, "execute") as execute, \
                patch("builtins.input", return_value="n") as prompt, \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(extract_dynamic.main(["--execute"]), 0)
        execute.assert_not_called()
        prompt.assert_called_once()
        with patch.object(extract_dynamic, "load_config", return_value=fake_config), \
                patch.object(extract_dynamic, "discover_videos", return_value=[(source, "J")]), \
                patch.object(extract_dynamic, "load_coverage_contract", return_value=coverage), \
                patch.object(extract_dynamic, "execute") as execute, \
                patch("builtins.input", return_value="y"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(extract_dynamic.main(["--execute"]), 0)
        execute.assert_called_once_with([(source, "J")], fake_config["paths"], coverage)


if __name__ == "__main__":
    unittest.main()
