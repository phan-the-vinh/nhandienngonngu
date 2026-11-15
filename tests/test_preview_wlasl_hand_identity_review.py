"""Offline checks for the saved-diagnostic WLASL continuity preview."""

import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools import preview_wlasl_hand_identity_review as tool


def frame(index, *, hand=True, label="Right", wrist=(0, 0), scale=10):
    return {
        "original_frame_index": index,
        "raw_hand_count": int(hand), "valid_hand_count": int(hand),
        "handedness_observed": [label] if hand else [],
        "wrist_pixel_xy": [list(wrist)] if hand else [],
        "palm_scales_px": [scale] if hand else [],
    }


class ContinuityPreviewTests(unittest.TestCase):
    def test_saved_events_are_located_without_inventing_shape_location(self):
        frames = [frame(0), frame(1, hand=False), frame(2, label="Left"),
                  frame(3, label="Left", wrist=(20, 0))]
        events = tool.event_locations(frames, {"max_shape_step_palm_widths": "1.5"})
        self.assertEqual(events["gaps"], [1])
        self.assertEqual(events["changes"], [2])
        self.assertEqual(events["wrist_jumps"], [3])
        self.assertEqual(events["shape_location"], "NOT_STORED_IN_FRAME_DIAGNOSTICS")

    def test_preview_frames_are_ordered_within_reviewed_segment(self):
        events = {"gaps": [20], "changes": [30], "unknown": [], "wrist_jumps": [40]}
        indices = tool.choose_preview_indices(10, 59, events)
        self.assertEqual((len(indices), indices[0], indices[-1]), (16, 10, 59))
        self.assertEqual(indices, sorted(set(indices)))
        self.assertTrue({20, 30, 40} <= set(indices))

    def test_real_check_only_never_decodes_or_writes(self):
        with patch.object(tool.cv2, "VideoCapture", side_effect=AssertionError("decoded")), \
             patch.object(Path, "mkdir", side_effect=AssertionError("wrote")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                result = tool.main(["--check-only"])
        self.assertEqual(result, 0)
        self.assertIn('"review_single_hand": 21', output.getvalue())
        self.assertIn('"priority": 11', output.getvalue())
        self.assertIn('"locatable_alert_windows": 20', output.getvalue())

    def test_alert_windows_use_only_reviewed_segment_and_preserve_shape_limit(self):
        entries = tool.alert_frame_plan(tool.preflight())
        self.assertEqual(len(entries), 21)
        self.assertEqual(sum(entry["center"] is not None for entry in entries), 20)
        unresolved = [entry for entry in entries if entry["center"] is None]
        self.assertEqual([entry["item"]["row"]["video_id"] for entry in unresolved], ["20666"])
        self.assertEqual(tool.alert_index_row(unresolved[0])["diagnostic_status"],
                         "NOT_LOCATABLE_WITH_SAVED_DIAGNOSTICS")
        for entry in entries:
            if entry["center"] is None:
                continue
            start = int(entry["item"]["row"]["segment_start"])
            end = int(entry["item"]["row"]["segment_end"])
            indices = entry["indices"]
            self.assertEqual(indices, list(range(max(start, entry["center"] - 4),
                                                 min(end, entry["center"] + 4) + 1)))
            self.assertTrue(all(start <= index <= end for index in indices))

    def test_existing_decoder_accepts_short_window_without_real_video(self):
        class Capture:
            def isOpened(self):
                return True

            def read(self):
                return True, np.zeros((3, 3, 3), dtype=np.uint8)

            def release(self):
                pass

        with patch.object(tool.cv2, "VideoCapture", return_value=Capture()):
            frames = tool.existing_preview.selected_frames(Path("not_a_video"), [2, 3, 4],
                                                            expected_count=3)
        self.assertEqual([index for index, _ in frames], [2, 3, 4])

    def test_exact_eight_jz_sampler_indices_and_warning_brackets(self):
        entries = tool.sampled_16_plan(tool.preflight(tool.TARGET_8))
        self.assertEqual([entry["item"]["row"]["video_id"] for entry in entries],
                         list(tool.TARGET_8))
        for entry in entries:
            video_id = entry["item"]["row"]["video_id"]
            self.assertEqual(entry["indices"], tool.PREVIOUS_SAMPLED_16[video_id])
            self.assertEqual(len(entry["indices"]), 16)
            self.assertEqual(entry["indices"], sorted(set(entry["indices"])))
        gap = entries[0]["warnings"]
        self.assertEqual([(warning["frame"], warning["left"], warning["right"])
                          for warning in gap], [(25, 24, 28), (32, 28, 33)])
        handedness = entries[-1]["warnings"]
        self.assertEqual([(warning["frame"], warning["position"])
                          for warning in handedness],
                         [(6, "BETWEEN_SELECTED_FRAMES"), (49, "AT_SELECTED_FRAME")])

    def test_prior_index_mismatch_stops_before_preview(self):
        with patch.dict(tool.PREVIOUS_SAMPLED_16, {"23769": list(range(16))}):
            with self.assertRaisesRegex(ValueError, "previously reported indices"):
                tool.sampled_16_plan(tool.preflight(tool.TARGET_8))

    def test_sampled_16_check_only_never_decodes_or_writes(self):
        with patch.object(tool.cv2, "VideoCapture", side_effect=AssertionError("decoded")), \
             patch.object(Path, "mkdir", side_effect=AssertionError("wrote")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                result = tool.main(["--check-only", "--sampled-16"])
        self.assertEqual(result, 0)
        self.assertIn('"videos": 8', output.getvalue())
        self.assertIn('"indices_match_prior_analysis": true', output.getvalue())
        self.assertIn('"approved_training_sequences": 0', output.getvalue())


if __name__ == "__main__":
    unittest.main()
