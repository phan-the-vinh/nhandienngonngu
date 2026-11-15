"""Preserved immediate-first16 regressions; motion-gated mode has separate tests."""
import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from ai.feature_extraction.dynamic_sequence import build_sequence_features
from app.inference import dynamic_jz_auto_camera_first16_v1 as auto
from tests.test_dynamic_jz_auto_camera_experiment import Model, hand_at


class First16Tests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.events = []
        self.session = auto.AutoJzSession(self.model, event_sink=self.events.append)

    def feed(self, index, *, hand=True, offset=None, now=None, error="", count=None):
        if hand is True:
            hand = hand_at(index*.008 if offset is None else offset)
        elif hand is False:
            hand = None
        now = index/30 if now is None else now
        self.session.update(index, hand, (640,480), now, detection_error=error,
                            hand_count=(1 if hand is not None else 0) if count is None else count)
        return self.session.process_pending(now)

    def start(self):
        for i in range(10):
            self.feed(i)
        self.assertEqual(self.session.state, "CAPTURING")
        self.assertEqual(self.session.valid_frames, 10)
        self.assertEqual(len(self.model.inputs), 0)

    def test_three_missing_frames_preserve_segment_and_resume_count(self):
        self.start()
        segment = self.session.segmenter.entries
        for i in range(10,13):
            self.feed(i, hand=False)
            self.assertIs(self.session.segmenter.entries, segment)
            self.assertEqual(self.session.valid_frames, 10)
            self.assertEqual(self.session.segmenter.missing_frames, i-9)
            self.assertEqual(self.session.state, "CAPTURING")
        self.feed(13, offset=.08)
        self.assertEqual(self.session.valid_frames, 11)
        self.assertEqual(self.session.segmenter.missing_frames, 0)
        self.assertEqual(self.session.attempts.counts["TOTAL_ATTEMPTS"], 1)

    def test_predict_immediately_on_16th_valid_frame_without_quiet_or_geometry_gate(self):
        with patch("app.inference.automatic_jz_experiment.geometry_candidates",
                   side_effect=AssertionError("whole-path gate must not block a first16 window")):
            for i in range(15):
                self.feed(i)
            self.assertEqual(self.session.valid_frames, 15)
            self.assertEqual(self.model.inputs, [])
            result = self.feed(15)
        self.assertEqual(result["status"], "RESULT")
        self.assertEqual(result["endpoint"], "FIRST_16_VALID_FRAMES")
        self.assertFalse(result["endpoint_verified"])
        self.assertEqual(result["feature_shape"], [16,69])
        self.assertEqual(result["selected_frame_indices"], list(range(16)))
        self.assertEqual(auto.display_lines(self.session,.5)["label"], "Z")
        self.assertEqual(len(self.model.inputs), 1)
        self.assertEqual(self.session.attempts.counts["CAPTURE_COMPLETED"], 1)

    def test_missing_frames_never_enter_feature_builder(self):
        self.start()
        for i in (10,11,12):
            self.feed(i, hand=False)
        for i in range(13,19):
            result = self.feed(i, offset=(i-3)*.008)
        indices = list(range(10)) + list(range(13,19))
        self.assertEqual(result["selected_frame_indices"], indices)
        expected, _ = build_sequence_features(
            [hand_at(i*.008 if i < 10 else (i-3)*.008) for i in indices], [(640,480)]*16)
        np.testing.assert_array_equal(self.model.inputs[0], expected.reshape(1,1104,order="C"))
        self.assertEqual(self.session.valid_frames, 16)
        self.assertEqual(self.session.attempts.counts["MODEL_INVOKED"], 1)

    def test_partial_and_nonfinite_landmark_frames_are_skipped_not_reset(self):
        self.start()
        self.feed(10, hand=hand_at()[:20])
        bad = hand_at()
        bad[8].x = float("nan")
        self.feed(11, hand=bad)
        self.assertEqual(self.session.valid_frames, 10)
        self.assertEqual(self.session.segmenter.missing_frames, 2)
        self.assertEqual(self.session.state, "CAPTURING")
        self.feed(12, offset=.08)
        self.assertEqual(self.session.valid_frames, 11)
        self.assertEqual(self.session.segmenter.missing_frames, 0)

    def test_detect_frame_invalid_21_count_becomes_bounded_gap(self):
        self.start()
        task = MagicMock()
        task.detect.return_value = SimpleNamespace(hand_landmarks=[hand_at()[:20]])
        hand,count,error = auto.detect_frame(task,np.zeros((10,10,3),dtype=np.uint8))
        self.assertIsNone(hand)
        self.assertEqual(count,1)
        self.assertTrue(error.startswith("INVALID_LANDMARK_FRAME:"))
        self.feed(10,hand=False,error=error,count=count)
        self.assertEqual(self.session.valid_frames,10)
        self.assertEqual(self.session.segmenter.missing_frames,1)

    def test_gap_over_frame_limit_resets_and_reports_reason(self):
        self.start()
        for i in range(10,16):
            self.feed(i, hand=False)
            self.assertEqual(self.session.valid_frames, 10)
        self.feed(16, hand=False)
        self.assertEqual(self.session.state, "REJECTED")
        self.assertEqual(self.session.reason, "MISSING_HAND_LIMIT")
        self.assertEqual(self.session.valid_frames, 0)
        self.assertEqual(self.session.segmenter.entries, [])
        self.assertEqual(self.model.inputs, [])
        self.feed(17, offset=.08)
        self.assertEqual(self.session.segmenter.entries, [])  # Old 10-frame segment is gone.

    def test_gap_over_time_limit_resets_even_with_few_missing_frames(self):
        self.start()
        self.feed(10, hand=False, now=.71)  # Last valid was .30; blind for >400ms.
        self.assertEqual(self.session.state, "REJECTED")
        self.assertEqual(self.session.valid_frames, 0)
        self.assertEqual(self.model.inputs, [])

    def test_custom_gap_limit_is_honored(self):
        settings = auto.replace(auto.SegmentSettings(),max_missing_frames=3,max_frame_gap_seconds=.4)
        self.session = auto.AutoJzSession(self.model,settings)
        self.start()
        for i in range(10,13):
            self.feed(i,hand=False)
            self.assertEqual(self.session.valid_frames,10)
        self.feed(13,hand=False)
        self.assertEqual(self.session.state,"REJECTED")

    def test_timeout_does_not_predict_with_less_than_16_valid_frames(self):
        settings = auto.replace(auto.SegmentSettings(),max_seconds=.15,
                                max_missing_frames=6,max_frame_gap_seconds=.4)
        self.session = auto.AutoJzSession(self.model,settings)
        for i in range(30):
            self.feed(i)
            if self.session.last_result:
                break
        self.assertEqual(self.session.reason,"SEGMENT_TIMEOUT")
        self.assertEqual(self.session.valid_frames,0)
        self.assertEqual(self.session.attempts.counts["MODEL_INVOKED"],0)

    def test_stationary_hand_never_starts_just_to_collect_16(self):
        for i in range(120):
            self.feed(i,offset=.0001*np.sin(i))
        self.assertEqual(self.session.state,"READY")
        self.assertEqual(self.model.inputs,[])

    def test_result_does_not_repeat_during_continuing_motion_or_held_endpoint(self):
        for i in range(120):
            self.feed(i,offset=min(i,80)*.004)
        self.assertEqual(len(self.model.inputs),1)
        self.assertEqual(len(self.session.results),1)

    def test_real_quiet_rearms_next_bout_before_display_expiry(self):
        for i in range(110):
            offset = min(i,20)*.004 if i < 60 else .08+min(i-60,20)*.004
            self.feed(i,offset=offset)
        self.assertEqual(len(self.model.inputs),2)
        self.assertLess(self.session.results[1]["frame_id"]/30,
                        self.session.results[0]["frame_id"]/30+4.)

    def test_release_rearms_but_one_frame_loss_does_not(self):
        for i in range(16):
            self.feed(i)
        self.feed(16,hand=False)
        self.assertTrue(self.session.segmenter.waiting_for_rearm)
        for i in range(17,23):
            self.feed(i,hand=False)
        self.assertFalse(self.session.segmenter.waiting_for_rearm)
        for i in range(23,45):
            self.feed(i,offset=(i-23)*.004)
        self.assertEqual(len(self.model.inputs),2)

    def test_ambiguous_or_discontinuous_hand_still_rejected(self):
        self.start()
        self.feed(10,count=2)
        self.assertEqual(self.session.reason,"MULTIPLE_HANDS_IDENTITY_UNVERIFIED")
        self.assertEqual(self.model.inputs,[])
        self.setUp()
        self.start()
        self.feed(10,offset=1.)
        self.assertEqual(self.session.reason,"PALM_POSITION_DISCONTINUITY")
        self.assertEqual(self.model.inputs,[])

    def test_debug_contains_valid_count_gap_count_state_and_prediction_reason(self):
        self.start()
        self.feed(10,hand=False)
        for i in range(11,17):
            self.feed(i,offset=(i-1)*.008)
        for event in self.events:
            for key in ("current_state","valid_frame_count","missing_gap_count","collection_mode"):
                self.assertIn(key,event)
        finished = [e for e in self.events if e["event"]=="ATTEMPT_FINISHED"]
        self.assertEqual(len(finished),1)
        self.assertEqual(finished[0]["result"]["reason"],"FIRST16_BINARY_RESULT_NOT_GESTURE_CONFIRMATION")
        json.dumps(self.events,allow_nan=False)

    def test_cli_default_and_gap_configuration_without_camera_or_writes(self):
        with (patch.object(auto.cv2,"VideoCapture",side_effect=AssertionError("camera/video")),
              patch.object(auto,"create_hand_landmarker",side_effect=AssertionError("MediaPipe")),
              patch.object(auto,"DebugTrace",side_effect=AssertionError("write")),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(auto.main(["--check-only","--debug","--max-gap-frames","4","--max-gap-ms","350"]),0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["collection_mode"],"first16")
        self.assertFalse(payload["quiet_end_required_for_prediction"])
        self.assertEqual(payload["settings"]["max_missing_frames"],4)
        self.assertEqual(payload["settings"]["max_frame_gap_seconds"],.35)
        self.assertEqual(payload["files_written"],0)

    def test_invalid_gap_cli_and_incomplete_timeout_prediction_rejected(self):
        with (patch.object(auto,"load_experiment_model",side_effect=AssertionError("load")),
              contextlib.redirect_stderr(io.StringIO())):
            for argv in (["--max-gap-frames","-1"], ["--max-gap-ms","nan"],
                         ["--max-gap-ms","0"], ["--predict-on-timeout"]):
                with self.assertRaises(SystemExit):
                    auto.main(argv)

    def test_previous_quiet_end_cli_remains_available_with_original_gap_defaults(self):
        with (patch.object(auto.cv2,"VideoCapture",side_effect=AssertionError("camera")),
              patch.object(auto,"create_hand_landmarker",side_effect=AssertionError("MediaPipe")),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(auto.main(["--check-only","--collection-mode","quiet-end"]),0)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["quiet_end_required_for_prediction"])
        self.assertEqual(payload["settings"]["max_missing_frames"],2)
        self.assertEqual(payload["settings"]["max_frame_gap_seconds"],.2)


if __name__ == "__main__":
    unittest.main()
