"""No webcam, raw video decode, training, extraction or project writes."""
import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from app.inference import dynamic_jz_auto_camera_experiment as auto
from ai.feature_extraction.dynamic_sequence import build_sequence_features, sample_valid_frame_indices
from tests.test_unified_camera_experiment import synthetic_hand


def hand_at(offset=0.):
    return [SimpleNamespace(x=p.x+offset, y=p.y, z=p.z) for p in synthetic_hand()]


class Model:
    classes_ = np.asarray(["J", "Z"])

    def __init__(self):
        self.inputs = []

    def predict_proba(self, features):
        self.inputs.append(features.copy())
        return np.asarray([[.07, .93]])


class AutoJzTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.session = auto.AutoJzSession(self.model, collection_mode="quiet-end")

    def feed(self, index, offset=0., *, hand=True, process=True):
        self.session.update(index, hand_at(offset) if hand else None, (640,480), index/30)
        if process:
            self.session.process_pending(index/30)

    def capture(self):
        for i in range(30):
            self.feed(i, i*.004)
        self.assertEqual(self.session.state, "CAPTURING")

    def test_stationary_and_small_jitter_never_start_or_predict(self):
        for i in range(240):
            self.feed(i, .00005*np.sin(i))
        self.assertEqual(self.session.state, "READY")
        self.assertEqual(self.model.inputs, [])
        self.assertIsNone(self.session.last_result)

    def test_no_hand_while_idle_stays_ready_not_rejected(self):
        for i in range(120):
            self.feed(i, hand=False)
        self.assertEqual(self.session.state, "READY")
        self.assertEqual(self.session.valid_frames, 0)
        self.assertIsNone(self.session.last_result)
        self.assertEqual(self.model.inputs, [])

    def test_starts_without_initial_quiet_and_retains_pre_roll(self):
        self.capture()
        self.assertEqual(self.session.segmenter.entries[0][0], 0)
        self.assertGreaterEqual(self.session.valid_frames, 16)
        self.assertIsNone(self.session.pending)
        self.assertEqual(self.model.inputs, [])  # Not a frame-16 stop.

    def test_quiet_end_then_processing_then_largest_probability(self):
        with patch("app.inference.automatic_jz_experiment.geometry_candidates", return_value={"Z"}):
            for i in range(100):
                self.feed(i, min(i,30)*.004, process=False)
                if self.session.pending:
                    break
            self.assertEqual(self.session.state, "PROCESSING")
            self.assertEqual(self.model.inputs, [])
            entries = self.session.pending[0]
            expected_indices = sample_valid_frame_indices([entry[0] for entry in entries])
            mapping = {entry[0]: entry for entry in entries}
            expected, _ = build_sequence_features([mapping[j][1] for j in expected_indices], [(640,480)]*16)
            result = self.session.process_pending(i/30)
        self.assertEqual(result["status"], "RESULT")
        self.assertEqual(result["model_result"]["prediction"], "Z")
        self.assertEqual(result["selected_frame_indices"], expected_indices)
        np.testing.assert_array_equal(self.model.inputs[0], expected.reshape(1,1104, order="C"))
        display = auto.display_lines(self.session, i/30)
        self.assertEqual(display["label"], "Z")
        self.assertIn("P(J)=0.0700", display["probabilities"])
        self.assertIn("P(Z)=0.9300", display["probabilities"])
        self.assertGreaterEqual(len(entries), 16)

    def test_held_endpoint_never_predicts_again(self):
        with patch("app.inference.automatic_jz_experiment.geometry_candidates", return_value={"Z"}):
            for i in range(300):
                self.feed(i, min(i,30)*.004)
        self.assertEqual(len(self.model.inputs), 1)
        self.assertEqual(len(self.session.results), 1)
        self.assertEqual(self.session.state, "READY")

    def test_new_gesture_works_without_key_or_display_cooldown(self):
        with patch("app.inference.automatic_jz_experiment.geometry_candidates", return_value={"Z"}):
            for i in range(180):
                offset = min(i,30)*.004 if i < 60 else .12+min(i-60,30)*.004
                self.feed(i, offset)
        self.assertEqual(len(self.model.inputs), 2)
        results = [row for row in self.session.results if row["status"] == "RESULT"]
        self.assertEqual(len(results), 2)
        captures = [row for row in self.session.transitions if row["new_state"] == "CAPTURING"]
        self.assertLess(captures[1]["timestamp_seconds"], results[0]["frame_id"]/30+self.session.result_seconds)

    def test_short_missing_gap_is_not_endpoint_and_not_interpolated(self):
        self.capture()
        self.feed(30, hand=False)
        self.assertEqual(self.session.state, "CAPTURING")
        self.assertIn("NOT_AN_ENDPOINT", self.session.reason)
        self.assertIsNone(self.session.pending)
        self.feed(31, .124)
        indices = [row[0] for row in self.session.segmenter.entries]
        self.assertNotIn(30, indices)
        self.assertIn(31, indices)
        self.assertEqual(self.model.inputs, [])

    def test_long_missing_gap_rejects_without_prediction(self):
        self.capture()
        for i in range(30,33):
            self.feed(i, hand=False)
        self.assertEqual(self.session.state, "REJECTED")
        self.assertEqual(self.session.reason, "MISSING_HAND_LIMIT")
        self.assertEqual(self.model.inputs, [])

    def test_timeout_rejected_by_default(self):
        self.session = auto.AutoJzSession(self.model, auto.replace(auto.SegmentSettings(), max_seconds=.4),
                                         collection_mode="quiet-end")
        for i in range(80):
            self.feed(i, i*.004)
            if self.session.last_result:
                break
        self.assertEqual(self.session.last_result["status"], "REJECTED")
        self.assertEqual(self.session.reason, "SEGMENT_TIMEOUT")
        self.assertEqual(self.model.inputs, [])

    def test_timeout_prediction_is_opt_in_uncertain_and_never_result(self):
        self.session = auto.AutoJzSession(self.model, auto.replace(auto.SegmentSettings(), max_seconds=.4),
                                         predict_on_timeout=True, collection_mode="quiet-end")
        with patch("app.inference.automatic_jz_experiment.geometry_candidates", return_value={"Z"}):
            for i in range(80):
                self.feed(i, i*.004)
                if self.session.last_result:
                    break
        result = self.session.last_result
        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertFalse(result["endpoint_verified"])
        self.assertEqual(len(self.model.inputs), 1)
        self.assertEqual(auto.display_lines(self.session, i/30)["label"], "? Z")
        self.assertIn("TIMEOUT", self.session.reason)

    def test_unsupported_geometry_not_forced_through_binary_model(self):
        for i in range(100):
            self.feed(i, min(i,30)*.004)
            if self.session.last_result:
                break
        self.assertEqual(self.session.state, "REJECTED")
        self.assertEqual(self.session.reason, "MOTION_OR_HANDSHAPE_NOT_SUPPORTED")
        self.assertEqual(self.model.inputs, [])
        self.assertEqual(auto.display_lines(self.session, i/30)["label"], "-")

    def test_model_geometry_disagreement_is_uncertain_not_wrong_label(self):
        with patch("app.inference.automatic_jz_experiment.geometry_candidates", return_value={"J"}):
            for i in range(100):
                self.feed(i, min(i,30)*.004)
                if self.session.last_result:
                    break
        self.assertEqual(self.session.state, "UNCERTAIN")
        self.assertEqual(auto.display_lines(self.session, i/30)["label"], "-")
        self.assertIn("P(Z)=0.9300", auto.display_lines(self.session, i/30)["probabilities"])

    def test_eof_is_not_a_quiet_endpoint(self):
        self.capture()
        result = self.session.end_of_file(30, 1.)
        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertEqual(result["reason"], "EOF_WITHOUT_OBSERVED_QUIET_END")
        self.assertEqual(self.model.inputs, [])

    def test_invalid_tracking_or_ambiguity_never_predicts(self):
        for kwargs in ({"detection_error": "API failure"}, {"hand_count": 2}):
            self.session.update(0, hand_at(), (640,480), 0., **kwargs)
            self.assertEqual(self.session.state, "REJECTED")
            self.assertIsNone(self.session.pending)
        bad = hand_at()
        bad[8].x = float("nan")
        self.session.update(1, bad, (640,480), .03)
        self.assertEqual(self.session.state, "REJECTED")
        self.assertEqual(self.model.inputs, [])

    def test_pending_processed_once(self):
        with patch("app.inference.automatic_jz_experiment.geometry_candidates", return_value={"Z"}):
            for i in range(100):
                self.feed(i, min(i,30)*.004, process=False)
                if self.session.pending:
                    break
            self.session.process_pending(i/30)
            self.assertIsNone(self.session.process_pending(i/30+.1))
        self.assertEqual(len(self.model.inputs), 1)

    def test_raw_pixels_are_not_flipped_and_same_image_options(self):
        raw = np.arange(4*6*3, dtype=np.uint8).reshape(4,6,3)
        task = MagicMock()
        task.detect.return_value = SimpleNamespace(hand_landmarks=[hand_at()])
        hand, count, error = auto.detect_frame(task, raw)
        np.testing.assert_array_equal(task.detect.call_args.args[0].numpy_view(), cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        self.assertEqual(count, 1)
        self.assertEqual(error, "")
        self.assertEqual(len(hand), 21)
        rendered = auto.render(raw, self.session, 0.)
        np.testing.assert_array_equal(rendered[:4,:6], raw)

    def test_camera_only_one_capture_no_start_keys_and_safe_close(self):
        capture, task = MagicMock(), MagicMock()
        task.__enter__.return_value = task
        task.detect.return_value = SimpleNamespace(hand_landmarks=[hand_at()])
        capture.read.return_value = (True, np.zeros((480,640,3), dtype=np.uint8))
        with (patch.object(auto, "create_hand_landmarker", return_value=task) as create,
              patch.object(auto.cv2, "VideoCapture", return_value=capture) as opened,
              patch.object(auto.cv2, "imshow"), patch.object(auto.cv2, "waitKey", return_value=ord("q")),
              patch.object(auto.cv2, "destroyAllWindows")):
            self.assertEqual(auto.run_camera(self.session, 0), 0)
        opened.assert_called_once_with(0)
        create.assert_called_once_with(auto.TASK_PATH, running_mode="IMAGE", num_hands=1)
        task.detect.assert_called_once()
        capture.release.assert_called_once()

    def test_camera_read_failure_releases_camera(self):
        capture, task = MagicMock(), MagicMock()
        task.__enter__.return_value = task
        capture.read.return_value = (False, None)
        with (patch.object(auto, "create_hand_landmarker", return_value=task),
              patch.object(auto.cv2, "VideoCapture", return_value=capture),
              patch.object(auto.cv2, "destroyAllWindows")):
            with self.assertRaises(OSError):
                auto.run_camera(self.session, 0)
        capture.release.assert_called_once()

    def test_check_only_never_opens_camera_task_or_video(self):
        with (patch.object(auto.cv2, "VideoCapture", side_effect=AssertionError("capture")),
              patch.object(auto, "create_hand_landmarker", side_effect=AssertionError("MediaPipe")),
              patch.object(auto, "replay_verified_videos", side_effect=AssertionError("replay")),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(auto.main(["--check-only", "--replay-verified-videos"]), 0)
        payload = json.loads(output.getvalue())
        self.assertFalse(payload["camera_opened"])
        self.assertFalse(payload["mediapipe_executed"])
        self.assertEqual(payload["feature_shape"], [16,69])
        self.assertEqual(payload["files_written"], 0)

    def test_invalid_cli_threshold_fails_before_model_loading(self):
        with (patch.object(auto, "load_experiment_model", side_effect=AssertionError("loaded")),
              contextlib.redirect_stderr(io.StringIO())):
            for argv in (["--max-seconds", "nan"], ["--end-quiet-seconds", "0"], ["--start-frames", "1"]):
                with self.assertRaises(SystemExit):
                    auto.main(argv)


if __name__ == "__main__":
    unittest.main()
