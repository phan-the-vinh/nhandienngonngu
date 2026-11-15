"""Synthetic offline coverage only; no camera, videos, extraction or training."""
import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from ai.feature_extraction.dynamic_sequence import build_sequence_features, sample_valid_frame_indices
from app.inference import automatic_jz_experiment as logic
from app.inference import unified_abc_dynamic_experiment as camera
from tests.test_unified_camera_experiment import FakePredictor, synthetic_hand
from app.inference.unified_static_coordinator import RecognitionCoordinator
from app.interaction.hold_confirmation import HoldSettings


class Model:
    classes_ = np.asarray(["J", "Z"])

    def __init__(self, label="J"):
        self.label = label
        self.input = None

    def predict_proba(self, features):
        self.input = features.copy()
        return np.asarray([[.96, .04] if self.label == "J" else [.04, .96]])


def moving_hand(offset=0.0):
    return [SimpleNamespace(x=p.x + offset, y=p.y, z=p.z) for p in synthetic_hand()]


class AutomaticJzTests(unittest.TestCase):
    def setUp(self):
        self.letters = FakePredictor("I", 0.0)
        self.digits = FakePredictor("0", .5)
        self.coordinator = logic.AbcDynamicCoordinator(self.letters, self.digits, Model(), True)
        self.hand = synthetic_hand()

    def feed(self, now, index=0):
        return self.coordinator.process_hand(self.hand, self.hand, (480, 640, 3), index, now)

    def lock(self, label="J", now=0):
        with (patch.object(self.coordinator.segmenter, "update", return_value=("COMPLETE", "", [])),
              patch.object(logic, "evaluate_segment", return_value=((label, .96, list(range(16))), ""))):
            return self.feed(now)

    def test_dynamic_label_locked_for_three_seconds_static_cannot_overwrite(self):
        self.lock("J")
        self.assertEqual(self.coordinator.hold.pending, "J")
        for now in (.1, .2, 2.9):
            self.assertIsNone(self.feed(now)[1])
        self.assertEqual(self.feed(3.0)[1], "J")
        for now in (3.1, 4, 7):
            self.assertIsNone(self.feed(now)[1])
        self.assertEqual(self.coordinator.composer.text, "J")
        self.assertEqual(self.letters.calls, 8)  # Predictions remain visible, never overwrite locked J.

    def test_z_and_repeat_after_real_release(self):
        self.lock("Z")
        self.feed(.1)
        self.feed(.2)
        self.assertEqual(self.feed(3)[1], "Z")
        self.coordinator.missing_hand(3.1)
        self.coordinator.missing_hand(3.4)
        self.assertTrue(self.coordinator.blocked)
        self.coordinator.missing_hand(3.8)
        self.assertFalse(self.coordinator.blocked)
        self.lock("Z", 4)
        self.feed(4.1)
        self.feed(4.2)
        self.assertEqual(self.feed(7)[1], "Z")
        self.assertEqual(self.coordinator.composer.text, "ZZ")

    def test_motion_cancels_i_hold_and_old_votes(self):
        with patch.object(self.coordinator.segmenter, "update", return_value=("IDLE", "", [])):
            for now in (0, .1, .2, 2.9):
                self.feed(now)
        self.assertEqual(self.coordinator.hold.pending, "I")
        resets = self.letters.resets
        with patch.object(self.coordinator.segmenter, "update", return_value=("CAPTURING", "", [])):
            self.assertIsNone(self.feed(3.0)[1])
        self.assertIsNone(self.coordinator.hold.pending)
        self.assertGreater(self.letters.resets, resets)
        self.assertEqual(self.coordinator.composer.text, "")

    def test_default_review_only_returns_control_to_static(self):
        self.coordinator.enable_commits = False
        self.lock()
        self.assertIn("REVIEW_ONLY_J", self.coordinator.status)
        self.assertFalse(self.coordinator.blocked)
        with patch.object(self.coordinator.segmenter, "update", return_value=("IDLE", "", [])):
            for now in (.1, .2, .3):
                self.assertIsNone(self.feed(now)[1])
            self.assertEqual(self.feed(3.1)[1], "I")
            self.assertIsNone(self.feed(9)[1])
        self.assertEqual(self.coordinator.composer.text, "I")  # Never J/Z by default.

    def test_invalid_motion_rejects_and_does_not_query_model(self):
        with (patch.object(self.coordinator.segmenter, "update", return_value=("REJECTED", "TOO_SHORT", [])),
              patch.object(self.coordinator.model, "predict_proba", side_effect=AssertionError("queried"))):
            self.assertIsNone(self.feed(0)[1])
        self.assertFalse(self.coordinator.blocked)
        self.assertIn("TOO_SHORT", self.coordinator.status)

    def test_suspected_motion_does_not_cancel_static_hold(self):
        with patch.object(self.coordinator.segmenter, "update", return_value=("SUSPECT", "", [])):
            self.assertIsNone(self.feed(0)[1])
        self.assertEqual(self.letters.calls, 1)
        self.assertEqual(self.coordinator.hold.pending, "I")
        self.assertEqual(self.letters.resets, 0)

    def test_malformed_sequence_cannot_fall_back_to_static_confirmation(self):
        with (patch.object(self.coordinator.segmenter, "update", return_value=("COMPLETE", "", [])),
              patch.object(logic, "evaluate_segment", side_effect=ValueError("bad features"))):
            self.assertIsNone(self.feed(0)[1])
        self.assertFalse(self.coordinator.blocked)
        self.assertEqual(self.letters.calls, 1)

    def test_tracking_loss_cancels_endpoint_hold(self):
        self.lock()
        self.feed(.1)
        self.feed(.2)
        self.coordinator.missing_hand(1)
        self.assertIsNone(self.coordinator.locked)
        self.assertFalse(self.coordinator.blocked)
        self.assertIsNone(self.feed(3.2)[1])

    def test_ambiguity_is_not_a_valid_release(self):
        self.lock()
        self.feed(.1)
        self.feed(.2)
        self.assertEqual(self.feed(3)[1], "J")
        for now in (4, 5, 6, 9):
            self.coordinator.uncertain_hand(now)
        self.assertTrue(self.coordinator.blocked)
        self.assertIsNone(self.coordinator.missing_since)
        self.coordinator.missing_hand(9.1)
        self.coordinator.missing_hand(9.2)
        self.assertTrue(self.coordinator.blocked)
        self.coordinator.missing_hand(9.8)
        self.assertFalse(self.coordinator.blocked)

    def test_endpoint_pose_motion_cancels_not_restarts_hold(self):
        self.lock()
        moved = moving_hand(.3)
        self.coordinator.process_hand(moved, moved, (480, 640, 3), 1, .1)
        self.assertFalse(self.coordinator.blocked)
        self.assertIsNone(self.coordinator.locked)
        self.assertEqual(self.coordinator.composer.text, "")
        self.assertFalse(self.coordinator.blocked)

    def test_static_digits_switch_preserves_text_and_cancels_dynamic(self):
        with patch.object(self.coordinator.segmenter, "update", return_value=("IDLE", "", [])):
            self.letters.label = "A"
            for i, now in enumerate((0, .1, .2, 3)):
                self.feed(now, i)
        self.assertEqual(self.coordinator.composer.text, "A")
        self.lock("J", 3.1)
        self.coordinator.switch_mode("123")
        self.assertIsNone(self.coordinator.locked)
        for i, now in enumerate((4, 4.1, 4.2, 7)):
            self.feed(now, i + 20)
        self.assertEqual(self.coordinator.composer.text, "A0")
        self.coordinator.switch_mode("ABC")
        self.assertEqual(self.coordinator.hold.text, "A0")
        with self.assertRaises(ValueError):
            self.coordinator.switch_mode("JZ")

    def test_official_feature_pipeline_and_indices_for_both_labels(self):
        entries = [(i, moving_hand(i * .001), (640, 480)) for i in range(40)]
        expected_indices = sample_valid_frame_indices(list(range(40)))
        expected, _ = build_sequence_features([entries[i][1] for i in expected_indices], [(640, 480)] * 16)
        for label in ("J", "Z"):
            model = Model(label)
            with patch.object(logic, "geometry_candidates", return_value={label}):
                result, reason = logic.evaluate_segment(entries, model)
            self.assertEqual(result[0], label)
            self.assertEqual(reason, "")
            self.assertEqual(result[2], expected_indices)
            np.testing.assert_array_equal(model.input, expected.reshape(1, 1104))

    def test_model_geometry_disagreement_rejected(self):
        entries = [(i, moving_hand(i * .001), (640, 480)) for i in range(20)]
        with patch.object(logic, "geometry_candidates", return_value={"J"}):
            result, reason = logic.evaluate_segment(entries, Model("Z"))
        self.assertIsNone(result)
        self.assertIn("DISAGREEMENT", reason)

    def test_geometry_gate_rejects_idle_and_rigid_translation(self):
        entries = [(i, moving_hand(i * .01), (640, 480)) for i in range(16)]
        features, _ = build_sequence_features([entry[1] for entry in entries], [(640, 480)] * 16)
        self.assertEqual(logic.geometry_candidates(features), set())
        features[:, 63:] = 0
        self.assertEqual(logic.geometry_candidates(features), set())

    def test_geometry_gate_j_and_z_examples(self):
        for label in ("J", "Z"):
            features = np.zeros((16, 69))
            shape = features[:, :63].reshape(16, 21, 3)
            for tip in (8, 12, 16, 20):
                shape[:, tip - 2, 1] = -.5
                shape[:, tip, 1] = -.4
            shape[:, 20 if label == "J" else 8, 1] = -1
            if label == "J":
                points = np.asarray([[0, 0], [0, .6], [.5, .4]])
                times = [0, 9, 15]
                col = 67
            else:
                points = np.asarray([[0, 0], [.5, 0], [0, .4], [.5, .4]])
                times = [0, 5, 10, 15]
                col = 65
            for axis in range(2):
                features[:, col + axis] = np.interp(np.arange(16), times, points[:, axis])
            self.assertEqual(logic.geometry_candidates(features), {label})

    def test_segmenter_whole_bout_then_quiet_endpoint(self):
        segmenter = logic.MotionSegmenter()
        events = []
        completed = []
        for i in range(90):
            offset = 0 if i < 10 else min(i - 10, 20) * .003
            state, reason, entries = segmenter.update(i, moving_hand(offset), (640, 480), i / 30)
            events.append(state)
            if state == "COMPLETE":
                completed = entries
                break
        self.assertIn("CAPTURING", events)
        self.assertEqual(events[-1], "COMPLETE")
        self.assertGreaterEqual(len(completed), 16)
        self.assertLess(completed[0][0], 30)
        self.assertGreater(completed[-1][0], 30)
        self.assertTrue(all(b[0] > a[0] for a, b in zip(completed, completed[1:])))

    def test_segmenter_temporal_gap_and_palm_jump_rejected(self):
        for defect in ("time", "jump"):
            segmenter = logic.MotionSegmenter()
            segmenter.update(0, self.hand, (640, 480), 0)
            segmenter.active = True
            hand = moving_hand(.5 if defect == "jump" else 0)
            state, reason, _ = segmenter.update(1, hand, (640, 480), .5 if defect == "time" else .03)
            self.assertEqual(state, "REJECTED", (defect, reason))

    def test_stationary_hand_does_not_start_dynamic_model(self):
        segmenter = logic.MotionSegmenter()
        for i in range(180):
            offset = .00005 * np.sin(i)
            state, _reason, _entries = segmenter.update(i, moving_hand(offset), (640, 480), i / 30)
            self.assertEqual(state, "IDLE")

    def test_orientation_correspondence_guard_does_not_modify_landmarks(self):
        mirrored = [SimpleNamespace(x=1-p.x, y=p.y, z=p.z) for p in self.hand]
        self.assertTrue(camera.oriented_passes_agree(mirrored, self.hand, (480, 640, 3)))
        self.assertFalse(camera.oriented_passes_agree(moving_hand(.4), self.hand, (480, 640, 3)))
        self.assertEqual(mirrored[5].x, 1 - self.hand[5].x)

    def test_two_buttons_and_mirrored_display_both_modes(self):
        self.assertEqual(camera.mode_from_click(700, 30, 640), "ABC")
        self.assertEqual(camera.mode_from_click(850, 30, 640), "123")
        self.assertIsNone(camera.mode_from_click(850, 65, 640))
        raw = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        for mode in ("ABC", "123", "ABC"):
            frame = camera.prepare_inference_frame(raw, mode)
            np.testing.assert_array_equal(camera.render_mirrored_frame(frame, mode), raw[:, ::-1])

    def test_check_only_no_camera_or_mediapipe_execution(self):
        for args in (["--check-only"], ["--check-only", "--enable-experimental-jz"]):
            with (patch.object(camera.cv2, "VideoCapture", side_effect=AssertionError("camera")),
                  patch.object(camera, "create_hand_landmarker", side_effect=AssertionError("task")),
                  contextlib.redirect_stdout(io.StringIO()) as output):
                self.assertEqual(camera.main(args), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["modes"], ["ABC", "123"])
            self.assertFalse(report["camera_opened"])
            self.assertEqual(report["jz_commits_enabled"], len(args) == 2)

    def test_one_camera_independent_baseline_video_and_raw_image_passes(self):
        raw = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        capture = MagicMock()
        capture.read.return_value = (True, raw)
        task, dynamic_task = MagicMock(), MagicMock()
        task.__enter__.return_value = task
        dynamic_task.__enter__.return_value = dynamic_task
        images = []

        def detect(image, *_timestamp):
            images.append(image.numpy_view().copy())
            return SimpleNamespace(hand_landmarks=[self.hand])

        task.detect_for_video.side_effect = detect
        dynamic_task.detect.side_effect = detect
        with (patch.object(camera.cv2, "VideoCapture", return_value=capture) as open_camera,
              patch.object(camera, "create_hand_landmarker", side_effect=[task, dynamic_task]) as create,
              patch.object(camera.cv2, "namedWindow"), patch.object(camera.cv2, "setMouseCallback"),
              patch.object(camera.cv2, "imshow"), patch.object(camera.cv2, "waitKey", return_value=ord("q")),
              patch.object(camera.cv2, "destroyAllWindows"),
              patch.object(self.coordinator, "process_hand", return_value=(None, None)) as process):
            self.assertEqual(camera.run_camera(self.coordinator, 0), 0)
        open_camera.assert_called_once_with(0)
        self.assertEqual(create.call_count, 2)
        self.assertEqual(create.call_args_list[0].kwargs["running_mode"], "VIDEO")
        self.assertEqual(create.call_args_list[0].kwargs["num_hands"], 1)
        self.assertEqual(create.call_args_list[0].kwargs["min_tracking_confidence"], .5)
        self.assertEqual(create.call_args_list[1].kwargs, {"running_mode": "IMAGE", "num_hands": 2})
        self.assertEqual(len(images), 2)
        np.testing.assert_array_equal(images[0], cv2.cvtColor(raw[:, ::-1], cv2.COLOR_BGR2RGB))
        np.testing.assert_array_equal(images[1], cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        process.assert_called_once()
        capture.release.assert_called_once()

    def test_multiple_dynamic_hands_do_not_disable_valid_static_prediction(self):
        capture = MagicMock()
        capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        task, dynamic_task = MagicMock(), MagicMock()
        task.__enter__.return_value = task
        dynamic_task.__enter__.return_value = dynamic_task
        task.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[self.hand])
        dynamic_task.detect.return_value = SimpleNamespace(hand_landmarks=[self.hand, self.hand])
        with (patch.object(camera.cv2, "VideoCapture", return_value=capture),
              patch.object(camera, "create_hand_landmarker", side_effect=[task, dynamic_task]),
              patch.object(camera.cv2, "namedWindow"), patch.object(camera.cv2, "setMouseCallback"),
              patch.object(camera.cv2, "imshow"), patch.object(camera.cv2, "waitKey", return_value=ord("q")),
              patch.object(camera.cv2, "destroyAllWindows"),
              patch.object(camera, "_panel", wraps=camera._panel) as panel):
            self.assertEqual(camera.run_camera(self.coordinator, 0), 0)
        self.assertEqual(self.letters.calls, 1)
        self.assertEqual(panel.call_args.args[2], "I")
        self.assertFalse(self.coordinator.blocked)
        self.assertEqual(self.coordinator.composer.text, "")

    def test_a_with_natural_jitter_keeps_hold_and_confirms_once(self):
        self.letters.label = "A"
        with patch.object(self.coordinator.model, "predict_proba", side_effect=AssertionError("idle JZ")):
            for i in range(210):
                hand = moving_hand(.004 * np.sin(i * .8))  # roughly 2.6 px jitter
                prediction, _ = self.coordinator.process_hand(hand, hand, (480, 640, 3), i, i / 30)
                self.assertEqual(prediction.instant_label, "A")
                self.assertEqual(self.coordinator.suppression_reason, "")
        self.assertEqual(self.coordinator.composer.text, "A")
        self.assertEqual(self.letters.resets, 1)  # Only the successful confirmation.

    def test_l_display_and_hold_survive_secondary_failure(self):
        self.letters.label = "L"
        for i in range(121):
            prediction, _ = self.coordinator.process_hand(self.hand, None, (480, 640, 3), i, i / 30)
            self.assertEqual(prediction.instant_label, "L")
        self.assertEqual(self.coordinator.composer.text, "L")
        self.assertFalse(self.coordinator.blocked)

    def test_apple_with_real_segmenter_monitoring_and_repeated_p(self):
        now = 0.
        index = 0
        for position, (label, shape) in enumerate(zip("APPLE", (0., .3, .3, .6, .9))):
            if position == 2:  # A real no-hand release is needed for PP.
                self.coordinator.missing_hand(now)
                now += .7
                self.coordinator.missing_hand(now)
            self.letters.label, self.letters.shape = label, shape
            for _ in range(100):
                hand = moving_hand(.001 * np.sin(index))
                self.coordinator.process_hand(hand, hand, (480, 640, 3), index, now)
                now += 1 / 30
                index += 1
        self.assertEqual(self.coordinator.composer.text, "APPLE")
        self.assertEqual(self.coordinator.hold.text, "APPLE")

    def test_low_fps_dynamic_gap_rejection_does_not_block_static_hold(self):
        self.letters.label = "A"
        for i in range(20):
            self.feed(i / 4, i)  # Over the unchanged 0.20s dynamic tracking limit.
        self.assertEqual(self.coordinator.composer.text, "A")
        self.assertFalse(self.coordinator.blocked)

    def test_rejection_paths_resume_l_without_hand_removal(self):
        for reason in ("NO_QUIET_START", "TEMPORAL_GAP", "TRACKING_OR_POSE_JUMP", "TOO_SHORT", "SEGMENT_TIMEOUT"):
            with self.subTest(reason=reason):
                self.setUp()
                with patch.object(self.coordinator.segmenter, "update", return_value=("CAPTURING", "", [])):
                    for i in range(5):
                        prediction, committed = self.feed(i / 30, i)
                        self.assertEqual(prediction.instant_label, "I")
                        self.assertIsNone(committed)
                self.assertEqual(self.letters.resets, 1)  # No per-frame vote wipe.
                with patch.object(self.coordinator.segmenter, "update", return_value=("REJECTED", reason, [])):
                    self.feed(.2, 6)
                self.assertFalse(self.coordinator.blocked)
                self.letters.label = "L"
                with patch.object(self.coordinator.segmenter, "update", return_value=("IDLE", "", [])):
                    for i in range(100):
                        self.feed(.3 + i / 30, i + 7)
                self.assertEqual(self.coordinator.composer.text, "L")

    def test_disabled_dynamic_matches_baseline_every_step(self):
        baseline_letters, baseline_digits = FakePredictor("A", 0), FakePredictor("0", .5)
        baseline = RecognitionCoordinator(baseline_letters, baseline_digits, HoldSettings())
        self.coordinator.dynamic_enabled = False
        with patch.object(self.coordinator.segmenter, "update", side_effect=AssertionError("detector disabled")):
            for i in range(180):
                now = i / 30
                if i == 110:
                    baseline.switch_mode("123")
                    self.coordinator.switch_mode("123")
                if i < 110:
                    baseline_letters.label = self.letters.label = "A" if i < 100 else "L"
                    baseline_letters.shape = self.letters.shape = 0 if i < 100 else .6
                left = baseline.process_hand(self.hand, (480, 640, 3), now)
                right = self.coordinator.process_hand(self.hand, None, (480, 640, 3), i, now)
                self.assertEqual(left, right)
                self.assertEqual(baseline.hold.state, self.coordinator.hold.state)
                self.assertEqual(baseline.hold.progress, self.coordinator.hold.progress)
                self.assertEqual(baseline.composer.text, self.coordinator.composer.text)

    def test_start_hysteresis_rejects_one_frame_spike(self):
        segmenter = logic.MotionSegmenter()
        states = []
        for i in range(50):
            hand = moving_hand(.06 if i == 15 else 0)
            states.append(segmenter.update(i, hand, (640, 480), i / 30)[0])
        self.assertNotIn("CAPTURING", states)

    def test_index_and_little_finger_motion_create_bout_not_just_arm_translation(self):
        for tip in (8, 20):
            segmenter = logic.MotionSegmenter()
            states, completed = [], []
            for i in range(100):
                hand = moving_hand()
                hand[tip].y += max(0, min(i - 12, 25)) * .003
                state, _, entries = segmenter.update(i, hand, (640, 480), i / 30)
                states.append(state)
                if state == "COMPLETE":
                    completed = entries
                    break
            self.assertIn("CAPTURING", states)
            self.assertEqual(states[-1], "COMPLETE")
            self.assertGreaterEqual(len(completed), 16)
            self.assertEqual(sample_valid_frame_indices([e[0] for e in completed])[0], completed[0][0])

    def test_disabled_check_only_does_not_load_dynamic_artifacts(self):
        with (patch.object(camera, "load_experiment_model", side_effect=AssertionError("dynamic loaded")),
              patch.object(camera.cv2, "VideoCapture", side_effect=AssertionError("camera opened")),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(camera.main(["--check-only", "--disable-dynamic"]), 0)
        report = json.loads(output.getvalue())
        self.assertFalse(report["dynamic_enabled"])
        self.assertEqual(report["mediapipe_instances"], 1)

    def test_disabled_camera_uses_exact_baseline_task_and_ui_prediction(self):
        self.coordinator.dynamic_enabled = False
        self.letters.label = "L"
        capture, task = MagicMock(), MagicMock()
        raw = np.zeros((480, 640, 3), dtype=np.uint8)
        capture.read.return_value = (True, raw)
        task.__enter__.return_value = task
        task.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[self.hand])
        with (patch.object(camera.cv2, "VideoCapture", return_value=capture),
              patch.object(camera, "create_hand_landmarker", return_value=task) as create,
              patch.object(camera.cv2, "namedWindow"), patch.object(camera.cv2, "setMouseCallback"),
              patch.object(camera.cv2, "imshow"), patch.object(camera.cv2, "waitKey", return_value=ord("q")),
              patch.object(camera.cv2, "destroyAllWindows"),
              patch.object(camera, "_panel", wraps=camera._panel) as panel):
            self.assertEqual(camera.run_camera(self.coordinator, 0), 0)
        create.assert_called_once_with(model_path=camera.HAND_MODEL_PATH, running_mode="VIDEO", num_hands=1,
                                       min_hand_detection_confidence=.5, min_hand_presence_confidence=.5,
                                       min_tracking_confidence=.5)
        task.detect_for_video.assert_called_once()
        task.detect.assert_not_called()
        self.assertEqual(panel.call_args.args[2], "L")

    def test_debug_log_rate_limited_and_contains_diagnostic_fields(self):
        self.coordinator.dynamic_enabled = False
        capture, task = MagicMock(), MagicMock()
        capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        task.__enter__.return_value = task
        task.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[self.hand])
        with (patch.object(camera.cv2, "VideoCapture", return_value=capture),
              patch.object(camera, "create_hand_landmarker", return_value=task),
              patch.object(camera.cv2, "namedWindow"), patch.object(camera.cv2, "setMouseCallback"),
              patch.object(camera.cv2, "imshow"), patch.object(camera.cv2, "waitKey", side_effect=[0, 0, 0, ord("q")]),
              patch.object(camera.cv2, "destroyAllWindows"),
              patch.object(camera.time, "monotonic", side_effect=[0, 0, .1, .2, .6]),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(camera.run_camera(self.coordinator, 0, True), 0)
        logs = [json.loads(line.removeprefix("MOTION_DEBUG ")) for line in output.getvalue().splitlines()]
        self.assertEqual(len(logs), 2)
        self.assertTrue(set(("ACTIVE_MODE", "HAND_DETECTED", "STATIC_RAW_PREDICTION", "STATIC_STABLE_PREDICTION",
                            "HOLD_STATE", "DYNAMIC_STATE", "DYNAMIC_MOTION_SCORE", "STATIC_SUPPRESSION_REASON",
                            "JZ_CANDIDATE", "FPS", "MEDIAPIPE_MS", "INFERENCE_MS")).issubset(logs[0]))
        self.assertEqual(logs[0]["STATIC_RAW_PREDICTION"], "I")

class MotionRevisionTests(unittest.TestCase):
    def test_motion_can_start_immediately_without_quiet_start(self):
        segmenter = logic.MotionSegmenter()
        states = []
        for i in range(45):
            states.append(segmenter.update(i, moving_hand(i*.004), (640,480), i/30)[0])
        self.assertIn("SUSPECT", states)
        self.assertIn("CAPTURING", states)
        self.assertNotIn("REJECTED", states)
        self.assertEqual(segmenter.entries[0][0], 0)  # pre-roll preserves start

    def test_slow_monotonic_motion_is_not_mistaken_for_quiet(self):
        segmenter = logic.MotionSegmenter()
        for i in range(90):
            state, _, _ = segmenter.update(i,moving_hand(i*.003),(640,480),i/30)
            self.assertNotEqual(state,"COMPLETE")
        self.assertTrue(segmenter.active)

    def test_fast_fingertip_rotation_alone_is_not_identity_failure(self):
        segmenter = logic.MotionSegmenter()
        segmenter.update(0,moving_hand(),(640,480),0.)
        hand = moving_hand()
        hand[20].y += .20  # old max-tip-step guard failed; palm is unchanged
        state, reason, _ = segmenter.update(1,hand,(640,480),1/30)
        self.assertNotEqual(state,"REJECTED",reason)
        self.assertGreater(segmenter.last_diagnostics["tip_step_ratio"],.65)
        self.assertEqual(segmenter.last_diagnostics["palm_step_ratio"],0.)

    def test_rotating_hand_and_foreshortening_do_not_fail_individually(self):
        segmenter = logic.MotionSegmenter()
        reasons = []
        for i in range(50):
            raw = np.array([[p.x,p.y,p.z] for p in moving_hand()])
            wrist = raw[0].copy()
            angle = min(i,30)*.04
            rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            raw[:,:2] = (raw[:,:2]-wrist[:2]) @ rotation.T * (1-.012*min(i,30)) + wrist[:2]
            hand = [SimpleNamespace(x=x,y=y,z=z) for x,y,z in raw]
            state, reason, _ = segmenter.update(i,hand,(640,480),i/30)
            if state=="REJECTED": reasons.append(reason)
        self.assertEqual(reasons,[])

    def test_one_spike_in_palm_coordinates_is_rejected(self):
        segmenter = logic.MotionSegmenter()
        segmenter.update(0,moving_hand(),(640,480),0.)
        self.assertEqual(segmenter.update(1,moving_hand(.5),(640,480),1/30)[:2],
                         ("REJECTED","PALM_POSITION_DISCONTINUITY"))

    def test_short_missing_gap_is_skipped_without_inventing_entries(self):
        segmenter = logic.MotionSegmenter()
        for i in range(30):
            segmenter.update(i,moving_hand(i*.004),(640,480),i/30)
        before = len(segmenter.entries)
        self.assertEqual(segmenter.update(30,None,(640,480),1.)[0],"GAP")
        self.assertEqual(segmenter.update(31,None,(640,480),31/30)[0],"GAP")
        self.assertEqual(len(segmenter.entries),before)
        self.assertEqual(segmenter.update(32,moving_hand(32*.004),(640,480),32/30)[0],"CAPTURING")
        indices = [e[0] for e in segmenter.entries]
        self.assertNotIn(30,indices)
        self.assertNotIn(31,indices)
        self.assertEqual(indices[-1],32)

    def test_long_gap_rejects_and_missing_does_not_count_as_quiet_end(self):
        segmenter = logic.MotionSegmenter()
        for i in range(25): segmenter.update(i,moving_hand(i*.004),(640,480),i/30)
        states = [segmenter.update(i,None,(640,480),i/30)[0] for i in (25,26,27)]
        self.assertEqual(states,["GAP","GAP","REJECTED"])
        self.assertFalse(segmenter.active)

    def test_low_coverage_cannot_complete(self):
        segmenter = logic.MotionSegmenter()
        reasons=[]
        for i in range(120):
            hand = None if i<45 and i%3==1 else moving_hand(min(i,45)*.004)
            state,reason,_=segmenter.update(i,hand,(640,480),i/30)
            if state=="REJECTED": reasons.append(reason)
        self.assertIn("INSUFFICIENT_SEGMENT_COVERAGE",reasons)

    def test_eof_is_review_only_never_a_live_completion(self):
        segmenter=logic.MotionSegmenter()
        for i in range(40): segmenter.update(i,moving_hand(i*.004),(640,480),i/30)
        state,reason,entries=segmenter.review_at_eof()
        self.assertEqual(state,"REVIEW_REQUIRED")
        self.assertEqual(reason,"EOF_WITHOUT_OBSERVED_QUIET_END")
        self.assertGreaterEqual(len(entries),16)
        self.assertFalse(segmenter.active)

    def test_eof_with_too_few_valid_frames_is_rejected(self):
        segmenter=logic.MotionSegmenter()
        for i in range(10): segmenter.update(i,moving_hand(i*.006),(640,480),i/30)
        self.assertEqual(segmenter.review_at_eof()[0],"REJECTED")

    def test_all_24_static_letters_keep_hold_and_predictions(self):
        for label in "ABCDEFGHIKLMNOPQRSTUVWXY":
            with self.subTest(label=label):
                letters=FakePredictor(label,0.)
                coordinator=logic.AbcDynamicCoordinator(letters,FakePredictor("0",.5),Model())
                for i in range(130):
                    hand=moving_hand(.001*np.sin(i))
                    prediction,_=coordinator.process_hand(hand,hand,(480,640,3),i,i/30)
                    self.assertEqual(prediction.instant_label,label)
                self.assertEqual(coordinator.composer.text,label)
                self.assertFalse(coordinator.blocked)

    def test_coordinator_bridges_only_explicit_missing_not_ambiguity(self):
        coordinator=logic.AbcDynamicCoordinator(FakePredictor("I",0.),FakePredictor("0",.5),Model())
        for i in range(30):
            hand=moving_hand(i*.004)
            coordinator.process_hand(hand,hand,(480,640,3),i,i/30)
        self.assertTrue(coordinator.segmenter.active)
        prediction,confirmed=coordinator.process_hand(moving_hand(.12),None,(480,640,3),30,1.,dynamic_missing=True)
        self.assertEqual(prediction.instant_label,"I")
        self.assertIsNone(confirmed)
        self.assertEqual(coordinator.dynamic_state,"GAP")
        coordinator.process_hand(moving_hand(.124),None,(480,640,3),31,31/30)
        self.assertFalse(coordinator.segmenter.active)  # ambiguity is not bridged
        self.assertEqual(coordinator.suppression_reason,"")
        self.assertFalse(coordinator.blocked)

    def test_primary_tracking_gap_is_bounded_and_resumes_static(self):
        coordinator=logic.AbcDynamicCoordinator(FakePredictor("I",0.),FakePredictor("0",.5),Model())
        for i in range(30):
            hand=moving_hand(i*.004)
            coordinator.process_hand(hand,hand,(480,640,3),i,i/30)
        coordinator.missing_hand(1.,index=30)
        self.assertEqual(coordinator.dynamic_state,"GAP")
        coordinator.missing_hand(31/30,index=31)
        coordinator.missing_hand(32/30,index=32)
        self.assertEqual(coordinator.dynamic_state,"REJECTED")
        self.assertFalse(coordinator.blocked)
        self.assertEqual(coordinator.suppression_reason,"")

    def test_non_jz_translation_does_not_use_probability_as_acceptance(self):
        entries=[(i,moving_hand(i*.01),(640,480)) for i in range(25)]
        model=Model()
        with patch.object(model,"predict_proba",side_effect=AssertionError("no NONE model")):
            candidate,reason=logic.evaluate_segment(entries,model)
        self.assertIsNone(candidate)
        self.assertEqual(reason,"MOTION_OR_HANDSHAPE_NOT_SUPPORTED")

    def test_j_corner_can_occur_late_without_relaxing_angle_limit(self):
        features=np.zeros((16,69))
        shape=features[:,:63].reshape(16,21,3)
        for tip in (8,12,16,20):
            shape[:,tip-2,1]=-.5
            shape[:,tip,1]=-.4
        shape[:,20,1]=-1.
        for axis,values in enumerate(([0,0,1],[0,1,1.6])):
            features[:,67+axis]=np.interp(np.arange(16),[0,12,15],values)
        path=features[:,67:69]
        first,hook=path[9]-path[0],path[-1]-path[9]
        self.assertGreater(np.dot(first,hook),.6*np.linalg.norm(first)*np.linalg.norm(hook))
        self.assertEqual(logic.geometry_candidates(features),{"J"})

    def test_real_timeout_then_idle_restores_static_hold(self):
        letters=FakePredictor("A",0.)
        coordinator=logic.AbcDynamicCoordinator(letters,FakePredictor("0",.5),Model())
        coordinator.segmenter=logic.MotionSegmenter(logic.replace(logic.SegmentSettings(),max_seconds=.3))
        rejected=False
        for i in range(200):
            hand=moving_hand(min(i,40)*.004)
            prediction,_=coordinator.process_hand(hand,hand,(480,640,3),i,i/30)
            self.assertEqual(prediction.instant_label,"A")
            rejected |= "SEGMENT_TIMEOUT" in coordinator.status
        self.assertTrue(rejected)
        self.assertEqual(coordinator.composer.text,"A")
        self.assertFalse(coordinator.blocked)
        self.assertEqual(coordinator.suppression_reason,"")


class LifecycleTraceTests(unittest.TestCase):
    def setUp(self):
        self.coordinator = logic.AbcDynamicCoordinator(FakePredictor("A", 0.), FakePredictor("0", .5), Model())
        self.events = []
        self.coordinator.event_sink = self.events.append

    def observe(self, i, offset, *, video=True, image=True, now=None):
        raw = moving_hand(offset)
        mirrored = [SimpleNamespace(x=1-p.x, y=p.y, z=p.z) for p in raw]
        return camera.process_camera_observations(
            self.coordinator, SimpleNamespace(hand_landmarks=[mirrored] if video else []),
            SimpleNamespace(hand_landmarks=[raw] if image else []), (480, 640, 3), i,
            i/30 if now is None else now)

    def capture(self):
        for i in range(30):
            self.observe(i, i*.004)
        self.assertTrue(self.coordinator.segmenter.active)

    def test_static_detected_image_missing_gap_then_resume(self):
        self.capture()
        prediction, committed, _ = self.observe(30, .12, image=False)
        self.assertEqual(prediction.instant_label, "A")
        self.assertIsNone(committed)
        gap = self.events[-1]
        self.assertEqual((gap["previous_state"], gap["new_state"]), ("CAPTURING", "GAP"))
        self.assertTrue(gap["video_hand_detected"])
        self.assertFalse(gap["image_hand_detected"])
        self.assertEqual(gap["segment"]["consecutive_missing_frames"], 1)
        self.assertFalse(gap["segment"]["quiet_end_met"])
        self.observe(31, .124)
        self.assertEqual((self.events[-1]["previous_state"], self.events[-1]["new_state"]),
                         ("GAP", "CAPTURING"))
        self.assertNotIn(30, [entry[0] for entry in self.coordinator.segmenter.entries])

    def test_gap_exceeding_bound_preserves_rejection_metrics_and_resumes_static(self):
        self.capture()
        valid = len(self.coordinator.segmenter.entries)
        for i in range(30, 33):
            prediction, _, _ = self.observe(i, .12, image=False)
            self.assertEqual(prediction.instant_label, "A")
        rejected = self.events[-1]
        self.assertEqual(rejected["reason"], "MISSING_HAND_LIMIT")
        self.assertEqual(rejected["segment"]["valid_frames"], valid)
        self.assertEqual(rejected["segment"]["consecutive_missing_frames"], 3)
        self.assertGreater(rejected["segment"]["segment_duration_seconds"], 0)
        self.assertFalse(rejected["model_called"])
        self.assertFalse(self.coordinator.blocked)
        self.assertFalse(self.coordinator.suppressed)
        for i in range(33, 160):
            self.observe(i, .12)
        self.assertEqual(self.coordinator.composer.text, "A")

    def test_video_missing_image_valid_does_not_manufacture_dynamic_gap(self):
        self.capture()
        count = len(self.coordinator.segmenter.entries)
        self.observe(30, .12, video=False)
        self.assertEqual(self.coordinator.dynamic_state, "CAPTURING")
        self.assertEqual(len(self.coordinator.segmenter.entries), count+1)
        self.assertEqual(self.coordinator.segmenter.missing_frames, 0)
        self.assertTrue(self.coordinator.identity_unverified)
        self.assertEqual(self.events[-1]["event"], "OBSERVATION_CHANGE")
        self.assertFalse(self.events[-1]["video_hand_detected"])
        self.assertTrue(self.events[-1]["image_hand_detected"])
        self.assertEqual(self.coordinator.segmenter.entries[-1][0], 30)
        np.testing.assert_array_equal(logic.landmarks_to_array(self.coordinator.segmenter.entries[-1][1]),
                                      logic.landmarks_to_array(moving_hand(.12)))

    def test_video_missing_never_starts_unmatched_bout(self):
        self.observe(0, 0, video=False)
        self.assertFalse(self.coordinator.segmenter.active)
        self.assertEqual(self.events[-1]["reason"], "VIDEO_MISSING_NO_MATCHED_ACTIVE_BOUT")

    def test_valid_quiet_end_records_model_and_candidate_without_committing(self):
        with patch.object(logic, "geometry_candidates", return_value={"J"}):
            for i in range(100):
                self.observe(i, min(i, 30)*.004)
                if self.coordinator.jz_candidate:
                    break
        evaluation = [e for e in self.events if e["event"] == "SEGMENT_EVALUATION"][-1]
        self.assertTrue(evaluation["model_called"])
        self.assertEqual(evaluation["candidate"], "J")
        self.assertEqual(evaluation["feature_shape"], [16, 69])
        self.assertEqual(len(evaluation["selected_frame_indices"]), 16)
        self.assertTrue(evaluation["segment"]["quiet_end_met"])
        self.assertEqual(evaluation["segment"]["completion"], "OBSERVED_QUIET_END")
        self.assertGreaterEqual(evaluation["segment"]["valid_frames"], 16)
        self.assertEqual(self.coordinator.composer.text, "")
        self.assertFalse(self.coordinator.suppressed)

    def test_unverified_bout_cannot_auto_commit_even_with_explicit_opt_in(self):
        self.coordinator.enable_commits = True
        self.capture()
        self.observe(30, .12, video=False)
        with patch.object(logic, "geometry_candidates", return_value={"J"}):
            for i in range(31, 100):
                self.observe(i, .12)
                if self.coordinator.jz_candidate:
                    break
        evaluation = [e for e in self.events if e["event"] == "SEGMENT_EVALUATION"][-1]
        self.assertTrue(evaluation["identity_unverified"])
        self.assertEqual(evaluation["candidate"], "J")
        self.assertIn("IDENTITY_UNVERIFIED", self.coordinator.status)
        self.assertIsNone(self.coordinator.locked)
        self.assertFalse(self.coordinator.blocked)
        self.assertEqual(self.coordinator.composer.text, "")

    def test_geometry_rejection_trace_proves_model_not_called(self):
        with patch.object(self.coordinator.model, "predict_proba", side_effect=AssertionError("model called")):
            for i in range(100):
                self.observe(i, min(i, 30)*.004)
                if any(e["event"] == "SEGMENT_EVALUATION" for e in self.events):
                    break
        evaluation = [e for e in self.events if e["event"] == "SEGMENT_EVALUATION"][-1]
        self.assertFalse(evaluation["model_called"])
        self.assertEqual(evaluation["geometry_candidates"], [])
        self.assertEqual(evaluation["reason"], "MOTION_OR_HANDSHAPE_NOT_SUPPORTED")
        self.assertFalse(self.coordinator.suppressed)

    def test_timeout_trace_has_quiet_window_evidence_and_static_recovers(self):
        self.coordinator.segmenter = logic.MotionSegmenter(logic.replace(logic.SegmentSettings(), max_seconds=.3))
        for i in range(50):
            self.observe(i, i*.004)
            if self.events and self.events[-1]["reason"] == "SEGMENT_TIMEOUT":
                break
        event = self.events[-1]
        self.assertEqual((event["previous_state"], event["new_state"]), ("CAPTURING", "REJECTED"))
        self.assertGreater(event["segment"]["segment_duration_seconds"], .3)
        self.assertFalse(event["segment"]["quiet_end_met"])
        self.assertFalse(self.coordinator.suppressed)

    def test_secondary_exception_is_not_a_no_hand_observation(self):
        self.capture()
        mirrored = [SimpleNamespace(x=1-p.x, y=p.y, z=p.z) for p in moving_hand(.12)]
        prediction, _, _ = camera.process_camera_observations(
            self.coordinator, SimpleNamespace(hand_landmarks=[mirrored]), None, (480,640,3), 30, 1.,
            image_error="RuntimeError: failed")
        self.assertEqual(prediction.instant_label, "A")
        self.assertEqual(self.events[-1]["reason"], "IMAGE_ERROR: RuntimeError: failed")
        self.assertIsNone(self.events[-1]["image_hand_detected"])
        self.assertFalse(self.coordinator.suppressed)

    def test_mismatch_and_two_hands_remain_distinct_rejections(self):
        self.capture()
        camera.process_camera_observations(
            self.coordinator, SimpleNamespace(hand_landmarks=[moving_hand(.5)]),
            SimpleNamespace(hand_landmarks=[moving_hand(.12)]), (480,640,3), 30, 1.)
        self.assertEqual(self.events[-1]["reason"], "ORIENTATION_CORRESPONDENCE_MISMATCH")
        camera.process_camera_observations(
            self.coordinator, SimpleNamespace(hand_landmarks=[moving_hand(.5)]),
            SimpleNamespace(hand_landmarks=[moving_hand(.12), moving_hand(.2)]), (480,640,3), 31, 31/30)
        self.assertEqual(self.events[-1]["reason"], "IMAGE_MULTIPLE_HANDS")

    def test_camera_runs_image_even_when_video_has_no_hand(self):
        capture = MagicMock()
        capture.read.return_value = (True, np.zeros((480,640,3), dtype=np.uint8))
        video_task, image_task = MagicMock(), MagicMock()
        video_task.__enter__.return_value = video_task
        image_task.__enter__.return_value = image_task
        video_task.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[])
        image_task.detect.return_value = SimpleNamespace(hand_landmarks=[synthetic_hand()])
        with (patch.object(camera.cv2, "VideoCapture", return_value=capture),
              patch.object(camera, "create_hand_landmarker", side_effect=[video_task, image_task]),
              patch.object(camera.cv2, "namedWindow"), patch.object(camera.cv2, "setMouseCallback"),
              patch.object(camera.cv2, "imshow"), patch.object(camera.cv2, "waitKey", return_value=ord("q")),
              patch.object(camera.cv2, "destroyAllWindows")):
            self.assertEqual(camera.run_camera(self.coordinator, 0), 0)
        video_task.detect_for_video.assert_called_once()
        image_task.detect.assert_called_once()
        self.assertFalse(self.coordinator.frame_context["video_hand_detected"])
        self.assertTrue(self.coordinator.frame_context["image_hand_detected"])

    def test_primary_ambiguity_cannot_be_bridged_as_secondary_only(self):
        self.capture()
        hand = synthetic_hand()
        camera.process_camera_observations(
            self.coordinator, SimpleNamespace(hand_landmarks=[hand, hand]),
            SimpleNamespace(hand_landmarks=[hand]), (480,640,3), 30, 1.)
        self.assertEqual(self.events[-1]["reason"], "HAND_IDENTITY_UNCERTAIN")
        self.assertFalse(self.coordinator.segmenter.active)

    def test_events_are_opt_in_and_not_emitted_for_every_capturing_frame(self):
        self.capture()
        count = len(self.events)
        for i in range(30, 40):
            self.observe(i, i*.004)
        self.assertEqual(len(self.events), count)
        self.coordinator.event_sink = None
        self.observe(40, .16, image=False)
        self.assertEqual(len(self.events), count)

    def test_check_only_debug_never_creates_trace_or_opens_camera(self):
        with (patch.object(camera.cv2, "VideoCapture", side_effect=AssertionError("camera")),
              patch.object(camera, "create_hand_landmarker", side_effect=AssertionError("MediaPipe")),
              patch.object(camera, "EventTrace", side_effect=AssertionError("trace file")),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(camera.main(["--check-only", "--debug-jz-events"]), 0)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["event_trace_enabled"])
        self.assertFalse(payload["jz_commits_enabled"])
        self.assertFalse(payload["camera_opened"])

    def test_trace_writer_is_buffered_exclusive_and_not_terminal_output(self):
        from pathlib import Path
        file = io.StringIO()
        with (patch.object(Path, "mkdir") as mkdir,
              patch.object(Path, "open", return_value=file) as opened,
              patch("builtins.print", side_effect=AssertionError("per-event Terminal output"))):
            writer = camera.EventTrace(camera.ROOT / "data/experiments/dynamic_jz_v1/metadata/test_trace.jsonl")
            writer({"event": "STATE_TRANSITION", "frame_id": 5})
            payload = json.loads(file.getvalue())
            self.assertEqual(payload["frame_id"], 5)
            self.assertIn("timestamp_utc", payload)
            opened.assert_called_once_with("x", encoding="utf-8", buffering=65536)
            mkdir.assert_called_once()


if __name__ == "__main__":
    unittest.main()
