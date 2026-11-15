"""Lightweight diagnostics regressions: no camera, decoding, MediaPipe or writes.

The four saved training sequences test only classifier forwarding. Cached J
landmarks were recorded with num_hands=2, NOT the live num_hands=1 task; their
replay tests detector equivalence only, not camera/generalization accuracy.
"""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

from app.inference import dynamic_jz_auto_camera_experiment as auto
from app.inference import dynamic_jz_auto_camera_experiment_v1 as before
from app.inference.dynamic_jz_auto_diagnostics import (
    AttemptStatistics, DebugTrace, PredictionProbe, failure_stage, handshape_evidence,
)
from tests.test_dynamic_jz_auto_camera_experiment import Model, hand_at


def drive(session, observations):
    for index, hand, size, now in observations:
        session.update(index, hand, size, now, hand_count=1 if hand else 0)
        session.process_pending(now)


def observations(offsets, missing=()):
    return [(i, None if i in missing else hand_at(value), (640,480), i/30)
            for i, value in enumerate(offsets)]


class DetectorEquivalenceTests(unittest.TestCase):
    def compare(self, rows, settings=None, gate=None):
        old_model, new_model = Model(), Model()
        settings = settings or auto.SegmentSettings()
        events = []
        old = before.AutoJzSession(old_model, settings, result_seconds=4.)
        new = auto.AutoJzSession(new_model, settings, event_sink=events.append, collection_mode="quiet-end")
        if gate is None:
            drive(old, rows)
            drive(new, rows)
        else:
            with patch("app.inference.automatic_jz_experiment.geometry_candidates", return_value=gate):
                drive(old, rows)
                drive(new, rows)
        self.assertEqual(list(old.transitions), list(new.transitions))
        self.assertEqual(old.state, new.state)
        self.assertEqual(len(old.results), len(new.results))
        for a, b in zip(old.results, new.results):
            for key in ("status", "reason", "frame_id", "model_result", "selected_frame_indices"):
                self.assertEqual(a.get(key), b.get(key))
        self.assertEqual(len(old_model.inputs), len(new_model.inputs))
        for a, b in zip(old_model.inputs, new_model.inputs):
            np.testing.assert_array_equal(a, b)
        return new, events

    def test_stationary_and_jitter_equivalent_no_false_result(self):
        for offsets in ([0.]*100, [.0002*np.sin(i) for i in range(100)]):
            session, _ = self.compare(observations(offsets))
            self.assertEqual(session.attempts.counts["TOTAL_ATTEMPTS"], 0)
            self.assertIsNone(session.last_result)

    def test_small_non_jz_translation_still_rejected_by_geometry(self):
        session, _ = self.compare(observations([min(i,30)*.004 for i in range(100)]))
        self.assertEqual(session.last_result["status"], "REJECTED")
        self.assertEqual(session.attempts.counts["MODEL_INVOKED"], 0)

    def test_fast_start_no_quiet_prerequisite_preserves_pre_roll(self):
        session, events = self.compare(observations([i*.004 for i in range(30)]))
        self.assertEqual(session.state, "CAPTURING")
        self.assertEqual(session.segmenter.entries[0][0], 0)
        transitions = [e["new_state"] for e in events if e["event"] == "DETECTOR_TRANSITION"]
        self.assertIn("SUSPECT", transitions)
        self.assertIn("CAPTURING", transitions)
        self.assertEqual(session.attempts.counts["CAPTURE_COMPLETED"], 0)

    def test_wrist_rotation_does_not_change_detector_behavior(self):
        rows = []
        for i in range(60):
            points = np.array([[p.x,p.y,p.z] for p in hand_at()])
            angle = min(i,30)*.015
            rotation = np.array([[np.cos(angle),-np.sin(angle)], [np.sin(angle),np.cos(angle)]])
            points[:,:2] = (points[:,:2]-points[0,:2]) @ rotation.T + points[0,:2]
            hand = [SimpleNamespace(x=x,y=y,z=z) for x,y,z in points]
            rows.append((i,hand,(640,480),i/30))
        self.compare(rows)

    def test_short_tracking_gap_bounded_no_synthetic_frame(self):
        session, events = self.compare(observations([i*.004 for i in range(35)], missing=(30,)))
        self.assertNotIn(30, [entry[0] for entry in session.segmenter.entries])
        states = [e["new_state"] for e in events if e["event"] == "DETECTOR_TRANSITION"]
        self.assertIn("GAP", states)
        self.assertEqual(states[-1], "CAPTURING")
        self.assertEqual(session.attempts.counts["TOTAL_ATTEMPTS"], 1)
        self.assertIsNone(session.last_result)

    def test_unfinished_motion_does_not_stop_at_frame_16(self):
        session, _ = self.compare(observations([i*.004 for i in range(60)]))
        self.assertGreater(session.valid_frames, 16)
        self.assertEqual(session.state, "CAPTURING")
        self.assertEqual(session.attempts.counts["MODEL_INVOKED"], 0)

    def test_quiet_end_model_and_result_are_traced_once(self):
        session, events = self.compare(observations([min(i,30)*.004 for i in range(100)]), gate={"Z"})
        counts = session.attempts.snapshot()
        for key in ("TOTAL_ATTEMPTS", "CAPTURE_STARTED", "CAPTURE_COMPLETED", "MODEL_INVOKED", "RESULT_CREATED"):
            self.assertEqual(counts[key], 1, key)
        self.assertEqual(counts["RESULT_DISPLAYED"], 0)  # Creation is not a UI acknowledgement.
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds.count("MODEL_INVOKED"), 1)
        self.assertEqual(kinds.count("ATTEMPT_FINISHED"), 1)
        self.assertLess(kinds.index("INFERENCE_BEGIN"), kinds.index("MODEL_INVOKED"))
        self.assertLess(kinds.index("MODEL_INVOKED"), kinds.index("INFERENCE_END"))
        self.assertTrue(session.last_result["endpoint_verified"])
        self.assertEqual(session.last_result["pre_roll_first_original_frame"], 0)
        self.assertIn("handshape_evidence", session.last_result)
        json.dumps(events, allow_nan=False)

    def test_timeout_remains_rejected_without_inference(self):
        settings = auto.replace(auto.SegmentSettings(), max_seconds=.4)
        session, _ = self.compare(observations([i*.004 for i in range(25)]), settings=settings)
        self.assertEqual(session.attempts.counts["TIMEOUT"], 1)
        self.assertEqual(session.attempts.counts["MODEL_INVOKED"], 0)
        self.assertEqual(session.last_result["failure_stage"], "END_DETECTION")

    def test_held_endpoint_no_repeat_and_new_capture_has_no_cooldown(self):
        offsets = [min(i,30)*.004 if i < 60 else .12+min(i-60,30)*.004 for i in range(180)]
        session, _ = self.compare(observations(offsets), gate={"Z"})
        self.assertEqual(session.attempts.counts["TOTAL_ATTEMPTS"], 2)
        self.assertEqual(session.attempts.counts["RESULT_CREATED"], 2)
        captures = [e for e in session.transitions if e["new_state"] == "CAPTURING"]
        self.assertLess(captures[1]["timestamp_seconds"], session.results[0]["frame_id"]/30+4.)


class DiagnosticCounterTests(unittest.TestCase):
    def test_persistent_rejection_counts_once_not_per_frame(self):
        stats = AttemptStatistics()
        stats.start(10,1.)
        stats.start(11,1.1)
        for _ in range(100):
            stats.finish({"status":"REJECTED", "reason":"SEGMENT_TIMEOUT"})
        self.assertEqual(stats.counts["TOTAL_ATTEMPTS"], 1)
        self.assertEqual(stats.counts["TIMEOUT"], 1)
        self.assertEqual(stats.counts["REJECTED"], 1)
        self.assertEqual(stats.reasons, {"SEGMENT_TIMEOUT":1})
        stats.start(120,4.)
        self.assertEqual(stats.counts["TOTAL_ATTEMPTS"], 2)

    def test_model_calls_and_result_display_are_separate_and_unique(self):
        stats = AttemptStatistics()
        stats.start(10,1.)
        stats.complete()
        stats.complete()
        model = Model()
        probe = PredictionProbe(model, stats.model_invoked)
        values = np.zeros((1,1104))
        np.testing.assert_array_equal(probe.predict_proba(values), [[.07,.93]])
        np.testing.assert_array_equal(model.inputs[0], values)
        np.testing.assert_array_equal(probe.classes_, model.classes_)
        result = {"status":"RESULT", "reason":"BINARY_EXPERIMENT_ONLY"}
        stats.finish(result)
        self.assertEqual(stats.counts["RESULT_DISPLAYED"], 0)
        self.assertTrue(stats.displayed(result))
        self.assertFalse(stats.displayed(result))
        self.assertEqual(stats.counts["MODEL_INVOKED"], 1)
        self.assertEqual(stats.counts["CAPTURE_COMPLETED"], 1)
        self.assertEqual(stats.counts["RESULT_DISPLAYED"], 1)

    def test_idle_errors_and_suspect_are_not_labelled_failed_attempts(self):
        events = []
        session = auto.AutoJzSession(Model(), auto.replace(auto.SegmentSettings(), start_frames=50),
                                    event_sink=events.append, collection_mode="quiet-end")
        drive(session, observations([i*.004 for i in range(20)]))
        self.assertEqual(session.attempts.counts["TOTAL_ATTEMPTS"], 0)
        self.assertTrue(any(e.get("new_state")=="SUSPECT" for e in events))
        for i in range(20,40):
            session.update(i,None,(640,480),i/30,detection_error="API failure",hand_count=0)
        self.assertEqual(session.attempts.counts["REJECTED"], 0)
        self.assertIn("NOT_USER_LABELLED", session.attempts.snapshot()["attempt_definition"])

    def test_shape_evidence_explains_gate_without_overriding_it(self):
        features = np.zeros((16,69))
        shape = features[:,:63].reshape(16,21,3)
        shape[:,18,1] = 1.
        shape[:,20,1] = 2.
        for tip in (8,12,16):
            shape[:,tip-2,1] = 1.
            shape[:,tip,1] = .1
        evidence = handshape_evidence(features)
        self.assertTrue(evidence["J"]["existing_pose_gate_met"])
        self.assertEqual(failure_stage("MOTION_OR_HANDSHAPE_NOT_SUPPORTED",evidence), "TRAJECTORY_GATE")
        self.assertEqual(auto.evaluate_segment.__module__, "app.inference.automatic_jz_experiment")
        evidence = handshape_evidence(np.zeros((16,69)))
        self.assertEqual(failure_stage("MOTION_OR_HANDSHAPE_NOT_SUPPORTED",evidence), "HANDSHAPE_GATE")
        self.assertEqual(failure_stage("MISSING_HAND_LIMIT"), "TRACKING")
        self.assertEqual(failure_stage("TOO_SHORT"), "FRAME_OR_COVERAGE_CHECK")


class DebugSafetyAndUiTests(unittest.TestCase):
    def test_check_only_debug_never_creates_trace_or_opens_task_camera(self):
        with (patch.object(auto,"DebugTrace",side_effect=AssertionError("write")),
              patch.object(auto.cv2,"VideoCapture",side_effect=AssertionError("camera/video")),
              patch.object(auto,"create_hand_landmarker",side_effect=AssertionError("MediaPipe")),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(auto.main(["--check-only","--debug"]),0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["files_written"],0)
        self.assertEqual(result["result_seconds"],4.)
        self.assertTrue(result["debug_enabled"])

    def test_bad_trace_paths_fail_before_loading_model_or_writing(self):
        with (patch.object(auto,"load_experiment_model",side_effect=AssertionError("model loaded")),
              patch.object(Path,"mkdir",side_effect=AssertionError("mkdir")),
              contextlib.redirect_stderr(io.StringIO())):
            for path in (str(auto.ROOT/"outside.jsonl"),
                         str(auto.ROOT/"data/experiments/dynamic_jz_v1/trace.csv")):
                with self.assertRaises(SystemExit):
                    auto.main(["--check-only","--debug","--debug-trace",path])

    def test_debug_trace_is_exclusive_buffered_and_json_finite(self):
        root = auto.ROOT/"data/experiments/dynamic_jz_v1"
        opened = mock_open()
        with patch.object(Path,"mkdir"), patch.object(Path,"open",opened):
            with DebugTrace(root/"unused_test_trace.jsonl",root) as trace:
                trace({"event":"TEST", "timestamp_monotonic":1.})
                trace({"event":"SESSION_SUMMARY"})
                with self.assertRaises(ValueError):
                    trace({"bad":float("nan")})
        opened.assert_called_once_with("x",encoding="utf-8",buffering=65536)
        opened().close.assert_called_once()
        opened().flush.assert_called_once()
        record = json.loads(opened().write.call_args_list[0].args[0])
        self.assertIn("timestamp_utc",record)

    def test_result_visible_four_seconds_not_inference_or_repeat_count(self):
        events = []
        session = auto.AutoJzSession(Model(),event_sink=events.append)
        session.attempts.start(10,1.)
        result = session._finish("RESULT",40,2.,"BINARY_EXPERIMENT_ONLY",{
            "model_result":{"status":"PREDICTED","prediction":"Z","probability_J":.07,"probability_Z":.93}})
        self.assertEqual(auto.display_lines(session,5.99)["label"],"Z")
        self.assertEqual(auto.display_lines(session,6.)["label"],"-")
        for now in (2.,2.1,3.,5.99):
            session.note_displayed(now)
        self.assertEqual(session.attempts.counts["RESULT_DISPLAYED"],1)
        self.assertEqual(sum(e["event"]=="RESULT_DISPLAYED" for e in events),1)
        self.assertEqual(result["attempt_id"],1)
        raw = np.arange(4*6*3,dtype=np.uint8).reshape(4,6,3)
        np.testing.assert_array_equal(auto.render(raw,session,3.)[:4,:6],raw)

    def test_trace_error_cannot_prevent_camera_release(self):
        session = auto.AutoJzSession(Model(),event_sink=lambda _: (_ for _ in ()).throw(OSError("disk")))
        task, capture = MagicMock(), MagicMock()
        task.__enter__.return_value = task
        capture.read.return_value = (False,None)
        with (patch.object(auto,"create_hand_landmarker",return_value=task),
              patch.object(auto.cv2,"VideoCapture",return_value=capture),
              patch.object(auto.cv2,"destroyAllWindows") as closed):
            with self.assertRaises(OSError):
                auto.run_camera(session,0)
        capture.release.assert_called_once()
        closed.assert_called_once()


class SavedEvidenceTests(unittest.TestCase):
    def test_cached_j_detector_before_after_identical_no_mediapipe(self):
        path = auto.ROOT/"data/experiments/dynamic_jz_v1/review/j246_j252_detector_revision_v1/frame_measurements.json"
        cache = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(cache["no_flip"])
        self.assertEqual(cache["num_hands"],2)  # Explicitly NOT a new live camera audit.
        self.assertEqual({v["video"] for v in cache["videos"]},{"J/246.avi","J/252.avi"})
        for video in cache["videos"]:
            with self.subTest(video=video["video"]):
                rows = []
                for frame in video["frames"]:
                    hand = ([SimpleNamespace(x=x,y=y,z=z) for x,y,z in frame["raw_xyz"]]
                            if frame["raw_xyz"] is not None else None)
                    rows.append((frame["index"],hand,tuple(frame["size"]),frame["index"]/video["fps"]))
                helper = DetectorEquivalenceTests()
                helper.compare(rows)

    def test_four_saved_training_sequences_classifier_forwarding_identical(self):
        path = auto.ROOT/"data/experiments/dynamic_jz_v1/processed/signnteam_jz_baseline_b_v1.jsonl"
        records = {}
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                # Only deserialize the four requested records, not other features.
                if not any('/'+video+'"' in line for video in auto.REPLAY_VIDEOS):
                    continue
                row = json.loads(line)
                video = "/".join(row["source_path"].split("/")[-2:])
                records[video] = row
                if len(records)==4:
                    break
        self.assertEqual(set(records),set(auto.REPLAY_VIDEOS))
        model,_ = auto.load_experiment_model()
        calls = []
        probe = PredictionProbe(model,lambda: calls.append(True))
        for video,row in records.items():
            with self.subTest(video=video):
                values = np.asarray(row["features"],dtype=float)
                self.assertEqual(values.shape,(16,69))
                flat = values.reshape(1,1104,order="C")
                old_probability = model.predict_proba(flat)
                new_probability = probe.predict_proba(flat)
                np.testing.assert_array_equal(old_probability,new_probability)
                self.assertEqual(model.classes_[np.argmax(new_probability)],row["label"])
        self.assertEqual(len(calls),4)


if __name__ == "__main__":
    unittest.main()
