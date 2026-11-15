"""Standalone EXPERIMENTAL J/Z auto camera: no SPACE, no text commits, no NONE.

Default first16 mode now requires post-onset motion evidence, at least 16 real
valid frames and minimum capture duration, then uniformly samples the segment.
This is NOT a NONE classifier or ASL verifier. --collection-mode quiet-end retains the
previous whole-bout experiment. Only opt-in debug writes temporary event JSONL.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.feature_extraction.hand_landmarks import create_hand_landmarker, landmarks_to_array  # noqa: E402
from app.inference.automatic_jz_experiment import MotionSegmenter, SegmentSettings, evaluate_segment  # noqa: E402
from app.inference.dynamic_jz_camera_test import (  # noqa: E402
    EXPERIMENT_DIR, TASK_PATH, _sha256, _synthetic_feature_check,
    load_experiment_model, predict_attempt,
)
from ai.feature_extraction.dynamic_sequence import build_sequence_features, sample_valid_frame_indices  # noqa: E402
from app.inference.dynamic_jz_auto_diagnostics import (  # noqa: E402
    AttemptStatistics, DebugTrace, PredictionProbe, handshape_evidence, failure_stage, readable_reason,
)
from app.inference.first16_jz_segmenter import First16Segmenter, MotionGateSettings  # noqa: E402

WINDOW = "EXPERIMENTAL automatic J/Z - no NONE model"
RAW_ROOT = ROOT / "data/raw/dynamic/external/signnteam_jz/SigNN Video Data"
REPLAY_VIDEOS = ("J/246.avi", "J/252.avi", "Z/278.avi", "Z/189.avi")


class AutoJzSession:
    """Motion collection independent of camera/UI; one prediction per bout.

    Motion-gated first16 re-arms on observed quiet/release to avoid repeated windows from one
    ongoing gesture. Quiet-end mode retains fresh anchoring after completion.
    Neither mode uses result display duration as a cooldown.
    """
    def __init__(self, model, settings=None, *, result_seconds=4., predict_on_timeout=False,
                 event_sink=None, collection_mode="first16", motion_gate=MotionGateSettings()):
        if not np.isfinite(result_seconds) or result_seconds <= 0:
            raise ValueError("result_seconds must be finite and positive")
        if collection_mode not in ("first16", "quiet-end"):
            raise ValueError("Unknown collection mode")
        if collection_mode == "first16" and predict_on_timeout:
            raise ValueError("first16 never predicts an incomplete timeout segment")
        if settings is None:
            settings = SegmentSettings()
            if collection_mode == "first16":
                settings = replace(settings, max_missing_frames=6, max_frame_gap_seconds=.4)
        self.collection_mode = collection_mode
        self.attempts = AttemptStatistics()
        self.event_sink = event_sink
        self.frame_context = {}
        self.last_sample_time = -float("inf")
        self.last_endpoint_reason = None
        self.last_detector_state = "READY"
        self.inference_timestamp = None
        self.model = PredictionProbe(model, self._model_invoked)
        self.segmenter = (First16Segmenter(settings,motion_gate) if collection_mode == "first16"
                          else MotionSegmenter(settings))
        self.result_seconds = result_seconds
        self.predict_on_timeout = predict_on_timeout
        self.state = "READY"
        self.reason = ""
        self.valid_frames = 0
        self.pending = None
        self.last_result = None
        self.visible_until = -float("inf")
        self.transitions = deque(maxlen=512)
        self.results = deque(maxlen=64)

    def _emit(self, event, now, **details):
        if self.event_sink:
            metrics = (self.segmenter.trace_metrics(now) if self.segmenter.active
                       else dict(self.segmenter.last_diagnostics))
            self.event_sink({"event": event, "timestamp_monotonic": now, **self.frame_context,
                             "attempt_id": self.attempts.active["attempt_id"] if self.attempts.active else None,
                             "state": self.state, "current_state": details.get("new_state", self.state),
                             "valid_frame_count": metrics.get("valid_frames", self.valid_frames),
                             "missing_gap_count": metrics.get("consecutive_missing_frames", 0),
                             "collection_mode": self.collection_mode, "segment": metrics, **details})

    def _model_invoked(self):
        self.attempts.model_invoked()
        self._emit("MODEL_INVOKED", self.inference_timestamp)

    def _state(self, state, index, now, reason=""):
        if state == "CAPTURING":
            self.attempts.start(index, now)
        if state == "PROCESSING" and self.pending and self.pending[2] in (
                "OBSERVED_QUIET_END", "FIRST_16_VALID_FRAMES", "MOTION_GATED_VALID_WINDOW"):
            self.attempts.complete()
        if state != self.state or reason != self.reason:
            self.transitions.append({"frame_id": index, "timestamp_seconds": now,
                                     "previous_state": self.state, "new_state": state, "reason": reason})
            self._emit("STATE_TRANSITION", now, previous_state=self.state, new_state=state, reason=reason)
        self.state, self.reason = state, reason

    def _finish(self, status, index, now, reason, diagnostics=None):
        result = {"status": status, "frame_id": index, "reason": reason,
                  "model_called": False, "model_result": None, **(diagnostics or {})}
        result["failure_stage"] = failure_stage(reason, result.get("handshape_evidence"))
        result["readable_reason"] = readable_reason(reason, result["failure_stage"])
        counted = self.attempts.finish(result)
        self.last_result = result
        self.results.append(result)
        self.visible_until = now + self.result_seconds
        self._state(status, index, now, reason)
        if counted is not None:
            self._emit("ATTEMPT_FINISHED", now, result=result, statistics=self.attempts.snapshot())
        return result

    def reject(self, reason, index, now):
        self.segmenter.reset()
        self.pending = None
        self.valid_frames = 0
        return self._finish("REJECTED", index, now, reason)

    def update(self, index, hand, size, now, *, detection_error="", hand_count=None):
        invalid_landmark_reason = ""
        if self.collection_mode == "first16":
            if detection_error.startswith("INVALID_LANDMARK_FRAME:"):
                invalid_landmark_reason, detection_error = detection_error, ""
                hand = None
            if hand is not None:
                try:
                    landmarks_to_array(hand)
                except (ValueError, TypeError, AttributeError) as exc:
                    invalid_landmark_reason = "INVALID_LANDMARK_FRAME: " + str(exc)
                    hand = None  # Same as a missing frame, never added to features.
        previous = (self.frame_context.get("hand_count"), self.frame_context.get("detection_error"))
        self.frame_context = {"frame_id": index, "timestamp": now, "hand_count": hand_count,
                              "hand_detected": hand is not None, "detection_error": detection_error,
                              "invalid_landmark_reason": invalid_landmark_reason}
        if (hand_count, detection_error) != previous:
            self._emit("OBSERVATION_CHANGE", now)
        result = self._update(index, hand, size, now, detection_error=detection_error, hand_count=hand_count)
        detector_state = ("END_DETECTED" if self.pending else
                          "WAIT_REARM" if getattr(self.segmenter, "waiting_for_rearm", False) else
                          "GAP" if self.segmenter.active and self.segmenter.missing_frames else
                          "CAPTURING" if self.segmenter.active else
                          "SUSPECT" if self.segmenter.start_count else "READY")
        if detector_state != self.last_detector_state:
            self._emit("DETECTOR_TRANSITION", now, previous_state=self.last_detector_state,
                       new_state=detector_state, start_count=self.segmenter.start_count,
                       rolling_buffer_frames=len(self.segmenter.history))
            self.last_detector_state = detector_state
        endpoint_reason = self.segmenter.last_diagnostics.get("quiet_end_reason")
        if endpoint_reason != self.last_endpoint_reason:
            self._emit("ENDPOINT_EVIDENCE_CHANGE", now, endpoint_reason=endpoint_reason)
            self.last_endpoint_reason = endpoint_reason
        if self.event_sink and now-self.last_sample_time >= .5:
            self._emit("MOTION_SAMPLE", now, detector_state=("CAPTURING" if self.segmenter.active else
                       "SUSPECT" if self.segmenter.start_count else "READY"),
                       start_count=self.segmenter.start_count,
                       rolling_buffer_frames=len(self.segmenter.history),
                       motion_score=self.segmenter.motion_score,
                       settings=asdict(self.segmenter.settings))
            self.last_sample_time = now
        return result

    def _update(self, index, hand, size, now, *, detection_error="", hand_count=None):
        if self.pending is not None:
            raise RuntimeError("Process the pending bout before consuming another frame")
        if detection_error:
            return self.reject("DETECTION_ERROR: " + detection_error, index, now)
        if hand_count is not None and hand_count > 1:
            return self.reject("MULTIPLE_HANDS_IDENTITY_UNVERIFIED", index, now)
        if (hand is None and not self.segmenter.active
                and not getattr(self.segmenter, "waiting_for_rearm", False)):
            # An empty READY camera is not a failed gesture. Forget an idle or
            # partially debounced anchor, never mix pre-roll across absence.
            self.segmenter.reset()
            self.valid_frames = 0
            if self.last_result is None or now >= self.visible_until:
                self._state("READY", index, now, "WAITING_FOR_HAND")
            return None
        # Keep a reference, not a copy on every frame. On timeout the existing
        # detector resets its buffer; this reference retains the REAL bout only.
        prior_entries = self.segmenter.entries
        try:
            state, reason, entries = self.segmenter.update(index, hand, size, now)
        except (ValueError, TypeError, AttributeError) as exc:
            return self.reject("INVALID_LANDMARK_OR_TIME: " + str(exc), index, now)
        metrics = (dict(self.segmenter.last_diagnostics) if state in ("COMPLETE", "REJECTED")
                   or reason == "STATIC_OR_ENTRY_MOTION_ONLY"
                   else self.segmenter.trace_metrics(now))
        self.valid_frames = int(metrics.get("valid_frames", 0))
        if state == "COMPLETE":
            endpoint = metrics.get("completion", "OBSERVED_QUIET_END")
            if endpoint in ("FIRST_16_VALID_FRAMES", "MOTION_GATED_VALID_WINDOW"):
                # Long idle pre-roll may already contain 16 frames when onset
                # debounce passes. Still count exactly one opened attempt.
                self.attempts.start(index, now)
            self.pending = (entries, index, endpoint, metrics)
            self._state("PROCESSING", index, now, endpoint if self.collection_mode == "first16" else "")
        elif state == "REJECTED":
            if reason == "SEGMENT_TIMEOUT" and self.predict_on_timeout:
                coverage = (len(prior_entries)/(prior_entries[-1][0]-prior_entries[0][0]+1)
                            if prior_entries else 0.)
                if len(prior_entries) >= 16 and coverage >= self.segmenter.settings.min_valid_fraction:
                    self.pending = (prior_entries, index, "TIMEOUT_ENDPOINT_UNVERIFIED", metrics)
                    self._state("PROCESSING", index, now, "TIMEOUT_DIAGNOSTIC_ONLY")
                    return None
            if self.collection_mode == "first16":
                self.valid_frames = 0
            self._finish("REJECTED", index, now, reason, {"segment": metrics})
        elif state in ("CAPTURING", "GAP") and self.segmenter.active:
            self._state("CAPTURING", index, now, "TRACKING_GAP_NOT_AN_ENDPOINT" if state == "GAP" else "")
        elif state == "WAIT_REARM":
            self._state("WAIT_REARM", index, now, reason)
        elif reason == "REARMED_AFTER_STILL_HAND_OR_RELEASE":
            self._state("READY", index, now, reason)
        elif reason == "STATIC_OR_ENTRY_MOTION_ONLY":
            # No J/Z result or lingering REJECTED UI for a hand entry/static pose.
            self.valid_frames = 0
            cancelled = {"status":"REJECTED", "reason":reason}
            if self.attempts.finish(cancelled) is not None:
                self._emit("ATTEMPT_FINISHED",now,result=cancelled,statistics=self.attempts.snapshot())
            self._state("READY",index,now,reason)
        else:
            if self.last_result is None or now >= self.visible_until:
                self._state("READY", index, now, "MOTION_DEBOUNCE" if state == "SUSPECT" else "")
        return None

    def process_pending(self, now):
        if self.pending is None:
            return None
        entries, index, endpoint, metrics = self.pending
        self.pending = None  # No repeated invocation for the same movement.
        diagnostics = {"model_called": False, "model_result": None, "geometry_candidates": [],
                       "endpoint": endpoint, "endpoint_verified": endpoint == "OBSERVED_QUIET_END",
                       "segment": metrics}
        self.inference_timestamp = now
        self._emit("INFERENCE_BEGIN", now, endpoint=endpoint, valid_frames=len(entries))
        if endpoint in ("FIRST_16_VALID_FRAMES", "MOTION_GATED_VALID_WINDOW"):
            # ORIGINAL uniform valid-frame sampler, 16x69 builder and pipeline.
            # Motion evidence is not an ASL verifier or a substitute NONE class.
            try:
                if len(entries) < 16:
                    raise ValueError("INSUFFICIENT_VALID_LANDMARK_FRAMES")
                result = predict_attempt(entries, index+1, self.model)
            except (ValueError, TypeError, AttributeError, RuntimeError, IndexError, KeyError) as exc:
                diagnostics.update(model_called=(self.attempts.active or {}).get("invoked", False))
                self._emit("INFERENCE_END", now, diagnostics=diagnostics, status="REJECTED", reason=str(exc))
                return self._finish("REJECTED", index, now, "INVALID_SEQUENCE: " + str(exc), diagnostics)
            diagnostics.update(model_result=result, model_called=(self.attempts.active or {}).get("invoked", False),
                               feature_shape=[16,69] if result["status"] == "PREDICTED" else None,
                               selected_frame_indices=result["selected_original_frame_indices"],
                               geometry_gate="MOTION_EVIDENCE_NOT_ASL_VERIFICATION",
                               collection_mode=self.collection_mode)
            status = "RESULT" if result["status"] == "PREDICTED" else "REJECTED"
            reason = ("MOTION_GATED_BINARY_RESULT_NOT_GESTURE_CONFIRMATION" if status == "RESULT"
                      else result["failure_reason"])
            self._emit("INFERENCE_END", now, diagnostics=diagnostics, status=status, reason=reason)
            return self._finish(status, index, now, reason, diagnostics)
        if self.event_sink:
            try:
                indices = sample_valid_frame_indices([entry[0] for entry in entries])
                mapping = {entry[0]: entry for entry in entries}
                features, _ = build_sequence_features([mapping[i][1] for i in indices],
                                                      [mapping[i][2] for i in indices])
                diagnostics["handshape_evidence"] = handshape_evidence(features)
                diagnostics["pre_roll_first_original_frame"] = entries[0][0]
                diagnostics["capture_started_frame"] = (self.attempts.active or {}).get("start_frame")
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                diagnostics["diagnostic_error"] = str(exc)  # Never changes acceptance.
        try:
            candidate, reason = evaluate_segment(entries, self.model, self.segmenter.settings,
                                                 diagnostics=diagnostics)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
            candidate, reason = None, "INVALID_SEQUENCE: " + str(exc)
        if candidate is None:
            status = "UNCERTAIN" if diagnostics.get("model_called") else "REJECTED"
        elif endpoint != "OBSERVED_QUIET_END":
            status, reason = "UNCERTAIN", "TIMEOUT_ENDPOINT_UNVERIFIED_NOT_GESTURE_CONFIRMATION"
        else:
            status, reason = "RESULT", "BINARY_EXPERIMENT_ONLY_NOT_GESTURE_CONFIRMATION"
        self._emit("INFERENCE_END", now, diagnostics=diagnostics, status=status, reason=reason)
        return self._finish(status, index, now, reason, diagnostics)

    def note_displayed(self, now):
        lines = display_lines(self, now)
        if lines["result_status"] == "RESULT" and lines["label"] in ("J", "Z"):
            if self.attempts.displayed(self.last_result):
                self._emit("RESULT_DISPLAYED", now, label=lines["label"],
                           result_attempt_id=self.last_result["attempt_id"])

    def close_debug_session(self, now):
        if self.attempts.active and not self.attempts.active["finished"]:
            self._finish("UNCERTAIN", self.frame_context.get("frame_id", -1), now,
                         "SESSION_ENDED_BEFORE_VERIFIED_RESULT")
        self._emit("SESSION_SUMMARY", now, statistics=self.attempts.snapshot())

    def end_of_file(self, index, now):
        """Offline reporting only: EOF is never a live quiet-end or a result."""
        if not self.segmenter.active:
            return None
        _state, reason, entries = self.segmenter.review_at_eof()
        return self._finish("UNCERTAIN" if entries else "REJECTED", index, now, reason,
                            {"endpoint_verified": False, "valid_frames": len(entries)})


def detect_frame(task, frame):
    """Same IMAGE/BGR->RGB/no-flip detection contract as the SPACE program."""
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = task.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        count = len(result.hand_landmarks)
        hand = result.hand_landmarks[0] if count == 1 else None
        if hand is not None:
            try:
                landmarks_to_array(hand)
            except (ValueError, TypeError, AttributeError) as exc:
                return None, count, "INVALID_LANDMARK_FRAME: " + str(exc)
        return hand, count, ""
    except (ValueError, RuntimeError, TypeError, AttributeError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def display_lines(session, now):
    """Testable display contract: never present rejected motion as J/Z."""
    result = session.last_result if now < session.visible_until else None
    model_result = result.get("model_result") if result else None
    label, probabilities = "-", "P(J)=--  P(Z)=--"
    if model_result and model_result.get("status") == "PREDICTED":
        probabilities = f"P(J)={model_result['probability_J']:.4f}  P(Z)={model_result['probability_Z']:.4f}"
        if result["status"] == "RESULT":
            label = model_result["prediction"]
        elif result.get("endpoint") == "TIMEOUT_ENDPOINT_UNVERIFIED":
            label = "? " + model_result["prediction"]
    return {"state": session.state, "label": label, "probabilities": probabilities,
            "result_status": result["status"] if result else "",
            "reason": result["reason"] if result else session.reason}


def render(frame, session, now):
    height, width = frame.shape[:2]
    canvas = np.full((max(height, 590 if session.event_sink else 460), width+390, 3), 245, dtype=np.uint8)
    canvas[:height, :width] = frame  # Original orientation, display only.
    lines = display_lines(session, now)

    def text(value, y, scale=.55, thickness=1):
        cv2.putText(canvas, value, (width+12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness, cv2.LINE_AA)

    text("EXPERIMENTAL J/Z AUTO", 28, .65, 2)
    text(lines["state"], 65, .75, 2)
    text(f"Valid frames: {session.valid_frames} / 16", 94)
    text(f"Motion score: {session.segmenter.motion_score:.3f}", 120)
    text(f"Gap: {session.segmenter.missing_frames} | {session.collection_mode}", 146, .48)
    if session.collection_mode == "first16":
        metrics = session.segmenter.trace_metrics(now)
        text(f"Observed: {metrics.get('capture_seconds_after_onset',0.):.2f}s / "
             f"{session.segmenter.motion_gate.min_capture_seconds:.2f}s",170,.48)
        text(metrics.get("prediction_wait_reason","WAITING_FOR_NEW_MOTION"),190,.40)
    text(lines["label"], 265, 2.8, 4)
    text(lines["probabilities"], 300, .55)
    text(lines["result_status"], 330, .65, 2)
    for row, start in enumerate(range(0, len(lines["reason"]), 36)):
        text(lines["reason"][start:start+36], 355+row*20, .43)
    text("Probability does NOT prove J/Z", 414, .48)
    text("One visible hand | Q / Esc: quit", 440, .48)
    if session.event_sink:
        result = session.last_result or {}
        readable = result.get("readable_reason", "Waiting for diagnostic attempt")
        for row, start in enumerate(range(0, len(readable), 43)):
            text(readable[start:start+43], 468+row*19, .43)
        counts = session.attempts.snapshot()
        text(f"Attempts {counts['TOTAL_ATTEMPTS']} | done {counts['CAPTURE_COMPLETED']}", 522, .48)
        text(f"Model {counts['MODEL_INVOKED']} | shown {counts['RESULT_DISPLAYED']}", 545, .48)
        text(f"Reject {counts['REJECTED']} | timeout {counts['TIMEOUT']}", 568, .48)
    return canvas


def run_camera(session, camera_index):
    camera = None
    try:
        # Exactly the task/mode/options used by the original SPACE test.
        with create_hand_landmarker(TASK_PATH, running_mode="IMAGE", num_hands=1) as task:
            camera = cv2.VideoCapture(camera_index)
            if not camera.isOpened():
                raise OSError(f"Cannot open camera index {camera_index}")
            index = 0
            while True:
                ok, frame = camera.read()
                now = time.monotonic()
                if not ok or frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                    raise OSError("Cannot read a valid camera frame")
                hand, count, error = detect_frame(task, frame)
                height, width = frame.shape[:2]
                session.update(index, hand, (width, height), now, detection_error=error, hand_count=count)
                cv2.imshow(WINDOW, render(frame, session, now))
                key = cv2.waitKey(1) & 0xff
                session.note_displayed(now)
                if key in (ord("q"), 27):
                    return 0
                # PROCESSING has been drawn before the short synchronous model
                # call. No sleep, second camera or start/stop keys.
                if session.pending is not None:
                    session.process_pending(time.monotonic())
                index += 1
    finally:
        try:
            if session.event_sink:
                session.close_debug_session(time.monotonic())
                print("JZ_SESSION_STATISTICS " + json.dumps(session.attempts.snapshot()), flush=True)
        finally:
            # A disk/logging error must never prevent camera cleanup.
            if camera is not None:
                camera.release()
            cv2.destroyAllWindows()


def replay_verified_videos(model, settings, predict_on_timeout=False, collection_mode="first16",
                           motion_gate=MotionGateSettings()):
    """Bounded read-only inference on four named files, never a dataset extract.

    Time = original frame index / source FPS. No tail frames are invented and
    EOF is not converted into quiet-end. Only summaries/indices, no features,
    are returned. All inputs are hashed before/after; nothing is saved.
    """
    split = json.loads((EXPERIMENT_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    membership = {row["source_path"]: name for name, rows in split["splits"].items() for row in rows}
    reports = []
    for relative in REPLAY_VIDEOS:
        path = (RAW_ROOT / relative).resolve()
        if not path.is_relative_to(RAW_ROOT.resolve()) or not path.is_file():
            raise ValueError(f"Missing or unsafe video: {relative}")
        before = _sha256(path)
        capture = cv2.VideoCapture(str(path))  # A FILE, never a camera index.
        entries, index = [], 0
        session = AutoJzSession(model, settings, predict_on_timeout=predict_on_timeout,
                               collection_mode=collection_mode,motion_gate=motion_gate)
        try:
            if not capture.isOpened():
                raise OSError(f"Cannot open video: {relative}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError(f"Invalid source FPS: {relative}")
            with create_hand_landmarker(TASK_PATH, running_mode="IMAGE", num_hands=1) as task:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    hand, count, error = detect_frame(task, frame)
                    size = (frame.shape[1], frame.shape[0])
                    if hand is not None:
                        entries.append((index, hand, size))
                    session.update(index, hand, size, index/fps, detection_error=error, hand_count=count)
                    session.process_pending(index/fps)
                    index += 1
        finally:
            capture.release()
        original = predict_attempt(entries, index, model)
        session.end_of_file(index, index/fps)
        if _sha256(path) != before:
            raise ValueError(f"Source changed during read-only replay: {relative}")
        reports.append({"video": relative, "expected_label": relative.split("/")[0],
                        "source_split": membership.get(path.relative_to(ROOT).as_posix(), "UNKNOWN"),
                        "signer_independent": False, "decoded_frames": index, "fps": fps,
                        "valid_frames": len(entries), "source_sha256": before,
                        "original_whole_clip_model": original, "auto_transitions": list(session.transitions),
                        "auto_results": list(session.results), "source_unchanged": True})
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="No camera/MediaPipe/video processing or writes")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--collection-mode", choices=("first16", "quiet-end"), default="first16",
                        help="Default: motion-gated >=16 frames + min duration; quiet-end retains older experiment")
    parser.add_argument("--min-capture-seconds", type=float, default=1.)
    parser.add_argument("--min-motion-seconds", type=float, default=.20)
    parser.add_argument("--min-motion-speed", type=float, default=.15, help="Palm scales/second")
    parser.add_argument("--min-motion-excursion", type=float, default=.25, help="Palm scales")
    parser.add_argument("--min-relative-excursion", type=float, default=.12, help="Tip relative to wrist, palm scales")
    parser.add_argument("--min-path-bend", type=float, default=.10, help="Bent path alternative, palm scales")
    parser.add_argument("--max-gap-frames", type=int, default=None,
                        help="Allowed consecutive missing frames (first16 default 6; quiet-end 2)")
    parser.add_argument("--max-gap-ms", type=float, default=None,
                        help="Maximum time since last valid frame (first16 default 400ms; quiet-end 200ms)")
    parser.add_argument("--start-displacement", type=float, default=.15)
    parser.add_argument("--start-frames", type=int, default=3)
    parser.add_argument("--pre-roll-seconds", type=float, default=.50)
    parser.add_argument("--end-quiet-seconds", type=float, default=.35,
                        help="first16: quiet re-arm only, never delays prediction; quiet-end: completion window")
    parser.add_argument("--max-seconds", type=float, default=6.)
    parser.add_argument("--result-seconds", type=float, default=4.)
    parser.add_argument("--debug", action="store_true", help="Temporary buffered JSONL and diagnostic panel")
    parser.add_argument("--debug-trace", type=Path, help="Unused JSONL inside the J/Z experiment; requires --debug")
    parser.add_argument("--predict-on-timeout", action="store_true", help="UNCERTAIN diagnostic only, never RESULT")
    parser.add_argument("--replay-verified-videos", action="store_true",
                        help="Read-only inference ONLY J/246,J/252,Z/278,Z/189; no webcam or file writes")
    args = parser.parse_args(argv)
    if args.collection_mode == "first16" and args.predict_on_timeout:
        parser.error("--predict-on-timeout requires --collection-mode quiet-end; first16 requires 16 valid frames")
    if args.max_gap_frames is None:
        args.max_gap_frames = 6 if args.collection_mode == "first16" else 2
    if args.max_gap_ms is None:
        args.max_gap_ms = 400. if args.collection_mode == "first16" else 200.
    if args.max_gap_frames < 0 or not np.isfinite(args.max_gap_ms) or args.max_gap_ms <= 0:
        parser.error("max-gap-frames must be >=0; max-gap-ms must be finite and positive")
    if args.debug_trace is not None and not args.debug:
        parser.error("--debug-trace requires --debug")
    trace_root = ROOT / "data/experiments/dynamic_jz_v1"
    trace_path = args.debug_trace
    if args.debug and trace_path is None:
        trace_path = trace_root / "metadata" / (
            "auto_camera_events_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
            + uuid4().hex[:8] + ".jsonl")
    if trace_path is not None:
        trace_path = trace_path.resolve()
        if (not trace_path.is_relative_to(trace_root.resolve()) or trace_path.exists()
                or trace_path.suffix.lower() != ".jsonl"):
            parser.error("Trace must be an unused JSONL inside data/experiments/dynamic_jz_v1")
    if args.debug and args.replay_verified_videos and not args.check_only:
        parser.error("Use --debug for the live auto session, not the separate four-video replay")
    for name in ("start_displacement", "pre_roll_seconds", "end_quiet_seconds", "max_seconds", "result_seconds",
                 "min_capture_seconds", "min_motion_seconds", "min_motion_speed", "min_motion_excursion",
                 "min_relative_excursion", "min_path_bend"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.start_frames < 2 or (args.collection_mode == "quiet-end" and args.max_seconds <= args.end_quiet_seconds):
        parser.error("start-frames must be >=2; quiet-end max-seconds must exceed end-quiet-seconds")
    settings = replace(SegmentSettings(), start_displacement=args.start_displacement,
                       start_frames=args.start_frames, pre_roll_seconds=args.pre_roll_seconds,
                       end_quiet_seconds=args.end_quiet_seconds, max_seconds=args.max_seconds,
                       max_missing_frames=args.max_gap_frames, max_frame_gap_seconds=args.max_gap_ms/1000.)
    motion_gate = MotionGateSettings(min_capture_seconds=args.min_capture_seconds,
        min_motion_seconds=args.min_motion_seconds,min_motion_speed=args.min_motion_speed,
        min_excursion=args.min_motion_excursion,min_relative_excursion=args.min_relative_excursion,
        min_path_bend=args.min_path_bend)
    if args.collection_mode == "first16" and args.min_capture_seconds >= args.max_seconds:
        parser.error("--min-capture-seconds must be below --max-seconds")
    try:
        model, model_hash = load_experiment_model()
        _synthetic_feature_check(model)
        if args.check_only:
            print(json.dumps({"status": "CHECK_ONLY_PASS", "camera_opened": False,
                              "mediapipe_executed": False, "files_written": 0,
                              "model_path": str(EXPERIMENT_DIR / "model_pipeline.joblib"),
                              "model_sha256": model_hash, "pipeline": ["StandardScaler", "LogisticRegression"],
                              "feature_shape": [16, 69], "flatten_order": "C", "no_flip": True,
                              "collection_mode": args.collection_mode,
                              "prediction_policy": ("MOTION_AND_DURATION_THEN_UNIFORM_VALID_16" if args.collection_mode == "first16"
                                                    else "WHOLE_BOUT_UNIFORM_16_AFTER_QUIET_END"),
                              "quiet_end_required_for_prediction": args.collection_mode == "quiet-end",
                              "geometry_gate_applied": args.collection_mode == "quiet-end",
                              "motion_gate": asdict(motion_gate), "pre_roll_counts_as_motion_or_duration": False,
                              "partial_gesture_distribution_risk": args.collection_mode == "first16",
                              "settings": asdict(settings), "result_seconds": args.result_seconds,
                              "predict_on_timeout": args.predict_on_timeout, "experimental": True,
                              "debug_enabled": args.debug, "debug_trace_path": str(trace_path) if trace_path else None,
                              "no_none_class": True, "replay_allowlist": list(REPLAY_VIDEOS)}))
            return 0
        if args.replay_verified_videos:
            print(json.dumps({"mode": "FOUR_FILE_OFFLINE_INFERENCE_NOT_GENERALIZATION_TEST",
                              "model_sha256": model_hash, "settings": asdict(settings),
                              "reports": replay_verified_videos(model, settings, args.predict_on_timeout,
                                                                args.collection_mode,motion_gate)}, allow_nan=False))
            return 0
        print("EXPERIMENTAL ONLY: automatic segmentation is unvalidated; probabilities do not prove J/Z.")
        if trace_path is None:
            session = AutoJzSession(model, settings, result_seconds=args.result_seconds,
                                   predict_on_timeout=args.predict_on_timeout, collection_mode=args.collection_mode,
                                   motion_gate=motion_gate)
            return run_camera(session, args.camera_index)
        with DebugTrace(trace_path, trace_root) as trace:
            print(f"JZ_AUTO_DEBUG_TRACE={trace_path}", flush=True)
            session = AutoJzSession(model, settings, result_seconds=args.result_seconds,
                                   predict_on_timeout=args.predict_on_timeout, event_sink=trace,
                                   collection_mode=args.collection_mode,motion_gate=motion_gate)
            session._emit("SESSION_START", time.monotonic(), model_sha256=model_hash,
                          feature_shape=[16,69], no_flip=True, settings=asdict(settings),
                          result_seconds=args.result_seconds, collection_mode=args.collection_mode,
                          motion_gate=asdict(motion_gate))
            return run_camera(session, args.camera_index)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
