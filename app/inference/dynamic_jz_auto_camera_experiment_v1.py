"""Standalone EXPERIMENTAL J/Z auto camera: no SPACE, no text commits, no NONE.

Reuses the existing motion detector and the original IMAGE/no-flip feature and
model path. Geometry gates are unvalidated rejection heuristics, NOT a NONE
classifier or an ASL verifier. Nothing is written to disk by this program.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

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

WINDOW = "EXPERIMENTAL automatic J/Z - no NONE model"
RAW_ROOT = ROOT / "data/raw/dynamic/external/signnteam_jz/SigNN Video Data"
REPLAY_VIDEOS = ("J/246.avi", "J/252.avi", "Z/278.avi", "Z/189.avi")


class AutoJzSession:
    """Motion collection independent of camera/UI; one prediction per bout.

    Fresh anchoring after a terminal result prevents the held endpoint from
    becoming another bout. Collection for the next genuine movement continues
    while the previous result is visible; display duration is NOT a cooldown.
    """
    def __init__(self, model, settings=SegmentSettings(), *, result_seconds=2., predict_on_timeout=False):
        if not np.isfinite(result_seconds) or result_seconds <= 0:
            raise ValueError("result_seconds must be finite and positive")
        self.model = model
        self.segmenter = MotionSegmenter(settings)
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

    def _state(self, state, index, now, reason=""):
        if state != self.state or reason != self.reason:
            self.transitions.append({"frame_id": index, "timestamp_seconds": now,
                                     "previous_state": self.state, "new_state": state, "reason": reason})
        self.state, self.reason = state, reason

    def _finish(self, status, index, now, reason, diagnostics=None):
        result = {"status": status, "frame_id": index, "reason": reason,
                  "model_called": False, "model_result": None, **(diagnostics or {})}
        self.last_result = result
        self.results.append(result)
        self.visible_until = now + self.result_seconds
        self._state(status, index, now, reason)
        return result

    def reject(self, reason, index, now):
        self.segmenter.reset()
        self.pending = None
        self.valid_frames = 0
        return self._finish("REJECTED", index, now, reason)

    def update(self, index, hand, size, now, *, detection_error="", hand_count=None):
        if self.pending is not None:
            raise RuntimeError("Process the pending bout before consuming another frame")
        if detection_error:
            return self.reject("DETECTION_ERROR: " + detection_error, index, now)
        if hand_count is not None and hand_count > 1:
            return self.reject("MULTIPLE_HANDS_IDENTITY_UNVERIFIED", index, now)
        if hand is None and not self.segmenter.active:
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
                   else self.segmenter.trace_metrics(now))
        self.valid_frames = int(metrics.get("valid_frames", 0))
        if state == "COMPLETE":
            self.pending = (entries, index, "OBSERVED_QUIET_END", metrics)
            self._state("PROCESSING", index, now)
        elif state == "REJECTED":
            if reason == "SEGMENT_TIMEOUT" and self.predict_on_timeout:
                coverage = (len(prior_entries)/(prior_entries[-1][0]-prior_entries[0][0]+1)
                            if prior_entries else 0.)
                if len(prior_entries) >= 16 and coverage >= self.segmenter.settings.min_valid_fraction:
                    self.pending = (prior_entries, index, "TIMEOUT_ENDPOINT_UNVERIFIED", metrics)
                    self._state("PROCESSING", index, now, "TIMEOUT_DIAGNOSTIC_ONLY")
                    return None
            self._finish("REJECTED", index, now, reason, {"segment": metrics})
        elif state in ("CAPTURING", "GAP") and self.segmenter.active:
            self._state("CAPTURING", index, now, "TRACKING_GAP_NOT_AN_ENDPOINT" if state == "GAP" else "")
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
        return self._finish(status, index, now, reason, diagnostics)

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
            landmarks_to_array(hand)
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
    canvas = np.full((max(height, 460), width+390, 3), 245, dtype=np.uint8)
    canvas[:height, :width] = frame  # Original orientation, display only.
    lines = display_lines(session, now)

    def text(value, y, scale=.55, thickness=1):
        cv2.putText(canvas, value, (width+12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness, cv2.LINE_AA)

    text("EXPERIMENTAL J/Z AUTO", 28, .65, 2)
    text(lines["state"], 65, .75, 2)
    text(f"Valid frames: {session.valid_frames}", 94)
    text(f"Motion score: {session.segmenter.motion_score:.3f}", 120)
    text(lines["label"], 230, 2.8, 4)
    text(lines["probabilities"], 268, .55)
    text(lines["result_status"], 300, .65, 2)
    for row, start in enumerate(range(0, len(lines["reason"]), 36)):
        text(lines["reason"][start:start+36], 325+row*20, .43)
    text("Probability does NOT prove J/Z", 414, .48)
    text("One visible hand | Q / Esc: quit", 440, .48)
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
                if cv2.waitKey(1) & 0xff in (ord("q"), 27):
                    return 0
                # PROCESSING has been drawn before the short synchronous model
                # call. No sleep, second camera, keys or frame-count stop.
                if session.pending is not None:
                    session.process_pending(time.monotonic())
                index += 1
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


def replay_verified_videos(model, settings, predict_on_timeout=False):
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
        session = AutoJzSession(model, settings, predict_on_timeout=predict_on_timeout)
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
    parser.add_argument("--start-displacement", type=float, default=.15)
    parser.add_argument("--start-frames", type=int, default=3)
    parser.add_argument("--pre-roll-seconds", type=float, default=.50)
    parser.add_argument("--end-quiet-seconds", type=float, default=.35)
    parser.add_argument("--max-seconds", type=float, default=6.)
    parser.add_argument("--result-seconds", type=float, default=2.)
    parser.add_argument("--predict-on-timeout", action="store_true", help="UNCERTAIN diagnostic only, never RESULT")
    parser.add_argument("--replay-verified-videos", action="store_true",
                        help="Read-only inference ONLY J/246,J/252,Z/278,Z/189; no webcam or file writes")
    args = parser.parse_args(argv)
    for name in ("start_displacement", "pre_roll_seconds", "end_quiet_seconds", "max_seconds", "result_seconds"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.start_frames < 2 or args.max_seconds <= args.end_quiet_seconds:
        parser.error("start-frames must be >=2; max-seconds must exceed end-quiet-seconds")
    settings = replace(SegmentSettings(), start_displacement=args.start_displacement,
                       start_frames=args.start_frames, pre_roll_seconds=args.pre_roll_seconds,
                       end_quiet_seconds=args.end_quiet_seconds, max_seconds=args.max_seconds)
    try:
        model, model_hash = load_experiment_model()
        _synthetic_feature_check(model)
        if args.check_only:
            print(json.dumps({"status": "CHECK_ONLY_PASS", "camera_opened": False,
                              "mediapipe_executed": False, "files_written": 0,
                              "model_path": str(EXPERIMENT_DIR / "model_pipeline.joblib"),
                              "model_sha256": model_hash, "pipeline": ["StandardScaler", "LogisticRegression"],
                              "feature_shape": [16, 69], "flatten_order": "C", "no_flip": True,
                              "settings": asdict(settings), "result_seconds": args.result_seconds,
                              "predict_on_timeout": args.predict_on_timeout, "experimental": True,
                              "no_none_class": True, "replay_allowlist": list(REPLAY_VIDEOS)}))
            return 0
        if args.replay_verified_videos:
            print(json.dumps({"mode": "FOUR_FILE_OFFLINE_INFERENCE_NOT_GENERALIZATION_TEST",
                              "model_sha256": model_hash, "settings": asdict(settings),
                              "reports": replay_verified_videos(model, settings, args.predict_on_timeout)}, allow_nan=False))
            return 0
        session = AutoJzSession(model, settings, result_seconds=args.result_seconds,
                               predict_on_timeout=args.predict_on_timeout)
        print("EXPERIMENTAL ONLY: automatic segmentation is unvalidated; probabilities do not prove J/Z.")
        return run_camera(session, args.camera_index)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
