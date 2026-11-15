"""Separate two-button ABC (static + automatic J/Z) / 123 experiment.

Original unified camera and production dispatcher are untouched. The primary
VIDEO task matches the trusted baseline; an independent IMAGE task is needed
for unmirrored J/Z because detector-mode/mirror equivalence is NOT established.
--disable-dynamic uses only the baseline task and one detection per frame.
J/Z commits are OFF unless explicitly enabled. No NONE guarantee.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.feature_extraction.hand_landmarks import create_hand_landmarker, landmarks_to_array  # noqa: E402
from app.inference.automatic_jz_experiment import AbcDynamicCoordinator  # noqa: E402
from app.inference.dynamic_jz_camera_test import load_experiment_model, _synthetic_feature_check  # noqa: E402
from app.inference.hold_confirmation_camera_test import _panel  # noqa: E402
from app.inference.letter_inference import (  # noqa: E402
    HAND_MODEL_PATH, DEFAULT_MODEL, IntegrationError, StaticAslInference,
    load_official_artifacts, _validate_deployment_config,
)
from app.inference.unified_camera_experiment import prepare_inference_frame, render_mirrored_frame  # noqa: E402
from app.inference.unified_static_coordinator import DigitPredictor, load_digit_model  # noqa: E402

WINDOW = "ABC A-Z / 123 - EXPERIMENT (JZ not validated)"


def mode_from_click(x, y, width):
    if 8 <= y <= 47:
        if 14 <= x - width <= 145:
            return "ABC"
        if 165 <= x - width <= 296:
            return "123"
    return None


def draw_controls(display, width, coordinator):
    for label, left, right in (("ABC", 14, 145), ("123", 165, 296)):
        color = (218, 240, 218) if coordinator.mode == label else (235, 235, 235)
        cv2.rectangle(display, (width + left, 8), (width + right, 47), color, -1)
        cv2.putText(display, label, (width + left + 34, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    .75, (0, 0, 0), 2, cv2.LINE_AA)
    # Non-clickable status in the light panel, not over the camera/hand.
    cv2.putText(display, coordinator.status[:42], (width + 8, 61), cv2.FONT_HERSHEY_SIMPLEX,
                .36, (0, 0, 0), 1, cv2.LINE_AA)
    detail = ("JZ OFF (detector disabled)" if not coordinator.dynamic_enabled
              else "JZ commits " + ("ON (experimental)" if coordinator.enable_commits else "OFF (diagnostic)")
              + " | last: " + (coordinator.jz_candidate or "-"))
    cv2.putText(display, detail,
                (width + 8, 76), cv2.FONT_HERSHEY_SIMPLEX, .36, (0, 0, 0), 1, cv2.LINE_AA)


def detect(landmarker, frame, timestamp_ms=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return (landmarker.detect(image) if timestamp_ms is None
            else landmarker.detect_for_video(image, timestamp_ms))


def debug_snapshot(coordinator, detected, fps, mediapipe_ms, inference_ms):
    prediction = coordinator.static_prediction
    return {
        "ACTIVE_MODE": coordinator.mode, "HAND_DETECTED": detected,
        **coordinator.frame_context,
        "STATIC_RAW_PREDICTION": prediction.instant_label if prediction else None,
        "STATIC_STABLE_PREDICTION": prediction.stable_label if prediction else None,
        "HOLD_STATE": coordinator.hold.state,
        "DYNAMIC_STATE": coordinator.dynamic_state,
        "DYNAMIC_DETAIL": coordinator.status,
        "DYNAMIC_MOTION_SCORE": round(coordinator.segmenter.motion_score, 4),
        "STATIC_SUPPRESSION_REASON": coordinator.suppression_reason or "NONE",
        "JZ_CANDIDATE": coordinator.jz_candidate,
        "FPS": round(fps, 1), "MEDIAPIPE_MS": round(mediapipe_ms, 1),
        "INFERENCE_MS": round(inference_ms, 1),
    }


class EventTrace:
    """Opt-in buffered JSONL; transitions/evaluations only, no per-frame print."""
    def __init__(self, path):
        root = (ROOT / "data/experiments/dynamic_jz_v1").resolve()
        self.path = Path(path).resolve()
        if not self.path.is_relative_to(root):
            raise ValueError("J/Z trace must stay under data/experiments/dynamic_jz_v1")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("x", encoding="utf-8", buffering=65536)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.file.close()

    def __call__(self, event):
        event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}
        self.file.write(json.dumps(event, allow_nan=False) + "\n")


def process_camera_observations(coordinator, video_result, image_result, shape, index, now,
                                *, video_error=None, image_error=None):
    """Route two independent same-frame observations; never fabricate landmarks."""
    video_hands = video_result.hand_landmarks if video_result is not None else []
    image_hands = image_result.hand_landmarks if image_result is not None else []
    video_count = len(video_hands) if video_result is not None else None
    image_count = len(image_hands) if image_result is not None else None
    raw_hand = None
    image_status = "NOT_RUN"
    if image_error:
        image_status = "IMAGE_ERROR: " + image_error
    elif image_result is not None:
        image_status = ("IMAGE_NO_HAND" if not image_hands else "IMAGE_MULTIPLE_HANDS"
                        if len(image_hands) != 1 else "IMAGE_SINGLE_HAND")
        if len(image_hands) == 1:
            raw_hand = image_hands[0]
            if len(video_hands) == 1:
                try:
                    agrees = oriented_passes_agree(video_hands[0], raw_hand, shape)
                except (ValueError, TypeError, AttributeError) as exc:
                    agrees = False
                    image_status = f"CORRESPONDENCE_ERROR: {type(exc).__name__}: {exc}"
                if agrees:
                    image_status = "MATCHED_RAW_IMAGE"
                elif image_status == "IMAGE_SINGLE_HAND":
                    image_status = "ORIENTATION_CORRESPONDENCE_MISMATCH"
                if not agrees:
                    raw_hand = None
    context = {
        "frame_id": index, "video_hand_detected": bool(video_hands) if video_count is not None else None,
        "image_hand_detected": bool(image_hands) if image_count is not None else None,
        "video_hand_count": video_count, "image_hand_count": image_count,
        "video_error": video_error, "image_status": image_status}
    if coordinator.event_sink:
        keys = ("video_hand_count", "image_hand_count", "video_error", "image_status")
        if any(context[key] != coordinator.frame_context.get(key) for key in keys):
            coordinator.event_sink({"event": "OBSERVATION_CHANGE", **context,
                                     "timestamp_monotonic": now, "previous_state": coordinator.dynamic_state,
                                     "new_state": coordinator.dynamic_state,
                                     "reason": "MEDIAPIPE_AVAILABILITY_OR_CORRESPONDENCE_CHANGED",
                                     "segment": coordinator.segmenter.trace_metrics(now),
                                     "model_called": False, "model_result": None, "geometry_candidates": None,
                                     "candidate": coordinator.jz_candidate,
                                     "identity_unverified": coordinator.identity_unverified})
    coordinator.frame_context = context
    if len(video_hands) == 1:
        hand = video_hands[0]
        prediction, committed = coordinator.process_hand(
            hand, raw_hand, shape, index, now, dynamic_missing=image_status == "IMAGE_NO_HAND",
            dynamic_failure_reason=image_status)
        return prediction, committed, hand
    if (not video_hands and coordinator.mode == "ABC" and coordinator.dynamic_enabled
            and raw_hand is not None):
        prediction, committed = coordinator.process_secondary_only(raw_hand, shape, index, now)
        return prediction, committed, None
    if len(video_hands) > 1 or len(image_hands) > 1:
        coordinator.uncertain_hand(now)
    elif image_error or video_error:
        coordinator.cancel_dynamic(image_status if image_error else "VIDEO_ERROR: " + video_error, now)
        coordinator.static.missing_hand(now)
        coordinator.static_prediction = None
    else:
        coordinator.missing_hand(now, index=index)
    return None, None, None


def oriented_passes_agree(mirrored_hand, raw_hand, frame_shape):
    """Correspondence guard only, never used to manufacture model features.

    One visible detection per pass is not proof of identity; reject substantial
    disagreement between mirrored/unmirrored detections. Subtle switches remain
    a limitation and require live visual validation.
    """
    height, width = frame_shape[:2]
    mirrored = landmarks_to_array(mirrored_hand)[:, :2].copy()
    mirrored[:, 0] = 1 - mirrored[:, 0]
    raw = landmarks_to_array(raw_hand)[:, :2]
    mirrored *= [width, height]
    raw = raw * [width, height]
    scale = float(np.median(np.linalg.norm(raw[[5, 9, 17]] - raw[0], axis=1)))
    error = np.linalg.norm(mirrored - raw, axis=1)
    return bool(scale > 1e-6 and error[0] / scale < .35 and np.mean(error) / scale < .35)


def run_camera(coordinator, camera_index, debug_motion=False, event_trace_path=None):
    camera = cv2.VideoCapture(camera_index)
    requested, width = [coordinator.mode], [0]

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            mode = mode_from_click(x, y, width[0])
            if mode:
                requested[0] = mode

    try:
        if not camera.isOpened():
            raise OSError("Cannot open camera")
        # Do not approximate raw J/Z landmarks by mirroring ABC coordinates.
        # Separate modes preserve baseline VIDEO tracking and training IMAGE
        # detection; only one physical camera, tasks reused throughout the loop.
        with ExitStack() as stack:
            if event_trace_path is not None:
                coordinator.event_sink = stack.enter_context(EventTrace(event_trace_path))
                print(f"JZ_EVENT_TRACE={event_trace_path}", flush=True)
            task = stack.enter_context(create_hand_landmarker(
                model_path=HAND_MODEL_PATH, running_mode="VIDEO", num_hands=1,
                min_hand_detection_confidence=.5, min_hand_presence_confidence=.5,
                min_tracking_confidence=.5))
            dynamic_task = (stack.enter_context(create_hand_landmarker(
                HAND_MODEL_PATH, running_mode="IMAGE", num_hands=2))
                if coordinator.dynamic_enabled else None)
            cv2.namedWindow(WINDOW)
            cv2.setMouseCallback(WINDOW, on_mouse)
            index = 0
            started = time.monotonic()
            last_timestamp = -1
            last_debug = -float("inf")
            previous_frame_time = None
            fps = 0.0
            while True:
                coordinator.switch_mode(requested[0])
                ok, raw = camera.read()
                now = time.monotonic()
                if not ok or raw is None:
                    coordinator.missing_hand(now)
                    raise OSError("Camera read failed")
                inference_frame = prepare_inference_frame(raw, coordinator.mode)
                width[0] = raw.shape[1]
                hand = None
                label = None
                mp_ms = 0.0
                inference_ms = 0.0
                timestamp_ms = max(int((now - started) * 1000), last_timestamp + 1)
                last_timestamp = timestamp_ms
                static_result, raw_result = None, None
                video_error, image_error = None, None
                tick = time.perf_counter()
                try:
                    static_result = detect(task, inference_frame, timestamp_ms)
                except (ValueError, RuntimeError, AttributeError, TypeError) as exc:
                    video_error = f"{type(exc).__name__}: {exc}"
                # IMAGE must be observed independently, even when VIDEO misses.
                # It always consumes the original unmirrored camera frame.
                if coordinator.mode == "ABC" and dynamic_task is not None:
                    try:
                        raw_result = detect(dynamic_task, raw)
                    except (ValueError, RuntimeError, AttributeError, TypeError) as exc:
                        image_error = f"{type(exc).__name__}: {exc}"
                mp_ms = (time.perf_counter() - tick) * 1000
                tick = time.perf_counter()
                try:
                    prediction, committed, hand = process_camera_observations(
                        coordinator, static_result, raw_result, raw.shape, index, now,
                        video_error=video_error, image_error=image_error)
                    if prediction:
                        label = prediction.stable_label or prediction.instant_label
                    if committed:
                        print(f"CONFIRMED={committed} TEXT={coordinator.composer.text}", flush=True)
                except (ValueError, IntegrationError, RuntimeError) as exc:
                    coordinator.uncertain_hand(now)
                    coordinator.status = f"PRIMARY_ERROR: {type(exc).__name__}"
                inference_ms = (time.perf_counter() - tick) * 1000
                index += 1
                frame = render_mirrored_frame(inference_frame, coordinator.mode, hand)
                display = _panel(frame, coordinator.hold, label, now)
                draw_controls(display, width[0], coordinator)
                if previous_frame_time is not None and now > previous_frame_time:
                    measured = 1 / (now - previous_frame_time)
                    fps = measured if fps == 0 else .2 * measured + .8 * fps
                previous_frame_time = now
                if debug_motion and now - last_debug >= .5:
                    print("MOTION_DEBUG " + json.dumps(debug_snapshot(
                        coordinator, hand is not None, fps, mp_ms, inference_ms)), flush=True)
                    last_debug = now
                cv2.imshow(WINDOW, display)
                if cv2.waitKey(1) & 0xff in (ord("q"), 27):
                    return 0
    finally:
        coordinator.event_sink = None
        camera.release()
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--debug-motion", action="store_true", help="Diagnostic JSON at most twice/second")
    parser.add_argument("--debug-jz-events", action="store_true",
                        help="Buffered transition/evaluation JSONL, no per-frame Terminal output")
    parser.add_argument("--jz-event-trace", type=Path,
                        help="New JSONL under data/experiments/dynamic_jz_v1; requires --debug-jz-events")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--disable-dynamic", action="store_true", help="Exact baseline static VIDEO path, no secondary pass")
    modes.add_argument("--enable-experimental-jz", action="store_true",
                        help="Opt in to UNVALIDATED automatic J/Z + endpoint HOLD commits")
    args = parser.parse_args(argv)
    if args.jz_event_trace is not None and not args.debug_jz_events:
        parser.error("--jz-event-trace requires --debug-jz-events")
    trace_path = args.jz_event_trace
    if args.debug_jz_events and trace_path is None:
        trace_path = ROOT / "data/experiments/dynamic_jz_v1/metadata" / (
            "camera_jz_events_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
            + uuid4().hex[:8] + ".jsonl")
    if trace_path is not None:
        trace_path = trace_path.resolve()
        if not trace_path.is_relative_to((ROOT / "data/experiments/dynamic_jz_v1").resolve()):
            parser.error("Event trace must stay under data/experiments/dynamic_jz_v1")
        if trace_path.exists():
            parser.error("Event trace already exists; refusing to overwrite")
    try:
        _validate_deployment_config()
        artifacts = load_official_artifacts(DEFAULT_MODEL)
        digits = load_digit_model()
        dynamic, model_hash = None, None
        if not args.disable_dynamic:
            dynamic, model_hash = load_experiment_model()
            _synthetic_feature_check(dynamic)
        coordinator = AbcDynamicCoordinator(StaticAslInference(artifacts), DigitPredictor(digits),
                                           dynamic, args.enable_experimental_jz, not args.disable_dynamic)
        if args.check_only:
            print(json.dumps({"status": "PASS", "camera_opened": False,
                              "modes": ["ABC", "123"], "display": "MIRRORED_BOTH_MODES",
                              "letters": "uvw_medium_epoch_81", "digits": "digits_final",
                              "dynamic_sha256": model_hash, "dynamic_feature_shape": [16, 69],
                              "jz_commits_enabled": args.enable_experimental_jz,
                              "segmentation": "HEURISTIC_UNVALIDATED_NO_NONE_MODEL",
                              "dynamic_enabled": not args.disable_dynamic,
                              "mediapipe": "BASELINE_VIDEO_PRIMARY + RAW_IMAGE_SECONDARY_WHEN_DYNAMIC_ENABLED",
                              "primary_mediapipe_matches_baseline": True,
                              "image_observed_independently_of_video": not args.disable_dynamic,
                              "unverified_bouts_auto_commit": False,
                              "mediapipe_instances": 1 if args.disable_dynamic else 2,
                              "camera_count": 1,
                              "hold_seconds": coordinator.hold.settings.hold_seconds,
                              "event_trace_enabled": args.debug_jz_events,
                              "event_trace_path": str(trace_path) if trace_path else None,
                              "production_changed": False}))
            return 0
        print("EXPERIMENT ONLY: automatic segmentation is unvalidated; binary scores do not prove J/Z.")
        return run_camera(coordinator, args.camera_index, args.debug_motion, trace_path)
    except (OSError, ValueError, IntegrationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
