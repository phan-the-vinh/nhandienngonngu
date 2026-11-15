# RECONSTRUCTED_PRE_APPLE comparison copy; not a verified historical original.
"""Experimental DOWN_CONFIRM camera for static letters, with PUSH backup.

This is deliberately separate from the production camera and the J/Z recorder.
It never writes text to disk and never interprets MediaPipe z as absolute depth.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.feature_extraction.hand_landmarks import create_hand_landmarker, landmarks_to_array  # noqa: E402
from app.interaction.down_confirmation import DownConfirmation, DownSettings  # noqa: E402
from app.inference.letter_inference import (  # noqa: E402
    DEFAULT_MODEL, HAND_MODEL_PATH, IntegrationError, StaticAslInference,
    _validate_deployment_config, draw_hand, load_official_artifacts,
)


@dataclass(frozen=True)
class PushSettings:
    hold_frames: int = 10
    push_frames: int = 2
    return_frames: int = 4
    min_confidence: float = 0.75
    push_ratio: float = 1.20
    width_ratio: float = 1.12
    return_ratio: float = 1.07
    max_ready_scale_cv: float = 0.06
    max_ready_wrist_span_palm_scales: float = 0.35
    max_lateral_palm_scales: float = 0.75
    max_shape_rms: float = 0.28

    def validate(self) -> None:
        if min(self.hold_frames, self.push_frames, self.return_frames) < 1:
            raise ValueError("Frame counts must be positive")
        if not (0 < self.min_confidence <= 1 and 0 < self.max_ready_scale_cv < 1):
            raise ValueError("Invalid confidence or calibration variation")
        if not (1 < self.return_ratio < self.push_ratio and 1 < self.width_ratio <= self.push_ratio):
            raise ValueError("Require 1 < return_ratio < push_ratio and 1 < width_ratio <= push_ratio")
        if (self.max_ready_wrist_span_palm_scales <= 0
                or self.max_lateral_palm_scales <= 0 or self.max_shape_rms <= 0):
            raise ValueError("Movement and shape tolerances must be positive")


@dataclass(frozen=True)
class HandObservation:
    label: str | None
    confidence: float
    scale_px: float
    width_px: float
    wrist_px: tuple[float, float]
    shape: tuple[float, ...]
    palm_center_px: tuple[float, float] | None = None


def observe_hand(hand: Sequence[object], prediction: object, frame_shape: tuple[int, ...]) -> HandObservation:
    """Measure 2D palm enlargement in pixel coordinates (aspect ratio preserved)."""
    raw = landmarks_to_array(hand)
    height, width = frame_shape[:2]
    xy = raw[:, :2] * np.asarray([width, height], dtype=np.float64)
    scale = float(np.median(np.linalg.norm(xy[[5, 9, 17]] - xy[0], axis=1)))
    palm_width = float(np.linalg.norm(xy[5] - xy[17]))
    shape = np.asarray(prediction.features, dtype=np.float64)
    if (scale <= 1e-6 or palm_width <= 1e-6 or shape.shape != (63,)
            or not np.isfinite(xy).all() or not np.isfinite(shape).all()):
        raise ValueError("Invalid palm geometry or shape")
    stable_label = prediction.stable_label
    if prediction.instant_label != stable_label:
        stable_label = None
    palm_center = np.mean(xy[[0, 5, 9, 17]], axis=0)
    return HandObservation(stable_label,
                           float(prediction.stable_confidence or 0),
                           scale, palm_width, tuple(map(float, xy[0])),
                           tuple(map(float, shape)), tuple(map(float, palm_center)))


class PushConfirmation:
    """One-shot latch: stable hold -> push -> one commit -> physical return."""

    def __init__(self, settings: PushSettings):
        settings.validate()
        self.settings = settings
        self.state = "IDLE"
        self.pending: str | None = None
        self.text = ""
        self._hold: deque[HandObservation] = deque(maxlen=settings.hold_frames)
        self._baseline_scale = 0.0
        self._baseline_width = 0.0
        self._baseline_wrist = np.zeros(2)
        self._baseline_shape = np.zeros(63)
        self._push_count = 0
        self._return_count = 0
        self.scale_ratio = 0.0

    def recalibrate(self) -> None:
        """Discard a calibration, but never unlock a confirmed push early."""
        if self.state not in ("CONFIRMED", "WAIT_RETURN"):
            self.state = "IDLE"
            self.pending = None
            self._hold.clear()
            self._push_count = 0

    def _eligible(self, obs: HandObservation | None) -> bool:
        return (obs is not None and obs.label is not None
                and obs.confidence >= self.settings.min_confidence
                and np.isfinite(obs.scale_px) and obs.scale_px > 0
                and np.isfinite(obs.width_px) and obs.width_px > 0)

    def update(self, obs: HandObservation | None, *, dynamic_active: bool = False) -> str | None:
        """Return a committed character on exactly one update, otherwise None."""
        if self.state == "CONFIRMED":
            self.state = "WAIT_RETURN"
        if dynamic_active:
            self.recalibrate()
            return None
        if self.state == "WAIT_RETURN":
            if obs is not None and self._baseline_scale > 0:
                self.scale_ratio = obs.scale_px / self._baseline_scale
                if self.scale_ratio <= self.settings.return_ratio:
                    self._return_count += 1
                    if self._return_count >= self.settings.return_frames:
                        self.state = "IDLE"
                        self.pending = None
                        self._hold.clear()
                        self._push_count = self._return_count = 0
                else:
                    self._return_count = 0
            else:
                self._return_count = 0  # lost hand is not proof of return
            return None
        if not self._eligible(obs):
            self.recalibrate()
            return None
        assert obs is not None
        if self.state == "IDLE":
            if self._hold and self._hold[-1].label != obs.label:
                self._hold.clear()
            self._hold.append(obs)
            if len(self._hold) == self.settings.hold_frames:
                scales = np.asarray([x.scale_px for x in self._hold])
                shapes = np.asarray([x.shape for x in self._hold])
                wrists = np.asarray([x.wrist_px for x in self._hold])
                wrist_span = float(np.max(np.linalg.norm(wrists - np.median(wrists, axis=0), axis=1))
                                   / np.median(scales))
                if (float(np.std(scales) / np.mean(scales)) <= self.settings.max_ready_scale_cv
                        and wrist_span <= self.settings.max_ready_wrist_span_palm_scales
                        and float(np.max(np.sqrt(np.mean((shapes - np.mean(shapes, axis=0)) ** 2, axis=1))))
                        <= self.settings.max_shape_rms):
                    self.pending = obs.label
                    self._baseline_scale = float(np.median(scales))
                    self._baseline_width = float(np.median([x.width_px for x in self._hold]))
                    self._baseline_wrist = np.median([x.wrist_px for x in self._hold], axis=0)
                    self._baseline_shape = np.median(shapes, axis=0)
                    self.state = "READY"
                else:
                    self._hold.popleft()
            return None
        if obs.label != self.pending:
            self.recalibrate()
            return None
        self.scale_ratio = obs.scale_px / self._baseline_scale
        width_ratio = obs.width_px / self._baseline_width
        lateral = float(np.linalg.norm(np.asarray(obs.wrist_px) - self._baseline_wrist)
                        / self._baseline_scale)
        shape_rms = float(np.sqrt(np.mean((np.asarray(obs.shape) - self._baseline_shape) ** 2)))
        if (lateral > self.settings.max_lateral_palm_scales
                or shape_rms > self.settings.max_shape_rms):
            self.recalibrate()
            return None
        if self.scale_ratio >= self.settings.push_ratio and width_ratio >= self.settings.width_ratio:
            self._push_count += 1
            if self.state == "PUSH_DETECTED":
                self.state = "CONFIRMED"
                self.text += self.pending or ""
                self._return_count = 0
                return self.pending
            if self._push_count >= self.settings.push_frames:
                self.state = "PUSH_DETECTED"
        else:
            self._push_count = 0
            self.state = "READY"
        return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental DOWN_CONFIRM for static letters, PUSH backup")
    parser.add_argument("--check-only", action="store_true", help="Verify artifacts without opening camera")
    parser.add_argument("--confirmation-mode", choices=("down", "push"), default="down",
                        help="Choose DOWN_CONFIRM (default) or the unchanged PUSH backup")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model", choices=("uvw_medium", "v1"), default=DEFAULT_MODEL)
    parser.add_argument("--no-mirror", action="store_true")
    for name, default, kind in (
        ("hold-frames", 10, int), ("push-frames", 2, int), ("return-frames", 4, int),
        ("min-confidence", .75, float), ("push-ratio", 1.20, float),
        ("width-ratio", 1.12, float), ("return-ratio", 1.07, float),
        ("max-ready-scale-cv", .06, float),
        ("max-ready-wrist-span-palm-scales", .35, float),
        ("max-lateral-palm-scales", .75, float),
        ("max-shape-rms", .28, float),
    ):
        parser.add_argument("--" + name, type=kind, default=default)
    for name, default, kind in (
        ("hold-frames", 7, int), ("down-frames", 2, int),
        ("return-frames", 4, int), ("min-confidence", .75, float),
        ("down-ratio", .25, float), ("return-ratio", .10, float),
        ("smoothing-alpha", .70, float), ("max-ready-scale-cv", .06, float),
        ("max-ready-center-span", .35, float),
        ("max-horizontal-ratio", .45, float),
        ("max-scale-change", .15, float), ("max-shape-rms", .28, float),
    ):
        flag = "--" + name if name.startswith("down-") else "--down-" + name
        parser.add_argument(flag, dest="down_" + name.replace("-", "_"),
                            type=kind, default=default)
    return parser.parse_args(argv)


def _overlay(frame: np.ndarray, lines: Sequence[str], *, first_line: int = 0) -> None:
    for index, line in enumerate(lines):
        origin = (18, 35 + (first_line + index) * 32)
        size, baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, .68, 2)
        cv2.rectangle(frame, (origin[0] - 6, origin[1] - size[1] - 5),
                      (origin[0] + size[0] + 6, origin[1] + baseline + 4),
                      (245, 245, 245), -1)
        cv2.putText(frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    .68, (0, 0, 0), 2, cv2.LINE_AA)


def run_camera(pipeline: StaticAslInference, confirmation: PushConfirmation | DownConfirmation,
               camera_index: int, mirror: bool, mode: str = "down") -> int:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        print("ERROR: Cannot open camera", file=sys.stderr)
        return 3
    started = time.perf_counter()
    last_timestamp = -1
    failed_reads = 0
    dynamic_guard = False
    frame_times: deque[float] = deque(maxlen=30)
    mediapipe_ms = 0.0
    inference_ms = 0.0
    try:
        with create_hand_landmarker(model_path=HAND_MODEL_PATH, running_mode="VIDEO",
                                    num_hands=1, min_hand_detection_confidence=.5,
                                    min_hand_presence_confidence=.5,
                                    min_tracking_confidence=.5) as landmarker:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    failed_reads += 1
                    pipeline.reset()
                    confirmation.update(None)
                    if failed_reads >= 10:
                        return 4
                    continue
                failed_reads = 0
                if mirror:
                    frame = cv2.flip(frame, 1)  # same orientation as static-letter runtime
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = max(int((time.perf_counter() - started) * 1000), last_timestamp + 1)
                last_timestamp = timestamp
                detection_started = time.perf_counter()
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                                data=rgb), timestamp)
                mediapipe_ms = (time.perf_counter() - detection_started) * 1000
                inference_ms = 0.0
                if result.hand_landmarks and not dynamic_guard:
                    hand = result.hand_landmarks[0]
                    try:
                        inference_started = time.perf_counter()
                        prediction = pipeline.predict_landmarks(hand)
                        observation = observe_hand(hand, prediction, frame.shape)
                        committed = confirmation.update(observation)
                        inference_ms = (time.perf_counter() - inference_started) * 1000
                        draw_hand(frame, hand)
                        movement_display = (
                            f"Down Y / hand scale: {confirmation.vertical_ratio:+.2f}"
                            if mode == "down" else
                            f"Scale / calibrated: {confirmation.scale_ratio:.2f}"
                        )
                        _overlay(frame, [f"Instant: {prediction.instant_label} {prediction.instant_confidence:.2f}",
                                         f"Stable: {prediction.stable_label or '-'}",
                                         movement_display])
                        if committed:
                            print(f"CONFIRMED={committed} TEXT={confirmation.text}", flush=True)
                    except (ValueError, IntegrationError):
                        pipeline.reset()
                        confirmation.update(None)
                else:
                    pipeline.reset()
                    confirmation.update(None, dynamic_active=dynamic_guard)
                frame_times.append(time.perf_counter())
                effective_fps = ((len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                                 if len(frame_times) > 1 and frame_times[-1] > frame_times[0] else 0.0)
                _overlay(frame, [f"Mode: {mode.upper()}  State: {confirmation.state}  Pending: {confirmation.pending or '-'}",
                                 f"Text: {confirmation.text[-35:]}",
                                 f"J/Z guard: {'ON' if dynamic_guard else 'OFF'} | D toggle C recalibrate Q quit"],
                         first_line=3)
                if mode == "down":
                    ready_ms = confirmation.last_ready_latency_ms
                    confirm_ms = confirmation.last_confirm_latency_ms
                    _overlay(frame, [
                        f"Stable->READY: {ready_ms:.0f} ms" if ready_ms is not None else "Stable->READY: -",
                        f"Down->CONFIRMED: {confirm_ms:.0f} ms" if confirm_ms is not None else "Down->CONFIRMED: -",
                        f"FPS: {effective_fps:.1f} | MP: {mediapipe_ms:.0f} ms | Infer: {inference_ms:.0f} ms",
                    ], first_line=6)
                cv2.imshow("Static letters - experimental push confirmation", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return 0
                if key == ord("c"):
                    confirmation.recalibrate()
                if key == ord("d"):
                    dynamic_guard = not dynamic_guard
                    pipeline.reset()
                    confirmation.recalibrate()
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.confirmation_mode == "down":
            settings = DownSettings(**{
                key: getattr(args, "down_" + key) for key in DownSettings.__dataclass_fields__
            })
            confirmation = DownConfirmation(settings)
        else:
            settings = PushSettings(**{
                key: getattr(args, key) for key in PushSettings.__dataclass_fields__
            })
            confirmation = PushConfirmation(settings)
        settings.validate()
        _validate_deployment_config()
        artifacts = load_official_artifacts(args.model)
        pipeline = StaticAslInference(artifacts)
    except (ValueError, IntegrationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.check_only:
        print(json.dumps({"status": "PASS", "model": artifacts.model_name,
                          "epoch": artifacts.best_epoch, "features": 63,
                          "confirmation_mode": args.confirmation_mode,
                          "classes": len(artifacts.labels), "hand_model_exists": HAND_MODEL_PATH.is_file(),
                          "camera_opened": False, "settings": vars(settings)}))
        return 0
    return run_camera(pipeline, confirmation, args.camera_index, not args.no_mirror,
                      mode=args.confirmation_mode)


if __name__ == "__main__":
    raise SystemExit(main())
