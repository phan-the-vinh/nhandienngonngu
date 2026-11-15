"""Standalone HOLD_CONFIRM camera experiment for the 24 static letters.

No camera is opened by import or by --check-only. J/Z are not supported.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.feature_extraction.hand_landmarks import create_hand_landmarker  # noqa: E402
from app.inference.letter_inference import (  # noqa: E402
    DEFAULT_MODEL, HAND_MODEL_PATH, IntegrationError, StaticAslInference,
    _validate_deployment_config, draw_hand, load_official_artifacts,
)
from app.inference.push_confirmation_camera_test import observe_hand  # noqa: E402
from app.interaction.hold_confirmation import HoldConfirmation, HoldSettings  # noqa: E402

FONT_PATH = Path("C:/Windows/Fonts/segoeui.ttf")
PANEL_WIDTH = 310


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone 3-second static-letter HOLD_CONFIRM test")
    parser.add_argument("--check-only", action="store_true", help="Load/check artifacts without camera")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model", choices=("uvw_medium", "v1"), default=DEFAULT_MODEL)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=.75)
    parser.add_argument("--max-center-drift-ratio", type=float, default=.25)
    parser.add_argument("--max-scale-change", type=float, default=.18)
    parser.add_argument("--max-shape-rms", type=float, default=.28)
    parser.add_argument("--release-seconds", type=float, default=.60)
    parser.add_argument("--min-new-pose-shape-rms", type=float, default=.06)
    return parser.parse_args(argv)


@lru_cache(maxsize=1)
def _confirmation_font() -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), 19)


def _panel(frame: np.ndarray, machine: HoldConfirmation, current_label: str | None,
           now: float) -> np.ndarray:
    height = frame.shape[0]
    panel = np.full((height, PANEL_WIDTH, 3), 245, dtype=np.uint8)
    center = (PANEL_WIDTH // 2, min(155, max(105, height // 3)))
    radius = 75
    progress = machine.progress if machine.state in ("HOLDING", "CONFIRMED") else 0.0
    cv2.circle(panel, center, radius, (205, 205, 205), 10, cv2.LINE_AA)
    if progress > 0:
        cv2.ellipse(panel, center, (radius, radius), 0, -90,
                    -90 + 360 * progress, (65, 170, 45), 10, cv2.LINE_AA)

    shown = (machine.pending if machine.state in ("STABILIZING", "HOLDING")
             else current_label) or "-"
    size, _ = cv2.getTextSize(shown, cv2.FONT_HERSHEY_SIMPLEX, 2.8, 4)
    cv2.putText(panel, shown, (center[0] - size[0] // 2, center[1] + size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0, 0, 0), 4, cv2.LINE_AA)

    below = center[1] + radius + 36
    title = "HOLD POSE" if machine.state in ("STABILIZING", "HOLDING") else machine.state
    cv2.putText(panel, title, (20, below), cv2.FONT_HERSHEY_SIMPLEX,
                .70, (0, 0, 0), 2, cv2.LINE_AA)
    countdown = f"{machine.remaining_seconds:.1f}s remaining"
    cv2.putText(panel, countdown, (20, below + 31), cv2.FONT_HERSHEY_SIMPLEX,
                .62, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(panel, f"Stable: {machine.stable_count}/{machine.settings.stable_frames}",
                (20, below + 59), cv2.FONT_HERSHEY_SIMPLEX,
                .53, (0, 0, 0), 1, cv2.LINE_AA)

    if (machine.last_confirmed_at is not None
            and now - machine.last_confirmed_at < 1.0
            and machine.state in ("CONFIRMED", "WAIT_RELEASE")):
        tick_y = min(height - 63, below + 97)
        cv2.line(panel, (22, tick_y - 8), (31, tick_y), (45, 170, 45), 5, cv2.LINE_AA)
        cv2.line(panel, (31, tick_y), (49, tick_y - 19), (45, 170, 45), 5, cv2.LINE_AA)
        image = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(image).text((59, tick_y - 24), "ĐÃ XÁC NHẬN",
                                   font=_confirmation_font(), fill=(0, 0, 0))
        panel = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    cv2.putText(panel, "TEXT:", (20, height - 57), cv2.FONT_HERSHEY_SIMPLEX,
                .60, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(panel, machine.text[-22:] or "-", (20, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX, .72, (0, 0, 0), 2, cv2.LINE_AA)
    return np.concatenate((frame, panel), axis=1)


def run_camera(pipeline: StaticAslInference, machine: HoldConfirmation,
               camera_index: int, mirror: bool) -> int:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        print("ERROR: Cannot open camera", file=sys.stderr)
        return 3
    started = time.monotonic()
    last_timestamp = -1
    failed_reads = 0
    try:
        with create_hand_landmarker(model_path=HAND_MODEL_PATH, running_mode="VIDEO",
                                    num_hands=1, min_hand_detection_confidence=.5,
                                    min_hand_presence_confidence=.5,
                                    min_tracking_confidence=.5) as landmarker:
            while True:
                ok, frame = capture.read()
                now = time.monotonic()
                if not ok or frame is None:
                    failed_reads += 1
                    pipeline.reset()
                    machine.update(None, timestamp=now)
                    if failed_reads >= 10:
                        return 4
                    continue
                failed_reads = 0
                if mirror:
                    frame = cv2.flip(frame, 1)  # Same orientation as the static-letter camera.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = max(int((now - started) * 1000), last_timestamp + 1)
                last_timestamp = timestamp_ms
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                current_label = None
                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]
                    try:
                        prediction = pipeline.predict_landmarks(hand)
                        observation = observe_hand(hand, prediction, frame.shape)
                        current_label = prediction.stable_label or prediction.instant_label
                        committed = machine.update(observation, timestamp=now)
                        if committed is not None:
                            print(f"CONFIRMED={committed} TEXT={machine.text}", flush=True)
                            pipeline.reset()  # No votes from this letter enter the next attempt.
                        elif machine.rearmed and machine.state == "IDLE":
                            pipeline.reset()  # Fresh votes after a completed no-hand release.
                        draw_hand(frame, hand)
                    except (ValueError, IntegrationError):
                        pipeline.reset()
                        machine.update(None, timestamp=now)
                else:
                    pipeline.reset()
                    machine.update(None, timestamp=now)
                cv2.imshow("Static letters - HOLD_CONFIRM experiment",
                           _panel(frame, machine, current_label, now))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return 0
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = HoldSettings(
            hold_seconds=args.hold_seconds, stable_frames=args.stable_frames,
            min_confidence=args.min_confidence,
            max_center_drift_ratio=args.max_center_drift_ratio,
            max_scale_change=args.max_scale_change,
            max_shape_rms=args.max_shape_rms,
            release_seconds=args.release_seconds,
            min_new_pose_shape_rms=args.min_new_pose_shape_rms,
        )
        settings.validate()
        if not FONT_PATH.is_file():
            raise OSError(f"Unicode UI font missing: {FONT_PATH}")
        _confirmation_font()
        _validate_deployment_config()
        artifacts = load_official_artifacts(args.model)
        pipeline = StaticAslInference(artifacts)
    except (OSError, ValueError, IntegrationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.check_only:
        print(json.dumps({"status": "PASS", "model": artifacts.model_name,
                          "epoch": artifacts.best_epoch, "features": 63,
                          "classes": len(artifacts.labels), "hand_model_exists": HAND_MODEL_PATH.is_file(),
                          "camera_opened": False, "settings": vars(settings)}))
        return 0
    return run_camera(pipeline, HoldConfirmation(settings), args.camera_index,
                      not args.no_mirror)


if __name__ == "__main__":
    raise SystemExit(main())
