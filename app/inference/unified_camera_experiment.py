"""One-camera ABC/123 + HOLD experiment; J/Z is explicitly disabled.

The binary J/Z model is integrity-checked, but no automatic gesture boundary or
NONE rejection exists. It is never queried on arbitrary live frames and cannot
commit a J/Z character. This is not a production dispatcher replacement.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.feature_extraction.hand_landmarks import create_hand_landmarker  # noqa: E402
from app.inference.dynamic_jz_camera_test import (  # noqa: E402
    _synthetic_feature_check, load_experiment_model,
)
from app.inference.hold_confirmation_camera_test import _panel  # noqa: E402
from app.inference.letter_inference import (  # noqa: E402
    DEFAULT_MODEL, HAND_MODEL_PATH, IntegrationError, StaticAslInference,
    _validate_deployment_config, draw_hand, load_official_artifacts,
)
from app.inference.unified_static_coordinator import (  # noqa: E402
    DigitPredictor, RecognitionCoordinator, load_digit_model,
)
from app.interaction.hold_confirmation import HoldSettings  # noqa: E402

WINDOW_NAME = "Unified ABC / 123 - HOLD experiment"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify models without opening camera")
    parser.add_argument("--camera-index", type=int, default=0)
    return parser.parse_args(argv)


def _draw_mode_controls(display, video_width: int, active_mode: str) -> None:
    for label, left, right in (("ABC", 14, 145), ("123", 165, 296)):
        x0, x1 = video_width + left, video_width + right
        color = (218, 240, 218) if label == active_mode else (235, 235, 235)
        cv2.rectangle(display, (x0, 8), (x1, 47), color, -1)
        cv2.rectangle(display, (x0, 8), (x1, 47), (65, 65, 65), 1)
        cv2.putText(display, label, (x0 + 34, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    .75, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(display, (video_width + 14, 52), (video_width + 296, 77),
                  (218, 240, 218) if active_mode == "JZ_PAUSED" else (235, 235, 235), -1)
    cv2.putText(display, "J/Z PAUSE: DYNAMIC DISABLED", (video_width + 19, 69),
                cv2.FONT_HERSHEY_SIMPLEX, .39, (0, 0, 0), 1, cv2.LINE_AA)


def mode_from_click(x: int, y: int, video_width: int) -> str | None:
    offset = x - video_width
    if 52 <= y <= 77 and 14 <= offset <= 296:
        return "JZ_PAUSED"
    if not 8 <= y <= 47:
        return None
    if 14 <= offset <= 145:
        return "ABC"
    if 165 <= offset <= 296:
        return "123"
    return None


def prepare_inference_frame(raw_frame, mode: str):
    """Preserve the model contracts: ABC mirrored, digits unmirrored."""
    return cv2.flip(raw_frame, 1) if mode == "ABC" else raw_frame


def render_mirrored_frame(inference_frame, mode: str, hand=None):
    """Render on a copy, then mirror pixels AND overlays for digits/pause.

    Landmark coordinates remain in their original inference coordinate system.
    UI text and buttons are added afterwards so they are never reversed.
    """
    display_frame = inference_frame.copy()
    if hand is not None:
        draw_hand(display_frame, hand)
    return display_frame if mode == "ABC" else cv2.flip(display_frame, 1)


def run_camera(coordinator: RecognitionCoordinator, camera_index: int) -> int:
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        print("ERROR: Cannot open camera", file=sys.stderr)
        return 3
    requested_mode = [coordinator.mode]
    video_width = [0]

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            selected = mode_from_click(x, y, video_width[0])
            if selected is not None:
                requested_mode[0] = selected

    started = time.monotonic()
    last_timestamp = -1
    failed_reads = 0
    try:
        # One task instance and one camera for both static modes. VIDEO mode is
        # retained for letters; digit features are still unmirrored/canonical.
        with create_hand_landmarker(model_path=HAND_MODEL_PATH, running_mode="VIDEO",
                                    num_hands=1, min_hand_detection_confidence=.5,
                                    min_hand_presence_confidence=.5,
                                    min_tracking_confidence=.5) as landmarker:
            cv2.namedWindow(WINDOW_NAME)
            cv2.setMouseCallback(WINDOW_NAME, on_mouse)
            while True:
                if requested_mode[0] != coordinator.mode:
                    coordinator.switch_mode(requested_mode[0])
                ok, raw_frame = camera.read()
                now = time.monotonic()
                if not ok or raw_frame is None:
                    failed_reads += 1
                    coordinator.missing_hand(now)
                    if failed_reads >= 10:
                        return 4
                    continue
                failed_reads = 0
                # Preserve each model's input orientation: letters mirror,
                # digits do not. Only the active model receives this hand.
                inference_frame = prepare_inference_frame(raw_frame, coordinator.mode)
                video_width[0] = inference_frame.shape[1]
                current_label = None
                display_hand = None
                if coordinator.mode == "JZ_PAUSED":
                    coordinator.missing_hand(now)
                else:
                    rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
                    timestamp_ms = max(int((now - started) * 1000), last_timestamp + 1)
                    last_timestamp = timestamp_ms
                    result = landmarker.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                    if result.hand_landmarks:
                        hand = result.hand_landmarks[0]
                        try:
                            prediction, committed = coordinator.process_hand(hand, inference_frame.shape, now)
                            current_label = prediction.stable_label or prediction.instant_label
                            if committed:
                                print(f"CONFIRMED={committed} TEXT={coordinator.composer.text}", flush=True)
                            display_hand = hand
                        except (ValueError, IntegrationError):
                            coordinator.missing_hand(now)
                    else:
                        coordinator.missing_hand(now)
                display_frame = render_mirrored_frame(inference_frame, coordinator.mode, display_hand)
                display = _panel(display_frame, coordinator.hold, current_label, now)
                _draw_mode_controls(display, video_width[0], coordinator.mode)
                cv2.imshow(WINDOW_NAME, display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _validate_deployment_config()
        letter_artifacts = load_official_artifacts(DEFAULT_MODEL)
        letters = StaticAslInference(letter_artifacts)
        digit_model = load_digit_model()
        digits = DigitPredictor(digit_model)
        dynamic_model, dynamic_hash = load_experiment_model()
        _synthetic_feature_check(dynamic_model)  # 16x69 contract, no camera/video.
        if not HAND_MODEL_PATH.is_file():
            raise FileNotFoundError(HAND_MODEL_PATH)
        coordinator = RecognitionCoordinator(letters, digits, HoldSettings())
    except (OSError, ValueError, IntegrationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.check_only:
        print(json.dumps({
            "status": "PASS", "camera_opened": False,
            "camera_count_when_running": 1, "landmarker_count_when_running": 1,
            "letters_model": letter_artifacts.model_name,
            "letters_epoch": letter_artifacts.best_epoch,
            "digits_model": "digits_final",
            "digits_classes": digits.labels,
            "dynamic_model_sha256": dynamic_hash,
            "dynamic_feature_shape": [16, 69],
            "dynamic_live_status": "EXPERIMENTAL_DISABLED_NO_NONE_OR_BOUNDARY_DETECTOR",
            "hold_seconds": coordinator.hold.settings.hold_seconds,
            "modes": ["ABC", "123", "JZ_PAUSED"],
        }))
        return 0
    return run_camera(coordinator, args.camera_index)


if __name__ == "__main__":
    raise SystemExit(main())
