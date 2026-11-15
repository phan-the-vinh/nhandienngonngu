"""Manually capture digit camera error cases into temporary digits_v2 data.

Run from the project root, for example:
    python tools/collect_digit_error_cases.py --true-label 4 --view front --session-id 01

Press S to save the current valid hand frame, or Q to quit. Recording a new
frame does not establish an independent gesture bout; use --gesture-bout-id to
identify frames from the same performance when that information is known.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "data" / "experiments" / "digits_v2"
RAW_DIR = EXPERIMENT_ROOT / "raw" / "error_cases"
METADATA_DIR = EXPERIMENT_ROOT / "metadata"
MODEL_PATH = ROOT / "models" / "digits" / "production" / "mlp_large_v2a.joblib"
MODEL_MANIFEST = ROOT / "models" / "digits" / "production" / "deployment_manifest_v1.json"
MEDIAPIPE_PATH = ROOT / "models" / "mediapipe" / "hand_landmarker.task"
WINDOW_NAME = "Digit v2 error-case collector"

# The canonical implementation used by the production digit camera and training.
sys.path.insert(0, str(ROOT / "ai" / "preprocessing"))
from landmark_utils import normalize_landmarks  # noqa: E402


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def ascii_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9]+", value):
        raise argparse.ArgumentTypeError("Use only ASCII letters and digits")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--true-label", required=True, type=int, choices=range(10))
    parser.add_argument(
        "--view", required=True,
        choices=("front", "left", "right", "top", "bot", "dif"),
    )
    parser.add_argument("--session-id", required=True, type=ascii_id)
    parser.add_argument("--gesture-bout-id", type=ascii_id, default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--camera-index", type=int, default=0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model():
    """Load only at camera start; --help and module import remain camera-free."""
    if not MODEL_PATH.is_file() or not MEDIAPIPE_PATH.is_file():
        raise FileNotFoundError("Production digit model or MediaPipe task is missing")
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    expected_hash = manifest["model"]["sha256"]
    actual_hash = sha256_file(MODEL_PATH)
    if actual_hash != expected_hash:
        raise ValueError("Production digit model SHA256 differs from deployment manifest")
    if sha256_file(MEDIAPIPE_PATH) != manifest["mediapipe"]["sha256"]:
        raise ValueError("MediaPipe task SHA256 differs from deployment manifest")

    # The production artifact includes its own StandardScaler; never fit it here.
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = joblib.load(MODEL_PATH)
    if (
        not isinstance(model, Pipeline)
        or len(model.steps) != 2
        or not isinstance(model.steps[0][1], StandardScaler)
        or not isinstance(model.steps[1][1], MLPClassifier)
        or model.n_features_in_ != 63
        or set(str(label) for label in model.classes_) != set("0123456789")
    ):
        raise ValueError("Unexpected production digit Pipeline, features, or label mapping")
    return model, actual_hash


def draw_landmarks(frame: np.ndarray, hand) -> np.ndarray:
    overlay = frame.copy()
    height, width = overlay.shape[:2]
    points = [(round(item.x * width), round(item.y * height)) for item in hand]
    for start, end in HAND_EDGES:
        cv2.line(overlay, points[start], points[end], (0, 180, 255), 2)
    for index, point in enumerate(points):
        color = (0, 0, 255) if index in (4, 8, 12) else (0, 255, 0)
        cv2.circle(overlay, point, 4, color, -1)
        cv2.putText(overlay, str(index), (point[0] + 4, point[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return overlay


def next_sample_paths(label: int, session: str, view: str):
    prefix = f"digit{label}_session{session}_{view}"
    for index in range(1, 1_000_000):
        stem = f"{prefix}_{index:04d}"
        raw = RAW_DIR / f"{stem}.png"
        overlay = RAW_DIR / f"{stem}_overlay.png"
        metadata = METADATA_DIR / f"{stem}.json"
        reserve = METADATA_DIR / f".{stem}.reserve"
        if any(path.exists() for path in (raw, overlay, metadata, reserve)):
            continue
        try:
            with reserve.open("x", encoding="ascii") as handle:
                handle.write(datetime.now(timezone.utc).isoformat())
        except FileExistsError:
            continue
        return stem, raw, overlay, metadata, reserve
    raise RuntimeError("No free sample index is available")


def write_png_exclusive(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Could not encode {path}")
    with path.open("xb") as target:
        target.write(encoded.tobytes())


def project_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def save_sample(args, frame, hand, features, prediction, probabilities,
                confidence, handedness, model_hash):
    stem, raw_path, overlay_path, metadata_path, reserve = next_sample_paths(
        args.true_label, args.session_id, args.view
    )
    # Metadata is written last. A failed save leaves an inspectable partial
    # capture and its reservation, never silently reusing its sample ID.
    try:
        write_png_exclusive(raw_path, frame)
        write_png_exclusive(overlay_path, draw_landmarks(frame, hand))
        metadata = {
            "schema_version": 1,
            "sample_id": stem,
            "true_label": args.true_label,
            "predicted_label": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "view": args.view,
            "session_id": args.session_id,
            "gesture_bout_id": args.gesture_bout_id,
            "handedness": handedness,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_image_path": project_relative(raw_path),
            "overlay_image_path": project_relative(overlay_path),
            "notes": args.notes,
            "landmarks": [
                {"index": index, "x": float(point.x), "y": float(point.y),
                 "z": float(point.z)}
                for index, point in enumerate(hand)
            ],
            "features": {f"f{index}": float(value)
                         for index, value in enumerate(features, start=1)},
            "feature_count": 63,
            "normalization": "ai/preprocessing/landmark_utils.py::normalize_landmarks",
            "model_path": project_relative(MODEL_PATH),
            "model_sha256": model_hash,
            "mediapipe_task_path": project_relative(MEDIAPIPE_PATH),
            "frame_flipped_before_mediapipe": False,
        }
        with metadata_path.open("x", encoding="utf-8") as target:
            json.dump(metadata, target, ensure_ascii=False, indent=2)
            target.write("\n")
    except Exception:
        print(f"Incomplete capture {stem}; inspect reservation and partial files", file=sys.stderr)
        raise
    else:
        reserve.unlink()
    return stem


def show_text(display, lines):
    for row, line in enumerate(lines):
        cv2.putText(display, line, (12, 28 + 30 * row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def run(args: argparse.Namespace) -> None:
    if args.camera_index < 0:
        raise ValueError("camera-index must be nonnegative")
    model, model_hash = load_model()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_ROOT / "processed").mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_ROOT / "splits").mkdir(parents=True, exist_ok=True)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MEDIAPIPE_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Cannot open camera")

    saved_count = 0
    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Cannot read camera frame")

                # Exactly the production order: unflipped BGR frame -> RGB ->
                # MediaPipe IMAGE mode -> canonical 63-feature normalizer.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                current = None
                if result.hand_landmarks and len(result.hand_landmarks[0]) == 21:
                    hand = result.hand_landmarks[0]
                    landmark_values = np.asarray(
                        [(p.x, p.y, p.z) for p in hand], dtype=float
                    )
                    if np.isfinite(landmark_values).all():
                        features = np.asarray(normalize_landmarks(hand), dtype=float)
                        if features.shape == (63,) and np.isfinite(features).all():
                            sample = features.reshape(1, 63)
                            predicted = int(model.predict(sample)[0])
                            probabilities = None
                            confidence = None
                            if hasattr(model, "predict_proba"):
                                values = model.predict_proba(sample)[0]
                                probabilities = {
                                    str(label): float(probability)
                                    for label, probability in zip(model.classes_, values)
                                }
                                confidence = probabilities[str(predicted)]
                            handedness = None
                            if result.handedness and result.handedness[0]:
                                handedness = result.handedness[0][0].category_name
                            current = (hand, features, predicted, probabilities,
                                       confidence, handedness)

                display = frame.copy()
                status = "NO VALID HAND" if current is None else f"PREDICTED: {current[2]}"
                confidence_text = "N/A" if current is None or current[4] is None else f"{current[4]:.3f}"
                show_text(display, (
                    f"TRUE LABEL: {args.true_label}   {status}",
                    f"CONFIDENCE: {confidence_text}   VIEW: {args.view}",
                    f"SESSION: {args.session_id}   SAVED COUNT: {saved_count}",
                    "S: SAVE current frame   Q: QUIT",
                ))
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("s"), ord("S")):
                    if current is None:
                        print("No valid hand in current frame; nothing saved")
                        continue
                    stem = save_sample(args, frame, *current, model_hash)
                    saved_count += 1
                    print(f"Saved {stem}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(parse_args())
