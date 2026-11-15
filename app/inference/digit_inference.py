import argparse
import importlib
from pathlib import Path
import sys

import cv2
import joblib
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 1. DUONG DAN PROJECT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MEDIAPIPE_MODEL = (
    ROOT
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)

DIGIT_MODEL = (
    ROOT
    / "models"
    / "digits"
    / "production"
    / "digits_final.joblib"
)


# ============================================================
# 2. IMPORT NORMALIZE LANDMARK
# ============================================================

if str(ROOT / "ai" / "preprocessing") not in sys.path:
    sys.path.insert(0, str(ROOT / "ai" / "preprocessing"))

from landmark_utils import normalize_landmarks


# ============================================================
# 3. KIEM TRA FILE
# ============================================================

def resolve_model_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Static digit camera (0-9)")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DIGIT_MODEL,
        help=f"Digit model to use (default: {DIGIT_MODEL})",
    )
    args = parser.parse_args(argv)
    model_path = resolve_model_path(args.model_path)

    if not MEDIAPIPE_MODEL.is_file():
        raise FileNotFoundError(f"Khong tim thay MediaPipe model:\n{MEDIAPIPE_MODEL}")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Digit model file not found at '{model_path}'. "
            "This public repository is distributed source-only without unverified pre-trained model weights. "
            "Please see docs/MODEL_DISTRIBUTION.md for details."
        )

    model = joblib.load(model_path)
    if getattr(model, "n_features_in_", None) != 63:
        raise ValueError("Digit model must accept 63 features")
    if set(map(str, model.classes_)) != {str(i) for i in range(10)}:
        raise ValueError("Digit model must have classes 0-9")
    alignment_error_type = None
    if "orientation_2d" in getattr(model, "named_steps", {}):
        # Each saved Pipeline owns its transformer. Keep old experimental
        # checkpoints loadable while final models import the stable module.
        transformer_module = model.named_steps["orientation_2d"].__class__.__module__
        if transformer_module not in {"hand_centric_2d", "orientation_2d_experimental"}:
            raise ValueError(f"Unsupported digit alignment module: {transformer_module}")
        alignment_error_type = importlib.import_module(transformer_module).HandAlignmentError
    print("Digit model loaded:", model_path)
    print("Classes:", model.classes_)
    print("Features:", model.n_features_in_)

    base_options = python.BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    camera = None
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError("Khong mo duoc camera.")

        print("\n==============================")
        print("STATIC DIGIT CAMERA")
        print("==============================")
        print("Nhan dien: 0-9")
        print("Nhan Q de thoat.\n")

        while True:
            success, frame = camera.read()
            if not success:
                print("Khong doc duoc frame camera.")
                break

            # BGR -> RGB; digit frames are deliberately not flipped.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            display_text = "NO HAND"
            confidence_text = ""

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                if len(hand) == 21:
                    features = normalize_landmarks(hand)
                    if len(features) == 63:
                        X = np.asarray(features, dtype=float).reshape(1, 63)
                        try:
                            prediction = model.predict(X)[0]
                            display_text = f"DIGIT: {prediction}"
                            if hasattr(model, "predict_proba"):
                                probabilities = model.predict_proba(X)[0]
                                confidence = float(np.max(probabilities))
                                confidence_text = f"Confidence: {confidence:.2f}"
                        except ValueError as error:
                            if alignment_error_type is None or not isinstance(error, alignment_error_type):
                                raise
                            display_text = "INVALID HAND ALIGNMENT"
                            confidence_text = ""

                        height, width = frame.shape[:2]
                        for landmark in hand:
                            x = int(landmark.x * width)
                            y = int(landmark.y * height)
                            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            cv2.putText(
                frame, display_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 2, cv2.LINE_AA,
            )
            if confidence_text:
                cv2.putText(
                    frame, confidence_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA,
                )
            cv2.putText(
                frame, "Q: Quit", (30, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.imshow("Static Digit Recognition 0-9", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("Camera closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
