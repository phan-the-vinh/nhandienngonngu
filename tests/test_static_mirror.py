"""Archived legacy camera implementation.

The executable entry point now delegates to ``AI/static_asl_camera.py`` so the
project has one maintained camera inference pipeline.

import mediapipe as mp
import pandas as pd
import joblib
import time
import sys

from pathlib import Path


# =====================================================
# 1. DUONG DAN THU MUC AI
# =====================================================

AI_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(AI_DIR)
)

from landmark_utils import normalize_landmarks


# =====================================================
# 2. DUONG DAN MODEL
# =====================================================

HAND_MODEL_PATH = (
    AI_DIR
    / "models"
    / "hand_landmarker.task"
)

MLP_MODEL_PATH = (
    AI_DIR
    / "models"
    / "mlp_static_mirror.pkl"
)


# =====================================================
# 3. KIEM TRA FILE
# =====================================================

if not HAND_MODEL_PATH.exists():
    print("Khong tim thay HandLandmarker:")
    print(HAND_MODEL_PATH)
    exit()

if not MLP_MODEL_PATH.exists():
    print("Khong tim thay model:")
    print(MLP_MODEL_PATH)
    exit()


# =====================================================
# 4. LOAD MODEL MLP
# =====================================================

model = joblib.load(
    MLP_MODEL_PATH
)

feature_columns = [
    f"f{i}"
    for i in range(1, 64)
]


print("Da load model:")
print(MLP_MODEL_PATH)


# =====================================================
# 5. DUONG NOI 21 LANDMARK
# =====================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (5, 9), (9, 10), (10, 11), (11, 12),

    (9, 13), (13, 14), (14, 15), (15, 16),

    (13, 17), (17, 18), (18, 19), (19, 20),

    (0, 17)
]


def draw_hand(frame, hand):
    height, width, _ = frame.shape

    points = []

    for landmark in hand:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # Ve cac duong noi
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )

    # Ve 21 diem
    for point in points:
        cv2.circle(
            frame,
            point,
            4,
            (0, 0, 255),
            -1
        )


# =====================================================
# 6. KHOI TAO MEDIAPIPE
# =====================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=str(
            HAND_MODEL_PATH
        )
    ),

    running_mode=RunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# =====================================================
# 7. MO CAMERA
# =====================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Khong mo duoc camera.")
    exit()


print("\n===== TEST STATIC MIRROR =====")
print("Model: mlp_static_mirror.pkl")
print("Hien dang test cac ky tu da train.")
print("Nhan Q de thoat.")


start_time = time.perf_counter()
last_timestamp = 0


# =====================================================
# 8. CHAY REALTIME
# =====================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Khong doc duoc camera.")
            break


        # Lat camera nhu guong
        frame = cv2.flip(
            frame,
            1
        )


        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        timestamp_ms = int(
            (
                time.perf_counter()
                - start_time
            )
            * 1000
        )


        if timestamp_ms <= last_timestamp:
            timestamp_ms = (
                last_timestamp + 1
            )

        last_timestamp = timestamp_ms


        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # =================================================
        # 9. NEU TIM THAY BAN TAY
        # =================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            draw_hand(
                frame,
                hand
            )


            # =============================================
            # CHUAN HOA LANDMARK
            # =============================================

            features = normalize_landmarks(
                hand
            )


            X = pd.DataFrame(
                [features],
                columns=feature_columns
            )


            # =============================================
            # DU DOAN
            # =============================================

            prediction = model.predict(
                X
            )[0]


            probabilities = model.predict_proba(
                X
            )[0]


            confidence = float(
                max(probabilities)
            )


            # =============================================
            # HIEN KET QUA
            # =============================================

            cv2.putText(
                frame,
                f"Prediction: {prediction}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )


            cv2.putText(
                frame,
                f"Confidence: {confidence * 100:.1f}%",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )


        else:

            cv2.putText(
                frame,
                "Khong tim thay ban tay",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2
            )


        # =================================================
        # 10. HIEN CAMERA
        # =================================================

        cv2.imshow(
            "STATIC MIRROR - Realtime Test",
            frame
        )


        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        if key == ord("q"):
            break


# =====================================================
# 11. DONG CAMERA
# =====================================================

camera.release()

cv2.destroyAllWindows()

print("Da dung test STATIC MIRROR.")
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.inference.letter_inference import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
