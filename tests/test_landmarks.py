import cv2
import mediapipe as mp
import time
import sys

from pathlib import Path


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]

    sys.path.insert(
        0,
        str(ROOT / "ai" / "preprocessing")
    )

    from landmark_utils import normalize_landmarks


    # =========================
    # 1. Duong dan model
    # =========================
    MODEL_PATH = (
        ROOT
        / "models"
        / "mediapipe"
        / "hand_landmarker.task"
    )

    if not MODEL_PATH.exists():
        print("Khong tim thay model:")
        print(MODEL_PATH)
        exit()


    # =========================
    # 2. Khoi tao MediaPipe
    # =========================
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )


    # =========================
    # 3. Mo camera
    # =========================
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Khong mo duoc camera")
        exit()

    print("===== TEST CHUAN HOA LANDMARK =====")
    print("Dua ban tay truoc camera.")
    print("Nhan S de xem du lieu.")
    print("Nhan Q de thoat.")

    start_time = time.perf_counter()
    last_timestamp = 0


    # =========================
    # 4. Chay MediaPipe
    # =========================
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:

            ret, frame = camera.read()

            if not ret:
                print("Khong doc duoc camera")
                break

            # Lat hinh giong nhu soi guong
            frame = cv2.flip(frame, 1)

            # OpenCV dung BGR
            # MediaPipe dung RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Chuyen sang MediaPipe Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Timestamp tang dan
            timestamp_ms = int(
                (time.perf_counter() - start_time) * 1000
            )

            if timestamp_ms <= last_timestamp:
                timestamp_ms = last_timestamp + 1

            last_timestamp = timestamp_ms

            # Nhan dien ban tay
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            hand = None

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                cv2.putText(
                    frame,
                    "Da phat hien ban tay",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "Chua phat hien ban tay",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.putText(
                frame,
                "S: xem du lieu | Q: thoat",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Test Normalize Landmarks",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # =========================
            # 5. Nhan S de xem 63 dac trung
            # =========================
            if key == ord("s"):

                if hand is None:

                    print(
                        "\nKhong co ban tay de lay du lieu."
                    )

                else:

                    features = normalize_landmarks(
                        hand
                    )

                    print("\n==========================")
                    print("So dac trung:", len(features))

                    print("10 gia tri dau:")

                    for i in range(10):
                        print(
                            f"f{i + 1}: {features[i]:.4f}"
                        )

                    print("==========================")

            # Nhan Q de thoat
            if key == ord("q"):
                break


    camera.release()
    cv2.destroyAllWindows()
