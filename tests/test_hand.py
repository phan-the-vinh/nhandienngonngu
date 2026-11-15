import cv2
import mediapipe as mp
import time
from pathlib import Path

# =========================
# 1. Đường dẫn tới model
# =========================
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]

    MODEL_PATH = (
        ROOT
        / "models"
        / "mediapipe"
        / "hand_landmarker.task"
    )

    # =========================
    # 2. Các đường nối 21 điểm bàn tay
    # =========================
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Ngón cái
        (0, 5), (5, 6), (6, 7), (7, 8),        # Ngón trỏ
        (5, 9), (9, 10), (10, 11), (11, 12),   # Ngón giữa
        (9, 13), (13, 14), (14, 15), (15, 16), # Ngón áp út
        (13, 17), (17, 18), (18, 19), (19, 20),# Ngón út
        (0, 17)
    ]

    # =========================
    # 3. Kiểm tra model
    # =========================
    if not MODEL_PATH.exists():
        print("Khong tim thay model:")
        print(MODEL_PATH)
        exit()

    print("Da tim thay model:")
    print(MODEL_PATH)

    # =========================
    # 4. Tạo Hand Landmarker
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
    # 5. Mở camera
    # =========================
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Khong mo duoc camera")
        exit()

    print("MediaPipe dang chay.")
    print("Dua ban tay truoc camera.")
    print("Nhan Q de thoat.")

    start_time = time.time()

    with HandLandmarker.create_from_options(options) as landmarker:

        while True:

            ret, frame = camera.read()

            if not ret:
                print("Khong doc duoc camera")
                break

            # Lật camera giống soi gương
            frame = cv2.flip(frame, 1)

            # OpenCV: BGR -> RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Chuyển ảnh OpenCV sang MediaPipe Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Timestamp phải tăng dần
            timestamp_ms = int(
                (time.time() - start_time) * 1000
            )

            # MediaPipe phát hiện bàn tay
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # =========================
            # 6. Nếu tìm thấy bàn tay
            # =========================
            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                height, width, _ = frame.shape

                points = []

                # Chuyển tọa độ 0-1 thành tọa độ pixel
                for landmark in hand:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    points.append((x, y))

                # Vẽ các đường nối
                for start, end in HAND_CONNECTIONS:

                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (0, 255, 0),
                        2
                    )

                # Vẽ 21 điểm
                for i, point in enumerate(points):

                    cv2.circle(
                        frame,
                        point,
                        5,
                        (0, 0, 255),
                        -1
                    )

                    # Hiện số thứ tự landmark
                    cv2.putText(
                        frame,
                        str(i),
                        point,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 0),
                        1
                    )

            cv2.imshow(
                "MediaPipe - 21 Hand Landmarks",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()
