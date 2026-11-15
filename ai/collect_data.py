import cv2
import mediapipe as mp
import time
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "preprocessing"))

from landmark_utils import normalize_landmarks  # noqa: E402


# =====================================================
# 1. CAU HINH THU DU LIEU
# =====================================================

CAPTURE_DURATION = 10.0      # Thu moi ky tu trong 10 giay
CAPTURE_INTERVAL = 0.20      # Moi 0.2 giay lay 1 mau

# 10 / 0.2 ~= 50 mau / ky tu


# =====================================================
# 2. DUONG NOI 21 LANDMARK
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
# 3. DUONG DAN
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)

LETTERS_PATH = (
    BASE_DIR
    / "config"
    / "ky_tu.txt"
)

RAW_FILE = (
    BASE_DIR
    / "raw_data"
    / "raw_landmarks.csv"
)

DATASET_FILE = (
    BASE_DIR
    / "dataset"
    / "dataset.csv"
)


# =====================================================
# 4. KIEM TRA FILE
# =====================================================

if not MODEL_PATH.exists():
    print("Khong tim thay model:")
    print(MODEL_PATH)
    exit()

if not LETTERS_PATH.exists():
    print("Khong tim thay ky_tu.txt")
    exit()


# =====================================================
# 5. DOC DANH SACH KY TU
# =====================================================

with open(
    LETTERS_PATH,
    "r",
    encoding="utf-8"
) as file:

    all_letters = [
        line.strip().upper()
        for line in file
        if line.strip()
    ]


print("Danh sach ky tu:")
print(" ".join(all_letters))


# =====================================================
# 6. NHAP NGUOI THU
# =====================================================

person_id = input(
    "\nNhap ma nguoi thu "
    "(VD: person_01): "
).strip()


if person_id == "":
    print("Ma nguoi khong duoc de trong.")
    exit()


# =====================================================
# 7. CHON KY TU CAN THU
# =====================================================

print("\nNhap ALL neu muon thu tat ca.")
print("Hoac nhap VD: ABC")


choice = input(
    "Ky tu can thu: "
).strip().upper()


if choice == "ALL":
    letters = all_letters

else:
    letters = []

    for letter in choice:

        if (
            letter in all_letters
            and letter not in letters
        ):
            letters.append(letter)


if len(letters) == 0:
    print("Khong co ky tu hop le.")
    exit()


print("\nSe thu:")
print(" ".join(letters))

print(
    f"Moi ky tu: {CAPTURE_DURATION:.0f} giay"
)

print(
    f"Khoang {int(CAPTURE_DURATION / CAPTURE_INTERVAL)} mau / ky tu"
)


# =====================================================
# 8. TAO THU MUC
# =====================================================

RAW_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

DATASET_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 9. TAO HEADER CSV
# =====================================================

raw_header = [
    "person",
    "label",
    "handedness"
]

for i in range(21):
    raw_header.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])


dataset_header = [
    "person",
    "label",
    "handedness"
]

for i in range(63):
    dataset_header.append(
        f"f{i + 1}"
    )


if not RAW_FILE.exists():

    with open(
        RAW_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)
        writer.writerow(raw_header)


if not DATASET_FILE.exists():

    with open(
        DATASET_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)
        writer.writerow(dataset_header)


# =====================================================
# 10. KHOI TAO MEDIAPIPE
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
        model_asset_path=str(MODEL_PATH)
    ),

    running_mode=RunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# =====================================================
# 11. MO CAMERA
# =====================================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():
    print("Khong mo duoc camera.")
    exit()


print("\n===== THU DU LIEU =====")
print("SPACE = bat dau thu ky tu")
print("ENTER = luu khi thu xong")
print("R = thu lai ky tu")
print("Q = thoat")


# =====================================================
# 12. CAC BIEN TRANG THAI
# =====================================================

current_letter_index = 0

collecting = False
waiting_confirmation = False

capture_start_time = 0
last_capture_time = 0

sample_count = 0


# Tam luu du lieu cua ky tu hien tai
current_raw_rows = []
current_dataset_rows = []


start_time = time.perf_counter()
last_timestamp = 0


# =====================================================
# 13. CHAY
# =====================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        if current_letter_index >= len(letters):
            break


        current_letter = (
            letters[current_letter_index]
        )


        ret, frame = camera.read()

        if not ret:
            print("Khong doc duoc camera.")
            break


        # Lat nhu soi guong
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
        # 14. LAY BAN TAY
        # =================================================

        hand = None
        handedness = "Unknown"


        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            draw_hand(
                frame,
                hand
            )


            if result.handedness:

                handedness = (
                    result
                    .handedness[0][0]
                    .category_name
                )


        # =================================================
        # 15. HIEN KY TU
        # =================================================

        cv2.putText(
            frame,
            f"Letter: {current_letter}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )


        # =================================================
        # 16. CHUA BAT DAU
        # =================================================

        if (
            not collecting
            and not waiting_confirmation
        ):

            cv2.putText(
                frame,
                "Giu dung ky hieu - chu yeu chinh dien",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Duoc di chuyen tay tu nhien, gan/xa nhe",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "SPACE: Bat dau",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )


        # =================================================
        # 17. DANG THU
        # =================================================

        if collecting:

            now = time.perf_counter()

            elapsed = (
                now - capture_start_time
            )

            remaining = max(
                0,
                CAPTURE_DURATION - elapsed
            )


            cv2.putText(
                frame,
                "DANG THU...",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                f"Time: {remaining:.1f}s",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Samples: {sample_count}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            # ---------------------------------------------
            # CO BAN TAY
            # ---------------------------------------------

            if hand is not None:

                if (
                    now - last_capture_time
                    >= CAPTURE_INTERVAL
                ):

                    # =========================
                    # RAW LANDMARK
                    # =========================

                    raw_values = []

                    for landmark in hand:

                        raw_values.extend([
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ])


                    # =========================
                    # CHUAN HOA
                    # =========================

                    features = (
                        normalize_landmarks(
                            hand
                        )
                    )


                    # =========================
                    # TAM LUU
                    # =========================

                    current_raw_rows.append(
                        [
                            person_id,
                            current_letter,
                            handedness
                        ]
                        + raw_values
                    )


                    current_dataset_rows.append(
                        [
                            person_id,
                            current_letter,
                            handedness
                        ]
                        + features
                    )


                    sample_count += 1

                    last_capture_time = now


            else:

                cv2.putText(
                    frame,
                    "Khong tim thay ban tay",
                    (20, 215),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )


            # ---------------------------------------------
            # HET THOI GIAN
            # ---------------------------------------------

            if elapsed >= CAPTURE_DURATION:

                collecting = False

                waiting_confirmation = True


                print(
                    "\n",
                    current_letter,
                    "THU XONG."
                )

                print(
                    "So mau:",
                    sample_count
                )

                print(
                    "ENTER = luu"
                )

                print(
                    "R = thu lai"
                )


        # =================================================
        # 18. CHO XAC NHAN
        # =================================================

        if waiting_confirmation:

            cv2.putText(
                frame,
                f"{current_letter} HOAN THANH",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                3
            )


            cv2.putText(
                frame,
                f"Samples: {sample_count}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                "ENTER: Luu va sang ky tu tiep",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                "R: Thu lai",
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2
            )


        # =================================================
        # 19. HIEN CAMERA
        # =================================================

        cv2.imshow(
            "Collect Dataset",
            frame
        )


        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        # =================================================
        # 20. Q = THOAT
        # =================================================

        if key == ord("q"):

            print("Da thoat.")

            break


        # =================================================
        # 21. SPACE = BAT DAU
        # =================================================

        if (
            key == 32
            and not collecting
            and not waiting_confirmation
        ):

            if hand is None:

                print(
                    "Chua thay ban tay. "
                    "Hay dua tay vao camera."
                )

            else:

                collecting = True

                capture_start_time = (
                    time.perf_counter()
                )

                last_capture_time = 0

                sample_count = 0

                current_raw_rows = []

                current_dataset_rows = []


                print(
                    "\nBat dau thu:",
                    current_letter
                )


        # =================================================
        # 22. XAC NHAN
        # =================================================

        if waiting_confirmation:

            # ENTER
            if key == 13 or key == 10:

                # -------------------------
                # LUU RAW
                # -------------------------

                with open(
                    RAW_FILE,
                    "a",
                    newline="",
                    encoding="utf-8"
                ) as file:

                    writer = csv.writer(file)

                    writer.writerows(
                        current_raw_rows
                    )


                # -------------------------
                # LUU DATASET
                # -------------------------

                with open(
                    DATASET_FILE,
                    "a",
                    newline="",
                    encoding="utf-8"
                ) as file:

                    writer = csv.writer(file)

                    writer.writerows(
                        current_dataset_rows
                    )


                print(
                    current_letter,
                    "DA LUU:",
                    sample_count,
                    "mau."
                )


                current_letter_index += 1

                collecting = False
                waiting_confirmation = False

                sample_count = 0

                current_raw_rows = []
                current_dataset_rows = []


            # ---------------------------------------------
            # R = THU LAI
            # ---------------------------------------------

            elif key == ord("r"):

                print(
                    "Thu lai ky tu:",
                    current_letter
                )


                collecting = False
                waiting_confirmation = False

                sample_count = 0

                current_raw_rows = []
                current_dataset_rows = []

                last_capture_time = 0


# =====================================================
# 23. DONG CAMERA
# =====================================================

camera.release()

cv2.destroyAllWindows()


print("\n============================")
print("Ket thuc thu du lieu.")
print("Raw data:")
print(RAW_FILE)
print("Dataset:")
print(DATASET_FILE)
print("============================")
