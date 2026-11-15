import sys
import hashlib
import math
import random
from pathlib import Path

import cv2
import pandas as pd
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 1. DUONG DAN PROJECT
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
AI_DIR = PROJECT_DIR / "ai"

SOURCE_DIR = PROJECT_DIR / "dulieu" / "chucai"

MODEL_PATH = PROJECT_DIR / "models" / "mediapipe" / "hand_landmarker.task"

OUTPUT_DATASET = AI_DIR / "dataset" / "dataset_chucai_all.csv"

OUTPUT_RAW = (
    AI_DIR
    / "raw_data"
    / "raw_landmarks_chucai_all.csv"
)

OUTPUT_REJECTED = (
    AI_DIR
    / "dataset"
    / "rejected_chucai.csv"
)


# ============================================================
# 2. SU DUNG HAM NORMALIZE CUA PROJECT
# ============================================================

sys.path.insert(0, str(AI_DIR / "preprocessing"))

from landmark_utils import normalize_landmarks


# ============================================================
# 3. CAU HINH THU NGHIEM
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# Chi xu ly 10 anh moi chu de TEST truoc.
#
# Khi chuong trinh chay dung,
# ta moi doi thanh None de xu ly toan bo.
MAX_PER_CLASS = None


# 24 chu cai tinh.
# J va Z se xu ly rieng bang chuyen dong.
VALID_LABELS = set(
    "ABCDEFGHIKLMNOPQRSTUVWXY"
)


# ============================================================
# 4. KHOI TAO MEDIAPIPE
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)


# ============================================================
# 5. TEN CAC COT DU LIEU
# ============================================================

feature_columns = [
    f"f{i}"
    for i in range(1, 64)
]

raw_columns = []

for i in range(21):
    raw_columns.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])


# ============================================================
# 6. NOI LUU KET QUA TAM
# ============================================================

dataset_rows = []
raw_rows = []
rejected_rows = []

# Dung de phat hien file anh trung hoan toan.
seen_hashes = set()


# ============================================================
# 7. KIEM TRA DUONG DAN
# ============================================================

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Khong tim thay thu muc anh:\n{SOURCE_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay MediaPipe model:\n{MODEL_PATH}"
    )


# ============================================================
# 8. HAM TINH SHA256
#    DUNG DE PHAT HIEN ANH TRUNG HOAN TOAN
# ============================================================

def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 9. CAC BIEN THONG KE
# ============================================================

total_images = 0
total_success = 0
total_rejected = 0


# ============================================================
# 10. BAT DAU DOC DATASET
# ============================================================

with vision.HandLandmarker.create_from_options(
    options
) as landmarker:

    class_folders = sorted(
        [
            path
            for path in SOURCE_DIR.iterdir()
            if path.is_dir()
        ],
        key=lambda p: p.name.upper()
    )

    for class_dir in class_folders:

        label = class_dir.name.upper()

        # Bo qua folder khong phai ky tu can train
        if label not in VALID_LABELS:
            print(
                f"Bo qua folder khong hop le: "
                f"{class_dir.name}"
            )
            continue

        # Tim tat ca anh ben trong folder,
        # ke ca folder con.
        image_files = sorted([
            path
            for path in class_dir.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ])

        # Chi lay 10 anh de test lan dau.
        if (
            MAX_PER_CLASS is not None
            and len(image_files) > MAX_PER_CLASS
        ):
            image_files = random.sample(
                image_files,
                MAX_PER_CLASS
            )

        class_success = 0
        class_rejected = 0

        print()
        print(
            f"===== {label}: "
            f"{len(image_files)} anh ====="
        )

        for image_path in image_files:

            total_images += 1

            relative_path = image_path.relative_to(
                PROJECT_DIR
            )

            # =================================================
            # FILTER 1:
            # ANH TRUNG HOAN TOAN
            # =================================================

            try:
                image_hash = calculate_sha256(
                    image_path
                )

            except Exception:
                rejected_rows.append({
                    "file": str(relative_path),
                    "label": label,
                    "reason": "HASH_ERROR"
                })

                class_rejected += 1
                total_rejected += 1

                continue

            if image_hash in seen_hashes:

                rejected_rows.append({
                    "file": str(relative_path),
                    "label": label,
                    "reason": "EXACT_DUPLICATE"
                })

                class_rejected += 1
                total_rejected += 1

                continue

            seen_hashes.add(image_hash)


            # =================================================
            # FILTER 2:
            # ANH KHONG DOC DUOC
            # =================================================

            frame = cv2.imread(
                str(image_path)
            )

            if frame is None:

                rejected_rows.append({
                    "file": str(relative_path),
                    "label": label,
                    "reason": "IMAGE_READ_ERROR"
                })

                class_rejected += 1
                total_rejected += 1

                continue


            # =================================================
            # CHUYEN BGR → RGB
            # =================================================

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )


            # =================================================
            # MEDIAPIPE PHAT HIEN TAY
            # =================================================

            result = landmarker.detect(
                mp_image
            )


            # =================================================
            # FILTER 3:
            # KHONG CO BAN TAY
            # =================================================

            if not result.hand_landmarks:

                rejected_rows.append({
                    "file": str(relative_path),
                    "label": label,
                    "reason": "NO_HAND"
                })

                class_rejected += 1
                total_rejected += 1

                continue


            hand = result.hand_landmarks[0]


            # =================================================
            # FILTER 4:
            # KHONG DU 21 LANDMARK
            # =================================================

            if len(hand) != 21:

                rejected_rows.append({
                    "file": str(relative_path),
                    "label": label,
                    "reason":
                    f"INVALID_LANDMARK_COUNT_{len(hand)}"
                })

                class_rejected += 1
                total_rejected += 1

                continue


            # =================================================
            # HANDEDNESS
            # =================================================

            handedness = "Unknown"

            if (
                result.handedness
                and result.handedness[0]
            ):
                handedness = (
                    result
                    .handedness[0][0]
                    .category_name
                )


            # =================================================
            # LUU 21 LANDMARK GOC
            # =================================================

            raw_features = []

            for landmark in hand:

                raw_features.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])


            # =================================================
            # CHUAN HOA BANG HAM CU CUA PROJECT
            # =================================================

            features = normalize_landmarks(
                hand
            )


            # =================================================
            # FILTER 5:
            # PHAI CO DUNG 63 DAC TRUNG
            # =================================================

            if len(features) != 63:

                rejected_rows.append({
                    "file": str(relative_path),
                    "label": label,
                    "reason":
                    "INVALID_FEATURE_COUNT"
                })

                class_rejected += 1
                total_rejected += 1

                continue


            # =================================================
            # FILTER 6:
            # KHONG DUOC CO NAN / INF
            # =================================================

            if not all(
                math.isfinite(value)
                for value in features
            ):

                rejected_rows.append({
                    "file": str(relative_path),
                    "label": label,
                    "reason":
                    "INVALID_NUMERIC_VALUE"
                })

                class_rejected += 1
                total_rejected += 1

                continue


            # =================================================
            # DU LIEU DAT YEU CAU
            # =================================================

            raw_rows.append(
                [
                    "internet",
                    str(relative_path),
                    label,
                    handedness
                ]
                + raw_features
            )

            dataset_rows.append(
                [
                    "internet",
                    str(relative_path),
                    label,
                    handedness
                ]
                + features
            )

            class_success += 1
            total_success += 1


        print(
            f"{label}: "
            f"giu = {class_success}, "
            f"loai = {class_rejected}"
        )


# ============================================================
# 11. TAO DATAFRAME NORMALIZED
# ============================================================

dataset_df = pd.DataFrame(
    dataset_rows,
    columns=[
        "source",
        "file",
        "label",
        "handedness"
    ]
    + feature_columns
)


# ============================================================
# 12. TAO DATAFRAME RAW
# ============================================================

raw_df = pd.DataFrame(
    raw_rows,
    columns=[
        "source",
        "file",
        "label",
        "handedness"
    ]
    + raw_columns
)


# ============================================================
# 13. DANH SACH BI LOAI
# ============================================================

rejected_df = pd.DataFrame(
    rejected_rows,
    columns=[
        "file",
        "label",
        "reason"
    ]
)


# ============================================================
# 14. TAO THU MUC NEU CHUA CO
# ============================================================

OUTPUT_DATASET.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_RAW.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 15. LUU FILE
# ============================================================

dataset_df.to_csv(
    OUTPUT_DATASET,
    index=False
)

raw_df.to_csv(
    OUTPUT_RAW,
    index=False
)

rejected_df.to_csv(
    OUTPUT_REJECTED,
    index=False
)


# ============================================================
# 16. THONG KE CUOI
# ============================================================

print()
print("===================================")
print("HOAN THANH KIEM TRA THU")
print("===================================")

print(
    f"Tong anh da kiem tra : {total_images}"
)

print(
    f"Anh dat yeu cau      : {total_success}"
)

print(
    f"Anh bi loai          : {total_rejected}"
)

print()

if not dataset_df.empty:

    print("SO MAU DAT YEU CAU MOI KY TU:")

    print(
        dataset_df
        .groupby("label")
        .size()
    )

print()
print("Dataset normalized:")
print(OUTPUT_DATASET)

print()
print("Raw landmarks:")
print(OUTPUT_RAW)

print()
print("Danh sach bi loai:")
print(OUTPUT_REJECTED)
