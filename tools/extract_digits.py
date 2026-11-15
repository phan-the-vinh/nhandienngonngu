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

SOURCE_DIR = PROJECT_DIR / "dulieu" / "chuso"

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)

OUTPUT_DIR = (
    AI_DIR
    / "dataset"
    / "kytuso"
    / "landmark"
)

OUTPUT_DATASET = (
    OUTPUT_DIR
    / "dataset_chuso_all.csv"
)

OUTPUT_RAW = (
    OUTPUT_DIR
    / "raw_landmarks_chuso_all.csv"
)

OUTPUT_REJECTED = (
    OUTPUT_DIR
    / "rejected_chuso.csv"
)


# ============================================================
# 2. SU DUNG HAM NORMALIZE CUA PROJECT
# ============================================================

sys.path.insert(0, str(AI_DIR / "preprocessing"))

from landmark_utils import normalize_landmarks


# ============================================================
# 3. CAU HINH THU NGHIEM
# ============================================================

# Dataset hien tai co 700 anh JPEG dai dien cho
# 700 basename duy nhat.
#
# 300 file PNG khong duoc dung trong pipeline v1
# de tranh lap lai cung mot mau.
IMAGE_EXTENSIONS = {
    ".jpeg"
}

# Smoke test:
# chi lay toi da 5 anh moi so.
#
# 10 so x 5 anh = toi da 50 anh.
#
# Khi test thanh cong:
# doi MAX_PER_CLASS = None
# de xu ly toan bo 700 anh JPEG.
MAX_PER_CLASS = None

# Giup lan smoke test sau chon lai cung mot tap anh.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 10 ky tu so tinh tu 0 den 9.
VALID_LABELS = set(
    "0123456789"
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


metadata_columns = [
    "source",
    "file",
    "label",
    "hand_id",
    "view",
    "handedness"
]


# ============================================================
# 6. HAM TINH SHA256
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
# 7. DOC METADATA TU TEN FILE
# ============================================================

def extract_filename_metadata(image_path):
    """
    Vi du:

    hand1_2_bot_seg_1_cropped.jpeg

    hand_id = hand1
    label   = 2
    view    = bot
    """

    parts = image_path.stem.split("_")

    if len(parts) >= 1:
        hand_id = parts[0]
    else:
        hand_id = "Unknown"

    if len(parts) >= 3:
        view = parts[2]
    else:
        view = "Unknown"

    return hand_id, view


# ============================================================
# 8. KIEM TRA DUONG DAN
# ============================================================

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Khong tim thay thu muc anh:\n{SOURCE_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay MediaPipe model:\n{MODEL_PATH}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 9. NOI LUU KET QUA TAM
# ============================================================

dataset_rows = []
raw_rows = []
rejected_rows = []

# Dung de phat hien file anh trung hoan toan.
seen_hashes = set()


# ============================================================
# 10. CAC BIEN THONG KE
# ============================================================

total_images = 0
total_success = 0
total_rejected = 0


# ============================================================
# 11. BAT DAU DOC DATASET
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
        key=lambda p: p.name
    )

    for class_dir in class_folders:

        label = class_dir.name

        # Chi xu ly folder 0-9.
        if label not in VALID_LABELS:
            print(
                f"Bo qua folder khong hop le: "
                f"{class_dir.name}"
            )
            continue

        # Chi tim file JPEG.
        image_files = sorted([
            path
            for path in class_dir.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ])

        # Smoke test:
        # chon toi da 5 anh moi class.
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

            hand_id, view = extract_filename_metadata(
                image_path
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
            # CHUYEN BGR -> RGB
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
            # CHUAN HOA BANG HAM CUA PROJECT
            # ============================================================

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

            metadata = [
                "dulieu_chuso",
                str(relative_path),
                label,
                hand_id,
                view,
                handedness
            ]

            raw_rows.append(
                metadata
                + raw_features
            )

            dataset_rows.append(
                metadata
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
# 12. TAO DATAFRAME NORMALIZED
# ============================================================

dataset_df = pd.DataFrame(
    dataset_rows,
    columns=
    metadata_columns
    + feature_columns
)


# ============================================================
# 13. TAO DATAFRAME RAW
# ============================================================

raw_df = pd.DataFrame(
    raw_rows,
    columns=
    metadata_columns
    + raw_columns
)


# ============================================================
# 14. DANH SACH BI LOAI
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
print("HOAN THANH TRICH LANDMARK KY TU SO")
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

    print(
        "SO MAU DAT YEU CAU MOI KY TU SO:"
    )

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
