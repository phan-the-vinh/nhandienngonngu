import sys
import math
from pathlib import Path

import cv2
import pandas as pd
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 1. DUONG DAN PROJECT
# ============================================================

AI_DIR = Path(__file__).resolve().parent
PROJECT_DIR = AI_DIR.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)

INPUT_REJECTED = (
    AI_DIR
    / "dataset"
    / "kytuso"
    / "landmark"
    / "rejected_chuso.csv"
)

OUTPUT_DIR = (
    AI_DIR
    / "dataset"
    / "kytuso"
    / "landmark"
    / "rescue"
)

OUTPUT_DATASET = (
    OUTPUT_DIR
    / "rescue_chuso_normalized.csv"
)

OUTPUT_RAW = (
    OUTPUT_DIR
    / "rescue_chuso_raw.csv"
)

OUTPUT_STILL_REJECTED = (
    OUTPUT_DIR
    / "rescue_still_rejected.csv"
)


# ============================================================
# 2. HAM NORMALIZE CUA PROJECT
# ============================================================

sys.path.insert(0, str(AI_DIR / "preprocessing"))

from landmark_utils import normalize_landmarks


# ============================================================
# 3. CAU HINH RESCUE
# ============================================================

RESCUE_DETECTION_CONFIDENCE = 0.30
RESCUE_PRESENCE_CONFIDENCE = 0.30


# ============================================================
# 4. KIEM TRA FILE
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay MediaPipe model:\n{MODEL_PATH}"
    )

if not INPUT_REJECTED.exists():
    raise FileNotFoundError(
        f"Khong tim thay rejected CSV:\n{INPUT_REJECTED}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 5. DOC DANH SACH NO_HAND
# ============================================================

rejected_df = pd.read_csv(
    INPUT_REJECTED,
    dtype={"label": str}
)

no_hand_df = rejected_df[
    rejected_df["reason"] == "NO_HAND"
].copy()

print(
    f"So anh NO_HAND can rescue: "
    f"{len(no_hand_df)}"
)


# ============================================================
# 6. COT DU LIEU
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
    "handedness",
    "rescue_detection_confidence",
    "rescue_presence_confidence"
]


# ============================================================
# 7. HAM DOC METADATA TU TEN FILE
# ============================================================

def extract_filename_metadata(image_path):

    parts = image_path.stem.split("_")

    hand_id = (
        parts[0]
        if len(parts) >= 1
        else "Unknown"
    )

    view = (
        parts[2]
        if len(parts) >= 3
        else "Unknown"
    )

    return hand_id, view


# ============================================================
# 8. KHOI TAO MEDIAPIPE RESCUE
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=(
        RESCUE_DETECTION_CONFIDENCE
    ),
    min_hand_presence_confidence=(
        RESCUE_PRESENCE_CONFIDENCE
    )
)


# ============================================================
# 9. NOI LUU KET QUA
# ============================================================

dataset_rows = []
raw_rows = []
still_rejected_rows = []

total_success = 0
total_failed = 0


# ============================================================
# 10. CHAY RESCUE
# ============================================================

with vision.HandLandmarker.create_from_options(
    options
) as landmarker:

    for _, row in no_hand_df.iterrows():

        relative_path = Path(
            str(row["file"])
        )

        image_path = (
            PROJECT_DIR
            / relative_path
        )

        label = str(row["label"])

        if not image_path.exists():

            still_rejected_rows.append({
                "file": str(relative_path),
                "label": label,
                "reason": "FILE_NOT_FOUND"
            })

            total_failed += 1
            continue


        # ----------------------------------------------------
        # DOC ANH
        # ----------------------------------------------------

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:

            still_rejected_rows.append({
                "file": str(relative_path),
                "label": label,
                "reason": "IMAGE_READ_ERROR"
            })

            total_failed += 1
            continue


        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ----------------------------------------------------
        # MEDIAPIPE
        # ----------------------------------------------------

        result = landmarker.detect(
            mp_image
        )

        if not result.hand_landmarks:

            still_rejected_rows.append({
                "file": str(relative_path),
                "label": label,
                "reason": "NO_HAND_AT_0.30"
            })

            total_failed += 1
            continue


        hand = result.hand_landmarks[0]


        # ----------------------------------------------------
        # KIEM TRA 21 LANDMARK
        # ----------------------------------------------------

        if len(hand) != 21:

            still_rejected_rows.append({
                "file": str(relative_path),
                "label": label,
                "reason":
                f"INVALID_LANDMARK_COUNT_{len(hand)}"
            })

            total_failed += 1
            continue


        # ----------------------------------------------------
        # HANDEDNESS
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # RAW LANDMARK
        # ----------------------------------------------------

        raw_features = []

        for landmark in hand:

            raw_features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])


        # ----------------------------------------------------
        # NORMALIZE GIONG PIPELINE CHINH
        # ----------------------------------------------------

        features = normalize_landmarks(
            hand
        )


        # ----------------------------------------------------
        # KIEM TRA 63 FEATURE
        # ----------------------------------------------------

        if len(features) != 63:

            still_rejected_rows.append({
                "file": str(relative_path),
                "label": label,
                "reason": "INVALID_FEATURE_COUNT"
            })

            total_failed += 1
            continue


        # ----------------------------------------------------
        # KIEM TRA NAN / INF
        # ----------------------------------------------------

        if not all(
            math.isfinite(value)
            for value in features
        ):

            still_rejected_rows.append({
                "file": str(relative_path),
                "label": label,
                "reason": "INVALID_NUMERIC_VALUE"
            })

            total_failed += 1
            continue


        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        hand_id, view = extract_filename_metadata(
            image_path
        )

        metadata = [
            "dulieu_chuso_rescue",
            str(relative_path),
            label,
            hand_id,
            view,
            handedness,
            RESCUE_DETECTION_CONFIDENCE,
            RESCUE_PRESENCE_CONFIDENCE
        ]


        # ----------------------------------------------------
        # LUU KET QUA
        # ----------------------------------------------------

        raw_rows.append(
            metadata
            + raw_features
        )

        dataset_rows.append(
            metadata
            + features
        )

        total_success += 1


# ============================================================
# 11. TAO DATAFRAME
# ============================================================

dataset_df = pd.DataFrame(
    dataset_rows,
    columns=
    metadata_columns
    + feature_columns
)

raw_df = pd.DataFrame(
    raw_rows,
    columns=
    metadata_columns
    + raw_columns
)

still_rejected_df = pd.DataFrame(
    still_rejected_rows,
    columns=[
        "file",
        "label",
        "reason"
    ]
)


# ============================================================
# 12. LUU FILE
# ============================================================

dataset_df.to_csv(
    OUTPUT_DATASET,
    index=False
)

raw_df.to_csv(
    OUTPUT_RAW,
    index=False
)

still_rejected_df.to_csv(
    OUTPUT_STILL_REJECTED,
    index=False
)


# ============================================================
# 13. THONG KE
# ============================================================

print()
print("====================================")
print("HOAN THANH RESCUE NO_HAND")
print("====================================")

print(
    f"Tong NO_HAND ban dau : "
    f"{len(no_hand_df)}"
)

print(
    f"Rescue thanh cong    : "
    f"{total_success}"
)

print(
    f"Van that bai         : "
    f"{total_failed}"
)

if not dataset_df.empty:

    print()
    print("RESCUE THANH CONG THEO LABEL:")

    print(
        dataset_df
        .groupby("label")
        .size()
    )

print()
print("Rescue normalized:")
print(OUTPUT_DATASET)

print()
print("Rescue raw:")
print(OUTPUT_RAW)

print()
print("Still rejected:")
print(OUTPUT_STILL_REJECTED)
