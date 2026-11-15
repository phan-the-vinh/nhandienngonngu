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
    / "rescue"
    / "rescue_still_rejected.csv"
)

OUTPUT_DIR = (
    AI_DIR
    / "dataset"
    / "kytuso"
    / "landmark"
    / "rescue_padding"
)

OUTPUT_DATASET = (
    OUTPUT_DIR
    / "rescue_padding_normalized.csv"
)

OUTPUT_RAW = (
    OUTPUT_DIR
    / "rescue_padding_raw.csv"
)

OUTPUT_STILL_REJECTED = (
    OUTPUT_DIR
    / "rescue_padding_still_rejected.csv"
)


# ============================================================
# 2. HAM NORMALIZE CUA PROJECT
# ============================================================

sys.path.insert(0, str(AI_DIR / "preprocessing"))

from landmark_utils import normalize_landmarks


# ============================================================
# 3. CAU HINH RESCUE
# ============================================================

DETECTION_CONFIDENCE = 0.30
PRESENCE_CONFIDENCE = 0.30

# Thu lan luot.
PADDING_RATIOS = [
    0.10,
    0.20,
    0.30
]


# ============================================================
# 4. KIEM TRA DUONG DAN
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay MediaPipe model:\n{MODEL_PATH}"
    )

if not INPUT_REJECTED.exists():
    raise FileNotFoundError(
        f"Khong tim thay file rejected:\n{INPUT_REJECTED}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 5. DOC 76 ANH CON THAT BAI
# ============================================================

rejected_df = pd.read_csv(
    INPUT_REJECTED,
    dtype={"label": str}
)

print(
    f"So anh can rescue bang padding: "
    f"{len(rejected_df)}"
)


# ============================================================
# 6. TEN COT DU LIEU
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
    "padding_ratio",
    "detection_confidence",
    "presence_confidence"
]


# ============================================================
# 7. DOC METADATA TU TEN FILE
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
# 8. THEM PADDING
# ============================================================

def add_padding(frame, ratio):

    height, width = frame.shape[:2]

    pad_y = int(height * ratio)
    pad_x = int(width * ratio)

    padded = cv2.copyMakeBorder(
        frame,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        borderType=cv2.BORDER_REPLICATE
    )

    return padded


# ============================================================
# 9. MEDIAPIPE
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=DETECTION_CONFIDENCE,
    min_hand_presence_confidence=PRESENCE_CONFIDENCE
)


# ============================================================
# 10. NOI LUU KET QUA
# ============================================================

dataset_rows = []
raw_rows = []
still_rejected_rows = []

total_success = 0
total_failed = 0


# ============================================================
# 11. CHAY RESCUE PADDING
# ============================================================

with vision.HandLandmarker.create_from_options(
    options
) as landmarker:

    for _, row in rejected_df.iterrows():

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


        rescue_result = None
        rescue_hand = None
        used_padding = None


        # ----------------------------------------------------
        # THU 10% -> 20% -> 30%
        # ----------------------------------------------------

        for padding_ratio in PADDING_RATIOS:

            padded_frame = add_padding(
                frame,
                padding_ratio
            )

            rgb = cv2.cvtColor(
                padded_frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect(
                mp_image
            )

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                if len(hand) == 21:

                    rescue_result = result
                    rescue_hand = hand
                    used_padding = padding_ratio

                    break


        # ----------------------------------------------------
        # VAN KHONG NHAN DUOC TAY
        # ----------------------------------------------------

        if rescue_hand is None:

            still_rejected_rows.append({
                "file": str(relative_path),
                "label": label,
                "reason": "NO_HAND_AFTER_PADDING"
            })

            total_failed += 1
            continue


        # ----------------------------------------------------
        # HANDEDNESS
        # ----------------------------------------------------

        handedness = "Unknown"

        if (
            rescue_result.handedness
            and rescue_result.handedness[0]
        ):
            handedness = (
                rescue_result
                .handedness[0][0]
                .category_name
            )


        # ----------------------------------------------------
        # RAW LANDMARKS
        # ----------------------------------------------------

        raw_features = []

        for landmark in rescue_hand:

            raw_features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])


        # ----------------------------------------------------
        # NORMALIZE GIONG PIPELINE CHINH
        # ----------------------------------------------------

        features = normalize_landmarks(
            rescue_hand
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
            "dulieu_chuso_rescue_padding",
            str(relative_path),
            label,
            hand_id,
            view,
            handedness,
            used_padding,
            DETECTION_CONFIDENCE,
            PRESENCE_CONFIDENCE
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
# 12. DATAFRAME
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
# 13. LUU FILE
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
# 14. THONG KE
# ============================================================

print()
print("====================================")
print("HOAN THANH RESCUE BANG PADDING")
print("====================================")

print(
    f"Tong anh dau vao     : "
    f"{len(rejected_df)}"
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
    print("RESCUE THEO LABEL:")

    print(
        dataset_df
        .groupby("label")
        .size()
    )

    print()
    print("RESCUE THEO PADDING:")

    print(
        dataset_df
        .groupby("padding_ratio")
        .size()
    )


print()
print("Normalized:")
print(OUTPUT_DATASET)

print()
print("Raw:")
print(OUTPUT_RAW)

print()
print("Still rejected:")
print(OUTPUT_STILL_REJECTED)
