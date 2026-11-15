import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. DUONG DAN
# ============================================================

AI_DIR = Path(__file__).resolve().parents[2] / "ai"

DATASET_DIR = (
    AI_DIR
    / "dataset"
    / "kytuso"
    / "datasetsach"
)

DATASET_PATH = (
    DATASET_DIR
    / "dataset_chuso_clean_v1.csv"
)

MERGE_SUMMARY_PATH = (
    DATASET_DIR
    / "merge_summary_v1.json"
)

OUTPUT_MANIFEST = (
    DATASET_DIR
    / "final_freeze_manifest_v1.json"
)


# ============================================================
# 2. CAU HINH MONG DOI
# ============================================================

EXPECTED_TOTAL = 699
EXPECTED_FEATURE_COUNT = 63

EXPECTED_LABEL_COUNTS = {
    "0": 69,
    "1": 70,
    "2": 70,
    "3": 70,
    "4": 70,
    "5": 70,
    "6": 70,
    "7": 70,
    "8": 70,
    "9": 70,
}

EXPECTED_HAND_COUNTS = {
    "hand1": 249,
    "hand2": 250,
    "hand3": 50,
    "hand4": 50,
    "hand5": 100,
}

EXPECTED_VIEW_COUNTS = {
    "bot": 200,
    "dif": 200,
    "left": 100,
    "right": 99,
    "top": 100,
}

EXPECTED_STAGE_COUNTS = {
    "main": 592,
    "rescue_030": 32,
    "rescue_padding": 75,
}

EXCLUDED_SAMPLE = {
    "file":
        r"dulieu\chuso\0\hand1_0_right_seg_2_cropped.jpeg",
    "label": "0",
    "reason": "NO_HAND_AFTER_PADDING",
}


# ============================================================
# 3. SHA256
# ============================================================

def calculate_sha256(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 4. KIEM TRA FILE
# ============================================================

if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"Khong tim thay dataset:\n"
        f"{DATASET_PATH}"
    )

if not MERGE_SUMMARY_PATH.exists():

    raise FileNotFoundError(
        f"Khong tim thay merge summary:\n"
        f"{MERGE_SUMMARY_PATH}"
    )


# ============================================================
# 5. DOC DATASET
# ============================================================

df = pd.read_csv(
    DATASET_PATH,
    dtype={
        "label": str
    }
)


# ============================================================
# 6. KIEM TRA FEATURE
# ============================================================

feature_columns = [
    col
    for col in df.columns
    if (
        col.startswith("f")
        and col[1:].isdigit()
    )
]

feature_columns = sorted(
    feature_columns,
    key=lambda col: int(col[1:])
)

if len(feature_columns) != EXPECTED_FEATURE_COUNT:

    raise ValueError(
        f"Feature count = "
        f"{len(feature_columns)}, "
        f"mong doi = "
        f"{EXPECTED_FEATURE_COUNT}"
    )


# ============================================================
# 7. KIEM TRA SO MAU
# ============================================================

if len(df) != EXPECTED_TOTAL:

    raise ValueError(
        f"Sample count = "
        f"{len(df)}, "
        f"mong doi = "
        f"{EXPECTED_TOTAL}"
    )


# ============================================================
# 8. KIEM TRA DUPLICATE FILE
# ============================================================

duplicate_count = int(
    df.duplicated(
        subset=["file"]
    ).sum()
)

if duplicate_count != 0:

    raise ValueError(
        f"Phat hien "
        f"{duplicate_count} file trung."
    )


# ============================================================
# 9. KIEM TRA NAN / INF
# ============================================================

feature_matrix = (
    df[feature_columns]
    .apply(
        pd.to_numeric,
        errors="raise"
    )
    .to_numpy(
        dtype=float
    )
)

if not np.isfinite(
    feature_matrix
).all():

    raise ValueError(
        "Dataset co NaN hoac Inf."
    )


# ============================================================
# 10. THONG KE
# ============================================================

label_counts = (
    df
    .groupby("label")
    .size()
    .to_dict()
)

hand_counts = (
    df
    .groupby("hand_id")
    .size()
    .to_dict()
)

view_counts = (
    df
    .groupby("view")
    .size()
    .to_dict()
)

stage_counts = (
    df
    .groupby("extraction_stage")
    .size()
    .to_dict()
)


# ============================================================
# 11. KIEM TRA PHAN BO
# ============================================================

if label_counts != EXPECTED_LABEL_COUNTS:

    raise ValueError(
        "Label distribution khong dung.\n"
        f"{label_counts}"
    )

if hand_counts != EXPECTED_HAND_COUNTS:

    raise ValueError(
        "Hand distribution khong dung.\n"
        f"{hand_counts}"
    )

if view_counts != EXPECTED_VIEW_COUNTS:

    raise ValueError(
        "View distribution khong dung.\n"
        f"{view_counts}"
    )

if stage_counts != EXPECTED_STAGE_COUNTS:

    raise ValueError(
        "Extraction stage distribution "
        "khong dung.\n"
        f"{stage_counts}"
    )


# ============================================================
# 12. SHA256 DATASET
# ============================================================

dataset_sha256 = calculate_sha256(
    DATASET_PATH
)

summary_sha256 = calculate_sha256(
    MERGE_SUMMARY_PATH
)


# ============================================================
# 13. TAO FREEZE MANIFEST
# ============================================================

manifest = {
    "freeze_version": "v1",

    "freeze_created_at": (
        datetime.now(
            timezone.utc
        ).isoformat()
    ),

    "dataset": {
        "path": str(
            DATASET_PATH
        ),
        "sha256": dataset_sha256,
        "sample_count": int(
            len(df)
        ),
        "feature_count": int(
            len(feature_columns)
        ),
        "duplicate_file_count":
            duplicate_count,
        "nan_inf_check": "PASS",
    },

    "label_counts": {
        str(k): int(v)
        for k, v
        in label_counts.items()
    },

    "hand_counts": {
        str(k): int(v)
        for k, v
        in hand_counts.items()
    },

    "view_counts": {
        str(k): int(v)
        for k, v
        in view_counts.items()
    },

    "extraction_stage_counts": {
        str(k): int(v)
        for k, v
        in stage_counts.items()
    },

    "merge_summary": {
        "path": str(
            MERGE_SUMMARY_PATH
        ),
        "sha256": summary_sha256,
    },

    "excluded_samples": [
        EXCLUDED_SAMPLE
    ],

    "validation": {
        "sample_count": "PASS",
        "feature_count": "PASS",
        "label_distribution": "PASS",
        "hand_distribution": "PASS",
        "view_distribution": "PASS",
        "extraction_stage_distribution":
            "PASS",
        "duplicate_check": "PASS",
        "nan_inf_check": "PASS",
    },

    "status":
        "FROZEN_CHUSO_DATASET_V1"
}


# ============================================================
# 14. LUU MANIFEST
# ============================================================

with open(
    OUTPUT_MANIFEST,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        manifest,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 15. KET QUA
# ============================================================

print()
print("====================================")
print("FREEZE DATASET CHUSO V1 COMPLETE")
print("====================================")

print(
    f"Samples       : "
    f"{len(df)}"
)

print(
    f"Features      : "
    f"{len(feature_columns)}"
)

print(
    f"Duplicate     : "
    f"{duplicate_count}"
)

print(
    "NaN / Inf     : PASS"
)

print()

print(
    "SHA256:"
)

print(
    dataset_sha256
)

print()

print(
    "Hand distribution:"
)

for key, value in hand_counts.items():

    print(
        f"{key}: {value}"
    )

print()

print(
    "View distribution:"
)

for key, value in view_counts.items():

    print(
        f"{key}: {value}"
    )

print()

print(
    "Freeze manifest:"
)

print(
    OUTPUT_MANIFEST
)

print()

print(
    "STATUS: FROZEN_CHUSO_DATASET_V1"
)
