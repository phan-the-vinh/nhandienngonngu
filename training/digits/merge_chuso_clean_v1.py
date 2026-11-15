import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. DUONG DAN
# ============================================================

AI_DIR = Path(__file__).resolve().parents[2] / "ai"

LANDMARK_DIR = (
    AI_DIR
    / "dataset"
    / "kytuso"
    / "landmark"
)

MAIN_CSV = (
    LANDMARK_DIR
    / "dataset_chuso_all.csv"
)

RESCUE_CSV = (
    LANDMARK_DIR
    / "rescue"
    / "rescue_chuso_normalized.csv"
)

PADDING_CSV = (
    LANDMARK_DIR
    / "rescue_padding"
    / "rescue_padding_normalized.csv"
)

OUTPUT_DIR = (
    AI_DIR
    / "dataset"
    / "kytuso"
    / "datasetsach"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "dataset_chuso_clean_v1.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "merge_summary_v1.json"
)


# ============================================================
# 2. FEATURE
# ============================================================

FEATURE_COLUMNS = [
    f"f{i}"
    for i in range(1, 64)
]

OUTPUT_COLUMNS = [
    "file",
    "label",
    "hand_id",
    "view",
    "handedness",
    "extraction_stage",
    "source_original",
    "detection_confidence",
    "presence_confidence",
    "padding_ratio",
] + FEATURE_COLUMNS


# ============================================================
# 3. KIEM TRA FILE DAU VAO
# ============================================================

for path in [
    MAIN_CSV,
    RESCUE_CSV,
    PADDING_CSV,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Khong tim thay:\n{path}"
        )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. DOC CSV
# ============================================================

main_df = pd.read_csv(
    MAIN_CSV,
    dtype={"label": str}
)

rescue_df = pd.read_csv(
    RESCUE_CSV,
    dtype={"label": str}
)

padding_df = pd.read_csv(
    PADDING_CSV,
    dtype={"label": str}
)


# ============================================================
# 5. HAM KIEM TRA COT
# ============================================================

def require_columns(df, columns, name):

    missing = [
        col
        for col in columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} thieu cot: {missing}"
        )


COMMON_COLUMNS = [
    "source",
    "file",
    "label",
    "hand_id",
    "view",
    "handedness",
] + FEATURE_COLUMNS

require_columns(
    main_df,
    COMMON_COLUMNS,
    "MAIN"
)

require_columns(
    rescue_df,
    COMMON_COLUMNS
    + [
        "rescue_detection_confidence",
        "rescue_presence_confidence",
    ],
    "RESCUE"
)

require_columns(
    padding_df,
    COMMON_COLUMNS
    + [
        "padding_ratio",
        "detection_confidence",
        "presence_confidence",
    ],
    "PADDING"
)


# ============================================================
# 6. CHUAN HOA MAIN
# ============================================================

main_clean = main_df.copy()

main_clean["extraction_stage"] = "main"
main_clean["source_original"] = main_clean["source"]

main_clean["detection_confidence"] = 0.50
main_clean["presence_confidence"] = 0.50
main_clean["padding_ratio"] = 0.0

main_clean = main_clean[
    OUTPUT_COLUMNS
]


# ============================================================
# 7. CHUAN HOA RESCUE 0.30
# ============================================================

rescue_clean = rescue_df.copy()

rescue_clean["extraction_stage"] = "rescue_030"
rescue_clean["source_original"] = rescue_clean["source"]

rescue_clean["detection_confidence"] = (
    rescue_clean[
        "rescue_detection_confidence"
    ]
)

rescue_clean["presence_confidence"] = (
    rescue_clean[
        "rescue_presence_confidence"
    ]
)

rescue_clean["padding_ratio"] = 0.0

rescue_clean = rescue_clean[
    OUTPUT_COLUMNS
]


# ============================================================
# 8. CHUAN HOA RESCUE PADDING
# ============================================================

padding_clean = padding_df.copy()

padding_clean["extraction_stage"] = (
    "rescue_padding"
)

padding_clean["source_original"] = (
    padding_clean["source"]
)

padding_clean = padding_clean[
    OUTPUT_COLUMNS
]


# ============================================================
# 9. GOP 3 NGUON
# ============================================================

merged_df = pd.concat(
    [
        main_clean,
        rescue_clean,
        padding_clean,
    ],
    ignore_index=True
)


# ============================================================
# 10. KIEM TRA TRUNG FILE
# ============================================================

duplicate_mask = merged_df.duplicated(
    subset=["file"],
    keep=False
)

if duplicate_mask.any():

    duplicate_files = (
        merged_df.loc[
            duplicate_mask,
            "file"
        ]
        .tolist()
    )

    raise ValueError(
        "Phat hien file bi trung sau merge:\n"
        + "\n".join(duplicate_files)
    )


# ============================================================
# 11. KIEM TRA LABEL
# ============================================================

valid_labels = set(
    "0123456789"
)

actual_labels = set(
    merged_df["label"].astype(str)
)

if not actual_labels.issubset(
    valid_labels
):
    raise ValueError(
        f"Co label khong hop le: "
        f"{actual_labels - valid_labels}"
    )


# ============================================================
# 12. KIEM TRA 63 FEATURE
# ============================================================

if len(FEATURE_COLUMNS) != 63:
    raise ValueError(
        "So feature khong bang 63."
    )

for col in FEATURE_COLUMNS:

    merged_df[col] = pd.to_numeric(
        merged_df[col],
        errors="raise"
    )


# ============================================================
# 13. KIEM TRA NAN / INF
# ============================================================

feature_matrix = merged_df[
    FEATURE_COLUMNS
].to_numpy(
    dtype=float
)

if not np.isfinite(
    feature_matrix
).all():

    raise ValueError(
        "Dataset co NaN hoac Inf."
    )


# ============================================================
# 14. KIEM TRA TONG SO MAU
# ============================================================

EXPECTED_TOTAL = 699

if len(merged_df) != EXPECTED_TOTAL:
    raise ValueError(
        f"So mau sau merge = "
        f"{len(merged_df)}, "
        f"mong doi = {EXPECTED_TOTAL}"
    )


# ============================================================
# 15. KIEM TRA PHAN BO LABEL
# ============================================================

expected_counts = {
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

actual_counts = (
    merged_df
    .groupby("label")
    .size()
    .to_dict()
)

if actual_counts != expected_counts:
    raise ValueError(
        "Phan bo label khong dung.\n"
        f"Thuc te: {actual_counts}\n"
        f"Mong doi: {expected_counts}"
    )


# ============================================================
# 16. SAP XEP
# ============================================================

merged_df["label_sort"] = (
    merged_df["label"]
    .astype(int)
)

merged_df = (
    merged_df
    .sort_values(
        by=[
            "label_sort",
            "hand_id",
            "view",
            "file",
        ]
    )
    .drop(
        columns=["label_sort"]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 17. THONG KE EXTRACTION STAGE
# ============================================================

stage_counts = (
    merged_df
    .groupby("extraction_stage")
    .size()
    .to_dict()
)


# ============================================================
# 18. LUU DATASET SACH
# ============================================================

merged_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 19. LUU SUMMARY
# ============================================================

summary = {
    "version": "v1",
    "total_samples": int(
        len(merged_df)
    ),
    "feature_count": 63,
    "duplicate_file_count": 0,
    "nan_inf_check": "PASS",
    "label_counts": {
        str(k): int(v)
        for k, v
        in actual_counts.items()
    },
    "extraction_stage_counts": {
        str(k): int(v)
        for k, v
        in stage_counts.items()
    },
    "input_files": {
        "main": str(MAIN_CSV),
        "rescue_030": str(
            RESCUE_CSV
        ),
        "rescue_padding": str(
            PADDING_CSV
        ),
    },
    "output_file": str(
        OUTPUT_CSV
    ),
}

with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 20. KET QUA
# ============================================================

print()
print("====================================")
print("HOAN THANH MERGE DATASET CHUSO V1")
print("====================================")

print(
    f"Tong mau      : "
    f"{len(merged_df)}"
)

print(
    f"So feature    : "
    f"{len(FEATURE_COLUMNS)}"
)

print()
print("THEO LABEL:")

print(
    merged_df
    .groupby("label")
    .size()
)

print()
print("THEO EXTRACTION STAGE:")

print(
    merged_df
    .groupby("extraction_stage")
    .size()
)

print()
print("Duplicate file : 0")
print("NaN / Inf      : PASS")

print()
print("Dataset sach:")
print(OUTPUT_CSV)

print()
print("Summary:")
print(OUTPUT_SUMMARY)
