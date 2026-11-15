from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# 1. DUONG DAN
# ============================================================

AI_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    AI_DIR
    / "dataset"
    / "dataset_chucai_all.csv"
)

OUTPUT_FILE = (
    AI_DIR
    / "dataset"
    / "landmark_outliers_chucai.csv"
)


# ============================================================
# 2. NGUONG PHAT HIEN
#
# Chi danh dau nhung mau RAT bat thuong.
# Khong xoa du lieu.
# ============================================================

OUTLIER_THRESHOLD = 6.0

# Nhom qua nho se khong tu danh gia outlier.
MIN_GROUP_SIZE = 200


# ============================================================
# 3. DOC DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("Da doc dataset.")
print("So mau:", len(df))


# ============================================================
# 4. TIM CHINH XAC 63 FEATURE
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if re.fullmatch(r"f\d+", column)
]

feature_columns = sorted(
    feature_columns,
    key=lambda name: int(name[1:])
)

print("So feature:", len(feature_columns))


if len(feature_columns) != 63:
    raise ValueError(
        f"Can 63 feature, nhung tim thay "
        f"{len(feature_columns)}"
    )


# ============================================================
# 5. NOI LUU KET QUA OUTLIER
# ============================================================

outlier_rows = []


# ============================================================
# 6. CHIA THEO LABEL + HANDEDNESS
# ============================================================

grouped = df.groupby(
    ["label", "handedness"],
    sort=True
)


for (label, handedness), group in grouped:

    print()
    print(
        f"===== {label} - {handedness} "
        f"({len(group)} mau) ====="
    )


    # ========================================================
    # Nhom qua nho thi khong danh gia tu dong
    # ========================================================

    if len(group) < MIN_GROUP_SIZE:

        print(
            "Bo qua: nhom qua nho de danh gia an toan."
        )

        continue


    # ========================================================
    # Chuyen 63 feature thanh numpy array
    # ========================================================

    X = group[
        feature_columns
    ].to_numpy(dtype=float)


    # ========================================================
    # 7. TIM TAM ROBUST BANG MEDIAN
    #
    # Moi feature co mot median rieng.
    # ========================================================

    center = np.median(
        X,
        axis=0
    )


    # ========================================================
    # 8. TINH KHOANG CACH TU MOI MAU DEN TAM
    # ========================================================

    differences = X - center

    distances = np.linalg.norm(
        differences,
        axis=1
    )


    # ========================================================
    # 9. MEDIAN CUA KHOANG CACH
    # ========================================================

    median_distance = np.median(
        distances
    )


    # ========================================================
    # 10. MAD
    #
    # Median Absolute Deviation
    # ========================================================

    absolute_deviations = np.abs(
        distances - median_distance
    )

    mad = np.median(
        absolute_deviations
    )


    # ========================================================
    # Tranh chia cho 0
    # ========================================================

    if mad < 1e-12:

        print(
            "MAD qua nho, bo qua nhom nay."
        )

        continue


    # ========================================================
    # 11. ROBUST Z-SCORE
    #
    # He so 0.6745 thuong dung voi MAD.
    # ========================================================

    robust_scores = (
        0.6745
        * (distances - median_distance)
        / mad
    )


    # ========================================================
    # 12. CHI DANH DAU MAU RAT BAT THUONG
    # ========================================================

    outlier_mask = (
        robust_scores
        > OUTLIER_THRESHOLD
    )


    outlier_indices = np.where(
        outlier_mask
    )[0]


    print(
        "So mau bi danh dau:",
        len(outlier_indices)
    )


    # ========================================================
    # 13. LUU THONG TIN MAU NGHI NGO
    # ========================================================

    for position in outlier_indices:

        original_row = group.iloc[
            position
        ]

        outlier_rows.append({
            "file": original_row["file"],
            "label": label,
            "handedness": handedness,
            "distance": distances[position],
            "robust_score": robust_scores[position]
        })


# ============================================================
# 14. TAO DATAFRAME OUTLIER
# ============================================================

outlier_df = pd.DataFrame(
    outlier_rows,
    columns=[
        "file",
        "label",
        "handedness",
        "distance",
        "robust_score"
    ]
)


# ============================================================
# 15. SAP XEP MAU BAT THUONG NHAT LEN DAU
# ============================================================

if not outlier_df.empty:

    outlier_df = outlier_df.sort_values(
        "robust_score",
        ascending=False
    )


# ============================================================
# 16. LUU FILE
# ============================================================

outlier_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 17. THONG KE
# ============================================================

print()
print("===================================")
print("HOAN THANH PHAT HIEN OUTLIER")
print("===================================")

print(
    "Tong dataset:",
    len(df)
)

print(
    "So mau nghi landmark bat thuong:",
    len(outlier_df)
)

if len(df) > 0:

    percentage = (
        len(outlier_df)
        / len(df)
        * 100
    )

    print(
        f"Ty le bi danh dau: "
        f"{percentage:.2f}%"
    )


if not outlier_df.empty:

    print()
    print("SO OUTLIER THEO KY TU:")

    print(
        outlier_df
        .groupby("label")
        .size()
    )


print()
print("Da luu tai:")
print(OUTPUT_FILE)
