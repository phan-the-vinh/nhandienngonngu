from pathlib import Path
import pandas as pd


AI_DIR = Path(__file__).resolve().parent

GEOMETRY_FILE = (
    AI_DIR
    / "dataset"
    / "quality_relative_outliers.csv"
)

IMAGE_FILE = (
    AI_DIR
    / "dataset"
    / "image_quality_outliers.csv"
)

OUTPUT_FILE = (
    AI_DIR
    / "dataset"
    / "reject_candidates_chucai.csv"
)


# ============================================================
# 1. DOC DU LIEU
# ============================================================

geometry = pd.read_csv(
    GEOMETRY_FILE
)

image_quality = pd.read_csv(
    IMAGE_FILE
)


# ============================================================
# 2. CHI LAY CAC COT CAN THIET
# ============================================================

image_quality = image_quality[
    [
        "file",
        "blur_hand",
        "hand_area",
        "image_risk_score"
    ]
]


# ============================================================
# 3. GHEP DU LIEU THEO FILE
# ============================================================

df = geometry.merge(
    image_quality,
    on="file",
    how="left"
)


# ============================================================
# 4. TIEU CHI REJECT CANDIDATE
# ============================================================

reject_mask = (
    (df["geometry_evidence_5"] >= 2)
    &
    (df["image_risk_score"] >= 0.90)
)


df["status"] = "REVIEW"

df.loc[
    reject_mask,
    "status"
] = "REJECT_CANDIDATE"


# ============================================================
# 5. LAY RIENG CAC MAU NGHI LOAI
# ============================================================

reject_df = df[
    df["status"]
    ==
    "REJECT_CANDIDATE"
].copy()


# ============================================================
# 6. SAP XEP MAU NGUY CO CAO LEN DAU
# ============================================================

reject_df = reject_df.sort_values(
    [
        "image_risk_score",
        "geometry_evidence_5",
        "geometry_score"
    ],
    ascending=[
        False,
        False,
        False
    ]
)


# ============================================================
# 7. LUU DANH SACH
# ============================================================

reject_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 8. THONG KE
# ============================================================

print()
print("===================================")
print("THONG KE REJECT CANDIDATE")
print("===================================")

print(
    "Tong robust outlier:",
    len(df)
)

print(
    "REJECT_CANDIDATE:",
    len(reject_df)
)

print(
    "REVIEW:",
    len(df) - len(reject_df)
)

print()

if len(df) > 0:

    rate = (
        len(reject_df)
        / len(df)
        * 100
    )

    print(
        f"Ty le reject trong robust outlier: "
        f"{rate:.2f}%"
    )


print()
print("SO REJECT THEO KY TU:")

print(
    reject_df
    .groupby("label")
    .size()
)


print()
print("Da luu tai:")
print(OUTPUT_FILE)
