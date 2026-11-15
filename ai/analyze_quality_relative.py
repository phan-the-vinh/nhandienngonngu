from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# 1. DUONG DAN
# ============================================================

AI_DIR = Path(__file__).resolve().parent

DATASET_FILE = (
    AI_DIR
    / "dataset"
    / "dataset_chucai_all.csv"
)

RAW_FILE = (
    AI_DIR
    / "raw_data"
    / "raw_landmarks_chucai_all.csv"
)

OUTLIER_FILE = (
    AI_DIR
    / "dataset"
    / "landmark_outliers_chucai.csv"
)

OUTPUT_FILE = (
    AI_DIR
    / "dataset"
    / "quality_relative_outliers.csv"
)


# ============================================================
# 2. CAC DOAN XUONG BAN TAY
# ============================================================

CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17),
]


# ============================================================
# 3. CAC ANH BAN DA XAC NHAN LANDMARK SAI
# ============================================================

BAD_RANKS = {
    1, 2, 3, 4, 5,
    6, 8, 9, 10, 11
}


# ============================================================
# 4. DOC DU LIEU
# ============================================================

print("Dang doc dataset...")

df = pd.read_csv(DATASET_FILE)

raw = pd.read_csv(RAW_FILE)

outliers = pd.read_csv(
    OUTLIER_FILE
).sort_values(
    "robust_score",
    ascending=False
).reset_index(drop=True)


print("Tong dataset:", len(df))
print("Tong robust outlier:", len(outliers))


# ============================================================
# 5. TIM DUNG 63 FEATURE
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

if len(feature_columns) != 63:
    raise ValueError(
        f"Can 63 feature, "
        f"nhung tim thay {len(feature_columns)}"
    )


# ============================================================
# 6. KIEM TRA RAW VA NORMALIZED CO CUNG FILE KHONG
# ============================================================

if len(df) != len(raw):
    raise ValueError(
        "So dong dataset va raw khong bang nhau."
    )

if not df["file"].equals(raw["file"]):
    raise ValueError(
        "Thu tu file cua dataset va raw khong trung nhau."
    )


# ============================================================
# 7. CHUYEN 63 FEATURE THANH:
#
# so_mau x 21 landmark x 3 toa_do
# ============================================================

X = df[
    feature_columns
].to_numpy(dtype=float)

points = X.reshape(
    -1,
    21,
    3
)


# ============================================================
# 8. HAM KHOANG CACH VECTOR
# ============================================================

def dist(a, b):
    return np.linalg.norm(
        a - b,
        axis=1
    )


# ============================================================
# 9. CAC CHI SO HINH HOC CHINH
# ============================================================

# Do rong long ban tay:
# goc ngon tro 5 -> goc ngon ut 17
palm_width = dist(
    points[:, 5, :],
    points[:, 17, :]
)

# Co tay 0 -> goc ngon giua 9
wrist_middle = dist(
    points[:, 0, :],
    points[:, 9, :]
)

# Chuoi ngang qua cac MCP:
# 5 -> 9 -> 13 -> 17
mcp_chain = (
    dist(
        points[:, 5, :],
        points[:, 9, :]
    )
    +
    dist(
        points[:, 9, :],
        points[:, 13, :]
    )
    +
    dist(
        points[:, 13, :],
        points[:, 17, :]
    )
)


# ============================================================
# 10. DO DAI CAC DOAN XUONG
# ============================================================

bone_lengths = []

for start, end in CONNECTIONS:

    length = dist(
        points[:, start, :],
        points[:, end, :]
    )

    bone_lengths.append(length)

bone_lengths = np.column_stack(
    bone_lengths
)


min_bone = bone_lengths.min(axis=1)

mean_bone = bone_lengths.mean(axis=1)

std_bone = bone_lengths.std(axis=1)

bone_cv = np.divide(
    std_bone,
    mean_bone,
    out=np.zeros_like(std_bone),
    where=mean_bone > 0
)


# ============================================================
# 11. KHOANG CACH NHO NHAT GIUA BAT KY 2 LANDMARK
# ============================================================

min_pair = np.full(
    len(df),
    np.inf
)

for i in range(21):

    for j in range(i + 1, 21):

        d = dist(
            points[:, i, :],
            points[:, j, :]
        )

        min_pair = np.minimum(
            min_pair,
            d
        )


# ============================================================
# 12. DO LAN RONG 3D CUA TOAN BO LANDMARK
# ============================================================

ranges = (
    points.max(axis=1)
    -
    points.min(axis=1)
)

spread_3d = np.linalg.norm(
    ranges,
    axis=1
)


# ============================================================
# 13. RAW BOUNDING BOX
#
# Chi dung x,y goc cua MediaPipe,
# truoc normalize.
# ============================================================

raw_x_columns = [
    f"x{i}"
    for i in range(21)
]

raw_y_columns = [
    f"y{i}"
    for i in range(21)
]


raw_x = raw[
    raw_x_columns
].to_numpy(dtype=float)

raw_y = raw[
    raw_y_columns
].to_numpy(dtype=float)


raw_bbox_width = (
    raw_x.max(axis=1)
    -
    raw_x.min(axis=1)
)

raw_bbox_height = (
    raw_y.max(axis=1)
    -
    raw_y.min(axis=1)
)


# ============================================================
# 14. THEM CAC CHI SO VAO DATAFRAME
# ============================================================

metrics = pd.DataFrame({
    "file": df["file"],
    "label": df["label"],
    "handedness": df["handedness"],

    "palm_width": palm_width,
    "wrist_middle": wrist_middle,
    "mcp_chain": mcp_chain,

    "min_bone": min_bone,
    "bone_cv": bone_cv,

    "min_pair": min_pair,
    "spread_3d": spread_3d,

    "raw_bbox_width": raw_bbox_width,
    "raw_bbox_height": raw_bbox_height,
})


# ============================================================
# 15. CAC CHI SO SE DUOC SO SANH TUONG DOI
# ============================================================

METRIC_COLUMNS = [
    "palm_width",
    "wrist_middle",
    "mcp_chain",
    "min_bone",
    "bone_cv",
    "min_pair",
    "spread_3d",
    "raw_bbox_width",
    "raw_bbox_height",
]


# ============================================================
# 16. HAM ROBUST Z-SCORE
#
# So sanh voi MEDIAN + MAD cua
# dung nhom label + handedness.
# ============================================================

def robust_absolute_z(values):

    values = np.asarray(
        values,
        dtype=float
    )

    median = np.median(values)

    mad = np.median(
        np.abs(
            values - median
        )
    )

    if mad < 1e-12:

        return np.zeros_like(values)

    z = (
        0.6745
        * (values - median)
        / mad
    )

    return np.abs(z)


# ============================================================
# 17. TINH Z-SCORE THEO TUNG NHOM
#
# A-Left  rieng
# A-Right rieng
# B-Left  rieng
# ...
# ============================================================

for metric in METRIC_COLUMNS:

    metrics[
        f"z_{metric}"
    ] = 0.0


groups = metrics.groupby(
    ["label", "handedness"],
    sort=True
)


for (label, handedness), group in groups:

    indexes = group.index

    for metric in METRIC_COLUMNS:

        values = group[
            metric
        ].to_numpy(dtype=float)

        z = robust_absolute_z(
            values
        )

        metrics.loc[
            indexes,
            f"z_{metric}"
        ] = z


# ============================================================
# 18. TONG HOP MUC BAT THUONG HINH HOC
#
# Khong chi dua vao 1 chi so.
#
# Lay 3 z-score lon nhat cua moi mau.
# ============================================================

z_columns = [
    f"z_{metric}"
    for metric in METRIC_COLUMNS
]

Z = metrics[
    z_columns
].to_numpy(dtype=float)


# Sap xep z-score tang dan tren tung dong
sorted_z = np.sort(
    Z,
    axis=1
)

# 3 chi so bat thuong nhat
top1 = sorted_z[:, -1]
top2 = sorted_z[:, -2]
top3 = sorted_z[:, -3]


# Geometry score =
# trung binh 3 dau hieu bat thuong nhat
geometry_score = (
    top1
    +
    top2
    +
    top3
) / 3.0


metrics[
    "geometry_score"
] = geometry_score


# Dem co bao nhieu chi so co robust z > 5
metrics[
    "geometry_evidence_5"
] = (
    Z > 5.0
).sum(axis=1)


# ============================================================
# 19. CHI LAY 1318 ROBUST OUTLIER
#
# Chua xoa du lieu.
# Chi ghep them geometry_score.
# ============================================================

result = outliers.merge(
    metrics,
    on=[
        "file",
        "label",
        "handedness"
    ],
    how="left"
)


# ============================================================
# 20. THEM RANK CUA ROBUST OUTLIER
# ============================================================

result.insert(
    0,
    "rank",
    np.arange(
        1,
        len(result) + 1
    )
)


# ============================================================
# 21. DANH DAU 20 ANH BAN DA KIEM TRA
# ============================================================

result[
    "manual_status"
] = ""


for rank in range(1, 21):

    if rank in BAD_RANKS:

        result.loc[
            result["rank"] == rank,
            "manual_status"
        ] = "BAD"

    else:

        result.loc[
            result["rank"] == rank,
            "manual_status"
        ] = "REVIEW"


# ============================================================
# 22. LUU TOAN BO 1318 OUTLIER
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 23. HIEN THI 20 ANH DA KIEM TRA
# ============================================================

print()
print("===================================")
print("20 MAU DA KIEM TRA THU CONG")
print("===================================")

show_columns = [
    "rank",
    "manual_status",
    "label",
    "robust_score",
    "geometry_score",
    "geometry_evidence_5",

    "z_palm_width",
    "z_mcp_chain",
    "z_min_bone",
    "z_min_pair",
    "z_bone_cv",
    "z_raw_bbox_width",
]


print(
    result
    .head(20)[show_columns]
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 24. SO SANH BAD VA REVIEW
# ============================================================

top20 = result.head(20)

print()
print("===================================")
print("TRUNG BINH BAD vs REVIEW")
print("===================================")

compare_columns = [
    "robust_score",
    "geometry_score",
    "geometry_evidence_5",
    "z_palm_width",
    "z_mcp_chain",
    "z_min_bone",
    "z_min_pair",
    "z_bone_cv",
    "z_raw_bbox_width",
]


print(
    top20
    .groupby(
        "manual_status"
    )[compare_columns]
    .mean()
    .round(3)
    .to_string()
)


# ============================================================
# 25. 20 MAU CO GEOMETRY SCORE CAO NHAT
# ============================================================

print()
print("===================================")
print("20 GEOMETRY SCORE CAO NHAT")
print("===================================")

print(
    result
    .sort_values(
        "geometry_score",
        ascending=False
    )
    .head(20)[
        [
            "rank",
            "label",
            "handedness",
            "robust_score",
            "geometry_score",
            "geometry_evidence_5",
            "file",
        ]
    ]
    .round(3)
    .to_string(index=False)
)


print()
print("Da luu ket qua:")
print(OUTPUT_FILE)
