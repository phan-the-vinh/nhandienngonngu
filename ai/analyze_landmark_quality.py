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
    / "quality_metrics_top20.csv"
)


# ============================================================
# 2. CAC ANH BAN DA XAC NHAN LANDMARK SAI
# ============================================================

BAD_RANKS = {
    1, 2, 3, 4, 5,
    6, 8, 9, 10, 11
}


# ============================================================
# 3. CAC DUONG XUONG BAN TAY
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
# 4. DOC DU LIEU
# ============================================================

dataset = pd.read_csv(DATASET_FILE)

raw = pd.read_csv(RAW_FILE)

outliers = pd.read_csv(
    OUTLIER_FILE
).sort_values(
    "robust_score",
    ascending=False
).head(20).reset_index(drop=True)


# ============================================================
# 5. TIM 63 FEATURE
# ============================================================

feature_columns = [
    column
    for column in dataset.columns
    if re.fullmatch(r"f\d+", column)
]

feature_columns = sorted(
    feature_columns,
    key=lambda name: int(name[1:])
)

if len(feature_columns) != 63:
    raise ValueError(
        f"Can 63 feature, tim thay "
        f"{len(feature_columns)}"
    )


# ============================================================
# 6. DAT FILE LAM INDEX DE TIM NHANH
# ============================================================

dataset_by_file = (
    dataset
    .drop_duplicates("file")
    .set_index("file")
)

raw_by_file = (
    raw
    .drop_duplicates("file")
    .set_index("file")
)


# ============================================================
# 7. HAM TINH KHOANG CACH
# ============================================================

def distance(point_a, point_b):
    return float(
        np.linalg.norm(
            point_a - point_b
        )
    )


# ============================================================
# 8. PHAN TICH 20 ANH
# ============================================================

results = []


for index, outlier in outliers.iterrows():

    rank = index + 1
    file_path = outlier["file"]

    if file_path not in dataset_by_file.index:
        print(
            f"Khong tim thay dataset: {file_path}"
        )
        continue

    if file_path not in raw_by_file.index:
        print(
            f"Khong tim thay raw: {file_path}"
        )
        continue


    # ========================================================
    # LANDMARK DA NORMALIZE
    # ========================================================

    row = dataset_by_file.loc[
        file_path
    ]

    values = (
        row[feature_columns]
        .to_numpy(dtype=float)
    )

    points = values.reshape(
        21,
        3
    )


    # ========================================================
    # 9. KHOANG CACH GIUA MOI CAP LANDMARK
    # ========================================================

    pair_distances = []

    for i in range(21):
        for j in range(i + 1, 21):

            pair_distances.append(
                distance(
                    points[i],
                    points[j]
                )
            )

    pair_distances = np.array(
        pair_distances
    )


    # ========================================================
    # 10. DO DAI CAC DOAN XUONG
    # ========================================================

    bone_lengths = []

    for start, end in CONNECTIONS:

        bone_lengths.append(
            distance(
                points[start],
                points[end]
            )
        )

    bone_lengths = np.array(
        bone_lengths
    )


    # ========================================================
    # 11. CAC CHI SO LONG BAN TAY
    # ========================================================

    # Khoang cach goc ngon tro -> goc ngon ut
    palm_width = distance(
        points[5],
        points[17]
    )

    # Co tay -> goc ngon giua
    wrist_middle = distance(
        points[0],
        points[9]
    )

    # Tong khoang cach ngang qua cac MCP
    mcp_chain = (
        distance(points[5], points[9])
        +
        distance(points[9], points[13])
        +
        distance(points[13], points[17])
    )


    # ========================================================
    # 12. DO LAN RONG TOAN BO BO LANDMARK
    # ========================================================

    ranges = (
        points.max(axis=0)
        -
        points.min(axis=0)
    )

    spread_3d = float(
        np.linalg.norm(ranges)
    )


    # ========================================================
    # 13. DEM CAC DIEM QUA GAN NHAU
    #
    # Chi la CHI SO PHAN TICH,
    # chua phai nguong de xoa.
    # ========================================================

    very_close_pairs = int(
        np.sum(
            pair_distances < 0.03
        )
    )


    # ========================================================
    # 14. CAC DOAN XUONG CUC NGAN
    # ========================================================

    tiny_bones = int(
        np.sum(
            bone_lengths < 0.05
        )
    )


    min_pair_distance = float(
        pair_distances.min()
    )

    min_bone_length = float(
        bone_lengths.min()
    )

    mean_bone_length = float(
        bone_lengths.mean()
    )


    # ========================================================
    # 15. DO BIEN THIEN DO DAI XUONG
    # ========================================================

    if mean_bone_length > 0:

        bone_cv = float(
            bone_lengths.std()
            / mean_bone_length
        )

    else:
        bone_cv = 0.0


    # ========================================================
    # 16. KIEM TRA LANDMARK NAM NGOAI ANH
    #
    # Dung toa do RAW x,y cua MediaPipe.
    # ========================================================

    raw_row = raw_by_file.loc[
        file_path
    ]

    outside_count = 0

    raw_x = []
    raw_y = []

    for landmark_index in range(21):

        x = float(
            raw_row[f"x{landmark_index}"]
        )

        y = float(
            raw_row[f"y{landmark_index}"]
        )

        raw_x.append(x)
        raw_y.append(y)

        if (
            x < 0
            or x > 1
            or y < 0
            or y > 1
        ):
            outside_count += 1


    raw_bbox_width = (
        max(raw_x)
        -
        min(raw_x)
    )

    raw_bbox_height = (
        max(raw_y)
        -
        min(raw_y)
    )


    # ========================================================
    # 17. NHAN DANH DAU CUA BAN
    # ========================================================

    if rank in BAD_RANKS:
        status = "BAD"
    else:
        status = "REVIEW"


    results.append({
        "rank": rank,
        "status": status,
        "file": file_path,
        "label": outlier["label"],
        "robust_score": outlier["robust_score"],

        "min_pair": min_pair_distance,
        "very_close_pairs": very_close_pairs,

        "min_bone": min_bone_length,
        "tiny_bones": tiny_bones,

        "palm_width": palm_width,
        "wrist_middle": wrist_middle,
        "mcp_chain": mcp_chain,

        "spread_3d": spread_3d,
        "bone_cv": bone_cv,

        "outside_count": outside_count,

        "raw_bbox_width": raw_bbox_width,
        "raw_bbox_height": raw_bbox_height,
    })


# ============================================================
# 18. TAO DATAFRAME
# ============================================================

result_df = pd.DataFrame(
    results
)


# ============================================================
# 19. LUU KET QUA
# ============================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 20. HIEN THI BANG QUAN TRONG
# ============================================================

display_columns = [
    "rank",
    "status",
    "label",
    "robust_score",
    "min_pair",
    "very_close_pairs",
    "min_bone",
    "tiny_bones",
    "palm_width",
    "wrist_middle",
    "outside_count"
]

print()
print("===================================")
print("PHAN TICH 20 OUTLIER")
print("===================================")

print(
    result_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# 21. SO SANH NHOM BAD VA REVIEW
# ============================================================

print()
print("===================================")
print("TRUNG BINH THEO NHOM")
print("===================================")

numeric_columns = [
    "robust_score",
    "min_pair",
    "very_close_pairs",
    "min_bone",
    "tiny_bones",
    "palm_width",
    "wrist_middle",
    "mcp_chain",
    "spread_3d",
    "bone_cv",
    "outside_count",
    "raw_bbox_width",
    "raw_bbox_height",
]

print(
    result_df
    .groupby("status")[
        numeric_columns
    ]
    .mean()
    .round(4)
    .to_string()
)


print()
print("Da luu ket qua tai:")
print(OUTPUT_FILE)
