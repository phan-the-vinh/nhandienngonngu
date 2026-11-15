from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# 1. DUONG DAN
# ============================================================

AI_DIR = Path(__file__).resolve().parent
PROJECT_DIR = AI_DIR.parent

INPUT_FILE = (
    AI_DIR
    / "dataset"
    / "quality_relative_outliers.csv"
)

RAW_FILE = (
    AI_DIR
    / "raw_data"
    / "raw_landmarks_chucai_all.csv"
)

OUTPUT_FILE = (
    AI_DIR
    / "dataset"
    / "image_quality_outliers.csv"
)


# ============================================================
# 2. KIEM TRA FILE
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Khong tim thay:\n{INPUT_FILE}"
    )

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Khong tim thay:\n{RAW_FILE}"
    )


# ============================================================
# 3. DOC DU LIEU
# ============================================================

print("Dang doc du lieu...")

df = pd.read_csv(INPUT_FILE)

raw = pd.read_csv(RAW_FILE)

print(
    "So robust outlier:",
    len(df)
)


# ============================================================
# 4. DAT RAW LANDMARK THEO TEN FILE
# ============================================================

raw_by_file = (
    raw
    .drop_duplicates("file")
    .set_index("file")
)


# ============================================================
# 5. HAM TINH BLUR SCORE
# ============================================================

def calculate_blur_score(image):

    if image is None or image.size == 0:
        return np.nan

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )


# ============================================================
# 6. NOI LUU KET QUA
# ============================================================

quality_rows = []


# ============================================================
# 7. PHAN TICH TUNG ANH
# ============================================================

for index, row in df.iterrows():

    file_path = row["file"]

    image_path = (
        PROJECT_DIR
        / Path(file_path)
    )


    # --------------------------------------------------------
    # Doc anh
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"Khong doc duoc: {file_path}"
        )

        continue


    height, width = image.shape[:2]


    # --------------------------------------------------------
    # Blur cua toan bo anh
    # --------------------------------------------------------

    blur_full = calculate_blur_score(
        image
    )


    # --------------------------------------------------------
    # Lay RAW landmark
    # --------------------------------------------------------

    if file_path not in raw_by_file.index:

        print(
            f"Khong tim thay raw: {file_path}"
        )

        continue


    raw_row = raw_by_file.loc[
        file_path
    ]


    xs = np.array([
        float(
            raw_row[f"x{i}"]
        )
        for i in range(21)
    ])

    ys = np.array([
        float(
            raw_row[f"y{i}"]
        )
        for i in range(21)
    ])


    # --------------------------------------------------------
    # Bounding box landmark theo toa do normalized
    # --------------------------------------------------------

    x_min = float(xs.min())
    x_max = float(xs.max())

    y_min = float(ys.min())
    y_max = float(ys.max())


    hand_width = (
        x_max
        -
        x_min
    )

    hand_height = (
        y_max
        -
        y_min
    )

    hand_area = (
        hand_width
        *
        hand_height
    )


    # --------------------------------------------------------
    # Chuyen bounding box sang pixel
    # --------------------------------------------------------

    px_min = int(
        x_min * width
    )

    px_max = int(
        x_max * width
    )

    py_min = int(
        y_min * height
    )

    py_max = int(
        y_max * height
    )


    # --------------------------------------------------------
    # Them margin 25%
    #
    # De crop khong qua sat landmark.
    # --------------------------------------------------------

    box_width = (
        px_max
        -
        px_min
    )

    box_height = (
        py_max
        -
        py_min
    )


    margin_x = int(
        box_width * 0.25
    )

    margin_y = int(
        box_height * 0.25
    )


    px_min = max(
        0,
        px_min - margin_x
    )

    px_max = min(
        width,
        px_max + margin_x
    )

    py_min = max(
        0,
        py_min - margin_y
    )

    py_max = min(
        height,
        py_max + margin_y
    )


    # --------------------------------------------------------
    # Crop vung ban tay
    # --------------------------------------------------------

    hand_crop = image[
        py_min:py_max,
        px_min:px_max
    ]


    # --------------------------------------------------------
    # Blur cua vung ban tay
    # --------------------------------------------------------

    if (
        hand_crop.size > 0
        and hand_crop.shape[0] >= 10
        and hand_crop.shape[1] >= 10
    ):

        blur_hand = (
            calculate_blur_score(
                hand_crop
            )
        )

    else:

        blur_hand = np.nan


    # --------------------------------------------------------
    # Luu ket qua
    # --------------------------------------------------------

    quality_rows.append({
        "file": file_path,
        "label": row["label"],
        "handedness": row["handedness"],

        "robust_score":
            row["robust_score"],

        "geometry_score":
            row["geometry_score"],

        "geometry_evidence_5":
            row["geometry_evidence_5"],

        "blur_full":
            blur_full,

        "blur_hand":
            blur_hand,

        "hand_width":
            hand_width,

        "hand_height":
            hand_height,

        "hand_area":
            hand_area,

        "image_width":
            width,

        "image_height":
            height,
    })


# ============================================================
# 8. TAO DATAFRAME
# ============================================================

quality = pd.DataFrame(
    quality_rows
)


# ============================================================
# 9. TINH PHAN VI
#
# Day chi de XEP HANG,
# KHONG PHAI NGUONG XOA.
# ============================================================

quality[
    "blur_percentile"
] = quality[
    "blur_hand"
].rank(
    pct=True
)


quality[
    "area_percentile"
] = quality[
    "hand_area"
].rank(
    pct=True
)


# ============================================================
# 10. IMAGE RISK SCORE
#
# blur_percentile thap
# → nguy co mo cao
#
# area_percentile thap
# → landmark/ban tay chiem vung nho
#
# Chi dung xep hang.
# ============================================================

quality[
    "image_risk_score"
] = (
    (
        1
        -
        quality[
            "blur_percentile"
        ]
    )
    +
    (
        1
        -
        quality[
            "area_percentile"
        ]
    )
) / 2.0


# ============================================================
# 11. LUU FILE
# ============================================================

quality.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 12. THONG KE PHAN VI
# ============================================================

print()
print("===================================")
print("PHAN VI BLUR_HAND")
print("===================================")

print(
    quality[
        "blur_hand"
    ].quantile(
        [
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
)


print()
print("===================================")
print("PHAN VI HAND_AREA")
print("===================================")

print(
    quality[
        "hand_area"
    ].quantile(
        [
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
)


# ============================================================
# 13. 20 MAU IMAGE RISK CAO NHAT
# ============================================================

print()
print("===================================")
print("20 IMAGE RISK CAO NHAT")
print("===================================")

show_columns = [
    "label",
    "robust_score",
    "geometry_score",
    "blur_hand",
    "hand_area",
    "image_risk_score",
    "file",
]

print(
    quality
    .sort_values(
        "image_risk_score",
        ascending=False
    )
    .head(20)[
        show_columns
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("Da luu tai:")
print(OUTPUT_FILE)
