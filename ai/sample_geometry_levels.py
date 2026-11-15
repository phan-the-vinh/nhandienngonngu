from pathlib import Path
import shutil

import cv2
import pandas as pd
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 1. DUONG DAN
# ============================================================

AI_DIR = Path(__file__).resolve().parent
PROJECT_DIR = AI_DIR.parent

INPUT_CSV = (
    AI_DIR
    / "dataset"
    / "quality_relative_outliers.csv"
)

HAND_MODEL = (
    AI_DIR
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)

OUTPUT_DIR = (
    AI_DIR
    / "review_geometry_levels"
)

OUTPUT_CSV = (
    AI_DIR
    / "dataset"
    / "geometry_level_sample.csv"
)


# ============================================================
# 2. SO MAU MOI NHOM
# ============================================================

SAMPLES_PER_GROUP = 10

RANDOM_STATE = 42


# ============================================================
# 3. DUONG NOI LANDMARK
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
# 4. KIEM TRA FILE
# ============================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Khong tim thay:\n{INPUT_CSV}"
    )

if not HAND_MODEL.exists():
    raise FileNotFoundError(
        f"Khong tim thay:\n{HAND_MODEL}"
    )


# ============================================================
# 5. DOC DU LIEU
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("Tong robust outlier:", len(df))


# ============================================================
# 6. BO 20 MAU DA KIEM TRA THU CONG TRUOC
# ============================================================

df_new = df[
    df["manual_status"].fillna("") == ""
].copy()

print(
    "Con lai sau khi bo mau da xem:",
    len(df_new)
)


# ============================================================
# 7. TINH CAC MOC PHAN VI
# ============================================================

q20 = df_new[
    "geometry_score"
].quantile(0.20)

q40 = df_new[
    "geometry_score"
].quantile(0.40)

q60 = df_new[
    "geometry_score"
].quantile(0.60)

q80 = df_new[
    "geometry_score"
].quantile(0.80)


print()
print("MOC GEOMETRY SCORE:")
print(f"20% = {q20:.3f}")
print(f"40% = {q40:.3f}")
print(f"60% = {q60:.3f}")
print(f"80% = {q80:.3f}")


# ============================================================
# 8. CHIA 3 NHOM
# ============================================================

low_pool = df_new[
    df_new["geometry_score"] <= q20
].copy()

medium_pool = df_new[
    (
        df_new["geometry_score"] >= q40
    )
    &
    (
        df_new["geometry_score"] <= q60
    )
].copy()

high_pool = df_new[
    df_new["geometry_score"] >= q80
].copy()


print()
print("SO MAU TRONG TUNG VUNG:")
print("LOW   :", len(low_pool))
print("MEDIUM:", len(medium_pool))
print("HIGH  :", len(high_pool))


# ============================================================
# 9. LAY NGAU NHIEN 10 MAU MOI NHOM
# ============================================================

low_sample = low_pool.sample(
    n=SAMPLES_PER_GROUP,
    random_state=RANDOM_STATE
).copy()

low_sample["level"] = "LOW"


medium_sample = medium_pool.sample(
    n=SAMPLES_PER_GROUP,
    random_state=RANDOM_STATE
).copy()

medium_sample["level"] = "MEDIUM"


high_sample = high_pool.sample(
    n=SAMPLES_PER_GROUP,
    random_state=RANDOM_STATE
).copy()

high_sample["level"] = "HIGH"


sample = pd.concat(
    [
        high_sample,
        medium_sample,
        low_sample
    ],
    ignore_index=True
)


# ============================================================
# 10. DAT SO THU TU TRONG TUNG NHOM
# ============================================================

sample["candidate"] = (
    sample.groupby("level").cumcount()
    + 1
)


# ============================================================
# 11. LUU DANH SACH MAU
# ============================================================

sample.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 12. TAO LAI THU MUC REVIEW
# ============================================================

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

for level in [
    "HIGH",
    "MEDIUM",
    "LOW"
]:

    (
        OUTPUT_DIR
        / level
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 13. KHOI TAO MEDIAPIPE
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=str(HAND_MODEL)
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)


# ============================================================
# 14. VE LANDMARK LEN 30 ANH
# ============================================================

with vision.HandLandmarker.create_from_options(
    options
) as landmarker:

    for row in sample.itertuples(
        index=False
    ):

        image_path = (
            PROJECT_DIR
            / Path(row.file)
        )

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:

            print(
                "Khong doc duoc:",
                image_path
            )

            continue


        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(
            mp_image
        )


        if not result.hand_landmarks:

            print(
                "Khong phat hien tay:",
                image_path
            )

            continue


        hand = result.hand_landmarks[0]

        height, width = frame.shape[:2]


        # ====================================================
        # 15. DOI LANDMARK SANG PIXEL
        # ====================================================

        points = []

        for landmark in hand:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            points.append(
                (x, y)
            )


        # ====================================================
        # 16. VE CAC DUONG NOI
        # ====================================================

        for start, end in CONNECTIONS:

            cv2.line(
                frame,
                points[start],
                points[end],
                (255, 255, 255),
                2
            )


        # ====================================================
        # 17. VE CAC DIEM 0-20
        # ====================================================

        for index, (x, y) in enumerate(
            points
        ):

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                str(index),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )


        # ====================================================
        # 18. GHI THONG TIN LEN ANH
        # ====================================================

        text1 = (
            f"{row.level} "
            f"{int(row.candidate):02d} | "
            f"Label: {row.label}"
        )

        text2 = (
            f"Geometry: "
            f"{row.geometry_score:.2f}"
        )

        text3 = (
            f"Robust: "
            f"{row.robust_score:.2f}"
        )


        cv2.putText(
            frame,
            text1,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            text2,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            text3,
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # 19. LUU VAO DUNG NHOM
        # ====================================================

        output_name = (
            f"{int(row.candidate):02d}"
            f"_{row.label}"
            f"_geo_{row.geometry_score:.2f}"
            f"_{image_path.name}"
        )

        output_path = (
            OUTPUT_DIR
            / row.level
            / output_name
        )

        cv2.imwrite(
            str(output_path),
            frame
        )


# ============================================================
# 20. HOAN THANH
# ============================================================

print()
print("===================================")
print("HOAN THANH")
print("===================================")

print(
    "Danh sach mau:"
)

print(
    OUTPUT_CSV
)

print()

print(
    "Thu muc review:"
)

print(
    OUTPUT_DIR
)
