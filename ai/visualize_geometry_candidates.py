from pathlib import Path

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

# File ket qua tu analyze_quality_relative.py
INPUT_CSV = (
    AI_DIR
    / "dataset"
    / "quality_relative_outliers.csv"
)

# Model MediaPipe
HAND_MODEL = (
    AI_DIR
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)

# Thu muc se tu dong duoc tao
OUTPUT_DIR = (
    AI_DIR
    / "review_geometry_landmarks"
)


# ============================================================
# 2. SO MAU MUON KIEM TRA
# ============================================================

TOP_N = 20


# ============================================================
# 3. CAC DUONG NOI CUA BAN TAY
# ============================================================

CONNECTIONS = [
    # Ngon cai
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Ngon tro
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Ngon giua
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ngon ap ut
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Ngon ut
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Long ban tay
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

print(
    f"Tong so robust outlier trong file: "
    f"{len(df)}"
)


# ============================================================
# 6. SAP XEP THEO GEOMETRY SCORE
# ============================================================

df = df.sort_values(
    "geometry_score",
    ascending=False
).head(TOP_N).reset_index(drop=True)


print(
    f"Se ve landmark cho "
    f"{len(df)} mau geometry cao nhat."
)


# ============================================================
# 7. TAO THU MUC KET QUA
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 8. KHOI TAO MEDIAPIPE
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=str(HAND_MODEL)
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,

    # Giu dung nguong da dung luc tao dataset
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)


# ============================================================
# 9. XU LY 20 ANH
# ============================================================

with vision.HandLandmarker.create_from_options(
    options
) as landmarker:

    for candidate_rank, row in enumerate(
        df.itertuples(index=False),
        start=1
    ):

        # row.rank la thu hang robust outlier cu
        robust_rank = row.rank

        image_path = (
            PROJECT_DIR
            / Path(row.file)
        )

        print()
        print(
            f"[{candidate_rank}/{len(df)}] "
            f"Label={row.label}"
        )

        print(
            f"Geometry score="
            f"{row.geometry_score:.3f}"
        )

        print(
            f"Robust score="
            f"{row.robust_score:.3f}"
        )

        print(
            f"Robust rank={robust_rank}"
        )

        print(
            f"Anh: {image_path}"
        )


        # ====================================================
        # 10. DOC ANH
        # ====================================================

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:

            print(
                "KHONG DOC DUOC ANH."
            )

            continue


        # ====================================================
        # 11. BGR -> RGB
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ====================================================
        # 12. CHAY MEDIAPIPE
        # ====================================================

        result = landmarker.detect(
            mp_image
        )

        if not result.hand_landmarks:

            print(
                "MediaPipe khong phat hien tay "
                "o lan kiem tra nay."
            )

            continue


        hand = result.hand_landmarks[0]

        print(
            f"So landmark: {len(hand)}/21"
        )


        if len(hand) != 21:

            print(
                "KHONG DU 21 LANDMARK."
            )

            continue


        # ====================================================
        # 13. CHUYEN SANG TOA DO PIXEL
        # ====================================================

        height, width = frame.shape[:2]

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
        # 14. VE DUONG NOI
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
        # 15. VE 21 DIEM
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
        # 16. GHI THONG TIN LEN ANH
        # ====================================================

        text1 = (
            f"Candidate: {candidate_rank} | "
            f"Label: {row.label}"
        )

        text2 = (
            f"Geometry: {row.geometry_score:.2f} | "
            f"Robust: {row.robust_score:.2f}"
        )

        text3 = (
            f"Robust rank: {robust_rank} | "
            f"Landmarks: {len(hand)}/21"
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
        # 17. TEN FILE KET QUA
        # ====================================================

        output_name = (
            f"{candidate_rank:02d}"
            f"_{row.label}"
            f"_geo_{row.geometry_score:.2f}"
            f"_robust_{row.robust_score:.2f}"
            f"_{image_path.name}"
        )

        output_path = (
            OUTPUT_DIR
            / output_name
        )


        # ====================================================
        # 18. LUU ANH
        # ====================================================

        success = cv2.imwrite(
            str(output_path),
            frame
        )

        if success:

            print(
                f"Da luu: {output_name}"
            )

        else:

            print(
                "LOI KHI LUU ANH."
            )


# ============================================================
# 19. HOAN THANH
# ============================================================

print()
print("===================================")
print("HOAN THANH")
print("===================================")

print(
    "Thu muc anh kiem tra:"
)

print(
    OUTPUT_DIR
)
