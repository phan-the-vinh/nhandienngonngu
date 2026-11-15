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

INPUT_FILE = (
    AI_DIR
    / "dataset"
    / "reject_candidates_chucai.csv"
)

HAND_MODEL = (
    AI_DIR
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)

OUTPUT_DIR = (
    AI_DIR
    / "review_reject_candidates"
)


# ============================================================
# 2. DUONG NOI 21 LANDMARK
# ============================================================

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (5, 9), (9, 10), (10, 11), (11, 12),

    (9, 13), (13, 14), (14, 15), (15, 16),

    (13, 17), (17, 18), (18, 19), (19, 20),

    (0, 17),
]


# ============================================================
# 3. DOC DANH SACH 16 REJECT CANDIDATE
# ============================================================

df = pd.read_csv(INPUT_FILE)

print(
    "So REJECT_CANDIDATE:",
    len(df)
)


# ============================================================
# 4. TAO THU MUC KET QUA
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 5. KHOI TAO MEDIAPIPE
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
# 6. VE LANDMARK CHO TUNG ANH
# ============================================================

with vision.HandLandmarker.create_from_options(
    options
) as landmarker:

    for index, row in df.reset_index(
        drop=True
    ).iterrows():

        number = index + 1

        image_path = (
            PROJECT_DIR
            / Path(row["file"])
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
                "Khong tim thay tay:",
                image_path
            )

            continue


        hand = result.hand_landmarks[0]

        height, width = frame.shape[:2]


        # ====================================================
        # 7. DOI LANDMARK SANG PIXEL
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
        # 8. VE DUONG NOI
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
        # 9. VE 21 DIEM
        # ====================================================

        for landmark_id, (x, y) in enumerate(
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
                str(landmark_id),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )


        # ====================================================
        # 10. GHI THONG TIN
        # ====================================================

        cv2.putText(
            frame,
            f"{number:02d} | Label {row['label']}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Geo {row['geometry_score']:.2f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Risk {row['image_risk_score']:.3f}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


        # ====================================================
        # 11. LUU ANH
        # ============================================================

        output_name = (
            f"{number:02d}"
            f"_{row['label']}"
            f"_geo_{row['geometry_score']:.2f}"
            f"_risk_{row['image_risk_score']:.3f}"
            f"_{image_path.name}"
        )

        cv2.imwrite(
            str(
                OUTPUT_DIR
                / output_name
            ),
            frame
        )


print()
print("===================================")
print("HOAN THANH")
print("===================================")

print(
    OUTPUT_DIR
)
