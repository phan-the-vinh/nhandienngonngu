from pathlib import Path

import cv2
import pandas as pd
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


AI_DIR = Path(__file__).resolve().parent
PROJECT_DIR = AI_DIR.parent

INPUT_CSV = (
    AI_DIR
    / "dataset"
    / "image_quality_outliers.csv"
)

HAND_MODEL = (
    AI_DIR
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)

OUTPUT_DIR = (
    AI_DIR
    / "review_image_risk"
)

TOP_N = 20


CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (5, 9), (9, 10), (10, 11), (11, 12),

    (9, 13), (13, 14), (14, 15), (15, 16),

    (13, 17), (17, 18), (18, 19), (19, 20),

    (0, 17),
]


df = pd.read_csv(INPUT_CSV)

df = (
    df
    .sort_values(
        "image_risk_score",
        ascending=False
    )
    .head(TOP_N)
    .reset_index(drop=True)
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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


with vision.HandLandmarker.create_from_options(
    options
) as landmarker:

    for rank, row in df.iterrows():

        number = rank + 1

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


        for start, end in CONNECTIONS:

            cv2.line(
                frame,
                points[start],
                points[end],
                (255, 255, 255),
                2
            )


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


        cv2.putText(
            frame,
            f"Risk {number:02d} | Label {row['label']}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Blur {row['blur_hand']:.1f} | Area {row['hand_area']:.4f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Geo {row['geometry_score']:.2f} | Risk {row['image_risk_score']:.3f}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


        output_name = (
            f"{number:02d}"
            f"_{row['label']}"
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
print("HOAN THANH")
print(OUTPUT_DIR)
