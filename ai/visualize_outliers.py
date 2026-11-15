from pathlib import Path

import cv2
import pandas as pd
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 1. DUONG DAN
# ============================================================

# Thu muc AI
AI_DIR = Path(__file__).resolve().parent

# Thu muc goc cua project: NhanDienCuChi
PROJECT_DIR = AI_DIR.parent

# Danh sach cac mau landmark nghi bat thuong
OUTLIER_CSV = (
    AI_DIR
    / "dataset"
    / "landmark_outliers_chucai.csv"
)

# Model MediaPipe da co san trong project
HAND_MODEL = (
    AI_DIR
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)

# Thu muc se chua anh co ve 21 landmark
OUTPUT_DIR = (
    AI_DIR
    / "review_outliers_landmarks"
)


# ============================================================
# 2. SO LUONG OUTLIER MUON XEM
# ============================================================

# Chi xem 20 mau co robust_score cao nhat truoc
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

    # Canh long ban tay
    (0, 17),
]


# ============================================================
# 4. KIEM TRA FILE CAN THIET
# ============================================================

if not OUTLIER_CSV.exists():
    raise FileNotFoundError(
        f"Khong tim thay:\n{OUTLIER_CSV}"
    )

if not HAND_MODEL.exists():
    raise FileNotFoundError(
        f"Khong tim thay:\n{HAND_MODEL}"
    )


# ============================================================
# 5. DOC DANH SACH OUTLIER
# ============================================================

df = pd.read_csv(OUTLIER_CSV)

# File nay da duoc sap xep score tu cao xuong thap,
# nhung ta sap xep lai de chac chan.
df = df.sort_values(
    "robust_score",
    ascending=False
)

df = df.head(TOP_N)

print(
    f"Se kiem tra {len(df)} "
    f"outlier co score cao nhat."
)


# ============================================================
# 6. TAO THU MUC KET QUA
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 7. KHOI TAO MEDIAPIPE
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=str(HAND_MODEL)
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,

    # Giu dung nguong giong luc extract dataset
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)


# ============================================================
# 8. DOC TUNG ANH OUTLIER
# ============================================================

with vision.HandLandmarker.create_from_options(
    options
) as landmarker:

    for rank, row in enumerate(
        df.itertuples(index=False),
        start=1
    ):

        # row.file co dang:
        # dulieu/chucai/x/X1972.jpg
        image_path = (
            PROJECT_DIR
            / Path(row.file)
        )

        print()
        print(
            f"[{rank}/{len(df)}] "
            f"{row.label} - "
            f"score={row.robust_score:.2f}"
        )

        print(
            f"Anh: {image_path}"
        )


        # ====================================================
        # 9. DOC ANH GOC
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
        # 10. BGR -> RGB
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
        # 11. MEDIAPIPE PHAT HIEN BAN TAY
        # ====================================================

        result = landmarker.detect(
            mp_image
        )


        if not result.hand_landmarks:

            print(
                "Lan kiem tra nay MediaPipe "
                "khong phat hien duoc tay."
            )

            continue


        hand = result.hand_landmarks[0]


        # ====================================================
        # 12. KIEM TRA SO LANDMARK
        # ====================================================

        print(
            f"So landmark: {len(hand)}/21"
        )

        if len(hand) != 21:

            print(
                "KHONG DU 21 LANDMARK."
            )

            continue


        # ====================================================
        # 13. CHUYEN TOA DO NORMALIZED -> PIXEL
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
        # 14. VE CAC DUONG NOI
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
        # 15. VE 21 DIEM + SO THU TU
        # ====================================================

        for index, point in enumerate(
            points
        ):

            x, y = point

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
            f"Label: {row.label} | "
            f"{row.handedness}"
        )

        text2 = (
            f"Landmarks: {len(hand)}/21 | "
            f"Score: {row.robust_score:.2f}"
        )

        cv2.putText(
            frame,
            text1,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            text2,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # 17. DAT TEN FILE KET QUA
        # ====================================================

        output_name = (
            f"{rank:02d}"
            f"_{row.label}"
            f"_{row.handedness}"
            f"_score_{row.robust_score:.2f}"
            f"_{image_path.name}"
        )

        output_path = (
            OUTPUT_DIR
            / output_name
        )


        # ====================================================
        # 18. LUU ANH
        # ====================================================

        cv2.imwrite(
            str(output_path),
            frame
        )

        print(
            f"Da luu: {output_name}"
        )


# ============================================================
# 19. HOAN THANH
# ============================================================

print()
print("===================================")
print("HOAN THANH")
print("===================================")

print(
    "Anh co ve landmark nam tai:"
)

print(
    OUTPUT_DIR
)
