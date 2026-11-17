# Dataset Documentation & Provenance Guide

This document provides the authoritative technical reference for all training, validation, and testing datasets across the four recognition branches of the **NhanDienCuChi** sign language recognition system.

- **GitHub Repository**: [https://github.com/phan-the-vinh/nhandienngonngu](https://github.com/phan-the-vinh/nhandienngonngu)
- **Dataset Documentation Link**: [https://github.com/phan-the-vinh/nhandienngonngu/blob/main/docs/dataset.md](https://github.com/phan-the-vinh/nhandienngonngu/blob/main/docs/dataset.md)
- **Google Drive Dataset Archive**: [https://drive.google.com/drive/u/0/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10](https://drive.google.com/drive/u/0/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10)

---

## 1. Feature Representation & Normalization Contract

All recognition models operate on 3D hand landmark coordinates extracted via Google MediaPipe Hand Landmarker (`models/mediapipe/hand_landmarker.task`):

- **Keypoints**: 21 3D hand landmarks ($x, y, z$).
- **Raw Dimension**: 63 raw float values ($21 \times 3$).
- **Normalization Pipeline** (`ai/preprocessing/landmark_utils.py`):
  1. **Wrist-Relative Centering**: Landmark 0 (wrist) coordinates are subtracted from all 21 points:
     $$p'_i = p_i - p_{\text{wrist}}$$
     Ensures translation invariance regardless of hand position in the webcam view.
  2. **Max-Distance Scaling**: All coordinates are scaled by the maximum Euclidean distance from the wrist to any landmark:
     $$p''_i = \frac{p'_i}{\max_{j} \|p'_j\|_2}$$
     Ensures scale invariance regardless of hand-to-camera distance.
  3. **Feature Naming**: Flattened into 63 continuous float features `f1, f2, ..., f63`.
  4. **Dynamic J/Z Trajectory**: 16 consecutive frames normalized to $[16, 69]$ (63 landmark features + 3 wrist displacement features + 3 motion velocity features).

---

## 2. Multi-Branch Dataset Breakdown

### 2.1. Static ASL Letters (Branch ABC)
Recognizes 24 static alphabet letters (`A–I`, `K–Y`). The letters `J` and `Z` are dynamic trajectories handled by the J/Z branch.

- **Total Clean Corpus (Frozen v3)**: **89,415 samples** ([final_freeze_manifest.json](file:///D:/NhanDienCuChi/ai/dataset/preprocessing_frozen_v3/final_freeze_manifest.json)).
- **Canonical Split v1 (70/15/15)**:
  - **Train**: **62,591 samples** (`ai/dataset/train_validation_test_split_v1/train.csv`).
  - **Validation**: **13,412 samples** (`ai/dataset/train_validation_test_split_v1/validation.csv`).
  - **Test (Hold-out)**: **13,412 samples** (`ai/dataset/train_validation_test_split_v1/test.csv`).
- **Production Model Training Set (`uvw_medium`)**:
  - **Total**: **62,648 samples** ([manifest.json](file:///D:/NhanDienCuChi/ai/dataset/mlp_uvw_training_data_v1/manifest.json)).
  - **Composition**: 62,591 base train samples + **57 webcam supplement samples** (`additional_uv_samples.csv`: 15 OLD_U, 15 OLD_V, 15 V_TO_W, 12 V_TO_U) to eliminate confusion between letters U, V, and W in live webcam video.
- **Upstream Data Sources**:
  - Kaggle ASL Alphabet (Akash Nagaraj, GPL-2.0): 45,236 samples ($50.59\%$).
  - Color-coded Hand Segmentation (`LICENSE_UNKNOWN`): 41,676 samples ($46.61\%$).
  - Web & auxiliary hand segmentation (`LICENSE_UNKNOWN`): 2,503 samples ($2.80\%$).

---

### 2.2. Static Digits (Branch 123)
Recognizes 10 static digit gestures (`0, 1, 2, 3, 4, 5, 6, 7, 8, 9`).

- **Total Digit Universe**: **899 samples** ([training_manifest_digits_final.json](file:///D:/NhanDienCuChi/data/metadata/digits/digits_final_repro/training_manifest_digits_final.json)).
- **Official Training Set (`digits_final`)**: **549 samples** (`data/processed/digits/digits_final_train_frozen.csv`).
  - **Internal Base Captures (Author)**: 349 samples (34 samples for digit 0; 35 samples each for digits 1–9).
  - **Arda Mavi Digits (Apache 2.0)**: 170 samples (20 samples for digits 0, 1, 2, 3, 5, 7, 9; 10 samples for digits 4, 6, 8).
  - **ASL-HG Mendeley (CC BY 4.0)**: 30 samples (10 samples each for digits 4, 6, 8 across 10 volunteer signers).
- **Validation Reference Set**: **100 samples** (`data/splits/digits/validation_reference.csv`, 10 balanced samples per class).
- **Regression Reference Set**: **250 samples** (`data/splits/digits/regression_reference.csv`, 25 balanced samples per class from `test_v1.csv`).

---

### 2.3. Dynamic Letters (Branch J/Z)
Recognizes continuous spatial-temporal trajectories for letters `J` (pinky hook) and `Z` (index zig-zag).

- **Original Source Videos**: 643 AVI video clips (316 J + 327 Z) from SigNN ASL Video Dataset (CC0 1.0 Universal).
- **Quality Filtering**: 3 corrupted/undetected video clips excluded (`J/99.avi`, `J/103.avi`, `J/250.avi`).
- **Valid Sequences**: **640 sequences** (265 J sequences + 375 Z sequences).
- **Canonical Split (80/10/10)** ([split_manifest.json](file:///D:/NhanDienCuChi/data/experiments/dynamic_jz_binary_baseline_v1/split_manifest.json)):
  - **Train**: **512 sequences** (80%).
  - **Validation**: **64 sequences** (10%).
  - **Test**: **64 sequences** (10%).
- **Feature Extraction**: First16Segmenter extracts 16 standardized time-step frames per sequence with motion speed gating.

---

### 2.4. SPACE Gesture (Branch SPACE)
Binary classification model (`final_space_model.joblib`) distinguishing open-palm space gestures from non-space hand poses.

- **Total Universe**: **91,813 samples** ([dataset_integrity_audit.csv](file:///D:/NhanDienCuChi/staging/space_training_readiness_audit_phase3_9_4_20261106/dataset_integrity_audit.csv)).
- **Training Set**: **64,189 samples** (`space_model_metadata.json`):
  - **Positive Class (`SPACE = 1`)**: **1,049 samples** ($1.63\%$) collected by the project author.
  - **Negative Class (`NOT_SPACE = 0`)**: **63,140 samples** ($98.37\%$), composed of:
    - 62,591 static letter samples from ABC base `train.csv`.
    - 549 digit samples from 123 `digits_final_train_frozen.csv`.
- **Validation Set**: **225 positive** / **13,512 negative** (13,412 ABC + 100 Digits).
- **Test Set**: **225 positive** / **13,662 negative** (13,412 ABC + 250 Digits).
- **Total Positive Captures**: **1,499 samples** ($1,049 \text{ train} + 225 \text{ val} + 225 \text{ test}$).

---

## 3. Summary Dataset Matrix

| Phân hệ | Lớp nhận diện | Tổng mẫu sạch | Tập Train | Tập Validation | Tập Test | Nguồn cấp phép chính |
|---|---|---:|---:|---:|---:|---|
| **ABC Letters** | 24 (`A–I, K–Y`) | 89.415 | 62.591 (base)<br>62.648 (`uvw_medium`) | 13.412 | 13.412 | Kaggle GPL-2.0 / Color-coded (`LICENSE_UNKNOWN`) |
| **123 Digits** | 10 (`0–9`) | 899 | 549 | 100 (ref) | 250 (ref) | Tự thu / Arda Mavi (Apache 2.0) / ASL-HG (CC BY 4.0) |
| **J/Z Dynamic** | 2 (`J, Z`) | 640 videos | 512 | 64 | 64 | SigNN ASL Video (CC0 1.0 Universal) |
| **SPACE Gesture** | 2 (`SPACE / NOT_SPACE`) | 91.813 | 64.189 (1.049+ / 63.140-) | 13.737 (225+ / 13.512-) | 13.887 (225+ / 13.662-) | Tự thu (+) / Lớp âm trích từ ABC & 123 (-) |

---

## 4. Google Drive Download & Installation Instructions

All extracted landmark feature CSV files and frozen dataset partitions are available for download from the project's dedicated Google Drive repository:

🔗 **Google Drive Folder**: [https://drive.google.com/drive/u/0/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10](https://drive.google.com/drive/u/0/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10)

### Archive Directory Layout:
When downloaded, extract the archive contents into the project directory structure as follows:

```
D:\NhanDienCuChi\
├── ai\dataset\
│   ├── train_validation_test_split_v1\
│   │   ├── train.csv                      # ABC 62,591 samples
│   │   ├── validation.csv                 # ABC 13,412 samples
│   │   ├── test.csv                       # ABC 13,412 samples
│   │   └── split_summary.json
│   └── mlp_uvw_training_data_v1\
│       ├── train_uvw_v1.csv               # ABC 62,648 samples
│       └── additional_uv_samples.csv      # 57 U/V/W supplement samples
├── data\
│   ├── processed\digits\
│   │   └── digits_final_train_frozen.csv  # Digits 549 samples
│   ├── splits\digits\
│   │   ├── validation_reference.csv       # Digits 100 samples
│   │   └── regression_reference.csv       # Digits 250 samples
│   └── experiments\dynamic_jz_binary_baseline_v1\
│       ├── split_manifest.json            # J/Z 512/64/64 split
│       └── training_manifest.json
└── models\
    ├── letters\production\uvw_medium_model.joblib
    ├── digits\production\digits_final.joblib
    └── space\production\final_space_model.joblib
```

---

## 5. Ethical Considerations & Privacy Notice

- **Publicity & Privacy Rights**: In accordance with Section 2(d) of the Creative Commons CC0 1.0 Universal license, the video recordings for the J/Z branch originate from volunteer signers whose publicity rights are reserved. Training pipelines use exclusively geometric wrist-relative landmarks without facial or identity markers.
- **Compliance**: The raw image sets for the ABC static letters contain historical web scrapes with unrecorded licenses. The project implements a **Source-Only** distribution policy, providing full preprocessing and model architecture code while excluding unverified binary image assets.
