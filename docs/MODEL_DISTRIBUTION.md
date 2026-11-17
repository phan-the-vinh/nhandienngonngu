# Model Distribution and Provenance Policy

This document details the official model distribution policy for the **NhanDienCuChi** open-source sign language recognition system.

- **GitHub Repository**: [https://github.com/phan-the-vinh/nhandienngonngu](https://github.com/phan-the-vinh/nhandienngonngu)
- **Dataset Documentation**: [https://github.com/phan-the-vinh/nhandienngonngu/blob/main/docs/dataset.md](https://github.com/phan-the-vinh/nhandienngonngu/blob/main/docs/dataset.md)
- **Google Drive Backup Folder**: [https://drive.google.com/drive/u/0/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10](https://drive.google.com/drive/u/0/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10)

---

## 1. Distribution Policy Overview (Batteries-Included Model Distribution)

This repository is distributed with a **Batteries-Included** runtime policy:

- **Included in Git Tree:** Complete Python source code, pre-trained model checkpoints (`.joblib` for static letters, digits, dynamic J/Z, and space gestures), MediaPipe landmark processor (`hand_landmarker.task`), feature normalization algorithms, dynamic trajectory verifiers, interaction coordinators, word/sentence completion engines, configuration manifests, test suites, and reproducible training scripts.
- **Hosted on Google Drive:** Heavy raw training datasets (source video recordings, extracted static frames, full multi-gigabyte coordinate archives, and dataset zip files) are maintained on the [Google Drive Dataset Archive](https://drive.google.com/drive/u/0/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10).

All production model weights combined comprise less than 1 Megabyte, enabling immediate, out-of-the-box execution upon cloning while keeping repository bloat minimal.

---

## 2. Model Asset Classification & Provenance Matrix

| Asset Name | Canonical File Path | Size | Upstream Origin & License | Repository & Distribution Status | Public Availability |
|---|---|---:|---|---|:---:|
| **MediaPipe Hand Landmarker** | `models/mediapipe/hand_landmarker.task` | 7.819.105 B | Google LLC (Apache License 2.0) | **Included** (Tracked in Git & downloadable via script) | **Publicly Available** (Google Cloud Storage) |
| **Word Suggestion Dictionary** | `data/dictionary/english_words_25k.tsv` | 291.528 B | SUBTLEX-US Corpus (CC BY-SA 4.0 with attribution) | **Included** (Tracked in Git) | **Publicly Available** (In Git tree) |
| **Static ASL Scaler** | `models/letters/production/uvw_medium_scaler.joblib` | 1.887 B | Fitted standard normalizer (63 coordinates) | **Included** (Tracked in Git) | **Publicly Available** (In Git tree) |
| **Static ASL Letters Model** | `models/letters/production/uvw_medium_model.joblib` | 218.456 B | 45,236 Kaggle ASL (GPL-2.0) + 41,676 color-coded images (`LICENSE_UNKNOWN`) | **Included** (Tracked in Git) | **Publicly Available** (In Git tree) |
| **Static Digits Model** | `models/digits/production/digits_final.joblib` | 464.265 B | 349 base author captures + 170 Arda Mavi (Apache 2.0) + 30 ASL-HG (CC BY 4.0) | **Included** (Tracked in Git) | **Publicly Available** (In Git tree) |
| **SPACE Gesture Model** | `models/space/production/final_space_model.joblib` | 1.375 B | 1,049 positive author captures + 63,140 negative samples (ABC & Digits) | **Included** (Tracked in Git) | **Publicly Available** (In Git tree) |
| **Dynamic J/Z Baseline Pipeline** | `data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib` | 36.657 B | SigNN ASL Video (CC0 1.0 Universal; Section 2(d) non-waiver) | **Included** (Tracked in Git) | **Publicly Available** (In Git tree) |
| **Dynamic J/Z Scaler** | `data/experiments/dynamic_jz_binary_baseline_v1/scaler.joblib` | 27.079 B | SigNN training set normalizer | **Included** (Tracked in Git) | **Publicly Available** (In Git tree) |

---

## 3. Clone & Verification Instructions

When cloning this repository, all required production models and assets are pre-installed. Running `python run.py --check-only` or `python tools/download_models.py --check-only` verifies that all components are intact and matching expected cryptographic signatures:

```text
=== NhanDienCuChi Model Asset Manager ===
[EXISTS] hand_landmarker.task: Verified matching SHA-256 at models\mediapipe\hand_landmarker.task

--- Verifying Project Production Runtime Assets ---
[OK] ASL Static Letters Model (uvw_medium) (models\letters\production\uvw_medium_model.joblib): Verified SHA-256
[OK] Static Digits Model (digits_final) (models\digits\production\digits_final.joblib): Verified SHA-256
[OK] SPACE Gesture Model (final_space_model) (models\space\production\final_space_model.joblib): Verified SHA-256
[OK] Dynamic J/Z Baseline Model (model_pipeline) (data\experiments\dynamic_jz_binary_baseline_v1\model_pipeline.joblib): Verified SHA-256
[OK] Offline English Word Suggestion Dictionary (25k words) (data\dictionary\english_words_25k.tsv): Verified SHA-256

[SUCCESS] All runtime models and assets are present and verified!
```

All production model weights, scalers, and runtime assets are pre-packaged in the repository root for immediate execution. If you need to re-download or inspect external backups:

### Option A: External Google Drive Backup
If you need to retrieve raw datasets, processed feature archives, or backup model checkpoints:
1. Navigate to: [https://drive.google.com/drive/u/0/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10](https://drive.google.com/drive/u/0/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10)
2. Download the needed `.joblib` model bundle or dataset archives.
3. Place files into their respective directories:
   - `models/letters/production/uvw_medium_model.joblib`
   - `models/letters/production/uvw_medium_scaler.joblib`
   - `models/digits/production/digits_final.joblib`
   - `models/space/production/final_space_model.joblib`
   - `data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib`
   - `data/experiments/dynamic_jz_binary_baseline_v1/scaler.joblib`
4. Run `python tools/download_models.py --check-only` to verify all SHA-256 checksums.

### Option B: Train Custom Replacement Models Using Clean Datasets
The repository includes complete scripts for training compatible models:
- **Digits (0–9)**: Execute `training/digits/` pipelines on verified datasets.
- **Letters (24 ASL Classes)**: Feature extraction and training templates in `training/letters/`.
- **Dynamic J/Z**: Trajectory training scripts in `training/dynamic/`.

---

## 4. Expected Model Input/Output Architectural Contracts

Any custom replacement model must satisfy the following architectural contracts:

- **Feature Vector**: Exactly 63 features (`f1..f63`), representing 21 normalized 3D hand landmarks produced by `ai.preprocessing.landmark_utils.normalize_landmarks`.
- **Letters Classifier (`uvw_medium`)**:
  - Classes: 24 classes: `A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y` (J and Z are dynamic).
  - Architecture: Scikit-learn MLPClassifier or Pipeline implementing `predict_proba`.
- **Digits Classifier (`digits_final`)**:
  - Classes: 10 classes: `'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'`.
  - Architecture: Scikit-learn Pipeline with steps `['orientation_2d', 'scaler', 'mlp']` implementing `predict_proba`.
- **SPACE Classifier (`final_space_model`)**:
  - Classes: Binary `[0, 1]` (0 = non-space, 1 = space).
  - Architecture: Scikit-learn LogisticRegression with wrist invariance.

---

## 5. Verification Checksums

For developers validating model artifacts against the official project manifests, the verified SHA-256 hashes are:

- **`hand_landmarker.task`**: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` (7,819,105 bytes)
- **`uvw_medium_model.joblib`**: `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` (218,456 bytes)
- **`uvw_medium_scaler.joblib`**: `93a0ddf102e456fef90cc54ddfa8cfadde60b72cce30e4f3c7810a0309fe0d4c` (1,887 bytes)
- **`digits_final.joblib`**: `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` (464,265 bytes)
- **`final_space_model.joblib`**: `453162d8aa0212016216a6c02798d3dcaa445c1fbb7a6340c896bb828e076121` (1,375 bytes)
- **`model_pipeline.joblib`** (J/Z baseline): `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497` (36,657 bytes)
- **`scaler.joblib`** (J/Z dynamic): `93804d73f4fc63aee7f2da08c1f0a879dc6d2bdc50ec27c0f5cfa4011b6fad7c` (27,079 bytes)
- **`english_words_25k.tsv`**: `986eead866a09df979d0c95a8250f235cb4bdf0bdadd5b3699db610e3ad4525e` (291,528 bytes)
