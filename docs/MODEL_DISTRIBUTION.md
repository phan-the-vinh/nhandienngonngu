# Model Distribution and Provenance Policy

This document details the model distribution policy for the **NhanDienCuChi** public open-source repository.

---

## 1. Distribution Policy Overview (Source-Only Release)

This repository is distributed as **Public Source-Only (Option B)**.

- **Included:** Complete Python source code, MediaPipe landmark processors, feature normalization algorithms, dynamic trajectory verifiers, interaction coordinators, word/sentence completion engines, configuration manifests, test suites, and reproducible training pipelines.
- **Excluded:** Unverified pre-trained model weights (`.joblib` binary artifacts) whose upstream training dataset licenses are unrecorded or pending explicit owner release declarations.

By decoupling the source code from unverified proprietary or third-party datasets, the project maintains strict open-source licensing integrity while allowing users, researchers, and developers full freedom to inspect, run, train, and validate the system.

---

## 2. Model Asset Classification & Provenance

| Asset Name | Canonical Path | Size | Upstream Origin & License | Repository Status |
|---|---|---|---|---|
| **MediaPipe Hand Landmarker** | `models/mediapipe/hand_landmarker.task` | ~7.8 MB | Google LLC (Apache 2.0) | **Included** (or via `python tools/download_models.py`) |
| **Dynamic J/Z Baseline Pipeline** | `data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib` | ~37 KB | SigNN ASL Video Dataset (CC0 1.0 Universal Public Domain) | **Included** (Tracked in Git) |
| **Word Suggestion Dictionary** | `data/dictionary/english_words_25k.tsv` | ~333 KB | SUBTLEX-US Corpus (CC BY-SA 4.0 with attribution) | **Included** (Tracked in Git) |
| **Static ASL Letters Model** | `models/letters/production/uvw_medium_model.joblib` | ~218 KB | Web image scrape dataset (License unrecorded) | **Excluded** (`.gitignore`) |
| **Static ASL Scaler** | `models/letters/production/uvw_medium_scaler.joblib` | ~1.2 KB | Derived from letter training set | **Included** (Feature normalization contract) |
| **Static Digits Model** | `models/digits/production/digits_final.joblib` | ~464 KB | Private author capture dataset | **Excluded** (`.gitignore`) |
| **SPACE Gesture Model** | `models/space/production/final_space_model.joblib` | ~1.4 KB | Private author capture dataset | **Excluded** (`.gitignore`) |
| **Experimental / Fallback Checkpoints** | `models/letters/fallback/*.joblib`, `models/digits/candidates/` | Various | Development iterations | **Excluded** (`.gitignore`) |

---

## 3. Instructions for Users on a Fresh Clone

When cloning this repository to a fresh machine without pre-trained model weights, the system provides clear, descriptive feedback:

```text
ERROR: Missing required pre-trained model file(s):
  Static ASL letter model (models/letters/production/uvw_medium_model.joblib)
  Static digit model (models/digits/production/digits_final.joblib)

This public repository is distributed source-only without unverified pre-trained weights.
Please refer to docs/MODEL_DISTRIBUTION.md for instructions.
```

To run full character and digit inference, you have two options:

### Option A: Provide Existing Pre-trained Checkpoints
If you have authorized access to pre-trained model weights (e.g., from a private backup or author release):
1. Place the letter classifier at:
   `models/letters/production/uvw_medium_model.joblib`
2. Place the digit classifier at:
   `models/digits/production/digits_final.joblib`
3. (Optional) Place the SPACE classifier at:
   `models/space/production/final_space_model.joblib`

These files match `.gitignore` and will never be accidentally committed to your public git tree.

### Option B: Train Your Own Models Using Provided Scripts
The repository includes complete scripts for training compatible models from custom landmark data:
- **Digits (0-9):**
  See `training/digits/train_digits.py` or the dataset preparation pipelines in `training/digits/`.
- **Letters (24 ASL classes):**
  See `training/` and `ai/dataset/` for feature extraction and training templates.
- **Dynamic J/Z:**
  See `training/dynamic/train_dynamic.py`.

---

## 4. Expected Model Input/Output Contracts

If you train custom replacement models, they must satisfy the following architectural contracts:

- **Feature Vector:** Exactly 63 features (`f1`..`f63`), representing 21 normalized 3D hand landmarks produced by `ai.preprocessing.landmark_utils.normalize_landmarks`.
- **Letters Classifier:**
  - Classes: 24 classes: `A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y` (J and Z are dynamic).
  - Implementation: Scikit-learn estimator or Pipeline implementing `predict_proba`.
- **Digits Classifier:**
  - Classes: 10 classes: `'0'`, `'1'`, `'2'`, `'3'`, `'4'`, `'5'`, `'6'`, `'7'`, `'8'`, `'9'`.
  - Implementation: Scikit-learn Pipeline with steps `['orientation_2d', 'scaler', 'mlp']` implementing `predict_proba`.
- **SPACE Classifier:**
  - Classes: Binary `[0, 1]` (0 = non-space, 1 = space).
  - Feature invariance: Zero or near-zero weight on wrist landmarks (features 0, 1, 2).

---

## 5. Verification Checksums

For developers working with the original private checkpoints, the expected SHA-256 hashes are:

- `uvw_medium_model.joblib`: `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45`
- `digits_final.joblib`: `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c`
- `final_space_model.joblib`: `453162d8aa0212016216a6c02798d3dcaa445c1fbb7a6340c896bb828e076121`
- `hand_landmarker.task`: `2a2fb49132512adbfb471fb6244ab742398547f877ce5d57d77b6392b45050be`
- `model_pipeline.joblib` (J/Z baseline): `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497`
