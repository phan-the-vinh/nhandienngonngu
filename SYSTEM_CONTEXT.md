# SYSTEM_CONTEXT — NhanDienCuChi

> Context document for AI assistants. Originally generated from a read-only survey (2026-10-02);
> updated after the folder-structure refactor and the `AI` → `ai` folder rename (same date).
> Root: `<project-root>`

---

## 0. Project layout after the structural refactor

The project was reorganized into `app/`, `ai/`, `models/`, `training/`, `tools/`, `tests/`,
`configs/`, `docs/`, `reports/`, `debug/`, `data/`. **No algorithm, model, or dataset changed.**

```
run.py                          -> main entry point (dispatches to app/camera/realtime_inference.py)
app/camera/realtime_inference.py -> dispatcher (letters | digits) via runpy
app/inference/letter_inference.py -> was AI/static_asl_camera.py (24 ASL letters, mirrored by default)
app/inference/digit_inference.py  -> was AI/static_digit_camera.py (digits 0-9, NOT mirrored)
app/inference/word_builder.py     -> placeholder only (word composition is not implemented)
ai/preprocessing/landmark_utils.py      -> was AI/landmark_utils.py (single normalize_landmarks)
ai/feature_extraction/hand_landmarks.py -> new MediaPipe helper (not yet wired into frozen cameras)
ai/common/constants.py                  -> new shared constants
models/mediapipe/hand_landmarker.task   -> was AI/models/mediapipe/hand_landmarker.task
models/letters/{production,fallback}    -> was AI/models/letters_static/{production,fallback}
models/digits/{production,candidates}   -> was AI/models/kytuso/{production,candidates}
configs/letters.json                    -> was AI/static_asl_model_config.json (paths updated)
configs/digits.json, configs/camera.json -> new documentation configs
tests/*.py                              -> was AI/test/*.py
training/letters/*, training/digits/*   -> was AI/train_*.py, AI/prepare_*.py, AI/evaluate_*.py
tools/extract_letters.py                -> was AI/extract_chucai.py
tools/extract_digits.py                 -> was AI/extract_chuso.py
reports/evaluation/{REFACTOR_REPORT,PATH_CHANGE_PLAN}.md -> moved from project root
docs/{architecture,dataset,model}.md    -> new, alongside the pre-existing docs
```

### `ai/` package layout

The legacy `AI` folder has been renamed to lowercase **`ai`**. It holds the `ai` package code
(`preprocessing/`, `feature_extraction/`, `common/`) together with the **un-moved data/tooling**
(`dataset/`, `raw_data/`, `camera_debug_*`, and the remaining audit/analysis scripts).

Runtime and tooling import `landmark_utils` / `constants` **by path** (adding `<root>/ai/preprocessing`
to `sys.path`) rather than via `import ai`. This keeps the modules portable and independent of how the
package directory is cased or of `import ai` succeeding.

A small number of **historical strings intentionally keep the `AI/...` form** because they are matched
against frozen snapshot/manifest content or are JSON keys: `normalize_landmarks` contract strings,
the `"AI/models/hand_landmarker.task"` hash key inside `training_config.json`, and the snapshot-key
prefixes in `ai/cleanup_snapshot_contract.py`. Do not rewrite those.

### Not moved (with reasons)

- `dulieu/` — protected source images; must not be moved (explicit constraint).
- `ai/dataset/`, `ai/raw_data/`, `ai/camera_debug_*`, review/audit trees — large and/or referenced
  by historical manifests, hashes, and audit scripts; a dedicated audited pass is required.
- `ai/final_landmark_dispositions.json`, `ai/preprocessing_semantic_observations.json`,
  `ai/PREPROCESSING_FREEZE_REVIEW.md` — still read by `finalize_static_dataset.py`,
  `finalize_preprocessing_minor_fixes.py`, and `tests/test_preprocessing_freeze_review.py`.
- Audit/analysis scripts (`audit_*.py`, `analyze_*.py`, `visualize_*.py`, `phase2_*`, `finalize_*`,
  `rescue_*`, `cleanup_snapshot_contract.py`, `collect_data.py`, `verify_static_asl_no_camera.py`) —
  they embed historical path/hash/self-signature contracts; relocating them safely needs its own pass.
- `ai/models/hand_landmarker.task` (legacy copy) — kept in place; its path is a hash key inside
  `training_config.json`, so moving it would break a recorded contract.
- `ai/dataset/deep_cleanup_snapshot_v1/` and `ai/_cleanup_quarantine_v1/` — frozen snapshots, read-only.
- `deployment_manifest_v3.json` / `deployment_manifest_v1.json` — historical evidence records whose
  embedded `AI/...` paths are intentionally frozen; the runtime does not read them.

---

## 1. What this project is

A hand-gesture recognition project with **two independent, standalone runtime programs**:

| Program | Recognizes | Entry point |
|---|---|---|
| Static ASL letters | 24 static classes `A–I, K–Y` (J and Z excluded — dynamic) | `app/inference/letter_inference.py` (via `run.py --mode letters`) |
| Static digits | `0–9` | `app/inference/digit_inference.py` (via `run.py --mode digits`) |

Both share the **same feature contract**: MediaPipe Hand Landmarker → 21 landmarks → wrist-relative,
scale-normalized → **63 features (`f1..f63`)** → scikit-learn `MLPClassifier`.

There is **no server / API / UI layer**. `backend/` and `database/` are empty.
There is **no dynamic-gesture runtime**; `dulieu/chuyendong/` data is unused by runtime code.

---

## 2. The shared feature path (canonical)

**File:** `ai/preprocessing/landmark_utils.py` → `normalize_landmarks(hand_landmarks) -> list[63]`

Algorithm:
1. `wrist = hand_landmarks[0]` (MediaPipe index 0 = wrist).
2. Each landmark → relative vector `(x - wrist.x, y - wrist.y, z - wrist.z)`.
3. `max_distance` = maximum Euclidean distance from wrist across all 21 relative points.
4. If `max_distance == 0` → set to `1.0` (guard).
5. Each relative `(x, y, z)` divided by `max_distance`, then flattened in landmark order → 63 floats.

Notes:
- This is **scale-normalized, wrist-relative**, but **not** rotation-normalized and **not** mirror-corrected.
- It has no notion of handedness. Feeding mirrored vs non-mirrored landmarks yields different features.
- This function is the single source of truth for normalization and must not be duplicated or altered
  without re-verifying every consumer below.

**Consumers of `normalize_landmarks`:**
- `app/inference/letter_inference.py` (via `features_from_landmarks()`)
- `app/inference/digit_inference.py` (imported directly)
- `tools/extract_letters.py`, `tools/extract_digits.py` (dataset creation)
- `training/letters/prepare_mlp_v2_training_data.py`, `training/letters/prepare_uvw_training_data.py`
- audit/analysis scripts still under `ai/` (`audit_*`, `analyze_*`, `rescue_*`, `run_landmark_quality_pipeline.py`)
- tests (`tests/test_landmarks.py`, `tests/test_static_asl_camera.py`, `tests/test_static_mirror.py`)

---

## 3. MediaPipe Hand Landmarker

- Library: `mediapipe.tasks.python.vision.HandLandmarker`.
- Runtime task asset for both cameras + both extractors: **`models/mediapipe/hand_landmarker.task`**
  (SHA-256 `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`).
- Legacy duplicate kept for old scripts / historical hashes: `ai/models/hand_landmarker.task` (byte-identical).

Config differences:

| Consumer | RunningMode | Detection call | num_hands |
|---|---|---|---|
| `app/inference/letter_inference.py` (was `static_asl_camera.py`) | `VIDEO` | `detect_for_video(mp_image, timestamp_ms)` | 1 |
| `app/inference/digit_inference.py` (was `static_digit_camera.py`) | `IMAGE` | `detect(mp_image)` | 1 |
| `tools/extract_letters.py`, `tools/extract_digits.py` | `IMAGE` | `detect(mp_image)` | 1 |

Both cameras convert BGR→RGB with `cv2.COLOR_BGR2RGB` before building `mp.Image` (SRGB).

---

## 4. Letter pipeline

### 4.1 Runtime
**Entry:** `app/inference/letter_inference.py` (CLI, argparse, hash-verified artifact loading; launched by `run.py --mode letters`).

Key structure:
- `_validate_deployment_config()` — validates `configs/letters.json` against hardcoded expectations.
- `load_official_artifacts(model_name)` → dispatches to `_load_v1_artifacts()` or `_load_uvw_medium_artifacts()`.
- `features_from_landmarks(hand)` — 21-landmark validation + calls `normalize_landmarks` → 63 features.
- `StaticAslInference.predict_features()` — **`scaler.transform(1×63)` → `model.predict_proba()` → argmax** over 24 classes.
- `TemporalSmoother` — display-only vote smoother (default: 4 votes in a 7-frame window). Never changes the model.
- `run_camera()` — mirrors frame by default (`--no-mirror` disables), runs `detect_for_video`, draws overlay, `Q`/`Esc` to quit.
- `--check-only` loads + verifies artifacts and prints JSON **without opening the camera**.

CLI:
```
python run.py --mode letters --check-only
python run.py --mode letters --camera-index 0
python run.py --mode letters --model v1            # fallback
python run.py --mode letters --no-mirror
python run.py --mode letters --debug-dir debug\letters --debug-label P --debug-view front
# or run the module directly:
python app\inference\letter_inference.py --check-only
```
Debug capture (`S` key) writes raw PNG + indexed overlay + JSON + CSV row under `<debug-dir>/<LABEL>_<view>/`.
Allowed labels: `P, Q, U, V`. Views: `front, horizontal`.
Optional `--debug-prediction-filter <label>` rejects saves unless the instant prediction matches.

### 4.2 Production artifacts

| Role | Path |
|---|---|
| **Active/default model** (`uvw_medium`, epoch 81) | `models/letters/production/uvw_medium_model.joblib` |
| Active scaler | `models/letters/production/uvw_medium_scaler.joblib` |
| Label mapping (24 classes) | `models/letters/production/label_mapping.json` |
| Selection / training / evaluation metadata | `models/letters/production/{selected_candidate,training_manifest,training_config,evaluation_summary}.json` |
| **Fallback model** (`v1_baseline`, epoch 143) | `models/letters/fallback/selected_model.joblib` |
| Fallback scaler / mapping / metadata | `models/letters/fallback/{scaler.joblib,label_mapping.json,baseline_model.joblib,selected_model.json,training_manifest.json,training_config.json,evaluation_manifest.json}` |
| Deployment config (ACTIVE) | `configs/letters.json` |
| Active deployment manifest v3 (historical evidence; contains pre-refactor paths) | `models/letters/production/deployment_manifest_v3.json` |

Loader integrity checks (uvw_medium / v1):
- Manifest status strings, selection flags, feature order `f1..f63`, 24-label list.
- **SHA-256 verification** of model, scaler, label mapping, and MediaPipe task.
- `scaler.n_features_in_ == 63`, expected `n_samples_seen_` (62648 uvw / 62591 v1).
- `model.n_features_in_ == 63`, `model.classes_ == np.arange(24)`.
- `uvw_medium.best_epoch == 81`; v1 contract uses epoch 143.

**Model + scaler are two separate joblib objects here** — scaling is applied manually via `scaler.transform`.

Labels (fixed order): `A B C D E F G H I K L M N O P Q R S T U V W X Y` (no J, no Z).

### 4.3 Training lineage (not runtime)
`dulieu/chucai/` → `tools/extract_letters.py` → `ai/dataset/dataset_chucai_all.csv`
→ `training/letters/{prepare_mlp_v2_training_data,prepare_uvw_training_data,finalize_static_dataset}.py`
→ `training/letters/create_train_validation_test_split.py` → `training/letters/{train_mlp_v2_validation,train_uvw_validation,train_compare_mlp_validation}.py`
→ selection metadata → copied into `models/letters/{production,fallback}/`.
`training/letters/train_mlp_mirror.py` is an experimental mirror-training script (not production).

MLP hyperparameters (letters, `train_mlp_v2_validation.py`): `hidden_layer_sizes=[64,32]`, `alpha=1e-4`,
`learning_rate_init=1e-3`, `batch_size=…`, `max_iter=1` per epoch with **external** epoch selection
(sklearn internal `early_stopping=False`, `tol=0.0`).

---

## 5. Digit pipeline

### 5.1 Runtime
**Entry:** `app/inference/digit_inference.py` — a flat top-level script (no argparse, no `main()`, no functions; launched by `run.py --mode digits`).

Flow:
1. Load `models/digits/production/mlp_large_v2a.joblib` **at import time** (`joblib.load`) → print classes & `n_features_in_`.
2. Create `HandLandmarker` (`RunningMode.IMAGE`, `num_hands=1`, det/presence conf 0.5).
3. Open `cv2.VideoCapture(0)`.
4. Loop: BGR→RGB → `mp.Image` → `landmarker.detect()` → if 21 landmarks →
   `normalize_landmarks(hand)` → `np.asarray(..., float).reshape(1, 63)` → `model.predict(X)` and
   `model.predict_proba(X)` → draw points + text.
5. `Q` quits.

Important characteristics:
- **The frame is NOT flipped/mirrored** before MediaPipe (this is the configuration the user confirms works well).
- **No separate scaler**: `mlp_large_v2a.joblib` is a single sklearn **`Pipeline([StandardScaler, MLPClassifier])`**
  (confirmed: joblib header contains `sklearn.pipeline.Pipeline`). Scaling happens inside `predict`.
- **No hash verification, no manifest validation, no `--check-only`, camera index hardcoded to 0.**
- Loads the model and opens the camera as an import side effect (no guard beyond `__main__`-less top-level code).

### 5.2 Production artifacts

| Role | Path |
|---|---|
| Production model (0–9, pipeline w/ scaler) | `models/digits/production/mlp_large_v2a.joblib` |
| Deployment manifest v1 (historical evidence) | `models/digits/production/deployment_manifest_v1.json` |
| Candidate artifacts + comparison reports | `models/digits/candidates/` |

Manifest facts: `classes = ["0".."9"]`, `feature_count = 63`, `feature_columns = f1..f63`,
`pipeline = ["StandardScaler", "MLPClassifier"]`, dataset `ai/dataset/kytuso/datasetsach/dataset_chuso_clean_v1.csv`
(699 samples), split by `hand_id`, final test accuracy 0.868 / macro-F1 0.863.

### 5.3 Training lineage (not runtime)
`dulieu/chuso/` → `tools/extract_digits.py` (JPEG only; PNG excluded to avoid duplicates) →
`ai/dataset/kytuso/landmark/dataset_chuso_all.csv` → `training/digits/{merge_chuso_clean_v1,freeze_chuso_dataset_v1}.py`
→ `ai/dataset/kytuso/datasetsach/dataset_chuso_clean_v1.csv` → `training/digits/split_chuso_dataset_v1.py`
(group split by `hand_id`: train = hand1/3/4, validation = hand5, test = rest)
→ `training/digits/train_chuso_mlp_candidates_v{1,2}.py` → `training/digits/evaluate_chuso_validation_v{1,2}.py`
→ promotion to `models/digits/production/`.

`training/digits/train_chuso_mlp_candidates_v2.py` builds `Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier(...))])`
and `joblib.dump`s the whole pipeline — this is why the runtime needs no scaler file.

---

## 6. File classification

### Runtime (production paths)
- `run.py` — main entry point
- `app/camera/realtime_inference.py` — mode dispatcher (letters | digits)
- `app/inference/letter_inference.py` — letters camera entry
- `app/inference/digit_inference.py` — digits camera entry
- `ai/preprocessing/landmark_utils.py` — shared normalization (runtime dependency of both)
- `configs/letters.json` — letters deployment config
- `models/letters/production/*`, `models/letters/fallback/*`
- `models/digits/production/*`
- `models/mediapipe/hand_landmarker.task`

### Testing / verification
- `tests/*.py` (83 unit tests; `test_static_asl_camera.py` mocks `VideoCapture`)
- `ai/verify_static_asl_no_camera.py` — runs verification while **forbidding** camera access

### Training only
- `tools/extract_letters.py`, `tools/extract_digits.py` (extraction)
- `training/letters/{prepare_mlp_v2_training_data,prepare_uvw_training_data,finalize_static_dataset}.py`
- `training/letters/{create_train_validation_test_split,train_mlp_mirror}.py`
- `training/letters/{train_mlp_v2_validation,train_uvw_validation,train_compare_mlp_validation}.py`
- `training/digits/{merge_chuso_clean_v1,freeze_chuso_dataset_v1,split_chuso_dataset_v1}.py`
- `training/digits/{train_chuso_mlp_candidates_v1,train_chuso_mlp_candidates_v2}.py`
- `training/digits/{evaluate_chuso_validation_v1,evaluate_chuso_validation_v2}.py`
- `ai/collect_data.py` (legacy live collector; reads `ai/config/ky_tu.txt`, 26 labels; not yet relocated)

### Dataset processing / audit / analysis only (still under `ai/`, not relocated)
- `ai/analyze_*.py`, `ai/audit_*.py`, `ai/visualize_*.py`
- `ai/detect_outliers_chucai.py`, `ai/count_reject_candidates.py`
- `ai/rescue_chuso_no_hand.py`, `ai/rescue_chuso_padding.py`
- `ai/run_landmark_quality_pipeline.py`, `ai/regenerate_phase2_quality_gate.py`
- `ai/phase2_review_promotion.py`, `ai/finalize_pq_debug_audit.py`, `ai/finalize_preprocessing_minor_fixes.py`
- `ai/cleanup_snapshot_contract.py`, `ai/sample_geometry_levels.py`

### Evidence / data (do not treat as runtime)
- `ai/camera_debug_*`, `ai/review_*`, `ai/quality_checkpoints/`, `ai/phase2_outputs/`, `ai/raw_data/`,
  `ai/final_promotion_audit/`, `ai/_cleanup_quarantine_v1/`, `ai/dataset/`
- `dulieu/{chucai,chuso,chuyendong}/` — raw images (large, immutable source data)
- `ai/dataset/deep_cleanup_snapshot_v1/payload/AI/…` — **frozen snapshot copies** of older source files.
  Never edit these; they are historical records, not active code.

---

## 7. Dependency graph

```
                    ai/preprocessing/landmark_utils.py::normalize_landmarks
                                   |
              +--------------------+--------------------+
              |                                         |
   app/inference/letter_inference.py       app/inference/digit_inference.py
              |                                         |
   configs/letters.json                     models/digits/production/
   models/letters/production/                  mlp_large_v2a.joblib
     uvw_medium_{model,scaler}.joblib           (Pipeline: scaler+MLP)
   models/letters/fallback/
     selected_model.joblib + scaler.joblib
              |                                         |
              +-------------> models/mediapipe/hand_landmarker.task <----------+
                                     ^
                    run.py -> app/camera/realtime_inference.py (dispatch)
                    tools/extract_letters.py, tools/extract_digits.py
```

- Both runtime entries depend on `ai/preprocessing/landmark_utils.py` and a MediaPipe `.task` file.
- Letters runtime additionally depends on `configs/letters.json` and the two model dirs under
  `models/letters/`; missing/mismatched hashes cause a hard `IntegrationError`.
- Digits runtime depends only on its one joblib pipeline (no config, no hash gate).
- The dispatcher launches both programs with `runpy.run_path(..., run_name="__main__")` instead of
  importing them, so neither camera opens as an import side effect.
- `landmark_utils` is imported by path (`<root>/ai/preprocessing` on `sys.path`) so the modules do not
  depend on `import ai` succeeding (see section 0).

---

## 8. Known structure / path risks

1. **Mirroring inconsistency.** `app/inference/letter_inference.py` mirrors the frame by default before
   MediaPipe; `app/inference/digit_inference.py` does not mirror at all. Extraction
   (`tools/extract_letters.py`, `tools/extract_digits.py`) used non-mirrored images. There is **no
   landmark-mirror augmentation**. Do not add mirroring, augmentation, or preprocessing changes without
   explicit approval.
2. **No J/Z runtime.** 24 classes only. `chuyendong/` (dynamic) is unused; there is no dynamic pipeline.
3. **Duplicated MediaPipe task files** (`ai/models/hand_landmarker.task` legacy vs
   `models/mediapipe/hand_landmarker.task` runtime). The legacy path is a hash key inside
   `training_config.json` and is referenced by legacy scripts, so it stays in place.
4. **Two different model-packaging conventions:** letters = separate scaler + model with SHA-256 gating;
   digits = single sklearn Pipeline with no gating. Any refactor must respect both.
5. **Digit camera has no CLI guard**: model load, landmarker creation, and camera open all happen at
   module top level. Importing `app/inference/digit_inference.py` has camera side effects; the
   dispatcher therefore uses `runpy` rather than importing it.
6. **Docs drift:** `docs/static_asl_camera.md` still describes the baseline epoch-143 model as the camera
   model, while the active default is `uvw_medium` epoch 81. Trust `configs/letters.json` and the runtime
   `--check-only` output over that doc.
7. **`backend/` and `database/` are empty** — no integration surface exists yet.
8. **Frozen snapshot tree** `ai/dataset/deep_cleanup_snapshot_v1/payload/…` contains stale copies of
   `static_asl_camera.py`, training scripts, and tests. Never mistake those for live code.
9. **Historical path contracts**: several manifests deliberately keep old artifact paths as evidence.
   Their presence does not mean the files are read at runtime.
10. Pre-existing known issue: `preprocessing_frozen_v3/README.md` hash differs from the value recorded in
    `final_freeze_manifest.json`. Unresolved, out of scope for runtime.
11. **Data still lives under the `ai/` folder.** The target `data/` tree is still empty; `ai/dataset`,
    `ai/raw_data`, and the debug/review trees must move only after the dedicated data audit.
12. **Historical contract strings** such as `"AI/landmark_utils.py::normalize_landmarks"` are stored in
    existing datasets/manifests and validated on read; they must NOT be rewritten to the new path.
13. **Audit/analysis scripts still under `ai/`** reference historical paths and self-signatures; they were
    intentionally left in place. Moving them needs its own reference/hash audit.

---

## 9. Quick commands

```powershell
# Letters — verify artifacts only (no camera)
python run.py --mode letters --check-only

# Letters — live camera (default uvw_medium)
python run.py --mode letters --camera-index 0

# Letters — fallback model
python run.py --mode letters --model v1

# Digits — live camera (no frame flip; Q to quit)
python run.py --mode digits

# Unit tests
python -m unittest discover -s tests -q
```

Environment note: docs reference a conda env named `nn`.

---

## 10. Open questions / needs confirmation

1. Should the digit pipeline get the same hash-verified artifact/manifest gating as the letter pipeline?
2. Is `static_digit_camera.py` intended to receive a CLI (`--camera-index`, `--check-only`) and a
   `main()` guard, or should it stay a minimal script?
3. Is mirroring meant to be unified (letters mirror vs digits no-mirror), or is the difference intentional
   because each model was trained on its own capture convention?
4. Are J and Z ever going to get a dynamic-gesture runtime using `dulieu/chuyendong/`?
5. Is there a planned `backend/`/`database/` integration, or are these placeholders?
6. Should the duplicated MediaPipe `.task` copies be consolidated, or kept for hash/provenance stability?
7. Confirm the timing of the `data/` migration (splitting the `ai/` data trees into `data/`) once the
   dedicated reference/hash audit is complete.
8. Confirm the `data/` migration plan (raw / processed / splits) and whether the remaining audit/analysis
   scripts should move to `tools/` in a dedicated, reference-checked pass.

---

### Ground rules for future work on this repo
- Do **not** modify `normalize_landmarks` (`ai/preprocessing/landmark_utils.py`) without re-validating both pipelines.
- Do **not** train/retrain or re-promote models as part of unrelated tasks.
- Do **not** add mirror landmarks, augmentation, or preprocessing changes silently.
- Do **not** move or delete original artifacts — training manifests and hash contracts still reference them.
- Treat `ai/dataset/deep_cleanup_snapshot_v1/` and all `camera_debug_*` / `review_*` trees as read-only evidence.

---

## 11. Project-wide architecture contract

This section is the **forward-looking contract for every module**, not a claim that
all existing files have already been migrated. Earlier inventory sections above
describe a point-in-time refactor; verify current paths before acting. Historical
datasets still under `ai/dataset/` stay there until a separate reference, hash,
and reproducibility audit authorizes a migration. This contract does not change
any current model, runtime, dataset, or historical manifest.

### 11.1 Goal and module boundaries

Keep the working project compact and understandable: runtime must not depend on
experiment data, large raw/debug data should be removable from the working
project **after approved finalization**, and compact processed/frozen datasets
and reproducibility code should remain. A new user or AI should be able to
identify each module's input, output, production model, and temporary artifacts.

| Module | Scope | Boundary |
|---|---|---|
| Static letters | Architectural shorthand `A–Y`; the current production classifier has **24 static ASL classes** `A–I, K–Y` | Do not add J/Z to the static classifier. |
| Digits | `0–9` | Separate digit model and data lifecycle. |
| Dynamic | `J`, `Z`, `NONE` | Separate sequence/video pipeline and model. |
| Interaction | Temporal decision, confirmation gesture, interaction state, word builder | Consumes recognition results; not part of the classifiers themselves. |
| Language | Word suggestion, spelling correction, NLP/autocomplete | Later layer; do not silently add it to the current runtime. |
| Future word-level | Whole-word/sign recognition using video/body landmarks | Later expansion after character-level work; do not mix into existing character pipelines. |

### 11.2 Standard lifecycle — all AI modules

`RAW / SOURCE DATA → EXPERIMENT / CURATION → PROCESSED DATASET → FROZEN DATASET → OFFICIAL SPLITS → TRAIN → EVALUATE → PROMOTE → VERIFY RUNTIME → FINALIZE → CLEAN TEMPORARY DATA`

Never promote directly from raw data without a processed/frozen dataset and
evaluation. Keep experiment, development, independent evaluation, and production
artifacts distinguishable. Cleanup is a separate, audited action after
finalization, never an automatic consequence of training or promotion.

### 11.3 Temporary versus permanent artifacts

New improvement/curation work should live under
`data/experiments/<experiment_name>/` (for example,
`data/experiments/digits_v2/`). Temporary artifacts may include raw error-case
images, videos, debug captures, overlays, manual-review working files,
intermediate landmarks, CSV/JSON, candidate datasets and splits, candidate
models, and draft reports. Production code/config must **never** depend directly
on `data/experiments/`. After finalization passes, the entire experiment folder
should be *eligible* for dependency-audited cleanup; only the user may manually
archive or delete its contents after the dependency audit.

Permanent compact data should ultimately be organized as:

```text
data/processed/{letters,digits,dynamic}/
data/splits/{letters,digits,dynamic}/
data/metadata/{letters,digits,dynamic}/
```

Prefer landmark/features/sequence data to retaining every source image or video
inside the working project. These are target locations, not permission to move
the historical `ai/dataset/` tree in this task.

Production and fallback models belong under
`models/{letters,digits,dynamic}/{production,fallback}/`; a promoted production
model requires a deployment manifest. Candidate models are development and
evaluation artifacts and may be cleanup candidates only after audit. Keep light
training/reproducibility code under `training/{letters,digits,dynamic}/`.

### 11.4 Training, raw data, and finalization gates

**AI/Codex must not execute preprocessing, sampling, dataset merging, split
creation, training, or retraining.** For these operations, it may write or
inspect scripts, audit and analyze inputs/outputs, provide commands, and check
results after the user runs them. Stop at execution, give the user the
command/configuration, and wait for the results.

Large raw images/videos need to remain local while extracting, curating,
improving, or reproducing processed data. Once a module is complete, raw data
may be archived to a separate backup/Google Drive and removed from the working
project only when the processed/frozen dataset, official splits, production
model, evaluation, and runtime verification exist; a dependency audit confirms
removal is safe; and the user performs the cleanup manually. Removing a local
copy is not equivalent to deleting a verified backup. Never delete raw data
automatically.

Before declaring any model version finalized, require at least:

1. Compact frozen processed dataset.
2. Official train and validation splits, plus an independent test split where appropriate.
3. Production model and deployment manifest.
4. Final evaluation report and production config.
5. Training/reproducibility code and relevant automated tests.
6. Passing runtime verification and dependency/integrity checks.

Only then audit temporary artifacts for cleanup. The experiment data used for
development must not masquerade as an independent final test set.

### 11.5 Target project layers

```text
app/       runtime application
ai/        shared AI logic: preprocessing, feature extraction, utilities
data/      processed, splits, metadata, experiments
models/    production and fallback models
training/  reproducible training code
tools/     collection, audit, and maintenance utilities
configs/   production configuration
tests/     automated tests
reports/   final evaluation evidence
docs/      human-readable architecture documentation
run.py     entry point
README.md  project overview
SYSTEM_CONTEXT.md  architecture contract and current context
```

Near project completion, perform **one coordinated documentation pass** over
`README.md`, `docs/ARCHITECTURE.md`, `docs/DATA_FLOW.md`,
`docs/MODEL_SUMMARY.md`, and `SYSTEM_CONTEXT.md`. It should explain what runs,
how to run it, folder roles, production models and their datasets, each model's
input/output and data flow, historical evidence, and which experiments may be
removed. Do not perform that documentation pass as part of this contract update.

### 11.6 Current digits-v2 example and protected rules

`data/experiments/digits_v2/` is temporary experiment data. A future digits-v2
finalization should place official compact outputs under
`data/processed/digits/`, `data/splits/digits/`, `data/metadata/digits/`,
`models/digits/production/`, and `reports/digits/`. After runtime verification,
the digit production path must remain functional without
`data/experiments/digits_v2/`.

### 11.7 TEMPORARY_ARTIFACT_LIFECYCLE (project-wide policy)

Every artifact produced during data supplementation, model improvement, debug,
audit, experimentation, candidate training, intermediate evaluation or
preprocessing, and external-dataset processing is **TEMPORARY by default**.
Only an explicit finalization/promotion decision can make an artifact final.
Being used in an experiment does not make it final. Temporary artifacts must
not become runtime/production dependencies or remain scattered across the
project long-term. During an experiment, keep them together under
`data/experiments/<experiment_name>/` or another architecturally designated
temporary area. Production/runtime must never depend on `data/experiments/`
or any temporary/debug artifact.

At experiment completion, identify **one definitive set of final artifacts**
for the module. For a final model, that set normally includes the final frozen
processed dataset, official train/validation/test splits, production model,
deployment manifest, final evaluation report, production config, and necessary
training/reproducibility code. The project should remain compact, understandable,
and free of scattered experimental versions after finalization.

Debug data, error cases, external raw copies inside an experiment, intermediate
CSV files, candidate datasets, failed-sample reports, temporary manifests,
comparison files, audit outputs, and candidate evaluation outputs remain
temporary unless individually promoted. After final results are confirmed,
consolidate temporary artifacts in their designated area and audit their
dependencies. The **user alone** then decides and manually performs any archive
or deletion. AI/Codex may audit, classify, propose, inspect dependencies, and
verify the system after user cleanup, but **must never run delete/remove
commands on any project file or folder**, even with user approval.

The user must personally run actual preprocessing, sampling, dataset merging,
split creation, training, and retraining. AI/Codex may prepare or check scripts,
audit and analyze, give exact commands, and inspect user-supplied results, but
must not execute those data-preparation or training steps. This restriction
applies even when a request asks AI/Codex to prepare a candidate dataset.

For `digits_v2`, **all** content under `data/experiments/digits_v2/` is
temporary, including `ardamavi_landmarks_v1.csv`,
`ardamavi_failed_samples_v1.csv`, `digits_v2a_train.csv`,
`digits_v2a_selected_external.csv`, `error_cases/`, and candidate reports.
None is a final artifact merely by existing or being used during development.
Once digits-v2 is finalized, keep only the explicitly promoted final set;
audit the remaining experiment artifacts and leave any archiving/deletion to
the user.

Do not silently alter production preprocessing, train, delete raw data, replace
a production model without an explicit promotion task, rewrite historical
manifests merely to modernize old paths, break hashes/reproducibility evidence,
or scan large raw/debug datasets when the task does not require it. Read the
smallest relevant scope first.

### 11.8 Unified training entry points (target architecture)

Use one stable training script and one separate evaluation script per module:

| Module | Training entry point | Evaluation entry point | Module config |
|---|---|---|---|
| Letters | `training/letters/train_letters.py` | `training/letters/evaluate_letters.py` | `configs/letters_training.json` |
| Digits | `training/digits/train_digits.py` | `training/digits/evaluate_digits.py` | `configs/digits_training.json` |
| Dynamic | `training/dynamic/train_dynamic.py` | `training/dynamic/evaluate_dynamic.py` | `configs/dynamic_training.json` |

These are **target entry points**, not a claim that all six files or config
directories already exist. Do not create another training script for each
dataset or model version. Represent experiment versions, input paths,
hyperparameters, and output paths in the relevant module config instead.
Training and evaluation must remain separate operations. Each training entry
point must work with VS Code's **Run Python File** action without requiring
command-line arguments and must ask for explicit confirmation before any
training starts; a missing or negative confirmation means no training. The
user, not AI/Codex, runs training. Existing versioned scripts
remain historical until a separately approved migration; this documentation
update does not alter them.
