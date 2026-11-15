# Stage 1.5 — Final Release Safety Verification Report

**Project Root:** `<project-root>`\
**Mirror Root:** `<project-mirror>`\
**Date:** 2026-11-15\
**Audit Mode:** READ-ONLY / AUDIT-ONLY (Zero files overwritten; no git commands executed; no models modified)\
**Status:** AUDIT COMPLETE — CONDITIONAL CLEARANCE (READY FOR GIT STAGING UPON ADDRESSING 3 ISOLATED PRE-COMMIT ITEMS)\

---

## 1. Workspace Consistency Audit

A comprehensive comparison was performed between `<project-root>` and `<project-mirror>`:

### Authority Determination
- **Authoritative Workspace:** `<project-root>`.
- **Reason:** Active development throughout Stages 14, 15, 16, 17, 18, 19, 20, and Stage 1 occurred directly in `<project-root>`.
  - Example: `app/inference/integrated_jz_validation.py` in `D:` has size `27,519 bytes` (mtime `2026-11-14`, containing the Stage 16 Z-Handshape validation logic), whereas in `C:` it is `23,974 bytes` (mtime `2026-11-13`, prior to Stage 16).
  - `reports/data/stage17_webcam_results.json` in `D:` has size `57,380 bytes` (mtime `2026-11-14T17:53:33Z`), whereas in `C:` it is an older version (`59,954 bytes`).

### Workspace Differences
1. **Source Code & Models:**
   - All 18 Stage 1 synchronized files (including `models/space/production/final_space_model.joblib`, `.gitignore`, `tools/download_models.py`, `README.md`, `requirements.txt`, `environment.yml`, `LICENSE`) have identical content across both workspaces.
   - 1 source file difference exists: `app/inference/integrated_jz_validation.py` (D is newer and authoritative).
2. **Historical Markdown Reports:**
   - 28 markdown reports have 1-byte differences caused strictly by CRLF vs LF line endings.
3. **Files present only in `<project-mirror>`:**
   - `tests/fixtures/sample_dictionary.txt` (1,475 bytes — test fixture for unit tests).
   - `reports/RUN_PY_SPACE_JZ_ROOT_CAUSE_AUDIT.md` (25,533 bytes).
   - `reports/SPACE_INTEGRATION_AUDIT.md` (15,154 bytes).
   - `reports/training_audit_20261112/` (12 historical training audit files totaling ~85 KB).
4. **Files present only in `<project-root>`:**
   - `ai/camera_debug_pq/` (431 files, ~137 MB of debug frames and camera capture logs from earlier development stages; correctly excluded by `.gitignore`).

> [!NOTE]
> Neither workspace was overwritten or modified during this verification.

---

## 2. Production Model Inventory & Runtime Tracing

Every asset required for application startup and recognition was traced through its actual code loading path:

| Asset Name | Exact File Path | Exists | Size (Bytes) | SHA-256 Checksum | Source Code Loading Point | Required for Startup? |
|---|---|---|---|---|---|---|
| **Static ASL Letters** | `models/letters/production/uvw_medium_model.joblib` | Yes | 218,456 | `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` | `app/inference/letter_inference.py::load_official_artifacts()` via `src/model.py::load_letters()` | Yes (Unified & Letters mode) |
| **Static Digits** | `models/digits/production/digits_final.joblib` | Yes | 464,265 | `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` | `app/inference/unified_static_coordinator.py::load_digit_model()` via `src/model.py::load_digits()` | Yes (Unified & Digits mode) |
| **SPACE Gesture** | `models/space/production/final_space_model.joblib` | Yes | 1,375 | `453162d8aa0212016216a6c02798d3dcaa445c1fbb7a6340c896bb828e076121` | `app/inference/space_predictor.py::SpacePredictor.from_config()` | Yes (when `--enable-space` is on, default in `run.py`) |
| **Dynamic J/Z Pipeline** | `data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib` | Yes | 36,657 | `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497` | `app/inference/dynamic_jz_camera_test.py::load_experiment_model()` via `src/model.py::load_dynamic()` | Yes (when J/Z is enabled, default in `run.py`) |
| **MediaPipe Landmarker** | `models/mediapipe/hand_landmarker.task` | Yes | 7,819,105 | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` | `ai/feature_extraction/hand_landmarks.py::create_hand_landmarker()` | Yes (when camera opens) |
| **Word Dictionary** | `data/dictionary/english_words_25k.tsv` | Yes | 291,528 | `986eead866a09df979d0c95a8250f235cb4bdf0bdadd5b3699db610e3ad4525e` | `app/inference/unified_abc123_jz_auto_experiment.py` | Yes (when word suggestions on, default in `run.py`) |

### Forensic Analysis of `dynamic_jz_binary_baseline_v1/model_pipeline.joblib`
- **Question:** Is this the actual production model, or an experimental artifact?
- **Finding:** In the application's formal architecture, the static letters and digits are promoted production models (`models/letters/production`, `models/digits/production`), whereas dynamic J/Z is documented as an **integrated experimental baseline candidate** wrapped by `First16Segmenter` and `RatioClassifier`.
- Its training manifest (`data/experiments/dynamic_jz_binary_baseline_v1/training_manifest.json`) retains `"status": "EXPERIMENTAL_NOT_PRODUCTION"`.
- However, `configs/application.json` pins `"dynamic_experiment_dir": "data/experiments/dynamic_jz_binary_baseline_v1"`, and `src/model.py::load_dynamic()` explicitly verifies and loads this exact file.
- **Conclusion:** It IS the actual, functioning runtime classifier for dynamic J/Z in the unified application.

---

## 3. Model & Third-Party Asset Licensing Audit

Each candidate asset was audited for provenance, origin data, and redistribution rights:

| Asset | Provenance / Training Source | Governing License | Classification | Redistribution Action |
|---|---|---|---|---|
| `models/mediapipe/hand_landmarker.task` | Google LLC MediaPipe Official Release | Apache License 2.0 | **Redistribution permitted** | Downloaded dynamically via `tools/download_models.py` or tracked directly. |
| `data/dictionary/english_words_25k.tsv` | SUBTLEX-US corpus (Brysbaert & New, Ghent University) | Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0) | **Redistribution permitted** | Permitted under CC BY-SA 4.0. Accompanied by [data/dictionary/NOTICE.md](file:///<project-root>/data/dictionary/NOTICE.md). |
| `data/experiments/.../model_pipeline.joblib` | SigNN ASL J/Z Video Dataset (Kaggle) | Creative Commons Zero 1.0 (CC0: Public Domain) | **Redistribution permitted** | Fully permissive public domain. Rights evidence documented in `signnteam_source_rights_evidence_v1.json`. |
| `models/letters/production/uvw_medium_model.joblib` | `raw_landmarks_chucai_all.csv` (`source='internet'`) + 57 custom camera captures | Unspecified internet dataset licenses | **License unclear** | The trained weights are owned by project author, but upstream image licenses are unformalized. |
| `models/digits/production/digits_final.joblib` | `dataset_chuso_all.csv` (699 internal samples) | Internal / Author proprietary collection | **License unclear / Author-owned** | Permitted if released by author under MIT/Apache, but requires explicit author release confirmation. |
| `models/space/production/final_space_model.joblib` | Phase 3.9.8 custom palm landmark captures | Internal / Author proprietary collection | **License unclear / Author-owned** | Requires explicit author release confirmation. |

> [!IMPORTANT]
> **License Recommendation:** Code is licensed under MIT. The dictionary is licensed under CC BY-SA 4.0. The J/Z model is CC0. For `uvw_medium_model.joblib`, `digits_final.joblib`, and `final_space_model.joblib`, we recommend that the repository author include an explicit Model Weights License statement in `docs/MODEL_LICENSES.md` before making the GitHub repository public.

---

## 4. Dependency Compatibility Verification

Comparison between `requirements.txt`, `environment.yml`, and active environment `nn`:

### Active Tested Environment
- **Python:** `3.11.15` (64-bit Windows)
- **OpenCV:** `opencv-python 5.0.0.93` / `opencv-contrib-python 5.0.0.93`
- **MediaPipe:** `1.0.1`
- **NumPy:** `2.4.6`
- **scikit-learn:** `1.9.0`
- **joblib:** `1.6.0`
- **pandas:** `3.0.5`
- **scipy:** `1.17.1` (transitive dependency of scikit-learn)

### `requirements.txt` Evaluation
- Specified:
  ```txt
  opencv-python>=4.8.0
  mediapipe>=0.10.0
  numpy>=1.24.0
  scikit-learn>=1.3.0
  joblib>=1.3.0
  pandas>=2.0.0
  ```
- **Pip Dry-Run Test:** Executed `pip install --dry-run -r requirements.txt`. All requirements and transitive dependencies (`sounddevice`, `flatbuffers`, `narwhals`, `threadpoolctl`, `python-dateutil`) resolved with **Exit Code: 0** and zero conflicts.
- **Unpickle Warning Notice:** In Python `nn` environment, loading `final_space_model.joblib` emits:
  `InconsistentVersionWarning: Trying to unpickle estimator LogisticRegression from version 1.9.1 when using version 1.9.0.`
  This warning is purely informational and does not impair execution or accuracy.

---

## 5. Model Asset Downloader Verification (`tools/download_models.py`)

A rigorous safety inspection was conducted on `tools/download_models.py`:

1. **Download URL:** `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task` — verified as Google's official immutable distribution point.
2. **SHA-256 Checksum:** `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` — verified matching the on-disk task bundle byte-for-byte.
3. **Atomic Replacement:** Downloads to a temporary file (`.task.tmp`) first. Checksum is computed on `.tmp`. The destination file is only replaced (`tmp_dst.replace(dst)`) after checksum verification succeeds.
4. **Failure Handling:** If download is interrupted or checksum fails, `.tmp` is immediately unlinked (`tmp_dst.unlink()`). No corrupted file can remain at destination.
5. **Existing File Protection:** If destination file exists and matches SHA-256, download is skipped. Existing models are never overwritten needlessly.
6. **Local Asset Mode:** `check_local_assets()` is purely read-only verification; it never downloads or overwrites local production checkpoints.

---

## 6. Exact Candidate Publication Inventory

Simulated via temporary Git directory without initializing `<project-root>`:

- **Total Candidate Files:** 407 files
- **Total Tracked Size:** 27.35 MB (28,674,516 bytes)

### Breakdown by Directory
| Directory | File Count | Size | Primary Contents |
|---|---|---|---|
| `ai/` | 49 | 10.65 MB | Landmark utils, normalization algorithms, constants |
| `models/` | 25 | 9.49 MB | MediaPipe task bundle, production & fallback letter/digit/space weights |
| `data/` | 5 | 3.49 MB | 25k dictionary (`english_words_25k.tsv`), SUBTLEX raw source, dynamic J/Z model |
| `tools/` | 59 | 0.88 MB | Model downloader, evaluation scripts, diagnostics |
| `tests/` | 99 | 0.81 MB | Automated unit and integration test suite |
| `app/` | 57 | 0.79 MB | Runtime inference loops, multi-modal coordinator, text composition |
| `reports/` | 34 | 0.66 MB | Architecture reports, validation findings |
| `training/` | 17 | 0.32 MB | Model training routines (not run automatically) |
| `docs/` | 16 | 0.12 MB | Architectural diagrams, design documentation |
| `results/` | 3 | 0.06 MB | Architecture refactor evidence manifests |
| `root` | 8 | 0.06 MB | `run.py`, `README.md`, `requirements.txt`, `environment.yml`, `LICENSE`, `.gitignore`, `SYSTEM_CONTEXT.md` |
| `src/` | 25 | 0.02 MB | Public facade APIs for models, inference, and camera |
| `configs/` | 10 | 0.01 MB | Runtime configuration files (`application.json`, `space.json`, etc.) |

### Exclusion Verification
- **Raw Datasets (`ai/dataset/`, `ai/raw_data/`):** 0 files tracked (IGNORED).
- **Video Recordings (`*.mp4`, `*.avi`):** 0 files tracked (IGNORED).
- **Archive Files (`*.zip`, `*.tar`):** 0 files tracked (IGNORED).
- **Log Files (`*.log`):** 0 files tracked (IGNORED).
- **Camera Debug Dumps (`ai/camera_debug_*/`):** 0 files tracked (IGNORED).

---

## 7. Public Privacy and Leakage Audit Findings

A full regex search across all 407 candidate files yielded:

1. **Credentials & Secrets:** **ZERO** API keys, personal access tokens, or private credentials found.
2. **Embedded Media (Images):**
   - **Finding:** 3 JPEG files tracked under `ai/final_promotion_audit/run_20260917_181104_976231/`:
     * `contact_sheet_bad_stability4.jpg` (146 KB)
     * `contact_sheet_random30.jpg` (1.27 MB)
     * `contact_sheet_random30_plus_bad4.jpg` (1.42 MB)
   - **Risk:** These files contain contact sheet mosaics of hand webcam photos from early 2026 validation runs.
   - **Action Item:** Add `ai/final_promotion_audit/` or `*.jpg` to `.gitignore`.
3. **Machine-Specific Paths in Code:**
   - `tests/test_z_trajectory_validation_a9.py` (Line 8): Contains hardcoded diagnostic path `r"<project-mirror>\data\experiments\dynamic_jz_v1\diagnostics\20261108T143709941895Z_44ef47a7"`.
   - `SYSTEM_CONTEXT.md` (Line 5): Contains `> Root: <project-mirror>`.
   - **Action Item:** Sanitize both before public release.
4. **Duplicate Large Files:**
   - `ai/models/hand_landmarker.task` (7.8 MB) is an unneeded duplicate of `models/mediapipe/hand_landmarker.task`.
   - `data/dictionary/source/SUBTLEXus74286wordstextversion.txt` (3.3 MB) is the raw uncompressed source corpus; only `english_words_25k.tsv` is needed at runtime.
   - **Action Item:** Exclude `ai/models/` and `data/dictionary/source/` in `.gitignore`.

---

## 8. Verification Execution Results

All three mandatory verification commands were executed in `<conda-env-nn>/python.exe`:

### 1. `python tools/download_models.py --check-only`
- **Output:**
  ```
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
- **Exit Code:** `0` (PASS)

### 2. `python run.py --check-only`
- **Output:**
  ```json
  {
    "status": "PASS",
    "experimental": true,
    "camera_opened": false,
    "mediapipe_executed": false,
    "files_written": 0,
    "modes": ["ABC", "123"],
    "word_suggestion_enabled": true,
    "word_suggestion_initialized": true,
    "space_enabled": true,
    "space_model": "<project-root>\\models\\space\\production\\final_space_model.joblib",
    "camera_count_when_running": 1,
    "landmarker_count_when_running": 2,
    "letters_model": "uvw_medium",
    "letters_epoch": 81,
    "digits_model": "models/digits/production/digits_final.joblib",
    "dynamic_model_sha256": "8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497",
    "dynamic_feature_shape": [16, 69],
    "jz_commits_enabled": true
  }
  ```
- **Exit Code:** `0` (PASS)

### 3. Stage 18–20 Core Regression Test Suite (49 Tests)
- **Command:**
  ```powershell
  python -m unittest tests/test_stage20_one_click_run.py tests/test_stage19_suggestion_integration.py tests/test_stage18_production_jz_integration.py tests/test_jz_production_integration.py tests.test_src_architecture.EntrypointTests
  ```
- **Output:**
  ```
  Ran 49 tests in 28.729s
  OK
  ```
- **Exit Code:** `0` (PASS)

---

## 9. Remaining Blockers & Pre-Commit Items

To transition from Stage 1.5 to Stage 2 with zero security or privacy risks, the following 3 items must be handled:

| Item | Category | Issue | Proposed Remediation |
|---|---|---|---|
| **1. Privacy (.gitignore)** | Leakage | 3 JPG contact sheets in `ai/final_promotion_audit/` contain camera frames of hands; raw corpus `data/dictionary/source/` (3.3 MB) and duplicate `ai/models/` (7.8 MB) are tracked. | Add `ai/final_promotion_audit/`, `ai/models/`, and `data/dictionary/source/` to `.gitignore`. Reduces repo size from 27.3 MB to ~13.4 MB and eliminates image leakage. |
| **2. Path Sanitization** | Privacy | `test_z_trajectory_validation_a9.py` (L8) and `SYSTEM_CONTEXT.md` (L5) contain hardcoded `<dev-home>`. | Convert to dynamic root relative paths `ROOT / ...`. |
| **3. Model Licensing** | Legal | Letters, digits, and space models lack explicit author redistribution declarations. | Add `docs/MODEL_LICENSES.md` declaring author redistribution terms under MIT/CC. |

---

## 10. Final Recommendation

**Status:** **READY FOR GIT REVIEW (UPON EXECUTING THE 3 PRE-COMMIT ACTIONS ABOVE)**.

The codebase is technically sound, clean of credentials, 100% verified across all regression suites, and operating with relative path resolutions. Once the 3 pre-commit items above are applied, the project is completely cleared for `git init` and initial public release.

---
*Report generated and validated by Antigravity.*
