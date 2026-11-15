# Stage 1.7 — Public Source-Only Preparation & Local Model Preservation Report

**Project Root:** `<project-root>`\
**Mirror Root:** `<project-mirror>`\
**Date:** 2026-11-15\
**Execution Mode:** SOURCE-ONLY REPOSITORY PREPARATION (Option B)\
**Safety Status:** ZERO model weights modified or deleted; ZERO retraining performed; ZERO git publishing actions run.

---

## 1. Executive Summary & Core Objective

In accordance with user authorization for **STAGE 1.7**, the repository has been prepared for a public open-source release on GitHub following **Option B (Source-Only Release)**.

The source code, MediaPipe landmark processors, feature normalization algorithms, dynamic J/Z models (CC0), word dictionaries (CC BY-SA 4.0), training pipelines, and test suites are configured for public tracking. Unverified pre-trained model weights (`.joblib` files for static letters, digits, and space) are excluded via `.gitignore` while being **fully preserved locally on disk** for developer execution.

### Key Deliverables Completed
1. **`.gitignore` Rules Updated:** Excluded the 3 unverified production models plus all fallback/candidate model checkpoints from git tracking. Retained Apache 2.0 MediaPipe model, CC0 Dynamic J/Z baseline, and CC BY-SA 4.0 dictionary assets.
2. **Model Loading & Informative Diagnostics:** Refined model loaders across `app/camera/realtime_inference.py`, `src/model.py`, `app/inference/space_predictor.py`, `app/inference/letter_inference.py`, `app/inference/unified_static_coordinator.py`, and `app/inference/digit_inference.py` to provide user-friendly error messages directing clean-clone users to `docs/MODEL_DISTRIBUTION.md`.
3. **Distribution Documentation:** Authored `docs/MODEL_DISTRIBUTION.md` and updated `README.md` with explicit instructions on model provenance, expected paths, and custom training scripts.
4. **Local Model Invariance Verified:** Cryptographic SHA-256 hashes of local models confirmed 100% identical before and after.
5. **Full Regression & Simulation Testing:** All 49 regression tests passed, `run.py --check-only` passed, and missing-model simulation verified 6/6 error-handling scenarios.
6. **Dual-Workspace Parity:** Perfect 100% hash parity achieved across `<project-root>` and `<project-mirror>`.

---

## 2. Model Inventory & Git Tracking Verification

An isolated temporary Git tracking probe (`git status --porcelain --ignored=matching`) was executed to verify ignore status without running `git init` in the project root:

| Asset Name | Canonical Path | Size | Git Ignore Status | Tracked in Git? | License / Provenance Classification |
|---|---|---|---|---|---|
| **Static ASL Letters Model** | `models/letters/production/uvw_medium_model.joblib` | 218,456 bytes | **`!!` (Ignored)** | **NO** | Proprietary / Unverified Internet Dataset |
| **Static Digits Model** | `models/digits/production/digits_final.joblib` | 464,265 bytes | **`!!` (Ignored)** | **NO** | Private author capture dataset |
| **SPACE Gesture Model** | `models/space/production/final_space_model.joblib` | 1,375 bytes | **`!!` (Ignored)** | **NO** | Private author capture dataset |
| **Letters Fallback Models** | `models/letters/fallback/*.joblib` | 438,136 bytes | **`!!` (Ignored)** | **NO** | Unverified / Obsolete baseline checkpoints |
| **Digits Fallback Models** | `models/digits/fallback/*.joblib`, `models/digits/production/mlp_large_v2a.joblib` | 928,530 bytes | **`!!` (Ignored)** | **NO** | Unverified / Obsolete digit checkpoints |
| **MediaPipe Hand Landmarker** | `models/mediapipe/hand_landmarker.task` | 7,842,504 bytes | **`??` (Tracked)** | **YES** | **Apache 2.0 (Google LLC)** |
| **Dynamic J/Z Baseline** | `data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib` | 36,657 bytes | **`??` (Tracked)** | **YES** | **CC0 1.0 Universal Public Domain (SigNN)** |
| **Dynamic J/Z Scaler** | `data/experiments/dynamic_jz_binary_baseline_v1/scaler.joblib` | 1,224 bytes | **`??` (Tracked)** | **YES** | **CC0 1.0 Universal Public Domain** |
| **Letters Scaler Contract** | `models/letters/production/uvw_medium_scaler.joblib` | 1,224 bytes | **`??` (Tracked)** | **YES** | Normalization contract (no learned weights) |
| **Word Dictionary (25k)** | `data/dictionary/english_words_25k.tsv` | 340,653 bytes | **`??` (Tracked)** | **YES** | **CC BY-SA 4.0 (SUBTLEX-US)** |
| **Dictionary Attribution** | `data/dictionary/NOTICE.md` | 1,189 bytes | **`??` (Tracked)** | **YES** | Mandatory open-source attribution notice |
| **Deployment Manifests** | `models/letters/production/deployment_manifest_v3.json`, etc. | Various | **`??` (Tracked)** | **YES** | Integrity and evaluation schema definitions |
| **Raw Datasets & Videos** | `dulieu/`, `ai/dataset/`, `ai/final_promotion_audit/`, `ai/models/` | ~3.8 GB | **`!!` (Ignored)** | **NO** | Heavy data / private captures excluded |

### Updated Candidate Publication Payload
- **Total Tracked Files:** **~392 files** (excluding unverified binary checkpoints).
- **Total Tracked Size:** **~12.06 MB** (down from 27.35 MB before audit, a **55.9% total size reduction**).
- **Core Binary Asset:** MediaPipe Hand Landmarker (`hand_landmarker.task`, 7.48 MB, Apache 2.0).
- **Total Code, Configs & Docs:** ~4.58 MB.

---

## 3. Cryptographic Verification of Local Models

To guarantee zero byte degradation, accidental deletion, or weight tampering, the cryptographic SHA-256 hashes of the three local production models were checked before and after Stage 1.7 execution:

| Model Asset | Pre-Stage 1.7 SHA-256 | Post-Stage 1.7 SHA-256 | Byte Size | Status |
|---|---|---|---|---|
| `uvw_medium_model.joblib` | `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` | `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` | 218,456 B | **IDENTICAL (100% UNCHANGED)** |
| `digits_final.joblib` | `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` | `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` | 464,265 B | **IDENTICAL (100% UNCHANGED)** |
| `final_space_model.joblib` | `453162d8aa0212016216a6c02798d3dcaa445c1fbb7a6340c896bb828e076121` | `453162d8aa0212016216a6c02798d3dcaa445c1fbb7a6340c896bb828e076121` | 1,375 B | **IDENTICAL (100% UNCHANGED)** |

---

## 4. Model Loader Error Handling & Diagnostics

The loader logic across all primary entrypoints was inspected and fortified to satisfy dual operational requirements:

### A. Local Machine (Developer Environment)
When models exist on disk, the system runs with zero change in performance, thresholding, or inference pipelines:
- `python run.py --check-only` returns `{"status": "PASS", "space_enabled": true, "word_suggestion_enabled": true, "jz_commits_enabled": true, ...}`.

### B. Fresh Clone (Public User Environment)
When cloned to a machine without model files, rather than throwing an unhandled exception or crashing silently, the entrypoint intercepts the missing files and provides clear, actionable instructions:
```text
ERROR: Missing required pre-trained model file(s):
  Static ASL letter model (models/letters/production/uvw_medium_model.joblib)
  Static digit model (models/digits/production/digits_final.joblib)

This public repository is distributed source-only without unverified pre-trained weights.
Please refer to docs/MODEL_DISTRIBUTION.md for instructions.
```
- **Zero Dummy Predictions:** The system does NOT inject fake predictions or dummy stubs.
- **Fail-Safe Parity:** Individual loaders (`SpacePredictor`, `src.model.load_letters`, `src.model.load_digits`, `digit_inference`, `letter_inference`) raise descriptive `FileNotFoundError` exceptions pointing to `docs/MODEL_DISTRIBUTION.md`.

---

## 5. Verification Test Results

All verification suites were run using the authoritative Conda environment Python `<conda-env-nn>/python.exe`:

| Verification Step | Command / Scope | Result / Output Summary | Exit Code | Status |
|---|---|---|---|---|
| **Python Syntax Check** | `python -m py_compile app/camera/realtime_inference.py app/inference/space_predictor.py app/inference/letter_inference.py app/inference/unified_static_coordinator.py app/inference/digit_inference.py src/model.py` | Compilation succeeded with zero warnings | `0` | **PASS** |
| **Runtime Preflight Check** | `python run.py --check-only` | All models loaded successfully; `status: PASS` | `0` | **PASS** |
| **Core Regression Suite** | `python -m unittest tests/test_stage20_one_click_run.py tests/test_stage19_suggestion_integration.py tests/test_stage18_production_jz_integration.py tests/test_jz_production_integration.py tests.test_src_architecture.EntrypointTests` | Ran 49 tests in 25.956s — `OK` (100% tests passed) | `0` | **PASS** |
| **Git Tracking Probes** | `scratch/verify_git_tracking.py` | 23 paths probed against `.gitignore`: 11 ignored, 12 tracked | `0` | **PASS** |
| **Missing Model Simulation** | `scratch/test_missing_models_simulation.py` | 6 simulated missing-model tests passed; clear error messages verified | `0` | **PASS** |
| **Dual-Workspace Parity** | `scratch/verify_dual_workspaces.py` | 9 modified/created files compared: 100% hash match across D: and C: | `0` | **PASS** |

---

## 6. Workspace Parity Table

All files modified or authored in Stage 1.7 were mirrored with identical SHA-256 hashes across both workspaces:

| Relative Path | Size | D: SHA-256 (Primary) | C: SHA-256 (Mirror) | Parity Status |
|---|---|---|---|---|
| `.gitignore` | 2,752 B | `c9eb0b03...` | `c9eb0b03...` | **PERFECT MATCH** |
| `docs/MODEL_DISTRIBUTION.md` | 4,374 B | `077cc5b8...` | `077cc5b8...` | **PERFECT MATCH** |
| `README.md` | 7,678 B | `46c452bd...` | `46c452bd...` | **PERFECT MATCH** |
| `app/camera/realtime_inference.py` | 2,953 B | `12a786c6...` | `12a786c6...` | **PERFECT MATCH** |
| `app/inference/space_predictor.py` | 6,325 B | `a8f02d5d...` | `a8f02d5d...` | **PERFECT MATCH** |
| `app/inference/digit_inference.py` | 5,556 B | `f268cc6b...` | `f268cc6b...` | **PERFECT MATCH** |
| `app/inference/letter_inference.py` | 31,185 B | `5d47bc29...` | `5d47bc29...` | **PERFECT MATCH** |
| `app/inference/unified_static_coordinator.py` | 13,090 B | `dfe7ccec...` | `dfe7ccec...` | **PERFECT MATCH** |
| `src/model.py` | 3,170 B | `e4f34f5a...` | `e4f34f5a...` | **PERFECT MATCH** |

---

## 7. Public GitHub Release Readiness Status

The project is now in a **clean, legally compliant, source-only state**:
- No copyright risks or unverified image model weights will be pushed to GitHub.
- Local developer workflow is completely unbroken: running `run.py` or clicking Run in VS Code continues to recognize ASL letters, digits, dynamic J/Z, space, and word suggestions.
- New users cloning from GitHub have full instructions in `README.md` and `docs/MODEL_DISTRIBUTION.md` on how to provide weights or train replacement models.
- All 49 regression tests pass 100%.

**STOP CONDITION MET:** The repository is fully prepared for Stage 2 (Git Initialization & Initial Commit). Awaiting user authorization.

---
*Report generated and validated by Antigravity.*
