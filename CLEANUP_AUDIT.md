# CLEANUP_AUDIT

> Read-only audit (2026-10-02). **No file was deleted, moved, or modified.**
> Scope: `app/`, `ai/` (excluding data), `training/`, `tools/`, `tests/`, `configs/`, `docs/`,
> `models/*.json`, `run.py`, `SYSTEM_CONTEXT.md`, `README.md`, `requirements.txt`, `.gitignore`.
> Excluded from scanning: `dulieu/`, `ai/dataset/` bulk data, `.joblib`, images/videos, caches,
> snapshots, debug trees.

## Method

1. Built a reference index over all in-scope text files: exact relative path, file name, and module
   name (imports + path literals), counting referencing files.
2. Ran targeted searches into the historical manifests under `ai/dataset/deep_cleanup_audit_v1/`
   (JSON only) to catch references that live in manifest/contract files rather than in code.
3. Reused the project's own prior disposition record
   `ai/dataset/deep_cleanup_audit_v1/final_cleanup_plan_v2.json` (`dispositions_explicit`, 1022 items).
   Every in-scope `.py` file it lists is already dispositioned **KEEP_REPRODUCIBILITY** or
   **KEEP_PRODUCTION** — no in-scope code file is marked `SAFE_TO_QUARANTINE`,
   `DELETE_AFTER_REFERENCE_REMOVAL`, or `DELETE_AFTER_SNAPSHOT` (those 152 items are all inside
   `ai/dataset/**` checkpoints/joblibs, out of scope here).

A file is **not** treated as unused merely because no runtime module imports it; tests, manifests,
snapshots, contracts, docs and historical evidence all count as references.

## Counts

| Category | Files |
|---|---|
| KEEP_RUNTIME | 23 |
| KEEP_CURRENT_TRAINING | 16 |
| KEEP_REPRODUCIBILITY | 49 |
| KEEP_TEST | 20 |
| POTENTIALLY_DELETABLE | 0 |
| DELETE_ONLY_AFTER_REFERENCE_REMOVAL | 2 |
| MANUAL_REVIEW | 7 |
| **Audited total** | **117** |

---

## KEEP_RUNTIME (23)

`run.py`-adjacent runtime and the artifacts/config the camera actually reads:

- `app/__init__.py`, `app/camera/__init__.py`, `app/camera/realtime_inference.py`,
  `app/inference/__init__.py`, `app/inference/letter_inference.py`, `app/inference/digit_inference.py`
- `ai/__init__.py`, `ai/preprocessing/__init__.py`, `ai/preprocessing/landmark_utils.py`
- `configs/letters.json`
- `models/letters/production/{evaluation_summary,label_mapping,selected_candidate,training_config,training_manifest}.json`
- `models/letters/fallback/{evaluation_manifest,label_mapping,selected_model,training_config,training_manifest}.json`
- `tests/__init__.py` (test package marker)

Not enumerated as files but equally required at runtime (not read during this audit):
`models/mediapipe/hand_landmarker.task`, `models/letters/{production,fallback}/*.joblib`,
`models/digits/production/mlp_large_v2a.joblib`.

## KEEP_CURRENT_TRAINING (16)

Current letter/digit train–evaluate pipelines (no newer replacement exists for any of these):

- `training/letters/`: `create_train_validation_test_split.py`, `finalize_static_dataset.py`,
  `prepare_mlp_v2_training_data.py`, `prepare_uvw_training_data.py`,
  `train_mlp_v2_validation.py`, `train_uvw_validation.py`, `train_compare_mlp_validation.py`
- `training/digits/`: `merge_chuso_clean_v1.py`, `freeze_chuso_dataset_v1.py`,
  `split_chuso_dataset_v1.py`, `train_chuso_mlp_candidates_v1.py`, `train_chuso_mlp_candidates_v2.py`,
  `evaluate_chuso_validation_v2.py`
- `tools/extract_letters.py`, `tools/extract_digits.py`

Note on old `train_*`: there is **no superseded letter trainer** — `train_mlp_v2_validation.py`,
`train_uvw_validation.py` and `train_compare_mlp_validation.py` are the current comparison pipeline.
`training/letters/train_mlp_mirror.py` is an older experimental generation, but the project's prior
audit records it as `KEEP_REPRODUCIBILITY` ("old model generator … keep source code").

## KEEP_TEST (20)

Automated tests plus the modules they import and the offline harness:

- Tests: `tests/test_analyze_train_balance.py`, `test_cleanup_snapshot_contract.py`,
  `test_create_train_validation_test_split.py`, `test_finalize_static_dataset.py`,
  `test_near_duplicate_audit.py`, `test_phase2_review_promotion.py`,
  `test_preprocessing_freeze_review.py`, `test_static_asl_camera.py`,
  `test_train_compare_mlp_validation.py`, `test_train_mlp_v2.py`, `test_train_uvw_validation.py`,
  `test_wrong_label_audit.py`
- Modules imported by those tests: `ai/analyze_train_balance.py`, `ai/audit_near_duplicates.py`,
  `ai/audit_wrong_labels.py`, `ai/cleanup_snapshot_contract.py`,
  `ai/finalize_preprocessing_minor_fixes.py`, `ai/phase2_review_promotion.py`,
  `ai/regenerate_phase2_quality_gate.py`
- Offline harness: `ai/verify_static_asl_no_camera.py` (runs the suite with `cv2.VideoCapture` forbidden)

## KEEP_REPRODUCIBILITY (49)

Historical analysis/audit helpers, evidence records, docs and evidence manifests. All are referenced
by the prior cleanup plan as `KEEP_REPRODUCIBILITY`, by tests/scripts, or by SYSTEM_CONTEXT/README:

- `ai/`: `analyze_image_quality.py`, `analyze_landmark_quality.py`, `analyze_pq_debug_results.py`,
  `analyze_quality_relative.py`, `audit_landmark_failure_modes.py`, `audit_uv_debug.py`,
  `audit_v1_v_errors.py`, `visualize_geometry_candidates.py`, `visualize_image_risk.py`,
  `visualize_outliers.py`, `visualize_reject_candidates.py`, `count_reject_candidates.py`,
  `detect_outliers_chucai.py`, `rescue_chuso_no_hand.py`, `rescue_chuso_padding.py`,
  `sample_geometry_levels.py`, `finalize_pq_debug_audit.py`, `run_landmark_quality_pipeline.py`,
  `collect_data.py`
- `ai/`: `final_landmark_dispositions.json`, `preprocessing_semantic_observations.json`,
  `PREPROCESSING_FREEZE_REVIEW.md`, `config/ky_tu.txt`
- `docs/`: `architecture.md`, `dataset.md`, `model.md`, `static_asl_camera.md`,
  `pq_horizontal_failure_analysis.md`, and `docs/model_v2_p_horizontal_design_v1/*`
- `models/`: `letters/production/deployment_manifest_v3.json`, all `digits/**/*.json`,
  `letters/candidates/README.md`, `digits/fallback/README.md`
- Root docs: `SYSTEM_CONTEXT.md`, `README.md`, `requirements.txt`, `.gitignore`

## POTENTIALLY_DELETABLE (0)

No in-scope file satisfies "reference_found: 0 and superseded and not needed for reproducibility".

- Every in-scope `.py` file is either referenced by runtime/training/tests/docs or is explicitly
  dispositioned `KEEP_REPRODUCIBILITY` / `KEEP_PRODUCTION` by the project's own
  `final_cleanup_plan_v2.json`.
- The analysis/audit scripts with a zero **code** reference count
  (`analyze_image_quality.py`, `analyze_landmark_quality.py`, `analyze_pq_debug_results.py`,
  `audit_landmark_failure_modes.py`, `audit_uv_debug.py`, `audit_v1_v_errors.py`, `visualize_*.py`)
  are all listed as inventory entries in the historical cleanup manifests, which is why the prior
  audit kept them for reproducibility.

## DELETE_ONLY_AFTER_REFERENCE_REMOVAL (2)

### 1. `tests/test_static_mirror.py`

- **reason**: archived legacy mirror-camera implementation that now lives entirely inside a module
  docstring; the runnable tail only delegates to `app/inference/letter_inference.py` and opens a real
  webcam, so it is not an automated test and duplicates `test_static_asl_camera.py`.
- **reference_found**: 2
  (`SYSTEM_CONTEXT.md`; `ai/dataset/deep_cleanup_audit_v1/final_cleanup_plan_v1.json` and
  `…_v2.json` — `KEEP_REPRODUCIBILITY`, kept as evidence of the `mlp_static_mirror` experiment; the
  archived docstring also references `landmark_utils`)
- **replaced_by**: `tests/test_static_asl_camera.py` (automated, `VideoCapture` mocked)
- **risk**: MEDIUM — it is the only remaining evidence record tying `mlp_static_mirror.pkl` to its
  generator; removing it would weaken the reproducibility chain of that experiment.

### 2. `training/digits/evaluate_chuso_validation_v1.py`

- **reason**: superseded by `evaluate_chuso_validation_v2.py`, which produced the validation analysis
  used for digit candidate selection (`candidate_comparison_v2.json`, `validation_analysis_v2.json`).
- **reference_found**: 2 (historical only — `models/digits/candidates/candidate_comparison_v1.json`
  and `models/digits/candidates/validation_analysis_v1.json`; no current script or test imports it)
- **replaced_by**: `training/digits/evaluate_chuso_validation_v2.py`
- **risk**: MEDIUM — the v1 result artifacts are retained, so the v1 script is the only way to
  re-derive them; delete only after those `candidates/*_v1.json` artifacts are also retired.

## MANUAL_REVIEW (7)

| File | Why it needs a human decision |
|---|---|
| `app/inference/word_builder.py` | Deliberate empty placeholder (word composition not implemented); imported by nothing. Keep as reserved location or remove. |
| `ai/common/constants.py` | New shared-constants module; currently imported only by `ai/feature_extraction/hand_landmarks.py`, not by runtime. Wire it in or let it idle as the canonical home. |
| `ai/feature_extraction/hand_landmarks.py` | New MediaPipe helper; not imported by any runtime or test. Intended for a future behaviour-checked consolidation. |
| `configs/digits.json` | Documentation-only (digit runtime reads no config file); no code reads it. |
| `configs/camera.json` | Documentation-only; no code reads it. |
| `tests/test_hand.py` | Manual webcam demo with no assertions, guarded by `if __name__ == "__main__"`; not an automated test. Prior plan marks `KEEP_REPRODUCIBILITY`. |
| `tests/test_landmarks.py` | Manual webcam demo with no assertions; not an automated test. Prior plan marks `KEEP_REPRODUCIBILITY`; referenced by `SYSTEM_CONTEXT.md`. |

---

## Specific checks requested

| Check | Result |
|---|---|
| Old `train_*` replaced by newer version | **No superseded letter/digit trainer found.** `train_mlp_v2_validation`, `train_uvw_validation`, `train_compare_mlp_validation`, `train_chuso_mlp_candidates_v1/v2` are all current or required for reproducibility. `train_mlp_mirror.py` is old but explicitly kept by the prior audit. |
| Old `evaluate_*` | `evaluate_chuso_validation_v1.py` is superseded by `_v2.py` → DELETE_ONLY_AFTER_REFERENCE_REMOVAL. `evaluate_chuso_validation_v2.py` is current. |
| Duplicate / unsuitable tests | `tests/test_static_mirror.py` duplicates the camera test and is non-automated → DELETE_ONLY_AFTER_REFERENCE_REMOVAL. `tests/test_hand.py`, `tests/test_landmarks.py` are non-asserting manual tools → MANUAL_REVIEW. |
| Old `audit_*`, `analyze_*`, `finalize_*`, `rescue_*` | All still referenced by the historical cleanup manifests (and some by tests) → KEEP_REPRODUCIBILITY / KEEP_TEST. |
| `verify_*` / `cleanup_*` still called | `ai/verify_static_asl_no_camera.py` = active offline verification harness → KEEP_TEST. `ai/cleanup_snapshot_contract.py` is imported by `tests/test_cleanup_snapshot_contract.py` → KEEP_TEST. |
| Historical scripts used by manifests/snapshots | Yes — `final_cleanup_plan_v1/v2.json`, `reference_review_v1.json`, `manual_review_resolution_v1.json`, `snapshot_manifest.json` list them; that is why they are KEEP_REPRODUCIBILITY. |

FILES_DELETED: 0
TRAINING_PERFORMED: False
STATUS: CLEANUP_AUDIT_COMPLETE
