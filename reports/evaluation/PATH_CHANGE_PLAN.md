# PATH_CHANGE_PLAN

Project root: `<project-mirror>`. This plan was prepared before moving any artifact. Large source images in `dulieu`, `AI/raw_data`, and `AI/dataset` are out of scope for automatic moves.

## Approved, bounded change

| Field | Value |
|---|---|
| ITEM | `ky_tu.txt` (legacy collection label list) |
| TYPE | CONFIG |
| OLD_PATH | `<project-mirror>\AI\ky_tu.txt` |
| NEW_PATH | `<project-mirror>\AI\config\ky_tu.txt` |
| REFERENCED_BY | Active: `AI/collect_data.py` (path construction at line 85; missing-file message at line 111). Historical only: `AI/dataset/deep_cleanup_audit_v1/final_cleanup_plan_v1.json`, `final_cleanup_plan_v2.json`, and its immutable snapshot copy. |
| REFERENCE_COUNT | 5 textual occurrences across 4 files (2 active, 3 historical). |
| REFERENCES_TO_UPDATE | Change only the path construction in `AI/collect_data.py` from `BASE_DIR / "ky_tu.txt"` to `BASE_DIR / "config" / "ky_tu.txt"`; leave historical audit records unchanged. |
| RUNTIME_CRITICAL | NO for the production camera; YES for the legacy `collect_data.py` collector. |
| SAFE_TO_MOVE | YES, conditional on byte-identical SHA256, destination vacancy, updated collector path, and offline validation. |

Preflight: source exists, 76 bytes, SHA256 `9f37bc6fe390f0e7dab61c76de3a697f45370ade427ea536768297cd55640e8c`; destination does not exist; `AI/config` may be created inside the project. File name and extension remain unchanged. No other active source/config references were found in `*.py`, `*.json`, `*.yaml`, `*.yml`, `*.toml`, `*.ini`, `*.md`, or `*.txt`.

## Deferred: do not move in this pass

| Atomic item | Current paths | Proposed destination | SAFE_TO_MOVE | Reason |
|---|---|---|---|---|
| Production model, scaler, mapping, config, manifests | `AI/dataset/mlp_uvw_validation_comparison_v1/`, `AI/static_asl_model_config.json`, `AI/dataset/static_asl_camera_deployment_v2/` | `AI/models/letters_static/production/` and `AI/config/` | NO | Camera, verification, training manifests, hash contracts, and historical evidence reference the current paths. A separate versioned deployment migration is needed. |
| Fallback model, scaler, mapping | `AI/dataset/mlp_validation_comparison_v1/` | `AI/models/letters_static/fallback/` | NO | Coupled to production fallback and historical validation/test manifests. |
| MediaPipe task | `AI/models/hand_landmarker.task` | `AI/models/mediapipe/hand_landmarker.task` | NO | Used by camera, collector, tests, and recorded hashes; must migrate atomically with runtime contracts. |
| Camera and normalization source | `AI/static_asl_camera.py`, `AI/landmark_utils.py` | `AI/src/inference/`, `AI/src/utils/` | NO | Moving changes import roots, command-line entry point, tests, and stored normalization contract. |
| Preprocessing/training source and outputs | `AI/*.py`, `AI/dataset/`, `AI/raw_data/` | `AI/src/`, `AI/training/`, `AI/dataset/letters/` | NO | Scripts use file-relative roots and manifests record source/output paths and hashes. |
| Debug/audit/snapshot trees | `AI/camera_debug_*`, `AI/dataset/*audit*`, `AI/dataset/deep_cleanup_snapshot_v1/` | `AI/debug/`, `AI/audit/`, `AI/backup/` | NO | Raw evidence, restore mapping, historical references, and large file counts require a separate bounded migration. |
| Original images | `dulieu/` | unchanged | NO | `LARGE_IMMUTABLE_SOURCE_DATA` by instruction. |

No move of a deferred item is authorized by this plan. Historical reports may retain old paths as historical facts; active runtime references may not.

## Phase 2 amendment — runtime artifacts (2026-10-01)

This amendment supersedes the **production, fallback, and MediaPipe rows only** in the deferred table above, for a **copy-and-switch migration**. Originals remain in place to preserve training, preprocessing, test, and historical hash contracts. No directory is recursively moved; only the explicitly listed files below may be copied. `SAFE_TO_MOVE=YES` means safe to **copy after source/destination/hash preflight**, not permission to delete the source.

Absolute path prefix for each `OLD_PATH` and `NEW_PATH` below: `<project-mirror>\`. All destinations are currently vacant. Source filename, extension, and bytes must be unchanged in the copy.

| Group / component | OLD_PATH (under prefix) | NEW_PATH (under prefix) | REFERENCED_BY / REFERENCES_TO_UPDATE | RUNTIME_CRITICAL | SAFE_TO_MOVE |
|---|---|---|---|---|---|
| PRODUCTION model | `AI\dataset\mlp_uvw_validation_comparison_v1\uvw_medium_model.joblib` | `AI\models\letters_static\production\uvw_medium_model.joblib` | `selected_candidate.json::source_model_path`, `training_manifest.json::outputs`, camera `UVW_MODEL_DIR`, active config, deployment v2. Keep basename selectors; change camera directory, config path, and publish deployment v3. | YES | YES (copy only) |
| PRODUCTION scaler | `AI\dataset\mlp_uvw_validation_comparison_v1\uvw_medium_scaler.joblib` | `AI\models\letters_static\production\uvw_medium_scaler.joblib` | Same selectors, config, and deployment references as model. | YES | YES (copy only) |
| PRODUCTION mapping | `AI\dataset\mlp_uvw_validation_comparison_v1\label_mapping.json` | `AI\models\letters_static\production\label_mapping.json` | `selected_candidate.json::label_mapping_path`, manifest, camera, config, deployment. | YES | YES (copy only) |
| PRODUCTION selection | `AI\dataset\mlp_uvw_validation_comparison_v1\selected_candidate.json` | `AI\models\letters_static\production\selected_candidate.json` | Camera and `training_manifest.json::outputs`; basename references remain valid after complete group copy. | YES | YES (copy only) |
| PRODUCTION training manifest | `AI\dataset\mlp_uvw_validation_comparison_v1\training_manifest.json` | `AI\models\letters_static\production\training_manifest.json` | Camera, config, deployment; source snapshot/output records inside it remain historical. | YES | YES (copy only) |
| PRODUCTION training config | `AI\dataset\mlp_uvw_validation_comparison_v1\training_config.json` | `AI\models\letters_static\production\training_config.json` | Camera and training manifest; historical input snapshots remain unchanged. | YES | YES (copy only) |
| PRODUCTION evaluation summary | `AI\dataset\mlp_uvw_validation_comparison_v1\evaluation_summary.json` | `AI\models\letters_static\production\evaluation_summary.json` | Camera checks best epoch 81. | YES | YES (copy only) |
| FALLBACK selected model | `AI\dataset\mlp_validation_comparison_v1\selected_model.joblib` | `AI\models\letters_static\fallback\selected_model.joblib` | Historical TEST evaluation manifest gives this basename and hash; camera must resolve the same basename inside the new fallback group. | YES | YES (copy only) |
| FALLBACK baseline alias | `AI\dataset\mlp_validation_comparison_v1\baseline_model.joblib` | `AI\models\letters_static\fallback\baseline_model.joblib` | Active config/deployment reference this byte-identical alias of selected model. | YES | YES (copy only) |
| FALLBACK scaler | `AI\dataset\mlp_validation_comparison_v1\scaler.joblib` | `AI\models\letters_static\fallback\scaler.joblib` | TEST manifest, camera, config, deployment. | YES | YES (copy only) |
| FALLBACK mapping | `AI\dataset\mlp_validation_comparison_v1\label_mapping.json` | `AI\models\letters_static\fallback\label_mapping.json` | TEST manifest, camera, config, deployment. | YES | YES (copy only) |
| FALLBACK selection | `AI\dataset\mlp_validation_comparison_v1\selected_model.json` | `AI\models\letters_static\fallback\selected_model.json` | Camera verifies selected baseline. | YES | YES (copy only) |
| FALLBACK training manifest | `AI\dataset\mlp_validation_comparison_v1\training_manifest.json` | `AI\models\letters_static\fallback\training_manifest.json` | Camera, config, deployment; historical output paths remain untouched. | YES | YES (copy only) |
| FALLBACK training config | `AI\dataset\mlp_validation_comparison_v1\training_config.json` | `AI\models\letters_static\fallback\training_config.json` | Camera checks features/labels and original MediaPipe asset hash. | YES | YES (copy only) |
| FALLBACK TEST evidence | `AI\dataset\mlp_baseline_test_evaluation_v1\evaluation_manifest.json` | `AI\models\letters_static\fallback\evaluation_manifest.json` | Camera uses selected-before-test/epoch/hash evidence. Old embedded paths are historical and must resolve by basename into copied fallback group; the original TEST report remains unchanged. | YES | YES (copy only) |
| MEDIAPIPE task | `AI\models\hand_landmarker.task` | `AI\models\mediapipe\hand_landmarker.task` | Camera `HAND_MODEL_PATH` and training-config hash key; change only the resolved camera path, keep historical hash key. Extraction/quality/test scripts still use the retained original asset; they are not modified. | YES | YES (copy only) |

Atomic switching order and gate:

1. Copy/verify all seven production files, then update only camera production directory and active config production paths; run 83 tests plus default/fallback `--check-only`. No next group on failure.
2. Copy/verify all eight fallback files, then update camera fallback directory and basename-based TEST evidence resolution plus active config fallback paths; run the same gate. Keep historical TEST manifest content intact.
3. Copy/verify MediaPipe task, change camera's resolved `HAND_MODEL_PATH`, and add the active task path to config; run offline unit/default/fallback and explicit Hand Landmarker create/close. The original asset remains for legacy pipelines.
4. Publish a new deployment manifest v3 with the changed source/config and copied artifact hashes; do not edit v1/v2 manifests. Verify active runtime paths, old source hashes, feature count, 24-class mapping, and prediction parity.

If any gate fails, stop and restore the immediately preceding camera/config references to their prior values; do not remove copied files without separate authorization. Existing dataset, model, scaler, label mapping, normalization, thresholds, preprocessing scripts, training scripts, and historical manifests are immutable in this phase. `preprocessing_frozen_v3/README.md` mismatch is `PREEXISTING_KNOWN_ISSUE` and out of scope.

### Phase 2 execution record

- Production group: 7/7 explicit copies, source/destination SHA256 identical. Camera and active config now resolve under `AI/models/letters_static/production/`. Immediate gate: 83/83 tests, default and fallback `--check-only`, and active-config path parity **PASS**.
- Fallback group: 8/8 explicit copies, source/destination SHA256 identical. Historical TEST evidence is copied byte-for-byte; the camera validates its exact historical paths but loads the corresponding new-basename artifacts. Immediate gate: 83/83 tests, both `--check-only` modes, and active-config path parity **PASS**.
- MediaPipe group: 1/1 explicit copy, source/destination SHA256 identical. Original task remains for legacy preprocessing scripts. Immediate gate: 83/83 tests, both `--check-only` modes, and offline Hand Landmarker create/close **PASS**.
- Active deployment contract: `AI/models/letters_static/production/deployment_manifest_v3.json`; v1/v2 are unchanged history. V3 verifies 16/16 copied artifacts and camera/config hashes; protected historical split/model checks remain 19/19 **PASS**.
- Resolved runtime path references to the former model/scaler/mapping/MediaPipe locations: **0**. Literal old paths retained only as historical TEST/training-evidence keys or in unmodified legacy scripts/manifests.
- No source artifact, model binary, scaler, mapping, dataset, image, historical manifest, or backup was deleted. Large data moved: **NO**.
