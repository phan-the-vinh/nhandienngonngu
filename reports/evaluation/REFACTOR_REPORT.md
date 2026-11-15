# REFACTOR_REPORT

STATUS: `SAFE_MODEL_ARTIFACT_REFACTOR_COMPLETE` for the runtime-model phase (2026-10-01). The wider source/dataset/debug/audit layout remains a separate future task. See [PATH_CHANGE_PLAN.md](PATH_CHANGE_PLAN.md) for the pre-copy path audit and group-by-group gate record.

## OLD_STRUCTURE

- `AI/ky_tu.txt`: 26-label list read by the legacy collection script.
- `AI/static_asl_camera.py`, `AI/landmark_utils.py`, `AI/static_asl_model_config.json`: current camera entry, normalization, and deployment config.
- `AI/models/hand_landmarker.task`: MediaPipe task; `AI/dataset/mlp_uvw_validation_comparison_v1/` and `AI/dataset/mlp_validation_comparison_v1/`: production and rollback artifacts.
- `AI/dataset/`, `AI/raw_data/`, `AI/camera_debug_*`, `dulieu/`: datasets and evidence at established paths.

## NEW_STRUCTURE

- `AI/config/ky_tu.txt`: legacy collection label list (prior phase).
- `AI/models/letters_static/production/`: hash-identical `uvw_medium` runtime model, scaler, mapping, and required verification metadata; active `deployment_manifest_v3.json`.
- `AI/models/letters_static/fallback/`: hash-identical baseline selected model, scaler, mapping, and required verification metadata.
- `AI/models/mediapipe/hand_landmarker.task`: active camera MediaPipe task copy.
- Original model folders, historical deployment v1/v2 manifests, and the original MediaPipe task remain unchanged for training/preprocessing provenance.

## RUNTIME_ARTIFACT_GRAPH

| COMPONENT | CURRENT_PATH before phase | REFERENCED_BY | RUNTIME_CRITICAL | PROPOSED_PATH, now active |
|---|---|---|---|---|
| Production model / scaler / mapping | `AI/dataset/mlp_uvw_validation_comparison_v1/{uvw_medium_model.joblib,uvw_medium_scaler.joblib,label_mapping.json}` | Camera UVW loader, selection/training metadata, active config, deployment v2 | YES | Same basenames under `AI/models/letters_static/production/` |
| Production metadata | `AI/dataset/mlp_uvw_validation_comparison_v1/{training_manifest.json,training_config.json,selected_candidate.json,evaluation_summary.json}` | Camera UVW loader, active config | YES | Same basenames under `AI/models/letters_static/production/` |
| Fallback model / scaler / mapping | `AI/dataset/mlp_validation_comparison_v1/{selected_model.joblib,scaler.joblib,label_mapping.json}` | Camera baseline loader, TEST evidence, active config | YES | Same basenames under `AI/models/letters_static/fallback/` |
| Fallback metadata | `AI/dataset/mlp_validation_comparison_v1/{training_manifest.json,training_config.json,selected_model.json}` and `AI/dataset/mlp_baseline_test_evaluation_v1/evaluation_manifest.json` | Camera baseline loader | YES | Same basenames under `AI/models/letters_static/fallback/` |
| MediaPipe task | `AI/models/hand_landmarker.task` | Camera, legacy extraction/quality code, training snapshot hashes | YES | `AI/models/mediapipe/hand_landmarker.task` for camera; original retained for legacy scripts |
| Active config / deployment manifest | `AI/static_asl_model_config.json`, `AI/dataset/static_asl_camera_deployment_v2/deployment_manifest.json` | Camera/config and deployment evidence | YES / evidence | Config stays at its stable entry path; new active `AI/models/letters_static/production/deployment_manifest_v3.json` |

## FILES_MOVED

| From | To | Result |
|---|---|---|
| `AI/ky_tu.txt` | `AI/config/ky_tu.txt` | 76 bytes; SHA256 before/after `9f37bc6fe390f0e7dab61c76de3a697f45370ade427ea536768297cd55640e8c` |

## FILES_COPIED_IN_THIS_PHASE

Seven production, eight fallback, and one MediaPipe artifact were copied individually to the paths in `deployment_manifest_v3.json` (16/16 byte-identical). This was a copy-and-switch migration, not a destructive move. No original was removed.

## FILES_LEFT_IN_PLACE

- Production entry/config remain at `AI/static_asl_camera.py` and `AI/static_asl_model_config.json`; their references changed, not their location.
- Original production, fallback, TEST-evaluation, and MediaPipe artifacts remain at `AI/dataset/mlp_uvw_validation_comparison_v1/`, `AI/dataset/mlp_validation_comparison_v1/`, `AI/dataset/mlp_baseline_test_evaluation_v1/`, and `AI/models/hand_landmarker.task`.
- Historical deployment v1/v2 manifests and `AI/landmark_utils.py` remain unmodified.
- Training/preprocessing scripts, tests, frozen/split datasets, `AI/raw_data/`, original images, camera-debug evidence, audits, and immutable cleanup snapshot. Their references and/or historical hash contracts make a bulk move unsafe in this pass.

## PATHS_UPDATED

- `AI/collect_data.py`: `LETTERS_PATH` now resolves to `AI/config/ky_tu.txt` from the script location. Content, collection logic, and other paths were not changed.
- Historical cleanup plans still record `AI/ky_tu.txt` as its original path; these are historical records, not active runtime references.
- `AI/static_asl_camera.py`: `MODEL_DIR`, `UVW_MODEL_DIR`, TEST evidence copy, and `HAND_MODEL_PATH` resolve to new runtime directories. Historical evidence strings are validated, but no old model/scaler/mapping/MediaPipe path is opened by the camera.
- `AI/static_asl_model_config.json`: default/fallback artifact paths and active MediaPipe/deployment-manifest paths point to new copies. Default remains `uvw_medium`; fallback remains `v1`.
- `AI/models/letters_static/production/deployment_manifest_v3.json`: new versioned runtime manifest. The old v1/v2 deployment manifests were not overwritten.

## PRODUCTION_ENTRY

`AI/static_asl_camera.py`; default `--model uvw_medium`, fallback `--model v1`. Tests use `--check-only` and do not open a webcam.

## PRODUCTION_MODEL

`uvw_medium`, epoch 81: `AI/models/letters_static/production/uvw_medium_model.joblib`, paired with its scaler and mapping in the same directory. The source original remains at `AI/dataset/mlp_uvw_validation_comparison_v1/`.

There is no separately loaded UVW/special classifier: `uvw_medium` itself is the active 24-class MLP.

## FALLBACK_MODEL

`v1_baseline`, epoch 143: `AI/models/letters_static/fallback/selected_model.joblib`, paired with its scaler and mapping. The copied `baseline_model.joblib` is byte-identical; the original files remain at `AI/dataset/mlp_validation_comparison_v1/`.

## MEDIAPIPE_MODEL

`AI/models/mediapipe/hand_landmarker.task` is used by the camera; `AI/models/hand_landmarker.task` remains unchanged for legacy preprocessing code and historical hashes.

## TEST_RESULTS

- Baseline and after each atomic group: **PASS**, 83/83 unit tests; `uvw_medium` and `v1_baseline` artifact/hash/load/mapping/inference checks **PASS**; webcam-opening attempts: **0**.
- Changed file: new path exists, old path absent, original SHA256 preserved, collector path resolves, collector source parses, and label list retains 26 entries: **PASS**.
- MediaPipe Hand Landmarker created and closed without a camera: **PASS**.
- Both camera `--check-only` modes run correctly from outside the project working directory: **PASS** (63 features, 24 static classes).
- New runtime manifest: **16/16** copied artifacts match source size and SHA256, camera/config hashes match, runtime paths match active config. Historical split and original model checks: **19/19 hashes PASS**. Frozen/TRAIN/VALIDATION/TEST and original model binaries were not edited.
- Final active-path audit for the moved file: no active source/config reference to its old location. Historical cleanup records retain old-path facts.

## MANUAL_DATA_MOVE_REQUIRED

None for this pass. `dulieu/` remains `LARGE_IMMUTABLE_SOURCE_DATA` at `<project-mirror>\dulieu`; no move, rename, copy, or internal reorganization was attempted. Moving the large debug, audit, or dataset trees to a future target structure would require a separate path/dependency audit and explicit review; this report does not authorize such a move.

## SAFE_TO_REMOVE_LATER

None established for original runtime artifacts: old paths remain referenced by historical/training/preprocessing contracts, so byte-identical runtime copies alone do **not** authorize deleting originals. No file, dataset, image, model, or snapshot was deleted.

## REMAINING_RISKS

- Wider `AI/src`, `AI/debug`, `AI/audit`, `AI/backup`, and dataset-layout changes are not in this runtime-only phase. Historical training and TEST manifests retain old path strings by design; original artifacts are still present to keep those contracts valid.
- Historical cleanup plans reference the original `AI/ky_tu.txt`; they are unchanged evidence and should not be interpreted as current runtime paths.
- Pre-existing frozen-v3 `README.md` hash differs from the value inside `final_freeze_manifest.json` (current 808 bytes versus recorded 1115 bytes). This pass did not touch either file; canonical dataset/group map and the split/model hashes checked above are unchanged. A separate documentation-integrity decision is needed before claiming every frozen-v3 output hash matches.
