# Dataset notes

Authoritative reference: [../SYSTEM_CONTEXT.md](../SYSTEM_CONTEXT.md).

## Current on-disk layout (NOT yet moved)

The target `data/` tree exists but is **empty**. Raw and processed datasets still live at their
historical paths because they are large and referenced by historical manifests:

| Purpose | Current location |
|---|---|
| Raw source images (letters) | `dulieu/chucai/` |
| Raw source images (digits) | `dulieu/chuso/` |
| Raw source images (dynamic, unused by runtime) | `dulieu/chuyendong/` |
| Letter extracted features + splits | `ai/dataset/*.csv`, `ai/dataset/train_validation_test_split_v1/` |
| Digit extracted features + splits | `ai/dataset/kytuso/{landmark,datasetsach,datatrain}/` |
| Camera debug evidence | `ai/camera_debug_*/` |
| Frozen snapshots (read-only) | `ai/dataset/deep_cleanup_snapshot_v1/`, `ai/_cleanup_quarantine_v1/` |

`dulieu/` must not be moved, renamed, or restructured.

## Feature contract

21 landmarks → wrist-relative, max-distance scaled → `f1..f63` (see
`ai/preprocessing/landmark_utils.py`).

## Digit dataset facts

- Clean dataset: `ai/dataset/kytuso/datasetsach/dataset_chuso_clean_v1.csv` (699 samples).
- Split by `hand_id` (group split), documented in
  `models/digits/production/deployment_manifest_v1.json`.
