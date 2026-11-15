# Architecture

Authoritative reference: [../SYSTEM_CONTEXT.md](../SYSTEM_CONTEXT.md).

## Two independent runtime programs

Both share one feature contract (MediaPipe Hand Landmarker → 21 landmarks → wrist-relative,
scale-normalized → 63 features) and one MLP classifier family, but they are separate processes.

| Program | Classes | Module | Model |
|---|---|---|---|
| Static ASL letters | 24 (`A–I, K–Y`) | `app/inference/letter_inference.py` | `models/letters/production/uvw_medium_model.joblib` + `uvw_medium_scaler.joblib` |
| Static digits | 10 (`0–9`) | `app/inference/digit_inference.py` | `models/digits/production/mlp_large_v2a.joblib` (bundled `Pipeline(StandardScaler, MLPClassifier)`) |

## Dispatch

`run.py` → `app/camera/realtime_inference.py` → launches the selected program with
`runpy.run_path(..., run_name="__main__")`, so each frozen program keeps its exact top-level
behaviour and remains directly runnable.

## Shared preprocessing

`ai/preprocessing/landmark_utils.py::normalize_landmarks` is the single normalization
implementation used by runtime, training, and tools. It is imported by path (see
SYSTEM_CONTEXT.md §8 for the Windows `ai` vs `AI` case caveat).

## Camera behaviour (do not change silently)

- Letters: frame is mirrored before MediaPipe by default (`--no-mirror` disables).
- Digits: frame is **not** mirrored before MediaPipe. This is required for current accuracy.

## Letter artifact integrity

The letter loader verifies SHA-256 hashes and metadata for model, scaler, label mapping, and the
MediaPipe task before inference. The digit loader does not (single joblib pipeline, no gate).
