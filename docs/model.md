# Model notes

Authoritative reference: [../SYSTEM_CONTEXT.md](../SYSTEM_CONTEXT.md).

## Letters

- Active: `uvw_medium`, best epoch 81, at `models/letters/production/`.
- Fallback: `v1_baseline`, best epoch 143, at `models/letters/fallback/`.
- Model and `StandardScaler` are separate joblib artifacts; scaling is applied explicitly with
  `scaler.transform()` before `predict_proba()`.
- Active deployment config: `configs/letters.json` (mirrors the validated
  `ai/static_asl_model_config.json` content with updated paths).
- Loader enforces SHA-256 hashes, 24-class label order `A–I, K–Y`, 63 input features, and the
  expected scaler fit row count.

## Digits

- Production: `mlp_large_v2a`, at `models/digits/production/mlp_large_v2a.joblib`.
- The joblib is a single scikit-learn `Pipeline([StandardScaler, MLPClassifier])`; the digit runtime
  calls `model.predict(X)` / `predict_proba(X)` directly and must not be split into separate
  artifacts.
- Classes `0–9`, 63 features.
- There is no digit fallback model (`models/digits/fallback/` is reserved).
- Candidate models and comparison reports: `models/digits/candidates/`.

## Provenance

Letter candidate models/reports remain under `ai/dataset/mlp_*validation_comparison_v*/` because
their manifests embed historical path/hash contracts; they were not relocated.
