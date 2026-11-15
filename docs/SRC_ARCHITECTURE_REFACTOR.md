# NhanDienCuChi — staged source architecture refactor

Date: 2026-10-14 (Asia/Saigon). Scope executed: A, B and C. No dataset/model
migration, model promotion, training, extraction or live-camera validation.

## Stage A — current implementation and baseline

The workspace is not a Git repository. Uncommitted-vs-committed status cannot
be established. Existing files were treated as user-owned. Baseline evidence is
under `results/experiments/architecture_refactor_v1/`:

- `baseline.json`: SHA256/size of 87 scoped source/config/model/manifest files
  and original source text for the four edited existing files.
- `rollback_bytes.json`: hash-verified original bytes for those four files,
  base64 encoded; no automatic restore is performed.

Baseline suite: 318 tests. Initial sandbox run: 315 passed, three permission
errors reading the letter validation CSV. Those exact three tests passed with
approved read access. No baseline algorithm failure was found.

Actual production dispatch is `run.py -> app/camera/realtime_inference.py`.
The latest integrated experiment is
`app/inference/unified_abc123_jz_auto_experiment.py`, backed by
`unified_abc123_jz_auto_coordinator.py`, not the earlier unified comparison cameras.

| Model | Artifact | Contract | Status |
|---|---|---|---|
| Letters | `models/letters/production/uvw_medium_model.joblib` + `uvw_medium_scaler.joblib` | 63 features; 24 mapped labels A-I/K-Y; epoch 81 | Production |
| Digits | `models/digits/production/digits_final.joblib` | 63 features; HandCentric2DAligner -> scaler -> MLP; 0-9 | Production |
| J/Z | `data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib` | (16,69), C-order flatten 1104; scaler -> LogisticRegression; J/Z | Experimental |
| MediaPipe | `models/mediapipe/hand_landmarker.task` | Existing task hash; VIDEO mirrored ABC / independent IMAGE raw J/Z | Shared task asset |

No trained 26-class model was found among the active artifacts. Alphabet
integration is arbitration over distinct input contracts, not merged weights.
Letter fallback remains `models/letters/fallback/`; digit fallback remains
`models/digits/fallback/mlp_large_v2a_pre_digits_final.joblib`.

The J/Z loader additionally requires and hashes its separate scaler,
`training_manifest.json`, `split_manifest.json` and `config_snapshot.json`.
Those dependencies and locations remain intact.

## Stages B/C — implemented adapter architecture

```text
run.py --mode letters/digits -> unchanged standalone programs
run.py --mode unified -> src/main.py -> src/camera/realtime.py
    -> existing OpenCV lifecycle, with injected model loader and AlphabetRecognizer

src/model.py        validated loaders + pinned application config
src/data.py         read-only CSV/JSONL contract readers; no preparation
src/predict.py      model-only static/dynamic API; no camera/UI
src/train.py        explicit user-only delegation to existing training
src/evaluate.py     explicit compatibility-aware evaluation routing
src/recognition/    static letters, digits, J/Z and A-Z facade
src/preprocessing/  aliases of canonical functions/classes
src/interaction/    unchanged HOLD, TextComposer, coordination, SHADOW exports
src/ui/             aliases of existing mirrored render/buttons
```

All implementation modules remain at their original paths. No original source
module is deleted, moved or replaced with a different algorithm. The new
`AlphabetRecognizer` inherits the existing coordinator without overriding its
recognition, confirmation, reset or suppression methods. The old integrated
entrypoint retains its original default factory/loader; only optional dependency
injection hooks were added so the new entrypoint uses the facade.

Letters still consume mirrored VIDEO observations. Digits/J/Z still consume
raw/no-flip observations. One physical camera and two independent MediaPipe
passes remain; this refactor does not collapse them. UI always stays mirrored.
Dynamic features remain canonical 63 shape values plus fixed-scale relative XY
trajectories for landmarks 0/8/20, sampled across valid frames into (16,69).

HOLD, TextComposer, no-duplicate commit/rearm, motion gates and Z start-pose
SHADOW/temporal evidence are unchanged. J/Z commits require the existing explicit
`--enable-experimental-jz`; no start-pose or historical geometry acceptance gate
was enabled. Existing diagnostics/replay keep their existing experiment paths.

### Serialized module compatibility

Digit joblib records `hand_centric_2d.HandCentric2DAligner`. The existing loader
still adds `ai/preprocessing` to sys.path before deserializing. Do not remove or
rename that module. The new `src/preprocessing/hand_centric.py` is an import
alias, not a new serialized transformer implementation. The older
`orientation_2d_experimental` module also remains for historical checkpoints.

## Workflow gaps — not silently repaired

- Letters have historical versioned scripts but no `train_letters.py`,
  `evaluate_letters.py` or `letters_training.json`. The facade returns a clear
  unsupported result, rather than choosing a historical training run.
- Digit training delegates to the existing frozen-replay entrypoint/config.
- The existing digit evaluator requires a two-step scaler/MLP pipeline; it is
  incompatible with production's three-step orientation pipeline. The facade
  blocks this route. It does not remove orientation or report old metrics as new.
- Dynamic evaluation's default experiment already has `metrics.json`. Explicit
  user-selected `--experiment-dir` and the existing no-overwrite checks remain.
- `src` workflow `--check-only` is a routing check, not dataset/rights/evaluation
  preflight. Underlying tools retain their more detailed checks.

## Stage D — deliberately deferred

[The migration map](architecture_migration_map.json) distinguishes API adapters
from proposed physical destinations. No physical move occurred. A full raw-data
inventory was not performed; only the relevant small artifacts were inventoried.
`dataset/` is not created as an empty/fake migrated dataset. Before any migration,
inventory counts/bytes, hash sources, inspect recorded source paths/lineage and
obtain approval for large or historical artifacts. Dynamic artifacts remain
experimental and must not be promoted by a path refactor.

Existing hardcoded project paths in loaders, training configs, extraction source
paths, frozen manifests, code-hash evidence and diagnostics output guards remain
dependencies. Historical strings such as `AI/...` must not be rewritten just to
make documentation look modern. Data/history must not be blindly moved into results.

## Stage E — partial documentation only

README documents the unified entry and workflow limitations. Existing dependency
pins in requirements.txt are unchanged; the adapters add no external dependency.
Complete dataset/results migration, training/evaluation standardization and
production finalization are not claimed.

## Verification

- Stage B: syntax/import checks passed for all 24 src modules, with model loading
  and camera access forbidden during imports.
- Stage C: 343 targeted offline regression tests passed with approved access to
  the existing letter validation CSV; 25 are new architecture/parity tests.
  This is the relevant runtime suite, not discovery over all historical tooling.
- Real artifact loading and synthetic-input probability parity passed for all
  three models. Digit serialized transformer path/classes remained unchanged.
- Exact model input, candidate/state/static suppression, HOLD, text and SHADOW
  parity passed over synthetic timelines and ABC/123 switching.
- Unified check-only passed; zero camera/MediaPipe execution/diagnostic writes.
- Additional task-loading check creates/closes IMAGE and VIDEO landmarkers from
  the configured task; no frame is passed to MediaPipe and no camera is opened.
- Baseline hash tests protect 83 unchanged scoped files (87 less four intended
  existing-file edits). They include models, manifests, normalization, builder,
  coordinator, thresholds, training code, configs and requirements.

These are offline/synthetic checks, not live-camera PASS or signer-independent
J/Z evaluation. Existing datasets and historical reports were not changed.
Historical replay may report a changed source hash for the integrated entrypoint
because its optional injection hooks changed. This is not a feature/model hash
match claim; historical manifests were not rewritten. Canonical builder and
model hashes are unchanged. New adapter hashes are recorded in the refactor
verification report; existing attempt capture/replay implementations remain.

## Commands and rollback

```powershell
python run.py --mode letters --check-only
python run.py --mode digits
python run.py --mode unified --check-only --z-start-pose-shadow
python run.py --mode unified
python run.py --mode unified --enable-experimental-jz
python -m src.train --module digits --check-only
python -m src.evaluate --module dynamic --check-only
```

To avoid the new integration, keep using the old letters/digits commands or run
the original integrated experiment directly. It retains its default coordinator
and loader. Full source rollback is a separate user action: inspect baseline
hashes, decode the four original files from `rollback_bytes.json`, verify their
SHA256 against baseline and restore only those exact paths after preserving any
newer user edits. Do not use git reset or delete src/history automatically.
