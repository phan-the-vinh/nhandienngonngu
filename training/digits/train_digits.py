"""Stable, user-operated entry point for digit candidate training."""

import argparse
import hashlib
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "digits_training.json"
OFFICIAL_TEST = (PROJECT_ROOT / "ai/dataset/kytuso/datatrain/test_v1.csv").resolve()
PRODUCTION_DIR = (PROJECT_ROOT / "models/digits/production").resolve()
PREPROCESSING_DIR = PROJECT_ROOT / "ai/preprocessing"
ORIENTATION_MODULE = PREPROCESSING_DIR / "hand_centric_2d.py"
FROZEN_TRAIN = (PROJECT_ROOT / "data/processed/digits/digits_final_train_frozen.csv").resolve()
FROZEN_VALIDATION = (PROJECT_ROOT / "data/splits/digits/validation_reference.csv").resolve()
FROZEN_SAMPLE_MAP = (PROJECT_ROOT / "data/metadata/digits/digits_final_sample_map.csv").resolve()
FROZEN_MANIFEST = (PROJECT_ROOT / "data/metadata/digits/digits_final_freeze_manifest.json").resolve()
FROZEN_REPLAY_SPEC = (PROJECT_ROOT / "data/metadata/digits/digits_final_replay_spec.json").resolve()
FROZEN_LINEAGE = (PROJECT_ROOT / "data/metadata/digits/digits_final_repro/training_manifest_digits_final.json").resolve()
REPLAY_OUTPUT_ROOT = (PROJECT_ROOT / "data/experiments/digits_final_replay").resolve()


def project_path(value):
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_counts(frame, labels):
    return {label: int((frame["label"] == label).sum()) for label in labels}


def format_class_counts(counts, labels):
    return " ".join(f"{label}={counts.get(label, 0)}" for label in labels)


def print_prepared_summary(config, selection):
    labels = [str(i) for i in range(10)]
    print("\nPrepared data summary:")
    print(f"  Experiment: {config['experiment_name']}")
    print(f"  Base samples: {selection['base_train_samples']}")
    for source in selection["external_sources"]:
        print(
            f"  {source['name']}: selected={source['selected_rows']}, "
            f"dedup removed={source['exact_duplicate_feature_rows_removed']}"
        )
        requested_labels = sorted(source["samples_per_class_requested"], key=int)
        print(
            "    Selected/class: "
            + format_class_counts(source["selected_per_class"], requested_labels)
        )
        if source["selected_group_counts"]:
            group_counts = source["selected_group_counts"]
            print(
                "    Source groups: "
                + ", ".join(f"{group}={count}" for group, count in sorted(group_counts.items()))
            )
    print(f"  Combined samples: {selection['combined_train_samples']}")
    print("  Combined class counts: " + format_class_counts(selection["combined_train_per_class"], labels))
    print(f"  Validation samples: {selection['validation_samples']}")


def sample_class(frame, count, seed, source_name, label, group_column):
    """Prefer a round-robin spread over known filename groups when available."""
    import numpy as np

    derived_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{source_name}:{label}".encode("utf-8")).digest()[:4], "big"
    )
    if group_column and group_column in frame.columns:
        groups = frame[group_column].fillna("").astype(str).str.strip()
        known = ~groups.str.upper().isin(("", "UNKNOWN", "NONE", "NULL", "NAN", "<NA>"))
        known_frame = frame.loc[known]
        if known_frame[group_column].nunique() >= 2:
            rng = np.random.default_rng(derived_seed)
            group_names = sorted(known_frame[group_column].unique().tolist())
            rng.shuffle(group_names)
            queues = {}
            for group in group_names:
                indexes = known_frame.loc[known_frame[group_column] == group].index.to_numpy().copy()
                rng.shuffle(indexes)
                queues[group] = indexes.tolist()
            selected_indexes = []
            while len(selected_indexes) < count and any(queues.values()):
                for group in group_names:
                    if queues[group] and len(selected_indexes) < count:
                        selected_indexes.append(queues[group].pop())
            if len(selected_indexes) < count:
                remainder = frame.index.difference(selected_indexes).to_numpy()
                rng.shuffle(remainder)
                selected_indexes.extend(remainder[:count - len(selected_indexes)].tolist())
            return frame.loc[selected_indexes].sort_index(kind="stable"), "round_robin_known_groups", derived_seed

    return frame.sample(n=count, random_state=derived_seed).sort_index(kind="stable"), "random_no_usable_groups", derived_seed


def prepare_training_data(config, base_path, source_specs, validation_path):
    """Read, filter, sample and merge in memory after user confirms preparation."""
    import numpy as np
    import pandas as pd

    paths = (base_path, validation_path, *(path for _, path in source_specs))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input CSV does not exist: {path}")

    labels = [str(i) for i in range(10)]
    feature_columns = [f"f{i}" for i in range(1, 64)]
    core_columns = ["label", *feature_columns]
    base_df = pd.read_csv(base_path, dtype={"label": str})
    val_df = pd.read_csv(validation_path, dtype={"label": str})
    for name, frame in (("Base TRAIN", base_df), ("Validation", val_df)):
        missing = [column for column in core_columns if column not in frame.columns]
        if missing or frame.empty or not frame["label"].isin(labels).all():
            raise ValueError(f"{name} has missing columns, no rows, or labels outside 0-9: {missing}")
        if not np.isfinite(frame[feature_columns].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} has NaN/Inf features")

    seed = config["random_seed"]
    if type(seed) is not int:
        raise ValueError("random_seed must be an integer")
    selected_frames = []
    source_details = []
    for source, path in source_specs:
        name = source["name"]
        frame = pd.read_csv(path, dtype={"label": str})
        required = ["sample_id", "source_path", *core_columns]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if frame["sample_id"].isna().any() or not frame["sample_id"].is_unique:
            raise ValueError(f"{name} requires unique non-null sample_id values")

        numeric_features = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
        finite = np.isfinite(numeric_features.to_numpy(dtype=float)).all(axis=1)
        eligible_mask = frame["label"].isin(labels) & finite
        for column, value in source.get("filters", {}).items():
            if column not in frame.columns:
                raise ValueError(f"{name} missing filter column: {column}")
            if column in ("width", "height"):
                eligible_mask &= pd.to_numeric(frame[column], errors="coerce").eq(value)
            elif column == "handedness":
                eligible_mask &= frame[column].eq(value)
            else:
                raise ValueError(f"Unsupported filter column for {name}: {column}")
        eligible_before_dedup = frame.loc[eligible_mask]
        # Deduplicate only within this external source's eligible pool, in source CSV
        # order. Keep the first exact f1..f63 row; never deduplicate base TRAIN.
        eligible_features = numeric_features.loc[eligible_mask]
        duplicate_mask = eligible_features.duplicated(
            subset=feature_columns, keep="first"
        )
        if duplicate_mask.any():
            duplicate_features = eligible_features.loc[
                eligible_features.duplicated(subset=feature_columns, keep=False)
            ].copy()
            duplicate_features["label"] = eligible_before_dedup.loc[
                duplicate_features.index, "label"
            ]
            if (duplicate_features.groupby(feature_columns)["label"].nunique() > 1).any():
                raise ValueError(f"{name} has identical features with conflicting labels")
        duplicate_rows = eligible_before_dedup.loc[duplicate_mask]
        eligible = eligible_before_dedup.loc[~duplicate_mask]
        selected_parts = []
        methods = {}
        derived_seeds = {}
        for label, count in source["samples_per_class"].items():
            pool = eligible.loc[eligible["label"] == label]
            if len(pool) < count:
                raise ValueError(f"{name} class {label}: {len(pool)} eligible, need {count}")
            part, method, derived_seed = sample_class(
                pool, count, seed, name, label, source.get("group_column")
            )
            selected_parts.append(part)
            methods[label] = method
            derived_seeds[label] = derived_seed
        selected = pd.concat(selected_parts).sort_index(kind="stable")
        if not np.isfinite(selected[feature_columns].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} selected non-finite features")
        selected_frames.append(selected[core_columns])
        group_column = source.get("group_column")
        source_details.append({
            "name": name,
            "path": str(path),
            "sha256": sha256_file(path),
            "total_rows": int(len(frame)),
            "filters": source.get("filters", {}),
            "eligible_rows_before_dedup": int(len(eligible_before_dedup)),
            "exact_duplicate_feature_rows_removed": int(len(duplicate_rows)),
            "exact_duplicate_feature_rows_removed_per_class": class_counts(duplicate_rows, labels),
            "dedup_rule": "Within this external source after filters; exact f1..f63; keep first CSV row",
            "eligible_rows": int(len(eligible)),
            "eligible_per_class": class_counts(eligible, labels),
            "samples_per_class_requested": source["samples_per_class"],
            "selected_rows": int(len(selected)),
            "selected_per_class": class_counts(selected, labels),
            "group_column": group_column,
            "group_sampling_method_per_class": methods,
            "derived_sampling_seed_per_class": derived_seeds,
            "selected_group_counts": (
                {str(key): int(value) for key, value in selected[group_column].fillna("UNKNOWN").value_counts().items()}
                if group_column and group_column in selected.columns else {}
            ),
            "selected_samples": [
                {
                    "source_row_index": int(index),
                    "sample_id": str(row["sample_id"]),
                    "label": str(row["label"]),
                    "source_path": str(row["source_path"]),
                    "source_group": str(row[group_column]) if group_column and group_column in selected.columns else "",
                }
                for index, row in selected.iterrows()
            ],
        })

    combined = pd.concat([base_df[core_columns], *selected_frames], ignore_index=True)
    if not np.isfinite(combined[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("Combined TRAIN contains NaN/Inf features")
    selection = {
        "base_train_data": str(base_path),
        "base_train_sha256": sha256_file(base_path),
        "base_train_samples": int(len(base_df)),
        "base_train_per_class": class_counts(base_df, labels),
        "external_sources": source_details,
        "external_selected_total": int(sum(item["selected_rows"] for item in source_details)),
        "combined_train_samples": int(len(combined)),
        "combined_train_per_class": class_counts(combined, labels),
        "validation_data": str(validation_path),
        "validation_sha256": sha256_file(validation_path),
        "validation_samples": int(len(val_df)),
        "validation_per_class": class_counts(val_df, labels),
        "random_seed": seed,
        "feature_columns": feature_columns,
        "test_set_used": False,
        "source_group_independence_verified": False,
        "source_group_note": "Filename groups guide sampling; they do not prove participant identity or independent gestures",
    }
    return combined, val_df, selection


def load_frozen_replay_data(config):
    """Verify permanent frozen inputs; never read or resample external sources."""
    import numpy as np
    import pandas as pd

    expected_paths = {
        "train_data": FROZEN_TRAIN,
        "validation_data": FROZEN_VALIDATION,
        "sample_map": FROZEN_SAMPLE_MAP,
        "freeze_manifest": FROZEN_MANIFEST,
        "replay_spec": FROZEN_REPLAY_SPEC,
        "lineage_manifest": FROZEN_LINEAGE,
    }
    for key, expected in expected_paths.items():
        if project_path(config.get(key, "")) != expected or not expected.is_file():
            raise ValueError(f"Frozen replay requires permanent {key}: {expected}")

    with FROZEN_MANIFEST.open("r", encoding="utf-8") as file:
        freeze = json.load(file)
    sidecar = project_path(freeze["manifest_sha256_sidecar"])
    if sidecar != FROZEN_MANIFEST.with_suffix(".sha256") or not sidecar.is_file():
        raise ValueError("Frozen manifest SHA256 sidecar is missing or misplaced")
    sidecar_parts = sidecar.read_text(encoding="ascii").split()
    if (len(sidecar_parts) != 2 or sidecar_parts[0] != sha256_file(FROZEN_MANIFEST)
            or sidecar_parts[1] != FROZEN_MANIFEST.name):
        raise ValueError("Frozen manifest SHA256 mismatch")
    if freeze.get("status") != "DIGITS_FINAL_DATA_FROZEN_NOT_PROMOTED":
        raise ValueError("Unexpected frozen dataset status")

    output_records = freeze["output_files"]
    if len({record["path"] for record in output_records}) != len(output_records):
        raise ValueError("Duplicate path in frozen output manifest")
    records = {record["path"]: record for record in output_records}
    for record in output_records:
        # The regression reference is historical evidence, never a training
        # input or model-selection signal; do not even read its CSV here.
        if record["role"] == "regression_reference_not_independent_test":
            continue
        path = project_path(record["path"])
        if not path.is_relative_to(PROJECT_ROOT) or not path.is_file():
            raise ValueError(f"Missing or external frozen artifact: {path}")
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Frozen artifact size/SHA256 mismatch: {path}")
    for path in expected_paths.values():
        if path != FROZEN_MANIFEST and path.relative_to(PROJECT_ROOT).as_posix() not in records:
            raise ValueError(f"Frozen manifest does not cover required file: {path}")
    transformer_snapshot = "data/metadata/digits/digits_final_repro/hand_centric_2d.py"
    if (transformer_snapshot not in records
            or sha256_file(ORIENTATION_MODULE) != records[transformer_snapshot]["sha256"]):
        raise ValueError("Active hand-centric transformer differs from frozen snapshot")

    with FROZEN_REPLAY_SPEC.open("r", encoding="utf-8") as file:
        spec = json.load(file)
    with FROZEN_LINEAGE.open("r", encoding="utf-8") as file:
        lineage = json.load(file)
    if (config["experiment_name"] != "digits_final_replay_v1"
            or config["feature_count"] != 63
            or config["random_seed"] != 42 or spec["random_seed"] != 42
            or config["selection_metric"] != "validation_macro_f1"
            or spec["selection_metric"] != "validation_macro_f1"
            or config["preprocessing"] != {"id": "hand_centric_2d_v1", "axis_epsilon": 1e-6}
            or config["preprocessing"] != {"id": spec["preprocessing"]["id"],
                                           "axis_epsilon": spec["preprocessing"]["axis_epsilon"]}
            or spec["pipeline_steps"] != ["orientation_2d", "scaler", "mlp"]
            or config["training"] != spec["training_parameters"]
            or config["training"] != lineage["training_parameters"]
            or config["candidates"] != spec["candidate_parameters"]
            or config["candidates"] != lineage["candidate_parameters"]
            or list(config["candidates"]) != ["mlp_large_v2a"]
            or lineage["best_candidate"] != "mlp_large_v2a"
            or spec["feature_columns"] != [f"f{i}" for i in range(1, 64)]
            or spec["regression_is_independent_final_test"] is not False
            or lineage["test_set_used"] is not False):
        raise ValueError("Frozen replay configuration differs from original digits_final training")
    if (project_path(spec["train_data"]) != FROZEN_TRAIN
            or project_path(spec["validation_reference"]) != FROZEN_VALIDATION
            or project_path(spec["source_code"]["transformer"]) != ORIENTATION_MODULE.resolve()):
        raise ValueError("Replay specification points outside permanent inputs")

    features = [f"f{i}" for i in range(1, 64)]
    train_df = pd.read_csv(FROZEN_TRAIN, dtype={"label": str})
    val_df = pd.read_csv(FROZEN_VALIDATION, dtype={"label": str})
    sample_map = pd.read_csv(FROZEN_SAMPLE_MAP, dtype={
        "label": str, "source": str, "original_sample_id": str, "sample_id": str,
    })
    if (list(train_df.columns) != ["label", *features]
            or len(train_df) != 549 or len(sample_map) != 549
            or freeze["train_samples"] != 549
            or list(sample_map["frozen_row_index"]) != list(range(549))
            or not sample_map["sample_id"].is_unique
            or sample_map["sample_id"].isna().any()
            or sample_map["label"].tolist() != train_df["label"].tolist()):
        raise ValueError("Frozen TRAIN schema, IDs or row order differ")
    labels = [str(i) for i in range(10)]
    if (class_counts(train_df, labels) != freeze["class_counts"]
            or not train_df["label"].isin(labels).all()
            or any(column not in val_df.columns for column in ["label", *features])
            or len(val_df) != 100
            or lineage["data_sources_and_sampling"]["validation_samples"] != 100
            or not val_df["label"].isin(labels).all()):
        raise ValueError("Frozen TRAIN or validation labels/counts differ")
    train_features = train_df[features].to_numpy(dtype=np.float64)
    val_features = val_df[features].to_numpy(dtype=np.float64)
    if not np.isfinite(train_features).all() or not np.isfinite(val_features).all():
        raise ValueError("Frozen TRAIN or validation has NaN/Inf features")
    source_counts = sample_map["source"].value_counts().to_dict()
    if source_counts != {"base_train_v1": 349, "ardamavi": 170, "asl_hg": 30}:
        raise ValueError("Frozen source composition differs")
    ordered = "\n".join(
        f"{row.source}|{row.original_sample_id}|{row.label}"
        for row in sample_map.itertuples(index=False)
    )
    ordered_digest = hashlib.sha256(ordered.encode("utf-8")).hexdigest()
    if (ordered_digest != freeze["ordered_source_id_label_sha256"]
            or ordered_digest != spec["ordered_source_id_label_sha256"]):
        raise ValueError("Frozen ordered sample digest mismatch")
    row_hashes = [hashlib.sha256(np.asarray(row, dtype="<f8").tobytes()).hexdigest()
                  for row in train_features]
    if row_hashes != sample_map["feature_float64_sha256"].tolist():
        raise ValueError("Frozen sample-to-feature mapping differs")

    selection = {
        "data_mode": "frozen_replay",
        "frozen_train_data": str(FROZEN_TRAIN),
        "frozen_train_sha256": records[FROZEN_TRAIN.relative_to(PROJECT_ROOT).as_posix()]["sha256"],
        "sample_map": str(FROZEN_SAMPLE_MAP),
        "sample_map_sha256": records[FROZEN_SAMPLE_MAP.relative_to(PROJECT_ROOT).as_posix()]["sha256"],
        "freeze_manifest": str(FROZEN_MANIFEST),
        "freeze_manifest_sha256": sha256_file(FROZEN_MANIFEST),
        "ordered_source_id_label_sha256": ordered_digest,
        "combined_train_samples": len(train_df),
        "combined_train_per_class": class_counts(train_df, labels),
        "source_counts": source_counts,
        "validation_data": str(FROZEN_VALIDATION),
        "validation_sha256": records[FROZEN_VALIDATION.relative_to(PROJECT_ROOT).as_posix()]["sha256"],
        "validation_samples": len(val_df),
        "random_seed": config["random_seed"],
        "feature_columns": features,
        "test_set_used": False,
        "regression_reference_used_for_training_or_selection": False,
    }
    return train_df, val_df, selection


def run_training(config, train_df, val_df, selection, output_dir):
    """Fit candidates only after main() receives explicit user confirmation."""
    import joblib
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    preprocessing = config.get("preprocessing", {"id": "none"})
    preprocessing_contract = {"id": preprocessing["id"]}
    if preprocessing["id"] == "hand_centric_2d_v1":
        import sys

        if not ORIENTATION_MODULE.is_file():
            raise FileNotFoundError(f"Hand-centric transformer missing: {ORIENTATION_MODULE}")
        if str(PREPROCESSING_DIR) not in sys.path:
            sys.path.insert(0, str(PREPROCESSING_DIR))
        from hand_centric_2d import HandCentric2DAligner

        preprocessing_contract.update({
            "axis_epsilon": preprocessing["axis_epsilon"],
            "module_path": str(ORIENTATION_MODULE),
            "module_sha256": sha256_file(ORIENTATION_MODULE),
            "pipeline_steps": ["orientation_2d", "scaler", "mlp"],
        })
    else:
        preprocessing_contract["pipeline_steps"] = ["scaler", "mlp"]

    feature_columns = [f"f{i}" for i in range(1, 64)]
    X_train = train_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy()
    X_val = val_df[feature_columns].to_numpy(dtype=float)
    y_val = val_df["label"].to_numpy()

    report_path = output_dir / config["report_filename"]
    manifest_path = output_dir / config["manifest_filename"]
    target_files = [report_path, manifest_path] + [
        output_dir / f"{name}.joblib" for name in config["candidates"]
    ]
    existing = [str(path) for path in target_files if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing artifacts: " + ", ".join(existing))
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    best_name = None
    best_macro_f1 = -1.0
    best_accuracy = -1.0
    shared = config["training"]

    for name, candidate in config["candidates"].items():
        print(f"\n====================================\nTRAINING: {name}\n====================================")
        steps = []
        if preprocessing["id"] == "hand_centric_2d_v1":
            steps.append(("orientation_2d", HandCentric2DAligner(
                axis_epsilon=preprocessing["axis_epsilon"]
            )))
        steps.extend([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=tuple(candidate["hidden_layer_sizes"]),
                activation=shared["activation"],
                solver=shared["solver"],
                alpha=candidate["alpha"],
                batch_size=shared["batch_size"],
                learning_rate_init=candidate["learning_rate_init"],
                max_iter=shared["max_iter"],
                early_stopping=shared["early_stopping"],
                validation_fraction=shared["validation_fraction"],
                n_iter_no_change=shared["n_iter_no_change"],
                random_state=shared["random_state"],
            )),
        ])
        pipeline = Pipeline(steps)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro")
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

        model_path = output_dir / f"{name}.joblib"
        joblib.dump(pipeline, model_path)
        mlp = pipeline.named_steps["mlp"]
        result = {
            "name": name,
            "model_path": str(model_path),
            "hidden_layer_sizes": list(candidate["hidden_layer_sizes"]),
            "alpha": candidate["alpha"],
            "learning_rate_init": candidate["learning_rate_init"],
            "iterations": int(mlp.n_iter_),
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "classification_report": report,
        }
        results.append(result)
        print(f"Iterations : {mlp.n_iter_}")
        print(f"Accuracy   : {accuracy:.4f}")
        print(f"Macro F1   : {macro_f1:.4f}")

        # Same selection rule as the legacy script: macro-F1, then accuracy.
        if macro_f1 > best_macro_f1 or (macro_f1 == best_macro_f1 and accuracy > best_accuracy):
            best_name = name
            best_macro_f1 = macro_f1
            best_accuracy = accuracy

    comparison = {
        "version": config["report_version"],
        "training_samples": int(len(train_df)),
        "validation_samples": int(len(val_df)),
        "feature_count": 63,
        "selection_metric": "validation_macro_f1",
        "test_set_used": False,
        "best_candidate": best_name,
        "best_validation_macro_f1": float(best_macro_f1),
        "best_validation_accuracy": float(best_accuracy),
        "candidates": results,
        "preprocessing": preprocessing_contract,
        "data_sources_and_sampling": selection,
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=4, ensure_ascii=False)

    manifest = {
        "experiment_name": config["experiment_name"],
        "data_sources_and_sampling": selection,
        "training_parameters": config["training"],
        "candidate_parameters": config["candidates"],
        "preprocessing": preprocessing_contract,
        "selection_metric": config["selection_metric"],
        "best_candidate": best_name,
        "model_files_sha256": {
            name: sha256_file(output_dir / f"{name}.joblib") for name in config["candidates"]
        },
        "report_path": str(report_path),
        "test_set_used": False,
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4, ensure_ascii=False)

    print("\n====================================\nTRAIN CANDIDATES COMPLETE\n====================================")
    for result in results:
        print(f"{result['name']}: accuracy={result['accuracy']:.4f}, macro_f1={result['macro_f1']:.4f}")
    print(f"\nBEST CANDIDATE : {best_name}")
    print(f"BEST MACRO F1  : {best_macro_f1:.4f}")
    print(f"BEST ACCURACY  : {best_accuracy:.4f}")
    print("\nTEST SET USED  : False\n")
    print("Comparison report:")
    print(report_path)
    print("Training manifest:")
    print(manifest_path)


def run_frozen_replay(config, args, parser):
    """User-confirmed replay from immutable permanent inputs only."""
    forbidden_overrides = {
        "--train-data/--base-train-data": args.base_train_data,
        "--external-data": args.external_data,
        "--external-samples-per-class": args.external_samples_per_class,
        "--random-seed": args.random_seed,
        "--validation-data": args.validation_data,
    }
    used = [name for name, value in forbidden_overrides.items() if value is not None]
    if used:
        parser.error("Frozen replay forbids input/sampling overrides: " + ", ".join(used))
    if config.get("external_sources") or config.get("ablation_reference_manifest"):
        parser.error("Frozen replay config must not reference external sampling or a v2e manifest")
    for key in ("report_filename", "manifest_filename"):
        name = config.get(key, "")
        if not name or Path(name).name != name or not name.endswith(".json"):
            parser.error(f"{key} must be a JSON basename")

    output_dir = project_path(args.output_dir or config["output_dir"])
    if output_dir == REPLAY_OUTPUT_ROOT or not output_dir.is_relative_to(REPLAY_OUTPUT_ROOT):
        parser.error(f"Replay output must be a separate directory under {REPLAY_OUTPUT_ROOT}")
    if output_dir == PRODUCTION_DIR or output_dir.is_relative_to(PRODUCTION_DIR):
        parser.error("Production model directory cannot be a replay output")
    planned = [
        output_dir / config["report_filename"],
        output_dir / config["manifest_filename"],
        *(output_dir / f"{name}.joblib" for name in config["candidates"]),
    ]
    existing = [str(path) for path in planned if path.exists() or path.is_symlink()]
    if existing:
        parser.error("Refusing to overwrite existing replay artifacts: " + ", ".join(existing))

    print("Frozen digits_final replay plan:")
    print(f"  TRAIN: {FROZEN_TRAIN}")
    print(f"  Validation: {FROZEN_VALIDATION}")
    print(f"  Sample map: {FROZEN_SAMPLE_MAP}")
    print(f"  Output: {output_dir}")
    print(f"  Candidate: {config['candidates']}")
    print(f"  Training: {config['training']}")
    print("  Regression reference: NOT used for training or selection")
    if not args.check_only:
        try:
            answer = input("Verify frozen replay inputs? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer != "y":
            print("Replay cancelled; no data or model was written.")
            return 0
    try:
        train_df, val_df, selection = load_frozen_replay_data(config)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        parser.error(f"Frozen replay verification failed: {error}")
    print(f"  Verified TRAIN: {selection['combined_train_samples']} samples")
    print("  Class counts: " + format_class_counts(selection["combined_train_per_class"], [str(i) for i in range(10)]))
    print(f"  Ordered sample digest: {selection['ordered_source_id_label_sha256']}")
    print(f"  Verified validation: {selection['validation_samples']} samples")
    if args.check_only:
        print("CHECK_ONLY: PASS; no sampling or training performed.")
        return 0
    try:
        answer = input("Start training? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer != "y":
        print("Training cancelled; no model was written.")
        return 0
    run_training(config, train_df, val_df, selection, output_dir)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train digit MLP candidates after confirmation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-train-data", "--train-data", dest="base_train_data", type=Path,
                        help="Override config base TRAIN CSV")
    parser.add_argument("--external-data", type=Path, help="Override first external source CSV")
    parser.add_argument("--external-samples-per-class", type=int,
                        help="Override per-class sample count for the first external source")
    parser.add_argument("--random-seed", type=int, help="Override external sampling seed")
    parser.add_argument("--validation-data", type=Path, help="Override config validation CSV")
    parser.add_argument("--output-dir", type=Path, help="Override config output directory")
    parser.add_argument("--check-only", action="store_true",
                        help="Verify frozen replay hashes, schema and order; do not train or write")
    args = parser.parse_args(argv)

    config_path = project_path(args.config)
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if config.get("data_mode") == "frozen_replay":
        return run_frozen_replay(config, args, parser)
    if args.check_only:
        parser.error("--check-only is supported only for frozen replay")
    if config.get("feature_count") != 63 or config.get("selection_metric") != "validation_macro_f1":
        parser.error("Config must use 63 features and validation macro-F1 selection")
    preprocessing = config.get("preprocessing", {"id": "none"})
    if not isinstance(preprocessing, dict) or preprocessing.get("id") not in (
        "none", "hand_centric_2d_v1"
    ):
        parser.error("Unsupported preprocessing id")
    if preprocessing["id"] == "hand_centric_2d_v1":
        epsilon = preprocessing.get("axis_epsilon")
        if (type(epsilon) not in (int, float) or not math.isfinite(epsilon)
                or epsilon <= 0):
            parser.error("axis_epsilon must be a finite positive number")
        if not config.get("ablation_reference_manifest"):
            parser.error("Orientation ablation requires a v2e reference manifest")
    for key in ("report_filename", "manifest_filename"):
        name = config.get(key, "")
        if not name or Path(name).name != name or not name.endswith(".json"):
            parser.error(f"{key} must be a JSON basename, not a path")
    if not config.get("candidates") or any(
        not name.isidentifier() for name in config["candidates"]
    ):
        parser.error("Candidate names must be nonempty safe identifiers")
    sources = config.get("external_sources")
    if not isinstance(sources, list) or not sources:
        parser.error("external_sources must be a nonempty list")
    names = []
    for source in sources:
        if not isinstance(source, dict) or not source.get("name") or not source.get("path"):
            parser.error("Each external source needs name and path")
        requested = source.get("samples_per_class")
        if (not isinstance(requested, dict) or not requested
                or any(label not in {str(i) for i in range(10)} or type(count) is not int or count <= 0
                       for label, count in requested.items())):
            parser.error(f"Invalid per-class sample counts for {source['name']}")
        names.append(source["name"])
    if len(set(names)) != len(names):
        parser.error("External source names must be unique")

    if args.external_data is not None:
        sources[0]["path"] = str(args.external_data)
    if args.external_samples_per_class is not None:
        if args.external_samples_per_class <= 0:
            parser.error("--external-samples-per-class must be positive")
        sources[0]["samples_per_class"] = {
            label: args.external_samples_per_class for label in sources[0]["samples_per_class"]
        }
    if args.random_seed is not None:
        config["random_seed"] = args.random_seed

    base_train_data = project_path(args.base_train_data or config["base_train_data"])
    source_specs = [(source, project_path(source["path"])) for source in sources]
    validation_data = project_path(args.validation_data or config["validation_data"])
    output_dir = project_path(args.output_dir or config["output_dir"])
    reference_manifest = None
    reference_path = None
    if preprocessing["id"] == "hand_centric_2d_v1":
        reference_path = project_path(config["ablation_reference_manifest"])
        if not reference_path.is_file():
            parser.error(f"Ablation reference manifest does not exist: {reference_path}")
        with reference_path.open("r", encoding="utf-8") as file:
            reference_manifest = json.load(file)
        reference_selection = reference_manifest.get("data_sources_and_sampling", {})
        reference_candidate = reference_manifest.get("candidate_parameters", {}).get("mlp_large_v2a")
        if (reference_manifest.get("experiment_name") != "digits_v2e"
                or config["training"] != reference_manifest.get("training_parameters")
                or config["candidates"] != {"mlp_large_v2a": reference_candidate}):
            parser.error("Ablation must use only the v2e winner and unchanged hyperparameters")
        reference_sources = reference_selection.get("external_sources", [])
        if (config["random_seed"] != reference_selection.get("random_seed")
                or str(base_train_data) != reference_selection.get("base_train_data")
                or str(validation_data) != reference_selection.get("validation_data")
                or len(source_specs) != len(reference_sources)):
            parser.error("Ablation seed, base, validation or source count differs from v2e")
        for (source, path), original in zip(source_specs, reference_sources):
            if (source["name"] != original["name"]
                    or str(path) != original["path"]
                    or source["samples_per_class"] != original["samples_per_class_requested"]
                    or source.get("filters", {}) != original["filters"]
                    or source.get("group_column") != original["group_column"]):
                parser.error(f"Ablation sampling rules differ from v2e for {source['name']}")
    input_paths = [base_train_data, validation_data, *(path for _, path in source_specs)]
    if OFFICIAL_TEST in input_paths:
        parser.error("The official test_v1.csv cannot be used for training, sampling or validation")
    if len(set(input_paths)) != len(input_paths):
        parser.error("Base, validation and external source paths must be distinct")
    if output_dir == PRODUCTION_DIR or PRODUCTION_DIR in output_dir.parents:
        parser.error("Production model directory cannot be a training output")
    planned_outputs = [
        output_dir / config["report_filename"],
        output_dir / config["manifest_filename"],
        *(output_dir / f"{name}.joblib" for name in config["candidates"]),
    ]
    existing = [str(path) for path in planned_outputs if path.exists() or path.is_symlink()]
    if existing:
        parser.error("Refusing to overwrite existing artifacts: " + ", ".join(existing))

    print("Training plan:")
    print(f"  Experiment: {config['experiment_name']}")
    print(f"  Base TRAIN: {base_train_data}")
    for source, path in source_specs:
        labels = sorted(source["samples_per_class"], key=int)
        print(
            f"  External {source['name']}: {path} "
            f"({format_class_counts(source['samples_per_class'], labels)})"
        )
    print(f"  Validation: {validation_data}")
    print(f"  Output: {output_dir}")
    print(f"  Seed: {config['random_seed']}; features: {config['feature_count']}; "
          f"selection: {config['selection_metric']}")
    print(f"  Preprocessing: {preprocessing['id']}")
    if reference_path is not None:
        print(f"  Ablation reference: {reference_path}")
    print(f"  Training: {config['training']}")
    for name, candidate in config["candidates"].items():
        print(f"  Candidate {name}: {candidate}")
    try:
        answer = input("Prepare source sampling in memory? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer != "y":
        print("Preparation cancelled; no dataset or model was written.")
        return 0

    train_df, val_df, selection = prepare_training_data(
        config, base_train_data, source_specs, validation_data
    )
    if reference_manifest is not None:
        if selection != reference_manifest["data_sources_and_sampling"]:
            raise ValueError("Selected training data differs from digits_v2e; training stopped")
        selection["ablation_control"] = {
            "reference_manifest_path": str(reference_path),
            "reference_manifest_sha256": sha256_file(reference_path),
            "exact_v2e_selection_match": True,
        }
    print_prepared_summary(config, selection)
    try:
        answer = input("Start training? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer != "y":
        print("Training cancelled; no dataset or model was written.")
        return 0

    run_training(config, train_df, val_df, selection, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
