"""Train exactly two U/V/W-boundary MLP candidates and compare on VALIDATION v1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from . import train_mlp_v2_validation as common
except ImportError:  # Direct execution: python AI/train_uvw_validation.py
    import train_mlp_v2_validation as common


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ai/dataset/mlp_uvw_training_data_v1"
SPLIT_DIR = ROOT / "ai/dataset/train_validation_test_split_v1"
V1_DIR = ROOT / "ai/dataset/mlp_validation_comparison_v1"
OUTPUT_DIR = ROOT / "ai/dataset/mlp_uvw_validation_comparison_v1"
FEATURES = common.FEATURES
LABELS = common.LABELS
LABEL_TO_ID = common.LABEL_TO_ID
SEED = common.SEED
HYPERPARAMETERS = common.HYPERPARAMETERS
VARIANTS = {
    "uvw_low": "weight_supplement_low",
    "uvw_medium": "weight_supplement_medium",
}
SIGNIFICANT_RECALL_DROP = 0.02


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload, replace=False):
    with Path(path).open("w" if replace else "x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, payload, replace=True)
    os.replace(temporary, path)


def atomic_joblib(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, temporary, compress=3)
    os.replace(temporary, path)


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def signature(path):
    path = Path(path)
    return {"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_csv(path, rows, fields=None):
    require(rows, f"No rows for {path}")
    fields = fields or list(rows[0])
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_inputs():
    fit_signature = inspect.signature(MLPClassifier.fit)
    partial_signature = inspect.signature(MLPClassifier.partial_fit)
    require("sample_weight" in fit_signature.parameters and
            "sample_weight" in partial_signature.parameters,
            "Installed MLPClassifier lacks native sample_weight")

    data_manifest_path = DATA_DIR / "manifest.json"
    integrity_path = DATA_DIR / "integrity_report.json"
    data_manifest = load_json(data_manifest_path)
    integrity = load_json(integrity_path)
    require(data_manifest["status"] == "UVW_TRAIN_DATA_PREPARED", "UVW TRAIN not ready")
    require(integrity["status"] == "PASS" and all(integrity["checks"].values()),
            "UVW TRAIN integrity is not PASS")
    require(data_manifest["counts"]["train_new"] == 62648 and
            data_manifest["counts"]["supplement"] == 57, "UVW TRAIN count mismatch")
    require(data_manifest["features"] == FEATURES and
            data_manifest["labels"] == list(LABELS), "UVW feature/label contract mismatch")
    require(data_manifest["pq_data_used"] is False and
            data_manifest["training_performed"] is False, "UVW preparation scope mismatch")

    snapshots = {}
    for name, item in data_manifest["input_hashes"].items():
        path = ROOT / name
        require(path.is_file() and path.stat().st_size == item["bytes"] and
                sha256(path) == item["sha256"], f"Protected input changed: {name}")
        snapshots[name] = item["sha256"]
    for name, item in data_manifest["output_hashes"].items():
        path = ROOT / name
        require(path.is_file() and path.stat().st_size == item["bytes"] and
                sha256(path) == item["sha256"], f"UVW data artefact changed: {name}")
        snapshots[name] = item["sha256"]
    snapshots[relative(data_manifest_path)] = sha256(data_manifest_path)

    v1_manifest_path = V1_DIR / "training_manifest.json"
    v1_manifest = load_json(v1_manifest_path)
    require(v1_manifest["status"] == "VALIDATION_COMPARISON_COMPLETE" and
            v1_manifest["selection"]["selected_model"] == "baseline" and
            v1_manifest["integrity"]["status"] == "PASS", "v1 baseline manifest mismatch")
    for item in v1_manifest["outputs"]:
        path = V1_DIR / item["path"]
        require(path.is_file() and path.stat().st_size == item["bytes"] and
                sha256(path) == item["sha256"], f"v1 artefact changed: {path}")
        snapshots[relative(path)] = item["sha256"]
    snapshots[relative(v1_manifest_path)] = sha256(v1_manifest_path)

    v1_config = load_json(V1_DIR / "training_config.json")
    validation_path = SPLIT_DIR / "validation.csv"
    test_path = SPLIT_DIR / "test.csv"
    for path in (validation_path, test_path):
        expected = v1_config["input_snapshots"][relative(path)]
        require(path.is_file() and sha256(path) == expected, f"Frozen split changed: {path}")
        snapshots[relative(path)] = expected
    test_audit = {
        "path": relative(test_path), "verification": "SIZE_AND_SHA256_ONLY",
        "csv_opened_or_parsed": False, "used_for_training": False,
        "used_for_model_selection": False, "evaluated": False,
    }
    return snapshots, test_audit, fit_signature, partial_signature


def load_data():
    expected_columns = ["source", "file", "label", "handedness", *FEATURES]
    train = pd.read_csv(DATA_DIR / "train_uvw_v1.csv")
    metadata = pd.read_csv(DATA_DIR / "train_uvw_sample_metadata.csv",
                           dtype=str, keep_default_na=False)
    weights = pd.read_csv(DATA_DIR / "train_uvw_weights.csv")
    validation = pd.read_csv(SPLIT_DIR / "validation.csv")
    require(train.columns.tolist() == expected_columns and
            validation.columns.tolist() == expected_columns, "Dataset schema mismatch")
    require(len(train) == len(metadata) == len(weights) == 62648 and
            len(validation) == 13412, "Dataset row count mismatch")
    require(train.file.nunique() == len(train) and validation.file.nunique() == len(validation),
            "Duplicate file ID")
    require(not set(train.file) & set(validation.file), "TRAIN/VALIDATION sample overlap")
    require(set(train.label) == set(validation.label) == set(LABELS), "Label set mismatch")
    require(metadata.v3_row_id.tolist() == weights.v3_row_id.tolist(), "Weight row IDs mismatch")
    for field in ("file", "label"):
        require(train[field].astype(str).tolist() == metadata[field].astype(str).tolist(),
                f"Metadata alignment mismatch: {field}")
    require(metadata.label.tolist() == weights.label.astype(str).tolist(), "Weight label mismatch")
    require(metadata.group_id.tolist() == weights.group_id.astype(str).tolist(),
            "Weight group mismatch")
    supplement = metadata.origin.ne("TRAIN_V1")
    require((~supplement).sum() == 62591 and supplement.sum() == 57,
            "TRAIN/supplement provenance mismatch")
    require(set(metadata.loc[supplement, "label"]) == {"U", "V"},
            "Unauthorized supplement labels")
    require(not metadata.loc[supplement, "origin"].str.contains("PQ", case=False).any(),
            "P/Q supplement detected")

    assignments = pd.read_csv(SPLIT_DIR / "split_assignments.csv", dtype=str,
                              keep_default_na=False)
    validation_groups = set(assignments.loc[assignments.split.eq("VALIDATION"), "group_id"])
    require(not set(metadata.group_id) & validation_groups, "TRAIN/VALIDATION group overlap")
    x_train = train[FEATURES].to_numpy(dtype=np.float64, copy=True)
    x_validation = validation[FEATURES].to_numpy(dtype=np.float64, copy=True)
    require(np.isfinite(x_train).all() and np.isfinite(x_validation).all(), "Non-finite feature")
    y_train = train.label.map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    y_validation = validation.label.map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    return train, metadata, weights, validation, x_train, y_train, x_validation, y_validation


def audit_weights(metadata, weights, variant, column):
    values = weights[column].to_numpy(dtype=np.float64)
    require(np.isfinite(values).all() and (values > 0).all(), f"Invalid weights: {variant}")
    require(math.isclose(values.sum(), len(values), rel_tol=0.0, abs_tol=1e-6),
            f"Total weight mismatch: {variant}")
    class_target = len(values) / len(LABELS)
    rows = []
    for label in LABELS:
        mask = metadata.label.eq(label).to_numpy()
        total = float(values[mask].sum())
        require(math.isclose(total, class_target, rel_tol=0.0, abs_tol=1e-6),
                f"Class weight mismatch: {variant}/{label}")
        rows.append({"candidate": variant, "scope": "CLASS", "label": label,
                     "supplement_group": "", "group_id": "",
                     "sample_count": int(mask.sum()), "weight_total": total,
                     "fraction_of_class": total / class_target})

    expected = {
        "uvw_low": {"OLD_U": .02, "OLD_V": .01, "V_TO_U": .02, "V_TO_W": .02},
        "uvw_medium": {"OLD_U": .05, "OLD_V": .02, "V_TO_U": .04, "V_TO_W": .04},
    }[variant]
    supplement = metadata.origin.ne("TRAIN_V1")
    require(metadata.loc[supplement, "group_id"].nunique() == 5,
            "Conservative supplement group count mismatch")
    for supplement_group, group_frame in metadata.loc[supplement].groupby(
            "supplement_group", sort=True):
        group_ids = group_frame.group_id.unique()
        aggregate_total = 0.0
        for group_id, frame in group_frame.groupby("group_id", sort=True):
            indexes = frame.index.to_numpy()
            group_total = float(values[indexes].sum())
            expected_group_fraction = expected[supplement_group] / len(group_ids)
            require(math.isclose(group_total / class_target, expected_group_fraction,
                                 rel_tol=0.0, abs_tol=1e-9),
                    f"Supplement group budget mismatch: {variant}/{group_id}")
            require(np.ptp(values[indexes]) < 1e-15,
                    f"Unequal within-group weights: {variant}/{group_id}")
            aggregate_total += group_total
            rows.append({"candidate": variant, "scope": "SUPPLEMENT_SESSION_GROUP",
                         "label": frame.label.iloc[0],
                         "supplement_group": supplement_group, "group_id": group_id,
                         "sample_count": len(frame), "weight_total": group_total,
                         "fraction_of_class": group_total / class_target})
        require(math.isclose(aggregate_total / class_target, expected[supplement_group],
                             rel_tol=0.0, abs_tol=1e-9),
                f"Supplement aggregate budget mismatch: {variant}/{supplement_group}")
    return values, rows


def evaluate_training_diagnostic(model, scaler, train, metadata, x_train, candidate):
    mask = metadata.origin.ne("TRAIN_V1").to_numpy()
    features = scaler.transform(x_train[mask])
    true_labels = train.loc[mask, "label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    probabilities = model.predict_proba(features)
    predictions = np.argmax(probabilities, axis=1)
    rows = []
    for local_index, source_index in enumerate(np.flatnonzero(mask)):
        rows.append({
            "candidate": candidate, "sample_id": metadata.iloc[source_index].sample_id,
            "supplement_group": metadata.iloc[source_index].supplement_group,
            "group_id": metadata.iloc[source_index].group_id,
            "original_label": train.iloc[source_index].label,
            "predicted_label": LABELS[int(predictions[local_index])],
            "confidence": float(probabilities[local_index].max()),
            "correct": bool(predictions[local_index] == true_labels[local_index]),
            "scope": "TRAINING_ROWS_DIAGNOSTIC_NOT_INDEPENDENT",
        })
    summary = []
    frame = pd.DataFrame(rows)
    for group, part in frame.groupby("supplement_group", sort=True):
        summary.append({
            "candidate": candidate, "supplement_group": group, "samples": len(part),
            "correct": int(part.correct.sum()), "accuracy": float(part.correct.mean()),
            "prediction_counts": part.predicted_label.value_counts().sort_index().to_dict(),
            "scope": "TRAINING_ROWS_DIAGNOSTIC_NOT_INDEPENDENT",
        })
    return rows, summary


def compare_candidate(name, evaluation, v1_evaluation):
    v1_by_label = {row["label"]: row for row in v1_evaluation["per_class"]}
    current_by_label = {row["label"]: row for row in evaluation["per_class"]}
    deltas = []
    for label in LABELS:
        deltas.append({
            "candidate": name, "label": label,
            "v1_precision": v1_by_label[label]["precision"],
            "candidate_precision": current_by_label[label]["precision"],
            "precision_delta": current_by_label[label]["precision"] - v1_by_label[label]["precision"],
            "v1_recall": v1_by_label[label]["recall"],
            "candidate_recall": current_by_label[label]["recall"],
            "recall_delta": current_by_label[label]["recall"] - v1_by_label[label]["recall"],
            "v1_f1": v1_by_label[label]["f1"],
            "candidate_f1": current_by_label[label]["f1"],
            "f1_delta": current_by_label[label]["f1"] - v1_by_label[label]["f1"],
        })
    regressions = [row for row in deltas if row["recall_delta"] < -SIGNIFICANT_RECALL_DROP]
    return {
        "accuracy_delta": evaluation["metrics"]["accuracy"] - v1_evaluation["metrics"]["accuracy"],
        "macro_f1_delta": evaluation["metrics"]["macro_f1"] - v1_evaluation["metrics"]["macro_f1"],
        "significant_recall_regressions": regressions,
        "per_class_deltas": deltas,
    }


def configuration(snapshots, test_audit, fit_signature, partial_signature):
    return {
        "schema_version": 1, "seed": SEED, "features": FEATURES,
        "labels": list(LABELS), "architecture": [63, 64, 32, 24],
        "hyperparameters": HYPERPARAMETERS, "variants": VARIANTS,
        "baseline": "v1_baseline epoch 143",
        "v1_parity": {
            "architecture_unchanged": True, "optimizer_unchanged": True,
            "seed_unchanged": True, "early_stopping_unchanged": True,
            "scaler_change": "Each candidate has its own StandardScaler fit only on identical TRAIN UVW rows.",
        },
        "sample_weight_support": {
            "fit_signature": str(fit_signature), "partial_fit_signature": str(partial_signature),
            "supported": True, "fallback_used": False,
        },
        "selection": {
            "primary": "VALIDATION v1 macro-F1", "tie_breakers": ["minimum_class_recall", "accuracy"],
            "significant_recall_drop": SIGNIFICANT_RECALL_DROP,
        },
        "libraries": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "joblib": version("joblib"),
        },
        "input_snapshots": snapshots, "test_access": test_audit,
        "implementation": {
            "script": signature(Path(__file__).resolve()),
            "shared_training_logic": signature(ROOT / "training" / "letters" / "train_mlp_v2_validation.py"),
            "tests": signature(ROOT / "tests" / "test_train_uvw_validation.py"),
        },
        "training_constraints": {
            "pq_data_used": False, "test_used": False, "variants_trained": list(VARIANTS),
            "extra_variants_trained": False,
        },
    }


def check_only():
    snapshots, test_audit, fit_sig, partial_sig = verify_inputs()
    train, metadata, weights, validation, _, _, _, _ = load_data()
    audit_rows = []
    for variant, column in VARIANTS.items():
        _, rows = audit_weights(metadata, weights, variant, column)
        audit_rows.extend(rows)
    require(not OUTPUT_DIR.exists(), "Output exists; inspect/resume instead of check-only")
    print(json.dumps({
        "status": "CHECK_ONLY_PASS", "training_started": False,
        "train_samples": len(train), "validation_samples": len(validation),
        "features": len(FEATURES), "classes": len(LABELS),
        "weight_audit_rows": len(audit_rows), "input_hashes_verified": len(snapshots),
        "test_access": test_audit,
    }))


def train(output_dir):
    snapshots, test_audit, fit_sig, partial_sig = verify_inputs()
    config = configuration(snapshots, test_audit, fit_sig, partial_sig)
    manifest_path = output_dir / "training_manifest.json"
    if output_dir.exists() and manifest_path.exists():
        manifest = load_json(manifest_path)
        require(manifest["status"] == "UVW_VALIDATION_COMPARISON_COMPLETE",
                "Existing manifest status mismatch")
        for item in manifest["outputs"]:
            path = output_dir / item["path"]
            require(path.is_file() and path.stat().st_size == item["bytes"] and
                    sha256(path) == item["sha256"], f"Completed output changed: {path}")
        print(f"STATUS={manifest['status']}")
        print(f"RECOMMENDED_CAMERA_CANDIDATE={manifest['selection']['recommended_for_camera_test']}")
        return
    if output_dir.exists():
        require(load_json(output_dir / "training_config.json") == config,
                "Checkpoint configuration/input hashes differ")
    else:
        output_dir.mkdir()
        (output_dir / "checkpoints").mkdir()
        write_json(output_dir / "training_config.json", config)
        atomic_json(output_dir / "run_state.json",
                    {"status": "INPUTS_VERIFIED", "completed_variants": []})

    (train_frame, metadata, weight_frame, validation_frame, x_train, y_train,
     x_validation, y_validation) = load_data()
    weights_by_variant = {}
    weight_audit_rows = []
    for variant, column in VARIANTS.items():
        values, rows = audit_weights(metadata, weight_frame, variant, column)
        weights_by_variant[variant] = values
        weight_audit_rows.extend(rows)
    write_csv(output_dir / "pretraining_weight_audit.csv", weight_audit_rows)
    write_json(output_dir / "pretraining_weight_audit.json", {
        "status": "PASS", "train_rows": len(train_frame), "classes": len(LABELS),
        "conservative_supplement_groups": 5,
        "near_duplicate_policy": "Near-duplicate IDs retained; fixed subgroup budget is divided across conservative session groups and then rows.",
        "rows": weight_audit_rows,
    }, replace=True)

    config_hash = sha256(output_dir / "training_config.json")
    states = {}
    scalers = {}
    completed = []
    for variant in VARIANTS:
        scaler_path = output_dir / f"{variant}_scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            require(int(scaler.n_samples_seen_) == len(train_frame),
                    f"Scaler is not TRAIN-only: {variant}")
        else:
            scaler = StandardScaler().fit(x_train)
            require(int(scaler.n_samples_seen_) == len(train_frame),
                    f"Scaler fit count mismatch: {variant}")
            atomic_joblib(scaler_path, scaler)
        scalers[variant] = scaler
        states[variant] = common.train_variant(
            variant, scaler.transform(x_train), y_train,
            scaler.transform(x_validation), y_validation,
            weights_by_variant[variant], output_dir / "checkpoints" / variant,
            config_hash)
        completed.append(variant)
        atomic_json(output_dir / "run_state.json",
                    {"status": "TRAINING", "completed_variants": completed})
    require(np.array_equal(scalers["uvw_low"].mean_, scalers["uvw_medium"].mean_) and
            np.array_equal(scalers["uvw_low"].scale_, scalers["uvw_medium"].scale_),
            "Candidate scalers differ despite identical TRAIN rows")

    v1_model = joblib.load(V1_DIR / "baseline_model.joblib")
    v1_scaler = joblib.load(V1_DIR / "scaler.joblib")
    require(int(v1_scaler.n_samples_seen_) == 62591, "v1 scaler fit count mismatch")
    v1_metrics, v1_per_class, v1_matrix, _, _, _ = common.evaluate(
        v1_model, v1_scaler.transform(x_validation), y_validation, validation_frame)
    saved_v1 = load_json(V1_DIR / "evaluation_summary.json")["baseline"]
    require(abs(v1_metrics["accuracy"] - saved_v1["metrics"]["accuracy"]) < 1e-15 and
            abs(v1_metrics["macro_f1"] - saved_v1["metrics"]["macro_f1"]) < 1e-15,
            "Recomputed v1 baseline differs from official metrics")
    v1_evaluation = {"metrics": v1_metrics, "per_class": v1_per_class,
                     "best_epoch": saved_v1["best_epoch"]}

    evaluations = {}
    delta_rows = []
    diagnostic_rows = []
    diagnostic_summary = []
    uvw_rows = []
    for variant, state in states.items():
        model = state["best_model"]
        scaler = scalers[variant]
        require(np.array_equal(model.classes_, np.arange(len(LABELS))),
                f"Class order mismatch: {variant}")
        metrics, per_class, matrix, predictions, _, _ = common.evaluate(
            model, scaler.transform(x_validation), y_validation, validation_frame)
        evaluation = {
            "metrics": metrics, "per_class": per_class,
            "best_epoch": state["best_epoch"], "epochs_run": state["epoch"],
            "stop_reason": state["stop_reason"],
        }
        comparison = compare_candidate(variant, evaluation, v1_evaluation)
        evaluation["comparison_to_v1"] = {
            key: value for key, value in comparison.items() if key != "per_class_deltas"
        }
        evaluations[variant] = evaluation
        delta_rows.extend(comparison["per_class_deltas"])

        atomic_joblib(output_dir / f"{variant}_model.joblib", model)
        write_csv(output_dir / f"{variant}_training_history.csv", state["history"])
        write_csv(output_dir / f"{variant}_per_class_metrics.csv", per_class)
        write_csv(output_dir / f"{variant}_validation_predictions.csv", predictions)
        with (output_dir / f"{variant}_confusion_matrix.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true\\predicted", *LABELS])
            for label, row in zip(LABELS, matrix, strict=True):
                writer.writerow([label, *map(int, row)])

        current = {row["label"]: row for row in per_class}
        baseline = {row["label"]: row for row in v1_per_class}
        for label in ("U", "V", "W"):
            uvw_rows.append({
                "candidate": variant, "label": label,
                "v1_precision": baseline[label]["precision"],
                "candidate_precision": current[label]["precision"],
                "v1_recall": baseline[label]["recall"],
                "candidate_recall": current[label]["recall"],
                "recall_delta": current[label]["recall"] - baseline[label]["recall"],
                "v1_f1": baseline[label]["f1"],
                "candidate_f1": current[label]["f1"],
            })
        rows, summary = evaluate_training_diagnostic(
            model, scaler, train_frame, metadata, x_train, variant)
        diagnostic_rows.extend(rows)
        diagnostic_summary.extend(summary)

    write_csv(output_dir / "per_class_comparison_to_v1.csv", delta_rows)
    write_csv(output_dir / "uvw_class_comparison.csv", uvw_rows)
    write_csv(output_dir / "supplement_training_diagnostic_predictions.csv", diagnostic_rows)
    write_json(output_dir / "supplement_training_diagnostic_summary.json", {
        "disclaimer": "These 57 rows were used for training; results are diagnostic only and not independent evaluation evidence.",
        "groups": diagnostic_summary,
    }, replace=True)

    order = sorted(VARIANTS, key=lambda name: (
        evaluations[name]["metrics"]["macro_f1"],
        evaluations[name]["metrics"]["minimum_class_recall"],
        evaluations[name]["metrics"]["accuracy"], name == "uvw_low"), reverse=True)
    winner = order[0]
    regressions = evaluations[winner]["comparison_to_v1"]["significant_recall_regressions"]
    selection = {
        "validation_winner": winner,
        "recommended_for_camera_test": winner if not regressions else None,
        "status": "CAMERA_TEST_RECOMMENDED" if not regressions else "RECALL_REGRESSION_REVIEW_REQUIRED",
        "criterion": "Highest VALIDATION v1 macro-F1; ties use minimum recall then accuracy.",
        "significant_recall_drop_threshold": SIGNIFICANT_RECALL_DROP,
        "significant_recall_regressions": regressions,
        "deployment_authorized": False, "camera_error_fixed_claimed": False,
        "test_used": False,
    }

    shutil.copyfile(output_dir / f"{winner}_model.joblib",
                    output_dir / "validation_winner_model.joblib")
    shutil.copyfile(output_dir / f"{winner}_scaler.joblib",
                    output_dir / "validation_winner_scaler.joblib")
    write_json(output_dir / "label_mapping.json", {
        "classes_in_model_order": list(LABELS), "label_to_index": LABEL_TO_ID,
        "index_to_label": {str(index): label for label, index in LABEL_TO_ID.items()},
    }, replace=True)
    write_json(output_dir / "evaluation_summary.json", {
        "train_samples": len(train_frame), "validation_samples": len(validation_frame),
        "baseline_v1": v1_evaluation, **evaluations, "selection": selection,
        "test_access": test_audit,
        "limitations": [
            "VALIDATION v1 has no camera-view metadata and cannot prove V->U/W correction.",
            "The 57 supplement rows are training data and not independent evaluation evidence.",
        ],
    }, replace=True)
    write_json(output_dir / "selected_candidate.json", {
        "validation_winner": winner,
        "recommended_for_camera_test": selection["recommended_for_camera_test"],
        "model_path": "validation_winner_model.joblib",
        "scaler_path": "validation_winner_scaler.joblib",
        "source_model_path": f"{winner}_model.joblib",
        "source_scaler_path": f"{winner}_scaler.joblib",
        "label_mapping_path": "label_mapping.json",
        "deployment_authorized": False, "selection": selection,
    }, replace=True)

    reload_checks = []
    for variant in VARIANTS:
        model = joblib.load(output_dir / f"{variant}_model.joblib")
        scaler = joblib.load(output_dir / f"{variant}_scaler.joblib")
        probabilities = model.predict_proba(scaler.transform(x_validation[:16]))
        require(probabilities.shape == (16, len(LABELS)) and
                np.isfinite(probabilities).all() and
                np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12),
                f"Reload inference failed: {variant}")
        reload_checks.append({"candidate": variant, "samples": 16, "status": "PASS"})

    test_run = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "tests.test_train_uvw_validation", "-q"],
        cwd=ROOT, capture_output=True, text=True, check=False)
    require(test_run.returncode == 0, f"Unit tests failed: {test_run.stderr[-1000:]}")
    write_json(output_dir / "test_report.json", {
        "status": "PASS", "unit_tests_run": 5,
        "reload_inference_checks": reload_checks,
    }, replace=True)

    for name, expected in snapshots.items():
        require(sha256(ROOT / name) == expected, f"Protected input changed: {name}")
    integrity = {
        "status": "PASS", "checks_run": 29,
        "train_samples": len(train_frame), "validation_samples": len(validation_frame),
        "features_used": 63, "labels": 24,
        "train_validation_sample_overlap": 0, "train_validation_group_overlap": 0,
        "scaler_train_rows": {variant: int(scalers[variant].n_samples_seen_)
                              for variant in VARIANTS},
        "candidate_scalers_identical": True,
        "sample_weight_native": True, "weight_class_totals_equal": True,
        "supplement_group_budgets_pass": True, "near_duplicate_control": "PASS",
        "models_scalers_reload_inference": "PASS", "unit_tests": "PASS",
        "test_csv_access": "SIZE_AND_SHA256_ONLY", "test_evaluated": False,
        "pq_data_used": False, "extra_variants_trained": False,
        "protected_inputs_hashes_unchanged": True,
        "static_camera_modified": False, "model_v1_modified": False,
        "frozen_or_split_data_modified": False,
    }
    write_json(output_dir / "integrity_report.json", integrity, replace=True)
    atomic_json(output_dir / "run_state.json", {
        "status": "COMPLETE", "completed_variants": list(VARIANTS),
        "validation_winner": winner,
        "recommended_for_camera_test": selection["recommended_for_camera_test"],
    })

    lines = ["# UVW validation comparison", "",
             f"TRAIN: {len(train_frame):,}; VALIDATION: {len(validation_frame):,}.", "",
             f"v1 baseline accuracy/macro-F1: {v1_metrics['accuracy']:.6f}/{v1_metrics['macro_f1']:.6f}."]
    for variant in VARIANTS:
        lines.append(f"{variant} accuracy/macro-F1: "
                     f"{evaluations[variant]['metrics']['accuracy']:.6f}/"
                     f"{evaluations[variant]['metrics']['macro_f1']:.6f}; "
                     f"best epoch {evaluations[variant]['best_epoch']}.")
    lines.extend(["", f"Validation winner: {winner}. Deployment is not authorized.",
                  "TEST v1 was hash-checked only. New unseen camera bouts are required."])
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    output_files = sorted(path for path in output_dir.rglob("*")
                          if path.is_file() and path.name != "training_manifest.json")
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "UVW_VALIDATION_COMPARISON_COMPLETE", "selection": selection,
        "integrity": integrity, "test_evaluated": False,
        "remaining_limitations": [
            "VALIDATION v1 lacks camera-view metadata.",
            "The 57 supplement rows are training data.",
            "A new unseen-bout camera challenge is required before deployment.",
        ],
        "outputs": [{"path": path.relative_to(output_dir).as_posix(),
                     "bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in output_files],
    }
    write_json(manifest_path, manifest, replace=True)
    print("STATUS=UVW_VALIDATION_COMPARISON_COMPLETE")
    print(f"RECOMMENDED_CAMERA_CANDIDATE={selection['recommended_for_camera_test']}")
    print(f"MANIFEST={manifest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    require(args.output_dir.resolve().parent == (ROOT / "ai/dataset").resolve(),
            "Output directory must be versioned under ai/dataset")
    if args.check_only:
        check_only()
    else:
        train(args.output_dir)


if __name__ == "__main__":
    main()
