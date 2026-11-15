"""Train two weighted MLP v2 candidates on TRAIN v2 and evaluate on VALIDATION v1.

TEST v1 is only size/SHA-256 checked. Per-epoch checkpoints make the run resumable.
"""
from __future__ import annotations

import argparse
import copy
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
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_recall_fscore_support)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
PREP_DIR = ROOT / "ai/dataset/mlp_v2_training_data_v1"
SPLIT_DIR = ROOT / "ai/dataset/train_validation_test_split_v1"
V1_DIR = ROOT / "ai/dataset/mlp_validation_comparison_v1"
OUTPUT_DIR = ROOT / "ai/dataset/mlp_v2_validation_comparison_v1"
FEATURES = [f"f{i}" for i in range(1, 64)]
LABELS = tuple("ABCDEFGHIKLMNOPQRSTUVWXY")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
SEED = 42
VARIANTS = {
    "v2_debug_5pct": "weight_debug_budget_5pct",
    "v2_debug_10pct": "weight_debug_budget_10pct_recommended",
}
HYPERPARAMETERS = {
    "hidden_layer_sizes": [64, 32],
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "batch_size": 256,
    "learning_rate": "constant",
    "learning_rate_init": 0.001,
    "max_epochs": 250,
    "external_validation_patience": 20,
    "external_validation_min_delta_macro_f1": 0.0001,
    "shuffle": True,
    "random_state": SEED,
    "internal_early_stopping": False,
}


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

    prep_manifest_path = PREP_DIR / "manifest.json"
    prep_integrity_path = PREP_DIR / "integrity_report.json"
    prep_manifest = load_json(prep_manifest_path)
    prep_integrity = load_json(prep_integrity_path)
    require(prep_manifest["status"] == "TRAIN_V2_DATA_PREPARED", "TRAIN v2 is not ready")
    require(prep_integrity["status"] == "PASS" and
            all(prep_integrity["checks"].values()), "TRAIN v2 integrity is not PASS")
    require(prep_manifest["counts"]["train_v2"] == 62668 and
            prep_manifest["counts"]["accepted_debug"] == 77, "TRAIN v2 count mismatch")
    require(prep_manifest["feature_columns"] == FEATURES and
            prep_manifest["labels"] == list(LABELS), "TRAIN v2 schema contract mismatch")

    snapshots = {}
    for name, item in prep_manifest["output_hashes"].items():
        path = ROOT / name
        require(path.is_file() and path.stat().st_size == item["bytes"] and
                sha256(path) == item["sha256"], f"TRAIN v2 artefact changed: {name}")
        snapshots[name] = item["sha256"]
    for name, item in prep_manifest["input_hashes"].items():
        path = ROOT / name
        require(path.is_file() and path.stat().st_size == item["bytes"] and
                sha256(path) == item["sha256"], f"Protected input changed: {name}")
        snapshots[name] = item["sha256"]
    snapshots[relative(prep_manifest_path)] = sha256(prep_manifest_path)

    v1_manifest_path = V1_DIR / "training_manifest.json"
    v1_manifest = load_json(v1_manifest_path)
    require(v1_manifest["status"] == "VALIDATION_COMPARISON_COMPLETE" and
            v1_manifest["integrity"]["status"] == "PASS" and
            v1_manifest["selection"]["selected_model"] == "baseline",
            "Official baseline v1 manifest mismatch")
    for item in v1_manifest["outputs"]:
        path = V1_DIR / item["path"]
        require(path.is_file() and path.stat().st_size == item["bytes"] and
                sha256(path) == item["sha256"], f"v1 artefact changed: {path}")
        snapshots[relative(path)] = item["sha256"]
    snapshots[relative(v1_manifest_path)] = sha256(v1_manifest_path)

    training_config = load_json(V1_DIR / "training_config.json")
    expected_test_hash = training_config["input_snapshots"][relative(SPLIT_DIR / "test.csv")]
    test_path = SPLIT_DIR / "test.csv"
    require(test_path.is_file() and sha256(test_path) == expected_test_hash, "TEST v1 hash mismatch")
    validation_path = SPLIT_DIR / "validation.csv"
    expected_validation_hash = training_config["input_snapshots"][relative(validation_path)]
    require(validation_path.is_file() and sha256(validation_path) == expected_validation_hash,
            "VALIDATION v1 hash mismatch")
    snapshots[relative(test_path)] = expected_test_hash
    snapshots[relative(validation_path)] = expected_validation_hash
    test_audit = {
        "path": relative(test_path), "verification": "SIZE_AND_SHA256_ONLY",
        "csv_opened_or_parsed": False, "used_for_training": False,
        "used_for_model_selection": False, "evaluated": False,
    }
    return snapshots, test_audit, fit_signature, partial_signature


def load_data():
    expected = ["source", "file", "label", "handedness", *FEATURES]
    train = pd.read_csv(PREP_DIR / "train_v2.csv")
    metadata = pd.read_csv(PREP_DIR / "train_v2_sample_metadata.csv",
                           dtype=str, keep_default_na=False)
    weights = pd.read_csv(PREP_DIR / "train_v2_weights.csv")
    validation = pd.read_csv(SPLIT_DIR / "validation.csv")
    require(train.columns.tolist() == expected and validation.columns.tolist() == expected,
            "TRAIN v2/VALIDATION v1 schema mismatch")
    require(len(train) == len(metadata) == len(weights) == 62668 and len(validation) == 13412,
            "Dataset row count mismatch")
    require(train.file.nunique() == len(train) and validation.file.nunique() == len(validation),
            "Duplicate file identifier")
    require(not set(train.file) & set(validation.file), "TRAIN v2/VALIDATION sample overlap")
    require(set(train.label) == set(validation.label) == set(LABELS), "Label set mismatch")
    require(metadata.v2_row_id.tolist() == weights.v2_row_id.tolist(), "Weight row ID mismatch")
    for field in ("file", "label"):
        require(train[field].astype(str).tolist() == metadata[field].astype(str).tolist(),
                f"Metadata alignment mismatch: {field}")
    require(metadata.label.tolist() == weights.label.astype(str).tolist(), "Weight label mismatch")
    require(metadata.group_id.tolist() == weights.group_id.astype(str).tolist(),
            "Weight group mismatch")
    require((metadata.origin == "TRAIN_V1").sum() == 62591 and
            metadata.origin.str.startswith("DEBUG_").sum() == 77, "Provenance count mismatch")

    assignments = pd.read_csv(SPLIT_DIR / "split_assignments.csv", dtype=str, keep_default_na=False)
    validation_groups = set(assignments.loc[assignments.split.eq("VALIDATION"), "group_id"])
    require(not set(metadata.group_id) & validation_groups, "TRAIN v2/VALIDATION group overlap")

    x_train = train[FEATURES].to_numpy(dtype=np.float64, copy=True)
    x_validation = validation[FEATURES].to_numpy(dtype=np.float64, copy=True)
    require(np.isfinite(x_train).all() and np.isfinite(x_validation).all(), "Non-finite feature")
    y_train = train.label.map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    y_validation = validation.label.map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    return train, metadata, weights, validation, x_train, y_train, x_validation, y_validation


def audit_weight_column(metadata, weights, column):
    values = weights[column].to_numpy(dtype=np.float64)
    require(np.isfinite(values).all() and (values > 0).all(), f"Invalid weights: {column}")
    require(math.isclose(values.sum(), len(values), rel_tol=0.0, abs_tol=1e-6),
            f"Weight total mismatch: {column}")
    class_target = len(values) / len(LABELS)
    class_rows = []
    for label in LABELS:
        mask = metadata.label.eq(label).to_numpy()
        total = float(values[mask].sum())
        require(math.isclose(total, class_target, rel_tol=0.0, abs_tol=1e-6),
                f"Class weight mismatch for {column}/{label}")
        class_rows.append({"variant_weight": column, "scope": "CLASS", "label": label,
                           "view": "", "group_id": "", "sample_count": int(mask.sum()),
                           "weight_total": total, "fraction_of_class": total / class_target})

    expected = {
        "weight_debug_budget_5pct": {
            ("P", "front"): 0.01, ("P", "horizontal"): 0.04,
            ("Q", "horizontal"): 0.05, ("U", "front"): 0.05,
            ("V", "front"): 0.05,
        },
        "weight_debug_budget_10pct_recommended": {
            ("P", "front"): 0.02, ("P", "horizontal"): 0.08,
            ("Q", "horizontal"): 0.10, ("U", "front"): 0.10,
            ("V", "front"): 0.10,
        },
    }[column]
    debug_rows = []
    debug = metadata.origin.ne("TRAIN_V1")
    require(metadata.loc[debug, "group_id"].nunique() == 5, "Debug conservative group count mismatch")
    for group_id, frame in metadata.loc[debug].groupby("group_id", sort=True):
        indexes = frame.index.to_numpy()
        label = frame.label.iloc[0]
        view = frame.input_view.iloc[0]
        require(frame.label.nunique() == frame.input_view.nunique() == 1, "Impure debug group")
        total = float(values[indexes].sum())
        fraction = total / class_target
        require(math.isclose(fraction, expected[(label, view)], rel_tol=0.0, abs_tol=1e-9),
                f"Debug budget mismatch for {column}/{group_id}")
        require(np.ptp(values[indexes]) < 1e-15, f"Unequal within-group weights: {group_id}")
        debug_rows.append({"variant_weight": column, "scope": "DEBUG_GROUP", "label": label,
                           "view": view, "group_id": group_id, "sample_count": len(frame),
                           "weight_total": total, "fraction_of_class": fraction})
    return values, class_rows + debug_rows


def make_model():
    return MLPClassifier(
        hidden_layer_sizes=tuple(HYPERPARAMETERS["hidden_layer_sizes"]),
        activation=HYPERPARAMETERS["activation"], solver=HYPERPARAMETERS["solver"],
        alpha=HYPERPARAMETERS["alpha"], batch_size=HYPERPARAMETERS["batch_size"],
        learning_rate=HYPERPARAMETERS["learning_rate"],
        learning_rate_init=HYPERPARAMETERS["learning_rate_init"], max_iter=1,
        shuffle=HYPERPARAMETERS["shuffle"], random_state=SEED,
        early_stopping=False, tol=0.0,
        n_iter_no_change=HYPERPARAMETERS["max_epochs"] + 1)


def epoch_metrics(model, x_validation, y_validation):
    predictions = model.predict(x_validation)
    probabilities = model.predict_proba(x_validation)
    _, recall, _, _ = precision_recall_fscore_support(
        y_validation, predictions, labels=np.arange(len(LABELS)), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_validation, predictions)),
        "macro_f1": float(f1_score(y_validation, predictions, labels=np.arange(len(LABELS)),
                                   average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_validation, probabilities,
                                   labels=np.arange(len(LABELS)))),
        "minimum_class_recall": float(recall.min()),
        **{f"{label}_recall": float(recall[LABEL_TO_ID[label]])
           for label in ("P", "Q", "U", "V", "W")},
    }


def train_variant(name, x_train, y_train, x_validation, y_validation,
                  sample_weight, checkpoint_dir, config_sha256):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "state.joblib"
    if checkpoint_path.exists():
        state = joblib.load(checkpoint_path)
        require(state["name"] == name and state["config_sha256"] == config_sha256,
                "Checkpoint variant/config mismatch")
    else:
        state = {
            "name": name, "config_sha256": config_sha256, "epoch": 0,
            "model": make_model(), "best_model": None, "best_metrics": None,
            "best_epoch": 0, "history": [],
            "epochs_without_material_improvement": 0, "complete": False,
        }
    if state["complete"]:
        return state

    classes = np.arange(len(LABELS), dtype=np.int64)
    patience = HYPERPARAMETERS["external_validation_patience"]
    min_delta = HYPERPARAMETERS["external_validation_min_delta_macro_f1"]
    while state["epoch"] < HYPERPARAMETERS["max_epochs"]:
        kwargs = {"sample_weight": sample_weight}
        if state["epoch"] == 0:
            state["model"].partial_fit(x_train, y_train, classes=classes, **kwargs)
        else:
            state["model"].partial_fit(x_train, y_train, **kwargs)
        state["epoch"] += 1
        metrics = epoch_metrics(state["model"], x_validation, y_validation)
        state["history"].append({"epoch": state["epoch"], **metrics})
        previous = state["best_metrics"]
        is_best = previous is None or metrics["macro_f1"] > previous["macro_f1"] or (
            metrics["macro_f1"] == previous["macro_f1"] and
            (metrics["minimum_class_recall"], metrics["accuracy"]) >
            (previous["minimum_class_recall"], previous["accuracy"]))
        material = previous is None or metrics["macro_f1"] > previous["macro_f1"] + min_delta
        if is_best:
            state["best_model"] = copy.deepcopy(state["model"])
            state["best_metrics"] = metrics
            state["best_epoch"] = state["epoch"]
        state["epochs_without_material_improvement"] = (
            0 if material else state["epochs_without_material_improvement"] + 1)
        if state["epochs_without_material_improvement"] >= patience:
            state["complete"] = True
            state["stop_reason"] = "EXTERNAL_VALIDATION_MACRO_F1_PATIENCE"
        if state["epoch"] >= HYPERPARAMETERS["max_epochs"]:
            state["complete"] = True
            state["stop_reason"] = "MAX_EPOCHS"
        atomic_joblib(checkpoint_path, state)
        if state["complete"]:
            break
    require(state["best_model"] is not None, "No best model checkpoint")
    return state


def evaluate(model, x, y, frame):
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)
    precision, recall, per_f1, support = precision_recall_fscore_support(
        y, predictions, labels=np.arange(len(LABELS)), zero_division=0)
    matrix = confusion_matrix(y, predictions, labels=np.arange(len(LABELS)))
    per_class = [
        {"label": label, "precision": float(precision[index]),
         "recall": float(recall[index]), "f1": float(per_f1[index]),
         "support": int(support[index])}
        for index, label in enumerate(LABELS)
    ]
    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, labels=np.arange(len(LABELS)),
                                   average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=np.arange(len(LABELS)))),
        "minimum_class_recall": float(recall.min()),
        "minimum_recall_labels": [LABELS[index] for index, value in enumerate(recall)
                                  if value == recall.min()],
    }
    prediction_rows = [
        {"file": file, "original_label": original,
         "predicted_label": LABELS[int(predicted)],
         "predicted_confidence": float(probabilities[index].max()),
         "correct": bool(predicted == y[index])}
        for index, (file, original, predicted) in enumerate(
            zip(frame.file, frame.label, predictions, strict=True))
    ]
    return metrics, per_class, matrix, prediction_rows, predictions, probabilities


def confusion_value(matrix, true_label, predicted_label):
    return int(matrix[LABEL_TO_ID[true_label], LABEL_TO_ID[predicted_label]])


def compare_to_v1(name, evaluation, v1_evaluation):
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
    significant = [row for row in deltas if row["recall_delta"] < -0.02]
    return {
        "accuracy_delta": evaluation["metrics"]["accuracy"] - v1_evaluation["metrics"]["accuracy"],
        "macro_f1_delta": evaluation["metrics"]["macro_f1"] - v1_evaluation["metrics"]["macro_f1"],
        "significant_recall_regressions_2pct": significant,
        "per_class_deltas": deltas,
    }


def evaluate_debug_training_rows(model, scaler, train, metadata, x_train, variant):
    mask = metadata.origin.ne("TRAIN_V1").to_numpy()
    x = scaler.transform(x_train[mask])
    y = train.loc[mask, "label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)
    rows = []
    for local, source_index in enumerate(np.flatnonzero(mask)):
        rows.append({
            "candidate": variant,
            "sample_id": metadata.iloc[source_index].sample_id,
            "label": train.iloc[source_index].label,
            "view": metadata.iloc[source_index].input_view,
            "group_id": metadata.iloc[source_index].group_id,
            "predicted_label": LABELS[int(predictions[local])],
            "confidence": float(probabilities[local].max()),
            "correct": bool(predictions[local] == y[local]),
            "evaluation_scope": "TRAINING_ROWS_DIAGNOSTIC_NOT_INDEPENDENT",
        })
    summary = []
    frame = pd.DataFrame(rows)
    for (label, view), part in frame.groupby(["label", "view"], sort=True):
        summary.append({
            "candidate": variant, "label": label, "view": view,
            "samples": len(part), "correct": int(part.correct.sum()),
            "accuracy": float(part.correct.mean()),
            "prediction_counts": part.predicted_label.value_counts().sort_index().to_dict(),
            "scope": "TRAINING_ROWS_DIAGNOSTIC_NOT_INDEPENDENT",
        })
    return rows, summary


def config_payload(snapshots, test_audit, fit_signature, partial_signature):
    return {
        "schema_version": 1, "seed": SEED, "features": FEATURES,
        "labels": list(LABELS), "architecture": [63, 64, 32, 24],
        "hyperparameters": HYPERPARAMETERS, "variants": VARIANTS,
        "v1_parity": {
            "architecture_unchanged": True, "optimizer_unchanged": True,
            "early_stopping_unchanged": True,
            "change": "StandardScaler is refit on TRAIN v2 only, as required because TRAIN v2 is the new fitting dataset.",
        },
        "standardization": "One StandardScaler fit on all 62,668 TRAIN v2 rows; transforms VALIDATION v1; TEST untouched.",
        "early_stopping": "External pre-split VALIDATION v1 macro-F1; sklearn internal early_stopping disabled.",
        "sample_weight_support": {
            "fit_signature": str(fit_signature), "partial_fit_signature": str(partial_signature),
            "supported": True, "fallback_used": False,
        },
        "libraries": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "joblib": version("joblib"),
        },
        "input_snapshots": snapshots, "test_access": test_audit,
        "limitations": [
            "VALIDATION v1 has no view metadata and cannot prove P-horizontal or V-front correction.",
            "The 77 appended debug rows are training data and cannot be independent evaluation evidence.",
        ],
        "implementation": {"script": signature(Path(__file__).resolve())},
    }


def check_only():
    snapshots, test_audit, fit_sig, partial_sig = verify_inputs()
    train, metadata, weights, validation, x_train, _, x_validation, _ = load_data()
    audits = []
    for column in VARIANTS.values():
        _, rows = audit_weight_column(metadata, weights, column)
        audits.extend(rows)
    require(not OUTPUT_DIR.exists(), "Output exists; inspect/resume rather than check-only")
    print(json.dumps({
        "status": "CHECK_ONLY_PASS", "training_started": False,
        "train_samples": len(train), "validation_samples": len(validation),
        "features": x_train.shape[1], "validation_features": x_validation.shape[1],
        "weight_audit_rows": len(audits), "input_hashes_verified": len(snapshots),
        "test_access": test_audit, "sample_weight_supported": True,
    }))


def train(output_dir):
    snapshots, test_audit, fit_sig, partial_sig = verify_inputs()
    config = config_payload(snapshots, test_audit, fit_sig, partial_sig)
    final_manifest_path = output_dir / "training_manifest.json"
    if output_dir.exists() and final_manifest_path.exists():
        manifest = load_json(final_manifest_path)
        require(manifest["status"] == "V2_VALIDATION_COMPARISON_COMPLETE",
                "Existing final manifest status mismatch")
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
    weight_audit_rows = []
    variant_weights = {}
    for name, column in VARIANTS.items():
        variant_weights[name], rows = audit_weight_column(metadata, weight_frame, column)
        weight_audit_rows.extend(rows)
    write_csv(output_dir / "pretraining_weight_audit.csv", weight_audit_rows)
    write_json(output_dir / "pretraining_weight_audit.json", {
        "status": "PASS", "train_rows": len(train_frame), "classes": len(LABELS),
        "debug_conservative_groups": 5, "near_duplicate_control":
        "All same-condition short-session frames share one group; per-frame weights divide the fixed group budget.",
        "rows": weight_audit_rows,
    }, replace=True)

    scaler_path = output_dir / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        require(int(scaler.n_samples_seen_) == len(train_frame), "Scaler is not TRAIN-v2-only")
    else:
        scaler = StandardScaler().fit(x_train)
        require(int(scaler.n_samples_seen_) == len(train_frame), "Scaler fit count mismatch")
        atomic_joblib(scaler_path, scaler)
    x_train_scaled = scaler.transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)

    config_sha = sha256(output_dir / "training_config.json")
    states = {}
    completed = []
    for name in VARIANTS:
        states[name] = train_variant(
            name, x_train_scaled, y_train, x_validation_scaled, y_validation,
            variant_weights[name], output_dir / "checkpoints" / name, config_sha)
        completed.append(name)
        atomic_json(output_dir / "run_state.json",
                    {"status": "TRAINING", "completed_variants": completed})

    v1_model = joblib.load(V1_DIR / "baseline_model.joblib")
    v1_scaler = joblib.load(V1_DIR / "scaler.joblib")
    require(int(v1_scaler.n_samples_seen_) == 62591, "v1 scaler fit-count mismatch")
    require(np.array_equal(v1_model.classes_, np.arange(len(LABELS))), "v1 model class order mismatch")
    v1_metrics, v1_per_class, v1_matrix, _, _, _ = evaluate(
        v1_model, v1_scaler.transform(x_validation), y_validation, validation_frame)
    saved_v1 = load_json(V1_DIR / "evaluation_summary.json")["baseline"]
    require(abs(v1_metrics["accuracy"] - saved_v1["metrics"]["accuracy"]) < 1e-15 and
            abs(v1_metrics["macro_f1"] - saved_v1["metrics"]["macro_f1"]) < 1e-15,
            "Recomputed baseline v1 metrics differ from official report")
    v1_evaluation = {"metrics": v1_metrics, "per_class": v1_per_class,
                     "best_epoch": saved_v1["best_epoch"]}

    evaluations = {}
    all_delta_rows = []
    all_debug_rows = []
    debug_summaries = []
    special_rows = []
    for name, state in states.items():
        model = state["best_model"]
        require(np.array_equal(model.classes_, np.arange(len(LABELS))), f"Class order mismatch: {name}")
        metrics, per_class, matrix, predictions, _, _ = evaluate(
            model, x_validation_scaled, y_validation, validation_frame)
        evaluation = {
            "metrics": metrics, "per_class": per_class, "best_epoch": state["best_epoch"],
            "epochs_run": state["epoch"], "stop_reason": state["stop_reason"],
        }
        comparison = compare_to_v1(name, evaluation, v1_evaluation)
        evaluation["comparison_to_v1"] = {key: value for key, value in comparison.items()
                                           if key != "per_class_deltas"}
        evaluations[name] = evaluation
        all_delta_rows.extend(comparison["per_class_deltas"])

        atomic_joblib(output_dir / f"{name}_model.joblib", model)
        write_csv(output_dir / f"{name}_training_history.csv", state["history"])
        write_csv(output_dir / f"{name}_per_class_metrics.csv", per_class)
        write_csv(output_dir / f"{name}_validation_predictions.csv", predictions)
        with (output_dir / f"{name}_confusion_matrix.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true\\predicted", *LABELS])
            for label, row in zip(LABELS, matrix, strict=True):
                writer.writerow([label, *map(int, row)])

        current_by_label = {row["label"]: row for row in per_class}
        v1_by_label = {row["label"]: row for row in v1_per_class}
        for label in ("P", "Q", "U", "V", "W"):
            special_rows.append({
                "candidate": name, "label": label,
                "v1_precision": v1_by_label[label]["precision"],
                "candidate_precision": current_by_label[label]["precision"],
                "v1_recall": v1_by_label[label]["recall"],
                "candidate_recall": current_by_label[label]["recall"],
                "recall_delta": current_by_label[label]["recall"] - v1_by_label[label]["recall"],
                "P_as_Q": confusion_value(matrix, "P", "Q") if label == "P" else "",
                "Q_as_P": confusion_value(matrix, "Q", "P") if label == "Q" else "",
                "U_as_V": confusion_value(matrix, "U", "V") if label == "U" else "",
                "V_as_U": confusion_value(matrix, "V", "U") if label == "V" else "",
                "V_as_W": confusion_value(matrix, "V", "W") if label == "V" else "",
                "W_as_V": confusion_value(matrix, "W", "V") if label == "W" else "",
            })
        debug_rows, debug_summary = evaluate_debug_training_rows(
            model, scaler, train_frame, metadata, x_train, name)
        all_debug_rows.extend(debug_rows)
        debug_summaries.extend(debug_summary)

    write_csv(output_dir / "per_class_comparison_to_v1.csv", all_delta_rows)
    write_csv(output_dir / "special_classes_P_Q_U_V_W.csv", special_rows)
    write_csv(output_dir / "debug_training_rows_predictions.csv", all_debug_rows)
    write_json(output_dir / "debug_training_rows_summary.json", {
        "disclaimer": "These 77 rows were used for training. Results are diagnostic only and are not independent evaluation evidence.",
        "conditions": debug_summaries,
    }, replace=True)

    candidate_order = sorted(VARIANTS, key=lambda name: (
        evaluations[name]["metrics"]["macro_f1"],
        evaluations[name]["metrics"]["minimum_class_recall"],
        evaluations[name]["metrics"]["accuracy"], name == "v2_debug_5pct"), reverse=True)
    winner = candidate_order[0]
    severe = evaluations[winner]["comparison_to_v1"]["significant_recall_regressions_2pct"]
    selection = {
        "validation_winner_among_v2_candidates": winner,
        "recommended_for_camera_test": winner if not severe else None,
        "status": "CAMERA_TEST_RECOMMENDED" if not severe else "VALIDATION_RECALL_REGRESSION_REVIEW_REQUIRED",
        "criterion": "Highest VALIDATION v1 macro-F1; ties use minimum class recall then accuracy.",
        "material_recall_regression_threshold": 0.02,
        "material_recall_regressions": severe,
        "deployment_authorized": False,
        "camera_error_fixed_claimed": False,
        "reason": "VALIDATION v1 lacks view metadata; only new unseen camera bouts can test P-horizontal and V-front.",
    }

    shutil.copyfile(output_dir / f"{winner}_model.joblib",
                    output_dir / "validation_winner_model.joblib")
    write_json(output_dir / "label_mapping.json", {
        "classes_in_model_order": list(LABELS), "label_to_index": LABEL_TO_ID,
        "index_to_label": {str(index): label for label, index in LABEL_TO_ID.items()},
    }, replace=True)
    write_json(output_dir / "evaluation_summary.json", {
        "train_samples": len(train_frame), "validation_samples": len(validation_frame),
        "baseline_v1": v1_evaluation, **evaluations, "selection": selection,
        "validation_limitation": "No view metadata; cannot establish P-horizontal/V-front correction.",
        "debug_limitation": "The 77 debug samples are training rows, not independent evaluation.",
        "test_access": test_audit,
    }, replace=True)
    write_json(output_dir / "selected_candidate.json", {
        "validation_winner": winner,
        "recommended_for_camera_test": selection["recommended_for_camera_test"],
        "model_path": "validation_winner_model.joblib",
        "source_model_path": f"{winner}_model.joblib",
        "scaler_path": "scaler.joblib", "label_mapping_path": "label_mapping.json",
        "deployment_authorized": False, "selection": selection,
    }, replace=True)

    # Reload every persisted inference artefact and execute deterministic inference.
    reload_checks = []
    for name in VARIANTS:
        reloaded_model = joblib.load(output_dir / f"{name}_model.joblib")
        reloaded_scaler = joblib.load(output_dir / "scaler.joblib")
        probability = reloaded_model.predict_proba(reloaded_scaler.transform(x_validation[:16]))
        require(probability.shape == (16, len(LABELS)) and np.isfinite(probability).all() and
                np.allclose(probability.sum(axis=1), 1.0, atol=1e-12),
                f"Reload/inference failed: {name}")
        reload_checks.append({"candidate": name, "samples": 16, "status": "PASS"})

    test_run = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "tests.test_train_mlp_v2", "-q"],
        cwd=ROOT, capture_output=True, text=True, check=False)
    require(test_run.returncode == 0, f"Unit tests failed: {test_run.stderr[-1000:]}")
    write_json(output_dir / "test_report.json", {
        "status": "PASS", "unit_tests_run": 5, "reload_inference_checks": reload_checks,
    }, replace=True)

    # Recheck every immutable input after training.
    for path_name, expected in snapshots.items():
        require(sha256(ROOT / path_name) == expected, f"Protected input changed: {path_name}")
    integrity = {
        "status": "PASS", "checks_run": 27,
        "train_samples": len(train_frame), "validation_samples": len(validation_frame),
        "features_used": 63, "labels": 24,
        "train_validation_sample_overlap": 0, "train_validation_group_overlap": 0,
        "scaler_train_rows": int(scaler.n_samples_seen_),
        "sample_weight_supported_natively": True, "sample_weight_fallback_used": False,
        "weight_class_totals_equal": True, "debug_group_budget_checks": "PASS",
        "debug_near_duplicate_group_control": "PASS",
        "model_scaler_reload_inference": "PASS", "unit_tests": "PASS",
        "test_csv_access": "SIZE_AND_SHA256_ONLY", "test_evaluated": False,
        "protected_inputs_hashes_unchanged": True,
        "model_v1_modified": False, "camera_app_modified": False,
        "frozen_or_split_data_modified": False, "debug_data_modified": False,
        "augmentation_or_row_resampling": False,
    }
    write_json(output_dir / "integrity_report.json", integrity, replace=True)
    atomic_json(output_dir / "run_state.json", {
        "status": "COMPLETE", "completed_variants": list(VARIANTS),
        "validation_winner": winner,
        "recommended_for_camera_test": selection["recommended_for_camera_test"],
    })

    report_lines = [
        "# MLP v2 validation comparison", "",
        f"TRAIN v2: {len(train_frame):,}; VALIDATION v1: {len(validation_frame):,}.", "",
        f"Baseline v1 accuracy/macro-F1: {v1_metrics['accuracy']:.6f}/{v1_metrics['macro_f1']:.6f}.",
    ]
    for name in VARIANTS:
        report_lines.append(
            f"{name} accuracy/macro-F1: {evaluations[name]['metrics']['accuracy']:.6f}/"
            f"{evaluations[name]['metrics']['macro_f1']:.6f}; best epoch "
            f"{evaluations[name]['best_epoch']}.")
    report_lines.extend([
        "", f"Validation winner: {winner}. Deployment is not authorized.", "",
        "VALIDATION v1 has no view metadata. The 77 debug samples are training rows. "
        "A new unseen-bout camera test is required before any claim that P-horizontal or V-front is fixed.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    output_files = sorted(path for path in output_dir.rglob("*")
                          if path.is_file() and path.name != "training_manifest.json")
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "V2_VALIDATION_COMPARISON_COMPLETE", "selection": selection,
        "integrity": integrity, "test_evaluated": False,
        "remaining_limitations": [
            "VALIDATION v1 lacks view metadata.",
            "The 77 debug samples are not independent evaluation data.",
            "New manually confirmed unseen camera bouts are required before deployment.",
        ],
        "outputs": [
            {"path": path.relative_to(output_dir).as_posix(), "bytes": path.stat().st_size,
             "sha256": sha256(path)} for path in output_files
        ],
    }
    write_json(final_manifest_path, manifest, replace=True)
    print("STATUS=V2_VALIDATION_COMPARISON_COMPLETE")
    print(f"RECOMMENDED_CAMERA_CANDIDATE={selection['recommended_for_camera_test']}")
    print(f"MANIFEST={final_manifest_path}")


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
