"""Train and compare baseline/weighted MLPs on the frozen ASL split.

Default mode trains with resumable per-epoch checkpoints. ``--check-only``
validates code, dependencies and data contracts without fitting any object or
creating the training output directory. TEST is hash-checked only and is never
opened as a CSV in this phase.
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
from collections import Counter
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
SPLIT_DIR = ROOT / "ai/dataset/train_validation_test_split_v1"
BALANCE_DIR = ROOT / "ai/dataset/train_balance_analysis_v1"
OUTPUT_DIR = ROOT / "ai/dataset/mlp_validation_comparison_v1"
FEATURES = [f"f{i}" for i in range(1, 64)]
LABELS = tuple("ABCDEFGHIKLMNOPQRSTUVWXY")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
SEED = 42
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
RECALL_FLOOR = 0.80
MATERIAL_RECALL_REGRESSION = 0.10


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
    with Path(path).open("w" if replace else "x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


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


def signature(path):
    path = Path(path)
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def verify_inputs():
    fit_signature = inspect.signature(MLPClassifier.fit)
    partial_signature = inspect.signature(MLPClassifier.partial_fit)
    require("sample_weight" in fit_signature.parameters and
            "sample_weight" in partial_signature.parameters,
            "Installed MLPClassifier lacks native sample_weight; no fallback is authorized")

    split_manifest_path = SPLIT_DIR / "split_manifest.json"
    split_manifest = load_json(split_manifest_path)
    require(split_manifest["status"] == "TRAIN_VALIDATION_TEST_SPLIT_READY",
            "Split manifest is not ready")
    require(split_manifest["seed"] == SEED and
            split_manifest["integrity"]["status"] == "PASS" and
            not split_manifest["remaining_blockers"], "Split integrity/seed mismatch")
    snapshots = {}
    for item in split_manifest["outputs"]:
        path = SPLIT_DIR / item["path"]
        require(path.is_file() and path.stat().st_size == item["bytes"] and
                sha256(path) == item["sha256"], f"Split artefact changed: {path}")
        snapshots[path.relative_to(ROOT).as_posix()] = item["sha256"]
    for path in [split_manifest_path, SPLIT_DIR / "test_report.json"]:
        require(path.is_file(), f"Missing split metadata: {path}")
        snapshots[path.relative_to(ROOT).as_posix()] = sha256(path)

    balance_manifest_path = BALANCE_DIR / "analysis_manifest.json"
    balance_manifest = load_json(balance_manifest_path)
    require(balance_manifest["status"] == "TRAIN_BALANCE_ANALYSIS_READY" and
            balance_manifest["integrity"]["status"] == "PASS",
            "Balance analysis is not ready")
    for item in balance_manifest["outputs"]:
        path = BALANCE_DIR / item["path"]
        require(path.is_file() and path.stat().st_size == item["bytes"] and
                sha256(path) == item["sha256"], f"Balance artefact changed: {path}")
        snapshots[path.relative_to(ROOT).as_posix()] = item["sha256"]
    snapshots[balance_manifest_path.relative_to(ROOT).as_posix()] = sha256(balance_manifest_path)
    recommendation = load_json(BALANCE_DIR / "training_recommendation.json")
    require(recommendation["preferred"] == "GROUP_AND_CLASS_ADJUSTED_SAMPLE_WEIGHTING",
            "Unexpected balance recommendation")

    test_item = next(item for item in split_manifest["outputs"] if item["path"] == "test.csv")
    test_path = SPLIT_DIR / "test.csv"
    require(test_path.is_file() and test_path.stat().st_size == test_item["bytes"] and
            sha256(test_path) == test_item["sha256"], "TEST hash mismatch")
    test_audit = {"path": test_path.relative_to(ROOT).as_posix(),
                  "verification": "SIZE_AND_SHA256_ONLY", "csv_opened_or_parsed": False,
                  "used_for_training": False, "used_for_model_selection": False,
                  "evaluated": False}
    return split_manifest, balance_manifest, snapshots, test_audit, fit_signature, partial_signature


def load_train_validation():
    expected = ["source", "file", "label", "handedness", *FEATURES]
    train = pd.read_csv(SPLIT_DIR / "train.csv")
    validation = pd.read_csv(SPLIT_DIR / "validation.csv")
    require(train.columns.tolist() == expected and validation.columns.tolist() == expected,
            "TRAIN/VALIDATION schema mismatch")
    require(len(train) == 62591 and len(validation) == 13412,
            "Unexpected TRAIN/VALIDATION row count")
    require(train.file.nunique() == len(train) and validation.file.nunique() == len(validation) and
            not set(train.file) & set(validation.file), "Duplicate/overlapping samples")
    require(set(train.label) == set(LABELS) and set(validation.label) == set(LABELS),
            "Static label set mismatch")

    assignments = pd.read_csv(SPLIT_DIR / "split_assignments.csv", dtype=str, keep_default_na=False)
    groups = pd.read_csv(SPLIT_DIR / "group_assignments.csv", dtype=str, keep_default_na=False)
    require(assignments.file.nunique() == len(assignments) == 89415, "Sample assignment scope mismatch")
    require(groups.group_id.nunique() == len(groups) == 60919, "Group assignment scope mismatch")
    assignment_by_file = assignments.set_index("file")
    for frame, split in [(train, "TRAIN"), (validation, "VALIDATION")]:
        mapped = assignment_by_file.loc[frame.file]
        require((mapped.split.to_numpy() == split).all() and
                (mapped.label.to_numpy() == frame.label.to_numpy()).all(),
                f"{split} assignment mismatch")
    train_group_ids = assignment_by_file.loc[train.file].group_id.to_numpy(dtype=str)
    validation_group_ids = assignment_by_file.loc[validation.file].group_id.to_numpy(dtype=str)
    require(not set(train_group_ids) & set(validation_group_ids), "TRAIN/VALIDATION group overlap")

    x_train = train[FEATURES].to_numpy(dtype=np.float64, copy=True)
    x_validation = validation[FEATURES].to_numpy(dtype=np.float64, copy=True)
    require(np.isfinite(x_train).all() and np.isfinite(x_validation).all(), "Non-finite feature")
    y_train = train.label.map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    y_validation = validation.label.map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    return (train, validation, x_train, y_train, x_validation, y_validation,
            train_group_ids, groups.set_index("group_id"))


def build_group_class_weights(labels, group_ids, group_table):
    require(len(labels) == len(group_ids), "Weight input length mismatch")
    group_sizes = Counter(group_ids)
    group_labels = {}
    class_group_counts = Counter()
    for label_id, group_id in zip(labels, group_ids, strict=True):
        label = LABELS[int(label_id)]
        if group_id not in group_labels:
            group_labels[group_id] = label
            class_group_counts[label] += 1
        require(group_labels[group_id] == label, "Impure TRAIN group")
    require(set(class_group_counts) == set(LABELS), "Class missing from weighted TRAIN")

    n = len(labels)
    weights = np.empty(n, dtype=np.float64)
    for index, (label_id, group_id) in enumerate(zip(labels, group_ids, strict=True)):
        label = LABELS[int(label_id)]
        expected_size = int(group_table.loc[group_id, "group_size"])
        require(expected_size == group_sizes[group_id], "TRAIN group-size mismatch")
        weights[index] = n / (len(LABELS) * class_group_counts[label] * group_sizes[group_id])
    require(np.isfinite(weights).all() and (weights > 0).all() and
            abs(weights.mean() - 1.0) < 1e-12, "Invalid normalized weights")
    target_class_total = n / len(LABELS)
    for label_id, label in enumerate(LABELS):
        require(abs(weights[labels == label_id].sum() - target_class_total) < 1e-8,
                f"Class weight mismatch: {label}")
        target_group_total = target_class_total / class_group_counts[label]
        for group_id in set(group_ids[labels == label_id]):
            require(abs(weights[group_ids == group_id].sum() - target_group_total) < 1e-8,
                    f"Group weight mismatch: {group_id}")
    return weights, class_group_counts, group_sizes


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
    return {"accuracy": float(accuracy_score(y_validation, predictions)),
            "macro_f1": float(f1_score(y_validation, predictions,
                                       labels=np.arange(len(LABELS)), average="macro", zero_division=0)),
            "log_loss": float(log_loss(y_validation, probabilities, labels=np.arange(len(LABELS)))),
            "minimum_class_recall": float(recall.min()),
            "M_recall": float(recall[LABEL_TO_ID["M"]]),
            "N_recall": float(recall[LABEL_TO_ID["N"]]),
            "T_recall": float(recall[LABEL_TO_ID["T"]])}


def train_variant(name, x_train, y_train, x_validation, y_validation,
                  sample_weight, checkpoint_dir):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "state.joblib"
    if checkpoint_path.exists():
        state = joblib.load(checkpoint_path)
        require(state["name"] == name and state["weighted"] == (sample_weight is not None),
                "Checkpoint variant mismatch")
        print(f"RESUME {name} epoch={state['epoch']}", flush=True)
    else:
        state = {"name": name, "weighted": sample_weight is not None, "epoch": 0,
                 "model": make_model(), "best_model": None, "best_metrics": None,
                 "best_epoch": 0, "history": [], "epochs_without_material_improvement": 0,
                 "complete": False}
    if state["complete"]:
        return state

    classes = np.arange(len(LABELS), dtype=np.int64)
    patience = HYPERPARAMETERS["external_validation_patience"]
    min_delta = HYPERPARAMETERS["external_validation_min_delta_macro_f1"]
    while state["epoch"] < HYPERPARAMETERS["max_epochs"]:
        kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
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
        if state["epoch"] == 1 or state["epoch"] % 5 == 0 or state["complete"]:
            print(f"[{name}] epoch={state['epoch']:03d} macro_f1={metrics['macro_f1']:.6f} "
                  f"accuracy={metrics['accuracy']:.6f} best_epoch={state['best_epoch']} "
                  f"patience={state['epochs_without_material_improvement']}/{patience}", flush=True)
        if state["complete"]:
            break
    require(state["best_model"] is not None, "No best model checkpoint")
    return state


def evaluate(model, x_validation, y_validation, validation_frame):
    predictions = model.predict(x_validation)
    probabilities = model.predict_proba(x_validation)
    precision, recall, per_f1, support = precision_recall_fscore_support(
        y_validation, predictions, labels=np.arange(len(LABELS)), zero_division=0)
    matrix = confusion_matrix(y_validation, predictions, labels=np.arange(len(LABELS)))
    per_class = [{"label": label, "precision": float(precision[index]),
                  "recall": float(recall[index]), "f1": float(per_f1[index]),
                  "support": int(support[index])} for index, label in enumerate(LABELS)]
    directed = []
    for true_id, true_label in enumerate(LABELS):
        for predicted_id, predicted_label in enumerate(LABELS):
            count = int(matrix[true_id, predicted_id])
            if true_id != predicted_id and count:
                directed.append({"true_label": true_label, "predicted_label": predicted_label,
                                 "count": count,
                                 "rate_within_true_class": count / int(matrix[true_id].sum())})
    directed.sort(key=lambda row: (-row["count"], -row["rate_within_true_class"],
                                   row["true_label"], row["predicted_label"]))
    symmetric = []
    for left in range(len(LABELS)):
        for right in range(left + 1, len(LABELS)):
            count = int(matrix[left, right] + matrix[right, left])
            if count:
                symmetric.append({"pair": f"{LABELS[left]}/{LABELS[right]}",
                                  "bidirectional_errors": count,
                                  "left_as_right": int(matrix[left, right]),
                                  "right_as_left": int(matrix[right, left])})
    symmetric.sort(key=lambda row: (-row["bidirectional_errors"], row["pair"]))
    metrics = {"accuracy": float(accuracy_score(y_validation, predictions)),
               "macro_f1": float(f1_score(y_validation, predictions,
                                           labels=np.arange(len(LABELS)), average="macro", zero_division=0)),
               "log_loss": float(log_loss(y_validation, probabilities, labels=np.arange(len(LABELS)))),
               "minimum_class_recall": float(recall.min()),
               "minimum_recall_labels": [LABELS[i] for i, value in enumerate(recall)
                                         if value == recall.min()],
               "M_recall": float(recall[LABEL_TO_ID["M"]]),
               "N_recall": float(recall[LABEL_TO_ID["N"]]),
               "T_recall": float(recall[LABEL_TO_ID["T"]])}
    prediction_rows = [{"file": file, "original_label": original,
                        "predicted_label": LABELS[int(predicted)],
                        "predicted_confidence": float(probabilities[index].max()),
                        "correct": bool(predicted == y_validation[index])}
                       for index, (file, original, predicted) in enumerate(
                           zip(validation_frame.file, validation_frame.label, predictions, strict=True))]
    return metrics, per_class, matrix, directed, symmetric, prediction_rows


def select_model(evaluations):
    ordered = sorted(evaluations, key=lambda name: (
        evaluations[name]["metrics"]["macro_f1"],
        evaluations[name]["metrics"]["minimum_class_recall"],
        evaluations[name]["metrics"]["accuracy"], name == "weighted"), reverse=True)
    winner, alternative = ordered
    alternate_recall = {row["label"]: row["recall"]
                        for row in evaluations[alternative]["per_class"]}
    regressions = [{"label": row["label"], "selected_recall": row["recall"],
                    "alternative_recall": alternate_recall[row["label"]],
                    "difference": row["recall"] - alternate_recall[row["label"]]}
                   for row in evaluations[winner]["per_class"]
                   if alternate_recall[row["label"]] - row["recall"] > MATERIAL_RECALL_REGRESSION]
    guard = (evaluations[winner]["metrics"]["minimum_class_recall"] >= RECALL_FLOOR
             and not regressions)
    return {"macro_f1_winner": winner, "selected_model": winner if guard else None,
            "selection_status": "SELECTED" if guard else "RECALL_GUARD_REQUIRES_REVIEW",
            "criterion": "Highest VALIDATION macro-F1; ties use minimum recall then accuracy.",
            "recall_guard_floor": RECALL_FLOOR,
            "material_recall_regression_threshold": MATERIAL_RECALL_REGRESSION,
            "recall_guard_pass": guard, "material_recall_regressions": regressions,
            "test_used": False}


def write_csv(path, rows, fields=None, replace=True):
    require(rows, f"No rows for {path}")
    fields = fields or list(rows[0])
    with Path(path).open("w" if replace else "x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def deterministic_config(snapshots, test_audit, fit_signature, partial_signature):
    return {
        "schema_version": 1, "seed": SEED, "features": FEATURES, "labels": list(LABELS),
        "architecture": [63, *HYPERPARAMETERS["hidden_layer_sizes"], 24],
        "hyperparameters": HYPERPARAMETERS,
        "standardization": "StandardScaler fit only on TRAIN; same scaler transforms VALIDATION; TEST untouched.",
        "early_stopping": "External pre-split VALIDATION macro-F1; sklearn internal early_stopping disabled.",
        "weighted_formula": "N_train / (24 * number_of_train_groups_in_class * train_group_size)",
        "sample_weight_support": {"fit_signature": str(fit_signature),
                                  "partial_fit_signature": str(partial_signature),
                                  "supported": True, "fallback_used": False},
        "selection_policy": {"primary": "VALIDATION macro-F1",
                             "tie_breakers": ["minimum_class_recall", "accuracy"],
                             "minimum_recall_guard": RECALL_FLOOR,
                             "material_recall_regression_guard": MATERIAL_RECALL_REGRESSION},
        "libraries": {"python": platform.python_version(), "numpy": np.__version__,
                      "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                      "joblib": version("joblib")},
        "input_snapshots": snapshots, "test_access": test_audit,
        "implementation": {"script": signature(Path(__file__).resolve()),
                           "tests": signature(ROOT / "tests" / "test_train_compare_mlp_validation.py")}}


def check_only():
    split_manifest, _, snapshots, test_audit, fit_sig, partial_sig = verify_inputs()
    train, validation, _, y_train, _, _, group_ids, group_table = load_train_validation()
    weights, class_groups, _ = build_group_class_weights(y_train, group_ids, group_table)
    require(abs(weights.mean() - 1.0) < 1e-12, "Weight check failed")
    require(not OUTPUT_DIR.exists(), "Training output already exists; inspect/resume instead of check-only")
    print(json.dumps({"status": "CHECK_ONLY_PASS", "training_started": False,
                      "output_created": False, "train_samples": len(train),
                      "validation_samples": len(validation), "test_access": test_audit,
                      "classes": len(class_groups), "features": len(FEATURES),
                      "sample_weight_supported": True,
                      "scikit_learn": sklearn.__version__, "input_hashes_verified": len(snapshots)}))


def train(output_dir):
    split_manifest, _, snapshots, test_audit, fit_sig, partial_sig = verify_inputs()
    for path in (ROOT / "ai/models").glob("*"):
        if path.is_file():
            snapshots[path.relative_to(ROOT).as_posix()] = sha256(path)
    config = deterministic_config(snapshots, test_audit, fit_sig, partial_sig)
    final_manifest_path = output_dir / "training_manifest.json"
    if output_dir.exists() and final_manifest_path.exists():
        manifest = load_json(final_manifest_path)
        require(manifest["status"] in {"VALIDATION_COMPARISON_COMPLETE",
                                       "VALIDATION_COMPARISON_REQUIRES_REVIEW"},
                "Existing final manifest has invalid status")
        for item in manifest["outputs"]:
            path = output_dir / item["path"]
            require(path.is_file() and path.stat().st_size == item["bytes"] and
                    sha256(path) == item["sha256"], f"Completed output changed: {path}")
        print(f"STATUS={manifest['status']}")
        print(f"SELECTED_MODEL={manifest['selection']['selected_model']}")
        print(f"MANIFEST={final_manifest_path}")
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

    (train_frame, validation_frame, x_train, y_train, x_validation, y_validation,
     train_group_ids, group_table) = load_train_validation()
    weights, class_groups, group_sizes = build_group_class_weights(
        y_train, train_group_ids, group_table)
    scaler_path = output_dir / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        require(int(scaler.n_samples_seen_) == len(train_frame), "Scaler checkpoint is not TRAIN-only")
    else:
        scaler = StandardScaler().fit(x_train)
        require(int(scaler.n_samples_seen_) == len(train_frame), "Scaler fit count mismatch")
        atomic_joblib(scaler_path, scaler)
    x_train_scaled = scaler.transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)

    weight_rows = [{"file": file, "label": label, "group_id": group_id,
                    "group_size": group_sizes[group_id], "baseline_weight": 1.0,
                    "weighted_sample_weight": float(weight)}
                   for file, label, group_id, weight in zip(
                       train_frame.file, train_frame.label, train_group_ids, weights, strict=True)]
    write_csv(output_dir / "training_weights.csv", weight_rows)

    states = {}
    completed = []
    for name, sample_weight in [("baseline", None), ("weighted", weights)]:
        states[name] = train_variant(name, x_train_scaled, y_train,
                                     x_validation_scaled, y_validation, sample_weight,
                                     output_dir / "checkpoints" / name)
        completed.append(name)
        atomic_json(output_dir / "run_state.json",
                    {"status": "TRAINING", "completed_variants": completed})

    evaluations = {}
    for name, state in states.items():
        metrics, per_class, matrix, directed, symmetric, predictions = evaluate(
            state["best_model"], x_validation_scaled, y_validation, validation_frame)
        evaluations[name] = {"metrics": metrics, "per_class": per_class,
                             "best_epoch": state["best_epoch"],
                             "epochs_run": state["epoch"],
                             "stop_reason": state["stop_reason"],
                             "top_directed_confusions": directed[:20],
                             "top_symmetric_confusions": symmetric[:20]}
        atomic_joblib(output_dir / f"{name}_model.joblib", state["best_model"])
        write_csv(output_dir / f"{name}_training_history.csv", state["history"])
        write_csv(output_dir / f"{name}_per_class_metrics.csv", per_class)
        with (output_dir / f"{name}_confusion_matrix.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true\\predicted", *LABELS])
            for label, row in zip(LABELS, matrix, strict=True):
                writer.writerow([label, *map(int, row)])
        write_csv(output_dir / f"{name}_validation_predictions.csv", predictions)
        write_json(output_dir / f"{name}_confusion_analysis.json",
                   {"top_directed": directed[:50], "top_symmetric": symmetric[:50],
                    "M_N_T": {label: {
                        "recall": per_class[LABEL_TO_ID[label]]["recall"],
                        "top_misclassifications": [row for row in directed
                                                   if row["true_label"] == label][:5]}
                        for label in ("M", "N", "T")}}, replace=True)

    selection = select_model(evaluations)
    selected = selection["selected_model"]
    if selected:
        shutil.copyfile(output_dir / f"{selected}_model.joblib",
                        output_dir / "selected_model.joblib")
    write_json(output_dir / "label_mapping.json",
               {"classes_in_model_order": list(LABELS), "label_to_index": LABEL_TO_ID,
                "index_to_label": {str(index): label for label, index in LABEL_TO_ID.items()}},
               replace=True)
    write_json(output_dir / "evaluation_summary.json",
               {"train_samples": len(train_frame), "validation_samples": len(validation_frame),
                "baseline": evaluations["baseline"], "weighted": evaluations["weighted"],
                "selection": selection, "test_access": test_audit}, replace=True)
    write_json(output_dir / "selected_model.json",
               {"selected": selected,
                "model_path": "selected_model.joblib" if selected else None,
                "source_model_path": f"{selected}_model.joblib" if selected else None,
                "scaler_path": "scaler.joblib", "label_mapping_path": "label_mapping.json",
                "selection": selection}, replace=True)
    (output_dir / "REPORT.md").write_text(
        "# MLP validation comparison\n\n"
        f"TRAIN: {len(train_frame):,}; VALIDATION: {len(validation_frame):,}. TEST was hash-checked only.\n\n"
        f"Baseline accuracy/macro-F1: {evaluations['baseline']['metrics']['accuracy']:.6f}/"
        f"{evaluations['baseline']['metrics']['macro_f1']:.6f}.\n\n"
        f"Weighted accuracy/macro-F1: {evaluations['weighted']['metrics']['accuracy']:.6f}/"
        f"{evaluations['weighted']['metrics']['macro_f1']:.6f}.\n\n"
        f"Selection status: {selection['selection_status']}; selected: {selected}.\n\n"
        "StandardScaler was fit only on TRAIN. Internal random validation was disabled. "
        "No TEST evaluation, augmentation, duplication, deletion or preprocessing occurred.\n",
        encoding="utf-8")

    test_run = subprocess.run(
        [sys.executable, "-B", "-m", "unittest",
         "tests.test_train_compare_mlp_validation", "-q"],
        cwd=ROOT, capture_output=True, text=True, check=False)
    require(test_run.returncode == 0, "Training unit tests failed")
    write_json(output_dir / "test_report.json",
               {"status": "PASS", "unit_tests_run": 6,
                "module": "tests.test_train_compare_mlp_validation"}, replace=True)
    for relative, expected in snapshots.items():
        require(sha256(ROOT / relative) == expected, f"Protected input changed: {relative}")
    integrity = {"status": "PASS", "train_samples": len(train_frame),
                 "validation_samples": len(validation_frame),
                 "test_csv_access": "SIZE_AND_SHA256_ONLY", "test_evaluated": False,
                 "train_validation_sample_overlap": 0,
                 "train_validation_group_overlap": 0,
                 "features_used": 63, "labels": 24,
                 "scaler_train_rows": int(scaler.n_samples_seen_),
                 "sample_weight_supported_natively": True,
                 "sample_weight_fallback_used": False,
                 "weighted_mean": float(weights.mean()),
                 "weighted_class_totals_equal": True,
                 "weighted_group_totals_equal_within_class": True,
                 "source_files_modified": 0, "old_model_files_modified": 0,
                 "augmentation_or_row_resampling": False, "unit_tests": "PASS"}
    write_json(output_dir / "integrity_report.json", integrity, replace=True)
    final_status = ("VALIDATION_COMPARISON_COMPLETE" if selected else
                    "VALIDATION_COMPARISON_REQUIRES_REVIEW")
    atomic_json(output_dir / "run_state.json",
                {"status": "COMPLETE", "completed_variants": ["baseline", "weighted"],
                 "selected": selected})
    output_files = sorted(path for path in output_dir.rglob("*")
                          if path.is_file() and path.name != "training_manifest.json")
    manifest = {"schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": final_status, "selection": selection,
                "integrity": integrity, "test_evaluated": False,
                "remaining_blockers": [] if selected else ["RECALL_GUARD_REQUIRES_REVIEW"],
                "outputs": [{"path": path.relative_to(output_dir).as_posix(),
                             "bytes": path.stat().st_size, "sha256": sha256(path)}
                            for path in output_files]}
    write_json(final_manifest_path, manifest, replace=True)
    print(f"STATUS={final_status}")
    print(f"SELECTED_MODEL={selected}")
    print(f"MANIFEST={final_manifest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="Validate dependencies/data without fitting or writing training outputs")
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
