"""Evaluate the single J/Z experimental baseline; never fit or promote a model.

The baseline has no candidate selection. Validation and test are reported once
from the saved source-video split, with signer-independence explicitly unknown.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.dynamic.train_dynamic import ROOT, load_checked_data, local_path, sha256


def validate_experiment(output: Path) -> tuple[object, list[dict], np.ndarray, np.ndarray, dict]:
    from joblib import load
    from sklearn.pipeline import Pipeline

    training = json.loads((output / "training_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((output / "config_snapshot.json").read_text(encoding="utf-8"))
    split = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
    for name, expected_hash in training["artifacts"].items():
        if sha256(output / name) != expected_hash:
            raise ValueError(f"Experiment artifact hash mismatch: {name}")
    if (local_path(config["output_dir"]) != output
            or training.get("status") != "EXPERIMENTAL_NOT_PRODUCTION"
            or training.get("data", {}).get("signer_session_metadata")
            != "ABSENT; source-video split is not signer-independent"):
        raise ValueError("Experiment lineage mismatch")
    records, X, y, audit = load_checked_data(config, materialize=True)
    if audit["input_sha256"] != training["data"]["input_sha256"]:
        raise ValueError("Input dataset hash changed since training")
    model = load(output / "model_pipeline.joblib")
    if (not isinstance(model, Pipeline) or list(model.named_steps) != ["scaler", "classifier"]
            or model.n_features_in_ != 1104
            or list(model.classes_) != ["J", "Z"]):
        raise ValueError("Unexpected pipeline/features/classes")
    source_index = {row["source_path"]: index for index, row in enumerate(records)}
    if len(source_index) != 640:
        raise ValueError("Duplicate source after data load")
    all_split_sources: set[str] = set()
    indices_by_split: dict[str, list[int]] = {}
    for name, expected_size in (("train", 512), ("validation", 64), ("test", 64)):
        items = split["splits"][name]
        if len(items) != expected_size:
            raise ValueError(f"Split size mismatch: {name}")
        indices = []
        for item in items:
            source = item["source_path"]
            if source not in source_index or source in all_split_sources:
                raise ValueError(f"Missing/overlapping source in split: {source}")
            index = source_index[source]
            if records[index]["label"] != item["label"]:
                raise ValueError(f"Split label mismatch: {source}")
            all_split_sources.add(source)
            indices.append(index)
        indices_by_split[name] = indices
    if len(all_split_sources) != 640 or split.get("source_overlap") != 0:
        raise ValueError("Split does not cover all unique sources")
    return model, records, X, y, {"indices": indices_by_split, "training": training,
                                 "split": split, "audit": audit}


def evaluate(model, records: list[dict], X: np.ndarray, y: np.ndarray, info: dict) -> dict:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)

    results = {}
    for name, indices in info["indices"].items():
        X_part = X[indices]
        truth = y[indices]
        prediction = model.predict(X_part)
        probabilities = model.predict_proba(X_part)
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, prediction, labels=["J", "Z"], zero_division=0)
        errors = []
        for position, index in enumerate(indices):
            if truth[position] == prediction[position]:
                continue
            predicted = str(prediction[position])
            confidence = float(probabilities[position, list(model.classes_).index(predicted)])
            errors.append({"source_path": records[index]["source_path"],
                           "true_label": str(truth[position]), "predicted_label": predicted,
                           "confidence": confidence,
                           "probability_J": float(probabilities[position, 0]),
                           "probability_Z": float(probabilities[position, 1])})
        results[name] = {
            "samples": len(indices), "class_counts": dict(Counter(truth)),
            "accuracy": float(accuracy_score(truth, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "per_class": {label: {"precision": float(precision[position]),
                                   "recall": float(recall[position]), "f1": float(f1[position]),
                                   "support": int(support[position])}
                          for position, label in enumerate(("J", "Z"))},
            "confusion_matrix": {"labels": ["J", "Z"],
                                 "rows_true_columns_predicted": confusion_matrix(
                                     truth, prediction, labels=["J", "Z"]).tolist()},
            "error_count": len(errors), "errors": errors,
        }
    return {"status": "EXPERIMENTAL_JZ_BINARY_EVALUATION",
            "model": "StandardScaler -> LogisticRegression", "test_used_for_tuning": False,
            "signer_independent_test": False,
            "limitation": "Signer/session identities unavailable; source-video split may leak signer style.",
            "splits": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--experiment-dir", type=Path,
                        default=ROOT / "data/experiments/dynamic_jz_binary_baseline_v1")
    args = parser.parse_args(argv)
    try:
        output = args.experiment_dir.resolve()
        if not output.is_relative_to(ROOT.resolve()):
            raise ValueError("Experiment directory must remain inside project")
        model, records, X, y, info = validate_experiment(output)
        metrics_path = output / "metrics.json"
        if metrics_path.exists():
            raise FileExistsError(f"No overwrite of existing evaluation: {metrics_path}")
        print(json.dumps({"status": "EVALUATION_PREFLIGHT_PASS", "samples": len(records),
                          "split_sizes": {name: len(values) for name, values in info["indices"].items()},
                          "model_loaded": True, "evaluation_performed": False}))
        if args.check_only:
            return 0
        if input("Start evaluation? [y/N] ").strip().lower() != "y":
            print("CANCELLED: no evaluation report written")
            return 0
        report = evaluate(model, records, X, y, info)
        report["training_manifest_sha256"] = sha256(output / "training_manifest.json")
        report["split_manifest_sha256"] = sha256(output / "split_manifest.json")
        metrics_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": "EVALUATION_COMPLETE", "metrics": metrics_path.relative_to(ROOT).as_posix()}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
