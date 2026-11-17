"""Independent Metrics Calculator for Webcam Evaluation.

Computes classification metrics, end-to-end character metrics, safety metrics (FAR/FRR),
latency statistics, and sequence accuracy across evaluation trials.
STRICTLY READ-ONLY with respect to production code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


def compute_latency_stats(latencies: Sequence[float]) -> Dict[str, Union[float, str]]:
    """Compute standard latency percentiles and mean in milliseconds."""
    valid = [float(x) for x in latencies if x is not None and np.isfinite(x) and x >= 0]
    if not valid:
        return {"mean_ms": "N/A", "p50_ms": "N/A", "p95_ms": "N/A", "min_ms": "N/A", "max_ms": "N/A"}
    arr = np.asarray(valid, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "sample_count": len(valid),
    }


def calculate_metrics(trials: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics from a sequence of trial dictionaries.

    Ensures that uncommitted trials remain in the denominator of End-to-End Character Accuracy,
    and Macro-F1 is strictly computed as the arithmetic mean of per-class F1 scores.
    """
    if not trials:
        return {
            "status": "EMPTY_DATA",
            "total_trials": 0,
            "classification_metrics": "N/A",
            "end_to_end_character_metrics": "N/A",
            "safety_metrics": "N/A",
            "latency_metrics": "N/A",
            "sequence_metrics": "N/A",
            "outcome_breakdown": {},
        }

    total_trials = len(trials)

    # 1. Outcome Breakdown
    outcome_counts: Dict[str, int] = {}
    for t in trials:
        outcome = str(t.get("primary_outcome", "UNKNOWN"))
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

    # 2. End-to-End Character Metrics
    # Every trial planned is in the denominator!
    success_count = sum(1 for t in trials if t.get("primary_outcome") == "SUCCESS")
    character_accuracy = float(success_count / total_trials) if total_trials > 0 else 0.0

    miss_count = sum(1 for t in trials if t.get("primary_outcome") in ("TIMEOUT_MISSED", "LANDMARK_DETECTION_FAILURE"))
    miss_rate = float(miss_count / total_trials) if total_trials > 0 else 0.0

    false_commit_count = sum(1 for t in trials if t.get("primary_outcome") == "CLASSIFICATION_ERROR")
    false_commit_rate = float(false_commit_count / total_trials) if total_trials > 0 else 0.0

    # Duplicate commit rate: denominator is trials that committed at least once
    committed_trials_count = sum(1 for t in trials if int(t.get("commit_count", 0)) > 0)
    duplicate_count = sum(1 for t in trials if t.get("primary_outcome") == "DUPLICATE_COMMIT" or int(t.get("commit_count", 0)) > 1)
    duplicate_commit_rate = float(duplicate_count / committed_trials_count) if committed_trials_count > 0 else 0.0

    # False rejection: valid positive trials rejected by validator or hold abort
    positive_trials_count = sum(1 for t in trials if str(t.get("ground_truth", "")).upper() not in ("NEGATIVE", "NOISE", "IDLE"))
    false_rejection_count = sum(1 for t in trials if str(t.get("ground_truth", "")).upper() not in ("NEGATIVE", "NOISE", "IDLE") and t.get("primary_outcome") in ("VALIDATOR_REJECTION", "HOLD_ABORTED"))
    false_rejection_rate = float(false_rejection_count / positive_trials_count) if positive_trials_count > 0 else 0.0

    # 3. Classification Metrics (Per-class Precision, Recall, F1, and Macro-F1)
    # Collect all unique ground truth labels that represent actual positive classes
    classes_set = set()
    for t in trials:
        gt = str(t.get("ground_truth", "")).strip()
        if gt and gt.upper() not in ("NEGATIVE", "NOISE", "IDLE"):
            classes_set.add(gt)
        # Also include committed classes if present
        commit = str(t.get("commit_text", "")).strip()
        if commit and commit.upper() not in ("NEGATIVE", "NOISE", "IDLE", "NONE"):
            classes_set.add(commit)

    sorted_classes = sorted(list(classes_set))
    per_class_metrics: Dict[str, Dict[str, float]] = {}
    f1_list: List[float] = []
    precision_list: List[float] = []
    recall_list: List[float] = []

    # Build confusion matrix: rows = true, cols = predicted
    class_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    conf_matrix = np.zeros((len(sorted_classes), len(sorted_classes)), dtype=np.int64)
    for t in trials:
        gt = str(t.get("ground_truth", "")).strip()
        pred = str(t.get("commit_text", "")).strip()
        if not pred or t.get("primary_outcome") in ("TIMEOUT_MISSED", "LANDMARK_DETECTION_FAILURE"):
            pred = ""
        if gt in class_to_idx and pred in class_to_idx:
            conf_matrix[class_to_idx[gt], class_to_idx[pred]] += 1

    for c in sorted_classes:
        tp = 0
        fp = 0
        fn = 0
        for t in trials:
            gt = str(t.get("ground_truth", "")).strip()
            # Determine effective prediction: either commit_text or primary model prediction
            pred = str(t.get("commit_text", "")).strip()
            if not pred or t.get("primary_outcome") in ("TIMEOUT_MISSED", "LANDMARK_DETECTION_FAILURE"):
                pred = ""

            if gt == c:
                if pred == c and t.get("primary_outcome") == "SUCCESS":
                    tp += 1
                else:
                    fn += 1
            else:
                if pred == c:
                    fp += 1

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class_metrics[c] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": tp + fn,
        }
        f1_list.append(f1)
        precision_list.append(prec)
        recall_list.append(rec)

    # Strictly compute Macro-F1 as arithmetic mean of per-class F1
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    macro_precision = float(np.mean(precision_list)) if precision_list else 0.0
    macro_recall = float(np.mean(recall_list)) if recall_list else 0.0

    # 4. SPACE Safety Metrics
    space_trials = [t for t in trials if t.get("branch") == "SPACE" or str(t.get("ground_truth", "")).upper() == "SPACE"]
    space_positives = [t for t in space_trials if str(t.get("ground_truth", "")).upper() == "SPACE"]
    space_negatives = [t for t in trials if str(t.get("ground_truth", "")).upper() != "SPACE"]

    far_space: Union[float, str] = "N/A"
    frr_space: Union[float, str] = "N/A"
    if space_negatives:
        false_space_commits = sum(1 for t in space_negatives if str(t.get("commit_text", "")).strip() in (" ", "SPACE"))
        far_space = float(false_space_commits / len(space_negatives))
    if space_positives:
        failed_space_commits = sum(1 for t in space_positives if str(t.get("commit_text", "")).strip() not in (" ", "SPACE"))
        frr_space = float(failed_space_commits / len(space_positives))

    # 5. J/Z Safety Metrics
    jz_trials = [t for t in trials if t.get("branch") == "JZ" or str(t.get("ground_truth", "")).upper() in ("J", "Z")]
    jz_positives = [t for t in jz_trials if str(t.get("ground_truth", "")).upper() in ("J", "Z")]
    jz_negatives = [t for t in trials if str(t.get("ground_truth", "")).upper() not in ("J", "Z")]

    far_jz: Union[float, str] = "N/A"
    frr_jz: Union[float, str] = "N/A"
    if jz_negatives:
        false_jz_commits = sum(1 for t in jz_negatives if str(t.get("commit_text", "")).strip().upper() in ("J", "Z"))
        far_jz = float(false_jz_commits / len(jz_negatives))
    if jz_positives:
        failed_jz_commits = sum(1 for t in jz_positives if str(t.get("commit_text", "")).strip().upper() not in ("J", "Z"))
        frr_jz = float(failed_jz_commits / len(jz_positives))

    # 6. Latency Metrics
    inference_latencies = [t.get("inference_latency_ms") for t in trials if t.get("inference_latency_ms") is not None]
    coordinator_latencies = [t.get("coordinator_latency_ms") for t in trials if t.get("coordinator_latency_ms") is not None]
    commit_latencies = [t.get("elapsed_time_ms", t.get("elapsed_time", 0.0) * 1000.0) for t in trials if t.get("primary_outcome") == "SUCCESS"]

    latency_metrics = {
        "model_inference": compute_latency_stats(inference_latencies),
        "coordinator_execution": compute_latency_stats(coordinator_latencies),
        "end_to_end_commit": compute_latency_stats(commit_latencies),
    }

    # 7. Sequence Metrics (if word/sequence ground truth is present)
    sequence_trials = [t for t in trials if t.get("sequence_ground_truth") is not None]
    sequence_metrics: Union[Dict[str, Any], str] = "N/A"
    if sequence_trials:
        exact_seq_success = sum(1 for t in sequence_trials if t.get("sequence_ground_truth") == t.get("sequence_committed"))
        exact_seq_acc = float(exact_seq_success / len(sequence_trials))
        sequence_metrics = {
            "total_sequences": len(sequence_trials),
            "exact_sequence_accuracy": exact_seq_acc,
            "sequence_error_rate": float(1.0 - exact_seq_acc),
        }

    return {
        "status": "COMPUTED",
        "total_trials": total_trials,
        "classification_metrics": {
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "per_class": per_class_metrics,
            "confusion_matrix": {
                "labels": sorted_classes,
                "matrix": conf_matrix.tolist(),
            },
        },
        "end_to_end_character_metrics": {
            "character_accuracy": character_accuracy,
            "miss_rate": miss_rate,
            "false_commit_rate": false_commit_rate,
            "duplicate_commit_rate": duplicate_commit_rate,
            "false_rejection_rate": false_rejection_rate,
        },
        "safety_metrics": {
            "far_space": far_space,
            "frr_space": frr_space,
            "far_jz": far_jz,
            "frr_jz": frr_jz,
        },
        "latency_metrics": latency_metrics,
        "sequence_metrics": sequence_metrics,
        "outcome_breakdown": outcome_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate evaluation metrics from webcam trial logs")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input JSON or JSONL file with trials")
    parser.add_argument("--output", "-o", type=Path, help="Optional output JSON path")
    args = parser.parse_args()

    input_path = args.input
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    trials: List[Dict[str, Any]] = []
    if input_path.suffix.lower() == ".jsonl":
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    trials.append(json.loads(line))
    else:
        data = json.loads(input_path.read_text(encoding="utf-8"))
        trials = data if isinstance(data, list) else data.get("trials", [])

    results = calculate_metrics(trials)
    output_str = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_str, encoding="utf-8")
        print(f"Metrics written to {args.output}")
    else:
        print(output_str)


if __name__ == "__main__":
    main()
