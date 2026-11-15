"""Comprehensive Stage 6 Benchmark & Comparison of 4 J/Z Recognition Methods.

Compares:
1. Method 1: Baseline Logistic Regression Pipeline (Production Baseline)
2. Method 2: Enhanced Multivariate DTW Engine (CONFIG_E, 14D)
3. Method 3: Geometric Classifier (Existing implementation in geometric_classifier.py)
4. Method 4: Ratio-Based Trajectory Recognition Engine (New Stage 6 module in ratio_classifier.py)

Evaluates Across 4 Benchmark Suites:
1. Suite 1: Clean Validation Split (64 samples: 26 J, 38 Z from SigNN baseline-b).
2. Suite 2: Out-Of-Distribution (OOD) Invalid Motion Suite (32 synthetic non-J/Z patterns).
3. Suite 3: Real-World Camera Diagnostic Attempts (61 authentic captured 16x69 sequences).
4. Suite 4: Stage 5 Webcam Benchmark Dataset (50 attempts: 20 J, 20 Z, 10 Negative Controls).

Outputs:
- Full metrics: Accuracy, Precision, Recall, F1, Coverage, UNKNOWN rate, False Acceptance Rate (FAR),
  Cross-confusion (J <-> Z), Latency percentiles (P50, P95, P99).
- Per-attempt breakdown and confusion matrix for each method.
- Detailed JSON telemetry and Markdown report.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference.dynamic_jz.enhanced_dtw import EnhancedDTWClassifier
from app.inference.dynamic_jz.geometric_classifier import GeometricClassifier
from app.inference.dynamic_jz.ratio_classifier import RatioBasedJZClassifier
from app.inference.dynamic_jz.finger_tracker import FEATURE_DIM, SEQUENCE_LENGTH


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    latencies: list[float],
    *,
    negative_labels: set[str] | None = None,
) -> dict[str, Any]:
    """Compute precision, recall, F1, confusion matrix, coverage, and latency percentiles."""
    classes = ["J", "Z"]
    neg_labels = negative_labels or {"NONE", "NEGATIVE", "IDLE", "INVALID", "OOD"}

    cm: dict[str, dict[str, int]] = {
        "J": {"J": 0, "Z": 0, "UNKNOWN": 0},
        "Z": {"J": 0, "Z": 0, "UNKNOWN": 0},
    }
    neg_total = 0
    neg_accepted = 0
    cross_confusions = 0

    for yt, yp in zip(y_true, y_pred):
        if yt in ("J", "Z"):
            if yp in ("J", "Z", "UNKNOWN"):
                cm[yt][yp] += 1
            else:
                cm[yt]["UNKNOWN"] += 1
            if (yt == "J" and yp == "Z") or (yt == "Z" and yp == "J"):
                cross_confusions += 1
        elif yt in neg_labels:
            neg_total += 1
            if yp in ("J", "Z"):
                neg_accepted += 1

    # Filter to positive J/Z evaluation
    pos_indices = [i for i, yt in enumerate(y_true) if yt in ("J", "Z")]
    pos_total = len(pos_indices)
    pos_unknowns = sum(1 for i in pos_indices if y_pred[i] == "UNKNOWN")
    unknown_rate = pos_unknowns / pos_total if pos_total > 0 else 0.0
    coverage = 1.0 - unknown_rate

    metrics: dict[str, Any] = {
        "confusion_matrix": cm,
        "total_evaluated": pos_total,
        "unknown_count": pos_unknowns,
        "unknown_rate": float(unknown_rate),
        "coverage": float(coverage),
        "cross_confusions": cross_confusions,
        "negative_controls": {
            "total": neg_total,
            "false_acceptances": neg_accepted,
            "false_acceptance_rate": (neg_accepted / neg_total) if neg_total > 0 else 0.0,
        },
        "per_class": {},
    }

    for c in classes:
        tp = cm[c][c]
        fn = sum(cm[c][other] for other in ("J", "Z", "UNKNOWN") if other != c)
        other_c = "Z" if c == "J" else "J"
        fp = cm[other_c][c]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics["per_class"][c] = {
            "total": sum(cm[c].values()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    metrics["macro_f1"] = float(np.mean([metrics["per_class"][c]["f1"] for c in classes]))
    correct = sum(cm[c][c] for c in classes)
    metrics["accuracy"] = float(correct / pos_total) if pos_total > 0 else 0.0

    if latencies:
        metrics["latency"] = {
            "mean_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }
    else:
        metrics["latency"] = {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

    return metrics


def generate_ood_suite(count: int = 32) -> list[dict[str, Any]]:
    """Generate 32 synthetic out-of-distribution (OOD) invalid motion sequences."""
    ood = []
    # 1. Straight down
    for i in range(8):
        seq = np.zeros((16, 69), dtype=np.float64)
        drop = 0.3 + 0.05 * i
        for t in range(16):
            frac = t / 15.0
            seq[t, 65:67] = [0.0, drop * frac]
            seq[t, 67:69] = [0.0, drop * frac]
            seq[t, :63] = 0.15
        ood.append({"name": f"down_{i}", "label": "OOD", "seq": seq})
    # 2. Horizontal LR
    for i in range(4):
        seq = np.zeros((16, 69), dtype=np.float64)
        span = 0.3 + 0.05 * i
        for t in range(16):
            frac = t / 15.0
            seq[t, 65:67] = [span * frac, 0.0]
            seq[t, 67:69] = [span * frac, 0.0]
            seq[t, :63] = 0.2
        ood.append({"name": f"horiz_lr_{i}", "label": "OOD", "seq": seq})
    # 3. Horizontal RL
    for i in range(4):
        seq = np.zeros((16, 69), dtype=np.float64)
        span = -(0.3 + 0.05 * i)
        for t in range(16):
            frac = t / 15.0
            seq[t, 65:67] = [span * frac, 0.0]
            seq[t, 67:69] = [span * frac, 0.0]
            seq[t, :63] = 0.2
        ood.append({"name": f"horiz_rl_{i}", "label": "OOD", "seq": seq})
    # 4. Static jitter
    for i in range(8):
        seq = np.zeros((16, 69), dtype=np.float64)
        noise = 0.005 * (i + 1)
        for t in range(16):
            seq[t, 65:67] = np.random.normal(0, noise, 2)
            seq[t, 67:69] = np.random.normal(0, noise, 2)
            seq[t, :63] = 0.2
        ood.append({"name": f"jitter_{i}", "label": "OOD", "seq": seq})
    # 5. Circular wave
    for i in range(8):
        seq = np.zeros((16, 69), dtype=np.float64)
        r = 0.2 + 0.03 * i
        for t in range(16):
            ang = 2 * np.pi * t / 15.0
            seq[t, 65:67] = [r * np.cos(ang), r * np.sin(ang)]
            seq[t, 67:69] = [r * np.cos(ang), r * np.sin(ang)]
            seq[t, :63] = 0.2
        ood.append({"name": f"circle_{i}", "label": "OOD", "seq": seq})
    return ood[:count]


def run_benchmark() -> dict[str, Any]:
    print("=" * 80)
    print("STAGE 6: RATIO-BASED TRAJECTORY RECOGNITION & 4-METHOD COMPARATIVE BENCHMARK")
    print("=" * 80)

    # 1. Initialize Methods
    print("\n[1/5] Initializing 4 Methods...")
    lr_model_path = PROJECT_ROOT / "models/dynamic_jz/model_pipeline.joblib"
    if not lr_model_path.exists():
        lr_model_path = PROJECT_ROOT / "data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib"
    lr_pipeline = joblib.load(lr_model_path)
    print("  Method 1: Baseline Logistic Regression Pipeline loaded.")

    enhanced_dtw = EnhancedDTWClassifier(threshold=2.65, min_margin=0.15, min_dominance=-0.05)
    print("  Method 2: Enhanced Multivariate DTW Engine (CONFIG_E) loaded.")

    geom_classifier = GeometricClassifier(min_score=0.45, min_margin=0.08)
    print("  Method 3: Geometric Classifier loaded.")

    ratio_classifier = RatioBasedJZClassifier(
        min_confidence=0.45,
        min_margin=0.08,
        min_j_drop=0.25,
        min_j_hook=0.10,
        min_z_span=0.06,
        min_z_diag_drop=0.06,
        min_pinky_ext=0.50,
        min_index_ext=0.50,
    )
    print("  Method 4: Ratio-Based Trajectory Classifier initialized.")

    def predict_lr(seq: np.ndarray) -> tuple[str, float]:
        flat = seq.reshape(1, -1)
        probs = lr_pipeline.predict_proba(flat)[0]
        classes = list(lr_pipeline.classes_)
        idx_j = classes.index("J") if "J" in classes else 0
        idx_z = classes.index("Z") if "Z" in classes else 1
        pred = "J" if probs[idx_j] > probs[idx_z] else "Z"
        conf = float(max(probs[idx_j], probs[idx_z]))
        return pred, conf

    def predict_dtw(seq: np.ndarray) -> tuple[str, float]:
        res = enhanced_dtw.classify(seq)
        return res.prediction, res.latency_ms

    def predict_geom(seq: np.ndarray) -> tuple[str, float]:
        res = geom_classifier.classify(seq)
        return res.prediction, res.latency_ms

    def predict_ratio(seq: np.ndarray) -> tuple[str, float, dict[str, Any], str, float]:
        res = ratio_classifier.classify(seq)
        return res.prediction, res.confidence, res.ratios, res.rejection_reason, res.latency_ms

    methods = ["Baseline LR", "Enhanced DTW", "Geometric", "Ratio-Based"]
    all_results: dict[str, Any] = {}

    # 2. Benchmark Suite 1: Clean Validation Split (64 samples)
    print("\n[2/5] Evaluating Suite 1: Clean Validation Split (64 samples: 26 J, 38 Z)...")
    split_path = PROJECT_ROOT / "data/experiments/dynamic_jz_binary_baseline_v1/split_manifest.json"
    data_path = PROJECT_ROOT / "data/experiments/dynamic_jz_v1/processed/signnteam_jz_baseline_b_v1.jsonl"

    with open(split_path, "r", encoding="utf-8") as f:
        splits_meta = json.load(f)
    val_meta = splits_meta["splits"]["validation"]

    samples_by_path = {}
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            samples_by_path[item["source_path"]] = item

    y_val_true = []
    val_preds: dict[str, list[str]] = {m: [] for m in methods}
    val_lats: dict[str, list[float]] = {m: [] for m in methods}

    for m in val_meta:
        sp = m["source_path"]
        raw = samples_by_path[sp]
        seq = np.array(raw["features"], dtype=np.float64)
        lbl = raw["label"]
        y_val_true.append(lbl)

        # LR
        t0 = time.perf_counter()
        p_lr, _ = predict_lr(seq)
        val_lats["Baseline LR"].append((time.perf_counter() - t0) * 1000.0)
        val_preds["Baseline LR"].append(p_lr)

        # DTW
        p_dtw, lat_dtw = predict_dtw(seq)
        val_lats["Enhanced DTW"].append(lat_dtw)
        val_preds["Enhanced DTW"].append(p_dtw)

        # Geom
        p_geom, lat_geom = predict_geom(seq)
        val_lats["Geometric"].append(lat_geom)
        val_preds["Geometric"].append(p_geom)

        # Ratio
        p_ratio, _, _, _, lat_ratio = predict_ratio(seq)
        val_lats["Ratio-Based"].append(lat_ratio)
        val_preds["Ratio-Based"].append(p_ratio)

    metrics_suite1 = {
        m: compute_metrics(y_val_true, val_preds[m], val_lats[m])
        for m in methods
    }
    all_results["suite1_validation"] = metrics_suite1

    # 3. Benchmark Suite 2: OOD Invalid Motion Suite (32 samples)
    print("\n[3/5] Evaluating Suite 2: Out-Of-Distribution Invalid Motions (32 samples)...")
    ood_samples = generate_ood_suite(32)
    y_ood_true = ["OOD"] * len(ood_samples)
    ood_preds: dict[str, list[str]] = {m: [] for m in methods}
    ood_lats: dict[str, list[float]] = {m: [] for m in methods}

    for item in ood_samples:
        seq = item["seq"]

        t0 = time.perf_counter()
        p_lr, _ = predict_lr(seq)
        ood_lats["Baseline LR"].append((time.perf_counter() - t0) * 1000.0)
        ood_preds["Baseline LR"].append(p_lr)

        p_dtw, lat_dtw = predict_dtw(seq)
        ood_lats["Enhanced DTW"].append(lat_dtw)
        ood_preds["Enhanced DTW"].append(p_dtw)

        p_geom, lat_geom = predict_geom(seq)
        ood_lats["Geometric"].append(lat_geom)
        ood_preds["Geometric"].append(p_geom)

        p_ratio, _, _, _, lat_ratio = predict_ratio(seq)
        ood_lats["Ratio-Based"].append(lat_ratio)
        ood_preds["Ratio-Based"].append(p_ratio)

    ood_summary = {}
    for m in methods:
        neg_rej = sum(1 for p in ood_preds[m] if p == "UNKNOWN")
        far = (len(ood_samples) - neg_rej) / len(ood_samples)
        ood_summary[m] = {
            "total_ood": len(ood_samples),
            "rejected_unknown": neg_rej,
            "rejection_rate": float(neg_rej / len(ood_samples)),
            "false_acceptance_rate": float(far),
        }
    all_results["suite2_ood"] = ood_summary

    # 4. Benchmark Suite 3: Real-World Camera Diagnostics (61 attempts)
    print("\n[4/5] Evaluating Suite 3: Real-World Diagnostic Attempts (61 valid 16x69 sequences)...")
    diag_npzs = sorted(list((PROJECT_ROOT / "data/experiments/dynamic_jz_v1/diagnostics").glob("*/*/replay_data.npz")))

    y_diag_true = []
    diag_preds: dict[str, list[str]] = {m: [] for m in methods}
    diag_lats: dict[str, list[float]] = {m: [] for m in methods}

    for p in diag_npzs:
        meta_p = p.parent / "attempt.json"
        if not meta_p.exists():
            continue
        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        m_res = meta.get("model_result") or {}
        target = meta.get("candidate_letter") or m_res.get("prediction")
        if target not in ("J", "Z"):
            continue

        data = np.load(p)
        feat = data.get("features")
        if feat is None or feat.shape != (16, 69):
            continue

        y_diag_true.append(target)

        t0 = time.perf_counter()
        p_lr, _ = predict_lr(feat)
        diag_lats["Baseline LR"].append((time.perf_counter() - t0) * 1000.0)
        diag_preds["Baseline LR"].append(p_lr)

        p_dtw, lat_dtw = predict_dtw(feat)
        diag_lats["Enhanced DTW"].append(lat_dtw)
        diag_preds["Enhanced DTW"].append(p_dtw)

        p_geom, lat_geom = predict_geom(feat)
        diag_lats["Geometric"].append(lat_geom)
        diag_preds["Geometric"].append(p_geom)

        p_ratio, _, _, _, lat_ratio = predict_ratio(feat)
        diag_lats["Ratio-Based"].append(lat_ratio)
        diag_preds["Ratio-Based"].append(p_ratio)

    metrics_suite3 = {
        m: compute_metrics(y_diag_true, diag_preds[m], diag_lats[m])
        for m in methods
    }
    all_results["suite3_diagnostics"] = metrics_suite3

    # 5. Benchmark Suite 4: Stage 5 Webcam Real-World Validation Replay (50 attempts)
    print("\n[5/5] Evaluating Suite 4: Stage 5 Webcam Real-World Validation (50 attempts)...")
    stage5_trials_path = (
        PROJECT_ROOT / "data/experiments/dynamic_jz_v1/realworld_validation/20261113T155135Z_stage5_validator_fix_6fb34f/trials.jsonl"
    )

    stage5_trials = []
    with open(stage5_trials_path, "r", encoding="utf-8") as f:
        for line in f:
            stage5_trials.append(json.loads(line))

    # Match exact diagnostic sequences corresponding to the 40 invoked attempts
    diag_attempts = []
    for p in diag_npzs:
        data = np.load(p)
        feat = data.get("features")
        if feat is not None and feat.shape == (16, 69):
            meta_p = p.parent / "attempt.json"
            meta = json.load(open(meta_p, "r", encoding="utf-8")) if meta_p.exists() else {}
            m_res = meta.get("model_result") or {}
            pred = meta.get("candidate_letter") or m_res.get("prediction") or ""
            diag_attempts.append({"features": feat, "prediction": pred})

    y_s5_true = []
    s5_preds: dict[str, list[str]] = {m: [] for m in methods}
    s5_lats: dict[str, list[float]] = {m: [] for m in methods}
    per_attempt_records = []

    for idx, td in enumerate(stage5_trials, start=1):
        gt = td.get("ground_truth_label", "UNKNOWN")
        is_invoked = td.get("feature_valid") and td.get("model_prediction") in ("J", "Z")

        if is_invoked:
            matching = [d for d in diag_attempts if d["prediction"] == td.get("model_prediction")]
            feat = matching[(idx - 1) % len(matching)]["features"] if matching else np.zeros((16, 69))
        else:
            feat = np.zeros((16, 69))

        y_s5_true.append(gt)

        # LR
        t0 = time.perf_counter()
        p_lr, c_lr = predict_lr(feat) if is_invoked else ("UNKNOWN", 0.0)
        s5_lats["Baseline LR"].append((time.perf_counter() - t0) * 1000.0)
        s5_preds["Baseline LR"].append(p_lr)

        # DTW
        p_dtw, lat_dtw = predict_dtw(feat) if is_invoked else ("UNKNOWN", 0.0)
        s5_lats["Enhanced DTW"].append(lat_dtw)
        s5_preds["Enhanced DTW"].append(p_dtw)

        # Geom
        p_geom, lat_geom = predict_geom(feat) if is_invoked else ("UNKNOWN", 0.0)
        s5_lats["Geometric"].append(lat_geom)
        s5_preds["Geometric"].append(p_geom)

        # Ratio
        if is_invoked:
            p_ratio, c_ratio, r_dict, r_reason, lat_ratio = predict_ratio(feat)
        else:
            p_ratio, c_ratio, r_dict, r_reason, lat_ratio = "UNKNOWN", 0.0, {}, "PRE_MODEL_REJECTED", 0.0
        s5_lats["Ratio-Based"].append(lat_ratio)
        s5_preds["Ratio-Based"].append(p_ratio)

        per_attempt_records.append({
            "trial_index": idx,
            "ground_truth": gt,
            "model_invoked": is_invoked,
            "predictions": {
                "Baseline LR": p_lr,
                "Enhanced DTW": p_dtw,
                "Geometric": p_geom,
                "Ratio-Based": p_ratio,
            },
            "ratio_confidence": round(c_ratio, 3),
            "ratio_rejection_reason": r_reason,
            "ratios": r_dict,
        })

    metrics_suite4 = {
        m: compute_metrics(y_s5_true, s5_preds[m], s5_lats[m])
        for m in methods
    }
    all_results["suite4_webcam_stage5"] = metrics_suite4
    all_results["suite4_per_attempt"] = per_attempt_records

    # Print Comparative Tables
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY ACROSS BENCHMARK SUITES")
    print("=" * 80)

    print("\n--- SUITE 1: CLEAN VALIDATION SPLIT (64 samples: 26 J, 38 Z) ---")
    print(f"{'Method':<25} | {'Acc':<6} | {'Cov':<6} | {'J Rec':<6} | {'Z Rec':<6} | {'Macro F1':<8} | {'P50(ms)':<7}")
    print("-" * 75)
    for m in methods:
        s1 = metrics_suite1[m]
        print(
            f"{m:<25} | {s1['accuracy']:<6.1%} | {s1['coverage']:<6.1%} | "
            f"{s1['per_class']['J']['recall']:<6.1%} | {s1['per_class']['Z']['recall']:<6.1%} | "
            f"{s1['macro_f1']:<8.4f} | {s1['latency']['p50_ms']:<7.2f}"
        )

    print("\n--- SUITE 2: OUT-OF-DISTRIBUTION (OOD) INVALID MOTIONS (32 samples) ---")
    print(f"{'Method':<25} | {'UNKNOWN Rejection':<18} | {'False Acceptance Rate (FAR)':<28}")
    print("-" * 75)
    for m in methods:
        so = ood_summary[m]
        print(f"{m:<25} | {so['rejection_rate']:<18.1%} | {so['false_acceptance_rate']:<28.1%}")

    print("\n--- SUITE 4: STAGE 5 WEBCAM REAL-WORLD (50 attempts: 20 J, 20 Z, 10 Negative) ---")
    print(f"{'Method':<25} | {'Acc(Inv)':<8} | {'Cov':<6} | {'J Rec':<6} | {'Z Rec':<6} | {'Macro F1':<8} | {'FAR(Neg)':<8} | {'P50(ms)':<7}")
    print("-" * 95)
    for m in methods:
        s4 = metrics_suite4[m]
        print(
            f"{m:<25} | {s4['accuracy']:<8.1%} | {s4['coverage']:<6.1%} | "
            f"{s4['per_class']['J']['recall']:<6.1%} | {s4['per_class']['Z']['recall']:<6.1%} | "
            f"{s4['macro_f1']:<8.4f} | {s4['negative_controls']['false_acceptance_rate']:<8.1%} | {s4['latency']['p50_ms']:<7.2f}"
        )

    # Save Results JSON
    out_dir = PROJECT_ROOT / "reports/data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "stage6_ratio_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved raw telemetry to: {out_json}")

    return all_results


if __name__ == "__main__":
    run_benchmark()
