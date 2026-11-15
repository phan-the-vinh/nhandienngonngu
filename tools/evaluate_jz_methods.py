"""Comprehensive Evaluation & Comparison of Four J/Z Recognition Methods (Stage 3).

Compares:
- Method 1: Baseline Logistic Regression (Current Production Pipeline)
- Method 2: Enhanced Multivariate DTW Engine (14D CONFIG_E, Shadow)
- Method 3: Geometric & Ratio-Based Multi-Criteria Classifier
- Method 4: Hybrid Decision Classifier (Consensus Fusion & OOD Gating)

Evaluates across 3 Benchmark Suites:
1. Validation Split (64 clean samples: 26 J, 38 Z from SigNN baseline-b).
2. Out-Of-Distribution (OOD) Invalid Motion Suite (32 non-J/Z motion patterns).
3. Real-World Camera Diagnostic Attempts (61 valid 16x69 captured attempts).

Computes:
- Accuracy, Precision, Recall, F1 (separate for J and Z).
- Coverage and UNKNOWN rate.
- False Acceptance Rate (FAR) on OOD invalid motions.
- Latency P50, P95, P99.
- Five explicit edge case breakdowns (LR correct & DTW wrong, etc.).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

import joblib
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference.dynamic_jz.enhanced_dtw import EnhancedDTWClassifier
from app.inference.dynamic_jz.geometric_classifier import GeometricClassifier
from app.inference.dynamic_jz.hybrid_jz_classifier import HybridJzClassifier
from app.inference.dynamic_jz.finger_tracker import FEATURE_DIM, SEQUENCE_LENGTH


def compute_metrics(y_true: list[str], y_pred: list[str], latencies: list[float]) -> dict[str, Any]:
    """Compute precision, recall, F1, confusion matrix, coverage, and latency percentiles."""
    classes = ["J", "Z"]
    cm: dict[str, dict[str, int]] = {c: {"J": 0, "Z": 0, "UNKNOWN": 0} for c in classes}

    for yt, yp in zip(y_true, y_pred):
        if yt in cm:
            if yp in cm[yt]:
                cm[yt][yp] += 1
            else:
                cm[yt]["UNKNOWN"] += 1

    total_samples = len(y_true)
    total_unknown = sum(1 for yp in y_pred if yp == "UNKNOWN")
    unknown_rate = total_unknown / total_samples if total_samples > 0 else 0.0
    coverage = 1.0 - unknown_rate

    metrics: dict[str, Any] = {
        "confusion_matrix": cm,
        "unknown_rate": float(unknown_rate),
        "coverage": float(coverage),
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

    # Macro F1
    metrics["macro_f1"] = float(
        np.mean([metrics["per_class"][c]["f1"] for c in classes])
    )
    # Accuracy over all samples
    correct = sum(cm[c][c] for c in classes)
    metrics["accuracy"] = float(correct / total_samples) if total_samples > 0 else 0.0

    # Latency percentiles
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
    """Generate 32 realistic synthetic out-of-distribution (OOD) invalid motion sequences."""
    ood_samples = []

    # 1. Straight vertical downward swipes (8 variations)
    for i in range(8):
        seq = np.zeros((16, 69), dtype=np.float64)
        drop = 0.3 + 0.05 * i
        for t in range(16):
            frac = t / 15.0
            seq[t, 65:67] = [0.0, drop * frac]
            seq[t, 67:69] = [0.0, drop * frac]
            seq[t, :63] = 0.15 + 0.01 * np.sin(t)
        ood_samples.append({
            "name": f"straight_down_{i}",
            "type": "STRAIGHT_DOWN",
            "sequence": seq,
        })

    # 2. Straight horizontal swipes (8 variations: 4 left-to-right, 4 right-to-left)
    for i in range(4):
        seq = np.zeros((16, 69), dtype=np.float64)
        span = 0.3 + 0.05 * i
        for t in range(16):
            frac = t / 15.0
            seq[t, 65:67] = [span * frac, 0.0]
            seq[t, 67:69] = [span * frac, 0.0]
            seq[t, :63] = 0.2
        ood_samples.append({
            "name": f"horizontal_lr_{i}",
            "type": "HORIZONTAL_LR",
            "sequence": seq,
        })
    for i in range(4):
        seq = np.zeros((16, 69), dtype=np.float64)
        span = -(0.3 + 0.05 * i)
        for t in range(16):
            frac = t / 15.0
            seq[t, 65:67] = [span * frac, 0.0]
            seq[t, 67:69] = [span * frac, 0.0]
            seq[t, :63] = 0.2
        ood_samples.append({
            "name": f"horizontal_rl_{i}",
            "type": "HORIZONTAL_RL",
            "sequence": seq,
        })

    # 3. Static holding with camera noise/jitter (8 variations)
    for i in range(8):
        seq = np.zeros((16, 69), dtype=np.float64)
        noise_level = 0.005 * (i + 1)
        for t in range(16):
            seq[t, 65:67] = np.random.normal(0, noise_level, 2)
            seq[t, 67:69] = np.random.normal(0, noise_level, 2)
            seq[t, :63] = 0.2 + np.random.normal(0, noise_level, 63)
        ood_samples.append({
            "name": f"static_jitter_{i}",
            "type": "STATIC_JITTER",
            "sequence": seq,
        })

    # 4. Circular / wavy hand motion without stroke reversals (8 variations)
    for i in range(8):
        seq = np.zeros((16, 69), dtype=np.float64)
        radius = 0.2 + 0.03 * i
        for t in range(16):
            angle = 2 * np.pi * t / 15.0
            seq[t, 65:67] = [radius * np.cos(angle), radius * np.sin(angle)]
            seq[t, 67:69] = [radius * np.cos(angle), radius * np.sin(angle)]
            seq[t, :63] = 0.2
        ood_samples.append({
            "name": f"circular_wave_{i}",
            "type": "CIRCULAR_WAVE",
            "sequence": seq,
        })

    return ood_samples[:count]


def main() -> None:
    print("=" * 80)
    print("STAGE 3: COMPREHENSIVE BENCHMARK OF 4 J/Z RECOGNITION METHODS (INC. HYBRID)")
    print("=" * 80)

    # 1. Initialize Methods
    print("\n[1/4] Initializing Methods...")

    # Method 1: Baseline Logistic Regression Pipeline
    lr_model_path = PROJECT_ROOT / "data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib"
    lr_pipeline = joblib.load(lr_model_path)
    print("  Method 1: Baseline LogisticRegression loaded.")

    # Method 2: Enhanced DTW Engine (CONFIG_E)
    enhanced_dtw = EnhancedDTWClassifier(threshold=2.65, min_margin=0.15, min_dominance=-0.05)
    print("  Method 2: Enhanced DTW Classifier loaded (CONFIG_E, 14D).")

    # Method 3: Geometric & Ratio-Based Classifier
    geom_classifier = GeometricClassifier(min_score=0.45, min_margin=0.08)
    print("  Method 3: Geometric & Ratio Classifier initialized.")

    # Method 4: Hybrid Decision Classifier
    hybrid_classifier = HybridJzClassifier(
        lr_model_path=lr_model_path,
        dtw_threshold=2.65,
        dtw_min_margin=0.15,
        dtw_min_dominance=-0.05,
        geom_min_score=0.45,
        geom_min_margin=0.08,
    )
    print("  Method 4: Hybrid J/Z Decision Classifier initialized.")

    # Unified wrappers
    def predict_lr(seq: np.ndarray) -> tuple[str, float]:
        flat = seq.reshape(1, -1)
        probs = lr_pipeline.predict_proba(flat)[0]
        pred = "J" if probs[0] > probs[1] else "Z"
        conf = float(max(probs[0], probs[1]))
        return pred, conf

    def predict_dtw(seq: np.ndarray) -> tuple[str, float]:
        res = enhanced_dtw.classify(seq)
        return res.prediction, res.latency_ms

    def predict_geom(seq: np.ndarray) -> tuple[str, float]:
        res = geom_classifier.classify(seq)
        return res.prediction, res.latency_ms

    def predict_hybrid(seq: np.ndarray) -> tuple[str, float, str, float]:
        res = hybrid_classifier.classify(seq)
        return res.prediction, res.confidence, res.decision_path, res.latency_ms

    # 2. Benchmark Suite 1: Clean Validation Split (64 samples)
    print("\n[2/4] Evaluating Suite 1: Clean Validation Split (64 samples: 26 J, 38 Z)...")
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

    y_true_val = []
    y_pred_lr = []
    y_pred_dtw = []
    y_pred_geom = []
    y_pred_hybrid = []

    latencies_lr = []
    latencies_dtw = []
    latencies_geom = []
    latencies_hybrid = []

    sample_breakdown = []

    for m in val_meta:
        sp = m["source_path"]
        raw = samples_by_path[sp]
        seq = np.array(raw["features"], dtype=np.float64)
        lbl = raw["label"]
        y_true_val.append(lbl)

        # LR
        t0 = time.perf_counter()
        p_l, c_l = predict_lr(seq)
        lat_l = (time.perf_counter() - t0) * 1000.0
        latencies_lr.append(lat_l)
        y_pred_lr.append(p_l)

        # DTW
        p_d, lat_d = predict_dtw(seq)
        latencies_dtw.append(lat_d)
        y_pred_dtw.append(p_d)

        # Geom
        p_g, lat_g = predict_geom(seq)
        latencies_geom.append(lat_g)
        y_pred_geom.append(p_g)

        # Hybrid
        p_h, c_h, path_h, lat_h = predict_hybrid(seq)
        latencies_hybrid.append(lat_h)
        y_pred_hybrid.append(p_h)

        sample_breakdown.append({
            "source_path": sp,
            "true_label": lbl,
            "pred_lr": p_l,
            "pred_dtw": p_d,
            "pred_geom": p_g,
            "pred_hybrid": p_h,
            "hybrid_conf": c_h,
            "hybrid_path": path_h,
        })

    metrics_lr_val = compute_metrics(y_true_val, y_pred_lr, latencies_lr)
    metrics_dtw_val = compute_metrics(y_true_val, y_pred_dtw, latencies_dtw)
    metrics_geom_val = compute_metrics(y_true_val, y_pred_geom, latencies_geom)
    metrics_hybrid_val = compute_metrics(y_true_val, y_pred_hybrid, latencies_hybrid)

    # 3. Analyze the 5 Specific Cases on Validation Split
    print("\n  Analyzing 5 explicit edge cases on Validation Suite:")
    # Case A: LR correct, DTW wrong (or UNKNOWN)
    case_lr_true_dtw_false = [
        s for s in sample_breakdown
        if s["pred_lr"] == s["true_label"] and s["pred_dtw"] != s["true_label"]
    ]
    # Case B: LR wrong, DTW correct
    case_lr_false_dtw_true = [
        s for s in sample_breakdown
        if s["pred_lr"] != s["true_label"] and s["pred_dtw"] == s["true_label"]
    ]
    # Case C: Geometric wrong (or UNKNOWN), but DTW correct
    case_geom_false_dtw_true = [
        s for s in sample_breakdown
        if s["pred_geom"] != s["true_label"] and s["pred_dtw"] == s["true_label"]
    ]
    # Case D: All three wrong
    case_all_three_false = [
        s for s in sample_breakdown
        if s["pred_lr"] != s["true_label"] and s["pred_dtw"] != s["true_label"] and s["pred_geom"] != s["true_label"]
    ]
    # Case E: All three correct
    case_all_three_true = [
        s for s in sample_breakdown
        if s["pred_lr"] == s["true_label"] and s["pred_dtw"] == s["true_label"] and s["pred_geom"] == s["true_label"]
    ]

    print(f"    Case A (LR correct, DTW wrong/unknown): {len(case_lr_true_dtw_false)} samples")
    for s in case_lr_true_dtw_false:
        print(f"      - {Path(s['source_path']).name}: true={s['true_label']}, LR={s['pred_lr']}, DTW={s['pred_dtw']}, Geom={s['pred_geom']} -> Hybrid={s['pred_hybrid']} ({s['hybrid_path']})")

    print(f"    Case B (LR wrong, DTW correct): {len(case_lr_false_dtw_true)} samples")
    print(f"    Case C (Geom wrong/unknown, DTW correct): {len(case_geom_false_dtw_true)} samples")
    for s in case_geom_false_dtw_true:
        print(f"      - {Path(s['source_path']).name}: true={s['true_label']}, LR={s['pred_lr']}, DTW={s['pred_dtw']}, Geom={s['pred_geom']} -> Hybrid={s['pred_hybrid']} ({s['hybrid_path']})")

    print(f"    Case D (All three wrong): {len(case_all_three_false)} samples")
    print(f"    Case E (All three correct): {len(case_all_three_true)} samples")

    # 4. Benchmark Suite 2: OOD & Invalid Motion Suite (32 samples)
    print("\n[3/4] Evaluating Suite 2: Out-Of-Distribution Invalid Motions (32 samples)...")
    ood_samples = generate_ood_suite(32)

    ood_preds_lr = []
    ood_preds_dtw = []
    ood_preds_geom = []
    ood_preds_hybrid = []

    lat_ood_lr, lat_ood_dtw, lat_ood_geom, lat_ood_hybrid = [], [], [], []

    for item in ood_samples:
        seq = item["sequence"]

        t0 = time.perf_counter()
        p_l, _ = predict_lr(seq)
        lat_ood_lr.append((time.perf_counter() - t0) * 1000.0)
        ood_preds_lr.append(p_l)

        p_d, lat_d = predict_dtw(seq)
        lat_ood_dtw.append(lat_d)
        ood_preds_dtw.append(p_d)

        p_g, lat_g = predict_geom(seq)
        lat_ood_geom.append(lat_g)
        ood_preds_geom.append(p_g)

        p_h, _, _, lat_h = predict_hybrid(seq)
        lat_ood_hybrid.append(lat_h)
        ood_preds_hybrid.append(p_h)

    # Rejection rate: percentage of samples correctly identified as UNKNOWN
    n_ood = len(ood_samples)
    rej_lr = sum(1 for p in ood_preds_lr if p == "UNKNOWN") / n_ood
    rej_dtw = sum(1 for p in ood_preds_dtw if p == "UNKNOWN") / n_ood
    rej_geom = sum(1 for p in ood_preds_geom if p == "UNKNOWN") / n_ood
    rej_hybrid = sum(1 for p in ood_preds_hybrid if p == "UNKNOWN") / n_ood

    # False Acceptance Rate (FAR): % falsely accepted as J or Z
    far_lr = 1.0 - rej_lr
    far_dtw = 1.0 - rej_dtw
    far_geom = 1.0 - rej_geom
    far_hybrid = 1.0 - rej_hybrid

    # 5. Benchmark Suite 3: Real-World Camera Diagnostic Attempts (61 attempts)
    print("\n[4/4] Evaluating Suite 3: Real-World Camera Diagnostic Attempts (61 valid 16x69 attempts)...")
    diag_npzs = sorted(list((PROJECT_ROOT / "data/experiments/dynamic_jz_v1/diagnostics").glob("*/*/*.npz")))

    diag_preds_lr = []
    diag_preds_dtw = []
    diag_preds_geom = []
    diag_preds_hybrid = []

    cam_lat_lr, cam_lat_dtw, cam_lat_geom, cam_lat_hybrid = [], [], [], []
    webcam_comparison = []

    for p in diag_npzs:
        data = np.load(p)
        seq = data.get("features")
        if seq is None or seq.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            continue

        t0 = time.perf_counter()
        p_l, c_l = predict_lr(seq)
        cam_lat_lr.append((time.perf_counter() - t0) * 1000.0)
        diag_preds_lr.append(p_l)

        p_d, lat_d = predict_dtw(seq)
        cam_lat_dtw.append(lat_d)
        diag_preds_dtw.append(p_d)

        p_g, lat_g = predict_geom(seq)
        cam_lat_geom.append(lat_g)
        diag_preds_geom.append(p_g)

        p_h, c_h, path_h, lat_h = predict_hybrid(seq)
        cam_lat_hybrid.append(lat_h)
        diag_preds_hybrid.append(p_h)

        attempt_json = p.parent / "attempt.json"
        val_status = "UNKNOWN"
        target = "UNKNOWN"
        if attempt_json.exists():
            with open(attempt_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
                val_status = meta.get("technical_validation", {}).get("status", "UNKNOWN")
                target = meta.get("candidate_letter", meta.get("model_result", {}).get("prediction", "UNKNOWN"))

        webcam_comparison.append({
            "attempt_id": p.parent.name,
            "target": target,
            "validator_status": val_status,
            "lr_pred": p_l,
            "lr_conf": c_l,
            "dtw_pred": p_d,
            "geom_pred": p_g,
            "hybrid_pred": p_h,
            "hybrid_conf": c_h,
            "hybrid_path": path_h,
        })

    n_cam = len(diag_preds_lr)
    cam_rej_lr = sum(1 for p in diag_preds_lr if p == "UNKNOWN") / n_cam
    cam_rej_dtw = sum(1 for p in diag_preds_dtw if p == "UNKNOWN") / n_cam
    cam_rej_geom = sum(1 for p in diag_preds_geom if p == "UNKNOWN") / n_cam
    cam_rej_hybrid = sum(1 for p in diag_preds_hybrid if p == "UNKNOWN") / n_cam

    # 6. Save JSON Results
    results_summary = {
        "suite_1_validation": {
            "sample_count": len(val_meta),
            "lr": metrics_lr_val,
            "enhanced_dtw": metrics_dtw_val,
            "geometric": metrics_geom_val,
            "hybrid": metrics_hybrid_val,
            "cases": {
                "case_a_lr_true_dtw_false": case_lr_true_dtw_false,
                "case_b_lr_false_dtw_true": case_lr_false_dtw_true,
                "case_c_geom_false_dtw_true": case_geom_false_dtw_true,
                "case_d_all_three_false": case_all_three_false,
                "case_e_all_three_true_count": len(case_all_three_true),
            },
        },
        "suite_2_ood_rejection": {
            "sample_count": n_ood,
            "lr": {
                "rejection_rate": rej_lr, "false_acceptance_rate": far_lr,
                "pred_distribution": {"J": sum(1 for p in ood_preds_lr if p == "J"), "Z": sum(1 for p in ood_preds_lr if p == "Z"), "UNKNOWN": sum(1 for p in ood_preds_lr if p == "UNKNOWN")},
            },
            "enhanced_dtw": {
                "rejection_rate": rej_dtw, "false_acceptance_rate": far_dtw,
                "pred_distribution": {"J": sum(1 for p in ood_preds_dtw if p == "J"), "Z": sum(1 for p in ood_preds_dtw if p == "Z"), "UNKNOWN": sum(1 for p in ood_preds_dtw if p == "UNKNOWN")},
            },
            "geometric": {
                "rejection_rate": rej_geom, "false_acceptance_rate": far_geom,
                "pred_distribution": {"J": sum(1 for p in ood_preds_geom if p == "J"), "Z": sum(1 for p in ood_preds_geom if p == "Z"), "UNKNOWN": sum(1 for p in ood_preds_geom if p == "UNKNOWN")},
            },
            "hybrid": {
                "rejection_rate": rej_hybrid, "false_acceptance_rate": far_hybrid,
                "pred_distribution": {"J": sum(1 for p in ood_preds_hybrid if p == "J"), "Z": sum(1 for p in ood_preds_hybrid if p == "Z"), "UNKNOWN": sum(1 for p in ood_preds_hybrid if p == "UNKNOWN")},
            },
        },
        "suite_3_camera_attempts": {
            "attempt_count": n_cam,
            "lr": {"j_count": sum(1 for p in diag_preds_lr if p == "J"), "z_count": sum(1 for p in diag_preds_lr if p == "Z"), "unknown_count": sum(1 for p in diag_preds_lr if p == "UNKNOWN"), "rejection_rate": cam_rej_lr},
            "enhanced_dtw": {"j_count": sum(1 for p in diag_preds_dtw if p == "J"), "z_count": sum(1 for p in diag_preds_dtw if p == "Z"), "unknown_count": sum(1 for p in diag_preds_dtw if p == "UNKNOWN"), "rejection_rate": cam_rej_dtw},
            "geometric": {"j_count": sum(1 for p in diag_preds_geom if p == "J"), "z_count": sum(1 for p in diag_preds_geom if p == "Z"), "unknown_count": sum(1 for p in diag_preds_geom if p == "UNKNOWN"), "rejection_rate": cam_rej_geom},
            "hybrid": {"j_count": sum(1 for p in diag_preds_hybrid if p == "J"), "z_count": sum(1 for p in diag_preds_hybrid if p == "Z"), "unknown_count": sum(1 for p in diag_preds_hybrid if p == "UNKNOWN"), "rejection_rate": cam_rej_hybrid},
            "trials": webcam_comparison,
        },
    }

    report_json_path = PROJECT_ROOT / "reports/data/stage3_hybrid_comparison_results.json"
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved benchmark results to {report_json_path}")

    # 7. Print Comprehensive Summary Table
    print("\n" + "=" * 90)
    print("COMPARATIVE BENCHMARK: BASELINE LR vs ENHANCED DTW vs GEOMETRIC vs HYBRID")
    print("=" * 90)
    print(f"{'METRIC':<32} | {'BASELINE LR':<12} | {'ENHANCED DTW':<12} | {'GEOMETRIC':<12} | {'HYBRID':<12}")
    print("-" * 90)
    v = results_summary["suite_1_validation"]
    print(f"{'Validation Accuracy (64)':<32} | {v['lr']['accuracy']*100:>11.2f}% | {v['enhanced_dtw']['accuracy']*100:>11.2f}% | {v['geometric']['accuracy']*100:>11.2f}% | {v['hybrid']['accuracy']*100:>11.2f}%")
    print(f"{'Validation Macro F1':<32} | {v['lr']['macro_f1']:>12.4f} | {v['enhanced_dtw']['macro_f1']:>12.4f} | {v['geometric']['macro_f1']:>12.4f} | {v['hybrid']['macro_f1']:>12.4f}")
    print(f"{'Coverage Rate':<32} | {v['lr']['coverage']*100:>11.2f}% | {v['enhanced_dtw']['coverage']*100:>11.2f}% | {v['geometric']['coverage']*100:>11.2f}% | {v['hybrid']['coverage']*100:>11.2f}%")
    print(f"{'Validation UNKNOWN Rate':<32} | {v['lr']['unknown_rate']*100:>11.2f}% | {v['enhanced_dtw']['unknown_rate']*100:>11.2f}% | {v['geometric']['unknown_rate']*100:>11.2f}% | {v['hybrid']['unknown_rate']*100:>11.2f}%")
    print(f"{'J Precision':<32} | {v['lr']['per_class']['J']['precision']:>12.4f} | {v['enhanced_dtw']['per_class']['J']['precision']:>12.4f} | {v['geometric']['per_class']['J']['precision']:>12.4f} | {v['hybrid']['per_class']['J']['precision']:>12.4f}")
    print(f"{'J Recall':<32} | {v['lr']['per_class']['J']['recall']:>12.4f} | {v['enhanced_dtw']['per_class']['J']['recall']:>12.4f} | {v['geometric']['per_class']['J']['recall']:>12.4f} | {v['hybrid']['per_class']['J']['recall']:>12.4f}")
    print(f"{'J F1':<32} | {v['lr']['per_class']['J']['f1']:>12.4f} | {v['enhanced_dtw']['per_class']['J']['f1']:>12.4f} | {v['geometric']['per_class']['J']['f1']:>12.4f} | {v['hybrid']['per_class']['J']['f1']:>12.4f}")
    print(f"{'Z Precision':<32} | {v['lr']['per_class']['Z']['precision']:>12.4f} | {v['enhanced_dtw']['per_class']['Z']['precision']:>12.4f} | {v['geometric']['per_class']['Z']['precision']:>12.4f} | {v['hybrid']['per_class']['Z']['precision']:>12.4f}")
    print(f"{'Z Recall':<32} | {v['lr']['per_class']['Z']['recall']:>12.4f} | {v['enhanced_dtw']['per_class']['Z']['recall']:>12.4f} | {v['geometric']['per_class']['Z']['recall']:>12.4f} | {v['hybrid']['per_class']['Z']['recall']:>12.4f}")
    print(f"{'Z F1':<32} | {v['lr']['per_class']['Z']['f1']:>12.4f} | {v['enhanced_dtw']['per_class']['Z']['f1']:>12.4f} | {v['geometric']['per_class']['Z']['f1']:>12.4f} | {v['hybrid']['per_class']['Z']['f1']:>12.4f}")
    print(f"{'Latency P50 (ms)':<32} | {v['lr']['latency']['p50_ms']:>12.3f} | {v['enhanced_dtw']['latency']['p50_ms']:>12.3f} | {v['geometric']['latency']['p50_ms']:>12.3f} | {v['hybrid']['latency']['p50_ms']:>12.3f}")
    print(f"{'Latency P95 (ms)':<32} | {v['lr']['latency']['p95_ms']:>12.3f} | {v['enhanced_dtw']['latency']['p95_ms']:>12.3f} | {v['geometric']['latency']['p95_ms']:>12.3f} | {v['hybrid']['latency']['p95_ms']:>12.3f}")
    print(f"{'Latency P99 (ms)':<32} | {v['lr']['latency']['p99_ms']:>12.3f} | {v['enhanced_dtw']['latency']['p99_ms']:>12.3f} | {v['geometric']['latency']['p99_ms']:>12.3f} | {v['hybrid']['latency']['p99_ms']:>12.3f}")

    print("-" * 90)
    o = results_summary["suite_2_ood_rejection"]
    print(f"{'OOD Rejection Rate (32 non-J/Z)':<32} | {o['lr']['rejection_rate']*100:>11.2f}% | {o['enhanced_dtw']['rejection_rate']*100:>11.2f}% | {o['geometric']['rejection_rate']*100:>11.2f}% | {o['hybrid']['rejection_rate']*100:>11.2f}%")
    print(f"{'OOD False Acceptance Rate':<32} | {o['lr']['false_acceptance_rate']*100:>11.2f}% | {o['enhanced_dtw']['false_acceptance_rate']*100:>11.2f}% | {o['geometric']['false_acceptance_rate']*100:>11.2f}% | {o['hybrid']['false_acceptance_rate']*100:>11.2f}%")
    print(f"{'OOD Falsely Pred as J':<32} | {o['lr']['pred_distribution']['J']:>12d} | {o['enhanced_dtw']['pred_distribution']['J']:>12d} | {o['geometric']['pred_distribution']['J']:>12d} | {o['hybrid']['pred_distribution']['J']:>12d}")
    print(f"{'OOD Falsely Pred as Z':<32} | {o['lr']['pred_distribution']['Z']:>12d} | {o['enhanced_dtw']['pred_distribution']['Z']:>12d} | {o['geometric']['pred_distribution']['Z']:>12d} | {o['hybrid']['pred_distribution']['Z']:>12d}")

    print("-" * 90)
    c = results_summary["suite_3_camera_attempts"]
    print(f"{'Camera Diagnostic Attempts':<32} | {n_cam:>12d} | {n_cam:>12d} | {n_cam:>12d} | {n_cam:>12d}")
    print(f"{'Camera Pred as J':<32} | {c['lr']['j_count']:>12d} | {c['enhanced_dtw']['j_count']:>12d} | {c['geometric']['j_count']:>12d} | {c['hybrid']['j_count']:>12d}")
    print(f"{'Camera Pred as Z':<32} | {c['lr']['z_count']:>12d} | {c['enhanced_dtw']['z_count']:>12d} | {c['geometric']['z_count']:>12d} | {c['hybrid']['z_count']:>12d}")
    print(f"{'Camera UNKNOWN Rejections':<32} | {c['lr']['unknown_count']:>12d} | {c['enhanced_dtw']['unknown_count']:>12d} | {c['geometric']['unknown_count']:>12d} | {c['hybrid']['unknown_count']:>12d}")
    print(f"{'Camera Rejection Rate':<32} | {c['lr']['rejection_rate']*100:>11.2f}% | {c['enhanced_dtw']['rejection_rate']*100:>11.2f}% | {c['geometric']['rejection_rate']*100:>11.2f}% | {c['hybrid']['rejection_rate']*100:>11.2f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
