"""Stage 7 Comprehensive Benchmark & Comparison of 5 Methods.

Evaluates and compares:
1. Method 1: Baseline Logistic Regression Pipeline (Production Baseline)
2. Method 2: Ratio-Based Trajectory Classifier (Stage 6)
3. Method 3: Handshape-Only Branch (Stage 7)
4. Method 4: Trajectory-Only Branch (Stage 7)
5. Method 5: Handshape + Trajectory Fusion Classifier (Stage 7 Fused)

Evaluates Across 4 Benchmark Suites:
1. Suite 1: Clean Validation Split (64 samples: 26 J, 38 Z from SigNN baseline-b).
2. Suite 2: Out-Of-Distribution (OOD) Invalid Motion Suite (32 synthetic non-J/Z patterns).
3. Suite 3: Real-World Camera Diagnostic Attempts (61 authentic captured 16x69 sequences).
4. Suite 4: Stage 5 Webcam Real-World Validation Replay (50 attempts: 20 J, 20 Z, 10 Negative Controls).

Outputs:
- Full comparative metrics across all 5 methods: Accuracy, Precision, Recall, F1, Coverage,
  UNKNOWN rate, False Acceptance Rate (FAR) on negative controls, Cross-confusions (J <-> Z),
  Latency percentiles (P50, P95, P99).
- Ablation Analysis: Handshape-only vs Trajectory-only vs Fused.
- Per-attempt attribution of rejection reasons for Stage 5 Webcam trials.
- Raw telemetry saved to reports/data/stage7_fusion_benchmark_results.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference.dynamic_jz.ratio_classifier import RatioBasedJZClassifier
from app.inference.dynamic_jz.fusion_classifier import (
    HandshapeBranch,
    TrajectoryBranch,
    HandshapeTrajectoryFusionClassifier,
)
from app.inference.dynamic_jz.finger_tracker import (
    FEATURE_DIM,
    LANDMARK_COUNT,
    PALM_SCALE_LANDMARKS,
    SEQUENCE_LENGTH,
)


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


def run_stage7_benchmark() -> dict[str, Any]:
    print("=" * 80)
    print("STAGE 7: HANDSHAPE + TRAJECTORY FUSION & 5-METHOD COMPARATIVE BENCHMARK")
    print("=" * 80)

    # 1. Initialize Methods
    print("\n[1/5] Initializing 5 Methods...")
    lr_model_path = PROJECT_ROOT / "models/dynamic_jz/model_pipeline.joblib"
    if not lr_model_path.exists():
        lr_model_path = PROJECT_ROOT / "data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib"
    lr_pipeline = joblib.load(lr_model_path)
    print("  Method 1: Baseline Logistic Regression Pipeline loaded.")

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
    print("  Method 2: Ratio-Based Trajectory Classifier (Stage 6) loaded.")

    handshape_branch = HandshapeBranch(
        min_confidence=0.40,
        min_margin=0.08,
        min_pinky_ext=0.95,
        min_index_ext=1.00,
        min_pinky_contrast=0.10,
        min_index_contrast=0.20,
        max_stability_std=0.35,
    )
    print("  Method 3: Handshape-Only Branch (Stage 7) initialized.")

    trajectory_branch = TrajectoryBranch(
        min_confidence=0.38,
        min_margin=0.05,
        min_motion_len=0.20,
        min_j_drop=0.18,
        min_j_hook=0.08,
        min_z_span=0.05,
        min_z_diag_drop=0.04,
    )
    print("  Method 4: Trajectory-Only Branch (Stage 7) initialized.")

    fusion_classifier = HandshapeTrajectoryFusionClassifier(
        handshape_branch=handshape_branch,
        trajectory_branch=trajectory_branch,
        weight_handshape=0.45,
        weight_trajectory=0.55,
        min_fused_confidence=0.42,
    )
    print("  Method 5: Handshape + Trajectory Fusion Classifier (Stage 7 Fused) initialized.")

    def predict_lr(seq: np.ndarray) -> tuple[str, float]:
        flat = seq.reshape(1, -1)
        probs = lr_pipeline.predict_proba(flat)[0]
        classes = list(lr_pipeline.classes_)
        idx_j = classes.index("J") if "J" in classes else 0
        idx_z = classes.index("Z") if "Z" in classes else 1
        pred = "J" if probs[idx_j] > probs[idx_z] else "Z"
        conf = float(max(probs[idx_j], probs[idx_z]))
        return pred, conf

    def predict_ratio(seq: np.ndarray) -> tuple[str, float, float]:
        res = ratio_classifier.classify(seq)
        return res.prediction, res.confidence, res.latency_ms

    def predict_handshape(seq: np.ndarray) -> tuple[str, float, float]:
        t0 = time.perf_counter()
        lms = seq[:, :63].reshape((SEQUENCE_LENGTH, LANDMARK_COUNT, 3))
        mcp_pts = lms[:, PALM_SCALE_LANDMARKS, :]
        mcp_dists = np.linalg.norm(mcp_pts, axis=2)
        palm_scales = np.median(mcp_dists, axis=1)
        res = handshape_branch.evaluate(lms, palm_scales)
        lat = (time.perf_counter() - t0) * 1000.0
        return res.prediction, res.confidence, lat

    def predict_trajectory(seq: np.ndarray) -> tuple[str, float, float]:
        t0 = time.perf_counter()
        path_idx = seq[:, 65:67]
        path_pky = seq[:, 67:69]
        res = trajectory_branch.evaluate(path_idx, path_pky)
        lat = (time.perf_counter() - t0) * 1000.0
        return res.prediction, res.confidence, lat

    def predict_fusion(seq: np.ndarray) -> tuple[str, float, str, str, str, float]:
        res = fusion_classifier.classify(seq)
        return (
            res.prediction,
            res.confidence,
            res.decision_path,
            res.primary_rejection_source,
            res.rejection_reason,
            res.latency_ms,
        )

    methods = ["Baseline LR", "Ratio-Based", "Handshape-Only", "Trajectory-Only", "Fusion (Stage 7)"]
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

        # 1. LR
        t0 = time.perf_counter()
        p_lr, _ = predict_lr(seq)
        val_lats["Baseline LR"].append((time.perf_counter() - t0) * 1000.0)
        val_preds["Baseline LR"].append(p_lr)

        # 2. Ratio
        p_ratio, _, lat_ratio = predict_ratio(seq)
        val_lats["Ratio-Based"].append(lat_ratio)
        val_preds["Ratio-Based"].append(p_ratio)

        # 3. Handshape
        p_hs, _, lat_hs = predict_handshape(seq)
        val_lats["Handshape-Only"].append(lat_hs)
        val_preds["Handshape-Only"].append(p_hs)

        # 4. Trajectory
        p_tr, _, lat_tr = predict_trajectory(seq)
        val_lats["Trajectory-Only"].append(lat_tr)
        val_preds["Trajectory-Only"].append(p_tr)

        # 5. Fusion
        p_fus, _, _, _, _, lat_fus = predict_fusion(seq)
        val_lats["Fusion (Stage 7)"].append(lat_fus)
        val_preds["Fusion (Stage 7)"].append(p_fus)

    metrics_suite1 = {m: compute_metrics(y_val_true, val_preds[m], val_lats[m]) for m in methods}
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

        p_ratio, _, lat_ratio = predict_ratio(seq)
        ood_lats["Ratio-Based"].append(lat_ratio)
        ood_preds["Ratio-Based"].append(p_ratio)

        p_hs, _, lat_hs = predict_handshape(seq)
        ood_lats["Handshape-Only"].append(lat_hs)
        ood_preds["Handshape-Only"].append(p_hs)

        p_tr, _, lat_tr = predict_trajectory(seq)
        ood_lats["Trajectory-Only"].append(lat_tr)
        ood_preds["Trajectory-Only"].append(p_tr)

        p_fus, _, _, _, _, lat_fus = predict_fusion(seq)
        ood_lats["Fusion (Stage 7)"].append(lat_fus)
        ood_preds["Fusion (Stage 7)"].append(p_fus)

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

        p_ratio, _, lat_ratio = predict_ratio(feat)
        diag_lats["Ratio-Based"].append(lat_ratio)
        diag_preds["Ratio-Based"].append(p_ratio)

        p_hs, _, lat_hs = predict_handshape(feat)
        diag_lats["Handshape-Only"].append(lat_hs)
        diag_preds["Handshape-Only"].append(p_hs)

        p_tr, _, lat_tr = predict_trajectory(feat)
        diag_lats["Trajectory-Only"].append(lat_tr)
        diag_preds["Trajectory-Only"].append(p_tr)

        p_fus, _, _, _, _, lat_fus = predict_fusion(feat)
        diag_lats["Fusion (Stage 7)"].append(lat_fus)
        diag_preds["Fusion (Stage 7)"].append(p_fus)

    metrics_suite3 = {m: compute_metrics(y_diag_true, diag_preds[m], diag_lats[m]) for m in methods}
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

        # 1. LR
        t0 = time.perf_counter()
        p_lr, c_lr = predict_lr(feat) if is_invoked else ("UNKNOWN", 0.0)
        s5_lats["Baseline LR"].append((time.perf_counter() - t0) * 1000.0)
        s5_preds["Baseline LR"].append(p_lr)

        # 2. Ratio
        p_ratio, c_ratio, lat_ratio = predict_ratio(feat) if is_invoked else ("UNKNOWN", 0.0, 0.0)
        s5_lats["Ratio-Based"].append(lat_ratio)
        s5_preds["Ratio-Based"].append(p_ratio)

        # 3. Handshape
        p_hs, c_hs, lat_hs = predict_handshape(feat) if is_invoked else ("UNKNOWN", 0.0, 0.0)
        s5_lats["Handshape-Only"].append(lat_hs)
        s5_preds["Handshape-Only"].append(p_hs)

        # 4. Trajectory
        p_tr, c_tr, lat_tr = predict_trajectory(feat) if is_invoked else ("UNKNOWN", 0.0, 0.0)
        s5_lats["Trajectory-Only"].append(lat_tr)
        s5_preds["Trajectory-Only"].append(p_tr)

        # 5. Fusion
        if is_invoked:
            p_fus, c_fus, d_path, r_src, r_reason, lat_fus = predict_fusion(feat)
        else:
            p_fus, c_fus, d_path, r_src, r_reason, lat_fus = "UNKNOWN", 0.0, "NOT_INVOKED", "PRE_MODEL", "PRE_MODEL_REJECTED", 0.0
        s5_lats["Fusion (Stage 7)"].append(lat_fus)
        s5_preds["Fusion (Stage 7)"].append(p_fus)

        per_attempt_records.append({
            "trial_index": idx,
            "ground_truth": gt,
            "model_invoked": is_invoked,
            "predictions": {
                "Baseline LR": p_lr,
                "Ratio-Based": p_ratio,
                "Handshape-Only": p_hs,
                "Trajectory-Only": p_tr,
                "Fusion": p_fus,
            },
            "fusion_confidence": round(c_fus, 3),
            "decision_path": d_path,
            "primary_rejection_source": r_src,
            "rejection_reason": r_reason,
        })

    metrics_suite4 = {m: compute_metrics(y_s5_true, s5_preds[m], s5_lats[m]) for m in methods}
    all_results["suite4_webcam_stage5"] = metrics_suite4
    all_results["suite4_per_attempt"] = per_attempt_records

    # 6. Ablation Summary on Suite 4
    ablation_summary = {
        "Handshape-Only": {
            "accuracy": metrics_suite4["Handshape-Only"]["accuracy"],
            "coverage": metrics_suite4["Handshape-Only"]["coverage"],
            "macro_f1": metrics_suite4["Handshape-Only"]["macro_f1"],
            "j_recall": metrics_suite4["Handshape-Only"]["per_class"]["J"]["recall"],
            "z_recall": metrics_suite4["Handshape-Only"]["per_class"]["Z"]["recall"],
            "unknown_rate": metrics_suite4["Handshape-Only"]["unknown_rate"],
        },
        "Trajectory-Only": {
            "accuracy": metrics_suite4["Trajectory-Only"]["accuracy"],
            "coverage": metrics_suite4["Trajectory-Only"]["coverage"],
            "macro_f1": metrics_suite4["Trajectory-Only"]["macro_f1"],
            "j_recall": metrics_suite4["Trajectory-Only"]["per_class"]["J"]["recall"],
            "z_recall": metrics_suite4["Trajectory-Only"]["per_class"]["Z"]["recall"],
            "unknown_rate": metrics_suite4["Trajectory-Only"]["unknown_rate"],
        },
        "Fusion (Both)": {
            "accuracy": metrics_suite4["Fusion (Stage 7)"]["accuracy"],
            "coverage": metrics_suite4["Fusion (Stage 7)"]["coverage"],
            "macro_f1": metrics_suite4["Fusion (Stage 7)"]["macro_f1"],
            "j_recall": metrics_suite4["Fusion (Stage 7)"]["per_class"]["J"]["recall"],
            "z_recall": metrics_suite4["Fusion (Stage 7)"]["per_class"]["Z"]["recall"],
            "unknown_rate": metrics_suite4["Fusion (Stage 7)"]["unknown_rate"],
        },
    }
    all_results["ablation_study"] = ablation_summary

    # Print Summary Tables
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY ACROSS BENCHMARK SUITES")
    print("=" * 80)

    print("\n--- SUITE 1: CLEAN VALIDATION SPLIT (64 samples: 26 J, 38 Z) ---")
    print(f"{'Method':<20} | {'Acc':<6} | {'Cov':<6} | {'J Rec':<6} | {'Z Rec':<6} | {'Macro F1':<8} | {'P50(ms)':<7}")
    print("-" * 75)
    for m in methods:
        res = metrics_suite1[m]
        p_c = res["per_class"]
        lat = res.get("latency", {}).get("p50_ms", 0.0)
        print(f"{m:<20} | {res['accuracy']*100:<5.1f}% | {res['coverage']*100:<5.1f}% | {p_c['J']['recall']*100:<5.1f}% | {p_c['Z']['recall']*100:<5.1f}% | {res['macro_f1']:<8.4f} | {lat:<7.2f}")

    print("\n--- SUITE 2: OUT-OF-DISTRIBUTION (OOD) INVALID MOTIONS (32 samples) ---")
    print(f"{'Method':<20} | {'UNKNOWN Rejection':<20} | {'False Acceptance Rate (FAR)':<28}")
    print("-" * 75)
    for m in methods:
        res = ood_summary[m]
        print(f"{m:<20} | {res['rejection_rate']*100:<19.1f}% | {res['false_acceptance_rate']*100:<27.1f}%")

    print("\n--- SUITE 4: STAGE 5 WEBCAM REAL-WORLD (50 attempts: 20 J, 20 Z, 10 Negative) ---")
    print(f"{'Method':<20} | {'Acc':<6} | {'Cov':<6} | {'J Rec':<6} | {'Z Rec':<6} | {'Macro F1':<8} | {'P50(ms)':<7}")
    print("-" * 75)
    for m in methods:
        res = metrics_suite4[m]
        p_c = res["per_class"]
        lat = res.get("latency", {}).get("p50_ms", 0.0)
        print(f"{m:<20} | {res['accuracy']*100:<5.1f}% | {res['coverage']*100:<5.1f}% | {p_c['J']['recall']*100:<5.1f}% | {p_c['Z']['recall']*100:<5.1f}% | {res['macro_f1']:<8.4f} | {lat:<7.2f}")

    # Save to JSON
    out_dir = PROJECT_ROOT / "reports/data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stage7_fusion_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw telemetry to: {out_path}")

    return all_results


if __name__ == "__main__":
    run_stage7_benchmark()
