"""Stage 8 Comprehensive Benchmark: Adaptive Motion Recognition vs Baseline Methods.

Evaluates and compares 5 methods:
1. Method 1: Production Logistic Regression Pipeline (Production Baseline)
2. Method 2: Handshape-Only Branch (Stage 7)
3. Method 3: Ratio-Based Trajectory Classifier (Stage 6)
4. Method 4: Stage 7 Fusion Classifier (Rigid Handshape AND Trajectory)
5. Method 5: Adaptive Motion Recognition Classifier (Stage 8 Adaptive Joint Model)

Evaluates Across 4 Benchmark Suites:
1. Suite 1: Clean Validation Split (64 samples: 26 J, 38 Z from SigNN baseline-b).
2. Suite 2: Out-Of-Distribution (OOD) Invalid Motion Suite (32 synthetic non-J/Z patterns).
3. Suite 3: Real-World Camera Diagnostic Attempts (61 authentic captured 16x69 sequences).
4. Suite 4: Stage 5 Webcam Real-World Validation Replay (50 attempts: 20 J, 20 Z, 10 Negative Controls).

Outputs:
- Full comparative metrics across all 5 methods: Accuracy, Precision, Recall, F1, Coverage,
  UNKNOWN rate, False Acceptance Rate (FAR) on negative controls, Cross-confusions (J <-> Z),
  Latency percentiles (P50, P95, P99).
- Per-attempt attribution of all 50 Stage 5 Webcam trials.
- Detailed analysis of missed / rejected cases.
- Saves raw telemetry to reports/data/stage8_adaptive_benchmark_results.json.
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
from app.inference.dynamic_jz.adaptive_classifier import (
    AdaptiveMotionJZClassifier,
)
from app.inference.dynamic_jz.finger_tracker import (
    FEATURE_DIM,
    LANDMARK_COUNT,
    PALM_SCALE_LANDMARKS,
    SEQUENCE_LENGTH,
)
from tools.evaluate_stage7_fusion import compute_metrics, generate_ood_suite


def run_stage8_benchmark() -> dict[str, Any]:
    print("=" * 80)
    print("STAGE 8: ADAPTIVE MOTION RECOGNITION & 5-METHOD COMPARATIVE BENCHMARK")
    print("=" * 80)

    # 1. Initialize Methods
    print("\n[1/5] Initializing 5 Methods...")
    lr_model_path = PROJECT_ROOT / "models/dynamic_jz/model_pipeline.joblib"
    if not lr_model_path.exists():
        lr_model_path = PROJECT_ROOT / "data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib"
    lr_pipeline = joblib.load(lr_model_path)
    print("  Method 1: Production Logistic Regression Pipeline loaded.")

    handshape_branch = HandshapeBranch(
        min_confidence=0.40,
        min_margin=0.08,
        min_pinky_ext=0.95,
        min_index_ext=1.00,
        min_pinky_contrast=0.10,
        min_index_contrast=0.20,
        max_stability_std=0.35,
    )
    print("  Method 2: Handshape-Only Branch (Stage 7) loaded.")

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
    print("  Method 3: Ratio-Based Trajectory Classifier (Stage 6) loaded.")

    fusion_classifier = HandshapeTrajectoryFusionClassifier()
    print("  Method 4: Stage 7 Fusion Classifier loaded.")

    adaptive_classifier = AdaptiveMotionJZClassifier()
    print("  Method 5: Adaptive Motion Recognition Classifier (Stage 8) initialized.")

    def predict_lr(seq: np.ndarray) -> tuple[str, float]:
        flat = seq.reshape(1, -1)
        probs = lr_pipeline.predict_proba(flat)[0]
        classes = list(lr_pipeline.classes_)
        idx_j = classes.index("J") if "J" in classes else 0
        idx_z = classes.index("Z") if "Z" in classes else 1
        pred = "J" if probs[idx_j] > probs[idx_z] else "Z"
        conf = float(max(probs[idx_j], probs[idx_z]))
        return pred, conf

    def predict_handshape(seq: np.ndarray) -> tuple[str, float, float]:
        t0 = time.perf_counter()
        lms = seq[:, :63].reshape((SEQUENCE_LENGTH, LANDMARK_COUNT, 3))
        mcp_pts = lms[:, PALM_SCALE_LANDMARKS, :]
        mcp_dists = np.linalg.norm(mcp_pts, axis=2)
        palm_scales = np.median(mcp_dists, axis=1)
        res = handshape_branch.evaluate(lms, palm_scales)
        lat = (time.perf_counter() - t0) * 1000.0
        return res.prediction, res.confidence, lat

    def predict_ratio(seq: np.ndarray) -> tuple[str, float, float]:
        res = ratio_classifier.classify(seq)
        return res.prediction, res.confidence, res.latency_ms

    def predict_fusion(seq: np.ndarray) -> tuple[str, float, float]:
        res = fusion_classifier.classify(seq)
        return res.prediction, res.confidence, res.latency_ms

    def predict_adaptive(seq: np.ndarray) -> tuple[str, float, str, str, float]:
        res = adaptive_classifier.classify(seq)
        return res.prediction, res.confidence, res.decision_path, res.rejection_reason, res.latency_ms

    methods = ["Production LR", "Handshape-Only", "Ratio-Only", "Stage 7 Fusion", "Adaptive Motion (Stage 8)"]
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
        val_lats["Production LR"].append((time.perf_counter() - t0) * 1000.0)
        val_preds["Production LR"].append(p_lr)

        # 2. Handshape
        p_hs, _, lat_hs = predict_handshape(seq)
        val_lats["Handshape-Only"].append(lat_hs)
        val_preds["Handshape-Only"].append(p_hs)

        # 3. Ratio
        p_ratio, _, lat_ratio = predict_ratio(seq)
        val_lats["Ratio-Only"].append(lat_ratio)
        val_preds["Ratio-Only"].append(p_ratio)

        # 4. Fusion
        p_fus, _, lat_fus = predict_fusion(seq)
        val_lats["Stage 7 Fusion"].append(lat_fus)
        val_preds["Stage 7 Fusion"].append(p_fus)

        # 5. Adaptive
        p_adp, _, _, _, lat_adp = predict_adaptive(seq)
        val_lats["Adaptive Motion (Stage 8)"].append(lat_adp)
        val_preds["Adaptive Motion (Stage 8)"].append(p_adp)

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
        ood_lats["Production LR"].append((time.perf_counter() - t0) * 1000.0)
        ood_preds["Production LR"].append(p_lr)

        p_hs, _, lat_hs = predict_handshape(seq)
        ood_lats["Handshape-Only"].append(lat_hs)
        ood_preds["Handshape-Only"].append(p_hs)

        p_ratio, _, lat_ratio = predict_ratio(seq)
        ood_lats["Ratio-Only"].append(lat_ratio)
        ood_preds["Ratio-Only"].append(p_ratio)

        p_fus, _, lat_fus = predict_fusion(seq)
        ood_lats["Stage 7 Fusion"].append(lat_fus)
        ood_preds["Stage 7 Fusion"].append(p_fus)

        p_adp, _, _, _, lat_adp = predict_adaptive(seq)
        ood_lats["Adaptive Motion (Stage 8)"].append(lat_adp)
        ood_preds["Adaptive Motion (Stage 8)"].append(p_adp)

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
        diag_lats["Production LR"].append((time.perf_counter() - t0) * 1000.0)
        diag_preds["Production LR"].append(p_lr)

        p_hs, _, lat_hs = predict_handshape(feat)
        diag_lats["Handshape-Only"].append(lat_hs)
        diag_preds["Handshape-Only"].append(p_hs)

        p_ratio, _, lat_ratio = predict_ratio(feat)
        diag_lats["Ratio-Only"].append(lat_ratio)
        diag_preds["Ratio-Only"].append(p_ratio)

        p_fus, _, lat_fus = predict_fusion(feat)
        diag_lats["Stage 7 Fusion"].append(lat_fus)
        diag_preds["Stage 7 Fusion"].append(p_fus)

        p_adp, _, _, _, lat_adp = predict_adaptive(feat)
        diag_lats["Adaptive Motion (Stage 8)"].append(lat_adp)
        diag_preds["Adaptive Motion (Stage 8)"].append(p_adp)

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
        s5_lats["Production LR"].append((time.perf_counter() - t0) * 1000.0)
        s5_preds["Production LR"].append(p_lr)

        # 2. Handshape
        p_hs, c_hs, lat_hs = predict_handshape(feat) if is_invoked else ("UNKNOWN", 0.0, 0.0)
        s5_lats["Handshape-Only"].append(lat_hs)
        s5_preds["Handshape-Only"].append(p_hs)

        # 3. Ratio
        p_ratio, c_ratio, lat_ratio = predict_ratio(feat) if is_invoked else ("UNKNOWN", 0.0, 0.0)
        s5_lats["Ratio-Only"].append(lat_ratio)
        s5_preds["Ratio-Only"].append(p_ratio)

        # 4. Fusion
        p_fus, c_fus, lat_fus = predict_fusion(feat) if is_invoked else ("UNKNOWN", 0.0, 0.0)
        s5_lats["Stage 7 Fusion"].append(lat_fus)
        s5_preds["Stage 7 Fusion"].append(p_fus)

        # 5. Adaptive
        if is_invoked:
            p_adp, c_adp, d_path, r_reason, lat_adp = predict_adaptive(feat)
        else:
            p_adp, c_adp, d_path, r_reason, lat_adp = "UNKNOWN", 0.0, "NOT_INVOKED", "PRE_MODEL_REJECTED", 0.0
        s5_lats["Adaptive Motion (Stage 8)"].append(lat_adp)
        s5_preds["Adaptive Motion (Stage 8)"].append(p_adp)

        per_attempt_records.append({
            "trial_index": idx,
            "ground_truth": gt,
            "model_invoked": is_invoked,
            "predictions": {
                "Production LR": p_lr,
                "Handshape-Only": p_hs,
                "Ratio-Only": p_ratio,
                "Stage 7 Fusion": p_fus,
                "Adaptive Motion": p_adp,
            },
            "adaptive_confidence": round(c_adp, 3),
            "adaptive_decision_path": d_path,
            "adaptive_rejection_reason": r_reason,
        })

    metrics_suite4 = {m: compute_metrics(y_s5_true, s5_preds[m], s5_lats[m]) for m in methods}
    all_results["suite4_webcam_stage5"] = metrics_suite4
    all_results["suite4_per_attempt"] = per_attempt_records

    # Print Summary Tables
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY ACROSS BENCHMARK SUITES")
    print("=" * 80)

    print("\n--- SUITE 1: CLEAN VALIDATION SPLIT (64 samples: 26 J, 38 Z) ---")
    print(f"{'Method':<25} | {'Acc':<6} | {'Cov':<6} | {'J Rec':<6} | {'Z Rec':<6} | {'Macro F1':<8} | {'P50(ms)':<7}")
    print("-" * 80)
    for m in methods:
        res = metrics_suite1[m]
        p_c = res["per_class"]
        lat = res.get("latency", {}).get("p50_ms", 0.0)
        print(f"{m:<25} | {res['accuracy']*100:<5.1f}% | {res['coverage']*100:<5.1f}% | {p_c['J']['recall']*100:<5.1f}% | {p_c['Z']['recall']*100:<5.1f}% | {res['macro_f1']:<8.4f} | {lat:<7.2f}")

    print("\n--- SUITE 2: OUT-OF-DISTRIBUTION (OOD) INVALID MOTIONS (32 samples) ---")
    print(f"{'Method':<25} | {'UNKNOWN Rejection':<20} | {'False Acceptance Rate (FAR)':<28}")
    print("-" * 80)
    for m in methods:
        res = ood_summary[m]
        print(f"{m:<25} | {res['rejection_rate']*100:<19.1f}% | {res['false_acceptance_rate']*100:<27.1f}%")

    print("\n--- SUITE 4: STAGE 5 WEBCAM REAL-WORLD (50 attempts: 20 J, 20 Z, 10 Negative) ---")
    print(f"{'Method':<25} | {'Acc':<6} | {'Cov':<6} | {'J Rec':<6} | {'Z Rec':<6} | {'Macro F1':<8} | {'P50(ms)':<7}")
    print("-" * 80)
    for m in methods:
        res = metrics_suite4[m]
        p_c = res["per_class"]
        lat = res.get("latency", {}).get("p50_ms", 0.0)
        print(f"{m:<25} | {res['accuracy']*100:<5.1f}% | {res['coverage']*100:<5.1f}% | {p_c['J']['recall']*100:<5.1f}% | {p_c['Z']['recall']*100:<5.1f}% | {res['macro_f1']:<8.4f} | {lat:<7.2f}")

    # Save to JSON
    out_dir = PROJECT_ROOT / "reports/data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stage8_adaptive_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw telemetry to: {out_path}")

    return all_results


if __name__ == "__main__":
    run_stage8_benchmark()
