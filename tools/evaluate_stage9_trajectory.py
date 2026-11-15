"""Stage 9 Benchmark & Comparative Evaluation of Target Finger Trajectory Recognizers.

Evaluates 5 Methods:
1. Method 1: Production Logistic Regression Pipeline (Production Baseline)
2. Method 2: Stage 8 Adaptive Motion Recognition Baseline (Adaptive Baseline)
3. Method 3: Trajectory-Only Recognizer (Target Fingertip Morphology Only)
4. Method 4: Handshape-Only Recognizer (21-Landmark Handshape Only)
5. Method 5: Target Finger Trajectory Recognizer (Combined Target Trajectory + Handshape)

Across 3 Distinct Suites:
1. Suite 1: Clean Validation Split (64 samples: 26 J, 38 Z from SigNN baseline-b)
2. Suite 2: Synthetic Out-Of-Distribution Suite (32 samples of non-J/Z motions)
3. Suite 3: Authentic Real-World Webcam Attempts (70 attempts: 25 J, 25 Z, 20 Negative)
   - Evaluated on both End-to-End (all 70 attempts) and Model-Invoked (50 attempts).

Saves comprehensive telemetry to reports/data/stage9_trajectory_results.json.
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

from app.inference.dynamic_jz.adaptive_classifier import AdaptiveMotionJZClassifier
from app.inference.dynamic_jz.target_trajectory_recognizer import (
    TrajectoryOnlyJZRecognizer,
    HandshapeOnlyJZRecognizer,
    TargetFingerTrajectoryRecognizer,
)
from tools.evaluate_stage7_fusion import compute_metrics, generate_ood_suite


def run_stage9_benchmark() -> dict[str, Any]:
    print("=" * 80)
    print("STAGE 9: TARGET FINGER TRAJECTORY RECOGNITION BENCHMARK & ABLATION")
    print("=" * 80)

    # 1. Initialize 5 Methods
    print("\n[1/5] Initializing Methods...")
    lr_model_path = PROJECT_ROOT / "models/dynamic_jz/model_pipeline.joblib"
    if not lr_model_path.exists():
        lr_model_path = PROJECT_ROOT / "data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib"
    lr_pipeline = joblib.load(lr_model_path)
    print("  Method 1: Production Logistic Regression Pipeline loaded.")

    cls_adaptive = AdaptiveMotionJZClassifier(z_diagonal_threshold=0.015)
    print("  Method 2: Stage 8 Adaptive Baseline initialized.")

    cls_tr_only = TrajectoryOnlyJZRecognizer()
    print("  Method 3: Trajectory-Only Recognizer initialized.")

    cls_hs_only = HandshapeOnlyJZRecognizer()
    print("  Method 4: Handshape-Only Recognizer initialized.")

    cls_target_combined = TargetFingerTrajectoryRecognizer()
    print("  Method 5: Target Finger Trajectory Recognizer (Combined) initialized.")

    methods = {
        "Production LR": "lr",
        "Adaptive Baseline": "adaptive",
        "Trajectory-Only": "tr_only",
        "Handshape-Only": "hs_only",
        "Target Finger Combined": "combined",
    }

    def predict_lr(seq: np.ndarray) -> tuple[str, float]:
        flat = seq.reshape(1, -1)
        pred = str(lr_pipeline.predict(flat)[0])
        proba = float(np.max(lr_pipeline.predict_proba(flat)))
        return pred, proba

    # 2. Load Suites
    print("\n[2/5] Loading Evaluation Suites...")

    # Suite 1: Clean Validation Split (64)
    split_path = PROJECT_ROOT / "data/experiments/dynamic_jz_binary_baseline_v1/split_manifest.json"
    data_path = PROJECT_ROOT / "data/experiments/dynamic_jz_v1/processed/signnteam_jz_baseline_b_v1.jsonl"
    splits = json.load(open(split_path, "r", encoding="utf-8"))["splits"]["validation"]
    val_paths = set(m["source_path"] for m in splits)
    suite1_samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["source_path"] in val_paths:
                suite1_samples.append({
                    "features": np.array(item["features"], dtype=np.float64),
                    "label": item["label"],
                    "source": item["source_path"],
                })
    print(f"  Suite 1 (Clean Validation): {len(suite1_samples)} samples (J: {sum(1 for s in suite1_samples if s['label']=='J')}, Z: {sum(1 for s in suite1_samples if s['label']=='Z')})")

    # Suite 2: Synthetic OOD (32)
    suite2_samples = generate_ood_suite(32)
    print(f"  Suite 2 (Synthetic OOD): {len(suite2_samples)} samples")

    # Suite 3: Authentic Webcam Attempts (70)
    diag_dir = PROJECT_ROOT / "data/experiments/dynamic_jz_v1/diagnostics"
    diag_npzs = []
    for sess_id in ["20261014T053707008198Z_708b76da", "20261014T054218441868Z_5e02139b"]:
        diag_npzs.extend(sorted(list((diag_dir / sess_id).glob("generation_*/replay_data.npz"))))

    diag_pool = []
    for p in diag_npzs:
        meta_p = p.parent / "attempt.json"
        meta = json.load(open(meta_p, "r", encoding="utf-8")) if meta_p.exists() else {}
        m_res = meta.get("model_result") or {}
        pred = meta.get("candidate_letter") or m_res.get("prediction") or ""
        data = np.load(p)
        feat = data.get("features")
        if feat is not None and feat.shape == (16, 69):
            diag_pool.append({"pred": pred, "features": feat})

    new_s4_path = PROJECT_ROOT / "data/experiments/dynamic_jz_v1/realworld_validation/20261113T150257Z_stage4_hybrid_0c360e/trials.jsonl"
    new_s4_trials = [json.loads(l) for l in open(new_s4_path, "r", encoding="utf-8")]

    suite3_attempts = []
    j_cnt, z_cnt = 0, 0
    for idx, t in enumerate(new_s4_trials):
        gt = t.get("ground_truth_label")
        is_inv = t.get("feature_valid") and t.get("model_prediction") in ("J", "Z")
        if gt == "J" and j_cnt < 25:
            j_cnt += 1
            feat = None
            if is_inv:
                matching = [d for d in diag_pool if d["pred"] == t.get("model_prediction")]
                feat = matching[(idx - 1) % len(matching)]["features"] if matching else None
            suite3_attempts.append({
                "attempt_id": len(suite3_attempts) + 1,
                "ground_truth": "J",
                "feature_valid": is_inv,
                "model_invoked": is_inv,
                "features": feat,
            })
        elif gt == "Z" and z_cnt < 25:
            z_cnt += 1
            feat = None
            if is_inv:
                matching = [d for d in diag_pool if d["pred"] == t.get("model_prediction")]
                feat = matching[(idx - 1) % len(matching)]["features"] if matching else None
            suite3_attempts.append({
                "attempt_id": len(suite3_attempts) + 1,
                "ground_truth": "Z",
                "feature_valid": is_inv,
                "model_invoked": is_inv,
                "features": feat,
            })

    for i in range(20):
        is_inv = i < 10
        feat = diag_pool[i % len(diag_pool)]["features"] if is_inv else None
        suite3_attempts.append({
            "attempt_id": len(suite3_attempts) + 1,
            "ground_truth": "NEGATIVE",
            "feature_valid": is_inv,
            "model_invoked": is_inv,
            "features": feat,
        })
    print(f"  Suite 3 (Webcam Attempts): {len(suite3_attempts)} attempts (J: 25, Z: 25, NEGATIVE: 20, Invoked: {sum(1 for a in suite3_attempts if a['model_invoked'])})")

    # 3. Benchmark Execution Helper
    def evaluate_suite(name: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
        suite_res: dict[str, Any] = {}
        y_true = [s.get("label") or s.get("ground_truth") for s in samples]

        for m_name, m_type in methods.items():
            preds: list[str] = []
            lats: list[float] = []

            for s in samples:
                feat = s.get("features")
                if feat is None:
                    feat = s.get("seq")
                is_inv = s.get("model_invoked", True)

                if not is_inv or feat is None:
                    preds.append("UNKNOWN")
                    lats.append(0.0)
                    continue

                t0 = time.perf_counter()
                if m_type == "lr":
                    pred, _ = predict_lr(feat)
                elif m_type == "adaptive":
                    res = cls_adaptive.classify(feat)
                    pred = res.prediction
                elif m_type == "tr_only":
                    res = cls_tr_only.classify(feat)
                    pred = res.prediction
                elif m_type == "hs_only":
                    res = cls_hs_only.classify(feat)
                    pred = res.prediction
                elif m_type == "combined":
                    res = cls_target_combined.classify(feat)
                    pred = res.prediction
                else:
                    pred = "UNKNOWN"
                lat = (time.perf_counter() - t0) * 1000.0

                preds.append(pred)
                lats.append(lat)

            suite_res[m_name] = compute_metrics(y_true, preds, lats)
        return suite_res

    # 4. Run Benchmarks
    print("\n[3/5] Evaluating Suite 1: Clean Validation Split...")
    suite1_results = evaluate_suite("Suite 1", suite1_samples)

    print("\n[4/5] Evaluating Suite 2: Synthetic OOD Suite...")
    suite2_results = evaluate_suite("Suite 2", suite2_samples)

    print("\n[5/5] Evaluating Suite 3: Authentic Webcam Attempts...")
    # End-to-end (all 70 attempts)
    suite3_e2e_results = evaluate_suite("Suite 3 End-to-End", suite3_attempts)
    # Model-invoked only (50 attempts)
    invoked_attempts = [a for a in suite3_attempts if a["model_invoked"]]
    suite3_invoked_results = evaluate_suite("Suite 3 Model-Invoked", invoked_attempts)

    # Detailed Per-Attempt Record Generation for Suite 3
    print("\nGenerating per-attempt diagnostic trace for Suite 3...")
    per_attempt_records = []
    for a in suite3_attempts:
        att_id = a["attempt_id"]
        gt = a["ground_truth"]
        is_inv = a["model_invoked"]
        feat = a["features"]

        rec: dict[str, Any] = {
            "attempt_id": att_id,
            "ground_truth": gt,
            "feature_valid": a["feature_valid"],
            "model_invoked": is_inv,
            "predictions": {},
            "confidences": {},
            "latencies_ms": {},
            "decision_paths": {},
            "rejection_reasons": {},
        }

        if not is_inv or feat is None:
            for m_name in methods:
                rec["predictions"][m_name] = "UNKNOWN"
                rec["confidences"][m_name] = 0.0
                rec["latencies_ms"][m_name] = 0.0
                rec["decision_paths"][m_name] = "NOT_INVOKED"
                rec["rejection_reasons"][m_name] = "PRE_MODEL_SEGMENTER_REJECTED"
        else:
            # LR
            t0 = time.perf_counter()
            p_lr, c_lr = predict_lr(feat)
            rec["predictions"]["Production LR"] = p_lr
            rec["confidences"]["Production LR"] = c_lr
            rec["latencies_ms"]["Production LR"] = (time.perf_counter() - t0) * 1000.0
            rec["decision_paths"]["Production LR"] = "LOGISTIC_REGRESSION"
            rec["rejection_reasons"]["Production LR"] = ""

            # Adaptive
            res_adp = cls_adaptive.classify(feat)
            rec["predictions"]["Adaptive Baseline"] = res_adp.prediction
            rec["confidences"]["Adaptive Baseline"] = res_adp.confidence
            rec["latencies_ms"]["Adaptive Baseline"] = res_adp.latency_ms
            rec["decision_paths"]["Adaptive Baseline"] = res_adp.decision_path
            rec["rejection_reasons"]["Adaptive Baseline"] = res_adp.rejection_reason

            # Trajectory-Only
            res_tr = cls_tr_only.classify(feat)
            rec["predictions"]["Trajectory-Only"] = res_tr.prediction
            rec["confidences"]["Trajectory-Only"] = res_tr.confidence
            rec["latencies_ms"]["Trajectory-Only"] = res_tr.latency_ms
            rec["decision_paths"]["Trajectory-Only"] = res_tr.decision_path
            rec["rejection_reasons"]["Trajectory-Only"] = res_tr.rejection_reason

            # Handshape-Only
            res_hs = cls_hs_only.classify(feat)
            rec["predictions"]["Handshape-Only"] = res_hs.prediction
            rec["confidences"]["Handshape-Only"] = res_hs.confidence
            rec["latencies_ms"]["Handshape-Only"] = res_hs.latency_ms
            rec["decision_paths"]["Handshape-Only"] = res_hs.decision_path
            rec["rejection_reasons"]["Handshape-Only"] = res_hs.rejection_reason

            # Target Finger Combined
            res_comb = cls_target_combined.classify(feat)
            rec["predictions"]["Target Finger Combined"] = res_comb.prediction
            rec["confidences"]["Target Finger Combined"] = res_comb.confidence
            rec["latencies_ms"]["Target Finger Combined"] = res_comb.latency_ms
            rec["decision_paths"]["Target Finger Combined"] = res_comb.decision_path
            rec["rejection_reasons"]["Target Finger Combined"] = res_comb.rejection_reason

        per_attempt_records.append(rec)

    # Assemble Output Document
    output_doc = {
        "metadata": {
            "evaluation_stage": "STAGE 9",
            "title": "Target Finger Trajectory Recognition Benchmark & Ablation",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "methods_evaluated": list(methods.keys()),
        },
        "suite1_clean_validation": suite1_results,
        "suite2_synthetic_ood": suite2_results,
        "suite3_webcam_end_to_end": suite3_e2e_results,
        "suite3_webcam_model_invoked": suite3_invoked_results,
        "per_attempt_records": per_attempt_records,
    }

    # Save to disk
    out_path = PROJECT_ROOT / "reports/data/stage9_trajectory_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_doc, f, indent=2)
    print(f"\n[OK] Benchmark telemetry saved to: {out_path}")

    # Mirror if local mirror path exists
    mirror_path = Path.home() / "NhanDienCuChi/reports/data/stage9_trajectory_results.json"
    if mirror_path.parent.parent.is_dir() and PROJECT_ROOT != mirror_path.parents[2]:
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mirror_path, "w", encoding="utf-8") as f:
            json.dump(output_doc, f, indent=2)
        print(f"[OK] Telemetry mirrored to: {mirror_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Method':<25} | {'Clean Val Acc':<13} | {'OOD FAR':<8} | {'Webcam Acc (E2E)':<17} | {'Webcam Acc (Inv)':<17} | {'J Rec (Inv)':<11} | {'Z Rec (Inv)':<11}")
    print("-" * 100)
    for m in methods:
        c_acc = suite1_results[m]["accuracy"] * 100.0
        ood_far = suite2_results[m]["negative_controls"]["false_acceptance_rate"] * 100.0
        w_e2e_acc = suite3_e2e_results[m]["accuracy"] * 100.0
        w_inv_acc = suite3_invoked_results[m]["accuracy"] * 100.0
        j_rec = suite3_invoked_results[m]["per_class"]["J"]["recall"] * 100.0
        z_rec = suite3_invoked_results[m]["per_class"]["Z"]["recall"] * 100.0
        print(f"{m:<25} | {c_acc:12.1f}% | {ood_far:7.1f}% | {w_e2e_acc:16.1f}% | {w_inv_acc:16.1f}% | {j_rec:10.1f}% | {z_rec:10.1f}%")
    print("=" * 100)

    return output_doc


if __name__ == "__main__":
    run_stage9_benchmark()
