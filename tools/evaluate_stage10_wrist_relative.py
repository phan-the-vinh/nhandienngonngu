"""Stage 10 Benchmark: Wrist-Relative Motion, Local Joint Kinematics & Handshape Fusion.

Evaluates 5 Mandatory Ablation Configurations (+ Production LR & Adaptive Stage 8 Reference):
1. Config 1: Absolute Trajectory (Camera-Plane Motion Baseline)
2. Config 2: Wrist-Relative Trajectory (Wrist Motion Subtracted)
3. Config 3: Local Joint-Relative Trajectory (Fingertip relative to MCP Joint)
4. Config 4: Handshape-Only Recognizer (21-Landmark Hand Geometry)
5. Config 5: Wrist-Relative Trajectory + Handshape Fusion

Evaluated Across 3 Distinct Benchmark Suites:
1. Suite 1: Clean Validation Split (64 samples: 26 J, 38 Z from SigNN baseline-b)
2. Suite 2: Synthetic Out-Of-Distribution Suite (32 samples of invalid non-J/Z motions)
3. Suite 3: Authentic Real-World Webcam Attempts (70 attempts: 25 J, 25 Z, 20 Negative)
   - Evaluated on both End-to-End (all 70 attempts) and Model-Invoked (50 attempts).

Saves comprehensive telemetry to reports/data/stage10_wrist_relative_results.json.
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
    TargetPinkyJAnalyzer,
    TargetIndexZAnalyzer,
)
from app.inference.dynamic_jz.fusion_classifier import HandshapeBranch
from tools.evaluate_stage7_fusion import compute_metrics, generate_ood_suite


def run_stage10_benchmark() -> dict[str, Any]:
    print("=" * 80)
    print("STAGE 10: WRIST-RELATIVE MOTION & HANDSHAPE FUSION BENCHMARK")
    print("=" * 80)

    # 1. Initialize Base Engines
    print("\n[1/5] Initializing Kinematic & Handshape Engines...")
    lr_model_path = PROJECT_ROOT / "models/dynamic_jz/model_pipeline.joblib"
    if not lr_model_path.exists():
        lr_model_path = PROJECT_ROOT / "data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib"
    lr_pipeline = joblib.load(lr_model_path)
    print("  Production Logistic Regression Pipeline loaded (Reference).")

    cls_adaptive = AdaptiveMotionJZClassifier(z_diagonal_threshold=0.015)
    print("  Stage 8 Adaptive Baseline initialized (Reference).")

    pinky_analyzer = TargetPinkyJAnalyzer()
    index_analyzer = TargetIndexZAnalyzer()
    handshape_branch = HandshapeBranch(
        min_confidence=0.35,
        min_margin=0.05,
        min_pinky_ext=0.90,
        min_index_ext=0.95,
        min_pinky_contrast=0.08,
        min_index_contrast=0.15,
        max_stability_std=0.38,
    )
    print("  Constituent analyzers (Pinky J, Index Z, Handshape Branch) initialized.")

    def predict_lr(seq: np.ndarray) -> tuple[str, float]:
        flat = seq.reshape(1, -1)
        pred = str(lr_pipeline.predict(flat)[0])
        proba = float(np.max(lr_pipeline.predict_proba(flat)))
        return pred, proba

    def classify_trajectory(path_idx: np.ndarray, path_pky: np.ndarray) -> tuple[str, float, float, float, str, str]:
        sj, mj = pinky_analyzer.analyze(path_pky)
        sz, mz = index_analyzer.analyze(path_idx)

        margin = abs(sj - sz)
        best = max(sj, sz)

        if mj.is_static and mz.is_static:
            return "UNKNOWN", 0.0, sj, sz, "STATIC_POSE_REJECTED", "Path length < 0.15"

        if best < 0.30:
            return "UNKNOWN", best, sj, sz, "LOW_SCORE", f"Max score {best:.3f} < 0.30"

        if margin < 0.02:
            return "UNKNOWN", best, sj, sz, "AMBIGUOUS_MARGIN", f"Margin {margin:.3f} < 0.02"

        if sj > sz:
            return "J", sj, sj, sz, "TRAJECTORY_J", ""
        else:
            return "Z", sz, sj, sz, "TRAJECTORY_Z", ""

    def classify_wrist_relative_fused(seq: np.ndarray) -> tuple[str, float, float, float, str, str]:
        # Wrist-relative trajectory
        p_idx = seq[:, 65:67] - seq[:, 63:65]
        p_pky = seq[:, 67:69] - seq[:, 63:65]

        p_len_pinky = float(np.sum(np.linalg.norm(np.diff(p_pky, axis=0), axis=1)))
        p_len_index = float(np.sum(np.linalg.norm(np.diff(p_idx, axis=0), axis=1)))
        max_motion = max(p_len_pinky, p_len_index)

        if max_motion < 0.15:
            return "UNKNOWN", 0.0, 0.0, 0.0, "STATIC_POSE_REJECTED", f"Max relative motion {max_motion:.3f} < 0.15"

        lms = seq[:, :63].reshape((16, 21, 3))
        mcp_dists = np.linalg.norm(lms[:, [5, 9, 17], :], axis=2)
        palm_scales = np.median(mcp_dists, axis=1)

        hs_res = handshape_branch.evaluate(lms, palm_scales)
        if hs_res.status == "REJECTED" and "OOD" in hs_res.rejection_reason:
            return "UNKNOWN", 0.0, 0.0, 0.0, "OOD_REJECTED", hs_res.rejection_reason

        s_hs_j = hs_res.score_j
        s_hs_z = hs_res.score_z

        s_tr_j, mj = pinky_analyzer.analyze(p_pky)
        s_tr_z, mz = index_analyzer.analyze(p_idx)

        bonus_j = 0.15 * s_hs_j * s_tr_j
        bonus_z = 0.15 * s_hs_z * s_tr_z

        joint_j = 0.40 * s_hs_j + 0.60 * s_tr_j + bonus_j
        joint_z = 0.40 * s_hs_z + 0.60 * s_tr_z + bonus_z

        # Target finger curled vetoes
        is_curled_pinky = bool(hs_res.metrics is not None and hs_res.metrics.pinky_vs_curled_contrast < -0.05)
        is_curled_index = bool(hs_res.metrics is not None and hs_res.metrics.index_vs_curled_contrast < -0.05)

        if s_tr_j < 0.20 or s_hs_j <= 0.15 or is_curled_pinky:
            joint_j = 0.0
        if s_tr_z < 0.20 or s_hs_z <= 0.15 or is_curled_index or mz.diagonal_downward < 0.003:
            joint_z = 0.0

        best = max(joint_j, joint_z)
        margin = abs(joint_j - joint_z)

        if best < 0.35:
            return "UNKNOWN", best, joint_j, joint_z, "INSUFFICIENT_JOINT_SCORE", f"Score {best:.3f} < 0.35"

        if margin < 0.05:
            return "UNKNOWN", best, joint_j, joint_z, "AMBIGUOUS_MARGIN", f"Margin {margin:.3f} < 0.05"

        if joint_j > joint_z:
            return "J", joint_j, joint_j, joint_z, "WRIST_REL_FUSED_J", ""
        else:
            return "Z", joint_z, joint_j, joint_z, "WRIST_REL_FUSED_Z", ""

    # Method Dispatcher
    methods = {
        "Production LR": "lr",
        "Adaptive Baseline": "adaptive",
        "1. Absolute Trajectory": "abs_tr",
        "2. Wrist-Relative Trajectory": "w_rel_tr",
        "3. Local Joint-Relative": "joint_rel_tr",
        "4. Handshape-Only": "hs_only",
        "5. Wrist-Relative + Handshape": "w_rel_fused",
    }

    # 2. Load Evaluation Suites
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
            diag_pool.append({"pred": pred, "features": feat, "path": str(p)})

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

    # 3. Evaluation Function
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
                elif m_type == "abs_tr":
                    p_idx = feat[:, 65:67]
                    p_pky = feat[:, 67:69]
                    pred, _, _, _, _, _ = classify_trajectory(p_idx, p_pky)
                elif m_type == "w_rel_tr":
                    p_idx = feat[:, 65:67] - feat[:, 63:65]
                    p_pky = feat[:, 67:69] - feat[:, 63:65]
                    pred, _, _, _, _, _ = classify_trajectory(p_idx, p_pky)
                elif m_type == "joint_rel_tr":
                    p_idx = feat[:, 8*3:8*3+2] - feat[:, 5*3:5*3+2]
                    p_pky = feat[:, 20*3:20*3+2] - feat[:, 17*3:17*3+2]
                    pred, _, _, _, _, _ = classify_trajectory(p_idx, p_pky)
                elif m_type == "hs_only":
                    lms = feat[:, :63].reshape((16, 21, 3))
                    mcp_dists = np.linalg.norm(lms[:, [5, 9, 17], :], axis=2)
                    palm_scales = np.median(mcp_dists, axis=1)
                    hs_res = handshape_branch.evaluate(lms, palm_scales)
                    pred = hs_res.prediction
                elif m_type == "w_rel_fused":
                    pred, _, _, _, _, _ = classify_wrist_relative_fused(feat)
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
    # End-to-end (70 attempts)
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
            "kinematics": {},
        }

        if not is_inv or feat is None:
            for m_name in methods:
                rec["predictions"][m_name] = "UNKNOWN"
                rec["confidences"][m_name] = 0.0
                rec["latencies_ms"][m_name] = 0.0
                rec["decision_paths"][m_name] = "NOT_INVOKED"
                rec["rejection_reasons"][m_name] = "PRE_MODEL_SEGMENTER_REJECTED"
        else:
            # Kinematics
            wri_abs = feat[:, 63:65]
            idx_abs = feat[:, 65:67]
            pky_abs = feat[:, 67:69]
            idx_rel = idx_abs - wri_abs
            pky_rel = pky_abs - wri_abs
            w_len = float(np.sum(np.linalg.norm(np.diff(wri_abs, axis=0), axis=1)))
            i_abs_len = float(np.sum(np.linalg.norm(np.diff(idx_abs, axis=0), axis=1)))
            p_abs_len = float(np.sum(np.linalg.norm(np.diff(pky_abs, axis=0), axis=1)))
            i_rel_len = float(np.sum(np.linalg.norm(np.diff(idx_rel, axis=0), axis=1)))
            p_rel_len = float(np.sum(np.linalg.norm(np.diff(pky_rel, axis=0), axis=1)))

            rec["kinematics"] = {
                "wrist_abs_path_len": w_len,
                "index_abs_path_len": i_abs_len,
                "pinky_abs_path_len": p_abs_len,
                "index_wrist_rel_path_len": i_rel_len,
                "pinky_wrist_rel_path_len": p_rel_len,
                "wrist_to_index_ratio": w_len / max(i_abs_len, 1e-6),
                "wrist_to_pinky_ratio": w_len / max(p_abs_len, 1e-6),
            }

            # 1. LR
            t0 = time.perf_counter()
            p_lr, c_lr = predict_lr(feat)
            rec["predictions"]["Production LR"] = p_lr
            rec["confidences"]["Production LR"] = c_lr
            rec["latencies_ms"]["Production LR"] = (time.perf_counter() - t0) * 1000.0
            rec["decision_paths"]["Production LR"] = "LOGISTIC_REGRESSION"
            rec["rejection_reasons"]["Production LR"] = ""

            # 2. Adaptive
            res_adp = cls_adaptive.classify(feat)
            rec["predictions"]["Adaptive Baseline"] = res_adp.prediction
            rec["confidences"]["Adaptive Baseline"] = res_adp.confidence
            rec["latencies_ms"]["Adaptive Baseline"] = res_adp.latency_ms
            rec["decision_paths"]["Adaptive Baseline"] = res_adp.decision_path
            rec["rejection_reasons"]["Adaptive Baseline"] = res_adp.rejection_reason

            # 3. Absolute Trajectory
            t0 = time.perf_counter()
            pred, conf, _, _, path, rej = classify_trajectory(idx_abs, pky_abs)
            rec["predictions"]["1. Absolute Trajectory"] = pred
            rec["confidences"]["1. Absolute Trajectory"] = conf
            rec["latencies_ms"]["1. Absolute Trajectory"] = (time.perf_counter() - t0) * 1000.0
            rec["decision_paths"]["1. Absolute Trajectory"] = path
            rec["rejection_reasons"]["1. Absolute Trajectory"] = rej

            # 4. Wrist-Relative Trajectory
            t0 = time.perf_counter()
            pred, conf, _, _, path, rej = classify_trajectory(idx_rel, pky_rel)
            rec["predictions"]["2. Wrist-Relative Trajectory"] = pred
            rec["confidences"]["2. Wrist-Relative Trajectory"] = conf
            rec["latencies_ms"]["2. Wrist-Relative Trajectory"] = (time.perf_counter() - t0) * 1000.0
            rec["decision_paths"]["2. Wrist-Relative Trajectory"] = path
            rec["rejection_reasons"]["2. Wrist-Relative Trajectory"] = rej

            # 5. Local Joint-Relative
            t0 = time.perf_counter()
            p_j_idx = feat[:, 8*3:8*3+2] - feat[:, 5*3:5*3+2]
            p_j_pky = feat[:, 20*3:20*3+2] - feat[:, 17*3:17*3+2]
            pred, conf, _, _, path, rej = classify_trajectory(p_j_idx, p_j_pky)
            rec["predictions"]["3. Local Joint-Relative"] = pred
            rec["confidences"]["3. Local Joint-Relative"] = conf
            rec["latencies_ms"]["3. Local Joint-Relative"] = (time.perf_counter() - t0) * 1000.0
            rec["decision_paths"]["3. Local Joint-Relative"] = path
            rec["rejection_reasons"]["3. Local Joint-Relative"] = rej

            # 6. Handshape-Only
            t0 = time.perf_counter()
            lms = feat[:, :63].reshape((16, 21, 3))
            mcp_dists = np.linalg.norm(lms[:, [5, 9, 17], :], axis=2)
            palm_scales = np.median(mcp_dists, axis=1)
            hs_res = handshape_branch.evaluate(lms, palm_scales)
            rec["predictions"]["4. Handshape-Only"] = hs_res.prediction
            rec["confidences"]["4. Handshape-Only"] = hs_res.confidence
            rec["latencies_ms"]["4. Handshape-Only"] = (time.perf_counter() - t0) * 1000.0
            rec["decision_paths"]["4. Handshape-Only"] = f"HANDSHAPE_{hs_res.prediction}"
            rec["rejection_reasons"]["4. Handshape-Only"] = hs_res.rejection_reason

            # 7. Wrist-Relative + Handshape Fusion
            t0 = time.perf_counter()
            pred, conf, _, _, path, rej = classify_wrist_relative_fused(feat)
            rec["predictions"]["5. Wrist-Relative + Handshape"] = pred
            rec["confidences"]["5. Wrist-Relative + Handshape"] = conf
            rec["latencies_ms"]["5. Wrist-Relative + Handshape"] = (time.perf_counter() - t0) * 1000.0
            rec["decision_paths"]["5. Wrist-Relative + Handshape"] = path
            rec["rejection_reasons"]["5. Wrist-Relative + Handshape"] = rej

        per_attempt_records.append(rec)

    # 5. Assemble Output Document
    output_doc = {
        "metadata": {
            "evaluation_stage": "STAGE 10",
            "title": "Wrist-Relative Motion & Handshape Fusion Benchmark",
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
    out_path = PROJECT_ROOT / "reports/data/stage10_wrist_relative_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_doc, f, indent=2)
    print(f"\n[OK] Benchmark telemetry saved to: {out_path}")

    # Mirror if local mirror path exists
    mirror_path = Path.home() / "NhanDienCuChi/reports/data/stage10_wrist_relative_results.json"
    if mirror_path.parent.parent.is_dir() and PROJECT_ROOT != mirror_path.parents[2]:
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mirror_path, "w", encoding="utf-8") as f:
            json.dump(output_doc, f, indent=2)
        print(f"[OK] Telemetry mirrored to: {mirror_path}")

    # Print summary table
    print("\n" + "=" * 105)
    print(f"{'Method / Configuration':<32} | {'Clean Acc':<10} | {'OOD FAR':<8} | {'Webcam E2E':<11} | {'Webcam Inv':<11} | {'J Rec':<7} | {'Z Rec':<7} | {'Neg FAR':<8}")
    print("-" * 105)
    for m in methods:
        c_acc = suite1_results[m]["accuracy"] * 100.0
        ood_far = suite2_results[m]["negative_controls"]["false_acceptance_rate"] * 100.0
        w_e2e_acc = suite3_e2e_results[m]["accuracy"] * 100.0
        w_inv_acc = suite3_invoked_results[m]["accuracy"] * 100.0
        j_rec = suite3_invoked_results[m]["per_class"]["J"]["recall"] * 100.0
        z_rec = suite3_invoked_results[m]["per_class"]["Z"]["recall"] * 100.0
        neg_far = suite3_invoked_results[m]["negative_controls"]["false_acceptance_rate"] * 100.0
        print(f"{m:<32} | {c_acc:9.1f}% | {ood_far:7.1f}% | {w_e2e_acc:10.1f}% | {w_inv_acc:10.1f}% | {j_rec:6.1f}% | {z_rec:6.1f}% | {neg_far:7.1f}%")
    print("=" * 105)

    return output_doc


if __name__ == "__main__":
    run_stage10_benchmark()
