"""Stage 2 Experiments Runner: Template Selection, Offline Evaluation, and Latency Benchmark.

Strictly follows Stage 2 Protocol:
- Train split (512 samples) exclusively used for template selection.
- Validation split (64 samples) used for evaluation and hyperparameter calibration.
- Test split (64 samples) is strictly untouched.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference.dynamic_jz.finger_tracker import FingerTracker
from app.inference.dynamic_jz.feature_builder import build_dtw_features, FEATURE_CONFIGURATIONS
from app.inference.dynamic_jz.template_store import (
    TemplateStore,
    DTWTemplate,
    select_k_medoids_for_class,
)
from app.inference.dynamic_jz.dtw_engine import dtw_multivariate
from app.inference.dynamic_jz.dtw_matcher import DTWMatcher, DTWMatchResult


def run_all() -> None:
    print("=" * 70)
    print("STAGE 2 — MULTIVARIATE DTW ENGINE & TEMPLATE SELECTION EXPERIMENTS")
    print("=" * 70)

    # 1. Load split manifest and data
    split_path = PROJECT_ROOT / "data/experiments/dynamic_jz_binary_baseline_v1/split_manifest.json"
    data_path = PROJECT_ROOT / "data/experiments/dynamic_jz_v1/processed/signnteam_jz_baseline_b_v1.jsonl"
    template_dir = PROJECT_ROOT / "models/dynamic_jz/dtw_templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    report_data_dir = PROJECT_ROOT / "reports/data"
    report_data_dir.mkdir(parents=True, exist_ok=True)

    with open(split_path, "r", encoding="utf-8") as f:
        splits_meta = json.load(f)

    samples_by_path = {}
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            samples_by_path[item["source_path"]] = item

    train_meta = splits_meta["splits"]["train"]
    val_meta = splits_meta["splits"]["validation"]
    test_meta = splits_meta["splits"]["test"]

    print(f"Loaded dataset: Train={len(train_meta)}, Val={len(val_meta)}, Test={len(test_meta)} (Test HELD OUT)")
    assert len(train_meta) == 512, f"Expected 512 train samples, got {len(train_meta)}"
    assert len(val_meta) == 64, f"Expected 64 validation samples, got {len(val_meta)}"
    assert len(test_meta) == 64, f"Expected 64 test samples, got {len(test_meta)}"

    # 2. Extract Finger Tracking for Train and Validation
    tracker = FingerTracker()
    print("\nExtracting Finger Tracking for Train split (512 samples)...")
    train_samples_j = []
    train_samples_z = []

    for m in train_meta:
        sp = m["source_path"]
        raw = samples_by_path[sp]
        seq = np.array(raw["features"], dtype=np.float64)
        lbl = raw["label"]
        frame_indices = raw.get("selected_original_frame_indices")
        res = tracker.extract(seq, letter_target=lbl, selected_original_frame_indices=frame_indices)
        if lbl == "J":
            train_samples_j.append((sp, res))
        else:
            train_samples_z.append((sp, res))

    print(f"Train extraction complete: J={len(train_samples_j)}, Z={len(train_samples_z)}")
    j_valid = sum(1 for _, r in train_samples_j if r.validity_status == "VALID")
    z_valid = sum(1 for _, r in train_samples_z if r.validity_status == "VALID")
    print(f"  J: VALID={j_valid}, SUSPICIOUS={len(train_samples_j) - j_valid}")
    print(f"  Z: VALID={z_valid}, SUSPICIOUS={len(train_samples_z) - z_valid}")

    print("\nExtracting Finger Tracking for Validation split (64 samples)...")
    val_samples = []
    for m in val_meta:
        sp = m["source_path"]
        raw = samples_by_path[sp]
        seq = np.array(raw["features"], dtype=np.float64)
        lbl = raw["label"]
        frame_indices = raw.get("selected_original_frame_indices")
        # Extract both J and Z hypothesis representations for validation
        res_j = tracker.extract(seq, letter_target="J", selected_original_frame_indices=frame_indices)
        res_z = tracker.extract(seq, letter_target="Z", selected_original_frame_indices=frame_indices)
        val_samples.append({
            "source_path": sp,
            "label": lbl,
            "sequence": seq,
            "res_j": res_j,
            "res_z": res_z,
        })
    print("Validation extraction complete (64 samples).")

    # 3. Generate Templates via k-Medoids across Configurations and Policies
    configs = ["CONFIG_A", "CONFIG_B", "CONFIG_C", "CONFIG_D"]
    policies = ["VALID_ONLY", "VALID_AND_SUSPICIOUS"]
    k_values = [1, 2, 4, 8]
    radii = [2, 3, 4]

    template_catalogue: dict[str, TemplateStore] = {}

    print("\n" + "=" * 70)
    print("GENERATING DTW TEMPLATES ON TRAIN SPLIT (k-Medoids)")
    print("=" * 70)

    for cfg in configs:
        for pol in policies:
            for k in k_values:
                store_key = f"{cfg}_{pol}_k{k}"
                save_file = template_dir / f"templates_{cfg.lower()}_{pol.lower()}_k{k}.json"

                # Run k-Medoids for J and Z
                medoids_j = select_k_medoids_for_class(
                    train_samples_j, label="J", k=k, config_name=cfg, policy=pol, radius=3
                )
                medoids_z = select_k_medoids_for_class(
                    train_samples_z, label="Z", k=k, config_name=cfg, policy=pol, radius=3
                )

                store = TemplateStore(medoids_j + medoids_z)
                store.save_json(save_file)
                template_catalogue[store_key] = store
                print(f"  Generated {store_key}: {len(medoids_j)} J + {len(medoids_z)} Z templates -> {save_file.name}")

    # 4. Run Offline Evaluation on Validation Split across all conditions
    print("\n" + "=" * 70)
    print("EVALUATING ON VALIDATION SPLIT (64 samples)")
    print("=" * 70)

    results_table = []

    for cfg in configs:
        for pol in policies:
            for k in k_values:
                for r in radii:
                    store_key = f"{cfg}_{pol}_k{k}"
                    store = template_catalogue[store_key]

                    # Instantiate Matcher with infinite threshold and 0 margin (raw classification)
                    matcher = DTWMatcher(
                        store,
                        config_name=cfg,
                        radius=r,
                        threshold_j=1e9,
                        threshold_z=1e9,
                        min_margin=0.0,
                    )

                    # Evaluate on all 64 validation samples
                    y_true = []
                    y_pred = []
                    margins = []
                    winning_dists = []
                    losing_dists = []

                    tp_j = fp_j = fn_j = tn_j = 0
                    tp_z = fp_z = fn_z = tn_z = 0

                    for item in val_samples:
                        lbl = item["label"]
                        y_true.append(lbl)

                        # Dual hypothesis match
                        match_res = matcher.match_dual(item["res_j"], item["res_z"])
                        pred = match_res.predicted_label
                        y_pred.append(pred)

                        margin = match_res.distance_margin
                        margins.append(margin)

                        if pred == "J":
                            winning_dists.append(match_res.distance_j)
                            losing_dists.append(match_res.distance_z)
                        else:
                            winning_dists.append(match_res.distance_z)
                            losing_dists.append(match_res.distance_j)

                        if lbl == "J":
                            if pred == "J":
                                tp_j += 1
                                tn_z += 1
                            else:
                                fn_j += 1
                                fp_z += 1
                        else:  # lbl == 'Z'
                            if pred == "Z":
                                tp_z += 1
                                tn_j += 1
                            else:
                                fn_z += 1
                                fp_j += 1

                    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
                    accuracy = correct / len(y_true)

                    prec_j = tp_j / (tp_j + fp_j) if (tp_j + fp_j) > 0 else 0.0
                    rec_j = tp_j / (tp_j + fn_j) if (tp_j + fn_j) > 0 else 0.0
                    f1_j = 2 * prec_j * rec_j / (prec_j + rec_j) if (prec_j + rec_j) > 0 else 0.0

                    prec_z = tp_z / (tp_z + fp_z) if (tp_z + fp_z) > 0 else 0.0
                    rec_z = tp_z / (tp_z + fn_z) if (tp_z + fn_z) > 0 else 0.0
                    f1_z = 2 * prec_z * rec_z / (prec_z + rec_z) if (prec_z + rec_z) > 0 else 0.0

                    res_entry = {
                        "config": cfg,
                        "policy": pol,
                        "k": k,
                        "radius": r,
                        "accuracy": float(accuracy),
                        "correct": correct,
                        "total": len(y_true),
                        "j_tp": tp_j,
                        "j_fp": fp_j,
                        "j_fn": fn_j,
                        "j_prec": float(prec_j),
                        "j_rec": float(rec_j),
                        "j_f1": float(f1_j),
                        "z_tp": tp_z,
                        "z_fp": fp_z,
                        "z_fn": fn_z,
                        "z_prec": float(prec_z),
                        "z_rec": float(rec_z),
                        "z_f1": float(f1_z),
                        "mean_margin": float(np.mean(margins)),
                        "median_margin": float(np.median(margins)),
                        "mean_winning_dist": float(np.mean(winning_dists)),
                        "mean_losing_dist": float(np.mean(losing_dists)),
                    }
                    results_table.append(res_entry)

    # Sort results by accuracy desc, mean_margin desc
    results_table.sort(key=lambda x: (x["accuracy"], x["mean_margin"]), reverse=True)

    print("\nTop 10 Evaluation Configurations on Validation Split:")
    print(f"{'Config':<10} {'Policy':<22} {'K':<3} {'R':<3} {'Accuracy':<10} {'J-F1':<8} {'Z-F1':<8} {'Mean Margin':<12}")
    print("-" * 80)
    for row in results_table[:10]:
        print(f"{row['config']:<10} {row['policy']:<22} {row['k']:<3} {row['radius']:<3} {row['accuracy']*100:>6.2f}%    {row['j_f1']:>6.4f}   {row['z_f1']:>6.4f}   {row['mean_margin']:>10.4f}")

    # 5. Threshold and Margin Sweep on Best Configuration
    best_cfg = results_table[0]
    print(f"\nBest Configuration: {best_cfg['config']} | Policy={best_cfg['policy']} | K={best_cfg['k']} | R={best_cfg['radius']}")
    print(f"Accuracy: {best_cfg['accuracy']*100:.2f}% | J-F1={best_cfg['j_f1']:.4f} | Z-F1={best_cfg['z_f1']:.4f}")

    best_store = template_catalogue[f"{best_cfg['config']}_{best_cfg['policy']}_k{best_cfg['k']}"]

    print("\nCalibrating Rejection Parameters (Threshold & Margin Sweep)...")
    calibration_results = []
    thresh_candidates = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    margin_candidates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    for th in thresh_candidates:
        for mg in margin_candidates:
            calib_matcher = DTWMatcher(
                best_store,
                config_name=best_cfg["config"],
                radius=best_cfg["radius"],
                threshold_j=th,
                threshold_z=th,
                min_margin=mg,
            )

            correct = 0
            unknown = 0
            error = 0

            for item in val_samples:
                lbl = item["label"]
                match_res = calib_matcher.match_dual(item["res_j"], item["res_z"])
                pred = match_res.predicted_label

                if pred == "UNKNOWN":
                    unknown += 1
                elif pred == lbl:
                    correct += 1
                else:
                    error += 1

            decided = correct + error
            acc_on_decided = correct / decided if decided > 0 else 0.0
            overall_effective_acc = correct / len(val_samples)

            calibration_results.append({
                "threshold": th,
                "margin": mg,
                "correct": correct,
                "unknown": unknown,
                "error": error,
                "unknown_rate": float(unknown / len(val_samples)),
                "acc_on_decided": float(acc_on_decided),
                "effective_accuracy": float(overall_effective_acc),
            })

    # 6. Latency Benchmarking (>= 500 iterations across K)
    print("\n" + "=" * 70)
    print("LATENCY BENCHMARKING (500 iterations per K)")
    print("=" * 70)

    sample_seq = val_samples[0]["sequence"]
    benchmark_results = {}

    for k in k_values:
        store = template_catalogue[f"CONFIG_A_VALID_ONLY_k{k}"]
        bench_matcher = DTWMatcher(store, config_name="CONFIG_A", radius=3)

        extraction_times = []
        matching_times = []
        total_times = []

        # Warmup (10 runs)
        for _ in range(10):
            t0 = time.perf_counter()
            rj = tracker.extract(sample_seq, letter_target="J")
            rz = tracker.extract(sample_seq, letter_target="Z")
            t1 = time.perf_counter()
            _ = bench_matcher.match_dual(rj, rz)
            t2 = time.perf_counter()

        # Benchmark 500 iterations
        for _ in range(500):
            t0 = time.perf_counter()
            rj = tracker.extract(sample_seq, letter_target="J")
            rz = tracker.extract(sample_seq, letter_target="Z")
            t1 = time.perf_counter()
            _ = bench_matcher.match_dual(rj, rz)
            t2 = time.perf_counter()

            extraction_times.append((t1 - t0) * 1000.0)
            matching_times.append((t2 - t1) * 1000.0)
            total_times.append((t2 - t0) * 1000.0)

        stats = {
            "templates_count": 2 * k,
            "extraction_mean_ms": float(np.mean(extraction_times)),
            "extraction_p95_ms": float(np.percentile(extraction_times, 95)),
            "extraction_p99_ms": float(np.percentile(extraction_times, 99)),
            "matching_mean_ms": float(np.mean(matching_times)),
            "matching_median_ms": float(np.median(matching_times)),
            "matching_p95_ms": float(np.percentile(matching_times, 95)),
            "matching_p99_ms": float(np.percentile(matching_times, 99)),
            "total_mean_ms": float(np.mean(total_times)),
            "total_median_ms": float(np.median(total_times)),
            "total_p95_ms": float(np.percentile(total_times, 95)),
            "total_p99_ms": float(np.percentile(total_times, 99)),
        }
        benchmark_results[f"K={k}"] = stats
        print(f"K={k} ({2*k} templates): Matching Mean={stats['matching_mean_ms']:.3f}ms, P95={stats['matching_p95_ms']:.3f}ms, P99={stats['matching_p99_ms']:.3f}ms | Total Mean={stats['total_mean_ms']:.3f}ms")

    # 7. Save complete experiments data
    export_payload = {
        "best_configuration": best_cfg,
        "all_evaluations": results_table,
        "calibration_sweep": calibration_results,
        "latency_benchmark": benchmark_results,
    }

    out_json = report_data_dir / "stage2_experiment_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    print(f"\nAll experiment results exported successfully to: {out_json}")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
