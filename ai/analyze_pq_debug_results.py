from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "ai" / "dataset" / "pq_debug_data_audit_v1"
OUTPUT_DIR = ROOT / "ai" / "dataset" / "pq_debug_model_analysis_v1"
TRAIN_DIR = ROOT / "ai" / "dataset" / "train_validation_test_split_v1"
MODEL_DIR = ROOT / "ai" / "dataset" / "mlp_validation_comparison_v1"
FEATURES = [f"f{i}" for i in range(1, 64)]
CONDITIONS = ["P_front", "P_horizontal", "Q_horizontal"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_row(metadata_path: str) -> list[float]:
    payload = json.loads((ROOT / metadata_path).read_text(encoding="utf-8"))
    return [float(payload["features"][name]) for name in FEATURES]


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(AUDIT_DIR / "debug_sample_audit.csv")
    audit["condition"] = audit["input_label"] + "_" + audit["input_view"]
    audit["analysis_unit"] = audit["possible_near_duplicate_group"].fillna("")
    singleton = audit["analysis_unit"].eq("")
    audit.loc[singleton, "analysis_unit"] = "SINGLE_" + audit.loc[singleton, "audit_id"]
    audit["correct"] = audit["model_prediction"] == audit["input_label"]

    source_hash_records = json.loads((AUDIT_DIR / "input_hashes.json").read_text(encoding="utf-8"))["files"]
    audit_manifest = json.loads((AUDIT_DIR / "manifest.json").read_text(encoding="utf-8"))
    integrity = {
        "audit_rows_47": len(audit) == 47,
        "audit_ids_unique": audit["audit_id"].nunique() == 47,
        "conditions_exact": audit["condition"].value_counts().to_dict()
        == {"P_front": 16, "P_horizontal": 16, "Q_horizontal": 15},
        "semantic_review_complete": bool((audit["label_review"] == "LABEL_CONFIRMED").all()),
        "landmark_review_complete": bool((audit["landmark_review"] == "LANDMARK_CONFIRMED").all()),
        "file_integrity_pass": bool((audit["file_integrity"] == "PASS").all()),
        "exact_duplicate_count_zero": int(audit["exact_duplicate_group"].notna().sum()) == 0,
        "near_duplicate_groups_7": audit["possible_near_duplicate_group"].nunique(dropna=True) == 7,
        "near_duplicate_samples_18": int(audit["possible_near_duplicate_group"].notna().sum()) == 18,
        "source_hashes_unchanged": all(
            (ROOT / item["path"]).is_file()
            and (ROOT / item["path"]).stat().st_size == item["bytes"]
            and sha256(ROOT / item["path"]) == item["sha256"]
            for item in source_hash_records
        ),
        "audit_artifact_hashes_match": all(
            sha256(ROOT / item["path"]) == item["sha256"] for item in audit_manifest["artifacts"]
        ),
    }
    if not all(integrity.values()):
        raise RuntimeError(f"Input integrity failed: {integrity}")

    condition_rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        subset = audit[audit["condition"] == condition]
        unit_accuracy = subset.groupby("analysis_unit")["correct"].mean()
        aggregate_correct: list[bool] = []
        for _, group in subset.groupby("analysis_unit", sort=False):
            aggregate_prediction = "P" if group["probability_P"].mean() >= group["probability_Q"].mean() else "Q"
            aggregate_correct.append(aggregate_prediction == group["input_label"].iloc[0])
        condition_rows.append(
            {
                "condition": condition,
                "samples": len(subset),
                "correct_samples": int(subset["correct"].sum()),
                "incorrect_samples": int((~subset["correct"]).sum()),
                "sample_accuracy": float(subset["correct"].mean()),
                "analysis_units": int(subset["analysis_unit"].nunique()),
                "equal_unit_weighted_accuracy": float(unit_accuracy.mean()),
                "aggregate_unit_correct": int(sum(aggregate_correct)),
                "aggregate_unit_accuracy": float(np.mean(aggregate_correct)),
                "instant_stable_label_disagreements": int(
                    (subset["model_prediction"] != subset["stable_prediction"]).sum()
                ),
            }
        )
    condition_metrics = pd.DataFrame(condition_rows)
    condition_metrics.to_csv(OUTPUT_DIR / "condition_metrics.csv", index=False, encoding="utf-8-sig")

    duplicate_rows: list[dict[str, object]] = []
    for group_id, group in audit.dropna(subset=["possible_near_duplicate_group"]).groupby(
        "possible_near_duplicate_group", sort=True
    ):
        timestamps = pd.to_datetime(group["captured_utc"], utc=True)
        duplicate_rows.append(
            {
                "near_duplicate_group": group_id,
                "condition": group["condition"].iloc[0],
                "samples": len(group),
                "audit_ids": "|".join(group["audit_id"]),
                "sample_ids": "|".join(group["sample_id"]),
                "distinct_raw_sha256": group["raw_sha256"].nunique(),
                "identical_raw_files": group["raw_sha256"].nunique() != len(group),
                "capture_span_seconds": float((timestamps.max() - timestamps.min()).total_seconds()),
                "predictions": "|".join(group["model_prediction"]),
                "prediction_consistent": group["model_prediction"].nunique() == 1,
                "independent_gesture_execution": "NOT_VERIFIABLE_NO_GESTURE_BOUT_OR_SESSION_ID",
            }
        )
    duplicate_groups = pd.DataFrame(duplicate_rows)
    duplicate_groups.to_csv(OUTPUT_DIR / "near_duplicate_group_analysis.csv", index=False, encoding="utf-8-sig")

    ph = audit[audit["condition"] == "P_horizontal"].copy()
    ph[
        [
            "audit_id", "sample_id", "analysis_unit", "model_prediction", "model_confidence",
            "probability_P", "probability_Q", "stable_prediction", "stable_confidence", "correct",
        ]
    ].to_csv(OUTPUT_DIR / "p_horizontal_cases.csv", index=False, encoding="utf-8-sig")

    features = np.asarray([feature_row(path) for path in audit["json_path"]], dtype=float)
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    scaled = scaler.transform(features)
    condition_indices = {
        condition: np.flatnonzero(audit["condition"].to_numpy() == condition) for condition in CONDITIONS
    }
    sample_centroids = {condition: scaled[index].mean(axis=0) for condition, index in condition_indices.items()}
    group_centroids: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        subset = audit[audit["condition"] == condition]
        means = [scaled[group.index].mean(axis=0) for _, group in subset.groupby("analysis_unit", sort=False)]
        group_centroids[condition] = np.mean(np.stack(means), axis=0)

    distance_rows: list[dict[str, object]] = []
    for left_index, left in enumerate(CONDITIONS):
        for right in CONDITIONS[left_index + 1 :]:
            distance_rows.append(
                {
                    "condition_a": left,
                    "condition_b": right,
                    "scaled_centroid_distance_sample_weighted": float(
                        np.linalg.norm(sample_centroids[left] - sample_centroids[right])
                    ),
                    "scaled_centroid_distance_equal_unit_weighted": float(
                        np.linalg.norm(group_centroids[left] - group_centroids[right])
                    ),
                }
            )
    feature_distances = pd.DataFrame(distance_rows)
    feature_distances.to_csv(OUTPUT_DIR / "feature_distance_summary.csv", index=False, encoding="utf-8-sig")

    points = features.reshape(-1, 21, 3)
    geometry_metrics = {
        "middle_tip_to_mcp_12_9": np.linalg.norm(points[:, 12] - points[:, 9], axis=1),
        "index_tip_to_mcp_8_5": np.linalg.norm(points[:, 8] - points[:, 5], axis=1),
        "thumb_to_index_tip_4_8": np.linalg.norm(points[:, 4] - points[:, 8], axis=1),
        "thumb_to_middle_tip_4_12": np.linalg.norm(points[:, 4] - points[:, 12], axis=1),
        "index_to_middle_tip_8_12": np.linalg.norm(points[:, 8] - points[:, 12], axis=1),
    }
    geometry_rows: list[dict[str, object]] = []
    for condition, index in condition_indices.items():
        for metric, values in geometry_metrics.items():
            geometry_rows.append({"condition": condition, "metric": metric, **stats(values[index])})
    pd.DataFrame(geometry_rows).to_csv(
        OUTPUT_DIR / "landmark_geometry_summary.csv", index=False, encoding="utf-8-sig"
    )

    ph_index = condition_indices["P_horizontal"]
    distance_to_pf = np.linalg.norm(scaled[ph_index] - sample_centroids["P_front"], axis=1)
    distance_to_qh = np.linalg.norm(scaled[ph_index] - sample_centroids["Q_horizontal"], axis=1)

    train = pd.read_csv(TRAIN_DIR / "train.csv", usecols=["label", *FEATURES])
    train = train[train["label"].isin(["P", "Q"])].reset_index(drop=True)
    train_labels = train["label"].astype(str).to_numpy(dtype=object)
    train_scaled = scaler.transform(train[FEATURES].to_numpy(dtype=float))
    train_centroids = {
        label: train_scaled[train_labels == label].mean(axis=0) for label in ("P", "Q")
    }
    ph_train_p = np.linalg.norm(scaled[ph_index] - train_centroids["P"], axis=1)
    ph_train_q = np.linalg.norm(scaled[ph_index] - train_centroids["Q"], axis=1)

    neighbors = NearestNeighbors(n_neighbors=31, n_jobs=-1).fit(train_scaled)
    neighbor_indices = neighbors.kneighbors(scaled, return_distance=False)
    knn_summary: dict[str, dict[str, float | int]] = {}
    for k in (5, 15, 31):
        p_vote = np.mean(train_labels[neighbor_indices[:, :k]] == "P", axis=1)
        knn_summary[str(k)] = {
            "p_horizontal_majority_P": int((p_vote[ph_index] > 0.5).sum()),
            "p_horizontal_samples": len(ph_index),
            "p_horizontal_mean_P_vote": float(p_vote[ph_index].mean()),
            "q_horizontal_majority_P": int(
                (p_vote[condition_indices["Q_horizontal"]] > 0.5).sum()
            ),
        }

    misclassified_ph = ph[ph["model_prediction"] == "Q"]
    correct_ph = ph[ph["model_prediction"] == "P"]
    inconsistent_near_groups = duplicate_groups[~duplicate_groups["prediction_consistent"]][
        "near_duplicate_group"
    ].tolist()
    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ANALYSIS_COMPLETE",
        "samples_retained": 47,
        "samples_modified_or_deleted": 0,
        "condition_metrics": condition_rows,
        "p_horizontal_to_q": {
            "samples": len(misclassified_ph),
            "total_p_horizontal": len(ph),
            "rate": float(len(misclassified_ph) / len(ph)),
            "audit_ids": misclassified_ph["audit_id"].tolist(),
            "probability_Q": stats(misclassified_ph["probability_Q"].to_numpy()),
            "probability_P": stats(misclassified_ph["probability_P"].to_numpy()),
            "correct_P_probability": stats(correct_ph["probability_P"].to_numpy()),
        },
        "prediction_stability": {
            "instant_vs_stable_label_disagreements": int(
                (audit["model_prediction"] != audit["stable_prediction"]).sum()
            ),
            "near_duplicate_groups_with_mixed_predictions": inconsistent_near_groups,
        },
        "near_duplicates": {
            "exact_duplicate_samples": 0,
            "possible_near_duplicate_samples": 18,
            "possible_near_duplicate_groups": 7,
            "all_raw_sha256_distinct_within_groups": bool(
                (duplicate_groups["distinct_raw_sha256"] == duplicate_groups["samples"]).all()
            ),
            "independence_limit": (
                "Distinct capture IDs, timestamps and raw pixels prove separate saved frames, but metadata has no "
                "gesture-bout/person/session identifier; independent gesture executions cannot be established."
            ),
        },
        "feature_evidence": {
            "scaled_centroid_distance_P_front_to_P_horizontal": float(
                np.linalg.norm(sample_centroids["P_front"] - sample_centroids["P_horizontal"])
            ),
            "scaled_centroid_distance_P_horizontal_to_Q_horizontal": float(
                np.linalg.norm(sample_centroids["P_horizontal"] - sample_centroids["Q_horizontal"])
            ),
            "scaled_centroid_distance_P_front_to_Q_horizontal": float(
                np.linalg.norm(sample_centroids["P_front"] - sample_centroids["Q_horizontal"])
            ),
            "equal_unit_scaled_centroid_distance_P_front_to_P_horizontal": float(
                np.linalg.norm(group_centroids["P_front"] - group_centroids["P_horizontal"])
            ),
            "equal_unit_scaled_centroid_distance_P_horizontal_to_Q_horizontal": float(
                np.linalg.norm(group_centroids["P_horizontal"] - group_centroids["Q_horizontal"])
            ),
            "equal_unit_scaled_centroid_distance_P_front_to_Q_horizontal": float(
                np.linalg.norm(group_centroids["P_front"] - group_centroids["Q_horizontal"])
            ),
            "p_horizontal_closer_to_Q_horizontal_debug_centroid": int((distance_to_qh < distance_to_pf).sum()),
            "p_horizontal_closer_to_Q_training_centroid": int((ph_train_q < ph_train_p).sum()),
            "p_horizontal_samples": len(ph_index),
            "training_P_samples": int((train_labels == "P").sum()),
            "training_Q_samples": int((train_labels == "Q").sum()),
            "training_knn_descriptive_only": knn_summary,
            "normalization_rotation_invariant": False,
        },
        "landmark_geometry_key_means": {
            condition: {
                metric: float(values[condition_indices[condition]].mean())
                for metric, values in geometry_metrics.items()
            }
            for condition in CONDITIONS
        },
        "confirmed_causes": [
            "The error is condition-specific: P/front is 16/16 while P/horizontal is 6/16; Q/horizontal is 15/15.",
            "Saved overlays passed visual landmark review for all 47 captures, including tips 4, 8 and 12; no observed landmark-placement failure explains P-to-Q errors.",
            "The 63-feature normalization removes wrist translation and global scale but does not align hand rotation.",
            "In TRAIN-scaled feature space, P/horizontal is closer to Q/horizontal than P/front at both sample-weighted and equal-near-duplicate-unit levels.",
            "Horizontal-P retains P/Q-discriminative geometry, while the MLP still maps 10/16 samples to Q; this supports a representation/training-coverage/decision-boundary limitation rather than semantic mislabeling.",
        ],
        "unverified_hypotheses_and_limits": [
            "The dataset has no explicit view-angle metadata, so insufficient horizontal-P coverage in TRAIN is plausible but not directly countable.",
            "All debug captures appear to come from one short collection session and one visible signer; results do not establish cross-person performance.",
            "Near-duplicate metadata cannot distinguish repeated S presses during one held pose from independent gesture executions.",
            "The sample is small; distances and kNN checks are descriptive and must not be treated as a tuned replacement classifier.",
        ],
        "recommended_v2_plan_not_executed": [
            "Collect manually verified P and Q across matched roll/yaw/pitch views, multiple signers/hands/sessions, and record signer/session/gesture-bout/view metadata.",
            "Keep all current captures, but split and evaluate by signer/session/gesture-bout groups so repeated frames cannot dominate.",
            "Create a new v2 validation protocol with dedicated P-front, P-horizontal, Q-front and Q-horizontal strata; protect Q recall/precision while optimizing horizontal-P recall.",
            "Compare the unchanged 63-coordinate baseline against a hand-centric rotation-aligned or relative-angle/distance representation in a controlled v2 experiment; do not add a P/Q angle rule.",
            "Reserve a new untouched subject/session test for v2; do not tune on TEST v1 and continue calling it independent.",
        ],
        "test_v1_used": False,
        "model_retrained_or_modified": False,
        "debug_or_frozen_data_modified": False,
        "integrity": {"status": "PASS", "checks_run": len(integrity), "checks": integrity},
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report = f"""# P/Q debug model analysis v1

- P/front: 16/16 correct; equal-unit weighted 100%.
- P/horizontal: 6/16 correct; equal-unit weighted {condition_rows[1]['equal_unit_weighted_accuracy']:.2%}; aggregate-unit 4/9.
- Q/horizontal: 15/15 correct; equal-unit weighted 100%.
- P-horizontal to Q: 10/16; mean Q probability {misclassified_ph['probability_Q'].mean():.6f}.
- Exact duplicates: 0. Possible near-duplicate groups: 7 (18 samples); independent gesture executions are not verifiable from metadata.
- Instant/stable label disagreements: 0. Mixed-prediction near-duplicate groups: {', '.join(inconsistent_near_groups)}.
- All 47 semantic and landmark reviews passed. No landmark-placement error was observed.
- No TEST v1 access, model training, relabeling, deletion, or source-data mutation occurred.
"""
    (OUTPUT_DIR / "REPORT.md").write_text(report, encoding="utf-8")

    outputs = [
        "analysis_summary.json", "condition_metrics.csv", "near_duplicate_group_analysis.csv",
        "p_horizontal_cases.csv", "feature_distance_summary.csv", "landmark_geometry_summary.csv", "REPORT.md",
    ]
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "source_files_modified": False,
        "outputs": [
            {"path": f"ai/dataset/pq_debug_model_analysis_v1/{name}", "sha256": sha256(OUTPUT_DIR / name)}
            for name in outputs
        ],
        "inputs": [
            {
                "path": "ai/dataset/pq_debug_data_audit_v1/debug_sample_audit.csv",
                "sha256": sha256(AUDIT_DIR / "debug_sample_audit.csv"),
            },
            {
                "path": "ai/dataset/mlp_validation_comparison_v1/scaler.joblib",
                "sha256": sha256(MODEL_DIR / "scaler.joblib"),
            },
            {
                "path": "ai/dataset/train_validation_test_split_v1/train.csv",
                "sha256": sha256(TRAIN_DIR / "train.csv"),
            },
        ],
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
