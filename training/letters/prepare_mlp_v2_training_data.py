from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "ai" / "preprocessing") not in sys.path:
    sys.path.insert(0, str(ROOT / "ai" / "preprocessing"))

from landmark_utils import normalize_landmarks  # noqa: E402
TRAIN_DIR = ROOT / "ai" / "dataset" / "train_validation_test_split_v1"
MODEL_DIR = ROOT / "ai" / "dataset" / "mlp_validation_comparison_v1"
PQ_AUDIT_DIR = ROOT / "ai" / "dataset" / "pq_debug_data_audit_v1"
UV_AUDIT_DIR = ROOT / "ai" / "dataset" / "uv_debug_data_audit_v1"
OUT_DIR = ROOT / "ai" / "dataset" / "mlp_v2_training_data_v1"

FEATURES = [f"f{i}" for i in range(1, 64)]
LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
TRAIN_COLUMNS = ["source", "file", "label", "handedness", *FEATURES]

INPUTS = {
    "train": TRAIN_DIR / "train.csv",
    "validation": TRAIN_DIR / "validation.csv",
    "test": TRAIN_DIR / "test.csv",
    "split_assignments": TRAIN_DIR / "split_assignments.csv",
    "group_assignments": TRAIN_DIR / "group_assignments.csv",
    "split_manifest": TRAIN_DIR / "split_manifest.json",
    "split_integrity": TRAIN_DIR / "integrity_report.json",
    "training_config": MODEL_DIR / "training_config.json",
    "pq_audit": PQ_AUDIT_DIR / "debug_sample_audit.csv",
    "pq_manifest": PQ_AUDIT_DIR / "manifest.json",
    "pq_integrity": PQ_AUDIT_DIR / "integrity_report.json",
    "pq_input_hashes": PQ_AUDIT_DIR / "input_hashes.json",
    "uv_audit": UV_AUDIT_DIR / "uv_debug_sample_audit.csv",
    "uv_manifest": UV_AUDIT_DIR / "manifest.json",
    "uv_integrity": UV_AUDIT_DIR / "integrity_report.json",
    "uv_input_hashes": UV_AUDIT_DIR / "input_hashes.json",
    "normalizer": ROOT / "ai" / "preprocessing" / "landmark_utils.py",
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_write(path: Path, obj) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def resolve_project_path(value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def strict_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def source_hash_records(path: Path) -> list[dict]:
    data = json_load(path)
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = None
        for key in ("files", "input_files", "source_files", "hashes"):
            if isinstance(data.get(key), list):
                records = data[key]
                break
            if isinstance(data.get(key), dict):
                records = [
                    {"path": name, "sha256": digest}
                    for name, digest in data[key].items()
                ]
                break
        if records is None and all(isinstance(v, str) for v in data.values()):
            records = [{"path": name, "sha256": digest} for name, digest in data.items()]
        if records is None:
            raise ValueError(f"Unsupported hash inventory schema: {path}")
    else:
        raise ValueError(f"Unsupported hash inventory schema: {path}")

    normalized = []
    for row in records:
        path_value = row.get("path") or row.get("file") or row.get("relative_path")
        digest = row.get("sha256") or row.get("hash")
        if path_value and digest:
            normalized.append({"path": str(path_value), "sha256": str(digest).lower()})
    if not normalized:
        raise ValueError(f"No file hashes found in {path}")
    return normalized


def verify_source_hash_inventory(path: Path) -> tuple[int, list[str]]:
    records = source_hash_records(path)
    errors = []
    for row in records:
        target = resolve_project_path(row["path"])
        if not target.is_file():
            errors.append(f"missing:{row['path']}")
        elif sha256(target) != row["sha256"]:
            errors.append(f"hash:{row['path']}")
    return len(records), errors


def feature_hashes(matrix: np.ndarray) -> list[str]:
    values = np.asarray(matrix, dtype="<f8")
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in values]


def conservative_group(origin: str, label: str, view: str, captured_utc: str) -> str:
    date = str(captured_utc)[:10].replace("-", "")
    return f"DBG-{origin}-{label}-{view}-{date}"


def load_debug(audit_path: Path, origin: str) -> tuple[pd.DataFrame, list[dict], list[str]]:
    audit = pd.read_csv(audit_path, dtype=str, keep_default_na=False)
    expected_labels = {"P", "Q"} if origin == "PQ" else {"U", "V"}
    errors: list[str] = []
    if audit["sample_id"].duplicated().any():
        errors.append(f"{origin}:duplicate_sample_id")
    if not set(audit["input_label"]).issubset(expected_labels):
        errors.append(f"{origin}:unexpected_label")

    accepted = audit.loc[
        (audit["file_integrity"] == "PASS")
        & (audit["label_review"] == "LABEL_CONFIRMED")
        & (audit["landmark_review"] == "LANDMARK_CONFIRMED")
    ].copy()

    rows = []
    metadata = []
    for _, item in accepted.sort_values(["input_label", "input_view", "sample_id"]).iterrows():
        json_path = resolve_project_path(item["json_path"])
        raw_path = resolve_project_path(item["raw_image_path"])
        overlay_path = resolve_project_path(item["overlay_image_path"])
        if not json_path.is_file() or not raw_path.is_file() or not overlay_path.is_file():
            errors.append(f"{origin}:{item['sample_id']}:missing_artifact")
            continue
        payload = json_load(json_path)
        if payload.get("capture_id") != item["sample_id"]:
            errors.append(f"{origin}:{item['sample_id']}:capture_id_mismatch")
        if payload.get("manual_label") != item["input_label"]:
            errors.append(f"{origin}:{item['sample_id']}:label_mismatch")
        if payload.get("manual_view") != item["input_view"]:
            errors.append(f"{origin}:{item['sample_id']}:view_mismatch")
        if payload.get("normalization_function") != "AI/landmark_utils.py::normalize_landmarks":
            errors.append(f"{origin}:{item['sample_id']}:normalizer_mismatch")

        stored = payload.get("features", {})
        if list(stored.keys()) != FEATURES:
            errors.append(f"{origin}:{item['sample_id']}:feature_order")
            continue
        vector = np.asarray([stored[name] for name in FEATURES], dtype=np.float64)
        if vector.shape != (63,) or not np.isfinite(vector).all():
            errors.append(f"{origin}:{item['sample_id']}:invalid_features")
            continue
        raw_landmarks = payload.get("raw_landmarks", [])
        if len(raw_landmarks) != 21 or [point.get("index") for point in raw_landmarks] != list(range(21)):
            errors.append(f"{origin}:{item['sample_id']}:invalid_landmarks")
            continue
        landmark_objects = [
            SimpleNamespace(x=float(point["x"]), y=float(point["y"]), z=float(point["z"]))
            for point in raw_landmarks
        ]
        recomputed = np.asarray(normalize_landmarks(landmark_objects), dtype=np.float64)
        if not np.allclose(recomputed, vector, rtol=0.0, atol=1e-12):
            errors.append(f"{origin}:{item['sample_id']}:feature_recompute")
        probabilities = payload.get("probabilities", {})
        if list(probabilities.keys()) != LABELS:
            errors.append(f"{origin}:{item['sample_id']}:probability_labels")
        elif not math.isclose(sum(float(v) for v in probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            errors.append(f"{origin}:{item['sample_id']}:probability_sum")

        row = {
            "source": f"camera_debug_{origin.lower()}",
            "file": rel(raw_path),
            "label": item["input_label"],
            "handedness": item["handedness"],
        }
        row.update({name: float(stored[name]) for name in FEATURES})
        rows.append(row)
        near_col = "possible_near_duplicate_group" if origin == "PQ" else "near_duplicate_group"
        collection = item.get("collection_dependence_group", "")
        group_id = conservative_group(origin, item["input_label"], item["input_view"], item["captured_utc"])
        metadata.append(
            {
                "source": row["source"],
                "file": row["file"],
                "label": row["label"],
                "handedness": row["handedness"],
                "origin": f"DEBUG_{origin}",
                "original_split": "DEBUG_OBSERVED",
                "group_id": group_id,
                "input_view": item["input_view"],
                "near_duplicate_group": item.get(near_col, ""),
                "collection_dependence_group": collection or group_id,
                "sample_id": item["sample_id"],
                "audit_id": item["audit_id"],
                "audit_status": "LABEL_AND_LANDMARK_CONFIRMED",
                "json_path": rel(json_path),
                "raw_image_path": rel(raw_path),
                "overlay_image_path": rel(overlay_path),
                "observed_during_v1_error_analysis": True,
            }
        )
    return pd.DataFrame(rows, columns=TRAIN_COLUMNS), metadata, errors


def make_weights(metadata: pd.DataFrame) -> pd.DataFrame:
    table = metadata[["v2_row_id", "label", "group_id", "origin", "input_view"]].copy()
    table["group_size"] = table.groupby("group_id")["group_id"].transform("size").astype(int)
    table["weight_uniform"] = 1.0
    n_rows = len(table)
    class_target = n_rows / len(LABELS)

    group_counts = table.groupby("label")["group_id"].nunique().to_dict()
    table["weight_group_class_balanced"] = [
        n_rows / (len(LABELS) * group_counts[label] * size)
        for label, size in zip(table["label"], table["group_size"])
    ]

    debug_fraction_5 = {
        ("P", "front"): 0.01,
        ("P", "horizontal"): 0.04,
        ("Q", "horizontal"): 0.05,
        ("U", "front"): 0.05,
        ("V", "front"): 0.05,
    }
    debug_fraction_10 = {
        ("P", "front"): 0.02,
        ("P", "horizontal"): 0.08,
        ("Q", "horizontal"): 0.10,
        ("U", "front"): 0.10,
        ("V", "front"): 0.10,
    }

    def budget_weights(fractions: dict[tuple[str, str], float]) -> np.ndarray:
        result = np.zeros(n_rows, dtype=np.float64)
        for label in LABELS:
            label_mask = table["label"].eq(label)
            legacy_mask = label_mask & table["origin"].eq("TRAIN_V1")
            debug_masks = []
            debug_total = 0.0
            for (target_label, view), fraction in fractions.items():
                if target_label != label:
                    continue
                mask = label_mask & table["origin"].str.startswith("DEBUG_") & table["input_view"].eq(view)
                if mask.any():
                    debug_masks.append((mask, fraction))
                    debug_total += fraction
            legacy_groups = table.loc[legacy_mask, "group_id"].nunique()
            legacy_share = 1.0 - debug_total
            if legacy_groups <= 0 or legacy_share <= 0:
                raise ValueError(f"Invalid legacy budget for class {label}")
            result[legacy_mask.to_numpy()] = [
                class_target * legacy_share / (legacy_groups * size)
                for size in table.loc[legacy_mask, "group_size"]
            ]
            for mask, fraction in debug_masks:
                debug_groups = table.loc[mask, "group_id"].nunique()
                result[mask.to_numpy()] = [
                    class_target * fraction / (debug_groups * size)
                    for size in table.loc[mask, "group_size"]
                ]
        return result

    table["weight_debug_budget_5pct"] = budget_weights(debug_fraction_5)
    table["weight_debug_budget_10pct_recommended"] = budget_weights(debug_fraction_10)
    return table


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [rel(path) for path in INPUTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    config = json_load(INPUTS["training_config"])
    snapshots = config["input_snapshots"]
    snapshot_checks = {}
    for name in ("train", "validation", "test", "split_assignments", "group_assignments", "split_manifest", "split_integrity"):
        path = INPUTS[name]
        expected = snapshots.get(rel(path))
        actual = sha256(path)
        snapshot_checks[name] = {"path": rel(path), "expected_sha256": expected, "actual_sha256": actual, "pass": actual == expected}

    pq_integrity = json_load(INPUTS["pq_integrity"])
    uv_integrity = json_load(INPUTS["uv_integrity"])
    pq_source_count, pq_source_errors = verify_source_hash_inventory(INPUTS["pq_input_hashes"])
    uv_source_count, uv_source_errors = verify_source_hash_inventory(INPUTS["uv_input_hashes"])

    train = pd.read_csv(INPUTS["train"])
    assignments = pd.read_csv(INPUTS["split_assignments"], dtype=str, keep_default_na=False)
    assignments = assignments.loc[assignments["split"].eq("TRAIN"), ["file", "label", "group_id"]]
    if train.columns.tolist() != TRAIN_COLUMNS:
        raise ValueError("TRAIN v1 schema/order does not match the required 63-feature schema")
    if len(train) != 62591:
        raise ValueError(f"Unexpected TRAIN v1 row count: {len(train)}")
    if set(train["label"]) != set(LABELS):
        raise ValueError("TRAIN v1 label set mismatch")
    if train["file"].duplicated().any():
        raise ValueError("TRAIN v1 contains duplicate file values")
    train_features = train[FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(train_features).all():
        raise ValueError("TRAIN v1 contains non-finite feature values")

    legacy_meta = train[["source", "file", "label", "handedness"]].merge(
        assignments, on=["file", "label"], how="left", validate="one_to_one"
    )
    if legacy_meta["group_id"].eq("").any() or legacy_meta["group_id"].isna().any():
        raise ValueError("Missing TRAIN group_id")
    legacy_meta = legacy_meta.assign(
        origin="TRAIN_V1",
        original_split="TRAIN",
        input_view="",
        near_duplicate_group="",
        collection_dependence_group=legacy_meta["group_id"],
        sample_id="",
        audit_id="",
        audit_status="FROZEN_V3_ACCEPTED",
        json_path="",
        raw_image_path=legacy_meta["file"],
        overlay_image_path="",
        observed_during_v1_error_analysis=False,
    )

    pq, pq_meta, pq_errors = load_debug(INPUTS["pq_audit"], "PQ")
    uv, uv_meta, uv_errors = load_debug(INPUTS["uv_audit"], "UV")
    debug = pd.concat([pq, uv], ignore_index=True)
    debug_meta = pd.DataFrame(pq_meta + uv_meta)
    expected_debug_counts = {"P": 32, "Q": 15, "U": 15, "V": 15}
    actual_debug_counts = debug["label"].value_counts().sort_index().to_dict()

    train_file_collisions = sorted(set(debug["file"]) & set(train["file"]))
    debug_file_duplicates = debug.loc[debug["file"].duplicated(keep=False), "file"].tolist()
    debug_sample_duplicates = debug_meta.loc[debug_meta["sample_id"].duplicated(keep=False), "sample_id"].tolist()

    train_hashes = feature_hashes(train_features)
    debug_features = debug[FEATURES].to_numpy(dtype=np.float64)
    debug_hashes = feature_hashes(debug_features)
    train_hash_map: dict[str, list[int]] = defaultdict(list)
    for index, digest in enumerate(train_hashes):
        train_hash_map[digest].append(index)
    exact_feature_matches = []
    for index, digest in enumerate(debug_hashes):
        for train_index in train_hash_map.get(digest, []):
            exact_feature_matches.append(
                {
                    "debug_file": debug.iloc[index]["file"],
                    "debug_label": debug.iloc[index]["label"],
                    "train_file": train.iloc[train_index]["file"],
                    "train_label": train.iloc[train_index]["label"],
                }
            )
    debug_exact_duplicate_count = len(debug_hashes) - len(set(debug_hashes))

    nearest_rows = []
    for label in sorted(set(debug["label"])):
        train_mask = train["label"].eq(label).to_numpy()
        debug_mask = debug["label"].eq(label).to_numpy()
        model = NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="auto")
        model.fit(train_features[train_mask])
        distances, indices = model.kneighbors(debug_features[debug_mask])
        train_indices = np.flatnonzero(train_mask)
        for debug_index, distance, local_index in zip(np.flatnonzero(debug_mask), distances[:, 0], indices[:, 0]):
            train_index = int(train_indices[local_index])
            nearest_rows.append(
                {
                    "debug_file": debug.iloc[debug_index]["file"],
                    "debug_label": label,
                    "nearest_train_file_same_label": train.iloc[train_index]["file"],
                    "nearest_feature_distance_same_label": float(distance),
                    "exact_feature_match": bool(distance == 0.0),
                }
            )

    train_v2 = pd.concat([train, debug], ignore_index=True)
    metadata = pd.concat([legacy_meta, debug_meta], ignore_index=True)
    metadata.insert(0, "v2_row_id", [f"V2-{index:06d}" for index in range(1, len(metadata) + 1)])
    if not train_v2[["source", "file", "label", "handedness"]].reset_index(drop=True).equals(
        metadata[["source", "file", "label", "handedness"]].reset_index(drop=True)
    ):
        raise ValueError("TRAIN v2 and metadata row alignment mismatch")
    weights = make_weights(metadata)

    additional = pd.concat(
        [
            metadata.tail(len(debug)).reset_index(drop=True),
            debug[FEATURES].reset_index(drop=True),
        ],
        axis=1,
    )
    nearest = pd.DataFrame(nearest_rows).sort_values(["debug_label", "debug_file"])
    duplicates = pd.DataFrame(exact_feature_matches, columns=["debug_file", "debug_label", "train_file", "train_label"])

    output_paths = {
        "train_v2": OUT_DIR / "train_v2.csv",
        "metadata": OUT_DIR / "train_v2_sample_metadata.csv",
        "weights": OUT_DIR / "train_v2_weights.csv",
        "additional": OUT_DIR / "additional_debug_samples.csv",
        "nearest": OUT_DIR / "debug_to_train_nearest_distances.csv",
        "exact_matches": OUT_DIR / "exact_feature_matches_with_train_v1.csv",
        "recommendation": OUT_DIR / "training_recommendation.json",
        "integrity": OUT_DIR / "integrity_report.json",
        "manifest": OUT_DIR / "manifest.json",
        "readme": OUT_DIR / "README.md",
    }
    train_v2.to_csv(output_paths["train_v2"], index=False, float_format="%.17g")
    metadata.to_csv(output_paths["metadata"], index=False)
    weights.to_csv(output_paths["weights"], index=False, float_format="%.17g")
    additional.to_csv(output_paths["additional"], index=False, float_format="%.17g")
    nearest.to_csv(output_paths["nearest"], index=False, float_format="%.17g")
    duplicates.to_csv(output_paths["exact_matches"], index=False)

    class_counts = train_v2["label"].value_counts().sort_index().to_dict()
    debug_group_counts = metadata.loc[metadata["origin"].ne("TRAIN_V1")].groupby(
        ["label", "input_view", "group_id"]
    ).size().reset_index(name="sample_count").to_dict(orient="records")
    near_counts = {
        "pq_samples_in_candidate_groups": int((debug_meta.loc[debug_meta["origin"].eq("DEBUG_PQ"), "near_duplicate_group"] != "").sum()),
        "pq_candidate_groups": int(debug_meta.loc[(debug_meta["origin"].eq("DEBUG_PQ")) & (debug_meta["near_duplicate_group"] != ""), "near_duplicate_group"].nunique()),
        "uv_samples_in_candidate_groups": int((debug_meta.loc[debug_meta["origin"].eq("DEBUG_UV"), "near_duplicate_group"] != "").sum()),
        "uv_candidate_groups": int(debug_meta.loc[(debug_meta["origin"].eq("DEBUG_UV")) & (debug_meta["near_duplicate_group"] != ""), "near_duplicate_group"].nunique()),
    }

    recommendation = {
        "status": "PREPARED_NOT_TRAINED",
        "primary_weight_column": "weight_debug_budget_10pct_recommended",
        "weighting": {
            "principle": "Equal total class mass; equal mass per legacy group; cap observed debug data by class and divide each condition budget across every frame in its conservative session-condition group.",
            "recommended_debug_class_mass": {
                "P_front": 0.02,
                "P_horizontal": 0.08,
                "Q_horizontal": 0.10,
                "U_front": 0.10,
                "V_front": 0.10,
            },
            "legacy_mass_for_affected_classes": 0.90,
            "lower_exposure_comparison_column": "weight_debug_budget_5pct",
            "controls": ["weight_uniform", "weight_group_class_balanced"],
            "near_duplicate_control": "All frames from one label/view/date collection are one conservative group; pHash candidate group is retained separately. Weights are divided by conservative group size.",
        },
        "development_protocol": {
            "validation": rel(INPUTS["validation"]),
            "validation_policy": "Keep VALIDATION v1 unchanged for 24-class monitoring and model selection safeguards.",
            "test_policy": "Do not use TEST v1 for training, checkpoint selection, hyperparameter tuning, or model-v2 selection.",
            "debug_policy": "The 77 debug samples were observed during v1 error analysis and are now training data; performance on them is diagnostic/training-fit only, never independent evidence.",
            "experimental_arms": [
                "v1 architecture with TRAIN v2 and weight_uniform",
                "v1 architecture with TRAIN v2 and weight_debug_budget_5pct",
                "v1 architecture with TRAIN v2 and weight_debug_budget_10pct_recommended",
            ],
            "guard_metrics": [
                "VALIDATION v1 macro-F1 and per-class recall",
                "Q, U, W recall",
                "P/K and Q/G confusion",
                "new unseen-camera challenge: P front/horizontal, Q front/horizontal, U front, V front, W front",
            ],
        },
        "pre_training_limitations": [
            "No signer_id, session_id, or gesture_bout_id proves independent performances in the 77 debug samples.",
            "Debug data come from short observed collection sessions and cannot provide an independent model-v2 evaluation.",
            "No Q/front or U/V horizontal debug coverage is present.",
            "A new manually confirmed camera challenge from unseen gesture bouts (preferably new sessions/signers) is required before claiming improvement.",
        ],
        "training_started": False,
    }
    json_write(output_paths["recommendation"], recommendation)

    legacy_numeric_preserved = np.array_equal(
        train_v2.iloc[: len(train)][FEATURES].to_numpy(dtype=np.float64), train_features
    )
    appended_numeric_preserved = np.array_equal(
        train_v2.iloc[len(train) :][FEATURES].to_numpy(dtype=np.float64), debug_features
    )
    weight_sums = {
        column: float(weights[column].sum())
        for column in ["weight_uniform", "weight_group_class_balanced", "weight_debug_budget_5pct", "weight_debug_budget_10pct_recommended"]
    }
    checks = {
        "all_required_inputs_present": not missing,
        "frozen_split_snapshot_hashes_match": all(row["pass"] for row in snapshot_checks.values()),
        "pq_prior_audit_pass": pq_integrity.get("status") == "PASS",
        "uv_prior_audit_pass": uv_integrity.get("status") == "PASS",
        "pq_source_hashes_match": not pq_source_errors,
        "uv_source_hashes_match": not uv_source_errors,
        "train_v1_rows_62591": len(train) == 62591,
        "train_v1_schema_exact": train.columns.tolist() == TRAIN_COLUMNS,
        "debug_rows_77": len(debug) == 77,
        "debug_counts_expected": actual_debug_counts == expected_debug_counts,
        "debug_metadata_rows_77": len(debug_meta) == 77,
        "debug_validation_errors_zero": not (pq_errors or uv_errors),
        "debug_file_collisions_with_train_zero": not train_file_collisions,
        "debug_file_duplicates_zero": not debug_file_duplicates,
        "debug_sample_id_duplicates_zero": not debug_sample_duplicates,
        "debug_exact_feature_matches_with_train_zero": not exact_feature_matches,
        "debug_exact_feature_duplicates_zero": debug_exact_duplicate_count == 0,
        "train_v2_rows_62668": len(train_v2) == 62668,
        "train_v2_schema_exact": train_v2.columns.tolist() == TRAIN_COLUMNS,
        "train_v2_labels_24": set(train_v2["label"]) == set(LABELS),
        "train_v2_features_finite": bool(np.isfinite(train_v2[FEATURES].to_numpy(dtype=np.float64)).all()),
        "legacy_labels_and_features_unchanged": bool(legacy_numeric_preserved and train_v2.iloc[: len(train)]["label"].equals(train["label"])),
        "appended_labels_and_features_match_sources": bool(appended_numeric_preserved and train_v2.iloc[len(train) :]["label"].reset_index(drop=True).equals(debug["label"].reset_index(drop=True))),
        "metadata_rows_align": len(metadata) == len(train_v2),
        "all_group_ids_present": bool(metadata["group_id"].notna().all() and metadata["group_id"].ne("").all()),
        "debug_conservative_groups_5": metadata.loc[metadata["origin"].ne("TRAIN_V1"), "group_id"].nunique() == 5,
        "weights_rows_align": len(weights) == len(train_v2),
        "weights_finite_positive": bool(np.isfinite(weights.filter(like="weight_").to_numpy()).all() and (weights.filter(like="weight_") > 0).all().all()),
        "weight_columns_mean_one": all(math.isclose(value, len(train_v2), rel_tol=0.0, abs_tol=1e-6) for value in weight_sums.values()),
        "validation_and_test_not_parsed": True,
        "training_not_performed": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    integrity = {
        "status": status,
        "checks_run": len(checks),
        "checks": checks,
        "snapshot_checks": snapshot_checks,
        "source_hash_verification": {
            "pq_files_verified": pq_source_count,
            "pq_errors": pq_source_errors,
            "uv_files_verified": uv_source_count,
            "uv_errors": uv_source_errors,
        },
        "debug_validation_errors": pq_errors + uv_errors,
        "collisions": {
            "file_with_train": train_file_collisions,
            "duplicate_debug_files": debug_file_duplicates,
            "duplicate_debug_sample_ids": debug_sample_duplicates,
            "exact_feature_matches_with_train": len(exact_feature_matches),
            "exact_feature_duplicates_within_debug": debug_exact_duplicate_count,
        },
        "weight_sums": weight_sums,
        "test_access": {
            "verification": "SHA256_ONLY",
            "csv_opened_or_parsed": False,
            "used_for_training_or_selection": False,
        },
    }
    json_write(output_paths["integrity"], integrity)

    readme = """# MLP v2 training-data preparation\n\nThis directory appends only audit-confirmed P/Q and U/V camera-debug rows to frozen TRAIN v1. It does not modify TRAIN/VALIDATION/TEST v1, does not fit a scaler, and does not train a model.\n\n- `train_v2.csv`: training-compatible 67-column dataset (`source,file,label,handedness,f1..f63`).\n- `train_v2_sample_metadata.csv`: provenance, conservative group IDs, view, audit and near-duplicate metadata.\n- `train_v2_weights.csv`: uniform, legacy group/class-balanced, and capped debug-budget weight options.\n- `additional_debug_samples.csv`: the 77 appended rows with provenance and features.\n- `training_recommendation.json`: experimental and evaluation protocol.\n\nThe 77 debug rows were already observed during v1 diagnosis. They are not an independent evaluation set.\n"""
    output_paths["readme"].write_text(readme, encoding="utf-8", newline="\n")

    output_hashes = {}
    for name, path in output_paths.items():
        if name == "manifest":
            continue
        output_hashes[rel(path)] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    input_hashes = {
        rel(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in INPUTS.values()
    }
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "TRAIN_V2_DATA_PREPARED" if status == "PASS" else "INTEGRITY_FAILED",
        "normalization": "AI/landmark_utils.py::normalize_landmarks (unchanged from model v1)",
        "feature_columns": FEATURES,
        "labels": LABELS,
        "counts": {
            "train_v1": len(train),
            "accepted_debug": len(debug),
            "accepted_by_label": actual_debug_counts,
            "train_v2": len(train_v2),
            "debug_conservative_groups": 5,
            "debug_groups": debug_group_counts,
            "near_duplicate_candidates": near_counts,
            "train_v2_by_label": class_counts,
        },
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "protected_inputs_modified": False,
        "validation_v1_changed": False,
        "test_v1_changed": False,
        "test_csv_parsed": False,
        "training_performed": False,
    }
    json_write(output_paths["manifest"], manifest)
    print(f"STATUS={manifest['status']}")
    print(f"TRAIN_V2_ROWS={len(train_v2)}")
    print(f"ACCEPTED_DEBUG={len(debug)}")
    print(f"INTEGRITY={status}")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
