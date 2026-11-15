"""Prepare a 24-class TRAIN set from TRAIN v1 plus audited U/V/W-boundary samples."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "ai" / "preprocessing") not in sys.path:
    sys.path.insert(0, str(ROOT / "ai" / "preprocessing"))

from landmark_utils import normalize_landmarks  # noqa: E402
SPLIT_DIR = ROOT / "ai/dataset/train_validation_test_split_v1"
V1_MODEL_DIR = ROOT / "ai/dataset/mlp_validation_comparison_v1"
OLD_AUDIT_DIR = ROOT / "ai/dataset/uv_debug_data_audit_v1"
NEW_AUDIT_DIR = ROOT / "ai/dataset/uv_v1_error_audit_v1"
OUT_DIR = ROOT / "ai/dataset/mlp_uvw_training_data_v1"
FEATURES = [f"f{i}" for i in range(1, 64)]
LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
TRAIN_COLUMNS = ["source", "file", "label", "handedness", *FEATURES]

PATHS = {
    "train": SPLIT_DIR / "train.csv",
    "validation": SPLIT_DIR / "validation.csv",
    "test": SPLIT_DIR / "test.csv",
    "split_assignments": SPLIT_DIR / "split_assignments.csv",
    "group_assignments": SPLIT_DIR / "group_assignments.csv",
    "split_manifest": SPLIT_DIR / "split_manifest.json",
    "split_integrity": SPLIT_DIR / "integrity_report.json",
    "v1_training_config": V1_MODEL_DIR / "training_config.json",
    "v1_training_manifest": V1_MODEL_DIR / "training_manifest.json",
    "old_audit_csv": OLD_AUDIT_DIR / "uv_debug_sample_audit.csv",
    "old_audit_manifest": OLD_AUDIT_DIR / "manifest.json",
    "old_audit_integrity": OLD_AUDIT_DIR / "integrity_report.json",
    "old_input_hashes": OLD_AUDIT_DIR / "input_hashes.json",
    "new_audit_csv": NEW_AUDIT_DIR / "v_error_sample_audit.csv",
    "new_audit_summary": NEW_AUDIT_DIR / "audit_summary.json",
    "new_audit_manifest": NEW_AUDIT_DIR / "manifest.json",
    "normalizer": ROOT / "ai" / "preprocessing" / "landmark_utils.py",
    "camera_app": ROOT / "app" / "inference" / "letter_inference.py",
}

LOW_BUDGETS = {
    "OLD_U": 0.02,
    "OLD_V": 0.01,
    "V_TO_U": 0.02,
    "V_TO_W": 0.02,
}
MEDIUM_BUDGETS = {
    "OLD_U": 0.05,
    "OLD_V": 0.02,
    "V_TO_U": 0.04,
    "V_TO_W": 0.04,
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
    ) + "\n",
                    encoding="utf-8")


def resolve(value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def verify_records(records: list[dict]) -> tuple[int, list[str]]:
    errors = []
    for item in records:
        path = resolve(item["path"])
        if not path.is_file():
            errors.append(f"missing:{item['path']}")
        elif "bytes" in item and path.stat().st_size != int(item["bytes"]):
            errors.append(f"size:{item['path']}")
        elif sha256(path) != item["sha256"]:
            errors.append(f"hash:{item['path']}")
    return len(records), errors


def verify_audits() -> dict:
    old_manifest = load_json(PATHS["old_audit_manifest"])
    old_integrity = load_json(PATHS["old_audit_integrity"])
    old_inputs = load_json(PATHS["old_input_hashes"])["files"]
    old_artifact_count, old_artifact_errors = verify_records(old_manifest["artifacts"])
    old_source_count, old_source_errors = verify_records(old_inputs)
    new_manifest = load_json(PATHS["new_audit_manifest"])
    new_summary = load_json(PATHS["new_audit_summary"])
    new_input_count, new_input_errors = verify_records(new_manifest["inputs"])
    new_reference_count, new_reference_errors = verify_records(new_manifest["reference_inputs"])
    new_output_count, new_output_errors = verify_records(new_manifest["outputs"])
    errors = (old_artifact_errors + old_source_errors + new_input_errors +
              new_reference_errors + new_output_errors)
    if old_manifest.get("status") != "PASS" or old_integrity.get("status") != "PASS":
        errors.append("old_audit_status")
    if (new_manifest.get("status") != "V1_V_ERROR_DATA_AUDIT_COMPLETE" or
            new_summary.get("status") != "V1_V_ERROR_DATA_AUDIT_COMPLETE" or
            new_summary.get("integrity", {}).get("status") != "PASS"):
        errors.append("new_audit_status")
    return {
        "status": "PASS" if not errors else "FAIL", "errors": errors,
        "old_source_files": old_source_count, "old_audit_artifacts": old_artifact_count,
        "new_source_files": new_input_count, "new_reference_files": new_reference_count,
        "new_audit_outputs": new_output_count,
    }


def validate_payload(audit_row: pd.Series, label: str) -> tuple[dict, np.ndarray, list[str]]:
    errors = []
    json_path = resolve(audit_row["json_path"])
    raw_path = resolve(audit_row["raw_image_path"])
    overlay_path = resolve(audit_row["overlay_image_path"])
    for path, name in ((json_path, "json"), (raw_path, "raw"), (overlay_path, "overlay")):
        if not path.is_file():
            errors.append(f"missing_{name}")
    if errors:
        return {}, np.empty(0), errors
    payload = load_json(json_path)
    if payload.get("capture_id") != audit_row["sample_id"]:
        errors.append("capture_id_mismatch")
    if payload.get("manual_label") != label:
        errors.append("label_mismatch")
    if payload.get("normalization_function") != "AI/landmark_utils.py::normalize_landmarks":
        errors.append("normalizer_mismatch")
    stored = payload.get("features", {})
    if list(stored) != FEATURES:
        errors.append("feature_order")
        return payload, np.empty(0), errors
    vector = np.asarray([stored[name] for name in FEATURES], dtype=np.float64)
    if vector.shape != (63,) or not np.isfinite(vector).all():
        errors.append("invalid_features")
    landmarks = payload.get("raw_landmarks", [])
    if len(landmarks) != 21 or [point.get("index") for point in landmarks] != list(range(21)):
        errors.append("landmark_schema")
    else:
        objects = [SimpleNamespace(x=float(point["x"]), y=float(point["y"]),
                                   z=float(point["z"])) for point in landmarks]
        recomputed = np.asarray(normalize_landmarks(objects), dtype=np.float64)
        if not np.allclose(vector, recomputed, rtol=0.0, atol=1e-12):
            errors.append("feature_recompute")
    if sha256(raw_path) != str(audit_row["raw_sha256"]):
        errors.append("raw_hash")
    if sha256(overlay_path) != str(audit_row["overlay_sha256"]):
        errors.append("overlay_hash")
    return payload, vector, errors


def load_supplement() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_rows = []
    metadata_rows = []
    errors = []

    old = pd.read_csv(PATHS["old_audit_csv"], dtype=str, keep_default_na=False)
    old_accepted = old.loc[
        old.file_integrity.eq("PASS") & old.label_review.eq("LABEL_CONFIRMED") &
        old.landmark_review.eq("LANDMARK_CONFIRMED") & old.input_label.isin(["U", "V"])
    ].copy()
    if len(old_accepted) != 30:
        errors.append(f"old_accept_count:{len(old_accepted)}")
    for _, item in old_accepted.sort_values(["input_label", "sample_id"]).iterrows():
        label = item["input_label"]
        payload, vector, row_errors = validate_payload(item, label)
        errors.extend(f"{item['sample_id']}:{error}" for error in row_errors)
        if row_errors:
            continue
        group_name = "OLD_U" if label == "U" else "OLD_V"
        raw_path = resolve(item["raw_image_path"])
        group_id = f"SUPP-{item['collection_dependence_group']}"
        row = {"source": "camera_debug_uv_audited", "file": rel(raw_path),
               "label": label, "handedness": item["handedness"]}
        row.update({name: float(value) for name, value in zip(FEATURES, vector)})
        train_rows.append(row)
        metadata_rows.append({
            "source": row["source"], "file": row["file"], "label": label,
            "handedness": row["handedness"], "origin": "AUDITED_UV_OLD",
            "supplement_group": group_name, "original_split": "DEBUG_OBSERVED",
            "group_id": group_id, "input_view": item["input_view"],
            "near_duplicate_group": item["near_duplicate_group"],
            "collection_dependence_group": item["collection_dependence_group"],
            "sample_id": item["sample_id"], "audit_status": "LABEL_AND_LANDMARK_CONFIRMED",
            "json_path": rel(resolve(item["json_path"])),
            "raw_image_path": row["file"], "overlay_image_path": rel(resolve(item["overlay_image_path"])),
            "observed_during_v1_error_analysis": True,
        })

    new = pd.read_csv(PATHS["new_audit_csv"], dtype=str, keep_default_na=False)
    new_accepted = new.loc[
        new.decision.eq("ACCEPT") & new.semantic_review.eq("ASL_V_CONFIRMED") &
        new.landmark_review.eq("LANDMARK_8_12_CONFIRMED") &
        new.automatic_integrity.eq("PASS") & new.error_group.isin(["V_TO_U", "V_TO_W"])
    ].copy()
    if len(new_accepted) != 27:
        errors.append(f"new_accept_count:{len(new_accepted)}")
    for _, item in new_accepted.sort_values(["error_group", "sample_id"]).iterrows():
        payload, vector, row_errors = validate_payload(item, "V")
        if payload and payload.get("instant_prediction") != ("U" if item["error_group"] == "V_TO_U" else "W"):
            row_errors.append("error_prediction_mismatch")
        errors.extend(f"{item['sample_id']}:{error}" for error in row_errors)
        if row_errors:
            continue
        raw_path = resolve(item["raw_image_path"])
        group_id = f"SUPP-NEW-{item['conservative_group_id']}"
        row = {"source": "camera_debug_v1_v_error_audited", "file": rel(raw_path),
               "label": "V", "handedness": item["handedness"]}
        row.update({name: float(value) for name, value in zip(FEATURES, vector)})
        train_rows.append(row)
        metadata_rows.append({
            "source": row["source"], "file": row["file"], "label": "V",
            "handedness": row["handedness"], "origin": "AUDITED_V1_V_ERROR_NEW",
            "supplement_group": item["error_group"], "original_split": "DEBUG_OBSERVED",
            "group_id": group_id, "input_view": item["manual_view"],
            "near_duplicate_group": item["near_duplicate_group"],
            "collection_dependence_group": item["conservative_group_id"],
            "sample_id": item["sample_id"], "audit_status": "ASL_V_AND_LANDMARK_CONFIRMED",
            "json_path": rel(resolve(item["json_path"])),
            "raw_image_path": row["file"], "overlay_image_path": rel(resolve(item["overlay_image_path"])),
            "observed_during_v1_error_analysis": True,
        })
    return (pd.DataFrame(train_rows, columns=TRAIN_COLUMNS),
            pd.DataFrame(metadata_rows), errors)


def feature_hashes(matrix: np.ndarray) -> list[str]:
    return [hashlib.sha256(np.asarray(row, dtype="<f8").tobytes()).hexdigest()
            for row in np.asarray(matrix, dtype=np.float64)]


def make_weights(metadata: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    table = metadata[["v3_row_id", "label", "group_id", "origin",
                      "supplement_group", "input_view"]].copy()
    table["group_size"] = table.groupby("group_id").group_id.transform("size").astype(int)
    n = len(table)
    class_target = n / len(LABELS)
    table["weight_uniform"] = 1.0

    group_counts = table.groupby("label").group_id.nunique().to_dict()
    table["weight_group_class_balanced"] = [
        n / (len(LABELS) * group_counts[label] * size)
        for label, size in zip(table.label, table.group_size)
    ]

    audit_rows = []
    def budget_column(name: str, budgets: dict[str, float]) -> np.ndarray:
        values = np.zeros(n, dtype=np.float64)
        for label in LABELS:
            label_mask = table.label.eq(label)
            legacy_mask = label_mask & table.origin.eq("TRAIN_V1")
            label_budgets = {group: fraction for group, fraction in budgets.items()
                             if ((group == "OLD_U" and label == "U") or
                                 (group in {"OLD_V", "V_TO_U", "V_TO_W"} and label == "V"))}
            supplemental_total = sum(label_budgets.values())
            legacy_groups = table.loc[legacy_mask, "group_id"].nunique()
            legacy_fraction = 1.0 - supplemental_total
            if legacy_groups <= 0 or legacy_fraction <= 0:
                raise ValueError(f"Invalid legacy budget for {name}/{label}")
            values[legacy_mask.to_numpy()] = [
                class_target * legacy_fraction / (legacy_groups * size)
                for size in table.loc[legacy_mask, "group_size"]
            ]
            for supplement_group, fraction in label_budgets.items():
                mask = label_mask & table.supplement_group.eq(supplement_group)
                group_ids = table.loc[mask, "group_id"].unique()
                if not len(group_ids):
                    raise ValueError(f"Missing supplement group: {supplement_group}")
                group_mass = class_target * fraction / len(group_ids)
                values[mask.to_numpy()] = [group_mass / size
                                           for size in table.loc[mask, "group_size"]]
        if not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError(f"Invalid weights: {name}")
        table[name] = values
        for label in LABELS:
            mask = table.label.eq(label).to_numpy()
            total = float(values[mask].sum())
            if not math.isclose(total, class_target, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"Class total mismatch: {name}/{label}")
            audit_rows.append({
                "weight_config": name, "scope": "CLASS", "label": label,
                "supplement_group": "", "group_id": "", "sample_count": int(mask.sum()),
                "weight_total": total, "fraction_of_class": total / class_target,
            })
        for (supplement_group, group_id), frame in table.loc[
                table.origin.ne("TRAIN_V1")].groupby(["supplement_group", "group_id"], sort=True):
            indexes = frame.index.to_numpy()
            total = float(values[indexes].sum())
            label = frame.label.iloc[0]
            audit_rows.append({
                "weight_config": name, "scope": "SUPPLEMENT_GROUP", "label": label,
                "supplement_group": supplement_group, "group_id": group_id,
                "sample_count": len(frame), "weight_total": total,
                "fraction_of_class": total / class_target,
            })
        return values

    budget_column("weight_supplement_low", LOW_BUDGETS)
    budget_column("weight_supplement_medium", MEDIUM_BUDGETS)
    return table, audit_rows


def main() -> int:
    for path in PATHS.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    camera_hash_before = sha256(PATHS["camera_app"])
    audit_verification = verify_audits()
    if audit_verification["status"] != "PASS":
        raise ValueError(f"Audit source verification failed: {audit_verification['errors']}")

    v1_config = load_json(PATHS["v1_training_config"])
    snapshots = v1_config["input_snapshots"]
    snapshot_checks = {}
    for key in ("train", "validation", "test", "split_assignments", "group_assignments",
                "split_manifest", "split_integrity"):
        path = PATHS[key]
        expected = snapshots[rel(path)]
        actual = sha256(path)
        snapshot_checks[key] = {"path": rel(path), "expected_sha256": expected,
                                "actual_sha256": actual, "pass": expected == actual}
    if not all(item["pass"] for item in snapshot_checks.values()):
        raise ValueError("Frozen split snapshot mismatch")

    train = pd.read_csv(PATHS["train"])
    if train.columns.tolist() != TRAIN_COLUMNS or len(train) != 62591:
        raise ValueError("TRAIN v1 schema/count mismatch")
    if train.file.duplicated().any() or set(train.label) != set(LABELS):
        raise ValueError("TRAIN v1 identity/label mismatch")
    train_features = train[FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(train_features).all():
        raise ValueError("TRAIN v1 contains non-finite features")

    assignments = pd.read_csv(PATHS["split_assignments"], dtype=str, keep_default_na=False)
    train_assignments = assignments.loc[assignments.split.eq("TRAIN"),
                                        ["file", "label", "group_id"]]
    legacy_meta = train[["source", "file", "label", "handedness"]].merge(
        train_assignments, on=["file", "label"], how="left", validate="one_to_one")
    if legacy_meta.group_id.isna().any() or legacy_meta.group_id.eq("").any():
        raise ValueError("TRAIN v1 group mapping incomplete")
    legacy_meta = legacy_meta.assign(
        origin="TRAIN_V1", supplement_group="", original_split="TRAIN", input_view="",
        near_duplicate_group="", collection_dependence_group=legacy_meta.group_id,
        sample_id="", audit_status="FROZEN_V3_ACCEPTED", json_path="",
        raw_image_path=legacy_meta.file, overlay_image_path="",
        observed_during_v1_error_analysis=False,
    )

    supplement, supplement_meta, supplement_errors = load_supplement()
    if supplement_errors:
        raise ValueError(f"Supplement validation errors: {supplement_errors}")
    expected_groups = {"OLD_U": 15, "OLD_V": 15, "V_TO_U": 12, "V_TO_W": 15}
    actual_groups = supplement_meta.supplement_group.value_counts().to_dict()
    if actual_groups != expected_groups or len(supplement) != 57:
        raise ValueError(f"Supplement count mismatch: {actual_groups}")
    if set(supplement.label) != {"U", "V"}:
        raise ValueError("Unauthorized supplement label")
    if supplement.file.duplicated().any() or supplement_meta.sample_id.duplicated().any():
        raise ValueError("Duplicate supplement identity")
    if set(supplement.file) & set(train.file):
        raise ValueError("Supplement file overlaps TRAIN v1")

    supplement_features = supplement[FEATURES].to_numpy(dtype=np.float64)
    train_feature_hashes = set(feature_hashes(train_features))
    supplement_feature_hash_list = feature_hashes(supplement_features)
    exact_feature_matches = sum(digest in train_feature_hashes
                                for digest in supplement_feature_hash_list)
    exact_supplement_duplicates = len(supplement_feature_hash_list) - len(set(supplement_feature_hash_list))
    if exact_feature_matches or exact_supplement_duplicates:
        raise ValueError("Exact feature duplication detected")

    train_new = pd.concat([train, supplement], ignore_index=True)
    metadata = pd.concat([legacy_meta, supplement_meta], ignore_index=True)
    metadata.insert(0, "v3_row_id", [f"UVW-{index:06d}" for index in range(1, len(metadata) + 1)])
    if len(train_new) != 62648 or len(metadata) != len(train_new):
        raise ValueError("TRAIN UVW row alignment mismatch")
    if not train_new[["source", "file", "label", "handedness"]].reset_index(drop=True).equals(
            metadata[["source", "file", "label", "handedness"]].reset_index(drop=True)):
        raise ValueError("Metadata alignment mismatch")
    weights, weight_audit_rows = make_weights(metadata)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": OUT_DIR / "train_uvw_v1.csv",
        "metadata": OUT_DIR / "train_uvw_sample_metadata.csv",
        "weights": OUT_DIR / "train_uvw_weights.csv",
        "additional": OUT_DIR / "additional_uv_samples.csv",
        "weight_audit_csv": OUT_DIR / "weight_audit.csv",
        "weight_audit_json": OUT_DIR / "weight_audit.json",
        "recommendation": OUT_DIR / "training_recommendation.json",
        "integrity": OUT_DIR / "integrity_report.json",
        "readme": OUT_DIR / "README.md",
        "manifest": OUT_DIR / "manifest.json",
    }
    train_new.to_csv(outputs["train"], index=False, float_format="%.17g")
    metadata.to_csv(outputs["metadata"], index=False)
    weights.to_csv(outputs["weights"], index=False, float_format="%.17g")
    additional = pd.concat([
        metadata.tail(57).reset_index(drop=True),
        supplement[FEATURES].reset_index(drop=True),
    ], axis=1)
    additional.to_csv(outputs["additional"], index=False, float_format="%.17g")
    pd.DataFrame(weight_audit_rows).to_csv(outputs["weight_audit_csv"], index=False,
                                           float_format="%.17g")

    weight_audit = {
        "status": "PASS", "train_rows": len(train_new), "class_target_weight": len(train_new) / 24,
        "supplement_counts": actual_groups,
        "conservative_supplement_groups": int(supplement_meta.group_id.nunique()),
        "near_duplicate_candidates": {
            "old_uv_groups": int(supplement_meta.loc[
                supplement_meta.origin.eq("AUDITED_UV_OLD") &
                supplement_meta.near_duplicate_group.ne(""), "near_duplicate_group"].nunique()),
            "old_uv_samples": int((supplement_meta.origin.eq("AUDITED_UV_OLD") &
                                    supplement_meta.near_duplicate_group.ne("")).sum()),
            "new_v_error_groups": int(supplement_meta.loc[
                supplement_meta.origin.eq("AUDITED_V1_V_ERROR_NEW") &
                supplement_meta.near_duplicate_group.ne(""), "near_duplicate_group"].nunique()),
            "new_v_error_samples": int((supplement_meta.origin.eq("AUDITED_V1_V_ERROR_NEW") &
                                         supplement_meta.near_duplicate_group.ne("")).sum()),
        },
        "budgets": {"low": LOW_BUDGETS, "medium": MEDIUM_BUDGETS},
        "rows": weight_audit_rows,
    }
    write_json(outputs["weight_audit_json"], weight_audit)

    recommendation = {
        "status": "PREPARED_NOT_TRAINED",
        "baseline": "v1_baseline epoch 143",
        "recommended_comparison": ["weight_supplement_low", "weight_supplement_medium"],
        "weighting": {
            "low": {
                "U": "2% audited old U; 98% legacy U",
                "V": "1% old V + 2% V_TO_U + 2% V_TO_W; 95% legacy V",
                "W": "100% legacy W",
            },
            "medium": {
                "U": "5% audited old U; 95% legacy U",
                "V": "2% old V + 4% V_TO_U + 4% V_TO_W; 90% legacy V",
                "W": "100% legacy W",
            },
            "group_control": "Each supplement budget is split equally across conservative session groups, then divided by group size; near-duplicate IDs are retained.",
            "class_control": "Every one of the 24 classes has equal total effective weight.",
        },
        "future_training": {
            "architecture": "Reuse the v1 63-64-32-24 MLP and external VALIDATION-v1 early stopping.",
            "validation": rel(PATHS["validation"]),
            "test": "TEST v1 remains hash-only and must not be used for selection/tuning.",
            "supplement_evaluation": "The 57 observed rows become TRAIN and cannot be independent evaluation evidence.",
            "camera_evaluation": "Collect new manually confirmed unseen gesture bouts after model selection.",
        },
        "pq_data_used": False, "training_performed": False,
    }
    write_json(outputs["recommendation"], recommendation)

    legacy_numeric_preserved = np.array_equal(
        train_new.iloc[:len(train)][FEATURES].to_numpy(dtype=np.float64), train_features)
    supplement_numeric_preserved = np.array_equal(
        train_new.iloc[len(train):][FEATURES].to_numpy(dtype=np.float64), supplement_features)
    weight_columns = ["weight_uniform", "weight_group_class_balanced",
                      "weight_supplement_low", "weight_supplement_medium"]
    weight_sums = {column: float(weights[column].sum()) for column in weight_columns}
    checks = {
        "audit_sources_verified": audit_verification["status"] == "PASS",
        "frozen_split_hashes_match": all(item["pass"] for item in snapshot_checks.values()),
        "train_v1_rows_62591": len(train) == 62591,
        "train_v1_schema_exact": train.columns.tolist() == TRAIN_COLUMNS,
        "supplement_rows_57": len(supplement) == 57,
        "supplement_counts_exact": actual_groups == expected_groups,
        "supplement_labels_only_U_V": set(supplement.label) == {"U", "V"},
        "pq_data_not_used": not supplement_meta.origin.str.contains("PQ", case=False).any(),
        "supplement_validation_errors_zero": not supplement_errors,
        "file_overlap_zero": not (set(supplement.file) & set(train.file)),
        "supplement_sample_ids_unique": supplement_meta.sample_id.nunique() == 57,
        "exact_feature_matches_train_zero": exact_feature_matches == 0,
        "exact_feature_duplicates_supplement_zero": exact_supplement_duplicates == 0,
        "train_new_rows_62648": len(train_new) == 62648,
        "train_new_schema_exact": train_new.columns.tolist() == TRAIN_COLUMNS,
        "train_new_features_finite": bool(np.isfinite(train_new[FEATURES].to_numpy(float)).all()),
        "labels_24": set(train_new.label) == set(LABELS),
        "legacy_labels_features_unchanged": bool(legacy_numeric_preserved and
                                                  train_new.iloc[:len(train)].label.equals(train.label)),
        "supplement_labels_features_unchanged": bool(supplement_numeric_preserved and
                                                      train_new.iloc[len(train):].label.reset_index(drop=True).equals(
                                                          supplement.label.reset_index(drop=True))),
        "metadata_rows_align": len(metadata) == len(train_new),
        "all_group_ids_present": metadata.group_id.notna().all() and metadata.group_id.ne("").all(),
        "supplement_conservative_groups_5": supplement_meta.group_id.nunique() == 5,
        "weight_rows_align": len(weights) == len(train_new),
        "weights_positive_finite": bool(np.isfinite(weights[weight_columns].to_numpy()).all() and
                                         (weights[weight_columns] > 0).all().all()),
        "weight_totals_mean_one": all(math.isclose(total, len(train_new), rel_tol=0.0,
                                                    abs_tol=1e-6) for total in weight_sums.values()),
        "weight_audit_pass": weight_audit["status"] == "PASS",
        "validation_not_modified": sha256(PATHS["validation"]) == snapshots[rel(PATHS["validation"])],
        "test_hash_only_not_parsed": sha256(PATHS["test"]) == snapshots[rel(PATHS["test"])],
        "static_camera_unchanged": sha256(PATHS["camera_app"]) == camera_hash_before,
        "training_not_performed": True,
    }
    integrity_status = "PASS" if all(checks.values()) else "FAIL"
    integrity = {
        "status": integrity_status, "checks_run": len(checks), "checks": checks,
        "audit_source_verification": audit_verification,
        "snapshot_checks": snapshot_checks,
        "duplicates": {"file_overlap_with_train": 0,
                       "exact_feature_matches_with_train": exact_feature_matches,
                       "exact_feature_duplicates_within_supplement": exact_supplement_duplicates},
        "weight_sums": weight_sums,
        "test_access": {"verification": "SHA256_ONLY", "csv_opened_or_parsed": False,
                        "used_for_training_or_selection": False},
        "protected_inputs_modified": False,
    }
    write_json(outputs["integrity"], integrity)
    outputs["readme"].write_text(
        "# U/V/W-boundary TRAIN preparation\n\n"
        "This directory contains TRAIN v1 plus 57 audit-confirmed U/V samples. "
        "No P/Q data is used. No model was trained. VALIDATION v1 remains unchanged; "
        "TEST v1 was SHA-256 checked only. The 57 appended rows are observed training data, "
        "not an independent evaluation set.\n", encoding="utf-8")

    input_hashes = {rel(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in PATHS.values()}
    output_hashes = {rel(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                     for key, path in outputs.items() if key != "manifest"}
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "UVW_TRAIN_DATA_PREPARED" if integrity_status == "PASS" else "INTEGRITY_FAILED",
        "baseline": "v1_baseline epoch 143", "normalization":
        "AI/landmark_utils.py::normalize_landmarks (unchanged)",
        "features": FEATURES, "labels": LABELS,
        "counts": {"train_v1": len(train), "supplement": len(supplement),
                   "supplement_groups": actual_groups, "train_new": len(train_new),
                   "conservative_supplement_groups": supplement_meta.group_id.nunique()},
        "weight_columns": weight_columns, "input_hashes": input_hashes,
        "output_hashes": output_hashes, "integrity": {"status": integrity_status,
                                                       "checks_run": len(checks)},
        "validation_v1_changed": False, "test_v1_parsed": False,
        "pq_data_used": False, "training_performed": False,
    }
    write_json(outputs["manifest"], manifest)
    print(f"STATUS={manifest['status']}")
    print(f"TRAIN_ROWS={len(train_new)} SUPPLEMENT_ROWS={len(supplement)}")
    print(f"INTEGRITY={integrity_status} CHECKS={len(checks)}")
    return 0 if integrity_status == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
