"""Leakage-safe, group-aware wrong-label candidate audit.

This script never changes labels or source data.  Every sample is scored by a
model, kNN index, and class-distance reference fitted without that sample's
``group_id``.  Intermediate fold artefacts are checkpointed and signature
checked so an interrupted run can resume without retraining completed folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [f"f{i}" for i in range(1, 64)]
N_SPLITS = 5
KNN_K = 7
SEED = 42
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def parse_args() -> argparse.Namespace:
    ai_dir = Path(__file__).resolve().parent
    dataset_dir = ai_dir / "dataset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=dataset_dir / "phase2_quality_gate_v2" / "dataset_chucai_clean_reviewed_v2.csv",
    )
    parser.add_argument("--group-map", type=Path, default=dataset_dir / "sample_group_map.csv")
    parser.add_argument(
        "--raw-landmarks",
        type=Path,
        default=ai_dir / "raw_data" / "raw_landmarks_chucai_all.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=dataset_dir)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=dataset_dir / "wrong_label_checkpoints_v1",
    )
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--knn-query-batch", type=int, default=2048)
    parser.add_argument("--skip-visuals", action="store_true")
    return parser.parse_args()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _file_signature(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _load_inputs(dataset_path: Path, group_path: Path) -> pd.DataFrame:
    data = pd.read_csv(dataset_path)
    group_map = pd.read_csv(group_path)
    required_data = {"source", "file", "label", "handedness", *FEATURE_COLUMNS}
    required_group = {"file", "label", "group_id", "group_type", "group_size"}
    if missing := required_data.difference(data.columns):
        raise ValueError(f"Canonical dataset is missing columns: {sorted(missing)}")
    if missing := required_group.difference(group_map.columns):
        raise ValueError(f"Group map is missing columns: {sorted(missing)}")
    if data["file"].duplicated().any() or group_map["file"].duplicated().any():
        raise ValueError("file must be unique in both canonical dataset and group map")
    merged = data.merge(
        group_map[["file", "label", "group_id", "group_type", "group_size"]],
        on="file", how="left", validate="one_to_one", indicator=True,
        suffixes=("", "_group"),
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError(f"{int((merged['_merge'] != 'both').sum())} rows lack group_id")
    mismatch = merged["label"].astype(str) != merged["label_group"].astype(str)
    if mismatch.any():
        raise ValueError(f"{int(mismatch.sum())} label mismatches between canonical and group map")
    if merged["group_id"].isna().any():
        raise ValueError("group_id contains NaN")
    purity = merged.groupby("group_id", sort=False)["label"].nunique()
    if (purity != 1).any():
        raise ValueError(f"{int((purity != 1).sum())} groups contain multiple labels")
    x = merged[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("f1..f63 contain NaN or infinity")
    merged = merged.drop(columns=["_merge", "label_group"])
    merged.insert(0, "row_id", np.arange(len(merged), dtype=np.int64))
    return merged


def _build_splits(frame: pd.DataFrame) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    y = frame["label"].astype(str).to_numpy()
    groups = frame["group_id"].astype(str).to_numpy()
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    audit_rows: list[dict] = []
    hits = np.zeros(len(frame), dtype=np.int8)
    dummy = np.zeros((len(frame), 1), dtype=np.uint8)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(dummy, y, groups)):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        overlap = train_groups.intersection(val_groups)
        if overlap:
            raise RuntimeError(f"Fold {fold} leaks {len(overlap)} groups")
        hits[val_idx] += 1
        splits.append((train_idx, val_idx))
        audit_rows.append({
            "fold": fold,
            "train_samples": len(train_idx),
            "validation_samples": len(val_idx),
            "train_groups": len(train_groups),
            "validation_groups": len(val_groups),
            "group_intersection_count": 0,
            "train_label_counts": json.dumps(pd.Series(y[train_idx]).value_counts().sort_index().to_dict()),
            "validation_label_counts": json.dumps(pd.Series(y[val_idx]).value_counts().sort_index().to_dict()),
        })
    if not np.all(hits == 1):
        raise RuntimeError("Every sample must occur in exactly one validation fold")
    return splits, pd.DataFrame(audit_rows)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    safe = np.clip(probabilities, 1e-15, 1.0)
    return -(safe * np.log(safe)).sum(axis=1)


def _centroid_reference(x_train: np.ndarray, y_train: np.ndarray, classes: np.ndarray) -> dict:
    reference: dict[str, dict[str, np.ndarray]] = {}
    for label in classes:
        members = x_train[y_train == label]
        centroid = members.mean(axis=0)
        distribution = np.sort(np.linalg.norm(members - centroid, axis=1))
        reference[str(label)] = {"centroid": centroid, "distribution": distribution}
    return reference


def _distance_and_percentile(
    x: np.ndarray, labels: np.ndarray, reference: dict
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.empty(len(x), dtype=np.float64)
    percentiles = np.empty(len(x), dtype=np.float64)
    for label in np.unique(labels):
        mask = labels == label
        item = reference[str(label)]
        current = np.linalg.norm(x[mask] - item["centroid"], axis=1)
        distances[mask] = current
        distribution = item["distribution"]
        percentiles[mask] = np.searchsorted(distribution, current, side="right") / len(distribution)
    return distances, percentiles


def _query_knn_batched(
    neighbors: NearestNeighbors,
    x_val: np.ndarray,
    batch_size: int,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    distance_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    batches = math.ceil(len(x_val) / batch_size)
    for batch, start in enumerate(range(0, len(x_val), batch_size), start=1):
        stop = min(start + batch_size, len(x_val))
        distances, indices = neighbors.kneighbors(x_val[start:stop], return_distance=True)
        distance_parts.append(distances)
        index_parts.append(indices)
        print(f"[fold {fold + 1}] kNN batch {batch}/{batches}", flush=True)
    return np.vstack(distance_parts), np.vstack(index_parts)


def _fold_result(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    x_all: np.ndarray,
    y_all: np.ndarray,
    checkpoint_dir: Path,
    max_iter: int,
    knn_batch: int,
) -> pd.DataFrame:
    complete_path = checkpoint_dir / f"fold_{fold}_complete.joblib"
    if complete_path.exists():
        saved = joblib.load(complete_path)
        if not np.array_equal(saved["validation_idx"], val_idx):
            raise RuntimeError(f"Stale validation indices in {complete_path}")
        print(f"[fold {fold + 1}/{N_SPLITS}] loading complete checkpoint", flush=True)
        # Historical checkpoints contain the modal alternative under a
        # nearest-alternative name. Repair reporting in memory; do not refit or
        # overwrite the immutable prediction checkpoint.
        return repair_knn_alternative_columns(saved["result"], y_all[val_idx])

    model_path = checkpoint_dir / f"fold_{fold}_model.joblib"
    if model_path.exists():
        saved_model = joblib.load(model_path)
        if not np.array_equal(saved_model["train_idx"], train_idx):
            raise RuntimeError(f"Stale training indices in {model_path}")
        scaler = saved_model["scaler"]
        mlp = saved_model["mlp"]
        print(f"[fold {fold + 1}/{N_SPLITS}] loading model checkpoint", flush=True)
    else:
        print(
            f"[fold {fold + 1}/{N_SPLITS}] fitting MLP train={len(train_idx):,} val={len(val_idx):,}",
            flush=True,
        )
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_all[train_idx])
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            batch_size=256, learning_rate_init=0.001, max_iter=max_iter,
            early_stopping=True, validation_fraction=0.10,
            n_iter_no_change=15, random_state=SEED + fold,
        )
        mlp.fit(x_train_scaled, y_all[train_idx])
        joblib.dump(
            {"train_idx": train_idx, "scaler": scaler, "mlp": mlp},
            model_path,
            compress=3,
        )

    x_train_scaled = scaler.transform(x_all[train_idx])
    x_val_scaled = scaler.transform(x_all[val_idx])
    probabilities = mlp.predict_proba(x_val_scaled)
    order = np.argsort(probabilities, axis=1)
    rows = np.arange(len(val_idx))
    top = order[:, -1]
    second = order[:, -2]
    predicted = mlp.classes_[top].astype(str)
    true_labels = y_all[val_idx]
    class_index = {str(label): index for index, label in enumerate(mlp.classes_)}
    true_probability = np.array([
        probabilities[row, class_index[str(label)]] for row, label in enumerate(true_labels)
    ])

    centroid_reference = _centroid_reference(x_train_scaled, y_all[train_idx], mlp.classes_)
    original_distance, original_percentile = _distance_and_percentile(
        x_val_scaled, true_labels, centroid_reference
    )
    predicted_distance, predicted_percentile = _distance_and_percentile(
        x_val_scaled, predicted, centroid_reference
    )

    print(f"[fold {fold + 1}/{N_SPLITS}] fitting/querying leakage-safe kNN-{KNN_K}", flush=True)
    neighbors = NearestNeighbors(n_neighbors=KNN_K, algorithm="brute", metric="euclidean", n_jobs=-1)
    neighbors.fit(x_train_scaled)
    knn_distances, knn_indices = _query_knn_batched(neighbors, x_val_scaled, knn_batch, fold)
    neighbor_labels = y_all[train_idx][knn_indices]
    knn_original = (neighbor_labels == true_labels[:, None]).mean(axis=1)
    knn_predicted = (neighbor_labels == predicted[:, None]).mean(axis=1)
    alternative = []
    serialized_neighbors = []
    for labels, original in zip(neighbor_labels, true_labels, strict=True):
        counts = Counter(str(value) for value in labels if str(value) != str(original))
        alternative.append(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if counts else "")
        serialized_neighbors.append("|".join(map(str, labels)))

    result = pd.DataFrame({
        "row_id": val_idx,
        "oof_fold": fold,
        "predicted_label": predicted,
        "true_label_probability": true_probability,
        "predicted_probability": probabilities[rows, top],
        "margin": probabilities[rows, top] - probabilities[rows, second],
        "entropy": _entropy(probabilities),
        "knn_labels": serialized_neighbors,
        "knn_original_agreement": knn_original,
        "knn_predicted_agreement": knn_predicted,
        "knn_mean_distance": knn_distances.mean(axis=1),
        "knn_median_distance": np.median(knn_distances, axis=1),
        "knn_nearest_distance": knn_distances[:, 0],
        "knn_nearest_alternative_class": alternative,
        "original_class_distance": original_distance,
        "predicted_class_distance": predicted_distance,
        "original_class_distance_percentile": original_percentile,
        "predicted_class_distance_percentile": predicted_percentile,
    })
    result = repair_knn_alternative_columns(result, true_labels)
    joblib.dump({"validation_idx": val_idx, "result": result}, complete_path, compress=3)
    return result


def alternative_neighbor_evidence(labels: list[str], original: str) -> tuple[str, str, str]:
    """Return first-distance-ordered alternative, mode, and search coverage.

    The result is restricted to the saved top-k neighbors. An empty nearest
    value means no alternative was observed there, not that none exists in
    the full training fold. The mode retains the historical alphabetical
    tie-break and is explicitly named as such.
    """
    alternatives = [str(label) for label in labels if str(label) != str(original)]
    if not alternatives:
        return "", "", "NOT_OBSERVED_IN_TOP_K"
    counts = Counter(alternatives)
    modal = sorted(counts, key=lambda label: (-counts[label], label))[0]
    return alternatives[0], modal, "FOUND_IN_TOP_K"


def repair_knn_alternative_columns(
    frame: pd.DataFrame, original_labels: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Repair reporting using saved distance-ordered labels only.

    No model, index, threshold, candidate status, or probability is recomputed.
    """
    if len(frame) != len(original_labels):
        raise ValueError("Original labels and cached predictions must align")
    evidence = []
    for serialized, original in zip(frame["knn_labels"], original_labels, strict=True):
        if not isinstance(serialized, str):
            raise ValueError("Missing serialized kNN labels")
        labels = serialized.split("|")
        if len(labels) != KNN_K or any(not label for label in labels):
            raise ValueError(f"Expected exactly {KNN_K} saved neighbor labels")
        evidence.append(alternative_neighbor_evidence(labels, str(original)))
    result = frame.copy()
    for index, column in enumerate((
        "knn_nearest_alternative_class", "knn_modal_alternative_class",
        "knn_alternative_search_status",
    )):
        result[column] = [values[index] for values in evidence]
    return result


def _class_thresholds(oof: pd.DataFrame, labels: list[str]) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    thresholds: dict[str, dict] = {}
    correct = oof["label"].astype(str) == oof["predicted_label"].astype(str)
    for label in labels:
        subset = oof.loc[correct & (oof["label"].astype(str) == label)]
        if len(subset) < 20:
            raise RuntimeError(f"Too few correct OOF samples to calibrate label {label}: {len(subset)}")
        item = {
            "model_reference_count": len(subset),
            "confidence_median": subset["predicted_probability"].quantile(0.50),
            "confidence_q75": subset["predicted_probability"].quantile(0.75),
            "margin_median": subset["margin"].quantile(0.50),
            "margin_q75": subset["margin"].quantile(0.75),
            "entropy_median": subset["entropy"].quantile(0.50),
            "entropy_q25": subset["entropy"].quantile(0.25),
        }
        thresholds[label] = item
        rows.append({"label": label, **item})
    return pd.DataFrame(rows), thresholds


def _confusion_statistics(oof: pd.DataFrame, labels: list[str]) -> tuple[pd.DataFrame, dict]:
    true = oof["label"].astype(str)
    pred = oof["predicted_label"].astype(str)
    total_by_label = true.value_counts()
    errors = oof.loc[true != pred].groupby(["label", "predicted_label"]).size().rename("count").reset_index()
    if errors.empty:
        return errors.assign(rate=[]), {"count_q75": None, "rate_q75": None, "frequent_pairs": []}
    errors["rate"] = [count / total_by_label[label] for label, count in zip(errors["label"], errors["count"], strict=True)]
    count_q75 = float(errors["count"].quantile(0.75))
    rate_q75 = float(errors["rate"].quantile(0.75))
    errors["frequent_confusion_pair"] = (errors["count"] >= count_q75) & (errors["rate"] >= rate_q75)
    errors = errors.sort_values(["count", "rate"], ascending=False).reset_index(drop=True)
    metadata = {
        "definition": "directed OOF error pair at or above empirical Q75 for both count and within-original-label rate",
        "count_q75": count_q75,
        "rate_q75": rate_q75,
        "frequent_pairs": errors.loc[errors["frequent_confusion_pair"], ["label", "predicted_label"]].values.tolist(),
    }
    return errors, metadata


def _apply_candidate_policy(
    oof: pd.DataFrame,
    thresholds: dict,
    confusion_meta: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = oof.copy()
    disagreement = result["label"].astype(str) != result["predicted_label"].astype(str)
    conf_floor = result["predicted_label"].map(lambda value: thresholds[str(value)]["confidence_median"]).astype(float)
    margin_floor = result["predicted_label"].map(lambda value: thresholds[str(value)]["margin_median"]).astype(float)
    entropy_ceiling = result["predicted_label"].map(lambda value: thresholds[str(value)]["entropy_median"]).astype(float)
    strict_conf = result["predicted_label"].map(lambda value: thresholds[str(value)]["confidence_q75"]).astype(float)
    strict_margin = result["predicted_label"].map(lambda value: thresholds[str(value)]["margin_q75"]).astype(float)
    strict_entropy = result["predicted_label"].map(lambda value: thresholds[str(value)]["entropy_q25"]).astype(float)
    result["model_strong_signal"] = disagreement & (
        (result["predicted_probability"] >= conf_floor)
        & (result["margin"] >= margin_floor)
        & (result["entropy"] <= entropy_ceiling)
    )
    result["model_very_strong_signal"] = disagreement & (
        (result["predicted_probability"] >= strict_conf)
        & (result["margin"] >= strict_margin)
        & (result["entropy"] <= strict_entropy)
    )
    result["knn_strong_signal"] = disagreement & (
        (result["knn_predicted_agreement"] >= 5 / KNN_K)
        & (result["knn_original_agreement"] <= 2 / KNN_K)
    )
    result["distance_strong_signal"] = disagreement & (
        (result["original_class_distance_percentile"] >= 0.90)
        & (result["predicted_class_distance_percentile"] <= 0.50)
        & (result["predicted_class_distance"] < result["original_class_distance"])
    )
    frequent_pairs = {tuple(pair) for pair in confusion_meta["frequent_pairs"]}
    result["frequent_confusion_pair"] = [
        (str(label), str(pred)) in frequent_pairs
        for label, pred in zip(result["label"], result["predicted_label"], strict=True)
    ]

    # Group evidence is evaluated only from OOF predictions; every member was
    # scored with its complete group absent from that fold's training data.
    group_rows: list[dict] = []
    group_signal_ids: set[str] = set()
    for group_id, group in result.groupby("group_id", sort=False):
        original = str(group["label"].iloc[0])
        alternatives = group.loc[group["predicted_label"].astype(str) != original, "predicted_label"].astype(str)
        if len(group) < 2 or alternatives.empty:
            continue
        counts = alternatives.value_counts()
        predicted = str(counts.index[0])
        count = int(counts.iloc[0])
        fraction = count / len(group)
        if count >= 2 and fraction >= 2 / 3:
            group_signal_ids.add(str(group_id))
            group_rows.append({
                "group_id": group_id,
                "label": original,
                "predicted_label": predicted,
                "group_size": len(group),
                "agreeing_alternative_prediction_count": count,
                "agreeing_alternative_prediction_fraction": fraction,
                "group_status": "GROUP_LABEL_DISAGREEMENT",
                "files": "|".join(group["file"].astype(str)),
            })
    result["group_label_disagreement"] = result["group_id"].astype(str).isin(group_signal_ids)
    signal_count = (
        result["model_strong_signal"].astype(int)
        + result["knn_strong_signal"].astype(int)
        + result["distance_strong_signal"].astype(int)
    )
    # Common confusions are held to a stricter model threshold (empirical Q75)
    # unless coherent group evidence independently supports the alternative.
    all_three = disagreement & (signal_count == 3)
    high = all_three & (
        ~result["frequent_confusion_pair"]
        | result["model_very_strong_signal"]
        | result["group_label_disagreement"]
    )
    possible = disagreement & ~high & (
        (signal_count >= 2)
        | (result["group_label_disagreement"] & (signal_count >= 1))
    )
    result["label_consistency_status"] = "LABEL_CONSISTENT"
    result.loc[possible, "label_consistency_status"] = "POSSIBLE_WRONG_LABEL"
    result.loc[high, "label_consistency_status"] = "HIGH_CONFIDENCE_WRONG_LABEL_CANDIDATE"
    reasons = []
    for row in result.itertuples(index=False):
        parts = []
        if row.model_strong_signal:
            parts.append("strong_oof_model")
        if row.knn_strong_signal:
            parts.append("knn_supermajority_for_prediction")
        if row.distance_strong_signal:
            parts.append("far_original_near_prediction_distribution")
        if row.group_label_disagreement:
            parts.append("group_label_disagreement")
        if row.frequent_confusion_pair:
            parts.append("frequent_confusion_strict_policy")
        reasons.append(";".join(parts) if parts else "no_multi_signal_evidence")
    result["reason"] = reasons
    group_frame = pd.DataFrame(group_rows, columns=[
        "group_id", "label", "predicted_label", "group_size",
        "agreeing_alternative_prediction_count", "agreeing_alternative_prediction_fraction",
        "group_status", "files",
    ])
    return result, group_frame


def _diverse_sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rng = np.random.default_rng(seed)
    pools = {}
    for label, part in frame.groupby("label", sort=True):
        indices = part.index.to_numpy().copy()
        rng.shuffle(indices)
        pools[str(label)] = list(indices)
    selected: list[int] = []
    labels = list(pools)
    while len(selected) < maximum and any(pools.values()):
        rng.shuffle(labels)
        for label in labels:
            if pools[label] and len(selected) < maximum:
                selected.append(pools[label].pop())
    return frame.loc[selected].copy().reset_index(drop=True)


def _load_selected_landmarks(raw_path: Path, files: set[str]) -> dict[str, np.ndarray]:
    columns = ["file"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    found: dict[str, np.ndarray] = {}
    for chunk in pd.read_csv(raw_path, usecols=columns, chunksize=10000):
        chosen = chunk[chunk["file"].astype(str).isin(files)]
        for row in chosen.itertuples(index=False, name=None):
            values = np.asarray(row[1:], dtype=float).reshape(21, 3)
            found[str(row[0])] = values
    return found


def _resolve_image(project_root: Path, relative: str) -> Path:
    return project_root / Path(str(relative).replace("\\", os.sep).replace("/", os.sep))


def _render_tile(row, landmarks: np.ndarray | None, project_root: Path) -> tuple[Image.Image, str]:
    width, height, header = 360, 300, 72
    tile = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(tile)
    image_path = _resolve_image(project_root, str(row.file))
    status = "OK"
    try:
        image = Image.open(image_path).convert("RGB")
        if landmarks is not None:
            overlay = ImageDraw.Draw(image)
            points = [(float(point[0]) * image.width, float(point[1]) * image.height) for point in landmarks]
            for a, b in HAND_CONNECTIONS:
                overlay.line((points[a], points[b]), fill=(0, 255, 0), width=max(2, image.width // 300))
            radius = max(2, image.width // 180)
            for x, y in points:
                overlay.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 40, 40))
        image.thumbnail((width, height - header), Image.Resampling.LANCZOS)
        x = (width - image.width) // 2
        y = header + (height - header - image.height) // 2
        tile.paste(image, (x, y))
    except Exception as exc:  # visual manifest records failures without hiding candidates
        status = f"ERROR: {type(exc).__name__}: {exc}"
        draw.text((8, header + 20), "IMAGE LOAD ERROR", fill="red")
    text = (
        f"{row.label} -> {row.predicted_label}  p={row.predicted_probability:.3f}\n"
        f"kNN orig/pred={row.knn_original_agreement:.2f}/{row.knn_predicted_agreement:.2f}\n"
        f"group={str(row.group_id)[:24]}"
    )
    draw.multiline_text((6, 5), text, fill="black", font=ImageFont.load_default(), spacing=2)
    return tile, status


def _create_visuals(
    audit: pd.DataFrame,
    raw_path: Path,
    output_dir: Path,
    project_root: Path,
) -> dict:
    visual_dir = output_dir / "wrong_label_audit_visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    selections = {
        "high": _diverse_sample(
            audit[audit["label_consistency_status"] == "HIGH_CONFIDENCE_WRONG_LABEL_CANDIDATE"], 30, SEED
        ),
        "possible": _diverse_sample(
            audit[audit["label_consistency_status"] == "POSSIBLE_WRONG_LABEL"], 30, SEED + 1
        ),
    }
    selected_files = {str(value) for frame in selections.values() for value in frame["file"]}
    landmarks = _load_selected_landmarks(raw_path, selected_files)
    manifest_rows: list[dict] = []
    page_paths: list[str] = []
    for category, selected in selections.items():
        tiles = []
        for row in selected.itertuples(index=False):
            tile, image_status = _render_tile(row, landmarks.get(str(row.file)), project_root)
            tiles.append(tile)
            manifest_rows.append({
                "category": category,
                "file": row.file,
                "label": row.label,
                "predicted_label": row.predicted_label,
                "predicted_probability": row.predicted_probability,
                "knn_original_agreement": row.knn_original_agreement,
                "knn_predicted_agreement": row.knn_predicted_agreement,
                "group_id": row.group_id,
                "raw_landmark_found": str(row.file) in landmarks,
                "image_status": image_status,
            })
        for page_number, start in enumerate(range(0, len(tiles), 15), start=1):
            current = tiles[start:start + 15]
            sheet = Image.new("RGB", (360 * 5, 300 * 3), (225, 225, 225))
            for index, tile in enumerate(current):
                sheet.paste(tile, ((index % 5) * 360, (index // 5) * 300))
            path = visual_dir / f"{category}_contact_sheet_{page_number}.jpg"
            sheet.save(path, quality=92)
            page_paths.append(str(path.resolve()))
    manifest = pd.DataFrame(manifest_rows)
    _write_csv(manifest, visual_dir / "contact_sheet_manifest.csv")
    return {
        "directory": str(visual_dir.resolve()),
        "selected_high": len(selections["high"]),
        "selected_possible": len(selections["possible"]),
        "raw_landmark_missing": int((~manifest["raw_landmark_found"]).sum()) if len(manifest) else 0,
        "image_errors": int((manifest["image_status"] != "OK").sum()) if len(manifest) else 0,
        "contact_sheets": page_paths,
        "selection_random_state": SEED,
        "thresholds_tuned_from_visual_sample": False,
        "limitation": "Static raw-landmark overlays support geometry inspection only; they do not prove the semantic label.",
    }


def main() -> int:
    args = parse_args()
    started = time.time()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        "version": 1,
        "dataset": _file_signature(args.dataset),
        "group_map": _file_signature(args.group_map),
        "config": {
            "n_splits": N_SPLITS, "knn_k": KNN_K, "seed": SEED,
            "max_iter": args.max_iter, "features": FEATURE_COLUMNS,
            "splitter": "StratifiedGroupKFold",
        },
    }
    signature_path = args.checkpoint_dir / "input_signature.json"
    if signature_path.exists():
        previous = json.loads(signature_path.read_text(encoding="utf-8"))
        if previous != signature:
            raise RuntimeError("Checkpoint signature differs from current inputs/configuration")
    else:
        _write_json(signature_path, signature)

    print("Loading canonical dataset and split-safe group map...", flush=True)
    frame = _load_inputs(args.dataset, args.group_map)
    if len(frame) != 89428:
        raise ValueError(f"Expected 89,428 canonical samples, found {len(frame):,}")
    splits, fold_audit = _build_splits(frame)
    _write_csv(fold_audit, args.checkpoint_dir / "fold_group_audit.csv")
    print(f"Validated {len(frame):,} samples and {frame['group_id'].nunique():,} pure groups", flush=True)

    x_all = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    y_all = frame["label"].astype(str).to_numpy()
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        fold_results.append(_fold_result(
            fold, train_idx, val_idx, x_all, y_all, args.checkpoint_dir,
            args.max_iter, args.knn_query_batch,
        ))
        _write_json(args.checkpoint_dir / "progress.json", {
            "completed_folds": fold + 1, "total_folds": N_SPLITS,
            "last_completed_fold": fold, "updated_epoch": time.time(),
        })

    scores = pd.concat(fold_results, ignore_index=True).sort_values("row_id")
    if len(scores) != len(frame) or scores["row_id"].duplicated().any():
        raise RuntimeError("OOF outputs are not exactly one-to-one with canonical rows")
    audit = frame.merge(scores, on="row_id", how="left", validate="one_to_one")
    audit.insert(audit.columns.get_loc("label") + 1, "true_label", audit["label"].astype(str))
    labels = sorted(audit["label"].astype(str).unique())
    threshold_frame, thresholds = _class_thresholds(audit, labels)
    confusion_pairs, confusion_meta = _confusion_statistics(audit, labels)
    audit, group_candidates = _apply_candidate_policy(audit, thresholds, confusion_meta)

    accuracy = accuracy_score(audit["label"], audit["predicted_label"])
    macro_f1 = f1_score(audit["label"], audit["predicted_label"], average="macro")
    matrix = confusion_matrix(audit["label"], audit["predicted_label"], labels=labels)
    statuses = [
        "LABEL_CONSISTENT", "POSSIBLE_WRONG_LABEL",
        "HIGH_CONFIDENCE_WRONG_LABEL_CANDIDATE",
    ]
    status_counts = audit["label_consistency_status"].value_counts().reindex(statuses, fill_value=0)
    candidates = audit[audit["label_consistency_status"] != "LABEL_CONSISTENT"].copy()
    candidates["status_rank"] = candidates["label_consistency_status"].map({
        "HIGH_CONFIDENCE_WRONG_LABEL_CANDIDATE": 0, "POSSIBLE_WRONG_LABEL": 1,
    })
    candidates = candidates.sort_values(
        ["status_rank", "predicted_probability", "knn_predicted_agreement"],
        ascending=[True, False, False],
    ).drop(columns="status_rank")

    output_audit = args.output_dir / "wrong_label_audit.csv"
    output_candidates = args.output_dir / "wrong_label_candidates.csv"
    output_groups = args.output_dir / "wrong_label_group_candidates.csv"
    output_summary = args.output_dir / "wrong_label_summary.json"
    _write_csv(audit.drop(columns="row_id"), output_audit)
    _write_csv(candidates.drop(columns="row_id"), output_candidates)
    _write_csv(group_candidates, output_groups)
    _write_csv(threshold_frame, args.checkpoint_dir / "class_model_thresholds.csv")
    _write_csv(confusion_pairs, args.checkpoint_dir / "oof_confusion_pairs.csv")

    visuals = {"skipped": True}
    if not args.skip_visuals:
        visuals = _create_visuals(
            audit, args.raw_landmarks, args.output_dir, Path(__file__).resolve().parent.parent
        )
    by_label_frame = (
        audit.groupby(["label", "label_consistency_status"]).size().unstack(fill_value=0)
        .reindex(columns=statuses, fill_value=0).reset_index()
    )
    label_quality = audit.assign(
        oof_correct=audit["label"].astype(str) == audit["predicted_label"].astype(str)
    ).groupby("label", as_index=False).agg(
        total_samples=("file", "size"),
        oof_correct=("oof_correct", "sum"),
        group_label_disagreement_samples=("group_label_disagreement", "sum"),
    )
    label_quality["oof_accuracy"] = label_quality["oof_correct"] / label_quality["total_samples"]
    by_label = by_label_frame.merge(label_quality, on="label", validate="one_to_one").to_dict(orient="records")
    special = {
        label: next(item for item in by_label if item["label"] == label)
        for label in ["M", "N", "T"]
    }
    summary = {
        "audit": "WRONG_LABEL_LABEL_CONSISTENCY",
        "canonical_samples": len(audit),
        "groups": int(audit["group_id"].nunique()),
        "labels": labels,
        "source_data_modified": False,
        "labels_modified": False,
        "production_model_trained": False,
        "oof": {
            "splitter": "StratifiedGroupKFold", "folds": N_SPLITS,
            "random_state": SEED, "all_group_intersections_empty": True,
            "each_sample_predicted_once": True,
            "accuracy": accuracy, "macro_f1": macro_f1,
            "confusion_matrix_labels": labels,
            "confusion_matrix": matrix.tolist(),
            "top_confusion_pairs": confusion_pairs.head(20).to_dict(orient="records"),
        },
        "knn": {
            "k": KNN_K, "fit_scope": "training fold only", "metric": "euclidean after fold StandardScaler",
            "strong_signal": "predicted-label agreement >= 5/7 and original-label agreement <= 2/7",
        },
        "class_distance": {
            "reference": "fold-training class centroid and empirical within-class distance CDF",
            "strong_signal": "original percentile >= 0.90, predicted percentile <= 0.50, predicted distance < original distance",
        },
        "candidate_policy": {
            "model_thresholds": "per-predicted-class empirical median among correct OOF predictions",
            "very_strong_model_thresholds": "per-predicted-class confidence/margin Q75 and entropy Q25 among correct OOF predictions",
            "possible": "alternative OOF prediction plus >=2 of model/kNN/distance signals, or group evidence plus >=1 signal",
            "high": "alternative OOF prediction and all 3 model/kNN/distance signals; empirically frequent confusion also requires very-strong model or group evidence",
            "visuals_used_to_tune_thresholds": False,
        },
        "confusion_aware": confusion_meta,
        "counts": {str(key): int(value) for key, value in status_counts.items()},
        "group_label_disagreement": len(group_candidates),
        "by_label": by_label,
        "special_labels_M_N_T": special,
        "visual_audit": visuals,
        "outputs": {
            "wrong_label_audit": str(output_audit.resolve()),
            "wrong_label_candidates": str(output_candidates.resolve()),
            "wrong_label_group_candidates": str(output_groups.resolve()),
            "wrong_label_summary": str(output_summary.resolve()),
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json(output_summary, summary)
    _write_json(args.checkpoint_dir / "progress.json", {
        "completed_folds": N_SPLITS, "total_folds": N_SPLITS,
        "outputs_complete": True, "updated_epoch": time.time(),
    })
    print("\n=== WRONG-LABEL AUDIT COMPLETE ===", flush=True)
    print(f"OOF accuracy={accuracy:.6f}; macro F1={macro_f1:.6f}", flush=True)
    print(status_counts.to_string(), flush=True)
    print(f"GROUP_LABEL_DISAGREEMENT={len(group_candidates):,}", flush=True)
    print(f"Outputs: {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
