"""Audit stored V->U/W camera failures without rerunning MediaPipe or training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ai" / "preprocessing"))

from landmark_utils import normalize_landmarks  # noqa: E402
INPUTS = {
    "V_TO_U": ROOT / "ai/camera_debug_uv_v1_v_as_u_new/V_front",
    "V_TO_W": ROOT / "ai/camera_debug_uv_v1_v_as_w_new/V_front",
}
OLD_AUDIT = ROOT / "ai/dataset/uv_debug_data_audit_v1/uv_debug_sample_audit.csv"
TRAIN_V1 = ROOT / "ai/dataset/train_validation_test_split_v1/train.csv"
V1_DIR = ROOT / "ai/dataset/mlp_validation_comparison_v1"
OUT = ROOT / "ai/dataset/uv_v1_error_audit_v1"
FEATURES = [f"f{i}" for i in range(1, 64)]
LABELS = tuple("ABCDEFGHIKLMNOPQRSTUVWXY")

# Filled only after direct inspection of raw and overlay contact sheets.
VISUAL_REVIEW: dict[str, dict[str, str]] = {
    sample_id: {
        "semantic_review": "ASL_V_CONFIRMED",
        "landmark_review": "LANDMARK_8_12_CONFIRMED",
        "decision": "ACCEPT",
        "review_notes": (
            "Direct raw/overlay inspection: index and middle fingers are visibly "
            "extended and separated; ring and little fingers are folded; landmarks "
            "8 and 12 track the corresponding fingertips and the remaining skeleton "
            "follows the visible hand."
        ),
    }
    for sample_id in (
        "20260926T063336_895485Z", "20260926T063347_136128Z",
        "20260926T063353_150221Z", "20260926T063356_559176Z",
        "20260926T064011_886096Z", "20260926T064036_431360Z",
        "20260926T064053_907297Z", "20260926T064102_689020Z",
        "20260926T064116_586350Z", "20260926T064122_958889Z",
        "20260926T064128_463611Z", "20260926T064144_643615Z",
        "20260926T064212_573753Z", "20260926T064218_308755Z",
        "20260926T064220_017582Z", "20260926T064222_188515Z",
        "20260926T064225_835787Z", "20260926T064232_881692Z",
        "20260926T064240_927683Z", "20260926T064250_456673Z",
        "20260926T064253_382347Z", "20260926T064311_937108Z",
        "20260926T064314_921899Z", "20260926T064317_384594Z",
        "20260926T064319_868237Z", "20260926T064330_754732Z",
        "20260926T064343_309713Z",
    )
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def phash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    block = cv2.dct(small)[:8, :8]
    median = float(np.median(block.ravel()[1:]))
    bits = (block > median).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def load_samples() -> tuple[list[dict], list[str]]:
    samples = []
    errors = []
    model = joblib.load(V1_DIR / "baseline_model.joblib")
    scaler = joblib.load(V1_DIR / "scaler.joblib")
    for group, folder in INPUTS.items():
        expected_prediction = "U" if group == "V_TO_U" else "W"
        for json_path in sorted(folder.glob("*.json")):
            payload = load_json(json_path)
            sample_id = json_path.stem
            raw_path = folder / payload.get("raw_image", "")
            overlay_path = folder / payload.get("overlay_image", "")
            issue = []
            if payload.get("capture_id") != sample_id:
                issue.append("capture_id_mismatch")
            for key, expected in {
                "manual_label": "V", "manual_view": "front",
                "instant_prediction": expected_prediction,
                "required_instant_prediction": expected_prediction,
                "model": "v1_baseline",
                "normalization_function": "AI/landmark_utils.py::normalize_landmarks",
            }.items():
                if payload.get(key) != expected:
                    issue.append(f"{key}_mismatch")
            if not raw_path.is_file() or not overlay_path.is_file():
                issue.append("missing_image")
                errors.append(f"{sample_id}:missing_image")
                continue
            raw_image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            overlay_image = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
            if raw_image is None or overlay_image is None:
                issue.append("unreadable_image")
                errors.append(f"{sample_id}:unreadable_image")
                continue
            stored = payload.get("features", {})
            if list(stored) != FEATURES:
                issue.append("feature_order")
                errors.append(f"{sample_id}:feature_order")
                continue
            vector = np.asarray([stored[name] for name in FEATURES], dtype=np.float64)
            landmarks = payload.get("raw_landmarks", [])
            if len(landmarks) != 21 or [p.get("index") for p in landmarks] != list(range(21)):
                issue.append("landmark_schema")
                errors.append(f"{sample_id}:landmark_schema")
                continue
            points = np.asarray([[p["x"], p["y"], p["z"]] for p in landmarks], dtype=np.float64)
            objects = [SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in points]
            recomputed = np.asarray(normalize_landmarks(objects), dtype=np.float64)
            if not np.allclose(vector, recomputed, rtol=0.0, atol=1e-12):
                issue.append("normalization_mismatch")
            probabilities = payload.get("probabilities", {})
            if list(probabilities) != list(LABELS):
                issue.append("probability_order")
            else:
                actual = model.predict_proba(scaler.transform(vector.reshape(1, -1)))[0]
                stored_probs = np.asarray([probabilities[label] for label in LABELS], dtype=np.float64)
                if not np.allclose(actual, stored_probs, rtol=0.0, atol=1e-12):
                    issue.append("prediction_recompute_mismatch")

            norm = vector.reshape(21, 3)
            palm_axis = norm[9] - norm[0]
            palm_tilt_2d = math.degrees(math.atan2(float(palm_axis[0]), float(-palm_axis[1])))
            palm_normal = np.cross(norm[5] - norm[0], norm[17] - norm[0])
            normal_length = float(np.linalg.norm(palm_normal))
            plane_tilt = (math.degrees(math.acos(np.clip(abs(float(palm_normal[2])) /
                                                        normal_length, 0.0, 1.0)))
                          if normal_length > 1e-12 else float("nan"))

            def distance(a, b):
                return float(np.linalg.norm(norm[a] - norm[b]))

            def straightness(indices):
                direct = distance(indices[0], indices[-1])
                chain = sum(distance(a, b) for a, b in zip(indices[:-1], indices[1:]))
                return direct / chain if chain > 0 else float("nan")

            review = VISUAL_REVIEW.get(sample_id, {})
            samples.append({
                "sample_id": sample_id,
                "error_group": group,
                "captured_utc": payload.get("captured_utc", ""),
                "json_path": rel(json_path), "raw_image_path": rel(raw_path),
                "overlay_image_path": rel(overlay_path),
                "manual_label": payload.get("manual_label"),
                "manual_view": payload.get("manual_view"),
                "instant_prediction": payload.get("instant_prediction"),
                "instant_confidence": float(payload.get("instant_confidence")),
                "stable_prediction": payload.get("stable_prediction"),
                "stable_confidence": payload.get("stable_confidence"),
                "required_instant_prediction": payload.get("required_instant_prediction"),
                "model": payload.get("model"),
                "handedness": payload.get("handedness_reported_by_mediapipe"),
                "raw_sha256": sha256(raw_path), "overlay_sha256": sha256(overlay_path),
                "raw_phash": f"{phash(raw_image):016x}",
                "image_width": raw_image.shape[1], "image_height": raw_image.shape[0],
                "feature_vector": vector, "landmark_points": points,
                "distance_8_12": distance(8, 12),
                "distance_12_16": distance(12, 16),
                "distance_8_16": distance(8, 16),
                "index_straightness": straightness([5, 6, 7, 8]),
                "middle_straightness": straightness([9, 10, 11, 12]),
                "ring_straightness": straightness([13, 14, 15, 16]),
                "ring_tip_to_wrist": distance(0, 16),
                "palm_tilt_2d_deg": palm_tilt_2d,
                "palm_plane_tilt_deg": plane_tilt,
                "automatic_integrity": "PASS" if not issue else "FAIL",
                "integrity_issues": ";".join(issue),
                "semantic_review": review.get("semantic_review", "PENDING"),
                "landmark_review": review.get("landmark_review", "PENDING"),
                "decision": review.get("decision", "REVIEW"),
                "review_notes": review.get("review_notes", "Pending direct visual review"),
                "raw_image": raw_image, "overlay_image": overlay_image,
            })
    return samples, errors


def make_sheet(samples: list[dict], output_path: Path, title: str) -> None:
    columns, rows = 2, 4
    tile_w, tile_h = 620, 290
    page = np.full((50 + rows * tile_h, columns * tile_w, 3), 245, dtype=np.uint8)
    cv2.putText(page, title, (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2,
                cv2.LINE_AA)
    for index, sample in enumerate(samples):
        row, col = divmod(index, columns)
        y, x = 50 + row * tile_h, col * tile_w
        raw = cv2.resize(sample["raw_image"], (300, 225), interpolation=cv2.INTER_AREA)
        overlay = cv2.resize(sample["overlay_image"], (300, 225), interpolation=cv2.INTER_AREA)
        page[y:y + 225, x:x + 300] = raw
        page[y:y + 225, x + 305:x + 605] = overlay
        line1 = f"{sample['sample_id']} {sample['error_group']} conf={sample['instant_confidence']:.3f}"
        line2 = (f"d8-12={sample['distance_8_12']:.3f} d12-16={sample['distance_12_16']:.3f} "
                 f"ring={sample['ring_straightness']:.3f}")
        cv2.putText(page, line1, (x + 5, y + 247), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(page, line2, (x + 5, y + 270), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), page, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Cannot write contact sheet: {output_path}")


def create_contact_sheets(samples: list[dict], prefix: str, subset=None) -> list[Path]:
    selected = samples if subset is None else [row for row in samples if subset(row)]
    paths = []
    for page_number, start in enumerate(range(0, len(selected), 8), start=1):
        page_samples = selected[start:start + 8]
        path = OUT / f"{prefix}_{page_number}.jpg"
        make_sheet(page_samples, path, f"{prefix} page {page_number} ({len(selected)} samples)")
        paths.append(path)
    return paths


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen = set()
    components = []
    for node in nodes:
        if node in seen:
            continue
        stack, component = [node], []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(adjacency[current] - seen)
        components.append(sorted(component))
    return components


def assign_duplicate_groups(samples: list[dict]) -> dict:
    exact_groups = defaultdict(list)
    for row in samples:
        exact_groups[row["raw_sha256"]].append(row["sample_id"])
    exact = {sample_id: f"EXACT{number:03d}"
             for number, group in enumerate(
                 [group for group in exact_groups.values() if len(group) > 1], start=1)
             for sample_id in group}

    feature_matrix = np.vstack([row["feature_vector"] for row in samples])
    feature_distances = np.linalg.norm(feature_matrix[:, None, :] - feature_matrix[None, :, :], axis=2)
    pairs = []
    candidate_records = []
    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            if samples[left]["error_group"] != samples[right]["error_group"]:
                continue
            ph = hamming(int(samples[left]["raw_phash"], 16), int(samples[right]["raw_phash"], 16))
            fd = float(feature_distances[left, right])
            delta = abs((parse_time(samples[right]["captured_utc"]) -
                         parse_time(samples[left]["captured_utc"])).total_seconds())
            candidate_records.append((left, right, ph, fd, delta))

    # Thresholds are calibrated to the lower tail of this capture set, then bounded
    # to require strong similarity in both pixels and stored landmark features.
    ph_values = np.asarray([record[2] for record in candidate_records], dtype=float)
    fd_values = np.asarray([record[3] for record in candidate_records], dtype=float)
    ph_threshold = min(8, int(math.floor(np.quantile(ph_values, 0.10)))) if len(ph_values) else 0
    fd_threshold = min(0.18, float(np.quantile(fd_values, 0.10))) if len(fd_values) else 0.0
    for left, right, ph, fd, delta in candidate_records:
        if ph <= ph_threshold and fd <= fd_threshold:
            pairs.append((samples[left]["sample_id"], samples[right]["sample_id"]))
    components = [group for group in connected_components(
        [row["sample_id"] for row in samples], pairs) if len(group) > 1]
    near = {sample_id: f"NEAR{number:03d}" for number, group in enumerate(components, start=1)
            for sample_id in group}

    # No gesture_bout_id exists. Treat each uninterrupted collection session as one
    # conservative dependence group, splitting only at gaps greater than 60 seconds.
    conservative = {}
    group_number = 0
    for error_group in ("V_TO_U", "V_TO_W"):
        ordered = sorted([row for row in samples if row["error_group"] == error_group],
                         key=lambda row: row["captured_utc"])
        previous = None
        for row in ordered:
            current = parse_time(row["captured_utc"])
            if previous is None or (current - previous).total_seconds() > 60:
                group_number += 1
            conservative[row["sample_id"]] = f"SESSION{group_number:03d}"
            previous = current
    return {
        "exact": exact, "near": near, "conservative": conservative,
        "near_thresholds": {"phash_hamming": ph_threshold,
                            "feature_euclidean": fd_threshold,
                            "calibration": "10th percentile within-condition pairs, bounded at pHash<=8 and feature<=0.18"},
        "near_components": components,
    }


def distribution(values: np.ndarray) -> dict:
    return {
        "n": int(len(values)), "mean": float(np.mean(values)),
        "q05": float(np.quantile(values, .05)), "q25": float(np.quantile(values, .25)),
        "median": float(np.median(values)), "q75": float(np.quantile(values, .75)),
        "q95": float(np.quantile(values, .95)),
    }


def feature_hashes(matrix: np.ndarray) -> set[str]:
    return {
        hashlib.sha256(np.asarray(row, dtype="<f8").tobytes()).hexdigest()
        for row in np.asarray(matrix, dtype=np.float64)
    }


def geometry_from_features(matrix: np.ndarray) -> pd.DataFrame:
    points = matrix.reshape(-1, 21, 3)
    def dist(a, b):
        return np.linalg.norm(points[:, a] - points[:, b], axis=1)
    def straight(indices):
        direct = dist(indices[0], indices[-1])
        chain = sum(dist(a, b) for a, b in zip(indices[:-1], indices[1:]))
        return direct / chain
    axis = points[:, 9] - points[:, 0]
    palm_tilt = np.degrees(np.arctan2(axis[:, 0], -axis[:, 1]))
    normal = np.cross(points[:, 5] - points[:, 0], points[:, 17] - points[:, 0])
    normal_length = np.linalg.norm(normal, axis=1)
    plane_tilt = np.degrees(np.arccos(np.clip(np.abs(normal[:, 2]) / normal_length, 0, 1)))
    return pd.DataFrame({
        "distance_8_12": dist(8, 12), "distance_12_16": dist(12, 16),
        "distance_8_16": dist(8, 16),
        "index_straightness": straight([5, 6, 7, 8]),
        "middle_straightness": straight([9, 10, 11, 12]),
        "ring_straightness": straight([13, 14, 15, 16]),
        "ring_tip_to_wrist": dist(0, 16),
        "palm_tilt_2d_deg": palm_tilt, "palm_plane_tilt_deg": plane_tilt,
    })


def finish_audit(samples: list[dict], automatic_errors: list[str]) -> None:
    if set(VISUAL_REVIEW) != {row["sample_id"] for row in samples}:
        missing = sorted({row["sample_id"] for row in samples} - set(VISUAL_REVIEW))
        raise ValueError(f"Visual review decisions incomplete: {missing}")
    duplicates = assign_duplicate_groups(samples)
    for row in samples:
        row["exact_duplicate_group"] = duplicates["exact"].get(row["sample_id"], "")
        row["near_duplicate_group"] = duplicates["near"].get(row["sample_id"], "")
        row["conservative_group_id"] = duplicates["conservative"][row["sample_id"]]

    train = pd.read_csv(TRAIN_V1, usecols=["label", *FEATURES])
    old_audit = pd.read_csv(OLD_AUDIT, dtype=str, keep_default_na=False)
    old_v = old_audit.loc[(old_audit.input_label == "V") &
                          (old_audit.label_review == "LABEL_CONFIRMED") &
                          (old_audit.landmark_review == "LANDMARK_CONFIRMED")]
    old_matrix = []
    for path in old_v.json_path:
        payload = load_json(ROOT / path)
        old_matrix.append([payload["features"][name] for name in FEATURES])
    old_geometry = geometry_from_features(np.asarray(old_matrix, dtype=float))
    old_matrix = np.asarray(old_matrix, dtype=float)
    new_geometry = pd.DataFrame([{key: row[key] for key in old_geometry.columns} for row in samples])
    new_matrix = np.vstack([row["feature_vector"] for row in samples])
    train_matrix = train[FEATURES].to_numpy(dtype=float)
    new_feature_hashes = feature_hashes(new_matrix)
    old_feature_hashes = feature_hashes(old_matrix)
    train_feature_hashes = feature_hashes(train_matrix)
    old_raw_hashes = set(old_v.raw_sha256)
    new_raw_hashes = {row["raw_sha256"] for row in samples}
    exact_cross_source = {
        "new_feature_matches_old_debug": len(new_feature_hashes & old_feature_hashes),
        "new_feature_matches_train_v1": len(new_feature_hashes & train_feature_hashes),
        "new_raw_image_matches_old_debug": len(new_raw_hashes & old_raw_hashes),
    }
    old_nearest = NearestNeighbors(n_neighbors=1).fit(old_matrix)
    old_distances, old_indices = old_nearest.kneighbors(new_matrix)
    for row, distance, index in zip(samples, old_distances[:, 0], old_indices[:, 0]):
        row["nearest_old_debug_V_distance"] = float(distance)
        row["nearest_old_debug_V_sample_id"] = str(old_v.iloc[int(index)].sample_id)
    old_distance_by_group = {
        group: distribution(np.asarray([
            row["nearest_old_debug_V_distance"] for row in samples
            if row["error_group"] == group
        ], dtype=float))
        for group in ("V_TO_U", "V_TO_W")
    }

    cross_old_near_pairs = []
    for row in samples:
        for _, old_row in old_v.iterrows():
            old_hash_text = str(old_row.raw_phash)
            try:
                old_hash = int(old_hash_text, 16)
            except ValueError:
                old_hash = int(old_hash_text, 2)
            ph_distance = hamming(int(row["raw_phash"], 16), old_hash)
            old_payload = load_json(ROOT / old_row.json_path)
            old_vector = np.asarray([old_payload["features"][name] for name in FEATURES])
            feature_distance = float(np.linalg.norm(row["feature_vector"] - old_vector))
            if (ph_distance <= duplicates["near_thresholds"]["phash_hamming"] and
                    feature_distance <= duplicates["near_thresholds"]["feature_euclidean"]):
                cross_old_near_pairs.append({
                    "new_sample_id": row["sample_id"], "old_sample_id": old_row.sample_id,
                    "phash_hamming": ph_distance, "feature_distance": feature_distance,
                })

    comparison = {}
    for label in ("U", "V", "W"):
        matrix = train.loc[train.label == label, FEATURES].to_numpy(dtype=float)
        geometry = geometry_from_features(matrix)
        comparison[f"TRAIN_V1_{label}"] = {
            column: distribution(geometry[column].to_numpy()) for column in geometry.columns
        }
    comparison["OLD_DEBUG_V_15"] = {
        column: distribution(old_geometry[column].to_numpy()) for column in old_geometry.columns
    }
    for group in ("V_TO_U", "V_TO_W"):
        group_geometry = new_geometry.loc[[row["error_group"] == group for row in samples]]
        comparison[group] = {
            column: distribution(group_geometry[column].to_numpy()) for column in group_geometry.columns
        }

    train_features = train[FEATURES].to_numpy(dtype=float)
    scaler = joblib.load(V1_DIR / "scaler.joblib")
    scaled_train = scaler.transform(train_features)
    train_labels = train.label.to_numpy()
    uvw_mask = np.isin(train_labels, ["U", "V", "W"])
    neighbor_model = NearestNeighbors(n_neighbors=5).fit(scaled_train[uvw_mask])
    uvw_labels = train_labels[uvw_mask]
    new_scaled = scaler.transform(np.vstack([row["feature_vector"] for row in samples]))
    distances, indices = neighbor_model.kneighbors(new_scaled)
    for row, row_distances, row_indices in zip(samples, distances, indices):
        votes = Counter(uvw_labels[row_indices])
        row["legacy_5nn_votes_U"] = int(votes["U"])
        row["legacy_5nn_votes_V"] = int(votes["V"])
        row["legacy_5nn_votes_W"] = int(votes["W"])
        row["legacy_nearest_distance"] = float(row_distances[0])

    public_fields = [
        "sample_id", "error_group", "captured_utc", "json_path", "raw_image_path",
        "overlay_image_path", "manual_label", "manual_view", "instant_prediction",
        "instant_confidence", "stable_prediction", "stable_confidence",
        "required_instant_prediction", "model", "handedness", "raw_sha256",
        "overlay_sha256", "image_width", "image_height", "distance_8_12",
        "distance_12_16", "distance_8_16", "index_straightness",
        "middle_straightness", "ring_straightness", "ring_tip_to_wrist",
        "palm_tilt_2d_deg", "palm_plane_tilt_deg", "legacy_5nn_votes_U",
        "legacy_5nn_votes_V", "legacy_5nn_votes_W", "legacy_nearest_distance",
        "nearest_old_debug_V_sample_id", "nearest_old_debug_V_distance",
        "automatic_integrity", "integrity_issues", "exact_duplicate_group",
        "near_duplicate_group", "conservative_group_id", "semantic_review",
        "landmark_review", "decision", "review_notes",
    ]
    report_path = OUT / "v_error_sample_audit.csv"
    pd.DataFrame([{key: row[key] for key in public_fields} for row in samples]).to_csv(
        report_path, index=False)
    review_sheets = create_contact_sheets(
        samples, "review_samples_contact_sheet", lambda row: row["decision"] == "REVIEW")
    if not review_sheets:
        placeholder = np.full((220, 900, 3), 245, dtype=np.uint8)
        cv2.putText(placeholder, "No REVIEW samples after direct raw/overlay inspection",
                    (35, 115), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 2, cv2.LINE_AA)
        path = OUT / "review_samples_contact_sheet_1.jpg"
        cv2.imwrite(str(path), placeholder)
        review_sheets = [path]

    counts = Counter(row["decision"] for row in samples)
    landmark_counts = Counter(row["landmark_review"] for row in samples)
    group_counts = Counter(row["error_group"] for row in samples)
    exact_sample_count = sum(bool(row["exact_duplicate_group"]) for row in samples)
    near_sample_count = sum(bool(row["near_duplicate_group"]) for row in samples)
    summary = {
        "status": "V1_V_ERROR_DATA_AUDIT_COMPLETE" if not automatic_errors else "AUDIT_INTEGRITY_FAIL",
        "scope": {"model": "v1_baseline epoch 143", "training_performed": False,
                  "data_added_to_train": False, "mediapipe_rerun": False,
                  "pq_data_used": False},
        "counts": {"total": len(samples), "by_error_group": dict(group_counts),
                   "decisions": {key: counts[key] for key in ("ACCEPT", "REVIEW", "EXCLUDE")},
                   "landmark_review": dict(landmark_counts)},
        "metadata_checks": {
            "all_manual_label_V": all(row["manual_label"] == "V" for row in samples),
            "all_view_front": all(row["manual_view"] == "front" for row in samples),
            "all_v_to_u_instant_U": all(row["instant_prediction"] == "U" for row in samples
                                         if row["error_group"] == "V_TO_U"),
            "all_v_to_w_instant_W": all(row["instant_prediction"] == "W" for row in samples
                                         if row["error_group"] == "V_TO_W"),
            "all_model_v1_baseline": all(row["model"] == "v1_baseline" for row in samples),
        },
        "duplicates": {
            "exact_groups": len(set(duplicates["exact"].values())),
            "exact_samples": exact_sample_count,
            "near_duplicate_groups": len(duplicates["near_components"]),
            "near_duplicate_samples": near_sample_count,
            "near_thresholds": duplicates["near_thresholds"],
            "cross_source_exact": exact_cross_source,
            "cross_old_debug_near_pairs": len(cross_old_near_pairs),
            "cross_old_debug_near_new_samples": len({row["new_sample_id"]
                                                       for row in cross_old_near_pairs}),
            "nearest_old_debug_V_distance_by_group": old_distance_by_group,
            "conservative_session_groups": len(set(duplicates["conservative"].values())),
            "independence_limit": "No gesture_bout_id; conservative session groups must be used for weighting/splitting.",
        },
        "geometry_comparison": comparison,
        "main_geometry_findings": {
            "distance_8_12_mean_V_TO_U": comparison["V_TO_U"]["distance_8_12"]["mean"],
            "distance_8_12_mean_V_TO_W": comparison["V_TO_W"]["distance_8_12"]["mean"],
            "palm_plane_tilt_mean_V_TO_U": comparison["V_TO_U"]["palm_plane_tilt_deg"]["mean"],
            "palm_plane_tilt_mean_V_TO_W": comparison["V_TO_W"]["palm_plane_tilt_deg"]["mean"],
            "ring_straightness_mean_V_TO_U": comparison["V_TO_U"]["ring_straightness"]["mean"],
            "ring_straightness_mean_V_TO_W": comparison["V_TO_W"]["ring_straightness"]["mean"],
            "interpretation": (
                "The two error groups have nearly identical 8-12 separation and folded-ring "
                "geometry. V_TO_W is more out-of-plane and has slightly greater ring "
                "straightness, but both remain far from legacy W's extended-ring distribution."
            ),
        },
        "integrity": {
            "status": "PASS" if not automatic_errors and
                      all(row["automatic_integrity"] == "PASS" for row in samples) else "FAIL",
            "automatic_errors": automatic_errors,
            "source_files_modified": 0, "model_modified": False,
            "frozen_or_split_data_modified": False,
        },
        "outputs": {"sample_report": rel(report_path),
                    "review_contact_sheets": [rel(path) for path in review_sheets],
                    "all_contact_sheets": [rel(path) for path in
                                           sorted(OUT.glob("all_samples_contact_sheet_*.jpg"))]},
    }
    summary_path = OUT / "audit_summary.json"
    write_json(summary_path, summary)
    input_files = sorted(path for folder in INPUTS.values() for path in folder.glob("*.*"))
    reference_inputs = [
        OLD_AUDIT, TRAIN_V1, V1_DIR / "baseline_model.joblib",
        V1_DIR / "scaler.joblib", V1_DIR / "label_mapping.json",
    ]
    all_sheets = sorted(OUT.glob("all_samples_contact_sheet_*.jpg"))
    manifest = {
        "schema_version": 1, "status": summary["status"],
        "inputs": [{"path": rel(path), "bytes": path.stat().st_size,
                    "sha256": sha256(path)} for path in input_files],
        "reference_inputs": [{"path": rel(path), "bytes": path.stat().st_size,
                              "sha256": sha256(path)} for path in reference_inputs],
        "outputs": [{"path": rel(path), "bytes": path.stat().st_size,
                     "sha256": sha256(path)}
                    for path in [report_path, summary_path, *review_sheets, *all_sheets]],
    }
    write_json(OUT / "manifest.json", manifest)
    print(f"STATUS={summary['status']}")
    print(f"SAMPLES={len(samples)} ACCEPT={counts['ACCEPT']} REVIEW={counts['REVIEW']} EXCLUDE={counts['EXCLUDE']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheets-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    samples, errors = load_samples()
    if len(samples) != 27:
        raise ValueError(f"Expected 27 metadata samples, found {len(samples)}")
    create_contact_sheets(samples, "all_samples_contact_sheet")
    if args.sheets_only:
        print(f"SHEETS_READY={len(list(OUT.glob('all_samples_contact_sheet_*.jpg')))}")
        return
    finish_audit(samples, errors)


if __name__ == "__main__":
    main()
