"""Scalable near-duplicate/variant audit for the canonical Phase-2 dataset.

No image extraction, model training, threshold tuning, or sample deletion is
performed.  Distances use globally standardized f1..f63.  Filename groups and
same-label landmark edges are combined into split-safe connected components;
cross-label matches are reported separately and are never merged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from phase2_review_promotion import FEATURE_COLUMNS, META_COLUMNS, make_group_key


RANDOM_STATE = 42


class UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            next_item = int(self.parent[item])
            self.parent[item] = root
            item = next_item
        return root

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def parse_args() -> argparse.Namespace:
    ai_dir = Path(__file__).resolve().parent
    dataset_dir = ai_dir / "dataset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=dataset_dir
        / "phase2_quality_gate_v2"
        / "dataset_chucai_clean_reviewed_v2.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=dataset_dir
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=dataset_dir / "near_duplicate_checkpoints_v1",
    )
    parser.add_argument("--same-label-neighbors", type=int, default=16)
    parser.add_argument("--cross-label-neighbors", type=int, default=32)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _dataset_signature(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _stable_group_id(label: str, files: list[str], singleton: bool) -> str:
    payload = label + "\x1f" + "\x1f".join(sorted(files))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{'UG' if singleton else 'DG'}-{label}-{digest}"


def _calibrate_distribution(distances: np.ndarray) -> dict:
    clean = distances[np.isfinite(distances)]
    if len(clean) < 100:
        raise ValueError("Too few finite nearest distances for calibration")
    positive = clean[clean > 0]
    epsilon = max(float(positive.min()) * 0.1 if len(positive) else 1e-12, 1e-12)
    logged = np.log(np.maximum(clean, epsilon)).reshape(-1, 1)
    one = GaussianMixture(
        n_components=1, covariance_type="full", random_state=RANDOM_STATE,
        reg_covar=1e-6, n_init=3,
    ).fit(logged)
    two = GaussianMixture(
        n_components=2, covariance_type="full", random_state=RANDOM_STATE,
        reg_covar=1e-6, n_init=5,
    ).fit(logged)
    means = two.means_.ravel()
    low_component = int(np.argmin(means))
    midpoint = float(means.mean())
    q1, q3 = np.quantile(logged.ravel(), [0.25, 0.75])
    iqr = float(q3 - q1)
    # Standard Tukey lower fences on log-distance: the inner fence marks mild
    # lower-tail outliers and the outer fence marks extreme lower-tail outliers.
    # These are distribution-derived distances, not hand-chosen metric values.
    possible_threshold = float(np.exp(q1 - 1.5 * iqr))
    high_threshold = float(np.exp(q1 - 3.0 * iqr))
    metadata = {
        "n": len(clean),
        "bic_1_component": one.bic(logged),
        "bic_2_component": two.bic(logged),
        "two_component_preferred": bool(two.bic(logged) < one.bic(logged)),
        "component_log_means": means.tolist(),
        "component_weights": two.weights_.tolist(),
        "low_component": low_component,
        "log_midpoint": midpoint,
        "epsilon": epsilon,
        "calibration_method": "Tukey lower fences on log nearest-distance",
        "log_q1": float(q1),
        "log_q3": float(q3),
        "log_iqr": iqr,
        "high_threshold": high_threshold,
        "possible_threshold": possible_threshold,
        "observed_high_count": int((clean <= high_threshold).sum()),
        "observed_possible_count": int(
            ((clean > high_threshold) & (clean <= possible_threshold)).sum()
        ),
        "nearest_distance_quantiles": {
            str(q): float(np.quantile(clean, q))
            for q in (0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
        },
    }
    return metadata


def _same_label_neighbors(
    frame: pd.DataFrame,
    scaled: np.ndarray,
    filename_keys: np.ndarray,
    checkpoint_dir: Path,
    base_neighbors: int,
    progress: dict,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    group_sizes = pd.Series(filename_keys).value_counts()
    neighbor_count = max(base_neighbors, int(group_sizes.max()) + 2)
    neighbor_count = min(neighbor_count, 64)
    labels = frame["label"].astype(str).to_numpy()
    cache: dict[str, dict[str, np.ndarray]] = {}
    nearest_any = np.full(len(frame), np.nan, dtype=np.float64)
    nearest_outside = np.full(len(frame), np.nan, dtype=np.float64)

    for number, label in enumerate(sorted(np.unique(labels)), start=1):
        path = checkpoint_dir / f"same_label_{label}.npz"
        indices = np.flatnonzero(labels == label)
        k = min(neighbor_count, len(indices))
        if path.exists():
            saved = np.load(path)
            if not np.array_equal(saved["row_indices"], indices):
                raise RuntimeError(f"Stale same-label checkpoint for {label}")
            distances = saved["distances"]
            neighbors = saved["neighbors"]
            print(f"[same-label {number}/24] {label}: resumed {len(indices):,}", flush=True)
        else:
            model = NearestNeighbors(
                n_neighbors=k, algorithm="brute", metric="euclidean", n_jobs=-1
            ).fit(scaled[indices])
            distances, local_neighbors = model.kneighbors(
                scaled[indices], return_distance=True
            )
            neighbors = indices[local_neighbors]
            np.savez_compressed(
                path,
                row_indices=indices,
                distances=distances.astype(np.float32),
                neighbors=neighbors.astype(np.int32),
            )
            print(
                f"[same-label {number}/24] {label}: computed {len(indices):,}, k={k}",
                flush=True,
            )
        cache[label] = {
            "row_indices": indices,
            "distances": distances,
            "neighbors": neighbors,
        }
        nearest_any[indices] = distances[:, 1] if k > 1 else np.nan
        for row_number, global_index in enumerate(indices):
            current_key = filename_keys[global_index]
            for position in range(1, distances.shape[1]):
                candidate = int(neighbors[row_number, position])
                if filename_keys[candidate] != current_key:
                    nearest_outside[global_index] = float(distances[row_number, position])
                    break
        progress["same_label_completed"] = number
        progress["same_label_last"] = label
        _write_json(checkpoint_dir / "progress.json", progress)
    if np.isnan(nearest_outside).any():
        raise RuntimeError("Could not find an out-of-filename-group neighbor for every row")
    return cache, nearest_any, nearest_outside


def _calibrate_by_label(
    labels: np.ndarray,
    nearest_outside: np.ndarray,
    checkpoint_dir: Path,
) -> dict[str, dict]:
    global_meta = _calibrate_distribution(nearest_outside)
    metadata: dict[str, dict] = {}
    for label in sorted(np.unique(labels)):
        distances = nearest_outside[labels == label]
        meta = _calibrate_distribution(distances)
        meta["calibration_scope"] = "label"
        metadata[label] = meta
    _write_json(
        checkpoint_dir / "calibration.json",
        {"global": global_meta, "by_label": metadata},
    )
    return metadata


def _build_same_label_edges(
    frame: pd.DataFrame,
    scaled: np.ndarray,
    filename_keys: np.ndarray,
    neighbor_cache: dict[str, dict[str, np.ndarray]],
    calibration: dict[str, dict],
) -> list[dict]:
    edges: list[dict] = []
    # A star per filename group is sufficient to make the whole variant set split-safe.
    filename_groups: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(filename_keys):
        filename_groups[str(key)].append(index)
    for members in filename_groups.values():
        if len(members) < 2:
            continue
        anchor = members[0]
        for other in members[1:]:
            distance = float(np.linalg.norm(scaled[anchor] - scaled[other]))
            edges.append(
                {
                    "row_a": anchor,
                    "row_b": other,
                    "label": str(frame.iloc[anchor]["label"]),
                    "edge_type": "EXACT_VARIANT_GROUP",
                    "distance": distance,
                    "threshold_ratio": np.nan,
                    "same_filename_group": True,
                }
            )

    for label, saved in neighbor_cache.items():
        rows = saved["row_indices"]
        distances = saved["distances"]
        neighbors = saved["neighbors"]
        candidate_a = np.repeat(rows, distances.shape[1] - 1)
        candidate_b = neighbors[:, 1:].reshape(-1)
        candidate_distance = distances[:, 1:].reshape(-1).astype(np.float64)
        mask = (
            (candidate_a < candidate_b)
            & (filename_keys[candidate_a] != filename_keys[candidate_b])
        )
        candidate_a = candidate_a[mask]
        candidate_b = candidate_b[mask]
        candidate_distance = candidate_distance[mask]
        possible_threshold = float(calibration[label]["possible_threshold"])
        high_threshold = float(calibration[label]["high_threshold"])
        selected = candidate_distance <= possible_threshold
        for a, b, distance in zip(
            candidate_a[selected],
            candidate_b[selected],
            candidate_distance[selected],
            strict=True,
        ):
            edges.append(
                {
                    "row_a": int(a),
                    "row_b": int(b),
                    "label": label,
                    "edge_type": (
                        "HIGH_CONFIDENCE_NEAR_DUPLICATE"
                        if distance <= high_threshold
                        else "POSSIBLE_NEAR_DUPLICATE"
                    ),
                    "distance": float(distance),
                    "threshold_ratio": float(distance / possible_threshold),
                    "same_filename_group": False,
                }
            )
    # kNN is symmetric often enough to emit the same pair twice; retain the
    # strongest classification and smallest numerically observed distance.
    priority = {
        "POSSIBLE_NEAR_DUPLICATE": 1,
        "HIGH_CONFIDENCE_NEAR_DUPLICATE": 2,
        "EXACT_VARIANT_GROUP": 3,
    }
    unique: dict[tuple[int, int], dict] = {}
    for edge in edges:
        key = (min(edge["row_a"], edge["row_b"]), max(edge["row_a"], edge["row_b"]))
        previous = unique.get(key)
        if previous is None or priority[edge["edge_type"]] > priority[previous["edge_type"]]:
            unique[key] = edge
        elif edge["distance"] < previous["distance"]:
            previous["distance"] = edge["distance"]
    return list(unique.values())


def _cross_label_matches(
    frame: pd.DataFrame,
    scaled: np.ndarray,
    calibration: dict[str, dict],
    components: int,
    neighbors: int,
    batch_size: int,
    checkpoint_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    labels = frame["label"].astype(str).to_numpy()
    pca = PCA(
        n_components=min(components, scaled.shape[1]),
        svd_solver="randomized",
        random_state=RANDOM_STATE,
    )
    projected = pca.fit_transform(scaled).astype(np.float32)
    k = min(neighbors + 1, len(frame))
    index = NearestNeighbors(
        n_neighbors=k, algorithm="kd_tree", leaf_size=40, metric="euclidean",
        n_jobs=-1,
    ).fit(projected)
    matches: dict[tuple[int, int], dict] = {}
    total_batches = math.ceil(len(frame) / batch_size)
    for batch_number, start in enumerate(range(0, len(frame), batch_size), start=1):
        stop = min(start + batch_size, len(frame))
        _, candidate = index.kneighbors(projected[start:stop], return_distance=True)
        a = np.repeat(np.arange(start, stop), k - 1)
        b = candidate[:, 1:].reshape(-1)
        cross = labels[a] != labels[b]
        a, b = a[cross], b[cross]
        ordered = a < b
        a, b = a[ordered], b[ordered]
        if len(a):
            distance = np.linalg.norm(scaled[a] - scaled[b], axis=1)
            threshold = np.array(
                [
                    min(
                        calibration[labels[first]]["possible_threshold"],
                        calibration[labels[second]]["possible_threshold"],
                    )
                    for first, second in zip(a, b, strict=True)
                ],
                dtype=np.float64,
            )
            keep = distance <= threshold
            for first, second, exact_distance, cutoff in zip(
                a[keep], b[keep], distance[keep], threshold[keep], strict=True
            ):
                high_cutoff = min(
                    calibration[labels[first]]["high_threshold"],
                    calibration[labels[second]]["high_threshold"],
                )
                key = (int(first), int(second))
                row = {
                    "row_a": int(first),
                    "row_b": int(second),
                    "label_a": labels[first],
                    "label_b": labels[second],
                    "file_a": frame.iloc[first]["file"],
                    "file_b": frame.iloc[second]["file"],
                    "distance": float(exact_distance),
                    "match_level": (
                        "HIGH_CONFIDENCE_CROSS_LABEL"
                        if exact_distance <= high_cutoff
                        else "POSSIBLE_CROSS_LABEL"
                    ),
                    "possible_threshold": float(cutoff),
                    "high_threshold": float(high_cutoff),
                }
                previous = matches.get(key)
                if previous is None or row["distance"] < previous["distance"]:
                    matches[key] = row
        print(
            f"[cross-label {batch_number}/{total_batches}] rows {start:,}:{stop:,}; matches={len(matches):,}",
            flush=True,
        )
    result = pd.DataFrame(matches.values())
    if result.empty:
        result = pd.DataFrame(
            columns=[
                "row_a", "row_b", "label_a", "label_b", "file_a", "file_b",
                "distance", "match_level", "possible_threshold", "high_threshold",
            ]
        )
    else:
        result = result.sort_values(["distance", "label_a", "label_b"]).reset_index(drop=True)
    result.to_csv(checkpoint_dir / "cross_label_candidates.csv", index=False)
    return result, {
        "method": "PCA randomized projection + exact-distance verification",
        "pca_components": projected.shape[1],
        "pca_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "projected_neighbors_per_sample": neighbors,
        "exact_distance_verified": True,
        "approximate_candidate_recall_not_guaranteed": True,
    }


def main() -> int:
    args = parse_args()
    outputs = {
        "audit": args.output_dir / "near_duplicate_audit.csv",
        "map": args.output_dir / "sample_group_map.csv",
        "cross": args.output_dir / "cross_label_near_matches.csv",
        "summary": args.output_dir / "near_duplicate_summary.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    signature = _dataset_signature(args.dataset)
    checkpoint_signature = {
        "dataset": signature,
        "configuration": {
            "same_label_neighbors": args.same_label_neighbors,
            "cross_label_neighbors": args.cross_label_neighbors,
            "pca_components": args.pca_components,
            "batch_size": args.batch_size,
            "random_state": RANDOM_STATE,
        },
    }
    signature_path = args.checkpoint_dir / "dataset_signature.json"
    if signature_path.exists():
        prior = json.loads(signature_path.read_text(encoding="utf-8"))
        if prior != checkpoint_signature:
            raise RuntimeError("Checkpoint signature/configuration does not match this run")
    else:
        _write_json(signature_path, checkpoint_signature)
    progress = {
        "stage": "load",
        "dataset_signature": signature,
        "same_label_completed": 0,
    }
    _write_json(args.checkpoint_dir / "progress.json", progress)

    frame = pd.read_csv(
        args.dataset, usecols=META_COLUMNS + FEATURE_COLUMNS, low_memory=False
    )
    if len(frame) != 89_428:
        raise ValueError(f"Expected 89,428 rows, found {len(frame):,}")
    if frame["file"].duplicated().any() or frame.duplicated(META_COLUMNS).any():
        raise ValueError("Canonical dataset contains duplicate file/metadata keys")
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(features).all():
        raise ValueError("Canonical f1..f63 contains NaN/Inf")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features).astype(np.float32)
    filename_keys = np.array(
        [
            make_group_key(label, source, file)
            for label, source, file in zip(
                frame["label"], frame["source"], frame["file"], strict=True
            )
        ],
        dtype=object,
    )
    labels = frame["label"].astype(str).to_numpy()
    progress["stage"] = "same_label_neighbors"
    _write_json(args.checkpoint_dir / "progress.json", progress)
    neighbor_cache, nearest_any, nearest_outside = _same_label_neighbors(
        frame,
        scaled,
        filename_keys,
        args.checkpoint_dir,
        args.same_label_neighbors,
        progress,
    )

    progress["stage"] = "calibration"
    _write_json(args.checkpoint_dir / "progress.json", progress)
    calibration = _calibrate_by_label(
        labels, nearest_outside, args.checkpoint_dir
    )
    edges = _build_same_label_edges(
        frame, scaled, filename_keys, neighbor_cache, calibration
    )
    union = UnionFind(len(frame))
    for edge in edges:
        if labels[edge["row_a"]] != labels[edge["row_b"]]:
            raise RuntimeError("Cross-label edge attempted to enter duplicate groups")
        union.union(edge["row_a"], edge["row_b"])
    roots = np.array([union.find(i) for i in range(len(frame))], dtype=np.int64)
    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index, root in enumerate(roots):
        members_by_root[int(root)].append(index)
    edge_types_by_root: dict[int, set[str]] = defaultdict(set)
    for edge in edges:
        edge_types_by_root[int(roots[edge["row_a"]])].add(edge["edge_type"])
    group_id_by_root: dict[int, str] = {}
    group_type_by_root: dict[int, str] = {}
    group_size_by_root: dict[int, int] = {}
    for root, members in members_by_root.items():
        component_labels = set(labels[members])
        if len(component_labels) != 1:
            raise RuntimeError("A duplicate component contains multiple labels")
        label = next(iter(component_labels))
        size = len(members)
        types = edge_types_by_root[root]
        if size == 1:
            group_type = "UNIQUE"
        elif "POSSIBLE_NEAR_DUPLICATE" in types:
            group_type = "POSSIBLE_NEAR_DUPLICATE"
        elif "HIGH_CONFIDENCE_NEAR_DUPLICATE" in types:
            group_type = "HIGH_CONFIDENCE_NEAR_DUPLICATE"
        else:
            group_type = "EXACT_VARIANT_GROUP"
        group_id_by_root[root] = _stable_group_id(
            label, frame.iloc[members]["file"].astype(str).tolist(), size == 1
        )
        group_type_by_root[root] = group_type
        group_size_by_root[root] = size

    group_ids = np.array([group_id_by_root[int(root)] for root in roots])
    group_types = np.array([group_type_by_root[int(root)] for root in roots])
    group_sizes = np.array([group_size_by_root[int(root)] for root in roots])
    filename_sizes = pd.Series(filename_keys).map(
        pd.Series(filename_keys).value_counts()
    ).to_numpy()
    sample_map = frame[META_COLUMNS].copy()
    sample_map["filename_group_key"] = filename_keys
    sample_map["filename_group_size"] = filename_sizes
    sample_map["group_id"] = group_ids
    sample_map["group_type"] = group_types
    sample_map["group_size"] = group_sizes
    sample_map["nearest_distance"] = nearest_any
    sample_map["nearest_outside_filename_group_distance"] = nearest_outside
    sample_map["duplicate_status"] = group_types

    edge_frame = pd.DataFrame(edges)
    if edge_frame.empty:
        edge_frame = pd.DataFrame(
            columns=[
                "row_a", "row_b", "label", "edge_type", "distance",
                "threshold_ratio", "same_filename_group",
            ]
        )
    edge_frame["file_a"] = edge_frame["row_a"].map(frame["file"])
    edge_frame["file_b"] = edge_frame["row_b"].map(frame["file"])
    edge_frame["group_id"] = edge_frame["row_a"].map(
        lambda index: group_ids[int(index)]
    )
    edge_frame["high_threshold"] = edge_frame["label"].map(
        lambda label: calibration[str(label)]["high_threshold"]
    )
    edge_frame["possible_threshold"] = edge_frame["label"].map(
        lambda label: calibration[str(label)]["possible_threshold"]
    )
    edge_frame = edge_frame[
        [
            "group_id", "label", "file_a", "file_b", "edge_type", "distance",
            "threshold_ratio", "same_filename_group", "high_threshold",
            "possible_threshold",
        ]
    ].sort_values(["label", "group_id", "distance"])

    progress["stage"] = "cross_label"
    _write_json(args.checkpoint_dir / "progress.json", progress)
    cross_label, cross_method = _cross_label_matches(
        frame,
        scaled,
        calibration,
        args.pca_components,
        args.cross_label_neighbors,
        args.batch_size,
        args.checkpoint_dir,
    )

    group_table = sample_map.drop_duplicates("group_id")
    size_distribution = (
        group_table["group_size"].value_counts().sort_index().to_dict()
    )
    filename_group_counts = pd.Series(filename_keys).value_counts()
    per_label: dict[str, dict] = {}
    for label, rows in sample_map.groupby("label"):
        groups = rows.drop_duplicates("group_id")
        per_label[str(label)] = {
            "samples": len(rows),
            "groups": len(groups),
            "singletons": int((groups["group_size"] == 1).sum()),
            "multi_sample_groups": int((groups["group_size"] > 1).sum()),
            "samples_in_filename_variant_groups": int(
                (rows["filename_group_size"] > 1).sum()
            ),
            "group_type_counts": groups["group_type"].value_counts().to_dict(),
            "sample_status_counts": rows["duplicate_status"].value_counts().to_dict(),
            "high_threshold": calibration[str(label)]["high_threshold"],
            "possible_threshold": calibration[str(label)]["possible_threshold"],
        }
    summary = {
        "status": "AUDIT_COMPLETE_NO_DELETION",
        "dataset": signature,
        "method": {
            "feature_space": "StandardScaler(f1..f63) + Euclidean distance",
            "same_label_search": "exact brute-force kNN within each label",
            "same_label_neighbors": args.same_label_neighbors,
            "threshold_calibration": (
                "Per-label Tukey lower fences on log nearest-distance outside "
                "filename groups: outer 3*IQR=HIGH, inner 1.5*IQR=POSSIBLE. "
                "1-vs-2 component GMM BIC is retained as a distribution diagnostic."
            ),
            "filename_group": (
                "Phase-2 label+source+parent+normalized-stem key; only terminal "
                "(2)..(9) and - Copy removed"
            ),
            "cross_label": cross_method,
            "random_state": RANDOM_STATE,
        },
        "counts": {
            "samples": len(frame),
            "groups": len(group_table),
            "singletons": int((group_table["group_size"] == 1).sum()),
            "multi_sample_groups": int((group_table["group_size"] > 1).sum()),
            "filename_variant_groups": int((filename_group_counts > 1).sum()),
            "samples_in_filename_variant_groups": int(
                filename_group_counts[filename_group_counts > 1].sum()
            ),
            "landmark_near_duplicate_groups": int(
                group_table["group_type"].isin(
                    ["HIGH_CONFIDENCE_NEAR_DUPLICATE", "POSSIBLE_NEAR_DUPLICATE"]
                ).sum()
            ),
            "high_confidence_groups": int(
                group_table["group_type"].eq(
                    "HIGH_CONFIDENCE_NEAR_DUPLICATE"
                ).sum()
            ),
            "possible_groups": int(
                group_table["group_type"].eq("POSSIBLE_NEAR_DUPLICATE").sum()
            ),
            "cross_label_near_match_pairs": len(cross_label),
            "cross_label_near_match_samples": int(
                len(set(cross_label.get("file_a", [])) | set(cross_label.get("file_b", [])))
            ),
        },
        "group_size_distribution": size_distribution,
        "edge_type_counts": edge_frame["edge_type"].value_counts().to_dict(),
        "cross_label_level_counts": cross_label.get(
            "match_level", pd.Series(dtype=str)
        ).value_counts().to_dict(),
        "calibration_by_label": calibration,
        "per_label": per_label,
        "special_labels": {
            label: per_label[label] for label in ("A", "B", "P", "W", "M", "N", "T")
        },
        "integrity": {
            "cross_label_groups": 0,
            "all_samples_have_one_group_id": bool(sample_map["group_id"].notna().all()),
            "duplicate_file_input": 0,
            "source_dataset_modified": False,
            "samples_deleted": 0,
        },
    }

    temporary_outputs = {
        name: path.with_suffix(path.suffix + ".tmp")
        for name, path in outputs.items()
    }
    edge_frame.to_csv(temporary_outputs["audit"], index=False)
    sample_map.to_csv(temporary_outputs["map"], index=False)
    cross_label.to_csv(temporary_outputs["cross"], index=False)
    temporary_outputs["summary"].write_text(
        json.dumps(_json_ready(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # Read-back validation before exposing any final output path.
    if len(pd.read_csv(temporary_outputs["map"])) != len(frame):
        raise RuntimeError("sample_group_map read-back row count failed")
    if len(pd.read_csv(temporary_outputs["audit"])) != len(edge_frame):
        raise RuntimeError("near_duplicate_audit read-back row count failed")
    if len(pd.read_csv(temporary_outputs["cross"])) != len(cross_label):
        raise RuntimeError("cross_label_near_matches read-back row count failed")
    for name, path in outputs.items():
        os.replace(temporary_outputs[name], path)
    progress["stage"] = "complete"
    progress["outputs"] = {name: str(path.resolve()) for name, path in outputs.items()}
    _write_json(args.checkpoint_dir / "progress.json", progress)
    print(json.dumps(_json_ready(summary["counts"]), indent=2), flush=True)
    print(f"Outputs: {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
