"""Build a conservative, resumable quality report for the static-letter dataset.

This script never modifies source images or the three extraction CSV files.  It
writes checkpoints under ``AI/quality_checkpoints`` and publishes new reports
under ``AI/dataset``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

AI_DIR = Path(__file__).resolve().parent
PROJECT_DIR = AI_DIR.parent
sys.path.insert(0, str(AI_DIR / "preprocessing"))

from landmark_utils import normalize_landmarks  # noqa: E402


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
DATASET_DIR = AI_DIR / "dataset"
RAW_DIR = AI_DIR / "raw_data"
CHECKPOINT_DIR = AI_DIR / "quality_checkpoints"

DATASET_FILE = DATASET_DIR / "dataset_chucai_all.csv"
RAW_FILE = RAW_DIR / "raw_landmarks_chucai_all.csv"
REJECTED_FILE = DATASET_DIR / "rejected_chucai.csv"
SOURCE_DIR = PROJECT_DIR / "dulieu" / "chucai"
MODEL_FILE = PROJECT_DIR / "models" / "mediapipe" / "hand_landmarker.task"

GEOMETRY_CHECKPOINT = CHECKPOINT_DIR / "geometry_quality.csv"
IMAGE_CHECKPOINT = CHECKPOINT_DIR / "image_quality.csv"
STABILITY_CHECKPOINT = CHECKPOINT_DIR / "landmark_stability.csv"
CHECKPOINT_META = CHECKPOINT_DIR / "checkpoint_meta.json"

REPORT_FILE = DATASET_DIR / "landmark_quality_report.csv"
REVIEW_FILE = DATASET_DIR / "review_chucai.csv"
EXCLUDED_FILE = DATASET_DIR / "excluded_chucai.csv"
CLEAN_FILE = DATASET_DIR / "dataset_chucai_clean.csv"
SUMMARY_FILE = DATASET_DIR / "landmark_quality_summary.json"

VALID_LABELS = tuple("ABCDEFGHIKLMNOPQRSTUVWXY")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONNECTIONS = np.asarray(
    [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ],
    dtype=np.int64,
)

PIPELINE_VERSION = 2
MIN_REFERENCE_GROUP = 200
SUSPICIOUS_Z = 4.0
BAD_Z = 6.0
PROGRESS_EVERY = 1000
IMAGE_FLUSH_EVERY = 500
STABILITY_FLUSH_EVERY = 100


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def source_signature() -> dict:
    return {
        str(path.relative_to(PROJECT_DIR)): {
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in (DATASET_FILE, RAW_FILE, REJECTED_FILE, MODEL_FILE)
    }


def load_meta() -> dict:
    if not CHECKPOINT_META.exists():
        return {}
    try:
        return json.loads(CHECKPOINT_META.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def checkpoint_is_current(meta: dict) -> bool:
    return (
        meta.get("pipeline_version") == PIPELINE_VERSION
        and meta.get("source_signature") == source_signature()
    )


def save_meta(meta: dict) -> None:
    meta["pipeline_version"] = PIPELINE_VERSION
    meta["source_signature"] = source_signature()
    atomic_json(meta, CHECKPOINT_META)


def robust_abs_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return result
    sample = values[finite]
    median = np.median(sample)
    mad = np.median(np.abs(sample - median))
    if mad < 1e-12:
        result[finite] = 0.0
    else:
        result[finite] = np.abs(0.67448975 * (sample - median) / mad)
    return result


def robust_low_z(values: np.ndarray, log_transform: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return result
    sample = values[finite]
    if log_transform:
        sample = np.log1p(np.maximum(sample, 0.0))
    median = np.median(sample)
    mad = np.median(np.abs(sample - median))
    if mad < 1e-12:
        result[finite] = 0.0
    else:
        result[finite] = np.maximum(
            0.0, 0.67448975 * (median - sample) / mad
        )
    return result


def grouped_robust_scores(
    frame: pd.DataFrame,
    columns: list[str],
    mode: str = "absolute",
    log_columns: set[str] | None = None,
) -> pd.DataFrame:
    """Score within label+handedness, falling back to label for small groups."""
    log_columns = log_columns or set()
    scored = pd.DataFrame(index=frame.index)
    for column in columns:
        scored[f"z_{column}"] = np.nan

    label_groups = {label: group for label, group in frame.groupby("label")}
    for (_, _), group in frame.groupby(["label", "handedness"], dropna=False):
        reference = group
        if len(group) < MIN_REFERENCE_GROUP:
            reference = label_groups[group["label"].iloc[0]]
        for column in columns:
            reference_values = reference[column].to_numpy(dtype=np.float64)
            target_values = group[column].to_numpy(dtype=np.float64)
            finite_reference = reference_values[np.isfinite(reference_values)]
            if len(finite_reference) == 0:
                continue
            transform = column in log_columns
            if transform:
                finite_reference = np.log1p(np.maximum(finite_reference, 0.0))
                target_values = np.log1p(np.maximum(target_values, 0.0))
            median = np.median(finite_reference)
            mad = np.median(np.abs(finite_reference - median))
            if mad < 1e-12:
                values = np.zeros(len(group), dtype=np.float64)
            elif mode == "low":
                values = np.maximum(
                    0.0, 0.67448975 * (median - target_values) / mad
                )
            else:
                values = np.abs(
                    0.67448975 * (target_values - median) / mad
                )
            scored.loc[group.index, f"z_{column}"] = values
    return scored


def discover_images() -> pd.DataFrame:
    rows: list[dict] = []
    for label_dir in sorted(SOURCE_DIR.iterdir(), key=lambda p: p.name.upper()):
        label = label_dir.name.upper()
        if not label_dir.is_dir() or label not in VALID_LABELS:
            continue
        for path in label_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append(
                    {
                        "file": str(path.relative_to(PROJECT_DIR)),
                        "label": label,
                    }
                )
    return pd.DataFrame(rows)


def validate_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (SOURCE_DIR, DATASET_FILE, RAW_FILE, REJECTED_FILE, MODEL_FILE):
        if not path.exists():
            raise FileNotFoundError(path)

    print("[A] Đọc và kiểm tra nguồn...")
    dataset = pd.read_csv(DATASET_FILE)
    raw = pd.read_csv(RAW_FILE)
    rejected = pd.read_csv(REJECTED_FILE)
    manifest = discover_images()
    feature_columns = [f"f{i}" for i in range(1, 64)]
    raw_columns = [f"{axis}{i}" for i in range(21) for axis in "xyz"]

    required_dataset = {"source", "file", "label", "handedness", *feature_columns}
    required_raw = {"source", "file", "label", "handedness", *raw_columns}
    if not required_dataset.issubset(dataset.columns):
        raise ValueError("dataset_chucai_all.csv thiếu cột bắt buộc")
    if not required_raw.issubset(raw.columns):
        raise ValueError("raw_landmarks_chucai_all.csv thiếu cột bắt buộc")
    if not {"file", "label", "reason"}.issubset(rejected.columns):
        raise ValueError("rejected_chucai.csv thiếu cột bắt buộc")
    if len(dataset) != len(raw):
        raise ValueError("Dataset normalized và raw khác số dòng")
    key_columns = ["file", "label", "handedness"]
    if not dataset[key_columns].equals(raw[key_columns]):
        raise ValueError("Dataset normalized và raw không cùng thứ tự/key")
    if dataset["file"].duplicated().any() or raw["file"].duplicated().any():
        raise ValueError("Có đường dẫn trùng trong dataset landmark hợp lệ")
    if rejected["file"].duplicated().any():
        raise ValueError("Có đường dẫn trùng trong rejected_chucai.csv")

    feature_values = dataset[feature_columns].to_numpy(dtype=np.float64)
    raw_values = raw[raw_columns].to_numpy(dtype=np.float64)
    invalid_feature = ~np.isfinite(feature_values).all(axis=1)
    invalid_raw = ~np.isfinite(raw_values).all(axis=1)
    dataset["hard_invalid_feature"] = invalid_feature | invalid_raw

    known = set(dataset["file"]) | set(rejected["file"])
    on_disk = set(manifest["file"])
    if known != on_disk:
        raise ValueError(
            f"Manifest không khớp CSV: thiếu={len(on_disk-known)}, "
            f"không tồn tại={len(known-on_disk)}"
        )
    if set(dataset["file"]) & set(rejected["file"]):
        raise ValueError("Một số file vừa hợp lệ vừa nằm trong rejected")
    if set(dataset["label"]) != set(VALID_LABELS):
        raise ValueError("Dataset không chứa đúng 24 lớp chữ cái tĩnh")

    print(
        f"    ảnh gốc={len(manifest):,}, landmark hợp lệ={len(dataset):,}, "
        f"rejected extraction={len(rejected):,}"
    )
    return dataset, raw, rejected, manifest


def build_geometry(dataset: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    print("[B] Tính geometry cho toàn bộ landmark hợp lệ...")
    feature_columns = [f"f{i}" for i in range(1, 64)]
    points = dataset[feature_columns].to_numpy(dtype=np.float64).reshape(-1, 21, 3)

    bone_vectors = points[:, CONNECTIONS[:, 1], :] - points[:, CONNECTIONS[:, 0], :]
    bone_lengths = np.linalg.norm(bone_vectors, axis=2)
    pair_distances = []
    for first in range(21):
        for second in range(first + 1, 21):
            pair_distances.append(
                np.linalg.norm(points[:, first, :] - points[:, second, :], axis=1)
            )
    pair_distances_array = np.column_stack(pair_distances)

    raw_x = raw[[f"x{i}" for i in range(21)]].to_numpy(dtype=np.float64)
    raw_y = raw[[f"y{i}" for i in range(21)]].to_numpy(dtype=np.float64)
    mean_bone = bone_lengths.mean(axis=1)
    metrics = pd.DataFrame(
        {
            "file": dataset["file"],
            "label": dataset["label"],
            "handedness": dataset["handedness"],
            "min_pair": pair_distances_array.min(axis=1),
            "very_close_pairs": (pair_distances_array < 0.03).sum(axis=1),
            "min_bone": bone_lengths.min(axis=1),
            "tiny_bones": (bone_lengths < 0.05).sum(axis=1),
            "bone_cv": np.divide(
                bone_lengths.std(axis=1),
                mean_bone,
                out=np.zeros_like(mean_bone),
                where=mean_bone > 0,
            ),
            "palm_width": np.linalg.norm(points[:, 5, :] - points[:, 17, :], axis=1),
            "wrist_middle": np.linalg.norm(points[:, 0, :] - points[:, 9, :], axis=1),
            "mcp_chain": (
                np.linalg.norm(points[:, 5, :] - points[:, 9, :], axis=1)
                + np.linalg.norm(points[:, 9, :] - points[:, 13, :], axis=1)
                + np.linalg.norm(points[:, 13, :] - points[:, 17, :], axis=1)
            ),
            "spread_3d": np.linalg.norm(points.max(axis=1) - points.min(axis=1), axis=1),
            "raw_bbox_width": raw_x.max(axis=1) - raw_x.min(axis=1),
            "raw_bbox_height": raw_y.max(axis=1) - raw_y.min(axis=1),
            "outside_landmarks": (
                (raw_x < 0) | (raw_x > 1) | (raw_y < 0) | (raw_y > 1)
            ).sum(axis=1),
        }
    )
    score_columns = [
        "min_pair", "min_bone", "bone_cv", "palm_width", "wrist_middle",
        "mcp_chain", "spread_3d", "raw_bbox_width", "raw_bbox_height",
    ]
    z_scores = grouped_robust_scores(metrics, score_columns)
    metrics = pd.concat([metrics, z_scores], axis=1)
    z_columns = [f"z_{column}" for column in score_columns]
    z_values = metrics[z_columns].to_numpy(dtype=np.float64)
    sorted_z = np.sort(np.nan_to_num(z_values, nan=0.0), axis=1)
    metrics["geometry_score"] = sorted_z[:, -3:].mean(axis=1)
    metrics["geometry_evidence_count"] = (z_values >= BAD_Z).sum(axis=1)
    metrics["geometry_warning_count"] = (z_values >= SUSPICIOUS_Z).sum(axis=1)
    metrics["bad_geometry"] = (
        (metrics["geometry_score"] >= BAD_Z)
        & (metrics["geometry_evidence_count"] >= 2)
    )
    metrics["geometry_candidate"] = (
        (metrics["geometry_score"] >= SUSPICIOUS_Z)
        | (metrics["geometry_warning_count"] >= 2)
        | (metrics["very_close_pairs"] >= 2)
        | (metrics["tiny_bones"] >= 2)
        | (metrics["outside_landmarks"] > 0)
    )
    atomic_csv(metrics, GEOMETRY_CHECKPOINT)
    print(
        f"    candidate geometry={metrics['geometry_candidate'].sum():,}, "
        f"bad_geometry={metrics['bad_geometry'].sum():,}"
    )
    return metrics


def blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def analyze_one_image(file_name: str, raw_row: pd.Series) -> dict:
    image = cv2.imread(str(PROJECT_DIR / Path(file_name)))
    if image is None or image.size == 0:
        return {
            "file": file_name,
            "image_read_error": True,
            "blur_full": np.nan,
            "blur_hand": np.nan,
            "hand_area": np.nan,
            "dark_fraction": np.nan,
            "bright_fraction": np.nan,
        }
    height, width = image.shape[:2]
    xs = np.asarray([raw_row[f"x{i}"] for i in range(21)], dtype=np.float64)
    ys = np.asarray([raw_row[f"y{i}"] for i in range(21)], dtype=np.float64)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    hand_area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    px_min, px_max = int(x_min * width), int(x_max * width)
    py_min, py_max = int(y_min * height), int(y_max * height)
    margin_x = int(max(1, px_max - px_min) * 0.25)
    margin_y = int(max(1, py_max - py_min) * 0.25)
    px_min, px_max = max(0, px_min - margin_x), min(width, px_max + margin_x)
    py_min, py_max = max(0, py_min - margin_y), min(height, py_max + margin_y)
    crop = image[py_min:py_max, px_min:px_max]
    if crop.size == 0 or min(crop.shape[:2]) < 10:
        crop_blur = np.nan
        dark_fraction = np.nan
        bright_fraction = np.nan
    else:
        crop_blur = blur_score(crop)
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        dark_fraction = float((gray_crop <= 5).mean())
        bright_fraction = float((gray_crop >= 250).mean())
    return {
        "file": file_name,
        "image_read_error": False,
        "blur_full": blur_score(image),
        "blur_hand": crop_blur,
        "hand_area": hand_area,
        "dark_fraction": dark_fraction,
        "bright_fraction": bright_fraction,
    }


def build_image_quality(dataset: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    print("[C] Đọc tuần tự ảnh và tính image quality...")
    completed = pd.DataFrame()
    if IMAGE_CHECKPOINT.exists():
        completed = pd.read_csv(IMAGE_CHECKPOINT)
        measurement_columns = [
            "file", "image_read_error", "blur_full", "blur_hand", "hand_area",
            "dark_fraction", "bright_fraction",
        ]
        if set(measurement_columns).issubset(completed.columns):
            completed = completed[measurement_columns].copy()
        else:
            completed = pd.DataFrame()
    completed_files = set(completed.get("file", pd.Series(dtype=str)))
    raw_index = raw.set_index("file", drop=False)
    pending_rows: list[dict] = []
    started = time.time()
    total = len(dataset)
    for number, row in enumerate(dataset.itertuples(index=False), start=1):
        if row.file not in completed_files:
            pending_rows.append(analyze_one_image(row.file, raw_index.loc[row.file]))
        if len(pending_rows) >= IMAGE_FLUSH_EVERY:
            completed = pd.concat([completed, pd.DataFrame(pending_rows)], ignore_index=True)
            atomic_csv(completed, IMAGE_CHECKPOINT)
            pending_rows.clear()
        if number % PROGRESS_EVERY == 0 or number == total:
            elapsed = max(time.time() - started, 1e-6)
            print(f"    image {number:,}/{total:,} ({number/elapsed:.1f} ảnh/s)")
    if pending_rows:
        completed = pd.concat([completed, pd.DataFrame(pending_rows)], ignore_index=True)
        atomic_csv(completed, IMAGE_CHECKPOINT)
    completed = completed.drop_duplicates("file", keep="last")
    completed = dataset[["file", "label", "handedness"]].merge(
        completed, on="file", how="left", validate="one_to_one"
    )

    quality_columns = ["blur_hand", "hand_area"]
    low_scores = grouped_robust_scores(
        completed, quality_columns, mode="low", log_columns=set(quality_columns)
    )
    completed = pd.concat([completed, low_scores], axis=1)
    completed["z_exposure"] = np.maximum(
        robust_abs_z(completed["dark_fraction"].to_numpy()),
        robust_abs_z(completed["bright_fraction"].to_numpy()),
    )
    image_z = completed[["z_blur_hand", "z_hand_area", "z_exposure"]].to_numpy()
    # Exposure varies strongly with background and cannot by itself make an image
    # suspicious.  The ranking score therefore reflects blur/hand area, while
    # exposure remains supporting evidence for the two-signal bad flag.
    completed["image_quality_score"] = completed[
        ["z_blur_hand", "z_hand_area"]
    ].max(axis=1)
    completed["image_evidence_count"] = (image_z >= BAD_Z).sum(axis=1)
    completed["image_warning_count"] = (image_z >= SUSPICIOUS_Z).sum(axis=1)
    completed["bad_image_quality"] = (
        (completed["image_quality_score"] >= BAD_Z)
        & (completed["image_evidence_count"] >= 2)
    )
    completed["image_candidate"] = (
        completed["image_read_error"].fillna(True)
        | (completed["image_quality_score"] >= SUSPICIOUS_Z)
        | (completed["image_warning_count"] >= 2)
    )
    atomic_csv(completed, IMAGE_CHECKPOINT)
    print(
        f"    candidate image={completed['image_candidate'].sum():,}, "
        f"bad_image_quality={completed['bad_image_quality'].sum():,}, "
        f"read_error={completed['image_read_error'].sum():,}"
    )
    return completed


def image_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("original", image),
        ("brightness_plus", np.clip(image.astype(np.int16) + 20, 0, 255).astype(np.uint8)),
        ("brightness_minus", np.clip(image.astype(np.int16) - 20, 0, 255).astype(np.uint8)),
        ("contrast_plus", cv2.convertScaleAbs(image, alpha=1.10, beta=0)),
        ("contrast_minus", cv2.convertScaleAbs(image, alpha=0.90, beta=0)),
    ]


def detect_normalized(landmarker, image: np.ndarray) -> np.ndarray | None:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks or len(result.hand_landmarks[0]) != 21:
        return None
    features = np.asarray(
        normalize_landmarks(result.hand_landmarks[0]), dtype=np.float64
    )
    if features.shape != (63,) or not np.isfinite(features).all():
        return None
    return features.reshape(21, 3)


def analyze_stability_one(landmarker, file_name: str, baseline: np.ndarray) -> dict:
    image = cv2.imread(str(PROJECT_DIR / Path(file_name)))
    if image is None or image.size == 0:
        return {
            "file": file_name,
            "stability_tested": True,
            "stability_failure_count": 5,
            "failure_rate": 1.0,
            "mean_drift": np.nan,
            "max_drift": np.nan,
            "variant_evidence": "IMAGE_READ_ERROR",
        }
    mean_drifts: list[float] = []
    max_drifts: list[float] = []
    failures: list[str] = []
    evidence: list[str] = []
    for variant_name, variant in image_variants(image):
        detected = detect_normalized(landmarker, variant)
        if detected is None:
            failures.append(variant_name)
            continue
        point_drift = np.linalg.norm(detected - baseline, axis=1)
        mean_drifts.append(float(point_drift.mean()))
        max_drifts.append(float(point_drift.max()))
        evidence.append(
            f"{variant_name}:mean={point_drift.mean():.6f},max={point_drift.max():.6f}"
        )
    if failures:
        evidence.append("failed=" + ",".join(failures))
    return {
        "file": file_name,
        "stability_tested": True,
        "stability_failure_count": len(failures),
        "failure_rate": len(failures) / 5.0,
        "mean_drift": float(np.mean(mean_drifts)) if mean_drifts else np.nan,
        "max_drift": float(np.max(max_drifts)) if max_drifts else np.nan,
        "variant_evidence": " | ".join(evidence),
    }


def build_stability(
    dataset: pd.DataFrame, geometry: pd.DataFrame, image_quality: pd.DataFrame
) -> pd.DataFrame:
    candidates = geometry[["file", "geometry_candidate"]].merge(
        image_quality[["file", "image_candidate", "image_read_error"]],
        on="file", validate="one_to_one"
    )
    candidates = candidates[
        candidates["geometry_candidate"] | candidates["image_candidate"]
    ].copy()
    candidate_files = set(candidates["file"])
    print(f"[D] Stability test trên {len(candidate_files):,} candidate...")

    completed = pd.DataFrame()
    if STABILITY_CHECKPOINT.exists():
        completed = pd.read_csv(STABILITY_CHECKPOINT)
        measurement_columns = [
            "file", "stability_tested", "stability_failure_count", "failure_rate",
            "mean_drift", "max_drift", "variant_evidence",
        ]
        if set(measurement_columns).issubset(completed.columns):
            completed = completed[measurement_columns].copy()
        else:
            completed = pd.DataFrame()
    completed_files = set(completed.get("file", pd.Series(dtype=str)))
    feature_columns = [f"f{i}" for i in range(1, 64)]
    dataset_index = dataset.set_index("file", drop=False)
    pending_rows: list[dict] = []
    ordered_candidates = [name for name in dataset["file"] if name in candidate_files]
    base_options = python.BaseOptions(model_asset_path=str(MODEL_FILE))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    started = time.time()
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for number, file_name in enumerate(ordered_candidates, start=1):
            if file_name not in completed_files:
                baseline = dataset_index.loc[file_name, feature_columns].to_numpy(
                    dtype=np.float64
                ).reshape(21, 3)
                pending_rows.append(
                    analyze_stability_one(landmarker, file_name, baseline)
                )
            if len(pending_rows) >= STABILITY_FLUSH_EVERY:
                completed = pd.concat(
                    [completed, pd.DataFrame(pending_rows)], ignore_index=True
                )
                atomic_csv(completed, STABILITY_CHECKPOINT)
                pending_rows.clear()
            if number % 100 == 0 or number == len(ordered_candidates):
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"    stability {number:,}/{len(ordered_candidates):,} "
                    f"({number/elapsed:.1f} candidate/s)"
                )
    if pending_rows:
        completed = pd.concat([completed, pd.DataFrame(pending_rows)], ignore_index=True)
    completed = completed[completed["file"].isin(candidate_files)]
    completed = completed.drop_duplicates("file", keep="last")

    metadata = dataset[["file", "label", "handedness"]].merge(
        completed, on="file", how="right", validate="one_to_one"
    )
    drift_scores = grouped_robust_scores(
        metadata, ["mean_drift", "max_drift"], mode="absolute"
    )
    metadata = pd.concat([metadata, drift_scores], axis=1)
    metadata["stability_score"] = metadata[
        ["z_mean_drift", "z_max_drift"]
    ].max(axis=1)
    metadata["stability_evidence_count"] = (
        metadata[["z_mean_drift", "z_max_drift"]] >= BAD_Z
    ).sum(axis=1) + (metadata["failure_rate"] >= 0.4).astype(int)
    metadata["bad_stability"] = (
        (metadata["stability_failure_count"] >= 2)
        | (
            (metadata["stability_score"] >= BAD_Z)
            & (metadata["stability_evidence_count"] >= 2)
        )
    )
    metadata["stability_candidate"] = (
        (metadata["stability_failure_count"] >= 1)
        | (metadata["stability_score"] >= SUSPICIOUS_Z)
    )
    atomic_csv(metadata, STABILITY_CHECKPOINT)
    print(
        f"    bad_stability={metadata['bad_stability'].sum():,}, "
        f"có ít nhất một detection failure="
        f"{(metadata['stability_failure_count'] > 0).sum():,}"
    )
    return metadata


def evidence_text(row: pd.Series) -> str:
    evidence: list[str] = []
    if bool(row.get("hard_invalid_feature", False)):
        evidence.append("feature/raw contains NaN, Inf or invalid value")
    if bool(row.get("image_read_error", False)):
        evidence.append("image cannot be read during quality pass")
    if bool(row.get("bad_geometry", False)):
        evidence.append(
            f"geometry_score={row.get('geometry_score', np.nan):.4f};"
            f"evidence={int(row.get('geometry_evidence_count', 0))}"
        )
    elif bool(row.get("geometry_candidate", False)):
        evidence.append(
            f"geometry_warning:score={row.get('geometry_score', np.nan):.4f};"
            f"warnings={int(row.get('geometry_warning_count', 0))}"
        )
    if bool(row.get("bad_image_quality", False)):
        evidence.append(
            f"image_quality_score={row.get('image_quality_score', np.nan):.4f};"
            f"evidence={int(row.get('image_evidence_count', 0))}"
        )
    elif bool(row.get("image_candidate", False)):
        evidence.append(
            f"image_warning:score={row.get('image_quality_score', np.nan):.4f};"
            f"warnings={int(row.get('image_warning_count', 0))}"
        )
    if bool(row.get("bad_stability", False)):
        evidence.append(
            f"stability_score={row.get('stability_score', np.nan):.4f};"
            f"failures={int(row.get('stability_failure_count', 0))}/5;"
            f"mean_drift={row.get('mean_drift', np.nan):.6f};"
            f"max_drift={row.get('max_drift', np.nan):.6f}"
        )
    elif bool(row.get("stability_candidate", False)):
        evidence.append(
            f"stability_warning:score={row.get('stability_score', np.nan):.4f};"
            f"failures={int(row.get('stability_failure_count', 0))}/5"
        )
    return " | ".join(evidence)


def combine_reports(
    dataset: pd.DataFrame,
    rejected: pd.DataFrame,
    manifest: pd.DataFrame,
    geometry: pd.DataFrame,
    image_quality: pd.DataFrame,
    stability: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[E] Gộp bằng chứng và quyết định trạng thái...")
    base = dataset[["source", "file", "label", "handedness", "hard_invalid_feature"]]
    report = base.merge(
        geometry.drop(columns=["label", "handedness"]), on="file", validate="one_to_one"
    ).merge(
        image_quality.drop(columns=["label", "handedness"]),
        on="file", validate="one_to_one"
    ).merge(
        stability.drop(columns=["label", "handedness"]),
        on="file", how="left", validate="one_to_one"
    )
    for column in [
        "stability_tested", "bad_stability", "stability_candidate",
    ]:
        report[column] = report[column].fillna(False).astype(bool)
    for column in ["bad_geometry", "bad_image_quality"]:
        report[column] = report[column].fillna(False).astype(bool)
    report["quality_evidence_count"] = (
        report["bad_geometry"].astype(int)
        + report["bad_image_quality"].astype(int)
        + report["bad_stability"].astype(int)
    )
    report["reason"] = ""
    report["final_status"] = "KEEP"

    hard_invalid = report["hard_invalid_feature"] | report["image_read_error"].fillna(True)
    multi_confirmed = report["bad_stability"] & (
        report["bad_geometry"] | report["bad_image_quality"]
    )
    exclude = hard_invalid | multi_confirmed
    report.loc[hard_invalid, "reason"] = "INVALID_FEATURE_OR_IMAGE"
    report.loc[
        multi_confirmed & report["bad_geometry"] & report["bad_image_quality"],
        "reason",
    ] = "MULTIPLE_QUALITY_FAILURES"
    report.loc[
        multi_confirmed & report["bad_geometry"] & ~report["bad_image_quality"],
        "reason",
    ] = "BAD_LANDMARK_GEOMETRY_AND_STABILITY"
    report.loc[
        multi_confirmed & ~report["bad_geometry"] & report["bad_image_quality"],
        "reason",
    ] = "BAD_IMAGE_QUALITY_AND_STABILITY"
    report.loc[exclude, "final_status"] = "EXCLUDE_FROM_TRAIN"

    review = (~exclude) & (
        report["geometry_candidate"]
        | report["image_candidate"]
        | report["stability_candidate"]
        | report["bad_geometry"]
        | report["bad_image_quality"]
        | report["bad_stability"]
    )
    report.loc[review, "final_status"] = "REVIEW"
    report["evidence"] = report.apply(evidence_text, axis=1)

    extraction = rejected.copy()
    extraction["source"] = "internet"
    extraction["handedness"] = "Unknown"
    extraction["extraction_rejected"] = True
    extraction["final_status"] = "EXCLUDE_FROM_TRAIN"
    extraction["evidence"] = "extraction:" + extraction["reason"].astype(str)
    extraction["quality_evidence_count"] = 1
    report["extraction_rejected"] = False
    all_columns = list(dict.fromkeys(list(report.columns) + list(extraction.columns)))
    full_report = pd.concat(
        [report.reindex(columns=all_columns), extraction.reindex(columns=all_columns)],
        ignore_index=True,
    )
    manifest_order = {name: index for index, name in enumerate(manifest["file"])}
    full_report["_order"] = full_report["file"].map(manifest_order)
    full_report = full_report.sort_values("_order").drop(columns="_order")

    review_report = full_report[full_report["final_status"] == "REVIEW"].copy()
    excluded_report = full_report[
        full_report["final_status"] == "EXCLUDE_FROM_TRAIN"
    ].copy()
    keep_files = set(report.loc[report["final_status"] == "KEEP", "file"])
    clean = dataset[dataset["file"].isin(keep_files)].drop(
        columns=["hard_invalid_feature"]
    )
    atomic_csv(full_report, REPORT_FILE)
    atomic_csv(review_report, REVIEW_FILE)
    atomic_csv(excluded_report, EXCLUDED_FILE)
    atomic_csv(clean, CLEAN_FILE)
    return full_report, review_report, excluded_report, clean


def quantile_summary(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0}
    quantiles = values.quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
    return {
        "count": int(len(values)),
        **{f"q{int(q * 100):02d}": float(value) for q, value in quantiles.items()},
    }


def build_summary(
    full_report: pd.DataFrame,
    dataset: pd.DataFrame,
    rejected: pd.DataFrame,
) -> dict:
    statuses = full_report["final_status"].value_counts()
    status_by_label = pd.crosstab(full_report["label"], full_report["final_status"])
    for status in ("KEEP", "REVIEW", "EXCLUDE_FROM_TRAIN"):
        if status not in status_by_label:
            status_by_label[status] = 0
    status_by_label = status_by_label[
        ["KEEP", "REVIEW", "EXCLUDE_FROM_TRAIN"]
    ]
    status_by_label["TOTAL"] = status_by_label.sum(axis=1)
    status_by_label["KEEP_RATE"] = (
        status_by_label["KEEP"] / status_by_label["TOTAL"]
    )

    valid_report = full_report[~full_report["extraction_rejected"].fillna(False)].copy()
    valid_by_label = pd.crosstab(valid_report["label"], valid_report["final_status"])
    for status in ("KEEP", "REVIEW", "EXCLUDE_FROM_TRAIN"):
        if status not in valid_by_label:
            valid_by_label[status] = 0
    valid_by_label["additional_not_keep_rate"] = (
        valid_by_label.get("REVIEW", 0) + valid_by_label.get("EXCLUDE_FROM_TRAIN", 0)
    ) / valid_by_label.sum(axis=1)
    overall_additional = float(
        (valid_report["final_status"] != "KEEP").mean()
    )
    warnings: list[str] = []
    for label, row in valid_by_label.iterrows():
        rate = float(row["additional_not_keep_rate"])
        if rate > max(0.10, overall_additional * 2.0):
            warnings.append(
                f"{label}: tỷ lệ REVIEW+EXCLUDE mới {rate:.2%}, "
                f"cao hơn mức chung {overall_additional:.2%}"
            )
    for label in ("A", "B", "P", "W"):
        row = status_by_label.loc[label]
        warnings.append(
            f"Theo dõi {label}: KEEP={int(row['KEEP'])}, REVIEW={int(row['REVIEW'])}, "
            f"EXCLUDE={int(row['EXCLUDE_FROM_TRAIN'])}, keep_rate={row['KEEP_RATE']:.2%}"
        )

    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "total_original_images": int(len(full_report)),
        "extraction_rejected": int(len(rejected)),
        "extraction_reasons": {
            str(key): int(value)
            for key, value in rejected["reason"].value_counts().items()
        },
        "valid_landmarks": int(len(dataset)),
        "status_counts": {
            status: int(statuses.get(status, 0))
            for status in ("KEEP", "REVIEW", "EXCLUDE_FROM_TRAIN")
        },
        "status_rates": {
            status: float(statuses.get(status, 0) / len(full_report))
            for status in ("KEEP", "REVIEW", "EXCLUDE_FROM_TRAIN")
        },
        "by_label": {
            str(label): {
                "KEEP": int(row["KEEP"]),
                "REVIEW": int(row["REVIEW"]),
                "EXCLUDE_FROM_TRAIN": int(row["EXCLUDE_FROM_TRAIN"]),
                "TOTAL": int(row["TOTAL"]),
                "KEEP_RATE": float(row["KEEP_RATE"]),
            }
            for label, row in status_by_label.iterrows()
        },
        "top_exclude_reasons": {
            str(key): int(value)
            for key, value in full_report.loc[
                full_report["final_status"] == "EXCLUDE_FROM_TRAIN", "reason"
            ].value_counts().head(10).items()
        },
        "score_distributions": {
            "geometry_score": quantile_summary(valid_report["geometry_score"]),
            "stability_score": quantile_summary(valid_report["stability_score"]),
            "image_quality_score": quantile_summary(valid_report["image_quality_score"]),
        },
        "thresholds": {
            "reference": "median/MAD robust z within label+handedness; label fallback below 200 samples",
            "suspicious_z": SUSPICIOUS_Z,
            "bad_z": BAD_Z,
            "exclusion_rule": "hard error, or bad stability plus bad geometry/image quality",
        },
        "warnings": warnings,
    }
    atomic_json(summary, SUMMARY_FILE)
    return summary


def print_summary(summary: dict) -> None:
    print("\n================ KẾT QUẢ THỰC TẾ ================")
    print(f"Tổng ảnh gốc             : {summary['total_original_images']:,}")
    print(f"Rejected từ extraction   : {summary['extraction_rejected']:,}")
    for reason, count in summary["extraction_reasons"].items():
        print(f"  {reason:<26}: {count:,}")
    print(f"Landmark hợp lệ          : {summary['valid_landmarks']:,}")
    for status, count in summary["status_counts"].items():
        print(f"{status:<26}: {count:,} ({summary['status_rates'][status]:.2%})")
    print("\nTheo label:")
    label_table = pd.DataFrame(summary["by_label"]).T
    print(label_table.to_string())
    print("\n10 lý do exclude phổ biến:")
    print(pd.Series(summary["top_exclude_reasons"]).to_string())
    print("\nPhân bố score:")
    print(pd.DataFrame(summary["score_distributions"]).T.to_string())
    print("\nCảnh báo:")
    for warning in summary["warnings"]:
        print("-", warning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Bỏ qua checkpoint nhưng vẫn chỉ ghi file kết quả mới.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, raw, rejected, manifest = validate_sources()
    original_signature = source_signature()
    meta = load_meta()
    current = checkpoint_is_current(meta) and not args.rebuild
    if args.rebuild:
        for checkpoint in (GEOMETRY_CHECKPOINT, IMAGE_CHECKPOINT, STABILITY_CHECKPOINT):
            if checkpoint.exists():
                checkpoint.unlink()

    if current and GEOMETRY_CHECKPOINT.exists():
        print("[B] Dùng checkpoint geometry.")
        geometry = pd.read_csv(GEOMETRY_CHECKPOINT)
    else:
        geometry = build_geometry(dataset, raw)
        save_meta(meta)

    if current and IMAGE_CHECKPOINT.exists() and len(pd.read_csv(IMAGE_CHECKPOINT, usecols=["file"])) == len(dataset):
        print("[C] Dùng checkpoint image quality.")
        image_quality = pd.read_csv(IMAGE_CHECKPOINT)
    else:
        image_quality = build_image_quality(dataset, raw)
        save_meta(meta)

    if not current and STABILITY_CHECKPOINT.exists():
        STABILITY_CHECKPOINT.unlink()
    stability = build_stability(dataset, geometry, image_quality)
    save_meta(meta)
    full_report, _, _, _ = combine_reports(
        dataset, rejected, manifest, geometry, image_quality, stability
    )
    summary = build_summary(full_report, dataset, rejected)
    if source_signature() != original_signature:
        raise RuntimeError("Một file nguồn được bảo vệ đã thay đổi trong lúc chạy")
    print_summary(summary)
    print("\nOutputs:")
    for path in (REPORT_FILE, REVIEW_FILE, EXCLUDED_FILE, CLEAN_FILE, SUMMARY_FILE):
        print(path)


if __name__ == "__main__":
    main()
