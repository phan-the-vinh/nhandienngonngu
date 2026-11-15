"""Extract the ArdaMavi digit images into TEMPORARY digits_v2 features.

This tool does not train, fit a scaler, load a classifier, flip images, or
modify source images. Run --pilot-per-class 10 before the full extraction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "data" / "experiments" / "digits_v2"
SOURCE = EXPERIMENT / "external" / "ardamavi_digits" / "Dataset"
TASK = ROOT / "models" / "mediapipe" / "hand_landmarker.task"
PROCESSED = EXPERIMENT / "processed" / "ardamavi_landmarks_v1.csv"
FAILED = EXPERIMENT / "metadata" / "ardamavi_failed_samples_v1.csv"
REPORT = EXPERIMENT / "metadata" / "ardamavi_extraction_report_v1.json"
LABELS = tuple(str(index) for index in range(10))
FEATURE_COLUMNS = tuple(f"f{index}" for index in range(1, 64))
COLUMNS = (
    "sample_id", "label", "source", "source_path", "filename",
    "width", "height", "handedness", "handedness_score", "source_group",
) + FEATURE_COLUMNS
FAILED_COLUMNS = ("sample_id", "label", "source_path", "reason")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Import the exact function used by the production digit camera and extractor.
sys.path.insert(0, str(ROOT / "ai" / "preprocessing"))
from landmark_utils import normalize_landmarks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-per-class", type=int, metavar="N",
        help="Read at most N evenly spaced images per class; write no output",
    )
    parser.add_argument(
        "--expected-count", type=int, default=2062,
        help="Refuse full extraction if the source image count differs (default: 2062)",
    )
    args = parser.parse_args()
    if args.pilot_per_class is not None and not 1 <= args.pilot_per_class <= 20:
        parser.error("--pilot-per-class must be between 1 and 20")
    if args.expected_count <= 0:
        parser.error("--expected-count must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def list_source() -> dict[str, list[Path]]:
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Missing source dataset: {SOURCE}")
    other_dirs = sorted(item.name for item in SOURCE.iterdir()
                        if item.is_dir() and item.name not in LABELS)
    if other_dirs:
        raise ValueError(f"Unexpected class folders: {other_dirs}")
    result: dict[str, list[Path]] = {}
    sample_ids: set[str] = set()
    for label in LABELS:
        folder = SOURCE / label
        if not folder.is_dir():
            raise ValueError(f"Missing digit class: {label}")
        paths = sorted((p for p in folder.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                       key=lambda p: p.name)
        if not paths:
            raise ValueError(f"No images in digit class: {label}")
        nested_images = [p for p in folder.rglob("*")
                         if p.is_file() and p.parent != folder
                         and p.suffix.lower() in IMAGE_SUFFIXES]
        if nested_images:
            raise ValueError(f"Unexpected nested images in class {label}")
        for path in paths:
            sample_id = f"ardamavi_{label}_{path.stem}"
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate sample ID: {sample_id}")
            sample_ids.add(sample_id)
        result[label] = paths
    return result


def pilot_subset(paths: list[Path], count: int) -> list[Path]:
    if count >= len(paths):
        return paths
    if count == 1:
        return [paths[len(paths) // 2]]
    return [paths[index * (len(paths) - 1) // (count - 1)]
            for index in range(count)]


def file_state(paths: dict[str, list[Path]]) -> dict[str, tuple[int, int, str]]:
    """Capture byte hashes as well as size/mtime for duplicate and source checks."""
    state = {}
    for label in LABELS:
        for path in paths[label]:
            info = path.stat()
            state[source_path(path)] = (info.st_size, info.st_mtime_ns,
                                        sha256_file(path))
    return state


def extract_one(path: Path, label: str, landmarker):
    sample_id = f"ardamavi_{label}_{path.stem}"
    relative = source_path(path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None, "IMAGE_DECODE_FAILED", None
    height, width = bgr.shape[:2]
    resolution = f"{width}x{height}"
    try:
        # Production digit input is NOT flipped. Do not change this order.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detected = landmarker.detect(mp_image)
    except Exception as exc:
        return None, f"MEDIAPIPE_ERROR:{type(exc).__name__}", resolution
    if not detected.hand_landmarks:
        return None, "NO_HAND_DETECTED", resolution
    hand = detected.hand_landmarks[0]
    if len(hand) != 21:
        return None, f"LANDMARK_COUNT_{len(hand)}", resolution
    if not all(math.isfinite(value)
               for point in hand for value in (point.x, point.y, point.z)):
        return None, "NONFINITE_LANDMARK", resolution
    features = normalize_landmarks(hand)
    if len(features) != 63 or not all(math.isfinite(value) for value in features):
        return None, "INVALID_63_FEATURES", resolution
    handedness = ""
    handedness_score = ""
    if detected.handedness and detected.handedness[0]:
        handedness = detected.handedness[0][0].category_name
        handedness_score = float(detected.handedness[0][0].score)
    row = {
        "sample_id": sample_id,
        "label": label,
        "source": "ardamavi_digits",
        "source_path": relative,
        "filename": path.name,
        "width": width,
        "height": height,
        "handedness": handedness,
        "handedness_score": handedness_score,
        # IMG_* does not establish signer/session/independent gesture group.
        "source_group": "",
    }
    row.update(zip(FEATURE_COLUMNS, features))
    return row, None, resolution


def duplicate_groups(values: dict[object, list[str]]) -> dict:
    groups = [sorted(ids) for ids in values.values() if len(ids) > 1]
    groups.sort(key=lambda ids: (-len(ids), ids[0]))
    return {
        "groups": len(groups),
        "extra_rows": sum(len(ids) - 1 for ids in groups),
        "affected_rows": sum(map(len, groups)),
        "members": groups,
    }


def temporary_path(final: Path) -> Path:
    return final.with_name(f".{final.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def validate_output(path: Path, expected_rows: int) -> None:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError("Processed CSV feature/order schema mismatch")
        count = 0
        seen = set()
        for row in reader:
            if row["label"] not in LABELS:
                raise ValueError("Invalid processed label")
            if row["sample_id"] in seen:
                raise ValueError("Duplicate processed sample_id")
            seen.add(row["sample_id"])
            if not all(math.isfinite(float(row[name])) for name in FEATURE_COLUMNS):
                raise ValueError("Nonfinite processed feature")
            count += 1
    if count != expected_rows:
        raise ValueError(f"Processed row count {count} != {expected_rows}")


def run(args: argparse.Namespace) -> None:
    by_class = list_source()
    total_images = sum(map(len, by_class.values()))
    if total_images != args.expected_count:
        raise ValueError(f"Source count changed: {total_images} != {args.expected_count}")
    if not TASK.is_file():
        raise FileNotFoundError(f"Missing production MediaPipe task: {TASK}")
    pilot = args.pilot_per_class is not None
    if not pilot:
        existing = [str(path) for path in (PROCESSED, FAILED, REPORT)
                    if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing output: {existing}")
        before = file_state(by_class)
    else:
        before = None

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(TASK)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    good: list[dict] = []
    failed: list[dict] = []
    class_stats = {}
    resolutions = Counter()
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for label in LABELS:
            images = (pilot_subset(by_class[label], args.pilot_per_class)
                      if pilot else by_class[label])
            success = 0
            for path in images:
                row, reason, resolution = extract_one(path, label, landmarker)
                resolutions[resolution or "UNREADABLE"] += 1
                if row is None:
                    failed.append({
                        "sample_id": f"ardamavi_{label}_{path.stem}",
                        "label": label,
                        "source_path": source_path(path),
                        "reason": reason,
                    })
                else:
                    good.append(row)
                    success += 1
            class_stats[label] = {
                "source_images": len(by_class[label]),
                "attempted": len(images),
                "success": success,
                "failed": len(images) - success,
                "success_rate": success / len(images),
            }
            print(f"class={label} attempted={len(images)} success={success} ",
                  f"failed={len(images) - success}", flush=True)

    if pilot:
        if any(item["success"] == 0 for item in class_stats.values()):
            raise RuntimeError("PILOT_FAILED: at least one class has no valid 63-feature sample")
        print(json.dumps({"status": "PILOT_PASS", "attempted": len(good) + len(failed),
                          "success": len(good), "failed": len(failed)}, sort_keys=True))
        return

    after = file_state(by_class)
    if after != before:
        raise RuntimeError("Source images changed during extraction; no output published")
    if len(good) + len(failed) != total_images:
        raise AssertionError("Success/failure accounting is incomplete")

    by_hash = defaultdict(list)
    for label in LABELS:
        for path in by_class[label]:
            by_hash[before[source_path(path)][2]].append(f"ardamavi_{label}_{path.stem}")
    by_features = defaultdict(list)
    for row in good:
        by_features[tuple(row[name] for name in FEATURE_COLUMNS)].append(row["sample_id"])
    file_duplicates = duplicate_groups(by_hash)
    feature_duplicates = duplicate_groups(by_features)
    handedness_counts = Counter(row["handedness"] or "UNKNOWN" for row in good)
    nonfinite_count = sum(not math.isfinite(row[name]) for row in good
                          for name in FEATURE_COLUMNS)

    report = {
        "schema_version": 1,
        "status": "ARDAMAVI_FULL_EXTRACTION_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": source_path(SOURCE),
        "source": "ardamavi_digits",
        "source_image_count": total_images,
        "success": len(good),
        "failed": len(failed),
        "success_rate": len(good) / total_images,
        "per_class": class_stats,
        "handedness_distribution": dict(sorted(handedness_counts.items())),
        "resolution_distribution": dict(sorted(resolutions.items())),
        "nonfinite_feature_values": nonfinite_count,
        "exact_duplicate_source_files": file_duplicates,
        "exact_duplicate_feature_rows": feature_duplicates,
        "source_files_unchanged_by_sha256_size_mtime": True,
        "source_group_status": "UNKNOWN: filename/folder do not establish signer or gesture group",
        "source_group_column": "blank",
        "sample_id_scheme": "ardamavi_<label>_<filename stem>",
        "mediapipe_task": source_path(TASK),
        "mediapipe_task_sha256": sha256_file(TASK),
        "mediapipe_running_mode": "IMAGE",
        "mediapipe_num_hands": 1,
        "mediapipe_detection_confidence": 0.5,
        "mediapipe_presence_confidence": 0.5,
        "normalization": "ai/preprocessing/landmark_utils.py::normalize_landmarks",
        "normalization_sha256": sha256_file(ROOT / "ai/preprocessing/landmark_utils.py"),
        "frame_flipped_before_mediapipe": False,
        "mirror_augmentation": False,
        "scaler_fit": False,
        "production_classifier_used": False,
        "feature_columns": list(FEATURE_COLUMNS),
        "processed_csv": source_path(PROCESSED),
        "failed_csv": source_path(FAILED),
    }

    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    FAILED.parent.mkdir(parents=True, exist_ok=True)
    tmp_processed = temporary_path(PROCESSED)
    tmp_failed = temporary_path(FAILED)
    tmp_report = temporary_path(REPORT)
    write_csv(tmp_processed, COLUMNS, good)
    validate_output(tmp_processed, len(good))
    write_csv(tmp_failed, FAILED_COLUMNS, failed)
    with tmp_failed.open("r", newline="", encoding="utf-8") as handle:
        if sum(1 for _ in csv.DictReader(handle)) != len(failed):
            raise ValueError("Failed-sample CSV row count mismatch")
    report["processed_csv_sha256"] = sha256_file(tmp_processed)
    report["failed_csv_sha256"] = sha256_file(tmp_failed)
    with tmp_report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    # Windows rename refuses existing destinations; never replace prior results.
    tmp_processed.rename(PROCESSED)
    tmp_failed.rename(FAILED)
    tmp_report.rename(REPORT)
    print(json.dumps({"status": report["status"], "source": total_images,
                      "success": len(good), "failed": len(failed)}, sort_keys=True))


if __name__ == "__main__":
    run(parse_args())
