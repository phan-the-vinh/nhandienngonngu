"""Extract ASL-HG digit 4/6/8 images into temporary digits_v2 landmarks.

Reuses the established digit MediaPipe/normalization extraction path. This
module never runs extraction on import; the user must execute it explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from extract_external_digits_v2 import (
    COLUMNS,
    FAILED_COLUMNS,
    FEATURE_COLUMNS,
    IMAGE_SUFFIXES,
    ROOT,
    TASK,
    duplicate_groups,
    extract_one,
    pilot_subset,
    sha256_file,
    temporary_path,
    validate_output,
    write_csv,
)


EXPERIMENT = ROOT / "data" / "experiments" / "digits_v2"
SOURCE = EXPERIMENT / "external" / "asl_hg"
PROCESSED = EXPERIMENT / "processed" / "asl_hg_4_6_8_landmarks_v1.csv"
FAILED = EXPERIMENT / "metadata" / "asl_hg_4_6_8_failed_samples_v1.csv"
REPORT = EXPERIMENT / "metadata" / "asl_hg_4_6_8_extraction_report_v1.json"
LABELS = ("4", "6", "8")
FILENAME_PATTERN = re.compile(r"^(P\d+)_([468])_(\d+)$")


def relative_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sample_id(path: Path, label: str) -> str:
    return f"asl_hg_{label}_{path.stem}"


def filename_group(path: Path, label: str) -> str:
    """Return the exact P-number token; do not assert it is a participant ID."""
    match = FILENAME_PATTERN.fullmatch(path.stem)
    if match is None:
        return ""
    if match.group(2) != label:
        raise ValueError(f"Filename label conflicts with folder label: {path}")
    return f"asl_hg_{match.group(1)}"


def list_source(expected_per_class: int) -> dict[str, list[Path]]:
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Missing ASL-HG source: {SOURCE}")
    unexpected = sorted(item.name for item in SOURCE.iterdir()
                        if item.is_dir() and item.name not in LABELS)
    if unexpected:
        raise ValueError(f"Unexpected label folders: {unexpected}")
    by_class = {}
    seen_ids = set()
    for label in LABELS:
        folder = SOURCE / label
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing label folder: {folder}")
        images = sorted(
            (path for path in folder.iterdir()
             if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda path: path.name,
        )
        if len(images) != expected_per_class:
            raise ValueError(
                f"Class {label}: found {len(images)} images, expected {expected_per_class}"
            )
        nested = [path for path in folder.rglob("*")
                  if path.is_file() and path.parent != folder
                  and path.suffix.lower() in IMAGE_SUFFIXES]
        if nested:
            raise ValueError(f"Unexpected nested images in {folder}")
        for path in images:
            identifier = sample_id(path, label)
            if identifier in seen_ids:
                raise ValueError(f"Duplicate sample ID: {identifier}")
            seen_ids.add(identifier)
            filename_group(path, label)  # Check embedded label if the pattern is known.
        by_class[label] = images
    return by_class


def source_state(by_class: dict[str, list[Path]]) -> dict[str, tuple[int, int, str]]:
    return {
        relative_path(path): (path.stat().st_size, path.stat().st_mtime_ns, sha256_file(path))
        for label in LABELS for path in by_class[label]
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-per-class", type=int, metavar="N",
                        help="Inspect at most N images per class; write no output")
    parser.add_argument("--expected-per-class", type=int, default=800,
                        help="Stop if a class does not contain this many images (default: 800)")
    args = parser.parse_args()
    if args.expected_per_class <= 0:
        parser.error("--expected-per-class must be positive")
    if args.pilot_per_class is not None and not 1 <= args.pilot_per_class <= 20:
        parser.error("--pilot-per-class must be between 1 and 20")
    return args


def run(args: argparse.Namespace) -> None:
    by_class = list_source(args.expected_per_class)
    if not TASK.is_file():
        raise FileNotFoundError(f"Missing canonical MediaPipe task: {TASK}")
    pilot = args.pilot_per_class is not None
    if not pilot:
        existing = [str(path) for path in (PROCESSED, FAILED, REPORT)
                    if path.exists() or path.is_symlink()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing output: {existing}")
        before = source_state(by_class)

    # Importing the ArdaMavi module reuses its exact extraction function,
    # including no pre-MediaPipe flip and canonical normalize_landmarks().
    from extract_external_digits_v2 import mp_python, vision

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(TASK)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    good = []
    failed = []
    per_class = {}
    resolutions = Counter()
    group_tokens = Counter()
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for label in LABELS:
            images = pilot_subset(by_class[label], args.pilot_per_class) if pilot else by_class[label]
            success = 0
            for path in images:
                row, reason, resolution = extract_one(path, label, landmarker)
                resolutions[resolution or "UNREADABLE"] += 1
                if row is None:
                    failed.append({
                        "sample_id": sample_id(path, label),
                        "label": label,
                        "source_path": relative_path(path),
                        "reason": reason,
                    })
                    continue
                token = filename_group(path, label)
                row.update({
                    "sample_id": sample_id(path, label),
                    "source": "asl_hg",
                    "source_group": token,
                })
                good.append(row)
                group_tokens[token or "UNKNOWN"] += 1
                success += 1
            per_class[label] = {
                "source_images": len(by_class[label]),
                "attempted": len(images),
                "success": success,
                "failed": len(images) - success,
            }
            print(f"class={label} attempted={len(images)} success={success} failed={len(images) - success}",
                  flush=True)

    if pilot:
        if any(item["success"] == 0 for item in per_class.values()):
            raise RuntimeError("PILOT_FAILED: a class has no usable 63-feature sample")
        print(json.dumps({"status": "PILOT_PASS", "success": len(good), "failed": len(failed)}))
        return

    after = source_state(by_class)
    if after != before:
        raise RuntimeError("Source images changed during extraction; no output published")
    if len(good) + len(failed) != sum(map(len, by_class.values())):
        raise AssertionError("Incomplete source accounting")

    by_hash = defaultdict(list)
    for label in LABELS:
        for path in by_class[label]:
            by_hash[before[relative_path(path)][2]].append(sample_id(path, label))
    by_features = defaultdict(list)
    for row in good:
        by_features[tuple(row[column] for column in FEATURE_COLUMNS)].append(row["sample_id"])
    report = {
        "status": "ASL_HG_4_6_8_EXTRACTION_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": relative_path(SOURCE),
        "source": "asl_hg",
        "source_images": sum(map(len, by_class.values())),
        "success": len(good),
        "failed": len(failed),
        "per_class": per_class,
        "resolution_distribution": dict(sorted(resolutions.items())),
        "handedness_distribution": dict(Counter(row["handedness"] or "UNKNOWN" for row in good)),
        "source_group_distribution": dict(sorted(group_tokens.items())),
        "source_group_rule": "Filename P-number token only, shared across labels; participant identity unverified",
        "participant_id_status": "UNKNOWN: no source documentation confirms P-number semantics",
        "exact_duplicate_source_files": duplicate_groups(by_hash),
        "exact_duplicate_feature_rows": duplicate_groups(by_features),
        "mediapipe_task": relative_path(TASK),
        "mediapipe_task_sha256": sha256_file(TASK),
        "normalization": "ai/preprocessing/landmark_utils.py::normalize_landmarks",
        "normalization_sha256": sha256_file(ROOT / "ai/preprocessing/landmark_utils.py"),
        "frame_flipped_before_mediapipe": False,
        "scaler_fit": False,
        "production_classifier_used": False,
        "feature_columns": list(FEATURE_COLUMNS),
        "processed_csv": relative_path(PROCESSED),
        "failed_csv": relative_path(FAILED),
        "source_files_unchanged_by_sha256_size_mtime": True,
    }

    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    FAILED.parent.mkdir(parents=True, exist_ok=True)
    tmp_processed = temporary_path(PROCESSED)
    tmp_failed = temporary_path(FAILED)
    tmp_report = temporary_path(REPORT)
    write_csv(tmp_processed, COLUMNS, good)
    validate_output(tmp_processed, len(good))
    with tmp_failed.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILED_COLUMNS)
        writer.writeheader()
        writer.writerows(failed)
    report["processed_csv_sha256"] = sha256_file(tmp_processed)
    report["failed_csv_sha256"] = sha256_file(tmp_failed)
    with tmp_report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp_processed.rename(PROCESSED)
    tmp_failed.rename(FAILED)
    tmp_report.rename(REPORT)
    print(json.dumps({"status": report["status"], "success": len(good), "failed": len(failed)}))


if __name__ == "__main__":
    run(parse_args())
