"""User-run, frame-level landmark coverage audit for the 712 SignNTeam J/Z AVIs.

This tool does not extract features, segment motion, or create training data.
It writes only a per-video CSV and summary JSON under data/metadata/dynamic/.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.extract_dynamic import discover_videos  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "dynamic.json"
METADATA_DIR = ROOT / "data" / "metadata" / "dynamic"
CSV_PATH = METADATA_DIR / "landmark_coverage_audit_v1.csv"
SUMMARY_PATH = METADATA_DIR / "landmark_coverage_summary_v1.json"
EXPECTED_COUNTS = {"J": 316, "Z": 396}
FIELDS = (
    "label", "source_path", "decoded_frame_count", "valid_landmark_frames",
    "invalid_landmark_frames", "landmark_coverage_ratio", "has_16_valid_frames",
    "audit_status", "failure_reason", "invalid_reason_counts",
)


def preflight() -> tuple[list[tuple[Path, str]], Path]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("task") != "dynamic_asl" or config.get("dynamic_labels") != ["J", "Z", "NONE"]:
        raise ValueError("Unexpected dynamic task or label schema")
    source_root = (ROOT / config["raw_external_root"]).resolve()
    task_path = (ROOT / config["sequence_extraction"]["mediapipe_model"]).resolve()
    if not source_root.is_relative_to(ROOT) or not source_root.is_dir():
        raise ValueError(f"Invalid source root: {source_root}")
    if not task_path.is_relative_to(ROOT) or not task_path.is_file():
        raise ValueError(f"Invalid MediaPipe task: {task_path}")
    videos = discover_videos(source_root, ["J", "Z"])
    counts = Counter(label for _, label in videos)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"Expected J=316, Z=396; found {dict(counts)}. No audit started")
    return videos, task_path


def audit_video(path: Path, label: str, landmarker) -> dict:
    """Decode every frame; a valid frame has one detected hand with 21 finite XYZ points."""
    import cv2
    import mediapipe as mp
    from ai.feature_extraction.hand_landmarks import landmarks_to_array

    decoded = valid = 0
    invalid_reasons: Counter[str] = Counter()
    status = "COMPLETE"
    failure_reason = ""
    capture = cv2.VideoCapture(str(path))  # File path only; never a webcam index.
    try:
        if not capture.isOpened():
            status, failure_reason = "ERROR", "VIDEO_OPEN_FAIL"
        else:
            while True:
                try:
                    ok, frame = capture.read()
                except Exception as exc:
                    status, failure_reason = "ERROR", f"VIDEO_READ_ERROR:{type(exc).__name__}"
                    break
                if not ok:
                    break
                if frame is None or frame.size == 0 or len(frame.shape) != 3 or frame.shape[2] != 3:
                    status, failure_reason = "ERROR", f"INVALID_DECODED_FRAME_{decoded}"
                    break
                decoded += 1
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # No flip or mirror.
                    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                except Exception as exc:
                    invalid_reasons[f"DETECTION_ERROR:{type(exc).__name__}"] += 1
                    continue
                if not result.hand_landmarks:
                    invalid_reasons["NO_HAND"] += 1
                    continue
                try:
                    landmarks_to_array(result.hand_landmarks[0])
                except (AttributeError, TypeError, ValueError):
                    invalid_reasons["INVALID_LANDMARKS"] += 1
                    continue
                valid += 1
            if decoded == 0 and status == "COMPLETE":
                status, failure_reason = "ERROR", "NO_DECODED_FRAMES"
            if status == "COMPLETE" and any(
                reason.startswith("DETECTION_ERROR:") for reason in invalid_reasons
            ):
                status, failure_reason = "REVIEW_REQUIRED", "MEDIAPIPE_DETECTION_ERRORS"
    finally:
        capture.release()
    return {
        "label": label,
        "source_path": path.relative_to(ROOT).as_posix(),
        "decoded_frame_count": decoded,
        "valid_landmark_frames": valid,
        "invalid_landmark_frames": decoded - valid,
        "landmark_coverage_ratio": valid / decoded if decoded else "",
        "has_16_valid_frames": valid >= 16,
        "audit_status": status,
        "failure_reason": failure_reason,
        "invalid_reason_counts": json.dumps(dict(invalid_reasons), sort_keys=True),
    }


def summarize(rows: list[dict]) -> dict:
    per_label = {}
    for label in ("J", "Z"):
        subset = [row for row in rows if row["label"] == label]
        ratios = [float(row["landmark_coverage_ratio"]) for row in subset
                  if row["landmark_coverage_ratio"] != ""]
        per_label[label] = {
            "videos": len(subset),
            "videos_at_least_16_valid_frames": sum(row["has_16_valid_frames"] for row in subset),
            "videos_fewer_than_16_valid_frames": sum(not row["has_16_valid_frames"] for row in subset),
            "video_audit_errors": sum(row["audit_status"] != "COMPLETE" for row in subset),
            "coverage_ratio": {
                "min": min(ratios) if ratios else None,
                "median": statistics.median(ratios) if ratios else None,
                "mean": statistics.mean(ratios) if ratios else None,
                "max": max(ratios) if ratios else None,
            },
            "coverage_ratio_samples": len(ratios),
        }
    return {
        "status": "COMPLETE" if len(rows) == 712 and all(
            row["audit_status"] == "COMPLETE" for row in rows) else "REVIEW_REQUIRED",
        "purpose": "LANDMARK_COVERAGE_ONLY",
        "official_dataset_created": False,
        "motion_segmentation_used": False,
        "feature_extraction_used": False,
        "minimum_valid_frames_for_proposed_baseline": 16,
        "video_count": len(rows),
        "per_label": per_label,
        "report_csv": CSV_PATH.relative_to(ROOT).as_posix(),
    }


def run_audit(videos: list[tuple[Path, str]], task_path: Path) -> dict:
    from ai.feature_extraction.hand_landmarks import create_hand_landmarker

    csv_partial = CSV_PATH.with_suffix(".csv.partial")
    summary_partial = SUMMARY_PATH.with_suffix(".json.partial")
    for target in (CSV_PATH, SUMMARY_PATH, csv_partial, summary_partial):
        if target.exists():
            raise FileExistsError(f"Existing report/checkpoint; refusing overwrite: {target}")
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    # A partial CSV remains available for inspection if the user interrupts the run.
    with create_hand_landmarker(task_path, running_mode="IMAGE", num_hands=1) as landmarker, \
            csv_partial.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, (path, label) in enumerate(videos, 1):
            row = audit_video(path, label, landmarker)
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            if index % 25 == 0 or index == len(videos):
                print(f"Coverage audit: {index}/{len(videos)} videos", flush=True)
    summary = summarize(rows)
    with summary_partial.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    csv_partial.rename(CSV_PATH)
    summary_partial.rename(SUMMARY_PATH)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Check paths and inventory; decode no video")
    mode.add_argument("--audit", action="store_true", help="Audit all 712 videos after confirmation")
    args = parser.parse_args(argv)
    try:
        videos, task_path = preflight()
        print(json.dumps({"status": "PREFLIGHT_PASS", "videos": len(videos),
                          "counts": EXPECTED_COUNTS, "task_path": task_path.relative_to(ROOT).as_posix(),
                          "output_csv": CSV_PATH.relative_to(ROOT).as_posix(),
                          "output_summary": SUMMARY_PATH.relative_to(ROOT).as_posix()}))
        if args.check_only:
            return 0
        if input("Run full 712-video landmark coverage audit? [y/N] ").strip().lower() != "y":
            print("Audit cancelled; no reports written")
            return 0
        summary = run_audit(videos, task_path)
        print(json.dumps({"status": summary["status"], "per_label": summary["per_label"]}))
        return 0 if summary["status"] == "COMPLETE" else 1
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
