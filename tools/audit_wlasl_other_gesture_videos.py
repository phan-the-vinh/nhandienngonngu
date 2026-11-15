"""Read-only technical decode audit of downloaded WLASL OTHER_GESTURE videos.

Only creates temporary CSV/JSON reports. Never invokes MediaPipe, extracts
landmarks, changes candidate decisions, or modifies source videos.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "experiments" / "dynamic_none_v1"
META = EXP / "metadata"
RAW = EXP / "raw" / "wlasl_other_gesture"
CANDIDATES = META / "wlasl_other_gesture_candidates_v1.csv"
AUDIT_V1 = META / "wlasl_other_gesture_download_audit_v1.csv"
AUDIT_V2 = META / "wlasl_other_gesture_download_audit_v2.csv"
OUTPUT_CSV = META / "wlasl_other_gesture_technical_audit_v1.csv"
OUTPUT_JSON = META / "wlasl_other_gesture_technical_summary_v1.json"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
FILENAME_RE = re.compile(r"^WLASL_([A-Za-z0-9_-]+)\.(mp4|mov|webm|mkv|avi)$", re.IGNORECASE)
SUCCESS = {"DOWNLOADED", "ALREADY_PRESENT_VERIFIED"}
CSV_FIELDS = (
    "video_id", "gloss", "signer_id", "source", "url", "split",
    "frame_start", "frame_end", "review_status", "local_path", "extension",
    "file_size_bytes", "recorded_sha256", "actual_sha256", "sha256_match",
    "provenance_match", "open_ok", "decode_ok", "decoded_frame_count",
    "container_frame_count", "fps", "duration_seconds", "width", "height",
    "short_under_16", "abnormal_reasons",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def latest_events(events: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for event in events:
        latest[event["video_id"]] = event
    return latest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values), "min": min(values),
        "median": statistics.median(values), "mean": statistics.fmean(values),
        "max": max(values),
    }


def decode_video(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "open_ok": False, "decode_ok": False, "decoded_frame_count": 0,
        "container_frame_count": None, "fps": None, "duration_seconds": None,
        "width": None, "height": None, "short_under_16": True,
        "abnormal_reasons": [],
    }
    issues: list[str] = result["abnormal_reasons"]  # type: ignore[assignment]
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            issues.append("VIDEO_OPEN_FAILED")
            return result
        result["open_ok"] = True
        reported_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if math.isfinite(reported_count) and reported_count >= 0:
            result["container_frame_count"] = int(round(reported_count))
        if math.isfinite(fps) and fps > 0:
            result["fps"] = float(fps)
        else:
            issues.append("INVALID_FPS")
        if math.isfinite(width) and width > 0:
            result["width"] = int(round(width))
        else:
            issues.append("INVALID_WIDTH")
        if math.isfinite(height) and height > 0:
            result["height"] = int(round(height))
        else:
            issues.append("INVALID_HEIGHT")

        decoded = 0
        first_shape = None
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            decoded += 1
            if first_shape is None:
                first_shape = frame.shape
        result["decoded_frame_count"] = decoded
        result["decode_ok"] = decoded > 0
        result["short_under_16"] = decoded < 16
        if decoded == 0:
            issues.append("NO_FRAMES_DECODED")
        if decoded < 16:
            issues.append("FEWER_THAN_16_DECODED_FRAMES")
        if result["fps"] is not None:
            result["duration_seconds"] = decoded / result["fps"]  # type: ignore[operator]
        if first_shape is not None and result["width"] is not None and result["height"] is not None:
            if first_shape[1] != result["width"] or first_shape[0] != result["height"]:
                issues.append("DECODED_DIMENSIONS_DIFFER_FROM_CONTAINER")
        # Container frame counts may be inaccurate: record discrepancy as a
        # warning, never substitute it for the actual decoded count.
        if result["container_frame_count"] is not None:
            if abs(decoded - result["container_frame_count"]) > max(2, int(0.05 * result["container_frame_count"])):
                issues.append("CONTAINER_DECODED_FRAME_COUNT_MISMATCH")
        return result
    finally:
        capture.release()


def main() -> None:
    for required in (CANDIDATES, AUDIT_V1, AUDIT_V2):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not RAW.is_dir():
        raise FileNotFoundError(RAW)
    if OUTPUT_CSV.exists() or OUTPUT_JSON.exists():
        raise FileExistsError("Technical audit output already exists; refusing to overwrite")

    candidate_rows = read_csv(CANDIDATES)
    if len(candidate_rows) != 220 or len({row["video_id"] for row in candidate_rows}) != 220:
        raise ValueError("Candidate manifest must have 220 unique video IDs")
    candidates = {row["video_id"]: row for row in candidate_rows}
    first = latest_events(read_csv(AUDIT_V1))
    second = latest_events(read_csv(AUDIT_V2))
    if set(first) != set(candidates) or set(second) != set(candidates):
        raise ValueError("Download audit coverage differs from candidate manifest")

    files = sorted(path for path in RAW.iterdir() if path.is_file())
    video_files = [path for path in files if path.suffix.lower() in VIDEO_EXTENSIONS]
    other_root_files = [path.name for path in files if path.suffix.lower() not in VIDEO_EXTENSIONS]
    staging = RAW / "_staging"
    staging_count = sum(1 for path in staging.iterdir() if path.is_file()) if staging.is_dir() else 0
    by_id: dict[str, list[Path]] = {}
    bad_filenames: list[str] = []
    for path in video_files:
        match = FILENAME_RE.fullmatch(path.name)
        if not match:
            bad_filenames.append(path.name)
            continue
        by_id.setdefault(match.group(1), []).append(path)
    duplicate_video_ids = sorted(video_id for video_id, paths in by_id.items() if len(paths) > 1)
    successes = {video_id: row for video_id, row in second.items() if row["download_status"] in SUCCESS}
    original_successes = {video_id for video_id, row in first.items() if row["download_status"] in SUCCESS}
    retry_successes = {video_id for video_id, row in second.items() if row.get("retry_status") == "RETRY_SUCCESS"}
    if original_successes & retry_successes:
        raise ValueError("Original successes overlap retry successes")
    audit_paths = [row["local_path"].casefold() for row in successes.values()]
    duplicate_local_paths = sorted(path for path, count in Counter(audit_paths).items() if count > 1)
    unexpected_file_ids = sorted(set(by_id) - set(successes))
    missing_file_ids = sorted(set(successes) - set(by_id))

    records: list[dict[str, object]] = []
    for path in video_files:
        match = FILENAME_RE.fullmatch(path.name)
        video_id = match.group(1) if match else ""
        candidate = candidates.get(video_id)
        event = successes.get(video_id)
        reasons: list[str] = []
        size = path.stat().st_size
        actual_hash = sha256(path)
        if size == 0:
            reasons.append("ZERO_BYTE_FILE")
        elif size < 1024:
            reasons.append("FILE_UNDER_1KB")
        if not match:
            reasons.append("INVALID_FILENAME")
        if candidate is None:
            reasons.append("NOT_IN_CANDIDATE_MANIFEST")
        if event is None:
            reasons.append("NO_SUCCESSFUL_DOWNLOAD_AUDIT")
        expected_path = path.relative_to(ROOT).as_posix()
        provenance = bool(candidate and event)
        if provenance:
            provenance = (
                event["local_path"] == expected_path
                and all(event[field] == candidate[field] for field in
                        ("gloss", "signer_id", "source", "url", "frame_start", "frame_end", "split"))
                and event["review_status"] == candidate["review_status"] == "REVIEW_REQUIRED"
            )
        if not provenance:
            reasons.append("PROVENANCE_MISMATCH")
        hash_match = bool(event and event["sha256"] == actual_hash)
        if not hash_match:
            reasons.append("DOWNLOAD_HASH_MISMATCH")
        if video_id in duplicate_video_ids:
            reasons.append("DUPLICATE_VIDEO_ID")
        decoded = decode_video(path)
        reasons.extend(decoded.pop("abnormal_reasons"))  # type: ignore[arg-type]
        record = {
            "video_id": video_id,
            "gloss": candidate["gloss"] if candidate else "",
            "signer_id": candidate["signer_id"] if candidate else "",
            "source": candidate["source"] if candidate else "",
            "url": candidate["url"] if candidate else "",
            "split": candidate["split"] if candidate else "",
            "frame_start": candidate["frame_start"] if candidate else "",
            "frame_end": candidate["frame_end"] if candidate else "",
            "review_status": candidate["review_status"] if candidate else "",
            "local_path": expected_path,
            "extension": path.suffix.lower(),
            "file_size_bytes": size,
            "recorded_sha256": event["sha256"] if event else "",
            "actual_sha256": actual_hash,
            "sha256_match": hash_match,
            "provenance_match": provenance,
            **decoded,
            "abnormal_reasons": ";".join(dict.fromkeys(reasons)),
        }
        records.append(record)
        if len(records) % 20 == 0 or len(records) == len(video_files):
            print(f"Audited {len(records)}/{len(video_files)} videos")

    frames = [int(row["decoded_frame_count"]) for row in records if row["decode_ok"]]
    durations = [float(row["duration_seconds"]) for row in records if row["duration_seconds"] is not None]
    fps_values = [float(row["fps"]) for row in records if row["fps"] is not None]
    long_full_video_ids = sorted(
        str(row["video_id"]) for row in records
        if row["duration_seconds"] is not None and float(row["duration_seconds"]) > 60
    )
    issue_counts = Counter(reason for row in records for reason in str(row["abnormal_reasons"]).split(";") if reason)
    summary = {
        "status": "WLASL_OTHER_GESTURE_TECHNICAL_AUDIT_COMPLETE",
        "source_video_count": len(video_files),
        "candidate_count": len(candidates),
        "initial_download_success": len(original_successes),
        "retry_success": len(retry_successes),
        "latest_download_success": len(successes),
        "latest_download_failed": sum(row["download_status"] == "DOWNLOAD_FAILED" for row in second.values()),
        "open_success": sum(bool(row["open_ok"]) for row in records),
        "open_failed": sum(not row["open_ok"] for row in records),
        "decode_success": sum(bool(row["decode_ok"]) for row in records),
        "decode_failed": sum(not row["decode_ok"] for row in records),
        "frame_statistics": summarize(frames),
        "duration_seconds_statistics": summarize(durations),
        "fps_statistics": summarize(fps_values),
        "full_video_over_60_seconds_count": len(long_full_video_ids),
        "full_video_over_60_seconds_ids": long_full_video_ids,
        "annotated_window_under_16_frames_count": sum(
            int(row["frame_end"]) >= 0
            and int(row["frame_end"]) - int(row["frame_start"]) + 1 < 16
            for row in records
        ),
        "annotated_window_unknown_end_count": sum(str(row["frame_end"]) == "-1" for row in records),
        "short_under_16_count": sum(bool(row["short_under_16"]) for row in records),
        "zero_byte_count": sum(row["file_size_bytes"] == 0 for row in records),
        "duplicate_video_ids": duplicate_video_ids,
        "duplicate_local_paths": duplicate_local_paths,
        "unexpected_file_ids": unexpected_file_ids,
        "missing_successful_file_ids": missing_file_ids,
        "bad_filenames": bad_filenames,
        "other_root_files": other_root_files,
        "staging_file_count_excluded": staging_count,
        "sha256_mismatch_count": sum(not row["sha256_match"] for row in records),
        "provenance_mismatch_count": sum(not row["provenance_match"] for row in records),
        "issue_counts": dict(sorted(issue_counts.items())),
        "review_status": "REVIEW_REQUIRED; no candidate accepted by this technical audit",
        "duration_method": "decoded_frame_count / container FPS; not a semantic gesture duration",
        "long_video_caveat": "Full source videos may contain many signs; WLASL frame_start/frame_end must be reviewed before any sequence extraction.",
        "container_frame_count_policy": "reported only; actual count from full decode",
        "mediapipe_run": False,
        "feature_extraction_run": False,
        "report_csv": OUTPUT_CSV.relative_to(ROOT).as_posix(),
    }
    with OUTPUT_CSV.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    with OUTPUT_JSON.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"source_video_count": len(video_files), "decode_success": summary["decode_success"], "short_under_16": summary["short_under_16_count"]}))


if __name__ == "__main__":
    main()
