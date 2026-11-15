"""User-operated J/Z/NONE sequence extraction (baseline B, 16 x 69).

Use --check-only to validate paths/schema without opening a video or writing.
Use --preview to make temporary 4x4 contact sheets, without MediaPipe or a dataset.
Use --motion-preview to compare those exact clips against a landmark-motion crop.
Official --execute samples 16 frames evenly from the list of frames with 21
valid landmarks. Motion-aware sampling remains preview-only and experimental.
Source videos are never modified, moved, or renamed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.feature_extraction.dynamic_sequence import (  # noqa: E402
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    DynamicSequenceError,
    build_sequence_features,
    detect_active_interval,
    sample_frame_indices,
    sample_valid_frame_indices,
)
from ai.feature_extraction.hand_landmarks import landmarks_to_array  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "dynamic.json"
PREVIEW_ROOT = ROOT / "data" / "experiments" / "dynamic_jz_v1" / "preview"
MOTION_PREVIEW_ROOT = ROOT / "data" / "experiments" / "dynamic_jz_v1" / "motion_preview"
AUDIT_FIELDS = (
    "source_path", "label", "z_variant", "decoded_frame_count",
    "valid_landmark_frames", "selected_original_frame_indices",
    "feature_dim", "sequence_length",
    "extraction_status", "failure_reason", "hand_scale_px",
)


class VideoExtractionError(ValueError):
    """A video or one of its selected frames cannot be extracted safely."""


def _project_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path outside project: {value}")
    return path


def load_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("task") != "dynamic_asl":
        raise ValueError("Unexpected dynamic task")
    labels = config.get("dynamic_labels")
    if labels != ["J", "Z", "NONE"]:
        raise ValueError("Expected J/Z/NONE label schema")
    spec = config.get("sequence_extraction", {})
    if (spec.get("feature_schema") != "baseline_b_v1"
            or spec.get("sequence_length") != SEQUENCE_LENGTH
            or spec.get("feature_dim") != FEATURE_DIM):
        raise ValueError("Sequence feature schema does not match baseline B")
    if spec.get("sampling_policy_status") != "VALID_LANDMARK_FRAMES_V1":
        raise ValueError("Unexpected official sequence sampling policy state")
    paths = {
        "source_root": _project_path(config["raw_external_root"]),
        "mediapipe_model": _project_path(spec["mediapipe_model"]),
        "output_jsonl": _project_path(spec["output_jsonl"]),
        "audit_csv": _project_path(spec["audit_csv"]),
        "coverage_audit_csv": _project_path(spec["coverage_audit_csv"]),
        "coverage_summary_json": _project_path(spec["coverage_summary_json"]),
    }
    experiment_root = _project_path("data/experiments/dynamic_jz_v1")
    for key in ("output_jsonl", "audit_csv"):
        if not paths[key].is_relative_to(experiment_root):
            raise ValueError(f"{key} must stay in the temporary dynamic experiment")
    metadata_root = _project_path("data/metadata/dynamic")
    for key in ("coverage_audit_csv", "coverage_summary_json"):
        if not paths[key].is_relative_to(metadata_root):
            raise ValueError(f"{key} must stay in dynamic metadata")
    if not paths["source_root"].is_dir():
        raise FileNotFoundError(f"Source video root missing: {paths['source_root']}")
    if not paths["mediapipe_model"].is_file():
        raise FileNotFoundError(f"MediaPipe task missing: {paths['mediapipe_model']}")
    if not (paths["source_root"] / "J").is_dir() or not (paths["source_root"] / "Z").is_dir():
        raise FileNotFoundError("Source J and Z directories are both required")
    motion = config.get("motion_preview", {})
    if not isinstance(motion, dict):
        raise ValueError("motion_preview config must be an object")
    required_integers = ("median_window", "mean_window", "min_active_edges",
                         "max_gap_frames", "margin_frames")
    required_floats = ("min_valid_fraction", "min_peak_score", "min_peak_noise_ratio",
                       "noise_mad_multiplier", "relative_peak_fraction",
                       "secondary_mass_ratio")
    for key in required_integers:
        if not isinstance(motion.get(key), int) or motion[key] < (0 if key in {"max_gap_frames", "margin_frames"} else 1):
            raise ValueError(f"Invalid motion_preview {key}")
    if motion["median_window"] % 2 != 1 or motion["mean_window"] % 2 != 1:
        raise ValueError("Motion smoothing windows must be odd")
    for key in required_floats:
        value = motion.get(key)
        if not isinstance(value, (int, float)) or not (0 < value <= 1 if key in {
                "min_valid_fraction", "relative_peak_fraction", "secondary_mass_ratio"} else value > 0):
            raise ValueError(f"Invalid motion_preview {key}")
    motion_paths = {
        "source_preview_manifest": _project_path(motion["source_preview_manifest"]),
        "output_dir": _project_path(motion["output_dir"]),
    }
    for key, path in motion_paths.items():
        if not path.is_relative_to(experiment_root):
            raise ValueError(f"{key} must stay inside the dynamic experiment")
    return {"labels": labels, "paths": paths, "spec": spec,
            "motion_params": {key: motion[key] for key in (*required_integers, *required_floats)},
            "motion_paths": motion_paths}


def discover_videos(source_root: Path, labels: list[str]) -> list[tuple[Path, str]]:
    videos = []
    for path in source_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".avi":
            continue
        relative = path.relative_to(source_root)
        label = relative.parts[0].upper()
        if label not in labels or len(relative.parts) < 2:
            raise ValueError(f"AVI outside a known label folder: {relative}")
        videos.append((path, label))
    return sorted(videos, key=lambda item: item[0].relative_to(source_root).as_posix().casefold())


def load_coverage_contract(videos: list[tuple[Path, str]], paths: dict) -> dict[str, dict]:
    """Require the complete 712-video audit before official extraction starts."""
    summary = json.loads(paths["coverage_summary_json"].read_text(encoding="utf-8"))
    counts = Counter(label for _, label in videos)
    if (summary.get("status") != "COMPLETE" or summary.get("video_count") != 712
            or counts != {"J": 316, "Z": 396}
            or summary.get("report_csv") != paths["coverage_audit_csv"].relative_to(ROOT).as_posix()):
        raise ValueError("Coverage summary or source inventory does not match the approved 712 videos")
    if any(summary.get("per_label", {}).get(label, {}).get("video_audit_errors") != 0
           for label in ("J", "Z")):
        raise ValueError("Coverage audit contains video errors")
    available = {path.relative_to(ROOT).as_posix(): label for path, label in videos}
    records = {}
    with paths["coverage_audit_csv"].open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source_path", "label", "decoded_frame_count", "valid_landmark_frames",
                    "invalid_landmark_frames", "landmark_coverage_ratio",
                    "has_16_valid_frames", "audit_status"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Coverage audit CSV schema mismatch")
        for row in reader:
            source = row["source_path"]
            if source not in available or source in records or row["label"] != available[source]:
                raise ValueError(f"Coverage audit source/label mismatch or duplicate: {source}")
            decoded = int(row["decoded_frame_count"])
            valid = int(row["valid_landmark_frames"])
            invalid = int(row["invalid_landmark_frames"])
            if (row["audit_status"] != "COMPLETE" or decoded <= 0 or valid < 0
                    or invalid < 0 or valid + invalid != decoded
                    or row["has_16_valid_frames"] not in {"True", "False"}
                    or (row["has_16_valid_frames"] == "True") != (valid >= SEQUENCE_LENGTH)
                    or abs(float(row["landmark_coverage_ratio"]) - valid / decoded) > 1e-12):
                raise ValueError(f"Invalid coverage audit values: {source}")
            records[source] = {"decoded_frame_count": decoded, "valid_landmark_frames": valid}
    if set(records) != set(available):
        raise ValueError("Coverage audit does not contain exactly the current source videos")
    for label, expected_usable in (("J", 308), ("Z", 375)):
        label_records = [record for source, record in records.items() if available[source] == label]
        if (summary["per_label"][label]["videos"] != len(label_records)
                or summary["per_label"][label]["videos_at_least_16_valid_frames"] != expected_usable
                or summary["per_label"][label]["videos_fewer_than_16_valid_frames"] != len(label_records) - expected_usable
                or sum(record["valid_landmark_frames"] >= SEQUENCE_LENGTH
                       for record in label_records) != expected_usable):
            raise ValueError(f"Coverage audit class counts mismatch: {label}")
    return records


def _count_decoded_frames(path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoExtractionError("VIDEO_OPEN_FAIL")
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise VideoExtractionError(f"EMPTY_DECODED_FRAME_{count}")
            count += 1
        if count == 0:
            raise VideoExtractionError("NO_DECODED_FRAMES")
        return count
    finally:
        capture.release()


def select_preview_videos(
    videos: list[tuple[Path, str]], source_root: Path,
    requested: list[str], count_per_label: int,
) -> list[tuple[Path, str]]:
    """Use explicit source-relative paths, or evenly spread 5 J + 5 Z by default."""
    if count_per_label < 1:
        raise ValueError("Preview count per label must be positive")
    if requested:
        available = {path.resolve(): (path, label) for path, label in videos}
        selected = []
        seen = set()
        for value in requested:
            candidate = (source_root / value).resolve()
            if not candidate.is_relative_to(source_root.resolve()) or candidate not in available:
                raise ValueError(f"Unknown or out-of-root preview video: {value}")
            if candidate in seen:
                raise ValueError(f"Duplicate preview video: {value}")
            selected.append(available[candidate])
            seen.add(candidate)
        return selected

    selected = []
    for label in ("J", "Z"):
        label_videos = [item for item in videos if item[1] == label]
        if not label_videos:
            raise ValueError(f"No {label} videos available for default preview")
        count = min(count_per_label, len(label_videos))
        positions = [0] if count == 1 else sample_frame_indices(len(label_videos), count)
        selected.extend(label_videos[index] for index in positions)
    return selected


def _selected_preview_frames(path: Path, indices: list[int]) -> list:
    """Decode source video sequentially; return only the 16 selected raw frames."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not capture.isOpened():
            raise VideoExtractionError("PREVIEW_VIDEO_OPEN_FAIL")
        selected_position = 0
        frame_index = 0
        while selected_position < len(indices):
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                raise VideoExtractionError(f"PREVIEW_DECODE_FAIL_{frame_index}")
            if frame_index == indices[selected_position]:
                frames.append(frame.copy())
                selected_position += 1
            frame_index += 1
    finally:
        capture.release()
    return frames


def make_preview_sheet(frames: list, indices: list[int], relative_path: str):
    """Draw 16 chronological frames with their actual zero-based video indices."""
    import cv2
    import numpy as np

    if len(frames) != SEQUENCE_LENGTH or len(indices) != SEQUENCE_LENGTH:
        raise VideoExtractionError("INVALID_PREVIEW_FRAME_COUNT")
    tile_width, tile_height, caption_height, banner_height = 320, 240, 32, 42
    sheet = np.full((banner_height + 4 * (tile_height + caption_height),
                     4 * tile_width, 3), 30, dtype=np.uint8)
    cv2.putText(sheet, f"{relative_path} | 16 frames, chronological order", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    for position, (frame, frame_index) in enumerate(zip(frames, indices)):
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise VideoExtractionError(f"INVALID_PREVIEW_FRAME_{frame_index}")
        height, width = frame.shape[:2]
        scale = min(tile_width / width, tile_height / height)
        resized = cv2.resize(frame, (max(1, round(width * scale)),
                                     max(1, round(height * scale))))
        row, column = divmod(position, 4)
        x0 = column * tile_width
        y0 = banner_height + row * (tile_height + caption_height)
        image_x = x0 + (tile_width - resized.shape[1]) // 2
        image_y = y0 + (tile_height - resized.shape[0]) // 2
        sheet[image_y:image_y + resized.shape[0], image_x:image_x + resized.shape[1]] = resized
        cv2.putText(sheet, f"{position + 1:02d}/16  frame {frame_index}",
                    (x0 + 8, y0 + tile_height + 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return sheet


def preview(videos: list[tuple[Path, str]], source_root: Path, output_dir: Path) -> list[dict]:
    """Write only temporary PNG sheets and a preview manifest; never JSONL."""
    import cv2

    experiment_root = _project_path("data/experiments/dynamic_jz_v1")
    if not output_dir.resolve().is_relative_to(experiment_root):
        raise ValueError("Preview directory must stay inside the dynamic experiment")
    manifest_path = output_dir / "preview_manifest.json"
    targets = []
    for path, label in videos:
        relative = path.relative_to(source_root).as_posix()
        safe_stem = re.sub(r"[^A-Za-z0-9_-]", "_", path.stem)[:40]
        suffix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:10]
        targets.append((path, label, relative, output_dir / f"{label}_{safe_stem}_{suffix}.png"))
    if len({target for _, _, _, target in targets}) != len(targets):
        raise ValueError("Preview output filename collision")
    for target in [manifest_path, *(target for _, _, _, target in targets)]:
        if target.exists():
            raise FileExistsError(f"Preview output already exists; no overwrite: {target}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path, label, relative, target in targets:
        decoded_count = _count_decoded_frames(path)
        indices = sample_frame_indices(decoded_count)
        frames = _selected_preview_frames(path, indices)
        sheet = make_preview_sheet(frames, indices, relative)
        ok, encoded = cv2.imencode(".png", sheet)
        if not ok:
            raise VideoExtractionError(f"PREVIEW_PNG_ENCODE_FAIL:{relative}")
        with target.open("xb") as handle:
            handle.write(encoded.tobytes())
        records.append({"source_path": path.relative_to(ROOT).as_posix(),
                        "label": label, "decoded_frame_count": decoded_count,
                        "sampled_frame_indices": indices,
                        "preview_path": target.relative_to(ROOT).as_posix()})
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump({"mode": "preview_only", "sequence_length": SEQUENCE_LENGTH,
                   "videos": records, "official_dataset_created": False},
                  handle, indent=2)
    return records


def original_preview_videos(manifest_path: Path, videos: list[tuple[Path, str]]) -> list[tuple[Path, str, dict]]:
    """Lock the comparison to the existing 5 J + 5 Z preview, not a new draw."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("videos", [])
    if manifest.get("mode") != "preview_only" or len(records) != 10:
        raise ValueError("Expected the original 10-video preview manifest")
    if Counter(record.get("label") for record in records) != {"J": 5, "Z": 5}:
        raise ValueError("Original preview must have exactly 5 J and 5 Z")
    available = {path.resolve(): (path, label) for path, label in videos}
    chosen = []
    seen = set()
    for record in records:
        path = _project_path(record["source_path"])
        if path not in available or path in seen or available[path][1] != record["label"]:
            raise ValueError(f"Original preview video missing or inconsistent: {path}")
        if not _project_path(record["preview_path"]).is_file():
            raise FileNotFoundError(f"Original preview sheet missing: {record['preview_path']}")
        chosen.append((*available[path], record))
        seen.add(path)
    return chosen


def _all_motion_landmarks(path: Path, landmarker) -> tuple[list, list[dict]]:
    """Decode chronologically and retain only pixel-XY landmarks, never images."""
    import cv2
    import mediapipe as mp
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    points = []
    detection_failures = []
    try:
        if not capture.isOpened():
            raise VideoExtractionError("MOTION_VIDEO_OPEN_FAIL")
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise VideoExtractionError(f"MOTION_EMPTY_DECODED_FRAME_{frame_index}")
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                raise VideoExtractionError(f"MOTION_INVALID_FRAME_SIZE_{frame_index}")
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # No flip or mirror.
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if not result.hand_landmarks:
                    raise ValueError("NO_HAND")
                if len(result.hand_landmarks[0]) != 21:
                    raise ValueError("INVALID_LANDMARK_COUNT")
                raw = np.asarray([(float(p.x), float(p.y)) for p in result.hand_landmarks[0]],
                                 dtype=np.float64)
                if raw.shape != (21, 2) or not np.isfinite(raw).all():
                    raise ValueError("INVALID_LANDMARK_VALUES")
                points.append(raw * np.array([width, height], dtype=np.float64))
            except Exception as exc:
                points.append(None)  # Auditable missing frame; never interpolate.
                detection_failures.append({"frame_index": frame_index,
                                           "reason": str(exc) if isinstance(exc, ValueError)
                                           else type(exc).__name__})
            frame_index += 1
    finally:
        capture.release()
    if not points:
        raise VideoExtractionError("MOTION_NO_DECODED_FRAMES")
    return points, detection_failures


def make_motion_comparison_sheet(full_sheet, motion_sheet, relative: str, result: dict):
    import cv2
    import numpy as np

    if motion_sheet is None:
        motion_sheet = np.full_like(full_sheet, 30)
        cv2.putText(motion_sheet, "B: REVIEW_REQUIRED", (24, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(motion_sheet, result["failure_reason"][:65], (24, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    banner = 52
    combined = np.full((full_sheet.shape[0] + banner, full_sheet.shape[1] * 2, 3),
                       30, dtype=np.uint8)
    combined[banner:, :full_sheet.shape[1]] = full_sheet
    combined[banner:, full_sheet.shape[1]:] = motion_sheet
    active = f"{result['active_start']}-{result['active_end']}" if result["active_start"] is not None else "UNKNOWN"
    margin = f"{result['margin_start']}-{result['margin_end']}" if result["margin_start"] is not None else "UNKNOWN"
    cv2.putText(combined, f"{relative} | active {active} | margin {margin} | {result['status']}",
                (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return combined


def motion_preview(videos: list[tuple[Path, str, dict | None]], source_root: Path,
                   output_dir: Path, model_path: Path, params: dict) -> list[dict]:
    """Save A/B contact sheets and score metadata; never make training JSONL."""
    import cv2
    from ai.feature_extraction.hand_landmarks import create_hand_landmarker

    experiment_root = _project_path("data/experiments/dynamic_jz_v1")
    if not output_dir.resolve().is_relative_to(experiment_root):
        raise ValueError("Motion preview directory must stay inside the dynamic experiment")
    manifest_path = output_dir / "motion_preview_manifest.json"
    targets = []
    for path, label, original in videos:
        relative = path.relative_to(source_root).as_posix()
        safe_stem = re.sub(r"[^A-Za-z0-9_-]", "_", path.stem)[:40]
        suffix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:10]
        targets.append((path, label, original, relative,
                        output_dir / f"{label}_{safe_stem}_{suffix}_comparison.png"))
    for target in [manifest_path, *(item[-1] for item in targets)]:
        if target.exists():
            raise FileExistsError(f"Motion preview already exists; no overwrite: {target}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with create_hand_landmarker(model_path, running_mode="IMAGE", num_hands=1) as landmarker:
        for path, label, original, relative, target in targets:
            points, detection_failures = _all_motion_landmarks(path, landmarker)
            count = len(points)
            full_indices = sample_frame_indices(count)
            result = detect_active_interval(points, params)
            if original is not None and (count != original["decoded_frame_count"]
                                         or full_indices != original["sampled_frame_indices"]):
                result["status"] = "REVIEW_REQUIRED"
                result["failure_reason"] = "SOURCE_CHANGED_SINCE_ORIGINAL_PREVIEW"
                result["sampled_frame_indices"] = []
            full_frames = _selected_preview_frames(path, full_indices)
            full_sheet = make_preview_sheet(full_frames, full_indices, f"A full clip: {relative}")
            motion_sheet = None
            if result["status"] == "OK":
                motion_indices = result["sampled_frame_indices"]
                motion_frames = _selected_preview_frames(path, motion_indices)
                motion_sheet = make_preview_sheet(motion_frames, motion_indices,
                                                  f"B motion-aware: {relative}")
            combined = make_motion_comparison_sheet(full_sheet, motion_sheet, relative, result)
            ok, encoded = cv2.imencode(".png", combined)
            if not ok:
                raise VideoExtractionError(f"MOTION_PREVIEW_PNG_ENCODE_FAIL:{relative}")
            with target.open("xb") as handle:
                handle.write(encoded.tobytes())
            records.append({"source_path": path.relative_to(ROOT).as_posix(), "label": label,
                            "decoded_frame_count": count, "full_clip_indices": full_indices,
                            "landmark_failures": detection_failures,
                            "comparison_path": target.relative_to(ROOT).as_posix(), **result})
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump({"mode": "motion_preview_only", "sequence_length": SEQUENCE_LENGTH,
                   "feature_formula_changed": False, "official_dataset_created": False,
                   "motion_parameters": params, "videos": records}, handle, indent=2,
                  allow_nan=False)
    return records


def _all_valid_landmarks(path: Path, landmarker) -> tuple[int, list[tuple[int, list, tuple[int, int]]], str]:
    """Decode every frame once and retain only validated 21-landmark frames."""
    import cv2
    import mediapipe as mp

    try:
        capture = cv2.VideoCapture(str(path))  # Video file path; never a webcam index.
    except Exception as exc:
        raise VideoExtractionError(f"VIDEO_OPEN_ERROR:{type(exc).__name__}") from exc
    valid_entries = []
    first_detection_error = ""
    decoded_count = 0
    decode_error = ""
    try:
        if not capture.isOpened():
            return 0, valid_entries, "VIDEO_OPEN_FAIL"
        while True:
            try:
                ok, frame = capture.read()
            except Exception as exc:
                decode_error = f"VIDEO_READ_ERROR_AT_FRAME_{decoded_count}:{type(exc).__name__}"
                break
            if not ok:
                break
            if (frame is None or frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3):
                decode_error = f"INVALID_DECODED_FRAME_{decoded_count}"
                break
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                decode_error = f"INVALID_FRAME_SIZE_{decoded_count}"
                break
            frame_index = decoded_count
            decoded_count += 1
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # No flip or mirror.
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            except Exception as exc:
                if not first_detection_error:
                    first_detection_error = f"LANDMARK_DETECTION_ERROR_AT_FRAME_{frame_index}:{type(exc).__name__}"
                continue
            if not result.hand_landmarks:
                continue
            hand = result.hand_landmarks[0]
            try:
                landmarks_to_array(hand)  # Enforce exactly 21 finite XYZ landmarks.
            except (AttributeError, TypeError, ValueError):
                continue
            valid_entries.append((frame_index, hand, (width, height)))
    finally:
        capture.release()
    if decoded_count == 0 and not decode_error:
        decode_error = "NO_DECODED_FRAMES"
    return decoded_count, valid_entries, decode_error or first_detection_error


def extract_one(path: Path, label: str, landmarker,
                expected_coverage: dict) -> tuple[dict, list[list[float]] | None]:
    row = {
        "source_path": path.relative_to(ROOT).as_posix(),
        "label": label,
        "z_variant": "UNKNOWN" if label == "Z" else "NOT_APPLICABLE",
        "decoded_frame_count": 0,
        "valid_landmark_frames": 0,
        "selected_original_frame_indices": [],
        "feature_dim": FEATURE_DIM,
        "sequence_length": SEQUENCE_LENGTH,
        "extraction_status": "REVIEW_REQUIRED",
        "failure_reason": "",
        "hand_scale_px": "",
    }
    try:
        count, valid_entries, video_error = _all_valid_landmarks(path, landmarker)
        row["decoded_frame_count"] = count
        row["valid_landmark_frames"] = len(valid_entries)
        if video_error:
            raise VideoExtractionError(video_error)
        if (count != expected_coverage["decoded_frame_count"]
                or len(valid_entries) != expected_coverage["valid_landmark_frames"]):
            raise VideoExtractionError("LANDMARK_COVERAGE_DIFFERS_FROM_AUDIT")
        original_indices = sample_valid_frame_indices([entry[0] for entry in valid_entries])
        row["selected_original_frame_indices"] = original_indices
        by_index = {index: (hand, size) for index, hand, size in valid_entries}
        frames = [by_index[index][0] for index in original_indices]
        sizes = [by_index[index][1] for index in original_indices]
        features, scale = build_sequence_features(frames, sizes)
        if features.shape != (SEQUENCE_LENGTH, FEATURE_DIM) or not np.isfinite(features).all():
            raise VideoExtractionError("INVALID_SEQUENCE_FEATURES")
        row["hand_scale_px"] = scale
        row["extraction_status"] = "SUCCESS"
        return row, features.tolist()
    except (VideoExtractionError, DynamicSequenceError) as exc:
        row["failure_reason"] = str(exc)
        return row, None


def execute(videos: list[tuple[Path, str]], paths: dict, coverage: dict[str, dict]) -> bool:
    from ai.feature_extraction.hand_landmarks import create_hand_landmarker

    output = paths["output_jsonl"]
    audit = paths["audit_csv"]
    partial_output = output.with_suffix(output.suffix + ".partial")
    partial_audit = audit.with_suffix(audit.suffix + ".partial")
    for target in (output, audit, partial_output, partial_audit):
        if target.exists():
            raise FileExistsError(f"Output already exists; no overwrite: {target}")
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)

    status_counts = Counter()
    per_label = {"J": Counter(), "Z": Counter()}
    unexpected_review_reasons = Counter()
    # Partial files remain visibly incomplete if the process is interrupted;
    # never auto-delete or overwrite an earlier extraction.
    with create_hand_landmarker(paths["mediapipe_model"], running_mode="IMAGE", num_hands=1) as landmarker, \
            partial_output.open("x", encoding="utf-8", newline="") as data_file, \
            partial_audit.open("x", encoding="utf-8", newline="") as audit_file:
        writer = csv.DictWriter(audit_file, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for position, (path, label) in enumerate(videos, 1):
            source = path.relative_to(ROOT).as_posix()
            row, features = extract_one(path, label, landmarker, coverage[source])
            writer.writerow({**row, "selected_original_frame_indices": json.dumps(
                row["selected_original_frame_indices"])})
            audit_file.flush()
            if features is not None:
                data_file.write(json.dumps({**row, "features": features}, allow_nan=False) + "\n")
                data_file.flush()
            status_counts[row["extraction_status"]] += 1
            per_label[label][row["extraction_status"]] += 1
            if row["extraction_status"] == "REVIEW_REQUIRED" and row["failure_reason"] != "INSUFFICIENT_VALID_LANDMARK_FRAMES":
                unexpected_review_reasons[row["failure_reason"]] += 1
            if position % 25 == 0 or position == len(videos):
                print(f"Official extraction: {position}/{len(videos)} videos", flush=True)

    partial_output.rename(output)
    partial_audit.rename(audit)
    expected_success = sum(record["valid_landmark_frames"] >= SEQUENCE_LENGTH
                           for record in coverage.values())
    expected_match = (status_counts["SUCCESS"] == expected_success
                      and status_counts["REVIEW_REQUIRED"] == len(videos) - expected_success
                      and not unexpected_review_reasons)
    print(json.dumps({"status": "COMPLETE_WITH_REVIEW" if expected_match else "REVIEW_REQUIRED",
                      "total": len(videos), "success": status_counts["SUCCESS"],
                      "review_required": status_counts["REVIEW_REQUIRED"],
                      "expected_success_from_coverage_audit": expected_success,
                      "matches_coverage_audit_counts": expected_match,
                      "per_label": {label: dict(counts) for label, counts in per_label.items()},
                      "unexpected_review_reasons": dict(unexpected_review_reasons),
                      "output": str(output),
                      "audit": str(audit)}))
    return expected_match


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Read-only config/path/schema check")
    mode.add_argument("--preview", action="store_true", help="Create temporary 16-frame contact sheets")
    mode.add_argument("--motion-preview", action="store_true",
                      help="Compare original 5 J + 5 Z previews with motion-aware sampling")
    mode.add_argument("--execute", action="store_true",
                      help="Official valid-landmark-frame extraction after confirmation")
    parser.add_argument("--video", action="append", default=[], metavar="J/1.avi",
                        help="Preview this source-relative AVI (repeatable)")
    parser.add_argument("--preview-count-per-label", type=int, default=5, metavar="N",
                        help="Default preview count for each of J and Z (default: 5)")
    parser.add_argument("--preview-dir", type=Path, default=PREVIEW_ROOT,
                        help="New output directory under data/experiments/dynamic_jz_v1/")
    args = parser.parse_args(argv)
    try:
        if not (args.preview or args.motion_preview) and args.video:
            raise ValueError("--video requires --preview or --motion-preview")
        if not args.preview and (args.preview_count_per_label != 5 or args.preview_dir != PREVIEW_ROOT):
            raise ValueError("--preview-count-per-label and --preview-dir require --preview")
        config = load_config()
        videos = discover_videos(config["paths"]["source_root"], config["labels"])
        counts = Counter(label for _, label in videos)
        if not videos:
            raise ValueError("No AVI videos found")
        coverage = load_coverage_contract(videos, config["paths"]) if args.check_only or args.execute else None
        preflight = {"status": "CHECK_PASS", "source_root": str(config["paths"]["source_root"]),
                     "videos": dict(counts), "sequence_shape": [SEQUENCE_LENGTH, FEATURE_DIM],
                     "official_sampling_policy": config["spec"]["sampling_policy_status"],
                     "coverage_audit_verified": coverage is not None,
                     "expected_usable": sum(record["valid_landmark_frames"] >= SEQUENCE_LENGTH
                                            for record in coverage.values()) if coverage is not None else None,
                     "output": str(config["paths"]["output_jsonl"]),
                     "audit": str(config["paths"]["audit_csv"])}
        print(json.dumps(preflight))
        if args.check_only:
            return 0
        if args.preview:
            selected = select_preview_videos(videos, config["paths"]["source_root"],
                                             args.video, args.preview_count_per_label)
            output_dir = _project_path(str(args.preview_dir))
            records = preview(selected, config["paths"]["source_root"], output_dir)
            print(json.dumps({"status": "PREVIEW_COMPLETE", "videos": len(records),
                              "output_dir": str(output_dir), "official_dataset_created": False}))
            return 0
        if args.motion_preview:
            if args.video:
                chosen = select_preview_videos(videos, config["paths"]["source_root"], args.video, 5)
                selected = [(path, label, None) for path, label in chosen]
            else:
                selected = original_preview_videos(
                    config["motion_paths"]["source_preview_manifest"], videos)
            records = motion_preview(selected, config["paths"]["source_root"],
                                     config["motion_paths"]["output_dir"],
                                     config["paths"]["mediapipe_model"], config["motion_params"])
            print(json.dumps({"status": "MOTION_PREVIEW_COMPLETE",
                              "videos": len(records),
                              "ok": sum(record["status"] == "OK" for record in records),
                              "review_required": sum(record["status"] == "REVIEW_REQUIRED" for record in records),
                              "output_dir": str(config["motion_paths"]["output_dir"]),
                              "official_dataset_created": False}))
            return 0
        if input("Extract all 712 J/Z videos using valid-landmark-frame sampling? [y/N] ").strip().lower() != "y":
            print("Official extraction cancelled; no dataset written")
            return 0
        return 0 if execute(videos, config["paths"], coverage) else 1
    except (OSError, ValueError, KeyError, EOFError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
