"""User-operated visual review sheets for downloaded WLASL OTHER_GESTURE candidates.

No MediaPipe, feature extraction, automatic acceptance, or source-video writes.
--check-only validates inputs and planned frame ranges without rendering.
--preview creates 16-frame contact sheets and a review manifest after confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "data" / "experiments" / "dynamic_none_v1"
RAW = EXPERIMENT / "raw" / "wlasl_other_gesture"
META = EXPERIMENT / "metadata"
CANDIDATES = META / "wlasl_other_gesture_candidates_v1.csv"
TECHNICAL = META / "wlasl_other_gesture_technical_audit_v1.csv"
OUTPUT = EXPERIMENT / "review" / "wlasl_other_gesture"
REVIEW_MANIFEST = OUTPUT / "review_manifest_v1.csv"
BATCH_INDEX = OUTPUT / "batch_index_v1.csv"
REVIEW_GUIDE = OUTPUT / "review_status_guide_v1.json"
METADATA_FPS = 25.0  # WLASL v0.3 frame_start/end: 1-based at 25 FPS.
LONG_VIDEO_SECONDS = 60.0
SEQUENCE_LENGTH = 16
DEFAULT_BATCH_SIZE = 25
REVIEW_OPTIONS = (
    "REVIEW_REQUIRED", "ACCEPT_OTHER_GESTURE", "REJECT_J_OR_Z_RISK",
    "REJECT_AMBIGUOUS", "REJECT_WRONG_SEGMENT", "RANGE_REVIEW_REQUIRED",
)
MANIFEST_FIELDS = (
    "video_id", "gloss", "signer_id", "source", "local_path",
    "metadata_frame_start", "metadata_frame_end", "metadata_fps_basis",
    "preview_start", "preview_end", "selected_original_frame_indices",
    "preview_mode", "range_review_required", "local_fps", "duration_seconds",
    "contact_sheet_path", "batch", "review_status", "review_reason",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def choose_range(
    metadata_start: int, metadata_end: int, decoded_count: int,
    local_fps: float, full_duration_seconds: float,
) -> tuple[int, int, str, bool, str]:
    """Return inclusive 0-based indices in the downloaded source video.

    WLASL's 1-based metadata indexes a 25-FPS decode. Convert endpoints through
    time if the downloaded video has another FPS; never assume same frame index.
    A navigation-only full-clip sheet is not an accepted gesture segment.
    """
    if decoded_count < SEQUENCE_LENGTH or not math.isfinite(local_fps) or local_fps <= 0:
        raise ValueError("Video lacks 16 decodable frames or valid local FPS")
    if not math.isfinite(full_duration_seconds) or full_duration_seconds <= 0:
        raise ValueError("Invalid source-video duration")
    whole_end = decoded_count - 1
    if metadata_start >= 1 and metadata_end >= metadata_start:
        start = round((metadata_start - 1) * local_fps / METADATA_FPS)
        end = round((metadata_end - 1) * local_fps / METADATA_FPS)
        if 0 <= start <= end < decoded_count and end - start + 1 >= SEQUENCE_LENGTH:
            reason = "METADATA_RANGE_25FPS_TIME_MAPPED"
            if abs(local_fps - METADATA_FPS) > 0.1:
                reason += ";VERIFY_FPS_MAPPING_AND_SEGMENT_BOUNDARIES"
            return start, end, "METADATA_RANGE_PREVIEW", False, reason
        return 0, whole_end, "FULL_CLIP_NAVIGATION_ONLY", True, "RANGE_REVIEW_REQUIRED:METADATA_RANGE_OUTSIDE_OR_TOO_SHORT"
    if metadata_start == 1 and metadata_end == -1:
        if full_duration_seconds > LONG_VIDEO_SECONDS:
            return 0, whole_end, "FULL_CLIP_NAVIGATION_ONLY", True, "RANGE_REVIEW_REQUIRED:LONG_VIDEO_WITH_UNKNOWN_END_NOT_A_GESTURE_SEGMENT"
        return 0, whole_end, "FULL_CLIP_PREVIEW", False, "FULL_CLIP_PREVIEW:METADATA_END_UNKNOWN"
    return 0, whole_end, "FULL_CLIP_NAVIGATION_ONLY", True, "RANGE_REVIEW_REQUIRED:INVALID_METADATA_RANGE"


def evenly_spaced_indices(start: int, end: int) -> list[int]:
    if end - start + 1 < SEQUENCE_LENGTH:
        raise ValueError("Preview range has fewer than 16 frames")
    indices = [round(start + (end - start) * i / (SEQUENCE_LENGTH - 1))
               for i in range(SEQUENCE_LENGTH)]
    if len(set(indices)) != SEQUENCE_LENGTH or indices[0] != start or indices[-1] != end:
        raise ValueError("Temporal preview sampling is not unique or complete")
    return indices


def load_plan() -> list[dict[str, object]]:
    for path in (CANDIDATES, TECHNICAL):
        if not path.is_file():
            raise FileNotFoundError(path)
    candidates = read_csv(CANDIDATES)
    technical = read_csv(TECHNICAL)
    if len(candidates) != 220 or len(technical) != 121:
        raise ValueError("Expected 220 candidates and 121 technical-audited videos")
    by_id = {row["video_id"]: row for row in candidates}
    if len(by_id) != 220 or len({row["video_id"] for row in technical}) != 121:
        raise ValueError("Duplicate video_id in input CSV")
    plan = []
    for row in sorted(technical, key=lambda item: item["video_id"]):
        video_id = row["video_id"]
        candidate = by_id.get(video_id)
        if candidate is None:
            raise ValueError(f"Missing candidate metadata for {video_id}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
            raise ValueError(f"Unsafe video_id: {video_id}")
        if candidate["review_status"] != row["review_status"] or row["review_status"] != "REVIEW_REQUIRED":
            raise ValueError(f"Unexpected review status: {video_id}")
        for field in ("gloss", "signer_id", "source"):
            if candidate[field] != row[field]:
                raise ValueError(f"Candidate/technical provenance mismatch: {video_id}/{field}")
        if candidate["frame_start"] != row["frame_start"] or candidate["frame_end"] != row["frame_end"]:
            raise ValueError(f"Candidate/technical range mismatch: {video_id}")
        if float(candidate["fps"]) != METADATA_FPS:
            raise ValueError(f"Unexpected metadata FPS basis: {video_id}")
        if row["sha256_match"] != "True" or row["provenance_match"] != "True" or row["decode_ok"] != "True":
            raise ValueError(f"Technical audit has unresolved integrity/decode issue: {video_id}")
        path = (ROOT / row["local_path"]).resolve()
        if not path.is_relative_to(RAW.resolve()) or path.parent != RAW.resolve():
            raise ValueError(f"Source path outside completed video directory: {video_id}")
        if path.stem != f"WLASL_{video_id}" or not path.is_file():
            raise ValueError(f"Missing or unexpected source file: {video_id}")
        if file_sha256(path) != row["actual_sha256"]:
            raise ValueError(f"Source video changed since technical audit: {video_id}")
        decoded_count = int(row["decoded_frame_count"])
        fps = float(row["fps"])
        duration = float(row["duration_seconds"])
        metadata_start = int(row["frame_start"])
        metadata_end = int(row["frame_end"])
        start, end, mode, range_review, reason = choose_range(
            metadata_start, metadata_end, decoded_count, fps, duration,
        )
        plan.append({
            "video_id": video_id, "gloss": row["gloss"],
            "signer_id": row["signer_id"], "source": row["source"],
            "path": path, "local_path": row["local_path"],
            "metadata_frame_start": metadata_start,
            "metadata_frame_end": metadata_end,
            "decoded_count": decoded_count, "local_fps": fps,
            "duration_seconds": duration, "preview_start": start,
            "preview_end": end, "preview_mode": mode,
            "range_review_required": range_review,
            "review_reason": reason,
            "indices": evenly_spaced_indices(start, end),
        })
    if len({str(item["local_path"]).casefold() for item in plan}) != len(plan):
        raise ValueError("Duplicate local_path in technical audit")
    return plan


def selected_frames(path: Path, indices: list[int], *, expected_count: int = SEQUENCE_LENGTH) -> list[tuple[int, object]]:
    """Sequential decode avoids inaccurate random seeking on mixed WLASL codecs."""
    if (len(indices) != expected_count or not indices or indices[0] < 0
            or indices != sorted(set(indices))):
        raise ValueError("Expected unique ascending frame indices")
    desired = set(indices)
    frames: dict[int, object] = {}
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open {path}")
        for original_index in range(indices[-1] + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"Decode stopped before frame {indices[-1]}; at {original_index}")
            if original_index in desired:
                frames[original_index] = frame
        if len(frames) != expected_count:
            raise ValueError("Not all selected frame indices decoded")
        return [(index, frames[index]) for index in indices]
    finally:
        capture.release()


def make_sheet(item: dict[str, object], indexed_frames: list[tuple[int, object]]) -> object:
    import numpy as np

    tile_width, tile_height = 320, 205
    header_height = 146
    sheet = np.full((header_height + 4 * tile_height, 4 * tile_width, 3), 242, dtype=np.uint8)
    lines = [
        f"WLASL {item['video_id']} | gloss={item['gloss']} | signer={item['signer_id']} | source={item['source']}",
        f"metadata frames (1-based @25fps): {item['metadata_frame_start']}..{item['metadata_frame_end']}",
        f"preview frames (local 0-based): {item['preview_start']}..{item['preview_end']} | {item['preview_mode']}",
        f"full video: {float(item['duration_seconds']):.2f}s @ {float(item['local_fps']):.3f} fps",
        f"REVIEW REQUIRED | {item['review_reason']}",
    ]
    for line_number, line in enumerate(lines):
        cv2.putText(sheet, line[:155], (14, 25 + line_number * 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.59, (25, 25, 25), 1, cv2.LINE_AA)
    for position, (original_index, frame) in enumerate(indexed_frames):
        row, column = divmod(position, 4)
        y0 = header_height + row * tile_height
        x0 = column * tile_width
        source_h, source_w = frame.shape[:2]
        ratio = min(tile_width / source_w, (tile_height - 25) / source_h)
        fitted = cv2.resize(frame, (max(1, round(source_w * ratio)), max(1, round(source_h * ratio))))
        offset_x = x0 + (tile_width - fitted.shape[1]) // 2
        offset_y = y0 + (tile_height - 25 - fitted.shape[0]) // 2
        sheet[offset_y:offset_y + fitted.shape[0], offset_x:offset_x + fitted.shape[1]] = fitted
        cv2.putText(sheet, f"{position + 1:02d} | original frame {original_index}",
                    (x0 + 8, y0 + tile_height - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.rectangle(sheet, (x0, y0), (x0 + tile_width - 1, y0 + tile_height - 1), (170, 170, 170), 1)
    return sheet


def write_preview(plan: list[dict[str, object]], batch_size: int) -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Review output already exists; refusing to overwrite: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    manifest_rows: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    for position, item in enumerate(plan):
        batch = f"batch_{position // batch_size + 1:03d}"
        batch_dir = OUTPUT / batch
        batch_dir.mkdir(exist_ok=True)
        sheet_path = batch_dir / f"WLASL_{item['video_id']}_contact_sheet.jpg"
        reason = str(item["review_reason"])
        relative_sheet = ""
        try:
            frames = selected_frames(item["path"], item["indices"])  # type: ignore[arg-type]
            sheet = make_sheet(item, frames)
            if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise OSError("cv2.imwrite failed")
            relative_sheet = sheet_path.relative_to(ROOT).as_posix()
        except (ValueError, OSError, cv2.error) as exc:
            reason += f";PREVIEW_RENDER_FAILED:{type(exc).__name__}:{exc}"
        manifest_rows.append({
            "video_id": item["video_id"], "gloss": item["gloss"],
            "signer_id": item["signer_id"], "source": item["source"],
            "local_path": item["local_path"],
            "metadata_frame_start": item["metadata_frame_start"],
            "metadata_frame_end": item["metadata_frame_end"],
            "metadata_fps_basis": METADATA_FPS,
            "preview_start": item["preview_start"], "preview_end": item["preview_end"],
            "selected_original_frame_indices": json.dumps(item["indices"]),
            "preview_mode": item["preview_mode"],
            "range_review_required": item["range_review_required"],
            "local_fps": item["local_fps"],
            "duration_seconds": item["duration_seconds"],
            "contact_sheet_path": relative_sheet, "batch": batch,
            "review_status": "REVIEW_REQUIRED", "review_reason": reason,
        })
        index_rows.append({"batch": batch, "video_id": item["video_id"],
                           "contact_sheet_path": relative_sheet,
                           "review_reason": reason})
        if (position + 1) % batch_size == 0 or position + 1 == len(plan):
            print(f"Rendered {position + 1}/{len(plan)}")
    with REVIEW_MANIFEST.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    with BATCH_INDEX.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("batch", "video_id", "contact_sheet_path", "review_reason"))
        writer.writeheader()
        writer.writerows(index_rows)
    with REVIEW_GUIDE.open("x", encoding="utf-8") as stream:
        json.dump({
            "allowed_review_statuses": REVIEW_OPTIONS,
            "initial_review_status": "REVIEW_REQUIRED",
            "range_review_flag": "range_review_required / review_reason; no automatic acceptance",
            "metadata_frame_indices": "1-based at 25 FPS, end inclusive; -1 means unknown end",
            "preview_frame_indices": "0-based indices of downloaded local video, end inclusive",
            "long_full_clip_mode": "FULL_CLIP_NAVIGATION_ONLY is not a gesture segment",
            "preview_only": True,
        }, stream, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Validate paths/hashes/ranges without creating files")
    mode.add_argument("--preview", action="store_true", help="Create contact sheets after confirmation")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Videos per batch (20-30; default 25)")
    args = parser.parse_args()
    if not 20 <= args.batch_size <= 30:
        parser.error("--batch-size must be between 20 and 30")
    plan = load_plan()
    counts = Counter(str(item["preview_mode"]) for item in plan)
    summary = {
        "videos": len(plan), "batch_size": args.batch_size,
        "batch_count": math.ceil(len(plan) / args.batch_size),
        "preview_modes": dict(counts),
        "range_review_required": sum(bool(item["range_review_required"]) for item in plan),
        "initial_review_status": "REVIEW_REQUIRED",
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "output_already_exists": OUTPUT.exists(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.check_only:
        return
    if OUTPUT.exists():
        raise FileExistsError(f"Review output already exists; refusing to overwrite: {OUTPUT}")
    if input(f"Create {len(plan)} contact sheets in {summary['batch_count']} batches? [y/N] ").strip().lower() != "y":
        print("Cancelled; no preview output created.")
        return
    write_preview(plan, args.batch_size)


if __name__ == "__main__":
    main()
