"""Technical OpenCV audit for the eight IPN Hand NONE pilot videos.

--check-only validates filenames, train membership, metadata and annotations
without opening a video. --audit must be run by the user; it decodes all frames
and creates temporary reports, but never runs MediaPipe or extracts features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IPN = ROOT / "data/experiments/dynamic_none_v1/raw/ipn_hand"
VIDEOS = IPN / "pilot_videos/videos"
ANNOTATIONS = IPN / "annotations"
REPORT_DIR = ROOT / "data/experiments/dynamic_none_v1/metadata"
VIDEO_REPORT = REPORT_DIR / "ipn_pilot_video_decode_audit_v1.csv"
SEGMENT_REPORT = REPORT_DIR / "ipn_pilot_segment_alignment_audit_v1.csv"
SUMMARY_REPORT = REPORT_DIR / "ipn_pilot_decode_summary_v1.json"

# Exact 24 previously proposed annotation segments; no labels are inferred.
PILOT = {
    "4CM11_1_R_#13": (("D0X", 1544, 1818), ("B0A", 3019, 3260), ("G01", 1819, 1870)),
    "4CM11_2_R_#37": (("D0X", 2599, 2844), ("B0B", 2285, 2598), ("G02", 3167, 3237)),
    "4CM11_5_L_#10": (("D0X", 2762, 2941), ("B0A", 445, 765), ("G03", 3325, 3386)),
    "4CM11_7_R_#33": (("D0X", 2043, 2238), ("B0B", 3346, 3626), ("G07", 3231, 3345)),
    "4CM11_19_R_#17": (("D0X", 2515, 3012), ("G08", 17, 62), ("G09", 3672, 3785)),
    "4CM11_23_R_#25": (("D0X", 1967, 2406), ("G01", 3648, 3719), ("G02", 2684, 2729)),
    "4CM11_26_R_#5": (("D0X", 1935, 2199), ("G03", 531, 641), ("G07", 800, 904)),
    "4CM11_29_R_#57": (("D0X", 1432, 1735), ("G08", 3933, 3975), ("G09", 1217, 1249)),
}

VIDEO_FIELDS = (
    "video_id", "video_path", "file_size_bytes", "special_large_video",
    "metadata_frames", "container_frames", "decoded_frames", "fps", "width", "height",
    "invalid_decoded_frames", "frame_position_jumps", "video_status", "reason",
)
SEGMENT_FIELDS = (
    "video_id", "annotation_label", "annotation_start_1based", "annotation_end_1based",
    "annotation_frames", "opencv_start_0based", "opencv_end_0based",
    "decoded_frames", "segment_status", "reason",
)


def read_csv(path: Path, fieldnames: list[str] | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, fieldnames=fieldnames))


def frame_range_0based(start: int, end: int, frames: int) -> tuple[int, int]:
    """IPN annotation bounds are 1-based inclusive; OpenCV uses 0-based."""
    if start < 1 or end < start or frames != end - start + 1:
        raise ValueError(f"Invalid 1-based inclusive annotation: {start}, {end}, {frames}")
    return start - 1, end - 1


def preflight() -> dict[str, object]:
    if not VIDEOS.is_dir():
        raise FileNotFoundError(VIDEOS)
    actual = sorted(path for path in VIDEOS.iterdir() if path.is_file() and path.suffix.lower() == ".avi")
    expected = {f"{video_id}.avi" for video_id in PILOT}
    names = {path.name for path in actual}
    if len(actual) != 8 or names != expected or len({name.casefold() for name in names}) != 8:
        raise ValueError(f"Expected exactly eight distinct pilot AVI files; missing={sorted(expected - names)}, "
                         f"unexpected={sorted(names - expected)}")
    if any(path.stat().st_size == 0 for path in actual):
        raise ValueError("One or more pilot AVI files are empty")

    metadata_rows = read_csv(ANNOTATIONS / "metadata.csv")
    metadata = {row["Video Name"]: row for row in metadata_rows}
    if len(metadata) != len(metadata_rows):
        raise ValueError("Duplicate video IDs in metadata.csv")
    train_ids = {line.split()[0] for line in (ANNOTATIONS / "Video_TrainList.txt")
                 .read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    all_rows = read_csv(ANNOTATIONS / "Annot_List.txt")
    train_rows = read_csv(ANNOTATIONS / "Annot_TrainList.txt",
                          ["video", "label", "id", "t_start", "t_end", "frames"])
    all_keys = Counter((row["video"], row["label"], row["t_start"], row["t_end"]) for row in all_rows)
    train_keys = Counter((row["video"], row["label"], row["t_start"], row["t_end"]) for row in train_rows)

    segments: list[dict[str, object]] = []
    for video_id, planned in PILOT.items():
        if video_id not in train_ids or video_id not in metadata or metadata[video_id]["Set"] != "train":
            raise ValueError(f"Pilot video is not in the annotated train split: {video_id}")
        metadata_frames = int(metadata[video_id]["Frames"])
        if metadata_frames <= 0:
            raise ValueError(f"Invalid metadata frame count: {video_id}")
        whole_video_rows = sorted((row for row in all_rows if row["video"] == video_id),
                                  key=lambda row: int(row["t_start"]))
        if (not whole_video_rows or int(whole_video_rows[0]["t_start"]) != 1
                or int(whole_video_rows[-1]["t_end"]) != metadata_frames):
            raise ValueError(f"Annotation does not span 1..metadata frame count: {video_id}")
        previous_end = 0
        for row in whole_video_rows:
            start, end, frames = int(row["t_start"]), int(row["t_end"]), int(row["frames"])
            frame_range_0based(start, end, frames)
            if start != previous_end + 1:
                raise ValueError(f"Annotation has a gap or overlap: {video_id}")
            previous_end = end
        for label, start, end in planned:
            key = (video_id, label, str(start), str(end))
            if all_keys[key] != 1 or train_keys[key] != 1:
                raise ValueError(f"Pilot segment missing/duplicated in annotations: {key}")
            matching = [row for row in all_rows if
                        (row["video"], row["label"], row["t_start"], row["t_end"]) == key]
            annotated_frames = int(matching[0]["frames"])
            matching_train = [row for row in train_rows if
                              (row["video"], row["label"], row["t_start"], row["t_end"]) == key]
            if (matching_train[0]["frames"] != matching[0]["frames"]
                    or matching_train[0]["id"] != matching[0]["id"]):
                raise ValueError(f"Train annotation differs from master annotation: {key}")
            start_zero, end_zero = frame_range_0based(start, end, annotated_frames)
            if end > metadata_frames:
                raise ValueError(f"Pilot segment exceeds metadata frame count: {key}")
            segments.append({"video_id": video_id, "label": label, "start": start, "end": end,
                             "frames": annotated_frames, "start_zero": start_zero, "end_zero": end_zero})
    if len(segments) != 24:
        raise ValueError("Expected exactly 24 pilot segments")
    return {"metadata": metadata, "segments": segments, "videos": actual}


def decode_video(path: Path, metadata_frames: int) -> dict[str, object]:
    """Read sequentially to EOF; a false read alone cannot prove corruption vs EOF."""
    import cv2  # Not imported, and no video opened, in --check-only mode.

    row: dict[str, object] = {
        "video_id": path.stem, "video_path": path.relative_to(ROOT).as_posix(),
        "file_size_bytes": path.stat().st_size,
        "special_large_video": path.stem == "4CM11_5_L_#10",
        "metadata_frames": metadata_frames, "container_frames": "", "decoded_frames": 0,
        "fps": "", "width": "", "height": "", "invalid_decoded_frames": 0,
        "frame_position_jumps": 0, "video_status": "FAIL", "reason": "",
    }
    capture = cv2.VideoCapture(str(path))  # A file path, never a webcam index.
    try:
        if not capture.isOpened():
            row["reason"] = "OPENCV_OPEN_FAILED"
            return row
        reported = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if math.isfinite(reported) and reported > 0:
            row["container_frames"] = int(round(reported))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if math.isfinite(fps) and fps > 0:
            row["fps"] = round(float(fps), 6)
        previous_position = 0.0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0 or frame.ndim < 2:
                row["invalid_decoded_frames"] = int(row["invalid_decoded_frames"]) + 1
                row["reason"] = "INVALID_DECODED_FRAME"
                break
            height, width = frame.shape[:2]
            if width <= 0 or height <= 0:
                row["invalid_decoded_frames"] = int(row["invalid_decoded_frames"]) + 1
                row["reason"] = "INVALID_FRAME_DIMENSIONS"
                break
            if not row["width"]:
                row["width"], row["height"] = width, height
            elif row["width"] != width or row["height"] != height:
                row["invalid_decoded_frames"] = int(row["invalid_decoded_frames"]) + 1
                row["reason"] = "INCONSISTENT_FRAME_DIMENSIONS"
                break
            row["decoded_frames"] = int(row["decoded_frames"]) + 1
            position = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if (math.isfinite(position) and position > 0 and previous_position > 0
                    and position - previous_position > 1.5):
                row["frame_position_jumps"] = int(row["frame_position_jumps"]) + 1
            if math.isfinite(position) and position > 0:
                previous_position = position
    except Exception as exc:
        row["reason"] = f"DECODE_EXCEPTION:{type(exc).__name__}:{exc}"
    finally:
        capture.release()

    if row["reason"] or int(row["decoded_frames"]) == 0:
        row["video_status"] = "FAIL"
        row["reason"] = row["reason"] or "NO_FRAMES_DECODED"
    elif int(row["decoded_frames"]) != metadata_frames:
        row["video_status"] = "REVIEW"
        row["reason"] = "DECODED_COUNT_DIFFERS_FROM_METADATA_OR_EARLY_STOP"
    elif row["container_frames"] not in ("", row["decoded_frames"]):
        row["video_status"] = "REVIEW"
        row["reason"] = "CONTAINER_COUNT_DIFFERS_FROM_DECODED"
    elif row["frame_position_jumps"]:
        row["video_status"] = "REVIEW"
        row["reason"] = "FRAME_POSITION_JUMP"
    elif not row["fps"]:
        row["video_status"] = "REVIEW"
        row["reason"] = "INVALID_FPS"
    else:
        row["video_status"] = "PASS"
        row["reason"] = "COUNTS_AND_DECODE_CONSISTENT"
    return row


def segment_result(segment: dict[str, object], video: dict[str, object]) -> dict[str, object]:
    decoded = int(video["decoded_frames"])
    end_zero = int(segment["end_zero"])
    if video["video_status"] == "FAIL" or end_zero >= decoded:
        status = "FAIL"
        reason = "VIDEO_DECODE_FAILED" if video["video_status"] == "FAIL" else "SEGMENT_OUTSIDE_DECODED_FRAMES"
    elif video["video_status"] == "REVIEW":
        status, reason = "REVIEW", "VIDEO_COUNT_OR_FRAME_POSITION_REVIEW"
    else:
        status, reason = "PASS", "ONE_BASED_TO_ZERO_BASED_RANGE_IN_BOUNDS"
    return {
        "video_id": segment["video_id"], "annotation_label": segment["label"],
        "annotation_start_1based": segment["start"], "annotation_end_1based": segment["end"],
        "annotation_frames": segment["frames"], "opencv_start_0based": segment["start_zero"],
        "opencv_end_0based": end_zero, "decoded_frames": decoded,
        "segment_status": status, "reason": reason,
    }


def write_csv_new(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Validate inputs without opening video or writing files")
    mode.add_argument("--audit", action="store_true", help="Decode eight AVI files and write temporary audit reports")
    args = parser.parse_args()
    try:
        checked = preflight()
        if args.check_only:
            print("PREFLIGHT=PASS VIDEOS=8 TRAIN_VIDEOS=8 ANNOTATION_SEGMENTS=24 "
                  "VIDEO_DECODED=0 REPORTS_WRITTEN=0")
            return 0
        outputs = (VIDEO_REPORT, SEGMENT_REPORT, SUMMARY_REPORT)
        if any(path.exists() for path in outputs):
            raise FileExistsError("Audit report already exists; refusing to overwrite")
        metadata = checked["metadata"]
        videos = [decode_video(path, int(metadata[path.stem]["Frames"])) for path in checked["videos"]]
        by_id = {row["video_id"]: row for row in videos}
        segments = [segment_result(segment, by_id[segment["video_id"]])
                    for segment in checked["segments"]]
        summary = {
            "status": "IPN_PILOT_DECODE_AUDIT_COMPLETE",
            "videos": len(videos), "segments": len(segments),
            "video_status_counts": dict(Counter(row["video_status"] for row in videos)),
            "segment_status_counts": dict(Counter(row["segment_status"] for row in segments)),
            "large_video_id": "4CM11_5_L_#10",
            "large_video_status": by_id["4CM11_5_L_#10"]["video_status"],
            "note": "PASS verifies OpenCV sequential decode counts and numeric annotation bounds, "
                    "not gesture semantics, hand landmarks, or every possible silent decoder error.",
        }
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        write_csv_new(VIDEO_REPORT, VIDEO_FIELDS, videos)
        write_csv_new(SEGMENT_REPORT, SEGMENT_FIELDS, segments)
        with SUMMARY_REPORT.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
