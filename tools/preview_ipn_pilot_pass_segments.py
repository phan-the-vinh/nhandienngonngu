"""User-run contact sheets for exactly 18 decode-PASS IPN pilot segments.

--check-only reads prior CSV reports and file metadata; it opens no video and
writes nothing. --execute previews only PASS segments, with no MediaPipe,
landmark extraction, dataset creation, or automatic review decision.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.feature_extraction.dynamic_sequence import sample_interval_indices


EXPERIMENT = ROOT / "data/experiments/dynamic_none_v1"
VIDEO_ROOT = EXPERIMENT / "raw/ipn_hand/pilot_videos/videos"
VIDEO_AUDIT = EXPERIMENT / "metadata/ipn_pilot_video_decode_audit_v1.csv"
SEGMENT_AUDIT = EXPERIMENT / "metadata/ipn_pilot_segment_alignment_audit_v1.csv"
OUTPUT = EXPERIMENT / "review/ipn_pilot_pass_v1"
REVIEW_TABLE = OUTPUT / "manual_preview_review_v1.csv"
EXPECTED_PASS_VIDEOS = {
    "4CM11_1_R_#13", "4CM11_2_R_#37", "4CM11_5_L_#10",
    "4CM11_7_R_#33", "4CM11_23_R_#25", "4CM11_26_R_#5",
}
REVIEW_FIELDS = (
    "video_id", "label", "segment_start_1based", "segment_end_1based",
    "preview_status", "manual_decision", "subtype_candidate", "jz_risk", "note",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_plan() -> list[dict[str, object]]:
    videos = read_csv(VIDEO_AUDIT)
    segments = read_csv(SEGMENT_AUDIT)
    if len(videos) != 8 or len(segments) != 24:
        raise ValueError("Expected eight audited videos and 24 audited segments")
    if Counter(row["video_status"] for row in videos) != {"PASS": 6, "REVIEW": 2}:
        raise ValueError("Video audit status counts changed")
    if Counter(row["segment_status"] for row in segments) != {"PASS": 18, "REVIEW": 5, "FAIL": 1}:
        raise ValueError("Segment audit status counts changed")
    by_video = {row["video_id"]: row for row in videos}
    if len(by_video) != 8:
        raise ValueError("Duplicate video ID in video audit")
    if {row["video_id"] for row in videos if row["video_status"] == "PASS"} != EXPECTED_PASS_VIDEOS:
        raise ValueError("Unexpected set of PASS videos")
    keys = [(row["video_id"], row["annotation_label"],
             row["annotation_start_1based"], row["annotation_end_1based"]) for row in segments]
    if len(set(keys)) != 24:
        raise ValueError("Duplicate segment in segment audit")

    plan: list[dict[str, object]] = []
    for row in segments:
        if row["segment_status"] != "PASS":
            continue  # REVIEW and FAIL are never opened by this tool.
        video_id = row["video_id"]
        video = by_video.get(video_id)
        if video is None or video["video_status"] != "PASS":
            raise ValueError(f"PASS segment is not from a PASS video: {video_id}")
        expected_path = (VIDEO_ROOT / f"{video_id}.avi").resolve()
        reported_path = (ROOT / video["video_path"]).resolve()
        if (reported_path != expected_path or expected_path.parent != VIDEO_ROOT.resolve()
                or not expected_path.is_file()
                or expected_path.stat().st_size != int(video["file_size_bytes"])):
            raise ValueError(f"Video path/size differs from audit: {video_id}")
        start_1 = int(row["annotation_start_1based"])
        end_1 = int(row["annotation_end_1based"])
        start_0 = int(row["opencv_start_0based"])
        end_0 = int(row["opencv_end_0based"])
        decoded = int(video["decoded_frames"])
        if (start_1 < 1 or end_1 < start_1 or start_0 != start_1 - 1
                or end_0 != end_1 - 1 or end_1 - start_1 + 1 != int(row["annotation_frames"])
                or not 0 <= start_0 < end_0 < decoded
                or int(row["decoded_frames"]) != decoded):
            raise ValueError(f"Invalid PASS segment frame alignment: {video_id}/{row['annotation_label']}")
        indices = sample_interval_indices(start_0, end_0)
        if len(indices) != 16 or indices[0] != start_0 or indices[-1] != end_0:
            raise ValueError("Preview frame sampling did not cover the segment")
        plan.append({"video_id": video_id, "label": row["annotation_label"],
                     "start_1": start_1, "end_1": end_1, "start_0": start_0,
                     "end_0": end_0, "indices": indices, "path": expected_path})
    if len(plan) != 18:
        raise ValueError("Expected exactly 18 PASS segments")
    return sorted(plan, key=lambda item: (str(item["video_id"]), int(item["start_0"])))


def make_sheet(item: dict[str, object], indexed_frames: list[tuple[int, object]]):
    import cv2
    import numpy as np

    if len(indexed_frames) != 16 or [index for index, _ in indexed_frames] != item["indices"]:
        raise ValueError("Contact sheet frames do not match the planned 16 indices")
    tile_w, tile_h, header_h = 320, 205, 92
    canvas = np.full((header_h + 4 * tile_h, 4 * tile_w, 3), 242, dtype=np.uint8)
    lines = (
        f"IPN {item['video_id']} | annotation {item['label']} | PREVIEW ONLY",
        f"1-based segment {item['start_1']}..{item['end_1']} | 0-based {item['start_0']}..{item['end_0']}",
        "16 ordered frames; no MediaPipe; NONE subtype and J/Z risk require human review",
    )
    for line_no, line in enumerate(lines):
        cv2.putText(canvas, line, (10, 23 + line_no * 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (25, 25, 25), 1, cv2.LINE_AA)
    for position, (frame_index, frame) in enumerate(indexed_frames):
        grid_row, grid_col = divmod(position, 4)
        x0, y0 = grid_col * tile_w, header_h + grid_row * tile_h
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("Invalid preview frame dimensions")
        scale = min(tile_w / width, (tile_h - 27) / height)
        fitted = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        x = x0 + (tile_w - fitted.shape[1]) // 2
        y = y0 + (tile_h - 27 - fitted.shape[0]) // 2
        canvas[y:y + fitted.shape[0], x:x + fitted.shape[1]] = fitted
        cv2.putText(canvas, f"{position + 1:02d} | source frame {frame_index} (0-based)",
                    (x0 + 8, y0 + tile_h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.47, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y0), (x0 + tile_w - 1, y0 + tile_h - 1), (170, 170, 170), 1)
    return canvas


def execute(plan: list[dict[str, object]]) -> None:
    import cv2
    from tools.preview_wlasl_other_gesture import selected_frames

    if OUTPUT.exists():
        raise FileExistsError(f"Preview output already exists; refusing to overwrite: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for item in plan:
        video_id, label = str(item["video_id"]), str(item["label"])
        filename = f"{video_id}_{label}_{item['start_1']}_{item['end_1']}_contact_sheet.jpg"
        sheet_path = OUTPUT / filename
        preview_status, note = "READY_FOR_MANUAL_REVIEW", ""
        try:
            frames = selected_frames(item["path"], item["indices"])
            sheet = make_sheet(item, frames)
            if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise OSError("cv2.imwrite failed")
        except (OSError, ValueError, cv2.error) as exc:
            preview_status = "PREVIEW_FAILED"
            note = f"{type(exc).__name__}: {exc}"
        rows.append({"video_id": video_id, "label": label,
                     "segment_start_1based": item["start_1"],
                     "segment_end_1based": item["end_1"],
                     "preview_status": preview_status,
                     "manual_decision": "PENDING_USER_REVIEW",
                     "subtype_candidate": "UNKNOWN", "jz_risk": "UNKNOWN", "note": note})
    with REVIEW_TABLE.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(str(row["preview_status"]) for row in rows)
    print(f"SEGMENTS=18 PREVIEW_READY={counts['READY_FOR_MANUAL_REVIEW']} "
          f"PREVIEW_FAILED={counts['PREVIEW_FAILED']} REVIEW_TABLE={REVIEW_TABLE}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Validate 18 PASS segments without opening video")
    mode.add_argument("--execute", action="store_true", help="Create 18 contact sheets and manual review CSV")
    args = parser.parse_args()
    try:
        plan = load_plan()
        if args.check_only:
            print(f"CHECK_ONLY=PASS VIDEOS={len({item['video_id'] for item in plan})} "
                  f"PASS_SEGMENTS={len(plan)} EXCLUDED_SEGMENTS=6 "
                  f"OUTPUT_EXISTS={OUTPUT.exists()} VIDEOS_OPENED=0 FILES_WRITTEN=0")
            return 0
        if OUTPUT.exists():
            raise FileExistsError(f"Preview output already exists; refusing to overwrite: {OUTPUT}")
        if input("Create 18 preview sheets? [y/N] ").strip().lower() != "y":
            print("CANCELLED: no video opened or file written")
            return 0
        execute(plan)
        return 0
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
