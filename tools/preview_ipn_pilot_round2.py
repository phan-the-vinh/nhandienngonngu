"""Preview exactly 18 new IPN Hand NONE candidates; never make review decisions.

--check-only validates metadata without decoding video or writing files.
--execute requires confirmation, then creates 16-frame contact sheets and a
pending manual-review CSV in a new experiment-only directory. No MediaPipe.
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

from tools import preview_ipn_pilot_pass_segments as pilot
from ai.feature_extraction.dynamic_sequence import sample_interval_indices


EXPERIMENT = pilot.EXPERIMENT
ANNOTATIONS = EXPERIMENT / "raw/ipn_hand/annotations"
OUTPUT = EXPERIMENT / "review/ipn_pilot_round2_v1"
REVIEW_TABLE = OUTPUT / "manual_preview_review_v1.csv"

# Annotation bounds are 1-based and inclusive. D0X comes first deliberately.
SEGMENTS = (
    ("4CM11_1_R_#13", "D0X", 2748, 2958),
    ("4CM11_2_R_#37", "D0X", 1405, 1615),
    ("4CM11_5_L_#10", "D0X", 2362, 2498),
    ("4CM11_23_R_#25", "D0X", 3229, 3437),
    ("4CM11_1_R_#13", "B0B", 998, 1234),
    ("4CM11_1_R_#13", "G08", 3471, 3569),
    ("4CM11_2_R_#37", "B0B", 1676, 1971),
    ("4CM11_2_R_#37", "G09", 731, 816),
    ("4CM11_5_L_#10", "B0B", 1222, 1527),
    ("4CM11_5_L_#10", "G07", 3633, 3714),
    ("4CM11_7_R_#33", "B0B", 655, 865),
    ("4CM11_7_R_#33", "G02", 1334, 1421),
    ("4CM11_7_R_#33", "G08", 304, 398),
    ("4CM11_23_R_#25", "B0B", 946, 1089),
    ("4CM11_23_R_#25", "G09", 1117, 1181),
    ("4CM11_26_R_#5", "B0B", 3077, 3288),
    ("4CM11_26_R_#5", "G02", 3664, 3740),
    ("4CM11_26_R_#5", "G08", 1650, 1719),
)


def _annotation_rows(path: Path, *, header: bool) -> list[tuple[str, str, int, int, int]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        if header and next(reader) != ["video", "label", "id", "t_start", "t_end", "frames"]:
            raise ValueError(f"Unexpected annotation schema: {path}")
        rows = []
        for row in reader:
            if len(row) != 6:
                raise ValueError(f"Invalid annotation row in {path}: {row}")
            rows.append((row[0], row[1], int(row[3]), int(row[4]), int(row[5])))
        return rows


def _train_video_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [line.strip().split() for line in stream if line.strip()]
    if any(len(row) != 2 for row in rows):
        raise ValueError("Invalid Video_TrainList schema")
    ids = [row[0] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate video ID in Video_TrainList")
    return set(ids)


def load_plan() -> list[dict[str, object]]:
    if len(SEGMENTS) != 18 or len(set(SEGMENTS)) != 18:
        raise ValueError("Expected 18 unique round-two candidates")
    if [row[1] for row in SEGMENTS[:4]] != ["D0X"] * 4 or any(
        row[1] == "D0X" for row in SEGMENTS[4:]
    ):
        raise ValueError("Four D0X candidates must be first")

    video_rows = pilot.read_csv(pilot.VIDEO_AUDIT)
    old_rows = pilot.read_csv(pilot.SEGMENT_AUDIT)
    if len(video_rows) != 8 or len(old_rows) != 24:
        raise ValueError("Prior pilot audit is incomplete")
    if Counter(row["video_status"] for row in video_rows) != {"PASS": 6, "REVIEW": 2}:
        raise ValueError("Unexpected video audit statuses")
    by_video = {row["video_id"]: row for row in video_rows}
    if len(by_video) != len(video_rows):
        raise ValueError("Duplicate video ID in decode audit")
    full_annotations = Counter(_annotation_rows(ANNOTATIONS / "Annot_List.txt", header=True))
    train_annotations = Counter(_annotation_rows(ANNOTATIONS / "Annot_TrainList.txt", header=False))
    train_videos = _train_video_ids(ANNOTATIONS / "Video_TrainList.txt")
    if len({(row["video_id"], row["annotation_label"],
             row["annotation_start_1based"], row["annotation_end_1based"])
            for row in old_rows}) != 24:
        raise ValueError("Duplicate segment in prior pilot")

    plan = []
    for video_id, label, start_1, end_1 in SEGMENTS:
        video = by_video.get(video_id)
        if video is None or video["video_status"] != "PASS" or video_id not in train_videos:
            raise ValueError(f"Not a decoded-PASS train video: {video_id}")
        length = end_1 - start_1 + 1
        annotation = (video_id, label, start_1, end_1, length)
        if full_annotations[annotation] != 1 or train_annotations[annotation] != 1:
            raise ValueError(f"Annotation/train split mismatch: {annotation}")
        decoded = int(video["decoded_frames"])
        if start_1 < 1 or end_1 > decoded or length < 16:
            raise ValueError(f"Segment outside decoded frames or too short: {annotation}")
        if any(row["video_id"] == video_id and not (
            end_1 < int(row["annotation_start_1based"])
            or start_1 > int(row["annotation_end_1based"])
        ) for row in old_rows):
            raise ValueError(f"Overlaps old pilot: {annotation}")
        if any(item["video_id"] == video_id and not (
            end_1 < item["start_1"] or start_1 > item["end_1"]
        ) for item in plan):
            raise ValueError(f"Overlaps another round-two segment: {annotation}")
        expected_path = (pilot.VIDEO_ROOT / f"{video_id}.avi").resolve()
        reported_path = (pilot.ROOT / video["video_path"]).resolve()
        if (expected_path.parent != pilot.VIDEO_ROOT.resolve()
                or expected_path != reported_path or not expected_path.is_file()
                or expected_path.stat().st_size != int(video["file_size_bytes"])):
            raise ValueError(f"Video path/size mismatch: {video_id}")
        indices = sample_interval_indices(start_1 - 1, end_1 - 1)
        if len(indices) != 16 or indices != sorted(set(indices)) or indices[0] != start_1 - 1 or indices[-1] != end_1 - 1:
            raise ValueError(f"Invalid preview indices: {annotation}")
        plan.append({"video_id": video_id, "label": label, "start_1": start_1,
                     "end_1": end_1, "start_0": start_1 - 1, "end_0": end_1 - 1,
                     "indices": indices, "path": expected_path})
    return plan


def review_row(item: dict[str, object], preview_status: str, note: str = "") -> dict[str, object]:
    return {"video_id": item["video_id"], "label": item["label"],
            "segment_start_1based": item["start_1"],
            "segment_end_1based": item["end_1"], "preview_status": preview_status,
            "manual_decision": "PENDING_USER_REVIEW", "subtype_candidate": "UNKNOWN",
            "jz_risk": "UNKNOWN", "note": note}


def execute(plan: list[dict[str, object]]) -> None:
    import cv2
    from tools.preview_wlasl_other_gesture import selected_frames

    if OUTPUT.exists():
        raise FileExistsError(f"Output exists; refusing to overwrite: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    rows = []
    for item in plan:
        name = f"{item['video_id']}_{item['label']}_{item['start_1']}_{item['end_1']}_contact_sheet.jpg"
        try:
            frames = selected_frames(item["path"], item["indices"])
            sheet = pilot.make_sheet(item, frames)
            if not cv2.imwrite(str(OUTPUT / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise OSError("cv2.imwrite failed")
            rows.append(review_row(item, "READY_FOR_MANUAL_REVIEW"))
        except (OSError, ValueError, cv2.error) as exc:
            rows.append(review_row(item, "PREVIEW_FAILED", f"{type(exc).__name__}: {exc}"))
    with REVIEW_TABLE.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=pilot.REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(str(row["preview_status"]) for row in rows)
    print(f"SEGMENTS=18 D0X_FIRST=4 READY={counts['READY_FOR_MANUAL_REVIEW']} "
          f"FAILED={counts['PREVIEW_FAILED']} REVIEW_TABLE={REVIEW_TABLE}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Metadata preflight; no video decode or writes")
    mode.add_argument("--execute", action="store_true", help="Create 18 sheets and pending review CSV")
    args = parser.parse_args()
    try:
        plan = load_plan()
        if args.check_only:
            print(f"CHECK_ONLY=PASS SEGMENTS={len(plan)} D0X={sum(item['label'] == 'D0X' for item in plan)} "
                  f"VIDEOS={len({item['video_id'] for item in plan})} OUTPUT_EXISTS={OUTPUT.exists()} "
                  "VIDEOS_OPENED=0 FILES_WRITTEN=0")
            return 0
        if OUTPUT.exists():
            raise FileExistsError(f"Output exists; refusing to overwrite: {OUTPUT}")
        if input("Create 18 round-two contact sheets? [y/N] ").strip().lower() != "y":
            print("CANCELLED: no video opened or file written")
            return 0
        execute(plan)
        return 0
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
