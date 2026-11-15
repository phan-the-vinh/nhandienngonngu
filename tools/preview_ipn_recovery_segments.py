"""User-run, report-driven video previews for three IPN Hand REVIEW segments.

--check-only validates existing audit/diagnostics and source-file metadata; it
does not decode video or write files. --execute writes the *entire* annotated
interval at source FPS, without MediaPipe, frame interpolation, mirroring,
cropping, landmark extraction, or automatic review decisions. A footer holds
diagnostic text so the original frame region is not covered by annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import preview_ipn_pilot_pass_segments as ipn_preview  # noqa: E402

EXPERIMENT = ROOT / "data/experiments/dynamic_none_v1"
AUDIT_CSV = EXPERIMENT / "metadata/ipn_pilot_accept_mediapipe_audit_v1.csv"
DIAGNOSTICS_JSONL = EXPERIMENT / "metadata/ipn_pilot_accept_mediapipe_frame_diagnostics_v1.jsonl"
SUMMARY_JSON = EXPERIMENT / "metadata/ipn_pilot_accept_mediapipe_summary_v1.json"
OUTPUT_DIR = EXPERIMENT / "review/ipn_recovery_preview_v1"
OUTPUT_METADATA = OUTPUT_DIR / "preview_metadata_v1.json"
FOOTER_HEIGHT = 96
TARGETS = (
    ("4CM11_23_R_#25", "G02", 2684, 2729),
    ("4CM11_2_R_#37", "G02", 3167, 3237),
    ("4CM11_5_L_#10", "G03", 3325, 3386),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Malformed CSV header: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV rows: {path}")
    return rows


def inclusive_ranges(indices: list[int]) -> list[dict[str, int]]:
    if not indices:
        return []
    result = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            result.append({"start_0based": start, "end_0based": previous,
                           "frame_count": previous - start + 1})
            start = index
        previous = index
    result.append({"start_0based": start, "end_0based": previous,
                   "frame_count": previous - start + 1})
    return result


def diagnostic_plan(frames: list[dict]) -> dict[str, object]:
    """Find candidate runs, never bridge a missed/multi-hand/changed frame."""
    missing = [f["original_frame_index"] for f in frames if f["raw_hand_count"] == 0]
    multiple = [f["original_frame_index"] for f in frames if f["raw_hand_count"] > 1]
    singles = [f for f in frames if f["raw_hand_count"] == f["valid_hand_count"] == 1]
    changes = []
    for left, right in zip(singles, singles[1:]):
        before, after = left["handedness_observed"][0], right["handedness_observed"][0]
        if before != after:
            changes.append({"previous_frame_0based": left["original_frame_index"],
                            "frame_0based": right["original_frame_index"],
                            "from": before, "to": after})

    runs = []
    current: list[dict] = []
    for frame in frames:
        unique = frame["raw_hand_count"] == frame["valid_hand_count"] == 1
        label = frame["handedness_observed"][0] if unique else ""
        if not unique or label not in ("Left", "Right"):
            if current:
                runs.append(current)
            current = []
            continue
        if current and (frame["original_frame_index"] != current[-1]["original_frame_index"] + 1
                        or label != current[-1]["handedness_observed"][0]):
            runs.append(current)
            current = []
        current.append(frame)
    if current:
        runs.append(current)
    candidates = [{"start_0based": run[0]["original_frame_index"],
                   "end_0based": run[-1]["original_frame_index"],
                   "frame_count": len(run), "observed_handedness": run[0]["handedness_observed"][0],
                   "status": "POTENTIAL_ONLY_NOT_TRAINING_APPROVAL"}
                  for run in runs if len(run) >= 16]
    return {"missing_hand_ranges": inclusive_ranges(missing),
            "multi_hand_ranges": inclusive_ranges(multiple),
            "handedness_changes": changes, "candidate_contiguous_ranges": candidates}


def preflight() -> list[dict[str, object]]:
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8-sig"))
    if (summary.get("segments") != 7 or summary.get("technical_status_counts") != {"PASS": 1, "REVIEW": 6}
            or summary.get("visual_review_status") != "AI_VISUAL_REVIEW_ONLY"
            or summary.get("training_approved") is not False
            or summary.get("training_dataset_created") is not False):
        raise ValueError("MediaPipe audit summary contract changed")
    audit_rows = read_csv(AUDIT_CSV)
    if len(audit_rows) != 7 or Counter(row["technical_status"] for row in audit_rows) != {"PASS": 1, "REVIEW": 6}:
        raise ValueError("Expected seven audit rows with one PASS and six REVIEW")
    audit_by_key = {}
    for row in audit_rows:
        key = (row["video_id"], row["annotation_label"], int(row["segment_start_1based"]),
               int(row["segment_end_1based"]))
        if key in audit_by_key:
            raise ValueError(f"Duplicate audit segment: {key}")
        audit_by_key[key] = row

    frames_by_key = {key: [] for key in TARGETS}
    with DIAGNOSTICS_JSONL.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("Blank diagnostics JSONL line")
            frame = json.loads(line)
            key = (frame["video_id"], frame["annotation_label"])
            for target in TARGETS:
                if key == target[:2]:
                    frames_by_key[target].append(frame)
                    break

    pass_plan = ipn_preview.load_plan()  # Confirms source path, size and decode-PASS annotations.
    pass_by_key = {(p["video_id"], p["label"], p["start_1"], p["end_1"]): p
                   for p in pass_plan}
    video_audit = {row["video_id"]: row for row in read_csv(ipn_preview.VIDEO_AUDIT)}
    plan = []
    for key in TARGETS:
        row, source = audit_by_key.get(key), pass_by_key.get(key)
        if row is None or source is None or row["technical_status"] != "REVIEW":
            raise ValueError(f"Target is not a decode-PASS/MediaPipe-REVIEW segment: {key}")
        if (row["visual_review_status"] != "AI_VISUAL_REVIEW_ONLY"
                or row["training_approved"] != "False"
                or (ROOT / row["source_video_path"]).resolve() != source["path"]):
            raise ValueError(f"Visual-review status or source path changed: {key}")
        frames = frames_by_key[key]
        expected = list(range(source["start_0"], source["end_0"] + 1))
        if ([f["original_frame_index"] for f in frames] != expected
                or len(frames) != int(row["decoded_segment_frames"])):
            raise ValueError(f"Diagnostics do not cover every annotated frame: {key}")
        if any(f["valid_hand_count"] > f["raw_hand_count"] or f["invalid_hands"]
               or f["detection_error"] for f in frames):
            raise ValueError(f"Unexpected invalid diagnostic frame: {key}")
        if (sum(f["raw_hand_count"] == 0 for f in frames) != int(row["missing_hand_frames"])
                or sum(f["raw_hand_count"] > 1 for f in frames) != int(row["multi_hand_frames"])
                or sum(f["valid_hand_count"] > 0 for f in frames) != int(row["frames_with_valid_21_landmarks"])):
            raise ValueError(f"Diagnostics summary differs from audit CSV: {key}")
        details = diagnostic_plan(frames)
        if len(details["handedness_changes"]) != int(row["single_hand_handedness_changes"]):
            raise ValueError(f"Handedness changes differ from audit CSV: {key}")
        video = video_audit[key[0]]
        fps = float(video["fps"])
        width, height = int(video["width"]), int(video["height"])
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Invalid audited video properties: {key[0]}")
        output = OUTPUT_DIR / f"{key[0]}_{key[1]}_full_annotation_preview.avi"
        plan.append({**source, "audit": row, "frames": frames, "details": details,
                     "fps": fps, "width": width, "height": height, "output": output})
    return plan


def render_frame(frame, index: int, details: dict[str, object], diagnostic: dict):
    """Put markers in a new footer; leave source-frame pixels untouched."""
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    canvas = np.zeros((height + FOOTER_HEIGHT, width, 3), dtype=np.uint8)
    canvas[:height] = frame
    candidate = next(((n, item) for n, item in enumerate(details["candidate_contiguous_ranges"], 1)
                      if item["start_0based"] <= index <= item["end_0based"]), None)
    change = next((item for item in details["handedness_changes"]
                   if item["frame_0based"] == index), None)
    flags = []
    if diagnostic["raw_hand_count"] == 0:
        flags.append("NO_HAND")
    if diagnostic["raw_hand_count"] > 1:
        flags.append("MULTI_HAND")
    if change:
        flags.append(f"HAND_CHANGE {change['from']}->{change['to']}")
    if candidate:
        number, interval = candidate
        flags.append(f"POTENTIAL {number}: {interval['start_0based']}-{interval['end_0based']}")
    if not flags:
        flags.append("TRACKED; NOT APPROVED")
    cv2.putText(canvas, f"Original frame (0-based): {index}", (10, height + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, " | ".join(flags), (10, height + 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54,
                (0, 90, 255) if change or diagnostic["raw_hand_count"] == 0 else (90, 230, 90),
                1, cv2.LINE_AA)
    candidate_summary = ", ".join(f"{r['start_0based']}-{r['end_0based']}"
                                  for r in details["candidate_contiguous_ranges"])
    cv2.putText(canvas, f"Potential ranges (0-based): {candidate_summary} | REVIEW ONLY",
                (10, height + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    return canvas


def write_preview(item: dict[str, object]) -> int:
    import cv2

    path, output = item["path"], item["output"]
    start, end = int(item["start_0"]), int(item["end_0"])
    cap = cv2.VideoCapture(str(path))  # Explicit file path; never a webcam index.
    writer = None
    written = 0
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open source video: {path}")
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(actual_fps) or abs(actual_fps - item["fps"]) > 0.05:
            raise ValueError(f"Source FPS differs from decode audit: {path}")
        if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) != item["width"]
                or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) != item["height"]):
            raise ValueError(f"Source resolution differs from decode audit: {path}")
        # Reserve the new path exclusively: an existing preview is never overwritten.
        descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(descriptor)
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"FFV1"),
                                 actual_fps, (item["width"], item["height"] + FOOTER_HEIGHT))
        if not writer.isOpened():
            raise ValueError("Lossless FFV1 AVI writer unavailable; output remains for manual inspection")
        for index in range(end + 1):
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                raise ValueError(f"Decode stopped at original frame {index}: {path}")
            if index < start:
                continue
            if frame.shape[:2] != (item["height"], item["width"]):
                raise ValueError(f"Frame dimensions changed at {index}: {path}")
            diagnostic = item["frames"][written]
            if diagnostic["original_frame_index"] != index:
                raise ValueError(f"Diagnostic frame index mismatch at {index}: {path}")
            writer.write(render_frame(frame, index, item["details"], diagnostic))
            written += 1
    finally:
        if writer is not None:
            writer.release()
        cap.release()
    if written != end - start + 1:
        raise ValueError(f"Preview frame count mismatch: {path}")
    verification = cv2.VideoCapture(str(output))
    decoded_output = 0
    try:
        if not verification.isOpened():
            raise ValueError(f"Cannot reopen written preview: {output}")
        output_fps = float(verification.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(output_fps) or abs(output_fps - item["fps"]) > 0.05:
            raise ValueError(f"Written preview FPS differs from source: {output}")
        while True:
            ok, frame = verification.read()
            if not ok:
                break
            if frame is None or frame.shape[:2] != (item["height"] + FOOTER_HEIGHT, item["width"]):
                raise ValueError(f"Written preview has an invalid frame: {output}")
            decoded_output += 1
    finally:
        verification.release()
    if decoded_output != written:
        raise ValueError(f"Written preview decoded {decoded_output}/{written} frames: {output}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Validate reports and paths; no video opened")
    mode.add_argument("--execute", action="store_true", help="Confirm, then write three full-segment video previews")
    args = parser.parse_args(argv)
    try:
        plan = preflight()
        view = [{"video_id": p["video_id"], "label": p["label"],
                 "annotation_1based": [p["start_1"], p["end_1"]],
                 "original_frame_0based": [p["start_0"], p["end_0"]],
                 "frame_count": len(p["frames"]), "fps": p["fps"],
                 "candidate_ranges": p["details"]["candidate_contiguous_ranges"],
                 "output": p["output"].relative_to(ROOT).as_posix()}
                for p in plan]
        print(json.dumps({"status": "PREFLIGHT_PASS", "segments": view,
                          "metadata": OUTPUT_METADATA.relative_to(ROOT).as_posix(),
                          "training_approved": False}))
        if args.check_only:
            return 0
        if any(p.exists() for p in [OUTPUT_METADATA, *(item["output"] for item in plan)]):
            raise ValueError("Preview output exists; refusing to overwrite")
        if input("Create three full-annotation IPN review videos? [y/N] ").strip().lower() != "y":
            print("Cancelled; no video opened")
            return 0
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        metadata = []
        for item in plan:
            written = write_preview(item)
            metadata.append({"video_id": item["video_id"], "label": item["label"],
                             "source_video_path": item["path"].relative_to(ROOT).as_posix(),
                             "output_video_path": item["output"].relative_to(ROOT).as_posix(),
                             "annotation_1based": [item["start_1"], item["end_1"]],
                             "original_frame_0based": [item["start_0"], item["end_0"]],
                             "source_fps": item["fps"], "output_fps": item["fps"],
                             "source_resolution": [item["width"], item["height"]],
                             "output_resolution": [item["width"], item["height"] + FOOTER_HEIGHT],
                             "frame_count": written, "diagnostics": item["details"],
                             "technical_status": "REVIEW", "visual_review_status": "AI_VISUAL_REVIEW_ONLY",
                             "training_approved": False})
        with OUTPUT_METADATA.open("x", encoding="utf-8") as stream:
            json.dump({"purpose": "IPN_RECOVERY_FULL_ANNOTATION_PREVIEW_ONLY",
                       "codec": "FFV1 lossless AVI", "frame_policy": "all annotation frames in source order at source FPS",
                       "original_image_region": "copied without flip/crop/resize; diagnostics in added footer",
                       "mediapipe_rerun": False, "sequence_created": False,
                       "training_approved": False, "segments": metadata},
                      stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
        print(json.dumps({"status": "PREVIEW_COMPLETE", "video_count": len(metadata),
                          "metadata": OUTPUT_METADATA.relative_to(ROOT).as_posix()}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
