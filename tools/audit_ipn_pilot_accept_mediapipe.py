"""User-run MediaPipe diagnostic for seven visually proposed IPN NONE segments.

--check-only reads metadata and file stats, without opening video or MediaPipe.
--audit decodes only the reviewed segment of each provisional ACCEPT. It never
creates 16x69 features, a training dataset, or a training approval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.feature_extraction.dynamic_sequence import FEATURE_DIM, SEQUENCE_LENGTH  # noqa: E402
from tools import audit_wlasl_hand_identity_continuity as continuity  # noqa: E402
from tools import preview_ipn_pilot_pass_segments as preview  # noqa: E402

EXPERIMENT = ROOT / "data/experiments/dynamic_none_v1"
REVIEW_CSV = preview.REVIEW_TABLE
TASK = ROOT / "models/mediapipe/hand_landmarker.task"
REPORT = EXPERIMENT / "metadata/ipn_pilot_accept_mediapipe_audit_v1.csv"
FRAMES = EXPERIMENT / "metadata/ipn_pilot_accept_mediapipe_frame_diagnostics_v1.jsonl"
SUMMARY = EXPERIMENT / "metadata/ipn_pilot_accept_mediapipe_summary_v1.json"
REPORT_FIELDS = (
    "video_id", "annotation_label", "segment_start_1based", "segment_end_1based",
    "source_video_path", "visual_review_status", "visual_decision",
    "subtype_candidate", "jz_risk", "decoded_segment_frames",
    "frames_with_detected_hand", "hand_detection_ratio",
    "frames_with_valid_21_landmarks", "landmark_coverage_ratio",
    "single_hand_frames", "multi_hand_frames",
    "missing_hand_frames", "invalid_hand_frames", "detection_error_frames",
    "single_hand_handedness_changes", "internal_detection_gaps",
    "max_wrist_step_palm_widths", "max_shape_step_palm_widths",
    "has_16_valid_frames", "can_sample_same_hand_16",
    "proposed_original_frame_indices", "technical_status", "decision_reason",
    "identity_not_proven", "training_approved", "sequence_length", "feature_dim",
)


def segment_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["video_id"], row["label"], row["segment_start_1based"],
            row["segment_end_1based"])


def preflight() -> list[dict[str, object]]:
    with REVIEW_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(preview.REVIEW_FIELDS):
            raise ValueError("Review CSV must have the exact nine-column schema")
        rows = list(reader)
    if len(rows) != 18 or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Review CSV must contain exactly 18 well-formed rows")
    if len({segment_key(row) for row in rows}) != 18:
        raise ValueError("Duplicate review segment")
    if Counter(row["manual_decision"] for row in rows) != {"ACCEPT": 7, "REVIEW": 7, "REJECT": 4}:
        raise ValueError("Review decision distribution differs from 7/7/4")
    if any(row["preview_status"] != "READY_FOR_MANUAL_REVIEW"
           or "AI_VISUAL_REVIEW_ONLY" not in row["note"] for row in rows):
        raise ValueError("Review status or AI_VISUAL_REVIEW_ONLY marker changed")

    pass_plan = preview.load_plan()  # Validates all 24 decode-audit segments and eight videos.
    by_key = {(item["video_id"], item["label"], str(item["start_1"]),
               str(item["end_1"])): item for item in pass_plan}
    if set(map(segment_key, rows)) != set(by_key):
        raise ValueError("Review rows must exactly match the 18 decode-PASS segments")
    selected = []
    for row in rows:
        if row["manual_decision"] != "ACCEPT":
            continue
        if row["subtype_candidate"] != "OTHER_GESTURE":
            raise ValueError("All seven ACCEPT candidates must be OTHER_GESTURE")
        selected.append({**by_key[segment_key(row)], "review": row})
    if len(selected) != 7 or not TASK.is_file():
        raise ValueError("Expected seven ACCEPT segments and the canonical MediaPipe task")
    if (SEQUENCE_LENGTH, FEATURE_DIM) != (16, 69):
        raise ValueError("Dynamic sequence contract changed")
    return sorted(selected, key=lambda item: (str(item["video_id"]), int(item["start_0"])))


def audit_segment(item: dict[str, object], landmarker) -> tuple[dict[str, object], list[dict]]:
    import cv2
    import mediapipe as mp
    from ai.feature_extraction.hand_landmarks import landmarks_to_array

    path = item["path"]
    start, end = int(item["start_0"]), int(item["end_0"])
    records: list[dict] = []
    failure = ""
    cap = cv2.VideoCapture(str(path))  # File path, never a webcam index.
    try:
        if not cap.isOpened():
            failure = "VIDEO_OPEN_FAILED"
        else:
            for index in range(end + 1):
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    failure = f"DECODE_FAILED_AT_FRAME_{index}"
                    break
                if index < start:
                    continue  # Decode for reliable seeking, but do not detect outside segment.
                height, width = frame.shape[:2]
                record = {"index": index, "raw_hand_count": 0, "hands": [],
                          "invalid_hands": 0, "detection_error": ""}
                if height <= 0 or width <= 0:
                    record["detection_error"] = "INVALID_FRAME_DIMENSIONS"
                    records.append(record)
                    continue
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # No flip or mirror.
                    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                    hands = result.hand_landmarks or []
                    record["raw_hand_count"] = len(hands)
                    handedness = result.handedness or []
                    for hand_index, hand in enumerate(hands):
                        try:
                            xyz = landmarks_to_array(hand)
                            points_xy = xyz[:, :2] * np.array([width, height], dtype=np.float64)
                            scale = continuity.palm_scale(points_xy)
                            if not math.isfinite(scale):
                                raise ValueError("Degenerate palm scale")
                        except (AttributeError, TypeError, ValueError):
                            record["invalid_hands"] += 1
                            continue
                        categories = handedness[hand_index] if hand_index < len(handedness) else []
                        label = categories[0].category_name if categories else ""
                        record["hands"].append({"points_xy": points_xy, "scale": scale,
                                                "handedness": label})
                except Exception as exc:
                    record["detection_error"] = type(exc).__name__
                records.append(record)
    finally:
        cap.release()

    expected_frames = end - start + 1
    if failure or len(records) != expected_frames:
        classification = {
            "single_hand_frames": sum(f["raw_hand_count"] == 1 and len(f["hands"]) == 1
                                      for f in records),
            "multi_hand_frames": sum(f["raw_hand_count"] > 1 for f in records),
            "missing_hand_frames": sum(not f["hands"] for f in records),
            "invalid_hand_frames": sum(bool(f["invalid_hands"]) for f in records),
            "detection_error_frames": sum(bool(f["detection_error"]) for f in records),
            "single_hand_handedness_changes": 0, "internal_detection_gaps": 0,
            "max_wrist_step_palm_widths": 0.0, "max_shape_step_palm_widths": 0.0,
            "can_sample_same_hand_16": False, "proposed_original_frame_indices": [],
            "continuity_decision": "FAIL", "decision_reason": failure or "INCOMPLETE_SEGMENT_DECODE",
            "identity_not_proven": True,
        }
    else:
        # Existing conservative WLASL classifier reuses position, shape and
        # handedness diagnostics. No prior IPN coverage audit exists, so its
        # prior-valid parameter is the observed count from this same run.
        observed_valid = sum(bool(f["hands"]) for f in records)
        classification = continuity.classify_frames(records, observed_valid)

    valid = sum(bool(f["hands"]) for f in records)
    review = item["review"]
    detected = sum(f["raw_hand_count"] > 0 for f in records)
    row = {
        "video_id": item["video_id"], "annotation_label": item["label"],
        "segment_start_1based": item["start_1"], "segment_end_1based": item["end_1"],
        "source_video_path": path.relative_to(ROOT).as_posix(),
        "visual_review_status": "AI_VISUAL_REVIEW_ONLY",
        "visual_decision": "ACCEPT", "subtype_candidate": review["subtype_candidate"],
        "jz_risk": review["jz_risk"], "decoded_segment_frames": len(records),
        "frames_with_detected_hand": detected,
        "hand_detection_ratio": detected / expected_frames,
        "frames_with_valid_21_landmarks": valid,
        "landmark_coverage_ratio": valid / expected_frames,
        "single_hand_frames": classification["single_hand_frames"],
        "multi_hand_frames": classification["multi_hand_frames"],
        "missing_hand_frames": classification["missing_hand_frames"],
        "invalid_hand_frames": classification["invalid_hand_frames"],
        "detection_error_frames": classification["detection_error_frames"],
        "single_hand_handedness_changes": classification["single_hand_handedness_changes"],
        "internal_detection_gaps": classification["internal_detection_gaps"],
        "max_wrist_step_palm_widths": classification["max_wrist_step_palm_widths"],
        "max_shape_step_palm_widths": classification["max_shape_step_palm_widths"],
        "has_16_valid_frames": valid >= SEQUENCE_LENGTH,
        "can_sample_same_hand_16": classification["can_sample_same_hand_16"],
        "proposed_original_frame_indices": classification["proposed_original_frame_indices"],
        "technical_status": classification["continuity_decision"],
        "decision_reason": classification["decision_reason"],
        "identity_not_proven": True, "training_approved": False,
        "sequence_length": SEQUENCE_LENGTH, "feature_dim": FEATURE_DIM,
    }
    diagnostics = [{
        "video_id": item["video_id"], "annotation_label": item["label"],
        "original_frame_index": f["index"], "raw_hand_count": f["raw_hand_count"],
        "valid_hand_count": len(f["hands"]), "handedness_observed":
        [hand["handedness"] for hand in f["hands"]],
        "wrist_pixel_xy": [hand["points_xy"][0].tolist() for hand in f["hands"]],
        "palm_scales_px": [hand["scale"] for hand in f["hands"]],
        "invalid_hands": f["invalid_hands"], "detection_error": f["detection_error"],
    } for f in records]
    return row, diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Validate CSV/audit/path only; no video or MediaPipe")
    mode.add_argument("--audit", action="store_true", help="Confirm then audit exactly seven provisional ACCEPT segments")
    args = parser.parse_args(argv)
    try:
        plan = preflight()
        print(json.dumps({"status": "PREFLIGHT_PASS", "review_rows": 18,
                          "decisions": {"ACCEPT": 7, "REVIEW": 7, "REJECT": 4},
                          "accept_segments": [f"{p['video_id']}:{p['label']}:{p['start_1']}-{p['end_1']}"
                                              for p in plan],
                          "training_approved": False,
                          "outputs": [p.relative_to(ROOT).as_posix() for p in (REPORT, FRAMES, SUMMARY)]}))
        if args.check_only:
            return 0
        if any(path.exists() for path in (REPORT, FRAMES, SUMMARY)):
            raise ValueError("Audit output already exists; refusing overwrite")
        if input("Audit MediaPipe on seven provisional ACCEPT segments? [y/N] ").strip().lower() != "y":
            print("Cancelled; no output written")
            return 0
        from ai.feature_extraction.hand_landmarks import create_hand_landmarker

        rows: list[dict] = []
        frame_rows: list[dict] = []
        with create_hand_landmarker(TASK, running_mode="IMAGE", num_hands=2) as landmarker:
            for item in plan:
                row, diagnostics = audit_segment(item, landmarker)
                rows.append(row)
                frame_rows.extend(diagnostics)
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        with REPORT.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "proposed_original_frame_indices": json.dumps(
                    row["proposed_original_frame_indices"])})
        with FRAMES.open("x", encoding="utf-8") as stream:
            for frame in frame_rows:
                stream.write(json.dumps(frame, allow_nan=False) + "\n")
        with SUMMARY.open("x", encoding="utf-8") as stream:
            json.dump({"purpose": "IPN_PILOT_ACCEPT_MEDIAPIPE_DIAGNOSTIC_ONLY",
                       "visual_review_status": "AI_VISUAL_REVIEW_ONLY",
                       "technical_status_counts": dict(Counter(row["technical_status"] for row in rows)),
                       "segments": len(rows), "sequence_contract": [SEQUENCE_LENGTH, FEATURE_DIM],
                       "sampling_policy": "JZ_VALID_LANDMARK_FRAMES_V1_ONLY_IF_CONTINUITY_PASS",
                       "training_approved": False, "training_dataset_created": False,
                       "report": REPORT.relative_to(ROOT).as_posix(),
                       "frame_diagnostics": FRAMES.relative_to(ROOT).as_posix()},
                      stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
        print(json.dumps({"status": "AUDIT_REPORT_WRITTEN", "segments": len(rows)}))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
