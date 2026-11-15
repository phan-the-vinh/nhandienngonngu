"""User-run continuity audit for 103 accepted WLASL OTHER_GESTURE segments.

The audit produces diagnostics only: no 16x69 features, training dataset, or
hand selection in two-hand frames. --check-only never decodes or runs MediaPipe.
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

from ai.feature_extraction.dynamic_sequence import (  # noqa: E402
    FEATURE_DIM, SEQUENCE_LENGTH, sample_valid_frame_indices,
)
from tools import audit_wlasl_other_gesture_landmark_coverage as coverage  # noqa: E402


EXPERIMENT = ROOT / "data" / "experiments" / "dynamic_none_v1"
COVERAGE_CSV = EXPERIMENT / "metadata" / "wlasl_other_gesture_landmark_coverage_audit_v1.csv"
COVERAGE_SUMMARY = EXPERIMENT / "metadata" / "wlasl_other_gesture_landmark_coverage_summary_v1.json"
RIGHTS_AUDIT = EXPERIMENT / "metadata" / "wlasl_other_gesture_source_rights_audit_v1.json"
REPORT = EXPERIMENT / "metadata" / "wlasl_hand_identity_continuity_audit_v1.csv"
FRAMES = EXPERIMENT / "metadata" / "wlasl_hand_identity_frame_diagnostics_v1.jsonl"
SUMMARY = EXPERIMENT / "metadata" / "wlasl_hand_identity_continuity_summary_v1.json"
EXPECTED_COVERAGE = {
    "COVERAGE_PASS_IDENTITY_UNVERIFIED": 13,
    "REVIEW_REQUIRED_HAND_IDENTITY": 88,
    "REVIEW_REQUIRED_LOW_COVERAGE": 2,
}
# Conservative diagnostic limits per adjacent decoded frame, in palm widths.
# Exceeding them means REVIEW, not proof of a hand swap or a new label.
MAX_WRIST_STEP = 1.5
MAX_SHAPE_STEP = 1.0
MIN_PALM_SCALE_PX = 1e-6
REPORT_FIELDS = (
    "video_id", "gloss", "signer_id", "source", "source_url", "split",
    "source_rights_status", "local_path", "source_sha256", "segment_start",
    "segment_end", "prior_coverage_status", "prior_valid_frames",
    "decoded_segment_frames", "frames_with_valid_hand", "single_hand_frames",
    "multi_hand_frames", "missing_hand_frames", "invalid_hand_frames",
    "detection_error_frames", "single_hand_handedness_changes",
    "internal_detection_gaps", "max_wrist_step_palm_widths",
    "max_shape_step_palm_widths", "proposed_original_frame_indices",
    "sequence_sampling_policy", "sequence_length", "feature_dim",
    "can_sample_same_hand_16", "continuity_decision", "decision_reason",
    "identity_not_proven", "feature_dataset_created",
)


def preflight() -> list[dict[str, object]]:
    plan = coverage.preflight()  # Validates the exact reviewed segment and 103/17/1 scope.
    rows = coverage.read_csv(COVERAGE_CSV, {
        "video_id", "source", "local_path", "source_sha256", "source_rights_status",
        "segment_start", "segment_end", "segment_decoded_frames",
        "valid_21_landmark_frames", "audit_status", "has_16_valid_frames",
    })
    summary = json.loads(COVERAGE_SUMMARY.read_text(encoding="utf-8"))
    rights = json.loads(RIGHTS_AUDIT.read_text(encoding="utf-8"))
    if len(rows) != 103 or summary.get("video_count") != 103:
        raise ValueError("Coverage inputs must contain exactly 103 videos")
    if summary.get("review_manifest_sha256") != coverage.sha256(coverage.REVIEW):
        raise ValueError("Review manifest changed since coverage audit")
    if summary.get("mediapipe_task_sha256") != coverage.sha256(coverage.TASK):
        raise ValueError("MediaPipe task changed since coverage audit")
    if summary.get("audit_status_counts") != EXPECTED_COVERAGE:
        raise ValueError("Coverage-status distribution differs from 13/88/2")
    if dict(Counter(row["audit_status"] for row in rows)) != EXPECTED_COVERAGE:
        raise ValueError("Coverage CSV and summary status counts differ")
    if (summary.get("at_least_16_valid_frames") != 101
            or summary.get("fewer_than_16_valid_frames") != 2):
        raise ValueError("Coverage summary must have 101 videos with >=16 and 2 below 16")
    if SEQUENCE_LENGTH != 16 or FEATURE_DIM != 69:
        raise ValueError("J/Z sequence contract changed; stop rather than infer a new schema")
    if rights.get("training_eligible_from_rights_audit") is not False:
        raise ValueError("Unexpected rights-audit eligibility")
    rights_sources = {entry["source"]: entry for entry in rights["sources"]}
    if set(rights_sources) != set(coverage.EXPECTED_SOURCES) or any(
        entry["source_rights_status"] != "UNVERIFIED"
        or entry["accepted_videos"] != coverage.EXPECTED_SOURCES[source]
        for source, entry in rights_sources.items()
    ):
        raise ValueError("The nine source rights must remain UNVERIFIED")
    by_id = {row["video_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate video_id in coverage audit")
    for item in plan:
        review = item["review"]
        prior = by_id.get(review["video_id"])
        if prior is None:
            raise ValueError(f"Missing coverage row: {review['video_id']}")
        if (prior["source"] != review["source"]
                or prior["local_path"] != review["local_path"]
                or prior["source_sha256"] != item["technical"]["actual_sha256"]
                or prior["source_rights_status"] != "SOURCE_RIGHTS_UNVERIFIED"
                or int(prior["segment_start"]) != item["start"]
                or int(prior["segment_end"]) != item["end"]
                or int(prior["segment_decoded_frames"]) != item["end"] - item["start"] + 1):
            raise ValueError(f"Coverage provenance/segment mismatch: {review['video_id']}")
        item["prior"] = prior
    return plan


def palm_scale(points_xy: np.ndarray) -> float:
    scale = float(np.median(np.linalg.norm(points_xy[[5, 9, 17]] - points_xy[0], axis=1)))
    return scale if math.isfinite(scale) and scale > MIN_PALM_SCALE_PX else math.nan


def adjacent_steps(left: dict, right: dict) -> tuple[float, float]:
    """Pixel/aspect-ratio-aware wrist and wrist-relative 21-point shape steps."""
    scale = (left["scale"] + right["scale"]) / 2
    if not math.isfinite(scale) or scale <= MIN_PALM_SCALE_PX:
        return math.inf, math.inf
    a, b = left["points_xy"], right["points_xy"]
    wrist = float(np.linalg.norm(b[0] - a[0]) / scale)
    shape = float(np.median(np.linalg.norm((b - b[0]) - (a - a[0]), axis=1)) / scale)
    return wrist, shape


def classify_frames(frames: list[dict], prior_valid: int) -> dict[str, object]:
    """PASS only for one uninterrupted, smooth, uniquely detected hand run.

    Two-hand detections are never assigned to a track. Handedness is only one
    corroborating signal. Even PASS is technical evidence, not identity proof.
    """
    if not frames or any(right["index"] != left["index"] + 1
                         for left, right in zip(frames, frames[1:])):
        raise ValueError("Frame diagnostics must be consecutive within one reviewed segment")
    valid = [f for f in frames if f["hands"]]
    single = [f for f in frames if f["raw_hand_count"] == 1 and len(f["hands"]) == 1]
    multi = sum(f["raw_hand_count"] > 1 for f in frames)
    invalid = sum(bool(f["invalid_hands"]) for f in frames)
    detection_errors = sum(bool(f["detection_error"]) for f in frames)
    missing = sum(not f["hands"] and not f["detection_error"] for f in frames)
    labels = [f["hands"][0]["handedness"] for f in single]
    label_changes = sum(a != b for a, b in zip(labels, labels[1:]) if a and b)
    missing_labels = sum(label not in ("Left", "Right") for label in labels)
    internal_gaps = 0
    if valid:
        between = [f for f in frames if valid[0]["index"] <= f["index"] <= valid[-1]["index"]]
        internal_gaps = sum(len(f["hands"]) != 1 or f["raw_hand_count"] != 1 for f in between)
    wrist_steps: list[float] = []
    shape_steps: list[float] = []
    for left, right in zip(single, single[1:]):
        if right["index"] != left["index"] + 1:
            continue
        wrist, shape = adjacent_steps(left["hands"][0], right["hands"][0])
        wrist_steps.append(wrist)
        shape_steps.append(shape)
    max_wrist = max(wrist_steps, default=0.0)
    max_shape = max(shape_steps, default=0.0)
    reasons = []
    if len(valid) != prior_valid:
        reasons.append("COVERAGE_CHANGED_SINCE_PRIOR_AUDIT")
    if len(valid) < SEQUENCE_LENGTH:
        decision = "FAIL"
        reasons.append("FEWER_THAN_16_VALID_21_LANDMARK_FRAMES")
    else:
        if multi:
            reasons.append("TWO_HAND_FRAMES_NO_AUTOMATIC_TRACK_SELECTION")
        if invalid or detection_errors:
            reasons.append("INVALID_LANDMARK_OR_DETECTION_ERROR")
        if internal_gaps:
            reasons.append("INTERNAL_GAP_OR_AMBIGUOUS_FRAME")
        if len(single) != len(valid):
            reasons.append("NOT_ALL_VALID_FRAMES_UNIQUELY_SINGLE_HAND")
        if label_changes or missing_labels:
            reasons.append("HANDEDNESS_CHANGE_OR_UNKNOWN_CORROBORATING_SIGNAL")
        if max_wrist > MAX_WRIST_STEP or max_shape > MAX_SHAPE_STEP:
            reasons.append("ABRUPT_POSITION_OR_21_LANDMARK_SHAPE_STEP")
        if not wrist_steps or not shape_steps:
            reasons.append("INSUFFICIENT_ADJACENT_MOTION_EVIDENCE")
        decision = "REVIEW" if reasons else "PASS"
    # Exactly the J/Z sampler, applied only when the continuous track is unique.
    # No arbitrary filling/padding or sampling from conflicting hand detections.
    indices = sample_valid_frame_indices([f["index"] for f in single]) if decision == "PASS" else []
    return {
        "decoded_segment_frames": len(frames),
        "frames_with_valid_hand": len(valid),
        "single_hand_frames": len(single),
        "multi_hand_frames": multi,
        "missing_hand_frames": missing,
        "invalid_hand_frames": invalid,
        "detection_error_frames": detection_errors,
        "single_hand_handedness_changes": label_changes,
        "internal_detection_gaps": internal_gaps,
        "max_wrist_step_palm_widths": max_wrist,
        "max_shape_step_palm_widths": max_shape,
        "proposed_original_frame_indices": indices,
        "can_sample_same_hand_16": decision == "PASS",
        "continuity_decision": decision,
        "decision_reason": "|".join(reasons) if reasons else "UNIQUE_CONTIGUOUS_SMOOTH_SINGLE_HAND_EVIDENCE",
        "identity_not_proven": True,
    }


def audit_video(item: dict[str, object], landmarker) -> tuple[dict[str, object], list[dict]]:
    import cv2
    import mediapipe as mp
    from ai.feature_extraction.hand_landmarks import landmarks_to_array

    review, prior, tech = item["review"], item["prior"], item["technical"]
    path, start, end = item["path"], int(item["start"]), int(item["end"])
    if coverage.sha256(path) != prior["source_sha256"]:
        raise ValueError(f"Source SHA256 changed: {review['video_id']}")
    frames = []
    cap = cv2.VideoCapture(str(path))  # Never a webcam index.
    try:
        if not cap.isOpened():
            raise ValueError(f"Video cannot be opened: {review['video_id']}")
        for index in range(end + 1):
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                raise ValueError(f"Decode failed at {review['video_id']} frame {index}")
            if index < start:
                continue  # Decode to seek reliably, but never detect outside review segment.
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                raise ValueError(f"Invalid frame dimensions: {review['video_id']}:{index}")
            record = {"index": index, "raw_hand_count": 0, "hands": [],
                      "invalid_hands": 0, "detection_error": ""}
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # No flip/mirror.
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                raw_hands = result.hand_landmarks or []
                record["raw_hand_count"] = len(raw_hands)
                for hand_index, hand in enumerate(raw_hands):
                    try:
                        xyz = landmarks_to_array(hand)
                    except (AttributeError, TypeError, ValueError):
                        record["invalid_hands"] += 1
                        continue
                    points_xy = xyz[:, :2] * np.array([width, height], dtype=np.float64)
                    categories = (result.handedness[hand_index]
                                  if hand_index < len(result.handedness) else [])
                    label = categories[0].category_name if categories else ""
                    record["hands"].append({"points_xy": points_xy,
                                            "scale": palm_scale(points_xy),
                                            "handedness": label})
            except Exception as exc:
                record["detection_error"] = type(exc).__name__
            frames.append(record)
    finally:
        cap.release()
    if len(frames) != end - start + 1:
        raise ValueError(f"Reviewed segment decode count changed: {review['video_id']}")
    result = classify_frames(frames, int(prior["valid_21_landmark_frames"]))
    row = {
        "video_id": review["video_id"], "gloss": review["gloss"],
        "signer_id": review["signer_id"], "source": review["source"],
        "source_url": tech["url"], "split": tech["split"],
        "source_rights_status": "SOURCE_RIGHTS_UNVERIFIED",
        "local_path": review["local_path"], "source_sha256": prior["source_sha256"],
        "segment_start": start, "segment_end": end,
        "prior_coverage_status": prior["audit_status"],
        "prior_valid_frames": prior["valid_21_landmark_frames"],
        "sequence_sampling_policy": "JZ_VALID_LANDMARK_FRAMES_V1",
        "sequence_length": SEQUENCE_LENGTH, "feature_dim": FEATURE_DIM,
        "feature_dataset_created": False,
        **result,
    }
    diagnostics = [{
        "video_id": review["video_id"], "original_frame_index": f["index"],
        "raw_hand_count": f["raw_hand_count"], "valid_hand_count": len(f["hands"]),
        "handedness_observed": [hand["handedness"] for hand in f["hands"]],
        "wrist_pixel_xy": [hand["points_xy"][0].tolist() for hand in f["hands"]],
        "palm_scales_px": [hand["scale"] if math.isfinite(hand["scale"]) else None for hand in f["hands"]],
        "invalid_hands": f["invalid_hands"], "detection_error": f["detection_error"],
        "source_rights_status": "SOURCE_RIGHTS_UNVERIFIED",
    } for f in frames]
    return row, diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Validate metadata and paths; no video/MediaPipe")
    mode.add_argument("--audit", action="store_true", help="Confirm then audit 103 reviewed video segments")
    args = parser.parse_args(argv)
    try:
        plan = preflight()
        print(json.dumps({
            "status": "PREFLIGHT_PASS", "videos": len(plan),
            "prior_coverage_statuses": EXPECTED_COVERAGE,
            "rights_status": "SOURCE_RIGHTS_UNVERIFIED_FOR_9_SOURCES",
            "sequence_contract": [SEQUENCE_LENGTH, FEATURE_DIM],
            "sampling_policy": "sample_valid_frame_indices over one continuous hand only",
            "outputs": [coverage.display_path(p) for p in (REPORT, FRAMES, SUMMARY)],
            "outputs_exist": any(p.exists() for p in (REPORT, FRAMES, SUMMARY)),
        }, indent=2))
        if args.check_only:
            return 0
        if any(p.exists() for p in (REPORT, FRAMES, SUMMARY)):
            raise ValueError("Audit output exists; refusing overwrite")
        if input("Audit hand continuity in 103 reviewed WLASL segments? [y/N] ").strip().lower() != "y":
            print("Cancelled; no audit output written")
            return 0
        from ai.feature_extraction.hand_landmarks import create_hand_landmarker

        rows = []
        with create_hand_landmarker(coverage.TASK, running_mode="IMAGE", num_hands=2) as landmarker, \
                REPORT.open("x", encoding="utf-8", newline="") as csv_stream, \
                FRAMES.open("x", encoding="utf-8") as frame_stream:
            writer = csv.DictWriter(csv_stream, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            for index, item in enumerate(plan, 1):
                row, diagnostics = audit_video(item, landmarker)
                writer.writerow({**row, "proposed_original_frame_indices": json.dumps(
                    row["proposed_original_frame_indices"])})
                rows.append(row)
                for frame in diagnostics:
                    frame_stream.write(json.dumps(frame, allow_nan=False) + "\n")
                csv_stream.flush()
                frame_stream.flush()
                if index % 10 == 0 or index == len(plan):
                    print(f"Continuity audit: {index}/{len(plan)}", flush=True)
        result = {
            "purpose": "HAND_IDENTITY_CONTINUITY_DIAGNOSTIC_ONLY",
            "video_count": len(rows),
            "review_manifest_sha256": coverage.sha256(coverage.REVIEW),
            "coverage_audit_sha256": coverage.sha256(COVERAGE_CSV),
            "coverage_summary_sha256": coverage.sha256(COVERAGE_SUMMARY),
            "mediapipe_task_sha256": coverage.sha256(coverage.TASK),
            "source_rights_status": "SOURCE_RIGHTS_UNVERIFIED_FOR_9_SOURCES",
            "decisions": dict(sorted(Counter(row["continuity_decision"] for row in rows).items())),
            "pass_meaning": "Strong single-hand temporal evidence only; not semantic, rights, or absolute identity proof.",
            "sequence_policy": "Exact J/Z valid-frame-list uniform sampler, only for PASS; 16x69 features not generated.",
            "diagnostic_limits_palm_widths": {
                "adjacent_wrist_step": MAX_WRIST_STEP,
                "adjacent_shape_step": MAX_SHAPE_STEP,
                "purpose": "Conservative REVIEW triggers, not model thresholds",
            },
            "output_csv": coverage.display_path(REPORT),
            "frame_diagnostics_jsonl": coverage.display_path(FRAMES),
            "training_dataset_created": False,
        }
        with SUMMARY.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
        print(json.dumps({"status": "AUDIT_REPORT_WRITTEN", "decisions": result["decisions"]}))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
