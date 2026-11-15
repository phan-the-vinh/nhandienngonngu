"""User-run, segment-bound landmark coverage audit for 103 reviewed WLASL videos.

--check-only reads CSV metadata and paths; --audit decodes only reviewed segments.
Neither mode creates 16x69 sequences, normalized features, or training samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT = ROOT / "data" / "experiments" / "dynamic_none_v1"
REVIEW = EXPERIMENT / "review" / "wlasl_other_gesture" / "review_manifest_v2.csv"
TECHNICAL = EXPERIMENT / "metadata" / "wlasl_other_gesture_technical_audit_v1.csv"
RAW = EXPERIMENT / "raw" / "wlasl_other_gesture"
TASK = ROOT / "models" / "mediapipe" / "hand_landmarker.task"
REPORT = EXPERIMENT / "metadata" / "wlasl_other_gesture_landmark_coverage_audit_v1.csv"
SUMMARY = EXPERIMENT / "metadata" / "wlasl_other_gesture_landmark_coverage_summary_v1.json"
EXPECTED_STATUS_COUNTS = {"ACCEPT_OTHER_GESTURE": 103, "REVIEW": 17, "REJECT": 1}
EXPECTED_SOURCES = {
    "aslbrick": 12, "asldeafined": 13, "aslu": 8,
    "elementalasl": 14, "scott": 10, "signschool": 14,
    "spreadthesign": 13, "startasl": 10, "valencia-asl": 9,
}
REPORT_FIELDS = (
    "video_id", "gloss", "signer_id", "source", "source_url", "split",
    "source_rights_status", "local_path", "source_sha256", "review_status",
    "manual_reason", "preview_mode", "segment_start", "segment_end",
    "metadata_frame_start", "metadata_frame_end", "technical_decoded_frame_count",
    "segment_expected_frames", "segment_decoded_frames", "valid_21_landmark_frames",
    "single_valid_hand_frames", "multiple_hand_frames", "invalid_landmark_frames",
    "no_hand_frames", "detection_error_frames", "handedness_missing_frames",
    "single_hand_handedness_changes", "observed_handedness", "coverage_ratio",
    "has_16_valid_frames", "hand_identity_status", "audit_status", "failure_reason",
)


def read_csv(path: Path, fields: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Invalid CSV header: {path}")
        if not fields.issubset(reader.fieldnames):
            raise ValueError(f"Missing CSV columns {sorted(fields - set(reader.fieldnames))}: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row: {path}")
    ids = [row["video_id"] for row in rows]
    if any(not video_id for video_id in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"Missing/duplicate video_id: {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def preflight() -> list[dict[str, object]]:
    """Validate scope, provenance, file identity and the exact reviewed segment."""
    from tools.preview_wlasl_other_gesture import choose_range

    review = read_csv(REVIEW, {
        "video_id", "gloss", "signer_id", "source", "local_path", "batch",
        "review_status", "manual_decision", "manual_reason", "preview_mode",
        "preview_start", "preview_end", "range_review_required", "metadata_frame_start",
        "metadata_frame_end", "metadata_fps_basis", "local_fps", "duration_seconds",
    })
    technical = read_csv(TECHNICAL, {
        "video_id", "gloss", "signer_id", "source", "url", "split", "local_path",
        "frame_start", "frame_end", "decoded_frame_count", "fps",
        "duration_seconds", "file_size_bytes", "actual_sha256", "sha256_match",
        "provenance_match", "decode_ok",
    })
    statuses = Counter(row["review_status"] for row in review)
    if dict(statuses) != EXPECTED_STATUS_COUNTS or len(technical) != 121:
        raise ValueError(f"Expected 103 ACCEPT, 17 REVIEW, 1 REJECT and 121 technical rows; found {statuses}")
    if not TASK.is_file():
        raise ValueError(f"MediaPipe task missing: {TASK}")
    by_id = {row["video_id"]: row for row in technical}
    accepted = [row for row in review if row["review_status"] == "ACCEPT_OTHER_GESTURE"]
    sources = Counter(row["source"] for row in accepted)
    if dict(sources) != EXPECTED_SOURCES:
        raise ValueError(f"Unexpected accepted-source distribution: {sources}")
    plan: list[dict[str, object]] = []
    seen_paths: set[Path] = set()
    for row in accepted:
        video_id = row["video_id"]  # Always a string: preserve leading zeros.
        tech = by_id.get(video_id)
        if tech is None:
            raise ValueError(f"No technical audit for video_id={video_id}")
        for left, right in (
            ("gloss", "gloss"), ("signer_id", "signer_id"),
            ("source", "source"), ("local_path", "local_path"),
            ("metadata_frame_start", "frame_start"),
            ("metadata_frame_end", "frame_end"),
        ):
            if row[left] != tech[right]:
                raise ValueError(f"Provenance mismatch {video_id}: {left}/{right}")
        if row["manual_decision"] != "ACCEPT_OTHER_GESTURE":
            raise ValueError(f"No explicit manual acceptance: {video_id}")
        if tech["sha256_match"] != "True" or tech["provenance_match"] != "True" or tech["decode_ok"] != "True":
            raise ValueError(f"Unresolved technical source issue: {video_id}")
        path = (ROOT / row["local_path"]).resolve()
        if path.parent != RAW.resolve() or path.stem != f"WLASL_{video_id}" or not path.is_file():
            raise ValueError(f"Invalid source path for {video_id}: {path}")
        if path in seen_paths:
            raise ValueError(f"Duplicate source path: {path}")
        seen_paths.add(path)
        if path.stat().st_size != int(tech["file_size_bytes"]):
            raise ValueError(f"Source size changed: {video_id}")
        decoded = int(tech["decoded_frame_count"])
        fps = float(tech["fps"])
        duration = float(tech["duration_seconds"])
        if not math.isfinite(fps) or not math.isfinite(duration) or float(row["local_fps"]) != fps:
            raise ValueError(f"Invalid/inconsistent video timing: {video_id}")
        if abs(float(row["duration_seconds"]) - duration) > 1e-6 or float(row["metadata_fps_basis"]) != 25.0:
            raise ValueError(f"Review/technical timing mismatch: {video_id}")
        expected_start, expected_end, mode, range_review, _ = choose_range(
            int(tech["frame_start"]), int(tech["frame_end"]), decoded, fps, duration,
        )
        start, end = int(row["preview_start"]), int(row["preview_end"])
        if (start, end, row["preview_mode"], row["range_review_required"]) != (
            expected_start, expected_end, mode, str(range_review),
        ):
            raise ValueError(f"Reviewed segment differs from source metadata: {video_id}")
        if range_review or mode == "FULL_CLIP_NAVIGATION_ONLY" or end - start + 1 < 16:
            raise ValueError(f"Unreviewed/short segment cannot be audited as accepted: {video_id}")
        plan.append({"review": row, "technical": tech, "path": path, "start": start, "end": end})
    return plan


def hand_identity_status(multiple: int, changes: int, missing: int) -> str:
    """Flag risks, never claim that per-frame detections prove continuous identity."""
    if multiple or changes or missing:
        return "REVIEW_REQUIRED_HAND_IDENTITY"
    return "HAND_IDENTITY_UNVERIFIED"


def audit_video(item: dict[str, object], landmarker) -> dict[str, object]:
    import cv2
    import mediapipe as mp
    from ai.feature_extraction.hand_landmarks import landmarks_to_array

    row = item["review"]
    tech = item["technical"]
    path = item["path"]
    start, end = int(item["start"]), int(item["end"])
    if sha256(path) != tech["actual_sha256"]:
        raise ValueError(f"Source SHA256 changed: {row['video_id']}")
    counts: Counter[str] = Counter()
    observed: set[str] = set()
    previous_single_handedness: str | None = None
    failure = ""
    cap = cv2.VideoCapture(str(path))  # Source file only; never webcam.
    try:
        if not cap.isOpened():
            failure = "VIDEO_OPEN_FAILED"
        else:
            # Sequential decode preserves original 0-based frame indices on mixed codecs.
            for frame_index in range(end + 1):
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    failure = f"DECODE_FAILED_AT_FRAME_{frame_index}"
                    break
                if frame_index < start:
                    continue
                counts["decoded"] += 1
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # No flip.
                    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                except Exception as exc:
                    counts["detection_error"] += 1
                    failure = f"MEDIAPIPE_ERROR:{type(exc).__name__}"
                    continue
                hands = result.hand_landmarks or []
                if len(hands) > 1:
                    counts["multiple"] += 1
                valid_indices = []
                invalid_in_frame = False
                for index, hand in enumerate(hands):
                    try:
                        landmarks_to_array(hand)
                        valid_indices.append(index)
                    except (AttributeError, TypeError, ValueError):
                        invalid_in_frame = True
                if invalid_in_frame:
                    counts["invalid_landmarks"] += 1
                if not valid_indices:
                    if not hands:
                        counts["no_hand"] += 1
                    continue
                counts["valid"] += 1  # At least one complete 21-landmark hand.
                if len(valid_indices) != 1 or len(hands) != 1:
                    continue  # Never choose an arbitrary hand from a two-hand frame.
                counts["single"] += 1
                categories = result.handedness[valid_indices[0]] if len(result.handedness) > valid_indices[0] else []
                label = categories[0].category_name if categories else ""
                if label not in ("Left", "Right"):
                    counts["missing_handedness"] += 1
                    continue
                observed.add(label)
                if previous_single_handedness is not None and label != previous_single_handedness:
                    counts["handedness_changes"] += 1
                previous_single_handedness = label
    finally:
        cap.release()
    expected = end - start + 1
    if counts["decoded"] != expected and not failure:
        failure = "SEGMENT_DECODE_COUNT_MISMATCH"
    identity = hand_identity_status(
        counts["multiple"], counts["handedness_changes"], counts["missing_handedness"],
    )
    if failure:
        status = "AUDIT_ERROR"
    elif counts["valid"] < 16:
        status = "REVIEW_REQUIRED_LOW_COVERAGE"
    elif identity == "REVIEW_REQUIRED_HAND_IDENTITY":
        status = "REVIEW_REQUIRED_HAND_IDENTITY"
    else:
        status = "COVERAGE_PASS_IDENTITY_UNVERIFIED"
    return {
        "video_id": row["video_id"], "gloss": row["gloss"],
        "signer_id": row["signer_id"], "source": row["source"],
        "source_url": tech["url"], "split": tech["split"],
        "source_rights_status": "SOURCE_RIGHTS_UNVERIFIED",
        "local_path": row["local_path"], "source_sha256": tech["actual_sha256"],
        "review_status": row["review_status"], "manual_reason": row["manual_reason"],
        "preview_mode": row["preview_mode"], "segment_start": start, "segment_end": end,
        "metadata_frame_start": row["metadata_frame_start"],
        "metadata_frame_end": row["metadata_frame_end"],
        "technical_decoded_frame_count": tech["decoded_frame_count"],
        "segment_expected_frames": expected, "segment_decoded_frames": counts["decoded"],
        "valid_21_landmark_frames": counts["valid"],
        "single_valid_hand_frames": counts["single"],
        "multiple_hand_frames": counts["multiple"],
        "invalid_landmark_frames": counts["invalid_landmarks"],
        "no_hand_frames": counts["no_hand"],
        "detection_error_frames": counts["detection_error"],
        "handedness_missing_frames": counts["missing_handedness"],
        "single_hand_handedness_changes": counts["handedness_changes"],
        "observed_handedness": "|".join(sorted(observed)),
        "coverage_ratio": counts["valid"] / counts["decoded"] if counts["decoded"] else "",
        "has_16_valid_frames": counts["valid"] >= 16,
        "hand_identity_status": identity, "audit_status": status,
        "failure_reason": failure,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Validate review/provenance/segments; no video decode")
    mode.add_argument("--audit", action="store_true", help="Confirm and audit only the 103 reviewed segments")
    args = parser.parse_args(argv)
    try:
        plan = preflight()
        print(json.dumps({
            "status": "PREFLIGHT_PASS", "accepted_videos": len(plan),
            "excluded_review": 17, "excluded_reject": 1,
            "sources": dict(sorted(Counter(item["review"]["source"] for item in plan).items())),
            "source_rights": "UNVERIFIED_PER_SOURCE; WLASL C-UDA alone is not source-specific clearance",
            "segment_policy": "review_manifest_v2 preview_start..preview_end inclusive, 0-based",
            "hand_identity": "flag risk; no automatic hand selection/identity claim",
            "report": display_path(REPORT),
            "summary": display_path(SUMMARY),
            "outputs_exist": REPORT.exists() or SUMMARY.exists(),
        }, indent=2))
        if args.check_only:
            return 0
        if REPORT.exists() or SUMMARY.exists():
            raise ValueError("Audit output exists; refusing overwrite")
        if input("Audit 103 accepted reviewed segments with MediaPipe? [y/N] ").strip().lower() != "y":
            print("Cancelled; no audit output written")
            return 0
        from ai.feature_extraction.hand_landmarks import create_hand_landmarker

        rows = []
        with create_hand_landmarker(TASK, running_mode="IMAGE", num_hands=2) as landmarker, \
                REPORT.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            for index, item in enumerate(plan, 1):
                result = audit_video(item, landmarker)
                rows.append(result)
                writer.writerow(result)
                stream.flush()  # An interrupted report remains inspectable, never overwritten.
                if index % 10 == 0 or index == len(plan):
                    print(f"Coverage audit: {index}/{len(plan)}", flush=True)
        summary = {
            "purpose": "SEGMENT_LANDMARK_COVERAGE_ONLY",
            "video_count": len(rows),
            "review_manifest_sha256": sha256(REVIEW),
            "technical_audit_sha256": sha256(TECHNICAL),
            "mediapipe_task_sha256": sha256(TASK),
            "reports": {"csv": display_path(REPORT)},
            "audit_status_counts": dict(sorted(Counter(row["audit_status"] for row in rows).items())),
            "at_least_16_valid_frames": sum(row["has_16_valid_frames"] for row in rows),
            "fewer_than_16_valid_frames": sum(not row["has_16_valid_frames"] for row in rows),
            "multiple_hand_videos": sum(row["multiple_hand_frames"] > 0 for row in rows),
            "handedness_change_videos": sum(row["single_hand_handedness_changes"] > 0 for row in rows),
            "per_source": {
                source: {
                    "videos": len(subset),
                    "at_least_16_valid_frames": sum(row["has_16_valid_frames"] for row in subset),
                    "rights_status": "SOURCE_RIGHTS_UNVERIFIED",
                }
                for source in sorted(EXPECTED_SOURCES)
                for subset in [[row for row in rows if row["source"] == source]]
            },
            "semantic_review_limit": "Manual decisions were based on contact sheets, not full-video or rights approval.",
            "hand_identity_limit": "IMAGE-mode per-frame detections do not prove continuous hand identity.",
            "training_dataset_created": False,
            "sequence_16x69_created": False,
        }
        with SUMMARY.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": "AUDIT_REPORT_WRITTEN", "video_count": len(rows)}))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
