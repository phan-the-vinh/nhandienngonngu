"""User-operated visual aid for 21 WLASL single-hand continuity REVIEW videos.

--check-only reads saved audits; --preview renders only the 11 priority videos.
--preview-alert-frames renders local chronological neighborhoods around saved alerts.
--check-only --sampled-16 and --preview-sampled-16 cover eight pending videos.
Neither mode runs MediaPipe, extracts features, or changes review decisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import preview_wlasl_other_gesture as existing_preview  # noqa: E402
from ai.feature_extraction.dynamic_sequence import (  # noqa: E402
    FEATURE_DIM, SEQUENCE_LENGTH, sample_valid_frame_indices,
)

EXPERIMENT = ROOT / "data/experiments/dynamic_none_v1"
AUDIT = EXPERIMENT / "metadata/wlasl_hand_identity_continuity_audit_v1.csv"
DIAGNOSTICS = EXPERIMENT / "metadata/wlasl_hand_identity_frame_diagnostics_v1.jsonl"
REVIEW = EXPERIMENT / "review/wlasl_other_gesture/review_manifest_v2.csv"
OUTPUT = EXPERIMENT / "review/wlasl_hand_identity_v1"
TABLE = OUTPUT / "manual_continuity_review_v1.csv"
ALERT_OUTPUT = OUTPUT / "alert_frame_neighborhoods"
ALERT_INDEX = ALERT_OUTPUT / "alert_frame_index_v1.csv"
SAMPLED_OUTPUT = OUTPUT / "sampled_16_visual_review_v1"
SAMPLED_TABLE = SAMPLED_OUTPUT / "sampled_16_visual_assessment_v1.csv"
PRIORITY = (
    "23769", "53343", "69418", "69527", "11768", "31093",
    "60554", "68010", "68054", "06331", "20666",
)
TARGET_8 = ("23769", "53343", "69418", "69527", "11768", "31093", "60554", "68010")
# Independently reported in the preceding read-only audit; comparison only.
# The source of the new indices remains sample_valid_frame_indices().
PREVIOUS_SAMPLED_16 = {
    "23769": [24, 28, 33, 36, 40, 43, 46, 50, 53, 57, 60, 63, 67, 70, 74, 77],
    "53343": [13, 19, 22, 26, 30, 34, 37, 41, 45, 49, 52, 56, 60, 64, 68, 73],
    "69418": [18, 20, 22, 25, 27, 29, 31, 33, 36, 38, 40, 42, 44, 47, 50, 52],
    "69527": [24, 26, 29, 32, 35, 37, 39, 42, 44, 47, 49, 51, 54, 56, 59, 61],
    "11768": [10, 13, 16, 19, 23, 26, 29, 32, 35, 38, 41, 44, 48, 51, 54, 57],
    "31093": [18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48],
    "60554": [14, 17, 20, 24, 27, 30, 33, 36, 40, 43, 46, 49, 52, 56, 59, 62],
    "68010": [4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49],
}
VISUAL_ASSESSMENTS = (
    "SAME_HAND_LIKELY", "UNCERTAIN", "DISCONTINUITY_OBSERVED",
)
SAMPLED_TABLE_FIELDS = (
    "video_id", "source", "gloss", "segment_start", "segment_end",
    "sampled_original_frame_indices", "selected_handedness", "warning_positions",
    "contact_sheet_path", "manual_note_from_prior_review", "visual_assessment",
    "allowed_visual_assessments", "visual_review_note", "manual_decision",
    "source_rights_status", "training_sample_created",
)
EXPECTED_REASON_COUNTS = {"INTERNAL_GAP_OR_AMBIGUOUS_FRAME": 4,
                          "HANDEDNESS_CHANGE_OR_UNKNOWN_CORROBORATING_SIGNAL": 5,
                          "ABRUPT_POSITION_OR_21_LANDMARK_SHAPE_STEP": 2}
ALERT_KIND = {
    "23769": "GAP", "53343": "GAP", "69418": "GAP", "69527": "GAP",
    "11768": "HANDEDNESS", "31093": "HANDEDNESS", "60554": "HANDEDNESS",
    "68010": "HANDEDNESS", "68054": "HANDEDNESS",
    "06331": "WRIST_JUMP", "20666": "SHAPE_JUMP",
}
ALERT_INDEX_FIELDS = (
    "video_id", "alert_kind", "alert_frame", "selected_original_frame_indices",
    "segment_start", "segment_end", "diagnostic_status", "image_path",
    "source_rights_status", "manual_decision",
)
TABLE_FIELDS = (
    "video_id", "priority", "source", "gloss", "local_path", "segment_start",
    "segment_end", "decision_reason", "internal_gap_indices", "handedness_change_indices",
    "unknown_handedness_indices", "wrist_jump_indices", "shape_jump_location",
    "evidence_note", "source_rights_status", "contact_sheet_path", "manual_decision",
    "manual_note",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_locations(frames: list[dict], row: dict[str, str]) -> dict[str, object]:
    """Locate only events recoverable from saved frame diagnostics.

    Shape-step positions cannot be recovered: the JSONL omits 21-point geometry.
    Handedness is a corroborating signal, never hand identity proof.
    """
    frames = sorted(frames, key=lambda frame: int(frame["original_frame_index"]))
    single = [frame for frame in frames if int(frame["raw_hand_count"]) == 1
              and int(frame["valid_hand_count"]) == 1]
    valid = [frame for frame in frames if int(frame["valid_hand_count"]) > 0]
    gaps = []
    if valid:
        first, last = (int(valid[0]["original_frame_index"]),
                       int(valid[-1]["original_frame_index"]))
        gaps = [int(frame["original_frame_index"]) for frame in frames
                if first <= int(frame["original_frame_index"]) <= last
                and (int(frame["raw_hand_count"]) != 1
                     or int(frame["valid_hand_count"]) != 1)]
    changes = []
    unknown = []
    jumps = []
    previous = None
    for frame in single:
        index = int(frame["original_frame_index"])
        label = frame["handedness_observed"][0]
        if label not in ("Left", "Right"):
            unknown.append(index)
        if previous is not None:
            previous_label = previous["handedness_observed"][0]
            if label and previous_label and label != previous_label:
                changes.append(index)
            if index == int(previous["original_frame_index"]) + 1:
                a, b = previous["wrist_pixel_xy"][0], frame["wrist_pixel_xy"][0]
                scale = (float(previous["palm_scales_px"][0])
                         + float(frame["palm_scales_px"][0])) / 2
                if math.isfinite(scale) and scale > 1e-6:
                    step = math.hypot(float(b[0])-float(a[0]), float(b[1])-float(a[1])) / scale
                    if step > 1.5:
                        jumps.append(index)
        previous = frame
    shape_alert = float(row["max_shape_step_palm_widths"]) > 1.0
    return {"gaps": gaps, "changes": changes, "unknown": unknown,
            "wrist_jumps": jumps,
            "shape_location": "NOT_STORED_IN_FRAME_DIAGNOSTICS" if shape_alert else "NOT_ALERTED"}


def preflight(video_ids: tuple[str, ...] | None = None) -> list[dict[str, object]]:
    rows = read_csv(AUDIT)
    review_rows = read_csv(REVIEW)
    if len(rows) != 103 or len(review_rows) != 121:
        raise ValueError("Unexpected audit/review manifest row count")
    by_id = {row["video_id"]: row for row in rows}
    review_by_id = {row["video_id"]: row for row in review_rows}
    if len(by_id) != 103 or len(review_by_id) != 121:
        raise ValueError("Duplicate video_id in CSV inputs")
    all_selected = [row for row in rows if row["continuity_decision"] == "REVIEW"
                    and int(row["multi_hand_frames"]) == 0]
    if len(all_selected) != 21 or not set(PRIORITY) <= {row["video_id"] for row in all_selected}:
        raise ValueError("Expected exactly 21 single-hand REVIEW videos and all 11 priorities")
    if video_ids is not None and video_ids != TARGET_8:
        raise ValueError("Sampled-16 review is restricted to the eight approved video IDs")
    selected = ([row for row in all_selected if row["video_id"] in video_ids]
                if video_ids is not None else all_selected)
    if video_ids is not None:
        if {row["video_id"] for row in selected} != set(TARGET_8):
            raise ValueError("Missing one or more of the eight target videos")
    else:
        reason_counts = Counter()
        for video_id in PRIORITY:
            reason_counts.update(by_id[video_id]["decision_reason"].split("|"))
        if dict(reason_counts) != EXPECTED_REASON_COUNTS:
            raise ValueError(f"Priority alert distribution changed: {dict(reason_counts)}")
    if set(ALERT_KIND) != set(PRIORITY) or (SEQUENCE_LENGTH, FEATURE_DIM) != (16, 69):
        raise ValueError("Alert scope or J/Z 16x69 feature contract changed")
    wanted = {row["video_id"] for row in selected}
    diagnostics: dict[str, list[dict]] = defaultdict(list)
    with DIAGNOSTICS.open("r", encoding="utf-8") as stream:
        for line in stream:
            frame = json.loads(line)
            if frame["video_id"] in wanted:
                diagnostics[frame["video_id"]].append(frame)
    plan = []
    for row in sorted(selected, key=lambda item: item["video_id"]):
        video_id = row["video_id"]
        review = review_by_id.get(video_id)
        if review is None or review["review_status"] != "ACCEPT_OTHER_GESTURE":
            raise ValueError(f"Missing accepted review segment: {video_id}")
        start, end = int(row["segment_start"]), int(row["segment_end"])
        if (start != int(review["preview_start"]) or end != int(review["preview_end"])
                or row["local_path"] != review["local_path"]):
            raise ValueError(f"Reviewed segment/path mismatch: {video_id}")
        frames = sorted(diagnostics[video_id], key=lambda f: int(f["original_frame_index"]))
        indices = [int(frame["original_frame_index"]) for frame in frames]
        if indices != list(range(start, end + 1)) or len(frames) != int(row["decoded_segment_frames"]):
            raise ValueError(f"Incomplete/duplicate frame diagnostics: {video_id}")
        if row["source_rights_status"] != "SOURCE_RIGHTS_UNVERIFIED" or any(
            frame["source_rights_status"] != "SOURCE_RIGHTS_UNVERIFIED" for frame in frames
        ):
            raise ValueError(f"Source rights status changed: {video_id}")
        if (int(row["sequence_length"]) != SEQUENCE_LENGTH
                or int(row["feature_dim"]) != FEATURE_DIM
                or row["sequence_sampling_policy"] != "JZ_VALID_LANDMARK_FRAMES_V1"):
            raise ValueError(f"J/Z sequence sampling contract changed: {video_id}")
        if any(int(frame["raw_hand_count"]) > 1 for frame in frames):
            raise ValueError(f"Two-hand frame in selected video: {video_id}")
        path = (ROOT / row["local_path"]).resolve()
        if (not path.is_relative_to((EXPERIMENT / "raw/wlasl_other_gesture").resolve())
                or not path.is_file() or path.stem != f"WLASL_{video_id}"):
            raise ValueError(f"Invalid source video path: {video_id}")
        if sha256(path) != row["source_sha256"]:
            raise ValueError(f"Source video changed since audit: {video_id}")
        events = event_locations(frames, row)
        if (len(events["gaps"]) != int(row["internal_detection_gaps"])
                or len(events["changes"]) != int(row["single_hand_handedness_changes"])):
            raise ValueError(f"Saved diagnostics disagree with summary: {video_id}")
        plan.append({"row": row, "review": review, "frames": frames,
                     "events": events, "path": path})
    if video_ids is not None and not TABLE.is_file():
        raise ValueError("The existing manual continuity review table is required")
    if TABLE.is_file():
        manual = read_csv(TABLE)
        manual_by_id = {row["video_id"]: row for row in manual}
        if (len(manual) != 21 or len(manual_by_id) != 21
                or ({item["row"]["video_id"] for item in plan} - set(manual_by_id))
                or {row["video_id"] for row in manual if row["priority"] == "True"} != set(PRIORITY)):
            raise ValueError("Manual review table does not match the 21-video audit scope")
        for item in plan:
            row = item["row"]
            note = manual_by_id[row["video_id"]]
            if (note["manual_decision"] != "PENDING_USER_REVIEW"
                    or note["source_rights_status"] != "SOURCE_RIGHTS_UNVERIFIED"
                    or note["segment_start"] != row["segment_start"]
                    or note["segment_end"] != row["segment_end"]
                    or note["decision_reason"] != row["decision_reason"]
                    or (row["video_id"] in PRIORITY and not note["manual_note"].strip())):
                raise ValueError(f"Unresolved manual review table mismatch: {row['video_id']}")
    return plan


def alert_frame_plan(plan: list[dict[str, object]]) -> list[dict[str, object]]:
    """One 4-before/4-after visual window per locatable saved warning."""
    result = []
    for item in plan:
        row = item["row"]
        video_id = row["video_id"]
        if video_id not in PRIORITY:
            continue
        kind = ALERT_KIND[video_id]
        events = item["events"]
        event_key = {"GAP": "gaps", "HANDEDNESS": "changes",
                     "WRIST_JUMP": "wrist_jumps"}.get(kind)
        centers = events[event_key] if event_key else []
        if kind == "SHAPE_JUMP" and events["shape_location"] != "NOT_STORED_IN_FRAME_DIAGNOSTICS":
            raise ValueError("Expected unlocatable shape-step alert for 20666")
        if event_key and not centers:
            raise ValueError(f"No saved {kind} frame for {video_id}")
        for center in centers or [None]:
            start, end = int(row["segment_start"]), int(row["segment_end"])
            indices = (list(range(max(start, center - 4), min(end, center + 4) + 1))
                       if center is not None else [])
            if center is not None and center not in indices:
                raise ValueError(f"Alert frame outside reviewed segment: {video_id}:{center}")
            path = (ALERT_OUTPUT / f"WLASL_{video_id}_{kind}_frame_{center:04d}.jpg"
                    if center is not None else None)
            result.append({"item": item, "kind": kind, "center": center,
                           "indices": indices, "path": path})
    return result


def alert_index_row(entry: dict[str, object]) -> dict[str, str]:
    row = entry["item"]["row"]
    return {
        "video_id": row["video_id"], "alert_kind": entry["kind"],
        "alert_frame": "" if entry["center"] is None else str(entry["center"]),
        "selected_original_frame_indices": json.dumps(entry["indices"]),
        "segment_start": row["segment_start"], "segment_end": row["segment_end"],
        "diagnostic_status": ("NOT_LOCATABLE_WITH_SAVED_DIAGNOSTICS"
                              if entry["center"] is None else "SAVED_DIAGNOSTIC_LOCATION"),
        "image_path": (str(entry["path"].relative_to(ROOT)) if entry["path"] else ""),
        "source_rights_status": "SOURCE_RIGHTS_UNVERIFIED",
        "manual_decision": "PENDING_USER_REVIEW",
    }


def sampled_16_plan(plan: list[dict[str, object]]) -> list[dict[str, object]]:
    """Compute hypothetical J/Z indices for visual review, never a training sample."""
    by_id = {item["row"]["video_id"]: item for item in plan}
    if set(by_id) != set(TARGET_8) or len(plan) != len(TARGET_8):
        raise ValueError("Sampled-16 scope must contain exactly the eight requested videos")
    result = []
    for video_id in TARGET_8:
        item = by_id[video_id]
        row = item["row"]
        if (row["continuity_decision"] != "REVIEW"
                or row["can_sample_same_hand_16"] != "False"
                or json.loads(row["proposed_original_frame_indices"]) != []):
            raise ValueError(f"Review gate or approved indices changed: {video_id}")
        by_frame = {int(frame["original_frame_index"]): frame for frame in item["frames"]}
        valid = [index for index, frame in sorted(by_frame.items())
                 if int(frame["raw_hand_count"]) == 1 and int(frame["valid_hand_count"]) == 1]
        if len(valid) != int(row["single_hand_frames"]):
            raise ValueError(f"Single-hand diagnostic count mismatch: {video_id}")
        indices = sample_valid_frame_indices(valid)
        if indices != PREVIOUS_SAMPLED_16[video_id]:
            raise ValueError(f"J/Z sampler differs from the previously reported indices: {video_id}")
        if (len(indices) != SEQUENCE_LENGTH or indices != sorted(set(indices))
                or indices[0] < int(row["segment_start"])
                or indices[-1] > int(row["segment_end"])):
            raise ValueError(f"Invalid hypothetical 16-frame sequence: {video_id}")
        warnings = []
        for kind, key in (("GAP", "gaps"), ("HANDEDNESS", "changes")):
            for frame_index in item["events"][key]:
                if frame_index in indices:
                    position = "AT_SELECTED_FRAME"
                    left = right = frame_index
                else:
                    left = max((index for index in indices if index < frame_index), default=None)
                    right = min((index for index in indices if index > frame_index), default=None)
                    if left is None or right is None:
                        raise ValueError(f"Warning outside selected frame span: {video_id}:{frame_index}")
                    position = "BETWEEN_SELECTED_FRAMES"
                warnings.append({"kind": kind, "frame": frame_index,
                                 "position": position, "left": left, "right": right})
        warnings.sort(key=lambda warning: (warning["frame"], warning["kind"]))
        expected_kind = "GAP" if video_id in TARGET_8[:4] else "HANDEDNESS"
        if not warnings or any(warning["kind"] != expected_kind for warning in warnings):
            raise ValueError(f"Unexpected warning evidence for {video_id}")
        labels = [by_frame[index]["handedness_observed"][0] for index in indices]
        result.append({"item": item, "indices": indices, "warnings": warnings,
                       "selected_handedness": labels,
                       "image_path": SAMPLED_OUTPUT / f"WLASL_{video_id}_sampled16.jpg"})
    return result


def sampled_16_table_row(entry: dict[str, object], manual: dict[str, str]) -> dict[str, str]:
    row = entry["item"]["row"]
    video_id = row["video_id"]
    if manual["manual_decision"] != "PENDING_USER_REVIEW" or not manual["manual_note"].strip():
        raise ValueError(f"Manual decision/note changed before sampled review: {video_id}")
    return {
        "video_id": video_id, "source": row["source"], "gloss": row["gloss"],
        "segment_start": row["segment_start"], "segment_end": row["segment_end"],
        "sampled_original_frame_indices": json.dumps(entry["indices"]),
        "selected_handedness": json.dumps(entry["selected_handedness"]),
        "warning_positions": json.dumps(entry["warnings"]),
        "contact_sheet_path": str(entry["image_path"].relative_to(ROOT)),
        "manual_note_from_prior_review": manual["manual_note"],
        "visual_assessment": "",  # Only the user fills one of VISUAL_ASSESSMENTS.
        "allowed_visual_assessments": "|".join(VISUAL_ASSESSMENTS),
        "visual_review_note": "", "manual_decision": "PENDING_USER_REVIEW",
        "source_rights_status": "SOURCE_RIGHTS_UNVERIFIED",
        "training_sample_created": "False",
    }


def choose_preview_indices(start: int, end: int, events: dict[str, object]) -> list[int]:
    """16 chronological visual frames, event-enriched, not model sampling."""
    if end - start + 1 < 16:
        raise ValueError("Reviewed segment too short for 16-frame visual sheet")
    event_set = {index + offset for key in ("gaps", "changes", "unknown", "wrist_jumps")
                 for index in events[key] for offset in (-1, 0, 1)
                 if start <= index + offset <= end}
    preferred = sorted(event_set - {start, end})
    if len(preferred) > 14:
        preferred = [preferred[round(i * (len(preferred) - 1) / 13)] for i in range(14)]
    chosen = {start, end, *preferred}
    for index in existing_preview.evenly_spaced_indices(start, end):
        if len(chosen) == 16:
            break
        chosen.add(index)
    for index in range(start, end + 1):
        if len(chosen) == 16:
            break
        chosen.add(index)
    return sorted(chosen)


def manual_row(item: dict[str, object], sheet_path: str = "") -> dict[str, str]:
    row, events = item["row"], item["events"]
    return {
        "video_id": row["video_id"], "priority": str(row["video_id"] in PRIORITY),
        "source": row["source"], "gloss": row["gloss"], "local_path": row["local_path"],
        "segment_start": row["segment_start"], "segment_end": row["segment_end"],
        "decision_reason": row["decision_reason"],
        "internal_gap_indices": json.dumps(events["gaps"]),
        "handedness_change_indices": json.dumps(events["changes"]),
        "unknown_handedness_indices": json.dumps(events["unknown"]),
        "wrist_jump_indices": json.dumps(events["wrist_jumps"]),
        "shape_jump_location": str(events["shape_location"]),
        "evidence_note": "Saved wrist/hand-count/handedness only; no 21-point overlay. "
                         "Handedness alone does not establish identity.",
        "source_rights_status": "SOURCE_RIGHTS_UNVERIFIED",
        "contact_sheet_path": sheet_path, "manual_decision": "PENDING_USER_REVIEW",
        "manual_note": "",
    }


def render(item: dict[str, object], indices: list[int], *, focus_frame: int | None = None,
           focus_kind: str = "", preview_mode: str = "CONTINUITY_AUDIT_SEGMENT",
           review_reason: str | None = None):
    row, review, events = item["row"], item["review"], item["events"]
    if (not indices or indices != sorted(set(indices))
            or indices[0] < int(row["segment_start"])
            or indices[-1] > int(row["segment_end"])):
        raise ValueError("Selected frames must be ordered and inside reviewed segment")
    indexed_frames = existing_preview.selected_frames(
        item["path"], indices, expected_count=len(indices))
    sheet = existing_preview.make_sheet({
        "video_id": row["video_id"], "gloss": row["gloss"],
        "signer_id": row["signer_id"], "source": row["source"],
        "metadata_frame_start": review["metadata_frame_start"],
        "metadata_frame_end": review["metadata_frame_end"],
        "preview_start": row["segment_start"], "preview_end": row["segment_end"],
        "preview_mode": "ALERT_NEIGHBORHOOD" if focus_frame is not None else preview_mode,
        "duration_seconds": review["duration_seconds"], "local_fps": review["local_fps"],
        "review_reason": (f"{focus_kind} at original frame {focus_frame}; blue dot=saved wrist"
                          if focus_frame is not None else review_reason or row["decision_reason"]),
    }, indexed_frames)
    diagnostics = {int(frame["original_frame_index"]): frame for frame in item["frames"]}
    for pos, (index, image) in enumerate(indexed_frames):
        flags = [name for name, key in (("GAP", "gaps"), ("HAND", "changes"),
                                        ("UNKNOWN", "unknown"), ("WRIST", "wrist_jumps"))
                 if index in events[key]]
        if index == focus_frame:
            flags.insert(0, "FOCUS")
        if flags:
            x, y = (pos % 4) * 320 + 8, 146 + (pos // 4) * 205 + 19
            cv2.putText(sheet, "/".join(flags), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (0, 0, 200), 2, cv2.LINE_AA)
        wrists = diagnostics[index]["wrist_pixel_xy"]
        if len(wrists) == 1:
            height, width = image.shape[:2]
            ratio = min(320 / width, 180 / height)
            fitted_width, fitted_height = max(1, round(width * ratio)), max(1, round(height * ratio))
            tile_x, tile_y = (pos % 4) * 320, 146 + (pos // 4) * 205
            offset_x = tile_x + (320 - fitted_width) // 2
            offset_y = tile_y + (180 - fitted_height) // 2
            wrist_x, wrist_y = wrists[0]
            if math.isfinite(wrist_x) and math.isfinite(wrist_y):
                cv2.circle(sheet, (offset_x + round(wrist_x * ratio),
                                   offset_y + round(wrist_y * ratio)), 4,
                           (255, 0, 0), -1, cv2.LINE_AA)
    return sheet


def render_sampled_16(entry: dict[str, object]):
    """Exactly 16 chronological visual frames; warnings remain annotated, not filtered."""
    indices = entry["indices"]
    sheet = render(entry["item"], indices, preview_mode="JZ_HYPOTHETICAL_16_NOT_APPROVED",
                   review_reason="PENDING_USER_REVIEW; rights UNVERIFIED; blue dot=saved wrist")
    for position, index in enumerate(indices):
        warnings = [warning for warning in entry["warnings"] if warning["right"] == index]
        if not warnings:
            continue
        between = [warning for warning in warnings
                   if warning["position"] == "BETWEEN_SELECTED_FRAMES"]
        at_frame = [warning for warning in warnings
                    if warning["position"] == "AT_SELECTED_FRAME"]
        messages = []
        if between:
            kinds = "/".join(sorted({warning["kind"] for warning in between}))
            frames = "/".join(str(warning["frame"]) for warning in between)
            messages.append(f"{kinds} {frames} BETWEEN {indices[position - 1]}-{index}")
        if at_frame:
            kinds = "/".join(sorted({warning["kind"] for warning in at_frame}))
            messages.append(f"{kinds} AT FRAME {index}")
        x, y = (position % 4) * 320 + 8, 146 + (position // 4) * 205 + 40
        for line_number, message in enumerate(messages):
            cv2.putText(sheet, message, (x, y + line_number * 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 0, 190), 1, cv2.LINE_AA)
    return sheet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Validate metadata, diagnostics and hashes; no video decoding")
    mode.add_argument("--preview", action="store_true", help="Confirm then render 11 priority sheets and 21-row manual table")
    mode.add_argument("--preview-alert-frames", action="store_true",
                      help="Confirm then render 4-before/4-after alert frames; no MediaPipe")
    mode.add_argument("--preview-sampled-16", action="store_true",
                      help="Confirm then render the hypothetical J/Z sampler's exact 16 frames for eight videos")
    parser.add_argument("--sampled-16", action="store_true",
                        help="With --check-only, restrict validation to the eight sampled-16 review videos")
    args = parser.parse_args(argv)
    if args.sampled_16 and not args.check_only:
        parser.error("--sampled-16 is only valid with --check-only")
    try:
        sampled_scope = args.sampled_16 or args.preview_sampled_16
        plan = preflight(TARGET_8 if sampled_scope else None)
        if sampled_scope:
            sampled = sampled_16_plan(plan)
            print(json.dumps({
                "status": "SAMPLED_16_PREFLIGHT_PASS", "videos": len(sampled),
                "indices_match_prior_analysis": True,
                "hypothetical_indices_only": {entry["item"]["row"]["video_id"]: entry["indices"]
                                              for entry in sampled},
                "approved_training_sequences": 0,
                "visual_assessment_options": VISUAL_ASSESSMENTS,
                "source_rights_status": "SOURCE_RIGHTS_UNVERIFIED",
                "feature_contract_unchanged": [SEQUENCE_LENGTH, FEATURE_DIM],
            }))
            if args.check_only:
                return 0
            if (SAMPLED_TABLE.exists()
                    or any(entry["image_path"].exists() for entry in sampled)):
                raise ValueError("Sampled-16 output exists; refusing overwrite")
            if input("Create eight hypothetical sampled-16 contact sheets and a blank visual assessment table? [y/N] ").strip().lower() != "y":
                return 0
            manual = {row["video_id"]: row for row in read_csv(TABLE)}
            table_rows = [sampled_16_table_row(entry, manual[entry["item"]["row"]["video_id"]])
                          for entry in sampled]
            SAMPLED_OUTPUT.mkdir(parents=True, exist_ok=True)
            for entry in sampled:
                if not cv2.imwrite(str(entry["image_path"]), render_sampled_16(entry)):
                    raise OSError(f"Cannot write {entry['image_path']}")
            with SAMPLED_TABLE.open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=SAMPLED_TABLE_FIELDS)
                writer.writeheader()
                writer.writerows(table_rows)
            print(json.dumps({"status": "SAMPLED_16_VISUAL_REVIEW_READY", "sheets": 8,
                              "manual_decisions_changed": 0,
                              "assessment_table": str(SAMPLED_TABLE.relative_to(ROOT))}))
            return 0
        alerts = alert_frame_plan(plan)
        print(json.dumps({"status": "PREFLIGHT_PASS", "review_single_hand": len(plan),
                          "priority": len(PRIORITY), "priority_ids": list(PRIORITY),
                          "locatable_alert_windows": sum(entry["center"] is not None for entry in alerts),
                          "priority_alert_frames": {
                              item["row"]["video_id"]: item["events"] for item in plan
                              if item["row"]["video_id"] in PRIORITY},
                          "shape_location_unavailable": [item["row"]["video_id"] for item in plan
                                                         if item["events"]["shape_location"] != "NOT_ALERTED"],
                          "source_rights_status": "SOURCE_RIGHTS_UNVERIFIED",
                          "feature_contract_unchanged": [16, 69],
                          "visual_windows_are_not_JZ_training_samples": True}))
        if args.check_only:
            return 0
        if args.preview_alert_frames:
            if not TABLE.is_file():
                raise ValueError("Manual review table is required before alert-frame preview")
            if ALERT_INDEX.exists() or any(
                entry["path"] is not None and entry["path"].exists() for entry in alerts
            ):
                raise ValueError("Alert preview output exists; refusing overwrite")
            if input("Create alert-frame neighborhood sheets from reviewed segments? [y/N] ").strip().lower() != "y":
                return 0
            ALERT_OUTPUT.mkdir(parents=True, exist_ok=True)
            for entry in alerts:
                if entry["center"] is None:
                    continue  # Shape-only 20666 cannot be localized from saved diagnostics.
                sheet = render(entry["item"], entry["indices"],
                               focus_frame=entry["center"], focus_kind=entry["kind"])
                if not cv2.imwrite(str(entry["path"]), sheet):
                    raise OSError(f"Cannot write {entry['path']}")
            with ALERT_INDEX.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=ALERT_INDEX_FIELDS)
                writer.writeheader()
                writer.writerows(alert_index_row(entry) for entry in alerts)
            print(json.dumps({"status": "ALERT_PREVIEW_READY_FOR_USER_REVIEW",
                              "sheets": sum(entry["center"] is not None for entry in alerts),
                              "unlocatable_shape_alert": ["20666"],
                              "index": str(ALERT_INDEX.relative_to(ROOT))}))
            return 0
        if TABLE.exists() or any((OUTPUT / f"WLASL_{video_id}_continuity.jpg").exists()
                                 for video_id in PRIORITY):
            raise ValueError("Review output exists; refusing overwrite")
        if input("Create 11 diagnostic contact sheets and a 21-row manual review table? [y/N] ").strip().lower() != "y":
            return 0
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for item in plan:
            if item["row"]["video_id"] not in PRIORITY:
                continue
            row = item["row"]
            indices = choose_preview_indices(int(row["segment_start"]),
                                             int(row["segment_end"]), item["events"])
            sheet = render(item, indices)
            path = OUTPUT / f"WLASL_{row['video_id']}_continuity.jpg"
            if not cv2.imwrite(str(path), sheet):
                raise OSError(f"Cannot write {path}")
        with TABLE.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=TABLE_FIELDS)
            writer.writeheader()
            for item in plan:
                video_id = item["row"]["video_id"]
                sheet_path = (OUTPUT / f"WLASL_{video_id}_continuity.jpg") if video_id in PRIORITY else None
                writer.writerow(manual_row(item, str(sheet_path.relative_to(ROOT)) if sheet_path else ""))
        print(json.dumps({"status": "PREVIEW_READY_FOR_USER_REVIEW", "sheets": 11,
                          "manual_rows": len(plan), "output": str(OUTPUT.relative_to(ROOT))}))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
