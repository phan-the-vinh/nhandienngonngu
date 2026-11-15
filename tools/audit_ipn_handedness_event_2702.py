"""User-run deep diagnostic of one IPN Hand handedness event (frame 2702).

--check-only validates saved reports, source metadata, and the original task;
it opens no video and runs no MediaPipe. --audit repeats the original IMAGE-mode
detection on all 46 annotated frames and draws overlays only for 2698..2706.
Results are diagnostic evidence, never a hand-identity or training approval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import audit_ipn_pilot_accept_mediapipe as prior_audit  # noqa: E402

VIDEO_ID = "4CM11_23_R_#25"
LABEL = "G02"
START, END = 2683, 2728  # Original OpenCV frame indices, inclusive.
OBSERVE_START, OBSERVE_END = 2698, 2706
EXPECTED_TASK_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
DIFFERENCE_TOLERANCE_PX = 0.01  # Diagnostic replay flag, not a model threshold.
EXPERIMENT = ROOT / "data/experiments/dynamic_none_v1"
OUTPUT_DIR = EXPERIMENT / "review/ipn_handedness_event_2702_v1"
OVERLAY_DIR = OUTPUT_DIR / "overlays"
FRAME_REPORT = OUTPUT_DIR / "frame_diagnostics_v1.jsonl"
STEP_REPORT = OUTPUT_DIR / "adjacent_steps_v1.csv"
SHEET = OUTPUT_DIR / "contact_sheet_2698_2706.png"
SUMMARY = OUTPUT_DIR / "audit_summary_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_paths() -> list[Path]:
    return [FRAME_REPORT, STEP_REPORT, SHEET, SUMMARY,
            *(OVERLAY_DIR / f"frame_{index:04d}.png"
              for index in range(OBSERVE_START, OBSERVE_END + 1))]


def preflight() -> dict:
    plans = [item for item in prior_audit.preflight()
             if item["video_id"] == VIDEO_ID and item["label"] == LABEL]
    if len(plans) != 1:
        raise ValueError("Target is not a unique visually ACCEPTed IPN segment")
    item = plans[0]
    if (item["start_0"], item["end_0"]) != (START, END):
        raise ValueError("Target annotation frame range changed")
    if sha256(prior_audit.TASK) != EXPECTED_TASK_SHA256:
        raise ValueError("MediaPipe task hash differs from the original contract")
    summary = json.loads(prior_audit.SUMMARY.read_text(encoding="utf-8-sig"))
    if (summary.get("segments") != 7 or summary.get("technical_status_counts") != {"PASS": 1, "REVIEW": 6}
            or summary.get("training_approved") is not False):
        raise ValueError("Prior MediaPipe audit summary changed")
    with prior_audit.REPORT.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    matches = [row for row in rows if row["video_id"] == VIDEO_ID and row["annotation_label"] == LABEL]
    if len(matches) != 1:
        raise ValueError("Expected one prior MediaPipe audit row for the target")
    old_row = matches[0]
    if (old_row["technical_status"] != "REVIEW" or old_row["visual_review_status"] != "AI_VISUAL_REVIEW_ONLY"
            or old_row["training_approved"] != "False" or old_row["segment_start_1based"] != str(START + 1)
            or old_row["segment_end_1based"] != str(END + 1)
            or int(old_row["decoded_segment_frames"]) != END - START + 1):
        raise ValueError("Target prior audit contract changed")
    old_frames = []
    with prior_audit.FRAMES.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            frame = json.loads(line)
            if frame["video_id"] == VIDEO_ID and frame["annotation_label"] == LABEL:
                old_frames.append(frame)
    if [frame["original_frame_index"] for frame in old_frames] != list(range(START, END + 1)):
        raise ValueError("Prior frame diagnostics are missing, duplicated or unordered")
    if (sum(frame["raw_hand_count"] == 0 for frame in old_frames) != 0
            or sum(frame["raw_hand_count"] != 1 or frame["valid_hand_count"] != 1 for frame in old_frames) != 0
            or [old_frames[index - START]["handedness_observed"] for index in (2701, 2702, 2703)]
            != [["Right"], ["Left"], ["Right"]]):
        raise ValueError("Prior 46-frame handedness/coverage evidence changed")
    technical = prior_audit.preview.read_csv(prior_audit.preview.VIDEO_AUDIT)
    video_rows = [row for row in technical if row["video_id"] == VIDEO_ID]
    if len(video_rows) != 1 or video_rows[0]["video_status"] != "PASS":
        raise ValueError("Missing decode-PASS source-video record")
    video = video_rows[0]
    if (item["path"].stat().st_size != int(video["file_size_bytes"])
            or (int(video["width"]), int(video["height"])) != (640, 480)):
        raise ValueError("Source file size or resolution differs from decode audit")
    return {"path": item["path"], "old_row": old_row, "old_frames": old_frames,
            "old_by_index": {f["original_frame_index"]: f for f in old_frames},
            "width": int(video["width"]), "height": int(video["height"]),
            "fps": float(video["fps"])}


def categories_for_hand(categories: list) -> list[dict]:
    result = []
    for category in categories:
        score = getattr(category, "score", None)
        score = float(score) if score is not None else None
        result.append({"label": getattr(category, "category_name", ""),
                       "score": score if score is None or math.isfinite(score) else None})
    return result


def compare_to_old(new: dict, old: dict) -> dict:
    reasons = []
    if new["raw_hand_count"] != old["raw_hand_count"]:
        reasons.append("RAW_HAND_COUNT_CHANGED")
    if new["valid_hand_count"] != old["valid_hand_count"]:
        reasons.append("VALID_HAND_COUNT_CHANGED")
    if new["handedness_observed"] != old["handedness_observed"]:
        reasons.append("HANDEDNESS_LABEL_CHANGED")
    wrist_delta = scale_delta = None
    if new["valid_hand_count"] == old["valid_hand_count"] == 1:
        old_wrist = np.asarray(old["wrist_pixel_xy"][0], dtype=float)
        wrist_delta = float(np.linalg.norm(np.asarray(new["wrist_pixel_xy"][0]) - old_wrist))
        scale_delta = abs(float(new["palm_scales_px"][0]) - float(old["palm_scales_px"][0]))
        if wrist_delta > DIFFERENCE_TOLERANCE_PX:
            reasons.append("WRIST_POSITION_DIFFERENCE_GT_0.01PX")
        if scale_delta > DIFFERENCE_TOLERANCE_PX:
            reasons.append("PALM_SCALE_DIFFERENCE_GT_0.01PX")
    return {"replay_difference_flags": reasons, "wrist_delta_px": wrist_delta,
            "palm_scale_delta_px": scale_delta,
            "prior_handedness_observed": old["handedness_observed"]}


def adjacent_metrics(records: list[dict]) -> list[dict]:
    rows = []
    for left, right in zip(records, records[1:]):
        wrist = shape = None
        reason = "ONE_VALID_HAND_EACH_FRAME"
        if left["valid_hand_count"] == right["valid_hand_count"] == 1 \
                and left["raw_hand_count"] == right["raw_hand_count"] == 1:
            wrist, shape = prior_audit.continuity.adjacent_steps(
                left["_valid_hands"][0], right["_valid_hands"][0])
            if not math.isfinite(wrist) or not math.isfinite(shape):
                wrist = shape = None
                reason = "DEGENERATE_HAND_GEOMETRY"
        else:
            reason = "MISSING_INVALID_OR_MULTIPLE_HANDS"
        rows.append({"previous_frame_0based": left["original_frame_index"],
                     "frame_0based": right["original_frame_index"],
                     "wrist_step_palm_widths": wrist, "shape_step_palm_widths": shape,
                     "step_status": reason})
    return rows


def run_audit(plan: dict) -> tuple[list[dict], list[dict], dict[int, np.ndarray]]:
    import cv2
    import mediapipe as mp
    from ai.feature_extraction.hand_landmarks import create_hand_landmarker, landmarks_to_array

    records = []
    images = {}
    cap = cv2.VideoCapture(str(plan["path"]))  # Explicit source path, never webcam.
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open source video")
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(actual_fps) or abs(actual_fps - plan["fps"]) > 0.05:
            raise ValueError("Source FPS differs from prior decode audit")
        with create_hand_landmarker(prior_audit.TASK, running_mode="IMAGE", num_hands=2,
                                    min_hand_detection_confidence=0.5,
                                    min_hand_presence_confidence=0.5) as landmarker:
            for index in range(END + 1):
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    raise ValueError(f"Source decode failed at frame {index}")
                if index < START:
                    continue  # Decode in order, but never detect outside annotation.
                height, width = frame.shape[:2]
                if (width, height) != (plan["width"], plan["height"]):
                    raise ValueError(f"Frame resolution changed at {index}")
                if OBSERVE_START <= index <= OBSERVE_END:
                    images[index] = frame.copy()
                record = {"video_id": VIDEO_ID, "annotation_label": LABEL,
                          "original_frame_index": index, "raw_hand_count": 0,
                          "valid_hand_count": 0, "invalid_hand_count": 0,
                          "handedness_observed": [], "hands": [], "wrist_pixel_xy": [],
                          "palm_scales_px": [], "detection_error": ""}
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # No flip/mirror.
                    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                    hands = result.hand_landmarks or []
                    record["raw_hand_count"] = len(hands)
                    handedness = result.handedness or []
                    valid_hands = []
                    for hand_index, hand in enumerate(hands):
                        try:
                            xyz = landmarks_to_array(hand)  # Exactly 21 finite XYZ points.
                            xy = xyz[:, :2] * np.array([width, height], dtype=float)
                            scale = prior_audit.continuity.palm_scale(xy)
                            if not math.isfinite(scale):
                                raise ValueError("DEGENERATE_PALM_SCALE")
                        except (AttributeError, TypeError, ValueError) as exc:
                            record["invalid_hand_count"] += 1
                            record["hands"].append({"valid": False, "reason": str(exc)})
                            continue
                        categories = categories_for_hand(
                            handedness[hand_index] if hand_index < len(handedness) else [])
                        label = categories[0]["label"] if categories else ""
                        record["hands"].append({"valid": True,
                                                "landmarks_xyz_normalized": xyz.tolist(),
                                                "landmarks_xy_pixels": xy.tolist(),
                                                "handedness_categories": categories,
                                                "handedness_label": label,
                                                "handedness_confidence": categories[0]["score"] if categories else None,
                                                "wrist_pixel_xy": xy[0].tolist(),
                                                "palm_scale_px": scale})
                        record["handedness_observed"].append(label)
                        record["wrist_pixel_xy"].append(xy[0].tolist())
                        record["palm_scales_px"].append(scale)
                        valid_hands.append({"points_xy": xy, "scale": scale, "handedness": label})
                    record["valid_hand_count"] = len(valid_hands)
                    record["_valid_hands"] = valid_hands  # In-memory only, never serialized.
                except Exception as exc:
                    record["detection_error"] = type(exc).__name__
                    record["_valid_hands"] = []
                record["prior_comparison"] = compare_to_old(
                    record, plan["old_by_index"][index])
                records.append(record)
    finally:
        cap.release()
    if len(records) != END - START + 1 or set(images) != set(range(OBSERVE_START, OBSERVE_END + 1)):
        raise ValueError("Deep audit did not cover the exact required frame ranges")
    return records, adjacent_metrics(records), images


def make_overlay(frame: np.ndarray, record: dict) -> np.ndarray:
    import cv2

    output = frame.copy()
    palette = [(0, 255, 0), (0, 180, 255)]
    for hand_index, hand in enumerate(record["hands"]):
        if not hand.get("valid"):
            continue
        color = palette[hand_index % len(palette)]
        for point_index, (x, y) in enumerate(hand["landmarks_xy_pixels"]):
            point = (round(x), round(y))
            cv2.circle(output, point, 4, color, -1, cv2.LINE_AA)
            cv2.putText(output, str(point_index), (point[0] + 4, point[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    banner = np.zeros((56, frame.shape[1], 3), dtype=np.uint8)
    first = record["hands"][0] if record["hands"] else {}
    score = first.get("handedness_confidence")
    score_text = "N/A" if score is None else f"{score:.4f}"
    identity_text = (f"Handedness: {first.get('handedness_label', 'NONE')} | score: {score_text}"
                     if record["raw_hand_count"] == record["valid_hand_count"] == 1
                     else "MULTI/NO VALID HAND: identity ambiguous")
    cv2.putText(banner, f"Frame {record['original_frame_index']} (0-based) | hands {record['raw_hand_count']}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, identity_text,
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack((banner, output))


def summary_for(plan: dict, records: list[dict], steps: list[dict]) -> dict:
    import cv2
    import mediapipe as mp

    differences = [r for r in records if r["prior_comparison"]["replay_difference_flags"]]
    shape_values = [r["shape_step_palm_widths"] for r in steps if r["shape_step_palm_widths"] is not None]
    wrist_values = [r["wrist_step_palm_widths"] for r in steps if r["wrist_step_palm_widths"] is not None]
    by_pair = {(r["previous_frame_0based"], r["frame_0based"]): r for r in steps}
    new_max_wrist, new_max_shape = max(wrist_values, default=None), max(shape_values, default=None)
    old_max_wrist = float(plan["old_row"]["max_wrist_step_palm_widths"])
    old_max_shape = float(plan["old_row"]["max_shape_step_palm_widths"])
    return {"purpose": "HANDEDNESS_EVENT_DIAGNOSTIC_ONLY", "video_id": VIDEO_ID,
            "annotation_label": LABEL, "segment_original_frames_0based": [START, END],
            "overlay_original_frames_0based": [OBSERVE_START, OBSERVE_END],
            "source_video_path": plan["path"].relative_to(ROOT).as_posix(),
            "source_video_sha256_current": sha256(plan["path"]),
            "mediapipe_task_path": prior_audit.TASK.relative_to(ROOT).as_posix(),
            "mediapipe_task_sha256": EXPECTED_TASK_SHA256,
            "library_versions_current": {"mediapipe": mp.__version__, "opencv": cv2.__version__},
            "prior_audit_library_versions_available": False,
            "mediapipe_configuration": {"running_mode": "IMAGE", "num_hands": 2,
                                        "min_hand_detection_confidence": 0.5,
                                        "min_hand_presence_confidence": 0.5,
                                        "min_tracking_confidence": None, "frame_flip": False},
            "prior_audit_sha256": sha256(prior_audit.REPORT),
            "prior_frame_diagnostics_sha256": sha256(prior_audit.FRAMES),
            "processed_frames": len(records),
            "valid_single_hand_frames": sum(r["raw_hand_count"] == r["valid_hand_count"] == 1
                                            for r in records),
            "replay_difference_count": len(differences),
            "replay_difference_frames": [r["original_frame_index"] for r in differences],
            "replay_geometry_flag_tolerance_px": DIFFERENCE_TOLERANCE_PX,
            "max_replay_wrist_delta_px": max((r["prior_comparison"]["wrist_delta_px"] for r in records
                                               if r["prior_comparison"]["wrist_delta_px"] is not None), default=None),
            "max_replay_palm_scale_delta_px": max((r["prior_comparison"]["palm_scale_delta_px"] for r in records
                                                    if r["prior_comparison"]["palm_scale_delta_px"] is not None),
                                                   default=None),
            "max_wrist_step_palm_widths": new_max_wrist,
            "max_shape_step_palm_widths": new_max_shape,
            "prior_max_wrist_step_palm_widths": old_max_wrist,
            "prior_max_shape_step_palm_widths": old_max_shape,
            "prior_vs_rerun_max_step_difference_palm_widths": {
                "wrist": abs(new_max_wrist - old_max_wrist) if new_max_wrist is not None else None,
                "shape": abs(new_max_shape - old_max_shape) if new_max_shape is not None else None},
            "prior_vs_rerun_max_step_flag_gt_0_0001": (
                new_max_wrist is None or new_max_shape is None
                or abs(new_max_wrist - old_max_wrist) > 0.0001
                or abs(new_max_shape - old_max_shape) > 0.0001),
            "full_21_landmark_replay_comparison_possible": False,
            "event_pairs": {f"{a}_to_{b}": by_pair[(a, b)] for a, b in ((2701, 2702), (2702, 2703))},
            "identity_conclusion": "UNDETERMINED", "review_status": "REVIEW_UNCHANGED",
            "training_approved": False, "sequence_created": False}


def write_outputs(records: list[dict], steps: list[dict], images: dict[int, np.ndarray],
                  summary: dict) -> None:
    import cv2

    if any(path.exists() for path in output_paths()):
        raise ValueError("An output already exists; refusing overwrite")
    overlays = {index: make_overlay(images[index], records[index - START])
                for index in range(OBSERVE_START, OBSERVE_END + 1)}
    sheet = np.vstack([np.hstack([overlays[i] for i in range(start, start + 3)])
                       for start in (OBSERVE_START, OBSERVE_START + 3, OBSERVE_START + 6)])
    encoded = {}
    for path, image in [(SHEET, sheet), *((OVERLAY_DIR / f"frame_{index:04d}.png", overlay)
                                         for index, overlay in overlays.items())]:
        ok, buffer = cv2.imencode(".png", image)
        if not ok:
            raise ValueError(f"Cannot encode PNG: {path}")
        encoded[path] = buffer.tobytes()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    for path, payload in encoded.items():
        with path.open("xb") as stream:
            stream.write(payload)
    with FRAME_REPORT.open("x", encoding="utf-8") as stream:
        for record in records:
            serializable = {key: value for key, value in record.items() if key != "_valid_hands"}
            stream.write(json.dumps(serializable, allow_nan=False) + "\n")
    with STEP_REPORT.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("previous_frame_0based", "frame_0based",
                                                   "wrist_step_palm_widths", "shape_step_palm_widths",
                                                   "step_status"))
        writer.writeheader()
        writer.writerows(steps)
    with SUMMARY.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Validate saved reports, video stats and task hash only")
    mode.add_argument("--audit", action="store_true", help="Confirm then rerun MediaPipe for only this 46-frame segment")
    args = parser.parse_args(argv)
    try:
        plan = preflight()
        print(json.dumps({"status": "PREFLIGHT_PASS", "video_id": VIDEO_ID, "label": LABEL,
                          "segment_frames_0based": [START, END], "overlay_frames_0based":
                          [OBSERVE_START, OBSERVE_END], "prior_status": "REVIEW",
                          "old_diagnostics": len(plan["old_frames"]),
                          "task_sha256": EXPECTED_TASK_SHA256,
                          "outputs_exist": any(path.exists() for path in output_paths()),
                          "output_directory": OUTPUT_DIR.relative_to(ROOT).as_posix()}))
        if args.check_only:
            return 0
        if any(path.exists() for path in output_paths()):
            raise ValueError("Audit output already exists; refusing overwrite")
        if input("Rerun MediaPipe for only 4CM11_23_R_#25/G02 frames 2683-2728? [y/N] ").strip().lower() != "y":
            print("Cancelled; no video or MediaPipe opened")
            return 0
        records, steps, images = run_audit(plan)
        summary = summary_for(plan, records, steps)
        write_outputs(records, steps, images, summary)
        print(json.dumps({"status": "DIAGNOSTIC_WRITTEN_REVIEW_UNCHANGED", "frames": len(records),
                          "replay_difference_count": summary["replay_difference_count"],
                          "output_directory": OUTPUT_DIR.relative_to(ROOT).as_posix()}))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
