"""User-run, manifest-only full-video previews for four J/Z label checks.

--check-only validates paths and reports without opening videos or writing.
--execute (after confirmation) copies every source frame, in order and at
source FPS, into lossless AVI previews with a separate diagnostic footer.
No MediaPipe, motion recomputation, extraction, or training is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "data/experiments/dynamic_jz_v1"
SOURCE_ROOT = ROOT / "data/raw/dynamic/external/signnteam_jz/SigNN Video Data"
PREVIEW_MANIFEST = EXPERIMENT / "preview/preview_manifest.json"
MOTION_MANIFEST = EXPERIMENT / "motion_preview/motion_preview_manifest.json"
TECHNICAL_AUDIT = ROOT / "data/metadata/dynamic/technical_audit.csv"
OUTPUT_DIR = EXPERIMENT / "review/video_label_check_v1"
OUTPUT_MANIFEST = OUTPUT_DIR / "video_preview_manifest_v1.json"
TARGETS = ("J/311.avi", "J/241.avi", "Z/278.avi", "Z/189.avi")
FOOTER_HEIGHT = 112


def _project_file(value: str, expected: Path) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe manifest path: {value}")
    actual = (ROOT / relative).resolve()
    if actual != expected.resolve() or not actual.is_file():
        raise ValueError(f"Manifest path missing or different: {value}")
    return actual


def _unique_by_source(rows: list[dict], name: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        source = row["source_path"]
        if source in result:
            raise ValueError(f"Duplicate source in {name}: {source}")
        result[source] = row
    return result


def preflight() -> list[dict]:
    preview = json.loads(PREVIEW_MANIFEST.read_text(encoding="utf-8-sig"))
    motion = json.loads(MOTION_MANIFEST.read_text(encoding="utf-8-sig"))
    if (preview.get("mode") != "preview_only" or motion.get("mode") != "motion_preview_only"
            or preview.get("sequence_length") != 16 or motion.get("sequence_length") != 16):
        raise ValueError("Unexpected preview manifest contract")
    preview_by_source = _unique_by_source(preview["videos"], "preview manifest")
    motion_by_source = _unique_by_source(motion["videos"], "motion manifest")
    with TECHNICAL_AUDIT.open("r", encoding="utf-8-sig", newline="") as stream:
        technical_rows = list(csv.DictReader(stream))
    technical_by_relative = {}
    for row in technical_rows:
        key = row["relative_path"]
        if key in technical_by_relative:
            raise ValueError(f"Duplicate technical audit path: {key}")
        technical_by_relative[key] = row

    if OUTPUT_DIR.resolve().is_relative_to(EXPERIMENT.resolve()) is False:
        raise ValueError("Output directory escapes dynamic experiment")
    plan = []
    for relative in TARGETS:
        label = relative.split("/", 1)[0]
        source = f"data/raw/dynamic/external/signnteam_jz/SigNN Video Data/{relative}"
        first, second, audit = (preview_by_source.get(source), motion_by_source.get(source),
                                technical_by_relative.get(relative))
        if first is None or second is None or audit is None:
            raise ValueError(f"Missing manifest/audit record: {relative}")
        if first["label"] != label or audit["label"] != label:
            raise ValueError(f"Label differs from target folder: {relative}")
        expected = SOURCE_ROOT / Path(relative)
        path = _project_file(source, expected)
        if path.parent != (SOURCE_ROOT / label).resolve():
            raise ValueError(f"Source outside J/Z label folder: {relative}")
        _project_file(first["preview_path"], EXPERIMENT / "preview" / Path(first["preview_path"]).name)
        _project_file(second["comparison_path"],
                      EXPERIMENT / "motion_preview" / Path(second["comparison_path"]).name)
        frames = int(audit["decoded_frame_count"])
        if (audit["technical_status"] != "OK" or frames < 16
                or frames != int(first["decoded_frame_count"])
                or frames != int(second["decoded_frame_count"])
                or frames != int(second["frame_count"])):
            raise ValueError(f"Decoded frame counts/status differ: {relative}")
        fps, width, height = float(audit["fps"]), int(audit["width"]), int(audit["height"])
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Invalid audited FPS/resolution: {relative}")
        full = first["sampled_frame_indices"]
        if (full != second["full_clip_indices"] or len(full) != 16
                or full != sorted(set(full)) or full[0] != 0 or full[-1] != frames - 1):
            raise ValueError(f"Original preview indices differ: {relative}")
        status, reason = second["status"], second["failure_reason"]
        bounds = tuple(second.get(key) for key in
                       ("active_start", "active_end", "margin_start", "margin_end"))
        if status == "OK":
            active_start, active_end, margin_start, margin_end = bounds
            if (not all(type(value) is int for value in bounds)
                    or not 0 <= margin_start <= active_start <= active_end <= margin_end < frames
                    or reason or len(second["sampled_frame_indices"]) != 16):
                raise ValueError(f"Invalid motion interval: {relative}")
        elif status == "REVIEW_REQUIRED":
            if not reason or any(value is not None for value in bounds) or second["sampled_frame_indices"]:
                raise ValueError(f"Untrusted REVIEW_REQUIRED interval: {relative}")
        else:
            raise ValueError(f"Unexpected motion status: {relative}/{status}")
        output = OUTPUT_DIR / f"{label}_{Path(relative).stem}_full_video_preview.avi"
        plan.append({"relative": relative, "source_path": source, "path": path,
                     "label": label, "frame_count": frames, "fps": fps,
                     "width": width, "height": height, "status": status,
                     "failure_reason": reason, "bounds": bounds, "output": output})
    if len(plan) != 4 or len({item["output"] for item in plan}) != 4:
        raise ValueError("Expected four distinct preview outputs")
    return plan


def frame_region(index: int, item: dict) -> str:
    if item["status"] == "REVIEW_REQUIRED":
        return "REVIEW_REQUIRED"
    active_start, active_end, margin_start, margin_end = item["bounds"]
    if active_start <= index <= active_end:
        return "ACTIVE_PROPOSAL"
    if margin_start <= index <= margin_end:
        return "MARGIN"
    return "OUTSIDE_PROPOSAL"


def render_frame(frame, index: int, item: dict):
    """Copy source pixels without flipping/cropping/resizing; annotate footer."""
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    canvas = np.zeros((height + FOOTER_HEIGHT, width, 3), dtype=np.uint8)
    canvas[:height] = frame
    lines = [f"{item['relative']} | original frame (0-based): {index} / {item['frame_count'] - 1}"]
    if item["status"] == "OK":
        active_start, active_end, margin_start, margin_end = item["bounds"]
        lines.extend((f"ACTIVE {active_start}-{active_end} | MARGIN {margin_start}-{margin_end}",
                      f"Region: {frame_region(index, item)} | motion interval is a PROPOSAL"))
        markers = [name for name, point in (("ACTIVE_START", active_start), ("ACTIVE_END", active_end),
                                           ("MARGIN_START", margin_start), ("MARGIN_END", margin_end))
                   if index == point]
        lines.append("Boundary: " + (", ".join(markers) if markers else "none"))
    else:
        lines.extend((f"REVIEW_REQUIRED: {item['failure_reason']}",
                      "No trusted active/margin interval; do not infer one", "Visual label check only"))
    for line_number, line in enumerate(lines):
        cv2.putText(canvas, line, (10, height + 24 + line_number * 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def write_video(item: dict) -> int:
    import cv2

    cap = cv2.VideoCapture(str(item["path"]))  # A validated video file, never a webcam index.
    writer = None
    written = 0
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open source: {item['relative']}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or abs(fps - item["fps"]) > 0.05:
            raise ValueError(f"FPS differs from technical audit: {item['relative']}")
        if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) != item["width"]
                or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) != item["height"]):
            raise ValueError(f"Resolution differs from technical audit: {item['relative']}")
        descriptor = os.open(item["output"], os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(descriptor)
        writer = cv2.VideoWriter(str(item["output"]), cv2.VideoWriter_fourcc(*"FFV1"),
                                 fps, (item["width"], item["height"] + FOOTER_HEIGHT))
        if not writer.isOpened():
            raise ValueError("Lossless FFV1 AVI writer unavailable; output left for manual inspection")
        for index in range(item["frame_count"]):
            ok, frame = cap.read()
            if not ok or frame is None or frame.shape[:2] != (item["height"], item["width"]):
                raise ValueError(f"Decode failed at frame {index}: {item['relative']}")
            writer.write(render_frame(frame, index, item))
            written += 1
        extra, _ = cap.read()
        if extra:
            raise ValueError(f"Source has more frames than audit: {item['relative']}")
    finally:
        if writer is not None:
            writer.release()
        cap.release()
    if written != item["frame_count"]:
        raise ValueError(f"Preview frame count mismatch: {item['relative']}")
    verify = cv2.VideoCapture(str(item["output"]))
    decoded = 0
    try:
        if not verify.isOpened() or abs(float(verify.get(cv2.CAP_PROP_FPS)) - item["fps"]) > 0.05:
            raise ValueError(f"Written preview cannot reopen at source FPS: {item['relative']}")
        while True:
            ok, frame = verify.read()
            if not ok:
                break
            if frame is None or frame.shape[:2] != (item["height"] + FOOTER_HEIGHT, item["width"]):
                raise ValueError(f"Written preview has invalid frame: {item['relative']}")
            decoded += 1
    finally:
        verify.release()
    if decoded != written:
        raise ValueError(f"Written preview decoded {decoded}/{written}: {item['relative']}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Validate manifests/paths; no video opened")
    mode.add_argument("--execute", action="store_true", help="Confirm, then create four full-video previews")
    args = parser.parse_args(argv)
    try:
        plan = preflight()
        if OUTPUT_DIR.exists():
            raise FileExistsError(f"Output directory already exists; no overwrite: {OUTPUT_DIR}")
        print(json.dumps({"status": "PREFLIGHT_PASS", "videos": [
            {"source": item["relative"], "frame_count": item["frame_count"],
             "fps": item["fps"], "motion_status": item["status"],
             "failure_reason": item["failure_reason"], "bounds": item["bounds"],
             "output": item["output"].relative_to(ROOT).as_posix()} for item in plan],
            "videos_opened": 0, "files_written": 0}, allow_nan=False))
        if args.check_only:
            return 0
        if input("Create four J/Z full-video previews? [y/N] ").strip().lower() != "y":
            print("CANCELLED: no video opened or file written")
            return 0
        OUTPUT_DIR.mkdir(parents=True)
        outputs = []
        for item in plan:
            written = write_video(item)
            outputs.append({"source_path": item["source_path"],
                            "output_path": item["output"].relative_to(ROOT).as_posix(),
                            "label_from_source_folder": item["label"],
                            "source_fps": item["fps"], "frames_written": written,
                            "motion_status_unchanged": item["status"],
                            "failure_reason_unchanged": item["failure_reason"],
                            "active_start_end_margin_start_end_0based": item["bounds"]})
        with OUTPUT_MANIFEST.open("x", encoding="utf-8") as stream:
            json.dump({"purpose": "JZ_FULL_VIDEO_LABEL_AND_MOTION_PROPOSAL_REVIEW_ONLY",
                       "codec": "FFV1 lossless AVI", "source_frames": "unaltered pixels and chronological order",
                       "frame_policy": "all original frames at original FPS; diagnostics in footer",
                       "mediapipe_rerun": False, "motion_recomputed": False,
                       "training_dataset_created": False, "review_decision_changed": False,
                       "videos": outputs}, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": "PREVIEW_COMPLETE", "video_count": len(outputs),
                          "manifest": OUTPUT_MANIFEST.relative_to(ROOT).as_posix()}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
