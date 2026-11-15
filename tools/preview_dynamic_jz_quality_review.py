"""User-run previews of nine J/Z quality-review videos; never recompute frame selection.

--check-only reads saved reports/JSONL and checks the H.264 encoder, but opens no
videos and writes nothing. --execute requires confirmation before decoding video.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.extract_dynamic import make_preview_sheet  # noqa: E402


EXPERIMENT = ROOT / "data/experiments/dynamic_jz_v1"
SOURCE_ROOT = ROOT / "data/raw/dynamic/external/signnteam_jz/SigNN Video Data"
QUALITY_CSV = EXPERIMENT / "metadata/sequence_quality_audit_v1.csv"
EXTRACTION_CSV = EXPERIMENT / "metadata/signnteam_jz_extraction_audit_v1.csv"
SEQUENCES_JSONL = EXPERIMENT / "processed/signnteam_jz_baseline_b_v1.jsonl"
TECHNICAL_CSV = ROOT / "data/metadata/dynamic/technical_audit.csv"
OUTPUT_DIR = EXPERIMENT / "review/quality_review_9_v1"
REVIEW_CSV = OUTPUT_DIR / "manual_quality_review_v1.csv"
OUTPUT_MANIFEST = OUTPUT_DIR / "quality_preview_manifest_v1.json"
TARGETS = ("J/100.avi", "J/102.avi", "J/104.avi", "Z/109.avi", "Z/260.avi",
           "Z/272.avi", "Z/273.avi", "J/210.avi", "Z/298.avi")
FOOTER_HEIGHT = 88


def _csv_index(path: Path, key: str) -> dict[str, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        name = row[key]
        if name in result:
            raise ValueError(f"Duplicate {key} in {path.name}: {name}")
        result[name] = row
    return result


def _saved_sequences() -> dict[str, dict]:
    expected = {f"data/raw/dynamic/external/signnteam_jz/SigNN Video Data/{name}"
                for name in TARGETS}
    found = {}
    with SEQUENCES_JSONL.open("r", encoding="utf-8") as stream:
        for line in stream:
            # Avoid materializing 683 feature arrays; parse only the nine targets.
            if not any(f'"{name}"' in line for name in expected):
                continue
            row = json.loads(line)
            source = row.get("source_path")
            if source not in expected:
                continue
            if source in found:
                raise ValueError(f"Duplicate JSONL source: {source}")
            found[source] = row
    if set(found) != expected:
        raise ValueError(f"Missing JSONL sequences: {sorted(expected - set(found))}")
    return found


def validate_indices(indices: object, frame_count: int) -> list[int]:
    if (not isinstance(indices, list) or len(indices) != 16
            or any(type(index) is not int for index in indices)
            or indices != sorted(set(indices))
            or indices[0] < 0 or indices[-1] >= frame_count):
        raise ValueError("Saved indices must be 16 distinct increasing in-range integers")
    return indices


def preflight() -> list[dict]:
    quality = _csv_index(QUALITY_CSV, "source_key")
    extraction = _csv_index(EXTRACTION_CSV, "source_path")
    technical = _csv_index(TECHNICAL_CSV, "relative_path")
    sequences = _saved_sequences()
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"No overwrite: {OUTPUT_DIR}")
    plan = []
    for relative in TARGETS:
        label = relative.split("/", 1)[0]
        source = f"data/raw/dynamic/external/signnteam_jz/SigNN Video Data/{relative}"
        path = (SOURCE_ROOT / relative).resolve()
        if (path != SOURCE_ROOT.resolve() / relative or not path.is_file()
                or path.parent != (SOURCE_ROOT / label).resolve()):
            raise ValueError(f"Missing/unsafe source path: {relative}")
        q, e, t, sequence = (quality[relative], extraction[source], technical[relative],
                             sequences[source])
        if (q["source_path"] != source or e["source_path"] != source
                or sequence["source_path"] != source
                or any(row["label"] != label for row in (q, e, t, sequence))
                or q["quality_decision"] != "REVIEW"
                or e["extraction_status"] != "SUCCESS"
                or t["technical_status"] != "OK"):
            raise ValueError(f"Label/status/source mismatch: {relative}")
        count = int(t["decoded_frame_count"])
        if count < 16 or any(int(row["decoded_frame_count"]) != count
                             for row in (q, e, sequence)):
            raise ValueError(f"Decoded frame-count mismatch: {relative}")
        fps, width, height = float(t["fps"]), int(t["width"]), int(t["height"])
        if not math.isfinite(fps) or fps <= 0 or width < 2 or height < 2:
            raise ValueError(f"Invalid FPS/dimensions: {relative}")
        indices = validate_indices(json.loads(e["selected_original_frame_indices"]), count)
        if indices != validate_indices(sequence["selected_original_frame_indices"], count):
            raise ValueError(f"JSONL/extraction frame-index mismatch: {relative}")
        if int(sequence["feature_dim"]) != 69 or int(sequence["sequence_length"]) != 16:
            raise ValueError(f"Unexpected sequence shape: {relative}")
        valid = int(q["valid_landmark_frames"])
        coverage = float(q["landmark_coverage_ratio"])
        if valid < 16 or abs(coverage - valid / count) > 1e-9:
            raise ValueError(f"Coverage mismatch: {relative}")
        reasons = json.loads(q["invalid_reason_counts"] or "{}")
        no_hand = int(reasons.get("NO_HAND", 0))
        if no_hand < 0 or no_hand > count - valid:
            raise ValueError(f"Invalid NO_HAND count: {relative}")
        # Saved quality reports give NO_HAND totals, not original-frame positions.
        # The video footer must not present any interval as established evidence.
        base = relative.replace("/", "_").replace(".avi", "")
        plan.append({"relative": relative, "label": label, "source_path": source,
                     "path": path, "frame_count": count, "fps": fps,
                     "width": width, "height": height, "indices": indices,
                     "coverage": coverage, "gap": int(q["max_selected_frame_gap"]),
                     "issue": q["review_reasons"], "no_hand_count": no_hand,
                     "no_hand_intervals": None,
                     "video_output": OUTPUT_DIR / f"{base}_full_h264.mp4",
                     "sheet_output": OUTPUT_DIR / f"{base}_extracted_16.png"})
    if len(plan) != 9 or len({str(item["video_output"]) for item in plan}) != 9:
        raise ValueError("Expected nine unique videos")
    return plan


def h264_encoder(ffmpeg: str | None) -> tuple[str | None, str]:
    executable = ffmpeg or shutil.which("ffmpeg")
    if not executable or not Path(executable).is_file():
        return None, "FFmpeg executable unavailable; install FFmpeg with libx264 or pass --ffmpeg"
    probe = subprocess.run([executable, "-hide_banner", "-encoders"], capture_output=True,
                           text=True, check=False, timeout=20)
    if probe.returncode != 0 or "libx264" not in probe.stdout:
        return None, "FFmpeg does not expose libx264 H.264 encoder"
    return executable, "libx264 available"


def render_frame(frame, index: int, item: dict):
    """Preserve original frame pixels; add only a separate diagnostic footer."""
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    canvas = np.zeros((height + FOOTER_HEIGHT, width, 3), dtype=np.uint8)
    canvas[:height] = frame
    selected = index in item["indices"]
    selection_kind = item.get("selection_kind", "EXTRACTED_16")
    lines = [f"{item['relative']} | original frame 0-based: {index}/{item['frame_count'] - 1}",
             f"{selection_kind}: {'YES' if selected else 'NO'} | NO_HAND total: {item['no_hand_count']}",
             "NO_HAND positions unavailable in saved metadata; no interval inferred"]
    for row, line in enumerate(lines):
        cv2.putText(canvas, line, (8, height + 21 + row * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def write_one(item: dict, ffmpeg: str) -> None:
    import cv2

    capture = cv2.VideoCapture(str(item["path"]))  # File path, never webcam index.
    process = None
    selected_frames = []
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {item['relative']}")
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if (abs(actual_fps - item["fps"]) > 0.05
                or int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) != item["width"]
                or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) != item["height"]):
            raise ValueError(f"Source FPS/dimensions differ from technical audit: {item['relative']}")
        width, height = item["width"], item["height"] + FOOTER_HEIGHT
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
                   "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
                   "-framerate", str(item["fps"]), "-i", "pipe:0", "-an", "-c:v",
                   "libx264", "-preset", "medium", "-crf", "18", "-vf",
                   "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(item["video_output"])]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE)
        selected_position = 0
        for index in range(item["frame_count"]):
            ok, frame = capture.read()
            if not ok or frame is None or frame.shape[:2] != (item["height"], item["width"]):
                raise ValueError(f"Decode failed at original frame {index}: {item['relative']}")
            if selected_position < 16 and index == item["indices"][selected_position]:
                selected_frames.append(frame.copy())
                selected_position += 1
            process.stdin.write(render_frame(frame, index, item).tobytes())
        if capture.read()[0] or selected_position != 16:
            raise ValueError(f"Decoded frame-count/selected frames mismatch: {item['relative']}")
        process.stdin.close()
        errors = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise ValueError(f"H.264 encode failed: {errors[-1000:]}")
        verification = cv2.VideoCapture(str(item["video_output"]))
        decoded = 0
        try:
            if not verification.isOpened() or abs(float(verification.get(cv2.CAP_PROP_FPS)) - item["fps"]) > 0.05:
                raise ValueError(f"Encoded preview cannot reopen at source FPS: {item['relative']}")
            while True:
                ok, preview_frame = verification.read()
                if not ok:
                    break
                if preview_frame is None or preview_frame.shape[0] < height:
                    raise ValueError(f"Encoded preview frame invalid: {item['relative']}")
                decoded += 1
        finally:
            verification.release()
        if decoded != item["frame_count"]:
            raise ValueError(f"Encoded preview count {decoded}/{item['frame_count']}: {item['relative']}")
        title = f"{item['relative']} | {item.get('selection_kind', 'EXTRACTED_16')}"
        sheet = make_preview_sheet(selected_frames, item["indices"], title)
        encoded, buffer = cv2.imencode(".png", sheet)
        if not encoded:
            raise ValueError(f"Contact sheet PNG encoding failed: {item['relative']}")
        with item["sheet_output"].open("xb") as stream:
            stream.write(buffer.tobytes())
    finally:
        capture.release()
        if process is not None and process.poll() is None:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            process.terminate()
            process.wait()


def write_review(plan: list[dict]) -> None:
    columns = ["video", "label", "coverage", "gap", "issue", "visual_review",
               "sequence_review", "notes"]
    with REVIEW_CSV.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in plan:
            writer.writerow({"video": item["relative"], "label": item["label"],
                             "coverage": item["coverage"], "gap": item["gap"],
                             "issue": item["issue"], "visual_review": "PENDING",
                             "sequence_review": "PENDING", "notes": ""})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--ffmpeg", help="Path to FFmpeg executable with libx264")
    args = parser.parse_args(argv)
    try:
        plan = preflight()
        encoder, encoder_status = h264_encoder(args.ffmpeg)
        print(json.dumps({"status": "PREFLIGHT_PASS" if encoder else "ENCODER_REQUIRED",
                          "videos": len(plan), "saved_indices_verified": 9,
                          "no_hand_intervals": "UNAVAILABLE_IN_SAVED_METADATA",
                          "h264_encoder": encoder_status, "videos_opened": 0,
                          "files_written": 0}))
        if args.check_only:
            return 0 if encoder else 2
        if encoder is None:
            raise ValueError(encoder_status)
        if input("Create nine H.264 videos, contact sheets and pending review CSV? [y/N] ").strip().lower() != "y":
            print("CANCELLED: no video opened or file written")
            return 0
        OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
        outputs = []
        for item in plan:
            write_one(item, encoder)
            outputs.append({"source_path": item["source_path"],
                            "video_output": item["video_output"].relative_to(ROOT).as_posix(),
                            "contact_sheet": item["sheet_output"].relative_to(ROOT).as_posix(),
                            "saved_original_frame_indices": item["indices"],
                            "source_fps": item["fps"], "source_frame_count": item["frame_count"],
                            "no_hand_count": item["no_hand_count"],
                            "no_hand_intervals": None})
        write_review(plan)
        with OUTPUT_MANIFEST.open("x", encoding="utf-8") as stream:
            json.dump({"purpose": "JZ_SEQUENCE_QUALITY_VISUAL_REVIEW_ONLY",
                       "encoder": "libx264 / H.264 MP4", "selected_indices_source":
                       "extraction audit and JSONL, exact match; never recomputed",
                       "source_frames": "chronological, no flip, no interpolation; footer only",
                       "no_hand_interval_policy": "counts only; positions unavailable, not inferred",
                       "mediapipe_rerun": False, "dataset_created": False,
                       "review_decisions": "PENDING", "videos": outputs}, stream,
                      indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": "PREVIEWS_CREATED", "videos": len(outputs),
                          "output_dir": OUTPUT_DIR.relative_to(ROOT).as_posix()}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            subprocess.SubprocessError, BrokenPipeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
