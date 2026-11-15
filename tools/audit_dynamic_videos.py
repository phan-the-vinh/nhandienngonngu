"""Read-only technical audit of external J/Z AVI videos (no MediaPipe).

Run manually from the project root:
    python tools/audit_dynamic_videos.py

An AVI may report CAP_PROP_FRAME_COUNT=0 yet decode correctly. This tool counts
frames by sequential reads; an ordinary end-of-stream cannot be distinguished
from every form of truncated tail, so OK means at least one frame decoded and
no explicit decode exception/invalid decoded frame was observed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "dynamic.json"
FIELDS = (
    "relative_path", "label", "filename", "can_open", "first_frame_ok",
    "decoded_frame_count", "metadata_frame_count", "fps", "width", "height",
    "duration_seconds", "technical_status", "error_message",
)


def config_paths() -> tuple[Path, Path]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("task") != "dynamic_asl" or config.get("dynamic_labels") != ["J", "Z", "NONE"]:
        raise ValueError("Unexpected dynamic task/labels in config")
    if config.get("raw_external_root") != "data/raw/dynamic/external/signnteam_jz/SigNN Video Data":
        raise ValueError("Unexpected raw external root in config")
    if config.get("metadata_dir") != "data/metadata/dynamic":
        raise ValueError("Unexpected metadata directory in config")
    return ROOT / config["raw_external_root"], ROOT / config["metadata_dir"]


def infer_label(relative_path: Path) -> str:
    directory_labels = {part.upper() for part in relative_path.parts[:-1] if part.upper() in {"J", "Z"}}
    prefix = re.match(r"^([JZ])(?=$|[0-9_. -])", relative_path.stem, re.IGNORECASE)
    filename_label = prefix.group(1).upper() if prefix else None
    if len(directory_labels) > 1:
        raise ValueError(f"Conflicting label directories: {relative_path}")
    directory_label = next(iter(directory_labels), None)
    if directory_label and filename_label and directory_label != filename_label:
        raise ValueError(f"Directory/filename label disagreement: {relative_path}")
    label = directory_label or filename_label
    if label is None:
        raise ValueError(f"Cannot determine J/Z label from path: {relative_path}")
    return label


def positive_number(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) and value > 0 else None


def audit_video(path: Path, raw_root: Path, label: str) -> dict:
    relative = path.relative_to(raw_root)
    row = dict.fromkeys(FIELDS, "")
    row.update(relative_path=relative.as_posix(), label=label, filename=path.name,
               can_open=False, first_frame_ok=False, decoded_frame_count=0,
               metadata_frame_count=0, technical_status="OPEN_FAIL")
    capture = None
    try:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            row["error_message"] = "OpenCV could not open the AVI"
            return row
        row["can_open"] = True
        row["technical_status"] = "DECODE_FAIL"
        reported = positive_number(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        row["metadata_frame_count"] = int(round(reported)) if reported is not None else 0
        fps = positive_number(capture.get(cv2.CAP_PROP_FPS))
        row["fps"] = round(fps, 6) if fps is not None else ""
        width = positive_number(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = positive_number(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        row["width"] = int(round(width)) if width is not None else 0
        row["height"] = int(round(height)) if height is not None else 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                row["error_message"] = "OpenCV returned an empty decoded frame"
                return row
            if row["decoded_frame_count"] == 0:
                row["first_frame_ok"] = True
                row["width"], row["height"] = int(frame.shape[1]), int(frame.shape[0])
            row["decoded_frame_count"] += 1

        if row["decoded_frame_count"] == 0:
            row["error_message"] = "No frames could be decoded"
        else:
            row["technical_status"] = "OK"
        if fps is not None:
            row["duration_seconds"] = round(row["decoded_frame_count"] / fps, 6)
    except Exception as exc:
        row["technical_status"] = "DECODE_FAIL" if row["can_open"] else "OPEN_FAIL"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
    finally:
        if capture is not None:
            capture.release()
    return row


def write_report(output: Path, rows: list[dict], replace: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not replace:
        raise FileExistsError(f"Report already exists: {output}; use --replace-report explicitly")
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", suffix=".tmp",
                                         prefix="technical_audit_", dir=output.parent,
                                         delete=False) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        if output.exists() and not replace:
            raise FileExistsError(f"Report became occupied: {output}")
        if replace:
            os.replace(temporary_name, output)
        else:
            Path(temporary_name).rename(output)  # Windows refuses existing destinations.
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace-report", action="store_true",
                        help="Explicitly replace an existing derived technical report")
    args = parser.parse_args()
    try:
        raw_root, metadata_dir = config_paths()
        if not raw_root.is_dir():
            raise FileNotFoundError(f"Raw J/Z directory is missing: {raw_root}")
        output = metadata_dir / "technical_audit.csv"
        if output.exists() and not args.replace_report:
            raise FileExistsError(f"Report already exists: {output}")
        videos = sorted((path for path in raw_root.rglob("*") if path.is_file()
                         and path.suffix.lower() == ".avi"),
                        key=lambda path: path.relative_to(raw_root).as_posix().casefold())
        if not videos:
            raise ValueError(f"No AVI videos found under {raw_root}")
        labelled = [(path, infer_label(path.relative_to(raw_root))) for path in videos]
        rows = [audit_video(path, raw_root, label) for path, label in labelled]
        write_report(output, rows, args.replace_report)
        counts = Counter(row["technical_status"] for row in rows)
        print(f"TOTAL={len(rows)} J_TOTAL={sum(row['label'] == 'J' for row in rows)} "
              f"Z_TOTAL={sum(row['label'] == 'Z' for row in rows)} "
              f"OK={counts['OK']} OPEN_FAIL={counts['OPEN_FAIL']} "
              f"DECODE_FAIL={counts['DECODE_FAIL']}")
        print(f"REPORT={output}")
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
