"""Human J/Z video curation in OpenCV; decisions live only in a CSV.

Run the technical audit first, then:
    python tools/review_dynamic_videos.py

SPACE play/pause; R replay; 1-4 handshape; G/I/U trajectory;
A/V/X decision; T note (terminal); N next; B previous; Q save and quit.
Completed decisions are saved immediately, so an interrupted session can resume.
No automatic gesture classification, MediaPipe, or raw-video modification.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "dynamic.json"
WINDOW_NAME = "Dynamic J/Z manual review"
FIELDS = (
    "relative_path", "label", "handshape_variant", "trajectory_quality",
    "decision", "notes", "reviewed_at",
)
TRAJECTORY_KEYS = {ord("g"): "complete", ord("i"): "incomplete", ord("u"): "unclear"}
DECISION_KEYS = {ord("a"): "GOOD", ord("v"): "REVIEW", ord("x"): "EXCLUDE"}


def config_paths() -> tuple[Path, Path]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("task") != "dynamic_asl" or config.get("dynamic_labels") != ["J", "Z", "NONE"]:
        raise ValueError("Unexpected dynamic task/labels in config")
    if config.get("raw_external_root") != "data/raw/dynamic/external/signnteam_jz":
        raise ValueError("Unexpected raw external root in config")
    if config.get("metadata_dir") != "data/metadata/dynamic":
        raise ValueError("Unexpected metadata directory in config")
    return ROOT / config["raw_external_root"], ROOT / config["metadata_dir"]


def read_audit(path: Path, raw_root: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        needed = {"relative_path", "label", "filename", "fps", "technical_status"}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError("Technical audit CSV is missing required columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Technical audit CSV contains no videos")
    seen = set()
    for row in rows:
        relative = Path(row["relative_path"])
        resolved = (raw_root / relative).resolve()
        if (relative.is_absolute() or not resolved.is_relative_to(raw_root.resolve())
                or relative.suffix.lower() != ".avi"):
            raise ValueError(f"Unsafe/non-AVI path in audit: {relative}")
        if row["relative_path"] in seen or row["label"] not in {"J", "Z"}:
            raise ValueError(f"Duplicate path or invalid label in audit: {relative}")
        seen.add(row["relative_path"])
    return rows


def read_decisions(path: Path) -> tuple[dict[str, dict], list[str]]:
    if not path.exists():
        return {}, list(FIELDS)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if not set(FIELDS).issubset(columns):
            raise ValueError("Existing curation CSV is missing required columns")
        decisions = {}
        for row in reader:
            key = row["relative_path"]
            if not key or key in decisions:
                raise ValueError(f"Empty/duplicate reviewed path: {key}")
            decisions[key] = row
    return decisions, columns


def save_decisions(path: Path, decisions: dict[str, dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", suffix=".tmp",
                                         prefix="curation_", dir=path.parent,
                                         delete=False) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for key in sorted(decisions, key=str.casefold):
                writer.writerow(decisions[key])
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def open_clip(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return None, None, "OpenCV could not open video"
    ok, frame = capture.read()
    if not ok or frame is None or frame.size == 0:
        capture.release()
        return None, None, "No first frame could be decoded"
    return capture, frame, ""


def draw_preview(frame, row: dict, position: int, total: int, values: dict,
                 playing: bool, message: str):
    if frame is None:
        canvas = np.zeros((720, 1100, 3), dtype=np.uint8)
    else:
        canvas = frame.copy()
        if canvas.shape[1] > 1100:
            scale = 1100 / canvas.shape[1]
            canvas = cv2.resize(canvas, (1100, round(canvas.shape[0] * scale)))
        if canvas.shape[1] < 1100:
            canvas = cv2.copyMakeBorder(canvas, 0, 0, 0, 1100 - canvas.shape[1],
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if canvas.shape[0] < 340:
            canvas = cv2.copyMakeBorder(canvas, 0, 340 - canvas.shape[0], 0, 0,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 225), (0, 0, 0), -1)
    fps = row["fps"] or "unknown"
    lines = (
        f"{position}/{total}  LABEL: {row['label']}  FPS: {fps}  "
        f"{'PLAY' if playing else 'PAUSE/END'}",
        f"FILE: {row['filename']}",
        f"HANDSHAPE: {values['handshape_variant'] or '-'}  "
        f"TRAJECTORY: {values['trajectory_quality'] or '-'}  "
        f"DECISION: {values['decision'] or '-'}",
        f"NOTES: {'set' if values['notes'] else '-'}  TECH: {row['technical_status']}",
        "SPACE play/pause | R replay | 1-4 shape | G/I/U trajectory",
        "A GOOD | V REVIEW | X EXCLUDE | T note | N next | B back | Q quit",
        message,
    )
    for index, line in enumerate(lines):
        cv2.putText(canvas, line[:105], (12, 25 + index * 31), cv2.FONT_HERSHEY_SIMPLEX,
                    0.56, (230, 230, 230) if index != 6 else (0, 215, 255), 1,
                    cv2.LINE_AA)
    return canvas


def review_clip(row: dict, position: int, total: int, raw_root: Path,
                decisions: dict[str, dict], columns: list[str], output: Path) -> str:
    key = row["relative_path"]
    previous = decisions.get(key, {})
    if previous and previous.get("label") != row["label"]:
        raise ValueError(f"Existing decision label differs from audit: {key}")
    values = {name: previous.get(name, "") for name in FIELDS}
    values["relative_path"], values["label"] = key, row["label"]
    path = raw_root / key
    capture, frame, message = open_clip(path)
    playing = capture is not None
    variants = ("index_standard", "pinch", "other", "unclear") if row["label"] == "Z" else (
        "standard", "variant", "other", "unclear")
    try:
        fps = float(row["fps"])
        delay_ms = max(1, min(200, round(1000 / fps))) if fps > 0 else 40
    except (ValueError, TypeError):
        delay_ms = 40

    def persist_if_complete() -> bool:
        nonlocal message
        if not all(values[name] for name in ("handshape_variant", "trajectory_quality", "decision")):
            return False
        if values["decision"] == "GOOD" and row["technical_status"] != "OK":
            message = "Cannot mark technically failed video GOOD"
            return False
        values["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        decisions[key] = {**previous, **values}
        save_decisions(output, decisions, columns)
        message = "SAVED"
        return True

    try:
        while True:
            cv2.imshow(WINDOW_NAME, draw_preview(frame, row, position, total, values,
                                                 playing, message))
            key_code = cv2.waitKey(delay_ms if playing else 50) & 0xFF
            lower = ord(chr(key_code).lower()) if 0 <= key_code < 128 else key_code
            changed = False
            if key_code == 32:
                if capture is not None:
                    playing = not playing
            elif lower == ord("r"):
                if capture is not None:
                    capture.release()
                capture, frame, message = open_clip(path)
                playing = capture is not None
            elif ord("1") <= key_code <= ord("4"):
                values["handshape_variant"] = variants[key_code - ord("1")]
                changed = True
            elif lower in TRAJECTORY_KEYS:
                values["trajectory_quality"] = TRAJECTORY_KEYS[lower]
                changed = True
            elif lower in DECISION_KEYS:
                values["decision"] = DECISION_KEYS[lower]
                changed = True
            elif lower == ord("t"):
                values["notes"] = input("Notes (blank clears): ").strip()
                changed = True
            elif lower == ord("n"):
                if persist_if_complete():
                    return "next"
                message = "Choose shape + trajectory + decision before next"
            elif lower == ord("b"):
                if persist_if_complete():
                    return "previous"
                message = "Choose shape + trajectory + decision before back"
            elif lower == ord("q") or key_code == 27:
                if not persist_if_complete():
                    print(f"CURRENT_CLIP_NOT_SAVED={key}")
                return "quit"

            if changed:
                if not persist_if_complete():
                    message = "Choose shape + trajectory + decision"
            if playing and capture is not None:
                ok, next_frame = capture.read()
                if ok and next_frame is not None and next_frame.size:
                    frame = next_frame
                else:
                    playing = False
                    message = "END OF VIDEO: R to replay"
    finally:
        if capture is not None:
            capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-reviewed", action="store_true",
                        help="Revisit and edit existing decisions instead of skipping them")
    args = parser.parse_args()
    try:
        raw_root, metadata_dir = config_paths()
        if not raw_root.is_dir():
            raise FileNotFoundError(f"Raw J/Z directory is missing: {raw_root}")
        audit_path = metadata_dir / "technical_audit.csv"
        rows = read_audit(audit_path, raw_root)
        output = metadata_dir / "curation_decisions.csv"
        decisions, columns = read_decisions(output)
        queue = rows if args.include_reviewed else [row for row in rows
                                                   if row["relative_path"] not in decisions]
        print(f"TOTAL={len(rows)} REVIEW_QUEUE={len(queue)} ALREADY_REVIEWED={len(decisions)}")
        index = 0
        while index < len(queue):
            action = review_clip(queue[index], index + 1, len(queue), raw_root,
                                 decisions, columns, output)
            if action == "quit":
                break
            index = max(0, index - 1) if action == "previous" else index + 1
        print(f"CURATION_CSV={output} SAVED_DECISIONS={len(decisions)}")
        return 0
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
