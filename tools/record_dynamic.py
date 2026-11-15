"""Record one J, Z, or NONE gesture bout per MP4; no inference or training.

From the project root, for example:
    python tools/record_dynamic.py --label J --signer-id 01 --session-id 01 \
        --hand tayphai --view gocchinh

R starts a fresh bout, S saves it, C cancels it, and Q quits when idle.
Release and reform the gesture between recordings: distinct files alone do not
prove independent performances. Saved clips are raw camera frames (not flipped).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "dynamic_capture.json"
WINDOW_NAME = "Dynamic gesture recorder"


@dataclass
class Recording:
    writer: cv2.VideoWriter
    temporary_video: Path
    final_video: Path
    temporary_metadata: Path
    final_metadata: Path
    reservation: Path
    index: int
    width: int
    height: int
    fps: float
    created_at: str
    start_monotonic: float
    frame_count: int = 0


def load_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != 1
        or config.get("output_root") != "data/raw/dynamic"
        or config.get("labels") != ["J", "Z", "NONE"]
        or not config.get("hands")
        or not config.get("views")
        or not isinstance(config.get("camera_index"), int)
        or config["camera_index"] < 0
        or not isinstance(config.get("codec"), str)
        or len(config["codec"]) != 4
    ):
        raise ValueError("Invalid dynamic capture configuration")
    for key in ("requested_width", "requested_height", "fps"):
        if not isinstance(config.get(key), (int, float)) or config[key] <= 0:
            raise ValueError(f"Invalid dynamic capture setting: {key}")
    for key in ("hands", "views"):
        if any(not re.fullmatch(r"[a-z0-9]+", value) for value in config[key]):
            raise ValueError(f"Non-ASCII/unsafe token in {key}")
    return config


def ascii_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9]+", value):
        raise argparse.ArgumentTypeError("Use only ASCII letters and digits")
    return value


def parse_args(config: dict) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, choices=config["labels"])
    parser.add_argument("--signer-id", required=True, type=ascii_id, help="For example 01")
    parser.add_argument("--session-id", required=True, type=ascii_id, help="For example 01")
    parser.add_argument("--hand", required=True, choices=config["hands"])
    parser.add_argument("--view", required=True, choices=config["views"])
    return parser.parse_args()


def reserve_name(folder: Path, prefix: str) -> tuple[int, Path, Path, Path, Path, Path]:
    """Reserve an unused per-bout name without overwriting existing clips."""
    for index in range(1, 1_000_000):
        stem = f"{prefix}_lan{index:03d}"
        video = folder / f"{stem}.mp4"
        metadata = folder / f"{stem}.json"
        temp_video = folder / f".{stem}.part.mp4"
        temp_metadata = folder / f".{stem}.part.json"
        reservation = folder / f".{stem}.reserve"
        if any(path.exists() for path in (video, metadata, temp_video, temp_metadata, reservation)):
            continue
        try:
            with reservation.open("x", encoding="ascii") as handle:
                handle.write(datetime.now(timezone.utc).isoformat() + "\n")
        except FileExistsError:
            continue
        if any(path.exists() for path in (video, metadata, temp_video, temp_metadata)):
            reservation.unlink()
            continue
        return index, video, metadata, temp_video, temp_metadata, reservation
    raise RuntimeError("No available sequence index")


def begin_recording(frame, args: argparse.Namespace, config: dict) -> Recording:
    folder = ROOT / config["output_root"] / args.label
    folder.mkdir(parents=True, exist_ok=True)
    prefix = (
        f"{args.label}_nguoi{args.signer_id}_phien{args.session_id}_"
        f"{args.hand}_{args.view}"
    )
    index, video, metadata, temp_video, temp_metadata, reservation = reserve_name(folder, prefix)
    height, width = frame.shape[:2]
    fps = float(config["fps"])
    writer = cv2.VideoWriter(
        str(temp_video), cv2.VideoWriter_fourcc(*config["codec"]), fps, (width, height)
    )
    if not writer.isOpened():
        writer.release()
        temp_video.unlink(missing_ok=True)
        reservation.unlink(missing_ok=True)
        raise RuntimeError("Could not open MP4 writer; no clip was saved")
    return Recording(
        writer, temp_video, video, temp_metadata, metadata, reservation,
        index, width, height, fps, datetime.now(timezone.utc).isoformat(),
        time.monotonic(),
    )


def cancel_recording(recording: Recording) -> None:
    recording.writer.release()
    recording.temporary_video.unlink(missing_ok=True)
    recording.temporary_metadata.unlink(missing_ok=True)
    recording.reservation.unlink(missing_ok=True)


def save_recording(recording: Recording, args: argparse.Namespace) -> None:
    recording.writer.release()
    if recording.frame_count < 1 or not recording.temporary_video.is_file():
        raise RuntimeError(f"Empty recording; inspect {recording.temporary_video}")
    if recording.temporary_video.stat().st_size == 0:
        raise RuntimeError(f"Empty MP4 file; inspect {recording.temporary_video}")
    if recording.final_video.exists() or recording.final_metadata.exists():
        raise FileExistsError("Clip name became occupied; temporary clip retained")

    metadata = {
        "schema_version": 1,
        "label": args.label,
        "signer_id": args.signer_id,
        "session_id": args.session_id,
        "gesture_bout_id": recording.final_video.stem,
        "hand": args.hand,
        "view": args.view,
        "sequence_index": recording.index,
        "video_path": recording.final_video.relative_to(ROOT).as_posix(),
        "fps": recording.fps,
        "width": recording.width,
        "height": recording.height,
        "frame_count": recording.frame_count,
        "duration_seconds": round(recording.frame_count / recording.fps, 6),
        "created_at": recording.created_at,
    }
    with recording.temporary_metadata.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    # On Windows, rename refuses to replace an existing target. The reservation
    # also prevents another instance of this tool from claiming the same name.
    recording.temporary_video.rename(recording.final_video)
    recording.temporary_metadata.rename(recording.final_metadata)
    recording.reservation.unlink(missing_ok=True)
    print(f"SAVED {recording.final_video.relative_to(ROOT)}")


def run(args: argparse.Namespace, config: dict) -> int:
    camera = cv2.VideoCapture(config["camera_index"])
    if not camera.isOpened():
        camera.release()
        print("ERROR: Camera could not be opened", file=sys.stderr)
        return 2
    recording: Recording | None = None
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config["requested_width"])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config["requested_height"])
        camera.set(cv2.CAP_PROP_FPS, config["fps"])
        print("R: record | S: save | C: cancel | Q: quit when idle")
        print("Release and reform the gesture between clips; each clip is one bout.")
        while True:
            ok, frame = camera.read()
            if not ok:
                print("ERROR: Camera frame read failed", file=sys.stderr)
                return 3
            if recording is not None:
                if frame.shape[1] != recording.width or frame.shape[0] != recording.height:
                    raise RuntimeError("Camera resolution changed during recording")
                recording.writer.write(frame)  # Save the raw, unflipped frame.
                recording.frame_count += 1

            preview = frame.copy()
            if recording is not None:
                elapsed = time.monotonic() - recording.start_monotonic
                cv2.putText(preview, "REC", (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 255), 3)
                cv2.putText(preview, f"Frames: {recording.frame_count}  Time: {elapsed:.1f}s",
                            (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                instructions = "S: save  C: cancel"
            else:
                instructions = "R: record  Q: quit"
            cv2.putText(preview, f"{args.label} | {instructions}",
                        (20, preview.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if recording is None:
                if key in (ord("r"), ord("R")):
                    recording = begin_recording(frame, args, config)
                    print(f"REC index={recording.index:03d}")
                elif key in (ord("q"), ord("Q"), 27):
                    break
            elif key in (ord("s"), ord("S")):
                finished, recording = recording, None
                save_recording(finished, args)
            elif key in (ord("c"), ord("C")):
                cancel_recording(recording)
                recording = None
                print("CANCELLED")
            elif key in (ord("q"), ord("Q"), 27):
                print("Save with S or cancel with C before quitting")
    finally:
        try:
            if recording is not None:
                cancel_recording(recording)
        finally:
            camera.release()
            cv2.destroyAllWindows()
    return 0


def main() -> int:
    try:
        config = load_config()
        args = parse_args(config)
        return run(args, config)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
