"""User-run J-only preview of the 38 unresolved motion-review videos.

Thirty saved sequences use their exact extraction indices. Eight videos without
a sequence receive a clearly marked whole-clip preview-only contact sheet.
Neither mode runs MediaPipe or changes a dataset or review decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.feature_extraction.dynamic_sequence import sample_frame_indices  # noqa: E402
from tools.audit_dynamic_j_motion import EXTRACTION, REPORT, SOURCE_DIR  # noqa: E402
from tools.preview_dynamic_jz_quality_review import (  # noqa: E402
    h264_encoder, validate_indices, write_one,
)

EXPERIMENT = ROOT / "data/experiments/dynamic_jz_v1"
CLEAN_JSONL = EXPERIMENT / "processed/signnteam_jz_exclude_j40_v1.jsonl"
CLEAN_MANIFEST = EXPERIMENT / "metadata/signnteam_jz_exclude_j40_manifest_v1.json"
TECHNICAL = ROOT / "data/metadata/dynamic/technical_audit.csv"
COVERAGE = ROOT / "data/metadata/dynamic/landmark_coverage_audit_v1.csv"
OUTPUT_ROOT = EXPERIMENT / "review/j_remaining_38_v1"
EXPECTED_SOURCE_PREFIX = "data/raw/dynamic/external/signnteam_jz/SigNN Video Data/"
REFERENCE_MOTIF = {
    "J/100.avi": "ARM_SWEEP_U", "J/102.avi": "ARM_SWEEP_U",
    "J/104.avi": "ARM_SWEEP_U", "J/210.avi": "AIR_DRAWN_J",
}


def _j_key(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    if len(parts) < 2 or parts[-2] != "J" or not parts[-1].lower().endswith(".avi"):
        raise ValueError(f"Not a J video path: {path}")
    return "J/" + parts[-1]


def _j_rows(path: Path, field: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("label") != "J":
                continue
            key = _j_key(row[field])
            if key in result:
                raise ValueError(f"Duplicate J row in {path.name}: {key}")
            result[key] = row
    return result


def _clean_j_records() -> dict[str, dict]:
    result: dict[str, dict] = {}
    with CLEAN_JSONL.open("r", encoding="utf-8") as stream:
        for line in stream:
            # Do not parse or otherwise process Z feature records.
            if f'"{EXPECTED_SOURCE_PREFIX}J/' not in line[:180]:
                continue
            record = json.loads(line)
            if record.get("label") != "J":
                raise ValueError("J source row has non-J label in filtered JSONL")
            key = _j_key(record["source_path"])
            if key in result:
                raise ValueError(f"Duplicate J source in filtered JSONL: {key}")
            result[key] = record
    return result


def review_groups() -> tuple[list[list[dict]], dict[str, dict]]:
    inventory = _j_rows(REPORT, "video")
    review = [row for row in inventory.values()
              if row["classification"] == "NEEDS_VISUAL_REVIEW"]
    if len(inventory) != 316 or len(review) != 38:
        raise ValueError("Expected 316 J inventory entries and 38 review entries")
    manifest = json.loads(CLEAN_MANIFEST.read_text(encoding="utf-8"))
    if (manifest.get("status") != "FILTERED_J40_VERIFIED"
            or set(manifest.get("remaining_needs_visual_review", []))
            != {row["video"] for row in review}):
        raise ValueError("Filtered manifest disagrees with review inventory")
    records = _clean_j_records()
    with_sequence = [row for row in review if row["video"] in records]
    without_sequence = [row for row in review if row["video"] not in records]
    if len(with_sequence) != 30 or len(without_sequence) != 8:
        raise ValueError("Expected 30 saved J sequences and 8 without sequences")
    with_sequence.sort(key=lambda row: (float(row["distance_to_exclusion"]), row["video"]))
    without_sequence.sort(key=lambda row: int(Path(row["video"]).stem))
    groups = [with_sequence[:10], with_sequence[10:20], with_sequence[20:30],
              without_sequence]
    return groups, records


def preflight(batch: int) -> tuple[list[dict], Path]:
    groups, records = review_groups()
    if batch not in range(1, 5):
        raise ValueError("Batch must be 1-4")
    output = OUTPUT_ROOT / f"batch_{batch:02d}"
    if output.exists():
        raise FileExistsError(f"No overwrite: {output}")
    extraction = _j_rows(EXTRACTION, "source_path")
    technical = _j_rows(TECHNICAL, "relative_path")
    coverage = _j_rows(COVERAGE, "source_path")
    plan: list[dict] = []
    for row in groups[batch - 1]:
        relative = row["video"]
        source = EXPECTED_SOURCE_PREFIX + relative
        filename = Path(relative).name
        if relative != f"J/{filename}" or not filename.lower().endswith(".avi"):
            raise ValueError(f"Unsafe J source name: {relative}")
        path = (SOURCE_DIR / filename).resolve()
        if path.parent != SOURCE_DIR.resolve() or not path.is_file():
            raise ValueError(f"Missing or unsafe source: {relative}")
        e, t, c = extraction[relative], technical[relative], coverage[relative]
        if any(meta["label"] != "J" for meta in (e, t, c)):
            raise ValueError(f"Metadata label mismatch: {relative}")
        if any(meta["source_path"] != source for meta in (e, c)):
            raise ValueError(f"Metadata source mismatch: {relative}")
        count = int(t["decoded_frame_count"])
        if count < 16 or any(int(meta["decoded_frame_count"]) != count for meta in (e, c)):
            raise ValueError(f"Frame-count mismatch: {relative}")
        fps, width, height = float(t["fps"]), int(t["width"]), int(t["height"])
        if not math.isfinite(fps) or fps <= 0 or width < 2 or height < 2:
            raise ValueError(f"Invalid FPS/dimensions: {relative}")
        saved = records.get(relative)
        if saved is not None:
            if e["extraction_status"] != "SUCCESS" or saved["source_path"] != source:
                raise ValueError(f"Saved sequence/extraction mismatch: {relative}")
            indices = validate_indices(json.loads(e["selected_original_frame_indices"]), count)
            if indices != validate_indices(saved["selected_original_frame_indices"], count):
                raise ValueError(f"Saved frame indices mismatch: {relative}")
            selection_kind = "EXTRACTED_16"
            sheet_suffix = "extracted_16"
        else:
            if e["extraction_status"] != "REVIEW_REQUIRED" or json.loads(e["selected_original_frame_indices"]) != []:
                raise ValueError(f"Unsequenced-video status mismatch: {relative}")
            indices = validate_indices(sample_frame_indices(count, 16), count)
            selection_kind = "PREVIEW_ONLY_NOT_EXTRACTED"
            sheet_suffix = "preview_only_16"
        reasons = json.loads(c["invalid_reason_counts"] or "{}")
        no_hand_count = int(reasons.get("NO_HAND", 0))
        nearest = row["nearest_exclusion"]
        if saved is not None and nearest not in REFERENCE_MOTIF:
            raise ValueError(f"Unknown exclusion reference: {relative}")
        plan.append({"relative": relative, "label": "J", "source_path": source,
                     "path": path, "frame_count": count, "fps": fps,
                     "width": width, "height": height, "indices": indices,
                     "selection_kind": selection_kind, "no_hand_count": no_hand_count,
                     "nearest_confirmed_exclusion": nearest,
                     "motif_to_compare": REFERENCE_MOTIF.get(nearest, "BOTH_MOTIFS_VISUAL_ONLY"),
                     "coverage": row["coverage"], "reason": row["reason"],
                     "video_output": output / f"J_{Path(filename).stem}_full_h264.mp4",
                     "sheet_output": output / f"J_{Path(filename).stem}_{sheet_suffix}.png"})
    if len(plan) > 10 or len({item["relative"] for item in plan}) != len(plan):
        raise ValueError("Invalid batch size or duplicate video")
    return plan, output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--batch", type=int, required=True, help="1-3: saved sequences; 4: no sequence")
    parser.add_argument("--ffmpeg", help="FFmpeg executable with libx264 H.264 encoder")
    args = parser.parse_args(argv)
    try:
        plan, output = preflight(args.batch)
        encoder, encoder_status = h264_encoder(args.ffmpeg)
        print(json.dumps({"status": "PREFLIGHT_PASS" if encoder else "ENCODER_REQUIRED",
                          "batch": args.batch, "videos": [item["relative"] for item in plan],
                          "saved_sequence_count": sum(item["selection_kind"] == "EXTRACTED_16" for item in plan),
                          "preview_only_count": sum(item["selection_kind"] != "EXTRACTED_16" for item in plan),
                          "h264_encoder": encoder_status, "videos_opened": 0, "files_written": 0}))
        if args.check_only:
            return 0 if encoder else 2
        if encoder is None:
            raise ValueError(encoder_status)
        if input(f"Create J review preview batch {args.batch}? [y/N] ").strip().lower() != "y":
            print("CANCELLED: no video opened or file written")
            return 0
        output.mkdir(parents=True, exist_ok=False)
        for item in plan:
            write_one(item, encoder)
        with (output / "manual_review_v1.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["video", "sequence_status", "coverage",
                                                          "nearest_confirmed_exclusion", "motif_to_compare",
                                                          "reason", "visual_decision", "notes"])
            writer.writeheader()
            for item in plan:
                writer.writerow({"video": item["relative"], "sequence_status": item["selection_kind"],
                                 "coverage": item["coverage"],
                                 "nearest_confirmed_exclusion": item["nearest_confirmed_exclusion"],
                                 "motif_to_compare": item["motif_to_compare"],
                                 "reason": item["reason"], "visual_decision": "PENDING_USER_REVIEW",
                                 "notes": ""})
        with (output / "preview_manifest_v1.json").open("x", encoding="utf-8") as stream:
            json.dump({"purpose": "J_REMAINING_VISUAL_REVIEW_ONLY", "batch": args.batch,
                       "source": "filtered J40 JSONL and saved extraction audit; no Z/NONE processing",
                       "no_sequence_policy": "uniform full-clip PREVIEW_ONLY, never an extracted sequence",
                       "no_hand_positions": "UNKNOWN; coverage audit provides counts only",
                       "review_decision": "PENDING_USER_REVIEW",
                       "videos": [{"source_path": item["source_path"],
                                   "selection_kind": item["selection_kind"],
                                   "indices": item["indices"],
                                   "video_output": item["video_output"].relative_to(ROOT).as_posix(),
                                   "sheet_output": item["sheet_output"].relative_to(ROOT).as_posix()}
                                  for item in plan]}, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"status": "PREVIEW_CREATED", "batch": args.batch,
                          "videos": len(plan), "output": output.relative_to(ROOT).as_posix()}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
