"""User-run, grouped full-video/16-frame previews of J exclusion candidates.

Uses saved extraction indices; never invokes MediaPipe or reruns extraction.
--check-only reads metadata without opening video or writing. --execute asks
before video decoding. Output is temporary and never changes source records.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_dynamic_j_motion import (  # noqa: E402
    BASE, EXTRACTION, OUT_DIR, REPORT, ROOT, SOURCE_DIR, SUMMARY,
)
from tools.preview_dynamic_jz_quality_review import (  # noqa: E402
    h264_encoder, validate_indices, write_one,
)


TECHNICAL = ROOT / "data/metadata/dynamic/technical_audit.csv"
COVERAGE = ROOT / "data/metadata/dynamic/landmark_coverage_audit_v1.csv"
BATCH_SIZE = 9


def _rows(path: Path, key: str) -> dict[str, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        values = list(csv.DictReader(stream))
    result = {}
    for row in values:
        name = row[key].replace("\\", "/")
        name = name[name.rfind("/J/") + 1:] if "/J/" in name else name
        if name in result:
            raise ValueError(f"Duplicate {key} in {path.name}: {name}")
        result[name] = row
    return result


def preflight(batch: int) -> tuple[list[dict], Path]:
    if batch < 1:
        raise ValueError("Batch must be positive")
    with REPORT.open("r", encoding="utf-8-sig", newline="") as stream:
        review = list(csv.DictReader(stream))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if len(review) != 316 or len({row["video"] for row in review}) != 316:
        raise ValueError("J inventory missing/duplicate rows")
    candidates = [row for row in review if row["classification"] == "EXCLUDE_CANDIDATE"]
    if set(row["video"] for row in candidates) != set(summary["candidate_videos"]):
        raise ValueError("Candidate list differs from saved summary")
    candidates.sort(key=lambda row: float(row["distance_to_exclusion"]))
    selected = candidates[(batch - 1) * BATCH_SIZE:batch * BATCH_SIZE]
    if not selected:
        raise ValueError(f"Empty candidate batch {batch}")
    output = OUT_DIR / f"preview_batch_{batch:02d}"
    if output.exists():
        raise FileExistsError(f"No overwrite: {output}")
    extraction = _rows(EXTRACTION, "source_path")
    technical = _rows(TECHNICAL, "relative_path")
    coverage = _rows(COVERAGE, "source_path")
    plan = []
    for row in selected:
        relative = row["video"]
        if not relative.startswith("J/") or Path(relative).name != relative[2:]:
            raise ValueError(f"Unsafe candidate video: {relative}")
        path = (SOURCE_DIR / relative[2:]).resolve()
        if path.parent != SOURCE_DIR.resolve() or not path.is_file():
            raise ValueError(f"Missing/unsafe source video: {relative}")
        e, t, c = extraction[relative], technical[relative], coverage[relative]
        if any(meta["label"] != "J" for meta in (e, t, c)) or e["extraction_status"] != "SUCCESS":
            raise ValueError(f"Source label/extraction mismatch: {relative}")
        count = int(t["decoded_frame_count"])
        if any(int(meta["decoded_frame_count"]) != count for meta in (e, c)):
            raise ValueError(f"Frame count mismatch: {relative}")
        indices = validate_indices(json.loads(e["selected_original_frame_indices"]), count)
        no_hand = int(json.loads(c["invalid_reason_counts"] or "{}").get("NO_HAND", 0))
        name = Path(relative).stem
        plan.append({"relative": relative, "label": "J", "source_path": e["source_path"],
                     "path": path, "frame_count": count, "fps": float(t["fps"]),
                     "width": int(t["width"]), "height": int(t["height"]),
                     "indices": indices, "no_hand_count": no_hand,
                     "video_output": output / f"J_{name}_full_h264.mp4",
                     "sheet_output": output / f"J_{name}_extracted_16.png",
                     "similarity_strength": row["similarity_strength"]})
    return plan, output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--batch", type=int, required=True, help="1-4; nine ranked videos per batch")
    parser.add_argument("--ffmpeg", help="FFmpeg executable with libx264 H.264 encoder")
    args = parser.parse_args(argv)
    try:
        plan, output = preflight(args.batch)
        encoder, encoder_status = h264_encoder(args.ffmpeg)
        print(json.dumps({"status": "PREFLIGHT_PASS" if encoder else "ENCODER_REQUIRED",
                          "batch": args.batch, "videos": [item["relative"] for item in plan],
                          "h264_encoder": encoder_status, "videos_opened": 0,
                          "files_written": 0}))
        if args.check_only:
            return 0 if encoder else 2
        if encoder is None:
            raise ValueError(encoder_status)
        if input(f"Create J candidate preview batch {args.batch}? [y/N] ").strip().lower() != "y":
            print("CANCELLED: no video opened or file written")
            return 0
        output.mkdir(parents=True, exist_ok=False)
        for item in plan:
            write_one(item, encoder)
        manifest = output / "preview_manifest_v1.json"
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump({"purpose": "J_EXCLUSION_CANDIDATE_VISUAL_REVIEW_ONLY",
                       "batch": args.batch, "review_decision": "PENDING_USER_REVIEW",
                       "saved_16_frame_indices_not_recomputed": True,
                       "no_hand_intervals": "UNAVAILABLE_IN_SAVED_METADATA",
                       "videos": [{"source_path": item["source_path"],
                                   "selected_original_frame_indices": item["indices"],
                                   "video_output": item["video_output"].relative_to(ROOT).as_posix(),
                                   "sheet_output": item["sheet_output"].relative_to(ROOT).as_posix()}
                                  for item in plan]}, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"status": "PREVIEW_COMPLETE", "batch": args.batch,
                          "count": len(plan), "output": output.relative_to(ROOT).as_posix()}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
