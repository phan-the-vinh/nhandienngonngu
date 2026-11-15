"""User-run, exact-manifest J exclusion from an existing J/Z JSONL.

--check-only verifies the 40 user-confirmed sources and predicts counts without
writing. --execute asks for confirmation and creates a *new* JSONL and result
manifest; original JSONL, video files, extraction reports, and Z lines remain
untouched. This is dataset preparation: the user, not Codex, runs --execute.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/experiments/dynamic_jz_v1"
RAW_J = ROOT / "data/raw/dynamic/external/signnteam_jz/SigNN Video Data/J"
ORIGINAL = BASE / "processed/signnteam_jz_baseline_b_v1.jsonl"
INVENTORY = BASE / "review/j_exclusion_search_v1/j_motion_review_inventory_v1.csv"
EXCLUSIONS = BASE / "review/j_exclusion_search_v1/exclude_confirmed_manifest_v1.csv"
BATCH_ROOT = BASE / "review/j_exclusion_search_v1"
EXTRACTION = BASE / "metadata/signnteam_jz_extraction_audit_v1.csv"
OUTPUT = BASE / "processed/signnteam_jz_exclude_j40_v1.jsonl"
OUTPUT_MANIFEST = BASE / "metadata/signnteam_jz_exclude_j40_manifest_v1.json"
SOURCE_PREFIX = "data/raw/dynamic/external/signnteam_jz/SigNN Video Data/"
INITIAL = {"J/100.avi", "J/102.avi", "J/104.avi", "J/210.avi"}
EXPECTED_BEFORE = {"J": 308, "Z": 375}
EXPECTED_AFTER = {"J": 268, "Z": 375}


def _read_csv(path: Path, fields: set[str]) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not fields.issubset(reader.fieldnames):
            raise ValueError(f"CSV schema mismatch: {path}")
        return list(reader)


def _video_key(source: str) -> str:
    source = source.replace("\\", "/")
    if not source.startswith(SOURCE_PREFIX):
        raise ValueError(f"Unexpected source path: {source}")
    suffix = source[len(SOURCE_PREFIX):]
    parts = suffix.split("/")
    if (len(parts) != 2 or parts[0] not in ("J", "Z") or not parts[1].lower().endswith(".avi")
            or parts[1] in ("", ".avi", "..avi")):
        raise ValueError(f"Invalid video key: {source}")
    return suffix


def validate_manifest_rows(rows: list[dict], batch_sets: dict[str, set[str]],
                           source_keys: set[str], inventory: dict[str, dict]) -> dict[str, dict]:
    if len(rows) != 40:
        raise ValueError(f"Expected 40 exclusions; got {len(rows)}")
    confirmed = {}
    counts = Counter()
    for row in rows:
        key = row["video"]
        batch = row["batch"]
        if key in confirmed:
            raise ValueError(f"Duplicate exclusion: {key}")
        if (_video_key(row["source_path"]) != key or row["label"] != "J"
                or row["decision"] != "EXCLUDE_CONFIRMED" or not row["reason"]
                or key not in source_keys or key not in inventory):
            raise ValueError(f"Exclusion source/label/decision mismatch: {key}")
        if batch == "INITIAL":
            if key not in INITIAL or inventory[key]["classification"] != "EXCLUDE_CONFIRMED":
                raise ValueError(f"Initial decision mismatch: {key}")
        elif batch in batch_sets:
            if key not in batch_sets[batch] or inventory[key]["classification"] != "EXCLUDE_CANDIDATE":
                raise ValueError(f"Batch membership or inventory mismatch: {key}")
        else:
            raise ValueError(f"Unknown batch: {batch}")
        counts[batch] += 1
        confirmed[key] = row
    if (counts != {"INITIAL": 4, "BATCH_01": 9, "BATCH_02": 9,
                   "BATCH_03": 9, "BATCH_04": 9}
            or set(confirmed) != INITIAL | set().union(*batch_sets.values())):
        raise ValueError("Manifest does not match the four complete preview batches")
    return confirmed


def _batch_sets() -> dict[str, set[str]]:
    result = {}
    for number in range(1, 5):
        name = f"BATCH_{number:02d}"
        path = BATCH_ROOT / f"preview_batch_{number:02d}/preview_manifest_v1.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("batch") != number or len(manifest.get("videos", [])) != 9:
            raise ValueError(f"Preview batch metadata mismatch: {name}")
        keys = [_video_key(video["source_path"]) for video in manifest["videos"]]
        if len(set(keys)) != 9 or any(not key.startswith("J/") for key in keys):
            raise ValueError(f"Duplicate/non-J preview entries: {name}")
        result[name] = set(keys)
    if len(set().union(*result.values())) != 36:
        raise ValueError("Preview batches overlap or omit J candidates")
    return result


def _scan_jsonl(path: Path) -> dict:
    counts = Counter()
    sources = set()
    z_hasher = hashlib.sha256()
    full_hasher = hashlib.sha256()
    j_records = {}
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line: {line_number}")
            full_hasher.update(line)
            row = json.loads(line)
            source = row["source_path"]
            key = _video_key(source)
            label = row["label"]
            if label not in ("J", "Z") or label != key[0] or source in sources:
                raise ValueError(f"Duplicate or mislabeled JSONL source on line {line_number}")
            sources.add(source)
            counts[label] += 1
            if label == "J":
                j_records[key] = row
            else:
                z_hasher.update(line)
    return {"counts": dict(counts), "source_paths": sources, "j_records": j_records,
            "sha256": full_hasher.hexdigest(), "z_lines_sha256": z_hasher.hexdigest()}


def preflight() -> dict:
    if OUTPUT.exists() or OUTPUT_MANIFEST.exists():
        raise FileExistsError("Filtered output or its manifest already exists; no overwrite")
    source_keys = {f"J/{path.name}" for path in RAW_J.glob("*.avi") if path.is_file()}
    if len(source_keys) != 316:
        raise ValueError(f"Expected 316 J source videos; got {len(source_keys)}")
    inventory_rows = _read_csv(INVENTORY, {"video", "classification", "source_path"})
    if len(inventory_rows) != 316 or len({row["video"] for row in inventory_rows}) != 316:
        raise ValueError("J inventory is incomplete/duplicated")
    inventory = {row["video"]: row for row in inventory_rows}
    if set(inventory) != source_keys:
        raise ValueError("J inventory/source set mismatch")
    remaining_review = {key for key, row in inventory.items()
                        if row["classification"] == "NEEDS_VISUAL_REVIEW"}
    if len(remaining_review) != 38:
        raise ValueError("Expected 38 J videos pending visual review")
    batch_sets = _batch_sets()
    rows = _read_csv(EXCLUSIONS, {"video", "label", "batch", "decision", "reason", "source_path"})
    confirmed = validate_manifest_rows(rows, batch_sets, source_keys, inventory)
    if remaining_review & set(confirmed):
        raise ValueError("Unreviewed J video was included in confirmed exclusions")
    extraction_rows = _read_csv(EXTRACTION, {"source_path", "label", "extraction_status",
                                             "selected_original_frame_indices"})
    extraction = {}
    for row in extraction_rows:
        if row["label"] != "J":
            continue
        key = _video_key(row["source_path"])
        if key in extraction:
            raise ValueError(f"Duplicate extraction row: {key}")
        extraction[key] = row
    if len(extraction) != 316 or set(extraction) != source_keys:
        raise ValueError("Extraction/J source set mismatch")
    original = _scan_jsonl(ORIGINAL)
    if original["counts"] != EXPECTED_BEFORE or len(original["source_paths"]) != 683:
        raise ValueError("Original JSONL count/uniqueness mismatch")
    if set(original["j_records"]) != {key for key, row in extraction.items()
                                      if row["extraction_status"] == "SUCCESS"}:
        raise ValueError("JSONL J/extraction success set mismatch")
    for key in confirmed:
        if key not in original["j_records"] or extraction[key]["extraction_status"] != "SUCCESS":
            raise ValueError(f"Confirmed source lacks one successful J record: {key}")
        saved_indices = original["j_records"][key]["selected_original_frame_indices"]
        if saved_indices != json.loads(extraction[key]["selected_original_frame_indices"]):
            raise ValueError(f"JSONL/extraction frame-index mismatch: {key}")
    missing_sequence = sorted(source_keys - set(original["j_records"]),
                              key=lambda key: int(Path(key).stem))
    if len(missing_sequence) != 8 or not set(missing_sequence).issubset(remaining_review):
        raise ValueError("Missing-sequence J review set mismatch")
    return {"confirmed": confirmed, "original": original,
            "exclusion_manifest_sha256": hashlib.sha256(EXCLUSIONS.read_bytes()).hexdigest(),
            "remaining_review": sorted(remaining_review, key=lambda key: int(Path(key).stem)),
            "missing_sequence": missing_sequence,
            "expected_after": EXPECTED_AFTER.copy()}


def line_action(line: bytes, excluded_sources: set[str]) -> tuple[bool, str, str]:
    """Return (keep, label, source), preserving retained bytes verbatim."""
    row = json.loads(line)
    label, source = row["label"], row["source_path"]
    if label not in ("J", "Z") or _video_key(source)[0] != label:
        raise ValueError("Unexpected label/source in JSONL line")
    if source in excluded_sources:
        if label != "J":
            raise ValueError("Attempted to exclude a non-J record")
        return False, label, source
    return True, label, source


def create_filtered(plan: dict) -> dict:
    excluded_sources = {row["source_path"] for row in plan["confirmed"].values()}
    before = Counter()
    removed = Counter()
    source_hasher = hashlib.sha256()
    with ORIGINAL.open("rb") as source, OUTPUT.open("xb") as target:
        for line in source:
            source_hasher.update(line)
            keep, label, source_path = line_action(line, excluded_sources)
            before[label] += 1
            if not keep:
                removed[source_path] += 1
            else:
                target.write(line)  # Preserve all retained records, especially Z, byte-for-byte.
    if (source_hasher.hexdigest() != plan["original"]["sha256"]
            or hashlib.sha256(EXCLUSIONS.read_bytes()).hexdigest() != plan["exclusion_manifest_sha256"]
            or dict(before) != EXPECTED_BEFORE or set(removed) != excluded_sources
            or any(count != 1 for count in removed.values())):
        raise ValueError("Exclusion count/set mismatch; output is not valid")
    verified = _scan_jsonl(OUTPUT)
    if (verified["counts"] != EXPECTED_AFTER
            or verified["z_lines_sha256"] != plan["original"]["z_lines_sha256"]
            or any(source in verified["source_paths"] for source in excluded_sources)):
        raise ValueError("Filtered JSONL verification failed; output is not valid")
    result = {"status": "FILTERED_J40_VERIFIED", "source_jsonl": ORIGINAL.relative_to(ROOT).as_posix(),
              "source_sha256": plan["original"]["sha256"],
              "exclusion_manifest": EXCLUSIONS.relative_to(ROOT).as_posix(),
              "exclusion_manifest_sha256": plan["exclusion_manifest_sha256"],
              "filtered_jsonl": OUTPUT.relative_to(ROOT).as_posix(),
              "filtered_sha256": verified["sha256"], "before_counts": EXPECTED_BEFORE,
              "after_counts": EXPECTED_AFTER, "excluded_j_records": len(removed),
              "z_lines_unchanged_sha256": verified["z_lines_sha256"],
              "remaining_needs_visual_review": plan["remaining_review"],
              "original_video_or_jsonl_modified": False,
              "note": "Temporary experiment dataset; not an official split or training approval"}
    with OUTPUT_MANIFEST.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = preflight()
        summary = {"status": "PREFLIGHT_PASS", "source_j_videos": 316,
                   "source_j_sequences": EXPECTED_BEFORE["J"],
                   "confirmed_exclusion_records": len(plan["confirmed"]),
                   "expected_after": plan["expected_after"],
                   "remaining_needs_visual_review": len(plan["remaining_review"]),
                   "missing_sequence_review": len(plan["missing_sequence"]),
                   "output": OUTPUT.relative_to(ROOT).as_posix(), "files_written": 0}
        print(json.dumps(summary, allow_nan=False))
        if args.check_only:
            return 0
        if input("Create new filtered J/Z JSONL excluding exactly 40 confirmed J records? [y/N] ").strip().lower() != "y":
            print("CANCELLED: no dataset created")
            return 0
        plan = preflight()  # Revalidate all manifests and the source immediately before writing.
        result = create_filtered(plan)
        print(json.dumps({"status": result["status"], "after_counts": result["after_counts"],
                          "filtered_sha256": result["filtered_sha256"],
                          "result_manifest": OUTPUT_MANIFEST.relative_to(ROOT).as_posix()}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
