"""Apply explicit human WLASL OTHER_GESTURE review decisions to a new manifest.

The tool does not inspect videos, infer decisions, change review_manifest_v1.csv,
or create training data. An absent decision leaves the original row unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = ROOT / "data" / "experiments" / "dynamic_none_v1" / "review" / "wlasl_other_gesture"
MANIFEST_V1 = REVIEW_DIR / "review_manifest_v1.csv"
MANUAL_DECISIONS = REVIEW_DIR / "manual_decisions_v1.csv"
MANIFEST_V2 = REVIEW_DIR / "review_manifest_v2.csv"
SUMMARY = REVIEW_DIR / "review_decision_summary_v1.json"

EXPECTED_MANIFEST_COUNT = 121
REQUIRED_MANIFEST_FIELDS = {"video_id", "batch", "review_status", "review_reason"}
DECISION_FIELDS = ("batch", "video_id", "decision", "reason")
VALID_DECISIONS = ("ACCEPT_OTHER_GESTURE", "REJECT", "REVIEW")
OUTPUT_FIELDS = ("manual_decision", "manual_reason")


class ReviewInputError(ValueError):
    """A manifest or manual decision is malformed or inconsistent."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path: Path, required: set[str]) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise ReviewInputError(f"Missing input CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames
        if not header or len(header) != len(set(header)) or not required.issubset(header):
            raise ReviewInputError(f"Invalid or incomplete CSV header: {path}")
        rows = list(reader)
    for line_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ReviewInputError(f"Malformed CSV row at {path}:{line_number}")
    return header, rows


def validate_inputs(
    manifest_path: Path = MANIFEST_V1,
    decision_path: Path = MANUAL_DECISIONS,
) -> tuple[list[str], list[dict[str, str]], dict[str, dict[str, str]]]:
    header, manifest = read_table(manifest_path, REQUIRED_MANIFEST_FIELDS)
    if len(manifest) != EXPECTED_MANIFEST_COUNT:
        raise ReviewInputError(
            f"Expected {EXPECTED_MANIFEST_COUNT} manifest rows, found {len(manifest)}"
        )
    if any(field in header for field in OUTPUT_FIELDS):
        raise ReviewInputError("Input manifest already contains manual output fields")
    by_id: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(manifest, start=2):
        video_id = row["video_id"]
        if not video_id or not row["batch"] or not row["review_status"]:
            raise ReviewInputError(f"Missing required manifest value at line {line_number}")
        if video_id in by_id:
            raise ReviewInputError(f"Duplicate video_id in manifest: {video_id}")
        by_id[video_id] = row

    _, manual_rows = read_table(decision_path, set(DECISION_FIELDS))
    decisions: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(manual_rows, start=2):
        if any(not row[field].strip() for field in DECISION_FIELDS):
            raise ReviewInputError(f"Missing required manual decision value at line {line_number}")
        video_id = row["video_id"]
        if video_id not in by_id:
            raise ReviewInputError(f"Unknown video_id in manual decisions: {video_id}")
        if video_id in decisions:
            raise ReviewInputError(f"Duplicate video_id in manual decisions: {video_id}")
        if row["batch"] != by_id[video_id]["batch"]:
            raise ReviewInputError(f"Batch mismatch for video_id {video_id}")
        if row["decision"] not in VALID_DECISIONS:
            raise ReviewInputError(f"Unsupported decision for video_id {video_id}: {row['decision']}")
        decisions[video_id] = row
    return header, manifest, decisions


def build_result(
    manifest: list[dict[str, str]], decisions: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    output: list[dict[str, str]] = []
    for original in manifest:
        row = dict(original)
        manual = decisions.get(row["video_id"])
        row["manual_decision"] = manual["decision"] if manual else ""
        row["manual_reason"] = manual["reason"] if manual else ""
        if manual:
            # Literal human decision; no model/semantic inference or remapping.
            row["review_status"] = manual["decision"]
        output.append(row)
    decision_counts = Counter(row["decision"] for row in decisions.values())
    summary: dict[str, object] = {
        "manifest_rows": len(manifest),
        "manual_decision_rows": len(decisions),
        "decisions": {name: decision_counts[name] for name in VALID_DECISIONS},
        "without_manual_decision": len(manifest) - len(decisions),
        "result_review_status_counts": dict(sorted(Counter(row["review_status"] for row in output).items())),
        "decision_policy": "Copy literal manual decision to review_status; preserve original preview review_reason and all other fields.",
        "automatic_decisions": False,
        "media_processed": False,
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-only", action="store_true", help="Validate files and decisions without writing")
    modes.add_argument("--apply", action="store_true", help="Confirm and write versioned review outputs")
    args = parser.parse_args()
    try:
        header, manifest, decisions = validate_inputs(MANIFEST_V1, MANUAL_DECISIONS)
        output, summary = build_result(manifest, decisions)
        summary.update({
            "source_manifest": MANIFEST_V1.relative_to(ROOT).as_posix(),
            "source_manifest_sha256": sha256(MANIFEST_V1),
            "manual_decisions": MANUAL_DECISIONS.relative_to(ROOT).as_posix(),
            "manual_decisions_sha256": sha256(MANUAL_DECISIONS),
            "output_manifest": MANIFEST_V2.relative_to(ROOT).as_posix(),
            "output_summary": SUMMARY.relative_to(ROOT).as_posix(),
            "output_files_exist": MANIFEST_V2.exists() or SUMMARY.exists(),
            "apply_ready": bool(decisions) and not (MANIFEST_V2.exists() or SUMMARY.exists()),
        })
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.check_only:
            return 0
        if not decisions:
            raise ReviewInputError("No manual decision rows; refusing to create an unchanged v2")
        if MANIFEST_V2.exists() or SUMMARY.exists():
            raise ReviewInputError("Versioned output already exists; refusing to overwrite")
        if input(f"Apply {len(decisions)} manual decisions to a new v2 manifest? [y/N] ").strip().lower() != "y":
            print("Cancelled; no output written.")
            return 0
        with MANIFEST_V2.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*header, *OUTPUT_FIELDS])
            writer.writeheader()
            writer.writerows(output)
        summary["output_manifest_sha256"] = sha256(MANIFEST_V2)
        summary["apply_ready"] = False
        summary["status"] = "MANUAL_REVIEW_DECISIONS_APPLIED"
        with SUMMARY.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print("MANUAL_REVIEW_DECISIONS_APPLIED")
        return 0
    except (ReviewInputError, OSError, ValueError) as exc:
        print(f"REVIEW_INPUT_ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
