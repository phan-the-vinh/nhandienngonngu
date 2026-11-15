"""Rank saved J sequences for visual review against user-decided J examples.

Only reads J source names, existing metadata, and J records of the (16,69)
JSONL. Scores are retrieval aids, never ASL-label or exclusion decisions.
--check-only writes nothing; --analyze writes temporary review reports only.
No video, MediaPipe, extraction, dataset filtering, or training is run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data/raw/dynamic/external/signnteam_jz/SigNN Video Data/J"
BASE = ROOT / "data/experiments/dynamic_jz_v1"
JSONL = BASE / "processed/signnteam_jz_baseline_b_v1.jsonl"
EXTRACTION = BASE / "metadata/signnteam_jz_extraction_audit_v1.csv"
QUALITY = BASE / "metadata/sequence_quality_audit_v1.csv"
COVERAGE = ROOT / "data/metadata/dynamic/landmark_coverage_audit_v1.csv"
OUT_DIR = BASE / "review/j_exclusion_search_v1"
REPORT = OUT_DIR / "j_motion_review_inventory_v1.csv"
SUMMARY = OUT_DIR / "j_motion_review_summary_v1.json"
CONFIRMED = ("J/100.avi", "J/102.avi", "J/104.avi", "J/210.avi")
REFERENCES = ("J/229.avi", "J/241.avi")


def _key(value: str) -> str:
    parts = value.replace("\\", "/").split("/")
    if len(parts) < 2 or parts[-2] != "J" or not parts[-1].lower().endswith(".avi"):
        raise ValueError(f"Not a J video path: {value}")
    return "J/" + parts[-1]


def _csv_index(path: Path, field: str) -> dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("label") != "J":
                continue
            key = _key(row[field])
            if key in rows:
                raise ValueError(f"Duplicate J metadata key in {path.name}: {key}")
            rows[key] = row
    return rows


def _signature(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frames.shape != (16, 69) or not np.isfinite(frames).all():
        raise ValueError("Invalid saved J sequence (16,69)")
    wrist, pinky = frames[:, 63:65], frames[:, 67:69]
    relative_tip = pinky - wrist
    shape = frames[:, :63].mean(axis=0)

    def path(points: np.ndarray) -> float:
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

    geometry = np.array((path(wrist), np.linalg.norm(wrist[-1]),
                         np.ptp(wrist[:, 0]), np.ptp(wrist[:, 1]),
                         path(pinky), np.linalg.norm(pinky[-1]),
                         path(relative_tip),
                         np.sqrt(np.mean((frames[:, :63] - shape) ** 2))), dtype=float)
    trajectory = np.concatenate((wrist.ravel(), pinky.ravel(), relative_tip.ravel()))
    return shape, trajectory, geometry


def load_j() -> tuple[dict[str, dict], dict[str, tuple], dict[str, dict], dict[str, dict]]:
    source_keys = {"J/" + path.name for path in SOURCE_DIR.glob("*.avi") if path.is_file()}
    extraction = _csv_index(EXTRACTION, "source_path")
    quality = _csv_index(QUALITY, "source_key")
    coverage = _csv_index(COVERAGE, "source_path")
    sequences = {}
    with JSONL.open("r", encoding="utf-8") as stream:
        for line in stream:
            if "SigNN Video Data/J/" not in line:
                continue
            row = json.loads(line)
            key = _key(row["source_path"])
            if key in sequences or row.get("label") != "J":
                raise ValueError(f"Duplicate or wrong-label J sequence: {key}")
            frames = np.asarray(row["features"], dtype=float)
            indices = row["selected_original_frame_indices"]
            if (row.get("extraction_status") != "SUCCESS" or row.get("sequence_length") != 16
                    or row.get("feature_dim") != 69 or not isinstance(indices, list)
                    or len(indices) != 16 or any(type(index) is not int for index in indices)
                    or any(right <= left for left, right in zip(indices, indices[1:]))):
                raise ValueError(f"Invalid J sequence metadata: {key}")
            if indices != json.loads(extraction[key]["selected_original_frame_indices"]):
                raise ValueError(f"Extraction/JSONL frame mismatch: {key}")
            sequences[key] = _signature(frames)
    successful = {key for key, row in extraction.items() if row["extraction_status"] == "SUCCESS"}
    failed = set(extraction) - successful
    if (len(source_keys) != 316 or len(extraction) != 316 or len(coverage) != 316
            or len(sequences) != 308 or source_keys != set(extraction)
            or source_keys != set(coverage) or successful != set(sequences)
            or set(quality) != successful or len(failed) != 8
            or not set(CONFIRMED + REFERENCES).issubset(sequences)):
        raise ValueError("J inventory or source/metadata/JSONL identity mismatch")
    for key in failed:
        if (extraction[key]["extraction_status"] != "REVIEW_REQUIRED"
                or extraction[key]["failure_reason"] != "INSUFFICIENT_VALID_LANDMARK_FRAMES"):
            raise ValueError(f"Unexpected failed-J reason: {key}")
    return extraction, sequences, quality, coverage


def _distances(sequences: dict[str, tuple]) -> dict[tuple[str, str], tuple[float, float, float, float]]:
    keys = list(sequences)
    shape = np.stack([sequences[key][0] for key in keys])
    trajectory = np.stack([sequences[key][1] for key in keys])
    geometry = np.stack([sequences[key][2] for key in keys])
    scales = (np.maximum(shape.std(axis=0), 0.12),
              np.maximum(trajectory.std(axis=0), 0.25),
              np.maximum(geometry.std(axis=0), 0.3))
    arrays = (shape, trajectory, geometry)
    positions = {key: index for index, key in enumerate(keys)}
    result = {}
    for key in keys:
        for prototype in CONFIRMED + REFERENCES:
            i, j = positions[key], positions[prototype]
            parts = tuple(float(np.sqrt(np.mean(((array[i] - array[j]) / scale) ** 2)))
                          for array, scale in zip(arrays, scales))
            result[key, prototype] = (parts[0] + parts[1] + 0.5 * parts[2], *parts)
    return result


def analyze() -> tuple[list[dict], dict]:
    extraction, sequences, quality, coverage = load_j()
    distances = _distances(sequences)
    rows = []
    for key in sorted(extraction, key=lambda name: int(Path(name).stem)):
        source = extraction[key]["source_path"]
        row = {"video": key, "source_path": source, "label": "J", "classification": "",
               "nearest_exclusion": "", "nearest_reference": "", "similarity_strength": "",
               "distance_to_exclusion": "", "distance_to_reference": "", "reference_margin": "",
               "shape_distance": "",
               "trajectory_distance": "", "geometry_distance": "", "coverage": "",
               "max_selected_frame_gap": "", "wrist_path": "", "pinky_relative_path": "",
               "shape_rms": "", "reason": "", "evidence_limit": ""}
        if key in CONFIRMED:
            row["classification"] = "EXCLUDE_CONFIRMED"
            row["reason"] = ("User-confirmed fixed handshape with U-shaped arm sweep"
                             if key != "J/210.avi" else
                             "User-confirmed fixed handshape with arm-drawn capital J")
        if key not in sequences:
            row["classification"] = "NEEDS_VISUAL_REVIEW"
            row["reason"] = "No saved sequence: fewer than 16 valid landmark frames"
            row["coverage"] = coverage[key]["landmark_coverage_ratio"]
            row["evidence_limit"] = "No (16,69) sequence; review source video only"
            rows.append(row)
            continue
        q = quality[key]
        signature = sequences[key][2]
        nearest_bad = min(CONFIRMED, key=lambda candidate: distances[key, candidate][0])
        bad = distances[key, nearest_bad]
        nearest_good = min(REFERENCES, key=lambda candidate: distances[key, candidate][0])
        good_distance = distances[key, nearest_good][0]
        margin = good_distance - bad[0]
        coverage_ratio = float(q["landmark_coverage_ratio"])
        row.update({"nearest_exclusion": nearest_bad, "nearest_reference": nearest_good,
                    "distance_to_exclusion": round(bad[0], 5),
                    "distance_to_reference": round(good_distance, 5),
                    "reference_margin": round(margin, 5),
                    "shape_distance": round(bad[1], 5),
                    "trajectory_distance": round(bad[2], 5),
                    "geometry_distance": round(bad[3], 5),
                    "coverage": q["landmark_coverage_ratio"],
                    "max_selected_frame_gap": q["max_selected_frame_gap"],
                    "wrist_path": round(float(signature[0]), 5),
                    "pinky_relative_path": round(float(signature[6]), 5),
                    "shape_rms": round(float(signature[7]), 5)})
        if key in CONFIRMED:
            pass
        elif (coverage_ratio >= 0.8 and nearest_bad == "J/210.avi"
              and bad[0] < 1.6 and margin > 1.0
              and bad[1] < 1.0 and bad[2] < 0.8):
            row["classification"] = "EXCLUDE_CANDIDATE"
            row["similarity_strength"] = "HIGH" if bad[0] < 1.0 and margin > 1.5 else "MEDIUM"
            row["reason"] = "J/210-like: concordant handshape, wrist/pinky trajectory and geometry; visually verify arm-drawn capital J"
        elif coverage_ratio < 0.8:
            row["classification"] = "NEEDS_VISUAL_REVIEW"
            row["reason"] = "Low landmark coverage; sampled trajectory may omit essential gesture phases"
        elif q["quality_decision"] == "REVIEW" or (nearest_bad in CONFIRMED[:3] and margin > 0.5):
            row["classification"] = "NEEDS_VISUAL_REVIEW"
            row["reason"] = "Quality flag or weak similarity to low-coverage U-arc example; inspect full video"
        else:
            row["classification"] = "NO_MATCH_FOUND"
            row["reason"] = "No strong match to confirmed exclusions in saved features; ASL correctness not established"
        row["evidence_limit"] = ("16 selected frames only; no unsampled motion, verified hand identity, "
                                  "signer/session ID, or semantic ASL decision")
        rows.append(row)
    from collections import Counter
    counts = dict(Counter(row["classification"] for row in rows))
    summary = {"status": "J_MOTION_RANKING_FOR_MANUAL_REVIEW",
               "source_j_videos": 316, "j_sequences": 308, "j_without_sequence": 8,
               "classification_counts": counts, "confirmed_exclusions": list(CONFIRMED),
               "user_confirmed_references": list(REFERENCES),
               "method": ("Three-part J-only normalized distance: mean 63-shape, 16-frame "
                          "wrist/pinky/pinky-minus-wrist XY trajectories, and eight geometry/motion "
                          "statistics. Candidate gates are retrieval heuristics, not exclusion rules."),
               "limitations": ["J/100, J/102 and J/104 have low landmark coverage; their "
                               "U-arc similarity is weak evidence", "No identity/signer metadata",
                               "Original video and unsampled motion require visual review",
                               "NO_MATCH_FOUND does not certify ASL correctness"],
               "candidate_videos": [row["video"] for row in rows
                                    if row["classification"] == "EXCLUDE_CANDIDATE"]}
    if sum(counts.values()) != 316 or counts.get("EXCLUDE_CONFIRMED") != 4:
        raise ValueError("Classification coverage mismatch")
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    args = parser.parse_args(argv)
    try:
        rows, summary = analyze()
        if args.check_only:
            print(json.dumps({**summary, "status": "PREFLIGHT_PASS",
                              "files_written": 0}, allow_nan=False))
            return 0
        if OUT_DIR.exists():
            raise FileExistsError(f"No overwrite: {OUT_DIR}")
        OUT_DIR.mkdir(parents=True, exist_ok=False)
        with REPORT.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with SUMMARY.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": summary["status"],
                          "classification_counts": summary["classification_counts"],
                          "report": REPORT.relative_to(ROOT).as_posix()}, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
