"""Audit saved J/Z (16,69) sequences; never decode video or invoke MediaPipe.

Statistical flags request review only. They do not prove hand identity, label
correctness, or license permission.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/experiments/dynamic_jz_v1"
JSONL = BASE / "processed/signnteam_jz_baseline_b_v1.jsonl"
EXTRACTION = BASE / "metadata/signnteam_jz_extraction_audit_v1.csv"
COVERAGE = ROOT / "data/metadata/dynamic/landmark_coverage_audit_v1.csv"
TECHNICAL = ROOT / "data/metadata/dynamic/technical_audit.csv"
REPORT_CSV = BASE / "metadata/sequence_quality_audit_v1.csv"
REPORT_JSON = BASE / "metadata/sequence_quality_summary_v1.json"
EXPECTED = {"J": 308, "Z": 375}
PAIRS = {0: (63, 64), 8: (65, 66), 20: (67, 68)}


def source_key(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    if len(parts) < 2 or parts[-2] not in EXPECTED or not parts[-1].lower().endswith(".avi"):
        raise ValueError("INVALID_SOURCE_PATH")
    return "/".join(parts[-2:])


def read_csv(path: Path, required: set[str]) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV_SCHEMA_MISMATCH: {path}")
        return list(reader)


def preflight() -> dict:
    for path in (JSONL, EXTRACTION, COVERAGE, TECHNICAL):
        if not path.is_file():
            raise FileNotFoundError(path)
    extraction = read_csv(EXTRACTION, {"source_path", "label", "extraction_status"})
    coverage = read_csv(COVERAGE, {"source_path", "label", "valid_landmark_frames"})
    technical = read_csv(TECHNICAL, {"relative_path", "label", "decoded_frame_count"})
    with JSONL.open(encoding="utf-8") as handle:
        lines = sum(bool(line.strip()) for line in handle)
    return {
        "jsonl_lines": lines,
        "extraction_rows": len(extraction),
        "extraction_success": sum(r["extraction_status"] == "SUCCESS" for r in extraction),
        "coverage_rows": len(coverage),
        "technical_rows": len(technical),
        "existing_outputs": [str(p) for p in (REPORT_CSV, REPORT_JSON) if p.exists()],
    }


def unique_map(rows: list[dict], field: str) -> tuple[dict, list[str]]:
    mapped, duplicates = {}, []
    for row in rows:
        key = source_key(row[field])
        if key in mapped:
            duplicates.append(key)
        else:
            mapped[key] = row
    return mapped, duplicates


def validate(row: dict) -> tuple[str, list[int], list[list[float]], list[str]]:
    errors = []
    try:
        key = source_key(str(row.get("source_path", "")))
    except ValueError:
        key = ""
        errors.append("INVALID_SOURCE_PATH")
    label = row.get("label")
    if not isinstance(label, str) or label not in EXPECTED or (key and not key.startswith(label + "/")):
        errors.append("LABEL_SOURCE_MISMATCH")
    if row.get("z_variant") != ("UNKNOWN" if label == "Z" else "NOT_APPLICABLE"):
        errors.append("Z_VARIANT_MISMATCH")
    if row.get("extraction_status") != "SUCCESS":
        errors.append("NON_SUCCESS_RECORD")
    if row.get("sequence_length") != 16 or row.get("feature_dim") != 69:
        errors.append("SCHEMA_METADATA_MISMATCH")
    indices = row.get("selected_original_frame_indices")
    if (not isinstance(indices, list) or len(indices) != 16
            or any(type(i) is not int or i < 0 for i in indices)
            or any(b <= a for a, b in zip(indices, indices[1:]))):
        errors.append("INVALID_FRAME_INDICES")
        indices = []
    try:
        decoded, valid = int(row["decoded_frame_count"]), int(row["valid_landmark_frames"])
        if decoded <= 0 or not 16 <= valid <= decoded or (indices and indices[-1] >= decoded):
            errors.append("INVALID_FRAME_COUNTS_OR_INDEX_RANGE")
    except (KeyError, TypeError, ValueError):
        errors.append("INVALID_FRAME_COUNTS_OR_INDEX_RANGE")
    scale = row.get("hand_scale_px")
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        errors.append("INVALID_HAND_SCALE")
    features = row.get("features")
    if (not isinstance(features, list) or len(features) != 16
            or any(not isinstance(frame, list) or len(frame) != 69 for frame in features)):
        errors.append("INVALID_FEATURE_SHAPE")
        features = []
    elif any(type(v) not in (int, float) or not math.isfinite(v)
             for frame in features for v in frame):
        errors.append("NONFINITE_OR_NONNUMERIC_FEATURE")
        features = []
    if features and any(abs(features[0][c]) > 1e-8 for c in range(63, 69)):
        errors.append("TRAJECTORY_ORIGIN_NONZERO")
    if features and any(abs(frame[c]) > 1e-8 for frame in features for c in range(3)):
        errors.append("WRIST_SHAPE_NONZERO")
    return key, indices, features, errors


def distance(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def metrics(features: list[list[float]], indices: list[int], label: str) -> dict[str, float]:
    means = [statistics.fmean(frame[c] for frame in features) for c in range(63)]
    result = {"shape_rms": math.sqrt(statistics.fmean(
        sum((frame[c] - means[c]) ** 2 for c in range(63)) for frame in features))}
    for landmark in (8, 20):
        start = landmark * 3
        center = [statistics.fmean(frame[start + c] for frame in features) for c in range(3)]
        result[f"shape_rms_{landmark}"] = math.sqrt(statistics.fmean(
            sum((frame[start + c] - center[c]) ** 2 for c in range(3)) for frame in features))
    gaps = [b - a for a, b in zip(indices, indices[1:])]
    result["max_selected_frame_gap"] = max(gaps)
    result["median_selected_frame_gap"] = statistics.median(gaps)
    shape_steps = [distance(a[:63], b[:63]) for a, b in zip(features, features[1:])]
    result["max_shape_step_per_source_frame"] = max(
        step / gap for step, gap in zip(shape_steps, gaps))
    result["max_shape_step"] = max(shape_steps)
    for landmark, (cx, cy) in PAIRS.items():
        points = [(frame[cx], frame[cy]) for frame in features]
        steps = [distance(a, b) for a, b in zip(points, points[1:])]
        prefix = f"landmark_{landmark}"
        result[f"{prefix}_path"] = sum(steps)
        result[f"{prefix}_net"] = distance(points[0], points[-1])
        result[f"{prefix}_max_step"] = max(steps)
        result[f"{prefix}_max_step_per_source_frame"] = max(
            step / gap for step, gap in zip(steps, gaps))
    tip = 20 if label == "J" else 8
    result["label_tip_path"] = result[f"landmark_{tip}_path"]
    result["label_tip_max_step_rate"] = result[f"landmark_{tip}_max_step_per_source_frame"]
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def audit() -> tuple[list[dict], dict]:
    extraction_rows = read_csv(EXTRACTION, {"source_path", "label", "extraction_status",
                                            "selected_original_frame_indices"})
    coverage_rows = read_csv(COVERAGE, {"source_path", "label", "decoded_frame_count",
                                        "valid_landmark_frames"})
    technical_rows = read_csv(TECHNICAL, {"relative_path", "label", "decoded_frame_count"})
    extraction, extraction_dupes = unique_map(extraction_rows, "source_path")
    coverage, coverage_dupes = unique_map(coverage_rows, "source_path")
    technical, technical_dupes = unique_map(technical_rows, "relative_path")
    rows, parse_errors = [], []
    source_counts = Counter()
    feature_sources = defaultdict(list)
    with JSONL.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("NOT_OBJECT")
            except (json.JSONDecodeError, ValueError) as exc:
                parse_errors.append(f"line {line_number}: {exc}")
                continue
            key, indices, features, errors = validate(item)
            source_counts[key] += 1
            label = item.get("label")
            if not isinstance(label, str):
                label = "INVALID_LABEL"
            row = {
                "line_number": line_number, "source_path": item.get("source_path", ""),
                "source_key": key, "label": label, "schema_errors": errors,
                "review_reasons": [], "identity_status": "UNVERIFIED",
                "decoded_frame_count": item.get("decoded_frame_count", ""),
                "valid_landmark_frames": item.get("valid_landmark_frames", ""),
            }
            e, c, t = extraction.get(key), coverage.get(key), technical.get(key)
            if not e or e.get("extraction_status") != "SUCCESS":
                errors.append("EXTRACTION_AUDIT_MISSING_OR_NOT_SUCCESS")
            elif indices:
                try:
                    if json.loads(e["selected_original_frame_indices"]) != indices:
                        errors.append("EXTRACTION_AUDIT_INDICES_MISMATCH")
                except (KeyError, TypeError, ValueError):
                    errors.append("EXTRACTION_AUDIT_INDICES_INVALID")
            if not c:
                errors.append("COVERAGE_AUDIT_MISSING")
            if not t:
                errors.append("TECHNICAL_AUDIT_MISSING")
            for meta in (e, c, t):
                if meta and (meta.get("label") != label
                             or str(meta.get("decoded_frame_count")) != str(item.get("decoded_frame_count"))):
                    errors.append("METADATA_LABEL_OR_FRAME_COUNT_MISMATCH")
            for meta in (e, c):
                if meta and str(meta.get("valid_landmark_frames")) != str(item.get("valid_landmark_frames")):
                    errors.append("METADATA_VALID_COUNT_MISMATCH")
            if c:
                row["landmark_coverage_ratio"] = c.get("landmark_coverage_ratio", "")
                row["invalid_reason_counts"] = c.get("invalid_reason_counts", "")
            if t:
                row.update({field: t.get(field, "") for field in ("fps", "width", "height")})
            if features and indices and label in EXPECTED:
                row.update(metrics(features, indices, label))
                digest = hashlib.sha256(json.dumps(features, separators=(",", ":"),
                                                   allow_nan=False).encode("utf-8")).hexdigest()
                row["feature_sha256"] = digest
                feature_sources[digest].append(key)
            rows.append(row)
    duplicates = {digest: sources for digest, sources in feature_sources.items() if len(sources) > 1}
    tail_spec = {"shape_rms": 0.01, "label_tip_path": 0.01,
                 "label_tip_max_step_rate": 0.99,
                 "max_shape_step_per_source_frame": 0.99}
    thresholds = {}
    for label in EXPECTED:
        subset = [r for r in rows if r.get("label") == label and "shape_rms" in r]
        thresholds[label] = {name: percentile([r[name] for r in subset], q)
                             for name, q in tail_spec.items()} if subset else {}
    for row in rows:
        reasons = list(row["schema_errors"])
        if source_counts[row["source_key"]] > 1:
            reasons.append("REVIEW_DUPLICATE_SOURCE")
        if row.get("feature_sha256") in duplicates:
            reasons.append("REVIEW_EXACT_DUPLICATE_FEATURES")
        label = row["label"]
        if "shape_rms" in row and label in EXPECTED:
            bounds = thresholds[label]
            if row["shape_rms"] < bounds["shape_rms"]:
                reasons.append("REVIEW_LOW_SHAPE_VARIATION_TAIL")
            if row["label_tip_path"] < bounds["label_tip_path"]:
                reasons.append("REVIEW_LOW_TIP_MOTION_TAIL")
            if row["label_tip_max_step_rate"] > bounds["label_tip_max_step_rate"]:
                reasons.append("REVIEW_HIGH_TIP_STEP_RATE_TAIL")
            if row["max_shape_step_per_source_frame"] > bounds["max_shape_step_per_source_frame"]:
                reasons.append("REVIEW_HIGH_SHAPE_STEP_RATE_TAIL")
            if row["max_selected_frame_gap"] > 2.5 * row["median_selected_frame_gap"]:
                reasons.append("REVIEW_LARGE_SELECTED_FRAME_GAP")
        try:
            if int(row["valid_landmark_frames"]) < 0.8 * int(row["decoded_frame_count"]):
                reasons.append("REVIEW_LOW_DETECTION_COVERAGE")
        except (TypeError, ValueError):
            pass
        row["review_reasons"] = sorted(set(reasons))
        row["quality_decision"] = "REVIEW" if reasons else "NO_AUTOMATED_FLAG_NOT_APPROVED"
    counts = Counter(row["label"] for row in rows)
    success_keys = {key for key, value in extraction.items()
                    if value.get("extraction_status") == "SUCCESS"}
    observed_keys = set(source_counts)
    issues = []
    if len(rows) != 683 or any(counts[label] != count for label, count in EXPECTED.items()):
        issues.append("RECORD_OR_LABEL_COUNT_MISMATCH")
    if success_keys != observed_keys:
        issues.append("EXTRACTION_SUCCESS_SOURCE_SET_MISMATCH")
    if any(len(group) != 712 for group in (extraction_rows, coverage_rows, technical_rows)):
        issues.append("METADATA_712_ROW_CONTRACT_MISMATCH")
    if parse_errors:
        issues.append("JSONL_PARSE_ERRORS")
    if any(row["schema_errors"] for row in rows):
        issues.append("RECORD_SCHEMA_OR_METADATA_ERRORS")
    if any(count > 1 for count in source_counts.values()):
        issues.append("DUPLICATE_SOURCE_VIDEO")
    if extraction_dupes or coverage_dupes or technical_dupes:
        issues.append("DUPLICATE_METADATA_KEYS")
    summary = {
        "status": "AUDIT_COMPLETE_WITH_INTEGRITY_ISSUES" if issues else "AUDIT_COMPLETE_REVIEW_REQUIRED",
        "purpose": "AUTOMATED_SEQUENCE_QUALITY_DIAGNOSTIC_ONLY",
        "record_count": len(rows), "label_counts": dict(counts), "expected_label_counts": EXPECTED,
        "dataset_issues": issues, "parse_errors": parse_errors,
        "missing_from_jsonl": sorted(success_keys - observed_keys),
        "unexpected_in_jsonl": sorted(observed_keys - success_keys),
        "duplicate_source_groups": {key: n for key, n in source_counts.items() if n > 1},
        "duplicate_exact_feature_groups": duplicates,
        "duplicate_metadata_keys": {
            "extraction": extraction_dupes, "coverage": coverage_dupes, "technical": technical_dupes},
        "review_record_count": sum(r["quality_decision"] == "REVIEW" for r in rows),
        "review_reason_counts": dict(Counter(reason for r in rows for reason in r["review_reasons"])),
        "statistical_tail_thresholds_by_label": thresholds,
        "threshold_policy": "Within-label empirical 1st/99th percentile; flags are review only.",
        "hand_identity_status": "UNVERIFIED: no per-frame identity/handedness in stored records",
        "signer_group_status": "UNVERIFIED: no trusted signer/session IDs in checked metadata",
        "source_rights_status": "UNVERIFIED: public availability or code license does not establish AVI rights",
        "semantic_label_status": "UNVERIFIED: motion outliers do not determine ASL correctness",
    }
    return rows, summary


def write_reports(rows: list[dict], summary: dict) -> None:
    if REPORT_CSV.exists() or REPORT_JSON.exists():
        raise FileExistsError("Refusing to overwrite existing sequence-quality report")
    fields = ["line_number", "source_path", "source_key", "label", "quality_decision",
              "review_reasons", "schema_errors", "identity_status", "decoded_frame_count",
              "valid_landmark_frames", "landmark_coverage_ratio", "invalid_reason_counts",
              "fps", "width", "height", "feature_sha256", "shape_rms", "shape_rms_8",
              "shape_rms_20", "max_shape_step", "max_shape_step_per_source_frame",
              "max_selected_frame_gap", "median_selected_frame_gap", "label_tip_path",
              "label_tip_max_step_rate"]
    for landmark in PAIRS:
        fields.extend(f"landmark_{landmark}_{name}" for name in
                      ("path", "net", "max_step", "max_step_per_source_frame"))
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            record = dict(row)
            record["review_reasons"] = "|".join(row["review_reasons"])
            record["schema_errors"] = "|".join(row["schema_errors"])
            writer.writerow(record)
    with REPORT_JSON.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Check inputs/counts; write nothing")
    mode.add_argument("--audit", action="store_true", help="Audit saved features and write temporary reports")
    args = parser.parse_args()
    checks = preflight()
    if args.check_only:
        print(json.dumps({"status": "INPUT_CHECK", **checks}, indent=2))
        return
    if checks["existing_outputs"]:
        raise FileExistsError(f"Existing outputs: {checks['existing_outputs']}")
    if input("Audit saved J/Z sequences (no video/MediaPipe/training)? [y/N] ").strip().lower() != "y":
        print("CANCELLED")
        return
    rows, summary = audit()
    write_reports(rows, summary)
    print(json.dumps({"status": summary["status"], "records": len(rows),
                      "review_records": summary["review_record_count"],
                      "report_csv": str(REPORT_CSV), "report_json": str(REPORT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
