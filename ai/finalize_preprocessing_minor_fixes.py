"""Finalize reporting from existing artefacts; never fit or extract landmarks.

Source CSVs, model checkpoints and group assignments are immutable inputs.
Ambiguous visual/semantic decisions prevent a READY freeze manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from audit_wrong_labels import alternative_neighbor_evidence

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ai/dataset"
EXPECTED_DATA = "7fdda53d591bfb7dbb3330f74b8ba3ae6328e5d9dcaad5869648de2275698639"
EXPECTED_GROUP = "0c04d591042630ceef12231b3d9ac0454007051a92f71946705aa5ae23f52b3b"
CANONICAL = DATA / "phase2_quality_gate_v2/dataset_chucai_clean_reviewed_v2.csv"
GROUP_MAP = DATA / "sample_group_map.csv"
ALPHABET_CONVENTION = "Alphabet convention: Standard ASL manual alphabet A-Z; J and Z handled as dynamic gestures; M and N use standard ASL forms."
STATIC_CLASSES = list("ABCDEFGHIKLMNOPQRSTUVWXY")
BLUR_FILES = [r"dulieu\chucai\a\color_0_0256.png", r"dulieu\chucai\c\mrinnec2.png"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(path: Path) -> dict:
    stat = path.stat()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path)}


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def required_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Invalid or missing Boolean: {value!r}")


def protected_files(selected: list[str]) -> list[Path]:
    files = {CANONICAL, GROUP_MAP, ROOT / "ai/raw_data/raw_landmarks_chucai_all.csv"}
    for name in ["dataset_chucai_all.csv", "dataset_chucai_clean.csv", "rejected_chucai.csv",
                 "landmark_quality_report.csv", "landmark_quality_summary.json",
                 "wrong_label_audit.csv", "wrong_label_candidates.csv",
                 "wrong_label_group_candidates.csv", "wrong_label_summary.json",
                 "near_duplicate_audit.csv", "near_duplicate_summary.json", "cross_label_near_matches.csv"]:
        files.add(DATA / name)
    for directory in [DATA / "phase2_quality_gate_v2", DATA / "wrong_label_checkpoints_v1",
                      DATA / "near_duplicate_checkpoints_v1", ROOT / "ai/quality_checkpoints",
                      ROOT / "ai/phase2_outputs/run_20260917_111456_626441"]:
        files.update(path for path in directory.rglob("*") if path.is_file())
    files.update(ROOT / name for name in selected)
    files.add(ROOT / "ai/models/hand_landmarker.task")
    return sorted(files)


def validate_inputs(*, reports_only: bool = False) -> tuple[list[dict], list[dict], dict]:
    assert sha256(CANONICAL) == EXPECTED_DATA, "Canonical hash changed"
    assert sha256(GROUP_MAP) == EXPECTED_GROUP, "Group map hash changed"
    groups = read_csv(GROUP_MAP)
    by_file = {row["file"]: row for row in groups}
    assert len(by_file) == len(groups) == 89428
    metadata = pd.read_csv(CANONICAL, usecols=["file", "label", "source", "handedness"])
    assert len(metadata) == metadata.file.nunique() == 89428
    assert set(metadata.label) == set(STATIC_CLASSES)
    assert set(metadata.file) == set(by_file)
    group_labels, group_sizes = {}, {}
    for row in metadata.itertuples(index=False):
        mapped = by_file[row.file]
        assert row.label == mapped["label"]
        group_labels.setdefault(mapped["group_id"], set()).add(row.label)
        group_sizes[mapped["group_id"]] = group_sizes.get(mapped["group_id"], 0) + 1
    assert all(len(labels) == 1 for labels in group_labels.values())
    assert all(int(row["group_size"]) == group_sizes[row["group_id"]] for row in groups)
    audit = pd.read_csv(DATA / "wrong_label_audit.csv", usecols=["file", "label", "group_id", "oof_fold"])
    assert audit.file.tolist() == metadata.file.tolist()
    assert audit.label.tolist() == metadata.label.tolist()
    assert audit.group_id.tolist() == [by_file[name]["group_id"] for name in audit.file]
    assert int(audit.groupby("group_id").oof_fold.nunique().max()) == 1
    hits = np.zeros(len(audit), dtype=np.int8)
    folds = []
    checkpoint_dir = DATA / "wrong_label_checkpoints_v1"
    progress = json.loads((checkpoint_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed_folds"] == progress["total_folds"] == 5
    assert progress["outputs_complete"] is True
    for fold in range(5):
        for kind in ["model", "complete"]:
            assert (checkpoint_dir / f"fold_{fold}_{kind}.joblib").stat().st_size > 0
        if reports_only:
            folds.append({"fold": fold, "files_present": True, "loaded_only": False,
                          "deserialized": False, "contents_revalidated": False})
            continue
        complete = joblib.load(checkpoint_dir / f"fold_{fold}_complete.joblib")
        model = joblib.load(checkpoint_dir / f"fold_{fold}_model.joblib")
        train, validation = model["train_idx"], complete["validation_idx"]
        assert len(np.intersect1d(train, validation)) == 0
        assert len(np.union1d(train, validation)) == len(audit)
        assert not set(audit.iloc[train].group_id) & set(audit.iloc[validation].group_id)
        assert np.array_equal(complete["result"].row_id.to_numpy(), validation)
        assert (audit.iloc[validation].oof_fold == fold).all()
        assert int(model["scaler"].n_samples_seen_) == len(train)
        assert len(model["mlp"].classes_) == 24
        hits[validation] += 1
        folds.append({"fold": fold, "train_rows": len(train), "validation_rows": len(validation),
                      "group_overlap": 0, "loaded_only": True})
    if not reports_only:
        assert (hits == 1).all()
    cache_rows = 0
    near_paths = sorted((DATA / "near_duplicate_checkpoints_v1").glob("same_label_*.npz"))
    assert len(near_paths) == 24
    for path in near_paths:
        if reports_only:
            assert path.stat().st_size > 0
            continue
        with np.load(path, allow_pickle=False) as cache:
            n = len(cache["row_indices"])
            assert cache["distances"].shape == cache["neighbors"].shape
            assert cache["distances"].shape[0] == n
            cache_rows += n
    if not reports_only:
        assert cache_rows == len(audit)
    quality_meta = json.loads((ROOT / "ai/quality_checkpoints/checkpoint_meta.json").read_text())
    for name, recorded in quality_meta["source_signature"].items():
        stat = (ROOT / name).stat()
        assert stat.st_size == recorded["size"] and stat.st_mtime_ns == recorded["mtime_ns"]
    promoted = read_csv(DATA / "phase2_quality_gate_v2/review_promoted_to_keep_v2.csv")
    promoted_files = {row["file"] for row in promoted}
    assert len(promoted_files) == len(promoted) == 230
    quality = pd.read_csv(DATA / "landmark_quality_report.csv", dtype=str, keep_default_na=False)
    assert not quality.file.duplicated().any()
    q = quality.set_index("file").loc[metadata.file]
    for flag in ["bad_geometry", "bad_stability", "bad_image_quality"]:
        assert not any(required_bool(value) for value in q[flag])
    assert q.final_status.value_counts().to_dict() == {"KEEP": 89198, "REVIEW": 230}
    assert set(q.index[q.final_status == "REVIEW"]) == promoted_files
    remaining = read_csv(DATA / "phase2_quality_gate_v2/review_remaining_v2.csv")
    assert not set(metadata.file) & {row["file"] for row in remaining}
    return read_csv(DATA / "wrong_label_candidates.csv"), read_csv(DATA / "wrong_label_group_candidates.csv"), {
        "samples": len(audit), "groups": len(group_sizes), "duplicate_files": 0,
        "bad_flag_violations": 0, "remaining_review_or_exclude_in_canonical": 0,
        "oof_checkpoints": folds, "near_cache_labels": 24,
        "near_cache_rows": None if reports_only else cache_rows,
        "checkpoint_contents_revalidated": not reports_only,
        "checkpoint_validation_scope": "File presence/current hash snapshots and saved OOF progress only; no joblib/NPZ deserialization." if reports_only else "Loaded existing checkpoints for structural checks; no inference or fitting.",
        "source_metadata_matches": True, "historical_hashes_available_for": ["canonical", "group_map"],
        "historical_hash_limitation": "Other files did not have recorded historical hashes; current snapshots provide forward integrity, not proof of their past byte identity.",
    }


def repair_csv(source: Path, target: Path) -> dict:
    changed = alternative_present = rows = 0
    with source.open(encoding="utf-8-sig", newline="") as src, target.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        assert reader.fieldnames is not None
        extra = ["knn_modal_alternative_class", "knn_alternative_search_status"]
        assert not set(extra) & set(reader.fieldnames)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames + extra)
        writer.writeheader()
        for original in reader:
            labels = original["knn_labels"].split("|")
            assert len(labels) == 7 and all(labels)
            nearest, modal, scope = alternative_neighbor_evidence(labels, original["label"])
            assert modal == original["knn_nearest_alternative_class"], "Legacy semantics differ from expected mode"
            fixed = dict(original)
            fixed["knn_nearest_alternative_class"] = nearest
            fixed["knn_modal_alternative_class"] = modal
            fixed["knn_alternative_search_status"] = scope
            changed += nearest != original["knn_nearest_alternative_class"]
            alternative_present += scope == "FOUND_IN_TOP_K"
            assert all(fixed[key] == value for key, value in original.items() if key != "knn_nearest_alternative_class")
            writer.writerow(fixed)
            rows += 1
    # Independent read-back verifies preservation of every original cell apart
    # from the one intentionally repaired reporting column.
    with source.open(encoding="utf-8-sig", newline="") as src, target.open(encoding="utf-8", newline="") as dst:
        before, after = csv.DictReader(src), csv.DictReader(dst)
        for left, right in zip(before, after, strict=True):
            assert all(left[key] == right[key] for key in left if key != "knn_nearest_alternative_class")
    return {"rows": rows, "nearest_alternative_changed": changed,
            "alternative_observed_in_top7": alternative_present,
            "other_original_cells_lexically_identical": True}


def blur_supplement(raw: pd.DataFrame) -> list[dict]:
    result = []
    for name in BLUR_FILES:
        image = cv2.imread(str(ROOT / name))
        assert image is not None
        h, w = image.shape[:2]
        row = raw.loc[name]
        xs = [float(row[f"x{i}"]) for i in range(21)]
        ys = [float(row[f"y{i}"]) for i in range(21)]
        x0, x1, y0, y1 = int(min(xs) * w), int(max(xs) * w), int(min(ys) * h), int(max(ys) * h)
        mx, my = int(max(1, x1 - x0) * .25), int(max(1, y1 - y0) * .25)
        left, right, top, bottom = max(0, x0 - mx), min(w, x1 + mx), max(0, y0 - my), min(h, y1 + my)
        crop = image[top:bottom, left:right]
        assert crop.size and min(crop.shape[:2]) < 10
        full_blur = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        # The numerical value of a too-small/mislocalized crop is retained only
        # as diagnostic evidence and must never be substituted for blur_hand.
        diagnostic = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        result.append({"file": name, "image_width": w, "image_height": h,
                       "roi_left": left, "roi_top": top, "roi_right": right, "roi_bottom": bottom,
                       "roi_width": crop.shape[1], "roi_height": crop.shape[0],
                       "quality_measurement_status": "UNAVAILABLE_ROI_TOO_SMALL",
                       "blur_hand_usable": False, "blur_hand": "",
                       "diagnostic_small_roi_laplacian_variance": diagnostic,
                       "diagnostic_full_image_laplacian_variance": full_blur,
                       "freeze_eligible_without_review": False,
                       "required_action": "Resolve saved-landmark/image correspondence and explicit keep/exclude disposition; missing measurement is not PASS."})
    return result


def gallery(path: Path, observations: list[dict], raw: pd.DataFrame) -> None:
    links = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
             (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
    cards = []
    for number, item in enumerate(observations, 1):
        name = item["file"]
        image = cv2.imread(str(ROOT / name))
        assert image is not None
        h, w = image.shape[:2]
        rel = os.path.relpath(ROOT / name, path.parent).replace("\\", "/")
        url = html.escape(rel, quote=True)
        row = raw.loc[name]
        points = [(float(row[f"x{i}"]) * w, float(row[f"y{i}"]) * h) for i in range(21)]
        svg = [f'<svg viewBox="0 0 {w} {h}" role="img"><image href="{url}" width="{w}" height="{h}"/>']
        for a, b in links:
            svg.append(f'<line x1="{points[a][0]}" y1="{points[a][1]}" x2="{points[b][0]}" y2="{points[b][1]}" stroke="#00ff70" stroke-width="{max(.7,w/200)}"/>')
        for i, (x, y) in enumerate(points):
            svg.append(f'<circle cx="{x}" cy="{y}" r="{max(.8,w/150)}" fill="red"/><text x="{x+1}" y="{y-1}" font-size="{max(3,w/50)}" fill="yellow">{i}</text>')
        svg.append('</svg>')
        cards.append(f'<article><h2>{number}. {html.escape(name)}</h2><div class="pair"><img src="{url}" alt="Original image"/>{"".join(svg)}</div><p>Semantic status: {html.escape(item["semantic_status"])}</p><p>{html.escape(item["visual_observation"])}</p><p>{html.escape(item["landmark_followup"])}</p><p>Disposition: {html.escape(item["disposition"])}. Reviewer: assistant visual inspection; not human certification.</p></article>')
    path.write_text('<!doctype html><meta charset="utf-8"><title>Pre-freeze semantic review</title><style>body{font:16px sans-serif;max-width:1200px;margin:auto;padding:24px}article{border-top:1px solid #bbb;padding:20px 0}.pair{display:flex;gap:20px}.pair>*{width:46%;height:360px;object-fit:contain;background:#eee}h2{font-size:17px}</style><h1>Original image and saved raw landmarks</h1><p>No MediaPipe rerun. Enlargement creates no additional image evidence. Static overlay does not measure stability. Yellow numbers are landmark indices.</p>' + ''.join(cards), encoding="utf-8")


def validate_semantic_review(review: dict) -> None:
    assert review["alphabet_convention_confirmed"] is True
    assert review["alphabet_convention"] == ALPHABET_CONVENTION
    assert review["static_classes"] == STATIC_CLASSES
    assert review["dynamic_classes_excluded_from_static_alternatives"] == ["J", "Z"]
    assert review["label_changes"] == review["excluded_files"] == []
    counts = {"SEMANTIC_ORIGINAL_LABEL_SUPPORTED": 0,
              "NEEDS_MANUAL_SEMANTIC_REVIEW": 0,
              "SEMANTIC_LABEL_MISMATCH_CANDIDATE": 0}
    for row in review["observations"]:
        assert row["original_label"] in STATIC_CLASSES
        assert row["label_action"] == "UNCHANGED"
        assert row["sample_action"] == "RETAINED_DURING_AUDIT"
        assert row["disposition"] in {"PENDING_REVIEW", "REVIEW_RESOLVED"}
        alternative = row["semantic_alternative_label"]
        assert alternative is None or (alternative in STATIC_CLASSES and alternative != row["original_label"])
        if row["review_scope"] != "MISSING_HAND_BLUR_FOLLOWUP":
            assert row["semantic_status"] in counts
            counts[row["semantic_status"]] += 1
        else:
            assert row["semantic_status"] == "NOT_ASSESSED_IN_THIS_SEMANTIC_SCOPE"
        if row["disposition"] == "REVIEW_RESOLVED":
            assert row.get("resolution_evidence") and row.get("resolution_reviewer")
    assert counts == review["semantic_counts"]
    assert sum(counts.values()) == 14


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "distribution_not_installed_under_this_name"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=ROOT / "ai/preprocessing_semantic_observations.json")
    parser.add_argument("--output-dir", type=Path, default=DATA / "preprocessing_final_review_v1")
    parser.add_argument("--reports-only", action="store_true",
                        help="No joblib/NPZ deserialization. Publish pending review reports, never a READY manifest.")
    args = parser.parse_args()
    assert not args.output_dir.exists(), "Refuse to overwrite an existing review package"
    review = json.loads(args.observations.read_text(encoding="utf-8"))
    validate_semantic_review(review)
    observations = review["observations"]
    selected = [row["file"] for row in observations]
    print("Verifying immutable inputs; reports-only=" + str(args.reports_only), flush=True)
    candidates, group_candidates, integrity = validate_inputs(reports_only=args.reports_only)
    expected = set(BLUR_FILES) | {row["file"] for row in candidates}
    expected |= {name for row in group_candidates for name in row["files"].split("|")}
    assert set(selected) == expected and len(selected) == len(expected) == 16
    expected_labels = {row["file"]: row["label"] for row in candidates}
    expected_labels.update({name: row["label"] for row in group_candidates for name in row["files"].split("|")})
    expected_labels.update(dict(zip(BLUR_FILES, ["A", "C"], strict=True)))
    assert all(item["original_label"] == expected_labels[item["file"]] for item in observations)
    inputs = protected_files(selected)
    before = {str(path): signature(path) for path in inputs}
    # Staging is retained for inspection after failures; no recursive deletion.
    stage = Path(tempfile.mkdtemp(prefix="preprocessing_review_staging_", dir=args.output_dir.parent))
    print(f"Writing bounded reporting repairs to {stage}", flush=True)
    write_json(stage / "resume_integrity.json", integrity)
    repair = repair_csv(DATA / "wrong_label_audit.csv", stage / "wrong_label_audit_v2.csv")
    assert repair["rows"] == 89428 and repair["nearest_alternative_changed"] == 433
    candidate_repair = repair_csv(DATA / "wrong_label_candidates.csv", stage / "wrong_label_candidates_v2.csv")
    assert candidate_repair["rows"] == 11
    raw = pd.read_csv(ROOT / "ai/raw_data/raw_landmarks_chucai_all.csv")
    assert not raw.file.duplicated().any()
    raw = raw.set_index("file").loc[selected]
    blur = blur_supplement(raw)
    write_csv(stage / "missing_blur_review.csv", blur)
    write_csv(stage / "semantic_review.csv", observations)
    write_json(stage / "semantic_review.json", review)
    (stage / "REVIEW.md").write_text((ROOT / "ai/PREPROCESSING_FREEZE_REVIEW.md").read_text(encoding="utf-8"), encoding="utf-8")
    gallery(stage / "semantic_review.html", observations, raw)
    summary = json.loads((DATA / "wrong_label_summary.json").read_text(encoding="utf-8"))
    summary["reporting_revision"] = {
        "version": 2, "predictions_recomputed": False, "candidate_policy_changed": False,
        "nearest_alternative": "First differently labelled neighbor in saved distance-ordered top7; blank means unobserved in top7, not absent globally.",
        "modal_alternative": "Historical value preserved under its correct modal name.",
        "LABEL_CONSISTENT_meaning": "No sufficient multi-signal flag; not semantic verification of the label.",
        "corrections": repair, "candidate_corrections": candidate_repair,
    }
    summary["outputs"]["wrong_label_audit"] = str(args.output_dir / "wrong_label_audit_v2.csv")
    summary["outputs"]["wrong_label_candidates"] = str(args.output_dir / "wrong_label_candidates_v2.csv")
    summary["outputs"]["wrong_label_summary"] = str(args.output_dir / "wrong_label_summary_v2.json")
    write_json(stage / "wrong_label_summary_v2.json", summary)
    after = {str(path): signature(path) for path in inputs}
    assert before == after, "A protected input changed during reporting repair"
    assert review["label_changes"] == [] and review["excluded_files"] == []
    # No automatic 'keep' decision is manufactured to make a freeze pass.
    unresolved = [item["file"] for item in observations if item["disposition"] != "REVIEW_RESOLVED"]
    semantic_pending = [item["file"] for item in observations if item["semantic_status"] in
                        {"NEEDS_MANUAL_SEMANTIC_REVIEW", "SEMANTIC_LABEL_MISMATCH_CANDIDATE"}]
    blockers = []
    if args.reports_only:
        blockers.append("CHECKPOINT_CONTENT_REVALIDATION_NOT_PERFORMED_IN_REPORTS_ONLY_MODE")
    if not review["alphabet_convention_confirmed"]:
        blockers.append("ALPHABET_CONVENTION_NOT_CONFIRMED")
    if unresolved:
        blockers.append("SEMANTIC_OR_LANDMARK_DISPOSITIONS_PENDING")
    if semantic_pending:
        blockers.append("MANUAL_SEMANTIC_REVIEW_PENDING")
    if any(not item["blur_hand_usable"] and item["file"] in unresolved for item in blur):
        blockers.append("TWO_MISSING_BLUR_DISPOSITIONS_PENDING")
    ready = not blockers
    post = {**integrity, "protected_inputs_unchanged": True,
            "canonical_sha256": EXPECTED_DATA, "group_map_sha256": EXPECTED_GROUP,
            "reporting_repairs": repair, "label_changes": 0, "excluded_samples": 0,
            "features_and_other_audit_cells_preserved": True,
            "retraining_or_extraction_performed": False, "unresolved_review_files": unresolved}
    write_json(stage / "post_fix_integrity.json", post)
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PREPROCESSING_READY_FOR_SPLIT" if ready else "NEEDS_MINOR_FIXES",
        "alphabet_convention": review["alphabet_convention"],
        "alphabet_convention_confirmed": True,
        "static_classes": review["static_classes"],
        "dynamic_classes": ["J", "Z"],
        "semantic_counts": review["semantic_counts"],
        "semantic_review_pending_files": semantic_pending,
        "freeze_authorized": ready, "blockers": blockers,
        "canonical_dataset": before[str(CANONICAL)], "sample_group_map": before[str(GROUP_MAP)],
        "protected_input_snapshots": list(before.values()),
        "current_code": [signature(path) for path in [Path(__file__).resolve(), ROOT / "ai/audit_wrong_labels.py",
                         ROOT / "ai/run_landmark_quality_pipeline.py", ROOT / "ai/preprocessing/landmark_utils.py",
                         ROOT / "ai/audit_near_duplicates.py", ROOT / "ai/PREPROCESSING_FREEZE_REVIEW.md", args.observations]],
        "packages": {name: package_version(name) for name in ["numpy", "pandas", "scikit-learn", "joblib", "opencv-python"]},
        "python": platform.python_version(),
        "review_files": unresolved,
        "limitations": [
            "89,198 original KEEP rows have no stability measurements; false flags are not measured stability passes.",
            "21 promoted rows retain stability_candidate warnings under the accepted policy.",
            "Group IDs constrain detected variants; they are not person/session identities.",
            "LABEL_CONSISTENT means not flagged, not human-confirmed correctness.",
            "OOF metrics describe the curated corpus; they are not subject-independent or untouched-test accuracy.",
            "Current hashes establish forward provenance where historical hashes were not stored.",
        ],
        "outputs": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in sorted(stage.iterdir()) if path.is_file()],
    }
    manifest_name = "final_freeze_manifest.json" if ready else "freeze_manifest.pending.json"
    write_json(stage / manifest_name, manifest)
    os.replace(stage, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "status": manifest["status"],
                      "blockers": blockers, "corrected_nearest_alternative_rows": repair["nearest_alternative_changed"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
