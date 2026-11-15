"""Publish a versioned static corpus from existing human/landmark dispositions.

No fitting, inference, extraction, normalization or neighbor search is performed.
All parent artefacts and source images remain immutable. Distances in the child
group map are explicitly historical; stable group IDs preserve split safety.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ai/dataset"
REVIEW = DATA / "preprocessing_final_review_v1"
PARENT = DATA / "phase2_quality_gate_v2/dataset_chucai_clean_reviewed_v2.csv"
PARENT_MAP = DATA / "sample_group_map.csv"
OUTPUT = DATA / "preprocessing_frozen_v3"
POINTER = DATA / "canonical_preprocessing_manifest.json"
EVIDENCE = ROOT / "ai/final_landmark_dispositions.json"
CONVENTION = "Alphabet convention: Standard ASL manual alphabet A-Z; J and Z handled as dynamic gestures; M and N use standard ASL forms."
CLASSES = set("ABCDEFGHIKLMNOPQRSTUVWXY")
FEATURES = [f"f{i}" for i in range(1, 64)]
RECONCILIATION_SHA = "e5d10a662708974cd3dbfa1644dc16404b88e5062742ce3ace930f329d2b6a3d"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, document):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames), list(reader)


def write_rows(path, columns, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parent_integrity():
    recon_path = REVIEW / "manual_review_reconciliation_v2.json"
    require(digest(recon_path) == RECONCILIATION_SHA, "Reconciliation checkpoint changed")
    recon = load_json(recon_path)
    manifest_path = REVIEW / "freeze_manifest.pending.json"
    manifest = load_json(manifest_path)
    checked = {}
    sections = [("protected_input_snapshots", ROOT), ("current_code", ROOT), ("outputs", REVIEW)]
    for section, base in sections:
        for item in manifest[section]:
            path = base / item["path"]
            expected = item["sha256"]
            if section == "outputs" and item["path"] == "semantic_review.csv":
                expected = recon["manual_decision_source"]["sha256"]
            require(path.is_file() and digest(path) == expected, f"Unexpected parent change: {path}")
            checked[path.relative_to(ROOT).as_posix()] = expected
    require(len(checked) == 110, "Unexpected parent verification scope")
    for path in [manifest_path, recon_path, EVIDENCE, Path(__file__).resolve(),
                 ROOT / "tests" / "test_finalize_static_dataset.py",
                 REVIEW / "semantic_review.html"]:
        checked[path.relative_to(ROOT).as_posix()] = digest(path)
    for row in load_json(EVIDENCE)["observations"]:
        path = (ROOT / row["saved_overlay"]).resolve()
        require(path.is_relative_to(ROOT.resolve()), "Overlay path outside project")
        checked[path.relative_to(ROOT).as_posix()] = digest(path)
    require(load_json(DATA / "wrong_label_checkpoints_v1/progress.json")["outputs_complete"] is True,
            "Saved OOF progress incomplete")
    require(load_json(DATA / "near_duplicate_checkpoints_v1/progress.json")["stage"] == "complete",
            "Saved near-duplicate progress incomplete")
    return checked


def decide(review_rows, evidence_rows):
    by_file = {row["file"]: row for row in evidence_rows}
    require(len(by_file) == len(evidence_rows), "Duplicate correspondence evidence")
    require(len({r["file"] for r in review_rows}) == len(review_rows), "Duplicate manual review file")
    require(set(by_file) <= {r["file"] for r in review_rows}, "Unexpected correspondence file")
    result = []
    for row in review_rows:
        name, original = row["file"], row["original_label"]
        require(original in CLASSES, "Original label outside static ASL classes")
        status = row["semantic_status"]
        target, reasons = original, []
        if status == "LABEL_CORRECT":
            require(not row["semantic_alternative_label"].strip(), "Conflicting correct-label alternative")
        elif status == "LABEL_INCORRECT":
            target = row["semantic_alternative_label"].strip()
            require(target in CLASSES and target != original, "Missing/invalid manually confirmed alternative")
        elif status == "NEEDS_MANUAL_SEMANTIC_REVIEW":
            require(not row["semantic_alternative_label"].strip(), "Unresolved alternative is not authorization")
            reasons.append("SEMANTIC_UNRESOLVED_AFTER_MANUAL_REVIEW")
        elif status == "NOT_ASSESSED_IN_THIS_SEMANTIC_SCOPE":
            require(row["review_scope"] == "MISSING_HAND_BLUR_FOLLOWUP" and name in by_file,
                    "Unassessed sample lacks separate quality disposition")
        else:
            raise ValueError(f"Unrecognized/unresolved manual status for {name}: {status}")
        evidence = by_file.get(name)
        if evidence:
            require(evidence["evidence"] and evidence["saved_overlay"], "Missing correspondence evidence")
            if evidence["decision"] == "EXCLUDE_CORRESPONDENCE":
                reasons.append("LANDMARK_IMAGE_CORRESPONDENCE_FAILED")
                if row["review_scope"] == "MISSING_HAND_BLUR_FOLLOWUP":
                    require(evidence["roi_invalid_due_to_correspondence"] is True, "Unexplained invalid ROI")
                    reasons.append("HAND_BLUR_ROI_INVALID_DUE_TO_CORRESPONDENCE")
            elif evidence["decision"] == "KEEP_CORRESPONDENCE":
                require(evidence.get("correspondence_valid") is True, "Unvalidated correspondence keep")
                if row["review_scope"] == "MISSING_HAND_BLUR_FOLLOWUP":
                    require(evidence.get("blur_hand_usable") is True and
                            isinstance(evidence.get("blur_hand"), (int, float)) and
                            math.isfinite(evidence["blur_hand"]) and evidence.get("blur_policy_pass") is True,
                            "No valid hand ROI/blur policy evidence")
            else:
                raise ValueError(f"Unresolved correspondence disposition: {name}")
        excluded = bool(reasons)
        result.append({"file": name, "original_label": original, "manual_status": status,
                       "decision": "EXCLUDE_FROM_STATIC_CANONICAL" if excluded else "KEEP",
                       "retained_label": "" if excluded else target,
                       "label_action": "UNCHANGED" if excluded or target == original else "MANUAL_RELABEL",
                       "reason": "|".join(reasons) if reasons else "USER_CONFIRMED_LABEL",
                       "image_action": "PRESERVE_ORIGINAL", "evidence": evidence["evidence"] if evidence else "User manual CSV and explicit unresolved-sample exclusion rule."})
    return result


def make_group_map(parent_rows, survivors, labels):
    rows = [row for row in parent_rows if row["file"] in survivors]
    require(len(rows) == len(survivors) == len({row["file"] for row in rows}), "Group-map membership mismatch")
    sizes = Counter(row["group_id"] for row in rows)
    filenames = Counter(row["filename_group_key"] for row in rows)
    group_labels = defaultdict(set)
    result = []
    for original in rows:
        row = dict(original)
        name = row["file"]
        row["label"] = labels[name]
        # Current task has no manual relabels. Fail closed if a future relabel
        # would need filename-key regeneration or mixed-label regrouping.
        require(row["label"] == original["label"], "Relabel requires separate group-key reconciliation")
        row["group_size"] = str(sizes[row["group_id"]])
        row["filename_group_size"] = str(filenames[row["filename_group_key"]])
        if sizes[row["group_id"]] == 1:
            row["group_type"] = row["duplicate_status"] = "UNIQUE"
        for column in ["nearest_distance", "nearest_outside_filename_group_distance"]:
            row["parent_" + column] = row[column]
            row[column] = ""
        row["distance_status"] = "NOT_RECOMPUTED_AFTER_CURATION_USE_PARENT_EVIDENCE_ONLY"
        group_labels[row["group_id"]].add(row["label"])
        result.append(row)
    require(all(len(values) == 1 for values in group_labels.values()), "Mixed-label group")
    return result, sizes


def metadata():
    by_file, counts = {}, Counter()
    with PARENT.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(all(feature in reader.fieldnames for feature in FEATURES), "Incomplete feature schema")
        for row in reader:
            require(None not in row and all(value is not None for value in row.values()), "Malformed canonical CSV")
            require(row["file"] not in by_file and row["label"] in CLASSES, "Duplicate file or invalid class")
            require(all(math.isfinite(float(row[f])) for f in FEATURES), "Nonfinite feature")
            by_file[row["file"]] = {key: row[key] for key in ["label", "source", "handedness"]}
            counts[row["label"]] += 1
    require(len(by_file) == 89428 and set(counts) == CLASSES, "Parent membership/class count changed")
    return by_file, counts


def emit_dataset(stage, decisions):
    by_file = {row["file"]: row for row in decisions}
    excluded, kept = 0, 0
    with PARENT.open(encoding="utf-8-sig", newline="") as src, \
            (stage / "dataset_chucai_clean_frozen_v3.csv").open("x", encoding="utf-8", newline="") as dst, \
            (stage / "excluded_samples.csv").open("x", encoding="utf-8", newline="") as exc:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        removed = csv.DictWriter(exc, fieldnames=reader.fieldnames + ["exclusion_reason"])
        writer.writeheader()
        removed.writeheader()
        for row in reader:
            decision = by_file.get(row["file"])
            if decision and decision["decision"] == "EXCLUDE_FROM_STATIC_CANONICAL":
                removed.writerow({**row, "exclusion_reason": decision["reason"]})
                excluded += 1
                continue
            if decision:
                row["label"] = decision["retained_label"]
            writer.writerow(row)
            kept += 1
    return kept, excluded


def verify_dataset(stage, decisions, survivors):
    by_file = {row["file"]: row for row in decisions}
    labels, seen = Counter(), set()
    expected_removed = []
    with PARENT.open(encoding="utf-8-sig", newline="") as src, \
            (stage / "dataset_chucai_clean_frozen_v3.csv").open(encoding="utf-8", newline="") as dst:
        before, after = csv.DictReader(src), csv.DictReader(dst)
        require(before.fieldnames == after.fieldnames, "Canonical schema mutation")
        for left in before:
            if left["file"] not in survivors:
                require(left["file"] in by_file and by_file[left["file"]]["decision"] == "EXCLUDE_FROM_STATIC_CANONICAL",
                        "Unexpected removal without decision")
                expected_removed.append({**left, "exclusion_reason": by_file[left["file"]]["reason"]})
                continue
            right = next(after, None)
            require(right is not None, "Truncated child dataset")
            expected = dict(left)
            if left["file"] in by_file:
                expected["label"] = by_file[left["file"]]["retained_label"]
            require(expected == right, f"Unexpected cell mutation: {left['file']}")
            require(right["file"] not in seen, "Duplicate child file")
            seen.add(right["file"])
            labels[right["label"]] += 1
        require(next(after, None) is None and seen == survivors, "Unexpected child membership")
    require(set(labels) == CLASSES, "A static class disappeared")
    _, excluded_rows = read_rows(stage / "excluded_samples.csv")
    require(excluded_rows == expected_removed, "Excluded ledger cells or membership changed")
    return labels


def verify_quality(survivors):
    _, promoted = read_rows(DATA / "phase2_quality_gate_v2/review_promoted_to_keep_v2.csv")
    promoted_files = {row["file"] for row in promoted}
    seen, statuses = set(), Counter()
    with (DATA / "landmark_quality_report.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["file"] not in survivors:
                continue
            require(row["file"] not in seen, "Duplicate quality join key")
            seen.add(row["file"])
            require(all(row[flag] == "False" for flag in ["bad_geometry", "bad_stability", "bad_image_quality"]),
                    "Bad or missing original quality flag in child")
            require(row["final_status"] == "KEEP" or
                    (row["final_status"] == "REVIEW" and row["file"] in promoted_files),
                    "Unpromoted REVIEW or EXCLUDE in child")
            statuses[row["final_status"]] += 1
    require(seen == survivors, "Missing quality evidence")
    return statuses


def pointer_for(manifest_path):
    document = load_json(manifest_path)
    return {"status": document["status"], "manifest": manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": digest(manifest_path), "canonical_dataset": document["canonical_dataset"],
            "sample_group_map": document["sample_group_map"], "alphabet_convention": CONVENTION}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-user-exclusion-policy", action="store_true", required=True)
    args = parser.parse_args()
    require(args.apply_user_exclusion_policy, "Explicit user exclusion policy required")
    snapshots = parent_integrity()
    if OUTPUT.exists():
        manifest_path = OUTPUT / "final_freeze_manifest.json"
        manifest = load_json(manifest_path)
        require(manifest["status"] == "PREPROCESSING_READY_FOR_SPLIT", "Existing output is not complete")
        require(manifest["protected_parent_snapshots"] == snapshots, "Existing freeze uses different inputs/code")
        for item in manifest["outputs"]:
            require(digest(OUTPUT / item["path"]) == item["sha256"], "Existing output hash differs")
        expected_pointer = pointer_for(manifest_path)
        if POINTER.exists():
            require(load_json(POINTER) == expected_pointer, "Existing canonical pointer differs")
        else:
            write_json(POINTER, expected_pointer)
        print(json.dumps({"status": manifest["status"], "resumed_complete_output": True}))
        return
    require(not POINTER.exists(), "Refuse to overwrite existing canonical pointer")
    manual_columns, manual = read_rows(REVIEW / "semantic_review.csv")
    evidence = load_json(EVIDENCE)
    require(evidence["alphabet_convention"] == CONVENTION, "Alphabet convention mismatch")
    require(len(manual) == 16 and len(evidence["observations"]) == 6, "Unexpected bounded review scope")
    decisions = decide(manual, evidence["observations"])
    parent_meta, parent_counts = metadata()
    for row in decisions:
        require(row["file"] in parent_meta and row["original_label"] == parent_meta[row["file"]]["label"],
                "Review/canonical label or file mismatch")
    excluded = {row["file"] for row in decisions if row["decision"] == "EXCLUDE_FROM_STATIC_CANONICAL"}
    survivors = set(parent_meta) - excluded
    labels = {name: row["label"] for name, row in parent_meta.items() if name in survivors}
    labels.update({row["file"]: row["retained_label"] for row in decisions if row["file"] in survivors})
    map_columns, parent_map = read_rows(PARENT_MAP)
    require(len(parent_map) == len(parent_meta) and {row["file"] for row in parent_map} == set(parent_meta),
            "Parent map does not cover parent corpus one-to-one")
    child_map, group_sizes = make_group_map(parent_map, survivors, labels)
    stage = Path(tempfile.mkdtemp(prefix="preprocessing_freeze_staging_", dir=DATA))
    require(stage.resolve().parent == DATA.resolve() and OUTPUT.resolve().parent == DATA.resolve(),
            "Publishing paths must remain inside dataset directory")
    write_json(stage / "progress.json", {"stage": "INPUTS_VALIDATED", "parent_rows": len(parent_meta)})
    kept_count, excluded_count = emit_dataset(stage, decisions)
    write_rows(stage / "sample_group_map_frozen_v3.csv", map_columns +
               ["parent_nearest_distance", "parent_nearest_outside_filename_group_distance", "distance_status"], child_map)
    write_rows(stage / "curation_decisions.csv", list(decisions[0]), decisions)
    shutil.copyfile(REVIEW / "semantic_review.csv", stage / "source_manual_review.csv")
    shutil.copyfile(EVIDENCE, stage / "landmark_dispositions.json")
    child_counts = verify_dataset(stage, decisions, survivors)
    require(kept_count == len(survivors) and excluded_count == len(excluded), "Dataset count mismatch")
    _, published_map = read_rows(stage / "sample_group_map_frozen_v3.csv")
    require(published_map == child_map, "Group-map readback differs")
    quality_counts = verify_quality(survivors)
    test = subprocess.run([sys.executable, "-B", "-m", "unittest", "tests.test_finalize_static_dataset", "-q"],
                          cwd=ROOT, capture_output=True, text=True, check=False)
    (stage / "unit_tests.txt").write_text(test.stdout + test.stderr, encoding="utf-8")
    require(test.returncode == 0, "Curation unit tests failed; staging retained")
    for relative, expected in snapshots.items():
        require(digest(ROOT / relative) == expected, f"Protected input changed during publication: {relative}")
    changes = sum(row["label_action"] == "MANUAL_RELABEL" for row in decisions)
    integrity = {"status": "PASS", "parent_samples": len(parent_meta), "retained_samples": kept_count,
                 "excluded_samples": excluded_count, "duplicate_files": 0, "missing_group_members": 0,
                 "mixed_label_groups": 0, "retained_group_ids_unchanged": True,
                 "features_lexically_identical": True, "unapproved_label_changes": 0,
                 "approved_label_changes": changes, "bad_quality_flag_violations": 0,
                 "unpromoted_review_or_exclude": 0, "original_quality_status_counts": dict(quality_counts),
                 "source_images_and_parent_artifacts_unchanged": True,
                 "protected_hashes_verified_before_and_after": len(snapshots),
                 "remaining_blockers": [], "unit_tests": "PASS"}
    write_json(stage / "integrity_report.json", integrity)
    summary = {"parent_label_counts": dict(sorted(parent_counts.items())),
               "frozen_label_counts": dict(sorted(child_counts.items())),
               "groups": len(group_sizes), "singleton_groups": sum(n == 1 for n in group_sizes.values()),
               "group_size_distribution": dict(sorted(Counter(group_sizes.values()).items())),
               "manual_status_counts": dict(Counter(row["semantic_status"] for row in manual)),
               "excluded_reason_counts_nonexclusive": dict(Counter(reason for row in decisions
                    if row["decision"] == "EXCLUDE_FROM_STATIC_CANONICAL" for reason in row["reason"].split("|"))),
               "approved_label_changes": changes, "no_images_deleted": True}
    write_json(stage / "freeze_summary.json", summary)
    (stage / "README.md").write_text(
        "# Frozen static ASL corpus v3\n\n" + CONVENTION + "\n\n"
        f"Canonical: dataset_chucai_clean_frozen_v3.csv ({kept_count:,} rows). "
        "Split grouping: sample_group_map_frozen_v3.csv, using group_id.\n\n"
        f"Removed {excluded_count} rows under explicit user curation policy; original images and all parent "
        "CSVs/checkpoints remain unchanged. Excluded rows are recoverable in excluded_samples.csv and the parent corpus.\n\n"
        "Group IDs are inherited unchanged, including conservative transitive groups; group_size and filename_group_size "
        "are recomputed on survivors. Singleton types are UNIQUE. parent_nearest_* values describe the old corpus; "
        "current nearest-distance columns are empty, not zero, because no neighbor audit was rerun.\n\n"
        "Earlier OOF scores apply to the parent corpus, not new independent validation of this curated subset. "
        "Groups are detected variants, not proven person/session identities. The accepted preprocessing policy "
        "still includes its documented unmeasured-stability and architecture limitations. No split or training was run.\n",
        encoding="utf-8")
    # Only this newly generated staging checkpoint is updated; parent artefacts
    # and user-edited manual decisions are never rewritten.
    (stage / "progress.json").write_text(json.dumps({"stage": "COMPLETE", "integrity": "PASS",
        "retained_samples": kept_count, "excluded_samples": excluded_count}, indent=2), encoding="utf-8")
    def output_signature(name):
        path = stage / name
        return {"path": (OUTPUT / name).relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                "sha256": digest(path)}
    manifest = {"schema_version": 3, "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "PREPROCESSING_READY_FOR_SPLIT", "freeze_authorized": True,
                "alphabet_convention": CONVENTION, "static_classes": sorted(CLASSES), "dynamic_classes": ["J", "Z"],
                "canonical_dataset": output_signature("dataset_chucai_clean_frozen_v3.csv"),
                "sample_group_map": output_signature("sample_group_map_frozen_v3.csv"),
                "sample_count": kept_count, "group_count": len(group_sizes), "remaining_blockers": [],
                "protected_parent_snapshots": snapshots, "integrity": integrity,
                "decision_authority": "User explicitly authorized exclusion of still-uncertain manual semantics and failed correspondence; label confirmations do not override landmark failures.",
                "normalization_and_quality_thresholds_changed": False,
                "pipelines_rerun": [], "split_created": False, "production_model_trained": False,
                "historical_checkpoint_scope": "Byte-integrity reuse; no model loading or inference.",
                "group_distance_semantics": "parent_nearest_* is historical evidence; current distances are uncomputed and blank.",
                "limitations_inherited_from_accepted_review": load_json(REVIEW / "freeze_manifest.pending.json")["limitations"],
                "outputs": [{"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
                            for path in sorted(stage.iterdir()) if path.is_file()]}
    write_json(stage / "final_freeze_manifest.json", manifest)
    require(not OUTPUT.exists(), "Output appeared concurrently; staging retained")
    stage.rename(OUTPUT)
    write_json(POINTER, pointer_for(OUTPUT / "final_freeze_manifest.json"))
    print(json.dumps({"status": manifest["status"], "samples": kept_count, "excluded": excluded_count,
                      "relabeled": changes, "groups": len(group_sizes), "integrity": "PASS"}))


if __name__ == "__main__":
    main()
