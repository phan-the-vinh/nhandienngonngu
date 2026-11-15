"""Analyze TRAIN class/group imbalance without modifying split artefacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "ai/dataset/train_validation_test_split_v1"
FROZEN_DIR = ROOT / "ai/dataset/preprocessing_frozen_v3"
OUTPUT = ROOT / "ai/dataset/train_balance_analysis_v1"
SPLITS = ("TRAIN", "VALIDATION", "TEST")
STATIC_CLASSES = tuple("ABCDEFGHIKLMNOPQRSTUVWXY")
FEATURES = [f"f{i}" for i in range(1, 64)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames), list(reader)


def write_csv(path, fields, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def signature(path, base=ROOT):
    path = Path(path)
    return {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def verify_inputs():
    manifest_path = SPLIT_DIR / "split_manifest.json"
    manifest = load_json(manifest_path)
    require(manifest["status"] == "TRAIN_VALIDATION_TEST_SPLIT_READY", "Split manifest is not ready")
    require(manifest["seed"] == 42 and manifest["integrity"]["status"] == "PASS", "Split integrity/seed mismatch")
    require(not manifest["remaining_blockers"], "Split manifest still has blockers")
    snapshots = {}
    for item in manifest["outputs"]:
        path = SPLIT_DIR / item["path"]
        require(path.is_file() and path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"],
                f"Split output changed: {path}")
        snapshots[path.relative_to(ROOT).as_posix()] = item["sha256"]
    for path in [manifest_path, SPLIT_DIR / "test_report.json",
                 FROZEN_DIR / "final_freeze_manifest.json",
                 FROZEN_DIR / "dataset_chucai_clean_frozen_v3.csv",
                 FROZEN_DIR / "sample_group_map_frozen_v3.csv"]:
        require(path.is_file(), f"Missing input: {path}")
        snapshots[path.relative_to(ROOT).as_posix()] = sha256(path)
    require(sha256(FROZEN_DIR / "dataset_chucai_clean_frozen_v3.csv") ==
            manifest["source_dataset"]["sha256"], "Frozen dataset differs from split source")
    require(sha256(FROZEN_DIR / "sample_group_map_frozen_v3.csv") ==
            manifest["source_group_map"]["sha256"], "Frozen group map differs from split source")
    return manifest, snapshots


def validate_split_data():
    _, assignment_rows = read_csv(SPLIT_DIR / "split_assignments.csv")
    _, group_rows = read_csv(SPLIT_DIR / "group_assignments.csv")
    require(len(assignment_rows) == 89415 and len({row["file"] for row in assignment_rows}) == 89415,
            "Sample assignment scope mismatch")
    require(len(group_rows) == 60919 and len({row["group_id"] for row in group_rows}) == 60919,
            "Group assignment scope mismatch")
    group_by_id = {row["group_id"]: row for row in group_rows}
    for row in assignment_rows:
        group = group_by_id.get(row["group_id"])
        require(group and group["split"] == row["split"] and group["label"] == row["label"],
                "Sample/group assignment mismatch")

    assignments = {row["file"]: row for row in assignment_rows}
    split_label_counts, seen = {}, set()
    for split in SPLITS:
        path = SPLIT_DIR / f"{split.lower()}.csv"
        counts = Counter()
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            require(reader.fieldnames[:4] == ["source", "file", "label", "handedness"] and
                    reader.fieldnames[4:] == FEATURES, "Split schema mismatch")
            for row in reader:
                name = row["file"]
                assignment = assignments.get(name)
                require(name not in seen and assignment and assignment["split"] == split and
                        assignment["label"] == row["label"], "Split membership/label mismatch")
                require(all(math.isfinite(float(row[feature])) for feature in FEATURES), "Non-finite feature")
                seen.add(name)
                counts[row["label"]] += 1
        require(set(counts) == set(STATIC_CLASSES), f"Class missing in {split}")
        split_label_counts[split] = counts
    require(len(seen) == 89415, "Missing or duplicated split sample")
    return assignment_rows, group_rows, split_label_counts


def class_group_distribution(train_counts, group_rows):
    train_groups = defaultdict(list)
    for row in group_rows:
        if row["split"] == "TRAIN":
            train_groups[row["label"]].append(int(row["group_size"]))
    total_samples = sum(train_counts.values())
    total_groups = sum(len(values) for values in train_groups.values())
    rows = []
    for label in sorted(STATIC_CLASSES):
        sizes = train_groups[label]
        samples = train_counts[label]
        require(sizes and sum(sizes) == samples, f"TRAIN group sizes do not sum to samples for {label}")
        rows.append({
            "label": label,
            "train_samples": samples,
            "train_sample_percent": samples / total_samples * 100,
            "train_groups": len(sizes),
            "train_group_percent": len(sizes) / total_groups * 100,
            "samples_per_group": samples / len(sizes),
            "median_group_size": statistics.median(sizes),
            "max_group_size": max(sizes),
            "singleton_groups": sum(size == 1 for size in sizes),
            "multi_sample_groups": sum(size > 1 for size in sizes),
            "samples_in_multi_sample_groups": sum(size for size in sizes if size > 1),
            "multi_sample_fraction": sum(size for size in sizes if size > 1) / samples,
        })
    return rows


def imbalance_summary(rows):
    sample_values = [row["train_samples"] for row in rows]
    group_values = [row["train_groups"] for row in rows]
    sample_mean = statistics.mean(sample_values)
    group_mean = statistics.mean(group_values)
    sample_ratio = max(sample_values) / min(sample_values)
    group_ratio = max(group_values) / min(group_values)
    return {
        "sample_count": {
            "minimum": min(sample_values), "minimum_labels": [r["label"] for r in rows if r["train_samples"] == min(sample_values)],
            "maximum": max(sample_values), "maximum_labels": [r["label"] for r in rows if r["train_samples"] == max(sample_values)],
            "max_to_min_ratio": sample_ratio,
            "coefficient_of_variation": statistics.pstdev(sample_values) / sample_mean,
        },
        "group_count": {
            "minimum": min(group_values), "minimum_labels": [r["label"] for r in rows if r["train_groups"] == min(group_values)],
            "maximum": max(group_values), "maximum_labels": [r["label"] for r in rows if r["train_groups"] == max(group_values)],
            "max_to_min_ratio": group_ratio,
            "coefficient_of_variation": statistics.pstdev(group_values) / group_mean,
        },
        "assessment": "HIGH_GROUP_IMBALANCE_AND_MATERIAL_SAMPLE_IMBALANCE" if group_ratio >= 5 or sample_ratio >= 3 else "MODERATE_OR_LOWER_IMBALANCE",
    }


def recommendation(rows, imbalance):
    by_label = {row["label"]: row for row in rows}
    return {
        "preferred": "GROUP_AND_CLASS_ADJUSTED_SAMPLE_WEIGHTING",
        "formula": "For sample i in class c and group g: raw_weight_i = 1 / (number_of_train_groups_in_class_c * train_group_size_g). Normalize weights to mean 1 over TRAIN. Equivalently normalized_weight_i = TRAIN_samples / (24 * class_group_count_c * group_size_g).",
        "rationale": [
            "Each class receives equal total training weight without copying or deleting samples.",
            "Each detected variant group receives equal total weight inside its class, so large variant groups cannot dominate merely through repeated views.",
            "Plain inverse sample-count class weighting alone does not correct within-class group redundancy, which is pronounced for M, N and T.",
        ],
        "training_protocol": [
            "Keep every TRAIN row and shuffle deterministically each epoch.",
            "Apply the group-and-class adjusted weight only to the training loss; do not alter VALIDATION or TEST distributions.",
            "Select model/early stopping with VALIDATION macro-F1 plus per-class recall; retain natural-distribution accuracy as a secondary metric.",
            "Do not inspect TEST until the complete training/model-selection procedure is frozen.",
        ],
        "fallback_if_loss_does_not_support_sample_weights": "Use class-uniform then group-uniform mini-batch sampling at runtime, selecting a member within the chosen group uniformly. Do not materialize duplicated rows or synthetic samples. Keep epoch accounting and seed explicit.",
        "not_recommended": [
            "No physical oversampling/row duplication.",
            "No undersampling or deletion for balance.",
            "No synthetic landmark generation or augmentation in this phase.",
            "No sample-only weighting as the sole correction for M/N/T redundancy.",
        ],
        "m_n_t": {label: {key: by_label[label][key] for key in ["train_samples", "train_groups", "samples_per_group", "median_group_size", "max_group_size", "multi_sample_fraction"]}
                  for label in ("M", "N", "T")},
        "imbalance_assessment": imbalance["assessment"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "Refuse to overwrite an existing analysis")
    require(args.output_dir.resolve().parent == (ROOT / "ai/dataset").resolve(), "Output must remain under ai/dataset")
    split_manifest, snapshots = verify_inputs()
    _, group_rows, split_counts = validate_split_data()
    rows = class_group_distribution(split_counts["TRAIN"], group_rows)
    imbalance = imbalance_summary(rows)
    proposal = recommendation(rows, imbalance)
    before = dict(snapshots)
    stage = Path(tempfile.mkdtemp(prefix="train_balance_analysis_staging_", dir=args.output_dir.parent))
    fields = list(rows[0])
    write_csv(stage / "train_class_group_distribution.csv", fields, rows)
    summary = {
        "status": "ANALYSIS_COMPLETE", "source_split": signature(SPLIT_DIR / "split_manifest.json"),
        "train_samples": sum(split_counts["TRAIN"].values()),
        "train_groups": sum(row["train_groups"] for row in rows),
        "validation_samples_read_only": sum(split_counts["VALIDATION"].values()),
        "test_samples_read_only_not_evaluated": sum(split_counts["TEST"].values()),
        "class_group_distribution": rows, "imbalance": imbalance,
        "m_n_t": proposal["m_n_t"],
    }
    write_json(stage / "train_balance_summary.json", summary)
    write_json(stage / "training_recommendation.json", proposal)
    md_rows = ["| Class | Samples | Groups | Samples/group | Median size | Max size | Multi-sample % |",
               "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        md_rows.append(f"| {row['label']} | {row['train_samples']} | {row['train_groups']} | {row['samples_per_group']:.3f} | {row['median_group_size']:.1f} | {row['max_group_size']} | {row['multi_sample_fraction']*100:.2f}% |")
    (stage / "REPORT.md").write_text(
        "# TRAIN balance analysis\n\n" + "\n".join(md_rows) + "\n\n"
        f"Sample max/min: {imbalance['sample_count']['max_to_min_ratio']:.3f}; group max/min: {imbalance['group_count']['max_to_min_ratio']:.3f}. "
        "M/N/T have few independent detected groups and high samples per group, so sample counts overstate their independent-source diversity.\n\n"
        "Preferred preparation: keep all rows and use normalized per-sample training-loss weight "
        "N_train / (24 × class_group_count × group_size). This equalizes classes and the total influence of each group without changing files. "
        "Leave VALIDATION/TEST natural and untouched; use validation macro-F1 and per-class recall for model selection.\n\n"
        "No balancing files, duplicates, deletions, synthetic samples, scaling, MLP training or TEST evaluation were produced. "
        "group_id covers detected variants only and does not prove person/session independence.\n",
        encoding="utf-8")
    for relative, expected in before.items():
        require(sha256(ROOT / relative) == expected, f"Input changed during analysis: {relative}")
    integrity = {
        "status": "PASS", "input_hashes_checked_before_and_after": len(before),
        "train_samples": sum(split_counts["TRAIN"].values()),
        "validation_samples": sum(split_counts["VALIDATION"].values()),
        "test_samples": sum(split_counts["TEST"].values()),
        "train_groups": sum(row["train_groups"] for row in rows),
        "classes": len(rows), "input_files_modified": 0,
        "rows_duplicated_or_deleted": 0, "training_performed": False,
        "test_evaluation_performed": False,
    }
    write_json(stage / "integrity_report.json", integrity)
    names = ["train_class_group_distribution.csv", "train_balance_summary.json",
             "training_recommendation.json", "REPORT.md", "integrity_report.json"]
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "TRAIN_BALANCE_ANALYSIS_READY", "integrity": integrity,
        "source_snapshots": before, "outputs": [signature(stage / name, base=stage) for name in names],
        "remaining_data_preparation_blockers": [],
        "model_training_authorized_or_performed": False,
    }
    write_json(stage / "analysis_manifest.json", manifest)
    stage.rename(args.output_dir)
    print(json.dumps({"status": manifest["status"], "train_samples": integrity["train_samples"],
                      "train_groups": integrity["train_groups"], "sample_ratio": imbalance["sample_count"]["max_to_min_ratio"],
                      "group_ratio": imbalance["group_count"]["max_to_min_ratio"], "integrity": "PASS"}))


if __name__ == "__main__":
    main()
