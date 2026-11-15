"""Create a reproducible group-aware 70/15/15 split from frozen v3.

This module never fits a model or transforms features. Frozen inputs are read
and verified byte-for-byte; all outputs are written to a new versioned folder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "ai/dataset/preprocessing_frozen_v3"
DATASET = FROZEN / "dataset_chucai_clean_frozen_v3.csv"
GROUP_MAP = FROZEN / "sample_group_map_frozen_v3.csv"
FREEZE_MANIFEST = FROZEN / "final_freeze_manifest.json"
CANONICAL_POINTER = ROOT / "ai/dataset/canonical_preprocessing_manifest.json"
OUTPUT = ROOT / "ai/dataset/train_validation_test_split_v1"
SPLITS = ("TRAIN", "VALIDATION", "TEST")
RATIOS = (0.70, 0.15, 0.15)
SEED = 42
FEATURES = [f"f{i}" for i in range(1, 64)]
STATIC_CLASSES = tuple("ABCDEFGHIKLMNOPQRSTUVWXY")


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


def verify_frozen_inputs():
    manifest = load_json(FREEZE_MANIFEST)
    require(manifest["status"] == "PREPROCESSING_READY_FOR_SPLIT", "Frozen manifest is not ready")
    require(manifest["freeze_authorized"] is True and not manifest["remaining_blockers"],
            "Frozen manifest is blocked")
    require(manifest["sample_count"] == 89415 and manifest["group_count"] == 60919,
            "Unexpected frozen corpus scope")
    require(sha256(DATASET) == manifest["canonical_dataset"]["sha256"], "Frozen dataset hash mismatch")
    require(sha256(GROUP_MAP) == manifest["sample_group_map"]["sha256"], "Frozen group-map hash mismatch")
    checked = {DATASET.relative_to(ROOT).as_posix(): sha256(DATASET),
               GROUP_MAP.relative_to(ROOT).as_posix(): sha256(GROUP_MAP),
               FREEZE_MANIFEST.relative_to(ROOT).as_posix(): sha256(FREEZE_MANIFEST)}
    for item in manifest["outputs"]:
        path = FROZEN / item["path"]
        require(path.is_file() and path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"],
                f"Frozen output changed: {path}")
        checked[path.relative_to(ROOT).as_posix()] = item["sha256"]
    pointer = load_json(CANONICAL_POINTER)
    require(pointer["status"] == manifest["status"], "Canonical pointer status mismatch")
    require(pointer["manifest_sha256"] == sha256(FREEZE_MANIFEST), "Canonical pointer manifest mismatch")
    require(pointer["canonical_dataset"]["sha256"] == sha256(DATASET), "Pointer dataset mismatch")
    require(pointer["sample_group_map"]["sha256"] == sha256(GROUP_MAP), "Pointer group-map mismatch")
    checked[CANONICAL_POINTER.relative_to(ROOT).as_posix()] = sha256(CANONICAL_POINTER)
    return manifest, checked


def target_matrix(label_counts, ratios=RATIOS):
    """Integer per-class targets with exact globally apportioned column totals."""
    labels = sorted(label_counts)
    total = sum(label_counts.values())
    raw_global = [total * ratio for ratio in ratios]
    global_targets = [math.floor(value) for value in raw_global]
    for index in sorted(range(3), key=lambda i: (-(raw_global[i] - global_targets[i]), i))[:total - sum(global_targets)]:
        global_targets[index] += 1

    base = {label: [math.floor(label_counts[label] * ratio) for ratio in ratios] for label in labels}
    deficits = tuple(global_targets[i] - sum(base[label][i] for label in labels) for i in range(3))
    remainders = {label: label_counts[label] - sum(base[label]) for label in labels}
    fractions = {label: [label_counts[label] * ratios[i] - base[label][i] for i in range(3)] for label in labels}

    # DP is a tiny constrained apportionment problem: at most 24 labels and
    # fewer than 72 residual units. It preserves each class total and the
    # exact requested global integer targets while minimizing rounding loss.
    states = {(0, 0): (0.0, [])}
    for label in labels:
        amount = remainders[label]
        options = list(itertools.combinations(range(3), amount))
        next_states = {}
        for (train_used, val_used), (cost, choices) in states.items():
            for option in options:
                nt = train_used + (0 in option)
                nv = val_used + (1 in option)
                test_used = sum(len(choice) - (0 in choice) - (1 in choice) for choice in choices) + (2 in option)
                if nt > deficits[0] or nv > deficits[1] or test_used > deficits[2]:
                    continue
                new_cost = cost - sum(fractions[label][i] for i in option)
                key = (nt, nv)
                candidate = (new_cost, choices + [option])
                if key not in next_states or candidate[0] < next_states[key][0] - 1e-15 or (
                        abs(candidate[0] - next_states[key][0]) <= 1e-15 and candidate[1] < next_states[key][1]):
                    next_states[key] = candidate
        states = next_states
    require((deficits[0], deficits[1]) in states, "Unable to apportion class targets")
    choices = states[(deficits[0], deficits[1])][1]
    result = {}
    for label, option in zip(labels, choices, strict=True):
        row = list(base[label])
        for index in option:
            row[index] += 1
        require(sum(row) == label_counts[label] and all(value > 0 for value in row), "Invalid class target")
        result[label] = tuple(row)
    require(tuple(sum(result[label][i] for label in labels) for i in range(3)) == tuple(global_targets),
            "Global target mismatch")
    return result, tuple(global_targets)


def _allocation(groups, target, seed, trial):
    rng = random.Random(seed + trial * 1_000_003)
    ordered = list(groups)
    if trial == 0:
        tie = {group_id: rng.random() for group_id, _ in ordered}
        ordered.sort(key=lambda item: (-item[1], tie[item[0]], item[0]))
    else:
        rng.shuffle(ordered)
    counts = [0, 0, 0]
    assignment = {}
    for group_id, size in ordered:
        deltas = []
        for index in range(3):
            before = counts[index] - target[index]
            after = counts[index] + size - target[index]
            deltas.append(after * after - before * before)
        best = min(deltas)
        candidates = [i for i, value in enumerate(deltas) if value == best]
        index = candidates[rng.randrange(len(candidates))]
        assignment[group_id] = SPLITS[index]
        counts[index] += size
    return assignment, tuple(counts)


def assign_groups(groups_by_label, targets, seed=SEED):
    assignments = {}
    actual = {}
    for label in sorted(groups_by_label):
        label_seed = int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:8], "big")
        best = None
        for trial in range(32):
            candidate, counts = _allocation(groups_by_label[label], targets[label], label_seed, trial)
            error = (sum(abs(counts[i] - targets[label][i]) for i in range(3)),
                     max(abs(counts[i] - targets[label][i]) for i in range(3)), trial)
            if best is None or error < best[0]:
                best = (error, candidate, counts)
                if error[0] == 0:
                    break
        require(best is not None and set(best[1]) == {group for group, _ in groups_by_label[label]},
                "Incomplete group assignment")
        assignments.update(best[1])
        actual[label] = best[2]
    return assignments, actual


def load_and_validate_membership():
    map_columns, map_rows = read_csv(GROUP_MAP)
    require(len(map_rows) == 89415 and len({row["file"] for row in map_rows}) == 89415,
            "Group map must cover unique samples")
    by_file = {row["file"]: row for row in map_rows}
    members = defaultdict(list)
    labels = defaultdict(set)
    for row in map_rows:
        require(row["label"] in STATIC_CLASSES, "Unexpected label in group map")
        members[row["group_id"]].append(row["file"])
        labels[row["group_id"]].add(row["label"])
    require(len(members) == 60919 and all(len(values) == 1 for values in labels.values()),
            "Group count or label purity mismatch")
    for row in map_rows:
        require(int(row["group_size"]) == len(members[row["group_id"]]), "Stale group size")

    header = None
    dataset_files, label_counts = set(), Counter()
    with DATASET.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames)
        require(header[:4] == ["source", "file", "label", "handedness"] and
                header[4:] == FEATURES, "Unexpected canonical schema")
        for row in reader:
            name = row["file"]
            require(name not in dataset_files and name in by_file, "Duplicate or unmapped canonical file")
            mapped = by_file[name]
            require(row["label"] == mapped["label"] and row["source"] == mapped["source"] and
                    row["handedness"] == mapped["handedness"], "Canonical/group metadata mismatch")
            require(all(math.isfinite(float(row[feature])) for feature in FEATURES), "Non-finite feature")
            dataset_files.add(name)
            label_counts[row["label"]] += 1
    require(dataset_files == set(by_file) and set(label_counts) == set(STATIC_CLASSES),
            "Canonical/group membership or classes mismatch")
    groups_by_label = defaultdict(list)
    for group_id, files in members.items():
        groups_by_label[next(iter(labels[group_id]))].append((group_id, len(files)))
    return header, map_columns, map_rows, by_file, groups_by_label, label_counts


def publish(stage, header, map_rows, by_file, group_assignments):
    paths = {split: stage / f"{split.lower()}.csv" for split in SPLITS}
    handles = {split: path.open("x", encoding="utf-8", newline="") for split, path in paths.items()}
    writers = {split: csv.DictWriter(handles[split], fieldnames=header) for split in SPLITS}
    for writer in writers.values():
        writer.writeheader()
    assignments, class_counts, sample_counts = [], {split: Counter() for split in SPLITS}, Counter()
    try:
        with DATASET.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                mapped = by_file[row["file"]]
                split = group_assignments[mapped["group_id"]]
                writers[split].writerow(row)
                assignments.append({"file": row["file"], "label": row["label"],
                                    "group_id": mapped["group_id"], "split": split})
                class_counts[split][row["label"]] += 1
                sample_counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    write_csv(stage / "split_assignments.csv", ["file", "label", "group_id", "split"], assignments)
    group_rows = []
    groups_seen = set()
    for row in map_rows:
        group_id = row["group_id"]
        if group_id in groups_seen:
            continue
        groups_seen.add(group_id)
        group_rows.append({"group_id": group_id, "label": row["label"],
                           "group_size": row["group_size"], "split": group_assignments[group_id]})
    write_csv(stage / "group_assignments.csv", ["group_id", "label", "group_size", "split"], group_rows)
    return paths, class_counts, sample_counts, assignments, group_rows


def verify_outputs(paths, assignments, group_rows, header):
    assignment_by_file = {row["file"]: row for row in assignments}
    require(len(assignment_by_file) == len(assignments) == 89415, "Assignment duplication or omission")
    group_split = {}
    for row in group_rows:
        require(row["group_id"] not in group_split, "Duplicate group assignment")
        group_split[row["group_id"]] = row["split"]
    require(len(group_split) == 60919, "Missing group assignment")

    readers, handles = {}, {}
    for split, path in paths.items():
        handles[split] = path.open(encoding="utf-8", newline="")
        readers[split] = csv.DictReader(handles[split])
        require(readers[split].fieldnames == header, "Split schema mismatch")
    seen, groups_per_split = set(), {split: set() for split in SPLITS}
    try:
        with DATASET.open(encoding="utf-8-sig", newline="") as source:
            for original in csv.DictReader(source):
                assignment = assignment_by_file[original["file"]]
                split = assignment["split"]
                child = next(readers[split], None)
                require(child == original, f"Label/feature/row mutation: {original['file']}")
                require(original["file"] not in seen, "Sample overlap")
                require(group_split[assignment["group_id"]] == split, "Group split mismatch")
                seen.add(original["file"])
                groups_per_split[split].add(assignment["group_id"])
        require(all(next(reader, None) is None for reader in readers.values()), "Unexpected split row")
    finally:
        for handle in handles.values():
            handle.close()
    require(len(seen) == 89415, "Missing sample")
    for left, right in itertools.combinations(SPLITS, 2):
        require(not groups_per_split[left] & groups_per_split[right], "Cross-split group leakage")
    return {split: len(groups_per_split[split]) for split in SPLITS}


def output_signature(stage, name):
    path = stage / name
    return {"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    require(args.seed == SEED, "This version is frozen to random seed 42")
    require(not args.output_dir.exists(), "Refuse to overwrite an existing split package")
    require(args.output_dir.resolve().parent == (ROOT / "ai/dataset").resolve(), "Output must be versioned under ai/dataset")

    freeze_manifest, protected = verify_frozen_inputs()
    header, map_columns, map_rows, by_file, groups_by_label, label_counts = load_and_validate_membership()
    targets, global_targets = target_matrix(label_counts)
    group_assignments, achieved = assign_groups(groups_by_label, targets, args.seed)
    repeat_assignments, repeat_achieved = assign_groups(groups_by_label, targets, args.seed)
    require(group_assignments == repeat_assignments and achieved == repeat_achieved,
            "Group assignment is not reproducible")

    stage = Path(tempfile.mkdtemp(prefix="train_validation_test_split_staging_", dir=args.output_dir.parent))
    paths, class_counts, sample_counts, assignments, group_rows = publish(
        stage, header, map_rows, by_file, group_assignments)
    group_counts = verify_outputs(paths, assignments, group_rows, header)
    total = sum(sample_counts.values())
    require(total == freeze_manifest["sample_count"] == 89415, "Split sample total mismatch")
    require(tuple(sample_counts[split] for split in SPLITS) == tuple(
        sum(achieved[label][i] for label in achieved) for i in range(3)), "Assignment count mismatch")
    require(all(set(class_counts[split]) == set(STATIC_CLASSES) for split in SPLITS), "Class missing from a split")
    for relative, expected in protected.items():
        require(sha256(ROOT / relative) == expected, f"Frozen input changed: {relative}")

    actual_ratios = {split: sample_counts[split] / total for split in SPLITS}
    per_class = {label: {SPLITS[i]: achieved[label][i] for i in range(3)} for label in sorted(achieved)}
    target_per_class = {label: {SPLITS[i]: targets[label][i] for i in range(3)} for label in sorted(targets)}
    integrity = {
        "status": "PASS", "source_samples": total,
        "split_samples": dict(sample_counts), "split_groups": group_counts,
        "sample_overlap_between_splits": 0, "group_overlap_between_splits": 0,
        "missing_samples": 0, "duplicate_samples": 0,
        "label_changes": 0, "feature_changes": 0, "schema_changes": 0,
        "all_24_classes_present_in_each_split": True,
        "assignment_reproducible_with_seed_42": True,
        "frozen_inputs_unchanged": True,
    }
    write_json(stage / "integrity_report.json", integrity)
    summary = {
        "status": "SPLIT_COMPLETE", "seed": args.seed,
        "target_ratios": dict(zip(SPLITS, RATIOS, strict=True)),
        "integer_global_targets": dict(zip(SPLITS, global_targets, strict=True)),
        "actual_samples": dict(sample_counts), "actual_ratios": actual_ratios,
        "actual_groups": group_counts, "per_class_samples": per_class,
        "per_class_integer_targets": target_per_class,
        "grouping_scope_limitation": "group_id prevents detected duplicate/variant groups from crossing splits. It does not establish subject-independent generalization because performer/person and session metadata are unavailable.",
        "operations_not_performed": ["balancing", "augmentation", "StandardScaler", "MLP training",
                                     "MediaPipe", "extraction", "preprocessing rerun", "audit rerun"],
    }
    write_json(stage / "split_summary.json", summary)
    (stage / "README.md").write_text(
        "# Group-aware train/validation/test split v1\n\n"
        "Source: frozen static ASL corpus v3. Ratios target 70/15/15; random seed 42. "
        "Rows within each split retain source order and are lexically identical to the frozen dataset.\n\n"
        "Use group_id from group_assignments.csv for provenance. No group_id or sample crosses splits. "
        "The grouping only constrains detected duplicate/variant groups; it does not prove generalization "
        "to unseen people or sessions because those identities are absent.\n\n"
        "No balancing, augmentation, scaling, model fitting, MediaPipe, extraction, preprocessing or audit rerun was performed.\n",
        encoding="utf-8")
    test = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "tests.test_create_train_validation_test_split", "-q"],
        cwd=ROOT, capture_output=True, text=True, check=False)
    require(test.returncode == 0, "Split unit tests failed; staging retained")
    write_json(stage / "test_report.json", {
        "status": "PASS", "unit_tests_run": 5,
        "unit_test_module": "tests.test_create_train_validation_test_split",
        "integration_integrity": integrity,
    })
    output_names = ["train.csv", "validation.csv", "test.csv", "split_assignments.csv",
                    "group_assignments.csv", "integrity_report.json", "split_summary.json",
                    "test_report.json", "README.md"]
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "TRAIN_VALIDATION_TEST_SPLIT_READY", "seed": args.seed,
        "source_freeze_manifest": signature(FREEZE_MANIFEST),
        "source_dataset": signature(DATASET), "source_group_map": signature(GROUP_MAP),
        "target_ratios": dict(zip(SPLITS, RATIOS, strict=True)),
        "actual_samples": dict(sample_counts), "actual_ratios": actual_ratios,
        "integrity": integrity, "remaining_blockers": [],
        "outputs": [output_signature(stage, name) for name in output_names],
        "limitations": [summary["grouping_scope_limitation"]],
    }
    write_json(stage / "split_manifest.json", manifest)
    require(not args.output_dir.exists(), "Output appeared concurrently; staging retained")
    stage.rename(args.output_dir)
    print(json.dumps({"status": manifest["status"], "samples": dict(sample_counts),
                      "ratios": actual_ratios, "groups": group_counts, "integrity": "PASS"}))


if __name__ == "__main__":
    main()
