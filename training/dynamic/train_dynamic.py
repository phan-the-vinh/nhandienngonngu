"""Single dynamic training entrypoint; J/Z baseline is gated by source rights.

--check-only validates inputs without split creation, fitting, or writes.
--train (or VS Code Run Python File) requires verified rights and confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/dynamic_training.json"
SOURCE_PREFIX = "data/raw/dynamic/external/signnteam_jz/SigNN Video Data/"
EXPECTED_EXCLUSIONS = {"J/99.avi", "J/103.avi", "J/250.avi"}
GROUP_FIELDS = ("signer_id", "session_id", "gesture_bout_id", "source_group")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(value: str) -> Path:
    candidate = (ROOT / value).resolve()
    if not candidate.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Path escapes project: {value}")
    return candidate


def load_config(path: Path = CONFIG) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if (config["labels"] != ["J", "Z"] or config["sequence_length"] != 16
            or config["feature_dim"] != 69
            or config["expected_source_counts"] != {"J": 268, "Z": 375}
            or config["expected_final_counts"] != {"J": 265, "Z": 375}
            or set(config["confirmed_additional_exclusions"]) != EXPECTED_EXCLUSIONS
            or len(config["confirmed_additional_exclusions"]) != 3
            or config["split"] != {"train": 0.8, "validation": 0.1, "test": 0.1,
                                   "seed": 42, "stratified": True}
            or config["model"] != {"pipeline": ["StandardScaler", "LogisticRegression"],
                                    "C": 1.0, "max_iter": 1000, "solver": "lbfgs",
                                    "random_state": 42}):
        raise ValueError("Unexpected J/Z baseline contract in config")
    for key in ("input_jsonl", "j40_manifest", "batch1_decisions", "output_dir"):
        local_path(config[key])
    return config


def source_key(record: dict) -> str:
    source = record.get("source_path")
    label = record.get("label")
    if not isinstance(source, str) or label not in ("J", "Z"):
        raise ValueError("Missing/invalid source or J/Z label")
    if not source.startswith(SOURCE_PREFIX + label + "/"):
        raise ValueError(f"Label/source mismatch: {source}")
    name = source.removeprefix(SOURCE_PREFIX)
    if (len(Path(name).parts) != 2 or Path(name).suffix.lower() != ".avi"
            or Path(name).name in ("", ".", "..")):
        raise ValueError(f"Unsafe source path: {source}")
    return name.replace("\\", "/")


def validate_decisions(config: dict) -> dict:
    decision_path = local_path(config["batch1_decisions"])
    decisions = json.loads(decision_path.read_text(encoding="utf-8"))
    if decisions.get("decision_source") != "USER_CONFIRMED":
        raise ValueError("Batch-1 decisions lack user confirmation")
    entries = decisions.get("decisions", [])
    if len(entries) != 10 or len({row["video"] for row in entries}) != 10:
        raise ValueError("Batch-1 decisions must contain 10 unique videos")
    excluded = {row["video"] for row in entries if row["decision"] == "EXCLUDE_CONFIRMED"}
    kept = {row["video"] for row in entries if row["decision"] == "KEEP_FOR_TRAINING"}
    if (excluded != EXPECTED_EXCLUSIONS or len(kept) != 7
            or excluded & kept or any(row.get("source") != "USER_CONFIRMED" for row in entries)
            or decisions.get("total_confirmed_exclusions") != 43):
        raise ValueError("Batch-1 decision contract mismatch")
    prior_path = local_path(decisions["prior_exclusion_manifest"])
    preview_path = local_path(decisions["preview_manifest"])
    if (sha256(prior_path) != decisions["prior_exclusion_manifest_sha256"]
            or sha256(preview_path) != decisions["preview_manifest_sha256"]):
        raise ValueError("Decision provenance hash mismatch")
    preview = json.loads(preview_path.read_text(encoding="utf-8"))
    preview_keys = {source_key({"source_path": row["source_path"], "label": "J"})
                    for row in preview["videos"]}
    if preview.get("batch") != 1 or preview_keys != {row["video"] for row in entries}:
        raise ValueError("Batch-1 decision IDs differ from preview manifest")
    with prior_path.open("r", encoding="utf-8-sig", newline="") as stream:
        old_exclusions = [row["video"] for row in csv.DictReader(stream)]
    if (len(old_exclusions) != 40 or len(set(old_exclusions)) != 40
            or set(old_exclusions) & preview_keys):
        raise ValueError("Prior J40 exclusion manifest overlaps or has wrong count")
    return decisions


def load_checked_data(config: dict, *, materialize: bool = False) -> tuple[list[dict], np.ndarray | None, np.ndarray | None, dict]:
    """Read and validate only; never write a filtered dataset or make a split."""
    decisions = validate_decisions(config)
    source_path = local_path(config["input_jsonl"])
    j40 = json.loads(local_path(config["j40_manifest"]).read_text(encoding="utf-8"))
    if (j40.get("status") != "FILTERED_J40_VERIFIED"
            or j40.get("filtered_jsonl") != config["input_jsonl"]
            or sha256(source_path) != j40.get("filtered_sha256")
            or j40.get("after_counts") != config["expected_source_counts"]):
        raise ValueError("J40 JSONL/manifest integrity mismatch")

    all_sources: set[str] = set()
    source_counts: Counter[str] = Counter()
    final_counts: Counter[str] = Counter()
    excluded_found: set[str] = set()
    records: list[dict] = []
    features: list[np.ndarray] = []
    labels: list[str] = []
    group_fields_present: set[str] = set()
    with source_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            key = source_key(record)
            if key in all_sources:
                raise ValueError(f"Duplicate source path: {key}")
            all_sources.add(key)
            label = record["label"]
            source_counts[label] += 1
            if (record.get("sequence_length") != 16 or record.get("feature_dim") != 69
                    or record.get("extraction_status") != "SUCCESS"):
                raise ValueError(f"Invalid metadata at JSONL line {line_number}")
            indices = record.get("selected_original_frame_indices")
            count = record.get("decoded_frame_count")
            if (not isinstance(indices, list) or len(indices) != 16
                    or any(type(index) is not int for index in indices)
                    or indices != sorted(set(indices))
                    or type(count) is not int or indices[0] < 0 or indices[-1] >= count):
                raise ValueError(f"Invalid original frame indices at line {line_number}")
            array = np.asarray(record.get("features"), dtype=np.float64)
            if array.shape != (16, 69) or not np.isfinite(array).all():
                raise ValueError(f"Invalid (16,69) features at line {line_number}")
            group_fields_present.update(field for field in GROUP_FIELDS if record.get(field) not in (None, "", "UNKNOWN"))
            if key in EXPECTED_EXCLUSIONS:
                excluded_found.add(key)
                continue
            records.append({"source_path": record["source_path"], "label": label,
                            "z_variant": record.get("z_variant")})
            if materialize:
                features.append(array)
                labels.append(label)
            final_counts[label] += 1
    if (dict(source_counts) != config["expected_source_counts"]
            or excluded_found != EXPECTED_EXCLUSIONS
            or dict(final_counts) != config["expected_final_counts"]
            or len(records) != 640 or group_fields_present):
        raise ValueError(f"Count/exclusion/group contract mismatch: source={dict(source_counts)}, "
                         f"final={dict(final_counts)}, excluded={sorted(excluded_found)}, "
                         f"group_fields={sorted(group_fields_present)}")
    X = np.stack(features).reshape(640, 1104, order="C") if materialize else None
    y = np.asarray(labels) if materialize else None
    return records, X, y, {"source_counts": dict(source_counts),
                          "final_counts": dict(final_counts),
                          "excluded": sorted(excluded_found),
                          "input_sha256": j40["filtered_sha256"],
                          "decision_sha256": sha256(local_path(config["batch1_decisions"])),
                          "signer_session_metadata": "ABSENT; source-video split is not signer-independent",
                          "decision_source": decisions["decision_source"]}


def rights_verified(config: dict) -> bool:
    rights = config["source_rights"]
    evidence = rights.get("evidence_path", "")
    return (rights.get("status") == "VERIFIED_FOR_ACADEMIC_ML_TRAINING"
            and isinstance(evidence, str) and bool(evidence)
            and local_path(evidence).is_file())


def make_split(y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Called only after the user confirms --train; never by --check-only."""
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(y))
    train, holdout = train_test_split(indices, test_size=0.2, random_state=seed, stratify=y)
    validation, test = train_test_split(holdout, test_size=0.5, random_state=seed,
                                        stratify=y[holdout])
    result = {"train": train, "validation": validation, "test": test}
    if {key: len(value) for key, value in result.items()} != {"train": 512, "validation": 64, "test": 64}:
        raise ValueError("Split size mismatch")
    if len(set(np.concatenate(list(result.values())))) != 640:
        raise ValueError("Split overlap or omission")
    return result


def train(config: dict, records: list[dict], X: np.ndarray, y: np.ndarray, audit: dict) -> Path:
    from joblib import dump
    from sklearn import __version__ as sklearn_version
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    output = local_path(config["output_dir"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment output: {output}")
    split = make_split(y, config["split"]["seed"])
    sources_by_split = {name: {records[int(index)]["source_path"] for index in indices}
                        for name, indices in split.items()}
    if any(sources_by_split[a] & sources_by_split[b]
           for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Source-video leakage across splits")
    spec = config["model"]
    model = Pipeline([("scaler", StandardScaler()),
                      ("classifier", LogisticRegression(C=spec["C"], max_iter=spec["max_iter"],
                                                        solver=spec["solver"],
                                                        random_state=spec["random_state"]))])
    model.fit(X[split["train"]], y[split["train"]])
    output.mkdir(parents=True, exist_ok=False)
    model_path = output / "model_pipeline.joblib"
    scaler_path = output / "scaler.joblib"
    dump(model, model_path)
    dump(model.named_steps["scaler"], scaler_path)
    split_manifest = {
        "seed": config["split"]["seed"], "strategy": "stratified_by_label_at_source_video_level",
        "ratios": {key: config["split"][key] for key in ("train", "validation", "test")},
        "group_metadata": audit["signer_session_metadata"],
        "splits": {name: [{"source_path": records[int(index)]["source_path"],
                           "label": records[int(index)]["label"]} for index in indices]
                   for name, indices in split.items()},
        "class_counts": {name: dict(Counter(y[indices])) for name, indices in split.items()},
        "source_overlap": 0,
    }
    split_path = output / "split_manifest.json"
    split_path.write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")
    config_path = output / "config_snapshot.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    training_manifest = {
        "experiment_name": config["experiment_name"], "status": "EXPERIMENTAL_NOT_PRODUCTION",
        "data": audit, "feature_shape": [16, 69], "flattened_feature_count": 1104,
        "flatten_order": "C: chronological frame 0..15, feature 0..68 within each frame",
        "pipeline": ["StandardScaler", "LogisticRegression"],
        "scaler_fit_on": "train_only", "test_used_for_model_selection": False,
        "rights_evidence_path": config["source_rights"]["evidence_path"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit_learn": sklearn_version},
        "artifacts": {"model_pipeline.joblib": sha256(model_path),
                      "scaler.joblib": sha256(scaler_path),
                      "split_manifest.json": sha256(split_path),
                      "config_snapshot.json": sha256(config_path)},
    }
    (output / "training_manifest.json").write_text(
        json.dumps(training_manifest, indent=2) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Validate data and rights; no split or fit")
    mode.add_argument("--train", action="store_true", help="Train after rights check and confirmation")
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        records, X, y, audit = load_checked_data(config, materialize=False)
        allowed = rights_verified(config)
        summary = {"status": "DATA_PASS_RIGHTS_VERIFIED" if allowed else "DATA_PASS_RIGHTS_BLOCKED",
                   "sequence_count": len(records), "class_counts": audit["final_counts"],
                   "input_shape": [16, 69], "flattened_features": 1104,
                   "excluded": audit["excluded"], "split_plan": "80/10/10 stratified seed=42",
                   "signer_leakage_risk": audit["signer_session_metadata"],
                   "rights_status": config["source_rights"]["status"],
                   "training_performed": False}
        print(json.dumps(summary))
        if args.check_only:
            return 0
        if not allowed:
            raise ValueError("SOURCE_RIGHTS_UNVERIFIED: verify video training rights and set evidence_path first")
        print(json.dumps({"experiment": config["experiment_name"], "output_dir": config["output_dir"],
                          "model": config["model"], "split": config["split"],
                          "class_counts": audit["final_counts"]}, indent=2))
        if input("Start training? [y/N] ").strip().lower() != "y":
            print("CANCELLED: no split or training performed")
            return 0
        records, X, y, audit = load_checked_data(config, materialize=True)
        output = train(config, records, X, y, audit)
        print(json.dumps({"status": "EXPERIMENT_TRAINED", "output_dir": output.relative_to(ROOT).as_posix()}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
