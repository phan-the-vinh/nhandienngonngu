"""User-operated, no-resampling freeze of the already-trained digits_final inputs.

Default and --check-only never write. --execute validates every source before
asking for confirmation and refuses to overwrite any output. This script does
not fit a scaler/model or evaluate a test set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEATURES = [f"f{i}" for i in range(1, 64)]
CORE = ["label", *FEATURES]
LABELS = {str(i) for i in range(10)}

BASE = ROOT / "ai/dataset/kytuso/datatrain/train_v1.csv"
VALIDATION = ROOT / "ai/dataset/kytuso/datatrain/validation_v1.csv"
REGRESSION = ROOT / "ai/dataset/kytuso/datatrain/test_v1.csv"
ARDA = ROOT / "data/experiments/digits_v2/processed/ardamavi_landmarks_v1.csv"
ASL_HG = ROOT / "data/experiments/digits_v2/processed/asl_hg_4_6_8_landmarks_v1.csv"
MODEL = ROOT / "data/experiments/digits_v2/models/digits_final/mlp_large_v2a.joblib"
TRAINING_MANIFEST = ROOT / "data/experiments/digits_v2/models/digits_final/training_manifest_digits_final.json"
COMPARISON = ROOT / "data/experiments/digits_v2/models/digits_final/candidate_comparison_digits_final.json"
CONFIG = ROOT / "configs/digits_training.json"
SPLIT_MANIFEST = ROOT / "ai/dataset/kytuso/datatrain/split_manifest_v1.json"
TRAINING_SCRIPT = ROOT / "training/digits/train_digits.py"
TRANSFORMER = ROOT / "ai/preprocessing/hand_centric_2d.py"
NORMALIZER = ROOT / "ai/preprocessing/landmark_utils.py"
REQUIREMENTS = ROOT / "requirements.txt"

FROZEN = ROOT / "data/processed/digits/digits_final_train_frozen.csv"
SAMPLE_MAP = ROOT / "data/metadata/digits/digits_final_sample_map.csv"
FREEZE_MANIFEST = ROOT / "data/metadata/digits/digits_final_freeze_manifest.json"
MANIFEST_HASH = ROOT / "data/metadata/digits/digits_final_freeze_manifest.sha256"
VALIDATION_COPY = ROOT / "data/splits/digits/validation_reference.csv"
REGRESSION_COPY = ROOT / "data/splits/digits/regression_reference.csv"
REPLAY_SPEC = ROOT / "data/metadata/digits/digits_final_replay_spec.json"
REPRO_DIR = ROOT / "data/metadata/digits/digits_final_repro"


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verified_hash(path: Path, expected: str | None = None) -> str:
    require(path.is_file(), f"Missing input: {path}")
    actual = sha256_file(path)
    if expected is not None:
        require(actual == expected.lower(), f"SHA256 mismatch: {path}")
    return actual


def load_json(path: Path) -> dict:
    require(path.is_file(), f"Missing input: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def raw_rows(path: Path, required: list[str]) -> list[dict[str, str]]:
    # Preserve original feature tokens. Parsing the same tokens in the frozen
    # CSV yields the same floats that pandas supplied to training.
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, f"Missing CSV header: {path}")
        require(all(name in reader.fieldnames for name in required), f"CSV schema mismatch: {path}")
        return list(reader)


def csv_bytes(header: list[str], rows: list[list[object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def class_counts(frame) -> dict[str, int]:
    return {label: int((frame["label"] == label).sum()) for label in sorted(LABELS)}


def check_training_contract(config: dict, manifest: dict, split: dict, report: dict) -> tuple[dict, Path]:
    selection = manifest["data_sources_and_sampling"]
    require(config["experiment_name"] == manifest["experiment_name"] == "digits_final", "Wrong experiment")
    require(manifest["best_candidate"] == report["best_candidate"] == "mlp_large_v2a", "Wrong selected model")
    require(config["feature_count"] == 63 and selection["feature_columns"] == FEATURES, "Feature schema differs")
    require(config["selection_metric"] == manifest["selection_metric"] == "validation_macro_f1", "Selection rule differs")
    require(config["training"] == manifest["training_parameters"], "Training hyperparameters differ")
    require(config["candidates"] == manifest["candidate_parameters"], "Candidate hyperparameters differ")
    require(config["random_seed"] == selection["random_seed"], "Random seed differs")
    require(not selection["test_set_used"] and not manifest["test_set_used"] and not report["test_set_used"], "TEST was used")
    require(resolve_project_path(config["base_train_data"]) == BASE.resolve(), "Base TRAIN path differs")
    require(resolve_project_path(config["validation_data"]) == VALIDATION.resolve(), "Validation path differs")
    require(resolve_project_path(selection["base_train_data"]) == BASE.resolve(), "Manifest base path differs")
    require(resolve_project_path(selection["validation_data"]) == VALIDATION.resolve(), "Manifest validation path differs")
    require(config["preprocessing"]["id"] == manifest["preprocessing"]["id"] == "hand_centric_2d_v1", "Transformer ID differs")
    require(config["preprocessing"]["axis_epsilon"] == manifest["preprocessing"]["axis_epsilon"], "Transformer epsilon differs")
    require(manifest["preprocessing"]["pipeline_steps"] == ["orientation_2d", "scaler", "mlp"], "Pipeline differs")
    require(resolve_project_path(manifest["preprocessing"]["module_path"]) == TRANSFORMER.resolve(), "Transformer path differs")
    require(len(config["external_sources"]) == len(selection["external_sources"]) == 2, "External source count differs")
    for configured, recorded, expected_path in zip(config["external_sources"], selection["external_sources"], (ARDA, ASL_HG)):
        require(configured["name"] == recorded["name"], "External source name differs")
        require(resolve_project_path(configured["path"]) == expected_path.resolve(), "External config path differs")
        require(resolve_project_path(recorded["path"]) == expected_path.resolve(), "External manifest path differs")
        require(configured["filters"] == recorded["filters"], "External filters differ")
        require(configured["samples_per_class"] == recorded["samples_per_class_requested"], "External sample counts differ")
        require(configured["group_column"] == recorded["group_column"], "External group column differs")
    require(resolve_project_path(config["ablation_reference_manifest"]) == resolve_project_path(selection["ablation_control"]["reference_manifest_path"]), "Ablation reference differs")
    reference = resolve_project_path(config["ablation_reference_manifest"])
    verified_hash(reference, selection["ablation_control"]["reference_manifest_sha256"])
    require(selection["ablation_control"]["exact_v2e_selection_match"] is True, "V2e selection contract not verified")
    verified_hash(BASE, selection["base_train_sha256"])
    verified_hash(VALIDATION, selection["validation_sha256"])
    verified_hash(REGRESSION, split["test"]["sha256"])
    require(split["train"]["sha256"] == selection["base_train_sha256"], "V1 TRAIN split hash differs")
    require(split["validation"]["sha256"] == selection["validation_sha256"], "V1 validation split hash differs")
    verified_hash(TRANSFORMER, manifest["preprocessing"]["module_sha256"])
    verified_hash(MODEL, manifest["model_files_sha256"]["mlp_large_v2a"])
    require(report["training_samples"] == selection["combined_train_samples"] == 549, "Training count differs")
    require(report["validation_samples"] == selection["validation_samples"] == 100, "Validation count differs")
    for path in (COMPARISON, CONFIG, SPLIT_MANIFEST, TRAINING_SCRIPT, NORMALIZER, REQUIREMENTS):
        verified_hash(path)
    return selection, reference


def prepare_freeze() -> tuple[dict[Path, tuple[bytes, str]], dict]:
    import numpy as np
    import pandas as pd

    manifest = load_json(TRAINING_MANIFEST)
    config = load_json(CONFIG)
    split = load_json(SPLIT_MANIFEST)
    report = load_json(COMPARISON)
    selection, reference = check_training_contract(config, manifest, split, report)

    base = pd.read_csv(BASE, dtype={"label": str, "file": str})
    base_raw = raw_rows(BASE, ["file", *CORE])
    require(len(base) == len(base_raw) == selection["base_train_samples"] == 349, "Base TRAIN count differs")
    require(base["file"].notna().all() and base["file"].is_unique, "Base file IDs not unique")
    require(class_counts(base) == selection["base_train_per_class"], "Base class counts differ")

    frozen_raw: list[list[str]] = []
    map_rows: list[list[object]] = []
    identities: set[str] = set()
    frames = [base[CORE]]

    def append(raw: dict[str, str], source: str, source_row: int, original_id: str,
               label: str, historical_path: str, group: str, dataset_hash: str, values) -> None:
        require(raw["label"] == label, f"Raw label mismatch: {source}:{original_id}")
        identity = f"{source}:{original_id}"
        require(identity not in identities, f"Duplicate sample ID: {identity}")
        identities.add(identity)
        features = np.asarray(values, dtype="<f8")
        require(features.shape == (63,) and np.isfinite(features).all(), f"Invalid features: {identity}")
        frozen_raw.append([raw[name] for name in CORE])
        map_rows.append([
            len(frozen_raw) - 1, identity, source, source_row, original_id,
            label, historical_path, group, dataset_hash, hashlib.sha256(features.tobytes()).hexdigest(),
        ])

    for index, raw in enumerate(base_raw):
        original_id = str(base.iloc[index]["file"])
        require(raw["file"] == original_id, f"Base file ID mismatch at row {index}")
        label = str(base.iloc[index]["label"])
        append(raw, "base_train_v1", index, original_id, label, original_id, "",
               selection["base_train_sha256"], base.loc[index, FEATURES].to_numpy(dtype=float))

    for source, expected_path in zip(selection["external_sources"], (ARDA, ASL_HG)):
        verified_hash(expected_path, source["sha256"])
        frame = pd.read_csv(expected_path, dtype={"label": str, "sample_id": str})
        raw = raw_rows(expected_path, ["sample_id", "source_path", *CORE])
        require(len(frame) == len(raw) == source["total_rows"], f"Source row count differs: {source['name']}")
        require(frame["sample_id"].notna().all() and frame["sample_id"].is_unique, f"Source IDs not unique: {source['name']}")
        selected = source["selected_samples"]
        positions = [item["source_row_index"] for item in selected]
        require(all(type(index) is int and 0 <= index < len(frame) for index in positions), "Invalid selected row index")
        require(positions == sorted(set(positions)), f"Selected row order/uniqueness differs: {source['name']}")
        require(len(selected) == source["selected_rows"], f"Selected count differs: {source['name']}")
        selected_frame = frame.iloc[positions]
        require(class_counts(selected_frame) == source["selected_per_class"], f"Selected labels differ: {source['name']}")
        for item in selected:
            index = item["source_row_index"]
            row = frame.iloc[index]
            raw_row = raw[index]
            require(str(row["sample_id"]) == raw_row["sample_id"] == item["sample_id"], "External sample ID mismatch")
            require(str(row["label"]) == raw_row["label"] == item["label"], "External label mismatch")
            require(str(row["source_path"]) == raw_row["source_path"] == item["source_path"], "External source path mismatch")
            group_column = source["group_column"]
            group = str(row[group_column]) if group_column and group_column in frame.columns else ""
            require(group == item["source_group"], "External group mismatch")
            for column, expected in source["filters"].items():
                actual = row[column]
                require(actual == expected or (isinstance(expected, int) and float(actual) == expected), "Selected row fails filter")
            append(raw_row, source["name"], index, item["sample_id"], item["label"],
                   item["source_path"], group, source["sha256"], row[FEATURES].to_numpy(dtype=float))
        frames.append(selected_frame[CORE])

    combined = pd.concat(frames, ignore_index=True)
    require(len(combined) == len(frozen_raw) == len(map_rows) == selection["combined_train_samples"] == 549,
            "Frozen TRAIN total differs")
    require(len(identities) == 549 and class_counts(combined) == selection["combined_train_per_class"],
            "Frozen IDs or class distribution differ")
    require(combined["label"].isin(LABELS).all(), "Invalid TRAIN label")
    require(np.isfinite(combined[FEATURES].to_numpy(dtype=float)).all(), "Non-finite TRAIN features")

    frozen_payload = csv_bytes(CORE, frozen_raw)
    replay = pd.read_csv(io.BytesIO(frozen_payload), dtype={"label": str})
    require(list(replay.columns) == CORE and len(replay) == 549, "Frozen CSV cannot be read")
    require(replay["label"].equals(combined["label"]), "Frozen label order differs")
    require(np.array_equal(replay[FEATURES].to_numpy(dtype=float),
                           combined[FEATURES].to_numpy(dtype=float)),
            "Frozen feature values differ from training input")

    map_header = ["frozen_row_index", "sample_id", "source", "source_row_index",
                  "original_sample_id", "label", "historical_source_path", "source_group",
                  "source_dataset_sha256", "feature_float64_sha256"]
    map_payload = csv_bytes(map_header, map_rows)
    id_digest = sha256_bytes("\n".join(f"{row[2]}|{row[4]}|{row[5]}" for row in map_rows).encode("utf-8"))
    replay_spec = {
        "version": 1,
        "status": "DATA_FROZEN_MODEL_NOT_PROMOTED",
        "train_data": relative(FROZEN),
        "validation_reference": relative(VALIDATION_COPY),
        "regression_reference": relative(REGRESSION_COPY),
        "regression_is_independent_final_test": False,
        "train_sample_count": 549,
        "ordered_source_id_label_sha256": id_digest,
        "feature_columns": FEATURES,
        "preprocessing": {"id": "hand_centric_2d_v1", "axis_epsilon": config["preprocessing"]["axis_epsilon"],
                          "module": relative(TRANSFORMER)},
        "pipeline_steps": ["orientation_2d", "scaler", "mlp"],
        "training_parameters": manifest["training_parameters"],
        "candidate_parameters": manifest["candidate_parameters"],
        "random_seed": selection["random_seed"],
        "selection_metric": manifest["selection_metric"],
        "source_code": {"training_entrypoint": relative(TRAINING_SCRIPT),
                        "transformer": relative(TRANSFORMER), "normalizer": relative(NORMALIZER),
                        "requirements": relative(REQUIREMENTS)},
        "replay_api": "training.digits.train_digits.run_training(config, train_df, val_df, selection, output_dir)",
        "replay_inputs": {
            "config": relative(REPRO_DIR / "digits_training_original.json"),
            "train_data": relative(FROZEN),
            "validation_data": relative(VALIDATION_COPY),
            "selection": relative(REPRO_DIR / "training_manifest_digits_final.json") + "::data_sources_and_sampling",
            "output_dir": "User-chosen new temporary directory, never production",
        },
        "replay_note": "The existing main CLI still checks historical experiment paths. A future user-operated replay can call run_training with the frozen CSV and saved selection metadata directly; that function does not read experiment inputs. The historical paths inside snapshots are provenance only. No training is performed by this freeze utility.",
    }
    snapshots = {
        REPRO_DIR / "digits_training_original.json": CONFIG,
        REPRO_DIR / "training_manifest_digits_final.json": TRAINING_MANIFEST,
        REPRO_DIR / "candidate_comparison_digits_final.json": COMPARISON,
        REPRO_DIR / "split_manifest_v1.json": SPLIT_MANIFEST,
        REPRO_DIR / "training_manifest_v2e_reference.json": reference,
        REPRO_DIR / "train_digits.py": TRAINING_SCRIPT,
        REPRO_DIR / "hand_centric_2d.py": TRANSFORMER,
        REPRO_DIR / "landmark_utils.py": NORMALIZER,
        REPRO_DIR / "requirements.txt": REQUIREMENTS,
    }
    outputs: dict[Path, tuple[bytes, str]] = {
        FROZEN: (frozen_payload, "exact_order_training_features"),
        SAMPLE_MAP: (map_payload, "row_id_and_source_mapping"),
        VALIDATION_COPY: (VALIDATION.read_bytes(), "validation_reference_byte_copy"),
        REGRESSION_COPY: (REGRESSION.read_bytes(), "regression_reference_not_independent_test"),
        REPLAY_SPEC: (json_bytes(replay_spec), "experiment_independent_replay_specification"),
    }
    outputs.update({destination: (source.read_bytes(), "historical_reproducibility_snapshot")
                    for destination, source in snapshots.items()})
    require(sha256_bytes(outputs[VALIDATION_COPY][0]) == selection["validation_sha256"], "Validation copy hash differs")
    require(sha256_bytes(outputs[REGRESSION_COPY][0]) == split["test"]["sha256"], "Regression copy hash differs")
    summary = {
        "train_samples": 549,
        "base_samples": 349,
        "external_samples": {source["name"]: source["selected_rows"] for source in selection["external_sources"]},
        "class_counts": class_counts(combined),
        "ordered_source_id_label_sha256": id_digest,
        "model_source_sha256": manifest["model_files_sha256"]["mlp_large_v2a"],
        "input_sha256": {
            relative(BASE): selection["base_train_sha256"],
            relative(VALIDATION): selection["validation_sha256"],
            relative(REGRESSION): split["test"]["sha256"],
            **{relative(path): source["sha256"] for source, path in zip(selection["external_sources"], (ARDA, ASL_HG))},
        },
    }
    return outputs, summary


def write_freeze(outputs: dict[Path, tuple[bytes, str]], summary: dict) -> None:
    targets = [*outputs, FREEZE_MANIFEST, MANIFEST_HASH]
    existing = [str(path) for path in targets if path.exists() or path.is_symlink()]
    require(not existing, "Refusing to overwrite existing output(s): " + ", ".join(existing))
    for path in targets:
        require(path.resolve().is_relative_to(ROOT.resolve()), f"Output outside project: {path}")
    manifest = {
        "version": 1,
        "status": "DIGITS_FINAL_DATA_FROZEN_NOT_PROMOTED",
        "model_source": relative(MODEL),
        "model_source_sha256": summary["model_source_sha256"],
        "train_samples": summary["train_samples"],
        "base_samples": summary["base_samples"],
        "external_samples": summary["external_samples"],
        "class_counts": summary["class_counts"],
        "ordered_source_id_label_sha256": summary["ordered_source_id_label_sha256"],
        "input_sha256": summary["input_sha256"],
        "regression_reference_notice": "test_v1.csv is historical regression/reference data, NOT an independent final test for digits_final; do not tune on it.",
        "production_dependency_notice": "Only frozen outputs and stable code may be used after experiment cleanup; historical snapshot paths are provenance, not runtime dependencies.",
        "output_files": [
            {"path": relative(path), "size_bytes": len(payload), "sha256": sha256_bytes(payload), "role": role}
            for path, (payload, role) in outputs.items()
        ],
        "manifest_sha256_sidecar": relative(MANIFEST_HASH),
        "self_hash_note": "A JSON file cannot contain its own file SHA256. The detached sidecar records the actual manifest-file hash.",
    }
    manifest_payload = json_bytes(manifest)
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
    for path, (payload, _) in outputs.items():
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        require(sha256_file(path) == sha256_bytes(payload), f"Written output hash mismatch: {path}")
    with FREEZE_MANIFEST.open("xb") as stream:
        stream.write(manifest_payload)
        stream.flush()
        os.fsync(stream.fileno())
    require(sha256_file(FREEZE_MANIFEST) == sha256_bytes(manifest_payload), "Freeze manifest hash mismatch")
    with MANIFEST_HASH.open("xb") as stream:
        stream.write((sha256_bytes(manifest_payload) + "  " + FREEZE_MANIFEST.name + "\n").encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())
    print(f"FROZEN: {summary['train_samples']} samples; {len(outputs)} hashed content outputs")
    print(f"MANIFEST: {relative(FREEZE_MANIFEST)}")
    print(f"MANIFEST_SHA256: {sha256_bytes(manifest_payload)}")
    print("REGRESSION_REFERENCE_IS_INDEPENDENT_FINAL_TEST: False")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify and freeze the exact digits_final TRAIN selection; never resample or train")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Validate hashes, IDs, order and in-memory output; write nothing")
    mode.add_argument("--execute", action="store_true", help="Validate, then ask before writing new frozen outputs")
    args = parser.parse_args()
    if not args.check_only and not args.execute:
        parser.print_help()
        return 0
    outputs, summary = prepare_freeze()
    print(f"VERIFIED: {summary['train_samples']} samples; base={summary['base_samples']}; external={summary['external_samples']}")
    print(f"ORDERED_SAMPLE_DIGEST: {summary['ordered_source_id_label_sha256']}")
    if args.check_only:
        print("CHECK_ONLY: no files written")
        return 0
    try:
        confirmation = input("Type FREEZE to create new artifacts (no overwrite): ").strip()
    except EOFError:
        confirmation = ""
    if confirmation != "FREEZE":
        print("Cancelled; no files written")
        return 0
    write_freeze(outputs, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
