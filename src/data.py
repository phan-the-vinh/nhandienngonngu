"""Read-only format adapters; no extraction, resampling, splitting or writes."""
import csv
import json
from pathlib import Path
import numpy as np
from src import ROOT

FEATURE_COLUMNS=tuple(f"f{i}" for i in range(1,64))
# dataset/ is a migration target, NOT a claim that historical data moved.
PLANNED_DATASET_ROOT=ROOT / "dataset"
KNOWN_INPUTS={
    "digits_frozen_train":"data/processed/digits/digits_final_train_frozen.csv",
    "digits_validation_reference":"data/splits/digits/validation_reference.csv",
    "digits_regression_reference":"data/splits/digits/regression_reference.csv",
    "letters_frozen":"ai/dataset/preprocessing_frozen_v3/dataset_chucai_clean_frozen_v3.csv",
    "letters_splits":"ai/dataset/train_validation_test_split_v1",
    "dynamic_candidate":"data/experiments/dynamic_jz_v1/processed/signnteam_jz_exclude_j40_v1.jsonl",
}

def dataset_path(name):
    if name not in KNOWN_INPUTS:
        raise ValueError(f"Unknown dataset role: {name}")
    return ROOT / KNOWN_INPUTS[name]

def read_static_csv(path, *, labels, label_column="label", feature_columns=FEATURE_COLUMNS):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream)
        if len(feature_columns)!=63 or not set((label_column,*feature_columns)).issubset(reader.fieldnames or []):
            raise ValueError("CSV must expose an explicit label and 63 ordered feature columns")
        rows=list(reader)
    if not rows or any(row[label_column] not in labels for row in rows):
        raise ValueError("Empty CSV or unexpected labels")
    values=np.asarray([[row[c] for c in feature_columns] for row in rows],dtype=np.float64)
    if values.shape!=(len(rows),63) or not np.isfinite(values).all():
        raise ValueError("CSV features must be finite")
    return rows,values

def validate_sequence_record(record, *, labels=("J","Z")):
    values=np.asarray(record.get("features"),dtype=np.float64)
    indices=record.get("selected_original_frame_indices",[])
    if (record.get("label") not in labels or values.shape!=(16,69)
            or not np.isfinite(values).all() or record.get("sequence_length")!=16
            or record.get("feature_dim")!=69 or len(indices)!=16
            or any(type(i) is not int or i<0 for i in indices)
            or any(a>=b for a,b in zip(indices,indices[1:])) or not record.get("source_path")):
        raise ValueError("Invalid dynamic record contract")
    return values

def read_sequences(path, *, labels=("J","Z")):
    seen=set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            record=json.loads(line)
            validate_sequence_record(record,labels=labels)
            if record["source_path"] in seen:
                raise ValueError("Duplicate sequence source")
            seen.add(record["source_path"])
            yield record
