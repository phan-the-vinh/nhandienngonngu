from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

V1_TRAIN = ROOT / "ai" / "dataset" / "kytuso" / "datatrain" / "train_v1.csv"

EXTERNAL = (
    ROOT
    / "data"
    / "experiments"
    / "digits_v2"
    / "processed"
    / "ardamavi_landmarks_v1.csv"
)

OUTPUT_DATASET = (
    ROOT
    / "data"
    / "experiments"
    / "digits_v2"
    / "processed"
    / "digits_v2a_train.csv"
)

OUTPUT_METADATA = (
    ROOT
    / "data"
    / "experiments"
    / "digits_v2"
    / "metadata"
    / "digits_v2a_selected_external.csv"
)

OUTPUT_MANIFEST = (
    ROOT
    / "data"
    / "experiments"
    / "digits_v2"
    / "metadata"
    / "digits_v2a_selection_manifest.json"
)

SEED = 42
SAMPLES_PER_CLASS = 70

FEATURE_COLUMNS = [f"f{i}" for i in range(1, 64)]
TRAIN_COLUMNS = ["label"] + FEATURE_COLUMNS


def main():
    # ---------------------------------------------------------
    # 1. Đọc dữ liệu
    # ---------------------------------------------------------
    v1 = pd.read_csv(V1_TRAIN)
    external = pd.read_csv(EXTERNAL)

    # ---------------------------------------------------------
    # 2. Kiểm tra schema feature
    # ---------------------------------------------------------
    missing_v1 = [c for c in TRAIN_COLUMNS if c not in v1.columns]
    missing_external = [c for c in TRAIN_COLUMNS if c not in external.columns]

    if missing_v1:
        raise RuntimeError(
            f"train_v1 thiếu các cột: {missing_v1}"
        )

    if missing_external:
        raise RuntimeError(
            f"ArdaMavi thiếu các cột: {missing_external}"
        )

    required_external_metadata = [
        "handedness",
        "width",
        "height",
    ]

    missing_metadata = [
        c for c in required_external_metadata
        if c not in external.columns
    ]

    if missing_metadata:
        raise RuntimeError(
            f"ArdaMavi thiếu metadata cần lọc: {missing_metadata}"
        )

    # ---------------------------------------------------------
    # 3. Chuẩn hóa label để kiểm tra
    # ---------------------------------------------------------
    v1["label"] = pd.to_numeric(v1["label"], errors="raise").astype(int)
    external["label"] = pd.to_numeric(
        external["label"], errors="raise"
    ).astype(int)

    valid_labels = set(range(10))

    if not set(v1["label"].unique()).issubset(valid_labels):
        raise RuntimeError("train_v1 có label ngoài 0-9")

    if not set(external["label"].unique()).issubset(valid_labels):
        raise RuntimeError("ArdaMavi có label ngoài 0-9")

    # ---------------------------------------------------------
    # 4. Kiểm tra NaN / Inf trên feature
    # ---------------------------------------------------------
    if v1[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("train_v1 có NaN")

    if external[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("ArdaMavi có NaN")

    # ---------------------------------------------------------
    # 5. Lọc ArdaMavi eligible
    # ---------------------------------------------------------
    eligible = external[
        (external["handedness"] == "Right")
        & (external["width"] == 100)
        & (external["height"] == 100)
    ].copy()

    print("=== V1 TRAIN ===")
    print(f"Samples: {len(v1)}")
    print(v1["label"].value_counts().sort_index())

    print("\n=== ARDAMAVI ELIGIBLE ===")
    print(f"Samples: {len(eligible)}")
    print(eligible["label"].value_counts().sort_index())

    # ---------------------------------------------------------
    # 6. Kiểm tra đủ 70 mẫu/class
    # ---------------------------------------------------------
    counts = eligible["label"].value_counts()

    for label in range(10):
        count = int(counts.get(label, 0))
        if count < SAMPLES_PER_CLASS:
            raise RuntimeError(
                f"Class {label} chỉ có {count} mẫu eligible, "
                f"không đủ {SAMPLES_PER_CLASS}"
            )

    # ---------------------------------------------------------
    # 7. Sampling deterministic 70/class
    # ---------------------------------------------------------
    selected_parts = []

    for label in range(10):
        class_rows = eligible[eligible["label"] == label]

        sampled = class_rows.sample(
            n=SAMPLES_PER_CLASS,
            random_state=SEED,
            replace=False,
        )

        selected_parts.append(sampled)

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # 8. Dataset train chỉ label + f1...f63
    # ---------------------------------------------------------
    v1_train_only = v1[TRAIN_COLUMNS].copy()
    external_train_only = selected[TRAIN_COLUMNS].copy()

    combined = pd.concat(
        [v1_train_only, external_train_only],
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # 9. Exact duplicate audit
    # ---------------------------------------------------------
    duplicate_mask = combined.duplicated(
        subset=TRAIN_COLUMNS,
        keep=False,
    )

    duplicate_count = int(duplicate_mask.sum())

    print("\n=== SELECTED EXTERNAL ===")
    print(f"Samples: {len(selected)}")
    print(selected["label"].value_counts().sort_index())

    print("\n=== COMBINED ===")
    print(f"Samples: {len(combined)}")
    print(combined["label"].value_counts().sort_index())

    print("\nExact duplicate rows:", duplicate_count)

    # Không tự deduplicate.
    if duplicate_count > 0:
        print(
            "\nSTOP: phát hiện exact duplicate. "
            "Không ghi dataset v2a."
        )
        return

    # ---------------------------------------------------------
    # 10. Tạo thư mục output nếu cần
    # ---------------------------------------------------------
    OUTPUT_DATASET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_METADATA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 11. Ghi output
    # ---------------------------------------------------------
    combined.to_csv(
        OUTPUT_DATASET,
        index=False,
    )

    selected.to_csv(
        OUTPUT_METADATA,
        index=False,
    )

    manifest = {
        "candidate_name": "digits_v2a",
        "random_seed": SEED,

        "v1_train_path": str(V1_TRAIN.relative_to(ROOT)),
        "v1_train_sample_count": int(len(v1)),

        "external_dataset_path": str(EXTERNAL.relative_to(ROOT)),
        "external_eligible_count": int(len(eligible)),

        "external_selection_rules": {
            "handedness": "Right",
            "width": 100,
            "height": 100,
            "samples_per_class": SAMPLES_PER_CLASS,
        },

        "external_selected_total": int(len(selected)),

        "external_selected_per_class": {
            str(k): int(v)
            for k, v in selected["label"]
            .value_counts()
            .sort_index()
            .items()
        },

        "combined_train_total": int(len(combined)),

        "combined_per_class": {
            str(k): int(v)
            for k, v in combined["label"]
            .value_counts()
            .sort_index()
            .items()
        },

        "training_schema": TRAIN_COLUMNS,
        "feature_count": 63,

        "validation_reference":
            "ai/dataset/kytuso/datatrain/validation_v1.csv",

        "test_reference":
            "ai/dataset/kytuso/datatrain/test_v1.csv",

        "validation_and_test_used_for_sampling": False,
        "training_performed": False,
    }

    with open(
        OUTPUT_MANIFEST,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== CREATED ===")
    print(OUTPUT_DATASET)
    print(OUTPUT_METADATA)
    print(OUTPUT_MANIFEST)

    print("\nSTATUS: DIGITS_V2A_TRAIN_DATASET_READY")
    print("TRAINING_PERFORMED: False")


if __name__ == "__main__":
    main()
