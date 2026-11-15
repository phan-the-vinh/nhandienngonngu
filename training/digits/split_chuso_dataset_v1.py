import hashlib
import json
from pathlib import Path

import pandas as pd


# ============================================================
# 1. DUONG DAN
# ============================================================

AI_DIR = Path(__file__).resolve().parents[2] / "ai"

CLEAN_DIR = (
    AI_DIR
    / "dataset"
    / "kytuso"
    / "datasetsach"
)

INPUT_DATASET = (
    CLEAN_DIR
    / "dataset_chuso_clean_v1.csv"
)

FREEZE_MANIFEST = (
    CLEAN_DIR
    / "final_freeze_manifest_v1.json"
)

OUTPUT_DIR = (
    AI_DIR
    / "dataset"
    / "kytuso"
    / "datatrain"
)

TRAIN_CSV = (
    OUTPUT_DIR
    / "train_v1.csv"
)

VALIDATION_CSV = (
    OUTPUT_DIR
    / "validation_v1.csv"
)

TEST_CSV = (
    OUTPUT_DIR
    / "test_v1.csv"
)

SPLIT_MANIFEST = (
    OUTPUT_DIR
    / "split_manifest_v1.json"
)


# ============================================================
# 2. CAU HINH SPLIT
# ============================================================

TRAIN_HANDS = {
    "hand1",
    "hand3",
    "hand4",
}

VALIDATION_HANDS = {
    "hand5",
}

TEST_HANDS = {
    "hand2",
}

EXPECTED_TOTAL = 699
EXPECTED_TRAIN = 349
EXPECTED_VALIDATION = 100
EXPECTED_TEST = 250


# ============================================================
# 3. HAM SHA256
# ============================================================

def calculate_sha256(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 4. KIEM TRA FILE
# ============================================================

if not INPUT_DATASET.exists():
    raise FileNotFoundError(
        f"Khong tim thay dataset:\n"
        f"{INPUT_DATASET}"
    )

if not FREEZE_MANIFEST.exists():
    raise FileNotFoundError(
        f"Khong tim thay freeze manifest:\n"
        f"{FREEZE_MANIFEST}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 5. KIEM TRA HASH DATASET VOI FREEZE
# ============================================================

with open(
    FREEZE_MANIFEST,
    "r",
    encoding="utf-8"
) as file:

    freeze_manifest = json.load(
        file
    )

expected_sha256 = (
    freeze_manifest[
        "dataset"
    ][
        "sha256"
    ]
)

actual_sha256 = calculate_sha256(
    INPUT_DATASET
)

if actual_sha256 != expected_sha256:

    raise ValueError(
        "SHA256 dataset khong khop freeze manifest.\n"
        f"Expected: {expected_sha256}\n"
        f"Actual  : {actual_sha256}"
    )


# ============================================================
# 6. DOC DATASET
# ============================================================

df = pd.read_csv(
    INPUT_DATASET,
    dtype={
        "label": str
    }
)

if len(df) != EXPECTED_TOTAL:

    raise ValueError(
        f"Tong mau = {len(df)}, "
        f"mong doi = {EXPECTED_TOTAL}"
    )


# ============================================================
# 7. KIEM TRA HAND GROUP
# ============================================================

all_split_hands = (
    TRAIN_HANDS
    | VALIDATION_HANDS
    | TEST_HANDS
)

actual_hands = set(
    df["hand_id"].unique()
)

if actual_hands != all_split_hands:

    raise ValueError(
        "Danh sach hand_id khong dung.\n"
        f"Dataset: {actual_hands}\n"
        f"Split  : {all_split_hands}"
    )


# ============================================================
# 8. CHIA DATASET
# ============================================================

train_df = (
    df[
        df["hand_id"].isin(
            TRAIN_HANDS
        )
    ]
    .copy()
)

validation_df = (
    df[
        df["hand_id"].isin(
            VALIDATION_HANDS
        )
    ]
    .copy()
)

test_df = (
    df[
        df["hand_id"].isin(
            TEST_HANDS
        )
    ]
    .copy()
)


# ============================================================
# 9. KIEM TRA SO MAU
# ============================================================

if len(train_df) != EXPECTED_TRAIN:

    raise ValueError(
        f"Train = {len(train_df)}, "
        f"mong doi = {EXPECTED_TRAIN}"
    )

if len(validation_df) != EXPECTED_VALIDATION:

    raise ValueError(
        f"Validation = "
        f"{len(validation_df)}, "
        f"mong doi = "
        f"{EXPECTED_VALIDATION}"
    )

if len(test_df) != EXPECTED_TEST:

    raise ValueError(
        f"Test = {len(test_df)}, "
        f"mong doi = {EXPECTED_TEST}"
    )


# ============================================================
# 10. KIEM TRA KHONG LEAK FILE
# ============================================================

train_files = set(
    train_df["file"]
)

validation_files = set(
    validation_df["file"]
)

test_files = set(
    test_df["file"]
)

if train_files & validation_files:
    raise ValueError(
        "Leak file TRAIN <-> VALIDATION"
    )

if train_files & test_files:
    raise ValueError(
        "Leak file TRAIN <-> TEST"
    )

if validation_files & test_files:
    raise ValueError(
        "Leak file VALIDATION <-> TEST"
    )


# ============================================================
# 11. KIEM TRA KHONG LEAK HAND
# ============================================================

train_hands_actual = set(
    train_df["hand_id"]
)

validation_hands_actual = set(
    validation_df["hand_id"]
)

test_hands_actual = set(
    test_df["hand_id"]
)

if (
    train_hands_actual
    & validation_hands_actual
):
    raise ValueError(
        "Leak hand TRAIN <-> VALIDATION"
    )

if train_hands_actual & test_hands_actual:
    raise ValueError(
        "Leak hand TRAIN <-> TEST"
    )

if (
    validation_hands_actual
    & test_hands_actual
):
    raise ValueError(
        "Leak hand VALIDATION <-> TEST"
    )


# ============================================================
# 12. KIEM TRA LABEL
# ============================================================

expected_labels = set(
    "0123456789"
)

for name, split_df in [
    ("TRAIN", train_df),
    ("VALIDATION", validation_df),
    ("TEST", test_df),
]:

    labels = set(
        split_df["label"].unique()
    )

    if labels != expected_labels:

        raise ValueError(
            f"{name} khong du 10 label.\n"
            f"{labels}"
        )


# ============================================================
# 13. SAP XEP
# ============================================================

def sort_split(split_df):

    result = split_df.copy()

    result["_label_sort"] = (
        result["label"]
        .astype(int)
    )

    result = (
        result
        .sort_values(
            by=[
                "_label_sort",
                "hand_id",
                "view",
                "file",
            ]
        )
        .drop(
            columns=[
                "_label_sort"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return result


train_df = sort_split(
    train_df
)

validation_df = sort_split(
    validation_df
)

test_df = sort_split(
    test_df
)


# ============================================================
# 14. LUU CSV
# ============================================================

train_df.to_csv(
    TRAIN_CSV,
    index=False
)

validation_df.to_csv(
    VALIDATION_CSV,
    index=False
)

test_df.to_csv(
    TEST_CSV,
    index=False
)


# ============================================================
# 15. THONG KE
# ============================================================

def get_label_counts(split_df):

    return {
        str(k): int(v)
        for k, v
        in (
            split_df
            .groupby("label")
            .size()
            .to_dict()
            .items()
        )
    }


def get_view_counts(split_df):

    return {
        str(k): int(v)
        for k, v
        in (
            split_df
            .groupby("view")
            .size()
            .to_dict()
            .items()
        )
    }


# ============================================================
# 16. SPLIT MANIFEST
# ============================================================

manifest = {

    "version": "v1",

    "source_dataset": {
        "path": str(
            INPUT_DATASET
        ),
        "sha256": actual_sha256,
        "freeze_manifest": str(
            FREEZE_MANIFEST
        ),
    },

    "strategy":
        "group_split_by_hand_id",

    "train": {
        "hands": sorted(
            TRAIN_HANDS
        ),
        "samples": int(
            len(train_df)
        ),
        "label_counts":
            get_label_counts(
                train_df
            ),
        "view_counts":
            get_view_counts(
                train_df
            ),
        "path": str(
            TRAIN_CSV
        ),
        "sha256":
            calculate_sha256(
                TRAIN_CSV
            ),
    },

    "validation": {
        "hands": sorted(
            VALIDATION_HANDS
        ),
        "samples": int(
            len(validation_df)
        ),
        "label_counts":
            get_label_counts(
                validation_df
            ),
        "view_counts":
            get_view_counts(
                validation_df
            ),
        "path": str(
            VALIDATION_CSV
        ),
        "sha256":
            calculate_sha256(
                VALIDATION_CSV
            ),
    },

    "test": {
        "hands": sorted(
            TEST_HANDS
        ),
        "samples": int(
            len(test_df)
        ),
        "label_counts":
            get_label_counts(
                test_df
            ),
        "view_counts":
            get_view_counts(
                test_df
            ),
        "path": str(
            TEST_CSV
        ),
        "sha256":
            calculate_sha256(
                TEST_CSV
            ),
    },

    "validation_checks": {
        "source_hash": "PASS",
        "sample_total": "PASS",
        "file_leakage": "PASS",
        "hand_leakage": "PASS",
        "all_labels_present": "PASS",
    },

    "status":
        "CHUSO_SPLIT_V1_COMPLETE",
}


with open(
    SPLIT_MANIFEST,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        manifest,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 17. KET QUA
# ============================================================

print()
print("====================================")
print("CHIA DATASET CHUSO V1 COMPLETE")
print("====================================")

print(
    f"TRAIN      : {len(train_df)}"
)

print(
    f"VALIDATION : "
    f"{len(validation_df)}"
)

print(
    f"TEST       : {len(test_df)}"
)

print(
    f"TOTAL      : "
    f"{len(train_df) + len(validation_df) + len(test_df)}"
)

print()

print("TRAIN LABELS:")
print(
    train_df
    .groupby("label")
    .size()
)

print()

print("VALIDATION LABELS:")
print(
    validation_df
    .groupby("label")
    .size()
)

print()

print("TEST LABELS:")
print(
    test_df
    .groupby("label")
    .size()
)

print()

print("TRAIN VIEWS:")
print(
    train_df
    .groupby("view")
    .size()
)

print()

print("VALIDATION VIEWS:")
print(
    validation_df
    .groupby("view")
    .size()
)

print()

print("TEST VIEWS:")
print(
    test_df
    .groupby("view")
    .size()
)

print()

print("FILE LEAKAGE : PASS")
print("HAND LEAKAGE : PASS")
print("SOURCE HASH  : PASS")

print()

print("Split manifest:")
print(SPLIT_MANIFEST)

print()

print(
    "STATUS: CHUSO_SPLIT_V1_COMPLETE"
)
