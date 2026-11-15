import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# 1. DUONG DAN
# ============================================================

AI_DIR = Path(__file__).resolve().parents[2] / "ai"

DATA_DIR = (
    AI_DIR
    / "dataset"
    / "kytuso"
    / "datatrain"
)

MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "digits"
    / "candidates"
)

VALIDATION_CSV = (
    DATA_DIR
    / "validation_v1.csv"
)

COMPARISON_JSON = (
    MODEL_DIR
    / "candidate_comparison_v2.json"
)

OUTPUT_REPORT = (
    MODEL_DIR
    / "validation_analysis_v2.json"
)

OUTPUT_CONFUSION = (
    MODEL_DIR
    / "validation_confusion_matrix_v2.csv"
)


# ============================================================
# 2. KIEM TRA FILE
# ============================================================

if not VALIDATION_CSV.exists():
    raise FileNotFoundError(
        f"Khong tim thay validation dataset:\n"
        f"{VALIDATION_CSV}"
    )

if not COMPARISON_JSON.exists():
    raise FileNotFoundError(
        f"Khong tim thay comparison v2:\n"
        f"{COMPARISON_JSON}"
    )


# ============================================================
# 3. DOC BEST CANDIDATE V2
# ============================================================

with open(
    COMPARISON_JSON,
    "r",
    encoding="utf-8"
) as file:
    comparison = json.load(file)


if "best_candidate" not in comparison:
    raise ValueError(
        "candidate_comparison_v2.json "
        "khong co best_candidate."
    )


best_candidate = str(
    comparison["best_candidate"]
)


# Model v2 duoc luu:
# mlp_large_v2a.joblib
# mlp_large_v2b.joblib
# mlp_large_v2c.joblib
MODEL_PATH = (
    MODEL_DIR
    / f"{best_candidate}.joblib"
)


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay model:\n"
        f"{MODEL_PATH}"
    )


# ============================================================
# 4. DOC VALIDATION DATASET
# ============================================================

df = pd.read_csv(
    VALIDATION_CSV,
    dtype={
        "label": str
    }
)


FEATURE_COLUMNS = [
    f"f{i}"
    for i in range(1, 64)
]


missing_features = [
    column
    for column in FEATURE_COLUMNS
    if column not in df.columns
]

if missing_features:
    raise ValueError(
        f"Validation thieu feature:\n"
        f"{missing_features}"
    )


required_metadata = [
    "file",
    "label",
    "hand_id",
    "view",
    "extraction_stage",
]

missing_metadata = [
    column
    for column in required_metadata
    if column not in df.columns
]

if missing_metadata:
    raise ValueError(
        f"Validation thieu metadata:\n"
        f"{missing_metadata}"
    )


X_val = (
    df[
        FEATURE_COLUMNS
    ]
    .to_numpy(
        dtype=float
    )
)

y_val = (
    df[
        "label"
    ]
    .astype(str)
    .to_numpy()
)


# ============================================================
# 5. LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# 6. DU DOAN VALIDATION
# ============================================================

y_pred = model.predict(
    X_val
)

y_pred = [
    str(value)
    for value in y_pred
]


# ============================================================
# 7. METRICS
# ============================================================

accuracy = accuracy_score(
    y_val,
    y_pred
)

macro_f1 = f1_score(
    y_val,
    y_pred,
    average="macro"
)


labels = [
    str(i)
    for i in range(10)
]


report = classification_report(
    y_val,
    y_pred,
    labels=labels,
    output_dict=True,
    zero_division=0
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=labels
)


cm_df = pd.DataFrame(
    cm,
    index=[
        f"true_{label}"
        for label in labels
    ],
    columns=[
        f"pred_{label}"
        for label in labels
    ]
)


cm_df.to_csv(
    OUTPUT_CONFUSION
)


# ============================================================
# 9. CAC MAU DU DOAN SAI
# ============================================================

error_mask = (
    y_val
    !=
    y_pred
)

error_df = df[
    error_mask
].copy()


error_df["predicted"] = [
    prediction
    for prediction, is_error
    in zip(
        y_pred,
        error_mask
    )
    if is_error
]


errors = []

for _, row in error_df.iterrows():

    errors.append({
        "file":
            str(row["file"]),

        "true_label":
            str(row["label"]),

        "predicted_label":
            str(row["predicted"]),

        "hand_id":
            str(row["hand_id"]),

        "view":
            str(row["view"]),

        "extraction_stage":
            str(
                row[
                    "extraction_stage"
                ]
            ),
    })


# ============================================================
# 10. THONG KE LOI THEO TRUE LABEL
# ============================================================

error_by_label = {}

if not error_df.empty:

    counts = (
        error_df
        .groupby("label")
        .size()
        .to_dict()
    )

    error_by_label = {
        str(k): int(v)
        for k, v
        in counts.items()
    }


# ============================================================
# 11. THONG KE LOI THEO VIEW
# ============================================================

error_by_view = {}

if not error_df.empty:

    counts = (
        error_df
        .groupby("view")
        .size()
        .to_dict()
    )

    error_by_view = {
        str(k): int(v)
        for k, v
        in counts.items()
    }


# ============================================================
# 12. LUU REPORT
# ============================================================

result = {

    "version":
        "validation_analysis_v2",

    "model":
        best_candidate,

    "model_path":
        str(MODEL_PATH),

    "comparison_source":
        str(COMPARISON_JSON),

    "dataset":
        str(VALIDATION_CSV),

    "test_set_used":
        False,

    "samples":
        int(len(df)),

    "feature_count":
        63,

    "accuracy":
        float(accuracy),

    "macro_f1":
        float(macro_f1),

    "class_7_recall":
        float(
            report["7"]["recall"]
        ),

    "classification_report":
        report,

    "error_count":
        int(len(error_df)),

    "error_by_label":
        error_by_label,

    "error_by_view":
        error_by_view,

    "errors":
        errors,
}


with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        result,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 13. HIEN THI KET QUA
# ============================================================

print()
print(
    "===================================="
)

print(
    "VALIDATION ANALYSIS V2"
)

print(
    "===================================="
)

print(
    f"Model      : "
    f"{best_candidate}"
)

print(
    f"Samples    : "
    f"{len(df)}"
)

print(
    f"Accuracy   : "
    f"{accuracy:.4f}"
)

print(
    f"Macro F1   : "
    f"{macro_f1:.4f}"
)

print(
    f"Errors     : "
    f"{len(error_df)}"
)

print(
    f"Recall 7   : "
    f"{report['7']['recall']:.4f}"
)


# ============================================================
# 14. PER-CLASS METRICS
# ============================================================

print()
print(
    "PER-CLASS METRICS:"
)

for label in labels:

    values = report[
        label
    ]

    print(
        f"{label}: "
        f"precision="
        f"{values['precision']:.3f}, "
        f"recall="
        f"{values['recall']:.3f}, "
        f"f1="
        f"{values['f1-score']:.3f}"
    )


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print()
print(
    "CONFUSION MATRIX:"
)

print(
    cm_df.to_string()
)


# ============================================================
# 16. HIEN THI CAC MAU SAI
# ============================================================

if not error_df.empty:

    print()
    print(
        "CAC MAU DU DOAN SAI:"
    )

    print(
        error_df[
            [
                "file",
                "label",
                "predicted",
                "view",
                "extraction_stage",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# 17. THONG KE ERROR
# ============================================================

print()
print(
    "ERROR THEO LABEL:"
)

if error_by_label:

    for key, value in sorted(
        error_by_label.items(),
        key=lambda item: int(item[0])
    ):

        print(
            f"{key}: {value}"
        )

else:

    print(
        "NONE"
    )


print()
print(
    "ERROR THEO VIEW:"
)

if error_by_view:

    for key, value in sorted(
        error_by_view.items()
    ):

        print(
            f"{key}: {value}"
        )

else:

    print(
        "NONE"
    )


# ============================================================
# 18. KET THUC
# ============================================================

print()
print(
    "TEST SET USED : False"
)

print()
print(
    "Report:"
)

print(
    OUTPUT_REPORT
)

print()
print(
    "Confusion matrix:"
)

print(
    OUTPUT_CONFUSION
)

print()

print(
    "STATUS: "
    "CHUSO_VALIDATION_V2_COMPLETE"
)
