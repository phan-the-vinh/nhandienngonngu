import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

# ============================================================
# 4. CAC MODEL CANDIDATE
# ============================================================

CANDIDATES = {

    "mlp_large_v2a": {
        "hidden_layer_sizes": (
            128,
            64,
            32,
        ),
        "alpha": 0.001,
        "learning_rate_init": 0.001,
    },

    "mlp_large_v2b": {
        "hidden_layer_sizes": (
            256,
            128,
            64,
        ),
        "alpha": 0.0005,
        "learning_rate_init": 0.001,
    },

    "mlp_large_v2c": {
        "hidden_layer_sizes": (
            128,
            128,
            64,
        ),
        "alpha": 0.001,
        "learning_rate_init": 0.0005,
    },
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train and compare the three digit MLP v2 candidates."
    )
    parser.add_argument(
        "--train-data", type=Path, required=True,
        help="Training CSV; required to prevent accidental training",
    )
    parser.add_argument(
        "--validation-data", type=Path, required=True,
        help="Validation CSV; the official test set is not allowed",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Candidate model/report directory; keep experiments outside production",
    )
    args = parser.parse_args(argv)

    train_csv = args.train_data.resolve()
    validation_csv = args.validation_data.resolve()
    output_dir = args.output_dir.resolve()
    production_dir = (Path(__file__).resolve().parents[2] / "models" / "digits" / "production").resolve()
    test_csv = (DATA_DIR / "test_v1.csv").resolve()

    if output_dir == production_dir or production_dir in output_dir.parents:
        parser.error("--output-dir cannot be the digit production directory or its descendant")
    if train_csv == test_csv or validation_csv == test_csv:
        parser.error("The official test_v1.csv cannot be used as training or validation data")
    if not train_csv.is_file():
        parser.error(f"Training CSV not found: {train_csv}")
    if not validation_csv.is_file():
        parser.error(f"Validation CSV not found: {validation_csv}")

    feature_columns = [f"f{i}" for i in range(1, 64)]
    train_df = pd.read_csv(train_csv, dtype={"label": str})
    val_df = pd.read_csv(validation_csv, dtype={"label": str})
    for col in feature_columns:
        if col not in train_df.columns:
            raise ValueError(f"Train thieu feature: {col}")
        if col not in val_df.columns:
            raise ValueError(f"Validation thieu feature: {col}")

    X_train = train_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy()
    X_val = val_df[feature_columns].to_numpy(dtype=float)
    y_val = val_df["label"].to_numpy()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "candidate_comparison_v2.json"

    results = []
    best_name = None
    best_macro_f1 = -1.0
    best_accuracy = -1.0

    for name, config in CANDIDATES.items():
        print(f"\n====================================\nTRAINING: {name}\n====================================")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=config["hidden_layer_sizes"],
                activation="relu",
                solver="adam",
                alpha=config["alpha"],
                batch_size=32,
                learning_rate_init=config["learning_rate_init"],
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=40,
                random_state=42,
            )),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro")
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

        model_path = output_dir / f"{name}.joblib"
        joblib.dump(pipeline, model_path)
        mlp = pipeline.named_steps["mlp"]
        result = {
            "name": name,
            "model_path": str(model_path),
            "hidden_layer_sizes": list(config["hidden_layer_sizes"]),
            "alpha": config["alpha"],
            "learning_rate_init": config["learning_rate_init"],
            "iterations": int(mlp.n_iter_),
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "classification_report": report,
        }
        results.append(result)
        print(f"Iterations : {mlp.n_iter_}")
        print(f"Accuracy   : {accuracy:.4f}")
        print(f"Macro F1   : {macro_f1:.4f}")

        # Chon theo Macro-F1. Neu bang nhau thi dung accuracy.
        if macro_f1 > best_macro_f1 or (macro_f1 == best_macro_f1 and accuracy > best_accuracy):
            best_name = name
            best_macro_f1 = macro_f1
            best_accuracy = accuracy

    comparison = {
        "version": "v2",
        "training_samples": int(len(train_df)),
        "validation_samples": int(len(val_df)),
        "feature_count": 63,
        "selection_metric": "validation_macro_f1",
        "test_set_used": False,
        "best_candidate": best_name,
        "best_validation_macro_f1": float(best_macro_f1),
        "best_validation_accuracy": float(best_accuracy),
        "candidates": results,
    }
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=4, ensure_ascii=False)

    print("\n====================================\nTRAIN CANDIDATES COMPLETE\n====================================")
    for result in results:
        print(
            f"{result['name']}: accuracy={result['accuracy']:.4f}, "
            f"macro_f1={result['macro_f1']:.4f}"
        )
    print(f"\nBEST CANDIDATE : {best_name}")
    print(f"BEST MACRO F1  : {best_macro_f1:.4f}")
    print(f"BEST ACCURACY  : {best_accuracy:.4f}")
    print("\nTEST SET USED  : False\n")
    print("Comparison report:")
    print(report_path)


if __name__ == "__main__":
    main()
