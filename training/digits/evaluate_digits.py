"""Evaluate one configured digit model on a user-selected held-out dataset."""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "digits_evaluation.json"
EXPERIMENT_ROOT = (PROJECT_ROOT / "data/experiments/digits_v2").resolve()


def project_path(value):
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def evaluate(model_path, test_path, report_path, labels):
    """Load and predict only; never fit a model or scaler."""
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not test_path.is_file():
        raise FileNotFoundError(
            f"Test data not found: {test_path}. Configure a new independent digits-v2 test set first."
        )

    model = joblib.load(model_path)
    if not isinstance(model, Pipeline) or len(model.steps) != 2:
        raise ValueError("Expected a two-step StandardScaler + MLPClassifier pipeline")
    if not isinstance(model.steps[0][1], StandardScaler) or not isinstance(model.steps[1][1], MLPClassifier):
        raise ValueError("Unexpected digit model pipeline components")
    if getattr(model, "n_features_in_", None) != 63:
        raise ValueError("Model does not accept exactly 63 features")
    if set(map(str, model.classes_)) != set(labels):
        raise ValueError("Model classes do not match the configured digit labels")

    df = pd.read_csv(test_path, dtype={"label": str})
    feature_columns = [f"f{i}" for i in range(1, 64)]
    missing = [col for col in ("label", *feature_columns) if col not in df.columns]
    if missing:
        raise ValueError(f"Test data is missing columns: {missing}")
    if df.empty:
        raise ValueError("Test data is empty")
    if df[feature_columns].isna().any().any():
        raise ValueError("Test data contains missing feature values")
    if not set(df["label"]).issubset(set(labels)):
        raise ValueError("Test data contains labels outside 0-9")

    X_test = df[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(X_test).all():
        raise ValueError("Test data contains non-finite feature values")
    y_true = df["label"].to_numpy()
    y_pred = [str(value) for value in model.predict(X_test)]
    if not set(y_pred).issubset(set(labels)):
        raise ValueError("Model predicted a label outside 0-9")

    per_class = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    errors_by_class = {
        label: int(sum(actual == label and actual != predicted for actual, predicted in zip(y_true, y_pred)))
        for label in labels
    }
    error_indices = [index for index, (actual, predicted) in enumerate(zip(y_true, y_pred)) if actual != predicted]
    errors = []
    for index in error_indices:
        item = {
            "row_index": int(index),
            "true_label": str(y_true[index]),
            "predicted_label": y_pred[index],
        }
        for field in ("sample_id", "file", "source_path", "view"):
            if field in df.columns:
                item[field] = str(df.iloc[index][field])
        errors.append(item)

    result = {
        "model": str(model_path),
        "test_data": str(test_path),
        "samples": int(len(df)),
        "feature_columns": feature_columns,
        "labels": labels,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(per_class[label]["precision"]),
                "recall": float(per_class[label]["recall"]),
                "f1": float(per_class[label]["f1-score"]),
                "support": int(per_class[label]["support"]),
            }
            for label in labels
        },
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": labels,
        "error_count": len(errors),
        "errors_by_true_class": errors_by_class,
        "errors": errors,
        "training_performed": False,
        "test_independence_verified": False,
    }

    confusion_path = report_path.with_name(report_path.stem + "_confusion_matrix.csv")
    existing = [str(path) for path in (report_path, confusion_path) if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError("Refusing to overwrite evaluation outputs: " + ", ".join(existing))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    ).to_csv(confusion_path)

    print(f"Accuracy: {result['accuracy']:.4f} | Macro-F1: {result['macro_f1']:.4f} | Errors: {len(errors)}")
    for label in labels:
        metrics = result["per_class"][label]
        print(
            f"{label}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, errors={errors_by_class[label]}"
        )
    print("Confusion matrix (rows=true, columns=predicted):")
    print(pd.DataFrame(matrix, index=labels, columns=labels).to_string())
    print(f"Report: {report_path}")
    print(f"Confusion CSV: {confusion_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate one digit model after user confirmation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    config_path = project_path(args.config)
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    labels = [str(value) for value in config["classes"]]
    if config.get("feature_count") != 63 or labels != [str(i) for i in range(10)]:
        parser.error("Evaluation config must specify 63 features and ordered labels 0-9")

    model_path = project_path(config["model"])
    test_path = project_path(config["test_data"])
    report_path = project_path(config["output_report"])
    if report_path != EXPERIMENT_ROOT and EXPERIMENT_ROOT not in report_path.parents:
        parser.error("Evaluation report must be inside data/experiments/digits_v2/")
    if report_path.suffix.lower() != ".json":
        parser.error("Evaluation report must have a .json extension")

    print("Evaluation configuration:")
    print(json.dumps({
        "config_path": str(config_path),
        "model": str(model_path),
        "test_data": str(test_path),
        "output_report": str(report_path),
        "feature_count": config["feature_count"],
        "classes": labels,
    }, indent=2, ensure_ascii=False))
    try:
        answer = input("Start evaluation? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer != "y":
        print("Evaluation cancelled; no report was written.")
        return 0

    evaluate(model_path, test_path, report_path, labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
